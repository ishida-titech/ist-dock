//! Vina スコアリング・エネルギーカーネルの SIMD 化（f32 / `wide`、高速化）。
//!
//! 探索の内側ループ（受容体全ペアの値＋勾配）を `f32x8`（AVX2 で 8 レーン）でベクトル化する。
//! 受容体は SoA（structure-of-arrays）で前計算して保持し、リガンド 1 原子 × 受容体 8 原子を
//! 一括評価する。`scoring.rs` のスカラ f64 版と関数形は同一（値は f32 精度内で一致）。
//!
//! 注: f32 精度のため探索（局所最適化）の目的関数に使う。最終スコアや `vina --score_only`
//! 一致ゲートはスカラ f64 版（`scoring::intermolecular_terms`）を引き続き使う。

use wide::{f32x8, f64x4, CmpGe, CmpGt, CmpLe, CmpLt};

use crate::geometry::Vec3;
use crate::scoring::{
    is_coordinating_element, metal_hbond, metal_hbond_all, metal_is_zn, metal_radii,
    partner_class, MetalRadii, ScoredAtom, CUTOFF,
    DEFAULT_METAL_COORD_RADIUS, W_GAUSS1, W_GAUSS2, W_HBOND, W_HYDROPHOBIC, W_REPULSION,
};

const LANES: usize = 8;

/// 受容体の SoA 表現（f32、8 の倍数にパディング）。パディング原子は遠方に置きカットオフ外にする。
#[derive(Debug, Clone)]
pub struct ReceptorSimd {
    rx: Vec<f32>,
    ry: Vec<f32>,
    rz: Vec<f32>,
    radius: Vec<f32>,
    /// 受容体原子が**亜鉛**かどうかのフラグ（1.0/0.0）。
    ///
    /// 段 124 で半径の表を相手の種別 8 通りに広げたので、種別ごとに実効半径の配列を
    /// 持つ方式（段 116 の `radius_n` / `radius_o`）では配列が 7 本に増えてしまう。
    /// 代わりに**金属フラグ 1 本と「差」のスカラ**を持つ方式に変えた。
    /// 相手の種別はリガンド原子ごとに 1 つに決まるので、差はループの外で計算できる。
    ///
    /// **半径の表が Vina の既定なら空である**（段 122.16 と同じ理由）。
    is_zn: Vec<f32>,
    /// 受容体原子が**亜鉛以外の金属**かどうかのフラグ。既定なら空。
    is_other_metal: Vec<f32>,
    /// **リガンド側が亜鉛のとき**に受容体原子ごとに引く差（1.2 − 表の値）。既定なら空。
    /// リガンドが金属の場合は、相手（受容体原子）の種別で差が決まるので配列になる。
    lig_zn_delta: Vec<f32>,
    /// **リガンド側が亜鉛以外の金属のとき**に引く差。既定なら空。
    lig_other_delta: Vec<f32>,
    /// 疎水フラグ（1.0/0.0）。
    hydro: Vec<f32>,
    donor: Vec<f32>,
    /// 受容原子フラグ（窒素・酸素をまとめたもの）。水素結合の係数に使う**常用の配列**。
    acceptor: Vec<f32>,
    /// 段 128 の門の列。**門が無効なら空**（作業集合を増やさない。段 122.16 と同じ方針）。
    ///
    /// `coord` は「金属と配位しうる元素か」（窒素・酸素・硫黄・フッ素）、
    /// `metal` は「金属か」。門が開いているとき、水素結合の係数は
    /// `max(don_l*acc_r, acc_l*don_r, 門の項)` になり、門の項は
    /// リガンド原子が金属なら `coord`、配位しうる元素なら `metal` を読む。
    /// 相手の元素で決まるので、フラグの積では表せない（だから列が要る）。
    coord: Vec<f32>,
    metal: Vec<f32>,
    /// 構築時に焼き込んだ半径の表（グリッドと同じ扱い）。
    /// 構築後に大域を変えても反映されない。
    radii: MetalRadii,
    /// 構築時に焼き込んだ水素結合の 2 つの旗。半径の表と同じ扱いで、
    /// **構築後に大域を変えても反映されない**。
    ///
    /// 受容体側のフラグ処理（段 112c の対照で金属の供与体・受容体を 0 にする）は
    /// もともと構築時に行っていたので、リガンド側だけが大域を直接読んでいた。
    /// 段 128.6 で両方をここに揃えた。
    metal_hbond: bool,
    metal_hbond_all: bool,
}

impl ReceptorSimd {
    pub fn new(receptor: &[ScoredAtom]) -> Self {
        Self::with_metal_radii(receptor, metal_radii())
    }

    /// 金属の配位時実効半径の表を明示して構築する（半径の大域を読まない）。
    /// 水素結合の 2 つの旗は大域から読む。
    pub fn with_metal_radii(receptor: &[ScoredAtom], radii: MetalRadii) -> Self {
        Self::with_options(receptor, radii, metal_hbond(), metal_hbond_all())
    }

    /// 表と水素結合の 2 つの旗をすべて明示して構築する（**大域を一切読まない**）。
    /// SIMD とスカラの一致を、大域を書き換えずに試験するために使う。
    pub fn with_options(
        receptor: &[ScoredAtom],
        radii: MetalRadii,
        metal_hbond: bool,
        metal_hbond_all: bool,
    ) -> Self {
        let n = receptor.len();
        let padded = n.div_ceil(LANES) * LANES;
        let mut rx = vec![1.0e9_f32; padded]; // 遠方パディング（カットオフ外）
        let mut ry = vec![1.0e9_f32; padded];
        let mut rz = vec![1.0e9_f32; padded];
        let mut radius = vec![0.0_f32; padded];
        let mut hydro = vec![0.0_f32; padded];
        let mut donor = vec![0.0_f32; padded];
        let mut acceptor = vec![0.0_f32; padded];
        // ★ 段 122.16: 表が Vina の既定なら金属関係の配列を**作らない**。
        // 既定では差が恒等的に 0 なので、持つと作業集合が増えるだけ損である。
        let split = !radii.is_vina_default();
        let mk = |on: bool| if on { vec![0.0_f32; padded] } else { Vec::new() };
        let mut is_zn = mk(split);
        let mut is_other_metal = mk(split);
        let mut lig_zn_delta = mk(split);
        let mut lig_other_delta = mk(split);
        // 段 128 の門。無効なら列を作らない（既定では 1 バイトも増えない）。
        // 旗を両方見るのは「金属の水素結合を全部外す」指定が門より強いためである。
        let gate = metal_hbond_all && metal_hbond;
        let mut coord = mk(gate);
        let mut metal = mk(gate);
        for (i, a) in receptor.iter().enumerate() {
            rx[i] = a.pos.x as f32;
            ry[i] = a.pos.y as f32;
            rz[i] = a.pos.z as f32;
            radius[i] = a.xs.radius() as f32;
            // 金属フラグと、「リガンドが金属のとき」に引く差を前計算する。
            // 受容体側が金属のときの差はリガンド原子の種別で決まるので、
            // ループの外でスカラとして計算できる（配列は要らない）。
            if split {
                let zn = metal_is_zn(a.xs);
                is_zn[i] = flag(zn == Some(true));
                is_other_metal[i] = flag(zn == Some(false));
                let cls = partner_class(a.xs);
                lig_zn_delta[i] =
                    (DEFAULT_METAL_COORD_RADIUS - radii.radius_for(true, cls)) as f32;
                lig_other_delta[i] =
                    (DEFAULT_METAL_COORD_RADIUS - radii.radius_for(false, cls)) as f32;
            }
            hydro[i] = flag(a.xs.is_hydrophobic());
            donor[i] = flag(a.xs.is_donor());
            acceptor[i] = flag(a.xs.is_acceptor());
            // 対照実験（段 112c）で金属の水素結合を外すときは、金属原子の供与体・受容体
            // フラグを両方 0 にする。`hb_pair = max(don_l*acc_r, acc_l*don_r)` なので、
            // これは「どちらかが金属なら水素結合なし」と**厳密に同じ**であり、
            // 内側ループに演算を 1 つも足さない。既定（有効）では何も変わらない。
            if !metal_hbond && metal_is_zn(a.xs).is_some() {
                donor[i] = 0.0;
                acceptor[i] = 0.0;
            }
            // 段 128 の門の列。判定は `scoring::is_coordinating_element` に委ねる
            // （定義を 2 箇所に置かないこと自体が再発防止である）。
            if gate {
                coord[i] = flag(is_coordinating_element(a.xs));
                metal[i] = flag(metal_is_zn(a.xs).is_some());
            }
        }
        ReceptorSimd {
            rx, ry, rz, radius, is_zn, is_other_metal, lig_zn_delta, lig_other_delta,
            hydro, donor, acceptor, coord, metal, radii, metal_hbond, metal_hbond_all,
        }
    }

    pub fn n_chunks(&self) -> usize {
        self.rx.len() / LANES
    }
}

#[inline]
fn flag(b: bool) -> f32 {
    if b { 1.0 } else { 0.0 }
}

#[inline]
fn chunk(v: &[f32], base: usize) -> f32x8 {
    f32x8::new(v[base..base + LANES].try_into().unwrap())
}

/// 分子間エネルギー（重み付き、f64 で集約）と各リガンド原子に働く力 `force[i] = -dE/dpos_i`
/// を SIMD で計算する。`force` は呼び出し側で 0 初期化（長さ = ligand.len()）。
pub fn intermolecular_value_grad_simd(
    ligand: &[ScoredAtom],
    rec: &ReceptorSimd,
    force: &mut [Vec3],
) -> f64 {
    let cutoff2 = f32x8::splat((CUTOFF * CUTOFF) as f32);
    let eps2 = f32x8::splat(1e-8);
    let two = f32x8::splat(2.0);
    let half_lo = f32x8::splat(0.5);
    let one_five = f32x8::splat(1.5);
    let three = f32x8::splat(3.0);
    let neg_half = f32x8::splat(-0.5);
    let neg_eight = f32x8::splat(-8.0);
    let inv_07 = f32x8::splat(1.0 / 0.7);
    let neg_inv_07 = f32x8::splat(-1.0 / 0.7);
    let w_g1 = f32x8::splat(W_GAUSS1 as f32);
    let w_g2 = f32x8::splat(W_GAUSS2 as f32);
    let w_rep = f32x8::splat(W_REPULSION as f32);
    let w_hyd = f32x8::splat(W_HYDROPHOBIC as f32);
    let w_hb = f32x8::splat(W_HBOND as f32);
    let zero = f32x8::ZERO;
    // 半径の表は受容体 SoA の構築時に焼き込んである（大域をここで読み直さない）。
    let radii = rec.radii;
    // 表が Vina の既定なら N／O 別の配列は作られていない（段 122.16）。
    // リガンドごとに判定する必要はないのでここで 1 回だけ求める。
    let split = !radii.is_vina_default();

    let nch = rec.n_chunks();
    let mut total = 0.0_f64;

    for (li, l) in ligand.iter().enumerate() {
        let lx = f32x8::splat(l.pos.x as f32);
        let ly = f32x8::splat(l.pos.y as f32);
        let lz = f32x8::splat(l.pos.z as f32);
        let hyd_l = f32x8::splat(flag(l.xs.is_hydrophobic()));
        // リガンド側も同じ扱い（対照実験のとき金属なら供与体・受容体を落とす）。
        // 旗は受容体 SoA に焼き込んだ値を読む（半径の表と同じ扱い。段 128.6 で揃えた）。
        let lig_metal = metal_is_zn(l.xs).is_some();
        let hb_off = lig_metal && !rec.metal_hbond;
        let don_l = f32x8::splat(if hb_off { 0.0 } else { flag(l.xs.is_donor()) });
        let acc_l = f32x8::splat(if hb_off { 0.0 } else { flag(l.xs.is_acceptor()) });

        // ★ 段 128 の門。半径の扱い（下）と同じ方針で、
        // 「どの列を読めば配位と判定できるか」を**リガンド原子ごとにループの外で**決める。
        // 門が無効なら `None` になり、内側ループは元のコードとビット一致する
        // （列を読みさえしない）。
        //
        //   リガンドが金属        → 相手が配位しうる元素か（`coord` を読む）
        //   リガンドが配位元素    → 相手が金属か（`metal` を読む）
        //   どちらでもない        → 門は何も足さない
        //
        // 金属同士は `coord` が 0 なので自動的に対象外になる。
        let gate_col: Option<&[f32]> = if rec.metal_hbond_all && rec.metal_hbond {
            if lig_metal {
                Some(&rec.coord)
            } else if is_coordinating_element(l.xs) {
                Some(&rec.metal)
            } else {
                None
            }
        } else {
            None
        };

        // ★ 半径の扱い（段 116・122.16・124）。既定の経路に演算を 1 つも足さないため、
        // 「差が 0 かどうか」をリガンド原子ごとにループの外で判定する。
        //
        // (0) 半径の表が Vina の既定なら差は恒等的に 0 で、金属関係の配列は作られていない。
        //     `d = r - (rl + rr)` となり、段 110 以前とビット一致する。
        // (1) 表が既定でなく**受容体側が金属**のとき、引く差はリガンド原子の種別だけで
        //     決まる（`1.2 − 表[亜鉛か][種別]`）。これはループの外のスカラなので、
        //     内側では金属フラグとの積和 2 つで済む。相手の種別が表に無いか値が既定なら
        //     差は 0 で、そのときは**配列を読まない**（例: 表が窒素と硫黄だけなら炭素は読まない）。
        // (2) 表が既定でなく**リガンド原子が金属**のとき、差は相手（受容体原子）の種別で
        //     決まるので、前計算した `lig_zn_delta` / `lig_other_delta` を引く。
        //     金属リガンドは稀なので分岐で切り分ける。
        let rl = f32x8::splat(l.xs.radius() as f32);
        let lcls = partner_class(l.xs);
        let (dz, dother) = if split {
            (
                DEFAULT_METAL_COORD_RADIUS - radii.radius_for(true, lcls),
                DEFAULT_METAL_COORD_RADIUS - radii.radius_for(false, lcls),
            )
        } else {
            (0.0, 0.0)
        };
        // 受容体側が金属のときの補正が要るか（差が両方 0 なら不要）。
        let adj_rec = dz != 0.0 || dother != 0.0;
        let dz_v = f32x8::splat(dz as f32);
        let dother_v = f32x8::splat(dother as f32);
        // リガンドが金属のときに引く差の配列（稀）。
        let adj_lig: Option<&[f32]> = if lig_metal && split {
            Some(if metal_is_zn(l.xs) == Some(true) {
                &rec.lig_zn_delta
            } else {
                &rec.lig_other_delta
            })
        } else {
            None
        };

        let mut e_acc = zero;
        let mut fx = zero;
        let mut fy = zero;
        let mut fz = zero;

        for c in 0..nch {
            let base = c * LANES;
            let rx = chunk(&rec.rx, base);
            let ry = chunk(&rec.ry, base);
            let rz = chunk(&rec.rz, base);
            let rr = chunk(&rec.radius, base);
            let hyd_r = chunk(&rec.hydro, base);
            let don_r = chunk(&rec.donor, base);
            // 常用は受容原子フラグ 1 本だけ（従来と同じ読み込み回数・同じ演算数）。
            let acc_r = chunk(&rec.acceptor, base);

            let dx = lx - rx;
            let dy = ly - ry;
            let dz = lz - rz;
            let r2 = dx * dx + dy * dy + dz * dz;
            let r = r2.sqrt();
            // 半径の和から引く「差」。既定（表が Vina）では `adj_rec` も `adj_lig` も
            // 偽・None なので `delta` は作られず、`d = r - (rl + rr)` とビット一致する
            // （段 110 以前と同じ）。`adj_rec` / `adj_lig` はループ内で不変なので
            // 分岐の予測は当たり、不要な配列は**読みさえしない**。
            let d = if let Some(ld) = adj_lig {
                r - (rl + rr - chunk(ld, base))
            } else if adj_rec {
                let delta = dz_v * chunk(&rec.is_zn, base)
                    + dother_v * chunk(&rec.is_other_metal, base);
                r - (rl + rr - delta)
            } else {
                r - (rl + rr)
            };

            // gauss1: z=d/0.5=2d, g=exp(-z^2), dg/dd=-8d*g。
            let z1 = d * two;
            let g1 = (zero - z1 * z1).exp();
            let dg1 = neg_eight * d * g1;
            // gauss2: z=(d-3)/2, g=exp(-z^2), dg/dd=-(d-3)/2*g。
            let z2 = (d - three) * half_lo;
            let g2 = (zero - z2 * z2).exp();
            let dg2 = neg_half * (d - three) * g2;
            // repulsion: d<0 ? d^2 : 0。
            let neg = d.cmp_lt(zero);
            let rep = neg.blend(d * d, zero);
            let drep = neg.blend(d * two, zero);
            // hydrophobic: d<=0.5→1, d>=1.5→0, else 1.5-d（疎水-疎水のみ）。
            let hyd_pair = hyd_l * hyd_r;
            let mid = one_five - d;
            let hv = d.cmp_le(half_lo).blend(
                f32x8::ONE,
                d.cmp_ge(one_five).blend(zero, mid),
            );
            let dhv = d.cmp_le(half_lo).blend(
                zero,
                d.cmp_ge(one_five).blend(zero, f32x8::splat(-1.0)),
            );
            // hbond: d<=-0.7→1, d>=0→0, else -d/0.7（ドナー-アクセプタのみ、どちら向きも）。
            // 段 128 の門が開いているときだけ、金属配位の項を `max` で足す。
            // `gate_col` はループ内で不変なので分岐の予測は当たり、
            // 門が無効なら列を読まないので既定の経路に演算は 1 つも増えない。
            let hb_pair = (don_l * acc_r).max(acc_l * don_r);
            let hb_pair = match gate_col {
                Some(g) => hb_pair.max(chunk(g, base)),
                None => hb_pair,
            };
            let hbv = d.cmp_le(f32x8::splat(-0.7)).blend(
                f32x8::ONE,
                d.cmp_ge(zero).blend(zero, d * neg_inv_07),
            );
            let dhbv = d.cmp_le(f32x8::splat(-0.7)).blend(
                zero,
                d.cmp_ge(zero).blend(zero, neg_inv_07),
            );
            let _ = inv_07;

            let value = w_g1 * g1
                + w_g2 * g2
                + w_rep * rep
                + w_hyd * hv * hyd_pair
                + w_hb * hbv * hb_pair;
            let dvalue = w_g1 * dg1
                + w_g2 * dg2
                + w_rep * drep
                + w_hyd * dhv * hyd_pair
                + w_hb * dhbv * hb_pair;

            // カットオフ内かつ非重複のみ採用（パディングは遠方で自動除外）。
            let inbox = r2.cmp_lt(cutoff2);
            let near = r2.cmp_gt(eps2);
            let keep_v = inbox.blend(near.blend(value, zero), zero);
            e_acc += keep_v;

            // 力: force_i -= diff * (dvdr / r)。マスク外は 0。
            let coeff = dvalue * r.recip();
            let keep_c = inbox.blend(near.blend(coeff, zero), zero);
            fx -= dx * keep_c;
            fy -= dy * keep_c;
            fz -= dz * keep_c;
        }

        total += hsum(e_acc) as f64;
        force[li] += Vec3::new(hsum(fx) as f64, hsum(fy) as f64, hsum(fz) as f64);
    }
    total
}

#[inline]
fn hsum(v: f32x8) -> f32 {
    v.to_array().iter().sum()
}

/// グリッド構築（スキャッタ）用: 1 受容体原子を x 連続 4 格子点（`px`= 4 レーンの x 座標, `py`/`pz` 共通）に対して
/// 評価し、**重み付きペアエネルギー値のみ**（力・勾配なし, カットオフ外は 0）を f64x4 で返す。
/// プローブ型（格子の対象 XS 型）とペア原子のスカラ属性を受け取る。f64 精度なのでスカラ f64 build と実質一致
/// （加算はグリッド側 f64 アキュムレータで行い、最終 f32 格納後はビット一致）。関数形は `scoring::eval_pair` /
/// `intermolecular_value_grad_simd` の値部と同一（gauss1/gauss2/repulsion/hydrophobic/hbond）。
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn probe_energy_f64x4(
    px: f64x4,
    py: f64,
    pz: f64,
    // **半径の和**（`scoring::pair_sum_radii(probe, atom)`）。配位時の金属半径の
    // 差し替え（段 110）を呼び出し側で済ませておくため、個別の半径ではなく和を受け取る。
    sum_radii: f64,
    // 疎水・水素結合のペア係数（`scoring::pair_factors`）。0.0 か 1.0。
    hyd_pair_s: f64,
    hb_pair_s: f64,
    ax: f64,
    ay: f64,
    az: f64,
) -> f64x4 {
    let zero = f64x4::ZERO;
    let cutoff2 = f64x4::splat(CUTOFF * CUTOFF);
    let two = f64x4::splat(2.0);
    let three = f64x4::splat(3.0);
    let half_lo = f64x4::splat(0.5);
    let one_five = f64x4::splat(1.5);
    let neg_07 = f64x4::splat(-0.7);
    let neg_inv_07 = f64x4::splat(-1.0 / 0.7);
    let w_g1 = f64x4::splat(W_GAUSS1);
    let w_g2 = f64x4::splat(W_GAUSS2);
    let w_rep = f64x4::splat(W_REPULSION);
    let w_hyd = f64x4::splat(W_HYDROPHOBIC);
    let w_hb = f64x4::splat(W_HBOND);

    let dx = px - f64x4::splat(ax);
    let dy = f64x4::splat(py - ay);
    let dz = f64x4::splat(pz - az);
    let r2 = dx * dx + dy * dy + dz * dz;
    let r = r2.sqrt();
    let d = r - f64x4::splat(sum_radii);

    let z1 = d * two;
    let g1 = (zero - z1 * z1).exp();
    let z2 = (d - three) * half_lo;
    let g2 = (zero - z2 * z2).exp();
    let rep = d.cmp_lt(zero).blend(d * d, zero);
    // hydrophobic / hbond のペア係数は原子・型で決まるスカラ（全レーン共通）。
    let hyd_pair = f64x4::splat(hyd_pair_s);
    let hv = d
        .cmp_le(half_lo)
        .blend(f64x4::ONE, d.cmp_ge(one_five).blend(zero, one_five - d));
    let hb_pair = f64x4::splat(hb_pair_s);
    let hbv = d
        .cmp_le(neg_07)
        .blend(f64x4::ONE, d.cmp_ge(zero).blend(zero, d * neg_inv_07));

    let value =
        w_g1 * g1 + w_g2 * g2 + w_rep * rep + w_hyd * hv * hyd_pair + w_hb * hbv * hb_pair;
    r2.cmp_lt(cutoff2).blend(value, zero)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atom_types::XsType;
    use crate::scoring::intermolecular_value_grad;

    fn atom(x: f64, y: f64, z: f64, xs: XsType) -> ScoredAtom {
        ScoredAtom { pos: Vec3::new(x, y, z), xs }
    }

    /// ★ 段 128 の門が SIMD 経路でも効くことを確かめる（段 128.6 の欠陥の回帰試験）。
    ///
    /// 段 128 の最初の実装では、SIMD の探索経路が水素結合の係数を
    /// `max(don_l*acc_r, acc_l*don_r)` として**原子ごとのフラグから直接**組んでいて、
    /// `scoring::h_bond_possible` を通らなかった。そのため門の旗を立てても
    /// `scoring = "direct"` では黙って無視された。
    ///
    /// この試験は 2 つを同時に固定する。
    ///   1. 旗の 4 通りすべてで SIMD とスカラが一致すること
    ///   2. 門を開けると SIMD の値が**実際に動く**こと（動かなければ無視されている）
    ///
    /// 大域は触らない（`with_options` と `pair_energy_grad_with` で旗を注入する）。
    #[test]
    fn simd_stage128_gate_matches_scalar() {
        use crate::scoring::{
            pair_energy_grad_with, sum_radii_with, MetalRadii, STAGE128_RADII,
        };
        // 門の対象（窒素・酸素・硫黄・フッ素）と対象外（炭素・リン・塩素・金属同士）を混ぜる。
        let receptor = vec![
            atom(2.0, 0.0, 0.0, XsType::MetalDZn),
            atom(-2.0, 0.0, 0.0, XsType::MetalD),
            atom(0.0, 2.0, 0.0, XsType::NitrogenD),     // 供与体のみ窒素（門で新たに入る）
            atom(0.0, -2.0, 0.0, XsType::NitrogenP),
            atom(0.0, 0.0, 2.0, XsType::Sulfur),        // 硫黄（門で新たに入る）
            atom(0.0, 0.0, -2.0, XsType::Fluorine),     // フッ素（門で新たに入る）
            atom(1.5, 1.5, 0.0, XsType::OxygenD),       // 供与体のみ酸素（門で新たに入る）
            atom(-1.5, 1.5, 0.0, XsType::OxygenA),      // 既定でも入る
            atom(1.5, -1.5, 0.0, XsType::NitrogenA),    // 既定でも入る
            atom(-1.5, -1.5, 0.0, XsType::CarbonH),     // 門の対象外
            atom(2.5, 0.0, 2.5, XsType::Phosphorus),    // 門の対象外
            atom(-2.5, 0.0, 2.5, XsType::Chlorine),     // 門の対象外
            atom(3.0, 3.0, 0.0, XsType::NitrogenDA),
            atom(-3.0, 3.0, 0.0, XsType::OxygenDA),
        ];
        let ligand = vec![
            atom(2.0, 0.0, 1.99, XsType::NitrogenD),    // 亜鉛–供与体のみ窒素の配位距離
            atom(2.0, 2.31, 0.0, XsType::Sulfur),       // 亜鉛–硫黄の配位距離
            atom(-2.0, 0.0, 1.996, XsType::Fluorine),   // 他の金属–フッ素の配位距離
            atom(0.3, 0.0, 0.0, XsType::MetalDZn),      // リガンド側が亜鉛
            atom(-0.6, 0.6, 0.0, XsType::MetalD),       // リガンド側が亜鉛以外の金属
            atom(0.0, 1.0, 1.0, XsType::OxygenD),
            atom(1.0, 0.0, 1.0, XsType::CarbonP),       // 門の対象外
            atom(0.0, 2.6, 0.6, XsType::NitrogenA),
            atom(-1.0, -1.0, 1.0, XsType::OxygenA),
            atom(2.6, 0.6, 0.0, XsType::Phosphorus),    // 門の対象外
        ];

        for (tname, table) in [("既定", MetalRadii::default()), ("段 128", STAGE128_RADII)] {
            for (hb, gate) in [(true, false), (true, true), (false, true), (false, false)] {
                let mut e_ref = 0.0;
                let mut f_ref = vec![Vec3::zeros(); ligand.len()];
                for (i, l) in ligand.iter().enumerate() {
                    for rec in &receptor {
                        let diff = l.pos - rec.pos;
                        let r = diff.norm();
                        if r < CUTOFF && r > 1e-8 {
                            let sum = sum_radii_with(l.xs, rec.xs, table);
                            let (v, dvdr) =
                                pair_energy_grad_with(l.xs, rec.xs, r, sum, hb, gate);
                            e_ref += v;
                            f_ref[i] -= diff * (dvdr / r);
                        }
                    }
                }
                let rec_simd = ReceptorSimd::with_options(&receptor, table, hb, gate);
                let mut f_simd = vec![Vec3::zeros(); ligand.len()];
                let e_simd = intermolecular_value_grad_simd(&ligand, &rec_simd, &mut f_simd);
                let what = format!("{tname} / 金属水素結合 {hb} / 門 {gate}");
                assert!(
                    (e_simd - e_ref).abs() < 1e-3 * e_ref.abs().max(1.0),
                    "{what}: SIMD {e_simd} とスカラ {e_ref} が不一致"
                );
                for (i, (a, b)) in f_simd.iter().zip(&f_ref).enumerate() {
                    assert!(
                        (a - b).norm() < 1e-3 * b.norm().max(1.0),
                        "{what} 原子 {i}: 力が不一致 {a:?} vs {b:?}"
                    );
                }
            }
        }

        // ★ 門を開けると SIMD の値が実際に動くこと。
        // これが無いと「門を無視していても一致試験は通る」ので、欠陥を捕まえられない。
        let t = MetalRadii::default();
        let mut f_off = vec![Vec3::zeros(); ligand.len()];
        let e_off = intermolecular_value_grad_simd(
            &ligand,
            &ReceptorSimd::with_options(&receptor, t, true, false),
            &mut f_off,
        );
        let mut f_on = vec![Vec3::zeros(); ligand.len()];
        let e_on = intermolecular_value_grad_simd(
            &ligand,
            &ReceptorSimd::with_options(&receptor, t, true, true),
            &mut f_on,
        );
        assert!(
            (e_on - e_off).abs() > 0.1,
            "門を開けても SIMD の値が動かない（無視されている）: {e_off} → {e_on}"
        );
        // 門は水素結合項を**足す**ので、エネルギーは必ず下がる（重みが負）。
        assert!(e_on < e_off, "門を開けるとエネルギーは下がるはず: {e_off} → {e_on}");

        // 金属の水素結合を全部外す指定は門より強い（優先順位の固定）。
        let mut f_x = vec![Vec3::zeros(); ligand.len()];
        let e_x = intermolecular_value_grad_simd(
            &ligand,
            &ReceptorSimd::with_options(&receptor, t, false, true),
            &mut f_x,
        );
        let mut f_y = vec![Vec3::zeros(); ligand.len()];
        let e_y = intermolecular_value_grad_simd(
            &ligand,
            &ReceptorSimd::with_options(&receptor, t, false, false),
            &mut f_y,
        );
        assert!(
            (e_x - e_y).abs() < 1e-9,
            "金属の水素結合を外したら門は効いてはいけない: {e_x} vs {e_y}"
        );
    }

    /// 金属の配位半径を下げたとき、SIMD 版が**スカラ版と同じ規則**で動くことを確かめる。
    /// 大域は触らない（他の試験と並行実行しても壊れないように、半径を引数で注入する）。
    /// 項の再実装が 2 つあるので、規則の食い違いはここでしか捕まらない。
    /// 段 124 で広げた表（打ち消しの無い相手にも効く欄）でも SIMD とスカラが一致すること。
    /// SIMD 側は段 124 で「種別ごとの実効半径の配列」から「金属フラグ＋差のスカラ」方式に
    /// 作り替えたので、ここが作り替えの検算になる。
    #[test]
    fn simd_stage124_radii_match_scalar() {
        use crate::scoring::{
            pair_energy_grad_with_sum_radii, sum_radii_with, MetalRadii, STAGE124_RADII,
            STAGE124_RADII_WITH_BACKBONE,
        };
        // 亜鉛・亜鉛以外の金属の両方と、表の全欄に当たる相手を混ぜる。
        let receptor = vec![
            atom(2.0, 0.0, 0.0, XsType::MetalDZn),
            atom(-2.5, 0.5, 0.0, XsType::OxygenA),
            atom(0.0, 2.2, 0.0, XsType::CarbonH),
            atom(1.0, -2.0, 1.0, XsType::NitrogenD),
            atom(3.5, 1.0, -1.0, XsType::MetalD),
            atom(-1.5, -1.5, 1.0, XsType::Sulfur),
            atom(0.5, 1.0, -2.0, XsType::Fluorine),
            atom(-3.0, 0.0, 2.0, XsType::Phosphorus),
            atom(2.5, -2.5, 0.0, XsType::NitrogenA),
        ];
        let ligand = vec![
            atom(0.1, 0.0, 0.0, XsType::NitrogenD),   // 供与体のみの窒素（段 124 の主役）
            atom(2.02, 0.0, 0.0, XsType::NitrogenD),  // 段 124.6 の配位距離
            atom(0.0, 0.0, 1.5, XsType::CarbonH),
            atom(2.0, 0.0, 2.32, XsType::Sulfur),
            atom(-1.0, 0.0, 0.0, XsType::MetalDZn),   // リガンド側が亜鉛
            atom(4.0, 1.0, -1.5, XsType::MetalD),     // リガンド側が亜鉛以外の金属
            atom(0.6, 1.2, -2.3, XsType::Fluorine),
            atom(-2.0, 0.6, 0.2, XsType::OxygenA),
            atom(-3.2, 0.3, 2.4, XsType::Phosphorus),
        ];

        for (name, table) in [
            ("段 124", STAGE124_RADII),
            ("段 124＋後ろの原子", STAGE124_RADII_WITH_BACKBONE),
            ("既定", MetalRadii::default()),
        ] {
            let mut e_ref = 0.0;
            let mut f_ref = vec![Vec3::zeros(); ligand.len()];
            for (i, l) in ligand.iter().enumerate() {
                for rec in &receptor {
                    let diff = l.pos - rec.pos;
                    let r = diff.norm();
                    if r < CUTOFF && r > 1e-8 {
                        let sum = sum_radii_with(l.xs, rec.xs, table);
                        let (v, dvdr) = pair_energy_grad_with_sum_radii(l.xs, rec.xs, r, sum);
                        e_ref += v;
                        f_ref[i] -= diff * (dvdr / r);
                    }
                }
            }
            let rec_simd = ReceptorSimd::with_metal_radii(&receptor, table);
            let mut f_simd = vec![Vec3::zeros(); ligand.len()];
            let e_simd = intermolecular_value_grad_simd(&ligand, &rec_simd, &mut f_simd);
            assert!(
                (e_simd - e_ref).abs() < 1e-3 * e_ref.abs().max(1.0),
                "{name}: SIMD {e_simd} とスカラ {e_ref} が不一致"
            );
            for (i, (a, b)) in f_simd.iter().zip(&f_ref).enumerate() {
                assert!(
                    (a - b).norm() < 1e-3 * b.norm().max(1.0),
                    "{name} 原子 {i}: 力が不一致 {a:?} vs {b:?}"
                );
            }
        }

        // 既定と段 124 の表で**実際に値が動く**こと（動かなければ試験が無意味）。
        let mut f0 = vec![Vec3::zeros(); ligand.len()];
        let e0 = intermolecular_value_grad_simd(
            &ligand,
            &ReceptorSimd::with_metal_radii(&receptor, MetalRadii::default()),
            &mut f0,
        );
        let mut f1 = vec![Vec3::zeros(); ligand.len()];
        let e1 = intermolecular_value_grad_simd(
            &ligand,
            &ReceptorSimd::with_metal_radii(&receptor, STAGE124_RADII),
            &mut f1,
        );
        assert!((e0 - e1).abs() > 1e-3, "段 124 の表で何も変わっていない");
    }

    #[test]
    fn simd_metal_coord_radius_matches_scalar() {
        use crate::scoring::{pair_energy_grad_with_sum_radii, sum_radii_with};
        // 亜鉛（MetalD）を含む受容体と、受容原子・非受容原子を混ぜたリガンド。
        let receptor = vec![
            atom(2.0, 0.0, 0.0, XsType::MetalD),
            atom(-2.5, 0.5, 0.0, XsType::OxygenA),
            atom(0.0, 2.2, 0.0, XsType::CarbonH),
            atom(1.0, -2.0, 1.0, XsType::NitrogenDA),
            atom(3.5, 1.0, -1.0, XsType::MetalD),
        ];
        let ligand = vec![
            atom(0.1, 0.0, 0.0, XsType::NitrogenA),  // 亜鉛と配位する受容原子
            atom(1.93, 0.0, 0.0, XsType::NitrogenA), // 段 86.3 の実測配位距離
            atom(0.0, 0.0, 1.5, XsType::CarbonH),    // 配位しない相手
            atom(2.0, 0.0, 2.07, XsType::OxygenA),
            atom(-1.0, 0.0, 0.0, XsType::MetalD),    // リガンド側が金属の場合
        ];

        for &metal_r in &[1.2, 1.0, 0.9, 0.6] {
            // スカラ基準（f64）。半径の和を明示して組むので大域に依存しない。
            let mut e_ref = 0.0;
            let mut f_ref = vec![Vec3::zeros(); ligand.len()];
            for (i, l) in ligand.iter().enumerate() {
                for rec in &receptor {
                    let diff = l.pos - rec.pos;
                    let r = diff.norm();
                    if r < CUTOFF && r > 1e-8 {
                        let sum = sum_radii_with(l.xs, rec.xs, MetalRadii::acceptors_only(metal_r));
                        let (v, dvdr) = pair_energy_grad_with_sum_radii(l.xs, rec.xs, r, sum);
                        e_ref += v;
                        f_ref[i] -= diff * (dvdr / r);
                    }
                }
            }

            let rec_simd = ReceptorSimd::with_metal_radii(&receptor, MetalRadii::acceptors_only(metal_r));
            let mut f_simd = vec![Vec3::zeros(); ligand.len()];
            let e_simd = intermolecular_value_grad_simd(&ligand, &rec_simd, &mut f_simd);

            // f32 精度なので相対許容。既定（1.2）では規則が発火しないことも同時に確かめる。
            assert!(
                (e_simd - e_ref).abs() < 1e-3 * e_ref.abs().max(1.0),
                "半径 {metal_r}: SIMD {e_simd} とスカラ {e_ref} が不一致"
            );
            for (i, (a, b)) in f_simd.iter().zip(&f_ref).enumerate() {
                assert!(
                    (a - b).norm() < 1e-3 * b.norm().max(1.0),
                    "半径 {metal_r} 原子 {i}: 力が不一致 {a:?} vs {b:?}"
                );
            }
            // 規則が実際に効いていること（効いていなければ試験が無意味）。
            if metal_r < 1.2 {
                let rec_default = ReceptorSimd::with_metal_radii(&receptor, MetalRadii::default());
                let mut f0 = vec![Vec3::zeros(); ligand.len()];
                let e0 = intermolecular_value_grad_simd(&ligand, &rec_default, &mut f0);
                assert!((e_simd - e0).abs() > 1e-3, "半径 {metal_r} で何も変わっていない");
            }
        }
    }

    #[test]
    fn simd_matches_scalar_value_and_force() {
        let ligand = vec![
            atom(0.0, 0.0, 0.0, XsType::CarbonH),
            atom(1.4, 0.3, 0.2, XsType::OxygenD),
            atom(-0.8, 1.1, 0.5, XsType::NitrogenA),
        ];
        // 受容体は 8 の倍数でない数（パディング検証）。
        let receptor = vec![
            atom(3.0, 0.5, 0.2, XsType::OxygenA),
            atom(-2.0, 1.0, 0.0, XsType::CarbonH),
            atom(2.5, -1.5, 1.0, XsType::NitrogenD),
            atom(0.5, 3.0, -1.0, XsType::OxygenDA),
            atom(-1.0, -2.0, 2.0, XsType::CarbonP),
            atom(4.0, 1.0, 0.0, XsType::Sulfur),
            atom(1.0, 1.0, 3.0, XsType::Fluorine),
            atom(2.0, 2.0, 2.0, XsType::CarbonH),
            atom(-3.0, 0.0, 1.0, XsType::NitrogenDA),
            atom(0.0, -3.0, -2.0, XsType::OxygenA),
        ];

        let mut f_scalar = vec![Vec3::zeros(); ligand.len()];
        let e_scalar = intermolecular_value_grad(&ligand, &receptor, &mut f_scalar);

        let rec_simd = ReceptorSimd::new(&receptor);
        let mut f_simd = vec![Vec3::zeros(); ligand.len()];
        let e_simd = intermolecular_value_grad_simd(&ligand, &rec_simd, &mut f_simd);

        // f32 精度内で一致（Vina ゲート許容 1e-2 より十分小さい）。
        assert!((e_scalar - e_simd).abs() < 1e-3, "energy scalar {e_scalar} simd {e_simd}");
        for (a, b) in f_scalar.iter().zip(&f_simd) {
            assert!((a - b).norm() < 1e-3, "force {a:?} vs {b:?}");
        }
    }
}
