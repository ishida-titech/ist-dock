//! Vina スコアリング・エネルギーカーネルの SIMD 化（f32 / `wide`、高速化）。
//!
//! 探索の内側ループ（受容体全ペアの値＋勾配）を `f32x8`（AVX2 で 8 レーン）でベクトル化する。
//! 受容体は SoA（structure-of-arrays）で前計算して保持し、リガンド 1 原子 × 受容体 8 原子を
//! 一括評価する。`scoring.rs` のスカラ f64 版と関数形は同一（値は f32 精度内で一致）。
//!
//! 注: f32 精度のため探索（局所最適化）の目的関数に使う。最終スコアや `vina --score_only`
//! 一致ゲートはスカラ f64 版（`scoring::intermolecular_terms`）を引き続き使う。

use wide::{f32x8, CmpGe, CmpGt, CmpLe, CmpLt};

use crate::geometry::Vec3;
use crate::scoring::{
    ScoredAtom, CUTOFF, W_GAUSS1, W_GAUSS2, W_HBOND, W_HYDROPHOBIC, W_REPULSION,
};

const LANES: usize = 8;

/// 受容体の SoA 表現（f32、8 の倍数にパディング）。パディング原子は遠方に置きカットオフ外にする。
#[derive(Debug, Clone)]
pub struct ReceptorSimd {
    rx: Vec<f32>,
    ry: Vec<f32>,
    rz: Vec<f32>,
    radius: Vec<f32>,
    /// 疎水フラグ（1.0/0.0）。
    hydro: Vec<f32>,
    donor: Vec<f32>,
    acceptor: Vec<f32>,
}

impl ReceptorSimd {
    pub fn new(receptor: &[ScoredAtom]) -> Self {
        let n = receptor.len();
        let padded = n.div_ceil(LANES) * LANES;
        let mut rx = vec![1.0e9_f32; padded]; // 遠方パディング（カットオフ外）
        let mut ry = vec![1.0e9_f32; padded];
        let mut rz = vec![1.0e9_f32; padded];
        let mut radius = vec![0.0_f32; padded];
        let mut hydro = vec![0.0_f32; padded];
        let mut donor = vec![0.0_f32; padded];
        let mut acceptor = vec![0.0_f32; padded];
        for (i, a) in receptor.iter().enumerate() {
            rx[i] = a.pos.x as f32;
            ry[i] = a.pos.y as f32;
            rz[i] = a.pos.z as f32;
            radius[i] = a.xs.radius() as f32;
            hydro[i] = flag(a.xs.is_hydrophobic());
            donor[i] = flag(a.xs.is_donor());
            acceptor[i] = flag(a.xs.is_acceptor());
        }
        ReceptorSimd { rx, ry, rz, radius, hydro, donor, acceptor }
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

    let nch = rec.n_chunks();
    let mut total = 0.0_f64;

    for (li, l) in ligand.iter().enumerate() {
        let lx = f32x8::splat(l.pos.x as f32);
        let ly = f32x8::splat(l.pos.y as f32);
        let lz = f32x8::splat(l.pos.z as f32);
        let rl = f32x8::splat(l.xs.radius() as f32);
        let hyd_l = f32x8::splat(flag(l.xs.is_hydrophobic()));
        let don_l = f32x8::splat(flag(l.xs.is_donor()));
        let acc_l = f32x8::splat(flag(l.xs.is_acceptor()));

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
            let acc_r = chunk(&rec.acceptor, base);

            let dx = lx - rx;
            let dy = ly - ry;
            let dz = lz - rz;
            let r2 = dx * dx + dy * dy + dz * dz;
            let r = r2.sqrt();
            let d = r - (rl + rr);

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
            let hb_pair = (don_l * acc_r).max(acc_l * don_r);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atom_types::XsType;
    use crate::scoring::intermolecular_value_grad;

    fn atom(x: f64, y: f64, z: f64, xs: XsType) -> ScoredAtom {
        ScoredAtom { pos: Vec3::new(x, y, z), xs }
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
