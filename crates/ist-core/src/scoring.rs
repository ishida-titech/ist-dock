//! Vina 互換スコアリング関数（Trott & Olson 2010 / Vina 1.2 "vina" 重み）。
//!
//! 原子ペア i,j、実距離 `r`、表面間距離 `d = r - (R_i + R_j)`、カットオフ `r < 8 Å`:
//!   gauss1      exp(-(d/0.5)^2)
//!   gauss2      exp(-((d-3)/2)^2)
//!   repulsion   d<0 ? d^2 : 0
//!   hydrophobic 区分線形（d<=0.5:1 → d>=1.5:0）、疎水-疎水ペアのみ
//!   hbond       区分線形（d<=-0.7:1 → d>=0:0）、ドナー-アクセプタペアのみ
//! 重み付き総和が分子間エネルギー e_inter。最終スコアは構造補正
//! `e_inter / (1 + w_rot * N_rot)`、`w_rot = 0.05846`。
//!
//! 検証ゲート: `vina --score_only` の "Ligand - Receptor" 値との一致（差 < 1e-2）。

use crate::atom_types::XsType;
use crate::geometry::{distance, Vec3};

/// 各項の重み（Vina 1.2 "vina" スコアリング）。
pub const W_GAUSS1: f64 = -0.035579;
pub const W_GAUSS2: f64 = -0.005156;
pub const W_REPULSION: f64 = 0.840245;
pub const W_HYDROPHOBIC: f64 = -0.035069;
pub const W_HBOND: f64 = -0.587439;
/// 構造補正（torsion）係数。
pub const W_ROT: f64 = 0.05846;

/// ペア相互作用のカットオフ（実距離 r、Å）。
pub const CUTOFF: f64 = 8.0;

/// スコアリング対象の重原子（位置と XS 型のみ）。水素は含めない。
#[derive(Debug, Clone, Copy)]
pub struct ScoredAtom {
    pub pos: Vec3,
    pub xs: XsType,
}

/// 5 項の素値（重み付け前）の和。
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Terms {
    pub gauss1: f64,
    pub gauss2: f64,
    pub repulsion: f64,
    pub hydrophobic: f64,
    pub hbond: f64,
}

impl Terms {
    fn add(&mut self, other: &Terms) {
        self.gauss1 += other.gauss1;
        self.gauss2 += other.gauss2;
        self.repulsion += other.repulsion;
        self.hydrophobic += other.hydrophobic;
        self.hbond += other.hbond;
    }

    /// 重み付き総和（= エネルギー寄与）。
    pub fn weighted_sum(&self) -> f64 {
        W_GAUSS1 * self.gauss1
            + W_GAUSS2 * self.gauss2
            + W_REPULSION * self.repulsion
            + W_HYDROPHOBIC * self.hydrophobic
            + W_HBOND * self.hbond
    }
}

#[inline]
fn gauss(d: f64, offset: f64, width: f64) -> f64 {
    (-((d - offset) / width).powi(2)).exp()
}

#[inline]
fn repulsion(d: f64) -> f64 {
    if d < 0.0 { d * d } else { 0.0 }
}

/// Vina の `slope_step`: `good` で 1、`bad` で 0 の線形ランプ（[0,1] にクランプ）。
#[inline]
fn slope_step(bad: f64, good: f64, x: f64) -> f64 {
    if bad < good {
        if x <= bad {
            0.0
        } else if x >= good {
            1.0
        } else {
            (x - bad) / (good - bad)
        }
    } else if x >= bad {
        0.0
    } else if x <= good {
        1.0
    } else {
        (x - bad) / (good - bad)
    }
}

#[inline]
fn hydrophobic_term(d: f64) -> f64 {
    // good = 0.5, bad = 1.5
    slope_step(1.5, 0.5, d)
}

#[inline]
fn hbond_term(d: f64) -> f64 {
    // good = -0.7, bad = 0.0
    slope_step(0.0, -0.7, d)
}

/// ドナー-アクセプタの組み合わせか（どちら向きでも可）。
#[inline]
fn h_bond_possible(a: XsType, b: XsType) -> bool {
    (a.is_donor() && b.is_acceptor()) || (a.is_acceptor() && b.is_donor())
}

/// 1 ペアの素値を評価する（カットオフ判定は呼び出し側）。
pub fn eval_pair(a: XsType, b: XsType, r: f64) -> Terms {
    let d = r - (a.radius() + b.radius());
    let mut t = Terms {
        gauss1: gauss(d, 0.0, 0.5),
        gauss2: gauss(d, 3.0, 2.0),
        repulsion: repulsion(d),
        hydrophobic: 0.0,
        hbond: 0.0,
    };
    if a.is_hydrophobic() && b.is_hydrophobic() {
        t.hydrophobic = hydrophobic_term(d);
    }
    if h_bond_possible(a, b) {
        t.hbond = hbond_term(d);
    }
    t
}

/// **反発項だけ**の重み付き値（B-1 用）。`r` は原子間距離。
///
/// 1-4 ペアに「立体反発だけ」を入れるための関数。Vina の分子内項をそのまま 1-4 に
/// 入れると引力（ガウス 2 項・疎水・水素結合）も一緒に入り、**折りたたみを促す**
/// （段 90・92.1）。反発だけなら折りたたみ圧力を増やさずに立体障害を表現できる。
/// 追加パラメータは無い（Vina の重み `W_REPULSION` と XS 半径をそのまま使う）。
#[inline]
pub fn pair_repulsion(a: XsType, b: XsType, r: f64) -> f64 {
    let d = r - (a.radius() + b.radius());
    W_REPULSION * repulsion(d)
}

/// `pair_repulsion` の値と `d/dr`。
#[inline]
pub fn pair_repulsion_grad(a: XsType, b: XsType, r: f64) -> (f64, f64) {
    let d = r - (a.radius() + b.radius());
    let (v, dv) = repulsion_value_grad(d); // dd/dr = 1
    (W_REPULSION * v, W_REPULSION * dv)
}

/// 分子間（受容体–リガンド）の素値合計。`r < CUTOFF` のペアのみ。
pub fn intermolecular_terms(ligand: &[ScoredAtom], receptor: &[ScoredAtom]) -> Terms {
    let mut total = Terms::default();
    for l in ligand {
        for rec in receptor {
            let r = distance(&l.pos, &rec.pos);
            if r < CUTOFF {
                let t = eval_pair(l.xs, rec.xs, r);
                total.add(&t);
            }
        }
    }
    total
}

/// 構造補正（torsion entropy）を適用した最終スコア（kcal/mol, Vina 互換 affinity）。
/// `num_tors` は Vina 1.2 の実効回転結合数（末端基は 0.5 カウント）＝ `Molecule::num_tors_vina()`。
/// 名目 TORSDOF ではないことに注意（末端結合を全数で数えると flexible 配位子を過剰にペナルティする）。
#[inline]
pub fn apply_torsion_correction(e_inter: f64, num_tors: f64) -> f64 {
    e_inter / (1.0 + W_ROT * num_tors)
}

/// Vina の `curl`（hunt_cap）: 探索中に**正のエネルギー（激突）を上限 `v` で滑らかに抑える**。
/// `e>0` のとき `e·v/(v+e)`（e→∞ で v に漸近）を返し、勾配は `(v/(v+e))²` 倍にスケールする。
/// 激突ポーズの巨大エネルギー・勾配が探索を支配・トラップするのを防ぎ、深い極小へ抜けやすくする。
/// 返り値 `(上限化エネルギー, 勾配スケール)`。`e<=0` や `v` 非有限なら素通り（スケール 1）。
#[inline]
pub fn curl(e: f64, v: f64) -> (f64, f64) {
    if e > 0.0 && v.is_finite() {
        let tmp = if v < 1e-12 { 0.0 } else { v / (v + e) };
        (e * tmp, tmp * tmp)
    } else {
        (e, 1.0)
    }
}

// --- 解析的勾配（M4 高速化） ---
//
// 各項の値とその d/dd（d=表面間距離、dd/dr=1 なので d/dr と等しい）を返す。`eval_pair` と
// 完全に同一の関数形を用いる（値が一致することを単体テストで担保）。

#[inline]
fn gauss_value_grad(d: f64, offset: f64, width: f64) -> (f64, f64) {
    let z = (d - offset) / width;
    let g = (-(z * z)).exp();
    // dg/dd = g * (-2 z / width)
    (g, g * (-2.0 * z / width))
}

#[inline]
fn repulsion_value_grad(d: f64) -> (f64, f64) {
    if d < 0.0 { (d * d, 2.0 * d) } else { (0.0, 0.0) }
}

/// 疎水項（slope_step(1.5, 0.5, d)）の値と d/dd。
#[inline]
fn hydrophobic_value_grad(d: f64) -> (f64, f64) {
    if d <= 0.5 {
        (1.0, 0.0)
    } else if d >= 1.5 {
        (0.0, 0.0)
    } else {
        (1.5 - d, -1.0)
    }
}

/// 水素結合項（slope_step(0.0, -0.7, d)）の値と d/dd。
#[inline]
fn hbond_value_grad(d: f64) -> (f64, f64) {
    if d <= -0.7 {
        (1.0, 0.0)
    } else if d >= 0.0 {
        (0.0, 0.0)
    } else {
        (-d / 0.7, -1.0 / 0.7)
    }
}

/// 1 ペアの重み付きエネルギーと d(energy)/dr を返す（カットオフ判定は呼び出し側）。
pub fn pair_energy_grad(a: XsType, b: XsType, r: f64) -> (f64, f64) {
    let d = r - (a.radius() + b.radius());
    let (g1, dg1) = gauss_value_grad(d, 0.0, 0.5);
    let (g2, dg2) = gauss_value_grad(d, 3.0, 2.0);
    let (rep, drep) = repulsion_value_grad(d);
    let (hyd, dhyd) = if a.is_hydrophobic() && b.is_hydrophobic() {
        hydrophobic_value_grad(d)
    } else {
        (0.0, 0.0)
    };
    let (hb, dhb) = if h_bond_possible(a, b) {
        hbond_value_grad(d)
    } else {
        (0.0, 0.0)
    };
    let value =
        W_GAUSS1 * g1 + W_GAUSS2 * g2 + W_REPULSION * rep + W_HYDROPHOBIC * hyd + W_HBOND * hb;
    let dvalue =
        W_GAUSS1 * dg1 + W_GAUSS2 * dg2 + W_REPULSION * drep + W_HYDROPHOBIC * dhyd + W_HBOND * dhb;
    (value, dvalue)
}

/// 分子間エネルギー（重み付き）と、各リガンド原子に働く力 `force[i] = -dE/dpos_i` を計算する。
/// `force` は長さ `ligand.len()`、呼び出し側で 0 初期化しておく（加算する）。
pub fn intermolecular_value_grad(
    ligand: &[ScoredAtom],
    receptor: &[ScoredAtom],
    force: &mut [Vec3],
) -> f64 {
    let mut e = 0.0;
    for (i, l) in ligand.iter().enumerate() {
        for rec in receptor {
            let diff = l.pos - rec.pos;
            let r = diff.norm();
            if r < CUTOFF && r > 1e-8 {
                let (v, dvdr) = pair_energy_grad(l.xs, rec.xs, r);
                e += v;
                // dE/dpos_i = dvdr * diff/r → force_i = -dvdr/r * diff
                force[i] -= diff * (dvdr / r);
            }
        }
    }
    e
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gauss_peaks() {
        // gauss1 は d=0 で 1、gauss2 は d=3 で 1。
        assert!((gauss(0.0, 0.0, 0.5) - 1.0).abs() < 1e-12);
        assert!((gauss(3.0, 3.0, 2.0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn repulsion_only_on_clash() {
        assert_eq!(repulsion(0.5), 0.0);
        assert!((repulsion(-0.5) - 0.25).abs() < 1e-12);
    }

    #[test]
    fn hydrophobic_ramp() {
        assert_eq!(hydrophobic_term(0.4), 1.0);
        assert_eq!(hydrophobic_term(1.6), 0.0);
        assert!((hydrophobic_term(1.0) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn hbond_ramp() {
        assert_eq!(hbond_term(-0.8), 1.0);
        assert_eq!(hbond_term(0.1), 0.0);
        assert!((hbond_term(-0.35) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn hydrophobic_only_between_hydrophobic_atoms() {
        // C_H–C_H は疎水項あり、C_H–O_A は無し。
        let t = eval_pair(XsType::CarbonH, XsType::CarbonH, 3.9);
        assert!(t.hydrophobic > 0.0);
        let t2 = eval_pair(XsType::CarbonH, XsType::OxygenA, 3.6);
        assert_eq!(t2.hydrophobic, 0.0);
    }

    #[test]
    fn hbond_only_donor_acceptor() {
        // ドナー(O_D)–アクセプタ(O_A) は hbond あり。
        let t = eval_pair(XsType::OxygenD, XsType::OxygenA, 3.0);
        assert!(t.hbond > 0.0);
        // アクセプタ同士は無し。
        let t2 = eval_pair(XsType::OxygenA, XsType::OxygenA, 3.0);
        assert_eq!(t2.hbond, 0.0);
    }

    #[test]
    fn pair_grad_value_matches_eval_pair() {
        // pair_energy_grad の値が eval_pair の重み付き和と一致する（複数の型・距離で）。
        let types = [
            XsType::CarbonH,
            XsType::OxygenA,
            XsType::OxygenD,
            XsType::NitrogenD,
        ];
        for &a in &types {
            for &b in &types {
                for &r in &[2.0, 3.5, 4.0, 5.5, 7.9] {
                    let (v, _) = pair_energy_grad(a, b, r);
                    let want = eval_pair(a, b, r).weighted_sum();
                    assert!((v - want).abs() < 1e-12, "{a:?},{b:?},{r}: {v} vs {want}");
                }
            }
        }
    }

    #[test]
    fn pair_grad_matches_numerical() {
        // d(energy)/dr が中心差分と一致する（区分線形の折れ点を避けた点で）。
        let types = [XsType::CarbonH, XsType::OxygenA, XsType::OxygenD];
        let eps = 1e-6;
        for &a in &types {
            for &b in &types {
                for &r in &[2.3, 3.1, 4.4, 5.2, 6.7] {
                    let (_, dv) = pair_energy_grad(a, b, r);
                    let (vp, _) = pair_energy_grad(a, b, r + eps);
                    let (vm, _) = pair_energy_grad(a, b, r - eps);
                    let num = (vp - vm) / (2.0 * eps);
                    assert!((dv - num).abs() < 1e-4, "{a:?},{b:?},{r}: {dv} vs {num}");
                }
            }
        }
    }

    #[test]
    fn intermolecular_grad_value_and_force() {
        let ligand = vec![
            ScoredAtom { pos: Vec3::new(0.0, 0.0, 0.0), xs: XsType::CarbonH },
            ScoredAtom { pos: Vec3::new(1.4, 0.3, 0.0), xs: XsType::OxygenD },
        ];
        let receptor = vec![
            ScoredAtom { pos: Vec3::new(3.0, 0.5, 0.2), xs: XsType::OxygenA },
            ScoredAtom { pos: Vec3::new(-2.0, 1.0, 0.0), xs: XsType::CarbonH },
            ScoredAtom { pos: Vec3::new(2.5, -1.5, 1.0), xs: XsType::NitrogenA },
        ];
        let mut force = vec![Vec3::zeros(); ligand.len()];
        let e = intermolecular_value_grad(&ligand, &receptor, &mut force);
        // 値はフル再計算と一致。
        let want = intermolecular_terms(&ligand, &receptor).weighted_sum();
        assert!((e - want).abs() < 1e-12, "{e} vs {want}");

        // 力 = -dE/dpos を、各リガンド原子で数値微分と比較。
        let eps = 1e-6;
        for (i, f) in force.iter().enumerate() {
            let mut num = Vec3::zeros();
            for axis in 0..3 {
                let mut lp = ligand.clone();
                lp[i].pos[axis] += eps;
                let ep = intermolecular_terms(&lp, &receptor).weighted_sum();
                let mut lm = ligand.clone();
                lm[i].pos[axis] -= eps;
                let em = intermolecular_terms(&lm, &receptor).weighted_sum();
                num[axis] = -(ep - em) / (2.0 * eps); // force = -grad
            }
            assert!((f - num).norm() < 1e-4, "atom {i}: force {f:?} vs {num:?}");
        }
    }
}
