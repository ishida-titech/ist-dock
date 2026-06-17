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

/// 構造補正を適用した最終スコア（kcal/mol）。
#[inline]
pub fn apply_torsion_correction(e_inter: f64, n_rot: u32) -> f64 {
    e_inter / (1.0 + W_ROT * n_rot as f64)
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
}
