//! ドッキングのエネルギー評価（M2）。
//!
//! 分子間（受容体–リガンド）エネルギーを Vina 互換スコアで計算する。受容体は探索ボックス
//! ＋カットオフ余白の範囲に枝刈りして直接原子間計算のコストを抑える（grid 化は M4）。
//! 探索を確実にボックス内へ閉じ込めるため、ボックス外に出たリガンド原子に罰則を課す
//! （Vina のグリッド外ペナルティ相当）。分子内エネルギーは多 torsion 対応時に追加予定。

use isct_core::geometry::Vec3;
use isct_core::scoring::{intermolecular_terms, ScoredAtom, CUTOFF};
use isct_core::scoring_simd::{intermolecular_value_grad_simd, ReceptorSimd};

/// ボックス外ペナルティ係数（kcal/mol/Å²。はみ出しの二乗に比例）。
const BOX_PENALTY: f64 = 2.0;

/// 枝刈り済み受容体に対しリガンドポーズを採点するスコアラ。
#[derive(Debug, Clone)]
pub struct Scorer {
    receptor: Vec<ScoredAtom>,
    /// 探索の値＋勾配カーネル用の SoA（f32）受容体（`receptor` と同一原子）。
    receptor_simd: ReceptorSimd,
    center: Vec3,
    half: Vec3,
}

impl Scorer {
    /// 探索ボックス（中心・サイズ）＋カットオフ余白に入る受容体原子だけを保持する。
    pub fn new(receptor_full: &[ScoredAtom], center: Vec3, size: Vec3) -> Self {
        let half = size * 0.5;
        let margin = Vec3::new(half.x + CUTOFF, half.y + CUTOFF, half.z + CUTOFF);
        let receptor: Vec<ScoredAtom> = receptor_full
            .iter()
            .filter(|a| {
                (a.pos.x - center.x).abs() <= margin.x
                    && (a.pos.y - center.y).abs() <= margin.y
                    && (a.pos.z - center.z).abs() <= margin.z
            })
            .copied()
            .collect();
        let receptor_simd = ReceptorSimd::new(&receptor);
        Scorer {
            receptor,
            receptor_simd,
            center,
            half,
        }
    }

    /// 枝刈り後の受容体原子数。
    pub fn receptor_atoms(&self) -> usize {
        self.receptor.len()
    }

    /// 枝刈り後の受容体原子（Phase 3 の Δエネルギー評価で使う）。
    pub fn receptor(&self) -> &[ScoredAtom] {
        &self.receptor
    }

    /// リガンドポーズの分子間エネルギー（kcal/mol。ペナルティを含まない素の値）。
    pub fn intermolecular(&self, ligand: &[ScoredAtom]) -> f64 {
        intermolecular_terms(ligand, &self.receptor).weighted_sum()
    }

    /// ボックス外はみ出しの罰則（全リガンド原子の各軸はみ出し²の和×係数）。
    pub fn box_penalty(&self, ligand: &[ScoredAtom]) -> f64 {
        let mut pen = 0.0;
        for a in ligand {
            let over = |p: f64, c: f64, h: f64| -> f64 {
                let d = (p - c).abs() - h;
                if d > 0.0 { d } else { 0.0 }
            };
            let ox = over(a.pos.x, self.center.x, self.half.x);
            let oy = over(a.pos.y, self.center.y, self.half.y);
            let oz = over(a.pos.z, self.center.z, self.half.z);
            pen += ox * ox + oy * oy + oz * oz;
        }
        BOX_PENALTY * pen
    }

    /// 探索用の目的関数値 = 分子間エネルギー + ボックス外ペナルティ。
    pub fn score(&self, ligand: &[ScoredAtom]) -> f64 {
        self.intermolecular(ligand) + self.box_penalty(ligand)
    }

    /// 目的関数値（分子間＋ボックス罰則）と各リガンド原子に働く力 `force[i] = -dE/dpos_i` を
    /// 加算して返す。`force` は呼び出し側で 0 初期化（長さ = ligand.len()）。
    pub fn value_and_force(&self, ligand: &[ScoredAtom], force: &mut [Vec3]) -> f64 {
        let mut e = intermolecular_value_grad_simd(ligand, &self.receptor_simd, force);
        // ボックス外ペナルティの値と力。
        for (i, a) in ligand.iter().enumerate() {
            let over = |p: f64, c: f64, h: f64| -> f64 {
                let d = (p - c).abs() - h;
                if d > 0.0 { d } else { 0.0 }
            };
            let comp = |p: f64, c: f64, h: f64, e: &mut f64, fcomp: &mut f64| {
                let o = over(p, c, h);
                if o > 0.0 {
                    *e += BOX_PENALTY * o * o;
                    // penalty = BOX*o^2, dpenalty/dp = BOX*2*o*sign(p-c) → force = -それ。
                    *fcomp -= BOX_PENALTY * 2.0 * o * (p - c).signum();
                }
            };
            let mut fx = 0.0;
            let mut fy = 0.0;
            let mut fz = 0.0;
            comp(a.pos.x, self.center.x, self.half.x, &mut e, &mut fx);
            comp(a.pos.y, self.center.y, self.half.y, &mut e, &mut fy);
            comp(a.pos.z, self.center.z, self.half.z, &mut e, &mut fz);
            force[i] += Vec3::new(fx, fy, fz);
        }
        e
    }
}
