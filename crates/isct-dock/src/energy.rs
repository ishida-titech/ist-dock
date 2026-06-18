//! ドッキングのエネルギー評価（M2 / 高速化）。
//!
//! 分子間（受容体–リガンド）エネルギーを Vina 互換スコアで計算する。受容体は探索ボックス
//! ＋カットオフ余白の範囲に枝刈りする。探索の値＋勾配は **バックエンド切替**:
//! `Direct`（f32 SIMD 直接原子間）/ `Grid`（Vina グリッドの trilinear 補間）。
//! 最終スコア（`intermolecular`）と `vina --score_only` 一致ゲートは常に厳密 f64。
//! 探索を確実にボックス内へ閉じ込めるため、ボックス外に出たリガンド原子に罰則を課す。

use isct_core::atom_types::XsType;
use isct_core::geometry::Vec3;
use isct_core::grid::VinaGrid;
use isct_core::scoring::{intermolecular_terms, ScoredAtom, CUTOFF};
use isct_core::scoring_simd::{intermolecular_value_grad_simd, ReceptorSimd};

/// ボックス外ペナルティ係数（kcal/mol/Å²。はみ出しの二乗に比例）。
const BOX_PENALTY: f64 = 2.0;

/// グリッド既定の格子間隔（Å）と探索窓への余白（Å）。
pub const GRID_SPACING: f64 = 0.375;
pub const GRID_MARGIN: f64 = 4.0;

/// 探索の分子間「値＋力」評価バックエンド。
#[derive(Debug, Clone)]
enum Backend {
    /// f32 SIMD 直接原子間計算（将来の非分解スコア・水分子もこの系統）。
    Direct(ReceptorSimd),
    /// Vina グリッドの trilinear 補間（探索専用近似）。
    Grid(VinaGrid),
}

/// 枝刈り済み受容体に対しリガンドポーズを採点するスコアラ。
#[derive(Debug, Clone)]
pub struct Scorer {
    receptor: Vec<ScoredAtom>,
    /// 探索の値＋勾配バックエンド（`receptor` と同一原子）。
    backend: Backend,
    center: Vec3,
    half: Vec3,
}

impl Scorer {
    /// 探索ボックス＋カットオフ余白に入る受容体原子を枝刈りする（共通）。
    fn prune(receptor_full: &[ScoredAtom], center: Vec3, size: Vec3) -> (Vec<ScoredAtom>, Vec3) {
        let half = size * 0.5;
        let margin = Vec3::new(half.x + CUTOFF, half.y + CUTOFF, half.z + CUTOFF);
        let receptor = receptor_full
            .iter()
            .filter(|a| {
                (a.pos.x - center.x).abs() <= margin.x
                    && (a.pos.y - center.y).abs() <= margin.y
                    && (a.pos.z - center.z).abs() <= margin.z
            })
            .copied()
            .collect();
        (receptor, half)
    }

    /// 直接（SIMD）バックエンドの Scorer を作る（既定）。
    pub fn new(receptor_full: &[ScoredAtom], center: Vec3, size: Vec3) -> Self {
        let (receptor, half) = Self::prune(receptor_full, center, size);
        let backend = Backend::Direct(ReceptorSimd::new(&receptor));
        Scorer {
            receptor,
            backend,
            center,
            half,
        }
    }

    /// グリッドバックエンドの Scorer を作る。`ligand_types` のマップを事前生成する
    /// （探索の並列実行前に呼ぶこと）。`spacing`/`margin` は既定 `GRID_SPACING`/`GRID_MARGIN`。
    pub fn new_grid(
        receptor_full: &[ScoredAtom],
        center: Vec3,
        size: Vec3,
        ligand_types: &[XsType],
    ) -> Self {
        let (receptor, half) = Self::prune(receptor_full, center, size);
        let mut grid = VinaGrid::new(&receptor, center, size, GRID_SPACING, GRID_MARGIN);
        grid.ensure(ligand_types);
        Scorer {
            receptor,
            backend: Backend::Grid(grid),
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
        let mut e = match &self.backend {
            Backend::Direct(rec) => intermolecular_value_grad_simd(ligand, rec, force),
            Backend::Grid(grid) => grid.value_and_force(ligand, force),
        };
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
