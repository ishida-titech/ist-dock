//! ドッキングのエネルギー評価（M2 / 高速化）。
//!
//! 分子間（受容体–リガンド）エネルギーを Vina 互換スコアで計算する。受容体は探索ボックス
//! ＋カットオフ余白の範囲に枝刈りする。探索の値＋勾配は **バックエンド切替**:
//! `Direct`（f32 SIMD 直接原子間）/ `Grid`（Vina グリッドの trilinear 補間）。
//! 最終スコア（`intermolecular`）と `vina --score_only` 一致ゲートは常に厳密 f64。
//! 探索を確実にボックス内へ閉じ込めるため、ボックス外に出たリガンド原子に罰則を課す。

use ist_core::atom_types::XsType;
use ist_core::geometry::Vec3;
use ist_core::grid::VinaGrid;
use ist_core::scoring::{intermolecular_terms, intermolecular_value_grad, ScoredAtom, CUTOFF};
use ist_core::scoring_simd::{intermolecular_value_grad_simd, ReceptorSimd};

/// ボックス外ペナルティ係数（kcal/mol/Å²。はみ出しの二乗に比例）。
const BOX_PENALTY: f64 = 2.0;

/// グリッド既定の格子間隔（Å）と探索窓への余白（Å）。
///
/// `GRID_MARGIN` は探索中にリガンド原子が箱外へはみ出しても正しいグリッド値を引ける緩衝。
/// margin シェルはグリッド体積の (box+2·margin)³ を占め構築費を大きく左右する（margin=4 で箱の 2.74 倍）。
/// margin スイープ実測（3ptb/1stp/1iep＋diverse 8 標的, docs/results.202607071...）で **2.0 を採用**:
/// margin=0 は 1lpz が 3.19Å へ退行（箱端 clamp）、margin≥1 で回復。2.0 は構築費を ~30% 削りつつ
/// 全 11 標的で RMSD<2Å 中立・Vina 相当の単発 dock 速度（3ptb で Vina 0.92s を下回る）を確保する安全側。
pub const GRID_SPACING: f64 = 0.375;
pub const GRID_MARGIN: f64 = 2.0;

/// 実効グリッド余白（Å）を解決する。基本は引数 `config_margin`（`[run].grid_margin` 由来, 既定 `GRID_MARGIN`）
/// だが、環境変数 `IST_GRID_MARGIN` が有効な非負値ならそれを優先する（スイープ検証・チューニング用の
/// デバッグ上書き）。探索表面のみに影響し、最終スコアは厳密 f64 polish で算出するため報告値には効かない。
fn effective_grid_margin(config_margin: f64) -> f64 {
    std::env::var("IST_GRID_MARGIN")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .filter(|m| m.is_finite() && *m >= 0.0)
        .unwrap_or(config_margin)
}

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
    /// （探索の並列実行前に呼ぶこと）。`spacing` は `GRID_SPACING`、余白は `margin`（`[run].grid_margin`
    /// 由来。既定 `GRID_MARGIN`=2.0。環境変数 `IST_GRID_MARGIN` があればそれが優先）。
    pub fn new_grid(
        receptor_full: &[ScoredAtom],
        center: Vec3,
        size: Vec3,
        ligand_types: &[XsType],
        margin: f64,
    ) -> Self {
        let (receptor, half) = Self::prune(receptor_full, center, size);
        let mut grid =
            VinaGrid::new(&receptor, center, size, GRID_SPACING, effective_grid_margin(margin));
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

    /// 厳密な目的関数値 = 厳密分子間（direct）+ ボックス外ペナルティ。報告・polish の直線探索用。
    pub fn score(&self, ligand: &[ScoredAtom]) -> f64 {
        self.intermolecular(ligand) + self.box_penalty(ligand)
    }

    /// **探索バックエンドに整合した**目的関数値 = 分子間（Grid なら補間, Direct なら直接）+ ボックス罰則。
    /// 探索の直線探索（値のみ評価, 1 BFGS 反復あたり最大 ~40 回）で使う。Grid 時は受容体全原子の直接和
    /// （O(n_lig×n_rec)）を避け補間（O(n_lig)）で大幅に高速化し、勾配（value_and_force）と同一表面に揃える。
    /// `cap`（hunt_cap）= 探索中の正エネルギー上限。無効化は `f64::INFINITY`。Direct は未適用（既定外）。
    pub fn search_score(&self, ligand: &[ScoredAtom], cap: f64) -> f64 {
        let inter = match &self.backend {
            Backend::Direct(_) => self.intermolecular(ligand),
            Backend::Grid(grid) => grid.value(ligand, cap),
        };
        inter + self.box_penalty(ligand)
    }

    /// 目的関数値（分子間＋ボックス罰則）と各リガンド原子に働く力 `force[i] = -dE/dpos_i` を
    /// 加算して返す。`force` は呼び出し側で 0 初期化（長さ = ligand.len()）。
    /// `cap`（hunt_cap）= 探索中の正エネルギー上限。無効化は `f64::INFINITY`。Direct は未適用（既定外）。
    pub fn value_and_force(&self, ligand: &[ScoredAtom], force: &mut [Vec3], cap: f64) -> f64 {
        let e = match &self.backend {
            Backend::Direct(rec) => intermolecular_value_grad_simd(ligand, rec, force),
            Backend::Grid(grid) => grid.value_and_force(ligand, force, cap),
        };
        e + self.add_box_penalty_force(ligand, force)
    }

    /// 厳密 f64 直接計算の分子間値＋力（バックエンドに依らない）。探索後の最終精密化（polish）と
    /// 報告スコアの整合に使う。grid バックエンドでも真の極小へ refine するための厳密勾配。
    pub fn exact_value_and_force(&self, ligand: &[ScoredAtom], force: &mut [Vec3]) -> f64 {
        let e = intermolecular_value_grad(ligand, &self.receptor, force);
        e + self.add_box_penalty_force(ligand, force)
    }

    /// ボックス外ペナルティの値を返し、対応する力を `force` に加算する。
    fn add_box_penalty_force(&self, ligand: &[ScoredAtom], force: &mut [Vec3]) -> f64 {
        let mut pen = 0.0;
        for (i, a) in ligand.iter().enumerate() {
            let over = |p: f64, c: f64, h: f64| -> f64 {
                let d = (p - c).abs() - h;
                if d > 0.0 { d } else { 0.0 }
            };
            let comp = |p: f64, c: f64, h: f64, e: &mut f64, fcomp: &mut f64| {
                let o = over(p, c, h);
                if o > 0.0 {
                    *e += BOX_PENALTY * o * o;
                    *fcomp -= BOX_PENALTY * 2.0 * o * (p - c).signum();
                }
            };
            let mut fx = 0.0;
            let mut fy = 0.0;
            let mut fz = 0.0;
            comp(a.pos.x, self.center.x, self.half.x, &mut pen, &mut fx);
            comp(a.pos.y, self.center.y, self.half.y, &mut pen, &mut fy);
            comp(a.pos.z, self.center.z, self.half.z, &mut pen, &mut fz);
            force[i] += Vec3::new(fx, fy, fz);
        }
        pen
    }
}
