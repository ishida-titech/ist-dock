//! Phase 1 — 代表分子による一括棄却（M3）。
//!
//! 代表分子（seed）を単一受容体へグローバルドッキングし、ベストポーズの目的関数値が
//! 棄却閾値を上回るクラスタは「基準未達」として子分子の計算を全スキップする。
//! 生き残ったクラスタは seed のベストポーズ（`Conformation`）を Phase 2 へ引き継ぐ。
//! M6 でマルチ受容体アンサンブルへ拡張する。複数ポーズの RMSD デデュープは後置。

use std::collections::HashSet;

use crate::energy::Scorer;
use crate::ligand::{Conformation, LigandModel};
use crate::search::{dock_constrained, SearchParams};

/// seed ドッキングの結果と棄却判定。
pub struct SeedResult {
    /// ベストポーズ。
    pub conf: Conformation,
    /// ベストポーズの目的関数値（分子間＋分子内＋ボックス罰則）。
    pub energy: f64,
    /// 一括棄却対象か（`energy > prune_threshold`）。
    pub pruned: bool,
}

/// seed を受容体ボックスへドッキングし、棄却判定して返す。
///
/// 共通コア（`core_atoms`＝全原子 idx）を **剛体**に保つため、コアを動かす torsion は固定し、
/// 位置・配向・R 基 torsion のみ大域探索する。これにより docked コアが曲がらず、子の剛体
/// 重ね合わせ（Phase 2）が厳密になる。
pub fn dock_seed(
    seed: &LigandModel,
    scorer: &Scorer,
    params: &SearchParams,
    core_atoms: &HashSet<usize>,
    prune_threshold: f64,
) -> SeedResult {
    let free = seed.rgroup_torsions(core_atoms);
    let result = dock_constrained(seed, scorer, params, &free);
    let pruned = result.energy > prune_threshold;
    SeedResult {
        conf: result.conf,
        energy: result.energy,
        pruned,
    }
}
