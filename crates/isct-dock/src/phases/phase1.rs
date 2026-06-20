//! Phase 1 — 代表分子による一括棄却（M3）。
//!
//! 代表分子（seed）を単一受容体へグローバルドッキングし、ベストポーズの目的関数値が
//! 棄却閾値を上回るクラスタは「基準未達」として子分子の計算を全スキップする。
//! 生き残ったクラスタは seed の **RMSD クラスタリングで区別される上位ポーズ群**（複数）を
//! Phase 2 へ引き継ぐ（子は各 seed ポーズへ配置して最良を採用＝拘束劣化の緩和）。
//! M6 でマルチ受容体アンサンブルへ拡張する。

use std::collections::HashSet;

use crate::energy::Scorer;
use crate::ligand::{Conformation, LigandModel};
use crate::search::{dock_constrained, SearchParams};

/// seed の 1 ポーズ（モード）。
pub struct SeedPose {
    pub conf: Conformation,
    /// 目的関数値（分子間＋分子内＋ボックス罰則）。
    pub energy: f64,
}

/// seed ドッキングの結果と棄却判定。
pub struct SeedResult {
    /// RMSD クラスタリングで区別される上位ポーズ群（エネルギー昇順, [0]=最良, 最大 `seed_poses` 個）。
    pub modes: Vec<SeedPose>,
    /// 一括棄却対象か（ベスト `energy > prune_threshold`）。
    pub pruned: bool,
}

impl SeedResult {
    /// ベストポーズ（最良モード）。
    pub fn best(&self) -> &SeedPose {
        &self.modes[0]
    }
}

/// seed を受容体ボックスへドッキングし、上位 `seed_poses` モードを保持して棄却判定する。
///
/// 共通コア（`core_atoms`＝全原子 idx）を **剛体**に保つため、コアを動かす torsion は固定し、
/// 位置・配向・R 基 torsion のみ大域探索する。これにより docked コアが曲がらず、子の剛体
/// 重ね合わせ（Phase 2）が厳密になる。`seed_poses` は子配置に使う保持モード数（1 で従来＝最良のみ）。
pub fn dock_seed(
    seed: &LigandModel,
    scorer: &Scorer,
    params: &SearchParams,
    core_atoms: &HashSet<usize>,
    prune_threshold: f64,
    seed_poses: usize,
) -> SeedResult {
    let free = seed.rgroup_torsions(core_atoms);
    // RMSD クラスタ済み多モード（昇順）。上位 seed_poses を保持。
    let modes: Vec<SeedPose> = dock_constrained(seed, scorer, params, &free)
        .into_iter()
        .take(seed_poses.max(1))
        .map(|m| SeedPose { conf: m.conf, energy: m.energy })
        .collect();
    let pruned = modes[0].energy > prune_threshold;
    SeedResult { modes, pruned }
}
