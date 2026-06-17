//! スクリーニング・オーケストレーション（M3）。
//!
//! クラスタライブラリを走査し、クラスタごとに Phase 1（seed ドッキング＋一括棄却）→
//! Phase 2（子の核固定配置）→ Phase 3（R 基 torsion 最適化＋Δエネルギー）を実行して、
//! 全分子のスコアとクラスタ生存統計を集計する。単一受容体・固定シードで決定論的。

use isct_core::geometry::Vec3;
use isct_core::scoring::{apply_torsion_correction, ScoredAtom};
use isct_io::library::{Library, MolEntry};

use crate::energy::Scorer;
use crate::ligand::LigandModel;
use crate::phases::{phase1, phase2, phase3};
use crate::search::SearchParams;

/// スクリーニング実行パラメータ。
#[derive(Debug, Clone)]
pub struct ScreenParams {
    /// 既定の探索ボックス中心（クラスタが pocket を持たない場合に使用）。
    pub center: Vec3,
    /// 既定の探索ボックスサイズ。
    pub size: Vec3,
    /// seed ドッキングの探索パラメータ（center/size はクラスタごとに上書き）。
    pub search: SearchParams,
    /// 一括棄却の閾値（seed 目的関数値がこれを超えると棄却）。
    pub prune_threshold: f64,
    /// 子分子 R 基最適化の最大反復。
    pub local_iters: usize,
}

/// 1 分子のスコア行。
#[derive(Debug, Clone)]
pub struct ScoredEntry {
    pub compound_id: String,
    pub cluster_id: String,
    /// 構造補正後の最終スコア（Vina 互換 affinity, kcal/mol）。
    pub total_score: f64,
    /// 分子間エネルギー（VdW 相当。Vina 5 項の合計）。
    pub e_inter: f64,
    /// 役割（"seed" または "child"）。
    pub role: &'static str,
}

/// クラスタ単位の処理結果。
#[derive(Debug, Clone)]
pub struct ClusterOutcome {
    pub cluster_id: String,
    pub seed_energy: f64,
    pub pruned: bool,
    /// 子分子数（棄却時はスキップされた数）。
    pub n_children: usize,
}

/// スクリーニング全体のレポート。
#[derive(Debug, Clone)]
pub struct ScreenReport {
    pub entries: Vec<ScoredEntry>,
    pub clusters: Vec<ClusterOutcome>,
}

impl ScreenReport {
    pub fn n_clusters(&self) -> usize {
        self.clusters.len()
    }
    pub fn n_pruned_clusters(&self) -> usize {
        self.clusters.iter().filter(|c| c.pruned).count()
    }
    /// 棄却によりスキップされた子分子数。
    pub fn n_skipped_children(&self) -> usize {
        self.clusters
            .iter()
            .filter(|c| c.pruned)
            .map(|c| c.n_children)
            .sum()
    }
}

/// ライブラリをスクリーニングする。
pub fn screen(receptor: &[ScoredAtom], lib: &Library, params: &ScreenParams) -> ScreenReport {
    let mut entries = Vec::new();
    let mut clusters = Vec::new();

    for cluster in &lib.clusters {
        // ポケット（探索ボックス）はクラスタ指定 > 既定。
        let (center, size) = match &cluster.pocket {
            Some(p) => (
                Vec3::new(p.center[0], p.center[1], p.center[2]),
                Vec3::new(p.size[0], p.size[1], p.size[2]),
            ),
            None => (params.center, params.size),
        };
        let scorer = Scorer::new(receptor, center, size);
        let mut sp = params.search.clone();
        sp.center = center;
        sp.size = size;

        // Phase 1: seed ドッキング＋一括棄却（コアを剛体に保つ）。
        let seed_lig = LigandModel::new(&cluster.seed.molecule);
        let seed_core: std::collections::HashSet<usize> =
            cluster.seed.core_atoms.iter().copied().collect();
        let seed_res = phase1::dock_seed(&seed_lig, &scorer, &sp, &seed_core, params.prune_threshold);

        clusters.push(ClusterOutcome {
            cluster_id: cluster.id.clone(),
            seed_energy: seed_res.energy,
            pruned: seed_res.pruned,
            n_children: cluster.children.len(),
        });

        if seed_res.pruned {
            // クラスタ一括棄却: 子分子は計算しない。
            continue;
        }

        // 生存クラスタ: seed もランキングに含める。
        let seed_atoms = seed_lig.scored_atoms(&seed_res.conf);
        let seed_inter = scorer.intermolecular(&seed_atoms);
        entries.push(ScoredEntry {
            compound_id: cluster.seed.compound_id.clone(),
            cluster_id: cluster.id.clone(),
            total_score: apply_torsion_correction(seed_inter, cluster.seed.molecule.torsdof),
            e_inter: seed_inter,
            role: "seed",
        });

        // docked seed のコア world 座標（Phase 2 の重ね合わせ目標）。
        let seed_world = seed_lig.world_coords(&seed_res.conf);
        let seed_core_world: Vec<Vec3> =
            cluster.seed.core_atoms.iter().map(|&i| seed_world[i]).collect();

        for child in &cluster.children {
            entries.push(score_child(&scorer, child, &cluster.id, &seed_core_world, params));
        }
    }

    ScreenReport { entries, clusters }
}

/// 1 子分子を Phase 2 + Phase 3 で配置・最適化してスコア行を作る。
fn score_child(
    scorer: &Scorer,
    child: &MolEntry,
    cluster_id: &str,
    seed_core_world: &[Vec3],
    params: &ScreenParams,
) -> ScoredEntry {
    let child_lig = LigandModel::new(&child.molecule);

    // 子の MCS コア参照 world 座標（seed.core_atoms と同順対応）。
    let child_core_ref: Vec<Vec3> =
        child.core_atoms.iter().map(|&i| child_lig.ref_coords()[i]).collect();

    // Phase 2: コアを docked seed コアへ重ねて核固定配置。
    let start = phase2::place_child(&child_lig, &child_core_ref, seed_core_world);

    // Phase 3: R 基 torsion 最適化＋Δエネルギー（コア＝全原子 idx 集合）。
    let core_set: std::collections::HashSet<usize> = child.core_atoms.iter().copied().collect();
    let result = phase3::optimize_child(&child_lig, scorer, start, &core_set, params.local_iters);

    ScoredEntry {
        compound_id: child.compound_id.clone(),
        cluster_id: cluster_id.to_string(),
        total_score: apply_torsion_correction(result.e_inter, child.molecule.torsdof),
        e_inter: result.e_inter,
        role: "child",
    }
}

/// ランキング TSV（仕様 4.2）を生成する。スコア昇順（最も負＝最良）に Rank を振る。
/// `Electrostatic_Energy` は Vina 関数に静電項が無いため 0、`Target_Protein_State_ID` は
/// 単一受容体のため 0 固定。
pub fn ranking_tsv(report: &ScreenReport) -> String {
    let mut ranked: Vec<&ScoredEntry> = report.entries.iter().collect();
    ranked.sort_by(|a, b| a.total_score.partial_cmp(&b.total_score).unwrap());

    let mut out = String::new();
    out.push_str("Rank\tCompound_ID\tCluster_ID\tTotal_Score\tVdW_Energy\tElectrostatic_Energy\tTarget_Protein_State_ID\n");
    for (i, e) in ranked.iter().enumerate() {
        out.push_str(&format!(
            "{}\t{}\t{}\t{:.4}\t{:.4}\t{:.4}\t{}\n",
            i + 1,
            e.compound_id,
            e.cluster_id,
            e.total_score,
            e.e_inter,
            0.0,
            0,
        ));
    }
    out
}
