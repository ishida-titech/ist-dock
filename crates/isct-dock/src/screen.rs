//! スクリーニング・オーケストレーション（M3）。
//!
//! クラスタライブラリを走査し、クラスタごとに Phase 1（seed ドッキング＋一括棄却）→
//! Phase 2（子の核固定配置）→ Phase 3（R 基 torsion 最適化＋Δエネルギー）を実行して、
//! 全分子のスコアとクラスタ生存統計を集計する。単一受容体・固定シードで決定論的。

use std::borrow::Cow;

use isct_core::atom_types::XsType;
use isct_core::geometry::Vec3;
use isct_core::scoring::{apply_torsion_correction, ScoredAtom};
use isct_io::library::{Cluster, Library, MolEntry};
use isct_io::library_bin::BinaryLibrary;
use isct_io::pdbqt;
use rayon::prelude::*;

use crate::energy::Scorer;
use crate::ligand::LigandModel;
use crate::phases::{phase1, phase2, phase3};
use crate::search::SearchParams;

/// クラスタの供給元（in-RAM の [`Library`] か memmap streaming の [`BinaryLibrary`]）。
/// `screen` をこの trait に汎用化し、バイナリでは**クラスタ単位の遅延デコード**でメモリ有界にする。
pub trait ClusterSource: Sync {
    fn n_clusters(&self) -> usize;
    /// クラスタ `i`（Library は借用、Binary は遅延デコードの所有値）。
    fn cluster(&self, i: usize) -> Cow<'_, Cluster>;
    /// pocket 未指定（既定ボックス）クラスタが存在するか（共有スコアラ構築の要否）。
    fn has_default_box(&self) -> bool;
    /// 既定ボックスクラスタの XS 型 union（共有 grid マップ生成用）。
    fn shared_xs_types(&self) -> Vec<XsType>;
}

impl ClusterSource for Library {
    fn n_clusters(&self) -> usize {
        self.clusters.len()
    }
    fn cluster(&self, i: usize) -> Cow<'_, Cluster> {
        Cow::Borrowed(&self.clusters[i])
    }
    fn has_default_box(&self) -> bool {
        self.clusters.iter().any(|c| c.pocket.is_none())
    }
    fn shared_xs_types(&self) -> Vec<XsType> {
        shared_xs_types(self)
    }
}

impl ClusterSource for BinaryLibrary {
    fn n_clusters(&self) -> usize {
        self.n_clusters()
    }
    fn cluster(&self, i: usize) -> Cow<'_, Cluster> {
        Cow::Owned(self.cluster(i).expect("binary library: cluster decode failed (corrupt file)"))
    }
    fn has_default_box(&self) -> bool {
        self.has_default_box()
    }
    fn shared_xs_types(&self) -> Vec<XsType> {
        self.xs_union().to_vec()
    }
}

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
    /// グリッドバックエンドを使うか（クラスタごとに seed+子の型でマップ生成）。
    pub use_grid: bool,
    /// 子配置に使う seed の保持ポーズ数（RMSD クラスタ済み上位 N）。1 で従来＝最良のみ。
    /// >1 で各子を複数 seed ポーズへ配置し最良を採用＝剛体コア固定の拘束劣化を緩和。
    pub seed_poses: usize,
    /// ベストポーズ座標を保持するか（ポーズ出力用。メモリ節約のため出力要求時のみ true）。
    pub keep_poses: bool,
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
    /// ベストポーズの PDBQT ボディ（`write_pose` 済み, `keep_poses` 有効時のみ）。ライブラリ非依存に
    /// 採点時へ書き込む（streaming 時に元 PDBQT を引けないため）。
    pub pose: Option<String>,
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

/// ライブラリ（in-RAM `Library` / streaming `BinaryLibrary`）をスクリーニングする。
pub fn screen<S: ClusterSource>(receptor: &[ScoredAtom], src: &S, params: &ScreenParams) -> ScreenReport {
    // 既定ボックス（pocket 未指定クラスタ用）のスコアラを **1 度だけ**構築して共有する。grid マップは
    // (受容体, ボックス, 型) のみに依存＝全 pocket 未指定クラスタで同一なので、クラスタ毎の再構築を避ける。
    // pocket 指定クラスタは process_cluster 内でローカル構築（フォールバック）。出力はビット一致・決定的。
    let shared: Option<Scorer> = src.has_default_box().then(|| {
        if params.use_grid {
            Scorer::new_grid(receptor, params.center, params.size, &src.shared_xs_types())
        } else {
            Scorer::new(receptor, params.center, params.size)
        }
    });

    // クラスタは独立 → 並列処理。binary は cluster() で遅延デコード→処理→破棄でメモリ有界。
    // インデックス順 collect で出力順序を保ち決定的。
    let per_cluster: Vec<(ClusterOutcome, Vec<ScoredEntry>)> = (0..src.n_clusters())
        .into_par_iter()
        .map(|i| process_cluster(receptor, &src.cluster(i), params, shared.as_ref()))
        .collect();

    let mut entries = Vec::new();
    let mut clusters = Vec::with_capacity(per_cluster.len());
    for (outcome, ents) in per_cluster {
        clusters.push(outcome);
        entries.extend(ents);
    }
    ScreenReport { entries, clusters }
}

/// ライブラリ中で **既定ボックスを使う（pocket 未指定）クラスタ**の seed＋子の XS 型の union。
/// 共有 grid スコアラのマップ生成に使う（≤17 型・典型 ~10）。
fn shared_xs_types(lib: &Library) -> Vec<isct_core::atom_types::XsType> {
    let mut set = std::collections::HashSet::new();
    for c in lib.clusters.iter().filter(|c| c.pocket.is_none()) {
        for a in c.seed.molecule.scored_atoms() {
            set.insert(a.xs);
        }
        for child in &c.children {
            for a in child.molecule.scored_atoms() {
                set.insert(a.xs);
            }
        }
    }
    set.into_iter().collect()
}

/// 1 クラスタ専用のローカルスコアラを構築する（pocket 指定クラスタ用フォールバック）。
/// grid 時は当該クラスタの seed＋子の型のみマップ生成。
fn build_local(receptor: &[ScoredAtom], center: Vec3, size: Vec3, params: &ScreenParams, cluster: &Cluster) -> Scorer {
    if params.use_grid {
        let mut types: Vec<_> = cluster.seed.molecule.scored_atoms().iter().map(|a| a.xs).collect();
        for child in &cluster.children {
            types.extend(child.molecule.scored_atoms().iter().map(|a| a.xs));
        }
        Scorer::new_grid(receptor, center, size, &types)
    } else {
        Scorer::new(receptor, center, size)
    }
}

/// 1 クラスタを Phase 1〜3 で処理し、クラスタ結果とスコア行（棄却時は空）を返す。
/// `shared` は既定ボックスの共有スコアラ（pocket 未指定クラスタはこれを再利用＝grid 再構築回避）。
fn process_cluster(
    receptor: &[ScoredAtom],
    cluster: &Cluster,
    params: &ScreenParams,
    shared: Option<&Scorer>,
) -> (ClusterOutcome, Vec<ScoredEntry>) {
    // ポケット（探索ボックス）はクラスタ指定 > 既定。
    let (center, size) = match &cluster.pocket {
        Some(p) => (
            Vec3::new(p.center[0], p.center[1], p.center[2]),
            Vec3::new(p.size[0], p.size[1], p.size[2]),
        ),
        None => (params.center, params.size),
    };
    // pocket 未指定 かつ 共有あり → 共有スコアラを再利用。さもなくばローカル構築（pocket 指定の例外）。
    let local;
    let scorer: &Scorer = match shared {
        Some(s) if cluster.pocket.is_none() => s,
        _ => {
            local = build_local(receptor, center, size, params, cluster);
            &local
        }
    };
    let mut sp = params.search.clone();
    sp.center = center;
    sp.size = size;

    // Phase 1: seed ドッキング＋一括棄却（コアを剛体に保つ）。上位 seed_poses モードを保持。
    let seed_lig = LigandModel::new(&cluster.seed.molecule);
    let seed_core: std::collections::HashSet<usize> =
        cluster.seed.core_atoms.iter().copied().collect();
    let seed_res =
        phase1::dock_seed(&seed_lig, scorer, &sp, &seed_core, params.prune_threshold, params.seed_poses);
    let best = seed_res.best();

    let outcome = ClusterOutcome {
        cluster_id: cluster.id.clone(),
        seed_energy: best.energy,
        pruned: seed_res.pruned,
        n_children: cluster.children.len(),
    };

    if seed_res.pruned {
        // クラスタ一括棄却: 子分子は計算しない。
        return (outcome, Vec::new());
    }

    let mut entries = Vec::with_capacity(1 + cluster.children.len());

    // 生存クラスタ: seed もランキングに含める（ベストポーズで採点）。
    let seed_atoms = seed_lig.scored_atoms(&best.conf);
    let seed_inter = scorer.intermolecular(&seed_atoms);
    entries.push(ScoredEntry {
        compound_id: cluster.seed.compound_id.clone(),
        cluster_id: cluster.id.clone(),
        total_score: apply_torsion_correction(seed_inter, cluster.seed.molecule.torsdof),
        e_inter: seed_inter,
        role: "seed",
        pose: params
            .keep_poses
            .then(|| pdbqt::write_pose(&cluster.seed.source_text, &seed_lig.world_coords(&best.conf)).ok())
            .flatten(),
    });

    // 各 seed モードの docked コア world 座標（Phase 2 の重ね合わせ目標。子は各モードへ配置し最良を採用）。
    let seed_cores: Vec<Vec<Vec3>> = seed_res
        .modes
        .iter()
        .map(|m| {
            let world = seed_lig.world_coords(&m.conf);
            cluster.seed.core_atoms.iter().map(|&i| world[i]).collect()
        })
        .collect();

    // 子分子も独立 → 並列。
    let child_entries: Vec<ScoredEntry> = cluster
        .children
        .par_iter()
        .map(|child| score_child(scorer, child, &cluster.id, &seed_cores, params))
        .collect();
    entries.extend(child_entries);

    (outcome, entries)
}

/// 1 子分子を Phase 2 + Phase 3 で配置・最適化してスコア行を作る。
/// `seed_cores` は各 seed モードの docked コア world 座標。各モードへ配置・最適化し、**最良 e_inter** を採用
/// （複数 seed ポーズ保持＝剛体コア固定の拘束劣化を緩和）。
fn score_child(
    scorer: &Scorer,
    child: &MolEntry,
    cluster_id: &str,
    seed_cores: &[Vec<Vec3>],
    params: &ScreenParams,
) -> ScoredEntry {
    let child_lig = LigandModel::new(&child.molecule);

    // 子の MCS コア参照 world 座標（seed.core_atoms と同順対応）。
    let child_core_ref: Vec<Vec3> =
        child.core_atoms.iter().map(|&i| child_lig.ref_coords()[i]).collect();
    let core_set: std::collections::HashSet<usize> = child.core_atoms.iter().copied().collect();

    // 各 seed モードへ核固定配置 → R 基最適化。最良 e_inter のポーズを採る。
    let best = seed_cores
        .iter()
        .map(|seed_core_world| {
            let start = phase2::place_child(&child_lig, &child_core_ref, seed_core_world);
            phase3::optimize_child(&child_lig, scorer, start, &core_set, params.local_iters)
        })
        .min_by(|a, b| a.e_inter.partial_cmp(&b.e_inter).unwrap_or(std::cmp::Ordering::Equal))
        .expect("at least one seed pose");

    ScoredEntry {
        compound_id: child.compound_id.clone(),
        cluster_id: cluster_id.to_string(),
        total_score: apply_torsion_correction(best.e_inter, child.molecule.torsdof),
        e_inter: best.e_inter,
        role: "child",
        pose: params
            .keep_poses
            .then(|| pdbqt::write_pose(&child.source_text, &child_lig.world_coords(&best.conf)).ok())
            .flatten(),
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

/// 上位 `top_n` ランク（スコア昇順）の分子のベストポーズを **マルチモデル PDBQT** で返す。
/// `keep_poses=true` でドッキングした場合のみポーズが入る（未保持エントリはスキップ）。各 MODEL に
/// REMARK で Rank/Compound/Cluster/Score/Role を付す。ポーズは採点時に書き込み済み（ライブラリ非依存）。
pub fn poses_pdbqt(report: &ScreenReport, top_n: usize) -> String {
    let mut ranked: Vec<&ScoredEntry> = report.entries.iter().filter(|e| e.pose.is_some()).collect();
    ranked.sort_by(|a, b| a.total_score.partial_cmp(&b.total_score).unwrap_or(std::cmp::Ordering::Equal));

    let mut out = String::new();
    for (i, e) in ranked.iter().take(top_n).enumerate() {
        out.push_str(&format!(
            "MODEL {}\nREMARK Rank {} Compound {} Cluster {} Score {:.4} Role {}\n",
            i + 1, i + 1, e.compound_id, e.cluster_id, e.total_score, e.role,
        ));
        out.push_str(e.pose.as_ref().unwrap());
        out.push_str("ENDMDL\n");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::SearchParams;
    use isct_core::atom_types::XsType;
    use isct_io::library::{Cluster, MolEntry, Pocket};
    use isct_io::pdbqt;

    const LIG: &str = "\
ROOT
ATOM      1  C   LIG A   1       0.000   0.000   0.000  0.00  0.00    +0.000 C
ATOM      2  C   LIG A   1       1.500   0.000   0.000  0.00  0.00    +0.000 C
ATOM      3  O   LIG A   1       0.000   1.500   0.000  0.00  0.00    -0.300 OA
ENDROOT
TORSDOF 0
";

    fn receptor() -> Vec<ScoredAtom> {
        vec![
            ScoredAtom { pos: Vec3::new(2.0, 0.0, 0.0), xs: XsType::OxygenA },
            ScoredAtom { pos: Vec3::new(0.0, 2.0, 0.0), xs: XsType::CarbonH },
            ScoredAtom { pos: Vec3::new(0.0, 0.0, 2.0), xs: XsType::NitrogenD },
        ]
    }

    fn entry(id: &str) -> MolEntry {
        MolEntry {
            compound_id: id.to_string(),
            molecule: pdbqt::parse_str(LIG).unwrap(),
            core_atoms: vec![],
            source_text: LIG.to_string(),
        }
    }

    fn library(pocket: Option<Pocket>, n: usize) -> Library {
        Library {
            clusters: (0..n)
                .map(|i| Cluster {
                    id: format!("c{i}"),
                    pocket: pocket.clone(),
                    seed: entry(&format!("cpd{i}")),
                    children: vec![],
                })
                .collect(),
        }
    }

    fn params() -> ScreenParams {
        let search = SearchParams { n_runs: 2, n_steps: 20, ..SearchParams::default() };
        ScreenParams {
            center: Vec3::zeros(),
            size: Vec3::new(12.0, 12.0, 12.0),
            search,
            prune_threshold: 1e9, // 棄却しない（seed を必ずランキング）
            local_iters: 20,
            use_grid: true,
            seed_poses: 1,
            keep_poses: false,
        }
    }

    #[test]
    fn shared_grid_matches_per_cluster() {
        let r = receptor();
        let p = params();
        // pocket=None → 共有スコアラ経由。pocket=Some(既定ボックス) → per-cluster ローカル構築。
        let shared = screen(&r, &library(None, 4), &p);
        let pocket = Some(Pocket { center: [0.0, 0.0, 0.0], size: [12.0, 12.0, 12.0] });
        let local = screen(&r, &library(pocket, 4), &p);
        assert_eq!(shared.entries.len(), local.entries.len());
        assert!(!shared.entries.is_empty());
        // 共有 grid と per-cluster grid は同一マップ → スコアはビット一致。
        for (a, b) in shared.entries.iter().zip(&local.entries) {
            assert_eq!(a.compound_id, b.compound_id);
            assert_eq!(a.total_score.to_bits(), b.total_score.to_bits(), "score差: {a:?} vs {b:?}");
            assert_eq!(a.e_inter.to_bits(), b.e_inter.to_bits());
        }
    }

    #[test]
    fn keep_poses_outputs_multimodel_pdbqt() {
        let r = receptor();
        let mut p = params();
        p.keep_poses = true;
        let lib = library(None, 2);
        let report = screen(&r, &lib, &p);
        // keep_poses 有効時は各エントリにポーズ座標が入る。
        assert!(report.entries.iter().all(|e| e.pose.is_some()));
        // マルチモデル PDBQT が生成され、REMARK 付き MODEL を含む。
        let poses = poses_pdbqt(&report, 100);
        assert!(poses.contains("MODEL 1\n"));
        assert!(poses.contains("REMARK Rank 1"));
        assert!(poses.contains("ENDMDL"));
    }
}
