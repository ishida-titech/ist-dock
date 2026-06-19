//! isct-dock CLI。
//!
//! サブコマンド:
//!   score  — 固定ポーズの Vina 互換スコア評価（M1。`vina --score_only` との一致ゲート）
//!   dock   — 単一リガンドのドッキング（M2）
//!   screen — クラスタライブラリのスクリーニング（M3〜）

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use isct_core::geometry::{self, Vec3};
use isct_core::scoring::{self, intermolecular_terms};
use isct_dock::energy::Scorer;
use isct_dock::ligand::LigandModel;
use isct_dock::screen::{ranking_tsv, screen, ScreenParams};
use isct_dock::search::{auto_n_steps, dock, SearchParams};
use isct_io::{config, library, pdbqt};

#[derive(Parser)]
#[command(name = "isct-dock", version, about = "ISCT-Dock screening engine")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// 固定ポーズの Vina 互換スコアを評価する（M1）。
    Score {
        /// 受容体 PDBQT。
        receptor: PathBuf,
        /// リガンド PDBQT。
        ligand: PathBuf,
    },
    /// 単一リガンドをドッキングする（M2）。
    Dock {
        /// 受容体 PDBQT。
        receptor: PathBuf,
        /// リガンド PDBQT（結晶ポーズ＝RMSD 基準としても使う）。
        ligand: PathBuf,
        /// TOML 設定（探索ボックス・シード等）。
        #[arg(short, long)]
        config: PathBuf,
        /// 出力ポーズ PDBQT（既定: <ligand>_isct_out.pdbqt）。
        #[arg(short, long)]
        out: Option<PathBuf>,
        /// 診断用: 大域探索をせず結晶（入力）ポーズから局所最適化のみ行う。
        #[arg(long)]
        local_only: bool,
    },
    /// クラスタライブラリをスクリーニングする（M3）。
    Screen {
        /// 受容体 PDBQT。
        receptor: PathBuf,
        /// クラスタライブラリのマニフェスト（TOML）。
        library: PathBuf,
        /// TOML 設定（探索ボックス・シード・棄却閾値等）。
        #[arg(short, long)]
        config: PathBuf,
        /// 出力ランキング TSV（既定: <library と同じディレクトリ>/ranking.tsv）。
        #[arg(short, long)]
        out: Option<PathBuf>,
    },
}

fn main() -> Result<()> {
    env_logger::init();
    let cli = Cli::parse();
    match cli.command {
        Command::Score { receptor, ligand } => cmd_score(&receptor, &ligand),
        Command::Dock {
            receptor,
            ligand,
            config,
            out,
            local_only,
        } => cmd_dock(&receptor, &ligand, &config, out.as_deref(), local_only),
        Command::Screen {
            receptor,
            library,
            config,
            out,
        } => cmd_screen(&receptor, &library, &config, out.as_deref()),
    }
}

/// クラスタライブラリをスクリーニングする（M3）。
fn cmd_screen(
    receptor: &PathBuf,
    library_path: &PathBuf,
    config_path: &PathBuf,
    out: Option<&std::path::Path>,
) -> Result<()> {
    let rec = pdbqt::parse_file(receptor)
        .with_context(|| format!("failed to parse receptor: {}", receptor.display()))?;
    let lib = library::load(library_path)
        .with_context(|| format!("failed to load library: {}", library_path.display()))?;
    let cfg = config::load(config_path)
        .with_context(|| format!("failed to load config: {}", config_path.display()))?;

    let center = Vec3::new(cfg.r#box.center[0], cfg.r#box.center[1], cfg.r#box.center[2]);
    let size = Vec3::new(cfg.r#box.size[0], cfg.r#box.size[1], cfg.r#box.size[2]);
    let rec_scored = rec.scored_atoms();

    let use_grid = cfg.run.scoring == config::Scoring::Grid;
    let params = ScreenParams {
        center,
        size,
        search: SearchParams {
            center,
            size,
            seed: cfg.run.seed,
            n_runs: cfg.run.n_runs,
            n_steps: cfg.run.n_steps,
            temp_init: cfg.run.temp_init,
            temp_final: cfg.run.temp_final,
            chain_iters: cfg.run.chain_iters,
            polish_iters: cfg.run.polish_iters,
            cluster_rmsd: cfg.run.cluster_rmsd,
            n_modes: cfg.run.n_modes,
            polish_modes: cfg.run.polish_modes,
            early_stop_patience: cfg.run.early_stop_patience,
            uniform_start: use_grid,
        },
        prune_threshold: cfg.screen.prune_threshold,
        local_iters: cfg.screen.local_iters,
        use_grid,
    };

    eprintln!(
        "screen: receptor(scored)={} atoms, clusters={}, children={}, seed={}, runs={}, steps={}, prune>{}",
        rec_scored.len(),
        lib.clusters.len(),
        lib.n_children(),
        params.search.seed,
        params.search.n_runs,
        params.search.n_steps,
        params.prune_threshold,
    );

    let report = screen(&rec_scored, &lib, &params);

    let out_path = out.map(PathBuf::from).unwrap_or_else(|| {
        library_path.with_file_name("ranking.tsv")
    });
    std::fs::write(&out_path, ranking_tsv(&report))?;

    println!("# isct-dock screen (ISCT-Dock Phase 1-3, M3)");
    println!("clusters total       : {}", report.n_clusters());
    println!("clusters pruned       : {}", report.n_pruned_clusters());
    println!("children skipped(prune): {}", report.n_skipped_children());
    println!("molecules scored      : {}", report.entries.len());
    println!("ranking TSV           : {}", out_path.display());
    Ok(())
}

/// 単一リガンドをボックスへドッキングする（M2）。
fn cmd_dock(
    receptor: &PathBuf,
    ligand: &PathBuf,
    config_path: &PathBuf,
    out: Option<&std::path::Path>,
    local_only: bool,
) -> Result<()> {
    let rec = pdbqt::parse_file(receptor)
        .with_context(|| format!("failed to parse receptor: {}", receptor.display()))?;
    let lig_text = std::fs::read_to_string(ligand)
        .with_context(|| format!("failed to read ligand: {}", ligand.display()))?;
    let lig_mol = pdbqt::parse_str(&lig_text)?;
    let cfg = config::load(config_path)
        .with_context(|| format!("failed to load config: {}", config_path.display()))?;

    let center = Vec3::new(cfg.r#box.center[0], cfg.r#box.center[1], cfg.r#box.center[2]);
    let size = Vec3::new(cfg.r#box.size[0], cfg.r#box.size[1], cfg.r#box.size[2]);

    let rec_scored = rec.scored_atoms();
    let lig = LigandModel::new(&lig_mol);

    // スコアリング・バックエンド（direct=SIMD 直接 / grid=Vina グリッド）。
    let use_grid = cfg.run.scoring == config::Scoring::Grid;
    let scorer = if use_grid {
        let lig_types: Vec<_> = lig_mol.scored_atoms().iter().map(|a| a.xs).collect();
        Scorer::new_grid(&rec_scored, center, size, &lig_types)
    } else {
        Scorer::new(&rec_scored, center, size)
    };

    let params = SearchParams {
        center,
        size,
        seed: cfg.run.seed,
        n_runs: cfg.run.n_runs,
        n_steps: cfg.run.n_steps,
        temp_init: cfg.run.temp_init,
        temp_final: cfg.run.temp_final,
        chain_iters: cfg.run.chain_iters,
        polish_iters: cfg.run.polish_iters,
        cluster_rmsd: cfg.run.cluster_rmsd,
        n_modes: cfg.run.n_modes,
        polish_modes: cfg.run.polish_modes,
        early_stop_patience: cfg.run.early_stop_patience,
        uniform_start: use_grid,
    };

    let eff_steps = if params.n_steps == 0 {
        auto_n_steps(lig.n_scored(), lig.n_torsions())
    } else {
        params.n_steps
    };
    eprintln!(
        "docking: receptor(pruned)={} atoms, ligand={} heavy atoms, torsions={}, runs={}, steps={} ({}), scoring={:?}",
        scorer.receptor_atoms(),
        lig.n_scored(),
        lig.n_torsions(),
        params.n_runs,
        eff_steps,
        if params.n_steps == 0 { "auto" } else { "fixed" },
        cfg.run.scoring,
    );

    let modes = if local_only {
        // 診断: 結晶（恒等）ポーズから局所最適化のみ。
        let objective = |c: &isct_dock::ligand::Conformation| {
            scorer.intermolecular(&lig.scored_atoms(c))
        };
        let start = isct_dock::ligand::Conformation::identity(lig.n_torsions());
        let free = isct_dock::optimizer::all_dofs(start.dim());
        let r = isct_dock::optimizer::minimize_free_fd(start, &objective, 200, &free);
        vec![isct_dock::search::DockResult {
            conf: r.conf,
            energy: r.energy,
        }]
    } else {
        dock(&lig, &scorer, &params)
    };

    let best = &modes[0];
    // 内訳・RMSD は最良モードで。
    let final_atoms = lig.scored_atoms(&best.conf);
    let e_inter = scorer.intermolecular(&final_atoms);
    let e_intra = lig.intramolecular(&final_atoms);
    let reference = lig.reference_scored_positions();
    let rmsd = geometry::rmsd(&lig.scored_positions(&best.conf), &reference);

    // マルチモデル PDBQT 出力（Vina 流）: 各モードを MODEL/ENDMDL で包む。
    let mut out_text = String::new();
    for (i, m) in modes.iter().enumerate() {
        let pose = pdbqt::write_pose(&lig_text, &lig.world_coords(&m.conf))?;
        out_text.push_str(&format!("MODEL {}\n", i + 1));
        out_text.push_str(&pose);
        out_text.push_str("ENDMDL\n");
    }
    let out_path = out.map(PathBuf::from).unwrap_or_else(|| {
        let stem = ligand.file_stem().and_then(|s| s.to_str()).unwrap_or("ligand");
        ligand.with_file_name(format!("{stem}_isct_out.pdbqt"))
    });
    std::fs::write(&out_path, out_text)?;

    println!("# isct-dock dock (Vina 互換, RMSD クラスタリング)");
    println!("intermolecular energy      : {e_inter:.4} kcal/mol");
    println!("intramolecular energy      : {e_intra:.4} kcal/mol");
    println!("total (objective)          : {:.4} kcal/mol", best.energy);
    println!("RMSD vs input(crystal) pose: {rmsd:.3} Å");
    println!("modes (RMSD-clustered)     : {}", modes.len());
    for (i, m) in modes.iter().enumerate() {
        let inter = scorer.intermolecular(&lig.scored_atoms(&m.conf));
        let rm = geometry::rmsd(&lig.scored_positions(&m.conf), &reference);
        println!("  mode {:2}  inter={:8.4}  rmsd_vs_crystal={:6.3} Å", i + 1, inter, rm);
    }
    println!("output poses               : {}", out_path.display());
    Ok(())
}

/// 固定ポーズの Vina 互換スコアを評価して出力する（M1）。
fn cmd_score(receptor: &PathBuf, ligand: &PathBuf) -> Result<()> {
    let rec = pdbqt::parse_file(receptor)
        .with_context(|| format!("failed to parse receptor: {}", receptor.display()))?;
    let lig = pdbqt::parse_file(ligand)
        .with_context(|| format!("failed to parse ligand: {}", ligand.display()))?;

    let rec_atoms = rec.scored_atoms();
    let lig_atoms = lig.scored_atoms();

    let terms = intermolecular_terms(&lig_atoms, &rec_atoms);
    let e_inter = terms.weighted_sum();
    let n_rot = lig.torsdof;
    let affinity = scoring::apply_torsion_correction(e_inter, n_rot);

    println!("# isct-dock score (Vina 互換)");
    println!("receptor atoms (scored): {}", rec_atoms.len());
    println!("ligand atoms   (scored): {}", lig_atoms.len());
    println!();
    println!("term         raw_sum        weighted");
    println!(
        "gauss1     {:>12.5}  {:>12.5}",
        terms.gauss1,
        scoring::W_GAUSS1 * terms.gauss1
    );
    println!(
        "gauss2     {:>12.5}  {:>12.5}",
        terms.gauss2,
        scoring::W_GAUSS2 * terms.gauss2
    );
    println!(
        "repulsion  {:>12.5}  {:>12.5}",
        terms.repulsion,
        scoring::W_REPULSION * terms.repulsion
    );
    println!(
        "hydrophob  {:>12.5}  {:>12.5}",
        terms.hydrophobic,
        scoring::W_HYDROPHOBIC * terms.hydrophobic
    );
    println!(
        "hbond      {:>12.5}  {:>12.5}",
        terms.hbond,
        scoring::W_HBOND * terms.hbond
    );
    println!();
    println!("Intermolecular energy (Ligand - Receptor): {e_inter:.5} kcal/mol");
    println!("Ligand torsional DOF (TORSDOF)           : {n_rot}");
    println!("Estimated Free Energy of Binding         : {affinity:.5} kcal/mol");
    Ok(())
}
