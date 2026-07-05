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
use isct_dock::search::{auto_n_steps_for, dock, SearchKind, SearchParams};
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
        /// QuickVina2（search="qvina2"）の勾配方向一致閾値（cos）を上書き。高い=保守的(速度↓精度↑)、
        /// 低い=攻める(速度↑精度↓)。省略時は config の qvina_cos（既定 0.90）。
        #[arg(long)]
        qvina_cos: Option<f64>,
        /// 探索アルゴリズムを上書き（standard=ドリフト鎖 / vina=Vina 忠実 MC / qvina2）。省略時は config の search。
        #[arg(long, value_parser = parse_search_mode)]
        search: Option<config::SearchMode>,
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
        /// 上位ポーズをマルチモデル PDBQT で書き出す先（省略時はポーズ出力なし）。
        #[arg(long)]
        poses_out: Option<PathBuf>,
        /// `--poses-out` で書き出す上位ランク数（既定 100）。
        #[arg(long, default_value_t = 100)]
        top: usize,
        /// ワーカスレッドを物理コアにピン留め（Linux。他 OS は no-op）。
        #[arg(long)]
        pin_threads: bool,
        /// 探索アルゴリズムを上書き（standard=ドリフト鎖 / vina=Vina 忠実 MC / qvina2）。省略時は config の search。
        #[arg(long, value_parser = parse_search_mode)]
        search: Option<config::SearchMode>,
    },
    /// TOML マニフェスト＋PDBQT のクラスタライブラリを memmap 用バイナリへ変換する（M4）。
    Pack {
        /// クラスタライブラリのマニフェスト（TOML）。
        manifest: PathBuf,
        /// 出力バイナリ（.bin）。
        out: PathBuf,
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
            qvina_cos,
            search,
        } => cmd_dock(&receptor, &ligand, &config, out.as_deref(), local_only, qvina_cos, search),
        Command::Screen {
            receptor,
            library,
            config,
            out,
            poses_out,
            top,
            pin_threads,
            search,
        } => cmd_screen(&receptor, &library, &config, out.as_deref(), poses_out.as_deref(), top, pin_threads, search),
        Command::Pack { manifest, out } => cmd_pack(&manifest, &out),
    }
}

/// `--search` の値をパースする（standard|drift|vina|qvina2|tpe）。`drift` は `standard` の別名。
fn parse_search_mode(s: &str) -> Result<config::SearchMode, String> {
    match s.to_ascii_lowercase().as_str() {
        "standard" | "drift" => Ok(config::SearchMode::Standard),
        "qvina2" => Ok(config::SearchMode::Qvina2),
        "vina" => Ok(config::SearchMode::Vina),
        "tpe" => Ok(config::SearchMode::Tpe),
        other => Err(format!("unknown search mode '{other}' (standard|drift|vina|qvina2|tpe)")),
    }
}

/// config の探索モード → 探索エンジンの `SearchKind`。
fn search_kind(m: config::SearchMode) -> SearchKind {
    match m {
        config::SearchMode::Standard => SearchKind::Drift,
        config::SearchMode::Qvina2 => SearchKind::Qvina2,
        config::SearchMode::Vina => SearchKind::Vina,
        config::SearchMode::Tpe => SearchKind::Tpe,
    }
}

/// `[run]` config から `SearchParams` を構築する（dock・screen 共通）。`qvina_cos_override` は
/// CLI `--qvina-cos`（dock のみ。screen は `None`）。`uniform_start` は grid バックエンドで true。
fn build_search_params(
    run: &config::RunOpts,
    center: Vec3,
    size: Vec3,
    mode: config::SearchMode,
    qvina_cos_override: Option<f64>,
    uniform_start: bool,
) -> SearchParams {
    SearchParams {
        center,
        size,
        seed: run.seed,
        n_runs: run.n_runs,
        n_steps: run.n_steps,
        temp_init: run.temp_init,
        temp_final: run.temp_final,
        chain_iters: run.chain_iters,
        polish_iters: run.polish_iters,
        cluster_rmsd: run.cluster_rmsd,
        n_modes: run.n_modes,
        polish_modes: run.polish_modes,
        early_stop_patience: run.early_stop_patience,
        uniform_start,
        kind: search_kind(mode),
        qvina_cos: qvina_cos_override.unwrap_or(run.qvina_cos),
        hunt_cap: run.hunt_cap,
        // TPE ハイパーパラメータは既定（TPEvina 準拠）。config 露出は将来対応。
        tpe: Default::default(),
    }
}

/// `--pin-threads` 時、rayon グローバルプールのワーカを物理コアへピン留めする（Linux）。
/// `RAYON_NUM_THREADS` を尊重。既に構築済み等で失敗しても続行（best-effort）。
fn setup_thread_pinning(pin: bool) {
    if !pin {
        return;
    }
    let core_ids = core_affinity::get_core_ids().unwrap_or_default();
    if core_ids.is_empty() {
        return;
    }
    let _ = rayon::ThreadPoolBuilder::new()
        .start_handler(move |i| {
            let id = core_ids[i % core_ids.len()];
            core_affinity::set_for_current(id);
        })
        .build_global();
}

/// TOML マニフェストを memmap 用バイナリへ変換する（M4）。
fn cmd_pack(manifest: &PathBuf, out: &PathBuf) -> Result<()> {
    let lib = library::load(manifest)
        .with_context(|| format!("failed to load library: {}", manifest.display()))?;
    isct_io::library_bin::pack(&lib, out)
        .with_context(|| format!("failed to pack: {}", out.display()))?;
    println!("packed {} clusters -> {}", lib.clusters.len(), out.display());
    Ok(())
}

/// クラスタライブラリをスクリーニングする（M3/M4）。ライブラリは TOML マニフェスト（小・デバッグ）か
/// `pack` 済みバイナリ（拡張子 .bin。大規模・memmap streaming）。
#[allow(clippy::too_many_arguments)]
fn cmd_screen(
    receptor: &PathBuf,
    library_path: &PathBuf,
    config_path: &PathBuf,
    out: Option<&std::path::Path>,
    poses_out: Option<&std::path::Path>,
    top: usize,
    pin_threads: bool,
    search: Option<config::SearchMode>,
) -> Result<()> {
    setup_thread_pinning(pin_threads);
    let rec = pdbqt::parse_file(receptor)
        .with_context(|| format!("failed to parse receptor: {}", receptor.display()))?;
    let cfg = config::load(config_path)
        .with_context(|| format!("failed to load config: {}", config_path.display()))?;

    let center = Vec3::new(cfg.r#box.center[0], cfg.r#box.center[1], cfg.r#box.center[2]);
    let size = Vec3::new(cfg.r#box.size[0], cfg.r#box.size[1], cfg.r#box.size[2]);
    let rec_scored = rec.scored_atoms();

    let mode = search.unwrap_or(cfg.run.search);
    let use_grid = cfg.run.scoring == config::Scoring::Grid;
    let params = ScreenParams {
        center,
        size,
        search: build_search_params(&cfg.run, center, size, mode, None, use_grid),
        prune_threshold: cfg.screen.prune_threshold,
        local_iters: cfg.screen.local_iters,
        use_grid,
        seed_poses: cfg.screen.seed_poses,
        keep_poses: poses_out.is_some(), // ポーズ出力要求時のみ座標を保持
    };

    let is_binary = library_path.extension().is_some_and(|e| e == "bin");
    eprintln!(
        "screen: receptor(scored)={} atoms, library={} ({}), seed={}, runs={}, steps={}, mode={:?}, prune>{}, pin={}",
        rec_scored.len(),
        library_path.display(),
        if is_binary { "binary/memmap" } else { "toml" },
        params.search.seed,
        params.search.n_runs,
        params.search.n_steps,
        params.search.kind,
        params.prune_threshold,
        pin_threads,
    );

    // TOML（in-RAM）か バイナリ（memmap streaming）かでソースを選び、汎用 screen を呼ぶ。
    let report = if is_binary {
        let src = isct_io::library_bin::BinaryLibrary::open(library_path)
            .with_context(|| format!("failed to open binary library: {}", library_path.display()))?;
        screen(&rec_scored, &src, &params)
    } else {
        let lib = library::load(library_path)
            .with_context(|| format!("failed to load library: {}", library_path.display()))?;
        screen(&rec_scored, &lib, &params)
    };

    let out_path = out.map(PathBuf::from).unwrap_or_else(|| {
        library_path.with_file_name("ranking.tsv")
    });
    std::fs::write(&out_path, ranking_tsv(&report))?;

    // 上位ポーズのマルチモデル PDBQT 出力（--poses-out 指定時。ポーズは採点時に書き込み済み＝lib 非依存）。
    if let Some(poses_path) = poses_out {
        std::fs::write(poses_path, isct_dock::screen::poses_pdbqt(&report, top))?;
    }

    println!("# isct-dock screen (ISCT-Dock Phase 1-3, M3)");
    println!("clusters total       : {}", report.n_clusters());
    println!("clusters pruned       : {}", report.n_pruned_clusters());
    println!("children skipped(prune): {}", report.n_skipped_children());
    println!("molecules scored      : {}", report.entries.len());
    println!("ranking TSV           : {}", out_path.display());
    if let Some(p) = poses_out {
        println!("poses PDBQT (top {top}) : {}", p.display());
    }
    Ok(())
}

/// 単一リガンドをボックスへドッキングする（M2）。
fn cmd_dock(
    receptor: &PathBuf,
    ligand: &PathBuf,
    config_path: &PathBuf,
    out: Option<&std::path::Path>,
    local_only: bool,
    qvina_cos: Option<f64>,
    search: Option<config::SearchMode>,
) -> Result<()> {
    let rec = pdbqt::parse_file(receptor)
        .with_context(|| format!("failed to parse receptor: {}", receptor.display()))?;
    let lig_text = std::fs::read_to_string(ligand)
        .with_context(|| format!("failed to read ligand: {}", ligand.display()))?;
    let lig_mol = pdbqt::parse_str(&lig_text)?;
    let cfg = config::load(config_path)
        .with_context(|| format!("failed to load config: {}", config_path.display()))?;
    let mode = search.unwrap_or(cfg.run.search);

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

    let params = build_search_params(&cfg.run, center, size, mode, qvina_cos, use_grid);

    let eff_steps = if params.n_steps == 0 {
        auto_n_steps_for(params.kind, lig.n_scored(), lig.n_torsions())
    } else {
        params.n_steps
    };
    eprintln!(
        "docking: receptor(pruned)={} atoms, ligand={} heavy atoms, torsions={}, runs={}, steps={} ({}), mode={:?}, scoring={:?}",
        scorer.receptor_atoms(),
        lig.n_scored(),
        lig.n_torsions(),
        params.n_runs,
        eff_steps,
        if params.n_steps == 0 { "auto" } else { "fixed" },
        params.kind,
        cfg.run.scoring,
    );

    let (modes, evals) = if local_only {
        // 診断: 結晶（恒等）ポーズから局所最適化のみ。
        let objective = |c: &isct_dock::ligand::Conformation| {
            scorer.intermolecular(&lig.scored_atoms(c))
        };
        let start = isct_dock::ligand::Conformation::identity(lig.n_torsions());
        let free = isct_dock::optimizer::all_dofs(start.dim());
        let r = isct_dock::optimizer::minimize_free_fd(start, &objective, 200, &free);
        (vec![isct_dock::search::DockResult { conf: r.conf, energy: r.energy }], 0u64)
    } else {
        dock(&lig, &scorer, &params)
    };

    let best = &modes[0];
    // 内訳・RMSD は最良モードで。
    let final_atoms = lig.scored_atoms(&best.conf);
    let e_inter = scorer.intermolecular(&final_atoms);
    let e_intra = lig.intramolecular(&final_atoms, f64::INFINITY);
    // Vina 互換の最終スコア（affinity）= 分子間項に torsion 補正を掛けたもの。
    // 順位付け・スクリーニングはこの値を使う（生 intermolecular はサイズ依存で識別力が落ちる）。
    let affinity = scoring::apply_torsion_correction(e_inter, lig_mol.torsdof);
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
    println!("affinity (Vina, kcal/mol)  : {affinity:.4}");
    println!("intermolecular energy      : {e_inter:.4} kcal/mol");
    println!("intramolecular energy      : {e_intra:.4} kcal/mol");
    println!("total (objective)          : {:.4} kcal/mol", best.energy);
    println!("RMSD vs input(crystal) pose: {rmsd:.3} Å");
    println!("search mode                : {:?}", params.kind);
    println!(
        "energy evals               : {} ({} /run)",
        evals,
        evals / params.n_runs.max(1) as u64
    );
    println!("modes (RMSD-clustered)     : {}", modes.len());
    for (i, m) in modes.iter().enumerate() {
        let inter = scorer.intermolecular(&lig.scored_atoms(&m.conf));
        let aff = scoring::apply_torsion_correction(inter, lig_mol.torsdof);
        let rm = geometry::rmsd(&lig.scored_positions(&m.conf), &reference);
        println!(
            "  mode {:2}  affinity={:8.4}  inter={:8.4}  rmsd_vs_crystal={:6.3} Å",
            i + 1,
            aff,
            inter,
            rm
        );
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
