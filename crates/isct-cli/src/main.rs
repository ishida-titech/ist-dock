//! isct-dock CLI。
//!
//! サブコマンド:
//!   score  — 固定ポーズの Vina 互換スコア評価（`vina --score_only` 相当）
//!   dock   — 単一リガンドを探索してベストポーズを出力
//!   screen — クラスタライブラリを一括ドッキングしてランキング出力
//!   pack   — TOML マニフェスト＋PDBQT ライブラリを memmap バイナリに変換

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
    /// Score a fixed ligand pose with the Vina-compatible function (no search; matches `vina --score_only`).
    #[command(about = "Score a fixed pose (no search)  [args: <RECEPTOR.pdbqt> <LIGAND.pdbqt>]")]
    Score {
        /// Receptor structure (rigid), PDBQT format.
        #[arg(value_name = "RECEPTOR.pdbqt")]
        receptor: PathBuf,
        /// Ligand pose to score, PDBQT format (coordinates are used as-is).
        #[arg(value_name = "LIGAND.pdbqt")]
        ligand: PathBuf,
    },
    /// Search for the best binding pose of a single ligand and write it out.
    #[command(about = "Dock one ligand and write the best pose  [args: <CONFIG.toml> <RECEPTOR.pdbqt> <LIGAND.pdbqt>]")]
    Dock {
        /// Run settings, TOML: search box (center/size), seed, search mode, etc.
        #[arg(value_name = "CONFIG.toml")]
        config: PathBuf,
        /// Receptor structure (rigid), PDBQT format.
        #[arg(value_name = "RECEPTOR.pdbqt")]
        receptor: PathBuf,
        /// Input ligand, PDBQT format. Its coordinates are also the reference pose for the reported RMSD.
        #[arg(value_name = "LIGAND.pdbqt")]
        ligand: PathBuf,
        /// Where to write the best pose (PDBQT). Default: <ligand>_isct_out.pdbqt next to the input.
        #[arg(short, long, value_name = "OUT.pdbqt")]
        out: Option<PathBuf>,
        /// Diagnostic: skip the global search and only locally minimize the input pose (measures the local optimizer alone).
        #[arg(long)]
        local_only: bool,
        /// Only for search=qvina2: gradient-direction agreement threshold (cos) that gates local-search skipping.
        /// Higher = more conservative (slower, more accurate); lower = more aggressive. Default: config qvina_cos (0.90).
        #[arg(long)]
        qvina_cos: Option<f64>,
        /// Search algorithm. tpe (default): fast low-budget preset (shared TPE + local chain + skip + rigid lightening);
        /// standard: drift-chain; drift_qvina2: drift + isct local-search skip; qvina2: faithful QuickVina2 (Vina MC + skip, exh=8);
        /// vina: faithful AutoDock Vina MC. Overrides the config search field (default tpe).
        #[arg(long, value_parser = parse_search_mode)]
        search: Option<config::SearchMode>,
        /// RNG seed override for reproducible runs (default: config seed). Vary it to sweep seeds when benchmarking.
        #[arg(long)]
        seed: Option<u64>,
        /// Diagnostic: dump every search step to this TSV path. Columns: run/step/energy/pos/rot/torsions plus
        /// mode-specific phase/greater_active/accepted. Ignored with --local-only.
        #[arg(long)]
        trace: Option<PathBuf>,
    },
    /// Dock every ligand in a cluster library and write a ranking (batch screening).
    #[command(about = "Screen a cluster library and rank it  [args: <CONFIG.toml> <RECEPTOR.pdbqt> <LIBRARY.toml>]")]
    Screen {
        /// Run settings, TOML: search box (center/size), seed, prune threshold, search mode, etc.
        #[arg(value_name = "CONFIG.toml")]
        config: PathBuf,
        /// Receptor structure (rigid), PDBQT format.
        #[arg(value_name = "RECEPTOR.pdbqt")]
        receptor: PathBuf,
        /// Cluster library manifest (TOML) listing the ligand PDBQTs to screen.
        #[arg(value_name = "LIBRARY.toml")]
        library: PathBuf,
        /// Where to write the ranking (TSV). Default: ranking.tsv in the library's directory.
        #[arg(short, long, value_name = "OUT.tsv")]
        out: Option<PathBuf>,
        /// Optionally also write the top poses as a multi-model PDBQT here (omit to skip pose output).
        #[arg(long, value_name = "POSES.pdbqt")]
        poses_out: Option<PathBuf>,
        /// How many top-ranked ligands to include in --poses-out (default 100).
        #[arg(long, default_value_t = 100)]
        top: usize,
        /// Pin worker threads to physical cores for stabler timing (Linux only; no-op elsewhere).
        #[arg(long)]
        pin_threads: bool,
        /// Search algorithm. tpe (default): fast low-budget preset (shared TPE + local chain + skip + rigid lightening);
        /// standard: drift-chain; drift_qvina2: drift + isct local-search skip; qvina2: faithful QuickVina2 (Vina MC + skip, exh=8);
        /// vina: faithful AutoDock Vina MC. Overrides the config search field (default tpe).
        #[arg(long, value_parser = parse_search_mode)]
        search: Option<config::SearchMode>,
    },
    /// Convert a TOML-manifest + PDBQT cluster library into a single memmap binary for fast loading.
    #[command(about = "Pack a cluster library into a memmap binary  [args: <MANIFEST.toml> <OUT.bin>]")]
    Pack {
        /// Cluster library manifest (TOML) to pack.
        #[arg(value_name = "MANIFEST.toml")]
        manifest: PathBuf,
        /// Output binary path.
        #[arg(value_name = "OUT.bin")]
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
            seed,
            trace,
        } => cmd_dock(
            &receptor, &ligand, &config, out.as_deref(), local_only, qvina_cos, search, seed, trace,
        ),
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

/// `--search` の値をパースする（tpe|standard|drift|drift_qvina2|qvina2|vina）。`tpe` が既定。`drift` は
/// `standard` の別名。`qvina2`=本来の QuickVina2 相当、`drift_qvina2`=ドリフト基盤の isct 独自版。
fn parse_search_mode(s: &str) -> Result<config::SearchMode, String> {
    match s.to_ascii_lowercase().as_str() {
        "tpe" => Ok(config::SearchMode::Tpe),
        "standard" | "drift" => Ok(config::SearchMode::Standard),
        "drift_qvina2" | "driftqvina2" => Ok(config::SearchMode::DriftQvina2),
        "qvina2" => Ok(config::SearchMode::Qvina2),
        "vina" => Ok(config::SearchMode::Vina),
        other => Err(format!(
            "unknown search mode '{other}' (tpe|standard|drift|drift_qvina2|qvina2|vina)"
        )),
    }
}

/// config の探索モード → 探索エンジンの `SearchKind`。
fn search_kind(m: config::SearchMode) -> SearchKind {
    match m {
        config::SearchMode::Standard => SearchKind::Drift,
        config::SearchMode::DriftQvina2 => SearchKind::DriftQvina2,
        config::SearchMode::Qvina2 => SearchKind::Qvina2,
        config::SearchMode::Vina => SearchKind::Vina,
        config::SearchMode::Tpe => SearchKind::Tpe,
    }
}

/// `[run]` config から `SearchParams` を構築する（dock・screen 共通）。`qvina_cos_override` は
/// CLI `--qvina-cos`（dock のみ。screen は `None`）。`uniform_start` は grid バックエンドで true。
///
/// **tpe（既定モード）の CLI プリセット**（dock/screen 共通, n_runs=1。config で `tpe_*` を明示すると上書き）。
/// 素の共有 TPE に以下を重畳し、標準 Astex85×10seed で Vina exh=8 に肉薄（68.8% vs 70.8%）を eval 20×・wall ~9×
/// 軽く達成（`docs/results.202607151422`）。各因子は旧の小標本結論を全85で訂正して確定（`docs/results.202607141930`）:
/// - 局所チェーン K=15（restart 代替）＋ **margin を skip 対象別に分離**: skip 有効（剛直 T<`SKIP_TORSDOF`）は 2.0
///   （skip がモデルを集中させ連鎖が過剰発火するため狭める）、skip 無効（柔軟）は 4.0（広く精密化して精度回収）。
///   旧「一律 2.0 で精度非退行」は Astex24 小標本の誤りで全85では柔軟巻き添え −3.5pt 退行。scale=0.75 は walk-off 抑制。
/// - **局所探索スキップ**（QuickVina2 相当, `skip_local`）を TORSDOF ゲート付きで既定 ON: `T<SKIP_TORSDOF(=7)` の
///   剛直のみスキップ（精度コストは高DOF に集中するため柔軟は温存, `docs/findings.202607141756`）。
/// - **剛直の軽量化**: `T<RIGID_TORSDOF(=6)` の剛直で局所チェーンを `RIGID_LOCAL_CHAIN(=8)` に、n_steps を
///   `scale(T)=min(1, 0.7+0.05·T)`（T=6 で満額）に縮小。K=0/ramp0.5 は剛直 native の pose を退行させるため K=8/ramp0.7。
const TPE_LOCAL_CHAIN: usize = 15;
const TPE_MARGIN: f64 = 2.0;
const TPE_NONSKIP_MARGIN: f64 = 4.0;
const TPE_SCALE: f64 = 0.75;
const TPE_RIGID_TORSDOF: usize = 6;
const TPE_RIGID_LOCAL_CHAIN: usize = 8;
const TPE_SKIP_TORSDOF: usize = 7;
const TPE_RIGID_STEP_SCALE: f64 = 0.7;
const TPE_RIGID_STEP_PER_TORS: f64 = 0.05;

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
        // n_runs 未指定時: tpe=1（局所チェーンで restart 代替）、vina/qvina2=8＝exhaustiveness、それ以外=60。
        // config 明示値は resolve_n_runs が優先。
        n_runs: run.resolve_n_runs(mode),
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
        // TPE ハイパーパラメータ。既定は TPEvina 準拠、EI 候補数と warmup 割合は config で上書き可（opt-in）。
        tpe: {
            let mut t = isct_dock::tpe::TpeParams::default();
            // CLI プリセット（tpe, dock/screen 共通）: 素の共有 TPE（ライブラリ既定は全機構 OFF）に局所チェーン・
            // 局所探索スキップ・剛直軽量化を CLI 層で重畳（config `tpe_*` 明示で上書き）。詳細は `build_search_params`
            // の doc（上）を参照。既定 tpe 以外のモードは素通し。
            if mode == config::SearchMode::Tpe {
                t.local_chain = TPE_LOCAL_CHAIN;
                t.local_chain_margin = TPE_MARGIN;
                t.nonskip_local_chain_margin = TPE_NONSKIP_MARGIN;
                t.local_chain_scale = TPE_SCALE;
                t.skip_local = true;
                t.skip_torsdof = TPE_SKIP_TORSDOF;
                t.rigid_torsdof = TPE_RIGID_TORSDOF;
                t.rigid_local_chain = TPE_RIGID_LOCAL_CHAIN;
                t.rigid_step_scale = TPE_RIGID_STEP_SCALE;
                t.rigid_step_per_tors = TPE_RIGID_STEP_PER_TORS;
            }
            if let Some(v) = run.tpe_ei_candidates {
                t.ei_candidates = v;
            }
            if let Some(v) = run.tpe_startup_ratio {
                t.startup_ratio = v;
            }
            if let Some(v) = run.tpe_step_ratio {
                t.step_ratio = v;
            }
            if let Some(v) = run.tpe_lower_cap {
                t.lower_cap = v;
            }
            if let Some(v) = run.tpe_greater_cap {
                t.greater_cap = v;
            }
            if let Some(v) = run.tpe_density_reuse {
                t.density_reuse = v;
            }
            if let Some(v) = run.tpe_shared {
                t.shared = v;
            }
            if let Some(v) = run.tpe_reanchor_period {
                t.reanchor_period = v;
            }
            if let Some(v) = run.tpe_coverage_weight {
                t.coverage_weight = v;
            }
            if let Some(v) = run.tpe_local_chain {
                t.local_chain = v;
            }
            if let Some(v) = run.tpe_local_chain_margin {
                t.local_chain_margin = v;
            }
            if let Some(v) = run.tpe_nonskip_local_chain_margin {
                t.nonskip_local_chain_margin = v;
            }
            if let Some(v) = run.tpe_local_chain_scale {
                t.local_chain_scale = v;
            }
            if let Some(v) = run.tpe_skip_local {
                t.skip_local = v;
            }
            if let Some(v) = run.tpe_skip_cos {
                t.skip_cos = v;
            }
            if let Some(v) = run.tpe_skip_torsdof {
                t.skip_torsdof = v;
            }
            if let Some(v) = run.tpe_rigid_torsdof {
                t.rigid_torsdof = v;
            }
            if let Some(v) = run.tpe_rigid_local_chain {
                t.rigid_local_chain = v;
            }
            if let Some(v) = run.tpe_rigid_step_scale {
                t.rigid_step_scale = v;
            }
            if let Some(v) = run.tpe_rigid_step_per_tors {
                t.rigid_step_per_tors = v;
            }
            if let Some(v) = run.tpe_rigid_step_floor {
                t.rigid_step_floor = v;
            }
            t
        },
        // トレースは dock 側で --tpe-trace 指定時のみ設定（screen では未使用）。
        trace_path: None,
    }
}

/// vina-mode（`--search vina`）の `n_runs` は AutoDock Vina の **exhaustiveness に対応**する量
/// （独立 MC run 本数）。per-run ステップ数は既に Vina のヒューリスティックに一致させてあるため、
/// n_runs を大きくすると Vina の n_runs/8 倍のコストになる。未指定なら既定 8（=exhaustiveness, Vina 既定）
/// だが、**config で明示的に大きな n_runs を指定した**場合は Vina(exh=8) の n_runs/8 倍遅くなるため
/// stderr で警告する（精度は run 本数で頭打ち＝別要因）。stdout・出力・決定論には一切影響しない。
fn warn_vina_exhaustiveness(kind: SearchKind, n_runs: usize) {
    const VINA_DEFAULT_EXHAUSTIVENESS: usize = 8;
    // vina と本来の qvina2（vina 基盤）は n_runs を exhaustiveness として使う。
    if matches!(kind, SearchKind::Vina | SearchKind::Qvina2) && n_runs > 2 * VINA_DEFAULT_EXHAUSTIVENESS {
        eprintln!(
            "warning: vina/qvina2 mode uses n_runs as exhaustiveness (AutoDock Vina default=8). \
             n_runs={n_runs} costs about {:.1}x of exh=8. For a fair comparison n_runs=8 is recommended.",
            n_runs as f64 / VINA_DEFAULT_EXHAUSTIVENESS as f64,
        );
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
    // 寛容ロード: 解釈不能なリガンドはスキップして残りをパックする（screen と同挙動）。
    let (lib, skipped) = library::load_lenient(manifest)
        .with_context(|| format!("failed to load library: {}", manifest.display()))?;
    isct_io::library_bin::pack(&lib, out)
        .with_context(|| format!("failed to pack: {}", out.display()))?;
    if !skipped.is_empty() {
        eprintln!("warning: skipped {} ligand(s) that could not be parsed", skipped.len());
    }
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
        grid_margin: cfg.run.grid_margin,
        seed_poses: cfg.screen.seed_poses,
        keep_poses: poses_out.is_some(), // ポーズ出力要求時のみ座標を保持
    };
    warn_vina_exhaustiveness(params.search.kind, params.search.n_runs);

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
    // TOML は**寛容ロード**: 解釈不能なリガンド（未知原子・不正 PDBQT）は全体を止めずスキップして継続。
    // バイナリは pack 時に解決済みなので skip は発生しない。
    let mut skipped = Vec::new();
    let report = if is_binary {
        let src = isct_io::library_bin::BinaryLibrary::open(library_path)
            .with_context(|| format!("failed to open binary library: {}", library_path.display()))?;
        screen(&rec_scored, &src, &params)
    } else {
        let (lib, skips) = library::load_lenient(library_path)
            .with_context(|| format!("failed to load library: {}", library_path.display()))?;
        skipped = skips;
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

    // 解釈不能でスキップしたリガンドがあれば skipped.tsv に記録（ランキングと同ディレクトリ）。
    if !skipped.is_empty() {
        let skip_path = out_path.with_file_name("skipped.tsv");
        let mut s = String::from("Cluster_ID\tCompound_ID\tReason\n");
        for r in &skipped {
            s.push_str(&format!("{}\t{}\t{}\n", r.cluster_id, r.compound_id, r.reason));
        }
        let _ = std::fs::write(&skip_path, s);
        eprintln!(
            "warning: skipped {} ligand(s) that could not be parsed (details: {})",
            skipped.len(),
            skip_path.display()
        );
    }

    println!("# isct-dock screen (ISCT-Dock Phase 1-3, M3)");
    println!("clusters total       : {}", report.n_clusters());
    println!("clusters pruned       : {}", report.n_pruned_clusters());
    println!("children skipped(prune): {}", report.n_skipped_children());
    println!("ligands skipped(parse): {}", skipped.len());
    println!("molecules scored      : {}", report.entries.len());
    println!("ranking TSV           : {}", out_path.display());
    if let Some(p) = poses_out {
        println!("poses PDBQT (top {top}) : {}", p.display());
    }
    Ok(())
}

/// 単一リガンドをボックスへドッキングする（M2）。
#[allow(clippy::too_many_arguments)]
fn cmd_dock(
    receptor: &PathBuf,
    ligand: &PathBuf,
    config_path: &PathBuf,
    out: Option<&std::path::Path>,
    local_only: bool,
    qvina_cos: Option<f64>,
    search: Option<config::SearchMode>,
    seed: Option<u64>,
    trace: Option<PathBuf>,
) -> Result<()> {
    let rec = pdbqt::parse_file(receptor)
        .with_context(|| format!("failed to parse receptor: {}", receptor.display()))?;
    let lig_text = std::fs::read_to_string(ligand)
        .with_context(|| format!("failed to read ligand: {}", ligand.display()))?;
    let lig_mol = pdbqt::parse_str(&lig_text)?;
    let mut cfg = config::load(config_path)
        .with_context(|| format!("failed to load config: {}", config_path.display()))?;
    if let Some(s) = seed {
        cfg.run.seed = s; // CLI --seed で config の seed を上書き
    }
    let mode = search.unwrap_or(cfg.run.search);

    let center = Vec3::new(cfg.r#box.center[0], cfg.r#box.center[1], cfg.r#box.center[2]);
    let size = Vec3::new(cfg.r#box.size[0], cfg.r#box.size[1], cfg.r#box.size[2]);

    let rec_scored = rec.scored_atoms();
    let lig = LigandModel::new(&lig_mol);

    // スコアリング・バックエンド（direct=SIMD 直接 / grid=Vina グリッド）。
    let use_grid = cfg.run.scoring == config::Scoring::Grid;
    let scorer = if use_grid {
        let lig_types: Vec<_> = lig_mol.scored_atoms().iter().map(|a| a.xs).collect();
        Scorer::new_grid(&rec_scored, center, size, &lig_types, cfg.run.grid_margin)
    } else {
        Scorer::new(&rec_scored, center, size)
    };

    let mut params = build_search_params(&cfg.run, center, size, mode, qvina_cos, use_grid);
    warn_vina_exhaustiveness(params.kind, params.n_runs);
    // 診断トレース（--trace）: 全探索モード対応。--local-only は別経路（大域探索なし）のため無効。
    if let Some(path) = trace {
        if local_only {
            eprintln!("warning: --trace is ignored with --local-only (no global search is performed).");
        } else {
            params.trace_path = Some(path);
        }
    }

    let eff_steps = if params.n_steps == 0 {
        auto_n_steps_for(
            params.kind,
            lig.n_scored(),
            lig.n_torsions(),
            params.tpe.step_ratio,
        )
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
    // num_tors は Vina 1.2 の実効回転結合数（末端基は 0.5, TORSDOF ではない）。
    let num_tors = lig_mol.num_tors_vina();
    let affinity = scoring::apply_torsion_correction(e_inter, num_tors);
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

    println!("# isct-dock dock (Vina-compatible, RMSD clustering)");
    println!("affinity (Vina, kcal/mol)  : {affinity:.4}");
    println!("intermolecular energy      : {e_inter:.4} kcal/mol");
    println!("intramolecular energy      : {e_intra:.4} kcal/mol");
    // モードの順位キーは inter（=affinity 基準）。目的関数値は inter+intra で別途表示（箱内 box=0）。
    println!("total (objective)          : {:.4} kcal/mol", e_inter + e_intra);
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
        let aff = scoring::apply_torsion_correction(inter, num_tors);
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

/// 固定ポーズの Vina 互換スコアを評価して出力する。
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
    let affinity = scoring::apply_torsion_correction(e_inter, lig.num_tors_vina());

    // ファイル名のみ（パスは出さない）。
    let base = |p: &PathBuf| {
        p.file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| p.display().to_string())
    };

    // ヘッダ: バージョンと git 情報（同一バージョンでもソースを一意に識別できるよう埋め込む）。
    println!(
        "# isct-dock score  v{} (git {})",
        env!("CARGO_PKG_VERSION"),
        env!("ISCT_GIT_HASH")
    );
    println!("# Vina-compatible scoring function");
    println!("Receptor : {}  ({} atoms scored)", base(receptor), rec_atoms.len());
    println!("Ligand   : {}  ({} atoms scored)", base(ligand), lig_atoms.len());
    println!();
    // 分子間項テーブル（列は空白区切りでパースしやすく整列）。
    println!("{:<12} {:>12} {:>12}", "TERM", "RAW_SUM", "WEIGHTED");
    let row = |name: &str, raw: f64, weight: f64| {
        println!("{name:<12} {raw:>12.5} {:>12.5}", weight * raw);
    };
    row("Gauss1", terms.gauss1, scoring::W_GAUSS1);
    row("Gauss2", terms.gauss2, scoring::W_GAUSS2);
    row("Repulsion", terms.repulsion, scoring::W_REPULSION);
    row("Hydrophobic", terms.hydrophobic, scoring::W_HYDROPHOBIC);
    row("HBond", terms.hbond, scoring::W_HBOND);
    println!();
    println!("Intermolecular energy (ligand-receptor) : {e_inter:.5} kcal/mol");
    println!("Ligand torsional DOF (TORSDOF)          : {n_rot}");
    println!("Estimated Free Energy of Binding        : {affinity:.5} kcal/mol");
    Ok(())
}
