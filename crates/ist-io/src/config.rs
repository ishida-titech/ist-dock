//! TOML 実行設定（M2〜）。
//!
//! 探索ボックス（中心・サイズ）と実行パラメータ（シード・探索網羅性）を `serde` で受ける。
//! 例:
//! ```toml
//! [box]
//! center = [-1.99, 14.25, 16.24]
//! size = [20.0, 20.0, 20.0]
//!
//! [run]
//! seed = 42
//! n_runs = 20
//! n_steps = 40
//! ```

use std::path::Path;

use serde::Deserialize;

use crate::IoError;

/// 実行設定全体。
#[derive(Debug, Clone, Deserialize)]
pub struct RunConfig {
    /// 受容体指定（任意。CLI 引数で渡す場合は不要）。
    #[serde(default)]
    pub receptor: Option<ReceptorConfig>,
    #[serde(rename = "box")]
    pub r#box: BoxConfig,
    #[serde(default)]
    pub run: RunOpts,
    /// スクリーニング（M3 `screen`）用パラメータ。
    #[serde(default)]
    pub screen: ScreenOpts,
}

/// `screen` 用パラメータ（既定値あり）。
#[derive(Debug, Clone, Deserialize)]
pub struct ScreenOpts {
    /// 一括棄却の閾値: seed のドッキング目的関数値が これを上回るクラスタは子を全スキップ。
    #[serde(default = "default_prune_threshold")]
    pub prune_threshold: f64,
    /// 子分子の R 基 torsion 局所最適化の最大反復。
    #[serde(default = "default_child_local_iters")]
    pub local_iters: usize,
    /// 子配置に使う seed の保持ポーズ数（RMSD クラスタ済み上位 N）。1=最良のみ（従来）、>1 で各子を
    /// 複数 seed ポーズへ配置し最良を採用＝剛体コア固定の拘束劣化を緩和（子コストは N 倍）。
    #[serde(default = "default_seed_poses")]
    pub seed_poses: usize,
}

impl Default for ScreenOpts {
    fn default() -> Self {
        ScreenOpts {
            prune_threshold: default_prune_threshold(),
            local_iters: default_child_local_iters(),
            seed_poses: default_seed_poses(),
        }
    }
}

fn default_prune_threshold() -> f64 {
    // 好ましい結合が無い（目的関数が非負）クラスタを棄却。標的に応じ調整可。
    0.0
}
fn default_child_local_iters() -> usize {
    120
}
fn default_seed_poses() -> usize {
    1
}

#[derive(Debug, Clone, Deserialize)]
pub struct ReceptorConfig {
    pub pdbqt: String,
}

/// 探索ボックス。
#[derive(Debug, Clone, Deserialize)]
pub struct BoxConfig {
    pub center: [f64; 3],
    pub size: [f64; 3],
}

/// 探索の分子間評価バックエンド。`direct`=直接原子間（SIMD）/ `grid`=Vina グリッド補間。
/// 既定は `grid`（受容体を前計算グリッドで補間＝auto ステップ予算でも高速・省メモリ、推奨）。
#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum Scoring {
    Direct,
    #[default]
    Grid,
}

/// 探索モード。`drift`=ドリフト鎖 ILS（旧称 standard・決定論）/ `drift_qvina2`=ドリフト鎖に QuickVina2 の局所探索
/// スキップを載せた ist 独自版（軽い per-run × best-of-60）/ `qvina2`=**本来の QuickVina2 相当**（Vina 忠実 MC＋
/// 局所探索スキップ, exhaustiveness=8）/ `vina`=Vina 忠実 MC（毎ステップ収束まで BFGS。実装レベル差の切り分け用）。
/// `qvinaw`（QuickVina-W: 協調 blind docking・非決定論）は後続フェーズ。
#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum SearchMode {
    /// **既定モード**（TPE, Tree-structured Parzen Estimator）。共有観測プールの逐次 Parzen サンプリング探索に
    /// 局所チェーン・局所探索スキップ・TORSDOF 条件付き軽量化を載せた低予算プリセット（n_runs=1）。dock/screen 共通。
    #[default]
    Tpe,
    /// ドリフト鎖 ILS（best-of-60）。旧既定。旧称 `standard`（後方互換で受理）。
    #[serde(alias = "standard")]
    Drift,
    /// ドリフト鎖 ILS（`drift`）に QuickVina2 の局所探索スキップだけを載せた ist 独自版（軽い per-run ×
    /// best-of-60, n_runs=60）。基盤が Vina でない点で本来の QuickVina2 とは別物（下記 `qvina2` が忠実版）。
    #[serde(rename = "drift_qvina2")]
    DriftQvina2,
    /// **本来の QuickVina2 相当**: Vina 忠実 MC（各ステップ Vina の per-step 局所探索上限・Vina 変異/温度）に
    /// 局所探索スキップを載せた版。n_runs=exhaustiveness=8（Vina と同予算）。
    Qvina2,
    Vina,
}

/// 実行パラメータ（既定値あり）。
#[derive(Debug, Clone, Deserialize)]
pub struct RunOpts {
    #[serde(default = "default_seed")]
    pub seed: u64,
    /// 独立 run 数（探索の網羅性）。**未指定なら探索モード別の既定**（`resolve_n_runs`）:
    /// tpe=1（局所チェーンで restart を代替）、vina/qvina2=8（exhaustiveness 相当）、drift/drift_qvina2=60
    /// （軽い per-run を best-of-N）。明示指定すればモードに依らずその値を使う。
    #[serde(default)]
    pub n_runs: Option<usize>,
    #[serde(default = "default_n_steps")]
    pub n_steps: usize,
    /// 探索の分子間評価方式（既定: grid）。
    #[serde(default)]
    pub scoring: Scoring,
    /// グリッド探索（scoring="grid"）のボックス余白（Å, 既定 2.0）。探索中に原子が箱外へはみ出しても
    /// 正しいグリッド値を引ける緩衝。小さいほど構築が速く省メモリだが、境界近傍のボーダーライン標的が
    /// 退行しうる（実測: 0 で一部標的が退行、2 で全標的中立）。環境変数 `IST_GRID_MARGIN` で更に上書き可。
    #[serde(default = "default_grid_margin")]
    pub grid_margin: f64,
    /// 探索モード（**既定: tpe**）。`drift`=drift 鎖 ILS、`vina`=Vina 忠実 MC、`qvina2`/`drift_qvina2` など。
    #[serde(default)]
    pub search: SearchMode,
    /// QuickVina2 の勾配方向一致閾値（cos, 既定 0.95）。CLI `--qvina-cos` で上書き可。
    #[serde(default = "default_qvina_cos")]
    pub qvina_cos: f64,
    /// 探索中の正エネルギー上限（Vina hunt_cap, 既定 10.0）。激突の暴走を抑える。大きい値（例 1e9）で無効化。
    #[serde(default = "default_hunt_cap")]
    pub hunt_cap: f64,
    /// 焼きなまし温度の初期値（高温で誤基脱出）。
    #[serde(default = "default_temp_init")]
    pub temp_init: f64,
    /// 焼きなまし温度の終値（低温で収束）。
    #[serde(default = "default_temp_final")]
    pub temp_final: f64,
    /// ILS 鎖の各ステップ局所最適化反復（浅く保ち、鎖を基底間でドリフトさせ広く探索）。
    #[serde(default = "default_chain_iters")]
    pub chain_iters: usize,
    /// 代表ポーズの最終精密化反復（厳密 f64 で厚く）。
    #[serde(default = "default_polish_iters")]
    pub polish_iters: usize,
    /// ポーズ RMSD クラスタリング閾値（Å）。
    #[serde(default = "default_cluster_rmsd")]
    pub cluster_rmsd: f64,
    /// 保持・出力する最大モード数。
    #[serde(default = "default_n_modes")]
    pub n_modes: usize,
    /// 最終精密化にかける代表モード数（≥ n_modes）。粗極小の near-native を救う漏斗を広げる。
    #[serde(default = "default_polish_modes")]
    pub polish_modes: usize,
    /// 早期停止 patience（新規モードも改善も無いステップ数の上限）。0 で無効。収束した易標的を短縮。
    #[serde(default = "default_early_stop_patience")]
    pub early_stop_patience: usize,
    /// TPE モード専用: EI 候補数（`TpeParams.ei_candidates`）の上書き。未指定なら既定 25。
    #[serde(default)]
    pub tpe_ei_candidates: Option<usize>,
    /// TPE モード専用: warmup 比率（`TpeParams.startup_ratio`, warmup = n_steps×この値, 0.0–1.0）の上書き。
    /// 未指定なら既定 0.05。
    #[serde(default)]
    pub tpe_startup_ratio: Option<f64>,
    /// TPE モード専用: auto n_steps の係数（`TpeParams.step_ratio`, auto_n_steps_vina×この値, **大きいほど深い**）の上書き。
    /// 未指定なら既定 0.1。n_steps を明示指定した場合は不使用。
    #[serde(default)]
    pub tpe_step_ratio: Option<f64>,
    /// TPE モード専用: lower（良群）サイズ上限（`TpeParams.lower_cap`）の上書き。未指定なら既定 10。
    #[serde(default)]
    pub tpe_lower_cap: Option<usize>,
    /// TPE モード専用: greater（悪群）カーネル上限（`TpeParams.greater_cap`）の上書き。未指定なら既定 10。
    #[serde(default)]
    pub tpe_greater_cap: Option<usize>,
    /// TPE モード専用: モデル再構築回数（`TpeParams.density_reuse`）の上書き。未指定なら既定 6。
    #[serde(default)]
    pub tpe_density_reuse: Option<usize>,
    /// TPE 共有観測モード（`TpeParams.shared`, **既定 true**）。総予算を単一の共有観測プールで逐次探索する
    /// （独立 n_runs の冗長を排し前回結果を再利用・warmup 1 回）。**false で従来の独立 n_runs**（`results.202607100834`
    /// とビット一致）へ戻す。未指定なら既定 true。
    #[serde(default)]
    pub tpe_shared: Option<bool>,
    /// TPE 共有モードのリアンカー周期（提案数, `TpeParams.reanchor_period`）。0=周期リアンカーなし（単一アンカー,
    /// 推奨）。>0 で N 提案ごとに探索的 restart。未指定なら 0。
    #[serde(default)]
    pub tpe_reanchor_period: Option<usize>,
    /// TPE 共有モードの未踏域カバレッジ重み（`TpeParams.coverage_weight`, 0.0=無効）。未指定なら 0.0。
    #[serde(default)]
    pub tpe_coverage_weight: Option<f64>,
    /// TPE 共有モードの局所チェーン歩数（`TpeParams.local_chain`, K）。0=OFF（既定・現行とビット一致）。>0 で
    /// competitive 提案を Drift MC で K 歩精密化＝MC の持続滞在を移植。未指定なら 0。
    #[serde(default)]
    pub tpe_local_chain: Option<usize>,
    /// 局所チェーンのゲート幅（`TpeParams.local_chain_margin`, kcal）。skip 有効リガンドに適用。未指定なら 2.0。
    #[serde(default)]
    pub tpe_local_chain_margin: Option<f64>,
    /// skip 無効リガンド用の局所チェーンゲート幅（`TpeParams.nonskip_local_chain_margin`）。0.0=分離無効（既定・
    /// 全リガンド `tpe_local_chain_margin`＝ビット不変）。>0.0 で `!skip_active` のリガンドのみこの広い margin へ切替＝
    /// 柔軟リガンドの精度回収。未指定なら 0.0。
    #[serde(default)]
    pub tpe_nonskip_local_chain_margin: Option<f64>,
    /// 局所チェーンの変異幅スケール（`TpeParams.local_chain_scale`）。1.0=Drift 幅（既定）、<1.0 で縮小し
    /// basin-hop（歩き去り）を抑制。未指定なら 1.0。
    #[serde(default)]
    pub tpe_local_chain_scale: Option<f64>,
    /// TPE 共有モードの局所探索スキップ（QuickVina2 相当, `TpeParams.skip_local`）。true で非 warmup 提案の
    /// start を勾配1回で判定し、既知 basin なら初期 BFGS を省く。tpe プリセットでは既定 ON。未指定なら
    /// ライブラリ既定 false（＝素の TPE とビット一致）。
    #[serde(default)]
    pub tpe_skip_local: Option<bool>,
    /// 局所探索スキップの勾配方向一致閾値（`TpeParams.skip_cos`, cosine）。高いほど保守的（スキップ少）。
    /// 未指定なら 0.90。
    #[serde(default)]
    pub tpe_skip_cos: Option<f64>,
    /// 局所探索スキップの TORSDOF ゲート閾値（`TpeParams.skip_torsdof`）。`free_torsions.len() < これ` の剛直
    /// リガンドのみスキップし、柔軟（T≥閾値）では無効化する。スキップの精度コストは高DOF に集中するため剛直側の
    /// 安価なスキップだけ残す（`docs/findings.202607141756`）。0=ゲート無し（既定・現行とビット一致）。未指定なら 0。
    #[serde(default)]
    pub tpe_skip_torsdof: Option<usize>,
    /// TORSDOF 条件付き局所チェーン抑制の閾値（`TpeParams.rigid_torsdof`）。`free_torsions.len() < これ` の
    /// 剛直リガンドで局所チェーンを `tpe_rigid_local_chain` に切替える。0=OFF（既定・現行とビット一致）。剛直
    /// デコイの過小スコア化（EF1% 低下）対策の opt-in。未指定なら 0。
    #[serde(default)]
    pub tpe_rigid_torsdof: Option<usize>,
    /// 剛直リガンドで用いる局所チェーン歩数 K（`TpeParams.rigid_local_chain`）。通常 `tpe_local_chain` より小、
    /// 0 で剛直はチェーン無効。`tpe_rigid_torsdof=0` なら不使用。未指定なら 0。
    #[serde(default)]
    pub tpe_rigid_local_chain: Option<usize>,
    /// n_steps 縮小の最小係数（`TpeParams.rigid_step_scale`, 最剛直に適用）。1.0=OFF（既定・ビット一致）。未指定なら 1.0。
    #[serde(default)]
    pub tpe_rigid_step_scale: Option<f64>,
    /// n_steps 縮小の段階ランプ勾配（`TpeParams.rigid_step_per_tors`）。T≥rigid_torsdof で 1.0 へ線形復帰させ低〜中
    /// 自由度も緩やかに削る。0.0 で従来の2段。未指定なら 0.0。
    #[serde(default)]
    pub tpe_rigid_step_per_tors: Option<f64>,
    /// n_steps 縮小時の下限（`TpeParams.rigid_step_floor`, warmup 確保）。未指定なら 200。
    #[serde(default)]
    pub tpe_rigid_step_floor: Option<usize>,
}

impl Default for RunOpts {
    fn default() -> Self {
        RunOpts {
            seed: default_seed(),
            n_runs: None,
            n_steps: default_n_steps(),
            scoring: Scoring::default(),
            grid_margin: default_grid_margin(),
            search: SearchMode::default(),
            qvina_cos: default_qvina_cos(),
            hunt_cap: default_hunt_cap(),
            temp_init: default_temp_init(),
            temp_final: default_temp_final(),
            chain_iters: default_chain_iters(),
            polish_iters: default_polish_iters(),
            cluster_rmsd: default_cluster_rmsd(),
            n_modes: default_n_modes(),
            polish_modes: default_polish_modes(),
            early_stop_patience: default_early_stop_patience(),
            tpe_ei_candidates: None,
            tpe_startup_ratio: None,
            tpe_step_ratio: None,
            tpe_lower_cap: None,
            tpe_greater_cap: None,
            tpe_density_reuse: None,
            tpe_shared: None,
            tpe_reanchor_period: None,
            tpe_coverage_weight: None,
            tpe_local_chain: None,
            tpe_local_chain_margin: None,
            tpe_local_chain_scale: None,
            tpe_nonskip_local_chain_margin: None,
            tpe_skip_local: None,
            tpe_skip_cos: None,
            tpe_skip_torsdof: None,
            tpe_rigid_torsdof: None,
            tpe_rigid_local_chain: None,
            tpe_rigid_step_scale: None,
            tpe_rigid_step_per_tors: None,
            tpe_rigid_step_floor: None,
        }
    }
}

fn default_temp_init() -> f64 {
    1.0
}
fn default_temp_final() -> f64 {
    // 既定は固定温度（init==final）。焼きなましは temp_init>temp_final を明示指定したとき有効。
    1.0
}
fn default_chain_iters() -> usize {
    // 浅い per-step BFGS で鎖を基底間にドリフトさせ広く探索する（Vina 流）。単独の浅化は収束を損ねる
    // ため、多 step（auto_n_steps 増）と最終深 polish（polish_iters/polish_modes）とセットで用いる。
    // 直線探索の warm-start(alpha) と併せて 35 を既定とする（Astex でエネルギー中立・RMSD 中立を保ちつつ
    // 高速化。warm-start 単独＋ci=30 の成功率 −2〜3pt を ci=35 が相殺、docs/results.202607062203）。
    35
}
fn default_polish_iters() -> usize {
    // grid 整合探索後、代表は既にほぼ最適。深い exact polish は精度をほぼ変えず高コストのため浅く。
    20
}
fn default_cluster_rmsd() -> f64 {
    2.0
}
fn default_n_modes() -> usize {
    9
}
fn default_polish_modes() -> usize {
    9
}
fn default_hunt_cap() -> f64 {
    10.0
}
fn default_qvina_cos() -> f64 {
    // 74 標的スイープで 0.90 が 0.95 を支配（1.27× 高速かつ精度も僅かに上）。0.85 は更に速いが精度低下大。
    0.90
}
fn default_early_stop_patience() -> usize {
    // 既定 OFF。収束ベースの早期停止はドリフト鎖のハード標的を退行させうるため opt-in に留める。
    0
}

fn default_seed() -> u64 {
    42
}
/// vina-mode（`SearchMode::Vina`）の未指定時 n_runs 既定。AutoDock Vina の exhaustiveness に対応し、
/// per-run が重いので Vina 既定の 8 に合わせる（`n_runs=60` のままだと Vina の 7.5 倍遅い）。
const DEFAULT_N_RUNS_VINA: usize = 8;
/// tpe-mode（`SearchMode::Tpe`）の既定 n_runs。**統一プリセット（dock/screen 共通）は低予算 n_runs=1**（局所チェーン
/// が restart を代替）。標準 Astex85×10seed で Vina exh=8 に肉薄（68.8% vs 70.8%）を eval 20×・wall ~9× 軽く達成
/// （`docs/results.202607151422`）。より高精度が要るときは config で n_runs を上げる。
const DEFAULT_N_RUNS_TPE: usize = 1;
/// vina/tpe 以外（drift/qvina2）の未指定時 n_runs 既定。軽い per-run を best-of-N で回して精度を出す
/// 水準（benzamidine 検証で複数シードが安定して RMSD<2Å）。
const DEFAULT_N_RUNS_OTHER: usize = 60;

/// 探索モード別の n_runs 既定（config で未指定のとき使う）。
pub fn default_n_runs_for(mode: SearchMode) -> usize {
    match mode {
        // vina と本来の qvina2 は exhaustiveness=8（少数・高コスト run）。
        SearchMode::Vina | SearchMode::Qvina2 => DEFAULT_N_RUNS_VINA,
        SearchMode::Tpe => DEFAULT_N_RUNS_TPE,
        // drift・drift_qvina2 は軽い per-run の best-of-60。
        _ => DEFAULT_N_RUNS_OTHER,
    }
}

impl RunOpts {
    /// 実効 n_runs を解決する。config で明示されていればその値、未指定ならモード別既定
    /// （[`default_n_runs_for`]）。`mode` は CLI `--search` 上書き後の実効モードを渡すこと。
    pub fn resolve_n_runs(&self, mode: SearchMode) -> usize {
        self.n_runs.unwrap_or_else(|| default_n_runs_for(mode))
    }
}
fn default_n_steps() -> usize {
    // 0 = 可動原子数から自動スケール（Vina 流。search::auto_n_steps）。
    0
}
fn default_grid_margin() -> f64 {
    // グリッド探索ボックスの余白（Å）。全標的で RMSD<2Å 中立を保つ安全側（docs/results.202607071919）。
    2.0
}

/// TOML ファイルを読み込む。
pub fn load(path: impl AsRef<Path>) -> Result<RunConfig, IoError> {
    let text = std::fs::read_to_string(path)?;
    Ok(toml::from_str(&text)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_minimal() {
        let cfg: RunConfig = toml::from_str(
            "[box]\ncenter = [1.0, 2.0, 3.0]\nsize = [20.0, 20.0, 20.0]\n",
        )
        .unwrap();
        assert_eq!(cfg.r#box.center, [1.0, 2.0, 3.0]);
        assert_eq!(cfg.run.seed, 42); // 既定値
        assert_eq!(cfg.run.grid_margin, 2.0); // 既定 margin
        assert!(cfg.receptor.is_none());
    }

    #[test]
    fn parses_grid_margin_override() {
        let cfg: RunConfig = toml::from_str(
            "[box]\ncenter=[0,0,0]\nsize=[20,20,20]\n[run]\ngrid_margin = 1.5\n",
        )
        .unwrap();
        assert_eq!(cfg.run.grid_margin, 1.5);
    }

    #[test]
    fn parses_full() {
        let cfg: RunConfig = toml::from_str(
            "[receptor]\npdbqt = \"r.pdbqt\"\n[box]\ncenter=[0,0,0]\nsize=[18,18,18]\n[run]\nseed=7\nn_runs=5\nn_steps=10\n",
        )
        .unwrap();
        assert_eq!(cfg.run.seed, 7);
        assert_eq!(cfg.run.n_runs, Some(5));
        assert_eq!(cfg.run.resolve_n_runs(SearchMode::Drift), 5); // 明示値はモードに依らず優先
        assert_eq!(cfg.receptor.unwrap().pdbqt, "r.pdbqt");
    }

    #[test]
    fn n_runs_defaults_per_mode_when_unset() {
        // n_runs 未指定 → モード別既定（tpe=1〔統一〕, vina=8, それ以外=60）。
        let cfg: RunConfig =
            toml::from_str("[box]\ncenter=[0,0,0]\nsize=[20,20,20]\n").unwrap();
        assert_eq!(cfg.run.n_runs, None);
        assert_eq!(cfg.run.resolve_n_runs(SearchMode::Vina), 8);
        assert_eq!(cfg.run.resolve_n_runs(SearchMode::Drift), 60);
        assert_eq!(cfg.run.resolve_n_runs(SearchMode::Tpe), 1);
        assert_eq!(SearchMode::default(), SearchMode::Tpe); // 既定モードは tpe
    }
}
