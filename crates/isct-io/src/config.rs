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
}

impl Default for ScreenOpts {
    fn default() -> Self {
        ScreenOpts {
            prune_threshold: default_prune_threshold(),
            local_iters: default_child_local_iters(),
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
#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum Scoring {
    #[default]
    Direct,
    Grid,
}

/// 探索モード。`standard`=毎ステップ局所探索（既定・決定論）/ `qvina2`=QuickVina2 相当の局所探索スキップ
/// （opt-in 高速化・決定論維持）。`qvinaw`（QuickVina-W: 協調 blind docking・非決定論）は後続フェーズ。
#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum SearchMode {
    #[default]
    Standard,
    Qvina2,
}

/// 実行パラメータ（既定値あり）。
#[derive(Debug, Clone, Deserialize)]
pub struct RunOpts {
    #[serde(default = "default_seed")]
    pub seed: u64,
    #[serde(default = "default_n_runs")]
    pub n_runs: usize,
    #[serde(default = "default_n_steps")]
    pub n_steps: usize,
    /// 探索の分子間評価方式（既定: direct）。
    #[serde(default)]
    pub scoring: Scoring,
    /// 探索モード（既定: standard。`qvina2` で局所探索スキップ＝opt-in 高速化）。
    #[serde(default)]
    pub search: SearchMode,
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
}

impl Default for RunOpts {
    fn default() -> Self {
        RunOpts {
            seed: default_seed(),
            n_runs: default_n_runs(),
            n_steps: default_n_steps(),
            scoring: Scoring::default(),
            search: SearchMode::default(),
            temp_init: default_temp_init(),
            temp_final: default_temp_final(),
            chain_iters: default_chain_iters(),
            polish_iters: default_polish_iters(),
            cluster_rmsd: default_cluster_rmsd(),
            n_modes: default_n_modes(),
            polish_modes: default_polish_modes(),
            early_stop_patience: default_early_stop_patience(),
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
    30
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
fn default_early_stop_patience() -> usize {
    // 既定 OFF。収束ベースの早期停止はドリフト鎖のハード標的を退行させうるため opt-in に留める。
    0
}

fn default_seed() -> u64 {
    42
}
fn default_n_runs() -> usize {
    // best-of-N が複数シードで安定して RMSD<2Å になる水準（benzamidine 検証）。
    60
}
fn default_n_steps() -> usize {
    // 0 = 可動原子数から自動スケール（Vina 流。search::auto_n_steps）。
    0
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
        assert!(cfg.receptor.is_none());
    }

    #[test]
    fn parses_full() {
        let cfg: RunConfig = toml::from_str(
            "[receptor]\npdbqt = \"r.pdbqt\"\n[box]\ncenter=[0,0,0]\nsize=[18,18,18]\n[run]\nseed=7\nn_runs=5\nn_steps=10\n",
        )
        .unwrap();
        assert_eq!(cfg.run.seed, 7);
        assert_eq!(cfg.run.n_runs, 5);
        assert_eq!(cfg.receptor.unwrap().pdbqt, "r.pdbqt");
    }
}
