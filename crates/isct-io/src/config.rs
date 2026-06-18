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
    /// 焼きなまし温度の初期値（高温で誤基脱出）。
    #[serde(default = "default_temp_init")]
    pub temp_init: f64,
    /// 焼きなまし温度の終値（低温で収束）。
    #[serde(default = "default_temp_final")]
    pub temp_final: f64,
}

impl Default for RunOpts {
    fn default() -> Self {
        RunOpts {
            seed: default_seed(),
            n_runs: default_n_runs(),
            n_steps: default_n_steps(),
            scoring: Scoring::default(),
            temp_init: default_temp_init(),
            temp_final: default_temp_final(),
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
