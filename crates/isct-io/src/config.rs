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

/// 実行パラメータ（既定値あり）。
#[derive(Debug, Clone, Deserialize)]
pub struct RunOpts {
    #[serde(default = "default_seed")]
    pub seed: u64,
    #[serde(default = "default_n_runs")]
    pub n_runs: usize,
    #[serde(default = "default_n_steps")]
    pub n_steps: usize,
}

impl Default for RunOpts {
    fn default() -> Self {
        RunOpts {
            seed: default_seed(),
            n_runs: default_n_runs(),
            n_steps: default_n_steps(),
        }
    }
}

fn default_seed() -> u64 {
    42
}
fn default_n_runs() -> usize {
    // best-of-N が複数シードで安定して RMSD<2Å になる水準（benzamidine 検証）。
    60
}
fn default_n_steps() -> usize {
    60
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
