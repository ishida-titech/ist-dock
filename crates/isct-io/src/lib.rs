//! isct-io — 入出力レイヤー。
//!
//! PDBQT パーサ（受容体・リガンド）、TOML 設定、クラスタライブラリのカスタムバイナリ I/O、
//! 出力（ランキング TSV・ポーズ）を提供する。

pub mod config;
pub mod library;
pub mod library_bin;
pub mod pdbqt;

/// クレート共通のエラー型。
#[derive(Debug, thiserror::Error)]
pub enum IoError {
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error("PDBQT parse error: {0}")]
    Pdbqt(String),
    #[error(transparent)]
    Core(#[from] isct_core::CoreError),
    #[error("config parse error: {0}")]
    Config(#[from] toml::de::Error),
    #[error("library error: {0}")]
    Library(String),
}
