//! isct-core — 純粋計算レイヤー。
//!
//! 原子型付与・Vina 互換スコアリング・幾何ユーティリティなど、I/O やドッキング探索に
//! 依存しない計算カーネルを提供する。M4 以降の SIMD/grid 最適化もこのクレートに入る。

pub mod atom_types;
pub mod delta_energy;
pub mod geometry;
pub mod scoring;

/// クレート共通のエラー型。
#[derive(Debug, thiserror::Error)]
pub enum CoreError {
    #[error("unknown atom type: {0}")]
    UnknownAtomType(String),
}
