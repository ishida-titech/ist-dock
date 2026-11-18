//! ist-dock — ドッキング探索エンジン。
//!
//! リガンドのトポロジー表現、Monte Carlo + L-BFGS 探索（M2）、
//! IST-Dock の Phase 1〜3（M3）、最適化（M4）、精度向上機能（M6）を提供する。

pub mod energy;
pub mod ligand;
pub mod optimizer;
pub mod phases;
pub mod qvina;
pub mod screen;
pub mod search;
pub mod tpe;
pub mod torsion;

/// クレート共通のエラー型。
#[derive(Debug, thiserror::Error)]
pub enum DockError {
    #[error(transparent)]
    Core(#[from] ist_core::CoreError),
    #[error(transparent)]
    Io(#[from] ist_io::IoError),
}
