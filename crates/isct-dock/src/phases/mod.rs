//! ISCT-Dock の階層的パイプライン（M3 で実装）。
//!
//! Phase 1: 代表分子のドッキングとクラスタ一括棄却。
//! Phase 2: 子分子への共通コア座標流用（M5 で動的バウンディングボックス追加）。
//! Phase 3: R 基の torsion 最適化と Delta エネルギー評価（M6 で Soft-core・柔軟側鎖追加）。

pub mod phase1;
pub mod phase2;
pub mod phase3;
