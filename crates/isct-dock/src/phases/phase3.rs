//! Phase 3 — R 基の最適化と Delta エネルギー評価（M3 で実装、M6 で拡張）。
//!
//! R 基の torsion のみ L-BFGS で最適化し、共通コアのエネルギーはキャッシュして
//! 差分(Delta)のみ追加計算する。M6 で Soft-core ポテンシャルと柔軟側鎖同時最適化を追加。

// TODO(M3): R 基 torsion 最適化と Delta スコア評価を実装する。
