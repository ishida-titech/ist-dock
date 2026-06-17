//! クラスタライブラリのカスタムバイナリ I/O（M3 で実装）。
//!
//! 既存ツール（RDKit/Meeko + tools/preprocess の薄い変換スクリプト）が生成する
//! クラスタライブラリを読む。スキーマ（仕様 4.1）:
//!   Cluster_ID / Target_Pocket_Constraint / Seed_Molecule(全自由度) /
//!   Children_Molecules(共通コアからの差分: 置換基トポロジー・初期座標・ねじれ自由度)。
//! M4 では memmap2 によるストリーミング読込に拡張する。

// TODO(M3): バイナリスキーマ定義とリーダを実装する。
