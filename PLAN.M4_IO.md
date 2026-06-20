# PLAN.M4_IO.md — M4 残実装（core_affinity ＋ バイナリ・ライブラリ＋memmap streaming）

大規模テストは別途（本計画では小〜中規模で検証）。確定方針: bincode 採用 / TOML パスと併存 / 大規模は後日。

## Context

M4 のうち grid/SIMD/ゼロアロケーション/rayon は実装済。残りは **(A) core_affinity（物理コアピン留め）** と
**(B) memmap2＋カスタムバイナリ・ライブラリ**。後者は 10M 化合物を全パース＝全 RAM ロードせずスクリーニング
する前提（現状 `library::load` は全 PDBQT をパースし `Vec<Cluster>` を RAM 保持＝非スケール,
docs/findings.202606200933）。

## 設計

### 0. 依存（`Cargo.toml`）
- `core_affinity = "0.8"`・`memmap2 = "0.9"` 有効化、`bincode = "1"` 追加。`nalgebra` に `serde-serialize` feature。

### 1. serde 派生
- `atom_types.rs`: `AdType`・`XsType`。`pdbqt.rs`: `Atom`・`Fragment`・`Molecule`。`library.rs`: `Pocket`・
  `MolEntry`・`Cluster` に `Serialize, Deserialize`。

### 2. バイナリ形式・pack・reader（新規 `isct-io/src/library_bin.rs`）
- `Header { magic, version, n_clusters, has_default_box, xs_union: Vec<XsType>, offsets: Vec<(u64,u64)> }` を
  bincode（長さ前置）→ 各クラスタ = bincode(`Cluster`)。`has_default_box`/`xs_union` は pack 時に前計算。
- `pack(lib, out)` / `BinaryLibrary::open(path)`（memmap2 map・Header デコード・`cluster(i)` 遅延 bincode・
  `#[cfg(unix)] advise(WillNeed)`）。

### 3. streaming screen（`screen.rs`）
- `trait ClusterSource: Sync { n_clusters; cluster(i)->Cow<Cluster>; has_default_box; shared_xs_types }`。
  `impl for Library`（in-RAM）/ `for BinaryLibrary`（遅延デコード・Header から union/flag）。
- `screen<S: ClusterSource>(receptor, src, params)`。共有 Scorer は src から構築。
  `(0..n).into_par_iter().map(|i| process_cluster(&src.cluster(i), ...))`＝遅延デコード→処理→破棄でメモリ有界・
  順序保持で決定論。

### 4. ポーズ出力を lib 非依存に
- `ScoredEntry.pose` を `Option<String>`（`write_pose` 済み body）に変更し採点時に書き込む。
  `poses_pdbqt(report, top)` は lib 不要。

### 5. CLI ＋ core_affinity
- `pack <manifest.toml> <out.bin>`。`screen` は拡張子/マジックで Binary/TOML 判定。
- `--pin-threads`（opt-in）: 起動時に rayon グローバルプールを `start_handler` 付きで構築しワーカ i を
  物理コアにピン留め（Linux。非対応は no-op）。`RAYON_NUM_THREADS` 尊重。

## 検証
1. `pack ampc_mini → lib.bin`。TOML 版とバイナリ版の `screen` で ranking TSV が sha1 一致。`--poses-out` も一致。
2. memmap 遅延デコードで動く（大規模メモリ実測は後日）。
3. `--pin-threads` で正常実行。
4. 単体テスト: library_bin ラウンドトリップ・ClusterSource 両 impl。既存 screen テスト維持。
5. `cargo test` 緑・`clippy` ゼロ。`PROGRESS.md` を M4 完了に更新。

## スコープ外
- 大規模テスト（10M 実測）＝後日。分散・M5・M6。
