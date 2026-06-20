# PLAN.SCREEN_GRID_REUSE.md — スクリーニングの grid 再利用（ボックス共有スコアラ）

## Context（なぜ）

10M 化合物スクリーニングのスループットに直結する余地（docs/findings.202606191932 / docs/findings.202606200933）。
現状 `screen.rs::process_cluster` は **クラスタごとに `Scorer::new_grid` を再構築**し、固定ポケットでも C/OA/N 等の
共通 XS 型マップ（`VinaGrid::build_map`＝grid の支配的構築コスト, rayon）を毎クラスタ作り直している。
grid マップは (受容体, ボックス, 型) のみに依存するため、**既定ボックスを使う全クラスタで同一**。1 度だけ構築して
読み取り専用の `&Scorer` を共有すれば、**出力ビット一致のまま** grid 構築をクラスタ数分から 1 回へ削減できる
（クラスタ数が多いほど効果大＝10M 向け）。

## 設計

### 1. 共有スコアラを par_iter 前に 1 度構築（`screen.rs::screen`）
- ポケット未指定（`cluster.pocket.is_none()`）のクラスタが 1 つ以上あれば、**既定ボックス**
  （`params.center/size`）のスコアラを 1 度だけ構築:
  - grid: ポケット未指定クラスタ群の seed＋子の **XS 型の union** で `Scorer::new_grid(receptor, center, size, &union)`。
  - direct: `Scorer::new(receptor, center, size)`（ReceptorSimd を 1 度だけ）。
- `shared: Option<Scorer>` として保持し、`process_cluster` に `Option<&Scorer>` で渡す（`&Scorer` は Sync＝
  rayon の `par_iter` で全クラスタが安全に共有・読み取り）。

### 2. クラスタ処理での選択（`screen.rs::process_cluster`）
- ボックス = `cluster.pocket` > 既定（現行通り）。
- スコアラ: **ポケット未指定 かつ shared 有り → shared を使用**（再構築しない）。ポケット指定あり（既定外
  ボックス）→ 従来通りローカル構築（フォールバック・少数の例外）。
  ```rust
  let local;
  let scorer: &Scorer = match shared {
      Some(s) if cluster.pocket.is_none() => s,
      _ => { local = Some(build_local(receptor, center, size, params, cluster)); local.as_ref().unwrap() }
  };
  ```
- 以降（`phase1::dock_seed`・`score_child` の nested par_iter）は `&Scorer` を受けるため変更最小。

### 3. 型 union ヘルパ
- `fn shared_xs_types(lib) -> Vec<XsType>`: ポケット未指定クラスタの seed＋子の `scored_atoms().xs` を
  重複排除（`HashSet`）。≤17 型・典型 ~10。

## 主要ファイル
- `crates/isct-dock/src/screen.rs` — `screen()` に共有スコアラ構築、`process_cluster` を
  `shared: Option<&Scorer>` 受け取りに、型 union ヘルパ、`build_local`（現行のスコアラ構築を抽出）。
- 変更不要・再利用: `energy.rs::Scorer::{new_grid,new}`、`grid.rs::VinaGrid`、`phase1`/`score_child`。

## 正しさ・効果
- **出力ビット一致**: 既定ボックスの grid マップは per-cluster 構築と同値（同一受容体・ボックス・型・build_map）。
  探索が引くマップ値が同一 → ランキング TSV がビット一致（sha1 で確認）。決定論維持。
- **効果**: grid 構築を「クラスタ数 × 型」から「union 型 × 1」へ。クラスタ数が多いほど削減大。単発・少数
  クラスタでは差は小。メモリは union 全型マップ 1 セット（~10–17 型, 20Å 箱で ~2MB/型）。

## 検証
1. **ビット一致**: `bench/data/screening/trypsin` で ranking TSV を実装前後で比較（sha1 一致）。
2. **速度**: クラスタを複製した多クラスタ・ライブラリで実装前後の壁時間を比較。
3. 単体テスト: 小ライブラリで共有経由と per-cluster のスコア行が一致。
4. `cargo test` 緑・`clippy` 警告ゼロ。結果は `docs/results.*`、知見は `docs/findings.*`。

## スコープ外（今回やらない）
- 複数の異なるポケットごとの共有（distinct box 毎に 1 grid）＝二次最適化。今回は既定ボックス共有に集中。
- パック済みバイナリ＋memmap ストリーミング（入力レイアウト課題, M4 別件）。
- grid マップの遅延・スレッド安全構築＝今回は union 一括構築（決定的・単純）。
