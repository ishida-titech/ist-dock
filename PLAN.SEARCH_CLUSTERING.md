# PLAN.SEARCH_CLUSTERING — ポーズ RMSD クラスタリング＋探索の幅向上

探索の「結合モードの広さ」不足（Vina との残る精度差）を埋めるための計画。実装はこの
ドキュメントをチェックリストとして進める。関連: [PLAN.md](PLAN.md) /
[PLAN.GRID.md](PLAN.GRID.md) / docs/findings.202606182316.md。

## Context（なぜ / 何を）

多様データセット検証で、isct は速い（~1.5×）が中柔軟リガンドで Vina に精度が劣り、1pmn/1eby では
**native 基を一度もサンプルできない**。原因の検討で、これは **障壁越えではなく「結合モードの広さ
（distinct mode のサンプリング・保持）」の問題**で、Vina の堅牢性は **(a) 複数モードの保持、
(b) ステップ毎の浅い局所最適化（exploitation 過剰の回避）** に由来する、と結論。焼きなまし
（大域温度＝障壁越え）は効かず既定オフにした。

本実装で **(a) ポーズ RMSD クラスタリング（複数モード保持・出力）** と **(b) ステップ毎の浅い局所
最適化** を入れ、探索の幅を上げて Vina との精度差を縮める。

### 確定した決定
- クラスタの **Top-N モードを dock 出力に反映**（Vina 流のマルチモデル PDBQT）。
- 関連調整は **浅いステップ局所最適化**（ILS 鎖は浅く、代表ポーズを最後に厚く精密化）。

## 現状（`crates/isct-dock/src/search.rs`）
- `dock_constrained`: run_seeds を rayon 並列 → `single_run`（**単一 best のみ**）→ reduce で global min →
  exact polish 1 回 → 単一 `DockResult`。
- `single_run`: contact_start → minimize(120) → ILS（n_steps, 毎ステップ minimize 120 ＋焼きなまし）→ best。
- RMSD 基盤は既存: `isct_core::geometry::rmsd`、`LigandModel::scored_positions(conf)`（main.rs で使用中）。

## 設計

### 1. クラスタリング（greedy, Vina 流）
候補 = `(Conformation, energy)`。**エネルギー昇順**処理、既存代表との
`rmsd(scored_positions(c), scored_positions(rep)) < cluster_rmsd`（既定 2.0Å）なら同クラスタ、無ければ新代表。
代表＝区別される最低エネルギーポーズ群。ヘルパ `cluster_poses(cands, threshold, cap)` を search.rs に追加。

### 2. per-run（`single_run`）
- ILS 鎖中、**各局所最小**（`minimize_free` の結果）を per-run プールへ。上限 cap（例 32）で逐次クラスタ
  （近傍代表より低ければ置換、無ければ追加、超過時は最悪退避）。
- 鎖・初期 minimize は **浅い `chain_iters`（既定 40）**。
- 返り値を **run の代表リスト** `Vec<(Conformation, f64)>`（昇順）に。

### 3. across-run（`dock_constrained`）
- `par_iter().map(single_run).collect()` で `Vec<Vec<(Conf,energy)>>`（run 順保持＝決定的）。
- flatten →（energy＋安定タイブレークで）昇順 → **大域 greedy クラスタ** → 上位 **N=`n_modes`（既定 9）**。
- 上位代表を **厳密 f64 で厚く精密化（`polish_iters` 既定 200）** → RMSD 再 dedupe・再ソート。
- 返り値 **`Vec<DockResult>`（昇順, [0]=best）**。決定論: 全順序ソート＋決定的 greedy。

### 4. 出力（CLI）
- `cmd_dock`: 上位 N モードを **マルチモデル PDBQT**（`pdbqt::write_pose` を MODEL/ENDMDL で包む）。各モードの
  厳密 intermolecular・結晶 RMSD を表示。best=モード1。
- `screen`/`phase1` は **best（モード0）** のみ使用（出力・ランキング不変）。

### 5. パラメータ（config `[run]` ＋ SearchParams）
`cluster_rmsd`(2.0), `n_modes`(9), `chain_iters`(40), `polish_iters`(200)。SearchParams の `local_iters` を
chain/polish に分離（phase3 側 `ScreenParams.local_iters` は別物・不変）。

## 主要ファイル
- `crates/isct-dock/src/search.rs` — `cluster_poses`、`single_run` 代表リスト返却、`dock_constrained` 大域クラスタ
  ＋複数代表 polish ＋ `Vec<DockResult>`、SearchParams 新フィールド、浅い chain_iters。
- `crates/isct-io/src/config.rs` — `cluster_rmsd/n_modes/chain_iters/polish_iters`。
- `crates/isct-cli/src/main.rs` — マルチモデル出力・各モード採点。
- `crates/isct-dock/src/{screen.rs, phases/phase1.rs}` — 新返り値（Vec, best=[0]）へ適応。

## 再利用する既存
- `isct_core::geometry::rmsd`、`LigandModel::scored_positions`、`pdbqt::write_pose`。
- `optimizer::minimize_free`、`Scorer::{value_and_force, exact_value_and_force, intermolecular}`。
- `bench/bench_diverse.py`・`bench/compare_vina_grid.py`。

## 想定効果と限界（正直に）
- クラスタリングは多様モードの保持・出力を与え、native が非大域モードでも拾える。浅い chain は幅を広げ
  native モードの可視化を助ける。1eby や中柔軟帯の改善を狙う。
- **1pmn のように native 完全未サンプル型は改善するか不確実**（要実測）。効果は標的依存。

## 検証
1. **単体テスト**: `cluster_poses` の分離・代表最低エネルギー・決定論。
2. **回帰**: 3ptb/1stp/1iep 再ドッキング best RMSD<2Å 維持、同一 seed で出力一致。**pdb/1iep 5/5 維持**。
3. **2 標的**: 1pmn/1eby の best RMSD・各モードを確認（改善有無を正直に記録）。
4. **広い再評価**: `bench_diverse.py` 部分集合で旧版比 退行なきこと。Vina 比較は任意。
5. `cargo test` 緑・`cargo clippy --all-targets` ゼロ。結果は `docs/results.*`、知見は `docs/findings.*`。

## 検証（E2E）
```sh
cargo test && cargo clippy --all-targets
./target/release/isct-dock dock <rec> <lig> --config <cfg(scoring="grid")>
python bench/bench_diverse.py --targets 1iep 1q1g 1pmn 1eby 1gpk 1ke5
```
