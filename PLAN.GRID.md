# PLAN.GRID — 受容体ポテンシャルのグリッド化（Vina 型・切替式）

`PLAN.OPTIMIZE_DOCKING.md` でスコープ外（別計画）とした「グリッド補間」の実装計画。
実装はこのドキュメントをチェックリストとして進める。関連: [PLAN.md](PLAN.md) /
[PLAN.OPTIMIZE_DOCKING.md](PLAN.OPTIMIZE_DOCKING.md)。

---

## Context（なぜ / 何を）

Vina 比較（docs/results.202606181201.md）で isct-dock のサンプル効率不足が判明。**本質原因は
グリッド不在**：直接原子間スコアはカットオフ 8Å 外で勾配ゼロ（平坦）のため、(1) 遠い初期配置から
最適化が効かず接触棄却で代償、(2) 評価が `O(n_lig×n_rec)` と重い。**受容体ポテンシャルをグリッド化**
すると、(a) 評価が `O(n_lig×8)` 定数化（支配ループ 10〜50×）、(b) ボックス全域に滑らかな勾配場が
でき**一様ランダム初期化で Vina 並に収束**＝速度と探索品質を同時に得る。

### 確定した前提
- スコアは当面 **Vina 型（分解可能・距離/型ベース）** でよい。
- 将来 **水分子の影響**を入れたいので、**「グリッド利用スコア」と「直接スコア」を切替可能**に実装する。
- グリッドマップは **Vina 流のオンデマンド生成**（AutoDock4 型の事前 .map ではなく Vina 型の 5 項を
  受容体について合算したマップ）。

---

## 設計

### 1. スコアリング・バックエンドの抽象化（切替式）
- 探索の評価点は `Scorer::value_and_force`（`crates/isct-dock/src/energy.rs`）に集約済み：
  **分子間（受容体–リガンド）値＋力** ＋ ボックス罰則。ここの**分子間部分のみ**をバックエンド化する。
- `Scorer` に `backend: ScoreBackend` を持たせ enum で分岐:
  - `Direct(ReceptorSimd)` — 現状の f32 SIMD 直接計算（将来の非分解スコア・水分子もここ系統）。
  - `Grid(GridRef)` — Vina グリッドの trilinear 補間。
  `value_and_force` は backend を呼ぶだけ（ボックス罰則・連鎖則 `lig.dof_gradient` は不変）。
- **最終スコア／M1 一致ゲートは常に直接 f64**（`Scorer::intermolecular` = `intermolecular_terms`）を維持。
  → グリッドは**探索専用近似**、報告・ランキングは厳密（既存の f32探索/f64報告と同じ二段構え）。
  screen のランキングも `scorer.intermolecular`（厳密）なので、グリッド由来の偽陽性は混入しない。

### 2. Vina グリッド（`crates/isct-core/src/grid.rs` 新規）
- `GridMap { origin: Vec3, spacing: f32, dims: [usize;3], data: Vec<f32> }`。
  受容体ボックス＋カットオフ余白を覆う格子。1 つの **リガンド XS 型 t** につき 1 枚。
  `data[ix,iy,iz] = Σ_{受容体原子 r, |p−r|<CUTOFF} pair_energy(t, type_r, |p−r|)`（重み付き＝既存
  `scoring::pair_energy(...).0` を再利用）。**1 原子型のプローブが各点で受ける分子間エネルギー**。
- `VinaGrid`: 受容体ボックスに対する型→マップの集合。**オンデマンド生成＋キャッシュ**:
  - 型は XsType の有限集合（~17）。`maps: [OnceCell<Arc<GridMap>>; N_XS]`（または `RwLock<HashMap>`）。
  - `ensure(types, receptor)`: 未生成の型マップだけ構築（スレッド安全）。**並列探索の前に**
    （dock=リガンド型、screen=クラスタのリガンド型）まとめて呼び、探索ループ中はロックなし読み取り。
  - 構築は格子点×近傍受容体（cell list で近傍のみ）。受容体あたり 1 回で**スクリーニングでは完全に償却**。
    まず逐次実装、必要なら rayon 並列。
- 補間: `interpolate_value_grad(map, pos) -> (f32, Vec3)` = trilinear 値とその**解析的勾配**
  （多重線形補間式の微分）。境界外はクランプ（ボックス罰則が引き戻す）。

### 3. グリッド・バックエンドの評価（`energy.rs`）
- `GridRef::value_and_force(ligand, force)`: 各リガンド原子 i について
  `(e_i, g_i) = interpolate_value_grad(map_for(xs_i), pos_i)`、`E += e_i`、`force[i] -= g_i`。
  型→マップは XsType を添字にした密配列で O(1)。**既存 `dof_gradient` 連鎖則がそのまま force を受ける**。

### 4. 切替（config / CLI）
- `crates/isct-io/src/config.rs` の `[run]` に `scoring = "grid" | "direct"`（既定は当面 `direct`、
  グリッド検証後に既定化を検討）。
- `crates/isct-cli/src/main.rs`・`crates/isct-dock/src/screen.rs`: 設定に応じ `Scorer` を構築、
  並列探索前に `VinaGrid::ensure(リガンド型)`。

### 5. 初期配置（グリッド前提で簡素化）
- grid バックエンド時は **一様ランダム初期化（接触棄却を省略）**を既定にできる（Vina 流）。
  direct バックエンドは現状の接触棄却を維持（切替で選択）。

---

## 主要ファイル
- `crates/isct-core/src/grid.rs`（新規）— `GridMap`/`VinaGrid`、構築（`scoring::pair_energy` 再利用）、trilinear 値＋勾配。
- `crates/isct-dock/src/energy.rs` — `Scorer` の `ScoreBackend`（Direct/Grid）分岐、`value_and_force` 委譲。
- `crates/isct-io/src/config.rs` — `scoring` 切替ノブ。
- `crates/isct-cli/src/main.rs`, `crates/isct-dock/src/screen.rs` — backend 構築・`ensure` 呼び出し。
- `crates/isct-dock/src/search.rs` — grid 時の初期配置（一様）切替。
- ベンチ: grid vs direct のマイクロベンチ、`bench/compare_vina_grid.py` 再利用。

## 再利用する既存
- `scoring::{pair_energy, eval_pair, Terms, W_*}`（マップ構築の項）／`intermolecular_terms`（厳密報告）。
- `energy::Scorer::value_and_force`（バックエンド差し込み点）、`box_penalty`。
- `ligand::dof_gradient`（force→DOF 勾配の連鎖則、グリッドでも不変）。
- `bench/run_redock.py::{heavy_coords, rmsd}`, `bench/compare_vina_grid.py`。

## 検証（段階的）
1. **正しさ（単体テスト）**: grid 値 vs 直接 f64 値が刻み依存許容内（例 0.25Å で <~0.3 kcal/mol）、
   grid 勾配 vs 補間値の数値微分 <1e-4、刻みを細かくして誤差が単調減少。
2. **精度回帰**: 3ptb/1stp/1iep 再ドッキング RMSD<2Å（grid）。報告スコアは厳密 f64 で不変。
3. **サンプル効率の再評価**: `compare_vina_grid.py` を grid で再実行し Vina との差の縮小を確認。
4. **速度**: grid vs direct eval マイクロベンチ、再ドッキング時間、grid 構築時間（償却確認）。
5. **決定論**・`cargo test` 緑・`cargo clippy --all-targets` ゼロ。
6. 結果は `docs/results.YYYYMMDDHHMM.md`、知見は `docs/findings.YYYYMMDDHHMM.md` に記録。

## スコープ外（将来）
- 水分子・非分解/ML スコア → `Direct`（または将来バックエンド）側。本計画の抽象化で差し替え可能に。
- 多解像度（Coarse/Fine）グリッド、柔軟側鎖のオフグリッド処理（M6）、動的バウンディングボックスの
  「グリッド添字範囲」化（M5）、Top-K 厳密再採点強化。

## 検証（E2E 手順）
```sh
cargo test && cargo clippy --all-targets
./target/release/isct-dock dock <rec> <lig> --config <cfg(scoring="grid")>
python bench/compare_vina_grid.py
cargo bench
```
