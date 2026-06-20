# PLAN.QUICKVINA.md — QuickVina2 相当の局所探索スキップ（opt-in 高速化モード）

QuickVina-W は後続フェーズ（末尾に設計概要）。

## Context（なぜ）

ユーザ要望: **オプションで QuickVina2 / QuickVina-W 相当の処理を選べるようにしたい**（主目的＝更なる高速化）。
確定事項（ユーザ回答）: **(1) QVina2 を先に、QVina-W は後で。(2) 主目的は高速化オプション。(3) QVina-W の
スレッド間共有は「忠実なリアルタイム共有＝非決定論」を許容**（QVina-W フェーズのみ。QVina2 は決定論を維持）。

アルゴリズム（論文確認済み, [QuickVina2](https://doi.org/10.1093/bioinformatics/btv082) /
[QuickVina-W](https://pmc.ncbi.nlm.nih.gov/articles/PMC5684369/)）:
- **QVina2**: 各スレッドの **circular history buffer**（過去に評価した点）に対し、変異後の試行点で
  **first-order-consistency check**（近傍の過去点と勾配符号を比較）を行い、「新しいエネルギー井戸でない」試行は
  **高価な BFGS 局所探索をスキップ**する。Vina 比 ~2× 高速・精度同等。**スレッドローカル＝決定論的**。
- **QVina-W**（後続）: QVina2 ＋ 全スレッド共有の **octree（探索済みリガンド中心位置, 近接半径 R=5Å）** で
  並列探索を協調＝blind docking。共有バッファは順序依存＝**非決定論**。

ISCT-Dock の現探索（`search.rs::single_run`）は既に Vina 流 ILS（mutate→`minimize_free`(BFGS)→Metropolis）で、
**毎ステップ無条件に局所探索**している。ここに QVina2 のスキップ判定を挿す。既定（Standard）は不変。

## 設計（QVina2, 本フェーズ）

### 1. モード切替（config / SearchParams）
- `config.rs` に `SearchMode` enum を追加（`scoring` の `Scoring` enum と同じ書式）: `standard`(既定) / `qvina2`。
  `[run] search = "qvina2"` で有効化。`RunOpts.search_mode`（既定 Standard）。
- `SearchParams.search_mode: SearchMode` を追加し、CLI（`main.rs` の dock/screen 構築 2 箇所）で受け渡し。

### 2. スキップ判定（新規 `crates/isct-dock/src/qvina.rs`）
- `HistoryBuffer`: 容量 `Q`（既定 `10·N`, N=`free_dofs.len()`）の circular buffer。各要素 = `(dof, grad)`。
- `dof_vector(conf, free_dofs) -> Vec<f64>`: DOF 埋め込み = [position(3), 配向の scaled-axis(3), 自由 torsion...]。
  近傍探索の距離計量用（勾配は `value_grad` が返す (並進3,回転3,torsion) と同空間）。
- `is_new_well(dof, grad, params) -> bool`: buffer 中の **P 近傍**（既定 `P=2·N`, 距離は DOF 空間 Euclidean,
  半径 `R` 内）に対し QVina2 規準で判定。「ある近傍点と、ある変数で勾配符号が逆（または一方が 0）」＝停留点を
  挟む＝新しい井戸 → **局所探索する**。全近傍と符号一致（同一斜面）→ 既知極小へ収束 → **スキップ**。
  正確な規準・パラメータ(Q,P,R)は QVina2 論文に従い、検証で調整。

### 3. ループ統合（`search.rs::single_run`）
```
mutate(&mut trial);
if qvina2 {
    let (_e, g_trial) = value_grad(&trial);        // 安価: 勾配 1 回（局所探索の代わり）
    let dof = dof_vector(&trial, free_dofs);
    let significant = history.is_new_well(&dof, &g_trial, &params);
    history.push(dof, g_trial);                     // circular
    if !significant { continue; }                   // 局所探索をスキップ（cur 不変＝棄却扱い）
}
let res = minimize_free(trial, &value, &value_grad, chain_iters, free_dofs);  // significant のみ
// 以降（cluster_insert, Metropolis）は不変
```
- Standard モードでは `if qvina2` を通らず**現行と完全同一**（決定論・精度不変）。
- QVina2 モードも **per-run history のみ＝決定論的**（共有状態なし, 固定 seed で再現）。

## 主要ファイル
- `crates/isct-dock/src/qvina.rs`（新規）— `HistoryBuffer` / `dof_vector` / `is_new_well`。
- `crates/isct-dock/src/search.rs` — `SearchParams.search_mode`、`single_run` ループにスキップ統合。`lib.rs` に `mod qvina;`。
- `crates/isct-io/src/config.rs` — `SearchMode` enum ＋ `RunOpts.search_mode`（既定 standard）。
- `crates/isct-cli/src/main.rs` — dock/screen の SearchParams 構築 2 箇所で受け渡し。

## 再利用
- `value_grad` クロージャ（search.rs, 試行点の勾配）、`free_dofs`、`Conformation`、`minimize_free`、
  `Scoring` enum（`SearchMode` の実装テンプレート）、`Conformation::orientation`（scaled-axis 取得）。

## 想定効果と限界（正直に）
- QVina の ~2× は **局所探索が高価な Vina（直接和）前提**。ISCT は既に grid で局所探索が安価なため **利得は
  小さめの可能性**。ただし局所探索 1 回 ≈ 30 eval（各 intramolecular exp 込み）をスキップ＝1 grad eval に置換
  できるので、スキップ率次第で実速度向上は見込める。
- スキップは探索軌道を変える → borderline 標的の reshuffle（精度感受性, `SIMD_OPTIMIZE.md` 参照）。
  **opt-in なので既定は不変**。QVina2 モードは「速度↑・精度はトレードオフ」と明示。

## 検証
1. 単体テスト: `dof_vector`/`is_new_well`/`HistoryBuffer`。Standard モードが現行とビット一致（QVina2 無効時 no-op）。
   QVina2 モードの**決定論**（同一 seed→出力 sha1 一致）。
2. 速度・精度: 多様 74 標的（部分集合可）で Standard vs QVina2 を比較。**QVina2 が有意に高速**であること、精度
   低下を定量・記録。スキップ率も計測。
3. `cargo test` 緑・`cargo clippy --all-targets` 警告ゼロ。結果は `docs/results.*`、知見は `docs/findings.*`。

### E2E
```sh
cargo test && cargo clippy --all-targets
./target/release/isct-dock dock <rec> <lig> --config <cfg search="qvina2" scoring="grid">
python bench/bench_diverse.py --targets 1iep 1q1g 1hwi 1gpk 1eby
```

## 後続フェーズ: QuickVina-W（本計画では未実装・設計のみ）
- `SearchMode::QVinaW` を追加。QVina2 のスキップ＋**全 run 共有の octree**（`Arc<Mutex<Octree>>` 等。
  探索済みリガンド中心, SMAX=8/cell, WMIN=0.1Å, 近接 R=5Å）。試行点が共有の高品質端点に近ければスキップ＝
  並列協調。large/blind ボックス（全表面）＋ step 増（S→4S）。
- **非決定論を許容**（ユーザ確定）。`search="qvinaw"` 選択時のみ＝既定の決定論は不変。出力に
  「non-deterministic」明記。blind docking 用途。
