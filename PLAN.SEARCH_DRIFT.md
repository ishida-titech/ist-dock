# 計画: Vina 流「ドリフト鎖」探索（浅い per-step ＋ 多 step ＋ 保持 ＋ 最終深 polish）

## Context（なぜ）

多様データセット検証で isct は速いが、中〜高柔軟リガンドで Vina に精度が劣り、とくに
1pmn(7 tor)/1eby(17 tor) で **native 基を一度もサンプルしない**（docs/findings.202606182223）。
原因を機構レベルで切り分けた結果（docs/findings.202606182316 とコード確認）:

- スコア・ボックス・グリッドは一致／妥当（差なし）。
- **初期配置は既に完全多様**（位置一様＋SO(3)一様配向 `random_quaternion` ＋全 torsion ±π 一様。
  `ligand.rs:397-410,414`）。変異も Vina とほぼ同一（`search.rs:327`）。run/step 量で負けていない。
  焼きなまし・クラスタリングも導入済。→ **量・初期多様性・温度・保持では埋まらない**。
- 残る唯一の構造差は **per-step 局所最適化の深さ**: isct は毎ステップ **フル BFGS(120反復)で底打ち**
  → decoy 基へスナップバック → T=1.0 で脱出棄却 → **鎖が decoy 基に貼り付く**。Vina は per-step BFGS が
  **浅く**、受理小移動が**基底間をドリフト蓄積**して別基(native)へ届き、最後にまとめて深く polish する。

先に `chain_iters=40` 単独で浅化したら退行した（3ptb 3.84/1iep 12.34）。理由は **step 数を増やさず反復
だけ削りドリフトの恩恵なしに収束性を失った**から。本計画は **「浅い per-step ＋ 多 step ＋ モード保持 ＋
最終深 polish」を 1 セットとして**導入する（Vina 流の未検証の組合せ）。

### 確定した決定（ユーザー）
- 方向は **Vina 流ドリフト鎖（本命）** を実装・検証する。

## 設計（`crates/isct-dock/src/search.rs` 中心）

現状: `single_run` は毎ステップ `minimize_free(..., chain_iters=120)` で底打ち→`cluster_insert` で保持。
`dock_constrained` は全代表を `cluster_poses`→上位 `n_modes` を exact-f64 で `polish_iters` 精密化。
**保持(`cluster_insert`)は受理/非受理を問わず全極小を拾う**ため、鎖が荒くてもモード保持は分離済（堅牢性の鍵）。

### 1. 浅い per-step ＋ 多 step（ドリフト）
- `chain_iters` 既定を **浅く**（開始値 30 程度。実装フェーズで 20–40 を走査）。per-step は粗極小でよい
  ——最終 polish が補正する。
- `auto_n_steps` を **浅化に見合って増やす**（per-step が ~4–6× 安価になる分、step を ~4–6× に）。
  開始値 `max(300, 30 * n_scored)`（現 `max(60, 7*n)`）。**計算量は概ね中立**（throughput 配慮）。
- 受理判定・温度・`mutate`・`cluster_insert` は不変（鎖の蓄積で別基へドリフト）。

### 2. polish の漏斗を広げる（粗極小→精密で native を救う）
- `dock_constrained`: 大域 `cluster_poses` を **`polish_modes`（既定 20, ≥ n_modes）** まで残し、その**全代表を
  exact-f64 で深く polish** → 再 `cluster_poses` で `n_modes`(=9) に絞り昇順返却。既存 `cluster_poses`＋polish
  ループの cap を差し替えるだけ（新規構造体不要）。粗い near-native 代表が精密化で勝ちモードへ昇格する機会を増やす。

### 3. パラメータ（config `[run]` ＋ SearchParams）
- 追加: `polish_modes`(usize, 既定 20)。
- 既定変更: `chain_iters` 120→30、`auto_n_steps` 係数増（上記）。`polish_iters`(200)/`n_modes`(9)/
  `cluster_rmsd`(2.0)/温度(1.0/1.0) は不変。**全て config で上書き可**（保守的運用へ戻せる）。

## 主要ファイル
- `crates/isct-dock/src/search.rs` — `auto_n_steps` 係数、`chain_iters` 既定、`dock_constrained` の polish cap を
  `polish_modes` に、SearchParams へ `polish_modes`。`single_run`/`mutate`/`cluster_insert` のロジックは不変。
- `crates/isct-io/src/config.rs` — `RunOpts.polish_modes`（default 20）、`chain_iters` 既定 30 と注記更新。
- `crates/isct-cli/src/main.rs`・`phases/phase1.rs` — `polish_modes` を SearchParams へ受け渡し（返り値型は不変）。

## 想定効果と限界（正直に）
- 仮説: 浅い鎖が decoy 基への貼り付きを緩め、ドリフト蓄積で native 基を**サンプル**、保持＋広い polish で
  **拾い上げる**。1pmn/1eby と中柔軟帯の改善を狙う。
- **不確実**: 高 torsion 次元(1eby 17)では効果が出ない可能性がある。本計画は**実験**であり、下記 go/no-go で
  判断する。負の結果でも記録し、易標的が維持される設定を残す。

## 検証（go/no-go）
1. **易標的の非退行（ガード）**: grid・seed42 で 3ptb<2Å, 1stp<2Å, 1iep<2Å, **pdb/1iep 5/5** を維持。
   退行するなら chain_iters 下限を上げる／polish を深く・広くして回復してから先へ（**この維持は必須条件**）。
2. **hard 標的**: 1pmn/1eby を 3 seed。best RMSD と「native 基を 1 度でもサンプルしたか」を記録。
   <2Å を 1/3 でも得れば前進、2–3Å でも改善として記録。改善ゼロなら負の結果として記録。
3. **広い再評価**: `bench/bench_diverse.py` 部分集合（易＋中柔軟＋2 hard）で成功率を現行と比較、**net 退行なきこと**。
4. **計算量**: 同条件で wall-time が概ね中立（浅化×step 増）であることを記録。
5. `cargo test` 緑・`cargo clippy --all-targets` 警告ゼロ。結果は `docs/results.*`、知見は `docs/findings.*`。

## 撤退条件
易標的ガード(1)が回復不能、または計算量が大幅悪化する場合は既定を現行（chain_iters=120）へ戻し、
本方向を「無効」として findings に記録して終了する。
