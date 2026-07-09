# TPE 探索モード（`--search tpe`）アルゴリズム仕様

ISCT-Dock の TPE モードは、**TPEvina-2.0**（PSOVina 派生, CBBio）の TPE 探索を移植したもの。
探索空間の各次元を独立な Gaussian Parzen 混合でモデル化し、良観測（lower）近傍から候補をサンプリングして
配置を提案、ISCT の BFGS で局所最適化する **EDA（分布推定）系**の探索。実装は
`crates/isct-dock/src/tpe.rs`（モデル）と `crates/isct-dock/src/search.rs::single_run_tpe`（1 run 駆動）。

本書は括弧内に **TPEvina 側の変数名**を添える（例: `warmup(start_search_num)`）。値は末尾表に isct / TPEvina 双方を掲載。

---

## 1. 全体構成

```
dock（CLI）
 └─ n_runs 本の独立 run を並列実行（seed 別・決定論）      … n_runs(num_tasks=exhaustiveness)
     └─ single_run_tpe（1 run）
         1. contact_start で参照配座 reference を決める
         2. n_steps 回の提案ループ:
             ├─ 前半 warmup 回: 一様ランダム提案
             └─ 後半: Parzen モデルから提案（周期 freq で再構築）
            各提案を BFGS(minimize_free) で局所最適化 → pool へクラスタ挿入・obs へ記録
         3. pool を inter 昇順で返す
 └─ 全 run の pool を統合 → 深い polish → RMSD クラスタで n_modes 出力
     → 最良モードに torsion 補正(num_tors)を掛けて affinity を算出
```

- **決定論**: 探索全体を seed 付き `ChaCha8Rng` で駆動（同一 seed でビット一致）。TPEvina 原実装は
  `random_device` シードで run 毎に非再現（作者も FIXME 明記）だが、ISCT は他モードと同じく決定論化している。
- **1 run の総提案数** = `n_steps(steps)`。**独立 run 数** = `n_runs(num_tasks)`。総探索量はこの積で決まる。

---

## 2. モデル空間

各配座を長さ `6 + n_free_torsions` のベクトルで表す:

```
[ pos(3, 絶対座標) , rot(3, 参照配座からの scaled-axis 差) , free_torsion(n, 参照からの差) ]
```

- 範囲 `bounds`: 位置 3 次元 = 探索ボックスの角 `center ± size/2`、回転・torsion = `[-π, π]`。
- 位置だけがボックス（＝Box サイズ）に依存し、回転・torsion は Box に依らず一定範囲。

---

## 3. 1 run の流れ（`single_run_tpe`）

1. **参照配座** `contact_start`: 受容体と好ましく接触する初期配置を複数試行から選ぶ（Vina の randomize 流）。
   TPE のモデル空間は「この参照からの差」で定義。
2. **ウォームアップ** `warmup(start_search_num)` = `n_steps / startup_frac`（下限 1）。
   最初の `warmup` 提案は **一様ランダム**（`warmup_sample`）で観測 `obs` を蓄積し、探索地形の初期標本を作る。
3. **モデル提案**（`p >= warmup`）: 観測から `TpeModel` を構築し、そこからサンプル。
   - モデル再構築周期 `freq` = `(n_steps − warmup) / density_reuse(kernel_density_frequency)`（下限 1）。
     すなわち後半で **density_reuse 回**だけモデルを作り直し、その間は同じモデルを使い回す。
   - 観測不足時は一様サンプルへフォールバック。
4. **局所最適化**: 各提案を `minimize_free`（ISCT の BFGS）で `local_iters` 回最適化。
   `local_iters` は `TpeParams.local_iters`（既定 0）→ 0 のとき `chain_iters`(=35) を使う。
5. **記録**:
   - `pool` へ `cluster_insert`（キー = **inter**〔分子間エネルギー〕, `cluster_rmsd`=2.0Å で重複排除, 上限
     `pool_cap = max(polish_modes, n_modes)`）。
   - ボックス内なら `obs` へ push（キー = **objective**〔探索地形＝curl 込みスコア〕）。
     **pool は inter・obs は objective** と、順位付けと学習元を分離している点に注意。

---

## 4. Parzen モデル（`tpe.rs`）

### 構築 `TpeModel::build(obs, iter)`
1. `obs` をスコア昇順に整列。
2. **lower / greater 分割**: lower サイズ `n_lower = clamp(0.25·√N, 2, lower_cap)`（N=観測数）。
   先頭 `n_lower` 個が lower（良）、残りが greater（悪）。
3. **greater 重み** `gw = clamp((upper_groups/2 − iter) / upper_groups, 0, 1)`。
   `iter` は提案カウンタ（= ループ変数 `p`）。既定 `upper_groups=10` では `gw = (5 − p)/10`。
4. 各次元ごとに lower/greater の Gaussian 混合（各観測を 1 ガウス、σ は隣接間隔から決定、greater は
   `upper_groups(num_upper_groups)` 個へ切詰め＋`gw` で重み付け）を構築。
5. `greater_active = gw > 0`。

### サンプル `TpeModel::sample`
- **高速経路（`greater_active == false`）**: g(x)=0 で候補選択が退化するため、**各次元 1 点だけ** lower 混合から引く。
  `ei_candidates` は使われない。
- **完全経路（`greater_active == true`）**: 各次元で lower 混合から `ei_candidates(sample_num)` 個引き、
  全次元にわたる `Σ log(g(x)/l(x))` が最小のスロットを採用（期待改善の最大化）。

> **重要な現状（と真因）**: 既定では `gw = (upper_groups/2 − p)/upper_groups = (5 − p)/10` が **p ≥ 5 で 0**。
> ここで `p` は **1 run の提案ループの通し番号**（`for p in 0..n_steps`, warmup 込み。§本節末の注記参照）で、
> モデル構築が始まる `p`（= warmup 以降, 通常 10 以上）では常に `gw ≤ 0 → greater_active == false`。よって
> **実効的には常に高速経路**で `ei_candidates` は一度も使われない（EI を 7↔25 に変えても出力ビット一致,
> `docs/findings.202607090823`）。現状 isct の TPE は「良観測近傍の Parzen サンプリング（exploitation）」として動作。
>
> **真因（`docs/findings.202607090942`）**: この減衰式 `(upper/2−i)/upper` は TPEvina の **`*_7s` 実験変種**由来で、
> その変種でも `i` は絶対通し番号ゆえ greater は死んでいる。一方 **出荷版の既定 `bfgs.h` は greater 重みが
> 順位ベース `(N_g−k−1)/(N_g−26)`（反復非依存）で常に活性**＝本来の TPEvina は `sample_num=25` の EI を実使用する。
> isct の EI/greater が死んでいるのは「カウンタが warmup 込み」ではなく「順位ベースの既定ではなく減衰変種を移植した」
> ため。**EI を生かすには順位ベース greater 重みの採用が正攻法**（検証中）。
>
> 注: `p` は warmup の一様提案も含む run 内通し番号で、最初のモデル構築時点（p ≥ warmup）で既に p > 5。

---

## 5. パラメータ一覧（意味・isct変数名(TPEvina変数名)・双方の値）

| isct 変数名 (TPEvina 変数名) | 意味 | isct 現在値（既定） | TPEvina 値 |
|---|---|---|---|
| `n_runs` (`num_tasks` = `exhaustiveness`) | 独立 run 数。総探索量 = n_runs×n_steps | **60**（tpe 既定, drift 流用）※検証中の推奨 8 | `exhaustiveness`（既定 **1**） |
| `n_steps` (`steps` = `num_steps/15`) | 1 run の提案数（探索の深さ） | **auto** = `6·min(N, 8+5·T)`（下限 200）※検証中の推奨 `7·(50+heuristic)` | `num_steps/15` = `7·(50 + 可動原子 + 10·(T+6))`（≈1000–1900） |
| `startup_frac` → warmup(`start_search_num`) | 一様探索比率。warmup = n_steps / startup_frac | **20**（warmup = n_steps/20）※検証で最適 10–15 | warmup = `steps/20` |
| `ei_candidates` (`sample_num`) | 1 提案あたり lower から引く候補数（EI 最大化用） | **7**（※現状 **不使用**, §4 参照） | **25**（有効な `bfgs.h`）／**7**（isct が参照した `*_7s` 変種） |
| `lower_cap` (`num_lower_groups`) | lower（良）集合サイズ上限。実サイズ = clamp(0.25√N, 2, これ) | **10** | 実サイズ = `max(10, min(0.25√N, 25))`（有効 `bfgs.h`）／isct コメントは 10 と記載 |
| `upper_groups` (`num_upper_groups`) | greater（悪）観測の上限数、かつ greater 重み減衰の分母 | **10** | **10**（isct 移植参照値。有効 `bfgs.h` は greater 全体を使用し分母に定数 26） |
| `density_reuse` (`kernel_density_frequency`) | 後半でモデルを作り直す回数（間は使い回し） | **4** | **4** |
| `local_iters` (—) | 1 提案あたりの BFGS 反復数。0 で `chain_iters` を使用 | **0** → `chain_iters` = **35** | full BFGS（`quasi_newton`, `ssd_par.evals`=`(25+可動原子)/3`） |
| `early_stop_patience` (`termination`) | 改善なし提案が続いたら打切る | **0**（tpe では無効＝全 n_steps 実行） | `termination = steps/25` |

補助（TPE 専用でないが影響する既定）: `chain_iters`=35, `cluster_rmsd`=2.0Å, `n_modes`=9, `polish_modes`=9,
`polish_iters`=20, `hunt_cap`（curl）。config `[run]` で `tpe_ei_candidates` / `tpe_startup_frac` を明示すると
`ei_candidates` / `startup_frac` を上書き可（opt-in, 既定不変）。

### 現在の既定値まとめ（双方）
- **isct（既定）**: `n_runs`=60, `n_steps`=auto(`6·min(N,8+5T)`,≥200), `startup_frac`=20, `ei_candidates`=7,
  `lower_cap`=10, `upper_groups`=10, `density_reuse`=4, `local_iters`=0(→chain_iters=35), `early_stop`=無効。
- **TPEvina（有効 `bfgs.h` / 既定 CLI）**: `num_tasks`=`exhaustiveness`(=1), `steps`=`num_steps/15`,
  warmup=`steps/20`, `sample_num`=25, lower=`max(10,min(0.25√N,25))`, greater=全体, reuse=4,
  local=full BFGS, `termination`=`steps/25`。

---

## 6. 既知の論点（チューニングの現状, 詳細は docs/）

- **n_runs=60 は Drift の既定流用で不適切**。TPEvina は少数・深い run（num_tasks=exhaustiveness × steps=num_steps/15）。
  検証で n_runs=8＋深い n_steps が best-of-3 を改善（`docs/results.202607090812`）。ただし run 数減で独立試行が減る副作用あり。
- **warmup が高レバレッジ**。`startup_frac`=10–15（warmup ≈ n_steps/10–15）が最適で、行き過ぎ(≤7)は非単調に悪化
  （`docs/results.202607090841`）。
- **EI は現状死んでいる**（§4）。有効化には `upper_groups` 拡大とセットが必要（`docs/findings.202607090823`）。
- **score と RMSD の乖離**: 深い探索ほど低エネルギー decoy を掴むため score は下がるが RMSD が改善しない標的がある
  （採点律速, `docs/findings.202607082223`）。評価の主指標は RMSD 成功率。
