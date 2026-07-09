# TPE 探索モード（`--search tpe`）アルゴリズム仕様

実装: `crates/isct-dock/src/tpe.rs`（Parzen モデル）と `crates/isct-dock/src/search.rs::single_run_tpe`（1 run 駆動）。
移植元は **TPEvina-2.0**（PSOVina 派生, CBBio）。本書は括弧内に **TPEvina 側の変数名**を添える（例
`warmup(start_search_num)`）。**drift 等と共有の変数は §6 と各所で明記**する。

---

## 1. TPE とはどんなアルゴリズムか

**TPE（Tree-structured Parzen Estimator）** は、過去の評価結果から探索空間の「良さ」を確率モデルで推定し、
次に評価すべき点を賢く選ぶ **逐次モデルベース最適化（Sequential Model-Based Optimization, SMBO）＝ベイズ最適化
（Bayesian optimization）** の一種。分布そのものを推定して次点を提案するため **分布推定アルゴリズム
（Estimation of Distribution Algorithm, EDA）** にも分類される。ドッキングでは「配座（ligand pose）→ エネルギー」を
目的関数（objective function）とし、低エネルギーの配座を探す。

手順の骨子:

1. **ウォームアップ（warm-up / start-up）**: 最初は探索空間を一様ランダム（uniform random）に評価し、
   **観測（observation）**（＝評価した点とその目的値）を貯める。モデルを作るための初期標本。
2. **良群 / 悪群への分割（good/bad split by quantile）**: 観測を目的値の**分位点（quantile）**で
   **良群（good set, "lower"）** と **悪群（bad set, "greater"）** に分ける。
3. **密度推定（density estimation）**: 各群を **カーネル密度推定（Kernel Density Estimation, KDE）＝Parzen 推定
   （Parzen estimator）** でモデル化。良群の密度を **l(x)**、悪群の密度を **g(x)** と書く（Gaussian 混合
   〔Gaussian mixture〕で表す）。
4. **候補提案と選択（candidate proposal & selection）**: 良密度 l(x) から複数の**候補（candidates）**を引き、
   **期待改善（Expected Improvement, EI）** が最大の候補を選ぶ。TPE では EI ∝ **l(x)/g(x)** に帰着する
   （「良い所に多く・悪い所に少ない」点を選ぶ＝比 g/l を最小化）。
5. **評価と更新（evaluate & update）**: 選んだ候補を目的関数で評価（ここでは **局所最適化（local optimization）
   ＝BFGS** を通す）し、観測へ追加してモデルを更新。以降 1 run 分これを繰り返す。

> 「Tree-structured」は原論文（Bergstra 2011）の**条件付き・階層的な変数空間**に由来する呼称。本実装は
> 各次元を独立にモデル化する簡略版（原 TPEvina も同様）。

ISCT 版は探索全体を seed 付き `ChaCha8Rng` で駆動し**同一 seed でビット一致（決定論, deterministic）**。
TPEvina 原実装は `random_device` シードで run 毎に非再現（作者も FIXME 明記）。

---

## 2. 概念 ↔ コード変数の対応（プロトコル）

§1 の概念が、コード（`single_run_tpe` / `tpe.rs`）でどの変数・関数に対応するか。**◆＝drift 等と共有**。

| §1 の概念（日本語 / English） | コード上の対応 | 補足 |
|---|---|---|
| 探索空間（search space）の 1 点＝配座ベクトル | モデル空間ベクトル `v`（長さ `6 + n_free_torsions`） | §3。`[pos(3), rot(3), torsion(n)]` |
| 探索範囲（bounds）下限/上限 | `l_limit` / `g_limit`（`tpe::bounds(center,size,…)`） | 位置のみ Box 依存。回転・torsion は ±π |
| 参照配座（reference conformation） | `reference`（`contact_start` の返り値）◆ | モデル空間は「参照からの差」で定義 |
| 観測集合（observations） | `obs: Vec<Obs>` | ボックス内の評価済み点のみ蓄積 |
| 観測の目的値（objective value） | `Obs.score` = `res.energy`（curl 込み探索目的） | **順位付け用の `inter` とは別**（下段） |
| ウォームアップ（warm-up） | `warmup`（回数）, `warmup_sample`（一様サンプル） | `warmup = n_steps / startup_frac` |
| 良群（good set, lower） | `lower_idx`, 各次元 `Dim.lower`, サイズ `n_lower` | スコア昇順の先頭 `n_lower` 個 |
| 悪群（bad set, greater） | `greater_idx`, 各次元 `Dim.greater`, サイズ `n_g` | lower を除いた残り |
| 良密度 l(x)（KDE） | `mixture_density(&d.lower, param, 1.0)` → `lx` | Gaussian 混合 |
| 悪密度 g(x)（KDE, 順位重み付き） | `mixture_density_weighted(&d.greater, param)` → `gx` | 各カーネルに順位重み `weight` |
| 期待改善（EI ∝ l/g） | `ratio = gx/lx`; `ei += ratio.ln()`; `ei_per_slot` 最小スロット採用 | Σ_dim log(g/l) 最小＝l/g 最大 |
| 候補数（number of candidates） | `ei_candidates`(=`sample_num`) | §5 |
| 代理モデル（surrogate model） | `TpeModel`（`model`） | 1 run 内で保持 |
| モデル再構築（rebuild cadence） | `freq`, `since_rebuild`, `density_reuse` | 後半で `density_reuse` 回だけ作り直す |
| greater 活性フラグ | `greater_active`（`= n_g >= 27`） | false で高速経路（EI 計算省略, §4） |
| 局所最適化（local optimization, BFGS） | `minimize_free(...)` ◆, `local_iters` | drift と同一の BFGS |
| 代表プール（candidate pool） | `pool`（`cluster_insert` で挿入）◆ | **`inter` で順位付け**（下段） |

### パラメータ以外で特に重要な変数
- **`Obs.score`（目的値・objective）と `inter`（分子間エネルギー・intermolecular energy）の使い分け**:
  TPE モデルの学習元（`obs`）は **objective**（`res.energy`, box penalty=curl 込みの探索地形）。一方、最終出力の
  **順位付け・pool 保持**は **`inter`（`res_inter`, 純分子間エネルギー）**。両者は意図的に分離している
  （探索は curl で滑らかな地形を学び、採点は Vina 互換 affinity の基＝inter で行う）。
- **`greater_active`**: g(x) を使うか否か。`n_g < 27` のとき全 greater 重みが 0 になり、候補選択が退化するので
  各次元 1 点だけ引く**高速経路**へ落ちる（`ei_candidates` 不使用）。`n_g >= 27` で **完全経路**（EI 選択）。
- **`reference`（参照配座）**: `contact_start` が受容体と好ましく接触する初期配置を複数試行から選ぶ ◆。
  モデル空間ベクトルはこの参照からの差分。
- **`pool` と `obs` の別**: `pool` は最終代表（inter 順・RMSD クラスタ済み, 上限 `pool_cap`）、`obs` は
  モデル学習用（objective 順）。役割が異なる。

---

## 3. モデル空間（探索空間ベクトル）

各配座を長さ `6 + n_free_torsions` のベクトルで表す:

```
[ pos(3, 絶対座標) , rot(3, 参照配座からの scaled-axis 差) , free_torsion(n, 参照からの差) ]
```

- 範囲: 位置 3 次元 = 探索ボックス角 `center ± size/2`、回転・torsion = `[-π, π]`。
- **位置だけが Box サイズに依存**し、回転・torsion は Box 非依存。
- 変換関数: `tpe::conf_from_vec`（ベクトル→配座）, `tpe::vec_from_conf`（配座→ベクトル）。

---

## 4. 1 run の流れ（`single_run_tpe`）と Parzen モデル

### 1 run（`single_run_tpe`）
1. `contact_start` で **参照配座** `reference` を得る ◆。
2. `warmup(start_search_num) = n_steps / startup_frac` 回は **一様提案**（`warmup_sample`）で `obs` を蓄積。
3. 以降は **モデル提案**: `TpeModel::build` で構築したモデルから `sample`。再構築周期 `freq =
   (n_steps − warmup) / density_reuse`。観測不足時は一様へフォールバック。
4. 各提案を `minimize_free`（BFGS, `local_iters` 回）で局所最適化 ◆。
5. **記録**: `pool` へ `cluster_insert`（キー=`inter`, `cluster_rmsd`=2.0Å）◆／ボックス内なら `obs` へ push
   （キー=`objective`）。
6. `pool` を `inter` 昇順で返す。全 run 分を `dock_constrained` が統合→深い polish→`n_modes` 出力 ◆。

### Parzen モデル（`tpe.rs`）
- **build**: `obs` をスコア昇順→ lower `n_lower = clamp(0.25√N, 2, lower_cap)` / greater（残り `n_g`）へ分割。
  各次元で lower/greater の Gaussian 混合（σ は隣接ギャップ規則）を構築。
- **greater 重み（順位ベース, rank-based）**: 各 greater 観測 `gi`（スコア順位, 0=最良greater）に
  `weight = clamp(gi/(n_g−26), 0, 1)`。**`n_g ≥ 27` で活性**（`greater_active`）。反復番号には依存しない。
- **sample**: `greater_active==false` → 各次元 1 点（高速経路）。`true` → 各次元 `ei_candidates` 個引き、
  Σ log(g/l) 最小の候補を採用（完全経路＝本来の EI）。

> **現状（commit f9e37e5 以降）**: greater 重みは上記**順位ベース**で、観測が 27 個以上たまると活性化し EI が機能する
> （出荷版 TPEvina `bfgs.h` 準拠）。**それ以前は `*_7s` 実験変種の減衰式 `(upper/2−iter)/upper` を移植しており
> greater が常に不活性＝EI が死んでいた**（`docs/findings.202607090942`）。EI 復活後は `ei_candidates`=25
> （＝TPEvina `sample_num`）が明確に良く（best-of-3 70→90%, `docs/results.202607090951`）、2026-07-09 に既定化済（§5）。

---

## 5. パラメータ一覧（意味・isct変数名(TPEvina変数名)・双方の値・共有）

| isct 変数名 (TPEvina 変数名) | 意味 | isct 現在値（既定） | TPEvina 値 | drift 等と共有 |
|---|---|---|---|---|
| `n_runs` (`num_tasks`=`exhaustiveness`) | 独立 run 数。総探索量=n_runs×n_steps | **8** | `exhaustiveness`（既定 **1**） | TPE 専用既定（`DEFAULT_N_RUNS_TPE`=8）。standard/qvina2=60, vina=8 |
| `n_steps` (`steps`=`num_steps/15`) | 1 run の提案数（探索の深さ） | **auto** = `auto_n_steps_vina/15`（≥200, `auto_n_steps_tpe`。TPEvina の num_steps/15） | `num_steps/15`≈1000–1900 | config 欄は共有だが **auto 式はモード別**（drift=`auto_n_steps`, vina=`auto_n_steps_vina`） |
| `startup_frac` → warmup(`start_search_num`) | 一様探索比率。warmup=n_steps/startup_frac | **20** | warmup=`steps/20` | TPE 専用 |
| `ei_candidates` (`sample_num`) | 1 提案あたり lower から引く候補数（EI 用） | **25** | **25**（既定 `bfgs.h`）/ 7（`*_7s` 変種） | TPE 専用 |
| `lower_cap` (`num_lower_groups`) | lower（良群）サイズ上限。実サイズ=clamp(0.25√N,2,これ) | **10** | `max(10,min(0.25√N,25))`（既定 bfgs.h） | TPE 専用 |
| `greater_cap` (`num_upper_groups`) | greater（悪）カーネルの上限数（切詰め）。`lower_cap` の greater 版 | **10** | 既定 bfgs.h は全 greater を使用 | TPE 専用 |
| `density_reuse` (`kernel_density_frequency`) | 後半でモデルを作り直す回数 | **4** | **4** | TPE 専用 |
| `local_iters` (—) | 1 提案あたり BFGS 反復数。0 で `chain_iters` | **0** → `chain_iters` | full BFGS | 実体 `chain_iters` は **◆共有** |
| `early_stop_patience` (`termination`) | 改善なし提案で打切り | **0**（tpe 無効） | `termination=steps/25` | ◆共有（全モードの config 欄） |

**TPE 専用ハイパラは `TpeParams`（`tpe.rs`）**: `ei_candidates`, `startup_frac`, `lower_cap`, `greater_cap`,
`density_reuse`, `local_iters`。config `[run]` で `tpe_ei_candidates` / `tpe_startup_frac` を opt-in 上書き可
（未指定なら上記の新既定）。

### drift 等と共有の変数・機構（◆）＝ TPE 固有でない点に注意
- `chain_iters`（=35）— drift は各 MC ステップの浅い BFGS、tpe は `local_iters=0` 時の局所最適化反復として流用。
- `minimize_free`（BFGS 局所最適化）, `contact_start`（初期配置）, `hunt_cap`（curl）, `EvalScratch`/`eval_value_grad`
  （スコア）— 全モード共通の機構。
- 集約・出力（`cluster_poses`, `cluster_rmsd`=2.0, `n_modes`=9, `polish_modes`=9, `polish_iters`=20）—
  `dock_constrained` で全モード共通。
- `n_steps` の **config 欄**は共有だが、未指定時の auto 値はモード別（`auto_n_steps_for`）。

### 現在の既定値まとめ（双方）
- **isct（既定, 2026-07-09 改定）**: `n_runs`=**8**, `n_steps`=auto(`auto_n_steps_vina/15`,≥200), `startup_frac`=**20**,
  `ei_candidates`=**25**, `lower_cap`=10, `greater_cap`=10, `density_reuse`=4, `local_iters`=0(→`chain_iters`=35◆),
  `early_stop`=無効。＝出荷版 TPEvina に忠実な設定（10標的×3seed, box22.5³ で best-of-3 90%, `docs/results.202607091051`）。
- **TPEvina（既定 `bfgs.h`）**: `num_tasks`=`exhaustiveness`(=1), `steps`=`num_steps/15`, warmup=`steps/20`,
  `sample_num`=25, lower=`max(10,min(0.25√N,25))`, greater=全体, reuse=4, local=full BFGS, `termination`=`steps/25`。

---

## 6. 既知の論点（チューニングの現状, 詳細は docs/）

- **既定改定済（2026-07-09）**: greater 順位ベース化（commit f9e37e5）で EI 復活後、`n_runs`=8・深 auto n_steps・
  `startup_frac`=20・`ei_candidates`=25 を既定化。box22.5³ 10標的×3seed で **best-of-3 90%・平均RMSD 3.50**
  （`docs/results.202607091051`）。出荷版 TPEvina に忠実な設定。
- **warmup 最適は greater 活性で 15→20 へ移動**（`docs/findings.202607091051`）。死 greater 時代は sf10–15 が最良
  （`docs/results.202607090841`）だったが、EI 活性後は sf20 が 1ig3/1owe を回復し優る。
- **未了**: n_runs=8 での独立試行減の影響（run 数掃引）、全 Astex での既定検証、1n2j 等の全設定失敗標的（探索非律速）。
- **score と RMSD の乖離**: 深い探索ほど低エネルギー decoy を掴むため score↓でも RMSD が改善しない標的がある
  （採点律速, `docs/findings.202607082223`）。評価の主指標は RMSD 成功率。
