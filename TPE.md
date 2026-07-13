# TPE 探索モード（`--search tpe` / `tpe_fast`）アルゴリズム仕様

実装: `crates/isct-dock/src/tpe.rs`（Parzen モデル）と `crates/isct-dock/src/search.rs`（run 駆動）。
移植元は **TPEvina-2.0**（PSOVina 派生, CBBio）。本書は括弧内に **TPEvina 側の変数名**を添える（例
`warmup(start_search_num)`）。**drift 等と共有の変数は明記**する（◆）。

> **現在の既定は「共有観測モード」**（`single_search_tpe_shared`, `shared=true`）。独立多始点の
> `single_run_tpe` は `tpe_shared=false` 時のレガシー経路。本書は共有モードを主として記述する（§4）。
> 加えて CLI 既定では **局所チェーン（dwelling）・局所探索スキップ・TORSDOF 条件付き抑制**が重畳される（§5）。
> それらの導入経緯・評価は `docs/results.2026071*` / `docs/findings.2026071*` と、要点メモリ
> [[tpe-shared-observation-mode]] / [[tpe-local-chain-prototype]] / [[tpe-fast-local-search-skip]] /
> [[tpe-fast-ef1-rigid-decoy]] を参照。

---

## 1. TPE とはどんなアルゴリズムか

**TPE（Tree-structured Parzen Estimator）** は、過去の評価結果から探索空間の「良さ」を確率モデルで推定し、
次に評価すべき点を賢く選ぶ **逐次モデルベース最適化（SMBO）＝ベイズ最適化** の一種。分布そのものを推定して
次点を提案するため **分布推定アルゴリズム（EDA）** にも分類される。ドッキングでは「配座（ligand pose）→
エネルギー」を目的関数とし、低エネルギーの配座を探す。

手順の骨子:

1. **ウォームアップ（warm-up）**: 最初は探索空間を一様ランダムに評価し、**観測（observation）**（評価点とその
   目的値）を貯める。モデルを作るための初期標本。
2. **良群 / 悪群への分割（good/bad split by quantile）**: 観測を目的値の**分位点**で **良群（good, "lower"）** と
   **悪群（bad, "greater"）** に分ける。
3. **密度推定（KDE / Parzen）**: 各群を Gaussian 混合でモデル化。良群密度 **l(x)**、悪群密度 **g(x)**。
4. **候補提案と選択（EI）**: 良密度 l(x) から複数候補を引き、**期待改善（EI）** 最大の候補を選ぶ。TPE では
   EI ∝ **l(x)/g(x)** に帰着（「良い所に多く・悪い所に少ない」点＝比 g/l を最小化）。
5. **評価と更新**: 選んだ候補を **局所最適化（BFGS）** を通して評価し、観測へ追加してモデルを更新。繰り返す。

> 「Tree-structured」は原論文（Bergstra 2011）の条件付き・階層的変数空間に由来。本実装は各次元を独立に
> モデル化する簡略版（原 TPEvina も同様）。

ISCT 版は探索全体を seed 付き `ChaCha8Rng` で駆動し**同一 seed でビット一致（決定論）**。TPEvina 原実装は
`random_device` シードで run 毎に非再現。

---

## 2. 概念 ↔ コード変数の対応

§1 の概念がコード（`tpe.rs` / `search.rs`）でどの変数・関数に対応するか。**◆＝drift 等と共有**。

| §1 の概念 | コード上の対応 | 補足 |
|---|---|---|
| 探索空間の 1 点＝配座ベクトル | モデル空間ベクトル `v`（長さ `6 + n_free_torsions`） | §3。`[pos(3), rot(3), torsion(n)]` |
| 探索範囲 下限/上限 | `l_limit` / `g_limit`（`tpe::bounds`） | 位置のみ Box 依存。回転・torsion は ±π |
| 参照配座 | `reference`（`contact_start`）◆ | モデル空間は「参照からの差」で定義。共有モードでは全観測・サンプルの固定基準 |
| 観測集合 | `obs: Vec<Obs>` | ボックス内の評価済み点のみ蓄積 |
| 観測の目的値 | `Obs.score` = `res.energy`（curl 込み探索目的） | **順位付け用の `inter` とは別** |
| ウォームアップ | `warmup`, `warmup_sample` | 共有モードは `warmup = budget × startup_ratio`（budget=n_runs×n_steps） |
| 良群 (lower) / 悪群 (greater) | `Dim.lower`/`Dim.greater`, `n_lower`/`n_g` | スコア昇順先頭 `n_lower` が lower |
| 良密度 l(x) / 悪密度 g(x) | `mixture_density(&d.lower,…)` / `mixture_density_weighted(&d.greater,…)` | greater は順位重み付き |
| 期待改善（EI ∝ l/g） | `ratio=gx/lx`; `ei += ratio.ln()`; 最小スロット採用 | Σ_dim log(g/l) 最小＝l/g 最大 |
| 候補数 | `ei_candidates`(=`sample_num`) | §6 |
| 代理モデル | `TpeModel`（`model`） | 共有モードは 1 本を再利用（`density_reuse`） |
| greater 活性フラグ | `greater_active`（`n_g >= 27`） | false で高速経路（EI 省略） |
| 局所最適化（BFGS） | `minimize_free(...)` ◆, `local_iters` | drift と同一の BFGS |
| 代表プール | `pool`（`cluster_insert`）◆ | **`inter` で順位付け** |

**重要な区別**:
- **`Obs.score`（objective, curl 込み探索地形）** で TPE を学習し、**`inter`（純分子間エネルギー）** で pool 順位・
  採点する（探索は滑らかな地形、採点は Vina 互換 affinity の基）。
- **`pool`（最終代表, inter 順・RMSD クラスタ, 上限 `pool_cap`）** と **`obs`（学習用, objective 順）** は役割が別。

---

## 3. モデル空間（探索空間ベクトル）

各配座を長さ `6 + n_free_torsions` のベクトルで表す:

```
[ pos(3, 絶対座標) , rot(3, 参照配座からの scaled-axis 差) , free_torsion(n, 参照からの差) ]
```

- 範囲: 位置 = 探索ボックス角 `center ± size/2`、回転・torsion = `[-π, π]`。位置だけ Box サイズ依存。
- 変換: `tpe::conf_from_vec`（ベクトル→配座）, `tpe::vec_from_conf`（配座→ベクトル）。

---

## 4. run 駆動：共有観測モード（既定）と独立モード

### 4.1 共有観測モード（`single_search_tpe_shared`, `shared=true`＝既定）
独立 run をやめ、**総予算 `budget = n_runs × n_steps` を 1 本の共有観測プール・単一 Parzen モデルで逐次探索**
する。TPE は初期依存性が小さいため、独立多始点の冗長（各 run で warmup をやり直す等）を排し、少予算で
n_runs=8 相当へ到達させる（利得源は**観測共有**そのもの。`docs/results.202607111622`, [[tpe-shared-observation-mode]]）。

ループ（提案 `p = 0..budget`）:
1. **スタート配座の決定**（`phase` で分岐）:
   - `p < warmup` → **一様**（`warmup_sample`）。`warmup = budget × startup_ratio`。
   - `reanchor_period>0 && p%reanchor_period==0` → **カバレッジ加重スタート**（`coverage_start`, 既知観測から
     遠い＝未踏域を優先。`coverage_weight>0` 時）。**既定 `reanchor_period=0`＝周期リアンカーなし**（単一アンカーの
     共有探索が推奨。有無で精度ほぼ不変）。
   - それ以外 → **モデル提案**（`TpeModel::build`→`sample`。再構築周期 `freq = (budget−warmup)/density_reuse`）。
2. **（任意）局所探索スキップ**（§5.2）: `skip_local` 時、非 warmup 提案の start を勾配1回で判定し既知 basin なら
   初期 BFGS 以降を省いて次提案へ。
3. **初期局所最適化** `minimize_free`（BFGS, `local_iters`）◆ → `pool` へ `cluster_insert`（キー=`inter`）。
4. **（任意）局所チェーン（dwelling）**（§5.1）: `chain_len>0` かつ非 warmup かつ competitive（`res.energy ≤
   best_e + local_chain_margin`）なら、Drift 流の短い MC チェーン（mutate+BFGS+Metropolis を `chain_len` 歩）で
   精密化。チェーンの最良ポーズを観測に採用、各局所最小は pool へ。
5. **記録**: `best_e` 更新、ボックス内なら `obs` へ push（キー=`objective`）。
6. 終了後 `pool` を `inter` 昇順で返す→`dock_constrained` が深い polish→`n_modes` 出力 ◆。

**決定論**: 単一 `ChaCha8Rng` の逐次消費で維持。各種既定 OFF（`local_chain=0`, `skip_local=false`,
`reanchor_period=0`, `rigid_torsdof=0`）なら **RNG 消費まで現行とビット一致**。

### 4.2 独立モード（`single_run_tpe`, `shared=false`＝レガシー）
`n_runs` 本の独立 run を rayon 並列。各 run が `contact_start`→warmup→モデル提案→BFGS→自 run の pool を返し、
`dock_constrained` が統合。共有モード導入前の挙動（`docs/results.202607100834` とビット一致）。**局所チェーン・
スキップ・TORSDOF 抑制は共有モード側のみ**に実装（独立モードは基本 TPE のまま）。

### 4.3 Parzen モデル（`tpe.rs`）
- **build**: `obs` をスコア昇順→ lower `n_lower = clamp(0.25√N, 2, lower_cap)` / greater（残り `n_g`）へ分割。
  各次元で lower/greater の Gaussian 混合（σ は隣接ギャップ規則）を構築。
- **greater 重み（順位ベース）**: 各 greater 観測 `gi`（0=最良greater）に `weight = clamp(gi/(n_g−26), 0, 1)`。
  **`n_g ≥ 27` で活性**（`greater_active`）。反復番号には依存しない（出荷版 TPEvina `bfgs.h` 準拠。**旧 `*_7s`
  減衰式は greater が死んで EI が無効化していた**バグを修正済＝`docs/findings.202607090942`, [[tpe-greater-rankbased-ei]]）。
- **sample**: `greater_active==false` → 各次元 1 点（高速経路）。`true` → 各次元 `ei_candidates` 個引き
  Σ log(g/l) 最小の候補を採用（完全経路＝本来の EI）。

---

## 5. 探索強化機構（CLI 既定で重畳。ライブラリ既定は全 OFF）

共有モードには、基本 TPE の上に **3 つの opt-in 機構**が載る。**ライブラリ `TpeParams` 既定は全 OFF ＝
基本 TPE とビット一致**。有効化は CLI プリセット（§7）または config で行う。

### 5.1 局所チェーン（dwelling）— `local_chain` K
**背景**: トレース解析（`docs/findings.20260711*`, 8標的）で、TPE の失敗は native 基を「見つけられない」のでは
なく「**留まって精密化できない**」ことに起因すると判明。MC（drift/vina）は native 基に連続滞在（streak 5–17）
するが、TPE は孤立単発（streak≈1）。この dwelling 欠如が深い native 最小への精密化不足・良ポーズの pool 残存性
不足を招く。

**機構**: 各提案の初期 BFGS 後、**competitive な basin**（`res.energy ≤ best_e + local_chain_margin`）なら Drift 流の
短い MC チェーン（`apply_mutation`(Refine{scale})→`minimize_free`→Metropolis を `chain_len` 歩）で「留まって
精密化」する＝MC の持続滞在を部分移植。チェーンの最良ポーズを観測に採用、各局所最小は pool へ挿入。
- ノブ: `local_chain`(K, 0=OFF) / `local_chain_margin`(ゲート幅 kcal, 広いほど連鎖多) /
  `local_chain_scale`(変異幅スケール, <1.0 で並進 ±2·scale Å・回転 ±scale rad・torsion ±π·scale に縮小＝
  basin-hop〔歩き去り〕抑制)。
- 効果: explore 律速で native が最深な標的（1lpz/1ig3 型）を救済。低 n_runs で独立 restart を代替し省コスト化
  （`docs/results.20260712*`, [[tpe-local-chain-prototype]]）。

### 5.2 局所探索スキップ（QuickVina2 相当）— `skip_local`
**背景**: tpe_fast の律速は初期 BFGS ではなく**局所チェーンの過剰発火**（易・中標的で eval が 1M 級に膨張）。
モデルが良 basin へ集中するほど近接した冗長提案が増え、その各々にチェーンがかかる。

**機構**: 既存の QuickVina2 スキップ（`qvina.rs` の `HistoryBuffer::is_new_well`）を共有ループに移植。**非 warmup**
提案の start を**勾配1回**で判定し、近傍の既知斜面と勾配方向が一致（cos>`skip_cos`）＝同一 basin なら初期 BFGS・
チェーン・pool・obs をすべて省いて次提案へ（既知 basin の極小は既に pool/obs に在る）。RNG 不使用＝決定論。
- ノブ: `skip_local`(bool, 既定 false) / `skip_cos`(0.90, 高いほど保守的＝スキップ少)。
- **skip はモデルを集中させるため、chain の `margin` を再較正しないと連鎖が過剰発火して逆効果**。tpe_fast では
  `margin=4.0→2.0` に絞って両立（Astex24 で eval −63%・wall −39%・精度非退行。`docs/results.202607130928`,
  [[tpe-fast-local-search-skip]]）。

### 5.3 TORSDOF 条件付き局所チェーン抑制 — `rigid_torsdof` / `rigid_local_chain`
**背景**: DUD-E 診断（`HANDOFF.ef1_diagnosis.md`）で tpe_fast の early 濃縮 EF1% 低下は**剛直・低 TORSDOF
デコイの過小スコア化**が主因と判明。tpe_fast は Vina exh8 より強い最適化器で、剛直デコイについて Vina 関数の
真の深い（偽陽性）極小を確実に掘り当て、それらが top1% に侵入してアクティブを押し出す（＝最適化力↑≠濃縮↑）。

**機構**: `chain_len = if free_torsions.len() < rigid_torsdof { rigid_local_chain } else { local_chain }`。
`free_torsions.len()`（=回転結合数, singleton は nominal TORSDOF に対応）で剛直/柔軟を分岐し、**剛直リガンドのみ
チェーンを弱める**（0 なら剛直はチェーン無効＝基本 BFGS のみで Vina 相当の探索深度へ寄せる）。柔軟リガンド
（アクティブに多い）は不変。
- ノブ: `rigid_torsdof`(閾値, 0=OFF＝`n<0` が常偽でビット不変) / `rigid_local_chain`(剛直時 K)。
- **剛直アクティブも同様に抑制される active/decoy トレードオフ**のため**既定 OFF**。EF1% 改善の可否は DUD-E で
  `rigid_torsdof∈{4,6,8}`×`rigid_local_chain∈{0,3,8}` を掃引して判定（`docs/results.202607131021`,
  [[tpe-fast-ef1-rigid-decoy]]）。

---

## 6. パラメータ一覧（`TpeParams`, `tpe.rs`）

| isct 変数 (TPEvina 名) | 意味 | 既定 | config キー |
|---|---|---|---|
| `ei_candidates` (`sample_num`) | EI 用に lower から引く候補数 | **25** | `tpe_ei_candidates` |
| `startup_ratio` (`start_search_num`) | warmup 比率（warmup=budget×これ） | **0.05** | `tpe_startup_ratio` |
| `step_ratio` (`1/15`) | auto n_steps 係数（`auto_n_steps_vina×これ`, 大=深） | **0.1** | `tpe_step_ratio` |
| `lower_cap` (`num_lower_groups`) | lower サイズ上限（実=clamp(0.25√N,2,これ)） | **10** | `tpe_lower_cap` |
| `greater_cap` (`num_upper_groups`) | greater カーネル上限 | **10** | `tpe_greater_cap` |
| `density_reuse` (`kernel_density_frequency`) | モデル再構築回数 | **6** | `tpe_density_reuse` |
| `local_iters` (—) | 1 提案あたり BFGS 反復（0→`chain_iters`◆） | **0** | — |
| `shared` (—) | 共有観測モード（§4.1） | **true** | `tpe_shared` |
| `reanchor_period` (—) | 共有モードのリアンカー周期（0=なし） | **0** | `tpe_reanchor_period` |
| `coverage_weight` (—) | 未踏域カバレッジ重み（0=無効） | **0.0** | `tpe_coverage_weight` |
| `local_chain` (—) | 局所チェーン歩数 K（§5.1, 0=OFF） | **0** | `tpe_local_chain` |
| `local_chain_margin` (—) | チェーンのゲート幅 kcal | **2.0** | `tpe_local_chain_margin` |
| `local_chain_scale` (—) | チェーンの変異幅スケール | **1.0** | `tpe_local_chain_scale` |
| `skip_local` (—) | 局所探索スキップ（§5.2, QuickVina2 相当） | **false** | `tpe_skip_local` |
| `skip_cos` (`qvina_cos`◆相当) | スキップの勾配方向一致閾値 | **0.90** | `tpe_skip_cos` |
| `rigid_torsdof` (—) | TORSDOF 条件付き抑制の閾値（§5.3, 0=OFF） | **0** | `tpe_rigid_torsdof` |
| `rigid_local_chain` (—) | 剛直リガンドの K | **0** | `tpe_rigid_local_chain` |

**drift 等と共有（◆）**: `chain_iters`(=35, `local_iters=0` 時の BFGS 反復) / `minimize_free`（BFGS） /
`contact_start`（初期配置） / `hunt_cap`（curl） / `eval_value_grad`（スコア） / 集約・出力（`cluster_rmsd`=2.0,
`n_modes`=9, `polish_modes`=9, `polish_iters`=20, `dock_constrained`）。`n_steps` の config 欄は共有だが未指定時の
auto 値はモード別（`auto_n_steps_for`：tpe=`auto_n_steps_tpe`）。

**n_runs（`num_tasks`=`exhaustiveness`）**: TPE の未指定既定は **dock=3 / screen(=tpe/tpe_fast)=1**
（`DEFAULT_N_RUNS_TPE=3`, `default_n_runs_for`）。共有モードでは n_runs は総予算 `budget=n_runs×n_steps` の係数。
旧既定 8 は独立モード時代の値（局所チェーンが restart を代替し低 n_runs 化。[[tpe-local-chain-prototype]]）。

---

## 7. CLI 既定プリセット（`build_search_params`, `main.rs`）

ライブラリ `TpeParams` 既定は全機構 OFF（基本 TPE）。CLI 層で **モード別プリセット**を適用（config で上書き可）。

| | dock（`--search tpe`） | screen / `--search tpe_fast` |
|---|---|---|
| `n_runs` | 3 | 1 |
| `local_chain` (K) | 15 | 15 |
| `local_chain_margin` | 2.0（既定） | **2.0** |
| `local_chain_scale` | 1.0（既定） | **0.75**（walk-off 抑制） |
| `skip_local` | false | **true** |
| `rigid_torsdof` | 0（OFF） | 0（OFF, DUD-E 掃引で opt-in） |

- **dock**: n_runs=3 + K=15。旧 n_runs=8 既定を精度同等以上・時間 −21% で達成（`docs/results.202607121032`）。
- **screen/tpe_fast**: n_runs=1 + K=15 + skip + margin=2.0 + scale=0.75。低予算スクリーニング向け。skip で冗長
  basin の初期 BFGS を削り、margin=2.0 が精度中立の最速点（Astex24 eval −63%・wall −39%・成功非退行）。
- **`rigid_torsdof` は両モードとも既定 OFF**（EF1% トレードオフ未検証。§5.3 の DUD-E 掃引で確定後にプリセット化を
  ユーザ判断）。

---

## 8. 既知の論点（詳細は docs/）

- **共有モードが既定**（2026-07-11〜）: 独立 n_runs の冗長を排し、同予算で旧既定を上回り／同精度を n_runs5–6 で
  1.3–1.5× 高速（`docs/results.202607111622`）。利得源は観測共有（restart/coverage は不要＝既定 0）。
- **局所チェーン**が低 n_runs での restart 代替＝省コストの本領（`docs/results.202607121032`/`.121135`）。
- **skip の律速理解**: tpe_fast の主コストは初期 BFGS でなく局所チェーン。skip 単体では不十分で margin 再較正が
  必須（[[tpe-fast-local-search-skip]]）。
- **EF1%＝剛直デコイの過小スコア化**（構造由来, grid バグではない）: 探索側は §5.3 の TORSDOF 条件付き抑制で
  対処を試みる（DUD-E 掃引で検証中）。スコア関数側（剛直デコイの偽陽性極小識別）は次段。`drift`(n_runs=60) は
  更に深く探索するため現象が強まる予測（[[tpe-fast-ef1-rigid-decoy]]）。
- **score と RMSD の乖離**: 深い探索ほど低エネルギー decoy を掴むため score↓でも RMSD が改善しない採点律速標的が
  ある（`docs/findings.202607082223`）。評価は RMSD 成功率と affinity を必ず併記（[[always-include-score-in-results]]）。
- **per-core 速度**: 公平 1 スレッド計測で TPE 2.70×Vina（[[per-core-speed-tpe-fastest]]）。
