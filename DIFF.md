# DIFF.md — ISCT-Dock と AutoDock Vina の違い（アルゴリズム / 実装）

ISCT-Dock は **Vina のスコア関数を厳密に再現**しつつ、**探索アルゴリズムと実装を置き換えた**決定論的
ドッキングエンジン。本書は両者の差を「アルゴリズムレベル」と「実装レベル」に分け、**約 3 倍速い理由**を示す。
（性能実測は [docs/results.202606212151](docs/results.202606212151.md)・[docs/results.202606220846](docs/results.202606220846.md)。
kif11: isct-dock 1.93 lig/s vs vina 0.61 lig/s ＝ **~3.2×**、探索ポーズのエネルギーは同等＝ −9.2〜−9.4。）

## 0. 一目で

| 観点 | AutoDock Vina | ISCT-Dock |
|---|---|---|
| スコア関数 | gauss1/gauss2/repulsion/hydrophobic/hbond＋torsion 補正 | **同一**（重み・XS 型・cutoff まで一致, `score` で <1e-2） |
| 大域探索 | Monte Carlo（exhaustiveness 本の独立 run） | **ドリフト鎖 ILS**（多数 run × 多 step、各 step 浅い局所最適化） |
| 局所最適化 | 各 MC step で **収束まで BFGS** | 各 step は **浅い BFGS（30 反復）**＋最終に少数モードのみ**深い厳密 polish** |
| 探索中の評価 | 受容体グリッド（補間, 直線探索含む） | 受容体グリッド（補間, 直線探索含む）＝**Vina と同等（差ではない）** |
| 乱数 | 確率的（seed 固定でも完全再現しにくい） | **完全決定論**（ChaCha 固定 seed, ビット再現） |
| 実装 | C++ | Rust（ゼロアロ探索・解析的勾配・rayon・f32 SIMD 直接バックエンドも併存） |

## 1. スコア関数 ＝ アルゴリズムレベルで「同一」

差ではなく**一致**だが、3 倍速が「精度を落とした近似」ではないことを示すため明記する。

- 5 項（`crates/isct-core/src/scoring.rs`）: `gauss1=exp(-(d/0.5)²)`, `gauss2=exp(-((d-3)/2)²)`,
  `repulsion=d<0?d²:0`, `hydrophobic`（区分線形, 疎水ペアのみ）, `hbond`（区分線形, ドナー-アクセプタのみ）。
- 重みも Vina と同値: `W_GAUSS1=-0.035579, W_GAUSS2=-0.005156, W_REPULSION=0.840245,`
  `W_HYDROPHOBIC=-0.035069, W_HBOND=-0.587439`、cutoff 8.0 Å。
- XS 原子型・距離 `d=r-(R_i+R_j)`・torsion 補正 `affinity = e_inter/(1+0.05846·N_rot)` も一致。
- **検証ゲート（M1）**: `isct-dock score` と `vina --score_only` の差 <1e-2 kcal/mol。
  → 同一ポーズなら出力スコアは Vina と一致する。速度は探索・実装で稼いでおり、スコア精度は犠牲にしていない。

## 2. 探索アルゴリズムの違い

### Vina（Monte Carlo + 各 step フル BFGS）
- `exhaustiveness`（既定 8）本の**独立 MC run**。各 run は: ランダム大域変異 → **収束までの局所 BFGS** →
  Metropolis 受理、を多数 step 繰り返す。**1 step あたりの局所最適化が重い**（毎回しっかり谷底まで降りる）。

### ISCT-Dock（ドリフト鎖 Iterated Local Search, `crates/isct-dock/src/search.rs`）
- **多数 run**（既定 `n_runs=60`）。各 run は受容体と productive に接触する初期配置から開始
  （`contact_start` の棄却サンプリングで浮遊・激突を排除）。
- **per-run step 数は自動スケール**: `auto_n_steps = max(30·min(n_atoms, 8+5·n_tors), 300)`。
- 各 step は「**1 自由度だけ変異 → 浅い局所最適化（`chain_iters=30` 反復, 収束させない）→ Metropolis 受理**」。
  浅い谷下りを**鎖のように多数つなぐ**ことで、フル最適化を毎回せずに広く探索する（＝ドリフト鎖）。
- run 内で訪れた極小を **RMSD クラスタリングでプール**し、全 run 集約後に大域 RMSD クラスタで上位
  `polish_modes=9` モードへ絞り、**そこだけ深く厳密に polish**（`polish_iters=20`, 厳密 f64 表面）。
- 焼きなまし（`temp_init→temp_final` 幾何冷却）・QuickVina2 相当の局所探索スキップ（opt-in）も持つ。

> 要点: Vina が「**毎 step で深く最適化**」するのに対し、ISCT は「**毎 step は浅く・最後に少数だけ深く**」。
> 同等のポーズ品質（実測エネルギー同等）を、総計算量を抑えて達成する。

## 3. 実装レベルの違い

- **言語**: Rust（Vina は C++）。
- **グリッド探索**: 受容体ポテンシャルを **XS 型ごとの 3D マップ**（spacing 0.375 Å, margin 4.0）に前計算し、
  **trilinear 補間**で値・勾配を引く（`crates/isct-core/src/grid.rs`）。探索の 1 評価が **O(n_ligand)**
  （原子数のみ）で、受容体原子数に依存しない。
- **直線探索もグリッド整合**: 局所最適化の line search の値評価まで **グリッド補間**（`search_score`/`grid.value`）
  を使う。以前は line search だけ直接和 O(n_lig×n_rec) になっており、ここをグリッド整合にした改修が
  **単独で約 3.5× の高速化**だった（後述 §4-①）。
- **解析的勾配**: 力・DOF 勾配を解析計算（有限差分なし）。`value_and_force`/`dof_gradient_into`。
- **ゼロアロケーション探索**: 評価バッファ（`EvalScratch`: world 座標・原子・力・勾配）を run/polish ごとに
  使い回し、ホットループで heap 確保しない。
- **厳密 f64 polish と探索近似の二段**: 探索はグリッド近似（速い）、最終 polish と報告スコアは厳密 direct f64
  （`score`/`exact_value_and_force`）で Vina 一致を担保。
- **並列**: run 単位で rayon 並列（スクリーニングではリガンド/クラスタ単位でも並列）。
- **決定論**: `rand_chacha` を seed から決定的に分岐。同一入力・seed でビット一致の結果。
- （補足）`Direct` バックエンド（f32 SIMD で原子間を直接計算）も併存するが、不規則 gather と精度感受性で
  グリッドに対し優位でなく、既定は Grid（[SIMD_OPTIMIZE.md](SIMD_OPTIMIZE.md)）。

## 4. なぜ約 3 倍速いのか（寄与の分解）

実測 per-ligand: **isct-dock ~0.5 s/件 vs vina ~1.65 s/件**（同一 40 物理コア, exhaustiveness 8 ≈ n_runs 60）。

> **重要な前提（誤解しやすい点）**: グリッド補間は **Vina も探索中に使っている**（受容体を原子型ごとのアフィニティ
> グリッドに前計算し、MC+BFGS の全評価＝直線探索を含めて trilinear 補間で引く）。したがって **グリッド整合は
> Vina に対する優位ではなく「パリティ（追いつき）」**である。ISCT の改修史にある「直線探索のグリッド整合 ~3.5×」
> は、ISCT 自身が以前 line search だけ直接和 O(n_lig×n_rec) に退化していた**内部バグを直して Vina 並みに戻した**
> もので、**Vina より速くなる理由ではない**。→ 「もし探索アルゴリズムを Vina と完全同一にしたら、グリッド由来の
> 差はゼロになり、3× の優位はほぼ消える（残るのは Rust vs C++ の実装定数倍のみ＝控えめ）」。

**Vina に対する 3× の真の源泉 ＝ 探索アルゴリズムが少ない計算量で同等ポーズに到達すること:**

1. **浅いドリフト鎖による評価回数の削減（主要因）**: 各 step を `chain_iters=30` の**浅い BFGS**に留め
   （収束させない）、深い厳密最適化は最終の `polish_modes=9` モードのみに限定。Vina の「**毎 MC step で収束まで
   フル BFGS**」に比べ、**総評価回数が大幅に少ない**。同じグリッド 1 評価コストでも、回数が少ないぶん速い。
   ＝ ここが Vina との差の中心。

2. **Rust ＋ ゼロアロ ＋ 解析的勾配（実装定数倍）**: ホットループで heap 確保ゼロ・有限差分なし・バッファ再利用。
   同じ計算量でも 1 評価の定数倍が小さい。

3. **接触開始サンプリング（無駄の削減）**: `contact_start` で「最適化が効かない浮遊/激突開始」を捨て、無駄な
   run/step を減らす。

> まとめ: **グリッドは Vina と同じ（差にならない）**。3× の優位は「**浅い鎖＋少数の深い polish で評価回数を
> 減らす探索設計**」＋「**Rust 実装の小さい定数倍**」から来る。スコア関数は Vina と同一なので、速度はアルゴリズム・
> 実装由来で、精度トレードオフではない。**探索アルゴリズムを Vina と同一化すれば、この優位は維持されない。**

## 5. 品質・限界（速いが何を失っていないか）

- **探索品質は同等**: 共通スケール（`vina --score_only` 再採点）で active のベストエネルギーは
  isct −9.24 / vina −9.35 / smina −9.20。isct のポーズを vina で採点すると AUC 0.827（vina 自身 0.827 と一致）。
  → 3 倍速でもポーズの質は落ちていない（[docs/results.202606220846](docs/results.202606220846.md)）。
- **native ランキング**: 以前 `dock` は生 intermolecular を出力し識別力が低かった（AUC 0.778）。
  Vina 互換 affinity（torsion 補正）を出力に追加して解消（~0.82, [docs/findings.202606221038](docs/findings.202606221038.md)）。
- **残課題**: 単一標的（kif11）では vina/smina に僅差で劣る指標もあり、複数 DUD-E 標的での確認が必要。
  ISCT-screen（scaffold 再利用）は探索が浅くなり別物（[docs/results.202606220300](docs/results.202606220300.md)）。
