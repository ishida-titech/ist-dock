# PROGRESS.md — ISCT-Dock 進捗

マイルストーン（[PLAN.md](PLAN.md)）の実装・検証状況を**随時更新する living document**。
（`docs/results.*`・`docs/findings.*` は一度作成したら更新しない時系列ログ。**本ファイルは現状を反映して
上書き・追記する**。新機能・検証を入れたらこの表と詳細・更新履歴を更新すること。）

最終更新: 2026-07-03

## マイルストーン状態

| マイルストーン | 状態 | 要点 |
|---|---|---|
| M1 Vina スコア一致 | **完了** | `score`。`vina --score_only` と差<1e-2。 |
| M2 Vina 互換ドッキング | **完了** | `dock`。torsion 木・剛体+回転+torsion・**解析的勾配**・MC+BFGS・決定論。再ドッキング<2Å。 |
| M3 クラスタ高速化 | **完了** | library/Phase1-3/delta_energy・前処理・manifest・効果検証・**screen ポーズ出力**・**複数 seed ポーズ保持**。 |
| M4 単一ノード最適化 | **完了** | grid・SIMD・ゼロアロケーション・rayon・解析的勾配・**core_affinity**・**memmap+バイナリ・ライブラリ streaming**。 |
| M5 追加高速化（動的 box） | 未着手 | PCA 身の丈ボックス等。 |
| M6 精度向上 | 未着手 | アンサンブル・Soft-core・柔軟側鎖。 |
| 分散（複数ノード） | スコープ外 | 別計画。 |

## 詳細

### M1 — Vina スコア一致【完了】
- PDBQT パーサ・XS 原子型・Vina スコア（gauss1/2・repulsion・hydrophobic・hbond, inter+intra）・`score` サブコマンド。
- `vina --score_only` 一致ゲート <1e-2 達成。

### M2 — Vina 互換ドッキング【完了】
- リガンド torsion 木、剛体並進(3)+回転(四元数)+torsion、ボックス罰則、接触開始、分子内エネルギー。
- **解析的勾配**（`dof_gradient` 連鎖則）＋ BFGS 局所最適化＋ Monte Carlo（ILS）。固定シードで決定論。
- 再ドッキング RMSD<2Å（3ptb/1stp/1iep, pdb/1iep 5/5）。

### M3 — クラスタ高速化【完了（中核＋効果検証）】
- クラスタデータ構造（Seed+Children/MCS, `isct-io/library.rs`）、前処理 `tools/preprocess/build_library.py`、
  manifest（`bench/data/library/ampc_*`）。
- Phase 1（seed ドッキング＋一括棄却）/ Phase 2（コア座標ロック・固定マージン）/ Phase 3（R 基 torsion＋
  Δエネルギーキャッシュ `delta_energy.rs`）。RMSD ポーズクラスタリング。
- 効果検証（`docs/results.202606172136`）: 削減率/棄却/偽陰性/Scaffold 拘束劣化/速度。結論＝ampc@0.55 は
  旨味小・拘束劣化が本質課題。
- **screen の上位ポーズ出力**: `screen --poses-out <file> [--top N]` でマルチモデル PDBQT（REMARK 付）。
  `screen::poses_pdbqt`（`docs/results.202606201202`）。
- **複数 seed ポーズ保持**: `[screen] seed_poses=N` で各子を上位 N seed モードへ配置し最良採用＝拘束劣化を
  緩和（ampc_mini で 114 −1.93→−5.31）。既定 1＝従来。
- **スクリーニング grid 再利用**（既定ボックス Scorer 共有, ビット一致, N=50 で 10.4×, `docs/results.202606201138`）。

### M4 — 単一ノード最適化【完了】
- グリッド補間（`grid.rs`）、SIMD f32x8（`scoring_simd.rs`, AVX2 codegen で有効）、ゼロアロケーション再利用
  バッファ（`EvalScratch`）、rayon スレッド並列、解析的勾配。
- **core_affinity**: `screen --pin-threads` でワーカを物理コアにピン留め（Linux）。
- **memmap+カスタムバイナリ・ライブラリ**: `isct-io/library_bin.rs`（serde+bincode）。`pack` で TOML→バイナリ、
  `screen <lib.bin>` は memmap で**クラスタ単位の遅延デコード**＝全 RAM ロード回避。`ClusterSource` trait で
  `Library`/`BinaryLibrary` 両対応の汎用 `screen`。TOML 経路とビット一致（`docs/results.202606201419`）。
- 検証: 最適化前後でスコア不変・スループット計測済（`docs/results.20260618*`, `docs/results.202606191907` 等）。
- 残: **10M 規模の大規模テスト（スループット/メモリ実測）は後日**（別途実施）。

### M5 / M6 — 未着手
- M5: 動的バウンディングボックス（PCA）・二面角優先度。
- M6: マルチ受容体アンサンブル・Soft-core・柔軟側鎖。

## マイルストーン外の追加実装・最適化（本体性能/機能）

- **探索の改良**: ドリフト鎖（浅い per-step＋多 step＋最終深 polish）、RMSD ポーズクラスタリング（多モード出力）、
  直線探索を grid 値に整合（3.5×）、深 polish 浅化、torsion 連動 step 予算。
  → 多様 74 標的で **63%（2.05s）、Vina(64%/7.6s) 比 ~3.7× 高速**（`docs/results.202606191907` 等）。
- **QuickVina2 相当の局所探索スキップ**（opt-in `search="qvina2"`, `qvina_cos` 可変, 既定 0.90, 決定論維持,
  `docs/results.202606201115`）。QuickVina-W は未対応。
- **スクリーニング grid 再利用**（既定ボックスの Scorer を 1 度構築・共有, ビット一致, クラスタ数に比例して
  高速化 N=50 で 10.4×, `docs/results.202606201138`）。
- 知見ドキュメント: `SIMD_OPTIMIZE.md`（SIMD は不規則 gather/精度感受性で割に合わず撤回）、`MANUAL.md`（利用法）。

## 大規模スクリーニング評価（外部ツール比較）

- 設計: [EVALUATION.SCREENING.md](EVALUATION.SCREENING.md)。ハーネス: `bench/{prep_dude_ligands,screen_eval,
  screen_metrics}.py`。比較手法: ISCT-dock / Vina / Smina（GWOVina・ISCT-screen は後続）。
- **第1弾 DUD-E kif11（6966 リガンド）完了**（`docs/results.202606211249`）。40 物理コアを全手法へ公平配分:
  - **速度: isct-dock 2.02 lig/s（57分）＝ vina/smina（0.61/0.62, 3.1–3.2h）の約 3.3 倍速**。
    vina≈smina は同エンジン（Vina 1.1.2 系）で公平性を裏付け（smina はシャード並列化が必須）。
  - **探索品質は 3 手法同等**（共通 vina 再採点で active −9.2〜−9.4、AUC isct 0.829≈vina 0.827>smina 0.794）。
  - **改善対象は探索でなく isct-dock の native ランキングスコア**（native AUC 0.776 が共通再採点で 0.829 へ）。
- 評価過程でハーネスの採点バグ（マルチMODEL／smina meeko 変換失敗による native フォールバック）を発見・是正
  （`docs/findings.202606211249`）。
- **CG0/G0（meeko マクロ環グルー原子）対応を実装**（isct-core, commit 27f6819）→ 従来除外の 257 件を解消し
  **isct-dock 全 6964 で再評価＝完全カバー版**（`docs/results.202606212151`）:
  - 共通スケール **AUC isct 0.827 ＝ vina 0.827 > smina 0.794**、EF@1%/BEDROC は isct 最良。速度 **約 3.1 倍**
    （1.93 lig/s, 60分）。結論は部分カバー版と不変（除外バイアスは小）。
  - マクロ環リガンドは閉環拘束未実装で ~30s/件と遅い（将来の最適化余地）。
- **ISCT-screen（scaffold 再利用）を kif11 で評価**（`docs/results.202606220300`）。全カバー版ライブラリ
  （`build_library.py --full-coverage`, 6966 全収録, 構築 4:33）で `isct-dock screen` を実行:
  - **再利用は 8%（565 children）のみ**＝DUD-E 多様性でコア≥55% の congeneric 子が少なく、92% は全自由度
    ドッキング。速度は isct-dock 比 **+10%（2.12 lig/s）**、濃縮はむしろ低下（共通 AUC 0.763 < isct-dock 0.827）。
  - → **ISCT-screen は congeneric ライブラリ向け**。多様な VS 集合では per-ligand の isct-dock が優位。用途で使い分け。
  - 実装修正: `library.rs`空コア許可 / `screen.rs` seed の n_steps 自動化（従来 steps=0 で不正） /
    `build_library.py --full-coverage`・in-process meeko・拘束 embed 上限化・コア割合フィルタ 0.55。
- 評価方針: **全ランキングパターン×全指標（AUC/EF@0.5/1/2/5%/BEDROC）を毎回そろえ複数標的で判断**
  （`screen_metrics.full_metrics`）。AUC 単独で優劣を決めない。
- 残: congeneric ライブラリでの ISCT-screen 評価、共通交差集合での再計算、マクロ環閉環拘束、akt1/hivrt 拡大、
  GWOVina、10M 本番、isct-dock の native ランキングスコア改良（torsion 補正の露出）。

## 更新履歴
- 2026-07-05: **TPE の「Parzen モデル未使用」バグ発見・修正 ＋ 高速化（①+ei実効1）**
  （`docs/findings.202607051047`）。高速化検討中に before/after ビット一致に気付き計装した結果、**aa38048 の TPE は
  観測が一切記録されず Parzen モデルが構築されていなかった**（実体はランダム再スタート+BFGS）。根本原因は位置
  パラメトリゼーションの取り違え: ISCT の `Conformation.position` は `root_pivot` からの**変位**なのに TPE モデルは
  次元 0–2 を**絶対座標**として扱い、`in_box` が常に false→obs 未記録。`conf_from_vec`/`vec_from_conf`/`in_box` に
  `root_pivot` を通し絶対⇄変位を正しく変換して修正（obs=60・model 稼働を確認）。あわせて高速化: greater 重みが
  0 のとき g(x)=0 で候補選択が退化するため、`sample()` に高速経路（各次元 1 点のみ・g/l と未使用候補を省略）を追加
  （既定で常用）。**壁時計効果は ~0%（BFGS 律速, 事前分析通り）**。修正で TPE が初めて真に稼働するため品質特性は
  再評価が必要（`results.202607051022` の数値はモデル未使用時のもので TPE 性能ではない）。決定論維持・clippy 警告
  ゼロ・test 全 pass（TPE 単体 5 件）。**未コミット**（挙動が大きく変わるためユーザ確認待ち）。
- 2026-07-05: **TPE 探索モード追加**（`--search tpe` / `search="tpe"`, `docs/results.202607051022`）。TPEvina-2.0
  （PSOVina 派生）の TPE（Tree-structured Parzen Estimator）を移植。新規 `crates/isct-dock/src/tpe.rs`
  （per-dim Gaussian Parzen モデル・良観測近傍サンプリング・g/l 期待改善）＋ `single_run_tpe`（`single_run` と
  同型, `dock_constrained` で分岐）。局所最適化=`minimize_free`(BFGS)・スコア/grid・**決定論(seeded ChaCha8Rng)**
  を共通機構で再利用。原実装は `std::random_device` で非決定的（作者 FIXME）だが本移植は決定的＝同一 seed で
  ビット一致（分布/FP 差のため原とはビット一致しない忠実再実装）。SearchKind/SearchMode/CLI/auto_n_steps_tpe を
  配線、screen も同経路で対応。1iep 再ドッキング 1.178Å（drift 1.238/vina 1.242 と競合）。品質は標的依存
  （深い漏斗で良好・探索性の要る標的で drift に劣る）。ハイパーパラメータは TPEvina 既定で固定（config 露出は将来）。
  clippy 警告ゼロ・test 全 pass（TPE 単体 4 件）。
- 2026-07-03〜04: **高速化の探索＋実測**（`docs/findings.202607030101`, 速度 A/B `docs/results.202607042210`,
  Astex 影響確認 `docs/results.202607042150`）。数値ビット不変のアロク削減を採用: Phase3 子 objective 脱アロク
  (T1-1)・`single_run` の lock_torsions/apply_mutation 脱アロク(T1-3)・**BFGS 逆ヘシアン更新の作業行列再利用**(T1-4,
  nalgebra `mul_to` で行列積の結合順を厳密一致→ビット一致)。**Astex Diverse 74 標的で HEAD と 74/74 ビット一致**
  （affinity・pose RMSD 完全一致＝回帰ゼロ）。clippy/test green。
  - **有効だった手法＝T1-4＋T1-3**: 単発 dock（Direct, 60 run × 多数 BFGS 反復）で n×n 行列 5–7 枚/反復の確保除去が
    効き **5.22→5.06s（~3%）**（交互計測・ビット一致）。
  - **中立＝T1-1**: 本 kif11 screen は seed 大域 docking 支配＋grid 探索/分子内 exp が計算律速でアロケータ律速でなく、
    壁時計は 103.0→102.9s（中立・ビット一致）。子 fan-out 大のライブラリでは効く可能性（未実証）。
  - **撤回＝T1-5（rayon 内側逐次化）**: 実測 +30%（97→127s）。ネスト par_iter の work-stealing がクラスタ間不均衡を
    吸収するため内側逐次化は末尾で遊休増→ revert。
  - **opt-in native ビルド**（`.cargo/config.toml.example`＋MANUAL§2.1）追加。本機 80 コアで dock(Direct)~5%
    （精度 −7/222 変化）・grid screen ~0%＋ビット一致（多コアで per-op ゲイン希釈・grid は f64）。
  結論: 既定 grid 探索は既に計算律速の床付近で、数値不変の安価な高速化は単発 dock の小幅（~3%）に留まる
  （SIMD_OPTIMIZE の再確認）。本質的向上は native（精度トレードオフ）か T3-1（子 objective の grid 化＝別計算機 DUD-E 検証）。
- 2026-06-22: 初期配置を **複数試行 best-start 選択**へ改良＋**残差の正体を診断特定**（`docs/findings.202606222119`）。
  Vina ポーズを `isct-dock score` でほぼ同値再採点でき＝**残差は scoring でなく ligand 依存の探索差**（小）。
  先頭 8 件（相関 subset）で isct 2/vina 6 が深い。全 kif11 では isct≈vina。残差追跡は多様セットで行うべき。
- 2026-06-22: hunt_cap を **intra（分子内ペア）にも適用**（Vina v[1] 相当, `docs/findings.202606222059`）。
  inter+intra 両対応で Vina 忠実化。ただし本セットでは追加効果ほぼ無し（埋め込み済みリガンドは intra 激突が稀）。
  実 Vina 残差 ~0.33 は依然＝次候補は初期ランダム化の多試行化。
- 2026-06-22: **hunt_cap（Vina curl：探索中の激突エネルギー上限）実装**（`docs/findings.202606222004`）。
  `SearchParams.hunt_cap`（既定 10.0、config `run.hunt_cap`）。探索 grid に per-atom 適用・polish は cap 無し。
  効果: vina-mode +0.14・drift 中立・速度不変・決定論維持。ただし実 Vina との残差 ~0.33 は未解消
  （初期ランダム化・intra curl・ステップ予算 等が候補）。vina-mode の変異振幅/温度も Vina 一致済（commit 325ee5d）。
- 2026-06-22: **速度差の切り分け実測**（`docs/results.202606221625`）。kif11 16 リガンド・8 コア:
  drift と vina-mode が同品質(−9.8)に到達するが drift は wall 2.05×・評価回数 3.0× 効率的＝**アルゴリズム差**。
  Rust vina-mode vs C++ Vina は wall 1.23×（品質を揃えれば ~1.0×）＝**実装差は小**。
  → **Vina に対する ~2.5× 高速化はほぼアルゴリズム（ドリフト鎖）由来**と確定。DIFF.md に反映。
- 2026-06-22: **Vina 忠実 MC 探索モード追加**（実装レベル差の切り分け用）。`SearchMode::Vina`／実行時
  `--search <standard|vina|qvina2>`（dock・screen 両対応）。vina-mode は各ステップ収束まで BFGS（drift は浅い
  chain_iters）。エネルギー評価回数カウンタを追加し dock 出力に表示（drift 870steps/143evals毎step vs
  vina 50steps/542evals毎step を確認）。決定論維持。clippy/test green。較正（実 Vina と品質を揃えた速度比＝
  実装差、vina-mode vs drift＝アルゴリズム差）は後続実験。
- 2026-06-22: ISCT-screen（scaffold 再利用）を kif11 で評価（`docs/results.202606220300`）。全カバー版
  ライブラリ実装（`build_library.py --full-coverage`）＋ engine 修正（`library.rs` 空コア・`screen.rs`
  auto-steps）。DUD-E 多様性で再利用 8% のみ＝利得限定の知見。全パターン×全指標の評価方針を確立。
- 2026-06-21: isct-dock の native 濃縮が低い原因＝ランキングに生 intermolecular を使い torsion 補正を露出
  していない報告上の問題と特定（`docs/findings.202606212224`）。CG0 対応で全 6964 完全カバー版を記録
  （`docs/results.202606212151`）。
- 2026-06-21: 大規模スクリーニング評価 第1弾（DUD-E kif11, ISCT-dock vs Vina vs Smina）を実施・記録
  （`docs/results.202606211249`, `docs/findings.{202606202120,202606211249}`）。ハーネス採点バグ是正。
- 2026-06-20: PROGRESS.md を PLAN.md から分離・新規作成。M1-M3 完了・M4 大部分実装を反映。QuickVina2・
  スクリーニング grid 再利用を追記。
- 2026-06-20: M3 残項目（screen 上位ポーズ出力・複数 seed ポーズ保持）を実装→ **M3 完了**に更新
  （`docs/results.202606201202`）。
- 2026-06-20: M4 残（core_affinity・memmap+バイナリ・ライブラリ streaming）を実装→ **M4 完了**に更新
  （`docs/results.202606201419`）。大規模テストは後日。
