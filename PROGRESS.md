# PROGRESS.md — ISCT-Dock 進捗

マイルストーン（[PLAN.md](PLAN.md)）の実装・検証状況を**随時更新する living document**。
（`docs/results.*`・`docs/findings.*` は一度作成したら更新しない時系列ログ。**本ファイルは現状を反映して
上書き・追記する**。新機能・検証を入れたらこの表と詳細・更新履歴を更新すること。）

最終更新: 2026-06-22

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
