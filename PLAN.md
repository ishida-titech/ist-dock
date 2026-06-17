# ISCT-Dock 開発計画

`isct-dock.md`（仕様書）の実装計画。実装言語は Rust。

> **関連文書**: 評価結果＝[EVALUATION.md](EVALUATION.md) / 調査ノート（時系列）＝[FINDING.md](FINDING.md)
> / 環境移行・引継ぎ＝[PORTING.md](PORTING.md) / プロジェクト常用コンテキスト＝[CLAUDE.md](CLAUDE.md)。

---

## 進捗状況（2026-06-17 時点）

| マイルストーン | 状態 | 要点 |
|---|---|---|
| M1 Vina スコア一致 | **完了** | `score` 実装。解析値＋実複合体3件で `vina --score_only` と差<1e-2（現実的ポーズ）。 |
| M2 Vina 互換ドッキング | **完了（正確性）** | `dock` 実装。1自由度変異＋ボックス罰則＋接触開始＋分子内エネルギー。再ドッキング RMSD<2Å を 3ptb/1stp/1iep で達成。解析的勾配は M4 送り。 |
| M3 クラスタリング高速化 | **検証データ設計まで** | DUD-E 5標的取得・特性化済み。クラスタリング＝FP+MCS（active/decoy 統合）、密検証は ampc 予定。**前処理パイプライン・Phase1-3 は未実装**。 |
| M4 単一ノード最適化 | 未着手 | 解析的勾配・grid・SIMD・スレッド化・mmap を Linux で集約実装。 |
| M5/M6 | 未着手 | 動的ボックス / アンサンブル・Soft-core・柔軟側鎖。 |

詳細な達成内容・既知の課題は EVALUATION.md / FINDING.md を参照。

---

## Context（なぜ作るのか / 何を作るのか）

`isct-dock.md` に記された「超大規模・高速分子ドッキングスクリーニングエンジン
（ISCT-Dock）」をゼロから実装する。仕様全体は (1) Vina 同等のドッキング中核、
(2) Scaffold 流用＋クラスタ一括棄却という独自アルゴリズム、(3) 単一ノード性能最適化、
(4) 分散 Work-Stealing の4大サブシステムを含み、一括実装は破綻する。

本開発は以下の**順序**で進める（後段ほどスコープが先送り）:

1. **Vina の Rust 実装**: Vina 互換スコア＋単一受容体ドッキング（剛体＋torsion）を作り切る。
2. **リガンドクラスタリング高速化の効果検証**: Scaffold 流用＋クラスタ一括棄却が
   「精度を保ったまま計算を削減できるか」を、最適化前の素直な実装で検証する。
3. **単一ノード最適化**: SIMD・ゼロアロケーション・スレッド並列・grid 補間・mmap 等を適用。
4. **追加高速化アイデアの検証**: 動的バウンディングボックス（PCA 身の丈ボックス等）を
   最適化済みベースライン上で導入し、効果を測定。
5. **精度向上機能の追加・検証**: マルチ受容体アンサンブルドッキング、Soft-core ポテンシャル
   （擬似 induced-fit）、重要側鎖の同時局所最適化（柔軟残基）を追加し、精度改善を検証。

**分散（複数ノード / Work-Stealing / gRPC）は今回の仕様から除外**（最後に別計画で）。

> 設計上の注意: 4 の最適化（M4）後に 5・6 の新機能を足すと内側ループ（特に Soft-core・
> 柔軟側鎖はスコア計算）が変化し再最適化が要りうる。M4 の基盤（再利用バッファ・SIMD
> カーネル・grid アクセサ）は後続機能を差し込める拡張点を残す設計とし、「最適化済み
> ベースライン上で各アイデアの限界効果を測る」という意図を満たす。

### 確定した重要決定

- **スコアリング関数は AutoDock Vina と完全に同一**にする（関数形・重み・原子型・
  半径まで一致させる）。
- **検証は Vina との「スコアの差」**で行う（順位相関ではなく、同一ポーズ/同一リガンドに
  対する絶対値の差）。
- **前処理（SDF→バイナリ変換、MCS/Scaffold クラスタリング）は既存ツール（RDKit/
  Meeko 等）で実施**。エンジンはバイナリ/PDBQT の読込のみ。変換スクリプトは薄く用意。
- 実装言語は **Rust**（仕様準拠）。
- **実行環境**: 本番実行環境は **Linux（x86_64）**。**2026-06-17 に開発環境を Arm Mac/OS X から
  Linux workstation へ移行**（移行手順・差分は [PORTING.md](PORTING.md)）。M1〜M3 のコードは純
  Rust で OS 非依存（Linux で無改変ビルド可）。M4 以降の OS 依存最適化は **Linux を第一級**で実装:
  - **SIMD**: Linux x86_64 では AVX2/AVX-512（`core::simd` または `wide`）。Mac の aarch64 NEON は
    cfg 分岐で両対応。
  - **`core_affinity`（コアピン留め）**: Linux で有効（Apple Silicon macOS は実質非対応で no-op だった）。
  - **`madvise(MADV_WILLNEED)`**: Linux で本来の挙動。`memmap2` プリフェッチを Linux 前提で実装。
  → 移行により M4 の OS 依存機能が本来の形でテスト可能になる（むしろ好都合）。
- **設定ファイルは TOML** を使用（受容体メタデータ＝ポケット中心・柔軟残基指定、実行パラメータ）。
  `serde` + `toml` でパース。

### 前提環境（要セットアップ — 現状未導入）

- `cargo` 1.93.1 ✓ 導入済み
- **AutoDock Vina**: 検証の基準値（`vina --score_only` 等）として必須。未導入 →
  `brew install autodock-vina` または conda で導入する。
- **RDKit + Meeko**: 前処理（PDBQT 化・MCS クラスタリング）に必要。未導入 →
  `conda install -c conda-forge rdkit meeko` で導入。

---

## 仕様の精緻化（実装に落とすための確定事項）

### スコアリング関数（Vina 完全互換）

Trott & Olson (2010) の Vina scoring function を厳密に再現する。原子ペア i,j について
表面間距離 `d = r_ij − (R_i + R_j)`、カットオフ `r_ij < 8 Å`:

| 項 | 関数形 | 重み w |
|---|---|---|
| gauss1 | `exp(−(d/0.5)²)` | −0.0356 |
| gauss2 | `exp(−((d−3)/2)²)` | −0.00516 |
| repulsion | `d<0 ? d² : 0` | +0.840 |
| hydrophobic | 区分線形（d<0.5:1 → d>1.5:0）、疎水-疎水のみ | −0.0351 |
| hbond | 区分線形（d<−0.7:1 → d=0:0）、ドナー-アクセプタのみ | −0.587 |

- 構造補正: 最終スコア = `e_inter / (1 + w_rot · N_rot)`、`w_rot = 0.0585`。
- 原子型は Vina の XS 原子型（C_H/C_P/N_D/N_A/O_DA/… 等）と各型の VdW 半径を再現。
- 分子内項（intra）のペア包含規則（結合グラフ上の近接ペア除外、剛体グループ内除外等）は
  **Vina 実装 `model::eval_*` を参照**して一致させる（誤実装回避のため文献値で過剰確定しない）。
- 検証の一次ゲートは `vina --score_only`（固定ポーズ）との一致。これは決定論的で、
  我々の `score` モードとの差が ~1e-2 kcal/mol 未満であることを要求する。
- ライセンス: Vina は Apache-2.0。関数形・重みは公開。ソースは参照のみ（コードコピーはしない）。

### 入出力（検証フェーズ）

- **受容体**: PDBQT を直接読む（Vina と同一の原子型・電荷を保証するため）。
  ポケット中心・探索ボックス・（後段の）柔軟残基指定・アンサンブル構造リスト等のメタデータは
  **TOML 設定ファイル**で受ける（例: `config.toml`）。
- **リガンド（検証初期）**: PDBQT を直接読む（Vina と同条件で比較するため）。
- **クラスタライブラリ（Milestone 3〜）**: 既存ツール（RDKit/Meeko + 薄い Python 変換
  スクリプト）が生成するカスタムバイナリを読む。スキーマは仕様 4.1 の
  `Cluster_ID / Target_Pocket_Constraint / Seed_Molecule / Children_Molecules（コア差分）`。
- **出力**: スクリーニングサマリ（ログ）、ランキング TSV（`Rank, Compound_ID, Cluster_ID,
  Total_Score, VdW_Energy, Electrostatic_Energy, Target_Protein_State_ID`）、上位ポーズの
  バイナリ（SDF/PDBQT 逆変換可）。

### 「決定論的」の扱い

Vina の探索は確率的（Monte Carlo）。本エンジンは**固定シードで決定論的**に動かす
（仕様の「決定論的」を満たす）。ただし Vina とは探索経路が異なるため、**ドッキング結果の
スコアは完全一致しない**。よって検証は二段構え（下記）にする。

---

## 開発マイルストーン

各マイルストーンは「動く成果物」を残す。順序は固定（M1〜M2=Vina実装、M3=クラスタ高速化
検証、M4=最適化、M5=追加高速化検証、M6=精度向上）。アンサンブル・Soft-core・柔軟側鎖・
動的バウンディングボックスは初期スコープから外し、後段へ送る。

### Milestone 1 — Vina スコア完全一致（score-only）【最重要ゲート】

- PDBQT パーサ（受容体・リガンド）と Vina XS 原子型付与。
- Vina スコアリング関数（inter + intra）を直接原子間計算で実装。
- `isct-dock score <receptor.pdbqt> <ligand.pdbqt>` モード（固定ポーズの評価）。
- **検証**: 複数複合体で `vina --score_only` の出力（Affinity と項別内訳）と比較し、
  差 < 1e-2 kcal/mol。これが合格するまで先に進まない。

主要ファイル: `crates/isct-core/src/{atom_types,scoring,geometry}.rs`,
`crates/isct-io/src/pdbqt.rs`, `crates/isct-cli`.

### Milestone 2 — Vina 互換ドッキング（単一受容体・単一リガンド）

「Vina の Rust 実装」を完成させる段階。アンサンブルはここでは扱わない（精度向上の M6）。

- リガンドのトポロジー表現（root + 可回転結合のツリー、torsion DOF）。
- 自由度: 剛体並進(3)＋回転(quaternion)＋全 torsion。単一受容体ボックスへドッキング。
- スコアの解析的勾配を実装（L-BFGS 用）。
- 探索: Vina 同様の Monte Carlo（反復局所探索）＋ L-BFGS 局所最適化。固定シードで決定論的。
- グリッドは使わず**直接原子間計算**（最適化前なので素直さ優先。grid は M4 で追加）。
- **検証（二段構え）**:
  1. *スコア一致*: 我々が出したポーズを `vina --score_only` で再採点し差を確認。
     逆に Vina のベストポーズを我々の `score` で採点し差を確認（スコア関数の正しさ）。
  2. *ドッキング比較*: 同一リガンドを Vina と本エンジンで独立ドッキングし、ベストスコアの
     差を記録（探索能力の検証。完全一致は要求しない、許容差を定義）。

主要ファイル: `crates/isct-dock/src/{ligand,search,optimizer,gradient}.rs`.

### Milestone 3 — リガンドクラスタリング高速化の効果検証（ISCT-Dock 中核）

Scaffold 流用＋クラスタ一括棄却の「効果」を検証する。**動的バウンディングボックスは含めず
（M5）、Soft-core・柔軟側鎖も含めない（M6）**。単一受容体で評価。

- クラスタデータ構造（Seed + Children、Scaffold=MCS）。既存ツール生成のバイナリを読込
  （`crates/isct-io/src/library.rs`）。RDKit による生成スクリプトは `tools/preprocess/` に薄く用意。
- **Phase 1**: 代表分子を（単一受容体へ）グローバルドッキング → 基準未達クラスタを一括棄却
  → ベストポーズ座標を確定（複数ポーズは RMSD クラスタリングでデデュープ）。
- **Phase 2（最小版）**: 子分子へ共通コア座標をロック流用。探索窓は当面**固定/単純マージン**
  （PCA 身の丈ボックスは M5）。
- **Phase 3（最小版）**: R 基の torsion のみ L-BFGS で最適化。共通コアのエネルギーはキャッシュし
  差分(Delta)のみ追加計算（Soft-core・柔軟側鎖はここでは無効）。
- **検証（アイデアの是非）**:
  - *精度*: Scaffold 拘束で得たスコアを、(a) 各子分子を独立フルドッキング、(b) Vina、と比較し
    スコア差を評価（拘束による劣化が許容範囲か）。
  - *効率*: クラスタ棄却率と、ナイーブ（全分子独立ドッキング）比の計算削減率を測定。

主要ファイル: `crates/isct-dock/src/phases/{phase1,phase2,phase3}.rs`,
`crates/isct-core/src/delta_energy.rs`.

### Milestone 4 — 単一ノード最適化

M3 で「アイデアが有効」と確認できてから着手。各項目はベンチで効果測定しつつ段階導入。
後続 M5/M6 の機能を差し込めるよう、バッファ・カーネル・grid アクセサに拡張点を残す。
**検証は当面 Arm Mac/OS X 上で行う**（OS 依存箇所は cfg 分岐、Linux は将来対応）。

- グリッド補間: Coarse/Fine 多解像度 grid と trilinear 補間で原子間計算を置換。
- SIMD: Mac は `aarch64` NEON（`core::simd` または `wide`）で距離計算・補間・座標更新を一括化。
- ゼロアロケーション: スレッド毎の再利用固定長バッファ（Clear & Reuse）でループ内 heap 確保を排除。
- スレッド並列: `rayon`（または std thread）でクラスタ/分子をデータ並列化。
- CPU アフィニティ: `core_affinity`（Mac はベストエフォート/no-op、Linux で物理コアピン留め）。
- I/O: `memmap2` での読込（`madvise(MADV_WILLNEED)` プリフェッチは cfg で Linux 向けに出し分け）。
- **検証**: 分子数/秒スループットの推移、最適化前後でスコアが不変（回帰テスト）。

### Milestone 5 — 追加高速化アイデアの検証（動的バウンディングボックス等）

最適化済みベースライン上で、追加の探索空間削減アイデアを導入し限界効果を測る。

- 動的バウンディングボックス（Dynamic Bounding Box）: 子分子ごとに PCA で非対称（直方体）な
  探索窓をインデックス範囲として算出。可回転結合数連動でマージン自動調整。窓外ポーズは即棄却。
- 二面角サンプリング優先度制御: コアに近い結合から優先最適化。
- **検証**: M4 ベースライン比のスループット改善と、スコアへの影響（精度劣化がないか）。

主要ファイル: `crates/isct-core/src/pca.rs`, `crates/isct-dock/src/phases/phase2.rs`.

### Milestone 6 — 精度向上機能の追加・検証

効果確認後、精度に効く機能を追加し検証する。

- マルチ受容体アンサンブルドッキング: M 個（3 構造程度）の Open/Closed 等へ Phase 1 を拡張し、
  勝ち受容体構造（`Target_Protein_State_ID`）を選別。
- Soft-core ポテンシャル: 初期ステップで VdW 反発の傾きを緩め擬似 induced-fit。
- 重要側鎖の同時局所最適化: 指定柔軟残基(1〜2)の側鎖二面角と R 基二面角を同時に L-BFGS。
- **検証**: 再ドッキング/既知結合データに対する精度（スコア・ポーズ）改善を、追加前後で比較。

主要ファイル: `crates/isct-dock/src/phases/{phase1,phase3}.rs`,
`crates/isct-core/src/softcore.rs`, `crates/isct-dock/src/flexible_residue.rs`.

### （スコープ外）複数ノード分散

gRPC/TCP の Work-Stealing は今回除外。将来必要になれば M6 完了後に別計画で。

---

## プロジェクト構成（Cargo ワークスペース）

```
isct-dock/
├─ PLAN.md                      # 本計画（成果物）
├─ Cargo.toml                   # workspace
├─ crates/
│  ├─ isct-core/                # 原子型・スコア・幾何・PCA・勾配（純計算、no_std 寄り）
│  ├─ isct-io/                  # PDBQT パーサ、カスタムバイナリ I/O
│  ├─ isct-dock/                # 探索・最適化・Phase1-3 エンジン
│  └─ isct-cli/                 # `isct-dock` 実行バイナリ（score / dock / screen）
├─ tools/preprocess/            # RDKit/Meeko 薄い前処理スクリプト（Python）
├─ bench/                       # Vina 比較検証ハーネス（Python + テストデータ）
│  └─ data/                     # 再ドッキング用の小規模複合体セット
└─ tests/                       # 統合テスト
```

### 採用クレート（暫定）

- 幾何/線形代数: `nalgebra`（PCA/SVD/四元数）または軽量に `glam`。
- 最適化: まず `argmin`(LBFGS) で素早く、M4 でゼロアロケーション版に置換検討。
- 乱数: `rand`（seedable, 決定論的）。
- 並列(M4): `rayon`, `core_affinity`。 I/O(M4): `memmap2`。 SIMD(M4): `core::simd` or `wide`。
- エラー: `anyhow`(cli) / `thiserror`(lib)。CLI: `clap`。設定: `serde` + `toml`。

---

## 検証（エンドツーエンド）

1. **前提セットアップ**: `brew install autodock-vina`、`conda install -c conda-forge rdkit meeko`。
   全フェーズの検証は当面 Arm Mac + OS X 上で実施（Linux 検証は後日）。
2. **単体テスト**: `cargo test` — 原子型付与、各スコア項、勾配（数値微分との一致）、PCA、PDBQT パース。
3. **スコア一致テスト（M1 ゲート）**: `bench/` の複合体群で
   `vina --score_only` vs `isct-dock score` を比較、差 < 1e-2 kcal/mol を assert。
4. **ドッキング比較（M2）**: 同一リガンドを Vina と本エンジンで実行。ベストスコア差を記録し、
   相互再採点（cross re-score）でスコア関数 vs 探索能力を切り分け。
5. **クラスタ高速化検証（M3）**: Scaffold 拘束 vs 独立フルドッキング/Vina のスコア差、
   クラスタ棄却率、ナイーブ比の計算削減率をレポート出力。
6. **性能回帰（M4）**: 最適化適用前後でスコア不変を確認しつつ、分子数/秒を測定。
7. **追加高速化検証（M5）**: 動的バウンディングボックス等の導入前後で、スループット改善と
   スコア不変（精度劣化なし）を確認。
8. **精度検証（M6）**: アンサンブル/Soft-core/柔軟側鎖の追加前後で、再ドッキング精度
   （スコア・ポーズ）の改善を比較。

検証ハーネスは `bench/run_vina_compare.py`（Vina 実行＋差分集計）として用意する。

---

## 最初の実行ステップ

1. `cargo` ワークスペース雛形（上記構成の空クレート群）を作成。
2. M1 着手: PDBQT パーサ → 原子型 → スコア関数 → `score` サブコマンド。
3. 並行して AutoDock Vina / RDKit のセットアップ手順を `bench/README` に記載。
