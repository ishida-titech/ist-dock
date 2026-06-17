# ISCT-Dock 評価ドキュメント

ISCT-Dock の正しさ・精度・性能を検証するためのデータセットと評価方法をまとめる。
詳細な計画は [PLAN.md](PLAN.md)、データ一覧は [bench/data/README.md](bench/data/README.md)、
ツール導入手順は [bench/README.md](bench/README.md) を参照。

---

## 1. 評価方針

- **スコアの正しさは AutoDock Vina を基準にする**。スコアリング関数は Vina と完全互換に
  実装しているため、同一ポーズ・同一リガンドに対する**スコアの差（絶対値）**で検証する
  （順位相関ではない）。
- Vina の探索は確率的なため、**ドッキング結果のスコアは完全一致しない**。そこで検証は
  「固定ポーズの再採点（決定論的・厳密一致を要求）」と「ドッキング結果の比較（許容差を定義）」
  に分けて行う（多段評価。第 4 章）。
- 当面の評価環境は **Arm Mac + OS X**（Linux は後日）。

### 前提ツール（検証済みバージョン）

| ツール | バージョン | 用途 |
|---|---|---|
| AutoDock Vina | 1.2.7 | スコア/ドッキングの基準値 |
| RDKit | 2025.09.5 | リガンド 3D 化・結合次数付与 |
| Meeko | 0.7.1 | PDBQT 調製（`mk_prepare_*`） |
| gemmi | 0.7.5 | Meeko 依存 |

導入手順は [bench/README.md](bench/README.md) を参照。バージョンは Arm Mac/OS X で検証した値。

> **環境移行（2026-06-17）**: 開発環境を Arm Mac/OS X から **Linux workstation（x86_64）** へ移行。
> 導入は Linux でも同じ（conda-forge: `vina rdkit` / pip: `meeko gemmi`、Rust は rustup）。OS 差分・
> 移行手順は [PORTING.md](PORTING.md)。RMSD 等の数値は移行後に再測定推奨（アーキ差による微小な
> 揺れの可能性。検証ロジック・閾値は不変）。

---

## 2. 評価データセット

すべて `bench/` のスクリプトで再生成可能。詳細は [bench/data/README.md](bench/data/README.md)。

| 区分 | ケース | 系 | リガンド | 受容体 atoms | TORSDOF | 生成元 |
|---|---|---|---|---|---|---|
| Vina 配布 | `vina_examples/1iep` | Abl キナーゼ | イマチニブ | 2702 | 7 | `curl`（AutoDock-Vina repo） |
| 再ドッキング | `pdb/3ptb` | トリプシン | benzamidine | 2009 | 1 | `prepare_pdb.py` |
| 再ドッキング | `pdb/1stp` | ストレプトアビジン | biotin | 1112 | 6 | `prepare_pdb.py` |
| スクリーニング | `screening/trypsin` | トリプシン | active 4 / decoy 8計 | 2009 | — | `make_screening.py` |

- **再ドッキング用**: `ligand.pdbqt` は結晶（共結晶）ポーズ。再ドッキング RMSD の基準に使える。
- **スクリーニング用**: 1 受容体 × 複数リガンド。`labels.tsv` に active/decoy、`config.toml` に
  探索ボックス（結晶 BEN 重心）、`vina_ranking.tsv` に Vina ドッキングの参照ランキングを同梱。

### スクリーニングセットの内訳（`screening/trypsin`）

トリプシン S1 ポケットに結合しうる正電荷性分子を active、中性芳香環を decoy とした。

- active: benzamidine, 4-aminobenzamidine, benzylamine, 4-guanidinotoluene
- decoy : benzene, phenol, toluene, naphthalene

---

## 3. 評価指標

| 指標 | 定義 | 適用 |
|---|---|---|
| スコア差 ΔE | \|E(isct-dock) − E(vina)\| (kcal/mol) | スコア一致（固定ポーズ） |
| 合格閾値 | ΔE < 1e-2 kcal/mol（現実的ポーズ） | M1 ゲート |
| 再ドッキング RMSD | 予測ベストポーズ vs 結晶ポーズの重原子 RMSD | M2 以降 |
| ドッキング成功率 | RMSD < 2.0 Å の割合 | M2 以降 |
| エンリッチメント | active が上位に来る度合い（EF / AUC 等） | M3 以降 |

---

## 4. 評価方法（多段）

### Level 0 — 解析値による回帰テスト（Vina 非依存）

`crates/isct-io/tests/score_parity.rs`。単純な 2 原子配置のエネルギーを手計算値と照合する。
Vina が無くても CI で常時実行でき、スコア関数の数式・重み・半径・カットオフを固定する。

```sh
cargo test
```

### Level 1 — スコア一致（固定ポーズ, vs `vina --score_only`）【M1 ゲート】

同一の受容体・リガンド（結合ポーズ）に対し、`vina --score_only` の "Ligand - Receptor"
分子間エネルギーと `isct-dock score` を比較し、差 < 1e-2 を要求する。

```sh
python3 bench/run_vina_compare.py <receptor.pdbqt> <ligand.pdbqt>
```

### Level 2 — ドッキング結果の比較（M2 で実施）

同一リガンドを Vina と本エンジンで独立にドッキングし、ベストスコアの差を記録する。
さらに**相互再採点（cross re-score）**——我々のポーズを Vina で、Vina のポーズを我々で
採点——により、「スコア関数の差」と「探索能力の差」を切り分ける。完全一致は要求せず、
許容差を定義する。

### Level 3 — 再ドッキング精度（M2 で実施）

`pdb/*` の結晶ポーズを基準に、ドッキング予測ベストポーズとの RMSD を測り、成功率
（RMSD < 2.0 Å）を集計する。

### Level 4 — スクリーニング・エンリッチメント（M3 で実施）

`screening/trypsin` を用い、active が decoy より上位に来るかをエンリッチメント指標で評価する。
基準として Vina ランキング（`vina_ranking.tsv`）を併用する。

---

## 5. 現在の結果（M1: スコア完全一致）

### Level 0（解析値）

孤立炭素 2 個（r=4.0 Å）の分子間エネルギー = 手計算 −0.066113 と一致（`vina --score_only`
も −0.066）。repulsion・カットオフを含め回帰テスト化済み。**全テスト PASS**。

### Level 1（固定ポーズ, vs Vina）

| ケース | Vina (kcal/mol) | isct-dock | ΔE |
|---|---|---|---|
| 1iep（Abl + イマチニブ）| −17.634 | −17.639 | 0.0047 |
| 3ptb（トリプシン + benzamidine）| −6.145 | −6.146 | 0.0009 |
| 1stp（ストレプトアビジン + biotin）| −8.691 | −8.698 | 0.0069 |

**3/3 ケースで ΔE < 1e-2 → PASS**。実タンパク質（2702 原子規模）でも許容内で一致。

#### 既知の差異（衝突ポーズ）

原子が深く衝突する人工的な配置では repulsion(d²) が支配的になり、Vina の事前計算
スプライン補間 vs 当方の厳密計算の差で ~0.04 kcal/mol 程度ずれる。ドッキングでは衝突ポーズが
棄却されるため実害はない。完全一致が必要なら将来 Vina のスプライン離散化を再現する。

### スクリーニング参照ランキング（Vina ドッキング, `screening/trypsin`）

| 順位 | 分子 | ラベル | Vina best (kcal/mol) |
|---|---|---|---|
| 1 | 4-aminobenzamidine | active | −6.343 |
| 2 | 4-guanidinotoluene | active | −6.118 |
| 3 | benzamidine | active | −6.083 |
| 4 | naphthalene | decoy | −5.093 |
| 5 | benzylamine | active | −5.009 |
| 6 | phenol | decoy | −4.679 |
| 7 | toluene | decoy | −4.347 |
| 8 | benzene | decoy | −3.662 |

上位 3 が amidine 系 active で decoy と明確に分離（識別力のある妥当なケース）。
naphthalene が benzylamine をわずかに上回るのは Vina の疎水性バイアスによる現実的挙動。

---

## 5b. 現在の結果（M2: Vina 互換ドッキング）

ドッキングの全機構（torsion 木ポーズ再構成・Vina 互換採点・有限差分 BFGS 局所最適化・
Monte Carlo 大域探索・config/ポーズ I/O・RMSD）を実装。コンポーネントは単体テスト済み。
実装メモ: 計画の「解析的勾配＋L-BFGS」に対し初版は**有限差分勾配**、エネルギーは**分子間のみ**
（分子内は多 torsion 対応時に追加予定）。

### 局所最適化の検証（`--local-only`, 3ptb 結晶ポーズから）

結晶ポーズ（恒等変換）から局所最適化のみ → **−6.29 kcal/mol, RMSD 0.32 Å**。
ポーズ再構成・採点・最適化が end-to-end で正しいことを示す（局所最適化は強力）。

### 大域再ドッキング（Level 3, `run_redock.py`, 既定 n_runs=60/n_steps=60, seed=42）

探索の改善（1 自由度変異＋ボックス外ペナルティ＋接触開始の棄却サンプリング＋分子内エネルギー。
詳細は [FINDING.md](FINDING.md) 2026-06-17）により、**剛体〜柔軟リガンドまで全ケースで RMSD<2 Å**
を達成した。

| ケース | torsion | isct-dock RMSD (Å) | Vina RMSD (Å) | 判定 |
|---|---|---|---|---|
| 3ptb (benzamidine) | 1 | 1.97 | 1.13 | PASS |
| 1stp (biotin) | 6 | 1.20 | 0.66 | PASS |
| 1iep (imatinib) | 7 | 1.21 | 0.91 | PASS |

- **シード依存は exhaustiveness（本数増）で吸収**: 3ptb で n_runs=60 は 5/5 マスターシードが <2 Å。
  Vina も 1 run では外し exhaustiveness で吸収する設計と同じ。
- スコア列は直接比較しない（Vina は torsion 補正後 affinity、当方表示は素の分子間エネルギー）。
  検証指標は RMSD。スコア関数自体の一致は Level 1（score parity）で担保済み。
- 実行時間（60/60・有限差分）: 3ptb ~35 秒、1stp ~51 秒、1iep ~196 秒。高速化は M4 へ集約。

### 次の作業

1. ~~1 自由度変異 / ボックス外ペナルティ / 分子内エネルギー~~ — **実装・検証済み（2026-06-17）**。
2. **M4 に集約**: 解析的勾配＋L-BFGS・grid 補間・SIMD・スレッド化（throughput が本命。grid と
   一体設計するため M2 では行わない）。
3. ポーズの RMSD クラスタリング（複数モード出力）・Level 2（相互再採点）は M3/出力整備時。

> 現状: 機構完成・スコア一致（Level 1）検証済み・再ドッキング（Level 3）は剛体〜7 torsion の
> 全 3 ケースで RMSD<2 Å。M2 の正確性目標を達成。残る速度面は M4 に集約。

---

## 6. 再現手順

```sh
# 0. 前提ツール（未導入時。詳細は bench/README.md）
conda install -y -c conda-forge vina rdkit
python3 -m pip install --upgrade meeko gemmi

# 1. データ生成
python3 bench/prepare_pdb.py 3PTB BEN "NC(=N)c1ccccc1"
python3 bench/prepare_pdb.py 1STP BTN "OC(=O)CCCC[C@@H]1SC[C@@H]2NC(=O)N[C@H]12"
python3 bench/make_screening.py
#   1iep は bench/README.md の curl 手順で取得

# 2. ビルド & 回帰テスト（Level 0）
cargo test

# 3. スコア一致（Level 1）
python3 bench/run_vina_compare.py bench/data/vina_examples/1iep/1iep_receptor.pdbqt bench/data/vina_examples/1iep/1iep_ligand.pdbqt
python3 bench/run_vina_compare.py bench/data/pdb/3ptb/receptor.pdbqt bench/data/pdb/3ptb/ligand.pdbqt
python3 bench/run_vina_compare.py bench/data/pdb/1stp/receptor.pdbqt bench/data/pdb/1stp/ligand.pdbqt
```

---

## 7. 今後の評価計画（マイルストーン連動）

| マイルストーン | 追加する評価 |
|---|---|
| M2（Vina 互換ドッキング）| Level 2（ドッキング比較・相互再採点）、Level 3（再ドッキング RMSD） |
| M3（クラスタリング高速化）| Level 4（エンリッチメント）、Scaffold 拘束 vs 独立ドッキングの精度差、計算削減率 |
| M4（単一ノード最適化）| スループット（分子数/秒）、最適化前後のスコア不変回帰 |
| M5（動的ボックス等）| 追加高速化の限界効果（スループット改善・精度不変） |
| M6（精度向上）| アンサンブル/Soft-core/柔軟側鎖 追加前後の RMSD・スコア改善 |

データセットは必要に応じて拡張する（再ドッキング複合体の追加、decoy セットの増強など）。
