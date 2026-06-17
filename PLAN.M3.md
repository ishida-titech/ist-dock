# PLAN.M3 — Milestone 3 詳細計画

`PLAN.md` の M3 節を補完する詳細計画。実装はこのドキュメントをチェックリストとして進める。
関連: 仕様＝[isct-dock.md](isct-dock.md) / 全体計画＝[PLAN.md](PLAN.md) / 調査ノート＝[docs/FINDING.md](docs/FINDING.md)。

---

## 1. 目的

`isct-dock.md` の中核「**Scaffold 流用 ＋ クラスタ一括棄却**」が *精度を保ったまま計算を削減
できるか* を、最適化前の素直な実装で実測検証する（M4 性能最適化 / M5 動的ボックス / M6 精度向上
は含めない。単一受容体）。

FINDING.md（2026-06-17）の結論が前提：
- **計算削減は主に「decoy の大クラスタを seed で一括棄却」から来る**（active は骨格多様で寄与小）。
- 測るべき主曲線は **クラスタリング Tanimoto しきい値による「速度削減 ↔ 偽陰性（active 取りこぼし）」**。
- 密検証ターゲットは **ampc**（最小・active/decoy 双方にクラスタ・酵素活性部位が明確）。

## 2. 設計判断（確定）

1. **薄い縦スライス先行**: まず 1 クラスタで E2E（seed dock → 子の核固定 dock → Δエネルギー）を通し、
   正しさを固めてから ampc 全体へスケール。
2. **可読な中間形式**: ライブラリ＝「PDBQT 群 ＋ TOML マニフェスト」。仕様 4.1 のカスタムバイナリ化は
   M4 の mmap ストリーミング対応とまとめて実装（新規依存ゼロ：既存 `serde`/`toml`）。
3. **前処理で拘束埋込**: RDKit constrained embed で各子の MCS コアを seed 参照配座に重ねて 3D 生成し、
   **子 PDBQT をコア＝ROOT でルート化**。エンジンは「固定コア＋R 基 torsion 木」を読むだけ。

---

## 3. 検証に必要なデータ（具体）— 本計画の主眼

### 3.1 現状あるもの（`bench/data/dude/<target>/`、全 5 標的同型）
| ファイル | 内容 | ampc の量 |
|---|---|---|
| `receptor.pdb` | 受容体（**PDBQT 未変換**） | 1（194 KB） |
| `crystal_ligand.mol2` | 結晶リガンド（box 中心算出用） | 34 原子 |
| `actives_final.ism` | active の SMILES＋ID | **48** |
| `decoys_final.ism` | decoy の SMILES＋ID | **2850** |

ampc 合計 **2898 分子**。active/decoy ラベルは由来ファイルから既知。

### 3.2 前処理で生成が必要なもの
| データ | 生成法（既存スクリプト流用先） | ampc の量 |
|---|---|---|
| `receptor.pdbqt` | `mk_prepare_receptor`（`bench/prepare_pdb.py` の受容体 prep を流用） | 1 |
| box（center / size） | `crystal_ligand.mol2` 重心 → `config.toml`（`make_screening.py:box_center_from_reference` 流用） | 1 |
| 全リガンド 3D PDBQT | SMILES→`AddHs`→`EmbedMolecule`→`MMFFOptimize`→`mk_prepare_ligand`（`make_screening.py:ligand_pdbqt` 流用） | 2898 |
| クラスタリング結果 | Morgan FP(r=2, 2048bit)+Tanimoto 前ふるい → 閾値超ペアに MCS（`FindMCS`）→ 共通コア十分大で同クラスタ化・seed 選定（`analyze_dude_clusters.py` を拡張） | active+decoy 統合 |
| `library.toml` ＋ 子の拘束 PDBQT | 各子を constrained embed（コア＝seed コア配座に拘束）→ **コア＝ROOT で PDBQT 化** ＋ `core_atoms` マップ | seed 数＋子数 |
| `labels.tsv` | `compound_id, label(active/decoy), cluster_id` | 2898 行 |

出力先: `bench/data/library/ampc/`。
ツール所在: rust=`~/.cargo`、vina/rdkit/meeko/gemmi=`~/miniconda3`（system python3 は rdkit 無し）。

### 3.3 計算量の現実性スコープ（曖昧さの核心 — 明示が必須）
M2 の有限差分ドッキングは 1 分子 **~30–200 秒**（3ptb 35s / 1stp 51s / 1iep 196s）。
**2898 分子の独立フルドッキング（ナイーブ基準）は数十〜百時間規模で非現実的**。
よって検証は計算対象を以下にスコープする：

| 検証量 | 対象 | 規模（ampc 目安） | コスト |
|---|---|---|---|
| クラスタ統計 | 全分子（SMILES のみ） | 2898 | 秒オーダー（全数） |
| seed ドッキング（フル） | クラスタ数ぶん | 数百（≪2898） | 数時間（全数、Phase1 で必要） |
| 子の拘束最適化（Phase3, torsion+Δ のみ） | 生存クラスタの子 | 数百〜千 | 速い（全数可） |
| ナイーブ基準（独立フルドッキング） | **サンプル 100–300 分子** | サンプルのみ | 1 分子コストを実測 → 総コスト外挿 |
| 精度比較（拘束 vs 独立 vs Vina） | **サンプル 50–100 分子**（active/decoy 混在） | サンプルのみ | 中 |

→ 「速度削減 ↔ 偽陰性」主曲線・精度・速度を、現実的計算量で取得する。
（フルセットの絶対スループットは M4 の grid/SIMD/スレッド化後に別途測る前提。）

---

## 4. 役割分担とファイル

| 層 | 成果物 | 状態 |
|---|---|---|
| 前処理（Python） | `tools/preprocess/{prepare_receptor,cluster_library,build_library}.py` | 現状 README のみ → 新規 |
| ライブラリ I/O（Rust） | `crates/isct-io/src/library.rs` | 現状 TODO → 実装 |
| Δエネルギー（Rust） | `crates/isct-core/src/delta_energy.rs` | 未存在 → 新規 |
| 幾何（Rust） | `crates/isct-core/src/geometry.rs` に `superpose`（Kabsch/SVD, `nalgebra`） | 拡張 |
| Phase 1〜3（Rust） | `crates/isct-dock/src/phases/{phase1,phase2,phase3}.rs` | 現状 TODO → 実装 |
| オーケストレーション | `crates/isct-dock/src/screen.rs` ＋ `crates/isct-cli/src/main.rs` の `Command::Screen` | screen 新規・CLI スタブ実装 |
| 設定 | `crates/isct-io/src/config.rs` に screen 用（library パス・prune 閾値・出力先）追加 | 拡張 |
| 検証（Python） | `bench/run_cluster_screen.py`（閾値スイープ・精度比較・速度集計） | 新規 |

### 再利用する既存 API（新規実装しない）
- `search::dock(lig, scorer, &SearchParams) -> DockResult`（Phase 1）。
- `LigandModel::{new, world_coords, scored_atoms, intramolecular}`, `Conformation`（コア固定＋torsion DOF をそのまま表現）。
- `energy::Scorer::{new, intermolecular, box_penalty, score}`。
- `optimizer::minimize`（Phase 3 の BFGS コア；torsion 制限版を派生）。
- `isct_core::scoring::{eval_pair, intermolecular_terms, apply_torsion_correction, Terms}`。
- `isct_io::pdbqt::{parse_file, Molecule, write_pose}`、`geometry::{distance, rmsd}`。
- Python: `make_screening.py:{ligand_pdbqt, box_center_from_reference}`, `analyze_dude_clusters.py:{read_ism, murcko}`, `prepare_pdb.py` 受容体 prep。

---

## 5. 実装（アルゴリズム要点）

### 5.1 ライブラリ・マニフェスト（`library.rs`、TOML）
```toml
[[cluster]]
id = "ampc_c0"
pocket_center = [x, y, z]   # 省略時は run config の box
pocket_size   = [sx, sy, sz]
seed = { compound_id = "...", pdbqt = "seed.pdbqt" }
[[cluster.child]]
compound_id = "..."
pdbqt = "child_0.pdbqt"     # コア=ROOT でルート化済み
core_atoms = [0,1,2,...]    # 子内 MCS コア原子インデックス（seed コアと 1:1 対応）
```
返す型: `Library{clusters}`, `Cluster{id, pocket, seed, children}`。各 PDBQT は `pdbqt::parse_file`。

### 5.2 Phase 1（`phase1.rs`）— seed ドッキング＋一括棄却
- seed を `search::dock` で単一受容体へ → `DockResult{conf, energy}`。
- **棄却判定**: `energy > prune_threshold`（config 可変）なら当クラスタの子を全スキップ。
- ベスト 1 ポーズ採用（複数ポーズ RMSD デデュープ＝`geometry::rmsd` は任意拡張、後置）。
- seed の docked コア世界座標を Phase 2 へ引継ぐ。

### 5.3 Phase 2（`phase2.rs`）— 子の核固定配置
- 子の参照コアを **docked seed コアへ剛体重ね合わせ**（対応＝`core_atoms`、`superpose`=Kabsch/SVD）。
  constrained embed によりコア RMSD≈0 で良条件。得た剛体変換を `Conformation`(pos/orient)、torsions は embed 値で初期化。
- コア（root）を固定し、R 基 torsion のみ自由 DOF（既存 `LigandModel` の root＋torsion 木をそのまま）。

### 5.4 Δエネルギー（`delta_energy.rs`）＋ Phase 3（`phase3.rs`）
- 子 scored 原子を **コア C（固定）/ R 基 G** に分割。`E_inter(child)=E_inter(C,recep)+E_inter(G,recep)`。
- `E_inter(C,recep)` は同クラスタ固定ポーズで全子共通かつ最適化中も不変 → **seed 確定時に 1 回計算しキャッシュ**。
- 各ステップは **G の対受容体項＋ G の分子内項（C×G, G×G）のみ再計算**＝Delta（`eval_pair`/`Terms` 再利用）。
- **単体テスト**: Δ評価 ＝ フル再計算（`Scorer::intermolecular`+`ligand::intramolecular`）と一致（~1e-9）。
- Phase 3: 既存 BFGS を **R 基 torsion DOF のみに制限**して最小化（pos/orient ロック、有限差分勾配は M3 のまま）。
- 最終スコア: `apply_torsion_correction(e_inter, n_rot)` で Vina 互換 affinity。

### 5.5 `screen` オーケストレーション
- ライブラリ走査 → クラスタごとに Phase1（棄却 or 通過）→ Phase2 → Phase3 → スコア収集。
- 出力 1: **ランキング TSV**（仕様 4.2: `Rank, Compound_ID, Cluster_ID, Total_Score, VdW_Energy,
  Electrostatic_Energy, Target_Protein_State_ID`）。**Vina 関数に静電項は無い**ため `Electrostatic_Energy`=0/該当なし
  として出力しその旨記録、`Target_Protein_State_ID` は単一受容体で固定。
- 出力 2: **サマリログ**（クラスタ生存率・棄却クラスタ数・スキップ分子数・スループット）。

---

## 6. ステップ 1 — 薄い縦スライス（最初の動く成果物）

ampc actives の明確な MCS 系列（例: 先頭 3 件 `S(Nc1...C(=O)O)(c2...scc2 C(=O)O)(=O)=O` …＝
アリールスルホニル–チオフェンカルボン酸骨格）を **seed＋子 2〜3 個**として手動/スクリプトで用意。
- 用意: ampc receptor.pdbqt、box（crystal 重心）、seed＋子の拘束 PDBQT＋`library_mini.toml`。
- 実装最小経路: `library.rs` → Phase1 → `superpose`/Phase2 → `delta_energy`/Phase3 → `screen`/CLI。
- **合格条件**: ① Δエネルギー単体テスト一致、② 子の Scaffold 拘束スコア ≈ 子の独立フルドッキング
  スコア（許容差を記録）。満たしてからステップ 2 へ。

## 7. ステップ 2 — ampc 密検証

### 7.1 前処理（`tools/preprocess/`）
- `prepare_receptor.py`: ampc `receptor.pdb`→`mk_prepare_receptor`、box=crystal 重心。
- `cluster_library.py`: **active+decoy 統合** Morgan FP→Tanimoto 前ふるい→閾値超ペアに `FindMCS`→
  共通コア十分大で同クラスタ化→seed 選定。**Tanimoto 閾値はパラメータ**（主曲線で振る）。
- `build_library.py`: seed 3D＋PDBQT、各子 constrained embed（コア拘束）→コア=ROOT で PDBQT→
  `library.toml`＋`core_atoms`＋`labels.tsv`。

### 7.2 検証プロトコル（`bench/run_cluster_screen.py`）
- **効率/主曲線**: 全クラスタ seed をドッキング（安価側）→ **prune 閾値**と**Tanimoto 閾値**を振り、
  各点で (i) スキップ子数＝計算削減率、(ii) active を含むクラスタが棄却された率＝偽陰性率 を集計し、
  削減↔偽陰性のトレードオフ曲線を出力。
- **精度**: サンプル 50–100 子で 拘束スコア vs (a) 独立フルドッキング(`search::dock`) vs (b) `vina`/`--score_only`
  のスコア差分布。
- **速度**: 拘束パイプライン総コスト（seed dock＋子 Phase3）vs ナイーブ（サンプル独立フルドッキングから外挿）
  の分子数/秒比。
- レポート: `docs/results.YYYYMMDDHHMM.md`（新規・上書き禁止）。新事実は `docs/findings.YYYYMMDDHHMM.md`。

---

## 8. 完了条件（M3 ゲート）

1. 縦スライス E2E が通り、子拘束スコア ≈ 独立スコア（許容差明文化）。
2. ampc で「削減率↔偽陰性」主曲線、精度分布、速度比を取得し `docs/results.*` に記録。
3. `cargo test`（library パース・`superpose`・`delta_energy` 一致）・`cargo clippy --all-targets` 警告ゼロ。
4. 「Scaffold 流用＋一括棄却が精度劣化許容内で計算削減に有効か」の結論を findings に記録。

## 9. 検証（エンドツーエンド手順）

1. `cargo test` / `cargo clippy --all-targets`。
2. 前処理: `python tools/preprocess/{prepare_receptor,cluster_library,build_library}.py`（ampc）。
3. 縦スライス: `isct-dock screen <ampc_receptor.pdbqt> bench/data/library/ampc_mini/library.toml --config run.toml`。
4. 密検証: `python bench/run_cluster_screen.py ampc`（閾値スイープ・精度・速度）。
5. 留意: 新規依存を増やさない（TOML=`serde`/`toml`、Kabsch=`nalgebra`、CLI=`clap`）。
