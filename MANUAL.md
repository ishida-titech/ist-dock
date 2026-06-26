# MANUAL.md — ISCT-Dock 利用マニュアル（現時点）

ISCT-Dock は、機械学習を使わず幾何・物理化学計算で行う高速・決定論的な分子ドッキング/スクリーニング
エンジン（Rust 実装）。AutoDock Vina 互換のスコア関数を用いる。本書は**現時点で動作する機能**の使い方を示す。
設計仕様は [isct-dock.md](isct-dock.md)、開発計画は [PLAN.md](PLAN.md) を参照。

---

## 1. 概要・現状

| サブコマンド | マイルストーン | 機能 | 状態 |
|---|---|---|---|
| `score`  | M1 | 固定ポーズの Vina 互換スコア評価 | 動作（`vina --score_only` と <1e-2 一致） |
| `dock`   | M2 | 単一リガンドのドッキング（剛体6＋torsion 大域探索） | 動作 |
| `screen` | M3 | クラスタライブラリの一括スクリーニング（scaffold 再利用） | 動作（基本機能） |
| `pack`   | M4 | クラスタライブラリ（TOML マニフェスト）を memmap 用バイナリ `.bin` へ変換 | 動作 |

- **決定論的**: 同一入力＋同一 `seed` で出力はビット一致（マルチスレッドでも）。
- **探索モード**: `--search standard`（ドリフト鎖, 既定）/ `vina`（Vina 忠実 MC）/ `qvina2`。詳細は [DIFF.md](DIFF.md)。
- **性能（多様 74 標的, grid, 8 run）**: 成功率 **63%（RMSD<2Å）/ 平均 2.05s**。AutoDock Vina（64% / 7.6s）比
  **約 3.7× 高速・精度ほぼ同等**（[docs/results.202606191907.md](docs/results.202606191907.md)）。

---

## 2. ビルド

```sh
cargo build --release        # 最適化ビルド（推奨）
cargo test                   # 単体テスト
cargo clippy --all-targets   # lint（警告ゼロを維持）
```

生成物: `./target/release/isct-dock`。

- Rust は stable で可（SIMD は `wide` クレートで stable のまま）。
- 検証ハーネス（`bench/`）には別途 Python＋`vina`/`rdkit`/`meeko`/`gemmi` が必要（環境は
  [PORTING.md](PORTING.md)・メモリ参照: rust=`~/.cargo`, それらは `~/miniconda3`）。
- 注: 既定ビルドは baseline x86-64 codegen（AVX2 無効）。`target-cpu=native` は速度が出る反面、精度・決定論・
  移植性に影響するため**既定では使わない**（理由は [SIMD_OPTIMIZE.md](SIMD_OPTIMIZE.md)）。

---

## 3. コマンド

### 3.1 `score` — 固定ポーズのスコア評価（M1）

```sh
./target/release/isct-dock score <receptor.pdbqt> <ligand.pdbqt>
```

入力ポーズ（ligand.pdbqt の座標）をそのまま Vina 互換スコアで採点。探索しない。出力例:

```
# isct-dock score (Vina 互換)
receptor atoms (scored): 1234
ligand atoms   (scored): 23
term         raw_sum        weighted
gauss1        ...            ...
gauss2        ...            ...
repulsion     ...            ...
hydrophob     ...            ...
hbond         ...            ...
Intermolecular energy (Ligand - Receptor): -8.12345 kcal/mol
Ligand torsional DOF (TORSDOF)           : 5
Estimated Free Energy of Binding         : -7.45678 kcal/mol
```

- `Intermolecular energy` が `vina --score_only` の "Ligand - Receptor" と一致（検証ゲート, 差<1e-2）。
- `Estimated Free Energy of Binding` = 構造補正 `e_inter/(1+0.05846·TORSDOF)`。

### 3.2 `dock` — 単一リガンドのドッキング（M2）

```sh
./target/release/isct-dock dock <receptor.pdbqt> <ligand.pdbqt> --config <run.toml> [--out OUT.pdbqt] [--local-only] [--search MODE]
```

| 引数 | 説明 |
|---|---|
| `receptor.pdbqt` | 受容体（剛体）。 |
| `ligand.pdbqt` | リガンド。**入力座標は RMSD 基準（結晶ポーズ）としても使われる**。 |
| `-c, --config` | TOML 設定（§4）。探索ボックス必須。 |
| `-o, --out` | 出力ポーズ PDBQT（既定: `<ligand>_isct_out.pdbqt`）。 |
| `--local-only` | 診断用。大域探索せず入力ポーズから局所最適化のみ。 |
| `--search <standard\|vina\|qvina2>` | 探索アルゴリズムを上書き（config `search` より優先）。`standard`(=`drift`)=ドリフト鎖 / `vina`=Vina 忠実 MC（毎ステップ収束 BFGS）/ `qvina2`。 |
| `--qvina-cos <f64>` | `search="qvina2"` 時の勾配方向一致閾値（cos）を上書き。高い=保守的(速度↓精度↑)、低い=攻める(速度↑精度↓)。省略時は config `qvina_cos`（既定 0.90）。 |

出力（stdout）例:

```
# isct-dock dock (Vina 互換, RMSD クラスタリング)
affinity (Vina, kcal/mol)  : -9.1701
intermolecular energy      : -9.4821 kcal/mol
intramolecular energy      :  0.3120 kcal/mol
total (objective)          : -9.1701 kcal/mol
RMSD vs input(crystal) pose:  0.913 Å
search mode                : Drift
energy evals               : 993429 (124178 /run)
modes (RMSD-clustered)     : 9
  mode  1  affinity= -9.1701  inter= -9.4821  rmsd_vs_crystal= 0.913 Å
  ...
output poses               : ligand_isct_out.pdbqt
```

- **`affinity (Vina)`** = torsion 補正後の Vina 互換最終スコア（`inter/(1+0.05846·N_rot)`）。**順位付け・
  スクリーニングはこの値を使う**（生 intermolecular はサイズ依存で識別力が落ちる）。
- `energy evals` = 探索のエネルギー評価回数（drift/vina モードの仕事量を比較する診断値）。
- 出力 PDBQT は **マルチモデル**（`MODEL n` / `ENDMDL`、Vina 流）。mode 1 が最良。
- `RMSD vs input` は最良モードの重原子 RMSD vs 入力ポーズ（再ドッキング検証用。対称性補正なし）。

### 3.3 `screen` — クラスタライブラリのスクリーニング（M3）

```sh
./target/release/isct-dock screen <receptor.pdbqt> <library.toml|library.bin> --config <run.toml> \
    [--out ranking.tsv] [--poses-out poses.pdbqt --top N] [--pin-threads] [--search MODE]
```

ライブラリは TOML マニフェスト（小・デバッグ）か `pack` 済みバイナリ（`.bin`、大規模・memmap streaming）。
`--search`（dock と同じ。seed 探索に適用）・`--poses-out`（上位 N をマルチモデル PDBQT 出力）・
`--pin-threads`（ワーカを物理コアへピン留め、Linux）。

クラスタ（代表 seed ＋ MCS コアを共有する子分子群）単位で処理する:
- **Phase 1**: seed をドッキングし、結合が悪い（目的関数 > `prune_threshold`）クラスタは**子を全スキップ**（一括棄却）。
- **Phase 2/3**: 生存クラスタの各子を、seed の docked コアへ剛体重ね合わせ→R 基 torsion を局所最適化。

出力: ランキング TSV（既定 `<library と同ディレクトリ>/ranking.tsv`）。列:
`Rank, Compound_ID, Cluster_ID, Total_Score, VdW_Energy, Electrostatic_Energy, Target_Protein_State_ID`。
`Total_Score` は torsion 補正後 affinity（Vina 互換）で昇順ランク。`--poses-out` 指定時は上位 `--top` 件を
マルチモデル PDBQT（各 MODEL に `REMARK Rank/Compound/Cluster/Score/Role`）で出力。

stdout にサマリ（クラスタ数・棄却数・スキップした子数・採点分子数）。

---

## 4. 設定ファイル（TOML）

注釈付きの雛形は **`configs/dock.example.toml` / `configs/screen.example.toml`**（`[box]` の center/size を
ターゲット値に置き換えて使う）。`dock`/`screen` 共通の最小例:

```toml
[box]
center = [-1.99, 14.25, 16.24]   # 探索ボックス中心 [x,y,z] Å（必須）
size   = [20.0, 20.0, 20.0]      # ボックス寸法 [x,y,z] Å（必須）

[run]
seed     = 42
n_runs   = 8
scoring  = "grid"                # ★ 速度のため grid 推奨（既定は "direct"）
```

### `[run]` パラメータと既定値

| キー | 既定 | 説明 |
|---|---|---|
| `seed` | 42 | 乱数シード（決定論）。 |
| `n_runs` | 60 | 独立 run 数（網羅性）。スクリーニングでは 8 程度で十分高精度。 |
| `n_steps` | 0 | per-run の MC ステップ数。**0=自動**（可動原子数×torsion 上限で算出）。 |
| `scoring` | `"direct"` | 探索の分子間評価。`"grid"`=Vina グリッド補間（高速・推奨）/ `"direct"`=直接原子間(SIMD)。 |
| `search` | `"standard"` | 探索モード。`"standard"`=ドリフト鎖 ILS（既定・決定論・高速）/ `"vina"`=Vina 忠実 MC（毎ステップ収束 BFGS。アルゴリズム/実装差の切り分け用, [DIFF.md](DIFF.md)）/ `"qvina2"`=QuickVina2 相当の局所探索スキップ（opt-in 高速化）。`--search` で上書き可。 |
| `qvina_cos` | 0.90 | `qvina2` の勾配方向一致閾値（cos）。低いほどスキップ多＝速度↑精度↓。`--qvina-cos` で上書き可。 |
| `hunt_cap` | 10.0 | 探索中の正エネルギー上限（Vina curl）。激突ポーズの暴走を抑え深い極小へ抜けやすくする。`1e9` で無効化。polish は常に無効（authentic）。 |
| `chain_iters` | 30 | ILS 各ステップの局所最適化反復（浅く保ち基底間をドリフト）。 |
| `polish_iters` | 20 | 代表ポーズの最終厳密 f64 精密化反復。 |
| `polish_modes` | 9 | 最終精密化にかける代表モード数。 |
| `n_modes` | 9 | 出力・保持する最大モード数（Vina 流マルチモデル）。 |
| `cluster_rmsd` | 2.0 | ポーズ RMSD クラスタリング閾値（Å）。 |
| `early_stop_patience` | 0 | 早期停止（0=無効）。収束ベース停止はハード標的を退行させうるため opt-in。 |
| `temp_init` / `temp_final` | 1.0 / 1.0 | Metropolis 温度（既定は固定温度。`init>final` で焼きなまし=opt-in）。 |

### `[screen]` パラメータ（`screen` のみ）

| キー | 既定 | 説明 |
|---|---|---|
| `prune_threshold` | 0.0 | seed の目的関数がこれを上回るクラスタは子を全スキップ（一括棄却）。 |
| `local_iters` | 120 | 子分子の R 基 torsion 局所最適化の最大反復。 |

> **推奨**: 速度重視なら `scoring = "grid"` を明示。grid は受容体ポテンシャルをオンデマンドで補間し、探索を
> 大幅高速化する（最終 polish と報告スコアは厳密 f64 のまま）。

### クラスタライブラリ・マニフェスト（`screen` の `<library.toml>`）

```toml
[[cluster]]
id = "ampc_c0"
pocket_center = [x, y, z]    # 任意（省略時は [box]）
pocket_size   = [sx, sy, sz] # 任意

[cluster.seed]
compound_id = "..."
pdbqt = "seed.pdbqt"         # マニフェストからの相対パス
core_atoms = [0, 1, 2]       # seed 側 MCS コア原子（全原子 idx）

[[cluster.child]]
compound_id = "..."
pdbqt = "child_0.pdbqt"      # コア=ROOT でルート化済み
core_atoms = [0, 1, 2]       # 子側コア（seed.core_atoms と 1:1 同順・同数）
```

不変条件: 子の `core_atoms` は seed と同数・同順で対応し、コアは剛体 root 内（torsion で動かない）にあること。

---

## 5. 入出力フォーマット

- **入力**: AutoDock **PDBQT**（受容体・リガンド）。RDKit/Meeko で生成（`mk_prepare_receptor`/`mk_prepare_ligand`）。
  - 柔軟マクロ環（meeko の `CG0/G0` グルー擬原子）は **対応済み**（`CG0`→脂肪族炭素として採点、`G0`→
    非採点ダミー）。剛体化（`--rigid_macrocycles`）入力も可。
  - 受容体に欠損原子がある実 PDB は `-a --default_altloc A` 等で前処理（[bench/prepare_pdb.py] 参照）。
- **出力**: マルチモデル PDBQT（`dock`）/ ランキング TSV（`screen`）。
- 並列度: 環境変数 **`RAYON_NUM_THREADS`** でスレッド数を制御（既定は論理コア数）。決定論は保たれる。

---

## 6. 検証・ベンチマーク（`bench/`、要 Python＋vina/rdkit/meeko）

| スクリプト | 用途 |
|---|---|
| `bench/run_vina_compare.py` | `score` の Vina 一致検証（M1 ゲート）。 |
| `bench/run_redock.py <case_dir>` | 1 標的の再ドッキング RMSD・スコアを Vina と比較。 |
| `bench/bench_diverse.py` | 多様データセット（74 標的×3 seed）で Vina と速度・精度を比較。 |
| `bench/guard_redock.py` | isct 単独の易標的ガード（grid, n_steps 自動）。 |
| `bench/make_diverse_set.py` | RCSB 自動取得→Meeko で多様データセット生成。 |
| `bench/make_screening.py` / `get_dude.py` / `analyze_dude_clusters.py` | スクリーニング/DUD-E データ生成・解析。 |
| `bench/prep_dude_ligands.py` / `screen_eval.py` / `screen_metrics.py` | DUD-E スクリーニング評価（ISCT vs Vina/Smina、AUC/EF/BEDROC・energy・速度）。設計は [EVALUATION.SCREENING.md](EVALUATION.SCREENING.md)。 |
| `tools/preprocess/build_library.py [--full-coverage]` | クラスタライブラリ生成（MCS クラスタ＋コア拘束 embed）。`--full-coverage` で全リガンドを 1 件も欠かさず収録（singleton は全自由度 seed）。→ `isct-dock pack` でバイナリ化。 |

評価方法とデータは [EVALUATION.md](EVALUATION.md)、評価結果のログは `docs/results.YYYYMMDDHHMM.md` に記録される。

---

## 7. リポジトリ構成

```
crates/isct-core   原子型・Vina スコア・幾何・grid・SIMD カーネル … 純計算
crates/isct-io     PDBQT パーサ・TOML config・クラスタライブラリ I/O
crates/isct-dock   ligand(torsion木)・optimizer(BFGS)・search(MC)・energy・screen・phases
crates/isct-cli    バイナリ `isct-dock`（score / dock / screen / pack）
bench/             検証スクリプト（Python）と data/
tools/preprocess/  前処理スクリプト置き場（build_library.py 等）
configs/           注釈付きサンプル設定（dock.example.toml / screen.example.toml）
docs/              評価結果(results.*) と知見(findings.*) のログ（追記のみ・更新しない）
```

主要アルゴリズム: Iterated Local Search（MC＋BFGS, 解析的勾配）/ Vina 互換スコア（grid 補間 or 直接 SIMD）/
RMSD ポーズクラスタリング / scaffold 固定ドッキング（Kabsch 重ね合わせ）。

---

## 8. 既知の制限

- 一部の**中〜高柔軟リガンド・高 torsion 標的**で Vina にわずかに劣る。kif11 の少数 subset では ligand 依存の
  探索差（scoring バグではない＝Vina ポーズをほぼ同値で再採点可能, [docs/findings.202606222119.md](docs/findings.202606222119.md)）。
  多様な全 DUD-E では isct ≈ vina。複数標的での確認が継続課題。
- RMSD は対称性補正なし（両エンジン同条件）。
- ISCT-screen（scaffold 再利用）は congeneric ライブラリ向け。多様な VS 集合（DUD-E）では再利用率が低く、
  per-ligand `dock` が速度・濃縮とも優位（[docs/results.202606220300.md](docs/results.202606220300.md)）。

---

## 9. 関連ドキュメント

- [isct-dock.md](isct-dock.md) — 設計仕様。
- [DIFF.md](DIFF.md) — AutoDock Vina との差（アルゴリズム/実装）と速度差の切り分け（`--search vina` で検証）。
- [PLAN.md](PLAN.md) / `PLAN.*.md` — 開発計画・各機能の設計。
- [EVALUATION.md](EVALUATION.md) / [EVALUATION.SCREENING.md](EVALUATION.SCREENING.md) — 評価方法・データ。
- `configs/dock.example.toml` / `configs/screen.example.toml` — 注釈付きサンプル設定（§4）。
- [PORTING.md](PORTING.md) — 環境移行（Mac→Linux）手順・OS 差分。
- [SIMD_OPTIMIZE.md](SIMD_OPTIMIZE.md) — SIMD 最適化の知見（律速・トレードオフ）。
- [CLAUDE.md](CLAUDE.md) — リポジトリ運用ルール（ログ記録規約等）。
- `docs/results.*` / `docs/findings.*` — 評価結果・知見の時系列ログ。
