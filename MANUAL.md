# MANUAL.md — ISCT-Dock 利用マニュアル

ISCT-Dock は、機械学習を使わず幾何・物理化学計算で行う高速・決定論的な分子ドッキング/スクリーニング
エンジン（Rust 実装）。AutoDock Vina 互換のスコア関数を用いる。本書はビルド方法と、`isct-dock` バイナリ
（`score`/`dock`/`screen`/`pack`）・`tools/preprocess/` スクリプトの使い方を示す。

---

## 1. ビルド

```sh
cargo build --release        # 最適化ビルド（推奨）
cargo test                   # 単体テスト
cargo clippy --all-targets   # lint（警告ゼロを維持）
```

生成物: `./target/release/isct-dock`。

- Rust は stable で可（SIMD は `wide` クレートで stable のまま）。workspace `rust-version = "1.92"`。
- 既定ビルドは baseline x86-64 codegen（AVX2 無効）・厳密 f64 で**決定論的**（同一 seed で全 CPU ビット一致）。

### 1.1 opt-in 高速ビルド（native-CPU、精度トレードオフあり）

速度を優先できる場合、実行機の CPU 向けに codegen する高速ビルドを opt-in できる。

```sh
# 方法A: 1 回だけ native ビルド（設定ファイルを置かない）
RUSTFLAGS="-C target-cpu=native" cargo build --release

# 方法B: 恒常化（テンプレートをコピーして有効化。戻すには .cargo/config.toml を削除）
cp .cargo/config.toml.example .cargo/config.toml
cargo build --release
```

**効果は workload・CPU 依存で、必ずしも大きくない（本機 = 80 コア Xeon での実測）**:
- **単発 dock（scoring=direct・SIMD f32 経路）**: ~5%。affinity がわずかに変化（−12.4797→−12.4750）
  ＝f32 FMA の丸め差で結果が変わる（accuracy トレードオフ）。
- **grid スクリーニング（既定・推奨）**: **速度差ほぼ無し・出力ビット一致**（grid 補間は f64 で native の
  影響を受けにくい）。→ 10M grid スクリーニングでは native は実効ゲインが乏しい。
- per-op の native ゲイン（~1.3×）は、多コアで多数の run/クラスタを並列実行するとスケジューリングに
  希釈される。**少コア・レイテンシ律速の単発 dock ほど効きやすい**。

**トレードオフ**: direct/SIMD 経路では f32 codegen 依存で borderline 標的が反転しうる（フル DUD-E で net
−7/222 を実測）。結果が CPU codegen 依存になり異機種間でビット一致しなくなる。既定ビルド（無効時）は
従来どおり厳密 f64・決定論的。有効化した `.cargo/config.toml` は gitignore 済み。詳細は
`.cargo/config.toml.example` のコメント参照。

---

## 2. コマンド

### 2.1 `score` — 固定ポーズのスコア評価

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

### 2.2 `dock` — 単一リガンドのドッキング

```sh
./target/release/isct-dock dock <receptor.pdbqt> <ligand.pdbqt> --config <run.toml> [--out OUT.pdbqt] [--local-only] [--search MODE]
```

| 引数 | 説明 |
|---|---|
| `receptor.pdbqt` | 受容体（剛体）。 |
| `ligand.pdbqt` | リガンド。**入力座標は RMSD 基準（結晶ポーズ）としても使われる**。 |
| `-c, --config` | TOML 設定（§3）。探索ボックス必須。 |
| `-o, --out` | 出力ポーズ PDBQT（既定: `<ligand>_isct_out.pdbqt`）。 |
| `--local-only` | 診断用。大域探索せず入力ポーズから局所最適化のみ。 |
| `--search <standard\|vina\|qvina2\|tpe>` | 探索アルゴリズムを上書き（config `search` より優先）。`standard`(=`drift`)=ドリフト鎖 / `vina`=Vina 忠実 MC（毎ステップ収束 BFGS）/ `qvina2` / `tpe`=TPE（Parzen サンプリング探索）。 |
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

### 2.3 `screen` — クラスタライブラリのスクリーニング

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

### 2.4 `pack` — クラスタライブラリのバイナリ化

```sh
./target/release/isct-dock pack <manifest.toml> <out.bin>
```

`screen` 用の TOML マニフェスト（クラスタ・seed/child の PDBQT 参照）を、memmap で読める単一バイナリ
（serde+bincode）に変換する。`screen` に `.bin` を渡すとクラスタ単位で遅延デコードし、全 RAM ロードを
回避できる（大規模ライブラリ向け）。TOML 直読みと出力はビット一致。

---

## 3. 設定ファイル（TOML）

注釈付きの雛形は **`configs/dock.example.toml` / `configs/screen.example.toml`**（`[box]` の center/size を
ターゲット値に置き換えて使う）。`dock`/`screen` 共通の最小例:

```toml
[box]
center = [-1.99, 14.25, 16.24]   # 探索ボックス中心 [x,y,z] Å（必須）
size   = [20.0, 20.0, 20.0]      # ボックス寸法 [x,y,z] Å（必須）

[run]
seed     = 42
n_runs   = 8                     # 省略可（未指定ならモード別既定: standard/qvina2/tpe=60, vina=8）
scoring  = "grid"                # ★ 速度のため grid 推奨（既定は "direct"）
```

### `[run]` パラメータと既定値

| キー | 既定 | 説明 |
|---|---|---|
| `seed` | 42 | 乱数シード（決定論）。 |
| `n_runs` | モード別（下記） | 独立 run 数（網羅性）。**未指定時の既定は探索モードで変わる**: `standard`/`qvina2`/`tpe`=60、`vina`=8。**`search="vina"` では n_runs = AutoDock Vina の exhaustiveness に対応**（Vina 既定=8）で、vina は 1 run が重いため既定を 8 にしてある。明示指定すればモードに依らずその値を使う（vina で大きな値を指定すると Vina の n_runs/8 倍遅くなる）。スクリーニングでは 8 程度で十分高精度。 |
| `n_steps` | 0 | per-run の MC ステップ数。**0=自動**（可動原子数×torsion 上限で算出）。 |
| `scoring` | `"direct"` | 探索の分子間評価。`"grid"`=Vina グリッド補間（高速・推奨）/ `"direct"`=直接原子間(SIMD)。 |
| `search` | `"standard"` | 探索モード。`"standard"`=ドリフト鎖 ILS（既定・決定論・高速）/ `"vina"`=Vina 忠実 MC（毎ステップ収束 BFGS）/ `"qvina2"`=QuickVina2 相当の局所探索スキップ（opt-in 高速化）/ `"tpe"`=TPE（Tree-structured Parzen Estimator。下記）。`--search` で上書き可。 |
| `qvina_cos` | 0.90 | `qvina2` の勾配方向一致閾値（cos）。低いほどスキップ多＝速度↑精度↓。`--qvina-cos` で上書き可。 |
| `hunt_cap` | 10.0 | 探索中の正エネルギー上限（Vina curl）。激突ポーズの暴走を抑え深い極小へ抜けやすくする。`1e9` で無効化。polish は常に無効（authentic）。 |
| `chain_iters` | 30 | ILS 各ステップの局所最適化反復（浅く保ち基底間をドリフト）。 |
| `polish_iters` | 20 | 代表ポーズの最終厳密 f64 精密化反復。 |
| `polish_modes` | 9 | 最終精密化にかける代表モード数。 |
| `n_modes` | 9 | 出力・保持する最大モード数（Vina 流マルチモデル）。 |
| `cluster_rmsd` | 2.0 | ポーズ RMSD クラスタリング閾値（Å）。 |
| `early_stop_patience` | 0 | 早期停止（0=無効）。収束ベース停止はハード標的を退行させうるため opt-in。 |
| `temp_init` / `temp_final` | 1.0 / 1.0 | Metropolis 温度（既定は固定温度。`init>final` で焼きなまし=opt-in）。 |

#### `tpe` モード（TPE=Tree-structured Parzen Estimator）
`--search tpe` / `search = "tpe"` で選択。TPEvina-2.0（PSOVina 派生）の TPE を移植したもの。各次元を
独立 Gaussian Parzen 混合でモデル化し、良観測（lower）近傍から候補をサンプル→ BFGS 局所最適化→観測更新を
繰り返す（前半はウォームアップの一様ランダム）。局所最適化・スコア・grid は他モードと共通機構を再利用。
- **決定論**: seeded `ChaCha8Rng` で駆動し**同一 seed でビット一致**。
- **既定挙動**: TPEvina 既定ハイパーパラメータ（`sample_num=7` 等）を踏襲。この既定では greater 重みが
  探索初期に 0 へ収束するため、実効的に**良ポーズ近傍の Parzen 探索（exploitation 主体）**になる。
- **ステップ予算**: `n_steps=0`（自動）で `auto_n_steps_tpe`（vina より多め＝モデル用ウォームアップ確保）。
- ハイパーパラメータは現状コード内既定（TPEvina 準拠）で固定。config 露出は将来対応。品質は標的依存
  （深い漏斗の標的で良好、探索性の要る標的では drift に劣ることがある）。

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
この形式のマニフェストは `tools/preprocess/build_library.py` で生成できる（§5）。

---

## 4. 入出力フォーマット

- **入力**: AutoDock **PDBQT**（受容体・リガンド）。RDKit/Meeko で生成（`mk_prepare_receptor`/`mk_prepare_ligand`、
  あるいは `tools/preprocess/prepare_receptor.py`）。
  - 柔軟マクロ環（meeko の `CG0/G0` グルー擬原子）は **対応済み**（`CG0`→脂肪族炭素として採点、`G0`→
    非採点ダミー）。剛体化（`--rigid_macrocycles`）入力も可。
- **出力**: マルチモデル PDBQT（`dock`）/ ランキング TSV（`screen`）/ バイナリ `.bin`（`pack`）。
- 並列度: 環境変数 **`RAYON_NUM_THREADS`** でスレッド数を制御（既定は論理コア数）。決定論は保たれる。

---

## 5. `tools/preprocess/` — クラスタライブラリ前処理スクリプト

`isct-dock` エンジン自体は前処理を行わない（読込専用）。`screen` 用のクラスタライブラリ
（seed＋child の PDBQT 群＋TOML マニフェスト）は、以下の Python スクリプト（RDKit/Meeko 使用）で用意する。

### 5.1 `build_library.py` — マニフェスト一式の生成（主エントリポイント）

```sh
python3 tools/preprocess/build_library.py <target> [オプション]
```

各子分子の MCS コアを seed コア配座へ constrained embed して 3D 生成し、PDBQT 化する。出力
（既定 `bench/data/library/<target>/`）: `receptor.pdbqt`, `mols/<id>.pdbqt`, `library.toml`, `labels.tsv`。

| オプション | 既定 | 説明 |
|---|---|---|
| `target` | — | 対象名（データソースのターゲット識別子）。 |
| `--out-dir` | `bench/data/library/<target>` | 出力先ディレクトリ。 |
| `--tanimoto` | 0.5 | クラスタリング閾値（Morgan FP Tanimoto）。 |
| `--max-each` | 無制限 | active/decoy 各読込上限。 |
| `--actives-only` | off | active のみ読み込む。 |
| `--max-clusters` | 1 | 生成する最大クラスタ数。 |
| `--max-children` | 3 | クラスタあたりの最大子分子数。 |
| `--box-size` | 22.5 | 探索ボックス寸法（Å）。 |
| `--seed` | `0xC0FFEE` | 乱数シード。 |
| `--full-coverage` | off | 全リガンドを 1 件も欠かさず収録（コア占有率が低い子は singleton＝全自由度 seed にフォールバック）。 |
| `--jobs` | 8 | `--full-coverage` 使用時の並列数。 |

使用例:
```sh
python3 tools/preprocess/build_library.py ampc --actives-only --max-each 60 \
    --tanimoto 0.5 --max-clusters 1 --max-children 3 --out-dir bench/data/library/ampc_mini
```

生成した `library.toml` は `isct-dock pack` でバイナリ化できる（§2.4）。

### 5.2 `prepare_receptor.py` — 受容体の PDBQT 化・box 中心算出

```sh
python3 tools/preprocess/prepare_receptor.py <target> [--out-dir DIR] [--box-size S]
```

受容体 PDB を `mk_prepare_receptor` で PDBQT 化し、結晶リガンドの重心を box 中心として算出する。
出力: `<out-dir>/receptor.pdbqt`（既定 `bench/data/library/<target>`）と box 中心（stdout, JSON）。
`build_library.py` から内部的にも呼ばれる。

### 5.3 `cluster_library.py` — Scaffold クラスタリング

```sh
python3 tools/preprocess/cluster_library.py <target> [--tanimoto 0.6] [--max N]
```

Morgan フィンガープリント（r=2, 2048bit）の Tanimoto 類似度で前ふるいし、連結成分をクラスタ候補とする。
CLI 単体ではクラスタサイズ分布の統計を出力する診断用途。`read_ism` / `morgan_fps` / `cluster_by_tanimoto` /
`mcs_of` は `build_library.py` からモジュールとして import されるコア関数。

---

## 6. リポジトリ構成

```
crates/isct-core   原子型・Vina スコア・幾何・grid・SIMD カーネル … 純計算
crates/isct-io     PDBQT パーサ・TOML config・クラスタライブラリ I/O
crates/isct-dock   ligand(torsion木)・optimizer(BFGS)・search(MC)・energy・screen・phases
crates/isct-cli    バイナリ `isct-dock`（score / dock / screen / pack）
tools/preprocess/  クラスタライブラリ前処理スクリプト（build_library.py 等, §5）
configs/           注釈付きサンプル設定（dock.example.toml / screen.example.toml）
```

主要アルゴリズム: Iterated Local Search（MC＋BFGS, 解析的勾配）/ Vina 互換スコア（grid 補間 or 直接 SIMD）/
RMSD ポーズクラスタリング / scaffold 固定ドッキング（Kabsch 重ね合わせ）。
