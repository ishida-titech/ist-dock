# MANUAL.md — IST-Dock 利用マニュアル

IST-Dock は、機械学習を使わず幾何・物理化学計算で行う高速・決定論的な分子ドッキング/スクリーニング
エンジン（Rust 実装）。AutoDock Vina 互換のスコア関数を用いる。本書はビルド方法と、`ist-dock` バイナリ
（`score`/`dock`/`screen`/`pack`）・`tools/preprocess/` スクリプトの使い方を示す。

---

## 1. ビルド

```sh
cargo build --release        # 最適化ビルド（推奨）
cargo test                   # 単体テスト
cargo clippy --all-targets   # lint（警告ゼロを維持）
```

生成物: `./target/release/ist-dock`。

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

# 方法C: AVX2 のみ（portable, 推奨）。native と違い AVX2 対応 CPU 間でビット一致を保つ。
#         .cargo/config.toml に rustflags = ["-C","target-cpu=x86-64-v3"] を書く（grid 経路が主目的ならこれで十分）。
RUSTFLAGS="-C target-cpu=x86-64-v3" cargo build --release
```

**効果は workload・CPU 依存（本機 = 80 コア Xeon での実測）**:
- **grid（既定・推奨、`scoring="grid"`）**: **AVX2 で速くなり、かつ出力ビット一致**。グリッド構築が
  スキャッタ＋`f64x4` SIMD 化されており AVX2 で本来のレーン幅で回る（グリッド床 ~2.9×、全 dock −10〜16%）。
  加算は f64 なので**スコア/RMSD は非 AVX2 とビット完全一致**（Astex 全85×3seed の 255×3 サンプルで確認,
  `docs/results.202607092101`）＝**精度トレードオフなしで速くなる**。**grid 経路では AVX2 有効化を推奨**。
  - ただし v3 非対応の**旧 CPU では動かない**ため、既定ビルド（無効）は portable・決定論のまま据え置く。
- **単発 dock（`scoring="direct"`・SIMD f32 経路）**: ~5%。**こちらは精度トレードオフあり** — f32 FMA の丸め差で
  affinity がわずかに変化（−12.4797→−12.4750）、borderline 標的が反転しうる（フル DUD-E で net −7/222）。
- per-op の native ゲインは多コアで多数の run/クラスタを並列実行するとスケジューリングに希釈される。
  **少コア・レイテンシ律速の単発 dock ほど効きやすい**。

**まとめ**: **grid 経路は AVX2 で「速く・ビット一致」＝有効化推奨**。direct/f32 経路のみ codegen 依存の丸め差で
結果が変わり異機種間ビット一致が崩れる（accuracy トレードオフ）。既定ビルド（無効時）は従来どおり baseline
x86-64・厳密 f64・決定論的。有効化した `.cargo/config.toml` は gitignore 済み。詳細は
`.cargo/config.toml.example` のコメント参照。

---

## 2. コマンド

### 2.1 `score` — 固定ポーズのスコア評価

```sh
./target/release/ist-dock score <receptor.pdbqt> <ligand.pdbqt>
```

入力ポーズ（ligand.pdbqt の座標）をそのまま Vina 互換スコアで採点。探索しない。出力例:

```
# IST-Dock  v1.0.0 (git <hash>)
# score: Vina-compatible scoring of a fixed pose (no search)
Receptor : receptor.pdbqt  (2526 atoms scored)
Ligand   : ligand.pdbqt  (22 atoms scored)

TERM              RAW_SUM     WEIGHTED
Gauss1           73.36092     -2.61011
Gauss2         1071.20048     -5.52311
Repulsion         4.72780      3.97251
Hydrophobic      16.81092     -0.58954
HBond             4.58334     -2.69243

Intermolecular energy (ligand-receptor) : -7.44268 kcal/mol
Ligand torsional DOF (TORSDOF)          : 10
Estimated Free Energy of Binding        : -4.78515 kcal/mol
```

- 先頭2行はヘッダ（全コマンド共通）: `# IST-Dock vX.Y.Z (git <hash>)`（版＋ビルド時 git ハッシュ。`-dirty`=未コミット変更あり）＋コマンド概要。
- `TERM` テーブルは各 Vina 項の `RAW_SUM`（重み付け前の生和）と `WEIGHTED`（重み積）。
- `Intermolecular energy (ligand-receptor)` が `vina --score_only` の "Ligand - Receptor" と一致（検証ゲート, 差<1e-2）。
- `Ligand torsional DOF (TORSDOF)` は探索木の回転結合数（BRANCH 数）。標準前処理なら PDBQT の TORSDOF 行と一致する。
- `Estimated Free Energy of Binding` = 構造補正 `e_inter/(1+0.05846·num_tors)`。`num_tors` は末端基を 0.5 換算した**実効**回転結合数（Vina 1.2 準拠。表示の TORSDOF とは一般に近いが別量）。

### 2.2 `dock` — 単一リガンドのドッキング

```sh
./target/release/ist-dock dock <run.toml> <receptor.pdbqt> <ligand.pdbqt> [--native NATIVE.pdbqt] [--out OUT.pdbqt] [--local-only] [--search MODE] [--trace TRACE.tsv]
```

| 引数 | 説明 |
|---|---|
| `run.toml` | TOML 設定（§3、位置引数・必須）。探索ボックス必須。 |
| `receptor.pdbqt` | 受容体（剛体）。 |
| `ligand.pdbqt` | ドッキングする入力（開始）構造。**native 前提は置かない**（再ドッキング検証の参照は `--native` で渡す）。 |
| `--native <PDBQT>` | 同一リガンドの参照（native/結晶）ポーズ。**指定時のみ**各モードに `rmsd_to_native` を出力（再ドッキング検証用）。原子数が入力と一致しないとエラー。 |
| `-o, --out` | 出力ポーズ PDBQT（既定: `<ligand>_ist_out.pdbqt`）。 |
| `--local-only` | 診断用。大域探索せず入力ポーズから局所最適化のみ。 |
| `--search <tpe\|drift\|drift_qvina2\|qvina2\|vina>` | 探索アルゴリズムを上書き（config `search` より優先）。**`tpe`=既定**（共有 TPE＋局所チェーン＋局所探索スキップ＋剛直軽量化の低予算プリセット, n_runs=1。dock/screen 共通）/ `drift`=ドリフト鎖（旧称 `standard`, 後方互換で受理）/ `drift_qvina2`=ドリフト鎖＋局所探索スキップ（ist 独自, n_runs=60）/ `qvina2`=**本来の QuickVina2 相当**（Vina 忠実 MC＋スキップ, exhaustiveness=8）/ `vina`=Vina 忠実 MC（毎ステップ収束 BFGS）。 |
| `--trace <TSV>` | **診断用**: 探索の全ステップを TSV 出力（全モード対応, §2.2.1）。探索結果には無影響・決定論維持。`--local-only` では無効。 |
| `--qvina-cos <f64>` | `search="qvina2"` 時の勾配方向一致閾値（cos）を上書き。高い=保守的(速度↓精度↑)、低い=攻める(速度↑精度↓)。省略時は config `qvina_cos`（既定 0.90）。**マイナーオプションのため `--help` には非表示**（機能は有効）。 |

出力（stdout）例:

出力は「ヘッダ → 入力分子 → 探索パラメータ → （空行）→ 結果 → （空行）→ 出力ファイル」の順（`--native` を付けた例）:

```
# IST-Dock  v1.0.0 (git <hash>)
# dock: single-ligand docking (Vina-compatible, RMSD clustering)
Receptor : receptor.pdbqt  (2526 atoms scored)
Ligand   : ligand.pdbqt  (22 atoms scored)

Search mode                    : TPE
Scoring backend                : Grid
Search box center (x,y,z)      : 65.607, 31.083, 117.843
Search box size (x,y,z)        : 15.750, 15.000, 18.750
Receptor atoms in box          : 777
Ligand torsional DOF (TORSDOF) : 10
Runs                           : 1
Steps per run                  : 2436 (auto)
Seed                           : 42

Affinity (Vina, kcal/mol)      : -6.6530
Intermolecular energy          : -10.3478 kcal/mol
Intramolecular energy          : 1.9354 kcal/mol
Total (objective)              : -8.4125 kcal/mol
Energy evals                   : 574594 (574594 /run)
Modes (RMSD-clustered)         : 7
  mode  1  affinity= -6.6530  inter=-10.3478  rmsd_to_native= 1.543 Å
  mode  2  affinity= -6.3597  inter= -9.8917  rmsd_to_native= 3.501 Å
  ...

Output poses                   : ligand_ist_out.pdbqt
```

- **パラメータ→結果の順**: `Search mode`〜`Seed` は実行前に確定するパラメータ、空行以降が結果。
- **`Receptor atoms in box`** はドッキング時に探索ボックス近傍へプルーニングした受容体原子数（`score` の全採点原子数とは異なる。`Receptor` 行の括弧内は全採点原子数）。
- **`Affinity (Vina)`** = torsion 補正後の Vina 互換最終スコア（`inter/(1+0.05846·num_tors)`）。**順位付け・
  スクリーニングはこの値を使う**（生 intermolecular はサイズ依存で識別力が落ちる）。
- `Energy evals` = 探索のエネルギー評価回数（探索モード間の仕事量を比較する診断値）。
- **`rmsd_to_native`** は `--native` 指定時のみ各モードに併記（最良モードの重原子 RMSD vs 参照ポーズ。再ドッキング検証用・対称性補正なし）。未指定なら `affinity`/`inter` のみ。
- 出力 PDBQT は **マルチモデル**（`MODEL n` / `ENDMDL`、Vina 流）。mode 1 が最良。

#### 2.2.1 探索トレース `--trace`（診断用・全モード対応）
`--trace <TSV>` で探索の**全ステップ**を TSV 出力する（`drift`/`drift_qvina2`/`qvina2`/`vina`/`tpe` 全モード）。各ステップに
**生成点（BFGS 前）と `bfgs`（最適化後）の 2 行**。探索結果・決定論には無影響（トレースのエネルギー評価は別
カウンタ・rng 不消費で、有無でスコアはビット一致）。`--local-only` では無効。列は**汎用情報→アルゴリズム依存情報**
の順:

| 列 | 区分 | 意味 |
|---|---|---|
| `run` | 汎用 | 独立 run のインデックス（TPE 共有モードは 0）。 |
| `step` | 汎用 | run 内のステップ/提案インデックス。 |
| `energy` | 汎用 | その点の探索目的値（inter+intra, hunt_cap 付, kcal/mol）。 |
| `pos_x/y/z` | 汎用 | リガンド root 重心の**絶対**座標（Å）。 |
| `rot_x/y/z` | 汎用 | **絶対**配向（scaled-axis, rad）。 |
| `torsions` | 汎用 | free torsion の**絶対**角（rad, `;` 区切り）。並び順はヘッダの `# TORSION k ...` に対応（下記）。 |
| `phase` | 依存 | 生成源。MC(`drift`/`drift_qvina2`/`qvina2`/`vina`)=`start`/`mutate`/`skip`/`bfgs`、TPE=`warmup`/`model`/`coverage`/`bfgs`。 |
| `greater_active` | 依存 | TPE `model` 行のみ 0/1（1=g(x) 活性＝観測 27 個以上で exploration 有効）。他は `-`。 |
| `accepted` | 依存 | MC `bfgs` 行のみ Metropolis 受理 0/1。他は `-`。 |

座標は**絶対値**でモード横断比較が可能。用途例: MC の `mutate`（現ポーズの局所変異＝ランダムウォーク）と TPE の
`model`（Parzen 大域サンプル）で**探索の広さ・動態の違い**を比較、`accepted` で MC の受理率、`greater_active` で
TPE の exploration 有効性、生成点→`bfgs` の energy 差で BFGS の寄与を可視化できる。
`awk -F'\t' '$11=="mutate"{...}'` 等で解析。

**探索ボックス（ヘッダ）**: ヘッダに探索ボックスも出力する（絶対座標 `pos_*` の探索範囲＝config `[box]` と同義）:
```
# BOX center 7.2880 6.2080 23.0460 size 18.0000 17.2500 18.7500
```
`center`=箱の中心、`size`=各辺長(Å)。

**torsion → ligand.pdbqt 対応（ヘッダ）**: `torsions` 列は内部フラグメント木の順で、そのままでは外部ツールで
構造再生できない。そこでヘッダに各列の対応を出力する:
```
# torsions column order maps to these ligand.pdbqt rotatable bonds:
# TORSION 1 serials 7-11 atoms N7-C17
# TORSION 2 serials 11-12 atoms C17-C18
...
```
`TORSION k` が `torsions` 列の k 番目（1 始まり）に対応し、`serials A-B` は ligand.pdbqt の `BRANCH A B` 行
（回転結合の 2 原子 serial）、`atoms` はその原子名。角度は参照（入力）ポーズからの絶対 torsion(rad)。これで
各 torsion 値がどの回転結合の回転角かを一意に対応付けて構造を再生できる。

### 2.3 `screen` — ライブラリのスクリーニング

```sh
./target/release/ist-dock screen <run.toml> <receptor.pdbqt> <library.toml|library.bin> \
    [--out ranking.tsv] [--poses-out poses.pdbqt --top N] [--pin-threads] [--search MODE]
```

単一受容体に対しライブラリ全分子をドッキングする。**共有グリッドを 1 度だけ構築して全分子で再利用**するため、
`dock` を分子数ぶん繰り返すのと違い、グリッド再構築・受容体パース・プロセス起動が **1 回**で済む（大規模ほど有利）。
ライブラリは TOML マニフェスト（小・デバッグ）か `pack` 済みバイナリ（`.bin`、大規模・memmap streaming）。共通オプション:
`--search`（dock と同じ。seed 探索に適用）・`--poses-out`（上位 `--top` をマルチモデル PDBQT 出力）・
`--pin-threads`（ワーカを物理コアへピン留め、Linux）。並列度は `RAYON_NUM_THREADS`。

**ライブラリの作り方で 2 通りの使い方がある**:

| | (A) クラスタリングして使う（§2.3.1） | (B) そのまま使う＝平坦/singleton（§2.3.2） |
|---|---|---|
| 対象 | congeneric シリーズ（共通スキャフォールド） | 多様な化合物ライブラリ（クラスタリング不要） |
| 各分子の扱い | seed＋MCS コア共有の子分子群 | 1 分子 = 1 クラスタ（子なし・全自由度） |
| 計算 | seed 1回 dock ＋ 子の核固定配置（Phase 1-3） | 各分子を独立に完全ドッキング（=`dock` と同一） |
| グリッド | クラスタが pocket を持たなければ共有 | 全 singleton が [box] 共有＝1回構築で再利用 |

いずれも出力はランキング TSV（既定 `<library と同ディレクトリ>/ranking.tsv`）。列:
`Rank, Compound_ID, Cluster_ID, Total_Score, VdW_Energy, Electrostatic_Energy, Target_Protein_State_ID`。
`Total_Score` は torsion 補正後 affinity（Vina 互換）で昇順ランク。`--poses-out` 指定時は上位 `--top` 件を
マルチモデル PDBQT（各 MODEL に `REMARK Rank/Compound/Cluster/Score/Role`）で出力。stdout にサマリ
（クラスタ数・棄却数・スキップした子数・採点分子数）。

> **堅牢性**: TOML ライブラリのロードは**寛容**で、解釈不能なリガンド（元素すら推定できない原子・不正 PDBQT）は
> **全体を止めずスキップして継続**する（10M 規模で 1 件が全ランを落とさない）。スキップ件数はサマリの
> `ligands skipped(parse)` に出し、詳細（Cluster_ID/Compound_ID/Reason）を `skipped.tsv`（ランキングと同ディレクトリ）
> に記録する。単発 `dock`/`score` は 1 分子を意図的に渡す用途のため、解釈不能な原子は明確にエラーにする。

#### 2.3.1 クラスタライブラリを使う場合（scaffold 再利用）

**用途**: 共通コアを持つ congeneric シリーズ。代表 seed を 1 回ドッキングし、子分子はそのコアへ重ねるだけで
探索を共有・削減する。クラスタ（代表 seed ＋ MCS コアを共有する子分子群）単位で処理する:
- **Phase 1**: seed をドッキングし、結合が悪い（目的関数 > `prune_threshold`）クラスタは**子を全スキップ**（一括棄却）。
- **Phase 2/3**: 生存クラスタの各子を、seed の docked コアへ剛体重ね合わせ→R 基 torsion を局所最適化。

マニフェストは `[[cluster]]` に `seed`（`core_atoms` 非空）と `[[cluster.child]]`（seed と 1:1 同順のコア）を持つ
（§3 の「クラスタライブラリ・マニフェスト」）。`tools/preprocess/build_library.py` で生成する（§5.1）。

> 注意: 子は seed コアへ剛体で重ねる前提。**多様性の高いライブラリ（例 DUD-E）では共通コアの再利用率が低く
> 利得が限定的**なことがある（評価は `docs/` を参照）。congeneric 最適化シリーズで効果が大きい。

#### 2.3.2 クラスタリングせずそのまま使う場合（平坦/singleton スクリーニング）

**用途**: 多様な化合物を単純に全件ドッキングしたい（クラスタリングするほどでない）場合。各リガンドを
**1 クラスタ（seed のみ・children なし・`core_atoms` 空＝全自由度ドッキング）** にする。`dock` と**厳密に同一の
探索・スコア（ビット一致）**でありながら、pocket を書かず `[box]` を共有するのでグリッドは 1 度だけ構築され全
リガンドで再利用される。

**ヘルパ `tools/ist_screening.py`** が受容体＋リガンドディレクトリからこの singleton ライブラリを生成し `screen`
を実行する（マニフェストを手書きしなくてよい）:

```sh
python3 tools/ist_screening.py <receptor.pdbqt> <ligand_dir> \
    [--n_dockings N] [--clean] [--center X Y Z] [--size S] \
    [--search MODE] [--n_runs N] [--seed S] [--prune_threshold T] [--workdir DIR] [--out PATH] [--bin PATH]
```

| オプション | 既定 | 説明 |
|---|---|---|
| `receptor` / `ligand_dir` | — | 第1=受容体 PDBQT、第2=リガンド PDBQT（`*.pdbqt`）のディレクトリ。 |
| `--n_dockings N` | 全件 | **ファイル名昇順で先頭 N 件**のみ処理。 |
| `--clean` | off（保持） | **指定時のみ**中間ファイル（`run.toml`/`library.toml`）を削除。ランキング TSV は常に残す。 |
| `--center X Y Z` | リガンド重心の平均 | 共有ボックス中心。省略時は対象リガンド群の重心平均を自動採用。 |
| `--size S` | 22.5 | ボックス一辺（Å）。 |
| `--search MODE` | `tpe` | 探索モード（`tpe`/`drift`/`drift_qvina2`/`qvina2`/`vina`）。既定 `tpe` が低予算プリセット（n_runs=1+局所チェーン＋スキップ＋剛直軽量化）でスクリーニング推奨。 |
| `--n_runs N` | モード既定 | 1 リガンドあたり独立 run 数（速度/精度トレード）。 |
| `--prune_threshold T` | `1e9`（全件出力） | seed エネルギー > T を除外。ヒットのみ欲しければ `0.0` 等。 |
| `--workdir` / `--out` / `--bin` | `./ist_screening_out` / `<workdir>/ranking.tsv` / `target/release/ist-dock` | 中間・出力・バイナリのパス。 |

- 入力 glob は `dock` の既定出力 `*_ist_out.pdbqt` を**自動除外**（同一 dir に dock 出力が混在しても再ドッキングしない）。
- **共有ボックスは全リガンドが同一ポケット付近にある前提**。中心が不適切なら `--center` で明示すること。
- 手書きで済ませる場合の singleton マニフェスト（`pocket_center/size` を書かず `[box]` 共有）:
  ```toml
  [[cluster]]
  id = "sng_0"
  [cluster.seed]
  compound_id = "mol1"
  pdbqt = "mol1.pdbqt"     # 相対パス（マニフェスト基準）または絶対パス
  core_atoms = []           # 空＝全自由度ドッキング（＝dock と同一）
  ```

### 2.4 `pack` — クラスタライブラリのバイナリ化

```sh
./target/release/ist-dock pack <manifest.toml> <out.bin>
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
n_runs   = 8                     # 省略可（未指定ならモード別既定: tpe=1, vina/qvina2=8, drift/drift_qvina2=60）
scoring  = "grid"                # ★ 速度のため grid 推奨（既定は "direct"）
```

### `[run]` パラメータと既定値

| キー | 既定 | 説明 |
|---|---|---|
| `seed` | 42 | 乱数シード（決定論）。 |
| `n_runs` | モード別（下記） | 独立 run 数（網羅性）。**未指定時の既定は探索モードで変わる**: `tpe`=1（dock/screen 共通・局所チェーンで restart 代替）、`vina`/`qvina2`=8、`drift`/`drift_qvina2`=60。`vina`・本来の `qvina2` の n_runs は AutoDock Vina の exhaustiveness に対応（Vina 既定=8）。`drift`/`drift_qvina2` は軽い run の best-of-N なので既定 60。明示指定すればモードに依らずその値を使う（vina で大きな値を指定すると Vina の n_runs/8 倍遅くなる）。**tpe は局所チェーンで restart を代替し低 n_runs 化**（標準 Astex85×10seed で Vina exh=8 に肉薄しつつ eval 20×・wall ~9× 軽量。`docs/results.202607151422`）。より高精度が要るなら config で `n_runs` を上げる。 |
| `n_steps` | 0 | per-run の MC ステップ数。**0=自動**（可動原子数×torsion 上限で算出）。 |
| `scoring` | `"grid"` | 探索の分子間評価。`"grid"`=Vina グリッド補間（高速・**既定**）/ `"direct"`=直接原子間(SIMD)。 |
| `grid_margin` | 2.0 | grid 探索ボックスの余白（Å）。探索中に原子が箱外へはみ出しても正しいグリッド値を引く緩衝。小さいほど**構築が速く省メモリ**だが、境界近傍のボーダーライン標的が退行しうる（実測: 0 で一部退行、2 で全標的 RMSD<2Å 中立）。環境変数 `IST_GRID_MARGIN` で更に上書き可（デバッグ用）。 |
| `search` | `"tpe"` | 探索モード。**`"tpe"`=既定**（共有 TPE の低予算プリセット, n_runs=1+局所チェーン＋スキップ＋剛直軽量化, dock/screen 共通。下記）/ `"drift"`=ドリフト鎖 ILS（旧称 standard・決定論・高速）/ `"drift_qvina2"`=ドリフト鎖＋局所探索スキップ（ist 独自, n_runs=60）/ `"qvina2"`=**本来の QuickVina2 相当**（Vina 忠実 MC＋局所探索スキップ, exhaustiveness=8）/ `"vina"`=Vina 忠実 MC（毎ステップ収束 BFGS）。`--search` で上書き可。 |
| `qvina_cos` | 0.90 | `drift_qvina2`/`qvina2` の勾配方向一致閾値（cos）。低いほどスキップ多＝速度↑精度↓。`--qvina-cos` で上書き可。 |
| `hunt_cap` | 10.0 | 探索中の正エネルギー上限（Vina curl）。激突ポーズの暴走を抑え深い極小へ抜けやすくする。`1e9` で無効化。polish は常に無効（authentic）。 |
| `chain_iters` | 30 | ILS 各ステップの局所最適化反復（浅く保ち基底間をドリフト）。 |
| `polish_iters` | 20 | 代表ポーズの最終厳密 f64 精密化反復。 |
| `polish_modes` | 9 | 最終精密化にかける代表モード数。 |
| `n_modes` | 9 | 出力・保持する最大モード数（Vina 流マルチモデル）。 |
| `cluster_rmsd` | 2.0 | ポーズ RMSD クラスタリング閾値（Å）。 |
| `early_stop_patience` | 0 | 早期停止（0=無効）。収束ベース停止はハード標的を退行させうるため opt-in。 |
| `temp_init` / `temp_final` | 1.0 / 1.0 | Metropolis 温度（既定は固定温度。`init>final` で焼きなまし=opt-in）。 |
| `intra_14_scale` | 0.0 | リガンド分子内エネルギーで **1-4 ペア**（結合グラフ距離がちょうど 3）に掛ける係数。**0.0 = AutoDock Vina と同じ規則**（1-4 を数えない）。1.0 にすると 1-4 を 1-5 以降と同じ重みで数える。 |
| `intra_15_scale` | 1.0 | 分子内エネルギーで **1-5 以降のペア**（距離 4 以上）に掛ける係数。**1.0 = AutoDock Vina と同じ。** |
| `intra_14_repulsion_scale` | 0.0 | 1-4 ペアに**反発項だけ**を入れる係数（0.0 = 無効）。Vina の分子内項をそのまま 1-4 に入れると引力も一緒に入って折りたたみを促すため、立体障害だけを表現したいとき用。 |
| `torsion_scale` | 0.0 | ねじれポテンシャルの係数（0.0 = 無効 ＝ AutoDock Vina と同じ）。sp2-sp3 単結合などの回転障壁を加える。 |
| `torsion_sp2_sp3` | true | `torsion_scale > 0` のとき、sp2-sp3 単結合の規則を使うか。 |
| `torsion_polish_only` | false | ねじれポテンシャルを最終精密化だけに掛けるか（探索中は掛けない）。 |
| `rank_by_objective` | false | 候補の保持と最終順位を**全目的関数**（分子間＋箱の罰則＋分子内＋ねじれ）で付けるか。**既定 false** は AutoDock Vina と同じで、最終順位は分子間だけで付ける。 |

#### `tpe` モード（TPE=Tree-structured Parzen Estimator）
`--search tpe` / `search = "tpe"` で選択。TPEvina-2.0（PSOVina 派生）の TPE を移植したもの。各次元を
独立 Gaussian Parzen 混合でモデル化し、良観測（lower）近傍から候補をサンプル→ BFGS 局所最適化→観測更新を
繰り返す（前半はウォームアップの一様ランダム）。局所最適化・スコア・grid は他モードと共通機構を再利用。
- **決定論**: seeded `ChaCha8Rng` で駆動し**同一 seed でビット一致**。
- **既定＝共有観測モード（`tpe_shared=true`）**: 総予算 `n_runs×auto_n_steps` を **1 本の共有観測プール・単一
  モデルで逐次探索**（warmup 1 回のみ・前回結果を再利用）。独立多始点（各 run が別 warmup/観測/モデル）は初期
  依存性の小さい TPE では冗長なため、これを排して効率化する。Astex 実測（`docs/results.202607111622`）:
  - **同予算(n_runs=8)で旧・独立モードを上回る**（探索62 成功率 94.6→95.2%・meanScore −9.5703→−9.5790, 時間同等）。
  - **同精度なら ~1.3–1.5× 高速**（`n_runs=5–6` で独立 n_runs=8 の成功率に到達, 時間 −20〜30%）。
  - 利得源は**観測共有**そのもの。任意ノブ `tpe_reanchor_period`（0=単一アンカー[既定] / >0 で N 提案ごと探索的
    restart）・`tpe_coverage_weight`（0=無効[既定] / >0 で未踏域重点。Astex では利得なし）は既定 off。
  - meanRMSD は微増するが成功率は非退行〜改善（score-RMSD 乖離）。
- **ステップ予算**: `n_steps=0`（自動）で `auto_n_steps_tpe`（vina より多め＝モデル用ウォームアップ確保）。
- **ハイパーパラメータは config `[run]` で可変**（未指定なら TPEvina 準拠の既定）:
  `tpe_ei_candidates`(25, EI 候補数) / `tpe_startup_ratio`(0.05, warmup 比) / `tpe_step_ratio`(0.10, auto n_steps 係数・大=深) /
  `tpe_lower_cap`(10) / `tpe_greater_cap`(10, Parzen カーネル上限) / `tpe_density_reuse`(6, モデル再構築回数)。
  加えて `chain_iters`(35, 1 提案あたり BFGS 反復) / `n_runs`（tpe 既定 1。局所チェーンで restart を代替）。
- **局所チェーン（`tpe_local_chain` K）＝CLI 既定 ON**: 共有モードで各提案の初期 BFGS 後、competitive な basin
  （初期 energy が `best_so_far + tpe_local_chain_margin` 以下）に対し Drift 流の短い MC チェーン（mutate+BFGS+
  Metropolis を K 歩）を回して精密化＝MC の持続滞在(dwelling)を移植。**explore 律速で native が最深な標的の dwelling
  失敗を救済**（例 1lpz 8.15→1.8Å, aff −10.58/inter −14.60）。低 n_runs で独立 restart を代替し**省コスト化**する
  （`docs/results.202607121032`,`.121135`）。**CLI 既定プリセット（`tpe`）＝dock/screen 共通の単一プリセット**（旧
  dock tpe(n_runs=3) と旧 screen tpe_fast(n_runs=1) を 2026-07-15 に一本化。config の各 `tpe_*` / `n_runs` で上書き可）。
  既定値（10seed 全85 で **68.8%/−9.0757/284k ev/3.04s wall**＝Vina exh=8(70.8%) に肉薄・eval 20×/wall ~9× 軽量,
  `docs/results.202607151422`）:
  - **`n_runs=1` + `tpe_local_chain=15` + `tpe_local_chain_scale=0.75`**（<1.0 で変異幅縮小＝walk-off 抑制）。
  - **局所探索スキップ `tpe_skip_local=true`**（QuickVina2 相当）で冗長 basin への初期 BFGS を省く。skip はモデルを
    良 basin へ集中させるため margin を skip 対象別に分離する（下記）。`tpe_skip_cos=0.90`（勾配方向一致閾値・高いほど
    保守的＝スキップ少）。
  - **margin 分割**: `tpe_local_chain_margin=2.0`（skip 有効＝剛直側の狭ゲートで連鎖の過剰発火を抑制）／
    **`tpe_nonskip_local_chain_margin=4.0`**（skip 無効＝柔軟側の広ゲートで精度回収）。旧「skip 時は一律 margin=2.0 で
    精度非退行」は Astex24 小標本の誤りで全85では柔軟を巻き添えに −3.5pt 退行、剛直 2.0・柔軟 4.0 の分割が正解
    （`docs/results.202607141930`）。攻めるなら margin=1.0 や `tpe_local_chain=0`（連鎖無効）。
  - **skip の TORSDOF ゲート `tpe_skip_torsdof=7`**: `free_torsions.len() < 7` の剛直のみスキップし、中〜高自由度
    (T≥7)では無効化＝BFGS 温存。スキップの精度コストは高DOF に集中する（native 盆地が難で BFGS 省略が native 消失に
    直結, `docs/results.202607141810*`）。`0`=全リガンドに適用で旧挙動とビット一致。
  - **剛直の TORSDOF 条件付き軽量化**（`free_torsions.len() < tpe_rigid_torsdof=6` の剛直に適用）:
    (a) 局所チェーンを **`tpe_rigid_local_chain=8`** へ切替（剛直・低自由度デコイが Vina 関数の偽陽性深極小を掘り
    early 濃縮 EF1% を下げる問題への対策, `HANDOFF.ef1_diagnosis.md`）、(b) n_steps を **`scale(T)=min(1, tpe_rigid_step_scale
    + tpe_rigid_step_per_tors·T)`**（既定 `0.7 + 0.05·T`＝T=6 で満額 1.0）へ多段階縮小（`tpe_rigid_step_floor=200` が下限）。
    **当初 K=0/BASE=0.5 は剛直 native(1u4d/1ke5/1tow,T2-4)の pose を −1.6/−2.4pt 退行させたため K=8/BASE=0.7 を採用**
    （affinity＝スコアは中立でも RMSD は退行＝スコアのみの小標本評価は不可。剛直チェーンの wall コストはほぼ無い,
    `docs/results.202607141930`）。EF1%/throughput 最優先なら config で `tpe_rigid_local_chain=0` へ。
  - **ライブラリ既定（`TpeParams::default`）は全機構 OFF**（`tpe_local_chain=0` かつ `tpe_skip_local=false`, 剛直軽量化も
    OFF）＝素の共有 TPE とビット一致（再現性契約）。上記プリセットは **CLI 層でのみ** 適用し、config で `tpe_*` を明示
    すると上書きする。
- 品質は標的依存（深い漏斗の標的で良好、探索性の要る標的では drift に劣ることがある）。

**既定 `tpe` が既に低予算プリセット（`n_runs=1`＋局所チェーン＋スキップ＋剛直軽量化）＝スクリーニング推奨設定**。
精度を上げたい場合のみ config で `n_runs` を増やす（局所チェーンが restart を代替するため通常は 1 で十分）:
  ```toml
  [run]
  search  = "tpe"
  scoring = "grid"
  # n_runs = 1        # 既定（低予算）。精度最優先なら 2–3 に増やす（wall はほぼ線形に増加）
  ```

**旧・独立モード（`tpe_shared=false`）**: 従来の独立 n_runs 探索へ戻す（`docs/results.202607100834` とビット一致）。
この独立モードでの補足知見:
- **既定パラメータは精度パレート最適**（Astex OFAT, `docs/results.202607111515`）: ei/step/chain/n_runs を振っても
  既定の meanScore を上回れない（`chain_iters↑`/`step_ratio↑`/`n_runs↑` は逆効果か ROI 極小）。
- **速度プリセット「tpe-fast」**（独立モードで ~10% 高速・成功率非退行・meanScore ほぼ不変。共有既定では基本不要）:
  ```toml
  [run]
  search            = "tpe"
  scoring           = "grid"
  n_runs            = 8
  tpe_shared        = false
  tpe_ei_candidates = 8    # 既定 25 → 提案あたり密度評価を削減（速度の主レバー）
  tpe_density_reuse = 15   # 既定 6 → モデル再構築を増やし ei 削減の精度劣化を無コスト補償
  ```

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
  - **原子型カバレッジ**: Vina の 17 XS 型（C/N/O/S/P/ハロゲン/金属）に加え **`Si`（ケイ素）を第一級対応**
    （Vina 1.2 準拠 XS 半径 2.2Å・立体項のみ）、`W`（AutoDock 水和水擬原子）は採点除外。金属は `Met_D` 扱い。
  - **未知原子型は元素推定で近似フォールバック**（例 `Se→S`, `B→C`, 未知金属→`Met_D`）＝異常終了しない。
    近似した型は初回に stderr 警告を出す。元素すら推定不能な型のみエラー（`score`/`dock`）またはスキップ（`screen`）。
- **出力**: マルチモデル PDBQT（`dock`）/ ランキング TSV（`screen`）/ バイナリ `.bin`（`pack`）。
- 並列度: 環境変数 **`RAYON_NUM_THREADS`** でスレッド数を制御（既定は論理コア数）。決定論は保たれる。

---

## 5. `tools/preprocess/` — クラスタライブラリ前処理スクリプト

`ist-dock` エンジン自体は前処理を行わない（読込専用）。`screen` 用のクラスタライブラリ
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

生成した `library.toml` は `ist-dock pack` でバイナリ化できる（§2.4）。

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
crates/ist-core   原子型・Vina スコア・幾何・grid・SIMD カーネル … 純計算
crates/ist-io     PDBQT パーサ・TOML config・クラスタライブラリ I/O
crates/ist-dock   ligand(torsion木)・optimizer(BFGS)・search(MC)・energy・screen・phases
crates/ist-cli    バイナリ `ist-dock`（score / dock / screen / pack）
tools/             ist_screening.py（平坦/singleton スクリーニング実行, §2.3.2）
tools/preprocess/  クラスタライブラリ前処理スクリプト（build_library.py 等, §5）
configs/           注釈付きサンプル設定（dock.example.toml / screen.example.toml）
```

主要アルゴリズム: Iterated Local Search（MC＋BFGS, 解析的勾配）/ Vina 互換スコア（grid 補間 or 直接 SIMD）/
RMSD ポーズクラスタリング / scaffold 固定ドッキング（Kabsch 重ね合わせ）。
