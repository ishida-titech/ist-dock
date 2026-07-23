# Astex Diverse Set — 調製方法（bench/data/astex）

Astex Diverse Set 85 複合体の再ドッキング用データ。全て **`bench/make_astex.py`** で自動生成する。
本 README は調製方針の記録（データと同じ場所に残す）。

## 再現コマンド
```sh
# ADFRsuite の bin を PATH に通す（prepare_receptor / prepare_ligand）
export PATH=~/ADFRsuite-1.0/bin:$PATH
python3 bench/make_astex.py                 # 全 85（既存はスキップ）
python3 bench/make_astex.py --force 1iep …  # 指定標的を強制再生成
```
入力リスト:
- `bench/data/astex.list`         … 85 PDB コード
- `bench/data/astex.ligand.list`  … `pdb_code,primary_ligand`（正解リガンド CCD。**選択の第一根拠**）

## 各標的ディレクトリ `bench/data/astex/<pdbid>/`
| ファイル | 内容 |
|---|---|
| `receptor.pdbqt` | 受容体（蛋白＋機能的ヘテロ、下記方針）|
| `ligand.pdbqt`   | 結晶リガンド（RMSD 基準かつドッキング開始構造）|
| `config.toml`    | `[box] center/size` と `[run] seed=42`（既定探索予算を使用）|
| `info.json`      | ligand CCD・box 中心・重原子数・torsdof・保持した `cofactors` 一覧 |

## 出典・座標
- 構造は **RCSB PDB** から取得（第 1 モデル、altloc は ' ' か 'A' のみ採用）。
- box は **ADFR 論文流のリガンド毎可変**（`PADDING=4.0 Å`, `GRID_SPACING=0.375 Å`）:
  - 中心 = リガンド**重原子 bounding box の中心**（各面がリガンド末端から ≧4 Å 空く）
  - 一辺（各次元独立＝矩形） = (重原子範囲 + 2×4.0) を 0.375 Å の**偶数**格子点へ切り上げ
  - 実測レンジ 9.75–25.5 Å（中央値 15.75）。`config.toml` の `size = [sx, sy, sz]` に格納。

## リガンド選択
1. `astex.ligand.list` の primary ligand（CCD）に一致する HETATM 残基を採用（複数コピーは重原子最多）。
2. リストに無い場合のみ従来ヒューリスティック（blocklist 外・炭素含む・重原子 10–70 の最大 HETATM）。
- `prepare_ligand -l lig.pdb -o ligand.pdbqt -A hydrogens`（水素付加、回転結合自動検出、重原子座標は保持）。

## 受容体調製（**補酵素・金属を含める**）
`receptor_lines()` が受容体 PDB を構成し、`prepare_receptor` で PDBQT 化する。

**保持するもの**
- 標準アミノ酸の蛋白原子（ATOM 行すべて）
- 機能的ヘテロ原子: 触媒金属（ZN/MG/CA/MN/FE/HG…）、有機補因子（HEM/FAD/NADP/SAM/ADP/PLP/FPP…）、
  修飾残基（CME/CSO/CSD…）。**裸の金属イオンも含む**（[[../../../docs/results.202607080937]] で除外は差し引きゼロと判明、
  スクリーニング前処理との整合も踏まえ全含有を採用）。

**除去するもの**
- 水: `WATERS = {HOH, DOD, WAT}`
- 結晶化添加物 blocklist `JUNK_HET`（緩衝剤・凍結保護剤・PEG類・界面活性剤・糖・単純緩衝イオン NA/CL/K・
  無機オキソアニオン VO4/WO4/MO4 等）
- ドッキング対象リガンド（全コピー）

**pdbqt 化フラグ**
```
prepare_receptor -r protein.pdb -o receptor.pdbqt -A hydrogens -U nphs_lps_waters
```
- `-A hydrogens`: 極性水素付加＋Gasteiger 電荷（金属は電荷 0＝無害。Vina は電荷非依存）
- `-U nphs_lps_waters`: 非極性 H マージ・孤立電子対除去・水除去。**非標準残基は削除しない**＝補因子/金属を保持。

**カラム補正 `fix_columns()`**
ADFRsuite が altloc 残基へ付加した水素で残基名が 4 文字化（例 `1LYS`）し、以降のカラムが 1 桁右へずれる
不具合を補正（標準カラムで座標が parse できない ATOM/HETATM 行のみ、altloc 直後の余分な数字を除去。冪等）。

**AutoDock 非対応原子**
バナジン酸 VO4（原子型 V）等は Vina/isct 双方が parse 不能のため `JUNK_HET` で除外。
isct-core 側も未知金属型 V/Mo/W を `Metal` として受理するよう拡張済み（`crates/isct-core/src/atom_types.rs`）。

## スクリーニング前処理との整合
方針は共通（標準AA＋機能的ヘテロを保持、水＋結晶化添加物を除去、`prepare_receptor -A hydrogens
-U nphs_lps_waters`、リガンド重心 box）。詳細差分は [[../../../docs/findings.202607081000]] を参照。
- 現状の軽微差: 糖（NAG 等）と NA/K を Astex 側は除去（スクリーニングは「曖昧→保持」）。ポケットから遠く
  box 外のためドッキング結果には無影響。
- box 方式: Astex は **ADFR 流 padding=4.0 / spacing=0.375 のリガンド毎可変**を採用（従来の 20³ 固定から変更）。

## 品質・検証
- 全 85 が isct-dock で parse 可能、リガンドは `astex.ligand.list` と一致。
- ベンチ結果は `bench/data/logs/vina.exhaustiveness8.astex.*.tsv`、まとめは `docs/results.*.md` を参照
  （現行 canonical: 補酵素・金属含む＋ADFR 流ボックス版 `docs/results.202607081016.md`, Vina exh=8 で 72.5%）。
