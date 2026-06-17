# bench — Vina 比較検証ハーネス

ISCT-Dock のスコア/ドッキング結果を AutoDock Vina と比較して検証する。
当面の検証環境は **Arm Mac + OS X**（Linux 検証は後日）。

## 前提ツールのセットアップ

Homebrew の Vina は `brewsci/bio`（未信頼 tap）にあり `brew trust` が必要なため、
**conda-forge に統一**して導入する。meeko は 0.6+ が conda-forge では python<3.12 要求の
ため、meeko/gemmi のみ pip で入れる（base が python 3.13 のため）。実績手順:

```sh
# AutoDock Vina（CLI + python binding）と RDKit を conda-forge から
conda install -y -c conda-forge vina rdkit

# meeko は python 3.13 対応の最新版を pip で（gemmi も必要）
python3 -m pip install --upgrade meeko gemmi
```

導入確認（検証済みバージョン）:

```sh
vina --version   # AutoDock Vina v1.2.7
python3 -c "import rdkit, meeko, gemmi; print(rdkit.__version__, meeko.__version__, gemmi.__version__)"
# rdkit 2025.09.5 | meeko 0.7.1 | gemmi 0.7.5
```

> 注: 当初 base conda に meeko 0.5.0 が入ったが、rdkit 2025 系で削除された `rdkit.six` を
> 参照して壊れていたため、pip で meeko 0.7.1 へ更新して解消した。

## 検証の流れ

1. **M1 スコア一致ゲート**: `data/` の複合体について `vina --score_only` と
   `isct-dock score` を比較し、差 < 1e-2 kcal/mol を確認する。
2. **M2 ドッキング比較**: 同一リガンドを Vina / 本エンジンでドッキングし、ベストスコア差を記録。
   相互再採点（cross re-score）でスコア関数の正しさと探索能力を切り分ける。
3. **M3 クラスタ高速化検証**: Scaffold 拘束 vs 独立フルドッキング/Vina のスコア差、
   クラスタ棄却率、計算削減率をレポートする。

## スクリプト

- `prepare_pdb.py <PDBID> <LIG_RESNAME> <SMILES>` — RCSB から共結晶複合体を取得し、
  受容体/リガンド（結晶ポーズ）の PDBQT を `data/pdb/<id>/` に調製する。
- `make_screening.py` — 1 受容体（トリプシン）× 複数リガンド（active/decoy）の
  スクリーニングケースを `data/screening/trypsin/` に生成する。
- `make_testcase.py <SMILES> <prefix> [shift_x]` — 単純な合成テストケース（リガンドを
  並進・剛体化した受容体）を生成する。スコア関数の素早い確認用。
- `run_vina_compare.py <receptor.pdbqt> <ligand.pdbqt>` — `vina --score_only` と
  `isct-dock score` の分子間エネルギーを比較し、差 < 1e-2 で PASS（M1）。
- `run_redock.py <case_dir>` — 結晶ポーズを基準に isct-dock と Vina で再ドッキングし、
  ベストスコアと重原子 RMSD を比較（M2）。`<case_dir>` に receptor.pdbqt/ligand.pdbqt が必要。

評価用テストセットの一覧と検証結果は `data/README.md`（manifest）を参照。

再現例:
```sh
# 共結晶複合体（再ドッキング基準）
python3 bench/prepare_pdb.py 3PTB BEN "NC(=N)c1ccccc1"
python3 bench/run_vina_compare.py bench/data/pdb/3ptb/receptor.pdbqt bench/data/pdb/3ptb/ligand.pdbqt
# スクリーニングケース
python3 bench/make_screening.py
```

## M1 検証結果（2026-06-16 時点）

- **解析値の一致**: 孤立炭素 2 個（r=4.0 Å）の分子間エネルギーは手計算 −0.066113 と一致
  （`vina --score_only` も −0.066）。`crates/isct-io/tests/score_parity.rs` に恒久回帰テスト化。
- **現実的ポーズでの一致**: アセトアミノフェンを 10–12 Å 並進した非衝突配置で、Vina との差は
  約 3e-4 kcal/mol（Vina 出力の 3 桁精度とほぼ同等）→ PASS。
- **既知の差異**: 原子が深く衝突する配置では repulsion(d²) が支配的になり、Vina の事前計算
  スプライン補間 vs 当方の厳密計算の差で ~0.04 kcal/mol 程度ずれる。ドッキングでは衝突ポーズが
  棄却されるため実害はない。完全一致が必要なら将来 Vina のスプライン離散化を再現する。

## data/

再ドッキング用の小規模複合体セット（受容体 PDBQT・リガンド PDBQT）を置く。
生成物・大容量データは `.gitignore` で除外している（`make_testcase.py` で再生成可能）。
