# CLAUDE.md — ISCT-Dock プロジェクト

## プロジェクト概要
- **ISCT-Dock**: 1000万化合物規模を狙う高速・決定論的分子ドッキングスクリーニングエンジン。
  仕様は [isct-dock.md](isct-dock.md)。機械学習を使わず幾何・物理化学計算で行う。

## 主要ドキュメント（必ず参照）
- [PLAN.md](PLAN.md) — 開発計画・マイルストーン（計画。不変寄り）。
- [PROGRESS.md](PROGRESS.md) — 進捗（マイルストーン状態・実装/検証状況）。**living document＝機能追加・検証ごとに随時更新（追記）すること。**
- [EVALUATION.md](EVALUATION.md) — 評価方法と使用するデータについて記載する。評価結果はこちらに記載しないこと。
- [PORTING.md](PORTING.md) — 環境移行（Mac→Linux）の手順・OS 差分・引継ぎ詳細。変更しないこと。

## ログ
- 評価を行うたびに、必ずその結果をdocs/以下に results.YYYYMMDDHHMM.mdとしてマークダウンで記録をすること
- 議論を行ったり、解析の結果新たな事実がわかった場合、必ずdocs/以下に findings.YYYYMMDDHHMM.mdとしてマークダウンで記録をすること
- これらのログ（docs/results.*・docs/findings.*）は一度作成したら更新しないこと。
- **進捗は [PROGRESS.md](PROGRESS.md) に随時反映（こちらは更新可・追記する living document）。**

## リポジトリ構成
```
crates/isct-core   原子型・Vina スコア・幾何・(将来)PCA   … 純計算
crates/isct-io     PDBQT パーサ・TOML config・(将来)クラスタバイナリ
crates/isct-dock   ligand(torsion木)・optimizer(BFGS)・search(MC)・energy・phases(M3未実装)
crates/isct-cli    バイナリ `isct-dock`（score / dock / screen）
bench/             検証スクリプト（Python: vina/rdkit/meeko 利用）と data/
tools/preprocess/  前処理スクリプト置き場（M3 で本格化）
```

## よく使うコマンド
```sh
cargo build --release
cargo test
cargo clippy --all-targets          # 警告ゼロを維持
./target/release/isct-dock score <receptor.pdbqt> <ligand.pdbqt>
./target/release/isct-dock dock  <receptor.pdbqt> <ligand.pdbqt> --config <run.toml> [--out X] [--local-only]
```
検証ハーネス（要 vina/rdkit/meeko）: `bench/run_vina_compare.py`(score一致), `bench/run_redock.py`(再ドッキング),
`bench/prepare_pdb.py`/`make_screening.py`/`get_dude.py`/`analyze_dude_clusters.py`(データ生成・解析)。


