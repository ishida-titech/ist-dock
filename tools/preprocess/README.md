# tools/preprocess — 前処理スクリプト（薄く用意）

エンジンは前処理を行わず読込専用。前処理は既存ツール（RDKit / Meeko）で実施する。
ここには薄い変換スクリプトのみを置く（M3 で本格化）。

予定スクリプト:

- `prepare_pdbqt.py` — RDKit/Meeko で受容体・リガンドを PDBQT 化（Vina と同条件）。
- `cluster_library.py` — RDKit の MCS/Murcko で化合物を Scaffold クラスタリングし、
  代表分子（Seed）＋子分子（Children: コア差分）のツリーを構築。
- `to_binary.py` — クラスタを ist-io が読むカスタムバイナリへ変換（スキーマは仕様 4.1）。
