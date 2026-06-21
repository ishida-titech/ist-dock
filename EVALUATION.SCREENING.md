# EVALUATION.SCREENING.md — 大規模スクリーニング性能評価（設計・方法論）

ISCT-Dock の実用スクリーニング性能を外部ツールと比較評価する**設計書**（方法論・指標・手順・ツール設定）。
評価**結果は記載しない**（結果は `docs/results.YYYYMMDDHHMM.md` に記録＝[EVALUATION.md](EVALUATION.md) と同流儀）。
進捗は [PROGRESS.md](PROGRESS.md)。

## 1. 目的・前提

- M1–M4 完了後、ISCT-Dock の **virtual screening 性能**を Vina/Smina（GWOVina は後続）と比較する。
- **全手法が ~AutoDock Vina スコア関数を共有**（Vina / Smina default=vina / ISCT=Vina 互換 / GWOVina=Vina scoring）。
  → 差は**探索アルゴリズムのみ**。よって「探索が見つけたポーズのエネルギー」を主要比較量に加える。
- 規模は **~1 万件オーダーの DUD-E から開始**（kif11）→ 後日 akt1/hivrt 等へ拡大 → 将来 10M 本番。

## 2. 比較手法（本フェーズ）

| 手法 | 実行 | 探索 | 備考 |
|---|---|---|---|
| ISCT-dock | `isct-dock dock` を 1 リガンドずつ（grid・全自由度） | ドリフト鎖 ILS | 探索アルゴ比較の本命 |
| ISCT-screen | `isct-dock screen`（MCS クラスタ・scaffold 再利用） | seed 全探索＋子 R 基最適化 | ISCT 独自スループット。energy は**拘束劣化**（参考扱い・注記） |
| Vina | `vina --exhaustiveness 8 --cpu N` | MC + BFGS | ベースライン |
| Smina | 全リガンド 1 ファイル一括 `smina -l all.sdf --cpu N`（grid 1 回・最速構成） | MC + BFGS | 複数リガンド対応の最速構成 |
| GWOVina | （後続フェーズ・ソースビルド） | Grey Wolf Optimizer | 本フェーズ未実施 |

## 3. データセット・前処理

- 標的: **DUD-E kif11**（actives 116 / decoys 6850 = **6966 リガンド**）。`bench/data/dude/kif11/`。
- 受容体: `receptor.pdb` → PDBQT（`mk_prepare_receptor` / `tools/preprocess/prepare_receptor.py`）。
- 探索ボックス: `crystal_ligand.mol2` 重心を中心、20–22.5 Å 立方。全手法で同一。
- リガンド: `actives_final.ism` / `decoys_final.ism`（SMILES）→ 3D（RDKit ETKDG）→ PDBQT（Meeko `mk_prepare_ligand`）。
  - per-ligand 個別 PDBQT（Vina/ISCT-dock 用）＋ **Smina 用に全リガンド 1 SDF**（一括入力）。
  - ISCT-screen 用: `tools/preprocess/build_library.py` で MCS クラスタリング＋**コア拘束 embed**→ マニフェスト→
    `isct-dock pack`（バイナリ）。
- ラベル: active=1 / decoy=0（prep 時に `labels.tsv` を生成）。

### 前処理時間の記録（参考情報）
速度比較の**参考**として、以下を別途計測・記録する（ドッキング wall とは分離）:
- 受容体 prep（1 回）。
- 全リガンド prep（SMILES→3D→PDBQT、並列、件数あたり時間も）。
- ISCT-screen の MCS クラスタリング＋拘束 embed ＋ pack。
- 各エンジンの grid 構築（Smina=一括 1 回 / ISCT-screen=既定ボックス共有 / ISCT-dock=per-ligand / Vina=内部）。

## 4. 評価指標

### (a) スクリーニング濃縮 — 各手法の **native スコア**でランク
- **ROC AUC**: actives を上位に並べる識別力（0.5=ランダム, 1.0=完全）。
- **EF@1% / EF@5%**: 上位 1% / 5% に含まれる active 比率 ÷ 全体 active 比率（濃縮係数。創薬で最重要）。
- **BEDROC(α=20)**: 早期濃縮（上位への偏り）を強調する指標。
- いずれも「より低いスコア（より良い結合）＝上位」。

### (b) 探索ポーズのエネルギー — **同一スケールで探索のみを比較**
- 各手法のベストポーズを **`vina --score_only` で再採点**し共通スケール化（スコア実装差を除去＝探索の優劣のみ）。
  - active/decoy 別のエネルギー分布（mean/median/箱ひげ）。
  - **手法間 per-ligand 比較**: ISCT が他手法 ≤（同等以下のエネルギーに到達）の割合。
- ISCT-dock の native energy（intermolecular）も併記。**ISCT-screen は拘束 energy のため参考**（R 基のみ最適化で
  劣るのは設計通り、[docs/results.202606172136](docs/results.202606172136.md)）。

### (c) 速度
- ドッキング**総 wall**・**スループット（ligands/sec）**・per-ligand 平均。スレッド数・exhaustiveness 等の条件を明記。
- **前処理時間**（§3）を参考併記。Smina の grid 一括・ISCT-screen の grid 再利用といった最速化要因も注記。

## 5. 手順（再現コマンド）

```sh
# 0) ツール: smina 導入（conda-forge / bioconda）。vina/rdkit/meeko は導入済。
# 1) 前処理（時間計測込み）
python3 bench/prep_dude_ligands.py kif11            # receptor PDBQT, per-ligand PDBQT, all.sdf, labels.tsv
python3 tools/preprocess/build_library.py kif11 ... # ISCT-screen 用クラスタライブラリ
./target/release/isct-dock pack <manifest> kif11.bin
# 2) 評価ハーネス（各手法実行→再採点→指標）
python3 bench/screen_eval.py kif11 --methods isct-dock isct-screen vina smina [--sample 200]
# → native_score / rescore_vina / wall を収集し AUC/EF/BEDROC・energy・速度を算出
```

- 段階実行: まず `--sample 200` でパイプライン・指標を疎通検証 → kif11 全 6966 で本実行（数時間・background）。

## 6. 出力・スコープ

- 本書 = **方法論のみ**。**結果は `docs/results.YYYYMMDDHHMM.md`**（AUC/EF/BEDROC 表・energy 分布・速度＋前処理）。
- 知見は `docs/findings.*`、進捗は `PROGRESS.md` に追記。
- スコープ外（後日）: **GWOVina** 追加、akt1/hivrt 等への規模拡大、10M 規模本番。

## 7. 注意・公平性

- ボックス・exhaustiveness（探索努力）を可能な限り揃える。Smina は最速構成（一括）だが探索努力は Vina と同等。
- 決定論: ISCT は固定 seed で決定論。Vina/Smina は seed 固定でも探索は確率的（複数 seed 平均は規模次第で検討）。
- ISCT-screen はスループット最適化（scaffold 再利用）で**探索アルゴリズムが異なる**ため、energy/enrichment は
  per-ligand dock とは別枠で解釈する（拘束による劣化と引き換えの高速化）。
