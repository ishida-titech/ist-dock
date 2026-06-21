#!/usr/bin/env python3
"""isct-dock / vina の既存ポーズ（マルチ MODEL）を先頭 MODEL 抽出して vina 再採点し、
**共通スケール（vina --score_only）**で AUC/EF/BEDROC・energy を再計算する（再ドッキング不要）。

元の本実行では rescore がマルチ MODEL を採点できず native へフォールバックしていた
（isct=intermolecular のみ, vina=affinity）。本スクリプトで 3 手法を同一スケールに揃える。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "bench")
import screen_eval as se
import screen_metrics as M

se.VINA = "/home/ishida/miniconda3/bin/vina"
D = Path("bench/data/screen_eval/kif11")
WALL = {"isct-dock": 3449.0, "vina": 11504.5}  # 本実行のドッキング wall


def main():
    box = json.loads((D / "box.json").read_text())
    labels = {}
    for ln in (D / "labels.tsv").read_text().splitlines()[1:]:
        i, lab = ln.split("\t")
        labels[i] = 1 if lab == "active" else 0

    for m in ["isct-dock", "vina"]:
        rows = []
        for ln in (D / f"res_{m}.tsv").read_text().splitlines()[1:]:
            p = ln.split("\t")
            lid, native = p[0], (None if p[1] in ("None", "") else float(p[1]))
            pose = D / f"poses_{m}" / f"{lid}.pdbqt"
            rows.append((lid, native, WALL[m] / max(1, len(rows) + 1), str(pose) if pose.exists() else ""))
        print(f"{m}: {len(rows)} rows, {sum(1 for r in rows if r[3])} with pose", flush=True)
        rescored = se.rescore(rows, D / "receptor.pdbqt", box, 40)
        n_rs = sum(1 for v in rescored.values() if v is not None)
        print(f"  vina-rescored {n_rs}/{len(rescored)}", flush=True)
        summ = se.summarize(m, rows, labels, rescored, WALL[m])
        # native AUC も併記
        nat = {r[0]: r[1] for r in rows if r[0] in labels and r[1] is not None}
        ids = list(nat)
        summ["AUC_native"] = round(M.roc_auc([nat[i] for i in ids], [labels[i] for i in ids]), 4)
        (D / f"summary_{m}_rescored.json").write_text(json.dumps(summ, indent=2))
        print(json.dumps(summ, indent=2), flush=True)


if __name__ == "__main__":
    main()
