#!/usr/bin/env python3
"""既存 smina シャード出力(out_*.sdf)のポーズを PDBQT 化→ vina 再採点し、
smina サマリを **vina-rescored 共通スケール**で再計算する（再ドッキング不要）。

元の run_smina は meeko 変換が sanitize=False 出力で失敗し native(minimizedAffinity)へ
フォールバックしていた。SanitizeMol を加えて変換し、isct-dock/vina と同一スケールに揃える。
"""
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, "bench")
from rdkit import Chem
from meeko import MoleculePreparation, PDBQTWriterLegacy
import screen_eval as se
import screen_metrics as M

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "bench" / "data" / "screen_eval" / "kif11"
SMINA_WALL = 11268.8  # bt6l6ljw3 のドッキング wall（コア飽和・公平条件）


def convert_one(a):
    """(lid, aff, molblock) -> pose を poses_smina に書き、(lid, aff, pose|"") を返す。"""
    lid, aff, mb = a
    pose = D / "poses_smina" / f"{lid}.pdbqt"
    try:
        mol = Chem.MolFromMolBlock(mb, removeHs=False, sanitize=False)
        Chem.SanitizeMol(mol)
        mh = Chem.AddHs(mol, addCoords=True)
        setups = MoleculePreparation().prepare(mh)
        pq, ok, _ = PDBQTWriterLegacy.write_string(setups[0])
        if ok:
            pose.write_text(pq)
            return (lid, aff, str(pose))
    except Exception:
        pass
    return (lid, aff, "")


def main():
    (D / "poses_smina").mkdir(exist_ok=True)
    box = json.loads((D / "box.json").read_text())
    labels = {}
    for ln in (D / "labels.tsv").read_text().splitlines()[1:]:
        i, lab = ln.split("\t")
        labels[i] = 1 if lab == "active" else 0

    # シャード出力からポーズ molblock を収集
    tasks = []
    for osdf in sorted((D / "smina_shards").glob("out_*.sdf")):
        supp = Chem.SDMolSupplier(str(osdf), removeHs=False, sanitize=False)
        for mol in supp:
            if mol is None:
                continue
            lid = mol.GetProp("_Name") if mol.HasProp("_Name") else None
            aff = float(mol.GetProp("minimizedAffinity")) if mol.HasProp("minimizedAffinity") else None
            tasks.append((lid, aff, Chem.MolToMolBlock(mol)))
    print(f"poses to convert: {len(tasks)}", flush=True)

    t0 = time.perf_counter()
    rows = []
    with Pool(40) as pool:
        for lid, aff, pose in pool.imap_unordered(convert_one, tasks, chunksize=16):
            rows.append((lid, aff, SMINA_WALL / max(len(tasks), 1), pose))
    n_pose = sum(1 for r in rows if r[3])
    print(f"converted {n_pose}/{len(rows)} poses to pdbqt in {time.perf_counter()-t0:.0f}s", flush=True)

    # vina 再採点（共通スケール）→ サマリ
    rescored = se.rescore(rows, D / "receptor.pdbqt", box, 40)
    n_rs = sum(1 for v in rescored.values() if v is not None)
    print(f"vina-rescored {n_rs}/{len(rescored)}", flush=True)
    summ = se.summarize("smina", rows, labels, rescored, SMINA_WALL)
    # native(minimizedAffinity)版 AUC も併記（参考）
    native_scores = {r[0]: r[1] for r in rows if r[0] in labels}
    ids = [i for i in native_scores if native_scores[i] is not None]
    summ["AUC_native_minaff"] = round(M.roc_auc([native_scores[i] for i in ids],
                                                [labels[i] for i in ids]), 4)
    (D / "summary_smina_rescored.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=2), flush=True)


if __name__ == "__main__":
    main()
