#!/usr/bin/env python3
"""DUD-E 標的のスクリーニング評価用 前処理（時間計測込み）。

`bench/data/dude/<target>/{receptor.pdb, crystal_ligand.mol2, actives_final.ism, decoys_final.ism}` から:
  - receptor.pdbqt（mk_prepare_receptor）と box 中心（結晶リガンド重心）
  - 各リガンドの 3D 生成（RDKit ETKDG+MMFF）→ per-ligand PDBQT（Meeko, Vina/ISCT-dock 用）
  - 全リガンド 1 SDF（all.sdf, Smina 一括入力用）
  - labels.tsv（id\tlabel: active/decoy）、box.json、prep_times.json
を生成する。前処理時間（受容体/リガンド）を記録する（速度比較の参考情報）。

使い方:
    python3 bench/prep_dude_ligands.py kif11 [--sample 200] [--jobs 20] [--box-size 22.5]
"""
import argparse
import json
import subprocess
import time
from multiprocessing import Pool
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem
from meeko import MoleculePreparation, PDBQTWriterLegacy

ROOT = Path(__file__).resolve().parent.parent
DUDE = ROOT / "bench" / "data" / "dude"


def read_ism(path: Path):
    """DUD-E .ism: 'SMILES ID [extra...]'。(id, smiles) を返す。"""
    recs = []
    for ln in path.read_text().splitlines():
        p = ln.split()
        if len(p) >= 2:
            recs.append((p[1], p[0]))  # (id, smiles)
    return recs


def prep_one(args):
    """1 リガンド: SMILES→3D→PDBQT 文字列＋SDF mol block。失敗時 None。"""
    lid, smiles, label = args
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return (lid, label, None, None)
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42  # 決定論的 3D
        if AllChem.EmbedMolecule(mol, params) != 0:
            return (lid, label, None, None)
        try:
            AllChem.MMFFOptimizeMolecule(mol)
        except Exception:
            pass
        mol.SetProp("_Name", lid)
        # PDBQT（Meeko, Python API）
        setups = MoleculePreparation().prepare(mol)
        pdbqt, ok, _err = PDBQTWriterLegacy.write_string(setups[0])
        if not ok:
            return (lid, label, None, None)
        sdf = Chem.MolToMolBlock(mol)
        return (lid, label, pdbqt, sdf)
    except Exception:
        return (lid, label, None, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target")
    ap.add_argument("--sample", type=int, default=0, help="0=全件。>0 で actives 一部＋decoys でサンプル")
    ap.add_argument("--jobs", type=int, default=20)
    ap.add_argument("--box-size", type=float, default=22.5)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    src = DUDE / args.target
    out = Path(args.out_dir) if args.out_dir else ROOT / "bench" / "data" / "screen_eval" / args.target
    (out / "ligands").mkdir(parents=True, exist_ok=True)

    # --- 受容体 prep（時間計測）---
    t0 = time.perf_counter()
    rec = subprocess.run(
        ["python3", str(ROOT / "tools" / "preprocess" / "prepare_receptor.py"), args.target,
         "--out-dir", str(out), "--box-size", str(args.box_size)],
        capture_output=True, text=True, cwd=ROOT)
    rec_t = time.perf_counter() - t0
    if rec.returncode != 0:
        raise SystemExit(f"receptor prep failed:\n{rec.stderr}")
    info = json.loads(rec.stdout)  # {receptor_pdbqt, box_center:[...], box_size:[...]}
    box = {"center": info["box_center"], "size": info["box_size"]}
    (out / "box.json").write_text(json.dumps(box))

    # --- リガンド集合（label 付き）---
    actives = [(i, s, "active") for i, s in read_ism(src / "actives_final.ism")]
    decoys = [(i, s, "decoy") for i, s in read_ism(src / "decoys_final.ism")]
    if args.sample > 0:
        n_act = min(max(args.sample // 10, 1), len(actives))
        actives = actives[:n_act]
        decoys = decoys[: max(args.sample - n_act, 0)]
    records = actives + decoys
    print(f"{args.target}: actives={len(actives)} decoys={len(decoys)} total={len(records)} (jobs={args.jobs})")

    # --- リガンド prep（並列・時間計測）---
    t0 = time.perf_counter()
    n_ok = n_fail = 0
    lig_dir = out / "ligands"
    sdf_path = out / "all.sdf"
    labels = ["id\tlabel"]
    with open(sdf_path, "w") as sdf_f, Pool(args.jobs) as pool:
        for lid, label, pdbqt, sdf in pool.imap_unordered(prep_one, records, chunksize=16):
            if pdbqt is None:
                n_fail += 1
                continue
            (lig_dir / f"{lid}.pdbqt").write_text(pdbqt)
            sdf_f.write(sdf)
            sdf_f.write("$$$$\n")
            labels.append(f"{lid}\t{label}")
            n_ok += 1
    lig_t = time.perf_counter() - t0
    (out / "labels.tsv").write_text("\n".join(labels) + "\n")

    times = {
        "receptor_prep_s": round(rec_t, 2),
        "ligand_prep_s": round(lig_t, 2),
        "n_ok": n_ok, "n_fail": n_fail,
        "per_ligand_ms": round(1000 * lig_t / max(n_ok, 1), 1),
        "jobs": args.jobs,
    }
    (out / "prep_times.json").write_text(json.dumps(times, indent=2))
    print(json.dumps(times, indent=2))
    print(f"out: {out}")


if __name__ == "__main__":
    main()
