#!/usr/bin/env python3
"""1 受容体 × 複数リガンドのスクリーニング用ケースを生成する。

トリプシン(3ptb) を受容体に、S1 ポケットに結合しうるアミジン/アミン類（actives）と
中性芳香環（decoys）を混在させたリガンド集合を作る。探索ボックス中心は結晶リガンド
(BEN) の重心とする。実際のドッキングは M2/M3 のエンジンで行う（ここでは入力一式を用意）。

使い方:
    python3 bench/make_screening.py
出力:
    bench/data/screening/trypsin/{receptor.pdbqt, config.toml, labels.tsv, ligands/*.pdbqt}
"""
import shutil
import subprocess
import sys
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem

ROOT = Path(__file__).resolve().parent.parent
PDB = ROOT / "bench" / "data" / "pdb" / "3ptb"
OUT = ROOT / "bench" / "data" / "screening" / "trypsin"

# (名前, SMILES, ラベル) — active はトリプシン S1 ポケットに結合しうる正電荷性
LIGANDS = [
    ("benzamidine", "NC(=N)c1ccccc1", "active"),
    ("4-aminobenzamidine", "NC(=N)c1ccc(N)cc1", "active"),
    ("benzylamine", "NCc1ccccc1", "active"),
    ("4-guanidinotoluene", "Cc1ccc(NC(=N)N)cc1", "active"),
    ("benzene", "c1ccccc1", "decoy"),
    ("phenol", "Oc1ccccc1", "decoy"),
    ("toluene", "Cc1ccccc1", "decoy"),
    ("naphthalene", "c1ccc2ccccc2c1", "decoy"),
]


def ligand_pdbqt(name: str, smiles: str, out_dir: Path) -> None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        sys.exit(f"invalid SMILES for {name}: {smiles}")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 0xC0FFEE
    if AllChem.EmbedMolecule(mol, params) != 0:
        sys.exit(f"embed failed for {name}")
    AllChem.MMFFOptimizeMolecule(mol)
    sdf = out_dir / f"{name}.sdf"
    w = Chem.SDWriter(str(sdf))
    w.write(mol)
    w.close()
    subprocess.run(
        ["mk_prepare_ligand.py", "-i", str(sdf), "-o", str(out_dir / f"{name}.pdbqt")],
        check=True, capture_output=True, text=True,
    )
    sdf.unlink()


def box_center_from_reference(ref_pdbqt: Path) -> tuple[float, float, float]:
    xs, ys, zs = [], [], []
    for ln in ref_pdbqt.read_text().splitlines():
        if ln.startswith(("ATOM", "HETATM")):
            xs.append(float(ln[30:38]))
            ys.append(float(ln[38:46]))
            zs.append(float(ln[46:54]))
    n = len(xs)
    return (sum(xs) / n, sum(ys) / n, sum(zs) / n)


def main() -> None:
    if not (PDB / "receptor.pdbqt").exists():
        sys.exit("先に bench/prepare_pdb.py 3PTB BEN ... で 3ptb を調製してください")
    lig_dir = OUT / "ligands"
    lig_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(PDB / "receptor.pdbqt", OUT / "receptor.pdbqt")

    cx, cy, cz = box_center_from_reference(PDB / "ligand.pdbqt")
    (OUT / "config.toml").write_text(
        "# トリプシン S1 ポケット（結晶 BEN の重心）への複数リガンドドッキング設定\n"
        '[receptor]\n'
        'pdbqt = "receptor.pdbqt"\n\n'
        "[box]\n"
        f"center = [{cx:.3f}, {cy:.3f}, {cz:.3f}]\n"
        "size = [20.0, 20.0, 20.0]\n\n"
        "[run]\n"
        "seed = 42\n"
    )

    labels = ["name\tlabel\tpdbqt"]
    for name, smiles, label in LIGANDS:
        ligand_pdbqt(name, smiles, lig_dir)
        labels.append(f"{name}\t{label}\tligands/{name}.pdbqt")
    (OUT / "labels.tsv").write_text("\n".join(labels) + "\n")
    print(f"screening case: {OUT} ({len(LIGANDS)} ligands, box center=({cx:.2f},{cy:.2f},{cz:.2f}))")


if __name__ == "__main__":
    main()
