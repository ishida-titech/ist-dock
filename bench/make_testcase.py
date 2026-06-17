#!/usr/bin/env python3
"""検証用 PDBQT を生成する。

実リガンドを RDKit で 3D 化 → Meeko で PDBQT 化し、その原子を並進＋torsion木除去で
剛体化したものを「受容体」とする。Meeko が付けた AutoDock 型をそのまま使うため、
現実的な原子型分布で `vina --score_only` と比較できる。

使い方:
    python3 bench/make_testcase.py "CC(=O)Nc1ccccc1O" bench/data/case1
出力:
    <prefix>_ligand.pdbqt, <prefix>_receptor.pdbqt
"""
import subprocess
import sys
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem

# 受容体を作るための既定の並進量（Å, x方向）。衝突を避けつつ一部ペアを 8 Å 以内に残す。
DEFAULT_SHIFT_X = 9.0


def build_ligand_pdbqt(smiles: str, sdf_path: Path, pdbqt_path: Path) -> None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        sys.exit(f"invalid SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 0xC0FFEE
    if AllChem.EmbedMolecule(mol, params) != 0:
        sys.exit("3D embedding failed")
    AllChem.MMFFOptimizeMolecule(mol)
    writer = Chem.SDWriter(str(sdf_path))
    writer.write(mol)
    writer.close()
    subprocess.run(
        ["mk_prepare_ligand.py", "-i", str(sdf_path), "-o", str(pdbqt_path)],
        check=True,
    )


def build_receptor_from_ligand(
    ligand_pdbqt: Path, receptor_pdbqt: Path, shift_x: float
) -> None:
    """リガンド PDBQT の ATOM 行を並進し、torsion 木を除いた剛体受容体を書く。"""
    out_lines = []
    serial = 0
    for line in ligand_pdbqt.read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")):
            serial += 1
            x = float(line[30:38]) + shift_x
            y = float(line[38:46])
            z = float(line[46:54])
            # 座標カラムだけ置き換え、電荷・型などの行末はそのまま保持する。
            new = (
                f"ATOM  {serial:>5} {line[12:30]}"
                f"{x:8.3f}{y:8.3f}{z:8.3f}{line[54:]}"
            )
            out_lines.append(new)
    out_lines.append("TER")
    receptor_pdbqt.write_text("\n".join(out_lines) + "\n")


def main() -> None:
    if len(sys.argv) not in (3, 4):
        sys.exit(__doc__)
    smiles, prefix = sys.argv[1], Path(sys.argv[2])
    shift_x = float(sys.argv[3]) if len(sys.argv) == 4 else DEFAULT_SHIFT_X
    prefix.parent.mkdir(parents=True, exist_ok=True)
    sdf = prefix.with_name(prefix.name + "_ligand.sdf")
    lig = prefix.with_name(prefix.name + "_ligand.pdbqt")
    rec = prefix.with_name(prefix.name + "_receptor.pdbqt")
    build_ligand_pdbqt(smiles, sdf, lig)
    build_receptor_from_ligand(lig, rec, shift_x)
    print(f"wrote {lig}")
    print(f"wrote {rec} (shift_x={shift_x})")


if __name__ == "__main__":
    main()
