#!/usr/bin/env python3
"""PDB の共結晶複合体を再ドッキング用の受容体/リガンド PDBQT に調製する。

手順:
  1. RCSB から PDB をダウンロード（キャッシュ）。
  2. 受容体 = タンパク質（ATOM 行のみ。水・イオン・リガンドを除外）→ mk_prepare_receptor。
  3. リガンド = 指定 HETATM 残基を抽出 → RDKit で SMILES テンプレートから結合次数を付与し
     水素を付加（結合ポーズの座標を保持）→ mk_prepare_ligand。
  4. 出力を bench/data/pdb/<id>/ に配置（ligand.pdbqt は結晶ポーズ = RMSD 基準）。

使い方:
    python3 bench/prepare_pdb.py <PDBID> <LIG_RESNAME> <LIG_SMILES>
例:
    python3 bench/prepare_pdb.py 3PTB BEN "NC(=N)c1ccccc1"
"""
import subprocess
import sys
import urllib.request
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "bench" / "data" / "pdb"


def download_pdb(pdb_id: str, dest: Path) -> None:
    if dest.exists():
        return
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    urllib.request.urlretrieve(url, dest)


def write_receptor(pdb_path: Path, out_pdbqt: Path) -> None:
    """タンパク質（ATOM 行のみ）を抽出し mk_prepare_receptor で PDBQT 化する。"""
    prot = out_pdbqt.with_name("protein.pdb")
    atom_lines = [
        ln for ln in pdb_path.read_text().splitlines() if ln.startswith("ATOM")
    ]
    prot.write_text("\n".join(atom_lines) + "\n")
    basename = str(out_pdbqt.with_suffix(""))
    subprocess.run(
        ["mk_prepare_receptor.py", "--read_pdb", str(prot), "-o", basename, "-p"],
        check=True, capture_output=True, text=True,
    )


def write_ligand(pdb_path: Path, resname: str, smiles: str, out_pdbqt: Path) -> None:
    """指定残基の HETATM を抽出し、結合次数・水素を付けて mk_prepare_ligand で PDBQT 化。"""
    het = [
        ln for ln in pdb_path.read_text().splitlines()
        if ln.startswith("HETATM") and ln[17:20].strip() == resname
    ]
    if not het:
        sys.exit(f"ligand {resname} not found in {pdb_path}")
    lig_pdb = out_pdbqt.with_name("ligand_raw.pdb")
    lig_pdb.write_text("\n".join(het) + "\nEND\n")

    raw = Chem.MolFromPDBFile(str(lig_pdb), sanitize=False, removeHs=True)
    if raw is None:
        sys.exit(f"RDKit could not read {lig_pdb}")
    template = Chem.MolFromSmiles(smiles)
    if template is None:
        sys.exit(f"invalid SMILES: {smiles}")
    mol = AllChem.AssignBondOrdersFromTemplate(template, raw)
    mol = Chem.AddHs(mol, addCoords=True)

    sdf = out_pdbqt.with_name("ligand.sdf")
    w = Chem.SDWriter(str(sdf))
    w.write(mol)
    w.close()
    subprocess.run(
        ["mk_prepare_ligand.py", "-i", str(sdf), "-o", str(out_pdbqt)],
        check=True, capture_output=True, text=True,
    )


def main() -> None:
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    pdb_id, resname, smiles = sys.argv[1], sys.argv[2], sys.argv[3]
    case = DATA / pdb_id.lower()
    case.mkdir(parents=True, exist_ok=True)
    pdb_path = case / f"{pdb_id.lower()}.pdb"
    download_pdb(pdb_id, pdb_path)
    write_receptor(pdb_path, case / "receptor.pdbqt")
    write_ligand(pdb_path, resname, smiles, case / "ligand.pdbqt")
    print(f"prepared {pdb_id}: {case}/receptor.pdbqt , {case}/ligand.pdbqt")


if __name__ == "__main__":
    main()
