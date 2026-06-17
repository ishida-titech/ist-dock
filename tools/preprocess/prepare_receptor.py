#!/usr/bin/env python3
"""DUD-E 受容体（PDB）を PDBQT 化し、結晶リガンドから探索ボックス中心を求める（M3 前処理）。

DUD-E の `bench/data/dude/<target>/` には `receptor.pdb`（PDBQT 未変換）と
`crystal_ligand.mol2`（box 中心算出用）がある。タンパク質（ATOM 行）を抽出して
`mk_prepare_receptor` で PDBQT 化し、結晶リガンドの重心を box 中心として返す。

使い方:
    python3 tools/preprocess/prepare_receptor.py <target> [--out-dir DIR] [--box-size S]
出力:
    <out-dir>/receptor.pdbqt  （既定 out-dir = bench/data/library/<target>）
    box 中心（stdout, JSON）
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DUDE = ROOT / "bench" / "data" / "dude"

# 標準アミノ酸（+ His プロトン化変種）。DUD-E PDB から水・非標準・水素を除く。
STD_RESIDUES = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "HID", "HIE", "HIP", "CYX", "CYM",
}


def _element_from_name(raw_name: str) -> str:
    """PDB 原子名（カラム 13–16）から元素記号を推定する（先頭の数字を除いた最初の英字）。"""
    s = raw_name.strip().lstrip("0123456789")
    return s[0].upper() if s else "C"


def clean_protein_pdb(src: Path, dst: Path) -> int:
    """DUD-E の素朴な PDB を整形する。

    DUD-E `receptor.pdb` はカラム 54 で切れ（occupancy/bfactor/element 欠如）、水・水素を含む。
    標準残基の重原子のみ残し、occupancy/bfactor と element カラム（77–78）を補って 80 桁化する。
    `mk_prepare_receptor -p` が水素・電荷を付け直すので入力水素は落とす。
    """
    out_lines = []
    for ln in src.read_text().splitlines():
        if not ln.startswith("ATOM"):
            continue
        resname = ln[17:20].strip()
        if resname not in STD_RESIDUES:
            continue
        elem = _element_from_name(ln[12:16])
        if elem == "H":
            continue
        head = ln[:54].ljust(54)
        out_lines.append(f"{head}  1.00  0.00          {elem:>2}")
    dst.write_text("\n".join(out_lines) + "\nEND\n")
    return len(out_lines)


def receptor_pdbqt(target: str, out_dir: Path) -> Path:
    """タンパク質を整形し mk_prepare_receptor で PDBQT 化する。"""
    src = DUDE / target / "receptor.pdb"
    if not src.exists():
        sys.exit(f"receptor not found: {src}")
    prot = out_dir / "protein.pdb"
    n = clean_protein_pdb(src, prot)
    if n == 0:
        sys.exit(f"no standard-residue atoms parsed from {src}")
    basename = str(out_dir / "receptor")
    subprocess.run(
        ["mk_prepare_receptor.py", "--read_pdb", str(prot), "-o", basename, "-p"],
        check=True, capture_output=True, text=True,
    )
    prot.unlink(missing_ok=True)
    return out_dir / "receptor.pdbqt"


def box_center_from_mol2(target: str) -> tuple[float, float, float]:
    """crystal_ligand.mol2 の @<TRIPOS>ATOM 座標の重心を返す。"""
    path = DUDE / target / "crystal_ligand.mol2"
    xs, ys, zs = [], [], []
    in_atoms = False
    for ln in path.read_text().splitlines():
        if ln.startswith("@<TRIPOS>"):
            in_atoms = ln.strip() == "@<TRIPOS>ATOM"
            continue
        if in_atoms and ln.strip():
            parts = ln.split()
            # mol2 ATOM: id name x y z type ...
            xs.append(float(parts[2]))
            ys.append(float(parts[3]))
            zs.append(float(parts[4]))
    n = len(xs)
    if n == 0:
        sys.exit(f"no atoms parsed from {path}")
    return (sum(xs) / n, sum(ys) / n, sum(zs) / n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("target")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--box-size", type=float, default=22.5)
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "bench" / "data" / "library" / args.target
    out_dir.mkdir(parents=True, exist_ok=True)

    pdbqt = receptor_pdbqt(args.target, out_dir)
    cx, cy, cz = box_center_from_mol2(args.target)
    info = {
        "receptor_pdbqt": str(pdbqt),
        "box_center": [round(cx, 3), round(cy, 3), round(cz, 3)],
        "box_size": [args.box_size] * 3,
    }
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
