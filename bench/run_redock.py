#!/usr/bin/env python3
"""再ドッキング検証: isct-dock と Vina の再ドッキング RMSD・スコアを比較する。

結晶ポーズ（ligand.pdbqt）を基準に、ボックス中心を結晶リガンド重心とし、両エンジンで
ドッキングして重原子 RMSD とベストスコアを出す。

使い方:
    python3 bench/run_redock.py <case_dir>   # 例: bench/data/pdb/3ptb
case_dir に receptor.pdbqt, ligand.pdbqt が必要。
"""
import math
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "target" / "release" / "isct-dock"


def heavy_coords(pdbqt_text: str, model: int = 1) -> list[tuple[float, float, float]]:
    """指定 MODEL の重原子（末尾 AutoDock 型が H/HD でない）座標を出現順に返す。"""
    coords = []
    cur = 0
    for ln in pdbqt_text.splitlines():
        if ln.startswith("MODEL"):
            cur = int(ln.split()[1])
        if cur in (0, model) and ln.startswith(("ATOM", "HETATM")):
            atype = ln.split()[-1]
            if atype in ("H", "HD", "HS"):
                continue
            coords.append((float(ln[30:38]), float(ln[38:46]), float(ln[46:54])))
        if ln.startswith("ENDMDL") and cur == model:
            break
    return coords


def rmsd(a, b) -> float:
    assert len(a) == len(b), f"atom count mismatch {len(a)} vs {len(b)}"
    s = sum((ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2
            for (ax, ay, az), (bx, by, bz) in zip(a, b))
    return math.sqrt(s / len(a))


def centroid(coords):
    n = len(coords)
    return (sum(c[0] for c in coords) / n, sum(c[1] for c in coords) / n,
            sum(c[2] for c in coords) / n)


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    case = Path(sys.argv[1])
    rec = case / "receptor.pdbqt"
    lig = case / "ligand.pdbqt"
    crystal = heavy_coords(lig.read_text())
    cx, cy, cz = centroid(crystal)

    # --- Vina ---
    with tempfile.TemporaryDirectory() as td:
        vout = Path(td) / "vina_out.pdbqt"
        res = subprocess.run(
            ["vina", "--receptor", str(rec), "--ligand", str(lig),
             "--center_x", str(cx), "--center_y", str(cy), "--center_z", str(cz),
             "--size_x", "20", "--size_y", "20", "--size_z", "20",
             "--seed", "42", "--exhaustiveness", "8", "--out", str(vout)],
            capture_output=True, text=True, check=True)
        vina_score = float(re.search(r"^\s+1\s+([-\d.]+)", res.stdout, re.M).group(1))
        vina_rmsd = rmsd(heavy_coords(vout.read_text(), model=1), crystal)

    # --- isct-dock ---
    cfg = case / "config.toml"
    if not cfg.exists():
        cfg.write_text(
            f"[box]\ncenter = [{cx:.3f}, {cy:.3f}, {cz:.3f}]\nsize = [20.0, 20.0, 20.0]\n"
            "[run]\nseed = 42\nn_runs = 60\nn_steps = 60\n")
    out = case / "ligand_isct_out.pdbqt"
    res = subprocess.run(
        [str(BIN), "dock", str(rec), str(lig), "--config", str(cfg), "--out", str(out)],
        capture_output=True, text=True, check=True)
    isct_score = float(re.search(r"intermolecular energy\s*:\s*([-\d.]+)", res.stdout).group(1))
    isct_rmsd = rmsd(heavy_coords(out.read_text()), crystal)

    print(f"case: {case.name}")
    print(f"{'engine':10} {'best_score':>12} {'rmsd(Å)':>10}")
    print(f"{'vina':10} {vina_score:>12.3f} {vina_rmsd:>10.3f}")
    print(f"{'isct-dock':10} {isct_score:>12.3f} {isct_rmsd:>10.3f}")
    ok = isct_rmsd < 2.0
    print(f"{'PASS' if ok else 'FAIL'} (isct-dock RMSD < 2.0 Å)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
