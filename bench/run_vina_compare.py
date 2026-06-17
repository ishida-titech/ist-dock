#!/usr/bin/env python3
"""`vina --score_only` と `isct-dock score` の分子間エネルギーを比較する（M1 ゲート）。

使い方:
    python3 bench/run_vina_compare.py <receptor.pdbqt> <ligand.pdbqt>
"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def vina_intermolecular(receptor: str, ligand: str) -> float:
    out = subprocess.run(
        [
            "vina", "--receptor", receptor, "--ligand", ligand,
            "--score_only", "--autobox",
        ],
        capture_output=True, text=True, check=True,
    ).stdout
    m = re.search(r"Ligand - Receptor\s*:\s*([-\d.]+)", out)
    if not m:
        sys.exit(f"could not parse vina output:\n{out}")
    return float(m.group(1))


def isct_intermolecular(receptor: str, ligand: str) -> float:
    out = subprocess.run(
        ["cargo", "run", "-q", "--release", "--bin", "isct-dock",
         "--", "score", receptor, ligand],
        capture_output=True, text=True, check=True, cwd=ROOT,
    ).stdout
    m = re.search(r"Intermolecular energy \(Ligand - Receptor\):\s*([-\d.]+)", out)
    if not m:
        sys.exit(f"could not parse isct-dock output:\n{out}")
    return float(m.group(1))


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    receptor, ligand = sys.argv[1], sys.argv[2]
    v = vina_intermolecular(receptor, ligand)
    i = isct_intermolecular(receptor, ligand)
    diff = i - v
    print(f"vina      : {v:+.5f} kcal/mol")
    print(f"isct-dock : {i:+.5f} kcal/mol")
    print(f"diff      : {diff:+.5f} kcal/mol")
    tol = 1e-2
    ok = abs(diff) < tol
    print(f"{'PASS' if ok else 'FAIL'} (tol={tol})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
