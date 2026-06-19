#!/usr/bin/env python3
"""易標的ガード: isct 単独で再ドッキング best モード RMSD を出す（grid, n_steps=auto）。

PLAN.SEARCH_DRIFT の非退行ガード用。Vina を回さない（測定干渉を避ける）。
  python3 bench/guard_redock.py 3ptb 1stp 1iep        # seed=42
  python3 bench/guard_redock.py 1iep --seeds 1 2 3 4 5 # pdb/1iep 5 seed
"""
import argparse
import subprocess
import tempfile
import time
import tomllib
from pathlib import Path

from run_redock import heavy_coords, rmsd

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "target" / "release" / "isct-dock"


def run_case(case_dir: Path, seed: int) -> tuple[float, float]:
    rec = case_dir / "receptor.pdbqt"
    lig = case_dir / "ligand.pdbqt"
    crystal = heavy_coords(lig.read_text())
    box = tomllib.loads((case_dir / "config.toml").read_text())["box"]
    c, s = box["center"], box["size"]
    with tempfile.TemporaryDirectory() as td:
        cfg = Path(td) / "run.toml"
        cfg.write_text(
            f"[box]\ncenter = [{c[0]}, {c[1]}, {c[2]}]\nsize = [{s[0]}, {s[1]}, {s[2]}]\n"
            f"[run]\nseed = {seed}\nn_runs = 60\nn_steps = 0\nscoring = \"grid\"\n")
        out = Path(td) / "out.pdbqt"
        t0 = time.perf_counter()
        subprocess.run([str(BIN), "dock", str(rec), str(lig), "--config", str(cfg),
                        "--out", str(out)], capture_output=True, text=True, check=True)
        dt = time.perf_counter() - t0
        best = rmsd(heavy_coords(out.read_text(), model=1), crystal)
    return best, dt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cases", nargs="+")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    args = ap.parse_args()
    for name in args.cases:
        case_dir = ROOT / "bench" / "data" / "pdb" / name
        for seed in args.seeds:
            best, dt = run_case(case_dir, seed)
            flag = "OK " if best < 2.0 else "** "
            print(f"{flag}{name:6s} seed={seed:<3d} best_rmsd={best:5.2f}  {dt:5.1f}s")


if __name__ == "__main__":
    main()
