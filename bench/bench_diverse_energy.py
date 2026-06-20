#!/usr/bin/env python3
"""多様データセットで isct-dock（現行=ドリフト鎖）のみを再ドッキングし、RMSD・時間に加えて
**結合エネルギー**を捕捉する。Vina は再実行せず、別途の以前の結果（results.*.md の表）を流用する。

各 target × 3 seed（1/42/123）。box=結晶リガンド重心・20Å、20 スレッド、grid・n_runs=8・auto step。
- isct_E: isct の docked best pose の intermolecular energy（kcal/mol, Vina 互換スケール=M1 検証, 平均）。
- cryst_E: 結晶ポーズの intermolecular energy（`isct-dock score` で算出, 同一 exact スケール）。
出力 tsv 列に isct_E, cryst_E を追加。Vina 列はマージ側（doc）から補う。

使い方: python3 bench/bench_diverse_energy.py [--targets ...] > out.tsv
"""
import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path

from run_redock import heavy_coords, rmsd

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "target" / "release" / "isct-dock"
DIVERSE = ROOT / "bench" / "data" / "diverse"
THREADS = 20
N_RUNS = 8

RE_INTER = re.compile(r"intermolecular energy\s*:\s*([-\d.]+)")
RE_SCORE_INTER = re.compile(r"Intermolecular energy \(Ligand - Receptor\):\s*([-\d.]+)")


def isct_dock(d: Path, center, size, seed, out: Path):
    search = os.environ.get("ISCT_SEARCH", "standard")  # standard / qvina2
    qcos = os.environ.get("ISCT_QVINA_COS", "0.95")     # qvina2 の cos 閾値
    cfg = out.with_suffix(".toml")
    cfg.write_text(
        f"[box]\ncenter = [{center[0]}, {center[1]}, {center[2]}]\n"
        f"size = [{size[0]}, {size[1]}, {size[2]}]\n"
        f'[run]\nseed = {seed}\nn_runs = {N_RUNS}\nscoring = "grid"\n'
        f'search = "{search}"\nqvina_cos = {qcos}\n'
    )
    env = {**os.environ, "RAYON_NUM_THREADS": str(THREADS)}
    t0 = time.perf_counter()
    r = subprocess.run([str(BIN), "dock", str(d / "receptor.pdbqt"), str(d / "ligand.pdbqt"),
                        "--config", str(cfg), "--out", str(out)],
                       check=True, capture_output=True, text=True, env=env)
    dt = time.perf_counter() - t0
    m = RE_INTER.search(r.stdout)
    return dt, float(m.group(1)) if m else float("nan")


def crystal_energy(d: Path):
    r = subprocess.run([str(BIN), "score", str(d / "receptor.pdbqt"), str(d / "ligand.pdbqt")],
                       check=True, capture_output=True, text=True)
    m = RE_SCORE_INTER.search(r.stdout)
    return float(m.group(1)) if m else float("nan")


def agg(vals):
    return sum(vals) / len(vals) if vals else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 42, 123])
    ap.add_argument("--targets", nargs="*", default=None)
    args = ap.parse_args()
    tmp = Path("/tmp/diverse_energy"); tmp.mkdir(exist_ok=True)

    if args.targets:
        targets = [DIVERSE / t for t in args.targets]
    else:
        targets = sorted(p.parent for p in DIVERSE.glob("*/info.json"))
    cols = ["pdbid", "lig", "heavy", "tors", "rings", "maxring", "macro", "MW",
            "isct_rmsd", "isct_ok", "isct_t", "isct_E", "cryst_E"]
    print("\t".join(cols))
    for d in targets:
        info = json.loads((d / "info.json").read_text())
        center = info["center"]; size = [20.0, 20.0, 20.0]
        crystal = heavy_coords((d / "ligand.pdbqt").read_text())
        ir, it, ie = [], [], []
        for seed in args.seeds:
            try:
                out = tmp / f"{info['pdbid']}_{seed}.pdbqt"
                dt, e = isct_dock(d, center, size, seed, out)
                ir.append(rmsd(heavy_coords(out.read_text()), crystal))
                it.append(dt); ie.append(e)
            except Exception:
                pass
        try:
            ce = crystal_energy(d)
        except Exception:
            ce = float("nan")
        iok = sum(1 for x in ir if x < 2.0)
        row = [info["pdbid"], info["ligand"], info["heavy_atoms"], info["torsdof"],
               info["n_rings"], info["max_ring"], "Y" if info["macrocycle"] else "n", info["mw"],
               f"{agg(ir):.2f}" if ir else "NA", f"{iok}/{len(ir)}", f"{agg(it):.1f}" if it else "NA",
               f"{agg(ie):.2f}" if ie else "NA", f"{ce:.2f}"]
        print("\t".join(str(x) for x in row), flush=True)

    print(f"\n# targets={len(targets)} (isct only; Vina は以前の結果を流用)")


if __name__ == "__main__":
    main()
