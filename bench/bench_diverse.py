#!/usr/bin/env python3
"""多様データセットで isct-dock vs Vina の速度・精度を比較する（各 target × 3 seed）。

bench/data/diverse/<pdbid>/ の {receptor.pdbqt, ligand.pdbqt, config.toml, info.json} を走査し、
各 target を 3 seed（1/42/123）で両エンジン再ドッキング。結晶 RMSD と壁時間を計測し、
リガンド性質（重原子数・torsion・大環状・最大環・MW）とともに 1 行/target のテーブルにまとめる。

- isct: grid + n_steps 自動 + exact polish（config.toml の scoring="grid"）, n_runs=8, 20 スレッド。
- vina: exhaustiveness 8, --cpu 20。
- 計測は両エンジン同一ハーネス・同一条件（time.perf_counter）。

使い方: python3 bench/bench_diverse.py [--seeds 1 42 123] > diverse_results.tsv
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
EXH = 8
N_RUNS = 8


def isct_dock(d: Path, center, size, seed, out: Path):
    cfg = out.with_suffix(".toml")
    cfg.write_text(
        f"[box]\ncenter = [{center[0]}, {center[1]}, {center[2]}]\n"
        f"size = [{size[0]}, {size[1]}, {size[2]}]\n"
        f'[run]\nseed = {seed}\nn_runs = {N_RUNS}\nscoring = "grid"\n'
    )
    env = {**os.environ, "RAYON_NUM_THREADS": str(THREADS)}
    t0 = time.perf_counter()
    subprocess.run([str(BIN), "dock", str(d / "receptor.pdbqt"), str(d / "ligand.pdbqt"),
                    "--config", str(cfg), "--out", str(out)],
                   check=True, capture_output=True, text=True, env=env)
    return time.perf_counter() - t0


def vina_dock(d: Path, center, size, seed, out: Path):
    t0 = time.perf_counter()
    subprocess.run(["vina", "--receptor", str(d / "receptor.pdbqt"), "--ligand", str(d / "ligand.pdbqt"),
                    "--center_x", str(center[0]), "--center_y", str(center[1]), "--center_z", str(center[2]),
                    "--size_x", str(size[0]), "--size_y", str(size[1]), "--size_z", str(size[2]),
                    "--seed", str(seed), "--cpu", str(THREADS), "--exhaustiveness", str(EXH),
                    "--out", str(out)],
                   check=True, capture_output=True, text=True)
    return time.perf_counter() - t0


def agg(vals):
    return sum(vals) / len(vals) if vals else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 42, 123])
    ap.add_argument("--targets", nargs="*", default=None, help="限定する pdbid（省略で全件）")
    args = ap.parse_args()
    tmp = Path("/tmp/diverse_bench"); tmp.mkdir(exist_ok=True)

    if args.targets:
        targets = [DIVERSE / t for t in args.targets]
    else:
        targets = sorted(p.parent for p in DIVERSE.glob("*/info.json"))
    cols = ["pdbid", "lig", "heavy", "tors", "rings", "maxring", "macro", "MW",
            "isct_rmsd", "isct_ok", "isct_t", "vina_rmsd", "vina_ok", "vina_t"]
    print("\t".join(cols))
    rows = []
    for d in targets:
        info = json.loads((d / "info.json").read_text())
        center = info["center"]; size = [20.0, 20.0, 20.0]
        crystal = heavy_coords((d / "ligand.pdbqt").read_text())
        ir, it, vr, vt = [], [], [], []
        for seed in args.seeds:
            try:
                t = isct_dock(d, center, size, seed, tmp / f"{info['pdbid']}_{seed}_i.pdbqt")
                ir.append(rmsd(heavy_coords((tmp / f"{info['pdbid']}_{seed}_i.pdbqt").read_text()), crystal))
                it.append(t)
            except Exception:
                pass
            try:
                t = vina_dock(d, center, size, seed, tmp / f"{info['pdbid']}_{seed}_v.pdbqt")
                vr.append(rmsd(heavy_coords((tmp / f"{info['pdbid']}_{seed}_v.pdbqt").read_text(), model=1), crystal))
                vt.append(t)
            except Exception:
                pass
        iok = sum(1 for x in ir if x < 2.0)
        vok = sum(1 for x in vr if x < 2.0)
        row = [info["pdbid"], info["ligand"], info["heavy_atoms"], info["torsdof"],
               info["n_rings"], info["max_ring"], "Y" if info["macrocycle"] else "n", info["mw"],
               f"{agg(ir):.2f}" if ir else "NA", f"{iok}/{len(ir)}", f"{agg(it):.1f}" if it else "NA",
               f"{agg(vr):.2f}" if vr else "NA", f"{vok}/{len(vr)}", f"{agg(vt):.1f}" if vt else "NA"]
        print("\t".join(str(x) for x in row), flush=True)
        rows.append((info, ir, it, vr, vt))

    # サマリ。
    n = len(rows)
    isct_succ = sum(sum(1 for x in r[1] if x < 2.0) for r in rows)
    vina_succ = sum(sum(1 for x in r[3] if x < 2.0) for r in rows)
    isct_runs = sum(len(r[1]) for r in rows)
    vina_runs = sum(len(r[3]) for r in rows)
    it_all = [t for r in rows for t in r[2]]
    vt_all = [t for r in rows for t in r[4]]
    print(f"\n# targets={n}")
    print(f"# isct success {isct_succ}/{isct_runs} ({100*isct_succ/max(isct_runs,1):.0f}%), "
          f"mean_time {agg(it_all):.1f}s")
    print(f"# vina success {vina_succ}/{vina_runs} ({100*vina_succ/max(vina_runs,1):.0f}%), "
          f"mean_time {agg(vt_all):.1f}s")
    print(f"# speed ratio (vina/isct mean time) = {agg(vt_all)/agg(it_all):.2f}x" if it_all else "")


if __name__ == "__main__":
    main()
