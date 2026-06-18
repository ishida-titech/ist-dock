#!/usr/bin/env python3
"""isct-dock vs Vina の精度・速度比較（スレッド数 × run 数の格子）。

再ドッキング 3 ケース（3ptb/1stp/1iep）について、(threads, runs) = (8,8),(8,20),(20,8),(20,20)
の 4 パターンで両エンジンを実行し、壁時間・結晶 RMSD・スコア（共通スケール）を比較する。

- スレッド数: isct-dock は RAYON_NUM_THREADS、Vina は --cpu。
- run 数: isct-dock は n_runs、Vina は --exhaustiveness（探索努力の対応物）。
- RMSD: 両エンジンの出力ポーズ重原子を結晶（入力 ligand.pdbqt）と比較（heavy_coords で統一）。
- スコア共通スケール: 両ポーズを `vina --score_only` で再採点した affinity（kcal/mol、低いほど良い）。

使い方:
    python3 bench/compare_vina_grid.py
"""
import os
import re
import subprocess
import time
import tomllib
from pathlib import Path

from run_redock import heavy_coords, rmsd

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "target" / "release" / "isct-dock"
CASES = ["3ptb", "1stp", "1iep"]
PATTERNS = [(8, 8), (8, 20), (20, 8), (20, 20)]
SEED = 42
N_STEPS = 60


def box(case_dir: Path):
    cfg = tomllib.loads((case_dir / "config.toml").read_text())
    c = cfg["box"]["center"]
    s = cfg["box"]["size"]
    return c, s


def vina_affinity(stdout: str) -> float | None:
    m = re.search(r"^\s+1\s+([-\d.]+)", stdout, re.M)  # 結果テーブルの mode 1
    if m:
        return float(m.group(1))
    m = re.search(r"Affinity:\s*([-\d.]+)", stdout)
    if m:
        return float(m.group(1))
    m = re.search(r"Estimated Free Energy of Binding\s*[:=]\s*([-\d.]+)", stdout)
    return float(m.group(1)) if m else None


def vina_score_only(rec: Path, lig: Path, center, size) -> float | None:
    # ポーズがはみ出さないよう余裕を持たせた box（スコアはカットオフ依存で box サイズに不感）。
    big = [s + 20.0 for s in size]
    r = subprocess.run(
        ["vina", "--score_only", "--receptor", str(rec), "--ligand", str(lig),
         "--center_x", str(center[0]), "--center_y", str(center[1]), "--center_z", str(center[2]),
         "--size_x", str(big[0]), "--size_y", str(big[1]), "--size_z", str(big[2])],
        capture_output=True, text=True,
    )
    m = re.search(r"Estimated Free Energy of Binding\s*[:=]\s*([-\d.]+)", r.stdout)
    if m:
        return float(m.group(1))
    return vina_affinity(r.stdout)


SCORING = os.environ.get("ISCT_SCORING", "direct")  # "direct" | "grid"


def run_isct(rec, lig, center, size, threads, runs, out):
    cfg = out.with_suffix(".toml")
    cfg.write_text(
        f"[box]\ncenter = [{center[0]}, {center[1]}, {center[2]}]\n"
        f"size = [{size[0]}, {size[1]}, {size[2]}]\n"
        f"[run]\nseed = {SEED}\nn_runs = {runs}\nn_steps = {N_STEPS}\n"
        f'scoring = "{SCORING}"\n'
    )
    env = {**os.environ, "RAYON_NUM_THREADS": str(threads)}
    t0 = time.perf_counter()
    subprocess.run(
        [str(BIN), "dock", str(rec), str(lig), "--config", str(cfg), "--out", str(out)],
        capture_output=True, text=True, check=True, env=env,
    )
    return time.perf_counter() - t0


def run_vina(rec, lig, center, size, threads, runs, out):
    t0 = time.perf_counter()
    r = subprocess.run(
        ["vina", "--receptor", str(rec), "--ligand", str(lig),
         "--center_x", str(center[0]), "--center_y", str(center[1]), "--center_z", str(center[2]),
         "--size_x", str(size[0]), "--size_y", str(size[1]), "--size_z", str(size[2]),
         "--seed", str(SEED), "--cpu", str(threads), "--exhaustiveness", str(runs),
         "--out", str(out)],
        capture_output=True, text=True, check=True,
    )
    return time.perf_counter() - t0, vina_affinity(r.stdout)


def main():
    tmp = Path("/tmp/isct_vina_cmp")
    tmp.mkdir(exist_ok=True)
    print(f"{'case':6} {'T/run':7} {'engine':9} {'time(s)':>8} {'rmsd(Å)':>8} {'vina_score':>11}")
    print("-" * 56)
    for case in CASES:
        d = ROOT / "bench" / "data" / "pdb" / case
        rec, lig = d / "receptor.pdbqt", d / "ligand.pdbqt"
        center, size = box(d)
        crystal = heavy_coords(lig.read_text())
        for (threads, runs) in PATTERNS:
            tag = f"{threads}/{runs}"
            # isct-dock
            iout = tmp / f"{case}_{threads}_{runs}_isct.pdbqt"
            it = run_isct(rec, lig, center, size, threads, runs, iout)
            irmsd = rmsd(heavy_coords(iout.read_text()), crystal)
            iscore = vina_score_only(rec, iout, center, size)
            print(f"{case:6} {tag:7} {'isct-dock':9} {it:8.2f} {irmsd:8.3f} "
                  f"{iscore if iscore is not None else float('nan'):11.3f}")
            # vina
            vout = tmp / f"{case}_{threads}_{runs}_vina.pdbqt"
            vt, vscore = run_vina(rec, lig, center, size, threads, runs, vout)
            vrmsd = rmsd(heavy_coords(vout.read_text(), model=1), crystal)
            print(f"{case:6} {tag:7} {'vina':9} {vt:8.2f} {vrmsd:8.3f} "
                  f"{vscore if vscore is not None else float('nan'):11.3f}")
        # 参考: isct-dock を既定の網羅性 n_runs=60（20 スレッド）で（精度パリティ確認用）。
        iout = tmp / f"{case}_ref60_isct.pdbqt"
        it = run_isct(rec, lig, center, size, 20, 60, iout)
        irmsd = rmsd(heavy_coords(iout.read_text()), crystal)
        iscore = vina_score_only(rec, iout, center, size)
        print(f"{case:6} {'20/60':7} {'isct-dock':9} {it:8.2f} {irmsd:8.3f} "
              f"{iscore if iscore is not None else float('nan'):11.3f}  (参考)")
        print()


if __name__ == "__main__":
    main()
