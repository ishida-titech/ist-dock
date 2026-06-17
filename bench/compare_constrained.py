#!/usr/bin/env python3
"""Scaffold 拘束スコア vs 独立フルドッキングの劣化を集計する（M3 密検証）。

`screen` が出した拘束スコア（ranking.tsv の VdW_Energy=分子間 e_inter）と、各子分子を
`isct-dock dock`（全自由度）で独立ドッキングした e_inter を比較し、Scaffold 拘束による
劣化分布（Δ=拘束−独立）を集計する。独立ドッキングは高コストなのでサンプルに限定する。

使い方:
    python3 bench/compare_constrained.py <library_dir> --config run.toml \
        [--sample 20] [--bin ./target/release/isct-dock]
"""
import argparse
import random
import re
import subprocess
import sys
from pathlib import Path

INTER = re.compile(r"intermolecular energy\s*:\s*(-?\d+\.\d+)")


def read_ranking(path: Path) -> dict:
    rows = {}
    for ln in path.read_text().splitlines()[1:]:
        f = ln.split("\t")
        if len(f) >= 5:
            rows[f[1]] = {"total": float(f[3]), "e_inter": float(f[4])}
    return rows


def read_labels(path: Path) -> dict:
    out = {}
    if not path.exists():
        return out
    for ln in path.read_text().splitlines()[1:]:
        f = ln.split("\t")
        if len(f) >= 4:
            out[f[0]] = {"label": f[1], "role": f[3]}
    return out


def independent_dock(binary: str, receptor: Path, ligand: Path, config: Path) -> float | None:
    r = subprocess.run(
        [binary, "dock", str(receptor), str(ligand), "--config", str(config),
         "--out", "/tmp/_cmp_out.pdbqt"],
        capture_output=True, text=True,
    )
    m = INTER.search(r.stdout)
    return float(m.group(1)) if m else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("library_dir")
    ap.add_argument("--config", required=True)
    ap.add_argument("--sample", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bin", default="./target/release/isct-dock")
    args = ap.parse_args()

    d = Path(args.library_dir)
    receptor = d / "receptor.pdbqt"
    ranking = read_ranking(d / "ranking.tsv")
    labels = read_labels(d / "labels.tsv")

    children = [cid for cid, m in labels.items() if m["role"] == "child" and cid in ranking]
    random.seed(args.seed)
    if len(children) > args.sample:
        children = random.sample(children, args.sample)
    children.sort()

    print(f"# 拘束 vs 独立フルドッキング（{d.name}, sample={len(children)}）")
    print("compound\tlabel\tconstrained\tindependent\tdelta(constr-indep)")
    deltas = []
    for cid in children:
        lig = d / "mols" / f"{cid}.pdbqt"
        indep = independent_dock(args.bin, receptor, lig, Path(args.config))
        c = ranking[cid]["e_inter"]
        if indep is None:
            print(f"{cid}\t{labels[cid]['label']}\t{c:.3f}\tFAIL\tNA")
            continue
        delta = c - indep
        deltas.append(delta)
        print(f"{cid}\t{labels[cid]['label']}\t{c:.3f}\t{indep:.3f}\t{delta:+.3f}", flush=True)

    if deltas:
        deltas.sort()
        n = len(deltas)
        mean = sum(deltas) / n
        median = deltas[n // 2]
        print(f"\n# 劣化 Δ=拘束−独立 (kcal/mol): n={n} mean={mean:+.3f} "
              f"median={median:+.3f} min={deltas[0]:+.3f} max={deltas[-1]:+.3f}")


if __name__ == "__main__":
    main()
