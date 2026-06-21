#!/usr/bin/env python3
"""ISCT-screen（クラスタ・scaffold 再利用）の kif11 評価。

- native: screen_ranking.tsv の Total_Score（＝torsion 補正後 affinity）でランク。
- common: screen_poses.pdbqt を compound ごとに分割→ vina --score_only で共通スケール再採点。
両方について全指標（AUC/EF@0.5/1/2/5%/BEDROC20）を出す。
"""
import json
import re
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, "bench")
import screen_metrics as M

VINA = "/home/ishida/miniconda3/bin/vina"
D = Path("bench/data/screen_eval/kif11")
RE_SCORE = re.compile(r"Estimated Free Energy of Binding\s*:\s*(-?\d+\.?\d*)")
RE_REMARK = re.compile(r"^REMARK Rank \d+ Compound (\S+) Cluster \S+ Score (-?\d+\.?\d*) Role (\S+)")


def labels():
    lab = {}
    for ln in (D / "labels.tsv").read_text().splitlines()[1:]:
        i, l = ln.split("\t")
        lab[i] = 1 if l == "active" else 0
    return lab


def native_scores():
    """screen_ranking.tsv → {compound: total_score}。"""
    sc = {}
    for ln in (D / "screen_ranking.tsv").read_text().splitlines()[1:]:
        p = ln.split("\t")
        if len(p) >= 4:
            sc[p[1]] = float(p[3])
    return sc


def split_poses():
    """screen_poses.pdbqt を compound ごとの単一 PDBQT に分割（MODEL/REMARK 除去）。"""
    pdir = D / "poses_screen"
    pdir.mkdir(exist_ok=True)
    cur, cid = [], None
    written = []
    for ln in (D / "screen_poses.pdbqt").read_text().splitlines():
        if ln.startswith("MODEL"):
            cur, cid = [], None
        elif ln.startswith("REMARK Rank"):
            m = RE_REMARK.match(ln)
            if m:
                cid = m.group(1)
        elif ln.startswith("ENDMDL"):
            if cid and cur:
                (pdir / f"{cid}.pdbqt").write_text("\n".join(cur) + "\n")
                written.append(cid)
            cur, cid = [], None
        else:
            cur.append(ln)
    return written


def _rescore_one(a):
    cid, box = a
    pose = D / "poses_screen" / f"{cid}.pdbqt"
    if not pose.exists():
        return (cid, None)
    c, s = box["center"], box["size"]
    big = [x + 20 for x in s]
    r = subprocess.run([VINA, "--score_only", "--receptor", str(D / "receptor.pdbqt"),
                        "--ligand", str(pose), "--center_x", str(c[0]), "--center_y", str(c[1]),
                        "--center_z", str(c[2]), "--size_x", str(big[0]), "--size_y", str(big[1]),
                        "--size_z", str(big[2])], capture_output=True, text=True)
    m = RE_SCORE.search(r.stdout)
    return (cid, float(m.group(1)) if m else None)


def report(tag, scoremap, lab):
    ids = [i for i in scoremap if i in lab and scoremap[i] is not None]
    s = [scoremap[i] for i in ids]
    l = [lab[i] for i in ids]
    mt = M.full_metrics(s, l)
    print(f"{tag:34s} n={len(ids)} act={sum(l)} | AUC {mt['AUC']:.3f} | EF0.5 {mt['EF0.5']:5.1f} | "
          f"EF1 {mt['EF1']:5.2f} | EF2 {mt['EF2']:5.2f} | EF5 {mt['EF5']:5.2f} | BEDROC {mt['BEDROC20']:.3f}")
    return dict(mt, n=len(ids), n_active=sum(l))


def main():
    box = json.loads((D / "box.json").read_text())
    lab = labels()
    out = {}
    nat = native_scores()
    out["native_total_score"] = report("ISCT-screen native(Total_Score)", nat, lab)
    n = split_poses()
    print(f"split {len(n)} poses; rescoring...", flush=True)
    with Pool(40) as pool:
        rs = dict(pool.imap_unordered(_rescore_one, [(c, box) for c in set(n)], chunksize=8))
    out["vina_rescored"] = report("ISCT-screen vina-rescored", rs, lab)
    (D / "summary_isct_screen.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
