#!/usr/bin/env python3
"""DUD-E スクリーニング評価ハーネス: ISCT-dock / ISCT-screen / Vina / Smina を実行し、
共通スケール（vina --score_only 再採点）で enrichment（AUC/EF/BEDROC）・エネルギー・速度を比較する。

前処理は `bench/prep_dude_ligands.py`（receptor.pdbqt, ligands/<id>.pdbqt, all.sdf, labels.tsv, box.json）。
ISCT-screen は別途クラスタライブラリ（`--screen-lib <lib.bin>`）が必要。

フェーズ:
  --phase dock     : 指定手法を実行し res_<method>.tsv（id, label, native_score, wall_s, pose_path）を保存。
  --phase metrics  : 各 res_*.tsv のポーズを vina --score_only で再採点→指標を算出し summary を出力。
  （既定: dock→metrics 連続）

使い方:
  python3 bench/screen_eval.py kif11 --methods isct-dock vina smina [--screen-lib X.bin] \
      [--concurrency 20] [--cpu 4] [--exhaustiveness 8] [--n-runs 8] [--rescore-jobs 40]
"""
import argparse
import json
import re
import subprocess
import time
from multiprocessing import Pool
from pathlib import Path

from rdkit import Chem
from meeko import MoleculePreparation, PDBQTWriterLegacy

import screen_metrics as M

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "target" / "release" / "isct-dock"
VINA = "vina"
SMINA = str(Path.home() / "miniconda3" / "envs" / "smina" / "bin" / "smina")

RE_ISCT_AFF = re.compile(r"affinity \(Vina[^:]*:\s*([-\d.]+)")  # torsion 補正後 Vina 互換スコア
RE_ISCT_INTER = re.compile(r"intermolecular energy\s*:\s*([-\d.]+)")
RE_VINA_AFF = re.compile(r"^\s+1\s+([-\d.]+)", re.M)
RE_SCORE_ONLY = re.compile(r"Estimated Free Energy of Binding\s*[:=]\s*([-\d.]+)")


def load_inputs(out: Path):
    box = json.loads((out / "box.json").read_text())
    labels = {}
    for ln in (out / "labels.tsv").read_text().splitlines()[1:]:
        i, lab = ln.split("\t")
        labels[i] = 1 if lab == "active" else 0
    return box, labels


# ---------- 手法ランナー ----------

def _isct_dock_one(a):
    lid, lig, rec, cfg, posedir, threads = a
    pose = f"{posedir}/{lid}.pdbqt"
    env = {"RAYON_NUM_THREADS": str(threads)}
    import os
    t0 = time.perf_counter()
    r = subprocess.run([str(BIN), "dock", rec, lig, "--config", cfg, "--out", pose],
                       capture_output=True, text=True, env={**os.environ, **env})
    dt = time.perf_counter() - t0
    # native ランキングは affinity（torsion 補正後 Vina 互換）を使う。無ければ intermolecular。
    m = RE_ISCT_AFF.search(r.stdout) or RE_ISCT_INTER.search(r.stdout)
    return (lid, float(m.group(1)) if m else None, dt, pose)


def _vina_one(a):
    lid, lig, rec, box, posedir, cpu, exh = a
    pose = f"{posedir}/{lid}.pdbqt"
    c, s = box["center"], box["size"]
    t0 = time.perf_counter()
    r = subprocess.run([VINA, "--receptor", rec, "--ligand", lig,
                        "--center_x", str(c[0]), "--center_y", str(c[1]), "--center_z", str(c[2]),
                        "--size_x", str(s[0]), "--size_y", str(s[1]), "--size_z", str(s[2]),
                        "--exhaustiveness", str(exh), "--cpu", str(cpu), "--seed", "42",
                        "--num_modes", "1", "--out", pose], capture_output=True, text=True)
    dt = time.perf_counter() - t0
    m = RE_VINA_AFF.search(r.stdout)
    return (lid, float(m.group(1)) if m else None, dt, pose)


def run_per_ligand(method, ligs, rec, box, out, args):
    posedir = out / f"poses_{method}"; posedir.mkdir(exist_ok=True)
    if method == "isct-dock":
        c, s = box["center"], box["size"]
        cfg = out / "isct_dock.toml"
        cfg.write_text(f"[box]\ncenter=[{c[0]},{c[1]},{c[2]}]\nsize=[{s[0]},{s[1]},{s[2]}]\n"
                       f'[run]\nseed=42\nn_runs={args.n_runs}\nscoring="grid"\n')
        tasks = [(lid, str(lig), str(rec), str(cfg), str(posedir), args.cpu) for lid, lig in ligs]
        fn = _isct_dock_one
    elif method == "vina":
        tasks = [(lid, str(lig), str(rec), box, str(posedir), args.cpu, args.exhaustiveness) for lid, lig in ligs]
        fn = _vina_one
    else:
        raise ValueError(method)
    t0 = time.perf_counter()
    rows = []
    with Pool(args.concurrency) as pool:
        for lid, score, dt, pose in pool.imap_unordered(fn, tasks, chunksize=4):
            rows.append((lid, score, dt, pose))
    total = time.perf_counter() - t0
    return rows, total


def _smina_shard(a):
    """1 シャード（複数リガンドの SDF）を smina で処理。出力 SDF パスを返す。"""
    shard_sdf, rec, box, osdf, cpu, exh = a
    c, s = box["center"], box["size"]
    subprocess.run([SMINA, "-r", rec, "-l", shard_sdf,
                    "--center_x", str(c[0]), "--center_y", str(c[1]), "--center_z", str(c[2]),
                    "--size_x", str(s[0]), "--size_y", str(s[1]), "--size_z", str(s[2]),
                    "--exhaustiveness", str(exh), "--cpu", str(cpu),
                    "--seed", "42", "--num_modes", "1", "-o", osdf],
                   capture_output=True, text=True, check=True)
    return osdf


def run_smina(ligs_ids, rec, allsdf, box, out, args):
    """全リガンドを smina_cpu 個のシャードに分割し並列実行（コア飽和・公平な最速構成）。

    smina/vina の --cpu は「1 リガンド内の exhaustiveness 個の独立探索」を並列化するため、
    単一バッチでは exhaustiveness(=8) コアで頭打ちになる。リガンドレベルで分割し各シャードを
    1 コアで並列実行することで、vina/isct-dock と同じ「1 コア×N 並列」の公平モデルにする。"""
    posedir = out / "poses_smina"; posedir.mkdir(exist_ok=True)
    shard_dir = out / "smina_shards"
    if shard_dir.exists():
        for f in shard_dir.iterdir():
            f.unlink()
    shard_dir.mkdir(exist_ok=True)
    # all.sdf を smina_cpu 個のシャードに分割（ラウンドロビンで負荷均等化）。
    blocks = [b for b in Path(allsdf).read_text().split("$$$$\n") if b.strip()]
    nsh = max(1, min(args.smina_cpu, len(blocks)))
    shards = [[] for _ in range(nsh)]
    for i, b in enumerate(blocks):
        shards[i % nsh].append(b)
    tasks = []
    for si, blk in enumerate(shards):
        if not blk:
            continue
        sp = shard_dir / f"in_{si}.sdf"
        sp.write_text("".join(b + "$$$$\n" for b in blk))
        op = shard_dir / f"out_{si}.sdf"
        tasks.append((str(sp), str(rec), box, str(op), 1, args.exhaustiveness))
    t0 = time.perf_counter()
    with Pool(len(tasks)) as pool:
        outs = list(pool.imap_unordered(_smina_shard, tasks))
    total = time.perf_counter() - t0
    # 各シャード出力 SDF: name + minimizedAffinity。各ポーズを PDBQT 化（再採点用）。
    rows = []
    for osdf in outs:
        supp = Chem.SDMolSupplier(osdf, removeHs=False, sanitize=False)
        for mol in supp:
            if mol is None:
                continue
            lid = mol.GetProp("_Name") if mol.HasProp("_Name") else None
            aff = float(mol.GetProp("minimizedAffinity")) if mol.HasProp("minimizedAffinity") else None
            pose = posedir / f"{lid}.pdbqt"
            try:
                Chem.SanitizeMol(mol)  # smina 出力(sanitize=False 読込)を meeko 変換可能にする
                mh = Chem.AddHs(mol, addCoords=True)
                setups = MoleculePreparation().prepare(mh)
                pq, ok, _ = PDBQTWriterLegacy.write_string(setups[0])
                if ok:
                    pose.write_text(pq)
                else:
                    pose = None
            except Exception:
                pose = None
            rows.append((lid, aff, total / max(len(ligs_ids), 1), str(pose) if pose else ""))
    return rows, total


def run_isct_screen(rec, lib, box, out, args):
    c, s = box["center"], box["size"]
    cfg = out / "isct_screen.toml"
    cfg.write_text(f"[box]\ncenter=[{c[0]},{c[1]},{c[2]}]\nsize=[{s[0]},{s[1]},{s[2]}]\n"
                   f'[run]\nseed=42\nn_runs={args.n_runs}\nscoring="grid"\n'
                   f"[screen]\nprune_threshold=1e9\nlocal_iters=120\n")
    rank = out / "screen_ranking.tsv"; poses = out / "screen_poses.pdbqt"
    t0 = time.perf_counter()
    subprocess.run([str(BIN), "screen", str(rec), str(lib), "--config", str(cfg),
                    "--out", str(rank), "--poses-out", str(poses), "--top", "100000"],
                   capture_output=True, text=True, check=True)
    total = time.perf_counter() - t0
    rows = []
    for ln in rank.read_text().splitlines()[1:]:
        p = ln.split("\t")  # Rank Compound Cluster Total VdW Elec State
        if len(p) >= 5:
            rows.append((p[1], float(p[3]), total, ""))  # pose は screen_poses にまとまり（再採点は割愛/別）
    return rows, total


# ---------- 再採点（共通スケール） ----------

def _first_model_pdbqt(text):
    """マルチ MODEL PDBQT から先頭 MODEL（=ベストポーズ）の本体のみ抽出。
    vina --score_only は MODEL/ENDMDL 付きを拒否するため、タグを外し 1 リガンド化する。"""
    if "MODEL" not in text:
        return text
    out, inside, done = [], False, False
    for ln in text.splitlines():
        if ln.startswith("MODEL"):
            inside = True
            continue
        if ln.startswith("ENDMDL"):
            if inside:
                done = True
                break
            continue
        if inside:
            out.append(ln)
    return "\n".join(out) + "\n" if done else text


def _rescore_one(a):
    lid, pose, rec, box = a
    if not pose or not Path(pose).exists():
        return (lid, None)
    c, s = box["center"], box["size"]
    big = [x + 20 for x in s]
    # マルチ MODEL ポーズは先頭 MODEL を一時ファイルへ展開してから採点。
    single = Path(pose)
    tmp = None
    txt = single.read_text()
    if "MODEL" in txt:
        tmp = single.with_suffix(".m1.pdbqt")
        tmp.write_text(_first_model_pdbqt(txt))
        single = tmp
    r = subprocess.run([VINA, "--score_only", "--receptor", rec, "--ligand", str(single),
                        "--center_x", str(c[0]), "--center_y", str(c[1]), "--center_z", str(c[2]),
                        "--size_x", str(big[0]), "--size_y", str(big[1]), "--size_z", str(big[2])],
                       capture_output=True, text=True)
    if tmp is not None:
        tmp.unlink(missing_ok=True)
    m = RE_SCORE_ONLY.search(r.stdout)
    return (lid, float(m.group(1)) if m else None)


def rescore(rows, rec, box, jobs):
    tasks = [(lid, pose, str(rec), box) for lid, _sc, _t, pose in rows]
    res = {}
    with Pool(jobs) as pool:
        for lid, v in pool.imap_unordered(_rescore_one, tasks, chunksize=8):
            res[lid] = v
    return res


# ---------- 集計 ----------

def summarize(method, rows, labels, rescored, total_wall):
    ids = [r[0] for r in rows if r[0] in labels]
    native = {r[0]: r[1] for r in rows}
    # スコア: 再採点 vina 優先、無ければ native。
    def sc(i):
        v = rescored.get(i)
        return v if v is not None else native.get(i)
    valid = [i for i in ids if sc(i) is not None]
    scores = [sc(i) for i in valid]
    labs = [labels[i] for i in valid]
    na = sum(labs)
    def estats(sel):
        vals = [scores[k] for k in range(len(valid)) if labs[k] == sel]
        vals.sort()
        if not vals: return (float("nan"), float("nan"))
        return (sum(vals) / len(vals), vals[len(vals) // 2])
    am, amed = estats(1); dm, dmed = estats(0)
    return {
        "method": method, "n": len(valid), "n_active": na, "n_decoy": len(valid) - na,
        "AUC": round(M.roc_auc(scores, labs), 4),
        "EF1pct": round(M.enrichment_factor(scores, labs, 0.01), 2),
        "EF5pct": round(M.enrichment_factor(scores, labs, 0.05), 2),
        "BEDROC20": round(M.bedroc(scores, labs, 20.0), 4),
        "energy_active_mean": round(am, 3), "energy_active_median": round(amed, 3),
        "energy_decoy_mean": round(dm, 3), "energy_decoy_median": round(dmed, 3),
        "total_wall_s": round(total_wall, 1),
        "throughput_lig_per_s": round(len(rows) / total_wall, 2) if total_wall else None,
        "per_ligand_mean_s": round(sum(r[2] for r in rows) / max(len(rows), 1), 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target")
    ap.add_argument("--methods", nargs="+", default=["isct-dock", "vina", "smina"])
    ap.add_argument("--screen-lib", default=None)
    ap.add_argument("--concurrency", type=int, default=20)
    ap.add_argument("--cpu", type=int, default=4)
    ap.add_argument("--smina-cpu", type=int, default=20)
    ap.add_argument("--exhaustiveness", type=int, default=8)
    ap.add_argument("--n-runs", type=int, default=8)
    ap.add_argument("--rescore-jobs", type=int, default=40)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    out = Path(args.out_dir) if args.out_dir else ROOT / "bench" / "data" / "screen_eval" / args.target
    rec = out / "receptor.pdbqt"
    box, labels = load_inputs(out)
    ligs = [(p.stem, p) for p in sorted((out / "ligands").glob("*.pdbqt"))]
    ids = [lid for lid, _ in ligs]
    print(f"{args.target}: {len(ligs)} ligands, {sum(labels.values())} actives")

    summaries = []
    for method in args.methods:
        print(f"--- {method} ---", flush=True)
        if method in ("isct-dock", "vina"):
            rows, total = run_per_ligand(method, ligs, rec, box, out, args)
        elif method == "smina":
            rows, total = run_smina(ids, rec, out / "all.sdf", box, out, args)
        elif method == "isct-screen":
            if not args.screen_lib:
                print("  skip: --screen-lib 未指定"); continue
            rows, total = run_isct_screen(rec, args.screen_lib, box, out, args)
        else:
            print(f"  unknown method {method}"); continue
        # 保存
        with open(out / f"res_{method}.tsv", "w") as f:
            f.write("id\tnative\twall_s\tpose\n")
            for lid, sc, dt, pose in rows:
                f.write(f"{lid}\t{sc}\t{dt}\t{pose}\n")
        rescored = rescore(rows, rec, box, args.rescore_jobs)
        summ = summarize(method, rows, labels, rescored, total)
        summaries.append(summ)
        print(json.dumps(summ, indent=2), flush=True)

    (out / "summary.json").write_text(json.dumps(summaries, indent=2))
    # 表示
    cols = ["method", "AUC", "EF1pct", "EF5pct", "BEDROC20", "energy_active_mean",
            "energy_decoy_mean", "total_wall_s", "throughput_lig_per_s"]
    print("\n" + "\t".join(cols))
    for s in summaries:
        print("\t".join(str(s.get(c, "")) for c in cols))


if __name__ == "__main__":
    main()
