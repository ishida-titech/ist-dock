#!/usr/bin/env python3
"""M3 クラスタ高速化の検証ハーネス（まず統計のみ＝ドッキングなし）。

active/decoy を統合してクラスタリングし、Tanimoto しきい値を振って「計算削減余地」と
「偽陰性曝露」を集計する。ドッキング前にこの曲線を見て、密ドッキング検証の規模を決める。

集計（しきい値ごと）:
  - クラスタ数 / 多member率 / seed としてドッキングする件数（= クラスタ数）
  - 削減余地上限 = 子分子率 = (N − クラスタ数)/N（子は Phase3 の安価評価 or 棄却で省ける）
  - decoy のみクラスタ（安全に一括棄却可）の分子数
  - active を含むクラスタ / 混在（active+decoy）クラスタ数
  - 偽陰性曝露 = 非 singleton の active 数（seed 棄却で巻き添えになりうる）
  - 上位数クラスタの MCS コア原子数（Scaffold 拘束の妥当性指標）

使い方:
    python3 bench/run_cluster_screen.py ampc [--thresholds 0.4 0.5 0.6 0.7 0.8] [--mcs-sample 3]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "preprocess"))
import cluster_library as cl  # noqa: E402
from rdkit import Chem  # noqa: E402


def summarize(target: str, recs, fps, thr: float, mcs_sample: int) -> dict:
    groups = cl.cluster_by_tanimoto(fps, thr)
    groups.sort(key=len, reverse=True)
    n = len(recs)
    n_clusters = len(groups)
    multi = [g for g in groups if len(g) > 1]
    n_in_multi = sum(len(g) for g in multi)

    decoy_only_mols = 0
    decoy_only_clusters = 0
    active_clusters = 0
    mixed_clusters = 0
    actives_singleton = 0
    actives_in_multi = 0
    for g in groups:
        labels = [recs[i]["label"] for i in g]
        has_a = "active" in labels
        has_d = "decoy" in labels
        n_a = labels.count("active")
        if has_a and not has_d:
            active_clusters += 1
        elif has_a and has_d:
            mixed_clusters += 1
            active_clusters += 1
        else:  # decoy only
            decoy_only_clusters += 1
            decoy_only_mols += len(g)
        if n_a:
            if len(g) == 1:
                actives_singleton += n_a
            else:
                actives_in_multi += n_a

    # 上位クラスタの MCS コア原子数（seed+最大数子の subset で計算、Scaffold 妥当性の目安）。
    mcs_sizes = []
    for g in multi[:mcs_sample]:
        subset = [recs[i]["mol"] for i in g[:4]]
        try:
            mcs = cl.mcs_of(subset, timeout=10)
            mcs_sizes.append(mcs.numAtoms)
        except Exception:
            mcs_sizes.append(-1)

    return {
        "thr": thr,
        "clusters": n_clusters,
        "frac_multi": n_in_multi / n,
        "reduction_upper": (n - n_clusters) / n,
        "decoy_only_clusters": decoy_only_clusters,
        "decoy_only_mols": decoy_only_mols,
        "active_clusters": active_clusters,
        "mixed_clusters": mixed_clusters,
        "actives_singleton": actives_singleton,
        "actives_in_multi": actives_in_multi,
        "top_mcs": mcs_sizes,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("target")
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.4, 0.5, 0.6, 0.7, 0.8])
    ap.add_argument("--mcs-sample", type=int, default=3)
    args = ap.parse_args()

    recs = cl.load_target(args.target)
    A = sum(1 for r in recs if r["label"] == "active")
    D = sum(1 for r in recs if r["label"] == "decoy")
    fps = cl.morgan_fps([r["mol"] for r in recs])
    n = len(recs)

    print(f"== {args.target}: 統合クラスタ統計（Morgan r2, active+decoy）==")
    print(f"分子数 N={n}  (active {A} / decoy {D})\n")
    hdr = ("thr", "clusters", "seedDock", "child%", "減上限", "decoyOnlyClu", "decoyOnlyMol",
           "actClu", "mixed", "actSingle", "actMulti", "topMCS")
    print("\t".join(hdr))
    for thr in args.thresholds:
        s = summarize(args.target, recs, fps, thr, args.mcs_sample)
        print("\t".join(str(x) for x in (
            f"{s['thr']:.2f}", s["clusters"], s["clusters"],
            f"{s['frac_multi']:.0%}", f"{s['reduction_upper']:.0%}",
            s["decoy_only_clusters"], s["decoy_only_mols"],
            s["active_clusters"], s["mixed_clusters"],
            s["actives_singleton"], s["actives_in_multi"], s["top_mcs"],
        )))
    print("\n注: 'seedDock'=実ドッキングする seed 数, 'child%'=多member率, '減上限'=子分子率")
    print("    'actMulti'=非 singleton の active 数（seed 棄却で偽陰性化しうる曝露）")


if __name__ == "__main__":
    main()
