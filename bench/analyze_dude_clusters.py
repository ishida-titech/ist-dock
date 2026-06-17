#!/usr/bin/env python3
"""DUD-E の actives+decoys を統合して Bemis-Murcko クラスタ構造を解析する。

実スクリーニングと同様に active/decoy をまとめてクラスタリングし、クラスタサイズ分布に加え、
「active と decoy が混在するクラスタ」を集計する。混在クラスタは、seed（代表）が非結合体で
一括棄却された場合に active を巻き添えにする＝偽陰性リスクを表す（M3 棄却の正しさの核心）。

使い方:
    python3 bench/analyze_dude_clusters.py akt1
"""
import sys
from collections import defaultdict
from pathlib import Path

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

ROOT = Path(__file__).resolve().parent.parent
DUDE = ROOT / "bench" / "data" / "dude"


def read_ism(path: Path) -> list[str]:
    return [ln.split()[0] for ln in path.read_text().splitlines() if ln.strip()]


def murcko(smi: str) -> str | None:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False) or "<acyclic>"
    except Exception:
        return "<error>"


def main() -> None:
    target = sys.argv[1] if len(sys.argv) > 1 else "akt1"
    d = DUDE / target
    actives = read_ism(d / "actives_final.ism")
    decoys = read_ism(d / "decoys_final.ism")

    # scaffold -> [n_active, n_decoy]
    clusters: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for smi in actives:
        s = murcko(smi)
        if s is not None:
            clusters[s][0] += 1
    for smi in decoys:
        s = murcko(smi)
        if s is not None:
            clusters[s][1] += 1

    n_act, n_dec = len(actives), len(decoys)
    n_total = n_act + n_dec
    n_clusters = len(clusters)
    sizes = sorted((a + dd for a, dd in clusters.values()), reverse=True)
    in_multi = sum(s for s in sizes if s >= 2)

    # 混在クラスタ（active>=1 かつ decoy>=1）
    mixed = {s: (a, dd) for s, (a, dd) in clusters.items() if a >= 1 and dd >= 1}
    actives_in_mixed = sum(a for a, _ in mixed.values())
    # active のみ / decoy のみのクラスタ
    act_only = sum(1 for a, dd in clusters.values() if a >= 1 and dd == 0)
    actives_singleton = sum(1 for a, dd in clusters.values() if a == 1 and dd == 0)

    print(f"== {target}（active+decoy 統合 Bemis-Murcko）==")
    print(f"総分子数: {n_total} (active {n_act}, decoy {n_dec})")
    print(f"クラスタ数: {n_clusters}")
    print(f"2件以上のクラスタに含まれる分子: {in_multi} ({100*in_multi/n_total:.0f}%)")
    print(f"最大クラスタ: {sizes[0]} 分子, 上位10サイズ: {sizes[:10]}")
    print("--- active の行き先 ---")
    print(f"active 単独 singleton クラスタ: {actives_singleton} active")
    print(f"active のみの複数クラスタ数: {act_only - actives_singleton}")
    print("--- 混在クラスタ（偽陰性リスク）---")
    print(f"混在クラスタ数: {len(mixed)}")
    print(f"混在クラスタ内の active: {actives_in_mixed} / {n_act} "
          f"({100*actives_in_mixed/max(n_act,1):.0f}% が decoy と同居)")
    # 混在クラスタの内訳トップ
    top = sorted(mixed.items(), key=lambda kv: kv[1][0] + kv[1][1], reverse=True)[:8]
    print("  混在クラスタ上位 (active, decoy):")
    for _, (a, dd) in top:
        print(f"    active={a:>3}  decoy={dd:>4}")


if __name__ == "__main__":
    main()
