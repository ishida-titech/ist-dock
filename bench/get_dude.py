#!/usr/bin/env python3
"""DUD-E の指定標的をダウンロードし、件数と Bemis-Murcko クラスタ構造を特性化する。

M3（リガンドクラスタリング高速化）の検証データ準備。actives/decoys の SMILES と受容体・
結晶リガンドを取得し、共通スキャフォールドでクラスタリングして「同族列がどれだけあるか」
（=コア流用・一括棄却が効く余地）を定量する。

使い方:
    python3 bench/get_dude.py akt1 ampc cxcr4 hivrt kif11
"""
import sys
import urllib.request
from collections import Counter
from pathlib import Path

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

ROOT = Path(__file__).resolve().parent.parent
DUDE = ROOT / "bench" / "data" / "dude"
BASE = "https://dude.docking.org/targets"
FILES = ["receptor.pdb", "crystal_ligand.mol2", "actives_final.ism", "decoys_final.ism"]


def download(target: str) -> Path:
    d = DUDE / target
    d.mkdir(parents=True, exist_ok=True)
    for f in FILES:
        dst = d / f
        if dst.exists() and dst.stat().st_size > 0:
            continue
        urllib.request.urlretrieve(f"{BASE}/{target}/{f}", dst)
    return d


def read_ism(path: Path) -> list[str]:
    """DUD-E .ism（1 行 = "SMILES ID ..."）から SMILES を返す。"""
    smis = []
    for ln in path.read_text().splitlines():
        ln = ln.strip()
        if ln:
            smis.append(ln.split()[0])
    return smis


def scaffold_clusters(smiles: list[str]) -> Counter:
    """汎用 Bemis-Murcko スキャフォールド SMILES でクラスタリングし、クラスタ→件数を返す。"""
    c = Counter()
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            c["<invalid>"] += 1
            continue
        try:
            scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        except Exception:
            scaf = "<error>"
        c[scaf or "<acyclic>"] += 1
    return c


def summarize(name: str, clusters: Counter) -> None:
    n = sum(clusters.values())
    n_clusters = len(clusters)
    sizes = sorted(clusters.values(), reverse=True)
    multi = [s for s in sizes if s >= 2]
    in_multi = sum(multi)
    print(f"  {name}: {n} 分子, {n_clusters} クラスタ "
          f"(2件以上のクラスタ {len(multi)} 個に {in_multi} 分子 = {100*in_multi/max(n,1):.0f}%)")
    print(f"    最大クラスタ: {sizes[0] if sizes else 0} 分子, 上位5サイズ: {sizes[:5]}")


def main() -> None:
    targets = sys.argv[1:] or ["akt1", "ampc", "cxcr4", "hivrt", "kif11"]
    for t in targets:
        d = download(t)
        actives = read_ism(d / "actives_final.ism")
        decoys = read_ism(d / "decoys_final.ism")
        print(f"== {t} ==")
        summarize("actives", scaffold_clusters(actives))
        summarize("decoys ", scaffold_clusters(decoys))


if __name__ == "__main__":
    main()
