#!/usr/bin/env python3
"""化合物ライブラリの Morgan FP + Tanimoto + MCS クラスタリング（M3 前処理）。

FINDING.md の設計どおり、Morgan フィンガープリント（r=2, 2048bit）の Tanimoto で前ふるい
（O(n²) だが安価）し、しきい値以上で結ばれた連結成分を「同族列の候補クラスタ」とする。
各クラスタには代表（seed）と子分子を割り当て、共通コアは MCS（FindMCS）で確定する。
active/decoy を **統合**してクラスタリングする（実スクリーニングと同様）。

importable: `read_ism`, `morgan_fps`, `cluster_by_tanimoto`, `mcs_of`。
CLI: クラスタサイズ分布の統計を出す。

使い方:
    python3 tools/preprocess/cluster_library.py <target> [--tanimoto 0.6] [--max N]
"""
import argparse
import sys
from pathlib import Path

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFMCS
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

ROOT = Path(__file__).resolve().parent.parent.parent
DUDE = ROOT / "bench" / "data" / "dude"

_MORGAN = GetMorganGenerator(radius=2, fpSize=2048)


def read_ism(path: Path) -> list[tuple[str, str]]:
    """`.ism`（SMILES<空白>ID）を [(smiles, id), ...] で返す。"""
    out = []
    for ln in path.read_text().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        parts = ln.split()
        smi = parts[0]
        cid = parts[1] if len(parts) > 1 else f"mol{len(out)}"
        out.append((smi, cid))
    return out


def load_target(target: str, max_each: int | None = None) -> list[dict]:
    """active/decoy を統合して [{smiles,id,label,mol}, ...] を返す（パース不能は除外）。"""
    recs: list[dict] = []
    for label, fname in (("active", "actives_final.ism"), ("decoy", "decoys_final.ism")):
        items = read_ism(DUDE / target / fname)
        if max_each is not None:
            items = items[:max_each]
        for smi, cid in items:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            recs.append({"smiles": smi, "id": cid, "label": label, "mol": mol})
    return recs


def morgan_fps(mols: list[Chem.Mol]):
    return [_MORGAN.GetFingerprint(m) for m in mols]


def cluster_by_tanimoto(fps, threshold: float) -> list[list[int]]:
    """Tanimoto >= threshold で結ばれた連結成分（単連結クラスタ）を返す。

    O(n²) の総当たり Union-Find。検証規模（数千）では十分。
    """
    n = len(fps)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1 :])
        for off, s in enumerate(sims):
            if s >= threshold:
                union(i, i + 1 + off)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def mcs_of(mols: list[Chem.Mol], timeout: int = 10):
    """クラスタ分子群の MCS（共通コア）を返す（rdFMCS.MCSResult）。"""
    return rdFMCS.FindMCS(
        mols,
        timeout=timeout,
        atomCompare=rdFMCS.AtomCompare.CompareElements,
        bondCompare=rdFMCS.BondCompare.CompareOrder,
        ringMatchesRingOnly=True,
        completeRingsOnly=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("target")
    ap.add_argument("--tanimoto", type=float, default=0.6)
    ap.add_argument("--max", type=int, default=None, help="active/decoy 各上限（高速確認用）")
    args = ap.parse_args()

    recs = load_target(args.target, args.max)
    fps = morgan_fps([r["mol"] for r in recs])
    clusters = cluster_by_tanimoto(fps, args.tanimoto)
    clusters.sort(key=len, reverse=True)

    n = len(recs)
    multi = [c for c in clusters if len(c) > 1]
    n_in_multi = sum(len(c) for c in multi)
    print(f"== {args.target} (Morgan r2 + Tanimoto>={args.tanimoto}, 統合) ==")
    print(f"分子数: {n}  クラスタ数: {len(clusters)}  多member率: {n_in_multi}/{n} = {n_in_multi / n:.0%}")
    print(f"上位クラスタサイズ: {[len(c) for c in clusters[:10]]}")
    # active を含むクラスタの混在状況
    mixed = 0
    for c in multi:
        labels = {recs[i]["label"] for i in c}
        if "active" in labels and "decoy" in labels:
            mixed += 1
    print(f"多member クラスタ {len(multi)} 個中、active/decoy 混在 {mixed} 個")


if __name__ == "__main__":
    main()
