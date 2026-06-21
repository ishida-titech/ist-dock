#!/usr/bin/env python3
"""クラスタライブラリ（PDBQT 群＋TOML マニフェスト）を生成する（M3 前処理）。

役割分担「前処理で拘束埋込」に従い、各子分子の MCS コアを seed コア配座へ **constrained
embed** して 3D 生成し、PDBQT 化する。コア原子の対応（seed と子で同順）は MCS クエリ原子順で
取り、PDBQT のどの ATOM 行かは座標最近傍で対応づける（Meeko は重原子座標を保つ）。

ROOT 化はエンジン側が「コアを動かす torsion を自動ロック」するため不要（Meeko の既定木で可）。

出力（既定 bench/data/library/<target>/）:
    receptor.pdbqt, mols/<id>.pdbqt, library.toml, labels.tsv

使い方（薄い縦スライス例）:
    python3 tools/preprocess/build_library.py ampc --actives-only --max-each 60 \
        --tanimoto 0.5 --max-clusters 1 --max-children 3 --out-dir bench/data/library/ampc_mini
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
from meeko import MoleculePreparation, PDBQTWriterLegacy

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cluster_library as cl  # noqa: E402
import prepare_receptor as pr  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent

# コアとして採用する MCS の最小原子数（小さすぎる共通部分は拘束に値しない）。
MIN_CORE_ATOMS = 6
# 子が scaffold 再利用する条件: MCS コアが子の重原子の一定割合以上を占めること。
# コアが小さいと R 基（分子の大半）が seed ポーズへ任意配置され受容体と衝突するため、
# 割合が低い子は再利用せず singleton（全自由度ドッキング）にフォールバックする。
MIN_CORE_FRACTION = 0.55


def embed(mol: Chem.Mol, seed: int, coord_map=None, fixed=None) -> Chem.Mol | None:
    """3D 埋め込み（任意で coord_map 拘束＋コア固定 MMFF）。失敗時 None。"""
    molh = Chem.AddHs(mol)
    if coord_map is not None:
        rc = AllChem.EmbedMolecule(molh, randomSeed=seed, coordMap=coord_map, useRandomCoords=True)
    else:
        params = AllChem.ETKDGv3()
        params.randomSeed = seed
        rc = AllChem.EmbedMolecule(molh, params)
    if rc != 0:
        rc = AllChem.EmbedMolecule(molh, randomSeed=seed + 1, useRandomCoords=True)
        if rc != 0:
            return None
    try:
        props = AllChem.MMFFGetMoleculeProperties(molh)
        ff = AllChem.MMFFGetMoleculeForceField(molh, props)
        if fixed:
            for a in fixed:
                ff.AddFixedPoint(a)
        ff.Minimize(maxIts=1000)
    except Exception:
        pass
    return molh


def embed_constrained(mol: Chem.Mol, seed: int, coord_map) -> Chem.Mol | None:
    """コア座標を coord_map に拘束して 3D 生成（子分子用）。失敗時 None（→ 呼び出し側で singleton 化）。
    無拘束フォールバックはしない（コアが seed と一致しないと再利用の意味がないため）。
    反復は ETKDG params の maxIterations で上限化する（無制限だと特定分子で停止するため）。"""
    molh = Chem.AddHs(mol)
    ps = AllChem.ETKDGv3()
    ps.randomSeed = seed
    ps.useRandomCoords = True
    ps.maxIterations = 200  # 暴走防止の上限
    ps.SetCoordMap(coord_map)
    rc = AllChem.EmbedMolecule(molh, ps)
    if rc != 0:
        return None
    try:
        props = AllChem.MMFFGetMoleculeProperties(molh)
        ff = AllChem.MMFFGetMoleculeForceField(molh, props)
        for a in coord_map:
            ff.AddFixedPoint(a)
        ff.Minimize(maxIts=400)
    except Exception:
        pass
    return molh


def write_pdbqt(molh: Chem.Mol, out_path: Path, rigid_smarts=None, rigid_bonds=None) -> None:
    """3D mol を SDF 経由で mk_prepare_ligand により PDBQT 化する。

    `rigid_smarts` + `rigid_bonds`（SMARTS 内の (i, j) ペア列）を渡すと、その全結合を
    剛体化する（コアを単一剛体にして「動かない Scaffold」を保証する）。これは CLI 経由（低速）。
    剛体指定が無い通常ケースは **meeko を in-process** で呼び高速化する（CLI のプロセス起動回避）。
    """
    if not (rigid_smarts and rigid_bonds):
        # 高速パス: in-process meeko（mk_prepare_ligand.py のプロセス起動 ~1.5s/件 を回避）。
        pdbqt, ok, err = PDBQTWriterLegacy.write_string(MoleculePreparation().prepare(molh)[0])
        if not ok:
            raise RuntimeError(f"meeko prepare failed: {err}")
        Path(out_path).write_text(pdbqt)
        return
    with tempfile.NamedTemporaryFile("w", suffix=".sdf", delete=False) as tf:
        sdf = Path(tf.name)
    w = Chem.SDWriter(str(sdf))
    w.write(molh)
    w.close()
    cmd = ["mk_prepare_ligand.py", "-i", str(sdf), "-o", str(out_path)]
    cmd += ["-r", rigid_smarts, "-b"]
    for i, j in rigid_bonds:
        cmd += [str(i), str(j)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    finally:
        sdf.unlink(missing_ok=True)


def pdbqt_coords(path: Path) -> list[tuple[float, float, float]]:
    """PDBQT の ATOM/HETATM 座標を出現順（Rust パーサと同順）で返す。"""
    out = []
    for ln in path.read_text().splitlines():
        if ln.startswith(("ATOM", "HETATM")):
            out.append((float(ln[30:38]), float(ln[38:46]), float(ln[46:54])))
    return out


def map_core_to_pdbqt(core_xyz, pdbqt_xyz) -> list[int]:
    """各コア原子座標に最近傍の PDBQT ATOM 行 idx を返す（重複・遠距離は失敗）。"""
    used: set[int] = set()
    idxs: list[int] = []
    for cx, cy, cz in core_xyz:
        best, bd = -1, 1e9
        for j, (px, py, pz) in enumerate(pdbqt_xyz):
            if j in used:
                continue
            d = (px - cx) ** 2 + (py - cy) ** 2 + (pz - cz) ** 2
            if d < bd:
                bd, best = d, j
        if best < 0 or bd > 0.25:  # 0.5 Å 超は不一致とみなす
            raise RuntimeError(f"core atom not matched in PDBQT (min dist²={bd:.3f})")
        used.add(best)
        idxs.append(best)
    return idxs


def conf_xyz(molh: Chem.Mol, atoms) -> list[tuple[float, float, float]]:
    conf = molh.GetConformer()
    return [tuple(conf.GetAtomPosition(a)) for a in atoms]


def build_cluster(members, query, out_dir: Path, cluster_idx: int, max_children: int, seed: int):
    """1 クラスタの seed・子を生成し、マニフェスト断片（dict）と labels を返す。"""
    mols_dir = out_dir / "mols"
    mols_dir.mkdir(parents=True, exist_ok=True)

    # 注: コアの剛体性はエンジン側が保証する（seed ドッキング・子最適化の双方で「コア原子を
    # 動かす torsion」を自動ロックする）ため、Meeko 側での剛体化は不要。

    seed_rec = members[0]
    seed_h = embed(seed_rec["mol"], seed)
    if seed_h is None:
        return None
    seed_match = seed_h.GetSubstructMatch(query)
    if len(seed_match) < MIN_CORE_ATOMS:
        return None
    # コアが seed の重原子の十分な割合を占めないクラスタは再利用に値しない（→ 全 singleton）。
    if len(seed_match) < MIN_CORE_FRACTION * seed_rec["mol"].GetNumHeavyAtoms():
        return None
    seed_core_xyz = conf_xyz(seed_h, seed_match)

    seed_pdbqt = mols_dir / f"{seed_rec['id']}.pdbqt"
    write_pdbqt(seed_h, seed_pdbqt)
    seed_core_idx = map_core_to_pdbqt(seed_core_xyz, pdbqt_coords(seed_pdbqt))

    seed_pt = [Point3D(*p) for p in seed_core_xyz]

    cluster = {
        "id": f"{out_dir.name}_c{cluster_idx}",
        "seed": {"compound_id": seed_rec["id"], "pdbqt": f"mols/{seed_rec['id']}.pdbqt",
                 "core_atoms": seed_core_idx},
        "children": [],
    }
    labels = [(seed_rec["id"], seed_rec["label"], cluster["id"], "seed")]

    for child_rec in members[1 : 1 + max_children]:
        cm = child_rec["mol"]
        molh0 = Chem.AddHs(cm)
        child_match = molh0.GetSubstructMatch(query)
        if len(child_match) != len(seed_match):
            continue
        # コアが子の重原子の十分な割合でなければ再利用しない（→ orphan＝singleton）。
        if len(child_match) < MIN_CORE_FRACTION * cm.GetNumHeavyAtoms():
            continue
        # コアを seed コア座標へ拘束 embed（反復上限付きで暴走防止）。エンジンは実行時にコアを
        # seed ドッキング後コアへ重ね合わせるため、子コアは seed コアと同一幾何である必要がある。
        coord_map = {child_match[q]: seed_pt[q] for q in range(len(child_match))}
        molh = embed_constrained(cm, seed + 1, coord_map)
        if molh is None:
            continue
        child_match = molh.GetSubstructMatch(query)
        if len(child_match) != len(seed_match):
            continue
        # コアを seed コア座標に厳密一致させる（残差ドリフト除去）。
        conf = molh.GetConformer()
        for q, a in enumerate(child_match):
            conf.SetAtomPosition(a, seed_pt[q])

        child_pdbqt = mols_dir / f"{child_rec['id']}.pdbqt"
        try:
            write_pdbqt(molh, child_pdbqt)
            child_core_idx = map_core_to_pdbqt(conf_xyz(molh, child_match), pdbqt_coords(child_pdbqt))
        except (RuntimeError, subprocess.CalledProcessError) as e:
            print(f"  skip child {child_rec['id']}: {e}", file=sys.stderr)
            continue

        cluster["children"].append({
            "compound_id": child_rec["id"],
            "pdbqt": f"mols/{child_rec['id']}.pdbqt",
            "core_atoms": child_core_idx,
        })
        labels.append((child_rec["id"], child_rec["label"], cluster["id"], "child"))

    if not cluster["children"]:
        return None
    return cluster, labels


def build_singleton(rec, out_dir: Path, cluster_idx: int, seed: int):
    """子を持たない seed のみクラスタ（＝通常の全自由度ドッキング）。
    singleton やクラスタ化に失敗した orphan を全件カバーするために使う。core_atoms は空
    （子が無いので不要。エンジンは seed を自由ドッキングする）。"""
    mols_dir = out_dir / "mols"
    mols_dir.mkdir(parents=True, exist_ok=True)
    h = embed(rec["mol"], seed)
    if h is None:
        return None
    pdbqt = mols_dir / f"{rec['id']}.pdbqt"
    try:
        write_pdbqt(h, pdbqt)
    except (RuntimeError, subprocess.CalledProcessError):
        return None
    cid = f"{out_dir.name}_s{cluster_idx}"
    cluster = {
        "id": cid,
        "seed": {"compound_id": rec["id"], "pdbqt": f"mols/{rec['id']}.pdbqt", "core_atoms": []},
        "children": [],
    }
    return cluster, [(rec["id"], rec["label"], cid, "seed")]


# 1 クラスタで MCS 計算・子拘束 embed に使うメンバー上限。大クラスタ全体で MCS を取ると
# 共通コアが痩せて拘束が無意味になり、かつ子 embed が逐次で重い。上限超過分は singleton 化する。
MAX_CLUSTER_MEMBERS = 12


def _proc_cluster(task):
    """並列ワーカ: 1 グループ（size>=2）を処理。
    返り値: ("cluster", cluster, labels, orphan_recs) | ("fallback", members)。
    上限超過メンバーは orphan として返し singleton にフォールバックさせる。
    どの分子で例外が出てもプール全体を落とさず singleton 化へフォールバックする。"""
    try:
        return _proc_cluster_inner(task)
    except Exception:  # noqa: BLE001
        return ("fallback", task[0])


def _proc_cluster_inner(task):
    members, out_dir_str, idx, seed = task
    out_dir = Path(out_dir_str)
    use = members[:MAX_CLUSTER_MEMBERS]
    overflow = members[MAX_CLUSTER_MEMBERS:]
    mcs = cl.mcs_of([m["mol"] for m in use], timeout=3)  # 大量クラスタ向けに MCS を短時間で
    if mcs.numAtoms < MIN_CORE_ATOMS or not mcs.smartsString:
        return ("fallback", members)
    query = Chem.MolFromSmarts(mcs.smartsString)
    if query is None:
        return ("fallback", members)
    built = build_cluster(use, query, out_dir, idx, len(use) - 1, seed)
    if built is None:
        return ("fallback", members)
    cluster, labels = built
    covered = {lab[0] for lab in labels}
    orphans = [m for m in use if m["id"] not in covered] + overflow
    return ("cluster", cluster, labels, orphans)


def _proc_singleton(task):
    """並列ワーカ: 1 分子を seed のみクラスタ化。例外は None（=失敗）に握り潰す。"""
    rec, out_dir_str, idx, seed = task
    try:
        return build_singleton(rec, Path(out_dir_str), idx, seed)
    except Exception:  # noqa: BLE001
        return None


def build_full_coverage(recs, groups, out_dir, box_center, box_size, seed, jobs):
    """全リガンドを 1 件残らずカバーするライブラリを構築する。
    size>=2 のグループは scaffold 再利用クラスタ（共通コアが小さい/embed 失敗の orphan は
    singleton にフォールバック）。size==1 は seed のみクラスタ。全件が seed か child になる。"""
    from multiprocessing import Pool

    multi = [g for g in groups if len(g) >= 2]
    singles = [g[0] for g in groups if len(g) == 1]
    print(f"full-coverage: groups(>=2)={len(multi)} singletons={len(singles)} total_mols={len(recs)}", flush=True)

    out_clusters, all_labels = [], []
    uncovered = [recs[i] for i in singles]  # singleton + 後で orphan を追加

    # Phase A: 多メンバークラスタ（並列）。
    tasks = [([recs[i] for i in g], str(out_dir), ci, seed) for ci, g in enumerate(multi)]
    n_cl = n_child = 0
    with Pool(jobs) as pool:
        for res in pool.imap_unordered(_proc_cluster, tasks, chunksize=4):
            if res[0] == "cluster":
                _, cluster, labels, orphans = res
                out_clusters.append(cluster)
                all_labels.extend(labels)
                uncovered.extend(orphans)
                n_cl += 1
                n_child += len(cluster["children"])
            else:  # fallback: 全メンバーを singleton 化
                uncovered.extend(res[1])
    print(f"  Phase A: {n_cl} reuse-clusters, {n_child} children; uncovered→singleton={len(uncovered)}", flush=True)

    # Phase B: singleton + orphan（並列）。
    stasks = [(rec, str(out_dir), si, seed) for si, rec in enumerate(uncovered)]
    n_single = n_fail = 0
    with Pool(jobs) as pool:
        for built in pool.imap_unordered(_proc_singleton, stasks, chunksize=8):
            if built is None:
                n_fail += 1
                continue
            cluster, labels = built
            out_clusters.append(cluster)
            all_labels.extend(labels)
            n_single += 1
    print(f"  Phase B: {n_single} singleton-seeds, {n_fail} embed失敗", flush=True)

    write_manifest(out_clusters, box_center, box_size, out_dir)
    lab_lines = ["compound_id\tlabel\tcluster_id\trole"]
    lab_lines += ["\t".join(x) for x in all_labels]
    (out_dir / "labels.tsv").write_text("\n".join(lab_lines) + "\n")
    covered = len(all_labels)
    print(f"library: {out_dir}/library.toml  clusters={len(out_clusters)} "
          f"(reuse={n_cl}, singleton={n_single}), covered={covered}/{len(recs)} "
          f"({100*covered/len(recs):.1f}%), children={n_child}", flush=True)


def toml_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def write_manifest(clusters, box_center, box_size, out_dir: Path) -> None:
    lines = []
    for c in clusters:
        lines.append("[[cluster]]")
        lines.append(f'id = "{toml_escape(c["id"])}"')
        lines.append(f"pocket_center = [{box_center[0]}, {box_center[1]}, {box_center[2]}]")
        lines.append(f"pocket_size = [{box_size[0]}, {box_size[1]}, {box_size[2]}]")
        lines.append("")
        s = c["seed"]
        lines.append("[cluster.seed]")
        lines.append(f'compound_id = "{toml_escape(s["compound_id"])}"')
        lines.append(f'pdbqt = "{s["pdbqt"]}"')
        lines.append(f"core_atoms = {s['core_atoms']}")
        lines.append("")
        for ch in c["children"]:
            lines.append("[[cluster.child]]")
            lines.append(f'compound_id = "{toml_escape(ch["compound_id"])}"')
            lines.append(f'pdbqt = "{ch["pdbqt"]}"')
            lines.append(f"core_atoms = {ch['core_atoms']}")
            lines.append("")
    (out_dir / "library.toml").write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("target")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--tanimoto", type=float, default=0.5)
    ap.add_argument("--max-each", type=int, default=None, help="active/decoy 各読込上限")
    ap.add_argument("--actives-only", action="store_true")
    ap.add_argument("--max-clusters", type=int, default=1)
    ap.add_argument("--max-children", type=int, default=3)
    ap.add_argument("--box-size", type=float, default=22.5)
    ap.add_argument("--seed", type=int, default=0xC0FFEE)
    ap.add_argument("--full-coverage", action="store_true",
                    help="全リガンドをカバー（size>=2 は再利用クラスタ、残りは seed のみ）。enrichment 評価用")
    ap.add_argument("--jobs", type=int, default=8, help="--full-coverage の並列数")
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "bench" / "data" / "library" / args.target
    out_dir.mkdir(parents=True, exist_ok=True)

    # 受容体 PDBQT ＋ box。
    pr.receptor_pdbqt(args.target, out_dir)
    box_center = [round(v, 3) for v in pr.box_center_from_mol2(args.target)]
    box_size = [args.box_size] * 3

    # 読込・クラスタリング。
    recs = cl.load_target(args.target, args.max_each)
    if args.actives_only:
        recs = [r for r in recs if r["label"] == "active"]
    fps = cl.morgan_fps([r["mol"] for r in recs])
    groups = cl.cluster_by_tanimoto(fps, args.tanimoto)

    if args.full_coverage:
        groups.sort(key=len, reverse=True)
        build_full_coverage(recs, groups, out_dir, box_center, box_size, args.seed, args.jobs)
        return

    groups = [g for g in groups if len(g) >= 2]
    groups.sort(key=len, reverse=True)
    print(f"clusters(size>=2)={len(groups)}; top sizes={[len(g) for g in groups[:8]]}")

    out_clusters = []
    all_labels = []
    ci = 0
    for g in groups:
        if ci >= args.max_clusters:
            break
        # MCS は「seed＋実際に拘束する子」の部分集合で計算する（クラスタ全体だと共通コアが
        # 小さくなりすぎ、Scaffold 拘束が無意味になる）。
        members = [recs[i] for i in g][: 1 + args.max_children]
        mcs = cl.mcs_of([m["mol"] for m in members])
        if mcs.numAtoms < MIN_CORE_ATOMS or not mcs.smartsString:
            continue
        query = Chem.MolFromSmarts(mcs.smartsString)
        if query is None:
            continue
        built = build_cluster(members, query, out_dir, ci, args.max_children, args.seed)
        if built is None:
            continue
        cluster, labels = built
        out_clusters.append(cluster)
        all_labels.extend(labels)
        print(f"  cluster {cluster['id']}: seed={cluster['seed']['compound_id']} "
              f"core={len(cluster['seed']['core_atoms'])} atoms, "
              f"children={len(cluster['children'])} (MCS {mcs.numAtoms} atoms)")
        ci += 1

    if not out_clusters:
        sys.exit("有効なクラスタを生成できませんでした（しきい値や MIN_CORE_ATOMS を調整）")

    write_manifest(out_clusters, box_center, box_size, out_dir)
    lab_lines = ["compound_id\tlabel\tcluster_id\trole"]
    lab_lines += ["\t".join(x) for x in all_labels]
    (out_dir / "labels.tsv").write_text("\n".join(lab_lines) + "\n")

    print(f"library: {out_dir}/library.toml  ({len(out_clusters)} clusters, "
          f"{sum(len(c['children']) for c in out_clusters)} children)  box_center={box_center}")


if __name__ == "__main__":
    main()
