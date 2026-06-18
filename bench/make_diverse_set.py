#!/usr/bin/env python3
"""多様な protein-ligand 複合体データセットを自動前処理する（isct-dock vs Vina 比較用）。

母集合は Astex Diverse Set（多様性ベンチの定番）＋大環状リガンド数件。各 PDB について:
  1. RCSB から PDB を取得（第 1 モデル・主要 altloc）。
  2. HETATM から **薬物様リガンドを自動選択**（緩衝剤・イオン・補酵素・糖を blocklist 除外、
     重原子 10〜70・炭素を含む最大残基）。
  3. RCSB ChemComp API からそのリガンドの SMILES_CANONICAL を取得。
  4. 受容体（蛋白 ATOM）→ mk_prepare_receptor、リガンド（HETATM＋SMILES で結合次数付与）
     → mk_prepare_ligand で PDBQT 化。box 中心＝リガンド重心。
  5. 性質（重原子数・torsion DOF・大環状有無・最大環・環数・MW・回転結合数）を info.json に記録。

出力: bench/data/diverse/<pdbid>/{receptor.pdbqt, ligand.pdbqt, config.toml, info.json}
失敗（共有結合・SMILES 取得不可・Meeko エラー等）はスキップしてログする。

使い方: python3 bench/make_diverse_set.py [pdbid ...]
"""
import json
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "bench" / "data" / "diverse"

# Astex Diverse Set（一部）＋ 大環状（1fkb=FK506, 1cwa=cyclosporin）。
PDB_IDS = """
1g9v 1gkc 1gm8 1gpk 1hnn 1hp0 1hq2 1hvy 1hwi 1hww 1ig3 1j3j 1jd0 1jje 1ke5
1kzk 1l2s 1l7f 1lpz 1lrh 1m2z 1meh 1mmv 1mzc 1n1m 1n2j 1n2v 1n46 1nav 1of1
1of6 1opk 1oq5 1owe 1oyt 1p2y 1p62 1pmn 1q1g 1q41 1q4g 1r1h 1r55 1r58 1r9o
1s19 1s3v 1sg0 1sj0 1sqn 1t40 1t46 1t9b 1tt1 1u4d 1unl 1uou 1v0p 1v48 1v4s
1xm6 1xoq 1y6b 1ydr 1ygc 1yqy 1yvf 1ywr 2bm2 2br1 2bsm 1fkb 1cwa
""".split()

BLOCKLIST = {
    "HOH", "DOD", "WAT",
    "NA", "CL", "MG", "ZN", "CA", "K", "MN", "FE", "CU", "NI", "CO", "CD", "HG", "FE2",
    "SO4", "PO4", "PI", "NO3", "NH4", "BCT", "ACT", "FMT", "ACY", "CIT", "FLC",
    "GOL", "EDO", "PEG", "PG4", "1PE", "P6G", "MPD", "DMS", "DMSO", "BME", "IMD",
    "TRS", "EPE", "MES", "EOH", "OXY", "PGE", "2PE", "12P", "15P", "MRD",
    "NAG", "BMA", "MAN", "GAL", "FUC", "GLC", "BGC", "NDG", "SUC",
    "NAD", "NAP", "NDP", "NAI", "FAD", "FMN", "HEM", "HEC", "HEA", "SAM", "SAH",
    "ATP", "ADP", "AMP", "ANP", "GTP", "GDP", "GNP", "COA", "PLP", "TPP", "FES",
}


def http_get(url: str, timeout: int = 30) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


def download_pdb(pdbid: str) -> str:
    return http_get(f"https://files.rcsb.org/download/{pdbid.upper()}.pdb").decode("utf-8", "replace")


def first_model_atom_lines(pdb_text: str):
    """第 1 モデルの ATOM/HETATM 行を返す（altloc は ' ' か 'A' のみ）。"""
    out = []
    for ln in pdb_text.splitlines():
        if ln.startswith("ENDMDL"):
            break
        if ln.startswith(("ATOM", "HETATM")):
            alt = ln[16] if len(ln) > 16 else " "
            if alt not in (" ", "A"):
                continue
            out.append(ln)
    return out


def element_of(line: str) -> str:
    e = line[76:78].strip() if len(line) >= 78 else ""
    if not e:
        name = line[12:16].strip()
        e = "".join(c for c in name if c.isalpha())[:1]
    return e.upper()


def pick_ligand(atom_lines):
    """blocklist 外・炭素含む・重原子 10〜70 の最大 HETATM 残基を選ぶ。(resname, chain, resseq)。"""
    groups = {}
    for ln in atom_lines:
        if not ln.startswith("HETATM"):
            continue
        res = ln[17:20].strip()
        if res in BLOCKLIST:
            continue
        key = (res, ln[21], ln[22:26].strip())
        groups.setdefault(key, []).append(ln)
    best, best_n = None, 0
    for key, lns in groups.items():
        heavy = [l for l in lns if element_of(l) != "H"]
        has_c = any(element_of(l) == "C" for l in heavy)
        n = len(heavy)
        if has_c and 10 <= n <= 70 and n > best_n:
            best, best_n = key, n
    return best


def fetch_smiles(ccd: str) -> str | None:
    try:
        d = json.loads(http_get(f"https://data.rcsb.org/rest/v1/core/chemcomp/{ccd}"))
    except Exception:
        return None
    for e in d.get("pdbx_chem_comp_descriptor", []):
        if e.get("type") == "SMILES_CANONICAL" and e.get("program", "").startswith("OpenEye"):
            return e["descriptor"]
    for e in d.get("pdbx_chem_comp_descriptor", []):
        if e.get("type") == "SMILES_CANONICAL":
            return e["descriptor"]
    return d.get("rcsb_chem_comp_descriptor", {}).get("smiles")


def write_receptor(atom_lines, out_dir: Path):
    prot = out_dir / "protein.pdb"
    prot.write_text("\n".join(l for l in atom_lines if l.startswith("ATOM")) + "\nEND\n")
    base = str(out_dir / "receptor")
    # -a: 欠損原子のある残基を削除（生 PDB の欠損・末端に頑健）。--default_altloc A。
    subprocess.run(["mk_prepare_receptor.py", "--read_pdb", str(prot), "-o", base, "-p",
                    "-a", "--default_altloc", "A"],
                   check=True, capture_output=True, text=True)
    prot.unlink(missing_ok=True)
    if not (out_dir / "receptor.pdbqt").exists():
        raise RuntimeError("receptor.pdbqt not produced")


def build_ligand(atom_lines, key, smiles, out_dir: Path):
    res, chain, seq = key
    het = [l for l in atom_lines if l.startswith("HETATM") and l[17:20].strip() == res
           and l[21] == chain and l[22:26].strip() == seq]
    lig_pdb = out_dir / "lig_raw.pdb"
    lig_pdb.write_text("\n".join(het) + "\nEND\n")
    raw = Chem.MolFromPDBFile(str(lig_pdb), sanitize=False, removeHs=True)
    if raw is None:
        raise RuntimeError("RDKit could not read ligand PDB")
    template = Chem.MolFromSmiles(smiles)
    if template is None:
        raise RuntimeError(f"invalid SMILES: {smiles}")
    mol = AllChem.AssignBondOrdersFromTemplate(template, raw)  # 結合次数を付与
    molh = Chem.AddHs(mol, addCoords=True)
    sdf = out_dir / "ligand.sdf"
    w = Chem.SDWriter(str(sdf))
    w.write(molh)
    w.close()
    # --rigid_macrocycles: 大環状/大環を入力配座のまま剛体化（Meeko の CG0/G0 グルー擬原子を回避。
    # 本エンジンは柔軟マクロ環の glue 表現を未対応のため）。
    subprocess.run(["mk_prepare_ligand.py", "-i", str(sdf), "-o", str(out_dir / "ligand.pdbqt"),
                    "--rigid_macrocycles"],
                   check=True, capture_output=True, text=True)
    lig_pdb.unlink(missing_ok=True)
    if not (out_dir / "ligand.pdbqt").exists():
        raise RuntimeError("ligand.pdbqt not produced")
    return mol


def ligand_properties(mol, lig_pdbqt: Path) -> dict:
    ri = mol.GetRingInfo()
    sizes = [len(r) for r in ri.AtomRings()]
    max_ring = max(sizes) if sizes else 0
    torsdof = 0
    heavy = 0
    for ln in lig_pdbqt.read_text().splitlines():
        if ln.startswith("TORSDOF"):
            torsdof = int(ln.split()[1])
        elif ln.startswith(("ATOM", "HETATM")):
            if ln.split()[-1] not in ("H", "HD", "HS"):
                heavy += 1
    return {
        "heavy_atoms": heavy,
        "torsdof": torsdof,
        "macrocycle": bool(max_ring >= 10),
        "max_ring": max_ring,
        "n_rings": rdMolDescriptors.CalcNumRings(mol),
        "rot_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "mw": round(Descriptors.MolWt(mol), 1),
    }


def box_center(lig_pdbqt: Path):
    xs, ys, zs = [], [], []
    for ln in lig_pdbqt.read_text().splitlines():
        if ln.startswith(("ATOM", "HETATM")):
            xs.append(float(ln[30:38])); ys.append(float(ln[38:46])); zs.append(float(ln[46:54]))
    n = len(xs)
    return [round(sum(xs) / n, 3), round(sum(ys) / n, 3), round(sum(zs) / n, 3)]


def prepare(pdbid: str) -> str:
    pdbid = pdbid.lower()
    out_dir = OUT / pdbid
    out_dir.mkdir(parents=True, exist_ok=True)
    pdb = download_pdb(pdbid)
    atoms = first_model_atom_lines(pdb)
    key = pick_ligand(atoms)
    if key is None:
        raise RuntimeError("no drug-like ligand found")
    smiles = fetch_smiles(key[0])
    if not smiles:
        raise RuntimeError(f"no SMILES for {key[0]}")
    write_receptor(atoms, out_dir)
    mol = build_ligand(atoms, key, smiles, out_dir)
    props = ligand_properties(mol, out_dir / "ligand.pdbqt")
    center = box_center(out_dir / "ligand.pdbqt")
    (out_dir / "config.toml").write_text(
        f"[box]\ncenter = [{center[0]}, {center[1]}, {center[2]}]\nsize = [20.0, 20.0, 20.0]\n"
        f'[run]\nseed = 42\nn_runs = 8\nscoring = "grid"\n'
    )
    info = {"pdbid": pdbid, "ligand": key[0], "smiles": smiles, "center": center, **props}
    (out_dir / "info.json").write_text(json.dumps(info))
    return (f"OK   {pdbid} lig={key[0]:4s} heavy={props['heavy_atoms']:2d} "
            f"tors={props['torsdof']:2d} macro={props['macrocycle']} maxring={props['max_ring']}")


def main():
    ids = sys.argv[1:] or PDB_IDS
    OUT.mkdir(parents=True, exist_ok=True)
    ok = 0
    for pid in ids:
        try:
            print(prepare(pid), flush=True)
            ok += 1
        except subprocess.CalledProcessError as e:
            print(f"FAIL {pid}: meeko/subprocess error", flush=True)
        except Exception as e:
            print(f"FAIL {pid}: {e}", flush=True)
    print(f"\nprepared {ok}/{len(ids)} targets in {OUT}")


if __name__ == "__main__":
    main()
