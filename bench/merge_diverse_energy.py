#!/usr/bin/env python3
"""新 isct（ドリフト鎖, エネルギー込み）tsv と、以前の Vina 結果（results.*.md の表）をマージして
全標的 markdown テーブルを出力する。Vina は再実行せず流用。

  python3 bench/merge_diverse_energy.py <new_isct.tsv> <prior_results.md>
"""
import re
import sys
from pathlib import Path


def parse_prior(md: str):
    """results.202606182210.md の 14 列表から pdbid→(prior_isct_rmsd,prior_isct_ok,vina_rmsd,vina_ok,vina_t)。"""
    out = {}
    for ln in md.splitlines():
        if not ln.startswith("| "):
            continue
        c = [x.strip() for x in ln.strip().strip("|").split("|")]
        if len(c) != 14 or c[0] in ("pdbid",) or set(c[0]) <= set("-"):
            continue
        pid = c[0]
        out[pid] = dict(isct_rmsd=c[8], isct_ok=c[9], vina_rmsd=c[11], vina_ok=c[12], vina_t=c[13])
    return out


def parse_new(tsv: str):
    out = {}
    rows = []
    for ln in tsv.splitlines():
        if not ln.strip() or ln.startswith("#"):
            continue
        c = ln.split("\t")
        if c[0] == "pdbid":
            continue
        rows.append(c)
        out[c[0]] = c
    return out, rows


def main():
    new_tsv = Path(sys.argv[1]).read_text()
    prior_md = Path(sys.argv[2]).read_text()
    prior = parse_prior(prior_md)
    new, rows = parse_new(new_tsv)

    hdr = ["pdbid", "lig", "heavy", "tors", "macro", "MW",
           "isct_rmsd", "isct_ok", "isct_t", "isct_E", "cryst_E",
           "vina_rmsd", "vina_ok", "vina_t", "Δprev_ok"]
    print("| " + " | ".join(hdr) + " |")
    print("|" + "|".join("---" for _ in hdr) + "|")

    n_isct = n_vina = n_runs = 0
    reached = 0  # isct_E <= cryst_E（native 以下の結合エネルギーに到達）
    for c in rows:
        pid, lig, heavy, tors, rings, maxring, macro, mw, irmsd, iok, it, iE, cE = c[:13]
        pv = prior.get(pid, {})
        vrmsd, vok, vt = pv.get("vina_rmsd", "NA"), pv.get("vina_ok", "NA"), pv.get("vina_t", "NA")
        prev_ok = pv.get("isct_ok", "NA")
        # Δprev_ok: 新 isct_ok - 旧 isct_ok（成功 run 数差）
        def num(ok):
            m = re.match(r"(\d+)/", ok or "")
            return int(m.group(1)) if m else None
        ni, pi = num(iok), num(prev_ok)
        delta = f"{ni-pi:+d}" if (ni is not None and pi is not None) else "—"
        print(f"| {pid} | {lig} | {heavy} | {tors} | {macro} | {mw} "
              f"| {irmsd} | {iok} | {it} | {iE} | {cE} | {vrmsd} | {vok} | {vt} | {delta} |")
        ni = num(iok) or 0; nv = num(vok) or 0
        n_isct += ni; n_vina += nv; n_runs += 3
        try:
            if float(iE) <= float(cE) + 1e-6:
                reached += 1
        except ValueError:
            pass

    print()
    print(f"- 標的数 {len(rows)}, isct 成功 {n_isct}/{n_runs} ({100*n_isct/n_runs:.0f}%), "
          f"vina 成功 {n_vina}/{n_runs} ({100*n_vina/n_runs:.0f}%)")
    print(f"- isct_E ≤ cryst_E（結晶以下の結合エネルギーに到達）: {reached}/{len(rows)} 標的")


if __name__ == "__main__":
    main()
