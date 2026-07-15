#!/usr/bin/env python3
"""isct_screening.py — 単一タンパク質に対しディレクトリ内リガンドを singleton ドッキングする。

各リガンドを「子を持たない seed のみクラスタ（core_atoms 空＝全自由度ドッキング）」として
1 本の screen ライブラリにまとめ、`isct-dock screen` を **1 プロセス**で実行する。
グリッドは共有ボックスで **1 度だけ**構築され全リガンドで再利用される（`dock` を N 回呼ぶのと違い
グリッド再構築・受容体パース・プロセス起動が 1 回で済む）。

使い方:
    isct_screening.py <receptor.pdbqt> <ligand_dir> [options]

位置引数:
    receptor            タンパク質 PDBQT ファイル
    ligand_dir          リガンド PDBQT を格納したディレクトリ（*.pdbqt）

主なオプション:
    --n_dockings N      ファイル名昇順で先頭 N 件のみ処理（既定: ディレクトリ内全件）
    --clean             実行後に中間ファイル（run.toml / library.toml）を削除（既定: 全て残す）
    --center X Y Z      共有探索ボックス中心（既定: 対象リガンド重心の平均）
    --size S            ボックス一辺（Å, 既定 22.5）
    --search MODE       探索モード tpe|vina|drift|standard|qvina2（既定 tpe）
    --n_runs N          1 リガンドあたりの独立 run 数（速度/精度トレード。既定: モード既定）
    --seed S            乱数シード（既定 42, 決定論）
    --prune_threshold T seed エネルギーが T を超えるリガンドをランキングから除外
                        （既定 1e9 ＝除外せず全件出力。ヒットのみ欲しければ 0.0 等に）
    --workdir DIR       中間/出力ファイル出力先（既定 ./isct_screening_out）
    --out PATH          ランキング TSV 出力先（既定 <workdir>/ranking.tsv）
    --bin PATH          isct-dock バイナリ（既定 <repo>/target/release/isct-dock）
"""
import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def heavy_centroid(pdbqt: Path):
    """PDBQT の重原子重心 (x,y,z) を返す（原子が無ければ None）。"""
    sx = sy = sz = 0.0
    n = 0
    for ln in pdbqt.read_text().splitlines():
        if ln.startswith(("ATOM", "HETATM")):
            elem = ln.split()[-1] if len(ln.split()) else ""
            if elem in ("H", "HD", "HS"):
                continue
            try:
                sx += float(ln[30:38]); sy += float(ln[38:46]); sz += float(ln[46:54])
            except ValueError:
                continue
            n += 1
    if n == 0:
        return None
    return (sx / n, sy / n, sz / n)


def main():
    ap = argparse.ArgumentParser(
        description="単一タンパク質にディレクトリ内リガンドを singleton ドッキング",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("receptor", type=Path, help="タンパク質 PDBQT")
    ap.add_argument("ligand_dir", type=Path, help="リガンド PDBQT ディレクトリ")
    ap.add_argument("--n_dockings", type=int, default=None,
                    help="ファイル名昇順で先頭 N 件のみ（既定: 全件）")
    ap.add_argument("--clean", action="store_true",
                    help="実行後に中間 TOML を削除（既定: 残す）")
    ap.add_argument("--center", type=float, nargs=3, metavar=("X", "Y", "Z"), default=None,
                    help="共有ボックス中心（既定: リガンド重心の平均）")
    ap.add_argument("--size", type=float, default=22.5, help="ボックス一辺 Å（既定 22.5）")
    ap.add_argument("--search", default="tpe",
                    choices=["tpe", "vina", "drift", "standard", "qvina2"],
                    help="探索モード（既定 tpe）")
    ap.add_argument("--n_runs", type=int, default=None,
                    help="1 リガンドあたり独立 run 数（既定: モード既定）")
    ap.add_argument("--seed", type=int, default=42, help="乱数シード（既定 42）")
    ap.add_argument("--prune_threshold", type=float, default=1e9,
                    help="seed エネルギー > T を除外（既定 1e9 ＝全件出力）")
    ap.add_argument("--workdir", type=Path, default=Path("isct_screening_out"),
                    help="中間/出力ファイル出力先（既定 ./isct_screening_out）")
    ap.add_argument("--out", type=Path, default=None,
                    help="ランキング TSV 出力先（既定 <workdir>/ranking.tsv）")
    ap.add_argument("--bin", type=Path, default=REPO / "target/release/isct-dock",
                    help="isct-dock バイナリ")
    args = ap.parse_args()

    # drift は CLI 上 standard の別名ではなく既定探索。--search の内部名は standard/vina/qvina2/tpe。
    # drift = 既定（standard）の直線探索ウォームスタート版なので standard にマップ。
    search_map = {"drift": "standard", "standard": "standard", "vina": "vina",
                  "qvina2": "qvina2", "tpe": "tpe"}
    search = search_map[args.search]

    receptor = args.receptor.resolve()
    if not receptor.is_file():
        sys.exit(f"error: receptor が見つからない: {receptor}")
    if not args.ligand_dir.is_dir():
        sys.exit(f"error: ligand_dir がディレクトリでない: {args.ligand_dir}")
    if not args.bin.is_file():
        sys.exit(f"error: isct-dock バイナリが無い: {args.bin}（cargo build --release 済？）")

    # リガンド収集: *.pdbqt をファイル名昇順、--n_dockings で先頭 N 件。
    # `*_isct_out.pdbqt`（dock の既定出力ポーズ）は入力から除外＝同一 dir に dock 出力が
    # 混在しても再ドッキングしない（マルチモデル pose の巨大 torsion 木化を防ぐ）。
    ligands = sorted((p for p in args.ligand_dir.glob("*.pdbqt")
                      if not p.stem.endswith("_isct_out")), key=lambda p: p.name)
    if not ligands:
        sys.exit(f"error: {args.ligand_dir} に *.pdbqt が無い")
    if args.n_dockings is not None:
        ligands = ligands[: args.n_dockings]
    print(f"[isct_screening] 対象リガンド: {len(ligands)} 件（{args.ligand_dir}）", flush=True)

    # 共有ボックス中心: 指定が無ければ対象リガンド重心の平均。
    if args.center is not None:
        center = tuple(args.center)
    else:
        cents = [c for c in (heavy_centroid(p) for p in ligands) if c is not None]
        if not cents:
            sys.exit("error: リガンドから座標を読めず中心を決定できない。--center を指定してください")
        center = (sum(c[0] for c in cents) / len(cents),
                  sum(c[1] for c in cents) / len(cents),
                  sum(c[2] for c in cents) / len(cents))
        print(f"[isct_screening] ボックス中心（リガンド重心平均）: "
              f"[{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}]", flush=True)

    workdir = args.workdir.resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    lib_path = workdir / "library.toml"
    cfg_path = workdir / "run.toml"
    out_path = (args.out.resolve() if args.out else workdir / "ranking.tsv")

    # run.toml: 共有ボックス・grid スコアリング（グリッド再利用）・探索設定。
    run_lines = [
        "# isct_screening.py 生成 — 全 singleton 共有ボックス screen 設定",
        "[box]",
        f"center = [{center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}]",
        f"size = [{args.size}, {args.size}, {args.size}]",
        "",
        "[run]",
        f"seed = {args.seed}",
        f'search = "{search}"',
        'scoring = "grid"',
    ]
    if args.n_runs is not None:
        run_lines.append(f"n_runs = {args.n_runs}")
    run_lines += ["", "[screen]", f"prune_threshold = {args.prune_threshold}", ""]
    cfg_path.write_text("\n".join(run_lines))

    # library.toml: 1 リガンド = 1 singleton クラスタ（seed のみ・core_atoms 空・pocket 指定なし）。
    # pocket を書かない＝既定ボックス共有＝グリッドを 1 度だけ構築し全リガンドで再利用。
    parts = []
    for i, lig in enumerate(ligands):
        cid = lig.stem
        parts.append(
            f'[[cluster]]\nid = "sng_{i}"\n\n'
            f'[cluster.seed]\ncompound_id = "{cid}"\n'
            f'pdbqt = "{lig.resolve()}"\ncore_atoms = []\n'
        )
    lib_path.write_text("\n".join(parts))

    # screen 実行（1 プロセス・rayon 並列）。
    cmd = [str(args.bin), "screen", str(cfg_path), str(receptor), str(lib_path),
           "--out", str(out_path)]
    print(f"[isct_screening] 実行: {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit(f"error: isct-dock screen 失敗 (exit {r.returncode})")

    print(f"[isct_screening] ランキング: {out_path}", flush=True)
    if args.clean:
        for p in (lib_path, cfg_path):
            p.unlink(missing_ok=True)
        print(f"[isct_screening] --clean: 中間ファイル削除（{lib_path.name}, {cfg_path.name}）。"
              f"ランキング TSV は保持。", flush=True)
    else:
        print(f"[isct_screening] 中間ファイル保持: {cfg_path}, {lib_path}", flush=True)


if __name__ == "__main__":
    main()
