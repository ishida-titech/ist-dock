# IST-Dock

**Incremental Scaffold-Templated Docking** — a fast, deterministic, ML-free molecular docking and virtual-screening engine written in Rust, designed to scale toward 10-million-compound libraries.

IST-Dock uses only geometry and physicochemical calculation (an AutoDock Vina–compatible scoring function) — no machine-learning model. Its scores match `vina --score_only` to within `<1e-2`, and every run is **bit-for-bit reproducible** for a given seed.

> Status: research software, `v1.0.0`. The detailed manual (`MANUAL.md`) is written in Japanese.

---

## Highlights

- **ML-free & deterministic** — pure geometric/physicochemical docking with a Vina-compatible force field (`gauss1`, `gauss2`, `repulsion`, `hydrophobic`, `hbond` + torsional entropy). Same seed → identical results across CPUs.
- **Fast default search (`tpe`)** — a low-budget preset built on a shared-observation Tree-structured Parzen Estimator with a local refinement chain, local-search skipping, and rigid-ligand lightening.
- **Scaffold-templated screening** — the `screen` mode clusters congeneric ligands by a shared scaffold (MCS core), docks a representative seed once, then places analogs by rigid core overlay + local tuning (CBdock-style), building a shared receptor grid once for the whole library.
- **Vina-faithful modes** available for validation (`qvina2` = faithful QuickVina2, `vina` = faithful AutoDock Vina MC).

### Accuracy / speed (Astex Diverse Set, 85 targets, 10 seeds)

The default `tpe` preset reaches **≈69% top-1 redocking success** (heavy-atom RMSD < 2 Å), close to AutoDock Vina at `exhaustiveness=8` (**≈71%**), while using **~20× fewer energy evaluations** and **~9× less wall-clock time**.

---

## Build

Requires a Rust toolchain (`rust-version = 1.92`, stable).

```sh
cargo build --release        # optimized build
cargo test                   # unit tests
```

The binary is produced at `./target/release/ist-dock`.

The default build is portable baseline `x86-64` with strict `f64` arithmetic (deterministic, bit-identical across machines). An optional AVX2 build (`-C target-cpu=x86-64-v3`) speeds up the grid path **without changing results**. See `MANUAL.md` §1.1.

---

## Quick start

### 1. Score a fixed pose (no search)

```sh
ist-dock score receptor.pdbqt ligand.pdbqt
```

Reports the Vina-compatible per-term breakdown, the intermolecular energy, and the estimated free energy of binding.

### 2. Dock a single ligand

Docking needs a TOML config with the search box (see below). The config is the **first positional argument**:

```sh
ist-dock dock run.toml receptor.pdbqt ligand.pdbqt --out pose.pdbqt
# redocking: add a reference pose to report per-mode RMSD
ist-dock dock run.toml receptor.pdbqt ligand.pdbqt --native crystal.pdbqt
```

### 3. Screen a cluster library

```sh
ist-dock screen run.toml receptor.pdbqt library.toml --out ranking.tsv
```

`library.toml` lists the ligands. Each entry may be a standalone molecule (a "singleton" cluster = independent full docking) or a scaffold cluster with a seed + children. See `MANUAL.md` §2.3.

### Minimal `run.toml`

```toml
[box]
center = [65.6, 31.1, 117.8]   # binding-site center (Å)
size   = [22.5, 22.5, 22.5]    # search box side lengths (Å)

[run]
seed    = 42
search  = "tpe"                # default search mode
scoring = "grid"               # "grid" (precomputed, recommended) or "direct"
```

Annotated example configs live in `configs/dock.example.toml` and `configs/screen.example.toml`.

---

## Commands

| Command | Purpose | Positional args |
|---|---|---|
| `score`  | Score a fixed pose (no search) | `<receptor.pdbqt> <ligand.pdbqt>` |
| `dock`   | Dock one ligand, write the best pose | `<config.toml> <receptor.pdbqt> <ligand.pdbqt>` |
| `screen` | Dock a cluster library, write a ranking | `<config.toml> <receptor.pdbqt> <library.toml>` |
| `pack`   | Pack a TOML library into a memmap binary | `<manifest.toml> <out.bin>` |

### Search modes (`--search`, or the config `search` field)

| Mode | Description |
|---|---|
| `tpe` *(default)* | Low-budget preset: shared TPE + local chain + local-search skip + rigid lightening (`n_runs=1`). |
| `drift` | Deterministic drift-chain ILS (best-of-60). |
| `drift_qvina2` | Drift chain + local-search skip (IST-specific). |
| `qvina2` | Faithful QuickVina2 (Vina MC + skip, `exhaustiveness=8`). |
| `vina` | Faithful AutoDock Vina Monte-Carlo. |

Inputs and outputs are AutoDock PDBQT; prepare receptors/ligands with the usual tools (e.g. Meeko / AutoDockTools). Parallelism is controlled by `RAYON_NUM_THREADS`.

---

## Repository layout

```
crates/ist-core   atom types, Vina scoring, geometry (pure computation)
crates/ist-io     PDBQT parser, TOML config, cluster library (+ memmap binary)
crates/ist-dock   ligand torsion tree, optimizer (BFGS), search (TPE/MC), energy, screen phases
crates/ist-cli    the `ist-dock` binary (score / dock / screen / pack)
tools/            preprocessing scripts (RDKit / Meeko)
```

---

## Documentation

- **`MANUAL.md`** — full user manual (build, every command and option, config reference, output format). *Written in Japanese.*

## License

Apache-2.0.
