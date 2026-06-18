# PLAN.OPTIMIZE_DOCKING — 通常ドッキングの高速化

中核ドッキング（search / optimizer / energy / scoring）の高速化計画。実装はこのドキュメントを
チェックリストとして進める。関連: 全体計画＝[PLAN.md](PLAN.md) / M3 詳細＝[PLAN.M3.md](PLAN.M3.md)。

---

## Context（なぜ / 何を）

M3 密検証で**テスト実行が遅すぎる**ことが判明（decoy screen 36 分、単一再ドッキング 35–196 秒、
docs/results.202606172136.md）。原因は中核ドッキングの計算が重いこと。大規模ライブラリでは
クラスタリングの性質が変わるため M3 検証は現状で保留し、**通常ドッキング中核の高速化を先に行う**
（当初 M4 の前倒し）。次の 4 つを段階導入し、各々ベンチで効果を測る:

1. **スレッド並列（rayon）** — テスト壁時間に最も効く（独立 run / 分子は埋め込み並列）。
2. **ゼロアロケーション再利用バッファ** — 評価ごとの heap 確保を排除（SIMD の前提）。
3. **解析的勾配** — 有限差分（1 勾配 = 2×n_dof エネルギー評価）を置換し評価回数を本質削減。
4. **エネルギーカーネルの SIMD（f32 / `wide`）** — 支配ループ（受容体全ペア）をレーン並列化。

ゴール: 単一ノードの分子/秒を大幅向上。回帰で再ドッキング精度（RMSD<2Å）維持、スコアは
許容差内（f32 SIMD の微小ドリフトを除き不変）。

### 確定した決定
- **解析的勾配を含める**（num_grad を置換）。
- **SIMD は f32 を狙う**（AVX2 で 8 レーン ~2×）。回帰は許容差（<1e-2 kcal/mol, Vina ゲート相当）で assert。

### ビルド前提（確認済み）
- Rust 1.96.0 stable / edition 2024 / x86_64-linux。`std::simd` は nightly のため **SIMD は `wide`(0.7)** を使う
  （workspace deps に予約済み）。`rayon` も予約済み。`profile.release`=opt3/thin-LTO/cu1/panic=abort。

---

## 現状のホットパス

- `search::dock_constrained`: `n_runs`(既定60) × (`contact_start`(最大60サンプル) + `n_steps`(60) × `minimize_dofs`)。
- `minimize_*` → **`num_grad`: 2×n_dof 回のエネルギー評価**（有限差分）＝支配要因。
- 1 エネルギー評価 = `lig.scored_atoms`→`world_coords`（`ref_coords` を毎回 clone・`Vec` 確保）+
  `scorer.score`（`intermolecular_terms`: **O(n_lig×n_rec ≈ 20×2765) の全ペア**、各ペア `sqrt`＋分岐 5 項）+
  `box_penalty` + `intramolecular`。
- `minimize` は `DMatrix`/`DVector` を呼び出し毎に確保。並列性: `n_runs`・`screen` 分子は独立。

---

## 段階 1 — スレッド並列（rayon）
- workspace deps の `rayon` を有効化（isct-dock）。
- **screen**（`crates/isct-dock/src/screen.rs`）: クラスタ走査を `par_iter` 並列化（粗粒度、理想スケール）。
- **dock_constrained**（`crates/isct-dock/src/search.rs`）: 独立 `n_runs` を並列化。
  - **決定論**: 親 rng から各 run の seed を順に生成（`seed_k = base_rng.next_u64()`）→ 各 run は
    `ChaCha8Rng::seed_from_u64(seed_k)`。実行順非依存・再現可能。best は min エネルギー（タイは run index 昇順）で reduce。
  - 注: 現行は単一 rng を全 run に直列消費するため、**本変更で乱数列が変わり数値結果も変わる**（依然決定的）。
    回帰は「RMSD<2Å 維持・スコア同等」で再検証（ビット一致は要求しない）。
- 任意: `core_affinity` でスレッドを物理コアにピン留め（Linux、後追い可）。
- 期待: ~物理コア数×。

## 段階 2 — ゼロアロケーション再利用バッファ
- `crates/isct-dock/src/ligand.rs`: `world_coords`/`scored_atoms` を**呼び出し側バッファ書込** API へ
  （例 `world_coords_into(&self, conf, &mut Vec<Vec3>)`）。作業領域は `dock`/`minimize` がスレッドローカルに保持。
- `crates/isct-dock/src/optimizer.rs`: BFGS の `DMatrix`/`DVector` を再利用（低 DOF なら固定長バッファ可）。
- `crates/isct-dock/src/energy.rs`: 受容体を **SoA（rx[],ry[],rz[],type[]）** で保持（SIMD 前提）。

## 段階 3 — 解析的勾配
- `crates/isct-core/src/scoring.rs`: 各ペアの `dE/dr`（gauss1/gauss2/repulsion と hydrophobic/hbond の区分線形）を実装し、
  原子間方向へ射影して**原子座標勾配**を出す `intermolecular_value_and_grad(...)` を追加。
- `crates/isct-dock/src/ligand.rs`: 原子座標勾配 → DOF 勾配の連鎖則（並進=Σ力、回転=Σ r×F、torsion=回転軸まわりのモーメント）。
- `crates/isct-dock/src/optimizer.rs`: 目的関数を「値＋勾配を一括返す」形へ切替、`num_grad` を置換。BFGS 本体は不変。
- 検証: 解析的勾配 vs 数値微分の一致（単体テスト <1e-6）。スコア値は不変。

## 段階 4 — エネルギーカーネルの SIMD（f32, `wide`）
- `crates/isct-core/src/scoring.rs`: 受容体 SoA に対し**リガンド 1 原子 × 受容体 8 原子（f32x8）**を一括評価する
  カーネル。`dx,dy,dz→r2→r=sqrt→d=r-(Ri+Rj)` と 5 項をレーン並列、カットオフ・型条件はマスク。値＋勾配寄与の双方。
- 型→半径/フラグは受容体型ごとに前計算した SoA 配列をレーンへ展開。
- f32 精度: 回帰は許容差（<1e-2 kcal/mol）で assert。`vina --score_only` 一致ゲートも許容差内を確認。

---

## 主要ファイル
- `crates/isct-core/src/scoring.rs` — 解析的勾配、SIMD(f32, wide) カーネル、型→半径/フラグの SoA。
- `crates/isct-dock/src/energy.rs` — `Scorer` を SoA・SIMD・値＋勾配対応へ。
- `crates/isct-dock/src/ligand.rs` — `world_coords`/`scored_atoms` をバッファ書込・座標勾配→DOF 勾配。
- `crates/isct-dock/src/optimizer.rs` — 解析的勾配で `num_grad` 置換、バッファ再利用。
- `crates/isct-dock/src/search.rs` — `rayon` で `n_runs` 並列、決定的 per-run seed。
- `crates/isct-dock/src/screen.rs` — `rayon` で分子/クラスタ並列。
- `Cargo.toml`（workspace）/ `crates/{isct-core,isct-dock}/Cargo.toml` — `rayon`,`wide`,（任意）`core_affinity`,criterion(dev)。
- `crates/{isct-core,isct-dock}/benches/*.rs` — criterion マイクロベンチ新規。

## 再利用する既存
- `isct_core::scoring::{eval_pair, intermolecular_terms, Terms, W_*}`（勾配・SIMD のベース）。
- `optimizer::minimize_reduced` の BFGS 本体（勾配源だけ差し替え、to_full は維持）。
- `search::{dock_constrained, minimize_dofs, minimize_torsion_subset}`（並列化は外側のみ）。
- `bench/run_redock.py`（再ドッキング RMSD/時間の回帰）。

## スコープ外（今回はやらない）
- **グリッド補間**（Coarse/Fine grid）: 最大の単スレッド algorithmic win だが「全ペア SIMD」と入れ替えになるため
  **別計画（次段）**。今回は direct atom-atom を SIMD/勾配/並列で速くする。
- mmap ストリーミング I/O、分散（Work-Stealing）。

---

## 段階導入と検証（各段で実施）
1. **ベースライン計測**: criterion（energy eval / 1 分子 dock）と `bench/run_redock.py` の RMSD・時間
   （3ptb/1stp/1iep）を記録。
2. 段階 1→2→3→4 の各完了時:
   - `cargo test` 緑・`cargo clippy --all-targets` 警告ゼロ。
   - **精度回帰**: 3ptb/1stp/1iep 再ドッキング RMSD<2Å 維持。スコアは段階 1–2 不変、段階 3 不変、
     段階 4（f32）は許容差 <1e-2。`vina --score_only` 一致ゲートも許容差内。
   - **決定論**: 同一 seed で並列実行が再現一致。
   - criterion で段階ごとの speedup を記録。
3. 結果を `docs/results.YYYYMMDDHHMM.md` に各段の speedup として記録（新規・上書き禁止）。

## 検証（E2E 手順）
```sh
cargo test && cargo clippy --all-targets
cargo bench
./target/release/isct-dock dock bench/data/pdb/3ptb/receptor.pdbqt bench/data/pdb/3ptb/ligand.pdbqt \
  --config bench/data/pdb/3ptb/config.toml
python bench/run_redock.py
```
