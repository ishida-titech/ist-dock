//! グローバル探索（M2）。
//!
//! Vina 流の反復局所探索（Monte Carlo）: ランダム初期配置から「摂動 → 局所最適化 →
//! Metropolis 受理」を繰り返し、複数の独立 run で最良ポーズを探す。`rand_chacha` の
//! 固定シードで完全に決定論的。

use isct_core::geometry::Vec3;
use rand::{Rng, RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

use crate::energy::Scorer;
use crate::ligand::{Conformation, LigandModel};
use crate::optimizer::{minimize_free, rigid_and_torsions};

/// 探索パラメータ。
#[derive(Debug, Clone)]
pub struct SearchParams {
    pub center: Vec3,
    pub size: Vec3,
    pub seed: u64,
    /// 独立 run 数（探索の網羅性）。
    pub n_runs: usize,
    /// 各 run の MC ステップ数。
    pub n_steps: usize,
    /// 局所最適化の最大反復。
    pub local_iters: usize,
    /// Metropolis 温度（kcal/mol スケール）。
    pub temperature: f64,
    /// 初期配置を一様ランダムにする（接触棄却を省く）。グリッドで勾配場が滑らかなとき有効。
    pub uniform_start: bool,
}

impl Default for SearchParams {
    fn default() -> Self {
        SearchParams {
            center: Vec3::zeros(),
            size: Vec3::new(20.0, 20.0, 20.0),
            seed: 42,
            n_runs: 60,
            n_steps: 60,
            local_iters: 120,
            temperature: 1.0,
            uniform_start: false,
        }
    }
}

/// ドッキング結果。
pub struct DockResult {
    pub conf: Conformation,
    pub energy: f64,
}

/// リガンドを受容体ボックスへドッキングし、最良ポーズを返す（全 torsion 自由）。
pub fn dock(lig: &LigandModel, scorer: &Scorer, params: &SearchParams) -> DockResult {
    let free: Vec<usize> = (0..lig.n_torsions()).collect();
    dock_constrained(lig, scorer, params, &free)
}

/// Scaffold 固定ドッキング: `free_torsions` の torsion と剛体自由度のみを動かし、その他の
/// torsion（コアを動かすもの）は参照値 0 に固定したまま探索する。Phase 1 で seed のコアを
/// 剛体のまま位置・配向・R 基だけ大域探索するのに使う。
pub fn dock_constrained(
    lig: &LigandModel,
    scorer: &Scorer,
    params: &SearchParams,
    free_torsions: &[usize],
) -> DockResult {
    // 各 run の seed を親 rng から順に生成する（実行順非依存・再現可能）。これにより run を
    // rayon で並列化しても結果が決定的になる。
    let mut seed_rng = ChaCha8Rng::seed_from_u64(params.seed);
    let run_seeds: Vec<u64> = (0..params.n_runs).map(|_| seed_rng.next_u64()).collect();

    // 最適化する全 DOF（剛体6 ＋ R 基 torsion）。コアを動かす torsion は除外される。
    let free_dofs = rigid_and_torsions(free_torsions);

    // 値クロージャ（直線探索用）= 分子間 + ボックス罰則 + 分子内。lig/scorer は &（Sync）。
    let value = |c: &Conformation| {
        let atoms = lig.scored_atoms(c);
        scorer.score(&atoms) + lig.intramolecular(&atoms)
    };
    // 値＋解析的勾配（全 DOF）。1 回の前向き計算で力を集約し連鎖則で DOF 勾配へ。
    let value_grad = |c: &Conformation| -> (f64, Vec<f64>) {
        let (world_pos, world_axis, world_pivot) = lig.world_coords_frames(c);
        let atoms = lig.scored_atoms_at(&world_pos);
        let mut force = vec![isct_core::geometry::Vec3::zeros(); atoms.len()];
        let mut e = scorer.value_and_force(&atoms, &mut force);
        e += lig.intramolecular_value_grad(&atoms, &mut force);
        let grad = lig.dof_gradient(c, &world_pos, &world_axis, &world_pivot, &force);
        (e, grad)
    };

    // 独立 run を並列実行し、(run_index, best) を決定的に reduce（min エネルギー、タイは index 昇順）。
    run_seeds
        .par_iter()
        .enumerate()
        .map(|(idx, &seed)| {
            (
                idx,
                single_run(lig, scorer, params, free_torsions, &free_dofs, &value, &value_grad, seed),
            )
        })
        .reduce_with(|a, b| {
            if a.1.energy < b.1.energy || (a.1.energy == b.1.energy && a.0 <= b.0) {
                a
            } else {
                b
            }
        })
        .map(|(_, res)| res)
        .expect("at least one run")
}

/// 1 つの独立 run（接触開始 → ILS）を実行し、その run の最良ポーズを返す。
#[allow(clippy::too_many_arguments)]
fn single_run<V, VG>(
    lig: &LigandModel,
    scorer: &Scorer,
    params: &SearchParams,
    free_torsions: &[usize],
    free_dofs: &[usize],
    value: &V,
    value_grad: &VG,
    seed: u64,
) -> DockResult
where
    V: Fn(&Conformation) -> f64 + Sync,
    VG: Fn(&Conformation) -> (f64, Vec<f64>) + Sync,
{
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    // 初期配置（grid 時は一様、direct 時は接触棄却）。固定 torsion は 0 に保つ。
    let start = contact_start(
        lig,
        scorer,
        &mut rng,
        params.center,
        params.size,
        free_torsions,
        params.uniform_start,
    );
    let mut cur = minimize_free(start, value, value_grad, params.local_iters, free_dofs);
    let mut best = DockResult {
        conf: cur.conf.clone(),
        energy: cur.energy,
    };

    // 反復局所探索（ILS）: 毎ステップ 1 自由度だけ変異 → 局所最適化 → Metropolis 受理。
    for _ in 0..params.n_steps {
        let mut trial = cur.conf.clone();
        mutate(&mut trial, &mut rng, free_torsions);
        let res = minimize_free(trial, value, value_grad, params.local_iters, free_dofs);

        let accept = res.energy < cur.energy
            || rng.r#gen::<f64>() < (-(res.energy - cur.energy) / params.temperature).exp();
        if accept {
            cur = res;
            if cur.energy < best.energy {
                best = DockResult {
                    conf: cur.conf.clone(),
                    energy: cur.energy,
                };
            }
        }
    }
    best
}

/// 受容体と好ましく接触する初期配置を棄却サンプリングで得る。
/// 浮遊（エネルギー≈0, 勾配≈0 で局所最適化が無効）や激突を避け、productive な開始に絞る。
#[allow(clippy::too_many_arguments)]
fn contact_start<R: Rng>(
    lig: &LigandModel,
    scorer: &Scorer,
    rng: &mut R,
    center: Vec3,
    size: Vec3,
    free_torsions: &[usize],
    uniform: bool,
) -> Conformation {
    // 固定 torsion を 0（参照＝コア剛体）に戻した初期配置を作る。
    let sample = |rng: &mut R| -> Conformation {
        let mut c = lig.random_in_box(rng, center, size);
        lock_torsions(&mut c, free_torsions);
        c
    };
    // grid 時は一様ランダム（接触棄却なし）。勾配場が滑らかなため最適化が引き込む。
    if uniform {
        return sample(rng);
    }
    let energy = |c: &Conformation| scorer.intermolecular(&lig.scored_atoms(c));
    let mut best_c = sample(rng);
    let mut best_e = energy(&best_c);
    for _ in 0..60 {
        // 好ましい接触（負エネルギー）かつ激突なし（過大正値でない）なら採用。
        if best_e < -1.0 {
            break;
        }
        let c = sample(rng);
        let e = energy(&c);
        if e < best_e {
            best_e = e;
            best_c = c;
        }
    }
    best_c
}

/// `free_torsions` に含まれない torsion を 0（参照）に固定する。
fn lock_torsions(conf: &mut Conformation, free_torsions: &[usize]) {
    let free: std::collections::HashSet<usize> = free_torsions.iter().copied().collect();
    for (k, t) in conf.torsions.iter_mut().enumerate() {
        if !free.contains(&k) {
            *t = 0.0;
        }
    }
}

/// Vina 流の 1 自由度変異: 並進・配向・**自由** torsion のうち 1 つだけを動かす。
/// 良いポーズを壊さず滑らかに探索でき、BFGS で徐々に基へ降りられる。固定 torsion は動かさない。
fn mutate<R: Rng>(conf: &mut Conformation, rng: &mut R, free_torsions: &[usize]) {
    use std::f64::consts::PI;
    // entity: 0=並進, 1=配向, 2..=自由 torsion
    let k = rng.gen_range(0..(2 + free_torsions.len()));
    let mut d = vec![0.0; conf.dim()];
    match k {
        0 => {
            // 並進（各成分 ±2 Å）
            for v in d.iter_mut().take(3) {
                *v = rng.gen_range(-2.0..2.0);
            }
        }
        1 => {
            // 配向（回転ベクトル各成分 ±1.0 rad）
            for v in d.iter_mut().take(6).skip(3) {
                *v = rng.gen_range(-1.0..1.0);
            }
        }
        _ => {
            // 1 つの自由 torsion を再サンプリング（±π）
            let t = free_torsions[k - 2];
            d[6 + t] = rng.gen_range(-PI..PI);
        }
    }
    conf.apply_delta(&d);
}
