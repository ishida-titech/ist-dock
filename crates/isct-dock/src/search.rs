//! グローバル探索（M2）。
//!
//! Vina 流の反復局所探索（Monte Carlo）: ランダム初期配置から「摂動 → 局所最適化 →
//! Metropolis 受理」を繰り返し、複数の独立 run で最良ポーズを探す。`rand_chacha` の
//! 固定シードで完全に決定論的。

use isct_core::geometry::Vec3;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::energy::Scorer;
use crate::ligand::{Conformation, LigandModel};
use crate::optimizer::minimize;

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
        }
    }
}

/// ドッキング結果。
pub struct DockResult {
    pub conf: Conformation,
    pub energy: f64,
}

/// リガンドを受容体ボックスへドッキングし、最良ポーズを返す。
pub fn dock(lig: &LigandModel, scorer: &Scorer, params: &SearchParams) -> DockResult {
    let mut rng = ChaCha8Rng::seed_from_u64(params.seed);
    // 目的関数 = 分子間エネルギー + ボックス外ペナルティ + 分子内エネルギー。
    let objective = |c: &Conformation| {
        let atoms = lig.scored_atoms(c);
        scorer.score(&atoms) + lig.intramolecular(&atoms)
    };

    let mut best: Option<DockResult> = None;

    for _ in 0..params.n_runs {
        // 接触する初期配置から開始（浮遊・激突を避ける）。
        let start = contact_start(lig, scorer, &mut rng, params.center, params.size);
        let mut cur = minimize(start, &objective, params.local_iters);
        consider(&mut best, &cur.conf, cur.energy);

        // 反復局所探索（ILS）: 毎ステップ 1 自由度だけ変異 → 局所最適化 → Metropolis 受理。
        for _ in 0..params.n_steps {
            let mut trial = cur.conf.clone();
            mutate(&mut trial, &mut rng);
            let res = minimize(trial, &objective, params.local_iters);

            let accept = res.energy < cur.energy
                || rng.r#gen::<f64>() < (-(res.energy - cur.energy) / params.temperature).exp();
            if accept {
                cur = res;
                consider(&mut best, &cur.conf, cur.energy);
            }
        }
    }

    best.expect("at least one run")
}

/// 受容体と好ましく接触する初期配置を棄却サンプリングで得る。
/// 浮遊（エネルギー≈0, 勾配≈0 で局所最適化が無効）や激突を避け、productive な開始に絞る。
fn contact_start<R: Rng>(
    lig: &LigandModel,
    scorer: &Scorer,
    rng: &mut R,
    center: Vec3,
    size: Vec3,
) -> Conformation {
    let energy = |c: &Conformation| scorer.intermolecular(&lig.scored_atoms(c));
    let mut best_c = lig.random_in_box(rng, center, size);
    let mut best_e = energy(&best_c);
    for _ in 0..60 {
        // 好ましい接触（負エネルギー）かつ激突なし（過大正値でない）なら採用。
        if best_e < -1.0 {
            break;
        }
        let c = lig.random_in_box(rng, center, size);
        let e = energy(&c);
        if e < best_e {
            best_e = e;
            best_c = c;
        }
    }
    best_c
}

fn consider(best: &mut Option<DockResult>, conf: &Conformation, energy: f64) {
    if best.as_ref().is_none_or(|b| energy < b.energy) {
        *best = Some(DockResult {
            conf: conf.clone(),
            energy,
        });
    }
}

/// Vina 流の 1 自由度変異: 並進・配向・各 torsion のうち 1 つだけを動かす。
/// 良いポーズを壊さず滑らかに探索でき、BFGS で徐々に基へ降りられる。
fn mutate<R: Rng>(conf: &mut Conformation, rng: &mut R) {
    use std::f64::consts::PI;
    let n_tors = conf.torsions.len();
    // entity: 0=並進, 1=配向, 2..=各 torsion
    let k = rng.gen_range(0..(2 + n_tors));
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
            // 1 つの torsion を再サンプリング（±π）
            d[6 + (k - 2)] = rng.gen_range(-PI..PI);
        }
    }
    conf.apply_delta(&d);
}
