//! 局所最適化（M2）。
//!
//! 目的関数（リガンドポーズのエネルギー）に対し、**有限差分勾配 ＋ BFGS（逆ヘシアン近似）
//! ＋ バックトラッキング直線探索**で局所最小化する。状態は Conformation の局所増分
//! （並進3＋回転3＋torsion n）で表す。
//!
//! 注: 計画では解析的勾配＋L-BFGS だが、まず動作・検証を優先して有限差分とした。
//! 解析的勾配は後続の最適化（性能向上）で導入する。

use nalgebra::{DMatrix, DVector};

use crate::ligand::Conformation;

/// 有限差分の刻み幅（Å / rad）。
const GRAD_EPS: f64 = 1e-4;
/// Armijo 条件の係数。
const ARMIJO_C1: f64 = 1e-4;

/// 局所最適化の結果。
pub struct LocalResult {
    pub conf: Conformation,
    pub energy: f64,
}

/// 目的関数 `f` を起点 `start` から BFGS で局所最小化する（全自由度: 並進3＋回転3＋torsion n）。
pub fn minimize<F>(start: Conformation, f: &F, max_iter: usize) -> LocalResult
where
    F: Fn(&Conformation) -> f64,
{
    let dim = start.dim();
    // 縮約ベクトル = 全自由度増分そのもの。
    minimize_reduced(start, f, max_iter, dim, |r| r.iter().copied().collect())
}

/// 全 torsion を自由度として局所最小化する（並進・回転＝コアはロック）。
pub fn minimize_torsions<F>(start: Conformation, f: &F, max_iter: usize) -> LocalResult
where
    F: Fn(&Conformation) -> f64,
{
    let free: Vec<usize> = (0..start.torsions.len()).collect();
    minimize_torsion_subset(start, f, max_iter, &free)
}

/// 指定した torsion 添字の部分集合のみを自由度として局所最小化する（その他は固定、並進・
/// 回転もロック）。Phase 3 で「R 基 torsion のみ」を最適化するのに使う。`free` は torsion
/// インデックス（0-based）の集合。
pub fn minimize_torsion_subset<F>(
    start: Conformation,
    f: &F,
    max_iter: usize,
    free: &[usize],
) -> LocalResult
where
    F: Fn(&Conformation) -> f64,
{
    let nt = start.torsions.len();
    let free: Vec<usize> = free.to_vec();
    let n = free.len();
    minimize_reduced(start, f, max_iter, n, move |r| {
        let mut full = vec![0.0; 6 + nt];
        for (j, &k) in free.iter().enumerate() {
            full[6 + k] = r[j];
        }
        full
    })
}

/// 並進・回転＋指定 torsion 部分集合のみを自由度として局所最小化する。コアを動かす torsion
/// をロックしたまま剛体配置と R 基だけを最適化する（Phase 1 の Scaffold 固定 seed 探索）用。
pub fn minimize_dofs<F>(
    start: Conformation,
    f: &F,
    max_iter: usize,
    free_torsions: &[usize],
) -> LocalResult
where
    F: Fn(&Conformation) -> f64,
{
    let nt = start.torsions.len();
    let free: Vec<usize> = free_torsions.to_vec();
    let n = 6 + free.len();
    minimize_reduced(start, f, max_iter, n, move |r| {
        let mut full = vec![0.0; 6 + nt];
        full[..6].copy_from_slice(&r.as_slice()[..6]);
        for (j, &k) in free.iter().enumerate() {
            full[6 + k] = r[6 + j];
        }
        full
    })
}

/// 縮約自由度空間での BFGS 最小化。`to_full` は縮約増分ベクトルを Conformation の全増分
/// （長さ dim）へ写す（`minimize`=恒等, `minimize_torsions`=先頭6つを0埋め）。
fn minimize_reduced<F, M>(
    start: Conformation,
    f: &F,
    max_iter: usize,
    n: usize,
    to_full: M,
) -> LocalResult
where
    F: Fn(&Conformation) -> f64,
    M: Fn(&DVector<f64>) -> Vec<f64>,
{
    if n == 0 {
        let energy = f(&start);
        return LocalResult {
            conf: start,
            energy,
        };
    }

    let mut conf = start;
    let mut energy = f(&conf);
    let mut grad = num_grad(&conf, f, n, &to_full);
    let mut hinv = DMatrix::<f64>::identity(n, n);

    for _ in 0..max_iter {
        let gnorm = grad.norm();
        if gnorm < 1e-4 {
            break;
        }
        // 探索方向 p = -Hinv * grad
        let mut p = -(&hinv * &grad);
        // 降下方向でなければ最急降下にフォールバック。
        if p.dot(&grad) >= 0.0 {
            p = -grad.clone();
            hinv = DMatrix::identity(n, n);
        }

        // バックトラッキング直線探索（Armijo）。
        let g_dot_p = grad.dot(&p);
        let mut step = 1.0;
        let mut accepted = false;
        let mut new_conf = conf.clone();
        let mut new_energy = energy;
        for _ in 0..40 {
            let s = step * &p;
            let mut trial = conf.clone();
            trial.apply_delta(&to_full(&s));
            let e = f(&trial);
            if e <= energy + ARMIJO_C1 * step * g_dot_p {
                new_conf = trial;
                new_energy = e;
                accepted = true;
                break;
            }
            step *= 0.5;
        }
        if !accepted {
            break;
        }

        let s = step * &p;
        let new_grad = num_grad(&new_conf, f, n, &to_full);
        let y = &new_grad - &grad;
        let sy = s.dot(&y);
        if sy > 1e-10 {
            bfgs_inverse_update(&mut hinv, &s, &y, sy);
        }
        conf = new_conf;
        energy = new_energy;
        grad = new_grad;
    }

    LocalResult { conf, energy }
}

/// 縮約自由度空間での中心差分勾配。`n`＝縮約次元、`to_full`＝縮約増分→全増分の写像。
fn num_grad<F, M>(conf: &Conformation, f: &F, n: usize, to_full: &M) -> DVector<f64>
where
    F: Fn(&Conformation) -> f64,
    M: Fn(&DVector<f64>) -> Vec<f64>,
{
    let mut g = DVector::zeros(n);
    let mut d = DVector::zeros(n);
    for k in 0..n {
        d[k] = GRAD_EPS;
        let mut cp = conf.clone();
        cp.apply_delta(&to_full(&d));
        let fp = f(&cp);
        d[k] = -GRAD_EPS;
        let mut cm = conf.clone();
        cm.apply_delta(&to_full(&d));
        let fm = f(&cm);
        d[k] = 0.0;
        g[k] = (fp - fm) / (2.0 * GRAD_EPS);
    }
    g
}

/// BFGS 逆ヘシアン更新: Hinv += ... （Sherman-Morrison 形式）。
fn bfgs_inverse_update(hinv: &mut DMatrix<f64>, s: &DVector<f64>, y: &DVector<f64>, sy: f64) {
    let rho = 1.0 / sy;
    let n = s.len();
    let i = DMatrix::<f64>::identity(n, n);
    let sy_mat = s * y.transpose() * rho; // s y^T / sy
    let left = &i - &sy_mat;
    let right = &i - &(y * s.transpose() * rho);
    let ss = s * s.transpose() * rho;
    *hinv = &left * &*hinv * &right + ss;
}

#[cfg(test)]
mod tests {
    use super::*;
    use isct_core::geometry::Vec3;

    // 並進のみの二次関数で BFGS が最小（原点）へ収束するか。
    #[test]
    fn minimizes_quadratic_translation() {
        let start = Conformation {
            position: Vec3::new(3.0, -2.0, 1.0),
            orientation: nalgebra::UnitQuaternion::identity(),
            torsions: vec![],
        };
        let f = |c: &Conformation| c.position.norm_squared();
        let res = minimize(start, &f, 50);
        assert!(res.energy < 1e-6, "energy {}", res.energy);
        assert!(res.conf.position.norm() < 1e-3);
    }

    // torsion 限定最適化は torsion のみ動かし、並進・回転（コア）は固定したまま。
    #[test]
    fn minimize_torsions_locks_rigid_dofs() {
        let start = Conformation {
            position: Vec3::new(3.0, -2.0, 1.0),
            orientation: nalgebra::UnitQuaternion::identity(),
            torsions: vec![0.0],
        };
        // torsion[0] の最小は 0.5。並進項も持たせて「動かさない」ことを確認する。
        let f = |c: &Conformation| (c.torsions[0] - 0.5).powi(2) + c.position.norm_squared();
        let res = minimize_torsions(start.clone(), &f, 50);
        assert!((res.conf.torsions[0] - 0.5).abs() < 1e-3, "torsion {}", res.conf.torsions[0]);
        // 並進は初期値のまま（ロック）。
        assert!((res.conf.position - start.position).norm() < 1e-12);
    }
}
