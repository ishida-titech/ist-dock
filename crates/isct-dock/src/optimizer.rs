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

/// 目的関数 `f` を起点 `start` から BFGS で局所最小化する。
pub fn minimize<F>(start: Conformation, f: &F, max_iter: usize) -> LocalResult
where
    F: Fn(&Conformation) -> f64,
{
    let n = start.dim();
    let mut conf = start;
    let mut energy = f(&conf);
    let mut grad = num_grad(&conf, f, energy);
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
            trial.apply_delta(s.as_slice());
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
        let new_grad = num_grad(&new_conf, f, new_energy);
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

/// 中心差分による勾配（`f0` は現在のエネルギー、使わないが将来の前方差分用に受ける）。
fn num_grad<F>(conf: &Conformation, f: &F, _f0: f64) -> DVector<f64>
where
    F: Fn(&Conformation) -> f64,
{
    let n = conf.dim();
    let mut g = DVector::zeros(n);
    let mut d = vec![0.0; n];
    for k in 0..n {
        d[k] = GRAD_EPS;
        let mut cp = conf.clone();
        cp.apply_delta(&d);
        let fp = f(&cp);
        d[k] = -GRAD_EPS;
        let mut cm = conf.clone();
        cm.apply_delta(&d);
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
}
