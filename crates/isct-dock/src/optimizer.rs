//! 局所最適化（M2 / 高速化）。
//!
//! 目的関数（リガンドポーズのエネルギー）に対し **BFGS（逆ヘシアン近似）＋ バックトラッキング
//! 直線探索**で局所最小化する。状態は Conformation の局所増分（並進3＋回転3＋torsion n）。
//!
//! 勾配は **解析的勾配**（`value_grad` クロージャ）を使う。解析的勾配を持たない呼び出し側
//! （Phase 3 の Δエネルギー等）向けに有限差分版 `minimize_free_fd` も用意する。
//! `free` は最適化する **全 DOF インデックス**（並進=0..3, 回転=3..6, torsion k=6+k）の部分集合。

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

/// 全 DOF（並進3＋回転3＋torsion n）のインデックス。
pub fn all_dofs(dim: usize) -> Vec<usize> {
    (0..dim).collect()
}

/// 剛体 6 自由度 ＋ 指定 R 基 torsion の DOF インデックス（Phase 1 の Scaffold 固定探索用）。
pub fn rigid_and_torsions(free_torsions: &[usize]) -> Vec<usize> {
    let mut v: Vec<usize> = (0..6).collect();
    v.extend(free_torsions.iter().map(|&t| 6 + t));
    v
}

/// 指定 torsion のみの DOF インデックス（Phase 3 の R 基最適化用）。
pub fn torsions_only(free_torsions: &[usize]) -> Vec<usize> {
    free_torsions.iter().map(|&t| 6 + t).collect()
}

/// 値クロージャ `value` と 値＋全 DOF 勾配クロージャ `value_grad` を用いて、`free` で指定した
/// DOF 部分空間のみ BFGS 局所最小化する。`value_grad` は全 DOF（長さ dim）の勾配を返し、
/// 本関数が `free` に射影する。
///
/// 不変条件（呼び出し側のバッファ再利用に利用）: `free` が非空のとき、**最後に評価する `value_grad` は
/// 返却する `conf` 上で呼ばれる**（収束 break・直線探索失敗 break・通常更新のいずれでも成立。直線探索は
/// `value` のみ呼ぶ）。これにより `value_grad` 側スクラッチは返却 conf の中間量を保持する。
pub fn minimize_free<V, VG>(
    start: Conformation,
    value: &V,
    value_grad: &VG,
    max_iter: usize,
    free: &[usize],
) -> LocalResult
where
    V: Fn(&Conformation) -> f64,
    VG: Fn(&Conformation) -> (f64, Vec<f64>),
{
    let n = free.len();
    if n == 0 {
        let energy = value(&start);
        return LocalResult {
            conf: start,
            energy,
        };
    }

    let dim = start.dim();
    let project = |gfull: &[f64]| -> DVector<f64> {
        DVector::from_iterator(n, free.iter().map(|&i| gfull[i]))
    };
    // `to_full` の毎試行アロク（`vec![0.0; dim]`, 直線探索で最大 40×/反復）を避ける再利用バッファ。
    let mut delta_full = vec![0.0; dim];

    let mut conf = start;
    let (mut energy, gfull) = value_grad(&conf);
    let mut grad = project(&gfull);
    let mut hinv = DMatrix::<f64>::identity(n, n);
    // BFGS 逆ヘシアン更新の作業行列（反復毎の n×n `DMatrix` 5–7 枚確保を回避）。
    let mut ws = BfgsWorkspace::new(n);

    for _ in 0..max_iter {
        if grad.norm() < 1e-4 {
            break;
        }
        // 探索方向 p = -Hinv * grad。降下方向でなければ最急降下へフォールバック。
        let mut p = -(&hinv * &grad);
        if p.dot(&grad) >= 0.0 {
            p = -grad.clone();
            set_identity(&mut hinv);
        }

        // バックトラッキング直線探索（Armijo）。値のみ評価。
        let g_dot_p = grad.dot(&p);
        let mut step = 1.0;
        let mut accepted = false;
        let mut new_conf = conf.clone();
        for _ in 0..40 {
            let s = step * &p;
            let mut trial = conf.clone();
            fill_full(&mut delta_full, &s, free);
            trial.apply_delta(&delta_full);
            let e = value(&trial);
            if e <= energy + ARMIJO_C1 * step * g_dot_p {
                new_conf = trial;
                accepted = true;
                break;
            }
            step *= 0.5;
        }
        if !accepted {
            break;
        }

        let s = step * &p;
        let (new_energy, new_gfull) = value_grad(&new_conf);
        let new_grad = project(&new_gfull);
        let y = &new_grad - &grad;
        let sy = s.dot(&y);
        if sy > 1e-10 {
            bfgs_inverse_update(&mut hinv, &s, &y, sy, &mut ws);
        }
        conf = new_conf;
        energy = new_energy;
        grad = new_grad;
    }

    LocalResult { conf, energy }
}

/// `free` DOF 部分空間のベクトル `r` を全 DOF 長（非 free は 0）へ書き戻す（再利用バッファ版 `to_full`）。
#[inline]
fn fill_full(full: &mut [f64], r: &DVector<f64>, free: &[usize]) {
    full.iter_mut().for_each(|v| *v = 0.0);
    for (j, &idx) in free.iter().enumerate() {
        full[idx] = r[j];
    }
}

/// 既存 n×n 行列を単位行列に上書き（`DMatrix::identity` の再確保を避ける）。
#[inline]
fn set_identity(m: &mut DMatrix<f64>) {
    m.fill(0.0);
    m.fill_diagonal(1.0);
}

/// BFGS 逆ヘシアン更新の作業行列。反復間で再利用しアロクを避ける。
struct BfgsWorkspace {
    left: DMatrix<f64>,
    right: DMatrix<f64>,
    tmp: DMatrix<f64>,
    out: DMatrix<f64>,
}

impl BfgsWorkspace {
    fn new(n: usize) -> Self {
        BfgsWorkspace {
            left: DMatrix::zeros(n, n),
            right: DMatrix::zeros(n, n),
            tmp: DMatrix::zeros(n, n),
            out: DMatrix::zeros(n, n),
        }
    }
}

/// 解析的勾配を持たない呼び出し側向け: 値クロージャだけから有限差分勾配を作って最小化する。
pub fn minimize_free_fd<V>(
    start: Conformation,
    value: &V,
    max_iter: usize,
    free: &[usize],
) -> LocalResult
where
    V: Fn(&Conformation) -> f64,
{
    let value_grad = |c: &Conformation| (value(c), fd_grad(value, c, free));
    minimize_free(start, value, &value_grad, max_iter, free)
}

/// `free` の DOF のみ中心差分で勾配を求める（全 DOF 長、非 free は 0）。
fn fd_grad<V: Fn(&Conformation) -> f64>(value: &V, conf: &Conformation, free: &[usize]) -> Vec<f64> {
    let dim = conf.dim();
    let mut g = vec![0.0; dim];
    let mut d = vec![0.0; dim];
    for &idx in free {
        d[idx] = GRAD_EPS;
        let mut cp = conf.clone();
        cp.apply_delta(&d);
        let fp = value(&cp);
        d[idx] = -GRAD_EPS;
        let mut cm = conf.clone();
        cm.apply_delta(&d);
        let fm = value(&cm);
        d[idx] = 0.0;
        g[idx] = (fp - fm) / (2.0 * GRAD_EPS);
    }
    g
}

/// BFGS 逆ヘシアン更新: Hinv ← (I - ρ s yᵀ) Hinv (I - ρ y sᵀ) + ρ s sᵀ（Sherman-Morrison 形式）。
///
/// アロク回避のため作業行列 `ws` を再利用する。**数値はビット一致を維持する**:
/// 外積（`s yᵀ`, `y sᵀ`, `s sᵀ`）は要素毎の単一積で手計算（累積無し＝丸め順に依存しない）、
/// 行列積は nalgebra の `mul_to`（`*` と同一カーネル）を使い、結合順・加算順を旧実装と厳密一致させる。
fn bfgs_inverse_update(
    hinv: &mut DMatrix<f64>,
    s: &DVector<f64>,
    y: &DVector<f64>,
    sy: f64,
    ws: &mut BfgsWorkspace,
) {
    let rho = 1.0 / sy;
    let n = s.len();
    // left = I - (s yᵀ)ρ, right = I - (y sᵀ)ρ（外積は単一積、旧: `(s*y^T)*rho` と同順）。
    for i in 0..n {
        for j in 0..n {
            let id = if i == j { 1.0 } else { 0.0 };
            ws.left[(i, j)] = id - (s[i] * y[j]) * rho;
            ws.right[(i, j)] = id - (y[i] * s[j]) * rho;
        }
    }
    // tmp = left * hinv;  out = tmp * right（旧: ((left*hinv)*right) と同結合順）。
    ws.left.mul_to(hinv, &mut ws.tmp);
    ws.tmp.mul_to(&ws.right, &mut ws.out);
    // out += (s sᵀ)ρ（旧: `+ ss` の要素毎加算と同順）。
    for i in 0..n {
        for j in 0..n {
            ws.out[(i, j)] += (s[i] * s[j]) * rho;
        }
    }
    hinv.copy_from(&ws.out);
}

#[cfg(test)]
mod tests {
    use super::*;
    use isct_core::geometry::Vec3;

    // 並進のみの二次関数で BFGS が最小（原点）へ収束するか（有限差分版）。
    #[test]
    fn minimizes_quadratic_translation() {
        let start = Conformation {
            position: Vec3::new(3.0, -2.0, 1.0),
            orientation: nalgebra::UnitQuaternion::identity(),
            torsions: vec![],
        };
        let f = |c: &Conformation| c.position.norm_squared();
        let res = minimize_free_fd(start.clone(), &f, 50, &all_dofs(start.dim()));
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
        let f = |c: &Conformation| (c.torsions[0] - 0.5).powi(2) + c.position.norm_squared();
        let res = minimize_free_fd(start.clone(), &f, 50, &torsions_only(&[0]));
        assert!((res.conf.torsions[0] - 0.5).abs() < 1e-3, "torsion {}", res.conf.torsions[0]);
        assert!((res.conf.position - start.position).norm() < 1e-12);
    }
}
