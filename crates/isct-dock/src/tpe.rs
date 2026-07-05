//! TPE（Tree-structured Parzen Estimator）探索の中核（opt-in 探索モード）。
//!
//! TPEvina-2.0（PSOVina 派生, CBBio）の TPE を ISCT-Dock へ移植したもの。探索空間の各次元を
//! **独立な Gaussian Parzen 混合**でモデル化し、良観測（lower）から候補をサンプリングして
//! 期待改善（g(x)/l(x) の比）で 1 点を選ぶ。局所最適化には ISCT の BFGS（`minimize_free`）を再利用する。
//!
//! ISCT 版の意図的な差分（原実装からの逸脱, いずれも移植上必須）:
//! - **決定論**: 原実装の TPE サンプリングは `std::random_device` シード（run 毎に非再現。作者も FIXME 明記）。
//!   本移植は探索全体を seeded `ChaCha8Rng` で駆動し**同一 seed でビット一致**（他モードと同じ ISCT 方針）。
//! - **数値**: boost/std の分布・浮動小数演算差、正規乱数は Box–Muller で自前生成のため、原実装とは
//!   ビット一致しない（=忠実な再実装であり byte clone ではない）。
//! - **既定パラメータ**: TPEvina 既定（`sample_num=7`, `num_lower_groups=10`, `num_upper_groups=10`,
//!   `kernel_density_frequency=4`, `start_search_num=20`）を踏襲。この既定では greater 重み
//!   `(upper_groups/2 - iter)/upper_groups` が探索初期に 0 へ収束するため、実効挙動は
//!   **良観測近傍の Parzen サンプリング（EDA 的 exploitation）**になる（原実装通り）。
//!
//! モデル空間ベクトル（長さ `6 + n_free_torsions`）: `[pos(3, 絶対), rot(3, 参照配座からの scaled-axis 差),
//! free_torsion(n, 参照からの差)]`。位置は探索ボックス角、回転・torsion は [-π, π] を範囲とする。

use rand::Rng;

use isct_core::geometry::Vec3;
use nalgebra::UnitQuaternion;

use crate::ligand::Conformation;

/// TPE ハイパーパラメータ（既定は TPEvina 準拠）。
#[derive(Debug, Clone, Copy)]
pub struct TpeParams {
    /// 1 提案あたり lower 混合から引く候補数（n_ei_candidates, TPEvina `sample_num`）。
    pub ei_candidates: usize,
    /// ウォームアップ（一様ランダム）割合: warmup = n_steps / startup_frac（TPEvina `start_search_num`）。
    pub startup_frac: usize,
    /// lower（良）集合サイズ上限（TPEvina `num_lower_groups`）。実サイズ = clamp(0.25√N, 2, これ)。
    pub lower_cap: usize,
    /// g(x) に使う greater（悪）観測数の上限、かつ greater 重み減衰の分母（TPEvina `num_upper_groups`）。
    pub upper_groups: usize,
    /// TPE フェーズでのモデル再構築回数（TPEvina `kernel_density_frequency`＝1 モデルの使い回し係数）。
    pub density_reuse: usize,
    /// 1 提案あたりの局所最適化（BFGS）反復数。0 なら呼び出し側の既定（`chain_iters`）を使う。
    pub local_iters: usize,
}

impl Default for TpeParams {
    fn default() -> Self {
        TpeParams {
            ei_candidates: 7,
            startup_frac: 20,
            lower_cap: 10,
            upper_groups: 10,
            density_reuse: 4,
            local_iters: 0,
        }
    }
}

/// 観測点（モデル空間ベクトル＋スコア）。スコア昇順が良い。
#[derive(Debug, Clone)]
pub struct Obs {
    pub v: Vec<f64>,
    pub score: f64,
}

/// 角度を [-π, π) へ正規化する（TPEvina `normalized_angle` 相当）。
#[inline]
pub fn normalize_angle(a: f64) -> f64 {
    use std::f64::consts::PI;
    let two_pi = 2.0 * PI;
    let mut x = (a + PI).rem_euclid(two_pi) - PI;
    if x == PI {
        x = -PI;
    }
    x
}

/// 各次元の探索範囲 [l_limit, g_limit]。位置=ボックス角、回転・torsion=[-π, π]。
/// `n_free_torsions` は free torsion 数（モデル次元 = 6 + n_free_torsions）。
pub fn bounds(center: Vec3, size: Vec3, n_free_torsions: usize) -> (Vec<f64>, Vec<f64>) {
    use std::f64::consts::PI;
    let half = size * 0.5;
    let mut l = vec![
        center.x - half.x,
        center.y - half.y,
        center.z - half.z,
    ];
    let mut g = vec![
        center.x + half.x,
        center.y + half.y,
        center.z + half.z,
    ];
    for _ in 0..(3 + n_free_torsions) {
        l.push(-PI);
        g.push(PI);
    }
    (l, g)
}

/// モデル空間ベクトル `v` を参照配座からの差として適用し `Conformation` を作る。
/// `v = [pos(3, 絶対), rot(3, 参照 orientation からの scaled-axis 差), free_torsion 差(n)]`。
pub fn conf_from_vec(reference: &Conformation, free_torsions: &[usize], v: &[f64]) -> Conformation {
    let mut c = reference.clone();
    c.position = Vec3::new(v[0], v[1], v[2]);
    let rot = Vec3::new(v[3], v[4], v[5]);
    c.orientation = UnitQuaternion::from_scaled_axis(rot) * reference.orientation;
    for (k, &t) in free_torsions.iter().enumerate() {
        c.torsions[t] = normalize_angle(reference.torsions[t] + v[6 + k]);
    }
    c
}

/// `Conformation`（局所最適化後）を参照からの差＝モデル空間ベクトルへ写す（観測の記録用）。
pub fn vec_from_conf(reference: &Conformation, free_torsions: &[usize], c: &Conformation) -> Vec<f64> {
    let mut v = Vec::with_capacity(6 + free_torsions.len());
    v.push(c.position.x);
    v.push(c.position.y);
    v.push(c.position.z);
    let diff = (c.orientation * reference.orientation.inverse()).scaled_axis();
    v.push(diff.x);
    v.push(diff.y);
    v.push(diff.z);
    for &t in free_torsions {
        v.push(normalize_angle(c.torsions[t] - reference.torsions[t]));
    }
    v
}

/// ウォームアップ提案: 各次元を範囲内で一様サンプル。
pub fn warmup_sample<R: Rng>(rng: &mut R, l_limit: &[f64], g_limit: &[f64]) -> Vec<f64> {
    l_limit
        .iter()
        .zip(g_limit)
        .map(|(&l, &g)| rng.gen_range(l..g))
        .collect()
}

/// 平均 `mean`・標準偏差 `sigma` の正規乱数（Box–Muller, 外部依存を避け決定論）。
fn sample_normal<R: Rng>(rng: &mut R, mean: f64, sigma: f64) -> f64 {
    use std::f64::consts::PI;
    // u1 ∈ (0,1] で ln(0) を回避。
    let u1: f64 = 1.0 - rng.r#gen::<f64>();
    let u2: f64 = rng.r#gen::<f64>();
    let z = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
    mean + sigma * z
}

/// 1 次元分の Parzen カーネル群（値と帯域）。
struct Dim {
    /// lower カーネル（value, sigma）。
    lower: Vec<(f64, f64)>,
    /// greater カーネル（value, sigma）。g(x) 用に上限 `upper_groups` 個へ切詰め済み。
    greater: Vec<(f64, f64)>,
    /// greater 全体に適用する重み（TPEvina: 反復依存でスカラ）。
    greater_weight: f64,
}

/// 全次元の Parzen モデル。`build` で観測から構築し `sample` で候補を引く。
pub struct TpeModel {
    dims: Vec<Dim>,
}

/// 近傍ギャップ規則の帯域 sigma（TPEvina）: 端はボックス境界まで、内側は左右ギャップの max。
/// 下限 = range / (1 + n_div)、上限 = range。
#[inline]
fn kernel_sigma(sorted_vals: &[f64], k: usize, lo: f64, hi: f64, n_div: usize) -> f64 {
    let range = hi - lo;
    let n = sorted_vals.len();
    let raw = if k == 0 {
        sorted_vals[k] - lo
    } else if k == n - 1 {
        hi - sorted_vals[k]
    } else {
        (sorted_vals[k + 1] - sorted_vals[k]).max(sorted_vals[k] - sorted_vals[k - 1])
    };
    let min_sigma = range / (1 + n_div) as f64;
    raw.max(min_sigma).min(range)
}

/// 1 次元の Gaussian 混合密度を `param` で評価（equal-weight または greater_weight 付き）。
#[inline]
fn mixture_density(kernels: &[(f64, f64)], param: f64, weight: f64) -> f64 {
    use std::f64::consts::PI;
    let norm = 1.0 / (2.0 * PI).sqrt();
    kernels
        .iter()
        .map(|&(val, sigma)| {
            let z = (param - val) / sigma;
            weight * norm / sigma * (-0.5 * z * z).exp()
        })
        .sum()
}

impl TpeModel {
    /// 観測 `obs` からモデルを構築する。`iter` は greater 重み減衰に使う提案カウンタ。
    /// 観測が 2 点未満のときは `None`（呼び出し側はウォームアップにフォールバック）。
    pub fn build(
        obs: &[Obs],
        p: &TpeParams,
        iter: usize,
        l_limit: &[f64],
        g_limit: &[f64],
    ) -> Option<TpeModel> {
        let n = obs.len();
        if n < 2 {
            return None;
        }
        // スコア昇順に並べ、lower / greater へ分割。
        let mut idx: Vec<usize> = (0..n).collect();
        idx.sort_by(|&a, &b| {
            obs[a]
                .score
                .partial_cmp(&obs[b].score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let n_lower = (((n as f64).sqrt() * 0.25) as usize).clamp(2, p.lower_cap).min(n);
        let lower_idx = &idx[..n_lower];
        let greater_idx = &idx[n_lower..];

        // greater 重み（TPEvina: min(1,max(0,(upper/2 - iter)/upper))）。iter が進むと 0 へ。
        let gw = (((p.upper_groups / 2) as f64 - iter as f64) / p.upper_groups as f64)
            .clamp(0.0, 1.0);

        let dim = l_limit.len();
        let mut dims = Vec::with_capacity(dim);
        for j in 0..dim {
            let (lo, hi) = (l_limit[j], g_limit[j]);
            // lower カーネル: 値でソート→sigma。
            let mut lv: Vec<f64> = lower_idx.iter().map(|&i| obs[i].v[j]).collect();
            lv.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let lower: Vec<(f64, f64)> = (0..lv.len())
                .map(|k| (lv[k], kernel_sigma(&lv, k, lo, hi, lv.len())))
                .collect();
            // greater カーネル: 値でソート→sigma、g(x) 用に upper_groups 個へ切詰め。
            let mut gv: Vec<f64> = greater_idx.iter().map(|&i| obs[i].v[j]).collect();
            gv.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let n_div = (1 + gv.len()).min(100).saturating_sub(1).max(1);
            let greater: Vec<(f64, f64)> = (0..gv.len())
                .map(|k| (gv[k], kernel_sigma(&gv, k, lo, hi, n_div)))
                .take(p.upper_groups)
                .collect();
            dims.push(Dim {
                lower,
                greater,
                greater_weight: gw,
            });
        }
        Some(TpeModel { dims })
    }

    /// 1 候補ベクトルをサンプルする。各次元で lower 混合から `ei_candidates` 個引き、
    /// 全次元にわたる Σ log(g/l) が最小の候補スロットを採用（tie は先頭）。
    pub fn sample<R: Rng>(
        &self,
        rng: &mut R,
        p: &TpeParams,
        l_limit: &[f64],
        g_limit: &[f64],
    ) -> Vec<f64> {
        let dim = self.dims.len();
        let m = p.ei_candidates.max(1);
        // 各候補スロット s の期待改善プロキシ Σ_dim log(g/l) を累積、per-dim の param を控える。
        let mut ei_per_slot = vec![0.0f64; m];
        let mut params = vec![vec![0.0f64; m]; dim]; // [dim][slot]
        for (j, d) in self.dims.iter().enumerate() {
            let (lo, hi) = (l_limit[j], g_limit[j]);
            for (s, ei) in ei_per_slot.iter_mut().enumerate() {
                // lower カーネルを一様に選び、その周りで正規サンプル（範囲外は再サンプル）。
                let mut param;
                loop {
                    let k = rng.gen_range(0..d.lower.len());
                    let (val, sigma) = d.lower[k];
                    param = sample_normal(rng, val, sigma);
                    if param >= lo && param <= hi {
                        break;
                    }
                }
                let lx = mixture_density(&d.lower, param, 1.0);
                let gx = mixture_density(&d.greater, param, d.greater_weight);
                // TPEvina は g/l を最小化（比小=良）。log で次元合成。
                let ratio = if lx > 0.0 { gx / lx } else { f64::INFINITY };
                *ei += ratio.ln();
                params[j][s] = param;
            }
        }
        // Σ_dim log(g/l) が最小のスロットを選ぶ（tie は先頭）。
        let best_slot = ei_per_slot
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(s, _)| s)
            .unwrap_or(0);
        params.iter().map(|row| row[best_slot]).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn normalize_angle_wraps() {
        use std::f64::consts::PI;
        assert!((normalize_angle(0.0)).abs() < 1e-12);
        assert!((normalize_angle(2.0 * PI)).abs() < 1e-9);
        assert!((normalize_angle(3.0 * PI) - (-PI)).abs() < 1e-9 || (normalize_angle(3.0 * PI) - PI).abs() < 1e-9);
        assert!(normalize_angle(4.0).abs() <= PI + 1e-12);
    }

    #[test]
    fn conf_vec_roundtrip() {
        let reference = Conformation {
            position: Vec3::new(1.0, 2.0, 3.0),
            orientation: UnitQuaternion::from_scaled_axis(Vec3::new(0.1, -0.2, 0.3)),
            torsions: vec![0.5, -0.3, 0.0],
        };
        let free = vec![0usize, 2];
        // モデルベクトル → conf → モデルベクトル でほぼ復元。
        let v = vec![4.0, -1.0, 2.5, 0.2, 0.1, -0.15, 0.4, -0.6];
        let c = conf_from_vec(&reference, &free, &v);
        let v2 = vec_from_conf(&reference, &free, &c);
        for (a, b) in v.iter().zip(&v2) {
            assert!((a - b).abs() < 1e-9, "{a} vs {b}");
        }
        // 固定 torsion(1) は参照のまま。
        assert!((c.torsions[1] - reference.torsions[1]).abs() < 1e-12);
    }

    #[test]
    fn model_build_and_sample_in_bounds() {
        let (l, g) = bounds(Vec3::zeros(), Vec3::new(10.0, 10.0, 10.0), 1); // dim=7
        let p = TpeParams::default();
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        // ランダム観測。
        let obs: Vec<Obs> = (0..30)
            .map(|i| Obs {
                v: warmup_sample(&mut rng, &l, &g),
                score: (i as f64 * 0.1).sin(),
            })
            .collect();
        let model = TpeModel::build(&obs, &p, 0, &l, &g).unwrap();
        for _ in 0..20 {
            let v = model.sample(&mut rng, &p, &l, &g);
            assert_eq!(v.len(), l.len());
            for j in 0..v.len() {
                assert!(v[j] >= l[j] - 1e-9 && v[j] <= g[j] + 1e-9, "dim {j} out of bounds: {}", v[j]);
            }
        }
    }

    #[test]
    fn determinism_same_seed() {
        let (l, g) = bounds(Vec3::zeros(), Vec3::new(8.0, 8.0, 8.0), 0);
        let p = TpeParams::default();
        let draw = || {
            let mut rng = ChaCha8Rng::seed_from_u64(42);
            let obs: Vec<Obs> = (0..20)
                .map(|i| Obs { v: warmup_sample(&mut rng, &l, &g), score: i as f64 })
                .collect();
            let model = TpeModel::build(&obs, &p, 0, &l, &g).unwrap();
            model.sample(&mut rng, &p, &l, &g)
        };
        assert_eq!(draw(), draw());
    }
}
