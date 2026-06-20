//! QuickVina2 相当の局所探索スキップ（first-order-consistency check）。opt-in 高速化モード。
//!
//! 各 run（スレッド）の circular history buffer に対し、変異後の試行点が「新しいエネルギー井戸」か
//! を勾配符号で判定し、そうでなければ高価な BFGS 局所探索（`minimize_free`）をスキップする。
//! スレッドローカル＝**共有状態なし＝決定論的**（QuickVina-W の共有 octree は別フェーズ・非決定論）。
//! 参照: QuickVina2 (Alhossary et al. 2015, doi:10.1093/bioinformatics/btv082)。

use std::collections::VecDeque;

use crate::ligand::Conformation;

/// QuickVina2 スキップ判定のパラメータ（N=自由 DOF 数から構成）。
#[derive(Debug, Clone, Copy)]
pub struct QVinaParams {
    /// circular history の容量（QVina ~10N）。
    pub buffer_cap: usize,
    /// 判定に使う最近傍数（QVina ~2N）。
    pub p_neighbors: usize,
    /// 勾配方向の一致閾値（cosine）。近傍の勾配と cos>これ＝同じ斜面（同一 basin）＝局所探索スキップ。
    /// 高いほど保守的（スキップ少）。1 ペア変数の符号一致だけ見る素朴版は高次元で無効化されるため、
    /// **勾配方向（first-order）の一致**で「既知の井戸へ向かう試行」を頑健に判定する。
    pub cos_threshold: f64,
}

impl QVinaParams {
    /// 自由 DOF 数 `n` から既定パラメータを構成。
    pub fn for_dofs(n: usize) -> Self {
        QVinaParams {
            buffer_cap: (10 * n).max(10),
            p_neighbors: (2 * n).max(2),
            cos_threshold: 0.95,
        }
    }
}

/// 試行 conformation の DOF 埋め込み = [position(3), 配向 scaled-axis(3), torsion...] を `free_dofs` に射影。
/// 近傍探索の距離計量用。勾配（`value_grad` 返り値）と同じ DOF 添字空間。
pub fn dof_vector(conf: &Conformation, free_dofs: &[usize]) -> Vec<f64> {
    let ax = conf.orientation.scaled_axis();
    free_dofs
        .iter()
        .map(|&i| match i {
            0 => conf.position.x,
            1 => conf.position.y,
            2 => conf.position.z,
            3 => ax.x,
            4 => ax.y,
            5 => ax.z,
            _ => conf.torsions[i - 6],
        })
        .collect()
}

/// 全 DOF 勾配（長さ 6+n_torsions）を `free_dofs` に射影。
pub fn project_grad(grad_full: &[f64], free_dofs: &[usize]) -> Vec<f64> {
    free_dofs.iter().map(|&i| grad_full[i]).collect()
}

/// 過去に評価した点（dof・射影済み勾配）の circular buffer。
pub struct HistoryBuffer {
    points: VecDeque<(Vec<f64>, Vec<f64>)>,
    params: QVinaParams,
}

impl HistoryBuffer {
    pub fn new(params: QVinaParams) -> Self {
        HistoryBuffer {
            points: VecDeque::with_capacity(params.buffer_cap),
            params,
        }
    }

    /// 評価点を追加（容量超過で最古を破棄＝circular）。
    pub fn push(&mut self, dof: Vec<f64>, grad: Vec<f64>) {
        if self.points.len() >= self.params.buffer_cap {
            self.points.pop_front();
        }
        self.points.push_back((dof, grad));
    }

    /// QVina2 相当の first-order-consistency: 最近傍 P 点のうち**勾配方向が十分一致**（cos>閾値）する点が
    /// あれば、その試行は**既に探索した斜面（同一 basin）**へ向かう＝局所探索しても既知極小に収束する
    /// ＝**スキップ**(false)。どの近傍とも方向が異なる＝新しい井戸＝**局所探索すべき**(true)。履歴が空・
    /// 勾配が極小（停留点近傍）なら true（探索/保持）。
    pub fn is_new_well(&self, dof: &[f64], grad: &[f64]) -> bool {
        if self.points.is_empty() {
            return true;
        }
        let gn = norm(grad);
        if gn < 1e-9 {
            return true; // ほぼ停留点（最小に近い）→ 局所探索で確定させる
        }
        let mut neigh: Vec<(f64, usize)> = self
            .points
            .iter()
            .enumerate()
            .map(|(i, (d, _))| (dist2(dof, d), i))
            .collect();
        let p = self.params.p_neighbors.min(neigh.len());
        if p < neigh.len() {
            neigh.select_nth_unstable_by(p, |a, b| {
                a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
            });
        }
        for &(_, idx) in neigh.iter().take(p) {
            let gh = &self.points[idx].1;
            let hn = norm(gh);
            if hn < 1e-9 {
                continue;
            }
            let cos = dot(grad, gh) / (gn * hn);
            if cos > self.params.cos_threshold {
                return false; // 既知斜面と同方向 → スキップ
            }
        }
        true // どの近傍とも方向が異なる → 新しい井戸 → 局所探索
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

fn dist2(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buffer_is_circular() {
        let mut h = HistoryBuffer::new(QVinaParams { buffer_cap: 2, p_neighbors: 2, cos_threshold: 0.95 });
        h.push(vec![0.0], vec![1.0]);
        h.push(vec![1.0], vec![1.0]);
        h.push(vec![2.0], vec![1.0]); // 最古(0.0)を破棄
        assert_eq!(h.points.len(), 2);
        assert_eq!(h.points.front().unwrap().0, vec![1.0]);
    }

    #[test]
    fn empty_history_is_significant() {
        let h = HistoryBuffer::new(QVinaParams::for_dofs(3));
        assert!(h.is_new_well(&[0.0, 0.0, 0.0], &[1.0, 1.0, 1.0]));
    }

    #[test]
    fn consistent_neighbor_is_skipped() {
        let mut h = HistoryBuffer::new(QVinaParams { buffer_cap: 8, p_neighbors: 4, cos_threshold: 0.95 });
        // 近傍で勾配が全変数同符号 → 既知斜面 → skip(false)。
        h.push(vec![0.0, 0.0], vec![1.0, 2.0]);
        assert!(!h.is_new_well(&[0.01, 0.01], &[1.5, 2.5]));
    }

    #[test]
    fn opposite_sign_neighbor_is_significant() {
        let mut h = HistoryBuffer::new(QVinaParams { buffer_cap: 8, p_neighbors: 4, cos_threshold: 0.95 });
        // ある変数で符号が逆 → 停留点を挟む → 局所探索(true)。
        h.push(vec![0.0, 0.0], vec![1.0, 2.0]);
        assert!(h.is_new_well(&[0.01, 0.01], &[-1.0, 2.5]));
    }
}
