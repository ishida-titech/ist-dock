//! 受容体ポテンシャルのグリッド化（Vina 型・高速化）。
//!
//! 受容体は固定なので、リガンド XS 原子型ごとに「その型のプローブを各格子点に置いたときの
//! 対受容体 Vina エネルギー」を事前計算する（`VinaGrid`）。評価時は各リガンド原子が自分の型の
//! マップを **trilinear 補間**で引き、値とその解析的勾配（＝力）を得る。これにより評価が
//! `O(n_lig × n_rec)` から `O(n_lig × 8)` に下がり、ボックス全域に滑らかな勾配場ができる。
//!
//! マップは AutoDock4 型の事前 .map ではなく **Vina 流のオンデマンド生成**（`ensure`）。関数形は
//! `scoring::pair_energy`（= `eval_pair` の重み付き和）と同一で、補間誤差を除き直接計算に一致する。
//! 最終スコアは別途厳密 f64（`scoring::intermolecular_terms`）で採点する前提（探索専用近似）。

use rayon::prelude::*;
use wide::f64x4;

use crate::atom_types::XsType;
use crate::geometry::Vec3;
use crate::scoring::{ScoredAtom, CUTOFF};
use crate::scoring_simd::probe_energy_f64x4;

/// bool → f64 フラグ（1.0/0.0）。スキャッタ SIMD のペア係数用。
#[inline]
fn flag64(b: bool) -> f64 {
    if b {
        1.0
    } else {
        0.0
    }
}

/// 1 つのリガンド XS 型に対する 3D エネルギーマップ（f32、行優先 [ix + nx*(iy + ny*iz)]）。
#[derive(Debug, Clone)]
pub struct GridMap {
    origin: Vec3,
    spacing: f64,
    nx: usize,
    ny: usize,
    nz: usize,
    data: Vec<f32>,
}

impl GridMap {
    #[inline]
    fn at(&self, ix: usize, iy: usize, iz: usize) -> f32 {
        self.data[ix + self.nx * (iy + self.ny * iz)]
    }

    /// `pos` における trilinear 補間値と、その解析的勾配 `∇value`。境界外はクランプ。
    pub fn interpolate_value_grad(&self, pos: Vec3) -> (f64, Vec3) {
        let inv = 1.0 / self.spacing;
        // セル座標（格子点間隔単位）。境界内にクランプ（最後のセルの手前まで）。
        let clamp = |t: f64, n: usize| -> (usize, f64) {
            let tc = t.clamp(0.0, (n - 1) as f64 - 1e-9);
            let i = tc.floor() as usize;
            (i.min(n - 2), tc - i as f64)
        };
        let (ix, fx) = clamp((pos.x - self.origin.x) * inv, self.nx);
        let (iy, fy) = clamp((pos.y - self.origin.y) * inv, self.ny);
        let (iz, fz) = clamp((pos.z - self.origin.z) * inv, self.nz);

        // 8 隅。c[dx][dy][dz]。
        let c000 = self.at(ix, iy, iz) as f64;
        let c100 = self.at(ix + 1, iy, iz) as f64;
        let c010 = self.at(ix, iy + 1, iz) as f64;
        let c110 = self.at(ix + 1, iy + 1, iz) as f64;
        let c001 = self.at(ix, iy, iz + 1) as f64;
        let c101 = self.at(ix + 1, iy, iz + 1) as f64;
        let c011 = self.at(ix, iy + 1, iz + 1) as f64;
        let c111 = self.at(ix + 1, iy + 1, iz + 1) as f64;

        let (gx, gy, gz) = (1.0 - fx, 1.0 - fy, 1.0 - fz);
        // 値: 多重線形補間。
        let value = c000 * gx * gy * gz
            + c100 * fx * gy * gz
            + c010 * gx * fy * gz
            + c110 * fx * fy * gz
            + c001 * gx * gy * fz
            + c101 * fx * gy * fz
            + c011 * gx * fy * fz
            + c111 * fx * fy * fz;

        // 勾配: 補間式を各軸で微分（dt/dpos = 1/spacing）。
        let dvx = (-(c000) * gy * gz + c100 * gy * gz - c010 * fy * gz + c110 * fy * gz
            - c001 * gy * fz
            + c101 * gy * fz
            - c011 * fy * fz
            + c111 * fy * fz)
            * inv;
        let dvy = (-(c000) * gx * gz - c100 * fx * gz + c010 * gx * gz + c110 * fx * gz
            - c001 * gx * fz
            - c101 * fx * fz
            + c011 * gx * fz
            + c111 * fx * fz)
            * inv;
        let dvz = (-(c000) * gx * gy - c100 * fx * gy - c010 * gx * fy - c110 * fx * fy
            + c001 * gx * gy
            + c101 * fx * gy
            + c011 * gx * fy
            + c111 * fx * fy)
            * inv;

        (value, Vec3::new(dvx, dvy, dvz))
    }

    /// `pos` における trilinear 補間値のみ（勾配を計算しない直線探索用の高速版）。境界外はクランプ。
    pub fn interpolate_value(&self, pos: Vec3) -> f64 {
        let inv = 1.0 / self.spacing;
        let clamp = |t: f64, n: usize| -> (usize, f64) {
            let tc = t.clamp(0.0, (n - 1) as f64 - 1e-9);
            let i = tc.floor() as usize;
            (i.min(n - 2), tc - i as f64)
        };
        let (ix, fx) = clamp((pos.x - self.origin.x) * inv, self.nx);
        let (iy, fy) = clamp((pos.y - self.origin.y) * inv, self.ny);
        let (iz, fz) = clamp((pos.z - self.origin.z) * inv, self.nz);

        let c000 = self.at(ix, iy, iz) as f64;
        let c100 = self.at(ix + 1, iy, iz) as f64;
        let c010 = self.at(ix, iy + 1, iz) as f64;
        let c110 = self.at(ix + 1, iy + 1, iz) as f64;
        let c001 = self.at(ix, iy, iz + 1) as f64;
        let c101 = self.at(ix + 1, iy, iz + 1) as f64;
        let c011 = self.at(ix, iy + 1, iz + 1) as f64;
        let c111 = self.at(ix + 1, iy + 1, iz + 1) as f64;

        let (gx, gy, gz) = (1.0 - fx, 1.0 - fy, 1.0 - fz);
        c000 * gx * gy * gz
            + c100 * fx * gy * gz
            + c010 * gx * fy * gz
            + c110 * fx * fy * gz
            + c001 * gx * gy * fz
            + c101 * fx * gy * fz
            + c011 * gx * fy * fz
            + c111 * fx * fy * fz
    }
}

/// 受容体ボックスに対する型→マップの集合（オンデマンド生成）。
#[derive(Debug, Clone)]
pub struct VinaGrid {
    origin: Vec3,
    spacing: f64,
    nx: usize,
    ny: usize,
    nz: usize,
    /// 枝刈り済み受容体原子（マップ構築のスキャッタで走査）。
    receptor: Vec<ScoredAtom>,
    /// XS 型インデックス → マップ（未生成は None）。
    maps: Vec<Option<GridMap>>,
}

impl VinaGrid {
    /// 探索ボックス（中心・サイズ）に余白 `margin` を足した格子を用意する（マップは未生成）。
    pub fn new(receptor: &[ScoredAtom], center: Vec3, size: Vec3, spacing: f64, margin: f64) -> Self {
        let half = size * 0.5;
        let extent = Vec3::new(half.x + margin, half.y + margin, half.z + margin);
        let origin = center - extent;
        let span = extent * 2.0;
        let n = |s: f64| ((s / spacing).ceil() as usize) + 1;
        let (nx, ny, nz) = (n(span.x), n(span.y), n(span.z));
        VinaGrid {
            origin,
            spacing,
            nx,
            ny,
            nz,
            receptor: receptor.to_vec(),
            maps: vec![None; XsType::COUNT],
        }
    }

    /// 指定型のマップを（未生成なら）構築する。探索の並列実行前に呼ぶこと。
    pub fn ensure(&mut self, types: &[XsType]) {
        for &t in types {
            let i = t.index();
            if self.maps[i].is_none() {
                self.maps[i] = Some(self.build_map(t));
            }
        }
    }

    /// 型 `t` のマップを構築する（AutoGrid 流**スキャッタ**＋SIMD）。
    ///
    /// 格子点ごとに近傍を集める（gather）のでなく、**受容体原子ごとにカットオフ球内の格子点行（x 連続）へ寄与を
    /// 加算**する。x 連続 4 点を `probe_energy_f64x4`（f64x4, 値のみ）で一括評価し、ギャザーの materialize を避けて
    /// `exp` を 4 レーンで回す。z スライス単位で rayon 並列＝スライスごとに独立の f64 アキュムレータへ加算するので
    /// 書き込み競合なし。**f64 で合算**するため、現行の「点ごと f64 和→f32 キャスト」と加算順のみ差（~1e-16）＝
    /// f32 格納後はビット一致（＝精度中立）。カットオフ・ペア集合は gather と同一（`r²<CUTOFF²`）。
    /// 前提: AVX2（`.cargo/config.toml` の `x86-64-v3`）。
    fn build_map(&self, t: XsType) -> GridMap {
        let (nx, ny, nz) = (self.nx, self.ny, self.nz);
        let (ox, oy, oz) = (self.origin.x, self.origin.y, self.origin.z);
        let sp = self.spacing;
        let cutoff2 = CUTOFF * CUTOFF;
        // プローブ型（格子の対象 XS 型）のスカラ属性。
        let p_radius = t.radius();
        let p_hydro = flag64(t.is_hydrophobic());
        let p_donor = flag64(t.is_donor());
        let p_acceptor = flag64(t.is_acceptor());
        let receptor = &self.receptor;
        let mut data = vec![0.0f32; nx * ny * nz];
        data.par_chunks_mut(nx * ny).enumerate().for_each(|(iz, slice)| {
            let pz = oz + iz as f64 * sp;
            // このスライス専用の f64 アキュムレータ（現行の点ごと f64 和と同じ精度）。
            let mut acc = vec![0.0f64; nx * ny];
            for a in receptor.iter() {
                let (ax, ay, az) = (a.pos.x, a.pos.y, a.pos.z);
                let dz = pz - az;
                let rxy2 = cutoff2 - dz * dz; // xy 平面に残るカットオフ半径²
                if rxy2 <= 0.0 {
                    continue; // この z はこの原子のカットオフ球外
                }
                let a_radius = a.xs.radius();
                let a_hydro = flag64(a.xs.is_hydrophobic());
                let a_donor = flag64(a.xs.is_donor());
                let a_acceptor = flag64(a.xs.is_acceptor());
                let rxy = rxy2.sqrt();
                let iy_lo = (((ay - rxy - oy) / sp).floor() as isize).max(0);
                let iy_hi = (((ay + rxy - oy) / sp).ceil() as isize).min(ny as isize - 1);
                for iy in iy_lo..=iy_hi {
                    let py = oy + iy as f64 * sp;
                    let dy = py - ay;
                    let rx2 = rxy2 - dy * dy;
                    if rx2 <= 0.0 {
                        continue;
                    }
                    let rx = rx2.sqrt();
                    let ix_lo = (((ax - rx - ox) / sp).floor() as isize).max(0);
                    let ix_hi = (((ax + rx - ox) / sp).ceil() as isize).min(nx as isize - 1);
                    if ix_hi < ix_lo {
                        continue;
                    }
                    let row = nx * iy as usize;
                    let mut ix = ix_lo;
                    while ix <= ix_hi {
                        let x0 = ox + ix as f64 * sp;
                        let pxv = f64x4::new([x0, x0 + sp, x0 + 2.0 * sp, x0 + 3.0 * sp]);
                        let vals = probe_energy_f64x4(
                            pxv, py, pz, p_radius, p_hydro, p_donor, p_acceptor, ax, ay, az,
                            a_radius, a_hydro, a_donor, a_acceptor,
                        )
                        .to_array();
                        // カットオフ外レーンは 0。ix_hi を超えるレーンは書かない（範囲外・OOB 防止）。
                        let mut lane = 0usize;
                        while lane < 4 && ix + lane as isize <= ix_hi {
                            acc[row + (ix as usize + lane)] += vals[lane];
                            lane += 1;
                        }
                        ix += 4;
                    }
                }
            }
            for (dst, &v) in slice.iter_mut().zip(acc.iter()) {
                *dst = v as f32;
            }
        });
        GridMap {
            origin: self.origin,
            spacing: self.spacing,
            nx,
            ny,
            nz,
            data,
        }
    }

    /// 型 `t` のマップ（`ensure` 済み前提）。
    #[inline]
    pub fn map(&self, t: XsType) -> &GridMap {
        self.maps[t.index()]
            .as_ref()
            .expect("grid map not built for this type; call ensure() first")
    }

    /// リガンド原子群の分子間エネルギーと各原子に働く力 `force[i] = -∇E_i`（補間）。
    /// `force` は呼び出し側で 0 初期化（長さ = ligand.len()）。`cap`（hunt_cap）で各原子の正エネルギーを
    /// 上限化する（探索用。無効化は `f64::INFINITY`）。
    pub fn value_and_force(&self, ligand: &[ScoredAtom], force: &mut [Vec3], cap: f64) -> f64 {
        let mut e = 0.0;
        for (i, a) in ligand.iter().enumerate() {
            let (v, g) = self.map(a.xs).interpolate_value_grad(a.pos);
            let (ce, scale) = crate::scoring::curl(v, cap);
            e += ce;
            force[i] -= g * scale; // force = -∇value（curl の勾配スケール込み）
        }
        e
    }

    /// リガンド原子群の分子間エネルギーのみ（勾配なし）。直線探索（値のみ評価）の高速版。
    /// `cap`（hunt_cap）で各原子の正エネルギーを上限化（無効化は `f64::INFINITY`）。
    pub fn value(&self, ligand: &[ScoredAtom], cap: f64) -> f64 {
        ligand
            .iter()
            .map(|a| crate::scoring::curl(self.map(a.xs).interpolate_value(a.pos), cap).0)
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scoring::intermolecular_terms;

    fn atom(x: f64, y: f64, z: f64, xs: XsType) -> ScoredAtom {
        ScoredAtom { pos: Vec3::new(x, y, z), xs }
    }

    fn receptor() -> Vec<ScoredAtom> {
        vec![
            atom(0.0, 0.0, 0.0, XsType::OxygenA),
            atom(2.5, 1.0, -0.5, XsType::CarbonH),
            atom(-1.5, 2.0, 1.0, XsType::NitrogenD),
            atom(1.0, -2.0, 2.0, XsType::CarbonP),
        ]
    }

    #[test]
    fn scatter_build_node_matches_direct_sum() {
        use crate::scoring::eval_pair;
        // スキャッタ構築の各格子点値が、その点における直接ペア合算（厳密 f64）と一致することを固定する。
        // 範囲計算の off-by-one・マスク漏れ・原子取りこぼしを検出する回帰テスト。全ノードを走査。
        let rec = receptor();
        let mut grid = VinaGrid::new(&rec, Vec3::zeros(), Vec3::new(12.0, 12.0, 12.0), 0.375, 2.0);
        let t = XsType::CarbonH;
        grid.ensure(&[t]);
        let map = grid.map(t);
        let mut max_err = 0.0f64;
        for iz in 0..map.nz {
            for iy in 0..map.ny {
                for ix in 0..map.nx {
                    let p = Vec3::new(
                        map.origin.x + ix as f64 * map.spacing,
                        map.origin.y + iy as f64 * map.spacing,
                        map.origin.z + iz as f64 * map.spacing,
                    );
                    let mut direct = 0.0;
                    for r in &rec {
                        let d = (p - r.pos).norm();
                        if d < CUTOFF {
                            direct += eval_pair(t, r.xs, d).weighted_sum();
                        }
                    }
                    let node = map.data[ix + map.nx * (iy + map.ny * iz)] as f64;
                    max_err = max_err.max((node - direct).abs());
                }
            }
        }
        // f32 格納 + f64 加算順の差のみ＝実質ビット一致（全ノード）。
        assert!(max_err < 1e-4, "max scatter-vs-direct node error {max_err}");
    }

    #[test]
    fn grid_value_matches_direct_within_tolerance() {
        let rec = receptor();
        let mut grid = VinaGrid::new(&rec, Vec3::zeros(), Vec3::new(12.0, 12.0, 12.0), 0.25, 2.0);
        let types = [XsType::CarbonH, XsType::OxygenD];
        grid.ensure(&types);

        // 数点で grid（補間）と直接 f64 を比較。
        for &(x, y, z, xs) in &[
            (1.2, 0.3, 0.4, XsType::CarbonH),
            (-0.7, 1.1, 0.6, XsType::OxygenD),
            (2.0, -1.0, 1.0, XsType::CarbonH),
        ] {
            let lig = vec![atom(x, y, z, xs)];
            let mut f = vec![Vec3::zeros(); 1];
            let e_grid = grid.value_and_force(&lig, &mut f, f64::INFINITY);
            let e_direct = intermolecular_terms(&lig, &rec).weighted_sum();
            assert!(
                (e_grid - e_direct).abs() < 0.3,
                "grid {e_grid} vs direct {e_direct} at ({x},{y},{z})"
            );
        }
    }

    #[test]
    fn grid_gradient_matches_numerical() {
        let rec = receptor();
        let mut grid = VinaGrid::new(&rec, Vec3::zeros(), Vec3::new(12.0, 12.0, 12.0), 0.25, 2.0);
        grid.ensure(&[XsType::CarbonH]);
        let map = grid.map(XsType::CarbonH);
        let p = Vec3::new(1.13, -0.42, 0.77);
        let (_, g) = map.interpolate_value_grad(p);
        let eps = 1e-4;
        for axis in 0..3 {
            let mut pp = p;
            pp[axis] += eps;
            let mut pm = p;
            pm[axis] -= eps;
            let num = (map.interpolate_value_grad(pp).0 - map.interpolate_value_grad(pm).0) / (2.0 * eps);
            assert!((g[axis] - num).abs() < 1e-3, "axis {axis}: {} vs {}", g[axis], num);
        }
    }

    #[test]
    fn finer_spacing_reduces_error() {
        let rec = receptor();
        let lig = vec![atom(0.83, 0.41, -0.27, XsType::CarbonH)];
        let direct = intermolecular_terms(&lig, &rec).weighted_sum();
        let err = |sp: f64| {
            let mut g = VinaGrid::new(&rec, Vec3::zeros(), Vec3::new(12.0, 12.0, 12.0), sp, 2.0);
            g.ensure(&[XsType::CarbonH]);
            let mut f = vec![Vec3::zeros(); 1];
            (g.value_and_force(&lig, &mut f, f64::INFINITY) - direct).abs()
        };
        assert!(err(0.1) <= err(0.5) + 1e-9, "finer spacing should not be worse");
    }
}
