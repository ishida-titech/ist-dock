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

use crate::atom_types::XsType;
use crate::geometry::Vec3;
use crate::scoring::{pair_energy_grad, ScoredAtom, CUTOFF};

/// 受容体原子の均一セルリスト（マップ構築で各格子点の近傍原子だけを走査するため）。
/// セル辺長 = CUTOFF なので、ある点の近傍 3×3×3 セルに「その点からカットオフ内の原子」が必ず入る。
#[derive(Debug, Clone)]
struct CellList {
    origin: Vec3,
    cell: f64,
    nx: usize,
    ny: usize,
    nz: usize,
    cells: Vec<Vec<u32>>,
    atoms: Vec<ScoredAtom>,
}

impl CellList {
    fn new(atoms: &[ScoredAtom]) -> Self {
        let cell = CUTOFF;
        let mut lo = Vec3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY);
        let mut hi = Vec3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
        for a in atoms {
            lo = lo.inf(&a.pos);
            hi = hi.sup(&a.pos);
        }
        if atoms.is_empty() {
            lo = Vec3::zeros();
            hi = Vec3::zeros();
        }
        // 余白 1 セルを足してクランプ時の取りこぼしを防ぐ。
        let origin = lo - Vec3::new(cell, cell, cell);
        let span = (hi - lo) + Vec3::new(2.0 * cell, 2.0 * cell, 2.0 * cell);
        let n = |s: f64| ((s / cell).ceil() as usize).max(1);
        let (nx, ny, nz) = (n(span.x), n(span.y), n(span.z));
        let idx = |v: f64, o: f64, m: usize| {
            (((v - o) / cell).floor() as isize).clamp(0, m as isize - 1) as usize
        };
        let mut cells = vec![Vec::new(); nx * ny * nz];
        for (i, a) in atoms.iter().enumerate() {
            let cx = idx(a.pos.x, origin.x, nx);
            let cy = idx(a.pos.y, origin.y, ny);
            let cz = idx(a.pos.z, origin.z, nz);
            cells[cx + nx * (cy + ny * cz)].push(i as u32);
        }
        CellList { origin, cell, nx, ny, nz, cells, atoms: atoms.to_vec() }
    }

    /// 点 `p` のカットオフ内候補（3×3×3 近傍セル）の受容体原子に `f` を適用する。
    #[inline]
    fn for_neighbors(&self, p: Vec3, mut f: impl FnMut(&ScoredAtom)) {
        let cx = (((p.x - self.origin.x) / self.cell).floor() as isize).clamp(0, self.nx as isize - 1);
        let cy = (((p.y - self.origin.y) / self.cell).floor() as isize).clamp(0, self.ny as isize - 1);
        let cz = (((p.z - self.origin.z) / self.cell).floor() as isize).clamp(0, self.nz as isize - 1);
        for dz in -1..=1 {
            let z = cz + dz;
            if z < 0 || z >= self.nz as isize {
                continue;
            }
            for dy in -1..=1 {
                let y = cy + dy;
                if y < 0 || y >= self.ny as isize {
                    continue;
                }
                for dx in -1..=1 {
                    let x = cx + dx;
                    if x < 0 || x >= self.nx as isize {
                        continue;
                    }
                    let c = x as usize + self.nx * (y as usize + self.ny * z as usize);
                    for &ai in &self.cells[c] {
                        f(&self.atoms[ai as usize]);
                    }
                }
            }
        }
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
    /// 受容体のセルリスト（マップ構築で近傍原子だけ走査）。
    cells: CellList,
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
            cells: CellList::new(receptor),
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

    /// 型 `t` のマップを構築する（各格子点で近傍受容体の Vina エネルギーを合算）。
    /// z スライス単位で rayon 並列、近傍はセルリストで限定。
    fn build_map(&self, t: XsType) -> GridMap {
        let (nx, ny, nz) = (self.nx, self.ny, self.nz);
        let cutoff2 = CUTOFF * CUTOFF;
        let mut data = vec![0.0f32; nx * ny * nz];
        data.par_chunks_mut(nx * ny).enumerate().for_each(|(iz, slice)| {
            let pz = self.origin.z + iz as f64 * self.spacing;
            for iy in 0..ny {
                let py = self.origin.y + iy as f64 * self.spacing;
                for ix in 0..nx {
                    let px = self.origin.x + ix as f64 * self.spacing;
                    let p = Vec3::new(px, py, pz);
                    let mut e = 0.0;
                    self.cells.for_neighbors(p, |r| {
                        let d2 = (p - r.pos).norm_squared();
                        if d2 < cutoff2 {
                            e += pair_energy_grad(t, r.xs, d2.sqrt()).0;
                        }
                    });
                    slice[ix + nx * iy] = e as f32;
                }
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
    /// `force` は呼び出し側で 0 初期化（長さ = ligand.len()）。
    pub fn value_and_force(&self, ligand: &[ScoredAtom], force: &mut [Vec3]) -> f64 {
        let mut e = 0.0;
        for (i, a) in ligand.iter().enumerate() {
            let (v, g) = self.map(a.xs).interpolate_value_grad(a.pos);
            e += v;
            force[i] -= g; // force = -∇value
        }
        e
    }

    /// リガンド原子群の分子間エネルギーのみ（勾配なし）。直線探索（値のみ評価）の高速版。
    pub fn value(&self, ligand: &[ScoredAtom]) -> f64 {
        ligand.iter().map(|a| self.map(a.xs).interpolate_value(a.pos)).sum()
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
            let e_grid = grid.value_and_force(&lig, &mut f);
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
            (g.value_and_force(&lig, &mut f) - direct).abs()
        };
        assert!(err(0.1) <= err(0.5) + 1e-9, "finer spacing should not be worse");
    }
}
