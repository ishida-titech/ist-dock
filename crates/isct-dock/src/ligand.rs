//! リガンドのトポロジー表現とポーズ再構成（M2）。
//!
//! PDBQT の torsion 木（フラグメント）から、剛体並進(3)＋回転(四元数)＋torsion(n) を
//! 自由度とするポーズ再構成を行う。参照座標（入力ポーズ）を基準に、torsion を木順
//! （親→子）で適用してから剛体変換を掛ける。恒等変換は入力ポーズを再現する。

use std::collections::HashSet;

use isct_core::atom_types::XsType;
use isct_core::geometry::{distance, Vec3};
use isct_core::scoring::{curl, eval_pair, pair_energy_grad, ScoredAtom, CUTOFF};
use isct_io::pdbqt::Molecule;
use nalgebra::{Unit, UnitQuaternion};
use rand::Rng;

/// リガンドの自由度状態。`torsions[k]` は非 root フラグメント(k+1) の絶対角（0=参照）。
#[derive(Debug, Clone)]
pub struct Conformation {
    pub position: Vec3,
    pub orientation: UnitQuaternion<f64>,
    pub torsions: Vec<f64>,
}

impl Conformation {
    /// 恒等変換（参照ポーズ）。
    pub fn identity(n_torsions: usize) -> Self {
        Conformation {
            position: Vec3::zeros(),
            orientation: UnitQuaternion::identity(),
            torsions: vec![0.0; n_torsions],
        }
    }

    /// 自由度数 = 並進3 + 回転3 + torsion n。
    pub fn dim(&self) -> usize {
        6 + self.torsions.len()
    }

    /// 増分ベクトル（長さ dim）を適用する。
    /// `d[0..3]`=並進, `d[3..6]`=回転ベクトル(軸×角), `d[6..]`=torsion 増分。
    pub fn apply_delta(&mut self, d: &[f64]) {
        self.position += Vec3::new(d[0], d[1], d[2]);
        let rotvec = Vec3::new(d[3], d[4], d[5]);
        self.orientation = UnitQuaternion::from_scaled_axis(rotvec) * self.orientation;
        for (t, dt) in self.torsions.iter_mut().zip(&d[6..]) {
            *t += dt;
        }
    }
}

/// リガンドの実行時モデル（参照座標・木構造・スコア対象重原子）。
#[derive(Debug, Clone)]
pub struct LigandModel {
    /// 全原子の参照座標（水素含む）。
    ref_coords: Vec<Vec3>,
    /// フラグメントごとの回転結合 (parent側 atom idx, 自身側 atom idx)。root は None。
    bonds: Vec<Option<(usize, usize)>>,
    /// 各フラグメントの部分木に属する全原子インデックス。
    subtree: Vec<Vec<usize>>,
    /// root 原子の重心（剛体変換のピボット）。
    root_pivot: Vec3,
    /// スコア対象重原子の (全原子 idx, XS 型)。
    scored: Vec<(usize, XsType)>,
    /// 分子内エネルギーを評価する重原子ペア（scored インデックス空間）。
    /// 異なるフラグメント間（torsion で距離が変わる）かつ結合グラフ距離 ≥3（1-2/1-3 除外）。
    intra_pairs: Vec<(usize, usize)>,
    /// フラグメント f（≥1）の部分木に属する **scored 原子インデックス**（torsion 勾配用）。
    scored_subtree: Vec<Vec<usize>>,
    n_torsions: usize,
}

impl LigandModel {
    pub fn new(mol: &Molecule) -> Self {
        let ref_coords: Vec<Vec3> = mol.atoms.iter().map(|a| a.pos).collect();
        let nf = mol.fragments.len();

        // 子リスト
        let mut children: Vec<Vec<usize>> = vec![Vec::new(); nf];
        for (i, f) in mol.fragments.iter().enumerate() {
            if let Some(p) = f.parent {
                children[p].push(i);
            }
        }
        // 部分木原子（DFS）
        let subtree: Vec<Vec<usize>> = (0..nf)
            .map(|f| {
                let mut acc = Vec::new();
                let mut stack = vec![f];
                while let Some(g) = stack.pop() {
                    acc.extend_from_slice(&mol.fragments[g].atoms);
                    stack.extend_from_slice(&children[g]);
                }
                acc
            })
            .collect();

        let bonds: Vec<Option<(usize, usize)>> = mol.fragments.iter().map(|f| f.bond).collect();

        let root_atoms = &mol.fragments[0].atoms;
        let root_pivot = root_atoms.iter().fold(Vec3::zeros(), |s, &i| s + ref_coords[i])
            / root_atoms.len().max(1) as f64;

        let xs = mol.xs_types();
        let scored: Vec<(usize, XsType)> = xs
            .iter()
            .enumerate()
            .filter_map(|(i, x)| x.map(|t| (i, t)))
            .collect();

        let intra_pairs = Self::build_intra_pairs(mol, &ref_coords, &scored);

        // 各フラグメントの部分木に属する scored 原子（all-atom idx → scored idx の逆引き）。
        let mut all_to_scored: Vec<Option<usize>> = vec![None; ref_coords.len()];
        for (k, &(ai, _)) in scored.iter().enumerate() {
            all_to_scored[ai] = Some(k);
        }
        let scored_subtree: Vec<Vec<usize>> = subtree
            .iter()
            .map(|atoms| atoms.iter().filter_map(|&ai| all_to_scored[ai]).collect())
            .collect();

        LigandModel {
            ref_coords,
            bonds,
            subtree,
            root_pivot,
            scored,
            intra_pairs,
            scored_subtree,
            n_torsions: nf - 1,
        }
    }

    /// 分子内エネルギー対象ペアを事前計算する。
    /// 異なるフラグメント間（torsion で距離が変化）かつ結合グラフ距離 ≥3（1-2/1-3 を除外）。
    fn build_intra_pairs(
        mol: &Molecule,
        ref_coords: &[Vec3],
        scored: &[(usize, XsType)],
    ) -> Vec<(usize, usize)> {
        let n = scored.len();
        let orig: Vec<usize> = scored.iter().map(|&(i, _)| i).collect();

        // 重原子間の結合隣接（scored インデックス空間）。
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
        for a in 0..n {
            for b in (a + 1)..n {
                let (ia, ib) = (orig[a], orig[b]);
                let ea = mol.atoms[ia].ad.element();
                let eb = mol.atoms[ib].ad.element();
                let r = distance(&ref_coords[ia], &ref_coords[ib]);
                if r < (ea.covalent_radius() + eb.covalent_radius()) * 1.1 {
                    adj[a].push(b);
                    adj[b].push(a);
                }
            }
        }

        // 1-2/1-3（グラフ距離 ≤2）を除外集合へ。
        let mut excluded: Vec<HashSet<usize>> = vec![HashSet::new(); n];
        for a in 0..n {
            for &b in &adj[a] {
                excluded[a].insert(b);
                for &c in &adj[b] {
                    if c != a {
                        excluded[a].insert(c);
                    }
                }
            }
        }

        let mut pairs = Vec::new();
        for a in 0..n {
            for b in (a + 1)..n {
                let diff_fragment = mol.atom_fragment[orig[a]] != mol.atom_fragment[orig[b]];
                if diff_fragment && !excluded[a].contains(&b) {
                    pairs.push((a, b));
                }
            }
        }
        pairs
    }

    /// 分子内エネルギー（フラグメント間の非結合重原子ペアの Vina スコア和）。
    /// `cap`（hunt_cap）で各ペアの正エネルギー（激突）を上限化する（探索用。無効化は `f64::INFINITY`）。
    pub fn intramolecular(&self, atoms: &[ScoredAtom], cap: f64) -> f64 {
        let mut e = 0.0;
        for &(a, b) in &self.intra_pairs {
            let r = distance(&atoms[a].pos, &atoms[b].pos);
            if r < CUTOFF {
                let pe = eval_pair(atoms[a].xs, atoms[b].xs, r).weighted_sum();
                e += curl(pe, cap).0;
            }
        }
        e
    }

    /// 分子内エネルギーと、各 scored 原子に働く力（`force[k] -= dE/dpos_k`）を加算する。
    /// `cap`（hunt_cap）で各ペアの正エネルギーを上限化し、勾配も curl のスケールを掛ける。
    pub fn intramolecular_value_grad(&self, atoms: &[ScoredAtom], force: &mut [Vec3], cap: f64) -> f64 {
        let mut e = 0.0;
        for &(a, b) in &self.intra_pairs {
            let diff = atoms[a].pos - atoms[b].pos;
            let r = diff.norm();
            if r < CUTOFF && r > 1e-8 {
                let (v, dvdr) = pair_energy_grad(atoms[a].xs, atoms[b].xs, r);
                let (cv, scale) = curl(v, cap);
                e += cv;
                // dE/dpos_a = dvdr*diff/r, dE/dpos_b = -それ。force = -dE/dpos。curl の勾配スケール込み。
                let f = diff * (dvdr / r * scale);
                force[a] -= f;
                force[b] += f;
            }
        }
        e
    }

    /// world 座標（全原子）に加え、各非 root フラグメントの **world 系**回転軸・ピボットを返す。
    /// torsion 勾配の連鎖則に使う。`world_coords` と同一のポーズ再構成を行う。
    pub fn world_coords_frames(&self, conf: &Conformation) -> (Vec<Vec3>, Vec<Vec3>, Vec<Vec3>) {
        let (mut pos, mut world_axis, mut world_pivot) = (Vec::new(), Vec::new(), Vec::new());
        let (mut ref_axis, mut ref_pivot) = (Vec::new(), Vec::new());
        self.world_coords_frames_into(
            conf, &mut pos, &mut ref_axis, &mut ref_pivot, &mut world_axis, &mut world_pivot,
        );
        (pos, world_axis, world_pivot)
    }

    /// `world_coords_frames` のゼロアロケーション版: 与えられた再利用バッファへ書き込む。
    /// `ref_axis`/`ref_pivot` は中間バッファ（呼び出し側は無視してよい）。
    pub fn world_coords_frames_into(
        &self,
        conf: &Conformation,
        pos: &mut Vec<Vec3>,
        ref_axis: &mut Vec<Vec3>,
        ref_pivot: &mut Vec<Vec3>,
        world_axis: &mut Vec<Vec3>,
        world_pivot: &mut Vec<Vec3>,
    ) {
        let nf = self.bonds.len();
        pos.clear();
        pos.extend_from_slice(&self.ref_coords);
        ref_axis.clear();
        ref_axis.resize(nf, Vec3::zeros());
        ref_pivot.clear();
        ref_pivot.resize(nf, Vec3::zeros());

        for f in 1..nf {
            let (a0, a1) = self.bonds[f].expect("non-root fragment must have a bond");
            let axis = pos[a1] - pos[a0];
            let an = axis.norm();
            // 親 torsion 適用後・子 torsion はピボット/軸を動かさないため、ここが最終 ref 系の値。
            ref_axis[f] = if an < 1e-9 { Vec3::zeros() } else { axis / an };
            ref_pivot[f] = pos[a1];

            let angle = conf.torsions[f - 1];
            if angle != 0.0 && an >= 1e-9 {
                let q = UnitQuaternion::from_axis_angle(&Unit::new_normalize(axis), angle);
                let pivot = pos[a1];
                for &i in &self.subtree[f] {
                    pos[i] = pivot + q * (pos[i] - pivot);
                }
            }
        }

        // 剛体変換（root 重心を中心に回転 → 並進）。
        for p in pos.iter_mut() {
            *p = self.root_pivot + conf.orientation * (*p - self.root_pivot) + conf.position;
        }
        // ref 系の軸・ピボットを world 系へ。
        world_axis.clear();
        world_axis.extend(ref_axis.iter().map(|a| conf.orientation * a));
        world_pivot.clear();
        world_pivot.extend(
            ref_pivot
                .iter()
                .map(|p| self.root_pivot + conf.orientation * (p - self.root_pivot) + conf.position),
        );
    }

    /// 事前計算した world 座標から scored 原子（位置＋XS 型）を作る（再計算を避ける）。
    pub fn scored_atoms_at(&self, world_pos: &[Vec3]) -> Vec<ScoredAtom> {
        let mut atoms = Vec::new();
        self.scored_atoms_into(world_pos, &mut atoms);
        atoms
    }

    /// `scored_atoms_at` のゼロアロケーション版: 再利用バッファ `atoms` へ書き込む。
    pub fn scored_atoms_into(&self, world_pos: &[Vec3], atoms: &mut Vec<ScoredAtom>) {
        atoms.clear();
        atoms.extend(self.scored.iter().map(|&(i, xs)| ScoredAtom { pos: world_pos[i], xs }));
    }

    /// scored 原子に働く力 `force[k] = -dE/dpos_k` から、DOF 勾配（長さ dim）を連鎖則で求める。
    /// `world_pos`/`world_axis`/`world_pivot` は `world_coords_frames` の出力。
    pub fn dof_gradient(
        &self,
        conf: &Conformation,
        world_pos: &[Vec3],
        world_axis: &[Vec3],
        world_pivot: &[Vec3],
        force: &[Vec3],
    ) -> Vec<f64> {
        let mut grad = Vec::new();
        self.dof_gradient_into(conf, world_pos, world_axis, world_pivot, force, &mut grad);
        grad
    }

    /// `dof_gradient` のゼロアロケーション版: 再利用バッファ `grad`（長さ dim）へ書き込む。
    pub fn dof_gradient_into(
        &self,
        conf: &Conformation,
        world_pos: &[Vec3],
        world_axis: &[Vec3],
        world_pivot: &[Vec3],
        force: &[Vec3],
        grad: &mut Vec<f64>,
    ) {
        grad.clear();
        grad.resize(6 + self.n_torsions, 0.0);
        // 回転のピボットは world 系の root（root_pivot + 並進）。
        let rot_pivot = self.root_pivot + conf.position;

        let mut g_trans = Vec3::zeros();
        let mut g_rot = Vec3::zeros();
        for (k, &(ai, _)) in self.scored.iter().enumerate() {
            let g = -force[k]; // dE/dpos
            g_trans += g;
            g_rot += (world_pos[ai] - rot_pivot).cross(&g);
        }
        grad[0] = g_trans.x;
        grad[1] = g_trans.y;
        grad[2] = g_trans.z;
        grad[3] = g_rot.x;
        grad[4] = g_rot.y;
        grad[5] = g_rot.z;

        for f in 1..self.bonds.len() {
            let mut acc = Vec3::zeros();
            for &k in &self.scored_subtree[f] {
                let ai = self.scored[k].0;
                let g = -force[k];
                acc += (world_pos[ai] - world_pivot[f]).cross(&g);
            }
            grad[6 + f - 1] = world_axis[f].dot(&acc);
        }
    }

    pub fn n_torsions(&self) -> usize {
        self.n_torsions
    }

    /// 分子内エネルギー評価ペア数。
    pub fn n_intra_pairs(&self) -> usize {
        self.intra_pairs.len()
    }

    pub fn n_scored(&self) -> usize {
        self.scored.len()
    }

    /// 全原子の参照座標数。
    pub fn n_atoms(&self) -> usize {
        self.ref_coords.len()
    }

    /// 全原子の参照座標（入力ポーズ）。Phase 2 で子コアの参照 world 座標を取るのに使う。
    pub fn ref_coords(&self) -> &[Vec3] {
        &self.ref_coords
    }

    /// 剛体変換のピボット（root 重心）。Phase 2 で剛体配置を Conformation に変換するのに使う。
    pub fn root_pivot(&self) -> Vec3 {
        self.root_pivot
    }

    /// スコア対象重原子それぞれの全原子インデックス（コア/R 基マスク構築用）。
    pub fn scored_all_indices(&self) -> Vec<usize> {
        self.scored.iter().map(|&(i, _)| i).collect()
    }

    /// R 基 torsion のインデックス（0-based, torsion k ↔ フラグメント k+1）。
    /// 「回しても **コア原子を 1 つも動かさない**」torsion のみ＝部分木にコアを含まないもの。
    /// これにより、コアが root でなくても（Meeko の root 選択に依存せず）コアを固定したまま
    /// R 基だけを最適化できる。`core_atoms` は全原子インデックス集合。
    pub fn rgroup_torsions(&self, core_atoms: &HashSet<usize>) -> Vec<usize> {
        (0..self.n_torsions)
            .filter(|&k| {
                let f = k + 1; // torsion k は非 root フラグメント k+1 に対応
                !self.subtree[f].iter().any(|i| core_atoms.contains(i))
            })
            .collect()
    }

    /// 与えられた conformation の全原子 world 座標を再構成する。
    pub fn world_coords(&self, conf: &Conformation) -> Vec<Vec3> {
        let mut pos = Vec::new();
        self.world_coords_into(conf, &mut pos);
        pos
    }

    /// `world_coords` のゼロアロケーション版: 再利用バッファ `pos` へ書き込む。
    pub fn world_coords_into(&self, conf: &Conformation, pos: &mut Vec<Vec3>) {
        pos.clear();
        pos.extend_from_slice(&self.ref_coords);

        // torsion をフラグメント木順（親→子, 添字昇順）に適用。
        for f in 1..self.bonds.len() {
            let angle = conf.torsions[f - 1];
            if angle == 0.0 {
                continue;
            }
            let (a0, a1) = self.bonds[f].expect("non-root fragment must have a bond");
            let axis = pos[a1] - pos[a0];
            if axis.norm() < 1e-9 {
                continue;
            }
            let q = UnitQuaternion::from_axis_angle(&Unit::new_normalize(axis), angle);
            let pivot = pos[a1];
            for &i in &self.subtree[f] {
                pos[i] = pivot + q * (pos[i] - pivot);
            }
        }

        // 剛体変換（root 重心を中心に回転 → 並進）。
        for p in pos.iter_mut() {
            *p = self.root_pivot + conf.orientation * (*p - self.root_pivot) + conf.position;
        }
    }

    /// スコア対象重原子（位置＋XS 型）を返す。
    pub fn scored_atoms(&self, conf: &Conformation) -> Vec<ScoredAtom> {
        let pos = self.world_coords(conf);
        self.scored
            .iter()
            .map(|&(i, xs)| ScoredAtom { pos: pos[i], xs })
            .collect()
    }

    /// スコア対象重原子の world 座標のみ（RMSD 用）。
    pub fn scored_positions(&self, conf: &Conformation) -> Vec<Vec3> {
        let pos = self.world_coords(conf);
        self.scored.iter().map(|&(i, _)| pos[i]).collect()
    }

    /// 参照ポーズ（恒等変換）の重原子 world 座標。
    pub fn reference_scored_positions(&self) -> Vec<Vec3> {
        self.scored.iter().map(|&(i, _)| self.ref_coords[i]).collect()
    }

    /// 探索ボックス内にランダム配置した初期 conformation を作る。
    pub fn random_in_box<R: Rng>(&self, rng: &mut R, center: Vec3, size: Vec3) -> Conformation {
        let target = Vec3::new(
            center.x + (rng.r#gen::<f64>() - 0.5) * size.x,
            center.y + (rng.r#gen::<f64>() - 0.5) * size.y,
            center.z + (rng.r#gen::<f64>() - 0.5) * size.z,
        );
        Conformation {
            position: target - self.root_pivot,
            orientation: random_quaternion(rng),
            torsions: (0..self.n_torsions)
                .map(|_| (rng.r#gen::<f64>() - 0.5) * 2.0 * std::f64::consts::PI)
                .collect(),
        }
    }
}

/// 一様ランダムな単位四元数。
pub fn random_quaternion<R: Rng>(rng: &mut R) -> UnitQuaternion<f64> {
    // Shoemake 法。
    let u1: f64 = rng.r#gen();
    let u2: f64 = rng.r#gen();
    let u3: f64 = rng.r#gen();
    let tau = std::f64::consts::TAU;
    let q = nalgebra::Quaternion::new(
        (u1).sqrt() * (tau * u3).cos(),
        (1.0 - u1).sqrt() * (tau * u2).sin(),
        (1.0 - u1).sqrt() * (tau * u2).cos(),
        (u1).sqrt() * (tau * u3).sin(),
    );
    UnitQuaternion::from_quaternion(q)
}

#[cfg(test)]
mod tests {
    use super::*;
    use isct_io::pdbqt;

    const LIG: &str = "\
ROOT
ATOM      1  C   LIG A   1       0.000   0.000   0.000  0.00  0.00    +0.000 C
ATOM      2  C   LIG A   1       1.500   0.000   0.000  0.00  0.00    +0.000 C
ENDROOT
BRANCH   2   3
ATOM      3  C   LIG A   1       3.000   0.000   0.000  0.00  0.00    +0.000 C
ATOM      4  O   LIG A   1       3.000   1.200   0.000  0.00  0.00    -0.300 OA
ENDBRANCH   2   3
TORSDOF 1
";

    #[test]
    fn identity_reproduces_reference() {
        let mol = pdbqt::parse_str(LIG).unwrap();
        let lig = LigandModel::new(&mol);
        let conf = Conformation::identity(lig.n_torsions());
        let coords = lig.world_coords(&conf);
        for (c, a) in coords.iter().zip(&mol.atoms) {
            assert!((c - a.pos).norm() < 1e-9, "got {c:?} want {:?}", a.pos);
        }
    }

    #[test]
    fn torsion_rotates_only_subtree() {
        let mol = pdbqt::parse_str(LIG).unwrap();
        let lig = LigandModel::new(&mol);
        let mut conf = Conformation::identity(lig.n_torsions());
        conf.torsions[0] = std::f64::consts::PI; // 180度
        let coords = lig.world_coords(&conf);
        // root 原子 (idx0,1) は不動。
        assert!((coords[0] - Vec3::new(0.0, 0.0, 0.0)).norm() < 1e-9);
        assert!((coords[1] - Vec3::new(1.5, 0.0, 0.0)).norm() < 1e-9);
        // bond 軸上の atom idx2 (3,0,0) は不動。
        assert!((coords[2] - Vec3::new(3.0, 0.0, 0.0)).norm() < 1e-9);
        // 軸(x方向)まわり180度で O(idx3) (3,1.2,0) → (3,-1.2,0)。
        assert!((coords[3] - Vec3::new(3.0, -1.2, 0.0)).norm() < 1e-6, "got {:?}", coords[3]);
    }

    #[test]
    fn intra_pairs_exclude_local_bonds() {
        // フラグメント0={C0,C1}, フラグメント1={C2,O3}。異フラグメント重原子ペアのうち
        // 1-2/1-3 を除いた C0-O3（グラフ距離3）だけが分子内ペアになる。
        let mol = pdbqt::parse_str(LIG).unwrap();
        let lig = LigandModel::new(&mol);
        assert_eq!(lig.n_intra_pairs(), 1);
        // 評価が有限値を返す（panic しない）。
        let e = lig.intramolecular(&lig.scored_atoms(&Conformation::identity(lig.n_torsions())), f64::INFINITY);
        assert!(e.is_finite());
    }

    #[test]
    fn rigid_translation() {
        let mol = pdbqt::parse_str(LIG).unwrap();
        let lig = LigandModel::new(&mol);
        let mut conf = Conformation::identity(lig.n_torsions());
        conf.apply_delta(&[10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let coords = lig.world_coords(&conf);
        for (c, a) in coords.iter().zip(&mol.atoms) {
            assert!((c.x - (a.pos.x + 10.0)).abs() < 1e-9);
        }
    }

    #[test]
    fn world_coords_frames_matches_world_coords() {
        // frames 版の world 座標が world_coords と一致する（ポーズ再構成の同一性）。
        let mol = pdbqt::parse_str(LIG).unwrap();
        let lig = LigandModel::new(&mol);
        let mut conf = Conformation::identity(lig.n_torsions());
        conf.position = Vec3::new(0.7, -0.3, 0.4);
        conf.orientation =
            UnitQuaternion::from_axis_angle(&Unit::new_normalize(Vec3::new(0.2, 1.0, -0.4)), 0.6);
        conf.torsions[0] = 0.5;
        let a = lig.world_coords(&conf);
        let (b, _, _) = lig.world_coords_frames(&conf);
        for (x, y) in a.iter().zip(&b) {
            assert!((x - y).norm() < 1e-12);
        }
    }

    #[test]
    fn analytic_dof_gradient_matches_numerical() {
        // 連鎖則（座標力→DOF 勾配）をスカラ f64 カーネルで厳密検証する。
        // SIMD(f32) との一致は scoring_simd の単体テストで別途検証している。
        use isct_core::scoring::{intermolecular_terms, intermolecular_value_grad, ScoredAtom};

        let mol = pdbqt::parse_str(LIG).unwrap();
        let lig = LigandModel::new(&mol);
        let receptor = vec![
            ScoredAtom { pos: Vec3::new(2.0, 2.0, 0.0), xs: XsType::OxygenA },
            ScoredAtom { pos: Vec3::new(-1.0, -1.0, 1.0), xs: XsType::CarbonH },
            ScoredAtom { pos: Vec3::new(4.0, 0.0, 0.5), xs: XsType::NitrogenD },
        ];

        let mut conf = Conformation::identity(lig.n_torsions());
        conf.position = Vec3::new(0.7, -0.3, 0.4);
        conf.orientation =
            UnitQuaternion::from_axis_angle(&Unit::new_normalize(Vec3::new(0.2, 1.0, -0.4)), 0.6);
        conf.torsions[0] = 0.5;

        // 値（罰則なし: 分子間＋分子内）。
        let value = |c: &Conformation| {
            let a = lig.scored_atoms(c);
            intermolecular_terms(&a, &receptor).weighted_sum() + lig.intramolecular(&a, f64::INFINITY)
        };

        let (wp, wa, wpv) = lig.world_coords_frames(&conf);
        let atoms = lig.scored_atoms_at(&wp);
        let mut force = vec![Vec3::zeros(); atoms.len()];
        let mut e = intermolecular_value_grad(&atoms, &receptor, &mut force);
        e += lig.intramolecular_value_grad(&atoms, &mut force, f64::INFINITY);
        let grad = lig.dof_gradient(&conf, &wp, &wa, &wpv, &force);

        assert!((e - value(&conf)).abs() < 1e-9, "value mismatch {e}");

        let eps = 1e-6;
        for k in 0..conf.dim() {
            let mut d = vec![0.0; conf.dim()];
            d[k] = eps;
            let mut cp = conf.clone();
            cp.apply_delta(&d);
            let fp = value(&cp);
            d[k] = -eps;
            let mut cm = conf.clone();
            cm.apply_delta(&d);
            let fm = value(&cm);
            let num = (fp - fm) / (2.0 * eps);
            assert!(
                (grad[k] - num).abs() < 1e-3,
                "dof {k}: analytic {} numerical {}",
                grad[k],
                num
            );
        }
    }
}
