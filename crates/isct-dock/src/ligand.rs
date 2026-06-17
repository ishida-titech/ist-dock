//! リガンドのトポロジー表現とポーズ再構成（M2）。
//!
//! PDBQT の torsion 木（フラグメント）から、剛体並進(3)＋回転(四元数)＋torsion(n) を
//! 自由度とするポーズ再構成を行う。参照座標（入力ポーズ）を基準に、torsion を木順
//! （親→子）で適用してから剛体変換を掛ける。恒等変換は入力ポーズを再現する。

use std::collections::HashSet;

use isct_core::atom_types::XsType;
use isct_core::geometry::{distance, Vec3};
use isct_core::scoring::{eval_pair, ScoredAtom, CUTOFF};
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

        LigandModel {
            ref_coords,
            bonds,
            subtree,
            root_pivot,
            scored,
            intra_pairs,
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
    pub fn intramolecular(&self, atoms: &[ScoredAtom]) -> f64 {
        let mut e = 0.0;
        for &(a, b) in &self.intra_pairs {
            let r = distance(&atoms[a].pos, &atoms[b].pos);
            if r < CUTOFF {
                e += eval_pair(atoms[a].xs, atoms[b].xs, r).weighted_sum();
            }
        }
        e
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

    /// 与えられた conformation の全原子 world 座標を再構成する。
    pub fn world_coords(&self, conf: &Conformation) -> Vec<Vec3> {
        let mut pos = self.ref_coords.clone();

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
        pos
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
        let e = lig.intramolecular(&lig.scored_atoms(&Conformation::identity(lig.n_torsions())));
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
}
