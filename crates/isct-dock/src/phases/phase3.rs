//! Phase 3 — R 基の最適化と Delta エネルギー評価（M3）。
//!
//! コアをロックした子分子の **R 基 torsion のみ**を L-BFGS（有限差分）で最適化する。
//! コア–受容体の分子間エネルギーは `DeltaScorer` にキャッシュし、各ステップでは R 基由来の
//! 項のみ再計算する（仕様の増分=Delta 評価）。Soft-core ポテンシャルと柔軟側鎖同時最適化は
//! M6 で追加する。

use std::collections::HashSet;

use isct_core::delta_energy::DeltaScorer;
use isct_core::scoring::ScoredAtom;

use crate::energy::Scorer;
use crate::ligand::{Conformation, LigandModel};
use crate::optimizer::{minimize_free_fd, torsions_only};

/// 子分子の最適化結果。
pub struct ChildScore {
    /// 最終ポーズ。
    pub conf: Conformation,
    /// 分子間エネルギー（kcal/mol、報告用フル計算）。
    pub e_inter: f64,
    /// 分子内エネルギー（kcal/mol）。
    pub e_intra: f64,
}

/// コア固定で R 基 torsion を最適化し、最終ポーズとエネルギーを返す。
///
/// `core_atoms` は子の MCS コア原子（**全原子インデックス**集合）。コアを動かす torsion
/// （部分木にコアを含むもの）は自動的にロックされ、R 基 torsion のみ最適化する。コアは
/// 最適化中に動かないため、コア–受容体エネルギーは 1 回だけ計算してキャッシュする。
pub fn optimize_child(
    child: &LigandModel,
    scorer: &Scorer,
    start: Conformation,
    core_atoms: &HashSet<usize>,
    local_iters: usize,
) -> ChildScore {
    let receptor = scorer.receptor();

    // scored 原子順のコア/R 基マスク（all-atom idx がコア集合にあれば true）。
    let core_mask: Vec<bool> = child
        .scored_all_indices()
        .iter()
        .map(|i| core_atoms.contains(i))
        .collect();

    // コア（固定）scored 原子を start ポーズ位置で取得し、コア–受容体エネルギーをキャッシュ。
    let start_atoms = child.scored_atoms(&start);
    let core: Vec<ScoredAtom> = start_atoms
        .iter()
        .zip(&core_mask)
        .filter_map(|(a, &m)| m.then_some(*a))
        .collect();
    let delta = DeltaScorer::new(&core, receptor);

    // 目的関数 = Δ分子間（コアキャッシュ＋R基–受容体）＋ ボックス罰則 ＋ 分子内。
    let objective = |c: &Conformation| {
        let atoms = child.scored_atoms(c);
        let rgroup: Vec<ScoredAtom> = atoms
            .iter()
            .zip(&core_mask)
            .filter_map(|(a, &m)| (!m).then_some(*a))
            .collect();
        delta.intermolecular(&rgroup, receptor) + scorer.box_penalty(&atoms) + child.intramolecular(&atoms, f64::INFINITY)
    };

    // R 基 torsion（コアを動かさない torsion）のみ最適化（Δ目的関数は有限差分勾配）。
    let free = child.rgroup_torsions(core_atoms);
    let res = minimize_free_fd(start, &objective, local_iters, &torsions_only(&free));

    // 報告用に最終ポーズで分子間・分子内をフル計算する。
    let final_atoms = child.scored_atoms(&res.conf);
    let e_inter = scorer.intermolecular(&final_atoms);
    let e_intra = child.intramolecular(&final_atoms, f64::INFINITY);
    ChildScore {
        conf: res.conf,
        e_inter,
        e_intra,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use isct_core::geometry::Vec3;
    use isct_core::scoring::ScoredAtom;
    use isct_io::pdbqt;

    const CHILD: &str = "\
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
    fn core_atoms_stay_fixed_during_optimization() {
        let mol = pdbqt::parse_str(CHILD).unwrap();
        let child = LigandModel::new(&mol);
        // 受容体: ポケット相当の原子を数個。
        let receptor_full = vec![
            ScoredAtom { pos: Vec3::new(3.0, 3.0, 0.0), xs: isct_core::atom_types::XsType::NitrogenD },
            ScoredAtom { pos: Vec3::new(0.0, -3.0, 0.0), xs: isct_core::atom_types::XsType::CarbonH },
        ];
        let scorer = Scorer::new(&receptor_full, Vec3::zeros(), Vec3::new(20.0, 20.0, 20.0));

        // コア: root={C0,C1}（全原子 idx）。
        let core_set: HashSet<usize> = [0usize, 1].into_iter().collect();

        let start = Conformation::identity(child.n_torsions());
        let start_world = child.world_coords(&start);

        let result = optimize_child(&child, &scorer, start, &core_set, 40);
        let end_world = child.world_coords(&result.conf);

        // コア原子（root の C0,C1）は不動。
        for &i in &core_set {
            assert!(
                (start_world[i] - end_world[i]).norm() < 1e-9,
                "core atom {i} moved"
            );
        }
        assert!(result.e_inter.is_finite() && result.e_intra.is_finite());
    }
}
