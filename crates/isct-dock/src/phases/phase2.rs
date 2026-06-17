//! Phase 2 — 子分子への座標流用（核固定配置）（M3）。
//!
//! 子分子の共通コア（MCS）を、Phase 1 で確定した docked seed のコア world 座標へ剛体
//! 重ね合わせ（Kabsch）し、コアをロックした初期 `Conformation` を作る。コア＝root
//! フラグメント（剛体）なので、得た剛体変換を `LigandModel::world_coords` と整合する
//! position/orientation に変換すればよい。R 基 torsion は参照（embed 値）= 0 で初期化し、
//! Phase 3 で最適化する。
//!
//! 動的バウンディングボックス（PCA 身の丈窓）は M5。ここでは固定/単純配置のみ。

use isct_core::geometry::{superpose, Vec3};

use crate::ligand::{Conformation, LigandModel};

/// 子のコアを docked seed コアへ重ね合わせ、コア固定の初期 conformation を返す。
///
/// - `child`: 子分子モデル（コア＝root 前提）。
/// - `child_core_ref`: 子の MCS コア原子の **参照（入力ポーズ）world 座標**。
/// - `seed_core_world`: docked seed の対応コア原子 world 座標（同順）。
///
/// `world_coords` は `p' = pivot + orientation·(p − pivot) + position`。剛体一致
/// `p' = R·p + t` には `orientation = R`, `position = t + R·pivot − pivot`。
pub fn place_child(
    child: &LigandModel,
    child_core_ref: &[Vec3],
    seed_core_world: &[Vec3],
) -> Conformation {
    let (rot, t) = superpose(child_core_ref, seed_core_world);
    let pivot = child.root_pivot();
    let position = t + rot * pivot - pivot;
    Conformation {
        position,
        orientation: rot,
        torsions: vec![0.0; child.n_torsions()],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use isct_io::pdbqt;

    // コア=root の子について、コアの参照座標を任意の剛体変換で動かした目標に重ねると、
    // place_child が返す conformation でコア原子がその目標に一致する。
    #[test]
    fn places_core_onto_target() {
        const CHILD: &str = "\
ROOT
ATOM      1  C   LIG A   1       0.000   0.000   0.000  0.00  0.00    +0.000 C
ATOM      2  C   LIG A   1       1.500   0.000   0.000  0.00  0.00    +0.000 C
ATOM      3  C   LIG A   1       0.000   1.500   0.000  0.00  0.00    +0.000 C
ENDROOT
BRANCH   3   4
ATOM      4  O   LIG A   1       0.000   3.000   0.000  0.00  0.00    -0.300 OA
ENDBRANCH   3   4
TORSDOF 1
";
        let mol = pdbqt::parse_str(CHILD).unwrap();
        let child = LigandModel::new(&mol);
        let core_idx = [0usize, 1, 2];
        let child_core_ref: Vec<Vec3> = core_idx.iter().map(|&i| child.ref_coords()[i]).collect();

        // 目標 = 既知の剛体変換をコア参照座標へ適用。
        let axis = nalgebra::Unit::new_normalize(Vec3::new(0.2, 1.0, -0.3));
        let rot = nalgebra::UnitQuaternion::from_axis_angle(&axis, 0.7);
        let trans = Vec3::new(4.0, -1.0, 2.0);
        let seed_core_world: Vec<Vec3> =
            child_core_ref.iter().map(|p| rot * p + trans).collect();

        let conf = place_child(&child, &child_core_ref, &seed_core_world);
        let world = child.world_coords(&conf);
        for (k, &i) in core_idx.iter().enumerate() {
            assert!(
                (world[i] - seed_core_world[k]).norm() < 1e-9,
                "core atom {i}: got {:?} want {:?}",
                world[i],
                seed_core_world[k]
            );
        }
    }
}
