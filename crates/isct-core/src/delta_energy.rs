//! 増分（Delta）エネルギー評価（M3）。
//!
//! 子分子のコア（Scaffold、固定）と R 基（可動）を分けて評価する。コアは Phase 2 で
//! docked seed ポーズへ重ねたあと **動かない**ため、コア–受容体の分子間エネルギーは
//! 1 回だけ計算してキャッシュし、R 基 torsion 最適化の各ステップでは **R 基–受容体の項
//! のみ再計算**する。これが仕様の「増分（Delta）エネルギー評価」の中核。
//!
//! 注: コア原子は同クラスタ内で同じ world 座標に置かれるが、付着炭素の XS 型が R 基に
//! よって変わりうるため、コア–受容体エネルギーは **子ごとに 1 回**計算する（子内の全
//! 最適化ステップで使い回す）。これにより正確さを保ったまま主要コストを削減する。
//! クラスタ全子での完全共有は近似となるため M3 では採らない。

use crate::scoring::{intermolecular_terms, ScoredAtom};

/// コア固定・R 基可動の子分子に対する増分スコアラ。
///
/// `new` でコア–受容体エネルギーをキャッシュし、`intermolecular` で
/// `E_inter(core,recep)[cache] + E_inter(rgroup,recep)` を返す。
#[derive(Debug, Clone)]
pub struct DeltaScorer {
    /// キャッシュ済みのコア–受容体分子間エネルギー（kcal/mol、重み付き）。
    core_receptor_energy: f64,
}

impl DeltaScorer {
    /// 固定コア原子（docked world 座標）と受容体から、コア–受容体エネルギーを 1 回計算する。
    pub fn new(core: &[ScoredAtom], receptor: &[ScoredAtom]) -> Self {
        DeltaScorer {
            core_receptor_energy: intermolecular_terms(core, receptor).weighted_sum(),
        }
    }

    /// キャッシュ済みのコア–受容体エネルギー。
    pub fn core_receptor_energy(&self) -> f64 {
        self.core_receptor_energy
    }

    /// 子分子の分子間エネルギー = コア（キャッシュ）＋ R 基–受容体（毎回計算）。
    pub fn intermolecular(&self, rgroup: &[ScoredAtom], receptor: &[ScoredAtom]) -> f64 {
        self.core_receptor_energy + intermolecular_terms(rgroup, receptor).weighted_sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atom_types::XsType;
    use crate::geometry::Vec3;

    fn atom(x: f64, y: f64, z: f64, xs: XsType) -> ScoredAtom {
        ScoredAtom {
            pos: Vec3::new(x, y, z),
            xs,
        }
    }

    /// 増分評価（コアキャッシュ＋R基）が、コア＋R基を結合した全原子のフル再計算と一致する。
    #[test]
    fn delta_equals_full_recompute() {
        let receptor = vec![
            atom(0.0, 0.0, 0.0, XsType::OxygenA),
            atom(2.0, 1.0, 0.0, XsType::CarbonH),
            atom(-1.0, 3.0, 1.0, XsType::NitrogenD),
            atom(4.0, -2.0, 2.0, XsType::CarbonP),
        ];
        let core = vec![
            atom(1.0, 0.5, 0.0, XsType::CarbonH),
            atom(1.5, 1.5, 0.5, XsType::CarbonP),
        ];
        let rgroup = vec![
            atom(3.0, 0.0, 1.0, XsType::OxygenD),
            atom(2.5, -1.0, 0.5, XsType::NitrogenA),
        ];

        let delta = DeltaScorer::new(&core, &receptor);
        let delta_e = delta.intermolecular(&rgroup, &receptor);

        let mut full_atoms = core.clone();
        full_atoms.extend_from_slice(&rgroup);
        let full_e = intermolecular_terms(&full_atoms, &receptor).weighted_sum();

        assert!(
            (delta_e - full_e).abs() < 1e-9,
            "delta {delta_e} vs full {full_e}"
        );
    }

    #[test]
    fn empty_rgroup_returns_core_only() {
        let receptor = vec![atom(0.0, 0.0, 0.0, XsType::OxygenA)];
        let core = vec![atom(2.0, 0.0, 0.0, XsType::CarbonP)];
        let delta = DeltaScorer::new(&core, &receptor);
        assert!((delta.intermolecular(&[], &receptor) - delta.core_receptor_energy()).abs() < 1e-12);
    }
}
