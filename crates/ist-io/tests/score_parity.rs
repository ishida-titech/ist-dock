//! Vina 互換スコアの解析値による回帰テスト（Vina 非依存の恒久ゲート）。
//!
//! 単純な 2 原子配置のエネルギーを手計算値と照合する。実際の `vina --score_only`
//! との突き合わせは bench/run_vina_compare.py（現実的ポーズで差 < 1e-2）で行う。

use ist_core::scoring::intermolecular_terms;
use ist_io::pdbqt;

/// 指定座標に孤立炭素 1 個を持つ受容体/リガンドの PDBQT を作る。
fn carbon_receptor(x: f64) -> String {
    format!("ATOM      1  C   REC A   1    {x:8.3}   0.000   0.000  0.00  0.00     0.000 C \nTER\n")
}
fn carbon_ligand(x: f64) -> String {
    format!(
        "ROOT\nATOM      1  C   LIG A   1    {x:8.3}   0.000   0.000  0.00  0.00     0.000 C \nENDROOT\nTORSDOF 0\n"
    )
}

fn intermolecular(rec: &str, lig: &str) -> f64 {
    let r = pdbqt::parse_str(rec).unwrap().scored_atoms();
    let l = pdbqt::parse_str(lig).unwrap().scored_atoms();
    intermolecular_terms(&l, &r).weighted_sum()
}

#[test]
fn two_carbons_attractive() {
    // r = 4.0, d = 4.0 - 3.8 = 0.2。gauss1/gauss2/hydrophobic（疎水炭素同士）。
    // 手計算: -0.035579*exp(-0.16) -0.005156*exp(-1.96) -0.035069*1 = -0.066113
    let e = intermolecular(&carbon_receptor(0.0), &carbon_ligand(4.0));
    assert!((e - (-0.066113)).abs() < 1e-5, "got {e}");
}

#[test]
fn two_carbons_clash_repulsion() {
    // r = 3.0, d = -0.8。repulsion = 0.64 が支配的。
    // 手計算: -0.035579*exp(-2.56) -0.005156*exp(-3.61) +0.840245*0.64 -0.035069*1 = 0.499799
    let e = intermolecular(&carbon_receptor(0.0), &carbon_ligand(3.0));
    assert!((e - 0.499799).abs() < 1e-5, "got {e}");
}

#[test]
fn beyond_cutoff_is_zero() {
    // r = 8.0 はカットオフ以上 → 寄与なし。
    let e = intermolecular(&carbon_receptor(0.0), &carbon_ligand(8.0));
    assert_eq!(e, 0.0);
}
