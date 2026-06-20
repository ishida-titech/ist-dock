//! スクリーニングの grid 再利用の効果測定: N クラスタ（同一リガンドを複製）を
//! 共有スコアラ（pocket=None, grid 1 回構築）vs per-cluster（pocket=既定ボックス, grid N 回構築）で
//! ドッキングし壁時間を比較する。
//! 使い方: cargo run --release --example screen_grid_timing -- <receptor.pdbqt> <ligand.pdbqt> cx cy cz [N]
use std::time::Instant;

use isct_core::geometry::Vec3;
use isct_dock::screen::{screen, ScreenParams};
use isct_dock::search::SearchParams;
use isct_io::library::{Cluster, Library, MolEntry, Pocket};
use isct_io::pdbqt;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let rec = pdbqt::parse_file(&a[1]).unwrap();
    let lig_text = std::fs::read_to_string(&a[2]).unwrap();
    let lig = pdbqt::parse_str(&lig_text).unwrap();
    let center = Vec3::new(a[3].parse().unwrap(), a[4].parse().unwrap(), a[5].parse().unwrap());
    let size = Vec3::new(20.0, 20.0, 20.0);
    let n: usize = a.get(6).map(|s| s.parse().unwrap()).unwrap_or(20);
    let rec_scored = rec.scored_atoms();

    let mk_lib = |pocket: Option<Pocket>| Library {
        clusters: (0..n)
            .map(|i| Cluster {
                id: format!("c{i}"),
                pocket: pocket.clone(),
                seed: MolEntry {
                    compound_id: format!("cpd{i}"),
                    molecule: pdbqt::parse_str(&lig_text).unwrap(),
                    core_atoms: vec![],
                    source_text: lig_text.clone(),
                },
                children: vec![],
            })
            .collect(),
    };

    let search = SearchParams { n_runs: 4, ..SearchParams::default() };
    let params = ScreenParams {
        center,
        size,
        search,
        prune_threshold: 1e9, // 棄却しない（全 seed をドッキング）
        local_iters: 50,
        use_grid: true,
    };

    println!("receptor {} scored atoms, ligand {} atoms, N={} clusters",
        rec_scored.len(), lig.scored_atoms().len(), n);

    // per-cluster（pocket=既定ボックス → grid を N 回構築）
    let local_lib = mk_lib(Some(Pocket { center: [center.x, center.y, center.z], size: [size.x, size.y, size.z] }));
    let t = Instant::now();
    let r_local = screen(&rec_scored, &local_lib, &params);
    let t_local = t.elapsed().as_secs_f64();

    // 共有（pocket=None → grid を 1 回構築）
    let shared_lib = mk_lib(None);
    let t = Instant::now();
    let r_shared = screen(&rec_scored, &shared_lib, &params);
    let t_shared = t.elapsed().as_secs_f64();

    println!("per-cluster (grid x{n}): {t_local:.2}s");
    println!("shared      (grid x1 ): {t_shared:.2}s");
    println!("speedup: {:.2}x", t_local / t_shared);
    // 結果一致の簡易確認
    let ok = r_local.entries.len() == r_shared.entries.len()
        && r_local.entries.iter().zip(&r_shared.entries)
            .all(|(a, b)| a.total_score.to_bits() == b.total_score.to_bits());
    println!("entries identical: {ok}");
}
