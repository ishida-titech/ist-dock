//! 探索 1 評価の内訳プロファイル: 各ホット関数の ns/call を計る（SIMD 化対象の特定用）。
//! 使い方: cargo run --release --example profile_eval -- <receptor.pdbqt> <ligand.pdbqt> cx cy cz
use std::time::Instant;

use isct_core::geometry::Vec3;
use isct_dock::energy::Scorer;
use isct_dock::ligand::LigandModel;
use isct_io::pdbqt;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let rec = pdbqt::parse_file(&a[1]).unwrap();
    let lig_mol = pdbqt::parse_str(&std::fs::read_to_string(&a[2]).unwrap()).unwrap();
    let center = Vec3::new(a[3].parse().unwrap(), a[4].parse().unwrap(), a[5].parse().unwrap());
    let size = Vec3::new(20.0, 20.0, 20.0);
    let rec_scored = rec.scored_atoms();
    let lig = LigandModel::new(&lig_mol);
    let types: Vec<_> = lig_mol.scored_atoms().iter().map(|x| x.xs).collect();
    let scorer = Scorer::new_grid(&rec_scored, center, size, &types, isct_dock::energy::GRID_MARGIN);

    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(7);
    use rand::SeedableRng;
    let conf = lig.random_in_box(&mut rng, center, size);

    let n = 200_000;
    let bench = |name: &str, mut f: Box<dyn FnMut()>| {
        f(); // warm
        let t = Instant::now();
        for _ in 0..n {
            f();
        }
        let ns = t.elapsed().as_nanos() as f64 / n as f64;
        println!("{name:32} {ns:8.1} ns/call");
    };

    println!("ligand: {} scored atoms, {} torsions, receptor {} pruned",
        lig.n_scored(), lig.n_torsions(), scorer.receptor_atoms());

    // 値（直線探索）パス
    bench("search_score(grid value)", {
        let lig = &lig; let scorer = &scorer; let conf = conf.clone();
        Box::new(move || {
            let atoms = lig.scored_atoms(&conf);
            std::hint::black_box(scorer.search_score(&atoms, f64::INFINITY));
        })
    });
    // 勾配パス
    bench("value_and_force(grid grad)", {
        let lig = &lig; let scorer = &scorer; let conf = conf.clone();
        Box::new(move || {
            let atoms = lig.scored_atoms(&conf);
            let mut force = vec![Vec3::zeros(); atoms.len()];
            std::hint::black_box(scorer.value_and_force(&atoms, &mut force, f64::INFINITY));
        })
    });
    bench("world_coords_frames", {
        let lig = &lig; let conf = conf.clone();
        Box::new(move || { std::hint::black_box(lig.world_coords_frames(&conf)); })
    });
    bench("scored_atoms(=world_coords+extract)", {
        let lig = &lig; let conf = conf.clone();
        Box::new(move || { std::hint::black_box(lig.scored_atoms(&conf)); })
    });
    bench("intramolecular_value_grad", {
        let lig = &lig; let conf = conf.clone();
        Box::new(move || {
            let atoms = lig.scored_atoms(&conf);
            let mut force = vec![Vec3::zeros(); atoms.len()];
            std::hint::black_box(lig.intramolecular_value_grad(&atoms, &mut force, f64::INFINITY));
        })
    });
    bench("dof_gradient", {
        let lig = &lig; let conf = conf.clone();
        Box::new(move || {
            let (wp, wa, wpiv) = lig.world_coords_frames(&conf);
            let atoms = lig.scored_atoms_at(&wp);
            let force = vec![Vec3::zeros(); atoms.len()];
            std::hint::black_box(lig.dof_gradient(&conf, &wp, &wa, &wpiv, &force));
        })
    });
}
