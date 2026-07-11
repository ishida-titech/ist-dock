//! クラスタライブラリ I/O（M3、可読 TOML マニフェスト版）。
//!
//! 既存ツール（RDKit/Meeko + `tools/preprocess/` の変換スクリプト）が生成する
//! クラスタライブラリを読む。M3 ではデバッグ容易性を優先し、**PDBQT 群 ＋ TOML
//! マニフェスト**で受け渡す（仕様 4.1 のカスタムバイナリ化と memmap2 ストリーミングは
//! M4 でまとめて実装）。
//!
//! マニフェスト・スキーマ:
//! ```toml
//! [[cluster]]
//! id = "ampc_c0"
//! pocket_center = [x, y, z]   # 任意（省略時は run config の box）
//! pocket_size   = [sx, sy, sz]  # 任意
//!
//! [cluster.seed]
//! compound_id = "..."
//! pdbqt = "seed.pdbqt"          # マニフェストからの相対パス
//! core_atoms = [0, 1, 2, ...]   # seed 側 MCS コア原子（全原子 idx、対応の基準順）
//!
//! [[cluster.child]]
//! compound_id = "..."
//! pdbqt = "child_0.pdbqt"       # コア=ROOT でルート化済み
//! core_atoms = [0, 1, 2, ...]   # 子側 MCS コア原子（seed.core_atoms と 1:1 同順）
//! ```
//!
//! 不変条件: 各子の `core_atoms.len()` は seed の `core_atoms.len()` に等しく、i 番目どうしが
//! 対応する（Phase 2 の剛体重ね合わせの点対応に使う）。コア原子は剛体 root 内にあること
//! （torsion で動かない）を前提とする。

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::pdbqt::{self, Molecule};
use crate::IoError;

/// ポケット拘束（探索ボックス）。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pocket {
    pub center: [f64; 3],
    pub size: [f64; 3],
}

/// ライブラリ中の 1 分子（seed または子）。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MolEntry {
    pub compound_id: String,
    /// パース済み分子（torsion 木を含む）。
    pub molecule: Molecule,
    /// MCS コア原子（全原子 idx）。seed と子で同順・同数。
    pub core_atoms: Vec<usize>,
    /// 元 PDBQT テキスト（ポーズ書き出し `pdbqt::write_pose` 用に保持）。
    pub source_text: String,
}

/// 1 クラスタ（代表 seed ＋ 子分子群）。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Cluster {
    pub id: String,
    pub pocket: Option<Pocket>,
    pub seed: MolEntry,
    pub children: Vec<MolEntry>,
}

/// クラスタライブラリ全体。
#[derive(Debug, Clone)]
pub struct Library {
    pub clusters: Vec<Cluster>,
}

impl Library {
    /// 子分子の総数（全クラスタ合計）。
    pub fn n_children(&self) -> usize {
        self.clusters.iter().map(|c| c.children.len()).sum()
    }
}

// --- serde 生スキーマ（パス・配列をそのまま受ける）---

#[derive(Deserialize)]
struct RawLibrary {
    #[serde(default, rename = "cluster")]
    clusters: Vec<RawCluster>,
}

#[derive(Deserialize)]
struct RawCluster {
    id: String,
    pocket_center: Option<[f64; 3]>,
    pocket_size: Option<[f64; 3]>,
    seed: RawMol,
    #[serde(default, rename = "child")]
    children: Vec<RawMol>,
}

#[derive(Deserialize)]
struct RawMol {
    compound_id: String,
    pdbqt: String,
    core_atoms: Vec<usize>,
}

/// マニフェスト（TOML）を読み、参照 PDBQT を解決してライブラリを構築する。
/// スキップされたクラスタ（リガンド）の記録。screen で解釈不能な原子・不正 PDBQT を含むクラスタを
/// 全体停止せずスキップした際に、その理由を残す（サマリ表示・`skipped.tsv` 用）。
#[derive(Debug, Clone)]
pub struct SkipRecord {
    pub cluster_id: String,
    pub compound_id: String,
    pub reason: String,
}

/// マニフェストを厳密ロードする（1 件でも解決失敗なら全体を Err）。単発検証・pack の既定に使う。
pub fn load(path: impl AsRef<Path>) -> Result<Library, IoError> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path)?;
    let raw: RawLibrary = toml::from_str(&text)?;
    let base = path.parent().unwrap_or_else(|| Path::new("."));

    let mut clusters = Vec::with_capacity(raw.clusters.len());
    for rc in raw.clusters {
        clusters.push(build_cluster(base, rc)?);
    }
    Ok(Library { clusters })
}

/// マニフェストを**寛容ロード**する（解釈不能なクラスタはスキップして継続）。ファイル読込・TOML 構文の
/// 失敗は致命（Err）だが、個々のクラスタの解決失敗（未知原子・不正 PDBQT・パス欠落・構造不整合）は
/// `SkipRecord` に記録して残りを続行する。10M スクリーニングで 1 リガンドが全体を落とさないための経路。
pub fn load_lenient(path: impl AsRef<Path>) -> Result<(Library, Vec<SkipRecord>), IoError> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path)?;
    let raw: RawLibrary = toml::from_str(&text)?;
    let base = path.parent().unwrap_or_else(|| Path::new("."));

    let mut clusters = Vec::with_capacity(raw.clusters.len());
    let mut skipped = Vec::new();
    for rc in raw.clusters {
        let cluster_id = rc.id.clone();
        let compound_id = rc.seed.compound_id.clone();
        match build_cluster(base, rc) {
            Ok(c) => clusters.push(c),
            Err(e) => skipped.push(SkipRecord {
                cluster_id,
                compound_id,
                reason: e.to_string(),
            }),
        }
    }
    Ok((Library { clusters }, skipped))
}

/// 1 クラスタ（seed＋子）を生の manifest エントリから構築する。`load`（厳密）と `load_lenient`（寛容）で共有。
fn build_cluster(base: &Path, rc: RawCluster) -> Result<Cluster, IoError> {
    let pocket = match (rc.pocket_center, rc.pocket_size) {
        (Some(center), Some(size)) => Some(Pocket { center, size }),
        (None, None) => None,
        _ => {
            return Err(IoError::Library(format!(
                "cluster {}: pocket_center と pocket_size は両方指定するか両方省略する",
                rc.id
            )))
        }
    };

    let seed = resolve_mol(base, rc.seed, &rc.id, "seed")?;
    let n_core = seed.core_atoms.len();

    // 子を持つクラスタは seed コアが非空でなければならない（子がそこへ重なるため）。
    // 子のいない seed のみクラスタは空コア＝全自由度ドッキングで正常。
    if !rc.children.is_empty() && n_core == 0 {
        return Err(IoError::Library(format!(
            "cluster {} seed {}: 子があるのに core_atoms が空",
            rc.id, seed.compound_id
        )));
    }

    let mut children = Vec::with_capacity(rc.children.len());
    for raw_child in rc.children {
        let child = resolve_mol(base, raw_child, &rc.id, "child")?;
        if child.core_atoms.len() != n_core {
            return Err(IoError::Library(format!(
                "cluster {}: 子 {} の core_atoms 数 {} が seed の {} と不一致",
                rc.id,
                child.compound_id,
                child.core_atoms.len(),
                n_core
            )));
        }
        children.push(child);
    }

    Ok(Cluster {
        id: rc.id,
        pocket,
        seed,
        children,
    })
}

/// 1 分子エントリを解決（PDBQT を読み・パースし、core_atoms 範囲を検証）。
fn resolve_mol(base: &Path, raw: RawMol, cluster_id: &str, role: &str) -> Result<MolEntry, IoError> {
    let pdbqt_path = base.join(&raw.pdbqt);
    let source_text = std::fs::read_to_string(&pdbqt_path).map_err(|e| {
        IoError::Library(format!(
            "cluster {cluster_id} {role} {}: PDBQT 読込失敗 {}: {e}",
            raw.compound_id,
            pdbqt_path.display()
        ))
    })?;
    let molecule = pdbqt::parse_str(&source_text)?;

    let n_atoms = molecule.atoms.len();
    for &idx in &raw.core_atoms {
        if idx >= n_atoms {
            return Err(IoError::Library(format!(
                "cluster {cluster_id} {role} {}: core_atoms idx {idx} が原子数 {n_atoms} を超過",
                raw.compound_id
            )));
        }
    }
    // 空コアの可否は呼び出し側で判定する（子のいない seed のみクラスタ＝全自由度ドッキングでは空が正）。

    Ok(MolEntry {
        compound_id: raw.compound_id,
        molecule,
        core_atoms: raw.core_atoms,
        source_text,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    const SEED: &str = "\
ROOT
ATOM      1  C   LIG A   1       0.000   0.000   0.000  0.00  0.00    +0.000 C
ATOM      2  C   LIG A   1       1.500   0.000   0.000  0.00  0.00    +0.000 C
ENDROOT
TORSDOF 0
";
    const CHILD: &str = "\
ROOT
ATOM      1  C   LIG A   1       0.000   0.000   0.000  0.00  0.00    +0.000 C
ATOM      2  C   LIG A   1       1.500   0.000   0.000  0.00  0.00    +0.000 C
ENDROOT
BRANCH   2   3
ATOM      3  O   LIG A   1       3.000   0.000   0.000  0.00  0.00    -0.300 OA
ENDBRANCH   2   3
TORSDOF 1
";

    fn write(dir: &Path, name: &str, body: &str) {
        let mut f = std::fs::File::create(dir.join(name)).unwrap();
        f.write_all(body.as_bytes()).unwrap();
    }

    #[test]
    fn loads_manifest_and_resolves_pdbqt() {
        let dir = std::env::temp_dir().join(format!("isct_lib_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        write(&dir, "seed.pdbqt", SEED);
        write(&dir, "child0.pdbqt", CHILD);
        let manifest = "\
[[cluster]]
id = \"c0\"
pocket_center = [1.0, 2.0, 3.0]
pocket_size = [20.0, 20.0, 20.0]

[cluster.seed]
compound_id = \"seed0\"
pdbqt = \"seed.pdbqt\"
core_atoms = [0, 1]

[[cluster.child]]
compound_id = \"child0\"
pdbqt = \"child0.pdbqt\"
core_atoms = [0, 1]
";
        write(&dir, "library.toml", manifest);

        let lib = load(dir.join("library.toml")).unwrap();
        assert_eq!(lib.clusters.len(), 1);
        assert_eq!(lib.n_children(), 1);
        let c = &lib.clusters[0];
        assert_eq!(c.id, "c0");
        assert_eq!(c.pocket.as_ref().unwrap().center, [1.0, 2.0, 3.0]);
        assert_eq!(c.seed.core_atoms, vec![0, 1]);
        assert_eq!(c.seed.molecule.atoms.len(), 2);
        assert_eq!(c.children[0].molecule.atoms.len(), 3);
        assert_eq!(c.children[0].compound_id, "child0");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn lenient_load_skips_bad_ligand_and_continues() {
        let dir = std::env::temp_dir().join(format!("isct_lib_lenient_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        write(&dir, "good.pdbqt", SEED);
        // 未知元素 Xx を含む壊れたリガンド（フォールバックでも救えない）。
        let bad = SEED.replace(" C\n", " Xx\n");
        write(&dir, "bad.pdbqt", &bad);
        let manifest = "\
[[cluster]]
id = \"good\"
[cluster.seed]
compound_id = \"g0\"
pdbqt = \"good.pdbqt\"
core_atoms = []

[[cluster]]
id = \"bad\"
[cluster.seed]
compound_id = \"b0\"
pdbqt = \"bad.pdbqt\"
core_atoms = []
";
        write(&dir, "library.toml", manifest);
        let path = dir.join("library.toml");

        // 厳密ロードは壊れたリガンドで全体 Err。
        assert!(load(&path).is_err());

        // 寛容ロードは good を残し bad をスキップ＋記録して継続。
        let (lib, skipped) = load_lenient(&path).unwrap();
        assert_eq!(lib.clusters.len(), 1);
        assert_eq!(lib.clusters[0].id, "good");
        assert_eq!(skipped.len(), 1);
        assert_eq!(skipped[0].cluster_id, "bad");
        assert_eq!(skipped[0].compound_id, "b0");
        assert!(skipped[0].reason.contains("Xx"));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn rejects_core_atom_count_mismatch() {
        let dir = std::env::temp_dir().join(format!("isct_lib_test_mm_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        write(&dir, "seed.pdbqt", SEED);
        write(&dir, "child0.pdbqt", CHILD);
        let manifest = "\
[[cluster]]
id = \"c0\"

[cluster.seed]
compound_id = \"seed0\"
pdbqt = \"seed.pdbqt\"
core_atoms = [0, 1]

[[cluster.child]]
compound_id = \"child0\"
pdbqt = \"child0.pdbqt\"
core_atoms = [0]
";
        write(&dir, "library.toml", manifest);
        let err = load(dir.join("library.toml")).unwrap_err();
        assert!(matches!(err, IoError::Library(_)));
        std::fs::remove_dir_all(&dir).ok();
    }
}
