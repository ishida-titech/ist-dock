//! バイナリ・クラスタライブラリ（M4 streaming）。
//!
//! TOML マニフェスト＋PDBQT（[`crate::library`]）をパース済み構造のまま **serde + bincode** で 1 ファイルに
//! パックし、`memmap2` でマップして **クラスタ単位に遅延デコード**する（全 RAM ロードを避ける＝10M 規模向け）。
//! 共有 grid 構築に必要な `has_default_box` / XS 型 union はパック時に前計算してヘッダへ持つ
//! （streaming 時にフルデコード走査をしないため）。
//!
//! レイアウト: `[u64 header_len(LE)] [bincode(Header)] [cluster record ...]`。
//! 各 record は `bincode(Cluster)`。Header の `offsets[i] = (ファイル先頭からの offset, byte 長)`。

use std::path::Path;

use isct_core::atom_types::XsType;
use memmap2::Mmap;
use serde::{Deserialize, Serialize};

use crate::library::{Cluster, Library};
use crate::IoError;

const MAGIC: [u8; 8] = *b"ISCTLIB1";

#[derive(Serialize, Deserialize)]
struct Header {
    magic: [u8; 8],
    version: u32,
    n_clusters: u64,
    /// pocket 未指定（既定ボックス）クラスタが存在するか。共有スコアラ構築の要否判定。
    has_default_box: bool,
    /// 既定ボックスクラスタの seed＋子の XS 型 union（共有 grid マップ生成用、前計算）。
    xs_union: Vec<XsType>,
    /// 各クラスタ record の (ファイル offset, byte 長)。
    offsets: Vec<(u64, u64)>,
}

fn bin_err(ctx: &str, e: bincode::Error) -> IoError {
    IoError::Library(format!("binary library {ctx}: {e}"))
}

/// 既定ボックス（pocket 未指定）クラスタの seed＋子の XS 型 union。
fn default_box_xs_union(lib: &Library) -> Vec<XsType> {
    let mut set = std::collections::HashSet::new();
    for c in lib.clusters.iter().filter(|c| c.pocket.is_none()) {
        for a in c.seed.molecule.scored_atoms() {
            set.insert(a.xs);
        }
        for child in &c.children {
            for a in child.molecule.scored_atoms() {
                set.insert(a.xs);
            }
        }
    }
    set.into_iter().collect()
}

/// `Library` をバイナリ・ライブラリへ書き出す。
pub fn pack(lib: &Library, out: impl AsRef<Path>) -> Result<(), IoError> {
    // 各クラスタを bincode 化。
    let records: Vec<Vec<u8>> = lib
        .clusters
        .iter()
        .map(|c| bincode::serialize(c).map_err(|e| bin_err("serialize cluster", e)))
        .collect::<Result<_, _>>()?;

    let mut header = Header {
        magic: MAGIC,
        version: 1,
        n_clusters: lib.clusters.len() as u64,
        has_default_box: lib.clusters.iter().any(|c| c.pocket.is_none()),
        xs_union: default_box_xs_union(lib),
        offsets: vec![(0, 0); records.len()],
    };
    // Header のサイズは offsets の **値**に依らず一定（固定長 int 符号化）なので、placeholder で長さを得てから
    // データ開始位置を確定し offsets を埋める（2 パス）。
    let header_len = bincode::serialized_size(&header).map_err(|e| bin_err("header size", e))?;
    let data_start = 8 + header_len;
    let mut off = data_start;
    for (i, r) in records.iter().enumerate() {
        header.offsets[i] = (off, r.len() as u64);
        off += r.len() as u64;
    }
    let header_bytes = bincode::serialize(&header).map_err(|e| bin_err("serialize header", e))?;
    debug_assert_eq!(header_bytes.len() as u64, header_len, "header size must be stable");

    let mut buf = Vec::with_capacity(off as usize);
    buf.extend_from_slice(&header_len.to_le_bytes());
    buf.extend_from_slice(&header_bytes);
    for r in &records {
        buf.extend_from_slice(r);
    }
    std::fs::write(out, buf)?;
    Ok(())
}

/// memmap した バイナリ・ライブラリ。クラスタを遅延デコードする。
pub struct BinaryLibrary {
    mmap: Mmap,
    header: Header,
}

impl BinaryLibrary {
    /// ファイルを memmap して開く（ヘッダのみデコード。クラスタは遅延）。
    pub fn open(path: impl AsRef<Path>) -> Result<Self, IoError> {
        let file = std::fs::File::open(path)?;
        // SAFETY: 読み取り専用マップ。プロセス実行中はファイルを変更しない前提。
        let mmap = unsafe { Mmap::map(&file)? };
        #[cfg(unix)]
        let _ = mmap.advise(memmap2::Advice::WillNeed); // madvise プリフェッチ（失敗は無視）

        if mmap.len() < 8 {
            return Err(IoError::Library("binary library: too short".into()));
        }
        let header_len = u64::from_le_bytes(mmap[0..8].try_into().unwrap()) as usize;
        let end = 8 + header_len;
        if mmap.len() < end {
            return Err(IoError::Library("binary library: truncated header".into()));
        }
        let header: Header =
            bincode::deserialize(&mmap[8..end]).map_err(|e| bin_err("deserialize header", e))?;
        if header.magic != MAGIC {
            return Err(IoError::Library("binary library: bad magic".into()));
        }
        Ok(BinaryLibrary { mmap, header })
    }

    pub fn n_clusters(&self) -> usize {
        self.header.n_clusters as usize
    }

    pub fn has_default_box(&self) -> bool {
        self.header.has_default_box
    }

    pub fn xs_union(&self) -> &[XsType] {
        &self.header.xs_union
    }

    /// クラスタ `i` を遅延デコードする（offset スライスを bincode で復元）。
    pub fn cluster(&self, i: usize) -> Result<Cluster, IoError> {
        let (off, len) = self.header.offsets[i];
        let (off, len) = (off as usize, len as usize);
        if self.mmap.len() < off + len {
            return Err(IoError::Library(format!("binary library: record {i} out of range")));
        }
        bincode::deserialize(&self.mmap[off..off + len]).map_err(|e| bin_err("deserialize cluster", e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::library::{Cluster, MolEntry};
    use crate::pdbqt;

    const LIG: &str = "\
ROOT
ATOM      1  C   LIG A   1       0.000   0.000   0.000  0.00  0.00    +0.000 C
ATOM      2  O   LIG A   1       1.500   0.000   0.000  0.00  0.00    -0.300 OA
ENDROOT
TORSDOF 0
";

    fn mol_entry(id: &str) -> MolEntry {
        MolEntry {
            compound_id: id.to_string(),
            molecule: pdbqt::parse_str(LIG).unwrap(),
            core_atoms: vec![0, 1],
            source_text: LIG.to_string(),
        }
    }

    #[test]
    fn pack_open_roundtrip() {
        let lib = Library {
            clusters: vec![
                Cluster { id: "c0".into(), pocket: None, seed: mol_entry("a"), children: vec![mol_entry("b")] },
                Cluster { id: "c1".into(), pocket: None, seed: mol_entry("c"), children: vec![] },
            ],
        };
        let tmp = std::env::temp_dir().join(format!("isct_libbin_{}.bin", std::process::id()));
        pack(&lib, &tmp).unwrap();
        let bin = BinaryLibrary::open(&tmp).unwrap();
        assert_eq!(bin.n_clusters(), 2);
        assert!(bin.has_default_box());
        assert!(!bin.xs_union().is_empty());
        // クラスタが元と一致（id・compound_id・原子数）。
        let c0 = bin.cluster(0).unwrap();
        assert_eq!(c0.id, "c0");
        assert_eq!(c0.seed.compound_id, "a");
        assert_eq!(c0.children.len(), 1);
        assert_eq!(c0.children[0].compound_id, "b");
        assert_eq!(c0.seed.molecule.atoms.len(), lib.clusters[0].seed.molecule.atoms.len());
        let c1 = bin.cluster(1).unwrap();
        assert_eq!(c1.id, "c1");
        assert!(c1.children.is_empty());
        let _ = std::fs::remove_file(&tmp);
    }
}
