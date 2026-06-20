//! PDBQT パーサ。
//!
//! 受容体・リガンドの PDBQT を読み、座標・部分電荷・AutoDock 原子型を取得する。
//! 結合検出（共有結合半径ベース）を行い、炭素の疎水性・N/O のドナー性を確定して
//! Vina XS 型を付与する。水素はスコアリング対象から除外する。
//!
//! リガンドの ROOT/BRANCH/ENDBRANCH からフラグメント木（剛体片と回転結合）を構築し、
//! M2 のポーズ再構成に使う。座標は PDB 固定カラム（31–54）、部分電荷と AutoDock 型は
//! 行末の 2 トークンから取る。

use std::collections::HashMap;
use std::path::Path;

use isct_core::atom_types::{assign_xs_type, AdType, Element};
use isct_core::geometry::{distance, Vec3};
use isct_core::scoring::ScoredAtom;
use serde::{Deserialize, Serialize};

use crate::IoError;

/// 結合判定の許容係数: `dist < (cov_i + cov_j) * BOND_TOLERANCE` を結合とみなす。
const BOND_TOLERANCE: f64 = 1.1;

/// PDBQT の 1 原子。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Atom {
    pub serial: u32,
    pub pos: Vec3,
    pub ad: AdType,
    pub charge: f64,
    pub name: String,
}

/// torsion 木のフラグメント（剛体片）。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fragment {
    /// 親フラグメント（root は None）。
    pub parent: Option<usize>,
    /// 親へつながる回転結合 (parent 側 atom idx, 自身側 atom idx)。root は None。
    pub bond: Option<(usize, usize)>,
    /// このフラグメントに直接属する原子インデックス（部分木は含まない）。
    pub atoms: Vec<usize>,
}

/// パース済み分子（全原子。水素も保持し、結合検出に使う）。
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Molecule {
    pub atoms: Vec<Atom>,
    /// フラグメント木。`fragments[0]` が root。受容体では root 1 個のみ。
    pub fragments: Vec<Fragment>,
    /// 原子 idx → 所属フラグメント idx。
    pub atom_fragment: Vec<usize>,
    /// リガンドの可回転結合数（TORSDOF 行。受容体では 0）。
    pub torsdof: u32,
}

impl Molecule {
    /// 結合検出と XS 型付与を行い、スコアリング対象の重原子を返す（水素は除外）。
    pub fn scored_atoms(&self) -> Vec<ScoredAtom> {
        let flags = self.bond_flags();
        let mut out = Vec::with_capacity(self.atoms.len());
        for (i, atom) in self.atoms.iter().enumerate() {
            if let Some(xs) = assign_xs_type(atom.ad, flags.hetero[i], flags.polar_h[i]) {
                out.push(ScoredAtom { pos: atom.pos, xs });
            }
        }
        out
    }

    /// 各原子の XS 型を返す（重原子のみ Some、水素は None）。ポーズ再構成・枝刈り用。
    pub fn xs_types(&self) -> Vec<Option<isct_core::atom_types::XsType>> {
        let flags = self.bond_flags();
        self.atoms
            .iter()
            .enumerate()
            .map(|(i, a)| assign_xs_type(a.ad, flags.hetero[i], flags.polar_h[i]))
            .collect()
    }

    fn bond_flags(&self) -> BondFlags {
        let n = self.atoms.len();
        let mut hetero = vec![false; n];
        let mut polar_h = vec![false; n];
        for i in 0..n {
            for j in (i + 1)..n {
                let ei = self.atoms[i].ad.element();
                let ej = self.atoms[j].ad.element();
                let r = distance(&self.atoms[i].pos, &self.atoms[j].pos);
                let bond_len = (ei.covalent_radius() + ej.covalent_radius()) * BOND_TOLERANCE;
                if r >= bond_len {
                    continue;
                }
                if ei == Element::C && ej.is_heteroatom() {
                    hetero[i] = true;
                }
                if ej == Element::C && ei.is_heteroatom() {
                    hetero[j] = true;
                }
                if self.atoms[j].ad == AdType::PolarH && matches!(ei, Element::N | Element::O) {
                    polar_h[i] = true;
                }
                if self.atoms[i].ad == AdType::PolarH && matches!(ej, Element::N | Element::O) {
                    polar_h[j] = true;
                }
            }
        }
        BondFlags { hetero, polar_h }
    }
}

struct BondFlags {
    hetero: Vec<bool>,
    polar_h: Vec<bool>,
}

/// PDBQT 文字列をパースする。
pub fn parse_str(text: &str) -> Result<Molecule, IoError> {
    let mut mol = Molecule::default();
    // root フラグメント。
    mol.fragments.push(Fragment {
        parent: None,
        bond: None,
        atoms: Vec::new(),
    });
    let mut serial_to_idx: HashMap<u32, usize> = HashMap::new();
    let mut stack: Vec<usize> = vec![0];
    // BRANCH の結合は (parent側serial, 自身側serial) を仮置きし、後で idx に解決する。
    let mut pending_bonds: Vec<(usize, u32, u32)> = Vec::new();

    for line in text.lines() {
        if line.starts_with("ATOM") || line.starts_with("HETATM") {
            let atom = parse_atom_line(line)?;
            let idx = mol.atoms.len();
            serial_to_idx.insert(atom.serial, idx);
            let frag = *stack.last().unwrap();
            mol.fragments[frag].atoms.push(idx);
            mol.atom_fragment.push(frag);
            mol.atoms.push(atom);
        } else if let Some(rest) = line.strip_prefix("BRANCH") {
            let (from_s, to_s) = parse_branch(rest, line)?;
            let parent = *stack.last().unwrap();
            let new_id = mol.fragments.len();
            mol.fragments.push(Fragment {
                parent: Some(parent),
                bond: None,
                atoms: Vec::new(),
            });
            pending_bonds.push((new_id, from_s, to_s));
            stack.push(new_id);
        } else if line.starts_with("ENDBRANCH") {
            if stack.len() > 1 {
                stack.pop();
            }
        } else if let Some(rest) = line.strip_prefix("TORSDOF") {
            mol.torsdof = rest
                .trim()
                .parse()
                .map_err(|_| IoError::Pdbqt(format!("invalid TORSDOF line: {line}")))?;
        }
    }

    if mol.atoms.is_empty() {
        return Err(IoError::Pdbqt("no ATOM/HETATM records found".into()));
    }

    // 結合 serial → idx を解決。
    for (frag_id, from_s, to_s) in pending_bonds {
        let from = *serial_to_idx
            .get(&from_s)
            .ok_or_else(|| IoError::Pdbqt(format!("BRANCH references unknown serial {from_s}")))?;
        let to = *serial_to_idx
            .get(&to_s)
            .ok_or_else(|| IoError::Pdbqt(format!("BRANCH references unknown serial {to_s}")))?;
        mol.fragments[frag_id].bond = Some((from, to));
    }

    Ok(mol)
}

/// PDBQT ファイルを読み込む。
pub fn parse_file(path: impl AsRef<Path>) -> Result<Molecule, IoError> {
    let text = std::fs::read_to_string(path)?;
    parse_str(&text)
}

/// 元の PDBQT テキストの ATOM/HETATM 行座標を新しい座標で置き換えてポーズを書き出す。
/// `coords` は ATOM/HETATM の出現順（= パース順）と一致する必要がある。
pub fn write_pose(original: &str, coords: &[Vec3]) -> Result<String, IoError> {
    let mut out = String::with_capacity(original.len());
    let mut i = 0;
    for line in original.lines() {
        if line.starts_with("ATOM") || line.starts_with("HETATM") {
            let c = coords.get(i).ok_or_else(|| {
                IoError::Pdbqt("write_pose: coords shorter than ATOM records".into())
            })?;
            // 座標カラム 31–54（0-indexed 30..54）を置換、それ以外は保持。
            let head = line.get(..30).unwrap_or(line);
            let tail = line.get(54..).unwrap_or("");
            out.push_str(head);
            out.push_str(&format!("{:8.3}{:8.3}{:8.3}", c.x, c.y, c.z));
            out.push_str(tail);
            out.push('\n');
            i += 1;
        } else {
            out.push_str(line);
            out.push('\n');
        }
    }
    if i != coords.len() {
        return Err(IoError::Pdbqt(format!(
            "write_pose: {} coords but {i} ATOM records",
            coords.len()
        )));
    }
    Ok(out)
}

fn parse_branch(rest: &str, line: &str) -> Result<(u32, u32), IoError> {
    let mut it = rest.split_whitespace();
    let from = it.next().and_then(|s| s.parse().ok());
    let to = it.next().and_then(|s| s.parse().ok());
    match (from, to) {
        (Some(f), Some(t)) => Ok((f, t)),
        _ => Err(IoError::Pdbqt(format!("invalid BRANCH line: {line}"))),
    }
}

fn parse_atom_line(line: &str) -> Result<Atom, IoError> {
    let serial = line
        .get(6..11)
        .and_then(|s| s.trim().parse::<u32>().ok())
        .ok_or_else(|| IoError::Pdbqt(format!("bad serial in: {line}")))?;

    // 座標は固定カラム 31–54（0-indexed 30..54）。
    let coord = |range: std::ops::Range<usize>, what: &str| -> Result<f64, IoError> {
        line.get(range)
            .and_then(|s| s.trim().parse::<f64>().ok())
            .ok_or_else(|| IoError::Pdbqt(format!("bad {what} coordinate in: {line}")))
    };
    let x = coord(30..38, "x")?;
    let y = coord(38..46, "y")?;
    let z = coord(46..54, "z")?;

    // 行末 2 トークン: charge（末尾から 2 番目）と AutoDock 型（末尾）。
    let mut tokens = line.split_whitespace().rev();
    let ad_str = tokens
        .next()
        .ok_or_else(|| IoError::Pdbqt(format!("missing atom type in: {line}")))?;
    let charge = tokens
        .next()
        .and_then(|s| s.parse::<f64>().ok())
        .ok_or_else(|| IoError::Pdbqt(format!("missing/invalid charge in: {line}")))?;

    let name = line.get(12..16).map(|s| s.trim().to_string()).unwrap_or_default();

    Ok(Atom {
        serial,
        pos: Vec3::new(x, y, z),
        ad: AdType::parse(ad_str)?,
        charge,
        name,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use isct_core::atom_types::XsType;

    const LIG: &str = "\
ROOT
ATOM      1  C   LIG A   1       0.000   0.000   0.000  0.00  0.00    +0.000 C
ATOM      2  O   LIG A   1       1.300   0.000   0.000  0.00  0.00    -0.300 OA
ENDROOT
BRANCH   2   3
ATOM      3  C   LIG A   1       2.500   0.000   0.000  0.00  0.00    +0.100 C
ENDBRANCH   2   3
TORSDOF 1
";

    #[test]
    fn parses_atoms_and_torsdof() {
        let mol = parse_str(LIG).unwrap();
        assert_eq!(mol.atoms.len(), 3);
        assert_eq!(mol.torsdof, 1);
        assert_eq!(mol.atoms[0].ad, AdType::C);
        assert_eq!(mol.atoms[1].ad, AdType::OAcceptor);
        assert!((mol.atoms[1].pos.x - 1.3).abs() < 1e-9);
        assert!((mol.atoms[1].charge + 0.3).abs() < 1e-9);
    }

    #[test]
    fn builds_fragment_tree() {
        let mol = parse_str(LIG).unwrap();
        // root + 1 branch
        assert_eq!(mol.fragments.len(), 2);
        assert_eq!(mol.fragments[0].atoms, vec![0, 1]);
        assert_eq!(mol.fragments[1].parent, Some(0));
        assert_eq!(mol.fragments[1].bond, Some((1, 2))); // O(idx1)-C(idx2)
        assert_eq!(mol.fragments[1].atoms, vec![2]);
        assert_eq!(mol.atom_fragment, vec![0, 0, 1]);
    }

    #[test]
    fn carbon_bonded_to_oxygen_is_polar() {
        let scored = parse_str(LIG).unwrap().scored_atoms();
        // 3 重原子（C, O, C）。idx0 C は O に結合 → C_P、idx1 O は O_A、idx2 C は C_P。
        assert_eq!(scored.len(), 3);
        assert_eq!(scored[0].xs, XsType::CarbonP);
        assert_eq!(scored[1].xs, XsType::OxygenA);
    }
}
