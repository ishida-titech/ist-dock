//! AutoDock 原子型 → Vina XS 原子型・VdW 半径・H 結合/疎水フラグ。
//!
//! AutoDock Vina の XS 原子型体系を再現する。PDBQT の最終列にある AutoDock 型
//! （C/A/N/NA/OA/HD/…）を読み、結合状況（炭素が異種原子に結合しているか、N/O に
//! 極性水素 HD が結合しているか）から XS 型を確定する。スコアリングの距離項
//! `d = r_ij - (R_i + R_j)` を Vina と一致させるため、半径と分類を厳密に合わせる。
//!
//! 参考: AutoDock Vina `atom_constants.h`（Apache-2.0、関数形・定数は公開）。

use serde::{Deserialize, Serialize};

use crate::CoreError;

/// 元素種別（結合検出・型付与に必要な範囲）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Element {
    C,
    N,
    O,
    S,
    P,
    H,
    F,
    Cl,
    Br,
    I,
    Metal,
}

impl Element {
    /// 結合検出用の共有結合半径（Å）。Vina の `covalent_radius` に準拠。
    pub fn covalent_radius(self) -> f64 {
        match self {
            Element::C => 0.77,
            Element::N => 0.75,
            Element::O => 0.73,
            Element::S => 1.02,
            Element::P => 1.06,
            Element::H => 0.37,
            Element::F => 0.71,
            Element::Cl => 0.99,
            Element::Br => 1.14,
            Element::I => 1.33,
            Element::Metal => 1.30,
        }
    }

    /// 異種原子（炭素・水素以外）か。炭素の疎水性判定に使う。
    pub fn is_heteroatom(self) -> bool {
        !matches!(self, Element::C | Element::H)
    }
}

/// PDBQT に現れる AutoDock 原子型。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AdType {
    /// 脂肪族炭素。
    C,
    /// 芳香族炭素。
    A,
    /// 窒素（非 H 結合）。
    N,
    /// 窒素アクセプタ（NA/NS）。
    NAcceptor,
    /// 酸素（非 H 結合）。
    O,
    /// 酸素アクセプタ（OA/OS）。
    OAcceptor,
    /// 硫黄（S/SA。Vina は H 結合しない S_P 扱い）。
    S,
    /// リン。
    P,
    /// フッ素。
    F,
    /// 塩素。
    Cl,
    /// 臭素。
    Br,
    /// ヨウ素。
    I,
    /// 極性水素（HD/HS）。N/O のドナー判定に使う。スコアでは無視。
    PolarH,
    /// 非極性水素（H/HS の一部）。スコアでは無視。
    NonpolarH,
    /// 金属（Mg/Zn/Ca/Mn/Fe/…）。Vina では Met_D（ドナー）。
    Metal,
    /// マクロ環グルーダミー（meeko の `G0`..`G9`）。切断した環の閉環を表す擬似原子で
    /// 相手の実炭素位置に重なる。Vina 同様**スコアからは除外**（相互作用ゼロ）。
    /// 元素は炭素扱い（重なる炭素を誤って極性炭素に分類させないため）。
    Glue,
}

/// `CG0`..`CG9`（meeko マクロ環の実炭素＝グルー炭素ラベル）か。脂肪族炭素として採点する。
fn is_glue_carbon(s: &str) -> bool {
    s.len() > 2 && s.starts_with("CG") && s[2..].bytes().all(|b| b.is_ascii_digit())
}

/// `G0`..`G9`（meeko マクロ環の閉環ダミー）か。スコア除外する。
fn is_glue_dummy(s: &str) -> bool {
    s.len() > 1 && s.starts_with('G') && s[1..].bytes().all(|b| b.is_ascii_digit())
}

impl AdType {
    /// PDBQT 最終列の AutoDock 型文字列から解釈する。
    pub fn parse(s: &str) -> Result<AdType, CoreError> {
        Ok(match s {
            "C" => AdType::C,
            "A" => AdType::A,
            "N" => AdType::N,
            "NA" | "NS" => AdType::NAcceptor,
            "O" => AdType::O,
            "OA" | "OS" => AdType::OAcceptor,
            "S" | "SA" => AdType::S,
            "P" => AdType::P,
            "F" => AdType::F,
            "Cl" | "CL" => AdType::Cl,
            "Br" | "BR" => AdType::Br,
            "I" => AdType::I,
            "HD" => AdType::PolarH,
            "H" | "HS" => AdType::NonpolarH,
            "Mg" | "MG" | "Mn" | "MN" | "Zn" | "ZN" | "Ca" | "CA" | "Fe" | "FE" | "Cu"
            | "CU" | "Na" | "NA " | "K" | "Hg" | "HG" | "Ni" | "NI" | "Co" | "CO" | "Cd"
            | "CD" => AdType::Metal,
            _ if is_glue_carbon(s) => AdType::C,
            _ if is_glue_dummy(s) => AdType::Glue,
            other => return Err(CoreError::UnknownAtomType(other.to_string())),
        })
    }

    /// この原子型の元素。
    pub fn element(self) -> Element {
        match self {
            AdType::C | AdType::A => Element::C,
            AdType::N | AdType::NAcceptor => Element::N,
            AdType::O | AdType::OAcceptor => Element::O,
            AdType::S => Element::S,
            AdType::P => Element::P,
            AdType::F => Element::F,
            AdType::Cl => Element::Cl,
            AdType::Br => Element::Br,
            AdType::I => Element::I,
            AdType::PolarH | AdType::NonpolarH => Element::H,
            AdType::Metal => Element::Metal,
            // グルーダミーは重なる実炭素を誤って極性炭素化させないため炭素扱い。
            AdType::Glue => Element::C,
        }
    }

    /// 水素か（スコアリングから除外する）。
    pub fn is_hydrogen(self) -> bool {
        matches!(self, AdType::PolarH | AdType::NonpolarH)
    }
}

/// Vina XS 原子型。スコアリングの半径・疎水性・H 結合性を担う。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum XsType {
    /// 疎水性炭素（異種原子に未結合）。
    CarbonH,
    /// 極性炭素（異種原子に結合）。
    CarbonP,
    NitrogenP,
    NitrogenD,
    NitrogenA,
    NitrogenDA,
    OxygenP,
    OxygenD,
    OxygenA,
    OxygenDA,
    Sulfur,
    Phosphorus,
    Fluorine,
    Chlorine,
    Bromine,
    Iodine,
    /// 金属（ドナー扱い）。
    MetalD,
}

impl XsType {
    /// XS 型の総数（グリッドの「型→マップ」配列サイズ）。`index()` は 0..COUNT を返す。
    pub const COUNT: usize = 17;

    /// 0..COUNT の連続インデックス（グリッドマップの添字）。
    pub fn index(self) -> usize {
        self as usize
    }

    /// XS VdW 半径（Å）。Vina の `xs_radius`。
    pub fn radius(self) -> f64 {
        use XsType::*;
        match self {
            CarbonH | CarbonP => 1.9,
            NitrogenP | NitrogenD | NitrogenA | NitrogenDA => 1.8,
            OxygenP | OxygenD | OxygenA | OxygenDA => 1.7,
            Sulfur => 2.0,
            Phosphorus => 2.1,
            Fluorine => 1.5,
            Chlorine => 1.8,
            Bromine => 2.0,
            Iodine => 2.2,
            MetalD => 1.2,
        }
    }

    /// 疎水項に寄与する疎水性原子か（C_H と各ハロゲン）。
    pub fn is_hydrophobic(self) -> bool {
        use XsType::*;
        matches!(self, CarbonH | Fluorine | Chlorine | Bromine | Iodine)
    }

    /// 水素結合ドナーか。
    pub fn is_donor(self) -> bool {
        use XsType::*;
        matches!(self, NitrogenD | NitrogenDA | OxygenD | OxygenDA | MetalD)
    }

    /// 水素結合アクセプタか。
    pub fn is_acceptor(self) -> bool {
        use XsType::*;
        matches!(self, NitrogenA | NitrogenDA | OxygenA | OxygenDA)
    }
}

/// 結合状況をもとに AutoDock 型から XS 型を確定する。
///
/// - `bonded_to_hetero`: 炭素が異種原子（C/H 以外）に結合しているか。
/// - `has_polar_h`: N/O に極性水素（HD）が結合しているか（ドナー判定）。
///
/// 水素は `None` を返す（スコアリングから除外）。
pub fn assign_xs_type(ad: AdType, bonded_to_hetero: bool, has_polar_h: bool) -> Option<XsType> {
    use XsType::*;
    Some(match ad {
        AdType::C | AdType::A => {
            if bonded_to_hetero {
                CarbonP
            } else {
                CarbonH
            }
        }
        AdType::N => {
            if has_polar_h {
                NitrogenD
            } else {
                NitrogenP
            }
        }
        AdType::NAcceptor => {
            if has_polar_h {
                NitrogenDA
            } else {
                NitrogenA
            }
        }
        AdType::O => {
            if has_polar_h {
                OxygenD
            } else {
                OxygenP
            }
        }
        AdType::OAcceptor => {
            if has_polar_h {
                OxygenDA
            } else {
                OxygenA
            }
        }
        AdType::S => Sulfur,
        AdType::P => Phosphorus,
        AdType::F => Fluorine,
        AdType::Cl => Chlorine,
        AdType::Br => Bromine,
        AdType::I => Iodine,
        AdType::Metal => MetalD,
        // 水素・グルーダミーはスコアから除外（幾何には残るが相互作用しない）。
        AdType::PolarH | AdType::NonpolarH | AdType::Glue => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn carbon_hydrophobicity() {
        assert_eq!(assign_xs_type(AdType::C, false, false), Some(XsType::CarbonH));
        assert_eq!(assign_xs_type(AdType::C, true, false), Some(XsType::CarbonP));
        assert!(XsType::CarbonH.is_hydrophobic());
        assert!(!XsType::CarbonP.is_hydrophobic());
    }

    #[test]
    fn oxygen_donor_acceptor() {
        // OA + 極性H → ドナー兼アクセプタ
        assert_eq!(assign_xs_type(AdType::OAcceptor, false, true), Some(XsType::OxygenDA));
        assert_eq!(assign_xs_type(AdType::OAcceptor, false, false), Some(XsType::OxygenA));
        assert!(XsType::OxygenDA.is_donor() && XsType::OxygenDA.is_acceptor());
    }

    #[test]
    fn hydrogens_excluded() {
        assert_eq!(assign_xs_type(AdType::PolarH, false, false), None);
        assert_eq!(assign_xs_type(AdType::NonpolarH, false, false), None);
    }

    #[test]
    fn macrocycle_glue_atoms() {
        // CG0..CG9 は脂肪族炭素として採点。G0..G9 はスコア除外（None）。
        assert_eq!(AdType::parse("CG0").unwrap(), AdType::C);
        assert_eq!(AdType::parse("CG1").unwrap(), AdType::C);
        assert_eq!(AdType::parse("G0").unwrap(), AdType::Glue);
        assert_eq!(AdType::parse("G9").unwrap(), AdType::Glue);
        assert_eq!(AdType::Glue.element(), Element::C); // 重なる炭素を極性化しない
        assert_eq!(assign_xs_type(AdType::Glue, false, false), None);
        assert_eq!(assign_xs_type(AdType::C, false, false), Some(XsType::CarbonH));
        // 紛らわしい既存型は誤分類しない。
        assert_eq!(AdType::parse("C").unwrap(), AdType::C);
        assert!(AdType::parse("GG").is_err());
        assert!(AdType::parse("CGx").is_err());
    }

    #[test]
    fn radii_match_vina() {
        assert_eq!(XsType::CarbonH.radius(), 1.9);
        assert_eq!(XsType::OxygenA.radius(), 1.7);
        assert_eq!(XsType::NitrogenD.radius(), 1.8);
    }
}
