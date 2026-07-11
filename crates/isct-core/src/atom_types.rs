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
    Si,
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
            Element::Si => 1.11,
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
    /// ケイ素（`Si`）。Vina/AutoDock の Si。疎水性・H 結合なしで炭素より大きい半径（XS 2.1Å）。
    Silicon,
    /// AutoDock 水和ドッキングの水擬原子（`W`）。Vina 同様**スコアから除外**（相互作用ゼロ）。
    /// 幾何には残す。tungsten ではなく水（AutoDock/meeko 慣例）。
    Water,
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

/// AD 型文字列の先頭 1–2 アルファベットを元素記号として正規化する（"SE"→"Se", "MG"→"Mg", "B"→"B"）。
/// 数字・末尾ラベルは無視する。
fn leading_element_symbol(s: &str) -> String {
    let mut it = s.chars().filter(|c| c.is_ascii_alphabetic());
    let mut out = String::new();
    if let Some(c0) = it.next() {
        out.push(c0.to_ascii_uppercase());
    }
    if let Some(c1) = it.next() {
        out.push(c1.to_ascii_lowercase());
    }
    out
}

/// 元素記号が（Vina で個別 XS 型を持たない）金属・遷移金属・ランタノイド等か。→ Met_D 扱い。
fn is_metal_symbol(sym: &str) -> bool {
    matches!(
        sym,
        "Li" | "Be" | "Na" | "Mg" | "Al" | "K" | "Ca" | "Sc" | "Ti" | "V" | "Cr" | "Mn"
            | "Fe" | "Co" | "Ni" | "Cu" | "Zn" | "Ga" | "Ge" | "Rb" | "Sr" | "Y" | "Zr"
            | "Nb" | "Mo" | "Tc" | "Ru" | "Rh" | "Pd" | "Ag" | "Cd" | "In" | "Sn" | "Sb"
            | "Cs" | "Ba" | "La" | "Ce" | "Pr" | "Nd" | "Pm" | "Sm" | "Eu" | "Gd" | "Tb"
            | "Dy" | "Ho" | "Er" | "Tm" | "Yb" | "Lu" | "Hf" | "Ta" | "Re" | "Os" | "Ir"
            | "Pt" | "Au" | "Hg" | "Tl" | "Pb" | "Bi" | "Po" | "Th" | "Pa" | "U"
    )
}

/// Vina 型集合に無い AD 型を、元素推定で最寄りの既知型へ寄せる（フォールバック）。Vina が採点する元素へ
/// マップして**クラッシュを避ける**。元素すら推定できない場合のみ `Err`（呼び出し側で screen はその
/// リガンドをスキップ、単発 dock は明確エラーにする）。近似が silent にならないよう型ごと 1 度だけ警告する。
fn fallback_ad_type(s: &str) -> Result<AdType, CoreError> {
    let sym = leading_element_symbol(s);
    let ad = match sym.as_str() {
        // カルコゲン（Se/Te）→ 硫黄相当。
        "Se" | "Te" => AdType::S,
        // アスタチン → 最寄りハロゲン（ヨウ素）。
        "At" => AdType::I,
        // ホウ素 → 炭素相当（疎水・半金属）。
        "B" => AdType::C,
        // ケイ素（保険。通常は exact 一致するが元素推定経路でも拾う）。
        "Si" => AdType::Silicon,
        // 金属・遷移金属・ランタノイド等 → Met_D。
        m if is_metal_symbol(m) => AdType::Metal,
        _ => return Err(CoreError::UnknownAtomType(s.to_string())),
    };
    warn_unknown_type_once(s, ad);
    Ok(ad)
}

/// 未知型フォールバックを型ごとに 1 度だけ stderr へ警告する（プロセス全体で重複排除）。
fn warn_unknown_type_once(orig: &str, mapped: AdType) {
    use std::sync::{Mutex, OnceLock};
    static SEEN: OnceLock<Mutex<std::collections::HashSet<String>>> = OnceLock::new();
    let seen = SEEN.get_or_init(|| Mutex::new(std::collections::HashSet::new()));
    if let Ok(mut set) = seen.lock() {
        if set.insert(orig.to_string()) {
            eprintln!(
                "warning: 未知の AutoDock 原子型 '{orig}' を元素推定で {mapped:?} として近似採点します"
            );
        }
    }
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
            "Si" | "SI" => AdType::Silicon,
            // W = AutoDock 水和ドッキングの水擬原子（tungsten ではない。Vina/meeko 慣例）。採点除外。
            "W" => AdType::Water,
            "HD" => AdType::PolarH,
            "H" | "HS" => AdType::NonpolarH,
            "Mg" | "MG" | "Mn" | "MN" | "Zn" | "ZN" | "Ca" | "CA" | "Fe" | "FE" | "Cu"
            | "CU" | "Na" | "K" | "Hg" | "HG" | "Ni" | "NI" | "Co" | "CO" | "Cd"
            | "CD" | "V" | "MO" => AdType::Metal,
            _ if is_glue_carbon(s) => AdType::C,
            _ if is_glue_dummy(s) => AdType::Glue,
            // Vina 型集合に無い型は元素推定で最寄りの既知型へフォールバック（クラッシュしない）。
            // 元素すら不明なら Err（screen はリガンドをスキップ、単発 dock は明確エラー）。
            other => fallback_ad_type(other)?,
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
            AdType::Silicon => Element::Si,
            // 水擬原子は採点除外だが元素は酸素（水和水）。covalent 距離外に置かれるため結合検出に影響しない。
            AdType::Water => Element::O,
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
    /// ケイ素（Vina 1.2 の `XS_TYPE_Si`）。半径 2.2Å・疎水非該当・H 結合なし＝立体項のみ寄与。
    Silicon,
}

impl XsType {
    /// XS 型の総数（グリッドの「型→マップ」配列サイズ）。`index()` は 0..COUNT を返す。
    pub const COUNT: usize = 18;

    /// 全 XS 型（`index()` 昇順）。テーブル構築などで型を走査するのに使う。
    pub const ALL: [XsType; Self::COUNT] = [
        XsType::CarbonH,
        XsType::CarbonP,
        XsType::NitrogenP,
        XsType::NitrogenD,
        XsType::NitrogenA,
        XsType::NitrogenDA,
        XsType::OxygenP,
        XsType::OxygenD,
        XsType::OxygenA,
        XsType::OxygenDA,
        XsType::Sulfur,
        XsType::Phosphorus,
        XsType::Fluorine,
        XsType::Chlorine,
        XsType::Bromine,
        XsType::Iodine,
        XsType::MetalD,
        XsType::Silicon,
    ];

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
            // ケイ素の XS 半径。Vina 1.2 `atom_constants.h` の xs_radius に一致（2.2Å）。
            Silicon => 2.2,
        }
    }

    /// 疎水項に寄与する疎水性原子か（C_H と各ハロゲン）。**Si は Vina 1.2 で疎水に含まれない**
    /// （`xs_is_hydrophobic` は C_H/F/Cl/Br/I のみ）ため除外＝Si は立体項のみに寄与。
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
        AdType::Silicon => Silicon,
        // 水素・グルーダミー・水擬原子はスコアから除外（幾何には残るが相互作用しない）。
        AdType::PolarH | AdType::NonpolarH | AdType::Glue | AdType::Water => return None,
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

    #[test]
    fn silicon_first_class() {
        // Si は正式対応: 専用 XS 型・半径 2.1・疎水性。
        assert_eq!(AdType::parse("Si").unwrap(), AdType::Silicon);
        assert_eq!(AdType::parse("SI").unwrap(), AdType::Silicon);
        assert_eq!(AdType::Silicon.element(), Element::Si);
        assert_eq!(assign_xs_type(AdType::Silicon, false, false), Some(XsType::Silicon));
        // Vina 1.2 パリティ: xs_radius=2.2, 疎水非該当, donor/acceptor なし（立体項のみ）。
        assert_eq!(XsType::Silicon.radius(), 2.2);
        assert!(!XsType::Silicon.is_hydrophobic());
        assert!(!XsType::Silicon.is_donor() && !XsType::Silicon.is_acceptor());
        // Si は heteroatom（Vina `ad_is_heteroatom`）＝隣接炭素を C_P 化する。
        assert!(Element::Si.is_heteroatom());
        // COUNT/ALL/index が整合（グリッド・LUT の添字健全性）。
        assert_eq!(XsType::ALL.len(), XsType::COUNT);
        assert_eq!(XsType::Silicon.index(), XsType::COUNT - 1);
    }

    #[test]
    fn water_pseudo_atom_excluded() {
        // W は AutoDock 水和水（tungsten ではない）→ 採点除外。
        assert_eq!(AdType::parse("W").unwrap(), AdType::Water);
        assert_eq!(assign_xs_type(AdType::Water, false, false), None);
    }

    #[test]
    fn unknown_types_fallback_by_element() {
        // カルコゲン → 硫黄、ホウ素 → 炭素、アスタチン → ヨウ素、金属 → Met_D。
        assert_eq!(AdType::parse("Se").unwrap(), AdType::S);
        assert_eq!(AdType::parse("SE").unwrap(), AdType::S);
        assert_eq!(AdType::parse("Te").unwrap(), AdType::S);
        assert_eq!(AdType::parse("B").unwrap(), AdType::C);
        assert_eq!(AdType::parse("At").unwrap(), AdType::I);
        // 金属記号（exact テーブルに無いもの）は Met_D へ。
        assert_eq!(AdType::parse("Al").unwrap(), AdType::Metal);
        assert_eq!(AdType::parse("Pt").unwrap(), AdType::Metal);
        assert_eq!(AdType::parse("Gd").unwrap(), AdType::Metal);
        // 元素すら推定不能なものは最終手段としてエラー（screen 側でスキップ／dock は明確エラー）。
        assert!(AdType::parse("Xx").is_err());
        assert!(AdType::parse("Zz").is_err());
    }

    #[test]
    fn element_symbol_normalization() {
        // 大小・数字混じりを正規化（1–2 文字）。ハロゲン C/Cl, B/Br の取り違えが無いこと。
        assert_eq!(leading_element_symbol("SE"), "Se");
        assert_eq!(leading_element_symbol("mg"), "Mg");
        assert_eq!(leading_element_symbol("B"), "B");
        assert_eq!(leading_element_symbol("Fe2"), "Fe");
        // 既存の正規型は exact 一致でフォールバックに来ない（回帰防止の意味で確認）。
        assert_eq!(AdType::parse("Cl").unwrap(), AdType::Cl);
        assert_eq!(AdType::parse("Br").unwrap(), AdType::Br);
        assert_eq!(AdType::parse("C").unwrap(), AdType::C);
    }
}
