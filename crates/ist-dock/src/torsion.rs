//! A-2 — 明示的なねじれポテンシャル（既定は無効）。
//!
//! # なぜ必要か
//!
//! AutoDock Vina の目的関数には**ねじれポテンシャルが無い**。回転結合の好ましい角度は
//! 分子内の非結合項（結合グラフ距離 ≥4 の重原子対）だけで決まる。段 91 の計測では、
//! Astex 10 標的の回転結合 70 本のうち
//!
//! - **10 本（14%）は動く重原子が 0 本**（末端の -OH・-NH2 で、動くのは水素だけ）。
//!   Vina は水素に XS 型を与えないので、この自由度は**どの項も変えられない**。
//!   ねじれポテンシャルを足しても意味は無い（別の対処が要る）。
//! - 残り 60 本のうち **20 本（33%）は障壁が 1 kcal/mol 未満**。実測の回転障壁は
//!   sp3-sp3 の単結合で 3 kcal/mol 前後あるので、ここが過小評価されている。
//!
//! # 方式
//!
//! **UFF（Rappé ら, J. Am. Chem. Soc. 1992, 114, 10024）の一般ねじれ則**を使う。
//! UFF を選ぶ理由は、必要な情報が**各原子の元素と混成だけ**で、PDBQT から自己完結に
//! 決められること。外部ファイルも RDKit も要らず、リガンドごとの前処理費用が 0 になる。
//! Astex のリガンドは ADFRsuite 調製で `REMARK SMILES IDX` を持たないため、SMILES の
//! 対応づけを要する方式（MMFF など）はそもそも使えない。
//!
//! 形は `V(φ) = Σ (Vₙ/2)·(1 + cos(n·φ − γₙ))`。UFF の規則は次のように移す。
//!
//! | 結合の混成 | n | V | γ |
//! |---|---|---|---|
//! | sp3–sp3 | 3 | `sqrt(Vi·Vj)` | 0 |
//! | sp3–sp3（両方 O・S） | 2 | 2.0（O）/ 6.8（S） | π/2 |
//! | sp2–sp2 | 2 | `5·sqrt(Ui·Uj)·(1 + 4.18·ln BO)` | π |
//! | sp2–sp3 | 6 | 1.0 | π |
//! | sp2–sp3（sp3 側が O・S） | 2 | `5·sqrt(Ui·Uj)·(1 + 4.18·ln BO)` | π |
//! | sp2–sp3（sp2 側が別の sp2 と共役） | 3 | 2.0 | π |
//! | sp が絡む | — | 0（直線なのでねじれが定義できない） | — |
//!
//! # ビアリールでの不一致（公開値で裏が取れている）
//!
//! **sp2–sp2 の式は C–C で V = 10 kcal/mol を与え、最小を平面に置く。
//! ビアリールではどちらも公開値と合わない。**
//!
//! - ビフェニルの気相の平衡ねじれ角は **44.4 ± 1.2°**（気相電子回折。Almenningen ら,
//!   J. Mol. Struct. 128 (1985) 59; 障壁の解析は Bastiansen・Samdal, 同巻 115）。
//!   最高水準の非経験計算では **45.8°**（Johansson・Olsen, J. Chem. Theory Comput.
//!   4 (2008) 1460）。**平面ではない。**
//! - 回転障壁は **平面で 6.0 ± 2.1 kJ/mol、直交で 6.5 ± 2.0 kJ/mol**（＝約 1.4 と
//!   1.6 kcal/mol、同上の 1985 年の実測）。非経験計算では 8.0 と 8.3 kJ/mol（約 1.9・2.0）。
//!   **UFF の V = 10 kcal/mol は実測障壁の 5〜7 倍。**
//!
//! なお `V = 5·sqrt(U_C·U_C) = 10` は UFF の公開パラメータからの算術で、
//! **UFF の論文がビアリールについてこの値を挙げているわけではない。**
//! 「UFF がビアリールを平面化しすぎる」と明記した文献は探したが見つからなかったので、
//! ここで主張しているのは**本リポジトリでの実測と上の公開値との不一致**に限る。
//!
//! 実在の分子ではオルト位の立体反発が平面化を打ち消すが、**Vina は水素を採点しないので
//! その反発が無い**（オルトの炭素どうしは結合グラフ距離 4 ＝ 1-4 なので、1-4 を数えない
//! 既定では寄与が 0）。つまり A-2 と A-1 は相互作用する。両方を振って測ること。
//!
//! **結晶データを直接当てはめてはいけない。** ビアリールの結晶構造の torsion 分布には
//! 平面に鋭い峰が立つが、これは**分子が結晶学的な反転中心に乗ることで強制された平面性**
//! であって配座の好みではない（Brock・Minton, J. Am. Chem. Soc. 111 (1989) 4586;
//! CCDC の技術ノート "Exploring symmetry related bias in conformational data from the
//! Cambridge Structural Database"）。峰を除いた広い成分は気相とよく一致する。
//!
//! # 角度の取り方
//!
//! エンジンの `Conformation::torsions[k]` は**参照ポーズからの相対角**なので、絶対二面角は
//! `φ = φ0 + s·t`（`φ0` は参照ポーズでの二面角、`s = ±1` は回転の向き）。`s` は構築時に
//! 微小回転を実際に適用して測る（エンジン内部の回転の向きの規約に依存しないため）。
//! この形なら勾配は `dV/dt = s·V'(φ)` で、原子座標への連鎖律が要らない。

use ist_core::geometry::{distance, Vec3};
use ist_io::pdbqt::Molecule;

/// 混成。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Hyb {
    Sp,
    Sp2,
    Sp3,
}

/// ねじれの種別（診断・記録用）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TorsionKind {
    /// 動く重原子が 0 本（水素しか動かない）。UFF の係数は与えるが実質 0 に近い。
    NoHeavyAtomMoves,
    Sp3Sp3,
    /// sp3–sp3 で両方が 16 族（O・S）。
    Sp3Sp3Group16,
    Sp2Sp2,
    /// sp2–sp2 のうち**アミド様**（sp2 の窒素と、末端酸素を持つ sp2 炭素の結合）。
    /// 結晶構造で平面が強く好まれる唯一の sp2–sp2 で、UFF の平面則が当たる種別。
    AmideLike,
    Sp2Sp3,
    /// sp2–sp3 で sp3 側が 16 族（フェノール・エステルなど）。
    Sp2Sp3Group16,
    /// sp2–sp3 で sp2 側が別の sp2 と共役している。
    Sp2Sp3Conjugated,
    /// sp が絡む、または分類できない（寄与 0）。
    None,
}

impl TorsionKind {
    /// **結晶構造の検証を通った種別か。** 通らない種別は係数を記録するだけで評価しない。
    ///
    /// Astex 85 標的の結晶リガンド（回転結合 476 本）に UFF の一般則をそのまま当て、
    /// 「結晶配座がそのポテンシャルの最小からどれだけ離れ、どれだけの歪みを負うか」を
    /// 測った結果（エンジン ist-dock、分子内項のみ、0.5° 刻み）:
    ///
    /// | 種別 | 本数 | 障壁 | 結晶の歪みの中央値 | 最小までの角度の中央値 | 判定 |
    /// |---|---:|---:|---:|---:|---|
    /// | sp3–sp3 | 20 | 2.12 | **0.03** | **7.2°** | **合格** |
    /// | sp2–sp3 | 47 | 1.00 | **0.24** | **9.7°** | **合格** |
    /// | sp2–sp3（共役） | 81 | 2.00 | 1.32 | 36.2° | 不合格 |
    /// | sp2–sp3（16 族） | 59 | 10.00 | 3.88 | 28.5° | 不合格 |
    /// | sp2–sp2 | 178 | 10.00 | **5.46** | **47.6°** | **不合格** |
    ///
    /// 不合格の 3 種はいずれも sp2 中心を含み、UFF が**平面を最小**と置くもの。
    /// 結晶構造はそうなっておらず、とくに sp2–sp2（回転結合の 37%）では結晶の角度が
    /// 平面から中央値 47.6° 離れていて、10 kcal/mol の 2 回対称を掛けると**結晶配座に
    /// 中央値 5.46 kcal/mol の罰則**が付く。リガンド 1 本あたりでは**中央値 16.0 kcal/mol**。
    /// 最小の**位置**が間違っているので、係数を小さくしても直らない（平面へ押す向きは変わらない）。
    ///
    /// 実在の分子ではオルト位の立体反発が平面化を打ち消すが、**Vina は水素を採点しない**
    /// のでその反発が無い（オルトの炭素どうしは結合グラフ距離 4 ＝ 1-4 で、既定では寄与 0）。
    /// **「UFF がビアリールを平面化しすぎる」と明記した文献は探したが見つからなかった**ので、
    /// ここでの主張は実測と公開値（気相 44.4 ± 1.2°、障壁 1.4〜1.6 kcal/mol）との
    /// 不一致に限る。詳しくはモジュールの冒頭。
    ///
    /// # 独立な集合での確認（段 95）
    ///
    /// 「平面から 47.6°」が Astex 固有かどうかを、**標的も調製ツールも重ならない集合**で
    /// 測り直した。水ベンチ 4 標的（HSP90α・BRD4・炭酸脱水酵素 II・FABP4、PDB 由来の
    /// 結晶リガンド 1,272 本・回転結合 6,761 本、Meeko 調製）。リガンドの重なりは 3 本だけ。
    ///
    /// sp2–sp2 を下位分類して平面からの距離の中央値を取ると:
    ///
    /// | 下位分類 | Astex（n） | 水ベンチ（n） | 動かす採点原子の割合 | 判定 |
    /// |---|---|---|---:|---|
    /// | 環–環（ビアリール型） | 44.7°（21） | **47.0°（529）** | 26% | 平面は誤り |
    /// | 環–鎖 | 41.1°（96） | 35.5°（1,582） | 31% | 二峰性（平面と直交） |
    /// | 鎖–鎖 | 55.6°（60） | 50.1°（862） | 21% | 二峰性 |
    /// | **アミド様** | 0.9°（1） | **2.7°（80）** | **38%** | **平面で正しい** |
    ///
    /// **「平面から 47.6°」は Astex 固有ではない。** ビアリールは 44.7° 対 47.0°
    /// （データ量 25 倍）で一致し、分布にも 40〜60° の峰がある（一様分布に対する χ² は
    /// 89.6、自由度 8）。構造上の一般的事実。
    ///
    /// **一方アミド様だけは平面が強く好まれる**（水ベンチで 76% が 10° 未満、χ² = 353）。
    /// UFF の平面則が当たる唯一の sp2–sp2 で、しかも**動かす採点原子の割合が最大（38%）**
    /// ＝ポーズを最も大きく変えるクラス。段 92.3 で sp2–sp2 を一括で落としたのは
    /// **この種別を一緒に捨てる誤りだった**ので、ここで有効にした（係数は UFF のまま、
    /// 追加パラメータなし）。
    ///
    /// なお sp2–sp3 が「通った」のは**ほぼ平坦な関数だったから**にすぎない。
    /// 6 回対称なので最小が 60° ごとに並び、最小までの距離は最大でも 30°。実測の中央値は
    /// 20.3°（Astex）/ 17.9°（水ベンチ）で、一様分布から期待される 15° とほぼ同じ。
    /// 障壁 1 kcal/mol・動かす原子の割合 9.5% で、**無害だが効果もない**。
    ///
    /// ビアリールに ±45° 付近の最小を置く形は、44.7°/47.0° を自分のデータから取ると
    /// 当てはめになる。**ただし公開値がこの数字を裏づけている**: 気相電子回折の
    /// 44.4 ± 1.2° と非経験計算の 45.8° はどちらも実測とほぼ一致する。外部の基準として
    /// 使えるものは次の 2 つ（実装は未着手）。
    ///
    /// 1. **ビフェニルの気相値**: 最小 44.4°、障壁は平面 1.4・直交 1.6 kcal/mol。
    /// 2. **Torsion Library**（Schärfer ら, J. Med. Chem. 56 (2013) 2016;
    ///    Guba ら, J. Chem. Inf. Model. 56 (2016) 1）。SMARTS で型付けする公開規則集で、
    ///    2013・2016 版のパターンは著者の配布ページから取得できる。ベンゼン–ベンゼンの
    ///    一般規則は **±90° を中心に緩い半幅 50°（＝40〜140° が許容）**で、
    ///    それに加えて 0°/180° に半幅 10° の狭い窓を持つ。オルト置換体の規則は
    ///    **平面の窓を持たず ±90° ± 40° のみ**。
    pub fn is_crystal_validated(self) -> bool {
        matches!(
            self,
            TorsionKind::Sp3Sp3 | TorsionKind::Sp2Sp3 | TorsionKind::AmideLike
        )
    }

    pub fn name(self) -> &'static str {
        match self {
            TorsionKind::NoHeavyAtomMoves => "空の自由度(水素のみ)",
            TorsionKind::Sp3Sp3 => "sp3-sp3",
            TorsionKind::Sp3Sp3Group16 => "sp3-sp3(16族)",
            TorsionKind::Sp2Sp2 => "sp2-sp2",
            TorsionKind::AmideLike => "アミド様",
            TorsionKind::Sp2Sp3 => "sp2-sp3",
            TorsionKind::Sp2Sp3Group16 => "sp2-sp3(16族)",
            TorsionKind::Sp2Sp3Conjugated => "sp2-sp3(共役)",
            TorsionKind::None => "なし",
        }
    }
}

/// 回転結合 1 本の素性（診断・調査用）。どの規則が当たったかを後から説明できるようにする。
#[derive(Debug, Clone)]
pub struct TorsionDetail {
    /// 結合の両端の全原子インデックス。
    pub atoms: (usize, usize),
    /// 二面角の基準に使った両端の原子。
    pub refs: (usize, usize),
    /// 両端の元素記号。
    pub elements: (String, String),
    /// 両端の混成。
    pub hyb: (Hyb, Hyb),
    /// 両端が環に属するか（回転結合そのものは環結合ではないが、端の原子が環にいることはある）。
    pub in_ring: (bool, bool),
    /// 両端の重原子隣接数。
    pub heavy_degree: (usize, usize),
    /// 片方が sp2 の窒素で、相手の sp2 炭素が末端酸素（カルボニル）を持つ＝アミド様か。
    pub amide_like: bool,
    /// 両端が属する最小環の大きさ（環に属さなければ 0）。
    pub ring_size: (usize, usize),
    /// その最小環が**炭素だけ**で出来ているか（複素環でないか）。
    pub ring_all_carbon: (bool, bool),
    /// **オルト位に付いている重原子の数**（両側の合計、最大 4）。
    ///
    /// ビアリールの公開値（ビフェニルの気相 44.4°）は**オルト位 4 箇所すべてが水素**の
    /// 場合の値なので、適用範囲を決めるのにこの数が必要。オルト置換が増えると
    /// ねじれは大きくなる（4 置換で約 85°）。環に属さない側は 0 と数える。
    pub ortho_heavy: usize,
}

impl TorsionDetail {
    /// sp2–sp2 結合の下位分類（環どうし／環と鎖／鎖どうし）。sp2–sp2 以外では `None`。
    pub fn sp2_sp2_subtype(&self) -> Option<&'static str> {
        if self.hyb != (Hyb::Sp2, Hyb::Sp2) {
            return None;
        }
        if self.amide_like {
            return Some("アミド様");
        }
        Some(match self.in_ring {
            (true, true) => "環-環（ビアリール型）",
            (true, false) | (false, true) => "環-鎖",
            (false, false) => "鎖-鎖",
        })
    }
}

/// torsion 1 本のポテンシャル。`V(φ) = Σ (v/2)(1 + cos(n·φ − γ))`、`φ = phi0 + sign·t`。
#[derive(Debug, Clone)]
pub struct TorsionTerm {
    /// 参照ポーズでの絶対二面角（rad）。
    pub phi0: f64,
    /// `torsions[k]` の増加に対する二面角の増加の向き（+1 または −1）。
    pub sign: f64,
    /// (n, V, γ) の並び。空なら寄与 0。
    pub harmonics: Vec<(f64, f64, f64)>,
    pub kind: TorsionKind,
    /// 分類の根拠（診断・調査用。探索計算では使わない）。
    pub detail: Option<TorsionDetail>,
}

impl TorsionTerm {
    /// 評価に使うか（結晶構造の検証を通った種別だけ使う）。
    #[inline]
    pub fn is_active(&self) -> bool {
        !self.harmonics.is_empty() && self.kind.is_crystal_validated()
    }

    /// 値と、`torsions[k]` に対する微分。**検証を通らない種別は 0 を返す。**
    #[inline]
    pub fn value_grad(&self, t: f64) -> (f64, f64) {
        if !self.kind.is_crystal_validated() {
            return (0.0, 0.0);
        }
        let phi = self.phi0 + self.sign * t;
        let mut e = 0.0;
        let mut g = 0.0;
        for &(n, v, gamma) in &self.harmonics {
            let a = n * phi - gamma;
            e += 0.5 * v * (1.0 + a.cos());
            g += -0.5 * v * n * a.sin();
        }
        (e, self.sign * g)
    }

    /// `t` における「このポテンシャルが与える歪み」と「最も近い最小までの角度（度）」。
    ///
    /// **A-2 の妥当性を結晶構造で確かめるための診断。** 結晶配座（`t = 0`）で歪みが
    /// 大きい、あるいは最小から大きく離れているなら、その結合種への当てはめが
    /// 間違っている（最小の位置が実際の構造と合っていない）。
    pub fn strain(&self, t: f64) -> (f64, f64) {
        if self.harmonics.is_empty() {
            return (0.0, 0.0);
        }
        let v = |phi: f64| -> f64 {
            self.harmonics
                .iter()
                .map(|&(n, vv, g)| 0.5 * vv * (1.0 + (n * phi - g).cos()))
                .sum()
        };
        let phi = self.phi0 + self.sign * t;
        let here = v(phi);
        let mut lo = f64::INFINITY;
        // 0.5° 刻みで最小値と、最小に十分近い点のうち最も近い角度を探す。
        let mut grid = Vec::with_capacity(720);
        for i in 0..720 {
            let p = (i as f64) * 0.5f64.to_radians();
            let e = v(p);
            lo = lo.min(e);
            grid.push((p, e));
        }
        let mut best_d = 180.0_f64;
        for &(p, e) in &grid {
            if e <= lo + 1e-6 {
                let mut d = (p - phi).to_degrees().rem_euclid(360.0);
                if d > 180.0 {
                    d = 360.0 - d;
                }
                best_d = best_d.min(d);
            }
        }
        (here - lo, best_d)
    }

    /// 1 周したときの障壁の高さ（最大 − 最小）。診断用。
    pub fn barrier(&self) -> f64 {
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for i in 0..360 {
            let phi = (i as f64).to_radians();
            let e: f64 = self
                .harmonics
                .iter()
                .map(|&(n, v, g)| 0.5 * v * (1.0 + (n * phi - g).cos()))
                .sum();
            lo = lo.min(e);
            hi = hi.max(e);
        }
        if self.harmonics.is_empty() {
            0.0
        } else {
            hi - lo
        }
    }
}

/// UFF の sp3 ねじれ係数 `V`（kcal/mol）。表に無い元素は炭素の値で代用する。
fn uff_v_sp3(el: &str) -> f64 {
    match el {
        "C" => 2.119,
        "N" => 0.450,
        "O" => 0.018,
        "S" => 0.484,
        "P" => 2.400,
        "Si" => 1.225,
        "Se" => 0.330,
        "F" | "Cl" | "Br" | "I" => 0.0, // 1 配位なので回転結合の中心にはならない
        _ => 2.119,
    }
}

/// UFF の sp2 ねじれ係数 `U`（kcal/mol）。
fn uff_u_sp2(el: &str) -> f64 {
    match el {
        "C" => 2.0,
        "N" => 2.0,
        "O" => 2.0,
        "S" => 8.0,
        "P" => 2.4,
        "Si" => 1.225,
        _ => 2.0,
    }
}

fn is_group16(el: &str) -> bool {
    matches!(el, "O" | "S" | "Se" | "Te")
}

/// 符号付き二面角 p-a-b-q（rad、−π..π）。
pub fn dihedral(p: &Vec3, a: &Vec3, b: &Vec3, q: &Vec3) -> f64 {
    let b1 = a - p;
    let b2 = b - a;
    let b3 = q - b;
    let n1 = b1.cross(&b2);
    let n2 = b2.cross(&b3);
    let m = n1.cross(&b2.normalize());
    let x = n1.dot(&n2);
    let y = m.dot(&n2);
    (-y).atan2(x)
}

/// 全原子の共有結合隣接（水素を含む）。参照座標と共有結合半径から決める
/// （`build_intra_pairs` と同じ 1.1 倍の規則）。
pub fn all_atom_adjacency(mol: &Molecule, ref_coords: &[Vec3]) -> Vec<Vec<usize>> {
    let n = ref_coords.len();
    let mut adj = vec![Vec::new(); n];
    for a in 0..n {
        for b in (a + 1)..n {
            let ea = mol.atoms[a].ad.element();
            let eb = mol.atoms[b].ad.element();
            let r = distance(&ref_coords[a], &ref_coords[b]);
            if r < (ea.covalent_radius() + eb.covalent_radius()) * 1.1 {
                adj[a].push(b);
                adj[b].push(a);
            }
        }
    }
    adj
}

/// 元素記号（AutoDock 型から。`A` は芳香族炭素）。
fn element_of(mol: &Molecule, i: usize) -> String {
    format!("{:?}", mol.atoms[i].ad.element())
}

/// 混成を参照座標の形から決める。
///
/// - 配位数 4 以上 → sp3
/// - 配位数 3 → 中心が 3 つの隣接の作る平面上に乗っていれば sp2、そうでなければ sp3
/// - 配位数 2 → O・S は sp3、それ以外は結合角 155° 超なら sp、以下なら sp2
/// - 配位数 1 以下 → sp3（回転結合の中心にはならない）
fn hybridization(mol: &Molecule, ref_coords: &[Vec3], adj: &[Vec<usize>], i: usize) -> Hyb {
    let nb = &adj[i];
    match nb.len() {
        0 | 1 => Hyb::Sp3,
        2 => {
            if is_group16(&element_of(mol, i)) {
                return Hyb::Sp3;
            }
            let u = (ref_coords[nb[0]] - ref_coords[i]).normalize();
            let v = (ref_coords[nb[1]] - ref_coords[i]).normalize();
            let ang = u.dot(&v).clamp(-1.0, 1.0).acos().to_degrees();
            if ang > 155.0 {
                Hyb::Sp
            } else {
                Hyb::Sp2
            }
        }
        3 => {
            // 中心から 3 隣接の平面までの距離。平面なら sp2。
            let a = ref_coords[nb[0]];
            let b = ref_coords[nb[1]];
            let c = ref_coords[nb[2]];
            let nrm = (b - a).cross(&(c - a));
            if nrm.norm() < 1e-9 {
                return Hyb::Sp2;
            }
            let d = (ref_coords[i] - a).dot(&nrm.normalize()).abs();
            if d < 0.25 {
                Hyb::Sp2
            } else {
                Hyb::Sp3
            }
        }
        _ => Hyb::Sp3,
    }
}

/// 原子 `a` が環に属するか。`a` の辺を 1 本外しても `a` とその相手が連結なら環結合で、
/// 環結合を 1 本でも持つ原子は環に属する。回転結合そのものは環結合にならない。
pub fn in_ring(adj: &[Vec<usize>], a: usize) -> bool {
    for &b in &adj[a] {
        // 辺 (a,b) を外した上で a から b に到達できるか（幅優先）。
        let mut seen = vec![false; adj.len()];
        let mut q = std::collections::VecDeque::new();
        seen[a] = true;
        q.push_back(a);
        while let Some(u) = q.pop_front() {
            for &v in &adj[u] {
                if (u == a && v == b) || (u == b && v == a) {
                    continue;
                }
                if v == b {
                    return true;
                }
                if !seen[v] {
                    seen[v] = true;
                    q.push_back(v);
                }
            }
        }
    }
    false
}

/// 原子 `x` が属する**最小環**の大きさと、その環上での `x` の 2 つの隣接（オルト位に相当）。
/// 環に属さなければ `None`。
///
/// `x` を取り除いたグラフで `x` の隣接同士の最短路を探し、`経路の辺数 + 1` を環の大きさとする。
pub fn smallest_ring(adj: &[Vec<usize>], x: usize) -> Option<(usize, Vec<usize>, usize, usize)> {
    let nb = &adj[x];
    let mut best: Option<(usize, Vec<usize>, usize, usize)> = None;
    for i in 0..nb.len() {
        for j in (i + 1)..nb.len() {
            let (u, v) = (nb[i], nb[j]);
            // x を通らない u→v の最短路（幅優先）。
            let mut prev = vec![usize::MAX; adj.len()];
            let mut seen = vec![false; adj.len()];
            seen[u] = true;
            seen[x] = true;
            let mut q = std::collections::VecDeque::new();
            q.push_back(u);
            let mut found = false;
            while let Some(w) = q.pop_front() {
                if w == v {
                    found = true;
                    break;
                }
                for &z in &adj[w] {
                    if !seen[z] {
                        seen[z] = true;
                        prev[z] = w;
                        q.push_back(z);
                    }
                }
            }
            if !found {
                continue;
            }
            // 経路を復元して環の原子列を作る。
            let mut path = vec![v];
            let mut w = v;
            while w != u {
                w = prev[w];
                path.push(w);
            }
            let size = path.len() + 1; // x を足す
            if best.as_ref().map_or(true, |b| size < b.0) {
                let mut ring = path.clone();
                ring.push(x);
                best = Some((size, ring, u, v));
            }
        }
    }
    best
}

/// 結合 (a,b) の**オルト位に付いている重原子の数**（両側の合計）。
///
/// 各側について、その原子が属する最小環上の 2 つの隣接（オルト位）を取り、
/// それぞれに「環の外の重原子」が付いていれば 1 と数える。環に属さない側は 0。
fn count_ortho_heavy(mol: &Molecule, adj: &[Vec<usize>], a: usize, b: usize) -> usize {
    let mut n = 0;
    for (x, _other) in [(a, b), (b, a)] {
        let Some((_, ring, u, v)) = smallest_ring(adj, x) else { continue };
        for o in [u, v] {
            let has_sub = adj[o].iter().any(|&z| {
                !mol.atoms[z].ad.is_hydrogen() && z != x && !ring.contains(&z)
            });
            if has_sub {
                n += 1;
            }
        }
    }
    n
}

/// 結合 (a,b) がアミド様か。片方が sp2 の窒素で、相手の sp2 炭素が末端の酸素
/// （カルボニル）を持つ場合。アミド結合は本来ほぼ平面なので、ビアリールとは区別する。
fn is_amide_like(
    mol: &Molecule,
    ref_coords: &[Vec3],
    adj: &[Vec<usize>],
    a: usize,
    b: usize,
) -> bool {
    for (n, c) in [(a, b), (b, a)] {
        if element_of(mol, n) != "N" || hybridization(mol, ref_coords, adj, n) != Hyb::Sp2 {
            continue;
        }
        if element_of(mol, c) != "C" || hybridization(mol, ref_coords, adj, c) != Hyb::Sp2 {
            continue;
        }
        // c に末端酸素（重原子隣接が c だけ）が付いているか。
        let has_carbonyl = adj[c].iter().any(|&o| {
            element_of(mol, o) == "O"
                && adj[o].iter().filter(|&&x| !mol.atoms[x].ad.is_hydrogen()).count() == 1
        });
        if has_carbonyl {
            return true;
        }
    }
    false
}

/// 結合 (a,b) の両側から二面角の基準原子を選ぶ。重原子を優先し、同じなら小さい添字。
fn pick_reference(mol: &Molecule, adj: &[Vec<usize>], a: usize, b: usize) -> Option<usize> {
    let mut heavy = None;
    let mut any = None;
    for &v in &adj[a] {
        if v == b {
            continue;
        }
        if any.is_none() {
            any = Some(v);
        }
        if !mol.atoms[v].ad.is_hydrogen() && heavy.is_none() {
            heavy = Some(v);
        }
    }
    heavy.or(any)
}

/// torsion ごとの UFF 係数を決める（`sign` と `phi0` は呼び出し側が入れる）。
fn classify(
    mol: &Molecule,
    ref_coords: &[Vec3],
    adj: &[Vec<usize>],
    a: usize,
    b: usize,
    moves_heavy: bool,
) -> (TorsionKind, Vec<(f64, f64, f64)>) {
    let ha = hybridization(mol, ref_coords, adj, a);
    let hb = hybridization(mol, ref_coords, adj, b);
    let ea = element_of(mol, a);
    let eb = element_of(mol, b);
    // 回転結合は非環状の単結合なので結合次数は 1（ln 1 = 0 ＝ 補正なし）。
    let bo_factor = 1.0_f64;

    let kind_base = match (ha, hb) {
        (Hyb::Sp, _) | (_, Hyb::Sp) => return (TorsionKind::None, Vec::new()),
        (Hyb::Sp3, Hyb::Sp3) => {
            if is_group16(&ea) && is_group16(&eb) {
                let v = if ea == "S" || eb == "S" { 6.8 } else { 2.0 };
                return (
                    TorsionKind::Sp3Sp3Group16,
                    vec![(2.0, v, std::f64::consts::FRAC_PI_2)],
                );
            }
            let v = (uff_v_sp3(&ea) * uff_v_sp3(&eb)).sqrt();
            (TorsionKind::Sp3Sp3, vec![(3.0, v, 0.0)])
        }
        (Hyb::Sp2, Hyb::Sp2) => {
            let v = 5.0 * (uff_u_sp2(&ea) * uff_u_sp2(&eb)).sqrt() * bo_factor;
            // **アミド様だけは別種別にする。** sp2–sp2 全体では UFF の平面則は結晶構造と
            // 合わないが、アミド様に限れば合う（段 95 の 2 集合の実測）。係数は UFF のまま。
            if is_amide_like(mol, ref_coords, adj, a, b) {
                (TorsionKind::AmideLike, vec![(2.0, v, std::f64::consts::PI)])
            } else {
                (TorsionKind::Sp2Sp2, vec![(2.0, v, std::f64::consts::PI)])
            }
        }
        (Hyb::Sp2, Hyb::Sp3) | (Hyb::Sp3, Hyb::Sp2) => {
            let (sp2, sp3, e2, e3) = if ha == Hyb::Sp2 {
                (a, b, ea.clone(), eb.clone())
            } else {
                (b, a, eb.clone(), ea.clone())
            };
            if is_group16(&e3) {
                let v = 5.0 * (uff_u_sp2(&e2) * uff_u_sp2(&e3)).sqrt() * bo_factor;
                (
                    TorsionKind::Sp2Sp3Group16,
                    vec![(2.0, v, std::f64::consts::PI)],
                )
            } else {
                // sp2 側がさらに別の sp2 と結合していれば共役（UFF の例外則）。
                let conj = adj[sp2].iter().any(|&v| {
                    v != sp3 && hybridization(mol, ref_coords, adj, v) == Hyb::Sp2
                });
                if conj {
                    (
                        TorsionKind::Sp2Sp3Conjugated,
                        vec![(3.0, 2.0, std::f64::consts::PI)],
                    )
                } else {
                    (TorsionKind::Sp2Sp3, vec![(6.0, 1.0, std::f64::consts::PI)])
                }
            }
        }
    };
    if moves_heavy {
        kind_base
    } else {
        // **水素しか動かない自由度は寄与 0 にする。**
        // エンジンは水素に XS 型を与えず全項から除外するので、この自由度を回しても
        // 分子間項・分子内項は一切変わらない。そこへねじれポテンシャルだけを足すと、
        // 目的関数に**物理的な対応物が無い構造**を持ち込むことになる（例: 水酸基の
        // 回転角が親和性を変えてしまう）。正しい対処はこの自由度を探索空間から
        // 外すことで、それは A-2 とは別の話（段 91 で A-4 として記録）。
        (TorsionKind::NoHeavyAtomMoves, Vec::new())
    }
}

/// `LigandModel` 構築後に呼び、torsion ごとのポテンシャルを作る。
///
/// `world_of` は「`torsions[k] = t` にしたときの全原子 world 座標」を返す関数
/// （`LigandModel::world_coords` を使う。向きの規約に依存しないよう実測で符号を決める）。
pub fn build_terms<F>(
    mol: &Molecule,
    ref_coords: &[Vec3],
    bonds: &[Option<(usize, usize)>],
    n_torsions: usize,
    moves_scored: &[bool],
    mut world_of: F,
) -> Vec<TorsionTerm>
where
    F: FnMut(usize, f64) -> Vec<Vec3>,
{
    let adj = all_atom_adjacency(mol, ref_coords);
    let mut out = Vec::with_capacity(n_torsions);
    const DELTA: f64 = 0.01;
    for k in 0..n_torsions {
        let bond = bonds.get(k + 1).and_then(|b| *b);
        let Some((a, b)) = bond else {
            out.push(TorsionTerm {
                phi0: 0.0,
                sign: 1.0,
                harmonics: Vec::new(),
                kind: TorsionKind::None,
                detail: None,
            });
            continue;
        };
        let (Some(p), Some(q)) = (
            pick_reference(mol, &adj, a, b),
            pick_reference(mol, &adj, b, a),
        ) else {
            out.push(TorsionTerm {
                phi0: 0.0,
                sign: 1.0,
                harmonics: Vec::new(),
                kind: TorsionKind::None,
                detail: None,
            });
            continue;
        };
        let w0 = world_of(k, 0.0);
        let w1 = world_of(k, DELTA);
        // 採点対象の重原子が動くか（A-4 の判定と同じものを使う。定義を 1 つに保つ）。
        let moves_heavy = moves_scored.get(k).copied().unwrap_or(true);
        let phi0 = dihedral(&w0[p], &w0[a], &w0[b], &w0[q]);
        let phi1 = dihedral(&w1[p], &w1[a], &w1[b], &w1[q]);
        let mut d = phi1 - phi0;
        while d > std::f64::consts::PI {
            d -= 2.0 * std::f64::consts::PI;
        }
        while d < -std::f64::consts::PI {
            d += 2.0 * std::f64::consts::PI;
        }
        // 二面角が動かない（基準原子の取り方が縮退している）場合は寄与 0 にする。
        let sign = if d.abs() < DELTA * 0.5 {
            0.0
        } else if d > 0.0 {
            1.0
        } else {
            -1.0
        };
        let (kind, harmonics) = classify(mol, ref_coords, &adj, a, b, moves_heavy);
        let hy = (
            hybridization(mol, ref_coords, &adj, a),
            hybridization(mol, ref_coords, &adj, b),
        );
        let hdeg = |x: usize| adj[x].iter().filter(|&&v| !mol.atoms[v].ad.is_hydrogen()).count();
        let detail = TorsionDetail {
            atoms: (a, b),
            refs: (p, q),
            elements: (element_of(mol, a), element_of(mol, b)),
            hyb: hy,
            in_ring: (in_ring(&adj, a), in_ring(&adj, b)),
            heavy_degree: (hdeg(a), hdeg(b)),
            amide_like: is_amide_like(mol, ref_coords, &adj, a, b),
            ring_size: (
                smallest_ring(&adj, a).map_or(0, |r| r.0),
                smallest_ring(&adj, b).map_or(0, |r| r.0),
            ),
            ring_all_carbon: (
                smallest_ring(&adj, a).map_or(false, |r| {
                    r.1.iter().all(|&z| element_of(mol, z) == "C")
                }),
                smallest_ring(&adj, b).map_or(false, |r| {
                    r.1.iter().all(|&z| element_of(mol, z) == "C")
                }),
            ),
            ortho_heavy: count_ortho_heavy(mol, &adj, a, b),
        };
        out.push(TorsionTerm {
            phi0,
            sign,
            harmonics: if sign == 0.0 { Vec::new() } else { harmonics },
            kind,
            detail: Some(detail),
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dihedral_signs_and_values() {
        // エタン型: p-a-b-q を 0°（重なり）と 180°（逆方向）で確かめる。
        let p = Vec3::new(1.0, 0.0, 0.0);
        let a = Vec3::new(0.0, 0.0, 0.0);
        let b = Vec3::new(0.0, 0.0, 1.5);
        let q_cis = Vec3::new(1.0, 0.0, 1.5);
        let q_trans = Vec3::new(-1.0, 0.0, 1.5);
        assert!(dihedral(&p, &a, &b, &q_cis).abs() < 1e-9);
        assert!((dihedral(&p, &a, &b, &q_trans).abs() - std::f64::consts::PI).abs() < 1e-9);
    }

    #[test]
    fn sp3_sp3_carbon_barrier_matches_uff() {
        // sqrt(2.119 * 2.119) = 2.119 kcal/mol の 3 回対称。障壁はちょうど V。
        let t = TorsionTerm {
            phi0: 0.0,
            sign: 1.0,
            harmonics: vec![(3.0, 2.119, 0.0)],
            detail: None,
            kind: TorsionKind::Sp3Sp3,
        };
        assert!((t.barrier() - 2.119).abs() < 1e-3);
        // φ=0（重なり）が最大、φ=60°（ねじれ）が最小。
        let (e0, _) = t.value_grad(0.0);
        let (e60, _) = t.value_grad(60f64.to_radians());
        assert!(e0 > e60 + 2.0, "e0={e0} e60={e60}");
        assert!(e60.abs() < 1e-9);
    }

    #[test]
    fn gradient_matches_finite_difference() {
        let t = TorsionTerm {
            phi0: 0.37,
            sign: -1.0,
            harmonics: vec![(2.0, 10.0, std::f64::consts::PI), (3.0, 2.0, 0.0)],
            detail: None,
            kind: TorsionKind::Sp3Sp3,
        };
        for x in [-1.3, -0.2, 0.0, 0.5, 2.4] {
            let (_, g) = t.value_grad(x);
            let h = 1e-6;
            let (ep, _) = t.value_grad(x + h);
            let (em, _) = t.value_grad(x - h);
            let fd = (ep - em) / (2.0 * h);
            assert!((g - fd).abs() < 1e-5, "x={x} g={g} fd={fd}");
        }
    }

    #[test]
    fn unvalidated_kinds_contribute_nothing() {
        // sp2–sp2 は結晶構造の検証に落ちた種別。係数は持つが評価されない。
        let t = TorsionTerm {
            phi0: 0.0,
            sign: 1.0,
            harmonics: vec![(2.0, 10.0, std::f64::consts::PI)],
            detail: None,
            kind: TorsionKind::Sp2Sp2,
        };
        assert!(!t.is_active());
        assert_eq!(t.value_grad(0.0), (0.0, 0.0));
        assert_eq!(t.value_grad(1.3), (0.0, 0.0));
        // 診断用の障壁は係数そのものを見るので 0 ではない（記録に残すため）。
        assert!((t.barrier() - 10.0).abs() < 1e-3);
    }

    #[test]
    fn empty_harmonics_contribute_nothing() {
        let t = TorsionTerm {
            phi0: 1.0,
            sign: 1.0,
            harmonics: Vec::new(),
            detail: None,
            kind: TorsionKind::None,
        };
        assert_eq!(t.value_grad(0.7), (0.0, 0.0));
        assert_eq!(t.barrier(), 0.0);
    }

    /// アミド様の結合が有効な種別として分類され、平面が最小になる。
    #[test]
    fn amide_like_bond_is_classified_and_planar() {
        // N-メチルアセトアミド。回転結合は C2(カルボニル)-N4。全原子を z=0 の平面に置く。
        const LIG: &str = "\
ROOT
ATOM      1  C1  LIG A   1      -1.500   0.000   0.000  0.00  0.00    +0.000 C
ATOM      2  C2  LIG A   1       0.000   0.000   0.000  0.00  0.00    +0.000 C
ATOM      3  O3  LIG A   1       0.600   1.100   0.000  0.00  0.00    -0.300 OA
ENDROOT
BRANCH   2   4
ATOM      4  N4  LIG A   1       0.650  -1.150   0.000  0.00  0.00    -0.300 N
ATOM      5  HN  LIG A   1       0.150  -2.000   0.000  0.00  0.00    +0.200 HD
ATOM      6  C5  LIG A   1       2.100  -1.250   0.000  0.00  0.00    +0.000 C
ENDBRANCH   2   4
TORSDOF 1
";
        let mol = ist_io::pdbqt::parse_str(LIG).unwrap();
        let mut lig = crate::ligand::LigandModel::new(&mol);
        assert_eq!(lig.n_torsions(), 1);
        let t = &lig.torsion_terms()[0];
        assert_eq!(t.kind, TorsionKind::AmideLike, "アミド様に分類されること");
        assert!(t.is_active(), "アミド様は検証を通った種別なので評価される");
        // UFF の sp2–sp2（C と N はどちらも U=2.0）→ V = 5*sqrt(2*2) = 10、2 回対称。
        assert!((t.barrier() - 10.0).abs() < 1e-2, "障壁 {}", t.barrier());
        // 入力は平面なので、結晶配座の歪みはほぼ 0。
        let (strain, dmin) = t.strain(0.0);
        assert!(strain < 0.05, "平面の入力で歪み {strain}");
        assert!(dmin < 5.0, "最小までの角度 {dmin}");
        // 90° 回すと最大の罰則（≈ V）。
        let (e90, _) = t.value_grad(std::f64::consts::FRAC_PI_2);
        let (e0, _) = t.value_grad(0.0);
        assert!(e90 - e0 > 9.0, "90° の罰則 {}", e90 - e0);
        // 係数 0（既定）なら寄与は厳密に 0。
        let conf = crate::ligand::Conformation::identity(1);
        assert_eq!(lig.torsional(&conf), 0.0);
        lig.set_torsion_scale(1.0);
        assert!(lig.torsional(&conf) < 0.05);
    }
}
