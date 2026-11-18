//! Vina 互換スコアリング関数（Trott & Olson 2010 / Vina 1.2 "vina" 重み）。
//!
//! 原子ペア i,j、実距離 `r`、表面間距離 `d = r - (R_i + R_j)`、カットオフ `r < 8 Å`:
//!   gauss1      exp(-(d/0.5)^2)
//!   gauss2      exp(-((d-3)/2)^2)
//!   repulsion   d<0 ? d^2 : 0
//!   hydrophobic 区分線形（d<=0.5:1 → d>=1.5:0）、疎水-疎水ペアのみ
//!   hbond       区分線形（d<=-0.7:1 → d>=0:0）、ドナー-アクセプタペアのみ
//! 重み付き総和が分子間エネルギー e_inter。最終スコアは構造補正
//! `e_inter / (1 + w_rot * N_rot)`、`w_rot = 0.05846`。
//!
//! 半径の和 `R_i + R_j` は `pair_sum_radii` が**唯一の窓口**で、金属が受容原子と対になる
//! ときだけ `metal_coord_radius` で差し替える（段 110・111）。既定 1.2 では Vina と
//! ビット一致するので、この窓口があっても既定の挙動は変わらない。
//!
//! 検証ゲート: `vina --score_only` の "Ligand - Receptor" 値との一致（差 < 1e-2）。

use crate::atom_types::XsType;
use crate::geometry::{distance, Vec3};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// 各項の重み（Vina 1.2 "vina" スコアリング）。
pub const W_GAUSS1: f64 = -0.035579;
pub const W_GAUSS2: f64 = -0.005156;
pub const W_REPULSION: f64 = 0.840245;
pub const W_HYDROPHOBIC: f64 = -0.035069;
pub const W_HBOND: f64 = -0.587439;
/// 構造補正（torsion）係数。
pub const W_ROT: f64 = 0.05846;

/// ペア相互作用のカットオフ（実距離 r、Å）。
pub const CUTOFF: f64 = 8.0;

/// 金属（XS 型 `MetalD`）が**水素結合の受容原子と対になるとき**に使う実効半径（Å）。
/// 既定値は `XsType::MetalD.radius()` と同じ 1.2 で、このとき AutoDock Vina と完全に同一。
///
/// # 何を直しているか（段 110 で診断、段 112 で実測）
///
/// Vina では反発が `0.840245·d²` で上限なく伸びるのに、水素結合項は `d ≤ −0.7` で
/// `−0.587439` に**飽和する**。そのため **表面距離が −0.836 Å より短い接触は、
/// 水素結合が可能であっても必ず正味で罰則になる**。亜鉛配位はちょうどこの領域にある:
/// Zn 半径 1.2 ＋ 窒素 1.8 = 3.0 に対して実際の配位距離は中央 1.93 Å
/// （段 86.3、489 リガンド中 390 ペア）なので d = −1.07、反発 +0.962 と水素結合 −0.587 で
/// **正味 +0.374 kcal/mol の罰則**になる。結晶ポーズの炭酸脱水酵素 II は 85.9% の
/// リガンドがこれで損をしている。
///
/// # ★ 支配している量は反発である（段 112.2 の対照で確定）
///
/// 金属の水素結合**だけ**を外す対照（`metal_hbond`）を走らせると成績は**悪化**した
/// （交差ドッキングの炭酸脱水酵素 II 439 リガンドで 100 → 87 本）。つまり水素結合項は
/// 現に役に立っており、この半径の効果は「見かけの水素結合を消すこと」では**ない**。
/// 半径の効果は水素結合の有無にほぼ依らず加算的だった（+22 対 +23 本）。
/// **半径は反発を通して効いている。**
///
/// # ★ 値に意味があるのは「反発が消えるか」だけで、報酬の深さは効かない
///
/// 段 112 で 1.2 から 0.0 まで 8 点を測ったところ、**0.6 以下は台地**だった
/// （成功数 118 / 122 / 118 / 120 で、0.4 対 0.0 の直接対比は 7 勝 9 敗）。
/// ところがペア 1 個のエネルギーは台地の中で大きく変わる ——
/// 亜鉛–窒素 1.93 Å の正味は x=0.6 で −0.224、x=0.4 で −0.192、x=0.13 で −0.036 と
/// **6 倍の幅**がある。それでも成績は動かない。
///
/// つまり**ポーズの弁別に効くのは配位距離での反発が消えることだけで、
/// 報酬がどれだけ深いかは効かない。** 反発は x=1.2 で +0.962、x=0.6 で +0.186、
/// x=0.4 で +0.061 なので、**およそ 0.2 kcal/mol を下回ると飽和する。**
///
/// したがって**このパラメータは連続的に調整するものではない。**
/// 台地の中から 1 点選べばよく、その推奨値は [`RECOMMENDED_METAL_COORD_RADIUS`]。
///
/// # 外した予測を 2 つ残しておく（同じ誤りを繰り返さないため）
///
/// - 段 110.5 で「段 86.3 の実測から罰則が消える条件は 0.97 以下」と値の根拠を立てたが、
///   **成績の上では意味が無かった**。0.97 には何の境界も無い
/// - 段 111.5 で「0.8 より下は水素結合項が飽和域から出て報酬を失うので**内部に最適がある**」
///   と予測したが**外れた**。報酬の深さが効かないので、報酬を失っても成績は落ちない
///
/// # 標本の限界
///
/// 測ったのは**炭酸脱水酵素 II 1 標的だけ**で、しかも段 86 以降この問題の診断に
/// 使ってきた標的である（＝探索に使った標本）。他の金属（Mg・Mn・Fe）や他の
/// 金属酵素では最適が違う可能性があるので、**半径は連続パラメータのまま残してある**
/// （入切の真偽値にすると測り直せなくなる）。
///
/// # ★ この値は亜鉛でしか校正していない（段 113.4）
///
/// 規則は XS 型 `MetalD` の**すべての金属に同じ値を適用する**。しかし校正したのは
/// 炭酸脱水酵素 II の亜鉛だけである（段 86.3 の実測も亜鉛–窒素 1.93 Å n=390、
/// 亜鉛–酸素 2.07 Å n=28 で亜鉛のみ）。鉄・マグネシウム・マンガンの配位距離は
/// 亜鉛と違うので、**他の金属に推奨値を当てるのは外挿である。**
///
/// 段 113 では**リガンド側に鉄を含むリガンド 2 本**（BRD4 の L8W、炭酸脱水酵素 II の 84）に
/// 実際に当たっているが、2 本では当たり外れを判定できない。
///
/// 金属ごとに半径を分けるのが筋だが、Vina の XS 型は金属を 1 種類（`MetalD`、半径 1.2）に
/// まとめているので、**型の分割が先に必要**になる。パラメータを増やす前に、
/// 標本外の金属標的で効果を確かめるべきである。
///
/// # 規則は受容体側とリガンド側の両方の金属に効く
///
/// `MetalD` がリガンドにあっても同じ規則が働く（分子内ペアにも効く）。
/// 段 113.2 で、鉄を含むリガンド 1 本について**スカラ・SIMD・格子・分子内の 4 経路すべて**で
/// 結果が動くことを確認した。金属を含まない 765 リガンドでは 1 件も動かない。
pub const DEFAULT_METAL_COORD_RADIUS: f64 = 1.2;

/// 金属の配位時実効半径の**推奨値**（Å）。既定ではない（既定は Vina の 1.2）。
///
/// 段 112 の交差ドッキング（炭酸脱水酵素 II 439 リガンド）で最良だった点。
/// 成功率 22.8% → 27.8%（+5.0 ポイント）、上位 1 位の対称補正 RMSD の中央が
/// 4.026 → 3.329 Å、35 勝 13 敗。
///
/// 0.0〜0.6 はどこを取ってもほぼ同じ成績なので（上の台地の説明）、この値は
/// 「台地の中の 1 点」として選んでいる。選んだ理由は 2 つ:
///
/// - 測った 8 点の中で最良だった（成功 122 本）
/// - 実測の配位距離がガウス 1 項の最適（表面距離 0）に近く、かつ反発がほぼ 0 になる。
///   酸素の最適は x=0.37、窒素の最適は x=0.13 なので、0.4 は**酸素側の最適**に当たり、
///   窒素でも反発は +0.061 まで落ちている
///
/// **既定値をこれに変えてはいない。** 標本外（別の金属標的）での確認が済んでいないため。
pub const RECOMMENDED_METAL_COORD_RADIUS: f64 = 0.4;

/// 金属の配位時実効半径の表（金属が亜鉛か否か × 相手が窒素か酸素か の 4 組）。
///
/// # なぜ 4 組に分けたか（段 115）
///
/// 段 112 では 1 つの値を全金属・全相手に当てていたが、**それは誤りだった。**
/// 段 115 の全数調査（分解能 2.0 Å 以下の PDB 7,786 構造、金属–リガンド配位 24,692 件）で、
/// 受容原子を相手にする 10 組の最適半径が **0.65〜1.15 Å** にばらつくと分かった。
/// 幅 0.50 Å なので単一の値では足りない。段 114 では実際に、炭酸脱水酵素 II で最良だった
/// 0.4 を MMP12（Zn–O）とアルギナーゼ 1（Mn–O）に当てると**結晶ポーズが悪化した**
/// （46 本中 28 本、23 本中 17 本）。
///
/// # ★ 既定で罰則になるのは亜鉛–窒素だけ
///
/// | 組 | 件数 | 配位距離 中央 | 既定（1.2）の正味 | 罰則の割合 |
/// |---|---:|---:|---:|---:|
/// | **Zn–N** | 1,640 | **2.034** | **+0.196** | **62%** |
/// | Mg–O | 6,011 | 2.103 | −0.040 | 36% |
/// | Zn–O | 3,564 | 2.295 | −0.036 | 22% |
/// | Mn–O | 2,108 | 2.188 | −0.153 | 13% |
/// | Ca–O | 2,004 | 2.434 | −0.214 | 1% |
///
/// 10 組のうち既定で正味が罰則なのは **Zn–N の 1 組だけ**で、他は既に報酬側にいる。
/// Ca–O の最適半径は 1.199 ＝ **Vina の既定 1.2 とほぼ一致**する（既定はカルシウム–酸素には
/// ちょうど良く、亜鉛–窒素には 0.5 Å 大きすぎる）。
///
/// そのため [`RECOMMENDED_METAL_RADII`] は**亜鉛–窒素だけ**を変える。残り 3 組を
/// 未測定のまま変えるのは、段 114 で実測した害を入れる危険がある。
/// 段 124 で 5 欄に広げたが、段 125.10 の標本外確認で亜鉛–窒素だけに差し戻した。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MetalRadii {
    /// 亜鉛が窒素の受容原子（`NitrogenA`/`NitrogenDA`）と対になるときの実効半径。
    pub zn_n: f64,
    /// 亜鉛が酸素の受容原子（`OxygenA`/`OxygenDA`）と対になるとき。
    pub zn_o: f64,
    /// 亜鉛以外の金属が窒素の受容原子と対になるとき。
    pub other_n: f64,
    /// 亜鉛以外の金属が酸素の受容原子と対になるとき。
    pub other_o: f64,
    // ── 段 124: 水素結合項が入らない相手（打ち消しの無い反発）──
    //
    // 段 124.6 の全数調査で、**打ち消しの無い相手が亜鉛の罰則総量の 53% を占める**
    // ことが分かった。受容原子（上の 4 組）より大きい。これらの組では水素結合項が
    // 0 なので正味の最小点が `d = 0` に来るため、最適半径は
    // **配位距離 −（相手の XS 半径）** であり、受容原子の `+ 0.465` が付かない。
    /// 亜鉛が**供与体のみの窒素**（`NitrogenD`/`NitrogenP`、AutoDock 型 `N`）と対になるとき。
    /// 段 124.6: 配位 141 件・中央 2.024 Å・**罰則 100%・正味 +0.800**（全組で最大）。
    pub zn_ndon: f64,
    /// 亜鉛が硫黄と対になるとき。段 124.6: 受容硫黄（`SA`）135 件が中央 2.315 Å で
    /// **罰則 100%・正味 +0.656**。ただし**読み取りが `S` と `SA` を区別していない**ので
    /// この欄は非配位の `S`（805 件・中央 3.040 Å）にも効く。
    pub zn_s: f64,
    /// 亜鉛がフッ素と対になるとき（段 124.6 では罰則なし。表の対称性のために置く）。
    pub zn_f: f64,
    /// 亜鉛が炭素と対になるとき。配位原子の**後ろにある**炭素が受ける反発
    /// （1,022 件・中央 2.861 Å・罰則 62%・正味 +0.019）。
    pub zn_c: f64,
    /// 亜鉛がリンと対になるとき（270 件・中央 2.999 Å・罰則 80%・正味 +0.051）。
    pub zn_p: f64,
    /// 亜鉛以外の金属が供与体のみの窒素と対になるとき。
    pub other_ndon: f64,
    /// 亜鉛以外の金属が硫黄と対になるとき。段 124.6: 鉄–`SA` が 96 件・中央 2.388 Å・
    /// **罰則 99%・正味 +0.552**。
    pub other_s: f64,
    /// 亜鉛以外の金属がフッ素と対になるとき。段 124.6: マグネシウム–フッ素が
    /// 132 件・中央 1.974 Å・**罰則 100%・正味 +0.438**。
    pub other_f: f64,
    /// 亜鉛以外の金属が炭素と対になるとき。**金属種で配位距離が大きく違う**ので
    /// （鉄 2.506 Å に対しマグネシウム・マンガン・コバルトは 2.9〜3.0 Å）、
    /// この 1 欄で揃えるのは粗い。既定のままを推奨する。
    pub other_c: f64,
    /// 亜鉛以外の金属がリンと対になるとき（マンガン 2.879 Å とマグネシウム 3.152 Å が混ざる）。
    pub other_p: f64,
}

impl Default for MetalRadii {
    /// 全組 1.2 ＝ **AutoDock Vina と完全に同一**。
    fn default() -> Self {
        Self::all(DEFAULT_METAL_COORD_RADIUS)
    }
}

impl MetalRadii {
    /// **14 欄すべて**に同じ値を入れる。`all(1.2)` が Vina の既定。
    pub const fn all(x: f64) -> Self {
        MetalRadii {
            zn_n: x, zn_o: x, other_n: x, other_o: x,
            zn_ndon: x, zn_s: x, zn_f: x, zn_c: x, zn_p: x,
            other_ndon: x, other_s: x, other_f: x, other_c: x, other_p: x,
        }
    }

    /// **受容原子の 4 欄だけ**に同じ値を入れ、残りは Vina の既定 1.2 のままにする。
    ///
    /// 段 112〜114 の `--metal-coord-radius` と**厳密に同じ意味**である。当時の
    /// `sum_radii_with` は受容原子の相手にしか半径を適用しなかったので、
    /// 「全組に同じ値」という当時の呼び方は実際には「受容原子の全組」だった。
    /// 段 124 で打ち消しの無い相手にも欄を増やしたため、
    /// **過去の測定を再現するにはこの関数を使う**（`all` ではない）。
    pub const fn acceptors_only(x: f64) -> Self {
        let mut r = Self::all(DEFAULT_METAL_COORD_RADIUS);
        r.zn_n = x;
        r.zn_o = x;
        r.other_n = x;
        r.other_o = x;
        r
    }

    /// 既定（＝ Vina）と完全に一致するか。一致するなら半径の計算を一切触らない。
    #[inline]
    pub fn is_vina_default(&self) -> bool {
        *self == Self::default()
    }

    /// 欄の数（大域の保持と入出力の順序を 1 箇所で決める）。
    pub const N_FIELDS: usize = 14;

    /// 欄を固定順の配列にする（大域への保存用）。順序は [`MetalRadii::from_array`] と対。
    pub const fn to_array(&self) -> [f64; Self::N_FIELDS] {
        [
            self.zn_n, self.zn_o, self.other_n, self.other_o,
            self.zn_ndon, self.zn_s, self.zn_f, self.zn_c, self.zn_p,
            self.other_ndon, self.other_s, self.other_f, self.other_c, self.other_p,
        ]
    }

    /// [`MetalRadii::to_array`] の逆。
    pub const fn from_array(a: [f64; Self::N_FIELDS]) -> Self {
        MetalRadii {
            zn_n: a[0], zn_o: a[1], other_n: a[2], other_o: a[3],
            zn_ndon: a[4], zn_s: a[5], zn_f: a[6], zn_c: a[7], zn_p: a[8],
            other_ndon: a[9], other_s: a[10], other_f: a[11], other_c: a[12], other_p: a[13],
        }
    }

    /// 既定（1.2）でない欄を `"zn_s=0.320"` の形で並べる（報告用。段 124）。
    /// 空文字列なら受容原子以外の欄はすべて既定である。
    pub fn non_default_extras(&self) -> String {
        const NAMES: [&str; MetalRadii::N_FIELDS] = [
            "zn_n", "zn_o", "other_n", "other_o", "zn_ndon", "zn_s", "zn_f", "zn_c", "zn_p",
            "other_ndon", "other_s", "other_f", "other_c", "other_p",
        ];
        let a = self.to_array();
        let mut out = Vec::new();
        // 先頭 4 欄（受容原子）は呼び出し側が常に表示するので省く。
        for i in 4..MetalRadii::N_FIELDS {
            if a[i] != DEFAULT_METAL_COORD_RADIUS {
                out.push(format!("{}={:.3}", NAMES[i], a[i]));
            }
        }
        out.join(" ")
    }

    /// 金属（亜鉛か否か）と相手の種別から実効半径を引く。
    /// 表に欄が無い相手（ハロゲン・ケイ素・金属同士など）は Vina の既定のまま。
    #[inline]
    pub fn radius_for(&self, zn: bool, c: PartnerClass) -> f64 {
        use PartnerClass as P;
        match (zn, c) {
            (true, P::NAcceptor) => self.zn_n,
            (true, P::OAcceptor) => self.zn_o,
            (true, P::NDonor) => self.zn_ndon,
            (true, P::Sulfur) => self.zn_s,
            (true, P::Fluorine) => self.zn_f,
            (true, P::Carbon) => self.zn_c,
            (true, P::Phosphorus) => self.zn_p,
            (false, P::NAcceptor) => self.other_n,
            (false, P::OAcceptor) => self.other_o,
            (false, P::NDonor) => self.other_ndon,
            (false, P::Sulfur) => self.other_s,
            (false, P::Fluorine) => self.other_f,
            (false, P::Carbon) => self.other_c,
            (false, P::Phosphorus) => self.other_p,
            (_, P::Other) => DEFAULT_METAL_COORD_RADIUS,
        }
    }
}

/// 金属の相手の種別（半径の表の列）。段 124 で受容原子以外にも広げた。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PartnerClass {
    /// 窒素の受容原子（AutoDock 型 `NA`）。
    NAcceptor,
    /// 酸素の受容原子（`OA`）。
    OAcceptor,
    /// 供与体のみ・非極性の窒素（`N`）。**水素結合項が入らない**。
    NDonor,
    /// 硫黄（`S` と `SA` の両方。読み取りが区別していない）。水素結合項が入らない。
    Sulfur,
    /// フッ素。水素結合項が入らない。
    Fluorine,
    /// 炭素（疎水・極性の両方）。
    Carbon,
    /// リン。
    Phosphorus,
    /// 表に欄が無いもの（酸素の供与体・ハロゲン・ケイ素・金属など）。既定のまま。
    Other,
}

/// XS 型を半径の表の列に分類する。
#[inline]
pub fn partner_class(t: XsType) -> PartnerClass {
    use PartnerClass as P;
    match t {
        XsType::NitrogenA | XsType::NitrogenDA => P::NAcceptor,
        XsType::OxygenA | XsType::OxygenDA => P::OAcceptor,
        XsType::NitrogenD | XsType::NitrogenP => P::NDonor,
        XsType::Sulfur => P::Sulfur,
        XsType::Fluorine => P::Fluorine,
        XsType::CarbonH | XsType::CarbonP => P::Carbon,
        XsType::Phosphorus => P::Phosphorus,
        _ => P::Other,
    }
}

/// 亜鉛–受容窒素だけ 0.70 にする表。**現在の推奨表**（別名 [`RECOMMENDED_METAL_RADII`]）。
///
/// 段 124 で一度推奨から外し（段 124.15: 選定に使った 3 標的で 0 勝 3 敗）、
/// 段 125.10 の標本外確認で**差し戻した**。差し戻しの根拠は 2 つ。
///
/// 1. 広げた表 [`STAGE124_RADII`] の利得が、欄の選定に使っていない 6 標的で再現しない
///    （Astex 10 標的で 0 勝 2 敗、金属 4 標的で 1 勝 4 敗。通算 1 勝 6 敗）
/// 2. この表は標本外でも無害（金属 4 標的 98 本で 28 本成功・既定 29 本。
///    亜鉛を持たない標的では既定と出力がバイト一致）
///
/// 段 122 で「炭酸脱水酵素以外で利得が無い」と書いたのは今も正しい。この表の利得は
/// 炭酸脱水酵素 II の 439 本で 100 → 122 本という 1 標的に依存している。
/// **利得が小さく無害な表**という位置づけであり、積極的な推奨ではない。
///
/// 0.70 は段 115.3 の予測式 `配位距離 − 相手の XS 半径 + 0.465` から出た値で、
/// 自由パラメータではない（亜鉛–窒素の配位距離の中央 2.034 Å、窒素の XS 半径 1.8 Å、
/// 0.465 は Vina のペア 5 項の正味が最小になる表面距離）。実測した最適値は 0.65 で、
/// 予測との差は 0.05 Å。段 112 の交差ドッキングで最良だった 0.4 ではなく 0.70 にしたのは、
/// 0.4 が「全金属一律」の条件下で選ばれた値であり、亜鉛–窒素に限れば予測式の値が
/// 根拠として強いからである。**どちらが良いかはドッキングで比べる必要がある（未測定）。**
///
/// **これは既定ではない。** 既定は [`MetalRadii::default`]（＝ Vina）。
pub const STAGE122_RADII: MetalRadii = {
    let mut r = MetalRadii::all(DEFAULT_METAL_COORD_RADIUS);
    r.zn_n = 0.70;
    r
};

/// 段 124 の表（段階 1 の案 1）。配位する相手だけを直す。
///
/// **推奨表ではない**（段 124.16 で推奨にしたが段 125.10 で差し戻した。
/// 現在の推奨は [`RECOMMENDED_METAL_RADII`] ＝ [`STAGE122_RADII`]）。
/// 調査を続けるために残してある。下の「段 124 の実測」の節は欄の選定に使った
/// 4 標的の結果で、**標本外では再現しない**。末尾の節を必ず読むこと。
///
/// 段 124.6 の実測（配位殻は「最初の山」で切り出したもの）から、予測式
/// 「配位距離 −（相手の XS 半径）＋（水素結合が入るなら 0.465）」で決めた。
/// 自由パラメータは 1 つも無い。
///
/// | 欄 | 値 | 由来（配位距離・件数・既定の正味） |
/// |---|---:|---|
/// | `zn_n`（受容窒素） | 0.70 | 2.034 Å・1,013 件・+0.290（段 115〜118） |
/// | `zn_ndon`（供与体のみ窒素） | 0.22 | 2.024 Å・141 件・**+0.800** |
/// | `zn_s`（硫黄） | 0.32 | 2.315 Å・135 件・**+0.656**（受容硫黄 `SA`） |
/// | `other_s`（鉄など） | 0.39 | 2.388 Å・96 件・**+0.552** |
/// | `other_f`（マグネシウムなど） | 0.47 | 1.974 Å・132 件・**+0.438** |
///
/// 炭素とリン（配位原子の**後ろ**にある原子）は入れていない。
/// [`STAGE124_RADII_WITH_BACKBONE`] を参照。
///
/// # ドッキングでの実測（段 124.12・124.15。既定に対する成功数、RMSD < 2.0 Å）
///
/// | 標的 | 本数 | 既定 | 段 122 表 | この表 |
/// |---|---:|---:|---:|---:|
/// | 炭酸脱水酵素 II（交差） | 439 | 100 | 122 | **123** |
/// | オートタキシン（自己） | 89 | 70 | 69 | **71** |
/// | VIM-2（自己） | 54 | 17 | 16 | **19** |
/// | HDAC6 CD2（自己） | 29 | 5 | 5 | **7** |
///
/// 炭酸脱水酵素以外の 3 標的で **7 勝 0 敗**。段 122 の表は同じ 3 標的で 0 勝 3 敗だった。
/// `zn_ndon` はオートタキシンと HDAC6、`zn_s` は VIM-2 と炭酸脱水酵素で効く（段 124.15）。
///
/// **`other_s` と `other_f` は幾何（配位距離と罰則の実測）だけが根拠で、
/// ドッキングでの確認が無い**（段 124.11: 非亜鉛金属の標的が作れない）。
///
/// # ★ 標本外での実測（段 125.10）。上の 7 勝 0 敗は再現しない
///
/// 上の 4 標的は**どの欄を使うかの選定に使った標的**である。選定に使っていない標的で
/// 測り直すと、段 122 の表に対する差は次のようになる。
///
/// | 標的（選定に未使用） | 本数 | 既定 | 段 122 表 | この表 | 差 |
/// |---|---:|---:|---:|---:|---|
/// | NDM-1（亜鉛 2 個・新規） | 13 | 2 | 2 | 2 | 0 勝 0 敗 |
/// | イソペニシリン N 合成酵素（鉄・新規） | 16 | 0 | 0 | 0 | 0 勝 0 敗 |
/// | MMP-12（亜鉛） | 46 | 11 | 10 | 8 | 1 勝 3 敗 |
/// | アルギナーゼ 1（マンガン 2 個） | 23 | 16 | 16 | 15 | 0 勝 1 敗 |
/// | Astex ポーズ再現 10 標的 × 10 種 | 100 | 52 | — | 50 | 0 勝 2 敗 |
///
/// 通算 **1 勝 6 敗**。欄の選定が 4 標的に過適合していたと判断した。
/// なお表がエネルギーを 0.05 kcal/mol 以上動かした 21 本では勝敗ゼロで、
/// 負け 5 本のうち 4 本はエネルギーがほぼ変わらない層（探索の揺れ）に入る。
/// つまり**害があるのではなく利得が無い**。ただし次の 1 件は明確な欠陥である。
///
/// `zn_ndon = 0.22` は亜鉛と供与体のみ窒素の半径和を 0.22 + 1.75 = **1.97 Å** に置く。
/// NDM-1 のカルバペネム（CCD `EFR`）で、既定では受容酸素が亜鉛に 2.346 Å で配位する
/// 天然の向きだったのが、この表ではリガンドが裏返って供与体のみ窒素が 2.00 Å・2.30 Å で
/// 亜鉛を掴む向きになり、RMSD が 2.37 → 6.26 Å、親和性は −5.750 → −6.043 と「改善」した。
/// **スコア関数の最小値そのものが非天然のポーズに移っている**（探索の揺れではない）。
/// 仮説（未検証）: 段 124.6 の全 PDB 調査に第 2 配位圏の窒素（隣の酸素が配位している
/// ために亜鉛の近くに来ているだけのアミド窒素）が混入していた。調査側で
/// 「その窒素が第 1 配位圏の最近接原子か」を条件に加えて測り直す必要がある。
///
/// `other_s` は標本外確認でも裏付けが得られなかった。PDB 全体でこの欄を試せる標的は
/// イソペニシリン N 合成酵素だけで、そこは 3 腕すべて 0/16（16 本中 12 本が 5 Å 超）と
/// ドッキングが成立しない。**この欄をドッキングで検証する手段は現時点で存在しない。**
///
/// **これは既定ではない。** 既定は [`MetalRadii::default`]（＝ Vina）。
pub const STAGE124_RADII: MetalRadii = {
    let mut r = MetalRadii::all(DEFAULT_METAL_COORD_RADIUS);
    r.zn_n = 0.70;
    r.zn_ndon = 0.22;
    r.zn_s = 0.32;
    r.other_s = 0.39;
    r.other_f = 0.47;
    r
};

/// **現在の推奨表**（段 125.10 の決定）。中身は [`STAGE122_RADII`]（亜鉛–受容窒素だけ 0.70）。
///
/// 段 124.16 では [`STAGE124_RADII`]（5 欄に広げた表）を推奨にしたが、
/// 段 125.10 の標本外確認で**差し戻した**。広げた 4 欄の利得は、
/// どの欄を使うかの選定に使った 4 標的（炭酸脱水酵素 II・オートタキシン・VIM-2・HDAC6）
/// でしか現れず、選定に使っていない 6 標的では通算 1 勝 6 敗だった。
/// `zn_ndon = 0.22` には具体的な欠陥も見つかっている（NDM-1 の EFR で、
/// 本来配位しないアミド窒素で亜鉛を掴むポーズがスコアで勝ち、RMSD が 2.37 → 6.26 Å）。
///
/// **これは既定ではない。** 既定は [`MetalRadii::default`]（＝ AutoDock Vina）。
pub const RECOMMENDED_METAL_RADII: MetalRadii = STAGE122_RADII;

/// 段 128 の表。[`set_metal_hbond_all`] と**組で使う**。
///
/// # 考え方: 欄を元素で切り、門の方を直す
///
/// 段 124 の 5 欄表（0.22〜0.47）が標本外で再現しなかった原因は 2 つあった。
/// 1 つは測定した母集団の型付けが誤っていたこと（段 126）。
/// もう 1 つは**非受容体の相手では水素結合項が 0 なので、
/// 反発を打ち消すものが無く、必要な半径が極端に小さくなる**ことである。
///
/// 後者は半径では直せない。打ち消しが無いまま半径を縮めても、
/// 得られる報酬はガウス 1 項の −0.036 しかない一方、
/// 立体の床が 0.9 Å 下がって非配位方向からの接近を許す。**費用だけで便益が無い。**
///
/// そこで門（[`set_metal_hbond_all`]）を開ける。すると配位距離での正味は
///
/// | 組 | 配位距離 | 既定 | 門を開く | 門＋この表 |
/// |---|---:|---:|---:|---:|
/// | 亜鉛–窒素（受容） | 1.992 | **+0.266** | +0.266 | **−0.222** |
/// | 亜鉛–窒素（供与体のみ） | 1.992 | **+0.853** | +0.266 | **−0.222** |
/// | 亜鉛–硫黄 | 2.312 | **+0.661** | **+0.073** | +0.073 |
/// | 亜鉛–酸素 | 2.210 | −0.184 | −0.184 | −0.184 |
/// | 他の金属–窒素 | 2.268 | −0.142 | −0.142 | −0.142 |
/// | 他の金属–酸素 | 2.192 | −0.171 | −0.171 | −0.171 |
/// | 他の金属–硫黄 | 2.389 | **+0.550** | **−0.037** | −0.037 |
/// | 他の金属–フッ素 | 1.996 | **+0.411** | **−0.176** | −0.176 |
///
/// **門を開けるだけで硫黄・フッ素・供与体のみ窒素の罰則が消える。
/// 新しいパラメータは 1 つも要らない。**
/// 残るのは亜鉛–窒素だけで、そこは既に標本外で裏づけのある 0.70 を使う
/// （段 127.8: 狙った 74 本で 9 勝 4 敗、揺れの幅は ±0〜+1）。
///
/// # この表の中身
///
/// 亜鉛–窒素の 2 欄（受容・供与体のみ）を 0.70 に揃えるだけ。
/// **揃えるのは同じ元素だから**である。配位距離は受容窒素 1.992 Å・
/// 供与体のみ窒素 1.992 Å で差が無い。段 124 がこの 2 欄に
/// 0.70 と 0.22 という 3 倍違う値を与えていたのは、
/// 水素結合項の有無という型付けの副作用であって、配位の化学ではない。
///
/// 酸素・硫黄・フッ素は**触らない**。門を開けた時点で罰則が消えているので、
/// 半径を動かす理由が無い（段 128.1 の実測では規則が与える値は
/// 亜鉛–酸素 0.98・亜鉛–硫黄 0.78・他の金属–硫黄 0.85・他の金属–フッ素 0.96 だが、
/// どれも正味が既に負なので、動かすのは費用だけで便益が無い）。
/// 炭素・リンも触らない（第 1 配位圏の中央が 2.98 Å で配位していない）。
pub const STAGE128_RADII: MetalRadii = MetalRadii {
    // 亜鉛 × 窒素。受容・供与体のみを**同じ元素だから**同じ値に揃える。
    zn_n: 0.70,
    zn_ndon: 0.70,
    // 残りはすべて既定 1.2（＝ AutoDock Vina）。門を開けた時点で罰則が消えている。
    zn_o: DEFAULT_METAL_COORD_RADIUS,
    zn_s: DEFAULT_METAL_COORD_RADIUS,
    zn_f: DEFAULT_METAL_COORD_RADIUS,
    zn_c: DEFAULT_METAL_COORD_RADIUS,
    zn_p: DEFAULT_METAL_COORD_RADIUS,
    other_n: DEFAULT_METAL_COORD_RADIUS,
    other_ndon: DEFAULT_METAL_COORD_RADIUS,
    other_o: DEFAULT_METAL_COORD_RADIUS,
    other_s: DEFAULT_METAL_COORD_RADIUS,
    other_f: DEFAULT_METAL_COORD_RADIUS,
    other_c: DEFAULT_METAL_COORD_RADIUS,
    other_p: DEFAULT_METAL_COORD_RADIUS,
};

/// 段 124 の表に「配位原子の後ろにある原子」を加えたもの（段階 1 の案 1・第 2 段）。
///
/// 亜鉛–炭素（1,022 件・2.861 Å・+0.019）と亜鉛–リン（270 件・2.999 Å・+0.051）は
/// 1 件あたりは小さいが件数が多く、亜鉛の罰則総量の 11% を占める。ただし
/// **立体の床を下げる**ので、配位しない方向からリガンドが金属に近づけるようになる。
/// **害が出るかは未測定**（段 124.16 の決定 4）。
pub const STAGE124_RADII_WITH_BACKBONE: MetalRadii = {
    let mut r = STAGE124_RADII;
    r.zn_c = 0.96;
    r.zn_p = 0.90;
    r
};

/// 実行時の表（f64 のビット列として欄ごとに保持）。
///
/// スコア関数の定数と同じ性格の量なので、受容体・リガンド・スレッドに依存しない
/// モジュール大域に 1 度だけ書き込む（`intra_lut::table()` と同じ方針）。
/// **`set_metal_radii` はドッキング開始前に 1 度だけ呼ぶこと。**
/// `grid::GridSet`・`intra_lut`・`scoring_simd::ReceptorSimd` はこの値を構築時に
/// 焼き込むので、構築後に変えても反映されない。
static MR: [AtomicU64; MetalRadii::N_FIELDS] =
    [const { AtomicU64::new(DEFAULT_METAL_COORD_RADIUS.to_bits()) }; MetalRadii::N_FIELDS];

/// 表を設定する（ドッキング開始前に 1 度だけ）。
pub fn set_metal_radii(r: MetalRadii) {
    for (slot, x) in MR.iter().zip(r.to_array()) {
        slot.store(x.to_bits(), Ordering::Relaxed);
    }
}

/// 現在の表。
#[inline]
pub fn metal_radii() -> MetalRadii {
    let mut a = [0.0_f64; MetalRadii::N_FIELDS];
    for (x, slot) in a.iter_mut().zip(MR.iter()) {
        *x = f64::from_bits(slot.load(Ordering::Relaxed));
    }
    MetalRadii::from_array(a)
}

/// 段 112〜114 の互換入口: 全組に同じ値を入れる。
pub fn set_metal_coord_radius(x: f64) {
    set_metal_radii(MetalRadii::acceptors_only(x));
}

/// 金属（亜鉛か否か）を判定する。金属でなければ `None`。
#[inline]
pub fn metal_is_zn(t: XsType) -> Option<bool> {
    match t {
        XsType::MetalDZn => Some(true),
        XsType::MetalD => Some(false),
        _ => None,
    }
}

/// 受容原子の種別（窒素か否か）。受容原子でなければ `None`。
/// Vina の型付けで受容原子になるのは窒素と酸素だけ（硫黄・フッ素はならない。段 115.4）。
#[inline]
pub fn acceptor_is_n(t: XsType) -> Option<bool> {
    match t {
        XsType::NitrogenA | XsType::NitrogenDA => Some(true),
        XsType::OxygenA | XsType::OxygenDA => Some(false),
        _ => None,
    }
}

/// ペアの半径の和（`d = r - sum` の `sum`）。**半径の和を決める唯一の窓口。**
///
/// 格子・SIMD・スカラの 3 経路がすべてこの関数（または `sum_radii_with`）を通る。
/// 段 111.2 で、格子構築だけがこれを通らずに規則を取りこぼしていた事故があったので、
/// 格子に渡す引数は「2 つの半径」ではなく「**半径の和 1 つ**」にしてある。
#[inline]
pub fn pair_sum_radii(a: XsType, b: XsType) -> f64 {
    sum_radii_with(a, b, metal_radii())
}

/// `pair_sum_radii` の純関数版（大域状態を読まない）。
/// 表を明示したいとき（試験・診断・SIMD 側の前計算）に使う。
///
/// 金属が対の一方のときだけ、金属側の半径を表の値に差し替える。
/// 相手の種別は [`partner_class`] が決める。表に欄が無い相手（酸素の供与体・ハロゲン・
/// ケイ素・金属同士）では 1.2 のままで、**立体の床を保つ**。
///
/// 段 124 で受容原子以外（供与体のみの窒素・硫黄・フッ素・炭素・リン）にも広げた。
/// それまでは受容原子だけが対象で、打ち消しの無い反発（段 124.6 で亜鉛の罰則総量の
/// 53% と判明）は手つかずだった。
#[inline]
pub fn sum_radii_with(a: XsType, b: XsType, r: MetalRadii) -> f64 {
    let s = a.radius() + b.radius();
    if r.is_vina_default() {
        return s; // 既定（＝Vina）。ビット一致を保つため一切触らない。
    }
    // どちらが金属かを決める。両方が金属なら相手の種別は `Other` になるので
    // どちらを金属と見ても結果は同じ（既定のまま）。
    let (zn, cls) = match metal_is_zn(a) {
        Some(z) => (z, partner_class(b)),
        None => match metal_is_zn(b) {
            Some(z) => (z, partner_class(a)),
            None => return s,
        },
    };
    s - (DEFAULT_METAL_COORD_RADIUS - r.radius_for(zn, cls))
}

/// スコアリング対象の重原子（位置と XS 型のみ）。水素は含めない。
#[derive(Debug, Clone, Copy)]
pub struct ScoredAtom {
    pub pos: Vec3,
    pub xs: XsType,
}

/// 5 項の素値（重み付け前）の和。
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Terms {
    pub gauss1: f64,
    pub gauss2: f64,
    pub repulsion: f64,
    pub hydrophobic: f64,
    pub hbond: f64,
}

impl Terms {
    fn add(&mut self, other: &Terms) {
        self.gauss1 += other.gauss1;
        self.gauss2 += other.gauss2;
        self.repulsion += other.repulsion;
        self.hydrophobic += other.hydrophobic;
        self.hbond += other.hbond;
    }

    /// 重み付き総和（= エネルギー寄与）。
    pub fn weighted_sum(&self) -> f64 {
        W_GAUSS1 * self.gauss1
            + W_GAUSS2 * self.gauss2
            + W_REPULSION * self.repulsion
            + W_HYDROPHOBIC * self.hydrophobic
            + W_HBOND * self.hbond
    }
}

#[inline]
fn gauss(d: f64, offset: f64, width: f64) -> f64 {
    (-((d - offset) / width).powi(2)).exp()
}

#[inline]
fn repulsion(d: f64) -> f64 {
    if d < 0.0 { d * d } else { 0.0 }
}

/// Vina の `slope_step`: `good` で 1、`bad` で 0 の線形ランプ（[0,1] にクランプ）。
#[inline]
fn slope_step(bad: f64, good: f64, x: f64) -> f64 {
    if bad < good {
        if x <= bad {
            0.0
        } else if x >= good {
            1.0
        } else {
            (x - bad) / (good - bad)
        }
    } else if x >= bad {
        0.0
    } else if x <= good {
        1.0
    } else {
        (x - bad) / (good - bad)
    }
}

#[inline]
fn hydrophobic_term(d: f64) -> f64 {
    // good = 0.5, bad = 1.5
    slope_step(1.5, 0.5, d)
}

#[inline]
fn hbond_term(d: f64) -> f64 {
    // good = -0.7, bad = 0.0
    slope_step(0.0, -0.7, d)
}

/// **対照実験用**の切り替え: 金属が関わるペアの水素結合項を無効にする。
/// 既定 `true` ＝ AutoDock Vina と同じ（金属は供与体なので受容原子と水素結合する）。
///
/// # これは機能案ではなく対照である（段 112c）
///
/// 段 112 で金属の実効半径を下げると成績が上がったが、半径を下げると反発と水素結合の
/// **両方**が同時に動くので、どちらが効いたのか分からない。この切り替えは水素結合だけを
/// 外して反発を残すので、両者を切り分けられる。
///
/// 炭酸脱水酵素 II では配位窒素が `NA`（受容原子）と型付けされており（489 本中 392 本）、
/// 水素結合項が罰則を部分的に打ち消している。亜鉛–窒素 2.01 Å で反発 +0.823・
/// 水素結合 −0.587 ＝ 正味 +0.236。これを外すと **+0.823 に悪化する見込み**で、
/// 悪化すれば「配位距離で反発が強すぎる」という機構が確定する。
static METAL_HBOND: AtomicBool = AtomicBool::new(true);

/// 金属が関わるペアの水素結合項を有効にするか（ドッキング開始前に 1 度だけ設定）。
pub fn set_metal_hbond(on: bool) {
    METAL_HBOND.store(on, Ordering::Relaxed);
}

/// 現在の設定。
#[inline]
pub fn metal_hbond() -> bool {
    METAL_HBOND.load(Ordering::Relaxed)
}

/// 金属が関わるペアの水素結合項を、**相手の供与体／受容体の型付けに依らず**
/// 有効にするか（段 128 の提案）。既定 false ＝ AutoDock Vina と同じ。
///
/// # なぜ要るか
///
/// Vina の水素結合項は「供与体 × 受容体」でしか入らない。金属は供与体なので、
/// 相手が受容体でない組（供与体のみの窒素・硫黄・フッ素）では水素結合項が 0 になり、
/// **反発を打ち消すものが無い**。その結果、正味が最小になる表面距離が
/// 受容体相手の −0.465 Å から 0 Å へ移り、必要な実効半径が
/// 「配位距離 − 相手の XS 半径」という極端に小さい値になる
/// （段 124 の表の 0.22〜0.47）。この表は標本外で再現しなかった（段 125.10、1 勝 6 敗）。
///
/// しかし**金属への配位は水素結合ではない**。相手が水素を持つかどうかは
/// 配位の可否を決めない（配位時に脱プロトン化する、孤立電子対が供与される）。
/// 供与体／受容体の門は金属配位に対しては誤ったモデルである。
///
/// この旗を立てると、金属と配位しうる元素（窒素・酸素・硫黄・フッ素）の組は
/// 型付けに依らず水素結合項を持つ。帰結は 3 つ。
///
/// 1. 井戸の深さが全組で揃う（受容体相手の −0.223 と同じ。現在は非受容体で −0.036）
/// 2. 半径の規則が 1 つになる（`配位距離 − 相手の XS 半径 + 0.465`）。
///    段 124 の極端な 4 欄が消え、物理的に妥当な 0.66〜0.98 になる（段 128.1 の実測）
/// 3. リガンドの窒素・酸素の**プロトン化の型付けに依存しなくなる**。
///    段 126 で見つけた型付けの欠陥（イミダゾリウム陽イオン・アゾールの互変異性・
///    脱プロトン化しうるアミド）が金属配位に影響しなくなる
///
/// 炭素・リン・塩素・臭素・ヨウ素は**対象にしない**。炭素は配位しない
/// （第 1 配位圏の中央 2.98 Å ＝ 配位原子の後ろにある非結合接触）。
static METAL_HBOND_ALL: AtomicBool = AtomicBool::new(false);

/// 段 128 の旗を設定する（ドッキング開始前に 1 度だけ）。
pub fn set_metal_hbond_all(on: bool) {
    METAL_HBOND_ALL.store(on, Ordering::Relaxed);
}

/// 現在の設定。
#[inline]
pub fn metal_hbond_all() -> bool {
    METAL_HBOND_ALL.load(Ordering::Relaxed)
}

/// 金属 × 配位しうる元素の組か（段 128）。
///
/// 配位しうる元素は窒素・酸素・硫黄・フッ素。炭素・リン・重ハロゲン・ケイ素・
/// 金属同士は含めない。金属が疎水性炭素を引き寄せるのを防ぐためである。
#[inline]
pub fn metal_coordinating_pair(a: XsType, b: XsType) -> bool {
    (metal_is_zn(a).is_some() && is_coordinating_element(b))
        || (metal_is_zn(b).is_some() && is_coordinating_element(a))
}

/// 金属と配位しうる元素か（段 128 の門の対象）。
///
/// 窒素・酸素・硫黄・フッ素。プロトン化の型付け（供与体／受容体）は見ない。
/// 炭素・リン・重ハロゲン・ケイ素・金属同士は含めない。
///
/// **この関数が門の対象の唯一の定義である。** SIMD 経路（`scoring_simd`）も
/// 受容体の列を作るときにこれを呼ぶ。段 128 の最初の実装では SIMD 経路が
/// 独自に水素結合の係数を組んでいて門に追随せず、旗を立てても黙って無視された。
/// 定義を 2 箇所に置かないことが再発防止である。
#[inline]
pub fn is_coordinating_element(t: XsType) -> bool {
    use XsType::*;
    matches!(
        t,
        NitrogenP | NitrogenD | NitrogenA | NitrogenDA
            | OxygenP | OxygenD | OxygenA | OxygenDA
            | Sulfur | Fluorine
    )
}

/// ペアの疎水係数と水素結合係数（0.0 または 1.0）。
///
/// 5 項のうち「ペアの型で 0/1 が決まる」2 項の係数を**ここ 1 箇所で決める**。
/// スカラ・SIMD・格子の 3 経路がこの関数（または同じ規則）を通るようにしてある。
#[inline]
pub fn pair_factors(a: XsType, b: XsType) -> (f64, f64) {
    let hyd = (a.is_hydrophobic() && b.is_hydrophobic()) as u32 as f64;
    let hb = h_bond_possible(a, b) as u32 as f64;
    (hyd, hb)
}

/// ドナー-アクセプタの組み合わせか（どちら向きでも可）。
#[inline]
fn h_bond_possible(a: XsType, b: XsType) -> bool {
    h_bond_possible_with(a, b, metal_hbond(), metal_hbond_all())
}

/// [`h_bond_possible`] の純関数版（大域状態を読まない）。
///
/// 旗を明示したいとき（試験・診断）に使う。大域の旗を書き換える試験は
/// 並列に走る他の試験と競合するので、判定の試験はこちらで行う。
///
/// 優先順位は「金属の水素結合を全部外す（`metal_hbond = false`）」が最も強く、
/// 次に「型に依らず入れる（`metal_hbond_all = true`、段 128）」、
/// 最後に Vina 本来の供与体 × 受容体である。
#[inline]
pub fn h_bond_possible_with(
    a: XsType,
    b: XsType,
    metal_hbond: bool,
    metal_hbond_all: bool,
) -> bool {
    if !metal_hbond && (metal_is_zn(a).is_some() || metal_is_zn(b).is_some()) {
        return false; // 対照実験: 金属の水素結合だけを外す
    }
    if metal_hbond_all && metal_coordinating_pair(a, b) {
        return true; // 段 128: 金属配位では供与体／受容体の門を使わない
    }
    (a.is_donor() && b.is_acceptor()) || (a.is_acceptor() && b.is_donor())
}

/// 1 ペアの素値を評価する（カットオフ判定は呼び出し側）。
pub fn eval_pair(a: XsType, b: XsType, r: f64) -> Terms {
    let d = r - pair_sum_radii(a, b);
    let mut t = Terms {
        gauss1: gauss(d, 0.0, 0.5),
        gauss2: gauss(d, 3.0, 2.0),
        repulsion: repulsion(d),
        hydrophobic: 0.0,
        hbond: 0.0,
    };
    if a.is_hydrophobic() && b.is_hydrophobic() {
        t.hydrophobic = hydrophobic_term(d);
    }
    if h_bond_possible(a, b) {
        t.hbond = hbond_term(d);
    }
    t
}

/// **反発項だけ**の重み付き値（B-1 用）。`r` は原子間距離。
///
/// 1-4 ペアに「立体反発だけ」を入れるための関数。Vina の分子内項をそのまま 1-4 に
/// 入れると引力（ガウス 2 項・疎水・水素結合）も一緒に入り、**折りたたみを促す**
/// （段 90・92.1）。反発だけなら折りたたみ圧力を増やさずに立体障害を表現できる。
/// 追加パラメータは無い（Vina の重み `W_REPULSION` と XS 半径をそのまま使う）。
#[inline]
pub fn pair_repulsion(a: XsType, b: XsType, r: f64) -> f64 {
    let d = r - pair_sum_radii(a, b);
    W_REPULSION * repulsion(d)
}

/// `pair_repulsion` の値と `d/dr`。
#[inline]
pub fn pair_repulsion_grad(a: XsType, b: XsType, r: f64) -> (f64, f64) {
    let d = r - pair_sum_radii(a, b);
    let (v, dv) = repulsion_value_grad(d); // dd/dr = 1
    (W_REPULSION * v, W_REPULSION * dv)
}

/// 分子間（受容体–リガンド）の素値合計。`r < CUTOFF` のペアのみ。
pub fn intermolecular_terms(ligand: &[ScoredAtom], receptor: &[ScoredAtom]) -> Terms {
    let mut total = Terms::default();
    for l in ligand {
        for rec in receptor {
            let r = distance(&l.pos, &rec.pos);
            if r < CUTOFF {
                let t = eval_pair(l.xs, rec.xs, r);
                total.add(&t);
            }
        }
    }
    total
}

/// 構造補正（torsion entropy）を適用した最終スコア（kcal/mol, Vina 互換 affinity）。
/// `num_tors` は Vina 1.2 の実効回転結合数（末端基は 0.5 カウント）＝ `Molecule::num_tors_vina()`。
/// 名目 TORSDOF ではないことに注意（末端結合を全数で数えると flexible 配位子を過剰にペナルティする）。
#[inline]
pub fn apply_torsion_correction(e_inter: f64, num_tors: f64) -> f64 {
    e_inter / (1.0 + W_ROT * num_tors)
}

/// Vina の `curl`（hunt_cap）: 探索中に**正のエネルギー（激突）を上限 `v` で滑らかに抑える**。
/// `e>0` のとき `e·v/(v+e)`（e→∞ で v に漸近）を返し、勾配は `(v/(v+e))²` 倍にスケールする。
/// 激突ポーズの巨大エネルギー・勾配が探索を支配・トラップするのを防ぎ、深い極小へ抜けやすくする。
/// 返り値 `(上限化エネルギー, 勾配スケール)`。`e<=0` や `v` 非有限なら素通り（スケール 1）。
#[inline]
pub fn curl(e: f64, v: f64) -> (f64, f64) {
    if e > 0.0 && v.is_finite() {
        let tmp = if v < 1e-12 { 0.0 } else { v / (v + e) };
        (e * tmp, tmp * tmp)
    } else {
        (e, 1.0)
    }
}

// --- 解析的勾配（M4 高速化） ---
//
// 各項の値とその d/dd（d=表面間距離、dd/dr=1 なので d/dr と等しい）を返す。`eval_pair` と
// 完全に同一の関数形を用いる（値が一致することを単体テストで担保）。

#[inline]
fn gauss_value_grad(d: f64, offset: f64, width: f64) -> (f64, f64) {
    let z = (d - offset) / width;
    let g = (-(z * z)).exp();
    // dg/dd = g * (-2 z / width)
    (g, g * (-2.0 * z / width))
}

#[inline]
fn repulsion_value_grad(d: f64) -> (f64, f64) {
    if d < 0.0 { (d * d, 2.0 * d) } else { (0.0, 0.0) }
}

/// 疎水項（slope_step(1.5, 0.5, d)）の値と d/dd。
#[inline]
fn hydrophobic_value_grad(d: f64) -> (f64, f64) {
    if d <= 0.5 {
        (1.0, 0.0)
    } else if d >= 1.5 {
        (0.0, 0.0)
    } else {
        (1.5 - d, -1.0)
    }
}

/// 水素結合項（slope_step(0.0, -0.7, d)）の値と d/dd。
#[inline]
fn hbond_value_grad(d: f64) -> (f64, f64) {
    if d <= -0.7 {
        (1.0, 0.0)
    } else if d >= 0.0 {
        (0.0, 0.0)
    } else {
        (-d / 0.7, -1.0 / 0.7)
    }
}

/// 1 ペアの重み付きエネルギーと d(energy)/dr を返す（カットオフ判定は呼び出し側）。
pub fn pair_energy_grad(a: XsType, b: XsType, r: f64) -> (f64, f64) {
    pair_energy_grad_with_sum_radii(a, b, r, pair_sum_radii(a, b))
}

/// `pair_energy_grad` の、半径の和を明示する版。
/// 金属の配位半径を変えた場合のスカラ基準を組むのに使う（SIMD 版との一致試験）。
///
/// **半径の大域は読まないが、水素結合の 2 つの旗は読む。**
/// 旗まで明示したい試験には [`pair_energy_grad_with`] を使う。
pub fn pair_energy_grad_with_sum_radii(
    a: XsType,
    b: XsType,
    r: f64,
    sum_radii: f64,
) -> (f64, f64) {
    pair_energy_grad_with(a, b, r, sum_radii, metal_hbond(), metal_hbond_all())
}

/// `pair_energy_grad_with_sum_radii` の**完全な純関数版**。
/// 半径の和と水素結合の 2 つの旗をすべて引数で受け取り、大域状態を一切読まない。
/// SIMD 経路との一致を、大域を書き換えずに試験するために使う。
pub fn pair_energy_grad_with(
    a: XsType,
    b: XsType,
    r: f64,
    sum_radii: f64,
    metal_hbond: bool,
    metal_hbond_all: bool,
) -> (f64, f64) {
    let d = r - sum_radii;
    let (g1, dg1) = gauss_value_grad(d, 0.0, 0.5);
    let (g2, dg2) = gauss_value_grad(d, 3.0, 2.0);
    let (rep, drep) = repulsion_value_grad(d);
    let (hyd, dhyd) = if a.is_hydrophobic() && b.is_hydrophobic() {
        hydrophobic_value_grad(d)
    } else {
        (0.0, 0.0)
    };
    let (hb, dhb) = if h_bond_possible_with(a, b, metal_hbond, metal_hbond_all) {
        hbond_value_grad(d)
    } else {
        (0.0, 0.0)
    };
    let value =
        W_GAUSS1 * g1 + W_GAUSS2 * g2 + W_REPULSION * rep + W_HYDROPHOBIC * hyd + W_HBOND * hb;
    let dvalue =
        W_GAUSS1 * dg1 + W_GAUSS2 * dg2 + W_REPULSION * drep + W_HYDROPHOBIC * dhyd + W_HBOND * dhb;
    (value, dvalue)
}

/// 分子間エネルギー（重み付き）と、各リガンド原子に働く力 `force[i] = -dE/dpos_i` を計算する。
/// `force` は長さ `ligand.len()`、呼び出し側で 0 初期化しておく（加算する）。
pub fn intermolecular_value_grad(
    ligand: &[ScoredAtom],
    receptor: &[ScoredAtom],
    force: &mut [Vec3],
) -> f64 {
    let mut e = 0.0;
    for (i, l) in ligand.iter().enumerate() {
        for rec in receptor {
            let diff = l.pos - rec.pos;
            let r = diff.norm();
            if r < CUTOFF && r > 1e-8 {
                let (v, dvdr) = pair_energy_grad(l.xs, rec.xs, r);
                e += v;
                // dE/dpos_i = dvdr * diff/r → force_i = -dvdr/r * diff
                force[i] -= diff * (dvdr / r);
            }
        }
    }
    e
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 金属の配位半径を触らない限り、半径の和は全型対で **`radius()+radius()` とビット一致**。
    /// これが AutoDock Vina との同一性の根拠（既定 1.2 では挙動が一切変わらない）。
    #[test]
    fn metal_coord_radius_default_is_bit_identical_to_vina() {
        assert!(metal_radii().is_vina_default());
        assert_eq!(DEFAULT_METAL_COORD_RADIUS, XsType::MetalD.radius());
        for &a in &XsType::ALL {
            for &b in &XsType::ALL {
                let vina = a.radius() + b.radius();
                assert_eq!(sum_radii_with(a, b, MetalRadii::default()).to_bits(), vina.to_bits());
                assert_eq!(pair_sum_radii(a, b).to_bits(), vina.to_bits());
            }
        }
    }

    /// 対照の切り替えは既定で入っておらず、入れると金属の水素結合だけが消える。
    /// 大域は触らない（`h_bond_possible` は非公開なので `pair_factors` 経由で確かめる）。
    #[test]
    fn metal_hbond_default_is_vina() {
        assert!(metal_hbond(), "既定は有効（＝Vina）でなければならない");
        for &a in &XsType::ALL {
            for &b in &XsType::ALL {
                let want_hb = ((a.is_donor() && b.is_acceptor())
                    || (a.is_acceptor() && b.is_donor())) as u32 as f64;
                let want_hyd = (a.is_hydrophobic() && b.is_hydrophobic()) as u32 as f64;
                assert_eq!(pair_factors(a, b), (want_hyd, want_hb), "{a:?} {b:?}");
                // `eval_pair` の水素結合項が 0 かどうかも係数と一致すること。
                let t = eval_pair(a, b, 1.5);
                assert_eq!((t.hbond != 0.0) as u32 as f64 * want_hb, (t.hbond != 0.0) as u32 as f64);
            }
        }
        // 金属が供与体で、受容原子と水素結合できる（Vina の挙動）。
        assert_eq!(pair_factors(XsType::MetalD, XsType::NitrogenA).1, 1.0);
        assert_eq!(pair_factors(XsType::MetalD, XsType::CarbonH).1, 0.0);
    }

    /// 半径を下げるのは**金属と受容原子の対だけ**。配位しない相手では 1.2 のまま
    /// （立体の床を保つ）。金属同士も対象外（`MetalD` は受容原子ではない）。
    #[test]
    fn metal_coord_radius_applies_only_to_acceptor_partners() {
        use XsType::*;
        let x = 0.9;
        // 受容原子（N/O の A・DA）が相手なら 0.3 縮む。順序は不問。
        for &acc in &[NitrogenA, NitrogenDA, OxygenA, OxygenDA] {
            assert!(acc.is_acceptor());
            let want = MetalD.radius() + acc.radius() - 0.3;
            assert!((sum_radii_with(MetalD, acc, MetalRadii::acceptors_only(x)) - want).abs() < 1e-12);
            assert!((sum_radii_with(acc, MetalD, MetalRadii::acceptors_only(x)) - want).abs() < 1e-12);
        }
        // 受容原子でない相手（炭素・非極性窒素・ドナー専用）では変わらない。
        for &other in &[CarbonH, CarbonP, NitrogenP, NitrogenD, OxygenD, Sulfur, MetalD] {
            assert!(!other.is_acceptor(), "{other:?} は受容原子ではない前提");
            let want = MetalD.radius() + other.radius();
            assert_eq!(sum_radii_with(MetalD, other, MetalRadii::acceptors_only(x)).to_bits(), want.to_bits());
        }
        // 金属が関わらない対は常に不変。
        assert_eq!(
            sum_radii_with(NitrogenA, OxygenDA, MetalRadii::acceptors_only(x)).to_bits(),
            (NitrogenA.radius() + OxygenDA.radius()).to_bits()
        );
    }

    /// 段 124 で広げた欄が、**指定した種別の相手にだけ**効くこと。
    /// 欄を埋めていない種別（ハロゲン・ケイ素・金属同士・酸素の供与体）は 1.2 のまま。
    #[test]
    fn stage124_radii_apply_to_the_named_partner_class_only() {
        use XsType::*;
        let mut t = MetalRadii::all(DEFAULT_METAL_COORD_RADIUS);
        t.zn_ndon = 0.22;
        t.zn_s = 0.32;
        t.other_f = 0.47;
        // 亜鉛 × 供与体のみの窒素: 1.2 − 0.22 = 0.98 縮む。順序は不問。
        for &n in &[NitrogenD, NitrogenP] {
            assert!(!n.is_acceptor());
            let want = MetalDZn.radius() + n.radius() - 0.98;
            assert!((sum_radii_with(MetalDZn, n, t) - want).abs() < 1e-12, "{n:?}");
            assert!((sum_radii_with(n, MetalDZn, t) - want).abs() < 1e-12, "{n:?}");
        }
        // 亜鉛 × 硫黄: 1.2 − 0.32 = 0.88 縮む。
        let want = MetalDZn.radius() + Sulfur.radius() - 0.88;
        assert!((sum_radii_with(MetalDZn, Sulfur, t) - want).abs() < 1e-12);
        // 亜鉛以外 × フッ素: 1.2 − 0.47 = 0.73 縮む。亜鉛では変わらない（欄が既定）。
        let want = MetalD.radius() + Fluorine.radius() - 0.73;
        assert!((sum_radii_with(MetalD, Fluorine, t) - want).abs() < 1e-12);
        assert_eq!(
            sum_radii_with(MetalDZn, Fluorine, t).to_bits(),
            (MetalDZn.radius() + Fluorine.radius()).to_bits()
        );
        // 欄を埋めていない相手は 1.2 のまま（立体の床を保つ）。
        for &o in &[CarbonH, CarbonP, OxygenD, Chlorine, Bromine, Iodine, Silicon, MetalD, MetalDZn,
                    Phosphorus]
        {
            let want = MetalDZn.radius() + o.radius();
            assert_eq!(
                sum_radii_with(MetalDZn, o, t).to_bits(), want.to_bits(),
                "{o:?} は欄が既定なので変わってはいけない"
            );
        }
        // 金属が関わらない対は常に不変。
        assert_eq!(
            sum_radii_with(NitrogenD, Sulfur, t).to_bits(),
            (NitrogenD.radius() + Sulfur.radius()).to_bits()
        );
    }

    /// 段 124 の表（`STAGE124_RADII`）の値が、測定した配位距離で**反発を消す**こと。
    /// 打ち消し（水素結合項）が無い組では正味の最小点が `d = 0` なので、
    /// 「配位距離 ≈ 半径の和」になっていれば良い。
    #[test]
    fn stage124_radii_zero_the_repulsion_at_the_measured_distance() {
        use XsType::*;
        // (相手, 金属が亜鉛か, 段 124.6 の配位距離中央)
        for &(partner, zn, r_coord) in &[
            (NitrogenD, true, 2.024),
            (Sulfur, true, 2.315),
            (Sulfur, false, 2.388),
            (Fluorine, false, 1.974),
        ] {
            let metal = if zn { MetalDZn } else { MetalD };
            let d_new = r_coord - sum_radii_with(metal, partner, STAGE124_RADII);
            let d_vina = r_coord - sum_radii_with(metal, partner, MetalRadii::default());
            // 既定では反発が立っている。
            assert!(
                W_REPULSION * repulsion(d_vina) > 0.4,
                "{partner:?} zn={zn}: 既定の反発が小さすぎる"
            );
            // 段 124 の表では反発がほぼ消える（表面距離が 0 近傍）。
            assert!(
                W_REPULSION * repulsion(d_new) < 0.05,
                "{partner:?} zn={zn}: 反発が残っている d={d_new}"
            );
            // 水素結合項が入らない組なので、報酬は増えていない（確認）。
            if partner != NitrogenA {
                assert_eq!(pair_factors(metal, partner).1, 0.0);
            }
        }
    }

    /// 段 128 の門: 金属 × 配位しうる元素だけを対象にする。
    ///
    /// 炭素を含めてしまうと金属が疎水性炭素を引き寄せるので、
    /// **含めてはならない**ことをここで固定する。
    #[test]
    fn stage128_gate_covers_only_coordinating_elements() {
        use XsType::*;
        for m in [MetalD, MetalDZn] {
            for t in [NitrogenP, NitrogenD, NitrogenA, NitrogenDA, OxygenP, OxygenD,
                      OxygenA, OxygenDA, Sulfur, Fluorine] {
                assert!(metal_coordinating_pair(m, t), "{m:?}×{t:?} は配位しうる");
                assert!(metal_coordinating_pair(t, m), "向きは問わない");
            }
            for t in [CarbonH, CarbonP, Phosphorus, Chlorine, Bromine, Iodine,
                      Silicon, MetalD, MetalDZn] {
                assert!(!metal_coordinating_pair(m, t), "{m:?}×{t:?} は対象外");
            }
        }
        // 金属が片方に居ないと成立しない。
        assert!(!metal_coordinating_pair(NitrogenA, OxygenA));
    }

    /// 段 128 の旗の効き方と優先順位。
    #[test]
    fn stage128_flag_opens_the_gate_without_touching_other_pairs() {
        use XsType::*;
        // 既定（旗を立てない）＝ Vina のまま。供与体 × 受容体だけ。
        assert!(h_bond_possible_with(MetalDZn, NitrogenA, true, false));
        assert!(!h_bond_possible_with(MetalDZn, NitrogenD, true, false));
        assert!(!h_bond_possible_with(MetalDZn, Sulfur, true, false));
        assert!(!h_bond_possible_with(MetalDZn, Fluorine, true, false));
        // 旗を立てると配位しうる元素は型に依らず入る。
        assert!(h_bond_possible_with(MetalDZn, NitrogenD, true, true));
        assert!(h_bond_possible_with(MetalDZn, Sulfur, true, true));
        assert!(h_bond_possible_with(MetalDZn, Fluorine, true, true));
        assert!(h_bond_possible_with(MetalD, Sulfur, true, true));
        // 炭素は旗を立てても入らない。
        assert!(!h_bond_possible_with(MetalDZn, CarbonH, true, true));
        assert!(!h_bond_possible_with(MetalDZn, CarbonP, true, true));
        // 金属が関わらないペアは旗に影響されない。
        assert!(!h_bond_possible_with(NitrogenD, Sulfur, true, true));
        assert!(h_bond_possible_with(NitrogenD, OxygenA, true, true));
        assert!(!h_bond_possible_with(NitrogenD, NitrogenD, true, true));
        // 優先順位: 金属の水素結合を全部外す指定が最も強い。
        assert!(!h_bond_possible_with(MetalDZn, NitrogenA, false, true));
        assert!(!h_bond_possible_with(MetalDZn, Sulfur, false, true));
    }

    /// 段 128 の主張の数値を固定する。配位距離での正味がどう動くか。
    ///
    /// 「門を開けるだけで硫黄・フッ素・供与体のみ窒素の罰則が消える」
    /// という主張が崩れたら落ちる。
    #[test]
    fn stage128_gate_removes_the_penalty_at_coordination_distance() {
        use XsType::*;
        /// 配位距離 `r` での 5 項の正味（疎水は金属に効かないので省く）。
        fn net(a: XsType, b: XsType, r: f64, radii: MetalRadii, all: bool) -> f64 {
            let d = r - sum_radii_with(a, b, radii);
            let hb = if h_bond_possible_with(a, b, true, all) {
                W_HBOND * hbond_term(d)
            } else {
                0.0
            };
            W_GAUSS1 * gauss(d, 0.0, 0.5)
                + W_GAUSS2 * gauss(d, 3.0, 2.0)
                + W_REPULSION * repulsion(d)
                + hb
        }
        let def = MetalRadii::default();
        let s128 = STAGE128_RADII;
        // (相手, 配位距離, 既定の正味, 門を開いた正味, 門＋表の正味)
        // 配位距離は段 128.1 の第 1 配位圏の中央値（元素で切ったもの）。
        let cases = [
            (MetalDZn, NitrogenA, 1.992, 0.266, 0.266, -0.222),
            (MetalDZn, NitrogenD, 1.992, 0.853, 0.266, -0.222),
            (MetalDZn, Sulfur, 2.312, 0.661, 0.073, 0.073),
            (MetalDZn, OxygenA, 2.210, -0.184, -0.184, -0.184),
            (MetalD, NitrogenA, 2.268, -0.142, -0.142, -0.142),
            (MetalD, OxygenA, 2.192, -0.171, -0.171, -0.171),
            (MetalD, Sulfur, 2.389, 0.550, -0.037, -0.037),
            (MetalD, Fluorine, 1.996, 0.411, -0.176, -0.176),
        ];
        for (m, t, r, e_def, e_gate, e_both) in cases {
            for (got, want, what) in [
                (net(m, t, r, def, false), e_def, "既定"),
                (net(m, t, r, def, true), e_gate, "門を開く"),
                (net(m, t, r, s128, true), e_both, "門＋段 128 の表"),
            ] {
                assert!((got - want).abs() < 2e-3,
                        "{m:?}×{t:?} {r} Å の{what}: {got:.3}（想定 {want:.3}）");
            }
        }
        // 主張の要点: 門を開けるだけで硫黄・フッ素・供与体のみ窒素は負になるか
        // ほぼ 0 になり、残る罰則は亜鉛–窒素だけである。
        assert!(net(MetalDZn, NitrogenA, 1.992, def, true) > 0.2,
                "亜鉛–窒素は門だけでは直らない（半径が要る）");
        assert!(net(MetalDZn, NitrogenA, 1.992, s128, true) < -0.2,
                "門＋表で亜鉛–窒素は有利になる");
    }

    /// 段 128 の表は亜鉛 × 窒素の 2 欄だけを動かす（同じ元素なので同じ値）。
    #[test]
    fn stage128_radii_touch_only_zinc_nitrogen() {
        let t = STAGE128_RADII;
        assert_eq!(t.zn_n.to_bits(), 0.70f64.to_bits());
        assert_eq!(t.zn_ndon.to_bits(), 0.70f64.to_bits(),
                   "受容窒素と供与体のみ窒素は同じ元素なので同じ値");
        let d = MetalRadii::default();
        for (name, got, want) in [
            ("zn_o", t.zn_o, d.zn_o), ("zn_s", t.zn_s, d.zn_s),
            ("zn_f", t.zn_f, d.zn_f), ("zn_c", t.zn_c, d.zn_c),
            ("zn_p", t.zn_p, d.zn_p), ("other_n", t.other_n, d.other_n),
            ("other_ndon", t.other_ndon, d.other_ndon), ("other_o", t.other_o, d.other_o),
            ("other_s", t.other_s, d.other_s), ("other_f", t.other_f, d.other_f),
            ("other_c", t.other_c, d.other_c), ("other_p", t.other_p, d.other_p),
        ] {
            assert_eq!(got.to_bits(), want.to_bits(), "{name} は既定 1.2 のまま");
        }
        // 段 124 の表とは別物（あちらは供与体のみ窒素を 0.22 にしていた）。
        assert_ne!(t, STAGE124_RADII);
        assert!(!t.is_vina_default());
    }

    /// 既定の旗は Vina と同一でなければならない（黙って変わると全結果が動く）。
    #[test]
    fn stage128_flag_defaults_to_vina_behaviour() {
        assert!(!metal_hbond_all(), "段 128 の旗の既定は false ＝ Vina と同じ");
        assert!(metal_hbond(), "金属の水素結合は既定で有効 ＝ Vina と同じ");
    }

    /// 推奨表が**亜鉛–受容窒素 0.70 の 1 欄だけ**であることを固定する（段 125.10 の決定）。
    ///
    /// 段 124.16 で 5 欄の `STAGE124_RADII` を推奨にしたが、段 125.10 の標本外確認で
    /// 利得が再現せず（選定に使っていない 6 標的で通算 1 勝 6 敗）、差し戻した。
    /// 推奨表は黙って戻りやすいので、ここで固定して取り違えを検出する。
    #[test]
    fn recommended_radii_table_is_zinc_acceptor_nitrogen_only() {
        let rec = RECOMMENDED_METAL_RADII;
        assert_eq!(rec, STAGE122_RADII, "推奨表は段 122 の表（亜鉛–受容窒素だけ）");
        assert_ne!(rec, STAGE124_RADII, "段 124 の 5 欄表は推奨ではない");
        // 亜鉛–受容窒素だけが既定と違う。
        assert_eq!(rec.zn_n.to_bits(), 0.70f64.to_bits());
        let d = MetalRadii::default();
        for (name, got, want) in [
            ("zn_o", rec.zn_o, d.zn_o), ("other_n", rec.other_n, d.other_n),
            ("other_o", rec.other_o, d.other_o), ("zn_ndon", rec.zn_ndon, d.zn_ndon),
            ("zn_s", rec.zn_s, d.zn_s), ("zn_f", rec.zn_f, d.zn_f),
            ("zn_c", rec.zn_c, d.zn_c), ("zn_p", rec.zn_p, d.zn_p),
            ("other_ndon", rec.other_ndon, d.other_ndon), ("other_s", rec.other_s, d.other_s),
            ("other_f", rec.other_f, d.other_f), ("other_c", rec.other_c, d.other_c),
            ("other_p", rec.other_p, d.other_p),
        ] {
            assert_eq!(got.to_bits(), want.to_bits(), "{name} は既定 1.2 のままでなければならない");
        }
        // 推奨表は既定ではない（既定は Vina そのまま）。
        assert!(!rec.is_vina_default());
        assert!(metal_radii().is_vina_default(), "実行時の既定は Vina のまま");
    }

    /// 推奨値で、**実測の配位距離における反発がほぼ消えている**ことを固定する。
    /// 段 112 で支配している量は反発だと分かったので、これが推奨値の本質的な条件である。
    /// 報酬の深さは成績に効かないので固定しない（段 112.4 で予測を外した箇所）。
    #[test]
    fn recommended_metal_radius_kills_the_repulsion() {
        use XsType::*;
        // 段 86.3 の実測配位距離。窒素 n=390、酸素 n=28。
        for &(partner, r_coord) in &[(NitrogenA, 1.93), (OxygenA, 2.07)] {
            let rep = |x: f64| {
                let d = r_coord - sum_radii_with(MetalD, partner, MetalRadii::acceptors_only(x));
                W_REPULSION * repulsion(d)
            };
            let vina = rep(DEFAULT_METAL_COORD_RADIUS);
            let rec = rep(RECOMMENDED_METAL_COORD_RADIUS);
            // Vina の既定では反発が立っている。
            assert!(vina > 0.5, "{partner:?}: Vina の反発が小さすぎる {vina}");
            // 推奨値では台地の条件（およそ 0.2 kcal/mol 未満）を満たす。
            assert!(rec < 0.2, "{partner:?}: 推奨値の反発が残っている {rec}");
            // 台地の全域（0.0〜0.6）で条件を満たす。
            for &x in &[0.0, 0.2, 0.4, 0.6] {
                assert!(rep(x) < 0.2, "{partner:?} x={x}: 反発 {} が残っている", rep(x));
            }
            // 台地の外（0.9 以上）では満たさない ＝ 条件が自明に真ではない。
            assert!(rep(0.9) > 0.2, "{partner:?}: x=0.9 で条件が緩すぎる");
        }
        // 推奨値は既定ではない（既定は Vina のまま）。
        assert_ne!(RECOMMENDED_METAL_COORD_RADIUS, DEFAULT_METAL_COORD_RADIUS);
        assert!(metal_radii().is_vina_default());
    }

    /// 段 110 の数値を固定する。亜鉛–窒素の実測配位距離 1.93 Å（段 86.3、n=390）で、
    /// 既定 1.2 では**正味 +0.375 kcal/mol の罰則**、0.9 に下げると報酬側（負）に入る。
    #[test]
    fn metal_coord_radius_removes_the_zinc_penalty() {
        use XsType::*;
        // 罰則の分岐点: 0.840245·d² = 0.587439 → d = -0.836。
        let d_breakeven = -(W_HBOND.abs() / W_REPULSION).sqrt();
        assert!((d_breakeven - (-0.836_0)).abs() < 1e-3, "d_breakeven = {d_breakeven}");

        // 5 項の重み付き和を、半径の和を指定して直接組む。
        let net = |a: XsType, b: XsType, r: f64, metal_r: f64| {
            let d = r - sum_radii_with(a, b, MetalRadii::acceptors_only(metal_r));
            W_GAUSS1 * gauss(d, 0.0, 0.5)
                + W_GAUSS2 * gauss(d, 3.0, 2.0)
                + W_REPULSION * repulsion(d)
                + if a.is_hydrophobic() && b.is_hydrophobic() { W_HYDROPHOBIC * hydrophobic_term(d) } else { 0.0 }
                + if h_bond_possible(a, b) { W_HBOND * hbond_term(d) } else { 0.0 }
        };

        // Zn–N(受容) 1.93 Å: 既定は罰則、0.9 では報酬。
        let n_default = net(MetalD, NitrogenA, 1.93, DEFAULT_METAL_COORD_RADIUS);
        let n_fixed = net(MetalD, NitrogenA, 1.93, 0.9);
        assert!(n_default > 0.3, "既定の Zn-N は罰則のはず: {n_default}");
        assert!(n_fixed < 0.0, "0.9 では報酬のはず: {n_fixed}");

        // Zn–O(受容) 2.07 Å: 既定でもわずかに報酬、0.9 でより深くなる。
        let o_default = net(MetalD, OxygenA, 2.07, DEFAULT_METAL_COORD_RADIUS);
        let o_fixed = net(MetalD, OxygenA, 2.07, 0.9);
        assert!(o_default < 0.0 && o_fixed < o_default, "{o_default} -> {o_fixed}");

        // 既定では `eval_pair` 経由の値と完全に一致する（大域が既定のままであること）。
        assert_eq!(
            eval_pair(MetalD, NitrogenA, 1.93).weighted_sum().to_bits(),
            n_default.to_bits()
        );
    }

    #[test]
    fn gauss_peaks() {
        // gauss1 は d=0 で 1、gauss2 は d=3 で 1。
        assert!((gauss(0.0, 0.0, 0.5) - 1.0).abs() < 1e-12);
        assert!((gauss(3.0, 3.0, 2.0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn repulsion_only_on_clash() {
        assert_eq!(repulsion(0.5), 0.0);
        assert!((repulsion(-0.5) - 0.25).abs() < 1e-12);
    }

    #[test]
    fn hydrophobic_ramp() {
        assert_eq!(hydrophobic_term(0.4), 1.0);
        assert_eq!(hydrophobic_term(1.6), 0.0);
        assert!((hydrophobic_term(1.0) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn hbond_ramp() {
        assert_eq!(hbond_term(-0.8), 1.0);
        assert_eq!(hbond_term(0.1), 0.0);
        assert!((hbond_term(-0.35) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn hydrophobic_only_between_hydrophobic_atoms() {
        // C_H–C_H は疎水項あり、C_H–O_A は無し。
        let t = eval_pair(XsType::CarbonH, XsType::CarbonH, 3.9);
        assert!(t.hydrophobic > 0.0);
        let t2 = eval_pair(XsType::CarbonH, XsType::OxygenA, 3.6);
        assert_eq!(t2.hydrophobic, 0.0);
    }

    #[test]
    fn hbond_only_donor_acceptor() {
        // ドナー(O_D)–アクセプタ(O_A) は hbond あり。
        let t = eval_pair(XsType::OxygenD, XsType::OxygenA, 3.0);
        assert!(t.hbond > 0.0);
        // アクセプタ同士は無し。
        let t2 = eval_pair(XsType::OxygenA, XsType::OxygenA, 3.0);
        assert_eq!(t2.hbond, 0.0);
    }

    #[test]
    fn pair_grad_value_matches_eval_pair() {
        // pair_energy_grad の値が eval_pair の重み付き和と一致する（複数の型・距離で）。
        let types = [
            XsType::CarbonH,
            XsType::OxygenA,
            XsType::OxygenD,
            XsType::NitrogenD,
        ];
        for &a in &types {
            for &b in &types {
                for &r in &[2.0, 3.5, 4.0, 5.5, 7.9] {
                    let (v, _) = pair_energy_grad(a, b, r);
                    let want = eval_pair(a, b, r).weighted_sum();
                    assert!((v - want).abs() < 1e-12, "{a:?},{b:?},{r}: {v} vs {want}");
                }
            }
        }
    }

    #[test]
    fn pair_grad_matches_numerical() {
        // d(energy)/dr が中心差分と一致する（区分線形の折れ点を避けた点で）。
        let types = [XsType::CarbonH, XsType::OxygenA, XsType::OxygenD];
        let eps = 1e-6;
        for &a in &types {
            for &b in &types {
                for &r in &[2.3, 3.1, 4.4, 5.2, 6.7] {
                    let (_, dv) = pair_energy_grad(a, b, r);
                    let (vp, _) = pair_energy_grad(a, b, r + eps);
                    let (vm, _) = pair_energy_grad(a, b, r - eps);
                    let num = (vp - vm) / (2.0 * eps);
                    assert!((dv - num).abs() < 1e-4, "{a:?},{b:?},{r}: {dv} vs {num}");
                }
            }
        }
    }

    #[test]
    fn intermolecular_grad_value_and_force() {
        let ligand = vec![
            ScoredAtom { pos: Vec3::new(0.0, 0.0, 0.0), xs: XsType::CarbonH },
            ScoredAtom { pos: Vec3::new(1.4, 0.3, 0.0), xs: XsType::OxygenD },
        ];
        let receptor = vec![
            ScoredAtom { pos: Vec3::new(3.0, 0.5, 0.2), xs: XsType::OxygenA },
            ScoredAtom { pos: Vec3::new(-2.0, 1.0, 0.0), xs: XsType::CarbonH },
            ScoredAtom { pos: Vec3::new(2.5, -1.5, 1.0), xs: XsType::NitrogenA },
        ];
        let mut force = vec![Vec3::zeros(); ligand.len()];
        let e = intermolecular_value_grad(&ligand, &receptor, &mut force);
        // 値はフル再計算と一致。
        let want = intermolecular_terms(&ligand, &receptor).weighted_sum();
        assert!((e - want).abs() < 1e-12, "{e} vs {want}");

        // 力 = -dE/dpos を、各リガンド原子で数値微分と比較。
        let eps = 1e-6;
        for (i, f) in force.iter().enumerate() {
            let mut num = Vec3::zeros();
            for axis in 0..3 {
                let mut lp = ligand.clone();
                lp[i].pos[axis] += eps;
                let ep = intermolecular_terms(&lp, &receptor).weighted_sum();
                let mut lm = ligand.clone();
                lm[i].pos[axis] -= eps;
                let em = intermolecular_terms(&lm, &receptor).weighted_sum();
                num[axis] = -(ep - em) / (2.0 * eps); // force = -grad
            }
            assert!((f - num).norm() < 1e-4, "atom {i}: force {f:?} vs {num:?}");
        }
    }
}
