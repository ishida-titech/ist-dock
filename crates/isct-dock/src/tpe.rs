//! TPE（Tree-structured Parzen Estimator）探索の中核（opt-in 探索モード）。
//!
//! TPEvina-2.0（PSOVina 派生, CBBio）の TPE を ISCT-Dock へ移植したもの。探索空間の各次元を
//! **独立な Gaussian Parzen 混合**でモデル化し、良観測（lower）から候補をサンプリングして
//! 期待改善（g(x)/l(x) の比）で 1 点を選ぶ。局所最適化には ISCT の BFGS（`minimize_free`）を再利用する。
//!
//! ISCT 版の意図的な差分（原実装からの逸脱, いずれも移植上必須）:
//! - **決定論**: 原実装の TPE サンプリングは `std::random_device` シード（run 毎に非再現。作者も FIXME 明記）。
//!   本移植は探索全体を seeded `ChaCha8Rng` で駆動し**同一 seed でビット一致**（他モードと同じ ISCT 方針）。
//! - **数値**: boost/std の分布・浮動小数演算差、正規乱数は Box–Muller で自前生成のため、原実装とは
//!   ビット一致しない（=忠実な再実装であり byte clone ではない）。
//! - **greater 重み**: 出荷版 TPEvina 既定 `bfgs.h` に合わせ**順位ベース** `w(gi)=clamp(gi/(N_g−26),0,1)`
//!   （gi=greater 内スコア順位, N_g=greater 観測数）。greater が 27 個以上たまると活性化し、g(x) による
//!   期待改善（g/l 最小化）で `ei_candidates(sample_num)` 個の候補から 1 点を選ぶ。旧移植は `*_7s` 実験変種の
//!   減衰式 `(upper/2−iter)/upper` を採っており greater が常に不活性だった（`docs/findings.202607090942`）。
//!   `num_lower_groups`(lower 上限)=10, `sample_num`(EI 候補)=25, `kernel_density_frequency`(再構築回数)=6,
//!   `start_search_num`(warmup 比率 startup_ratio)=0.05 を既定とする（`docs/results.202607091051`,
//!   `docs/findings.202607091051`）。
//!
//! モデル空間ベクトル（長さ `6 + n_free_torsions`）: `[pos(3, 絶対), rot(3, 参照配座からの scaled-axis 差),
//! free_torsion(n, 参照からの差)]`。位置は探索ボックス角、回転・torsion は [-π, π] を範囲とする。

use rand::Rng;

use isct_core::geometry::Vec3;
use nalgebra::UnitQuaternion;

use crate::ligand::Conformation;

/// TPE ハイパーパラメータ（既定は TPEvina 準拠）。
#[derive(Debug, Clone, Copy)]
pub struct TpeParams {
    /// 1 提案あたり lower 混合から引く候補数（n_ei_candidates, TPEvina `sample_num`）。
    pub ei_candidates: usize,
    /// ウォームアップ（一様ランダム）割合: warmup = n_steps × startup_ratio（0.0–1.0）。
    /// TPEvina `start_search_num`（あちらは除数だが isct では直感的な比率にした。既定 0.05 = 従来の 1/20）。
    pub startup_ratio: f64,
    /// auto n_steps の係数: `auto_n_steps_tpe = auto_n_steps_vina × step_ratio`（**大きいほど深い**）。
    /// 既定 0.1（TPEvina の num_steps/15 相当は 1/15≈0.0667）。n_steps を明示指定した場合は不使用。
    pub step_ratio: f64,
    /// lower（良）集合サイズ上限（TPEvina `num_lower_groups`）。実サイズ = clamp(0.25√N, 2, これ)。
    pub lower_cap: usize,
    /// greater（悪）カーネル数の上限（`lower_cap` の greater 版）。g(x) はこの個数まで切詰めて使う。
    /// TPEvina 変数名は `num_upper_groups`（＝本来「greater」の集合。名称が紛らわしいため isct では greater_cap）。
    pub greater_cap: usize,
    /// TPE フェーズでのモデル再構築回数（TPEvina `kernel_density_frequency`＝1 モデルの使い回し係数）。
    pub density_reuse: usize,
    /// 1 提案あたりの局所最適化（BFGS）反復数。0 なら呼び出し側の既定（`chain_iters`）を使う。
    pub local_iters: usize,
    /// **共有観測モード**（**既定 true**）。`n_runs` 個の独立 run ではなく、総予算
    /// `n_runs × auto_n_steps_tpe` を **1 本の共有観測プール・単一モデル**で逐次探索する（warmup 1 回のみ、
    /// 前回結果を再利用）。初期依存性の小さい TPE で独立多始点の冗長を除き、少予算で n_runs=8 相当へ到達させる。
    /// `false` にすると従来の独立 n_runs（`results.202607100834` とビット一致）へ戻せる。
    pub shared: bool,
    /// 共有モードのリアンカー周期（提案数）。**0=周期リアンカーなし（単一アンカーの共有探索, 推奨・既定)**。
    /// >0 で N 提案ごとに探索的 restart（観測は保持）を注入する。Astex 実測ではリアンカー有無で精度はほぼ
    /// 不変（`docs/results.202607111615`）＝共有観測プール自体が利得源。>0 は探索性を要する用途向けの任意ノブ。
    pub reanchor_period: usize,
    /// 共有モードの未踏域カバレッジ重み（0.0 で無効＝従来のエネルギー基準アンカー）。>0 のとき新アンカー候補を
    /// 既存観測への最小距離（未探索度）で加重選択し、未踏域を重点探索する。
    pub coverage_weight: f64,
    /// **局所チェーン歩数**（K, 既定 0=OFF＝現行と完全一致）。共有モードで各提案の初期 BFGS 後、competitive な
    /// basin（`local_chain_margin` ゲート内）なら Drift 流の短い MC チェーン（mutate+BFGS+Metropolis を K 歩）で
    /// 精密化する＝MC の「持続滞在」を部分移植し、native 基が浅い探索律速標的（1ig3/1lpz 型）を救う。
    /// チェーンの最良ポーズを観測に採用し、各局所最小は pool へ挿入。0 で rng 消費ゼロ＝既定挙動とビット一致。
    pub local_chain: usize,
    /// 局所チェーンのゲート幅（kcal, 既定 2.0）。提案の初期 BFGS energy が `best_so_far + margin` 以下のときのみ
    /// チェーンを起動する。native/decoy 双方の competitive basin を精密化しつつ激突提案には浪費しない＝コスト集中。
    /// **skip が有効なリガンド**（`skip_active`）に適用する margin。skip はモデルを集中させ連鎖が過剰発火するため
    /// 狭める（2.0）。skip 無効リガンドには `nonskip_local_chain_margin` を使い分ける。
    pub local_chain_margin: f64,
    /// **skip が無効なリガンド**（柔軟 T≥`skip_torsdof` 等）に適用する局所チェーンゲート幅（既定 0.0=分離無効＝全
    /// リガンドで `local_chain_margin` を使用＝ビット不変）。skip 無効側はモデルが集中せず過剰発火しないため、広い
    /// margin（例 4.0）で competitive basin を広く精密化して柔軟リガンドの精度を回収できる。`>0.0` で `!skip_active`
    /// のリガンドのみこの値へ切替（skip の margin 再較正が柔軟を巻き添えにする問題への対策, `docs/results.20260714*`）。
    pub nonskip_local_chain_margin: f64,
    /// 局所チェーンの変異幅スケール（既定 1.0＝Drift 幅と完全一致・ビット一致）。<1.0 で並進 ±2·scale Å・回転
    /// ±scale rad・torsion ±π·scale に縮小し、基内 dwelling（refine）寄りにして basin-hop（歩き去り）を抑える。
    pub local_chain_scale: f64,
    /// **局所探索スキップ**（QuickVina2 相当, 既定 false=OFF＝現行とビット一致）。共有モードで各**非 warmup**
    /// 提案の start を安価な勾配1回で判定し、近傍の既知斜面と勾配方向が一致（cos>`skip_cos`）＝同一 basin なら
    /// 高価な初期 BFGS（提案あたり全 eval の ~78%）をスキップする。tpe プリセットの律速（冗長 basin への再最適化）を
    /// 削る。スキップ提案は BFGS/チェーン/pool/obs をすべて省く（既知 basin の極小は既に pool/obs に在る）。
    pub skip_local: bool,
    /// 局所探索スキップの勾配方向一致閾値（cosine, 既定 0.90＝`qvina_cos` と同値）。高いほど保守的（スキップ少）。
    pub skip_cos: f64,
    /// **局所探索スキップの TORSDOF ゲート閾値**（既定 0=ゲート無し＝`skip_local` を全リガンドに適用で現行とビット
    /// 一致）。`skip_torsdof>0` のとき、スキップは `free_torsions.len() < skip_torsdof` の剛直リガンドに限定し、
    /// 柔軟リガンド（T≥閾値）では無効化する。スキップの精度コストは柔軟側に集中する（高DOF は native 盆地が難で
    /// 初期 BFGS 省略が native 消失に直結。剛直は盆地が易・冗長でスキップ無害, `docs/findings.202607141756`）ため、
    /// 剛直だけスキップして速度を保ちつつ柔軟の精度を回収する。0 では `T<0` を判定に使わず全適用＝バイト不変。
    pub skip_torsdof: usize,
    /// **TORSDOF 条件付き局所チェーン抑制の閾値**（既定 0=OFF＝現行とビット一致）。`free_torsions.len() <
    /// rigid_torsdof` のリガンドを「剛直」とみなし、局所チェーン歩数を `rigid_local_chain` に切替える。剛直・
    /// 低自由度デコイで tpe が Vina 関数の深い偽陽性極小を掘り当てて過小スコア化→EF1% を下げる問題
    /// （`HANDOFF.ef1_diagnosis.md`）への対策。0 では実 ligand（n≥0）で `n<0` が常に偽＝抑制なし＝バイト不変。
    pub rigid_torsdof: usize,
    /// 剛直リガンド（`free_torsions.len() < rigid_torsdof`）で用いる局所チェーン歩数 K（既定 0）。通常 `local_chain`
    /// より小さくし、0 なら剛直ではチェーン無効＝基本 BFGS のみで Vina 相当の探索深度に寄せる。`rigid_torsdof=0`
    /// なら不使用。
    pub rigid_local_chain: usize,
    /// **n_steps（提案数）縮小スケールの最小値**（既定 1.0=OFF）。複雑さ連動の段階スケール
    /// `step_scale(T) = min(1.0, rigid_step_scale + rigid_step_per_tors·T)`（T=`free_torsions.len()`）の下限＝
    /// 最剛直（T=0）に適用する係数。低自由度ほど n=1 で収束済みゆえ削れる（`HANDOFF.regularization.md` §5.1）。
    /// 1.0 で全 T が素通り＝現行とビット一致。共有モードでのみ有効（budget=n_runs×n_steps を縮小）。
    pub rigid_step_scale: f64,
    /// n_steps 縮小スケールの**段階ランプ勾配**（1 torsion あたりの増分, 既定 0.0）。`step_scale(T)` を T とともに
    /// 1.0 へ線形復帰させる（`T ≥ (1−rigid_step_scale)/rigid_step_per_tors` で full）。低〜中自由度を緩やかに削り、
    /// 高自由度（最適化不足が大）は full に保つ多段階化。0.0 だと全 T で `rigid_step_scale` 一律（=非推奨。柔軟も削れる）
    /// ため、`rigid_step_scale<1.0` で使うときは正の勾配を設定すること。`rigid_step_scale>=1.0` なら全体 OFF。
    pub rigid_step_per_tors: f64,
    /// n_steps 縮小時の下限（既定 200）。`n_steps × step_scale(T)` がこれを下回らないよう clamp し、warmup と
    /// 最低限の探索を確保する。`step_scale(T)>=1.0` なら不使用。
    pub rigid_step_floor: usize,
}

impl Default for TpeParams {
    fn default() -> Self {
        TpeParams {
            ei_candidates: 25,
            startup_ratio: 0.05,
            step_ratio: 0.1,
            lower_cap: 10,
            greater_cap: 10,
            density_reuse: 6,
            local_iters: 0,
            shared: true,
            reanchor_period: 0,
            coverage_weight: 0.0,
            local_chain: 0,
            local_chain_margin: 2.0,
            nonskip_local_chain_margin: 0.0,
            local_chain_scale: 1.0,
            skip_local: false,
            skip_cos: 0.90,
            skip_torsdof: 0,
            rigid_torsdof: 0,
            rigid_local_chain: 0,
            rigid_step_scale: 1.0,
            rigid_step_per_tors: 0.0,
            rigid_step_floor: 200,
        }
    }
}

/// 観測点（モデル空間ベクトル＋スコア）。スコア昇順が良い。
#[derive(Debug, Clone)]
pub struct Obs {
    pub v: Vec<f64>,
    pub score: f64,
}

/// 角度を [-π, π) へ正規化する（TPEvina `normalized_angle` 相当）。
#[inline]
pub fn normalize_angle(a: f64) -> f64 {
    use std::f64::consts::PI;
    let two_pi = 2.0 * PI;
    let mut x = (a + PI).rem_euclid(two_pi) - PI;
    if x == PI {
        x = -PI;
    }
    x
}

/// 各次元の探索範囲 [l_limit, g_limit]。位置=ボックス角、回転・torsion=[-π, π]。
/// `n_free_torsions` は free torsion 数（モデル次元 = 6 + n_free_torsions）。
pub fn bounds(center: Vec3, size: Vec3, n_free_torsions: usize) -> (Vec<f64>, Vec<f64>) {
    use std::f64::consts::PI;
    let half = size * 0.5;
    let mut l = vec![
        center.x - half.x,
        center.y - half.y,
        center.z - half.z,
    ];
    let mut g = vec![
        center.x + half.x,
        center.y + half.y,
        center.z + half.z,
    ];
    for _ in 0..(3 + n_free_torsions) {
        l.push(-PI);
        g.push(PI);
    }
    (l, g)
}

/// モデル空間ベクトル `v` を `Conformation` へ写す。
/// `v = [pos(3, 絶対ボックス座標), rot(3, 参照 orientation からの scaled-axis 差), free_torsion 差(n)]`。
/// ISCT の `Conformation.position` は `root_pivot` からの**変位**なので、絶対座標から `root_pivot` を引く
/// （`random_in_box` の `target - root_pivot` と同一規約 = リガンド root 重心が絶対座標 `v[0..3]` に載る）。
pub fn conf_from_vec(
    reference: &Conformation,
    free_torsions: &[usize],
    root_pivot: Vec3,
    v: &[f64],
) -> Conformation {
    let mut c = reference.clone();
    c.position = Vec3::new(v[0], v[1], v[2]) - root_pivot;
    let rot = Vec3::new(v[3], v[4], v[5]);
    c.orientation = UnitQuaternion::from_scaled_axis(rot) * reference.orientation;
    for (k, &t) in free_torsions.iter().enumerate() {
        c.torsions[t] = normalize_angle(reference.torsions[t] + v[6 + k]);
    }
    c
}

/// `Conformation`（局所最適化後）をモデル空間ベクトルへ写す（観測の記録用）。位置は絶対座標
/// `root_pivot + position` に戻す。
pub fn vec_from_conf(
    reference: &Conformation,
    free_torsions: &[usize],
    root_pivot: Vec3,
    c: &Conformation,
) -> Vec<f64> {
    let mut v = Vec::with_capacity(6 + free_torsions.len());
    let abs = c.position + root_pivot;
    v.push(abs.x);
    v.push(abs.y);
    v.push(abs.z);
    let diff = (c.orientation * reference.orientation.inverse()).scaled_axis();
    v.push(diff.x);
    v.push(diff.y);
    v.push(diff.z);
    for &t in free_torsions {
        v.push(normalize_angle(c.torsions[t] - reference.torsions[t]));
    }
    v
}

/// モデル空間 2 ベクトル間の距離（位置=ユークリッド Å, 回転・torsion=角度差ラジアン）。共有モードの
/// カバレッジ（未踏域）評価用。角度側は `ang_scale`（Å/rad）で Å 換算し位置と合成する。
pub fn model_dist(a: &[f64], b: &[f64], ang_scale: f64) -> f64 {
    let mut s = 0.0;
    for j in 0..3 {
        let d = a[j] - b[j];
        s += d * d;
    }
    for j in 3..a.len() {
        let d = normalize_angle(a[j] - b[j]) * ang_scale;
        s += d * d;
    }
    s.sqrt()
}

/// 観測集合 `obs` への最小距離（＝未探索度; 大きいほど未踏）。`obs` 空なら `f64::INFINITY`。
pub fn min_dist_to_obs(v: &[f64], obs: &[Obs], ang_scale: f64) -> f64 {
    obs.iter()
        .map(|o| model_dist(v, &o.v, ang_scale))
        .fold(f64::INFINITY, f64::min)
}

/// ウォームアップ提案: 各次元を範囲内で一様サンプル。
pub fn warmup_sample<R: Rng>(rng: &mut R, l_limit: &[f64], g_limit: &[f64]) -> Vec<f64> {
    l_limit
        .iter()
        .zip(g_limit)
        .map(|(&l, &g)| rng.gen_range(l..g))
        .collect()
}

/// 平均 `mean`・標準偏差 `sigma` の正規乱数（Box–Muller, 外部依存を避け決定論）。
fn sample_normal<R: Rng>(rng: &mut R, mean: f64, sigma: f64) -> f64 {
    use std::f64::consts::PI;
    // u1 ∈ (0,1] で ln(0) を回避。
    let u1: f64 = 1.0 - rng.r#gen::<f64>();
    let u2: f64 = rng.r#gen::<f64>();
    let z = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
    mean + sigma * z
}

/// 1 次元分の Parzen カーネル群（値と帯域）。
struct Dim {
    /// lower カーネル（value, sigma）。
    lower: Vec<(f64, f64)>,
    /// greater カーネル（value, sigma, weight）。g(x) 用に上限 `greater_cap` 個へ切詰め済み。
    /// weight は観測のスコア順位由来（順位ベース, 既定 bfgs.h `(N_g−k−1)/(N_g−26)` 相当）。
    greater: Vec<(f64, f64, f64)>,
}

/// 全次元の Parzen モデル。`build` で観測から構築し `sample` で候補を引く。
pub struct TpeModel {
    dims: Vec<Dim>,
    /// greater が活性か（順位ベース重みが正になる = greater 観測が 27 個以上）。`false` のとき g(x)=0 で
    /// 候補選択が退化し、高速経路（各次元 1 点のみサンプル・g/l 計算省略）に切り替える。
    greater_active: bool,
}

/// 近傍ギャップ規則の帯域 sigma（TPEvina）: 端はボックス境界まで、内側は左右ギャップの max。
/// 下限 = range / (1 + n_div)、上限 = range。
#[inline]
fn kernel_sigma(sorted_vals: &[f64], k: usize, lo: f64, hi: f64, n_div: usize) -> f64 {
    let range = hi - lo;
    let n = sorted_vals.len();
    let raw = if k == 0 {
        sorted_vals[k] - lo
    } else if k == n - 1 {
        hi - sorted_vals[k]
    } else {
        (sorted_vals[k + 1] - sorted_vals[k]).max(sorted_vals[k] - sorted_vals[k - 1])
    };
    let min_sigma = range / (1 + n_div) as f64;
    raw.max(min_sigma).min(range)
}

/// lower 混合から 1 点サンプル: カーネルを一様に選び、その周りで正規サンプル（範囲外は再サンプル）。
#[inline]
fn draw_lower<R: Rng>(rng: &mut R, lower: &[(f64, f64)], lo: f64, hi: f64) -> f64 {
    loop {
        let k = rng.gen_range(0..lower.len());
        let (val, sigma) = lower[k];
        let param = sample_normal(rng, val, sigma);
        if param >= lo && param <= hi {
            return param;
        }
    }
}

/// 1 次元の Gaussian 混合密度を `param` で評価（equal-weight, lower 用）。
#[inline]
fn mixture_density(kernels: &[(f64, f64)], param: f64, weight: f64) -> f64 {
    use std::f64::consts::PI;
    let norm = 1.0 / (2.0 * PI).sqrt();
    kernels
        .iter()
        .map(|&(val, sigma)| {
            let z = (param - val) / sigma;
            weight * norm / sigma * (-0.5 * z * z).exp()
        })
        .sum()
}

/// 順位ベースの per-要素重み付き Gaussian 混合密度（greater/g(x) 用）。
#[inline]
fn mixture_density_weighted(kernels: &[(f64, f64, f64)], param: f64) -> f64 {
    use std::f64::consts::PI;
    let norm = 1.0 / (2.0 * PI).sqrt();
    kernels
        .iter()
        .map(|&(val, sigma, w)| {
            let z = (param - val) / sigma;
            w * norm / sigma * (-0.5 * z * z).exp()
        })
        .sum()
}

impl TpeModel {
    /// greater（g(x)）が活性か（観測 27 個以上）。トレース診断用: exploration(g/l 選択) が効いているかを示す。
    pub fn greater_active(&self) -> bool {
        self.greater_active
    }

    /// 観測 `obs` からモデルを構築する。`_iter`（提案カウンタ）は順位ベース greater 重みでは不要だが、
    /// 呼び出し側の互換のため残す。観測が 2 点未満のときは `None`（呼び出し側はウォームアップにフォールバック）。
    pub fn build(
        obs: &[Obs],
        p: &TpeParams,
        _iter: usize,
        l_limit: &[f64],
        g_limit: &[f64],
    ) -> Option<TpeModel> {
        let n = obs.len();
        if n < 2 {
            return None;
        }
        // スコア昇順に並べ、lower / greater へ分割。
        let mut idx: Vec<usize> = (0..n).collect();
        idx.sort_by(|&a, &b| {
            obs[a]
                .score
                .partial_cmp(&obs[b].score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let n_lower = (((n as f64).sqrt() * 0.25) as usize).clamp(2, p.lower_cap).min(n);
        let lower_idx = &idx[..n_lower];
        let greater_idx = &idx[n_lower..];

        // 順位ベース greater 重み（既定 bfgs.h: w(gi)=clamp(gi/(N_g−26),0,1)）。gi=greater 内スコア順位
        // （0=最良greater→重み0, 悪いほど大）。N_g<27 で全 0＝greater 不活性。反復 iter には依存しない。
        let n_g = greater_idx.len();
        let greater_active = n_g >= 27;
        let gw_of = |gi: usize| -> f64 {
            if n_g < 27 {
                0.0
            } else {
                (gi as f64 / (n_g as f64 - 26.0)).clamp(0.0, 1.0)
            }
        };

        let dim = l_limit.len();
        let mut dims = Vec::with_capacity(dim);
        for j in 0..dim {
            let (lo, hi) = (l_limit[j], g_limit[j]);
            // lower カーネル: 値でソート→sigma。
            let mut lv: Vec<f64> = lower_idx.iter().map(|&i| obs[i].v[j]).collect();
            lv.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let lower: Vec<(f64, f64)> = (0..lv.len())
                .map(|k| (lv[k], kernel_sigma(&lv, k, lo, hi, lv.len())))
                .collect();
            // greater カーネル: (値, 順位重み) を値でソート→sigma、g(x) 用に greater_cap 個へ切詰め。
            let mut gvw: Vec<(f64, f64)> = greater_idx
                .iter()
                .enumerate()
                .map(|(gi, &i)| (obs[i].v[j], gw_of(gi)))
                .collect();
            gvw.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            let gv: Vec<f64> = gvw.iter().map(|x| x.0).collect();
            let n_div = (1 + gv.len()).min(100).saturating_sub(1).max(1);
            let greater: Vec<(f64, f64, f64)> = (0..gvw.len())
                .map(|k| (gvw[k].0, kernel_sigma(&gv, k, lo, hi, n_div), gvw[k].1))
                .take(p.greater_cap)
                .collect();
            dims.push(Dim { lower, greater });
        }
        Some(TpeModel {
            dims,
            greater_active,
        })
    }

    /// 1 候補ベクトルをサンプルする。
    ///
    /// **高速経路（`greater_active==false`）**: greater 重みが全 0（greater 観測が 27 個未満）で g(x)=0 の
    /// とき全候補が同点（Σlog(g/l)=-∞）となり `ei_candidates` 本引いても先頭が選ばれるだけ＝候補選択も
    /// g/l 計算も無意味。よって各次元 **1 点だけ**引いて返す（exp 計算と未使用候補のサンプリングを省く）。
    ///
    /// **完全経路（`greater_active==true`）**: greater が活性（観測 27 個以上）のときの本来の TPE。各次元で
    /// lower 混合から `ei_candidates` 個引き、全次元にわたる Σ log(g/l) が最小の候補スロットを採用（tie は先頭）。
    pub fn sample<R: Rng>(
        &self,
        rng: &mut R,
        p: &TpeParams,
        l_limit: &[f64],
        g_limit: &[f64],
    ) -> Vec<f64> {
        // 高速経路: g(x)=0 で候補選択が退化 → 各次元 1 点のみ。
        if !self.greater_active {
            return self
                .dims
                .iter()
                .enumerate()
                .map(|(j, d)| draw_lower(rng, &d.lower, l_limit[j], g_limit[j]))
                .collect();
        }

        let dim = self.dims.len();
        let m = p.ei_candidates.max(1);
        // 各候補スロット s の期待改善プロキシ Σ_dim log(g/l) を累積、per-dim の param を控える。
        let mut ei_per_slot = vec![0.0f64; m];
        let mut params = vec![vec![0.0f64; m]; dim]; // [dim][slot]
        for (j, d) in self.dims.iter().enumerate() {
            let (lo, hi) = (l_limit[j], g_limit[j]);
            for (s, ei) in ei_per_slot.iter_mut().enumerate() {
                let param = draw_lower(rng, &d.lower, lo, hi);
                let lx = mixture_density(&d.lower, param, 1.0);
                let gx = mixture_density_weighted(&d.greater, param);
                // TPEvina は g/l を最小化（比小=良）。log で次元合成。
                let ratio = if lx > 0.0 { gx / lx } else { f64::INFINITY };
                *ei += ratio.ln();
                params[j][s] = param;
            }
        }
        // Σ_dim log(g/l) が最小のスロットを選ぶ（tie は先頭）。
        let best_slot = ei_per_slot
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(s, _)| s)
            .unwrap_or(0);
        params.iter().map(|row| row[best_slot]).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn normalize_angle_wraps() {
        use std::f64::consts::PI;
        assert!((normalize_angle(0.0)).abs() < 1e-12);
        assert!((normalize_angle(2.0 * PI)).abs() < 1e-9);
        assert!((normalize_angle(3.0 * PI) - (-PI)).abs() < 1e-9 || (normalize_angle(3.0 * PI) - PI).abs() < 1e-9);
        assert!(normalize_angle(4.0).abs() <= PI + 1e-12);
    }

    #[test]
    fn conf_vec_roundtrip() {
        let reference = Conformation {
            position: Vec3::new(1.0, 2.0, 3.0),
            orientation: UnitQuaternion::from_scaled_axis(Vec3::new(0.1, -0.2, 0.3)),
            torsions: vec![0.5, -0.3, 0.0],
        };
        let free = vec![0usize, 2];
        let root_pivot = Vec3::new(5.0, -4.0, 1.0);
        // モデルベクトル → conf → モデルベクトル でほぼ復元。
        let v = vec![4.0, -1.0, 2.5, 0.2, 0.1, -0.15, 0.4, -0.6];
        let c = conf_from_vec(&reference, &free, root_pivot, &v);
        let v2 = vec_from_conf(&reference, &free, root_pivot, &c);
        for (a, b) in v.iter().zip(&v2) {
            assert!((a - b).abs() < 1e-9, "{a} vs {b}");
        }
        // 固定 torsion(1) は参照のまま。
        assert!((c.torsions[1] - reference.torsions[1]).abs() < 1e-12);
    }

    #[test]
    fn model_build_and_sample_in_bounds() {
        let (l, g) = bounds(Vec3::zeros(), Vec3::new(10.0, 10.0, 10.0), 1); // dim=7
        let p = TpeParams::default();
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        // ランダム観測。
        let obs: Vec<Obs> = (0..30)
            .map(|i| Obs {
                v: warmup_sample(&mut rng, &l, &g),
                score: (i as f64 * 0.1).sin(),
            })
            .collect();
        let model = TpeModel::build(&obs, &p, 0, &l, &g).unwrap();
        for _ in 0..20 {
            let v = model.sample(&mut rng, &p, &l, &g);
            assert_eq!(v.len(), l.len());
            for j in 0..v.len() {
                assert!(v[j] >= l[j] - 1e-9 && v[j] <= g[j] + 1e-9, "dim {j} out of bounds: {}", v[j]);
            }
        }
    }

    #[test]
    fn fast_path_when_few_greater() {
        // greater 観測が 27 個未満（順位ベース重み全 0）→ 高速経路。各次元 1 点だけ引き範囲内を返す。
        let (l, g) = bounds(Vec3::zeros(), Vec3::new(10.0, 10.0, 10.0), 1); // dim=7
        let p = TpeParams::default();
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        // 20 obs → n_lower=2, n_g=18 (<27) → greater 不活性。
        let obs: Vec<Obs> = (0..20)
            .map(|i| Obs { v: warmup_sample(&mut rng, &l, &g), score: i as f64 })
            .collect();
        let model = TpeModel::build(&obs, &p, 0, &l, &g).unwrap();
        assert!(!model.greater_active, "greater should be inactive with <27 greater obs");
        for _ in 0..20 {
            let v = model.sample(&mut rng, &p, &l, &g);
            assert_eq!(v.len(), l.len());
            for j in 0..v.len() {
                assert!(v[j] >= l[j] - 1e-9 && v[j] <= g[j] + 1e-9);
            }
        }
    }

    #[test]
    fn greater_active_when_enough_obs() {
        // greater 観測が 27 個以上 → 順位ベース重みが正で活性化。完全経路でも範囲内を返す。
        let (l, g) = bounds(Vec3::zeros(), Vec3::new(10.0, 10.0, 10.0), 1); // dim=7
        let p = TpeParams::default();
        let mut rng = ChaCha8Rng::seed_from_u64(5);
        // 40 obs → n_lower=2, n_g=38 (>=27) → greater 活性。
        let obs: Vec<Obs> = (0..40)
            .map(|i| Obs { v: warmup_sample(&mut rng, &l, &g), score: i as f64 })
            .collect();
        let model = TpeModel::build(&obs, &p, 0, &l, &g).unwrap();
        assert!(model.greater_active, "greater should be active with >=27 greater obs");
        for _ in 0..20 {
            let v = model.sample(&mut rng, &p, &l, &g);
            assert_eq!(v.len(), l.len());
            for j in 0..v.len() {
                assert!(v[j] >= l[j] - 1e-9 && v[j] <= g[j] + 1e-9);
            }
        }
    }

    #[test]
    fn determinism_same_seed() {
        let (l, g) = bounds(Vec3::zeros(), Vec3::new(8.0, 8.0, 8.0), 0);
        let p = TpeParams::default();
        let draw = || {
            let mut rng = ChaCha8Rng::seed_from_u64(42);
            let obs: Vec<Obs> = (0..20)
                .map(|i| Obs { v: warmup_sample(&mut rng, &l, &g), score: i as f64 })
                .collect();
            let model = TpeModel::build(&obs, &p, 0, &l, &g).unwrap();
            model.sample(&mut rng, &p, &l, &g)
        };
        assert_eq!(draw(), draw());
    }
}
