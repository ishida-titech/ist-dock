//! 分子内ペアポテンシャルの事前計算テーブル（AutoDock Vina `precalculate` 相当）。
//!
//! `scoring::pair_energy_grad` が返す「重み付きペアエネルギーとその d/dr」を、XS 型対 × r²（距離²）で
//! 事前に **(値, d値/d(r²))** の線形補間テーブルに焼き込む。探索中の分子内評価はこのテーブルを引くだけで、
//! `exp`・`sqrt`・`/r` を一切呼ばない（r² 添字なので sqrt 不要、力は `2·dvalue_dr2·diff` で `/r` も不要）。
//!
//! テーブルは受容体・リガンド非依存（スコア関数定数のみに依存）なので `OnceLock` で 1 度だけ構築し、
//! 全リガンド・全評価・全スレッドで共有する。**探索（近似許容）専用**。polish/最終スコアは呼び出し側で
//! 従来の厳密 exp 経路（`scoring::pair_energy_grad` / `eval_pair`）を使い、出力の厳密性を保つ。

use crate::atom_types::XsType;
use crate::scoring::{pair_energy_grad, CUTOFF};
use std::sync::OnceLock;

/// r² ビンの解像度（1 Å² あたりのビン数）。`bin = (r2 * FACTOR) as usize`。
/// r∈[0,8]→r²∈[0,64] を `FACTOR` 刻みで格子化。32 で厳密比 max <1.5e-3 kcal/pair（区分線形 kink 由来）、
/// テーブル ~9.5MB＝キャッシュに優しく最速。探索専用（polish は厳密）なのでこの誤差は許容。
const FACTOR: f64 = 32.0;
/// r² 上限（`CUTOFF²`）。
const R2_MAX: f64 = CUTOFF * CUTOFF;
/// ビン数（`R2_MAX * FACTOR` ＋ 補間で i+1 を引くための番兵 2）。
const N_BINS: usize = (R2_MAX * FACTOR) as usize + 2;
/// 型対数（対称だが添字簡単化のため full COUNT×COUNT）。
const N_PAIRS: usize = XsType::COUNT * XsType::COUNT;

/// 分子内ペアポテンシャルの補間テーブル。各エントリ = `(value, dvalue_dr2)`。
pub struct IntraTable {
    /// `[type_pair * N_BINS + bin]` で `(value, dvalue_dr2)`。
    data: Vec<(f64, f64)>,
}

/// 型対 `(a,b)` の連続インデックス（full 格納）。
#[inline]
fn pair_index(a: XsType, b: XsType) -> usize {
    a.index() * XsType::COUNT + b.index()
}

impl IntraTable {
    /// スコア関数からテーブルを構築する（純関数・決定論）。
    pub fn build() -> IntraTable {
        let mut data = vec![(0.0f64, 0.0f64); N_PAIRS * N_BINS];
        for &a in &XsType::ALL {
            for &b in &XsType::ALL {
                let base = pair_index(a, b) * N_BINS;
                for bin in 0..N_BINS {
                    // 全ビンに厳密値を格納（カットオフは `value()` の `r2 >= R2_MAX` ガードで課す）。
                    // こうすると cutoff 直下ビンの補間が厳密値のまま保たれる。r=0 は特異なので下限クランプ
                    // （実 intra ペアは近接しないので未使用域）。
                    let r2 = bin as f64 / FACTOR;
                    let r = r2.max(1e-12).sqrt();
                    let (v, dvdr) = pair_energy_grad(a, b, r);
                    // d(value)/d(r²) = d(value)/dr · dr/d(r²) = dvdr / (2r)。
                    data[base + bin] = (v, dvdr / (2.0 * r));
                }
            }
        }
        IntraTable { data }
    }

    /// 値のみ（Metropolis 判定・値評価用）。`r2` は距離²。カットオフ外は 0。
    #[inline]
    pub fn value(&self, a: XsType, b: XsType, r2: f64) -> f64 {
        if r2 >= R2_MAX {
            return 0.0;
        }
        let base = pair_index(a, b) * N_BINS;
        let x = r2 * FACTOR;
        let i = x as usize; // 0..N_BINS-2（r2<R2_MAX なので安全）
        let frac = x - i as f64;
        let (v0, _) = self.data[base + i];
        let (v1, _) = self.data[base + i + 1];
        v0 + (v1 - v0) * frac
    }

    /// 値と d(value)/d(r²)（力の計算用）。`force -= 2·dvalue_dr2·diff` で使う。カットオフ外は (0,0)。
    #[inline]
    pub fn value_grad_r2(&self, a: XsType, b: XsType, r2: f64) -> (f64, f64) {
        if r2 >= R2_MAX {
            return (0.0, 0.0);
        }
        let base = pair_index(a, b) * N_BINS;
        let x = r2 * FACTOR;
        let i = x as usize;
        let frac = x - i as f64;
        let (v0, d0) = self.data[base + i];
        let (v1, d1) = self.data[base + i + 1];
        (v0 + (v1 - v0) * frac, d0 + (d1 - d0) * frac)
    }
}

/// プロセス共有の分子内テーブル（初回アクセスで 1 度だけ構築）。
pub fn table() -> &'static IntraTable {
    static TABLE: OnceLock<IntraTable> = OnceLock::new();
    TABLE.get_or_init(IntraTable::build)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn value_matches_exact() {
        let t = table();
        // 全型対 × r グリッドで LUT 値が厳密値と一致（探索用補間誤差 < 1.5e-3 kcal/pair）。
        let mut maxerr = 0.0f64;
        for &a in &XsType::ALL {
            for &b in &XsType::ALL {
                let mut r = 0.5;
                while r < CUTOFF {
                    let (want, _) = pair_energy_grad(a, b, r);
                    let got = t.value(a, b, r * r);
                    maxerr = maxerr.max((got - want).abs());
                    r += 0.05;
                }
            }
        }
        assert!(maxerr < 1.5e-3, "max intra LUT value error {maxerr} exceeds 1.5e-3");
    }

    #[test]
    fn grad_r2_matches_numerical() {
        let t = table();
        // d(value)/d(r²) が LUT 値の r² 数値微分と一致（折れ点を避けた点で）。
        let eps = 1e-4;
        for &a in &[XsType::CarbonH, XsType::OxygenA, XsType::OxygenD, XsType::NitrogenD] {
            for &b in &[XsType::CarbonH, XsType::OxygenA, XsType::NitrogenA] {
                for &r in &[2.3, 3.1, 4.4, 5.2, 6.7] {
                    let r2 = r * r;
                    let (_, d) = t.value_grad_r2(a, b, r2);
                    let num = (t.value(a, b, r2 + eps) - t.value(a, b, r2 - eps)) / (2.0 * eps);
                    assert!((d - num).abs() < 1e-3, "{a:?},{b:?},r={r}: {d} vs {num}");
                }
            }
        }
    }

    #[test]
    fn zero_beyond_cutoff() {
        let t = table();
        assert_eq!(t.value(XsType::CarbonH, XsType::CarbonH, 65.0), 0.0);
        assert_eq!(t.value_grad_r2(XsType::CarbonH, XsType::CarbonH, 100.0), (0.0, 0.0));
    }
}
