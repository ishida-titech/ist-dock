//! 幾何ユーティリティ（座標・距離・剛体重ね合わせ）。
//!
//! 座標は `nalgebra` の 3D ベクトルで表す。M3 で Kabsch 重ね合わせ（`superpose`）、
//! M5 で PCA を追加する。

use nalgebra::{Matrix3, Rotation3, UnitQuaternion};

/// 3D 座標 / ベクトル（f64）。
pub type Vec3 = nalgebra::Vector3<f64>;

/// 2 点間のユークリッド距離。
#[inline]
pub fn distance(a: &Vec3, b: &Vec3) -> f64 {
    (a - b).norm()
}

/// 2 点間の距離の二乗（平方根を避けたいカットオフ判定用）。
#[inline]
pub fn distance_sq(a: &Vec3, b: &Vec3) -> f64 {
    (a - b).norm_squared()
}

/// 同順の点群間 RMSD（重ね合わせなし。受容体固定の再ドッキング RMSD 用）。
pub fn rmsd(a: &[Vec3], b: &[Vec3]) -> f64 {
    assert_eq!(a.len(), b.len(), "rmsd: length mismatch");
    if a.is_empty() {
        return 0.0;
    }
    let sum: f64 = a.iter().zip(b).map(|(p, q)| (p - q).norm_squared()).sum();
    (sum / a.len() as f64).sqrt()
}

/// 点群の重心。
fn centroid(p: &[Vec3]) -> Vec3 {
    if p.is_empty() {
        return Vec3::zeros();
    }
    p.iter().fold(Vec3::zeros(), |s, v| s + v) / p.len() as f64
}

/// Kabsch 法による剛体重ね合わせ。`R*src_i + t ≈ dst_i` となる回転 `R`（四元数）と並進 `t`
/// を返す（対応は同順、最小二乗）。Phase 2 で子のコアを docked seed コアへ重ねるのに使う。
///
/// `src` と `dst` は同数（≥1）。鏡映（det<0）は回転に補正する。3 点未満や共線配置では
/// 回転が一意でないが、与えた対応に対する最小二乗解を返す。
pub fn superpose(src: &[Vec3], dst: &[Vec3]) -> (UnitQuaternion<f64>, Vec3) {
    assert_eq!(src.len(), dst.len(), "superpose: length mismatch");
    assert!(!src.is_empty(), "superpose: empty point set");

    let cs = centroid(src);
    let cd = centroid(dst);

    // 相互共分散 H = Σ (src_i - cs)(dst_i - cd)^T。
    let mut h = Matrix3::zeros();
    for (p, q) in src.iter().zip(dst) {
        h += (p - cs) * (q - cd).transpose();
    }

    let svd = h.svd(true, true);
    let u = svd.u.expect("superpose: SVD U");
    let v_t = svd.v_t.expect("superpose: SVD V^T");
    let v = v_t.transpose();
    let ut = u.transpose();

    // 鏡映補正: det(V U^T) が負なら V の最終列を反転。
    let d = if (v * ut).determinant() < 0.0 { -1.0 } else { 1.0 };
    let diag = Matrix3::new(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, d);
    let r = v * diag * ut;

    let rot = Rotation3::from_matrix_unchecked(r);
    let q = UnitQuaternion::from_rotation_matrix(&rot);
    let t = cd - r * cs;
    (q, t)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distance_basic() {
        let a = Vec3::new(0.0, 0.0, 0.0);
        let b = Vec3::new(3.0, 4.0, 0.0);
        assert!((distance(&a, &b) - 5.0).abs() < 1e-12);
        assert!((distance_sq(&a, &b) - 25.0).abs() < 1e-12);
    }

    #[test]
    fn superpose_recovers_known_transform() {
        // 既知の回転＋並進で dst を作り、superpose が復元するか。
        let src = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 2.0, 0.0),
            Vec3::new(0.0, 0.0, 3.0),
        ];
        let axis = nalgebra::Unit::new_normalize(Vec3::new(0.3, -0.7, 1.0));
        let true_rot = UnitQuaternion::from_axis_angle(&axis, 0.9);
        let true_t = Vec3::new(5.0, -2.0, 1.5);
        let dst: Vec<Vec3> = src.iter().map(|p| true_rot * p + true_t).collect();

        let (q, t) = superpose(&src, &dst);
        for (p, d) in src.iter().zip(&dst) {
            let mapped = q * p + t;
            assert!((mapped - d).norm() < 1e-9, "got {mapped:?} want {d:?}");
        }
        // 回転・並進そのものも一致。
        assert!((t - true_t).norm() < 1e-9);
        assert!(q.angle_to(&true_rot) < 1e-7);
    }
}
