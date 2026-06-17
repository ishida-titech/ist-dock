//! 幾何ユーティリティ（座標・距離）。
//!
//! 座標は `nalgebra` の 3D ベクトルで表す。M2 で剛体変換（並進＋四元数）、
//! M5 で PCA を追加する。

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
}
