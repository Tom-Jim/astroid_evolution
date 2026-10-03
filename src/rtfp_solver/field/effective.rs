//! Effective potential `W = U + ½|ω × x|²`, its gradient, and its Hessian.
//!
//! Spin about `z` reduces the centrifugal term to `½ ω² (x² + y²)`.

use crate::common::math::DVec3;

#[inline]
pub fn centrifugal_potential(omega: DVec3, pos: DVec3) -> f64 {
    0.5 * omega.cross(pos).length_squared()
}

#[inline]
pub fn centrifugal_acceleration(omega: DVec3, pos: DVec3) -> DVec3 {
    -omega.cross(omega.cross(pos))
}

/// `H_c = |ω|² I − ω ωᵀ`.
#[inline]
pub fn centrifugal_hessian(omega: DVec3) -> [[f64; 3]; 3] {
    let w2 = omega.length_squared();
    let w = omega.to_array();
    let mut h = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let diag = if i == j { w2 } else { 0.0 };
            h[i][j] = diag - w[i] * w[j];
        }
    }
    h
}

#[inline]
pub fn effective_potential(gravitational_potential: f64, omega: DVec3, pos: DVec3) -> f64 {
    gravitational_potential + centrifugal_potential(omega, pos)
}

#[inline]
pub fn effective_gravity(gravitational_accel: DVec3, omega: DVec3, pos: DVec3) -> DVec3 {
    gravitational_accel + centrifugal_acceleration(omega, pos)
}

#[inline]
pub fn effective_hessian(gravitational_hessian: &[[f64; 3]; 3], omega: DVec3) -> [[f64; 3]; 3] {
    let h_c = centrifugal_hessian(omega);
    let mut h = *gravitational_hessian;
    for i in 0..3 {
        for j in 0..3 {
            h[i][j] += h_c[i][j];
        }
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn z_spin_matches_half_omega_squared_cylindrical_radius() {
        let omega = DVec3::new(0.0, 0.0, 1.0);
        let pos = DVec3::new(10.0, 0.0, 0.0);
        let w = effective_potential(-150.0, omega, pos);
        assert!((w - (-100.0)).abs() < 1e-12);
    }

    #[test]
    fn centrifugal_acceleration_and_hessian_about_z() {
        let omega = DVec3::new(0.0, 0.0, 2.0);
        let pos = DVec3::new(3.0, 4.0, 5.0);
        assert!((centrifugal_potential(omega, pos) - 50.0).abs() < 1e-12);
        let a = centrifugal_acceleration(omega, pos);
        assert!((a.x - 12.0).abs() < 1e-12 && (a.y - 16.0).abs() < 1e-12 && a.z.abs() < 1e-12);
        let h = centrifugal_hessian(omega);
        assert!((h[0][0] - 4.0).abs() < 1e-12);
        assert!((h[1][1] - 4.0).abs() < 1e-12);
        assert!(h[2][2].abs() < 1e-12);
    }
}
