//! 有效势能 W(x) = U(x) + U_c(x)
//! 严格依照 docs/software.md 第 8.1 节 与 docs/mathstrict.md 第 8 节

use crate::model_calculation::centrifugal::centrifugal_potential;

/// 合成总有效势能 W = U + U_c
#[inline]
pub fn effective_potential(gravitational_potential: f64, centrifugal_potential: f64) -> f64 {
    gravitational_potential + centrifugal_potential
}

/// 计算指定空间场点处包含离心势的有效势能 W(x)
#[inline]
pub fn compute_effective_potential_at_point(
    gravitational_potential: f64,
    pos: [f64; 3],
    omega: [f64; 3],
) -> f64 {
    let uc = centrifugal_potential(omega, pos);
    effective_potential(gravitational_potential, uc)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_effective_potential() {
        let u = -150.0;
        let omega = [0.0, 0.0, 1.0];
        let pos = [10.0, 0.0, 0.0];

        // U_c = 0.5 * 1^2 * 100 = 50.0
        // W = -150.0 + 50.0 = -100.0
        let w = compute_effective_potential_at_point(u, pos, omega);
        assert!((w - (-100.0)).abs() < 1e-12);
    }
}
