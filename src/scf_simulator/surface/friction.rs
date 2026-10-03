#[inline]
pub fn effective_projection_mu(
    cohesion: f64,
    mu_proj: f64,
    pressure: f64,
    t_tide_frobenius_norm: f64,
    p_star: f64,
) -> f64 {
    if p_star <= 1e-14 {
        return 0.0;
    }
    let numerator =
        cohesion + mu_proj * pressure + t_tide_frobenius_norm / std::f64::consts::SQRT_2;
    (numerator / p_star).max(0.0)
}
#[inline]
pub fn friction_angle(mu_eff: f64) -> f64 {
    mu_eff.max(0.0).atan()
}
#[inline]
pub fn sin2_friction_angle(mu_eff: f64) -> f64 {
    let mu2 = mu_eff * mu_eff;
    mu2 / (1.0 + mu2)
}
use glam::DVec3;
pub fn check_slope_violation(g_eff: [f64; 3], normal: [f64; 3], phi: f64) -> (bool, f64, f64) {
    let g = DVec3::from_array(g_eff);
    let n = DVec3::from_array(normal);
    let cross_norm_sq = g.cross(n).length_squared();
    let g_norm_sq = g.length_squared();
    let n_norm_sq = n.length_squared();
    let denom = g_norm_sq * n_norm_sq;
    let sin2_slope = if denom > 1e-20 {
        (cross_norm_sq / denom).min(1.0)
    } else {
        0.0
    };
    let sin_phi = phi.sin();
    let sin2_phi = sin_phi * sin_phi;
    let is_violated = sin2_slope > sin2_phi + 1e-12;
    (is_violated, sin2_slope, sin2_phi)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_effective_projection_mu_and_angle() {
        let c = 5.0;
        let mu_proj = 0.6;
        let p = 100.0;
        let t_tide_norm = 10.0;
        let p_c = 10.0;
        let p_star = p + p_c;
        let mu_eff = effective_projection_mu(c, mu_proj, p, t_tide_norm, p_star);
        let expected_num = 5.0 + 0.6 * 100.0 + 10.0 / std::f64::consts::SQRT_2;
        let expected_mu = expected_num / 110.0;
        assert!((mu_eff - expected_mu).abs() < 1e-12);
        let phi = friction_angle(mu_eff);
        assert!((phi - expected_mu.atan()).abs() < 1e-12);
        let sin2 = sin2_friction_angle(mu_eff);
        assert!((sin2 - phi.sin().powi(2)).abs() < 1e-12);
    }
    #[test]
    fn test_slope_violation() {
        let g = [0.0, 0.0, -9.8];
        let normal_aligned = [0.0, 0.0, 1.0];
        let phi = 0.5;
        let (violated, sin2_slope, _) = check_slope_violation(g, normal_aligned, phi);
        assert!(!violated);
        assert!(sin2_slope < 1e-12);
        let normal_steep = [1.0, 0.0, 1.0];
        let (violated_steep, sin2_slope_steep, sin2_phi) =
            check_slope_violation(g, normal_steep, phi);
        assert!(violated_steep);
        assert!((sin2_slope_steep - 0.5).abs() < 1e-12);
        assert!(sin2_slope_steep > sin2_phi);
    }
}
