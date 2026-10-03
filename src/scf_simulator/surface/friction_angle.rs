#[inline]
pub fn effective_projection_mu(
    cohesion: f64,
    mu_proj: f64,
    pressure: f64,
    t_tide_frobenius: f64,
    p_star: f64,
) -> f64 {
    if p_star <= 1e-14 {
        return 0.0;
    }
    let numerator =
        cohesion + mu_proj * pressure.max(0.0) + t_tide_frobenius / std::f64::consts::SQRT_2;
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
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_friction_angle() {
        let c = 0.0;
        let mu_proj = 0.5;
        let p = 100.0;
        let t_tide_f = 0.0;
        let p_star = p;
        let mu_eff = effective_projection_mu(c, mu_proj, p, t_tide_f, p_star);
        assert!((mu_eff - 0.5).abs() < 1e-14);
        let phi = friction_angle(mu_eff);
        assert!((phi - 0.5_f64.atan()).abs() < 1e-14);
        let sin2 = sin2_friction_angle(mu_eff);
        assert!((sin2 - 0.2).abs() < 1e-14);
    }
}
