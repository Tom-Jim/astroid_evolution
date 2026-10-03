#[inline]
pub fn cohesive_pressure(
    c_c: f64,
    z: f64,
    nu: f64,
    cohesion_force: f64,
    grain_diameter: f64,
) -> f64 {
    if grain_diameter <= 0.0 {
        return 0.0;
    }
    let d2 = grain_diameter * grain_diameter;
    c_c * (z * nu * cohesion_force) / d2
}
#[inline]
pub fn effective_pressure(pressure: f64, p_c: f64) -> f64 {
    (pressure + p_c).max(0.0)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_cohesive_pressure() {
        let c_c = 1.0;
        let z = 6.0;
        let nu = 0.6;
        let f_h = 1e-7;
        let d = 1e-3;
        let pc = cohesive_pressure(c_c, z, nu, f_h, d);
        assert!((pc - 0.36).abs() < 1e-12);
        let p_star = effective_pressure(10.0, pc);
        assert!((p_star - 10.36).abs() < 1e-12);
    }
}
