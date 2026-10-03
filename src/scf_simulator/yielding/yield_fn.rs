#[inline]
pub fn combined_yield_function(
    stress_dev: &[[f64; 3]; 3],
    t_tide: &[[f64; 3]; 3],
    cohesion: f64,
    mu_y: f64,
    pressure: f64,
) -> f64 {
    let mut contraction = 0.0;
    for i in 0..3 {
        for j in 0..3 {
            let diff = stress_dev[i][j] - t_tide[i][j];
            contraction += diff * diff;
        }
    }
    let effective_shear_stress = (0.5 * contraction).max(0.0).sqrt();
    effective_shear_stress - cohesion - mu_y * pressure.max(0.0)
}
#[inline]
pub fn is_yielded(f_comb: f64) -> bool {
    f_comb > 0.0
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TensorYieldCertificate {
    pub residual: f64,
    pub effective_shear: f64,
    pub threshold: f64,
    pub satisfied: bool,
}
#[inline]
pub fn tensor_yield_certificate(
    stress_dev: &[[f64; 3]; 3],
    t_tide: &[[f64; 3]; 3],
    cohesion: f64,
    mu_y: f64,
    pressure: f64,
) -> Result<TensorYieldCertificate, String> {
    if !cohesion.is_finite()
        || !mu_y.is_finite()
        || !pressure.is_finite()
        || cohesion < 0.0
        || mu_y < 0.0
        || pressure < 0.0
        || stress_dev.iter().flatten().any(|x| !x.is_finite())
        || t_tide.iter().flatten().any(|x| !x.is_finite())
    {
        return Err("张量屈服证书输入必须为有限的非负材料参数和有限张量".into());
    }
    let mut contraction = 0.0;
    for i in 0..3 {
        for j in 0..3 {
            let d = stress_dev[i][j] - t_tide[i][j];
            contraction += d * d;
        }
    }
    let effective_shear = (0.5 * contraction).sqrt();
    let threshold = cohesion + mu_y * pressure;
    let residual = effective_shear - threshold;
    Ok(TensorYieldCertificate {
        residual,
        effective_shear,
        threshold,
        satisfied: residual <= 0.0,
    })
}
#[inline]
pub fn classic_yield_function(
    stress_dev: &[[f64; 3]; 3],
    cohesion: f64,
    mu_y: f64,
    pressure: f64,
) -> f64 {
    let zero_tide = [[0.0; 3]; 3];
    combined_yield_function(stress_dev, &zero_tide, cohesion, mu_y, pressure)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_yield_condition() {
        let p = 1000.0;
        let c = 50.0;
        let mu_y = 0.4;
        let yield_threshold = c + mu_y * p;
        let s_below = [
            [400.0 * 2.0_f64.sqrt(), 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ];
        let f_below = classic_yield_function(&s_below, c, mu_y, p);
        assert!((f_below - (400.0 - yield_threshold)).abs() < 1e-10);
        assert!(!is_yielded(f_below));
        let s_above = [
            [500.0 * 2.0_f64.sqrt(), 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ];
        let f_above = classic_yield_function(&s_above, c, mu_y, p);
        assert!((f_above - (500.0 - yield_threshold)).abs() < 1e-10);
        assert!(is_yielded(f_above));
    }
    #[test]
    fn test_tide_counteraction() {
        let p = 1000.0;
        let c = 50.0;
        let mu_y = 0.4;
        let s = [
            [500.0 * 2.0_f64.sqrt(), 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ];
        let t_tide = [
            [200.0 * 2.0_f64.sqrt(), 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ];
        let f_comb = combined_yield_function(&s, &t_tide, c, mu_y, p);
        assert!((f_comb - (300.0 - 450.0)).abs() < 1e-10);
        assert!(!is_yielded(f_comb));
    }
    #[test]
    fn tensor_certificate_is_exact_and_fail_closed() {
        let s = [
            [300.0 * 2.0_f64.sqrt(), 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ];
        let t = [[0.0; 3]; 3];
        let cert = tensor_yield_certificate(&s, &t, 50.0, 0.4, 1000.0).unwrap();
        assert!((cert.effective_shear - 300.0).abs() < 1e-10);
        assert!(cert.satisfied);
        assert!(tensor_yield_certificate(&s, &t, -1.0, 0.4, 1000.0).is_err());
    }
}
