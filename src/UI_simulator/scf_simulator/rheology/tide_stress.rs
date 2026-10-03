use nalgebra::Matrix3;
#[inline]
pub fn tidal_backstress(
    c_t: f64,
    rho_p: f64,
    nu: f64,
    grain_diameter: f64,
    hessian_w: &[[f64; 3]; 3],
) -> [[f64; 3]; 3] {
    let flat = [
        hessian_w[0][0],
        hessian_w[0][1],
        hessian_w[0][2],
        hessian_w[1][0],
        hessian_w[1][1],
        hessian_w[1][2],
        hessian_w[2][0],
        hessian_w[2][1],
        hessian_w[2][2],
    ];
    let h = Matrix3::from_row_slice(&flat);
    let dev_h = h - (h.trace() / 3.0) * Matrix3::identity();
    let coeff = c_t * rho_p * nu * (grain_diameter * grain_diameter);
    let t_tide = coeff * dev_h;
    let mut out = [[0.0; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            out[r][c] = t_tide[(r, c)];
        }
    }
    out
}
#[inline]
pub fn frobenius_norm(tensor: &[[f64; 3]; 3]) -> f64 {
    let flat = [
        tensor[0][0],
        tensor[0][1],
        tensor[0][2],
        tensor[1][0],
        tensor[1][1],
        tensor[1][2],
        tensor[2][0],
        tensor[2][1],
        tensor[2][2],
    ];
    Matrix3::from_row_slice(&flat).norm()
}
#[inline]
pub fn equivalent_stress(tensor: &[[f64; 3]; 3]) -> f64 {
    frobenius_norm(tensor) / std::f64::consts::SQRT_2
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_tidal_backstress() {
        let c_t = 1.0;
        let rho_p = 2000.0;
        let nu = 0.5;
        let d = 0.1;
        let h_w = [[2.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]];
        let t_tide = tidal_backstress(c_t, rho_p, nu, d, &h_w);
        assert!((t_tide[0][0] - 20.0).abs() < 1e-12);
        assert!((t_tide[1][1] - (-10.0)).abs() < 1e-12);
        assert!((t_tide[2][2] - (-10.0)).abs() < 1e-12);
        let f_norm = frobenius_norm(&t_tide);
        assert!((f_norm - 600.0_f64.sqrt()).abs() < 1e-12);
        let eq_stress = equivalent_stress(&t_tide);
        assert!((eq_stress - 300.0_f64.sqrt()).abs() < 1e-12);
    }
}
