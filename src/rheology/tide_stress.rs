//! 潮汐背应力张量与 Frobenius 范数计算
//! 严格依照 docs/software.md 第 9.4 节 与 docs/mathstrict.md 第 4.6 节

/// 计算潮汐/有效引力位 Hessian 偏量诱导的微观潮汐背应力张量 T_tide
///
/// 公式依据 docs/software.md 第 9.4 节 与 docs/mathstrict.md 式 (3858):
/// T_tide = C_T * rho_p * nu * d^2 * dev(H_W)
///
/// 其中 dev(H_W) = H_W - 1/3 Tr(H_W) I
#[inline]
pub fn tidal_backstress(
    c_t: f64,
    rho_p: f64,
    nu: f64,
    grain_diameter: f64,
    hessian_w: &[[f64; 3]; 3],
) -> [[f64; 3]; 3] {
    let tr_h = hessian_w[0][0] + hessian_w[1][1] + hessian_w[2][2];
    let mean_h = tr_h / 3.0;

    let coeff = c_t * rho_p * nu * (grain_diameter * grain_diameter);

    let mut t_tide = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let mut dev_h = hessian_w[i][j];
            if i == j {
                dev_h -= mean_h;
            }
            t_tide[i][j] = coeff * dev_h;
        }
    }
    t_tide
}

/// 计算 3x3 矩阵/张量的 Frobenius 范数 ||A||_F = sqrt(sum A_ij^2)
#[inline]
pub fn frobenius_norm(tensor: &[[f64; 3]; 3]) -> f64 {
    let mut sum_sq = 0.0;
    for i in 0..3 {
        for j in 0..3 {
            sum_sq += tensor[i][j] * tensor[i][j];
        }
    }
    sum_sq.sqrt()
}

/// 计算偏应力等效应力范数 ||A||_eq = ||A||_F / sqrt(2) = sqrt(1/2 A : A)
///
/// 与屈服函数定义式 sqrt(1/2 (s - T) : (s - T)) 保持严格一致
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
        let d = 0.1; // 10 cm, d^2 = 0.01

        // coeff = 1.0 * 2000.0 * 0.5 * 0.01 = 10.0
        let h_w = [
            [2.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
        ]; // Tr(H) = 0, dev(H) = H

        let t_tide = tidal_backstress(c_t, rho_p, nu, d, &h_w);
        assert!((t_tide[0][0] - 20.0).abs() < 1e-12);
        assert!((t_tide[1][1] - (-10.0)).abs() < 1e-12);
        assert!((t_tide[2][2] - (-10.0)).abs() < 1e-12);

        // Frobenius 范数: sqrt(400 + 100 + 100) = sqrt(600) = 10 * sqrt(6)
        let f_norm = frobenius_norm(&t_tide);
        assert!((f_norm - 600.0_f64.sqrt()).abs() < 1e-12);

        // 等效应力: sqrt(600) / sqrt(2) = sqrt(300) = 10 * sqrt(3)
        let eq_stress = equivalent_stress(&t_tide);
        assert!((eq_stress - 300.0_f64.sqrt()).abs() < 1e-12);
    }
}
