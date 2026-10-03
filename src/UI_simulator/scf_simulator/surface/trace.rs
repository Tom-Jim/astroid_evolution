use glam::DVec2;
#[inline]
pub fn compute_b_k(
    grad_s_q_k: [f64; 2],
    q_k: f64,
    grad_s_r: [f64; 2],
    r: f64,
    k_order: usize,
) -> [f64; 2] {
    let k = k_order as f64;
    let inv_r = if r > 1e-12 { 1.0 / r } else { 0.0 };
    let coeff = (1.0 - k) * q_k * inv_r;
    let g_q = DVec2::from_array(grad_s_q_k);
    let g_r = DVec2::from_array(grad_s_r);
    (g_q + coeff * g_r).to_array()
}
pub fn compute_surface_tangential_velocity_2d(
    r: f64,
    grad_s_r: [f64; 2],
    q: &[f64; 4],
    grad_s_q: &[[f64; 2]; 4],
) -> [f64; 2] {
    let mut sum_b = DVec2::ZERO;
    for k in 1..=4 {
        let b_k = DVec2::from_array(compute_b_k(grad_s_q[k - 1], q[k - 1], grad_s_r, r, k));
        let weight = (k + 2) as f64;
        sum_b += weight * b_k;
    }
    (r * sum_b).to_array()
}
pub fn compute_radial_shear_rate_f_rad(
    r: f64,
    grad_s_r: [f64; 2],
    q: &[f64; 4],
    grad_s_q: &[[f64; 2]; 4],
) -> f64 {
    let mut sum_vec = DVec2::ZERO;
    for k in 1..=4 {
        let b_k = DVec2::from_array(compute_b_k(grad_s_q[k - 1], q[k - 1], grad_s_r, r, k));
        let weight = ((k + 2) * (k - 1)) as f64;
        sum_vec += weight * b_k;
    }
    sum_vec.length()
}
#[inline]
pub fn compute_f_rad_simplified(
    grad_s_q2: [f64; 2],
    grad_s_q3: [f64; 2],
    grad_s_q4: [f64; 2],
) -> f64 {
    let vx = 4.0 * grad_s_q2[0] + 10.0 * grad_s_q3[0] + 18.0 * grad_s_q4[0];
    let vy = 4.0 * grad_s_q2[1] + 10.0 * grad_s_q3[1] + 18.0 * grad_s_q4[1];
    (vx * vx + vy * vy).sqrt()
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_f_rad_degeneracy() {
        let r = 100.0;
        let zero_grad_r = [0.0, 0.0];
        let q = [1.2, 0.5, -0.3, 0.1];
        let grad_s_q = [[1.0, 2.0], [0.1, 0.2], [-0.05, 0.1], [0.02, -0.01]];
        let f_rad_full = compute_radial_shear_rate_f_rad(r, zero_grad_r, &q, &grad_s_q);
        let f_rad_simp = compute_f_rad_simplified(grad_s_q[1], grad_s_q[2], grad_s_q[3]);
        assert!((f_rad_full - f_rad_simp).abs() < 1e-12);
        let mut grad_s_q_alt = grad_s_q;
        grad_s_q_alt[0] = [100.0, -50.0];
        let f_rad_alt = compute_radial_shear_rate_f_rad(r, zero_grad_r, &q, &grad_s_q_alt);
        assert!((f_rad_full - f_rad_alt).abs() < 1e-12);
    }
}
