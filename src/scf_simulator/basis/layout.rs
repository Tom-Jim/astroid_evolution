pub const K4_ORDER: usize = 4;

/// Spherical-harmonic major layout: eight contiguous (X, k) slots per mode.
/// N_u = 8 * L_max * (L_max + 2).
pub fn velocity_dof_count(l_max: usize) -> usize {
    8 * l_max * (l_max + 2)
}

pub fn dof_index(l: usize, m_index: usize, slot: usize) -> usize {
    let before = if l == 0 { 0 } else { (l - 1) * (l + 1) };
    8 * (before + m_index) + slot
}
/// Four radial modes at `ℓ = 1`: `ψ_1k`, not the monomials `ζ²…ζ⁵`.
#[inline]
pub fn basis_k4(zeta: f64) -> [f64; 4] {
    std::array::from_fn(|k| super::jacobi::psi(1, k, zeta))
}
#[inline]
pub fn basis_k4_derivative(zeta: f64) -> [f64; 4] {
    std::array::from_fn(|k| super::jacobi::psi_derivative(1, k, zeta).1)
}
#[inline]
pub fn basis_k4_second_derivative(zeta: f64) -> [f64; 4] {
    std::array::from_fn(|k| super::jacobi::psi_second(1, k, zeta))
}
#[inline]
pub fn evaluate_modal_field(q: &[f64; 4], zeta: f64) -> f64 {
    let psi = basis_k4(zeta);
    q[0] * psi[0] + q[1] * psi[1] + q[2] * psi[2] + q[3] * psi[3]
}
#[inline]
pub fn evaluate_modal_derivative(q: &[f64; 4], zeta: f64) -> f64 {
    let dpsi = basis_k4_derivative(zeta);
    q[0] * dpsi[0] + q[1] * dpsi[1] + q[2] * dpsi[2] + q[3] * dpsi[3]
}
#[inline]
pub fn evaluate_modal_second_derivative(q: &[f64; 4], zeta: f64) -> f64 {
    let d2psi = basis_k4_second_derivative(zeta);
    q[0] * d2psi[0] + q[1] * d2psi[1] + q[2] * d2psi[2] + q[3] * d2psi[3]
}
pub use basis_k4_derivative as basis_k4_first_derivative;
pub use evaluate_modal_derivative as eval_modal_first_derivative;
pub use evaluate_modal_field as eval_modal_field;
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_basis_k4_values_and_derivatives() {
        let zeta = 0.5;
        let psi = basis_k4(zeta);
        assert!((psi[0] - 0.5).abs() < 1e-12);
        assert!((psi[1] - (6.0 * 0.25 - 2.5)).abs() < 1e-12);
        assert!((basis_k4(1.0)[2] - 1.0).abs() < 1e-12);
        assert!(
            (basis_k4_derivative(1.0)[1] - crate::scf_simulator::basis::jacobi::sigma(1, 1)).abs()
                < 1e-9
        );
        let eps = 1e-6;
        let p_plus = basis_k4(zeta + eps);
        let p_minus = basis_k4(zeta - eps);
        let dpsi_ana = basis_k4_derivative(zeta);
        for k in 0..4 {
            let num = (p_plus[k] - p_minus[k]) / (2.0 * eps);
            assert!((dpsi_ana[k] - num).abs() < 1e-8);
        }
        let dp_plus = basis_k4_derivative(zeta + eps);
        let dp_minus = basis_k4_derivative(zeta - eps);
        let d2psi_ana = basis_k4_second_derivative(zeta);
        for k in 0..4 {
            let num2 = (dp_plus[k] - dp_minus[k]) / (2.0 * eps);
            assert!((d2psi_ana[k] - num2).abs() < 1e-8);
        }
    }
}
