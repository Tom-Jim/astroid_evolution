pub const K4_ORDER: usize = 4;
#[inline]
pub fn basis_k4(zeta: f64) -> [f64; 4] {
    let z2 = zeta * zeta;
    let z3 = z2 * zeta;
    let z4 = z3 * zeta;
    let z5 = z4 * zeta;
    [z2, z3, z4, z5]
}
#[inline]
pub fn basis_k4_derivative(zeta: f64) -> [f64; 4] {
    let z = zeta;
    let z2 = z * z;
    let z3 = z2 * z;
    let z4 = z3 * z;
    [2.0 * z, 3.0 * z2, 4.0 * z3, 5.0 * z4]
}
#[inline]
pub fn basis_k4_second_derivative(zeta: f64) -> [f64; 4] {
    let z = zeta;
    let z2 = z * z;
    let z3 = z2 * z;
    [2.0, 6.0 * z, 12.0 * z2, 20.0 * z3]
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
        assert_eq!(psi[0], 0.25);
        assert_eq!(psi[1], 0.125);
        assert_eq!(psi[2], 0.0625);
        assert_eq!(psi[3], 0.03125);
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
