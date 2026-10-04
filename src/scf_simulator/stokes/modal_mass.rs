#[inline]
pub fn compute_conservative_modal_state(m_mat: &[[f64; 4]; 4], q: &[f64; 4]) -> [f64; 4] {
    let mut m = [0.0; 4];
    for i in 0..4 {
        m[i] = m_mat[i][0] * q[0] + m_mat[i][1] * q[1] + m_mat[i][2] * q[2] + m_mat[i][3] * q[3];
    }
    m
}
#[inline]
pub fn recover_physical_modal_state(inv_m: &[[f64; 4]; 4], m: &[f64; 4]) -> [f64; 4] {
    let mut q = [0.0; 4];
    for i in 0..4 {
        q[i] = inv_m[i][0] * m[0] + inv_m[i][1] * m[1] + inv_m[i][2] * m[2] + inv_m[i][3] * m[3];
    }
    q
}
pub fn compute_conservative_states_batch(
    m_mat: &[[f64; 4]; 4],
    q_batch: &[[f64; 4]],
) -> Vec<[f64; 4]> {
    q_batch
        .iter()
        .map(|q| compute_conservative_modal_state(m_mat, q))
        .collect()
}
pub fn recover_physical_states_batch(inv_m: &[[f64; 4]; 4], m_batch: &[[f64; 4]]) -> Vec<[f64; 4]> {
    m_batch
        .iter()
        .map(|m| recover_physical_modal_state(inv_m, m))
        .collect()
}
#[cfg(test)]
mod tests {
    use super::super::assemble::{assemble_mass_matrix, invert_mass_matrix};
    use super::*;
    #[test]
    fn test_conservative_roundtrip() {
        let m_mat = assemble_mass_matrix();
        let inv_m = invert_mass_matrix(&m_mat).unwrap();
        let q_orig = [1.2, -0.4, 0.85, 0.3];
        let m = compute_conservative_modal_state(&m_mat, &q_orig);
        let q_rec = recover_physical_modal_state(&inv_m, &m);
        for i in 0..4 {
            assert!(
                (q_orig[i] - q_rec[i]).abs() < 1e-10,
                "第 {i} 分量还原误差过大: orig {}, rec {}",
                q_orig[i],
                q_rec[i]
            );
        }
    }
}
