use serde::{Deserialize, Serialize};
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ScfResiduals {
    pub geometry_residual: f64,
    pub mass_residual: f64,
    pub slope_residual: f64,
    pub transport_residual: f64,
    pub kkt_residual: f64,
}
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ScfTolerances {
    pub geometry_tol: f64,
    pub mass_tol: f64,
    pub slope_tol: f64,
    pub transport_tol: f64,
    pub kkt_tol: f64,
}
impl Default for ScfTolerances {
    fn default() -> Self {
        Self {
            geometry_tol: 1e-4,
            mass_tol: 1e-3,
            slope_tol: 1e-4,
            transport_tol: 1e-3,
            kkt_tol: 1e-3,
        }
    }
}
pub fn compute_geometry_residual(r_next: &[f64], r_curr: &[f64]) -> f64 {
    assert_eq!(r_next.len(), r_curr.len());
    let mut diff_sq = 0.0;
    let mut curr_sq = 0.0;
    for (&rn, &rc) in r_next.iter().zip(r_curr.iter()) {
        let d = rn - rc;
        diff_sq += d * d;
        curr_sq += rc * rc;
    }
    (diff_sq.sqrt() / curr_sq.sqrt().max(1e-12)).max(0.0)
}
pub fn compute_mass_residual(actual_mass: f64, target_mass: f64) -> f64 {
    (actual_mass - target_mass).abs() / target_mass.max(1e-12)
}
pub fn compute_slope_residual(sin2_slope_list: &[f64], sin2_phi_list: &[f64]) -> f64 {
    let mut max_viol = 0.0;
    for (&s2, &p2) in sin2_slope_list.iter().zip(sin2_phi_list.iter()) {
        let diff = s2 - p2;
        if diff > max_viol {
            max_viol = diff;
        }
    }
    max_viol
}
pub fn compute_transport_residual(m_next: &[[f64; 4]], m_curr: &[[f64; 4]]) -> f64 {
    assert_eq!(m_next.len(), m_curr.len());
    let mut diff_sq = 0.0;
    let mut curr_sq = 0.0;
    for (mn, mc) in m_next.iter().zip(m_curr.iter()) {
        for k in 0..4 {
            let d = mn[k] - mc[k];
            diff_sq += d * d;
            curr_sq += mc[k] * mc[k];
        }
    }
    (diff_sq.sqrt() / curr_sq.sqrt().max(1e-12)).max(0.0)
}
pub fn compute_kkt_residual(
    r_sol: &[f64],
    r_eq: &[f64],
    weights: &[f64],
    mass_coeffs: &[f64],
    lambda: f64,
) -> f64 {
    let mut max_grad = 0.0;
    for i in 0..r_sol.len() {
        let kkt_stationarity = weights[i] * (r_sol[i] - r_eq[i]) + lambda * mass_coeffs[i];
        let abs_val = kkt_stationarity.abs();
        if abs_val > max_grad {
            max_grad = abs_val;
        }
    }
    max_grad
}
pub fn check_scf_convergence(
    residuals: &ScfResiduals,
    tols: &ScfTolerances,
) -> (bool, Vec<String>) {
    let mut failed_reasons = Vec::new();
    if residuals.geometry_residual > tols.geometry_tol {
        failed_reasons.push(format!(
            "几何残差未收敛: {:.2e} > 容差 {:.2e}",
            residuals.geometry_residual, tols.geometry_tol
        ));
    }
    if residuals.mass_residual > tols.mass_tol {
        failed_reasons.push(format!(
            "质量残差未收敛: {:.2e} > 容差 {:.2e}",
            residuals.mass_residual, tols.mass_tol
        ));
    }
    if residuals.slope_residual > tols.slope_tol {
        failed_reasons.push(format!(
            "坡度超限未收敛: {:.2e} > 容差 {:.2e}",
            residuals.slope_residual, tols.slope_tol
        ));
    }
    if residuals.transport_residual > tols.transport_tol {
        failed_reasons.push(format!(
            "输运残差未收敛: {:.2e} > 容差 {:.2e}",
            residuals.transport_residual, tols.transport_tol
        ));
    }
    if residuals.kkt_residual > tols.kkt_tol {
        failed_reasons.push(format!(
            "KKT 残差未收敛: {:.2e} > 容差 {:.2e}",
            residuals.kkt_residual, tols.kkt_tol
        ));
    }
    let is_converged = failed_reasons.is_empty();
    (is_converged, failed_reasons)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_residuals_and_convergence() {
        let r_curr = vec![100.0, 100.0, 100.0];
        let r_next = vec![100.005, 99.995, 100.002];
        let geom_res = compute_geometry_residual(&r_next, &r_curr);
        assert!(geom_res < 1e-4);
        let mass_res = compute_mass_residual(1.0002e9, 1.0e9);
        assert!((mass_res - 0.0002).abs() < 1e-8);
        let slope_res = compute_slope_residual(&[0.3, 0.4], &[0.35, 0.45]);
        assert_eq!(slope_res, 0.0);
        let residuals = ScfResiduals {
            geometry_residual: geom_res,
            mass_residual: mass_res,
            slope_residual: slope_res,
            transport_residual: 1e-5,
            kkt_residual: 1e-4,
        };
        let tols = ScfTolerances::default();
        let (converged, reasons) = check_scf_convergence(&residuals, &tols);
        assert!(converged, "残差应判定为收敛, 失败原因: {:?}", reasons);
    }
}
