//! SCF 几何自洽场收敛性判定与多物理场残差诊断
//! 严格依照 docs/software.md 第 11 节 与 docs/mathstrict.md 第 10.4 节
//!
//! 文档强制判据：只有同时满足几何、质量、输运守恒和投影 KKT 残差才判定为收敛。

use serde::{Deserialize, Serialize};

/// SCF 迭代残差集合
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ScfResiduals {
    /// 几何外形相对相对残差 ||R^(n+1) - R^(n)||_2 / ||R^(n)||_2
    pub geometry_residual: f64,
    /// 真实封闭质量相对相对残差 |M(R^(n+1)) - M_target| / M_target
    pub mass_residual: f64,
    /// 最大坡度违背量 max_i max(0, sin^2(theta_i) - sin^2(phi_i))
    pub slope_residual: f64,
    /// 模态输运守恒相对残差 ||m^(n+1/2) - m^(n)||_2 / ||m^(n)||_2
    pub transport_residual: f64,
    /// 凸优化 KKT 一阶最优性残差
    pub kkt_residual: f64,
}

/// SCF 收敛容差设定
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

/// 计算几何外形相对残差: ||R_next - R_curr||_2 / ||R_curr||_2
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

/// 计算质量守恒相对残差: |M - M_target| / M_target
pub fn compute_mass_residual(actual_mass: f64, target_mass: f64) -> f64 {
    (actual_mass - target_mass).abs() / target_mass.max(1e-12)
}

/// 计算全表面最大坡度超限残差
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

/// 计算模态输运守恒残差
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

/// 计算投影 QP 的 KKT 一阶梯度最优性残差
pub fn compute_kkt_residual(
    r_sol: &[f64],
    r_eq: &[f64],
    weights: &[f64],
    mass_coeffs: &[f64],
    lambda: f64,
) -> f64 {
    let mut max_grad = 0.0;
    for i in 0..r_sol.len() {
        // 目标函数对 R_i 梯度: w_i * (R_i - R_eq,i)
        // 约束项对 R_i 梯度: lambda * a_i
        let kkt_stationarity = weights[i] * (r_sol[i] - r_eq[i]) + lambda * mass_coeffs[i];
        let abs_val = kkt_stationarity.abs();
        if abs_val > max_grad {
            max_grad = abs_val;
        }
    }
    max_grad
}

/// 综合收敛性检验
///
/// 严格要求全部 5 项物理残差均达到阈值
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
