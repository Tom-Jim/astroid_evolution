#[derive(Clone, Copy, Debug)]
pub struct SurfaceMetric {
    pub radius: f64,
    pub jacobian_gamma: f64,
    pub normal: [f64; 3],
    pub e_theta: [f64; 3],
    pub e_phi: [f64; 3],
    pub g_metric: [[f64; 2]; 2],
    pub inv_g_metric: [[f64; 2]; 2],
}
use glam::{DMat2, DVec2, DVec3};
pub fn compute_surface_metric(
    theta: f64,
    phi: f64,
    r: f64,
    r_theta: f64,
    r_phi: f64,
) -> SurfaceMetric {
    let sin_t = theta.sin().max(1e-6);
    let cos_t = theta.cos();
    let sin_p = phi.sin();
    let cos_p = phi.cos();
    let u = DVec3::new(sin_t * cos_p, sin_t * sin_p, cos_t);
    let u_theta = DVec3::new(cos_t * cos_p, cos_t * sin_p, -sin_t);
    let u_phi = DVec3::new(-sin_t * sin_p, sin_t * cos_p, 0.0);
    let e_theta_vec = r_theta * u + r * u_theta;
    let e_phi_vec = r_phi * u + r * u_phi;
    let cross_prod = e_theta_vec.cross(e_phi_vec);
    let jacobian_gamma = cross_prod.length().max(1e-30);
    let normal_vec = cross_prod.normalize_or_zero();
    let g_tt = e_theta_vec.dot(e_theta_vec);
    let g_tp = e_theta_vec.dot(e_phi_vec);
    let g_pp = e_phi_vec.dot(e_phi_vec);
    let g_metric = [[g_tt, g_tp], [g_tp, g_pp]];
    let g_mat = DMat2::from_cols(DVec2::new(g_tt, g_tp), DVec2::new(g_tp, g_pp));
    let inv_g_mat = g_mat.inverse();
    let inv_g_metric = [
        [inv_g_mat.x_axis.x, inv_g_mat.y_axis.x],
        [inv_g_mat.x_axis.y, inv_g_mat.y_axis.y],
    ];
    SurfaceMetric {
        radius: r,
        jacobian_gamma,
        normal: normal_vec.to_array(),
        e_theta: e_theta_vec.to_array(),
        e_phi: e_phi_vec.to_array(),
        g_metric,
        inv_g_metric,
    }
}
pub fn compute_surface_metric_checked(
    theta: f64,
    phi: f64,
    r: f64,
    r_theta: f64,
    r_phi: f64,
) -> Result<SurfaceMetric, String> {
    if !theta.is_finite()
        || !phi.is_finite()
        || !r.is_finite()
        || !r_theta.is_finite()
        || !r_phi.is_finite()
        || r <= 0.0
    {
        return Err("表面度量参数必须为有限数且 R>0".into());
    }
    if theta.sin().abs() <= 1e-8 {
        return Err("极点处球极坐标 chart 奇异；请切换到无极点局部 chart".into());
    }
    let metric = compute_surface_metric(theta, phi, r, r_theta, r_phi);
    if !metric.jacobian_gamma.is_finite() || metric.jacobian_gamma <= 0.0 {
        return Err("表面 Jacobian 非正或非有限".into());
    }
    Ok(metric)
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;
    #[test]
    fn test_sphere_metric_degeneracy() {
        let r0 = 450.0;
        let theta = PI / 3.0;
        let phi = PI / 4.0;
        let metric = compute_surface_metric(theta, phi, r0, 0.0, 0.0);
        let expected_j = r0 * r0 * theta.sin();
        assert!(
            (metric.jacobian_gamma - expected_j).abs() < 1e-6,
            "球面度量 Jacobian 检验失败: 得到 {}, 期望 {}",
            metric.jacobian_gamma,
            expected_j
        );
        let u = [
            theta.sin() * phi.cos(),
            theta.sin() * phi.sin(),
            theta.cos(),
        ];
        for i in 0..3 {
            assert!((metric.normal[i] - u[i]).abs() < 1e-10);
        }
    }
}
