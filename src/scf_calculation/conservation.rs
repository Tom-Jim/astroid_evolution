//! 曲面坐标映射的 Jacobian 与模态守恒修正
//! 严格依照 docs/software.md 第 10.5 节 与 docs/mathstrict.md 第 9.4 节

/// 计算二维曲面坐标映射的反向特征 Jacobian 绝对值
///
/// 对新坐标 a = (theta, phi) 与反向特征坐标 a_d(a)，
/// J_i = | det (∂a_d / ∂a) | = | J_11 * J_22 - J_12 * J_21 |
///
/// 严格遵循文档约束：Jacobian 必须是二维曲面坐标映射的 Jacobian，不能对三维嵌入向量直接取行列式。
#[inline]
pub fn surface_map_jacobian_from_grad(da_d_da: &[[f64; 2]; 2]) -> f64 {
    let det = da_d_da[0][0] * da_d_da[1][1] - da_d_da[0][1] * da_d_da[1][0];
    det.abs()
}

/// 保守模态变量的时间步更新
///
/// 公式依据 docs/software.md 第 10.5 节 与 docs/mathstrict.md 式 (7831):
/// J_{Gamma, i}^{n+1} m_i^{n+1} = J_i J_{Gamma, d} m_i^d + Δt J_{Gamma, i}^{n+1} S_{m, i}
///
/// 解出 m_i^{n+1}:
/// m_i^{n+1} = (J_i J_{Gamma, d} / J_{Gamma, i}^{n+1}) * m_i^d + Δt * S_{m, i}
///
/// 若 J_{Gamma, i}^{n+1} <= 0 (非物理退化点)，则回退为不放大面积权重的 m_i^d + Δt S_{m, i}
pub fn conservative_modal_update(
    j_gamma_next: f64,
    j_gamma_dep: f64,
    mapping_jacobian: f64,
    m_dep: &[f64; 4],
    source: Option<&[f64; 4]>,
    dt: f64,
) -> [f64; 4] {
    let weight = if j_gamma_next > 1e-14 {
        (mapping_jacobian * j_gamma_dep) / j_gamma_next
    } else {
        1.0
    };

    let mut m_next = [0.0; 4];
    for k in 0..4 {
        let adv = weight * m_dep[k];
        let src = match source {
            Some(s) => dt * s[k],
            None => 0.0,
        };
        m_next[k] = adv + src;
    }
    m_next
}

/// 批量执行全表面节点的守恒模态更新
pub fn conservative_modal_update_batch(
    j_gamma_next_list: &[f64],
    j_gamma_dep_list: &[f64],
    jacobians: &[f64],
    m_dep_list: &[[f64; 4]],
    sources: Option<&[[f64; 4]]>,
    dt: f64,
) -> Vec<[f64; 4]> {
    let n = m_dep_list.len();
    assert_eq!(j_gamma_next_list.len(), n);
    assert_eq!(j_gamma_dep_list.len(), n);
    assert_eq!(jacobians.len(), n);
    if let Some(src) = sources {
        assert_eq!(src.len(), n);
    }

    let mut result = Vec::with_capacity(n);
    for i in 0..n {
        let src_i = sources.map(|s| &s[i]);
        let m_i = conservative_modal_update(
            j_gamma_next_list[i],
            j_gamma_dep_list[i],
            jacobians[i],
            &m_dep_list[i],
            src_i,
            dt,
        );
        result.push(m_i);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_surface_map_jacobian() {
        let identity_grad = [[1.0, 0.0], [0.0, 1.0]];
        assert!((surface_map_jacobian_from_grad(&identity_grad) - 1.0).abs() < 1e-12);

        let stretch_grad = [[1.5, 0.2], [-0.1, 2.0]];
        // det = 1.5 * 2.0 - 0.2 * (-0.1) = 3.0 + 0.02 = 3.02
        assert!((surface_map_jacobian_from_grad(&stretch_grad) - 3.02).abs() < 1e-12);
    }

    #[test]
    fn test_conservative_update_pure_advection() {
        let j_gamma_next = 2.0;
        let j_gamma_dep = 1.8;
        let mapping_j = 1.0;
        let m_dep = [1.0, 2.0, 3.0, 4.0];

        // weight = 1.0 * 1.8 / 2.0 = 0.9
        let m_next = conservative_modal_update(
            j_gamma_next,
            j_gamma_dep,
            mapping_j,
            &m_dep,
            None,
            0.1,
        );

        assert!((m_next[0] - 0.9).abs() < 1e-12);
        assert!((m_next[1] - 1.8).abs() < 1e-12);
        assert!((m_next[2] - 2.7).abs() < 1e-12);
        assert!((m_next[3] - 3.6).abs() < 1e-12);
    }

    #[test]
    fn test_conservative_update_with_source() {
        let j_gamma_next = 1.0;
        let j_gamma_dep = 1.0;
        let mapping_j = 1.0;
        let m_dep = [1.0, 1.0, 1.0, 1.0];
        let src = [0.5, -0.5, 2.0, 0.0];
        let dt = 0.2;

        let m_next = conservative_modal_update(
            j_gamma_next,
            j_gamma_dep,
            mapping_j,
            &m_dep,
            Some(&src),
            dt,
        );

        assert!((m_next[0] - 1.1).abs() < 1e-12);
        assert!((m_next[1] - 0.9).abs() < 1e-12);
        assert!((m_next[2] - 1.4).abs() < 1e-12);
        assert!((m_next[3] - 1.0).abs() < 1e-12);
    }
}
