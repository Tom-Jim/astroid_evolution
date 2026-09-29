//! 经典与状态依赖动态摩擦系数定律
//! 严格依照 docs/software.md 第 9.3 节 与 docs/mathstrict.md 第 4.1、4.5 节

/// 计算传统 Jop-Forterre-Pouliquen mu(I) 经典摩擦系数
///
/// mu_cl(I) = mu_s + (mu_2 - mu_s) * I / (I_0 + I)
#[inline]
pub fn mu_classic(i: f64, mu_s: f64, mu_2: f64, i_0: f64) -> f64 {
    if i <= 0.0 {
        return mu_s;
    }
    let denom = i_0 + i;
    if denom <= 1e-14 {
        return mu_s;
    }
    mu_s + (mu_2 - mu_s) * (i / denom)
}

/// 计算扩展联合模型的动态摩擦系数 mu_d(I_c)
///
/// 公式依据 docs/software.md 第 9.3 节 与 docs/mathstrict.md 式 (3825):
/// mu_d(I_c) = mu_s + (mu_2 - mu_s) * I_c / (I_0 + I_c)
#[inline]
pub fn mu_dynamic(i_c: f64, mu_s: f64, mu_2: f64, i_0: f64) -> f64 {
    if i_c <= 0.0 {
        return mu_s;
    }
    let denom = i_0 + i_c;
    if denom <= 1e-14 {
        return mu_s;
    }
    mu_s + (mu_2 - mu_s) * (i_c / denom)
}

/// 状态依赖摩擦系数（考虑密实度与配位数影响的唯象参数修正）
///
/// 当密实度 nu 增加或配位数 Z 增大时，咬合作用增强，静摩擦角升高。
#[inline]
pub fn state_dependent_friction_params(
    base_mu_s: f64,
    base_mu_2: f64,
    base_i_0: f64,
    solid_fraction: f64,
    coordination_number: f64,
    reference_nu: f64,
    reference_z: f64,
) -> (f64, f64, f64) {
    let delta_nu = (solid_fraction - reference_nu).clamp(-0.2, 0.2);
    let delta_z = ((coordination_number - reference_z) / reference_z.max(1.0)).clamp(-0.5, 0.5);

    let mu_s = (base_mu_s * (1.0 + 0.5 * delta_nu + 0.2 * delta_z)).max(0.01);
    let mu_2 = (base_mu_2 * (1.0 + 0.3 * delta_nu + 0.1 * delta_z)).max(mu_s + 0.01);
    let i_0 = (base_i_0 * (1.0 - 0.2 * delta_nu)).max(1e-4);

    (mu_s, mu_2, i_0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_friction_laws() {
        let mu_s = 0.38;
        let mu_2 = 0.64;
        let i_0 = 0.279;

        // I = 0 时退化为静摩擦 mu_s
        assert!((mu_classic(0.0, mu_s, mu_2, i_0) - mu_s).abs() < 1e-12);
        assert!((mu_dynamic(0.0, mu_s, mu_2, i_0) - mu_s).abs() < 1e-12);

        // I -> 无穷大 时趋近于 mu_2
        assert!((mu_classic(1e8, mu_s, mu_2, i_0) - mu_2).abs() < 1e-6);
        assert!((mu_dynamic(1e8, mu_s, mu_2, i_0) - mu_2).abs() < 1e-6);

        // I = I_0 时取 (mu_s + mu_2) / 2
        let mid = (mu_s + mu_2) / 2.0;
        assert!((mu_classic(i_0, mu_s, mu_2, i_0) - mid).abs() < 1e-12);
    }
}
