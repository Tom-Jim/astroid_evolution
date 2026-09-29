//! 流变学状态变量与微观颗粒物理参数
//! 严格依照 docs/software.md 第 9 节 与 docs/mathstrict.md 第 4 节

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RheologyState {
    /// 宏观静水/正压力 p (Pa)
    pub pressure: f64,
    /// 密实度 / 固体体积分数 nu (无量纲, 0 < nu <= 1)
    pub solid_fraction: f64,
    /// 颗粒平均配位数 Z (无量纲, Z >= 0)
    pub coordination_number: f64,
    /// 颗粒平均粒径 d (m, d > 0)
    pub grain_diameter: f64,
    /// 颗粒本征物质密度 rho_p (kg/m^3, rho_p > 0)
    pub grain_density: f64,
}

impl RheologyState {
    /// 构造流变学状态并进行有效性校验
    pub fn new(
        pressure: f64,
        solid_fraction: f64,
        coordination_number: f64,
        grain_diameter: f64,
        grain_density: f64,
    ) -> Result<Self, String> {
        if solid_fraction <= 0.0 || solid_fraction > 1.0 {
            return Err(format!(
                "固体体积分数 nu 必须在 (0, 1] 范围内，收到: {solid_fraction}"
            ));
        }
        if coordination_number < 0.0 {
            return Err(format!(
                "配位数 Z 必须 >= 0，收到: {coordination_number}"
            ));
        }
        if grain_diameter <= 0.0 {
            return Err(format!(
                "颗粒粒径 d 必须 > 0，收到: {grain_diameter}"
            ));
        }
        if grain_density <= 0.0 {
            return Err(format!(
                "颗粒本征密度 rho_p 必须 > 0，收到: {grain_density}"
            ));
        }

        Ok(Self {
            pressure: pressure.max(0.0),
            solid_fraction,
            coordination_number,
            grain_diameter,
            grain_density,
        })
    }

    /// 宏观堆积介质等效体密度 rho = nu * rho_p
    #[inline]
    pub fn bulk_density(&self) -> f64 {
        self.solid_fraction * self.grain_density
    }
}

/// 经典与修正流变学接触与本构模型参数
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RheologyParams {
    /// 准静态摩擦系数 mu_s
    pub mu_s: f64,
    /// 高速流动态极限摩擦系数 mu_2 (mu_2 > mu_s)
    pub mu_2: f64,
    /// 惯性数特征过渡值 I_0
    pub i_0: f64,
    /// 内聚几何系数 C_c
    pub c_c: f64,
    /// 潮汐应力耦合系数 C_T
    pub c_t: f64,
    /// 单个颗粒接触内聚力 F_H (N)
    pub cohesion_force: f64,
    /// 宏观屈服面黏聚力 c (Pa)
    pub cohesion_macro: f64,
    /// 屈服摩擦系数 mu_y
    pub mu_y: f64,
    /// 投影摩擦系数 mu_proj
    pub mu_proj: f64,
    /// 黏度低剪切率正则化参数 gamma_dot_eps (1/s)
    pub gamma_dot_eps: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RheologyBranch {
    InertialFlow,
    StaticYield,
    RegularizedZeroShear,
}

pub fn select_branch(pressure: f64, shear_rate: f64, pressure_eps: f64, shear_eps: f64) -> RheologyBranch {
    if pressure <= pressure_eps { RheologyBranch::StaticYield }
    else if shear_rate <= shear_eps { RheologyBranch::RegularizedZeroShear }
    else { RheologyBranch::InertialFlow }
}

impl Default for RheologyParams {
    fn default() -> Self {
        Self {
            mu_s: 0.38,
            mu_2: 0.64,
            i_0: 0.279,
            c_c: 1.0,
            c_t: 1.0,
            cohesion_force: 0.0,
            cohesion_macro: 0.0,
            mu_y: 0.38,
            mu_proj: 0.38,
            gamma_dot_eps: 1e-6,
        }
    }
}

#[cfg(test)]
mod branch_tests {
    use super::*;
    #[test]
    fn zero_pressure_uses_static_branch() {
        assert_eq!(select_branch(0.0, 10.0, 1e-12, 1e-12), RheologyBranch::StaticYield);
        assert_eq!(select_branch(10.0, 0.0, 1e-12, 1e-12), RheologyBranch::RegularizedZeroShear);
    }
}
