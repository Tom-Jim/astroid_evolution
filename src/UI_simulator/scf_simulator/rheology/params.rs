#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RheologyState {
    pub pressure: f64,
    pub solid_fraction: f64,
    pub coordination_number: f64,
    pub grain_diameter: f64,
    pub grain_density: f64,
}
impl RheologyState {
    pub fn new(
        pressure: f64,
        solid_fraction: f64,
        coordination_number: f64,
        grain_diameter: f64,
        grain_density: f64,
    ) -> Result<Self, String> {
        if !pressure.is_finite()
            || !solid_fraction.is_finite()
            || !coordination_number.is_finite()
            || !grain_diameter.is_finite()
            || !grain_density.is_finite()
        {
            return Err("流变状态参数必须全部为有限数".into());
        }
        if pressure < 0.0 {
            return Err(format!("压力 p 必须为非负数，收到: {pressure}"));
        }
        if solid_fraction <= 0.0 || solid_fraction > 1.0 {
            return Err(format!(
                "固体体积分数 nu 必须在 (0, 1] 范围内，收到: {solid_fraction}"
            ));
        }
        if coordination_number < 0.0 {
            return Err(format!("配位数 Z 必须 >= 0，收到: {coordination_number}"));
        }
        if grain_diameter <= 0.0 {
            return Err(format!("颗粒粒径 d 必须 > 0，收到: {grain_diameter}"));
        }
        if grain_density <= 0.0 {
            return Err(format!(
                "颗粒本征密度 rho_p 必须 > 0，收到: {grain_density}"
            ));
        }
        Ok(Self {
            pressure,
            solid_fraction,
            coordination_number,
            grain_diameter,
            grain_density,
        })
    }
    #[inline]
    pub fn bulk_density(&self) -> f64 {
        self.solid_fraction * self.grain_density
    }
}
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RheologyParams {
    pub mu_s: f64,
    pub mu_2: f64,
    pub i_0: f64,
    pub c_c: f64,
    pub c_t: f64,
    pub cohesion_force: f64,
    pub cohesion_macro: f64,
    pub mu_y: f64,
    pub mu_proj: f64,
    pub gamma_dot_eps: f64,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RheologyBranch {
    InertialFlow,
    StaticYield,
    RegularizedZeroShear,
}
pub fn select_branch(
    pressure: f64,
    shear_rate: f64,
    pressure_eps: f64,
    shear_eps: f64,
) -> RheologyBranch {
    if pressure <= pressure_eps {
        RheologyBranch::StaticYield
    } else if shear_rate <= shear_eps {
        RheologyBranch::RegularizedZeroShear
    } else {
        RheologyBranch::InertialFlow
    }
}
impl Default for RheologyParams {
    fn default() -> Self {
        Self {
            mu_s: 0.75,
            mu_2: 1.10,
            i_0: 0.279,
            c_c: 1.0,
            c_t: 1.0,
            cohesion_force: 5e-8,
            cohesion_macro: 10.0,
            mu_y: 0.75,
            mu_proj: 0.75,
            gamma_dot_eps: 1e-6,
        }
    }
}
#[cfg(test)]
mod branch_tests {
    use super::*;
    #[test]
    fn zero_pressure_uses_static_branch() {
        assert_eq!(
            select_branch(0.0, 10.0, 1e-12, 1e-12),
            RheologyBranch::StaticYield
        );
        assert_eq!(
            select_branch(10.0, 0.0, 1e-12, 1e-12),
            RheologyBranch::RegularizedZeroShear
        );
    }
}
