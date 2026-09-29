//! 单位系统与显式单位转换
//! 避免配置单位和代码内部单位产生隐式转换混乱
//! 内部标准物理量全部使用 SI 单位 (m, kg, s, rad, kg/m³)

use std::f64::consts::PI;

/// 长度单位
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LengthUnit {
    Meter,
    Kilometer,
    Millimeter,
    AstronomicalUnit,
}

impl LengthUnit {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s.trim().to_lowercase().as_str() {
            "m" | "meter" | "meters" | "metre" => Ok(Self::Meter),
            "km" | "kilometer" | "kilometers" => Ok(Self::Kilometer),
            "mm" | "millimeter" | "millimeters" => Ok(Self::Millimeter),
            "au" | "astronomical_unit" => Ok(Self::AstronomicalUnit),
            other => Err(format!("未知长度单位: '{other}'，支持: m, km, mm, au")),
        }
    }

    /// 转换为 SI 标准米 (m) 的比例因子
    pub fn to_meters_scale(&self) -> f64 {
        match self {
            Self::Meter => 1.0,
            Self::Kilometer => 1_000.0,
            Self::Millimeter => 0.001,
            Self::AstronomicalUnit => 149_597_870_700.0,
        }
    }
}

/// 密度单位
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DensityUnit {
    KgPerM3,
    GPerCm3,
}

impl DensityUnit {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s.trim().to_lowercase().as_str() {
            "kg/m3" | "kg/m^3" | "kg_m3" => Ok(Self::KgPerM3),
            "g/cm3" | "g/cm^3" | "g_cm3" => Ok(Self::GPerCm3),
            other => Err(format!("未知密度单位: '{other}'，支持: kg/m3, g/cm3")),
        }
    }

    /// 转换为 SI 标准 kg/m³ 的比例因子
    pub fn to_si_scale(&self) -> f64 {
        match self {
            Self::KgPerM3 => 1.0,
            Self::GPerCm3 => 1_000.0, // 1 g/cm³ = 1000 kg/m³
        }
    }
}

/// 角速度单位
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AngularVelocityUnit {
    RadPerSec,
    RevPerDay,
    Rpm,
}

impl AngularVelocityUnit {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s.trim().to_lowercase().as_str() {
            "rad/s" | "rad_s" | "rad/sec" => Ok(Self::RadPerSec),
            "rev/day" | "rot/day" => Ok(Self::RevPerDay),
            "rpm" | "rev/min" => Ok(Self::Rpm),
            other => Err(format!("未知角速度单位: '{other}'，支持: rad/s, rev/day, rpm")),
        }
    }

    /// 转换为 SI 标准 rad/s 的比例因子
    pub fn to_si_scale(&self) -> f64 {
        match self {
            Self::RadPerSec => 1.0,
            Self::RevPerDay => 2.0 * PI / 86_400.0,
            Self::Rpm => 2.0 * PI / 60.0,
        }
    }
}

/// 显式单位系统配置
#[derive(Clone, Debug)]
pub struct UnitSystem {
    pub length: LengthUnit,
    pub density: DensityUnit,
    pub angular_velocity: AngularVelocityUnit,
}

impl Default for UnitSystem {
    fn default() -> Self {
        Self {
            length: LengthUnit::Meter,
            density: DensityUnit::KgPerM3,
            angular_velocity: AngularVelocityUnit::RadPerSec,
        }
    }
}

impl UnitSystem {
    pub fn new(length_str: &str, density_str: &str) -> Result<Self, String> {
        Ok(Self {
            length: LengthUnit::parse(length_str)?,
            density: DensityUnit::parse(density_str)?,
            angular_velocity: AngularVelocityUnit::RadPerSec,
        })
    }

    /// 长度值缩放到米
    #[inline]
    pub fn to_meters(&self, val: f64) -> f64 {
        val * self.length.to_meters_scale()
    }

    /// 三维坐标缩放到米
    #[inline]
    pub fn coords_to_meters(&self, p: [f64; 3]) -> [f64; 3] {
        let s = self.length.to_meters_scale();
        [p[0] * s, p[1] * s, p[2] * s]
    }

    /// 密度缩放到 kg/m³
    #[inline]
    pub fn to_si_density(&self, val: f64) -> f64 {
        val * self.density.to_si_scale()
    }

    /// Cauchy 核参数显式转换到 SI:
    /// - c_k 具有长度量纲: c_SI = c * scale_length
    /// - sigma_k 具有 1/长度量纲: sigma_SI = sigma / scale_length
    /// - w_k 具有密度量纲: w_SI = w * scale_density
    /// - alpha_k 为无量纲量
    pub fn convert_cauchy_kernel(
        &self,
        center: [f64; 3],
        sigma: f64,
        weight: f64,
        alpha: f64,
    ) -> ([f64; 3], f64, f64, f64) {
        let l_scale = self.length.to_meters_scale();
        let rho_scale = self.density.to_si_scale();
        let c_si = [center[0] * l_scale, center[1] * l_scale, center[2] * l_scale];
        let sigma_si = sigma / l_scale;
        let w_si = weight * rho_scale;
        (c_si, sigma_si, w_si, alpha)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_length_conversion() {
        let u = UnitSystem::new("km", "kg/m3").unwrap();
        assert_eq!(u.to_meters(1.5), 1500.0);
        assert_eq!(u.coords_to_meters([1.0, 2.0, 3.0]), [1000.0, 2000.0, 3000.0]);
    }

    #[test]
    fn test_cauchy_kernel_conversion() {
        let u = UnitSystem::new("km", "g/cm3").unwrap();
        // km: l_scale = 1000. sigma in 1/km -> 1/m is sigma / 1000
        // g/cm3: rho_scale = 1000. w in g/cm3 -> kg/m3 is w * 1000
        let (c, sig, w, a) = u.convert_cauchy_kernel([0.1, 0.2, 0.3], 2.0, 1.2, 1.5);
        assert_eq!(c, [100.0, 200.0, 300.0]);
        assert_eq!(sig, 0.002);
        assert_eq!(w, 1200.0);
        assert_eq!(a, 1.5);
    }
}
