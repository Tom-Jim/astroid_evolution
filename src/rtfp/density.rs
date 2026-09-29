//! Cauchy 变密度场与核配置
//! 参照 ryugu_cauchy_gravity/src/rust/compute/native/density.rs

use serde::Deserialize;
use std::fs;
use std::path::Path;

#[derive(Clone, Copy, Debug)]
pub struct KernelSi {
    /// 核心位置 (米)
    pub c: [f64; 3],
    /// 衰减尺度 σ (m⁻¹)
    pub sigma: f64,
    /// 权重 w (kg/m³)
    pub w: f64,
    /// 广义指数 α
    pub alpha: f64,
}

#[derive(Clone, Debug, Deserialize)]
pub struct TomlCauchyKernel {
    pub c: [f64; 3],
    pub sigma: f64,
    pub w: f64,
    #[serde(default = "default_alpha")]
    pub alpha: f64,
}

fn default_alpha() -> f64 {
    1.0
}

#[derive(Clone, Debug, Deserialize)]
struct TomlDensityFile {
    #[serde(default = "default_bulk")]
    pub mean_density: f64,
    #[serde(default)]
    pub kernels: Vec<TomlCauchyKernel>,
}

fn default_bulk() -> f64 {
    1190.0
}

#[derive(Clone, Debug)]
pub struct Density {
    kernels: Vec<KernelSi>,
    pub bulk_density: f64,
}

impl Density {
    /// 纯均匀恒定密度
    pub fn homogeneous(bulk: f64) -> Self {
        Self {
            kernels: Vec::new(),
            bulk_density: bulk,
        }
    }

    /// 从 TOML 配置文件加载变密度 Cauchy 混合核 (例如 assets/density/cauchy_elliptic.toml)
    pub fn from_toml(path: &Path) -> Result<Self, String> {
        let content = fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let parsed: TomlDensityFile = toml::from_str(&content)
            .map_err(|e| format!("{}: TOML parse error: {e}", path.display()))?;

        // 此低层 API 只接受 SI；外层项目加载必须通过 UnitSystem 显式转换。
        let kernels_si: Vec<KernelSi> = parsed
            .kernels
            .iter()
            .map(|k| KernelSi {
                c: k.c,
                sigma: k.sigma,
                w: k.w,
                alpha: k.alpha,
            })
            .collect();

        Ok(Self {
            kernels: kernels_si,
            bulk_density: parsed.mean_density,
        })
    }

    /// 龙宫 (Ryugu) 默认 Cauchy 变密度预设 (基底 + 十字不对称异常结构)
    pub fn default_ryugu() -> Self {
        Self {
            kernels: vec![
                KernelSi { c: [0.0, 0.0, 0.0], sigma: 0.8 / 1000.0, w: 920.0, alpha: 1.0 },
                KernelSi { c: [0.0, 0.0, 0.0], sigma: 2.0 / 1000.0, w: 150.0, alpha: 1.0 },
                KernelSi { c: [0.0, 250.0, 0.0], sigma: 3.5 / 1000.0, w: 40.0, alpha: 1.0 },
                KernelSi { c: [0.0, -250.0, 0.0], sigma: 3.5 / 1000.0, w: 40.0, alpha: 1.0 },
                // 表面近区致密块
                KernelSi { c: [0.0, 0.0, 432.0], sigma: 25.0 / 1000.0, w: 3500.0, alpha: 1.0 },
            ],
            bulk_density: 1190.0,
        }
    }

    pub fn kernels(&self) -> &[KernelSi] {
        &self.kernels
    }

    /// 计算给定位置 x 处的密度: ρ(x) = Σ_k w_k * [1 + σ_k² ||x - c_k||²]^(-α_k)
    pub fn rho(&self, x: &[f64; 3]) -> f64 {
        if self.kernels.is_empty() {
            return self.bulk_density;
        }
        let mut sum = 0.0;
        for k in &self.kernels {
            let dx = x[0] - k.c[0];
            let dy = x[1] - k.c[1];
            let dz = x[2] - k.c[2];
            let r2 = dx * dx + dy * dy + dz * dz;
            let denom = 1.0 + k.sigma * k.sigma * r2;
            sum += k.w * denom.powf(-k.alpha);
        }
        sum
    }

    /// 转换为跨模块数据契约结构 DensityKernel 集合
    pub fn to_density_kernels(&self) -> Vec<crate::model_calculation::contracts::DensityKernel> {
        self.kernels
            .iter()
            .map(|k| crate::model_calculation::contracts::DensityKernel {
                center: k.c,
                sigma: k.sigma,
                weight: k.w,
                alpha: k.alpha,
            })
            .collect()
    }
}
