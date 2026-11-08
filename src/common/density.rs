//! Cauchy density `ρ_K` and its classical derivatives (MS §1.1, §6.1).
//!
//! `ρ_K(y) = bulk + Σ_k w_k [1 + (y − c_k)ᵀ S_k (y − c_k)]^{−α_k}` with a symmetric
//! positive-definite metric `S_k` (dimension L⁻²). The isotropic core is
//! `S_k = σ_k² I`. The core value at `c_k` is `w_k` and its Hessian there is
//! `−2 α_k w_k S_k`; MS §9.2 transports exactly these.

use crate::common::error::SimError;
use crate::common::math::{DMat3, DVec3, Sym6, mat_sym, sym_mat};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CauchyKernel {
    pub center: DVec3,
    pub weight: f64,
    /// `S` in the layout `(xx, yy, zz, xy, xz, yz)`.
    pub metric: Sym6,
    pub alpha: f64,
}

impl CauchyKernel {
    /// An isotropic core `w [1 + σ² |y − c|²]^{−α}`.
    pub fn try_new(center: DVec3, weight: f64, sigma: f64, alpha: f64) -> Result<Self, SimError> {
        if !(sigma.is_finite() && sigma > 0.0) {
            return Err(SimError::model_not_closed(
                "Cauchy kernel sigma must be finite and positive",
            ));
        }
        let s2 = sigma * sigma;
        Self::try_anisotropic(center, weight, [s2, s2, s2, 0.0, 0.0, 0.0], alpha)
    }

    /// A core with metric `S` (symmetric positive definite).
    pub fn try_anisotropic(
        center: DVec3,
        weight: f64,
        metric: Sym6,
        alpha: f64,
    ) -> Result<Self, SimError> {
        if !center.is_finite() || !weight.is_finite() {
            return Err(SimError::model_not_closed(
                "Cauchy kernel center and weight must be finite",
            ));
        }
        if !(alpha.is_finite() && alpha > 0.0) {
            return Err(SimError::model_not_closed(
                "Cauchy kernel alpha must be finite and positive; no implicit alpha=1",
            ));
        }
        if !metric.iter().all(|v| v.is_finite()) || !is_spd(&metric) {
            return Err(SimError::model_not_closed(
                "Cauchy kernel metric must be symmetric positive definite",
            ));
        }
        Ok(Self {
            center,
            weight,
            metric,
            alpha,
        })
    }

    pub fn metric_mat(&self) -> DMat3 {
        sym_mat(self.metric)
    }

    /// `w ∫_{ℝ³} [1 + xᵀSx]^{−α} dx = w π^{3/2} Γ(α − 3/2)/Γ(α) / √det S` for α > 3/2.
    pub fn total_mass(&self) -> Option<f64> {
        if self.alpha <= 1.5 {
            return None;
        }
        let det = self.metric_mat().determinant();
        let ratio = (ln_gamma(self.alpha - 1.5) - ln_gamma(self.alpha)).exp();
        Some(self.weight * std::f64::consts::PI.powf(1.5) * ratio / det.sqrt())
    }
}

fn is_spd(m: &Sym6) -> bool {
    // Sylvester's criterion.
    let a = sym_mat(*m);
    let d1 = m[0];
    let d2 = m[0] * m[1] - m[3] * m[3];
    d1 > 0.0 && d2 > 0.0 && a.determinant() > 0.0
}

/// `ln Γ(x)` for x > 0 (Lanczos, g = 7, relative error below 1e-15).
pub fn ln_gamma(x: f64) -> f64 {
    const G: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        let pi = std::f64::consts::PI;
        return (pi / (pi * x).sin()).ln() - ln_gamma(1.0 - x);
    }
    let x = x - 1.0;
    let mut a = G[0];
    let t = x + 7.5;
    for (i, g) in G.iter().enumerate().skip(1) {
        a += g / (x + i as f64);
    }
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

/// Physical density is `bulk + Σ kernels`. `bulk` may be zero. A negative bulk is rejected.
#[derive(Clone, Debug, PartialEq)]
pub struct CauchyDensity {
    pub bulk: f64,
    pub kernels: Vec<CauchyKernel>,
}

impl CauchyDensity {
    pub fn try_new(bulk: f64, kernels: Vec<CauchyKernel>) -> Result<Self, SimError> {
        if !(bulk.is_finite() && bulk >= 0.0) {
            return Err(SimError::model_not_closed(
                "bulk density must be finite and non-negative",
            ));
        }
        if bulk == 0.0 && kernels.is_empty() {
            return Err(SimError::model_not_closed(
                "density model is empty: bulk is zero and no Cauchy kernel was supplied",
            ));
        }
        Ok(Self { bulk, kernels })
    }

    pub fn uniform(bulk: f64) -> Result<Self, SimError> {
        Self::try_new(bulk, Vec::new())
    }

    #[inline]
    pub fn rho(&self, y: DVec3) -> f64 {
        self.bulk + self.kernels.iter().map(|k| kernel_rho(k, y)).sum::<f64>()
    }

    #[inline]
    pub fn grad(&self, y: DVec3) -> DVec3 {
        self.kernels.iter().map(|k| kernel_grad(k, y)).sum()
    }

    #[inline]
    pub fn hessian(&self, y: DVec3) -> Sym6 {
        let mut h = [0.0; 6];
        for k in &self.kernels {
            let hk = kernel_hessian(k, y);
            for i in 0..6 {
                h[i] += hk[i];
            }
        }
        h
    }
}

/// `q = 1 + dᵀ S d` and `S d`.
#[inline]
fn quad(k: &CauchyKernel, y: DVec3) -> (f64, DVec3) {
    let d = y - k.center;
    let sd = k.metric_mat() * d;
    (1.0 + d.dot(sd), sd)
}

#[inline]
pub fn kernel_rho(k: &CauchyKernel, y: DVec3) -> f64 {
    let (q, _) = quad(k, y);
    k.weight * q.powf(-k.alpha)
}

/// `∇ρ = −2 α w q^{−α−1} S (y − c)`.
#[inline]
pub fn kernel_grad(k: &CauchyKernel, y: DVec3) -> DVec3 {
    let (q, sd) = quad(k, y);
    sd * (-2.0 * k.alpha * k.weight * q.powf(-k.alpha - 1.0))
}

/// `∇∇ρ = −2 α w q^{−α−1} S + 4 α (α + 1) w q^{−α−2} (S d)(S d)ᵀ`, layout `(xx, yy, zz, xy, xz, yz)`.
#[inline]
pub fn kernel_hessian(k: &CauchyKernel, y: DVec3) -> Sym6 {
    let (q, sd) = quad(k, y);
    let a = -2.0 * k.alpha * k.weight * q.powf(-k.alpha - 1.0);
    let b = 4.0 * k.alpha * (k.alpha + 1.0) * k.weight * q.powf(-k.alpha - 2.0);
    let m = k.metric_mat() * a + DMat3::from_cols(sd * sd.x, sd * sd.y, sd * sd.z) * b;
    mat_sym(m)
}
