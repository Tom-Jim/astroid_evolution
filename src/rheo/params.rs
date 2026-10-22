//! Material constants of the joint rheology (mathstrict.md §4, §10.3).
//!
//! Every constant is required: there are no defaults, and a missing or
//! out-of-range one is `ModelNotClosed` (§10.4.4 step 0).

use serde::Deserialize;

use crate::common::error::SimError;

/// How the friction angle is projected (§10.3 flow-chain switch).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MuProjMode {
    /// `mu_proj = mu_d`: the shear-rate chain acts on the geometric angle.
    Dynamic,
    /// `mu_proj = mu_y = mu_s`: a static geometric relaxation proxy.
    Static,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaterialSpec {
    /// Packing fraction `nu_0`.
    pub nu0: f64,
    /// Coordination number `Z_0`.
    pub z0: f64,
    /// Grain diameter `d_p`.
    pub d_p: f64,
    /// Grain density `rho_p`.
    pub rho_p: f64,
    pub mu_s: f64,
    pub mu_2: f64,
    pub i_0: f64,
    /// Macroscopic cohesion `c_b` of the interior (§4.3a).
    pub cohesion: f64,
    /// Cohesion `c_s` of the free surface: the weak regolith layer (§4.3a).
    pub cohesion_surface: f64,
    /// e-folding depth `h_c` of the regolith-to-interior transition (§4.3a).
    pub cohesion_depth: f64,
    /// Cohesive pressure factor `C_c`.
    pub c_c: f64,
    /// Contact cohesion force `F_H`.
    pub f_h: f64,
    /// Tidal back-stress factor `C_T`.
    pub c_t: f64,
    /// Shear-rate regularisation `gd_eps`.
    pub gamma_eps: f64,
    /// Pressure floor `p_min`.
    pub p_min: f64,
    /// Active surface-layer thickness `h_s`.
    pub h_s: f64,
    /// Overhang margin `epsilon` in `g . n <= -epsilon |g|`.
    pub eps_overhang: f64,
    /// `"dynamic"` or `"static"`.
    pub mu_proj_mode: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SolverSpec {
    /// Cap of the per-node frozen-viscosity fixed point.
    pub max_iter: u32,
    /// Relative tolerance of `gamma^2` in that fixed point.
    pub tol: f64,
}

#[derive(Clone, Debug)]
pub struct Material {
    pub spec: MaterialSpec,
    pub mode: MuProjMode,
    /// `rho_bulk = rho_p nu_0`.
    pub rho_bulk: f64,
    /// `p_c = C_c Z_0 nu_0 F_H / d_p^2`.
    pub p_c: f64,
    /// `mu_y = mu_s` (§2.8 B2).
    pub mu_y: f64,
    /// `C_T rho_p nu_0 d_p^2`, the factor of `T_tide = coef * dev H_W`.
    pub tide_coef: f64,
    pub max_iter: u32,
    pub tol: f64,
}

fn positive(name: &str, v: f64) -> Result<(), SimError> {
    if v.is_finite() && v > 0.0 {
        Ok(())
    } else {
        Err(SimError::model_not_closed(format!(
            "{name} must be finite and positive, got {v}"
        )))
    }
}

fn non_negative(name: &str, v: f64) -> Result<(), SimError> {
    if v.is_finite() && v >= 0.0 {
        Ok(())
    } else {
        Err(SimError::model_not_closed(format!(
            "{name} must be finite and non-negative, got {v}"
        )))
    }
}

impl Material {
    pub fn new(spec: MaterialSpec, solver: SolverSpec) -> Result<Self, SimError> {
        positive("nu0", spec.nu0)?;
        if spec.nu0 >= 1.0 {
            return Err(SimError::model_not_closed("nu0 must be below 1"));
        }
        positive("z0", spec.z0)?;
        positive("d_p", spec.d_p)?;
        positive("rho_p", spec.rho_p)?;
        positive("mu_s", spec.mu_s)?;
        positive("i_0", spec.i_0)?;
        positive("gamma_eps", spec.gamma_eps)?;
        positive("p_min", spec.p_min)?;
        positive("h_s", spec.h_s)?;
        non_negative("cohesion", spec.cohesion)?;
        non_negative("cohesion_surface", spec.cohesion_surface)?;
        positive("cohesion_depth", spec.cohesion_depth)?;
        non_negative("c_c", spec.c_c)?;
        non_negative("f_h", spec.f_h)?;
        non_negative("c_t", spec.c_t)?;
        non_negative("eps_overhang", spec.eps_overhang)?;
        if spec.eps_overhang >= 1.0 {
            return Err(SimError::model_not_closed("eps_overhang must be below 1"));
        }
        if !(spec.mu_2.is_finite() && spec.mu_2 > spec.mu_s) {
            return Err(SimError::model_not_closed("mu_2 must exceed mu_s"));
        }
        if solver.max_iter == 0 {
            return Err(SimError::model_not_closed("max_iter must be positive"));
        }
        positive("tol", solver.tol)?;
        let mode = match spec.mu_proj_mode.as_str() {
            "dynamic" => MuProjMode::Dynamic,
            "static" => MuProjMode::Static,
            other => {
                return Err(SimError::model_not_closed(format!(
                    "mu_proj_mode must be \"dynamic\" or \"static\", got \"{other}\""
                )));
            }
        };
        let rho_bulk = spec.rho_p * spec.nu0;
        let p_c = spec.c_c * spec.z0 * spec.nu0 * spec.f_h / (spec.d_p * spec.d_p);
        let tide_coef = spec.c_t * spec.rho_p * spec.nu0 * spec.d_p * spec.d_p;
        Ok(Self {
            mu_y: spec.mu_s,
            rho_bulk,
            p_c,
            tide_coef,
            mode,
            max_iter: solver.max_iter,
            tol: solver.tol,
            spec,
        })
    }
}

impl Material {
    /// `c(d) = c_b + (c_s − c_b) e^{−d/h_c}` at depth `d ≥ 0` below the free surface (§4.3a).
    pub fn cohesion_at(&self, depth: f64) -> f64 {
        let s = &self.spec;
        s.cohesion + (s.cohesion_surface - s.cohesion) * (-depth.max(0.0) / s.cohesion_depth).exp()
    }

    /// `dc/dd` of the same profile.
    pub fn cohesion_slope(&self, depth: f64) -> f64 {
        let s = &self.spec;
        -(s.cohesion_surface - s.cohesion) / s.cohesion_depth
            * (-depth.max(0.0) / s.cohesion_depth).exp()
    }
}
