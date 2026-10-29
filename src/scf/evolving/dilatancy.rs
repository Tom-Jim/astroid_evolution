//! Hypothesis A-Ξ (§11.3–11.4): rate-type dilatancy and the material solid fraction.

use serde::Deserialize;

use crate::common::error::SimError;

use super::markers::Range;

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DilatancyParams {
    /// `K_Ξ ≥ 0` of `Ξ = K_Ξ (ν − ν_c(I_c))`.
    pub k: f64,
    /// `ν_c(I_c) = ν_c⁰ − Δν_c I_c / (I_ν + I_c)`.
    pub nu_c0: f64,
    pub d_nu: f64,
    pub i_nu: f64,
    /// Uniform `ν⁽⁰⁾` and the deposit value `ν_dep`.
    pub initial: f64,
    pub deposit: f64,
    /// Random close packing: `ν_hi` must stay below it.
    pub close_packing: f64,
    /// `θ_Ξ ≤ 1` of the step bound `Δt_Ξ`.
    pub theta_step: f64,
    /// Degree `d` of the polynomial space of the dilatational potential `χ_h` and its
    /// regularization `ε̂_χ` (§11.4).
    pub degree: usize,
    pub regularization: f64,
    /// Level 3 spawn (§11.4): at most `spawn_max` cores per step (0 disables), exponent,
    /// metric scale, and the factor by which `e_ρ` must drop to keep one.
    pub spawn_max: usize,
    pub spawn_alpha: f64,
    pub spawn_scale: f64,
    pub spawn_gain: f64,
}

impl DilatancyParams {
    /// `[ν_lo, ν_hi]` of A-Ξ item 4.
    pub fn range(&self) -> Range {
        Range {
            lo: self.initial.min(self.nu_c0 - self.d_nu).min(self.deposit),
            hi: self.initial.max(self.nu_c0).max(self.deposit),
        }
    }

    /// `μ_min = μ_s − K_Ξ (ν_c⁰ − ν_lo)`.
    pub fn mu_min(&self, mu_s: f64) -> f64 {
        mu_s - self.k * (self.nu_c0 - self.range().lo)
    }

    pub fn validate(&self, mu_s: f64) -> Result<(), SimError> {
        let open = |v: f64| v.is_finite() && v > 0.0 && v < 1.0;
        if !(self.k.is_finite() && self.k >= 0.0)
            || !open(self.nu_c0)
            || !(self.d_nu.is_finite() && self.d_nu >= 0.0 && self.d_nu < self.nu_c0)
            || !(self.i_nu.is_finite() && self.i_nu > 0.0)
            || !open(self.initial)
            || !open(self.deposit)
            || !open(self.close_packing)
        {
            return Err(SimError::model_not_closed(
                "dilatancy needs k >= 0, 0 < ν_c⁰, ν⁽⁰⁾, ν_dep, close_packing < 1, 0 <= Δν_c < ν_c⁰, I_ν > 0",
            ));
        }
        if self.range().hi >= self.close_packing {
            return Err(SimError::model_not_closed("dilatancy: ν_hi must stay below close packing"));
        }
        if !(self.mu_min(mu_s) > 0.0) {
            return Err(SimError::model_not_closed(format!(
                "dilatancy: μ_min = μ_s − K_Ξ (ν_c⁰ − ν_lo) = {:.3} must be positive",
                self.mu_min(mu_s)
            )));
        }
        if !(self.theta_step > 0.0 && self.theta_step <= 1.0) {
            return Err(SimError::model_not_closed("dilatancy.theta_step must lie in (0, 1]"));
        }
        if !(2..=8).contains(&self.degree) || !(self.regularization.is_finite() && self.regularization > 0.0) {
            return Err(SimError::model_not_closed(
                "dilatancy.degree must lie in 2..=8 and dilatancy.regularization be positive",
            ));
        }
        if self.spawn_max > 0
            && !(self.spawn_alpha > 1.5 && self.spawn_scale > 0.0 && self.spawn_gain > 0.0 && self.spawn_gain < 1.0)
        {
            return Err(SimError::model_not_closed(
                "dilatancy spawn needs spawn_alpha > 3/2, spawn_scale > 0 and spawn_gain in (0, 1)",
            ));
        }
        Ok(())
    }

    /// `ν_c(I_c)`.
    pub fn nu_c(&self, inertial: f64) -> f64 {
        self.nu_c0 - self.d_nu * inertial / (self.i_nu + inertial)
    }

    /// Proposition 11.5: `ν' = ν_c / (1 − (1 − ν_c/ν) e^{−K_Ξ ν_c Δγ_p})`.
    pub fn exact_nu(&self, nu: f64, nu_c: f64, d_gamma: f64) -> f64 {
        if !(d_gamma > 0.0) || self.k == 0.0 {
            return nu;
        }
        nu_c / (1.0 - (1.0 - nu_c / nu) * (-self.k * nu_c * d_gamma).exp())
    }
}
