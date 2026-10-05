//! Degree-dependent spectral gains. The host multiplies `c_lm` before the device runs.

use super::error::ShError;
use super::harmonics::lm_index;

fn unit_gain(name: &str, gain: f64) -> Result<(), ShError> {
    if gain.is_finite() && (0.0..=1.0).contains(&gain) {
        Ok(())
    } else {
        Err(ShError::invalid_input(format!(
            "{name} gain must be a finite number in [0, 1]"
        )))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Filter {
    /// Keep degrees `l ≤ l_cut` and drop the rest.
    Truncate { l_cut: usize },
    /// Heat kernel `exp(-κ l(l+1))` on the sphere.
    Heat { kappa: f64 },
    /// Raised cosine from `l0` (gain 1) to `l1` (gain 0).
    Cosine { l0: f64, l1: f64 },
    /// `exp(-(l/σ)²)`.
    Gaussian { sigma: f64 },
}

impl Filter {
    pub fn validate(&self, l_max: usize) -> Result<(), ShError> {
        match *self {
            Filter::Truncate { l_cut } => {
                if l_cut > l_max {
                    return Err(ShError::invalid_input(format!(
                        "truncate l_cut {l_cut} is above l_max {l_max}"
                    )));
                }
            }
            Filter::Heat { kappa } => {
                if !(kappa.is_finite() && kappa >= 0.0) {
                    return Err(ShError::invalid_input(
                        "heat kappa must be a finite number >= 0",
                    ));
                }
            }
            Filter::Cosine { l0, l1 } => {
                if !(l0.is_finite() && l1.is_finite() && 0.0 <= l0 && l0 < l1) {
                    return Err(ShError::invalid_input("cosine taper needs 0 <= l0 < l1"));
                }
            }
            Filter::Gaussian { sigma } => {
                if !(sigma.is_finite() && sigma > 0.0) {
                    return Err(ShError::invalid_input(
                        "gaussian sigma must be a finite number > 0",
                    ));
                }
            }
        }
        Ok(())
    }

    pub fn gain(&self, l: usize) -> f64 {
        let lf = l as f64;
        match *self {
            Filter::Truncate { l_cut } => {
                if l <= l_cut {
                    1.0
                } else {
                    0.0
                }
            }
            Filter::Heat { kappa } => (-kappa * lf * (lf + 1.0)).exp(),
            Filter::Cosine { l0, l1 } => {
                if lf <= l0 {
                    1.0
                } else if lf >= l1 {
                    0.0
                } else {
                    let t = (lf - l0) / (l1 - l0);
                    0.5 * (1.0 + (std::f64::consts::PI * t).cos())
                }
            }
            Filter::Gaussian { sigma } => (-(lf / sigma).powi(2)).exp(),
        }
    }

    pub fn label(&self) -> String {
        match *self {
            Filter::Truncate { l_cut } => format!("truncate:l_cut={l_cut}"),
            Filter::Heat { kappa } => format!("heat:kappa={kappa}"),
            Filter::Cosine { l0, l1 } => format!("cosine:l0={l0},l1={l1}"),
            Filter::Gaussian { sigma } => format!("gaussian:sigma={sigma}"),
        }
    }

    /// Filtered coefficients, then scales on `Y_2^0` and `Y_4^0`.
    /// A gain of `0.3` attenuates that zonal undulation by 70%. A gain of `0` removes it.
    pub fn ancient(
        &self,
        l_max: usize,
        coeff: &[f64],
        y20_gain: f64,
        y40_gain: f64,
    ) -> Result<Vec<f64>, ShError> {
        unit_gain("Y20", y20_gain)?;
        unit_gain("Y40", y40_gain)?;
        if y20_gain < 1.0 && l_max < 2 {
            return Err(ShError::invalid_input(
                "Y20 oblateness attenuation needs l_max >= 2",
            ));
        }
        if y40_gain < 1.0 && l_max < 4 {
            return Err(ShError::invalid_input(
                "Y40 equatorial-ridge attenuation needs l_max >= 4",
            ));
        }
        let mut out = self.apply(l_max, coeff);
        if l_max >= 2 {
            out[lm_index(2, 0)] *= y20_gain;
        }
        if l_max >= 4 {
            out[lm_index(4, 0)] *= y40_gain;
        }
        Ok(out)
    }

    pub fn apply(&self, l_max: usize, coeff: &[f64]) -> Vec<f64> {
        let mut out = coeff.to_vec();
        for l in 0..=l_max {
            let g = self.gain(l);
            for m in -(l as i32)..=(l as i32) {
                out[lm_index(l, m)] *= g;
            }
        }
        out
    }
}
