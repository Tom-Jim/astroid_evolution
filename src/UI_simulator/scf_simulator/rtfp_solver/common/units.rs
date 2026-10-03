//! SI ↔ internal units. `L0` is the reference length, `ρ0` the reference density.
//!
//! Internal gravity uses `G' = 1` after the potential is scaled by `G ρ0 L0²`.

use crate::common::error::CommonError;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UnitSystem {
    pub length_m: f64,
    pub density_kg_m3: f64,
    pub g_si: f64,
}

impl UnitSystem {
    pub fn try_new(length_m: f64, density_kg_m3: f64, g_si: f64) -> Result<Self, CommonError> {
        if !(length_m.is_finite() && length_m > 0.0) {
            return Err(CommonError::model_not_closed(
                "UnitSystem.length_m must be a finite positive reference length",
            ));
        }
        if !(density_kg_m3.is_finite() && density_kg_m3 > 0.0) {
            return Err(CommonError::model_not_closed(
                "UnitSystem.density_kg_m3 must be a finite positive reference density",
            ));
        }
        if !(g_si.is_finite() && g_si > 0.0) {
            return Err(CommonError::model_not_closed(
                "UnitSystem.g_si must be a finite positive gravitational constant",
            ));
        }
        Ok(Self {
            length_m,
            density_kg_m3,
            g_si,
        })
    }

    #[inline]
    pub fn potential_scale(self) -> f64 {
        self.g_si * self.density_kg_m3 * self.length_m * self.length_m
    }

    #[inline]
    pub fn length_to_internal(self, meters: f64) -> f64 {
        meters / self.length_m
    }

    #[inline]
    pub fn length_to_si(self, internal: f64) -> f64 {
        internal * self.length_m
    }

    #[inline]
    pub fn density_to_internal(self, kg_m3: f64) -> f64 {
        kg_m3 / self.density_kg_m3
    }

    #[inline]
    pub fn potential_to_internal(self, u_si: f64) -> f64 {
        u_si / self.potential_scale()
    }

    #[inline]
    pub fn potential_to_si(self, u_internal: f64) -> f64 {
        u_internal * self.potential_scale()
    }
}
