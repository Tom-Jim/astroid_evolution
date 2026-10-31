//! Evolving material state and angular momentum (MS Chapter 11).
//!
//! Each hypothesis is an optional project section; an absent section is the switch in
//! its off position, and the loop then reduces exactly to §10.4.4 (Theorem 11.13).
//!
//! * `spin`, A-L (§11.7): the spin follows the angular momentum and the settled inertia.
//! * `thermal`, A-Θ (§11.6): diurnal thermal fatigue of the surface skin and the
//!   radiative torque of law L2.
//! * `markers`, `material`, A-ϖ (§11.5) and the marker carriers of §11.1–11.2.
//! * `dilatancy`, A-Ξ (§11.3–11.4).

pub mod dilatancy;
pub mod dilatation;
pub mod markers;
pub mod material;
pub mod spin;
pub mod thermal;

use serde::{Deserialize, Serialize};

pub use spin::{SpinLaw, SpinLawParams, SpinReport, SpinState};
pub use dilatancy::DilatancyParams;
pub use markers::{MarkerParams, Markers};
pub use material::{DamageParams, MaterialReport, StateField};
pub use thermal::{Thermal, ThermalParams, ThermalReport};

/// The persistent state of Chapter 11, carried by the run state and the snapshot.
/// Every field is empty while its hypothesis is off.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Evolving {
    /// A-L: spin angular momentum and its torque reference.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spin: Option<SpinState>,
    /// A-Θ: surface damage `ϖ_Γ` per vertex.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub surface_damage: Option<Vec<f64>>,
    /// A-ϖ, A-Ξ: the state markers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub markers: Option<Markers>,
    /// A-O (§10.4.6): the semi-major axis and the mass ledger.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub orbit: Option<super::orbit::OrbitState>,
}
