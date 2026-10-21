pub mod common;

#[cfg(feature = "tier1")]
pub mod rtfp_solver;

#[cfg(feature = "rheo")]
pub mod rheo_friction;

pub mod sh_smooth;

#[cfg(feature = "tier2")]
pub mod scf_simulator;

#[cfg(any(feature = "tier3", feature = "bevy-web"))]
pub mod asteroid_app;

#[cfg(all(target_arch = "wasm32", any(feature = "tier3", feature = "bevy-web")))]
pub mod browser;
