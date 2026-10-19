//! Tier 2: quasi-static geometric self-consistent-field (SCF) evolution of a rotating
//! granular star-shaped body (mathstrict.md §2.8–2.16, §8–§10).
//!
//! Gravity is Tier 1 (`rtfp_solver`) and the friction-angle chain is `rheo_friction`;
//! both are called, never re-implemented.

pub mod basis;
#[cfg(target_os = "macos")]
mod basis_gpu;
pub mod body;
pub mod classify;
pub mod error;
pub mod evolve;
pub mod forward;
pub mod frame;
pub mod harmonics;
pub mod io;
pub mod jets;
mod mode_c_gpu;
mod mode_c_spectral;
pub mod mechanics;
pub mod params;
pub mod quad;
pub mod relax;
pub mod socp;
mod socp_gpu;
mod socp_pdhg;
pub mod spin;
pub mod stokes;
pub mod timing;
pub mod transport;
