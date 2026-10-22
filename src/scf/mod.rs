//! Tier 2: quasi-static geometric self-consistent-field (SCF) evolution of a rotating
//! granular star-shaped body (MS §2.8–2.16, §8–§10).
//!
//! Gravity is Tier 1 (`rtfp`) and the friction-angle chain is `rheo`; both are
//! called, never re-implemented.

pub mod basis;
mod basis_gpu;
pub mod body;
pub mod classify;
pub mod cores;
pub mod evolve;
pub mod forward;
pub mod frame;
pub mod harmonics;
pub mod history;
pub mod io;
pub mod jets;
pub mod mechanics;
mod mode_c_gpu;
pub mod mode_c_spectral;
pub mod params;
pub mod quad;
pub mod relax;
pub mod socp;
pub mod socp_gpu;
pub mod socp_pdhg;
pub mod spin;
pub mod stokes;
pub mod timing;
pub mod transport;
