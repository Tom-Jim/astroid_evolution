//! Tier 2: quasi-static geometric self-consistent-field (SCF) evolution of a rotating
//! granular star-shaped body (mathstrict.md §2.8–2.16, §8–§10).
//!
//! Gravity is Tier 1 (`rtfp_solver`) and the friction-angle chain is `rheo_friction`;
//! both are called, never re-implemented.

pub mod basis;
pub mod body;
pub mod error;
pub mod harmonics;
pub mod jets;
pub mod quad;
pub mod spin;
pub mod forward;
pub mod params;
pub mod stokes;
pub mod classify;
pub mod transport;
pub mod socp;
pub mod frame;
pub mod mechanics;
pub mod relax;
pub mod evolve;
pub mod io;
