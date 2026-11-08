//! Tier 2: quasi-static geometric self-consistent-field (SCF) evolution of a rotating
//! granular body on the general polyhedral domain of hypothesis (G) (MS §10.0–10.0.11, §8–§10).
//!
//! Gravity is Tier 1 (`rtfp`) and the friction-angle chain is `rheo`; both are
//! called, never re-implemented.

pub mod basis;
mod basis_gpu;
pub mod body;
pub mod breakup;
pub mod cell_bvh;
pub mod classify;
pub mod cores;
pub mod cycle;
pub mod domain;
pub mod evolve;
pub mod evolving;
pub mod forward;
pub mod frame;
pub mod gbasis;
pub mod gbasis_gpu;
pub mod ggram_gpu;
pub mod harmonics;
pub mod history;
pub mod io;
pub mod jets;
pub mod mechanics;
pub mod nbody;
pub mod orbit;
pub mod params;
pub mod quad;
pub mod relax;
pub mod relief;
pub mod runout;
pub mod settle;
pub mod socp;
pub mod socp_ipm;
pub mod spin;
pub mod stokes;
pub mod surface;
pub mod timing;
pub mod transport;
