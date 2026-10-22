//! Standalone rheology and friction-angle solver (mathstrict.md §4, §10.2, §10.3).
//!
//! The chain runs on the GPU in double-single arithmetic:
//! velocity coefficients -> surface velocity trace (Prop. 10.2) -> curvature-aware
//! surface shear rate (§10.2.3) -> `I_c`, `mu_d`, `eta` (§4) -> `mu_eff`, `phi` (§10.3).
//! It is independent of the RTFP tier: the gravity, pressure and tide tensor it
//! reads are plain per-node inputs.

pub mod basis;
pub mod closure;
pub mod gpu;
pub mod model;
pub mod params;
pub mod project;
pub mod reference;

pub use gpu::{ChainResult, ChainStats, NodeResult, RheoGpu};
pub use model::{Environment, NodeBatch, StarSurface, VelocityField};
pub use params::{Material, MaterialSpec, MuProjMode, SolverSpec};
