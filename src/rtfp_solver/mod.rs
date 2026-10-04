//! Tier 1: ray-traced finite-part (RTFP) gravity.
//!
//! [`rtfp`] is the solver and runs only on hardware ray tracing.
//! [`reference`] holds the Feynman, Carlson and Werner benchmarks, which are
//! used only when an accuracy comparison is requested.

pub mod api;
pub mod error;
pub mod io;
pub mod reference;
pub mod rtfp;
pub mod volume;

pub use api::{FieldBatch, FieldMask, PointRole, PointSet, VolumeIntegrals};
pub use error::RtfpError;
pub use rtfp::{RtfpOptions, RtfpSolver};
