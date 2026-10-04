//! Shared data definitions for every tier.
//!
//! This crate layer holds units, tensors, meshes, and density samples.
//! It does not solve a field equation and it does not touch the filesystem.

pub mod density;
pub mod error;
pub mod geometry;
pub mod math;
pub mod par;
pub mod units;

pub use density::{CauchyDensity, CauchyKernel};
pub use error::CommonError;
pub use geometry::{StarMesh, TriMesh};
pub use math::{DMat3, DVec3, Sym6};
pub use units::UnitSystem;
