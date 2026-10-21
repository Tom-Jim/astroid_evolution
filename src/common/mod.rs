//! Shared data definitions for every tier.
//!
//! This crate layer holds tensors, meshes, and density samples.
//! It does not solve a field equation and it does not touch the filesystem.

pub mod density;
pub mod error;
pub mod geometry;
pub mod math;
pub mod time;

pub use density::{CauchyDensity, CauchyKernel};
pub use error::CommonError;
pub use geometry::TriMesh;
pub use math::{DVec3, Sym6};
