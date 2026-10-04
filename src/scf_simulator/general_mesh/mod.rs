//! Non-star Tier2 path. Surface degrees of freedom are vectors in R^3.
//! This module does not call radial radii, radial Galerkin, or radial SOCP.

pub mod geometry;
pub mod geometry_update;
pub mod loop_;
pub mod pressure;
pub mod rheology;
pub mod stokes;
pub mod volume;
pub mod yielding;

pub use loop_::execute;
