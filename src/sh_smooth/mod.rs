//! Spherical-harmonic smoothing of a triangle mesh.
//!
//! The host fits `R(ω̂) = Σ c_lm Y_lm(ω̂)`, keeps `residual = R_orig - R_macro`,
//! and filters the low-frequency coefficients (including an optional `Y_4^0` ridge scale).
//! The device evaluates `R_final = R_ancient + residual` at every input vertex.
//! Vertex positions move radially. The face list is copied unchanged, so the
//! output has the same face count and the same triangle indices as the input.

pub mod error;
pub mod eval;
pub mod filter;
pub mod fit;
pub mod fuse;
pub mod gpu;
pub mod harmonics;
mod math;
pub mod mesh;
pub mod selfcheck;
pub mod smooth;

pub use error::ShError;
pub use filter::Filter;
pub use fuse::{FuseOptions, FuseOutput, FuseReport, fuse};
pub use gpu::ShGpu;
pub use mesh::{Center, Mesh};
pub use smooth::{SmoothOptions, SmoothOutput, smooth};
