//! Tier 1: ray-traced finite-part (RTFP) gravity of a polyhedron with a Cauchy density.

pub mod api;
pub mod chords;
pub mod lebedev;
pub(crate) mod lebedev_table;
pub mod mesh;
pub mod moments;
pub mod reference;
pub(crate) mod segments;
pub mod solver;
pub mod tree;
pub mod volume;

pub use api::{FieldBatch, FieldMask, PointRole, PointSet, VolumeIntegrals};
pub use mesh::BodyMesh;
pub use solver::{RtfpOptions, RtfpSolver};
