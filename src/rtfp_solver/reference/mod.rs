//! Accuracy benchmarks for the RTFP solver. They share no code with it and
//! never run unless a comparison is requested.
//!
//! * [`werner`]: exact uniform polyhedron (Werner–Scheeres).
//! * [`feynman`]: Feynman-parameter evaluation of the generalized-Cauchy kernels.
//! * [`carlson`] and [`ellipsoid`]: Carlson elliptic integrals and the exact homogeneous ellipsoid.

pub mod carlson;
pub mod ellipsoid;
pub mod feynman;
pub mod werner;
