//! Tier 1 RTFP. CPU f64 is the authority. Backend choice is data, not a trait.

use crate::common::density::CauchyDensity;
use crate::common::geometry::{FlattenedBvh, TriMesh};
use crate::common::math::DVec3;
use crate::rtfp_solver::backend::cpu_rayon::CpuRayonBackend;
use crate::rtfp_solver::radial::gauss_legendre::{directions_for_budget, gauss_legendre};

pub mod api;
pub mod backend;
pub mod density;
#[cfg(feature = "diagnostics")]
pub mod diagnostics;
pub mod error;
pub mod field;
pub mod geometry;
pub mod radial;
pub mod ray_radii;
pub mod volume;

pub use api::{FieldBatch, FieldMask, FieldRequest, PointRole, PointSet, VolumeIntegrals};
pub use error::RtfpError;

#[derive(Clone, Debug)]
pub struct RtfpOptions {
    pub directions: usize,
    pub r0: f64,
    pub ell0: f64,
    pub g: f64,
    pub quadrature_nodes: usize,
}

impl Default for RtfpOptions {
    fn default() -> Self {
        Self {
            directions: 2048,
            r0: 0.05,
            ell0: 1.0,
            g: 6.674_30e-11,
            quadrature_nodes: 32,
        }
    }
}

pub fn bind_cpu(
    options: &RtfpOptions,
    mesh: TriMesh,
    density: CauchyDensity,
) -> Result<CpuRayonBackend, RtfpError> {
    if options.directions < 8 {
        return Err(RtfpError::model_not_closed(
            "RTFP direction budget must be at least 8",
        ));
    }
    if !(options.r0.is_finite() && options.r0 > 0.0) {
        return Err(RtfpError::model_not_closed(
            "near-field radius r0 must be a finite positive length",
        ));
    }
    if !(options.ell0.is_finite() && options.ell0 > 0.0) {
        return Err(RtfpError::model_not_closed(
            "finite-part scale ell0 must be a finite positive length",
        ));
    }
    if !(options.g.is_finite() && options.g > 0.0) {
        return Err(RtfpError::model_not_closed(
            "gravitational constant must be finite and positive",
        ));
    }
    let bvh = FlattenedBvh::build(&mesh);
    let dirs = directions_for_budget(options.directions)
        .into_iter()
        .map(|(u, w)| (DVec3::from_array(u), w))
        .collect();
    let (gl_nodes, gl_weights) = gauss_legendre(options.quadrature_nodes.max(8));
    Ok(CpuRayonBackend {
        mesh,
        bvh,
        density,
        dirs,
        gl_nodes,
        gl_weights,
        g: options.g,
        r0: options.r0,
        ell0: options.ell0,
    })
}

pub fn bind_and_eval(
    options: RtfpOptions,
    mesh: &TriMesh,
    density: CauchyDensity,
    points: &PointSet,
    role: PointRole,
    want: FieldMask,
) -> Result<FieldBatch, RtfpError> {
    let backend = bind_cpu(&options, mesh.clone(), density)?;
    backend.eval_points(points, role, want)
}
