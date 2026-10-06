//! The RTFP solver: ray-traced finite-part gravity of a polyhedron with a
//! generalized-Cauchy density. The angular integral is one fixed Lebedev–Laikov
//! rule. Ray intersection, chord integrals and the field reduction run on the GPU.

pub mod chords;
pub mod field;
pub mod hw_rt;
pub mod lebedev;
pub mod moments;

use std::f64::consts::PI;
use std::num::NonZeroUsize;

use gauss_quad::GaussLegendre;

use crate::common::density::CauchyDensity;
use crate::common::geometry::TriMesh;
use crate::common::math::DVec3;
use crate::rtfp_solver::api::{FieldBatch, FieldMask, PointRole, PointSet};
use crate::rtfp_solver::error::RtfpError;

use field::Medium;
use hw_rt::{DeviceKernel, FieldJob, HardwareMesh};

#[cfg_attr(feature = "tier3", derive(bevy::prelude::Resource))]
#[derive(Clone, Debug)]
pub struct RtfpOptions {
    /// Gravitational constant.
    pub g: f64,
    /// Kept for the field-record file. The angular integral does not use it.
    pub angular_ratio: f64,
    /// Gauss–Legendre nodes per radial panel for kernels with `α ≠ 1`.
    pub radial_nodes: usize,
}

impl Default for RtfpOptions {
    fn default() -> Self {
        Self {
            g: 6.674_30e-11,
            angular_ratio: 0.06,
            radial_nodes: 12,
        }
    }
}

/// Rays kept in one GPU launch.
const GROUP_RAYS: usize = 1_500_000;
const MAX_GROUP: usize = 256;

pub struct RtfpSolver {
    options: RtfpOptions,
    mesh: TriMesh,
    density: CauchyDensity,
    medium: Medium,
    /// Gauss–Legendre nodes for kernels whose `α` is not 1, and for mass moments.
    quad: Vec<[f32; 2]>,
    hw: HardwareMesh,
}

impl RtfpSolver {
    /// Bind a closed mesh and a density to the hardware ray tracer.
    pub fn bind(
        options: RtfpOptions,
        mesh: TriMesh,
        density: CauchyDensity,
    ) -> Result<Self, RtfpError> {
        if !(options.g.is_finite() && options.g > 0.0) {
            return Err(RtfpError::model_not_closed(
                "gravitational constant must be finite and positive",
            ));
        }
        if !(options.angular_ratio.is_finite() && options.angular_ratio > 0.0) {
            return Err(RtfpError::model_not_closed(
                "angular ratio must be finite and positive",
            ));
        }
        mesh.validate_solid()?;
        let hw = HardwareMesh::build(&mesh)?;
        Ok(Self {
            medium: Medium::new(&density),
            quad: legendre_nodes(options.radial_nodes.max(16)),
            options,
            mesh,
            density,
            hw,
        })
    }

    pub fn mesh(&self) -> &TriMesh {
        &self.mesh
    }

    pub fn density(&self) -> &CauchyDensity {
        &self.density
    }

    pub fn hardware(&self) -> &HardwareMesh {
        &self.hw
    }

    pub fn options(&self) -> &RtfpOptions {
        &self.options
    }

    /// Potential, gradient and Hessian at `points`.
    pub fn eval(
        &self,
        points: &PointSet,
        role: PointRole,
        want: FieldMask,
    ) -> Result<FieldBatch, RtfpError> {
        if role == PointRole::SurfaceVertex && want.hessian {
            return Err(RtfpError::model_not_closed(
                "MS §10.3: a vertex or edge Hessian must not be sampled; use SurfaceFace",
            ));
        }
        if !(want.potential || want.gradient || want.hessian) {
            return Err(RtfpError::model_not_closed(
                "field request must include potential, gradient, or hessian",
            ));
        }
        let n = points.len();
        let mut batch = FieldBatch {
            potential: vec![0.0; n],
            gradient: vec![DVec3::ZERO; n],
            hessian: vec![[0.0; 6]; n],
            theta: vec![0.0; n],
            solid_angle: vec![0.0; n],
        };
        let group = (GROUP_RAYS / lebedev::COUNT).clamp(1, MAX_GROUP);
        let mut start = 0;
        while start < n {
            let end = (start + group).min(n);
            let xs: Vec<DVec3> = (start..end).map(|i| points.point(i)).collect();
            if xs.iter().any(|x| !x.is_finite()) {
                return Err(RtfpError::geometry("evaluation point must be finite"));
            }
            self.eval_group(&xs, role, want, &mut batch, start)?;
            start = end;
        }
        Ok(batch)
    }

    fn eval_group(
        &self,
        xs: &[DVec3],
        role: PointRole,
        want: FieldMask,
        out: &mut FieldBatch,
        offset: usize,
    ) -> Result<(), RtfpError> {
        let sphere = lebedev::rule();
        let inside = match role {
            PointRole::Field => self.hw.inside(xs)?,
            _ => Vec::new(),
        };
        let mut thetas = Vec::with_capacity(xs.len());
        let mut rhos = Vec::with_capacity(xs.len());
        for p in 0..xs.len() {
            let (theta, solid) = match role {
                PointRole::SurfaceFace => (0.5, 2.0 * PI),
                PointRole::SurfaceVertex => (0.0, 0.0),
                PointRole::Field => {
                    let inside = u8::from(inside[p]);
                    (f64::from(inside), f64::from(inside) * 4.0 * PI)
                }
            };
            thetas.push(theta);
            rhos.push(self.density.rho(xs[p]));
            let i = offset + p;
            out.theta[i] = theta;
            out.solid_angle[i] = solid;
        }
        let count = sphere.dirs.len() as u32;
        let ranges = vec![[0u32, count]; xs.len()];
        let kernels: Vec<DeviceKernel> = self
            .medium
            .kernels()
            .iter()
            .map(|k| DeviceKernel {
                center: k.center,
                inv_sigma2: k.inv_sigma2,
                coeff: k.coeff,
                alpha: k.alpha,
            })
            .collect();
        let columns = self.hw.eval_field(&FieldJob {
            points: xs,
            thetas: &thetas,
            rhos: &rhos,
            ranges: &ranges,
            dirs: &sphere.dirs,
            weights: &sphere.weights,
            bulk: self.medium.bulk(),
            kernels: &kernels,
            nodes: &self.quad,
            g: self.options.g,
            want_potential: want.potential,
            want_hessian: want.hessian,
        })?;
        for (p, col) in columns.iter().enumerate() {
            let i = offset + p;
            out.potential[i] = col[0];
            out.gradient[i] = DVec3::new(col[1], col[2], col[3]);
            if want.hessian {
                out.hessian[i] = [col[4], col[5], col[6], col[7], col[8], col[9]];
            }
        }
        Ok(())
    }
}

fn legendre_nodes(count: usize) -> Vec<[f32; 2]> {
    let rule = GaussLegendre::new(NonZeroUsize::new(count.max(4)).expect("at least four nodes"));
    let mut nodes = Vec::new();
    for &(x, w) in rule.as_node_weight_pairs() {
        nodes.push([x as f32, w as f32]);
    }
    nodes
}
