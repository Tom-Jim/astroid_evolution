//! The RTFP solver: ray-traced finite-part gravity of a polyhedron with a
//! generalized-Cauchy density. Every surface query runs on the hardware
//! ray-tracing units; there is no software ray tracer.

pub mod chords;
pub mod field;
pub mod hw_rt;
pub mod moments;
pub mod patches;
pub mod radial;

use std::f64::consts::PI;

use rayon::prelude::*;

use crate::common::density::CauchyDensity;
use crate::common::geometry::TriMesh;
use crate::common::math::{DVec3, Sym6};
use crate::rtfp_solver::api::{FieldBatch, FieldMask, PointRole, PointSet};
use crate::rtfp_solver::error::RtfpError;

use field::{Medium, PointKernels, RayContext, Sums};
use hw_rt::HardwareMesh;
use patches::Faces;

#[cfg_attr(feature = "tier3", derive(bevy::prelude::Resource))]
#[derive(Clone, Debug)]
pub struct RtfpOptions {
    /// Gravitational constant.
    pub g: f64,
    /// A surface patch is split while its edge exceeds this fraction of its distance from the point.
    pub angular_ratio: f64,
    /// Gauss–Legendre nodes per radial panel for kernels with `α ≠ 1`.
    pub radial_nodes: usize,
}

impl Default for RtfpOptions {
    fn default() -> Self {
        Self {
            g: 6.674_30e-11,
            angular_ratio: patches::DEFAULT_ANGULAR_RATIO,
            radial_nodes: 12,
        }
    }
}

/// Rays kept in flight per evaluation group.
const GROUP_RAYS: usize = 1_500_000;
const MAX_GROUP: usize = 256;
/// Rays handled by one rayon task during accumulation.
const TASK_RAYS: usize = 2048;

pub struct RtfpSolver {
    options: RtfpOptions,
    mesh: TriMesh,
    density: CauchyDensity,
    medium: Medium,
    general: radial::General,
    faces: Faces,
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
            return Err(RtfpError::model_not_closed("angular ratio must be finite and positive"));
        }
        mesh.validate_solid()?;
        let faces = Faces::new(&mesh)?;
        let hw = HardwareMesh::build(&mesh)?;
        Ok(Self {
            medium: Medium::new(&density),
            general: radial::General::new(options.radial_nodes),
            options,
            mesh,
            density,
            faces,
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
        let group = (GROUP_RAYS / (self.faces.len() + 2000)).clamp(1, MAX_GROUP);
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
        let leaves = patches::settle(&self.hw, &self.faces, xs, self.options.angular_ratio)?;

        // Phase 2: every accepted patch ray, with all of its crossings.
        let samples: Vec<Vec<patches::Sample>> = xs
            .iter()
            .zip(&leaves)
            .map(|(x, list)| patches::flatten(*x, list, &self.faces))
            .collect();
        let mut batch = self.hw.new_batch();
        let mut starts = Vec::with_capacity(xs.len());
        for (x, list) in xs.iter().zip(&samples) {
            starts.push(batch.len());
            let parts: Vec<_> = list
                .par_chunks(4096)
                .map(|piece| {
                    let mut part = self.hw.new_batch();
                    for s in piece {
                        part.push(*x, s.dir, None);
                    }
                    part
                })
                .collect();
            for mut part in parts {
                batch.append(&mut part);
            }
        }
        let hits = self.hw.trace(&batch, false)?;

        let inside = match role {
            PointRole::Field => self.hw.inside(xs)?,
            _ => Vec::new(),
        };

        let kernels: Vec<PointKernels> = xs.iter().map(|x| PointKernels::new(&self.medium, *x)).collect();
        let tasks: Vec<(usize, usize, usize)> = samples
            .iter()
            .enumerate()
            .flat_map(|(p, list)| {
                (0..list.len())
                    .step_by(TASK_RAYS)
                    .map(move |lo| (p, lo, (lo + TASK_RAYS).min(list.len())))
            })
            .collect();
        let partial: Vec<(usize, Sums)> = tasks
            .par_iter()
            .map(|&(p, lo, hi)| {
                let ctx = RayContext {
                    medium: &self.medium,
                    point: &kernels[p],
                    general: &self.general,
                    want,
                    ell: self.hw.scale(),
                    body_scale: self.hw.scale(),
                };
                let slice = &samples[p][lo..hi];
                let dirs: Vec<DVec3> = slice.iter().map(|s| s.dir).collect();
                let weights: Vec<f64> = slice.iter().map(|s| s.weight).collect();
                (p, field::accumulate(&ctx, &hits, starts[p] + lo, &dirs, &weights))
            })
            .collect();
        let mut sums = vec![Sums::ZERO; xs.len()];
        for (p, s) in &partial {
            sums[*p].merge(s);
        }

        let g = self.options.g;
        for (p, s) in sums.iter().enumerate() {
            let i = offset + p;
            let theta = match role {
                PointRole::SurfaceFace => 0.5,
                PointRole::SurfaceVertex => s.solid_angle / (4.0 * PI),
                PointRole::Field => f64::from(u8::from(inside[p])),
            };
            let contact = -4.0 * PI / 3.0 * g * theta * self.density.rho(xs[p]);
            let mut h: Sym6 = [0.0; 6];
            if want.hessian {
                for c in 0..6 {
                    h[c] = g * s.hessian[c];
                }
                for c in 0..3 {
                    h[c] += contact;
                }
            }
            out.potential[i] = g * s.potential;
            out.gradient[i] = s.gradient * g;
            out.hessian[i] = h;
            out.theta[i] = theta;
            out.solid_angle[i] = s.solid_angle;
        }
        Ok(())
    }
}
