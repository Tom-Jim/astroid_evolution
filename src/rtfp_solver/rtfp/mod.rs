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

/// Complete field points in one command buffer. Every point still integrates the
/// full Lebedev rule; the split is only between points, so a partial launch is
/// the same sum. `output/out.obj` has 196608 faces and fits in one launch.
const POINTS_PER_LAUNCH: usize = 262_144;

pub struct RtfpSolver {
    options: RtfpOptions,
    mesh: TriMesh,
    density: CauchyDensity,
    medium: Medium,
    /// Base Gauss–Legendre rule, then the 32-node rule used on long chords.
    /// Mass moments read only the base prefix.
    quad: Vec<[f32; 2]>,
    quad_base: usize,
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
        let mut quad = legendre_nodes(options.radial_nodes.max(16));
        let quad_base = quad.len();
        quad.extend(legendre_nodes(32));
        Ok(Self {
            medium: Medium::new(&density),
            quad,
            quad_base,
            options,
            mesh,
            density,
            hw,
        })
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
        self.eval_mode(points, role, want, true)
    }

    /// The previous 256-lane reduction, for the packet comparison.
    #[cfg(all(test, target_os = "macos"))]
    fn eval_strided(
        &self,
        points: &PointSet,
        role: PointRole,
        want: FieldMask,
    ) -> Result<FieldBatch, RtfpError> {
        self.eval_mode(points, role, want, false)
    }

    fn eval_mode(
        &self,
        points: &PointSet,
        role: PointRole,
        want: FieldMask,
        packet: bool,
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
        let group = POINTS_PER_LAUNCH;
        let mut start = 0;
        while start < n {
            let end = (start + group).min(n);
            let xs: Vec<DVec3> = (start..end).map(|i| points.point(i)).collect();
            if xs.iter().any(|x| !x.is_finite()) {
                return Err(RtfpError::geometry("evaluation point must be finite"));
            }
            self.eval_group(&xs, role, want, &mut batch, start, packet)?;
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
        packet: bool,
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
        let job = FieldJob {
            points: xs,
            thetas: &thetas,
            rhos: &rhos,
            dirs: &sphere.dirs,
            weights: &sphere.weights,
            bulk: self.medium.bulk(),
            kernels: &kernels,
            nodes: &self.quad,
            node_base: self.quad_base as u32,
            g: self.options.g,
            want_potential: want.potential,
            want_hessian: want.hessian,
        };
        let columns = if packet {
            self.hw.eval_field(&job)?
        } else {
            #[cfg(all(test, target_os = "macos"))]
            {
                self.hw.eval_field_strided(&job)?
            }
            #[cfg(not(all(test, target_os = "macos")))]
            {
                let _ = packet;
                return Err(RtfpError::gpu_unavailable(
                    "the strided field kernel is a macOS test oracle",
                ));
            }
        };
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

#[cfg(all(test, target_os = "macos"))]
#[test]
fn packet_field_matches_the_strided_reduction() {
    let mesh = TriMesh::unit_cube();
    let tri = mesh.triangle(0);
    let centre = DVec3::new(
        (tri[0].x + tri[1].x + tri[2].x) / 3.0,
        (tri[0].y + tri[1].y + tri[2].y) / 3.0,
        (tri[0].z + tri[1].z + tri[2].z) / 3.0,
    );
    let points = PointSet::from_points(&[DVec3::new(0.2, -0.1, 0.3), centre]);
    let solver = RtfpSolver::bind(
        RtfpOptions::default(),
        mesh,
        CauchyDensity::uniform(1.0).expect("uniform density"),
    )
    .expect("hardware ray tracing");
    let want = FieldMask::ALL;
    let packet = solver
        .eval(&points, PointRole::Field, want)
        .expect("packet field");
    let interior = solver
        .eval_strided(&points, PointRole::Field, want)
        .expect("strided field");
    let face = solver
        .eval(
            &PointSet::from_points(&[centre]),
            PointRole::SurfaceFace,
            want,
        )
        .expect("packet face");
    let face_ref = solver
        .eval_strided(
            &PointSet::from_points(&[centre]),
            PointRole::SurfaceFace,
            want,
        )
        .expect("strided face");
    for (got, want_batch) in [(&packet, &interior), (&face, &face_ref)] {
        let n = got.gradient.len();
        for i in 0..n {
            let comps = [
                got.potential[i],
                got.gradient[i].x,
                got.gradient[i].y,
                got.gradient[i].z,
                got.hessian[i][0],
                got.hessian[i][1],
                got.hessian[i][2],
                got.hessian[i][3],
                got.hessian[i][4],
                got.hessian[i][5],
            ];
            let refs = [
                want_batch.potential[i],
                want_batch.gradient[i].x,
                want_batch.gradient[i].y,
                want_batch.gradient[i].z,
                want_batch.hessian[i][0],
                want_batch.hessian[i][1],
                want_batch.hessian[i][2],
                want_batch.hessian[i][3],
                want_batch.hessian[i][4],
                want_batch.hessian[i][5],
            ];
            for (a, b) in comps.iter().zip(refs.iter()) {
                let scale = b.abs().max(1e-12);
                let rel = (a - b).abs() / scale;
                assert!(
                    rel < 1e-6,
                    "packet {a} and strided {b} differ by relative {rel}"
                );
            }
        }
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
