//! The RTFP solver: ray-traced finite-part gravity of a polyhedron with a
//! generalized (anisotropic) Cauchy density (MS §1, §3).
//!
//! The field is split by the density difference of §3.9,
//! `ρ(y) = ρ(x) + [ρ(y) − ρ(x)]`. The constant part is the closed-form field of the
//! uniform polyhedron (`rtfp/shader/uniform.wesl`), which carries the contact term,
//! the face jump and every edge logarithm exactly. The difference vanishes at the
//! field point and is integrated along the rays of one Lebedev–Laikov rule
//! (`rtfp/shader/field.wesl`); a constant density never reaches the rays, so the
//! angular rule only sees the variation of the cores.
//!
//! [`crate::rtfp::reference`] holds the independent Werner benchmark of the tests.

use std::f64::consts::PI;
use std::num::NonZeroUsize;

use gauss_quad::GaussLegendre;

use crate::common::density::CauchyDensity;
use crate::common::error::SimError;
use crate::common::geometry::TriMesh;
use crate::common::math::{DVec3, Sym6};
use crate::gpu::{Job, Res, kernels};
use crate::rtfp::api::{FieldBatch, FieldMask, PointRole, PointSet};
use crate::rtfp::chords::{MERGE_ABS, MERGE_REL, ORIGIN_ABS};
use crate::rtfp::lebedev;
use crate::rtfp::mesh::{BodyMesh, LAUNCH_OFFSET};

#[cfg_attr(feature = "tier3", derive(bevy::prelude::Resource))]
#[derive(Clone, Debug)]
pub struct RtfpOptions {
    /// Gravitational constant.
    pub g: f64,
    /// Gauss–Legendre nodes per part of the composite radial rule (§3.3): 8, 16 or 32.
    pub radial_nodes: usize,
    /// Half-width of the tangent-substitution window in units of `d` (§3.3, κ).
    pub kappa: f64,
    /// Opening angle of the face tree of the uniform part (`rtfp/tree.rs`).
    pub theta: f64,
}

impl Default for RtfpOptions {
    fn default() -> Self {
        Self {
            g: 6.674_30e-11,
            radial_nodes: 16,
            kappa: 4.0,
            theta: super::tree::THETA,
        }
    }
}

/// Field points per dispatch: one thread each, below the one-dimension group limit.
const POINTS_PER_LAUNCH: usize = 64 * 65_535;
/// Floats per point in the field output: ten double-single pairs and a flag.
const FIELD_STRIDE: usize = 21;

/// `Kern` of `rtfp/shader/chord.wesl`, in the unit frame.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct GpuKernel {
    c: [f32; 3],
    alpha: f32,
    sd: [f32; 3],
    w: f32,
    so: [f32; 3],
    pad: f32,
}

/// `Chords` of `rtfp/shader/chord.wesl`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct ChordParams {
    pub kernel_count: u32,
    pub node_count: u32,
    pub ray_count: u32,
    pub item_count: u32,
    pub want_u: u32,
    pub want_h: u32,
    pub bulk: f32,
    pub kappa: f32,
    pub merge_abs: f32,
    pub merge_rel: f32,
    pub origin_abs: f32,
    pub launch: f32,
    /// Read intervals from the segment table 𝒯 instead of tracing (MS §3.15.1).
    pub use_segs: u32,
    /// First point index covered by `seg_counts`/`seg_pairs`.
    pub seg_base: u32,
    /// First float of the launch's output block inside `out`.
    pub out_base: u32,
    pub pad1: f32,
}

/// Capacity-reuse device buffers for `eval_group`: points in, both kernel
/// outputs (uniform half, ray half) in one `out` buffer.
struct EvalScratch {
    points: Res,
    own: Res,
    out: Res,
    cap: usize,
}

pub struct RtfpSolver {
    pub(crate) options: RtfpOptions,
    pub(crate) mesh: TriMesh,
    pub(crate) density: CauchyDensity,
    pub(crate) body: BodyMesh,
    /// Lebedev directions and weights, `[dx, dy, dz, ω]`.
    pub(crate) rays: Res,
    /// Gauss–Legendre nodes `[x, w]` on [−1, 1].
    pub(crate) nodes: Res,
    pub(crate) node_count: u32,
    /// Segment tables 𝒯 of the point sets seen so far (MS §3.15.1): a set is
    /// traced on its first evaluation, and its table is built at the second
    /// and reused from then on. Density updates (`set_density`) leave the
    /// tables valid (Proposition 3.15.1).
    segs: std::sync::Mutex<std::collections::HashMap<u64, SegState>>,
    /// Device buffers reused across `eval` calls (`EvalScratch`).
    scratch: std::sync::Mutex<Option<EvalScratch>>,
}

/// Double-single pair to f64.
fn ds(hi: f32, lo: f32) -> f64 {
    f64::from(hi) + f64::from(lo)
}

/// A point set's row in the segment-table cache: `Seen` after the first
/// evaluation, `Ready` once `rtfp_segments` has built its 𝒯.
enum SegState {
    Seen,
    Ready(std::sync::Arc<crate::rtfp::segments::SegmentTable>),
}

impl RtfpSolver {
    /// Bind a closed mesh and a density to the GPU ray tracer.
    pub fn bind(
        options: RtfpOptions,
        mesh: TriMesh,
        density: CauchyDensity,
    ) -> Result<Self, SimError> {
        if !(options.g.is_finite() && options.g > 0.0) {
            return Err(SimError::model_not_closed(
                "gravitational constant must be finite and positive",
            ));
        }
        if !(options.theta > 0.0 && options.theta < 1.0) {
            return Err(SimError::model_not_closed(
                "tree opening angle θ must lie in (0, 1)",
            ));
        }
        if !(options.kappa.is_finite() && options.kappa > 0.0) {
            return Err(SimError::model_not_closed("κ must be finite and positive"));
        }
        mesh.validate_solid()?;
        let body = BodyMesh::build(&mesh)?;
        let sphere = lebedev::rule();
        let rays: Vec<[f32; 4]> = sphere
            .dirs
            .iter()
            .zip(&sphere.weights)
            .map(|(d, w)| [d.x as f32, d.y as f32, d.z as f32, *w as f32])
            .collect();
        let nodes = legendre_nodes(options.radial_nodes);
        let gpu = |e: String| SimError::gpu(e);
        Ok(Self {
            rays: Res::from_slice(&rays).map_err(gpu)?,
            node_count: nodes.len() as u32,
            nodes: Res::from_slice(&nodes).map_err(gpu)?,
            options,
            mesh,
            density,
            body,
            segs: std::sync::Mutex::new(std::collections::HashMap::new()),
            scratch: std::sync::Mutex::new(None),
        })
    }

    pub fn options(&self) -> &RtfpOptions {
        &self.options
    }

    pub fn mesh(&self) -> &TriMesh {
        &self.mesh
    }

    pub fn density(&self) -> &CauchyDensity {
        &self.density
    }

    /// Replace the density on the same mesh; only the core table is re-uploaded.
    pub fn set_density(&mut self, density: CauchyDensity) {
        self.density = density;
    }

    /// Rows of the cached segment tables 𝒯, their dirty rays and their
    /// overflow rays (MS §3.15.1, §3.6 step 4), for diagnostics.
    pub fn seg_stats(&self) -> (usize, u64, u64) {
        let map = self.segs.lock().unwrap();
        map.values().fold((0, 0, 0), |(sets, d, o), s| match s {
            SegState::Seen => (sets, d, o),
            SegState::Ready(t) => (
                sets + t.rows,
                d + u64::from(t.dirty),
                o + u64::from(t.overflow),
            ),
        })
    }

    /// The cores in the unit frame: `c_u = (c − centre)/scale`, `S_u = scale² S`.
    pub(crate) fn unit_kernels(&self) -> Vec<GpuKernel> {
        let f = self.body.frame;
        let s2 = f.scale * f.scale;
        self.density
            .kernels
            .iter()
            .map(|k| {
                let c = f.to_unit(k.center);
                let m = k.metric.map(|v| (v * s2) as f32);
                GpuKernel {
                    c: [c.x as f32, c.y as f32, c.z as f32],
                    alpha: k.alpha as f32,
                    sd: [m[0], m[1], m[2]],
                    w: k.weight as f32,
                    so: [m[3], m[4], m[5]],
                    pad: 0.0,
                }
            })
            .collect()
    }

    pub(crate) fn chord_params(
        &self,
        item_count: u32,
        want_u: bool,
        want_h: bool,
        use_segs: bool,
        seg_base: u32,
        out_base: u32,
    ) -> ChordParams {
        ChordParams {
            kernel_count: self.density.kernels.len() as u32,
            node_count: self.node_count,
            ray_count: lebedev::COUNT as u32,
            item_count,
            want_u: u32::from(want_u),
            want_h: u32::from(want_h),
            bulk: self.density.bulk as f32,
            kappa: self.options.kappa as f32,
            merge_abs: MERGE_ABS as f32,
            merge_rel: MERGE_REL as f32,
            origin_abs: ORIGIN_ABS as f32,
            launch: LAUNCH_OFFSET,
            use_segs: u32::from(use_segs),
            seg_base,
            out_base,
            pad1: 0.0,
        }
    }

    /// Potential, gradient and Hessian at `points`.
    pub fn eval(
        &self,
        points: &PointSet,
        role: PointRole,
        want: FieldMask,
    ) -> Result<FieldBatch, SimError> {
        if role == PointRole::SurfaceVertex && want.hessian {
            return Err(SimError::model_not_closed(
                "MS §10.2: a vertex or edge Hessian must not be sampled; use SurfaceFace",
            ));
        }
        if !(want.potential || want.gradient || want.hessian) {
            return Err(SimError::model_not_closed(
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
            anomalies: 0,
        };
        let table = self.seg_table(points)?;
        if let Some(t) = &table {
            batch.anomalies += t.dirty;
        }
        let mut start = 0;
        while start < n {
            let end = (start + POINTS_PER_LAUNCH).min(n);
            let xs: Vec<DVec3> = (start..end).map(|i| points.point(i)).collect();
            if xs.iter().any(|x| !x.is_finite()) {
                return Err(SimError::geometry("evaluation point must be finite"));
            }
            self.eval_group(&xs, role, want, &mut batch, start, table.as_deref())?;
            start = end;
        }
        Ok(batch)
    }

    /// The segment table of a point set (MS §3.15.1): `None` on its first
    /// evaluation and whenever it cannot be cached, the table from the second
    /// evaluation on. Density changes leave every table valid.
    fn seg_table(
        &self,
        points: &PointSet,
    ) -> Result<Option<std::sync::Arc<crate::rtfp::segments::SegmentTable>>, SimError> {
        use crate::rtfp::segments::{SegmentTable, point_key};
        if self.density.kernels.is_empty() || !SegmentTable::fits(points.len()) {
            return Ok(None);
        }
        let key = point_key(points);
        let mut map = self.segs.lock().unwrap();
        match map.get(&key) {
            Some(SegState::Ready(t)) => Ok(Some(t.clone())),
            Some(SegState::Seen) => {
                let t = std::sync::Arc::new(SegmentTable::build(self, points)?);
                map.insert(key, SegState::Ready(t.clone()));
                Ok(Some(t))
            }
            None => {
                map.insert(key, SegState::Seen);
                Ok(None)
            }
        }
    }

    fn eval_group(
        &self,
        xs: &[DVec3],
        role: PointRole,
        want: FieldMask,
        out: &mut FieldBatch,
        offset: usize,
        table: Option<&crate::rtfp::segments::SegmentTable>,
    ) -> Result<(), SimError> {
        let frame = self.body.frame;
        let n = xs.len();
        let rho: Vec<f64> = xs.iter().map(|x| self.density.rho(*x)).collect();
        let pts: Vec<[f32; 4]> = xs
            .iter()
            .zip(&rho)
            .map(|(x, r)| {
                let u = frame.to_unit(*x);
                [u.x as f32, u.y as f32, u.z as f32, *r as f32]
            })
            .collect();
        let own: Vec<u32> = match role {
            PointRole::SurfaceFace => xs
                .iter()
                .map(|x| {
                    self.body.face_of(&self.mesh, *x).ok_or_else(|| {
                        SimError::geometry("a SurfaceFace point lies on no face interior")
                    })
                })
                .collect::<Result<_, _>>()?,
            _ => vec![u32::MAX; n],
        };
        // Capacity-reuse scratch: one resident buffer per role, grown to the
        // largest batch seen. A call is then two writes, the two dispatches
        // sharing one output buffer, and a single read-back — the Metal
        // round-trip floor dominates everything else.
        let mut guard = self.scratch.lock().unwrap();
        if guard.as_ref().is_none_or(|s: &EvalScratch| s.cap < n) {
            let cap = n.next_power_of_two();
            *guard = Some(EvalScratch {
                points: Res::zeroed((cap * 16) as u64).map_err(SimError::gpu)?,
                own: Res::zeroed((cap * 4) as u64).map_err(SimError::gpu)?,
                out: Res::zeroed((2 * cap * FIELD_STRIDE * 4) as u64).map_err(SimError::gpu)?,
                cap,
            });
        }
        let s = guard.as_ref().expect("scratch just ensured");
        let mut job = Job::new();
        job.write(&s.points, 0, &pts);
        job.write(&s.own, 0, &own);
        // Exact part ρ(x) Φ_V (Werner–Scheeres), first half of `out`.
        let ku = &kernels::RTFP_UNIFORM;
        let uparams = job.value(&[
            n as u32,
            self.body.node_count as u32,
            (self.options.theta as f32).to_bits(),
            0,
        ]);
        job.run(
            ku,
            ku.groups(n),
            &[
                (&s.points).into(),
                (&s.own).into(),
                (&self.body.faces).into(),
                (&self.body.nodes).into(),
                uparams,
                (&s.out).into(),
            ],
        );
        // Ray part of ρ − ρ(x), second half of `out`; uniform density skips it.
        let has_rays = !self.density.kernels.is_empty();
        if has_rays {
            let k = &kernels::RTFP_FIELD;
            let cores = job.input(&self.unit_kernels());
            let params = job.value(&self.chord_params(
                n as u32,
                want.potential,
                want.hessian,
                table.is_some(),
                offset as u32,
                (n * FIELD_STRIDE) as u32,
            ));
            job.run(
                k,
                k.groups(n),
                &[
                    (&s.points).into(),
                    (&self.rays).into(),
                    cores,
                    (&self.nodes).into(),
                    params,
                    (&s.out).into(),
                    table.map_or((&self.nodes).into(), |t| (&t.counts).into()),
                    table.map_or((&self.nodes).into(), |t| (&t.pairs).into()),
                    (&self.body.accel).into(),
                ],
            );
        }
        let rd = job.read_all::<f32>(
            &s.out,
            if has_rays { 2 } else { 1 } * n * FIELD_STRIDE,
        );
        let outs = job.submit().map_err(SimError::gpu)?;
        let halves = outs.get::<f32>(rd);
        let (uni, ray) = if has_rays {
            let (u, r) = halves.split_at(n * FIELD_STRIDE);
            (u.to_vec(), Some(r.to_vec()))
        } else {
            (halves, None)
        };
        // Unit frame → world: U scales with L², ∇U with L, the Hessian not at all (§3.1).
        let g = self.options.g;
        let (s1, s2) = (frame.scale, frame.scale * frame.scale);
        for p in 0..n {
            let col_u = &uni[p * FIELD_STRIDE..(p + 1) * FIELD_STRIDE];
            let mut v = [0.0f64; 10];
            for (m, vm) in v.iter_mut().enumerate() {
                *vm = rho[p] * ds(col_u[2 * m], col_u[2 * m + 1]);
            }
            if let Some(ray) = &ray {
                let col = &ray[p * FIELD_STRIDE..(p + 1) * FIELD_STRIDE];
                let flags = col[20] as u32;
                if flags & 1 != 0 {
                    return Err(SimError::geometry(format!(
                        "a ray from point {} crosses the surface more than 64 times",
                        offset + p
                    )));
                }
                out.anomalies += flags >> 1;
                for (m, vm) in v.iter_mut().enumerate() {
                    *vm += ds(col[2 * m], col[2 * m + 1]);
                }
            }
            // The total solid angle is 4π inside and 0 outside.
            let inside = f64::from(col_u[20]) > 2.0 * PI;
            let (theta, solid) = match role {
                PointRole::SurfaceFace => (0.5, 2.0 * PI),
                PointRole::SurfaceVertex => (0.0, 0.0),
                PointRole::Field => {
                    let f = f64::from(u8::from(inside));
                    (f, f * 4.0 * PI)
                }
            };
            let i = offset + p;
            out.theta[i] = theta;
            out.solid_angle[i] = solid;
            out.potential[i] = g * s2 * v[0];
            out.gradient[i] = DVec3::new(v[1], v[2], v[3]) * (g * s1);
            if want.hessian {
                let h: Sym6 = [v[4], v[5], v[6], v[7], v[8], v[9]].map(|x| g * x);
                out.hessian[i] = h;
            }
        }
        Ok(())
    }
}

fn legendre_nodes(count: usize) -> Vec<[f32; 2]> {
    let rule = GaussLegendre::new(NonZeroUsize::new(count.max(4)).expect("at least four nodes"));
    rule.as_node_weight_pairs()
        .iter()
        .map(|&(x, w)| [x as f32, w as f32])
        .collect()
}
