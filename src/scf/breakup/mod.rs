//! Breakup (§12.1–12.2): the collapse mechanism of a certified limit load cut into rigid
//! blocks, and each block made a body of its own, with the parent's density and mass.

mod mesh;
mod partition;

use serde::Deserialize;

use crate::common::density::CauchyKernel;
use crate::common::error::SimError;
use crate::common::math::{DMat3, DVec3, mat_sym, sym_mat};

use super::body::Body;
use super::forward::Forward;
use super::frame::realign;
use super::params::Project;
use super::stokes::Operators;
use super::surface::{PointGrid, SurfaceIndex};

pub use mesh::{EMPTY, Voxels};
pub use partition::{PartitionOptions, Rigid, graph, partition};

/// The `[breakup]` section of a project: the cut (§12.1–12.2), the fragments' dynamics
/// (§12.3–12.4) and the debris ring (§12.5).
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BreakupParams {
    /// Most blocks a mechanism is cut into, the fraction by which each further block must
    /// lower the misfit, and the smallest block as a fraction of the mass.
    pub max_blocks: usize,
    pub gain: f64,
    pub min_fraction: f64,
    /// Voxels along the longest side of a body that is cut or merged.
    pub voxels: usize,
    /// Pieces smaller than this diameter (m) join the debris ring.
    pub debris_diameter: f64,
    /// `J₂` of the primary.
    pub primary_j2: f64,
    /// Length of the dynamic phase (s), the substeps per orbit at least, the interval of the
    /// stability checks (orbits), and of the recorded frames (s).
    pub dynamic_time: f64,
    pub steps_per_orbit: usize,
    pub check_every: f64,
    pub output_every: f64,
    /// Radial cells of the ring, its particle radius (m), and the time it evolves (s).
    pub ring_cells: usize,
    pub ring_particle_radius: f64,
    pub ring_time: f64,
}

impl BreakupParams {
    pub fn validate(&self) -> Result<(), SimError> {
        let positive = [
            ("gain", self.gain),
            ("min_fraction", self.min_fraction),
            ("debris_diameter", self.debris_diameter),
            ("dynamic_time", self.dynamic_time),
            ("check_every", self.check_every),
            ("output_every", self.output_every),
            ("ring_particle_radius", self.ring_particle_radius),
        ];
        for (name, v) in positive {
            if !(v.is_finite() && v > 0.0) {
                return Err(SimError::model_not_closed(format!("breakup.{name} must be finite and positive")));
            }
        }
        if self.max_blocks < 2 || self.voxels < 8 || self.steps_per_orbit < 50 || self.ring_cells < 8 {
            return Err(SimError::model_not_closed(
                "breakup needs max_blocks ≥ 2, voxels ≥ 8, steps_per_orbit ≥ 50 and ring_cells ≥ 8",
            ));
        }
        if !(self.gain < 1.0 && self.min_fraction < 0.5 && self.primary_j2.is_finite() && self.ring_time >= 0.0) {
            return Err(SimError::model_not_closed("breakup: gain < 1, min_fraction < 0.5, a finite J₂ and ring_time ≥ 0"));
        }
        Ok(())
    }
}

/// One block of a cut, as a body in its own principal frame.
pub struct Piece {
    pub body: Body,
    pub fwd: Forward,
    /// Its centroid in the parent's frame, and the rotation from its frame to the parent's.
    pub com: DVec3,
    pub rot: DMat3,
    /// The block's rigid motion in the mechanism (the direction it separates in).
    pub motion: Rigid,
}

/// The pieces of a cut, the misfit of each accepted number of blocks, and the relative
/// error of the pieces' total mass.
pub struct Split {
    pub pieces: Vec<Piece>,
    pub misfit: Vec<f64>,
    pub mass_error: f64,
}

/// Cuts `body` (bound to `fwd`, with the operators `ops` of its geometry) along the
/// collapse mechanism `u`; `None` when the mechanism is one rigid block.
pub fn split(project: &Project, prm: &BreakupParams, body: &Body, fwd: &Forward, ops: &Operators, u: &[f64]) -> Result<Option<Split>, SimError> {
    let _span = super::timing::span("breakup_split");
    let x = &ops.vol.pts;
    let (vel, _) = ops.vol_velocity(u);
    let v: Vec<DVec3> = vel.chunks_exact(3).map(|c| DVec3::new(c[0], c[1], c[2])).collect();
    let m: Vec<f64> = ops.vol.w.iter().zip(&ops.vol.rho).map(|(w, r)| w * r).collect();
    let nb = graph(x, 10);
    let opt = PartitionOptions { max_blocks: prm.max_blocks, gain: prm.gain, min_fraction: prm.min_fraction };
    let (labels, misfit) = partition(x, &v, &m, &nb, &opt);
    let k = labels.iter().max().map_or(0, |l| l + 1);
    if k < 2 {
        return Ok(None);
    }
    let index = SurfaceIndex::build(&body.positions, &body.topo.faces);
    let nodes = PointGrid::build(x);
    let (lo, hi) = bounds(&body.positions);
    let mut vox = Voxels::new(lo, hi, prm.voxels, |p| if index.inside(p) { labels[nodes.nearest(p)] as u32 } else { EMPTY });
    vox.connect();
    let parent_level = body.topo.icosphere.unwrap_or(project.body.level);
    let parent_area: f64 = body.face_geometry().iter().map(|f| f.2).sum();
    let density = body.density()?;
    // Block masses from the parent density over the voxels, scaled to the parent's mass.
    let cell = vox.h.powi(3);
    let mut target = vec![0.0; k];
    for (id, l) in vox.label.iter().enumerate() {
        if *l != EMPTY {
            target[*l as usize] += density.rho(vox.centre(id)) * cell;
        }
    }
    let scale = fwd.moments.mass / target.iter().sum::<f64>();
    let mut pieces = Vec::with_capacity(k);
    for b in 0..k {
        let idx: Vec<usize> = (0..x.len()).filter(|&i| labels[i] == b).collect();
        let motion = Rigid::fit(x, &v, &m, &idx).ok_or_else(|| SimError::solver("breakup: a block has no rigid motion"))?;
        let kernels: Vec<CauchyKernel> = body.kernels.iter().filter(|kn| labels[nodes.nearest(kn.center)] == b).cloned().collect();
        let solid = |l: u32| l == b as u32;
        let inner = vox.deepest(solid).ok_or_else(|| SimError::geometry("breakup: an empty block"))?;
        let piece = piece(project, &vox.surface(solid), inner, kernels, body.bulk, target[b] * scale, parent_level, parent_area)?;
        pieces.push(Piece { motion, ..piece });
    }
    let total: f64 = pieces.iter().map(|p| p.fwd.moments.mass).sum();
    Ok(Some(Split { pieces, misfit, mass_error: (total - fwd.moments.mass).abs() / fwd.moments.mass }))
}

/// A body of the closed surface `(verts, faces)` with `kernels` (all in one frame), its bulk
/// density set so that its mass is `mass`, moved to its principal frame. It is re-triangulated
/// by the rays from `inner`, a point deep inside it; the icosphere level keeps the parent's
/// edge length (`parent_level` on `parent_area`).
#[allow(clippy::too_many_arguments)]
pub fn piece(
    project: &Project,
    (verts, faces): &(Vec<DVec3>, Vec<[u32; 3]>),
    inner: DVec3,
    kernels: Vec<CauchyKernel>,
    bulk: f64,
    mass: f64,
    parent_level: usize,
    parent_area: f64,
) -> Result<Piece, SimError> {
    let area: f64 = faces
        .iter()
        .map(|t| {
            let [a, b, c] = t.map(|i| verts[i as usize]);
            0.5 * (b - a).cross(c - a).length()
        })
        .sum();
    let level = (parent_level as f64 + 0.5 * (area / parent_area).log2()).round().clamp(2.0, parent_level as f64) as usize;
    let (topo, positions, origin) = Body::rays_from(level, verts, faces, inner)?;
    let kernels = kernels
        .into_iter()
        .map(|kn| CauchyKernel::try_anisotropic(kn.center - origin, kn.weight, kn.metric, kn.alpha))
        .collect::<Result<Vec<_>, _>>()?;
    let rho_s = vec![project.material.rho_bulk * project.material.spec.h_s; topo.n_vertices()];
    let body = Body::new(topo.clone(), positions.clone(), bulk, kernels.clone(), rho_s.clone())?;
    let fwd = Forward::bind(&body, project.g)?;
    // The bulk density takes up the mass the cut moved between blocks.
    let bulk = bulk + (mass - fwd.moments.mass) / fwd.moments.volume;
    if !(bulk > 0.0) {
        return Err(SimError::model_not_closed(format!("breakup: a piece would need the bulk density {bulk:.3e}")));
    }
    let body = Body::new(topo, positions, bulk, kernels, rho_s)?;
    let (body, rep, fwd) = realign(&body, None, project.g, &project.geometry, true)?;
    Ok(Piece {
        body,
        fwd,
        com: origin + rep.translation,
        rot: DMat3::from_cols_array_2d(&rep.rotation),
        motion: Rigid { a: DVec3::ZERO, theta: DVec3::ZERO, c: DVec3::ZERO },
    })
}

/// `kernel` moved rigidly: `x ↦ rot x + shift`.
pub fn moved_kernel(kernel: &CauchyKernel, rot: DMat3, shift: DVec3) -> Result<CauchyKernel, SimError> {
    let metric = mat_sym(rot * sym_mat(kernel.metric) * rot.transpose());
    CauchyKernel::try_anisotropic(rot * kernel.center + shift, kernel.weight, metric, kernel.alpha)
}

pub fn bounds(pts: &[DVec3]) -> (DVec3, DVec3) {
    pts.iter().fold((DVec3::splat(f64::INFINITY), DVec3::splat(f64::NEG_INFINITY)), |(lo, hi), p| (lo.min(*p), hi.max(*p)))
}
