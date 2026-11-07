//! Breakup (§12.1–12.2): the collapse mechanism of a certified limit load cut into rigid
//! blocks, and each block made a body of its own, with the parent's density and mass.

mod mesh;
mod partition;

use serde::Deserialize;
use std::collections::HashMap;

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
        let piece = piece(project, &vox.surface(solid), kernels, body.bulk, target[b] * scale, parent_level, parent_area)?;
        pieces.push(Piece { motion, ..piece });
    }
    let total: f64 = pieces.iter().map(|p| p.fwd.moments.mass).sum();
    Ok(Some(Split { pieces, misfit, mass_error: (total - fwd.moments.mass).abs() / fwd.moments.mass }))
}

/// A body of the closed surface `(verts, faces)` with `kernels` (all in one frame), its bulk
/// density set so that its mass is `mass`, moved to its principal frame; the edge length
/// follows the parent's (`parent_level` on `parent_area`).
#[allow(clippy::too_many_arguments)]
pub fn piece(
    project: &Project,
    (verts, faces): &(Vec<DVec3>, Vec<[u32; 3]>),
    kernels: Vec<CauchyKernel>,
    bulk: f64,
    mass: f64,
    parent_level: usize,
    parent_area: f64,
) -> Result<Piece, SimError> {
    // §12.2: keep the marching-cubes surface of the block and remesh it
    // isotropically to the parent's edge length — topology-preserving, so a
    // block's concavities and necks survive; the radial re-triangulation of
    // `rays_from` would impose a star-shaped shape on the piece.
    let mut rv = verts.clone();
    let mut rf = faces.clone();
    let target_edge = (parent_area / (20.0 * 4f64.powi(parent_level as i32))).sqrt() * 1.1;
    remesh(&mut rv, &mut rf, target_edge);
    let (topo, positions, origin) = Body::from_mesh(&rv, rf).or_else(|_| {
        // The remesh could not keep the geometric embedding (§12.2's outer
        // acceptance): the un-remeshed block surface is the topology-preserving
        // fallback — voxel-faceted but still the block's own surface, never a
        // radial re-projection.
        Body::from_mesh(verts, faces.clone())
    })?;
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

/// Topology-preserving isotropic remeshing of §12.2: edges longer than
/// `1.4·target` are split at their midpoint, edges shorter than `0.6·target`
/// collapsed when the collapse keeps a valid manifold, and vertices are then
/// relaxed tangentially (Laplacian smoothing projected onto the vertex normal
/// plane). Two rounds; the surface is re-validated by `TriMesh::try_new` on
/// the call site (`Body::from_mesh`), so a failed remesh can never ship.
fn remesh(verts: &mut Vec<DVec3>, faces: &mut Vec<[u32; 3]>, target: f64) {
    for _ in 0..2 {
        split_long(verts, faces, 1.4 * target);
        smooth(verts, faces, 0.4, 2);
        collapse_short(verts, faces, 0.6 * target);
        smooth(verts, faces, 0.4, 1);
    }
}

/// Undirected edge → the (up to two) incident faces.
fn edge_map(faces: &[[u32; 3]]) -> HashMap<(u32, u32), Vec<usize>> {
    let mut map: HashMap<(u32, u32), Vec<usize>> = HashMap::new();
    for (f, tri) in faces.iter().enumerate() {
        for e in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])] {
            map.entry(if e.0 < e.1 { (e.0, e.1) } else { (e.1, e.0) })
                .or_default()
                .push(f);
        }
    }
    map
}

fn split_long(verts: &mut Vec<DVec3>, faces: &mut Vec<[u32; 3]>, max_len: f64) {
    // Every edge longer than `max_len` is split in both of its faces at once,
    // so the subdivision stays a closed manifold.
    for _ in 0..4 {
        let mut midpoints: HashMap<(u32, u32), u32> = HashMap::new();
        let mut out = Vec::with_capacity(faces.len());
        let mut split_any = false;
        for tri in faces.iter() {
            let mut mid = [None; 3];
            for e in 0..3 {
                let (a, b) = (tri[e], tri[(e + 1) % 3]);
                if verts[a as usize].distance(verts[b as usize]) > max_len {
                    let key = if a < b { (a, b) } else { (b, a) };
                    mid[e] = Some(*midpoints.entry(key).or_insert_with(|| {
                        verts.push((verts[a as usize] + verts[b as usize]) * 0.5);
                        (verts.len() - 1) as u32
                    }));
                    split_any = true;
                }
            }
            let (v0, v1, v2) = (tri[0], tri[1], tri[2]);
            match mid {
                [None, None, None] => out.push(*tri),
                // One edge split in two.
                [Some(m), None, None] => {
                    out.push([v0, m, v2]);
                    out.push([m, v1, v2]);
                }
                [None, Some(m), None] => {
                    out.push([v0, v1, m]);
                    out.push([v0, m, v2]);
                }
                [None, None, Some(m)] => {
                    out.push([v0, v1, m]);
                    out.push([m, v1, v2]);
                }
                // Two edges: a corner triangle and a quad split on a diagonal.
                [Some(m01), Some(m12), None] => {
                    out.push([v0, m01, v2]);
                    out.push([m01, m12, v2]);
                    out.push([m01, v1, m12]);
                }
                [Some(m01), None, Some(m20)] => {
                    out.push([m20, v0, m01]);
                    out.push([m01, v1, v2]);
                    out.push([m01, v2, m20]);
                }
                [None, Some(m12), Some(m20)] => {
                    out.push([m12, v2, m20]);
                    out.push([v0, v1, m12]);
                    out.push([v0, m12, m20]);
                }
                // All three edges: the regular 1-to-4 subdivision.
                [Some(m01), Some(m12), Some(m20)] => {
                    out.push([v0, m01, m20]);
                    out.push([m01, v1, m12]);
                    out.push([m20, m12, v2]);
                    out.push([m01, m12, m20]);
                }
            }
        }
        *faces = out;
        if !split_any {
            break;
        }
    }
}

/// Collapse edges shorter than `min_len` onto their midpoint when the
/// resulting faces stay non-degenerate and the mesh stays a closed
/// manifold — the topology check of §12.2 is the outer `TriMesh::try_new`,
/// but degenerate or folded faces are rejected here.
fn collapse_short(verts: &mut Vec<DVec3>, faces: &mut Vec<[u32; 3]>, min_len: f64) {
    let edges = edge_map(faces);
    let mut kill: Vec<u32> = (0..verts.len() as u32).collect();
    let mut replace = faces.clone();
    // Vertex → the vertices of its one-ring, for the link condition.
    let mut ring: Vec<std::collections::HashSet<u32>> = vec![std::collections::HashSet::new(); verts.len()];
    for tri in faces.iter() {
        for e in 0..3 {
            ring[tri[e] as usize].insert(tri[(e + 1) % 3]);
            ring[tri[e] as usize].insert(tri[(e + 2) % 3]);
        }
    }
    for (&(a, b), fs) in &edges {
        if fs.len() != 2 {
            continue;
        }
        let (ra, rb) = (kill[a as usize], kill[b as usize]);
        if ra == rb {
            continue;
        }
        let (va, vb) = (verts[ra as usize], verts[rb as usize]);
        if va.distance(vb) >= min_len {
            continue;
        }
        // Link condition (§12.2 topology preservation): the endpoints' common
        // neighbours must be exactly the two apexes of the edge's faces, else
        // the collapse would pinch off a handle or merge a seam.
        let apexes: std::collections::HashSet<u32> = fs
            .iter()
            .flat_map(|&f| faces[f].iter().copied())
            .filter(|&v| v != a && v != b)
            .map(|v| kill[v as usize])
            .collect();
        let common: std::collections::HashSet<u32> = ring[ra as usize]
            .intersection(&ring[rb as usize])
            .map(|&v| kill[v as usize])
            .collect();
        if apexes.len() != 2 || common != apexes {
            continue;
        }
        // Collapse rb → ra at the midpoint; drop faces containing the edge.
        let mid = (va + vb) * 0.5;
        verts[ra as usize] = mid;
        for f in replace.iter_mut() {
            for v in f.iter_mut() {
                if *v == rb {
                    *v = ra;
                }
            }
        }
        for v in kill.iter_mut() {
            if *v == rb {
                *v = ra;
            }
        }
        let moved: Vec<u32> = ring[rb as usize].iter().copied().collect();
        ring[ra as usize].extend(moved.iter().map(|&v| kill[v as usize]));
        ring[ra as usize].remove(&rb);
        for v in moved {
            let kv = kill[v as usize] as usize;
            ring[kv].remove(&rb);
            ring[kv].insert(ra);
        }
    }
    let mut out = Vec::with_capacity(replace.len());
    for tri in replace {
        if tri[0] == tri[1] || tri[1] == tri[2] || tri[2] == tri[0] {
            continue;
        }
        let [a, b, c] = tri.map(|i| verts[i as usize]);
        let n = (b - a).cross(c - a);
        let scale = (b - a).length() * (c - a).length();
        if scale > 1e-30 && n.length() > 1e-12 * scale * 1e-9 {
            out.push(tri);
        }
    }
    if out.len() >= 4 {
        *faces = out;
    }
}

/// Tangential Laplacian smoothing: each vertex moves `λ` toward its ring
/// centroid, the displacement projected onto the plane of the vertex normal
/// (the mean of its face normals), so the relaxation does not shrink the
/// surface across its tangent.
fn smooth(verts: &mut Vec<DVec3>, faces: &[[u32; 3]], lambda: f64, rounds: usize) {
    for _ in 0..rounds {
        let mut ring: Vec<(DVec3, u32)> = vec![(DVec3::ZERO, 0); verts.len()];
        let mut normals = vec![DVec3::ZERO; verts.len()];
        for tri in faces {
            let [a, b, c] = tri.map(|i| verts[i as usize]);
            let n = (b - a).cross(c - a);
            for (i, j) in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])] {
                ring[i as usize].0 += verts[j as usize];
                ring[i as usize].1 += 1;
            }
            for v in tri {
                normals[*v as usize] += n;
            }
        }
        let next: Vec<DVec3> = verts
            .iter()
            .enumerate()
            .map(|(i, &x)| {
                let (sum, k) = ring[i];
                if k == 0 {
                    return x;
                }
                let step = (sum / k as f64 - x) * lambda;
                let n = normals[i].normalize_or_zero();
                x + step - n * step.dot(n)
            })
            .collect();
        verts.copy_from_slice(&next);
    }
}

/// `kernel` moved rigidly: `x ↦ rot x + shift`.
pub fn moved_kernel(kernel: &CauchyKernel, rot: DMat3, shift: DVec3) -> Result<CauchyKernel, SimError> {
    let metric = mat_sym(rot * sym_mat(kernel.metric) * rot.transpose());
    CauchyKernel::try_anisotropic(rot * kernel.center + shift, kernel.weight, metric, kernel.alpha)
}

pub fn bounds(pts: &[DVec3]) -> (DVec3, DVec3) {
    pts.iter().fold((DVec3::splat(f64::INFINITY), DVec3::splat(f64::NEG_INFINITY)), |(lo, hi), p| (lo.min(*p), hi.max(*p)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::geometry::tri_mesh::TriMesh;

    /// §12.2: a remeshed dumbbell keeps its neck — the waist must stay far
    /// narrower than the lobes, where the radial `rays_from` path would fill
    /// it in — and the result still validates as a closed oriented solid.
    #[test]
    fn a_dumbbell_keeps_its_neck() {
        let vox = Voxels::new(DVec3::splat(-30.0), DVec3::splat(30.0), 30, |p| {
            let lobe = (p.x.abs() - 12.0).powi(2) + p.y.powi(2) + p.z.powi(2) < 10.0f64.powi(2);
            let neck = p.x.abs() <= 12.0 && (p.y.powi(2) + p.z.powi(2)) < 3.5f64.powi(2);
            if lobe || neck { 0 } else { EMPTY }
        });
        let (mut verts, mut faces) = vox.surface(&|l| l != EMPTY);
        let mesh0 = TriMesh::try_new(verts.clone(), faces.clone()).expect("a closed mesh");
        mesh0.validate_solid().expect("an oriented solid");
        let vol0 = mesh0.signed_volume();

        let edge0: Vec<f64> = faces
            .iter()
            .flat_map(|t| [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])])
            .map(|(a, b)| verts[a as usize].distance(verts[b as usize]))
            .collect();
        let mean_edge = edge0.iter().sum::<f64>() / edge0.len() as f64;
        remesh(&mut verts, &mut faces, mean_edge * 0.9);

        let mesh = TriMesh::try_new(verts.clone(), faces.clone()).expect("remesh stays a mesh");
        mesh.validate_solid().expect("remesh stays an oriented solid");
        let vol = mesh.signed_volume();
        assert!((vol - vol0).abs() < 0.25 * vol0, "volume {vol} vs {vol0}");

        // The waist near x=0: the largest radial distance there stays ≪ lobe radius.
        let waist = verts
            .iter()
            .filter(|v| v.x.abs() < 1.0)
            .map(|v| (v.y.powi(2) + v.z.powi(2)).sqrt())
            .fold(0.0, f64::max);
        let lobe_r = verts
            .iter()
            .filter(|v| (v.x.abs() - 12.0).abs() < 3.0)
            .map(|v| (v.y.powi(2) + v.z.powi(2)).sqrt())
            .fold(0.0, f64::max);
        assert!(waist > 0.0, "neck vertices exist");
        assert!(waist < 0.75 * lobe_r, "waist {waist} vs lobe {lobe_r}");
    }
}
