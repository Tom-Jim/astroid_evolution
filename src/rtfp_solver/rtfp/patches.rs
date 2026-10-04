//! Face-partition solid-angle quadrature (MS §3.2).
//!
//! The solid angle seen from `x` is split along the mesh faces. A surface patch
//! `P` contributes the solid-angle element `|n·û| dA / s²` in direction `û`
//! at distance `s`, evaluated at the patch centroid (midpoint rule). Faces are
//! subdivided until each patch spans a small angle as seen from `x`, so the
//! rule converges for points arbitrarily close to the surface.
//!
//! A patch is a valid sample only when the first surface crossing along its ray
//! is the patch itself. Patches occluded by another part of the body are
//! rejected by the ray-tracing hardware; where validity changes between
//! neighbouring faces the patches are refined so the visibility boundary is
//! resolved to the same angular tolerance.

use rayon::prelude::*;

use crate::common::geometry::TriMesh;
use crate::common::math::DVec3;
use crate::rtfp_solver::error::RtfpError;
use crate::rtfp_solver::rtfp::hw_rt::{HardwareMesh, RayBatch};

/// A patch is split while its longest edge exceeds this fraction of its distance from the point.
pub const DEFAULT_ANGULAR_RATIO: f64 = 0.06;
/// Maximum subdivision depth of a patch beyond the ratio rule, used to resolve occlusion boundaries.
const VISIBILITY_DEPTH: usize = 4;
/// Patches whose plane contains the point carry no solid angle.
const GRAZING: f64 = 1.0e-9;
/// Rays per hardware launch while validating patches.
const BATCH: usize = 1 << 21;

#[derive(Clone, Copy)]
pub struct FaceGeometry {
    pub tri: [DVec3; 3],
    pub normal: DVec3,
    pub diameter: f64,
}

/// Per-face data the quadrature reads, plus edge adjacency.
pub struct Faces {
    pub geometry: Vec<FaceGeometry>,
    neighbors: Vec<[u32; 3]>,
}

impl Faces {
    pub fn new(mesh: &TriMesh) -> Result<Self, RtfpError> {
        let mut geometry = Vec::with_capacity(mesh.faces.len());
        for f in 0..mesh.faces.len() {
            let tri = mesh.triangle(f);
            let normal = mesh
                .face_normal(f)
                .ok_or_else(|| RtfpError::geometry(format!("face {f} is degenerate")))?;
            let diameter = (tri[1] - tri[0])
                .length()
                .max((tri[2] - tri[1]).length())
                .max((tri[0] - tri[2]).length());
            geometry.push(FaceGeometry {
                tri,
                normal,
                diameter,
            });
        }
        let mut edges: Vec<(u32, u32, u32)> = Vec::with_capacity(3 * mesh.faces.len());
        for (f, face) in mesh.faces.iter().enumerate() {
            for e in 0..3 {
                let (a, b) = (face[e], face[(e + 1) % 3]);
                edges.push((a.min(b), a.max(b), f as u32));
            }
        }
        edges.sort_unstable();
        let mut neighbors = vec![[u32::MAX; 3]; mesh.faces.len()];
        let mut slot = vec![0usize; mesh.faces.len()];
        for pair in edges.windows(2) {
            if pair[0].0 == pair[1].0 && pair[0].1 == pair[1].1 {
                let (f, g) = (pair[0].2 as usize, pair[1].2 as usize);
                neighbors[f][slot[f] % 3] = g as u32;
                neighbors[g][slot[g] % 3] = f as u32;
                slot[f] += 1;
                slot[g] += 1;
            }
        }
        Ok(Self {
            geometry,
            neighbors,
        })
    }

    pub fn len(&self) -> usize {
        self.geometry.len()
    }

    pub fn is_empty(&self) -> bool {
        self.geometry.is_empty()
    }
}

/// A triangular piece of one mesh face.
#[derive(Clone, Copy)]
pub struct Leaf {
    tri: [DVec3; 3],
    pub face: u32,
    diameter: f64,
}

/// A patch seen from a point: unit direction, distance, and solid-angle weight `|n·û| A / s²`.
#[derive(Clone, Copy)]
pub struct Sample {
    pub dir: DVec3,
    pub dist: f64,
    pub weight: f64,
}

impl Leaf {
    fn of_face(index: usize, face: &FaceGeometry) -> Self {
        Self {
            tri: face.tri,
            face: index as u32,
            diameter: face.diameter,
        }
    }

    fn centroid(&self) -> DVec3 {
        (self.tri[0] + self.tri[1] + self.tri[2]) * (1.0 / 3.0)
    }

    fn area(&self) -> f64 {
        0.5 * (self.tri[1] - self.tri[0])
            .cross(self.tri[2] - self.tri[0])
            .length()
    }

    fn children(&self) -> [Leaf; 4] {
        let [a, b, c] = self.tri;
        let (ab, bc, ca) = ((a + b) * 0.5, (b + c) * 0.5, (c + a) * 0.5);
        let d = self.diameter * 0.5;
        let make = |tri| Leaf {
            tri,
            face: self.face,
            diameter: d,
        };
        [
            make([a, ab, ca]),
            make([ab, b, bc]),
            make([ca, bc, c]),
            make([ab, bc, ca]),
        ]
    }

    /// The three interior samples (barycentric `(2/3, 1/6, 1/6)`, a degree-2 rule of weight `A/3`
    /// each) of the patch as seen from `x`, or `None` when its plane contains `x`.
    pub fn samples(&self, x: DVec3, normal: DVec3) -> Option<[Sample; 3]> {
        let [a, b, c] = self.tri;
        let area = self.area();
        let mut out = [Sample {
            dir: DVec3::ZERO,
            dist: 0.0,
            weight: 0.0,
        }; 3];
        for (slot, point) in out.iter_mut().zip([
            a * (2.0 / 3.0) + (b + c) * (1.0 / 6.0),
            b * (2.0 / 3.0) + (c + a) * (1.0 / 6.0),
            c * (2.0 / 3.0) + (a + b) * (1.0 / 6.0),
        ]) {
            let rel = point - x;
            let dist = rel.length();
            if dist == 0.0 {
                return None;
            }
            let dir = rel * (1.0 / dist);
            let cos = normal.dot(dir).abs();
            if cos <= GRAZING {
                return None;
            }
            *slot = Sample {
                dir,
                dist,
                weight: cos * area / (3.0 * dist * dist),
            };
        }
        Some(out)
    }
}

/// How many of a patch's three sample rays see the patch itself first.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Status {
    Hidden,
    Visible,
    Partial,
}

/// All samples of the accepted patches of one point, flattened.
pub fn flatten(x: DVec3, leaves: &[Leaf], faces: &Faces) -> Vec<Sample> {
    leaves
        .iter()
        .flat_map(|leaf| {
            leaf.samples(x, faces.geometry[leaf.face as usize].normal)
                .expect("accepted patch has samples")
        })
        .collect()
}

/// Patches of every face for point `x`, split to the angular ratio.
fn base_leaves(x: DVec3, faces: &Faces, ratio: f64) -> Vec<Leaf> {
    (0..faces.len())
        .into_par_iter()
        .flat_map_iter(|f| {
            let face = &faces.geometry[f];
            let mut out = Vec::new();
            let mut stack = vec![Leaf::of_face(f, face)];
            let mut first = true;
            while let Some(leaf) = stack.pop() {
                if leaf.samples(x, face.normal).is_none() {
                    if first {
                        break;
                    }
                    continue;
                }
                first = false;
                if leaf.diameter > ratio * (leaf.centroid() - x).length() {
                    stack.extend(leaf.children());
                } else {
                    out.push(leaf);
                }
            }
            out
        })
        .collect()
}

/// Visibility status of every patch of every point.
fn validity(
    hw: &HardwareMesh,
    faces: &Faces,
    xs: &[DVec3],
    leaves: &[Vec<Leaf>],
) -> Result<Vec<Vec<Status>>, RtfpError> {
    let mut out: Vec<Vec<Status>> = leaves
        .iter()
        .map(|l| vec![Status::Hidden; l.len()])
        .collect();
    let jobs: Vec<(usize, usize)> = leaves
        .iter()
        .enumerate()
        .flat_map(|(p, l)| (0..l.len()).map(move |i| (p, i)))
        .collect();
    for chunk in jobs.chunks(BATCH) {
        let parts: Vec<RayBatch> = chunk
            .par_chunks(4096)
            .map(|piece| {
                let mut batch = hw.new_batch();
                for &(p, i) in piece {
                    let leaf = &leaves[p][i];
                    let samples = leaf
                        .samples(xs[p], faces.geometry[leaf.face as usize].normal)
                        .expect("leaf has samples");
                    for s in samples {
                        batch.push(xs[p], s.dir, Some(s.dist));
                    }
                }
                batch
            })
            .collect();
        let mut batch = hw.new_batch();
        for mut part in parts {
            batch.append(&mut part);
        }
        let hits = hw.trace(&batch, true)?;
        for (k, &(p, i)) in chunk.iter().enumerate() {
            let seen = (0..3).filter(|&j| hits.count(3 * k + j) > 0).count();
            out[p][i] = match seen {
                0 => Status::Hidden,
                3 => Status::Visible,
                _ => Status::Partial,
            };
        }
    }
    Ok(out)
}

/// The valid patches of every point in `xs`, with occlusion boundaries refined.
pub fn settle(
    hw: &HardwareMesh,
    faces: &Faces,
    xs: &[DVec3],
    ratio: f64,
) -> Result<Vec<Vec<Leaf>>, RtfpError> {
    struct Pending {
        leaf: Leaf,
        status: Status,
    }

    let mut leaves: Vec<Vec<Leaf>> = xs.iter().map(|&x| base_leaves(x, faces, ratio)).collect();
    let status = validity(hw, faces, xs, &leaves)?;
    let mut done: Vec<Vec<Leaf>> = vec![Vec::new(); xs.len()];
    let mut pending: Vec<Vec<Pending>> = (0..xs.len()).map(|_| Vec::new()).collect();

    // A face is flagged when it, or a neighbouring face, mixes visibility;
    // flagged patches are the ones that may straddle an occlusion boundary.
    for p in 0..xs.len() {
        let mut all = vec![true; faces.len()];
        let mut any = vec![false; faces.len()];
        let mut has = vec![false; faces.len()];
        for (leaf, &v) in leaves[p].iter().zip(&status[p]) {
            let f = leaf.face as usize;
            all[f] &= v == Status::Visible;
            any[f] |= v != Status::Hidden;
            has[f] = true;
        }
        let uniform = |f: usize| all[f] == any[f];
        for (leaf, &v) in std::mem::take(&mut leaves[p]).into_iter().zip(&status[p]) {
            let f = leaf.face as usize;
            let straddles = v == Status::Partial
                || !uniform(f)
                || faces.neighbors[f].iter().any(|&g| {
                    g != u32::MAX
                        && has[g as usize]
                        && (any[g as usize] != any[f] || !uniform(g as usize))
                });
            if straddles {
                pending[p].push(Pending { leaf, status: v });
            } else if v == Status::Visible {
                done[p].push(leaf);
            }
        }
    }

    for _ in 0..VISIBILITY_DEPTH {
        if pending.iter().all(Vec::is_empty) {
            break;
        }
        let children: Vec<Vec<Leaf>> = (0..xs.len())
            .map(|p| {
                pending[p]
                    .iter()
                    .flat_map(|q| q.leaf.children())
                    .filter(|c| {
                        c.samples(xs[p], faces.geometry[c.face as usize].normal)
                            .is_some()
                    })
                    .collect()
            })
            .collect();
        let child_status = validity(hw, faces, xs, &children)?;
        let mut next: Vec<Vec<Pending>> = (0..xs.len()).map(|_| Vec::new()).collect();
        for p in 0..xs.len() {
            // Weightless children were filtered out, so walk them in the same order.
            let mut cursor = 0;
            for q in &pending[p] {
                let normal = faces.geometry[q.leaf.face as usize].normal;
                let mut group = Vec::with_capacity(4);
                for kid in q.leaf.children() {
                    if kid.samples(xs[p], normal).is_some() {
                        group.push(Pending {
                            leaf: kid,
                            status: child_status[p][cursor],
                        });
                        cursor += 1;
                    }
                }
                let first = group.first().map(|k| k.status);
                let uniform =
                    first != Some(Status::Partial) && group.iter().all(|k| Some(k.status) == first);
                if !uniform {
                    next[p].extend(group);
                } else if first == Some(q.status) {
                    if q.status == Status::Visible {
                        done[p].push(q.leaf);
                    }
                } else {
                    done[p].extend(
                        group
                            .into_iter()
                            .filter(|k| k.status == Status::Visible)
                            .map(|k| k.leaf),
                    );
                }
            }
        }
        pending = next;
    }
    for p in 0..xs.len() {
        for q in std::mem::take(&mut pending[p]) {
            if q.status == Status::Visible {
                done[p].push(q.leaf);
            }
        }
    }
    Ok(done)
}
