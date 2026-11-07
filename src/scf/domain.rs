//! The integration domain `𝒱` of hypothesis (G) (§8.1): the volume rule on ray segments
//! from the origin (G2) with adaptive refinement of cut angular cells, the depth (G6),
//! and the boundary rule on `Γ` of the Nitsche terms (G3).

use std::collections::HashMap;

use crate::common::density::CauchyDensity;
use crate::common::error::SimError;
use crate::common::math::DVec3;
use crate::rtfp::chords;
use crate::rtfp::mesh::BodyMesh;

use super::quad::{gauss_jacobi_zeta2, gauss_legendre};
use super::surface::SurfaceIndex;

/// Volume quadrature points of `𝒱`.
#[derive(Clone, Debug)]
pub struct VolumeSet {
    pub pts: Vec<DVec3>,
    /// `dV` weights (m³).
    pub w: Vec<f64>,
    pub rho: Vec<f64>,
    pub grho: Vec<DVec3>,
    /// Depth `d = dist(x, Γ)` and `∇d = (x − π(x))/d` (§4.3a, G6).
    pub depth: Vec<f64>,
    pub gdepth: Vec<DVec3>,
    /// Rays of the angular rule, and those inside refined cut cells.
    pub rays: usize,
    pub cut_rays: usize,
    /// Rays whose orientation-rule construction saw an anomaly and fell back to
    /// the parity rule (MS §3.6 step 4).
    pub dirty_rays: usize,
}

/// One angular cell `[z0, z1] × [φ0, φ1]` of the product rule (`z = cos θ`).
#[derive(Clone, Copy)]
struct Cell {
    z: [f64; 2],
    phi: [f64; 2],
    depth: usize,
}

/// Segment structure of one ray: the occupied intervals along it.
type Segments = Vec<(f64, f64)>;

/// One ray of the angular rule: direction, solid-angle weight, segments.
type Ray = (DVec3, f64, Segments);

fn direction(z: f64, phi: f64) -> DVec3 {
    let s = (1.0 - z * z).max(0.0).sqrt();
    DVec3::new(s * phi.cos(), s * phi.sin(), z)
}

/// `(segment count, starts inside)`: rays with equal signatures see the same topology.
fn signature(seg: &Segments) -> (usize, bool) {
    (seg.len(), seg.first().is_some_and(|s| s.0 == 0.0))
}

/// `(segments, dirty count)`: the occupied intervals come from the orientation
/// rule of MS §3.6; a ray whose signed crossings do not alternate is dirty and
/// is rebuilt by the parity rule, which stays correct while the mesh is
/// watertight.
fn trace(mesh: &BodyMesh, dirs: &[DVec3]) -> Result<(Vec<Segments>, usize), SimError> {
    let mut batch = mesh.new_batch();
    for d in dirs {
        batch.push(DVec3::ZERO, *d, None);
    }
    let hits = mesh.trace(&batch, false)?;
    let (mut t, mut out) = (Vec::new(), Vec::with_capacity(dirs.len()));
    let mut dirty = 0;
    for i in 0..dirs.len() {
        t.clear();
        hits.crossings_into(i, &mut t);
        let mut seg = Vec::new();
        if chords::intervals_oriented(&t, mesh.scale(), &mut seg) > 0 {
            dirty += 1;
            let dist: Vec<f64> = t.iter().map(|h| h.0).collect();
            chords::intervals(&dist, mesh.scale(), &mut seg);
        }
        out.push(seg);
    }
    Ok((out, dirty))
}

/// The 2 × 2 Gauss points of a cell, with their solid-angle weights.
fn cell_points(c: &Cell) -> [(DVec3, f64); 4] {
    const X: f64 = 0.577_350_269_189_625_8;
    let (zm, zh) = (0.5 * (c.z[0] + c.z[1]), 0.5 * (c.z[1] - c.z[0]));
    let (pm, ph) = (0.5 * (c.phi[0] + c.phi[1]), 0.5 * (c.phi[1] - c.phi[0]));
    let w = zh * ph;
    let mut out = [(DVec3::ZERO, 0.0); 4];
    for (k, (a, b)) in [(-X, -X), (X, -X), (-X, X), (X, X)].into_iter().enumerate() {
        out[k] = (direction(zm + a * zh, pm + b * ph), w);
    }
    out
}

/// Angular rays of `𝒱` with their weights and segments: the product rule, and inside
/// every cell whose ray topology differs from a neighbour's, a quadtree of 2 × 2 Gauss
/// cells refined while their four rays disagree, down to `max_depth`.
fn angular_rays(
    mesh: &BodyMesh,
    n_theta: usize,
    n_phi: usize,
    max_depth: usize,
) -> Result<(Vec<Ray>, usize, usize, usize), SimError> {
    let (z, wz) = gauss_legendre(n_theta);
    let dphi = 2.0 * std::f64::consts::PI / n_phi as f64;
    let mut dirs = Vec::with_capacity(n_theta * n_phi);
    for zi in &z {
        for j in 0..n_phi {
            dirs.push(direction(*zi, dphi * (j as f64 + 0.5)));
        }
    }
    let (base, mut dirty) = trace(mesh, &dirs)?;
    let at = |i: usize, j: usize| i * n_phi + j;
    let mut out = Vec::with_capacity(base.len());
    let mut cells = Vec::new();
    let mut z0 = -1.0;
    for i in 0..n_theta {
        let z1 = z0 + wz[i];
        for j in 0..n_phi {
            let sig = signature(&base[at(i, j)]);
            let mut nb = vec![at(i, (j + 1) % n_phi), at(i, (j + n_phi - 1) % n_phi)];
            if i > 0 {
                nb.push(at(i - 1, j));
            }
            if i + 1 < n_theta {
                nb.push(at(i + 1, j));
            }
            if max_depth > 0 && nb.iter().any(|&k| signature(&base[k]) != sig) {
                let phi0 = dphi * j as f64;
                cells.push(Cell {
                    z: [z0, z1],
                    phi: [phi0, phi0 + dphi],
                    depth: 1,
                });
            } else {
                out.push((dirs[at(i, j)], wz[i] * dphi, base[at(i, j)].clone()));
            }
        }
        z0 = z1;
    }
    let rays = out.len() + cells.len();
    let mut cut = 0;
    while !cells.is_empty() {
        let pts: Vec<[(DVec3, f64); 4]> = cells.iter().map(cell_points).collect();
        let dirs: Vec<DVec3> = pts.iter().flat_map(|p| p.iter().map(|x| x.0)).collect();
        let (segs, d) = trace(mesh, &dirs)?;
        dirty += d;
        let mut next = Vec::new();
        for (k, c) in cells.iter().enumerate() {
            let four = &segs[4 * k..4 * k + 4];
            let uniform = four.iter().all(|s| signature(s) == signature(&four[0]));
            if uniform || c.depth >= max_depth {
                for (p, s) in pts[k].iter().zip(four) {
                    out.push((p.0, p.1, s.clone()));
                }
                cut += 4;
            } else {
                let (zm, pm) = (0.5 * (c.z[0] + c.z[1]), 0.5 * (c.phi[0] + c.phi[1]));
                for (za, zb) in [(c.z[0], zm), (zm, c.z[1])] {
                    for (pa, pb) in [(c.phi[0], pm), (pm, c.phi[1])] {
                        next.push(Cell {
                            z: [za, zb],
                            phi: [pa, pb],
                            depth: c.depth + 1,
                        });
                    }
                }
            }
        }
        cells = next;
    }
    Ok((out, rays, cut, dirty))
}

/// The ray-segment rule of (G2): on each segment, `n_radial` Gauss–Jacobi nodes with
/// weight `r²` when it starts at the origin, Gauss–Legendre on `f r²` otherwise. Returns
/// points, weights, the counts of angular rays, of rays in refined cut cells, and of
/// dirty rays (MS §3.6 step 4).
pub fn volume_rule(
    mesh: &BodyMesh,
    n_theta: usize,
    n_phi: usize,
    n_radial: usize,
    max_depth: usize,
) -> Result<(Vec<DVec3>, Vec<f64>, usize, usize, usize), SimError> {
    let (rays, n_rays, cut_rays, dirty) = angular_rays(mesh, n_theta, n_phi, max_depth)?;
    let (zeta, wj) = gauss_jacobi_zeta2(n_radial);
    let (xl, wl) = gauss_legendre(n_radial);
    let (mut pts, mut w) = (Vec::new(), Vec::new());
    for (dir, wa, segs) in &rays {
        for &(a, b) in segs {
            if a == 0.0 {
                for (z, wz) in zeta.iter().zip(&wj) {
                    pts.push(*dir * (z * b));
                    w.push(wa * wz * b * b * b);
                }
            } else {
                for (x, wx) in xl.iter().zip(&wl) {
                    let r = a + 0.5 * (b - a) * (1.0 + x);
                    pts.push(*dir * r);
                    w.push(wa * 0.5 * (b - a) * wx * r * r);
                }
            }
        }
    }
    Ok((pts, w, n_rays, cut_rays, dirty))
}

impl VolumeSet {
    pub fn build(
        mesh: &BodyMesh,
        index: &SurfaceIndex,
        density: &CauchyDensity,
        n_theta: usize,
        n_phi: usize,
        n_radial: usize,
        max_depth: usize,
    ) -> Result<Self, SimError> {
        let (pts, w, n_rays, cut_rays, dirty) =
            volume_rule(mesh, n_theta, n_phi, n_radial, max_depth)?;
        let (depth, gdepth) = depth(index, &pts);
        Ok(Self {
            rho: pts.iter().map(|p| density.rho(*p)).collect(),
            grho: pts.iter().map(|p| density.grad(*p)).collect(),
            pts,
            w,
            depth,
            gdepth,
            rays: n_rays,
            cut_rays,
            dirty_rays: dirty,
        })
    }

    pub fn mass(&self) -> f64 {
        self.w.iter().zip(&self.rho).map(|(w, r)| w * r).sum()
    }
}

/// `d = |x − π(x)|` and `∇d = (x − π(x))/d` for points of `𝒱` (G6). A point on `Γ`
/// gets `d = 0` and the inward normal of its face.
pub fn depth(index: &SurfaceIndex, pts: &[DVec3]) -> (Vec<f64>, Vec<DVec3>) {
    use rayon::prelude::*;
    pts.par_iter()
        .map(|x| {
            let c = index.closest(*x);
            let d = c.dist2.sqrt();
            if d > 0.0 {
                (d, (*x - c.point) * (1.0 / d))
            } else {
                let [a, b, cc] = index.triangle(c.face);
                (0.0, -(b - a).cross(cc - a).normalize_or_zero())
            }
        })
        .unzip()
}

/// Points on `Γ` with unit outward normals and area weights: the face centres (the nodes
/// of the friction chain and of `p_Γ`), or the boundary rule of the Nitsche terms (G3).
#[derive(Clone, Debug)]
pub struct SurfacePoints {
    pub pts: Vec<DVec3>,
    pub normals: Vec<DVec3>,
    pub w: Vec<f64>,
}

impl SurfacePoints {
    /// Every face centre with its normal and area.
    pub fn faces(faces: &[(DVec3, DVec3, f64)]) -> Self {
        Self {
            pts: faces.iter().map(|f| f.0).collect(),
            normals: faces.iter().map(|f| f.1).collect(),
            w: faces.iter().map(|f| f.2).collect(),
        }
    }

    /// The boundary rule of (G3): faces grouped by a voxel grid of spacing `√(|Γ| / target)`.
    /// Each group is one point (the face centre nearest its area centroid, so the point
    /// lies on `Γ`), its unit mean normal, and its area.
    pub fn clustered(faces: &[(DVec3, DVec3, f64)], target: usize) -> Self {
        let total: f64 = faces.iter().map(|f| f.2).sum();
        let h = (total / target.max(1) as f64).sqrt();
        let key = |c: DVec3| ((c / h).floor().as_i64vec3()).to_array();
        // Per group: area, area-weighted centroid, vector area, member faces.
        let mut groups: HashMap<[i64; 3], (f64, DVec3, DVec3, Vec<usize>)> = HashMap::new();
        for (f, (c, n, a)) in faces.iter().enumerate() {
            let g = groups.entry(key(*c)).or_insert((0.0, DVec3::ZERO, DVec3::ZERO, Vec::new()));
            g.0 += a;
            g.1 += *c * *a;
            g.2 += *n * *a;
            g.3.push(f);
        }
        let mut out = Self {
            pts: Vec::with_capacity(groups.len()),
            normals: Vec::with_capacity(groups.len()),
            w: Vec::with_capacity(groups.len()),
        };
        let mut sorted: Vec<_> = groups.into_iter().collect();
        sorted.sort_by_key(|g| g.0);
        for (_, (area, moment, vector, members)) in sorted {
            let centroid = moment / area;
            let rep = members
                .iter()
                .copied()
                .min_by(|a, b| {
                    (faces[*a].0 - centroid)
                        .length_squared()
                        .total_cmp(&(faces[*b].0 - centroid).length_squared())
                })
                .expect("a group has a face");
            out.pts.push(faces[rep].0);
            out.normals.push(vector.try_normalize().unwrap_or(faces[rep].1));
            out.w.push(area);
        }
        out
    }
}
