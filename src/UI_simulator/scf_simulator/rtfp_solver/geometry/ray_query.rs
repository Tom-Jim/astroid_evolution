//! Ray–triangle queries shared by the BVH and the brute-force checker.
//!
//! The same function is the only intersection predicate. Counting and writing
//! therefore cannot disagree because of two copies of the arithmetic.

use crate::common::geometry::{FlattenedBvh, TriMesh};
use crate::common::math::DVec3;

#[derive(Clone, Copy, Debug)]
pub struct RayHit {
    pub t: f64,
    pub face: u32,
}

/// Möller–Trumbore with a scale-aware parallel reject.
///
/// Shared edges are accepted by both incident triangles and later coalesced by
/// `segments_from_hits`, which restores a single crossing. A coplanar ray is rejected.
pub fn ray_triangle(origin: DVec3, dir: DVec3, v0: DVec3, v1: DVec3, v2: DVec3) -> Option<f64> {
    let e1 = v1 - v0;
    let e2 = v2 - v0;
    let p = dir.cross(e2);
    let det = e1.dot(p);
    let scale = e1.length() * e2.length() * dir.length();
    if scale == 0.0 || det.abs() <= 1e-14 * scale {
        return None;
    }
    let inv = 1.0 / det;
    let tvec = origin - v0;
    let u = tvec.dot(p) * inv;
    if !(-1e-12..=1.0 + 1e-12).contains(&u) {
        return None;
    }
    let q = tvec.cross(e1);
    let v = dir.dot(q) * inv;
    if v < -1e-12 || u + v > 1.0 + 1e-12 {
        return None;
    }
    let t = e2.dot(q) * inv;
    if t > 0.0 && t.is_finite() {
        Some(t)
    } else {
        None
    }
}

pub fn intersect_bruteforce(mesh: &TriMesh, origin: DVec3, dir: DVec3) -> Vec<RayHit> {
    let mut hits = Vec::new();
    for (face, _) in mesh.faces.iter().enumerate() {
        let [a, b, c] = mesh.triangle(face);
        if let Some(t) = ray_triangle(origin, dir, a, b, c) {
            hits.push(RayHit {
                t,
                face: face as u32,
            });
        }
    }
    hits.sort_by(|p, q| p.t.partial_cmp(&q.t).unwrap_or(std::cmp::Ordering::Equal));
    hits
}

pub fn intersect_bvh(mesh: &TriMesh, bvh: &FlattenedBvh, origin: DVec3, dir: DVec3) -> Vec<RayHit> {
    if bvh.min.is_empty() {
        return Vec::new();
    }
    let inv = DVec3::new(safe_inv(dir.x), safe_inv(dir.y), safe_inv(dir.z));
    let mut hits = Vec::new();
    let mut stack = [0u32; 64];
    let mut sp = 1usize;
    while sp > 0 {
        sp -= 1;
        let node = stack[sp] as usize;
        if !slab(origin, inv, dir, bvh.min[node], bvh.max[node]) {
            continue;
        }
        let count = bvh.count[node];
        if count > 0 {
            let start = bvh.start[node] as usize;
            for k in 0..count as usize {
                let face = bvh.order[start + k] as usize;
                let [a, b, c] = mesh.triangle(face);
                if let Some(t) = ray_triangle(origin, dir, a, b, c) {
                    hits.push(RayHit {
                        t,
                        face: face as u32,
                    });
                }
            }
        } else if sp + 2 <= stack.len() {
            stack[sp] = bvh.left[node];
            sp += 1;
            stack[sp] = bvh.right[node];
            sp += 1;
        }
    }
    hits.sort_by(|p, q| p.t.partial_cmp(&q.t).unwrap_or(std::cmp::Ordering::Equal));
    hits
}

fn safe_inv(c: f64) -> f64 {
    if c.abs() < 1e-30 {
        1e30_f64.copysign(c)
    } else {
        1.0 / c
    }
}

fn slab(origin: DVec3, inv: DVec3, dir: DVec3, bmin: DVec3, bmax: DVec3) -> bool {
    let mut t0 = f64::NEG_INFINITY;
    let mut t1 = f64::INFINITY;
    for axis in 0..3 {
        let o = component(origin, axis);
        let d = component(dir, axis);
        let mn = component(bmin, axis);
        let mx = component(bmax, axis);
        if d.abs() < 1e-30 {
            if o < mn || o > mx {
                return false;
            }
            continue;
        }
        let inv_c = component(inv, axis);
        let mut ta = (mn - o) * inv_c;
        let mut tb = (mx - o) * inv_c;
        if ta > tb {
            std::mem::swap(&mut ta, &mut tb);
        }
        t0 = t0.max(ta);
        t1 = t1.min(tb);
        if t1 < t0 {
            return false;
        }
    }
    t1 >= 0.0 && t1 >= t0
}

fn component(v: DVec3, axis: usize) -> f64 {
    match axis {
        0 => v.x,
        1 => v.y,
        _ => v.z,
    }
}

/// Odd crossing count along `+e_x` after coalescing duplicate edge hits.
pub fn point_inside(mesh: &TriMesh, bvh: &FlattenedBvh, x: DVec3) -> bool {
    let hits = intersect_bvh(mesh, bvh, x, DVec3::new(1.0, 1.0e-3, -2.0e-3));
    let mut n = 0u32;
    let mut last = f64::NEG_INFINITY;
    for h in hits {
        let tol = (1e-9 * h.t).max(1e-12);
        if h.t - last > tol {
            n += 1;
            last = h.t;
        }
    }
    n % 2 == 1
}

pub fn distance_to_mesh(mesh: &TriMesh, x: DVec3) -> f64 {
    let mut best = f64::INFINITY;
    for f in 0..mesh.faces.len() {
        best = best.min(distance_to_triangle(x, mesh.triangle(f)));
    }
    best
}

fn distance_to_triangle(p: DVec3, tri: [DVec3; 3]) -> f64 {
    let [a, b, c] = tri;
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return ap.length();
    }
    let bp = p - b;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 {
        return bp.length();
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return (a + ab * v - p).length();
    }
    let cp = p - c;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= 0.0 && d5 <= d6 {
        return cp.length();
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return (a + ac * w - p).length();
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return (b + (c - b) * w - p).length();
    }
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    (a + ab * v + ac * w - p).length()
}
