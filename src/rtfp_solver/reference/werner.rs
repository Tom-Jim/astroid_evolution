//! Closed-form field of a uniform polyhedron.
//!
//! Vectors run from the field point to the vertex. `L` is the Werner edge
//! factor and `ω` is the signed solid angle of an outward triangle. The
//! Hessian is the symmetric part of the dyad sum. Its trace is `-4π G ρ`
//! inside, `-2π G ρ` at a face interior point, and `0` outside.

use crate::common::geometry::TriMesh;
use crate::common::math::{DVec3, Sym6};

#[derive(Clone, Copy, Debug)]
pub struct UniformField {
    pub potential: f64,
    pub gradient: DVec3,
    pub hessian: Sym6,
}

pub fn uniform_polyhedron(mesh: &TriMesh, x: DVec3, g: f64, rho: f64) -> UniformField {
    let mut integral = 0.0;
    let mut gradient = DVec3::ZERO;
    let mut tensor = [[0.0; 3]; 3];
    let own_face = face_containing(mesh, x);
    for f in 0..mesh.faces.len() {
        let [a, b, c] = mesh.triangle(f);
        let ra = a - x;
        let rb = b - x;
        let rc = c - x;
        let Some(n) = (rb - ra).cross(rc - ra).try_normalize() else {
            continue;
        };
        // Seen from a point on its own plane the face subtends +2π from inside and −2π from
        // outside; the mean of the one-sided Hessians takes zero.
        let omega = if own_face == Some(f) {
            0.0
        } else {
            solid_angle(ra, rb, rc)
        };
        let h = n.dot(ra);
        integral -= 0.5 * h * h * omega;
        gradient = gradient + n * (h * omega);
        let nv = n.to_array();
        for i in 0..3 {
            for j in 0..3 {
                tensor[i][j] -= nv[i] * nv[j] * omega;
            }
        }
        let verts = [ra, rb, rc];
        for e in 0..3 {
            let r1 = verts[e];
            let r2 = verts[(e + 1) % 3];
            let Some(ehat) = (r2 - r1).try_normalize() else {
                continue;
            };
            let n_edge = ehat.cross(n);
            let along = n_edge.dot(r1);
            let l_e = werner_edge_factor(r1, r2);
            integral += 0.5 * h * along * l_e;
            gradient = gradient - n * (along * l_e);
            let ev = n_edge.to_array();
            for i in 0..3 {
                for j in 0..3 {
                    tensor[i][j] += nv[i] * ev[j] * l_e;
                }
            }
        }
    }
    let scale = g * rho;
    let hessian = [
        scale * tensor[0][0],
        scale * tensor[1][1],
        scale * tensor[2][2],
        scale * 0.5 * (tensor[0][1] + tensor[1][0]),
        scale * 0.5 * (tensor[0][2] + tensor[2][0]),
        scale * 0.5 * (tensor[1][2] + tensor[2][1]),
    ];
    UniformField {
        potential: scale * integral,
        gradient: gradient * scale,
        hessian,
    }
}

/// The face whose interior contains `x`, if `x` lies on one.
pub fn face_containing(mesh: &TriMesh, x: DVec3) -> Option<usize> {
    for f in 0..mesh.faces.len() {
        let [a, b, c] = mesh.triangle(f);
        let n = (b - a).cross(c - a);
        let area2 = n.length();
        if area2 == 0.0 {
            continue;
        }
        if n.dot(x - a).abs() > 1e-8 * area2 {
            continue;
        }
        let c0 = (b - x).cross(c - x).dot(n);
        let c1 = (c - x).cross(a - x).dot(n);
        let c2 = (a - x).cross(b - x).dot(n);
        let sum = c0 + c1 + c2;
        if sum.abs() <= 1e-18 {
            continue;
        }
        let u = c0 / sum;
        let v = c1 / sum;
        let w = c2 / sum;
        if u > 1e-8 && v > 1e-8 && w > 1e-8 {
            return Some(f);
        }
    }
    None
}

/// Distance from `x` to the nearest point of the triangle surface.
pub fn distance_to_surface(mesh: &TriMesh, x: DVec3) -> f64 {
    (0..mesh.faces.len())
        .map(|f| distance_to_triangle(x, mesh.triangle(f)))
        .fold(f64::INFINITY, f64::min)
}

fn distance_to_triangle(p: DVec3, [a, b, c]: [DVec3; 3]) -> f64 {
    let (ab, ac, ap) = (b - a, c - a, p - a);
    let (d1, d2) = (ab.dot(ap), ac.dot(ap));
    if d1 <= 0.0 && d2 <= 0.0 {
        return ap.length();
    }
    let bp = p - b;
    let (d3, d4) = (ab.dot(bp), ac.dot(bp));
    if d3 >= 0.0 && d4 <= d3 {
        return bp.length();
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        return (a + ab * (d1 / (d1 - d3)) - p).length();
    }
    let cp = p - c;
    let (d5, d6) = (ab.dot(cp), ac.dot(cp));
    if d6 >= 0.0 && d5 <= d6 {
        return cp.length();
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        return (a + ac * (d2 / (d2 - d6)) - p).length();
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && d4 - d3 >= 0.0 && d5 - d6 >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return (b + (c - b) * w - p).length();
    }
    let denom = 1.0 / (va + vb + vc);
    (a + ab * (vb * denom) + ac * (vc * denom) - p).length()
}

fn werner_edge_factor(r1: DVec3, r2: DVec3) -> f64 {
    let a = r1.length();
    let b = r2.length();
    let e = (r2 - r1).length();
    if e == 0.0 || !a.is_finite() || !b.is_finite() {
        return 0.0;
    }
    let den = (a + b - e).max(1e-300);
    ((a + b + e) / den).ln()
}

fn solid_angle(a: DVec3, b: DVec3, c: DVec3) -> f64 {
    let num = a.dot(b.cross(c));
    let den = a.length() * b.length() * c.length()
        + a.dot(b) * c.length()
        + b.dot(c) * a.length()
        + c.dot(a) * b.length();
    2.0 * num.atan2(den)
}
