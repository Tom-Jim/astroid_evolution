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
