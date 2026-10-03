//! Closed-form uniform polyhedron field (Werner–Scheeres).
//!
//! Vectors are measured from the field point to the vertex. The solid angle uses the
//! Van Oosterom–Strackee two-argument form so the interior and exterior signs differ.
//! The potential is the positive Newtonian integral `G ρ ∫ 1/R dV`.

use crate::common::geometry::TriMesh;
use crate::common::math::{DVec3, Sym6};

#[derive(Clone, Copy, Debug)]
pub struct PolyhedronField {
    pub potential: f64,
    pub gradient: DVec3,
    pub hessian: Sym6,
}

pub fn uniform_field(mesh: &TriMesh, x: DVec3, g: f64, rho: f64) -> PolyhedronField {
    let integral = newton_volume_integral(mesh, x);
    let potential = g * rho * integral;
    let step = 1e-5;
    let mut gradient = DVec3::ZERO;
    let axes = [DVec3::new(step, 0.0, 0.0), DVec3::new(0.0, step, 0.0), DVec3::new(0.0, 0.0, step)];
    let samples: Vec<f64> = axes
        .iter()
        .flat_map(|a| {
            [
                newton_volume_integral(mesh, x + *a),
                newton_volume_integral(mesh, x - *a),
            ]
        })
        .collect();
    gradient.x = g * rho * (samples[0] - samples[1]) / (2.0 * step);
    gradient.y = g * rho * (samples[2] - samples[3]) / (2.0 * step);
    gradient.z = g * rho * (samples[4] - samples[5]) / (2.0 * step);
    let mut hessian = [0.0; 6];
    for (i, axis) in axes.iter().enumerate() {
        let gp = field_gradient_component(mesh, x + *axis, i);
        let gm = field_gradient_component(mesh, x - *axis, i);
        hessian[i] = g * rho * (gp - gm) / (2.0 * step);
    }
    hessian[3] = g * rho
        * (field_gradient_component(mesh, x + axes[0], 1)
            - field_gradient_component(mesh, x - axes[0], 1))
        / (2.0 * step);
    hessian[4] = g * rho
        * (field_gradient_component(mesh, x + axes[0], 2)
            - field_gradient_component(mesh, x - axes[0], 2))
        / (2.0 * step);
    hessian[5] = g * rho
        * (field_gradient_component(mesh, x + axes[1], 2)
            - field_gradient_component(mesh, x - axes[1], 2))
        / (2.0 * step);
    PolyhedronField {
        potential,
        gradient,
        hessian,
    }
}

fn field_gradient_component(mesh: &TriMesh, x: DVec3, axis: usize) -> f64 {
    let step = 1e-5;
    let mut d = DVec3::ZERO;
    match axis {
        0 => d.x = step,
        1 => d.y = step,
        _ => d.z = step,
    }
    (newton_volume_integral(mesh, x + d) - newton_volume_integral(mesh, x - d)) / (2.0 * step)
}

/// `∫_V |x−y|^{-1} dV` for a constant-density polyhedron.
///
/// The identity `∫_V R^{-1} dV = ½ Σ_f h_f ∫_{face} R^{-1} dS` reduces each
/// planar face to edge logarithms and the signed solid angle. On every face,
/// `∫ dS/R = −Σ_e (n × ê · r) L_e − h ω`.
fn newton_volume_integral(mesh: &TriMesh, x: DVec3) -> f64 {
    let mut sum = 0.0;
    for f in 0..mesh.faces.len() {
        let [a, b, c] = mesh.triangle(f);
        let ra = a - x;
        let rb = b - x;
        let rc = c - x;
        let Some(n) = (rb - ra).cross(rc - ra).try_normalize() else {
            continue;
        };
        let omega = solid_angle(ra, rb, rc);
        let h = n.dot(ra);
        let mut edges = 0.0;
        let verts = [ra, rb, rc];
        for e in 0..3 {
            let r1 = verts[e];
            let r2 = verts[(e + 1) % 3];
            let Some(ehat) = (r2 - r1).try_normalize() else {
                continue;
            };
            let n_edge = n.cross(ehat);
            edges += n_edge.dot(r1) * edge_log(r1, r2);
        }
        let surface = -edges - h * omega;
        sum += 0.5 * h * surface;
    }
    sum
}

fn edge_log(r1: DVec3, r2: DVec3) -> f64 {
    let edge = r2 - r1;
    let elen = edge.length();
    if elen == 0.0 {
        return 0.0;
    }
    let ehat = edge * (1.0 / elen);
    let s1 = r1.dot(ehat);
    let s2 = r2.dot(ehat);
    let num = r2.length() + s2;
    let den = (r1.length() + s1).abs().max(1e-300);
    // ∫ ds / R along the edge. The numerator stays positive for a finite segment
    // that does not pass through the field point.
    if num <= 0.0 {
        let l1 = r1.length();
        let l2 = r2.length();
        ((l1 + l2 + elen) / (l1 + l2 - elen).max(1e-300)).ln()
    } else {
        (num / den).ln()
    }
}

fn solid_angle(a: DVec3, b: DVec3, c: DVec3) -> f64 {
    let num = a.dot(b.cross(c));
    let den = a.length() * b.length() * c.length()
        + a.dot(b) * c.length()
        + b.dot(c) * a.length()
        + c.dot(a) * b.length();
    2.0 * num.atan2(den)
}

#[cfg(test)]
mod face_formula {
    use super::*;
    use crate::common::geometry::TriMesh;

    fn tri_inv_r(a: DVec3, b: DVec3, c: DVec3, depth: u32) -> f64 {
        let area = (b - a).cross(c - a).length() * 0.5;
        let centroid = (a + b + c) * (1.0 / 3.0);
        let span = (b - a).length().max((c - a).length());
        if depth >= 8 || span < 1e-3 * centroid.length() {
            return area / centroid.length();
        }
        let ab = (a + b) * 0.5;
        let bc = (b + c) * 0.5;
        let ca = (c + a) * 0.5;
        tri_inv_r(a, ab, ca, depth + 1)
            + tri_inv_r(ab, b, bc, depth + 1)
            + tri_inv_r(ca, bc, c, depth + 1)
            + tri_inv_r(ab, bc, ca, depth + 1)
    }

    #[test]
    fn one_face_matches_edge_reduction() {
        let mesh = TriMesh::unit_cube();
        let x = DVec3::new(3.0, 0.4, -0.25);
        for f in 0..mesh.faces.len() {
            let [a, b, c] = mesh.triangle(f);
            let ra = a - x;
            let rb = b - x;
            let rc = c - x;
            let n = (b - a).cross(c - a).try_normalize().unwrap();
            let omega = solid_angle(ra, rb, rc);
            let h = n.dot(ra);
            let mut edges = 0.0;
            let verts = [ra, rb, rc];
            for e in 0..3 {
                let r1 = verts[e];
                let r2 = verts[(e + 1) % 3];
                let ehat = (r2 - r1).try_normalize().unwrap();
                let n_edge = n.cross(ehat);
                edges += n_edge.dot(r1) * edge_log(r1, r2);
            }
            let surface = -edges - h * omega;
            let numeric = tri_inv_r(a - x, b - x, c - x, 0);
            assert!(
                (numeric - surface).abs() < 2e-3,
                "face {f} numeric {numeric} analytic {surface}"
            );
        }
    }
}

/// Singular-free exterior check: recursively integrate `1/R` over the centroid fan.
pub fn exterior_newton_potential(mesh: &TriMesh, x: DVec3, g: f64, rho: f64) -> f64 {
    let integrals = crate::rtfp_solver::volume::uniform_integrals(mesh, 1.0);
    let c = integrals.centroid;
    let mut acc = 0.0;
    for f in 0..mesh.faces.len() {
        let [a, b, d] = mesh.triangle(f);
        acc += tet_inv_r(c, a, b, d, x, 0);
    }
    g * rho * acc
}

fn tet_inv_r(a: DVec3, b: DVec3, c: DVec3, d: DVec3, x: DVec3, depth: u32) -> f64 {
    let vol = (b - a).dot((c - a).cross(d - a)).abs() / 6.0;
    if vol == 0.0 {
        return 0.0;
    }
    let centroid = (a + b + c + d) * 0.25;
    let span = (b - a)
        .length()
        .max((c - a).length())
        .max((d - a).length());
    let dist = (centroid - x).length();
    if depth >= 7 || span < 0.15 * dist {
        return vol / dist;
    }
    let ab = (a + b) * 0.5;
    let ac = (a + c) * 0.5;
    let ad = (a + d) * 0.5;
    let bc = (b + c) * 0.5;
    let bd = (b + d) * 0.5;
    let cd = (c + d) * 0.5;
    tet_inv_r(a, ab, ac, ad, x, depth + 1)
        + tet_inv_r(b, ab, bc, bd, x, depth + 1)
        + tet_inv_r(c, ac, bc, cd, x, depth + 1)
        + tet_inv_r(d, ad, bd, cd, x, depth + 1)
        + tet_inv_r(ab, ac, ad, bc, x, depth + 1)
        + tet_inv_r(ab, ad, bd, bc, x, depth + 1)
        + tet_inv_r(ac, ad, bc, cd, x, depth + 1)
        + tet_inv_r(ad, bc, bd, cd, x, depth + 1)
}

pub fn homogeneous_sphere(radius: f64, rho: f64, g: f64, r: f64) -> PolyhedronField {
    let m = 4.0 / 3.0 * std::f64::consts::PI * rho * radius.powi(3);
    if r < radius {
        let u = 2.0 * std::f64::consts::PI * g * rho * (radius * radius - r * r / 3.0);
        let gp = -(4.0 * std::f64::consts::PI * g * rho / 3.0) * r;
        let h = -(4.0 * std::f64::consts::PI * g * rho / 3.0);
        PolyhedronField {
            potential: u,
            gradient: DVec3::new(gp, 0.0, 0.0),
            hessian: [h, h, h, 0.0, 0.0, 0.0],
        }
    } else {
        let u = g * m / r;
        let gp = -g * m / (r * r);
        let h_rr = 2.0 * g * m / r.powi(3);
        let h_tt = -g * m / r.powi(3);
        PolyhedronField {
            potential: u,
            gradient: DVec3::new(gp, 0.0, 0.0),
            hessian: [h_rr, h_tt, h_tt, 0.0, 0.0, 0.0],
        }
    }
}
