//! Forces and torques on rigid fragments (§12.3): the primary as a point mass with `J₂`, and
//! between fragments MacCullagh's quadrupole expansion far apart and the RTFP field of one
//! integrated over the volume nodes of the other close together. Every pair force is applied
//! with its reaction and its torques, so momentum and angular momentum are conserved exactly.

use crate::common::error::SimError;
use crate::common::math::{DMat3, DVec3};

use super::Fragment;

/// Gravitational constant (m³ kg⁻¹ s⁻²).
pub const G: f64 = 6.6743e-11;

/// The primary: `GM`, equatorial radius and `J₂`, with its spin axis along `z`.
#[derive(Clone, Copy, Debug)]
pub struct Primary {
    pub gm: f64,
    pub radius: f64,
    pub j2: f64,
}

impl Primary {
    /// `−∇Φ` with `Φ = −(GM/r)[1 − J₂ (R/r)² P₂(z/r)]`.
    pub fn accel(&self, x: DVec3) -> DVec3 {
        let r2 = x.length_squared();
        let r = r2.sqrt();
        let k = 1.5 * self.j2 * self.gm * self.radius * self.radius / r.powi(5);
        let s = 5.0 * x.z * x.z / r2;
        x * (-self.gm / (r2 * r)) + DVec3::new(x.x * (s - 1.0), x.y * (s - 1.0), x.z * (s - 3.0)) * k
    }

    /// `Φ` per unit mass.
    pub fn potential(&self, x: DVec3) -> f64 {
        let r = x.length();
        let p2 = 1.5 * (x.z / r).powi(2) - 0.5;
        -self.gm / r * (1.0 - self.j2 * (self.radius / r).powi(2) * p2)
    }
}

/// `Q(r) = (tr I − 3 r̂ᵀ I r̂) / (2 r³)`: the quadrupole part of a body's potential per `G`.
fn quad(i: DMat3, r: DVec3) -> f64 {
    let r2 = r.length_squared();
    let tr = i.x_axis.x + i.y_axis.y + i.z_axis.z;
    (tr - 3.0 * r.dot(i * r) / r2) / (2.0 * r2 * r2.sqrt())
}

/// `∇Q(r)`.
fn quad_grad(i: DMat3, r: DVec3) -> DVec3 {
    let r2 = r.length_squared();
    let r5 = r2 * r2 * r2.sqrt();
    let tr = i.x_axis.x + i.y_axis.y + i.z_axis.z;
    r * (-1.5 * tr / r5) - (i * r) * (3.0 / r5) + r * (7.5 * r.dot(i * r) / (r5 * r2))
}

/// Gravity-gradient torque on a body of inertia `i` from a mass `gm` at `rho` from its centre.
fn gradient_torque(i: DMat3, rho: DVec3, gm: f64) -> DVec3 {
    rho.cross(i * rho) * (3.0 * gm / rho.length().powi(5))
}

/// Centre distance, in sums of radii, below which a pair is integrated over volume nodes.
pub const NEAR: f64 = 3.0;

/// Forces and torques on every fragment, and the mutual potential energy (with the
/// quadrupole expansion also for close pairs, so it is a diagnostic there).
pub fn forces(primary: &Primary, frags: &[Fragment]) -> Result<(Vec<(DVec3, DVec3)>, f64), SimError> {
    let n = frags.len();
    let mut out = vec![(DVec3::ZERO, DVec3::ZERO); n];
    let mut energy = 0.0;
    for (k, f) in frags.iter().enumerate() {
        let iw = f.inertia_world();
        out[k].0 += primary.accel(f.x) * f.mass + quad_grad(iw, f.x) * (primary.gm * 1.0);
        out[k].1 += gradient_torque(iw, -f.x, primary.gm);
        energy += f.mass * primary.potential(f.x) - primary.gm * quad(iw, f.x);
    }
    for j in 0..n {
        for k in j + 1..n {
            let (a, b) = (&frags[j], &frags[k]);
            let r = b.x - a.x;
            let (ia, ib) = (a.inertia_world(), b.inertia_world());
            energy -= G * (a.mass * b.mass / r.length() + b.mass * quad(ia, r) + a.mass * quad(ib, r));
            let (fb, ta, tb) = if r.length() < NEAR * (a.radius + b.radius) {
                near(a, b)?
            } else {
                let fb = r * (-G * a.mass * b.mass / r.length().powi(3)) + quad_grad(ia, r) * (G * b.mass) + quad_grad(ib, r) * (G * a.mass);
                (fb, gradient_torque(ia, r, G * b.mass), gradient_torque(ib, -r, G * a.mass))
            };
            out[k].0 += fb;
            out[j].0 -= fb;
            out[j].1 += ta;
            out[k].1 += tb;
        }
    }
    Ok((out, energy))
}

/// The pair force on `b` and the torques on `a` and `b` from the RTFP field of the heavier
/// body integrated over the volume nodes of the lighter: each node's force `m_i g(y_i)` acts
/// at the node, and its reaction on the heavier body at the same point.
fn near(a: &Fragment, b: &Fragment) -> Result<(DVec3, DVec3, DVec3), SimError> {
    let (heavy, light, light_is_b) = if a.mass >= b.mass { (a, b, true) } else { (b, a, false) };
    let pts: Vec<DVec3> = light.nodes.iter().map(|p| light.x + light.rot * *p).collect();
    let local: Vec<DVec3> = pts.iter().map(|y| heavy.rot.transpose() * (*y - heavy.x)).collect();
    let field = heavy.fwd.eval_exterior(&local)?;
    let (mut force, mut t_light, mut t_heavy) = (DVec3::ZERO, DVec3::ZERO, DVec3::ZERO);
    for ((y, (g, _)), m) in pts.iter().zip(&field).zip(&light.node_mass) {
        let fi = heavy.rot * *g * *m;
        force += fi;
        t_light += (*y - light.x).cross(fi);
        t_heavy -= (*y - heavy.x).cross(fi);
    }
    Ok(if light_is_b { (force, t_heavy, t_light) } else { (-force, t_light, t_heavy) })
}
