//! f64 reference of the surface velocity trace, built from the definitions
//! rather than from the closed-form trace of Section 10.1.
//!
//! The potentials `Phi = R^2 sum q psi Y`, `Psi = R sum s psi Y` are evaluated at
//! arbitrary radius and direction, and the identities of Section 10.1,
//! `v_r = L^2 Phi / r`, `v_S = r^-1 grad d_r(r Phi)`, `v^T = grad Psi x e_r`,
//! are differentiated on the unit sphere by fourth-order great-circle stencils.
//! This checks the trace formula of the GPU kernel independently of the jets.

use super::basis::{K, lm_index, poly_derivative, poly_eval, psi_poly, sh_all};
use super::model::{StarSurface, VelocityField, local_frame};
use crate::common::math::DVec3;

pub fn radius(surface: &StarSurface, dir: DVec3) -> f64 {
    let y = sh_all(surface.l_max, dir.to_array());
    surface.coeff.iter().zip(&y).map(|(c, y)| c * y).sum()
}

const H: f64 = 2.0e-3;

fn on_circle(w: DVec3, t: DVec3, a: f64) -> DVec3 {
    w * a.cos() + t * a.sin()
}

/// Tangent gradient of `f` on the unit sphere at `w`.
pub fn gradient(f: &dyn Fn(DVec3) -> f64, w: DVec3) -> DVec3 {
    let (t1, t2) = local_frame(w);
    let mut g = DVec3::ZERO;
    for t in [t1, t2] {
        let d = (8.0 * (f(on_circle(w, t, H)) - f(on_circle(w, t, -H)))
            - (f(on_circle(w, t, 2.0 * H)) - f(on_circle(w, t, -2.0 * H))))
            / (12.0 * H);
        g = g + t * d;
    }
    g
}

fn second(f: &dyn Fn(DVec3) -> f64, w: DVec3, t: DVec3) -> f64 {
    let f0 = f(w);
    (-f(on_circle(w, t, 2.0 * H)) + 16.0 * f(on_circle(w, t, H)) - 30.0 * f0
        + 16.0 * f(on_circle(w, t, -H))
        - f(on_circle(w, t, -2.0 * H)))
        / (12.0 * H * H)
}

/// Laplace–Beltrami operator on the unit sphere.
pub fn laplacian(f: &dyn Fn(DVec3) -> f64, w: DVec3) -> f64 {
    let (t1, t2) = local_frame(w);
    second(f, w, t1) + second(f, w, t2)
}

/// Covariant Hessian of `f` on the unit sphere in the frame `(t1, t2)`: `[h11, h12, h22]`.
pub fn hessian(f: &dyn Fn(DVec3) -> f64, w: DVec3) -> [f64; 3] {
    let (t1, t2) = local_frame(w);
    let h11 = second(f, w, t1);
    let h22 = second(f, w, t2);
    let d = (t1 + t2) * std::f64::consts::FRAC_1_SQRT_2;
    let h12 = second(f, w, d) - 0.5 * (h11 + h22);
    [h11, h12, h22]
}

struct Potentials<'a> {
    surface: &'a StarSurface,
    velocity: &'a VelocityField,
    l_max: usize,
    psi: Vec<Vec<f64>>,
    dpsi: Vec<Vec<f64>>,
}

impl<'a> Potentials<'a> {
    fn new(surface: &'a StarSurface, velocity: &'a VelocityField) -> Self {
        let l_max = surface.l_max.max(velocity.l_max);
        let mut psi = Vec::new();
        let mut dpsi = Vec::new();
        for l in 0..=l_max {
            for k in 0..K {
                let p = psi_poly(l.max(1), k);
                let d = poly_derivative(&p);
                // l = 0 has no velocity; keep indices aligned.
                psi.push(p);
                dpsi.push(d);
            }
        }
        Self {
            surface,
            velocity,
            l_max,
            psi,
            dpsi,
        }
    }

    /// Returns `(Phi, d_r (r Phi), Psi)` at radius `r` along `w`.
    fn eval(&self, r: f64, w: DVec3) -> (f64, f64, f64) {
        let y = sh_all(self.l_max, w.to_array());
        let rr: f64 = self.surface.coeff.iter().zip(&y).map(|(c, y)| c * y).sum();
        let zeta = r / rr;
        let (mut phi, mut g, mut psi_v) = (0.0, 0.0, 0.0);
        for l in 1..=self.velocity.l_max {
            for m in -(l as i32)..=(l as i32) {
                let idx = lm_index(l, m);
                for k in 0..K {
                    let q = self.velocity.q[idx][k];
                    let s = self.velocity.s[idx][k];
                    if q == 0.0 && s == 0.0 {
                        continue;
                    }
                    let p = poly_eval(&self.psi[l * K + k], zeta);
                    let dp = poly_eval(&self.dpsi[l * K + k], zeta);
                    phi += q * rr * rr * p * y[idx];
                    g += q * rr * rr * y[idx] * (p + zeta * dp);
                    psi_v += s * rr * p * y[idx];
                }
            }
        }
        (phi, g, psi_v)
    }
}

/// Surface velocity `v|_Gamma` at the direction `w`, from the potentials.
pub fn trace_velocity(surface: &StarSurface, velocity: &VelocityField, w: DVec3) -> DVec3 {
    let pot = Potentials::new(surface, velocity);
    let r = radius(surface, w);
    let phi = |d: DVec3| pot.eval(r, d).0;
    let g = |d: DVec3| pot.eval(r, d).1;
    let psi = |d: DVec3| pot.eval(r, d).2;
    // L^2 = -Delta_S.
    let v_r = -laplacian(&phi, w) / r;
    let v_s = gradient(&g, w) * (1.0 / r);
    let v_t = gradient(&psi, w).cross(w);
    w * v_r + v_s + v_t
}
