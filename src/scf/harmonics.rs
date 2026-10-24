//! Spherical-harmonic jets and the spherical-harmonic fit `R_SH(ω̂)` of the body surface.
//!
//! `rheo_friction` and the Stokes basis both live on the surface
//! `R(ω̂) = Σ r_lm Y_lm(ω̂)`; the polyhedron is only the gravity geometry. The fit is
//! the single bridge between the two and its error is reported on every run.

use std::f64::consts::{PI, SQRT_2};


use crate::rheo::basis::lm_index;

use super::jets::Jet;

/// Real orthonormal harmonics up to `l_max`, as jets in the *unit-vector* jets `x`.
///
/// This is the recursion of `rheo::basis::sh_all`, which uses only products and
/// sums, so the result is a polynomial jet in the arguments.
pub fn sh_jets(l_max: usize, x: &[Jet; 3]) -> Vec<Jet> {
    let ord = x[0].ord.min(x[1].ord).min(x[2].ord);
    let zero = Jet::constant(0.0, ord);
    let mut out = vec![zero; (l_max + 1) * (l_max + 1)];
    let mut cre = Jet::constant(1.0, ord);
    let mut cim = zero;
    let mut pmm = (1.0 / (4.0 * PI)).sqrt();
    for m in 0..=l_max {
        if m > 0 {
            let nre = cre.mul(&x[0]).sub(&cim.mul(&x[1]));
            let nim = cre.mul(&x[1]).add(&cim.mul(&x[0]));
            cre = nre;
            cim = nim;
            pmm *= ((2 * m + 1) as f64 / (2 * m) as f64).sqrt();
        }
        let mut p2 = zero;
        let mut p1 = zero;
        for l in m..=l_max {
            let p = if l == m {
                Jet::constant(pmm, ord)
            } else {
                let lf = l as f64;
                let mf = m as f64;
                let a = ((4.0 * lf * lf - 1.0) / (lf * lf - mf * mf)).sqrt();
                let b = (((lf - 1.0) * (lf - 1.0) - mf * mf)
                    / (4.0 * (lf - 1.0) * (lf - 1.0) - 1.0))
                    .sqrt();
                if l == m + 1 {
                    x[2].mul(&p1).scale(a)
                } else {
                    x[2].mul(&p1).sub(&p2.scale(b)).scale(a)
                }
            };
            if m == 0 {
                out[lm_index(l, 0)] = p;
            } else {
                out[lm_index(l, m as i32)] = p.mul(&cre).scale(SQRT_2);
                out[lm_index(l, -(m as i32))] = p.mul(&cim).scale(SQRT_2);
            }
            p2 = p1;
            p1 = p;
        }
    }
    out
}

/// Spherical-harmonic surface `R(ω̂) = Σ c_lm Y_lm(ω̂)`.
#[derive(Clone, Debug)]
pub struct SurfaceFit {
    pub l_max: usize,
    pub coeff: Vec<f64>,
}

impl SurfaceFit {
    /// The constant radius `R ≡ r`: the enclosing ball of the general basis (§8.0, G1).
    pub fn sphere(r: f64) -> Self {
        Self {
            l_max: 0,
            coeff: vec![r * (4.0 * std::f64::consts::PI).sqrt()],
        }
    }
}
