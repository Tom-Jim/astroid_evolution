//! Spherical-harmonic jets and the spherical-harmonic fit `R_SH(ω̂)` of the body surface.
//!
//! `rheo_friction` and the Stokes basis both live on the surface
//! `R(ω̂) = Σ r_lm Y_lm(ω̂)`; the polyhedron is only the gravity geometry. The fit is
//! the single bridge between the two and its error is reported on every run.

use std::f64::consts::{PI, SQRT_2};

use nalgebra::{Cholesky, DMatrix, DVector};

use crate::common::math::DVec3;
use crate::rheo::basis::{lm_index, sh_all};

use super::jets::Jet;
use crate::common::error::SimError;

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
    /// Weighted least squares through `(ω̂_i, R_i)`. A tiny Laplace–Beltrami ridge keeps
    /// the normal equations definite when the sample is sparse at high degree.
    pub fn fit(
        l_max: usize,
        dirs: &[DVec3],
        radii: &[f64],
        weights: &[f64],
    ) -> Result<Self, SimError> {
        let n = (l_max + 1) * (l_max + 1);
        if dirs.len() < n {
            return Err(SimError::geometry(format!(
                "{} surface samples cannot determine {} harmonic coefficients",
                dirs.len(),
                n
            )));
        }
        let mut gram = DMatrix::<f64>::zeros(n, n);
        let mut rhs = DVector::<f64>::zeros(n);
        for ((d, r), w) in dirs.iter().zip(radii).zip(weights) {
            let y = sh_all(l_max, d.to_array());
            for a in 0..n {
                rhs[a] += w * r * y[a];
                for b in 0..=a {
                    gram[(a, b)] += w * y[a] * y[b];
                }
            }
        }
        for a in 0..n {
            for b in 0..a {
                gram[(b, a)] = gram[(a, b)];
            }
        }
        let scale = gram.trace() / n as f64;
        for l in 0..=l_max {
            for m in -(l as i32)..=(l as i32) {
                let i = lm_index(l, m);
                gram[(i, i)] += 1.0e-12 * scale * (1.0 + (l * (l + 1)) as f64);
            }
        }
        let chol = Cholesky::new(gram).ok_or_else(|| {
            SimError::geometry("spherical-harmonic normal equations are singular")
        })?;
        let coeff = chol.solve(&rhs).iter().copied().collect();
        Ok(Self { l_max, coeff })
    }

    pub fn radius(&self, dir: DVec3) -> f64 {
        let y = sh_all(self.l_max, dir.to_array());
        self.coeff.iter().zip(&y).map(|(c, y)| c * y).sum()
    }

    /// The `StarSurface` consumed by `rheo_friction`.
    pub fn to_rheo(&self) -> crate::rheo::StarSurface {
        crate::rheo::StarSurface {
            l_max: self.l_max,
            coeff: self.coeff.clone(),
        }
    }

    /// Largest `|R_SH(ω̂_i) − R_i| / R_i`.
    pub fn max_relative_error(&self, dirs: &[DVec3], radii: &[f64]) -> f64 {
        dirs.iter()
            .zip(radii)
            .map(|(d, r)| (self.radius(*d) - r).abs() / r)
            .fold(0.0, f64::max)
    }

    /// Radius and analytic outward unit normal of the surface `r = R(ω̂)` at the direction `w`.
    pub fn normal(&self, w: DVec3) -> (f64, DVec3) {
        self.normals(std::slice::from_ref(&w))
            .into_iter()
            .next()
            .unwrap_or((0.0, w))
    }

    /// Radius and outward unit normal at every direction, one GPU launch.
    pub fn normals(&self, dirs: &[DVec3]) -> Vec<(f64, DVec3)> {
        super::basis_gpu::normals(self, dirs)
            .unwrap_or_else(|err| panic!("surface normal GPU: {err}"))
    }
}
