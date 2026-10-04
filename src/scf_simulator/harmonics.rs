//! Spherical-harmonic jets and the spherical-harmonic fit `R_SH(ω̂)` of the body surface.
//!
//! `rheo_friction` and the Stokes basis both live on the surface
//! `R(ω̂) = Σ r_lm Y_lm(ω̂)`; the polyhedron is only the gravity geometry. The fit is
//! the single bridge between the two and its error is reported on every run.

use std::f64::consts::{PI, SQRT_2};

use nalgebra::{Cholesky, DMatrix, DVector};

use crate::common::math::DVec3;
use crate::rheo_friction::basis::{lm_index, sh_all};

use super::error::ScfError;
use super::jets::Jet;

/// Real orthonormal harmonics up to `l_max`, as jets in the *unit-vector* jets `x`.
///
/// This is the recursion of `rheo_friction::basis::sh_all`, which uses only products and
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
    ) -> Result<Self, ScfError> {
        let n = (l_max + 1) * (l_max + 1);
        if dirs.len() < n {
            return Err(ScfError::geometry(format!(
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
        let chol = Cholesky::new(gram)
            .ok_or_else(|| ScfError::geometry("spherical-harmonic normal equations are singular"))?;
        let coeff = chol.solve(&rhs).iter().copied().collect();
        Ok(Self { l_max, coeff })
    }

    pub fn radius(&self, dir: DVec3) -> f64 {
        let y = sh_all(self.l_max, dir.to_array());
        self.coeff.iter().zip(&y).map(|(c, y)| c * y).sum()
    }

    /// `R` as a jet, from harmonic jets of at least degree `l_max`.
    pub fn radius_jet(&self, y: &[Jet]) -> Jet {
        let mut r = Jet::constant(0.0, y[0].ord);
        for (c, yj) in self.coeff.iter().zip(y) {
            r.axpy(*c, yj);
        }
        r
    }

    /// The `StarSurface` consumed by `rheo_friction`.
    pub fn to_rheo(&self) -> crate::rheo_friction::StarSurface {
        crate::rheo_friction::StarSurface {
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
        let x = [
            Jet::var(0, w.x, 1),
            Jet::var(1, w.y, 1),
            Jet::var(2, w.z, 1),
        ];
        let r2 = x[0].mul(&x[0]).add(&x[1].mul(&x[1])).add(&x[2].mul(&x[2]));
        let inv = r2.sqrt().recip();
        let u = [x[0].mul(&inv), x[1].mul(&inv), x[2].mul(&inv)];
        let r = self.radius_jet(&sh_jets(self.l_max, &u));
        // `∇F = e_r − R⁻¹ ∇_S R` for `F = r − R(ω̂)`; the jet gradient at |x| = 1 is `∇_S R`.
        let g = r.grad();
        let k = 1.0 / r.val();
        let n = DVec3::new(w.x - g[0] * k, w.y - g[1] * k, w.z - g[2] * k);
        (r.val(), n.try_normalize().unwrap_or(w))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn point(i: usize, n: usize) -> DVec3 {
        let golden = PI * (3.0 - 5.0_f64.sqrt());
        let z = 1.0 - 2.0 * (i as f64 + 0.5) / n as f64;
        let r = (1.0 - z * z).sqrt();
        DVec3::new(r * (golden * i as f64).cos(), r * (golden * i as f64).sin(), z)
    }

    #[test]
    fn jets_reproduce_the_scalar_recursion_and_its_gradient() {
        let l_max = 5;
        let w = point(7, 31);
        let x = [Jet::var(0, w.x, 3), Jet::var(1, w.y, 3), Jet::var(2, w.z, 3)];
        let jets = sh_jets(l_max, &x);
        let plain = sh_all(l_max, w.to_array());
        let h = 1e-6;
        for k in 0..plain.len() {
            assert!((jets[k].val() - plain[k]).abs() < 1e-13, "value {k}");
            // The recursion is a polynomial in the arguments, so shifting the argument
            // (without renormalising) must follow the jet.
            let shifted = |axis: usize, s: f64| {
                let mut p = w.to_array();
                p[axis] += s;
                sh_all(l_max, p)[k]
            };
            for axis in 0..3 {
                let fd = (shifted(axis, h) - shifted(axis, -h)) / (2.0 * h);
                assert!((jets[k].grad()[axis] - fd).abs() < 1e-6, "gradient {k} {axis}");
            }
        }
    }

    #[test]
    fn fit_recovers_a_low_degree_surface() {
        let n = 400;
        let dirs: Vec<DVec3> = (0..n).map(|i| point(i, n)).collect();
        let truth = |d: DVec3| 100.0 + 8.0 * d.z * d.z + 3.0 * d.x * d.y;
        let radii: Vec<f64> = dirs.iter().map(|d| truth(*d)).collect();
        let w = vec![4.0 * PI / n as f64; n];
        let fit = SurfaceFit::fit(4, &dirs, &radii, &w).unwrap();
        assert!(fit.max_relative_error(&dirs, &radii) < 1e-8);
        let (r, nrm) = fit.normal(DVec3::new(0.0, 0.0, 1.0));
        assert!((r - 108.0).abs() < 1e-6);
        assert!((nrm.z - 1.0).abs() < 1e-9);
    }
}
