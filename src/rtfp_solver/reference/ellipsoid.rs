//! Exact field of a homogeneous ellipsoid through Carlson integrals.
//!
//! With semi-axes `a_i`, `Δ(s) = √Π(a_i² + s)` and `λ` the largest root of
//! `Σ ξ_i² / (a_i² + λ) = 1` (`λ = 0` inside),
//!
//! ```text
//! U    = π G ρ abc [ 2 R_F − (2/3) Σ ξ_i² R_D,i ]
//! ∂_i U = −2 π G ρ abc ξ_i J_i,                 J_i = (2/3) R_D,i
//! ∂_ij U = −2 π G ρ abc [ δ_ij J_i − 2 ξ_i ξ_j / ((a_i²+λ)(a_j²+λ) S Δ(λ)) ],
//! S = Σ ξ_i² / (a_i² + λ)²   (the second term is absent inside),
//! ```
//!
//! where `ξ = x − center` and `R_D,i = R_D(a_j²+λ, a_k²+λ, a_i²+λ)`.

use std::f64::consts::PI;

use super::carlson::{rd, rf};
use gauss_quad::GaussLegendre;
use std::num::NonZeroUsize;

use crate::common::density::CauchyDensity;
use crate::common::math::{DVec3, Sym6};

#[derive(Clone, Copy, Debug)]
pub struct Ellipsoid {
    pub center: DVec3,
    pub semi_axes: [f64; 3],
}

#[derive(Clone, Copy, Debug)]
pub struct EllipsoidField {
    pub potential: f64,
    pub gradient: DVec3,
    pub hessian: Sym6,
}

impl Ellipsoid {
    pub fn new(center: DVec3, semi_axes: [f64; 3]) -> Self {
        assert!(
            semi_axes.iter().all(|a| a.is_finite() && *a > 0.0),
            "ellipsoid semi-axes must be finite and positive"
        );
        Self { center, semi_axes }
    }

    pub fn volume(&self) -> f64 {
        4.0 / 3.0 * PI * self.semi_axes.iter().product::<f64>()
    }

    /// Whether `x` is strictly inside.
    pub fn contains(&self, x: DVec3) -> bool {
        self.level(x) < 1.0
    }

    fn level(&self, x: DVec3) -> f64 {
        let xi = (x - self.center).to_array();
        (0..3).map(|i| (xi[i] / self.semi_axes[i]).powi(2)).sum()
    }

    /// Largest root `λ` of `Σ ξ_i² / (a_i² + λ) = 1`, or 0 inside.
    fn lambda(&self, xi: [f64; 3]) -> f64 {
        let a2 = self.semi_axes.map(|a| a * a);
        let f = |l: f64| (0..3).map(|i| xi[i] * xi[i] / (a2[i] + l)).sum::<f64>() - 1.0;
        if f(0.0) <= 0.0 {
            return 0.0;
        }
        let (mut lo, mut hi) = (0.0, xi.iter().map(|v| v * v).sum::<f64>());
        for _ in 0..200 {
            let mid = 0.5 * (lo + hi);
            if f(mid) > 0.0 { lo = mid } else { hi = mid }
            if hi - lo <= 1e-16 * hi {
                break;
            }
        }
        0.5 * (lo + hi)
    }

    pub fn field(&self, x: DVec3, g: f64, rho: f64) -> EllipsoidField {
        let xi = (x - self.center).to_array();
        let a2 = self.semi_axes.map(|a| a * a);
        let lambda = self.lambda(xi);
        let s = [a2[0] + lambda, a2[1] + lambda, a2[2] + lambda];
        let delta = (s[0] * s[1] * s[2]).sqrt();
        let rf_val = rf(s[0], s[1], s[2]);
        let rd_i = [rd(s[1], s[2], s[0]), rd(s[2], s[0], s[1]), rd(s[0], s[1], s[2])];
        let j = rd_i.map(|v| 2.0 / 3.0 * v);
        let abc: f64 = self.semi_axes.iter().product();
        let k = PI * g * rho * abc;

        let potential = k * (2.0 * rf_val - (0..3).map(|i| xi[i] * xi[i] * j[i]).sum::<f64>());
        let gradient = DVec3::new(-2.0 * k * xi[0] * j[0], -2.0 * k * xi[1] * j[1], -2.0 * k * xi[2] * j[2]);

        let mut h = [[0.0; 3]; 3];
        let outside = lambda > 0.0;
        let big_s: f64 = (0..3).map(|i| xi[i] * xi[i] / (s[i] * s[i])).sum();
        for i in 0..3 {
            for jx in 0..3 {
                let mut v = if i == jx { j[i] } else { 0.0 };
                if outside {
                    v -= 2.0 * xi[i] * xi[jx] / (s[i] * s[jx] * big_s * delta);
                }
                h[i][jx] = -2.0 * k * v;
            }
        }
        EllipsoidField {
            potential,
            gradient,
            hessian: [h[0][0], h[1][1], h[2][2], h[0][1], h[0][2], h[1][2]],
        }
    }
}

impl Ellipsoid {
    /// Field of a density that depends only on the ellipsoidal radius `m = √Σ(ξ_i/a_i)²`,
    /// `ρ(m) = bulk + Σ w (1 + s² m²)^{-α}`, as a stack of homogeneous ellipsoids:
    /// `F = ρ(1) F₁(1) − ∫₀¹ ρ'(m) F₁(m) dm`, where `F₁(m)` is the unit-density ellipsoid with
    /// semi-axes `m a`. Kernels must be centred on the ellipsoid and the ellipsoid a sphere
    /// (an isotropic kernel is a function of `m` only then).
    pub fn cauchy_field(&self, x: DVec3, g: f64, density: &CauchyDensity) -> Result<EllipsoidField, String> {
        let a = self.semi_axes;
        if !density.kernels.is_empty() {
            if !(a[0] == a[1] && a[1] == a[2]) {
                return Err("a Cauchy kernel is isotropic: use a sphere for the Carlson benchmark".into());
            }
            if density.kernels.iter().any(|k| (k.center - self.center).length() > 1e-12 * a[0]) {
                return Err("Carlson Cauchy benchmark needs kernels centred on the sphere".into());
            }
        }
        let profile = |m: f64| -> (f64, f64) {
            let mut rho = density.bulk;
            let mut slope = 0.0;
            for k in &density.kernels {
                let s2 = (k.sigma * a[0]).powi(2);
                let base = 1.0 + s2 * m * m;
                rho += k.weight * base.powf(-k.alpha);
                slope -= 2.0 * k.alpha * k.weight * s2 * m * base.powf(-k.alpha - 1.0);
            }
            (rho, slope)
        };
        let unit = |m: f64| Ellipsoid::new(self.center, [a[0] * m, a[1] * m, a[2] * m]).field(x, g, 1.0);
        let rule = GaussLegendre::new(NonZeroUsize::new(12).expect("nonzero"));
        let kink = self.level(x).sqrt().min(1.0);
        let mut cuts = vec![0.0];
        if kink > 1e-9 && kink < 1.0 {
            cuts.push(kink);
        }
        cuts.push(1.0);
        let mut acc = [0.0_f64; 10];
        let pack = |f: &EllipsoidField| -> [f64; 10] {
            [f.potential, f.gradient.x, f.gradient.y, f.gradient.z, f.hessian[0], f.hessian[1], f.hessian[2], f.hessian[3], f.hessian[4], f.hessian[5]]
        };
        for seg in cuts.windows(2) {
            let panels = 24;
            let h = (seg[1] - seg[0]) / panels as f64;
            for p in 0..panels {
                let lo = seg[0] + h * p as f64;
                for &(node, weight) in rule.as_node_weight_pairs() {
                    let m = lo + 0.5 * h * (node + 1.0);
                    let v = pack(&unit(m));
                    let w = 0.5 * h * weight * profile(m).1;
                    for i in 0..10 {
                        acc[i] -= w * v[i];
                    }
                }
            }
        }
        let edge = pack(&unit(1.0));
        let rho1 = profile(1.0).0;
        for i in 0..10 {
            acc[i] += rho1 * edge[i];
        }
        Ok(EllipsoidField {
            potential: acc[0],
            gradient: DVec3::new(acc[1], acc[2], acc[3]),
            hessian: [acc[4], acc[5], acc[6], acc[7], acc[8], acc[9]],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sphere_matches_the_closed_forms() {
        let (g, rho, r) = (1.0, 2.0, 1.5);
        let body = Ellipsoid::new(DVec3::ZERO, [r, r, r]);
        let mass = 4.0 / 3.0 * PI * r.powi(3) * rho;
        let inside = body.field(DVec3::new(0.3, -0.4, 0.5), g, rho);
        let x = DVec3::new(0.3, -0.4, 0.5);
        assert!((inside.gradient.x + 4.0 / 3.0 * PI * g * rho * x.x).abs() < 1e-12);
        assert!((inside.hessian[0] + 4.0 / 3.0 * PI * g * rho).abs() < 1e-12);
        let p = DVec3::new(2.0, 1.0, -2.0);
        let outside = body.field(p, g, rho);
        let d = p.length();
        assert!((outside.potential - g * mass / d).abs() < 1e-12);
        assert!((outside.gradient.x + g * mass * p.x / d.powi(3)).abs() < 1e-12);
        let trace = outside.hessian[0] + outside.hessian[1] + outside.hessian[2];
        assert!(trace.abs() < 1e-12);
    }

    #[test]
    fn triaxial_field_is_continuous_across_the_surface_and_harmonic_outside() {
        let body = Ellipsoid::new(DVec3::ZERO, [3.0, 2.0, 1.0]);
        let n = DVec3::new(0.6, 0.64, 0.48);
        let on = DVec3::new(3.0 * n.x, 2.0 * n.y, n.z);
        let scale = 1.0 / (on.x * on.x / 9.0 + on.y * on.y / 4.0 + on.z * on.z).sqrt();
        let on = on * scale;
        let inner = body.field(on * (1.0 - 1e-9), 1.0, 1.0);
        let outer = body.field(on * (1.0 + 1e-9), 1.0, 1.0);
        assert!((inner.potential - outer.potential).abs() < 1e-7);
        assert!((inner.gradient - outer.gradient).length() < 1e-7);
        let far = body.field(DVec3::new(4.0, -3.0, 2.0), 1.0, 1.0);
        assert!((far.hessian[0] + far.hessian[1] + far.hessian[2]).abs() < 1e-12);
    }
}
