//! Radial integrals of one generalized-Cauchy kernel along one ray.
//!
//! Along `y = x − R u` the kernel is `A(R)^{-α}` with
//! `A(R) = (R − b)² + d²`, `b = (x − c)·u`, `d² = σ⁻² + |(x − c) − b u|²`.
//! Over a radial interval `[r0, r1]` the field needs
//!
//! * `I_U = ∫ R A^{-α} dR`,
//! * `I_g = ∫ A^{-α} dR`,
//! * `I_H = ∫ A^{-α} / R dR`, taken as a Hadamard finite part when `r0 = 0`.
//!
//! `α = 1` is closed form. Any other `α` uses `R − b = d sinh s`, which makes
//! `I_U` and `I_g` smooth `cosh^{1−2α}` integrals, and writes
//! `I_H = ∫ (A^{-α} − A(0)^{-α}) / R dR + A(0)^{-α} ln(r1/r0)`, whose integrand
//! is regular on the whole interval.

use gauss_quad::GaussLegendre;
use std::num::NonZeroUsize;

/// Radial integrals `(I_U, I_g, I_H)` of one kernel on one interval.
#[derive(Clone, Copy, Default, Debug)]
pub struct Integrals {
    pub u: f64,
    pub g: f64,
    pub h: f64,
}

/// Which integrals the caller needs; the logarithms are skipped when only the gradient is wanted.
#[derive(Clone, Copy, Debug)]
pub struct Need {
    pub potential: bool,
    pub hessian: bool,
}

/// Kernel geometry on one ray.
#[derive(Clone, Copy, Debug)]
pub struct RayKernel {
    pub b: f64,
    pub d: f64,
    /// `A(0) = b² + d²`.
    pub a0: f64,
}

/// Closed form for `α = 1`. `ln_ratio` is `ln(r1/r0)`, or `ln(r1/ℓ)` for the finite part.
pub fn alpha_one(k: RayKernel, r0: f64, r1: f64, ln_ratio: f64, need: Need) -> Integrals {
    let x0 = (r0 - k.b) / k.d;
    let x1 = (r1 - k.b) / k.d;
    // atan(x1) − atan(x0), accurate for short intervals.
    let dat = (x1 - x0).atan2(1.0 + x0 * x1);
    let bd = k.b / k.d;
    let mut out = Integrals { g: dat / k.d, ..Integrals::default() };
    if need.potential || need.hessian {
        let half_ln = 0.5 * ((1.0 + x1 * x1) / (1.0 + x0 * x0)).ln();
        out.u = half_ln + bd * dat;
        out.h = (ln_ratio - half_ln + bd * dat) / k.a0;
    }
    out
}

/// Gauss–Legendre integrator for `α ≠ 1`.
pub struct General {
    rule: GaussLegendre,
}

impl General {
    pub fn new(nodes: usize) -> Self {
        let n = NonZeroUsize::new(nodes.max(4)).expect("at least four nodes");
        Self { rule: GaussLegendre::new(n) }
    }

    pub fn integrals(&self, alpha: f64, k: RayKernel, r0: f64, r1: f64, ln_ratio: f64) -> Integrals {
        let s0 = ((r0 - k.b) / k.d).asinh();
        let s1 = ((r1 - k.b) / k.d).asinh();
        let panels = ((s1 - s0).ceil() as usize).max(1);
        let h = (s1 - s0) / panels as f64;
        let amp = k.d.powf(1.0 - 2.0 * alpha);
        let a0_pow = k.a0.powf(-alpha);
        let reg_scale = 1.0e-14 * (k.b.abs() + k.d);
        let (mut iu, mut ig, mut ih) = (0.0, 0.0, 0.0);
        for p in 0..panels {
            let lo = s0 + h * p as f64;
            for &(node, weight) in self.rule.as_node_weight_pairs() {
                let s = lo + 0.5 * h * (node + 1.0);
                let w = 0.5 * h * weight;
                let ch = s.cosh();
                let r = (k.b + k.d * s.sinh()).max(0.0);
                let g = amp * ch.powf(1.0 - 2.0 * alpha);
                iu += w * g * r;
                ig += w * g;
                let regular = if r > reg_scale {
                    a0_pow * (-alpha * (r * (r - 2.0 * k.b) / k.a0).ln_1p()).exp_m1() / r
                } else {
                    2.0 * alpha * k.b * a0_pow / k.a0
                };
                ih += w * regular * k.d * ch;
            }
        }
        Integrals { u: iu, g: ig, h: ih + a0_pow * ln_ratio }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const NEED: Need = Need { potential: true, hessian: true };

    /// Direct fine-grained Simpson reference for the three integrals.
    fn brute(alpha: f64, k: RayKernel, r0: f64, r1: f64) -> (f64, f64, f64) {
        let n = 400_000;
        let a_of = |r: f64| (r - k.b).powi(2) + k.d * k.d;
        let (mut iu, mut ig, mut ih) = (0.0, 0.0, 0.0);
        let h = (r1 - r0) / n as f64;
        for i in 0..n {
            let r = r0 + h * (i as f64 + 0.5);
            let a = a_of(r).powf(-alpha);
            iu += a * r * h;
            ig += a * h;
            ih += a / r * h;
        }
        (iu, ig, ih)
    }

    fn kernel(b: f64, d: f64) -> RayKernel {
        RayKernel { b, d, a0: b * b + d * d }
    }

    #[test]
    fn closed_form_matches_quadrature() {
        let k = kernel(0.7, 0.4);
        let (r0, r1) = (0.3, 2.0);
        let exact = alpha_one(k, r0, r1, (r1 / r0).ln(), NEED);
        let (iu, ig, ih) = brute(1.0, k, r0, r1);
        assert!((exact.u - iu).abs() < 1e-9);
        assert!((exact.g - ig).abs() < 1e-9);
        assert!((exact.h - ih).abs() < 1e-8);
    }

    #[test]
    fn general_rule_reproduces_alpha_one() {
        let rule = General::new(12);
        for &(b, d, r0, r1) in &[(0.7, 0.4, 0.3, 2.0), (-1.2, 0.05, 0.0, 3.0), (5.0, 2.0, 1e-4, 9.0)] {
            let k = kernel(b, d);
            let ln_ratio = if r0 > 0.0 { (r1 / r0).ln() } else { (r1 / 0.7_f64).ln() };
            let a = alpha_one(k, r0, r1, ln_ratio, NEED);
            let g = rule.integrals(1.0, k, r0, r1, ln_ratio);
            assert!((a.u - g.u).abs() < 1e-9 * (1.0 + a.u.abs()), "{a:?} {g:?}");
            assert!((a.g - g.g).abs() < 1e-9 * (1.0 + a.g.abs()), "{a:?} {g:?}");
            assert!((a.h - g.h).abs() < 1e-9 * (1.0 + a.h.abs()), "{a:?} {g:?}");
        }
    }

    #[test]
    fn general_rule_matches_quadrature_for_other_alpha() {
        let rule = General::new(12);
        let k = kernel(0.7, 0.4);
        for alpha in [0.5, 1.5, 2.5] {
            let (r0, r1) = (0.3, 2.0);
            let g = rule.integrals(alpha, k, r0, r1, (r1 / r0).ln());
            let (iu, ig, ih) = brute(alpha, k, r0, r1);
            assert!((g.u - iu).abs() < 1e-9, "alpha {alpha}");
            assert!((g.g - ig).abs() < 1e-9, "alpha {alpha}");
            assert!((g.h - ih).abs() < 1e-8, "alpha {alpha}");
        }
    }
}
