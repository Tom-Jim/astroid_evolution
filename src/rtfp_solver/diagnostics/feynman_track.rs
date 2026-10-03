//! Independent Newtonian samples used to cross-check the ray reduction.
//!
//! The Feynman parameter integral below is the kernel identity of MS §1.7 evaluated
//! at a single source point. It is not added to the ray field.

use crate::common::density::CauchyKernel;
use crate::common::math::DVec3;

pub fn feynman_kernel_potential(x: DVec3, y: DVec3, kernel: CauchyKernel) -> f64 {
    let a = (x - y).length_squared();
    let b = (y - kernel.center).length_squared() + 1.0 / (kernel.sigma * kernel.sigma);
    let alpha = kernel.alpha;
    let p = alpha + 0.5;
    let pref = gamma(p) / (std::f64::consts::PI.sqrt() * gamma(alpha))
        * kernel.sigma.powf(-2.0 * alpha)
        * kernel.weight;
    let (nodes, weights) = crate::rtfp_solver::radial::gauss_legendre::gauss_legendre(96);
    let mut acc = 0.0;
    for (t, w) in nodes.iter().zip(weights) {
        // s ∈ (0,1), u = s² is the complement parameter of MS §1.7.
        let s = 0.5 * (t + 1.0);
        let u = s * s;
        let one = (1.0 - u).max(0.0);
        let q = u * a + one * b;
        acc += w * 0.5 * 2.0 * one.powf(alpha - 1.0) * q.powf(-p);
    }
    pref * acc
}

fn gamma(z: f64) -> f64 {
    if z < 0.5 {
        return std::f64::consts::PI / ((std::f64::consts::PI * z).sin() * gamma(1.0 - z));
    }
    const C: [f64; 9] = [
        0.9999999999998099,
        676.5203681218851,
        -1259.1392167224028,
        771.3234287776531,
        -176.6150291621406,
        12.507343278686905,
        -0.13857109526572012,
        9.984369578019572e-6,
        1.5056327351493116e-7,
    ];
    let z = z - 1.0;
    let mut x = C[0];
    for (i, c) in C.iter().enumerate().skip(1) {
        x += c / (z + i as f64);
    }
    let t = z + 7.5;
    (2.0 * std::f64::consts::PI).sqrt() * t.powf(z + 0.5) * (-t).exp() * x
}

/// Direct `w (1+σ²‖y−c‖²)^{−α} / ‖x−y‖` versus the Feynman parameter integral.
pub fn kernel_matches_newton(x: DVec3, y: DVec3, kernel: CauchyKernel) -> (f64, f64) {
    let newton = kernel.weight
        * (1.0 + kernel.sigma * kernel.sigma * (y - kernel.center).length_squared())
            .powf(-kernel.alpha)
        / (x - y).length();
    (newton, feynman_kernel_potential(x, y, kernel))
}
