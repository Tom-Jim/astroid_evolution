//! `α = 1` antiderivatives of MS §3.2, including the zero-endpoint finite part of MS §1.6.
//!
//! `D(R) = (R − b)² + d²` is the geometric quadratic. The density factor `w σ^{−2α}`
//! is applied by the caller. `A0 = b² + d² = ‖x − c‖² + σ^{−2}` does not depend on direction.

#[inline]
pub fn geom_d2(r: f64, b: f64, d2: f64) -> f64 {
    let diff = r - b;
    diff * diff + d2
}

/// `P(R) = ½ log D + (b/d) arctan((R−b)/d)`, with the removable value at `R = 0`.
#[inline]
pub fn p_of(r: f64, b: f64, d: f64, d2: f64) -> f64 {
    if r == 0.0 {
        let a0 = b * b + d2;
        0.5 * a0.ln() - (b / d) * (b / d).atan()
    } else {
        0.5 * geom_d2(r, b, d2).ln() + (b / d) * ((r - b) / d).atan()
    }
}

#[inline]
pub fn q_of(r: f64, b: f64, d: f64) -> f64 {
    if r == 0.0 {
        -(1.0 / d) * (b / d).atan()
    } else {
        (1.0 / d) * ((r - b) / d).atan()
    }
}

#[inline]
pub fn l_of(r: f64, a0: f64, b: f64, d: f64, d2: f64) -> f64 {
    let term = r.ln() - 0.5 * geom_d2(r, b, d2).ln() + (b / d) * ((r - b) / d).atan();
    term / a0
}

#[inline]
pub fn i_u(b: f64, d: f64, d2: f64, r0: f64, r1: f64) -> f64 {
    p_of(r1, b, d, d2) - p_of(r0, b, d, d2)
}

#[inline]
pub fn i_g(b: f64, d: f64, r0: f64, r1: f64) -> f64 {
    q_of(r1, b, d) - q_of(r0, b, d)
}

#[inline]
pub fn i_h(a0: f64, b: f64, d: f64, d2: f64, r0: f64, r1: f64) -> f64 {
    l_of(r1, a0, b, d, d2) - l_of(r0, a0, b, d, d2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alpha_one_matches_gauss_legendre_to_1e_12() {
        let b = 0.3;
        let d = 1.7;
        let d2 = d * d;
        let a0 = b * b + d2;
        let r0 = 0.4;
        let r1 = 2.5;
        let (nodes, weights) = crate::rtfp_solver::radial::gauss_legendre::gauss_legendre(32);
        let h = 0.5 * (r1 - r0);
        let mid = 0.5 * (r1 + r0);
        let mut iu = 0.0;
        let mut ig = 0.0;
        let mut ih = 0.0;
        for i in 0..nodes.len() {
            let r = mid + h * nodes[i];
            let den = geom_d2(r, b, d2);
            iu += weights[i] * r / den;
            ig += weights[i] / den;
            ih += weights[i] / (r * den);
        }
        assert!((i_u(b, d, d2, r0, r1) - iu * h).abs() < 1e-12);
        assert!((i_g(b, d, r0, r1) - ig * h).abs() < 1e-12);
        assert!((i_h(a0, b, d, d2, r0, r1) - ih * h).abs() < 1e-12);
    }

    #[test]
    fn zero_endpoint_matches_the_limit_definition() {
        let b = -0.4;
        let d = 0.8;
        let d2 = d * d;
        let a0 = b * b + d2;
        let l = 1.25;
        let ell0 = 0.2;
        let eps = 1e-9;
        let raw = i_h(a0, b, d, d2, eps, l);
        let removed = raw - (1.0 / a0) * (ell0 / eps).ln();
        let closed = crate::rtfp_solver::radial::finite_part::i_h_finite_part(a0, b, d, d2, l, ell0);
        assert!(
            (removed - closed).abs() < 1e-8,
            "removed {removed} closed {closed}"
        );
    }
}

