//! Near-field finite part of the Hessian radial integral (MS §1.6).

use super::analytic_alpha1::geom_d2;

/// Boxed finite part from the singular endpoint `0` to `L`, at scale `ell0`.
#[inline]
pub fn i_h_finite_part(a0: f64, b: f64, d: f64, d2: f64, l: f64, ell0: f64) -> f64 {
    let dl = geom_d2(l, b, d2);
    let term_log = (l / ell0).ln() - 0.5 * (dl / a0).ln();
    let term_atan = (b / d) * (((l - b) / d).atan() + (b / d).atan());
    (term_log + term_atan) / a0
}
