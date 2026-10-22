//! Host reference evaluation of `R(ω̂) = Σ c_lm Y_lm(ω̂)` in f64.

use super::harmonics::{coeff_count, sh_all};

pub fn radius(l_max: usize, coeff: &[f64], dir: [f64; 3]) -> f64 {
    let y = sh_all(l_max, dir);
    let n = coeff_count(l_max);
    let mut sum = 0.0;
    for i in 0..n {
        sum += coeff[i] * y[i];
    }
    sum
}
