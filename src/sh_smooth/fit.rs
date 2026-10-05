//! Host weighted least squares of `R(ω̂)` onto real spherical harmonics.
//!
//! A Laplace–Beltrami ridge `ε (1 + l(l+1))` keeps the normal equations definite
//! when a high degree is weakly seen by the samples.

use super::error::ShError;
use super::harmonics::{check_l_max, coeff_count, lm_index, sh_all};

#[derive(Clone, Debug)]
pub struct Spectrum {
    pub l_max: usize,
    pub coeff: Vec<f64>,
}

pub fn fit(
    l_max: usize,
    dirs: &[[f64; 3]],
    radii: &[f64],
    weights: &[f64],
    ridge: f64,
) -> Result<Spectrum, ShError> {
    check_l_max(l_max)?;
    if !(ridge.is_finite() && ridge >= 0.0) {
        return Err(ShError::invalid_input("ridge must be a finite number >= 0"));
    }
    let n = coeff_count(l_max);
    if dirs.len() != radii.len() || dirs.len() != weights.len() {
        return Err(ShError::invalid_input(
            "direction, radius and weight arrays differ in length",
        ));
    }
    if dirs.len() < n {
        return Err(ShError::invalid_input(format!(
            "{} samples cannot determine {n} harmonic coefficients at l_max {l_max}",
            dirs.len()
        )));
    }
    let mut gram = vec![0.0; n * n];
    let mut rhs = vec![0.0; n];
    for ((dir, &radius), &weight) in dirs.iter().zip(radii).zip(weights) {
        if !(weight.is_finite() && weight >= 0.0 && radius.is_finite()) {
            return Err(ShError::invalid_input(
                "sample weights must be >= 0 and radii finite",
            ));
        }
        if weight == 0.0 {
            continue;
        }
        let y = sh_all(l_max, *dir);
        for a in 0..n {
            rhs[a] += weight * radius * y[a];
            let wa = weight * y[a];
            for b in 0..=a {
                gram[a * n + b] += wa * y[b];
            }
        }
    }
    for a in 0..n {
        for b in 0..a {
            gram[b * n + a] = gram[a * n + b];
        }
    }
    let trace: f64 = (0..n).map(|i| gram[i * n + i]).sum();
    if !(trace.is_finite() && trace > 0.0) {
        return Err(ShError::geometry(
            "spherical-harmonic samples have no positive weight",
        ));
    }
    let scale = trace / n as f64;
    if ridge > 0.0 {
        for l in 0..=l_max {
            let add = ridge * scale * (1.0 + (l * (l + 1)) as f64);
            for m in -(l as i32)..=(l as i32) {
                let i = lm_index(l, m);
                gram[i * n + i] += add;
            }
        }
    }
    let coeff = solve_spd(&gram, &rhs, n)?;
    if coeff.iter().any(|c| !c.is_finite()) {
        return Err(ShError::geometry(
            "spherical-harmonic fit produced a non-finite coefficient",
        ));
    }
    Ok(Spectrum { l_max, coeff })
}

fn solve_spd(gram: &[f64], rhs: &[f64], n: usize) -> Result<Vec<f64>, ShError> {
    let mut lower = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = gram[i * n + j];
            for k in 0..j {
                sum -= lower[i * n + k] * lower[j * n + k];
            }
            if i == j {
                if !(sum.is_finite() && sum > 0.0) {
                    return Err(ShError::geometry(
                        "spherical-harmonic normal equations are singular",
                    ));
                }
                lower[i * n + j] = sum.sqrt();
            } else {
                lower[i * n + j] = sum / lower[j * n + j];
            }
        }
    }
    let mut y = vec![0.0; n];
    for i in 0..n {
        let mut sum = rhs[i];
        for k in 0..i {
            sum -= lower[i * n + k] * y[k];
        }
        y[i] = sum / lower[i * n + i];
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = y[i];
        for k in (i + 1)..n {
            sum -= lower[k * n + i] * x[k];
        }
        x[i] = sum / lower[i * n + i];
    }
    Ok(x)
}
