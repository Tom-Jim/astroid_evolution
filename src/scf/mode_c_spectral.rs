//! Degree-32 zonal convolution for the mode C deposition kernel.
//!
//! The configured weight is the cutoff Gaussian of the angular separation.
//! Its Legendre truncation `G_L` is applied by the addition theorem, so each
//! source is one spherical-harmonic analysis and one synthesis rather than a
//! walk of the cap. The same `G_L` normalises the source, which keeps
//! `∑ J_dep,i area_i = ∑ J_ero,i area_i`. The truncation is used only when its
//! relative L1 against the cutoff Gaussian is at most `1e-3`; a narrower
//! kernel falls back to the cap walk. Negative lobes of `G_L`, and any source
//! whose synthesised norm is not positive, also fall back.

use std::sync::Mutex;

use rayon::prelude::*;

use crate::common::math::DVec3;
use crate::shsmooth::harmonics::{coeff_count, sh_all_into};

use super::quad::gauss_legendre;

pub const L_MAX: usize = 32;
/// Relative L1 of `G_L` against the cutoff Gaussian, on the Legendre measure.
const REL_L1: f64 = 1.0e-3;

struct Cache {
    bits: u64,
    beta: Option<Vec<f64>>,
}

static CACHE: Mutex<Option<Cache>> = Mutex::new(None);

/// `β_l = 2π ∫ G P_l dμ` when the truncation is inside [`REL_L1`].
pub fn spectrum(width: f64) -> Option<Vec<f64>> {
    if !(width.is_finite() && width > 0.0) {
        return None;
    }
    let bits = width.to_bits();
    let mut guard = CACHE.lock().unwrap_or_else(|err| err.into_inner());
    if let Some(hit) = guard.as_ref() {
        if hit.bits == bits {
            return hit.beta.clone();
        }
    }
    let beta = truncate(width);
    *guard = Some(Cache {
        bits,
        beta: beta.clone(),
    });
    beta
}

fn truncate(width: f64) -> Option<Vec<f64>> {
    let (mu, wgt) = gauss_legendre(512);
    let n = mu.len();
    let cutoff = (4.0 * width).min(std::f64::consts::PI);
    let mut g = vec![0.0; n];
    for i in 0..n {
        let ang = mu[i].clamp(-1.0, 1.0).acos();
        if ang < cutoff {
            g[i] = (-0.5 * (ang / width).powi(2)).exp();
        }
    }
    let mut legendre = Vec::with_capacity(L_MAX + 1);
    legendre.push(vec![1.0; n]);
    if L_MAX >= 1 {
        legendre.push(mu.clone());
    }
    for l in 1..L_MAX {
        let lf = l as f64;
        let prev = &legendre[l - 1];
        let cur = &legendre[l];
        let next: Vec<f64> = (0..n)
            .map(|i| ((2.0 * lf + 1.0) * mu[i] * cur[i] - lf * prev[i]) / (lf + 1.0))
            .collect();
        legendre.push(next);
    }
    let mut beta = vec![0.0; L_MAX + 1];
    let mut alpha = vec![0.0; L_MAX + 1];
    for l in 0..=L_MAX {
        let integral: f64 = g
            .iter()
            .zip(&legendre[l])
            .zip(&wgt)
            .map(|((gk, pk), wk)| gk * pk * wk)
            .sum();
        beta[l] = 2.0 * std::f64::consts::PI * integral;
        alpha[l] = (2 * l + 1) as f64 / 2.0 * integral;
    }
    let mut num = 0.0;
    let mut den = 0.0;
    for i in 0..n {
        let mut approx = 0.0;
        for l in 0..=L_MAX {
            approx += alpha[l] * legendre[l][i];
        }
        num += (approx - g[i]).abs() * wgt[i];
        den += g[i].abs() * wgt[i];
    }
    if den == 0.0 || num / den > REL_L1 {
        return None;
    }
    Some(beta)
}

/// Deposition through `G_L`. `None` means a source norm left the positive cone.
pub fn deposit(dirs: &[DVec3], area: &[f64], j_ero: &[f64], beta: &[f64]) -> Option<Vec<f64>> {
    let n = dirs.len();
    if area.len() != n || j_ero.len() != n || beta.len() != L_MAX + 1 {
        return None;
    }
    let mass: Vec<f64> = j_ero
        .iter()
        .zip(area)
        .map(|(j, a)| if *j > 0.0 { j * a } else { 0.0 })
        .collect();
    if mass.iter().all(|m| *m == 0.0) {
        return Some(vec![0.0; n]);
    }
    let l_max = beta.len() - 1;
    let area_hat = analyse(dirs, area, l_max);
    let norm = synthesise(dirs, &area_hat, beta);
    let mut scaled = vec![0.0; n];
    for i in 0..n {
        if mass[i] > 0.0 {
            if !(norm[i].is_finite() && norm[i] > 0.0) {
                return None;
            }
            scaled[i] = mass[i] / norm[i];
        }
    }
    let scaled_hat = analyse(dirs, &scaled, l_max);
    let j_dep = synthesise(dirs, &scaled_hat, beta);
    if j_dep.iter().any(|v| !v.is_finite()) {
        None
    } else {
        Some(j_dep)
    }
}

fn analyse(dirs: &[DVec3], weight: &[f64], l_max: usize) -> Vec<f64> {
    let ncoef = coeff_count(l_max);
    dirs.par_iter()
        .zip(weight.par_iter())
        .fold(
            || (vec![0.0; ncoef], vec![0.0; ncoef]),
            |mut state, (dir, &weight)| {
                if weight != 0.0 {
                    sh_all_into(l_max, dir.to_array(), &mut state.1);
                    for (slot, &harmonic) in state.0.iter_mut().zip(state.1.iter()) {
                        *slot += harmonic * weight;
                    }
                }
                state
            },
        )
        .reduce(
            || (vec![0.0; ncoef], Vec::new()),
            |mut left, right| {
                for (slot, value) in left.0.iter_mut().zip(right.0) {
                    *slot += value;
                }
                left
            },
        )
        .0
}

fn synthesise(dirs: &[DVec3], coeff: &[f64], beta: &[f64]) -> Vec<f64> {
    let l_max = beta.len() - 1;
    let ncoef = coeff_count(l_max);
    dirs.par_iter()
        .map(|dir| {
            let mut y = vec![0.0; ncoef];
            sh_all_into(l_max, dir.to_array(), &mut y);
            let mut sum = 0.0;
            let mut offset = 0usize;
            for l in 0..=l_max {
                let count = 2 * l + 1;
                let mut band = 0.0;
                for k in 0..count {
                    band += y[offset + k] * coeff[offset + k];
                }
                sum += beta[l] * band;
                offset += count;
            }
            sum
        })
        .collect()
}
