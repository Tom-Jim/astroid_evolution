//! Real orthonormal spherical harmonics.
//!
//! No Condon–Shortley phase. `m > 0` carries `cos(m phi)`, `m < 0` carries
//! `sin(|m| phi)`. The flat index of `(l, m)` is `l^2 + l + m`.

use std::f64::consts::PI;

use super::error::ShError;

/// Highest degree the device tables are built for. Matches the constant-buffer budget.
pub const L_MAX_LIMIT: usize = 48;

#[inline]
pub fn lm_index(l: usize, m: i32) -> usize {
    ((l * l + l) as i64 + i64::from(m)) as usize
}

pub fn coeff_count(l_max: usize) -> usize {
    (l_max + 1) * (l_max + 1)
}

pub fn check_l_max(l_max: usize) -> Result<(), ShError> {
    if l_max > L_MAX_LIMIT {
        return Err(ShError::invalid_input(format!(
            "l_max {l_max} exceeds the device limit {L_MAX_LIMIT}"
        )));
    }
    Ok(())
}

/// All real orthonormal harmonics up to `l_max` at the unit vector `d`.
pub fn sh_all(l_max: usize, d: [f64; 3]) -> Vec<f64> {
    let mut out = vec![0.0; coeff_count(l_max)];
    sh_all_into(l_max, d, &mut out);
    out
}

/// `sh_all` into a caller-owned buffer. `out` must hold `(l_max + 1)²` values.
pub fn sh_all_into(l_max: usize, d: [f64; 3], out: &mut [f64]) {
    let n = coeff_count(l_max);
    assert!(out.len() >= n, "spherical-harmonic buffer is short");
    let (x, y, z) = (d[0], d[1], d[2]);
    let (mut cre, mut cim) = (1.0_f64, 0.0_f64);
    let mut pmm = (1.0 / (4.0 * PI)).sqrt();
    for m in 0..=l_max {
        if m > 0 {
            let (nre, nim) = (cre * x - cim * y, cre * y + cim * x);
            cre = nre;
            cim = nim;
            pmm *= ((2 * m + 1) as f64 / (2 * m) as f64).sqrt();
        }
        let mut p2 = 0.0;
        let mut p1 = 0.0;
        for l in m..=l_max {
            let p = if l == m {
                pmm
            } else {
                let lf = l as f64;
                let mf = m as f64;
                let a = ((4.0 * lf * lf - 1.0) / (lf * lf - mf * mf)).sqrt();
                let b = (((lf - 1.0) * (lf - 1.0) - mf * mf)
                    / (4.0 * (lf - 1.0) * (lf - 1.0) - 1.0))
                    .sqrt();
                if l == m + 1 {
                    a * z * p1
                } else {
                    a * (z * p1 - b * p2)
                }
            };
            if m == 0 {
                out[lm_index(l, 0)] = p;
            } else {
                let s2 = std::f64::consts::SQRT_2;
                out[lm_index(l, m as i32)] = s2 * p * cre;
                out[lm_index(l, -(m as i32))] = s2 * p * cim;
            }
            p2 = p1;
            p1 = p;
        }
    }
}

/// Power `Σ_m c_lm²` of each degree.
pub fn degree_power(l_max: usize, coeff: &[f64]) -> Vec<f64> {
    let mut power = vec![0.0; l_max + 1];
    for l in 0..=l_max {
        for m in -(l as i32)..=(l as i32) {
            let c = coeff[lm_index(l, m)];
            power[l] += c * c;
        }
    }
    power
}

/// `[hi, lo]` with `hi = f32(x)`, `lo = f32(x - hi)`.
#[inline]
pub fn ds_pair(x: f64) -> [f32; 2] {
    let hi = x as f32;
    [hi, (x - f64::from(hi)) as f32]
}

fn push_ds(out: &mut Vec<f32>, x: f64) {
    out.extend_from_slice(&ds_pair(x));
}

/// `a_lm`, `b_lm` of the normalised Legendre recursion, then `pmm`, as DS pairs.
/// Pair `(l, m)` lives at float offset `(l(l+1)/2 + m) * 4`.
/// `pmm[m]` starts at `(L+1)(L+2)/2 * 4`.
pub fn recurrence_table(l_max: usize) -> Vec<f32> {
    let mut out = Vec::new();
    for l in 0..=l_max {
        for m in 0..=l {
            let (lf, mf) = (l as f64, m as f64);
            let a = if l > m {
                ((4.0 * lf * lf - 1.0) / (lf * lf - mf * mf)).sqrt()
            } else {
                0.0
            };
            let b = if l >= m + 2 {
                (((lf - 1.0) * (lf - 1.0) - mf * mf) / (4.0 * (lf - 1.0) * (lf - 1.0) - 1.0)).sqrt()
            } else {
                0.0
            };
            push_ds(&mut out, a);
            push_ds(&mut out, b);
        }
    }
    let mut pmm = (1.0 / (4.0 * PI)).sqrt();
    for m in 0..=l_max {
        if m > 0 {
            pmm *= ((2 * m + 1) as f64 / (2 * m) as f64).sqrt();
        }
        push_ds(&mut out, pmm);
    }
    out
}

/// Coefficients consumed by the kernel. `m ≠ 0` is premultiplied by `√2`,
/// which the device recurrence does not apply itself.
pub fn pack_coefficients(l_max: usize, coeff: &[f64]) -> Vec<f32> {
    let mut out = Vec::with_capacity(coeff_count(l_max) * 2);
    for l in 0..=l_max {
        for m in -(l as i32)..=(l as i32) {
            let mut c = coeff[lm_index(l, m)];
            if m != 0 {
                c *= std::f64::consts::SQRT_2;
            }
            push_ds(&mut out, c);
        }
    }
    out
}

pub fn pack_scalars(values: &[f64]) -> Vec<f32> {
    let mut out = Vec::with_capacity(values.len() * 2);
    for v in values {
        push_ds(&mut out, *v);
    }
    out
}

pub fn pack_directions(dirs: &[[f64; 3]]) -> Vec<f32> {
    let mut out = Vec::with_capacity(dirs.len() * 6);
    for d in dirs {
        push_ds(&mut out, d[0]);
        push_ds(&mut out, d[1]);
        push_ds(&mut out, d[2]);
    }
    out
}

/// Fibonacci lattice on the unit sphere.
pub fn fibonacci_directions(n: usize) -> Vec<[f64; 3]> {
    let golden = PI * (3.0 - 5.0_f64.sqrt());
    (0..n)
        .map(|i| {
            let z = 1.0 - 2.0 * (i as f64 + 0.5) / n as f64;
            let rho = (1.0 - z * z).max(0.0).sqrt();
            let phi = golden * i as f64;
            [rho * phi.cos(), rho * phi.sin(), z]
        })
        .collect()
}
