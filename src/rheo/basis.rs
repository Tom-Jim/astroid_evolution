//! Radial Jacobi basis traces (mathstrict_EN.md §8.1.3, §10.1), real spherical
//! harmonics, and the host-side reduction of a velocity field to the mode sums
//! the GPU consumes.
//!
//! The real spherical harmonics are orthonormal on the unit sphere, carry no
//! Condon–Shortley phase, use `cos(m phi)` for `m > 0` and `sin(|m| phi)` for
//! `m < 0`. The flat index of `(l, m)` is `l^2 + l + m`.

use super::model::{StarSurface, VelocityField};
use crate::common::error::SimError;

/// Radial truncation `K = 4` (§10).
pub const K: usize = 4;

pub use crate::shsmooth::harmonics::{
    L_MAX_LIMIT, ds_pair, lm_index, pack_scalars, recurrence_table, sh_all,
};

/// `sigma_{lk} = psi'_{lk}(1)`.
#[inline]
pub fn sigma(l: usize, k: usize) -> f64 {
    let (l, k) = (l as f64, k as f64);
    l + k * (k + 2.0 * l + 3.0)
}

/// `kappa_{lk} = psi''_{lk}(1)` (§10.1 table).
pub fn kappa(l: usize, k: usize) -> f64 {
    let l = l as f64;
    match k {
        0 => l * (l - 1.0),
        1 => l * (5.0 * l + 7.0),
        2 => (l + 2.0) * (13.0 * l + 15.0),
        3 => (l + 2.0) * (25.0 * l + 63.0),
        _ => f64::NAN,
    }
}

#[inline]
pub fn lambda(l: usize) -> f64 {
    (l * (l + 1)) as f64
}

/// `c^P_{lk} = Lambda + kappa - 2` of the sphere constraint `D_rS = 0` (§10.1).
pub fn constraint_poloidal(l: usize, k: usize) -> f64 {
    lambda(l) + kappa(l, k) - 2.0
}

/// `c^T_{lk} = sigma - 1`.
pub fn constraint_toroidal(l: usize, k: usize) -> f64 {
    sigma(l, k) - 1.0
}

/// Monomial coefficients (ascending in zeta) of `psi_{lk}(zeta) = zeta^l P_k^{(0, 2l+2)}(2 zeta - 1)`.
pub fn psi_poly(l: usize, k: usize) -> Vec<f64> {
    let beta = 2 * l + 2;
    let binom = |n: usize, r: usize| -> f64 {
        if r > n {
            return 0.0;
        }
        let mut v = 1.0;
        for i in 0..r {
            v = v * (n - i) as f64 / (i + 1) as f64;
        }
        v
    };
    let mut total = vec![0.0; l + k + 1];
    for s in 0..=k {
        // C(k, k - s) C(k + beta, s) (zeta - 1)^s zeta^(k - s)
        let weight = binom(k, k - s) * binom(k + beta, s);
        let mut poly = vec![1.0];
        for _ in 0..s {
            let mut next = vec![0.0; poly.len() + 1];
            for (i, c) in poly.iter().enumerate() {
                next[i] -= c;
                next[i + 1] += c;
            }
            poly = next;
        }
        for (i, c) in poly.iter().enumerate() {
            total[l + k - s + i] += weight * c;
        }
    }
    total
}

pub fn poly_eval(p: &[f64], x: f64) -> f64 {
    p.iter().rev().fold(0.0, |acc, c| acc * x + c)
}

pub fn poly_derivative(p: &[f64]) -> Vec<f64> {
    p.iter()
        .enumerate()
        .skip(1)
        .map(|(i, c)| c * i as f64)
        .collect()
}

/// Directions as double-single triples, the layout `pack_dirs` in the GPU callers uses.
pub fn pack_dirs(vs: &[crate::common::math::DVec3]) -> Vec<f32> {
    use rayon::prelude::*;
    let mut out = vec![0.0f32; vs.len() * 6];
    out.par_chunks_mut(6)
        .zip(vs.par_iter())
        .for_each(|(slot, v)| {
            let x = ds_pair(v.x);
            let y = ds_pair(v.y);
            let z = ds_pair(v.z);
            slot.copy_from_slice(&[x[0], x[1], y[0], y[1], z[0], z[1]]);
        });
    out
}

/// Number of DS values per mode in the table.
pub const MODE_DS: usize = 9;

/// Per-mode sums that turn Section 10.1 into linear combinations of `Y_lm`:
/// `[r, -Lambda r, z1, z3, z4, z5, z6, z7, p0]` with
/// `z1 = sum (2-sigma) q`, `z3 = sum (2-2 sigma+kappa) q`, `z4 = sum (1+sigma) q`,
/// `z5 = sum (2-kappa) q`, `z6 = sum s`, `z7 = sum (1-sigma) s`, `p0 = Lambda sum q`.
/// Modes with `m != 0` carry the `sqrt 2` of the real harmonic, which the kernel omits.
pub fn mode_sums(
    surface: &StarSurface,
    velocity: &VelocityField,
    l_max: usize,
) -> Result<Vec<[f64; MODE_DS]>, SimError> {
    let count = (l_max + 1) * (l_max + 1);
    let mut rows = vec![[0.0; MODE_DS]; count];
    for l in 0..=l_max {
        for m in -(l as i32)..=(l as i32) {
            let idx = lm_index(l, m);
            let fold = if m == 0 {
                1.0
            } else {
                std::f64::consts::SQRT_2
            };
            let lam = lambda(l);
            let r = if idx < surface.coeff.len() {
                surface.coeff[idx]
            } else {
                0.0
            };
            let (mut a, mut ss, mut sk, mut st, mut sts) = (0.0, 0.0, 0.0, 0.0, 0.0);
            if l >= 1 && idx < velocity.q.len() {
                for k in 0..K {
                    let q = velocity.q[idx][k];
                    let s = velocity.s[idx][k];
                    a += q;
                    ss += q * sigma(l, k);
                    sk += q * kappa(l, k);
                    st += s;
                    sts += s * sigma(l, k);
                }
            }
            rows[idx] = [
                r * fold,
                -lam * r * fold,
                (2.0 * a - ss) * fold,
                (2.0 * a - 2.0 * ss + sk) * fold,
                (a + ss) * fold,
                (2.0 * a - sk) * fold,
                st * fold,
                (st - sts) * fold,
                lam * a * fold,
            ];
        }
    }
    Ok(rows)
}

pub fn mode_table(rows: &[[f64; MODE_DS]]) -> Vec<f32> {
    pack_scalars(rows.as_flattened())
}
