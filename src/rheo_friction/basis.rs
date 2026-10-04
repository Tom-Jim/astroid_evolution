//! Radial Jacobi basis traces (mathstrict.md §10.1, §10.2.1), real spherical
//! harmonics, and the host-side reduction of a velocity field to the mode sums
//! the GPU consumes.
//!
//! The real spherical harmonics are orthonormal on the unit sphere, carry no
//! Condon–Shortley phase, use `cos(m phi)` for `m > 0` and `sin(|m| phi)` for
//! `m < 0`. The flat index of `(l, m)` is `l^2 + l + m`.

use std::f64::consts::PI;

use super::error::RheoError;
use super::model::{StarSurface, VelocityField};

/// Radial truncation `K = 4` (§10).
pub const K: usize = 4;

/// Highest spherical-harmonic degree the GPU kernel is validated for.
pub const L_MAX_LIMIT: usize = 48;

#[inline]
pub fn lm_index(l: usize, m: i32) -> usize {
    ((l * l + l) as i64 + i64::from(m)) as usize
}

/// `sigma_{lk} = psi'_{lk}(1)`.
#[inline]
pub fn sigma(l: usize, k: usize) -> f64 {
    let (l, k) = (l as f64, k as f64);
    l + k * (k + 2.0 * l + 3.0)
}

/// `kappa_{lk} = psi''_{lk}(1)` (§10.2.1 table).
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

/// `c^P_{lk} = Lambda + kappa - 2` of the sphere constraint `D_rS = 0` (§10.2.4).
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

/// All real orthonormal spherical harmonics up to `l_max` at the unit vector `d`.
pub fn sh_all(l_max: usize, d: [f64; 3]) -> Vec<f64> {
    let mut out = vec![0.0; (l_max + 1) * (l_max + 1)];
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
        // pmm carries sqrt((2m+1)/(4 pi)) prod_{j<=m} (2j-1)/(2j) after the update above.
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
    out
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

/// `a_lm, b_lm` of the normalised Legendre recursion and `pmm`, as DS pairs.
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

/// Number of DS values per mode in the table.
pub const MODE_DS: usize = 9;

/// Per-mode sums that turn Proposition 10.2 into linear combinations of `Y_lm`:
/// `[r, -Lambda r, z1, z3, z4, z5, z6, z7, p0]` with
/// `z1 = sum (2-sigma) q`, `z3 = sum (2-2 sigma+kappa) q`, `z4 = sum (1+sigma) q`,
/// `z5 = sum (2-kappa) q`, `z6 = sum s`, `z7 = sum (1-sigma) s`, `p0 = Lambda sum q`.
/// Modes with `m != 0` carry the `sqrt 2` of the real harmonic, which the kernel omits.
pub fn mode_sums(
    surface: &StarSurface,
    velocity: &VelocityField,
    l_max: usize,
) -> Result<Vec<[f64; MODE_DS]>, RheoError> {
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
    let mut out = Vec::with_capacity(rows.len() * MODE_DS * 2);
    for row in rows {
        for v in row {
            push_ds(&mut out, *v);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigma_kappa_match_the_jacobi_polynomials() {
        for l in 1..=8 {
            for k in 0..K {
                let p = psi_poly(l, k);
                assert!(
                    (poly_eval(&p, 1.0) - 1.0).abs() < 1e-9,
                    "psi(1) l={l} k={k}"
                );
                let d1 = poly_derivative(&p);
                let d2 = poly_derivative(&d1);
                let s = poly_eval(&d1, 1.0);
                let kp = poly_eval(&d2, 1.0);
                assert!(
                    (s - sigma(l, k)).abs() < 1e-7 * s.abs().max(1.0),
                    "sigma l={l} k={k}"
                );
                assert!(
                    (kp - kappa(l, k)).abs() < 1e-7 * kp.abs().max(1.0),
                    "kappa l={l} k={k}"
                );
            }
        }
    }

    #[test]
    fn jacobi_basis_is_l2_orthogonal() {
        // int psi_j psi_k zeta^2 dzeta = delta / (2k + 2l + 3).
        let l = 3;
        for j in 0..K {
            for k in 0..K {
                let (pj, pk) = (psi_poly(l, j), psi_poly(l, k));
                // Exact: multiply the monomial coefficients and integrate term by term.
                let mut acc = 0.0;
                for (a, ca) in pj.iter().enumerate() {
                    for (b, cb) in pk.iter().enumerate() {
                        acc += ca * cb / (a + b + 3) as f64;
                    }
                }
                let want = if j == k {
                    1.0 / (2 * k + 2 * l + 3) as f64
                } else {
                    0.0
                };
                assert!((acc - want).abs() < 1e-9, "j={j} k={k} {acc}");
            }
        }
    }

    #[test]
    fn rigid_modes_have_no_surface_shear_constraint() {
        for l in 1..=1 {
            assert_eq!(constraint_poloidal(l, 0), 0.0);
            assert_eq!(constraint_toroidal(l, 0), 0.0);
        }
    }

    #[test]
    fn harmonics_are_orthonormal() {
        let l_max = 5;
        let n = 60000;
        let golden = PI * (3.0 - 5.0_f64.sqrt());
        let count = (l_max + 1) * (l_max + 1);
        let mut gram = vec![0.0; count * count];
        for i in 0..n {
            let z = 1.0 - 2.0 * (i as f64 + 0.5) / n as f64;
            let r = (1.0 - z * z).sqrt();
            let phi = golden * i as f64;
            let y = sh_all(l_max, [r * phi.cos(), r * phi.sin(), z]);
            for a in 0..count {
                for b in 0..count {
                    gram[a * count + b] += y[a] * y[b] * 4.0 * PI / n as f64;
                }
            }
        }
        for a in 0..count {
            for b in 0..count {
                let want = if a == b { 1.0 } else { 0.0 };
                assert!((gram[a * count + b] - want).abs() < 2e-3, "{a} {b}");
            }
        }
    }
}
