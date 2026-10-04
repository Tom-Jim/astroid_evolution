//! Shifted Jacobi radial basis `ψ_ℓk(ζ) = ζ^ℓ P_k^{(0, 2ℓ+2)}(2ζ−1)`, `k = 0..3`.

/// `P_k^{(α,β)}(x)` by the three-term recurrence.
pub fn jacobi(k: usize, alpha: f64, beta: f64, x: f64) -> f64 {
    if k == 0 {
        return 1.0;
    }
    let mut prev = 1.0;
    let mut curr = 0.5 * (alpha - beta + (alpha + beta + 2.0) * x);
    if k == 1 {
        return curr;
    }
    for n in 1..k {
        let nf = n as f64;
        let denom = 2.0 * (nf + 1.0) * (nf + alpha + beta + 1.0) * (2.0 * nf + alpha + beta);
        let mid = (2.0 * nf + alpha + beta + 1.0)
            * ((2.0 * nf + alpha + beta) * (2.0 * nf + alpha + beta + 2.0) * x + alpha * alpha
                - beta * beta);
        let last = 2.0 * (nf + alpha) * (nf + beta) * (2.0 * nf + alpha + beta + 2.0);
        let next = if denom.abs() < 1e-18 {
            0.0
        } else {
            (mid * curr - last * prev) / denom
        };
        prev = curr;
        curr = next;
    }
    curr
}

/// `ψ_ℓk` and `dψ_ℓk/dζ`.
pub fn psi_derivative(l: usize, k: usize, zeta: f64) -> (f64, f64) {
    let beta = 2.0 * l as f64 + 2.0;
    let x = 2.0 * zeta - 1.0;
    let poly = jacobi(k, 0.0, beta, x);
    let dpoly = if k == 0 {
        0.0
    } else {
        0.5 * (k as f64 + beta + 1.0) * jacobi(k - 1, 1.0, beta + 1.0, x)
    };
    let z_pow = if l == 0 { 1.0 } else { zeta.powi(l as i32) };
    let z_pow_m = if l == 0 { 0.0 } else { zeta.powi(l as i32 - 1) };
    let psi = z_pow * poly;
    let dpsi = l as f64 * z_pow_m * poly + z_pow * dpoly * 2.0;
    (psi, dpsi)
}

pub fn psi(l: usize, k: usize, zeta: f64) -> f64 {
    psi_derivative(l, k, zeta).0
}

/// `d²ψ_ℓk/dζ²`. At `ζ = 1` this is `κ_ℓk`.
pub fn psi_second(l: usize, k: usize, zeta: f64) -> f64 {
    let beta = 2.0 * l as f64 + 2.0;
    let x = 2.0 * zeta - 1.0;
    let poly = jacobi(k, 0.0, beta, x);
    let dpoly = if k == 0 {
        0.0
    } else {
        0.5 * (k as f64 + beta + 1.0) * jacobi(k - 1, 1.0, beta + 1.0, x)
    };
    let d2poly = if k < 2 {
        0.0
    } else {
        0.25 * (k as f64 + beta + 1.0) * (k as f64 + beta + 2.0) * jacobi(k - 2, 2.0, beta + 2.0, x)
    };
    let zm2 = if l < 2 { 0.0 } else { zeta.powi(l as i32 - 2) };
    let zm1 = if l == 0 { 0.0 } else { zeta.powi(l as i32 - 1) };
    let zp = if l == 0 { 1.0 } else { zeta.powi(l as i32) };
    let lf = l as f64;
    lf * (lf - 1.0) * zm2 * poly + 4.0 * lf * zm1 * dpoly + 4.0 * zp * d2poly
}

/// `σ_ℓk = ψ'_ℓk(1) = ℓ + k(k + 2ℓ + 3)`.
pub fn sigma(l: usize, k: usize) -> f64 {
    let lf = l as f64;
    let kf = k as f64;
    lf + kf * (kf + 2.0 * lf + 3.0)
}

/// `κ_ℓk = ψ''_ℓk(1)` from the closed table in §10.2.1.
pub fn kappa(l: usize, k: usize) -> f64 {
    let lf = l as f64;
    match k {
        0 => lf * (lf - 1.0),
        1 => lf * (5.0 * lf + 7.0),
        2 => (lf + 2.0) * (13.0 * lf + 15.0),
        3 => (lf + 2.0) * (25.0 * lf + 63.0),
        _ => 0.0,
    }
}

/// `∫_0^1 ψ_ℓj ψ_ℓk ζ² dζ = δ_jk / (2k + 2ℓ + 3)`.
pub fn weighted_orthogonality(l: usize, j: usize, k: usize) -> f64 {
    if j != k {
        0.0
    } else {
        1.0 / (2.0 * k as f64 + 2.0 * l as f64 + 3.0)
    }
}

pub fn n_tilde(l: usize) -> [[f64; 4]; 4] {
    let mut n = [[0.0; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            n[i][j] = integrate(|zeta| psi(l, i, zeta) * psi(l, j, zeta));
        }
    }
    n
}

pub fn p_tilde(l: usize) -> [[f64; 4]; 4] {
    let mut p = [[0.0; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            p[i][j] = integrate(|zeta| {
                let (pi, di) = psi_derivative(l, i, zeta);
                let (pj, dj) = psi_derivative(l, j, zeta);
                let di = pi + zeta * di;
                let dj = pj + zeta * dj;
                di * dj
            });
        }
    }
    p
}

/// Reference toroidal mass, diagonal in `k`.
pub fn toroidal_mass(l: usize, density: f64, radius: f64) -> [f64; 4] {
    let lambda = l as f64 * (l as f64 + 1.0);
    let scale = density * radius.powi(5) * lambda;
    std::array::from_fn(|k| scale * weighted_orthogonality(l, k, k))
}

/// Reference poloidal mass `ρ R^5 Λ (Λ Ñ + P̃)`.
pub fn poloidal_mass(l: usize, density: f64, radius: f64) -> [[f64; 4]; 4] {
    let lambda = l as f64 * (l as f64 + 1.0);
    let scale = density * radius.powi(5) * lambda;
    let n = n_tilde(l);
    let p = p_tilde(l);
    let mut m = [[0.0; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            m[i][j] = scale * (lambda * n[i][j] + p[i][j]);
        }
    }
    m
}

fn integrate(f: impl Fn(f64) -> f64) -> f64 {
    let n = 512;
    let h = 1.0 / n as f64;
    let mut acc = f(0.0) + f(1.0);
    for i in 1..n {
        let x = i as f64 * h;
        let weight = if i % 2 == 0 { 2.0 } else { 4.0 };
        acc += weight * f(x);
    }
    acc * h / 3.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoint_sigma_and_kappa_match_the_closed_forms() {
        for l in [1usize, 2, 3, 5] {
            for k in 0..4 {
                assert!((psi(l, k, 1.0) - 1.0).abs() < 1e-12, "ψ({l},{k},1)");
                let (value, slope) = psi_derivative(l, k, 1.0);
                assert!((value - 1.0).abs() < 1e-12);
                assert!((slope - sigma(l, k)).abs() < 1e-9, "σ({l},{k})");
                let eps = 1e-5;
                let second = (psi(l, k, 1.0 + eps) - 2.0 * psi(l, k, 1.0) + psi(l, k, 1.0 - eps))
                    / (eps * eps);
                assert!(
                    (second - kappa(l, k)).abs() < 1e-4 * (1.0 + kappa(l, k).abs()),
                    "κ({l},{k}) numeric {second} formula {}",
                    kappa(l, k)
                );
            }
        }
    }

    #[test]
    fn weighted_inner_product_is_diagonal() {
        for l in [1usize, 2, 4] {
            for j in 0..4 {
                for k in 0..4 {
                    let numeric = integrate(|zeta| psi(l, j, zeta) * psi(l, k, zeta) * zeta * zeta);
                    let exact = weighted_orthogonality(l, j, k);
                    assert!(
                        (numeric - exact).abs() < 1e-6,
                        "ℓ={l} ({j},{k}) {numeric} vs {exact}"
                    );
                }
            }
        }
    }

    #[test]
    fn ell_one_polynomials_and_reference_blocks() {
        let zeta = 0.5_f64;
        let expected = [
            zeta,
            6.0 * zeta * zeta - 5.0 * zeta,
            28.0 * zeta.powi(3) - 42.0 * zeta * zeta + 15.0 * zeta,
            120.0 * zeta.powi(4) - 252.0 * zeta.powi(3) + 168.0 * zeta * zeta - 35.0 * zeta,
        ];
        for k in 0..4 {
            assert!((psi(1, k, zeta) - expected[k]).abs() < 1e-12);
        }
        let n = n_tilde(1);
        let exact_n = [
            [1.0 / 3.0, -1.0 / 6.0, 0.1, -1.0 / 15.0],
            [-1.0 / 6.0, 8.0 / 15.0, -0.4, 61.0 / 210.0],
            [0.1, -0.4, 0.8, -0.7],
            [-1.0 / 15.0, 61.0 / 210.0, -0.7, 17.0 / 15.0],
        ];
        for i in 0..4 {
            for j in 0..4 {
                assert!((n[i][j] - exact_n[i][j]).abs() < 1e-8);
            }
        }
        let toroidal = toroidal_mass(1, 1.0, 1.0);
        let exact_t = [2.0 / 5.0, 2.0 / 7.0, 2.0 / 9.0, 2.0 / 11.0];
        for k in 0..4 {
            assert!((toroidal[k] - exact_t[k]).abs() < 1e-12);
        }
        let poloidal = poloidal_mass(1, 2.0, 1.0);
        let unit = poloidal_mass(1, 1.0, 1.0);
        for i in 0..4 {
            for j in 0..4 {
                assert!((poloidal[i][j] - 2.0 * unit[i][j]).abs() < 1e-8);
                assert!((unit[i][j] - unit[j][i]).abs() < 1e-12);
            }
        }
    }
}
