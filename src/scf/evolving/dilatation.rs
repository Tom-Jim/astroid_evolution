//! The dilatational velocity of A-Ξ (§11.4): `v_ε = ∇χ_h` with `Δχ_h ≈ ε̇_v`.
//!
//! `χ_h` lives in the polynomials of degree at most `d` without the constant, in the
//! scaled coordinates `x / R_*`. It is the minimizer of
//! `½‖Δχ − ε̇_v‖² + ½ ε_χ ‖∇χ‖²` on the volume rule; the normal matrix depends on the
//! geometry only and is factored once per geometry. All derivatives are analytic.

use nalgebra::{Cholesky, DMatrix, DVector, Dyn};

use crate::common::error::SimError;
use crate::common::math::DVec3;

/// Exponents `(a, b, c)` with `1 ≤ a + b + c ≤ d`.
fn exponents(d: usize) -> Vec<[u8; 3]> {
    let mut out = Vec::new();
    for total in 1..=d {
        for a in (0..=total).rev() {
            for b in (0..=total - a).rev() {
                out.push([a as u8, b as u8, (total - a - b) as u8]);
            }
        }
    }
    out
}

/// `∂^{(i,j,k)} (ξ_x^a ξ_y^b ξ_z^c)` at `ξ`, without the `1/s^{i+j+k}` factor.
fn derivative(e: [u8; 3], o: [u8; 3], xi: [f64; 3]) -> f64 {
    let mut v = 1.0;
    for k in 0..3 {
        let (a, n) = (e[k] as i32, o[k] as i32);
        if n > a {
            return 0.0;
        }
        for m in 0..n {
            v *= f64::from(a - m);
        }
        v *= xi[k].powi(a - n);
    }
    v
}

const ORDER1: [[u8; 3]; 3] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
/// `(xx, yy, zz, xy, xz, yz)`.
const ORDER2: [[u8; 3]; 6] = [[2, 0, 0], [0, 2, 0], [0, 0, 2], [1, 1, 0], [1, 0, 1], [0, 1, 1]];

/// A solved dilatational potential.
#[derive(Clone, Debug)]
pub struct Dilatational {
    exps: Vec<[u8; 3]>,
    coef: Vec<f64>,
    scale: f64,
}

impl Dilatational {
    /// `∂^o χ` at `x`, with `χ = R_*² Σ a_i q_i(x / R_*)`.
    fn d(&self, x: DVec3, o: [u8; 3]) -> f64 {
        let xi = (x / self.scale).to_array();
        let n = i32::from(o[0] + o[1] + o[2]);
        self.exps.iter().zip(&self.coef).map(|(e, c)| c * derivative(*e, o, xi)).sum::<f64>()
            * self.scale.powi(2 - n)
    }

    /// `v_ε = ∇χ`.
    pub fn velocity(&self, x: DVec3) -> DVec3 {
        DVec3::new(self.d(x, ORDER1[0]), self.d(x, ORDER1[1]), self.d(x, ORDER1[2]))
    }

    /// `D(v_ε) = ∇∇χ` as the scaled six-vector `(xx, yy, zz, √2 xy, √2 xz, √2 yz)`.
    pub fn strain(&self, x: DVec3) -> [f64; 6] {
        let h: [f64; 6] = std::array::from_fn(|k| self.d(x, ORDER2[k]));
        let s = std::f64::consts::SQRT_2;
        [h[0], h[1], h[2], s * h[3], s * h[4], s * h[5]]
    }

    /// `ε̇_{v,h} = Δχ`.
    pub fn divergence(&self, x: DVec3) -> f64 {
        self.d(x, ORDER2[0]) + self.d(x, ORDER2[1]) + self.d(x, ORDER2[2])
    }

    /// `Δ v_ε = ∇Δχ`.
    pub fn laplacian(&self, x: DVec3) -> DVec3 {
        let t = |a: [u8; 3]| -> f64 {
            (0..3).map(|k| {
                let mut o = a;
                o[k] += 2;
                self.d(x, o)
            })
            .sum()
        };
        DVec3::new(t(ORDER1[0]), t(ORDER1[1]), t(ORDER1[2]))
    }

    /// `∂_k D(v_ε)` for `k = x, y, z`, each a scaled six-vector.
    pub fn strain_gradient(&self, x: DVec3) -> [f64; 18] {
        let s = std::f64::consts::SQRT_2;
        let mut out = [0.0; 18];
        for k in 0..3 {
            for (c, o2) in ORDER2.iter().enumerate() {
                let mut o = *o2;
                o[k] += 1;
                out[6 * k + c] = self.d(x, o) * if c >= 3 { s } else { 1.0 };
            }
        }
        out
    }
}

/// Deviator of a scaled strain six-vector.
pub fn deviator(d: [f64; 6]) -> [f64; 6] {
    let m = (d[0] + d[1] + d[2]) / 3.0;
    [d[0] - m, d[1] - m, d[2] - m, d[3], d[4], d[5]]
}

/// The space and its factored normal matrix on one volume rule.
pub struct DilatationSpace {
    exps: Vec<[u8; 3]>,
    scale: f64,
    /// `Δq_a` at every volume node, node-major.
    lap: Vec<f64>,
    chol: Cholesky<f64, Dyn>,
}

impl DilatationSpace {
    pub fn build(pts: &[DVec3], w: &[f64], degree: usize, regularization: f64, r_star: f64) -> Result<Self, SimError> {
        let exps = exponents(degree);
        let n = exps.len();
        let s = r_star;
        // With χ = R_*² Σ a_i q_i(ξ), ξ = x / R_*: Δχ = Σ a_i Δ_ξ q_i and
        // ε_χ ‖∇χ‖² = ε̂_χ ‖Σ a_i ∇_ξ q_i‖², so the normal matrix is O(1).
        let mut lap = vec![0.0; pts.len() * n];
        let mut a = DMatrix::<f64>::zeros(n, n);
        let mut grad = vec![0.0; 3 * n];
        for (q, (x, wq)) in pts.iter().zip(w).enumerate() {
            let xi = (*x / s).to_array();
            let row = &mut lap[q * n..(q + 1) * n];
            for (i, e) in exps.iter().enumerate() {
                row[i] = (0..3).map(|k| derivative(*e, ORDER2[k], xi)).sum();
                for k in 0..3 {
                    grad[3 * i + k] = derivative(*e, ORDER1[k], xi);
                }
            }
            for i in 0..n {
                for j in i..n {
                    let g: f64 = (0..3).map(|k| grad[3 * i + k] * grad[3 * j + k]).sum();
                    a[(i, j)] += wq * (row[i] * row[j] + regularization * g);
                }
            }
        }
        for i in 0..n {
            for j in 0..i {
                a[(i, j)] = a[(j, i)];
            }
        }
        let chol = Cholesky::new(a)
            .ok_or_else(|| SimError::solver("A-Ξ: the normal matrix of χ_h is not positive definite"))?;
        Ok(Self { exps, scale: s, lap, chol })
    }

    /// `χ_h` for the target `ε̇_v` at the volume nodes (weights `w`), and the projection
    /// defect `e_ε = ‖Δχ_h − ε̇_v‖ / ‖ε̇_v‖`.
    pub fn solve(&self, eps_v: &[f64], w: &[f64]) -> (Dilatational, f64) {
        let n = self.exps.len();
        // χ(x) = R_*² Σ a_i q_i(x / R_*), so Δχ = Σ a_i Δ_ξ q_i: the target is ε̇_v itself.
        let mut b = DVector::<f64>::zeros(n);
        for (q, (e, wq)) in eps_v.iter().zip(w).enumerate() {
            if *e != 0.0 {
                for i in 0..n {
                    b[i] += wq * self.lap[q * n + i] * e;
                }
            }
        }
        let x = self.chol.solve(&b);
        let (mut num, mut den) = (0.0, 0.0);
        for (q, (e, wq)) in eps_v.iter().zip(w).enumerate() {
            let fit: f64 = (0..n).map(|i| self.lap[q * n + i] * x[i]).sum();
            num += wq * (fit - e).powi(2);
            den += wq * e * e;
        }
        let dil = Dilatational {
            exps: self.exps.clone(),
            coef: x.iter().copied().collect(),
            scale: self.scale,
        };
        (dil, (num / den.max(1e-300)).sqrt())
    }
}

/// The dilatancy source of A-Ξ on one geometry (`stokes::DilatancySource`): the plastic
/// rate and yield mask at the volume nodes from the current state, the nonlocal driving
/// rate (§11.2), `ε̇_v = Ξ γ̇_p^⋆` (§11.3), and the least-squares `χ_h` (§11.4).
pub struct Source<'a> {
    pub space: DilatationSpace,
    pub nonlocal: super::markers::Nonlocal<'a>,
    pub params: &'a super::DilatancyParams,
    pub delta_y: f64,
    pub pts: &'a [DVec3],
    pub w: &'a [f64],
}

impl crate::scf::stokes::DilatancySource for Source<'_> {
    fn field(
        &self,
        m: &crate::rheo::Material,
        strength: &[crate::scf::stokes::Local],
        shear: &[f64],
        ptil: &[f64],
        eps: f64,
    ) -> crate::scf::stokes::DilatancyState {
        use rayon::prelude::*;
        let d = self.params;
        // F_δ = η γ̇ − (1 − δ_Y)(c + (μ_y + ξ) p̃) at the volume nodes (B2, §11.3).
        let (rate, mask, xi): (Vec<f64>, Vec<bool>, Vec<f64>) = {
            let rows: Vec<(f64, bool, f64)> = (0..shear.len())
                .into_par_iter()
                .map(|q| {
                    let l = &strength[q];
                    let ev = crate::scf::stokes::eta_eval(m, l, shear[q], ptil[q], eps);
                    let tau_y = l.c + (m.mu_y + l.xi) * ptil[q];
                    let yielded = ev.eta * shear[q] - (1.0 - self.delta_y) * tau_y > 0.0;
                    // Ξ = K_Ξ (ν − ν_c(I_c)) with ν = ν_c⁰ + ξ / K_Ξ.
                    let nu = if d.k > 0.0 { d.nu_c0 + l.xi / d.k } else { d.nu_c0 };
                    let big_xi = d.k * (nu - d.nu_c(ev.inertial));
                    (if yielded { shear[q] } else { 0.0 }, yielded, big_xi)
                })
                .collect();
            (rows.iter().map(|r| r.0).collect(), rows.iter().map(|r| r.1).collect(), rows.iter().map(|r| r.2).collect())
        };
        let drive = self.nonlocal.driving(&rate, self.w, self.pts, &mask);
        let eps_v: Vec<f64> = drive.iter().zip(&xi).map(|(g, x)| g * x).collect();
        let (field, defect) = self.space.solve(&eps_v, self.w);
        crate::scf::stokes::DilatancyState {
            field,
            defect,
            driving_max: drive.iter().copied().fold(0.0, f64::max),
        }
    }
}
