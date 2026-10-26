//! The primal–dual iteration of §10.4.2: Chambolle–Pock in the metric
//! `P = G + γ_AL B_rbᵀB_rb` on the strain samples, with the pressure updated every `N_p`
//! iterations.

use nalgebra::{Cholesky, DVector};

use crate::common::error::SimError;
use crate::common::linalg;
use crate::common::math::DVec3;
use crate::rheo::Material;

use super::super::params::StokesParams;
use super::pressure::Pressure;
use super::{
    Guess, Loads, Operators, StokesSolution, assemble_f, d6_dot, eta_eval, load_scale, pd_metric,
    solve_range, strain_fields,
};

/// `φ'(s) = s η(s)` and `φ''(s) = η + s ∂η/∂γ̇` at frozen `c`, `p̃` (Proposition 2.17.1).
fn flow_slope(m: &Material, c: f64, p_tilde: f64, eps: f64, s: f64) -> (f64, f64) {
    let ev = eta_eval(m, c, s, p_tilde, eps);
    (s * ev.eta, ev.eta + s * ev.d_shear)
}

/// The root `s ≥ 0` of `τ s + φ'(s) = t`: safeguarded Newton on `[0, t/τ]` from `s0`.
fn prox_rate(m: &Material, c: f64, p_tilde: f64, eps: f64, tau: f64, t: f64, s0: f64) -> f64 {
    let (mut lo, mut hi) = (0.0, t / tau);
    let mut s = s0.clamp(lo, hi);
    for _ in 0..100 {
        let (f1, f2) = flow_slope(m, c, p_tilde, eps, s);
        let h = tau * s + f1 - t;
        if h.abs() <= 1e-14 * t {
            break;
        }
        if h > 0.0 {
            hi = s;
        } else {
            lo = s;
        }
        let next = s - h / (tau + f2);
        s = if next > lo && next < hi { next } else { 0.5 * (lo + hi) };
        if hi - lo <= 1e-15 * hi {
            break;
        }
    }
    s
}

/// `∫_a^b (φ'(r) − t) dr` by 8-point Gauss–Legendre: the convexity part of the
/// Fenchel–Young gap, with `φ'(a) = t`.
fn slope_excess(m: &Material, c: f64, p_tilde: f64, eps: f64, a: f64, b: f64, t: f64) -> f64 {
    const X: [f64; 4] = [
        0.183_434_642_495_649_8,
        0.525_532_409_916_329,
        0.796_666_477_413_626_7,
        0.960_289_856_497_536_3,
    ];
    const W: [f64; 4] = [
        0.362_683_783_378_362,
        0.313_706_645_877_887_3,
        0.222_381_034_453_374_5,
        0.101_228_536_290_376_3,
    ];
    let (mid, half) = (0.5 * (a + b), 0.5 * (b - a));
    let mut acc = 0.0;
    for k in 0..4 {
        for sign in [-1.0, 1.0] {
            let r = mid + sign * half * X[k];
            acc += W[k] * (flow_slope(m, c, p_tilde, eps, r).0 - t);
        }
    }
    acc * half
}

/// `Σ_i w_i e_iᵀ ς_i` with `e_i = √2 D_i u`.
fn strain_adjoint(ops: &Operators, sig: &DVector<f64>) -> DVector<f64> {
    let s2 = std::f64::consts::SQRT_2;
    let mut ws = sig.clone();
    for (i, w) in ops.vol.w.iter().enumerate() {
        for c in 0..6 {
            ws[6 * i + c] *= w * s2;
        }
    }
    linalg::tr_mul(&ops.rows.d, &ws)
}

/// Primal–dual iteration of §10.4.2 (Chambolle–Pock in the metric `P = G + γ_AL B_rbᵀB_rb`).
///
/// The strain samples are `e_i = √2 D(x_i)` (six scaled components, `|e_i| = γ̇`), so
/// `G = Σ w_i e_iᵀ e_i` is the stiffness at `η ≡ 1`. The dual `ς_i` is the flow stress over
/// `√2`. The pressure is updated every `N_p` iterations and the iteration continues warm.
pub(super) fn solve_pd(
    ops: &Operators,
    ld: &Loads,
    prm: &StokesParams,
    m: &Material,
    guess: Option<&Guess>,
) -> Result<StokesSolution, SimError> {
    let n_u = ops.lay.n_u;
    let np = ops.vol.pts.len();
    let p_min = m.spec.p_min;
    let f_load = assemble_f(ops, ld);
    let f_norm = load_scale(ops, ld);
    let coh: Vec<f64> = ops.vol.depth.iter().map(|d| m.cohesion_at(*d)).collect();
    let gcoh: Vec<DVec3> = (0..np)
        .map(|p| ops.vol.gdepth[p] * m.cohesion_slope(ops.vol.depth[p]))
        .collect();
    let w = &ops.vol.w;
    let s2 = std::f64::consts::SQRT_2;
    // P is factored once per geometry; Π_K^P P⁻¹ r is the range-space solve.
    let (metric, rs) = pd_metric(ops)?;
    // The equilibrium residual F − Σ w_i e_iᵀ ς_i − B_rbᵀ λ with the multiplier that the
    // rigid modes absorb, λ = (B Bᵀ)⁻¹ B (F − Σ w_i e_iᵀ ς_i).
    let bbt = Cholesky::new(&ops.rigid * ops.rigid.transpose())
        .ok_or_else(|| SimError::rigid_rank("B_rb B_rbᵀ is singular"))?;
    let equilibrium = |sig: &DVector<f64>| {
        let r = &f_load - strain_adjoint(ops, sig);
        let lambda = bbt.solve(&(&ops.rigid * &r));
        let resid = &r - ops.rigid.transpose() * &lambda;
        (resid.norm() / f_norm, lambda)
    };
    let p_norm = |v: &DVector<f64>| v.dot(&(metric * v)).max(0.0).sqrt();
    let strain = |u: &DVector<f64>| linalg::mul(&ops.rows.d, u) * s2;
    let start = match guess {
        Some(g) if g.u.len() == n_u => DVector::from_column_slice(&g.u),
        _ => DVector::zeros(n_u),
    };
    // P-orthogonal projection of the warm start onto K = ker B_rb.
    let start = {
        let lam = rs.s_chol.solve(&(&ops.rigid * &start));
        &start - &rs.y * lam
    };

    let mut levels = vec![0usize];
    let mut ladder = prm.continuation > 0;
    let mut total = 0usize;
    let mut at = 0usize;
    let mut warm = start.clone();
    // The pressure is itself a fixed-point iteration in p̃ (Pressure::update); it starts
    // from the guess's coefficients, then from each level's.
    let p_start: &[f64] = guess.map_or(&[], |g| &g.p);
    let mut warm_p = p_start.to_vec();
    loop {
        let level = levels[at];
        at += 1;
        let eps = m.spec.gamma_eps * 10f64.powi(level as i32);
        let mut u = warm.clone();
        let mut pr = Pressure::warm(ops, p_min, &warm_p);
        // The pressure of the warm start, so that the first dual steps see its strength.
        let mut fields = strain_fields(ops, &u);
        pr.update(ops, ld, prm, m, eps, &coh, &gcoh, &u, &fields)?;
        let mut p_prev = pr.p_vals.clone();
        // ς⁰ = η e(u⁰); the dual step τ_d is η_ref, the weighted geometric mean of η⁰, and
        // τ_p τ_d < 1 (Proposition 10.4).
        let e0 = strain(&u);
        let mut sig = DVector::zeros(6 * np);
        let mut rate = vec![0.0; np];
        let (mut lsum, mut wsum) = (0.0, 0.0);
        for i in 0..np {
            let ev = eta_eval(m, coh[i], fields.shear[i], pr.ptil[i], eps);
            for c in 0..6 {
                sig[6 * i + c] = ev.eta * e0[6 * i + c];
            }
            rate[i] = fields.shear[i];
            lsum += w[i] * ev.eta.ln();
            wsum += w[i];
        }
        let eta_ref = (lsum / wsum).exp();
        let tau_d = eta_ref;
        let tau_p = PD_THETA / eta_ref;
        let mut ubar = u.clone();
        let mut converged = false;
        let mut last = (f64::INFINITY, f64::INFINITY, f64::INFINITY);
        for k in 0..prm.pd_max {
            total += 1;
            // (D) pointwise prox of φ_i* at the frozen pressure: ς = z φ'(s)/|z| with
            // τ_d s + φ'(s) = |z| (Moreau).
            let eb = strain(&ubar);
            for i in 0..np {
                let mut z = [0.0; 6];
                for c in 0..6 {
                    z[c] = sig[6 * i + c] + tau_d * eb[6 * i + c];
                }
                let t = d6_dot(&z, &z).sqrt();
                if t == 0.0 {
                    rate[i] = 0.0;
                    for c in 0..6 {
                        sig[6 * i + c] = 0.0;
                    }
                    continue;
                }
                let s = prox_rate(m, coh[i], pr.ptil[i], eps, tau_d, t, rate[i]);
                rate[i] = s;
                let mag = flow_slope(m, coh[i], pr.ptil[i], eps, s).0;
                for c in 0..6 {
                    sig[6 * i + c] = z[c] * (mag / t);
                }
            }
            // (U) u ← u − τ_p Π_K^P P⁻¹ (Σ w_i e_iᵀ ς_i − F), then the extrapolation.
            let r = strain_adjoint(ops, &sig) - &f_load;
            let (step, _) = solve_range(ops, rs, &r);
            let un = &u - step * tau_p;
            let du = p_norm(&(&un - &u)) / p_norm(&un).max(1e-300);
            ubar = &un * 2.0 - &u;
            u = un;
            // (P) and (S) every N_p iterations.
            if (k + 1) % prm.pd_pressure_every.max(1) == 0 {
                fields = strain_fields(ops, &u);
                let e = strain(&u);
                // Relative Fenchel–Young gap at the frozen pressure,
                // φ(γ̇) + φ*(|ς|) − ⟨ς, e⟩, with φ'(s*) = |ς| from the prox.
                let (mut gap, mut scale) = (0.0, 0.0);
                for i in 0..np {
                    let sl = &sig.as_slice()[6 * i..6 * i + 6];
                    let el = &e.as_slice()[6 * i..6 * i + 6];
                    let t = d6_dot(sl, sl).sqrt();
                    let g = fields.shear[i];
                    let fy = slope_excess(m, coh[i], pr.ptil[i], eps, rate[i], g, t)
                        + (t * g - d6_dot(sl, el));
                    gap += w[i] * fy.max(0.0);
                    scale += w[i] * t * g.max(rate[i]);
                }
                // The primal–dual gap at the frozen pressure: the Fenchel–Young gap of the
                // pointwise pairs and the dual infeasibility, which is the equilibrium residual.
                let (fy, eq) = (gap / scale.max(1e-300), equilibrium(&sig).0);
                let r_gap = fy.max(eq);
                pr.update(ops, ld, prm, m, eps, &coh, &gcoh, &u, &fields)?;
                let num: f64 = (0..np).map(|p| w[p] * (pr.p_vals[p] - p_prev[p]).powi(2)).sum();
                let den: f64 = (0..np).map(|p| w[p] * pr.p_vals[p].powi(2)).sum();
                let dp = (num / den.max(1e-300)).sqrt();
                p_prev.clone_from(&pr.p_vals);
                last = (du, r_gap, dp);
                let du_floor = p_norm(&u) <= 1e-300;
                if (du <= prm.pd_tol || du_floor) && r_gap <= prm.pd_tol && dp <= prm.tol_picard {
                    converged = true;
                    break;
                }
            }
        }
        if converged && level == 0 {
            // Criterion (b) on F − Σ w_i e_iᵀ ς_i − Bᵀλ.
            let (eps_momentum, lambda) = equilibrium(&sig);
            let eps_lambda = (ops.rigid.transpose() * &lambda).norm() / f_norm;
            if eps_momentum > prm.tol_momentum {
                return Err(SimError::rheology_not_converged(format!(
                    "weak momentum residual {eps_momentum:.3e} exceeds {:.1e}",
                    prm.tol_momentum
                )));
            }
            if eps_lambda > prm.tol_momentum {
                return Err(SimError::rigid_rank(format!(
                    "rigid-body multiplier {eps_lambda:.3e} exceeds tol_momentum {:.1e}: \
                     the load is incompatible with the rigid modes (§2.15(b), §A.3)",
                    prm.tol_momentum
                )));
            }
            let fields = strain_fields(ops, &u);
            let eta_final: Vec<f64> = (0..np)
                .map(|p| eta_eval(m, coh[p], fields.shear[p], pr.ptil[p], eps).eta)
                .collect();
            let eps_rot = pr.rotational_residual(ops, ld, m, eps, &coh, &gcoh, &u, &fields);
            let p_surface = pr.face_traction(ops, ld, prm, m, eps, &u)?;
            let (pres_c, lift) = pr.split(ops);
            let wsum: f64 = w.iter().sum();
            let tensile = (0..np)
                .filter(|&p| pr.p_vals[p] < 0.0)
                .map(|p| w[p])
                .sum::<f64>()
                / wsum;
            let eta_lo = eta_final.iter().copied().fold(f64::INFINITY, f64::min);
            let eta_hi = eta_final.iter().copied().fold(0.0, f64::max);
            return Ok(StokesSolution {
                u: u.iter().copied().collect(),
                lambda: std::array::from_fn(|i| lambda[i]),
                pres_c,
                lift,
                eta: eta_final,
                p_tilde: pr.ptil,
                p_surface,
                picard_iterations: total,
                eps_picard: last.0.max(last.1),
                eps_momentum,
                eps_lambda,
                eps_rot,
                eta_lo,
                eta_hi,
                tensile_fraction: tensile,
                inverse_ekman: 0.0,
                gamma_eps: eps,
            });
        }
        if level == 0 && ladder {
            // Continuation ladder 10^c γ̇_ε, …, 10 γ̇_ε, γ̇_ε from the incoming guess,
            // warm-started from level to level.
            ladder = false;
            levels.extend((1..=prm.continuation).rev());
            levels.push(0);
            warm = start.clone();
            warm_p = p_start.to_vec();
        } else {
            warm = u;
            warm_p = pr.coeff.as_slice().to_vec();
        }
        if at >= levels.len() {
            return Err(SimError::rheology_not_converged(format!(
                "primal–dual iteration: velocity change {:.3e}, gap {:.3e}, pressure change \
                 {:.3e} after {total} iterations (tolerances {:.1e}, {:.1e})",
                last.0, last.1, last.2, prm.pd_tol, prm.tol_picard
            )));
        }
    }
}

/// `τ_p τ_d` of the primal–dual steps (below 1, Proposition 10.4).
const PD_THETA: f64 = 0.99;

