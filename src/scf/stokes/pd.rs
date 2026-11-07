//! The primal–dual iteration of §10.3.2: Chambolle–Pock in the metric
//! `P = G + γ_AL B_rbᵀB_rb` on the strain samples, with the pressure updated every `N_p`
//! iterations.

use nalgebra::{Cholesky, DVector};

use crate::common::error::SimError;
use crate::common::linalg;
use crate::rheo::Material;

use super::super::params::StokesParams;
use super::super::evolving::dilatation::deviator;
use super::pressure::Pressure;
use super::{
    DilatancyState, Guess, Loads, Local, Operators, StokesSolution, assemble_f, d6_dot, eta_eval, load_scale,
    pd_metric, solve_range, strain_fields, strain_fields_with,
};

/// `φ'(s) = s η(s)` and `φ''(s) = η + s ∂η/∂γ̇` at frozen `c`, `p̃` (Proposition 10.0.12.1).
fn flow_slope(m: &Material, l: &Local, p_tilde: f64, eps: f64, s: f64) -> (f64, f64) {
    let ev = eta_eval(m, l, s, p_tilde, eps);
    (s * ev.eta, ev.eta + s * ev.d_shear)
}

/// The root `s ≥ 0` of `τ s + φ'(s) = t`: safeguarded Newton on `[0, t/τ]` from `s0`.
fn prox_rate(m: &Material, l: &Local, p_tilde: f64, eps: f64, tau: f64, t: f64, s0: f64) -> f64 {
    let (mut lo, mut hi) = (0.0, t / tau);
    let mut s = s0.clamp(lo, hi);
    for _ in 0..100 {
        let (f1, f2) = flow_slope(m, l, p_tilde, eps, s);
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
fn slope_excess(m: &Material, l: &Local, p_tilde: f64, eps: f64, a: f64, b: f64, t: f64) -> f64 {
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
            acc += W[k] * (flow_slope(m, l, p_tilde, eps, r).0 - t);
        }
    }
    acc * half
}

/// The frozen dilatational shift of A-Ξ (§11.4): `e_i^ε = √2 dev D(v_ε)(x_i)`, the
/// rigid offset `b_ε = (∫ρ v_ε, ∫ρ x × v_ε)`, and the state that produced them.
struct Shift {
    e: DVector<f64>,
    b: DVector<f64>,
    state: Option<DilatancyState>,
}

impl Shift {
    fn none(np: usize) -> Self {
        Self { e: DVector::zeros(6 * np), b: DVector::zeros(6), state: None }
    }

    fn set(ops: &Operators, state: DilatancyState) -> Self {
        let s2 = std::f64::consts::SQRT_2;
        let np = ops.vol.pts.len();
        let mut e = DVector::zeros(6 * np);
        let mut b = DVector::zeros(6);
        for (p, x) in ops.vol.pts.iter().enumerate() {
            let d = deviator(state.field.strain(*x));
            for c in 0..6 {
                e[6 * p + c] = s2 * d[c];
            }
            let v = state.field.velocity(*x) * (ops.vol.w[p] * ops.vol.rho[p]);
            let m = x.cross(v);
            for a in 0..3 {
                b[a] += v[a];
                b[3 + a] += m[a];
            }
        }
        Self { e, b, state: Some(state) }
    }

    fn active(&self) -> Option<&DVector<f64>> {
        self.state.as_ref().map(|_| &self.e)
    }

    fn field(&self) -> Option<&super::super::evolving::dilatation::Dilatational> {
        self.state.as_ref().map(|s| &s.field)
    }
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

/// Primal–dual iteration of §10.3.2 (Chambolle–Pock in the metric `P = G + γ_AL B_rbᵀB_rb`).
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
    let coh = &ld.strength.vol;
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
    // P-orthogonal projection onto the affine K_ε = {B_rb u = −b_ε} (K = ker B_rb
    // without A-Ξ, Proposition 11.8): the solve step stays in K, so only the start moves.
    let project = |u: &DVector<f64>, b: &DVector<f64>| {
        let lam = rs.s_chol.solve(&(&ops.rigid * u + b));
        u - &rs.y * lam
    };
    let start = project(&start, &DVector::zeros(6));

    let mut levels = vec![0usize];
    let mut ladder = prm.continuation > 0;
    let mut total = 0usize;
    let mut at = 0usize;
    let mut warm = start.clone();
    // The pressure is itself a fixed-point iteration in p̃ (Pressure::update); it starts
    // from the guess's coefficients, then from each level's.
    let p_start: &[f64] = guess.map_or(&[], |g| &g.p);
    let mut warm_p = p_start.to_vec();
    // Extensions of a level whose suspected collapse the bracket did not confirm, and the
    // gap at the last one: an extension must reduce it.
    let (mut extensions, mut gap_at_extension) = (0usize, f64::INFINITY);
    loop {
        let level = levels[at];
        at += 1;
        let eps = m.spec.gamma_eps * 10f64.powi(level as i32);
        let mut u = warm.clone();
        let mut pr = Pressure::warm(ops, p_min, &warm_p);
        // The pressure of the warm start, so that the first dual steps see its strength.
        let mut fields = strain_fields(ops, &u);
        let mut shift = Shift::none(np);
        pr.update(ops, ld, prm, m, eps, &u, &fields, None)?;
        // A-Ξ: the first dilatational field, the affine start, and its pressure.
        if let Some(src) = ld.dilatancy {
            shift = Shift::set(ops, src.field(m, coh, &fields.shear, &pr.ptil, eps));
            u = project(&u, &shift.b);
            fields = strain_fields_with(ops, &u, shift.active());
            pr.update(ops, ld, prm, m, eps, &u, &fields, shift.field())?;
        }
        let mut p_prev = pr.p_vals.clone();
        // ς⁰ = η e(u⁰); the dual step τ_d is η_ref, the weighted geometric mean of η⁰, and
        // τ_p τ_d < 1 (Section 10.3.2).
        let e0 = strain(&u);
        let mut sig = DVector::zeros(6 * np);
        let mut rate = vec![0.0; np];
        let (mut lsum, mut wsum) = (0.0, 0.0);
        for i in 0..np {
            let ev = eta_eval(m, &coh[i], fields.shear[i], pr.ptil[i], eps);
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
        // ‖u‖_P halfway through the level, for the collapse certificate.
        let mut u_half = 0.0;
        for k in 0..prm.pd_max {
            if k == prm.pd_max / 2 {
                u_half = p_norm(&u);
            }
            total += 1;
            // (D) pointwise prox of φ_i* at the frozen pressure: ς = z φ'(s)/|z| with
            // τ_d s + φ'(s) = |z| (Moreau).
            let eb = strain(&ubar);
            for i in 0..np {
                let mut z = [0.0; 6];
                for c in 0..6 {
                    z[c] = sig[6 * i + c] + tau_d * (eb[6 * i + c] + shift.e[6 * i + c]);
                }
                let t = d6_dot(&z, &z).sqrt();
                if t == 0.0 {
                    rate[i] = 0.0;
                    for c in 0..6 {
                        sig[6 * i + c] = 0.0;
                    }
                    continue;
                }
                let s = prox_rate(m, &coh[i], pr.ptil[i], eps, tau_d, t, rate[i]);
                rate[i] = s;
                let mag = flow_slope(m, &coh[i], pr.ptil[i], eps, s).0;
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
                fields = strain_fields_with(ops, &u, shift.active());
                let e = strain(&u) + &shift.e;
                // Relative Fenchel–Young gap at the frozen pressure,
                // φ(γ̇) + φ*(|ς|) − ⟨ς, e⟩, with φ'(s*) = |ς| from the prox.
                let (mut gap, mut scale) = (0.0, 0.0);
                for i in 0..np {
                    let sl = &sig.as_slice()[6 * i..6 * i + 6];
                    let el = &e.as_slice()[6 * i..6 * i + 6];
                    let t = d6_dot(sl, sl).sqrt();
                    let g = fields.shear[i];
                    let fy = slope_excess(m, &coh[i], pr.ptil[i], eps, rate[i], g, t)
                        + (t * g - d6_dot(sl, el));
                    gap += w[i] * fy.max(0.0);
                    scale += w[i] * t * g.max(rate[i]);
                }
                // The primal–dual gap at the frozen pressure: the Fenchel–Young gap of the
                // pointwise pairs and the dual infeasibility, which is the equilibrium residual.
                let (fy, eq) = (gap / scale.max(1e-300), equilibrium(&sig).0);
                let r_gap = fy.max(eq);
                pr.update(ops, ld, prm, m, eps, &u, &fields, shift.field())?;
                // A-Ξ: the dilatational field follows the strain and pressure (PO-23).
                if let Some(src) = ld.dilatancy {
                    shift = Shift::set(ops, src.field(m, coh, &fields.shear, &pr.ptil, eps));
                    u = project(&u, &shift.b);
                    ubar = u.clone();
                }
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
                     the load is incompatible with the rigid modes (§10.0.10(b), §A.3)",
                    prm.tol_momentum
                )));
            }
            let fields = strain_fields_with(ops, &u, shift.active());
            let eta_final: Vec<f64> = (0..np)
                .map(|p| eta_eval(m, &coh[p], fields.shear[p], pr.ptil[p], eps).eta)
                .collect();
            let eps_rot = pr.rotational_residual(ops, ld, m, eps, &u, &fields, shift.field());
            let p_surface = pr.face_traction(ops, ld, prm, m, eps, &u, shift.field())?;
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
                dilatancy: shift.state,
            });
        }
        // Collapse (Proposition 10.0.12.1): ‖u‖_P growing while the stress is pinned at
        // τ_max = c + (μ_2 + ξ + K_Ξ Δν_c) p̃ somewhere only suggests L* > 1; a near-critical
        // state also converges slowly at large rates. The certified bracket of
        // Proposition 10.0.12.3 at this frozen pressure decides. L* does not depend on γ̇_ε, so
        // the continuation ladder cannot help.
        let tau_max: Vec<f64> = (0..np)
            .map(|i| {
                let mu_max = m.spec.mu_2 + coh[i].xi + m.dilatancy.map_or(0.0, |d| d[0]);
                coh[i].c + mu_max * pr.ptil[i]
            })
            .collect();
        let saturation = (0..np)
            .map(|i| {
                let s = &sig.as_slice()[6 * i..6 * i + 6];
                d6_dot(s, s).sqrt() / tau_max[i]
            })
            .fold(0.0, f64::max);
        let growth = p_norm(&u) / u_half.max(1e-300);
        if saturation >= COLLAPSE_SATURATION && growth >= COLLAPSE_GROWTH {
            let suspect = format!(
                "the velocity grew {growth:.2}× over the second half of {} primal–dual iterations with \
                 the stress at {saturation:.4} of τ_max (equilibrium residual {:.3e})",
                prm.pd_max, last.1
            );
            // Under A-Ξ the admissible set is affine (Proposition 11.8) and the bracket does
            // not apply: the suspicion stands as the verdict.
            if ld.dilatancy.is_some() {
                return Err(SimError::limit_load_collapse(format!("{suspect}: no quasi-static state at this pressure")));
            }
            let b = super::limit::bracket_with(ops, &f_load, &tau_max, CERTIFY_GAP)?;
            if b.lo > 1.0 {
                return Err(SimError::limit_load_collapse(format!(
                    "{suspect}; certified L* in [{:.4}, {:.4}] > 1: no quasi-static state at this pressure",
                    b.lo, b.hi
                )));
            }
            if extensions >= PD_EXTENSIONS || last.1 > PD_PROGRESS * gap_at_extension {
                return Err(SimError::rheology_not_converged(format!(
                    "{suspect}; certified L* in [{:.4}, {:.4}] does not exclude a quasi-static state, \
                     which {extensions} further levels of iterations did not reach",
                    b.lo, b.hi
                )));
            }
            // Not a collapse: the same level again from where it stopped, while the gap falls.
            extensions += 1;
            gap_at_extension = last.1;
            levels.insert(at, level);
            warm = u;
            warm_p = pr.coeff.as_slice().to_vec();
            continue;
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

/// `τ_p τ_d` of the primal–dual steps (below 1, Section 10.3.2).
const PD_THETA: f64 = 0.99;
/// Collapse certificate: `max |ς_i| / τ_max,i` at least this, and `‖u‖_P` at least this
/// many times its value halfway through the level (linear growth gives 2, convergence 1).
const COLLAPSE_SATURATION: f64 = 0.999;
const COLLAPSE_GROWTH: f64 = 1.8;
/// Relative width at which the bracket of a suspected collapse stops; the further levels of
/// `pd_max` iterations at most that a suspicion the bracket does not confirm gets, each
/// only while the primal–dual gap still falls below this fraction of its last value.
const CERTIFY_GAP: f64 = 1e-3;
const PD_EXTENSIONS: usize = 10;
const PD_PROGRESS: f64 = 0.9;

