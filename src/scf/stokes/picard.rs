//! The fixed-viscosity Picard map of §8.7, kept as a check of the primal–dual iteration
//! (Proposition 2.17.1(1): the same fixed points).

use nalgebra::DVector;

use crate::common::error::SimError;
use crate::rheo::Material;

use super::super::params::StokesParams;
use super::pressure::Pressure;
use super::{
    Guess, Loads, Operators, StokesSolution, assemble_f, assemble_k, eta_eval, factor_range, load_scale,
    solve_range, strain_fields,
};

/// Picard solve at fixed geometry: the fixed-viscosity check of §10.4.2.
pub(super) fn solve_picard(
    ops: &Operators,
    ld: &Loads,
    prm: &StokesParams,
    m: &Material,
    guess: Option<&Guess>,
) -> Result<StokesSolution, SimError> {
    if ld.dilatancy.is_some() {
        return Err(SimError::model_not_closed(
            "A-Ξ (dilatancy) needs stokes.solver = \"primal-dual\"",
        ));
    }
    let n_u = ops.lay.n_u;
    let np = ops.vol.pts.len();
    let p_min = m.spec.p_min;
    let f_load = assemble_f(ops, ld);
    let f_norm = load_scale(ops, ld);
    let mut u = match guess {
        Some(g) if g.u.len() == n_u => DVector::from_column_slice(&g.u),
        _ => DVector::zeros(n_u),
    };
    let coh = &ld.strength.vol;
    let p_start: &[f64] = guess.map_or(&[], |g| &g.p);
    let mut pr = Pressure::warm(ops, p_min, p_start);
    let mut eta_prev: Vec<f64> = Vec::new();
    let mut total_iters = 0;
    let mut converged = false;
    let mut last = (f64::INFINITY, 0.0);
    let mut lambda = DVector::zeros(6);
    // Previous-sweep un-truncated Poisson pressure (for the L² pressure criterion (a)).
    let mut p_prev = pr.p_vals.clone();
    let mut eps_used = m.spec.gamma_eps;
    let mut theta = prm.theta_eta;
    let mut theta_retries = 0u32;
    let u_guess = u.clone();
    let theta_guess = theta;
    // §10.4.2: solve the target γ̇_ε first. The coarse-to-fine ladder runs only
    // when that attempt does not converge, and it restarts from the incoming guess.
    let mut plan = vec![(0usize, prm.continuation == 0)];
    let mut plan_at = 0usize;
    let mut homotopy_armed = prm.continuation > 0;

    while plan_at < plan.len() {
        let (level, retry_theta) = plan[plan_at];
        plan_at += 1;
        let eps = m.spec.gamma_eps * 10f64.powi(level as i32);
        eps_used = eps;
        loop {
            converged = false;
            eta_prev.clear();
            // The pressure history is local to each continuation level.
            let mut p_prev_set = false;
            for _ in 0..prm.picard_max {
                total_iters += 1;
                let fields = strain_fields(ops, &u);
                // (P1) pressure from the frozen (u^k, p̃^k).
                pr.update(ops, ld, prm, m, eps, &u, &fields, None)?;
                // (P2) viscosity from (γ̇(u^k), p̃^{k+1}).
                let mut eta: Vec<f64> = (0..np)
                    .map(|p| eta_eval(m, &coh[p], fields.shear[p], pr.ptil[p], eps).eta)
                    .collect();
                if !eta_prev.is_empty() {
                    for p in 0..np {
                        eta[p] = theta * eta[p] + (1.0 - theta) * eta_prev[p];
                    }
                }
                // (P3) velocity.
                let k = assemble_k(ops, &eta);
                let rs = factor_range(ops, &k)?;
                let (un, lam) = solve_range(ops, &rs, &f_load);
                // (P4) criterion (a): velocity in K-norm, pressure in L².
                // ‖u^{k+1} - u^k‖_K / max{‖u^{k+1}‖_K, ε_u}
                let du_vec = &un - &u;
                let ku_num = du_vec.dot(&(&k * &du_vec)).max(0.0).sqrt();
                let ku_den = un.dot(&(&k * &un)).max(0.0).sqrt().max(1e-300);
                let du = ku_num / ku_den;
                // ‖p^{k+1} - p^k‖_{L²} / max{‖p^{k+1}‖_{L²}, ε_p}
                let dp = {
                    let num: f64 = (0..np)
                        .map(|p| ops.vol.w[p] * (pr.p_vals[p] - p_prev[p]).powi(2))
                        .sum();
                    let den: f64 = (0..np).map(|p| ops.vol.w[p] * pr.p_vals[p].powi(2)).sum();
                    // First Picard sweep: p_prev is all p_min, treat as unconverged.
                    if p_prev_set {
                        (num / den.max(1e-300)).sqrt()
                    } else {
                        f64::INFINITY
                    }
                };
                let du_floor = un.norm() <= 1e-300;
                u = un;
                lambda = lam;
                p_prev.clone_from(&pr.p_vals);
                p_prev_set = true;
                eta_prev = eta;
                last = (du.max(dp), dp);
                if (du <= prm.tol_picard || du_floor) && dp <= prm.tol_picard {
                    converged = true;
                    break;
                }
            }
            if converged {
                break;
            }
            // §10.4.2: on the finest γ̇_ε, decrease θ_η and retry this level only.
            if retry_theta && theta > 0.25 + 1e-12 && theta_retries < 3 {
                theta = (0.5 * theta).max(0.25);
                theta_retries += 1;
                continue;
            }
            break;
        }
        if converged && level == 0 {
            break;
        }
        if !converged && level == 0 && homotopy_armed {
            u = u_guess.clone();
            pr = Pressure::warm(ops, p_min, p_start);
            p_prev.clone_from(&pr.p_vals);
            eta_prev.clear();
            theta = theta_guess;
            theta_retries = 0;
            homotopy_armed = false;
            plan.extend((1..=prm.continuation).rev().map(|lv| (lv, false)));
            plan.push((0, true));
            continue;
        }
        if !converged && level == 0 {
            return Err(SimError::rheology_not_converged(format!(
                "Picard iteration: relative change {:.3e} after {} sweeps (tolerance {:.1e})",
                last.0, total_iters, prm.tol_picard
            )));
        }
    }
    debug_assert!(converged);

    // Criterion (b) on the final iterate with the unrelaxed viscosity.
    let fields = strain_fields(ops, &u);
    let eta_final: Vec<f64> = (0..np)
        .map(|p| eta_eval(m, &coh[p], fields.shear[p], pr.ptil[p], eps_used).eta)
        .collect();
    let k = assemble_k(ops, &eta_final);
    let resid = &k * &u + ops.rigid.transpose() * &lambda - &f_load;
    let eps_momentum = resid.norm() / f_norm;
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
    // Strong-form rotational residual f - ∇p (diagnostic of §8.7).
    let eps_rot = pr.rotational_residual(ops, ld, m, eps_used, &u, &fields, None);
    let p_surface = pr.face_traction(ops, ld, prm, m, eps_used, &u, None)?;
    let (pres_c, lift) = pr.split(ops);
    let wsum: f64 = ops.vol.w.iter().sum();
    let tensile = (0..np)
        .filter(|&p| pr.p_vals[p] < 0.0)
        .map(|p| ops.vol.w[p])
        .sum::<f64>()
        / wsum;
    let eta_lo = eta_final.iter().copied().fold(f64::INFINITY, f64::min);
    let eta_hi = eta_final.iter().copied().fold(0.0, f64::max);
    Ok(StokesSolution {
        u: u.iter().copied().collect(),
        lambda: std::array::from_fn(|i| lambda[i]),
        pres_c,
        lift,
        eta: eta_final,
        p_tilde: pr.ptil,
        p_surface,
        picard_iterations: total_iters,
        eps_picard: last.0,
        eps_momentum,
        eps_lambda,
        eps_rot,
        eta_lo,
        eta_hi,
        tensile_fraction: tensile,
        inverse_ekman: 0.0,
        gamma_eps: eps_used,
        dilatancy: None,
    })
}
