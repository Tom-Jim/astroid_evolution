//! f64 reference of the scalar rheology and friction-angle closure
//! (§4.1–§4.6, §10.2), and the classical-limit checks of §4.7–§4.20.
//!
//! The GPU evaluates the same formulas in double-single; the acceptance checks
//! compare the two, and the f64 path is the `f32` implementation's required
//! recheck (§10.2).

use super::params::{Material, MuProjMode};

/// `mu_cl(I) = mu_s + (mu_2 - mu_s) I / (I_0 + I)` (§4.1).
pub fn mu_of_inertial(i: f64, mu_s: f64, mu_2: f64, i_0: f64) -> f64 {
    if i <= 0.0 {
        return mu_s;
    }
    mu_s + (mu_2 - mu_s) * (i / (i_0 + i))
}

/// Classical inertial number `I = d gd / sqrt(p / rho_p)`.
pub fn inertial_number(d: f64, shear: f64, p: f64, rho_p: f64) -> f64 {
    if p <= 0.0 {
        return 0.0;
    }
    d * shear / (p / rho_p).sqrt()
}

/// `p_c = C_c Z nu F_H / d^2` (§4.3).
pub fn cohesive_pressure(c_c: f64, z: f64, nu: f64, f_h: f64, d: f64) -> f64 {
    c_c * z * nu * f_h / (d * d)
}

/// `mu_d` of §10.2 as the rational form that stays finite as `p_* -> 0`.
pub fn dynamic_friction(m: &Material, shear: f64, p_star: f64) -> (f64, f64) {
    let s = &m.spec;
    let scale = (p_star / s.rho_p).sqrt();
    let df = s.d_p * shear;
    let den = s.i_0 * scale + df;
    if den > 0.0 {
        (s.mu_s + (s.mu_2 - s.mu_s) * df / den, df / scale)
    } else {
        (s.mu_s, 0.0)
    }
}

/// Inputs of the scalar closure at one surface node.
#[derive(Clone, Copy, Debug)]
pub struct ClosureInput {
    /// `S0 = 2[(tr M)^2 + tr(M^2)]`, the shear rate squared without the tide traction.
    pub s0: f64,
    /// `t_a g^{ab} t_b` with `t_a = n . T_tide . e_a`.
    pub tg: f64,
    pub pressure: f64,
    /// `g . n`.
    pub g_dot_n: f64,
    /// `|g|`.
    pub g_norm: f64,
    /// `|g x n|`.
    pub g_cross_n: f64,
    /// `||T_tide||_F`.
    pub tide_frob: f64,
    /// Continuum cohesion `c` at the node (flow strength), `c_s` of §4.3a when undamaged.
    pub cohesion: f64,
    /// Shallow cohesion `c_Γ` of the repose proxy (§10.2, §11.6), `c_s` when undamaged.
    pub shallow_cohesion: f64,
    /// Static dilatancy shift `ξ = K_Ξ (ν − ν_c⁰)` of the friction (A-Ξ), zero without it.
    pub xi: f64,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ClosureOutput {
    pub shear: f64,
    pub inertial: f64,
    pub mu_d: f64,
    pub eta: f64,
    pub tau_eq: f64,
    pub p_tilde: f64,
    pub p_star: f64,
    pub p_ref: f64,
    pub mu_proj: f64,
    pub mu_eff: f64,
    pub phi: f64,
    pub sin2_phi: f64,
    pub sin_slope: f64,
    pub slack: f64,
    pub overhang_margin: f64,
    pub iterations: u32,
    pub converged: bool,
}

/// Frozen-viscosity fixed point `gd^2 = S0 + tg (gd^2 + eps^2) / tau(gd)^2`, then the chain
/// `I_c -> mu_d -> eta -> mu_eff -> phi` and the slope slack of the SOCP row.
pub fn closure(m: &Material, input: &ClosureInput) -> ClosureOutput {
    let s = &m.spec;
    let p_tilde = input.pressure.max(0.0).max(s.p_min);
    let p_star = p_tilde + m.p_c;
    let eps2 = s.gamma_eps * s.gamma_eps;
    let mut g2 = input.s0;
    let mut converged = true;
    let mut iterations = 0;
    if input.tg > 0.0 {
        converged = false;
        for it in 0..m.max_iter {
            let (mu_cs, i_c) = dynamic_friction(m, g2.sqrt(), p_star);
            let mu_d = mu_cs + input.xi + m.dilatancy_rate_term(i_c).0;
            let tau = input.cohesion + mu_d * p_tilde;
            let next = input.s0 + input.tg / (tau * tau) * (g2 + eps2);
            iterations = it + 1;
            if !next.is_finite() || next > 1.0e30 {
                g2 = next;
                break;
            }
            let diff = (next - g2).abs();
            g2 = next;
            if diff <= m.tol * (next + eps2) {
                converged = true;
                break;
            }
        }
    }
    let shear = g2.sqrt();
    let (mu_cs, inertial) = dynamic_friction(m, shear, p_star);
    let mu_d = mu_cs + input.xi + m.dilatancy_rate_term(inertial).0;
    let tau = input.cohesion + mu_d * p_tilde;
    let eta = tau / (g2 + eps2).sqrt();
    let p_ref = m.rho_bulk * input.g_dot_n.abs() * s.h_s;
    let p_star_ref = (p_ref + m.p_c).max(s.p_min);
    let mu_proj = match m.mode {
        MuProjMode::Dynamic => mu_d,
        MuProjMode::Static => m.mu_y + input.xi,
    };
    let num = input.shallow_cohesion + mu_proj * p_ref - input.tide_frob / std::f64::consts::SQRT_2;
    let mu_eff = num.max(0.0) / p_star_ref;
    let sin2 = mu_eff * mu_eff / (1.0 + mu_eff * mu_eff);
    let sin_slope = input.g_cross_n / input.g_norm;
    ClosureOutput {
        shear,
        inertial,
        mu_d,
        eta,
        tau_eq: tau,
        p_tilde,
        p_star,
        p_ref,
        mu_proj,
        mu_eff,
        phi: mu_eff.atan(),
        sin2_phi: sin2,
        sin_slope,
        slack: (sin_slope - sin2.sqrt()).max(0.0),
        overhang_margin: -input.g_dot_n / input.g_norm - s.eps_overhang,
        iterations,
        converged,
    }
}

/// Residuals of the classical limit (§4.7–§4.20) for one parameter set.
#[derive(Clone, Copy, Debug)]
pub struct DegeneracyReport {
    pub cohesion_ratio: f64,
    pub tide_ratio: f64,
    pub inertial_rel_diff: f64,
    pub friction_abs_diff: f64,
    pub flow_stress_rel_diff: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct DegeneracyCase {
    pub pressure: f64,
    pub nu: f64,
    pub z: f64,
    pub d_p: f64,
    pub rho_p: f64,
    pub f_h: f64,
    pub c_c: f64,
    pub c_t: f64,
    pub dev_h_frob: f64,
    pub shear: f64,
    pub mu_s: f64,
    pub mu_2: f64,
    pub i_0: f64,
    pub cohesion: f64,
    pub gamma_eps: f64,
}

/// Conditions A–F with `p_c/p -> 0`, `T_tide/p -> 0`, `c/p -> 0`, `gd_eps/gd -> 0`:
/// the joint model has to collapse onto `tau = mu_cl(I) p` (§4.19).
pub fn degeneracy(case: &DegeneracyCase) -> DegeneracyReport {
    let p = case.pressure;
    let p_c = cohesive_pressure(case.c_c, case.z, case.nu, case.f_h, case.d_p);
    let t_norm = case.c_t * case.rho_p * case.nu * case.d_p * case.d_p * case.dev_h_frob;
    let i_cl = inertial_number(case.d_p, case.shear, p, case.rho_p);
    let i_c = inertial_number(case.d_p, case.shear, p + p_c, case.rho_p);
    let mu_cl = mu_of_inertial(i_cl, case.mu_s, case.mu_2, case.i_0);
    let mu_d = mu_of_inertial(i_c, case.mu_s, case.mu_2, case.i_0);
    let eta =
        (case.cohesion + mu_d * p) / (case.shear * case.shear + case.gamma_eps.powi(2)).sqrt();
    // tau_eq = ||2 eta D'||_eq = eta gd for a deviatoric D' with gd = sqrt(2 D':D').
    let tau_eq = eta * case.shear;
    let expected = mu_cl * p;
    DegeneracyReport {
        cohesion_ratio: p_c / p,
        tide_ratio: t_norm / p,
        inertial_rel_diff: (i_c - i_cl).abs() / i_cl.max(1e-300),
        friction_abs_diff: (mu_d - mu_cl).abs(),
        flow_stress_rel_diff: (tau_eq - expected).abs() / expected,
    }
}
