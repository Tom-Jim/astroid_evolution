//! The certified bracket of the limit-load multiplier `L*` (§10.0.12.3).
//!
//! `L* = 1 / min{N(u) : ℓᵀu = 1, B u = 0}` with `N(u) = Σ_i w_i τ_i γ̇_i(u)`. For any positive
//! weights `q`, the solution of `Σ q_i E_iᵀE_i u = λ ℓ + Bᵀμ` gives the kinematic bound
//! `ℓᵀu / N(u) ≤ L*` and the static bound `L* ≤ max_i q_i γ̇_i(u) / (w_i τ_i)`
//! (Proposition 10.0.12.3); the weights of the smoothed minimizer close the gap.

use nalgebra::DVector;

use crate::common::error::SimError;
use crate::common::linalg;
use crate::rheo::Material;

use super::super::params::StokesParams;
use super::pressure::Pressure;
use super::{Fields, Loads, Operators, assemble_f, factor_range, solve_range};

/// The load vector `ℓ` and the pressure `p₀` of the body at rest under its loads.
#[derive(Clone, Debug)]
pub struct LimitData {
    pub load: DVector<f64>,
    pub p0: Vec<f64>,
}

impl LimitData {
    /// `a + t (b − a)`: both parts are affine in `n²`.
    pub fn lerp(a: &Self, b: &Self, t: f64) -> Self {
        Self {
            load: &a.load + (&b.load - &a.load) * t,
            p0: a.p0.iter().zip(&b.p0).map(|(x, y)| x + t * (y - x)).collect(),
        }
    }
}

/// `ℓ` of Proposition 10.0.12.1 and `p₀`, the Nitsche pressure (§8.1.3) with `τ = T_tide`.
pub fn data(ops: &Operators, ld: &Loads, prm: &StokesParams, m: &Material) -> Result<LimitData, SimError> {
    let np = ops.vol.pts.len();
    let mut pr = Pressure::new(ops, m.spec.p_min);
    let zero = DVector::zeros(ops.n_u);
    let rest = Fields { shear: vec![0.0; np], d: DVector::zeros(6 * np) };
    pr.update(ops, ld, prm, m, m.spec.gamma_eps, &zero, &rest, None)?;
    Ok(LimitData { load: assemble_f(ops, ld), p0: pr.p_vals })
}

/// A certified bracket `lo ≤ L* ≤ hi`, and the mechanism of its kinematic bound: velocity
/// coefficients `u` with `N(u) = 1` and their shear rates `γ̇_i(u)` at the volume nodes.
/// `stress` is the static field of the upper bound, `σ̂_i = q_i E_i u_q / w_i` (equivalent
/// norm `|σ̂_i|`), which equilibrates the load.
#[derive(Clone, Debug)]
pub struct Bracket {
    pub lo: f64,
    pub hi: f64,
    pub iterations: usize,
    pub rates: Vec<f64>,
    pub u: Vec<f64>,
    pub stress: Vec<[f64; 6]>,
}

/// Reweighting passes allowed; each is one weighted Gram matrix and one dense solve.
const MAX_PASSES: usize = 200;

/// The bracket of `L*` with the strength `τ_i = c_i + μ max(p₀_i, p_min)`. Stops once the
/// bracket excludes 1 or its relative width is below `gap`.
pub fn bracket(
    ops: &Operators,
    data: &LimitData,
    cohesion: &[f64],
    mu: f64,
    p_min: f64,
    gap: f64,
) -> Result<Bracket, SimError> {
    let tau: Vec<f64> = (0..cohesion.len()).map(|i| cohesion[i] + mu * data.p0[i].max(p_min)).collect();
    bracket_with(ops, &data.load, &tau, gap, false)
}

/// The bracket of `L*` for the load vector `load` and the strengths `tau` at the volume nodes.
/// `certify = false` exits as soon as the bracket excludes 1; `certify = true` keeps
/// tightening until the relative width meets `gap` (or `MAX_PASSES`), so the caller
/// decides on a converged bracket instead of an early-exit bound.
pub fn bracket_with(ops: &Operators, load: &DVector<f64>, tau: &[f64], gap: f64, certify: bool) -> Result<Bracket, SimError> {
    let _span = super::super::timing::span("limit_load");
    let np = ops.vol.pts.len();
    let cw: Vec<f64> = (0..np).map(|i| ops.vol.w[i] * tau[i]).collect();
    if cw.iter().any(|c| !(*c > 0.0)) {
        return Err(SimError::model_not_closed("limit load: the strength must be positive"));
    }
    let mut q = cw.clone();
    let (mut lo, mut hi) = (0.0_f64, f64::INFINITY);
    let mut rates = vec![0.0; np];
    let mut mech = vec![0.0; ops.n_u];
    let mut stress = vec![[0.0; 6]; np];
    let mut theta = 1.0;
    for pass in 1..=MAX_PASSES {
        // Σ q_i E_iᵀE_i with E_i = √2 D_i, so that |E_i u| = γ̇_i(u).
        let two_q: Vec<f64> = q.iter().map(|v| 2.0 * v).collect();
        let k = linalg::weighted_gram(&ops.rows.d, 6, &two_q);
        let rs = factor_range(ops, &k)?;
        let (u, _) = solve_range(ops, &rs, load);
        let c = load.dot(&u);
        if !(c > 0.0) {
            // ℓ vanishes on the admissible velocities: no load, L* = 0.
            return Ok(Bracket { lo: 0.0, hi: 0.0, iterations: pass, rates, u: mech, stress });
        }
        let d = linalg::mul(&ops.rows.d, &u);
        let shear: Vec<f64> = (0..np)
            .map(|i| {
                let s = &d.as_slice()[6 * i..6 * i + 6];
                (2.0 * s.iter().map(|x| x * x).sum::<f64>()).sqrt()
            })
            .collect();
        let norm: f64 = cw.iter().zip(&shear).map(|(c, g)| c * g).sum();
        let stat = (0..np).map(|i| q[i] * shear[i] / cw[i]).fold(0.0_f64, f64::max);
        if c / norm > lo {
            lo = c / norm;
            rates = shear.iter().map(|g| g / norm).collect();
            mech = u.iter().map(|c| c / norm).collect();
        }
        if stat < hi {
            hi = stat;
            let s2 = std::f64::consts::SQRT_2;
            stress = (0..np).map(|i| std::array::from_fn(|c| q[i] / ops.vol.w[i] * s2 * d[6 * i + c])).collect();
        }
        if (hi - lo <= gap * hi) || (!certify && (lo > 1.0 || hi < 1.0)) {
            return Ok(Bracket { lo, hi, iterations: pass, rates, u: mech, stress });
        }
        // Smoothed-stationarity weights with a shrinking smoothing ε = θ · rms γ̇.
        let rms = (shear.iter().map(|g| g * g).sum::<f64>() / np as f64).sqrt();
        let eps = theta * rms;
        theta = (theta * 0.5).max(1e-8);
        for i in 0..np {
            q[i] = cw[i] / (shear[i] * shear[i] + eps * eps).sqrt();
        }
    }
    Ok(Bracket { lo, hi, iterations: MAX_PASSES, rates, u: mech, stress })
}
