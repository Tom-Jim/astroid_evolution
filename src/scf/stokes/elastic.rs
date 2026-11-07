//! The elastic cycle of A-Λ (§10.3.7): its stress (Proposition 10.3.7.4), the peak and
//! shakedown brackets (Proposition 10.3.7.5), and the steady cycle (Algorithm 10.3.7.6).
//!
//! Stress samples use the convention of Proposition 10.0.12.3: `σ̂ = τ/√2` in the Mandel
//! six-vector, so `|σ̂|` is the equivalent stress and `σ̂ · E u` the work with `E = √2 D`.

use std::f64::consts::{PI, SQRT_2};

use nalgebra::DVector;

use crate::common::error::SimError;
use crate::common::linalg;
use crate::rheo::Material;

use super::super::cycle::Cycle;
use crate::common::math::mat_sym;
use super::limit::Bracket;
use super::{Loads, Operators, Strengths, assemble_f, pd_metric, pressure, solve_range};

/// The two components (cos, sin) of the cycle at the volume nodes, modulus-free.
pub struct ElasticCycle {
    /// Load vectors `ℓ_c`, `ℓ_s`.
    pub load: [DVector<f64>; 2],
    /// `σ̂_c`, `σ̂_s` and `δp_c`, `δp_s` at the volume nodes.
    pub sigma: [Vec<[f64; 6]>; 2],
    pub dp: [Vec<f64>; 2],
}

impl ElasticCycle {
    /// `σ̂ᴱ_i(M)`.
    pub fn sigma_at(&self, i: usize, m: f64) -> [f64; 6] {
        let (s, c) = m.sin_cos();
        std::array::from_fn(|k| c * self.sigma[0][i][k] + s * self.sigma[1][i][k])
    }

    pub fn dp_at(&self, i: usize, m: f64) -> f64 {
        m.cos() * self.dp[0][i] + m.sin() * self.dp[1][i]
    }

    fn dp_amplitude(&self, i: usize) -> f64 {
        self.dp[0][i].hypot(self.dp[1][i])
    }
}

/// Proposition 10.3.7.4: the incompressible elastic response to each component of `cycle`,
/// solved with the unit-viscosity stiffness of this geometry. `tide_coef` scales the
/// backstress `C_T ρ_p ν d² dev H`.
pub fn elastic_cycle(ops: &Operators, cycle: &Cycle, m: &Material, strength: &Strengths, g_const: f64) -> Result<ElasticCycle, SimError> {
    let _span = super::super::timing::span("elastic_cycle");
    let (_, rs) = pd_metric(ops)?;
    let np = ops.vol.pts.len();
    let mut out: Vec<(DVector<f64>, Vec<[f64; 6]>, Vec<f64>)> = Vec::with_capacity(2);
    for k in 0..2 {
        let g_vol: Vec<_> = ops.vol.pts.iter().map(|x| cycle.component(k, *x)).collect();
        let g_bnd: Vec<_> = ops.bnd.pts.iter().map(|x| cycle.component(k, *x)).collect();
        let h = mat_sym(cycle.h[k]);
        let tr = (h[0] + h[1] + h[2]) / 3.0;
        let t = [h[0] - tr, h[1] - tr, h[2] - tr, h[3], h[4], h[5]].map(|v| m.tide_coef * v);
        let tide = vec![t; np];
        let tide_bnd = vec![t; ops.bnd.pts.len()];
        let tide_nn: Vec<f64> = ops.faces.normals.iter().map(|n| super::super::mechanics::quad_form(t, *n)).collect();
        let ld = Loads {
            g_eff: &g_vol,
            tide: &tide,
            g_bnd: &g_bnd,
            tide_bnd: &tide_bnd,
            tide_nn: &tide_nn,
            g_const,
            strength,
            dilatancy: None,
        };
        let f = assemble_f(ops, &ld);
        let (u, _) = solve_range(ops, rs, &f);
        let d = linalg::mul(&ops.rows.d, &u);
        // Unit viscosity: τ = 2 D(u), σ̂ = √2 D(u).
        let sigma = (0..np).map(|i| std::array::from_fn(|c| SQRT_2 * d[6 * i + c])).collect();
        let dp = pressure::uniform(ops, &ld, 1.0, &u);
        out.push((f, sigma, dp));
    }
    let (f1, s1, p1) = out.pop().expect("sin");
    let (f0, s0, p0) = out.pop().expect("cos");
    Ok(ElasticCycle { load: [f0, f1], sigma: [s0, s1], dp: [p0, p1] })
}

fn norm(v: &[f64; 6]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn dot(a: &[f64; 6], b: &[f64]) -> f64 {
    (0..6).map(|c| a[c] * b[c]).sum()
}

/// The bound of `L_peak` and the bracket of `L_sd` (Proposition 10.3.7.5).
#[derive(Clone, Copy, Debug, Default)]
pub struct CycleBracket {
    pub peak_lo: f64,
    pub shakedown_lo: f64,
    pub shakedown_hi: f64,
}

/// Proposition 10.3.7.5 about the mean bracket `mean` of the load `load`, with the cyclic
/// fields of `el` scaled by `scale`, the cohesion and friction of the frozen strength at the
/// rest pressure `p0`, and `phases` samples of the Melan bound.
#[allow(clippy::too_many_arguments)]
pub fn cycle_bracket(
    ops: &Operators,
    mean: &Bracket,
    load: &DVector<f64>,
    el: &ElasticCycle,
    scale: f64,
    cohesion: &[f64],
    mu: f64,
    p0: &[f64],
    p_min: f64,
    phases: usize,
) -> CycleBracket {
    let np = ops.vol.pts.len();
    let tau = |i: usize, sign: f64| cohesion[i] + mu * (p0[i] + sign * scale * el.dp_amplitude(i)).max(p_min);
    let u = DVector::from_column_slice(&mean.u);
    let d = linalg::mul(&ops.rows.d, &u);
    let eu = |i: usize| -> [f64; 6] { std::array::from_fn(|c| SQRT_2 * d[6 * i + c]) };
    let n_plus: f64 = (0..np).map(|i| ops.vol.w[i] * tau(i, 1.0) * mean.rates[i]).sum();
    let lbar = load.dot(&u);
    let (lc, ls) = (scale * el.load[0].dot(&u), scale * el.load[1].dot(&u));
    let koiter: f64 = (0..np)
        .map(|i| {
            let e = eu(i);
            ops.vol.w[i] * scale * dot(&el.sigma[0][i], &e).hypot(dot(&el.sigma[1][i], &e))
        })
        .sum();
    let melan = (0..np)
        .map(|i| {
            let s = &mean.stress[i];
            let peak = (0..phases)
                .map(|j| {
                    let e = el.sigma_at(i, Cycle::phase(j, phases));
                    norm(&std::array::from_fn(|c| s[c] + scale * e[c]))
                })
                .fold(0.0, f64::max);
            let lip = scale * (norm(&el.sigma[0][i]) + norm(&el.sigma[1][i])) * PI / phases as f64;
            (peak + lip) / tau(i, -1.0)
        })
        .fold(0.0, f64::max);
    CycleBracket {
        peak_lo: (lbar + lc.hypot(ls)) / n_plus,
        shakedown_lo: (lbar + koiter) / n_plus,
        shakedown_hi: melan,
    }
}

/// Options of Algorithm 10.3.7.6.
#[derive(Clone, Copy, Debug)]
pub struct CycleOptions {
    pub rigidity: f64,
    pub phases: usize,
    pub cycles_max: usize,
    pub tol: f64,
    pub tol_cycle: f64,
}

/// The steady cycle: the ratchet displacement per cycle (coefficients, m), the alternating
/// plastic shear per cycle and the utilization at the volume nodes, and the cycles run
/// (0: the cycle is elastic at every node of `trigger`).
pub struct SteadyCycle {
    pub cycles: usize,
    pub ratchet: Vec<f64>,
    pub ratcheting: Vec<bool>,
    pub range: Vec<f64>,
    pub utilization: Vec<f64>,
}

/// Initial-stress iterations allowed at one phase.
const RETURN_MAX: usize = 50;

/// Algorithm 10.3.7.6 on the mean stress `creep` (σ̂ per volume node) at the pressure
/// `p_bar`. The phase loop runs only if a node of `trigger` leaves the yield surface at
/// some phase; otherwise the cycle is elastic.
#[allow(clippy::too_many_arguments)]
pub fn steady_cycle(
    ops: &Operators,
    el: &ElasticCycle,
    creep: &[[f64; 6]],
    p_bar: &[f64],
    cohesion: &[f64],
    mu: f64,
    p_min: f64,
    trigger: &[bool],
    opt: &CycleOptions,
) -> Result<SteadyCycle, SimError> {
    let _span = super::super::timing::span("steady_cycle");
    let np = ops.vol.pts.len();
    let nm = opt.phases;
    let (_, rs) = pd_metric(ops)?;
    let tau: Vec<Vec<f64>> = (0..nm)
        .map(|j| {
            let m = Cycle::phase(j, nm);
            (0..np).map(|i| cohesion[i] + mu * (p_bar[i] + el.dp_at(i, m)).max(p_min)).collect()
        })
        .collect();
    let elastic: Vec<Vec<[f64; 6]>> = (0..nm)
        .map(|j| (0..np).map(|i| el.sigma_at(i, Cycle::phase(j, nm))).collect())
        .collect();
    let total = |j: usize, i: usize, r: &[f64]| -> [f64; 6] { std::array::from_fn(|c| creep[i][c] + elastic[j][i][c] + r[6 * i + c]) };
    let zero = vec![0.0; 6 * np];
    let utilization = |r: &[f64]| -> Vec<f64> {
        (0..np).map(|i| (0..nm).map(|j| norm(&total(j, i, r)) / tau[j][i]).fold(0.0, f64::max)).collect()
    };
    let leaves = (0..np).any(|i| trigger[i] && (0..nm).any(|j| norm(&total(j, i, &zero)) > tau[j][i] * (1.0 + opt.tol)));
    if !leaves {
        return Ok(SteadyCycle {
            cycles: 0,
            ratchet: vec![0.0; ops.n_u],
            ratcheting: vec![false; np],
            range: vec![0.0; np],
            utilization: utilization(&zero),
        });
    }
    let mu_e = opt.rigidity;
    // `u_ε` of a plastic strain field: K₁ u = Σ w √2 Dᵀ ε̂ (the modulus cancels in u).
    let displacement = |eps: &[f64]| -> DVector<f64> {
        let wv = DVector::from_iterator(6 * np, (0..6 * np).map(|k| ops.vol.w[k / 6] * SQRT_2 * eps[k]));
        solve_range(ops, rs, &linalg::tr_mul(&ops.rows.d, &wv)).0
    };
    // σ̂ᴿ = μ_e (E u_ε − ε̂).
    let residual = |eps: &[f64]| -> Vec<f64> {
        let d = linalg::mul(&ops.rows.d, &displacement(eps));
        (0..6 * np).map(|k| mu_e * (SQRT_2 * d[k] - eps[k])).collect()
    };
    let mut eps = vec![0.0; 6 * np];
    let mut r = vec![0.0; 6 * np];
    let tau_max = tau.iter().flatten().copied().fold(0.0, f64::max);
    let (mut cycles, mut range, mut start) = (0, vec![0.0; np], eps.clone());
    while cycles < opt.cycles_max {
        cycles += 1;
        start.copy_from_slice(&eps);
        let r_start = r.clone();
        range.iter_mut().for_each(|v| *v = 0.0);
        for j in 0..nm {
            for _ in 0..RETURN_MAX {
                let mut worst = 0.0_f64;
                for i in 0..np {
                    let s = total(j, i, &r);
                    let (len, t) = (norm(&s), tau[j][i]);
                    if len > t {
                        let k = (len - t) / (mu_e * len);
                        (0..6).for_each(|c| eps[6 * i + c] += k * s[c]);
                        range[i] += (len - t) / mu_e;
                        worst = worst.max((len - t) / t);
                    }
                }
                if worst <= opt.tol {
                    break;
                }
                r = residual(&eps);
            }
        }
        let change = r.iter().zip(&r_start).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        if change <= opt.tol_cycle * tau_max {
            break;
        }
    }
    let delta: Vec<f64> = eps.iter().zip(&start).map(|(a, b)| a - b).collect();
    let ratcheting = (0..np)
        .map(|i| {
            let d: [f64; 6] = std::array::from_fn(|c| delta[6 * i + c]);
            norm(&d) > opt.tol * tau[0][i] / mu_e
        })
        .collect();
    let ratchet = displacement(&delta).as_slice().to_vec();
    Ok(SteadyCycle { cycles, ratchet, ratcheting, range, utilization: utilization(&r) })
}
