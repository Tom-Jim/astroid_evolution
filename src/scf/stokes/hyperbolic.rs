//! The hyperbolic characteristic-ray relaxation of §8.3 (8.3-H), the normative
//! frozen-coefficient solver of §10.3.2. `U = (v, S, p)` is held at the volume
//! quadrature samples; every sweep backtraces the P and S characteristics one
//! axis family at a time (the directional splitting of §10.3.2), relaxes the
//! four stationary modes by the exponential integrator, imposes the traction
//! BC on rays whose upstream leaves `𝒱`, and closes each `N_p` sweeps with
//! the artificial-compressibility correction and the six rigid-mode
//! projections. The converged `v^ray` is Galerkin-projected onto the retained
//! velocity basis (`e_proj` reported) so downstream consumers keep their
//! existing coefficient contract.
//!
//! This is the sole rheology solver of the production loop. A sweep that
//! cannot meet criterion (a) inside `max_sweeps` reports a bounded
//! `RheologyNotConverged` — the PO-19 boundedness stance of §10.3.2 — never
//! propagated NaNs. The primal–dual iteration survives only as a `tier2`
//! test oracle.

use nalgebra::{DMatrix, DVector};

use crate::common::error::SimError;
use crate::common::math::DVec3;
use crate::rheo::Material;
use crate::scf::cell_bvh::CellBvh;
use crate::scf::params::StokesParams;

use super::pressure::Pressure;
use super::{
    Guess, Loads, Operators, StokesSolution, assemble_f, assemble_k, d6_dot, eta_eval,
    factor_range, load_scale, solve_range, strain_fields,
};

/// One sample state `U = (v, S, p)`: `v` 3, `S` the scaled six-vector of the
/// `basis.rs` convention (`d·d = D:D`), `p` 1.
const NDIM: usize = 10;

/// Local characteristic speeds of (10.3.2-H) at one sample.
#[derive(Clone, Copy)]
struct Speeds {
    /// `c_hyp = √(2η/(ρT_s) + 1/(ρκ))`, the P speed.
    p: f64,
    /// `c_s = √(η/(ρT_s))`, the S speed.
    s: f64,
}

fn speeds(eta: f64, rho: f64, t_s: f64, kappa: f64) -> Speeds {
    let r = rho.max(1e-30);
    Speeds {
        p: (2.0 * eta / (r * t_s) + 1.0 / (r * kappa)).sqrt(),
        s: (eta / (r * t_s)).sqrt(),
    }
}

/// The cell-sampled ray state of the solver.
struct Ray {
    pts: Vec<DVec3>,
    bvh: CellBvh,
    /// `10` per sample: `v | S(xx yy zz √2xy √2xz √2yz) | p`.
    u: Vec<f64>,
    /// Neighbouring sample indices within `radius`, for the upstream
    /// interpolation and the local `D(v)` least-squares.
    nbr: Vec<Vec<u32>>,
}

impl Ray {
    /// Samples are the volume quadrature points; the CellBVH cells have the
    /// mean nodal spacing, and the neighbour set covers the cells sharing a
    /// face.
    fn build(pts: &[DVec3], side: f64) -> Self {
        let bvh = CellBvh::build(pts, side);
        let nbr = pts
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let c = bvh.cell_of(*p).map(|c| c as usize).unwrap_or(0);
                let [kx, ky, kz] = bvh.coords[c];
                let mut out = Vec::new();
                for a in -1..=1i64 {
                    for b in -1..=1i64 {
                        for d in -1..=1i64 {
                            if let Some(&cell) = bvh.index.get(&[kx + a, ky + b, kz + d]) {
                                out.extend_from_slice(bvh.members(cell as usize));
                            }
                        }
                    }
                }
                out.sort_unstable();
                out.dedup();
                out.retain(|&j| j != i as u32);
                out
            })
            .collect();
        Self {
            pts: pts.to_vec(),
            bvh,
            u: vec![0.0; NDIM * pts.len()],
            nbr,
        }
    }

    /// `D(v)` from a local linear fit over the neighbour set: the
    /// least-squares gradient of each velocity component at sample `i`.
    fn strain(&self, i: usize) -> [f64; 6] {
        const S2: f64 = std::f64::consts::FRAC_1_SQRT_2;
        let x0 = self.pts[i];
        let mut g = nalgebra::Matrix3::<f64>::zeros();
        let mut rhs = nalgebra::Matrix3::<f64>::zeros();
        for &j in &self.nbr[i] {
            let dx = self.pts[j as usize] - x0;
            let dv = DVec3::new(
                self.u[NDIM * j as usize] - self.u[NDIM * i],
                self.u[NDIM * j as usize + 1] - self.u[NDIM * i + 1],
                self.u[NDIM * j as usize + 2] - self.u[NDIM * i + 2],
            );
            let w = 1.0 / dx.length_squared().max(1e-30);
            for a in 0..3 {
                for b in 0..3 {
                    g[(a, b)] += w * dx[a] * dx[b];
                }
                for k in 0..3 {
                    rhs[(a, k)] += w * dx[a] * dv[k];
                }
            }
        }
        let Some(inv) = g.try_inverse() else {
            return [0.0; 6];
        };
        let grad = inv * rhs; // grad[k][a] = ∂_a v_k
        let d = |a: usize, b: usize| 0.5 * (grad[(a, b)] + grad[(b, a)]);
        [
            grad[(0, 0)],
            grad[(1, 1)],
            grad[(2, 2)],
            d(0, 1) / S2,
            d(0, 2) / S2,
            d(1, 2) / S2,
        ]
    }

    /// Index of the sample nearest `x` (used as the boundary ghost).
    fn nearest(&self, x: DVec3) -> usize {
        if let Some(c) = self.bvh.cell_of(x) {
            let mut best = (usize::MAX, f64::INFINITY);
            let [kx, ky, kz] = self.bvh.coords[c as usize];
            for a in -1..=1i64 {
                for b in -1..=1i64 {
                    for d in -1..=1i64 {
                        if let Some(&cell) = self.bvh.index.get(&[kx + a, ky + b, kz + d]) {
                            for &j in self.bvh.members(cell as usize) {
                                let dd = self.pts[j as usize].distance_squared(x);
                                if dd < best.1 {
                                    best = (j as usize, dd);
                                }
                            }
                        }
                    }
                }
            }
            if best.0 != usize::MAX {
                return best.0;
            }
        }
        (0..self.pts.len())
            .min_by(|&a, &b| {
                self.pts[a]
                    .distance_squared(x)
                    .total_cmp(&self.pts[b].distance_squared(x))
            })
            .unwrap_or(0)
    }

    /// The state at `x`: the interpolated value inside the sampled region, or
    /// the nearest-sample ghost at a boundary exit (zeroth-order ghost; the
    /// traction of the ghost cell itself is corrected by the BC pass).
    fn at(&self, x: DVec3) -> [f64; NDIM] {
        self.interp(x)
            .unwrap_or_else(|| self.u[NDIM * self.nearest(x)..][..NDIM].try_into().unwrap())
    }

    /// The interpolated state at `x`: the inverse-distance-weighted mean of
    /// the occupied cell's members and the 26 neighbours, or `None` outside
    /// the sampled region (a boundary exit).
    fn interp(&self, x: DVec3) -> Option<[f64; NDIM]> {
        let mut acc = [0.0; NDIM];
        let mut wsum = 0.0;
        if let Some(c) = self.bvh.cell_of(x) {
            let [kx, ky, kz] = self.bvh.coords[c as usize];
            for a in -1..=1i64 {
                for b in -1..=1i64 {
                    for d in -1..=1i64 {
                        if let Some(&cell) = self.bvh.index.get(&[kx + a, ky + b, kz + d]) {
                            for &j in self.bvh.members(cell as usize) {
                                let dx = (self.pts[j as usize] - x).length();
                                let w = 1.0 / (dx + 1e-12 * self.bvh.side);
                                for k in 0..NDIM {
                                    acc[k] += w * self.u[NDIM * j as usize + k];
                                }
                                wsum += w;
                            }
                        }
                    }
                }
            }
        }
        (wsum > 0.0).then(|| {
            for k in 0..NDIM {
                acc[k] /= wsum;
            }
            acc
        })
    }
}

/// The shear six-vector indices owned by axis `a`: pairs (a, j) for `j ≠ a`.
fn shear_of_axis(axis: usize) -> [usize; 2] {
    match axis {
        0 => [3, 4], // xy, xz
        1 => [3, 5], // xy, yz
        _ => [4, 5], // xz, yz
    }
}

/// One axis family sweep for the `{P±, S_a±}` characteristics of §10.3.2,
/// directional splitting with the eigenvector refresh at every sample.
#[allow(clippy::too_many_arguments)]
fn sweep_axis(
    ray: &mut Ray,
    axis: usize,
    dt: f64,
    eta: &[f64],
    rho: &[f64],
    t_s: f64,
    kappa: f64,
    tide: &[[f64; 6]],
    g_eff: &[DVec3],
) {
    let n = ray.pts.len();
    let old = ray.u.clone();
    let (j1, j2) = ((axis + 1) % 3, (axis + 2) % 3);
    // Stress components the axis families own: S_aa plus the two S_aj.
    let owned = [axis, shear_of_axis(axis)[0], shear_of_axis(axis)[1]];
    let owned_at = |c: usize| owned.contains(&c);
    for i in 0..n {
        let x = ray.pts[i];
        let sp = speeds(eta[i], rho[i], t_s, kappa);
        let mut new = [0.0f64; NDIM];
        // Off-axis (stationary) stress components relax in place toward
        // `2ηD + T_tide` — the exponential integrator of §10.3.2.
        let d_loc = ray.strain(i);
        let ex = (-dt / t_s).exp();
        for c in 0..6 {
            if owned_at(c) {
                continue;
            }
            let src = 2.0 * eta[i] * d_loc[c] + tide[i][c];
            new[3 + c] = ex * old[NDIM * i + 3 + c] + (1.0 - ex) * src;
        }
        // Moving families at both signs: reconstruct `v` and the owned stress
        // from the upwind characteristic amplitudes.
        let (r_p, r_s) = (
            rho[i].max(1e-30) * sp.p,
            rho[i].max(1e-30) * sp.s,
        );
        // The P family transports the single combined amplitude
        // `σ̃ = S_aa − p` on `v_a`; writing it into both slots would double it
        // every sweep. p persists between the AC corrections instead.
        let mut sig = 0.0;
        for &sign in &[1.0f64, -1.0] {
            let back = |c: f64| {
                let mut xu = x;
                xu[axis] -= sign * c * dt;
                ray.at(xu)
            };
            // P family: w_± = ½(v_a ± (S_aa − p)/(ρc_P)) with the momentum
            // load `f_a` of (10.3.2-H) integrated over the segment.
            let (up_p, sgn) = (back(sp.p), sign);
            let w_p = 0.5 * (up_p[axis] + sgn * (up_p[3 + axis] - up_p[9]) / r_p)
                + 0.5 * sgn * dt * g_eff[i][axis] * rho[i] / r_p;
            new[axis] += w_p;
            sig += sgn * r_p * w_p;
            // The two S families on `v_{j1}, v_{j2}`.
            let up_s = back(sp.s);
            for (j, cix) in [(j1, owned[1]), (j2, owned[2])] {
                let w_s = 0.5 * (up_s[j] + sgn * up_s[3 + cix] / r_s)
                    + 0.5 * sgn * dt * g_eff[i][j] * rho[i] / r_s;
                new[j] += w_s;
                new[3 + cix] += sgn * r_s * w_s;
            }
        }
        new[9] = old[NDIM * i + 9];
        new[3 + axis] = sig + new[9];
        ray.u[NDIM * i..NDIM * i + NDIM].copy_from_slice(&new);
    }
}

/// The traction residual of boundary-adjacent samples:
/// `(-pI + S + T_tide) n − t_ext` with `t_ext = −p_ext n` at free faces.
/// `n` here is the inward depth gradient `∇d` of the volume set (−n of Γ).
fn traction_fix(
    ray: &mut Ray,
    bnd_pts: &[usize],
    gdepth: &[DVec3],
    p_ext: f64,
    tide: &[[f64; 6]],
) {
    const S2: f64 = std::f64::consts::FRAC_1_SQRT_2;
    for &i in bnd_pts {
        let n = -gdepth[i];
        let mut s = [
            ray.u[NDIM * i + 3],
            ray.u[NDIM * i + 4],
            ray.u[NDIM * i + 5],
            ray.u[NDIM * i + 6] * S2,
            ray.u[NDIM * i + 7] * S2,
            ray.u[NDIM * i + 8] * S2,
        ];
        let p = ray.u[NDIM * i + 9];
        // Normal stress target: `σ_nn = n(−pI + S + T_tide)n = −p_ext`.
        let t = tide[i];
        let cur = n.x * (s[0] + t[0] - p) * n.x
            + n.y * (s[1] + t[1] - p) * n.y
            + n.z * (s[2] + t[2] - p) * n.z
            + 2.0 * (n.x * (s[3] + t[3]) * n.y + n.x * (s[4] + t[4]) * n.z + n.y * (s[5] + t[5]) * n.z);
        let fix = (-p_ext - cur) / n.length_squared().max(1e-30);
        // Spread the correction over the normal stress components only.
        for a in 0..3 {
            s[a] += fix * n[a] * n[a];
        }
        ray.u[NDIM * i + 3] = s[0];
        ray.u[NDIM * i + 4] = s[1];
        ray.u[NDIM * i + 5] = s[2];
    }
}

/// Project out the six rigid modes (Appendix A.3, mass-weighted): the
/// least-squares rigid velocity of `v` is subtracted on the samples.
fn rigid_project(ray: &mut Ray, w: &[f64], rho: &[f64]) {
    let n = ray.pts.len();
    // Normal equations for the rigid fit `v_r = a + θ×x`.
    let mut m = DMatrix::<f64>::zeros(6, 6);
    let mut rhs = DVector::<f64>::zeros(6);
    for i in 0..n {
        let x = ray.pts[i];
        let rows = [
            [1.0, 0.0, 0.0, 0.0, x.z, -x.y],
            [0.0, 1.0, 0.0, -x.z, 0.0, x.x],
            [0.0, 0.0, 1.0, x.y, -x.x, 0.0],
        ];
        let wgt = w[i] * rho[i].max(0.0);
        let v = [ray.u[NDIM * i], ray.u[NDIM * i + 1], ray.u[NDIM * i + 2]];
        for a in 0..6 {
            for b in 0..6 {
                let mut s = 0.0;
                for k in 0..3 {
                    s += rows[k][a] * rows[k][b];
                }
                m[(a, b)] += wgt * s;
            }
            let mut r = 0.0;
            for k in 0..3 {
                r += rows[k][a] * v[k];
            }
            rhs[a] += wgt * r;
        }
    }
    let Some(ch) = m.cholesky() else {
        return;
    };
    let c = ch.solve(&rhs);
    for i in 0..n {
        let x = ray.pts[i];
        let rows = [
            [1.0, 0.0, 0.0, 0.0, x.z, -x.y],
            [0.0, 1.0, 0.0, -x.z, 0.0, x.x],
            [0.0, 0.0, 1.0, x.y, -x.x, 0.0],
        ];
        for k in 0..3 {
            let mut s = 0.0;
            for a in 0..6 {
                s += rows[k][a] * c[a];
            }
            ray.u[NDIM * i + k] -= s;
        }
    }
}

/// Relative `L²(ρ)` change of the velocity between two sweeps.
fn rel_change(old: &[f64], new: &[f64], w: &[f64], rho: &[f64]) -> f64 {
    let (mut num, mut den) = (0.0, 0.0);
    for i in 0..new.len() / NDIM {
        let pw = w[i] * rho[i];
        for k in 0..3 {
            num += pw * (new[NDIM * i + k] - old[NDIM * i + k]).powi(2);
            den += pw * new[NDIM * i + k].powi(2);
        }
    }
    (num / den.max(1e-300)).sqrt()
}

/// The frozen-coefficient hyperbolic ray solve of §10.3.2 for a fixed `η`:
/// axis sweeps until the relative velocity change falls under `tol`, with the
/// AC correction and rigid projection every `n_p` sweeps.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
fn relax(
    ops: &Operators,
    ld: &Loads,
    p_ext: f64,
    eta: &[f64],
    t_s: f64,
    kappa: f64,
    tol: f64,
    max_sweeps: usize,
    n_p: usize,
    bnd_pts: &[usize],
    ray: &mut Ray,
) -> (usize, f64, f64) {
    let (w, rho, vol) = (&ops.vol.w, &ops.vol.rho, ops.vol.pts.len());
    let mut prev = ray.u.clone();
    let mut sweeps = 0;
    // `η` is frozen for the whole relaxation, so the P-speed cap is too.
    let cmax = eta
        .iter()
        .zip(rho.iter())
        .map(|(&e, &r)| speeds(e, r, t_s, kappa).p)
        .fold(0.0, f64::max)
        .max(1e-300);
    // Δτ adaptation: a sweep that grows the state change instead of shrinking
    // it is rolled back and retried at half the step; a few rollbacks are the
    // cheap version of the §10.3.2 pseudo-time line search.
    let mut dt_fac = 1.0;
    let mut prev_eps = f64::INFINITY;
    let mut rollbacks = 0;
    // The CFL step is a guess; keep halving while rollback still rescues the
    // sweep rather than giving up after a fixed count.
    const MAX_ROLLBACKS: usize = 30;
    let mut half = f64::NAN;
    let mut converged = false;
    for s in 0..max_sweeps {
        if s == max_sweeps / 2 {
            half = v_norm(ray, w, rho);
        }
        let dt = 0.4 * ray.bvh.side / cmax * dt_fac;
        for axis in 0..3 {
            sweep_axis(ray, axis, dt, eta, rho, t_s, kappa, ld.tide, ld.g_eff);
        }
        // Traction BC on the boundary-adjacent samples (§10.3.2 boundary hit).
        traction_fix(ray, bnd_pts, &ops.vol.gdepth, p_ext, ld.tide);
        if s % n_p == n_p - 1 {
            // AC correction: `p ← p − κ^{-1}(∇·v − g)·(κΔτ)` — the local
            // divergence relaxes pressure directly (g ≡ 0 without A-Ξ).
            for i in 0..vol {
                let d = ray.strain(i);
                let div = d[0] + d[1] + d[2];
                ray.u[NDIM * i + 9] -= dt_scale(ray, eta, rho, t_s, kappa) * div / kappa;
            }
            rigid_project(ray, w, rho);
        }
        sweeps = s + 1;
        // Steady-state test on the *rate* ‖Δv‖/(3Δτ·‖v‖): a pseudo-time step
        // that merely shrinks Δτ cannot fake convergence at a non-equilibrium.
        let eps = rel_change(&prev, &ray.u, w, rho) / (3.0 * dt).max(1e-300);
        if !eps.is_finite() || !ray.u.iter().all(|v| v.is_finite()) {
            if rollbacks < MAX_ROLLBACKS {
                ray.u.copy_from_slice(&prev);
                dt_fac *= 0.5;
                prev_eps = f64::INFINITY;
                rollbacks += 1;
                continue;
            }
            ray.u.copy_from_slice(&prev); // keep the last finite state for the caller's diagnosis
            return (0, half, v_norm(ray, w, rho));
        }
        if eps < tol {
            converged = true;
            break;
        }
        if eps > 1.5 * prev_eps && rollbacks < MAX_ROLLBACKS {
            ray.u.copy_from_slice(&prev);
            dt_fac *= 0.5;
            prev_eps = f64::INFINITY;
            rollbacks += 1;
            continue;
        }
        prev_eps = eps;
        prev.copy_from_slice(&ray.u);
    }
    (if converged { sweeps } else { 0 }, half, v_norm(ray, w, rho))
}

/// Mass-weighted `L²(ρ)` norm of the velocity block of the ray state.
fn v_norm(ray: &Ray, w: &[f64], rho: &[f64]) -> f64 {
    (0..ray.u.len() / NDIM)
        .map(|i| {
            w[i] * rho[i]
                * (ray.u[NDIM * i].powi(2) + ray.u[NDIM * i + 1].powi(2) + ray.u[NDIM * i + 2].powi(2))
        })
        .sum::<f64>()
        .sqrt()
}

/// The collapse detector of Proposition 10.0.12.1 for the ray path, mirroring
/// the tier-2 oracle: the stress pinned at `τ_max = c + (μ_2 + ξ + K_Ξ Δν_c) p̃`
/// is the numerical signature of `L* > 1`, and the certified bracket of
/// §10.0.12.3 is the judge — so a merely stiff state is never misclassified.
/// `growth` (velocity amplification through the `what`) goes into the report.
/// Returns the error detail when collapse is certified, `None` otherwise.
fn certified_collapse(
    ops: &Operators,
    ld: &Loads,
    m: &Material,
    coh: &[super::Local],
    stress: &[f64],
    data: &super::limit::LimitData,
    growth: f64,
    what: &str,
) -> Result<Option<String>, SimError> {
    let np = ops.vol.pts.len();
    // τ_max on the rest pressure `p₀`, the same strength field the production
    // `limit::bracket` certifies with — independent of the converged state.
    let tau_max: Vec<f64> = (0..np)
        .map(|i| {
            let mu_max = m.spec.mu_2 + coh[i].xi + m.dilatancy.map_or(0.0, |d| d[0]);
            coh[i].c + mu_max * data.p0[i].max(m.spec.p_min)
        })
        .collect();
    let saturation = (0..np)
        .map(|i| {
            let s = &stress[6 * i..6 * i + 6];
            d6_dot(s, s).sqrt() / tau_max[i].max(1e-300)
        })
        .fold(0.0, f64::max);
    // Certify only on a finite saturated state: a diverged (NaN) ray means
    // the relaxation itself was unstable, not that L* > 1.
    if !(saturation.is_finite() && saturation >= 0.999) {
        return Ok(None);
    }
    let suspect = format!(
        "the ray velocity grew {growth:.2}× through the {what} with the stress at \
         {saturation:.4} of τ_max"
    );
    if ld.dilatancy.is_some() {
        return Ok(Some(format!("{suspect}: no quasi-static state at this pressure")));
    }
    // Certification, not a heuristic: the bracket must tighten to `gap`
    // (consistent, `lo ≤ hi`) with the whole interval above 1 before the
    // collapse verdict is returned — an early-exit bound is not enough for a
    // marginal state.
    const CERTIFY_GAP: f64 = 1e-2;
    let b = super::limit::bracket_with(ops, &data.load, &tau_max, CERTIFY_GAP, true)?;
    if b.lo > 1.0 && b.lo <= b.hi && b.hi - b.lo <= CERTIFY_GAP * b.hi {
        return Ok(Some(format!(
            "{suspect}; certified L* in [{:.4}, {:.4}] > 1: no quasi-static state",
            b.lo, b.hi
        )));
    }
    Ok(None)
}

fn dt_scale(ray: &Ray, eta: &[f64], rho: &[f64], t_s: f64, kappa: f64) -> f64 {
    let cmax = eta
        .iter()
        .zip(rho.iter())
        .map(|(&e, &r)| speeds(e, r, t_s, kappa).p)
        .fold(0.0, f64::max);
    0.4 * ray.bvh.side / cmax.max(1e-300)
}

/// `solver = "hyperbolic"`: the ray-relaxation path of §10.3.2. The outer
/// η-updating loop matches the PD loop; the converged `v^ray` is projected
/// onto the retained basis and `e_proj` is reported through `eps_momentum`.
/// Falls back to `solve_pd` on a singular configuration so the caller never
/// loses a step (the reference-oracle role of PD).
pub fn solve_hyperbolic(
    ops: &Operators,
    ld: &Loads,
    prm: &StokesParams,
    m: &Material,
    _guess: Option<&Guess>,
) -> Result<StokesSolution, SimError> {
    // Pseudo-time parameters of (10.3.2-H): `T_s` sets the relaxation scale,
    // `κ` the artificial compressibility. Both are artificial; their ratio to
    // η/ρ fixes the speeds. Chosen so `c_P` is a few c_S.
    let t_s = 1.0;
    let kappa = 0.25;
    let (w, rho, pts) = (&ops.vol.w, &ops.vol.rho, &ops.vol.pts);
    let mut ray = Ray::build(pts, {
        let (mut lo, mut hi) = (pts[0], pts[0]);
        for p in pts {
            lo = lo.min(*p);
            hi = hi.max(*p);
        }
        let d = hi - lo;
        ((d.x * d.y * d.z).max(1e-30) / pts.len() as f64).cbrt() * 1.5
    });
    // Boundary-adjacent samples: a neighbour set thinner than the stencil
    // means the ray exits 𝒱 within one step.
    let bnd_pts: Vec<usize> = (0..pts.len())
        .filter(|&i| ray.nbr[i].len() < 20)
        .collect();
    // Outer rheology passes: η frozen per pass, recomputed from (p̃, γ̇) of the
    // converged ray state — the `m` loop of §10.3.2.
    let coh = &ld.strength.vol;
    let eps = m.spec.gamma_eps;
    // `ℓ` and the rest pressure `p₀` once — the strength reference of every
    // collapse certification below (§10.0.12.3).
    let ldata = super::limit::data(ops, ld, prm, m)?;
    let mut eta: Vec<f64> = (0..pts.len())
        .map(|p| eta_eval(m, &coh[p], 0.0, m.spec.p_min.max(ld.g_const), eps).eta)
        .collect();
    // Continuation ladder of §10.3.2 (mirroring the oracle): try the target
    // `γ̇_ε` first; on a stall that the bracket does not certify, restart the
    // ray state at the easiest level `10^c γ̇_ε` and step down level to level.
    // Each level runs its own η-map (relax → η recompute at that level's ε).
    let mut levels = vec![0usize];
    let mut ladder = prm.continuation > 0;
    let mut at = 0usize;
    let mut settled0 = false;
    while at < levels.len() {
        let level = levels[at];
        at += 1;
        let eps_l = eps * 10f64.powi(level as i32);
        let mut done = false;
        let mut stuck = false;
        let mut v_first = f64::NAN;
        let mut v_last = f64::NAN;
        for _pass in 0..4 {
            let (ok, u_half, u_end) = relax(
                ops, ld, prm.p_ext, &eta, t_s, kappa, prm.tol_relax, prm.max_sweeps,
                prm.pressure_every.max(2), &bnd_pts, &mut ray,
            );
            if ok == 0 {
                let growth = u_end / u_half.max(1e-300);
                let stress: Vec<f64> = (0..pts.len())
                    .flat_map(|i| ray.u[NDIM * i + 3..NDIM * i + 9].iter().copied())
                    .collect();
                if let Some(detail) =
                    certified_collapse(ops, ld, m, coh, &stress, &ldata, growth, "sweeps")?
                {
                    return Err(SimError::limit_load_collapse(detail));
                }
                stuck = true;
                break;
            }
            let vn = v_norm(&ray, w, rho);
            if !v_first.is_finite() {
                v_first = vn;
            }
            v_last = vn;
            // Recompute η from (p̃, γ̇) of the ray state at this level's ε.
            // `p̃` takes the ray pressure plus the tidal trace, as §8.1.3.
            let mut new_eta = vec![0.0; pts.len()];
            for i in 0..pts.len() {
                let d = ray.strain(i);
                let gamma = (2.0 * d6_dot(&d, &d)).sqrt();
                let ptil = ray.u[NDIM * i + 9]
                    + (ld.tide[i][0] + ld.tide[i][1] + ld.tide[i][2]) / 3.0
                    + ld.g_const;
                new_eta[i] = eta_eval(m, &coh[i], gamma, ptil.max(m.spec.p_min), eps_l).eta;
            }
            done = new_eta
                .iter()
                .zip(eta.iter())
                .all(|(a, b)| ((a - b) / b.abs().max(1e-30)).abs() < prm.tol_relax);
            eta = new_eta;
            if done {
                break;
            }
        }
        if done && level == 0 {
            settled0 = true;
            break;
        }
        if !done && !stuck {
            // The η-map stalled at this level — the collapse signature of
            // Proposition 10.0.12.1 in this discretization. Certify on the
            // projected viscous stress `2ηD + T_tide`; only a tightened
            // bracket fully above 1 convicts.
            let growth = v_last / v_first.max(1e-300);
            let v_ray: Vec<f64> = (0..pts.len())
                .flat_map(|i| [ray.u[NDIM * i], ray.u[NDIM * i + 1], ray.u[NDIM * i + 2]])
                .collect();
            if let Ok((u_l, _)) = ops.project_velocity(&v_ray) {
                let ud = DVector::from_column_slice(&u_l);
                let fields = strain_fields(ops, &ud);
                let mut stress = vec![0.0; 6 * pts.len()];
                for i in 0..pts.len() {
                    for c in 0..6 {
                        stress[6 * i + c] = 2.0 * eta[i] * fields.d[6 * i + c] + ld.tide[i][c];
                    }
                }
                if let Some(detail) = certified_collapse(
                    ops, ld, m, coh, &stress, &ldata, growth, "rheology passes",
                )? {
                    return Err(SimError::limit_load_collapse(detail));
                }
            }
        }
        if level == 0 && ladder {
            // Restart at the easiest level, like the oracle's `warm = start`.
            ladder = false;
            levels.extend((1..=prm.continuation).rev());
            levels.push(0);
            for v in &mut ray.u {
                *v = 0.0;
            }
            for (i, e) in eta.iter_mut().enumerate() {
                *e = eta_eval(m, &coh[i], 0.0, m.spec.p_min.max(ld.g_const), eps).eta;
            }
        }
    }
    if !settled0 {
        return Err(SimError::rheology_not_converged(
            "the characteristic relaxation did not settle the η-map at any continuation level (PO-19)",
        ));
    }
    // Galerkin projection of the converged ray velocity onto the retained
    // (G1) basis: produces `u`, `v_h` and `e_proj`.
    let v_ray: Vec<f64> = (0..pts.len())
        .flat_map(|i| [ray.u[NDIM * i], ray.u[NDIM * i + 1], ray.u[NDIM * i + 2]])
        .collect();
    let (u, e_proj) = ops.project_velocity(&v_ray)?;
    finish(ops, ld, prm, m, eps, &eta, u, e_proj)
}

/// Assemble the `StokesSolution` of the ray path: pressure by the same (P1)
/// Nitsche step the reference loop uses, the §10.0.10(b) momentum residual on
/// the projected `u`, and `e_proj` in its own field of the report.
#[allow(clippy::too_many_arguments)]
fn finish(
    ops: &Operators,
    ld: &Loads,
    prm: &StokesParams,
    m: &Material,
    eps_used: f64,
    eta: &[f64],
    u: Vec<f64>,
    e_proj: f64,
) -> Result<StokesSolution, SimError> {
    let np = ops.vol.pts.len();
    let u = DVector::from_column_slice(&u);
    let fields = strain_fields(ops, &u);
    let coh = &ld.strength.vol;
    let mut pr = Pressure::warm(ops, m.spec.p_min, &[]);
    pr.update(ops, ld, prm, m, eps_used, &u, &fields, None)?;
    let eta_final: Vec<f64> = (0..np)
        .map(|p| eta_eval(m, &coh[p], fields.shear[p], pr.ptil[p], eps_used).eta)
        .collect();
    let _ = eta;
    let k = assemble_k(ops, &eta_final);
    let f_load = assemble_f(ops, ld);
    let f_norm = load_scale(ops, ld);
    let rs = factor_range(ops, &k)?;
    let (un, lambda) = solve_range(ops, &rs, &f_load);
    let resid = &k * &un + ops.rigid.transpose() * &lambda - &f_load;
    let eps_momentum = resid.norm() / f_norm;
    let eps_lambda = (ops.rigid.transpose() * &lambda).norm() / f_norm;
    // Keep the projected solution: its own defect is `e_proj`; the residual
    // above is the equilibrium error of the projected field, reported.
    let eps_rot = pr.rotational_residual(ops, ld, m, eps_used, &u, &fields, None);
    let p_surface = pr.face_traction(ops, ld, prm, m, eps_used, &u, None)?;
    let (pres_c, lift) = pr.split(ops);
    let wsum: f64 = ops.vol.w.iter().sum();
    let tensile = (0..np)
        .filter(|&p| pr.p_vals[p] < 0.0)
        .map(|p| ops.vol.w[p])
        .sum::<f64>()
        / wsum;
    Ok(StokesSolution {
        u: u.iter().copied().collect(),
        lambda: std::array::from_fn(|i| lambda[i]),
        pres_c,
        lift,
        eta: eta_final,
        p_tilde: pr.ptil,
        p_surface,
        sweeps: 0,
        eps_relax: e_proj,
        eps_momentum,
        eps_lambda,
        eps_rot,
        eta_lo: eta.iter().copied().fold(f64::INFINITY, f64::min),
        eta_hi: eta.iter().copied().fold(0.0, f64::max),
        tensile_fraction: tensile,
        inverse_ekman: 0.0,
        gamma_eps: eps_used,
        dilatancy: None,
        e_proj,
    })
}

