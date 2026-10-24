//! One geometric increment of §10.4.3 with its outer acceptance (§10.4.4 step 8a): the cone
//! program, the routing of the relief (§2.8b), the trust-radius retries, the mass correction
//! of §2.14, and the loose-layer ledger of the promoted vertices (§2.8c). The creep step and
//! the surface settling of §10.4.4 both call it.

use std::borrow::Cow;

use crate::common::density::CauchyKernel;
use crate::common::error::SimError;
use crate::common::math::DVec3;

use super::body::Body;
use super::forward::Forward;
use super::params::Project;
use super::relax::{AcceptError, accept, mass_correction};
use super::runout::{self, Arrival, RunoutField};
use super::socp::{Kind, SocpInput, SocpOutput, SocpSeed, solve};
use super::transport::Spread;

/// Where the removed relief runs out: the frozen field, the vertex areas and spread, and
/// the free vertices of the slope faces (`V_slope`).
pub struct Routing<'a> {
    pub field: &'a RunoutField<'a>,
    pub area: &'a [f64],
    pub spread: &'a Spread,
    pub slope_vertex: &'a [bool],
}

/// The data of one increment. `scale` multiplies the accepted increment (`λ` of a creep
/// step, 1 when settling).
pub struct Relief<'a> {
    pub project: &'a Project,
    pub body: &'a Body,
    pub kernels: &'a [CauchyKernel],
    pub normals: &'a [DVec3],
    pub kind: &'a [Kind],
    pub d_pred: &'a [f64],
    pub weight: &'a [f64],
    pub gravity: &'a [DVec3],
    pub sin_phi: &'a [f64],
    pub slope_row: &'a [bool],
    pub jac: &'a [f64],
    pub lower: &'a [f64],
    pub upper: &'a [f64],
    pub mass_target: f64,
    pub scale: f64,
    pub routing: Option<Routing<'a>>,
    pub step: usize,
}

/// The accepted increment.
pub struct Relieved {
    pub out: SocpOutput,
    /// Largest applied displacement (m).
    pub dmax: f64,
    pub body: Body,
    pub fwd: Forward,
    pub mass_residual: f64,
    /// The prediction of the solve that was accepted (with the routed deposit).
    pub d_used: Vec<f64>,
    /// The routed relief and the routing gap.
    pub routed: Option<(runout::Routing, f64)>,
    pub trust: f64,
    pub retries: usize,
    /// The cone program missed its tolerance down to the trust floor; the geometry is kept.
    pub unresolved: bool,
    pub max_locked_step: f64,
    pub max_deposit_miss: f64,
}

/// Solve, route, build and accept one increment, shrinking the trust radius on a rejected
/// attempt; time does not advance on a rejected attempt.
pub fn relieve(r: &Relief, fwd: &Forward, seed: &mut Option<SocpSeed>) -> Result<Relieved, SimError> {
    let g = &r.project.geometry;
    let topo = &r.body.topo;
    let nv = topo.n_vertices();
    let r_mean = r.body.r_mean();
    let mut trust = g.trust0 * r_mean;
    let mut retries = 0usize;
    let mut rejected: Option<String> = None;
    let build = |positions: Vec<DVec3>, from: &Body| {
        let _span = super::timing::span("geometry_build");
        Body::new(topo.clone(), positions, from.bulk, r.kernels.to_vec(), from.rho_s.clone())
    };
    loop {
        if trust < g.trust_min * r_mean {
            let why = rejected.as_deref().unwrap_or("rejected");
            if !why.starts_with("cone solve") {
                return Err(SimError::geometry(format!(
                    "step {}: no acceptable geometry down to Δ = {:.3e} m ({why})",
                    r.step,
                    trust / g.trust_shrink,
                )));
            }
            // §10.4.3: the cone program itself, not the geometry, failed at the floor.
            // The geometry is kept and the increment is flagged.
            eprintln!("step {}: geometry kept, {why}", r.step);
            let body = build(r.body.positions.clone(), r.body)?;
            let fwd = Forward::bind(&body, r.project.g)?;
            let mass_residual = (fwd.moments.mass - r.mass_target).abs() / r.mass_target;
            return Ok(Relieved {
                out: SocpOutput::unresolved(nv, topo.n_faces()),
                dmax: 0.0,
                body,
                fwd,
                mass_residual,
                d_used: r.d_pred.to_vec(),
                routed: None,
                trust,
                retries,
                unresolved: true,
                max_locked_step: 0.0,
                max_deposit_miss: 0.0,
            });
        }
        let mut reject = |why: String, trust: &mut f64, retries: &mut usize| {
            rejected = Some(why);
            *trust *= g.trust_shrink;
            *retries += 1;
        };
        // One cone solve. A warm seed is never discarded, and a mass or volume miss does
        // not start PDHG again.
        let inp = SocpInput {
            body: r.body,
            kind: r.kind,
            d_pred: r.d_pred,
            weight: r.weight,
            gravity: r.gravity,
            sin_phi: r.sin_phi,
            slope_row: r.slope_row,
            mass_jac: r.jac,
            mass_residual: r.mass_target - fwd.moments.mass,
            mass: r.mass_target,
            trust,
            eps_n: g.eps_n,
            lower: r.lower,
            upper: r.upper,
            varrho: g.varrho,
            warm: seed.as_ref(),
        };
        let out = match solve(&inp) {
            Ok(out) => out,
            // A cone solve that misses KKT_TOL is a rejected attempt like a bad geometry:
            // the smaller box is a better-conditioned program.
            Err(err) if err.kind == "SolverFailure" => {
                *seed = None;
                reject(format!("cone solve: {}", err.detail), &mut trust, &mut retries);
                continue;
            }
            Err(err) => return Err(err),
        };
        // §2.8b: route the mass the relief removed from the slope vertices, then solve
        // again from the first solution with the deposit added to the prediction.
        let removed: Vec<f64> = match &r.routing {
            Some(rt) => (0..nv)
                .map(|i| if rt.slope_vertex[i] { (r.d_pred[i] - out.d[i]).max(0.0) } else { 0.0 })
                .collect(),
            None => Vec::new(),
        };
        let (out, kind_used, d_used, routed) = match (&r.routing, removed.iter().any(|m| *m > 0.0)) {
            (Some(rt), true) => {
                let routed = {
                    let _span = super::timing::span("relief_routing");
                    let exterior = |p: &[DVec3]| fwd.eval_exterior(p);
                    runout::route_relief(rt.field, rt.area, r.jac, &removed, rt.spread, Some(&exterior))?
                };
                let mut d2 = r.d_pred.to_vec();
                let mut k2 = r.kind.to_vec();
                for j in 0..nv {
                    if routed.delta[j] > 0.0 {
                        d2[j] += routed.delta[j];
                        if k2[j] == Kind::Locked {
                            k2[j] = Kind::Deposit;
                        }
                    }
                }
                let second = SocpInput {
                    kind: &k2,
                    d_pred: &d2,
                    warm: Some(&out.seed),
                    ..inp
                };
                match solve(&second) {
                    Ok(out2) => {
                        let first: f64 = routed.masses.iter().sum();
                        let again: f64 = (0..nv)
                            .filter(|&i| rt.slope_vertex[i])
                            .map(|i| r.jac[i] * (d2[i] - out2.d[i]).max(0.0))
                            .sum();
                        let gap = if first > 0.0 { (again - first).abs() / first } else { 0.0 };
                        (out2, Cow::Owned(k2), d2, Some((routed, gap)))
                    }
                    Err(err) if err.kind == "SolverFailure" => {
                        *seed = None;
                        reject(format!("cone solve (routed): {}", err.detail), &mut trust, &mut retries);
                        continue;
                    }
                    Err(err) => return Err(err),
                }
            }
            _ => (out, Cow::Borrowed(r.kind), r.d_pred.to_vec(), None),
        };
        *seed = Some(out.seed.clone());
        let (max_locked_step, max_deposit_miss) = class_steps(&kind_used, &d_used, &out.d);
        // 8a: X⁽ⁿ⁺¹⁾ = X⁽ⁿ⁾ + scale · d_* N (G4).
        let moved: Vec<DVec3> = (0..nv)
            .map(|i| r.body.positions[i] + r.normals[i] * (r.scale * out.d[i]))
            .collect();
        let b = match build(moved, r.body) {
            Ok(b) => b,
            Err(e) => {
                reject(format!("body construction: {e}"), &mut trust, &mut retries);
                continue;
            }
        };
        let f = match Forward::bind(&b, r.project.g) {
            Ok(f) => f,
            Err(e) => {
                reject(format!("{e}"), &mut trust, &mut retries);
                continue;
            }
        };
        let dmax = out.d.iter().fold(0.0_f64, |m, v| m.max(v.abs())) * r.scale;
        // Mass misses get one mobile-set correction (§2.14). That correction is not a
        // second cone solve; a residual that remains is recorded.
        let (body, fwd, mass_residual) = match accept(&f, r.mass_target, g.eps_mass) {
            Ok(res) => (b, f, res),
            Err(AcceptError::Geometry(why)) => {
                reject(why, &mut trust, &mut retries);
                continue;
            }
            Err(AcceptError::Mass(res)) => {
                let dm = f.moments.mass - r.mass_target;
                let mut chosen = (b, f, res);
                if let Some(rr) = mass_correction(&chosen.0.positions, r.normals, r.kind, r.jac, dm, trust) {
                    if let Ok(cb) = build(rr, &chosen.0) {
                        if let Ok(cf) = Forward::bind(&cb, r.project.g) {
                            match accept(&cf, r.mass_target, g.eps_mass) {
                                Ok(cr) | Err(AcceptError::Mass(cr)) => chosen = (cb, cf, cr),
                                Err(AcceptError::Geometry(_)) => {}
                            }
                        }
                    }
                }
                chosen
            }
        };
        return Ok(Relieved {
            out,
            dmax,
            body,
            fwd,
            mass_residual,
            d_used,
            routed,
            trust,
            retries,
            unresolved: false,
            max_locked_step,
            max_deposit_miss,
        });
    }
}

/// §2.8c: the relief of a promoted vertex comes out of its loose layer and is laid down
/// where the routing of §2.8b sends it, so `Σ A ρ_s` is unchanged. Returns the debited
/// and credited masses per vertex.
pub fn layer_ledger(
    promoted: &[bool],
    relieved: &Relieved,
    scale: f64,
    jac: &[f64],
    rho_s: &[f64],
    area: &[f64],
    spread: &Spread,
) -> (Vec<f64>, Vec<f64>) {
    let nv = promoted.len();
    let d = &relieved.out.d;
    let debit: Vec<f64> = (0..nv)
        .map(|i| {
            if promoted[i] && d.len() == nv {
                // Capped by the carried layer plus what arrives on it in the increment.
                let du = relieved.d_used[i];
                let m = scale * jac[i] * (du - d[i]).max(0.0);
                m.min(rho_s[i] * area[i] + scale * jac[i] * du.max(0.0))
            } else {
                0.0
            }
        })
        .collect();
    if debit.iter().all(|m| *m <= 0.0) {
        return (debit, vec![0.0; nv]);
    }
    let per_area: Vec<f64> = (0..nv).map(|i| debit[i] / area[i]).collect();
    let routed: Vec<Arrival> = relieved
        .routed
        .as_ref()
        .map(|(r, _)| r.arrivals.iter().filter(|a| promoted[a.source]).cloned().collect())
        .unwrap_or_default();
    let laid = spread.spread(&runout::moved_source(area, &per_area, &routed));
    let credit = (0..nv).map(|j| laid[j] * area[j]).collect();
    (debit, credit)
}

/// The loose-mass bound of §2.8c: a promoted vertex cannot cut below its carried layer plus
/// the step's deposit, `d_i ≥ min(d_pred,i, 0) − ρ_s,i A_i / (DM)_i ≤ 0`, so `d = 0` stays
/// feasible; `−∞` elsewhere.
pub fn floor_of(promoted: &[bool], d_pred: &[f64], rho_s: &[f64], area: &[f64], jac: &[f64]) -> Vec<f64> {
    (0..promoted.len())
        .map(|i| {
            if promoted[i] && jac[i] > 0.0 {
                d_pred[i].min(0.0) - rho_s[i] * area[i] / jac[i]
            } else {
                f64::NEG_INFINITY
            }
        })
        .collect()
}

/// Per-vertex bounds of the cone program: the clearance rows `−θ_g t⁻ ≤ d ≤ θ_g t⁺` of
/// (G4), with `t^±` the distance along `±N_i` to the next crossing of `Γ`, and the
/// loose-mass `floor`.
pub fn clearance(
    body: &Body,
    normals: &[DVec3],
    mesh: &crate::rtfp::mesh::BodyMesh,
    theta_g: f64,
    floor: &[f64],
) -> Result<(Vec<f64>, Vec<f64>), SimError> {
    let _span = super::timing::span("clearance");
    let nv = normals.len();
    let eps = 1e-7 * body.r_mean();
    let mut batch = mesh.new_batch();
    for i in 0..nv {
        for s in [1.0, -1.0] {
            batch.push(body.positions[i] + normals[i] * (s * eps), normals[i] * s, None);
        }
    }
    let hits = mesh.trace(&batch, true)?;
    let mut buf = Vec::new();
    let mut dist = |k: usize| {
        buf.clear();
        hits.distances_into(k, &mut buf);
        buf.first().map_or(f64::INFINITY, |t| t + eps)
    };
    let mut lower = Vec::with_capacity(nv);
    let mut upper = Vec::with_capacity(nv);
    for i in 0..nv {
        upper.push(theta_g * dist(2 * i));
        lower.push((-theta_g * dist(2 * i + 1)).max(floor[i]));
    }
    Ok((lower, upper))
}

/// Largest step of a locked vertex and largest miss of a deposition vertex.
fn class_steps(kind: &[Kind], d_pred: &[f64], d: &[f64]) -> (f64, f64) {
    let (mut locked, mut deposit) = (0.0_f64, 0.0_f64);
    for i in 0..kind.len().min(d.len()) {
        match kind[i] {
            Kind::Locked => locked = locked.max(d[i].abs()),
            Kind::Deposit => deposit = deposit.max((d[i] - d_pred[i]).abs()),
            Kind::Mobile => {}
        }
    }
    (locked, deposit)
}
