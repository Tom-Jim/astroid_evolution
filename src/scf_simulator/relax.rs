//! Supercritical-slope relaxation with crater protection.
//!
//! Relaxation is the geometric SOCP of §10.4.3 with a zero predictor (`d_pred = 0`).
//! The slope cone is imposed only on faces whose slack exceeds `tol_slack` and that
//! are not fully crater-locked. Every other vertex is in the locked set for that
//! sweep, so a subcritical face stays in static friction (`d = 0`). That is the
//! relaxation reading of `S_slope ⊆ A`: locked nodes do not receive the repose
//! proxy, and the mass row runs only inside the mobile patch. Crater features are
//! protected by the geometric detector (radius anomaly against the neighbour mean,
//! curvature above the configured quantile) or by observed floor/wall/rim masks.
//! Protected vertices are hard-locked (weight ≥ `lock_weight`) or made
//! proportionally stiffer. Mass is restored inside the mobile set, never by a
//! global rescale. A GPU, solver, or acceptance failure is returned as itself.

use std::path::Path;

use crate::common::math::DVec3;
use crate::rheo_friction::{RheoGpu, VelocityField};

use super::body::{Body, Topology};
use super::error::ScfError;
use super::forward::{Forward, deviator, effective_gravity, hessian_w};
use super::mechanics::{face_chain, fit_surface, quad_form};
use super::params::Project;
use super::socp::{Kind, SocpInput, mass_jacobian, solve};

/// Per-vertex protection weight in `[0, 1]`.
#[derive(Clone, Debug)]
pub struct Protection {
    pub weight: Vec<f64>,
    /// `"observed"` when the floor/wall/rim masks were read, `"geometric"` otherwise.
    pub identification: &'static str,
}

fn read_mask(
    path: &Path,
    nv: usize,
    nf: usize,
    body: &Body,
    name: &str,
) -> Result<Vec<f64>, ScfError> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| ScfError::io(format!("cannot read {name} {}: {e}", path.display())))?;
    let mut values = Vec::new();
    for (k, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let v: f64 = line.parse().map_err(|_| {
            ScfError::model_not_closed(format!("{name} line {} is not a weight", k + 1))
        })?;
        if !(0.0..=1.0).contains(&v) {
            return Err(ScfError::model_not_closed(format!(
                "{name} line {} is outside [0, 1]",
                k + 1
            )));
        }
        values.push(v);
    }
    if values.len() == nv {
        return Ok(values);
    }
    if values.len() == nf {
        // Face weights -> vertices (mean over the faces of the vertex).
        let mut acc = vec![0.0; nv];
        let mut cnt = vec![0.0; nv];
        for (f, t) in body.topo.faces.iter().enumerate() {
            for v in t {
                acc[*v as usize] += values[f];
                cnt[*v as usize] += 1.0;
            }
        }
        return Ok(acc.iter().zip(&cnt).map(|(a, c)| a / c).collect());
    }
    Err(ScfError::model_not_closed(format!(
        "{name} has {} weights; the mesh has {nv} vertices and {nf} faces",
        values.len()
    )))
}

/// Vertex normals (area weighted).
pub fn vertex_normals(body: &Body) -> Vec<DVec3> {
    let p = body.positions();
    let mut n = vec![DVec3::ZERO; p.len()];
    for t in &body.topo.faces {
        let [a, b, c] = t.map(|x| x as usize);
        let area = (p[b] - p[a]).cross(p[c] - p[a]);
        for v in [a, b, c] {
            n[v] = n[v] + area;
        }
    }
    n.iter()
        .map(|x| x.try_normalize().unwrap_or(DVec3::new(0.0, 0.0, 1.0)))
        .collect()
}

/// Crater-feature detector: radius anomaly against the neighbour mean, and curvature rank.
pub fn geometric_protection(project: &Project, body: &Body) -> Vec<f64> {
    let topo = &body.topo;
    let nv = topo.n_vertices();
    let rmax = body.radius.iter().cloned().fold(0.0, f64::max);
    let normals = vertex_normals(body);
    let mut w = vec![0.0_f64; nv];
    let mut curv = vec![0.0; nv];
    for i in 0..nv {
        let nb = &topo.vertex_neighbors[i];
        let mean = nb.iter().map(|j| body.radius[*j]).sum::<f64>() / nb.len() as f64;
        if (mean - body.radius[i]).abs() > project.relief.radius_anomaly * rmax {
            w[i] = 1.0;
        }
        curv[i] = nb
            .iter()
            .map(|j| 1.0 - normals[i].dot(normals[*j]).clamp(-1.0, 1.0))
            .sum::<f64>()
            / nb.len() as f64;
    }
    let mut sorted = curv.clone();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let q = project.relief.curvature_quantile;
    let pivot = sorted[((sorted.len() as f64 * q) as usize).min(sorted.len() - 1)];
    let peak = *sorted.last().unwrap();
    if peak > pivot {
        for i in 0..nv {
            if curv[i] > pivot {
                w[i] = w[i].max(((curv[i] - pivot) / (peak - pivot)).clamp(0.0, 1.0));
            }
        }
    }
    w
}

pub fn protection(project: &Project, body: &Body) -> Result<Protection, ScfError> {
    let topo = &body.topo;
    let nv = topo.n_vertices();
    let mut w = geometric_protection(project, body);
    let mut identification = "geometric";
    let r = &project.relief;
    if let (Some(fl), Some(wa), Some(ri)) = (&r.floor, &r.wall, &r.rim) {
        let nf = topo.n_faces();
        let floor = read_mask(&project.resolve(fl), nv, nf, body, "relief.floor")?;
        let wall = read_mask(&project.resolve(wa), nv, nf, body, "relief.wall")?;
        let rim = read_mask(&project.resolve(ri), nv, nf, body, "relief.rim")?;
        let combined: Vec<f64> = (0..nv).map(|i| floor[i].max(wall[i]).max(rim[i])).collect();
        // Spread the mask over one ring of neighbours (transition zone).
        for i in 0..nv {
            let mut acc = combined[i];
            let mut cnt = 1.0;
            for j in &topo.vertex_neighbors[i] {
                acc += combined[*j];
                cnt += 1.0;
            }
            w[i] = w[i].max(acc / cnt);
        }
        identification = "observed";
    }
    Ok(Protection {
        weight: w.iter().map(|v| v.clamp(0.0, 1.0)).collect(),
        identification,
    })
}

/// Failure kind returned by [`accept`].
#[derive(Clone, Debug)]
pub enum AcceptError {
    /// Positive volume, star-shape (`n·e_r ≥ c0`) or topology failure.
    /// The caller must shrink `Δ` and re-solve the SOCP (§2.13, §10.4.3).
    Geometry(String),
    /// Mass residual `|ℳ − M| > ε_M`.  The caller must apply a conservative
    /// correction in the mobile set; `Δ` must **not** be shrunk (§2.14).
    Mass(f64),
}

impl std::fmt::Display for AcceptError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AcceptError::Geometry(s) => write!(f, "geometry: {s}"),
            AcceptError::Mass(r) => write!(f, "mass residual {r:.3e}"),
        }
    }
}

/// Outer geometric acceptance (§10.4.4 step 8a): star-shape, positive volume, mass.
///
/// Returns `Ok(relative_mass_residual)` on pass.
/// Returns `Err(AcceptError::Geometry(_))` for mesh/topology failures that require trust shrink.
/// Returns `Err(AcceptError::Mass(resid))` for mass failures that require conservative correction.
pub fn accept(
    body: &Body,
    fwd: &Forward,
    mass_target: f64,
    c0: f64,
    eps_mass: f64,
) -> Result<f64, AcceptError> {
    if !(fwd.moments.volume > 0.0) {
        return Err(AcceptError::Geometry("non-positive volume".into()));
    }
    let geo = body.face_geometry();
    for (f, (_, n, _)) in geo.iter().enumerate() {
        let c = n.dot(body.topo.face_dirs[f]);
        if !(c >= c0) {
            return Err(AcceptError::Geometry(format!(
                "face {f}: n·e_r = {c:.4} < c0 = {c0}"
            )));
        }
    }
    let resid = (fwd.moments.mass - mass_target).abs() / mass_target;
    if resid > eps_mass {
        return Err(AcceptError::Mass(resid));
    }
    Ok(resid)
}

#[derive(Clone, Debug)]
pub struct SweepRecord {
    pub sweep: usize,
    pub max_slack: f64,
    pub n_supercritical: usize,
    pub trust: f64,
    pub trust_retries: usize,
    pub socp_iterations: u32,
    /// Vertices free in this sweep (incident to a supercritical face, not crater-locked).
    pub n_active: usize,
    /// Clarabel's own `solve_time`, in seconds.
    pub socp_seconds: f64,
    pub dropped_overhang: usize,
    pub mass_residual: f64,
    pub max_displacement: f64,
}

#[derive(Clone, Debug)]
pub struct RelaxReport {
    pub converged: bool,
    pub stalled: bool,
    pub identification: &'static str,
    pub n_protected: usize,
    pub n_locked: usize,
    /// Supercritical faces with all three vertices locked (preserved terrain).
    pub preserved_supercritical: usize,
    pub slack_before: f64,
    pub slack_after: f64,
    pub sweeps: Vec<SweepRecord>,
}

struct SlopeState {
    sin_phi: Vec<f64>,
    slack: Vec<f64>,
    gravity: Vec<DVec3>,
}

fn slope_state(
    project: &Project,
    gpu: &RheoGpu,
    body: &Body,
    fwd: &mut Forward,
    omega: f64,
) -> Result<SlopeState, ScfError> {
    let (surf, _) = fit_surface(project, body)?;
    fwd.eval_faces(body)?;
    let centres = fwd.face.points.clone();
    let dirs: Vec<DVec3> = centres.iter().map(|c| c.try_normalize().unwrap()).collect();
    let gravity: Vec<DVec3> = fwd
        .face
        .grad_u
        .iter()
        .zip(&centres)
        .map(|(g, x)| effective_gravity(*g, *x, omega))
        .collect();
    let hess_w: Vec<[f64; 6]> = fwd
        .face
        .hess_u
        .iter()
        .map(|h| hessian_w(*h, omega))
        .collect();
    let normals = surf.normals(&dirs);
    let pressure: Vec<f64> = hess_w
        .iter()
        .zip(&normals)
        .map(|(h, (_, n))| {
            project.stokes.p_ext + project.material.tide_coef * quad_form(deviator(*h), *n)
        })
        .collect();
    let (nodes, _) = face_chain(
        gpu,
        project,
        &surf,
        &VelocityField::zero(),
        &dirs,
        &gravity,
        hess_w,
        pressure,
    )?;
    Ok(SlopeState {
        sin_phi: nodes.iter().map(|n| n.sin_phi).collect(),
        slack: nodes.iter().map(|n| n.slack).collect(),
        gravity,
    })
}

/// Mobile set of one relaxation sweep.
///
/// A face enters `S_slope` when its slack exceeds `tol` and at least one vertex is
/// not already crater-locked. Vertices that do not touch such a face are locked,
/// which is the static-friction condition on subcritical faces. Crater-locked
/// vertices stay locked even if a neighbour face is supercritical.
fn friction_lock(
    topo: &Topology,
    protect: &[Kind],
    slack: &[f64],
    tol: f64,
) -> (Vec<Kind>, Vec<bool>) {
    use rayon::prelude::*;
    let nv = topo.n_vertices();
    let slope_row: Vec<bool> = (0..topo.n_faces())
        .into_par_iter()
        .map(|f| {
            slack[f] > tol
                && !topo.faces[f]
                    .iter()
                    .all(|v| protect[*v as usize] == Kind::Locked)
        })
        .collect();
    let mut touch = vec![false; nv];
    for (f, on) in slope_row.iter().enumerate() {
        if *on {
            for v in topo.faces[f] {
                let i = v as usize;
                if protect[i] != Kind::Locked {
                    touch[i] = true;
                }
            }
        }
    }
    let kind = (0..nv)
        .map(|i| {
            if protect[i] == Kind::Locked || !touch[i] {
                Kind::Locked
            } else {
                Kind::Mobile
            }
        })
        .collect();
    (kind, slope_row)
}

/// Relax the supercritical slopes of `body` under the spin `omega`.
pub fn relax(
    project: &Project,
    gpu: &RheoGpu,
    body: &Body,
    omega: f64,
) -> Result<(Body, RelaxReport), ScfError> {
    let topo = body.topo.clone();
    let nv = topo.n_vertices();
    let g = &project.geometry;
    let prot = protection(project, body)?;
    let kind: Vec<Kind> = prot
        .weight
        .iter()
        .map(|w| {
            if *w >= project.relief.lock_weight {
                Kind::Locked
            } else {
                Kind::Mobile
            }
        })
        .collect();
    let weight: Vec<f64> = prot
        .weight
        .iter()
        .map(|w| 1.0 / (1.0 - w).max(1e-3))
        .collect();
    let n_locked = kind.iter().filter(|k| **k == Kind::Locked).count();
    let n_protected = prot.weight.iter().filter(|w| **w > 0.0).count();
    let face_locked: Vec<bool> = topo
        .faces
        .iter()
        .map(|t| t.iter().all(|v| kind[*v as usize] == Kind::Locked))
        .collect();

    let mut current = body.clone();
    let mut fwd = Forward::bind(&current, project.g)?;
    let mass_target = fwd.moments.mass;
    let mut sweeps = Vec::new();
    let mut state = slope_state(project, gpu, &current, &mut fwd, omega)?;
    let measure = |st: &SlopeState| -> (f64, usize, usize) {
        let mut max_slack = 0.0_f64;
        let mut count = 0;
        let mut preserved = 0;
        for f in 0..st.slack.len() {
            if st.slack[f] > project.relief.tol_slack {
                if face_locked[f] {
                    preserved += 1;
                } else {
                    count += 1;
                    max_slack = max_slack.max(st.slack[f]);
                }
            }
        }
        (max_slack, count, preserved)
    };
    let (slack_before, _, _) = measure(&state);
    let mut converged = false;
    let mut stalled = false;
    let mut last_slack = slack_before;
    let mut stall_count = 0;

    for sweep in 0..project.relief.max_sweeps {
        let (max_slack, n_super, _) = measure(&state);
        if n_super == 0 {
            converged = true;
            break;
        }
        if state.slack.iter().any(|s| !s.is_finite())
            || state.sin_phi.iter().any(|s| !s.is_finite())
        {
            return Err(ScfError::solver(format!(
                "relaxation sweep {sweep}: friction chain returned a non-finite slope slack or sin φ"
            )));
        }
        let (sweep_kind, slope_row) =
            friction_lock(&topo, &kind, &state.slack, project.relief.tol_slack);
        let n_active = sweep_kind.iter().filter(|k| **k == Kind::Mobile).count();
        if n_active == 0 {
            return Err(ScfError::solver(format!(
                "relaxation sweep {sweep}: {n_super} supercritical faces but the mobile set is empty"
            )));
        }
        let r_mean = current.radius.iter().sum::<f64>() / nv as f64;
        let mut trust = g.trust0 * r_mean;
        let jac = mass_jacobian(&current, &current.density()?);
        let d_pred = vec![0.0; nv];
        let mut retries = 0;
        let mut last_failure;
        let shrink =
            |trust: &mut f64, retries: &mut usize, last_failure: &str| -> Result<(), ScfError> {
                *trust *= g.trust_shrink;
                *retries += 1;
                if *trust < g.trust_min * r_mean {
                    return Err(ScfError::trust_exhausted(format!(
                        "relaxation sweep {sweep}: {last_failure} down to Δ = {trust:.3e} m"
                    )));
                }
                Ok(())
            };
        let accepted = loop {
            let inp = SocpInput {
                body: &current,
                kind: &sweep_kind,
                d_pred: &d_pred,
                weight: &weight,
                gravity: &state.gravity,
                sin_phi: &state.sin_phi,
                slope_row: &slope_row,
                mass_jac: &jac,
                mass_residual: mass_target - fwd.moments.mass,
                mass: mass_target,
                trust,
                c0: g.c0,
                eps_n: g.eps_n,
                varrho: g.varrho,
            };
            let out = solve(&inp)?;
            let mut cand = current.clone();
            for i in 0..nv {
                cand.radius[i] += out.d[i];
            }
            let built = match Body::new(
                topo.clone(),
                cand.radius.clone(),
                cand.bulk,
                cand.kernels.clone(),
                cand.rho_s.clone(),
            ) {
                Ok(b) => b,
                Err(e) => {
                    last_failure = format!("geometry: {e}");
                    shrink(&mut trust, &mut retries, &last_failure)?;
                    continue;
                }
            };
            let bound = match Forward::bind(&built, project.g) {
                Ok(f) => f,
                Err(e) => {
                    last_failure = format!("geometry: {e}");
                    shrink(&mut trust, &mut retries, &last_failure)?;
                    continue;
                }
            };
            match accept(&built, &bound, mass_target, g.c0, g.eps_mass) {
                Ok(resid) => {
                    let dmax = out.d.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
                    break (built, bound, resid, out, dmax);
                }
                Err(AcceptError::Geometry(why)) => {
                    last_failure = format!("geometry: {why}");
                    shrink(&mut trust, &mut retries, &last_failure)?;
                }
                Err(AcceptError::Mass(resid)) => {
                    // §2.14: the nonlinear remainder is corrected in the mobile set.
                    // A degenerate Jacobian is reported; it is not turned into a
                    // smaller trust radius.
                    let dm = bound.moments.mass - mass_target;
                    let denom: f64 = sweep_kind
                        .iter()
                        .enumerate()
                        .filter(|(_, k)| **k == Kind::Mobile)
                        .map(|(i, _)| jac[i] * jac[i])
                        .sum();
                    if !(denom.is_finite() && denom > 0.0) {
                        return Err(ScfError::solver(format!(
                            "relaxation sweep {sweep}: mass residual {resid:.3e} and the mobile mass Jacobian is degenerate"
                        )));
                    }
                    let alpha = (-dm / denom).clamp(-trust, trust);
                    let mut rr = built.radius.clone();
                    for i in 0..nv {
                        if sweep_kind[i] == Kind::Mobile {
                            rr[i] += alpha * jac[i];
                        }
                    }
                    let corrected = Body::new(
                        topo.clone(),
                        rr,
                        built.bulk,
                        built.kernels.clone(),
                        built.rho_s.clone(),
                    )
                    .and_then(|cb| Forward::bind(&cb, project.g).map(|cf| (cb, cf)));
                    match corrected {
                        Ok((cb, cf)) => match accept(&cb, &cf, mass_target, g.c0, g.eps_mass) {
                            Ok(cr) => {
                                let dmax = out.d.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
                                break (cb, cf, cr, out, dmax);
                            }
                            Err(e) => {
                                last_failure = format!(
                                    "mass residual {resid:.3e} after mobile-set correction ({e})"
                                );
                                shrink(&mut trust, &mut retries, &last_failure)?;
                            }
                        },
                        Err(e) => {
                            last_failure = format!(
                                "mass residual {resid:.3e}; mobile-set correction was rejected ({e})"
                            );
                            shrink(&mut trust, &mut retries, &last_failure)?;
                        }
                    }
                }
            }
        };
        let (b, f, resid, out, dmax) = accepted;
        current = b;
        fwd = f;
        state = slope_state(project, gpu, &current, &mut fwd, omega)?;
        sweeps.push(SweepRecord {
            sweep,
            max_slack,
            n_supercritical: n_super,
            trust,
            trust_retries: retries,
            socp_iterations: out.iterations,
            n_active,
            socp_seconds: out.solve_seconds,
            dropped_overhang: out.dropped_overhang,
            mass_residual: resid,
            max_displacement: dmax,
        });
        let (new_slack, _, _) = measure(&state);
        if new_slack > 0.99 * last_slack {
            stall_count += 1;
        } else {
            stall_count = 0;
        }
        last_slack = new_slack;
        if stall_count >= 3 {
            stalled = true;
            break;
        }
    }
    let (slack_after, n_after, preserved) = measure(&state);
    if n_after == 0 {
        converged = true;
    }
    Ok((
        current,
        RelaxReport {
            converged,
            stalled,
            identification: prot.identification,
            n_protected,
            n_locked,
            preserved_supercritical: preserved,
            slack_before,
            slack_after,
            sweeps,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scf_simulator::body::Topology;

    #[test]
    fn subcritical_vertices_stay_locked() {
        let topo = Topology::icosphere(2);
        let nv = topo.n_vertices();
        let nf = topo.n_faces();
        let tri = topo.faces[0];
        let mut slack = vec![0.0; nf];
        slack[0] = 0.1;

        let protect = vec![Kind::Mobile; nv];
        let (kind, slope) = friction_lock(&topo, &protect, &slack, 1.0e-3);
        assert!(slope[0]);
        assert_eq!(slope.iter().filter(|on| **on).count(), 1);
        for i in 0..nv {
            let on_face = tri.contains(&(i as u32));
            assert_eq!(
                kind[i] == Kind::Mobile,
                on_face,
                "vertex {i} mobile={} but on the supercritical face={on_face}",
                kind[i] == Kind::Mobile
            );
        }

        let mut protect = vec![Kind::Mobile; nv];
        protect[tri[0] as usize] = Kind::Locked;
        let (kind, slope) = friction_lock(&topo, &protect, &slack, 1.0e-3);
        assert!(slope[0]);
        assert_eq!(kind[tri[0] as usize], Kind::Locked);
        assert_eq!(kind[tri[1] as usize], Kind::Mobile);
        assert_eq!(kind[tri[2] as usize], Kind::Mobile);

        let protect = vec![Kind::Locked; nv];
        let (kind, slope) = friction_lock(&topo, &protect, &slack, 1.0e-3);
        assert!(!slope[0]);
        assert!(kind.iter().all(|k| *k == Kind::Locked));
    }
}
