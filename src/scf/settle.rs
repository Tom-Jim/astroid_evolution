//! The fast operator of the time-scale split (§10.4.4 step S, hypothesis A-T1): after a
//! creep step the surface settles to quasi-static equilibrium on the frozen yield state.
//!
//! Erosion, runout and shallow slides act on minutes to hours; creep on the step of the
//! slow operator. Each sweep takes the face gravity of the current shape, erodes the loose
//! layer of the yielded surface by the exact exponential over the step `τ` (first sweep only:
//! material laid on yielding ground leaves with the next step), runs the packets out, and
//! relieves the supercritical loose faces (slope promotion, §2.8c) with the routed cone of
//! §10.4.3. Sweeps stop when no loose face is supercritical and nothing erodes, when a sweep
//! no longer reduces the supercritical set (the rest is unrelievable), or when the geometry
//! admits no further increment. Time does not advance.

use serde::Serialize;

use crate::common::error::SimError;
use crate::common::math::DVec3;

use super::body::Body;
use super::mechanics::FaceLaw;
use super::orbit::Necks;
use super::forward::{Forward, Loading};
use super::params::Project;
use super::relax::slope_state;
use super::relief::{Relief, Routing, clearance, floor_of, layer_ledger, relieve};
use super::runout::{self, RunoutField, RunoutReport};
use super::socp::{Kind, mass_jacobian};
use super::surface::SurfaceIndex;
use super::transport::{ModeC, Spread, mode_c_over};

/// What settling did in one step.
#[derive(Clone, Debug, Default, Serialize)]
pub struct SettleReport {
    pub sweeps: usize,
    /// No loose face is supercritical and nothing erodes.
    pub converged: bool,
    /// A sweep no longer reduced the supercritical set; the rest is unrelievable.
    pub stalled: bool,
    /// The geometry admitted no further increment down to the trust floor.
    pub geometry_limited: bool,
    /// Supercritical faces and their largest slack before and after.
    pub n_supercritical_before: usize,
    pub max_slack_before: f64,
    pub n_supercritical_after: usize,
    pub max_slack_after: f64,
    /// Largest number of promoted vertices in a sweep.
    pub n_promoted: usize,
    /// Mass eroded from the yielded surface (Mode C, kg).
    pub eroded_mass: f64,
    /// Mass the relief removed from promoted vertices and routed (kg).
    pub relief_mass: f64,
    /// Loose-layer mass the relief moved through the ledger of §2.8c (kg).
    pub layer_mass: f64,
    /// Largest normal displacement of a vertex over all sweeps (m).
    pub max_displacement: f64,
    /// Shortest erosion e-folding time `σ₀ / (k_e max [F_δ]₊)` (s); the split needs it far
    /// below the creep step `τ` (A-T1).
    pub tau_surface: f64,
    pub trust_retries: usize,
    pub cone_unresolved: bool,
    pub mass_residual: f64,
    /// A-O: eroded and routed mass lost through a neck (kg).
    pub escaped_runout: f64,
    pub escaped_relief: f64,
    /// A-O: vertices whose loose layer detached under outward effective gravity.
    pub n_detached: usize,
}

/// The settled body and what the viewer shows of it.
pub struct Settled {
    pub body: Body,
    pub fwd: Forward,
    pub report: SettleReport,
    /// Mode C of the first sweep (fluxes over `τ`).
    pub mode_c: ModeC,
    /// Accumulated normal displacement per vertex (m).
    pub d_normal: Vec<f64>,
    pub runout: RunoutReport,
    pub paths: Vec<[f64; 4]>,
}

/// Settle `body` under `load` on the frozen yield state `yielded`, `f_delta` (per vertex)
/// of the creep step of length `tau` (= λΔt). With the `necks` of A-O, packets beyond a
/// neck plane leave the body, and a loose layer pulled outward harder than its cohesion
/// holds detaches (§10.4.6).
#[allow(clippy::too_many_arguments)]
pub fn settle(
    project: &Project,
    load: Loading,
    mut body: Body,
    mut fwd: Forward,
    yielded: &[bool],
    f_delta: &[f64],
    tau: f64,
    mass_target: f64,
    step: usize,
    laws: Option<&[FaceLaw]>,
    necks: Option<&Necks>,
) -> Result<Settled, SimError> {
    let prm = &project.mode_c;
    let tol = project.relief.tol_slack;
    let nv = body.topo.n_vertices();
    let supercritical = |slack: &[f64]| {
        let n = slack.iter().filter(|s| **s > tol).count();
        (n, slack.iter().copied().fold(0.0_f64, f64::max))
    };
    // Without excess there is no erosion: its e-folding time is infinite.
    let mut rep = SettleReport { tau_surface: f64::INFINITY, ..SettleReport::default() };
    let mut first_mc: Option<ModeC> = None;
    let mut d_normal = vec![0.0; nv];
    let (mut weights, mut packets) = (Vec::new(), Vec::new());
    let mut target = mass_target;
    let mut slope = slope_state(project, &body, &mut fwd, load, laws)?;
    (rep.n_supercritical_before, rep.max_slack_before) = supercritical(&slope.slack);
    for sweep in 0..project.relief.max_sweeps {
        let topo = body.topo.clone();
        let area = body.vertex_areas();
        let r_mean = body.r_mean();
        let spread = Spread::new(&body, &area, prm.kernel_width * r_mean);
        let normals = body.vertex_normals();
        let density = body.density()?;
        let positions = body.positions.clone();
        let index = SurfaceIndex::build(&positions, &topo.faces);
        let rho_face: Vec<f64> = topo
            .faces
            .iter()
            .map(|t| density.rho(t.iter().map(|v| positions[*v as usize]).sum::<DVec3>() / 3.0))
            .collect();
        let hess_face = fwd.face.hess_u.clone();
        let field = RunoutField {
            topo: &topo,
            index: &index,
            positions: &positions,
            g_face: &slope.gravity,
            hess_face: &hess_face,
            rho_face: &rho_face,
            grav_const: project.g,
            load,
            material: &project.material,
            necks,
            time_limit: prm.runout_time,
            stop: prm.runout_stop,
            theta: runout::THETA,
        };
        // Mode C over τ: the exact exponential, so a layer whose e-folding time is far
        // below τ is stripped and laid down where its packets come to rest.
        let detached = match necks {
            Some(_) if sweep == 0 => detachment(project, &body, &slope.gravity, laws),
            _ => vec![false; nv],
        };
        rep.n_detached = rep.n_detached.max(detached.iter().filter(|d| **d).count());
        let sources: Vec<usize> = (0..nv)
            .filter(|&i| {
                sweep == 0 && body.rho_s[i] > 0.0 && ((yielded[i] && f_delta[i] > 0.0) || detached[i])
            })
            .collect();
        let arrivals = if prm.runout && !sources.is_empty() {
            let _span = super::timing::span("runout");
            let exterior = |p: &[DVec3]| fwd.eval_exterior(p);
            field.trace_with(&sources, Some(&exterior))?
        } else {
            Vec::new()
        };
        let mc = {
            let _span = super::timing::span("mode_c");
            if sweep == 0 {
                // A detached layer leaves within the step: β → ∞ in the exact update.
                let movable: Vec<bool> = (0..nv).map(|i| yielded[i] || detached[i]).collect();
                let excess: Vec<f64> =
                    (0..nv).map(|i| if detached[i] { f64::INFINITY } else { f_delta[i] }).collect();
                mode_c_over(
                    &spread,
                    &area,
                    &body.rho_s,
                    &movable,
                    &excess,
                    prm,
                    prm.runout.then_some(arrivals.as_slice()),
                    tau,
                )
            } else {
                ModeC::zero(nv)
            }
        };
        let excess = (0..nv)
            .filter(|&i| sweep == 0 && yielded[i] && body.rho_s[i] > 0.0)
            .map(|i| f_delta[i].max(0.0))
            .fold(0.0_f64, f64::max);
        if excess > 0.0 {
            rep.tau_surface = prm.sigma_0 / (prm.k_e * excess);
        }
        let (_, promoted) = slope_promotion(&topo.faces, yielded, &slope.slack, &body.rho_s, tol);
        let n_promoted = promoted.iter().filter(|p| **p).count();
        let eroding = mc.eroded_rate > 0.0;
        if n_promoted == 0 && !eroding {
            rep.converged = true;
            first_mc.get_or_insert(mc);
            break;
        }
        rep.sweeps = sweep + 1;
        rep.n_promoted = rep.n_promoted.max(n_promoted);
        let eroded = runout::eroded_masses(&area, &mc.j_ero, &arrivals, tau);
        rep.eroded_mass += eroded.iter().sum::<f64>();
        weights.extend(eroded);
        packets.extend(arrivals);

        // Promoted vertices are free; the erosion and deposit of the others is their
        // fixed increment Δh*_C (§2.8). Yielded vertices keep d = 0 here: they moved with
        // the creep step, and their loose layer leaves by Mode C.
        let mut d_pred = vec![0.0; nv];
        let mut kind = vec![Kind::Locked; nv];
        for i in 0..nv {
            let dh = tau * (mc.j_dep[i] - mc.j_ero[i]) / density.rho(positions[i]).max(1e-300);
            if promoted[i] {
                d_pred[i] = dh;
                kind[i] = Kind::Mobile;
            } else if dh != 0.0 {
                d_pred[i] = dh;
                kind[i] = Kind::Deposit;
            }
        }
        let slope_row: Vec<bool> = (0..topo.n_faces())
            .map(|f| {
                let v = topo.faces[f].map(|x| x as usize);
                slope.sin_phi[f].is_finite()
                    && v.iter().any(|i| kind[*i] == Kind::Mobile)
                    && v.iter().map(|i| body.rho_s[*i]).sum::<f64>() > 0.0
            })
            .collect();
        let mut slope_vertex = vec![false; nv];
        for (f, on) in slope_row.iter().enumerate() {
            if *on {
                for v in topo.faces[f] {
                    if kind[v as usize] == Kind::Mobile {
                        slope_vertex[v as usize] = true;
                    }
                }
            }
        }
        let mean_area = area.iter().sum::<f64>() / nv as f64;
        let weight: Vec<f64> = area.iter().map(|a| a / mean_area).collect();
        let jac = {
            let _span = super::timing::span("mass_jacobian");
            mass_jacobian(&body, &density)
        };
        let floor = floor_of(&promoted, &d_pred, &body.rho_s, &area, &jac);
        let (lower, upper) =
            clearance(&body, &normals, fwd.body_mesh(), project.geometry.theta_g, &floor)?;
        let relief = Relief {
            project,
            body: &body,
            kernels: &body.kernels,
            normals: &normals,
            kind: &kind,
            d_pred: &d_pred,
            weight: &weight,
            gravity: &slope.gravity,
            sin_phi: &slope.sin_phi,
            slope_row: &slope_row,
            jac: &jac,
            lower: &lower,
            upper: &upper,
            mass_target: target - tau * mc.lost_rate,
            scale: 1.0,
            routing: (prm.runout && slope_vertex.iter().any(|v| *v)).then_some(Routing {
                field: &field,
                area: &area,
                spread: &spread,
                slope_vertex: &slope_vertex,
            }),
            step,
        };
        let rel = match relieve(&relief, &fwd) {
            Ok(rel) => rel,
            // No admissible increment down to the trust floor: keep the last valid shape.
            Err(err) if err.kind == "GeometryInvalid" => {
                eprintln!("step {step}: settling stopped, {}", err.detail);
                rep.geometry_limited = true;
                first_mc.get_or_insert(mc);
                break;
            }
            Err(err) => return Err(err),
        };
        rep.escaped_runout += tau * mc.lost_rate;
        rep.escaped_relief += rel.lost;
        target -= tau * mc.lost_rate + rel.lost;
        let (debit, credit) = layer_ledger(&promoted, &rel, 1.0, &jac, &body.rho_s, &area, &spread);
        rep.layer_mass += debit.iter().sum::<f64>();
        rep.trust_retries += rel.retries;
        rep.cone_unresolved |= rel.unresolved;
        rep.mass_residual = rel.mass_residual;
        rep.max_displacement = rep.max_displacement.max(rel.dmax);
        if let Some((routed, _)) = &rel.routed {
            rep.relief_mass += routed.masses.iter().sum::<f64>();
            weights.extend(routed.masses.iter().copied());
            packets.extend(routed.arrivals.iter().cloned());
        }
        for (acc, d) in d_normal.iter_mut().zip(&rel.out.d) {
            *acc += d;
        }
        let area_new = rel.body.vertex_areas();
        let rho_s: Vec<f64> = (0..nv)
            .map(|i| {
                (body.rho_s[i] * area[i] / area_new[i]
                    + tau * (mc.j_dep[i] - mc.j_ero[i])
                    + (credit[i] - debit[i]) / area_new[i])
                    .max(0.0)
            })
            .collect();
        body = rel.body;
        body.rho_s = rho_s;
        fwd = rel.fwd;
        first_mc.get_or_insert(mc);
        let (n_before, slack_before) = supercritical(&slope.slack);
        slope = slope_state(project, &body, &mut fwd, load, laws)?;
        let (n_after, slack_after) = supercritical(&slope.slack);
        // A sweep that removes under 5 % of the supercritical faces without lowering the
        // largest slack has met what the relief cannot reduce.
        if rel.unresolved
            || (!eroding && n_after as f64 > 0.95 * n_before as f64 && slack_after >= slack_before)
        {
            rep.stalled = true;
            break;
        }
    }
    (rep.n_supercritical_after, rep.max_slack_after) = supercritical(&slope.slack);
    let runout = runout::report(&weights, &packets);
    let paths = runout::paths(&weights, &packets, super::evolve::RUNOUT_PATHS);
    let mode_c = first_mc.unwrap_or_else(|| ModeC::zero(nv));
    Ok(Settled {
        body,
        fwd,
        report: rep,
        mode_c,
        d_normal,
        runout,
        paths,
    })
}

/// The vertices whose loose layer detaches (§10.4.6): `ρ_s g_n > c_Γ`, with `g_n` the
/// area-weighted mean of `g_eff · n` over the incident faces and `c_Γ` their mean shallow
/// cohesion.
fn detachment(project: &Project, body: &Body, gravity: &[DVec3], laws: Option<&[FaceLaw]>) -> Vec<bool> {
    let geo = body.face_geometry();
    let c_s = project.material.spec.cohesion_surface;
    (0..body.topo.n_vertices())
        .map(|i| {
            let (mut gn, mut c, mut w) = (0.0, 0.0, 0.0);
            for &f in &body.topo.vertex_faces[i] {
                let (_, n, a) = geo[f];
                gn += a * gravity[f].dot(n);
                c += a * laws.map_or(c_s, |l| l[f].shallow);
                w += a;
            }
            w > 0.0 && body.rho_s[i] * gn / w > c / w
        })
        .collect()
}

/// Slope promotion (§2.8c, hypothesis A-C4): the vertices of a face whose repose slack
/// exceeds `tol`, that are not yielded and carry loose material (`ρ_s > 0`). Returns the
/// movable set (yielded or promoted) and the promoted flags.
pub fn slope_promotion(
    faces: &[[u32; 3]],
    yielded: &[bool],
    slack: &[f64],
    rho_s: &[f64],
    tol: f64,
) -> (Vec<bool>, Vec<bool>) {
    let mut movable = yielded.to_vec();
    let mut promoted = vec![false; yielded.len()];
    for (tri, s) in faces.iter().zip(slack) {
        if !(*s > tol) {
            continue;
        }
        for &v in tri {
            let i = v as usize;
            if !yielded[i] && rho_s[i] > 0.0 {
                movable[i] = true;
                promoted[i] = true;
            }
        }
    }
    (movable, promoted)
}
