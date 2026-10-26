//! The quasi-static evolution loop of §10.4.4.
//!
//! One pass of [`run`] is steps 1–9 of the doc: forward gravity (Tier 1), the
//! Stokes–pressure solve, yield classification, friction-angle chain, mode C, step size,
//! geometric SOCP in normal increments with outer acceptance, `ρ_s` transport, and the
//! rigid frame re-alignment, all on the general polyhedron of hypothesis (G). The spin
//! schedule is an exogenous input; neither angular momentum nor any Tier 3 coupling enters.

use nalgebra::{Matrix3, Vector3};
use serde::Serialize;

use crate::common::math::DVec3;

use super::basis::{Want, contract_velocity_rays};
use super::body::Body;
use super::cores::{self, CoreReport, Quadrature};
use super::domain::volume_rule;
use super::forward::Forward;
use super::forward::Loading;
use super::frame::{FrameReport, realign};
use super::mechanics::{Mechanics, evaluate};
use super::params::Project;
use super::relief::{Relief, clearance, relieve};
use super::runout::RunoutReport;
use super::settle::{SettleReport, settle};
use super::socp::{Kind, SocpSeed, mass_jacobian};
use super::spin::SpinSchedule;
use super::stokes::Guess;
use super::transport::{surface_flow, transport};
use crate::common::error::SimError;

/// Everything that defines a restartable state of the run.
#[derive(Clone, Debug)]
pub struct State {
    pub body: Body,
    pub level: usize,
    pub t: f64,
    pub step: usize,
    pub omega: f64,
    pub frame_rotation: [[f64; 3]; 3],
    pub frame_translation: DVec3,
    /// Velocity coefficients of the last solve (`u⁰` of the next step).
    pub velocity: Vec<f64>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct StepRecord {
    /// Core transport and refit of §9.10.7.
    pub cores: CoreReport,
    pub step: usize,
    pub t: f64,
    /// Physical time advanced by this record.
    pub advance: f64,
    pub omega: f64,
    /// `"geometry"`, `"stage-equilibrium"` or `"spin-only"`.
    pub kind: &'static str,
    pub dt: f64,
    pub dt_bound: &'static str,
    pub dt_cfl: f64,
    pub dt_c: f64,
    pub dt_char: f64,
    pub dt_omega: f64,
    pub picard_iterations: usize,
    pub eps_picard: f64,
    pub eps_momentum: f64,
    pub eps_lambda: f64,
    pub eps_rot: f64,
    pub eta_lo: f64,
    pub eta_hi: f64,
    pub tensile_fraction: f64,
    pub inverse_ekman: f64,
    pub quasi_static: bool,
    pub yielded_fraction: f64,
    pub mobile_area_fraction: f64,
    pub core_mass_fraction: f64,
    pub core_degenerate: bool,
    pub core_creep: f64,
    pub plastic_dissipation: f64,
    pub n_mobile: usize,
    pub n_deposit: usize,
    pub n_locked: usize,
    pub n_supercritical_faces: usize,
    pub max_slack: f64,
    pub eroded_rate: f64,
    /// `|∫ J_dep − ∫ J_ero| / max(∫ J_ero, ∫ J_dep)`, zero when both vanish.
    pub flux_gap: f64,
    /// Largest `|d|` on the locked set. The cone fixes these nodes at zero.
    pub max_locked_step: f64,
    /// Largest `|d − d_pred|` on the pure-deposition set.
    pub max_deposit_miss: f64,
    pub trust: f64,
    pub trust_retries: usize,
    pub socp_iterations: u32,
    pub slope_slack: f64,
    pub mass_slack: f64,
    pub dropped_overhang: usize,
    pub mass_residual: f64,
    pub max_displacement: f64,
    /// The cone program failed at the trust-radius floor and the geometry was kept.
    pub cone_unresolved: bool,
    pub transport_fix: f64,
    pub rho_s_total: f64,
    pub frame_applied: bool,
    pub frame_angle_deg: f64,
    pub frame_mass_change: f64,
    /// `C_N` of Proposition 8.0.6 for the Nitsche pressure problem of this geometry.
    pub nitsche_c: f64,
    pub rheo_unconverged: u32,
    pub rheo_seconds: f64,
    pub mass: f64,
    pub centroid_offset: f64,
    /// Runout of the eroded and the routed packets together (§2.8b).
    pub runout: RunoutReport,
    /// Mass the slope relief removed and routed (kg, §2.8b).
    pub relief_mass: f64,
    /// The surface settling after the creep step (step S).
    pub settle: SettleReport,
}

#[derive(Clone, Debug)]
pub struct RunOptions {
    pub max_steps: Option<usize>,
    /// Stop at this time instead of `T_end` of the schedule.
    pub until: Option<f64>,
}

pub struct RunResult {
    pub state: State,
    pub records: Vec<StepRecord>,
    pub complete: bool,
}

/// Surface fields of one accepted step, indexed like the body the observer receives.
///
/// `positions_from` are the vertices at the start of the step, in the frame of the body
/// the observer receives. `d_normal` is `λ d` on a geometry step and zero on a spin or
/// stage-equilibrium step. `movable` is the classifier flag (true means the vertex is
/// outside the rigid core).
/// Runout paths handed to the viewer per step (§2.8b).
pub const RUNOUT_PATHS: usize = 256;

#[derive(Clone, Debug)]
pub struct StepFields {
    pub positions_from: Vec<DVec3>,
    pub f_delta: Vec<f64>,
    pub movable: Vec<bool>,
    pub v_sl: Vec<DVec3>,
    pub normal: Vec<DVec3>,
    pub j_ero: Vec<f64>,
    pub j_dep: Vec<f64>,
    pub d_normal: Vec<f64>,
    /// Per face. Incident-face scatter happens when the snapshot is packed.
    pub slack: Vec<f64>,
    pub g_face: Vec<DVec3>,
    /// Cauchy stress at the face centres (Pa, `(xx, yy, zz, xy, xz, yz)`).
    pub stress: Vec<[f64; 6]>,
    /// Runout paths, `runout::PATH_SAMPLES` points each (m; `w = 1` airborne).
    pub runout: Vec<[f64; 4]>,
    /// False on the spin-only and stage-equilibrium exits. Core creep is not drift.
    pub flow_active: bool,
}

/// Compose the accumulated frame map with the report of one more re-alignment.
pub fn compose_frame(
    state_rot: [[f64; 3]; 3],
    state_trans: DVec3,
    rep: &FrameReport,
) -> ([[f64; 3]; 3], DVec3) {
    let m = |a: [[f64; 3]; 3]| {
        Matrix3::new(
            a[0][0], a[0][1], a[0][2], a[1][0], a[1][1], a[1][2], a[2][0], a[2][1], a[2][2],
        )
    };
    let r1 = m(state_rot);
    let r2 = m(rep.rotation);
    let r = r2 * r1;
    let t1 = Vector3::new(state_trans.x, state_trans.y, state_trans.z);
    let t2 = Vector3::new(rep.translation.x, rep.translation.y, rep.translation.z);
    let t = r1.transpose() * t2 + t1;
    (
        [
            [r[(0, 0)], r[(0, 1)], r[(0, 2)]],
            [r[(1, 0)], r[(1, 1)], r[(1, 2)]],
            [r[(2, 0)], r[(2, 1)], r[(2, 2)]],
        ],
        DVec3::new(t.x, t.y, t.z),
    )
}

pub const IDENTITY: [[f64; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

/// Fresh state: `ρ_s⁰` from the project and the mandatory initial frame alignment (step 0).
pub fn start(project: &Project, body: Body) -> Result<(State, Forward), SimError> {
    let (body, rep, fwd) = realign(&body, None, project.g, &project.geometry, true)?;
    let (rot, trans) = compose_frame(IDENTITY, DVec3::ZERO, &rep);
    Ok((
        State {
            level: project.body.level,
            body,
            t: 0.0,
            step: 0,
            omega: 0.0,
            frame_rotation: rot,
            frame_translation: trans,
            velocity: Vec::new(),
        },
        fwd,
    ))
}

#[allow(clippy::too_many_lines)]
pub fn run(
    project: &Project,
    spin: &SpinSchedule,
    mut state: State,
    mut fwd: Forward,
    opts: &RunOptions,
    observer: &mut dyn FnMut(&StepRecord, &State, &StepFields) -> Result<(), SimError>,
) -> Result<RunResult, SimError> {
    spin.validate()?;
    let g = &project.geometry;
    let st = &project.step;
    let t_end = opts.until.unwrap_or_else(|| spin.t_end());
    let snap_tol = 1e-12 * t_end.max(1.0);
    let mut records = Vec::new();
    let mut mass_target = fwd.moments.mass;
    // A resumed snapshot carries the velocity only; its pressure starts cold.
    let mut guess: Option<Guess> = (!state.velocity.is_empty()).then(|| Guess {
        u: state.velocity.clone(),
        p: Vec::new(),
    });
    let mut socp_seed: Option<SocpSeed> = None;
    let mut steps_done = 0usize;

    while state.t < t_end - snap_tol {
        if let Some(m) = opts.max_steps {
            if steps_done >= m {
                break;
            }
        }
        if steps_done >= st.max_steps {
            return Err(SimError::step_limit_exhausted(format!(
                "step.max_steps = {} exhausted at t = {:.6e}",
                st.max_steps, state.t
            )));
        }
        let topo = state.body.topo.clone();
        let nv = topo.n_vertices();
        let positions_from = state.body.positions.clone();
        let sample = spin.sample(state.t);
        let t_next = sample.t_next.min(t_end);
        let omega = sample.omega;
        state.omega = omega;

        // Steps 1–4: the deep state of the slow operator (§10.4.4).
        let load = Loading {
            omega,
            tide: spin.tide(state.t),
        };
        let mech = evaluate(project, &state.body, &mut fwd, load, guess.as_ref())?;
        let cl = &mech.class;
        let area = {
            let _span = super::timing::span("vertex_areas");
            state.body.vertex_areas()
        };
        let r_mean = state.body.r_mean();
        // χ_Y from the unyielded core is authoritative for creep (§2.8 B3). Erosion, runout
        // and slope promotion are the fast operator: they settle after the creep step (S).
        let yielded = cl.movable.clone();
        let slack: Vec<f64> = mech.rheo.iter().map(|node| node.slack).collect();
        let n_mobile = yielded.iter().filter(|m| **m).count();
        let chi_a: Vec<bool> = (0..nv)
            .map(|i| yielded[i] && state.body.rho_s[i] > 0.0)
            .collect();

        let mut rec = StepRecord {
            step: state.step,
            t: state.t,
            omega,
            kind: "geometry",
            dt_cfl: f64::INFINITY,
            dt_c: f64::INFINITY,
            dt_char: f64::INFINITY,
            dt_omega: f64::INFINITY,
            picard_iterations: mech.sol.picard_iterations,
            eps_picard: mech.sol.eps_picard,
            eps_momentum: mech.sol.eps_momentum,
            eps_lambda: mech.sol.eps_lambda,
            eps_rot: mech.sol.eps_rot,
            eta_lo: mech.sol.eta_lo,
            eta_hi: mech.sol.eta_hi,
            tensile_fraction: mech.sol.tensile_fraction,
            inverse_ekman: cl.inverse_ekman,
            quasi_static: cl.inverse_ekman <= st.eps_inertia,
            yielded_fraction: cl.yielded_fraction,
            mobile_area_fraction: cl.mobile_area_fraction,
            core_mass_fraction: cl.core_mass_fraction,
            core_degenerate: cl.core_degenerate,
            core_creep: cl.core_creep,
            plastic_dissipation: cl.plastic_dissipation,
            n_mobile,
            n_supercritical_faces: slack.iter().filter(|s| **s > project.relief.tol_slack).count(),
            max_slack: slack.iter().copied().fold(0.0, f64::max),
            mass_residual: (fwd.moments.mass - mass_target).abs() / mass_target,
            rho_s_total: (0..nv).map(|i| area[i] * state.body.rho_s[i]).sum(),
            nitsche_c: mech.ops.c_nitsche,
            rheo_unconverged: mech.rheo_stats.unconverged,
            rheo_seconds: mech.rheo_stats.seconds,
            mass: fwd.moments.mass,
            centroid_offset: fwd.moments.centroid.length(),
            ..StepRecord::default()
        };
        guess = Some(mech.sol.guess());
        state.velocity = mech.sol.u.clone();

        // Stage equilibrium (2.15(c)): the whole body in the rigid core (an interior
        // yielded pocket would move the cores even with a locked surface) and a settled
        // surface. The stage then jumps to its end.
        let all_core = cl.in_core.iter().all(|c| *c);
        if n_mobile == 0 && spin.is_stages() && all_core {
            let settled = settle(
                project,
                load,
                state.body.clone(),
                fwd,
                &yielded,
                &cl.f_delta,
                t_next - state.t,
                mass_target,
                &mut socp_seed,
                state.step,
            )?;
            let quiet = settled.report.sweeps == 0;
            state.body = settled.body;
            fwd = settled.fwd;
            if quiet {
                let dt = t_next - state.t;
                rec.kind = "stage-equilibrium";
                rec.dt = dt;
                rec.advance = dt;
                rec.dt_bound = "t_next";
                rec.n_locked = nv;
                rec.settle = settled.report;
                advance_time(&mut state, spin, dt, t_next, snap_tol);
                state.step += 1;
                steps_done += 1;
                let fields = step_fields(
                    &positions_from,
                    &mech,
                    &settled.mode_c,
                    &slack,
                    vec![0.0; nv],
                    &settled.paths,
                    false,
                    None,
                );
                observer(&rec, &state, &fields)?;
                records.push(rec);
                continue;
            }
        }

        // Step 6: step size of the slow operator.
        let omega_bound = if spin.is_stages() {
            f64::INFINITY
        } else {
            st.eps_omega * omega.max(st.omega_ref) / sample.omega_dot.abs().max(1e-30)
        };
        rec.dt_omega = omega_bound;
        // Single-valued core map: τ ‖dev ∇v_adv‖_F ≤ θ_ch at every moving core, with the
        // gradient at the current centres (§10.4.1); step 6b rechecks it at the midpoints.
        let cores_on = project.cores.lagrangian() && !state.body.kernels.is_empty();
        let probe_h = 1e-3 * r_mean;
        let v_adv = |p: &[DVec3]| advection(&mech, p);
        let in_body = |c: DVec3| mech.index.inside(c);
        let core_rate = if cores_on {
            cores::strain_rate(&state.body.kernels, &v_adv, &in_body, probe_h, &project.cores)
        } else {
            0.0
        };
        let dt_core = if core_rate > 0.0 {
            st.theta_char / (st.lambda * core_rate)
        } else {
            f64::INFINITY
        };
        // 6b: Level 0 of the core transport over τ = λΔt. Where a midpoint gradient breaks
        // the bound, Δt shrinks by the violation ratio and the O(K) prediction repeats.
        let predict_cores = |dt: &mut f64, rep: &mut CoreReport| {
            if !cores_on || core_rate == 0.0 {
                return state.body.kernels.clone();
            }
            let _span = super::timing::span("core_transport");
            let mut tries = 0;
            loop {
                let mut r = CoreReport::default();
                let ks = cores::predict(
                    &state.body.kernels,
                    st.lambda * *dt,
                    &v_adv,
                    &in_body,
                    probe_h,
                    &project.cores,
                    &mut r,
                );
                tries += 1;
                if r.strain_step <= st.theta_char * (1.0 + 1e-9) || tries >= CORE_REPEATS {
                    *rep = r;
                    return ks;
                }
                *dt *= st.theta_char / r.strain_step;
            }
        };
        let (creep_d, frep) = if n_mobile == 0 {
            // No yielded surface: the creep step leaves the surface in place, but an
            // interior yielded pocket still moves the cores (6b, 8c), and the frame is
            // re-aligned (8b).
            let bounds = [
                ("core", dt_core),
                ("omega", omega_bound),
                ("dt_max", st.dt_max),
                ("t_next", t_next - state.t),
            ];
            let (mut name, mut dt) = bounds
                .iter()
                .fold(("dt_max", f64::INFINITY), |a, b| if b.1 < a.1 { *b } else { a });
            let chosen = dt;
            let kernels_pred = predict_cores(&mut dt, &mut rec.cores);
            if dt < chosen {
                name = "core";
            }
            let advance = st.lambda * dt;
            let mut body = state.body.clone();
            if core_rate > 0.0 {
                body.kernels = kernels_pred;
                fwd.set_density(body.density()?)?;
                close_cores(project, &mech, &state.body, &mut body, &mut fwd, mass_target, advance, &mut rec.cores)?;
            }
            let (body, frep, new_fwd) = realign(&body, Some(fwd), project.g, g, false)?;
            rec.kind = "spin-only";
            rec.dt = dt;
            rec.advance = advance;
            rec.dt_bound = name;
            state.body = body;
            fwd = new_fwd;
            (vec![0.0; nv], frep)
        } else {
            // §10.4.1 writes θ h_Γ,min / max s, with θ ≤ 1/2. That product is a
            // sufficient bound. The accuracy it protects is local: a movable node
            // advances by at most θ times its own nearest spacing. The per-node
            // minimum is that bound, and it is larger whenever the fastest node
            // does not sit on the shortest edge.
            let dt_cfl = cfl_dt(&state.body, &yielded, &cl.normal, &cl.v_sl, st.theta_cfl);
            let flow = {
                let _span = super::timing::span("surface_flow");
                surface_flow(&state.body, &cl.normal, &cl.v_sl, &chi_a)
            };
            let dt_char = if flow.grad_max > 0.0 {
                st.theta_char / flow.grad_max
            } else {
                f64::INFINITY
            };
            let bounds = [
                ("core", dt_core),
                ("cfl", dt_cfl),
                ("characteristic", dt_char),
                ("omega", omega_bound),
                ("dt_max", st.dt_max),
                ("t_next", t_next - state.t),
            ];
            let (mut name, mut dt) = bounds
                .iter()
                .fold(("dt_max", f64::INFINITY), |a, b| if b.1 < a.1 { *b } else { a });
            if !(dt.is_finite() && dt > 0.0) {
                return Err(SimError::solver(format!("step size {dt} is not positive")));
            }
            // 6b before the step is used anywhere: the cone's mass row sees the predicted
            // density ρ_cand, so dense material carried outward is not mistaken for a
            // geometric mass error (§9.10.7).
            let chosen = dt;
            let kernels_pred = predict_cores(&mut dt, &mut rec.cores);
            if dt < chosen {
                name = "core";
            }
            rec.dt = dt;
            rec.dt_bound = name;
            rec.dt_cfl = dt_cfl;
            rec.dt_char = dt_char;
            rec.advance = st.lambda * dt;

            // Step 7: the creep predictor on the yielded surface; every other vertex locked.
            let mut d_pred = vec![0.0; nv];
            let mut kind = vec![Kind::Locked; nv];
            for i in 0..nv {
                if yielded[i] {
                    d_pred[i] = dt * cl.v_sl[i].dot(cl.normal[i]);
                    kind[i] = Kind::Mobile;
                }
            }
            rec.n_locked = nv - n_mobile;
            // The repose rows of the loose faces of the yielded set keep creep from
            // steepening them past repose (§2.8c, S_slope).
            let slope_row: Vec<bool> = mech
                .rheo
                .iter()
                .enumerate()
                .map(|(f, node)| {
                    let v = topo.faces[f].map(|x| x as usize);
                    node.sin_phi.is_finite()
                        && v.iter().any(|i| kind[*i] == Kind::Mobile)
                        && v.iter().map(|i| state.body.rho_s[*i]).sum::<f64>() > 0.0
                })
                .collect();
            let sin_phi: Vec<f64> = mech.rheo.iter().map(|n| n.sin_phi).collect();
            // Dimensionless fidelity weights around 1 (area over its mean), so that the
            // exact-penalty scale ϱ ≥ 10 h̄² of §2.13 holds with d in metres.
            let mean_area = area.iter().sum::<f64>() / nv as f64;
            let weight: Vec<f64> = area.iter().map(|a| a / mean_area).collect();
            let density_cand =
                crate::common::density::CauchyDensity::try_new(state.body.bulk, kernels_pred.clone())?;
            if cores_on {
                fwd.set_density(density_cand.clone())?;
            }
            let jac = {
                let _span = super::timing::span("mass_jacobian");
                mass_jacobian(&state.body, &density_cand)
            };
            // Clearance rows of (G4): −θ_g t⁻ ≤ d ≤ θ_g t⁺ along the vertex normals, with
            // t^± the distance to the next crossing of Γ (two first-hit queries).
            let (lower, upper) = clearance(
                &state.body,
                &cl.normal,
                fwd.body_mesh(),
                g.theta_g,
                &vec![f64::NEG_INFINITY; nv],
            )?;
            let relief = Relief {
                project,
                body: &state.body,
                kernels: &kernels_pred,
                normals: &cl.normal,
                kind: &kind,
                d_pred: &d_pred,
                weight: &weight,
                gravity: &mech.g_face,
                sin_phi: &sin_phi,
                slope_row: &slope_row,
                jac: &jac,
                lower: &lower,
                upper: &upper,
                mass_target,
                scale: st.lambda,
                routing: None,
                step: state.step,
            };
            let rel = relieve(&relief, &fwd, &mut socp_seed)?;
            rec.trust = rel.trust;
            rec.trust_retries = rel.retries;
            rec.cone_unresolved = rel.unresolved;
            rec.socp_iterations = rel.out.iterations;
            rec.slope_slack = rel.out.slope_slack;
            rec.mass_slack = rel.out.mass_slack;
            rec.dropped_overhang = rel.out.dropped_overhang;
            rec.mass_residual = rel.mass_residual;
            rec.max_displacement = rel.dmax;
            rec.max_locked_step = rel.max_locked_step;
            rec.max_deposit_miss = rel.max_deposit_miss;
            let creep_d: Vec<f64> = rel.out.d.iter().map(|d| st.lambda * d).collect();

            // ρ_s: semi-Lagrangian transport along the creep slip over λΔt.
            let area_new = {
                let _span = super::timing::span("vertex_areas");
                rel.body.vertex_areas()
            };
            let (rho_s, fix) = {
                let _span = super::timing::span("transport");
                transport(&state.body, &flow, &state.body.rho_s, &area, &area_new, rec.advance)
            };
            rec.transport_fix = fix;
            let mut body = rel.body;
            body.rho_s = rho_s;
            let mut new_fwd = rel.fwd;
            // 8c: Levels 1 and 2 of the core transport (§9.10.4–5).
            if cores_on {
                close_cores(
                    project,
                    &mech,
                    &state.body,
                    &mut body,
                    &mut new_fwd,
                    mass_target,
                    rec.advance,
                    &mut rec.cores,
                )?;
            }
            // 8b: frame re-alignment (only when triggered).
            let (body, frep, new_fwd) = realign(&body, Some(new_fwd), project.g, g, false)?;
            state.body = body;
            fwd = new_fwd;
            (creep_d, frep)
        };
        rec.frame_applied = frep.applied;
        rec.frame_angle_deg = frep.angle_deg;
        rec.frame_mass_change = frep.mass_change;
        if frep.applied {
            let (rot, trans) = compose_frame(state.frame_rotation, state.frame_translation, &frep);
            state.frame_rotation = rot;
            state.frame_translation = trans;
            guess = None;
            state.velocity.clear();
            // The seed is indexed by vertex, and realignment keeps the vertices, so it
            // stays a warm start for the next cone.
            mass_target = fwd.moments.mass;
        }

        // Step S: the surface settles on the frozen yield state over the step (A-T1).
        let settled = settle(
            project,
            load,
            state.body.clone(),
            fwd,
            &yielded,
            &cl.f_delta,
            rec.advance,
            mass_target,
            &mut socp_seed,
            state.step,
        )?;
        state.body = settled.body;
        fwd = settled.fwd;
        let rep = &settled.report;
        if n_mobile == 0 && rep.sweeps > 0 {
            rec.kind = "settle";
        }
        rec.dt_c = rep.tau_surface;
        rec.eroded_rate = settled.mode_c.eroded_rate;
        rec.flux_gap = flux_gap(&settled.mode_c, &area);
        rec.relief_mass = rep.relief_mass;
        rec.max_displacement = rec.max_displacement.max(rep.max_displacement);
        rec.cone_unresolved |= rep.cone_unresolved;
        if rep.sweeps > 0 {
            rec.mass_residual = rep.mass_residual;
        }
        rec.runout = settled.runout;
        rec.rho_s_total = {
            let a = state.body.vertex_areas();
            (0..nv).map(|i| a[i] * state.body.rho_s[i]).sum()
        };
        rec.settle = settled.report;
        let moved = n_mobile > 0 || rec.settle.sweeps > 0;
        let d_normal: Vec<f64> = creep_d.iter().zip(&settled.d_normal).map(|(c, s)| c + s).collect();

        // 9: advance time.
        advance_time(&mut state, spin, rec.advance, t_next, snap_tol);
        state.step += 1;
        steps_done += 1;
        let fields = step_fields(
            &positions_from,
            &mech,
            &settled.mode_c,
            &slack,
            d_normal,
            &settled.paths,
            moved,
            Some(&frep),
        );
        observer(&rec, &state, &fields)?;
        records.push(rec);
    }
    let complete = state.t >= t_end - snap_tol;
    Ok(RunResult {
        state,
        records,
        complete,
    })
}

/// The masked slip velocity `v_adv = χ_Y (v − a − θ × x)` at arbitrary points (§9.10.2).
/// `χ_Y` is read at the nearest point of the classification grid `X_Y`.
fn advection(mech: &Mechanics, pts: &[DVec3]) -> Vec<DVec3> {
    let cl = &mech.class;
    // Every point of X_Y in the rigid core: χ_Y = 0 everywhere and the result is zero,
    // so the velocity basis is not evaluated.
    if cl.in_core.iter().all(|c| *c) {
        return vec![DVec3::ZERO; pts.len()];
    }
    let v = contract_velocity_rays(&mech.surf, &mech.ops.lay, pts, 1, Want::V, &mech.sol.u).v;
    pts.iter()
        .enumerate()
        .map(|(k, x)| {
            if !cl.movable_near(*x) {
                return DVec3::ZERO;
            }
            DVec3::new(v[3 * k], v[3 * k + 1], v[3 * k + 2]) - cl.core_a - cl.core_theta.cross(*x)
        })
        .collect()
}

/// Levels 1–2 of §9.10: mass closure of the predicted cores on the new body,
/// then, when the misfit or the closure exceeds its threshold, the refit.
#[allow(clippy::too_many_arguments)]
/// 8c: Levels 1 and 2 of the core transport (§9.10.4–5) on the new body, which carries
/// the predicted cores; `fwd` is bound to it and gets the closed density.
#[allow(clippy::too_many_arguments)]
fn close_cores(
    project: &Project,
    mech: &Mechanics,
    old: &Body,
    body: &mut Body,
    fwd: &mut Forward,
    mass_target: f64,
    tau: f64,
    rep: &mut CoreReport,
) -> Result<(), SimError> {
    let _span = super::timing::span("core_refit");
    let (ks, refit) = closed_cores(project, mech, old, body, fwd, mass_target, tau, rep)?;
    body.kernels = ks;
    fwd.set_density(body.density()?)?;
    if refit {
        // The refit holds the quadrature mass only; close the exact mass again.
        let bulk_mass = body.bulk * fwd.moments.volume;
        cores::close_mass(&mut body.kernels, fwd.moments.mass, bulk_mass, mass_target);
        fwd.set_density(body.density()?)?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn closed_cores(
    project: &Project,
    mech: &Mechanics,
    old: &Body,
    new: &Body,
    fwd: &Forward,
    mass_target: f64,
    tau: f64,
    rep: &mut CoreReport,
) -> Result<(Vec<crate::common::density::CauchyKernel>, bool), SimError> {
    let mut ks = new.kernels.clone();
    let bulk_mass = new.bulk * fwd.moments.volume;
    rep.xi = cores::close_mass(&mut ks, fwd.moments.mass, bulk_mass, mass_target);
    // Volume quadrature of V⁽ⁿ⁺¹⁾: the ray-segment rule of (G2).
    let s = &project.stokes;
    let (pts, w, _, _) = volume_rule(fwd.body_mesh(), s.n_theta, s.n_phi, s.n_radial, s.cut_depth)?;
    let quad = Quadrature { pts, w };
    let target = cores::departure_density(&old.density()?, &quad, tau, &|p| advection(mech, p));
    let closed = crate::common::density::CauchyDensity::try_new(new.bulk, ks.clone())?;
    rep.e_rho = cores::misfit(&closed, &quad, &target);
    let refit = rep.e_rho > project.cores.eps_rho || rep.xi.abs() > project.cores.eps_xi;
    if refit {
        ks = cores::refit(new.bulk, &ks, &quad, &target, &project.cores, rep);
    }
    Ok((ks, refit))
}

/// Advance the clock; the state then carries the spin of its own time, so a snapshot
/// written after the last step reports the spin at the end of the schedule.
fn advance_time(state: &mut State, spin: &SpinSchedule, dt: f64, t_next: f64, tol: f64) {
    state.t += dt;
    if (state.t - t_next).abs() < tol {
        state.t = t_next;
    }
    state.omega = spin.sample(state.t).omega;
}

/// Slope promotion (§2.8c, hypothesis A-C4): the geometric mobile set and the promoted flags.
///
/// `yielded` is `χ_Y` of the classifier. Every vertex of a face whose repose slack exceeds
/// `tol` that is not already yielded and carries loose material (`ρ_s > 0`) is promoted.
/// The rule is local: a yielded vertex elsewhere on the body does not switch it off.
/// A promoted vertex stays in the rigid core for creep, mode C, runout sources and the
/// core advection; it only becomes a free variable of the slope cone, with `d_pred = 0`.
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

/// Repetitions of the step-6b core prediction after a midpoint violation.
const CORE_REPEATS: usize = 8;

fn flux_gap(mc: &super::transport::ModeC, area: &[f64]) -> f64 {
    let ero: f64 = mc.j_ero.iter().zip(area).map(|(j, a)| j * a).sum();
    let dep: f64 = mc.j_dep.iter().zip(area).map(|(j, a)| j * a).sum();
    (dep - ero).abs() / ero.max(dep).max(1e-30)
}

/// `θ min_i h_i / |v_sl,i · N_i|` over the yielded movable set (G10).
///
/// `h_i` is the shortest edge at the node: a node advances by at most `θ` times its
/// own nearest spacing. `θ` is the configured CFL factor, at most 1/2.
fn cfl_dt(body: &Body, yielded: &[bool], normal: &[DVec3], v_sl: &[DVec3], theta: f64) -> f64 {
    let position = body.positions();
    let mut dt = f64::INFINITY;
    for i in 0..yielded.len() {
        if !yielded[i] {
            continue;
        }
        let speed = v_sl[i].dot(normal[i]).abs();
        if !(speed.is_finite() && speed > 0.0) {
            continue;
        }
        let mut spacing = f64::INFINITY;
        for &j in &body.topo.vertex_neighbors[i] {
            spacing = spacing.min((position[i] - position[j]).length());
        }
        if spacing.is_finite() && spacing > 0.0 {
            dt = dt.min(theta * spacing / speed);
        }
    }
    dt
}

/// Surface fields of the step for the observer. A frame realignment is a rigid motion
/// (G8), so vertex and face identities are kept: vectors turn with `Q`, tensors as
/// `Q σ Qᵀ`, and points as `Q (x − T)`.
#[allow(clippy::too_many_arguments)]
pub fn step_fields(
    from: &[DVec3],
    mech: &Mechanics,
    mc: &super::transport::ModeC,
    slack: &[f64],
    d_normal: Vec<f64>,
    runout: &[[f64; 4]],
    flow_active: bool,
    frame: Option<&FrameReport>,
) -> StepFields {
    let cl = &mech.class;
    let (q, t) = match frame.filter(|r| r.applied) {
        Some(r) => (r.rotation, r.translation),
        None => (IDENTITY, DVec3::ZERO),
    };
    let turn = |v: &DVec3| mul_q(q, *v);
    StepFields {
        positions_from: from.iter().map(|x| mul_q(q, *x - t)).collect(),
        f_delta: cl.f_delta.clone(),
        movable: cl.movable.clone(),
        v_sl: cl.v_sl.iter().map(turn).collect(),
        normal: cl.normal.iter().map(turn).collect(),
        j_ero: mc.j_ero.clone(),
        j_dep: mc.j_dep.clone(),
        d_normal,
        slack: slack.to_vec(),
        g_face: mech.g_face.iter().map(turn).collect(),
        stress: mech.surface_stress.iter().map(|s| rotate_sym(q, *s)).collect(),
        runout: runout
            .iter()
            .map(|p| {
                let x = mul_q(q, DVec3::new(p[0], p[1], p[2]) - t);
                [x.x, x.y, x.z, p[3]]
            })
            .collect(),
        flow_active,
    }
}

/// `Q T Qᵀ` of a symmetric `(xx, yy, zz, xy, xz, yz)` tensor.
fn rotate_sym(q: [[f64; 3]; 3], t: [f64; 6]) -> [f64; 6] {
    let m = [[t[0], t[3], t[4]], [t[3], t[1], t[5]], [t[4], t[5], t[2]]];
    let mut r = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            r[i][j] = (0..3)
                .map(|k| (0..3).map(|l| q[i][k] * m[k][l] * q[j][l]).sum::<f64>())
                .sum();
        }
    }
    [r[0][0], r[1][1], r[2][2], r[0][1], r[0][2], r[1][2]]
}

fn mul_q(q: [[f64; 3]; 3], v: DVec3) -> DVec3 {
    DVec3::new(
        q[0][0] * v.x + q[0][1] * v.y + q[0][2] * v.z,
        q[1][0] * v.x + q[1][1] * v.y + q[1][2] * v.z,
        q[2][0] * v.x + q[2][1] * v.y + q[2][2] * v.z,
    )
}
