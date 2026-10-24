//! The quasi-static evolution loop of §10.4.4.
//!
//! One pass of [`run`] is steps 1–9 of the doc: forward gravity (Tier 1), the
//! Stokes–pressure solve, yield classification, friction-angle chain, mode C, step size,
//! geometric SOCP in normal increments with outer acceptance, `ρ_s` transport, and the
//! rigid frame re-alignment, all on the general polyhedron of hypothesis (G). The spin
//! schedule is an exogenous input; neither angular momentum nor any Tier 3 coupling enters.

use std::borrow::Cow;

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
use super::relax::{AcceptError, accept, mass_correction};
use super::socp::{Kind, SocpInput, SocpOutput, SocpSeed, mass_jacobian, solve};
use super::spin::SpinSchedule;
use super::runout::{self, RunoutField, RunoutReport};
use super::transport::{Spread, mode_c, mode_c_over, surface_flow, transport};
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
    /// Smallest `n·ê` on the body kept by this step, and the face that carries it.
    /// Runout of the eroded and the routed packets together (§2.8b).
    pub runout: RunoutReport,
    /// Mass the slope relief removed and routed (kg), and the routing gap (§2.8b).
    pub relief_mass: f64,
    pub relief_gap: f64,
    /// Loose-layer mass the relief of promoted vertices moved through the ledger (kg, §2.8c).
    pub relief_layer: f64,
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
    let spec = &project.material.spec;
    let rho_bulk = spec.rho_p * spec.nu0;
    let t_end = opts.until.unwrap_or_else(|| spin.t_end());
    let snap_tol = 1e-12 * t_end.max(1.0);
    let mut records = Vec::new();
    let mut mass_target = fwd.moments.mass;
    let mut guess: Option<Vec<f64>> = if state.velocity.is_empty() {
        None
    } else {
        Some(state.velocity.clone())
    };
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

        // Steps 1–4.
        let load = Loading {
            omega,
            tide: spin.tide(state.t),
        };
        let mech = evaluate(project, &state.body, &mut fwd, load, guess.as_deref())?;
        let cl = &mech.class;
        let density = state.body.density()?;
        let area = {
            let _span = super::timing::span("vertex_areas");
            state.body.vertex_areas()
        };
        let r_mean = state.body.r_mean();
        let spread = Spread::new(&state.body, &area, project.mode_c.kernel_width * r_mean);
        // χ_Y from the unyielded core is authoritative for creep and mode C (§2.8 B3).
        // Slope promotion (§2.8c, A-C4) admits the loose-layer vertices of supercritical
        // faces to the geometric mobile set only.
        let yielded = cl.movable.clone();
        let slack: Vec<f64> = mech.rheo.iter().map(|node| node.slack).collect();
        let (movable, slope_promoted) = slope_promotion(
            &topo.faces,
            &yielded,
            &slack,
            &state.body.rho_s,
            project.relief.tol_slack,
        );
        let n_mobile = movable.iter().filter(|m| **m).count();
        let chi_a: Vec<bool> = (0..nv)
            .map(|i| yielded[i] && state.body.rho_s[i] > 0.0)
            .collect();

        // Step 5: mode C stays on the true yield set. Promoted slope nodes have F_δ ≤ 0.
        // §2.8b: where each eroded packet comes to rest. The sources are the nodes with
        // J_ero > 0; the arrivals do not depend on the step length, so both mode C
        // evaluations of this step share them.
        let positions = state.body.positions.clone();
        // ρ_K at the face centres, for the exterior Hessian of the flight expansion.
        let rho_face: Vec<f64> = topo
            .faces
            .iter()
            .map(|t| density.rho(t.iter().map(|v| positions[*v as usize]).sum::<DVec3>() / 3.0))
            .collect();
        // The frozen field of this step; the cone's density update must not reach it.
        let hess_face = fwd.face.hess_u.clone();
        let field = RunoutField {
            topo: &topo,
            index: &mech.index,
            positions: &positions,
            g_face: &mech.g_face,
            hess_face: &hess_face,
            rho_face: &rho_face,
            grav_const: project.g,
            load,
            material: &project.material,
            time_limit: project.mode_c.runout_time,
            stop: project.mode_c.runout_stop,
            theta: runout::THETA,
        };
        let arrivals = if project.mode_c.runout {
            let _span = super::timing::span("runout");
            let sources: Vec<usize> = (0..nv)
                .filter(|&i| yielded[i] && state.body.rho_s[i] > 0.0 && cl.f_delta[i] > 0.0)
                .collect();
            let exterior = |p: &[DVec3]| fwd.eval_exterior(p);
            field.trace_with(&sources, Some(&exterior))?
        } else {
            Vec::new()
        };
        let arrived = project.mode_c.runout.then_some(arrivals.as_slice());
        let mc = {
            let _span = super::timing::span("mode_c");
            mode_c(
                &spread,
                &area,
                &state.body.rho_s,
                &yielded,
                &cl.f_delta,
                &project.mode_c,
                arrived,
            )
        };
        let eroded = runout::eroded_masses(&area, &mc.j_ero, &arrivals, 1.0);
        let paths = runout::paths(&eroded, &arrivals, RUNOUT_PATHS);

        let mut rec = StepRecord {
            runout: RunoutReport::default(),
            relief_mass: 0.0,
            relief_gap: 0.0,
            relief_layer: 0.0,
            cores: CoreReport::default(),
            step: state.step,
            t: state.t,
            advance: 0.0,
            omega,
            kind: "geometry",
            dt: 0.0,
            dt_bound: "",
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
            n_deposit: 0,
            n_locked: 0,
            n_supercritical_faces: mech
                .rheo
                .iter()
                .filter(|n| n.slack > project.relief.tol_slack)
                .count(),
            max_slack: mech.rheo.iter().map(|n| n.slack).fold(0.0, f64::max),
            eroded_rate: mc.eroded_rate,
            flux_gap: flux_gap(&mc, &area),
            max_locked_step: 0.0,
            max_deposit_miss: 0.0,
            trust: 0.0,
            trust_retries: 0,
            socp_iterations: 0,
            slope_slack: 0.0,
            mass_slack: 0.0,
            dropped_overhang: 0,
            mass_residual: (fwd.moments.mass - mass_target).abs() / mass_target,
            max_displacement: 0.0,
            cone_unresolved: false,
            transport_fix: 0.0,
            rho_s_total: (0..nv).map(|i| area[i] * state.body.rho_s[i]).sum(),
            frame_applied: false,
            frame_angle_deg: 0.0,
            frame_mass_change: 0.0,
            nitsche_c: mech.ops.c_nitsche,
            rheo_unconverged: mech.rheo_stats.unconverged,
            rheo_seconds: mech.rheo_stats.seconds,
            mass: fwd.moments.mass,
            centroid_offset: fwd.moments.centroid.length(),
        };
        rec.runout = runout::report(&eroded, &arrivals);
        guess = Some(mech.sol.u.clone());
        state.velocity = mech.sol.u.clone();

        // Step 6: step size and the two shortcuts for an empty mobile set.
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
        // Stage equilibrium (2.15(c)) needs the whole body in the rigid core: an interior
        // yielded pocket would move the cores even with a locked surface.
        let all_core = cl.in_core.iter().all(|c| *c);
        if n_mobile == 0 && spin.is_stages() && all_core {
            // Stage equilibrium (criterion (c) of §2.15): jump to the end of the stage.
            let dt = t_next - state.t;
            rec.kind = "stage-equilibrium";
            rec.dt = dt;
            rec.advance = dt;
            rec.dt_bound = "t_next";
            rec.n_locked = nv;
            advance_time(&mut state, spin, dt, t_next, snap_tol);
            state.step += 1;
            steps_done += 1;
            let fields = step_fields(
                &positions_from,
                &mech,
                &mc,
                &slack,
                vec![0.0; nv],
                &paths,
                false,
                None,
            );
            observer(&rec, &state, &fields)?;
            records.push(rec);
            continue;
        }
        if n_mobile == 0 {
            // An empty movable set leaves the surface unchanged, but an interior yielded
            // pocket still moves the cores (6b, 8c), and the frame is re-aligned (8b).
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
            rec.frame_applied = frep.applied;
            rec.frame_angle_deg = frep.angle_deg;
            rec.frame_mass_change = frep.mass_change;
            state.body = body;
            fwd = new_fwd;
            if frep.applied {
                let (rot, trans) =
                    compose_frame(state.frame_rotation, state.frame_translation, &frep);
                state.frame_rotation = rot;
                state.frame_translation = trans;
                guess = None;
                state.velocity.clear();
                // The seed is indexed by vertex, and realignment resamples radii
                // onto the same rays, so it stays a warm start for the next cone.
                mass_target = fwd.moments.mass;
            }
            advance_time(&mut state, spin, rec.advance, t_next, snap_tol);
            state.step += 1;
            steps_done += 1;
            let fields = step_fields(
                &positions_from,
                &mech,
                &mc,
                &slack,
                vec![0.0; nv],
                &paths,
                false,
                Some(&frep),
            );
            observer(&rec, &state, &fields)?;
            records.push(rec);
            continue;
        }

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
        let max_flux = (0..nv)
            .map(|i| (mc.j_dep[i] - mc.j_ero[i]).abs())
            .fold(0.0, f64::max);
        let mut dt_c = f64::INFINITY;
        if spec.h_s > 0.0 && max_flux > 0.0 {
            dt_c = rho_bulk * spec.h_s / max_flux;
        }
        // The erosion itself is integrated exactly over the step (below), so only
        // the change of layer thickness bounds the step, not the decay rate.
        dt_c *= st.theta_c;
        let dt_char = if flow.grad_max > 0.0 {
            st.theta_char / flow.grad_max
        } else {
            f64::INFINITY
        };
        let bounds = [
            ("core", dt_core),
            ("cfl", dt_cfl),
            ("mode_c", dt_c),
            ("characteristic", dt_char),
            ("omega", omega_bound),
            ("dt_max", st.dt_max),
            ("t_next", t_next - state.t),
        ];
        let (mut name, mut dt) =
            bounds.iter().fold(
                ("dt_max", f64::INFINITY),
                |a, b| if b.1 < a.1 { *b } else { a },
            );
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
        rec.dt_c = dt_c;
        rec.dt_char = dt_char;
        rec.advance = st.lambda * dt;
        // Mode C over the chosen step: the exact average of the exponential layer decay.
        let mc = if mc.eroded_rate > 0.0 {
            let _span = super::timing::span("mode_c");
            mode_c_over(
                &spread,
                &area,
                &state.body.rho_s,
                &yielded,
                &cl.f_delta,
                &project.mode_c,
                arrived,
                rec.advance,
            )
        } else {
            mc
        };

        // Step 7: d_pred and the node classes L / D / A.
        let mut d_pred = vec![0.0; nv];
        let mut kind = vec![Kind::Locked; nv];
        let pos = &positions;
        for i in 0..nv {
            let n = cl.normal[i];
            let dh_star = dt * (mc.j_dep[i] - mc.j_ero[i]) / density.rho(pos[i]).max(1e-300);
            if movable[i] {
                // Slope-promoted nodes are still in the rigid core. Use the relax predictor
                // d_pred = 0 there; the slope cone is what moves them. Copying v_sl would
                // integrate the regularised creep the classifier is required to drop.
                let slide = if slope_promoted[i] {
                    0.0
                } else {
                    dt * cl.v_sl[i].dot(n)
                };
                d_pred[i] = slide + dh_star;
                kind[i] = Kind::Mobile;
            } else if dh_star != 0.0 {
                d_pred[i] = dh_star;
                kind[i] = Kind::Deposit;
            }
        }
        rec.n_deposit = kind.iter().filter(|k| **k == Kind::Deposit).count();
        rec.n_locked = kind.iter().filter(|k| **k == Kind::Locked).count();
        // S_slope (§2.8c): the loose faces with at least one free vertex, supercritical
        // or not. The subcritical rows start inactive and keep the relief from pushing
        // a neighbour past repose; fixed vertices enter n̂ with their fixed increments.
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
        // V_slope, the free vertices of the slope faces: where the relief removes mass (§2.8b).
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
        let route = project.mode_c.runout && slope_vertex.iter().any(|v| *v);
        let mut routing: Option<(runout::Routing, f64)> = None;
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
        let (lower, upper) = clearance(&state.body, &cl.normal, fwd.body_mesh(), g.theta_g, &floor_of(&slope_promoted, &d_pred, &state.body.rho_s, &area, &jac))?;
        // Outer geometric acceptance (§2.13, §10.4.4 step 8a): a self-intersection,
        // an overhang or a lost volume shrinks the trust radius and re-solves step 7;
        // time does not advance on a rejected attempt.
        let mut trust = g.trust0 * r_mean;
        let mut retries = 0usize;
        let mut rejected: Option<String> = None;
        let (out, dmax, new_body, new_fwd, resid, d_used) = loop {
            if trust < g.trust_min * r_mean {
                let why = rejected.as_deref().unwrap_or("rejected");
                if !why.starts_with("cone solve") {
                    return Err(SimError::geometry(format!(
                        "step {}: no acceptable geometry down to Δ = {:.3e} m ({why})",
                        state.step,
                        trust / g.trust_shrink,
                    )));
                }
                // §10.4.3: the cone program itself, not the geometry, failed at the
                // floor. The step keeps its geometry and is flagged.
                eprintln!("step {}: geometry kept, {why}", state.step);
                rec.cone_unresolved = true;
                let b = Body::new(
                    topo.clone(),
                    state.body.positions.clone(),
                    state.body.bulk,
                    kernels_pred.clone(),
                    state.body.rho_s.clone(),
                )?;
                let f = Forward::bind(&b, project.g)?;
                let resid = (f.moments.mass - mass_target).abs() / mass_target;
                break (SocpOutput::unresolved(nv, topo.n_faces()), 0.0, b, f, resid, d_pred.clone());
            }
            // One cone solve. A warm seed is never discarded, and a mass or volume
            // miss does not start PDHG again.
            let inp = SocpInput {
                body: &state.body,
                kind: &kind,
                d_pred: &d_pred,
                weight: &weight,
                gravity: &mech.g_face,
                sin_phi: &sin_phi,
                slope_row: &slope_row,
                mass_jac: &jac,
                mass_residual: mass_target - fwd.moments.mass,
                mass: mass_target,
                trust,
                eps_n: g.eps_n,
                lower: &lower,
                upper: &upper,
                varrho: g.varrho,
                warm: socp_seed.as_ref(),
            };
            let out = match solve(&inp) {
                Ok(out) => out,
                // A cone solve that misses KKT_TOL is a rejected attempt like a bad
                // geometry: the smaller box is a better-conditioned program.
                Err(err) if err.kind == "SolverFailure" => {
                    rejected = Some(format!("cone solve: {}", err.detail));
                    socp_seed = None;
                    trust *= g.trust_shrink;
                    retries += 1;
                    continue;
                }
                Err(err) => return Err(err),
            };
            // §2.8b: route the mass the relief removed from the slope vertices, then solve
            // again from the first solution with the deposit added to the prediction.
            let removed: Vec<f64> = if route {
                (0..nv)
                    .map(|i| if slope_vertex[i] { (d_pred[i] - out.d[i]).max(0.0) } else { 0.0 })
                    .collect()
            } else {
                Vec::new()
            };
            let (out, kind_used, d_pred_used) = if removed.iter().any(|r| *r > 0.0) {
                let routed = {
                    let _span = super::timing::span("relief_routing");
                    let exterior = |p: &[DVec3]| fwd.eval_exterior(p);
                    runout::route_relief(&field, &area, &jac, &removed, &spread, Some(&exterior))?
                };
                let mut d2 = d_pred.clone();
                let mut k2 = kind.clone();
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
                            .filter(|&i| slope_vertex[i])
                            .map(|i| jac[i] * (d2[i] - out2.d[i]).max(0.0))
                            .sum();
                        let gap = if first > 0.0 { (again - first).abs() / first } else { 0.0 };
                        routing = Some((routed, gap));
                        (out2, Cow::Owned(k2), Cow::Owned(d2))
                    }
                    Err(err) if err.kind == "SolverFailure" => {
                        rejected = Some(format!("cone solve (routed): {}", err.detail));
                        socp_seed = None;
                        trust *= g.trust_shrink;
                        retries += 1;
                        continue;
                    }
                    Err(err) => return Err(err),
                }
            } else {
                routing = None;
                (out, Cow::Borrowed(&kind), Cow::Borrowed(&d_pred))
            };
            socp_seed = Some(out.seed.clone());
            let (max_locked_step, max_deposit_miss) = class_steps(&kind_used, &d_pred_used, &out.d);
            rec.max_locked_step = max_locked_step;
            rec.max_deposit_miss = max_deposit_miss;
            // 8a: X⁽ⁿ⁺¹⁾ = X⁽ⁿ⁾ + λ d_* N (G4).
            let moved: Vec<DVec3> = (0..nv)
                .map(|i| state.body.positions[i] + cl.normal[i] * (st.lambda * out.d[i]))
                .collect();
            let b = match {
                let _span = super::timing::span("geometry_build");
                Body::new(
                    topo.clone(),
                    moved,
                    state.body.bulk,
                    kernels_pred.clone(),
                    state.body.rho_s.clone(),
                )
            } {
                Ok(b) => b,
                Err(e) => {
                    rejected = Some(format!("body construction: {e}"));
                    trust *= g.trust_shrink;
                    retries += 1;
                    continue;
                }
            };
            let f = match Forward::bind(&b, project.g) {
                Ok(f) => f,
                Err(e) => {
                    rejected = Some(format!("{e}"));
                    trust *= g.trust_shrink;
                    retries += 1;
                    continue;
                }
            };
            let dmax = out.d.iter().fold(0.0_f64, |m, v| m.max(v.abs())) * st.lambda;
            // Mass misses get one mobile-set correction (§2.14). That correction is
            // not a second cone solve; a residual that remains is recorded.
            let (new_body, new_fwd, resid) = match accept(&f, mass_target, g.eps_mass) {
                Ok(r) => (b, f, r),
                Err(AcceptError::Geometry(why)) => {
                    rejected = Some(why.to_string());
                    trust *= g.trust_shrink;
                    retries += 1;
                    continue;
                }
                Err(AcceptError::Mass(resid)) => {
                    let dm = f.moments.mass - mass_target;
                    let mut chosen = (b, f, resid);
                    if let Some(rr) =
                        mass_correction(&chosen.0.positions, &cl.normal, &kind, &jac, dm, trust)
                    {
                        if let Ok(cb) = {
                            let _span = super::timing::span("geometry_build");
                            Body::new(
                                topo.clone(),
                                rr,
                                chosen.0.bulk,
                                chosen.0.kernels.clone(),
                                chosen.0.rho_s.clone(),
                            )
                        } {
                            if let Ok(cf) = Forward::bind(&cb, project.g) {
                                match accept(&cf, mass_target, g.eps_mass) {
                                    Ok(cr) | Err(AcceptError::Mass(cr)) => chosen = (cb, cf, cr),
                                    Err(AcceptError::Geometry(_)) => {}
                                }
                            }
                        }
                    }
                    chosen
                }
            };
            break (out, dmax, new_body, new_fwd, resid, d_pred_used.into_owned());
        };
        rec.trust = trust;
        rec.trust_retries = retries;
        rec.socp_iterations = out.iterations;
        rec.slope_slack = out.slope_slack;
        rec.mass_slack = out.mass_slack;
        rec.dropped_overhang = out.dropped_overhang;
        rec.mass_residual = resid;
        rec.max_displacement = dmax;
        // 2.15(c), second route: on a stage with the whole body in the core, no erosion and
        // no deposit, a resolved cone program that leaves every free vertex in place is a
        // fixed point; the remaining slack is unrelievable. The step jumps to the stage end
        // with the geometry unchanged.
        let fixed_point = spin.is_stages()
            && all_core
            && rec.n_deposit == 0
            && mc.j_ero.iter().all(|j| *j == 0.0)
            && !rec.cone_unresolved
            && out.d.iter().all(|d| d.abs() <= D_ZERO * r_mean);
        if fixed_point {
            drop((new_body, new_fwd));
            let dt = t_next - state.t;
            rec.kind = "stage-equilibrium";
            rec.dt = dt;
            rec.advance = dt;
            rec.dt_bound = "t_next";
            rec.max_displacement = 0.0;
            advance_time(&mut state, spin, dt, t_next, snap_tol);
            state.step += 1;
            steps_done += 1;
            let fields = step_fields(
                &positions_from,
                &mech,
                &mc,
                &slack,
                vec![0.0; nv],
                &paths,
                false,
                None,
            );
            observer(&rec, &state, &fields)?;
            records.push(rec);
            continue;
        }
        // §2.8c: the relief of a promoted vertex comes out of its loose layer and is laid
        // down where the routing of §2.8b sends it; Σ A ρ_s is unchanged.
        let (layer_debit, layer_credit) = {
            let debit: Vec<f64> = (0..nv)
                .map(|i| {
                    if slope_promoted[i] && out.d.len() == nv {
                        // Capped by the carried layer plus what arrives on it this step.
                        let m = st.lambda * jac[i] * (d_used[i] - out.d[i]).max(0.0);
                        m.min(state.body.rho_s[i] * area[i] + st.lambda * jac[i] * d_used[i].max(0.0))
                    } else {
                        0.0
                    }
                })
                .collect();
            if debit.iter().any(|m| *m > 0.0) {
                let per_area: Vec<f64> = (0..nv).map(|i| debit[i] / area[i]).collect();
                let routed: Vec<runout::Arrival> = routing
                    .as_ref()
                    .map(|(r, _)| {
                        r.arrivals
                            .iter()
                            .filter(|a| slope_promoted[a.source])
                            .cloned()
                            .collect()
                    })
                    .unwrap_or_default();
                let moved = runout::moved_source(&area, &per_area, &routed);
                let laid = spread.spread(&moved);
                let credit: Vec<f64> = (0..nv).map(|j| laid[j] * area[j]).collect();
                (debit, credit)
            } else {
                (debit, vec![0.0; nv])
            }
        };
        rec.relief_layer = layer_debit.iter().sum();
        // Runout of the eroded packets over the step and of the routed relief together.
        let mut weights = runout::eroded_masses(&area, &mc.j_ero, &arrivals, rec.advance);
        let mut packets = arrivals;
        if let Some((routed, gap)) = routing {
            rec.relief_mass = routed.masses.iter().sum();
            rec.relief_gap = gap;
            weights.extend(routed.masses);
            packets.extend(routed.arrivals);
        }
        rec.runout = runout::report(&weights, &packets);
        let paths = runout::paths(&weights, &packets, RUNOUT_PATHS);

        // ρ_s: semi-Lagrangian transport (λΔt), then the mode C source.
        let adv = st.lambda * dt;
        let area_new = {
            let _span = super::timing::span("vertex_areas");
            new_body.vertex_areas()
        };
        let (mut rho_s, fix) = {
            let _span = super::timing::span("transport");
            transport(&state.body, &flow, &state.body.rho_s, &area, &area_new, adv)
        };
        for i in 0..nv {
            // The step-averaged J_ero is the exact exponential decay, applied once (§2.8).
            rho_s[i] = (rho_s[i]
                + adv * (mc.j_dep[i] - mc.j_ero[i])
                + (layer_credit[i] - layer_debit[i]) / area_new[i])
                .max(0.0);
        }
        rec.transport_fix = fix;
        let mut body = new_body;
        body.rho_s = rho_s;
        let mut new_fwd = new_fwd;
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
        rec.rho_s_total = (0..nv).map(|i| area_new[i] * body.rho_s[i]).sum();

        // 8b: frame re-alignment (only when triggered).
        let (body, frep, new_fwd) = realign(&body, Some(new_fwd), project.g, g, false)?;
        rec.frame_applied = frep.applied;
        rec.frame_angle_deg = frep.angle_deg;
        rec.frame_mass_change = frep.mass_change;
        if frep.applied {
            let (rot, trans) = compose_frame(state.frame_rotation, state.frame_translation, &frep);
            state.frame_rotation = rot;
            state.frame_translation = trans;
            guess = None;
            state.velocity.clear();
            // Same rays after the frame update: keep the cone seed.
            mass_target = new_fwd.moments.mass;
        }
        state.body = body;
        fwd = new_fwd;

        // 9: advance time.
        advance_time(&mut state, spin, adv, t_next, snap_tol);
        state.step += 1;
        steps_done += 1;
        let fields = step_fields(
            &positions_from,
            &mech,
            &mc,
            &slack,
            out.d.iter().map(|d| st.lambda * d).collect(),
            &paths,
            true,
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

/// The loose-mass bound of §2.8c: a promoted vertex cannot cut below its carried layer plus
/// the step's deposit, `d_i ≥ min(d_pred,i, 0) − ρ_s,i A_i / (DM)_i ≤ 0`, so `d = 0` stays
/// feasible; `−∞` elsewhere.
fn floor_of(promoted: &[bool], d_pred: &[f64], rho_s: &[f64], area: &[f64], jac: &[f64]) -> Vec<f64> {
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
fn clearance(
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

/// `|d_*| ≤ D_ZERO R̄` is a zero displacement for the stage fixed point (round-off only).
const D_ZERO: f64 = 1e-12;
/// Repetitions of the step-6b core prediction after a midpoint violation.
const CORE_REPEATS: usize = 8;

fn flux_gap(mc: &super::transport::ModeC, area: &[f64]) -> f64 {
    let ero: f64 = mc.j_ero.iter().zip(area).map(|(j, a)| j * a).sum();
    let dep: f64 = mc.j_dep.iter().zip(area).map(|(j, a)| j * a).sum();
    (dep - ero).abs() / ero.max(dep).max(1e-30)
}

fn class_steps(kind: &[Kind], d_pred: &[f64], d: &[f64]) -> (f64, f64) {
    let mut locked = 0.0_f64;
    let mut deposit = 0.0_f64;
    for i in 0..kind.len() {
        match kind[i] {
            Kind::Locked => locked = locked.max(d[i].abs()),
            Kind::Deposit => deposit = deposit.max((d[i] - d_pred[i]).abs()),
            Kind::Mobile => {}
        }
    }
    (locked, deposit)
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
