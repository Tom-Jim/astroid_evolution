//! The quasi-static evolution loop of §10.4.4.
//!
//! One pass of [`run`] is steps 1–9 of the doc: forward gravity (Tier 1), Stokes–pressure
//! Picard, yield classification, friction-angle chain (`rheo_friction`), mode C, step size,
//! geometric SOCP with outer acceptance, `ρ_s` transport, frame re-alignment. The spin
//! schedule is an exogenous input; neither angular momentum nor any Tier 3 coupling enters.

use nalgebra::{Matrix3, Vector3};
use serde::Serialize;

use crate::common::math::DVec3;
use crate::rheo_friction::RheoGpu;

use super::body::Body;
use super::error::ScfError;
use super::forward::Forward;
use super::frame::{FrameReport, realign, vertex_areas};
use super::mechanics::evaluate;
use super::params::Project;
use super::relax::{AcceptError, accept};
use super::socp::{Kind, SocpInput, SocpSeed, mass_jacobian, solve};
use super::spin::SpinSchedule;
use super::transport::{DirIndex, locate_face, locate_vertex, mode_c, surface_flow, transport};

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
    pub transport_fix: f64,
    pub rho_s_total: f64,
    pub frame_applied: bool,
    pub frame_angle_deg: f64,
    pub frame_mass_change: f64,
    pub fit_error: f64,
    pub rheo_unconverged: u32,
    pub rheo_seconds: f64,
    pub mass: f64,
    pub centroid_offset: f64,
    /// Smallest `n·ê` on the body kept by this step, and the face that carries it.
    pub min_radial: f64,
    pub min_radial_face: usize,
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
/// `radius_from` is the ray radius at the start of the step. `d_radial` is `λ d` on a
/// geometry step and zero on a spin or stage-equilibrium step; a frame realignment does
/// not resample it. Every other vertex field is resampled onto the post-realignment body
/// when the frame report was applied. `movable` is the classifier flag (true means the
/// vertex is outside the rigid core).
#[derive(Clone, Debug)]
pub struct StepFields {
    pub radius_from: Vec<f64>,
    pub f_delta: Vec<f64>,
    pub movable: Vec<bool>,
    pub v_sl: Vec<DVec3>,
    pub normal: Vec<DVec3>,
    pub j_ero: Vec<f64>,
    pub j_dep: Vec<f64>,
    pub d_radial: Vec<f64>,
    /// Per face. Incident-face scatter happens when the snapshot is packed.
    pub slack: Vec<f64>,
    pub g_face: Vec<DVec3>,
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
pub fn start(project: &Project, body: Body) -> Result<(State, Forward), ScfError> {
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
    gpu: &RheoGpu,
    spin: &SpinSchedule,
    mut state: State,
    mut fwd: Forward,
    opts: &RunOptions,
    observer: &mut dyn FnMut(&StepRecord, &State, &StepFields) -> Result<(), ScfError>,
) -> Result<RunResult, ScfError> {
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
            return Err(ScfError::step_limit_exhausted(format!(
                "step.max_steps = {} exhausted at t = {:.6e}",
                st.max_steps, state.t
            )));
        }
        let topo = state.body.topo.clone();
        let nv = topo.n_vertices();
        let radius_from = state.body.radius.clone();
        let sample = spin.sample(state.t);
        let t_next = sample.t_next.min(t_end);
        let omega = sample.omega;
        state.omega = omega;

        // Steps 1–4.
        let mech = evaluate(project, gpu, &state.body, &mut fwd, omega, guess.as_deref())?;
        let cl = &mech.class;
        let density = state.body.density()?;
        let area = {
            let _span = super::timing::span("vertex_areas");
            vertex_areas(&state.body)
        };
        // χ_Y from the unyielded core is authoritative for creep and mode C (§2.8 B3).
        // A separate case: the whole surface sits in that core, but the loose-layer
        // friction chain still reports slack above `relief.tol_slack`. The regularised
        // creep is then below γ̇_Y, so F_δ never opens A, and the slope SOCP of §10.4.3
        // would be skipped even though the rheology proxy says the sheet is supercritical.
        // Those vertices are admitted to the geometric mobile set only. Their v_sl is
        // core creep and must not enter d_pred (§2.8: χ_Y = 0 drops that creep).
        let yielded = cl.movable.clone();
        let mut movable = yielded.clone();
        let mut slope_promoted = vec![false; nv];
        if yielded.iter().all(|m| !*m) {
            for (f, node) in mech.rheo.iter().enumerate() {
                if node.slack > project.relief.tol_slack {
                    for &v in &topo.faces[f] {
                        let i = v as usize;
                        if state.body.rho_s[i] > 0.0 {
                            movable[i] = true;
                            slope_promoted[i] = true;
                        }
                    }
                }
            }
        }
        let n_mobile = movable.iter().filter(|m| **m).count();
        let chi_a: Vec<bool> = (0..nv)
            .map(|i| yielded[i] && state.body.rho_s[i] > 0.0)
            .collect();

        // Step 5: mode C stays on the true yield set. Promoted slope nodes have F_δ ≤ 0.
        let mc = {
            let _span = super::timing::span("mode_c");
            mode_c(
                &topo,
                &area,
                &state.body.rho_s,
                &yielded,
                &cl.f_delta,
                &project.mode_c,
            )
        };
        let slack: Vec<f64> = mech.rheo.iter().map(|node| node.slack).collect();

        let mut rec = StepRecord {
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
            transport_fix: 0.0,
            rho_s_total: (0..nv).map(|i| area[i] * state.body.rho_s[i]).sum(),
            frame_applied: false,
            frame_angle_deg: 0.0,
            frame_mass_change: 0.0,
            fit_error: mech.fit_error,
            rheo_unconverged: mech.rheo_stats.unconverged,
            rheo_seconds: mech.rheo_stats.seconds,
            mass: fwd.moments.mass,
            centroid_offset: fwd.moments.centroid.length(),
            min_radial: 0.0,
            min_radial_face: 0,
        };
        guess = Some(mech.sol.u.clone());
        state.velocity = mech.sol.u.clone();

        // Step 6: step size and the two shortcuts for an empty mobile set.
        let omega_bound = if spin.is_stages() {
            f64::INFINITY
        } else {
            st.eps_omega * omega.max(st.omega_ref) / sample.omega_dot.abs().max(1e-30)
        };
        rec.dt_omega = omega_bound;
        if n_mobile == 0 && spin.is_stages() {
            let dt = if spin.is_stages() {
                t_next - state.t
            } else {
                omega_bound.min(st.dt_max).min(t_next - state.t)
            };
            rec.kind = if spin.is_stages() {
                "stage-equilibrium"
            } else {
                "spin-only"
            };
            rec.dt = dt;
            rec.advance = dt;
            rec.dt_bound = if spin.is_stages() {
                "t_next"
            } else {
                "omega/max/t_next"
            };
            rec.n_locked = nv;
            advance_time(&mut state, dt, t_next, snap_tol);
            state.step += 1;
            steps_done += 1;
            stamp_radial(&mut rec, &state.body);
            let fields = step_fields(
                &radius_from,
                &cl.f_delta,
                &cl.movable,
                &cl.v_sl,
                &cl.normal,
                &mc.j_ero,
                &mc.j_dep,
                &radial_increments(false, st.lambda, nv, None),
                &slack,
                &mech.g_face,
                false,
                None,
                &state.body,
            );
            observer(&rec, &state, &fields)?;
            records.push(rec);
            continue;
        }
        if n_mobile == 0 {
            // A spin-only step still executes frame re-alignment (§10.4.4 step 8b),
            // even though the geometry has no movable vertices.
            let (body, frep, new_fwd) = realign(&state.body, Some(fwd), project.g, g, false)?;
            rec.kind = "spin-only";
            rec.dt = (omega_bound.min(st.dt_max)).min(t_next - state.t);
            rec.advance = rec.dt;
            rec.dt_bound = "omega/max/t_next";
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
            advance_time(&mut state, rec.advance, t_next, snap_tol);
            state.step += 1;
            steps_done += 1;
            stamp_radial(&mut rec, &state.body);
            let fields = step_fields(
                &radius_from,
                &cl.f_delta,
                &cl.movable,
                &cl.v_sl,
                &cl.normal,
                &mc.j_ero,
                &mc.j_dep,
                &radial_increments(false, st.lambda, nv, None),
                &slack,
                &mech.g_face,
                false,
                Some(&frep),
                &state.body,
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
        let dt_cfl = cfl_dt(
            &state.body,
            &yielded,
            &cl.normal,
            &cl.v_sl,
            st.theta_cfl,
        );
        let flow = {
            let _span = super::timing::span("surface_flow");
            surface_flow(&topo, &cl.v_sl, &chi_a, &cl.radius)
        };
        let max_flux = (0..nv)
            .map(|i| (mc.j_dep[i] - mc.j_ero[i]).abs())
            .fold(0.0, f64::max);
        let mut dt_c = f64::INFINITY;
        if spec.h_s > 0.0 && max_flux > 0.0 {
            dt_c = rho_bulk * spec.h_s / max_flux;
        }
        if project.mode_c.k_e > 0.0 && mc.max_excess > 0.0 {
            dt_c = dt_c.min(project.mode_c.sigma_0 / (project.mode_c.k_e * mc.max_excess));
        }
        dt_c *= st.theta_c;
        let dt_char = if flow.grad_max > 0.0 {
            st.theta_char / flow.grad_max
        } else {
            f64::INFINITY
        };
        let bounds = [
            ("cfl", dt_cfl),
            ("mode_c", dt_c),
            ("characteristic", dt_char),
            ("omega", omega_bound),
            ("dt_max", st.dt_max),
            ("t_next", t_next - state.t),
        ];
        let (name, dt) =
            bounds.iter().fold(
                ("dt_max", f64::INFINITY),
                |a, b| if b.1 < a.1 { *b } else { a },
            );
        if !(dt.is_finite() && dt > 0.0) {
            return Err(ScfError::solver(format!("step size {dt} is not positive")));
        }
        rec.dt = dt;
        rec.dt_bound = name;
        rec.dt_cfl = dt_cfl;
        rec.dt_c = dt_c;
        rec.dt_char = dt_char;
        rec.advance = st.lambda * dt;

        // Step 7: d_pred and the node classes L / D / A.
        let mut d_pred = vec![0.0; nv];
        let mut kind = vec![Kind::Locked; nv];
        let pos = state.body.positions();
        for i in 0..nv {
            let n = cl.normal[i];
            let mu = n.dot(topo.dirs[i]);
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
                d_pred[i] = radial_increment(slide + dh_star, mu);
                kind[i] = Kind::Mobile;
            } else if dh_star != 0.0 {
                d_pred[i] = radial_increment(dh_star, mu);
                kind[i] = Kind::Deposit;
            }
        }
        rec.n_deposit = kind.iter().filter(|k| **k == Kind::Deposit).count();
        rec.n_locked = kind.iter().filter(|k| **k == Kind::Locked).count();
        // S_slope is the supercritical loose sheet only. Subcritical faces stay
        // in static friction: their yielded vertices may still follow d_pred,
        // but they do not enter the slope cone.
        let slope_row: Vec<bool> = mech
            .rheo
            .iter()
            .enumerate()
            .map(|(f, node)| {
                let v = topo.faces[f].map(|x| x as usize);
                node.slack > project.relief.tol_slack
                    && node.sin_phi.is_finite()
                    && v.iter().all(|i| kind[*i] == Kind::Mobile)
                    && v.iter().map(|i| state.body.rho_s[*i]).sum::<f64>() > 0.0
            })
            .collect();
        let sin_phi: Vec<f64> = mech.rheo.iter().map(|n| n.sin_phi).collect();
        let weight: Vec<f64> = (0..nv).map(|i| area[i]).collect();
        let jac = {
            let _span = super::timing::span("mass_jacobian");
            mass_jacobian(&state.body, &density)
        };
        let r_mean = state.body.radius.iter().sum::<f64>() / nv as f64;
        let trust = g.trust0 * r_mean;
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
            varrho: g.varrho,
            warm: socp_seed.as_ref(),
        };
        let out = match solve(&inp) {
            Ok(out) => out,
            Err(err) if err.kind == "SolverFailure" => {
                return Err(ScfError::solver(format!(
                    "step {}: {}",
                    state.step, err.detail
                )));
            }
            Err(err) => return Err(err),
        };
        socp_seed = Some(out.seed.clone());
        let (max_locked_step, max_deposit_miss) = class_steps(&kind, &d_pred, &out.d);
        rec.max_locked_step = max_locked_step;
        rec.max_deposit_miss = max_deposit_miss;
        // 8a: R⁽ⁿ⁺¹⁾ = R⁽ⁿ⁾ + λ d_*.
        let mut radius = state.body.radius.clone();
        for i in 0..nv {
            radius[i] += st.lambda * out.d[i];
        }
        let b = match {
            let _span = super::timing::span("geometry_build");
            Body::new(
                topo.clone(),
                radius,
                state.body.bulk,
                state.body.kernels.clone(),
                state.body.rho_s.clone(),
            )
        } {
            Ok(b) => b,
            Err(e) => {
                return Err(ScfError::geometry(format!(
                    "step {}: body construction failed at Δ = {trust:.3e} m ({e})",
                    state.step
                )));
            }
        };
        let f = match Forward::bind(&b, project.g) {
            Ok(f) => f,
            Err(e) => {
                return Err(ScfError::geometry(format!(
                    "step {}: Forward::bind failed at Δ = {trust:.3e} m ({e})",
                    state.step
                )));
            }
        };
        let dmax = out.d.iter().fold(0.0_f64, |m, v| m.max(v.abs())) * st.lambda;
        // Mass misses get one mobile-set correction (§2.14). That correction is
        // not a second cone solve; a residual that remains is recorded.
        let (new_body, new_fwd, resid) = match accept(&f, mass_target, g.eps_mass) {
            Ok(r) => (b, f, r),
            Err(AcceptError::Geometry(why)) => {
                return Err(ScfError::geometry(format!(
                    "step {}: geometry rejected at Δ = {trust:.3e} m ({why})",
                    state.step
                )));
            }
            Err(AcceptError::Mass(resid)) => {
                let dm = f.moments.mass - mass_target;
                let denom: f64 = kind
                    .iter()
                    .enumerate()
                    .filter(|(_, k)| **k == Kind::Mobile)
                    .map(|(i, _)| jac[i] * jac[i])
                    .sum();
                let mut chosen = (b, f, resid);
                if denom > 0.0 {
                    let alpha = (-dm / denom).clamp(-trust, trust);
                    let mut rr = chosen.0.radius.clone();
                    for i in 0..nv {
                        if kind[i] == Kind::Mobile {
                            rr[i] += alpha * jac[i];
                        }
                    }
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
        rec.trust = trust;
        rec.trust_retries = 0;
        rec.socp_iterations = out.iterations;
        rec.slope_slack = out.slope_slack;
        rec.mass_slack = out.mass_slack;
        rec.dropped_overhang = out.dropped_overhang;
        rec.mass_residual = resid;
        rec.max_displacement = dmax;

        // ρ_s: semi-Lagrangian transport (λΔt), then the mode C source.
        let adv = st.lambda * dt;
        let area_new = {
            let _span = super::timing::span("vertex_areas");
            vertex_areas(&new_body)
        };
        let j_old: Vec<f64> = (0..nv).map(|i| area[i] / topo.vertex_solid[i]).collect();
        let j_new: Vec<f64> = (0..nv)
            .map(|i| area_new[i] / topo.vertex_solid[i])
            .collect();
        let (mut rho_s, fix) = {
            let _span = super::timing::span("transport");
            transport(&topo, &flow, &state.body.rho_s, &j_old, &j_new, adv)
                .map_err(|e| ScfError::solver(format!("surface transport failed: {e}")))?
        };
        for i in 0..nv {
            rho_s[i] = (rho_s[i] + adv * (mc.j_dep[i] - mc.j_ero[i])).max(0.0);
        }
        rec.transport_fix = fix;
        let mut body = new_body;
        body.rho_s = rho_s;
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
        advance_time(&mut state, adv, t_next, snap_tol);
        state.step += 1;
        steps_done += 1;
        stamp_radial(&mut rec, &state.body);
        let fields = step_fields(
            &radius_from,
            &cl.f_delta,
            &cl.movable,
            &cl.v_sl,
            &cl.normal,
            &mc.j_ero,
            &mc.j_dep,
            &radial_increments(true, st.lambda, nv, Some(&out.d)),
            &slack,
            &mech.g_face,
            true,
            Some(&frep),
            &state.body,
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

fn advance_time(state: &mut State, dt: f64, t_next: f64, tol: f64) {
    state.t += dt;
    if (state.t - t_next).abs() < tol {
        state.t = t_next;
    }
}

/// `|n·ê|` below this is left unconverted. The vertex takes no radial step from
/// that normal displacement, and it does not enter the CFL speed.
const MU_MIN: f64 = 0.05;

fn radial_increment(normal_disp: f64, mu: f64) -> f64 {
    if mu.abs() >= MU_MIN {
        normal_disp / mu
    } else {
        0.0
    }
}

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

/// `θ min_i h_i / s_i` over the yielded movable set.
///
/// `h_i` is the shortest edge at the node and `s_i = |v_sl · n| / |n · ê|` is
/// the radial rate of §10.4.1. `θ` is the configured CFL factor, at most 1/2.
fn cfl_dt(body: &Body, yielded: &[bool], normal: &[DVec3], v_sl: &[DVec3], theta: f64) -> f64 {
    let position = body.positions();
    let dirs = &body.topo.dirs;
    let mut dt = f64::INFINITY;
    for i in 0..yielded.len() {
        if !yielded[i] {
            continue;
        }
        let mu = normal[i].dot(dirs[i]);
        if mu.abs() < MU_MIN {
            continue;
        }
        let speed = v_sl[i].dot(normal[i]).abs() / mu;
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

fn stamp_radial(rec: &mut StepRecord, body: &Body) {
    let (mu, face) = body.min_radial();
    rec.min_radial = mu;
    rec.min_radial_face = face;
}

/// `λ d` when this step moved the geometry, otherwise a zero increment per vertex.
pub fn radial_increments(flow_active: bool, lambda: f64, nv: usize, d: Option<&[f64]>) -> Vec<f64> {
    if !flow_active {
        return vec![0.0; nv];
    }
    let mut out = Vec::with_capacity(nv);
    for i in 0..nv {
        let value = d.and_then(|row| row.get(i)).copied().unwrap_or(0.0);
        out.push(lambda * value);
    }
    out
}

#[allow(clippy::too_many_arguments)]
pub fn step_fields(
    radius_from: &[f64],
    f_delta: &[f64],
    movable: &[bool],
    v_sl: &[DVec3],
    normal: &[DVec3],
    j_ero: &[f64],
    j_dep: &[f64],
    d_radial: &[f64],
    slack: &[f64],
    g_face: &[DVec3],
    flow_active: bool,
    frame: Option<&FrameReport>,
    body: &Body,
) -> StepFields {
    let nv = body.topo.n_vertices();
    let nf = body.topo.n_faces();
    let Some(frame) = frame.filter(|report| report.applied) else {
        return StepFields {
            radius_from: fit_copy(radius_from, nv, 0.0),
            f_delta: fit_copy(f_delta, nv, 0.0),
            movable: fit_copy(movable, nv, false),
            v_sl: fit_copy(v_sl, nv, DVec3::ZERO),
            normal: fit_copy(normal, nv, DVec3::ZERO),
            j_ero: fit_copy(j_ero, nv, 0.0),
            j_dep: fit_copy(j_dep, nv, 0.0),
            d_radial: fit_copy(d_radial, nv, 0.0),
            slack: fit_copy(slack, nf, 0.0),
            g_face: fit_copy(g_face, nf, DVec3::ZERO),
            flow_active,
        };
    };
    let q = frame.rotation;
    let index = DirIndex::build(&body.topo.dirs);
    let movable_w: Vec<f64> = (0..nv)
        .map(|i| {
            if movable.get(i).copied().unwrap_or(false) {
                1.0
            } else {
                0.0
            }
        })
        .collect();
    let v_rot: Vec<DVec3> = (0..v_sl.len()).map(|i| mul_q(q, v_sl[i])).collect();
    let n_rot: Vec<DVec3> = (0..normal.len())
        .map(|i| {
            let turned = mul_q(q, normal[i]);
            turned.try_normalize().unwrap_or(turned)
        })
        .collect();
    let g_rot: Vec<DVec3> = (0..g_face.len()).map(|i| mul_q(q, g_face[i])).collect();
    let mut out_f = vec![0.0; nv];
    let mut out_m = vec![false; nv];
    let mut out_v = vec![DVec3::ZERO; nv];
    let mut out_n = vec![DVec3::ZERO; nv];
    let mut out_ero = vec![0.0; nv];
    let mut out_dep = vec![0.0; nv];
    for i in 0..nv {
        let x_new = body.topo.dirs[i] * body.radius[i];
        let dir = old_direction(q, frame.translation, x_new, body.topo.dirs[i]);
        let (ids, w) = locate_vertex(&body.topo, &index, dir);
        out_f[i] = blend_f64(f_delta, ids, w);
        out_m[i] = blend_f64(&movable_w, ids, w) > 0.5;
        out_v[i] = blend_vec(&v_rot, ids, w);
        out_n[i] = blend_vec(&n_rot, ids, w)
            .try_normalize()
            .unwrap_or(body.topo.dirs[i]);
        out_ero[i] = blend_f64(j_ero, ids, w);
        out_dep[i] = blend_f64(j_dep, ids, w);
    }
    let positions = body.positions();
    let mut out_slack = vec![0.0; nf];
    let mut out_g = vec![DVec3::ZERO; nf];
    for (f, tri) in body.topo.faces.iter().enumerate() {
        let [a, b, c] = tri.map(|i| positions[i as usize]);
        let centroid = (a + b + c) * (1.0 / 3.0);
        let fallback = body
            .topo
            .face_dirs
            .get(f)
            .copied()
            .unwrap_or(DVec3::new(0.0, 0.0, 1.0));
        let dir = old_direction(q, frame.translation, centroid, fallback);
        let face = locate_face(&body.topo, &index, dir);
        out_slack[f] = slack.get(face).copied().unwrap_or(0.0);
        out_g[f] = g_rot.get(face).copied().unwrap_or(DVec3::ZERO);
    }
    StepFields {
        radius_from: fit_copy(radius_from, nv, 0.0),
        f_delta: out_f,
        movable: out_m,
        v_sl: out_v,
        normal: out_n,
        j_ero: out_ero,
        j_dep: out_dep,
        d_radial: fit_copy(d_radial, nv, 0.0),
        slack: out_slack,
        g_face: out_g,
        flow_active,
    }
}

fn fit_copy<T: Copy>(src: &[T], n: usize, zero: T) -> Vec<T> {
    let mut out = vec![zero; n];
    for (slot, value) in out.iter_mut().zip(src) {
        *slot = *value;
    }
    out
}

fn mul_q(q: [[f64; 3]; 3], v: DVec3) -> DVec3 {
    DVec3::new(
        q[0][0] * v.x + q[0][1] * v.y + q[0][2] * v.z,
        q[1][0] * v.x + q[1][1] * v.y + q[1][2] * v.z,
        q[2][0] * v.x + q[2][1] * v.y + q[2][2] * v.z,
    )
}

fn mul_qt(q: [[f64; 3]; 3], v: DVec3) -> DVec3 {
    DVec3::new(
        q[0][0] * v.x + q[1][0] * v.y + q[2][0] * v.z,
        q[0][1] * v.x + q[1][1] * v.y + q[2][1] * v.z,
        q[0][2] * v.x + q[1][2] * v.y + q[2][2] * v.z,
    )
}

fn old_direction(q: [[f64; 3]; 3], translation: DVec3, x_new: DVec3, fallback: DVec3) -> DVec3 {
    (mul_qt(q, x_new) + translation)
        .try_normalize()
        .unwrap_or(fallback)
}

fn blend_f64(field: &[f64], ids: [usize; 3], w: [f64; 3]) -> f64 {
    ids.iter()
        .zip(w)
        .map(|(id, weight)| field.get(*id).copied().unwrap_or(0.0) * weight)
        .sum()
}

fn blend_vec(field: &[DVec3], ids: [usize; 3], w: [f64; 3]) -> DVec3 {
    ids.iter().zip(w).fold(DVec3::ZERO, |acc, (id, weight)| {
        acc + field.get(*id).copied().unwrap_or(DVec3::ZERO) * weight
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::radial_increment;
    use super::{IDENTITY, radial_increments, step_fields};
    use crate::common::density::CauchyKernel;
    use crate::common::math::DVec3;
    use crate::scf_simulator::body::{Body, Topology};
    use crate::scf_simulator::frame::{FrameDeparture, FrameReport};

    #[test]
    fn radial_increment_keeps_a_transverse_face_and_drops_a_grazing_one() {
        assert!((radial_increment(0.1, 0.2) - 0.5).abs() < 1e-12);
        assert!((radial_increment(0.1, -0.2) + 0.5).abs() < 1e-12);
        assert_eq!(radial_increment(10.0, 0.01), 0.0);
        assert_eq!(radial_increment(10.0, -0.01), 0.0);
        assert_eq!(radial_increment(10.0, 0.0), 0.0);
    }

    #[test]
    fn a_spin_step_records_zero_radial_increment() {
        let zeros = radial_increments(false, 0.4, 5, Some(&[1.0, 2.0, 3.0, 4.0, 5.0]));
        assert!(zeros.iter().all(|value| *value == 0.0));
        let scaled = radial_increments(true, 0.5, 3, Some(&[1.0, 2.0, 4.0]));
        assert_eq!(scaled, vec![0.5, 1.0, 2.0]);
    }

    #[test]
    fn realigned_fields_keep_the_post_step_vertex_count() {
        let topo = Arc::new(Topology::icosphere(0));
        let nv = topo.n_vertices();
        let nf = topo.n_faces();
        let body = Body::new(
            topo.clone(),
            vec![1.0; nv],
            1000.0,
            Vec::<CauchyKernel>::new(),
            vec![0.0; nv],
        )
        .unwrap();
        let f_delta: Vec<f64> = (0..nv).map(|i| i as f64).collect();
        let movable: Vec<bool> = (0..nv).map(|i| i % 2 == 0).collect();
        let v_sl = topo.dirs.clone();
        let slack: Vec<f64> = (0..nf).map(|f| f as f64).collect();
        let g_face = topo.face_dirs.clone();
        let d_radial: Vec<f64> = (0..nv).map(|i| 0.1 * i as f64).collect();
        let report = FrameReport {
            applied: true,
            passes: 1,
            translation: DVec3::ZERO,
            rotation: IDENTITY,
            angle_deg: 0.0,
            before: FrameDeparture {
                centroid: 0.0,
                off_diagonal: 0.0,
                ordering: 0.0,
            },
            after: FrameDeparture {
                centroid: 0.0,
                off_diagonal: 0.0,
                ordering: 0.0,
            },
            mass_change: 0.0,
            volume_change: 0.0,
            max_radius_change: 0.0,
        };
        let fields = step_fields(
            &vec![2.0; nv],
            &f_delta,
            &movable,
            &v_sl,
            &topo.dirs,
            &f_delta,
            &f_delta,
            &d_radial,
            &slack,
            &g_face,
            false,
            Some(&report),
            &body,
        );
        assert_eq!(fields.f_delta.len(), body.radius.len());
        assert_eq!(fields.v_sl.len(), body.radius.len());
        assert_eq!(fields.normal.len(), body.radius.len());
        assert_eq!(fields.movable.len(), body.radius.len());
        assert_eq!(fields.d_radial.len(), body.radius.len());
        assert_eq!(fields.slack.len(), body.topo.n_faces());
        assert_eq!(fields.radius_from, vec![2.0; nv]);
        assert!(!fields.flow_active);
        for i in 0..nv {
            assert!(
                (fields.f_delta[i] - f_delta[i]).abs() < 1.0e-6,
                "vertex {i}: {} vs {}",
                fields.f_delta[i],
                f_delta[i]
            );
            assert_eq!(fields.movable[i], movable[i]);
            assert!((fields.d_radial[i] - d_radial[i]).abs() < 1.0e-12);
            assert!((fields.v_sl[i] - v_sl[i]).length() < 1.0e-6);
        }
        for f in 0..nf {
            assert!(
                (fields.slack[f] - slack[f]).abs() < 1.0e-6,
                "face {f}: {} vs {}",
                fields.slack[f],
                slack[f]
            );
        }
    }
}
