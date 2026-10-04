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
use super::relax::accept;
use super::socp::{Kind, SocpInput, mass_jacobian, solve};
use super::spin::SpinSchedule;
use super::transport::{mode_c, surface_flow, transport};

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

#[derive(Clone, Debug, Serialize)]
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
    observer: &mut dyn FnMut(&StepRecord, &State) -> Result<(), ScfError>,
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
    let mut steps_done = 0usize;

    while state.t < t_end - snap_tol {
        if let Some(m) = opts.max_steps {
            if steps_done >= m {
                break;
            }
        }
        if steps_done >= st.max_steps {
            return Err(ScfError::trust_exhausted(format!(
                "step.max_steps = {} exhausted at t = {:.6e}",
                st.max_steps, state.t
            )));
        }
        let topo = state.body.topo.clone();
        let nv = topo.n_vertices();
        let sample = spin.sample(state.t);
        let t_next = sample.t_next.min(t_end);
        let omega = sample.omega;
        state.omega = omega;

        // Steps 1–4.
        let mech = evaluate(project, gpu, &state.body, &mut fwd, omega, guess.as_deref())?;
        let cl = &mech.class;
        let density = state.body.density()?;
        let area = vertex_areas(&state.body);
        let n_mobile = cl.movable.iter().filter(|m| **m).count();
        let chi_a: Vec<bool> = (0..nv).map(|i| cl.movable[i] && state.body.rho_s[i] > 0.0).collect();

        // Step 5: mode C.
        let mc = mode_c(&topo, &area, &state.body.rho_s, &cl.movable, &cl.f_delta, &project.mode_c);

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
        if n_mobile == 0 {
            let dt = if spin.is_stages() {
                t_next - state.t
            } else {
                omega_bound.min(st.dt_max).min(t_next - state.t)
            };
            rec.kind = if spin.is_stages() { "stage-equilibrium" } else { "spin-only" };
            rec.dt = dt;
            rec.advance = dt;
            rec.dt_bound = if spin.is_stages() { "t_next" } else { "omega/max/t_next" };
            rec.n_locked = nv;
            advance_time(&mut state, dt, t_next, snap_tol);
            state.step += 1;
            steps_done += 1;
            observer(&rec, &state)?;
            records.push(rec);
            continue;
        }

        let h_min = state.body.min_edge();
        let mut max_normal_speed = 0.0_f64;
        for i in 0..nv {
            if cl.movable[i] {
                let n = cl.normal[i];
                let s = cl.v_sl[i].dot(n).abs() / n.dot(topo.dirs[i]);
                max_normal_speed = max_normal_speed.max(s);
            }
        }
        let flow = surface_flow(&topo, &cl.v_sl, &chi_a, &cl.radius);
        let dt_cfl = if max_normal_speed > 0.0 {
            st.theta_cfl * h_min / max_normal_speed
        } else {
            f64::INFINITY
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
        let (name, dt) = bounds
            .iter()
            .fold(("dt_max", f64::INFINITY), |a, b| if b.1 < a.1 { *b } else { a });
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
            let cos = n.dot(topo.dirs[i]);
            let dh_star = dt * (mc.j_dep[i] - mc.j_ero[i]) / density.rho(pos[i]).max(1e-300);
            if cl.movable[i] {
                d_pred[i] = (dt * cl.v_sl[i].dot(n) + dh_star) / cos;
                kind[i] = Kind::Mobile;
            } else if dh_star != 0.0 {
                d_pred[i] = dh_star / cos;
                kind[i] = Kind::Deposit;
            }
        }
        rec.n_deposit = kind.iter().filter(|k| **k == Kind::Deposit).count();
        rec.n_locked = kind.iter().filter(|k| **k == Kind::Locked).count();
        let slope_row: Vec<bool> = topo
            .faces
            .iter()
            .map(|t| {
                let v = t.map(|x| x as usize);
                v.iter().all(|i| kind[*i] == Kind::Mobile)
                    && v.iter().map(|i| state.body.rho_s[*i]).sum::<f64>() > 0.0
            })
            .collect();
        let sin_phi: Vec<f64> = mech.rheo.iter().map(|n| n.sin_phi).collect();
        let weight: Vec<f64> = (0..nv).map(|i| area[i]).collect();
        let jac = mass_jacobian(&state.body, &density);
        let r_mean = state.body.radius.iter().sum::<f64>() / nv as f64;
        let mut trust = g.trust0 * r_mean;
        let mut retries = 0usize;
        let (new_body, new_fwd, resid, out, dmax) = loop {
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
                c0: g.c0,
                eps_n: g.eps_n,
                varrho: g.varrho,
            };
            let out = solve(&inp)?;
            // 8a: R⁽ⁿ⁺¹⁾ = R⁽ⁿ⁾ + λ d_*.
            let mut radius = state.body.radius.clone();
            for i in 0..nv {
                radius[i] += st.lambda * out.d[i];
            }
            let verdict = Body::new(
                topo.clone(),
                radius,
                state.body.bulk,
                state.body.kernels.clone(),
                state.body.rho_s.clone(),
            )
            .map_err(|e| e.to_string())
            .and_then(|b| {
                let f = Forward::bind(&b, project.g).map_err(|e| e.to_string())?;
                let r = accept(&b, &f, mass_target, g.c0, g.eps_mass)?;
                Ok((b, f, r))
            });
            match verdict {
                Ok((b, f, r)) => {
                    let dmax = out.d.iter().fold(0.0_f64, |m, v| m.max(v.abs())) * st.lambda;
                    break (b, f, r, out, dmax);
                }
                Err(why) => {
                    trust *= g.trust_shrink;
                    retries += 1;
                    if trust < g.trust_min * r_mean {
                        return Err(ScfError::trust_exhausted(format!(
                            "step {}: geometry rejected down to Δ = {trust:.3e} m ({why})",
                            state.step
                        )));
                    }
                }
            }
        };
        rec.trust = trust;
        rec.trust_retries = retries;
        rec.socp_iterations = out.iterations;
        rec.slope_slack = out.slope_slack;
        rec.mass_slack = out.mass_slack;
        rec.dropped_overhang = out.dropped_overhang;
        rec.mass_residual = resid;
        rec.max_displacement = dmax;

        // ρ_s: semi-Lagrangian transport (λΔt), then the mode C source.
        let adv = st.lambda * dt;
        let area_new = vertex_areas(&new_body);
        let j_old: Vec<f64> = (0..nv).map(|i| area[i] / topo.vertex_solid[i]).collect();
        let j_new: Vec<f64> = (0..nv).map(|i| area_new[i] / topo.vertex_solid[i]).collect();
        let (mut rho_s, fix) = transport(&topo, &flow, &state.body.rho_s, &j_old, &j_new, adv);
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
            mass_target = new_fwd.moments.mass;
        }
        state.body = body;
        fwd = new_fwd;

        // 9: advance time.
        advance_time(&mut state, adv, t_next, snap_tol);
        state.step += 1;
        steps_done += 1;
        observer(&rec, &state)?;
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
