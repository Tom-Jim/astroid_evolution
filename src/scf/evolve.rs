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
use super::evolving::spin::principal;
use super::evolving::markers::{Driving, Reconstruction};
use super::evolving::material;
use super::evolving::{Evolving, MaterialReport, Markers, SpinLaw, SpinReport, StateField, Thermal, ThermalReport};
use super::stokes::MaterialField;
use super::surface::SurfaceIndex;
use super::domain::volume_rule;
use super::forward::Forward;
use super::forward::Loading;
use super::frame::{FrameReport, realign};
use super::mechanics::{Evolution, Mechanics, evaluate};
use super::params::Project;
use super::relief::{Relief, clearance, relieve};
use super::runout::RunoutReport;
use super::settle::{SettleReport, settle};
use super::socp::{Kind, mass_jacobian};
use super::cycle::{self, CycleOut, CycleReport};
use super::orbit::{Necks, OrbitLaw, OrbitReport, OrbitState};
use super::spin::SpinSchedule;
use super::stokes::Guess;
use super::transport::{advect, surface_flow, transport};
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
    /// The state of Chapter 11.
    pub evolving: Evolving,
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
    /// The deep state (Stokes, classification, face chain) was reused from an earlier
    /// step within the lag budget (§10.4.4).
    pub lagged: bool,
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
    /// A-L (§11.7); inactive while the spin is the exogenous table. Inactive reports
    /// are left out of the step log.
    #[serde(skip_serializing_if = "SpinReport::inactive")]
    pub spin: SpinReport,
    /// A-Θ (§11.6); inactive without the thermal section.
    #[serde(skip_serializing_if = "ThermalReport::inactive")]
    pub thermal: ThermalReport,
    /// A-O (§10.4.6); inactive without the orbit section.
    #[serde(skip_serializing_if = "OrbitReport::inactive")]
    pub orbit: OrbitReport,
    /// A-Λ (§10.4.7); inactive without the cycle section.
    #[serde(skip_serializing_if = "CycleReport::inactive")]
    pub cycle: CycleReport,
    /// The state markers, A-ϖ and A-Ξ (§11.1–11.5); inactive without them.
    #[serde(skip_serializing_if = "MaterialReport::inactive")]
    pub material: MaterialReport,
}

#[derive(Clone, Debug)]
pub struct RunOptions {
    pub max_steps: Option<usize>,
    /// Stop at this time instead of `T_end` of the schedule.
    pub until: Option<f64>,
    /// Reuse the deep state within the lag budget (§10.4.4 step 4+); `false` evaluates it
    /// on every step (audit runs).
    pub lag: bool,
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
            evolving: Evolving::default(),
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
    let law = SpinLaw::new(project.spin_law.as_ref(), spin)?;
    if let Some(l) = &law {
        l.init(&mut state.evolving.spin, state.t, &fwd.moments);
    }
    let mut thermal = project.thermal.as_ref().map(Thermal::new);
    if thermal.is_some() && state.evolving.surface_damage.is_none() {
        state.evolving.surface_damage = Some(vec![0.0; state.body.topo.n_vertices()]);
    }
    let rho_bulk = project.material.rho_bulk;
    // §11.1: one marker per volume node of the initial body (A-ϖ, A-Ξ).
    if project.markers.is_some() && state.evolving.markers.is_none() {
        let s = &project.stokes;
        let (pts, ..) = volume_rule(fwd.body_mesh(), s.n_theta, s.n_phi, s.n_radial, s.cut_depth)?;
        state.evolving.markers = Some(Markers::seed(
            &pts,
            project.dilatancy.as_ref().map_or(0.0, |d| d.initial),
            project.damage.as_ref().map_or(0.0, |d| d.initial),
        ));
    }
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
    let mut steps_done = 0usize;
    // The last evaluated deep state, and whether the next step must evaluate a new one.
    let mut deep: Option<Deep> = None;
    let mut fresh_deep = true;
    // Chapter 11 state follows the fresh fields every step, so it never lags.
    let mut geometry_cache = super::mechanics::GeometryCache::default();
    // A-O (§10.4.6): the orbit follows the ledger mass, which leaves through the necks.
    let orbit_law = match &project.orbit {
        Some(p) => {
            let o = *state
                .evolving
                .orbit
                .get_or_insert_with(|| OrbitState {
                    e: project.cycle.as_ref().map_or(0.0, |c| c.eccentricity),
                    ..OrbitLaw::start(spin, state.t, fwd.moments.mass)
                });
            Some(OrbitLaw::new(p, spin, o.m0)?)
        }
        None => None,
    };
    let lag_allowed = opts.lag && law.is_none() && thermal.is_none() && project.markers.is_none() && project.cycle.is_none();

    while state.t < t_end - snap_tol {
        if orbit_law.zip(state.evolving.orbit).is_some_and(|(l, o)| l.ended(&o)) {
            break;
        }
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
        // A-L: ω_n = L_n / I_1⁽ⁿ⁾ on the current (settled) inertia; else the table.
        let orbit = orbit_law.zip(state.evolving.orbit);
        let omega = match (&state.evolving.spin, orbit) {
            (Some(s), _) if law.is_some() => SpinLaw::omega(s, &fwd.moments),
            (_, Some((l, o))) => l.mean_motion(o.a),
            _ => sample.omega,
        };
        state.omega = omega;
        let r_star = state.body.r_star();
        // Steps 1–4: the deep state of the slow operator (§10.4.4).
        let load = match orbit {
            Some((l, o)) => l.load(o.a),
            None => Loading { omega, tide: spin.tide(state.t) },
        };
        let necks = match orbit {
            Some((_, o)) => Some(Necks::locate(&fwd, project.g, o.mass, load)?),
            None => None,
        };
        let lagged = lag_allowed
            && !fresh_deep
            && deep.as_ref().is_some_and(|d| d.origin.within(&state.body, load, st.eps_omega));
        // Released before a new evaluation: its operators are reused only when no one
        // else holds them.
        let kept = deep.take().filter(|_| lagged);
        let mut origin = kept.as_ref().map(|d| d.origin.clone());
        // What Chapter 11 contributes to the step; the default is the reference closure.
        let (mut mech, fallbacks) = if let Some(d) = kept {
            (d.mech, 0)
        } else {
            let index = super::mechanics::geometry(&mut geometry_cache, project, &state.body, &fwd)?.0;
            let field = match (&state.evolving.markers, &project.markers) {
                (Some(mk), Some(mp)) => Some(StateField::new(
                    mk,
                    &index,
                    mp.radius * state.body.r_mean(),
                    project.damage.is_some(),
                    project.dilatancy.as_ref(),
                )),
                _ => None,
            };
            let ev = Evolution {
                field: field.as_ref().map(|f| f as &dyn MaterialField),
                surface_damage: state.evolving.surface_damage.as_deref(),
            };
            let mech = evaluate(&mut geometry_cache, project, &state.body, &mut fwd, load, guess.as_ref(), ev)?;
            origin = Some(DeepOrigin::of(&state.body, load));
            (mech, field.map_or(0, |f| f.fallbacks.into_inner()))
        };
        fresh_deep = false;
        // A-Λ (§10.4.7): the tidal cycle on the deep state, and its ratchet.
        let cyc: Option<CycleOut> = match (&project.cycle, orbit) {
            (Some(p), Some((l, o))) if o.e > 0.0 => Some(cycle::evaluate(project, p, &mech, &fwd, l.mean_motion(o.a), o.e)?),
            _ => None,
        };
        let ratchet_v: Vec<DVec3> = match &cyc {
            Some(c) if c.ratchet.iter().any(|v| *v != 0.0) => {
                mech.ratchet = c.ratchet.clone();
                let v = contract_velocity_rays(&mech.surf, &mech.ops.lay, &state.body.positions, 1, Want::V, &c.ratchet).v;
                v.chunks_exact(3).map(|w| DVec3::new(w[0], w[1], w[2])).collect()
            }
            _ => Vec::new(),
        };
        // A-Θ on the frozen shape: fatigue rates and the radiative torque of law L2.
        let heat = thermal
            .as_mut()
            .map(|th| {
                // §11.6: a satellite's primary eclipses the Sun and lights the faces toward it.
                if let (Some(p), Some((_, o))) = (&project.orbit, orbit) {
                    th.set_primary(p.primary_radius, o.a);
                }
                th.evaluate(&state.body, &mech.index, omega, fwd.moments.centroid)
            });
        let t_rad = heat.as_ref().map_or(0.0, |h| h.report.torque);
        if let (Some(l), Some(s)) = (&law, state.evolving.spin.as_mut()) {
            l.calibrate(s, state.t, t_rad)?;
        }
        let dt_theta = match (&thermal, &heat) {
            (Some(th), Some(h)) => th.dt_bound(&h.rate, &state.body.rho_s, rho_bulk, omega),
            _ => f64::INFINITY,
        };
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
            thermal: heat.as_ref().map(|h| h.report).unwrap_or_default(),
            material: MaterialReport {
                active: state.evolving.markers.is_some(),
                fallbacks,
                ..MaterialReport::default()
            },
            ..StepRecord::default()
        };
        rec.lagged = lagged;
        if let Some(c) = &cyc {
            rec.cycle = CycleReport {
                ratchet_speed: ratchet_v.iter().map(|v| v.length()).fold(0.0, f64::max),
                ..c.report
            };
        }
        if let (Some((l, o)), Some(nk)) = (orbit, &necks) {
            rec.orbit = OrbitReport {
                active: true,
                a: o.a,
                n: l.mean_motion(o.a),
                mass: o.mass,
                r_hill: nk.r_hill,
                lobe_margin: nk.margin(&state.body.positions),
                ..OrbitReport::default()
            };
            // §2.17.3: a certified L* > 1 ends the quasi-static route; under A-Λ the peak
            // multiplier of the cycle does (Proposition 10.4.7.5).
            let every = project.orbit.as_ref().map_or(0, |p| p.limit_every);
            if every > 0 && !lagged && state.step % every == 0 {
                let (lo, what) = match (&cyc, &project.cycle) {
                    (Some(c), Some(p)) => {
                        let (b, cb) = super::mechanics::limit_cycle(project, &mech, &fwd, load, &c.elastic, p.phases)?;
                        (rec.orbit.limit_lo, rec.orbit.limit_hi) = (b.lo, b.hi);
                        rec.cycle.peak_lo = cb.peak_lo;
                        (rec.cycle.shakedown_lo, rec.cycle.shakedown_hi) = (cb.shakedown_lo, cb.shakedown_hi);
                        (cb.peak_lo, "L_peak (Proposition 10.4.7.5)")
                    }
                    _ => {
                        let b = super::mechanics::limit_bracket(project, &mech, &fwd, load)?;
                        (rec.orbit.limit_lo, rec.orbit.limit_hi) = (b.lo, b.hi);
                        (b.lo, "L* (§2.17.3)")
                    }
                };
                if lo > 1.0 {
                    return Err(SimError::limit_load_collapse(format!(
                        "certified {what} ≥ {lo:.4} > 1 at a = {:.6e} m",
                        o.a
                    )));
                }
            }
        }
        if !lagged {
            guess = Some(mech.sol.guess());
            state.velocity = mech.sol.u.clone();
        }

        // Stage equilibrium (2.15(c)): the whole body in the rigid core (an interior
        // yielded pocket would move the cores even with a locked surface) and a settled
        // surface. The stage then jumps to its end.
        let all_core = cl.in_core.iter().all(|c| *c);
        // Proposition 11.14: the stage shortcut needs the slow state of Chapter 11 at rest.
        let thermal_rest = heat.as_ref().is_none_or(|h| h.rate.iter().all(|r| *r == 0.0))
            && project.thermal.as_ref().is_none_or(|p| {
                p.healing == 0.0
                    || state.evolving.surface_damage.as_ref().is_none_or(|d| d.iter().all(|v| *v == 0.0))
            });
        let material_rest = project.damage.as_ref().is_none_or(|d| {
            d.healing == 0.0
                || state.evolving.markers.as_ref().is_none_or(|m| m.damage.iter().all(|v| *v == 0.0))
        });
        let stage_jump = spin.is_stages()
            && law.as_ref().is_none_or(SpinLaw::allows_stage_jump)
            && thermal_rest
            && material_rest;
        if n_mobile == 0 && stage_jump && all_core {
            let creep_m = fwd.moments;
            let settled = settle(
                project,
                load,
                state.body.clone(),
                fwd,
                &yielded,
                &cl.f_delta,
                t_next - state.t,
                mass_target,
                state.step,
                Some(&mech.face_law),
                None,
                None,
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
                let t0 = state.t;
                advance_time(&mut state, spin, dt, t_next, snap_tol);
                spin_step(&law, &mut state, t0, t_rad, omega, &creep_m, &fwd.moments, &mut rec)?;
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

        // Definition 11.3: the driving rate on the volume nodes of this geometry, shared by
        // Δt_ϖ, J_ν and the markers.
        // A-Λ adds the alternating plastic shear of the cycle and the fatigue rate (§10.4.7).
        let plastic: Vec<f64> = match &cyc {
            Some(c) => mech.class.plastic_rate.iter().zip(&c.alternating).map(|(p, a)| p + a).collect(),
            None => mech.class.plastic_rate.clone(),
        };
        let driving = match (&state.evolving.markers, &project.markers) {
            (Some(_), Some(mp)) => {
                let d = Driving::new(
                    &mech.ops.vol.pts,
                    &mech.ops.vol.w,
                    &plastic,
                    &mech.class.inertial_vol,
                    mp.nonlocal_length(r_star, project.stokes.l_velocity),
                    mp.radius * r_mean,
                    &mech.index,
                );
                Some(match &cyc {
                    Some(c) => d.with_fatigue(c.fatigue.clone()),
                    None => d,
                })
            }
            _ => None,
        };
        // A-ϖ (§11.5): Δt_ϖ.
        let dt_damage = match (&project.damage, &state.evolving.markers, &project.markers, &driving) {
            (Some(d), Some(mk), Some(mp), Some(dr)) => {
                let rec_d = Reconstruction::new(mk, &mech.index, mp.radius * r_mean);
                material::dt_bound(d, mk, &rec_d, &dr.rate, &dr.fatigue, &mech.ops.vol.pts)
            }
            _ => f64::INFINITY,
        };
        if project.damage.is_some() {
            rec.material.dt_damage = dt_damage;
        }
        // A-Ξ (§11.4): the density ratio J_ν of the material over the step, and Δt_Ξ.
        let dil_ratio = match (&project.dilatancy, &state.evolving.markers, &project.markers, &driving) {
            (Some(d), Some(mk), Some(mp), Some(dr)) => Some(DilatationRatio::new(d, mk, mp, &mech, dr, r_mean)),
            _ => None,
        };
        let dt_xi = match (&project.dilatancy, &mech.sol.dilatancy) {
            (Some(d), Some(s)) if d.k > 0.0 && s.driving_max > 0.0 => {
                d.theta_step / (d.k * d.range().hi * s.driving_max)
            }
            _ => f64::INFINITY,
        };
        if let Some(s) = &mech.sol.dilatancy {
            rec.material.dilatation_defect = s.defect;
            rec.material.dt_xi = dt_xi;
        }

        // Step 6: step size of the slow operator.
        let omega_bound = match (&law, &state.evolving.spin) {
            (Some(l), Some(s)) => {
                let torque = l.torque(s, state.t, t_rad);
                if torque == 0.0 {
                    f64::INFINITY
                } else {
                    let i1 = principal(&fwd.moments).moments[0];
                    SpinLaw::dt_bound(st.eps_omega, st.omega_ref, omega, i1, torque)
                }
            }
            _ if spin.is_stages() => f64::INFINITY,
            _ => {
                let rate = orbit.map_or(sample.omega_dot, |(l, o)| l.ndot(o.a, o.mass));
                st.eps_omega * omega.max(st.omega_ref) / rate.abs().max(1e-30)
            }
        };
        // A-O: the step that brings the orbit to the primary ends exactly there.
        let primary_bound = orbit.map_or(f64::INFINITY, |(l, o)| l.time_to_floor(&o) / st.lambda);
        // A-Λ: e ∝ a^(C_e + β) over the step (Proposition 10.4.7.1), and the step in which e
        // changes by the fraction ε_ω.
        let e_exponent = match (&project.cycle, orbit) {
            (Some(p), Some((l, o))) if o.e > 0.0 => {
                let n = l.mean_motion(o.a);
                cycle::eccentricity_exponent(project, p, &l, &fwd, n * n * o.a.powi(3), o.mass, n)
            }
            _ => 0.0,
        };
        let omega_bound = match orbit {
            Some((l, o)) if e_exponent != 0.0 => omega_bound.min(st.eps_omega * o.a / (e_exponent.abs() * l.adot(o.a, o.mass).abs())),
            _ => omega_bound,
        };
        // The ratchet moves every vertex: θ_CFL h / |v_r · N| (§10.4.7).
        let dt_ratchet = if ratchet_v.is_empty() {
            f64::INFINITY
        } else {
            cfl_dt(&state.body, &vec![true; nv], &cl.normal, &ratchet_v, st.theta_cfl)
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
                let tau = st.lambda * *dt;
                let ratio = |m: &[DVec3]| dil_ratio.as_ref().map_or_else(|| vec![1.0; m.len()], |d| d.at(m, tau));
                let ks = cores::predict(
                    &state.body.kernels,
                    tau,
                    &v_adv,
                    &in_body,
                    probe_h,
                    &project.cores,
                    dil_ratio.as_ref().map(|_| &ratio as &cores::Ratio),
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
        let (creep_d, frep, creep_m) = if n_mobile == 0 {
            // No yielded surface: the creep step leaves the surface in place, but an
            // interior yielded pocket still moves the cores (6b, 8c), and the frame is
            // re-aligned (8b).
            let bounds = [
                ("core", dt_core),
                ("ratchet", dt_ratchet),
                ("thermal", dt_theta),
                ("dilatancy", dt_xi),
                ("damage", dt_damage),
                ("omega", omega_bound),
                ("primary", primary_bound),
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
            if ratchet(&mut body, &ratchet_v, &cl.normal, advance) {
                fwd = Forward::bind(&body, project.g)?;
            }
            if core_rate > 0.0 || dil_ratio.is_some() {
                body.kernels = kernels_pred;
                fwd.set_density(body.density()?)?;
                close_cores(
                    project,
                    &mech,
                    &state.body,
                    &mut body,
                    &mut fwd,
                    mass_target,
                    advance,
                    dil_ratio.as_ref(),
                    &mut rec.cores,
                )?;
            }
            if let (Some(th), Some(h)) = (&thermal, &heat) {
                let dmg = state.evolving.surface_damage.as_mut().expect("A-Θ state");
                rec.thermal.produced_mass =
                    th.advance(advance, omega, &h.rate, &area, rho_bulk, dmg, &mut body.rho_s);
            }
            let creep_m = fwd.moments;
            let (body, frep, new_fwd) = realign(&body, Some(fwd), project.g, g, false)?;
            rec.kind = "spin-only";
            rec.dt = dt;
            rec.advance = advance;
            rec.dt_bound = name;
            state.body = body;
            fwd = new_fwd;
            (vec![0.0; nv], frep, creep_m)
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
                ("ratchet", dt_ratchet),
                ("cfl", dt_cfl),
                ("characteristic", dt_char),
                ("thermal", dt_theta),
                ("dilatancy", dt_xi),
                ("damage", dt_damage),
                ("omega", omega_bound),
                ("primary", primary_bound),
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
            let mut jac = {
                let _span = super::timing::span("mass_jacobian");
                mass_jacobian(&state.body, &density_cand)
            };
            // A-Ξ: ρ_cand = ρ_K(θ_pred) − ρ_b (1 − J_ν). The deficit enters (DM)_i at the
            // vertices and, integrated over the body, the target of the mass row (§11.4).
            let mut cone_target = mass_target;
            if let Some(r) = &dil_ratio {
                let tau = st.lambda * dt;
                let bulk = state.body.bulk;
                let jv = r.arriving(&state.body.positions, tau);
                let av = state.body.area_vectors();
                for i in 0..nv {
                    jac[i] -= bulk * (1.0 - jv[i]) * av[i].length();
                }
                let vol = &mech.ops.vol;
                let jq = r.arriving(&vol.pts, tau);
                cone_target += bulk * vol.w.iter().zip(&jq).map(|(w, j)| w * (1.0 - j)).sum::<f64>();
            }
            // Clearance rows of (G4): −θ_g t⁻ ≤ d ≤ θ_g t⁺ along the vertex normals, with
            // t^± the distance to the next crossing of Γ (two first-hit queries).
            let (lower, mut upper) = clearance(
                &state.body,
                &cl.normal,
                fwd.body_mesh(),
                g.theta_g,
                &vec![f64::NEG_INFINITY; nv],
            )?;
            // A-O lobe cap (§10.4.6): movable material carried beyond a neck plane is in
            // the outer region at rest and leaves; the cap bounds the increment there.
            if let Some(nk) = &necks {
                for i in 0..nv {
                    let Some(cap) = nk.cap(state.body.positions[i], cl.normal[i]) else { continue };
                    if kind[i] != Kind::Mobile || st.lambda * d_pred[i] <= cap {
                        continue;
                    }
                    upper[i] = (cap / st.lambda).min(upper[i]).max(lower[i]);
                    rec.orbit.escaped_cap += jac[i] * st.lambda * (d_pred[i] - upper[i]).max(0.0);
                }
                mass_target -= rec.orbit.escaped_cap;
                cone_target -= rec.orbit.escaped_cap;
            }
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
                mass_target: cone_target,
                scale: st.lambda,
                routing: None,
                step: state.step,
            };
            let rel = relieve(&relief, &fwd)?;
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
            // A-Θ over λΔt: ϖ_Γ moves with the slipping rock surface (§11.6), then the
            // surface damage and the shielded production (Proposition 11.11).
            if let (Some(th), Some(h)) = (&thermal, &heat) {
                let dmg = state.evolving.surface_damage.as_mut().expect("A-Θ state");
                let rock = surface_flow(&state.body, &cl.normal, &cl.v_sl, &yielded);
                *dmg = advect(&state.body, &rock, dmg, rec.advance);
                rec.thermal.produced_mass =
                    th.advance(rec.advance, omega, &h.rate, &area_new, rho_bulk, dmg, &mut body.rho_s);
            }
            let mut new_fwd = rel.fwd;
            if ratchet(&mut body, &ratchet_v, &cl.normal, rec.advance) {
                new_fwd = Forward::bind(&body, project.g)?;
            }
            // 8c: Levels 1 and 2 of the core transport (§9.10.4–5).
            if cores_on || dil_ratio.is_some() {
                close_cores(
                    project,
                    &mech,
                    &state.body,
                    &mut body,
                    &mut new_fwd,
                    mass_target,
                    rec.advance,
                    dil_ratio.as_ref(),
                    &mut rec.cores,
                )?;
            }
            // 8b: frame re-alignment (only when triggered).
            let creep_m = new_fwd.moments;
            let (body, frep, new_fwd) = realign(&body, Some(new_fwd), project.g, g, false)?;
            state.body = body;
            fwd = new_fwd;
            (creep_d, frep, creep_m)
        };
        // 8a+ (§11.1): the markers move along v_adv with the exact state update at the
        // midpoints (Propositions 11.5, 11.9), then follow the rigid motion of 8b.
        if let (Some(mk), Some(mp), Some(dr)) = (state.evolving.markers.as_mut(), &project.markers, &driving) {
            let rate = |pts: &[DVec3]| {
                let (g, ic) = dr.at(pts);
                let nu_c = ic
                    .iter()
                    .map(|i| project.dilatancy.as_ref().map_or(0.0, |d| d.nu_c(*i)))
                    .collect();
                (g, nu_c, dr.fatigue_at(pts))
            };
            material::advance(
                mk,
                rec.advance,
                &|p| advection(&mech, p),
                &rate,
                project.damage.as_ref(),
                project.dilatancy.as_ref(),
            );
            if frep.applied {
                mk.realign(frep.rotation, frep.translation);
            }
            rec.material.ell_ratio = mp.nonlocal;
        }
        rec.frame_applied = frep.applied;
        // A realignment moves every vertex: the next step evaluates afresh.
        fresh_deep |= frep.applied;
        rec.frame_angle_deg = frep.angle_deg;
        rec.frame_mass_change = frep.mass_change;
        if frep.applied {
            let (rot, trans) = compose_frame(state.frame_rotation, state.frame_translation, &frep);
            state.frame_rotation = rot;
            state.frame_translation = trans;
            guess = None;
            state.velocity.clear();
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
            state.step,
            Some(&mech.face_law),
            necks.as_ref(),
            cyc.as_ref().zip(project.cycle.as_ref()).map(|(c, p)| (&c.cycle, p.phases)),
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
        rec.orbit.escaped_runout = rep.escaped_runout;
        rec.orbit.escaped_relief = rep.escaped_relief;
        mass_target -= rep.escaped_runout + rep.escaped_relief;
        rec.rho_s_total = {
            let a = state.body.vertex_areas();
            (0..nv).map(|i| a[i] * state.body.rho_s[i]).sum()
        };
        rec.settle = settled.report;
        rec.thermal.max_surface_damage = state
            .evolving
            .surface_damage
            .as_ref()
            .map_or(0.0, |d| d.iter().copied().fold(0.0, f64::max));
        let moved = n_mobile > 0 || rec.settle.sweeps > 0;
        let d_normal: Vec<f64> = creep_d.iter().zip(&settled.d_normal).map(|(c, s)| c + s).collect();
        // S7 (§11.1): marker maintenance on the settled geometry.
        if let (Some(mk), Some(mp)) = (state.evolving.markers.as_mut(), &project.markers) {
            maintain_markers(project, mp, mk, &state.body, &fwd, &yielded, &d_normal, &mut rec.material)?;
        }

        // 9: advance time.
        let t0 = state.t;
        advance_time(&mut state, spin, rec.advance, t_next, snap_tol);
        spin_step(&law, &mut state, t0, t_rad, omega, &creep_m, &fwd.moments, &mut rec)?;
        if let (Some(l), Some(o)) = (&orbit_law, state.evolving.orbit.as_mut()) {
            let r = &mut rec.orbit;
            l.advance(o, rec.advance, r.escaped_runout + r.escaped_relief + r.escaped_cap, e_exponent)?;
            r.escaped_total = o.escaped;
            r.recoil_bound = 2.0 * o.escaped / o.mass * r.r_hill / r.a;
            state.omega = l.mean_motion(o.a);
        }
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
        deep = origin.map(|origin| Deep { mech, origin });
        records.push(rec);
    }
    let complete = state.t >= t_end - snap_tol;
    Ok(RunResult {
        state,
        records,
        complete,
    })
}

/// Largest vertex displacement since the deep state, as a fraction of the shortest edge,
/// that still reuses it (§10.4.4): the lag error is then of the order of one explicit step.
const LAG_THETA: f64 = 0.1;

/// Where a deep state (Stokes solution, classification, face chain) was evaluated.
#[derive(Clone)]
struct DeepOrigin {
    positions: Vec<DVec3>,
    kernels: Vec<crate::common::density::CauchyKernel>,
    load: Loading,
    /// `θ_lag h_Γ,min` of that geometry.
    budget: f64,
}

impl DeepOrigin {
    fn of(body: &Body, load: Loading) -> Self {
        let p = &body.positions;
        let h_min = body
            .topo
            .faces
            .iter()
            .flat_map(|t| (0..3).map(move |e| (t[e] as usize, t[(e + 1) % 3] as usize)))
            .map(|(a, b)| (p[a] - p[b]).length())
            .fold(f64::INFINITY, f64::min);
        Self { positions: p.clone(), kernels: body.kernels.clone(), load, budget: LAG_THETA * h_min }
    }

    /// The body, its density cores and the loads are still within the lag budget.
    fn within(&self, body: &Body, load: Loading, eps_load: f64) -> bool {
        let near = |a: f64, b: f64| (a - b).abs() <= eps_load * b.abs() || a == b;
        near(load.omega, self.load.omega)
            && near(load.tide, self.load.tide)
            && body.positions.len() == self.positions.len()
            && body.positions.iter().zip(&self.positions).all(|(x, x0)| (*x - *x0).length() <= self.budget)
            && body.kernels.len() == self.kernels.len()
            && body.kernels.iter().zip(&self.kernels).all(|(k, k0)| {
                (k.center - k0.center).length() <= self.budget
                    && (k.weight - k0.weight).abs() <= 1e-3 * k0.weight.abs()
                    && k.metric.iter().zip(&k0.metric).all(|(m, m0)| (m - m0).abs() <= 1e-3 * m0.abs().max(1e-300))
                    && k.alpha == k0.alpha
            })
    }
}

/// The deep state kept for the next step.
struct Deep {
    mech: Mechanics,
    origin: DeepOrigin,
}

/// The masked slip velocity `v_adv = χ_Y (v − a − θ × x)` at arbitrary points (§9.10.2).
/// `χ_Y` is read at the nearest point of the classification grid `X_Y`.
fn advection(mech: &Mechanics, pts: &[DVec3]) -> Vec<DVec3> {
    let cl = &mech.class;
    // Every point of X_Y in the rigid core: χ_Y = 0 everywhere and the result is zero,
    // so the velocity basis is not evaluated.
    if cl.in_core.iter().all(|c| *c) && mech.ratchet.is_empty() {
        return vec![DVec3::ZERO; pts.len()];
    }
    let v = contract_velocity_rays(&mech.surf, &mech.ops.lay, pts, 1, Want::V, &mech.sol.u).v;
    let r = (!mech.ratchet.is_empty()).then(|| contract_velocity_rays(&mech.surf, &mech.ops.lay, pts, 1, Want::V, &mech.ratchet).v);
    pts.iter()
        .enumerate()
        .map(|(k, x)| {
            // A-Λ: the ratchet is real everywhere, so it is not masked (§10.4.7).
            let vr = r.as_ref().map_or(DVec3::ZERO, |r| DVec3::new(r[3 * k], r[3 * k + 1], r[3 * k + 2]));
            if !cl.movable_near(*x) {
                return vr;
            }
            // The total velocity, with the dilatational field of A-Ξ (§11.4).
            DVec3::new(v[3 * k], v[3 * k + 1], v[3 * k + 2]) + mech.sol.v_eps(*x)
                - cl.core_a
                - cl.core_theta.cross(*x)
                + vr
        })
        .collect()
}

/// A-Λ: every vertex of `body` moves by `τ (v_r·N) N`, with `v_r` and `N` at its
/// start-of-step position (§10.4.7): the tangential part only relabels the surface, as for
/// the creep predictor. Whether anything moved.
fn ratchet(body: &mut Body, v: &[DVec3], normal: &[DVec3], tau: f64) -> bool {
    if v.is_empty() {
        return false;
    }
    for ((x, vi), n) in body.positions.iter_mut().zip(v).zip(normal) {
        *x += *n * (vi.dot(*n) * tau);
    }
    true
}

/// Levels 1–2 of §9.10: mass closure of the predicted cores on the new body,
/// then, when the misfit or the closure exceeds its threshold, the refit.
#[allow(clippy::too_many_arguments)]
/// 8c: Levels 1 and 2 of the core transport (§9.10.4–5) on the new body, which carries
/// the predicted cores; `fwd` is bound to it and gets the closed density. Under A-Ξ
/// the closure is the two-scalar fit of §11.4, which may rescale `ρ_b`, and Level 3 may
/// spawn signed cores.
#[allow(clippy::too_many_arguments)]
fn close_cores(
    project: &Project,
    mech: &Mechanics,
    old: &Body,
    body: &mut Body,
    fwd: &mut Forward,
    mass_target: f64,
    tau: f64,
    ratio: Option<&DilatationRatio>,
    rep: &mut CoreReport,
) -> Result<(), SimError> {
    let _span = super::timing::span("core_refit");
    let (bulk, ks, refit) = closed_cores(project, mech, old, body, fwd, mass_target, tau, ratio, rep)?;
    body.bulk = bulk;
    body.kernels = ks;
    fwd.set_density(body.density()?)?;
    if refit {
        // The refit holds the quadrature mass only; close the exact mass again, on the
        // cores, or on ρ_b when there is no core mass to scale.
        let bulk_mass = body.bulk * fwd.moments.volume;
        let core = fwd.moments.mass - bulk_mass;
        if body.kernels.is_empty() || core.abs() < 1e-12 * fwd.moments.mass.abs() {
            body.bulk = (mass_target - core) / fwd.moments.volume;
        } else {
            cores::close_mass(&mut body.kernels, fwd.moments.mass, bulk_mass, mass_target);
        }
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
    ratio: Option<&DilatationRatio>,
    rep: &mut CoreReport,
) -> Result<(f64, Vec<crate::common::density::CauchyKernel>, bool), SimError> {
    use crate::common::density::CauchyDensity;
    let mut ks = new.kernels.clone();
    let mut bulk = new.bulk;
    // Volume quadrature of V⁽ⁿ⁺¹⁾: the ray-segment rule of (G2).
    let s = &project.stokes;
    let (pts, w, _, _) = volume_rule(fwd.body_mesh(), s.n_theta, s.n_phi, s.n_radial, s.cut_depth)?;
    let quad = Quadrature { pts, w };
    let mut target = cores::departure_density(&old.density()?, &quad, tau, &|p| advection(mech, p));
    match ratio {
        Some(r) => {
            // Level 2 target J_ν ρ_K(X_d) and the two-scalar Level 1 (§11.4).
            let j = r.arriving(&quad.pts, tau);
            target.iter_mut().zip(&j).for_each(|(t, j)| *t *= j);
            let core_mass = fwd.moments.mass - bulk * fwd.moments.volume;
            let (sb, sc) = cores::close_two(bulk, &ks, &quad, &target, fwd.moments.volume, core_mass, mass_target);
            bulk *= sb;
            ks.iter_mut().for_each(|k| k.weight *= sc);
            rep.s_bulk = sb;
            rep.s_core = sc;
            rep.xi = sc - 1.0;
        }
        None => {
            rep.xi = cores::close_mass(&mut ks, fwd.moments.mass, bulk * fwd.moments.volume, mass_target);
        }
    }
    rep.e_rho = cores::misfit(&CauchyDensity::try_new(bulk, ks.clone())?, &quad, &target);
    let mut refit = rep.e_rho > project.cores.eps_rho || rep.xi.abs() > project.cores.eps_xi;
    if refit && !ks.is_empty() {
        ks = cores::refit(bulk, &ks, &quad, &target, &project.cores, rep);
        rep.e_rho = cores::misfit(&CauchyDensity::try_new(bulk, ks.clone())?, &quad, &target);
    }
    // Level 3 spawn (optional, A-Ξ): a signed core at the largest residual cluster, kept
    // only when e_ρ drops by the configured factor and ρ_K stays positive at the nodes.
    if let (Some(_), Some(d)) = (ratio, &project.dilatancy) {
        while rep.spawned < d.spawn_max && rep.e_rho > project.cores.eps_rho {
            let current = CauchyDensity::try_new(bulk, ks.clone())?;
            let Some(k) = cores::spawn(&current, &quad, &target, d.spawn_alpha, d.spawn_scale) else { break };
            let mut trial = ks.clone();
            trial.push(k);
            let mut r2 = rep.clone();
            let trial = cores::refit(bulk, &trial, &quad, &target, &project.cores, &mut r2);
            let dens = CauchyDensity::try_new(bulk, trial.clone())?;
            let e = cores::misfit(&dens, &quad, &target);
            if !(e < d.spawn_gain * rep.e_rho) || quad.pts.iter().any(|x| dens.rho(*x) <= 0.0) {
                break;
            }
            ks = trial;
            rep.e_rho = e;
            rep.spawned += 1;
            refit = true;
        }
    }
    Ok((bulk, ks, refit))
}

/// The dilatational density ratio `J_ν` of A-Ξ on one step (Proposition 11.5): `ν` from
/// the markers, the nonlocal driving rate and `ν_c(I_c)` of the classification.
struct DilatationRatio<'a> {
    d: &'a super::evolving::DilatancyParams,
    rec: Reconstruction<'a>,
    nu: Vec<f64>,
    driving: &'a Driving<'a>,
    mech: &'a Mechanics,
}

impl<'a> DilatationRatio<'a> {
    fn new(
        d: &'a super::evolving::DilatancyParams,
        mk: &Markers,
        mp: &super::evolving::MarkerParams,
        mech: &'a Mechanics,
        driving: &'a Driving<'a>,
        r_mean: f64,
    ) -> Self {
        Self {
            d,
            rec: Reconstruction::new(mk, &mech.index, mp.radius * r_mean),
            nu: mk.nu.clone(),
            driving,
            mech,
        }
    }

    /// `J_ν` of the material whose step midpoints are `mid`, over `τ`.
    fn at(&self, mid: &[DVec3], tau: f64) -> Vec<f64> {
        let (fit, _) = self.rec.fit(mid, &[(&self.nu, self.d.range())]);
        let (g, ic) = self.driving.at(mid);
        fit[0]
            .iter()
            .zip(&g)
            .zip(&ic)
            .map(|((f, g), i)| self.d.exact_nu(f.value, self.d.nu_c(*i), tau * g) / f.value)
            .collect()
    }

    /// `J_ν` of the material arriving at `pts` over `τ` (midpoints `x − τ/2 v_adv(x)`).
    fn arriving(&self, pts: &[DVec3], tau: f64) -> Vec<f64> {
        let v = advection(self.mech, pts);
        let mid: Vec<DVec3> = pts.iter().zip(&v).map(|(x, v)| *x - *v * (0.5 * tau)).collect();
        self.at(&mid, tau)
    }
}

/// S7 of §11.1: delete, insert and merge markers on the settled body, and report the
/// state ranges. A new point carries the deposit state when its closest vertex was
/// locked and received a deposit at least as thick as the point's depth.
#[allow(clippy::too_many_arguments)]
fn maintain_markers(
    project: &Project,
    mp: &super::evolving::MarkerParams,
    mk: &mut Markers,
    body: &Body,
    fwd: &Forward,
    yielded: &[bool],
    d_normal: &[f64],
    rep: &mut MaterialReport,
) -> Result<(), SimError> {
    let index = SurfaceIndex::build(&body.positions, &body.topo.faces);
    let s = &project.stokes;
    let (nodes, ..) = volume_rule(fwd.body_mesh(), s.n_theta, s.n_phi, s.n_radial, s.cut_depth)?;
    let deposit_at = |x: DVec3| {
        let c = index.closest(x);
        let v = body.topo.faces[c.face]
            .map(|v| v as usize)
            .into_iter()
            .min_by(|a, b| (body.positions[*a] - x).length().total_cmp(&(body.positions[*b] - x).length()))
            .expect("triangle");
        !yielded[v] && d_normal[v] > 0.0 && c.dist2.sqrt() <= d_normal[v]
    };
    let state = (
        project.dilatancy.as_ref().map_or(0.0, |d| d.deposit),
        project.damage.as_ref().map_or(0.0, |d| d.deposit),
    );
    material::maintain(mk, &index, &nodes, mp.radius * body.r_mean(), mp.n_min, mp.n_max, state, &deposit_at, rep);
    let n = mk.len().max(1) as f64;
    rep.max_damage = mk.damage.iter().copied().fold(0.0, f64::max);
    rep.mean_damage = mk.damage.iter().sum::<f64>() / n;
    rep.nu_min = mk.nu.iter().copied().fold(f64::INFINITY, f64::min);
    rep.nu_max = mk.nu.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if let Some(d) = &project.dilatancy {
        rep.max_xi = mk.nu.iter().map(|n| (d.k * (n - d.nu_c0)).abs()).fold(0.0, f64::max);
    }
    Ok(())
}

/// Step 9+ under A-L: advance `L` over `[t0, state.t]` and read `ω` on the settled inertia.
#[allow(clippy::too_many_arguments)]
fn spin_step(
    law: &Option<SpinLaw>,
    state: &mut State,
    t0: f64,
    t_rad: f64,
    omega_n: f64,
    creep: &super::forward::Moments,
    settled: &super::forward::Moments,
    rec: &mut StepRecord,
) -> Result<(), SimError> {
    if let (Some(l), Some(s)) = (law, state.evolving.spin.as_mut()) {
        let (omega, report) = l.finish(s, t0, state.t, t_rad, omega_n, creep, settled)?;
        state.omega = omega;
        rec.spin = report;
    }
    Ok(())
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
