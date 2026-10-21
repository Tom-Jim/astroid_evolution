//! Snapshot JSON between the worker instance and the page instance.
//!
//! `StepSnapshot` keeps `&'static str` for the step kind. The known strings are
//! the ones Tier 2 already uses. Anything else is leaked once so the viewer
//! type does not change.

use serde::{Deserialize, Serialize};

use crate::asteroid_app::snapshot::{StepAudit, StepSnapshot, StreamPoint, VertexSample};

fn intern(text: &str) -> &'static str {
    match text {
        "geometry" => "geometry",
        "stage-equilibrium" => "stage-equilibrium",
        "spin-only" => "spin-only",
        "cfl" => "cfl",
        "mode_c" => "mode_c",
        "characteristic" => "characteristic",
        "omega" => "omega",
        "dt_max" => "dt_max",
        "t_next" => "t_next",
        "" => "",
        other => Box::leak(other.to_string().into_boxed_str()),
    }
}

#[derive(Serialize, Deserialize)]
struct WireAudit {
    eps_momentum: f64,
    eps_lambda: f64,
    eps_picard: f64,
    picard_iterations: usize,
    tensile_fraction: f64,
    inverse_ekman: f64,
    mass_residual: f64,
    mobile_area_fraction: f64,
    core_mass_fraction: f64,
    slope_slack: f64,
    mass_slack: f64,
    n_mobile: usize,
    n_locked: usize,
    n_deposit: usize,
    socp_iterations: u32,
    trust: f64,
    trust_retries: usize,
    dt: f64,
    dt_cfl: f64,
    dt_c: f64,
    dt_char: f64,
    flux_gap: f64,
    max_locked_step: f64,
    max_deposit_miss: f64,
    rheo_unconverged: u32,
}

#[derive(Serialize, Deserialize)]
struct WireVertex {
    f_delta: f32,
    movable: f32,
    v_sl: [f32; 3],
    rho_s: f32,
    j_net: f32,
    d_radial: f32,
    slack_max: f32,
    slack_avg: f32,
}

#[derive(Serialize, Deserialize)]
struct WireStream {
    verts: [u32; 3],
    bary: [f32; 3],
}

#[derive(Serialize, Deserialize)]
struct WireSnapshot {
    step: usize,
    time: f64,
    omega: f64,
    mass: f64,
    radius_min: f64,
    radius_max: f64,
    yielded_fraction: f64,
    max_slack: f64,
    rho_s_total: f64,
    max_displacement: f64,
    frame_applied: bool,
    wall_seconds: f64,
    flow_active: bool,
    kind: String,
    dt_bound: String,
    audit: WireAudit,
    slack_tol: f32,
    f_delta_p95: f32,
    slack_p95: f32,
    rho_s_p95: f32,
    faces: Vec<[u32; 3]>,
    vertices_m: Vec<[f64; 3]>,
    position_from: Vec<[f32; 3]>,
    position_to: Vec<[f32; 3]>,
    normal_from: Vec<[f32; 3]>,
    normal_to: Vec<[f32; 3]>,
    field: Vec<WireVertex>,
    stream: Vec<WireStream>,
}

fn audit_wire(audit: &StepAudit) -> WireAudit {
    WireAudit {
        eps_momentum: audit.eps_momentum,
        eps_lambda: audit.eps_lambda,
        eps_picard: audit.eps_picard,
        picard_iterations: audit.picard_iterations,
        tensile_fraction: audit.tensile_fraction,
        inverse_ekman: audit.inverse_ekman,
        mass_residual: audit.mass_residual,
        mobile_area_fraction: audit.mobile_area_fraction,
        core_mass_fraction: audit.core_mass_fraction,
        slope_slack: audit.slope_slack,
        mass_slack: audit.mass_slack,
        n_mobile: audit.n_mobile,
        n_locked: audit.n_locked,
        n_deposit: audit.n_deposit,
        socp_iterations: audit.socp_iterations,
        trust: audit.trust,
        trust_retries: audit.trust_retries,
        dt: audit.dt,
        dt_cfl: audit.dt_cfl,
        dt_c: audit.dt_c,
        dt_char: audit.dt_char,
        flux_gap: audit.flux_gap,
        max_locked_step: audit.max_locked_step,
        max_deposit_miss: audit.max_deposit_miss,
        rheo_unconverged: audit.rheo_unconverged,
    }
}

pub fn encode_snapshot(snapshot: &StepSnapshot) -> String {
    let wire = WireSnapshot {
        step: snapshot.step,
        time: snapshot.time,
        omega: snapshot.omega,
        mass: snapshot.mass,
        radius_min: snapshot.radius_min,
        radius_max: snapshot.radius_max,
        yielded_fraction: snapshot.yielded_fraction,
        max_slack: snapshot.max_slack,
        rho_s_total: snapshot.rho_s_total,
        max_displacement: snapshot.max_displacement,
        frame_applied: snapshot.frame_applied,
        wall_seconds: snapshot.wall_seconds,
        flow_active: snapshot.flow_active,
        kind: snapshot.kind.to_string(),
        dt_bound: snapshot.dt_bound.to_string(),
        audit: audit_wire(&snapshot.audit),
        slack_tol: snapshot.slack_tol,
        f_delta_p95: snapshot.f_delta_p95,
        slack_p95: snapshot.slack_p95,
        rho_s_p95: snapshot.rho_s_p95,
        faces: snapshot.faces.clone(),
        vertices_m: snapshot.vertices_m.clone(),
        position_from: snapshot.position_from.clone(),
        position_to: snapshot.position_to.clone(),
        normal_from: snapshot.normal_from.clone(),
        normal_to: snapshot.normal_to.clone(),
        field: snapshot
            .field
            .iter()
            .map(|v| WireVertex {
                f_delta: v.f_delta,
                movable: v.movable,
                v_sl: v.v_sl,
                rho_s: v.rho_s,
                j_net: v.j_net,
                d_radial: v.d_radial,
                slack_max: v.slack_max,
                slack_avg: v.slack_avg,
            })
            .collect(),
        stream: snapshot
            .stream
            .iter()
            .map(|p| WireStream {
                verts: p.verts,
                bary: p.bary,
            })
            .collect(),
    };
    serde_json::to_string(&wire).unwrap_or_else(|err| format!("{{\"error\":\"{err}\"}}"))
}

pub fn decode_snapshot(payload: &str) -> Result<StepSnapshot, String> {
    let wire: WireSnapshot =
        serde_json::from_str(payload).map_err(|err| format!("snapshot: {err}"))?;
    Ok(StepSnapshot {
        step: wire.step,
        time: wire.time,
        omega: wire.omega,
        mass: wire.mass,
        radius_min: wire.radius_min,
        radius_max: wire.radius_max,
        yielded_fraction: wire.yielded_fraction,
        max_slack: wire.max_slack,
        rho_s_total: wire.rho_s_total,
        max_displacement: wire.max_displacement,
        frame_applied: wire.frame_applied,
        wall_seconds: wire.wall_seconds,
        flow_active: wire.flow_active,
        kind: intern(&wire.kind),
        dt_bound: intern(&wire.dt_bound),
        audit: StepAudit {
            eps_momentum: wire.audit.eps_momentum,
            eps_lambda: wire.audit.eps_lambda,
            eps_picard: wire.audit.eps_picard,
            picard_iterations: wire.audit.picard_iterations,
            tensile_fraction: wire.audit.tensile_fraction,
            inverse_ekman: wire.audit.inverse_ekman,
            mass_residual: wire.audit.mass_residual,
            mobile_area_fraction: wire.audit.mobile_area_fraction,
            core_mass_fraction: wire.audit.core_mass_fraction,
            slope_slack: wire.audit.slope_slack,
            mass_slack: wire.audit.mass_slack,
            n_mobile: wire.audit.n_mobile,
            n_locked: wire.audit.n_locked,
            n_deposit: wire.audit.n_deposit,
            socp_iterations: wire.audit.socp_iterations,
            trust: wire.audit.trust,
            trust_retries: wire.audit.trust_retries,
            dt: wire.audit.dt,
            dt_cfl: wire.audit.dt_cfl,
            dt_c: wire.audit.dt_c,
            dt_char: wire.audit.dt_char,
            flux_gap: wire.audit.flux_gap,
            max_locked_step: wire.audit.max_locked_step,
            max_deposit_miss: wire.audit.max_deposit_miss,
            rheo_unconverged: wire.audit.rheo_unconverged,
        },
        slack_tol: wire.slack_tol,
        f_delta_p95: wire.f_delta_p95,
        slack_p95: wire.slack_p95,
        rho_s_p95: wire.rho_s_p95,
        faces: wire.faces,
        vertices_m: wire.vertices_m,
        position_from: wire.position_from,
        position_to: wire.position_to,
        normal_from: wire.normal_from,
        normal_to: wire.normal_to,
        field: wire
            .field
            .into_iter()
            .map(|v| VertexSample {
                f_delta: v.f_delta,
                movable: v.movable,
                v_sl: v.v_sl,
                rho_s: v.rho_s,
                j_net: v.j_net,
                d_radial: v.d_radial,
                slack_max: v.slack_max,
                slack_avg: v.slack_avg,
            })
            .collect(),
        stream: wire
            .stream
            .into_iter()
            .map(|p| StreamPoint {
                verts: p.verts,
                bary: p.bary,
            })
            .collect(),
    })
}
