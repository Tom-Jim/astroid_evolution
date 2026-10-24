use std::path::PathBuf;
use std::sync::Arc;

use astroid_evolution::app::AsteroidPlugin;
use astroid_evolution::app::pipeline::{SurfaceFrame, prepare, run_prepared};
use astroid_evolution::app::resources::{
    AppConfig, Evolution, Phase, Scene, display_imported, display_positions, lock_scale,
};
use astroid_evolution::app::snapshot::posed_positions;
use astroid_evolution::app::bridge::{self, Command, RunRequest};
use astroid_evolution::app::history::History;
use astroid_evolution::app::playback::VisMode;
use astroid_evolution::common::density::CauchyKernel;
use astroid_evolution::scf::body::{Body, Topology};
use astroid_evolution::scf::timing;
use bevy::prelude::*;

fn project_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("dist/.stage/assets/projects")
}

fn project_text() -> String {
    std::fs::read_to_string(project_dir().join("ryugu.toml")).expect("ryugu project")
}

fn config() -> AppConfig {
    AppConfig {
        project_text: project_text(),
        project_dir: project_dir(),
        project_path: "assets/projects/ryugu.toml".into(),
        mesh_path: "assets/models/ryugu.glb".into(),
        mesh_unit: "km".into(),
        spin: "const:4.986655e-4@3.15576e13".into(),
        max_steps: Some(1),
        snapshot: None,
    }
}

#[test]
fn the_view_keeps_file_axes_and_locks_scale_from_the_imported_mesh() {
    let samples = [[3.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 5.0]];
    let scale = lock_scale(&samples);
    assert!((scale - 0.2).abs() < 1.0e-12);
    let shown = display_positions(&[[0.0, 4.0, 5.0]], scale);
    assert!((shown[0][0]).abs() < 1.0e-6);
    assert!((shown[0][1] - 0.8).abs() < 1.0e-6);
    assert!((shown[0][2] - 1.0).abs() < 1.0e-6);
    let placed = display_imported(&[[1.0, 2.0, 3.0]], [1.0, 0.0, 0.0], 0.5);
    assert!((placed[0][0]).abs() < 1.0e-6);
    assert!((placed[0][1] - 1.0).abs() < 1.0e-6);
    assert!((placed[0][2] - 1.5).abs() < 1.0e-6);
}

#[test]
fn a_frame_is_the_tier2_body_itself() {
    let (topo, dirs) = Topology::icosphere(1);
    let topo = Arc::new(topo);
    let count = topo.n_vertices();
    let body = Body::new(
        topo.clone(),
        dirs.iter().map(|d| *d * 2.0).collect(),
        1000.0,
        Vec::<CauchyKernel>::new(),
        vec![1.0; count],
    )
    .unwrap();
    let frame = SurfaceFrame::from_body(&body, 4, 12.0, 3.0e-4, 9.0);
    assert_eq!(frame.vertices.len(), count);
    assert_eq!(frame.faces, topo.faces);
    assert_eq!(
        (frame.step, frame.time, frame.omega, frame.mass),
        (4, 12.0, 3.0e-4, 9.0)
    );
    assert!((frame.radius_min - 2.0).abs() < 1.0e-12);
    assert!((frame.radius_max - 2.0).abs() < 1.0e-12);
    for (shown, position) in frame.vertices.iter().zip(body.positions()) {
        assert!((shown[0] - position.x).abs() < 1.0e-12);
        assert!((shown[1] - position.y).abs() < 1.0e-12);
        assert!((shown[2] - position.z).abs() < 1.0e-12);
    }
}

fn scene() -> Scene {
    Scene {
        vertices: Vec::new(),
        faces: Vec::new(),
        center: [0.0, 0.0, 0.0],
        scale: 0.002,
    }
}

fn running() -> Evolution {
    Evolution {
        phase: Phase::Running,
        status: "tier 2 running".into(),
        stats: Some(astroid_evolution::app::resources::FrameStats {
            step: 3,
            time: 3.15576e13,
            omega: 4.986655e-4,
            mass: 1.0,
            radius_min: 400.0,
            radius_max: 500.0,
            yielded_fraction: 0.25,
            max_slack: 1.0e-3,
            rho_s_total: 10.0,
            max_displacement: 0.4,
            eps_momentum: 2.0e-3,
            eps_lambda: 1.0e-3,
            eps_picard: 4.0e-2,
            picard_iterations: 6,
            tensile_fraction: 0.01,
            inverse_ekman: 0.2,
            mass_residual: 1.0e-5,
            mobile_area_fraction: 0.4,
            dt: 1.0e4,
            dt_cfl: 2.0e4,
            dt_bound: "mode_c",
            tension_max: 12.5,
            runout_traced: 40,
            runout_distance: 35.0,
            runout_dlat: -0.05,
            runout_hops: 2,
        }),
        ..Evolution::default()
    }
}

#[test]
fn the_status_names_the_run_and_reads_back_the_slider_keys() {
    let waiting = bridge::status(
        &config(),
        &scene(),
        &Evolution {
            phase: Phase::Preparing,
            status: "computing the first tier 2 step".into(),
            ..Evolution::default()
        },
        &History::default(),
        &VisMode::default(),
        [0.0, 1.0],
        false,
    );
    let json = serde_json::to_value(&waiting).unwrap();
    assert_eq!(json["phase"], "preparing");
    assert_eq!(json["project"], "assets/projects/ryugu.toml");
    assert_eq!(json["mesh"], "assets/models/ryugu.glb");
    assert_eq!(json["mesh_unit"], "km");
    assert_eq!(json["spin"], "const:4.986655e-4@3.15576e13");
    assert!(json["stats"].is_null());
    assert_eq!(json["view_m"], 500.0);
    assert_eq!(json["layer"], 4);
    for key in bridge::PARAMS {
        assert!(json["params"][key].is_number(), "{key}");
    }
    assert_eq!(json["params"]["material.cohesion_surface"], 0.5);
    assert_eq!(json["params"]["body.level"], 5.0);

    let shown = serde_json::to_value(bridge::status(
        &config(),
        &scene(),
        &running(),
        &History::default(),
        &VisMode::default(),
        [0.0, 1.0],
        true,
    ))
    .unwrap();
    assert_eq!(shown["phase"], "running");
    assert_eq!(shown["stats"]["step"], 3);
    assert_eq!(shown["stats"]["dt_bound"], "mode_c");
    assert_eq!(shown["stats"]["picard_iterations"], 6);
    assert!((shown["time_myr"].as_f64().unwrap() - 1.0).abs() < 1e-12);
    assert!((shown["period_h"].as_f64().unwrap() - 3.5).abs() < 1e-3);
    assert!(shown["orbit"].is_null());
    assert_eq!(shown["wireframe"], true);
}

#[test]
fn panel_commands_parse_and_overrides_rewrite_the_project() {
    let restart: Command = serde_json::from_str(
        r#"{"cmd":"restart","project":"assets/projects/phobos.toml","spin":"const:2e-4",
            "overrides":{"material.cohesion":10,"body.level":4.4},"snapshot":null}"#,
    )
    .unwrap();
    let Command::Restart(RunRequest {
        project, overrides, ..
    }) = restart
    else {
        panic!("not a restart");
    };
    assert_eq!(project, "assets/projects/phobos.toml");
    assert!(bridge::send(r#"{"cmd":"layer","mode":2}"#).is_ok());
    assert!(bridge::send(r#"{"cmd":"pointer","x":1,"y":2,"button":2,"down":true}"#).is_ok());
    assert!(bridge::send(r#"{"cmd":"warp"}"#).is_err());

    let mut edited = config();
    edited.apply_overrides(&overrides).unwrap();
    let table: toml::Table = edited.project_text.parse().unwrap();
    assert_eq!(table["material"]["cohesion"].as_float(), Some(10.0));
    assert_eq!(table["body"]["level"].as_integer(), Some(4));
    let mut bad = config();
    let unknown = [("material.nothing".to_string(), 1.0)].into_iter().collect();
    assert!(bad.apply_overrides(&unknown).is_err());
}

#[test]
fn a_headless_viewer_records_the_input_and_does_not_start_tier2() {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins).add_plugins(AsteroidPlugin {
        config: config(),
        windowed: false,
    });
    app.update();
    let scene = app.world().resource::<Scene>();
    assert_eq!(scene.vertices.len(), 98306);
    assert_eq!(scene.faces.len(), 196608);
    assert!(
        scene.scale > 0.001 && scene.scale < 0.003,
        "{}",
        scene.scale
    );
    let evolution = app.world().resource::<Evolution>();
    assert_eq!(evolution.phase, Phase::Idle);
    assert!(evolution.receiver.lock().unwrap().is_none());
    assert!(evolution.stats.is_none());
    let stored = app.world().resource::<AppConfig>();
    assert_eq!(stored.project_path, "assets/projects/ryugu.toml");
    assert_eq!(stored.mesh_path, "assets/models/ryugu.glb");
    assert_eq!(stored.mesh_unit, "km");
    assert!(stored.spin.starts_with("const:"));
}

/// One geological step on the same path the window uses (`prepare` + `run_prepared`).
///
/// Hard checks are the exogenous spin, criterion (b), mode C balance, the locked
/// and deposition sets, and the displayed endpoints. Slack, yield, and the mass
/// residual are printed. They are not convergence tests (§2.15).
#[test]
fn ryugu_step_matches_the_quasi_static_audit() {
    timing::ensure_enabled();
    let cfg = config();
    let started = std::time::Instant::now();
    let prepared = prepare(&cfg.project_text, &cfg.project_dir, &cfg.spin, None).expect("prepare");
    let mut snaps = Vec::new();
    let complete = run_prepared(
        prepared,
        Some(1),
        1.0,
        || Ok(()),
        |snap| {
            snaps.push(snap);
            Ok(())
        },
    )
    .expect("tier 2 step");
    let wall = started.elapsed().as_secs_f64();
    assert_eq!(snaps.len(), 1, "the step must emit one snapshot");
    let snap = &snaps[0];
    let omega = 4.986655e-4;
    assert!(
        (snap.omega - omega).abs() <= 1.0e-12 * omega,
        "spin {} left the exogenous value {omega}",
        snap.omega
    );
    let tol = 1.0e-2;
    assert!(
        snap.audit.eps_momentum <= tol,
        "eps_mom {} exceeds {tol}",
        snap.audit.eps_momentum
    );
    assert!(
        snap.audit.eps_lambda <= tol,
        "eps_lam {} exceeds {tol}",
        snap.audit.eps_lambda
    );
    assert_eq!(
        snap.audit.rheo_unconverged, 0,
        "a passing step reported rheology not converged"
    );
    assert!(
        snap.audit.flux_gap <= 1.0e-6,
        "mode C balance {}",
        snap.audit.flux_gap
    );
    assert!(
        snap.audit.max_locked_step <= 1.0e-6,
        "locked node moved by {} m",
        snap.audit.max_locked_step
    );
    assert!(
        snap.audit.max_deposit_miss <= 1.0e-4,
        "deposition left d_pred by {} m",
        snap.audit.max_deposit_miss
    );
    assert_eq!(
        snap.audit.trust_retries, 0,
        "the inner cone must not shrink the trust region"
    );
    assert_eq!(
        posed_positions(&snap.position_from, &snap.position_to, 0.0),
        snap.position_from
    );
    assert_eq!(
        posed_positions(&snap.position_from, &snap.position_to, 1.0),
        snap.position_to
    );
    for (shown, vertex) in snap.position_to.iter().zip(&snap.vertices_m) {
        for k in 0..3 {
            assert!(
                (f64::from(shown[k]) - vertex[k]).abs() < 1.0e-2,
                "displayed endpoint {shown:?} left the body vertex {vertex:?}"
            );
        }
    }

    let audit = &snap.audit;
    eprintln!(
        "tier3 census wall={wall:.3}s complete={complete} bound={} dt={:.3e} dt_cfl={:.3e} dt_c={:.3e} dt_char={:.3e}",
        snap.dt_bound.0, audit.dt, audit.dt_cfl, audit.dt_c, audit.dt_char
    );
    eprintln!(
        "tier3 census omega={:.6e} mass={:.6e} residual={:.3e} yielded={:.3e} mobile_area={:.3e} core_mass={:.3e} n_mobile={} n_locked={} n_deposit={}",
        snap.omega,
        snap.mass,
        audit.mass_residual,
        snap.yielded_fraction,
        audit.mobile_area_fraction,
        audit.core_mass_fraction,
        audit.n_mobile,
        audit.n_locked,
        audit.n_deposit
    );
    eprintln!(
        "tier3 census eps_mom={:.3e} eps_lam={:.3e} eps_pic={:.3e} picard={} tensile={:.3e} Ek^-1={:.3e} flux={:.3e}",
        audit.eps_momentum,
        audit.eps_lambda,
        audit.eps_picard,
        audit.picard_iterations,
        audit.tensile_fraction,
        audit.inverse_ekman,
        audit.flux_gap
    );
    eprintln!(
        "tier3 census slack={:.3e} slope_slack={:.3e} mass_slack={:.3e} socp={} trust={:.3e} locked_step={:.3e} deposit_miss={:.3e} |dR|={:.3e}",
        snap.max_slack,
        audit.slope_slack,
        audit.mass_slack,
        audit.socp_iterations,
        audit.trust,
        audit.max_locked_step,
        audit.max_deposit_miss,
        snap.max_displacement
    );
    let phases = timing::drain();
    let spanned: f64 = phases.iter().map(|row| row.seconds).sum();
    eprintln!("tier3 census spans={spanned:.3}s");
    for row in &phases {
        eprintln!(
            "  {:20} {:8.3}s  n={:3}  avg={:7.3}s",
            row.name,
            row.seconds,
            row.calls,
            row.seconds / row.calls as f64
        );
    }
}
