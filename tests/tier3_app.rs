use std::path::PathBuf;
use std::sync::Arc;

use astroid_evolution::asteroid_app::AsteroidPlugin;
use astroid_evolution::asteroid_app::pipeline::{SurfaceFrame, prepare, run_prepared};
use astroid_evolution::asteroid_app::resources::{
    AppConfig, Evolution, Phase, Scene, display_imported, display_positions, lock_scale,
};
use astroid_evolution::asteroid_app::snapshot::posed_positions;
use astroid_evolution::asteroid_app::ui::hud_lines;
use astroid_evolution::scf_simulator::timing;
use astroid_evolution::common::density::CauchyKernel;
use astroid_evolution::scf_simulator::body::{Body, Topology};
use bevy::prelude::*;

fn project_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets/projects")
}

fn project_text() -> String {
    std::fs::read_to_string(project_dir().join("ryugu_cauchy.toml")).expect("ryugu project")
}

fn config() -> AppConfig {
    AppConfig {
        project_text: project_text(),
        project_dir: project_dir(),
        project_path: "assets/projects/ryugu_cauchy.toml".into(),
        mesh_path: "assets/models/ryugu.glb".into(),
        mesh_unit: "km".into(),
        spin: "const:4.986655e-4@3.15576e13".into(),
        max_steps: Some(1),
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
    let topo = Arc::new(Topology::icosphere(1));
    let count = topo.n_vertices();
    let body = Body::new(
        topo.clone(),
        vec![2.0; count],
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

#[test]
fn the_corner_label_names_the_project_the_mesh_and_the_constant_spin() {
    let waiting = hud_lines(
        &config(),
        &scene(),
        &Evolution {
            phase: Phase::Preparing,
            status: "computing the first tier 2 step".into(),
            ..Evolution::default()
        },
        0.0,
        "combined",
    )
    .join("\n");
    assert!(waiting.contains("project  assets/projects/ryugu_cauchy.toml"));
    assert!(waiting.contains("mesh     assets/models/ryugu.glb  (km)"));
    assert!(waiting.contains("spin     const:4.986655e-4@3.15576e13"));
    assert!(waiting.contains("showing  imported glb"));
    assert!(waiting.contains("file axes"));
    assert!(waiting.contains("1 = 500.0 m"));

    let shown = hud_lines(
        &config(),
        &scene(),
        &Evolution {
            phase: Phase::Running,
            status: "tier 2 running".into(),
            stats: Some(astroid_evolution::asteroid_app::resources::FrameStats {
                step: 3,
                time: 1.0,
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
            }),
            ..Evolution::default()
        },
        0.4,
        "yield",
    )
    .join("\n");
    assert!(shown.contains("showing  tier 2 step 3"));
    assert!(shown.contains("tier 2 frame"));
    assert!(shown.contains("balance  eps_mom 2.000e-3   eps_lam 1.000e-3   picard 6 (4.000e-2)"));
    assert!(shown.contains("sheet    mobile 0.400   tensile 1.000e-2   Ek^-1 2.000e-1   mass residual 1.000e-5"));
    assert!(shown.contains("cfl      bound mode_c   dt 1.000e4 s   dt_cfl 2.000e4 s"));
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
    assert_eq!(stored.project_path, "assets/projects/ryugu_cauchy.toml");
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
    let prepared = prepare(&cfg.project_text, &cfg.project_dir, &cfg.spin).expect("prepare");
    let mut snaps = Vec::new();
    let complete = run_prepared(prepared, Some(1), 1.0, || Ok(()), |snap| {
        snaps.push(snap);
        Ok(())
    })
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
        "tier3 census wall={wall:.3}s complete={complete} kind={} bound={} dt={:.3e} dt_cfl={:.3e} dt_c={:.3e} dt_char={:.3e}",
        snap.kind, snap.dt_bound, audit.dt, audit.dt_cfl, audit.dt_c, audit.dt_char
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
