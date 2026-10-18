//! Tier 2 acceptance: closed project, spin grammar, snapshot round trip, relaxation,
//! frame alignment and a short quasi-static run on the GPU.

use std::path::PathBuf;

use astroid_evolution::rheo_friction::RheoGpu;
use astroid_evolution::scf_simulator::evolve::{self, RunOptions};
use astroid_evolution::scf_simulator::frame;
use astroid_evolution::scf_simulator::io::{Snapshot, initial_body, read_snapshot, write_snapshot};
use astroid_evolution::scf_simulator::params::Project;
use astroid_evolution::scf_simulator::relax::relax;
use astroid_evolution::scf_simulator::spin::SpinSchedule;

fn sample_text(level: usize) -> String {
    let text = include_str!("../assets/projects/sample_project.toml");
    text.replace("level = 3 ", &format!("level = {level} "))
}

fn project(level: usize) -> Project {
    Project::parse(&sample_text(level), PathBuf::from(".")).expect("sample project is closed")
}

#[test]
fn sample_project_is_closed_and_incomplete_ones_are_refused() {
    project(2);
    let broken = sample_text(2).replace("[yielding]\ndelta_y = 0.05\n", "");
    let err = Project::parse(&broken, PathBuf::from(".")).unwrap_err();
    assert_eq!(err.code(), 10);
}

#[test]
fn spin_schedule_grammar() {
    let s = SpinSchedule::parse("stages:100@1e-4,200@2e-4").unwrap();
    s.validate().unwrap();
    assert!(s.is_stages());
    assert_eq!(s.t_end(), 300.0);
    assert_eq!(s.sample(150.0).omega, 2e-4);
    assert_eq!(s.sample(50.0).t_next, 100.0);
    let l = SpinSchedule::parse("linear:1e-4,1e-9,1000").unwrap();
    assert!((l.sample(500.0).omega - (1e-4 + 5e-7)).abs() < 1e-15);
    assert!(SpinSchedule::parse("bogus").is_err());
}

#[test]
fn snapshot_round_trip_and_standard_frame() {
    let p = project(2);
    let (state, _) = evolve::start(&p, initial_body(&p).unwrap()).unwrap();
    let snap = Snapshot::from_body(
        &state.body,
        2,
        0.0,
        0,
        0.0,
        state.frame_rotation,
        state.frame_translation,
        vec![],
    );
    let path = std::env::temp_dir().join("scf_tier2_snapshot.json");
    write_snapshot(&path, &snap).unwrap();
    let back = read_snapshot(&path).unwrap().to_body().unwrap();
    assert_eq!(back.radius, state.body.radius);
    let _ = std::fs::remove_file(path);
}

#[test]
fn relaxation_and_stage_run_on_gpu() {
    let clock = std::time::Instant::now();
    let mark = |label: &str, start: std::time::Instant| {
        eprintln!(
            "time {label}: {:.3}s  (elapsed {:.3}s)",
            start.elapsed().as_secs_f64(),
            clock.elapsed().as_secs_f64()
        );
    };
    let t = std::time::Instant::now();
    // out.obj (metres) with Ryugu's three Cauchy kernels. Centres stay in metres.
    let text = sample_text(2).replace(
        "kernels = []",
        "kernels = [\n  { center = [0.0, 0.0, 0.0], weight = 280.0, sigma = 0.0020, alpha = 1.2 },\n  { center = [0.0, 80.0, 140.0], weight = 160.0, sigma = 0.0080, alpha = 0.55 },\n  { center = [-110.0, 40.0, 20.0], weight = 140.0, sigma = 0.0100, alpha = 2.4 },\n]",
    );
    let p = Project::parse(&text, PathBuf::from(".")).expect("Ryugu density on out.obj");
    mark("parse project", t);
    let t = std::time::Instant::now();
    let gpu = RheoGpu::new().expect("Metal device");
    mark("Metal rheology pipeline", t);
    let t = std::time::Instant::now();
    let body = initial_body(&p).unwrap();
    eprintln!(
        "mesh: {} vertices, {} faces",
        body.topo.n_vertices(),
        body.topo.faces.len()
    );
    mark("load output/out.obj", t);
    let t = std::time::Instant::now();
    let (state, fwd) = evolve::start(&p, body).unwrap();
    mark("start (frame realign + gravity)", t);
    let mass0 = fwd.moments.mass;

    // Relaxation never changes the mass beyond eps_mass.
    let omega = 2.0 * std::f64::consts::PI / (3.8 * 3600.0);
    let t = std::time::Instant::now();
    let (relaxed, rep) = relax(&p, &gpu, &state.body, omega).unwrap();
    mark("relax", t);
    assert!(rep.slack_after <= rep.slack_before + 1e-12);
    let t = std::time::Instant::now();
    let m1 = astroid_evolution::scf_simulator::forward::Forward::bind(&relaxed, p.g)
        .unwrap()
        .moments
        .mass;
    mark("mass check gravity", t);
    assert!(((m1 - mass0) / mass0).abs() <= p.geometry.eps_mass * 1.0001);

    // 3.8 h per revolution. Do not require stage equilibrium: that was the empty-kernel spin.
    let schedule = SpinSchedule::parse(&format!("const:{omega:.8e}@1e5")).unwrap();
    let mut seen = 0;
    let t = std::time::Instant::now();
    let res = evolve::run(
        &p,
        &gpu,
        &schedule,
        state,
        fwd,
        &RunOptions {
            max_steps: Some(5),
            until: None,
        },
        &mut |_, _, _| {
            seen += 1;
            Ok(())
        },
    )
    .unwrap();
    mark("evolve", t);
    eprintln!("time total: {:.3}s", clock.elapsed().as_secs_f64());
    assert_eq!(seen, res.records.len());
    assert!(!res.records.is_empty());
    for r in &res.records {
        eprintln!(
            "record kind={} mass={:.6e} eps_momentum={:.3e}",
            r.kind, r.mass, r.eps_momentum
        );
    }
}

/// Relax is only the pre-evolution shape. Tier 2 then calls RTFP, the rheology
/// chain, and the SCF geometry step. A loose layer that is still supercritical
/// after that partial relax must change radius.
#[test]
fn cauchy_meshes_evolve_past_relax() {
    // 4 h. Fast enough for topographic slopes, short enough that one geometry
    // step is the whole stage when the mobile set is the slope proxy.
    let omega = 2.0 * std::f64::consts::PI / (4.0 * 3600.0);
    let schedule = SpinSchedule::parse(&format!("const:{omega:.8}@1e4")).unwrap();
    let gpu = RheoGpu::new().expect("Metal device");
    let only = std::env::var("SCF_ONLY").ok();
    let mut any_geometry = false;
    for (name, file) in [
        (
            "ryugu",
            include_str!("../assets/projects/ryugu_cauchy.toml"),
        ),
        (
            "phobos",
            include_str!("../assets/projects/phobos_cauchy.toml"),
        ),
        (
            "deimos",
            include_str!("../assets/projects/deimos_cauchy.toml"),
        ),
    ] {
        let p = Project::parse(file, PathBuf::from("assets/projects")).unwrap_or_else(|e| {
            panic!("{name} project: {e}");
        });
        if only.as_deref().is_some_and(|want| want != name) {
            continue;
        }
        let alphas: Vec<f64> = p.kernels().iter().map(|k| k.alpha).collect();
        assert!(
            alphas.iter().any(|a| (*a - 1.0).abs() > 1e-9),
            "{name} alphas {alphas:?} are all 1"
        );
        let t0 = std::time::Instant::now();
        let (mut state, _) = evolve::start(&p, initial_body(&p).unwrap())
            .unwrap_or_else(|e| panic!("{name} start: {e}"));
        let (relaxed, rep) =
            relax(&p, &gpu, &state.body, omega).unwrap_or_else(|e| panic!("{name} relax: {e}"));
        let (body, _, fwd) = frame::realign(&relaxed, None, p.g, &p.geometry, false)
            .unwrap_or_else(|e| panic!("{name} realign: {e}"));
        let radius_in = body.radius.clone();
        state.body = body;
        state.velocity.clear();
        let res = evolve::run(
            &p,
            &gpu,
            &schedule,
            state,
            fwd,
            &RunOptions {
                max_steps: Some(2),
                until: None,
            },
            &mut |_, _, _| Ok(()),
        )
        .unwrap_or_else(|e| panic!("{name} evolve: {e}"));
        let rec = res
            .records
            .first()
            .unwrap_or_else(|| panic!("{name} produced no step"));
        let radius_out = &res.state.body.radius;
        let (mut max_abs, mut max_rel) = (0.0_f64, 0.0_f64);
        for (a, b) in radius_in.iter().zip(radius_out) {
            let d = (a - b).abs();
            max_abs = max_abs.max(d);
            max_rel = max_rel.max(d / a.abs().max(1e-30));
        }
        eprintln!(
            "{name}: {:.2}s mass={:.4e} alphas={alphas:?} relax_sweeps={} slack {:.3e}->{:.3e} kind={} n_mobile={} yielded={:.3e} super={} max_slack={:.3e} dR={:.4e} m ({:.3e} rel) d*={:.4e} socp={} frame={} ({:.3e} deg) picard={} eps_mom={:.3e} eps_lam={:.3e}",
            t0.elapsed().as_secs_f64(),
            rec.mass,
            rep.sweeps.len(),
            rep.slack_before,
            rep.slack_after,
            rec.kind,
            rec.n_mobile,
            rec.yielded_fraction,
            rec.n_supercritical_faces,
            rec.max_slack,
            max_abs,
            max_rel,
            rec.max_displacement,
            rec.socp_iterations,
            rec.frame_applied,
            rec.frame_angle_deg,
            rec.picard_iterations,
            rec.eps_momentum,
            rec.eps_lambda,
        );
        assert!(rec.mass.is_finite() && rec.mass > 0.0, "{name} mass");
        assert!(rec.eps_momentum < p.stokes.tol_momentum, "{name} momentum");
        assert!(rec.rheo_seconds.is_finite(), "{name} rheo");
        if rec.n_supercritical_faces > 0 {
            any_geometry = true;
            assert_eq!(rec.kind, "geometry", "{name} stayed at {}", rec.kind);
            assert!(rec.n_mobile > 0, "{name} supercritical but n_mobile=0");
            assert!(
                rec.max_displacement > 1.0,
                "{name} geometry step did not move radii (max |d*|={})",
                rec.max_displacement
            );
            assert!(
                max_abs > 1.0,
                "{name} relax→final radius change is {max_abs} m"
            );
        }
    }
    assert!(
        any_geometry,
        "none of Ryugu, Phobos, Deimos entered a geometry step"
    );
}
