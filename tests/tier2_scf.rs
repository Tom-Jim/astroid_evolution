//! Tier 2 acceptance: closed project, spin grammar, snapshot round trip, relaxation,
//! frame alignment and a short quasi-static run on the GPU.

use std::path::PathBuf;

use astroid_evolution::rheo_friction::RheoGpu;
use astroid_evolution::scf_simulator::evolve::{self, RunOptions};
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
    let p = project(2);
    let gpu = RheoGpu::new().expect("Metal device");
    let (state, fwd) = evolve::start(&p, initial_body(&p).unwrap()).unwrap();
    let mass0 = fwd.moments.mass;

    // Relaxation never changes the mass beyond eps_mass.
    let (relaxed, rep) = relax(&p, &gpu, &state.body, 2.0e-4).unwrap();
    assert!(rep.slack_after <= rep.slack_before + 1e-12);
    let m1 = astroid_evolution::scf_simulator::forward::Forward::bind(&relaxed, p.g)
        .unwrap()
        .moments
        .mass;
    assert!(((m1 - mass0) / mass0).abs() <= p.geometry.eps_mass * 1.0001);

    // A cohesive body at a modest spin is rigid: the stage ends in one equilibrium step.
    let schedule = SpinSchedule::parse("const:2e-4@1e5").unwrap();
    let mut seen = 0;
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
        &mut |_, _| {
            seen += 1;
            Ok(())
        },
    )
    .unwrap();
    assert!(res.complete);
    assert_eq!(seen, res.records.len());
    let r = &res.records[0];
    assert_eq!(r.kind, "stage-equilibrium");
    assert!(r.eps_momentum < 1e-6);
}
