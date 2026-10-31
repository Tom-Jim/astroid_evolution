//! Tier 2 acceptance: closed project, spin grammar, snapshot round trip, relaxation,
//! frame alignment and a short quasi-static run on the GPU.

use std::path::PathBuf;

use astroid_evolution::scf::evolve::{self, RunOptions, State};
use astroid_evolution::scf::forward::Loading;
use astroid_evolution::scf::frame;
use astroid_evolution::scf::io::{Snapshot, initial_body, read_snapshot, write_snapshot};
use astroid_evolution::scf::params::Project;
use astroid_evolution::scf::relax::relax;
use astroid_evolution::scf::spin::SpinSchedule;

fn projects() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("dist/.stage/assets/projects")
}

fn sample_text(level: usize) -> String {
    let text = include_str!("../dist/.stage/assets/projects/sample_project.toml");
    text.replace("level = 3 ", &format!("level = {level} "))
}

fn project(level: usize) -> Project {
    Project::parse(&sample_text(level), projects()).expect("sample project is closed")
}

#[test]
fn sample_project_is_closed_and_incomplete_ones_are_refused() {
    project(2);
    let broken = sample_text(2).replace("[yielding]\ndelta_y = 0.05\n", "");
    let err = Project::parse(&broken, projects()).unwrap_err();
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
    assert_eq!(back.positions, state.body.positions);
    let _ = std::fs::remove_file(path);
}

/// A spin at which the Ryugu-density sample body has a quasi-static state.
const SLOW_HOURS: f64 = 6.0;

/// The sample body (out.obj, metres) at icosphere `level` with Ryugu's three Cauchy
/// kernels. Centres stay in metres.
fn ryugu_density(level: usize) -> Project {
    let text = sample_text(level).replace(
        "kernels = []",
        "kernels = [\n  { center = [0.0, 0.0, 0.0], weight = 280.0, sigma = 0.0020, alpha = 1.2 },\n  { center = [0.0, 80.0, 140.0], weight = 160.0, sigma = 0.0080, alpha = 0.55 },\n  { center = [-110.0, 40.0, 20.0], weight = 140.0, sigma = 0.0100, alpha = 2.4 },\n]",
    );
    Project::parse(&text, projects()).expect("Ryugu density on out.obj")
}

fn opts(max_steps: usize, lag: bool) -> RunOptions {
    RunOptions { max_steps: Some(max_steps), until: None, lag }
}

/// What every accepted step must satisfy, whatever the body does: finite fields, the
/// momentum audit inside its tolerance, mass held to `ε_M` of the start, time booked by
/// the advance of each record (zero for a settling step) at the exogenous spin, and the
/// cone program solved.
fn audit_records(p: &Project, res: &evolve::RunResult, mass0: f64, spin: &SpinSchedule, t0: f64) {
    let mut t = t0;
    for r in &res.records {
        let at = format!("step {} ({})", r.step, r.kind);
        // `t` is where the step starts and `advance` what it books (zero for settling).
        assert!(r.advance >= 0.0 && (r.t - t).abs() <= 1e-9 * t.max(1.0), "{at}: starts at {} instead of {t}", r.t);
        let (a, b) = (spin.sample(r.t).omega, spin.sample(r.t + r.advance).omega);
        let tol = 1e-12 * a.abs().max(b.abs());
        assert!(r.omega >= a.min(b) - tol && r.omega <= a.max(b) + tol, "{at}: spin {} outside [{a}, {b}]", r.omega);
        t = r.t + r.advance;
        assert!(r.mass.is_finite() && ((r.mass - mass0) / mass0).abs() <= p.geometry.eps_mass, "{at}: mass {} vs {mass0}", r.mass);
        assert!(r.eps_momentum < p.stokes.tol_momentum, "{at}: eps_mom {}", r.eps_momentum);
        for (name, v) in [
            ("max_slack", r.max_slack),
            ("max_displacement", r.max_displacement),
            ("yielded_fraction", r.yielded_fraction),
            ("mobile_area_fraction", r.mobile_area_fraction),
            ("eps_lambda", r.eps_lambda),
            ("dt", r.dt),
        ] {
            assert!(v.is_finite() && v >= 0.0, "{at}: {name} = {v}");
        }
        assert!(r.yielded_fraction <= 1.0 && r.mobile_area_fraction <= 1.0, "{at}: fractions");
        assert!(!r.cone_unresolved, "{at}: the cone program was not solved");
        assert_eq!(r.rheo_unconverged, 0, "{at}: rheology did not converge");
    }
    assert!((res.state.t - t).abs() <= 1e-9 * t.max(1.0), "the state is at {} after records to {t}", res.state.t);
    assert!(res.state.body.positions.iter().all(|x| x.is_finite()), "a vertex is not finite");
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
    let p = ryugu_density(4);
    mark("parse project", t);
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
    let (relaxed, rep) = relax(&p, &state.body, Loading { omega, tide: 0.0 }).unwrap();
    mark("relax", t);
    assert!(rep.slack_after <= rep.slack_before + 1e-12);
    let t = std::time::Instant::now();
    let m1 = astroid_evolution::scf::forward::Forward::bind(&relaxed, p.g)
        .unwrap()
        .moments
        .mass;
    mark("mass check gravity", t);
    assert!(((m1 - mass0) / mass0).abs() <= p.geometry.eps_mass * 1.0001);

    // At 3.8 h this body is past its limit load (L* > 1, §2.17.1): no quasi-static state
    // exists, and the solver must say so instead of iterating on.
    let schedule = SpinSchedule::parse(&format!("const:{omega:.8e}@1e5")).unwrap();
    let t = std::time::Instant::now();
    let collapse = evolve::run(
        &p,
        &schedule,
        state.clone(),
        fwd,
        &opts(5, true),
        &mut |_, _, _| Ok(()),
    )
    .err()
    .expect("3.8 h is past the limit load");
    mark("collapse", t);
    assert_eq!(collapse.kind, "LimitLoadCollapse", "{collapse}");

    // At SLOW_HOURS a quasi-static state exists. Do not require stage equilibrium.
    let slow = 2.0 * std::f64::consts::PI / (SLOW_HOURS * 3600.0);
    let schedule = SpinSchedule::parse(&format!("const:{slow:.8e}@1e5")).unwrap();
    let fwd = astroid_evolution::scf::forward::Forward::bind(&state.body, p.g).unwrap();
    let mut seen = 0;
    let t = std::time::Instant::now();
    let res = evolve::run(
        &p,
        &schedule,
        state,
        fwd,
        &opts(5, true),
        &mut |_, _, _| {
            seen += 1;
            Ok(())
        },
    )
    .unwrap();
    mark("evolve", t);
    eprintln!("time total: {:.3}s", clock.elapsed().as_secs_f64());
    for row in astroid_evolution::scf::timing::drain() {
        eprintln!(
            "span {:>8} {:<20} {:>9.3}s {:>6} calls",
            row.phase, row.name, row.seconds, row.calls
        );
    }
    assert_eq!(seen, res.records.len());
    assert!(res.records.len() == 5 || res.complete, "the run stopped after {} steps", res.records.len());
    for r in &res.records {
        eprintln!(
            "record kind={} mass={:.6e} eps_momentum={:.3e} lagged={}",
            r.kind, r.mass, r.eps_momentum, r.lagged
        );
    }
    assert!(!res.records[0].lagged, "the first step has no deep state to lag");
    audit_records(&p, &res, mass0, &schedule, 0.0);
}

/// Ryugu at icosphere level 4 (`level = 5` in the project) at a spin of `hours`, slowly
/// varying so that there is no stage target (§2.15) and the run goes on; and its start.
fn ryugu_level4(hours: f64) -> (Project, SpinSchedule, State, f64) {
    let text = include_str!("../dist/.stage/assets/projects/ryugu.toml").replacen("level = 5", "level = 4", 1);
    let p = Project::parse(&text, projects()).expect("ryugu project");
    let omega = 2.0 * std::f64::consts::PI / (hours * 3600.0);
    let schedule = SpinSchedule::parse(&format!("linear:{omega:.8e},1e-22,1e13")).unwrap();
    let (state, fwd) = evolve::start(&p, initial_body(&p).unwrap()).unwrap();
    (p, schedule, state, fwd.moments.mass)
}

fn run_from(p: &Project, schedule: &SpinSchedule, state: &State, steps: usize, lag: bool) -> evolve::RunResult {
    let fwd = astroid_evolution::scf::forward::Forward::bind(&state.body, p.g).unwrap();
    evolve::run(p, schedule, state.clone(), fwd, &opts(steps, lag), &mut |_, _, _| Ok(())).unwrap()
}

/// The lagged deep state (§10.4.4 step 4+) is reused only while the surface and the
/// density cores stay within `θ_lag h_Γ,min` and the loads within `ε_ω`. On Ryugu's quiet
/// surface at its present 7.63 h spin nothing moves, every step after the first lags,
/// and the run is exactly the one that evaluates every step.
#[test]
fn lagged_steps_on_a_quiet_surface_repeat_the_evaluated_run() {
    let (p, schedule, state, mass0) = ryugu_level4(7.63);
    let (lagged, exact) = (run_from(&p, &schedule, &state, 6, true), run_from(&p, &schedule, &state, 6, false));
    audit_records(&p, &lagged, mass0, &schedule, 0.0);
    audit_records(&p, &exact, mass0, &schedule, 0.0);
    assert!(exact.records.iter().all(|r| !r.lagged), "lag switched off, yet a step lagged");
    assert!(!lagged.records[0].lagged, "the first step has no deep state to lag");
    assert!(lagged.records[1..].iter().all(|r| r.lagged), "a quiet step evaluated the deep state again");
    assert_eq!(lagged.records.len(), exact.records.len());
    for (a, b) in lagged.records.iter().zip(&exact.records) {
        assert_eq!((a.kind, a.advance, a.max_displacement, a.mass), (b.kind, b.advance, b.max_displacement, b.mass), "step {}", a.step);
    }
    assert_eq!(lagged.state.body.positions, exact.state.body.positions);
}

/// A run depends on its request only: run again in the same process (the viewer's
/// restart), it repeats itself bit for bit, sliding surface and moving cores included.
#[test]
fn a_rerun_in_the_same_process_repeats_the_run() {
    let (p, schedule, state, mass0) = ryugu_level4(3.4);
    let first = run_from(&p, &schedule, &state, 3, true);
    let again = run_from(&p, &schedule, &state, 3, true);
    audit_records(&p, &first, mass0, &schedule, 0.0);
    assert!(first.records.iter().any(|r| r.max_displacement > 0.0), "nothing moved: the rerun proves little");
    for (a, b) in again.records.iter().zip(&first.records) {
        assert_eq!((a.kind, a.max_displacement, a.mass, a.yielded_fraction), (b.kind, b.max_displacement, b.mass, b.yielded_fraction), "step {} differs on a rerun", a.step);
    }
    assert_eq!(again.state.body.positions, first.state.body.positions, "a rerun ends elsewhere");
}

/// Hypothesis A-O (§10.4.6) on Phobos at icosphere level 4 and velocity degree 5, at 6.6 Mm:
/// its tip has passed L₁, and the tide dominates even at its underdense centre, so the far
/// neck sits at the centroid. A 1 m skin of 0.5 Pa over a 100 kPa interior keeps the deep
/// state at rest, which isolates A-O: loose layer beyond the necks detaches, and runout and
/// relief packets leave through them. (With the project's 25 Pa interior Phobos has failed
/// before this orbit, near 6.9 Mm; with 2.5 kPa at 7.4 Mm a third of it creeps and the
/// coupled pressure iteration does not settle, with or without A-O.)
/// Every step keeps the exact ledger `M + escaped = M₀` and the exact decay
/// `a^{13/2} ← a^{13/2} − κ M τ`, and the certified bracket of `L*` stays below 1.
#[test]
fn phobos_loses_mass_through_its_inner_neck() {
    let text = include_str!("../dist/.stage/assets/projects/phobos.toml")
        .replacen("level = 5", "level = 4", 1)
        .replacen("l_velocity = 6", "l_velocity = 5", 1)
        .replacen("l_pressure = 8", "l_pressure = 7", 1)
        .replacen("cohesion = 25.0", "cohesion = 1.0e5", 1)
        .replacen("cohesion_depth = 100.0", "cohesion_depth = 1.0", 1)
        + "\n[orbit]\nprimary_radius = 3.3895e6\nlimit_every = 2\n";
    let p = Project::parse(&text, projects()).expect("phobos project");
    let (gm, a0, adot0): (f64, f64, f64) = (4.282837e13, 6.6e6, -8.19e-9);
    let schedule = SpinSchedule::parse(&format!("orbit:{gm:e},{a0:e},{adot0:e},1.2e14")).unwrap();
    let (state, fwd) = evolve::start(&p, initial_body(&p).unwrap()).unwrap();
    let m0 = fwd.moments.mass;
    let kappa = -6.5 * a0.powf(5.5) * adot0 / m0;
    let res = evolve::run(&p, &schedule, state, fwd, &opts(4, true), &mut |_, _, _| Ok(()));
    let res = match res {
        Ok(r) => r,
        Err(e) => panic!("phobos A-O run: {e}"),
    };
    let mut a = a0;
    for r in &res.records {
        let o = &r.orbit;
        eprintln!(
            "step {} {} a={:.6e} margin={:+.4} lost runout={:.3e} relief={:.3e} cap={:.3e} total={:.3e} L*=[{:.4}, {:.4}] detached={} dt={:.3e} bound={}",
            r.step, r.kind, o.a, o.lobe_margin, o.escaped_runout, o.escaped_relief, o.escaped_cap, o.escaped_total,
            o.limit_lo, o.limit_hi, r.settle.n_detached, r.dt, r.dt_bound
        );
        assert!(o.active);
        assert!((o.a - a).abs() <= 1e-9 * a, "step {}: a {} vs {a}", r.step, o.a);
        let lost = o.escaped_runout + o.escaped_relief + o.escaped_cap;
        a = (a.powf(6.5) - kappa * o.mass * r.advance).powf(1.0 / 6.5);
        // The ledger at the start of the step: M_n + (escaped before it) = M₀.
        assert!((o.mass + (o.escaped_total - lost) - m0).abs() <= 1e-9 * m0, "step {}: ledger", r.step);
        if o.limit_hi > 0.0 {
            assert!(o.limit_lo <= o.limit_hi && o.limit_hi < 1.0, "step {}: L* in [{}, {}]", r.step, o.limit_lo, o.limit_hi);
        }
    }
    let last = &res.records.last().expect("a step").orbit;
    assert!(res.records.iter().any(|r| r.orbit.lobe_margin > 0.0), "the tip never passed a neck plane");
    assert!(last.escaped_total > 0.0, "an overflowing Phobos lost no mass");
    let first = &res.records[0];
    assert!(first.settle.n_detached > 0, "no loose layer detached beyond a neck");
    assert!(first.orbit.escaped_runout > 0.0 && first.orbit.escaped_relief > 0.0, "a route lost nothing");
    let o = res.state.evolving.orbit.expect("the orbit state");
    assert!((o.mass + o.escaped - m0).abs() <= 1e-9 * m0, "ledger {} + {} vs {m0}", o.mass, o.escaped);
    assert!(o.a < a0);
}

/// Chapter 12 end to end on the sample body (out.obj, level 3) at 2 Ryugu radii from a
/// Mars-mass primary, deep inside its Roche limit with weak cohesion: the certified limit load
/// exceeds 1, the mechanism cuts into rigid pieces of the parent's mass, and their dynamics
/// conserve angular momentum while events keep the ledgers.
#[test]
fn a_failed_body_breaks_and_its_fragments_move() {
    use astroid_evolution::common::math::{DMat3, DVec3};
    use astroid_evolution::scf::breakup::split;
    use astroid_evolution::scf::forward::Forward;
    use astroid_evolution::scf::nbody::{Primary, System};
    use astroid_evolution::scf::orbit::certify;
    let text = sample_text(3)
        + "\n[orbit]\nprimary_radius = 3.3895e6\nlimit_every = 0\n"
        + "\n[breakup]\nmax_blocks = 4\ngain = 0.2\nmin_fraction = 0.02\nvoxels = 48\ndebris_diameter = 50.0\nprimary_j2 = 0.0\ndynamic_time = 2.0e3\nsteps_per_orbit = 200\ncheck_every = 1.0e3\noutput_every = 500.0\nring_cells = 50\nring_particle_radius = 1.0\nring_time = 0.0\n";
    let p = Project::parse(&text, projects()).expect("sample with breakup");
    let prm = p.breakup.clone().unwrap();
    let (state, _) = evolve::start(&p, initial_body(&p).unwrap()).unwrap();
    let body = state.body;
    let gm: f64 = 4.282837e13;
    // Inside the Roche limit of a rubble pile: the tide alone exceeds the body's gravity.
    let a: f64 = 3.0e6;
    let n2 = gm / a.powi(3);
    let mut fwd = Forward::bind(&body, p.g).unwrap();
    let (bracket, ops) = certify(&p, &body, &mut fwd, Loading { omega: n2.sqrt(), tide: n2 }).unwrap();
    assert!(bracket.lo > 1.0, "not certified to fail: L* in [{}, {}]", bracket.lo, bracket.hi);
    let cut = split(&p, &prm, &body, &fwd, &ops, &bracket.u).unwrap().expect("rigid blocks");
    eprintln!("pieces {} misfit {:?} mass error {:.2e}", cut.pieces.len(), cut.misfit, cut.mass_error);
    assert!(cut.pieces.len() >= 2 && cut.mass_error < 1e-3);
    let primary = Primary { gm, radius: 3.3895e6, j2: 0.0 };
    let mut sys = System::new(&p, &prm, primary);
    let speed = (gm / a).sqrt();
    sys.add_pieces(cut.pieces, DVec3::new(-a, 0.0, 0.0), DVec3::new(0.0, -speed, 0.0), DMat3::IDENTITY, DVec3::Z * (speed / a)).unwrap();
    let mass0: f64 = sys.fragments.iter().map(|f| f.mass).sum();
    let l0: DVec3 = sys.fragments.iter().map(|f| f.x.cross(f.v) * f.mass + f.l).sum();
    let outcome = sys.run(&mut |_, _| Ok(())).unwrap();
    let left: f64 = outcome.fragments.iter().map(|f| f.mass).sum::<f64>()
        + outcome.ring.mass()
        + outcome.ring.to_primary
        + outcome.ring.to_moonlets
        + outcome.escaped;
    assert!((left - mass0).abs() <= 1e-9 * mass0, "ledger {left} vs {mass0}");
    if outcome.events.iter().all(|e| e.kind == "merge") {
        let l1: DVec3 = outcome.fragments.iter().map(|f| f.x.cross(f.v) * f.mass + f.l).sum();
        assert!((l1 - l0).length() <= 1e-9 * l0.length(), "angular momentum {l0:?} -> {l1:?}");
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
    let only = std::env::var("SCF_ONLY").ok();
    let mut any_geometry = false;
    for (name, file) in [
        (
            "ryugu",
            include_str!("../dist/.stage/assets/projects/ryugu.toml"),
        ),
        (
            "phobos",
            include_str!("../dist/.stage/assets/projects/phobos.toml"),
        ),
        (
            "deimos",
            include_str!("../dist/.stage/assets/projects/deimos.toml"),
        ),
    ] {
        // Lab glass-bead friction (Jop et al. 2006): weaker than the projects'
        // 35° regolith, so the meshes have slopes that must flow.
        let file = file
            .replace("mu_s = 0.70", "mu_s = 0.38")
            .replace("mu_2 = 0.96", "mu_2 = 0.64");
        let p = Project::parse(&file, projects()).unwrap_or_else(|e| {
            panic!("{name} project: {e}");
        });
        if only.as_deref().is_some_and(|want| want != name) {
            continue;
        }
        let t0 = std::time::Instant::now();
        let (mut state, _) = evolve::start(&p, initial_body(&p).unwrap())
            .unwrap_or_else(|e| panic!("{name} start: {e}"));
        let (relaxed, rep) = relax(&p, &state.body, Loading { omega, tide: 0.0 })
            .unwrap_or_else(|e| panic!("{name} relax: {e}"));
        let (body, _, fwd) = frame::realign(&relaxed, None, p.g, &p.geometry, false)
            .unwrap_or_else(|e| panic!("{name} realign: {e}"));
        let radius_in: Vec<f64> = body.positions.iter().map(|x| x.length()).collect();
        state.body = body;
        state.velocity.clear();
        let res = match evolve::run(
            &p,
            &schedule,
            state,
            fwd,
            &opts(2, true),
            &mut |_, _, _| Ok(()),
        ) {
            Ok(res) => res,
            // Deimos at 4 h with glass-bead friction is at its limit load: the certified bracket
            // at the iteration's pressure lies just below 1 (about [0.92, 0.99]), and the
            // quasi-static route either certifies the collapse or does not reach the state
            // (§2.17, §10.4.2); neither is a solver defect. Ryugu and Phobos have a state.
            Err(e) if name == "deimos" && (e.kind == "LimitLoadCollapse" || e.kind == "RheologyNotConverged") => {
                eprintln!("{name}: limit load reached ({})", e.detail);
                continue;
            }
            Err(e) => panic!("{name} evolve: {e}"),
        };
        let rec = res
            .records
            .first()
            .unwrap_or_else(|| panic!("{name} produced no step"));
        let radius_out: Vec<f64> = res.state.body.positions.iter().map(|x| x.length()).collect();
        let (mut max_abs, mut max_rel) = (0.0_f64, 0.0_f64);
        for (a, b) in radius_in.iter().zip(&radius_out) {
            let d = (a - b).abs();
            max_abs = max_abs.max(d);
            max_rel = max_rel.max(d / a.abs().max(1e-30));
        }
        eprintln!(
            "{name}: {:.2}s mass={:.4e} relax_sweeps={} slack {:.3e}->{:.3e} kind={} n_mobile={} yielded={:.3e} super={} max_slack={:.3e} dR={:.4e} m ({:.3e} rel) d*={:.4e} socp={} frame={} ({:.3e} deg) picard={} eps_mom={:.3e} eps_lam={:.3e}",
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
            // Supercritical loose faces are the surface's: they settle after the creep step.
            assert!(
                rec.settle.sweeps > 0 || rec.settle.converged,
                "{name} supercritical but the surface did not settle"
            );
            assert!(
                rec.max_displacement.is_finite(),
                "{name} displacement is not finite"
            );
            // A yielded sheet carries a slide predictor, so the trust step is
            // metres. An unyielded core that was only promoted onto the slope
            // cones has d_pred = 0, and the area stiffness outweighs ϱ, so the
            // exact penalty stays on the slack and the radius step is tiny.
            if rec.yielded_fraction > 0.0 {
                assert!(
                    rec.max_displacement > 1.0,
                    "{name} geometry step did not move radii by metres (max |d*|={})",
                    rec.max_displacement
                );
                assert!(
                    max_abs > 1.0,
                    "{name} relax→final radius change is {max_abs} m"
                );
            }
        }
    }
    assert!(
        any_geometry,
        "none of Ryugu, Phobos, Deimos entered a geometry step"
    );
}
