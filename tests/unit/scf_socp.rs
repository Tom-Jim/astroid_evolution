//! Unit tests of `scf::socp` (§2.13, §10.4.3) in normal increments (G4).

use astroid_evolution::common::math::DVec3;
use astroid_evolution::scf::body::Body;

use astroid_evolution::scf::socp::*;
use std::sync::Arc;

use astroid_evolution::scf::body::Topology;

/// A sphere of `level` with the vertex radii `r(i)`, and its unit directions.
fn star(level: usize, r: impl Fn(usize) -> f64) -> (Arc<Topology>, Vec<DVec3>, Body) {
    let (topo, dirs) = Topology::icosphere(level);
    let topo = Arc::new(topo);
    let nv = dirs.len();
    let x = (0..nv).map(|i| dirs[i] * r(i)).collect();
    let body = Body::new(topo.clone(), x, 1000.0, vec![], vec![0.0; nv]).unwrap();
    (topo, dirs, body)
}

/// `body` moved by `d_i` along its vertex normals.
fn moved(body: &Body, d: &[f64]) -> Body {
    let n = body.vertex_normals();
    let mut next = body.clone();
    for i in 0..d.len() {
        next.positions[i] += n[i] * d[i];
    }
    next
}

/// The interior point reached its full tolerances in the tens of iterations it takes on a
/// well-posed program: `AlmostSolved` or a long run here is a conditioning regression.
fn assert_solved(out: &SocpOutput) {
    assert_eq!(out.status, "Solved", "after {} iterations", out.iterations);
    assert!((1..=60).contains(&out.iterations), "{} iterations", out.iterations);
}

/// `Σ (Dℳ)_i d_i + s_M`, the linearised mass change the step books.
fn mass_row(jac: &[f64], out: &SocpOutput) -> f64 {
    jac.iter().zip(&out.d).map(|(j, d)| j * d).sum::<f64>() + out.mass_slack
}

fn max_slope(body: &Body, g: &[DVec3]) -> f64 {
    body.face_geometry()
        .iter()
        .zip(g)
        .map(|((_, n, _), g)| g.try_normalize().unwrap().cross(*n).length())
        .fold(0.0, f64::max)
}

#[test]
fn spike_is_relaxed_with_mass_conserved() {
    let (topo, _, body) = star(2, |i| if i == 0 { 1.25 } else { 1.0 });
    let nv = topo.n_vertices();
    let dens = body.density().unwrap();
    let geo = body.face_geometry();
    let gravity: Vec<DVec3> = geo
        .iter()
        .map(|(c, _, _)| c.try_normalize().unwrap() * -9.8)
        .collect();
    let nf = topo.n_faces();
    let jac = mass_jacobian(&body, &dens);
    let inp = SocpInput {
        body: &body,
        kind: &vec![Kind::Mobile; nv],
        d_pred: &vec![0.0; nv],
        weight: &vec![1.0; nv],
        gravity: &gravity,
        sin_phi: &vec![0.35; nf],
        slope_row: &vec![true; nf],
        mass_jac: &jac,
        mass_residual: 0.0,
        mass: 4.0e3,
        trust: 0.05,
        eps_n: 0.01,
        lower: &[],
        upper: &[],
        varrho: 1e3,
    };
    let out = solve(&inp).unwrap();
    assert_solved(&out);
    let next = moved(&body, &out.d);
    let before = max_slope(&body, &gravity);
    let after = max_slope(&next, &gravity);
    assert!(out.d[0] < -0.01, "spike moved by {}", out.d[0]);
    assert!(after < before, "{before} -> {after}");
    // A reachable target is a hard equality, met to the feasibility tolerance of the
    // normalised row; the slack stays zero.
    let dm = mass_row(&jac, &out);
    assert!(dm.abs() <= 1.0e-6 * 4.0e3, "mass change {dm}");
    assert!(out.mass_slack.abs() <= 1.0e-6 * 4.0e3, "mass slack {}", out.mass_slack);
    assert!(out.d.iter().all(|d| d.abs() <= 0.05 * (1.0 + 1.0e-6)), "trust box left");
}

/// Same spike, with Ryugu's radius and mass. The boxed penalty is `ϱ|s_M|`
/// in kilograms, so this is the scale the equilibration has to carry.
#[test]
fn spike_at_ryugu_scale_still_meets_kkt() {
    let (topo, _, mut body) = star(2, |i| if i == 0 { 500.0 } else { 400.0 });
    body.bulk = 2000.0;
    let nv = topo.n_vertices();
    let dens = body.density().unwrap();
    let geo = body.face_geometry();
    let gravity: Vec<DVec3> = geo
        .iter()
        .map(|(c, _, _)| c.try_normalize().unwrap() * -0.0001)
        .collect();
    let nf = topo.n_faces();
    let jac = mass_jacobian(&body, &dens);
    let mass = 4.5e11;
    let inp = SocpInput {
        body: &body,
        kind: &vec![Kind::Mobile; nv],
        d_pred: &vec![0.0; nv],
        weight: &vec![1.0; nv],
        gravity: &gravity,
        sin_phi: &vec![0.35; nf],
        slope_row: &vec![true; nf],
        mass_jac: &jac,
        mass_residual: 0.0,
        mass,
        trust: 8.0,
        eps_n: 0.01,
        lower: &[],
        upper: &[],
        varrho: 1e3,
    };
    let out = solve(&inp).expect("ryugu-scale spike");
    assert_solved(&out);
    assert!(out.d[0] < -0.5, "spike moved by {}", out.d[0]);
    assert!(after_slope_drops(&body, &out.d, &gravity));
    let dm = mass_row(&jac, &out);
    assert!(dm.abs() <= 1.0e-6 * mass, "mass change {dm} kg");
    assert!(out.d.iter().all(|d| d.abs() <= 8.0 * (1.0 + 1.0e-6)), "trust box left");
}

fn after_slope_drops(body: &Body, d: &[f64], g: &[DVec3]) -> bool {
    max_slope(&moved(body, d), g) < max_slope(body, g)
}

/// A yielded patch at Ryugu's mass and radius: thousands of free nodes, a few
/// hundred slope cones, area weights, and a mass residual the box cannot
/// absorb alone. The unscaled mass columns stay ±1 and the linear cost of
/// σ is ϱ M, which is ϱ |s_M|.
#[test]
fn ryugu_patch_meets_kkt_with_relative_mass_slack() {
    let (topo, dirs, mut body) = star(5, |_| 400.0);
    body.bulk = 2000.0;
    let nv = topo.n_vertices();
    let nf = topo.n_faces();
    let area = body.vertex_areas();
    let kind: Vec<Kind> = dirs
        .iter()
        .map(|d| {
            if d.z > 0.0 {
                Kind::Mobile
            } else {
                Kind::Locked
            }
        })
        .collect();
    let n_mobile = kind.iter().filter(|k| **k == Kind::Mobile).count();
    assert!(
        (4_000..6_000).contains(&n_mobile),
        "expected a few thousand free nodes, got {n_mobile}"
    );
    let slope_faces: Vec<usize> = topo
        .faces
        .iter()
        .enumerate()
        .filter(|(_, tri)| tri.iter().all(|v| kind[*v as usize] == Kind::Mobile))
        .map(|(f, _)| f)
        .take(400)
        .collect();
    assert_eq!(slope_faces.len(), 400);
    let mut slope_row = vec![false; nf];
    for f in &slope_faces {
        slope_row[*f] = true;
    }
    let mut d_pred = vec![0.0; nv];
    for i in 0..nv {
        if kind[i] == Kind::Mobile {
            d_pred[i] = 20.0;
        }
    }
    let gravity: Vec<DVec3> = body
        .face_geometry()
        .iter()
        .map(|(c, _, _)| {
            let radial = c.try_normalize().unwrap();
            (radial + DVec3::new(0.8, 0.0, 0.0))
                .try_normalize()
                .unwrap()
                * -1.0e-4
        })
        .collect();
    let jac = mass_jacobian(&body, &body.density().unwrap());
    let mass = 5.0e11;
    let varrho = 1.0e3;
    let trust = 6.0;
    let mass_residual = 3.0e10;
    let out = solve(&SocpInput {
        body: &body,
        kind: &kind,
        d_pred: &d_pred,
        weight: &area,
        gravity: &gravity,
        sin_phi: &vec![0.2; nf],
        slope_row: &slope_row,
        mass_jac: &jac,
        mass_residual,
        mass,
        trust,
        eps_n: 0.01,
        lower: &[],
        upper: &[],
        varrho,
    })
    .expect("ryugu patch");
    assert_solved(&out);
    let closed = mass_row(&jac, &out);
    assert!(
        (closed - mass_residual).abs() <= 1e-6 * mass,
        "mass row {closed} vs residual {mass_residual}"
    );
    // The box cannot absorb the residual, so the exact penalty books the rest in s_M.
    assert!(out.mass_slack.abs() > 0.0, "an unreachable residual left no mass slack");
    for (i, d) in out.d.iter().enumerate() {
        if kind[i] == Kind::Mobile {
            assert!(
                d.abs() <= trust * (1.0 + 1.0e-3) + 1e-6,
                "free d[{i}] = {d} outside ±{trust}"
            );
        } else {
            assert!(d.abs() < 1e-6, "locked d[{i}] = {d}");
        }
    }
    assert!(
        out.d.iter().fold(0.0_f64, |m, d| m.max(d.abs())) > 1.0,
        "the patch did not move by a metre"
    );
}

/// A trust radius of millimetres at Ryugu's mass (late in a relaxation that shrank Δ):
/// the box cannot reach the mass residual, so the program folds `ϱ M |b − a·x|` into
/// its linear cost, of order 10¹¹ against a quadratic of order 10⁵. The interior point
/// must still solve it rather than report an unbounded direction, and book the
/// unreachable rest in `s_M` with every free node at the box edge it is pushed to.
#[test]
fn a_folded_mass_row_at_a_millimetre_trust_radius_is_solved() {
    let (topo, _, mut body) = star(4, |i| 400.0 + 3.0 * ((i * 13 % 7) as f64 / 7.0));
    body.bulk = 2000.0;
    let nv = topo.n_vertices();
    let nf = topo.n_faces();
    let jac = mass_jacobian(&body, &body.density().unwrap());
    let gravity: Vec<DVec3> = body.face_geometry().iter().map(|(c, _, _)| c.try_normalize().unwrap() * -1.0e-4).collect();
    let (mass, trust, residual) = (4.5e11, 3.0e-3, 1.0e9);
    let out = solve(&SocpInput {
        body: &body,
        kind: &vec![Kind::Mobile; nv],
        d_pred: &vec![0.0; nv],
        weight: &body.vertex_areas(),
        gravity: &gravity,
        sin_phi: &vec![0.35; nf],
        slope_row: &vec![true; nf],
        mass_jac: &jac,
        mass_residual: residual,
        mass,
        trust,
        eps_n: 0.01,
        lower: &[],
        upper: &[],
        varrho: 1e3,
    })
    .expect("folded cone program");
    assert_solved(&out);
    let reach: f64 = jac.iter().map(|j| j.abs() * trust).sum();
    assert!(reach < residual, "the box reaches the residual: the fold is not exercised");
    assert!((mass_row(&jac, &out) - residual).abs() <= 1e-6 * mass, "mass row {} vs {residual}", mass_row(&jac, &out));
    // The penalty outweighs the slope terms: every node goes to the edge that adds mass.
    let at_edge = out.d.iter().zip(&jac).filter(|(d, j)| (**d - trust * j.signum()).abs() <= 1e-6 * trust).count();
    assert!(at_edge * 10 >= nv * 9, "{at_edge} of {nv} nodes at the box edge");
}

/// A program with nothing free (only locked and deposition vertices, no slope cone) is the
/// fixed deposition itself: no solver call, and the mass row books what it misses in `s_M`.
#[test]
fn a_program_with_nothing_free_is_its_fixed_deposition() {
    let (topo, _, body) = star(2, |_| 1.0);
    let (nv, nf) = (topo.n_vertices(), topo.n_faces());
    let jac = mass_jacobian(&body, &body.density().unwrap());
    let mut kind = vec![Kind::Locked; nv];
    let mut d_pred = vec![0.0; nv];
    for i in (0..nv).step_by(5) {
        kind[i] = Kind::Deposit;
        d_pred[i] = 1e-3;
    }
    let gravity: Vec<DVec3> = body.face_geometry().iter().map(|(c, _, _)| c.normalize() * -9.8).collect();
    let out = solve(&SocpInput {
        body: &body,
        kind: &kind,
        d_pred: &d_pred,
        weight: &vec![1.0; nv],
        gravity: &gravity,
        sin_phi: &vec![0.35; nf],
        slope_row: &vec![false; nf],
        mass_jac: &jac,
        mass_residual: 7.0,
        mass: 4.0e3,
        trust: 0.05,
        eps_n: 0.01,
        lower: &[],
        upper: &[],
        varrho: 1e3,
    })
    .expect("an empty program is not an error");
    assert_eq!(out.iterations, 0);
    for i in 0..nv {
        assert!((out.d[i] - d_pred[i]).abs() <= 1e-15, "vertex {i}: {} vs {}", out.d[i], d_pred[i]);
    }
    let booked: f64 = jac.iter().zip(&out.d).map(|(j, d)| j * d).sum();
    assert!((booked + out.mass_slack - 7.0).abs() <= 1e-12 * 7.0);
}

/// `∂n̂/∂x_k` of the CPU face frame against central differences of the normal of the
/// face whose vertex `k` moves along its vertex normal.
#[test]
fn face_frames_match_finite_differences() {
    let (topo, _, body) = star(3, |i| 1.0 + 0.08 * ((i * 17) % 11) as f64 / 11.0);
    let r0 = body.r_mean();
    let ps: Vec<DVec3> = body.positions.iter().map(|p| *p / r0).collect();
    let normals = body.vertex_normals();
    let unit = |p: &[DVec3], t: [u32; 3]| {
        let [a, b, c] = t.map(|v| p[v as usize]);
        (b - a).cross(c - a).normalize()
    };
    let h = 1e-6;
    let mut worst = 0.0_f64;
    for t in topo.faces.iter().take(200) {
        let frame = face_frame(&ps, &normals, *t);
        assert!((frame.n - unit(&ps, *t)).length() < 1e-14);
        for k in 0..3 {
            let v = t[k] as usize;
            let (mut a, mut b) = (ps.clone(), ps.clone());
            a[v] += normals[v] * h;
            b[v] -= normals[v] * h;
            let fd = (unit(&a, *t) - unit(&b, *t)) / (2.0 * h);
            worst = worst.max((fd - frame.nj[k]).length() / fd.length().max(1e-12));
        }
    }
    assert!(worst < 1e-6, "normal sensitivity off by {worst:.3e}");
}

#[test]
fn supercritical_patch_socp_finishes_quickly() {
    // Icosphere level 7 is 327680 faces, past the 196k mesh that made the
    // full-sphere cone program take more than an hour. Only a polar cap
    // (about three percent of the faces) is mobile.
    let (topo, dirs, body) = star(7, |_| 1.0);
    let nv = topo.n_vertices();
    let nf = topo.n_faces();
    let axis = DVec3::new(0.0, 0.0, 1.0);
    let mut score: Vec<(usize, f64)> = topo
        .faces
        .iter()
        .enumerate()
        .map(|(f, tri)| {
            let mean = tri
                .iter()
                .map(|v| dirs[*v as usize])
                .fold(DVec3::ZERO, |acc, d| acc + d);
            (f, mean.try_normalize().unwrap().dot(axis))
        })
        .collect();
    score.sort_by(|a, b| b.1.total_cmp(&a.1));
    let n_cap = (0.03 * nf as f64).round() as usize;
    let mut slope_row = vec![false; nf];
    let mut touch = vec![false; nv];
    for (f, _) in score.iter().take(n_cap) {
        slope_row[*f] = true;
        for v in topo.faces[*f] {
            touch[v as usize] = true;
        }
    }
    let n_active = touch.iter().filter(|t| **t).count();
    let kind: Vec<Kind> = touch
        .iter()
        .map(|on| if *on { Kind::Mobile } else { Kind::Locked })
        .collect();
    let dens = body.density().unwrap();
    let geo = body.face_geometry();
    let gravity: Vec<DVec3> = geo
        .iter()
        .map(|(c, _, _)| c.try_normalize().unwrap() * -9.8)
        .collect();
    let jac = mass_jacobian(&body, &dens);
    let inp = SocpInput {
        body: &body,
        kind: &kind,
        d_pred: &vec![0.0; nv],
        weight: &vec![1.0; nv],
        gravity: &gravity,
        sin_phi: &vec![0.25; nf],
        slope_row: &slope_row,
        mass_jac: &jac,
        mass_residual: 0.0,
        mass: 4.0e3,
        trust: 0.05,
        eps_n: 0.01,
        lower: &[],
        upper: &[],
        varrho: 1e3,
    };
    let wall = std::time::Instant::now();
    let out = solve(&inp).expect("local SOCP");
    let wall_s = wall.elapsed().as_secs_f64();
    let moved_locked = (0..nv)
        .filter(|i| kind[*i] == Kind::Locked)
        .any(|i| out.d[i].abs() > 1e-12);
    eprintln!(
        "local SOCP: faces={nf} slope_rows={n_cap} mobile={n_active} ipm={:.3}s wall={:.3}s iters={} status={}",
        out.solve_seconds, wall_s, out.iterations, out.status
    );
    assert!(!moved_locked, "a locked vertex left static friction");
    assert_solved(&out);
    // The interior point is the smaller share of the call; assembly of 327k faces the rest.
    assert!(out.solve_seconds <= wall_s, "reported {:.2}s > wall {wall_s:.2}s", out.solve_seconds);
}
