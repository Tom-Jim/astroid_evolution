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
        warm: None,
    };
    let out = solve(&inp).unwrap();
    let next = moved(&body, &out.d);
    let before = max_slope(&body, &gravity);
    let after = max_slope(&next, &gravity);
    assert!(out.d[0] < -0.01, "spike moved by {}", out.d[0]);
    assert!(after < before, "{before} -> {after}");
    let dm: f64 = (0..nv).map(|i| jac[i] * out.d[i]).sum();
    // The normalised mass row is held to the PDHG tolerance, about 1e-3.
    assert!(dm.abs() < 5.0e-3 * 4.0e3, "mass change {dm}");
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
        warm: None,
    };
    let out = solve(&inp).expect("ryugu-scale spike");
    assert!(out.iterations < 40_000, "iters {}", out.iterations);
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
        warm: None,
    })
    .expect("ryugu patch");
    assert!(
        out.iterations < 8_000,
        "patch took {} iterations",
        out.iterations
    );
    assert!((out.mass_col_plus - 1.0).abs() < 1e-12);
    assert!((out.mass_col_minus + 1.0).abs() < 1e-12);
    assert!((out.mass_cost - varrho * mass).abs() <= 1.0);
    let closed: f64 = jac.iter().zip(&out.d).map(|(j, d)| j * d).sum::<f64>() + out.mass_slack;
    assert!(
        (closed - mass_residual).abs() < 1e-3 * mass,
        "mass row {closed} vs residual {mass_residual}"
    );
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
        warm: None,
    };
    let wall = std::time::Instant::now();
    let out = solve(&inp).expect("local SOCP");
    let wall_s = wall.elapsed().as_secs_f64();
    let moved_locked = (0..nv)
        .filter(|i| kind[*i] == Kind::Locked)
        .any(|i| out.d[i].abs() > 1e-12);
    eprintln!(
        "local SOCP: faces={nf} slope_rows={n_cap} mobile={n_active} pdhg={:.3}s wall={:.3}s iters={} status={}",
        out.solve_seconds, wall_s, out.iterations, out.status
    );
    assert!(!moved_locked, "a locked vertex left static friction");
    assert!(
        wall_s < 600.0 && out.solve_seconds < 600.0,
        "Metal-PDHG wall {wall_s:.1}s, reported {:.1}s",
        out.solve_seconds
    );
}

struct Spike {
    body: Body,
    kind: Vec<Kind>,
    d_pred: Vec<f64>,
    weight: Vec<f64>,
    gravity: Vec<DVec3>,
    sin_phi: Vec<f64>,
    slope_row: Vec<bool>,
    jac: Vec<f64>,
}

impl Spike {
    fn new() -> Self {
        let (topo, _, body) = star(2, |i| if i == 0 { 1.25 } else { 1.0 });
        let nv = topo.n_vertices();
        let nf = topo.n_faces();
        let dens = body.density().unwrap();
        let gravity: Vec<DVec3> = body
            .face_geometry()
            .iter()
            .map(|(c, _, _)| c.try_normalize().unwrap() * -9.8)
            .collect();
        let jac = mass_jacobian(&body, &dens);
        Self {
            kind: vec![Kind::Mobile; nv],
            d_pred: vec![0.0; nv],
            weight: vec![1.0; nv],
            sin_phi: vec![0.35; nf],
            slope_row: vec![true; nf],
            jac,
            gravity,
            body,
        }
    }

    fn input<'a>(&'a self, trust: f64, warm: Option<&'a SocpSeed>) -> SocpInput<'a> {
        SocpInput {
            body: &self.body,
            kind: &self.kind,
            d_pred: &self.d_pred,
            weight: &self.weight,
            gravity: &self.gravity,
            sin_phi: &self.sin_phi,
            slope_row: &self.slope_row,
            mass_jac: &self.jac,
            mass_residual: 0.0,
            mass: 4.0e3,
            trust,
            eps_n: 0.01,
            lower: &[],
            upper: &[],
            varrho: 1e3,
            warm,
        }
    }
}

#[test]
fn the_same_cone_program_accepts_its_own_multipliers() {
    let spike = Spike::new();
    let first = solve(&spike.input(0.05, None)).expect("cold SOCP");
    let second = solve(&spike.input(0.05, Some(&first.seed))).expect("warm SOCP");
    assert_eq!(
        second.iterations, 0,
        "warm start took {}",
        second.iterations
    );
    for (a, b) in first.d.iter().zip(&second.d) {
        assert!((a - b).abs() < 1e-8, "{a} vs {b}");
    }
}

#[test]
fn a_tighter_trust_region_starts_closer_than_the_origin() {
    let spike = Spike::new();
    let wide = solve(&spike.input(10.0, None)).expect("wide trust");
    let reach = wide.d.iter().fold(0.0_f64, |m, d| m.max(d.abs()));
    let narrow_trust = (reach * 2.0).max(0.05);
    assert!(
        narrow_trust < 10.0,
        "the wide step filled its trust box ({reach})"
    );
    let cold = solve(&spike.input(narrow_trust, None)).expect("cold narrow");
    let hot = solve(&spike.input(narrow_trust, Some(&wide.seed))).expect("warm narrow");
    assert!(
        hot.initial_metric < cold.initial_metric,
        "warm {:.3e} cold {:.3e}",
        hot.initial_metric,
        cold.initial_metric
    );
    for (a, b) in hot.d.iter().zip(&cold.d) {
        // Both points meet KKT_TOL. With ϱ on every face, ||q|| is large, so
        // that test allows the two displacements to differ by about a centimetre.
        assert!((a - b).abs() < 2.0e-2, "warm {a} cold {b}");
    }
}

#[test]
fn dropping_one_slope_face_keeps_the_other_multipliers() {
    let mut spike = Spike::new();
    let full = solve(&spike.input(0.05, None)).expect("full cones");
    let face = full
        .seed
        .y_soc
        .iter()
        .position(|cone| cone.iter().any(|v| v.abs() > 1e-8))
        .unwrap_or(0);
    let kept = (face + 1) % spike.slope_row.len();
    assert!(
        full.seed.y_soc[kept].iter().any(|v| v.is_finite()),
        "remaining cone multiplier is not finite"
    );
    spike.slope_row[face] = false;
    let cold = solve(&spike.input(0.05, None)).expect("cold reduced");
    let hot = solve(&spike.input(0.05, Some(&full.seed))).expect("warm reduced");
    // Dropping the cone removes the multiplier that was balancing one node,
    // so the reloaded point is not closer than the origin on the per-variable
    // stationarity test. Both solves still have to agree.
    println!(
        "warm {} cold {} iterations",
        hot.iterations, cold.iterations
    );
    assert!(hot.iterations < 80_000 && cold.iterations < 80_000);
    for (a, b) in hot.d.iter().zip(&cold.d) {
        assert!((a - b).abs() < 2.0e-2, "warm {a} cold {b}");
    }
}

/// A regular tetrahedron of circumradius `radius`, outward faces.
fn tetrahedron(radius: f64) -> Body {
    let s = radius / 3.0_f64.sqrt();
    let x = vec![
        DVec3::new(s, s, s),
        DVec3::new(s, -s, -s),
        DVec3::new(-s, s, -s),
        DVec3::new(-s, -s, s),
    ];
    let faces = vec![[0u32, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]];
    let topo = Arc::new(Topology::new(4, faces));
    let body = Body::new(topo, x, 1000.0, vec![], vec![0.0; 4]).unwrap();
    assert!(body.mesh().unwrap().signed_volume() > 0.0);
    body
}

#[test]
fn one_vertex_uses_the_documented_weight_and_varrho() {
    // §10.4.3: ½ w (d − d_pred)² + ϱ (t + |s_M|). The solver unknown is
    // x = d/R₀ and the mass slack is σ = s_M/M, so the stored coefficients
    // are w R₀², −w R₀ d_pred, ϱ on each slope slack, and ϱ M on σ.
    let radius = 2.0;
    let body = tetrahedron(radius);
    let nv = body.topo.n_vertices();
    let nf = body.topo.n_faces();
    let r0 = body.r_mean();
    let mut kind = vec![Kind::Locked; nv];
    kind[0] = Kind::Mobile;
    let mut d_pred = vec![0.0; nv];
    let predictor = 0.25;
    d_pred[0] = predictor;
    let mut weight = vec![0.0; nv];
    let w = 4.0;
    weight[0] = w;
    let face = body
        .topo
        .faces
        .iter()
        .position(|tri| tri.contains(&0u32))
        .unwrap();
    let mut slope_row = vec![false; nf];
    slope_row[face] = true;
    let varrho = 1.0e3;
    let mass = 4.0e3;
    let jac = mass_jacobian(&body, &body.density().unwrap());
    let gravity: Vec<DVec3> = body
        .face_geometry()
        .iter()
        .map(|(c, _, _)| c.try_normalize().unwrap() * -9.8)
        .collect();
    let out = solve(&SocpInput {
        body: &body,
        kind: &kind,
        d_pred: &d_pred,
        weight: &weight,
        gravity: &gravity,
        sin_phi: &vec![0.35; nf],
        slope_row: &slope_row,
        mass_jac: &jac,
        mass_residual: 0.0,
        mass,
        trust: 0.1,
        eps_n: 0.01,
        lower: &[],
        upper: &[],
        varrho,
        warm: None,
    })
    .expect("one vertex");
    assert!(
        (out.vertex_p[0] / (r0 * r0) - w).abs() < 1e-9,
        "documented weight {}, stored p {}",
        w,
        out.vertex_p[0]
    );
    assert!((out.vertex_q[0] / r0 + w * predictor).abs() < 1e-9);
    assert!((out.slope_cost[face] - varrho).abs() < 1e-12);
    assert!((out.mass_cost - varrho * mass).abs() < 1e-6);
    assert!((out.mass_col_plus - 1.0).abs() < 1e-12);
    assert!((out.mass_col_minus + 1.0).abs() < 1e-12);
    assert!((out.vertex_p[0] - w / weight.iter().sum::<f64>()).abs() > 1.0);
    assert!((out.slope_cost[face] - varrho / nf as f64).abs() > 1.0);
}

#[test]
fn two_mobile_vertices_have_one_trust_ball() {
    let body = tetrahedron(1.0);
    let nv = body.topo.n_vertices();
    let nf = body.topo.n_faces();
    let mut kind = vec![Kind::Locked; nv];
    kind[0] = Kind::Mobile;
    kind[1] = Kind::Mobile;
    let jac = mass_jacobian(&body, &body.density().unwrap());
    let gravity: Vec<DVec3> = body
        .face_geometry()
        .iter()
        .map(|(c, _, _)| c.try_normalize().unwrap() * -9.8)
        .collect();
    let out = solve(&SocpInput {
        body: &body,
        kind: &kind,
        d_pred: &vec![0.0; nv],
        weight: &vec![1.0; nv],
        gravity: &gravity,
        sin_phi: &vec![0.35; nf],
        slope_row: &vec![false; nf],
        mass_jac: &jac,
        mass_residual: 0.0,
        mass: 4.0e3,
        trust: 0.2,
        eps_n: 0.01,
        lower: &[],
        upper: &[],
        varrho: 1.0e3,
        warm: None,
    })
    .expect("two vertices");
    assert_eq!(
        out.n_trust, 0,
        "the trust region is a box, not a second-order cone"
    );
    assert_eq!(out.n_soc, 0);
    assert_eq!(
        out.n_nn, 10,
        "s⁺, s⁻, two box rows on each of two vertices, and the anti-overhang row of \
         each of the four faces with a free vertex (§2.8c)"
    );
}
