//! Unit tests of `scf::socp`, moved out of `src/scf/socp.rs`.

mod ctx {
    pub use astroid_evolution::common::density::CauchyDensity;
    pub use astroid_evolution::common::error::SimError;
    pub use astroid_evolution::common::math::DVec3;
    pub use astroid_evolution::scf::body::Body;
    pub use astroid_evolution::scf::socp_gpu::{FaceFrame, face_frames};
    pub use rayon::prelude::*;
}
use ctx::*;

use astroid_evolution::scf::socp::*;
use std::sync::Arc;

use astroid_evolution::scf::body::Topology;

fn max_slope(body: &Body, g: &[DVec3]) -> f64 {
    body.face_geometry()
        .iter()
        .zip(g)
        .map(|((_, n, _), g)| g.try_normalize().unwrap().cross(*n).length())
        .fold(0.0, f64::max)
}

#[test]
fn spike_is_relaxed_with_mass_conserved() {
    let topo = Arc::new(Topology::icosphere(2));
    let nv = topo.n_vertices();
    let mut radius = vec![1.0; nv];
    radius[0] = 1.25;
    let body = Body::new(topo.clone(), radius, 1000.0, vec![], vec![0.0; nv]).unwrap();
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
        trust: 0.1,
        eps_n: 0.01,
        varrho: 1e3,
        warm: None,
    };
    let out = solve(&inp).unwrap();
    let mut next = body.clone();
    for i in 0..nv {
        next.radius[i] += out.d[i];
    }
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
    let topo = Arc::new(Topology::icosphere(2));
    let nv = topo.n_vertices();
    let mut radius = vec![400.0; nv];
    radius[0] = 500.0;
    let body = Body::new(topo.clone(), radius, 2000.0, vec![], vec![0.0; nv]).unwrap();
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
    let topo = Arc::new(Topology::icosphere(5));
    let nv = topo.n_vertices();
    let nf = topo.n_faces();
    let body = Body::new(topo.clone(), vec![400.0; nv], 2000.0, vec![], vec![0.0; nv]).unwrap();
    let area = astroid_evolution::scf::frame::vertex_areas(&body);
    let kind: Vec<Kind> = topo
        .dirs
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

/// f64 oracle for the face frame. Production assembly does not call this.
fn reference_frame(pa: DVec3, pb: DVec3, pc: DVec3, w: [DVec3; 3]) -> (DVec3, [DVec3; 3]) {
    let area_vec = (pb - pa).cross(pc - pa);
    let len = area_vec.length();
    let n = area_vec * (1.0 / len);
    let da = [
        w[0].cross(pb - pc),
        w[1].cross(pc - pa),
        w[2].cross(pa - pb),
    ];
    let proj = |v: DVec3| (v - n * v.dot(n)) * (1.0 / len);
    (n, [proj(da[0]), proj(da[1]), proj(da[2])])
}

fn relative(a: DVec3, b: DVec3) -> f64 {
    (a - b).length() / a.length().max(b.length()).max(1e-30)
}

#[test]
fn gpu_face_frames_match_f64() {
    let topo = Topology::icosphere(4);
    let nv = topo.n_vertices();
    let radius: Vec<f64> = (0..nv)
        .map(|i| 1.0 + 0.08 * ((i * 17) % 11) as f64 / 11.0)
        .collect();
    let r0 = radius.iter().sum::<f64>() / nv as f64;
    let ps: Vec<DVec3> = topo
        .dirs
        .iter()
        .zip(&radius)
        .map(|(d, r)| *d * (*r / r0))
        .collect();
    let gpu = astroid_evolution::scf::socp_gpu::face_frames(&ps, &topo.dirs, &topo.faces)
        .expect("Metal face-frame kernel");
    let mut worst_n = 0.0_f64;
    let mut worst_j = 0.0_f64;
    for (f, tri) in topo.faces.iter().enumerate() {
        let v = tri.map(|i| i as usize);
        let (n, nj) = reference_frame(ps[v[0]], ps[v[1]], ps[v[2]], v.map(|i| topo.dirs[i]));
        worst_n = worst_n.max(relative(n, gpu[f].n));
        for k in 0..3 {
            worst_j = worst_j.max(relative(nj[k], gpu[f].nj[k]));
        }
    }
    eprintln!("face-frame ds error: normal {worst_n:.3e}, sensitivity {worst_j:.3e}");
    assert!(
        worst_n < 1e-6,
        "unit normal left the f64 oracle by {worst_n:.3e}"
    );
    assert!(
        worst_j < 1e-6,
        "normal sensitivity left the f64 oracle by {worst_j:.3e}"
    );
}

#[test]
fn supercritical_patch_socp_finishes_quickly() {
    // Icosphere level 7 is 327680 faces, past the 196k mesh that made the
    // full-sphere cone program take more than an hour. Only a polar cap
    // (about three percent of the faces) is mobile.
    let topo = Arc::new(Topology::icosphere(7));
    let nv = topo.n_vertices();
    let nf = topo.n_faces();
    let radius = vec![1.0; nv];
    let axis = DVec3::new(0.0, 0.0, 1.0);
    let mut score: Vec<(usize, f64)> = topo
        .faces
        .iter()
        .enumerate()
        .map(|(f, tri)| {
            let mean = tri
                .iter()
                .map(|v| topo.dirs[*v as usize])
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
    let body = Body::new(topo.clone(), radius, 1000.0, vec![], vec![0.0; nv]).unwrap();
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
        let topo = Arc::new(Topology::icosphere(2));
        let nv = topo.n_vertices();
        let nf = topo.n_faces();
        let mut radius = vec![1.0; nv];
        radius[0] = 1.25;
        let body = Body::new(topo.clone(), radius, 1000.0, vec![], vec![0.0; nv]).unwrap();
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
            varrho: 1e3,
            warm,
        }
    }
}

#[test]
fn the_same_cone_program_accepts_its_own_multipliers() {
    let spike = Spike::new();
    let first = solve(&spike.input(0.1, None)).expect("cold SOCP");
    let second = solve(&spike.input(0.1, Some(&first.seed))).expect("warm SOCP");
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
    let full = solve(&spike.input(0.1, None)).expect("full cones");
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
    let cold = solve(&spike.input(0.1, None)).expect("cold reduced");
    let hot = solve(&spike.input(0.1, Some(&full.seed))).expect("warm reduced");
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

fn tetrahedron(invert: Option<usize>, radius: f64) -> Body {
    let s = 1.0 / 3.0_f64.sqrt();
    let dirs = vec![
        DVec3::new(s, s, s),
        DVec3::new(s, -s, -s),
        DVec3::new(-s, s, -s),
        DVec3::new(-s, -s, s),
    ];
    let mut faces = vec![[0u32, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]];
    let build = |faces: Vec<[u32; 3]>| {
        let topo = Arc::new(Topology::new(dirs.clone(), faces));
        Body::new(topo, vec![radius; 4], 1000.0, vec![], vec![0.0; 4]).unwrap()
    };
    let trial = build(faces.clone());
    let geo = trial.face_geometry();
    for (f, (_, n, _)) in geo.iter().enumerate() {
        if n.dot(trial.topo.face_dirs[f]) < 0.0 {
            faces[f].swap(1, 2);
        }
    }
    if let Some(f) = invert {
        faces[f].swap(1, 2);
    }
    build(faces)
}

fn face_radial(body: &Body) -> Vec<f64> {
    body.face_geometry()
        .iter()
        .enumerate()
        .map(|(f, (_, n, _))| n.dot(body.topo.face_dirs[f]))
        .collect()
}

#[test]
fn one_vertex_uses_the_documented_weight_and_varrho() {
    // §10.4.3: ½ w (d − d_pred)² + ϱ (t + |s_M|). The solver unknown is
    // x = d/R₀ and the mass slack is σ = s_M/M, so the stored coefficients
    // are w R₀², −w R₀ d_pred, ϱ on each slope slack, and ϱ M on σ.
    let radius = 2.0;
    let body = tetrahedron(None, radius);
    let nv = body.topo.n_vertices();
    let nf = body.topo.n_faces();
    let r0 = body.radius.iter().sum::<f64>() / nv as f64;
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
    let body = tetrahedron(None, 1.0);
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
        out.n_nn, 6,
        "s⁺, s⁻, and two box rows on each of two vertices"
    );
}

#[test]
fn an_inward_face_adds_no_normal_floor() {
    let body = tetrahedron(Some(3), 1.0);
    let radial = face_radial(&body);
    assert!(
        radial[3] < -0.25,
        "face 3 should sit below the old floor, n·ê = {}",
        radial[3]
    );
    assert_eq!(radial.iter().filter(|c| **c > -0.25).count(), 3);
    let nv = body.topo.n_vertices();
    let nf = body.topo.n_faces();
    let jac = mass_jacobian(&body, &body.density().unwrap());
    let gravity: Vec<DVec3> = body
        .face_geometry()
        .iter()
        .map(|(c, _, _)| c.try_normalize().unwrap() * -9.8)
        .collect();
    let out = solve(&SocpInput {
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
        trust: 0.1,
        eps_n: 0.01,
        varrho: 1.0e3,
        warm: None,
    })
    .expect("inward face");
    // Three outward faces each contribute t ≥ 0 and one overhang row.
    // s⁺ and s⁻ make two more. Four mobile vertices add eight trust-box rows.
    // The inward face, and every n·ê row, add none.
    assert_eq!(out.n_nn, 16);
    assert_eq!(out.n_soc, 3);
    assert_eq!(out.n_trust, 0);
}
