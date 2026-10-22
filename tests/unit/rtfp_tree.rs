//! Unit tests of `rtfp::tree`, moved out of `src/rtfp/tree.rs`.

mod ctx {
    pub use astroid_evolution::common::math::DVec3;
    pub use rayon::prelude::*;
}
use ctx::*;

use astroid_evolution::common::geometry::TriMesh;
use astroid_evolution::rtfp::reference::werner;
use astroid_evolution::rtfp::tree::*;

#[test]
fn root_expansion_matches_werner_far_away() {
    let mesh = TriMesh::uv_sphere(16, 1.0);
    let tris: Vec<[DVec3; 3]> = (0..mesh.faces.len()).map(|f| mesh.triangle(f)).collect();
    let (nodes, order) = build(&tris);
    assert_eq!(order.len(), tris.len());
    for x in [DVec3::new(6.0, 1.0, -2.0), DVec3::new(-3.0, 8.0, 4.0)] {
        let w = werner::uniform_polyhedron(&mesh, x, 1.0, 1.0);
        let (u, g, h) = far_field(&nodes[0], x);
        assert!(
            (u - w.potential).abs() < 2e-4 * w.potential.abs(),
            "{u} {}",
            w.potential
        );
        assert!((g - w.gradient).length() < 2e-4 * w.gradient.length());
        let hs = w.hessian.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        for k in 0..6 {
            assert!(
                (h[k] - w.hessian[k]).abs() < 2e-3 * hs,
                "{k}: {} {}",
                h[k],
                w.hessian[k]
            );
        }
    }
}

#[test]
fn node_expansion_is_fourth_order() {
    let doc = astroid_evolution::io::load_mesh_file(astroid_evolution::io::ImportRequest {
        path: std::path::Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/dist/.stage/assets/models/Phobos.glb"
        )),
        length_unit: Some("km"),
        reconstruct: None,
    })
    .unwrap();
    let mesh = doc.mesh;
    let tris: Vec<[DVec3; 3]> = (0..mesh.faces.len()).map(|f| mesh.triangle(f)).collect();
    let (nodes, order) = build(&tris);
    let slot: Vec<[DVec3; 3]> = order.iter().map(|&i| tris[i as usize]).collect();
    for idx in [5usize, 40, 300] {
        let n = &nodes[idx];
        let c = DVec3::new(n.c[0].into(), n.c[1].into(), n.c[2].into());
        let mut range = vec![idx];
        let mut faces = vec![];
        while let Some(i) = range.pop() {
            let m = &nodes[i];
            if m.count > 0 {
                faces.extend(m.first as usize..(m.first + m.count) as usize);
            } else {
                range.push(m.a as usize);
                range.push(m.b as usize);
            }
        }
        for th in [0.4, 0.2, 0.1] {
            let x = c + DVec3::new(0.48, 0.6, 0.64) * (f64::from(n.h) / th);
            let mut ex = astroid_evolution::rtfp::reference::werner::FaceTerm::default();
            for &f in &faces {
                ex.add(&astroid_evolution::rtfp::reference::werner::face_term(
                    slot[f], x, false,
                ));
            }
            let w = ex.scaled(1.0);
            let (u, g, _) = far_field(n, x);
            let (du, dg) = (
                (u - w.potential).abs() / w.potential.abs(),
                (g - w.gradient).length() / w.gradient.length(),
            );
            // Octupole order: the error falls at least as θ⁴.
            let bound = 4e-3 * th.powi(4) / 0.1f64.powi(4) * 1e-3;
            assert!(
                du < bound && dg < bound,
                "node {idx} θ {th}: {du:.2e} {dg:.2e}"
            );
        }
    }
}

/// Relative error of the traversal against brute force, on and inside a real shape.
#[test]
fn traversal_matches_werner_on_phobos() {
    let doc = astroid_evolution::io::load_mesh_file(astroid_evolution::io::ImportRequest {
        path: std::path::Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/dist/.stage/assets/models/Phobos.glb"
        )),
        length_unit: Some("km"),
        reconstruct: None,
    })
    .unwrap();
    let mesh = doc.mesh;
    let tris: Vec<[DVec3; 3]> = (0..mesh.faces.len()).map(|f| mesh.triangle(f)).collect();
    let scale = tris
        .iter()
        .flatten()
        .map(|p| p.length())
        .fold(0.0, f64::max);
    let tris: Vec<[DVec3; 3]> = tris.iter().map(|t| t.map(|p| p / scale)).collect();
    let (nodes, order) = build(&tris);
    let slot: Vec<[DVec3; 3]> = order.iter().map(|&i| tris[i as usize]).collect();
    let unit = TriMesh {
        vertices: mesh.vertices.iter().map(|p| *p / scale).collect(),
        faces: mesh.faces.clone(),
    };
    let mut pts: Vec<(DVec3, Option<usize>)> =
        vec![(DVec3::ZERO, None), (DVec3::new(0.3, -0.2, 0.1), None)];
    for s in (0..order.len()).step_by(order.len() / 40) {
        let t = slot[s];
        pts.push(((t[0] + t[1] + t[2]) / 3.0, Some(s)));
    }
    for theta in [THETA, 0.2, 0.3] {
        let (mut eu, mut eg, mut eh) = (0.0f64, 0.0f64, 0.0f64);
        for &(x, own) in &pts {
            let w = werner::uniform_polyhedron(&unit, x, 1.0, 1.0);
            let (u, g, h, om) = eval(&nodes, &slot, x, theta, own);
            eu = eu.max((u - w.potential).abs() / w.potential.abs());
            eg = eg.max((g - w.gradient).length() / w.gradient.length().max(0.5));
            let hs = w.hessian.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            for k in 0..6 {
                eh = eh.max((h[k] - w.hessian[k]).abs() / hs);
            }
            let inside = own.is_none();
            assert!(
                if inside {
                    om > 3.0 * std::f64::consts::PI
                } else {
                    (om - 2.0 * std::f64::consts::PI).abs() < 0.5
                },
                "{om}"
            );
        }
        eprintln!("theta {theta}: U {eu:.2e}  grad {eg:.2e}  hess {eh:.2e}");
        if theta <= THETA {
            assert!(eu < 3e-6 && eg < 1e-5 && eh < 3e-5, "theta {theta}");
        }
    }
}
