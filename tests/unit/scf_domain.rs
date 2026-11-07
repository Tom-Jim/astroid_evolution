//! Unit tests of `scf::domain` and `scf::surface` (§8.1 G2, G3, G6) on a body that is not
//! star-shaped: a torus, whose origin lies outside `𝒱`.

use std::f64::consts::PI;
use std::sync::Arc;

use astroid_evolution::common::math::DVec3;
use astroid_evolution::rtfp::mesh::BodyMesh;
use astroid_evolution::scf::body::{Body, Topology};
use astroid_evolution::scf::domain::{SurfacePoints, volume_rule};
use astroid_evolution::scf::surface::{SurfaceIndex, closest_on_triangle};

/// A torus of radii `(R, a)` about `e_z`, `nu × nv` quads split into outward triangles.
fn torus(big: f64, a: f64, nu: usize, nv: usize) -> Body {
    let mut x = Vec::with_capacity(nu * nv);
    for i in 0..nu {
        let u = 2.0 * PI * i as f64 / nu as f64;
        for j in 0..nv {
            let v = 2.0 * PI * j as f64 / nv as f64;
            let r = big + a * v.cos();
            x.push(DVec3::new(r * u.cos(), r * u.sin(), a * v.sin()));
        }
    }
    let id = |i: usize, j: usize| ((i % nu) * nv + j % nv) as u32;
    let mut faces = Vec::with_capacity(2 * nu * nv);
    for i in 0..nu {
        for j in 0..nv {
            faces.push([id(i, j), id(i + 1, j), id(i + 1, j + 1)]);
            faces.push([id(i, j), id(i + 1, j + 1), id(i, j + 1)]);
        }
    }
    let n = x.len();
    let body = Body::new(Arc::new(Topology::new(n, faces)), x, 1.0, vec![], vec![0.0; n]).unwrap();
    assert!(body.mesh().unwrap().signed_volume() > 0.0, "torus faces must be outward");
    body
}

/// The ray-segment rule (G2) integrates the volume of a body the rays from the origin
/// cross twice; refining the cut cells reduces the silhouette error.
#[test]
fn ray_segments_integrate_a_torus_whose_origin_is_outside() {
    let body = torus(2.0, 0.7, 64, 32);
    let exact = body.mesh().unwrap().signed_volume();
    let mesh = BodyMesh::build(&body.mesh().unwrap()).unwrap();
    let err = |depth: usize| {
        let (_, w, _, cut) = volume_rule(&mesh, 16, 32, 3, depth).unwrap();
        ((w.iter().sum::<f64>() - exact) / exact, cut)
    };
    let (coarse, _) = err(0);
    let (fine, cut) = err(4);
    assert!(cut > 0, "the torus silhouette must cut cells");
    assert!(fine.abs() < 0.02 && fine.abs() < coarse.abs(), "{coarse:.3e} -> {fine:.3e}");
}

/// Closest points of the BVH (G6) against a brute-force scan of all triangles.
#[test]
fn closest_points_match_brute_force() {
    let body = torus(2.0, 0.7, 24, 12);
    let index = SurfaceIndex::build(&body.positions, &body.topo.faces);
    for k in 0..200 {
        let t = k as f64 * 0.37;
        let q = DVec3::new(3.0 * t.sin(), 3.0 * (1.3 * t).cos(), 1.2 * (0.7 * t).sin());
        let brute = body
            .topo
            .faces
            .iter()
            .map(|f| {
                let [a, b, c] = f.map(|v| body.positions[v as usize]);
                (closest_on_triangle(q, a, b, c) - q).length_squared()
            })
            .fold(f64::INFINITY, f64::min);
        assert!((index.closest(q).dist2 - brute).abs() <= 1e-12 * brute.max(1.0), "{k}");
    }
    assert!(index.inside(DVec3::new(2.0, 0.0, 0.0)) && !index.inside(DVec3::ZERO));
}

/// The boundary rule of (G3) keeps the area and the closure `∮ n dS = 0`.
#[test]
fn boundary_rule_keeps_area_and_closure() {
    let body = torus(2.0, 0.7, 48, 24);
    let faces = body.face_geometry();
    let rule = SurfacePoints::clustered(&faces, 256);
    let area: f64 = faces.iter().map(|f| f.2).sum();
    assert!((rule.w.iter().sum::<f64>() - area).abs() < 1e-12 * area);
    let closure = (0..rule.pts.len()).fold(DVec3::ZERO, |s, k| s + rule.normals[k] * rule.w[k]);
    assert!(closure.length() < 0.05 * area, "{closure:?}");
    assert!(rule.pts.len() < faces.len());
}
