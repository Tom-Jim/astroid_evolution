//! Acceptance of spherical-harmonic smoothing.
//! The radius evaluation runs on the Metal GPU in double-single arithmetic.
//! Input and output keep the same triangle list.

use std::sync::OnceLock;

use astroid_evolution::shsmooth::ShGpu;
#[path = "support/sh_selfcheck.rs"]
mod selfcheck;

use selfcheck::Check;

fn gpu() -> &'static ShGpu {
    static GPU: OnceLock<ShGpu> = OnceLock::new();
    GPU.get_or_init(|| ShGpu::new().expect("a Metal device is available"))
}

fn assert_pass(c: &Check) {
    println!("{:<64} {:.3e} <= {:.1e}", c.name, c.value, c.limit);
    assert!(
        c.passed(),
        "{}: {:e} exceeds {:e}",
        c.name,
        c.value,
        c.limit
    );
}

#[test]
fn harmonics_match_closed_form() {
    assert_pass(&selfcheck::closed_form());
}

#[test]
fn quadrature_is_orthonormal_through_degree_2() {
    assert_pass(&selfcheck::orthonormality());
}

#[test]
fn least_squares_recovers_a_band_limited_shape() {
    assert_pass(&selfcheck::coefficient_recovery().unwrap());
}

#[test]
fn spectral_gains_match_their_formulas() {
    assert_pass(&selfcheck::filter_gains());
}

#[test]
fn device_evaluation_matches_the_f64_reference() {
    assert_pass(&selfcheck::gpu_matches_reference(gpu()).unwrap());
}

#[test]
fn smoothing_keeps_the_face_list_and_an_identity_filter() {
    for c in selfcheck::faces_and_identity(gpu()).unwrap() {
        assert_pass(&c);
    }
}

#[test]
fn heat_filter_attenuates_a_degree_6_ripple() {
    assert_pass(&selfcheck::heat_attenuates(gpu()).unwrap());
}

#[test]
fn truncate_to_degree_0_keeps_faces_and_the_mean_radius() {
    assert_pass(&selfcheck::truncate_projects_to_mean(gpu()).unwrap());
}

#[test]
fn a_sphere_is_left_in_place() {
    assert_pass(&selfcheck::sphere_is_fixed(gpu()).unwrap());
}

#[test]
fn device_adds_the_high_frequency_residual() {
    assert_pass(&selfcheck::gpu_adds_residual(gpu()).unwrap());
}

#[test]
fn a_crater_survives_filtering_the_macro_surface() {
    assert_pass(&selfcheck::crater_survives_the_macro_filter(gpu()).unwrap());
}

#[test]
fn weakening_y40_keeps_the_crater_and_drops_the_ridge() {
    for c in selfcheck::equatorial_ridge_attenuation_keeps_the_crater(gpu()).unwrap() {
        assert_pass(&c);
    }
}

#[test]
fn y20_gain_cuts_the_oblate_gap_to_thirty_percent() {
    assert_pass(&selfcheck::y20_attenuation_cuts_oblateness(gpu()).unwrap());
}

#[test]
fn a_45m_cosine_lowers_the_equator_and_leaves_the_pole() {
    for c in selfcheck::equator_drop_lowers_only_the_belt(gpu()).unwrap() {
        assert_pass(&c);
    }
}

#[test]
fn equatorial_spikes_are_cut_and_polar_craters_remain() {
    for c in selfcheck::equatorial_detail_is_cut_and_polar_craters_stay(gpu()).unwrap() {
        assert_pass(&c);
    }
}

#[test]
fn ancient_ryugu_rollback_drops_the_equator() {
    for c in selfcheck::ancient_rollback_lowers_the_equator(gpu()).unwrap() {
        assert_pass(&c);
    }
}

#[test]
fn y40_gain_outside_the_unit_interval_is_rejected() {
    let err = astroid_evolution::shsmooth::Filter::Heat { kappa: 0.0 }
        .ancient(8, &vec![0.0; 81], 1.0, 1.5)
        .unwrap_err();
    assert_eq!(err.kind, "InvalidInput");
}

#[test]
fn quads_and_degenerate_faces_are_refused() {
    let quad = "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3 4\n";
    let err = astroid_evolution::shsmooth::Mesh::parse_obj(quad).unwrap_err();
    assert_eq!(err.kind, "ModelNotClosed");
    let degenerate = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 1 2\n";
    let err = astroid_evolution::shsmooth::Mesh::parse_obj(degenerate).unwrap_err();
    assert_eq!(err.kind, "ModelNotClosed");
}

#[test]
fn a_vertex_at_the_center_is_not_star_shaped() {
    let mesh = astroid_evolution::shsmooth::Mesh::parse_obj(
        "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nf 1 2 3\nf 1 3 4\nf 1 4 2\nf 2 4 3\n",
    )
    .unwrap();
    let err = mesh
        .radial_samples(
            astroid_evolution::shsmooth::Center::Origin
                .resolve(&mesh)
                .unwrap(),
        )
        .unwrap_err();
    assert_eq!(err.kind, "GeometryInvalid");
}

#[test]
fn too_few_samples_cannot_determine_the_fit() {
    let err = astroid_evolution::shsmooth::fit::fit(2, &[[0.0, 0.0, 1.0]], &[1.0], &[1.0], 1e-12)
        .unwrap_err();
    assert_eq!(err.kind, "InvalidInput");
}

fn radius_about(center: [f64; 3], p: [f64; 3]) -> f64 {
    let d = [p[0] - center[0], p[1] - center[1], p[2] - center[2]];
    (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
}

#[test]
fn fuse_of_a_mesh_with_itself_is_the_identity() {
    let mesh = astroid_evolution::shsmooth::Mesh::from_radial(10, |_, _| 1.4).unwrap();
    let out = astroid_evolution::shsmooth::fuse(
        &mesh,
        &mesh,
        &astroid_evolution::shsmooth::FuseOptions {
            center: astroid_evolution::shsmooth::Center::Origin,
        },
    )
    .unwrap();
    assert_eq!(out.mesh.faces, mesh.faces);
    assert_eq!(out.report.high_faces, mesh.face_count());
    assert_eq!(out.report.coarse_faces, mesh.face_count());
    assert!(out.report.faces_preserved);
    let mut err = 0.0_f64;
    for (a, b) in mesh.vertices.iter().zip(&out.mesh.vertices) {
        let d = radius_about([0.0; 3], [a[0] - b[0], a[1] - b[1], a[2] - b[2]]);
        err = err.max(d);
    }
    assert!(err < 1e-8, "identity fuse moved a vertex by {err}");
    assert!(out.report.delta_rms < 1e-8);
}

#[test]
fn fuse_adds_a_coarse_bulge_and_keeps_a_crater() {
    let mut high = astroid_evolution::shsmooth::Mesh::from_radial(12, |_, _| 1.2).unwrap();
    let crater = high
        .vertices
        .iter()
        .enumerate()
        .filter(|(_, v)| v[2] > 0.0)
        .min_by(|(_, a), (_, b)| a[0].abs().total_cmp(&b[0].abs()))
        .map(|(i, _)| i)
        .unwrap();
    let r0 = radius_about([0.0; 3], high.vertices[crater]);
    for c in &mut high.vertices[crater] {
        *c *= (r0 - 0.25) / r0;
    }
    let depth_in = {
        let mut sum = 0.0;
        let mut n = 0.0;
        for f in &high.faces {
            if f.contains(&(crater as u32)) {
                for v in f {
                    if *v as usize != crater {
                        sum += radius_about([0.0; 3], high.vertices[*v as usize]);
                        n += 1.0;
                    }
                }
            }
        }
        sum / n - radius_about([0.0; 3], high.vertices[crater])
    };
    let coarse =
        astroid_evolution::shsmooth::Mesh::from_radial(5, |theta, _| 1.2 + 0.3 * theta.sin())
            .unwrap();
    assert_ne!(coarse.face_count(), high.face_count());
    let out = astroid_evolution::shsmooth::fuse(
        &high,
        &coarse,
        &astroid_evolution::shsmooth::FuseOptions {
            center: astroid_evolution::shsmooth::Center::Origin,
        },
    )
    .unwrap();
    assert_eq!(out.mesh.faces, high.faces);
    assert_eq!(out.mesh.face_count(), high.face_count());
    let depth_out = {
        let mut sum = 0.0;
        let mut n = 0.0;
        for f in &out.mesh.faces {
            if f.contains(&(crater as u32)) {
                for v in f {
                    if *v as usize != crater {
                        sum += radius_about([0.0; 3], out.mesh.vertices[*v as usize]);
                        n += 1.0;
                    }
                }
            }
        }
        sum / n - radius_about([0.0; 3], out.mesh.vertices[crater])
    };
    let retained = (depth_in - depth_out).abs() / depth_in;
    assert!(
        retained < 0.15,
        "crater depth {depth_in} became {depth_out}"
    );
    let eq = out
        .mesh
        .vertices
        .iter()
        .filter(|v| v[2].abs() < 0.15)
        .map(|v| radius_about([0.0; 3], *v))
        .sum::<f64>()
        / out
            .mesh
            .vertices
            .iter()
            .filter(|v| v[2].abs() < 0.15)
            .count() as f64;
    assert!(eq > 1.35, "equator stayed at {eq}");
}

#[test]
fn fuse_refuses_a_coarse_mesh_that_misses_a_direction() {
    let high = astroid_evolution::shsmooth::Mesh::from_radial(4, |_, _| 1.0).unwrap();
    let coarse = astroid_evolution::shsmooth::Mesh {
        vertices: vec![[0.0, 0.0, 2.0], [1.0, 0.0, 2.0], [0.0, 1.0, 2.0]],
        faces: vec![[0, 1, 2]],
        length_unit: "1".to_string(),
    };
    match astroid_evolution::shsmooth::fuse(
        &high,
        &coarse,
        &astroid_evolution::shsmooth::FuseOptions {
            center: astroid_evolution::shsmooth::Center::Origin,
        },
    ) {
        Err(err) => assert_eq!(err.kind, "GeometryInvalid"),
        Ok(_) => panic!("a coarse triangle must not cover every high-resolution direction"),
    }
}
