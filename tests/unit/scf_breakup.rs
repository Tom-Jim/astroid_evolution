//! Unit tests of `scf::breakup` (§12.1–12.2).

use astroid_evolution::common::geometry::tri_mesh::TriMesh;
use astroid_evolution::common::math::DVec3;
use astroid_evolution::scf::body::Body;
use astroid_evolution::scf::breakup::{EMPTY, PartitionOptions, Rigid, Voxels, graph, partition};

/// The surface of a voxelized ball is a closed solid whose volume is the voxels' to within a
/// voxel layer, and the surfaces of two labels together hold the whole ball.
#[test]
fn voxel_surfaces_are_closed_solids_of_their_labels() {
    let r = 100.0;
    let vox = Voxels::new(DVec3::splat(-r), DVec3::splat(r), 40, |p| {
        if p.length() > r { EMPTY } else if p.x < 0.0 { 0 } else { 1 }
    });
    let volume = |which: &dyn Fn(u32) -> bool| {
        let (v, f) = vox.surface(which);
        let mesh = TriMesh::try_new(v, f).expect("a closed mesh");
        mesh.validate_solid().expect("an oriented solid");
        mesh.signed_volume()
    };
    let ball = 4.0 / 3.0 * std::f64::consts::PI * r.powi(3);
    let all = volume(&|l| l != EMPTY);
    // Midpoint vertices put the surface half a voxel inside the outermost centres.
    assert!((all - ball).abs() < 0.08 * ball, "ball volume {all} vs {ball}");
    let (left, right) = (volume(&|l| l == 0), volume(&|l| l == 1));
    assert!(left > 0.0 && right > 0.0 && (left - right).abs() < 0.05 * all, "{left} / {right}");
    assert!((left + right - all).abs() < 0.05 * all, "halves {} vs whole {all}", left + right);
}

/// A stray island of a label is given to the label that surrounds it.
#[test]
fn stray_voxel_islands_join_their_surroundings() {
    let mut vox = Voxels::new(DVec3::splat(-10.0), DVec3::splat(10.0), 20, |p| {
        if p.length() > 9.0 { EMPTY } else if (p - DVec3::new(5.0, 0.0, 0.0)).length() < 1.2 || p.x < -3.0 { 1 } else { 0 }
    });
    vox.connect();
    let island = (0..vox.label.len()).filter(|&i| (vox.centre(i) - DVec3::new(5.0, 0.0, 0.0)).length() < 1.0);
    for i in island {
        assert_eq!(vox.label[i], 0, "the island kept its label");
    }
    assert!(vox.label.iter().any(|l| *l == 1), "the connected part of label 1 is gone");
}

/// Two halves of a box moving as different rigid bodies are found as two blocks along their
/// interface; one rigid motion is one block.
#[test]
fn the_partition_recovers_rigid_blocks() {
    let mut x = Vec::new();
    for i in 0..20 {
        for j in 0..10 {
            for k in 0..10 {
                x.push(DVec3::new(i as f64 - 9.5, j as f64 - 4.5, k as f64 - 4.5));
            }
        }
    }
    let left = Rigid { a: DVec3::new(-1.0, 0.0, 0.2), theta: DVec3::new(0.0, 0.0, 0.05), c: DVec3::new(-5.0, 0.0, 0.0) };
    let right = Rigid { a: DVec3::new(0.8, 0.1, 0.0), theta: DVec3::new(0.03, -0.02, 0.0), c: DVec3::new(5.0, 0.0, 0.0) };
    let v: Vec<DVec3> = x.iter().map(|p| if p.x < 0.0 { left.at(*p) } else { right.at(*p) }).collect();
    let m = vec![1.0; x.len()];
    let nb = graph(&x, 10);
    let opt = PartitionOptions { max_blocks: 5, gain: 0.3, min_fraction: 0.01 };
    let (labels, misfit) = partition(&x, &v, &m, &nb, &opt);
    assert_eq!(misfit.len(), 2, "blocks found: {misfit:?}");
    assert!(misfit[1] < 1e-6, "two exact rigid motions leave misfit {}", misfit[1]);
    let side = |p: &DVec3| usize::from(p.x >= 0.0);
    let (a, b) = (labels[0], labels[x.len() - 1]);
    assert_ne!(a, b);
    let wrong = x.iter().zip(&labels).filter(|(p, l)| **l != if side(p) == 0 { a } else { b }).count();
    assert_eq!(wrong, 0, "{wrong} nodes on the wrong side");

    let one: Vec<DVec3> = x.iter().map(|p| left.at(*p)).collect();
    let (labels, misfit) = partition(&x, &one, &m, &nb, &opt);
    assert_eq!(misfit.len(), 1);
    assert!(labels.iter().all(|l| *l == 0));
}

/// A crescent has its centroid outside: rays from there miss it, rays from its deepest voxel
/// re-triangulate it.
#[test]
fn a_crescent_is_retriangulated_from_inside() {
    let vox = Voxels::new(DVec3::splat(-12.0), DVec3::splat(12.0), 48, |p| {
        let r = (p.x * p.x + p.y * p.y).sqrt();
        if (7.0..11.0).contains(&r) && p.z.abs() < 2.0 && p.x > -3.0 { 0 } else { EMPTY }
    });
    let (v, f) = vox.surface(|l| l == 0);
    assert!(Body::rays_from_mesh(3, &v, &f).is_err(), "rays from an outside centroid should miss");
    let inner = vox.deepest(|l| l == 0).unwrap();
    let r = (inner.x * inner.x + inner.y * inner.y).sqrt();
    assert!((7.0..11.0).contains(&r) && inner.z.abs() < 2.0, "deepest point {inner:?} is not inside");
    let (_, pos, origin) = Body::rays_from(3, &v, &f, inner).unwrap();
    assert_eq!(origin, inner);
    assert!(pos.iter().all(|p| p.is_finite()));
}
