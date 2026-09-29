//! 阶段 1：输入和几何 完备单元与集成测试套件
//! 依照 docs/software.md 与 docs/mathstrict.md

use std::path::Path;
use astroid_evolution::file_support::load_project_toml;
use astroid_evolution::file_support::point_cloud::PointCloud;
use astroid_evolution::file_support::surface_reconstruction::{reconstruct_surface, ReconstructionMethod};
use astroid_evolution::file_support::unit_system::UnitSystem;
use astroid_evolution::rtfp::bvh::{build_flattened_bvh, intersect_ray_bvh, is_point_inside};
use astroid_evolution::rtfp::geometry_validation::validate_closed_polyhedron;
use astroid_evolution::rtfp::mesh::Mesh;
use astroid_evolution::rtfp::ray_backend::{CpuReferenceBackend, RayBackend, RayBatch};
use astroid_evolution::rtfp::ray_segments::{segments_from_hits, split_segments_near_far};

#[test]
fn test_step1_unit_system() {
    let units = UnitSystem::new("km", "g/cm3").unwrap();
    // 长度: 1.5 km -> 1500 m
    assert_eq!(units.to_meters(1.5), 1500.0);
    // 密度: 1.2 g/cm3 -> 1200 kg/m3
    assert_eq!(units.to_si_density(1.2), 1200.0);

    // Cauchy 核转换:
    // center [0.1, 0.2, 0.3] km -> [100, 200, 300] m
    // sigma 2.0 km^-1 -> 0.002 m^-1
    // weight 1.5 g/cm3 -> 1500 kg/m3
    let (c, sig, w, a) = units.convert_cauchy_kernel([0.1, 0.2, 0.3], 2.0, 1.5, 1.0);
    assert_eq!(c, [100.0, 200.0, 300.0]);
    assert_eq!(sig, 0.002);
    assert_eq!(w, 1500.0);
    assert_eq!(a, 1.0);
}

#[test]
fn test_step2_point_cloud_io() {
    let units = UnitSystem::new("m", "kg/m3").unwrap();
    let pc_path = Path::new("assets/point_cloud/sample_asteroid.xyz");
    let pc = PointCloud::load_from_file(pc_path, &units).expect("点云加载失败");

    assert!(!pc.points.is_empty());
    assert!(pc.normals.is_some());

    let (min, max) = pc.bounding_box();
    assert!(max[0] > min[0]);
    assert!(max[1] > min[1]);
    assert!(max[2] > min[2]);

    let spacing = pc.estimate_average_spacing();
    assert!(spacing > 0.0 && spacing.is_finite());
}

#[test]
fn test_step3_surface_reconstruction() {
    let units = UnitSystem::new("m", "kg/m3").unwrap();
    let pc_path = Path::new("assets/point_cloud/sample_asteroid.xyz");
    let pc = PointCloud::load_from_file(pc_path, &units).unwrap();

    // 1. 测试球极网格投影重建 (Spherical Grid)
    let mesh_sphere = reconstruct_surface(
        &pc,
        &ReconstructionMethod::SphericalGrid { n_lat: 16, n_lon: 32 },
        true,
    ).expect("球极网格重建失败");
    assert!(mesh_sphere.vertex_count() > 0);
    assert!(mesh_sphere.face_count() > 0);

    // 2. 测试凸包表面重建 (Convex Hull)
    let mesh_hull = reconstruct_surface(
        &pc,
        &ReconstructionMethod::ConvexHull,
        true,
    ).expect("凸包表面重建失败");
    assert!(mesh_hull.vertex_count() >= 4);
    assert!(mesh_hull.face_count() >= 4);

    // 3. 跨模块数据契约互操作测试 (GeometryState)
    let geo_state = mesh_sphere.to_geometry_state();
    assert_eq!(geo_state.vertices.len(), mesh_sphere.vertex_count());
    assert_eq!(geo_state.faces.len(), mesh_sphere.face_count());
    let restored_mesh = Mesh::from_geometry_state(&geo_state);
    assert_eq!(restored_mesh.vertex_count(), mesh_sphere.vertex_count());
    assert_eq!(restored_mesh.face_count(), mesh_sphere.face_count());
}

#[test]
fn test_step4_geometry_validation() {
    let units = UnitSystem::new("m", "kg/m3").unwrap();
    let pc_path = Path::new("assets/point_cloud/sample_asteroid.xyz");
    let pc = PointCloud::load_from_file(pc_path, &units).unwrap();

    // 重建流形
    let mesh = reconstruct_surface(
        &pc,
        &ReconstructionMethod::SphericalGrid { n_lat: 20, n_lon: 40 },
        true,
    ).unwrap();

    // 执行封闭多面体严格验证
    let report = validate_closed_polyhedron(&mesh);
    println!("封闭多面体验证报告: {report:#?}");

    assert!(report.non_degenerate, "面片必须非退化");
    assert!(report.is_closed_manifold, "必须为封闭二维流形 (无边界开孔)");
    assert!(report.consistent_orientation, "半边定向必须严格一致");
    assert!(report.signed_volume > 0.0, "多面体体积必须严格为正");
    assert!(!report.has_self_intersection, "不应存在面片自相交");
    assert!(report.is_valid, "总体多面体验证必须通过");
}

#[test]
fn test_step5_cpu_bvh_and_ray_backend() {
    let units = UnitSystem::new("m", "kg/m3").unwrap();
    let pc_path = Path::new("assets/point_cloud/sample_asteroid.xyz");
    let pc = PointCloud::load_from_file(pc_path, &units).unwrap();
    let mesh = reconstruct_surface(
        &pc,
        &ReconstructionMethod::SphericalGrid { n_lat: 16, n_lon: 32 },
        true,
    ).unwrap();

    // 构建扁平 SoA BVH
    let bvh = build_flattened_bvh(&mesh);
    assert!(bvh.node_count() > 0);
    assert_eq!(bvh.triangle_indices.len(), mesh.face_count());

    // 内部/外部点判定
    let inside = is_point_inside(&bvh, &mesh, [0.0, 0.0, 0.0]);
    assert!(inside, "质心原点应该位于封闭多面体内部");

    let outside = is_point_inside(&bvh, &mesh, [2000.0, 2000.0, 2000.0]);
    assert!(!outside, "远场点应该位于多面体外部");

    // 单光线求交测试 (从外部射入穿透实体)
    let orig = [0.0, 0.0, 1000.0];
    let dir = [0.0, 0.0, -1.0];
    let hits = intersect_ray_bvh(&bvh, &mesh, orig, dir);
    assert!(hits.len() >= 2, "射线必须穿透实体产生至少两个交点");
    assert!(hits[0] < hits[1]);

    // 交段生成 (外部点 inside_at_zero = false)
    let segs_ext = segments_from_hits(&hits, false);
    assert!(!segs_ext.is_empty());
    assert_eq!(segs_ext[0].r_min, hits[0]);
    assert_eq!(segs_ext[0].r_max, hits[1]);

    // 近场/远场切锥切分测试
    let (near, far) = split_segments_near_far(&segs_ext, 700.0);
    assert!(!near.is_empty() || !far.is_empty());

    // RayBackend Trait 与 CpuReferenceBackend 批次测试
    let backend = CpuReferenceBackend::new(&bvh, &mesh);
    let batch = RayBatch::new(
        vec![[0.0, 0.0, 1000.0], [1000.0, 0.0, 0.0]],
        vec![[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0]],
    ).unwrap();

    let hit_buffer = backend.intersect_batch(&batch);
    assert_eq!(hit_buffer.hits_per_ray.len(), 2);
    assert!(hit_buffer.hits_per_ray[0].len() >= 2);
    assert!(hit_buffer.hits_per_ray[1].len() >= 2);
}

#[test]
fn test_step6_load_project_toml_pipeline() {
    let toml_path = Path::new("assets/projects/sample_project.toml");
    let loaded = load_project_toml(toml_path).expect("完整项目 TOML 流水线加载失败");

    assert_eq!(loaded.config.project.name, "ryugu_scf");
    assert_eq!(loaded.kernels.len(), 2);
    assert!(loaded.initial_mass > 0.0);
    assert!(loaded.validation_report.is_valid);
    println!("初始估计质量 M: {:.4e} kg", loaded.initial_mass);
}
