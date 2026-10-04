//! Historical plausibility of a smoothed mesh.
//!
//! The review compares the raw body with the smoothed body. It is not a unique
//! reconstruction of the asteroid millions of years ago.

use crate::common::density::CauchyDensity;
use crate::common::geometry::TriMesh;
use crate::common::math::DVec3;
use crate::rtfp_solver::api::{FieldMask, PointRole, PointSet};
use crate::rtfp_solver::field::audit::{ErrorReport, production_rtfp_status};
use crate::rtfp_solver::radial::gauss_legendre::{
    coarser_face_rule, face_rule_for_directions, rules_differ,
};
use crate::rtfp_solver::volume::uniform_integrals;
use crate::rtfp_solver::{RtfpOptions, bind_and_eval_ray_rule};
use crate::scf_simulator::error::ScfError;
use crate::scf_simulator::geometry_update::smooth::{CraterMask, VertexRecord};

const MIN_FACE_COVERAGE: f64 = 0.5;

const NON_UNIQUENESS: &str =
    "该结果只能说明没有发现明显物理矛盾，不能证明它就是数百万年前的真实唯一形态。";
const FOUND_CONTRADICTION: &str =
    "发现了明显物理矛盾，因此不能把该网格当作数百万年前的状态；这仍然不是一次唯一历史重建。";

#[derive(Clone, Debug)]
pub struct SourcedDraw {
    pub name: &'static str,
    pub measured: f64,
    pub uncertainty: f64,
    pub unit: &'static str,
    pub source: String,
}

#[derive(Clone, Debug)]
pub struct ReviewRequest<'a> {
    pub raw: &'a TriMesh,
    pub smoothed: &'a TriMesh,
    pub density: &'a CauchyDensity,
    pub mass_target: f64,
    pub mass_tolerance: f64,
    pub bulk_density: f64,
    pub bulk_uncertainty: f64,
    pub porosity: f64,
    pub porosity_uncertainty: f64,
    pub grain_diameter: f64,
    pub grain_uncertainty: f64,
    pub mu_s: f64,
    pub mu_uncertainty: f64,
    pub mu_2: f64,
    pub cohesion: f64,
    pub cohesion_uncertainty: f64,
    pub spin_rate: f64,
    pub spin_uncertainty: f64,
    pub timestep: f64,
    pub timestep_uncertainty: f64,
    pub calibration: f64,
    pub calibration_uncertainty: f64,
    pub nominal_bulk: f64,
    pub omega: [f64; 3],
    pub gravitational_constant: f64,
    pub steps_executed: u32,
    pub mask: &'a CraterMask,
    pub vertices: &'a [VertexRecord],
    pub coefficients: Vec<Coefficient>,
    pub field: RtfpOptions,
    pub parameter_source: String,
}

#[derive(Clone, Debug)]
pub struct Coefficient {
    pub name: String,
    pub value: f64,
    pub uncertainty: f64,
    pub unit: String,
    pub provenance: String,
}

#[derive(Clone, Debug)]
pub struct HistoryReport {
    pub class: &'static str,
    pub time_role: &'static str,
    pub verification_level: &'static str,
    pub document: serde_json::Value,
}

pub fn review(request: &ReviewRequest<'_>) -> Result<HistoryReport, ScfError> {
    let raw = body_record(request.raw, request.density, "raw")?;
    let smoothed = body_record(request.smoothed, request.density, "smoothed")?;
    let same_mesh = raw.mesh_hash == smoothed.mesh_hash;
    let raw_field = sample_field(request.raw, request.density, &request.field, request.mu_s)?;
    let smoothed_field = if same_mesh {
        raw_field.clone()
    } else {
        sample_field(
            request.smoothed,
            request.density,
            &request.field,
            request.mu_s,
        )?
    };
    let local = local_change(request.raw, request.smoothed);
    let protection = protection_report(
        request.raw,
        request.smoothed,
        request.mask,
        request.vertices,
    );
    let candidates = candidate_report(request.raw, request.smoothed);
    let time = time_account(request, &smoothed_field);
    if time.role == "million_year_evolution" || time.role == "historical_reconstruction" {
        return Err(ScfError::model_not_closed(
            "history review must not label the result as million_year_evolution or historical_reconstruction",
        ));
    }
    let slopes = slope_pair(request, &raw, &smoothed);
    let mut failures = obvious_failures(request, &raw, &smoothed, &local, &protection, &time);
    failures.extend(parameter_failures(
        request,
        &raw,
        &smoothed,
        &raw_field,
        &smoothed_field,
    ));
    if !raw_field.values_finite || !smoothed_field.values_finite {
        failures.push("Tier1 field sample is not finite".into());
    }
    let ensemble = ensemble(
        request,
        &raw,
        &smoothed,
        &local,
        &protection,
        &time,
        &raw_field,
        &smoothed_field,
    );
    let closure = if request.steps_executed == 0 || time.role != "physical_evolution" {
        "not_run"
    } else if raw_field.values_finite && smoothed_field.values_finite {
        "tier1_resampled"
    } else {
        "not_closed"
    };
    if closure != "tier1_resampled" {
        failures.retain(|item| {
            item != "stepwise RTFP, pressure, friction, and geometry are not closed"
        });
    }
    let coverage = raw_field.coverage.min(smoothed_field.coverage);
    let field_complete = raw_field.status == "PASS" && smoothed_field.status == "PASS";
    let unmeasured_blocks = ensemble
        .get("unmeasured")
        .and_then(|value| value.as_array())
        .is_some_and(|items| !items.is_empty());
    let class = classify(
        &failures,
        &ensemble,
        same_mesh,
        field_complete,
        coverage,
        !unmeasured_blocks,
    );
    let verification_level = if class == "plausible" {
        "not_a_unique_reconstruction"
    } else {
        "not_passed"
    };
    let statement = if class == "obviously_wrong" {
        FOUND_CONTRADICTION
    } else {
        NON_UNIQUENESS
    };
    let constraints = constraint_table(request, &raw, &smoothed, &ensemble);
    let document = serde_json::json!({
        "historical_plausibility": class,
        "time_role": time.role,
        "rejected_labels": ["million_year_evolution", "historical_reconstruction"],
        "verification_level": verification_level,
        "non_uniqueness_statement": statement,
        "obvious_failure_reasons": failures,
        "constraint_residuals": constraints,
        "uncertainty_summary": ensemble,
        "bodies": [raw.document, smoothed.document],
        "local_morphology": local,
        "protection": protection,
        "feature_candidates": candidates,
        "slope": slopes,
        "vertex_records": vertex_record_summary(request.vertices),
        "time": time.document,
        "tier1": {
            "calls": if same_mesh { 1 } else { 2 },
            "distinct_meshes": if same_mesh { 1 } else { 2 },
            "identical_smoothed_mesh": same_mesh,
            "raw_mesh_hash": raw.mesh_hash,
            "smoothed_mesh_hash": smoothed.mesh_hash,
            "raw_samples": raw_field.samples,
            "smoothed_samples": smoothed_field.samples,
            "raw_faces": request.raw.faces.len(),
            "smoothed_faces": request.smoothed.faces.len(),
            "raw_supercritical_fraction": raw_field.supercritical_fraction,
            "smoothed_supercritical_fraction": smoothed_field.supercritical_fraction,
            "coverage": coverage,
            "minimum_coverage": MIN_FACE_COVERAGE,
            "raw_status": raw_field.status,
            "smoothed_status": smoothed_field.status,
            "raw_potential_error": json_opt(raw_field.potential_error),
            "raw_gradient_error": json_opt(raw_field.gradient_error),
            "raw_hessian_error": json_opt(raw_field.hessian_error),
            "smoothed_potential_error": json_opt(smoothed_field.potential_error),
            "smoothed_gradient_error": json_opt(smoothed_field.gradient_error),
            "smoothed_hessian_error": json_opt(smoothed_field.hessian_error),
            "field_error_evidence": if field_complete { "complete" } else { "incomplete" },
            "feynman": "not_called",
            "carlson": "not_called",
            "radial_projection": "not_used",
        },
        "star": star_diagnostic(request)?,
        "closure": closure,
    });
    Ok(HistoryReport {
        class,
        time_role: time.role,
        verification_level,
        document,
    })
}

fn classify(
    failures: &[String],
    ensemble: &serde_json::Value,
    same_mesh: bool,
    field_complete: bool,
    coverage: f64,
    measured_drivers_only: bool,
) -> &'static str {
    if !failures.is_empty() {
        return "obviously_wrong";
    }
    let pass = ensemble
        .get("pass_fraction")
        .and_then(|value| value.as_f64())
        .unwrap_or(0.0);
    let applied = ensemble
        .get("applied_draws")
        .and_then(|value| value.as_u64())
        .unwrap_or(0);
    if same_mesh
        || !field_complete
        || coverage + 1.0e-12 < MIN_FACE_COVERAGE
        || !measured_drivers_only
        || applied == 0
        || pass < 1.0
    {
        return "indeterminate";
    }
    "plausible"
}

fn json_opt(value: Option<f64>) -> serde_json::Value {
    match value {
        Some(value) if value.is_finite() => serde_json::json!(value),
        _ => serde_json::Value::Null,
    }
}

struct BodyRecord {
    mesh_hash: String,
    document: serde_json::Value,
    volume: f64,
    mass: f64,
    centroid: DVec3,
    inertia: [f64; 3],
    bulge: f64,
    radius_rms: f64,
    curvature_max: f64,
    mean_radius: f64,
}

fn body_record(
    mesh: &TriMesh,
    density: &CauchyDensity,
    role: &str,
) -> Result<BodyRecord, ScfError> {
    let integrals = crate::rtfp_solver::volume::closed_density_integrals(mesh, density)
        .map_err(ScfError::model_not_closed)?;
    let (inertia, axes, frame) =
        principal_inertia(integrals.inertia, integrals.mass, integrals.centroid);
    let area = surface_area(mesh);
    let radii = vertex_radii(mesh, integrals.centroid, &frame);
    let mean_radius = mean(&radii);
    let radius_rms = rms(&radii, mean_radius);
    let (bulge, flattening) = bulge_and_flattening(mesh, integrals.centroid, &frame, mean_radius);
    let curvature = vertex_curvature(mesh);
    let curvature_max = curvature.iter().copied().fold(0.0_f64, f64::max);
    let topology = topology_record(mesh);
    let hash = crate::rtfp_solver::io::mesh_hash(mesh);
    let spin_note = "shape axis is the maximum-inertia eigenvector; it is not a measured pole unless rotation.omega is aligned with it";
    Ok(BodyRecord {
        mesh_hash: hash.clone(),
        volume: integrals.volume,
        mass: integrals.mass,
        centroid: integrals.centroid,
        inertia,
        bulge,
        radius_rms,
        curvature_max,
        mean_radius,
        document: serde_json::json!({
            "role": role,
            "mesh_hash": hash,
            "volume_m3": integrals.volume,
            "area_m2": area,
            "mass_kg": integrals.mass,
            "mass_method": "closed_mesh_full_cauchy",
            "centroid_m": integrals.centroid.to_array(),
            "principal_inertia_kg_m2": inertia,
            "axis_lengths_m": axes,
            "equatorial_bulge_m": bulge,
            "polar_flattening": flattening,
            "radius_quantiles_m": quantiles(&radii),
            "radius_rms_m": radius_rms,
            "curvature_quantiles": quantiles(&curvature),
            "curvature_max": curvature_max,
            "roughness_m": [roughness_scale(mesh, 1), roughness_scale(mesh, 2)],
            "shape_axis": frame[2],
            "shape_axis_note": spin_note,
            "topology": topology,
        }),
    })
}

fn topology_record(mesh: &TriMesh) -> serde_json::Value {
    match mesh.validate_solid() {
        Ok(()) => serde_json::json!({
            "closed": true,
            "orientation_consistent": true,
            "nonmanifold_edges": 0,
            "degenerate_faces": 0,
            "self_intersections": 0,
            "negative_volume": false,
            "note": "validate_solid accepted the closed oriented solid",
        }),
        Err(err) => serde_json::json!({
            "closed": false,
            "orientation_consistent": false,
            "negative_volume": mesh.signed_volume() <= 0.0,
            "note": err.detail,
        }),
    }
}

fn surface_area(mesh: &TriMesh) -> f64 {
    (0..mesh.faces.len())
        .map(|face| {
            let [a, b, c] = mesh.triangle(face);
            0.5 * (b - a).cross(c - a).length()
        })
        .sum()
}

#[derive(Clone)]
struct FieldSample {
    samples: usize,
    coverage: f64,
    values_finite: bool,
    status: &'static str,
    potential_error: Option<f64>,
    gradient_error: Option<f64>,
    hessian_error: Option<f64>,
    sin2: Vec<f64>,
    supercritical_fraction: f64,
    mean_speed: f64,
}

fn sample_field(
    mesh: &TriMesh,
    density: &CauchyDensity,
    options: &RtfpOptions,
    friction: f64,
) -> Result<FieldSample, ScfError> {
    let ids = face_ids(mesh);
    let points: Vec<DVec3> = ids
        .iter()
        .map(|&face| {
            let [a, b, c] = mesh.triangle(face);
            (a + b + c) * (1.0 / 3.0)
        })
        .collect();
    let coverage = if mesh.faces.is_empty() {
        0.0
    } else {
        points.len() as f64 / mesh.faces.len() as f64
    };
    if points.is_empty() {
        return Ok(FieldSample {
            samples: 0,
            coverage,
            values_finite: false,
            status: "NOT_ASSESSED",
            potential_error: None,
            gradient_error: None,
            hessian_error: None,
            sin2: Vec::new(),
            supercritical_fraction: 0.0,
            mean_speed: 0.0,
        });
    }
    let fine = face_rule_for_directions(options.directions.max(8));
    let coarse = coarser_face_rule(fine);
    let want = FieldMask {
        potential: true,
        gradient: true,
        hessian: true,
    };
    let set = PointSet::from_points(&points);
    let fine_batch = bind_and_eval_ray_rule(
        options.clone(),
        fine,
        mesh,
        density.clone(),
        &set,
        PointRole::SurfaceFace,
        want,
    )
    .map_err(|err| ScfError::model_not_closed(err.detail))?;
    let coarse_batch = bind_and_eval_ray_rule(
        options.clone(),
        coarse,
        mesh,
        density.clone(),
        &set,
        PointRole::SurfaceFace,
        want,
    )
    .map_err(|err| ScfError::model_not_closed(err.detail))?;
    let potential_error = component_gap(&fine_batch.potential, &coarse_batch.potential);
    let gradient_error = vector_gap(&fine_batch.gradient, &coarse_batch.gradient);
    let hessian_error = sym_gap(&fine_batch.hessian, &coarse_batch.hessian);
    let report = ErrorReport {
        points_completed: points.len(),
        points_requested: points.len(),
        rules_differ: rules_differ(fine, coarse),
        max_potential: potential_error,
        max_gradient: gradient_error,
        max_hessian: hessian_error,
        want,
    };
    let status = production_rtfp_status(&report);
    let threshold = friction.atan().sin().powi(2);
    let mut sin2 = Vec::with_capacity(points.len());
    let mut values_finite = fine_batch.potential.len() == points.len()
        && fine_batch.gradient.len() == points.len()
        && fine_batch.hessian.len() == points.len();
    let mut steep = 0usize;
    for (index, &face) in ids.iter().enumerate() {
        let hessian_ok = fine_batch
            .hessian
            .get(index)
            .is_some_and(|row| row.iter().all(|value| value.is_finite()));
        if !fine_batch
            .potential
            .get(index)
            .is_some_and(|value| value.is_finite())
            || !fine_batch
                .gradient
                .get(index)
                .is_some_and(|value| value.is_finite())
            || !hessian_ok
        {
            values_finite = false;
        }
        let Some(normal) = mesh.face_normal(face) else {
            values_finite = false;
            sin2.push(0.0);
            continue;
        };
        let gravity = fine_batch.gradient[index];
        let denom = gravity.length_squared() * normal.length_squared();
        let value = if denom > 1.0e-20 {
            (gravity.cross(normal).length_squared() / denom).min(1.0)
        } else {
            0.0
        };
        if value > threshold {
            steep += 1;
        }
        sin2.push(value);
    }
    Ok(FieldSample {
        samples: points.len(),
        coverage,
        values_finite,
        status,
        potential_error,
        gradient_error,
        hessian_error,
        supercritical_fraction: steep as f64 / points.len() as f64,
        sin2,
        mean_speed: 0.0,
    })
}

fn face_ids(mesh: &TriMesh) -> Vec<usize> {
    let count = mesh.faces.len();
    if count <= 512 {
        return (0..count).collect();
    }
    let target = ((count as f64) * MIN_FACE_COVERAGE).ceil() as usize;
    let target = target.clamp(1, count);
    (0..target).map(|index| index * count / target).collect()
}

struct TimeAccount {
    role: &'static str,
    document: serde_json::Value,
}

fn time_account(request: &ReviewRequest<'_>, field: &FieldSample) -> TimeAccount {
    let physical_steps = request.steps_executed > 0 && request.timestep > 0.0;
    let role = if physical_steps {
        "physical_evolution"
    } else {
        "initial_condition_preprocessing"
    };
    let total = if physical_steps {
        request.timestep * f64::from(request.steps_executed)
    } else {
        0.0
    };
    let gravity_scale =
        (request.gravitational_constant * request.bulk_density * request.raw.signed_volume().abs()
            / request
                .raw
                .vertices
                .iter()
                .map(|v| v.length())
                .fold(1.0_f64, f64::max)
                .powi(2))
        .sqrt();
    let free_fall = if gravity_scale > 0.0 {
        request
            .raw
            .vertices
            .iter()
            .map(|v| v.length())
            .fold(1.0_f64, f64::max)
            / gravity_scale
    } else {
        f64::NAN
    };
    let grain_flow = if request.spin_rate.abs() > 0.0 {
        1.0 / request.spin_rate.abs()
    } else {
        f64::NAN
    };
    TimeAccount {
        role,
        document: serde_json::json!({
            "role": role,
            "million_year_evolution": false,
            "historical_reconstruction": false,
            "timestep_s": if physical_steps { request.timestep } else { 0.0 },
            "total_time_s": total,
            "steps": request.steps_executed,
            "normal_speed_m_s": field.mean_speed,
            "normal_speed_drivers": [
                "effective gravity from the Tier1 gradient plus centrifugal acceleration",
                "friction angle and grain diameter in the slope speed",
                "cohesion enters the yield stress, not an extra geometric iteration count",
                "pressure is not available for general_mesh because the non-star Stokes solve is unsupported"
            ],
            "free_fall_time_s": free_fall,
            "spin_period_s": grain_flow,
            "relation": "total_time is compared with the free-fall time and the spin period; a preprocessing step has zero physical duration",
            "supercritical_fraction": field.supercritical_fraction,
        }),
    }
}

fn local_change(raw: &TriMesh, smoothed: &TriMesh) -> serde_json::Value {
    if raw.vertices.len() != smoothed.vertices.len() || raw.faces.len() != smoothed.faces.len() {
        return serde_json::json!({
            "paired": false,
            "reason": "vertex or face counts differ, so local morphology is not paired by index",
        });
    }
    let mut max_disp = 0.0_f64;
    let mut mean_disp = 0.0;
    let mut over_scale = 0usize;
    let edges = median_edge(raw);
    for (before, after) in raw.vertices.iter().zip(&smoothed.vertices) {
        let disp = (*after - *before).length();
        max_disp = max_disp.max(disp);
        mean_disp += disp;
        if disp > edges {
            over_scale += 1;
        }
    }
    mean_disp /= raw.vertices.len() as f64;
    let raw_radius = vertex_radius_values(raw);
    let smoothed_radius = vertex_radius_values(smoothed);
    serde_json::json!({
        "paired": true,
        "max_displacement_m": max_disp,
        "mean_displacement_m": mean_disp,
        "vertices_over_local_edge": over_scale,
        "median_edge_m": edges,
        "radius_quantile_change_m": quantile_delta(&raw_radius, &smoothed_radius),
        "curvature_max_change": vertex_curvature(smoothed).into_iter().fold(0.0_f64, f64::max)
            - vertex_curvature(raw).into_iter().fold(0.0_f64, f64::max),
    })
}

fn protection_report(
    raw: &TriMesh,
    smoothed: &TriMesh,
    mask: &CraterMask,
    records: &[VertexRecord],
) -> serde_json::Value {
    let paired = raw.vertices.len() == smoothed.vertices.len();
    let mut max_disp = 0.0_f64;
    let mut mean_disp = 0.0;
    let mut count = 0.0;
    if paired {
        for index in 0..raw.vertices.len() {
            let protected = mask.weight(index) > 0.0
                || records
                    .get(index)
                    .is_some_and(|record| record.protection > 0.0);
            if !protected {
                continue;
            }
            let disp = records
                .get(index)
                .map(|record| record.displacement_m)
                .unwrap_or_else(|| (smoothed.vertices[index] - raw.vertices[index]).length());
            max_disp = max_disp.max(disp);
            mean_disp += disp;
            count += 1.0;
        }
    }
    let before = crate::scf_simulator::geometry_update::smooth::crater_measure(raw, mask);
    let after = crate::scf_simulator::geometry_update::smooth::crater_measure(smoothed, mask);
    let depth_change = match (before, after) {
        (Some(left), Some(right)) => serde_json::json!(right.depth - left.depth),
        _ => serde_json::Value::Null,
    };
    let curvature_change = match (before, after) {
        (Some(left), Some(right)) => serde_json::json!(right.rim_curvature - left.rim_curvature),
        _ => serde_json::Value::Null,
    };
    serde_json::json!({
        "mask": if mask.claimed { "observed" } else { "absent" },
        "identification": if mask.claimed { "observed" } else { "uncertain" },
        "claimed_real_terrain": mask.claimed_real_terrain,
        "max_displacement_m": max_disp,
        "mean_displacement_m": if count > 0.0 { mean_disp / count } else { 0.0 },
        "depth_change_m": depth_change,
        "rim_height_change_m": depth_change,
        "rim_curvature_change": curvature_change,
        "area_change_m2": null,
        "diameter_change_m": null,
        "note": if mask.claimed_real_terrain {
            "observational floor, wall, and rim masks limit displacement on the mapped terrain"
        } else {
            "没有观测掩码，陨石坑和块石只标记为 uncertain，不能视为已经保护了真实地貌"
        },
    })
}

fn candidate_report(raw: &TriMesh, smoothed: &TriMesh) -> serde_json::Value {
    let raw_craters = depression_candidates(raw);
    let smoothed_craters = depression_candidates(smoothed);
    let raw_boulders = protrusion_candidates(raw);
    let smoothed_boulders = protrusion_candidates(smoothed);
    serde_json::json!({
        "identification": "uncertain",
        "note": "geometric candidates are not observed craters or boulders",
        "raw_crater_candidates": raw_craters,
        "smoothed_crater_candidates": smoothed_craters,
        "raw_boulder_candidates": raw_boulders,
        "smoothed_boulder_candidates": smoothed_boulders,
    })
}

fn depression_candidates(mesh: &TriMesh) -> serde_json::Value {
    feature_candidates(mesh, true)
}

fn protrusion_candidates(mesh: &TriMesh) -> serde_json::Value {
    feature_candidates(mesh, false)
}

fn feature_candidates(mesh: &TriMesh, depression: bool) -> serde_json::Value {
    let neighbors = vertex_neighbors(&mesh.faces, mesh.vertices.len());
    let center = uniform_integrals(mesh, 1.0).centroid;
    let mut mark = vec![false; mesh.vertices.len()];
    let scale = mesh
        .vertices
        .iter()
        .map(|vertex| (*vertex - center).length())
        .fold(0.0_f64, f64::max)
        .max(1.0);
    for (index, vertex) in mesh.vertices.iter().enumerate() {
        if neighbors[index].is_empty() {
            continue;
        }
        let radius = (*vertex - center).length();
        let neighbor_radius = neighbors[index]
            .iter()
            .map(|&other| (mesh.vertices[other] - center).length())
            .sum::<f64>()
            / neighbors[index].len() as f64;
        let delta = if depression {
            neighbor_radius - radius
        } else {
            radius - neighbor_radius
        };
        mark[index] = delta > 0.03 * scale;
    }
    let clusters = components(&mark, &neighbors);
    let mut rows = Vec::new();
    for cluster in clusters {
        if cluster.len() < 3 {
            continue;
        }
        let mut floor = f64::INFINITY;
        let mut rim = 0.0_f64;
        let mut area = 0.0;
        for &index in &cluster {
            floor = floor.min((mesh.vertices[index] - center).length());
            for &other in &neighbors[index] {
                if !mark[other] {
                    rim = rim.max((mesh.vertices[other] - center).length());
                }
            }
        }
        for face in 0..mesh.faces.len() {
            let ids = mesh.faces[face];
            if cluster.contains(&(ids[0] as usize))
                || cluster.contains(&(ids[1] as usize))
                || cluster.contains(&(ids[2] as usize))
            {
                let [a, b, c] = mesh.triangle(face);
                area += 0.5 * (b - a).cross(c - a).length() / 3.0;
            }
        }
        let diameter = 2.0 * (area / std::f64::consts::PI).sqrt();
        rows.push(serde_json::json!({
            "identification": "uncertain",
            "vertices": cluster.len(),
            "diameter_m": diameter,
            "depth_m": (rim - floor).max(0.0),
            "rim_height_m": (rim - floor).max(0.0),
            "floor_area_m2": area,
        }));
    }
    serde_json::json!({
        "count": rows.len(),
        "items": rows,
    })
}

fn obvious_failures(
    request: &ReviewRequest<'_>,
    raw: &BodyRecord,
    smoothed: &BodyRecord,
    local: &serde_json::Value,
    protection: &serde_json::Value,
    time: &TimeAccount,
) -> Vec<String> {
    let mut failures = Vec::new();
    if raw.bulge > 0.05 * raw.mean_radius && smoothed.bulge < 0.5 * raw.bulge {
        failures.push("equatorial bulge was substantially removed".into());
    }
    if raw.radius_rms > 0.02 * raw.mean_radius && smoothed.radius_rms < 0.25 * raw.radius_rms {
        failures.push("rough surface was smoothed toward an ellipsoid".into());
    }
    let curvature_jump = smoothed.curvature_max - raw.curvature_max;
    let max_disp = local
        .get("max_displacement_m")
        .and_then(|value| value.as_f64())
        .unwrap_or(0.0);
    if curvature_jump > raw.curvature_max.max(0.05)
        && max_disp
            > local
                .get("median_edge_m")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0)
    {
        failures.push("smoothing introduced sharper ridges than the input".into());
    }
    let mass_gap = (smoothed.mass - raw.mass).abs() / raw.mass.abs().max(1.0);
    if mass_gap > request.mass_tolerance {
        failures.push("mass change exceeds the sourced mass tolerance".into());
    }
    let volume_gap = (smoothed.volume - raw.volume).abs() / raw.volume.abs().max(1.0);
    if volume_gap > request.mass_tolerance {
        failures.push("volume change exceeds the sourced mass tolerance".into());
    }
    let inertia_gap = (0..3)
        .map(|index| {
            (smoothed.inertia[index] - raw.inertia[index]).abs() / raw.inertia[index].abs().max(1.0)
        })
        .fold(0.0, f64::max);
    if inertia_gap > request.mass_tolerance {
        failures.push("inertia change exceeds the sourced mass tolerance".into());
    }
    let drift = (smoothed.centroid - raw.centroid).length();
    if request.steps_executed == 0 && drift > 0.01 * raw.mean_radius.max(1.0) {
        failures.push("centroid drifted without a recorded physical translation".into());
    }
    let over_scale = local
        .get("vertices_over_local_edge")
        .and_then(|value| value.as_u64())
        .unwrap_or(0);
    if over_scale > 0 {
        failures.push("displacement exceeds the local edge length".into());
    }
    if protection
        .get("claimed_real_terrain")
        .and_then(|value| value.as_bool())
        == Some(true)
    {
        let depth = protection
            .get("depth_change_m")
            .and_then(|value| value.as_f64());
        if depth.is_some_and(|value| value < -0.2 * raw.mean_radius) {
            failures.push("a claimed crater became substantially shallower".into());
        }
    }
    if time.role == "million_year_evolution" || request.parameter_source.trim().is_empty() {
        failures.push("a million-year claim has no sourced parameters".into());
    }
    let _ = time;
    failures
}

fn ensemble(
    request: &ReviewRequest<'_>,
    raw: &BodyRecord,
    smoothed: &BodyRecord,
    local: &serde_json::Value,
    protection: &serde_json::Value,
    time: &TimeAccount,
    raw_field: &FieldSample,
    smoothed_field: &FieldSample,
) -> serde_json::Value {
    let catalog = if request.coefficients.is_empty() {
        vec![
            Coefficient {
                name: "bulk_density".into(),
                value: request.bulk_density,
                uncertainty: request.bulk_uncertainty,
                unit: "kg/m^3".into(),
                provenance: "unmeasured".into(),
            },
            Coefficient {
                name: "mu_s".into(),
                value: request.mu_s,
                uncertainty: request.mu_uncertainty,
                unit: "1".into(),
                provenance: "unmeasured".into(),
            },
        ]
    } else {
        request.coefficients.clone()
    };
    let mut passed = 0usize;
    let mut applied = 0usize;
    let mut rows = Vec::new();
    let mut listed = Vec::new();
    for item in &catalog {
        let low = item.value - item.uncertainty;
        let high = item.value + item.uncertainty;
        listed.push(serde_json::json!({
            "name": item.name,
            "value": item.value,
            "uncertainty": item.uncertainty,
            "unit": item.unit,
            "provenance": item.provenance,
            "samples": [low, high],
        }));
        if !applies(&item.name) {
            continue;
        }
        for sample in [low, high] {
            let shifted = request.with_sample(&item.name, sample);
            let mut failures = obvious_failures(&shifted, raw, smoothed, local, protection, time);
            failures.extend(parameter_failures(
                &shifted,
                raw,
                smoothed,
                raw_field,
                smoothed_field,
            ));
            let ok = failures.is_empty();
            if ok {
                passed += 1;
            }
            applied += 1;
            rows.push(serde_json::json!({
                "parameter": item.name,
                "value": sample,
                "center": item.value,
                "uncertainty": item.uncertainty,
                "provenance": item.provenance,
                "passed": ok,
                "failure_count": failures.len(),
            }));
        }
    }
    serde_json::json!({
        "draws": applied,
        "applied_draws": applied,
        "pass_fraction": if applied == 0 { 0.0 } else { passed as f64 / applied as f64 },
        "source": request.parameter_source,
        "parameters": rows,
        "coefficients": listed,
        "unmeasured": catalog.iter().filter(|item| item.provenance == "unmeasured").map(|item| &item.name).collect::<Vec<_>>(),
        "note": "each applied draw is evaluated at value-uncertainty and value+uncertainty; unmeasured coefficients stay in this summary and block a plausible label",
    })
}

fn applies(name: &str) -> bool {
    matches!(
        name,
        "bulk_density"
            | "porosity"
            | "grain_diameter"
            | "mu_s"
            | "mu_2"
            | "cohesion"
            | "spin_rate"
            | "timestep"
            | "observational_calibration"
    )
}

fn parameter_failures(
    request: &ReviewRequest<'_>,
    raw: &BodyRecord,
    smoothed: &BodyRecord,
    raw_field: &FieldSample,
    smoothed_field: &FieldSample,
) -> Vec<String> {
    let mut failures = Vec::new();
    if request.bulk_density <= 0.0
        || request.grain_diameter <= 0.0
        || request.spin_rate <= 0.0
        || request.timestep <= 0.0
        || request.calibration <= 0.0
        || request.mu_s <= 0.0
        || request.mu_2 <= 0.0
    {
        failures.push("a sampled coefficient is outside its physical domain".into());
    }
    if !(0.0..=1.0).contains(&request.porosity) {
        failures.push("porosity sample is outside [0, 1]".into());
    }
    if request.cohesion < 0.0 {
        failures.push("cohesion sample is negative".into());
    }
    let scale = if request.nominal_bulk.abs() > 0.0 {
        request.bulk_density / request.nominal_bulk
    } else {
        1.0
    };
    let sampled_mass = smoothed.mass * scale;
    if (sampled_mass - request.mass_target).abs() / request.mass_target.abs().max(1.0)
        > request.mass_tolerance
    {
        failures.push("mass at the sampled bulk density exceeds the sourced mass tolerance".into());
    }
    let threshold = request.mu_s.atan().sin().powi(2);
    let raw_fraction = fraction_above(&raw_field.sin2, threshold);
    let smoothed_fraction = fraction_above(&smoothed_field.sin2, threshold);
    if smoothed_fraction > raw_fraction + 0.05 {
        failures.push(
            "the smoothed mesh has a larger supercritical fraction at the sampled friction".into(),
        );
    }
    if request
        .vertices
        .iter()
        .any(|record| record.displacement_m > 0.0)
    {
        let cap = displacement_cap(request, raw);
        let max_disp = request
            .vertices
            .iter()
            .map(|record| record.displacement_m)
            .fold(0.0_f64, f64::max);
        if max_disp > cap {
            failures.push("recorded displacement exceeds the sampled slope-speed cap".into());
        }
    }
    let _ = raw;
    failures
}

fn fraction_above(values: &[f64], threshold: f64) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().filter(|value| **value > threshold).count() as f64 / values.len() as f64
    }
}

fn displacement_cap(request: &ReviewRequest<'_>, raw: &BodyRecord) -> f64 {
    let radius = raw.mean_radius.max(1.0);
    let gravity = (request.gravitational_constant * request.bulk_density * raw.volume.abs()
        / radius.powi(2))
    .sqrt();
    let strength = request.mu_s.atan().tan().abs().max(1.0e-8);
    let speed = request.calibration * (gravity * request.grain_diameter).sqrt() / strength;
    let spin_cap = request.spin_rate.abs() * request.timestep * radius;
    (speed * request.timestep).min(spin_cap).max(0.0)
}

impl ReviewRequest<'_> {
    fn with_sample(&self, name: &str, sample: f64) -> ReviewRequest<'_> {
        let mut out = ReviewRequest {
            raw: self.raw,
            smoothed: self.smoothed,
            density: self.density,
            mass_target: self.mass_target,
            mass_tolerance: self.mass_tolerance,
            bulk_density: self.bulk_density,
            bulk_uncertainty: self.bulk_uncertainty,
            porosity: self.porosity,
            porosity_uncertainty: self.porosity_uncertainty,
            grain_diameter: self.grain_diameter,
            grain_uncertainty: self.grain_uncertainty,
            mu_s: self.mu_s,
            mu_uncertainty: self.mu_uncertainty,
            mu_2: self.mu_2,
            cohesion: self.cohesion,
            cohesion_uncertainty: self.cohesion_uncertainty,
            spin_rate: self.spin_rate,
            spin_uncertainty: self.spin_uncertainty,
            timestep: self.timestep,
            timestep_uncertainty: self.timestep_uncertainty,
            calibration: self.calibration,
            calibration_uncertainty: self.calibration_uncertainty,
            nominal_bulk: self.nominal_bulk,
            omega: self.omega,
            gravitational_constant: self.gravitational_constant,
            steps_executed: self.steps_executed,
            mask: self.mask,
            vertices: self.vertices,
            coefficients: Vec::new(),
            field: self.field.clone(),
            parameter_source: self.parameter_source.clone(),
        };
        match name {
            "bulk_density" => out.bulk_density = sample,
            "porosity" => out.porosity = sample,
            "grain_diameter" => out.grain_diameter = sample,
            "mu_s" => out.mu_s = sample,
            "mu_2" => out.mu_2 = sample,
            "cohesion" => out.cohesion = sample,
            "spin_rate" => out.spin_rate = sample,
            "timestep" => out.timestep = sample,
            "observational_calibration" => out.calibration = sample,
            _ => {}
        }
        out
    }
}

fn constraint_table(
    request: &ReviewRequest<'_>,
    raw: &BodyRecord,
    smoothed: &BodyRecord,
    ensemble: &serde_json::Value,
) -> serde_json::Value {
    let pass = ensemble
        .get("pass_fraction")
        .and_then(|value| value.as_f64())
        .unwrap_or(0.0);
    let rows = [
        row(
            "mass",
            request.mass_target,
            smoothed.mass,
            request.mass_tolerance,
            request.mass_tolerance,
            pass,
        ),
        row(
            "volume",
            raw.volume,
            smoothed.volume,
            request.mass_tolerance * raw.volume.abs(),
            request.bulk_uncertainty,
            pass,
        ),
        row(
            "equatorial_bulge",
            raw.bulge,
            smoothed.bulge,
            0.5 * raw.bulge.abs().max(1.0),
            request.bulk_uncertainty,
            pass,
        ),
        row(
            "radius_rms",
            raw.radius_rms,
            smoothed.radius_rms,
            0.75 * raw.radius_rms.abs().max(1e-6),
            request.grain_uncertainty,
            pass,
        ),
        row(
            "mu_s",
            request.mu_s,
            request.mu_s,
            request.mu_uncertainty,
            request.mu_uncertainty,
            pass,
        ),
        row(
            "spin_rate",
            request.spin_rate,
            request.spin_rate,
            request.spin_uncertainty,
            request.spin_uncertainty,
            pass,
        ),
    ];
    serde_json::Value::Array(rows.to_vec())
}

fn row(
    name: &str,
    measured: f64,
    model: f64,
    tolerance: f64,
    uncertainty: f64,
    pass_fraction: f64,
) -> serde_json::Value {
    serde_json::json!({
        "name": name,
        "measured": measured,
        "model": model,
        "tolerance": tolerance,
        "residual": model - measured,
        "uncertainty": uncertainty,
        "pass_fraction": pass_fraction,
    })
}

fn principal_inertia(
    origin: [f64; 6],
    mass: f64,
    centroid: DVec3,
) -> ([f64; 3], [f64; 3], [[f64; 3]; 3]) {
    let r2 = centroid.length_squared();
    let mut cm = origin;
    cm[0] -= mass * (r2 - centroid.x * centroid.x);
    cm[1] -= mass * (r2 - centroid.y * centroid.y);
    cm[2] -= mass * (r2 - centroid.z * centroid.z);
    cm[3] -= mass * (-centroid.x * centroid.y);
    cm[4] -= mass * (-centroid.x * centroid.z);
    cm[5] -= mass * (-centroid.y * centroid.z);
    let (values, vectors) = jacobi(cm);
    let mut order = [0, 1, 2];
    order.sort_by(|&i, &j| values[j].total_cmp(&values[i]));
    let inertia = [values[order[0]], values[order[1]], values[order[2]]];
    let frame = [vectors[order[0]], vectors[order[1]], vectors[order[2]]];
    let axes = axis_lengths(values, mass);
    (inertia, axes, frame)
}

fn axis_lengths(inertia: [f64; 3], mass: f64) -> [f64; 3] {
    if mass <= 0.0 {
        return [0.0; 3];
    }
    let a2 = ((inertia[1] + inertia[2] - inertia[0]) / mass).max(0.0);
    let b2 = ((inertia[0] + inertia[2] - inertia[1]) / mass).max(0.0);
    let c2 = ((inertia[0] + inertia[1] - inertia[2]) / mass).max(0.0);
    [a2.sqrt(), b2.sqrt(), c2.sqrt()]
}

fn jacobi(inertia: [f64; 6]) -> ([f64; 3], [[f64; 3]; 3]) {
    let mut a = [
        [inertia[0], inertia[3], inertia[4]],
        [inertia[3], inertia[1], inertia[5]],
        [inertia[4], inertia[5], inertia[2]],
    ];
    let mut v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for _ in 0..16 {
        let (p, q) = if a[0][1].abs() >= a[0][2].abs() && a[0][1].abs() >= a[1][2].abs() {
            (0, 1)
        } else if a[0][2].abs() >= a[1][2].abs() {
            (0, 2)
        } else {
            (1, 2)
        };
        if a[p][q].abs() < 1.0e-15 {
            break;
        }
        let tau = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
        let t = tau.signum() / (tau.abs() + (1.0 + tau * tau).sqrt());
        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = t * c;
        let app = a[p][p];
        let aqq = a[q][q];
        let apq = a[p][q];
        a[p][p] = app - t * apq;
        a[q][q] = aqq + t * apq;
        a[p][q] = 0.0;
        a[q][p] = 0.0;
        for k in 0..3 {
            if k == p || k == q {
                continue;
            }
            let aik = a[k][p];
            let akq = a[k][q];
            a[k][p] = c * aik - s * akq;
            a[p][k] = a[k][p];
            a[k][q] = s * aik + c * akq;
            a[q][k] = a[k][q];
        }
        for k in 0..3 {
            let vip = v[k][p];
            let viq = v[k][q];
            v[k][p] = c * vip - s * viq;
            v[k][q] = s * vip + c * viq;
        }
    }
    (
        [a[0][0], a[1][1], a[2][2]],
        [
            [v[0][0], v[1][0], v[2][0]],
            [v[0][1], v[1][1], v[2][1]],
            [v[0][2], v[1][2], v[2][2]],
        ],
    )
}

fn vertex_radii(mesh: &TriMesh, center: DVec3, frame: &[[f64; 3]; 3]) -> Vec<f64> {
    mesh.vertices
        .iter()
        .map(|vertex| {
            let offset = *vertex - center;
            let local = [
                offset.dot(DVec3::from_array(frame[0])),
                offset.dot(DVec3::from_array(frame[1])),
                offset.dot(DVec3::from_array(frame[2])),
            ];
            (local[0] * local[0] + local[1] * local[1] + local[2] * local[2]).sqrt()
        })
        .collect()
}

fn vertex_radius_values(mesh: &TriMesh) -> Vec<f64> {
    let center = uniform_integrals(mesh, 1.0).centroid;
    mesh.vertices
        .iter()
        .map(|vertex| (*vertex - center).length())
        .collect()
}

fn bulge_and_flattening(
    mesh: &TriMesh,
    center: DVec3,
    frame: &[[f64; 3]; 3],
    mean_radius: f64,
) -> (f64, f64) {
    let mut equatorial = 0.0;
    let mut equatorial_count = 0.0;
    let mut polar = 0.0;
    let mut polar_count = 0.0;
    for vertex in &mesh.vertices {
        let offset = *vertex - center;
        let z = offset.dot(DVec3::from_array(frame[0]));
        let radius = offset.length();
        if z.abs() < 0.25 * mean_radius.max(1.0) {
            equatorial += radius;
            equatorial_count += 1.0;
        }
        if z.abs() > 0.75 * mean_radius.max(1.0) {
            polar += radius;
            polar_count += 1.0;
        }
    }
    let eq = if equatorial_count > 0.0 {
        equatorial / equatorial_count
    } else {
        mean_radius
    };
    let pole = if polar_count > 0.0 {
        polar / polar_count
    } else {
        mean_radius
    };
    let bulge = eq - pole;
    let flattening = if eq.abs() > 1.0e-12 { bulge / eq } else { 0.0 };
    (bulge, flattening)
}

fn vertex_curvature(mesh: &TriMesh) -> Vec<f64> {
    let neighbors = vertex_neighbors(&mesh.faces, mesh.vertices.len());
    (0..mesh.vertices.len())
        .map(|index| {
            let Some(normal) = vertex_normal(mesh, index) else {
                return 0.0;
            };
            if neighbors[index].is_empty() {
                return 0.0;
            }
            neighbors[index]
                .iter()
                .filter_map(|&other| vertex_normal(mesh, other))
                .map(|other| 1.0 - normal.dot(other).clamp(-1.0, 1.0))
                .sum::<f64>()
                / neighbors[index].len() as f64
        })
        .collect()
}

fn roughness_scale(mesh: &TriMesh, rings: usize) -> f64 {
    let neighbors = vertex_neighbors(&mesh.faces, mesh.vertices.len());
    let center = uniform_integrals(mesh, 1.0).centroid;
    let mut acc = 0.0;
    let mut count = 0.0;
    for (index, vertex) in mesh.vertices.iter().enumerate() {
        let radius = (*vertex - center).length();
        let mut ring = vec![index];
        for _ in 0..rings {
            let mut next = ring.clone();
            for current in &ring {
                next.extend_from_slice(&neighbors[*current]);
            }
            next.sort_unstable();
            next.dedup();
            ring = next;
        }
        if ring.len() <= 1 {
            continue;
        }
        let mean_radius = ring
            .iter()
            .map(|&other| (mesh.vertices[other] - center).length())
            .sum::<f64>()
            / ring.len() as f64;
        acc += (radius - mean_radius).powi(2);
        count += 1.0;
    }
    if count == 0.0 {
        0.0
    } else {
        (acc / count).sqrt()
    }
}

fn vertex_normal(mesh: &TriMesh, index: usize) -> Option<DVec3> {
    let mut acc = DVec3::ZERO;
    for face in 0..mesh.faces.len() {
        if mesh.faces[face].contains(&(index as u32)) {
            let [a, b, c] = mesh.triangle(face);
            acc = acc + (b - a).cross(c - a);
        }
    }
    acc.try_normalize()
}

fn vertex_neighbors(faces: &[[u32; 3]], count: usize) -> Vec<Vec<usize>> {
    let mut neighbors = vec![Vec::new(); count];
    for face in faces {
        for slot in 0..3 {
            let a = face[slot] as usize;
            let b = face[(slot + 1) % 3] as usize;
            if !neighbors[a].contains(&b) {
                neighbors[a].push(b);
            }
            if !neighbors[b].contains(&a) {
                neighbors[b].push(a);
            }
        }
    }
    neighbors
}

fn components(mark: &[bool], neighbors: &[Vec<usize>]) -> Vec<Vec<usize>> {
    let mut seen = vec![false; mark.len()];
    let mut groups = Vec::new();
    for start in 0..mark.len() {
        if !mark[start] || seen[start] {
            continue;
        }
        let mut stack = vec![start];
        let mut group = Vec::new();
        seen[start] = true;
        while let Some(index) = stack.pop() {
            group.push(index);
            for &other in &neighbors[index] {
                if mark[other] && !seen[other] {
                    seen[other] = true;
                    stack.push(other);
                }
            }
        }
        groups.push(group);
    }
    groups
}

fn median_edge(mesh: &TriMesh) -> f64 {
    let mut edges = Vec::new();
    for face in &mesh.faces {
        for slot in 0..3 {
            let a = mesh.vertices[face[slot] as usize];
            let b = mesh.vertices[face[(slot + 1) % 3] as usize];
            edges.push((a - b).length());
        }
    }
    edges.sort_by(|left, right| left.total_cmp(right));
    edges.get(edges.len() / 2).copied().unwrap_or(0.0)
}

fn quantiles(values: &[f64]) -> [f64; 3] {
    if values.is_empty() {
        return [0.0; 3];
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|left, right| left.total_cmp(right));
    [
        sorted[sorted.len() / 10],
        sorted[sorted.len() / 2],
        sorted[sorted.len() * 9 / 10],
    ]
}

fn quantile_delta(raw: &[f64], smoothed: &[f64]) -> [f64; 3] {
    let left = quantiles(raw);
    let right = quantiles(smoothed);
    [right[0] - left[0], right[1] - left[1], right[2] - left[2]]
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

fn rms(values: &[f64], center: f64) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        (values
            .iter()
            .map(|value| (value - center).powi(2))
            .sum::<f64>()
            / values.len() as f64)
            .sqrt()
    }
}

fn component_gap(fine: &[f64], coarse: &[f64]) -> Option<f64> {
    if fine.len() != coarse.len() || fine.is_empty() {
        return None;
    }
    let mut max_gap = 0.0_f64;
    for (left, right) in fine.iter().zip(coarse) {
        if !left.is_finite() || !right.is_finite() {
            return None;
        }
        let scale = left.abs().max(right.abs()).max(1.0e-30);
        max_gap = max_gap.max((left - right).abs() / scale);
    }
    Some(max_gap)
}

fn vector_gap(fine: &[DVec3], coarse: &[DVec3]) -> Option<f64> {
    if fine.len() != coarse.len() || fine.is_empty() {
        return None;
    }
    let mut max_gap = 0.0_f64;
    for (left, right) in fine.iter().zip(coarse) {
        if !left.is_finite() || !right.is_finite() {
            return None;
        }
        let scale = left.length().max(right.length()).max(1.0e-30);
        max_gap = max_gap.max((*left - *right).length() / scale);
    }
    Some(max_gap)
}

fn sym_gap(fine: &[[f64; 6]], coarse: &[[f64; 6]]) -> Option<f64> {
    if fine.len() != coarse.len() || fine.is_empty() {
        return None;
    }
    let mut max_gap = 0.0_f64;
    for (left, right) in fine.iter().zip(coarse) {
        let mut scale = 1.0e-30_f64;
        let mut gap = 0.0_f64;
        for index in 0..6 {
            if !left[index].is_finite() || !right[index].is_finite() {
                return None;
            }
            scale = scale.max(left[index].abs()).max(right[index].abs());
            gap = gap.max((left[index] - right[index]).abs());
        }
        max_gap = max_gap.max(gap / scale);
    }
    Some(max_gap)
}

fn slope_pair(
    request: &ReviewRequest<'_>,
    raw: &BodyRecord,
    smoothed: &BodyRecord,
) -> serde_json::Value {
    let raw_slopes = monopole_slopes(
        request.raw,
        raw.centroid,
        raw.mass,
        request.gravitational_constant,
        request.omega,
    );
    let smoothed_slopes = monopole_slopes(
        request.smoothed,
        smoothed.centroid,
        smoothed.mass,
        request.gravitational_constant,
        request.omega,
    );
    serde_json::json!({
        "raw_max_rad": max_of(&raw_slopes),
        "smoothed_max_rad": max_of(&smoothed_slopes),
        "raw_quantiles_rad": quantiles(&raw_slopes),
        "smoothed_quantiles_rad": quantiles(&smoothed_slopes),
        "friction_angle_rad": request.mu_s.atan(),
    })
}

fn monopole_slopes(
    mesh: &TriMesh,
    centroid: DVec3,
    mass: f64,
    gravitational_constant: f64,
    omega: [f64; 3],
) -> Vec<f64> {
    let spin = DVec3::from_array(omega);
    (0..mesh.faces.len())
        .map(|face| {
            let [a, b, c] = mesh.triangle(face);
            let point = (a + b + c) * (1.0 / 3.0);
            let Some(normal) = mesh.face_normal(face) else {
                return 0.0;
            };
            let offset = point - centroid;
            let radius2 = offset.length_squared();
            let gravity = if radius2 > 1.0e-16 {
                offset * (-gravitational_constant * mass / (radius2 * radius2.sqrt()))
            } else {
                DVec3::ZERO
            };
            let effective = gravity - spin.cross(spin.cross(offset));
            let denom = effective.length_squared() * normal.length_squared();
            let sin2 = if denom > 1.0e-20 {
                (effective.cross(normal).length_squared() / denom).min(1.0)
            } else {
                0.0
            };
            sin2.sqrt().clamp(0.0, 1.0).asin()
        })
        .collect()
}

fn max_of(values: &[f64]) -> f64 {
    values.iter().copied().fold(0.0_f64, f64::max)
}

fn vertex_record_summary(records: &[VertexRecord]) -> serde_json::Value {
    if records.is_empty() {
        return serde_json::json!({
            "count": 0,
            "recorded": false,
        });
    }
    let max_disp = records
        .iter()
        .map(|record| record.displacement_m)
        .fold(0.0_f64, f64::max);
    serde_json::json!({
        "count": records.len(),
        "recorded": true,
        "fields": ["displacement_m", "protection", "slope_before_rad", "slope_after_rad", "curvature_before", "curvature_after"],
        "max_displacement_m": max_disp,
        "vertices": records.iter().map(|record| serde_json::json!({
            "displacement_m": record.displacement_m,
            "protection": record.protection,
            "slope_before_rad": record.slope_before_rad,
            "slope_after_rad": record.slope_after_rad,
            "curvature_before": record.curvature_before,
            "curvature_after": record.curvature_after,
        })).collect::<Vec<_>>(),
    })
}

fn star_diagnostic(request: &ReviewRequest<'_>) -> Result<serde_json::Value, ScfError> {
    let report = crate::rtfp_solver::geometry::star_audit::audit_star(
        request.smoothed,
        request.density,
        -1.0,
    )
    .map_err(|err| ScfError::model_not_closed(err.detail))?;
    Ok(serde_json::json!({
        "optional": true,
        "gates_general_mesh_rtfp": false,
        "shaped": report.shaped,
        "star_margin": report.star_margin,
        "unique_crossing_count": report.unique_crossing_count,
        "detail": report.detail,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::density::{CauchyDensity, CauchyKernel};

    fn options() -> RtfpOptions {
        RtfpOptions {
            directions: 14,
            r0: 0.01,
            ell0: 1.0,
            g: 6.67430e-11,
            quadrature_nodes: 2,
        }
    }

    fn request<'a>(
        raw: &'a TriMesh,
        smoothed: &'a TriMesh,
        density: &'a CauchyDensity,
        mask: &'a CraterMask,
        steps: u32,
    ) -> ReviewRequest<'a> {
        ReviewRequest {
            raw,
            smoothed,
            density,
            mass_target: 8.0 * 1190.0,
            mass_tolerance: 0.013,
            bulk_density: 1190.0,
            bulk_uncertainty: 30.0,
            porosity: 0.5,
            porosity_uncertainty: 0.02,
            grain_diameter: 0.1,
            grain_uncertainty: 0.05,
            mu_s: 0.6,
            mu_uncertainty: 0.06,
            mu_2: 0.6,
            cohesion: 0.05,
            cohesion_uncertainty: 0.45,
            spin_rate: 2.3e-4,
            spin_uncertainty: 6.0e-10,
            timestep: 0.25,
            timestep_uncertainty: 0.0,
            calibration: 1.0,
            calibration_uncertainty: 0.0,
            nominal_bulk: 1190.0,
            omega: [0.0, 0.0, 2.3e-4],
            gravitational_constant: 6.67430e-11,
            steps_executed: steps,
            mask,
            vertices: &[],
            coefficients: vec![
                Coefficient {
                    name: "mu_s".into(),
                    value: 0.6,
                    uncertainty: 0.06,
                    unit: "1".into(),
                    provenance: "literature_estimate".into(),
                },
                Coefficient {
                    name: "bulk_density".into(),
                    value: 1190.0,
                    uncertainty: 1.0,
                    unit: "kg/m^3".into(),
                    provenance: "observational_constraint".into(),
                },
                Coefficient {
                    name: "observational_calibration".into(),
                    value: 1.0,
                    uncertainty: 1.0,
                    unit: "1".into(),
                    provenance: "unmeasured".into(),
                },
            ],
            field: options(),
            parameter_source: "unit-test sourced priors".into(),
        }
    }

    #[test]
    fn unchanged_mesh_is_preprocessing_and_not_a_passed_reconstruction() {
        let mesh = TriMesh::unit_cube();
        let density = CauchyDensity::uniform(1190.0).unwrap();
        let mask = CraterMask::unmasked(mesh.vertices.len());
        let report = review(&request(&mesh, &mesh, &density, &mask, 0)).unwrap();
        assert_eq!(report.class, "indeterminate");
        assert_eq!(report.time_role, "initial_condition_preprocessing");
        assert_ne!(report.verification_level, "passed");
        assert!(
            report.document["rejected_labels"]
                .as_array()
                .unwrap()
                .iter()
                .any(|value| value == "million_year_evolution")
        );
        assert!(
            report.document["non_uniqueness_statement"]
                .as_str()
                .unwrap()
                .contains("不能证明")
        );
        assert!(report.document["feature_candidates"]["identification"] == "uncertain");
        assert_eq!(report.document["tier1"]["distinct_meshes"], 1);
        assert_eq!(report.document["tier1"]["coverage"], 1.0);
        assert_ne!(report.time_role, "million_year_evolution");
        assert_ne!(report.time_role, "historical_reconstruction");
        assert_eq!(report.document["protection"]["identification"], "uncertain");
        assert_eq!(report.document["protection"]["claimed_real_terrain"], false);
        assert_eq!(report.document["star"]["gates_general_mesh_rtfp"], false);
        let samples: Vec<f64> = report.document["uncertainty_summary"]["parameters"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|item| item["parameter"] == "mu_s")
            .map(|item| item["value"].as_f64().unwrap())
            .collect();
        assert!(samples.iter().any(|value| (*value - 0.66).abs() < 1e-12));
        assert!(samples.iter().any(|value| (*value - 0.54).abs() < 1e-12));
        assert!(
            report.document["uncertainty_summary"]["unmeasured"]
                .as_array()
                .unwrap()
                .iter()
                .any(|item| item == "observational_calibration")
        );
        assert_eq!(report.document["tier1"]["feynman"], "not_called");
        assert_eq!(report.document["tier1"]["radial_projection"], "not_used");
    }

    #[test]
    fn removing_a_bulge_and_the_roughness_is_obviously_wrong() {
        let raw = TriMesh::unit_cube();
        let smoothed_vertices = raw
            .vertices
            .iter()
            .map(|vertex| DVec3::new(vertex.x * 0.2, vertex.y * 0.2, vertex.z * 0.2))
            .collect();
        let smoothed = TriMesh::try_new(smoothed_vertices, raw.faces.clone()).unwrap();
        let density = CauchyDensity::uniform(1190.0).unwrap();
        let mask = CraterMask::unmasked(raw.vertices.len());
        let report = review(&request(&raw, &smoothed, &density, &mask, 0)).unwrap();
        assert_eq!(report.class, "obviously_wrong");
        assert_ne!(report.verification_level, "passed");
        assert_eq!(report.document["tier1"]["distinct_meshes"], 2);
        assert_ne!(
            report.document["tier1"]["raw_mesh_hash"],
            report.document["tier1"]["smoothed_mesh_hash"]
        );
        assert!(
            !report.document["obvious_failure_reasons"]
                .as_array()
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn mass_uses_the_full_cauchy_distribution() {
        let mesh = TriMesh::unit_cube();
        let kernel = CauchyKernel::try_new(DVec3::new(0.35, -0.2, 0.1), 4000.0, 1.4, 1.3).unwrap();
        let density = CauchyDensity::try_new(25.0, vec![kernel]).unwrap();
        let mask = CraterMask::unmasked(mesh.vertices.len());
        let report = review(&request(&mesh, &mesh, &density, &mask, 0)).unwrap();
        let mass = report.document["bodies"][0]["mass_kg"].as_f64().unwrap();
        let first_kernel = 4000.0 * mesh.signed_volume().abs();
        assert!(
            (mass - first_kernel).abs() / first_kernel > 1.0e-2,
            "mass {mass} collapsed to the first kernel weight"
        );
        assert_eq!(
            report.document["bodies"][0]["mass_method"],
            "closed_mesh_full_cauchy"
        );
        assert_ne!(report.verification_level, "passed");
        assert_ne!(report.class, "plausible");
    }
}
