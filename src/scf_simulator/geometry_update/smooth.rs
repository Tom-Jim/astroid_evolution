//! Crater-preserving slope relaxation.
//!
//! The normal step is the grain-inertial speed on supercritical slopes. Neighbor
//! averaging is used only to spread the protection weight, not to move vertices.
//! Volume is not restored by a uniform scale.

use std::path::{Path, PathBuf};

use crate::common::geometry::TriMesh;
use crate::common::math::DVec3;
use crate::scf_simulator::error::ScfError;

#[derive(Clone, Debug)]
pub struct SourcedParameter {
    pub name: String,
    pub value: f64,
    pub unit: String,
    pub uncertainty: f64,
    pub source: String,
    pub calibration: String,
    /// `observational_constraint`, `literature_estimate`, `numerical_stability`, or `unmeasured`.
    pub provenance: String,
}

#[derive(Clone, Debug)]
pub struct SlopeLaw {
    pub grain_diameter: SourcedParameter,
    pub friction_angle: SourcedParameter,
    pub critical_slope: SourcedParameter,
    pub spin_rate: SourcedParameter,
    pub gravitational_constant: SourcedParameter,
    pub timestep: SourcedParameter,
    pub calibration: SourcedParameter,
}

#[derive(Clone, Debug)]
pub struct SlopeRequest {
    pub law: SlopeLaw,
    pub preserve_topology: bool,
    pub penalty: Option<SourcedParameter>,
    pub floor_path: Option<PathBuf>,
    pub wall_path: Option<PathBuf>,
    pub rim_path: Option<PathBuf>,
}

#[derive(Clone, Debug)]
pub struct CraterMask {
    pub floor: Vec<f64>,
    pub wall: Vec<f64>,
    pub rim: Vec<f64>,
    pub protection: Vec<f64>,
    pub penalty: Option<f64>,
    pub claimed: bool,
    /// `observed` when floor, wall, and rim masks were read. Otherwise `uncertain`.
    pub identification: &'static str,
    /// True only for an observational mask. Geometric candidates never set this.
    pub claimed_real_terrain: bool,
}

#[derive(Clone, Debug)]
pub struct VertexRecord {
    pub displacement_m: f64,
    pub protection: f64,
    pub slope_before_rad: f64,
    pub slope_after_rad: f64,
    pub curvature_before: f64,
    pub curvature_after: f64,
}

#[derive(Clone, Debug)]
pub struct Relaxation {
    pub mesh: TriMesh,
    pub mask: CraterMask,
    pub vertices: Vec<VertexRecord>,
    pub method: &'static str,
}

impl CraterMask {
    pub fn unmasked(count: usize) -> Self {
        Self {
            floor: vec![0.0; count],
            wall: vec![0.0; count],
            rim: vec![0.0; count],
            protection: vec![0.0; count],
            penalty: None,
            claimed: false,
            identification: "uncertain",
            claimed_real_terrain: false,
        }
    }

    pub fn weight(&self, index: usize) -> f64 {
        self.protection.get(index).copied().unwrap_or(0.0)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct CraterMeasure {
    pub depth: f64,
    pub rim_curvature: f64,
}

pub fn parse_slope_request(
    table: &toml::Value,
    gravity: f64,
) -> Result<(bool, f64, SlopeRequest), ScfError> {
    if table.get("lambda").is_some() || table.get("mu").is_some() {
        return Err(ScfError::model_not_closed(
            "smoothing.lambda and smoothing.mu are not physical coefficients",
        ));
    }
    if table.get("preserve_volume").is_some() || table.get("iterations").is_some() {
        return Err(ScfError::model_not_closed(
            "volume is a Lagrange constraint; uniform rescaling and a fixed iteration count are not used",
        ));
    }
    let method = required_str(table, "method", "smoothing")?;
    if method == "taubin" || method == "hc" {
        return Err(ScfError::model_not_closed(
            "Taubin/HC smoothing is not a material law; use method = \"slope_relaxation\"",
        ));
    }
    if method != "slope_relaxation" {
        return Err(ScfError::model_not_closed(
            "smoothing.method must be slope_relaxation",
        ));
    }
    let preserve_topology = required_bool(table, "preserve_topology", "smoothing")?;
    if !preserve_topology {
        return Err(ScfError::geometry_invalid(
            "smoothing.preserve_topology is false and this solver does not rebuild faces",
        ));
    }
    let star_min_cos = required_f64(table, "star_min_cos", "smoothing")?;
    if !(star_min_cos.is_finite() && (-1.0..=1.0).contains(&star_min_cos)) {
        return Err(ScfError::model_not_closed(
            "smoothing.star_min_cos must lie in [-1, 1]",
        ));
    }
    let require_star = required_bool(table, "require_star", "smoothing")?;
    let law = SlopeLaw {
        grain_diameter: sourced(table, "grain_diameter", "m")?,
        friction_angle: sourced(table, "friction_angle", "rad")?,
        critical_slope: sourced(table, "critical_slope", "rad")?,
        spin_rate: sourced(table, "spin_rate", "rad/s")?,
        gravitational_constant: sourced(table, "gravitational_constant", "m^3/kg/s^2")?,
        timestep: sourced(table, "timestep", "s")?,
        calibration: sourced(table, "observational_calibration", "1")?,
    };
    if law.grain_diameter.value <= 0.0
        || law.spin_rate.value <= 0.0
        || law.gravitational_constant.value <= 0.0
        || law.timestep.value <= 0.0
        || law.calibration.value <= 0.0
    {
        return Err(ScfError::model_not_closed(
            "grain diameter, spin, G, timestep, and the calibration factor must be positive",
        ));
    }
    if !(0.0 < law.friction_angle.value && law.friction_angle.value < std::f64::consts::FRAC_PI_2) {
        return Err(ScfError::model_not_closed(
            "friction_angle must lie in (0, pi/2)",
        ));
    }
    if !(0.0 < law.critical_slope.value && law.critical_slope.value < std::f64::consts::FRAC_PI_2) {
        return Err(ScfError::model_not_closed(
            "critical_slope must lie in (0, pi/2)",
        ));
    }
    let slope_gap = (law.critical_slope.value - law.friction_angle.value).abs();
    if slope_gap > law.critical_slope.uncertainty + law.friction_angle.uncertainty {
        return Err(ScfError::model_not_closed(
            "critical_slope and friction_angle disagree by more than their sourced uncertainties",
        ));
    }
    let g_gap = (law.gravitational_constant.value - gravity).abs();
    if g_gap > law.gravitational_constant.uncertainty.max(0.0) {
        return Err(ScfError::model_not_closed(
            "stokes.g is outside the sourced gravitational_constant uncertainty",
        ));
    }
    let (penalty, floor_path, wall_path, rim_path) = parse_mask(table)?;
    Ok((
        require_star,
        star_min_cos,
        SlopeRequest {
            law,
            preserve_topology,
            penalty,
            floor_path,
            wall_path,
            rim_path,
        },
    ))
}

pub fn load_crater_mask(
    request: &SlopeRequest,
    mesh: &TriMesh,
    project_dir: &Path,
) -> Result<CraterMask, ScfError> {
    let paths = [
        request.floor_path.as_ref(),
        request.wall_path.as_ref(),
        request.rim_path.as_ref(),
    ];
    if paths.iter().all(|path| path.is_none()) {
        if request.penalty.is_some() {
            return Err(ScfError::model_not_closed(
                "a crater penalty requires floor, wall, and rim masks",
            ));
        }
        return Ok(CraterMask::unmasked(mesh.vertices.len()));
    }
    if paths.iter().any(|path| path.is_none()) {
        return Err(ScfError::model_not_closed(
            "crater_floor, crater_wall, and crater_rim must all be present",
        ));
    }
    let floor = read_mask(
        project_dir,
        request.floor_path.as_ref().unwrap(),
        mesh,
        "crater_floor",
    )?;
    let wall = read_mask(
        project_dir,
        request.wall_path.as_ref().unwrap(),
        mesh,
        "crater_wall",
    )?;
    let rim = read_mask(
        project_dir,
        request.rim_path.as_ref().unwrap(),
        mesh,
        "crater_rim",
    )?;
    let combined: Vec<f64> = floor
        .iter()
        .zip(&wall)
        .zip(&rim)
        .map(|((a, b), c)| a.max(*b).max(*c).clamp(0.0, 1.0))
        .collect();
    let neighbors = vertex_neighbors(&mesh.faces, mesh.vertices.len());
    let protection = transition_weights(&combined, &neighbors);
    Ok(CraterMask {
        floor,
        wall,
        rim,
        protection,
        penalty: request.penalty.as_ref().map(|item| item.value),
        claimed: true,
        identification: "observed",
        claimed_real_terrain: true,
    })
}

pub const PROVENANCE_OBSERVED: &str = "observational_constraint";
pub const PROVENANCE_LITERATURE: &str = "literature_estimate";
pub const PROVENANCE_NUMERICAL: &str = "numerical_stability";
pub const PROVENANCE_UNMEASURED: &str = "unmeasured";

pub fn classify_provenance(
    explicit: Option<&str>,
    source: &str,
    calibration: &str,
) -> Result<String, ScfError> {
    if let Some(name) = explicit {
        if matches!(
            name,
            PROVENANCE_OBSERVED
                | PROVENANCE_LITERATURE
                | PROVENANCE_NUMERICAL
                | PROVENANCE_UNMEASURED
        ) {
            return Ok(name.to_string());
        }
        return Err(ScfError::model_not_closed(format!(
            "provenance '{name}' is not observational_constraint, literature_estimate, numerical_stability, or unmeasured"
        )));
    }
    let text = format!("{source} {calibration}").to_ascii_lowercase();
    let class = if text.contains("numerical")
        || text.contains("model step")
        || text.contains("model stage")
        || text.contains("pressure floor")
        || text.contains("shear-rate floor")
        || text.contains("solver penalty")
    {
        PROVENANCE_NUMERICAL
    } else if text.contains("no ryugu")
        || text.contains("no measured")
        || text.contains("no separate")
        || text.contains("no closed")
        || text.contains("unsourced")
        || text.contains("not a glass")
        || text.contains("does not enter")
        || text.contains("not an observation")
        || text.contains("not an observed")
    {
        PROVENANCE_UNMEASURED
    } else if text.contains("measured")
        || text.contains("spacecraft")
        || text.contains("codata")
        || text.contains("gravity mass")
    {
        PROVENANCE_OBSERVED
    } else if text.contains("doi") || text.contains("et al") {
        PROVENANCE_LITERATURE
    } else {
        PROVENANCE_UNMEASURED
    };
    Ok(class.to_string())
}

/// Slope relaxation. Vertices move only where the local slope exceeds the
/// friction angle, and only by the unprotected fraction of the normal step.
/// Faces are unchanged. Volume is restored by a normal Lagrange offset on
/// unprotected vertices, not by a uniform scale.
pub fn relax_slopes(
    mesh: &TriMesh,
    density: &crate::common::density::CauchyDensity,
    request: &SlopeRequest,
    mask: &CraterMask,
    omega: [f64; 3],
    mass_tolerance: f64,
) -> Result<Relaxation, ScfError> {
    if !request.preserve_topology {
        return Err(ScfError::geometry_invalid(
            "smoothing.preserve_topology is false and this solver does not rebuild faces",
        ));
    }
    let moments = crate::rtfp_solver::volume::closed_density_integrals(mesh, density)
        .map_err(ScfError::model_not_closed)?;
    let mass0 = moments.mass;
    let centroid = moments.centroid;
    let volume0 = mesh.signed_volume();
    if !(volume0.is_finite() && volume0 > 0.0) {
        return Err(ScfError::geometry_invalid(
            "slope relaxation requires a positive volume",
        ));
    }
    let count = mesh.vertices.len();
    let neighbors = vertex_neighbors(&mesh.faces, count);
    let curvature_before = vertex_curvature(mesh, &neighbors);
    let normals = vertex_normals(mesh);
    let geometric = geometric_protection(mesh, &neighbors, &curvature_before);
    let mut protection = if mask.claimed {
        mask.protection.clone()
    } else {
        vec![0.0; count]
    };
    if protection.len() != count {
        protection.resize(count, 0.0);
    }
    for (slot, weight) in protection.iter_mut().zip(&geometric) {
        *slot = slot.max(*weight).clamp(0.0, 1.0);
    }
    let slope_before = vertex_slopes(
        mesh,
        centroid,
        mass0,
        request.law.gravitational_constant.value,
        omega,
        &normals,
    );
    let mut step = vec![0.0; count];
    for index in 0..count {
        let point = mesh.vertices[index];
        let gravity = effective_gravity(
            point,
            centroid,
            mass0,
            request.law.gravitational_constant.value,
            omega,
        );
        let normal = normals[index];
        let hard_lock = protection[index] >= 1.0 - 1.0e-9;
        step[index] = slope_normal_displacement(
            &request.law,
            gravity.to_array(),
            normal.to_array(),
            request.law.timestep.value,
            point.length().max(1.0),
            protection[index],
            hard_lock,
        )
        .map_err(ScfError::model_not_closed)?;
    }
    let mut accepted = mesh.vertices.clone();
    let mut placed = false;
    let mut scale = 1.0;
    if step.iter().any(|value| value.abs() > 0.0) {
        for _ in 0..8 {
            let mut proposed = mesh.vertices.clone();
            for index in 0..count {
                proposed[index] = proposed[index] + normals[index] * (step[index] * scale);
            }
            let lock: Vec<f64> = step
                .iter()
                .zip(&protection)
                .map(|(disp, weight)| if disp.abs() > 0.0 { *weight } else { 1.0 })
                .collect();
            proposed = restore_volume(&proposed, &normals, &lock, &mesh.faces, volume0);
            if let Some(candidate) = accept_candidate(
                &proposed,
                &mesh.faces,
                density,
                mass0,
                volume0,
                mass_tolerance,
            )? {
                accepted = candidate;
                placed = true;
                break;
            }
            scale *= 0.5;
        }
    }
    if !placed {
        accepted = mesh.vertices.clone();
    }
    let smoothed = TriMesh::try_new(accepted, mesh.faces.clone())
        .map_err(|err| ScfError::geometry_invalid(err.detail))?;
    if smoothed.faces != mesh.faces {
        return Err(ScfError::geometry_invalid(
            "slope relaxation changed the face topology",
        ));
    }
    let after_neighbors = vertex_neighbors(&smoothed.faces, smoothed.vertices.len());
    let curvature_after = vertex_curvature(&smoothed, &after_neighbors);
    let normals_after = vertex_normals(&smoothed);
    let after_moments = crate::rtfp_solver::volume::closed_density_integrals(&smoothed, density)
        .map_err(ScfError::model_not_closed)?;
    let slope_after = vertex_slopes(
        &smoothed,
        after_moments.centroid,
        after_moments.mass,
        request.law.gravitational_constant.value,
        omega,
        &normals_after,
    );
    let vertices = (0..count)
        .map(|index| VertexRecord {
            displacement_m: (smoothed.vertices[index] - mesh.vertices[index]).length(),
            protection: protection[index],
            slope_before_rad: slope_before[index],
            slope_after_rad: slope_after[index],
            curvature_before: curvature_before[index],
            curvature_after: curvature_after[index],
        })
        .collect();
    let returned = CraterMask {
        floor: if mask.floor.len() == count {
            mask.floor.clone()
        } else {
            vec![0.0; count]
        },
        wall: if mask.wall.len() == count {
            mask.wall.clone()
        } else {
            vec![0.0; count]
        },
        rim: if mask.rim.len() == count {
            mask.rim.clone()
        } else {
            vec![0.0; count]
        },
        protection,
        penalty: mask.penalty,
        claimed: mask.claimed,
        identification: if mask.claimed {
            "observed"
        } else {
            "uncertain"
        },
        claimed_real_terrain: mask.claimed,
    };
    Ok(Relaxation {
        mesh: smoothed,
        mask: returned,
        vertices,
        method: "slope_relaxation",
    })
}

/// Inward normal step on slopes steeper than the sourced friction angle.
/// A fully protected vertex returns zero. Subcritical slopes return zero.
pub fn slope_normal_displacement(
    law: &SlopeLaw,
    g_eff: [f64; 3],
    normal: [f64; 3],
    dt: f64,
    radius: f64,
    protection: f64,
    hard_lock: bool,
) -> Result<f64, String> {
    if !(dt.is_finite() && dt >= 0.0) {
        return Err("slope timestep is not finite and non-negative".into());
    }
    let g = DVec3::from_array(g_eff);
    let n = DVec3::from_array(normal);
    let denom = g.length_squared() * n.length_squared();
    let sin2 = if denom > 1.0e-20 {
        (g.cross(n).length_squared() / denom).min(1.0)
    } else {
        0.0
    };
    let excess = (sin2 - law.friction_angle.value.sin().powi(2)).max(0.0);
    let guard = protection.clamp(0.0, 1.0);
    if excess == 0.0 || (hard_lock && guard >= 1.0) {
        return Ok(0.0);
    }
    let strength = law.friction_angle.value.tan().abs().max(1.0e-8);
    let speed =
        law.calibration.value * (g.length() * law.grain_diameter.value).sqrt() * excess / strength;
    let mut disp = -speed * dt * (1.0 - guard);
    let cap = law.spin_rate.value.abs() * dt * radius.max(law.grain_diameter.value);
    if cap.is_finite() && cap > 0.0 {
        disp = disp.clamp(-cap, cap);
    }
    if !disp.is_finite() {
        return Err("slope displacement is not finite".into());
    }
    Ok(disp)
}

pub fn transition_weights(mask: &[f64], neighbors: &[Vec<usize>]) -> Vec<f64> {
    mask.iter()
        .enumerate()
        .map(|(index, value)| {
            let mut acc = value.clamp(0.0, 1.0);
            let mut count = 1.0;
            for neighbor in &neighbors[index] {
                acc += mask[*neighbor].clamp(0.0, 1.0);
                count += 1.0;
            }
            acc / count
        })
        .collect()
}

pub fn crater_measure(mesh: &TriMesh, mask: &CraterMask) -> Option<CraterMeasure> {
    if !mask.claimed {
        return None;
    }
    let floor = mean_position(mesh, &mask.floor)?;
    let rim = mean_position(mesh, &mask.rim)?;
    let outward = rim.try_normalize()?;
    let depth = (rim - floor).dot(outward);
    let neighbors = vertex_neighbors(&mesh.faces, mesh.vertices.len());
    let mut acc = 0.0;
    let mut count = 0.0;
    for (index, weight) in mask.rim.iter().enumerate() {
        if *weight <= 0.0 {
            continue;
        }
        let normal = vertex_normal(mesh, index)?;
        let mut turn = 0.0;
        let mut n = 0.0;
        for neighbor in &neighbors[index] {
            if let Some(other) = vertex_normal(mesh, *neighbor) {
                turn += 1.0 - normal.dot(other).clamp(-1.0, 1.0);
                n += 1.0;
            }
        }
        if n > 0.0 {
            acc += *weight * turn / n;
            count += *weight;
        }
    }
    let rim_curvature = if count > 0.0 { acc / count } else { 0.0 };
    if depth.is_finite() && rim_curvature.is_finite() {
        Some(CraterMeasure {
            depth,
            rim_curvature,
        })
    } else {
        None
    }
}

fn mean_position(mesh: &TriMesh, weight: &[f64]) -> Option<DVec3> {
    let mut acc = DVec3::ZERO;
    let mut sum = 0.0;
    for (vertex, w) in mesh.vertices.iter().zip(weight) {
        if *w > 0.0 {
            acc = acc + *vertex * *w;
            sum += *w;
        }
    }
    if sum > 0.0 {
        Some(acc * (1.0 / sum))
    } else {
        None
    }
}

fn vertex_normal(mesh: &TriMesh, index: usize) -> Option<DVec3> {
    let mut acc = DVec3::ZERO;
    for face in &mesh.faces {
        if !face.contains(&(index as u32)) {
            continue;
        }
        let [a, b, c] = [
            mesh.vertices[face[0] as usize],
            mesh.vertices[face[1] as usize],
            mesh.vertices[face[2] as usize],
        ];
        acc = acc + (b - a).cross(c - a);
    }
    acc.try_normalize()
}

fn parse_mask(
    table: &toml::Value,
) -> Result<
    (
        Option<SourcedParameter>,
        Option<PathBuf>,
        Option<PathBuf>,
        Option<PathBuf>,
    ),
    ScfError,
> {
    let Some(mask) = table.get("crater_mask") else {
        return Ok((None, None, None, None));
    };
    let penalty = if mask.get("penalty").is_some() {
        let item = sourced(mask, "penalty", "1/m")?;
        if item.value < 0.0 || !item.value.is_finite() {
            return Err(ScfError::model_not_closed(
                "crater penalty must be a finite non-negative number",
            ));
        }
        Some(item)
    } else {
        None
    };
    Ok((
        penalty,
        Some(path_of(mask, "crater_floor")?),
        Some(path_of(mask, "crater_wall")?),
        Some(path_of(mask, "crater_rim")?),
    ))
}

fn path_of(table: &toml::Value, key: &str) -> Result<PathBuf, ScfError> {
    let text = required_str(table, key, "smoothing.crater_mask")?;
    if text.is_empty() {
        return Err(ScfError::model_not_closed(format!(
            "smoothing.crater_mask.{key} is empty"
        )));
    }
    Ok(PathBuf::from(text))
}

fn read_mask(
    project_dir: &Path,
    path: &Path,
    mesh: &TriMesh,
    name: &str,
) -> Result<Vec<f64>, ScfError> {
    let full = if path.is_absolute() {
        path.to_path_buf()
    } else {
        project_dir.join(path)
    };
    let text = std::fs::read_to_string(&full).map_err(|err| {
        ScfError::model_not_closed(format!("cannot read {name} {}: {err}", full.display()))
    })?;
    let mut values = Vec::new();
    for (line_number, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let value = line.parse::<f64>().map_err(|_| {
            ScfError::model_not_closed(format!("{name} line {} is not a weight", line_number + 1))
        })?;
        if !value.is_finite() || !(0.0..=1.0).contains(&value) {
            return Err(ScfError::model_not_closed(format!(
                "{name} line {} is outside [0, 1]",
                line_number + 1
            )));
        }
        values.push(value);
    }
    if values.len() == mesh.vertices.len() {
        return Ok(values);
    }
    if values.len() == mesh.faces.len() {
        return Ok(faces_to_vertices(mesh, &values));
    }
    Err(ScfError::model_not_closed(format!(
        "{name} has {} weights; the mesh has {} vertices and {} faces",
        values.len(),
        mesh.vertices.len(),
        mesh.faces.len()
    )))
}

fn faces_to_vertices(mesh: &TriMesh, face_weight: &[f64]) -> Vec<f64> {
    let mut weight = vec![0.0_f64; mesh.vertices.len()];
    for (face, value) in mesh.faces.iter().zip(face_weight) {
        for index in *face {
            weight[index as usize] = weight[index as usize].max(*value);
        }
    }
    weight
}

fn sourced(table: &toml::Value, name: &str, unit: &str) -> Result<SourcedParameter, ScfError> {
    let item = table.get(name).ok_or_else(|| {
        ScfError::model_not_closed(format!("missing smoothing.{name}; a source is required"))
    })?;
    let value = required_f64(item, "value", &format!("smoothing.{name}"))?;
    let declared = required_str(item, "unit", &format!("smoothing.{name}"))?;
    let uncertainty = required_f64(item, "uncertainty", &format!("smoothing.{name}"))?;
    let source = required_str(item, "source", &format!("smoothing.{name}"))?;
    let calibration = required_str(item, "calibration", &format!("smoothing.{name}"))?;
    let explicit = item.get("provenance").and_then(|entry| entry.as_str());
    let provenance = classify_provenance(explicit, &source, &calibration)?;
    if declared != unit {
        return Err(ScfError::model_not_closed(format!(
            "smoothing.{name}.unit must be {unit}"
        )));
    }
    if source.trim().is_empty() || calibration.trim().is_empty() {
        return Err(ScfError::model_not_closed(format!(
            "smoothing.{name} is missing its source or calibration"
        )));
    }
    if !value.is_finite() || !uncertainty.is_finite() || uncertainty < 0.0 {
        return Err(ScfError::model_not_closed(format!(
            "smoothing.{name} value or uncertainty is not finite and non-negative"
        )));
    }
    Ok(SourcedParameter {
        name: name.into(),
        value,
        unit: declared,
        uncertainty,
        source,
        calibration,
        provenance,
    })
}

fn required_str(table: &toml::Value, key: &str, owner: &str) -> Result<String, ScfError> {
    table
        .get(key)
        .and_then(|item| item.as_str())
        .map(|text| text.to_string())
        .ok_or_else(|| ScfError::model_not_closed(format!("missing {owner}.{key}")))
}

fn required_f64(table: &toml::Value, key: &str, owner: &str) -> Result<f64, ScfError> {
    let value = table
        .get(key)
        .and_then(|item| {
            item.as_float()
                .or_else(|| item.as_integer().map(|n| n as f64))
        })
        .ok_or_else(|| ScfError::model_not_closed(format!("missing {owner}.{key}")))?;
    if !value.is_finite() {
        return Err(ScfError::model_not_closed(format!(
            "{owner}.{key} is not finite"
        )));
    }
    Ok(value)
}

fn required_bool(table: &toml::Value, key: &str, owner: &str) -> Result<bool, ScfError> {
    table
        .get(key)
        .and_then(toml::Value::as_bool)
        .ok_or_else(|| ScfError::model_not_closed(format!("{owner}.{key} must be a boolean")))
}

fn vertex_normals(mesh: &TriMesh) -> Vec<DVec3> {
    (0..mesh.vertices.len())
        .map(|index| vertex_normal(mesh, index).unwrap_or(DVec3::new(0.0, 0.0, 1.0)))
        .collect()
}

fn vertex_curvature(mesh: &TriMesh, neighbors: &[Vec<usize>]) -> Vec<f64> {
    (0..mesh.vertices.len())
        .map(|index| {
            let Some(normal) = vertex_normal(mesh, index) else {
                return 0.0;
            };
            let mut turn = 0.0;
            let mut count = 0.0;
            for &other in &neighbors[index] {
                if let Some(neighbor) = vertex_normal(mesh, other) {
                    turn += 1.0 - normal.dot(neighbor).clamp(-1.0, 1.0);
                    count += 1.0;
                }
            }
            if count > 0.0 { turn / count } else { 0.0 }
        })
        .collect()
}

fn geometric_protection(mesh: &TriMesh, neighbors: &[Vec<usize>], curvature: &[f64]) -> Vec<f64> {
    let center = crate::rtfp_solver::volume::uniform_integrals(mesh, 1.0).centroid;
    let scale = mesh
        .vertices
        .iter()
        .map(|vertex| (*vertex - center).length())
        .fold(0.0_f64, f64::max)
        .max(1.0);
    let mut weight = vec![0.0; mesh.vertices.len()];
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
        if (neighbor_radius - radius).abs() > 0.03 * scale {
            weight[index] = 1.0;
        }
    }
    let mut sorted = curvature.to_vec();
    sorted.sort_by(|left, right| left.total_cmp(right));
    let pivot = sorted[sorted.len() * 9 / 10];
    let peak = sorted.last().copied().unwrap_or(pivot);
    if peak > pivot {
        for (slot, value) in weight.iter_mut().zip(curvature) {
            if *value > pivot {
                *slot = f64::max(*slot, ((*value - pivot) / (peak - pivot)).clamp(0.0, 1.0));
            }
        }
    }
    weight
}

fn effective_gravity(
    point: DVec3,
    centroid: DVec3,
    mass: f64,
    gravitational_constant: f64,
    omega: [f64; 3],
) -> DVec3 {
    let offset = point - centroid;
    let radius2 = offset.length_squared();
    let gravity = if radius2 > 1.0e-16 && mass.is_finite() && gravitational_constant.is_finite() {
        offset * (-gravitational_constant * mass / (radius2 * radius2.sqrt()))
    } else {
        DVec3::ZERO
    };
    let spin = DVec3::from_array(omega);
    gravity - spin.cross(spin.cross(offset))
}

fn vertex_slopes(
    mesh: &TriMesh,
    centroid: DVec3,
    mass: f64,
    gravitational_constant: f64,
    omega: [f64; 3],
    normals: &[DVec3],
) -> Vec<f64> {
    mesh.vertices
        .iter()
        .zip(normals)
        .map(|(point, normal)| {
            let gravity = effective_gravity(*point, centroid, mass, gravitational_constant, omega);
            let denom = gravity.length_squared() * normal.length_squared();
            let sin2 = if denom > 1.0e-20 {
                (gravity.cross(*normal).length_squared() / denom).min(1.0)
            } else {
                0.0
            };
            sin2.sqrt().clamp(0.0, 1.0).asin()
        })
        .collect()
}

fn restore_volume(
    vertices: &[DVec3],
    normals: &[DVec3],
    protection: &[f64],
    faces: &[[u32; 3]],
    target: f64,
) -> Vec<DVec3> {
    let current = polyhedron_volume(vertices, faces);
    if !current.is_finite() || (current - target).abs() <= 1.0e-8 * target.abs().max(1.0) {
        return vertices.to_vec();
    }
    let probe = 1.0e-4
        * vertices
            .iter()
            .map(|vertex| vertex.length())
            .fold(1.0_f64, f64::max);
    let probed = offset_unprotected(vertices, normals, protection, probe);
    let probed_volume = polyhedron_volume(&probed, faces);
    let slope = (probed_volume - current) / probe;
    if !slope.is_finite() || slope.abs() < 1.0e-14 {
        return vertices.to_vec();
    }
    let lambda = ((target - current) / slope).clamp(-20.0 * probe, 20.0 * probe);
    offset_unprotected(vertices, normals, protection, lambda)
}

fn offset_unprotected(
    vertices: &[DVec3],
    normals: &[DVec3],
    protection: &[f64],
    lambda: f64,
) -> Vec<DVec3> {
    vertices
        .iter()
        .zip(normals)
        .zip(protection)
        .map(|((vertex, normal), weight)| {
            *vertex + *normal * (lambda * (1.0 - weight).clamp(0.0, 1.0))
        })
        .collect()
}

fn polyhedron_volume(vertices: &[DVec3], faces: &[[u32; 3]]) -> f64 {
    let mut volume = 0.0;
    for face in faces {
        let a = vertices[face[0] as usize];
        let b = vertices[face[1] as usize];
        let c = vertices[face[2] as usize];
        volume += a.dot(b.cross(c));
    }
    volume / 6.0
}

fn accept_candidate(
    vertices: &[DVec3],
    faces: &[[u32; 3]],
    density: &crate::common::density::CauchyDensity,
    mass0: f64,
    volume0: f64,
    mass_tolerance: f64,
) -> Result<Option<Vec<DVec3>>, ScfError> {
    let volume = polyhedron_volume(vertices, faces);
    if !(volume.is_finite() && volume > 0.0) {
        return Ok(None);
    }
    let candidate = match TriMesh::try_new(vertices.to_vec(), faces.to_vec()) {
        Ok(mesh) => mesh,
        Err(_) => return Ok(None),
    };
    if candidate.faces.len() != faces.len() {
        return Ok(None);
    }
    let mass = crate::rtfp_solver::volume::closed_density_integrals(&candidate, density)
        .map_err(ScfError::model_not_closed)?
        .mass;
    let mass_gap = (mass - mass0).abs() / mass0.abs().max(1.0e-30);
    let volume_gap = (volume - volume0).abs() / volume0.abs().max(1.0e-30);
    let limit = if mass_tolerance.is_finite() && mass_tolerance >= 0.0 {
        mass_tolerance
    } else {
        0.0
    };
    if mass_gap <= limit.max(1.0e-8) && volume_gap <= limit.max(1.0e-6) {
        Ok(Some(candidate.vertices))
    } else {
        Ok(None)
    }
}

pub fn vertex_neighbors(faces: &[[u32; 3]], count: usize) -> Vec<Vec<usize>> {
    let mut neighbors = vec![Vec::new(); count];
    for face in faces {
        let ids = [face[0] as usize, face[1] as usize, face[2] as usize];
        for a in 0..3 {
            for b in 0..3 {
                if a != b && !neighbors[ids[a]].contains(&ids[b]) {
                    neighbors[ids[a]].push(ids[b]);
                }
            }
        }
    }
    neighbors
}

#[cfg(test)]
mod tests {
    use super::*;

    fn law() -> SlopeLaw {
        let item = |name: &str, value: f64, unit: &str| SourcedParameter {
            name: name.into(),
            value,
            unit: unit.into(),
            uncertainty: 0.0,
            source: "test".into(),
            calibration: "test".into(),
            provenance: "literature_estimate".into(),
        };
        SlopeLaw {
            grain_diameter: item("grain_diameter", 1.0, "m"),
            friction_angle: item("friction_angle", 0.6, "rad"),
            critical_slope: item("critical_slope", 0.4, "rad"),
            spin_rate: item("spin_rate", 1.0, "rad/s"),
            gravitational_constant: item("gravitational_constant", 1.0, "m^3/kg/s^2"),
            timestep: item("timestep", 1.0, "s"),
            calibration: item("observational_calibration", 1.0, "1"),
        }
    }

    #[test]
    fn subcritical_slope_and_full_protection_do_not_move() {
        let flat = slope_normal_displacement(
            &law(),
            [0.0, 0.0, -1.0],
            [0.0, 0.0, 1.0],
            1.0,
            1.0,
            0.0,
            true,
        )
        .unwrap();
        assert_eq!(flat, 0.0);
        let locked = slope_normal_displacement(
            &law(),
            [1.0, 0.0, -0.2],
            [0.0, 0.0, 1.0],
            1.0,
            1.0,
            1.0,
            true,
        )
        .unwrap();
        assert_eq!(locked, 0.0);
    }

    #[test]
    fn supercritical_slope_moves_inward_and_the_mask_edge_is_between_zero_and_one() {
        let moved = slope_normal_displacement(
            &law(),
            [1.0, 0.0, -0.05],
            [0.0, 0.0, 1.0],
            0.1,
            2.0,
            0.0,
            true,
        )
        .unwrap();
        assert!(moved < 0.0, "{moved}");
        let mask = [0.0, 1.0, 1.0, 0.0];
        let neighbors = vec![vec![1], vec![0, 2], vec![1, 3], vec![2]];
        let weights = transition_weights(&mask, &neighbors);
        assert!(weights[0] > 0.0 && weights[0] < 1.0, "{}", weights[0]);
        assert!(weights[3] > 0.0 && weights[3] < 1.0, "{}", weights[3]);
        assert!(weights[1] > weights[0]);
        let between = slope_normal_displacement(
            &law(),
            [0.55, 0.0, -1.0],
            [0.0, 0.0, 1.0],
            1.0,
            1.0,
            0.0,
            true,
        )
        .unwrap();
        assert_eq!(
            between, 0.0,
            "a slope between the critical slope and the friction angle stays put"
        );
    }

    #[test]
    fn a_missing_source_is_model_not_closed() {
        let table = toml::from_str(
            r#"
            method = "slope_relaxation"
            preserve_topology = true
            star_min_cos = 0.0
            require_star = false
            [grain_diameter]
            value = 1.0
            unit = "m"
            uncertainty = 0.1
            calibration = "test"
            "#,
        )
        .unwrap();
        let err = parse_slope_request(&table, 1.0).unwrap_err();
        assert_eq!(err.kind, "ModelNotClosed");
        assert!(err.detail.contains("source"));
    }

    #[test]
    fn provenance_keeps_unmeasured_coefficients_out_of_the_observational_class() {
        assert_eq!(
            classify_provenance(None, "no Ryugu migration-rate fit", "placeholder").unwrap(),
            "unmeasured"
        );
        assert_eq!(
            classify_provenance(Some("numerical_stability"), "doi:10.1/example", "et al").unwrap(),
            "numerical_stability"
        );
        let bad = classify_provenance(Some("measured"), "doi", "et al").unwrap_err();
        assert_eq!(bad.kind, "ModelNotClosed");
    }

    #[test]
    fn slope_relaxation_locks_a_protected_vertex_and_keeps_the_solid() {
        let mut vertices = TriMesh::unit_cube().vertices;
        vertices[6] = DVec3::new(1.2, 0.4, 1.7);
        let faces = TriMesh::unit_cube().faces;
        let mesh = TriMesh::try_new(vertices, faces.clone()).unwrap();
        let density = crate::common::density::CauchyDensity::uniform(1.0).unwrap();
        let mut mask = CraterMask::unmasked(mesh.vertices.len());
        mask.protection[0] = 1.0;
        let mut request_law = law();
        request_law.friction_angle.value = 0.15;
        request_law.critical_slope.value = 0.15;
        request_law.timestep.value = 0.2;
        let request = SlopeRequest {
            law: request_law,
            preserve_topology: true,
            penalty: None,
            floor_path: None,
            wall_path: None,
            rim_path: None,
        };
        let relaxed =
            relax_slopes(&mesh, &density, &request, &mask, [0.0, 0.0, 0.2], 0.05).unwrap();
        assert_eq!(relaxed.method, "slope_relaxation");
        assert_eq!(relaxed.mesh.faces, faces);
        assert!(relaxed.mesh.signed_volume() > 0.0);
        assert_eq!(relaxed.vertices.len(), mesh.vertices.len());
        assert!(relaxed.vertices[0].displacement_m < 1.0e-8);
        assert_eq!(relaxed.mask.identification, "uncertain");
        assert!(!relaxed.mask.claimed_real_terrain);
        assert!(relaxed.vertices.iter().all(|record| {
            record.slope_before_rad.is_finite()
                && record.slope_after_rad.is_finite()
                && record.curvature_before.is_finite()
                && record.curvature_after.is_finite()
                && record.protection.is_finite()
        }));
        let moved = relaxed
            .vertices
            .iter()
            .any(|record| record.protection < 1.0 && record.displacement_m > 0.0);
        assert!(moved, "an unprotected supercritical vertex should move");
        let phi = request.law.friction_angle.value;
        assert!(relaxed.vertices.iter().all(|record| {
            record.slope_before_rad + 1.0e-8 >= phi
                || record.protection >= 1.0
                || record.displacement_m < 1.0e-8
        }));
    }
}
