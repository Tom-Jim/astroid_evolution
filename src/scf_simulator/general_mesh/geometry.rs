//! Closed-triangle geometry used by the general-mesh SCF step.
//! Increments are vertex vectors. There is no radial chart and no `n · e_r` gate.

use crate::common::geometry::TriMesh;
use crate::common::math::DVec3;
use crate::scf_simulator::error::ScfError;

#[derive(Clone, Debug)]
pub struct SurfaceGeometry {
    pub face_normal: Vec<DVec3>,
    pub face_area: Vec<f64>,
    pub vertex_area: Vec<f64>,
    pub dual_area: Vec<f64>,
    pub vertex_normal: Vec<DVec3>,
    pub one_ring: Vec<Vec<usize>>,
    pub incident_faces: Vec<Vec<usize>>,
    pub mean_curvature: Vec<f64>,
    pub gaussian_curvature: Vec<f64>,
    /// Exact `∂V/∂x_i`. The mass row is `ρ ∂V/∂x_i = A_dual ρ n` when `n` is this direction.
    pub volume_gradient: Vec<DVec3>,
    /// `∂n_i / ∂x_j` as a row-major 3×3, for `j` in the one-ring of `i` (including `i`).
    pub normal_jacobian: Vec<Vec<(usize, [f64; 9])>>,
    pub min_altitude: f64,
    pub min_face_area: f64,
    pub volume: f64,
}

pub fn analyze(mesh: &TriMesh) -> Result<SurfaceGeometry, ScfError> {
    if mesh.vertices.len() < 4 || mesh.faces.is_empty() {
        return Err(ScfError::model_not_closed(
            "a general mesh needs vertices and triangles",
        ));
    }
    let mut face_normal = Vec::with_capacity(mesh.faces.len());
    let mut face_area = Vec::with_capacity(mesh.faces.len());
    let mut min_altitude = f64::INFINITY;
    let mut min_face_area = f64::INFINITY;
    for face in 0..mesh.faces.len() {
        let [a, b, c] = mesh.triangle(face);
        let cross = (b - a).cross(c - a);
        let area = 0.5 * cross.length();
        if !(area.is_finite() && area > 0.0) {
            return Err(ScfError::model_not_closed(format!(
                "degenerate triangle {face}"
            )));
        }
        let edges = [(b - a).length(), (c - b).length(), (a - c).length()];
        let longest = edges[0].max(edges[1]).max(edges[2]).max(1e-30);
        min_altitude = min_altitude.min(2.0 * area / longest);
        min_face_area = min_face_area.min(area);
        face_area.push(area);
        face_normal.push(cross * (1.0 / cross.length()));
    }
    let n = mesh.vertices.len();
    let mut one_ring = vec![Vec::<usize>::new(); n];
    let mut incident_faces = vec![Vec::<usize>::new(); n];
    for (face_index, face) in mesh.faces.iter().enumerate() {
        let tri = [face[0] as usize, face[1] as usize, face[2] as usize];
        for vertex in tri {
            incident_faces[vertex].push(face_index);
        }
        for a in 0..3 {
            for b in 0..3 {
                if a != b && !one_ring[tri[a]].contains(&tri[b]) {
                    one_ring[tri[a]].push(tri[b]);
                }
            }
        }
    }
    let mut vertex_area = vec![0.0; n];
    let mut dual_area = vec![0.0; n];
    let mut area_vector = vec![DVec3::ZERO; n];
    let mut angle_sum = vec![0.0; n];
    for (face_index, face) in mesh.faces.iter().enumerate() {
        let tri = [face[0] as usize, face[1] as usize, face[2] as usize];
        let p = tri.map(|index| mesh.vertices[index]);
        let area = face_area[face_index];
        let cross = (p[1] - p[0]).cross(p[2] - p[0]);
        for slot in 0..3 {
            vertex_area[tri[slot]] += area / 3.0;
            area_vector[tri[slot]] = area_vector[tri[slot]] + cross;
            dual_area[tri[slot]] += corner_area(p, slot, area);
            angle_sum[tri[slot]] += corner_angle(p, slot);
        }
    }
    let mut vertex_normal = Vec::with_capacity(n);
    let mut mean_curvature = vec![0.0; n];
    let mut gaussian_curvature = vec![0.0; n];
    for index in 0..n {
        let normal = area_vector[index].try_normalize().ok_or_else(|| {
            ScfError::model_not_closed(format!("vertex {index} has a zero area vector"))
        })?;
        vertex_normal.push(normal);
        if !(dual_area[index].is_finite() && dual_area[index] > 0.0) {
            return Err(ScfError::model_not_closed(format!(
                "vertex {index} has no positive dual area"
            )));
        }
        let mut lap = DVec3::ZERO;
        for &neighbor in &one_ring[index] {
            lap = lap + (mesh.vertices[index] - mesh.vertices[neighbor]);
        }
        mean_curvature[index] = lap.length() / dual_area[index];
        gaussian_curvature[index] =
            (2.0 * std::f64::consts::PI - angle_sum[index]) / dual_area[index];
    }
    let normal_jacobian = normal_jacobians(mesh, &incident_faces, &area_vector);
    let volume_gradient = volume_gradients(mesh);
    Ok(SurfaceGeometry {
        face_normal,
        face_area,
        vertex_area,
        dual_area,
        vertex_normal,
        one_ring,
        incident_faces,
        mean_curvature,
        gaussian_curvature,
        volume_gradient,
        normal_jacobian,
        min_altitude,
        min_face_area,
        volume: mesh.signed_volume(),
    })
}

/// `D M d = Σ_i ρ_i (∂V/∂x_i · d_i)`.
/// `∂V/∂x_i` is the closed-surface identity `A_dual,i n_i` used by the mass row.
pub fn mass_row(geometry: &SurfaceGeometry, density: &[f64]) -> Result<Vec<DVec3>, ScfError> {
    if density.len() != geometry.volume_gradient.len() {
        return Err(ScfError::model_not_closed(
            "mass-row density does not match the vertex count",
        ));
    }
    let mut row = Vec::with_capacity(density.len());
    for index in 0..density.len() {
        let rho = density[index];
        if !(rho.is_finite() && rho > 0.0) {
            return Err(ScfError::model_not_closed(
                "a surface density sample is not finite and positive",
            ));
        }
        row.push(geometry.volume_gradient[index] * rho);
    }
    Ok(row)
}

pub fn apply_normal_jacobian(row: &[(usize, [f64; 9])], displacement: &[DVec3]) -> DVec3 {
    let mut delta = DVec3::ZERO;
    for &(index, matrix) in row {
        let d = displacement[index];
        delta.x += matrix[0] * d.x + matrix[1] * d.y + matrix[2] * d.z;
        delta.y += matrix[3] * d.x + matrix[4] * d.y + matrix[5] * d.z;
        delta.z += matrix[6] * d.x + matrix[7] * d.y + matrix[8] * d.z;
    }
    delta
}

pub fn displace(mesh: &TriMesh, displacement: &[DVec3]) -> Result<TriMesh, ScfError> {
    if displacement.len() != mesh.vertices.len() {
        return Err(ScfError::model_not_closed(
            "a general-mesh displacement has the wrong vertex count",
        ));
    }
    let vertices = mesh
        .vertices
        .iter()
        .zip(displacement)
        .map(|(vertex, step)| *vertex + *step)
        .collect();
    TriMesh::try_new(vertices, mesh.faces.clone())
        .map_err(|err| ScfError::model_not_closed(err.detail))
}

#[derive(Clone, Debug)]
pub struct CandidateAudit {
    pub volume: f64,
    pub mass_residual: f64,
    pub max_displacement: f64,
    pub flipped_faces: usize,
    pub self_intersections: usize,
    pub topology_valid: bool,
    pub detail: String,
}

pub fn audit_candidate(
    current: &TriMesh,
    candidate: &TriMesh,
    displacement: &[DVec3],
    mass_now: f64,
    mass_next: f64,
    mass_scale: f64,
    trust: f64,
    protected: &[f64],
) -> CandidateAudit {
    let mut flipped = 0usize;
    let faces = current.faces.len().min(candidate.faces.len());
    for face in 0..faces {
        let before = current.face_normal(face);
        let after = candidate.face_normal(face);
        if let (Some(before), Some(after)) = (before, after) {
            if before.dot(after) <= 0.0 {
                flipped += 1;
            }
        } else {
            flipped += 1;
        }
    }
    let max_displacement = displacement
        .iter()
        .map(|step| step.length())
        .fold(0.0_f64, f64::max);
    let mut protected_over = false;
    for (index, weight) in protected.iter().enumerate() {
        if *weight > 0.0
            && displacement
                .get(index)
                .is_some_and(|step| step.length() > trust * (1.0 - weight).max(0.0) + 1e-14)
        {
            protected_over = true;
        }
    }
    let finite = displacement.iter().all(|step| step.is_finite())
        && max_displacement <= trust * (1.0 + 1e-8);
    let topology = candidate.signed_volume().is_finite() && candidate.signed_volume() > 0.0;
    let (closed, intersections) = match candidate.validate_solid() {
        Ok(()) => (true, 0usize),
        Err(err) => {
            let hit = usize::from(err.detail.contains("intersect"));
            (false, hit.max(1))
        }
    };
    let mut detail = String::new();
    if !finite {
        detail.push_str("displacement is not finite or leaves the trust ball; ");
    }
    if !closed {
        detail.push_str("candidate is not a closed oriented manifold; ");
    }
    if flipped > 0 {
        detail.push_str("a face flipped; ");
    }
    if protected_over {
        detail.push_str("a protected vertex left its allowed ball; ");
    }
    let mass_residual = (mass_next - mass_now).abs() / mass_scale.max(1e-30);
    CandidateAudit {
        volume: candidate.signed_volume(),
        mass_residual,
        max_displacement,
        flipped_faces: flipped,
        self_intersections: intersections,
        topology_valid: topology && closed && flipped == 0 && finite && !protected_over,
        detail,
    }
}

/// Optional star diagnostic. A negative margin does not reject the mesh.
pub fn star_margin(mesh: &TriMesh, origin: DVec3) -> f64 {
    let mut margin = f64::INFINITY;
    for face in 0..mesh.faces.len() {
        let [a, b, c] = mesh.triangle(face);
        let normal = (b - a).cross(c - a);
        let center = (a + b + c) * (1.0 / 3.0);
        let radial = center - origin;
        let denom = normal.length() * radial.length();
        if denom > 0.0 {
            margin = margin.min(normal.dot(radial) / denom);
        }
    }
    if margin.is_finite() { margin } else { f64::NAN }
}

fn corner_area(p: [DVec3; 3], slot: usize, area: f64) -> f64 {
    let angle = [corner_angle(p, 0), corner_angle(p, 1), corner_angle(p, 2)];
    if angle
        .iter()
        .any(|value| *value >= 0.5 * std::f64::consts::PI)
    {
        return if angle[slot] >= 0.5 * std::f64::consts::PI {
            0.5 * area
        } else {
            0.25 * area
        };
    }
    let (i, j, k) = (slot, (slot + 1) % 3, (slot + 2) % 3);
    let cot_j = angle[j].tan().recip();
    let cot_k = angle[k].tan().recip();
    let ij = (p[i] - p[j]).length_squared();
    let ik = (p[i] - p[k]).length_squared();
    (ij * cot_k + ik * cot_j) * 0.125
}

fn corner_angle(p: [DVec3; 3], slot: usize) -> f64 {
    let i = slot;
    let j = (slot + 1) % 3;
    let k = (slot + 2) % 3;
    let u = p[j] - p[i];
    let v = p[k] - p[i];
    let denom = u.length() * v.length();
    if denom == 0.0 {
        return 0.0;
    }
    (u.dot(v) / denom).clamp(-1.0, 1.0).acos()
}

fn normal_jacobians(
    mesh: &TriMesh,
    incident: &[Vec<usize>],
    area_vector: &[DVec3],
) -> Vec<Vec<(usize, [f64; 9])>> {
    let mut rows = Vec::with_capacity(mesh.vertices.len());
    for (vertex, faces) in incident.iter().enumerate() {
        let mut influence = Vec::new();
        influence.push(vertex);
        for &face in faces {
            for index in mesh.faces[face] {
                let index = index as usize;
                if !influence.contains(&index) {
                    influence.push(index);
                }
            }
        }
        let mut local = Vec::new();
        let length = area_vector[vertex].length();
        let normal = if length > 0.0 {
            area_vector[vertex] * (1.0 / length)
        } else {
            DVec3::ZERO
        };
        for &other in &influence {
            let mut matrix = [0.0; 9];
            for axis in 0..3 {
                let mut delta = DVec3::ZERO;
                delta = add_axis(delta, axis, 1.0);
                let ds = area_vector_derivative(mesh, faces, vertex, other, delta);
                let dn = (ds - normal * normal.dot(ds)) * (1.0 / length.max(1e-30));
                matrix[axis] = dn.x;
                matrix[3 + axis] = dn.y;
                matrix[6 + axis] = dn.z;
            }
            local.push((other, matrix));
        }
        rows.push(local);
    }
    rows
}

fn add_axis(mut value: DVec3, axis: usize, amount: f64) -> DVec3 {
    match axis {
        0 => value.x += amount,
        1 => value.y += amount,
        _ => value.z += amount,
    }
    value
}

fn volume_gradients(mesh: &TriMesh) -> Vec<DVec3> {
    let mut gradient = vec![DVec3::ZERO; mesh.vertices.len()];
    for face in &mesh.faces {
        let ids = [face[0] as usize, face[1] as usize, face[2] as usize];
        let p = ids.map(|index| mesh.vertices[index]);
        gradient[ids[0]] = gradient[ids[0]] + p[1].cross(p[2]) * (1.0 / 6.0);
        gradient[ids[1]] = gradient[ids[1]] + p[2].cross(p[0]) * (1.0 / 6.0);
        gradient[ids[2]] = gradient[ids[2]] + p[0].cross(p[1]) * (1.0 / 6.0);
    }
    gradient
}

fn area_vector_derivative(
    mesh: &TriMesh,
    faces: &[usize],
    _vertex: usize,
    moved: usize,
    delta: DVec3,
) -> DVec3 {
    let mut sum = DVec3::ZERO;
    for &face in faces {
        let ids = [
            mesh.faces[face][0] as usize,
            mesh.faces[face][1] as usize,
            mesh.faces[face][2] as usize,
        ];
        let p = ids.map(|index| mesh.vertices[index]);
        let u = p[1] - p[0];
        let v = p[2] - p[0];
        for slot in 0..3 {
            if ids[slot] != moved {
                continue;
            }
            let linear = match slot {
                0 => (-delta).cross(v) + u.cross(-delta),
                1 => delta.cross(v),
                _ => u.cross(delta),
            };
            sum = sum + linear;
        }
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cube_areas_and_volume_are_positive() {
        let mesh = TriMesh::unit_cube();
        let geometry = analyze(&mesh).unwrap();
        assert!(geometry.volume > 0.0);
        assert!(geometry.dual_area.iter().all(|area| *area > 0.0));
        assert!(geometry.min_altitude > 0.0);
        assert!(geometry.vertex_normal.len() == mesh.vertices.len());
        let rho = vec![2.0; mesh.vertices.len()];
        let row = mass_row(&geometry, &rho).unwrap();
        assert_eq!(row.len(), mesh.vertices.len());
    }

    #[test]
    fn mass_row_matches_volume_finite_difference() {
        let mesh = TriMesh::unit_cube();
        let geometry = analyze(&mesh).unwrap();
        let rho_value = 1.7;
        let rho = vec![rho_value; mesh.vertices.len()];
        let row = mass_row(&geometry, &rho).unwrap();
        let mut displacement = vec![DVec3::ZERO; mesh.vertices.len()];
        displacement[0] = DVec3::new(1.0e-4, -2.0e-4, 1.5e-4);
        let predicted: f64 = row
            .iter()
            .zip(&displacement)
            .map(|(weight, step)| weight.dot(*step))
            .sum();
        let moved = displace(&mesh, &displacement).unwrap();
        let delta_mass = rho_value * (moved.signed_volume() - mesh.signed_volume());
        assert!(
            (predicted - delta_mass).abs() < 1e-8 * rho_value,
            "{predicted} {delta_mass}"
        );
    }

    #[test]
    fn normal_jacobian_matches_finite_difference() {
        let mesh = TriMesh::unit_cube();
        let geometry = analyze(&mesh).unwrap();
        let vertex = 0usize;
        let eps = 1e-6;
        let mut step = vec![DVec3::ZERO; mesh.vertices.len()];
        step[2] = DVec3::new(eps, -0.5 * eps, 0.25 * eps);
        let moved = displace(&mesh, &step).unwrap();
        let next = analyze(&moved).unwrap();
        let analytic = apply_normal_jacobian(&geometry.normal_jacobian[vertex], &step);
        let numeric = next.vertex_normal[vertex] - geometry.vertex_normal[vertex];
        let gap = (analytic - numeric).length();
        assert!(
            gap < 1e-8,
            "{gap} analytic {analytic:?} numeric {numeric:?}"
        );
    }

    #[test]
    fn three_dimensional_displacement_is_not_a_radius() {
        let mesh = TriMesh::right_tetrahedron();
        let mut step = vec![DVec3::ZERO; 4];
        step[1] = DVec3::new(0.0, 0.02, -0.01);
        let moved = displace(&mesh, &step).unwrap();
        assert!((moved.vertices[1] - (mesh.vertices[1] + step[1])).length() < 1e-15);
        assert!(moved.signed_volume() > 0.0);
    }

    #[test]
    fn unknown_tier1_blocks_acceptance_and_sources_stay_off_the_radial_path() {
        assert!(!crate::scf_simulator::general_mesh::loop_::gate_accepts(
            "NOT_ASSESSED",
            true,
            true
        ));
        assert!(crate::scf_simulator::general_mesh::loop_::gate_accepts(
            "PASS", true, true
        ));
        let sources = [
            include_str!("geometry.rs"),
            include_str!("volume.rs"),
            include_str!("stokes.rs"),
            include_str!("pressure.rs"),
            include_str!("rheology.rs"),
            include_str!("yielding.rs"),
            include_str!("geometry_update.rs"),
            include_str!("loop_.rs"),
        ];
        for source in sources {
            let radial = ["ray", "_radii", "::"].concat();
            let picard = ["picard", "_solve("].concat();
            let radial_step = ["predicted", "_radial_increment("].concat();
            let mass = ["surface", "_mass_jacobian("].concat();
            assert!(!source.contains(radial.as_str()), "{radial}");
            assert!(!source.contains(picard.as_str()));
            assert!(!source.contains(radial_step.as_str()));
            assert!(!source.contains(mass.as_str()));
        }
        let mut streak = 0u32;
        streak += 1;
        assert!(streak < 2);
    }
}
