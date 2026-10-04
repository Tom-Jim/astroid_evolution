//! Yield classification, rigid-core connectivity, and the surface friction angle.
//! The friction angle uses the trace of the current Stokes velocity, not a radial formula.

use crate::common::math::DVec3;
use crate::scf_simulator::error::ScfError;
use crate::rheo_friction::closure::{
    equivalent_flow_stress, flow_stress, inertial_number, mu_of_inertial,
};
use crate::scf_simulator::yielding::core_fit::{fit_core_motion, slip_velocity};
use crate::scf_simulator::yielding::core_graph::rigid_core;
use crate::scf_simulator::yielding::yield_fn::combined_yield_function;

use super::geometry::SurfaceGeometry;
use super::stokes::{sample_velocity, tet_strain};
use super::volume::TetMesh;

#[derive(Clone, Debug)]
pub struct YieldReport {
    pub yielded: Vec<bool>,
    pub core: Vec<bool>,
    pub isolated: Vec<bool>,
    pub chi: Vec<f64>,
    pub f_delta: Vec<f64>,
    pub core_mass: f64,
    pub core_volume_fraction: f64,
    pub yield_volume_fraction: f64,
    pub mobile_area_fraction: f64,
    pub fit_residual: f64,
    pub max_slip: f64,
    pub slip: Vec<DVec3>,
    pub vertex_yielded: Vec<bool>,
    pub vertex_core: Vec<bool>,
}

pub fn classify(
    volume: &TetMesh,
    velocity: &[DVec3],
    bubble: &[DVec3],
    eta: &[f64],
    pressure: &[f64],
    tide: &[[[f64; 3]; 3]],
    cohesion: f64,
    mu_y: f64,
    surface_vertices: usize,
) -> Result<YieldReport, ScfError> {
    let n = volume.tets.len();
    let mut yielded = vec![false; n];
    let mut f_delta = vec![0.0; n];
    let mut mass = vec![0.0; n];
    for tet in 0..n {
        let strain = tet_strain(volume, velocity, bubble, tet);
        let dev = deviator(&strain);
        let tau = flow_stress(eta[tet], &dev);
        let p = pressure.get(tet).copied().unwrap_or(0.0).max(0.0);
        let tide_tet = tide.get(tet).copied().unwrap_or([[0.0; 3]; 3]);
        f_delta[tet] = combined_yield_function(&tau, &tide_tet, cohesion, mu_y, p);
        yielded[tet] = f_delta[tet] > 0.0;
        mass[tet] = volume.volumes[tet];
        let _ = equivalent_flow_stress(&tau);
    }
    let neighbors = tet_neighbors(volume);
    let core = if yielded.iter().all(|flag| *flag) {
        vec![false; n]
    } else {
        rigid_core(&yielded, &neighbors, &mass)?
    };
    let mut isolated = vec![false; n];
    for tet in 0..n {
        isolated[tet] = !yielded[tet] && !core[tet];
    }
    let total = mass.iter().sum::<f64>().max(1e-30);
    let core_mass: f64 = mass
        .iter()
        .zip(&core)
        .filter(|(_, flag)| **flag)
        .map(|(m, _)| *m)
        .sum();
    let yield_mass: f64 = mass
        .iter()
        .zip(&yielded)
        .filter(|(_, flag)| **flag)
        .map(|(m, _)| *m)
        .sum();
    let (slip, fit_residual, max_slip) = slips(volume, velocity, &core)?;
    let (vertex_yielded, vertex_core, chi) =
        map_vertices(volume, &yielded, &core, surface_vertices);
    let mobile_area = mobile_fraction(volume, &vertex_yielded);
    Ok(YieldReport {
        yielded,
        core,
        isolated,
        chi,
        f_delta,
        core_mass,
        core_volume_fraction: core_mass / total,
        yield_volume_fraction: yield_mass / total,
        mobile_area_fraction: mobile_area,
        fit_residual,
        max_slip,
        slip,
        vertex_yielded,
        vertex_core,
    })
}

pub fn surface_friction(
    faces: &[[u32; 3]],
    volume: &TetMesh,
    velocity: &[DVec3],
    bubble: &[DVec3],
    pressure: &[f64],
    movable: &[bool],
    material_mu_s: f64,
    material_mu_2: f64,
    i_0: f64,
    grain: f64,
    rho_grain: f64,
    geometry: &SurfaceGeometry,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), ScfError> {
    let mut face_phi = vec![0.0; geometry.face_area.len()];
    let mut face_mu = vec![material_mu_s; geometry.face_area.len()];
    let mut face_shear_rate = vec![0.0; geometry.face_area.len()];
    let mut weight = vec![0.0; geometry.face_area.len()];
    for boundary in &volume.boundary {
        if boundary.surface_face >= face_phi.len() {
            continue;
        }
        let ids = faces[boundary.surface_face];
        if !ids
            .iter()
            .any(|index| movable.get(*index as usize).copied().unwrap_or(false))
        {
            continue;
        }
        let corners = boundary
            .vertices
            .map(|index| volume.vertices[index as usize]);
        let speeds = boundary.vertices.map(|index| {
            sample_velocity(
                volume,
                velocity,
                bubble,
                boundary.tet,
                volume.vertices[index as usize],
            )
        });
        let p = pressure.get(boundary.tet).copied().unwrap_or(0.0).max(0.0);
        let (shear, mu) = face_shear(
            &corners,
            &speeds,
            boundary.normal,
            p,
            material_mu_s,
            material_mu_2,
            i_0,
            grain,
            rho_grain,
        );
        face_phi[boundary.surface_face] += mu.max(0.0).atan() * boundary.area;
        face_mu[boundary.surface_face] += mu * boundary.area;
        face_shear_rate[boundary.surface_face] += shear * boundary.area;
        weight[boundary.surface_face] += boundary.area;
    }
    for face in 0..face_phi.len() {
        if weight[face] > 0.0 {
            face_phi[face] /= weight[face];
            face_mu[face] /= weight[face];
            face_shear_rate[face] /= weight[face];
        }
    }
    let vertex_phi = map_face_to_vertices(
        faces,
        &face_phi,
        &geometry.face_area,
        geometry.vertex_normal.len(),
    );
    let mut guarded = vertex_phi;
    for (index, flag) in movable.iter().enumerate() {
        if !*flag {
            guarded[index] = 0.0;
        }
    }
    Ok((guarded, face_mu, face_shear_rate))
}

pub fn map_face_to_vertices(
    faces: &[[u32; 3]],
    face_value: &[f64],
    face_area: &[f64],
    vertices: usize,
) -> Vec<f64> {
    let mut value = vec![0.0; vertices];
    let mut weight = vec![0.0; vertices];
    for (face, ids) in faces.iter().enumerate() {
        for index in ids {
            value[*index as usize] += face_value[face] * face_area[face];
            weight[*index as usize] += face_area[face];
        }
    }
    for index in 0..vertices {
        if weight[index] > 0.0 {
            value[index] /= weight[index];
        }
    }
    value
}

fn face_shear(
    corners: &[DVec3; 3],
    speeds: &[DVec3; 3],
    normal: DVec3,
    pressure: f64,
    mu_s: f64,
    mu_2: f64,
    i_0: f64,
    grain: f64,
    rho: f64,
) -> (f64, f64) {
    let tangent = orthogonal(normal);
    let bitangent = normal.cross(tangent);
    let grad = in_plane_gradient(corners, speeds, tangent, bitangent, normal);
    let shear = (2.0 * dev_dot(&grad)).sqrt();
    let inertial = inertial_number(grain, shear, pressure.max(1e-12), rho.max(1e-12));
    let mu = mu_of_inertial(inertial, mu_s, mu_2, i_0);
    (shear, mu)
}

fn in_plane_gradient(
    corners: &[DVec3; 3],
    speeds: &[DVec3; 3],
    tangent: DVec3,
    bitangent: DVec3,
    normal: DVec3,
) -> [[f64; 3]; 3] {
    let coords = corners.map(|point| [tangent.dot(point), bitangent.dot(point)]);
    let mut gradient = [[0.0; 3]; 3];
    for row in 0..3 {
        let values = speeds.map(|speed| match row {
            0 => speed.x,
            1 => speed.y,
            _ => speed.z,
        });
        let (ds, dt) = plane_derivative(coords, values);
        gradient[row][0] = ds * tangent.x + dt * bitangent.x;
        gradient[row][1] = ds * tangent.y + dt * bitangent.y;
        gradient[row][2] = ds * tangent.z + dt * bitangent.z;
        let normal_component =
            gradient[row][0] * normal.x + gradient[row][1] * normal.y + gradient[row][2] * normal.z;
        gradient[row][0] -= normal_component * normal.x;
        gradient[row][1] -= normal_component * normal.y;
        gradient[row][2] -= normal_component * normal.z;
    }
    gradient
}

fn plane_derivative(coords: [[f64; 2]; 3], values: [f64; 3]) -> (f64, f64) {
    let e1 = [coords[1][0] - coords[0][0], coords[1][1] - coords[0][1]];
    let e2 = [coords[2][0] - coords[0][0], coords[2][1] - coords[0][1]];
    let det = e1[0] * e2[1] - e1[1] * e2[0];
    if det.abs() < 1e-18 {
        return (0.0, 0.0);
    }
    let dv1 = values[1] - values[0];
    let dv2 = values[2] - values[0];
    let ds = (dv1 * e2[1] - dv2 * e1[1]) / det;
    let dt = (e1[0] * dv2 - e2[0] * dv1) / det;
    (ds, dt)
}

fn orthogonal(normal: DVec3) -> DVec3 {
    let trial = if normal.x.abs() < 0.9 {
        DVec3::new(1.0, 0.0, 0.0)
    } else {
        DVec3::new(0.0, 1.0, 0.0)
    };
    normal
        .cross(trial)
        .try_normalize()
        .unwrap_or(DVec3::new(1.0, 0.0, 0.0))
}

fn deviator(strain: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let trace = (strain[0][0] + strain[1][1] + strain[2][2]) / 3.0;
    let mut out = *strain;
    out[0][0] -= trace;
    out[1][1] -= trace;
    out[2][2] -= trace;
    out
}

fn dev_dot(strain: &[[f64; 3]; 3]) -> f64 {
    let dev = deviator(strain);
    let mut sum = 0.0;
    for row in 0..3 {
        for col in 0..3 {
            sum += dev[row][col] * dev[row][col];
        }
    }
    sum
}

fn tet_neighbors(volume: &TetMesh) -> Vec<Vec<usize>> {
    let mut neighbors = vec![Vec::new(); volume.tets.len()];
    for (tet, adj) in volume.neighbors.iter().enumerate() {
        for other in adj {
            if *other >= 0 {
                neighbors[tet].push(*other as usize);
            }
        }
    }
    neighbors
}

fn slips(
    volume: &TetMesh,
    velocity: &[DVec3],
    core: &[bool],
) -> Result<(Vec<DVec3>, f64, f64), ScfError> {
    let mut points = Vec::new();
    let mut speeds = Vec::new();
    for (tet, flag) in core.iter().enumerate() {
        if *flag {
            let center = volume.tets[tet].map(|index| volume.vertices[index as usize]);
            let mid = (center[0] + center[1] + center[2] + center[3]) * 0.25;
            let speed = volume.tets[tet]
                .iter()
                .map(|index| velocity[*index as usize])
                .fold(DVec3::ZERO, |acc, value| acc + value)
                * 0.25;
            points.push(mid.to_array());
            speeds.push(speed.to_array());
        }
    }
    if points.len() < 4 {
        return Ok((vec![DVec3::ZERO; velocity.len()], 0.0, 0.0));
    }
    let motion = match fit_core_motion(&points, &speeds) {
        Ok(motion) => motion,
        Err(err) if err.kind == "CoreDegenerate" => {
            let mut mean = [0.0; 3];
            for speed in &speeds {
                mean[0] += speed[0];
                mean[1] += speed[1];
                mean[2] += speed[2];
            }
            let scale = speeds.len() as f64;
            crate::scf_simulator::yielding::core_fit::CoreMotion {
                translation: [mean[0] / scale, mean[1] / scale, mean[2] / scale],
                rotation: [0.0; 3],
            }
        }
        Err(err) => return Err(err),
    };
    let mut slip = Vec::with_capacity(velocity.len());
    let mut residual = 0.0_f64;
    let mut max_slip = 0.0_f64;
    for (index, speed) in velocity.iter().enumerate() {
        let relative = slip_velocity(volume.vertices[index].to_array(), speed.to_array(), motion);
        let vector = DVec3::from_array(relative);
        residual = residual.max(vector.length());
        max_slip = max_slip.max(vector.length());
        slip.push(vector);
    }
    let fit = points
        .iter()
        .zip(&speeds)
        .map(|(point, speed)| {
            let relative = slip_velocity(*point, *speed, motion);
            DVec3::from_array(relative).length()
        })
        .fold(0.0_f64, f64::max);
    Ok((slip, fit, max_slip.max(residual)))
}

fn map_vertices(
    volume: &TetMesh,
    yielded: &[bool],
    core: &[bool],
    surface_vertices: usize,
) -> (Vec<bool>, Vec<bool>, Vec<f64>) {
    let mut vertex_yielded = vec![false; surface_vertices.max(volume.vertices.len())];
    let mut vertex_core = vec![false; vertex_yielded.len()];
    let mut seen_yield = vec![false; vertex_yielded.len()];
    let mut seen_core = vec![false; vertex_yielded.len()];
    for (tet, flag) in yielded.iter().enumerate() {
        for vertex in volume.tets[tet] {
            let index = vertex as usize;
            if index >= vertex_yielded.len() {
                continue;
            }
            if *flag {
                vertex_yielded[index] = true;
                seen_yield[index] = true;
            }
            if core[tet] {
                vertex_core[index] = true;
                seen_core[index] = true;
            }
        }
    }
    let _ = (seen_yield, seen_core);
    let chi = vertex_yielded
        .iter()
        .map(|flag| if *flag { 1.0 } else { 0.0 })
        .collect();
    (vertex_yielded, vertex_core, chi)
}

fn mobile_fraction(volume: &TetMesh, vertex_yielded: &[bool]) -> f64 {
    let mut mobile = 0.0;
    let mut total = 0.0;
    for face in &volume.boundary {
        total += face.area;
        if face.vertices.iter().any(|index| {
            vertex_yielded
                .get(*index as usize)
                .copied()
                .unwrap_or(false)
        }) {
            mobile += face.area;
        }
    }
    if total > 0.0 { mobile / total } else { 0.0 }
}
