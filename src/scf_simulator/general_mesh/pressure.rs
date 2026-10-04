//! Scalar pressure Poisson on the tetrahedral mesh, followed by the tension cutoff.
//! `p_tilde` is not reported as the exact solution of the unmodified Poisson problem.

use nalgebra::{DMatrix, DVector};

use crate::common::math::DVec3;
use crate::scf_simulator::error::ScfError;

use super::volume::TetMesh;

#[derive(Clone, Debug)]
pub struct PressureProblem<'a> {
    pub density: &'a [f64],
    pub gravity: &'a [DVec3],
    pub tide_divergence: &'a [DVec3],
    pub flow_divergence: &'a [DVec3],
    pub boundary_pressure: &'a [f64],
    pub p_min: f64,
    pub velocity_is_initial: bool,
}

#[derive(Clone, Debug)]
pub struct PressureSolution {
    pub raw: Vec<f64>,
    pub tilde: Vec<f64>,
    pub residual: f64,
    pub boundary_residual: f64,
    pub tension_fraction: f64,
    pub p_min_value: f64,
    pub p_max: f64,
    pub tilde_min: f64,
    pub tilde_max: f64,
    pub initial_velocity: bool,
}

pub fn solve_pressure(
    mesh: &TetMesh,
    problem: &PressureProblem<'_>,
) -> Result<PressureSolution, ScfError> {
    let n = mesh.vertices.len();
    if problem.density.len() != mesh.quad_points.len()
        || problem.gravity.len() != mesh.quad_points.len()
        || problem.tide_divergence.len() != mesh.quad_points.len()
        || problem.flow_divergence.len() != mesh.quad_points.len()
    {
        return Err(ScfError::model_not_closed(
            "pressure sources must be sampled on the tet quadrature",
        ));
    }
    let mut matrix = DMatrix::<f64>::zeros(n, n);
    let mut rhs = DVector::<f64>::zeros(n);
    for (sample, point) in mesh.quad_points.iter().enumerate() {
        let tet = mesh.quad_tet[sample];
        let grads = vertex_grads(mesh, mesh.tets[tet]);
        let lambda = bary(mesh, mesh.tets[tet], *point);
        let weight = mesh.quad_weights[sample];
        let source = problem.gravity[sample] * problem.density[sample]
            - problem.tide_divergence[sample]
            - problem.flow_divergence[sample];
        for a in 0..4 {
            let ia = mesh.tets[tet][a] as usize;
            rhs[ia] += -source.dot(grads[a]) * weight;
            for b in a..4 {
                let ib = mesh.tets[tet][b] as usize;
                let form = grads[a].dot(grads[b]) * weight;
                matrix[(ia, ib)] += form;
                if a != b {
                    matrix[(ib, ia)] += form;
                }
            }
        }
        let _ = lambda;
    }
    for face in &mesh.boundary {
        if problem.boundary_pressure.is_empty() {
            break;
        }
        let prescribed = problem
            .boundary_pressure
            .get(face.surface_face)
            .copied()
            .unwrap_or(0.0);
        if prescribed == 0.0 {
            continue;
        }
        for vertex in face.vertices {
            rhs[vertex as usize] += prescribed * face.area / 3.0;
        }
    }
    let original = matrix.clone();
    let original_rhs = rhs.clone();
    for col in 0..n {
        matrix[(0, col)] = 0.0;
    }
    matrix[(0, 0)] = 1.0;
    rhs[0] = 0.0;
    let raw_vec = matrix
        .lu()
        .solve(&rhs)
        .ok_or_else(|| ScfError::model_not_closed("the pressure Poisson matrix is singular"))?;
    let mut raw: Vec<f64> = raw_vec.iter().copied().collect();
    let mean = raw.iter().sum::<f64>() / n as f64;
    for value in &mut raw {
        *value -= mean;
    }
    let centered = DVector::from_column_slice(&raw);
    let mut poisson = &original * &centered - &original_rhs;
    let poisson_mean = poisson.iter().sum::<f64>() / n as f64;
    for value in poisson.iter_mut() {
        *value -= poisson_mean;
    }
    let poisson_residual = poisson.norm() / original_rhs.norm().max(1.0);
    let mut tension = 0usize;
    let mut tilde = Vec::with_capacity(n);
    for value in &raw {
        if *value < 0.0 {
            tension += 1;
        }
        let positive = value.max(0.0);
        tilde.push(positive.max(problem.p_min));
    }
    let mut boundary_gap = 0.0;
    let mut boundary_scale = 0.0;
    if !problem.boundary_pressure.is_empty() {
        for face in &mesh.boundary {
            let prescribed = problem
                .boundary_pressure
                .get(face.surface_face)
                .copied()
                .unwrap_or(0.0);
            let mut average = 0.0;
            for vertex in face.vertices {
                average += raw[vertex as usize];
            }
            average /= 3.0;
            boundary_gap += (average - prescribed).abs() * face.area;
            boundary_scale += face.area * prescribed.abs().max(1.0);
        }
    }
    let p_min_value = raw.iter().copied().fold(f64::INFINITY, f64::min);
    let p_max = raw.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let tilde_min = tilde.iter().copied().fold(f64::INFINITY, f64::min);
    let tilde_max = tilde.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !p_min_value.is_finite() || !tilde_max.is_finite() {
        return Err(ScfError::model_not_closed("pressure is not finite"));
    }
    Ok(PressureSolution {
        raw,
        tilde,
        residual: poisson_residual,
        boundary_residual: boundary_gap / boundary_scale.max(1e-30),
        tension_fraction: tension as f64 / n as f64,
        p_min_value,
        p_max,
        tilde_min,
        tilde_max,
        initial_velocity: problem.velocity_is_initial,
    })
}

fn vertex_grads(mesh: &TetMesh, tet: [u32; 4]) -> [DVec3; 4] {
    let center = {
        let p = tet.map(|index| mesh.vertices[index as usize]);
        (p[0] + p[1] + p[2] + p[3]) * 0.25
    };
    let base = bary(mesh, tet, center);
    let scale = mesh
        .vertices
        .iter()
        .map(|vertex| vertex.length())
        .fold(1.0_f64, f64::max)
        .max(1.0);
    let eps = 1e-6 * scale;
    let mut grads = [DVec3::ZERO; 4];
    for axis in 0..3 {
        let mut shifted = center;
        match axis {
            0 => shifted.x += eps,
            1 => shifted.y += eps,
            _ => shifted.z += eps,
        }
        let next = bary(mesh, tet, shifted);
        for local in 0..4 {
            let slope = (next[local] - base[local]) / eps;
            match axis {
                0 => grads[local].x = slope,
                1 => grads[local].y = slope,
                _ => grads[local].z = slope,
            }
        }
    }
    grads
}

fn bary(mesh: &TetMesh, tet: [u32; 4], point: DVec3) -> [f64; 4] {
    let p = tet.map(|index| mesh.vertices[index as usize]);
    let vol = |a: DVec3, b: DVec3, c: DVec3, d: DVec3| (b - a).dot((c - a).cross(d - a));
    let total = vol(p[0], p[1], p[2], p[3]);
    if total == 0.0 {
        return [0.25; 4];
    }
    let b0 = vol(point, p[1], p[2], p[3]) / total;
    let b1 = vol(p[0], point, p[2], p[3]) / total;
    let b2 = vol(p[0], p[1], point, p[3]) / total;
    [b0, b1, b2, 1.0 - b0 - b1 - b2]
}
