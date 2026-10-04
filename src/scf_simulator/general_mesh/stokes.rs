//! Mixed MINI Stokes on a tetrahedral mesh: P1 plus a cubic bubble, P0 pressure,
//! and six rigid-body constraints. The discrete divergence is the weak form
//! `∫ q div v`, not a nodal finite-difference stencil.

use nalgebra::{DMatrix, DVector};

use crate::common::math::DVec3;
use crate::scf_simulator::error::ScfError;

use super::volume::TetMesh;

#[derive(Clone, Debug)]
pub struct StokesInput<'a> {
    pub eta: &'a [f64],
    pub force: &'a [DVec3],
    pub vertex_mass: &'a [f64],
    pub boundary_stress: Option<[[f64; 3]; 3]>,
}

#[derive(Clone, Debug)]
pub struct StokesSolution {
    pub vertex_velocity: Vec<DVec3>,
    pub bubble: Vec<DVec3>,
    pub pressure: Vec<f64>,
    pub rigid_multiplier: [f64; 6],
    pub linear_residual: f64,
    pub divergence_residual: f64,
    pub rigid_residual: f64,
    pub iterations: usize,
    pub diagonal_condition: f64,
    pub rigid_rank: usize,
    pub sigma_min: f64,
    pub velocity_finite: bool,
    pub pressure_finite: bool,
}

pub fn solve_stokes(mesh: &TetMesh, input: &StokesInput<'_>) -> Result<StokesSolution, ScfError> {
    if input.eta.len() != mesh.tets.len() || input.force.len() != mesh.tets.len() {
        return Err(ScfError::model_not_closed(
            "Stokes viscosity and force must be given on every tetrahedron",
        ));
    }
    if input.vertex_mass.len() != mesh.vertices.len() {
        return Err(ScfError::model_not_closed(
            "rigid-mass weights must match the tet vertices",
        ));
    }
    if input.eta.iter().any(|eta| !(eta.is_finite() && *eta > 0.0)) {
        return Err(ScfError::model_not_closed(
            "Stokes viscosity must be finite and positive on every tetrahedron",
        ));
    }
    let (rank, sigma_min) = rigid_spectrum(mesh, input.vertex_mass);
    if rank < 6 {
        return Err(ScfError::rigid_rank_deficient(format!(
            "rank(B_rb) {rank} sigma_min {sigma_min}"
        )));
    }
    let nv = mesh.vertices.len();
    let nt = mesh.tets.len();
    let n_vel = 3 * nv + 3 * nt;
    let n = n_vel + nt + 6;
    let mut matrix = DMatrix::<f64>::zeros(n, n);
    let mut rhs = DVector::<f64>::zeros(n);
    let mut diagonal_max = 0.0_f64;
    let mut diagonal_min = f64::INFINITY;
    for (tet_index, tet) in mesh.tets.iter().enumerate() {
        assemble_tet(
            mesh,
            *tet,
            tet_index,
            input.eta[tet_index],
            input.force[tet_index],
            n_vel,
            &mut matrix,
            &mut rhs,
            &mut diagonal_max,
            &mut diagonal_min,
        );
    }
    if let Some(stress) = input.boundary_stress {
        assemble_traction(mesh, stress, &mut rhs);
    }
    assemble_rigid(mesh, input.vertex_mass, n_vel + nt, &mut matrix);
    let solved = matrix
        .clone()
        .lu()
        .solve(&rhs)
        .ok_or_else(|| ScfError::model_not_closed("the mixed Stokes matrix is singular"))?;
    let residual_vec = &matrix * &solved - &rhs;
    let linear_residual = residual_vec.norm() / rhs.norm().max(1e-30);
    let mut vertex_velocity = vec![DVec3::ZERO; nv];
    for vertex in 0..nv {
        vertex_velocity[vertex] = DVec3::new(
            solved[3 * vertex],
            solved[3 * vertex + 1],
            solved[3 * vertex + 2],
        );
    }
    let mut bubble = vec![DVec3::ZERO; nt];
    for tet in 0..nt {
        let base = 3 * nv + 3 * tet;
        bubble[tet] = DVec3::new(solved[base], solved[base + 1], solved[base + 2]);
    }
    let pressure: Vec<f64> = (0..nt).map(|tet| solved[n_vel + tet]).collect();
    let mut rigid_multiplier = [0.0; 6];
    for mode in 0..6 {
        rigid_multiplier[mode] = solved[n_vel + nt + mode];
    }
    let divergence_residual = divergence_norm(mesh, &vertex_velocity, &bubble);
    let rigid_residual = rigid_norm(mesh, input.vertex_mass, &vertex_velocity);
    let velocity_finite = vertex_velocity
        .iter()
        .chain(bubble.iter())
        .all(|v| v.is_finite());
    let pressure_finite = pressure.iter().all(|p| p.is_finite());
    if !(linear_residual.is_finite()
        && divergence_residual.is_finite()
        && rigid_residual.is_finite())
    {
        return Err(ScfError::model_not_closed(
            "Stokes residuals are not finite",
        ));
    }
    if !velocity_finite || !pressure_finite {
        return Err(ScfError::model_not_closed(
            "Stokes velocity or pressure is not finite",
        ));
    }
    Ok(StokesSolution {
        vertex_velocity,
        bubble,
        pressure,
        rigid_multiplier,
        linear_residual,
        divergence_residual,
        rigid_residual,
        iterations: 1,
        diagonal_condition: diagonal_max / diagonal_min.max(1e-30),
        rigid_rank: rank,
        sigma_min,
        velocity_finite,
        pressure_finite,
    })
}

pub fn divergence_norm(mesh: &TetMesh, velocity: &[DVec3], bubble: &[DVec3]) -> f64 {
    let mut accum = 0.0;
    let mut scale = 0.0;
    for (tet_index, tet) in mesh.tets.iter().enumerate() {
        let grads = shape_gradients(mesh, *tet);
        let mut div = 0.0;
        for local in 0..4 {
            div += velocity[tet[local] as usize].dot(grads[local]);
        }
        let center = centroid(mesh, *tet);
        let lambda = barycentric(mesh, *tet, center);
        let grad_beta = bubble_grad(&grads, lambda);
        div += bubble[tet_index].dot(grad_beta);
        accum += div * div * mesh.volumes[tet_index];
        scale += velocity[tet[0] as usize].length_squared() * mesh.volumes[tet_index];
    }
    (accum.sqrt()) / scale.sqrt().max(1e-30)
}

pub fn tet_strain(
    mesh: &TetMesh,
    velocity: &[DVec3],
    bubble: &[DVec3],
    tet_index: usize,
) -> [[f64; 3]; 3] {
    let tet = mesh.tets[tet_index];
    let grads = shape_gradients(mesh, tet);
    let center = centroid(mesh, tet);
    let lambda = barycentric(mesh, tet, center);
    let mut gradient = [[0.0; 3]; 3];
    for local in 0..4 {
        let speed = velocity[tet[local] as usize];
        for axis in 0..3 {
            let component = match axis {
                0 => speed.x,
                1 => speed.y,
                _ => speed.z,
            };
            gradient[axis][0] += component * grads[local].x;
            gradient[axis][1] += component * grads[local].y;
            gradient[axis][2] += component * grads[local].z;
        }
    }
    let grad_beta = bubble_grad(&grads, lambda);
    let speed = bubble[tet_index];
    for axis in 0..3 {
        let component = match axis {
            0 => speed.x,
            1 => speed.y,
            _ => speed.z,
        };
        gradient[axis][0] += component * grad_beta.x;
        gradient[axis][1] += component * grad_beta.y;
        gradient[axis][2] += component * grad_beta.z;
    }
    symmetric(gradient)
}

pub fn sample_velocity(
    mesh: &TetMesh,
    velocity: &[DVec3],
    bubble: &[DVec3],
    tet: usize,
    point: DVec3,
) -> DVec3 {
    let grads = shape_gradients(mesh, mesh.tets[tet]);
    let lambda = barycentric(mesh, mesh.tets[tet], point);
    let mut speed = bubble[tet] * bubble_value(lambda);
    for local in 0..4 {
        speed = speed + velocity[mesh.tets[tet][local] as usize] * lambda[local];
    }
    let _ = grads;
    speed
}

fn assemble_tet(
    mesh: &TetMesh,
    tet: [u32; 4],
    tet_index: usize,
    eta: f64,
    force: DVec3,
    n_vel: usize,
    matrix: &mut DMatrix<f64>,
    rhs: &mut DVector<f64>,
    diagonal_max: &mut f64,
    diagonal_min: &mut f64,
) {
    if !(eta.is_finite() && eta > 0.0) {
        return;
    }
    let grads = shape_gradients(mesh, tet);
    let dofs = tet_dofs(mesh.vertices.len(), tet, tet_index);
    let alpha = 0.585_410_196_624_968_5;
    let beta = 0.138_196_601_125_010_5;
    let bary = [
        [alpha, beta, beta, beta],
        [beta, alpha, beta, beta],
        [beta, beta, alpha, beta],
        [beta, beta, beta, alpha],
    ];
    let weight = mesh.volumes[tet_index] / 4.0;
    for coords in bary {
        let lambda = coords;
        let grad_beta = bubble_grad(&grads, lambda);
        let beta_value = bubble_value(lambda);
        let mut strain = Vec::with_capacity(15);
        for local in 0..4 {
            for axis in 0..3 {
                strain.push(basis_strain(grads[local], axis, 0.0, DVec3::ZERO));
            }
        }
        for axis in 0..3 {
            strain.push(basis_strain(DVec3::ZERO, axis, beta_value, grad_beta));
        }
        for a in 0..15 {
            let load = match a {
                0..=11 => {
                    let local = a / 3;
                    let axis = a % 3;
                    force_component(force, axis) * lambda[local]
                }
                _ => force_component(force, a - 12) * beta_value,
            };
            rhs[dofs[a]] += weight * load;
            let div_a = strain_div(&strain[a]);
            matrix[(dofs[a], n_vel + tet_index)] += -div_a * weight;
            matrix[(n_vel + tet_index, dofs[a])] += -div_a * weight;
            for b in a..15 {
                let form = 2.0 * eta * weight * double_dot(&strain[a], &strain[b]);
                matrix[(dofs[a], dofs[b])] += form;
                if a != b {
                    matrix[(dofs[b], dofs[a])] += form;
                }
                if a == b {
                    *diagonal_max = diagonal_max.max(matrix[(dofs[a], dofs[a])].abs());
                    if matrix[(dofs[a], dofs[a])].abs() > 0.0 {
                        *diagonal_min = diagonal_min.min(matrix[(dofs[a], dofs[a])].abs());
                    }
                }
            }
        }
    }
}

fn assemble_traction(mesh: &TetMesh, stress: [[f64; 3]; 3], rhs: &mut DVector<f64>) {
    for face in &mesh.boundary {
        let traction = mat_vec(stress, face.normal);
        let share = face.area / 3.0;
        for vertex in face.vertices {
            let base = 3 * vertex as usize;
            rhs[base] += share * traction.x;
            rhs[base + 1] += share * traction.y;
            rhs[base + 2] += share * traction.z;
        }
    }
}

fn assemble_rigid(mesh: &TetMesh, mass: &[f64], lambda_row: usize, matrix: &mut DMatrix<f64>) {
    let origin = mass_center(mesh, mass);
    for (vertex, &weight) in mass.iter().enumerate() {
        if weight == 0.0 {
            continue;
        }
        let arm = mesh.vertices[vertex] - origin;
        let base = 3 * vertex;
        for axis in 0..3 {
            matrix[(base + axis, lambda_row + axis)] += weight;
            matrix[(lambda_row + axis, base + axis)] += weight;
        }
        let rotational = [
            [0.0, -arm.z, arm.y],
            [arm.z, 0.0, -arm.x],
            [-arm.y, arm.x, 0.0],
        ];
        for row in 0..3 {
            for col in 0..3 {
                let value = weight * rotational[row][col];
                if value != 0.0 {
                    matrix[(base + col, lambda_row + 3 + row)] += value;
                    matrix[(lambda_row + 3 + row, base + col)] += value;
                }
            }
        }
    }
}

fn rigid_spectrum(mesh: &TetMesh, mass: &[f64]) -> (usize, f64) {
    let origin = mass_center(mesh, mass);
    let mut columns = Vec::new();
    for (vertex, &weight) in mass.iter().enumerate() {
        let arm = mesh.vertices[vertex] - origin;
        let modes = [
            DVec3::new(weight, 0.0, 0.0),
            DVec3::new(0.0, weight, 0.0),
            DVec3::new(0.0, 0.0, weight),
            DVec3::new(0.0, weight * arm.z, -weight * arm.y),
            DVec3::new(-weight * arm.z, 0.0, weight * arm.x),
            DVec3::new(weight * arm.y, -weight * arm.x, 0.0),
        ];
        if columns.is_empty() {
            columns = vec![Vec::new(); 6];
        }
        for mode in 0..6 {
            columns[mode].push(modes[mode].x);
            columns[mode].push(modes[mode].y);
            columns[mode].push(modes[mode].z);
        }
    }
    if columns.is_empty() {
        return (0, 0.0);
    }
    let rows = columns[0].len();
    let mut gram = DMatrix::<f64>::zeros(6, 6);
    for a in 0..6 {
        for b in a..6 {
            let dot = columns[a]
                .iter()
                .zip(&columns[b])
                .map(|(left, right)| left * right)
                .sum::<f64>();
            gram[(a, b)] = dot;
            gram[(b, a)] = dot;
        }
    }
    let _ = rows;
    let eigenvalues = gram.symmetric_eigenvalues();
    let mut values: Vec<f64> = eigenvalues
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect();
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let sigma_max = values.last().copied().unwrap_or(0.0).sqrt();
    let sigma_min = values.first().copied().unwrap_or(0.0).max(0.0).sqrt();
    let rank = values
        .iter()
        .filter(|value| value.sqrt() > 1e-8 * sigma_max.max(1.0))
        .count();
    (rank, sigma_min)
}

fn rigid_norm(mesh: &TetMesh, mass: &[f64], velocity: &[DVec3]) -> f64 {
    let origin = mass_center(mesh, mass);
    let mut linear = DVec3::ZERO;
    let mut angular = DVec3::ZERO;
    let mut scale = 0.0;
    for (vertex, &weight) in mass.iter().enumerate() {
        linear = linear + velocity[vertex] * weight;
        angular = angular + (mesh.vertices[vertex] - origin).cross(velocity[vertex]) * weight;
        scale += weight * velocity[vertex].length();
    }
    (linear.length() + angular.length()) / scale.max(1e-30)
}

fn mass_center(mesh: &TetMesh, mass: &[f64]) -> DVec3 {
    let mut weight = 0.0;
    let mut moment = DVec3::ZERO;
    for (vertex, &value) in mass.iter().enumerate() {
        weight += value;
        moment = moment + mesh.vertices[vertex] * value;
    }
    if weight > 0.0 {
        moment * (1.0 / weight)
    } else {
        DVec3::ZERO
    }
}

fn tet_dofs(nv: usize, tet: [u32; 4], tet_index: usize) -> [usize; 15] {
    let mut dofs = [0; 15];
    for local in 0..4 {
        for axis in 0..3 {
            dofs[local * 3 + axis] = 3 * tet[local] as usize + axis;
        }
    }
    for axis in 0..3 {
        dofs[12 + axis] = 3 * nv + 3 * tet_index + axis;
    }
    dofs
}

fn shape_gradients(mesh: &TetMesh, tet: [u32; 4]) -> [DVec3; 4] {
    let center = centroid(mesh, tet);
    let base = barycentric(mesh, tet, center);
    let scale = mesh
        .vertices
        .iter()
        .map(|v| v.length())
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
        let next = barycentric(mesh, tet, shifted);
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

fn centroid(mesh: &TetMesh, tet: [u32; 4]) -> DVec3 {
    let p = tet.map(|index| mesh.vertices[index as usize]);
    (p[0] + p[1] + p[2] + p[3]) * 0.25
}

fn barycentric(mesh: &TetMesh, tet: [u32; 4], point: DVec3) -> [f64; 4] {
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

fn bubble_value(lambda: [f64; 4]) -> f64 {
    256.0 * lambda[0] * lambda[1] * lambda[2] * lambda[3]
}

fn bubble_grad(grads: &[DVec3; 4], lambda: [f64; 4]) -> DVec3 {
    let mut grad = DVec3::ZERO;
    for index in 0..4 {
        let mut product = 256.0;
        for other in 0..4 {
            if other != index {
                product *= lambda[other];
            }
        }
        grad = grad + grads[index] * product;
    }
    grad
}

fn basis_strain(grad_lambda: DVec3, axis: usize, _beta: f64, grad_beta: DVec3) -> [[f64; 3]; 3] {
    let mut gradient = [[0.0; 3]; 3];
    let shape = if grad_beta == DVec3::ZERO && _beta == 0.0 {
        grad_lambda
    } else if grad_lambda == DVec3::ZERO {
        grad_beta
    } else {
        grad_lambda
    };
    gradient[axis] = [shape.x, shape.y, shape.z];
    symmetric(gradient)
}

fn strain_div(strain: &[[f64; 3]; 3]) -> f64 {
    strain[0][0] + strain[1][1] + strain[2][2]
}

fn symmetric(gradient: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0; 3]; 3];
    for row in 0..3 {
        for col in 0..3 {
            out[row][col] = 0.5 * (gradient[row][col] + gradient[col][row]);
        }
    }
    out
}

fn double_dot(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> f64 {
    let mut sum = 0.0;
    for row in 0..3 {
        for col in 0..3 {
            sum += a[row][col] * b[row][col];
        }
    }
    sum
}

fn force_component(force: DVec3, axis: usize) -> f64 {
    match axis {
        0 => force.x,
        1 => force.y,
        _ => force.z,
    }
}

fn mat_vec(matrix: [[f64; 3]; 3], vector: DVec3) -> DVec3 {
    DVec3::new(
        matrix[0][0] * vector.x + matrix[0][1] * vector.y + matrix[0][2] * vector.z,
        matrix[1][0] * vector.x + matrix[1][1] * vector.y + matrix[1][2] * vector.z,
        matrix[2][0] * vector.x + matrix[2][1] * vector.y + matrix[2][2] * vector.z,
    )
}

pub fn extension_stress() -> [[f64; 3]; 3] {
    [[2.0, 0.0, 0.0], [0.0, -2.0, 0.0], [0.0, 0.0, 0.0]]
}

#[cfg(test)]
mod tests {
    use super::super::volume::tetrahedralize;
    use super::*;
    use crate::common::geometry::TriMesh;

    fn unit_mass(mesh: &TetMesh) -> Vec<f64> {
        let mut mass = vec![0.0; mesh.vertices.len()];
        for (tet, &volume) in mesh.volumes.iter().enumerate() {
            for vertex in mesh.tets[tet] {
                mass[vertex as usize] += volume / 4.0;
            }
        }
        mass
    }

    #[test]
    fn constant_velocity_has_zero_divergence() {
        let mesh = tetrahedralize(&TriMesh::unit_cube()).unwrap();
        let velocity = vec![DVec3::new(0.2, -0.4, 0.1); mesh.vertices.len()];
        let bubble = vec![DVec3::ZERO; mesh.tets.len()];
        let div = divergence_norm(&mesh, &velocity, &bubble);
        assert!(div < 1e-8, "{div}");
    }

    #[test]
    fn rigid_motions_span_a_six_dimensional_kernel() {
        let mesh = tetrahedralize(&TriMesh::unit_cube()).unwrap();
        let mass = unit_mass(&mesh);
        let (rank, sigma) = rigid_spectrum(&mesh, &mass);
        assert_eq!(rank, 6, "{sigma}");
        assert!(sigma > 0.0);
        let velocity = vec![DVec3::new(0.3, -0.2, 0.1); mesh.vertices.len()];
        let bubble = vec![DVec3::ZERO; mesh.tets.len()];
        assert!(divergence_norm(&mesh, &velocity, &bubble) < 1e-8);
    }

    #[test]
    fn cube_extension_matches_the_manufactured_field() {
        let mesh = tetrahedralize(&TriMesh::unit_cube()).unwrap();
        let mass = unit_mass(&mesh);
        let eta = vec![1.0; mesh.tets.len()];
        let force = vec![DVec3::ZERO; mesh.tets.len()];
        let solved = solve_stokes(
            &mesh,
            &StokesInput {
                eta: &eta,
                force: &force,
                vertex_mass: &mass,
                boundary_stress: Some(extension_stress()),
            },
        )
        .unwrap();
        assert!(solved.linear_residual < 1e-6, "{}", solved.linear_residual);
        assert!(
            solved.divergence_residual < 1e-5,
            "{}",
            solved.divergence_residual
        );
        assert!(solved.rigid_residual < 1e-5, "{}", solved.rigid_residual);
        let mut err = 0.0;
        let mut scale = 0.0;
        for (vertex, point) in mesh.vertices.iter().enumerate() {
            let exact = DVec3::new(point.x, -point.y, 0.0);
            err += (solved.vertex_velocity[vertex] - exact).length_squared();
            scale += exact.length_squared();
        }
        assert!(err.sqrt() < 1e-4 * scale.sqrt().max(1.0), "{err}");
    }
}
