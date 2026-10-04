//! Three-dimensional SCF increment, friction cone, and outer trust loop.
//! Clarabel sees free vertex vectors. The trust radius shrinks only after the
//! outer geometry check rejects a candidate.

use clarabel::algebra::CscMatrix;
use clarabel::solver::{
    DefaultSettings, DefaultSolver, IPSolver, NonnegativeConeT, SecondOrderConeT, SolverStatus,
    SupportedConeT, ZeroConeT,
};
use nalgebra::{Matrix3, SymmetricEigen};

use crate::common::density::{CauchyDensity, CauchyKernel};
use crate::common::geometry::TriMesh;
use crate::common::math::DVec3;
use crate::scf_simulator::error::ScfError;

use super::geometry::{SurfaceGeometry, apply_normal_jacobian};

const PENALTY: f64 = 1.0e4;

#[derive(Clone, Debug)]
pub struct VectorIncrement {
    pub d_pred: Vec<DVec3>,
    pub locked: Vec<bool>,
    pub deposit: Vec<bool>,
    pub weight: Vec<f64>,
    pub mass_coeff: Vec<DVec3>,
    pub current_mass: f64,
    pub target_mass: f64,
    pub gravity: Vec<DVec3>,
    pub normal: Vec<DVec3>,
    pub normal_jacobian: Vec<Vec<(usize, [f64; 9])>>,
    pub phi: Vec<f64>,
    pub slope_node: Vec<bool>,
}

#[derive(Clone, Debug)]
pub struct IncrementSolution {
    pub displacement: Vec<DVec3>,
    pub kkt_residual: f64,
    pub clarabel_called: bool,
    pub free_count: usize,
    pub mass_slack: f64,
}

pub fn predict_displacement(
    chi: &[f64],
    dt: f64,
    slip: &[DVec3],
    normal: &[DVec3],
    delta_h: &[f64],
    rho_bulk: f64,
    rho_surface: &[f64],
    locked: &[bool],
) -> Result<Vec<DVec3>, ScfError> {
    let n = chi.len();
    if slip.len() != n
        || normal.len() != n
        || delta_h.len() != n
        || rho_surface.len() != n
        || locked.len() != n
    {
        return Err(ScfError::model_not_closed(
            "predicted displacement arrays differ in length",
        ));
    }
    let mut out = Vec::with_capacity(n);
    for index in 0..n {
        if locked[index] {
            out.push(DVec3::ZERO);
            continue;
        }
        let rho = rho_surface[index];
        if !(rho.is_finite() && rho > 0.0) || !delta_h[index].is_finite() || !dt.is_finite() {
            return Err(ScfError::model_not_closed(
                "a predicted displacement coefficient is not finite",
            ));
        }
        let deposition = normal[index] * ((rho_bulk / rho) * delta_h[index]);
        let motion = slip[index] * (chi[index] * dt);
        let step = motion + deposition;
        if !step.is_finite() {
            return Err(ScfError::model_not_closed(
                "a predicted displacement is not finite",
            ));
        }
        out.push(step);
    }
    Ok(out)
}

pub fn solve_increment(
    problem: &VectorIncrement,
    trust: f64,
) -> Result<IncrementSolution, ScfError> {
    let free: Vec<usize> = (0..problem.d_pred.len())
        .filter(|&index| !problem.locked[index] && !problem.deposit[index])
        .collect();
    if free.is_empty() {
        return Ok(stamp(problem, &[]));
    }
    if !(trust.is_finite() && trust > 0.0) {
        return Err(ScfError::model_not_closed(
            "the trust radius is not positive",
        ));
    }
    if free.len() > 400 {
        return gauss_seidel(problem, &free, trust);
    }
    coupled(problem, &free, trust)
}

pub fn trust_scale(initial: f64, rejects: u32) -> Result<f64, ScfError> {
    let radius = initial * 0.5f64.powi(rejects as i32);
    let floor = initial * 1.0e-6;
    if radius < floor {
        Err(ScfError::trust_radius_exhausted(
            "outer geometry or mass acceptance failed at the trust-radius floor",
        ))
    } else {
        Ok(radius)
    }
}

pub fn align_frame(
    mesh: &TriMesh,
    density: &CauchyDensity,
) -> Result<(TriMesh, CauchyDensity, f64, f64), ScfError> {
    let (centroid, inertia) = frame_inertia(mesh);
    let shifted: Vec<DVec3> = mesh
        .vertices
        .iter()
        .map(|vertex| *vertex - centroid)
        .collect();
    let rotation = principal_rotation(&inertia);
    let rotated: Vec<DVec3> = shifted
        .iter()
        .map(|vertex| rotate(rotation, *vertex))
        .collect();
    let mut kernels = Vec::new();
    for kernel in &density.kernels {
        kernels.push(
            CauchyKernel::try_new(
                rotate(rotation, kernel.center - centroid),
                kernel.weight,
                kernel.sigma,
                kernel.alpha,
            )
            .map_err(|err| ScfError::model_not_closed(err.detail))?,
        );
    }
    let density = CauchyDensity::try_new(density.bulk, kernels)
        .map_err(|err| ScfError::model_not_closed(err.detail))?;
    let mesh = TriMesh::try_new(rotated, mesh.faces.clone())
        .map_err(|err| ScfError::model_not_closed(err.detail))?;
    let (next_centroid, next_inertia) = frame_inertia(&mesh);
    Ok((
        mesh,
        density,
        next_centroid.length(),
        (next_inertia[(0, 2)].abs() + next_inertia[(1, 2)].abs()),
    ))
}

fn coupled(
    problem: &VectorIncrement,
    free: &[usize],
    trust: f64,
) -> Result<IncrementSolution, ScfError> {
    let n = free.len();
    let nvar = 3 * n + n + 2;
    let mut p_diag = vec![0.0; nvar];
    let mut q = vec![0.0; nvar];
    for (local, &node) in free.iter().enumerate() {
        let weight = problem.weight[node].max(1e-8);
        for axis in 0..3 {
            let slot = 3 * local + axis;
            p_diag[slot] = weight;
            let pred = component(problem.d_pred[node], axis);
            q[slot] = -weight * pred;
        }
        p_diag[3 * n + local] = 1e-8;
        q[3 * n + local] = 1.0;
    }
    q[nvar - 2] = PENALTY;
    q[nvar - 1] = PENALTY;
    let p = CscMatrix::new(
        nvar,
        nvar,
        (0..=nvar).collect(),
        (0..nvar).collect(),
        p_diag,
    );
    let mut columns = vec![Vec::<(usize, f64)>::new(); nvar];
    let mut b = Vec::new();
    let mut cones: Vec<SupportedConeT<f64>> = Vec::new();
    let mass_scale = problem.target_mass.abs().max(1.0);
    b.push((problem.target_mass - problem.current_mass) / mass_scale);
    for (local, &node) in free.iter().enumerate() {
        for axis in 0..3 {
            let coeff = component(problem.mass_coeff[node], axis) / mass_scale;
            if coeff != 0.0 {
                columns[3 * local + axis].push((0, coeff));
            }
        }
    }
    columns[nvar - 2].push((0, 1.0 / mass_scale));
    columns[nvar - 1].push((0, -1.0 / mass_scale));
    cones.push(ZeroConeT(1));
    let mut row = 1usize;
    for local in 0..n {
        b.extend_from_slice(&[trust, 0.0, 0.0, 0.0]);
        for axis in 0..3 {
            columns[3 * local + axis].push((row + 1 + axis, -1.0));
        }
        cones.push(SecondOrderConeT(4));
        row += 4;
    }
    for (local, &node) in free.iter().enumerate() {
        if !problem.slope_node[node] {
            continue;
        }
        let gravity = problem.gravity[node];
        let gn = gravity.length();
        if !(gn.is_finite() && gn > 1e-16) {
            continue;
        }
        let normal = problem.normal[node];
        let known = normal;
        for &(other, matrix) in &problem.normal_jacobian[node] {
            if problem.locked[other] {
                continue;
            }
            if let Some(local_other) = free.iter().position(|index| *index == other) {
                let _ = (local_other, matrix);
            }
        }
        b.push(-1e-6 * gn - gravity.dot(known));
        for (local_other, &other) in free.iter().enumerate() {
            if let Some(matrix) = matrix_of(&problem.normal_jacobian[node], other) {
                for axis in 0..3 {
                    let dn = column(matrix, axis);
                    let slope = gravity.dot(dn);
                    if slope != 0.0 {
                        columns[3 * local_other + axis].push((row, slope));
                    }
                }
            }
        }
        cones.push(NonnegativeConeT(1));
        row += 1;
        let cross = gravity.cross(normal);
        b.push(gn * problem.phi[node].sin().abs());
        b.push(-cross.x);
        b.push(-cross.y);
        b.push(-cross.z);
        columns[3 * n + local].push((row, -gn));
        for (local_other, &other) in free.iter().enumerate() {
            if let Some(matrix) = matrix_of(&problem.normal_jacobian[node], other) {
                for axis in 0..3 {
                    let dn = column(matrix, axis);
                    let crossed = gravity.cross(dn);
                    for component_index in 0..3 {
                        let value = component(crossed, component_index);
                        if value != 0.0 {
                            columns[3 * local_other + axis]
                                .push((row + 1 + component_index, value));
                        }
                    }
                }
            }
        }
        cones.push(SecondOrderConeT(4));
        row += 4;
    }
    b.push(0.0);
    b.push(0.0);
    columns[nvar - 2].push((row, -1.0));
    columns[nvar - 1].push((row + 1, -1.0));
    cones.push(NonnegativeConeT(2));
    let matrix = csc_from_columns(b.len(), &columns);
    let settings = DefaultSettings {
        verbose: false,
        max_iter: 80,
        ..Default::default()
    };
    let mut solver = DefaultSolver::new(&p, &q, &matrix, &b, &cones, settings)
        .map_err(|err| ScfError::model_not_closed(format!("SOCP assembly: {err}")))?;
    solver.solve();
    if !matches!(
        solver.solution.status,
        SolverStatus::Solved | SolverStatus::AlmostSolved
    ) {
        return Err(ScfError::model_not_closed(format!(
            "general-mesh SOCP: {:?}",
            solver.solution.status
        )));
    }
    let mut solution = stamp(problem, &solver.solution.x[..3 * n]);
    solution.kkt_residual = solver
        .solution
        .r_prim
        .abs()
        .max(solver.solution.r_dual.abs());
    solution.clarabel_called = true;
    solution.free_count = n;
    solution.mass_slack = solver.solution.x[nvar - 2] - solver.solution.x[nvar - 1];
    if !solution.kkt_residual.is_finite() {
        return Err(ScfError::model_not_closed(
            "SOCP KKT residual is not finite",
        ));
    }
    Ok(solution)
}

fn gauss_seidel(
    problem: &VectorIncrement,
    free: &[usize],
    trust: f64,
) -> Result<IncrementSolution, ScfError> {
    let mut current = problem.d_pred.clone();
    for node in 0..current.len() {
        if problem.locked[node] {
            current[node] = DVec3::ZERO;
        }
    }
    let mut lambda = 0.0;
    let mut kkt = 0.0_f64;
    for _ in 0..3 {
        for &node in free {
            let mut local = problem.clone();
            for other in 0..local.locked.len() {
                if other != node && !local.deposit[other] {
                    local.locked[other] = true;
                    local.d_pred[other] = current[other];
                }
            }
            local.d_pred[node] = problem.d_pred[node]
                - problem.mass_coeff[node] * (lambda / problem.weight[node].max(1e-8));
            let solved = coupled(&local, &[node], trust)?;
            current[node] = solved.displacement[node];
            kkt = kkt.max(solved.kkt_residual);
        }
        let mut predicted = 0.0;
        for (coeff, step) in problem.mass_coeff.iter().zip(&current) {
            predicted += coeff.dot(*step);
        }
        let gap = problem.target_mass - problem.current_mass - predicted;
        lambda += gap / problem.target_mass.abs().max(1.0);
    }
    let mut solution = IncrementSolution {
        displacement: current,
        kkt_residual: kkt,
        clarabel_called: true,
        free_count: free.len(),
        mass_slack: 0.0,
    };
    for (index, locked) in problem.locked.iter().enumerate() {
        if *locked {
            solution.displacement[index] = DVec3::ZERO;
        } else if problem.deposit[index] {
            solution.displacement[index] = problem.d_pred[index];
        }
    }
    Ok(solution)
}

fn stamp(problem: &VectorIncrement, free_values: &[f64]) -> IncrementSolution {
    let mut displacement = vec![DVec3::ZERO; problem.d_pred.len()];
    let mut cursor = 0usize;
    for index in 0..problem.d_pred.len() {
        if problem.locked[index] {
            displacement[index] = DVec3::ZERO;
        } else if problem.deposit[index] {
            displacement[index] = problem.d_pred[index];
        } else if free_values.len() >= cursor + 3 {
            displacement[index] = DVec3::new(
                free_values[cursor],
                free_values[cursor + 1],
                free_values[cursor + 2],
            );
            cursor += 3;
        } else {
            displacement[index] = problem.d_pred[index];
        }
    }
    IncrementSolution {
        displacement,
        kkt_residual: 0.0,
        clarabel_called: false,
        free_count: cursor / 3,
        mass_slack: 0.0,
    }
}

fn frame_inertia(mesh: &TriMesh) -> (DVec3, Matrix3<f64>) {
    let mut centroid = DVec3::ZERO;
    for vertex in &mesh.vertices {
        centroid = centroid + *vertex;
    }
    centroid = centroid * (1.0 / mesh.vertices.len() as f64);
    let mut inertia = Matrix3::zeros();
    for vertex in &mesh.vertices {
        let arm = *vertex - centroid;
        let r2 = arm.length_squared();
        inertia[(0, 0)] += r2 - arm.x * arm.x;
        inertia[(1, 1)] += r2 - arm.y * arm.y;
        inertia[(2, 2)] += r2 - arm.z * arm.z;
        inertia[(0, 1)] -= arm.x * arm.y;
        inertia[(0, 2)] -= arm.x * arm.z;
        inertia[(1, 2)] -= arm.y * arm.z;
    }
    inertia[(1, 0)] = inertia[(0, 1)];
    inertia[(2, 0)] = inertia[(0, 2)];
    inertia[(2, 1)] = inertia[(1, 2)];
    (centroid, inertia)
}

fn principal_rotation(inertia: &Matrix3<f64>) -> Matrix3<f64> {
    let eigen = SymmetricEigen::new(inertia.clone());
    let mut rotation = eigen.eigenvectors;
    if rotation.determinant() < 0.0 {
        let mut column = rotation.column_mut(0);
        column *= -1.0;
    }
    rotation
}

fn rotate(rotation: Matrix3<f64>, point: DVec3) -> DVec3 {
    let rotated = rotation.transpose() * nalgebra::Vector3::new(point.x, point.y, point.z);
    DVec3::new(rotated.x, rotated.y, rotated.z)
}

fn component(value: DVec3, axis: usize) -> f64 {
    match axis {
        0 => value.x,
        1 => value.y,
        _ => value.z,
    }
}

fn column(matrix: [f64; 9], axis: usize) -> DVec3 {
    DVec3::new(matrix[axis], matrix[3 + axis], matrix[6 + axis])
}

fn matrix_of(rows: &[(usize, [f64; 9])], node: usize) -> Option<[f64; 9]> {
    rows.iter()
        .find(|(index, _)| *index == node)
        .map(|(_, matrix)| *matrix)
}

fn csc_from_columns(rows: usize, columns: &[Vec<(usize, f64)>]) -> CscMatrix<f64> {
    let mut colptr = vec![0];
    let mut rowval = Vec::new();
    let mut nzval = Vec::new();
    for column in columns {
        let mut entries = column.clone();
        entries.sort_by_key(|entry| entry.0);
        entries.retain(|entry| entry.0 < rows && entry.1 != 0.0);
        let mut last = usize::MAX;
        let mut collapsed: Vec<(usize, f64)> = Vec::new();
        for (row, value) in entries {
            if row == last {
                collapsed.last_mut().unwrap().1 += value;
            } else {
                collapsed.push((row, value));
                last = row;
            }
        }
        for (row, value) in collapsed {
            rowval.push(row);
            nzval.push(value);
        }
        colptr.push(rowval.len());
    }
    CscMatrix::new(rows, columns.len(), colptr, rowval, nzval)
}

pub fn normal_update(geometry: &SurfaceGeometry, displacement: &[DVec3], vertex: usize) -> DVec3 {
    geometry.vertex_normal[vertex]
        + apply_normal_jacobian(&geometry.normal_jacobian[vertex], displacement)
}

#[cfg(test)]
mod tests {
    use super::super::geometry::analyze;
    use super::*;
    use crate::common::geometry::TriMesh;

    #[test]
    fn empty_active_set_does_not_call_clarabel() {
        let problem = VectorIncrement {
            d_pred: vec![DVec3::new(0.1, 0.0, 0.0), DVec3::ZERO],
            locked: vec![true, false],
            deposit: vec![false, true],
            weight: vec![1.0, 1.0],
            mass_coeff: vec![DVec3::new(1.0, 0.0, 0.0), DVec3::ZERO],
            current_mass: 1.0,
            target_mass: 1.0,
            gravity: vec![DVec3::new(0.0, 0.0, -1.0); 2],
            normal: vec![DVec3::new(0.0, 0.0, 1.0); 2],
            normal_jacobian: vec![Vec::new(); 2],
            phi: vec![0.3, 0.3],
            slope_node: vec![false, false],
        };
        let solved = solve_increment(&problem, 1.0).unwrap();
        assert!(!solved.clarabel_called);
        assert_eq!(solved.displacement[0], DVec3::ZERO);
        assert_eq!(solved.displacement[1], problem.d_pred[1]);
    }

    #[test]
    fn trust_floor_stops_without_advancing_a_clock() {
        let mut time = 10.0;
        let initial = 1.0;
        let mut rejects = 0u32;
        let mut advanced = false;
        loop {
            match trust_scale(initial, rejects) {
                Ok(radius) => {
                    let accept = false;
                    if accept {
                        time += 1.0;
                        advanced = true;
                        let _ = radius;
                        break;
                    }
                    rejects += 1;
                }
                Err(err) => {
                    assert_eq!(err.kind, "TrustRadiusExhausted");
                    break;
                }
            }
        }
        assert!(!advanced);
        assert!((time - 10.0_f64).abs() < 1e-15);
    }

    #[test]
    fn predicted_vector_has_no_radial_denominator() {
        let step = predict_displacement(
            &[1.0],
            0.5,
            &[DVec3::new(0.2, -0.1, 0.0)],
            &[DVec3::new(0.0, 0.0, 1.0)],
            &[0.0],
            1.0,
            &[2.0],
            &[false],
        )
        .unwrap();
        assert!((step[0] - DVec3::new(0.1, -0.05, 0.0)).length() < 1e-15);
    }

    #[test]
    fn frame_alignment_keeps_the_vertices_without_ray_resampling() {
        let mesh = TriMesh::unit_cube();
        let mut shifted = mesh.vertices.clone();
        for vertex in &mut shifted {
            *vertex = *vertex + DVec3::new(0.4, -0.2, 0.1);
        }
        let mesh = TriMesh::try_new(shifted, mesh.faces.clone()).unwrap();
        let density = CauchyDensity::uniform(1.0).unwrap();
        let (aligned, _, centroid, _) = align_frame(&mesh, &density).unwrap();
        assert!(centroid < 1e-8, "{centroid}");
        assert_eq!(aligned.vertices.len(), mesh.vertices.len());
        assert_eq!(aligned.faces, mesh.faces);
        let geometry = analyze(&aligned).unwrap();
        assert!(geometry.volume > 0.0);
    }
}
