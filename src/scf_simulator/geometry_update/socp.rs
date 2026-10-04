//! Frozen-field incremental SOCP. Clarabel sees only the movable nodes.
//!
//! Locked increments are written as `+0.0` and deposit increments are copied
//! from `d_pred`. The trust radius shrinks only after the outer callback
//! rejects a candidate; a KKT residual is recorded and never accepted on its own.

use crate::common::geometry::TriMesh;
use crate::common::math::DVec3;
use crate::scf_simulator::error::ScfError;
use crate::scf_simulator::yielding::yield_fn::{TensorYieldCertificate, tensor_yield_certificate};
use clarabel::{
    algebra::CscMatrix,
    solver::{
        DefaultSettings, DefaultSolver, IPSolver, NonnegativeConeT, SecondOrderConeT, SolverStatus,
        SupportedConeT, ZeroConeT,
    },
};

const PENALTY: f64 = 1.0e6;
const OVERHANG: f64 = 1.0e-6;
const TRUST_FLOOR_RATIO: f64 = 1.0e-6;

#[derive(Clone, Debug)]
pub struct FrozenIncrement {
    pub d_pred: Vec<f64>,
    pub locked: Vec<bool>,
    pub deposit: Vec<bool>,
    pub objective_weight: Vec<f64>,
    pub mass_jacobian: Vec<f64>,
    pub current_mass: f64,
    pub target_mass: f64,
    pub volume_jacobian: Option<Vec<f64>>,
    pub current_volume: f64,
    pub target_volume: f64,
    pub gravity: Vec<[f64; 3]>,
    pub normals: Vec<[f64; 3]>,
    pub normal_jacobian: Vec<Vec<(usize, [f64; 3])>>,
    pub phi: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct IncrementSolution {
    pub displacement: Vec<f64>,
    pub kkt_residual: f64,
    pub clarabel_called: bool,
    pub free_count: usize,
    pub mass_slack: f64,
}

pub enum OuterVerdict {
    Accept,
    Reject(String),
    Abort(String),
}

#[derive(Clone, Debug)]
pub struct TrustedProjection {
    pub radii: Vec<f64>,
    pub solution: IncrementSolution,
    pub trust_used: f64,
}

#[derive(Clone, Debug)]
pub struct TensorYieldConstraint {
    pub stress_deviator: [[f64; 3]; 3],
    pub tide: [[f64; 3]; 3],
    pub cohesion: f64,
    pub mu_y: f64,
    pub pressure: f64,
}

impl TensorYieldConstraint {
    pub fn certificate(&self) -> Result<TensorYieldCertificate, String> {
        tensor_yield_certificate(
            &self.stress_deviator,
            &self.tide,
            self.cohesion,
            self.mu_y,
            self.pressure,
        )
    }
}

pub fn normalized_normal_jacobian(
    mesh: &TriMesh,
    directions: &[DVec3],
    radii: &[f64],
) -> Vec<Vec<(usize, [f64; 3])>> {
    let n = mesh.vertices.len();
    let mut neighbors = vec![Vec::<usize>::new(); n];
    let mut incident = vec![Vec::<usize>::new(); n];
    for (face_index, face) in mesh.faces.iter().enumerate() {
        let tri = [face[0] as usize, face[1] as usize, face[2] as usize];
        for index in tri {
            incident[index].push(face_index);
        }
        for a in 0..3 {
            for b in 0..3 {
                if a != b && !neighbors[tri[a]].contains(&tri[b]) {
                    neighbors[tri[a]].push(tri[b]);
                }
            }
        }
    }
    let mut columns = vec![Vec::new(); n];
    for j in 0..n {
        let step = (radii.get(j).copied().unwrap_or(1.0).abs() * 1.0e-5).max(1.0e-7);
        let direction = directions.get(j).copied().unwrap_or(DVec3::ZERO);
        for i in std::iter::once(j).chain(neighbors[j].iter().copied()) {
            let plus = local_normal(mesh, &incident, i, j, direction, step);
            let minus = local_normal(mesh, &incident, i, j, direction, -step);
            let derivative = (plus - minus) * (1.0 / (2.0 * step));
            columns[i].push((j, derivative.to_array()));
        }
    }
    columns
}

fn local_normal(
    mesh: &TriMesh,
    incident: &[Vec<usize>],
    vertex: usize,
    perturbed: usize,
    direction: DVec3,
    delta: f64,
) -> DVec3 {
    let mut sum = DVec3::ZERO;
    for &face in &incident[vertex] {
        let [ia, ib, ic] = mesh.faces[face];
        let point = |index: u32| {
            let position = mesh.vertices[index as usize];
            if index as usize == perturbed {
                position + direction * delta
            } else {
                position
            }
        };
        sum = sum + (point(ib) - point(ia)).cross(point(ic) - point(ia));
    }
    sum.try_normalize().unwrap_or(DVec3::ZERO)
}

/// Solve one frozen SOCP at a fixed trust radius. This function does not shrink it.
pub fn solve_frozen_increment(
    problem: &FrozenIncrement,
    trust: f64,
) -> Result<IncrementSolution, String> {
    let movable = movable_index(problem)?;
    if !trust.is_finite() || trust <= 0.0 {
        return Err("trust radius is not positive".into());
    }
    if movable.is_empty() {
        return Ok(stamp(problem, &[]));
    }
    let n = movable.len();
    let volume_on = problem.volume_jacobian.is_some();
    let eq = if volume_on { 2 } else { 1 };
    let nvar = 2 * n + if volume_on { 4 } else { 2 };
    let mut q = vec![0.0; nvar];
    let mut p_diag = vec![0.0; nvar];
    for (local, &node) in movable.iter().enumerate() {
        let weight = problem.objective_weight[node];
        if !(weight.is_finite() && weight > 0.0) {
            return Err("a movable node has a non-positive objective weight".into());
        }
        p_diag[local] = weight;
        q[local] = -weight * problem.d_pred[node];
        q[n + local] = PENALTY;
    }
    q[2 * n] = PENALTY;
    q[2 * n + 1] = PENALTY;
    if volume_on {
        q[2 * n + 2] = PENALTY;
        q[2 * n + 3] = PENALTY;
    }
    let p = CscMatrix::new(
        nvar,
        nvar,
        (0..=nvar).collect(),
        (0..nvar).collect(),
        p_diag,
    );

    let slack_rows = if volume_on { 4 } else { 2 };
    let mut rows = eq + 3 * n + slack_rows;
    let mut slope_nodes = Vec::new();
    for &node in &movable {
        let gravity = DVec3::from_array(problem.gravity[node]);
        if gravity.length() > 1.0e-30 && gravity.length().is_finite() {
            slope_nodes.push(node);
            rows += 1 + 4;
        }
    }
    let mut columns = vec![Vec::<(usize, f64)>::new(); nvar];
    let mut b = vec![0.0; rows];
    let scale = problem.target_mass;
    if !(scale.is_finite() && scale > 0.0)
        || !problem.current_mass.is_finite()
        || problem.mass_jacobian.iter().any(|value| !value.is_finite())
    {
        return Err("the mass row is not finite".into());
    }
    let mut fixed = 0.0;
    for (node, deposit) in problem.deposit.iter().enumerate() {
        if *deposit {
            fixed += problem.mass_jacobian[node] * problem.d_pred[node];
        }
    }
    b[0] = (problem.target_mass - problem.current_mass - fixed) / scale;
    columns[2 * n].push((0, 1.0));
    columns[2 * n + 1].push((0, -1.0));
    if let Some(jacobian) = &problem.volume_jacobian {
        if jacobian.len() != problem.d_pred.len()
            || !problem.current_volume.is_finite()
            || !problem.target_volume.is_finite()
        {
            return Err("the volume row is not finite".into());
        }
        let volume_scale = problem.target_volume.abs().max(1.0);
        let mut fixed_volume = 0.0;
        for (node, deposit) in problem.deposit.iter().enumerate() {
            if *deposit {
                fixed_volume += jacobian[node] * problem.d_pred[node];
            }
        }
        b[1] = (problem.target_volume - problem.current_volume - fixed_volume) / volume_scale;
        columns[2 * n + 2].push((1, 1.0));
        columns[2 * n + 3].push((1, -1.0));
        for (local, &node) in movable.iter().enumerate() {
            columns[local].push((1, jacobian[node] / volume_scale));
        }
    }
    for (local, &node) in movable.iter().enumerate() {
        columns[local].push((0, problem.mass_jacobian[node] / scale));
        let upper = eq + local;
        let lower = eq + n + local;
        columns[local].push((upper, 1.0));
        columns[local].push((lower, -1.0));
        b[upper] = trust;
        b[lower] = trust;
        let slack = eq + 2 * n + local;
        columns[n + local].push((slack, -1.0));
        b[slack] = 0.0;
    }
    let plus_row = eq + 3 * n;
    columns[2 * n].push((plus_row, -1.0));
    columns[2 * n + 1].push((plus_row + 1, -1.0));
    b[plus_row] = 0.0;
    b[plus_row + 1] = 0.0;
    if volume_on {
        columns[2 * n + 2].push((plus_row + 2, -1.0));
        columns[2 * n + 3].push((plus_row + 3, -1.0));
        b[plus_row + 2] = 0.0;
        b[plus_row + 3] = 0.0;
    }
    let mut cursor = plus_row + slack_rows;
    for &node in &slope_nodes {
        let gravity = DVec3::from_array(problem.gravity[node]);
        let gn = gravity.length();
        let normal = DVec3::from_array(problem.normals[node]);
        let mut cross = gravity.cross(normal);
        let mut radial = gravity.dot(normal);
        for &(column, derivative) in &problem.normal_jacobian[node] {
            if !problem.deposit.get(column).copied().unwrap_or(false) {
                continue;
            }
            let dn = DVec3::from_array(derivative) * problem.d_pred[column];
            cross = cross + gravity.cross(dn);
            radial += gravity.dot(dn);
        }
        let overhang = cursor;
        cursor += 1;
        b[overhang] = -OVERHANG * gn - radial;
        for (local, &free) in movable.iter().enumerate() {
            if let Some(derivative) = column_of(&problem.normal_jacobian[node], free) {
                let slope = gravity.dot(DVec3::from_array(derivative));
                if slope != 0.0 {
                    columns[local].push((overhang, slope));
                }
            }
        }
        let soc = cursor;
        cursor += 4;
        let sine = problem.phi[node].sin();
        if !sine.is_finite() {
            return Err("a friction angle is not finite".into());
        }
        b[soc] = gn * sine.abs();
        let local = movable.iter().position(|&free| free == node).unwrap();
        columns[n + local].push((soc, -gn));
        let cross = cross.to_array();
        for k in 0..3 {
            b[soc + 1 + k] = -cross[k];
        }
        for (local, &free) in movable.iter().enumerate() {
            if let Some(derivative) = column_of(&problem.normal_jacobian[node], free) {
                let crossed = gravity.cross(DVec3::from_array(derivative)).to_array();
                for k in 0..3 {
                    if crossed[k] != 0.0 {
                        columns[local].push((soc + 1 + k, crossed[k]));
                    }
                }
            }
        }
    }
    let mut cones: Vec<SupportedConeT<f64>> = vec![
        ZeroConeT(eq),
        NonnegativeConeT(2 * n),
        NonnegativeConeT(n),
        NonnegativeConeT(slack_rows),
    ];
    for _ in &slope_nodes {
        cones.push(NonnegativeConeT(1));
        cones.push(SecondOrderConeT(4));
    }
    let a = csc_from_columns(rows, &columns);
    let settings = DefaultSettings {
        verbose: false,
        tol_gap_abs: 1e-8,
        tol_gap_rel: 1e-8,
        tol_feas: 1e-8,
        max_iter: 200,
        ..Default::default()
    };
    let mut solver = DefaultSolver::new(&p, &q, &a, &b, &cones, settings)
        .map_err(|err| format!("SOCP assembly: {err}"))?;
    solver.solve();
    if !matches!(
        solver.solution.status,
        SolverStatus::Solved | SolverStatus::AlmostSolved
    ) {
        return Err(format!("frozen SOCP: {:?}", solver.solution.status));
    }
    let kkt = solver
        .solution
        .r_prim
        .abs()
        .max(solver.solution.r_dual.abs());
    if !kkt.is_finite() {
        return Err("frozen SOCP KKT residual is not finite".into());
    }
    let mut solution = stamp(problem, &solver.solution.x[..n]);
    solution.kkt_residual = kkt;
    solution.clarabel_called = true;
    solution.free_count = n;
    solution.mass_slack = solver.solution.x[2 * n] - solver.solution.x[2 * n + 1];
    Ok(solution)
}

pub fn project_trusted_increment<F>(
    problem: &FrozenIncrement,
    radii: &[f64],
    trust: f64,
    mut accept: F,
) -> Result<TrustedProjection, ScfError>
where
    F: FnMut(&[f64], &IncrementSolution) -> OuterVerdict,
{
    if radii.len() != problem.d_pred.len() || !trust.is_finite() || trust <= 0.0 {
        return Err(ScfError::model_not_closed(
            "incremental SOCP dimensions or trust radius are invalid",
        ));
    }
    let movable = movable_index(problem).map_err(ScfError::model_not_closed)?;
    let floor = trust * TRUST_FLOOR_RATIO;
    let mut delta = trust;
    loop {
        let solution =
            solve_frozen_increment(problem, delta).map_err(ScfError::model_not_closed)?;
        let candidate: Vec<f64> = radii
            .iter()
            .zip(&solution.displacement)
            .map(|(radius, increment)| radius + increment)
            .collect();
        match accept(&candidate, &solution) {
            OuterVerdict::Accept => {
                return Ok(TrustedProjection {
                    radii: candidate,
                    solution,
                    trust_used: delta,
                });
            }
            OuterVerdict::Abort(detail) => return Err(ScfError::model_not_closed(detail)),
            OuterVerdict::Reject(_) => {
                if movable.is_empty() || delta * 0.5 < floor {
                    return Err(ScfError::trust_radius_exhausted(
                        "outer geometry or mass acceptance failed at the trust-radius floor",
                    ));
                }
                delta *= 0.5;
            }
        }
    }
}

fn movable_index(problem: &FrozenIncrement) -> Result<Vec<usize>, String> {
    let n = problem.d_pred.len();
    let same = [
        problem.locked.len(),
        problem.deposit.len(),
        problem.objective_weight.len(),
        problem.mass_jacobian.len(),
        problem.gravity.len(),
        problem.normals.len(),
        problem.normal_jacobian.len(),
        problem.phi.len(),
    ]
    .iter()
    .all(|len| *len == n);
    if n == 0 || !same {
        return Err("incremental SOCP node arrays differ".into());
    }
    let mut movable = Vec::new();
    for i in 0..n {
        match (problem.locked[i], problem.deposit[i]) {
            (true, false) | (false, true) => {}
            (false, false) => movable.push(i),
            (true, true) => {
                return Err("a node is both locked and a deposit node".into());
            }
        }
        if !problem.d_pred[i].is_finite() {
            return Err("d_pred is not finite".into());
        }
    }
    Ok(movable)
}

fn stamp(problem: &FrozenIncrement, free: &[f64]) -> IncrementSolution {
    let n = problem.d_pred.len();
    let mut displacement = vec![0.0; n];
    let mut cursor = 0;
    for i in 0..n {
        if problem.locked[i] {
            displacement[i] = 0.0;
        } else if problem.deposit[i] {
            displacement[i] = problem.d_pred[i];
        } else {
            displacement[i] = free[cursor];
            cursor += 1;
        }
    }
    for i in 0..n {
        if problem.locked[i] {
            displacement[i] = 0.0;
        } else if problem.deposit[i] {
            displacement[i] = problem.d_pred[i];
        }
    }
    IncrementSolution {
        displacement,
        kkt_residual: 0.0,
        clarabel_called: false,
        free_count: free.len(),
        mass_slack: 0.0,
    }
}

fn column_of(columns: &[(usize, [f64; 3])], node: usize) -> Option<[f64; 3]> {
    columns
        .iter()
        .find(|(index, _)| *index == node)
        .map(|(_, derivative)| *derivative)
}

fn csc_from_columns(rows: usize, columns: &[Vec<(usize, f64)>]) -> CscMatrix<f64> {
    let mut colptr = vec![0];
    let mut rowval = Vec::new();
    let mut nzval = Vec::new();
    for column in columns {
        let mut merged = column.clone();
        merged.sort_by_key(|entry| entry.0);
        let mut acc: Vec<(usize, f64)> = Vec::new();
        for (row, value) in merged {
            if !value.is_finite() || value == 0.0 || row >= rows {
                continue;
            }
            if let Some(last) = acc.last_mut() {
                if last.0 == row {
                    last.1 += value;
                    continue;
                }
            }
            acc.push((row, value));
        }
        for (row, value) in acc {
            if value != 0.0 {
                rowval.push(row);
                nzval.push(value);
            }
        }
        colptr.push(rowval.len());
    }
    CscMatrix::new(rows, columns.len(), colptr, rowval, nzval)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scf_simulator::geometry_update::accept::{
        audit_radial_mesh, finite_gravitational_field,
    };
    use crate::scf_simulator::geometry_update::mass::{flux_density_weight, surface_mass_jacobian};

    fn blank(n: usize) -> FrozenIncrement {
        FrozenIncrement {
            d_pred: vec![0.0; n],
            locked: vec![false; n],
            deposit: vec![false; n],
            objective_weight: vec![1.0; n],
            mass_jacobian: vec![0.0; n],
            current_mass: 1.0,
            target_mass: 1.0,
            volume_jacobian: None,
            current_volume: 1.0,
            target_volume: 1.0,
            gravity: vec![[0.0, 0.0, -1.0]; n],
            normals: vec![[0.0, 0.0, 1.0]; n],
            normal_jacobian: vec![Vec::new(); n],
            phi: vec![std::f64::consts::FRAC_PI_2; n],
        }
    }

    #[test]
    fn locked_is_bit_zero_deposit_is_copied_and_only_movable_is_free() {
        let mut problem = blank(3);
        problem.d_pred = vec![4.0, 0.25, 0.1];
        problem.locked[0] = true;
        problem.deposit[1] = true;
        let projection =
            project_trusted_increment(&problem, &[1.0, 1.0, 1.0], 1.0, |_, _| OuterVerdict::Accept)
                .unwrap();
        assert!(projection.solution.clarabel_called);
        assert_eq!(projection.solution.free_count, 1);
        assert_eq!(
            projection.solution.displacement[0].to_bits(),
            0.0_f64.to_bits()
        );
        assert_eq!(
            projection.solution.displacement[1].to_bits(),
            0.25_f64.to_bits()
        );
        assert!((projection.solution.displacement[2] - 0.1).abs() < 1e-6);
    }

    #[test]
    fn mass_row_uses_the_jacobian_and_the_density_weight() {
        let jacobian = surface_mass_jacobian(1.0, 4.0, 2.0);
        assert!((jacobian - 16.0).abs() < 1e-12);
        let weight = flux_density_weight(2.0, 4.0).unwrap();
        let delta_h = 0.2;
        let radial = weight * delta_h;
        assert!((jacobian * radial - 2.0 * 1.0 * 4.0 * delta_h).abs() < 1e-12);

        let mut problem = blank(1);
        problem.current_mass = 10.0;
        problem.target_mass = 10.5;
        problem.mass_jacobian = vec![jacobian];
        problem.d_pred = vec![0.0];
        let projection = project_trusted_increment(&problem, &[2.0], 1.0, |_, solution| {
            if solution.kkt_residual.is_finite() {
                OuterVerdict::Accept
            } else {
                OuterVerdict::Reject("non-finite kkt".into())
            }
        })
        .unwrap();
        assert!(projection.solution.displacement[0] > 0.02);
        assert!(projection.solution.kkt_residual < 1e-4);
    }

    #[test]
    fn kkt_success_does_not_keep_a_rejected_geometry() {
        let mut problem = blank(1);
        problem.d_pred = vec![-2.0];
        let mut saw_invalid = false;
        let projection = project_trusted_increment(&problem, &[1.0], 3.0, |radii, solution| {
            assert!(solution.kkt_residual.is_finite());
            if radii[0] <= 0.0 {
                saw_invalid = true;
                OuterVerdict::Reject("non-positive radius".into())
            } else {
                OuterVerdict::Accept
            }
        })
        .unwrap();
        assert!(saw_invalid);
        assert!(projection.radii[0] > 0.0);
        assert!(projection.trust_used < 3.0);
        assert!((projection.radii[0] - (1.0 - 2.0)).abs() > 0.5);
    }

    #[test]
    fn outer_failure_down_to_the_floor_is_trust_radius_exhausted() {
        let problem = blank(1);
        let err = project_trusted_increment(&problem, &[1.0], 1.0e-3, |_, _| {
            OuterVerdict::Reject("self-intersection".into())
        })
        .unwrap_err();
        assert_eq!(err.kind, "TrustRadiusExhausted");
        assert_eq!(err.code, 23);
    }

    #[test]
    fn an_inner_solver_failure_does_not_shrink_the_trust_radius() {
        let mut problem = blank(1);
        problem.gravity = vec![[0.0, 0.0, 1.0]];
        problem.normals = vec![[0.0, 0.0, 1.0]];
        let mut accepts = 0;
        let err = project_trusted_increment(&problem, &[1.0], 1.0, |_, _| {
            accepts += 1;
            OuterVerdict::Accept
        })
        .unwrap_err();
        assert_eq!(accepts, 0);
        assert_eq!(err.kind, "ModelNotClosed");
    }

    #[test]
    fn octahedron_candidate_passes_volume_intersection_and_star_checks() {
        let directions = [
            DVec3::new(1.0, 0.0, 0.0),
            DVec3::new(-1.0, 0.0, 0.0),
            DVec3::new(0.0, 1.0, 0.0),
            DVec3::new(0.0, -1.0, 0.0),
            DVec3::new(0.0, 0.0, 1.0),
            DVec3::new(0.0, 0.0, -1.0),
        ];
        let faces = [
            [0, 2, 4],
            [0, 4, 3],
            [0, 3, 5],
            [0, 5, 2],
            [1, 4, 2],
            [1, 3, 4],
            [1, 5, 3],
            [1, 2, 5],
        ];
        let audit = audit_radial_mesh(&directions, &[1.0; 6], &faces).unwrap();
        assert!(audit.volume > 0.0);
        assert!(audit.star_margin >= 1.0e-4);
        assert!(!finite_gravitational_field(
            &[f64::NAN],
            &[[0.0; 3]],
            &[[0.0; 6]]
        ));
    }

    #[test]
    fn exact_tensor_yield_constraint_is_not_a_geometry_acceptance() {
        let feasible = TensorYieldConstraint {
            stress_deviator: [[0.0; 3]; 3],
            tide: [[0.0; 3]; 3],
            cohesion: 1.0,
            mu_y: 0.4,
            pressure: 1.0,
        };
        let cert = feasible.certificate().unwrap();
        assert!(cert.satisfied);
        assert!(cert.residual < 0.0);
    }
}
