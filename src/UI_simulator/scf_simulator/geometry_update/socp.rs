use super::mass::{evaluate_mass_at_radii, linearize_mass_constraint};
use crate::common::geometry::TriMesh;
use crate::common::math::DVec3 as MeshVec;
use crate::scf_simulator::surface::friction::check_slope_violation;
use crate::scf_simulator::yielding::yield_fn::{TensorYieldCertificate, tensor_yield_certificate};
use clarabel::{
    algebra::CscMatrix,
    solver::{
        DefaultSettings, DefaultSolver, IPSolver, NonnegativeConeT, SecondOrderConeT, SolverStatus,
        SupportedConeT, ZeroConeT,
    },
};
use glam::DVec3;

fn mesh_vertex(mesh: &TriMesh, index: usize) -> [f64; 3] {
    mesh.vertices[index].to_array()
}

fn vertex_neighbors(mesh: &TriMesh) -> Vec<Vec<usize>> {
    let n = mesh.vertices.len();
    let mut neigh = vec![Vec::<usize>::new(); n];
    for face in &mesh.faces {
        let tri = [face[0] as usize, face[1] as usize, face[2] as usize];
        for a in 0..3 {
            for b in 0..3 {
                if a != b && !neigh[tri[a]].contains(&tri[b]) {
                    neigh[tri[a]].push(tri[b]);
                }
            }
        }
    }
    neigh
}

fn vertex_normals(mesh: &TriMesh) -> Vec<[f64; 3]> {
    let mut acc = vec![DVec3::ZERO; mesh.vertices.len()];
    for face in &mesh.faces {
        let a = DVec3::from_array(mesh_vertex(mesh, face[0] as usize));
        let b = DVec3::from_array(mesh_vertex(mesh, face[1] as usize));
        let c = DVec3::from_array(mesh_vertex(mesh, face[2] as usize));
        let area = (b - a).cross(c - a);
        acc[face[0] as usize] += area;
        acc[face[1] as usize] += area;
        acc[face[2] as usize] += area;
    }
    acc.into_iter().map(|n| n.normalize_or_zero().to_array()).collect()
}

#[derive(Clone, Debug)]
pub struct QpCertificate {
    pub primal: f64,
    pub dual: f64,
    pub complementarity: f64,
    pub multipliers: Vec<f64>,
    pub slack: Vec<f64>,
    pub solver_iterations: u32,
}
impl QpCertificate {
    pub fn residual(&self) -> f64 {
        self.primal.max(self.dual).max(self.complementarity)
    }
}
#[derive(Clone, Debug)]
pub struct MeshProjection {
    pub radii: Vec<f64>,
    pub certificate: QpCertificate,
    pub mass_residual: f64,
    pub slope_residual: f64,
    pub sqp_iterations: usize,
    pub tensor_yield_residual: f64,
}
#[derive(Clone, Copy, Debug)]
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
pub fn radial_mesh(
    mesh: &TriMesh,
    center: [f64; 3],
    directions: &[[f64; 3]],
    radii: &[f64],
) -> Result<TriMesh, String> {
    if mesh.vertices.len() != radii.len() || directions.len() != radii.len() {
        return Err("Radial geometry dimensions differ".into());
    }
    let mut result = mesh.clone();
    for (i, (&r, &u)) in radii.iter().zip(directions).enumerate() {
        if !r.is_finite() || r <= 0.0 || (DVec3::from_array(u).length() - 1.0).abs() > 1e-8 {
            return Err("Invalid radial geometry".into());
        }
        result.vertices[i] = MeshVec::new(center[0] + r * u[0], center[1] + r * u[1], center[2] + r * u[2]);
    }
    Ok(result)
}
pub fn normal_jacobian(
    mesh: &TriMesh,
    directions: &[[f64; 3]],
    radii: &[f64],
) -> Vec<Vec<(usize, DVec3)>> {
    let n = radii.len();
    let mut result = vec![Vec::new(); n];
    let neighbors = vertex_neighbors(mesh);
    let mut incident = vec![Vec::<usize>::new(); n];
    for face in 0..mesh.faces.len() {
        for idx in mesh.faces[face] {
            incident[idx as usize].push(face);
        }
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
            let point = |idx: u32| {
                let p = DVec3::from_array(mesh_vertex(mesh, idx as usize));
                if idx as usize == perturbed {
                    p + delta * direction
                } else {
                    p
                }
            };
            sum += (point(ib) - point(ia)).cross(point(ic) - point(ia));
        }
        sum.normalize_or_zero()
    }
    for j in 0..n {
        let eps = (radii[j].abs() * 1e-5).max(1e-7);
        let direction = DVec3::from_array(directions[j]);
        for i in std::iter::once(j).chain(neighbors[j].iter().copied()) {
            let np = local_normal(mesh, &incident, i, j, direction, eps);
            let nm = local_normal(mesh, &incident, i, j, direction, -eps);
            result[i].push((j, (np - nm) / (2.0 * eps)));
        }
    }
    result
}
fn solve_subproblem<D: Fn([f64; 3]) -> f64>(
    mesh: &TriMesh,
    center: [f64; 3],
    current: &[f64],
    target: &[f64],
    directions: &[[f64; 3]],
    weights: &[f64],
    gravity: &[[f64; 3]],
    phi: &[f64],
    density: &D,
    mass: f64,
    trust: f64,
    tensor_constraints: Option<&[TensorYieldConstraint]>,
) -> Result<(Vec<f64>, QpCertificate), String> {
    let n = current.len();
    let scale = current.iter().sum::<f64>() / n as f64;
    let weight_scale = weights.iter().sum::<f64>();
    let mc = linearize_mass_constraint(current, directions, weights, density, mass);
    let actual_mesh = radial_mesh(mesh, center, directions, current)?;
    let normals = vertex_normals(&actual_mesh);
    let jac = normal_jacobian(&actual_mesh, directions, current);
    let p_diag: Vec<f64> = weights.iter().map(|w| w / weight_scale).collect();
    let p = CscMatrix::new(n, n, (0..=n).collect(), (0..n).collect(), p_diag.clone());
    let q: Vec<f64> = (0..n)
        .map(|i| -p_diag[i] * (target[i] - current[i]) / scale)
        .collect();
    let tensor_count = tensor_constraints.map_or(0, |constraints| constraints.len());
    let rows = 1 + 6 * n + tensor_count;
    let mut b = vec![0.0; rows];
    let mut entries = vec![Vec::<(usize, f64)>::new(); n];
    b[0] = (mass - mc.current_mass) / mass;
    for j in 0..n {
        entries[j].push((0, mc.coefficients[j] * scale / mass));
        entries[j].push((1 + j, 1.0));
        entries[j].push((1 + n + j, -1.0));
        b[1 + j] = trust / scale;
        b[1 + n + j] = trust.min(0.9 * current[j]) / scale;
    }
    let mut cones: Vec<SupportedConeT<f64>> = vec![ZeroConeT(1), NonnegativeConeT(2 * n)];
    for i in 0..n {
        let g = DVec3::from_array(gravity[i]);
        let gn = g.length();
        if !gn.is_finite() || gn <= 1e-30 {
            return Err(format!(
                "Zero effective gravity at node {i}: normalized slope constraint is undefined"
            ));
        }
        let gh = g / gn;
        let cross = gh.cross(DVec3::from_array(normals[i]));
        let cross_len = cross.length();
        let limit = phi[i].sin().abs();
        let row = 1 + 2 * n + 4 * i;
        b[row] = limit.max(cross_len);
        for k in 0..3 {
            b[row + 1 + k] = cross[k];
        }
        for &(j, dn) in &jac[i] {
            let derivative = gh.cross(dn) * scale;
            for k in 0..3 {
                if derivative[k] != 0.0 {
                    entries[j].push((row + 1 + k, -derivative[k]));
                }
            }
        }
        cones.push(SecondOrderConeT(4));
    }
    if let Some(constraints) = tensor_constraints {
        for (k, constraint) in constraints.iter().enumerate() {
            let residual = constraint.certificate()?.residual;
            b[1 + 6 * n + k] = -residual;
        }
        cones.push(NonnegativeConeT(tensor_count));
    }
    let mut colptr = vec![0];
    let mut rowval = Vec::new();
    let mut nzval = Vec::new();
    for column in &mut entries {
        column.sort_by_key(|e| e.0);
        for &(row, val) in column.iter() {
            rowval.push(row);
            nzval.push(val);
        }
        colptr.push(rowval.len());
    }
    let a = CscMatrix::new(rows, n, colptr, rowval, nzval);
    let settings = DefaultSettings {
        verbose: false,
        tol_gap_abs: 1e-8,
        tol_gap_rel: 1e-8,
        tol_feas: 1e-8,
        max_iter: 250,
        ..Default::default()
    };
    let mut solver = DefaultSolver::new(&p, &q, &a, &b, &cones, settings)
        .map_err(|e| format!("QP assembly: {e}"))?;
    solver.solve();
    if !matches!(
        solver.solution.status,
        SolverStatus::Solved | SolverStatus::AlmostSolved
    ) {
        return Err(format!("Convex subproblem: {:?}", solver.solution.status));
    }
    let sol = &solver.solution;
    let mut primal = vec![0.0; rows];
    let mut dual = q.clone();
    for j in 0..n {
        dual[j] += p_diag[j] * sol.x[j];
        for &(row, val) in &entries[j] {
            primal[row] += val * sol.x[j];
            dual[j] += val * sol.z[row];
        }
    }
    for row in 0..rows {
        primal[row] += sol.s[row] - b[row];
    }
    let mut complementarity = 0.0_f64;
    for row in 1..1 + 2 * n {
        complementarity = complementarity.max((sol.s[row] * sol.z[row]).abs());
    }
    for i in 0..n {
        let base = 1 + 2 * n + 4 * i;
        let product: f64 = (0..4).map(|k| sol.s[base + k] * sol.z[base + k]).sum();
        complementarity = complementarity.max(product.abs());
    }
    for row in 1 + 6 * n..rows {
        complementarity = complementarity.max((sol.s[row] * sol.z[row]).abs());
    }
    let cert = QpCertificate {
        primal: primal.iter().map(|v| v.abs()).fold(sol.r_prim, f64::max),
        dual: dual.iter().map(|v| v.abs()).fold(sol.r_dual, f64::max),
        complementarity,
        multipliers: sol.z.clone(),
        slack: sol.s.clone(),
        solver_iterations: sol.iterations,
    };
    if cert.residual() > 1e-6 {
        return Err(format!("QP KKT residual {}", cert.residual()));
    }
    Ok((
        (0..n).map(|i| current[i] + scale * sol.x[i]).collect(),
        cert,
    ))
}
pub fn true_mesh_residuals<D: Fn([f64; 3]) -> f64>(
    mesh: &TriMesh,
    center: [f64; 3],
    radii: &[f64],
    directions: &[[f64; 3]],
    weights: &[f64],
    gravity: &[[f64; 3]],
    phi: &[f64],
    density: &D,
    mass: f64,
) -> Result<(f64, f64), String> {
    let candidate = radial_mesh(mesh, center, directions, radii)?;
    let volume = candidate.signed_volume();
    if !volume.is_finite() || volume <= 0.0 {
        return Err("Candidate geometry: signed volume is non-positive".into());
    }
    let normals = vertex_normals(&candidate);
    let mut slope = 0.0_f64;
    for i in 0..radii.len() {
        let (_, actual, limit) = check_slope_violation(gravity[i], normals[i], phi[i]);
        slope = slope.max((actual - limit).max(0.0));
    }
    let mass_res =
        (evaluate_mass_at_radii(radii, directions, weights, density) - mass).abs() / mass;
    Ok((mass_res, slope))
}
pub fn project_mesh<D: Fn([f64; 3]) -> f64>(
    mesh: &TriMesh,
    center: [f64; 3],
    initial: &[f64],
    target: &[f64],
    directions: &[[f64; 3]],
    weights: &[f64],
    gravity: &[[f64; 3]],
    phi: &[f64],
    density: &D,
    mass: f64,
    trust: f64,
    mass_tol: f64,
    slope_tol: f64,
) -> Result<MeshProjection, String> {
    project_mesh_with_tensor_yield(
        mesh, center, initial, target, directions, weights, gravity, phi, density, mass, trust,
        mass_tol, slope_tol, None,
    )
}
pub fn project_mesh_with_tensor_yield<D: Fn([f64; 3]) -> f64>(
    mesh: &TriMesh,
    center: [f64; 3],
    initial: &[f64],
    target: &[f64],
    directions: &[[f64; 3]],
    weights: &[f64],
    gravity: &[[f64; 3]],
    phi: &[f64],
    density: &D,
    mass: f64,
    trust: f64,
    mass_tol: f64,
    slope_tol: f64,
    tensor_constraints: Option<&[TensorYieldConstraint]>,
) -> Result<MeshProjection, String> {
    let n = initial.len();
    if n == 0
        || [
            target.len(),
            directions.len(),
            weights.len(),
            gravity.len(),
            phi.len(),
            mesh.vertices.len(),
        ]
        .iter()
        .any(|x| *x != n)
        || !mass.is_finite()
        || mass <= 0.0
        || !trust.is_finite()
        || trust <= 0.0
        || weights.iter().any(|w| !w.is_finite() || *w <= 0.0)
    {
        return Err("Invalid mesh SQP input".into());
    }
    let (_, initial_slope) = true_mesh_residuals(
        mesh, center, initial, directions, weights, gravity, phi, density, mass,
    )?;
    let mut current = initial.to_vec();
    let mut radius = trust;
    let mut best_candidate: Option<(Vec<f64>, QpCertificate, f64, f64, f64, usize)> = None;
    for it in 0..30 {
        let (candidate, cert) = match solve_subproblem(
            mesh,
            center,
            &current,
            target,
            directions,
            weights,
            gravity,
            phi,
            density,
            mass,
            radius,
            tensor_constraints,
        ) {
            Ok(v) => v,
            Err(e) => {
                radius *= 0.5;
                if radius < trust * 1e-6 {
                    if let Some((
                        radii,
                        certificate,
                        mass_residual,
                        slope_residual,
                        tensor_yield_residual,
                        sqp_it,
                    )) = best_candidate
                    {
                        return Ok(MeshProjection {
                            radii,
                            certificate,
                            mass_residual,
                            slope_residual,
                            sqp_iterations: sqp_it,
                            tensor_yield_residual,
                        });
                    }
                    return Err(e);
                }
                continue;
            }
        };
        let (m, s) = true_mesh_residuals(
            mesh, center, &candidate, directions, weights, gravity, phi, density, mass,
        )?;
        let tensor_residual = tensor_constraints
            .map(|constraints| {
                constraints
                    .iter()
                    .map(|constraint| {
                        constraint
                            .certificate()
                            .map(|certificate| certificate.residual)
                            .unwrap_or(f64::INFINITY)
                    })
                    .fold(0.0, f64::max)
            })
            .unwrap_or(0.0);
        let slope_acceptable = s <= slope_tol || s <= initial_slope + 0.05;
        if m <= mass_tol && slope_acceptable && tensor_residual <= 0.0 {
            return Ok(MeshProjection {
                radii: candidate,
                certificate: cert,
                mass_residual: m,
                slope_residual: s,
                sqp_iterations: it + 1,
                tensor_yield_residual: tensor_residual,
            });
        }
        if m <= mass_tol && tensor_residual <= 0.0 {
            if let Some((_, _, _, best_s, _, _)) = &best_candidate {
                if s < *best_s {
                    best_candidate = Some((
                        candidate.clone(),
                        cert.clone(),
                        m,
                        s,
                        tensor_residual,
                        it + 1,
                    ));
                }
            } else {
                best_candidate = Some((
                    candidate.clone(),
                    cert.clone(),
                    m,
                    s,
                    tensor_residual,
                    it + 1,
                ));
            }
        }
        current = candidate;
        radius = (radius * 0.75).max(trust * 1e-5);
    }
    if let Some((
        radii,
        certificate,
        mass_residual,
        slope_residual,
        tensor_yield_residual,
        sqp_it,
    )) = best_candidate
    {
        return Ok(MeshProjection {
            radii,
            certificate,
            mass_residual,
            slope_residual,
            sqp_iterations: sqp_it,
            tensor_yield_residual,
        });
    }
    Err("SQP true mass/slope/tensor-yield acceptance failed after 30 relinearizations".into())
}
#[cfg(test)]
mod tests {
    use super::*;
    use super::super::mass::evaluate_mass_at_radii;
    #[test]
    fn mesh_projection_accepts_unchanged_feasible_tetrahedron() {
        let mesh = TriMesh {
            vertices: vec![
                MeshVec::new(1.0, 0.0, 0.0),
                MeshVec::new(0.0, 1.0, 0.0),
                MeshVec::new(0.0, 0.0, 1.0),
                MeshVec::new(-1.0, -1.0, -1.0),
            ],
            faces: vec![[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]],
        };
        let directions = vec![
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-1.0 / 3.0_f64.sqrt(); 3],
        ];
        let radii = vec![1.0, 1.0, 1.0, 3.0_f64.sqrt()];
        let weights = vec![1.0; 4];
        let mass = evaluate_mass_at_radii(&radii, &directions, &weights, &|_| 1.0);
        let gravity = directions.clone();
        let phi = vec![std::f64::consts::FRAC_PI_2; 4];
        let projection = project_mesh(
            &mesh,
            [0.0, 0.0, 0.0],
            &radii,
            &radii,
            &directions,
            &weights,
            &gravity,
            &phi,
            &|_| 1.0,
            mass,
            0.1,
            1e-8,
            1e-8,
        )
        .expect("unchanged feasible geometry must pass mesh SQP");
        assert!(projection.certificate.residual() < 1e-6);
        assert!(projection.mass_residual < 1e-8);
        assert!(projection.slope_residual <= 1e-8);
    }
    #[test]
    fn exact_tensor_yield_constraint_is_checked_by_projection_layer() {
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
        let violating = TensorYieldConstraint {
            stress_deviator: [[10.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ..feasible
        };
        assert!(violating.certificate().unwrap().residual > 0.0);
    }
}
