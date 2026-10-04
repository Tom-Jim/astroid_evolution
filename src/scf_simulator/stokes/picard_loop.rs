//! One geological step of the Stokes–Poisson Picard loop (MS 10.4.2, 8.7).
//! `M` and `K` are reassembled from the current mesh, `ρ_K` and `η`.

use nalgebra::{DMatrix, DVector};

use crate::common::density::CauchyDensity;
use crate::common::geometry::TriMesh;
use crate::common::math::DVec3;
use crate::rtfp_solver::backend::hw_rt::HardwareMesh;
use crate::scf_simulator::basis::jacobi::psi;
use crate::scf_simulator::basis::layout::velocity_dof_count;
use crate::scf_simulator::error::ScfError;
use crate::scf_simulator::rheology::closure::flow_viscosity;
use crate::scf_simulator::rheology::picard::apply_under_relaxation;

use super::range_space::range_space_velocity;
use super::rigid::{assemble_rigid_constraint, real_harmonics, velocity_bases};

#[derive(Clone, Copy, Debug)]
pub struct PicardMaterial {
    pub cohesion: f64,
    pub mu: f64,
    pub gamma_eps: f64,
    pub p_min: f64,
    pub gravity: f64,
    pub tide_coefficient: f64,
    pub omega_z: f64,
    pub relaxation: f64,
    pub aug_lagrange: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct GravSample {
    pub position: [f64; 3],
    pub g_eff: [f64; 3],
}

#[derive(Clone, Debug)]
pub struct PicardOutput {
    pub coefficients: Vec<f64>,
    pub pressure: Vec<f64>,
    pub vertex_velocity: Vec<[f64; 3]>,
    pub vertex_pressure: Vec<f64>,
    pub vertex_strain: Vec<[f64; 9]>,
    pub shear_rate: Vec<f64>,
    pub picard_residual: f64,
    pub pcg_residual: f64,
    pub strong_residual: f64,
    pub tension_fraction: f64,
    pub eta: Vec<f64>,
}

struct Quad {
    position: DVec3,
    radius: f64,
    weight: f64,
    g_eff: DVec3,
}

struct PMode {
    l: usize,
    m: i32,
    k: usize,
}

pub fn picard_solve(
    l_max: usize,
    k_p: usize,
    mesh: &TriMesh,
    density: &CauchyDensity,
    gravity: &[GravSample],
    material: PicardMaterial,
    iterations: usize,
    mut velocity: Vec<f64>,
) -> Result<PicardOutput, ScfError> {
    let n_u = velocity_dof_count(l_max);
    if velocity.len() != n_u {
        velocity = vec![0.0; n_u];
    }
    let modes = pressure_modes(l_max, k_p.max(1));
    let mut pressure = vec![0.0; modes.len()];
    let quad = quadrature(mesh, density, gravity)?;
    let mut picard_residual = 0.0;
    let mut pcg_residual = 0.0;
    let mut strong_residual = 0.0;
    let mut tension_fraction = 0.0;
    let mut eta = vec![material.p_min; quad.len()];
    for _ in 0..iterations.max(1) {
        let previous = velocity.clone();
        let (p_next, pcg_p, strong, tension) = poisson(
            l_max, mesh, density, &quad, &modes, &velocity, &eta, material,
        )?;
        pressure = p_next;
        pcg_residual = pcg_p;
        strong_residual = strong;
        tension_fraction = tension;
        eta = viscosities(l_max, &quad, &modes, &pressure, &velocity, material);
        let (stiffness, load) = viscous_system(l_max, density, &quad, &eta)?;
        let rigid_samples: Vec<([f64; 3], [f64; 3], f64)> = quad
            .iter()
            .map(|sample| {
                let rho = density.rho(sample.position);
                (
                    sample.position.to_array(),
                    sample.g_eff.to_array(),
                    rho * sample.weight,
                )
            })
            .collect();
        let rigid = assemble_rigid_constraint(l_max, mesh, density, &rigid_samples)?;
        let solved = range_space_velocity(&stiffness, &rigid.matrix, &load, material.aug_lagrange)
            .map_err(ScfError::model_not_closed)?;
        pcg_residual = pcg_residual.max(solved.pcg_residual);
        let relaxed =
            apply_under_relaxation(&previous, solved.velocity.as_slice(), material.relaxation);
        picard_residual = k_increment(&stiffness, &previous, &relaxed);
        velocity = relaxed;
        eprintln!(
            "picard_residual {picard_residual} pcg_residual {pcg_residual} strong_residual {strong_residual} tension_fraction {tension_fraction}"
        );
    }
    let mut vertex_velocity = Vec::with_capacity(mesh.vertices.len());
    let mut vertex_pressure = Vec::with_capacity(mesh.vertices.len());
    let mut vertex_strain = Vec::with_capacity(mesh.vertices.len());
    let mut shear_rate = Vec::with_capacity(mesh.vertices.len());
    for vertex in &mesh.vertices {
        let radius = vertex.length().max(1e-8);
        let speed = reconstruct(l_max, &velocity, *vertex, radius);
        let rate = strain_of(l_max, &velocity, *vertex, radius);
        let shear = (2.0 * double_dot(&rate, &rate)).sqrt();
        let probe = Quad {
            position: *vertex,
            radius,
            weight: 0.0,
            g_eff: DVec3::ZERO,
        };
        vertex_pressure.push(pressure_at(&modes, &pressure, &probe));
        vertex_strain.push(rate);
        shear_rate.push(shear);
        vertex_velocity.push(speed.to_array());
    }
    Ok(PicardOutput {
        coefficients: velocity,
        pressure,
        vertex_velocity,
        vertex_pressure,
        vertex_strain,
        shear_rate,
        picard_residual,
        pcg_residual,
        strong_residual,
        tension_fraction,
        eta,
    })
}

/// Kinetic mass `∫ ρ B_α·B_β dV` and viscous `K` for the current geometry and density.
pub fn kinetic_mass(
    l_max: usize,
    mesh: &TriMesh,
    density: &CauchyDensity,
) -> Result<DMatrix<f64>, ScfError> {
    let quad = quadrature(mesh, density, &[])?;
    let n = velocity_dof_count(l_max);
    let mut mass = DMatrix::zeros(n, n);
    for sample in &quad {
        let rho = density.rho(sample.position);
        let bases = velocity_bases(l_max, sample.position, sample.radius);
        for a in 0..n {
            for b in a..n {
                let value = rho * sample.weight * bases[a].dot(bases[b]);
                mass[(a, b)] += value;
                if a != b {
                    mass[(b, a)] += value;
                }
            }
        }
    }
    Ok(mass)
}

fn viscous_system(
    l_max: usize,
    density: &CauchyDensity,
    quad: &[Quad],
    eta: &[f64],
) -> Result<(DMatrix<f64>, DVector<f64>), ScfError> {
    let n = velocity_dof_count(l_max);
    let mut stiffness = DMatrix::<f64>::zeros(n, n);
    let mut load = DVector::<f64>::zeros(n);
    for (sample, eta_q) in quad.iter().zip(eta) {
        let rho = density.rho(sample.position);
        let bases = velocity_bases(l_max, sample.position, sample.radius);
        let strains = basis_strains(l_max, sample);
        for a in 0..n {
            load[a] += sample.weight * rho * bases[a].dot(sample.g_eff);
            for b in a..n {
                let value = 2.0 * eta_q * sample.weight * double_dot(&strains[a], &strains[b]);
                stiffness[(a, b)] += value;
                if a != b {
                    stiffness[(b, a)] += value;
                }
            }
        }
    }
    Ok((stiffness, load))
}

fn poisson(
    l_max: usize,
    mesh: &TriMesh,
    density: &CauchyDensity,
    quad: &[Quad],
    modes: &[PMode],
    velocity: &[f64],
    eta: &[f64],
    material: PicardMaterial,
) -> Result<(Vec<f64>, f64, f64, f64), ScfError> {
    let lift = boundary_lift(l_max, mesh, density, velocity, eta, material)?;
    let n = modes.len();
    let mut stiffness = DMatrix::zeros(n, n);
    let mut rhs = DVector::zeros(n);
    let mut force_norm = 0.0;
    let mut strong_num = 0.0;
    let mut strong_den = 0.0;
    let mut tension_num = 0.0;
    let mut tension_den = 0.0;
    let mut pressure_values = Vec::with_capacity(quad.len());
    let grads: Vec<Vec<DVec3>> = modes
        .iter()
        .map(|mode| {
            quad.iter()
                .map(|sample| gradient_fd(|x, radius| pressure_basis(mode, x, radius), sample))
                .collect()
        })
        .collect();
    for (q, sample) in quad.iter().enumerate() {
        let div_tide = tide_divergence(density, sample.position, material);
        let div_tau = flow_divergence(l_max, velocity, eta, quad, q, material);
        let grad_pb = gradient_fd(|x, radius| lift_value(&lift, x, radius), sample);
        let body = sample.g_eff * density.rho(sample.position) + div_tide + div_tau - grad_pb;
        force_norm += sample.weight * body.length_squared();
        for a in 0..n {
            rhs[a] += sample.weight * body.dot(grads[a][q]);
            for b in a..n {
                let value = sample.weight * grads[a][q].dot(grads[b][q]);
                stiffness[(a, b)] += value;
                if a != b {
                    stiffness[(b, a)] += value;
                }
            }
        }
    }
    for i in 0..n {
        stiffness[(i, i)] += 1e-12;
    }
    let (coeffs, pcg) = conjugate_gradient(&stiffness, &rhs);
    for sample in quad {
        let mut p0 = 0.0;
        let mut grad = DVec3::ZERO;
        for (a, mode) in modes.iter().enumerate() {
            p0 += coeffs[a] * pressure_basis(mode, sample.position, sample.radius);
        }
        let p = lift_value(&lift, sample.position, sample.radius) + p0;
        for (a, mode) in modes.iter().enumerate() {
            grad =
                grad + gradient_fd(|x, radius| pressure_basis(mode, x, radius), sample) * coeffs[a];
        }
        grad = grad + gradient_fd(|x, radius| lift_value(&lift, x, radius), sample);
        let div_tide = tide_divergence(density, sample.position, material);
        let body = sample.g_eff * density.rho(sample.position) + div_tide;
        strong_num += sample.weight * (body - grad).length_squared();
        strong_den += sample.weight * body.length_squared();
        let negative = p.min(0.0);
        tension_num += sample.weight * negative * negative;
        tension_den += sample.weight * p * p;
        pressure_values.push(p);
    }
    let _ = (force_norm, pressure_values);
    let strong = (strong_num / strong_den.max(1e-30)).sqrt();
    let tension = (tension_num / tension_den.max(1e-30)).sqrt();
    Ok((coeffs, pcg, strong, tension))
}

fn viscosities(
    l_max: usize,
    quad: &[Quad],
    modes: &[PMode],
    pressure: &[f64],
    velocity: &[f64],
    material: PicardMaterial,
) -> Vec<f64> {
    quad.iter()
        .map(|sample| {
            let p = pressure_at(modes, pressure, sample);
            let p_tilde = p.max(0.0).max(material.p_min);
            let rate = strain_of(l_max, velocity, sample.position, sample.radius);
            let shear = (2.0 * double_dot(&rate, &rate)).sqrt();
            flow_viscosity(
                material.cohesion,
                material.mu,
                p_tilde,
                shear,
                material.gamma_eps,
            )
        })
        .collect()
}

fn pressure_at(modes: &[PMode], coeffs: &[f64], sample: &Quad) -> f64 {
    modes
        .iter()
        .zip(coeffs)
        .map(|(mode, coeff)| coeff * pressure_basis(mode, sample.position, sample.radius))
        .sum()
}

fn tide_divergence(density: &CauchyDensity, position: DVec3, material: PicardMaterial) -> DVec3 {
    let grad = density.grad(position);
    grad * (-8.0 * std::f64::consts::PI / 3.0 * material.gravity * material.tide_coefficient)
}

fn flow_divergence(
    l_max: usize,
    velocity: &[f64],
    eta: &[f64],
    quad: &[Quad],
    index: usize,
    material: PicardMaterial,
) -> DVec3 {
    if velocity.iter().all(|c| c.abs() < 1e-18) {
        return DVec3::ZERO;
    }
    let sample = &quad[index];
    let h = (sample.radius * 1e-4).max(1e-6);
    let eta0 = eta.get(index).copied().unwrap_or(material.gamma_eps);
    let mut lap = DVec3::ZERO;
    for axis in 0..3 {
        let step = axis_vector(axis) * h;
        let plus = reconstruct(l_max, velocity, sample.position + step, sample.radius);
        let minus = reconstruct(l_max, velocity, sample.position - step, sample.radius);
        lap = lap
            + (plus + minus - reconstruct(l_max, velocity, sample.position, sample.radius) * 2.0)
                * (1.0 / (h * h));
    }
    let rate = strain_of(l_max, velocity, sample.position, sample.radius);
    let mut grad_eta = DVec3::ZERO;
    for axis in 0..3 {
        let step = axis_vector(axis) * h;
        let eta_plus = local_eta(
            l_max,
            velocity,
            sample.position + step,
            sample.radius,
            material,
        );
        let eta_minus = local_eta(
            l_max,
            velocity,
            sample.position - step,
            sample.radius,
            material,
        );
        let slope = (eta_plus - eta_minus) / (2.0 * h);
        grad_eta = grad_eta + axis_vector(axis) * slope;
    }
    let mut extra = DVec3::ZERO;
    for i in 0..3 {
        for j in 0..3 {
            extra = extra + axis_vector(i) * (2.0 * rate[i * 3 + j] * grad_eta.to_array()[j]);
        }
    }
    lap * eta0 + extra
}

fn local_eta(
    l_max: usize,
    velocity: &[f64],
    position: DVec3,
    radius: f64,
    material: PicardMaterial,
) -> f64 {
    let rate = strain_of(l_max, velocity, position, radius);
    let shear = (2.0 * double_dot(&rate, &rate)).sqrt();
    flow_viscosity(
        material.cohesion,
        material.mu,
        material.p_min,
        shear,
        material.gamma_eps,
    )
}

fn boundary_lift(
    l_max: usize,
    mesh: &TriMesh,
    density: &CauchyDensity,
    velocity: &[f64],
    eta: &[f64],
    material: PicardMaterial,
) -> Result<Vec<f64>, ScfError> {
    let mut harmonics = Vec::new();
    for l in 0..=l_max {
        for m in -(l as i32)..=l as i32 {
            harmonics.push((l, m));
        }
    }
    let n = harmonics.len();
    let mut normal = DMatrix::zeros(n, n);
    let mut rhs = DVector::zeros(n);
    let scene = HardwareMesh::build(mesh).map_err(|err| ScfError::model_not_closed(err.detail))?;
    for face in 0..mesh.faces.len() {
        let [a, b, c] = mesh.triangle(face);
        let area = (b - a).cross(c - a).length() * 0.5;
        let point = (a + b + c) * (1.0 / 3.0);
        let normal_v = (b - a)
            .cross(c - a)
            .try_normalize()
            .unwrap_or(DVec3::new(0.0, 0.0, 1.0));
        let dir = point.try_normalize().unwrap_or(DVec3::new(0.0, 0.0, 1.0));
        let hits = scene
            .intersect_one(DVec3::ZERO, dir)
            .map_err(|err| ScfError::model_not_closed(err.detail))?;
        let radius = hits
            .into_iter()
            .map(|hit| hit.t)
            .filter(|t| *t > 0.0)
            .fold(point.length(), f64::max);
        let tide = centrifugal_traction(normal_v, material);
        let flow = flow_traction(l_max, velocity, point, radius, normal_v, eta, material);
        let p_gamma = tide + flow + density.rho(point) * 0.0;
        let (theta, phi) = angles(dir);
        for (col, (l, m)) in harmonics.iter().enumerate() {
            let y = real_harmonics(*l, *m, theta, phi).0;
            rhs[col] += area * p_gamma * y;
            for (row, (l2, m2)) in harmonics.iter().enumerate() {
                let y2 = real_harmonics(*l2, *m2, theta, phi).0;
                normal[(row, col)] += area * y * y2;
            }
        }
    }
    for i in 0..n {
        normal[(i, i)] += 1e-12;
    }
    Ok(conjugate_gradient(&normal, &rhs).0)
}

fn lift_value(coeffs: &[f64], position: DVec3, radius: f64) -> f64 {
    let r = position.length();
    let zeta = if radius > 0.0 {
        (r / radius).clamp(0.0, 2.0)
    } else {
        0.0
    };
    let dir = if r > 0.0 {
        position * (1.0 / r)
    } else {
        DVec3::new(0.0, 0.0, 1.0)
    };
    let (theta, phi) = angles(dir);
    let mut acc = 0.0;
    let mut slot = 0;
    let l_max = ((coeffs.len() as f64).sqrt() as usize).saturating_sub(1);
    for l in 0..=l_max {
        for m in -(l as i32)..=l as i32 {
            if slot < coeffs.len() {
                acc += coeffs[slot] * zeta.powi(l as i32) * real_harmonics(l, m, theta, phi).0;
                slot += 1;
            }
        }
    }
    acc
}

fn centrifugal_traction(normal: DVec3, material: PicardMaterial) -> f64 {
    let w2 = material.omega_z * material.omega_z;
    let h = [w2, w2, 0.0];
    let trace = (h[0] + h[1] + h[2]) / 3.0;
    let dev = [h[0] - trace, h[1] - trace, h[2] - trace];
    let n = normal.to_array();
    material.tide_coefficient * (dev[0] * n[0] * n[0] + dev[1] * n[1] * n[1] + dev[2] * n[2] * n[2])
}

fn flow_traction(
    l_max: usize,
    velocity: &[f64],
    position: DVec3,
    radius: f64,
    normal: DVec3,
    eta: &[f64],
    material: PicardMaterial,
) -> f64 {
    let rate = strain_of(l_max, velocity, position, radius);
    let eta0 = eta
        .iter()
        .copied()
        .fold(0.0, f64::max)
        .max(material.gamma_eps);
    let n = normal.to_array();
    let mut traction = 0.0;
    for i in 0..3 {
        for j in 0..3 {
            traction += n[i] * (2.0 * eta0 * rate[i * 3 + j]) * n[j];
        }
    }
    traction
}

fn pressure_modes(l_max: usize, k_p: usize) -> Vec<PMode> {
    let mut modes = Vec::new();
    for l in 0..=l_max {
        for m in -(l as i32)..=l as i32 {
            for k in 0..k_p.min(4) {
                modes.push(PMode { l, m, k });
            }
        }
    }
    modes
}

fn pressure_basis(mode: &PMode, position: DVec3, radius: f64) -> f64 {
    let r = position.length();
    if radius <= 0.0 {
        return 0.0;
    }
    let zeta = (r / radius).clamp(0.0, 2.0);
    let dir = if r > 1e-14 {
        position * (1.0 / r)
    } else {
        DVec3::new(0.0, 0.0, 1.0)
    };
    let (theta, phi) = angles(dir);
    (1.0 - zeta) * psi(mode.l, mode.k, zeta) * real_harmonics(mode.l, mode.m, theta, phi).0
}

fn quadrature(
    mesh: &TriMesh,
    density: &CauchyDensity,
    gravity: &[GravSample],
) -> Result<Vec<Quad>, ScfError> {
    let scene = HardwareMesh::build(mesh).map_err(|err| ScfError::model_not_closed(err.detail))?;
    let alpha = 0.5854101966249685;
    let beta = 0.1381966011250105;
    let bary = [
        [alpha, beta, beta, beta],
        [beta, alpha, beta, beta],
        [beta, beta, alpha, beta],
        [beta, beta, beta, alpha],
    ];
    let mut quad = Vec::new();
    for face in 0..mesh.faces.len() {
        let [a, b, c] = mesh.triangle(face);
        let volume = a.dot(b.cross(c)) / 6.0;
        if volume.abs() <= 1e-18 {
            continue;
        }
        for coords in bary {
            let position = a * coords[1] + b * coords[2] + c * coords[3];
            let dir = match position.try_normalize() {
                Some(dir) => dir,
                None => continue,
            };
            let hits = scene
                .intersect_one(DVec3::ZERO, dir)
                .map_err(|err| ScfError::model_not_closed(err.detail))?;
            let radius = hits
                .into_iter()
                .map(|hit| hit.t)
                .filter(|t| *t > 0.0)
                .fold(0.0, f64::max);
            if radius <= 0.0 || !density.rho(position).is_finite() {
                continue;
            }
            quad.push(Quad {
                position,
                radius,
                weight: volume.abs() * 0.25,
                g_eff: nearest_gravity(position, gravity),
            });
        }
    }
    if quad.is_empty() {
        return Err(ScfError::model_not_closed(
            "Stokes quadrature has no interior samples",
        ));
    }
    Ok(quad)
}

fn nearest_gravity(position: DVec3, gravity: &[GravSample]) -> DVec3 {
    let mut best = f64::INFINITY;
    let mut value = DVec3::ZERO;
    for sample in gravity {
        let dist = (position - DVec3::from_array(sample.position)).length_squared();
        if dist < best {
            best = dist;
            value = DVec3::from_array(sample.g_eff);
        }
    }
    value
}

fn reconstruct(l_max: usize, velocity: &[f64], position: DVec3, radius: f64) -> DVec3 {
    let bases = velocity_bases(l_max, position, radius);
    let mut field = DVec3::ZERO;
    for (coeff, basis) in velocity.iter().zip(bases) {
        field = field + basis * *coeff;
    }
    field
}

fn strain_of(l_max: usize, velocity: &[f64], position: DVec3, radius: f64) -> [f64; 9] {
    let h = (radius * 1e-4).max(1e-6);
    let center = reconstruct(l_max, velocity, position, radius);
    let mut grad = [0.0; 9];
    for axis in 0..3 {
        let step = axis_vector(axis) * h;
        let plus = reconstruct(l_max, velocity, position + step, radius);
        let minus = reconstruct(l_max, velocity, position - step, radius);
        let slope = (plus - minus) * (1.0 / (2.0 * h));
        let s = slope.to_array();
        for row in 0..3 {
            grad[row * 3 + axis] = s[row];
        }
    }
    let mut strain = [0.0; 9];
    for i in 0..3 {
        for j in 0..3 {
            strain[i * 3 + j] = 0.5 * (grad[i * 3 + j] + grad[j * 3 + i]);
        }
    }
    let _ = center;
    strain
}

fn basis_strains(l_max: usize, sample: &Quad) -> Vec<[f64; 9]> {
    let h = (sample.radius * 1e-4).max(1e-6);
    let n = velocity_dof_count(l_max);
    let mut grad = vec![[0.0; 9]; n];
    for axis in 0..3 {
        let step = axis_vector(axis) * h;
        let plus = velocity_bases(l_max, sample.position + step, sample.radius);
        let minus = velocity_bases(l_max, sample.position - step, sample.radius);
        for mode in 0..n {
            let slope = (plus[mode] - minus[mode]) * (1.0 / (2.0 * h));
            let s = slope.to_array();
            for row in 0..3 {
                grad[mode][row * 3 + axis] = s[row];
            }
        }
    }
    for mode in &mut grad {
        let raw = *mode;
        for i in 0..3 {
            for j in 0..3 {
                mode[i * 3 + j] = 0.5 * (raw[i * 3 + j] + raw[j * 3 + i]);
            }
        }
    }
    grad
}

fn gradient_fd(field: impl Fn(DVec3, f64) -> f64, sample: &Quad) -> DVec3 {
    let h = (sample.radius * 1e-4).max(1e-6);
    let mut grad = [0.0; 3];
    for axis in 0..3 {
        let step = axis_vector(axis) * h;
        let plus = field(sample.position + step, sample.radius);
        let minus = field(sample.position - step, sample.radius);
        grad[axis] = (plus - minus) / (2.0 * h);
    }
    DVec3::from_array(grad)
}

fn k_increment(stiffness: &DMatrix<f64>, previous: &[f64], next: &[f64]) -> f64 {
    let mut delta = 0.0;
    let mut scale = 0.0;
    for i in 0..next.len() {
        for j in 0..next.len() {
            let diff = next[i] - previous.get(i).copied().unwrap_or(0.0);
            let step = next[j] - previous.get(j).copied().unwrap_or(0.0);
            delta += diff * stiffness[(i, j)] * step;
            scale += next[i] * stiffness[(i, j)] * next[j];
        }
    }
    (delta.max(0.0) / scale.max(0.0).max(1e-30)).sqrt()
}

fn double_dot(a: &[f64; 9], b: &[f64; 9]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn conjugate_gradient(matrix: &DMatrix<f64>, rhs: &DVector<f64>) -> (Vec<f64>, f64) {
    let n = rhs.len();
    let mut x = DVector::zeros(n);
    let mut r = rhs.clone();
    let mut p = r.clone();
    let mut rr = r.dot(&r);
    if rr < 1e-30 {
        return (vec![0.0; n], 0.0);
    }
    for _ in 0..(n * 4).max(8) {
        let ap = matrix * &p;
        let denom = p.dot(&ap);
        if denom.abs() < 1e-30 {
            break;
        }
        let alpha = rr / denom;
        x += &p * alpha;
        r -= &ap * alpha;
        let next = r.dot(&r);
        if next.sqrt() <= 1e-10 * rhs.norm().max(1.0) {
            break;
        }
        p = &r + p * (next / rr);
        rr = next;
    }
    let residual = (matrix * &x - rhs).norm() / rhs.norm().max(1e-30);
    (x.iter().copied().collect(), residual)
}

fn angles(dir: DVec3) -> (f64, f64) {
    let theta = dir.z.clamp(-1.0, 1.0).acos();
    (theta, dir.y.atan2(dir.x))
}

fn axis_vector(axis: usize) -> DVec3 {
    match axis {
        0 => DVec3::new(1.0, 0.0, 0.0),
        1 => DVec3::new(0.0, 1.0, 0.0),
        _ => DVec3::new(0.0, 0.0, 1.0),
    }
}
