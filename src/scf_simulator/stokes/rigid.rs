//! ρ_K-weighted rigid constraint `B_rb` (MS A.3). Rows are
//! `∫ ρ B_α dV` and `∫ ρ x × B_α dV`. The matrix is never a planted identity.

use nalgebra::DMatrix;

use crate::common::density::CauchyDensity;
use crate::common::geometry::TriMesh;
use crate::common::math::DVec3;
use crate::rtfp_solver::backend::hw_rt::HardwareMesh;
use crate::scf_simulator::basis::jacobi::psi_derivative;
use crate::scf_simulator::basis::layout::{dof_index, velocity_dof_count};
use crate::scf_simulator::error::ScfError;

#[derive(Clone, Debug)]
pub struct RigidReport {
    pub matrix: DMatrix<f64>,
    pub rank: usize,
    pub sigma_min: f64,
    pub force: [f64; 3],
    pub torque: [f64; 3],
}

/// Assemble `B_rb` on `mesh` with density `density`. `samples` are `(x, g_eff, weight)`
/// for the force and torque compatibility residuals; an empty list reports a zero load.
pub fn assemble_rigid_constraint(
    l_max: usize,
    mesh: &TriMesh,
    density: &CauchyDensity,
    samples: &[([f64; 3], [f64; 3], f64)],
) -> Result<RigidReport, ScfError> {
    let n_u = velocity_dof_count(l_max);
    if n_u < 6 {
        return Err(ScfError::rigid_rank_deficient(
            "rank 0 sigma_min 0 velocity space is smaller than six rigid modes",
        ));
    }
    let scene = HardwareMesh::build(mesh).map_err(|err| ScfError::model_not_closed(err.detail))?;
    let quad = volume_samples(mesh, &scene)?;
    if quad.is_empty() {
        return Err(ScfError::rigid_rank_deficient(
            "rigid constraint quadrature has no interior samples",
        ));
    }
    let mut matrix = DMatrix::<f64>::zeros(6, n_u);
    for sample in &quad {
        let rho = density.rho(sample.position);
        if !rho.is_finite() {
            return Err(ScfError::model_not_closed(
                "density in the rigid constraint integral is not finite",
            ));
        }
        let weight = rho * sample.weight;
        if weight == 0.0 {
            continue;
        }
        for (column, mode) in mode_order(l_max).into_iter().enumerate() {
            let basis = basis_vector(mode, sample.position, sample.radius);
            if !basis.is_finite() {
                return Err(ScfError::model_not_closed(
                    "a rigid-constraint basis vector is not finite",
                ));
            }
            let moment = sample.position.cross(basis);
            for axis in 0..3 {
                matrix[(axis, column)] += weight * basis.to_array()[axis];
                matrix[(3 + axis, column)] += weight * moment.to_array()[axis];
            }
        }
    }
    let singular = matrix.clone().svd(false, false).singular_values;
    if singular.iter().any(|value| !value.is_finite()) {
        return Err(ScfError::rigid_rank_deficient(
            "singular values of B_rb are not finite",
        ));
    }
    let sigma_max = singular.iter().copied().fold(0.0_f64, f64::max);
    let cutoff = (1e-8 * sigma_max).max(1e-12);
    let rank = singular.iter().filter(|value| **value > cutoff).count();
    let sigma_min = singular.iter().copied().fold(f64::INFINITY, f64::min);
    if rank < 6 {
        return Err(ScfError::rigid_rank_deficient(format!(
            "rank(B_rb) = {rank}, sigma_min = {sigma_min}"
        )));
    }
    let (force, torque) = compatibility(samples);
    Ok(RigidReport {
        matrix,
        rank,
        sigma_min,
        force,
        torque,
    })
}

pub fn singular_values(l_max: usize) -> Result<Vec<f64>, ScfError> {
    let mesh = TriMesh::unit_cube();
    let density =
        CauchyDensity::uniform(1.0).map_err(|err| ScfError::model_not_closed(err.detail))?;
    let report = assemble_rigid_constraint(l_max, &mesh, &density, &[])?;
    Ok(report
        .matrix
        .svd(false, false)
        .singular_values
        .iter()
        .copied()
        .collect())
}

pub fn compatibility(samples: &[([f64; 3], [f64; 3], f64)]) -> ([f64; 3], [f64; 3]) {
    let mut force = [0.0; 3];
    let mut torque = [0.0; 3];
    for (position, gravity, weight) in samples {
        let x = DVec3::from_array(*position);
        let g = DVec3::from_array(*gravity);
        let moment = x.cross(g);
        for axis in 0..3 {
            force[axis] += weight * g.to_array()[axis];
            torque[axis] += weight * moment.to_array()[axis];
        }
    }
    (force, torque)
}

struct VolumeSample {
    position: DVec3,
    radius: f64,
    weight: f64,
}

fn volume_samples(mesh: &TriMesh, scene: &HardwareMesh) -> Result<Vec<VolumeSample>, ScfError> {
    let mut pending = Vec::new();
    for face in 0..mesh.faces.len() {
        let [a, b, c] = mesh.triangle(face);
        let volume = a.dot(b.cross(c)) / 6.0;
        if volume.abs() <= 1e-18 {
            continue;
        }
        let centroid = (a + b + c) * 0.25;
        let Some(direction) = centroid.try_normalize() else {
            continue;
        };
        pending.push((centroid, direction, volume));
    }
    let dirs: Vec<DVec3> = pending.iter().map(|(_, direction, _)| *direction).collect();
    let rows = scene
        .intersect_rays(DVec3::ZERO, &dirs)
        .map_err(|err| ScfError::model_not_closed(err.detail))?;
    let mut samples = Vec::new();
    for ((centroid, _, volume), row) in pending.into_iter().zip(rows) {
        let radius = row
            .iter()
            .map(|hit| hit.t)
            .filter(|t| *t > 0.0 && t.is_finite())
            .fold(0.0_f64, f64::max);
        if radius <= 0.0 {
            continue;
        }
        samples.push(VolumeSample {
            position: centroid,
            radius,
            weight: volume,
        });
    }
    Ok(samples)
}

#[derive(Clone, Copy)]
struct Mode {
    l: usize,
    m: i32,
    poloidal: bool,
    k: usize,
}

fn mode_order(l_max: usize) -> Vec<Mode> {
    let mut modes = Vec::new();
    for l in 1..=l_max {
        for m_index in 0..(2 * l + 1) {
            let m = m_index as i32 - l as i32;
            for k in 0..4 {
                modes.push(Mode {
                    l,
                    m,
                    poloidal: true,
                    k,
                });
            }
            for k in 0..4 {
                modes.push(Mode {
                    l,
                    m,
                    poloidal: false,
                    k,
                });
            }
        }
    }
    debug_assert_eq!(modes.len(), velocity_dof_count(l_max));
    debug_assert_eq!(dof_index(1, 0, 0), 0);
    modes
}

fn basis_vector(mode: Mode, position: DVec3, radius: f64) -> DVec3 {
    let r = position.length();
    if r <= 1e-14 * radius.max(1.0) || radius <= 0.0 {
        return DVec3::ZERO;
    }
    let zeta = (r / radius).clamp(0.0, 2.0);
    let (psi, dpsi) = radial_basis(mode.l, mode.k, zeta);
    let dir = position * (1.0 / r);
    let (theta, phi, sin_theta) = spherical_angles(dir);
    let (y, dy_theta, dy_phi) = real_harmonics(mode.l, mode.m, theta, phi);
    let (e_theta, e_phi) = spherical_basis(theta, phi);
    if mode.poloidal {
        let radial_factor = psi + zeta * dpsi;
        let p_r = mode.l as f64 * (mode.l as f64 + 1.0) / r * radius * radius * psi * y;
        let p_theta = radius * radius / r * radial_factor * dy_theta;
        let p_phi = if sin_theta > 1e-8 {
            radius * radius / (r * sin_theta) * radial_factor * dy_phi
        } else {
            0.0
        };
        dir * p_r + e_theta * p_theta + e_phi * p_phi
    } else {
        let t_theta = if sin_theta > 1e-8 {
            radius * psi / sin_theta * dy_phi
        } else {
            0.0
        };
        let t_phi = -radius * psi * dy_theta;
        e_theta * t_theta + e_phi * t_phi
    }
}

fn radial_basis(l: usize, k: usize, zeta: f64) -> (f64, f64) {
    psi_derivative(l, k, zeta)
}

pub fn velocity_bases(l_max: usize, position: DVec3, radius: f64) -> Vec<DVec3> {
    mode_order(l_max)
        .into_iter()
        .map(|mode| basis_vector(mode, position, radius))
        .collect()
}

fn spherical_angles(dir: DVec3) -> (f64, f64, f64) {
    let cos_theta = dir.z.clamp(-1.0, 1.0);
    let theta = cos_theta.acos();
    let phi = dir.y.atan2(dir.x);
    (theta, phi, theta.sin())
}

fn spherical_basis(theta: f64, phi: f64) -> (DVec3, DVec3) {
    let (st, ct) = (theta.sin(), theta.cos());
    let (sp, cp) = (phi.sin(), phi.cos());
    let e_theta = DVec3::new(ct * cp, ct * sp, -st);
    let e_phi = DVec3::new(-sp, cp, 0.0);
    (e_theta, e_phi)
}

pub(crate) fn real_harmonics(l: usize, m: i32, theta: f64, phi: f64) -> (f64, f64, f64) {
    let abs_m = m.unsigned_abs() as usize;
    let p = assoc_legendre(l, abs_m, theta.cos());
    let norm = harmonic_norm(l, abs_m);
    let azimuth = abs_m as f64 * phi;
    let (angular, dphi) = match m.cmp(&0) {
        std::cmp::Ordering::Greater => {
            let scale = norm * std::f64::consts::SQRT_2 * p;
            (
                scale * azimuth.cos(),
                scale * (-(abs_m as f64) * azimuth.sin()),
            )
        }
        std::cmp::Ordering::Less => {
            let scale = norm * std::f64::consts::SQRT_2 * p;
            (
                scale * azimuth.sin(),
                scale * (abs_m as f64 * azimuth.cos()),
            )
        }
        std::cmp::Ordering::Equal => (norm * p, 0.0),
    };
    let dp = d_legendre_dtheta(l, abs_m, theta);
    let dtheta = if m == 0 {
        norm * dp
    } else {
        norm * std::f64::consts::SQRT_2 * dp * if m > 0 { azimuth.cos() } else { azimuth.sin() }
    };
    (angular, dtheta, dphi)
}

fn harmonic_norm(l: usize, m: usize) -> f64 {
    let mut ratio = 1.0;
    if m > 0 {
        for i in (l + 1 - m)..=(l + m) {
            ratio /= i as f64;
        }
    }
    ((2.0 * l as f64 + 1.0) / (4.0 * std::f64::consts::PI) * ratio).sqrt()
}

fn assoc_legendre(l: usize, m: usize, x: f64) -> f64 {
    if m > l {
        return 0.0;
    }
    let mut pmm = 1.0;
    if m > 0 {
        let somx2 = (1.0 - x * x).max(0.0).sqrt();
        let mut fact = 1.0;
        for _ in 1..=m {
            pmm *= -fact * somx2;
            fact += 2.0;
        }
    }
    if l == m {
        return pmm;
    }
    let mut pmmp1 = x * (2.0 * m as f64 + 1.0) * pmm;
    if l == m + 1 {
        return pmmp1;
    }
    let mut pll = 0.0;
    for ll in (m + 2)..=l {
        pll = (x * (2.0 * ll as f64 - 1.0) * pmmp1 - (ll as f64 + m as f64 - 1.0) * pmm)
            / (ll as f64 - m as f64);
        pmm = pmmp1;
        pmmp1 = pll;
    }
    pll
}

fn d_legendre_dtheta(l: usize, m: usize, theta: f64) -> f64 {
    let step = 1e-6;
    let plus = assoc_legendre(l, m, (theta + step).cos());
    let minus = assoc_legendre(l, m, (theta - step).cos());
    (plus - minus) / (2.0 * step)
}
