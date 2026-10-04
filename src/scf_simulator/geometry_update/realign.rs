//! Step 8b. Translate to the ρ_K centroid, diagonalize the inertia tensor,
//! align the maximum-inertia axis with e_z and the minimum with e_x, rotate
//! the Cauchy centers with the same proper rotation, then resample fixed rays.

use nalgebra::Matrix3;

use crate::common::density::{CauchyDensity, CauchyKernel};
use crate::common::geometry::TriMesh;
use crate::common::math::DVec3;
use crate::rtfp_solver::backend::hw_rt::HardwareMesh;
use crate::rtfp_solver::ray_radii::ray_radii;
use crate::rtfp_solver::volume::uniform_integrals;
use crate::scf_simulator::error::ScfError;

#[derive(Clone, Debug)]
pub struct Realignment {
    pub applied: bool,
    pub centroid: [f64; 3],
    pub rotation: [[f64; 3]; 3],
    pub det: f64,
    pub radii: Vec<f64>,
    pub density: CauchyDensity,
    pub centroid_residual: f64,
    pub ixz: f64,
    pub iyz: f64,
    pub i1: f64,
}

/// `eps_frame` and `eps_deg` are the MS 8b thresholds. A body already inside
/// the trigger band is returned unmoved so repeated resampling does not drift.
pub fn realign_body(
    mesh: &TriMesh,
    directions: &[DVec3],
    density: &CauchyDensity,
    eps_frame: f64,
    eps_deg: f64,
) -> Result<Realignment, ScfError> {
    let (mass, centroid, inertia) = density_moments(mesh, density)?;
    if !(mass.is_finite() && mass > 0.0) {
        return Err(ScfError::model_not_closed(
            "realignment mass is not positive",
        ));
    }
    let r_ref = mesh
        .vertices
        .iter()
        .map(|v| v.length())
        .fold(0.0_f64, f64::max)
        .max(1.0e-12);
    let products = (inertia[(0, 2)].abs(), inertia[(1, 2)].abs());
    let i1_now = inertia
        .symmetric_eigen()
        .eigenvalues
        .iter()
        .copied()
        .fold(0.0_f64, f64::max);
    let triggered = centroid.length() > eps_frame * r_ref / 2.0
        || products.0.max(products.1) > eps_frame * i1_now.max(1e-30) / 2.0;
    if !triggered {
        return Ok(Realignment {
            applied: false,
            centroid: centroid.to_array(),
            rotation: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            det: 1.0,
            radii: mesh.vertices.iter().map(|vertex| vertex.length()).collect(),
            density: density.clone(),
            centroid_residual: centroid.length(),
            ixz: inertia[(0, 2)],
            iyz: inertia[(1, 2)],
            i1: i1_now,
        });
    }
    let shifted = translate_mesh(mesh, centroid)?;
    let translated = rotate_density(density, centroid, identity())?;
    let (_, _, shifted_inertia) = density_moments(&shifted, &translated)?;
    let (rotation, i1) = principal_frame(&shifted_inertia, eps_deg)?;
    let det = rotation_det(rotation);
    if !(det.is_finite() && (det - 1.0).abs() <= 1e-8) {
        return Err(ScfError::model_not_closed(format!(
            "realignment det(Q) = {det}"
        )));
    }
    let rotated = rotate_mesh(&shifted, rotation)?;
    let density = rotate_density(density, centroid, rotation)?;
    let scene =
        HardwareMesh::build(&rotated).map_err(|err| ScfError::model_not_closed(err.detail))?;
    let radii = ray_radii(&scene, DVec3::ZERO, directions)
        .map_err(|err| ScfError::model_not_closed(err.to_string()))?;
    let resampled = star_mesh(directions, &radii, &rotated.faces)?;
    let (mass_after, centroid_after, inertia_after) = density_moments(&resampled, &density)?;
    let i1_after = inertia_after
        .symmetric_eigen()
        .eigenvalues
        .iter()
        .copied()
        .fold(0.0_f64, f64::max)
        .max(i1);
    let centroid_residual = centroid_after.length();
    let ixz = inertia_after[(0, 2)];
    let iyz = inertia_after[(1, 2)];
    if centroid_residual > eps_frame * r_ref
        || ixz.abs().max(iyz.abs()) > eps_frame * i1_after.max(1e-30)
    {
        return Err(ScfError::model_not_closed(format!(
            "realignment residual |x_cm| = {centroid_residual}, Ixz = {ixz}, Iyz = {iyz}"
        )));
    }
    let _ = mass_after;
    Ok(Realignment {
        applied: true,
        centroid: centroid.to_array(),
        rotation,
        det,
        radii,
        density,
        centroid_residual,
        ixz,
        iyz,
        i1,
    })
}

pub fn principal_frame(
    inertia: &Matrix3<f64>,
    eps_deg: f64,
) -> Result<([[f64; 3]; 3], f64), ScfError> {
    let eigen = inertia.symmetric_eigen();
    let mut order = [0usize, 1, 2];
    order.sort_by(|&i, &j| {
        eigen.eigenvalues[j]
            .partial_cmp(&eigen.eigenvalues[i])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let values = order.map(|index| eigen.eigenvalues[index]);
    let mut axes = order.map(|index| {
        let column = eigen.eigenvectors.column(index);
        DVec3::new(column[0], column[1], column[2])
    });
    for axis in &mut axes {
        if let Some(unit) = axis.try_normalize() {
            *axis = unit;
        } else {
            return Err(ScfError::model_not_closed("an inertia eigenvector is zero"));
        }
    }
    if axes[0].dot(DVec3::new(0.0, 0.0, 1.0)) < 0.0 {
        axes[0] = axes[0] * -1.0;
    }
    if axes[2].dot(DVec3::new(1.0, 0.0, 0.0)) < 0.0 {
        axes[2] = axes[2] * -1.0;
    }
    let i1 = values[0];
    let gap = eps_deg * i1.abs().max(1e-30);
    let all_degenerate = values[0] - values[2] <= gap;
    let e_z = if all_degenerate {
        DVec3::new(0.0, 0.0, 1.0)
    } else if values[0] - values[1] <= gap {
        project_or_lowest(DVec3::new(0.0, 0.0, 1.0), &[axes[0], axes[1]])
    } else {
        axes[0]
    };
    let e_z = e_z
        .try_normalize()
        .ok_or_else(|| ScfError::model_not_closed("e_z vanished"))?;
    let e_x = if all_degenerate {
        DVec3::new(1.0, 0.0, 0.0)
    } else if values[1] - values[2] <= gap {
        let in_plane = intersect_perp(e_z, axes[1], axes[2]);
        sign_toward(in_plane, DVec3::new(1.0, 0.0, 0.0))
    } else {
        sign_toward(axes[2] - e_z * axes[2].dot(e_z), DVec3::new(1.0, 0.0, 0.0))
    };
    let e_x = e_x.try_normalize().unwrap_or(axes[2]);
    let e_x = (e_x - e_z * e_x.dot(e_z))
        .try_normalize()
        .ok_or_else(|| ScfError::model_not_closed("e_x vanished"))?;
    let e_y = e_z
        .cross(e_x)
        .try_normalize()
        .ok_or_else(|| ScfError::model_not_closed("e_y vanished"))?;
    let rotation = [e_x.to_array(), e_y.to_array(), e_z.to_array()];
    Ok((rotation, i1))
}

fn project_or_lowest(target: DVec3, span: &[DVec3]) -> DVec3 {
    let projected = span
        .iter()
        .fold(DVec3::ZERO, |acc, axis| acc + *axis * target.dot(*axis));
    if projected.length() > 1e-10 {
        projected
    } else {
        span[0]
    }
}

fn intersect_perp(normal: DVec3, a: DVec3, b: DVec3) -> DVec3 {
    let da = a.dot(normal);
    let db = b.dot(normal);
    if da.abs() <= 1e-12 && db.abs() <= 1e-12 {
        return a;
    }
    let candidate = a * db - b * da;
    if candidate.length() > 1e-12 {
        candidate
    } else {
        a
    }
}

fn sign_toward(vector: DVec3, preferred: DVec3) -> DVec3 {
    if vector.dot(preferred) < 0.0 {
        vector * -1.0
    } else {
        vector
    }
}

fn rotation_det(q: [[f64; 3]; 3]) -> f64 {
    let ex = DVec3::from_array(q[0]);
    let ey = DVec3::from_array(q[1]);
    let ez = DVec3::from_array(q[2]);
    ex.dot(ey.cross(ez))
}

fn identity() -> [[f64; 3]; 3] {
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
}

fn apply_q(q: [[f64; 3]; 3], v: DVec3) -> DVec3 {
    DVec3::new(
        q[0][0] * v.x + q[0][1] * v.y + q[0][2] * v.z,
        q[1][0] * v.x + q[1][1] * v.y + q[1][2] * v.z,
        q[2][0] * v.x + q[2][1] * v.y + q[2][2] * v.z,
    )
}

fn translate_mesh(mesh: &TriMesh, centroid: DVec3) -> Result<TriMesh, ScfError> {
    let vertices = mesh.vertices.iter().map(|v| *v - centroid).collect();
    TriMesh::try_new(vertices, mesh.faces.clone())
        .map_err(|err| ScfError::model_not_closed(err.detail))
}

fn rotate_mesh(mesh: &TriMesh, rotation: [[f64; 3]; 3]) -> Result<TriMesh, ScfError> {
    let vertices = mesh
        .vertices
        .iter()
        .map(|v| apply_q(rotation, *v))
        .collect();
    TriMesh::try_new(vertices, mesh.faces.clone())
        .map_err(|err| ScfError::model_not_closed(err.detail))
}

fn star_mesh(directions: &[DVec3], radii: &[f64], faces: &[[u32; 3]]) -> Result<TriMesh, ScfError> {
    if directions.len() != radii.len() {
        return Err(ScfError::model_not_closed(
            "ray resampling changed the direction count",
        ));
    }
    let vertices = directions
        .iter()
        .zip(radii)
        .map(|(dir, radius)| *dir * *radius)
        .collect();
    TriMesh::try_new(vertices, faces.to_vec()).map_err(|err| ScfError::model_not_closed(err.detail))
}

fn rotate_density(
    density: &CauchyDensity,
    centroid: DVec3,
    rotation: [[f64; 3]; 3],
) -> Result<CauchyDensity, ScfError> {
    let kernels = density
        .kernels
        .iter()
        .map(|kernel| {
            let shifted = apply_q(rotation, kernel.center - centroid);
            CauchyKernel::try_new(shifted, kernel.weight, kernel.sigma, kernel.alpha)
                .map_err(|err| ScfError::model_not_closed(err.detail))
        })
        .collect::<Result<Vec<_>, _>>()?;
    CauchyDensity::try_new(density.bulk, kernels)
        .map_err(|err| ScfError::model_not_closed(err.detail))
}

fn density_moments(
    mesh: &TriMesh,
    density: &CauchyDensity,
) -> Result<(f64, DVec3, Matrix3<f64>), ScfError> {
    let probes = [DVec3::ZERO]
        .into_iter()
        .chain(mesh.vertices.iter().copied())
        .map(|point| density.rho(point))
        .collect::<Vec<_>>();
    let rho_max = probes.iter().copied().fold(0.0_f64, f64::max);
    let rho_min = probes.iter().copied().fold(f64::INFINITY, f64::min);
    if rho_max > 0.0 && (rho_max - rho_min) <= 1e-2 * rho_max {
        let integrals = uniform_integrals(mesh, rho_max);
        return Ok((
            integrals.mass,
            integrals.centroid,
            inertia_matrix(integrals.inertia),
        ));
    }
    let mut mass = 0.0;
    let mut moment = DVec3::ZERO;
    let mut quad = [0.0; 6];
    // Symmetric 4-point tetrahedron rule on the origin fan. Exact for quadratic
    // polynomials, so a centered cube or octahedron stays on its principal axes.
    let alpha = 0.5854101966249685;
    let beta = 0.1381966011250105;
    let bary = [
        [alpha, beta, beta, beta],
        [beta, alpha, beta, beta],
        [beta, beta, alpha, beta],
        [beta, beta, beta, alpha],
    ];
    for face in 0..mesh.faces.len() {
        let [a, b, c] = mesh.triangle(face);
        let volume = a.dot(b.cross(c)) / 6.0;
        if volume.abs() <= 1e-18 {
            continue;
        }
        for coords in bary {
            let position = a * coords[1] + b * coords[2] + c * coords[3];
            let sample = density.rho(position) * volume * 0.25;
            if !sample.is_finite() {
                return Err(ScfError::model_not_closed(
                    "realignment density sample is not finite",
                ));
            }
            mass += sample;
            moment = moment + position * sample;
            let p = position.to_array();
            quad[0] += sample * p[0] * p[0];
            quad[1] += sample * p[1] * p[1];
            quad[2] += sample * p[2] * p[2];
            quad[3] += sample * p[0] * p[1];
            quad[4] += sample * p[0] * p[2];
            quad[5] += sample * p[1] * p[2];
        }
    }
    if !(mass.is_finite() && mass > 0.0) {
        return Err(ScfError::model_not_closed(
            "realignment quadrature mass is not positive",
        ));
    }
    Ok((mass, moment * (1.0 / mass), inertia_from_products(quad)))
}

fn inertia_matrix(packed: [f64; 6]) -> Matrix3<f64> {
    Matrix3::new(
        packed[0], packed[3], packed[4], packed[3], packed[1], packed[5], packed[4], packed[5],
        packed[2],
    )
}

fn inertia_from_products(quad: [f64; 6]) -> Matrix3<f64> {
    let (xx, yy, zz, xy, xz, yz) = (quad[0], quad[1], quad[2], quad[3], quad[4], quad[5]);
    Matrix3::new(yy + zz, -xy, -xz, -xy, xx + zz, -yz, -xz, -yz, xx + yy)
}
