//! Step 8b of §10.4.4: re-alignment of the standard frame.
//!
//! The standard frame has the centroid at the origin, `e_z` along the largest principal
//! axis of inertia (the spin axis) and `e_x` along the smallest. When the body drifts out of
//! it, the **whole** model is rigidly moved: the mesh vertices, and the centres of the Cauchy
//! density kernels with the same map `x' = Q (x − x_cm)`. Radii along the fixed directions are
//! then re-sampled by hardware ray casting from the new origin, the loose layer `ρ_s` is
//! transferred barycentrically, and the residuals are re-verified.

use nalgebra::{Matrix3, SymmetricEigen, Vector3};

use crate::common::density::CauchyKernel;
use crate::common::math::DVec3;

use super::body::Body;
use super::error::ScfError;
use super::forward::{Forward, Moments, inertia_matrix};
use super::params::GeometryParams;
use super::transport::interp_vertex_field;

fn v3(v: DVec3) -> Vector3<f64> {
    Vector3::new(v.x, v.y, v.z)
}

fn d3(v: Vector3<f64>) -> DVec3 {
    DVec3::new(v.x, v.y, v.z)
}

/// Inertia about the centroid, `I_c = I_O − M (|c|² 1 − c cᵀ)`.
pub fn inertia_about_centroid(m: &Moments) -> Matrix3<f64> {
    let c = v3(m.centroid);
    inertia_matrix(m.inertia) - m.mass * (Matrix3::identity() * c.norm_squared() - c * c.transpose())
}

/// Dimensionless departure of the current frame from the standard one.
#[derive(Clone, Copy, Debug)]
pub struct FrameDeparture {
    /// `|x_cm| / R_ref`.
    pub centroid: f64,
    /// `max(|I_xz|, |I_yz|) / I_1` (and `|I_xy|` when `I_xx ≠ I_yy`).
    pub off_diagonal: f64,
    /// Violation of `I_zz ≥ I_yy ≥ I_xx` relative to `I_1`.
    pub ordering: f64,
}

pub fn departure(m: &Moments, r_ref: f64, eps_deg: f64) -> FrameDeparture {
    let ic = inertia_about_centroid(m);
    let i1 = ic[(0, 0)].max(ic[(1, 1)]).max(ic[(2, 2)]);
    let mut off = ic[(0, 2)].abs().max(ic[(1, 2)].abs());
    if (ic[(0, 0)] - ic[(1, 1)]).abs() > eps_deg * i1 {
        off = off.max(ic[(0, 1)].abs());
    }
    let ordering = (ic[(0, 0)] - ic[(2, 2)])
        .max(ic[(1, 1)] - ic[(2, 2)])
        .max(ic[(0, 0)] - ic[(1, 1)])
        .max(0.0);
    FrameDeparture {
        centroid: m.centroid.length() / r_ref,
        off_diagonal: off / i1,
        ordering: ordering / i1,
    }
}

pub fn needs_realign(m: &Moments, r_ref: f64, prm: &GeometryParams) -> bool {
    let d = departure(m, r_ref, prm.eps_deg);
    d.centroid > 0.5 * prm.eps_frame
        || d.off_diagonal > 0.5 * prm.eps_frame
        || d.ordering > prm.eps_deg
}

/// Rotation `Q` (rows `e_x', e_y', e_z'`) of the principal frame of the inertia `ic`,
/// with the sign and degeneracy conventions of §10.4.4 step 8b. The old frame is the identity.
pub fn principal_rotation(ic: &Matrix3<f64>, eps_deg: f64) -> Matrix3<f64> {
    let eig = SymmetricEigen::new(*ic);
    let mut idx = [0usize, 1, 2];
    idx.sort_by(|a, b| eig.eigenvalues[*b].partial_cmp(&eig.eigenvalues[*a]).unwrap());
    let val = idx.map(|i| eig.eigenvalues[i]);
    let vec = idx.map(|i| eig.eigenvectors.column(i).into_owned());
    let i1 = val[0];
    let tol = eps_deg * i1.abs().max(1e-300);
    let top_degenerate = (val[0] - val[1]).abs() <= tol;
    let bottom_degenerate = (val[1] - val[2]).abs() <= tol;

    let ez_old = Vector3::z();
    let ex_old = Vector3::x();
    let orthogonal_part = |v: &Vector3<f64>, to: &Vector3<f64>| -> Vector3<f64> {
        let p = v - to * v.dot(to);
        p.try_normalize(1e-12).unwrap_or_else(|| {
            // v parallel to `to`: any perpendicular
            let h = if to.x.abs() < 0.9 { Vector3::x() } else { Vector3::y() };
            (h - to * h.dot(to)).normalize()
        })
    };
    let (ez, ex) = match (top_degenerate, bottom_degenerate) {
        (true, true) => (ez_old, ex_old),
        (true, false) => {
            // The minimum axis is unique; the maximum axis lies anywhere in the plane ⟂ a3.
            let ex = vec[2];
            let ez = orthogonal_part(&ez_old, &ex);
            (ez, ex)
        }
        (false, true) => {
            // The maximum axis is unique; e_x is free in the plane ⟂ a1.
            let ez = vec[0];
            let ex = orthogonal_part(&ex_old, &ez);
            (ez, ex)
        }
        (false, false) => (vec[0], vec[2]),
    };
    // Signs: keep the new axes in the same half-spaces as the old ones.
    let ez = if ez.dot(&ez_old) < 0.0 { -ez } else { ez };
    let ex = if ex.dot(&ex_old) < 0.0 { -ex } else { ex };
    let ey = ez.cross(&ex);
    // Re-orthogonalise e_x against e_z (they are orthogonal up to round-off).
    let ex = ey.cross(&ez).normalize();
    let ey = ey.normalize();
    Matrix3::from_rows(&[ex.transpose(), ey.transpose(), ez.transpose()])
}

#[derive(Clone, Debug)]
pub struct FrameReport {
    pub applied: bool,
    pub passes: usize,
    /// Accumulated translation (old origin → new origin, in the old frame).
    pub translation: DVec3,
    /// Accumulated rotation `Q` (rows `e_x', e_y', e_z'` in the old frame).
    pub rotation: [[f64; 3]; 3],
    pub angle_deg: f64,
    pub before: FrameDeparture,
    pub after: FrameDeparture,
    pub mass_change: f64,
    pub volume_change: f64,
    pub max_radius_change: f64,
}

fn mean_radius(b: &Body) -> f64 {
    b.radius.iter().sum::<f64>() / b.radius.len() as f64
}

/// Apply `x' = Q (x − c)` to the model and re-sample the star-shaped surface.
fn apply_once(body: &Body, c: DVec3, q: &Matrix3<f64>) -> Result<Body, ScfError> {
    let p = body.positions();
    let moved: Vec<DVec3> = p.iter().map(|x| d3(q * (v3(*x) - v3(c)))).collect();
    let mesh = crate::common::geometry::TriMesh::try_new(moved, body.topo.faces.clone())
        .map_err(ScfError::from)?;
    let radius = Body::radii_from_mesh(&mesh, DVec3::ZERO, &body.topo.dirs)?;
    let kernels: Vec<CauchyKernel> = body
        .kernels
        .iter()
        .map(|k| {
            let centre = d3(q * (v3(k.center) - v3(c)));
            CauchyKernel::try_new(centre, k.weight, k.sigma, k.alpha).map_err(ScfError::from)
        })
        .collect::<Result<_, _>>()?;
    // Material transfer of ρ_s: the new vertex i sits at x' = R'_i ω̂_i, i.e. at the old point
    // Qᵀ x' + c.
    let area_before = vertex_areas(body);
    let mass_before: f64 = (0..radius.len()).map(|i| area_before[i] * body.rho_s[i]).sum();
    let mut rho_s = vec![0.0; radius.len()];
    for i in 0..radius.len() {
        let x_new = v3(body.topo.dirs[i]) * radius[i];
        let x_old = q.transpose() * x_new + v3(c);
        let dir_old = d3(x_old).try_normalize().unwrap_or(body.topo.dirs[i]);
        rho_s[i] = interp_vertex_field(&body.topo, &body.rho_s, dir_old);
    }
    let next = Body::new(body.topo.clone(), radius, body.bulk, kernels, rho_s)?;
    let area_after = vertex_areas(&next);
    let mass_after: f64 = (0..next.radius.len()).map(|i| area_after[i] * next.rho_s[i]).sum();
    let mut next = next;
    if mass_after > 0.0 && mass_before > 0.0 {
        let s = mass_before / mass_after;
        next.rho_s.iter_mut().for_each(|v| *v *= s);
    }
    Ok(next)
}

/// Dual area of every vertex (`Σ_f A_f / 3`).
pub fn vertex_areas(body: &Body) -> Vec<f64> {
    let geo = body.face_geometry();
    let mut a = vec![0.0; body.topo.n_vertices()];
    for (f, t) in body.topo.faces.iter().enumerate() {
        for v in t {
            a[*v as usize] += geo[f].2 / 3.0;
        }
    }
    a
}

/// Re-align `body` (centroid at the origin, `I_zz` maximal, `I_xx` minimal). `force` skips the
/// trigger test for the first pass (initialisation). At most three passes: the ray re-sampling perturbs the moments at the discretisation level, so the frame
/// is re-verified after every pass and the loop stops as soon as the thresholds hold.
pub fn realign(
    body: &Body,
    forward: Option<Forward>,
    g: f64,
    prm: &GeometryParams,
    force: bool,
) -> Result<(Body, FrameReport, Forward), ScfError> {
    let mut current = body.clone();
    let mut fwd = match forward {
        Some(f) => f,
        None => Forward::bind(&current, g)?,
    };
    let r_ref = mean_radius(&current);
    let before = departure(&fwd.moments, r_ref, prm.eps_deg);
    let m0 = fwd.moments.mass;
    let v0 = fwd.moments.volume;
    let r0 = current.radius.clone();
    let mut q_acc = Matrix3::identity();
    let mut t_acc = Vector3::zeros();
    let mut passes = 0;
    while passes < 3 && ((force && passes == 0) || needs_realign(&fwd.moments, r_ref, prm)) {
        let c = fwd.moments.centroid;
        let ic = inertia_about_centroid(&fwd.moments);
        let q = principal_rotation(&ic, prm.eps_deg);
        current = apply_once(&current, c, &q)?;
        // x_orig = q_accᵀ x_cur + t_acc  ⇒  compose.
        t_acc += q_acc.transpose() * v3(c);
        q_acc = q * q_acc;
        fwd = Forward::bind(&current, g)?;
        passes += 1;
    }
    let after = departure(&fwd.moments, r_ref, prm.eps_deg);
    let trace = q_acc.trace();
    let angle = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0).acos().to_degrees();
    let max_change = current
        .radius
        .iter()
        .zip(&r0)
        .map(|(a, b)| ((a - b) / b).abs())
        .fold(0.0, f64::max);
    let rotation = [
        [q_acc[(0, 0)], q_acc[(0, 1)], q_acc[(0, 2)]],
        [q_acc[(1, 0)], q_acc[(1, 1)], q_acc[(1, 2)]],
        [q_acc[(2, 0)], q_acc[(2, 1)], q_acc[(2, 2)]],
    ];
    let report = FrameReport {
        applied: passes > 0,
        passes,
        translation: d3(t_acc),
        rotation,
        angle_deg: angle,
        before,
        after,
        mass_change: (fwd.moments.mass - m0) / m0,
        volume_change: (fwd.moments.volume - v0) / v0,
        max_radius_change: max_change,
    };
    Ok((current, report, fwd))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use super::super::body::Topology;

    fn prm() -> GeometryParams {
        GeometryParams {
            c0: 0.2,
            eps_n: 0.01,
            eps_mass: 1e-6,
            trust0: 0.01,
            trust_min: 1e-5,
            trust_shrink: 0.5,
            varrho: 1e3,
            sh_fit_tol: 1e-2,
            eps_frame: 1e-3,
            eps_deg: 1e-6,
        }
    }

    #[test]
    fn tilted_offset_ellipsoid_is_brought_to_the_standard_frame() {
        let topo = Arc::new(Topology::icosphere(4));
        let axes = [0.6, 0.8, 1.0];
        // Ellipsoid whose long axis is e_x and short axis e_z, i.e. a wrong frame.
        let radius = Body::ellipsoid(topo.clone(), axes);
        let nv = topo.n_vertices();
        let body = Body::new(topo.clone(), radius, 1000.0, vec![], vec![1.0; nv]).unwrap();
        // The frame is wrong: I_zz is the smallest moment.
        let g = 1.0e-9;
        let (out, rep, _) = realign(&body, None, g, &prm(), false).unwrap();
        assert!(rep.applied);
        assert!(rep.after.ordering < 1e-3, "{:?}", rep.after);
        assert!(rep.after.off_diagonal < 1e-3, "{:?}", rep.after);
        assert!(rep.after.centroid < 1e-3, "{:?}", rep.after);
        // The new long axis (max R) is along e_x? No: e_z is the max-inertia (short body) axis,
        // so the shortest radius must now lie on z and the longest on x or y.
        let zi = (0..nv).max_by(|a, b| topo.dirs[*a].z.partial_cmp(&topo.dirs[*b].z).unwrap()).unwrap();
        assert!((out.radius[zi] - 0.6).abs() < 0.03, "{}", out.radius[zi]);
        assert!(rep.mass_change.abs() < 5e-3);
    }

    #[test]
    fn kernel_centres_follow_the_mesh() {
        let topo = Arc::new(Topology::icosphere(4));
        let nv = topo.n_vertices();
        let radius = Body::ellipsoid(topo.clone(), [1.0, 0.8, 0.6]);
        let k = CauchyKernel::try_new(DVec3::new(0.3, 0.1, 0.2), 800.0, 0.3, 2.0).unwrap();
        let body = Body::new(topo, radius, 1000.0, vec![k], vec![0.0; nv]).unwrap();
        let g = 1.0e-9;
        let (out, rep, _) = realign(&body, None, g, &prm(), false).unwrap();
        assert!(rep.applied);
        assert!(rep.after.centroid < 1e-3, "{:?}", rep.after);
        assert!(rep.after.off_diagonal < 1e-3, "{:?}", rep.after);
        // The kernel moved with the rigid map x' = Q (x - c) (a rigid map preserves distances
        // to the origin only up to the translation).
        let c = rep.translation;
        let q = rep.rotation;
        let old = DVec3::new(0.3, 0.1, 0.2) - c;
        let expect = DVec3::new(
            q[0][0] * old.x + q[0][1] * old.y + q[0][2] * old.z,
            q[1][0] * old.x + q[1][1] * old.y + q[1][2] * old.z,
            q[2][0] * old.x + q[2][1] * old.y + q[2][2] * old.z,
        );
        let got = out.kernels[0].center;
        assert!((got - expect).length() < 1e-9, "{got:?} vs {expect:?}");
    }
}
