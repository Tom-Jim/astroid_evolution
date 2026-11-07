//! Step 8b of §10.3.4: re-alignment of the standard frame.
//!
//! The standard frame has the centroid at the origin, `e_z` along the largest principal
//! axis of inertia (the spin axis) and `e_x` along the smallest. When the body drifts out of
//! it, the **whole** model is rigidly moved: the mesh vertices, and the centres of the Cauchy
//! density kernels with the same map `x' = Q (x − x_cm)`. Nothing is re-sampled and no
//! surface field is remapped; only the residuals are re-verified against the new frame.

use nalgebra::{Matrix3, SymmetricEigen, Vector3};

use crate::common::density::CauchyKernel;
use crate::common::math::DVec3;

use super::body::Body;
use super::forward::{Forward, Moments, inertia_matrix};
use super::params::GeometryParams;
use crate::common::error::SimError;

fn v3(v: DVec3) -> Vector3<f64> {
    Vector3::new(v.x, v.y, v.z)
}

fn d3(v: Vector3<f64>) -> DVec3 {
    DVec3::new(v.x, v.y, v.z)
}

/// Inertia about the centroid, `I_c = I_O − M (|c|² 1 − c cᵀ)`.
pub fn inertia_about_centroid(m: &Moments) -> Matrix3<f64> {
    let c = v3(m.centroid);
    inertia_matrix(m.inertia)
        - m.mass * (Matrix3::identity() * c.norm_squared() - c * c.transpose())
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
/// with the sign and degeneracy conventions of §10.3.4 step 8b. The old frame is the identity.
pub fn principal_rotation(ic: &Matrix3<f64>, eps_deg: f64) -> Matrix3<f64> {
    let eig = SymmetricEigen::new(*ic);
    let mut idx = [0usize, 1, 2];
    idx.sort_by(|a, b| {
        eig.eigenvalues[*b]
            .partial_cmp(&eig.eigenvalues[*a])
            .unwrap()
    });
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
            let h = if to.x.abs() < 0.9 {
                Vector3::x()
            } else {
                Vector3::y()
            };
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
}

/// The rigid motion `x' = Q (x − c)` of the vertices and the cores (G8): `ρ_s` stays on its
/// vertices, and a core keeps its shape, `S' = Q S Qᵀ`. Nothing is re-sampled.
fn rigid(body: &Body, q: &Matrix3<f64>, c: DVec3, t: DVec3) -> Result<Body, SimError> {
    let qm = crate::common::math::DMat3::from_cols_array_2d(&[
        [q[(0, 0)], q[(1, 0)], q[(2, 0)]],
        [q[(0, 1)], q[(1, 1)], q[(2, 1)]],
        [q[(0, 2)], q[(1, 2)], q[(2, 2)]],
    ]);
    let map = |x: DVec3| qm * (x - c) + t;
    let kernels: Vec<CauchyKernel> = body
        .kernels
        .iter()
        .map(|k| {
            let metric = crate::common::math::mat_sym(qm * k.metric_mat() * qm.transpose());
            CauchyKernel::try_anisotropic(map(k.center), k.weight, metric, k.alpha)
        })
        .collect::<Result<_, _>>()?;
    Body::new(
        body.topo.clone(),
        body.positions.iter().map(|x| map(*x)).collect(),
        body.bulk,
        kernels,
        body.rho_s.clone(),
    )
}

/// The inverse of the accumulated frame map: a body expressed in a realigned frame
/// `x = R (x₀ − T)` is returned in the frame of `x₀` (`x₀ = Rᵀ x + T`).
pub fn undo_frame(
    body: &Body,
    rotation: [[f64; 3]; 3],
    translation: DVec3,
) -> Result<Body, SimError> {
    let r = rotation;
    let q = Matrix3::new(
        r[0][0], r[0][1], r[0][2], r[1][0], r[1][1], r[1][2], r[2][0], r[2][1], r[2][2],
    );
    rigid(body, &q.transpose(), DVec3::ZERO, translation)
}

/// Re-align `body` (centroid at the origin, `I_zz` maximal, `I_xx` minimal). `force` skips the
/// trigger test for the first pass (initialisation). The motion is rigid, so one pass
/// normally suffices; the frame is re-verified after every pass, up to three.
///
/// After three passes, the residual departure is verified. If it still exceeds
/// `ε_frame · R_ref` (centroid) or `ε_frame · I₁` (off-diagonal), the function returns
/// `RigidRankDeficient` per §10.3.4 step 8b (vii) and §A.3.
pub fn realign(
    body: &Body,
    forward: Option<Forward>,
    g: f64,
    prm: &GeometryParams,
    force: bool,
) -> Result<(Body, FrameReport, Forward), SimError> {
    let mut clock = super::timing::span("frame_realign");
    let mut current = body.clone();
    let mut fwd = match forward {
        Some(f) => f,
        None => {
            clock.pause();
            let bound = Forward::bind(&current, g)?;
            clock.resume();
            bound
        }
    };
    let r_ref = current.r_mean();
    let before = departure(&fwd.moments, r_ref, prm.eps_deg);
    let m0 = fwd.moments.mass;
    let v0 = fwd.moments.volume;
    let mut q_acc = Matrix3::identity();
    let mut t_acc = Vector3::zeros();
    let mut passes = 0;
    while passes < 3 && ((force && passes == 0) || needs_realign(&fwd.moments, r_ref, prm)) {
        let c = fwd.moments.centroid;
        let ic = inertia_about_centroid(&fwd.moments);
        let q = principal_rotation(&ic, prm.eps_deg);
        current = rigid(&current, &q, c, DVec3::ZERO)?;
        // x_orig = q_accᵀ x_cur + t_acc  ⇒  compose.
        t_acc += q_acc.transpose() * v3(c);
        q_acc = q * q_acc;
        clock.pause();
        fwd = Forward::bind(&current, g)?;
        clock.resume();
        passes += 1;
    }
    let after = departure(&fwd.moments, r_ref, prm.eps_deg);
    // §10.3.4 step 8b (vii): re-verify after at most three passes.
    // If the frame is still out of tolerance, the rigid-body condition is incompatible.
    if passes > 0 && (after.centroid > prm.eps_frame || after.off_diagonal > prm.eps_frame) {
        return Err(SimError::rigid_rank(format!(
            "frame re-alignment did not converge after {passes} passes: \
             centroid offset = {:.3e} (tol {}), off-diagonal = {:.3e} (tol {})",
            after.centroid, prm.eps_frame, after.off_diagonal, prm.eps_frame
        )));
    }
    let trace = q_acc.trace();
    let angle = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0).acos().to_degrees();
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
    };
    Ok((current, report, fwd))
}
