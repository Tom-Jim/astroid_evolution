//! Step 1 of §10.4.4: the Tier 1 forward model on the current polyhedron.
//!
//! * surface: `SurfaceFace` role at the face centres (gradient and finite-part Hessian);
//!   a vertex or edge Hessian is never requested (MS §10.3);
//! * interior: `Field` role at the Stokes quadrature points (gradient and Hessian);
//! * moments: mass, centroid and inertia of `ρ_K` from `density_integrals`.

use crate::common::math::{DVec3, Sym6};
use crate::rtfp_solver::api::{FieldMask, PointRole, PointSet};
use crate::rtfp_solver::{RtfpOptions, RtfpSolver};

use super::body::Body;
use super::error::ScfError;

/// Mass, centroid and origin inertia of the body.
#[derive(Clone, Copy, Debug)]
pub struct Moments {
    pub volume: f64,
    pub mass: f64,
    pub centroid: DVec3,
    /// `(xx, yy, zz, xy, xz, yz)` of `I_ij = ∫ ρ (r² δ_ij − x_i x_j) dV` about the origin.
    pub inertia: Sym6,
}

/// Raw Tier 1 fields at a point set (no rotation terms).
#[derive(Clone, Debug, Default)]
pub struct PointFields {
    pub points: Vec<DVec3>,
    pub grad_u: Vec<DVec3>,
    pub hess_u: Vec<Sym6>,
}

pub struct Forward {
    pub solver: RtfpSolver,
    pub moments: Moments,
    pub face: PointFields,
}

/// `g_eff = ∇U + ω² x_⊥`.
pub fn effective_gravity(grad_u: DVec3, x: DVec3, omega: f64) -> DVec3 {
    let o2 = omega * omega;
    DVec3::new(grad_u.x + o2 * x.x, grad_u.y + o2 * x.y, grad_u.z)
}

/// `H_W = Hess U + ω² diag(1, 1, 0)`.
pub fn hessian_w(h: Sym6, omega: f64) -> Sym6 {
    let o2 = omega * omega;
    [h[0] + o2, h[1] + o2, h[2], h[3], h[4], h[5]]
}

/// `dev H` of a symmetric tensor.
pub fn deviator(h: Sym6) -> Sym6 {
    let t = (h[0] + h[1] + h[2]) / 3.0;
    [h[0] - t, h[1] - t, h[2] - t, h[3], h[4], h[5]]
}

impl Forward {
    pub fn bind(body: &Body, g: f64) -> Result<Self, ScfError> {
        let options = RtfpOptions {
            g,
            ..RtfpOptions::default()
        };
        let solver = RtfpSolver::bind(options, body.mesh()?, body.density()?)?;
        let v = solver.density_integrals()?;
        let moments = Moments {
            volume: v.volume,
            mass: v.mass,
            centroid: v.centroid,
            inertia: v.inertia,
        };
        Ok(Self {
            solver,
            moments,
            face: PointFields::default(),
        })
    }

    /// Gradient and Hessian at the face centres of the polyhedron.
    pub fn eval_faces(&mut self, body: &Body) -> Result<(), ScfError> {
        let centres: Vec<DVec3> = body.face_geometry().iter().map(|f| f.0).collect();
        // Gradient and finite-part Hessian only. The potential is not an input of the
        // Stokes load or the slope row, so it is not requested.
        let batch = self.solver.eval(
            &PointSet::from_points(&centres),
            PointRole::SurfaceFace,
            FieldMask {
                potential: false,
                gradient: true,
                hessian: true,
            },
        )?;
        self.face = PointFields {
            points: centres,
            grad_u: batch.gradient,
            hess_u: batch.hessian,
        };
        Ok(())
    }

    /// Gradient and Hessian at interior points (classical convergence).
    pub fn eval_field(&self, points: &[DVec3], hessian: bool) -> Result<PointFields, ScfError> {
        let mask = if hessian {
            FieldMask {
                potential: false,
                gradient: true,
                hessian: true,
            }
        } else {
            FieldMask::GRAD
        };
        let batch = self
            .solver
            .eval(&PointSet::from_points(points), PointRole::Field, mask)?;
        Ok(PointFields {
            points: points.to_vec(),
            grad_u: batch.gradient,
            hess_u: batch.hessian,
        })
    }
}

/// Inertia as a nalgebra matrix, for the eigen-decomposition of the frame.
pub fn inertia_matrix(i: Sym6) -> nalgebra::Matrix3<f64> {
    nalgebra::Matrix3::new(i[0], i[3], i[4], i[3], i[1], i[5], i[4], i[5], i[2])
}
