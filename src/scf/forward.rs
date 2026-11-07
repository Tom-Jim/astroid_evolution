//! Step 1 of §10.3.4: the Tier 1 forward model on the current polyhedron.
//!
//! * surface: `SurfaceFace` role at the face centres (gradient and finite-part Hessian);
//!   a vertex or edge Hessian is never requested (MS §10.2);
//! * interior: `Field` role at the Stokes quadrature points (gradient and Hessian);
//! * moments: mass, centroid and inertia of `ρ_K` from `density_integrals`.

use crate::common::math::{DMat3, DVec3, Sym6, sym_mat};
use crate::rtfp::api::{FieldMask, PointRole, PointSet};
use crate::rtfp::{RtfpOptions, RtfpSolver};

use super::body::Body;
use crate::common::error::SimError;

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

/// The exogenous loads of one step: spin about `z` and the tide of a primary on
/// the `+x` axis with strength `GM/a³` (zero without one).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Loading {
    pub omega: f64,
    pub tide: f64,
}

/// `g_eff = ∇U + ω² x_⊥ + g_ext`, `g_ext = ∇[(GM/a³)(x² − (y² + z²)/2)]`.
pub fn effective_gravity(grad_u: DVec3, x: DVec3, l: Loading) -> DVec3 {
    let o2 = l.omega * l.omega;
    grad_u + DVec3::new(o2 * x.x, o2 * x.y, 0.0) + DVec3::new(2.0 * x.x, -x.y, -x.z) * l.tide
}

/// `H_W = Hess U + ω² diag(1, 1, 0) + (GM/a³) diag(2, −1, −1)`.
pub fn hessian_w(h: Sym6, l: Loading) -> Sym6 {
    let o2 = l.omega * l.omega;
    [
        h[0] + o2 + 2.0 * l.tide,
        h[1] + o2 - l.tide,
        h[2] - l.tide,
        h[3],
        h[4],
        h[5],
    ]
}

/// `dev H` of a symmetric tensor.
pub fn deviator(h: Sym6) -> Sym6 {
    let t = (h[0] + h[1] + h[2]) / 3.0;
    [h[0] - t, h[1] - t, h[2] - t, h[3], h[4], h[5]]
}

impl Forward {
    pub fn bind(body: &Body, g: f64) -> Result<Self, SimError> {
        let options = RtfpOptions {
            g,
            ..RtfpOptions::default()
        };
        let solver = {
            let _span = super::timing::span("rtfp_bind");
            RtfpSolver::bind(options, body.mesh()?, body.density()?)?
        };
        let v = {
            let _span = super::timing::span("rtfp_moments");
            solver.density_integrals()?
        };
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

    /// Replace the density on the same polyhedron: new moments, cached fields dropped.
    pub fn set_density(
        &mut self,
        density: crate::common::density::CauchyDensity,
    ) -> Result<(), SimError> {
        self.solver.set_density(density);
        let v = self.solver.density_integrals()?;
        self.moments = Moments {
            volume: v.volume,
            mass: v.mass,
            centroid: v.centroid,
            inertia: v.inertia,
        };
        self.face = PointFields::default();
        Ok(())
    }

    /// Gradient and Hessian at the face centres of the polyhedron.
    pub fn eval_faces(&mut self, body: &Body) -> Result<(), SimError> {
        let centres: Vec<DVec3> = body.face_geometry().iter().map(|f| f.0).collect();
        // Same bound mesh, density and G, and the same sample points: the integral
        // is already stored. A vertex move shifts the centres and misses this.
        if self.face.points == centres
            && self.face.grad_u.len() == centres.len()
            && self.face.hess_u.len() == centres.len()
        {
            return Ok(());
        }
        // Gradient and finite-part Hessian only. The potential is not an input of the
        // Stokes load or the slope row, so it is not requested.
        let batch = {
            let _span = super::timing::span("rtfp_faces");
            self.solver.eval(
                &PointSet::from_points(&centres),
                PointRole::SurfaceFace,
                FieldMask {
                    potential: false,
                    gradient: true,
                    hessian: true,
                },
            )?
        };
        self.face = PointFields {
            points: centres,
            grad_u: batch.gradient,
            hess_u: batch.hessian,
        };
        Ok(())
    }

    /// The GPU acceleration structure of the bound polyhedron (ray queries of §8.1).
    pub fn body_mesh(&self) -> &crate::rtfp::mesh::BodyMesh {
        &self.solver.body
    }

    /// Gradient and finite-part Hessian at points on face interiors of `Γ`.
    pub fn eval_surface(&self, points: &[DVec3]) -> Result<PointFields, SimError> {
        let batch = {
            let _span = super::timing::span("rtfp_surface");
            self.solver.eval(
                &PointSet::from_points(points),
                PointRole::SurfaceFace,
                FieldMask {
                    potential: false,
                    gradient: true,
                    hessian: true,
                },
            )?
        };
        Ok(PointFields {
            points: points.to_vec(),
            grad_u: batch.gradient,
            hess_u: batch.hessian,
        })
    }

    /// Gradient and Hessian at interior points (classical convergence).
    pub fn eval_field(&self, points: &[DVec3], hessian: bool) -> Result<PointFields, SimError> {
        let mask = if hessian {
            FieldMask {
                potential: false,
                gradient: true,
                hessian: true,
            }
        } else {
            FieldMask::GRAD
        };
        let batch = {
            let _span = super::timing::span("rtfp_field");
            self.solver
                .eval(&PointSet::from_points(points), PointRole::Field, mask)?
        };
        Ok(PointFields {
            points: points.to_vec(),
            grad_u: batch.gradient,
            hess_u: batch.hessian,
        })
    }

    /// Gravity and its Hessian at points outside the body: the exterior route of the
    /// runout flights (§10.0.2).
    pub fn eval_exterior(&self, points: &[DVec3]) -> Result<Vec<(DVec3, DMat3)>, SimError> {
        let f = self.eval_field(points, true)?;
        Ok(f.grad_u.into_iter().zip(f.hess_u).map(|(g, h)| (g, sym_mat(h))).collect())
    }
}

/// Inertia as a nalgebra matrix, for the eigen-decomposition of the frame.
pub fn inertia_matrix(i: Sym6) -> nalgebra::Matrix3<f64> {
    nalgebra::Matrix3::new(i[0], i[3], i[4], i[3], i[1], i[5], i[4], i[5], i[2])
}
