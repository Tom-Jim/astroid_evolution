//! Tier 1 request and result types. Coordinates are SI f64, stored as SoA.

use crate::common::math::{DVec3, Sym6};

/// What kind of point is evaluated; it fixes the contact term of the Hessian.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PointRole {
    /// A mesh node. The Hessian is refused: MS §10.2 forbids a vertex or edge Hessian.
    SurfaceVertex,
    /// A face interior point. The one-sided finite-part Hessian has `θ = 1/2`.
    SurfaceFace,
    /// A point strictly inside or strictly outside the body, told apart by the hardware ray parity.
    Field,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FieldMask {
    pub potential: bool,
    pub gradient: bool,
    pub hessian: bool,
}

impl FieldMask {
    pub const ALL: Self = Self {
        potential: true,
        gradient: true,
        hessian: true,
    };

    pub const GRAD: Self = Self {
        potential: false,
        gradient: true,
        hessian: false,
    };
}

#[derive(Clone, Debug)]
pub struct PointSet {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub z: Vec<f64>,
}

impl PointSet {
    pub fn from_points(points: &[DVec3]) -> Self {
        Self {
            x: points.iter().map(|p| p.x).collect(),
            y: points.iter().map(|p| p.y).collect(),
            z: points.iter().map(|p| p.z).collect(),
        }
    }

    pub fn len(&self) -> usize {
        self.x.len()
    }

    pub fn is_empty(&self) -> bool {
        self.x.is_empty()
    }

    pub fn point(&self, i: usize) -> DVec3 {
        DVec3::new(self.x[i], self.y[i], self.z[i])
    }
}

#[derive(Clone, Debug)]
pub struct FieldBatch {
    pub potential: Vec<f64>,
    pub gradient: Vec<DVec3>,
    pub hessian: Vec<Sym6>,
    /// Fraction used by the contact term: 1 inside, 0 outside, 1/2 on a face.
    pub theta: Vec<f64>,
    /// Spherical measure of the tangent cone: `4π` inside, `2π` on a face, `0` outside and at a vertex.
    pub solid_angle: Vec<f64>,
    /// Rays whose orientation-rule chord construction saw a same-sign anomaly
    /// and fell back to the parity rule (MS §3.6 step 4).
    pub anomalies: u32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VolumeIntegrals {
    pub volume: f64,
    pub mass: f64,
    pub centroid: DVec3,
    /// Inertia about the origin, `(xx, yy, zz, xy, xz, yz)`, with `I_ij = ∫ (r² δ_ij − x_i x_j) ρ dV`.
    pub inertia: Sym6,
    /// Rays whose orientation-rule chord construction saw an anomaly and fell
    /// back to the parity rule (MS §3.6 step 4).
    pub anomalies: u32,
}
