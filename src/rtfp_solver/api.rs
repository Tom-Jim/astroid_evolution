//! Tier 1 request and result types. Coordinates are SI f64, stored as SoA.

use crate::common::math::{DVec3, Sym6};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PointRole {
    /// Node samples. Hessian is refused: MS §10.3 forbids a vertex Hessian in `T_tide`.
    SurfaceVertex,
    /// Face-center samples. The finite-part Hessian is independent of `ℓ0` when `C = 0`.
    SurfaceFace,
    /// Interior Stokes nodes. Only the gradient is classical and required.
    VolumeQuad,
    /// Interior classification nodes. Gradient and classical Hessian.
    ClassifyGrid,
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

pub struct FieldRequest<'a> {
    pub points: &'a PointSet,
    pub role: PointRole,
    pub want: FieldMask,
}

#[derive(Clone, Debug)]
pub struct FieldBatch {
    pub potential: Vec<f64>,
    pub gradient: Vec<DVec3>,
    pub hessian: Vec<Sym6>,
    /// Solid-angle fraction used by the contact term. Interior is 1, exterior is 0.
    pub theta: Vec<f64>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VolumeIntegrals {
    pub volume: f64,
    pub mass: f64,
    pub centroid: DVec3,
    /// Inertia about the origin, `(xx, yy, zz, xy, xz, yz)`, with `I_ij = ∫ (r² δ_ij − x_i x_j) ρ dV`.
    pub inertia: Sym6,
}
