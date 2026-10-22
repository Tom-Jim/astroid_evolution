//! Inputs of the chain: the star surface, the modal velocity, and per-node fields.

pub use crate::common::math::fibonacci_directions;
use std::f64::consts::PI;

use super::basis::{K, L_MAX_LIMIT, lm_index};
use crate::common::error::SimError;
use crate::common::math::{DVec3, Sym6};

/// `R(omega) = sum r_lm Y_lm(omega)`.
#[derive(Clone, Debug)]
pub struct StarSurface {
    pub l_max: usize,
    pub coeff: Vec<f64>,
}

impl StarSurface {
    pub fn sphere(radius: f64) -> Self {
        Self {
            l_max: 0,
            coeff: vec![radius * (4.0 * PI).sqrt()],
        }
    }

    /// `entries` are `(l, m, r_lm)`; repeated entries add.
    pub fn from_entries(entries: &[(usize, i32, f64)]) -> Result<Self, SimError> {
        let l_max = entries.iter().map(|e| e.0).max().unwrap_or(0);
        if l_max > L_MAX_LIMIT {
            return Err(SimError::invalid_input(format!(
                "surface degree {l_max} exceeds {L_MAX_LIMIT}"
            )));
        }
        let mut coeff = vec![0.0; (l_max + 1) * (l_max + 1)];
        for &(l, m, v) in entries {
            if m.unsigned_abs() as usize > l || !v.is_finite() {
                return Err(SimError::invalid_input(format!(
                    "bad surface mode ({l}, {m})"
                )));
            }
            coeff[lm_index(l, m)] += v;
        }
        if coeff.first().copied().unwrap_or(0.0) <= 0.0 {
            return Err(SimError::geometry("the (0,0) coefficient must be positive"));
        }
        Ok(Self { l_max, coeff })
    }
}

/// Poloidal `q_{lkm}` and toroidal `s_{lkm}` coefficients (§10).
#[derive(Clone, Debug)]
pub struct VelocityField {
    pub l_max: usize,
    pub q: Vec<[f64; K]>,
    pub s: Vec<[f64; K]>,
}

impl VelocityField {
    pub fn zero() -> Self {
        Self {
            l_max: 0,
            q: vec![[0.0; K]],
            s: vec![[0.0; K]],
        }
    }

    /// `poloidal` and `toroidal` entries are `(l, m, k, value)`.
    pub fn from_entries(
        poloidal: &[(usize, i32, usize, f64)],
        toroidal: &[(usize, i32, usize, f64)],
    ) -> Result<Self, SimError> {
        let l_max = poloidal
            .iter()
            .chain(toroidal)
            .map(|e| e.0)
            .max()
            .unwrap_or(0);
        if l_max > L_MAX_LIMIT {
            return Err(SimError::invalid_input(format!(
                "velocity degree {l_max} exceeds {L_MAX_LIMIT}"
            )));
        }
        let n = (l_max + 1) * (l_max + 1);
        let mut out = Self {
            l_max,
            q: vec![[0.0; K]; n],
            s: vec![[0.0; K]; n],
        };
        for (set, target) in [(poloidal, 0), (toroidal, 1)] {
            for &(l, m, k, v) in set {
                if l < 1 || m.unsigned_abs() as usize > l || k >= K || !v.is_finite() {
                    return Err(SimError::invalid_input(format!(
                        "bad velocity mode (l={l}, m={m}, k={k})"
                    )));
                }
                let idx = lm_index(l, m);
                if target == 0 {
                    out.q[idx][k] += v;
                } else {
                    out.s[idx][k] += v;
                }
            }
        }
        Ok(out)
    }
}

/// Everything the chain reads at one surface node.
#[derive(Clone, Debug)]
pub struct NodeBatch {
    /// Unit directions of the nodes on the sphere.
    pub dirs: Vec<DVec3>,
    /// Effective gravity `g_eff` at the node.
    pub gravity: Vec<DVec3>,
    /// `H_W = Hess U + omega^2 diag(1,1,0)` at the face centre (§10.3), `(xx,yy,zz,xy,xz,yz)`.
    pub hessian_w: Vec<Sym6>,
    /// Pressure at the node (`p_ext + n.tau.n`).
    pub pressure: Vec<f64>,
}

impl NodeBatch {
    pub fn len(&self) -> usize {
        self.dirs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.dirs.is_empty()
    }

    pub fn validate(&self) -> Result<(), SimError> {
        let n = self.dirs.len();
        if self.gravity.len() != n || self.hessian_w.len() != n || self.pressure.len() != n {
            return Err(SimError::invalid_input("node arrays differ in length"));
        }
        for i in 0..n {
            let d = self.dirs[i];
            if !(d.is_finite() && (d.length() - 1.0).abs() < 1e-9) {
                return Err(SimError::invalid_input(format!(
                    "direction {i} is not a unit vector"
                )));
            }
            let g = self.gravity[i];
            if !(g.is_finite() && g.length() > 0.0) {
                return Err(SimError::invalid_input(format!(
                    "gravity {i} is zero or not finite"
                )));
            }
            if !self.hessian_w[i].iter().all(|v| v.is_finite()) || !self.pressure[i].is_finite() {
                return Err(SimError::invalid_input(format!(
                    "node {i} has non-finite data"
                )));
            }
        }
        Ok(())
    }
}

/// Orthonormal tangent frame `(t1, t2)` with `t1 x t2 = w`.
pub fn local_frame(w: DVec3) -> (DVec3, DVec3) {
    let axis = if w.x.abs() <= w.y.abs() && w.x.abs() <= w.z.abs() {
        DVec3::new(1.0, 0.0, 0.0)
    } else if w.y.abs() <= w.z.abs() {
        DVec3::new(0.0, 1.0, 0.0)
    } else {
        DVec3::new(0.0, 0.0, 1.0)
    };
    let t1 = (axis - w * w.dot(axis))
        .try_normalize()
        .expect("axis is never parallel to w");
    let t2 = w.cross(t1);
    (t1, t2)
}

/// Direction of the chart point `(u, v)` of the graph chart at `w`.
pub fn chart_direction(w: DVec3, u: f64, v: f64) -> DVec3 {
    let (t1, t2) = local_frame(w);
    let s = (1.0 - u * u - v * v).sqrt();
    t1 * u + t2 * v + w * s
}

/// Closed-form environment used by the CLI and the acceptance checks:
/// `g_eff = -k x + omega^2 x_perp`, `H_W = -k I + omega^2 diag(1,1,0)` (uniform
/// sphere interior plus the rotating frame), constant pressure `p_ext`.
#[derive(Clone, Copy, Debug)]
pub struct Environment {
    pub gravity_k: f64,
    pub omega: f64,
    pub p_ext: f64,
}

impl Environment {
    pub fn nodes(&self, surface: &StarSurface, dirs: Vec<DVec3>) -> NodeBatch {
        let o2 = self.omega * self.omega;
        let k = self.gravity_k;
        let mut gravity = Vec::with_capacity(dirs.len());
        for d in &dirs {
            let r = super::reference::radius(surface, *d);
            let x = *d * r;
            gravity.push(DVec3::new(
                -k * x.x + o2 * x.x,
                -k * x.y + o2 * x.y,
                -k * x.z,
            ));
        }
        let h: Sym6 = [-k + o2, -k + o2, -k, 0.0, 0.0, 0.0];
        NodeBatch {
            hessian_w: vec![h; dirs.len()],
            pressure: vec![self.p_ext; dirs.len()],
            gravity,
            dirs,
        }
    }
}
