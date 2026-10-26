//! The pressure problem of §8.7 by Nitsche's method (G3), the free-surface datum, and the
//! strong-form residual. Shared by both velocity solvers.

use std::f64::consts::PI;

use nalgebra::DVector;

use crate::common::error::SimError;
use crate::common::linalg::{self, Entry};
use crate::common::math::DVec3;
use crate::rheo::Material;

use super::super::basis::{VelRows, contract_pressure};
use super::super::params::StokesParams;
use super::{Fields, Loads, Operators, d_apply, d6_dot, eta_eval};

/// `f = ρ g_eff + ∇·T_tide + η Δv + 2 D ∇η` at one point, with `∇η` by the chain rule
/// through `γ̇`, `p̃` and `c` (§8.7). `p` is the unclipped pressure there.
#[allow(clippy::too_many_arguments)]
fn body_force(
    m: &Material,
    eps: f64,
    (c, gc): (f64, DVec3),
    (d6, lap, gd): (&[f64], &[f64], &[f64]),
    (ptil, p, gp): (f64, f64, DVec3),
    rho_g: DVec3,
    tide_div: DVec3,
) -> DVec3 {
    let shear = (2.0 * d6_dot(d6, d6)).sqrt();
    let ev = eta_eval(m, c, shear, ptil, eps);
    let mut grad_eta = DVec3::ZERO;
    if shear > 1e-300 {
        let gs: [f64; 3] =
            std::array::from_fn(|k| 2.0 * d6_dot(d6, &gd[6 * k..6 * k + 6]) / shear);
        grad_eta = DVec3::from_array(gs) * ev.d_shear;
    }
    if ptil > m.spec.p_min && p > m.spec.p_min {
        grad_eta += gp * ev.d_p;
    }
    // The cohesion profile varies in space (§4.3a): ∂η/∂c ∇c.
    grad_eta += gc * ev.d_c;
    rho_g + tide_div + DVec3::from_slice(lap) * ev.eta + d_apply(d6, grad_eta) * 2.0
}

/// `v`, `D`, `Δv`, `∂D` of the coefficient vector `u` at every point of `rows`.
fn strains<T: Entry>(rows: &VelRows<T>, u: &DVector<f64>) -> [DVector<f64>; 3] {
    [linalg::mul(&rows.d, u), linalg::mul(&rows.lap, u), linalg::mul(&rows.gd, u)]
}

/// Pressure fields carried between updates.
pub(super) struct Pressure {
    /// `p`, `p̃ = max(p, p_min)` and `∇p` at the volume points.
    pub p_vals: Vec<f64>,
    pub ptil: Vec<f64>,
    pub gradp: Vec<DVec3>,
    /// The same at the boundary rule.
    bnd_p: Vec<f64>,
    bnd_gradp: Vec<DVec3>,
    /// Coefficients in `ℚ_h`.
    pub coeff: DVector<f64>,
}

impl Pressure {
    pub fn new(ops: &Operators, p_min: f64) -> Self {
        let (np, nb) = (ops.vol.pts.len(), ops.bnd.pts.len());
        Self {
            p_vals: vec![p_min; np],
            ptil: vec![p_min; np],
            gradp: vec![DVec3::ZERO; np],
            bnd_p: vec![p_min; nb],
            bnd_gradp: vec![DVec3::ZERO; nb],
            coeff: DVector::zeros(ops.q_vol.val.ncols()),
        }
    }

    /// The pressure of the coefficients `coeff` (a nearby solution's), or `p_min` when they
    /// do not fit this space. Only the start of the fixed-point iteration changes.
    pub fn warm(ops: &Operators, p_min: f64, coeff: &[f64]) -> Self {
        let mut pr = Self::new(ops, p_min);
        if coeff.len() == pr.coeff.len() && coeff.iter().all(|c| c.is_finite()) {
            pr.coeff = DVector::from_column_slice(coeff);
            pr.sample(ops, p_min);
        }
        pr
    }

    /// `p`, `p̃`, `∇p` at the volume and boundary rules from `coeff`.
    fn sample(&mut self, ops: &Operators, p_min: f64) {
        let pv = linalg::mul(&ops.q_vol.val, &self.coeff);
        let gp = linalg::mul(&ops.q_vol.grad, &self.coeff);
        for p in 0..ops.vol.pts.len() {
            self.p_vals[p] = pv[p];
            self.ptil[p] = pv[p].max(p_min);
            self.gradp[p] = DVec3::new(gp[3 * p], gp[3 * p + 1], gp[3 * p + 2]);
        }
        let pb = &ops.q_bnd.val * &self.coeff;
        let gb = &ops.q_bnd.grad * &self.coeff;
        for b in 0..ops.bnd.pts.len() {
            self.bnd_p[b] = pb[b];
            self.bnd_gradp[b] = DVec3::new(gb[3 * b], gb[3 * b + 1], gb[3 * b + 2]);
        }
    }

    /// One Nitsche solve (G3) with the flow stress of the current velocity `u`:
    /// `a_N(p, q) = ∫ f·∇q − ∮ (f·n) q − ∮ p_Γ ∂_n q + (γ_N/h_N) ∮ p_Γ q`, with
    /// `p_Γ = p_ext + n·T_tide·n + 2η n·D·n` on the boundary rule.
    #[allow(clippy::too_many_arguments)]
    pub fn update(
        &mut self,
        ops: &Operators,
        ld: &Loads,
        prm: &StokesParams,
        m: &Material,
        eps: f64,
        coh: &[f64],
        gcoh: &[DVec3],
        u: &DVector<f64>,
        fields: &Fields,
    ) -> Result<(), SimError> {
        let (np, nb) = (ops.vol.pts.len(), ops.bnd.pts.len());
        let p_min = m.spec.p_min;
        let tide_div = tide_divergence(ld, m);
        let [_, lap, gd] = strains(&ops.rows, u);
        let mut wf = DVector::zeros(3 * np);
        for p in 0..np {
            let f = body_force(
                m,
                eps,
                (coh[p], gcoh[p]),
                (
                    &fields.d.as_slice()[6 * p..6 * p + 6],
                    &lap.as_slice()[3 * p..3 * p + 3],
                    &gd.as_slice()[18 * p..18 * p + 18],
                ),
                (self.ptil[p], self.p_vals[p], self.gradp[p]),
                ld.g_eff[p] * ops.vol.rho[p],
                ops.vol.grho[p] * tide_div,
            );
            for a in 0..3 {
                wf[3 * p + a] = ops.vol.w[p] * f[a];
            }
        }
        let mut rhs = linalg::tr_mul(&ops.q_vol.grad, &wf);
        // Boundary terms on the interior trace, depth 0: c = c_s and ∇c = −c'(0) n.
        let [d, lap, gd] = strains(&ops.bnd_rows, u);
        let (c_s, slope) = (m.cohesion_at(0.0), m.cohesion_slope(0.0));
        let (mut wq, mut wdn) = (DVector::zeros(nb), DVector::zeros(nb));
        for b in 0..nb {
            let n = ops.bnd.normals[b];
            let d6 = &d.as_slice()[6 * b..6 * b + 6];
            let ptil = self.bnd_p[b].max(p_min);
            let f = body_force(
                m,
                eps,
                (c_s, n * -slope),
                (d6, &lap.as_slice()[3 * b..3 * b + 3], &gd.as_slice()[18 * b..18 * b + 18]),
                (ptil, self.bnd_p[b], self.bnd_gradp[b]),
                ld.g_bnd[b] * ops.bnd_rho[b],
                ops.bnd_grho[b] * tide_div,
            );
            let p_gamma = traction(m, prm, eps, c_s, d6, ptil, &ld.tide_bnd[b], n);
            wq[b] = ops.bnd.w[b] * (ops.gamma_over_h * p_gamma - f.dot(n));
            wdn[b] = ops.bnd.w[b] * p_gamma;
        }
        rhs += ops.q_bnd.val.tr_mul(&wq) - ops.q_bnd_dn.tr_mul(&wdn);
        self.coeff = ops.nitsche.solve(&rhs);
        self.sample(ops, p_min);
        Ok(())
    }

    /// The coefficients split into the interior family and the `(r/R_*)^l` family.
    pub fn split(&self, ops: &Operators) -> (Vec<f64>, Vec<f64>) {
        let a = ops.play.n_p();
        let c = self.coeff.as_slice();
        (c[..a].to_vec(), c[a..].to_vec())
    }

    /// `p_Γ` at the face centres for the converged `u`, with `p̃` from the pressure there.
    pub fn face_traction(
        &self,
        ops: &Operators,
        ld: &Loads,
        prm: &StokesParams,
        m: &Material,
        eps: f64,
        u: &DVector<f64>,
    ) -> Result<Vec<f64>, SimError> {
        let (pres_c, lift) = self.split(ops);
        let p_face = contract_pressure(&ops.surf, &ops.play, &ops.faces.pts, &pres_c, &lift);
        let d = ops.surf_d.contract(u.as_slice()).map_err(SimError::gpu)?;
        let c_s = m.cohesion_at(0.0);
        Ok((0..ops.faces.pts.len())
            .map(|f| {
                let ev = eta_eval(
                    m,
                    c_s,
                    (2.0 * d6_dot(&d[6 * f..6 * f + 6], &d[6 * f..6 * f + 6])).sqrt(),
                    p_face[f].max(m.spec.p_min),
                    eps,
                );
                let n = ops.faces.normals[f];
                prm.p_ext + ld.tide_nn[f] + 2.0 * ev.eta * d_apply(&d[6 * f..6 * f + 6], n).dot(n)
            })
            .collect())
    }

    /// `ε_rot = ‖f − ∇p‖ / ‖f‖` over the volume points (strong-form diagnostic of §8.7).
    #[allow(clippy::too_many_arguments)]
    pub fn rotational_residual(
        &self,
        ops: &Operators,
        ld: &Loads,
        m: &Material,
        eps: f64,
        coh: &[f64],
        gcoh: &[DVec3],
        u: &DVector<f64>,
        fields: &Fields,
    ) -> f64 {
        let tide_div = tide_divergence(ld, m);
        let [_, lap, gd] = strains(&ops.rows, u);
        let (mut num, mut den) = (0.0, 0.0);
        for p in 0..ops.vol.pts.len() {
            let f = body_force(
                m,
                eps,
                (coh[p], gcoh[p]),
                (
                    &fields.d.as_slice()[6 * p..6 * p + 6],
                    &lap.as_slice()[3 * p..3 * p + 3],
                    &gd.as_slice()[18 * p..18 * p + 18],
                ),
                (self.ptil[p], self.p_vals[p], self.gradp[p]),
                ld.g_eff[p] * ops.vol.rho[p],
                ops.vol.grho[p] * tide_div,
            );
            num += ops.vol.w[p] * (f - self.gradp[p]).length_squared();
            den += ops.vol.w[p] * f.length_squared();
        }
        (num / den.max(1e-300)).sqrt()
    }
}

/// `∇·T_tide = −(8π/3) G C_T ρ_p ν₀ d_p² ∇ρ_K` per unit `∇ρ_K` (§8.7).
fn tide_divergence(ld: &Loads, m: &Material) -> f64 {
    -(8.0 * PI / 3.0) * ld.g_const * m.tide_coef
}

/// `p_Γ = p_ext + n·T_tide·n + 2η n·D·n` at one boundary point.
#[allow(clippy::too_many_arguments)]
fn traction(
    m: &Material,
    prm: &StokesParams,
    eps: f64,
    c: f64,
    d6: &[f64],
    ptil: f64,
    tide: &[f64; 6],
    n: DVec3,
) -> f64 {
    let ev = eta_eval(m, c, (2.0 * d6_dot(d6, d6)).sqrt(), ptil, eps);
    prm.p_ext + super::super::mechanics::quad_form(*tide, n) + 2.0 * ev.eta * d_apply(d6, n).dot(n)
}
