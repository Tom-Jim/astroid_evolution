//! Quasi-static Stokes–pressure solve of §8.4′, §8.7 and §10.4.2 on the general domain
//! of hypothesis (G) (§8.0).
//!
//! * velocity: the enclosing-ball basis of radius `R_*` (G1), exactly divergence-free,
//!   restricted to `𝒱` and integrated by the ray-segment rule (G2);
//! * pressure: the scalar Poisson problem with the free-surface datum `p_Γ` imposed by
//!   Nitsche's method (G3), whose matrix is factored once per geometry;
//! * the pair is closed by the primal–dual iteration ([`pd`]), or by the
//!   fixed-viscosity Picard map kept as a check ([`picard`]).

mod pd;
mod picard;
mod pressure;

use std::f64::consts::PI;

use nalgebra::{Cholesky, DMatrix, DVector, Dyn};

use crate::common::density::CauchyDensity;
use crate::common::math::DVec3;
use crate::rheo::Material;
use crate::rheo::closure::dynamic_friction;
use crate::rtfp::mesh::BodyMesh;

use super::basis::{
    PresLayout, SurfaceStrain, VelLayout, VelRows, Want, eval_pressure, eval_velocity_rays_as,
    surface_strain,
};
use super::domain::{SurfacePoints, VolumeSet};
use super::harmonics::SurfaceFit;
use super::params::StokesParams;
use super::surface::SurfaceIndex;
use crate::common::error::SimError;
use crate::common::linalg::{self, Entry};

use pd::solve_pd;
use picard::solve_picard;

/// Values and gradients of the pressure space `ℚ_h` at a point set: the interior family
/// `(1 − r/R_*) ψ_{lk} Y_{lm}` and the family `(r/R_*)^l Y_{lm}`, all free under (G3).
struct QRows<T: Entry = f64> {
    val: DMatrix<T>,
    grad: DMatrix<T>,
}

impl<T: Entry> QRows<T> {
    fn eval(surf: &SurfaceFit, play: &PresLayout, pts: &[DVec3]) -> Self {
        let r = eval_pressure(surf, play, pts, true);
        let a = play.n_p();
        // Interior columns, then the lift columns, stored as `T`.
        let join = |inner: &DMatrix<f64>, lift: &DMatrix<f64>| {
            DMatrix::from_fn(inner.nrows(), a + lift.ncols(), |i, j| {
                T::of(if j < a { inner[(i, j)] } else { lift[(i, j - a)] })
            })
        };
        Self {
            val: join(&r.val, &r.lift_val),
            grad: join(&r.grad, &r.lift_grad),
        }
    }

    /// `∂_n q` at each point.
    fn normal(&self, normals: &[DVec3]) -> DMatrix<f64> {
        let mut dn = DMatrix::zeros(normals.len(), self.val.ncols());
        for (p, n) in normals.iter().enumerate() {
            for c in 0..dn.ncols() {
                dn[(p, c)] = (0..3).map(|a| n[a] * self.grad[(3 * p + a, c)].into()).sum();
            }
        }
        dn
    }
}

/// Geometry-dependent operators, built once per geometry and density.
pub struct Operators {
    /// The enclosing-ball fit `R ≡ R_*` that drives the basis kernels (G1).
    pub surf: SurfaceFit,
    pub lay: VelLayout,
    pub play: PresLayout,
    pub vol: VolumeSet,
    /// The boundary rule of the Nitsche terms.
    pub bnd: SurfacePoints,
    /// Face centres of `Γ`: the nodes of the friction chain and of `p_Γ`.
    pub faces: SurfacePoints,
    /// Volume rows in f32: the largest store of the solve (30 rows per point).
    pub rows: VelRows<f32>,
    bnd_rows: VelRows<f32>,
    bnd_rho: Vec<f64>,
    bnd_grho: Vec<DVec3>,
    q_vol: QRows<f32>,
    q_bnd: QRows,
    q_bnd_dn: DMatrix<f64>,
    nitsche: Cholesky<f64, Dyn>,
    /// `γ_N / h_N` and `h_N = R_*`; `C_N` of Proposition 8.0.6 is reported.
    gamma_over_h: f64,
    pub c_nitsche: f64,
    /// Strain basis at the face centres, resident on the device.
    pub surf_d: SurfaceStrain,
    pub rigid: DMatrix<f64>,
    /// Smallest singular value of the row-normalised rigid-body constraint matrix.
    pub rigid_sigma_min: f64,
    /// Velocity rows on the classification mesh `X_Y`, kept from the second use of this
    /// geometry on (a step that moves the geometry classifies it once).
    pub class_rows: std::sync::OnceLock<super::basis::ResidentVelocity>,
    pub class_uses: std::sync::atomic::AtomicUsize,
    /// The classification mesh `X_Y` of (G7).
    pub class_mesh: std::sync::OnceLock<super::classify::ClassMesh>,
    /// Gravity at the volume and boundary points; independent of the spin.
    pub vol_field: std::sync::OnceLock<super::forward::PointFields>,
    pub bnd_field: std::sync::OnceLock<super::forward::PointFields>,
}

impl Operators {
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        surf: &SurfaceFit,
        density: &CauchyDensity,
        mesh: &BodyMesh,
        index: &SurfaceIndex,
        faces: SurfacePoints,
        prm: &StokesParams,
    ) -> Result<Self, SimError> {
        let lay = VelLayout::new(prm.l_velocity);
        let play = PresLayout {
            l_max: prm.l_pressure,
            k_max: prm.k_pressure,
        };
        let vol = VolumeSet::build(
            mesh,
            index,
            density,
            prm.n_theta,
            prm.n_phi,
            prm.n_radial,
            prm.cut_depth,
        )?;
        let face_triples: Vec<(DVec3, DVec3, f64)> = (0..faces.pts.len())
            .map(|f| (faces.pts[f], faces.normals[f], faces.w[f]))
            .collect();
        let bnd = SurfacePoints::clustered(&face_triples, prm.n_boundary);
        let rows = eval_velocity_rays_as(surf, &lay, &vol.pts, 1, Want::ALL);
        let bnd_rows = eval_velocity_rays_as(surf, &lay, &bnd.pts, 1, Want::ALL);
        let surf_d = surface_strain(surf, &lay, &faces.pts).map_err(SimError::gpu)?;
        let rigid = rigid_matrix(&vol, &rows, lay.n_u);
        let rigid_sigma_min = rigid_rank(&rigid, prm.rigid_sigma_min)?;

        // Nitsche matrix a_N (G3) with h_N = R_* and γ_N = 4 C_N.
        let q_vol = QRows::eval(surf, &play, &vol.pts);
        let q_bnd = QRows::eval(surf, &play, &bnd.pts);
        let q_bnd_dn = q_bnd.normal(&bnd.normals);
        let h_n = surf.coeff[0] / (4.0 * PI).sqrt();
        let stiff = linalg::weighted_gram(&q_vol.grad, 3, &vol.w);
        let weighted = |m: &DMatrix<f64>| {
            let mut out = m.clone();
            for (p, w) in bnd.w.iter().enumerate() {
                out.row_mut(p).scale_mut(*w);
            }
            out
        };
        let dn_w = weighted(&q_bnd_dn);
        let flux = dn_w.tr_mul(&q_bnd.val);
        let mass = weighted(&q_bnd.val).tr_mul(&q_bnd.val);
        let c_nitsche = nitsche_constant(&stiff, &(dn_w.tr_mul(&q_bnd_dn) * h_n));
        let gamma_over_h = (4.0 * c_nitsche).max(1.0) / h_n;
        let a_n = &stiff - &flux - flux.transpose() + mass * gamma_over_h;
        let nitsche = Cholesky::new(a_n)
            .ok_or_else(|| SimError::solver("the Nitsche pressure matrix is not positive definite"))?;
        Ok(Self {
            surf: surf.clone(),
            lay,
            play,
            bnd_rho: bnd.pts.iter().map(|p| density.rho(*p)).collect(),
            bnd_grho: bnd.pts.iter().map(|p| density.grad(*p)).collect(),
            vol,
            bnd,
            faces,
            rows,
            bnd_rows,
            q_vol,
            q_bnd,
            q_bnd_dn,
            nitsche,
            gamma_over_h,
            c_nitsche,
            surf_d,
            rigid,
            rigid_sigma_min,
            class_rows: std::sync::OnceLock::new(),
            class_uses: std::sync::atomic::AtomicUsize::new(0),
            class_mesh: std::sync::OnceLock::new(),
            vol_field: std::sync::OnceLock::new(),
            bnd_field: std::sync::OnceLock::new(),
        })
    }
}

/// Smallest singular value of the row-normalised `B_rb`; an error below `sigma_min`.
fn rigid_rank(rigid: &DMatrix<f64>, sigma_min: f64) -> Result<f64, SimError> {
    let mut norm = rigid.clone();
    for a in 0..6 {
        let n = norm.row(a).norm();
        if !(n.is_finite() && n > 0.0) {
            return Err(SimError::rigid_rank("a rigid-body constraint row vanishes"));
        }
        norm.row_mut(a).scale_mut(1.0 / n);
    }
    let smin = (&norm * norm.transpose())
        .symmetric_eigen()
        .eigenvalues
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min)
        .max(0.0)
        .sqrt();
    if smin < sigma_min {
        return Err(SimError::rigid_rank(format!(
            "rank of B_rb is deficient: smallest normalised singular value {smin:.3e}"
        )));
    }
    Ok(smin)
}

/// `C_N = max h_N ‖∂_n q‖²_Γ / ‖∇q‖²_𝒱` (Proposition 8.0.6): the largest eigenvalue of
/// `boundary` relative to `stiff`. Constants (`∇q = 0`, `∂_n q = 0`) are removed by a
/// relative shift of `stiff` that does not change the other eigenvalues materially.
fn nitsche_constant(stiff: &DMatrix<f64>, boundary: &DMatrix<f64>) -> f64 {
    let n = stiff.nrows();
    let shift = 1e-10 * stiff.trace().max(1e-300) / n as f64;
    let Some(l) = Cholesky::new(stiff + DMatrix::identity(n, n) * shift) else {
        return 0.0;
    };
    let li = l.l().try_inverse().unwrap_or_else(|| DMatrix::identity(n, n));
    let c = &li * boundary * li.transpose();
    c.symmetric_eigen().eigenvalues.iter().copied().fold(0.0, f64::max)
}

/// `B_rb` of §8.5: rows `∫ρ B·e_a`, `∫ρ (x × B)·e_a`.
fn rigid_matrix(vol: &VolumeSet, rows: &VelRows<f32>, n_u: usize) -> DMatrix<f64> {
    let mut b = DMatrix::zeros(6, n_u);
    for (p, x) in vol.pts.iter().enumerate() {
        let wr = vol.w[p] * vol.rho[p];
        for j in 0..n_u {
            let [vx, vy, vz] = std::array::from_fn(|a| f64::from(rows.v[(3 * p + a, j)]));
            b[(0, j)] += wr * vx;
            b[(1, j)] += wr * vy;
            b[(2, j)] += wr * vz;
            b[(3, j)] += wr * (x.y * vz - x.z * vy);
            b[(4, j)] += wr * (x.z * vx - x.x * vz);
            b[(5, j)] += wr * (x.x * vy - x.y * vx);
        }
    }
    b
}

/// Viscosity of §4 and its partial derivatives, all analytic.
#[derive(Clone, Copy, Debug)]
pub struct EtaEval {
    pub eta: f64,
    pub d_shear: f64,
    pub d_p: f64,
    /// `∂η/∂c`.
    pub d_c: f64,
    pub mu_d: f64,
    pub inertial: f64,
}

/// `η = (c + μ_d(I_c) p̃) / √(γ̇² + γ̇_ε²)`, `I_c = d_p γ̇ / √((p̃ + p_c)/ρ_p)`,
/// with the local cohesion `c = c(d)` of §4.3a.
pub fn eta_eval(m: &Material, c: f64, shear: f64, p_tilde: f64, eps: f64) -> EtaEval {
    let s = &m.spec;
    let p_star = p_tilde + m.p_c;
    let (mu_d, inertial) = dynamic_friction(m, shear, p_star);
    let scale = (p_star / s.rho_p).sqrt();
    let df = s.d_p * shear;
    let den = s.i_0 * scale + df;
    let dmu = s.mu_2 - s.mu_s;
    let (dmu_dg, dmu_dp) = if den > 0.0 {
        (
            dmu * s.d_p * s.i_0 * scale / (den * den),
            -dmu * df * s.i_0 / (2.0 * scale * s.rho_p) / (den * den),
        )
    } else {
        (0.0, 0.0)
    };
    let big = (shear * shear + eps * eps).sqrt();
    let a = c + mu_d * p_tilde;
    EtaEval {
        eta: a / big,
        d_c: 1.0 / big,
        d_shear: p_tilde * dmu_dg / big - a * shear / (big * big * big),
        d_p: (mu_d + p_tilde * dmu_dp) / big,
        mu_d,
        inertial,
    }
}

/// Loads of one geometric step, frozen for the solve.
pub struct Loads<'a> {
    /// `g_eff` and `T_tide` (`(xx, yy, zz, xy, xz, yz)`) at the volume points.
    pub g_eff: &'a [DVec3],
    pub tide: &'a [[f64; 6]],
    /// The same at the points of the boundary rule.
    pub g_bnd: &'a [DVec3],
    pub tide_bnd: &'a [[f64; 6]],
    /// `n · T_tide · n` at the face centres.
    pub tide_nn: &'a [f64],
    pub g_const: f64,
}

#[derive(Clone, Debug)]
pub struct StokesSolution {
    pub u: Vec<f64>,
    pub lambda: [f64; 6],
    /// Pressure coefficients: the interior family, then the `(r/R_*)^l Y_{lm}` family.
    pub pres_c: Vec<f64>,
    pub lift: Vec<f64>,
    /// Final viscosity at the volume points.
    pub eta: Vec<f64>,
    /// Regularised pressure `p̃` at the volume points.
    pub p_tilde: Vec<f64>,
    /// `p_Γ = p_ext + n·τ·n` at the face centres.
    pub p_surface: Vec<f64>,
    pub picard_iterations: usize,
    pub eps_picard: f64,
    pub eps_momentum: f64,
    pub eps_lambda: f64,
    pub eps_rot: f64,
    pub eta_lo: f64,
    pub eta_hi: f64,
    pub tensile_fraction: f64,
    pub inverse_ekman: f64,
    pub gamma_eps: f64,
}

pub(super) fn d6_dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// `D · v` from the scaled six-vector.
pub(super) fn d_apply(d: &[f64], v: DVec3) -> DVec3 {
    let s = std::f64::consts::FRAC_1_SQRT_2;
    let (xy, xz, yz) = (d[3] * s, d[4] * s, d[5] * s);
    DVec3::new(
        d[0] * v.x + xy * v.y + xz * v.z,
        xy * v.x + d[1] * v.y + yz * v.z,
        xz * v.x + yz * v.y + d[2] * v.z,
    )
}

pub(super) struct Fields {
    /// `γ̇` at the volume points.
    pub(super) shear: Vec<f64>,
    /// `D` at the volume points (six per point, scaled).
    pub(super) d: DVector<f64>,
}

pub(super) fn strain_fields(ops: &Operators, u: &DVector<f64>) -> Fields {
    let d = linalg::mul(&ops.rows.d, u);
    let np = ops.vol.pts.len();
    let shear = (0..np)
        .map(|p| {
            let dd = d6_dot(
                &d.as_slice()[6 * p..6 * p + 6],
                &d.as_slice()[6 * p..6 * p + 6],
            );
            (2.0 * dd).sqrt()
        })
        .collect();
    Fields { shear, d }
}

pub(super) fn assemble_k(ops: &Operators, eta: &[f64]) -> DMatrix<f64> {
    // K = Σ_q 2 η_q w_q D_qᵀ D_q.
    let s: Vec<f64> = eta.iter().zip(&ops.vol.w).map(|(e, w)| 2.0 * e * w).collect();
    linalg::weighted_gram(&ops.rows.d, 6, &s)
}

pub(super) fn assemble_f(ops: &Operators, ld: &Loads) -> DVector<f64> {
    let np = ops.vol.pts.len();
    let mut gv = DVector::zeros(3 * np);
    let mut tv = DVector::zeros(6 * np);
    for p in 0..np {
        let wr = ops.vol.w[p] * ops.vol.rho[p];
        gv[3 * p] = wr * ld.g_eff[p].x;
        gv[3 * p + 1] = wr * ld.g_eff[p].y;
        gv[3 * p + 2] = wr * ld.g_eff[p].z;
        let t = ld.tide[p];
        let s2 = std::f64::consts::SQRT_2;
        let w = ops.vol.w[p];
        tv[6 * p] = w * t[0];
        tv[6 * p + 1] = w * t[1];
        tv[6 * p + 2] = w * t[2];
        tv[6 * p + 3] = w * s2 * t[3];
        tv[6 * p + 4] = w * s2 * t[4];
        tv[6 * p + 5] = w * s2 * t[5];
    }
    linalg::tr_mul(&ops.rows.v, &gv) - linalg::tr_mul(&ops.rows.d, &tv)
}

/// Magnitude scale of the load: the same sums with absolute values, so that a load that
/// cancels exactly (a uniform sphere in its own gravity) does not make relative residuals
/// meaningless.
pub(super) fn load_scale(ops: &Operators, ld: &Loads) -> f64 {
    let np = ops.vol.pts.len();
    let n_u = ops.lay.n_u;
    let mut acc = DVector::<f64>::zeros(n_u);
    for p in 0..np {
        let wg = ops.vol.w[p] * ops.vol.rho[p] * ld.g_eff[p].length();
        let t = ld.tide[p];
        let wt = ops.vol.w[p] * t.iter().map(|x| x * x).sum::<f64>().sqrt();
        for j in 0..n_u {
            let vmag = (0..3)
                .map(|a| f64::from(ops.rows.v[(3 * p + a, j)]).powi(2))
                .sum::<f64>()
                .sqrt();
            let dmag = (0..6)
                .map(|c| f64::from(ops.rows.d[(6 * p + c, j)]).powi(2))
                .sum::<f64>()
                .sqrt();
            acc[j] += wg * vmag + wt * dmag;
        }
    }
    acc.norm().max(1e-300)
}

pub(super) struct RangeSolve {
    chol: Cholesky<f64, Dyn>,
    pub(super) y: DMatrix<f64>,
    pub(super) s_chol: Cholesky<f64, Dyn>,
}

pub(super) fn factor_range(ops: &Operators, k: &DMatrix<f64>) -> Result<RangeSolve, SimError> {
    let n = k.nrows();
    let bt = ops.rigid.transpose();
    let btb = &bt * &ops.rigid;
    let kmax = (0..n).map(|i| k[(i, i)]).fold(0.0, f64::max);
    let bmax = (0..n).map(|i| btb[(i, i)]).fold(0.0, f64::max);
    let gamma = kmax / bmax.max(1e-300);
    let kg = k + btb * gamma;
    let chol = Cholesky::new(kg)
        .ok_or_else(|| SimError::rigid_rank("K + γ B_rbᵀB_rb is not positive definite"))?;
    let y = chol.solve(&bt);
    let s = &ops.rigid * &y;
    let s_chol = Cholesky::new(s)
        .ok_or_else(|| SimError::rigid_rank("the 6×6 rigid-body Schur complement is singular"))?;
    Ok(RangeSolve { chol, y, s_chol })
}

pub(super) fn solve_range(ops: &Operators, rs: &RangeSolve, f: &DVector<f64>) -> (DVector<f64>, DVector<f64>) {
    let y0 = rs.chol.solve(f);
    let lam = rs.s_chol.solve(&(&ops.rigid * &y0));
    let u = y0 - &rs.y * &lam;
    (u, lam)
}

/// The Stokes–pressure pair at fixed geometry (§10.4.2): the primal–dual iteration, or the
/// fixed-viscosity Picard map that has the same fixed points.
pub fn solve(
    ops: &Operators,
    ld: &Loads,
    prm: &StokesParams,
    m: &Material,
    guess: Option<&[f64]>,
) -> Result<StokesSolution, SimError> {
    match prm.solver.as_str() {
        "picard" => solve_picard(ops, ld, prm, m, guess),
        "primal-dual" => solve_pd(ops, ld, prm, m, guess),
        other => Err(SimError::solver(format!("unknown Stokes solver {other:?}"))),
    }
}

