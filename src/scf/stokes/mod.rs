//! Quasi-static Stokes–pressure solve of §8.4′, §8.7 and §10.4.2 on the general domain
//! of hypothesis (G) (§8.0).
//!
//! * velocity: the enclosing-ball basis of radius `R_*` (G1), exactly divergence-free,
//!   restricted to `𝒱` and integrated by the ray-segment rule (G2);
//! * pressure: the scalar Poisson problem with the free-surface datum `p_Γ` imposed by
//!   Nitsche's method (G3), whose matrix is factored once per geometry;
//! * the pair is closed by the primal–dual iteration ([`pd`]), or by the
//!   fixed-viscosity Picard map kept as a check ([`picard`]);
//! * the certified bracket of the limit load of §2.17.3 ([`limit`]).

pub mod elastic;
pub mod limit;
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
use super::evolving::dilatation::Dilatational;
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

/// Where a point's basis rows come from when a geometry is rebuilt: point `i` of the
/// previous volume rule, or point `j` of the newly evaluated ones.
#[derive(Clone, Copy)]
enum Source {
    Old(usize),
    New(usize),
}

/// Relative distance `δ / R_ref` within which a node takes the rows of an old node: the
/// integrand of that node is then off by `O(L δ / R_ref)` (§8.1), below 1e-4 relative.
const REUSE_DISTANCE: f64 = 1.0e-5;

/// Each point of `pts` mapped to the nearest point of `old` within `reach`, or to the
/// list of points to evaluate.
fn reuse_map(pts: &[DVec3], old: &[DVec3], reach: f64) -> (Vec<Source>, Vec<DVec3>) {
    let cell = |p: &DVec3| (*p / reach).floor().as_i64vec3().to_array();
    let mut grid: std::collections::HashMap<[i64; 3], Vec<usize>> = std::collections::HashMap::new();
    for (i, p) in old.iter().enumerate() {
        grid.entry(cell(p)).or_default().push(i);
    }
    let mut fresh = Vec::new();
    let src = pts
        .iter()
        .map(|p| {
            let c = cell(p);
            let mut best: Option<(f64, usize)> = None;
            for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        for &i in grid.get(&[c[0] + dx, c[1] + dy, c[2] + dz]).into_iter().flatten() {
                            let d = (old[i] - *p).length();
                            if d <= reach && best.is_none_or(|b| d < b.0) {
                                best = Some((d, i));
                            }
                        }
                    }
                }
            }
            match best {
                Some((_, i)) => Source::Old(i),
                None => {
                    fresh.push(*p);
                    Source::New(fresh.len() - 1)
                }
            }
        })
        .collect();
    (src, fresh)
}

/// Rows of `stride` per point, each taken from the old or the new block: one contiguous
/// copy per point and column, the columns in parallel.
fn gather<T: Entry + Send + Sync>(old: &DMatrix<T>, new: &DMatrix<T>, src: &[Source], stride: usize) -> DMatrix<T> {
    use rayon::prelude::*;
    let cols = old.ncols().max(new.ncols());
    let rows = stride * src.len();
    let mut out = DMatrix::from_element(rows, cols, T::of(0.0));
    if rows == 0 {
        return out;
    }
    let (no, nn) = (old.nrows(), new.nrows());
    out.as_mut_slice().par_chunks_mut(rows).enumerate().for_each(|(c, col)| {
        for (p, s) in src.iter().enumerate() {
            let (from, at) = match *s {
                Source::Old(i) => (&old.as_slice()[c * no..(c + 1) * no], stride * i),
                Source::New(j) => (&new.as_slice()[c * nn..(c + 1) * nn], stride * j),
            };
            col[stride * p..stride * (p + 1)].copy_from_slice(&from[at..at + stride]);
        }
    });
    out
}

/// What the next geometry takes over from this one: the basis scale `R_ref` (§8.1) and
/// the volume rows by point. Everything else of the old geometry is released first.
pub struct Reuse {
    pub r_ref: f64,
    pts: Vec<DVec3>,
    rows: VelRows<f32>,
    q_vol: QRows<f32>,
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
    /// The classification mesh `X_Y` of (G7).
    pub class_mesh: std::sync::OnceLock<super::classify::ClassMesh>,
    /// Gravity at the volume and boundary points; independent of the spin.
    pub vol_field: std::sync::OnceLock<super::forward::PointFields>,
    pub bnd_field: std::sync::OnceLock<super::forward::PointFields>,
    /// The primal–dual metric `G` (the stiffness at `η ≡ 1`) and its range-space
    /// factorization: fixed by the geometry, so a spin-only step reuses them.
    pd_metric: std::sync::OnceLock<(DMatrix<f64>, RangeSolve)>,
}

impl Operators {
    #[allow(clippy::too_many_arguments)]
    /// `v` and the scaled strain six-vector `D` at the volume nodes, from the host rows.
    pub fn vol_velocity(&self, u: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let u = DVector::from_column_slice(u);
        let v = linalg::mul(&self.rows.v, &u);
        let d = linalg::mul(&self.rows.d, &u);
        (v.as_slice().to_vec(), d.as_slice().to_vec())
    }

    /// The pressure `P c + lift` at the volume nodes, from the host rows.
    pub fn vol_pressure(&self, pres_c: &[f64], lift: &[f64]) -> Vec<f64> {
        let coeff = DVector::from_iterator(pres_c.len() + lift.len(), pres_c.iter().chain(lift).copied());
        linalg::mul(&self.q_vol.val, &coeff).as_slice().to_vec()
    }

    /// The rows and scale the next geometry can reuse; the rest is released.
    pub fn into_reuse(self) -> Reuse {
        Reuse {
            r_ref: self.surf.coeff[0] / (4.0 * PI).sqrt(),
            pts: self.vol.pts,
            rows: self.rows,
            q_vol: self.q_vol,
        }
    }

    /// `prev` (same basis scale) supplies the rows of the volume nodes that moved by at
    /// most `REUSE_DISTANCE · R_ref` (the crossings of most rays barely move in a step).
    pub fn build(
        surf: &SurfaceFit,
        density: &CauchyDensity,
        mesh: &BodyMesh,
        index: &SurfaceIndex,
        faces: SurfacePoints,
        prm: &StokesParams,
        prev: Option<Reuse>,
    ) -> Result<Self, SimError> {
        let lay = VelLayout::new(prm.l_velocity);
        let play = PresLayout {
            l_max: prm.l_pressure,
            k_max: prm.k_pressure,
        };
        let span = super::timing::span("operators_rule");
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
        drop(span);
        let span = super::timing::span("operators_volume_rows");
        // Rows of another basis scale are other functions: they are never taken over.
        let h_n = surf.coeff[0] / (4.0 * PI).sqrt();
        let prev = prev.filter(|old| (old.r_ref - h_n).abs() <= 1e-12 * h_n);
        let (rows, q_vol) = match prev {
            Some(old) => {
                let (src, fresh) = reuse_map(&vol.pts, &old.pts, REUSE_DISTANCE * old.r_ref);
                let r = eval_velocity_rays_as::<f32>(surf, &lay, &fresh, 1, Want::ALL);
                let q = QRows::<f32>::eval(surf, &play, &fresh);
                let _gather = super::timing::span("operators_volume_gather");
                (
                    VelRows {
                        n_points: vol.pts.len(),
                        v: gather(&old.rows.v, &r.v, &src, 3),
                        d: gather(&old.rows.d, &r.d, &src, 6),
                        lap: gather(&old.rows.lap, &r.lap, &src, 3),
                        gd: gather(&old.rows.gd, &r.gd, &src, 18),
                    },
                    QRows {
                        val: gather(&old.q_vol.val, &q.val, &src, 1),
                        grad: gather(&old.q_vol.grad, &q.grad, &src, 3),
                    },
                )
            }
            None => (
                eval_velocity_rays_as(surf, &lay, &vol.pts, 1, Want::ALL),
                QRows::eval(surf, &play, &vol.pts),
            ),
        };
        let rigid = rigid_matrix(&vol, &rows, lay.n_u);
        let rigid_sigma_min = rigid_rank(&rigid, prm.rigid_sigma_min)?;
        drop(span);
        let span = super::timing::span("operators_boundary_rows");
        let bnd_rows = eval_velocity_rays_as(surf, &lay, &bnd.pts, 1, Want::ALL);
        let surf_d = surface_strain(surf, &lay, &faces.pts).map_err(SimError::gpu)?;
        drop(span);
        let _span = super::timing::span("operators_nitsche");

        // Nitsche matrix a_N (G3) with h_N = R_* and γ_N = 4 C_N.
        let q_bnd = QRows::eval(surf, &play, &bnd.pts);
        let q_bnd_dn = q_bnd.normal(&bnd.normals);
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
            class_mesh: std::sync::OnceLock::new(),
            vol_field: std::sync::OnceLock::new(),
            bnd_field: std::sync::OnceLock::new(),
            pd_metric: std::sync::OnceLock::new(),
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

/// Local strength of the viscoplastic law at a point: the cohesion `c` (§4.3a, under A-ϖ
/// `(1 − ϖ) c`), the static dilatancy shift `ξ = K_Ξ (ν − ν_c⁰)` of the friction (A-Ξ,
/// §11.3), and their gradients for `∇η` (§8.7).
#[derive(Clone, Copy, Debug, Default)]
pub struct Local {
    pub c: f64,
    pub xi: f64,
    pub gc: DVec3,
    pub gxi: DVec3,
}

/// The local strengths at the volume nodes, the boundary rule and the face centres.
#[derive(Clone, Debug, Default)]
pub struct Strengths {
    pub vol: Vec<Local>,
    pub bnd: Vec<Local>,
    pub face: Vec<Local>,
}

impl Strengths {
    /// The intact profile of A-C2: `c(d)` and `∇c = c'(d) ∇d`, no dilatancy shift. The
    /// boundary and face values are the interior traces at depth 0, `∇d = −n`.
    pub fn profile(ops: &Operators, m: &Material) -> Self {
        let trace = |n: &DVec3| Local {
            c: m.cohesion_at(0.0),
            gc: *n * -m.cohesion_slope(0.0),
            ..Local::default()
        };
        Self {
            vol: (0..ops.vol.pts.len())
                .map(|p| Local {
                    c: m.cohesion_at(ops.vol.depth[p]),
                    gc: ops.vol.gdepth[p] * m.cohesion_slope(ops.vol.depth[p]),
                    ..Local::default()
                })
                .collect(),
            bnd: ops.bnd.normals.iter().map(trace).collect(),
            face: ops.faces.normals.iter().map(trace).collect(),
        }
    }
}

/// The evolving material of Chapter 11 seen by the solver: the local strengths at any
/// points, given the intact ones there.
pub trait MaterialField: Sync {
    fn locals(&self, m: &Material, pts: &[DVec3], intact: &[Local]) -> Vec<Local>;
}

/// The rate-type dilatancy of A-Ξ seen by the solver (§11.4): from the total shear rate,
/// `p̃` and the strengths at the volume nodes, the dilatational field of the next sweeps.
pub trait DilatancySource: Sync {
    fn field(&self, m: &Material, strength: &[Local], shear: &[f64], ptil: &[f64], eps: f64) -> DilatancyState;
}

/// One update of the dilatational field: `v_ε`, the projection defect `e_ε`, and the
/// largest driving rate `γ̇_p^⋆` (for `Δt_Ξ`).
#[derive(Clone, Debug)]
pub struct DilatancyState {
    pub field: Dilatational,
    pub defect: f64,
    pub driving_max: f64,
}

/// Viscosity of §4 and its partial derivatives, all analytic.
#[derive(Clone, Copy, Debug)]
pub struct EtaEval {
    pub eta: f64,
    pub d_shear: f64,
    pub d_p: f64,
    /// `∂η/∂c` and `∂η/∂ξ`.
    pub d_c: f64,
    pub d_xi: f64,
    pub mu_d: f64,
    pub inertial: f64,
}

/// `η = (c + μ_d p̃) / √(γ̇² + γ̇_ε²)`, `I_c = d_p γ̇ / √((p̃ + p_c)/ρ_p)`, with
/// `μ_d = μ_cs(I_c) + ξ + K_Ξ Δν_c I_c/(I_ν + I_c)` (the last two under A-Ξ only).
pub fn eta_eval(m: &Material, l: &Local, shear: f64, p_tilde: f64, eps: f64) -> EtaEval {
    let s = &m.spec;
    let p_star = p_tilde + m.p_c;
    let (mu_cs, inertial) = dynamic_friction(m, shear, p_star);
    let (extra, d_extra) = m.dilatancy_rate_term(inertial);
    let mu_d = mu_cs + l.xi + extra;
    let scale = (p_star / s.rho_p).sqrt();
    let df = s.d_p * shear;
    let den = s.i_0 * scale + df;
    let dmu = s.mu_2 - s.mu_s;
    // ∂I/∂γ̇ = d_p / scale, ∂I/∂p̃ = −I / (2 p_*).
    let (dmu_dg, dmu_dp) = if den > 0.0 {
        (
            dmu * s.d_p * s.i_0 * scale / (den * den) + d_extra * s.d_p / scale,
            -dmu * df * s.i_0 / (2.0 * scale * s.rho_p) / (den * den) - d_extra * inertial / (2.0 * p_star),
        )
    } else {
        (0.0, 0.0)
    };
    let big = (shear * shear + eps * eps).sqrt();
    let a = l.c + mu_d * p_tilde;
    EtaEval {
        eta: a / big,
        d_c: 1.0 / big,
        d_xi: p_tilde / big,
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
    /// Local strengths (§4.3a; Chapter 11).
    pub strength: &'a Strengths,
    /// A-Ξ; `None` keeps the velocity exactly divergence-free.
    pub dilatancy: Option<&'a dyn DilatancySource>,
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
    /// A-Ξ: the converged dilatational field (the total velocity is `v_ε + Σ u B`).
    pub dilatancy: Option<DilatancyState>,
}

impl StokesSolution {
    /// `v_ε(x)` (zero without A-Ξ).
    pub fn v_eps(&self, x: DVec3) -> DVec3 {
        self.dilatancy.as_ref().map_or(DVec3::ZERO, |d| d.field.velocity(x))
    }

    /// `D(v_ε)(x)` as the scaled six-vector, and `ε̇_{v,h}(x)` (zeros without A-Ξ).
    pub fn d_eps(&self, x: DVec3) -> ([f64; 6], f64) {
        self.dilatancy
            .as_ref()
            .map_or(([0.0; 6], 0.0), |d| (d.field.strain(x), d.field.divergence(x)))
    }
}

/// A warm start of the Stokes–pressure pair: the velocity and pressure coefficients of a
/// nearby solution. A part whose length does not fit the current spaces is ignored.
#[derive(Clone, Debug, Default)]
pub struct Guess {
    pub u: Vec<f64>,
    pub p: Vec<f64>,
}

impl StokesSolution {
    pub fn guess(&self) -> Guess {
        Guess {
            u: self.u.clone(),
            p: [self.pres_c.as_slice(), &self.lift].concat(),
        }
    }
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

/// `D'` of the total velocity and its shear rate at the volume nodes: the solenoidal part
/// of `u` plus the deviatoric shift `e^ε/√2` of A-Ξ (`None`: no shift).
pub(super) fn strain_fields_with(ops: &Operators, u: &DVector<f64>, shift: Option<&DVector<f64>>) -> Fields {
    let mut d = linalg::mul(&ops.rows.d, u);
    if let Some(e) = shift {
        d += e * std::f64::consts::FRAC_1_SQRT_2;
    }
    let np = ops.vol.pts.len();
    let shear = (0..np)
        .map(|p| (2.0 * d6_dot(&d.as_slice()[6 * p..6 * p + 6], &d.as_slice()[6 * p..6 * p + 6])).sqrt())
        .collect();
    Fields { shear, d }
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

/// `G` and its range-space factorization, built on the first primal–dual solve of this
/// geometry.
pub(super) fn pd_metric(ops: &Operators) -> Result<&(DMatrix<f64>, RangeSolve), SimError> {
    if let Some(m) = ops.pd_metric.get() {
        return Ok(m);
    }
    let _span = super::timing::span("pd_metric");
    let metric = assemble_k(ops, &vec![1.0; ops.vol.pts.len()]);
    let rs = factor_range(ops, &metric)?;
    Ok(ops.pd_metric.get_or_init(|| (metric, rs)))
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
    guess: Option<&Guess>,
) -> Result<StokesSolution, SimError> {
    match prm.solver.as_str() {
        "picard" => solve_picard(ops, ld, prm, m, guess),
        "primal-dual" => solve_pd(ops, ld, prm, m, guess),
        other => Err(SimError::solver(format!("unknown Stokes solver {other:?}"))),
    }
}

