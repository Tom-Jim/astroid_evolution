//! Quasi-static Stokes–pressure solve of §8.4′, §8.7 and §10.4.2.
//!
//! * velocity: exactly divergence-free modal Galerkin space, `K u + B_rbᵀ λ = F`, `B_rb u = 0`,
//!   solved in the range space with a dense Cholesky factor of `K + γ B_rbᵀ B_rb`;
//! * pressure: independent scalar Poisson problem with Dirichlet data `p_Γ` from the
//!   free-surface traction and the `(1 − ζ) ψ_{lk} Y_{lm}` basis;
//! * the two are closed by a Picard iteration with `γ̇_ε`-continuation and viscosity
//!   under-relaxation.

use std::f64::consts::PI;

use nalgebra::{Cholesky, DMatrix, DVector, Dyn};

use crate::common::density::CauchyDensity;
use crate::common::math::DVec3;
use crate::rheo_friction::Material;
use crate::rheo_friction::basis::sh_all;
use crate::rheo_friction::closure::dynamic_friction;

use super::basis::{PresLayout, PresRows, VelLayout, VelRows, Want, eval_pressure, eval_velocity};
use super::error::ScfError;
use super::harmonics::SurfaceFit;
use super::params::StokesParams;
use super::quad::{gauss_jacobi_zeta2, sphere_grid};

/// Volume quadrature points of the star domain `r < R_SH(ω̂)`.
#[derive(Clone, Debug)]
pub struct VolumeSet {
    pub pts: Vec<DVec3>,
    /// `dV` weights `ω_q R³ Ω_j`.
    pub w: Vec<f64>,
    pub rho: Vec<f64>,
    pub grho: Vec<DVec3>,
}

impl VolumeSet {
    pub fn build(
        surf: &SurfaceFit,
        density: &CauchyDensity,
        n_theta: usize,
        n_phi: usize,
        n_radial: usize,
    ) -> Self {
        let (dirs, wa) = sphere_grid(n_theta, n_phi);
        let (zeta, wr) = gauss_jacobi_zeta2(n_radial);
        let mut pts = Vec::with_capacity(dirs.len() * n_radial);
        let mut w = Vec::with_capacity(pts.capacity());
        for (d, wa) in dirs.iter().zip(&wa) {
            let r = surf.radius(*d);
            for (z, wr) in zeta.iter().zip(&wr) {
                pts.push(*d * (z * r));
                w.push(wr * r * r * r * wa);
            }
        }
        let rho = pts.iter().map(|p| density.rho(*p)).collect();
        let grho = pts.iter().map(|p| density.grad(*p)).collect();
        Self { pts, w, rho, grho }
    }

    pub fn mass(&self) -> f64 {
        self.w.iter().zip(&self.rho).map(|(w, r)| w * r).sum()
    }
}

/// Surface nodes at the face centres, lifted onto the harmonic surface.
#[derive(Clone, Debug)]
pub struct SurfaceSet {
    pub dirs: Vec<DVec3>,
    pub pts: Vec<DVec3>,
    pub normals: Vec<DVec3>,
    /// Solid angle of each face (quadrature weight on `S²`).
    pub solid: Vec<f64>,
}

impl SurfaceSet {
    pub fn build(surf: &SurfaceFit, dirs: Vec<DVec3>, solid: Vec<f64>) -> Self {
        let mut pts = Vec::with_capacity(dirs.len());
        let mut normals = Vec::with_capacity(dirs.len());
        for d in &dirs {
            let (r, n) = surf.normal(*d);
            pts.push(*d * r);
            normals.push(n);
        }
        Self {
            dirs,
            pts,
            normals,
            solid,
        }
    }
}

/// Geometry-dependent operators, built once per geometry and reused over Picard sweeps.
pub struct Operators {
    pub lay: VelLayout,
    pub play: PresLayout,
    pub vol: VolumeSet,
    pub surface: SurfaceSet,
    pub rows: VelRows,
    pub pres: PresRows,
    pub surf_d: DMatrix<f64>,
    pub rigid: DMatrix<f64>,
    /// Smallest singular value of the row-normalised rigid-body constraint matrix.
    pub rigid_sigma_min: f64,
    a_chol: Cholesky<f64, Dyn>,
    lift_chol: Cholesky<f64, Dyn>,
    lift_y: DMatrix<f64>,
}

impl Operators {
    pub fn build(
        surf: &SurfaceFit,
        density: &CauchyDensity,
        surface: SurfaceSet,
        prm: &StokesParams,
    ) -> Result<Self, ScfError> {
        let lay = VelLayout::new(prm.l_velocity);
        let play = PresLayout {
            l_max: prm.l_pressure,
            k_max: prm.k_pressure,
        };
        let vol = VolumeSet::build(surf, density, prm.n_theta, prm.n_phi, prm.n_radial);
        let rows = eval_velocity(surf, &lay, &vol.pts, Want::ALL);
        let pres = eval_pressure(surf, &play, &vol.pts, true);
        let surf_rows = eval_velocity(
            surf,
            &lay,
            &surface.pts,
            Want {
                v: false,
                d: true,
                lap: false,
                gd: false,
            },
        );
        let rigid = rigid_matrix(&vol, &rows, lay.n_u);
        // Row-normalised singular values.
        let mut norm = rigid.clone();
        for a in 0..6 {
            let n = norm.row(a).norm();
            if !(n.is_finite() && n > 0.0) {
                return Err(ScfError::rigid_rank("a rigid-body constraint row vanishes"));
            }
            norm.row_mut(a).scale_mut(1.0 / n);
        }
        let gram = &norm * norm.transpose();
        let eig = gram.symmetric_eigen();
        let smin = eig
            .eigenvalues
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min)
            .max(0.0)
            .sqrt();
        if smin < prm.rigid_sigma_min {
            return Err(ScfError::rigid_rank(format!(
                "rank of B_rb is deficient: smallest normalised singular value {smin:.3e}"
            )));
        }
        // Poisson stiffness.
        let np = vol.pts.len();
        let mut gs = pres.grad.clone();
        for p in 0..np {
            let s = vol.w[p].sqrt();
            for a in 0..3 {
                gs.row_mut(3 * p + a).scale_mut(s);
            }
        }
        let a_chol = Cholesky::new(gs.tr_mul(&gs))
            .ok_or_else(|| ScfError::solver("the pressure stiffness matrix is not positive definite"))?;
        // Boundary-lift least squares on the face-centre nodes.
        let nl = play.n_lift();
        let ns = surface.dirs.len();
        let mut y = DMatrix::zeros(ns, nl);
        for (s, d) in surface.dirs.iter().enumerate() {
            let v = sh_all(play.l_max, d.to_array());
            for j in 0..nl {
                y[(s, j)] = v[j];
            }
        }
        let mut yw = y.clone();
        for s in 0..ns {
            yw.row_mut(s).scale_mut(surface.solid[s].sqrt());
        }
        let mut gl = yw.tr_mul(&yw);
        let tr = gl.trace() / nl as f64;
        for j in 0..nl {
            gl[(j, j)] += 1e-12 * tr;
        }
        let lift_chol = Cholesky::new(gl)
            .ok_or_else(|| ScfError::solver("boundary-lift normal equations are singular"))?;
        Ok(Self {
            lay,
            play,
            vol,
            surface,
            rows,
            pres,
            surf_d: surf_rows.d,
            rigid,
            rigid_sigma_min: smin,
            a_chol,
            lift_chol,
            lift_y: y,
        })
    }
}

/// `B_rb` of §8.5: rows `∫ρ B·e_a`, `∫ρ (x × B)·e_a`.
fn rigid_matrix(vol: &VolumeSet, rows: &VelRows, n_u: usize) -> DMatrix<f64> {
    let mut b = DMatrix::zeros(6, n_u);
    for (p, x) in vol.pts.iter().enumerate() {
        let wr = vol.w[p] * vol.rho[p];
        for j in 0..n_u {
            let (vx, vy, vz) = (
                rows.v[(3 * p, j)],
                rows.v[(3 * p + 1, j)],
                rows.v[(3 * p + 2, j)],
            );
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
    pub mu_d: f64,
    pub inertial: f64,
}

/// `η = (c + μ_d(I_c) p̃) / √(γ̇² + γ̇_ε²)`, `I_c = d_p γ̇ / √((p̃ + p_c)/ρ_p)`.
pub fn eta_eval(m: &Material, shear: f64, p_tilde: f64, eps: f64) -> EtaEval {
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
    let a = s.cohesion + mu_d * p_tilde;
    EtaEval {
        eta: a / big,
        d_shear: p_tilde * dmu_dg / big - a * shear / (big * big * big),
        d_p: (mu_d + p_tilde * dmu_dp) / big,
        mu_d,
        inertial,
    }
}

/// Loads of one geometric step, frozen for the Picard iteration.
pub struct Loads<'a> {
    pub g_eff: &'a [DVec3],
    /// `T_tide` at the volume points, `(xx, yy, zz, xy, xz, yz)`.
    pub tide: &'a [[f64; 6]],
    /// `n · T_tide · n` at the surface nodes.
    pub tide_nn: &'a [f64],
    pub g_const: f64,
}

#[derive(Clone, Debug)]
pub struct StokesSolution {
    pub u: Vec<f64>,
    pub lambda: [f64; 6],
    pub pres_c: Vec<f64>,
    pub lift: Vec<f64>,
    /// Final viscosity at the volume points.
    pub eta: Vec<f64>,
    /// Regularised pressure `p̃` at the volume points.
    pub p_tilde: Vec<f64>,
    /// `p_Γ` at the surface nodes.
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

fn d6_dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// `D · v` from the scaled six-vector.
fn d_apply(d: &[f64], v: DVec3) -> DVec3 {
    let s = std::f64::consts::FRAC_1_SQRT_2;
    let (xy, xz, yz) = (d[3] * s, d[4] * s, d[5] * s);
    DVec3::new(
        d[0] * v.x + xy * v.y + xz * v.z,
        xy * v.x + d[1] * v.y + yz * v.z,
        xz * v.x + yz * v.y + d[2] * v.z,
    )
}

struct Fields {
    /// `γ̇` at the volume points.
    shear: Vec<f64>,
    /// `D` at the volume points (six per point, scaled).
    d: DVector<f64>,
}

fn strain_fields(ops: &Operators, u: &DVector<f64>) -> Fields {
    let d = &ops.rows.d * u;
    let np = ops.vol.pts.len();
    let shear = (0..np)
        .map(|p| {
            let dd = d6_dot(&d.as_slice()[6 * p..6 * p + 6], &d.as_slice()[6 * p..6 * p + 6]);
            (2.0 * dd).sqrt()
        })
        .collect();
    Fields { shear, d }
}

fn assemble_k(ops: &Operators, eta: &[f64]) -> DMatrix<f64> {
    let np = ops.vol.pts.len();
    let mut ds = ops.rows.d.clone();
    for p in 0..np {
        let s = (2.0 * eta[p] * ops.vol.w[p]).sqrt();
        for c in 0..6 {
            ds.row_mut(6 * p + c).scale_mut(s);
        }
    }
    ds.tr_mul(&ds)
}

fn assemble_f(ops: &Operators, ld: &Loads) -> DVector<f64> {
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
    ops.rows.v.tr_mul(&gv) - ops.rows.d.tr_mul(&tv)
}

/// Magnitude scale of the load: the same sums with absolute values, so that a load that
/// cancels exactly (a uniform sphere in its own gravity) does not make relative residuals
/// meaningless.
fn load_scale(ops: &Operators, ld: &Loads) -> f64 {
    let np = ops.vol.pts.len();
    let n_u = ops.lay.n_u;
    let mut acc = DVector::<f64>::zeros(n_u);
    for p in 0..np {
        let wg = ops.vol.w[p] * ops.vol.rho[p] * ld.g_eff[p].length();
        let t = ld.tide[p];
        let wt = ops.vol.w[p] * t.iter().map(|x| x * x).sum::<f64>().sqrt();
        for j in 0..n_u {
            let vmag = (ops.rows.v[(3 * p, j)].powi(2)
                + ops.rows.v[(3 * p + 1, j)].powi(2)
                + ops.rows.v[(3 * p + 2, j)].powi(2))
            .sqrt();
            let dmag = (0..6).map(|c| ops.rows.d[(6 * p + c, j)].powi(2)).sum::<f64>().sqrt();
            acc[j] += wg * vmag + wt * dmag;
        }
    }
    acc.norm().max(1e-300)
}

struct RangeSolve {
    chol: Cholesky<f64, Dyn>,
    y: DMatrix<f64>,
    s_chol: Cholesky<f64, Dyn>,
}

fn factor_range(ops: &Operators, k: &DMatrix<f64>) -> Result<RangeSolve, ScfError> {
    let n = k.nrows();
    let bt = ops.rigid.transpose();
    let btb = &bt * &ops.rigid;
    let kmax = (0..n).map(|i| k[(i, i)]).fold(0.0, f64::max);
    let bmax = (0..n).map(|i| btb[(i, i)]).fold(0.0, f64::max);
    let gamma = kmax / bmax.max(1e-300);
    let kg = k + btb * gamma;
    let chol = Cholesky::new(kg)
        .ok_or_else(|| ScfError::rigid_rank("K + γ B_rbᵀB_rb is not positive definite"))?;
    let y = chol.solve(&bt);
    let s = &ops.rigid * &y;
    let s_chol = Cholesky::new(s)
        .ok_or_else(|| ScfError::rigid_rank("the 6×6 rigid-body Schur complement is singular"))?;
    Ok(RangeSolve { chol, y, s_chol })
}

fn solve_range(
    ops: &Operators,
    rs: &RangeSolve,
    f: &DVector<f64>,
) -> (DVector<f64>, DVector<f64>) {
    let y0 = rs.chol.solve(f);
    let lam = rs.s_chol.solve(&(&ops.rigid * &y0));
    let u = y0 - &rs.y * &lam;
    (u, lam)
}

/// Picard solve at fixed geometry (§10.4.2).
pub fn solve(
    ops: &Operators,
    ld: &Loads,
    prm: &StokesParams,
    m: &Material,
    guess: Option<&[f64]>,
) -> Result<StokesSolution, ScfError> {
    let n_u = ops.lay.n_u;
    let np = ops.vol.pts.len();
    let ns = ops.surface.pts.len();
    let p_min = m.spec.p_min;
    let f_load = assemble_f(ops, ld);
    let f_norm = load_scale(ops, ld);
    let mut u = match guess {
        Some(g) if g.len() == n_u => DVector::from_column_slice(g),
        _ => DVector::zeros(n_u),
    };
    let grho_scale = -(8.0 * PI / 3.0) * ld.g_const * m.tide_coef;
    let mut ptil = vec![p_min; np];
    let mut gradp = vec![DVec3::ZERO; np];
    let mut p_surf_til = vec![p_min; ns];
    let mut p_gamma = vec![p_min; ns];
    let mut eta_prev: Vec<f64> = Vec::new();
    let mut total_iters = 0;
    let mut converged = false;
    let mut last = (f64::INFINITY, 0.0);
    let mut pres_c = DVector::zeros(ops.play.n_p());
    let mut lift = DVector::zeros(ops.play.n_lift());
    let mut lambda = DVector::zeros(6);
    let mut p_vals = vec![p_min; np];
    let mut eps_used = m.spec.gamma_eps;

    for level in (0..=prm.continuation).rev() {
        let eps = m.spec.gamma_eps * 10f64.powi(level as i32);
        eps_used = eps;
        converged = false;
        eta_prev.clear();
        for _ in 0..prm.picard_max {
            total_iters += 1;
            let fields = strain_fields(ops, &u);
            let lapv = &ops.rows.lap * &u;
            let gdv = &ops.rows.gd * &u;
            // (P1) pressure from the frozen (u^k, p̃^k).
            let mut rhs_f = DVector::zeros(3 * np);
            for p in 0..np {
                let d6 = &fields.d.as_slice()[6 * p..6 * p + 6];
                let ev = eta_eval(m, fields.shear[p], ptil[p], eps);
                let sh = fields.shear[p];
                let mut grad_eta = DVec3::ZERO;
                if sh > 1e-300 {
                    let mut gs = [0.0; 3];
                    for k in 0..3 {
                        let gd6 = &gdv.as_slice()[18 * p + 6 * k..18 * p + 6 * k + 6];
                        gs[k] = 2.0 * d6_dot(d6, gd6) / sh;
                    }
                    grad_eta = DVec3::from_array(gs) * ev.d_shear;
                }
                if ptil[p] > p_min + 0.0 && p_vals[p] > p_min {
                    grad_eta = grad_eta + gradp[p] * ev.d_p;
                }
                let lap = DVec3::new(lapv[3 * p], lapv[3 * p + 1], lapv[3 * p + 2]);
                let div_flow = lap * ev.eta + d_apply(d6, grad_eta) * 2.0;
                let f = ld.g_eff[p] * ops.vol.rho[p]
                    + ops.vol.grho[p] * grho_scale
                    + div_flow;
                rhs_f[3 * p] = f.x;
                rhs_f[3 * p + 1] = f.y;
                rhs_f[3 * p + 2] = f.z;
            }
            // Dirichlet data p_Γ at the surface nodes.
            let ds = &ops.surf_d * &u;
            let mut weighted = DVector::zeros(ns);
            for s in 0..ns {
                let d6 = &ds.as_slice()[6 * s..6 * s + 6];
                let shear = (2.0 * d6_dot(d6, d6)).sqrt();
                let ev = eta_eval(m, shear, p_surf_til[s], eps);
                let n = ops.surface.normals[s];
                let nn = 2.0 * ev.eta * d_apply(d6, n).dot(n);
                p_gamma[s] = prm.p_ext + ld.tide_nn[s] + nn;
                weighted[s] = ops.surface.solid[s] * p_gamma[s];
            }
            lift = ops.lift_chol.solve(&ops.lift_y.tr_mul(&weighted));
            let grad_pb = &ops.pres.lift_grad * &lift;
            let mut wf = DVector::zeros(3 * np);
            for p in 0..np {
                for a in 0..3 {
                    wf[3 * p + a] = ops.vol.w[p] * (rhs_f[3 * p + a] - grad_pb[3 * p + a]);
                }
            }
            pres_c = ops.a_chol.solve(&ops.pres.grad.tr_mul(&wf));
            let pv = &ops.pres.val * &pres_c + &ops.pres.lift_val * &lift;
            let gp = &ops.pres.grad * &pres_c + grad_pb;
            let mut new_ptil = vec![0.0; np];
            for p in 0..np {
                p_vals[p] = pv[p];
                new_ptil[p] = pv[p].max(p_min);
                gradp[p] = DVec3::new(gp[3 * p], gp[3 * p + 1], gp[3 * p + 2]);
            }
            for s in 0..ns {
                p_surf_til[s] = p_gamma[s].max(p_min);
            }
            // (P2) viscosity from (γ̇(u^k), p̃^{k+1}).
            let mut eta: Vec<f64> = (0..np)
                .map(|p| eta_eval(m, fields.shear[p], new_ptil[p], eps).eta)
                .collect();
            if !eta_prev.is_empty() {
                for p in 0..np {
                    eta[p] = prm.theta_eta * eta[p] + (1.0 - prm.theta_eta) * eta_prev[p];
                }
            }
            // (P3) velocity.
            let k = assemble_k(ops, &eta);
            let rs = factor_range(ops, &k)?;
            let (un, lam) = solve_range(ops, &rs, &f_load);
            // (P4) criterion (a).
            let du = (&un - &u).norm() / un.norm().max(1e-300);
            let deta = if eta_prev.is_empty() {
                f64::INFINITY
            } else {
                let num: f64 = (0..np)
                    .map(|p| ops.vol.w[p] * (eta[p] - eta_prev[p]).powi(2))
                    .sum();
                let den: f64 = (0..np).map(|p| ops.vol.w[p] * eta[p].powi(2)).sum();
                (num / den.max(1e-300)).sqrt()
            };
            let du_floor = un.norm() <= 1e-300;
            u = un;
            lambda = lam;
            ptil = new_ptil;
            eta_prev = eta;
            last = (du.max(deta), deta);
            if (du <= prm.tol_picard || du_floor) && deta <= prm.tol_picard {
                converged = true;
                break;
            }
        }
        if !converged && level == 0 {
            return Err(ScfError::rheology_not_converged(format!(
                "Picard iteration: relative change {:.3e} after {} sweeps (tolerance {:.1e})",
                last.0, total_iters, prm.tol_picard
            )));
        }
    }
    debug_assert!(converged);

    // Criterion (b) on the final iterate with the unrelaxed viscosity.
    let fields = strain_fields(ops, &u);
    let eta_final: Vec<f64> = (0..np)
        .map(|p| eta_eval(m, fields.shear[p], ptil[p], eps_used).eta)
        .collect();
    let k = assemble_k(ops, &eta_final);
    let resid = &k * &u + ops.rigid.transpose() * &lambda - &f_load;
    let eps_momentum = resid.norm() / f_norm;
    let eps_lambda = (ops.rigid.transpose() * &lambda).norm() / f_norm;
    if eps_momentum > prm.tol_momentum {
        return Err(ScfError::rheology_not_converged(format!(
            "weak momentum residual {eps_momentum:.3e} exceeds {:.1e}",
            prm.tol_momentum
        )));
    }
    if eps_lambda > prm.tol_momentum.max(1e-3) {
        return Err(ScfError::rigid_rank(format!(
            "rigid-body multiplier {eps_lambda:.3e}: the load is incompatible with the rigid modes"
        )));
    }
    // Strong-form rotational residual f - ∇p (diagnostic of §8.7).
    let lapv = &ops.rows.lap * &u;
    let gdv = &ops.rows.gd * &u;
    let (mut num, mut den) = (0.0, 0.0);
    for p in 0..np {
        let d6 = &fields.d.as_slice()[6 * p..6 * p + 6];
        let ev = eta_eval(m, fields.shear[p], ptil[p], eps_used);
        let sh = fields.shear[p];
        let mut grad_eta = DVec3::ZERO;
        if sh > 1e-300 {
            let mut gs = [0.0; 3];
            for kk in 0..3 {
                gs[kk] = 2.0 * d6_dot(d6, &gdv.as_slice()[18 * p + 6 * kk..18 * p + 6 * kk + 6]) / sh;
            }
            grad_eta = DVec3::from_array(gs) * ev.d_shear;
        }
        if p_vals[p] > p_min {
            grad_eta = grad_eta + gradp[p] * ev.d_p;
        }
        let lap = DVec3::new(lapv[3 * p], lapv[3 * p + 1], lapv[3 * p + 2]);
        let f = ld.g_eff[p] * ops.vol.rho[p]
            + ops.vol.grho[p] * grho_scale
            + lap * ev.eta
            + d_apply(d6, grad_eta) * 2.0;
        let r = f - gradp[p];
        num += ops.vol.w[p] * r.length_squared();
        den += ops.vol.w[p] * f.length_squared();
    }
    let eps_rot = (num / den.max(1e-300)).sqrt();
    let wsum: f64 = ops.vol.w.iter().sum();
    let tensile = (0..np)
        .filter(|&p| p_vals[p] < 0.0)
        .map(|p| ops.vol.w[p])
        .sum::<f64>()
        / wsum;
    let eta_lo = eta_final.iter().copied().fold(f64::INFINITY, f64::min);
    let eta_hi = eta_final.iter().copied().fold(0.0, f64::max);
    Ok(StokesSolution {
        u: u.iter().copied().collect(),
        lambda: std::array::from_fn(|i| lambda[i]),
        pres_c: pres_c.iter().copied().collect(),
        lift: lift.iter().copied().collect(),
        eta: eta_final,
        p_tilde: ptil,
        p_surface: p_gamma,
        picard_iterations: total_iters,
        eps_picard: last.0,
        eps_momentum,
        eps_lambda,
        eps_rot,
        eta_lo,
        eta_hi,
        tensile_fraction: tensile,
        inverse_ekman: 0.0,
        gamma_eps: eps_used,
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::rheo_friction::{MaterialSpec, SolverSpec};

    pub fn material() -> Material {
        Material::new(
            MaterialSpec {
                nu0: 0.6,
                z0: 6.0,
                d_p: 0.05,
                rho_p: 2500.0,
                mu_s: 0.38,
                mu_2: 0.64,
                i_0: 0.279,
                cohesion: 25.0,
                c_c: 1.0,
                f_h: 1.0e-4,
                c_t: 1.0,
                gamma_eps: 1.0e-9,
                p_min: 1.0e-6,
                h_s: 5.0,
                eps_overhang: 0.01,
                mu_proj_mode: "dynamic".into(),
            },
            SolverSpec {
                max_iter: 100,
                tol: 1.0e-12,
            },
        )
        .unwrap()
    }

    pub fn params() -> StokesParams {
        StokesParams {
            l_surface: 4,
            l_velocity: 3,
            l_pressure: 4,
            k_pressure: 4,
            n_theta: 10,
            n_phi: 20,
            n_radial: 5,
            p_ext: 0.0,
            picard_max: 60,
            tol_picard: 1e-8,
            tol_momentum: 1e-6,
            theta_eta: 1.0,
            continuation: 2,
            rigid_sigma_min: 1e-6,
        }
    }

    #[test]
    fn uniform_sphere_is_hydrostatic_and_at_rest() {
        let (r0, rho, g) = (1000.0, 2000.0, 6.6743e-11);
        let k = 4.0 * PI / 3.0 * g * rho;
        let surf = SurfaceFit {
            l_max: 0,
            coeff: vec![r0 * (4.0 * PI).sqrt()],
        };
        let density = CauchyDensity::uniform(rho).unwrap();
        let prm = params();
        // A closed set of surface nodes: the product grid.
        let (dirs, solid) = sphere_grid(8, 16);
        let surface = SurfaceSet::build(&surf, dirs, solid);
        let ops = Operators::build(&surf, &density, surface, &prm).unwrap();
        let g_eff: Vec<DVec3> = ops.vol.pts.iter().map(|x| *x * -k).collect();
        let tide = vec![[0.0; 6]; ops.vol.pts.len()];
        let tide_nn = vec![0.0; ops.surface.pts.len()];
        let ld = Loads {
            g_eff: &g_eff,
            tide: &tide,
            tide_nn: &tide_nn,
            g_const: g,
        };
        let sol = solve(&ops, &ld, &prm, &material(), None).unwrap();
        let umax = sol.u.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        assert!(umax < 1e-12, "velocity {umax}");
        // p(r) = rho k (R^2 - r^2) / 2
        for (p, x) in ops.vol.pts.iter().enumerate().step_by(37) {
            let exact = 0.5 * rho * k * (r0 * r0 - x.length_squared());
            let got = sol.p_tilde[p];
            assert!(
                (got - exact).abs() < 2e-3 * exact.abs().max(1.0) + 1e-3,
                "p at r={}: {got} vs {exact}",
                x.length()
            );
        }
        assert!(sol.eps_rot < 1e-2, "rot {}", sol.eps_rot);
    }
}
