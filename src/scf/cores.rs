//! Lagrangian transport and adaptive refit of the Cauchy density cores (MS §9.10).
//!
//! * Level 0 (`predict`): exact 2-jet transport of every core along the masked
//!   slip velocity, `c' = Φ(c)`, `S' = F⁻ᵀ S F⁻¹`, `F = exp(τ dev ∇v)`, midpoint rule.
//! * Level 1 (`close_mass`): uniform scaling of the core weights to the target mass.
//! * Level 2 (`refit`): Gauss–Newton fit of `{w, c, S = L Lᵀ}` to the semi-Lagrangian
//!   transported density on volume quadrature points, with the mass as a linear
//!   constraint, triggered by the misfit indicator `e_ρ` or by `|ξ|`.

use nalgebra::{DMatrix, DVector, Matrix3};
use rayon::prelude::*;

use crate::common::density::{CauchyDensity, CauchyKernel};
use crate::common::math::{DMat3, DVec3, mat_sym};

use crate::common::error::SimError;

/// Configuration (`[cores]` of the project).
#[derive(Clone, Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CoreParams {
    /// `"lagrangian_cores"` (hypothesis A-ρL, §9.10) or `"frozen"` (hypothesis A-ρ,
    /// regression only).
    pub density_mode: String,
    /// `"anisotropic"` (full metric) or `"isotropic"` (`σ'² = det(S')^{1/3}`).
    pub core_shape: String,
    /// Level-2 trigger on the relative mass closure `|ξ|`.
    pub eps_xi: f64,
    /// Level-2 trigger and target of the relative misfit `e_ρ`.
    pub eps_rho: f64,
    /// Gauss–Newton iterations at most.
    pub gn_max: usize,
}

impl CoreParams {
    pub fn validate(&self) -> Result<(), SimError> {
        let bad = |m: &str| Err(SimError::model_not_closed(m.to_string()));
        if !matches!(self.density_mode.as_str(), "lagrangian_cores" | "frozen") {
            return bad("cores.density_mode must be \"lagrangian_cores\" or \"frozen\"");
        }
        if !matches!(self.core_shape.as_str(), "anisotropic" | "isotropic") {
            return bad("cores.core_shape must be \"anisotropic\" or \"isotropic\"");
        }
        if !(self.eps_xi > 0.0 && self.eps_rho > 0.0 && self.gn_max > 0) {
            return bad(
                "cores.eps_xi and cores.eps_rho must be positive and cores.gn_max at least 1",
            );
        }
        Ok(())
    }

    pub fn lagrangian(&self) -> bool {
        self.density_mode == "lagrangian_cores"
    }
}

/// Reported quantities of §9.10.7.
#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct CoreReport {
    pub anchored: usize,
    pub max_shift: f64,
    pub max_condition: f64,
    pub xi: f64,
    pub e_rho: f64,
    pub gn_iterations: usize,
    /// `τ max |∇v|` over the moving cores.
    pub strain_step: f64,
    /// A-Ξ: the two-scalar closure `(s_b, s_c)` and the spawned cores (§11.4).
    pub s_bulk: f64,
    pub s_core: f64,
    pub spawned: usize,
}

/// The dilatational density ratio `J_ν` at a batch of points (A-Ξ, Proposition 11.5).
pub type Ratio<'a> = dyn Fn(&[DVec3]) -> Vec<f64> + 'a;

/// The masked advection velocity `v_adv = χ_Y v_sl` at a batch of points.
pub type Velocity<'a> = dyn Fn(&[DVec3]) -> Vec<DVec3> + 'a;

fn mat(m: DMat3) -> Matrix3<f64> {
    Matrix3::from_column_slice(&m.to_cols_array())
}

fn dmat(m: Matrix3<f64>) -> DMat3 {
    DMat3::from_cols_slice(m.as_slice())
}

/// `max ‖dev ∇v_adv(c_k)‖_F` over the moving cores at their current centres: the rate
/// of the core step bound `Δt_core` (§10.4.1). Zero when every core is anchored.
pub fn strain_rate(
    kernels: &[CauchyKernel],
    v_adv: &Velocity<'_>,
    inside: &dyn Fn(DVec3) -> bool,
    h: f64,
    params: &CoreParams,
) -> f64 {
    if kernels.is_empty() || !params.lagrangian() {
        return 0.0;
    }
    let centres: Vec<DVec3> = kernels.iter().map(|k| k.center).collect();
    let v0 = v_adv(&centres);
    let mut probe = Vec::with_capacity(centres.len() * 6);
    for c in &centres {
        for a in DVec3::AXES {
            probe.push(*c + a * h);
            probe.push(*c - a * h);
        }
    }
    let vp = v_adv(&probe);
    let mut rate = 0.0_f64;
    for (k, kern) in kernels.iter().enumerate() {
        if !(inside(kern.center) && v0[k] != DVec3::ZERO) {
            continue;
        }
        let row = &vp[6 * k..6 * k + 6];
        let grad = DMat3::from_cols(
            (row[0] - row[1]) / (2.0 * h),
            (row[2] - row[3]) / (2.0 * h),
            (row[4] - row[5]) / (2.0 * h),
        );
        let trace = grad.x_axis.x + grad.y_axis.y + grad.z_axis.z;
        rate = rate.max(mat(grad - DMat3::IDENTITY * (trace / 3.0)).norm());
    }
    rate
}

/// Level 0 (§9.10.3). `inside(c)` tells a core centre inside the moving material;
/// a core outside it, or where the velocity is masked, is anchored.
pub fn predict(
    kernels: &[CauchyKernel],
    tau: f64,
    v_adv: &Velocity<'_>,
    inside: &dyn Fn(DVec3) -> bool,
    h: f64,
    params: &CoreParams,
    ratio: Option<&Ratio<'_>>,
    report: &mut CoreReport,
) -> Vec<CauchyKernel> {
    if kernels.is_empty() || tau == 0.0 || !params.lagrangian() {
        return kernels.to_vec();
    }
    let centres: Vec<DVec3> = kernels.iter().map(|k| k.center).collect();
    let v0 = v_adv(&centres);
    let mid: Vec<DVec3> = centres
        .iter()
        .zip(&v0)
        .map(|(c, v)| *c + *v * (0.5 * tau))
        .collect();
    // v at the midpoint and its six central-difference neighbours.
    let mut probe = Vec::with_capacity(mid.len() * 7);
    for m in &mid {
        probe.push(*m);
        for a in DVec3::AXES {
            probe.push(*m + a * h);
            probe.push(*m - a * h);
        }
    }
    let vp = v_adv(&probe);
    // A-Ξ (§11.4): J_ν at the midpoints; the physical dilatation −ln J / τ is added back
    // to the deviatoric gradient, and the weight follows the density, w' = J w.
    let jac = ratio.map_or_else(|| vec![1.0; mid.len()], |r| r(&mid));
    let mut out = Vec::with_capacity(kernels.len());
    for (k, kern) in kernels.iter().enumerate() {
        let moving = inside(kern.center) && v0[k] != DVec3::ZERO;
        if !moving {
            report.anchored += 1;
            out.push(*kern);
            continue;
        }
        let row = &vp[7 * k..7 * k + 7];
        let grad = DMat3::from_cols(
            (row[1] - row[2]) / (2.0 * h),
            (row[3] - row[4]) / (2.0 * h),
            (row[5] - row[6]) / (2.0 * h),
        );
        // Columns are ∂v/∂x_j, so `grad` is ∇v with (i, j) = ∂v_i/∂x_j.
        let trace = grad.x_axis.x + grad.y_axis.y + grad.z_axis.z;
        let l = grad - DMat3::IDENTITY * (trace / 3.0);
        report.strain_step = report.strain_step.max(tau * mat(l).norm());
        let l = l + DMat3::IDENTITY * (-jac[k].ln() / (3.0 * tau));
        let f = dmat((mat(l) * tau).exp());
        let finv = f.inverse();
        let c_new = kern.center + row[0] * tau;
        let s = finv.transpose() * kern.metric_mat() * finv;
        let metric = if params.core_shape == "isotropic" {
            let s2 = s.determinant().cbrt();
            [s2, s2, s2, 0.0, 0.0, 0.0]
        } else {
            mat_sym(s)
        };
        report.max_shift = report.max_shift.max((c_new - kern.center).length());
        match CauchyKernel::try_anisotropic(c_new, kern.weight * jac[k], metric, kern.alpha) {
            Ok(n) => out.push(n),
            Err(_) => {
                report.anchored += 1;
                out.push(*kern);
            }
        }
    }
    report.max_condition = out.iter().map(condition).fold(0.0, f64::max);
    out
}

fn condition(k: &CauchyKernel) -> f64 {
    let e = mat(k.metric_mat()).symmetric_eigenvalues();
    e.max() / e.min().max(1e-300)
}

/// Level 1 (§9.10.4): scale the core weights so the total mass is `target`.
/// `mass` is the mass of the current density on the new body and `bulk_mass`
/// its constant part. Returns `ξ` relative to the core mass.
pub fn close_mass(kernels: &mut [CauchyKernel], mass: f64, bulk_mass: f64, target: f64) -> f64 {
    let core = mass - bulk_mass;
    if kernels.is_empty() || core.abs() < 1e-12 * mass.abs() {
        return 0.0;
    }
    let xi = (target - bulk_mass) / core - 1.0;
    for k in kernels.iter_mut() {
        k.weight *= 1.0 + xi;
    }
    xi
}

/// Level 1 under A-Ξ (§11.4): the mass-constrained two-scalar fit
/// `min Σ ω (s_b ρ_b + s_c ρ_c − ρ̃)²` subject to `s_b ρ_b V + s_c M_c = M`, a 3 × 3 KKT
/// system. Falls back to one scalar when a part is absent or the two are dependent.
/// Returns `(s_b, s_c)`; the mass is exact in every case.
pub fn close_two(
    bulk: f64,
    kernels: &[CauchyKernel],
    quad: &Quadrature,
    target: &[f64],
    volume: f64,
    core_mass: f64,
    mass: f64,
) -> (f64, f64) {
    let bulk_mass = bulk * volume;
    let has_core = !kernels.is_empty() && core_mass.abs() > 1e-12 * mass.abs();
    if bulk_mass <= 0.0 {
        return (1.0, if has_core { mass / core_mass } else { 1.0 });
    }
    if !has_core {
        return ((mass - core_mass) / bulk_mass, 1.0);
    }
    let cores = CauchyDensity::try_new(0.0, kernels.to_vec()).expect("validated cores");
    let (mut a11, mut a12, mut a22, mut b1, mut b2) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for ((x, w), t) in quad.pts.iter().zip(&quad.w).zip(target) {
        let c = cores.rho(*x);
        a11 += w * bulk * bulk;
        a12 += w * bulk * c;
        a22 += w * c * c;
        b1 += w * bulk * t;
        b2 += w * c * t;
    }
    // [[a11 a12 V_b], [a12 a22 M_c], [V_b M_c 0]] [s_b s_c λ] = [b1 b2 M].
    let k = nalgebra::Matrix3::new(a11, a12, bulk_mass, a12, a22, core_mass, bulk_mass, core_mass, 0.0);
    let dependent = (a11 * a22 - a12 * a12).abs() <= 1e-10 * a11 * a22;
    match k.try_inverse().filter(|_| !dependent) {
        Some(inv) => {
            let s = inv * nalgebra::Vector3::new(b1, b2, mass);
            (s[0], s[1])
        }
        None => (1.0, (mass - bulk_mass) / core_mass),
    }
}

/// Level 3 spawn under A-Ξ (§11.4): a signed core at the largest residual cluster of
/// `ρ̃ − ρ_K` on the quadrature, with exponent `alpha`, metric `scale (C + h² I)⁻¹` from the
/// `|r|`-weighted covariance `C` of the cluster, and the residual of its node nearest the
/// centre as weight. `None` when there is no residual.
pub fn spawn(density: &CauchyDensity, quad: &Quadrature, target: &[f64], alpha: f64, scale: f64) -> Option<CauchyKernel> {
    let r: Vec<f64> = quad.pts.iter().zip(target).map(|(x, t)| t - density.rho(*x)).collect();
    let (top, peak) = r.iter().enumerate().fold((0, 0.0f64), |a, (i, v)| if v.abs() > a.1 { (i, v.abs()) } else { a });
    if !(peak > 0.0) {
        return None;
    }
    // The connected cluster of {|r| ≥ peak/2} around the peak, linked within two node spacings.
    let spacing = quad.w.iter().map(|w| w.cbrt()).sum::<f64>() / quad.w.len() as f64;
    let grid = super::surface::PointGrid::build(&quad.pts);
    let sign = r[top].signum();
    let mut seen = vec![false; r.len()];
    let mut stack = vec![top];
    let mut cluster = Vec::new();
    seen[top] = true;
    while let Some(i) = stack.pop() {
        cluster.push(i);
        for (_, j) in grid.within(quad.pts[i], 2.0 * spacing) {
            if !seen[j] && r[j] * sign >= 0.5 * peak {
                seen[j] = true;
                stack.push(j);
            }
        }
    }
    let wsum: f64 = cluster.iter().map(|i| quad.w[*i] * r[*i].abs()).sum();
    let c = cluster.iter().fold(DVec3::ZERO, |s, i| s + quad.pts[*i] * (quad.w[*i] * r[*i].abs())) / wsum;
    let mut cov = Matrix3::identity() * (spacing * spacing);
    for i in &cluster {
        let d = quad.pts[*i] - c;
        let d = nalgebra::Vector3::new(d.x, d.y, d.z);
        cov += d * d.transpose() * (quad.w[*i] * r[*i].abs() / wsum);
    }
    let metric = cov.try_inverse()? * scale;
    let near = cluster
        .iter()
        .min_by(|a, b| (quad.pts[**a] - c).length().total_cmp(&(quad.pts[**b] - c).length()))?;
    CauchyKernel::try_anisotropic(c, r[*near], mat_sym(dmat(metric)), alpha).ok()
}

/// Volume quadrature for Level 2: points, weights.
pub struct Quadrature {
    pub pts: Vec<DVec3>,
    pub w: Vec<f64>,
}

/// Number of parameters per core: weight, centre, Cholesky factor (log diagonal).
const NP: usize = 10;

fn pack(k: &CauchyKernel) -> [f64; NP] {
    let l = mat(k.metric_mat())
        .cholesky()
        .map(|c| c.l())
        .unwrap_or_else(Matrix3::identity);
    [
        k.weight,
        k.center.x,
        k.center.y,
        k.center.z,
        l[(0, 0)].ln(),
        l[(1, 1)].ln(),
        l[(2, 2)].ln(),
        l[(1, 0)],
        l[(2, 0)],
        l[(2, 1)],
    ]
}

fn factor(p: &[f64]) -> Matrix3<f64> {
    Matrix3::new(
        p[4].exp(),
        0.0,
        0.0,
        p[7],
        p[5].exp(),
        0.0,
        p[8],
        p[9],
        p[6].exp(),
    )
}

fn unpack(p: &[f64], alpha: f64) -> Option<CauchyKernel> {
    let l = factor(p);
    let s = dmat(l * l.transpose());
    CauchyKernel::try_anisotropic(DVec3::new(p[1], p[2], p[3]), p[0], mat_sym(s), alpha).ok()
}

/// ρ_k and ∂ρ_k/∂θ_k at `x`.
fn kernel_jet(p: &[f64], alpha: f64, x: DVec3) -> (f64, [f64; NP]) {
    let l = factor(p);
    let s = l * l.transpose();
    let d = nalgebra::Vector3::new(x.x - p[1], x.y - p[2], x.z - p[3]);
    let sd = s * d;
    let q = 1.0 + d.dot(&sd);
    let base = q.powf(-alpha);
    let w = p[0];
    let rho = w * base;
    let mut g = [0.0; NP];
    g[0] = base;
    let dq = 2.0 * alpha * w * base / q;
    g[1] = dq * sd.x;
    g[2] = dq * sd.y;
    g[3] = dq * sd.z;
    // ∂ρ/∂S = −α w q^{−α−1} d dᵀ, ∂ρ/∂L = 2 (∂ρ/∂S) L.
    let gs = d * d.transpose() * (-alpha * w * base / q);
    let gl = gs * l * 2.0;
    g[4] = gl[(0, 0)] * l[(0, 0)];
    g[5] = gl[(1, 1)] * l[(1, 1)];
    g[6] = gl[(2, 2)] * l[(2, 2)];
    g[7] = gl[(1, 0)];
    g[8] = gl[(2, 0)];
    g[9] = gl[(2, 1)];
    (rho, g)
}

/// Quadrature sums run over this many contiguous blocks, added in order: the result
/// does not depend on how rayon splits the work, so runs repeat bit for bit.
const SUM_BLOCKS: usize = 64;

fn blocks(len: usize) -> impl IndexedParallelIterator<Item = std::ops::Range<usize>> {
    let size = len.div_ceil(SUM_BLOCKS).max(1);
    (0..len.div_ceil(size)).into_par_iter().map(move |b| b * size..((b + 1) * size).min(len))
}

/// `e_ρ` of §9.10.5 between a density and target values on a quadrature.
pub fn misfit(density: &CauchyDensity, quad: &Quadrature, target: &[f64]) -> f64 {
    let parts: Vec<(f64, f64)> = blocks(quad.pts.len())
        .map(|range| {
            range.fold((0.0, 0.0), |(num, den), i| {
                let (w, t) = (quad.w[i], target[i]);
                let r = density.rho(quad.pts[i]) - t;
                (num + w * r * r, den + w * t * t)
            })
        })
        .collect();
    let (num, den) = parts.iter().fold((0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1));
    (num / den.max(1e-300)).sqrt()
}

/// Level 2 (§9.10.5): Gauss–Newton with Levenberg–Marquardt damping and the
/// quadrature mass held at its current value. Returns the refitted cores.
pub fn refit(
    bulk: f64,
    kernels: &[CauchyKernel],
    quad: &Quadrature,
    target: &[f64],
    params: &CoreParams,
    report: &mut CoreReport,
) -> Vec<CauchyKernel> {
    let nk = kernels.len();
    if nk == 0 {
        return Vec::new();
    }
    let n = NP * nk;
    let alphas: Vec<f64> = kernels.iter().map(|k| k.alpha).collect();
    let mut theta: Vec<f64> = kernels.iter().flat_map(pack).collect();
    // Objective, normal matrix, gradient, mass and mass gradient over the points.
    let assemble = |theta: &[f64]| {
        let zero = || {
            (
                0.0,
                DMatrix::<f64>::zeros(n, n),
                DVector::<f64>::zeros(n),
                0.0,
                DVector::<f64>::zeros(n),
            )
        };
        let parts: Vec<_> = blocks(quad.pts.len())
            .map(|range| {
                let (mut f, mut h, mut g, mut m, mut gm) = zero();
                let mut j = vec![0.0; n];
                for i in range {
                    let (x, w, t) = (quad.pts[i], quad.w[i], target[i]);
                    let mut rho = bulk;
                    for k in 0..nk {
                        let (r, d) = kernel_jet(&theta[NP * k..NP * (k + 1)], alphas[k], x);
                        rho += r;
                        j[NP * k..NP * (k + 1)].copy_from_slice(&d);
                    }
                    let res = rho - t;
                    f += 0.5 * w * res * res;
                    m += w * rho;
                    for a in 0..n {
                        if j[a] == 0.0 {
                            continue;
                        }
                        g[a] += w * res * j[a];
                        gm[a] += w * j[a];
                        for b in a..n {
                            h[(a, b)] += w * j[a] * j[b];
                        }
                    }
                }
                (f, h, g, m, gm)
            })
            .collect();
        parts.into_iter().fold(zero(), |a, b| {
            (a.0 + b.0, a.1 + b.1, a.2 + b.2, a.3 + b.3, a.4 + b.4)
        })
    };
    let (mut f, mut h, mut g, m0, mut gm) = assemble(&theta);
    let mut mass = m0;
    let mut mu = 1e-3;
    let target_norm: f64 = quad
        .w
        .iter()
        .zip(target)
        .map(|(w, t)| w * t * t)
        .sum::<f64>()
        .max(1e-300);
    for _ in 0..params.gn_max {
        if (2.0 * f / target_norm).sqrt() <= 0.5 * params.eps_rho {
            break;
        }
        for a in 0..n {
            for b in 0..a {
                h[(a, b)] = h[(b, a)];
            }
        }
        // KKT system [H + μ diag H, gm; gmᵀ, 0] [δ; λ] = [−g; m0 − m].
        let mut kkt = DMatrix::<f64>::zeros(n + 1, n + 1);
        let mut rhs = DVector::<f64>::zeros(n + 1);
        for a in 0..n {
            for b in 0..n {
                kkt[(a, b)] = h[(a, b)];
            }
            kkt[(a, a)] += mu * h[(a, a)].max(1e-30);
            kkt[(a, n)] = gm[a];
            kkt[(n, a)] = gm[a];
            rhs[a] = -g[a];
        }
        rhs[n] = m0 - mass;
        let Some(step) = kkt.lu().solve(&rhs) else {
            mu *= 10.0;
            continue;
        };
        let trial: Vec<f64> = theta.iter().zip(step.iter()).map(|(t, s)| t + s).collect();
        let ok = (0..nk).all(|k| unpack(&trial[NP * k..NP * (k + 1)], alphas[k]).is_some());
        report.gn_iterations += 1;
        if ok {
            let (f2, h2, g2, m2, gm2) = assemble(&trial);
            if f2 < f {
                let gain = (f - f2) / f.max(1e-300);
                theta = trial;
                (f, h, g, mass, gm) = (f2, h2, g2, m2, gm2);
                mu = (mu * 0.3).max(1e-9);
                if gain < 1e-3 {
                    break;
                }
                continue;
            }
        }
        mu *= 10.0;
        if mu > 1e8 {
            break;
        }
    }
    report.e_rho = (2.0 * f / target_norm).sqrt();
    (0..nk)
        .map(|k| unpack(&theta[NP * k..NP * (k + 1)], alphas[k]).unwrap_or(kernels[k]))
        .collect()
}

/// The transported target `ρ̃_q = ρ_K(X_d(x_q); θ⁽ⁿ⁾)` with the midpoint departure
/// point `X_d(x) = x − τ v(x − τ/2 v(x))` (§9.10.5).
pub fn departure_density(
    old: &CauchyDensity,
    quad: &Quadrature,
    tau: f64,
    v_adv: &Velocity<'_>,
) -> Vec<f64> {
    let v1 = v_adv(&quad.pts);
    let mid: Vec<DVec3> = quad
        .pts
        .iter()
        .zip(&v1)
        .map(|(x, v)| *x - *v * (0.5 * tau))
        .collect();
    let v2 = v_adv(&mid);
    quad.pts
        .par_iter()
        .zip(v2.par_iter())
        .map(|(x, v)| old.rho(*x - *v * tau))
        .collect()
}
