//! Hypothesis A-O (§10.3.6): the tidal decay of a synchronous satellite coupled to its mass,
//! the necks of its Hill region through which it loses mass, and the failure orbit of a
//! frozen body (§10.0.12.3).
//!
//! The orbit is the `orbit` spin schedule of §10.3.1 with its rate scale tied to the mass:
//! `d(a^{13/2})/dt = −κ m`, integrated exactly over each step at the mass of the step.

use nalgebra::{Matrix3, SymmetricEigen};
use serde::{Deserialize, Serialize};

use crate::common::error::SimError;
use crate::common::math::{DMat3, DVec3};

use super::body::Body;
use super::forward::{Forward, Loading};
use super::mechanics::{GeometryCache, LoadSet, geometry};
use super::params::Project;
use super::spin::SpinSchedule;
use super::cycle::{Cycle, CycleParams, Libration, chi_of, eccentricity_coefficient, love_number};
use super::stokes::Operators;
use super::stokes::elastic::{CycleBracket, ElasticCycle, cycle_bracket, elastic_cycle};
use super::stokes::limit::{self, Bracket, LimitData};

/// The `[orbit]` section of a project.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OrbitParams {
    /// Radius of the primary (m): the run stops when the semi-major axis reaches it.
    pub primary_radius: f64,
    /// Steps between certified limit-load brackets of the run (§10.0.12.3); 0 for none.
    #[serde(default)]
    pub limit_every: usize,
}

impl OrbitParams {
    pub fn validate(&self) -> Result<(), SimError> {
        if self.primary_radius.is_finite() && self.primary_radius > 0.0 {
            Ok(())
        } else {
            Err(SimError::model_not_closed("orbit.primary_radius must be finite and positive"))
        }
    }
}

/// `(a_n, M_n)`, the mass `m₀` at which the configured rate `ȧ₀` holds, and the mass lost
/// so far; `M_n + escaped = M₀` exactly. `e` is the eccentricity of A-Λ (zero without it).
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct OrbitState {
    pub a: f64,
    pub mass: f64,
    pub m0: f64,
    pub escaped: f64,
    #[serde(default)]
    pub e: f64,
}

/// What A-O reports per step (inactive without it).
#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct OrbitReport {
    pub active: bool,
    /// Semi-major axis (m) and mean motion (rad/s) of the step.
    pub a: f64,
    pub n: f64,
    /// Ledger mass at the start of the step, and the mass lost in it by route (kg).
    pub mass: f64,
    pub escaped_runout: f64,
    pub escaped_relief: f64,
    pub escaped_cap: f64,
    pub escaped_total: f64,
    pub r_hill: f64,
    /// Largest distance of `Γ` beyond a neck plane over `r_H` (positive: overflow).
    pub lobe_margin: f64,
    /// Bound on `|Δa|/a` from the neglected recoil (Remark 10.3.6.4).
    pub recoil_bound: f64,
    /// Certified bracket of `L*` with the strength `c + μ₂ p̃` (zero when not evaluated).
    pub limit_lo: f64,
    pub limit_hi: f64,
    /// Mountain-pass critical energy of Definition 10.3.6.2 and its saddle bound
    /// `min_k{−W(L_k)}`, sampled on the neck planes each step.
    pub e_crit: f64,
    pub e_crit_saddle: f64,
}

impl OrbitReport {
    pub fn inactive(&self) -> bool {
        !self.active
    }
}

/// The decay law `d(a^{13/2})/dt = −κ m` of Hypothesis A-O, item 1.
#[derive(Clone, Copy, Debug)]
pub struct OrbitLaw {
    gm: f64,
    kappa: f64,
    floor: f64,
}

impl OrbitLaw {
    /// The law of the `orbit` schedule `spin` for a satellite whose rate `ȧ₀` held at `m0`.
    pub fn new(params: &OrbitParams, spin: &SpinSchedule, m0: f64) -> Result<Self, SimError> {
        let SpinSchedule::Orbit { gm, a0, adot0, .. } = *spin else {
            return Err(SimError::model_not_closed("[orbit] (A-O) needs an orbit spin schedule"));
        };
        let kappa = -6.5 * a0.powf(5.5) * adot0 / m0;
        if !(kappa.is_finite() && kappa > 0.0) {
            return Err(SimError::model_not_closed(
                "A-O needs a decaying orbit (ȧ₀ < 0) and a positive mass",
            ));
        }
        Ok(Self { gm, kappa, floor: params.primary_radius })
    }

    /// The state at time `t` of a satellite of mass `mass` that has lost nothing yet.
    pub fn start(spin: &SpinSchedule, t: f64, mass: f64) -> OrbitState {
        OrbitState { a: spin.semi_major_axis(t), mass, m0: mass, escaped: 0.0, e: 0.0 }
    }

    pub fn mean_motion(&self, a: f64) -> f64 {
        (self.gm / a.powi(3)).sqrt()
    }

    pub fn load(&self, a: f64) -> Loading {
        let n2 = self.gm / a.powi(3);
        Loading { omega: n2.sqrt(), tide: n2 }
    }

    /// `ȧ = −κ m / ((13/2) a^{11/2})`.
    pub fn adot(&self, a: f64, mass: f64) -> f64 {
        -self.kappa * mass / (6.5 * a.powf(5.5))
    }

    /// `ṅ = −(3/2)(n/a) ȧ`.
    pub fn ndot(&self, a: f64, mass: f64) -> f64 {
        -1.5 * self.mean_motion(a) / a * self.adot(a, mass)
    }

    pub fn floor(&self) -> f64 {
        self.floor
    }

    /// `β_n` of Proposition 10.3.7.1 for a satellite of mass `mass` and radius `r_s` whose
    /// own tide has `k2_over_q`, about a primary of mass `m_p`.
    pub fn beta(&self, m_p: f64, k2_over_q: f64, r_s: f64, mass: f64) -> f64 {
        68.25 * k2_over_q * m_p * r_s.powi(5) * self.gm.sqrt() / (self.kappa * mass * mass)
    }

    /// The time in which the orbit of `s` reaches the primary at its present mass.
    pub fn time_to_floor(&self, s: &OrbitState) -> f64 {
        ((s.a.powf(6.5) - self.floor.powf(6.5)) / (self.kappa * s.mass)).max(0.0)
    }

    /// The exact update over `tau` at the mass of the step, then the loss `lost`. The
    /// eccentricity follows `e ∝ a^exponent` (Proposition 10.3.7.1; zero without A-Λ).
    pub fn advance(&self, s: &mut OrbitState, tau: f64, lost: f64, exponent: f64) -> Result<(), SimError> {
        let base = s.a.powf(6.5) - self.kappa * s.mass * tau;
        let a = if base > 0.0 { base.powf(1.0 / 6.5) } else { 0.0 };
        if !(a >= self.floor * (1.0 - 1e-12)) {
            return Err(SimError::model_not_closed(format!(
                "A-O: a step of {tau:.3e} s would take the orbit from {:.6e} m below the primary ({:.6e} m)",
                s.a, self.floor
            )));
        }
        if !(lost >= 0.0 && lost < s.mass) {
            return Err(SimError::model_not_closed(format!(
                "A-O: the step loses {lost:.6e} kg of a satellite of {:.6e} kg",
                s.mass
            )));
        }
        // Within the end tolerance the orbit is at the primary.
        let a = if a <= self.floor * (1.0 + 1e-9) { self.floor } else { a };
        s.e *= (a / s.a).powf(exponent);
        s.a = a;
        s.mass -= lost;
        s.escaped += lost;
        Ok(())
    }

    /// Whether the orbit has reached the primary.
    pub fn ended(&self, s: &OrbitState) -> bool {
        s.a <= self.floor * (1.0 + 1e-9)
    }
}

/// A saddle `L_k` of `W` and the outward axis `e_k` of its neck plane `Π_k`.
#[derive(Clone, Copy, Debug)]
pub struct Neck {
    pub point: DVec3,
    pub axis: DVec3,
}

/// The two necks of the Hill region of a synchronous satellite (Definition 10.3.6.2).
#[derive(Clone, Debug)]
pub struct Necks {
    pub necks: [Neck; 2],
    pub r_hill: f64,
    /// The centroid they were located about.
    pub centre: DVec3,
}

/// Points of the axial scan for a neck, bisections of its bracket, Newton iterations
/// allowed after them, and the step (over `r_H`) that ends those: the RTFP field locates a
/// saddle to about `10⁻⁷ r_H`, far below any mesh spacing.
const NECK_SCAN: usize = 64;
const NECK_BISECT: usize = 14;
const NEWTON_MAX: usize = 20;
const NEWTON_TOL: f64 = 1e-6;

/// The unit eigenvector of the one positive eigenvalue of `h`, if exactly one is positive.
fn index_one(h: DMat3) -> Option<DVec3> {
    let m = h.to_cols_array_2d();
    let eig = SymmetricEigen::new(Matrix3::from_fn(|i, j| m[j][i]));
    let positive: Vec<usize> = (0..3).filter(|&i| eig.eigenvalues[i] > 0.0).collect();
    (positive.len() == 1).then(|| {
        let v = eig.eigenvectors.column(positive[0]);
        DVec3::new(v[0], v[1], v[2]).normalize()
    })
}

impl Necks {
    /// The saddles of `W = U + ½ n²(3x² − z²)` on either side of the centroid. Along the axis
    /// through it, `±∂W/∂x` is positive beyond a neck all the way out, inside or outside the
    /// body: a scan to `2 r_H` brackets the outermost sign change, bisection narrows it, and Newton's iteration on `∇W = 0` with the RTFP field moves to the saddle itself,
    /// which an asymmetric body puts off the axis. A saddle that Newton's iteration does not
    /// reach near the axial point is that point.
    pub fn locate(fwd: &Forward, g: f64, mass: f64, load: Loading) -> Result<Self, SimError> {
        let n2 = load.tide;
        let c = fwd.moments.centroid;
        let r_hill = (g * mass / (3.0 * n2)).cbrt();
        let hill = DMat3::from_diagonal(DVec3::new(3.0 * n2, 0.0, -n2));
        let grad_w = |x: DVec3, grad_u: DVec3| grad_u + DVec3::new(3.0 * n2 * x.x, 0.0, -n2 * x.z);
        let side = [1.0, -1.0];
        let at = |k: usize, t: f64| c + DVec3::X * (side[k] * t);
        let ts: Vec<f64> = (1..=NECK_SCAN).map(|j| 2.0 * r_hill * j as f64 / NECK_SCAN as f64).collect();
        let pts: Vec<DVec3> = (0..2).flat_map(|k| ts.iter().map(move |t| at(k, *t))).collect();
        let f = fwd.eval_exterior(&pts)?;
        let (mut lo, mut hi) = ([0.0; 2], [0.0; 2]);
        for k in 0..2 {
            let p = k * NECK_SCAN;
            let slope = |j: usize| side[k] * grad_w(pts[p + j], f[p + j].0).x;
            if !(slope(NECK_SCAN - 1) > 0.0) {
                return Err(SimError::solver("A-O: no neck within 2 r_H of the centroid"));
            }
            // The outermost sign change: beyond the neck the tide dominates all the way out,
            // while density structure can add sign changes inside a body near its Roche limit.
            // Without any, the tide dominates at the centroid itself: the neck is there, and
            // that half of the body stands by its strength alone.
            let j = (0..NECK_SCAN).rev().find(|&j| !(slope(j) > 0.0)).map_or(0, |j| j + 1);
            (lo[k], hi[k]) = (if j == 0 { 0.0 } else { ts[j - 1] }, ts[j]);
        }
        for _ in 0..NECK_BISECT {
            let mid = [0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1])];
            let x = [at(0, mid[0]), at(1, mid[1])];
            let f = fwd.eval_exterior(&x)?;
            for k in 0..2 {
                if side[k] * grad_w(x[k], f[k].0).x > 0.0 {
                    hi[k] = mid[k];
                } else {
                    lo[k] = mid[k];
                }
            }
        }
        let axial = [at(0, 0.5 * (lo[0] + hi[0])), at(1, 0.5 * (lo[1] + hi[1]))];
        let mut x = axial;
        let mut h_w = [DMat3::ZERO; 2];
        let mut converged = [false; 2];
        for _ in 0..NEWTON_MAX {
            let f = fwd.eval_exterior(&x)?;
            for k in 0..2 {
                h_w[k] = f[k].1 + hill;
                if converged[k] || !(h_w[k].determinant().abs() > 0.0) {
                    continue;
                }
                let mut dx = -(h_w[k].inverse() * grad_w(x[k], f[k].0));
                let len = dx.length();
                if len > 0.05 * r_hill {
                    dx *= 0.05 * r_hill / len;
                }
                x[k] += dx;
                converged[k] = len <= NEWTON_TOL * r_hill;
            }
            if converged.iter().all(|c| *c) {
                break;
            }
        }
        let f_axial = fwd.eval_exterior(&axial)?;
        let neck = |k: usize| -> Result<Neck, SimError> {
            let saddle = converged[k] && (x[k] - axial[k]).length() <= 0.25 * r_hill && index_one(h_w[k]).is_some();
            let (point, h) = if saddle { (x[k], h_w[k]) } else { (axial[k], f_axial[k].1 + hill) };
            let mut axis = index_one(h).ok_or_else(|| {
                SimError::solver(format!("A-O: no neck near {point:?}: H_W has not exactly one positive eigenvalue"))
            })?;
            if (point - c).dot(axis) < 0.0 {
                axis = -axis;
            }
            Ok(Neck { point, axis })
        };
        Ok(Self { necks: [neck(0)?, neck(1)?], r_hill, centre: c })
    }

    /// Whether `x` lies in the outer half-space of a neck plane.
    pub fn beyond(&self, x: DVec3) -> bool {
        self.necks.iter().any(|n| (x - n.point).dot(n.axis) > 0.0)
    }

    /// The normal increment that brings `x` onto the first neck plane ahead along `normal`;
    /// `None` when no plane lies ahead.
    pub fn cap(&self, x: DVec3, normal: DVec3) -> Option<f64> {
        self.necks
            .iter()
            .filter(|n| normal.dot(n.axis) > 0.0)
            .map(|n| (n.point - x).dot(n.axis) / normal.dot(n.axis))
            .min_by(f64::total_cmp)
    }

    /// The mountain-pass critical energy of Definition 10.3.6.2 and its saddle bound
    /// `min_k{−W(L_k)}`, returned as `(E_crit, bound)`. On each neck plane Π_k the
    /// connecting level is the maximum of `W = U + ½n²(3x² − z²)` over a sampling disk
    /// of radius `1.5 r_H` about L_k; for a point mass it sits at the saddle itself, so
    /// `E_crit = −max_k pass_k` attains the bound there. For an elongated body the pass
    /// maximum can exceed `W(L_k)` away from `L_k`, and `E_crit` falls below the bound.
    pub fn critical_energy(&self, fwd: &Forward, tide: f64) -> Result<(f64, f64), SimError> {
        const M: usize = 24;
        let n_g = (2 * M + 1) * (2 * M + 1);
        let w = |x: DVec3, u: f64| u + 0.5 * tide * (3.0 * x.x * x.x - x.z * x.z);
        let mut pts = Vec::with_capacity(2 * n_g + 2);
        for n in &self.necks {
            let t = if n.axis.z.abs() < 0.9 { DVec3::Z } else { DVec3::Y };
            let u1 = n.axis.cross(t).normalize();
            let u2 = n.axis.cross(u1);
            for i in 0..=2 * M {
                for j in 0..=2 * M {
                    let s = (i as f64 / M as f64 - 1.0) * 1.5 * self.r_hill;
                    let t2 = (j as f64 / M as f64 - 1.0) * 1.5 * self.r_hill;
                    pts.push(n.point + u1 * s + u2 * t2);
                }
            }
        }
        pts.extend(self.necks.iter().map(|n| n.point));
        let u = fwd.eval_potential(&pts)?;
        let pass = |k: usize| {
            (0..n_g)
                .map(|i| w(pts[k * n_g + i], u[k * n_g + i]))
                .fold(f64::NEG_INFINITY, f64::max)
        };
        let e_crit = -pass(0).max(pass(1));
        let bound = self
            .necks
            .iter()
            .enumerate()
            .map(|(k, n)| -w(n.point, u[2 * n_g + k]))
            .fold(f64::INFINITY, f64::min);
        Ok((e_crit, bound))
    }

    /// `max_{x ∈ pts} max_k (x − L_k)·e_k / r_H`.
    pub fn margin(&self, pts: &[DVec3]) -> f64 {
        pts.iter()
            .flat_map(|x| self.necks.iter().map(move |n| (*x - n.point).dot(n.axis)))
            .fold(f64::NEG_INFINITY, f64::max)
            / self.r_hill
    }
}

/// The certified bracket of `L*` (strength `c + μ₂ p̃`, rest pressure; §10.0.12.3) of `body`,
/// bound to `fwd`, frozen under `load`, with the operators of its geometry.
pub fn certify(project: &Project, body: &Body, fwd: &mut Forward, load: Loading) -> Result<(Bracket, std::sync::Arc<super::stokes::Operators>), SimError> {
    fwd.eval_faces(body)?;
    let mut cache = GeometryCache::default();
    let (_, ops) = geometry(&mut cache, project, body, fwd)?;
    let set = LoadSet::of(project, &ops, fwd, load)?;
    let m = &project.material;
    let data = limit::data(&ops, &set.loads(project.g), &project.stokes, m)?;
    let b = limit::bracket(&ops, &data, &set.cohesion(), m.spec.mu_2, m.spec.p_min, 1e-3)?;
    Ok((b, ops))
}

/// Algorithm 10.0.12.4: where a frozen body first fails as its orbit shrinks.
#[derive(Clone, Debug, Serialize)]
pub struct FailureOrbit {
    /// The smallest certified-stable and the largest certified-failed semi-major axis (m);
    /// the failure orbit lies between them. `None` when none was found in the range.
    pub stable: Option<f64>,
    pub failed: Option<f64>,
    /// `(a, L_lo, L_hi)` of every evaluated orbit, in order.
    pub scan: Vec<(f64, f64, f64)>,
    /// Shear rates of the collapse mechanism at the failed orbit, normalized to `N(u) = 1`,
    /// as `[x, y, z, γ̇]` per volume node.
    pub mechanism: Vec<[f64; 4]>,
    /// Under A-Λ (§10.3.7) `stable` is certified shakedown and `failed` a certified peak
    /// collapse; `ratchet` is the largest semi-major axis certified not to shake down, and
    /// `cycle_scan` holds `(a, e, L_peak lower bound, L_sd lo, L_sd hi)` per orbit.
    pub ratchet: Option<f64>,
    pub cycle_scan: Vec<[f64; 5]>,
}

/// Options of [`failure_orbit`].
#[derive(Clone, Copy, Debug)]
pub struct FailureOptions {
    pub gm: f64,
    pub a_max: f64,
    pub a_min: f64,
    /// Relative step of the downward scan, and the relative width the bisection stops at.
    pub scan: f64,
    pub tol: f64,
    /// Friction of the strength `c + μ p̃` (`μ₂` or `μ_s`).
    pub mu: f64,
    /// Davis' reduced strength `(c cos φ, sin φ)` with `tan φ = μ` (Conjecture A.2).
    pub davis: bool,
    /// Relative bracket gap below which an orbit counts as undecided.
    pub gap: f64,
    /// Under A-Λ: the semi-major axis at which the configured eccentricity holds, and the
    /// rate `ȧ` of the primary's tide there.
    pub a_ref: Option<(f64, f64)>,
}

/// The failure orbit of `body` frozen (Algorithm 10.0.12.4).
pub fn failure_orbit(project: &Project, body: &Body, opt: &FailureOptions) -> Result<FailureOrbit, SimError> {
    let mut fwd = Forward::bind(body, project.g)?;
    fwd.eval_faces(body)?;
    let mut cache = GeometryCache::default();
    let (_, ops) = geometry(&mut cache, project, body, &fwd)?;
    let at = |s: f64| -> Result<LimitData, SimError> {
        let set = LoadSet::of(project, &ops, &fwd, Loading { omega: s.sqrt(), tide: s })?;
        limit::data(&ops, &set.loads(project.g), &project.stokes, &project.material)
    };
    // ℓ and p₀ are affine in s = n²: two evaluations give every orbit.
    let s_hi = opt.gm / opt.a_min.powi(3);
    let (d0, d1) = (at(0.0)?, at(s_hi)?);
    let set = LoadSet::of(project, &ops, &fwd, Loading::default())?;
    let (mut cohesion, mut mu) = (set.cohesion(), opt.mu);
    if opt.davis {
        let cos = 1.0 / (1.0 + mu * mu).sqrt();
        cohesion.iter_mut().for_each(|c| *c *= cos);
        mu *= cos;
    }
    let p_min = project.material.spec.p_min;
    if let Some(prm) = &project.cycle {
        let a_ref = opt.a_ref.ok_or_else(|| SimError::invalid_input("A-Λ: the failure orbit needs the orbit at which the eccentricity holds and its rate"))?;
        let ctx = CycleScan::new(project, prm, &ops, &fwd, opt.gm, a_ref)?;
        return cycle_failure_orbit(&ops, &ctx, (&d0, &d1, s_hi), &cohesion, mu, p_min, opt);
    }
    let mut scan = Vec::new();
    let mut eval = |a: f64| -> Result<Bracket, SimError> {
        let s = opt.gm / a.powi(3);
        let data = LimitData::lerp(&d0, &d1, s / s_hi);
        let b = limit::bracket(&ops, &data, &cohesion, mu, p_min, opt.gap)?;
        scan.push((a, b.lo, b.hi));
        Ok(b)
    };
    let (mut stable, mut failed, mut mech) = (None, None, None);
    let mut a = opt.a_max;
    while a >= opt.a_min {
        let b = eval(a)?;
        if b.hi < 1.0 {
            stable = Some(a);
        } else if b.lo > 1.0 {
            failed = Some(a);
            mech = Some(b.rates);
            break;
        }
        a *= 1.0 - opt.scan;
    }
    if let (Some(mut hi), Some(mut lo)) = (stable, failed) {
        while (hi - lo) > opt.tol * hi {
            let mid = (hi * lo).sqrt();
            let b = eval(mid)?;
            if b.hi < 1.0 {
                hi = mid;
            } else if b.lo > 1.0 {
                lo = mid;
                mech = Some(b.rates);
            } else {
                break;
            }
        }
        (stable, failed) = (Some(hi), Some(lo));
    }
    let mechanism = mech
        .map(|r| ops.vol.pts.iter().zip(r).map(|(x, g)| [x.x, x.y, x.z, g]).collect())
        .unwrap_or_default();
    Ok(FailureOrbit { stable, failed, scan, mechanism, ratchet: None, cycle_scan: Vec::new() })
}

/// The frozen body's tidal cycle of unit `n² e` and the law `e(a)` (Proposition 10.3.7.1).
struct CycleScan {
    unit: ElasticCycle,
    phases: usize,
    gm: f64,
    a_ref: f64,
    e_ref: f64,
    exponent: f64,
}

impl CycleScan {
    fn new(project: &Project, prm: &CycleParams, ops: &Operators, fwd: &Forward, gm: f64, (a_ref, adot_ref): (f64, f64)) -> Result<Self, SimError> {
        let chi = chi_of(fwd.moments.inertia);
        let unit = Cycle::new(1.0, 1.0, Libration::new(1.0, chi, prm.q_satellite));
        let strength = super::stokes::Strengths::profile(ops, &project.material);
        let unit = elastic_cycle(ops, &unit, &project.material, &strength, project.g)?;
        let mo = &fwd.moments;
        let r_s = (3.0 * mo.volume / (4.0 * std::f64::consts::PI)).cbrt();
        let rho = mo.mass / mo.volume;
        let k2 = love_number(prm.rigidity, rho, project.g * mo.mass / (r_s * r_s), r_s);
        let n_ref = (gm / a_ref.powi(3)).sqrt();
        // κ m = −(13/2) a^{11/2} ȧ at the reference orbit (A-O, item 1).
        let kappa_m = -6.5 * a_ref.powf(5.5) * adot_ref;
        if !(kappa_m > 0.0) {
            return Err(SimError::invalid_input("A-Λ: the reference rate ȧ must be negative (a decaying orbit)"));
        }
        let beta = 68.25 * k2 / prm.q_satellite * (gm / project.g) * r_s.powi(5) * gm.sqrt() / (kappa_m * mo.mass);
        let exponent = eccentricity_coefficient(prm.primary_spin, n_ref) + beta;
        Ok(Self { unit, phases: prm.phases, gm, a_ref, e_ref: prm.eccentricity, exponent })
    }
}


impl CycleScan {
    /// `e(a)` and the scale `n² e` of the unit cycle.
    fn at(&self, a: f64) -> (f64, f64) {
        let e = self.e_ref * (a / self.a_ref).powf(self.exponent);
        (e, self.gm / a.powi(3) * e)
    }
}

/// Algorithm 10.0.12.4 under A-Λ: the scan down to the first certified peak collapse, then
/// bisections of the shakedown and peak boundaries (Proposition 10.3.7.5).
#[allow(clippy::too_many_arguments)]
fn cycle_failure_orbit(
    ops: &Operators,
    ctx: &CycleScan,
    (d0, d1, s_hi): (&LimitData, &LimitData, f64),
    cohesion: &[f64],
    mu: f64,
    p_min: f64,
    opt: &FailureOptions,
) -> Result<FailureOrbit, SimError> {
    let (mut scan, mut cycle_scan) = (Vec::new(), Vec::new());
    let mut eval = |a: f64| -> Result<(Bracket, CycleBracket), SimError> {
        let s = opt.gm / a.powi(3);
        let data = LimitData::lerp(d0, d1, s / s_hi);
        let b = limit::bracket(ops, &data, cohesion, mu, p_min, opt.gap)?;
        let (e, scale) = ctx.at(a);
        let c = cycle_bracket(ops, &b, &data.load, &ctx.unit, scale, cohesion, mu, &data.p0, p_min, ctx.phases);
        scan.push((a, b.lo, b.hi));
        cycle_scan.push([a, e, c.peak_lo, c.shakedown_lo, c.shakedown_hi]);
        Ok((b, c))
    };
    let (mut stable, mut ratchet, mut failed, mut mech) = (None, None, None, None);
    let (mut last_sound, mut last_up) = (None, None);
    let mut a = opt.a_max;
    while a >= opt.a_min {
        let (b, c) = eval(a)?;
        if c.shakedown_hi < 1.0 {
            stable = Some(a);
        }
        if c.shakedown_lo > 1.0 && ratchet.is_none() {
            ratchet = Some(a);
            last_sound = stable;
        }
        if c.peak_lo > 1.0 {
            failed = Some(a);
            mech = Some(b.rates);
            break;
        }
        last_up = Some(a);
        a *= 1.0 - opt.scan;
    }
    // The shakedown boundary between the last certified shakedown and the first ratchet.
    if let (Some(mut hi), Some(mut lo)) = (last_sound, ratchet) {
        while (hi - lo) > opt.tol * hi {
            let mid = (hi * lo).sqrt();
            let (_, c) = eval(mid)?;
            if c.shakedown_hi < 1.0 {
                hi = mid;
            } else if c.shakedown_lo > 1.0 {
                lo = mid;
            } else {
                break;
            }
        }
        (stable, ratchet) = (Some(hi), Some(lo));
    }
    // The peak boundary: the largest orbit certified to collapse at its worst phase.
    if let (Some(mut hi), Some(mut lo)) = (last_up, failed) {
        while (hi - lo) > opt.tol * hi {
            let mid = (hi * lo).sqrt();
            let (b, c) = eval(mid)?;
            if c.peak_lo > 1.0 {
                lo = mid;
                mech = Some(b.rates);
            } else {
                hi = mid;
            }
        }
        failed = Some(lo);
    }
    let mechanism = mech
        .map(|r| ops.vol.pts.iter().zip(r).map(|(x, g)| [x.x, x.y, x.z, g]).collect())
        .unwrap_or_default();
    Ok(FailureOrbit { stable, failed, scan, mechanism, ratchet, cycle_scan })
}
