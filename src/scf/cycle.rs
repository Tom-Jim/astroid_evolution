//! Hypothesis A-Λ (§10.3.7): the eccentricity of a synchronous satellite, its forced
//! libration, and the tidal cycle they impose in the body frame, at first order in `e`.
//!
//! The cyclic body force per unit mass is `δg(x, M) = H(M) x + E(M) e_z × x` with
//! `H(M) = H_c cos M + H_s sin M` and `E(M) = E_c cos M + E_s sin M` (the boxed formula of
//! §10.3.7); its elastic response, the brackets and the steady cycle are in
//! [`super::stokes::elastic`].

use serde::{Deserialize, Serialize};

use crate::common::error::SimError;
use crate::common::math::{DMat3, DVec3};

use super::forward::Forward;
use super::mechanics::Mechanics;
use super::orbit::{Neck, Necks, OrbitLaw};
use super::params::Project;
use super::stokes::elastic::{CycleOptions, ElasticCycle, elastic_cycle, steady_cycle};

/// The `[cycle]` section of a project (needs `[orbit]`).
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CycleParams {
    /// Eccentricity at the start of the run.
    pub eccentricity: f64,
    /// Spin rate of the primary (rad/s), for the signs of its tidal lags.
    pub primary_spin: f64,
    /// Shear modulus `μ_e` of the elastic cycle (Pa) and the satellite's tidal `Q_s`.
    pub rigidity: f64,
    pub q_satellite: f64,
    /// Phases `N_M` of the sampled cycle, and the largest admissible libration angle (rad).
    pub phases: usize,
    pub psi_max: f64,
    /// Algorithm 10.3.7.6: cycles allowed, the tolerance of the return at a phase and of
    /// the periodic residual stress (relative to the strength).
    pub cycles_max: usize,
    pub tol: f64,
    pub tol_cycle: f64,
    /// S–N line `s = b₀ − b₁ log₁₀ N` and endurance limit `s_e` of the bonds; without it no
    /// high-cycle fatigue.
    #[serde(default)]
    pub fatigue: Option<FatigueParams>,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FatigueParams {
    pub intercept: f64,
    pub slope: f64,
    pub limit: f64,
}

impl FatigueParams {
    /// `r_F` (s⁻¹) at utilization `s` and mean motion `n`.
    pub fn rate(&self, s: f64, n: f64) -> f64 {
        if s > self.limit {
            n / (2.0 * std::f64::consts::PI) * 10f64.powf(-(self.intercept - s) / self.slope)
        } else {
            0.0
        }
    }
}

impl CycleParams {
    pub fn validate(&self) -> Result<(), SimError> {
        let finite = |name: &str, v: f64, ok: bool| {
            if v.is_finite() && ok {
                Ok(())
            } else {
                Err(SimError::model_not_closed(format!("cycle.{name} is out of range: {v}")))
            }
        };
        finite("eccentricity", self.eccentricity, (0.0..1.0).contains(&self.eccentricity))?;
        finite("primary_spin", self.primary_spin, self.primary_spin >= 0.0)?;
        finite("rigidity", self.rigidity, self.rigidity > 0.0)?;
        finite("q_satellite", self.q_satellite, self.q_satellite > 0.0)?;
        finite("psi_max", self.psi_max, self.psi_max > 0.0 && self.psi_max < 1.0)?;
        finite("tol", self.tol, self.tol > 0.0 && self.tol < 1.0)?;
        finite("tol_cycle", self.tol_cycle, self.tol_cycle > 0.0 && self.tol_cycle < 1.0)?;
        if self.phases < 4 || self.cycles_max == 0 {
            return Err(SimError::model_not_closed("cycle.phases must be at least 4 and cycle.cycles_max positive"));
        }
        if let Some(f) = &self.fatigue {
            finite("fatigue.slope", f.slope, f.slope > 0.0)?;
            finite("fatigue.limit", f.limit, f.limit > 0.0 && f.limit < f.intercept)?;
        }
        Ok(())
    }
}

/// `C_e` of Proposition 10.3.7.1: `ė/e = C_e ȧ_p/a` for the primary's tide at constant `Q`.
pub fn eccentricity_coefficient(primary_spin: f64, n: f64) -> f64 {
    let sgn = |x: f64| if x >= 0.0 { 1.0 } else { -1.0 };
    let (s1, s2, s3) = (sgn(primary_spin - n), sgn(2.0 * primary_spin - n), sgn(2.0 * primary_spin - 3.0 * n));
    (-0.75 * s1 - 3.0 / 16.0 * s2 + 147.0 / 16.0 * s3 - 9.0 / 8.0) / (3.0 * s1)
}

/// `k₂ = (3/2)/(1 + μ̃)`, `μ̃ = 19 μ_e / (2 ρ g R)` of a homogeneous body of mean density
/// `rho`, surface gravity `g` and radius `r`.
pub fn love_number(rigidity: f64, rho: f64, g: f64, r: f64) -> f64 {
    1.5 / (1.0 + 19.0 * rigidity / (2.0 * rho * g * r))
}

/// The libration `γ = γ_s sin M + γ_c cos M` of Proposition 10.3.7.2.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Libration {
    pub chi: f64,
    pub gamma_s: f64,
    pub gamma_c: f64,
}

impl Libration {
    /// `Γ = 2eχ / (χ − 1 + i√χ/Q_s)`.
    pub fn new(e: f64, chi: f64, q: f64) -> Self {
        let (re, im) = (chi - 1.0, chi.max(0.0).sqrt() / q);
        let den = re * re + im * im;
        let k = 2.0 * e * chi / den;
        Self { chi, gamma_s: k * re, gamma_c: -k * im }
    }

    pub fn amplitude(&self) -> f64 {
        self.gamma_s.hypot(self.gamma_c)
    }
}

/// `χ = 3(B − A)/C` of the inertia `(xx, yy, zz, …)` in the synchronous frame (primary on `+x`).
pub fn chi_of(inertia: [f64; 6]) -> f64 {
    3.0 * (inertia[1] - inertia[0]) / inertia[2]
}

/// The tidal cycle of one step in the body frame.
#[derive(Clone, Copy, Debug)]
pub struct Cycle {
    pub n: f64,
    pub e: f64,
    pub lib: Libration,
    /// `H_c`, `H_s` and `E_c`, `E_s`.
    pub h: [DMat3; 2],
    pub euler: [f64; 2],
}

impl Cycle {
    pub fn new(n: f64, e: f64, lib: Libration) -> Self {
        let n2 = n * n;
        let d2 = DMat3::from_diagonal(DVec3::new(2.0, -1.0, -1.0));
        let p = DMat3::from_diagonal(DVec3::new(1.0, 1.0, 0.0));
        let s = DMat3::from_cols(DVec3::Y, DVec3::X, DVec3::ZERO);
        let (gs, gc) = (lib.gamma_s, lib.gamma_c);
        let h_c = (d2 * (3.0 * e) - s * (3.0 * gc) + p * (2.0 * gs)) * n2;
        let h_s = (s * (3.0 * (2.0 * e - gs)) - p * (2.0 * gc)) * n2;
        Self { n, e, lib, h: [h_c, h_s], euler: [n2 * gc, n2 * gs] }
    }

    /// Component `k` (0: cos, 1: sin) of `δg` at `x`.
    pub fn component(&self, k: usize, x: DVec3) -> DVec3 {
        self.h[k] * x + DVec3::Z.cross(x) * self.euler[k]
    }

    pub fn delta_g(&self, x: DVec3, m: f64) -> DVec3 {
        self.component(0, x) * m.cos() + self.component(1, x) * m.sin()
    }

    /// `ψ(M) = 2e sin M − γ(M)`, the direction of the primary in the body frame.
    pub fn psi(&self, m: f64) -> f64 {
        (2.0 * self.e - self.lib.gamma_s) * m.sin() - self.lib.gamma_c * m.cos()
    }

    pub fn psi_amplitude(&self) -> f64 {
        (2.0 * self.e - self.lib.gamma_s).hypot(self.lib.gamma_c)
    }

    /// The spin rate `n + γ̇`.
    pub fn spin(&self, m: f64) -> f64 {
        self.n * (1.0 + self.lib.gamma_s * m.cos() - self.lib.gamma_c * m.sin())
    }

    /// The necks of the instantaneous Hill region at phase `m`.
    pub fn necks_at(&self, necks: &Necks, m: f64) -> Necks {
        let c = necks.centre;
        let (sn, cs) = self.psi(m).sin_cos();
        let rot = |v: DVec3| DVec3::new(cs * v.x - sn * v.y, sn * v.x + cs * v.y, v.z);
        let scale = 1.0 - self.e * m.cos();
        let at = |k: &Neck| Neck { point: c + rot(k.point - c) * scale, axis: rot(k.axis) };
        Necks { necks: [at(&necks.necks[0]), at(&necks.necks[1])], r_hill: necks.r_hill * scale, centre: c }
    }

    /// The `j`-th of `n` equispaced phases.
    pub fn phase(j: usize, n: usize) -> f64 {
        2.0 * std::f64::consts::PI * j as f64 / n as f64
    }
}

/// What A-Λ reports per step (inactive without it).
#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct CycleReport {
    pub active: bool,
    pub e: f64,
    pub chi: f64,
    pub libration: f64,
    pub psi: f64,
    /// Relative torque residual of Proposition 10.3.7.3.
    pub torque_residual: f64,
    /// `nR/√(μ_e/ρ)`, `1/μ̃`, `1/Q_s`.
    pub inertia: f64,
    pub self_gravity: f64,
    pub lag: f64,
    /// Bound of `L_peak` and bracket of `L_sd` (zero when not evaluated).
    pub peak_lo: f64,
    pub shakedown_lo: f64,
    pub shakedown_hi: f64,
    /// Algorithm 10.3.7.6: cycles run (0: the cycle is elastic), volume fractions that
    /// ratchet and that alternate, the largest ratchet speed (m/s) and fatigue rate (s⁻¹).
    pub cycles: usize,
    pub ratchet_fraction: f64,
    pub alternating_fraction: f64,
    pub ratchet_speed: f64,
    pub fatigue_max: f64,
}

impl CycleReport {
    pub fn inactive(&self) -> bool {
        !self.active
    }
}

/// One evaluation of A-Λ on the deep state of a step.
pub struct CycleOut {
    pub cycle: Cycle,
    pub elastic: ElasticCycle,
    /// Ratchet velocity coefficients (m/s per unit coefficient), alternating plastic shear
    /// rate and fatigue rate `r_F` at the volume nodes.
    pub ratchet: Vec<f64>,
    pub alternating: Vec<f64>,
    pub fatigue: Vec<f64>,
    pub report: CycleReport,
}

/// The cycle of the step at mean motion `n` and eccentricity `e` on the deep state `mech`
/// of the body bound to `fwd` (the principal frame of §10.3.4, step 8).
pub fn evaluate(project: &Project, prm: &CycleParams, mech: &Mechanics, fwd: &Forward, n: f64, e: f64) -> Result<CycleOut, SimError> {
    let _span = super::timing::span("cycle");
    let mo = &fwd.moments;
    let chi = chi_of(mo.inertia);
    let lib = Libration::new(e, chi, prm.q_satellite);
    let cycle = Cycle::new(n, e, lib);
    if lib.amplitude() > prm.psi_max || cycle.psi_amplitude() > prm.psi_max {
        return Err(SimError::model_not_closed(format!(
            "A-Λ: libration {:.3e} rad and primary swing {:.3e} rad exceed ψ_max = {:.3e} (χ = {chi:.4}, e = {e:.4e}): the linear libration, and possibly the lock, fails",
            lib.amplitude(),
            cycle.psi_amplitude(),
            prm.psi_max
        )));
    }
    let m = &project.material;
    let ops = &mech.ops;
    let elastic = elastic_cycle(ops, &cycle, m, &mech.strength, project.g)?;
    let np = ops.vol.pts.len();
    let (_, d) = ops.vol_velocity(&mech.sol.u);
    let creep: Vec<[f64; 6]> = (0..np)
        .map(|i| std::array::from_fn(|c| mech.sol.eta[i] * std::f64::consts::SQRT_2 * d[6 * i + c]))
        .collect();
    let cohesion: Vec<f64> = mech.strength.vol.iter().map(|l| l.c).collect();
    let unyielded: Vec<bool> = mech.class.plastic_rate.iter().map(|r| *r == 0.0).collect();
    let opt = CycleOptions {
        rigidity: prm.rigidity,
        phases: prm.phases,
        cycles_max: prm.cycles_max,
        tol: prm.tol,
        tol_cycle: prm.tol_cycle,
    };
    let steady = steady_cycle(ops, &elastic, &creep, &mech.sol.p_tilde, &cohesion, m.spec.mu_2, m.spec.p_min, &unyielded, &opt)?;
    let per_cycle = n / (2.0 * std::f64::consts::PI);
    let ratchet: Vec<f64> = steady.ratchet.iter().map(|c| c * per_cycle).collect();
    let alternating: Vec<f64> = steady.range.iter().map(|r| r * per_cycle).collect();
    let fatigue: Vec<f64> = (0..np)
        .map(|i| match &prm.fatigue {
            Some(f) if unyielded[i] && steady.range[i] == 0.0 => f.rate(steady.utilization[i], n),
            _ => 0.0,
        })
        .collect();
    let r_s = (3.0 * mo.volume / (4.0 * std::f64::consts::PI)).cbrt();
    let rho = mo.mass / mo.volume;
    let g = project.g * mo.mass / (r_s * r_s);
    let vol: f64 = ops.vol.w.iter().sum();
    let frac = |f: &dyn Fn(usize) -> bool| (0..np).filter(|i| f(*i)).map(|i| ops.vol.w[i]).sum::<f64>() / vol;
    let report = CycleReport {
        active: true,
        e,
        chi,
        libration: lib.amplitude(),
        psi: cycle.psi_amplitude(),
        torque_residual: chi.max(0.0).sqrt() / (prm.q_satellite * (1.0 - chi).abs()),
        inertia: n * r_s / (prm.rigidity / rho).sqrt(),
        self_gravity: 2.0 * rho * g * r_s / (19.0 * prm.rigidity),
        lag: 1.0 / prm.q_satellite,
        cycles: steady.cycles,
        ratchet_fraction: frac(&|i| steady.ratcheting[i]),
        alternating_fraction: frac(&|i| steady.range[i] > 0.0),
        fatigue_max: fatigue.iter().copied().fold(0.0, f64::max),
        ..CycleReport::default()
    };
    Ok(CycleOut { cycle, elastic, ratchet, alternating, fatigue, report })
}

/// `C_e + β` of Proposition 10.3.7.1 for the satellite bound to `fwd` at `(a, mass)`.
pub fn eccentricity_exponent(project: &Project, prm: &CycleParams, law: &OrbitLaw, fwd: &Forward, gm: f64, mass: f64, n: f64) -> f64 {
    let mo = &fwd.moments;
    let r_s = (3.0 * mo.volume / (4.0 * std::f64::consts::PI)).cbrt();
    let rho = mo.mass / mo.volume;
    let k2 = love_number(prm.rigidity, rho, project.g * mo.mass / (r_s * r_s), r_s);
    eccentricity_coefficient(prm.primary_spin, n) + law.beta(gm / project.g, k2 / prm.q_satellite, r_s, mass)
}
