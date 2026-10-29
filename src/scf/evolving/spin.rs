//! Hypothesis A-L (§11.7): the spin follows the angular momentum.
//!
//! The rigid rows of (B1) make the frame a Tisserand frame, so `L = I Ω`. The scalar spin
//! angular momentum `L_n` is advanced by the axial torque and the spin rate is read from
//! the settled inertia, `ω_n = L_n / I_1⁽ⁿ⁾`. Principal-axis rotation (A-S) is kept; the
//! realignment of step 8b is the true polar wander of Proposition 11.12.

use nalgebra::{Matrix3, SymmetricEigen, Vector3};
use serde::{Deserialize, Serialize};

use crate::common::error::SimError;
use crate::scf::forward::Moments;
use crate::scf::frame::inertia_about_centroid;
use crate::scf::spin::SpinSchedule;

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpinLawParams {
    /// `"schedule"` (law L1: `T_∥ = I_ref ω̇` of the spin schedule, impulses at its jumps)
    /// or `"radiative"` (law L2: the calibrated radiative torque of A-Θ).
    pub torque: String,
    /// L2: the calibration is refused when `|T_rad| < floor · |T_cal|` at step 0.
    #[serde(default)]
    pub calibration_floor: Option<f64>,
}

impl SpinLawParams {
    pub fn validate(&self) -> Result<(), SimError> {
        match self.torque.as_str() {
            "schedule" => Ok(()),
            "radiative" => match self.calibration_floor {
                Some(f) if f.is_finite() && f > 0.0 && f < 1.0 => Ok(()),
                _ => Err(SimError::model_not_closed(
                    "spin_law.calibration_floor in (0, 1) is required for torque = \"radiative\"",
                )),
            },
            other => Err(SimError::model_not_closed(format!(
                "spin_law.torque must be \"schedule\" or \"radiative\", got \"{other}\""
            ))),
        }
    }
}

/// `L_n`, the moment `I_ref` that turns the schedule into a torque, and `κ_Y` of law L2
/// (`None` until the calibration state is reached).
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct SpinState {
    pub l: f64,
    pub i_ref: f64,
    #[serde(default)]
    pub kappa_y: Option<f64>,
}

/// Per-step diagnostics of §11.10 under A-L.
#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct SpinReport {
    pub active: bool,
    pub angular_momentum: f64,
    pub torque: f64,
    pub i1: f64,
    /// `ε_ad`: creep drift of the principal axis per wobble period.
    pub eps_adiabatic: f64,
    pub omega_wobble: f64,
    /// Relative gap `(I_1 − I_2) / I_1`.
    pub gap: f64,
    /// `ϑ_L` (degrees) and `ΔE` (J) of the settling (Proposition 11.12(2)).
    pub theta_l_deg: f64,
    pub delta_energy: f64,
    pub eps_euler: f64,
}

impl SpinReport {
    pub fn inactive(&self) -> bool {
        !self.active
    }
}

/// Principal moments about the centroid, largest first, and the axis of the largest.
pub struct Principal {
    pub inertia: Matrix3<f64>,
    pub moments: [f64; 3],
    pub a1: Vector3<f64>,
}

pub fn principal(m: &Moments) -> Principal {
    let inertia = inertia_about_centroid(m);
    let eig = SymmetricEigen::new(inertia);
    let mut idx = [0usize, 1, 2];
    idx.sort_by(|a, b| eig.eigenvalues[*b].total_cmp(&eig.eigenvalues[*a]));
    Principal {
        inertia,
        moments: idx.map(|i| eig.eigenvalues[i]),
        a1: eig.eigenvectors.column(idx[0]).into_owned(),
    }
}

/// Angle (rad) between an axis and `e_z`, sign-free.
fn axis_angle(a: &Vector3<f64>) -> f64 {
    (a.z.abs() / a.norm().max(1e-300)).min(1.0).acos()
}

#[derive(Clone, Copy, PartialEq)]
enum Torque {
    Schedule,
    Radiative,
}

/// A-L for one run.
pub struct SpinLaw<'a> {
    torque: Torque,
    floor: f64,
    schedule: &'a SpinSchedule,
}

impl<'a> SpinLaw<'a> {
    /// `None` when A-L is off. The synchronous orbit schedule is a tidal lock (§11.7).
    pub fn new(params: Option<&SpinLawParams>, schedule: &'a SpinSchedule) -> Result<Option<Self>, SimError> {
        let Some(p) = params else { return Ok(None) };
        if matches!(schedule, SpinSchedule::Orbit { .. }) {
            return Err(SimError::model_not_closed(
                "spin_law (A-L) does not apply to the synchronous orbit schedule",
            ));
        }
        Ok(Some(Self {
            torque: if p.torque == "radiative" { Torque::Radiative } else { Torque::Schedule },
            floor: p.calibration_floor.unwrap_or(0.0),
            schedule,
        }))
    }

    pub fn radiative(&self) -> bool {
        self.torque == Torque::Radiative
    }

    /// The stage shortcut of step 6 needs a zero axial torque inside the stage
    /// (Proposition 11.14); law L1 has one, its impulses sit on the stage boundaries.
    pub fn allows_stage_jump(&self) -> bool {
        self.torque == Torque::Schedule
    }

    /// `L_0 = I_1⁽⁰⁾ ω(t)`, once, after the initial realignment.
    pub fn init(&self, state: &mut Option<SpinState>, t: f64, m: &Moments) {
        if state.is_none() {
            let i1 = principal(m).moments[0];
            *state = Some(SpinState {
                l: i1 * self.schedule.sample(t).omega,
                i_ref: i1,
                kappa_y: None,
            });
        }
    }

    /// `ω = L / I_1`.
    pub fn omega(s: &SpinState, m: &Moments) -> f64 {
        s.l / principal(m).moments[0]
    }

    /// Fix `κ_Y = T_cal / T_rad` at the first radiative step, with `T_cal = I_ref ω̇(t)`.
    pub fn calibrate(&self, s: &mut SpinState, t: f64, t_rad: f64) -> Result<(), SimError> {
        if self.torque != Torque::Radiative || s.kappa_y.is_some() {
            return Ok(());
        }
        let t_cal = s.i_ref * self.schedule.sample(t).omega_dot;
        if t_rad.abs() < self.floor * t_cal.abs() || (t_cal != 0.0 && t_rad == 0.0) {
            return Err(SimError::model_not_closed(format!(
                "A-L calibration: radiative torque {t_rad:.3e} N m is below the floor of the \
                 calibration torque {t_cal:.3e} N m"
            )));
        }
        s.kappa_y = Some(if t_rad == 0.0 { 0.0 } else { t_cal / t_rad });
        Ok(())
    }

    /// Axial torque at `t` (N m), for the step bound.
    pub fn torque(&self, s: &SpinState, t: f64, t_rad: f64) -> f64 {
        match self.torque {
            Torque::Schedule => s.i_ref * self.schedule.sample(t).omega_dot,
            Torque::Radiative => s.kappa_y.unwrap_or(0.0) * t_rad,
        }
    }

    /// `Δt_ω = ε_ω max(ω, ω_ref) I_1 / max(|T|, ε_T)` of §11.7.
    pub fn dt_bound(eps_omega: f64, omega_ref: f64, omega: f64, i1: f64, torque: f64) -> f64 {
        eps_omega * omega.max(omega_ref) * i1 / torque.abs().max(1e-300)
    }

    /// `L_{n+1} − L_n` over `[t0, t1]`: `I_ref (ω(t1) − ω(t0))` under L1, which is exact
    /// for a rate constant on the step and carries the jumps of a staged schedule; under
    /// L2 the torque of the step start held over the step.
    fn increment(&self, s: &SpinState, t0: f64, t1: f64, t_rad: f64) -> f64 {
        match self.torque {
            Torque::Schedule => s.i_ref * (self.schedule.sample(t1).omega - self.schedule.sample(t0).omega),
            Torque::Radiative => (t1 - t0) * s.kappa_y.unwrap_or(0.0) * t_rad,
        }
    }

    /// Step 9+: advance `L`, read `ω` on the settled inertia and report the indicators.
    /// `creep` is the inertia after the creep update and before settling.
    #[allow(clippy::too_many_arguments)]
    pub fn finish(
        &self,
        s: &mut SpinState,
        t0: f64,
        t1: f64,
        t_rad: f64,
        omega_n: f64,
        creep: &Moments,
        settled: &Moments,
    ) -> Result<(f64, SpinReport), SimError> {
        let dl = self.increment(s, t0, t1, t_rad);
        let l = s.l + dl;
        if !(l > 0.0) {
            return Err(SimError::model_not_closed(format!(
                "A-L: the angular momentum {l:.3e} kg m²/s is no longer positive"
            )));
        }
        s.l = l;
        let pc = principal(creep);
        let ps = principal(settled);
        let [i1, i2, _] = ps.moments;
        let omega = l / i1;
        let tau = t1 - t0;
        let wobble = |p: &Principal, w: f64| {
            let [a, b, c] = p.moments;
            w * (((a - b) * (a - c)) / (b * c)).max(0.0).sqrt()
        };
        let w_creep = wobble(&pc, omega_n);
        // Proposition 11.12(2): L along e_z of the step, the settled inertia I'.
        let inv_zz = ps.inertia.try_inverse().map_or(1.0 / i1, |m| m[(2, 2)]);
        let report = SpinReport {
            active: true,
            angular_momentum: l,
            torque: if tau > 0.0 { dl / tau } else { 0.0 },
            i1,
            eps_adiabatic: if tau > 0.0 && w_creep > 0.0 {
                axis_angle(&pc.a1) / (tau * w_creep)
            } else {
                0.0
            },
            omega_wobble: wobble(&ps, omega),
            gap: (i1 - i2) / i1,
            theta_l_deg: axis_angle(&ps.a1).to_degrees(),
            delta_energy: (0.5 * l * l * (inv_zz - 1.0 / i1)).max(0.0),
            eps_euler: if tau > 0.0 && omega_n > 0.0 {
                (l / pc.moments[0] - omega_n).abs() / (tau * omega_n * omega_n)
            } else {
                0.0
            },
        };
        Ok((omega, report))
    }
}
