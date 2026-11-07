//! Exogenous spin `t -> omega(t)` (§10.3.1). It is a command-line input of `scf-cli`.
//!
//! The spin axis is the body `z` axis (assumption A-S); `omega` is never updated from
//! the inertia tensor or from any angular momentum.
//!
//! Grammar of the `--spin` argument:
//!
//! * `stages:T1@w1,T2@w2,...`  stage `j` lasts `T_j` seconds at `w_j` rad/s;
//! * `const:w@T`               one stage;
//! * `linear:w0,rate,T`        `omega(t) = w0 + rate t` on `[0, T]`;
//! * `table:t0@w0,t1@w1,...`   piecewise linear, `t_k` strictly increasing;
//! * `table-file:PATH`         the same table, one `t w` pair per line.
//! * `orbit:GM,a0,adot0,T`      synchronous rotation on a tidally decaying orbit
//!   about a primary of gravitational parameter `GM` on the body `+x` axis
//!   (MS §10.0, `g_ext`): `ω = n = √(GM/a³)`, and the tidal Hill potential
//!   `n²(x² − (y² + z²)/2)` enters the effective potential. The semi-major axis
//!   follows the classical tidal law `ȧ ∝ a^{−11/2}`, i.e.
//!   `a(t)^{13/2} = a0^{13/2} + (13/2) a0^{11/2} ȧ0 t`, with `ȧ0` its rate at `t = 0`.

use crate::common::error::SimError;

fn read_spin_table(path: &str) -> Result<String, SimError> {
    crate::io::memfs::read_to_string(std::path::Path::new(path))
        .map_err(|e| SimError::io(format!("spin table: {e}")))
}

#[derive(Clone, Debug, PartialEq)]
pub enum SpinSchedule {
    /// `(duration, omega)` per stage.
    Stages(Vec<(f64, f64)>),
    Linear {
        omega0: f64,
        rate: f64,
        t_end: f64,
    },
    Table {
        times: Vec<f64>,
        omegas: Vec<f64>,
    },
    Orbit {
        gm: f64,
        a0: f64,
        adot0: f64,
        t_end: f64,
    },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpinSample {
    pub omega: f64,
    /// `d omega / dt` of the current segment (zero inside a stage).
    pub omega_dot: f64,
    /// End of the current stage, table segment, or the schedule.
    pub t_next: f64,
}

fn number(token: &str, what: &str) -> Result<f64, SimError> {
    token
        .trim()
        .parse::<f64>()
        .ok()
        .filter(|v| v.is_finite())
        .ok_or_else(|| {
            SimError::model_not_closed(format!("spin: {what} '{token}' is not a finite number"))
        })
}

fn pair(token: &str) -> Result<(f64, f64), SimError> {
    let (a, b) = token.split_once('@').ok_or_else(|| {
        SimError::model_not_closed(format!("spin: '{token}' is not of the form A@B"))
    })?;
    Ok((number(a, "time")?, number(b, "omega")?))
}

impl SpinSchedule {
    pub fn parse(spec: &str) -> Result<Self, SimError> {
        let (kind, body) = spec
            .split_once(':')
            .ok_or_else(|| SimError::model_not_closed("spin: expected KIND:ARGS"))?;
        let schedule = match kind.trim() {
            "const" => {
                let (omega, duration) = {
                    let (w, t) = body
                        .split_once('@')
                        .ok_or_else(|| SimError::model_not_closed("spin: const needs w@T"))?;
                    (number(w, "omega")?, number(t, "duration")?)
                };
                Self::Stages(vec![(duration, omega)])
            }
            "stages" => {
                let mut stages = Vec::new();
                for token in body.split(',').filter(|t| !t.trim().is_empty()) {
                    let (duration, omega) = pair(token)?;
                    stages.push((duration, omega));
                }
                Self::Stages(stages)
            }
            "linear" => {
                let parts: Vec<&str> = body.split(',').collect();
                if parts.len() != 3 {
                    return Err(SimError::model_not_closed("spin: linear needs w0,rate,T"));
                }
                Self::Linear {
                    omega0: number(parts[0], "omega0")?,
                    rate: number(parts[1], "rate")?,
                    t_end: number(parts[2], "T")?,
                }
            }
            "table" => {
                let mut times = Vec::new();
                let mut omegas = Vec::new();
                for token in body.split(',').filter(|t| !t.trim().is_empty()) {
                    let (t, w) = pair(token)?;
                    times.push(t);
                    omegas.push(w);
                }
                Self::Table { times, omegas }
            }
            "table-file" => {
                let text = read_spin_table(body.trim())?;
                let mut times = Vec::new();
                let mut omegas = Vec::new();
                for line in text
                    .lines()
                    .map(str::trim)
                    .filter(|l| !l.is_empty() && !l.starts_with('#'))
                {
                    let mut it = line.split_whitespace();
                    let (t, w) = (it.next(), it.next());
                    match (t, w) {
                        (Some(t), Some(w)) => {
                            times.push(number(t, "time")?);
                            omegas.push(number(w, "omega")?);
                        }
                        _ => return Err(SimError::model_not_closed("spin table line needs 't w'")),
                    }
                }
                Self::Table { times, omegas }
            }
            "orbit" => {
                let p: Vec<f64> = body
                    .split(',')
                    .map(|t| number(t, "orbit parameter"))
                    .collect::<Result<_, _>>()?;
                if p.len() != 4 {
                    return Err(SimError::model_not_closed(
                        "spin: orbit needs GM,a0,adot0,T",
                    ));
                }
                Self::Orbit {
                    gm: p[0],
                    a0: p[1],
                    adot0: p[2],
                    t_end: p[3],
                }
            }
            other => {
                return Err(SimError::model_not_closed(format!(
                    "spin: unknown kind '{other}' (stages, const, linear, table, table-file, orbit)"
                )));
            }
        };
        schedule.validate()?;
        Ok(schedule)
    }

    pub fn validate(&self) -> Result<(), SimError> {
        match self {
            Self::Stages(stages) => {
                if stages.is_empty() {
                    return Err(SimError::model_not_closed("spin: no stage"));
                }
                for &(duration, omega) in stages {
                    if !(duration.is_finite() && duration > 0.0) {
                        return Err(SimError::model_not_closed(
                            "spin: stage duration must be positive",
                        ));
                    }
                    if !(omega.is_finite() && omega >= 0.0) {
                        return Err(SimError::model_not_closed(
                            "spin: omega must be non-negative",
                        ));
                    }
                }
            }
            Self::Linear {
                omega0,
                rate,
                t_end,
            } => {
                if !(t_end.is_finite() && *t_end > 0.0) {
                    return Err(SimError::model_not_closed(
                        "spin: linear T must be positive",
                    ));
                }
                if !(omega0.is_finite() && rate.is_finite())
                    || *omega0 < 0.0
                    || omega0 + rate * t_end < 0.0
                {
                    return Err(SimError::model_not_closed(
                        "spin: linear omega must stay non-negative on [0, T]",
                    ));
                }
            }
            Self::Table { times, omegas } => {
                if times.len() < 2 || times.len() != omegas.len() {
                    return Err(SimError::model_not_closed(
                        "spin: a table needs at least two nodes",
                    ));
                }
                if times.windows(2).any(|w| w[1] <= w[0]) || times[0] < 0.0 {
                    return Err(SimError::model_not_closed(
                        "spin: table times must start at or after 0 and strictly increase",
                    ));
                }
                if omegas.iter().any(|w| !w.is_finite() || *w < 0.0) {
                    return Err(SimError::model_not_closed(
                        "spin: omega must be non-negative",
                    ));
                }
            }
            Self::Orbit {
                gm,
                a0,
                adot0,
                t_end,
            } => {
                if !(*gm > 0.0 && *a0 > 0.0 && *t_end > 0.0 && adot0.is_finite()) {
                    return Err(SimError::model_not_closed(
                        "spin: orbit needs GM > 0, a0 > 0, T > 0 and a finite rate",
                    ));
                }
                if !(self.semi_major_axis(*t_end) > 0.0) {
                    return Err(SimError::model_not_closed(
                        "spin: the orbit decays to zero before T",
                    ));
                }
            }
        }
        Ok(())
    }

    /// Semi-major axis of an `orbit` schedule; zero for the others.
    pub fn semi_major_axis(&self, t: f64) -> f64 {
        match self {
            Self::Orbit {
                a0, adot0, t_end, ..
            } => {
                let k = 6.5_f64;
                let base = a0.powf(k) + k * a0.powf(k - 1.0) * adot0 * t.min(*t_end);
                if base > 0.0 { base.powf(1.0 / k) } else { 0.0 }
            }
            _ => 0.0,
        }
    }

    /// Tidal strength `GM/a³` (s⁻²) of the primary; zero without one.
    pub fn tide(&self, t: f64) -> f64 {
        match self {
            Self::Orbit { gm, .. } => gm / self.semi_major_axis(t).powi(3),
            _ => 0.0,
        }
    }

    /// `T_end`.
    pub fn t_end(&self) -> f64 {
        match self {
            Self::Stages(stages) => stages.iter().map(|s| s.0).sum(),
            Self::Linear { t_end, .. } => *t_end,
            Self::Table { times, .. } => *times.last().unwrap_or(&0.0),
            Self::Orbit { t_end, .. } => *t_end,
        }
    }

    pub fn is_stages(&self) -> bool {
        matches!(self, Self::Stages(_))
    }

    pub fn sample(&self, t: f64) -> SpinSample {
        match self {
            Self::Orbit {
                a0, adot0, t_end, ..
            } => {
                // n = √(GM/a³), dn/dt = −(3/2) (n/a) ȧ with ȧ = ȧ0 (a0/a)^{11/2}.
                let a = self.semi_major_axis(t);
                let n = self.tide(t).sqrt();
                let adot = adot0 * (a0 / a).powf(5.5);
                SpinSample {
                    omega: n,
                    omega_dot: -1.5 * n / a * adot,
                    t_next: *t_end,
                }
            }
            Self::Stages(stages) => {
                let mut end = 0.0;
                for &(duration, omega) in stages {
                    end += duration;
                    if t < end {
                        return SpinSample {
                            omega,
                            omega_dot: 0.0,
                            t_next: end,
                        };
                    }
                }
                SpinSample {
                    omega: stages.last().map_or(0.0, |s| s.1),
                    omega_dot: 0.0,
                    t_next: end,
                }
            }
            Self::Linear {
                omega0,
                rate,
                t_end,
            } => SpinSample {
                omega: (omega0 + rate * t.min(*t_end)).max(0.0),
                omega_dot: *rate,
                t_next: *t_end,
            },
            Self::Table { times, omegas } => {
                let last = times.len() - 1;
                // Before the first node the spin is held at ω₀; the step ends at t₀.
                if t < times[0] {
                    return SpinSample {
                        omega: omegas[0],
                        omega_dot: 0.0,
                        t_next: times[0],
                    };
                }
                for k in 0..last {
                    if t < times[k + 1] {
                        let slope = (omegas[k + 1] - omegas[k]) / (times[k + 1] - times[k]);
                        return SpinSample {
                            omega: omegas[k] + slope * (t - times[k]),
                            omega_dot: slope,
                            t_next: times[k + 1],
                        };
                    }
                }
                SpinSample {
                    omega: omegas[last],
                    omega_dot: 0.0,
                    t_next: times[last],
                }
            }
        }
    }
}
