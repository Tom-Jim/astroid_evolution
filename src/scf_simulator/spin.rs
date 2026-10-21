//! Exogenous spin `t -> omega(t)` (§10.4.1). It is a command-line input of `scf-cli`.
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

use super::error::ScfError;

fn read_spin_table(path: &str) -> Result<String, ScfError> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        std::fs::read_to_string(path).map_err(|e| ScfError::io(format!("spin table {path}: {e}")))
    }
    #[cfg(target_arch = "wasm32")]
    {
        crate::rtfp_solver::io::memfs::read_to_string(std::path::Path::new(path))
            .map_err(|e| ScfError::io(format!("spin table {path}: {e}")))
    }
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
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpinSample {
    pub omega: f64,
    /// `d omega / dt` of the current segment (zero inside a stage).
    pub omega_dot: f64,
    /// End of the current stage, table segment, or the schedule.
    pub t_next: f64,
}

fn number(token: &str, what: &str) -> Result<f64, ScfError> {
    token
        .trim()
        .parse::<f64>()
        .ok()
        .filter(|v| v.is_finite())
        .ok_or_else(|| {
            ScfError::model_not_closed(format!("spin: {what} '{token}' is not a finite number"))
        })
}

fn pair(token: &str) -> Result<(f64, f64), ScfError> {
    let (a, b) = token.split_once('@').ok_or_else(|| {
        ScfError::model_not_closed(format!("spin: '{token}' is not of the form A@B"))
    })?;
    Ok((number(a, "time")?, number(b, "omega")?))
}

impl SpinSchedule {
    pub fn parse(spec: &str) -> Result<Self, ScfError> {
        let (kind, body) = spec
            .split_once(':')
            .ok_or_else(|| ScfError::model_not_closed("spin: expected KIND:ARGS"))?;
        let schedule = match kind.trim() {
            "const" => {
                let (omega, duration) = {
                    let (w, t) = body
                        .split_once('@')
                        .ok_or_else(|| ScfError::model_not_closed("spin: const needs w@T"))?;
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
                    return Err(ScfError::model_not_closed("spin: linear needs w0,rate,T"));
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
                        _ => return Err(ScfError::model_not_closed("spin table line needs 't w'")),
                    }
                }
                Self::Table { times, omegas }
            }
            other => {
                return Err(ScfError::model_not_closed(format!(
                    "spin: unknown kind '{other}' (stages, const, linear, table, table-file)"
                )));
            }
        };
        schedule.validate()?;
        Ok(schedule)
    }

    pub fn validate(&self) -> Result<(), ScfError> {
        match self {
            Self::Stages(stages) => {
                if stages.is_empty() {
                    return Err(ScfError::model_not_closed("spin: no stage"));
                }
                for &(duration, omega) in stages {
                    if !(duration.is_finite() && duration > 0.0) {
                        return Err(ScfError::model_not_closed(
                            "spin: stage duration must be positive",
                        ));
                    }
                    if !(omega.is_finite() && omega >= 0.0) {
                        return Err(ScfError::model_not_closed(
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
                    return Err(ScfError::model_not_closed(
                        "spin: linear T must be positive",
                    ));
                }
                if !(omega0.is_finite() && rate.is_finite())
                    || *omega0 < 0.0
                    || omega0 + rate * t_end < 0.0
                {
                    return Err(ScfError::model_not_closed(
                        "spin: linear omega must stay non-negative on [0, T]",
                    ));
                }
            }
            Self::Table { times, omegas } => {
                if times.len() < 2 || times.len() != omegas.len() {
                    return Err(ScfError::model_not_closed(
                        "spin: a table needs at least two nodes",
                    ));
                }
                if times.windows(2).any(|w| w[1] <= w[0]) || times[0] < 0.0 {
                    return Err(ScfError::model_not_closed(
                        "spin: table times must start at or after 0 and strictly increase",
                    ));
                }
                if omegas.iter().any(|w| !w.is_finite() || *w < 0.0) {
                    return Err(ScfError::model_not_closed(
                        "spin: omega must be non-negative",
                    ));
                }
            }
        }
        Ok(())
    }

    /// `T_end`.
    pub fn t_end(&self) -> f64 {
        match self {
            Self::Stages(stages) => stages.iter().map(|s| s.0).sum(),
            Self::Linear { t_end, .. } => *t_end,
            Self::Table { times, .. } => *times.last().unwrap_or(&0.0),
        }
    }

    pub fn is_stages(&self) -> bool {
        matches!(self, Self::Stages(_))
    }

    pub fn sample(&self, t: f64) -> SpinSample {
        match self {
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
                if t <= times[0] {
                    return SpinSample {
                        omega: omegas[0],
                        omega_dot: (omegas[1] - omegas[0]) / (times[1] - times[0]),
                        t_next: times[1],
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stages_hold_omega_until_their_end() {
        let s = SpinSchedule::parse("stages:100@1e-4,50@2e-4").unwrap();
        assert_eq!(s.t_end(), 150.0);
        let a = s.sample(10.0);
        assert_eq!((a.omega, a.t_next), (1e-4, 100.0));
        let b = s.sample(100.0);
        assert_eq!((b.omega, b.t_next), (2e-4, 150.0));
    }

    #[test]
    fn linear_and_table_interpolate() {
        let l = SpinSchedule::parse("linear:1e-4,1e-7,1000").unwrap();
        assert!((l.sample(500.0).omega - 1.5e-4).abs() < 1e-18);
        let t = SpinSchedule::parse("table:0@1e-4,10@3e-4").unwrap();
        assert!((t.sample(5.0).omega - 2e-4).abs() < 1e-18);
        assert!(SpinSchedule::parse("linear:1e-4,-1e-3,1000").is_err());
        assert!(SpinSchedule::parse("table:0@1,0@2").is_err());
        assert!(SpinSchedule::parse("sometimes:1").is_err());
    }
}
