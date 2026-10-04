/// Exogenous spin. No feedback from inertia or internal flow.

#[derive(Clone, Debug)]
pub struct SpinStage {
    pub t_end: f64,
    pub omega: [f64; 3],
}

#[derive(Clone, Debug)]
pub enum SpinSchedule {
    Stages(Vec<SpinStage>),
    Linear {
        omega0: [f64; 3],
        omega1: [f64; 3],
        t0: f64,
        t1: f64,
    },
    Table {
        times: Vec<f64>,
        omegas: Vec<[f64; 3]>,
    },
}

#[derive(Clone, Copy, Debug)]
pub struct SpinSample {
    pub omega: [f64; 3],
    pub omega_dot: [f64; 3],
    pub t_next: f64,
}

fn lerp(a: [f64; 3], b: [f64; 3], s: f64) -> [f64; 3] {
    [
        a[0] + (b[0] - a[0]) * s,
        a[1] + (b[1] - a[1]) * s,
        a[2] + (b[2] - a[2]) * s,
    ]
}

impl SpinSchedule {
    pub fn sample(&self, t: f64) -> Result<SpinSample, super::ScfError> {
        if !t.is_finite() {
            return Err(super::ScfError::model_not_closed("spin time is not finite"));
        }
        match self {
            Self::Stages(stages) => {
                if stages.is_empty() {
                    return Err(super::ScfError::model_not_closed("spin stages are empty"));
                }
                for stage in stages {
                    if t <= stage.t_end {
                        return Ok(SpinSample {
                            omega: stage.omega,
                            omega_dot: [0.0; 3],
                            t_next: stage.t_end,
                        });
                    }
                }
                let last = stages.last().unwrap();
                Ok(SpinSample {
                    omega: last.omega,
                    omega_dot: [0.0; 3],
                    t_next: last.t_end,
                })
            }
            Self::Linear {
                omega0,
                omega1,
                t0,
                t1,
            } => {
                let span = t1 - t0;
                if span <= 0.0 {
                    return Err(super::ScfError::model_not_closed(
                        "linear spin interval is empty",
                    ));
                }
                let s = ((t - t0) / span).clamp(0.0, 1.0);
                let rate = 1.0 / span;
                Ok(SpinSample {
                    omega: lerp(*omega0, *omega1, s),
                    omega_dot: [
                        (omega1[0] - omega0[0]) * rate,
                        (omega1[1] - omega0[1]) * rate,
                        (omega1[2] - omega0[2]) * rate,
                    ],
                    t_next: *t1,
                })
            }
            Self::Table { times, omegas } => {
                if times.len() != omegas.len() || times.len() < 2 {
                    return Err(super::ScfError::model_not_closed(
                        "spin table is incomplete",
                    ));
                }
                if t <= times[0] {
                    return Ok(SpinSample {
                        omega: omegas[0],
                        omega_dot: [0.0; 3],
                        t_next: times[0],
                    });
                }
                for i in 0..times.len() - 1 {
                    if t <= times[i + 1] {
                        let span = times[i + 1] - times[i];
                        let s = (t - times[i]) / span;
                        return Ok(SpinSample {
                            omega: lerp(omegas[i], omegas[i + 1], s),
                            omega_dot: [
                                (omegas[i + 1][0] - omegas[i][0]) / span,
                                (omegas[i + 1][1] - omegas[i][1]) / span,
                                (omegas[i + 1][2] - omegas[i][2]) / span,
                            ],
                            t_next: times[i + 1],
                        });
                    }
                }
                let last = omegas.len() - 1;
                Ok(SpinSample {
                    omega: omegas[last],
                    omega_dot: [0.0; 3],
                    t_next: times[last],
                })
            }
        }
    }
}
