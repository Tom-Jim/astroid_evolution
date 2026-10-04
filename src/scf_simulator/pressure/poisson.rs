use nalgebra::{DMatrix, DVector};

#[derive(Clone, Debug)]
pub struct PressureField {
    pub pressure: Vec<f64>,
    pub tension_norm: f64,
}

/// Scalar Poisson solve followed by the tension cutoff p̃ = max(p, p_min).
pub fn solve_pressure(
    stiffness: &DMatrix<f64>,
    divergence: &DVector<f64>,
    p_min: f64,
) -> PressureField {
    let raw = stiffness
        .clone()
        .cholesky()
        .expect("pressure stiffness")
        .solve(divergence);
    let mut tension = 0.0;
    let pressure = raw
        .iter()
        .map(|p| {
            if *p < 0.0 {
                tension += -p;
            }
            p.max(p_min)
        })
        .collect();
    PressureField {
        pressure,
        tension_norm: tension,
    }
}
