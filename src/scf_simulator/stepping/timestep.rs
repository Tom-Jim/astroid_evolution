/// Δt = min{CFL, C, char, ω, max, next}. An empty active set lifts the CFL bound.

#[derive(Clone, Copy, Debug)]
pub struct TimestepBounds {
    pub cfl: f64,
    pub coupling: f64,
    pub characteristic: f64,
    pub spin: f64,
    pub max: f64,
    pub until_next: f64,
    pub active_set_empty: bool,
}

pub fn decide_timestep(bounds: TimestepBounds) -> f64 {
    let cfl = if bounds.active_set_empty {
        f64::INFINITY
    } else {
        bounds.cfl
    };
    let mut best = f64::INFINITY;
    for value in [
        cfl,
        bounds.coupling,
        bounds.characteristic,
        bounds.spin,
        bounds.max,
        bounds.until_next,
    ] {
        if value.is_nan() {
            return f64::NAN;
        }
        if value.is_finite() && value > 0.0 {
            best = best.min(value);
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bounds() -> TimestepBounds {
        TimestepBounds {
            cfl: 0.4,
            coupling: 0.5,
            characteristic: 0.8,
            spin: 2.0,
            max: 1.0,
            until_next: 3.0,
            active_set_empty: false,
        }
    }

    #[test]
    fn nan_bound_does_not_drop_out_of_the_minimum() {
        let mut sample = bounds();
        sample.characteristic = f64::NAN;
        assert!(decide_timestep(sample).is_nan());
    }

    #[test]
    fn empty_active_set_lifts_only_the_cfl_bound() {
        let mut sample = bounds();
        sample.active_set_empty = true;
        assert!((decide_timestep(sample) - 0.5).abs() < 1e-15);
    }
}
