use super::jacobi::{kappa, sigma, weighted_orthogonality};

#[derive(Clone, Copy, Debug)]
pub struct RadialTraces {
    pub sigma: [f64; 4],
    pub kappa: [f64; 4],
    pub c_p: [f64; 4],
    pub c_t: [f64; 4],
}

/// Surface traces and the weighted Jacobi norms for one angular degree.
pub fn radial_traces(l: usize) -> RadialTraces {
    RadialTraces {
        sigma: std::array::from_fn(|k| sigma(l, k)),
        kappa: std::array::from_fn(|k| kappa(l, k)),
        c_p: std::array::from_fn(|k| weighted_orthogonality(l, k, k)),
        c_t: std::array::from_fn(|k| weighted_orthogonality(l, k, k)),
    }
}
