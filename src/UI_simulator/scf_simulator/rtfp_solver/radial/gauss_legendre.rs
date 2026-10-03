//! Fixed-node Gauss–Legendre tables. Gauss–Legendre is the Jacobi `(0,0)` rule
//! required by MS §3.3 on a nonsingular segment mapped to `[-1, 1]`.

pub fn gauss_legendre(n: usize) -> (Vec<f64>, Vec<f64>) {
    let n = n.max(1);
    let mut nodes = vec![0.0; n];
    let mut weights = vec![0.0; n];
    for i in 0..n {
        let mut x = ((i as f64 + 0.75) / (n as f64 + 0.5) * std::f64::consts::PI).cos();
        let mut dp = 0.0;
        for _ in 0..80 {
            let (mut p1, mut p2) = (1.0, 0.0);
            for j in 0..n {
                let p3 = p2;
                p2 = p1;
                p1 = ((2 * j + 1) as f64 * x * p2 - j as f64 * p3) / (j + 1) as f64;
            }
            dp = n as f64 * (x * p1 - p2) / (x * x - 1.0);
            let dx = p1 / dp;
            x -= dx;
            if dx.abs() < 1e-16 {
                break;
            }
        }
        nodes[i] = x;
        weights[i] = 2.0 / ((1.0 - x * x) * dp * dp);
    }
    (nodes, weights)
}

/// Product rule on `S²`. Weights sum to `4π` for every `n_theta ≥ 1`.
pub fn sphere_directions(n_theta: usize) -> Vec<([f64; 3], f64)> {
    let n_theta = n_theta.max(2);
    let n_phi = 2 * n_theta;
    let (nodes, weights) = gauss_legendre(n_theta);
    let mut dirs = Vec::with_capacity(n_theta * n_phi);
    let dphi = 2.0 * std::f64::consts::PI / n_phi as f64;
    for i in 0..n_theta {
        let z = nodes[i];
        let r = (1.0 - z * z).max(0.0).sqrt();
        for j in 0..n_phi {
            let phi = dphi * (j as f64 + 0.5);
            dirs.push(([r * phi.cos(), r * phi.sin(), z], weights[i] * dphi));
        }
    }
    dirs
}

pub fn directions_for_budget(total: usize) -> Vec<([f64; 3], f64)> {
    let n_theta = ((total.max(8) as f64 / 2.0).sqrt().round() as usize).max(2);
    sphere_directions(n_theta)
}

/// Nonsingular segment quadrature of `(I_U, I_g, I_H)` for `D^{-α}`.
/// The caller must not use this on a segment that starts at `R = 0`.
pub fn quadrature_triplet(
    alpha: f64,
    b: f64,
    d2: f64,
    r0: f64,
    r1: f64,
    nodes: &[f64],
    weights: &[f64],
) -> (f64, f64, f64) {
    let h = 0.5 * (r1 - r0);
    let mid = 0.5 * (r1 + r0);
    let mut iu = 0.0;
    let mut ig = 0.0;
    let mut ih = 0.0;
    for (node, weight) in nodes.iter().zip(weights) {
        let r = mid + h * node;
        let diff = r - b;
        let den = (diff * diff + d2).powf(-alpha);
        iu += weight * r * den;
        ig += weight * den;
        ih += weight * den / r;
    }
    (iu * h, ig * h, ih * h)
}

/// Finite-part integrand `(D^{-α} − D(0)^{-α}) / R` on `(0, L]`.
/// The node at `R = 0` is replaced by the analytic limit `2 α b D(0)^{−α−1}`.
pub fn quadrature_hessian_finite_part(
    alpha: f64,
    b: f64,
    d2: f64,
    l: f64,
    ell0: f64,
    nodes: &[f64],
    weights: &[f64],
) -> f64 {
    let a0 = b * b + d2;
    let rho_factor = a0.powf(-alpha);
    let limit = 2.0 * alpha * b * a0.powf(-alpha - 1.0);
    let h = 0.5 * l;
    let mut acc = 0.0;
    for (node, weight) in nodes.iter().zip(weights) {
        let r = h * (1.0 + node);
        let ratio = if r <= 1e-14 * l.max(1.0) {
            limit
        } else {
            let diff = r - b;
            ((diff * diff + d2).powf(-alpha) - rho_factor) / r
        };
        acc += weight * ratio;
    }
    acc * h + rho_factor * (l / ell0).ln()
}
