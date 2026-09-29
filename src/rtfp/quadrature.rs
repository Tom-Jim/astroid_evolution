//! 球面高斯-勒让德求积采样 (满足 Σω = 4π 且 Σω T(u) = 0)
//! 参照 ryugu_cauchy_gravity/src/rust/compute/native/quadrature.rs

pub fn gauss_legendre(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut nodes = vec![0.0; n];
    let mut weights = vec![0.0; n];
    for i in 0..n {
        let mut x = (std::f64::consts::PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
        let mut dp = 0.0;
        for _ in 0..100 {
            let (mut p1, mut p2) = (1.0, 0.0);
            for j in 0..n {
                let p3 = p2;
                p2 = p1;
                p1 = ((2 * j + 1) as f64 * x * p2 - j as f64 * p3) / (j + 1) as f64;
            }
            dp = n as f64 * (x * p1 - p2) / (x * x - 1.0);
            let dx = p1 / dp;
            x -= dx;
            if dx.abs() < 1e-15 {
                break;
            }
        }
        nodes[i] = x;
        weights[i] = 2.0 / ((1.0 - x * x) * dp * dp);
    }
    (nodes, weights)
}

pub fn product_rule(n_theta: usize) -> Vec<([f64; 3], f64)> {
    let n_phi = 2 * n_theta;
    let (nodes, weights) = gauss_legendre(n_theta);
    let mut dirs = Vec::with_capacity(n_theta * n_phi);
    for i in 0..n_theta {
        let r = (1.0 - nodes[i] * nodes[i]).max(0.0).sqrt();
        for j in 0..n_phi {
            let phi = 2.0 * std::f64::consts::PI * (j as f64 + 0.5) / n_phi as f64;
            dirs.push((
                [r * phi.cos(), r * phi.sin(), nodes[i]],
                weights[i] * 2.0 * std::f64::consts::PI / n_phi as f64,
            ));
        }
    }
    dirs
}

/// 获取接近目标数量的方向集合 (默认 288 = 12 × 24)
pub fn directions(total: usize) -> Vec<([f64; 3], f64)> {
    let want = total.max(8) as f64;
    let n_theta = ((want / 2.0).sqrt().round() as usize).max(2);
    product_rule(n_theta)
}
