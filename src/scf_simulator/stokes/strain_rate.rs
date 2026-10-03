use nalgebra::Matrix3;
#[inline]
pub fn compute_strain_rate(velocity_gradient: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let flat = [
        velocity_gradient[0][0],
        velocity_gradient[0][1],
        velocity_gradient[0][2],
        velocity_gradient[1][0],
        velocity_gradient[1][1],
        velocity_gradient[1][2],
        velocity_gradient[2][0],
        velocity_gradient[2][1],
        velocity_gradient[2][2],
    ];
    let vg = Matrix3::from_row_slice(&flat);
    let d = 0.5 * (vg + vg.transpose());
    let mut out = [[0.0; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            out[r][c] = d[(r, c)];
        }
    }
    out
}
#[inline]
pub fn compute_deviatoric_strain_rate(d: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let flat = [
        d[0][0], d[0][1], d[0][2], d[1][0], d[1][1], d[1][2], d[2][0], d[2][1], d[2][2],
    ];
    let mat = Matrix3::from_row_slice(&flat);
    let dev = mat - (mat.trace() / 3.0) * Matrix3::identity();
    let mut out = [[0.0; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            out[r][c] = dev[(r, c)];
        }
    }
    out
}
#[inline]
pub fn compute_shear_rate(d_prime: &[[f64; 3]; 3]) -> f64 {
    let flat = [
        d_prime[0][0],
        d_prime[0][1],
        d_prime[0][2],
        d_prime[1][0],
        d_prime[1][1],
        d_prime[1][2],
        d_prime[2][0],
        d_prime[2][1],
        d_prime[2][2],
    ];
    let mat = Matrix3::from_row_slice(&flat);
    (2.0 * mat.norm_squared()).max(0.0).sqrt()
}
#[inline]
pub fn compute_shear_rate_from_vel_grad(velocity_gradient: &[[f64; 3]; 3]) -> f64 {
    let d = compute_strain_rate(velocity_gradient);
    let d_prime = compute_deviatoric_strain_rate(&d);
    compute_shear_rate(&d_prime)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_simple_shear() {
        let gamma0 = 2.5;
        let mut vel_grad = [[0.0; 3]; 3];
        vel_grad[0][1] = gamma0;
        let d = compute_strain_rate(&vel_grad);
        assert!((d[0][1] - 0.5 * gamma0).abs() < 1e-14);
        assert!((d[1][0] - 0.5 * gamma0).abs() < 1e-14);
        assert!((d[0][0]).abs() < 1e-14);
        let d_prime = compute_deviatoric_strain_rate(&d);
        assert!((d_prime[0][1] - 0.5 * gamma0).abs() < 1e-14);
        let shear_rate = compute_shear_rate(&d_prime);
        assert!((shear_rate - gamma0).abs() < 1e-12);
    }
    #[test]
    fn test_pure_dilation_zero_shear() {
        let vel_grad = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let d = compute_strain_rate(&vel_grad);
        let d_prime = compute_deviatoric_strain_rate(&d);
        for i in 0..3 {
            for j in 0..3 {
                assert!(d_prime[i][j].abs() < 1e-14);
            }
        }
        let shear_rate = compute_shear_rate(&d_prime);
        assert!(shear_rate < 1e-14);
    }
}
