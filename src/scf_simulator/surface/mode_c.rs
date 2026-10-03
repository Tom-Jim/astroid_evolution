use glam::{DVec3, DVec4};
#[inline]
pub fn backtrace_characteristic_1st_order(
    x_arrival: [f64; 3],
    velocity: [f64; 3],
    dt: f64,
) -> [f64; 3] {
    (DVec3::from_array(x_arrival) - dt * DVec3::from_array(velocity)).to_array()
}
#[inline]
pub fn backtrace_characteristic_2nd_order(
    x_arrival: [f64; 3],
    velocity_half: [f64; 3],
    dt: f64,
) -> [f64; 3] {
    (DVec3::from_array(x_arrival) - dt * DVec3::from_array(velocity_half)).to_array()
}
pub fn triangle_barycentric_coordinates(
    p: [f64; 3],
    a: [f64; 3],
    b: [f64; 3],
    c: [f64; 3],
) -> [f64; 3] {
    let pt = DVec3::from_array(p);
    let va = DVec3::from_array(a);
    let vb = DVec3::from_array(b);
    let vc = DVec3::from_array(c);
    let v0 = vb - va;
    let v1 = vc - va;
    let v2 = pt - va;
    let d00 = v0.dot(v0);
    let d01 = v0.dot(v1);
    let d11 = v1.dot(v1);
    let d20 = v2.dot(v0);
    let d21 = v2.dot(v1);
    let denom = (d00 * d11 - d01 * d01).max(1e-30);
    let v = (d11 * d20 - d01 * d21) / denom;
    let w = (d00 * d21 - d01 * d20) / denom;
    let u = 1.0 - v - w;
    [u, v, w]
}
pub fn barycentric_interpolate_modal_state(
    bary: [f64; 3],
    m_a: &[f64; 4],
    m_b: &[f64; 4],
    m_c: &[f64; 4],
) -> [f64; 4] {
    let ma = DVec4::from_slice(m_a);
    let mb = DVec4::from_slice(m_b);
    let mc = DVec4::from_slice(m_c);
    (bary[0] * ma + bary[1] * mb + bary[2] * mc).to_array()
}
pub use backtrace_characteristic_1st_order as backtrace_surface_characteristic;
pub use backtrace_characteristic_2nd_order as backtrace_surface_characteristic_rk2;
pub use triangle_barycentric_coordinates as project_point_to_triangle_barycentric;
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_backtrace_and_barycentric() {
        let x_i = [100.0, 50.0, 20.0];
        let vel = [10.0, -5.0, 2.0];
        let dt = 0.5;
        let x_d = backtrace_characteristic_1st_order(x_i, vel, dt);
        assert_eq!(x_d, [95.0, 52.5, 19.0]);
        let a = [0.0, 0.0, 0.0];
        let b = [1.0, 0.0, 0.0];
        let c = [0.0, 1.0, 0.0];
        let bary_a = triangle_barycentric_coordinates(a, a, b, c);
        assert!((bary_a[0] - 1.0).abs() < 1e-10);
        assert!(bary_a[1].abs() < 1e-10);
        assert!(bary_a[2].abs() < 1e-10);
        let mid = [0.3333333333333333, 0.3333333333333333, 0.0];
        let bary_mid = triangle_barycentric_coordinates(mid, a, b, c);
        assert!((bary_mid[0] - 0.3333333333333333).abs() < 1e-6);
        assert!((bary_mid[1] - 0.3333333333333333).abs() < 1e-6);
        assert!((bary_mid[2] - 0.3333333333333333).abs() < 1e-6);
    }
}
