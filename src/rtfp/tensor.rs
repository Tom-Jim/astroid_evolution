//! 对称 3x3 张量打包为 6 个独立分量 [xx, yy, zz, xy, xz, yz]
//! 参照 ryugu_cauchy_gravity/src/rust/compute/native/tensor.rs

pub type Sym6 = [f64; 6];

/// 对称张量的 Frobenius 范数 (非对角线项计入两次)
pub fn frobenius(a: &Sym6) -> f64 {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2] + 2.0 * (a[3] * a[3] + a[4] * a[4] + a[5] * a[5]))
        .sqrt()
}

/// 张量迹 Tr(H) = H_xx + H_yy + H_zz
pub fn trace(a: &Sym6) -> f64 {
    a[0] + a[1] + a[2]
}

/// t += s · T(u)，其中 T(u) = 3uu^T - I (射线有限部分的核心张量核)
pub fn add_tensor_term(t: &mut Sym6, u: &[f64; 3], s: f64) {
    t[0] += s * (3.0 * u[0] * u[0] - 1.0);
    t[1] += s * (3.0 * u[1] * u[1] - 1.0);
    t[2] += s * (3.0 * u[2] * u[2] - 1.0);
    t[3] += s * 3.0 * u[0] * u[1];
    t[4] += s * 3.0 * u[0] * u[2];
    t[5] += s * 3.0 * u[1] * u[2];
}

/// 张量与向量收缩: (H · v)
pub fn contract_vector(h: &Sym6, v: &[f64; 3]) -> [f64; 3] {
    [
        h[0] * v[0] + h[3] * v[1] + h[4] * v[2],
        h[3] * v[0] + h[1] * v[1] + h[5] * v[2],
        h[4] * v[0] + h[5] * v[1] + h[2] * v[2],
    ]
}

/// 依据自由空间 (-1次调和引力场) 经典关系: 由 Hessian 恢复引力加速度 g
/// 对于无源自由空间 (ΔU = 0)，g(x) = -1/2 * H(x) · (x - x_cm)
pub fn gravity_from_hessian(h: &Sym6, point: &[f64; 3], center: &[f64; 3]) -> [f64; 3] {
    let r = [
        point[0] - center[0],
        point[1] - center[1],
        point[2] - center[2],
    ];
    let hr = contract_vector(h, &r);
    [-0.5 * hr[0], -0.5 * hr[1], -0.5 * hr[2]]
}
