//! Vector types. `DVec3` is `glam::DVec3`; the Hessian is the six-component symmetric tensor `Sym6`.

pub use glam::{DMat3, DMat4, DVec3};

/// Symmetric tensor packed as `(xx, yy, zz, xy, xz, yz)`.
pub type Sym6 = [f64; 6];

/// `Sym6` as a full matrix.
#[inline]
pub fn sym_mat(s: Sym6) -> DMat3 {
    DMat3::from_cols_array(&[s[0], s[3], s[4], s[3], s[1], s[5], s[4], s[5], s[2]])
}

/// The symmetric part of `m` packed as `Sym6`.
#[inline]
pub fn mat_sym(m: DMat3) -> Sym6 {
    let a = m.to_cols_array_2d();
    [
        a[0][0],
        a[1][1],
        a[2][2],
        0.5 * (a[1][0] + a[0][1]),
        0.5 * (a[2][0] + a[0][2]),
        0.5 * (a[2][1] + a[1][2]),
    ]
}

/// `[hi, lo]` with `hi = f32(x)` and `lo = f32(x − hi)`: the double-single pair the kernels read.
#[inline]
pub fn ds_pair(x: f64) -> [f32; 2] {
    let hi = x as f32;
    [hi, (x - f64::from(hi)) as f32]
}

/// `n` nearly uniform unit directions on the golden-angle spiral, regular at both poles.
pub fn fibonacci_directions(n: usize) -> Vec<DVec3> {
    let golden = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
    (0..n)
        .map(|i| {
            let z = 1.0 - 2.0 * (i as f64 + 0.5) / n as f64;
            let r = (1.0 - z * z).max(0.0).sqrt();
            let phi = golden * i as f64;
            DVec3::new(r * phi.cos(), r * phi.sin(), z)
        })
        .collect()
}
