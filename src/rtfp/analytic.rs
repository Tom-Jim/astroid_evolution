//! 均匀密度多面体引力梯度张量 (Werner 闭合解析面元积分)
//! 参照 ryugu_cauchy_gravity/src/rust/compute/native/analytic.rs

use super::mesh::{cross, dot, norm, Mesh};
use super::tensor::Sym6;

pub const G_SI: f64 = 6.674_30e-11;

#[derive(Clone, Copy, Debug, Default)]
pub struct Face {
    pub corners: [[f64; 3]; 3],
    pub n: [f64; 3],
    pub weight: f64,
}

/// 预计算面元独立几何数据
pub fn precompute(mesh: &Mesh) -> Vec<Face> {
    let mut out = Vec::with_capacity(mesh.face_count());
    for f in 0..mesh.face_count() {
        let [i0, i1, i2] = mesh.face(f);
        let corners = [
            mesh.vertex(i0 as usize),
            mesh.vertex(i1 as usize),
            mesh.vertex(i2 as usize),
        ];
        let n = normalize(cross(
            sub(corners[1], corners[0]),
            sub(corners[2], corners[1]),
        ));
        out.push(Face {
            corners,
            n,
            weight: 1.0,
        });
    }
    out
}

/// ∫ ds / √(s² + L²) 沿边积分 ln((s_B + l_B)/(s_A + l_A))
#[inline]
pub fn edge_log(s_a: f64, s_b: f64, l: f64) -> f64 {
    let l_a = (s_a * s_a + l * l).sqrt();
    let l_b = (s_b * s_b + l * l).sqrt();
    if s_a >= 0.0 || s_b > 0.0 {
        ((s_b + l_b) / (s_a + l_a).max(f64::MIN_POSITIVE)).ln()
    } else {
        ((l_a - s_a) / (l_b - s_b).max(f64::MIN_POSITIVE)).ln()
    }
}

/// Van Oosterom–Strackee 三角形有符号立体角
#[inline]
pub fn solid_angle(a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> f64 {
    let (an, bn, cn) = (normalize(a), normalize(b), normalize(c));
    let denom = 1.0 + dot(an, bn) + dot(bn, cn) + dot(cn, an);
    2.0 * (-dot(an, cross(bn, cn))).atan2(denom)
}

/// 单个面元矢量表面积分 I_F(x) = ∫_F (x - y) / ||x - y||³ dS
pub fn face_integral(f: &Face, x: [f64; 3]) -> [f64; 3] {
    let mut i_vec = [0.0f64; 3];
    for e in 0..3 {
        let a = f.corners[e];
        let b = f.corners[(e + 1) % 3];
        let dir = normalize(sub(b, a));
        let outward = cross(dir, f.n);
        let r_a = sub(a, x);
        let par = dot(r_a, dir);
        let perp = sub(r_a, scale(dir, par));
        let l = norm(perp);
        let log_term = edge_log(par, dot(sub(b, x), dir), l);
        for k in 0..3 {
            i_vec[k] += outward[k] * log_term;
        }
    }
    let omega = solid_angle(
        sub(f.corners[0], x),
        sub(f.corners[1], x),
        sub(f.corners[2], x),
    );
    for (slot, n) in i_vec.iter_mut().zip(f.n) {
        *slot += omega * n;
    }
    i_vec
}

/// 累加面元到 W_ij = G Σ_F n_j · I_F[i]
#[inline]
pub fn add_face(w: &mut Sym6, f: &Face, x: [f64; 3]) {
    let i_vec = face_integral(f, x);
    let k = G_SI * f.weight;
    w[0] += k * f.n[0] * i_vec[0];
    w[1] += k * f.n[1] * i_vec[1];
    w[2] += k * f.n[2] * i_vec[2];
    w[3] += k * f.n[0] * i_vec[1];
    w[4] += k * f.n[0] * i_vec[2];
    w[5] += k * f.n[1] * i_vec[2];
}

/// 计算单位密度下的封闭多面体 Hessian 张量 W(x)
pub fn hessian6(faces: &[Face], x: [f64; 3]) -> Sym6 {
    let mut w = [0.0f64; 6];
    for f in faces {
        add_face(&mut w, f, x);
    }
    w
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn scale(a: [f64; 3], k: f64) -> [f64; 3] {
    [a[0] * k, a[1] * k, a[2] * k]
}

fn normalize(a: [f64; 3]) -> [f64; 3] {
    let n = norm(a);
    if n > f64::MIN_POSITIVE {
        [a[0] / n, a[1] / n, a[2] / n]
    } else {
        [0.0, 0.0, 0.0]
    }
}
