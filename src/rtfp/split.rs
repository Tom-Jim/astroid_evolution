//! 近场对数奇点剔除与光滑余项积分 (Analytic Near-field Split)
//! 参照 ryugu_cauchy_gravity/src/rust/compute/native/split.rs

use super::density::KernelSi;

pub const MAX_INTERVALS: usize = 16;

#[inline]
fn d_of(r: f64, a: f64, b: f64, s2: f64) -> f64 {
    (a - 2.0 * s2 * b * r + s2 * r * r).max(1e-300)
}

/// 将交点距离排序列表组织为可视区间对 (r0, r1)
pub fn intervals(hits: &[f64], inside: bool) -> ([Option<(f64, f64)>; MAX_INTERVALS], bool) {
    let mut out = [None; MAX_INTERVALS];
    let mut n = 0usize;
    if inside {
        if hits.is_empty() {
            return (out, false);
        }
        out[n] = Some((0.0, hits[0]));
        n += 1;
        let mut i = 1;
        while i + 1 < hits.len() && n < MAX_INTERVALS {
            out[n] = Some((hits[i], hits[i + 1]));
            n += 1;
            i += 2;
        }
    } else {
        let mut i = 0;
        while i + 1 < hits.len() && n < MAX_INTERVALS {
            out[n] = Some((hits[i], hits[i + 1]));
            n += 1;
            i += 2;
        }
    }
    let needed = if inside && !hits.is_empty() {
        1 + (hits.len() - 1) / 2
    } else {
        hits.len() / 2
    };
    (out, needed > MAX_INTERVALS)
}

/// 计算单个方向 u 上的光滑余项标量: Σ_k w_k R_k(u)
/// 已扣除 ln(r1/r0) 奇异对数发散项，消除了有限部分参考尺度 ℓ0 的依赖
pub fn remainder_scalar(
    kernels: &[KernelSi],
    x: [f64; 3],
    u: [f64; 3],
    ivals: &[Option<(f64, f64)>; MAX_INTERVALS],
) -> f64 {
    if kernels.is_empty() || ivals.iter().all(|s| s.is_none()) {
        return 0.0;
    }
    let mut total = 0.0;
    for k in kernels {
        let p = [x[0] - k.c[0], x[1] - k.c[1], x[2] - k.c[2]];
        let p2 = p[0] * p[0] + p[1] * p[1] + p[2] * p[2];
        let s2 = k.sigma * k.sigma;
        let a = 1.0 + s2 * p2;
        let b = p[0] * u[0] + p[1] * u[1] + p[2] * u[2];
        let d = (1.0 / s2 + p2 - b * b).max(1e-30).sqrt();
        let mut jk = 0.0;
        for slot in ivals.iter().flatten() {
            let (r0, r1) = *slot;
            if r1 <= r0 {
                continue;
            }
            let dr1 = d_of(r1, a, b, s2);
            let phi1 = (b / d) * (r1 - b).atan2(d);
            jk += if r0 > 0.0 {
                let dr0 = d_of(r0, a, b, s2);
                (-0.5 * (dr1 / dr0).ln() + phi1 - (b / d) * (r0 - b).atan2(d)) / a
            } else {
                (-0.5 * (dr1 / a).ln() + phi1 + (b / d) * b.atan2(d)) / a
            };
        }
        total += k.w * jk;
    }
    total
}
