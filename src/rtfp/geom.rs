//! BVH 射线遍历与三角形高精度求交 (Möller-Trumbore)
//! 参照 ryugu_cauchy_gravity/src/rust/compute/native/geom.rs

use super::bvh::Bvh;
use super::mesh::{cross, dot, Mesh};

pub const MAX_HITS: usize = 32;

/// 遍历扁平二叉 BVH，计算射线 o + t·d 的所有穿透交点距离
pub fn bvh_crossings(
    bvh: &Bvh,
    mesh: &Mesh,
    o: [f64; 3],
    d: [f64; 3],
    t_min: f64,
    t_max: f64,
    out: &mut Vec<f64>,
) {
    out.clear();
    let mut stack = vec![0u32];
    while let Some(node) = stack.pop() {
        let (lo, hi) = (
            &bvh.bounds[2 * node as usize],
            &bvh.bounds[2 * node as usize + 1],
        );
        let mut near = t_min;
        let mut far = t_max;
        for k in 0..3 {
            let inv = 1.0 / d[k];
            let a = (lo[k] as f64 - o[k]) * inv;
            let b = (hi[k] as f64 - o[k]) * inv;
            near = near.max(a.min(b));
            far = far.min(a.max(b));
        }
        if far < near {
            continue;
        }
        let meta = bvh.meta[node as usize];
        if meta[3] > 0 {
            for i in 0..meta[3] {
                let tri = bvh.order[(meta[2] + i) as usize];
                let [i0, i1, i2] = mesh.face(tri as usize);
                let v0 = mesh.vertex(i0 as usize);
                let v1 = mesh.vertex(i1 as usize);
                let v2 = mesh.vertex(i2 as usize);
                if let Some(t) = intersect(v0, v1, v2, o, d) {
                    if t > t_min && t < t_max {
                        out.push(t);
                    }
                }
            }
        } else {
            stack.push(meta[0]);
            stack.push(meta[1]);
        }
    }
    out.sort_by(|a, b| a.partial_cmp(b).unwrap());
    dedup_crossings(out);
    out.truncate(MAX_HITS);
}

fn dedup_crossings(hits: &mut Vec<f64>) {
    hits.dedup_by(|a, b| {
        let scale = a.abs().max(b.abs()).max(1.0);
        (*a - *b).abs() <= (8e-7 * scale).max(1e-5)
    });
}

/// f64 Möller-Trumbore 三角形求交 (无背面剔除)
fn intersect(v0: [f64; 3], v1: [f64; 3], v2: [f64; 3], o: [f64; 3], d: [f64; 3]) -> Option<f64> {
    let e1 = sub(v1, v0);
    let e2 = sub(v2, v0);
    let pvec = cross(d, e2);
    let det = dot(e1, pvec);
    if det.abs() < 1e-18 {
        return None;
    }
    let inv = 1.0 / det;
    let tvec = sub(o, v0);
    let u = dot(tvec, pvec) * inv;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let qvec = cross(tvec, e1);
    let v = dot(d, qvec) * inv;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t = dot(e2, qvec) * inv;
    if t.is_finite() { Some(t) } else { None }
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
