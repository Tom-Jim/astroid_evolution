// Single ray–triangle predicate. P1 (`ray_count.wgsl`) and P3 (`ray_write.wgsl`)
// are concatenated with this file by the host. Do not paste a second copy.

fn ray_slab_hit(orig: vec3<f32>, dir: vec3<f32>, bmin: vec3<f32>, bmax: vec3<f32>) -> bool {
    var t0 = -1e30;
    var t1 = 1e30;
    for (var axis = 0u; axis < 3u; axis = axis + 1u) {
        let o = orig[axis];
        let d = dir[axis];
        let mn = bmin[axis];
        let mx = bmax[axis];
        if abs(d) < 1e-20 {
            if o < mn || o > mx {
                return false;
            }
            continue;
        }
        var ta = (mn - o) / d;
        var tb = (mx - o) / d;
        if ta > tb {
            let swap = ta;
            ta = tb;
            tb = swap;
        }
        t0 = max(t0, ta);
        t1 = min(t1, tb);
        if t1 < t0 {
            return false;
        }
    }
    return t1 >= 0.0;
}

fn ray_triangle_hit(
    orig: vec3<f32>,
    dir: vec3<f32>,
    v0: vec3<f32>,
    v1: vec3<f32>,
    v2: vec3<f32>,
    t_min: f32,
    t_max: f32,
) -> f32 {
    let e1 = v1 - v0;
    let e2 = v2 - v0;
    let pvec = cross(dir, e2);
    let det = dot(e1, pvec);
    let scale = length(e1) * length(e2) * max(length(dir), 1e-20);
    if abs(det) <= 1e-6 * scale {
        return -1.0;
    }
    let inv_det = 1.0 / det;
    let tvec = orig - v0;
    let u = dot(tvec, pvec) * inv_det;
    if u < -1e-5 || u > 1.0 + 1e-5 {
        return -1.0;
    }
    let qvec = cross(tvec, e1);
    let v = dot(dir, qvec) * inv_det;
    if v < -1e-5 || u + v > 1.0 + 1e-5 {
        return -1.0;
    }
    let t = dot(e2, qvec) * inv_det;
    if t > t_min && t < t_max {
        return t;
    }
    return -1.0;
}
