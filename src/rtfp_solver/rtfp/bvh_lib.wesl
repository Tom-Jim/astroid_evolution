// GPU LBVH walked by the wasm trace and field kernels. WebGPU has no
// acceleration structure. macOS keeps the Metal one and does not use this.
//
// An internal node stores both children. A leaf has right == 0xffffffff and
// left equal to the triangle index (or 0xffffffff when the mesh is empty).
// The root is at index 0.

struct BvhNode {
    min_x: f32,
    min_y: f32,
    min_z: f32,
    left: u32,
    max_x: f32,
    max_y: f32,
    max_z: f32,
    right: u32,
}

@group(0) @binding(7) var<storage, read> bvh_nodes: array<BvhNode>;
@group(0) @binding(8) var<storage, read> bvh_tris: array<f32>;

const BVH_LEAF: u32 = 0xffffffffu;
const BVH_STACK: i32 = 256;

fn isfinite_f(x: f32) -> bool {
    return (bitcast<u32>(x) & 0x7fffffffu) < 0x7f800000u;
}

fn next_up(t: f32) -> f32 {
    if (!isfinite_f(t)) {
        return t;
    }
    if (t == 0.0) {
        return bitcast<f32>(1u);
    }
    let bits = bitcast<u32>(t);
    if (t > 0.0) {
        return bitcast<f32>(bits + 1u);
    }
    return bitcast<f32>(bits - 1u);
}

fn edge_shift(dir: vec3<f32>) -> vec3<f32> {
    // The triangle test keeps shared edges. Offsetting the origin misses a
    // centroid ray that should leave the body.
    let _unused = dir.x;
    return vec3<f32>(0.0, 0.0, 0.0);
}

fn tri_vert(index: u32, corner: u32) -> vec3<f32> {
    let base = index * 9u + corner * 3u;
    return vec3<f32>(bvh_tris[base], bvh_tris[base + 1u], bvh_tris[base + 2u]);
}

fn ray_aabb(origin: vec3<f32>, inv: vec3<f32>, t_min: f32, t_max: f32, node: BvhNode) -> bool {
    let bmin = vec3<f32>(node.min_x, node.min_y, node.min_z);
    let bmax = vec3<f32>(node.max_x, node.max_y, node.max_z);
    let t1 = (bmin - origin) * inv;
    let t2 = (bmax - origin) * inv;
    let tnear = max(max(min(t1.x, t2.x), min(t1.y, t2.y)), min(t1.z, t2.z));
    let tfar = min(min(max(t1.x, t2.x), max(t1.y, t2.y)), max(t1.z, t2.z));
    return tfar >= max(tnear, t_min) && tnear <= t_max;
}

fn ray_tri(origin: vec3<f32>, dir: vec3<f32>, index: u32, t_min: f32, t_max: f32) -> f32 {
    if (index >= arrayLength(&bvh_tris) / 9u) {
        return -1.0;
    }
    let a = tri_vert(index, 0u) - origin;
    let b = tri_vert(index, 1u) - origin;
    let c = tri_vert(index, 2u) - origin;
    var kz = 0;
    if (abs(dir.y) > abs(dir.x) && abs(dir.y) >= abs(dir.z)) {
        kz = 1;
    } else if (abs(dir.z) > abs(dir.x) && abs(dir.z) > abs(dir.y)) {
        kz = 2;
    }
    let kx = (kz + 1) % 3;
    let ky = (kx + 1) % 3;
    let dz = dir[kz];
    if (dz == 0.0) {
        return -1.0;
    }
    let sx = -dir[kx] / dz;
    let sy = -dir[ky] / dz;
    let sz = 1.0 / dz;
    var p0 = vec3<f32>(a[kx], a[ky], a[kz]);
    var p1 = vec3<f32>(b[kx], b[ky], b[kz]);
    var p2 = vec3<f32>(c[kx], c[ky], c[kz]);
    p0.x = fma(sx, p0.z, p0.x);
    p0.y = fma(sy, p0.z, p0.y);
    p1.x = fma(sx, p1.z, p1.x);
    p1.y = fma(sy, p1.z, p1.y);
    p2.x = fma(sx, p2.z, p2.x);
    p2.y = fma(sy, p2.z, p2.y);
    let e0 = fma(p1.x, p2.y, -(p1.y * p2.x));
    let e1 = fma(p2.x, p0.y, -(p2.y * p0.x));
    let e2 = fma(p0.x, p1.y, -(p0.y * p1.x));
    if ((e0 < 0.0 || e1 < 0.0 || e2 < 0.0) && (e0 > 0.0 || e1 > 0.0 || e2 > 0.0)) {
        return -1.0;
    }
    let det = e0 + e1 + e2;
    if (det == 0.0) {
        return -1.0;
    }
    let t = fma(e0, p0.z, fma(e1, p1.z, e2 * p2.z)) * (1.0 / det) * sz;
    if (t >= t_min && t <= t_max) {
        return t;
    }
    return -1.0;
}

// Nearest hit at or beyond t_min. A negative return is a miss.
fn bvh_closest(origin: vec3<f32>, dir: vec3<f32>, t_min: f32) -> f32 {
    let inv = vec3<f32>(
        select(1.0 / dir.x, 1.0e30, dir.x == 0.0),
        select(1.0 / dir.y, 1.0e30, dir.y == 0.0),
        select(1.0 / dir.z, 1.0e30, dir.z == 0.0),
    );
    var stack: array<u32, 256>;
    var sp: i32 = 0;
    stack[0] = 0u;
    var best = -1.0;
    var limit = 1.0e20;
    let count = arrayLength(&bvh_nodes);
    while (sp >= 0) {
        let index = stack[sp];
        sp = sp - 1;
        if (index >= count) {
            continue;
        }
        let node = bvh_nodes[index];
        if (!ray_aabb(origin, inv, t_min, limit, node)) {
            continue;
        }
        if (node.right == BVH_LEAF) {
            if (node.left != BVH_LEAF) {
                let t = ray_tri(origin, dir, node.left, t_min, limit);
                if (t >= t_min && (best < 0.0 || t < best)) {
                    best = t;
                    limit = t;
                }
            }
        } else {
            if (sp + 2 < BVH_STACK) {
                sp = sp + 1;
                stack[sp] = node.left;
                sp = sp + 1;
                stack[sp] = node.right;
            }
        }
    }
    return best;
}

fn bvh_collect(
    origin_u: vec3<f32>,
    dir: vec3<f32>,
    body_scale: f32,
    launch_offset: f32,
    origin_abs: f32,
    merge_abs: f32,
    merge_rel: f32,
    merged: ptr<function, array<f32, 128>>,
    overflow: ptr<function, bool>,
) -> u32 {
    let shifted = origin_u + edge_shift(dir);
    var t_min = launch_offset;
    var n = 0u;
    var last = -1.0e30;
    *overflow = false;
    for (var step = 0u; step < 8192u; step = step + 1u) {
        let t = bvh_closest(shifted, dir, t_min);
        if (!(t >= t_min)) {
            return n;
        }
        let nxt = next_up(t);
        if (!isfinite_f(t) || !(nxt > t)) {
            return n;
        }
        t_min = nxt;
        let world = t * body_scale;
        if (!(isfinite_f(world) && world > origin_abs)) {
            continue;
        }
        if (n > 0u && (world - last) <= (merge_abs + merge_rel * world)) {
            continue;
        }
        if (n >= 128u) {
            *overflow = true;
            return n;
        }
        (*merged)[n] = world;
        last = world;
        n = n + 1u;
    }
    *overflow = true;
    return n;
}
