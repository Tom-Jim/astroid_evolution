// GPU LBVH for the wasm software ray tracer.
//
// 63-bit Morton codes (21 bits per axis), a stable 8-bit LSD radix sort,
// Karras' binary radix tree with index tie-breaks for duplicate codes, then
// a device-scope atomic bottom-up AABB fit. macOS does not compile this.

struct U64 {
    lo: u32,
    hi: u32,
}

struct LbvParams {
    n: u32,
    groups: u32,
    digit: u32,
    pad: u32,
}

const LBVH_NONE: u32 = 0xffffffffu;
const ORDERED_POS_INF: u32 = 0xff800000u;
const ORDERED_NEG_INF: u32 = 0x007fffffu;

fn u64_or(a: U64, b: U64) -> U64 {
    return U64(a.lo | b.lo, a.hi | b.hi);
}

fn u64_and(a: U64, b: U64) -> U64 {
    return U64(a.lo & b.lo, a.hi & b.hi);
}

fn u64_shl(a: U64, s: u32) -> U64 {
    if (s == 0u) {
        return a;
    }
    if (s >= 32u) {
        return U64(0u, a.lo << (s - 32u));
    }
    return U64(a.lo << s, (a.hi << s) | (a.lo >> (32u - s)));
}

fn part1by2(v: u32) -> U64 {
    var x = U64(v & 0x001fffffu, 0u);
    x = u64_and(u64_or(x, u64_shl(x, 32u)), U64(0x0000ffffu, 0x001f0000u));
    x = u64_and(u64_or(x, u64_shl(x, 16u)), U64(0xff0000ffu, 0x001f0000u));
    x = u64_and(u64_or(x, u64_shl(x, 8u)), U64(0x0f00f00fu, 0x100f00f0u));
    x = u64_and(u64_or(x, u64_shl(x, 4u)), U64(0xc30c30c3u, 0x10c30c30u));
    x = u64_and(u64_or(x, u64_shl(x, 2u)), U64(0x49249249u, 0x12492492u));
    return x;
}

fn morton3(x: u32, y: u32, z: u32) -> U64 {
    return u64_or(part1by2(x), u64_or(u64_shl(part1by2(y), 1u), u64_shl(part1by2(z), 2u)));
}

fn quantize(c: f32, lo: f32, hi: f32) -> u32 {
    let extent = hi - lo;
    if (!(extent > 0.0)) {
        return 0u;
    }
    let t = clamp((c - lo) / extent, 0.0, 1.0);
    return min(u32(t * 2097151.0), 2097151u);
}

fn digit_of(lo: u32, hi: u32, slot: u32) -> u32 {
    if (slot < 4u) {
        return (lo >> (slot * 8u)) & 255u;
    }
    return (hi >> ((slot - 4u) * 8u)) & 255u;
}

fn clz64(hi: u32, lo: u32) -> i32 {
    if (hi != 0u) {
        return i32(countLeadingZeros(hi));
    }
    return 32 + i32(countLeadingZeros(lo));
}

fn prefix_len(ahi: u32, alo: u32, bhi: u32, blo: u32, i: i32, j: i32) -> i32 {
    let xhi = ahi ^ bhi;
    let xlo = alo ^ blo;
    if (xhi != 0u || xlo != 0u) {
        return clz64(xhi, xlo);
    }
    return 64 + i32(countLeadingZeros(u32(i) ^ u32(j)));
}

fn ordered_from_float(f: f32) -> u32 {
    let bits = bitcast<u32>(f);
    if ((bits & 0x80000000u) != 0u) {
        return ~bits;
    }
    return bits | 0x80000000u;
}

fn float_from_ordered(bits: u32) -> f32 {
    if ((bits & 0x80000000u) != 0u) {
        return bitcast<f32>(bits & 0x7fffffffu);
    }
    return bitcast<f32>(~bits);
}

fn expand_axis(lo: f32, hi: f32) -> vec2<f32> {
    let pad = max(abs(hi - lo), 1.0e-6) * 1.0e-4 + 1.0e-6;
    return vec2<f32>(lo - pad, hi + pad);
}

// ENTRY lbvh_bounds
@group(0) @binding(0) var<storage, read> tris: array<f32>;
@group(0) @binding(1) var<storage, read_write> aabb: array<f32>;
@group(0) @binding(2) var<storage, read_write> centroid: array<f32>;
@group(0) @binding(3) var<storage, read> params: LbvParams;
@compute @workgroup_size(256)
fn lbvh_bounds(@builtin(global_invocation_id) gid3: vec3<u32>) {
    let gid = gid3.x;
    if (gid >= params.n) {
        return;
    }
    let base = gid * 9u;
    let a = vec3<f32>(tris[base], tris[base + 1u], tris[base + 2u]);
    let b = vec3<f32>(tris[base + 3u], tris[base + 4u], tris[base + 5u]);
    let c = vec3<f32>(tris[base + 6u], tris[base + 7u], tris[base + 8u]);
    let mn = min(a, min(b, c));
    let mx = max(a, max(b, c));
    let ex = expand_axis(mn.x, mx.x);
    let ey = expand_axis(mn.y, mx.y);
    let ez = expand_axis(mn.z, mx.z);
    let dst = gid * 6u;
    aabb[dst] = ex.x;
    aabb[dst + 1u] = ey.x;
    aabb[dst + 2u] = ez.x;
    aabb[dst + 3u] = ex.y;
    aabb[dst + 4u] = ey.y;
    aabb[dst + 5u] = ez.y;
    let mid = (a + b + c) * (1.0 / 3.0);
    let cd = gid * 3u;
    centroid[cd] = mid.x;
    centroid[cd + 1u] = mid.y;
    centroid[cd + 2u] = mid.z;
}
// END ENTRY lbvh_bounds

// ENTRY lbvh_reduce
@group(0) @binding(0) var<storage, read> centroid_r: array<f32>;
@group(0) @binding(1) var<storage, read_write> partial: array<f32>;
@group(0) @binding(2) var<storage, read> params_r: LbvParams;
var<workgroup> red_min: array<vec3<f32>, 256>;
var<workgroup> red_max: array<vec3<f32>, 256>;
@compute @workgroup_size(256)
fn lbvh_reduce(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let tid = lid.x;
    let gid = wid.x * 256u + tid;
    var bmin = vec3<f32>(1.0e20, 1.0e20, 1.0e20);
    var bmax = vec3<f32>(-1.0e20, -1.0e20, -1.0e20);
    if (gid < params_r.n) {
        let cd = gid * 3u;
        bmin = vec3<f32>(centroid_r[cd], centroid_r[cd + 1u], centroid_r[cd + 2u]);
        bmax = bmin;
    }
    red_min[tid] = bmin;
    red_max[tid] = bmax;
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride = stride / 2u) {
        if (tid < stride) {
            red_min[tid] = min(red_min[tid], red_min[tid + stride]);
            red_max[tid] = max(red_max[tid], red_max[tid + stride]);
        }
        workgroupBarrier();
    }
    if (tid == 0u) {
        let dst = wid.x * 6u;
        partial[dst] = red_min[0].x;
        partial[dst + 1u] = red_min[0].y;
        partial[dst + 2u] = red_min[0].z;
        partial[dst + 3u] = red_max[0].x;
        partial[dst + 4u] = red_max[0].y;
        partial[dst + 5u] = red_max[0].z;
    }
}
// END ENTRY lbvh_reduce

// ENTRY lbvh_scene
@group(0) @binding(0) var<storage, read> partial_s: array<f32>;
@group(0) @binding(1) var<storage, read_write> scene: array<f32>;
@group(0) @binding(2) var<storage, read> params_s: LbvParams;
var<workgroup> scene_min: array<vec3<f32>, 256>;
var<workgroup> scene_max: array<vec3<f32>, 256>;
@compute @workgroup_size(256)
fn lbvh_scene(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    var bmin = vec3<f32>(1.0e20, 1.0e20, 1.0e20);
    var bmax = vec3<f32>(-1.0e20, -1.0e20, -1.0e20);
    for (var g = tid; g < params_s.groups; g = g + 256u) {
        let src = g * 6u;
        bmin = min(bmin, vec3<f32>(partial_s[src], partial_s[src + 1u], partial_s[src + 2u]));
        bmax = max(bmax, vec3<f32>(partial_s[src + 3u], partial_s[src + 4u], partial_s[src + 5u]));
    }
    scene_min[tid] = bmin;
    scene_max[tid] = bmax;
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride = stride / 2u) {
        if (tid < stride) {
            scene_min[tid] = min(scene_min[tid], scene_min[tid + stride]);
            scene_max[tid] = max(scene_max[tid], scene_max[tid + stride]);
        }
        workgroupBarrier();
    }
    if (tid == 0u) {
        scene[0] = scene_min[0].x;
        scene[1] = scene_min[0].y;
        scene[2] = scene_min[0].z;
        scene[3] = scene_max[0].x;
        scene[4] = scene_max[0].y;
        scene[5] = scene_max[0].z;
    }
}
// END ENTRY lbvh_scene

// ENTRY lbvh_morton
@group(0) @binding(0) var<storage, read> centroid_m: array<f32>;
@group(0) @binding(1) var<storage, read> scene_m: array<f32>;
@group(0) @binding(2) var<storage, read_write> keys_lo: array<u32>;
@group(0) @binding(3) var<storage, read_write> keys_hi: array<u32>;
@group(0) @binding(4) var<storage, read_write> order: array<u32>;
@group(0) @binding(5) var<storage, read> params_m: LbvParams;
@compute @workgroup_size(256)
fn lbvh_morton(@builtin(global_invocation_id) gid3: vec3<u32>) {
    let gid = gid3.x;
    if (gid >= params_m.n) {
        return;
    }
    let cd = gid * 3u;
    let code = morton3(
        quantize(centroid_m[cd], scene_m[0], scene_m[3]),
        quantize(centroid_m[cd + 1u], scene_m[1], scene_m[4]),
        quantize(centroid_m[cd + 2u], scene_m[2], scene_m[5]),
    );
    keys_lo[gid] = code.lo;
    keys_hi[gid] = code.hi;
    order[gid] = gid;
}
// END ENTRY lbvh_morton

// ENTRY lbvh_hist
@group(0) @binding(0) var<storage, read> hist_lo: array<u32>;
@group(0) @binding(1) var<storage, read> hist_hi: array<u32>;
@group(0) @binding(2) var<storage, read_write> hist: array<u32>;
@group(0) @binding(3) var<storage, read> params_h: LbvParams;
var<workgroup> hist_bins: array<atomic<u32>, 256>;
@compute @workgroup_size(256)
fn lbvh_hist(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let tid = lid.x;
    atomicStore(&hist_bins[tid], 0u);
    workgroupBarrier();
    let gid = wid.x * 256u + tid;
    if (gid < params_h.n) {
        let bin = digit_of(hist_lo[gid], hist_hi[gid], params_h.digit);
        atomicAdd(&hist_bins[bin], 1u);
    }
    workgroupBarrier();
    hist[wid.x * 256u + tid] = atomicLoad(&hist_bins[tid]);
}
// END ENTRY lbvh_hist

// ENTRY lbvh_bin_total
@group(0) @binding(0) var<storage, read> hist_t: array<u32>;
@group(0) @binding(1) var<storage, read_write> bin_total: array<u32>;
@group(0) @binding(2) var<storage, read> params_t: LbvParams;
@compute @workgroup_size(256)
fn lbvh_bin_total(@builtin(local_invocation_id) lid: vec3<u32>) {
    let bin = lid.x;
    var sum = 0u;
    for (var g = 0u; g < params_t.groups; g = g + 1u) {
        sum = sum + hist_t[g * 256u + bin];
    }
    bin_total[bin] = sum;
}
// END ENTRY lbvh_bin_total

// ENTRY lbvh_bin_scan
@group(0) @binding(0) var<storage, read> total_in: array<u32>;
@group(0) @binding(1) var<storage, read_write> bin_start: array<u32>;
var<workgroup> scan_bin: array<u32, 256>;
@compute @workgroup_size(256)
fn lbvh_bin_scan(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    let value = total_in[tid];
    scan_bin[tid] = value;
    workgroupBarrier();
    for (var offset = 1u; offset < 256u; offset = offset * 2u) {
        var next = scan_bin[tid];
        if (tid >= offset) {
            next = scan_bin[tid] + scan_bin[tid - offset];
        }
        workgroupBarrier();
        scan_bin[tid] = next;
        workgroupBarrier();
    }
    bin_start[tid] = scan_bin[tid] - value;
}
// END ENTRY lbvh_bin_scan

// ENTRY lbvh_tile_offset
@group(0) @binding(0) var<storage, read> hist_o: array<u32>;
@group(0) @binding(1) var<storage, read> start_o: array<u32>;
@group(0) @binding(2) var<storage, read_write> tile_offset: array<u32>;
@group(0) @binding(3) var<storage, read> params_o: LbvParams;
@compute @workgroup_size(256)
fn lbvh_tile_offset(@builtin(local_invocation_id) lid: vec3<u32>) {
    let bin = lid.x;
    var running = start_o[bin];
    for (var g = 0u; g < params_o.groups; g = g + 1u) {
        tile_offset[g * 256u + bin] = running;
        running = running + hist_o[g * 256u + bin];
    }
}
// END ENTRY lbvh_tile_offset

// ENTRY lbvh_scatter
@group(0) @binding(0) var<storage, read> in_lo: array<u32>;
@group(0) @binding(1) var<storage, read> in_hi: array<u32>;
@group(0) @binding(2) var<storage, read> in_order: array<u32>;
@group(0) @binding(3) var<storage, read_write> out_lo: array<u32>;
@group(0) @binding(4) var<storage, read_write> out_hi: array<u32>;
@group(0) @binding(5) var<storage, read_write> out_order: array<u32>;
@group(0) @binding(6) var<storage, read> tile_s: array<u32>;
@group(0) @binding(7) var<storage, read> params_sc: LbvParams;
var<workgroup> sc_bin: array<u32, 256>;
var<workgroup> sc_lo: array<u32, 256>;
var<workgroup> sc_hi: array<u32, 256>;
var<workgroup> sc_order: array<u32, 256>;
@compute @workgroup_size(256)
fn lbvh_scatter(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let tid = lid.x;
    let gid = wid.x * 256u + tid;
    sc_bin[tid] = 0xffffffffu;
    if (gid < params_sc.n) {
        sc_lo[tid] = in_lo[gid];
        sc_hi[tid] = in_hi[gid];
        sc_order[tid] = in_order[gid];
        sc_bin[tid] = digit_of(sc_lo[tid], sc_hi[tid], params_sc.digit);
    }
    workgroupBarrier();
    if (gid < params_sc.n) {
        var rank = 0u;
        for (var i = 0u; i < tid; i = i + 1u) {
            if (sc_bin[i] == sc_bin[tid]) {
                rank = rank + 1u;
            }
        }
        let dest = tile_s[wid.x * 256u + sc_bin[tid]] + rank;
        out_lo[dest] = sc_lo[tid];
        out_hi[dest] = sc_hi[tid];
        out_order[dest] = sc_order[tid];
    }
}
// END ENTRY lbvh_scatter

// ENTRY lbvh_clear_parent
@group(0) @binding(0) var<storage, read_write> parent_c: array<u32>;
@group(0) @binding(1) var<storage, read> params_c: LbvParams;
@compute @workgroup_size(256)
fn lbvh_clear_parent(@builtin(global_invocation_id) gid3: vec3<u32>) {
    let gid = gid3.x;
    let count = params_c.n * 2u - 1u;
    if (gid < count) {
        parent_c[gid] = LBVH_NONE;
    }
}
// END ENTRY lbvh_clear_parent

// ENTRY lbvh_zero
@group(0) @binding(0) var<storage, read_write> words_z: array<u32>;
@group(0) @binding(1) var<storage, read> params_z: LbvParams;
@compute @workgroup_size(256)
fn lbvh_zero(@builtin(global_invocation_id) gid3: vec3<u32>) {
    let gid = gid3.x;
    let count = (params_z.n * 2u - 1u) * 8u;
    if (gid < count) {
        words_z[gid] = 0u;
    }
}
// END ENTRY lbvh_zero

// ENTRY lbvh_leaves
@group(0) @binding(0) var<storage, read> aabb_l: array<f32>;
@group(0) @binding(1) var<storage, read> order_l: array<u32>;
@group(0) @binding(2) var<storage, read_write> words_l: array<u32>;
@group(0) @binding(3) var<storage, read> params_l: LbvParams;
@compute @workgroup_size(256)
fn lbvh_leaves(@builtin(global_invocation_id) gid3: vec3<u32>) {
    let sorted = gid3.x;
    if (sorted >= params_l.n) {
        return;
    }
    let prim = order_l[sorted];
    let src = prim * 6u;
    let node = (params_l.n - 1u + sorted) * 8u;
    words_l[node] = bitcast<u32>(aabb_l[src]);
    words_l[node + 1u] = bitcast<u32>(aabb_l[src + 1u]);
    words_l[node + 2u] = bitcast<u32>(aabb_l[src + 2u]);
    words_l[node + 3u] = prim;
    words_l[node + 4u] = bitcast<u32>(aabb_l[src + 3u]);
    words_l[node + 5u] = bitcast<u32>(aabb_l[src + 4u]);
    words_l[node + 6u] = bitcast<u32>(aabb_l[src + 5u]);
    words_l[node + 7u] = LBVH_NONE;
}
// END ENTRY lbvh_leaves

// ENTRY lbvh_karras
@group(0) @binding(0) var<storage, read> k_lo: array<u32>;
@group(0) @binding(1) var<storage, read> k_hi: array<u32>;
@group(0) @binding(2) var<storage, read_write> words_k: array<u32>;
@group(0) @binding(3) var<storage, read_write> parent_k: array<u32>;
@group(0) @binding(4) var<storage, read> params_k: LbvParams;

fn delta_at(i: i32, j: i32, n: i32) -> i32 {
    if (i < 0 || j < 0 || i >= n || j >= n) {
        return -1;
    }
    return prefix_len(k_hi[u32(i)], k_lo[u32(i)], k_hi[u32(j)], k_lo[u32(j)], i, j);
}

@compute @workgroup_size(256)
fn lbvh_karras(@builtin(global_invocation_id) gid3: vec3<u32>) {
    let gid = gid3.x;
    let n = i32(params_k.n);
    if (n < 2 || gid >= params_k.n - 1u) {
        return;
    }
    let i = i32(gid);
    let d = select(i32(-1), i32(1), delta_at(i, i + 1, n) - delta_at(i, i - 1, n) >= 0);
    let delta_min = delta_at(i, i - d, n);
    var l_max = 2;
    var guard = 0;
    loop {
        if (!(delta_at(i, i + l_max * d, n) > delta_min)) {
            break;
        }
        l_max = l_max * 2;
        guard = guard + 1;
        if (guard > 31) {
            break;
        }
    }
    var len = 0;
    var step = l_max / 2;
    loop {
        if (step < 1) {
            break;
        }
        if (delta_at(i, i + (len + step) * d, n) > delta_min) {
            len = len + step;
        }
        step = step / 2;
    }
    let j = i + len * d;
    let delta_node = delta_at(i, j, n);
    var split = 0;
    var span = len;
    loop {
        span = (span + 1) / 2;
        if (delta_at(i, i + (split + span) * d, n) > delta_node) {
            split = split + span;
        }
        if (span <= 1) {
            break;
        }
    }
    let gamma = i + split * d + min(d, i32(0));
    var left_child = u32(gamma);
    if (min(i, j) == gamma) {
        left_child = (params_k.n - 1u) + u32(gamma);
    }
    var right_child = u32(gamma + 1);
    if (max(i, j) == gamma + 1) {
        right_child = (params_k.n - 1u) + u32(gamma + 1);
    }
    let slot = gid * 8u;
    words_k[slot + 3u] = left_child;
    words_k[slot + 7u] = right_child;
    parent_k[left_child] = gid;
    parent_k[right_child] = gid;
}
// END ENTRY lbvh_karras

// ENTRY lbvh_init_aabb
@group(0) @binding(0) var<storage, read_write> words_i: array<u32>;
@group(0) @binding(1) var<storage, read> params_i: LbvParams;
@compute @workgroup_size(256)
fn lbvh_init_aabb(@builtin(global_invocation_id) gid3: vec3<u32>) {
    let gid = gid3.x;
    if (gid >= params_i.n - 1u) {
        return;
    }
    let node = gid * 8u;
    words_i[node] = ORDERED_POS_INF;
    words_i[node + 1u] = ORDERED_POS_INF;
    words_i[node + 2u] = ORDERED_POS_INF;
    words_i[node + 4u] = ORDERED_NEG_INF;
    words_i[node + 5u] = ORDERED_NEG_INF;
    words_i[node + 6u] = ORDERED_NEG_INF;
}
// END ENTRY lbvh_init_aabb

// ENTRY lbvh_root
@group(0) @binding(0) var<storage, read> parent_r: array<u32>;
@group(0) @binding(1) var<storage, read_write> root_r: array<u32>;
@group(0) @binding(2) var<storage, read> params_root: LbvParams;
@compute @workgroup_size(256)
fn lbvh_root(@builtin(global_invocation_id) gid3: vec3<u32>) {
    if (gid3.x != 0u) {
        return;
    }
    root_r[0] = 0u;
    let internal = params_root.n - 1u;
    for (var i = 0u; i < internal; i = i + 1u) {
        if (parent_r[i] == LBVH_NONE) {
            root_r[0] = i;
            return;
        }
    }
}
// END ENTRY lbvh_root

// ENTRY lbvh_fit
@group(0) @binding(0) var<storage, read_write> words_f: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read> parent_f: array<u32>;
@group(0) @binding(2) var<storage, read> params_f: LbvParams;
@compute @workgroup_size(256)
fn lbvh_fit(@builtin(global_invocation_id) gid3: vec3<u32>) {
    let gid = gid3.x;
    if (gid >= params_f.n) {
        return;
    }
    var current = params_f.n - 1u + gid;
    let leaf = current * 8u;
    var omin = vec3<u32>(
        ordered_from_float(bitcast<f32>(atomicLoad(&words_f[leaf]))),
        ordered_from_float(bitcast<f32>(atomicLoad(&words_f[leaf + 1u]))),
        ordered_from_float(bitcast<f32>(atomicLoad(&words_f[leaf + 2u]))),
    );
    var omax = vec3<u32>(
        ordered_from_float(bitcast<f32>(atomicLoad(&words_f[leaf + 4u]))),
        ordered_from_float(bitcast<f32>(atomicLoad(&words_f[leaf + 5u]))),
        ordered_from_float(bitcast<f32>(atomicLoad(&words_f[leaf + 6u]))),
    );
    for (var hop = 0u; hop < params_f.n; hop = hop + 1u) {
        let p = parent_f[current];
        if (p == LBVH_NONE) {
            return;
        }
        let base = p * 8u;
        atomicMin(&words_f[base], omin.x);
        atomicMin(&words_f[base + 1u], omin.y);
        atomicMin(&words_f[base + 2u], omin.z);
        atomicMax(&words_f[base + 4u], omax.x);
        atomicMax(&words_f[base + 5u], omax.y);
        atomicMax(&words_f[base + 6u], omax.z);
        current = p;
    }
}
// END ENTRY lbvh_fit

// ENTRY lbvh_convert
@group(0) @binding(0) var<storage, read_write> words_v: array<u32>;
@group(0) @binding(1) var<storage, read> params_v: LbvParams;
@compute @workgroup_size(256)
fn lbvh_convert(@builtin(global_invocation_id) gid3: vec3<u32>) {
    let gid = gid3.x;
    if (gid >= params_v.n - 1u) {
        return;
    }
    let node = gid * 8u;
    words_v[node] = bitcast<u32>(float_from_ordered(words_v[node]));
    words_v[node + 1u] = bitcast<u32>(float_from_ordered(words_v[node + 1u]));
    words_v[node + 2u] = bitcast<u32>(float_from_ordered(words_v[node + 2u]));
    words_v[node + 4u] = bitcast<u32>(float_from_ordered(words_v[node + 4u]));
    words_v[node + 5u] = bitcast<u32>(float_from_ordered(words_v[node + 5u]));
    words_v[node + 6u] = bitcast<u32>(float_from_ordered(words_v[node + 6u]));
}
// END ENTRY lbvh_convert

// ENTRY lbvh_swap_root
@group(0) @binding(0) var<storage, read_write> words_w: array<u32>;
@group(0) @binding(1) var<storage, read> root_w: array<u32>;
@compute @workgroup_size(256)
fn lbvh_swap_root(@builtin(global_invocation_id) gid3: vec3<u32>) {
    if (gid3.x != 0u) {
        return;
    }
    let root = root_w[0];
    if (root == 0u) {
        return;
    }
    let src = root * 8u;
    for (var i = 0u; i < 8u; i = i + 1u) {
        let tmp = words_w[i];
        words_w[i] = words_w[src + i];
        words_w[src + i] = tmp;
    }
}
// END ENTRY lbvh_swap_root

// ENTRY lbvh_patch
@group(0) @binding(0) var<storage, read_write> words_p: array<u32>;
@group(0) @binding(1) var<storage, read> root_p: array<u32>;
@group(0) @binding(2) var<storage, read> params_p: LbvParams;

fn retarget(child: u32, root: u32) -> u32 {
    if (child == 0u) {
        return root;
    }
    if (child == root) {
        return 0u;
    }
    return child;
}

@compute @workgroup_size(256)
fn lbvh_patch(@builtin(global_invocation_id) gid3: vec3<u32>) {
    let gid = gid3.x;
    let root = root_p[0];
    if (root == 0u || gid >= params_p.n - 1u) {
        return;
    }
    let node = gid * 8u;
    words_p[node + 3u] = retarget(words_p[node + 3u], root);
    words_p[node + 7u] = retarget(words_p[node + 7u], root);
}
// END ENTRY lbvh_patch
