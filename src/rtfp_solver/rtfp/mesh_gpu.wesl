// GPU mesh load, radial-cone alignment, and self-intersection for the wasm
// viewer. macOS keeps the f64 CPU implementations and does not compile this.

struct LoadParams {
    pos_count: u32,
    pos_offset: u32,
    pos_stride: u32,
    index_count: u32,
    index_offset: u32,
    index_stride: u32,
    index_width: u32,
    pad: u32,
}

struct MeshParams {
    nvert: u32,
    nface: u32,
    cells: u32,
    cap: u32,
    side: u32,
    scale_bits: u32,
    vert_groups: u32,
    face_groups: u32,
    origin_x: u32,
    origin_y: u32,
    origin_z: u32,
    extent_bits: u32,
}

struct TriHit {
    hit: u32,
    gap: f32,
}

fn bits_f32(bits: u32) -> f32 {
    return bitcast<f32>(bits);
}

fn finite_f(x: f32) -> bool {
    return (bitcast<u32>(x) & 0x7fffffffu) < 0x7f800000u;
}

fn shared_n(a0: u32, a1: u32, a2: u32, b0: u32, b1: u32, b2: u32) -> u32 {
    var n = 0u;
    let a = array<u32, 3>(a0, a1, a2);
    let b = array<u32, 3>(b0, b1, b2);
    for (var i = 0u; i < 3u; i = i + 1u) {
        for (var j = 0u; j < 3u; j = j + 1u) {
            if (a[i] == b[j]) {
                n = n + 1u;
            }
        }
    }
    return n;
}

struct ds2 {
    x: ds,
    y: ds,
}

struct ds3 {
    x: ds,
    y: ds,
    z: ds,
}

fn ds_max(a: ds, b: ds) -> ds {
    if (ds_gt(a, b)) { return a; }
    return b;
}

fn ds_min(a: ds, b: ds) -> ds {
    if (ds_gt(a, b)) { return b; }
    return a;
}

fn ds_clamp(x: ds, lo: ds, hi: ds) -> ds {
    var y = x;
    if (ds_gt(lo, y)) { y = lo; }
    if (ds_gt(y, hi)) { y = hi; }
    return y;
}

fn d3_add(a: ds3, b: ds3) -> ds3 {
    return ds3(ds_add(a.x, b.x), ds_add(a.y, b.y), ds_add(a.z, b.z));
}

fn d3_sub(a: ds3, b: ds3) -> ds3 {
    return ds3(ds_sub(a.x, b.x), ds_sub(a.y, b.y), ds_sub(a.z, b.z));
}

fn d3_scale(a: ds3, s: ds) -> ds3 {
    return ds3(ds_mul(a.x, s), ds_mul(a.y, s), ds_mul(a.z, s));
}

fn d3_dot(a: ds3, b: ds3) -> ds {
    return ds_add(ds_add(ds_mul(a.x, b.x), ds_mul(a.y, b.y)), ds_mul(a.z, b.z));
}

fn d3_cross(a: ds3, b: ds3) -> ds3 {
    return ds3(
        ds_sub(ds_mul(a.y, b.z), ds_mul(a.z, b.y)),
        ds_sub(ds_mul(a.z, b.x), ds_mul(a.x, b.z)),
        ds_sub(ds_mul(a.x, b.y), ds_mul(a.y, b.x)),
    );
}

fn d3_len(a: ds3) -> ds {
    return ds_sqrt(d3_dot(a, a));
}

fn norm_ds(v: ds3) -> ds3 {
    let len = d3_len(v);
    if (!ds_gt(len, ds_zero())) {
        return ds3(ds_zero(), ds_zero(), ds_f(1.0));
    }
    return d3_scale(v, ds_recip(len));
}

fn axis_of(n: ds3) -> u32 {
    let ax = ds_abs(n.x);
    let ay = ds_abs(n.y);
    let az = ds_abs(n.z);
    if (!ds_gt(ay, ax) && !ds_gt(az, ax)) {
        return 0u;
    }
    if (!ds_gt(az, ay)) {
        return 1u;
    }
    return 2u;
}

fn drop_axis(v: ds3, axis: u32) -> ds2 {
    if (axis == 0u) { return ds2(v.y, v.z); }
    if (axis == 1u) { return ds2(v.x, v.z); }
    return ds2(v.x, v.y);
}

fn orient2(a: ds2, b: ds2, c: ds2) -> ds {
    return ds_sub(
        ds_mul(ds_sub(b.x, a.x), ds_sub(c.y, a.y)),
        ds_mul(ds_sub(b.y, a.y), ds_sub(c.x, a.x)),
    );
}

fn proper2d(a: ds2, b: ds2, c: ds2, d: ds2, eps: ds) -> bool {
    let left = ds_mul(orient2(a, b, c), orient2(a, b, d));
    let right = ds_mul(orient2(c, d, a), orient2(c, d, b));
    let neg = ds_neg(eps);
    return ds_gt(neg, left) && ds_gt(neg, right);
}

fn bary2(p: ds2, a: ds2, b: ds2, c: ds2) -> ds3 {
    let den = ds_add(
        ds_mul(ds_sub(b.y, c.y), ds_sub(a.x, c.x)),
        ds_mul(ds_sub(c.x, b.x), ds_sub(a.y, c.y)),
    );
    if (!ds_gt(ds_abs(den), ds_f(1e-18))) {
        return ds3(ds_zero(), ds_zero(), ds_zero());
    }
    let w0 = ds_div(
        ds_add(ds_mul(ds_sub(b.y, c.y), ds_sub(p.x, c.x)), ds_mul(ds_sub(c.x, b.x), ds_sub(p.y, c.y))),
        den,
    );
    let w1 = ds_div(
        ds_add(ds_mul(ds_sub(c.y, a.y), ds_sub(p.x, c.x)), ds_mul(ds_sub(a.x, c.x), ds_sub(p.y, c.y))),
        den,
    );
    return ds3(w0, w1, ds_sub(ds_sub(ds_f(1.0), w0), w1));
}

fn inside_tri(p: ds3, a: ds3, b: ds3, c: ds3, eps: ds) -> bool {
    let n = d3_cross(d3_sub(b, a), d3_sub(c, a));
    let axis = axis_of(n);
    let w = bary2(drop_axis(p, axis), drop_axis(a, axis), drop_axis(b, axis), drop_axis(c, axis));
    return ds_gt(w.x, eps) && ds_gt(w.y, eps) && ds_gt(w.z, eps);
}

fn seg_cross3(a: ds3, b: ds3, c: ds3, d: ds3, eps: ds) -> bool {
    let n = d3_cross(d3_sub(b, a), d3_sub(d, c));
    let nlen = d3_len(n);
    if (!ds_gt(nlen, eps)) {
        return false;
    }
    if (ds_gt(ds_abs(d3_dot(n, d3_sub(c, a))), ds_mul(eps, nlen))) {
        return false;
    }
    let axis = axis_of(n);
    return proper2d(drop_axis(a, axis), drop_axis(b, axis), drop_axis(c, axis), drop_axis(d, axis), eps);
}

fn edge_dist(point: ds3, p: ds3, q: ds3) -> ds {
    let side = d3_sub(q, p);
    let len = d3_len(side);
    if (!ds_gt(len, ds_zero())) {
        return ds_f(1e20);
    }
    return ds_div(d3_len(d3_cross(d3_sub(point, p), side)), len);
}

fn boundary_gap(point: ds3, a: ds3, b: ds3, c: ds3) -> ds {
    return ds_min(edge_dist(point, a, b), ds_min(edge_dist(point, b, c), edge_dist(point, c, a)));
}

fn seg_gap(p: ds3, q: ds3, a: ds3, b: ds3, c: ds3, eps: ds) -> TriHit {
    let miss = TriHit(0u, 0.0);
    let dir = d3_sub(q, p);
    let e1 = d3_sub(b, a);
    let e2 = d3_sub(c, a);
    let pv = d3_cross(dir, e2);
    let det = d3_dot(e1, pv);
    let scale = ds_mul(ds_mul(d3_len(e1), d3_len(e2)), d3_len(dir));
    if (!ds_gt(scale, ds_zero()) || !ds_gt(ds_abs(det), ds_mul(ds_f(1e-12), scale))) {
        return miss;
    }
    let inv = ds_recip(det);
    let tvec = d3_sub(p, a);
    let u = ds_mul(d3_dot(tvec, pv), inv);
    let qv = d3_cross(tvec, e1);
    let v = ds_mul(d3_dot(dir, qv), inv);
    let t = ds_mul(d3_dot(e2, qv), inv);
    let margin = ds_clamp(ds_div(eps, ds_max(scale, eps)), ds_f(1e-8), ds_f(1e-3));
    let upper = ds_sub(ds_f(1.0), margin);
    if (ds_gt(t, margin) && ds_gt(upper, t) && ds_gt(u, margin) && ds_gt(v, margin) && ds_gt(upper, ds_add(u, v))) {
        let point = d3_add(p, d3_scale(dir, t));
        return TriHit(1u, ds_float(boundary_gap(point, a, b, c)));
    }
    return miss;
}

fn nearer(a: TriHit, b: TriHit) -> TriHit {
    if (b.hit == 0u) {
        return a;
    }
    if (a.hit == 0u || b.gap < a.gap) {
        return b;
    }
    return a;
}

fn tri_hit(a0: ds3, a1: ds3, a2: ds3, b0: ds3, b1: ds3, b2: ds3, eps: ds) -> TriHit {
    let miss = TriHit(0u, 0.0);
    let na = d3_cross(d3_sub(a1, a0), d3_sub(a2, a0));
    let nb = d3_cross(d3_sub(b1, b0), d3_sub(b2, b0));
    let a_scale = ds_max(d3_len(na), eps);
    let b_scale = ds_max(d3_len(nb), eps);
    let da0 = d3_dot(na, d3_sub(b0, a0));
    let da1 = d3_dot(na, d3_sub(b1, a0));
    let da2 = d3_dot(na, d3_sub(b2, a0));
    let db0 = d3_dot(nb, d3_sub(a0, b0));
    let db1 = d3_dot(nb, d3_sub(a1, b0));
    let db2 = d3_dot(nb, d3_sub(a2, b0));
    let a_band = ds_mul(ds_f(1e-9), a_scale);
    let b_band = ds_mul(ds_f(1e-9), b_scale);
    let a_coplanar = !ds_gt(ds_abs(da0), a_band) && !ds_gt(ds_abs(da1), a_band) && !ds_gt(ds_abs(da2), a_band);
    let b_coplanar = !ds_gt(ds_abs(db0), b_band) && !ds_gt(ds_abs(db1), b_band) && !ds_gt(ds_abs(db2), b_band);
    if (a_coplanar && b_coplanar) {
        if (inside_tri(b0, a0, a1, a2, eps) || inside_tri(b1, a0, a1, a2, eps) || inside_tri(b2, a0, a1, a2, eps) || inside_tri(a0, b0, b1, b2, eps) || inside_tri(a1, b0, b1, b2, eps) || inside_tri(a2, b0, b1, b2, eps) || seg_cross3(a0, a1, b0, b1, eps) || seg_cross3(a0, a1, b1, b2, eps) || seg_cross3(a0, a1, b2, b0, eps) || seg_cross3(a1, a2, b0, b1, eps) || seg_cross3(a1, a2, b1, b2, eps) || seg_cross3(a1, a2, b2, b0, eps) || seg_cross3(a2, a0, b0, b1, eps) || seg_cross3(a2, a0, b1, b2, eps) || seg_cross3(a2, a0, b2, b0, eps)) {
            return TriHit(1u, 0.0);
        }
        return miss;
    }
    let neg = ds_neg(eps);
    let a_pos = ds_gt(da0, eps) && ds_gt(da1, eps) && ds_gt(da2, eps);
    let a_neg = ds_gt(neg, da0) && ds_gt(neg, da1) && ds_gt(neg, da2);
    let b_pos = ds_gt(db0, eps) && ds_gt(db1, eps) && ds_gt(db2, eps);
    let b_neg = ds_gt(neg, db0) && ds_gt(neg, db1) && ds_gt(neg, db2);
    if (a_pos || a_neg || b_pos || b_neg) {
        return miss;
    }
    var best = miss;
    best = nearer(best, seg_gap(a0, a1, b0, b1, b2, eps));
    best = nearer(best, seg_gap(a1, a2, b0, b1, b2, eps));
    best = nearer(best, seg_gap(a2, a0, b0, b1, b2, eps));
    best = nearer(best, seg_gap(b0, b1, a0, a1, a2, eps));
    best = nearer(best, seg_gap(b1, b2, a0, a1, a2, eps));
    best = nearer(best, seg_gap(b2, b0, a0, a1, a2, eps));
    return best;
}

fn slot_of(ix: u32, iy: u32, iz: u32, side: u32) -> u32 {
    return (ix * side + iy) * side + iz;
}

// ENTRY mesh_load_pos
@group(0) @binding(0) var<storage, read> bin_p: array<u32>;
@group(0) @binding(1) var<storage, read> load_p: LoadParams;
@group(0) @binding(2) var<storage, read> matrix_p: array<f32>;
@group(0) @binding(3) var<storage, read_write> out_p: array<f32>;
@group(0) @binding(4) var<storage, read_write> bad_p: array<atomic<u32>>;
fn word_p(offset: u32) -> u32 {
    let word = offset >> 2u;
    let shift = (offset & 3u) * 8u;
    if (shift == 0u) {
        return bin_p[word];
    }
    return (bin_p[word] >> shift) | (bin_p[word + 1u] << (32u - shift));
}
@compute @workgroup_size(256)
fn mesh_load_pos(@builtin(global_invocation_id) gid3: vec3<u32>) {
    let gid = gid3.x;
    if (gid >= load_p.pos_count) {
        return;
    }
    let off = load_p.pos_offset + gid * load_p.pos_stride;
    let x = bits_f32(word_p(off));
    let y = bits_f32(word_p(off + 4u));
    let z = bits_f32(word_p(off + 8u));
    if (!finite_f(x) || !finite_f(y) || !finite_f(z)) {
        atomicAdd(&bad_p[0], 1u);
    }
    let px = matrix_p[0] * x + matrix_p[4] * y + matrix_p[8] * z + matrix_p[12];
    let py = matrix_p[1] * x + matrix_p[5] * y + matrix_p[9] * z + matrix_p[13];
    let pz = matrix_p[2] * x + matrix_p[6] * y + matrix_p[10] * z + matrix_p[14];
    if (!finite_f(px) || !finite_f(py) || !finite_f(pz)) {
        atomicAdd(&bad_p[0], 1u);
    }
    out_p[gid * 3u] = px;
    out_p[gid * 3u + 1u] = py;
    out_p[gid * 3u + 2u] = pz;
}
// END ENTRY mesh_load_pos

// ENTRY mesh_load_idx
@group(0) @binding(0) var<storage, read> bin_i: array<u32>;
@group(0) @binding(1) var<storage, read> load_i: LoadParams;
@group(0) @binding(2) var<storage, read_write> out_i: array<u32>;
@group(0) @binding(3) var<storage, read_write> bad_i: array<atomic<u32>>;
fn word_i(offset: u32) -> u32 {
    let word = offset >> 2u;
    let shift = (offset & 3u) * 8u;
    if (shift == 0u) {
        return bin_i[word];
    }
    return (bin_i[word] >> shift) | (bin_i[word + 1u] << (32u - shift));
}
@compute @workgroup_size(256)
fn mesh_load_idx(@builtin(global_invocation_id) gid3: vec3<u32>) {
    let gid = gid3.x;
    if (gid >= load_i.index_count) {
        return;
    }
    let bits = word_i(load_i.index_offset + gid * load_i.index_stride);
    var value = bits;
    if (load_i.index_width == 1u) {
        value = bits & 255u;
    } else if (load_i.index_width == 2u) {
        value = bits & 65535u;
    }
    if (value >= load_i.pos_count) {
        atomicAdd(&bad_i[1], 1u);
    }
    out_i[gid] = value;
}
// END ENTRY mesh_load_idx

// ENTRY mesh_zero_counts
@group(0) @binding(0) var<storage, read_write> counts_z: array<u32>;
@group(0) @binding(1) var<storage, read> params_zc: MeshParams;
@compute @workgroup_size(256)
fn mesh_zero_counts(@builtin(global_invocation_id) gid3: vec3<u32>) {
    let gid = gid3.x;
    if (gid < params_zc.nvert) {
        counts_z[gid] = 0u;
    }
}
// END ENTRY mesh_zero_counts

// ENTRY mesh_align_count
@group(0) @binding(0) var<storage, read> faces_c: array<u32>;
@group(0) @binding(1) var<storage, read_write> counts_c: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read> params_c: MeshParams;
@compute @workgroup_size(256)
fn mesh_align_count(@builtin(global_invocation_id) gid3: vec3<u32>) {
    let gid = gid3.x;
    if (gid >= params_c.nface) {
        return;
    }
    let base = gid * 3u;
    atomicAdd(&counts_c[faces_c[base]], 1u);
    atomicAdd(&counts_c[faces_c[base + 1u]], 1u);
    atomicAdd(&counts_c[faces_c[base + 2u]], 1u);
}
// END ENTRY mesh_align_count

// ENTRY mesh_align_scan
@group(0) @binding(0) var<storage, read> counts_s: array<u32>;
@group(0) @binding(1) var<storage, read_write> offsets_s: array<u32>;
@group(0) @binding(2) var<storage, read_write> cursors_s: array<u32>;
@group(0) @binding(3) var<storage, read> params_s: MeshParams;
@compute @workgroup_size(256)
fn mesh_align_scan(@builtin(global_invocation_id) gid3: vec3<u32>) {
    if (gid3.x != 0u) {
        return;
    }
    var sum = 0u;
    for (var i = 0u; i < params_s.nvert; i = i + 1u) {
        offsets_s[i] = sum;
        cursors_s[i] = sum;
        sum = sum + counts_s[i];
    }
    offsets_s[params_s.nvert] = sum;
}
// END ENTRY mesh_align_scan

// ENTRY mesh_align_fill
@group(0) @binding(0) var<storage, read> faces_f: array<u32>;
@group(0) @binding(1) var<storage, read_write> cursors_f: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> face_ids: array<u32>;
@group(0) @binding(3) var<storage, read> params_f: MeshParams;
@compute @workgroup_size(256)
fn mesh_align_fill(@builtin(global_invocation_id) gid3: vec3<u32>) {
    let gid = gid3.x;
    if (gid >= params_f.nface) {
        return;
    }
    let base = gid * 3u;
    let cap = params_f.nface * 3u;
    for (var k = 0u; k < 3u; k = k + 1u) {
        let v = faces_f[base + k];
        let slot = atomicAdd(&cursors_f[v], 1u);
        if (slot < cap) {
            face_ids[slot] = gid;
        }
    }
}
// END ENTRY mesh_align_fill

// ENTRY mesh_zero_flags
@group(0) @binding(0) var<storage, read_write> flags_z: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read> params_zf: MeshParams;
@compute @workgroup_size(256)
fn mesh_zero_flags(@builtin(global_invocation_id) gid3: vec3<u32>) {
    let gid = gid3.x;
    if (gid < params_zf.nvert) {
        atomicStore(&flags_z[gid], 0u);
    }
}
// END ENTRY mesh_zero_flags

// ENTRY mesh_align_test
@group(0) @binding(0) var<storage, read> faces_t: array<u32>;
@group(0) @binding(1) var<storage, read> pos_t: array<f32>;
@group(0) @binding(2) var<storage, read> offsets_t: array<u32>;
@group(0) @binding(3) var<storage, read> ids_t: array<u32>;
@group(0) @binding(4) var<storage, read_write> flags_t: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> bad_t: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read> params_t: MeshParams;
fn corner_t(face: u32, k: u32) -> u32 {
    return faces_t[face * 3u + k];
}
fn load_t(index: u32) -> ds3 {
    let b = index * 6u;
    return ds3(ds(pos_t[b], pos_t[b + 1u]), ds(pos_t[b + 2u], pos_t[b + 3u]), ds(pos_t[b + 4u], pos_t[b + 5u]));
}
fn tangent_of(dir: ds3, e1: ds3, e2: ds3) -> ds2 {
    return ds2(d3_dot(dir, e1), d3_dot(dir, e2));
}
fn dir_of(index: u32) -> ds3 {
    return norm_ds(load_t(index));
}
@compute @workgroup_size(256)
fn mesh_align_test(@builtin(global_invocation_id) gid3: vec3<u32>) {
    let gid = gid3.x;
    if (gid >= params_t.nvert) {
        return;
    }
    let start = offsets_t[gid];
    let end = offsets_t[gid + 1u];
    if (end < start + 2u) {
        return;
    }
    let sd = dir_of(gid);
    var tmp = ds3(ds_zero(), ds_f(1.0), ds_zero());
    if (ds_gt(ds_f(0.9), ds_abs(sd.x))) {
        tmp = ds3(ds_f(1.0), ds_zero(), ds_zero());
    }
    let c1 = d3_cross(sd, tmp);
    if (!ds_gt(d3_len(c1), ds_zero())) {
        return;
    }
    let e1 = norm_ds(c1);
    let c2 = d3_cross(sd, e1);
    if (!ds_gt(d3_len(c2), ds_zero())) {
        return;
    }
    let e2 = norm_ds(c2);
    for (var i = start; i < end; i = i + 1u) {
        for (var j = i + 1u; j < end; j = j + 1u) {
            let fa = ids_t[i];
            let fb = ids_t[j];
            let a0 = corner_t(fa, 0u);
            let a1 = corner_t(fa, 1u);
            let a2 = corner_t(fa, 2u);
            let b0 = corner_t(fb, 0u);
            let b1 = corner_t(fb, 1u);
            let b2 = corner_t(fb, 2u);
            if (shared_n(a0, a1, a2, b0, b1, b2) != 1u) {
                continue;
            }
            var crossed = false;
            let av = array<u32, 3>(a0, a1, a2);
            let bv = array<u32, 3>(b0, b1, b2);
            for (var e = 0u; e < 3u; e = e + 1u) {
                let pa = tangent_of(dir_of(av[e]), e1, e2);
                let pb = tangent_of(dir_of(av[(e + 1u) % 3u]), e1, e2);
                for (var g = 0u; g < 3u; g = g + 1u) {
                    let pc = tangent_of(dir_of(bv[g]), e1, e2);
                    let pd = tangent_of(dir_of(bv[(g + 1u) % 3u]), e1, e2);
                    let ab_c = orient2(pa, pb, pc);
                    let ab_d = orient2(pa, pb, pd);
                    let cd_a = orient2(pc, pd, pa);
                    let cd_b = orient2(pc, pd, pb);
                    // f64 rounds a shared-endpoint orientation to 0. Double-single
                    // leaves about 1e-20, which a strict sign test calls a crossing.
                    // Real crossings are at least ~1e-7; 1e-12 is only the zero test.
                    if (ds_gt(ds_abs(ab_c), ds_f(1e-12)) && ds_gt(ds_abs(ab_d), ds_f(1e-12)) && ds_gt(ds_abs(cd_a), ds_f(1e-12)) && ds_gt(ds_abs(cd_b), ds_f(1e-12)) && ds_gt(ds_zero(), ds_mul(ab_c, ab_d)) && ds_gt(ds_zero(), ds_mul(cd_a, cd_b))) {
                        crossed = true;
                    }
                }
            }
            if (crossed) {
                atomicAdd(&bad_t[0], 1u);
                atomicStore(&flags_t[a0], 1u);
                atomicStore(&flags_t[a1], 1u);
                atomicStore(&flags_t[a2], 1u);
                atomicStore(&flags_t[b0], 1u);
                atomicStore(&flags_t[b1], 1u);
                atomicStore(&flags_t[b2], 1u);
            }
        }
    }
}
// END ENTRY mesh_align_test

// ENTRY mesh_align_relax
@group(0) @binding(0) var<storage, read_write> pos_r: array<f32>;
@group(0) @binding(1) var<storage, read_write> flags_r: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read> offsets_r: array<u32>;
@group(0) @binding(3) var<storage, read> ids_r: array<u32>;
@group(0) @binding(4) var<storage, read> faces_r: array<u32>;
@group(0) @binding(5) var<storage, read_write> bad_r: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read> params_r: MeshParams;
fn load_r(index: u32) -> ds3 {
    let base = index * 6u;
    return ds3(ds(pos_r[base], pos_r[base + 1u]), ds(pos_r[base + 2u], pos_r[base + 3u]), ds(pos_r[base + 4u], pos_r[base + 5u]));
}
@compute @workgroup_size(1)
fn mesh_align_relax(@builtin(global_invocation_id) gid3: vec3<u32>) {
    if (gid3.x != 0u) {
        return;
    }
    let begin = params_r.cells;
    let stop = begin + params_r.cap;
    for (var gid = begin; gid < stop; gid = gid + 1u) {
    if (atomicLoad(&flags_r[gid]) == 0u) {
        continue;
    }
    let start = offsets_r[gid];
    let stop_f = offsets_r[gid + 1u];
    var nbr: array<u32, 48>;
    var ncount = 0u;
    for (var i = start; i < stop_f; i = i + 1u) {
        let face = ids_r[i];
        for (var k = 0u; k < 3u; k = k + 1u) {
            let v = faces_r[face * 3u + k];
            if (v == gid) {
                continue;
            }
            var seen = false;
            for (var t = 0u; t < ncount; t = t + 1u) {
                if (nbr[t] == v) {
                    seen = true;
                }
            }
            if (!seen) {
                if (ncount >= 48u) {
                    atomicMax(&bad_r[1], 1u);
                } else {
                    nbr[ncount] = v;
                    ncount = ncount + 1u;
                }
            }
        }
    }
    if (ncount == 0u) {
        continue;
    }
    let here = load_r(gid);
    let radius = d3_len(here);
    if (!ds_gt(radius, ds_zero())) {
        continue;
    }
    var acc = ds3(ds_zero(), ds_zero(), ds_zero());
    for (var t = 0u; t < ncount; t = t + 1u) {
        acc = d3_add(acc, norm_ds(load_r(nbr[t])));
    }
    let blended = d3_add(
        d3_scale(d3_scale(here, ds_recip(radius)), ds_f(0.8)),
        d3_scale(norm_ds(d3_scale(acc, ds_f(1.0 / f32(ncount)))), ds_f(0.2)),
    );
    if (!ds_gt(d3_len(blended), ds_zero())) {
        continue;
    }
    let next = d3_scale(norm_ds(blended), radius);
    let b = gid * 6u;
    pos_r[b] = next.x.hi;
    pos_r[b + 1u] = next.x.lo;
    pos_r[b + 2u] = next.y.hi;
    pos_r[b + 3u] = next.y.lo;
    pos_r[b + 4u] = next.z.hi;
    pos_r[b + 5u] = next.z.lo;
    }
}
// END ENTRY mesh_align_relax

// ENTRY mesh_vert_partial
@group(0) @binding(0) var<storage, read> pos_b: array<f32>;
@group(0) @binding(1) var<storage, read_write> partial_b: array<f32>;
@group(0) @binding(2) var<storage, read> params_b: MeshParams;
var<workgroup> red_min: array<vec3<f32>, 256>;
var<workgroup> red_max: array<vec3<f32>, 256>;
@compute @workgroup_size(256)
fn mesh_vert_partial(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let tid = lid.x;
    let gid = wid.x * 256u + tid;
    var bmin = vec3<f32>(1e30, 1e30, 1e30);
    var bmax = vec3<f32>(-1e30, -1e30, -1e30);
    if (gid < params_b.nvert) {
        let b = gid * 6u;
        bmin = vec3<f32>(pos_b[b], pos_b[b + 2u], pos_b[b + 4u]);
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
        partial_b[dst] = red_min[0].x;
        partial_b[dst + 1u] = red_min[0].y;
        partial_b[dst + 2u] = red_min[0].z;
        partial_b[dst + 3u] = red_max[0].x;
        partial_b[dst + 4u] = red_max[0].y;
        partial_b[dst + 5u] = red_max[0].z;
    }
}
// END ENTRY mesh_vert_partial

// ENTRY mesh_scene
@group(0) @binding(0) var<storage, read> partial_sc: array<f32>;
@group(0) @binding(1) var<storage, read_write> scene_sc: array<f32>;
@group(0) @binding(2) var<storage, read> params_sc: MeshParams;
var<workgroup> scene_min: array<vec3<f32>, 256>;
var<workgroup> scene_max: array<vec3<f32>, 256>;
@compute @workgroup_size(256)
fn mesh_scene(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    var bmin = vec3<f32>(1e30, 1e30, 1e30);
    var bmax = vec3<f32>(-1e30, -1e30, -1e30);
    for (var g = tid; g < params_sc.vert_groups; g = g + 256u) {
        let src = g * 6u;
        bmin = min(bmin, vec3<f32>(partial_sc[src], partial_sc[src + 1u], partial_sc[src + 2u]));
        bmax = max(bmax, vec3<f32>(partial_sc[src + 3u], partial_sc[src + 4u], partial_sc[src + 5u]));
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
        let axis = max(max(scene_max[0].x - scene_min[0].x, scene_max[0].y - scene_min[0].y), scene_max[0].z - scene_min[0].z);
        let extent = max(max(axis, bits_f32(params_sc.scale_bits)), 1e-12);
        scene_sc[0] = scene_min[0].x;
        scene_sc[1] = scene_min[0].y;
        scene_sc[2] = scene_min[0].z;
        scene_sc[3] = scene_max[0].x;
        scene_sc[4] = scene_max[0].y;
        scene_sc[5] = scene_max[0].z;
        scene_sc[6] = extent;
        scene_sc[7] = 0.0;
    }
}
// END ENTRY mesh_scene

// ENTRY mesh_tri_cell
@group(0) @binding(0) var<storage, read> faces_tc: array<u32>;
@group(0) @binding(1) var<storage, read> pos_tc: array<f32>;
@group(0) @binding(2) var<storage, read> scene_tc: array<f32>;
@group(0) @binding(3) var<storage, read_write> cells_tc: array<u32>;
@group(0) @binding(4) var<storage, read_write> cover_tc: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read> params_tc: MeshParams;
fn cell_index(value: f32, origin: f32, extent: f32, cells: u32) -> u32 {
    let cell = extent / f32(cells);
    var i = i32(floor((value - origin) / cell));
    if (i < 0) {
        i = 0;
    }
    if (i > i32(cells)) {
        i = i32(cells);
    }
    return u32(i);
}
fn vert_tc(index: u32) -> vec3<f32> {
    let b = index * 6u;
    return vec3<f32>(pos_tc[b], pos_tc[b + 2u], pos_tc[b + 4u]);
}
@compute @workgroup_size(256)
fn mesh_tri_cell(@builtin(global_invocation_id) gid3: vec3<u32>) {
    let gid = gid3.x;
    if (gid >= params_tc.nface) {
        return;
    }
    let base = gid * 3u;
    let a = vert_tc(faces_tc[base]);
    let b = vert_tc(faces_tc[base + 1u]);
    let c = vert_tc(faces_tc[base + 2u]);
    let lo = min(min(a, b), c);
    let hi = max(max(a, b), c);
    let origin = vec3<f32>(scene_tc[0], scene_tc[1], scene_tc[2]);
    let extent = scene_tc[6];
    let ix0 = cell_index(lo.x, origin.x, extent, params_tc.cells);
    let iy0 = cell_index(lo.y, origin.y, extent, params_tc.cells);
    let iz0 = cell_index(lo.z, origin.z, extent, params_tc.cells);
    let ix1 = cell_index(hi.x, origin.x, extent, params_tc.cells);
    let iy1 = cell_index(hi.y, origin.y, extent, params_tc.cells);
    let iz1 = cell_index(hi.z, origin.z, extent, params_tc.cells);
    let dst = gid * 6u;
    cells_tc[dst] = ix0;
    cells_tc[dst + 1u] = iy0;
    cells_tc[dst + 2u] = iz0;
    cells_tc[dst + 3u] = ix1;
    cells_tc[dst + 4u] = iy1;
    cells_tc[dst + 5u] = iz1;
    let cover = (ix1 - ix0 + 1u) * (iy1 - iy0 + 1u) * (iz1 - iz0 + 1u);
    atomicAdd(&cover_tc[0], cover);
}
// END ENTRY mesh_tri_cell

// ENTRY mesh_zero_cells
@group(0) @binding(0) var<storage, read_write> cells_zz: array<u32>;
@group(0) @binding(1) var<storage, read> params_zz: MeshParams;
@compute @workgroup_size(256)
fn mesh_zero_cells(@builtin(global_invocation_id) gid3: vec3<u32>) {
    let gid = gid3.x;
    let n = params_zz.side * params_zz.side * params_zz.side;
    if (gid < n) {
        cells_zz[gid] = 0u;
    }
}
// END ENTRY mesh_zero_cells

// ENTRY mesh_cell_count
@group(0) @binding(0) var<storage, read> tri_cc: array<u32>;
@group(0) @binding(1) var<storage, read_write> count_cc: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read> params_cc: MeshParams;
@compute @workgroup_size(256)
fn mesh_cell_count(@builtin(global_invocation_id) gid3: vec3<u32>) {
    let gid = gid3.x;
    if (gid >= params_cc.nface) {
        return;
    }
    let b = gid * 6u;
    let ix0 = tri_cc[b];
    let iy0 = tri_cc[b + 1u];
    let iz0 = tri_cc[b + 2u];
    let ix1 = tri_cc[b + 3u];
    let iy1 = tri_cc[b + 4u];
    let iz1 = tri_cc[b + 5u];
    for (var ix = ix0; ix <= ix1; ix = ix + 1u) {
        for (var iy = iy0; iy <= iy1; iy = iy + 1u) {
            for (var iz = iz0; iz <= iz1; iz = iz + 1u) {
                atomicAdd(&count_cc[slot_of(ix, iy, iz, params_cc.side)], 1u);
            }
        }
    }
}
// END ENTRY mesh_cell_count

// ENTRY mesh_cell_scan
@group(0) @binding(0) var<storage, read> count_cs: array<u32>;
@group(0) @binding(1) var<storage, read_write> start_cs: array<u32>;
@group(0) @binding(2) var<storage, read_write> cursor_cs: array<u32>;
@group(0) @binding(3) var<storage, read> params_cs: MeshParams;
@compute @workgroup_size(256)
fn mesh_cell_scan(@builtin(global_invocation_id) gid3: vec3<u32>) {
    if (gid3.x != 0u) {
        return;
    }
    let n = params_cs.side * params_cs.side * params_cs.side;
    var sum = 0u;
    for (var i = 0u; i < n; i = i + 1u) {
        start_cs[i] = sum;
        cursor_cs[i] = sum;
        sum = sum + count_cs[i];
    }
}
// END ENTRY mesh_cell_scan

// ENTRY mesh_cell_fill
@group(0) @binding(0) var<storage, read> tri_cf: array<u32>;
@group(0) @binding(1) var<storage, read_write> cursor_cf: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> ids_cf: array<u32>;
@group(0) @binding(3) var<storage, read> params_cf: MeshParams;
@compute @workgroup_size(256)
fn mesh_cell_fill(@builtin(global_invocation_id) gid3: vec3<u32>) {
    let gid = gid3.x;
    if (gid >= params_cf.nface) {
        return;
    }
    let b = gid * 6u;
    for (var ix = tri_cf[b]; ix <= tri_cf[b + 3u]; ix = ix + 1u) {
        for (var iy = tri_cf[b + 1u]; iy <= tri_cf[b + 4u]; iy = iy + 1u) {
            for (var iz = tri_cf[b + 2u]; iz <= tri_cf[b + 5u]; iz = iz + 1u) {
                let slot = atomicAdd(&cursor_cf[slot_of(ix, iy, iz, params_cf.side)], 1u);
                if (slot < params_cf.cap) {
                    ids_cf[slot] = gid;
                }
            }
        }
    }
}
// END ENTRY mesh_cell_fill

// ENTRY mesh_isect
@group(0) @binding(0) var<storage, read> faces_x: array<u32>;
@group(0) @binding(1) var<storage, read> pos_x: array<f32>;
@group(0) @binding(2) var<storage, read> tri_x: array<u32>;
@group(0) @binding(3) var<storage, read> start_x: array<u32>;
@group(0) @binding(4) var<storage, read> count_x: array<u32>;
@group(0) @binding(5) var<storage, read> ids_x: array<u32>;
@group(0) @binding(6) var<storage, read_write> hits_x: array<u32>;
@group(0) @binding(7) var<storage, read> params_x: MeshParams;
fn load_x(index: u32) -> ds3 {
    let b = index * 6u;
    return ds3(ds(pos_x[b], pos_x[b + 1u]), ds(pos_x[b + 2u], pos_x[b + 3u]), ds(pos_x[b + 4u], pos_x[b + 5u]));
}
fn eps_x() -> ds {
    return ds_mul(ds_f(1e-10), ds(bits_f32(params_x.scale_bits), bits_f32(params_x.extent_bits)));
}
fn owned_x(cell: vec3<u32>, a: vec3<u32>, b: vec3<u32>) -> bool {
    return cell.x == max(a.x, b.x) && cell.y == max(a.y, b.y) && cell.z == max(a.z, b.z);
}
@compute @workgroup_size(256)
fn mesh_isect(@builtin(global_invocation_id) gid3: vec3<u32>) {
    let gid = gid3.x;
    if (gid >= params_x.nface) {
        return;
    }
    hits_x[gid * 2u] = 0xffffffffu;
    hits_x[gid * 2u + 1u] = 0u;
    let b = gid * 6u;
    let lo = vec3<u32>(tri_x[b], tri_x[b + 1u], tri_x[b + 2u]);
    let hi = vec3<u32>(tri_x[b + 3u], tri_x[b + 4u], tri_x[b + 5u]);
    let ia0 = faces_x[gid * 3u];
    let ia1 = faces_x[gid * 3u + 1u];
    let ia2 = faces_x[gid * 3u + 2u];
    let eps = eps_x();
    let a0 = load_x(ia0);
    let a1 = load_x(ia1);
    let a2 = load_x(ia2);
    var best_hi = 0xffffffffu;
    var best_gap = 0.0;
    var found = false;
    for (var ix = lo.x; ix <= hi.x; ix = ix + 1u) {
        for (var iy = lo.y; iy <= hi.y; iy = iy + 1u) {
            for (var iz = lo.z; iz <= hi.z; iz = iz + 1u) {
                let slot = slot_of(ix, iy, iz, params_x.side);
                let begin = start_x[slot];
                let count = count_x[slot];
                for (var k = 0u; k < count; k = k + 1u) {
                    let other = ids_x[begin + k];
                    if (other <= gid) {
                        continue;
                    }
                    let ob = other * 6u;
                    let olo = vec3<u32>(tri_x[ob], tri_x[ob + 1u], tri_x[ob + 2u]);
                    if (!owned_x(vec3<u32>(ix, iy, iz), lo, olo)) {
                        continue;
                    }
                    let ib0 = faces_x[other * 3u];
                    let ib1 = faces_x[other * 3u + 1u];
                    let ib2 = faces_x[other * 3u + 2u];
                    if (shared_n(ia0, ia1, ia2, ib0, ib1, ib2) >= 2u) {
                        continue;
                    }
                    let hit = tri_hit(a0, a1, a2, load_x(ib0), load_x(ib1), load_x(ib2), eps);
                    if (hit.hit != 0u && (!found || other < best_hi)) {
                        found = true;
                        best_hi = other;
                        best_gap = hit.gap;
                    }
                }
            }
        }
    }
    if (found) {
        hits_x[gid * 2u] = best_hi;
        hits_x[gid * 2u + 1u] = bitcast<u32>(best_gap);
    }
}
// END ENTRY mesh_isect

// ENTRY mesh_best
@group(0) @binding(0) var<storage, read> hits_b: array<u32>;
@group(0) @binding(1) var<storage, read_write> out_b: array<u32>;
@group(0) @binding(2) var<storage, read> params_best: MeshParams;
@compute @workgroup_size(256)
fn mesh_best(@builtin(global_invocation_id) gid3: vec3<u32>) {
    if (gid3.x != 0u) {
        return;
    }
    out_b[0] = 0u;
    out_b[1] = 0xffffffffu;
    out_b[2] = 0xffffffffu;
    out_b[3] = 0u;
    for (var i = 0u; i < params_best.nface; i = i + 1u) {
        let hi = hits_b[i * 2u];
        if (hi != 0xffffffffu) {
            out_b[0] = 1u;
            out_b[1] = i;
            out_b[2] = hi;
            out_b[3] = hits_b[i * 2u + 1u];
            return;
        }
    }
}
// END ENTRY mesh_best
