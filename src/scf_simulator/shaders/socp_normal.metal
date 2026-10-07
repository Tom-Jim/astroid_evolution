// Face frame of the §10.4.3 slope row, in double-single arithmetic.
//
// One thread per face. Positions are the scaled vertices `x / R₀`. Directions
// `w` are the fixed radial unit vectors. The outputs are the unit normal and
// the three columns of `∂n̂/∂R`, matching the f64 formula
//
//   A = (p_b − p_a) × (p_c − p_a),   n̂ = A / |A|,
//   N_j = (I − n̂ n̂ᵀ) (w_j × e_j) / |A|.

#include <metal_stdlib>
using namespace metal;

struct ds3 {
    ds x;
    ds y;
    ds z;
};

inline ds3 ds3_load(constant float* p) {
    return ds3{ds_mk(p[0], p[1]), ds_mk(p[2], p[3]), ds_mk(p[4], p[5])};
}

inline ds3 ds3_sub(ds3 a, ds3 b) {
    return ds3{ds_subp(a.x, b.x), ds_subp(a.y, b.y), ds_subp(a.z, b.z)};
}

inline ds3 ds3_cross(ds3 a, ds3 b) {
    return ds3{
        ds_subp(ds_mul(a.y, b.z), ds_mul(a.z, b.y)),
        ds_subp(ds_mul(a.z, b.x), ds_mul(a.x, b.z)),
        ds_subp(ds_mul(a.x, b.y), ds_mul(a.y, b.x)),
    };
}

inline ds ds3_dot(ds3 a, ds3 b) {
    dsacc s = acc_zero();
    acc_fma(s, a.x, b.x);
    acc_fma(s, a.y, b.y);
    acc_fma(s, a.z, b.z);
    return acc_done(s);
}

inline ds3 ds3_scale(ds3 a, ds s) {
    return ds3{ds_mul(a.x, s), ds_mul(a.y, s), ds_mul(a.z, s)};
}

inline void ds3_store(device float* o, ds3 a) {
    o[0] = a.x.hi;
    o[1] = a.x.lo;
    o[2] = a.y.hi;
    o[3] = a.y.lo;
    o[4] = a.z.hi;
    o[5] = a.z.lo;
}

kernel void socp_face_frame(
    constant float* pos [[buffer(0)]],
    constant float* dir [[buffer(1)]],
    constant uint* faces [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& nfaces [[buffer(4)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= nfaces) {
        return;
    }
    uint base = tid * 3u;
    uint ia = faces[base];
    uint ib = faces[base + 1u];
    uint ic = faces[base + 2u];
    ds3 pa = ds3_load(pos + ia * 6u);
    ds3 pb = ds3_load(pos + ib * 6u);
    ds3 pc = ds3_load(pos + ic * 6u);
    ds3 area = ds3_cross(ds3_sub(pb, pa), ds3_sub(pc, pa));
    ds len = ds_sqrt(ds3_dot(area, area));
    ds inv = ds_div(ds_f(1.0f), len);
    ds3 n = ds3_scale(area, inv);

    ds3 w0 = ds3_load(dir + ia * 6u);
    ds3 w1 = ds3_load(dir + ib * 6u);
    ds3 w2 = ds3_load(dir + ic * 6u);
    ds3 da0 = ds3_cross(w0, ds3_sub(pb, pc));
    ds3 da1 = ds3_cross(w1, ds3_sub(pc, pa));
    ds3 da2 = ds3_cross(w2, ds3_sub(pa, pb));
    ds3 cols[3] = {da0, da1, da2};

    device float* o = out + tid * 24u;
    ds3_store(o, n);
    for (int k = 0; k < 3; ++k) {
        ds nd = ds3_dot(n, cols[k]);
        ds3 tang = ds3_sub(cols[k], ds3_scale(n, nd));
        ds3_store(o + 6u + uint(k) * 6u, ds3_scale(tang, inv));
    }
}
