#include <metal_stdlib>
using namespace metal;
using namespace metal::raytracing;

// Fused RTFP field. One threadgroup owns one sample point; each thread owns a
// strided set of patch rays. Crossings stay in a private array, chords are
// paired by occupancy parity, and every density kernel is integrated on the
// spot: the α = 1 antiderivative, or a fixed Gauss–Legendre rule otherwise.
// The threadgroup then tree-reduces the ten field components. No atomics.

#define TG 256u
#define MAX_MERGED 128u
#define MAX_STEPS 8192u

struct GpuPoint {
    float ox, oy, oz, theta;
    float wx, wy, wz, rho;
};

struct GpuRay {
    float dx, dy, dz, weight;
};

struct GpuRange {
    uint begin;
    uint count;
};

struct GpuKernel {
    float cx, cy, cz, inv_sigma2;
    float coeff, alpha;
};

struct GpuNode {
    float x;
    float w;
};

struct FieldParams {
    uint kernel_count;
    uint node_count;
    uint want_potential;
    uint want_hessian;
    float bulk;
    float ell;
    float body_scale;
    float g;
    float merge_rel;
    float merge_abs;
    float origin_abs;
    float launch_offset;
};

struct MomentParams {
    uint kernel_count;
    uint node_count;
    uint ray_count;
    uint max_panels;
    float bulk;
    float finest;
    float body_scale;
    float launch_offset;
    float merge_rel;
    float merge_abs;
    float origin_abs;
    float ox, oy, oz;
    float wx, wy, wz;
    float pad;
};

struct Scalar {
    ds u;
    ds g;
    ds h;
};

inline Scalar scalar_zero() {
    return Scalar{ds_zero(), ds_zero(), ds_zero()};
}

inline ds chord_ln(float r0, float r1, float ell, uint want_h) {
    if (want_h == 0u) {
        return ds_zero();
    }
    if (r0 == 0.0f) {
        return ds_ln(ds_div(ds_from_float(r1), ds_from_float(ell)));
    }
    return ds_ln(ds_div(ds_from_float(r1), ds_from_float(r0)));
}

// Hardware walk. Hits come back nearest-first; coincident edge reports are
// fused with the same absolute/relative tolerance as the occupancy test.
// The walk always steps past a hit, including hits that the fuse drops.
uint collect_hits(
    float3 origin_u,
    float3 dir,
    primitive_acceleration_structure accel,
    float body_scale,
    float launch_offset,
    float origin_abs,
    float merge_abs,
    float merge_rel,
    thread float* merged,
    thread bool& overflow
) {
    intersector<triangle_data> isect;
    isect.assume_geometry_type(geometry_type::triangle);
    isect.force_opacity(forced_opacity::opaque);
    ray r;
    r.origin = origin_u;
    r.direction = dir;
    r.min_distance = launch_offset;
    r.max_distance = 1.0e20f;

    uint n = 0u;
    float last = -1.0e30f;
    overflow = false;
    for (uint step = 0u; step < MAX_STEPS; ++step) {
        intersection_result<triangle_data> hit = isect.intersect(r, accel);
        if (hit.type != intersection_type::triangle) {
            return n;
        }
        float t = hit.distance;
        float nxt = nextafter(t, INFINITY);
        if (!isfinite(t) || !(nxt > t)) {
            return n;
        }
        r.min_distance = nxt;
        float world = t * body_scale;
        if (!(isfinite(world) && world > origin_abs)) {
            continue;
        }
        if (n > 0u && (world - last) <= (merge_abs + merge_rel * world)) {
            continue;
        }
        if (n >= MAX_MERGED) {
            overflow = true;
            return n;
        }
        merged[n] = world;
        last = world;
        n += 1u;
    }
    overflow = true;
    return n;
}

inline void add_alpha_one(
    thread Scalar& acc,
    float b,
    float d,
    float a0,
    float coeff,
    float r0,
    float r1,
    ds ln_ratio,
    uint want_u,
    uint want_h
) {
    ds db = ds_from_float(b);
    ds dd = ds_from_float(d);
    ds x0 = ds_div(ds_sub(ds_from_float(r0), db), dd);
    ds x1 = ds_div(ds_sub(ds_from_float(r1), db), dd);
    ds dat = ds_atan2(ds_sub(x1, x0), ds_add(ds_from_float(1.0f), ds_mul(x0, x1)));
    ds ig = ds_div(dat, dd);
    ds iu = ds_zero();
    ds ih = ds_zero();
    if (want_u != 0u || want_h != 0u) {
        ds half_ln = ds_mul(
            ds_from_float(0.5f),
            ds_ln(ds_div(
                ds_add(ds_from_float(1.0f), ds_mul(x1, x1)),
                ds_add(ds_from_float(1.0f), ds_mul(x0, x0)))));
        ds bd_dat = ds_mul(ds_div(db, dd), dat);
        iu = ds_add(half_ln, bd_dat);
        if (want_h != 0u) {
            ih = ds_div(ds_add(ds_sub(ln_ratio, half_ln), bd_dat), ds_from_float(a0));
        }
    }
    ds c = ds_from_float(coeff);
    acc.u = ds_add(acc.u, ds_mul(c, iu));
    acc.g = ds_add(acc.g, ds_mul(c, ig));
    acc.h = ds_add(acc.h, ds_mul(c, ih));
}

inline void add_general(
    thread Scalar& acc,
    float alpha,
    float b,
    float d,
    float a0,
    float coeff,
    float r0,
    float r1,
    ds ln_ratio,
    uint node_count,
    device const GpuNode* nodes,
    uint want_u,
    uint want_h
) {
    ds db = ds_from_float(b);
    ds dd = ds_from_float(d);
    ds da0 = ds_from_float(a0);
    ds s0 = ds_asinh(ds_div(ds_sub(ds_from_float(r0), db), dd));
    ds s1 = ds_asinh(ds_div(ds_sub(ds_from_float(r1), db), dd));
    float span = (s1.hi + s1.lo) - (s0.hi + s0.lo);
    uint panels = 1u;
    if (isfinite(span) && span > 0.0f) {
        float rounded = ceil(span);
        panels = rounded > 128.0f ? 128u : max((uint)rounded, 1u);
    }
    float h = (panels == 0u) ? 0.0f : span / float(panels);
    ds amp = ds_pow(dd, 1.0f - 2.0f * alpha);
    ds a0_pow = ds_pow(da0, -alpha);
    float reg_scale = 1.0e-14f * (fabs(b) + d);
    ds iu = ds_zero();
    ds ig = ds_zero();
    ds ih = ds_zero();
    float s_base = s0.hi + s0.lo;
    for (uint p = 0u; p < panels; ++p) {
        float lo = s_base + h * float(p);
        for (uint n = 0u; n < node_count; ++n) {
            float s = lo + 0.5f * h * (nodes[n].x + 1.0f);
            float gw = 0.5f * h * nodes[n].w;
            ds sd = ds_from_float(s);
            ds ch = ds_cosh(sd);
            ds radius = ds_add(db, ds_mul(dd, ds_sinh(sd)));
            if (radius.hi < 0.0f) {
                radius = ds_zero();
            }
            ds shape = ds_mul(amp, ds_pow(ch, 1.0f - 2.0f * alpha));
            ds wd = ds_from_float(gw);
            iu = ds_add(iu, ds_mul(ds_mul(wd, shape), radius));
            ig = ds_add(ig, ds_mul(wd, shape));
            if (want_h != 0u) {
                float rf = radius.hi + radius.lo;
                ds regular;
                if (rf > reg_scale) {
                    ds arg = ds_div(
                        ds_mul(radius, ds_sub(radius, ds_mul(ds_from_float(2.0f), db))),
                        da0);
                    regular = ds_div(
                        ds_mul(a0_pow, ds_expm1(ds_mul(ds_from_float(-alpha), ds_ln1p(arg)))),
                        radius);
                } else {
                    regular = ds_div(
                        ds_mul(ds_from_float(2.0f * alpha * b), a0_pow),
                        da0);
                }
                ih = ds_add(ih, ds_mul(ds_mul(wd, regular), ds_mul(dd, ch)));
            }
        }
    }
    if (want_h != 0u) {
        ih = ds_add(ih, ds_mul(a0_pow, ln_ratio));
    }
    ds c = ds_from_float(coeff);
    if (want_u != 0u || want_h != 0u) {
        acc.u = ds_add(acc.u, ds_mul(c, iu));
    }
    acc.g = ds_add(acc.g, ds_mul(c, ig));
    if (want_h != 0u) {
        acc.h = ds_add(acc.h, ds_mul(c, ih));
    }
}

inline void integrate_chord(
    thread Scalar& acc,
    float r0,
    float r1,
    float3 world,
    float3 unit_from_point,
    float ell,
    float bulk,
    uint kernel_count,
    uint node_count,
    uint want_u,
    uint want_h,
    device const GpuKernel* kernels,
    device const GpuNode* nodes
) {
    if (!(r1 > r0)) {
        return;
    }
    ds ln_ratio = chord_ln(r0, r1, ell, want_h);
    if (bulk != 0.0f) {
        ds r0d = ds_from_float(r0);
        ds r1d = ds_from_float(r1);
        ds dbulk = ds_from_float(bulk);
        ds r2 = ds_sub(ds_mul(r1d, r1d), ds_mul(r0d, r0d));
        acc.u = ds_add(acc.u, ds_mul(ds_mul(ds_from_float(0.5f), dbulk), r2));
        acc.g = ds_add(acc.g, ds_mul(dbulk, ds_sub(r1d, r0d)));
        if (want_h != 0u) {
            acc.h = ds_add(acc.h, ds_mul(dbulk, ln_ratio));
        }
    }
    for (uint k = 0u; k < kernel_count; ++k) {
        GpuKernel src = kernels[k];
        if (src.coeff == 0.0f) {
            continue;
        }
        float3 p = world - float3(src.cx, src.cy, src.cz);
        float b = dot(p, unit_from_point);
        float a0 = src.inv_sigma2 + dot(p, p);
        float3 perp = p - unit_from_point * b;
        float d2 = src.inv_sigma2 + dot(perp, perp);
        float d = sqrt(d2);
        if (!(d > 0.0f) || !isfinite(d) || !isfinite(a0)) {
            continue;
        }
        if (src.alpha == 1.0f) {
            add_alpha_one(acc, b, d, a0, src.coeff, r0, r1, ln_ratio, want_u, want_h);
        } else {
            add_general(
                acc, src.alpha, b, d, a0, src.coeff, r0, r1, ln_ratio, node_count, nodes, want_u,
                want_h);
        }
    }
}

inline void store_pair(threadgroup float* hi, threadgroup float* lo, uint slot, ds value) {
    hi[slot] = value.hi;
    lo[slot] = value.lo;
}

inline ds load_pair(threadgroup float* hi, threadgroup float* lo, uint slot) {
    return ds{hi[slot], lo[slot]};
}

inline void reduce10(
    threadgroup float* hi,
    threadgroup float* lo,
    threadgroup uint* overflow_bits,
    uint tid
) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = TG / 2u; s > 0u; s >>= 1u) {
        if (tid < s) {
            for (uint m = 0u; m < 10u; ++m) {
                uint slot = m * TG + tid;
                uint other = m * TG + tid + s;
                ds sum = ds_add(load_pair(hi, lo, slot), load_pair(hi, lo, other));
                store_pair(hi, lo, slot, sum);
            }
            overflow_bits[tid] |= overflow_bits[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void rtfp_field(
    device const GpuPoint* points [[buffer(0)]],
    device const GpuRange* ranges [[buffer(1)]],
    device const GpuRay* rays [[buffer(2)]],
    device const GpuKernel* kernels [[buffer(3)]],
    device const GpuNode* nodes [[buffer(4)]],
    constant FieldParams& params [[buffer(5)]],
    device float* out [[buffer(6)]],
    device uint* flags [[buffer(7)]],
    primitive_acceleration_structure accel [[buffer(8)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]]
) {
    GpuPoint pt = points[tgid];
    GpuRange range = ranges[tgid];
    float3 origin_u = float3(pt.ox, pt.oy, pt.oz);
    float3 world = float3(pt.wx, pt.wy, pt.wz);
    uint want_h = params.want_hessian;
    uint want_u = (params.want_potential != 0u || want_h != 0u) ? 1u : 0u;

    ds priv[10];
    for (uint m = 0u; m < 10u; ++m) {
        priv[m] = ds_zero();
    }
    bool overflow = false;
    float merged[MAX_MERGED];

    for (uint i = tid; i < range.count; i += TG) {
        GpuRay ray = rays[range.begin + i];
        if (!isfinite(ray.weight) || ray.weight == 0.0f) {
            continue;
        }
        float3 dir = float3(ray.dx, ray.dy, ray.dz);
        float len2 = dot(dir, dir);
        if (!(len2 > 0.0f) || !isfinite(len2)) {
            continue;
        }
        dir *= rsqrt(len2);
        bool ray_overflow = false;
        uint n = collect_hits(
            origin_u, dir, accel, params.body_scale, params.launch_offset, params.origin_abs,
            params.merge_abs, params.merge_rel, merged, ray_overflow);
        if (ray_overflow) {
            overflow = true;
            continue;
        }
        Scalar acc = scalar_zero();
        float open_r = (n & 1u) != 0u ? 0.0f : -1.0f;
        for (uint h = 0u; h < n; ++h) {
            float t = merged[h];
            if (open_r >= 0.0f) {
                integrate_chord(
                    acc, open_r, t, world, -dir, params.ell, params.bulk, params.kernel_count,
                    params.node_count, want_u, want_h, kernels, nodes);
                open_r = -1.0f;
            } else {
                open_r = t;
            }
        }
        ds weight = ds_from_float(ray.weight);
        ds wu = ds_mul(weight, acc.u);
        ds wg = ds_mul(weight, acc.g);
        priv[0] = ds_add(priv[0], wu);
        priv[1] = ds_add(priv[1], ds_mul(ds_from_float(dir.x), wg));
        priv[2] = ds_add(priv[2], ds_mul(ds_from_float(dir.y), wg));
        priv[3] = ds_add(priv[3], ds_mul(ds_from_float(dir.z), wg));
        if (want_h != 0u) {
            ds wh = ds_mul(weight, acc.h);
            float txx = fma(3.0f * dir.x, dir.x, -1.0f);
            float tyy = fma(3.0f * dir.y, dir.y, -1.0f);
            float tzz = fma(3.0f * dir.z, dir.z, -1.0f);
            priv[4] = ds_add(priv[4], ds_mul(ds_from_float(txx), wh));
            priv[5] = ds_add(priv[5], ds_mul(ds_from_float(tyy), wh));
            priv[6] = ds_add(priv[6], ds_mul(ds_from_float(tzz), wh));
            priv[7] = ds_add(priv[7], ds_mul(ds_from_float(3.0f * dir.x * dir.y), wh));
            priv[8] = ds_add(priv[8], ds_mul(ds_from_float(3.0f * dir.x * dir.z), wh));
            priv[9] = ds_add(priv[9], ds_mul(ds_from_float(3.0f * dir.y * dir.z), wh));
        }
    }

    if (overflow) {
        for (uint m = 0u; m < 10u; ++m) {
            priv[m] = ds_zero();
        }
    }

    threadgroup float red_hi[10 * TG];
    threadgroup float red_lo[10 * TG];
    threadgroup uint ov[TG];
    for (uint m = 0u; m < 10u; ++m) {
        store_pair(red_hi, red_lo, m * TG + tid, priv[m]);
    }
    ov[tid] = overflow ? 1u : 0u;
    reduce10(red_hi, red_lo, ov, tid);

    if (tid == 0u) {
        ds gscale = ds_from_float(params.g);
        for (uint m = 0u; m < 10u; ++m) {
            uint slot = m * TG;
            store_pair(red_hi, red_lo, slot, ds_mul(load_pair(red_hi, red_lo, slot), gscale));
        }
        if (want_h != 0u) {
            ds contact = ds_mul(
                ds_mul(ds_from_float(-4.0f / 3.0f), ds_pi()),
                ds_mul(gscale, ds_mul(ds_from_float(pt.theta), ds_from_float(pt.rho))));
            for (uint m = 4u; m < 7u; ++m) {
                uint slot = m * TG;
                store_pair(red_hi, red_lo, slot, ds_add(load_pair(red_hi, red_lo, slot), contact));
            }
        }
        uint base = tgid * 20u;
        for (uint m = 0u; m < 10u; ++m) {
            ds value = load_pair(red_hi, red_lo, m * TG);
            out[base + 2u * m] = value.hi;
            out[base + 2u * m + 1u] = value.lo;
        }
        flags[tgid] = ov[0];
    }
}

inline float density_at(float3 y, float bulk, device const GpuKernel* kernels, uint kernel_count) {
    float rho = bulk;
    for (uint k = 0u; k < kernel_count; ++k) {
        GpuKernel src = kernels[k];
        float3 d = y - float3(src.cx, src.cy, src.cz);
        float base = src.inv_sigma2 + dot(d, d);
        rho += src.coeff * pow(base, -src.alpha);
    }
    return rho;
}

kernel void rtfp_moments(
    device const GpuRay* rays [[buffer(0)]],
    device const GpuKernel* kernels [[buffer(1)]],
    device const GpuNode* nodes [[buffer(2)]],
    constant MomentParams& params [[buffer(3)]],
    device float* out [[buffer(4)]],
    device uint* flags [[buffer(5)]],
    primitive_acceleration_structure accel [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]]
) {
    float3 origin_u = float3(params.ox, params.oy, params.oz);
    float3 world = float3(params.wx, params.wy, params.wz);
    ds priv[10];
    for (uint m = 0u; m < 10u; ++m) {
        priv[m] = ds_zero();
    }
    bool overflow = false;
    float merged[MAX_MERGED];
    for (uint i = tid; i < params.ray_count; i += TG) {
        GpuRay ray = rays[i];
        if (!isfinite(ray.weight) || ray.weight == 0.0f) {
            continue;
        }
        float3 dir = float3(ray.dx, ray.dy, ray.dz);
        float len2 = dot(dir, dir);
        if (!(len2 > 0.0f) || !isfinite(len2)) {
            continue;
        }
        dir *= rsqrt(len2);
        bool ray_overflow = false;
        uint n = collect_hits(
            origin_u, dir, accel, params.body_scale, params.launch_offset, params.origin_abs,
            params.merge_abs, params.merge_rel, merged, ray_overflow);
        if (ray_overflow) {
            overflow = true;
            continue;
        }
        float open_r = (n & 1u) != 0u ? 0.0f : -1.0f;
        for (uint h = 0u; h < n; ++h) {
            float t = merged[h];
            if (open_r < 0.0f) {
                open_r = t;
                continue;
            }
            float r0 = open_r;
            float r1 = t;
            open_r = -1.0f;
            float span = r1 - r0;
            if (!(span > 0.0f)) {
                continue;
            }
            float ratio = span / params.finest;
            uint panels = 1u;
            if (!isfinite(ratio) || ratio > float(params.max_panels)) {
                panels = params.max_panels;
            } else {
                panels = max((uint)ceil(ratio), 1u);
            }
            float step = span / float(panels);
            for (uint p = 0u; p < panels; ++p) {
                float lo = r0 + step * float(p);
                for (uint q = 0u; q < params.node_count; ++q) {
                    float radius = lo + 0.5f * step * (nodes[q].x + 1.0f);
                    float gw = 0.5f * step * nodes[q].w;
                    float3 y = world + dir * radius;
                    float rho = density_at(y, params.bulk, kernels, params.kernel_count);
                    float mass = ray.weight * rho * radius * radius * gw;
                    ds dm = ds_from_float(mass);
                    priv[0] = ds_add(priv[0], dm);
                    priv[1] = ds_add(priv[1], ds_mul(dm, ds_from_float(y.x)));
                    priv[2] = ds_add(priv[2], ds_mul(dm, ds_from_float(y.y)));
                    priv[3] = ds_add(priv[3], ds_mul(dm, ds_from_float(y.z)));
                    priv[4] = ds_add(priv[4], ds_mul(dm, ds_from_float(y.x * y.x)));
                    priv[5] = ds_add(priv[5], ds_mul(dm, ds_from_float(y.y * y.y)));
                    priv[6] = ds_add(priv[6], ds_mul(dm, ds_from_float(y.z * y.z)));
                    priv[7] = ds_add(priv[7], ds_mul(dm, ds_from_float(y.x * y.y)));
                    priv[8] = ds_add(priv[8], ds_mul(dm, ds_from_float(y.x * y.z)));
                    priv[9] = ds_add(priv[9], ds_mul(dm, ds_from_float(y.y * y.z)));
                }
            }
        }
    }
    if (overflow) {
        for (uint m = 0u; m < 10u; ++m) {
            priv[m] = ds_zero();
        }
    }
    threadgroup float red_hi[10 * TG];
    threadgroup float red_lo[10 * TG];
    threadgroup uint ov[TG];
    for (uint m = 0u; m < 10u; ++m) {
        store_pair(red_hi, red_lo, m * TG + tid, priv[m]);
    }
    ov[tid] = overflow ? 1u : 0u;
    reduce10(red_hi, red_lo, ov, tid);
    if (tid == 0u) {
        for (uint m = 0u; m < 10u; ++m) {
            ds value = load_pair(red_hi, red_lo, m * TG);
            out[2u * m] = value.hi;
            out[2u * m + 1u] = value.lo;
        }
        flags[0] = ov[0];
    }
}
