// Mass, first and second moments of the density on the body (MS §3, §8.1.3).
//
// Rays leave one origin inside the body; one thread per direction. The radial
// weight is R² (the volume element). The host integrates a constant c exactly by
// the divergence theorem and passes `bulk − c` here, so the angular rule only
// sees ρ − c; the constant part of each segment is closed and each core uses the
// composite rule of `chord.wesl` on the moments R², R³, R⁴. Output per ray:
// double-single (m, m x, m y, m z, m xx, m yy, m zz, m xy, m xz, m yz) already
// multiplied by the direction weight, then an overflow flag. The host sums them.

// #include "rtfp/shader/chord.wesl"

@group(0) @binding(0)
var<storage, read> origin: vec4<f32>;
@group(0) @binding(1)
var<storage, read> rays: array<Ray>;
@group(0) @binding(2)
var<storage, read> kerns: array<Kern>;
@group(0) @binding(3)
var<storage, read> gl_nodes: array<Node>;
@group(0) @binding(4)
var<uniform> params: Chords;
@group(0) @binding(5)
var<storage, read_write> out: array<f32>;

const MOMENT_STRIDE: u32 = 21u;

// ∫ R^{2+j} over [lo, hi] for j = 0, 1, 2.
fn powers(lo: f32, hi: f32) -> vec3<f32> {
    let a3 = lo * lo * lo;
    let b3 = hi * hi * hi;
    return vec3<f32>(
            (b3 - a3) / 3.0,
            (b3 * hi - a3 * lo) / 4.0,
            (b3 * hi * hi - a3 * lo * lo) / 5.0,
        );
}

// Core moments ∫ ρ R^{2+j} dR over one part of the composite rule.
fn core_part(
    mode: u32,
    lo: f32,
    hi: f32,
    b: f32,
    d: f32,
    s: f32,
    w: f32,
    alpha: f32,
    n_nodes: u32,
) -> vec3<f32> {
    var t0: f32;
    var t1: f32;
    if mode == 0u {
        t0 = atan((lo - b) / d);
        t1 = atan((hi - b) / d);
    } else if mode == 1u {
        t0 = log(lo - b);
        t1 = log(hi - b);
    } else {
        t0 = log(b - lo);
        t1 = log(b - hi);
    }
    let half = 0.5 * (t1 - t0);
    let mid = 0.5 * (t1 + t0);
    var acc = vec3<f32>(0.0);
    for(var j = 0u; j < n_nodes; j = j + 1u) {
        let nd = gl_nodes[j];
        let t = mid + half * nd.x;
        var delta: f32;
        var jac: f32;
        if mode == 0u {
            let ct = cos(t);
            delta = d * tan(t);
            jac = half * nd.w * d / (ct * ct);
        } else {
            let e = exp(t);
            delta = select(-e, e, mode == 1u);
            jac = abs(half) * nd.w * e;
        }
        let r = b + delta;
        let f = jac * w * pow(s * (delta * delta + d * d), -alpha) * r * r;
        acc = acc + f * vec3<f32>(1.0, r, r * r);
    }
    return acc;
}

fn core_moments(
    lo: f32,
    hi: f32,
    b: f32,
    d: f32,
    s: f32,
    w: f32,
    alpha: f32,
    p: Chords,
) -> vec3<f32> {
    let pl = b - p.kappa * d;
    let pr = b + p.kappa * d;
    var sum = vec3<f32>(0.0);
    if lo < min(hi, pl) {
        sum = sum + core_part(2u, lo, min(hi, pl), b, d, s, w, alpha, p.node_count);
    }
    let a = max(lo, pl);
    let z = min(hi, pr);
    if a < z {
        sum = sum + core_part(0u, a, z, b, d, s, w, alpha, p.node_count);
    }
    if max(lo, pr) < hi {
        sum = sum + core_part(1u, max(lo, pr), hi, b, d, s, w, alpha, p.node_count);
    }
    return sum;
}

@compute @workgroup_size(64)
fn rtfp_moments(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let p = params;
    if i >= p.ray_count {
        return;
    }
    let o = origin.xyz;
    let ray = rays[i];
    let dir = ray.d;
    var hits: array<f32, 64>;
    let n = collect_hits(o, dir, p, &hits);
    let base = i * MOMENT_STRIDE;
    if n > RT_MAX_HITS {
        for(var m = 0u; m < 20u; m = m + 1u) {
            out[base + m] = 0.0;
        }
        out[base + 20u] = 1.0;
        return;
    }
    // Radial moments M_j = ∫ ρ R^{2+j} dR summed over the occupied segments.
    var m0 = ds_zero();
    var m1 = ds_zero();
    var m2 = ds_zero();
    var open_r = select(-1.0, 0.0, (n & 1u) != 0u);
    for(var h = 0u; h < n; h = h + 1u) {
        let t = hits[h];
        if open_r < 0.0 {
            open_r = t;
            continue;
        }
        let lo = open_r;
        let hi = t;
        open_r = -1.0;
        var seg = p.bulk * powers(lo, hi);
        // Along y = o + R dir the core is seen from x = o with u = −dir.
        let u = -dir;
        for(var k = 0u; k < p.kernel_count; k = k + 1u) {
            let src = kerns[k];
            if src.w == 0.0 {
                continue;
            }
            let q0 = o - src.c;
            let su = sym_mul(src, u);
            let s = dot(u, su);
            let b = dot(su, q0) / s;
            let q = q0 - b * u;
            let d = sqrt((1.0 + dot(q, sym_mul(src, q))) / s);
            if !(d > 0.0) || !isfinite_f(d) {
                continue;
            }
            seg = seg + core_moments(lo, hi, b, d, s, src.w, src.alpha, p);
        }
        m0 = ds_add(m0, ds_f(seg.x));
        m1 = ds_add(m1, ds_f(seg.y));
        m2 = ds_add(m2, ds_f(seg.z));
    }
    // y = o + R dir: ∫ρ y = o M0 + dir M1, ∫ρ y yᵀ = o oᵀ M0 + (o dirᵀ + dir oᵀ) M1 + dir dirᵀ M2.
    let w = ds_f(ray.weight);
    m0 = ds_mul(w, m0);
    m1 = ds_mul(w, m1);
    m2 = ds_mul(w, m2);
    var v: array<ds, 10>;
    v[0] = m0;
    let oo = array<f32, 3>(o.x, o.y, o.z);
    let dd = array<f32, 3>(dir.x, dir.y, dir.z);
    for(var a = 0u; a < 3u; a = a + 1u) {
        v[1u + a] = ds_add(ds_mul(ds_f(oo[a]), m0), ds_mul(ds_f(dd[a]), m1));
    }
    let ia = array<u32, 6>(0u, 1u, 2u, 0u, 0u, 1u);
    let ib = array<u32, 6>(0u, 1u, 2u, 1u, 2u, 2u);
    for(var c = 0u; c < 6u; c = c + 1u) {
        let a = ia[c];
        let b = ib[c];
        let t0 = ds_mul(ds_mul(ds_f(oo[a]), ds_f(oo[b])), m0);
        let t1 = ds_mul(
                ds_add(ds_mul(ds_f(oo[a]), ds_f(dd[b])), ds_mul(ds_f(dd[a]), ds_f(oo[b]))),
                m1,
            );
        let t2 = ds_mul(ds_mul(ds_f(dd[a]), ds_f(dd[b])), m2);
        v[4u + c] = ds_add(ds_add(t0, t1), t2);
    }
    for(var m = 0u; m < 10u; m = m + 1u) {
        out[base + 2u * m] = v[m].hi;
        out[base + 2u * m + 1u] = v[m].lo;
    }
    out[base + 20u] = 0.0;
}
