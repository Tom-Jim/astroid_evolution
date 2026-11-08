// Ray chords and the radial integrals of the RTFP field (MS §1.4, Lemma 1.4′, §3.1–3.3, §3.9).
//
// The field kernel integrates the density difference ρ(y) − ρ(x) (the exact
// uniform part ρ(x) Φ_V is `uniform.wesl`), so the constant density passed to
// `integrate_segment` is `bulk − ρ(x)`.
//
// Everything is in the unit frame of the body: lengths divided by the bounding
// radius, the metric `S` multiplied by its square. Along the ray `y = x − R u`
// (`u = −dir`) one anisotropic Cauchy core is
//
//   ρ_k(y) = w [1 + (p − R u)ᵀ S (p − R u)]^{−α} = w [s ((R − b)² + d²)]^{−α},
//   p = x − c,  s = uᵀSu,  b = uᵀSp / s,  d² = (1 + qᵀSq) / s,  q = p − b u.
//
// `q` is the S-orthogonal residual of `p`, so `d²` carries no cancellation.
// The three radial integrals per segment [R⁻, R⁺] are
//
//   I_U = ∫ ρ R dR,   I_g = ∫ ρ dR,   I_H = f.p.∫ ρ / R dR = ∫ (ρ − ρ₀)/R dR + ρ₀ ln(R⁺/R⁻),
//
// with `R⁻ = 0` replaced by the reference length `ℓ₀ = 1` (§3.9). `α = 1` uses the
// closed antiderivatives in double-single. Any other `α` uses the composite rule
// of §3.3: the tangent substitution `R = b + d tan θ` on |R − b| ≤ κ d and
// Gauss–Legendre in `ln|R − b|` outside, with the same nodes for I_g and I_H;
// `I_U = b I_g + [E(R⁺) − E(R⁻)] / (2 − 2α)`, `E = ρ D`, is closed.

// #include "gpu/shader/ds.wesl"
// #include "gpu/shader/rt.wesl"

struct Kern {
    c: vec3<f32>,
    alpha: f32,
    // Diagonal and off-diagonal (xy, xz, yz) entries of S.
    sd: vec3<f32>,
    w: f32,
    so: vec3<f32>,
    pad: f32,
}

struct Ray {
    d: vec3<f32>,
    weight: f32,
}

struct Node {
    x: f32,
    w: f32,
}

struct Chords {
    kernel_count: u32,
    node_count: u32,
    ray_count: u32,
    item_count: u32,
    want_u: u32,
    want_h: u32,
    bulk: f32,
    kappa: f32,
    merge_abs: f32,
    merge_rel: f32,
    origin_abs: f32,
    launch: f32,
    // Segment table 𝒯 of MS §3.15.1 (rtfp/shader/segments.wesl): when use_segs
    // is set, field threads read intervals instead of tracing, at
    // (seg_base + m)·ray_count + q.
    use_segs: u32,
    seg_base: u32,
    pad0: f32,
    pad1: f32,
}

struct Radial {
    u: ds,
    g: ds,
    h: ds,
}

const RT_MAX_HITS: u32 = 64u;

fn sym_mul(k: Kern, v: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
            k.sd.x * v.x + k.so.x * v.y + k.so.y * v.z,
            k.so.x * v.x + k.sd.y * v.y + k.so.z * v.z,
            k.so.y * v.x + k.so.z * v.y + k.sd.z * v.z,
        );
}

// log(1 + x) and exp(x) − 1 to a few ulps (Goldberg, Kahan).
fn log1p_f(x: f32) -> f32 {
    let u = 1.0 + x;
    if u == 1.0 {
        return x;
    }
    return log(u) * (x / (u - 1.0));
}

fn expm1_f(x: f32) -> f32 {
    if abs(x) < 1.0e-5 {
        return x + 0.5 * x * x;
    }
    let u = exp(x);
    if u == 1.0 {
        return x;
    }
    let um1 = u - 1.0;
    if um1 == -1.0 {
        return -1.0;
    }
    return um1 * (x / log(u));
}

// (eˣ − 1)/x, equal to 1 at 0.
fn expm1_over(x: f32) -> f32 {
    if abs(x) < 1.0e-5 {
        return 1.0 + 0.5 * x;
    }
    return expm1_f(x) / x;
}

// Every crossing of the ray with the surface at t ≥ launch, sorted, coincident
// reports merged and crossings at the launch point dropped. Returns the count, or
// RT_MAX_HITS + 1 when the ray crosses more often than the buffer holds.
fn collect_hits(
    origin: vec3<f32>,
    dir: vec3<f32>,
    p: Chords,
    hits: ptr<function, array<f32, 64>>,
) -> u32 {
    var raw: array<f32, 64>;
    let n = rt_all(origin, dir, p.launch, &raw);
    if n > RT_MAX_HITS {
        return RT_MAX_HITS + 1u;
    }
    // Insertion sort: a ray crosses a closed surface a few times.
    for(var i = 1u; i < n; i = i + 1u) {
        let v = raw[i];
        var j = i;
        while j > 0u && raw[j - 1u] > v {
            raw[j] = raw[j - 1u];
            j = j - 1u;
        }
        raw[j] = v;
    }
    var m = 0u;
    var last = -1.0e30;
    for(var i = 0u; i < n; i = i + 1u) {
        let t = raw[i];
        if !(isfinite_f(t) && t > p.origin_abs) {
            continue;
        }
        if m > 0u && t - last <= p.merge_abs + p.merge_rel * t {
            continue;
        }
        (*hits)[m] = t;
        last = t;
        m = m + 1u;
    }
    return m;
}

// The orientation rule of MS §3.6: every crossing carries s = sign(dir·n̂) of
// the crossed face (rt_all_signed). Reports closer than the geometric merge
// tolerance are one vertex/edge report group; a group's net sign decides
// whether occupancy changes — a tangency sums to zero and is dropped.
// `hits` receives (t, s) pairs, s ∈ {−1, +1}. Returns the count, or
// RT_MAX_HITS + 1 on buffer overflow.
fn collect_hits_o(
    origin: vec3<f32>,
    dir: vec3<f32>,
    p: Chords,
    hits: ptr<function, array<vec2<f32>, 64>>,
) -> u32 {
    var raw: array<vec2<f32>, 64>;
    let n = rt_all_signed(origin, dir, p.launch, &raw);
    if n > RT_MAX_HITS {
        return RT_MAX_HITS + 1u;
    }
    for(var i = 1u; i < n; i = i + 1u) {
        let v = raw[i];
        var j = i;
        while j > 0u && raw[j - 1u].x > v.x {
            raw[j] = raw[j - 1u];
            j = j - 1u;
        }
        raw[j] = v;
    }
    var m = 0u;
    var last = -1.0e30;
    var acc = 0.0;
    for(var i = 0u; i < n; i = i + 1u) {
        let t = raw[i].x;
        if !(isfinite_f(t) && t > p.origin_abs) {
            continue;
        }
        if m > 0u && t - last <= p.merge_abs + p.merge_rel * t {
            acc = acc + raw[i].y;
            continue;
        }
        if m > 0u {
            // Close the previous group: keep only its net sign.
            if abs(acc) < 0.5 {
                m = m - 1u;
            } else {
                (*hits)[m - 1u].y = sign(acc);
            }
        }
        (*hits)[m].x = t;
        acc = raw[i].y;
        last = t;
        m = m + 1u;
    }
    if m > 0u {
        if abs(acc) < 0.5 {
            m = m - 1u;
        } else {
            (*hits)[m - 1u].y = sign(acc);
        }
    }
    return m;
}

// α = 1: the antiderivatives P, Q, L of §3.2 with the logarithm of L split off
// into `ln_ratio`. `c1 = w / s`.
fn add_alpha_one(
    acc: ptr<function, Radial>,
    b: f32,
    d: f32,
    a0: f32,
    c1: f32,
    r0: f32,
    r1: f32,
    ln_ratio: ds,
    want_u: bool,
    want_h: bool,
) {
    let db = ds_f(b);
    let dd = ds_f(d);
    let x0 = ds_div(ds_sub(ds_f(r0), db), dd);
    let x1 = ds_div(ds_sub(ds_f(r1), db), dd);
    let dat = ds_atan2(ds_sub(x1, x0), ds_add(ds_f(1.0), ds_mul(x0, x1)));
    let c = ds_f(c1);
    (*acc).g = ds_add((*acc).g, ds_mul(c, ds_div(dat, dd)));
    if want_u || want_h {
        let half_ln = ds_mul(
                ds_f(0.5),
                ds_ln(ds_div(ds_add(ds_f(1.0), ds_mul(x1, x1)), ds_add(ds_f(1.0), ds_mul(x0, x0)))),
            );
        let bd_dat = ds_mul(ds_div(db, dd), dat);
        (*acc).u = ds_add((*acc).u, ds_mul(c, ds_add(half_ln, bd_dat)));
        if want_h {
            let ih = ds_div(ds_add(ds_sub(ln_ratio, half_ln), bd_dat), ds_f(a0));
            (*acc).h = ds_add((*acc).h, ds_mul(c, ih));
        }
    }
}

// One part of the composite rule. mode 0: tangent substitution on [lo, hi];
// mode 1: ln(R − b); mode 2: ln(b − R). Returns (∫ρ dR, ∫(ρ − ρ₀)/R dR).
fn radial_part(
    mode: u32,
    lo: f32,
    hi: f32,
    b: f32,
    d: f32,
    s: f32,
    w: f32,
    alpha: f32,
    a0: f32,
    f0: f32,
    n_nodes: u32,
    want_h: bool,
) -> vec2<f32> {
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
    var ig = 0.0;
    var ih = 0.0;
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
        let f = w * pow(s * (delta * delta + d * d), -alpha);
        ig = ig + jac * f;
        if want_h {
            // (ρ − ρ₀)/R without cancellation near R = 0: ρ/ρ₀ = (D/A₀)^{−α}. At
            // R → 0 the quotient tends to ∂ρ/∂R(0) = 2αb ρ₀/A₀ (§3.9, step 3).
            var quot: f32;
            if !(abs(r) > 1.0e-6 * (abs(b) + d)) {
                quot = 2.0 * alpha * b * f0 / a0;
            } else {
                let arg = r * (r - 2.0 * b) / a0;
                if abs(arg) < 0.5 {
                    quot = f0 * expm1_f(-alpha * log1p_f(arg)) / r;
                } else {
                    quot = (f - f0) / r;
                }
            }
            ih = ih + jac * quot;
        }
    }
    // mode 2 runs from b − lo down to b − hi: the orientation is in `abs(half)`.
    return vec2<f32>(ig, ih);
}

fn add_general(
    acc: ptr<function, Radial>,
    b: f32,
    d: f32,
    s: f32,
    w: f32,
    alpha: f32,
    a0: f32,
    f0: f32,
    r0: f32,
    r1: f32,
    ln_ratio: ds,
    p: Chords,
    want_u: bool,
    want_h: bool,
) {
    let pl = b - p.kappa * d;
    let pr = b + p.kappa * d;
    var sum = vec2<f32>(0.0, 0.0);
    if r0 < min(r1, pl) {
        sum = sum
            + radial_part(2u, r0, min(r1, pl), b, d, s, w, alpha, a0, f0, p.node_count, want_h);
    }
    let lo = max(r0, pl);
    let hi = min(r1, pr);
    if lo < hi {
        sum = sum + radial_part(0u, lo, hi, b, d, s, w, alpha, a0, f0, p.node_count, want_h);
    }
    if max(r0, pr) < r1 {
        sum = sum
            + radial_part(1u, max(r0, pr), r1, b, d, s, w, alpha, a0, f0, p.node_count, want_h);
    }
    (*acc).g = ds_add((*acc).g, ds_f(sum.x));
    if want_u || want_h {
        // I_U = b I_g + [E(R⁺) − E(R⁻)]/(2 − 2α), E = w s^{−α} D^{1−α}, in the form
        // E(R⁻) (e^{(1−α)L} − 1)/(2(1 − α)) with L = ln(D⁺/D⁻), stable at α → 1.
        let dm0 = r0 - b;
        let d0 = dm0 * dm0 + d * d;
        let e0 = w * pow(s * d0, -alpha) * d0;
        let l = log1p_f((r1 - r0) * (r1 + r0 - 2.0 * b) / d0);
        let iu = b * sum.x + 0.5 * e0 * l * expm1_over((1.0 - alpha) * l);
        (*acc).u = ds_add((*acc).u, ds_f(iu));
    }
    if want_h {
        (*acc).h = ds_add((*acc).h, ds_add(ds_f(sum.y), ds_mul(ds_f(f0), ln_ratio)));
    }
}

// The three integrals of one occupied segment [r0, r1] of the ray leaving `x`
// along `-u`, summed over the constant density `bulk` and every core.
fn integrate_segment(
    acc: ptr<function, Radial>,
    r0: f32,
    r1: f32,
    x: vec3<f32>,
    u: vec3<f32>,
    bulk: f32,
    p: Chords,
) {
    if !(r1 > r0) {
        return;
    }
    let want_h = p.want_h != 0u;
    let want_u = p.want_u != 0u;
    var ln_ratio = ds_zero();
    if want_h {
        // ℓ₀ = 1 in the unit frame. The term cancels in the angular sum wherever
        // the tangent cone is a sphere or a half-space (§3.9).
        ln_ratio = ds_ln(ds_div(ds_f(r1), ds_f(select(r0, 1.0, r0 == 0.0))));
    }
    if bulk != 0.0 {
        let a = ds_f(r0);
        let z = ds_f(r1);
        let rho = ds_f(bulk);
        (*acc).u = ds_add(
                (*acc).u,
                ds_mul(ds_mul(ds_f(0.5), rho), ds_sub(ds_mul(z, z), ds_mul(a, a))),
            );
        (*acc).g = ds_add((*acc).g, ds_mul(rho, ds_sub(z, a)));
        if want_h {
            (*acc).h = ds_add((*acc).h, ds_mul(rho, ln_ratio));
        }
    }
    for(var k = 0u; k < p.kernel_count; k = k + 1u) {
        let src = kerns[k];
        if src.w == 0.0 {
            continue;
        }
        let q0 = x - src.c;
        let su = sym_mul(src, u);
        let s = dot(u, su);
        let b = dot(su, q0) / s;
        let q = q0 - b * u;
        let d2 = (1.0 + dot(q, sym_mul(src, q))) / s;
        let d = sqrt(d2);
        let psp = dot(q0, sym_mul(src, q0));
        let a0 = b * b + d2;
        if !(d > 0.0) || !isfinite_f(d) || !isfinite_f(a0) {
            continue;
        }
        if src.alpha == 1.0 {
            add_alpha_one(acc, b, d, a0, src.w / s, r0, r1, ln_ratio, want_u, want_h);
        } else {
            let f0 = src.w * pow(1.0 + psp, -src.alpha);
            add_general(
                acc,
                b,
                d,
                s,
                src.w,
                src.alpha,
                a0,
                f0,
                r0,
                r1,
                ln_ratio,
                p,
                want_u,
                want_h,
            );
        }
    }
}
