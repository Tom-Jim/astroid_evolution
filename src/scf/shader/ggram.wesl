// (G1) Gram assembly of `scf/gbasis.rs`: the mass Gram `∫ρw B_b·B_c` and the
// strain Gram `∫ρw D_b:D_c` of Proposition 8.1.3 in one submission. The CPU
// path scans all `n_b²` mode pairs and sorts the union of two support lists
// per surviving pair; here per-support point lists are built once, the Gram
// is split into kernel×kernel, kernel×global and global×global blocks, and
// each workgroup accumulates its output pair over the smaller support list —
// no pair scan, no union sort.
//
// Layout: `supp_list[supp_off[s] .. supp_off[s+1])` holds the quadrature
// indices inside the ball of support s (sorted by construction). Kernel mode
// b of support s is `supp_base[s] + axis`. Non-kernel ("global") modes are
// evaluated once into `gvals[(p*n_global + g)*9 + c]`: value components 0..3,
// symmetric-strain Voigt components 3..9.

struct GgParams {
    n_points: u32,
    n_modes: u32,
    n_supp: u32,
    n_global: u32,
    n_kk: u32,
    n_gk: u32,
    n_gg: u32,
    want_strain: u32,
}

const SQRT2: f32 = 1.4142135623730951;

fn axis3(i: u32) -> vec3<f32> {
    var a = vec3<f32>(0.0);
    a[i] = 1.0;
    return a;
}

// A kernel mode's value and scaled symmetric strain at `x` for support
// `(c,h)` and unit axis `e`: `B = k(s)·(r×e)` with the Wendland C² profile
// (`k = −20q³/h²`, `k′ = 60q²/h³`), the same math as `gbasis.wesl` and the
// CPU `Mode::Kernel::{value,strain}`.
fn supp_vd(cx: f32, cy: f32, cz: f32, h: f32, e: vec3<f32>, x: vec3<f32>, out: ptr<function, array<f32, 9>>) {
    for (var t = 0u; t < 9u; t = t + 1u) {
        (*out)[t] = 0.0;
    }
    let r = x - vec3<f32>(cx, cy, cz);
    let s2 = dot(r, r);
    if s2 >= h * h || s2 <= 1e-30 {
        return;
    }
    let s = sqrt(s2);
    let q = 1.0 - s / h;
    let h2 = h * h;
    let k = -20.0 * q * q * q / h2;
    let dk = 60.0 * q * q / (h2 * h);
    let rh = r / s;
    let rx_e = cross(r, e);
    (*out)[0] = k * rx_e.x;
    (*out)[1] = k * rx_e.y;
    (*out)[2] = k * rx_e.z;
    // jac[i][j] = ∂_i B_j = dk·r̂_i·(r×e)_j + k·(e_i×e)_j.
    var j01 = 0.0;
    var j02 = 0.0;
    var j10 = 0.0;
    var j12 = 0.0;
    var j20 = 0.0;
    var j21 = 0.0;
    var jd = array<f32, 3>(0.0, 0.0, 0.0);
    for (var i = 0u; i < 3u; i = i + 1u) {
        let ci = cross(axis3(i), e);
        let j0 = dk * rh[i] * rx_e.x + k * ci.x;
        let j1 = dk * rh[i] * rx_e.y + k * ci.y;
        let j2 = dk * rh[i] * rx_e.z + k * ci.z;
        if i == 0u {
            jd[0] = j0;
            j01 = j1;
            j02 = j2;
        } else if i == 1u {
            j10 = j0;
            jd[1] = j1;
            j12 = j2;
        } else {
            j20 = j0;
            j21 = j1;
            jd[2] = j2;
        }
    }
    (*out)[3] = jd[0];
    (*out)[4] = jd[1];
    (*out)[5] = jd[2];
    (*out)[6] = SQRT2 * 0.5 * (j01 + j10);
    (*out)[7] = SQRT2 * 0.5 * (j02 + j20);
    (*out)[8] = SQRT2 * 0.5 * (j12 + j21);
}

fn ipow(x: f32, n: u32) -> f32 {
    var r = 1.0;
    for (var t = 0u; t < n; t = t + 1u) {
        r = r * x;
    }
    return r;
}

// Mixed partial of the monomial x^i y^j z^k — the `mono_d` of `gbasis.wesl`.
fn mono_d(ijk: vec3<u32>, nn: vec3<u32>, x: vec3<f32>) -> f32 {
    var v = 1.0;
    let xs = array<f32, 3>(x.x, x.y, x.z);
    for (var c = 0u; c < 3u; c = c + 1u) {
        if ijk[c] < nn[c] {
            return 0.0;
        }
        var coef = 1.0;
        for (var t = 0u; t < nn[c]; t = t + 1u) {
            coef = coef * f32(ijk[c] - t);
        }
        v = v * coef * ipow(xs[c], ijk[c] - nn[c]);
    }
    return v;
}

fn mono_dercol(ijk: vec3<u32>, nn: vec3<u32>, x: vec3<f32>) -> vec3<f32> {
    var g = vec3<f32>(0.0);
    for (var a = 0u; a < 3u; a = a + 1u) {
        var m = nn;
        m[a] = m[a] + 1u;
        g[a] = mono_d(ijk, m, x);
    }
    return g;
}

// Membership of point `p` in the support ball `(c,h)` — the predicate the
// fill kernel used to build `supp_list`.
fn memtest(p: u32, cx: f32, cy: f32, cz: f32, h: f32) -> bool {
    let dx = pts[3u * p] - cx;
    let dy = pts[3u * p + 1u] - cy;
    let dz = pts[3u * p + 2u] - cz;
    return dx * dx + dy * dy + dz * dz < h * h;
}

var<workgroup> red: array<f32, 64>;
var<workgroup> red64: array<u32, 64>;

// Each entry section declares only the bindings its kernel uses; the bind
// table is dense from 0 inside a section.

// ENTRY ggram_supp_count
@group(0) @binding(0) var<uniform> P: GgParams;
@group(0) @binding(1) var<storage, read> pts: array<f32>;
@group(0) @binding(2) var<storage, read> supps: array<f32>;
@group(0) @binding(3) var<storage, read_write> cnt: array<u32>;

@compute @workgroup_size(64)
fn ggram_supp_count(@builtin(global_invocation_id) gid: vec3<u32>) {
    let s = gid.x;
    if s >= P.n_supp {
        return;
    }
    let c = vec3<f32>(supps[4u * s], supps[4u * s + 1u], supps[4u * s + 2u]);
    let h = supps[4u * s + 3u];
    var n = 0u;
    for (var p = 0u; p < P.n_points; p = p + 1u) {
        let dx = pts[3u * p] - c.x;
        let dy = pts[3u * p + 1u] - c.y;
        let dz = pts[3u * p + 2u] - c.z;
        if dx * dx + dy * dy + dz * dz < h * h {
            n = n + 1u;
        }
    }
    cnt[s] = n;
}
// END ENTRY ggram_supp_count

// ENTRY ggram_supp_prefix
@group(0) @binding(0) var<uniform> P: GgParams;
@group(0) @binding(1) var<storage, read> cnt: array<u32>;
@group(0) @binding(2) var<storage, read_write> supp_off: array<u32>;

@compute @workgroup_size(64)
fn ggram_supp_prefix(@builtin(local_invocation_id) lid: vec3<u32>) {
    var acc = 0u;
    for (var s0 = 0u; s0 < P.n_supp; s0 = s0 + 64u) {
        let s = s0 + lid.x;
        var v = 0u;
        if s < P.n_supp {
            v = cnt[s];
        }
        // Hillis–Steele inclusive scan over 64 lanes.
        red64[lid.x] = v;
        workgroupBarrier();
        for (var d = 1u; d < 64u; d = d << 1u) {
            var t = 0u;
            if lid.x >= d {
                t = red64[lid.x - d];
            }
            workgroupBarrier();
            red64[lid.x] = red64[lid.x] + t;
            workgroupBarrier();
        }
        if s < P.n_supp {
            supp_off[s] = acc + red64[lid.x] - v;
        }
        acc = acc + red64[63];
        workgroupBarrier();
    }
    if lid.x == 0u {
        supp_off[P.n_supp] = acc;
    }
}
// END ENTRY ggram_supp_prefix

// ENTRY ggram_supp_fill
@group(0) @binding(0) var<uniform> P: GgParams;
@group(0) @binding(1) var<storage, read> pts: array<f32>;
@group(0) @binding(2) var<storage, read> supps: array<f32>;
@group(0) @binding(3) var<storage, read> supp_off: array<u32>;
@group(0) @binding(4) var<storage, read_write> supp_list: array<u32>;

@compute @workgroup_size(64)
fn ggram_supp_fill(@builtin(global_invocation_id) gid: vec3<u32>) {
    let s = gid.x;
    if s >= P.n_supp {
        return;
    }
    let c = vec3<f32>(supps[4u * s], supps[4u * s + 1u], supps[4u * s + 2u]);
    let h = supps[4u * s + 3u];
    var at = supp_off[s];
    for (var p = 0u; p < P.n_points; p = p + 1u) {
        let dx = pts[3u * p] - c.x;
        let dy = pts[3u * p + 1u] - c.y;
        let dz = pts[3u * p + 2u] - c.z;
        if dx * dx + dy * dy + dz * dz < h * h {
            supp_list[at] = p;
            at = at + 1u;
        }
    }
}
// END ENTRY ggram_supp_fill

// ENTRY ggram_eval_g
@group(0) @binding(0) var<uniform> P: GgParams;
@group(0) @binding(1) var<storage, read> pts: array<f32>;
@group(0) @binding(2) var<storage, read> modes: array<f32>;
@group(0) @binding(3) var<storage, read> globals: array<u32>;
@group(0) @binding(4) var<storage, read_write> gvals: array<f32>;

// The rigid, flux and polynomial mode values of `gbasis.wesl` — value in
// slots 0..3, scaled symmetric strain in 3..9.
fn gval_vd(m: u32, x: vec3<f32>, out: ptr<function, array<f32, 9>>) {
    let b = 8u * m;
    let f0 = vec3<f32>(modes[b], modes[b + 1u], modes[b + 2u]);
    let f1 = vec3<f32>(modes[b + 3u], modes[b + 4u], modes[b + 5u]);
    let kind = u32(modes[b + 7u] + 0.5);
    for (var k = 0u; k < 9u; k = k + 1u) {
        (*out)[k] = 0.0;
    }
    if kind == 0u {
        // Rigid: B = a + w×x; D = 0.
        let v = f0 + cross(f1, x);
        (*out)[0] = v.x;
        (*out)[1] = v.y;
        (*out)[2] = v.z;
    } else if kind == 2u {
        // Flux: B = −r/r³; J = −δ/r³ + 3rr/r⁵ (symmetric).
        let r = x - f0;
        let r2 = dot(r, r);
        if r2 > 1e-20 {
            let ir3 = 1.0 / (r2 * sqrt(r2));
            let ir5 = ir3 / r2;
            let v = -r * ir3;
            (*out)[0] = v.x;
            (*out)[1] = v.y;
            (*out)[2] = v.z;
            (*out)[3] = -ir3 + 3.0 * r.x * r.x * ir5;
            (*out)[4] = -ir3 + 3.0 * r.y * r.y * ir5;
            (*out)[5] = -ir3 + 3.0 * r.z * r.z * ir5;
            (*out)[6] = SQRT2 * (3.0 * r.x * r.y * ir5);
            (*out)[7] = SQRT2 * (3.0 * r.x * r.z * ir5);
            (*out)[8] = SQRT2 * (3.0 * r.y * r.z * ir5);
        }
    } else {
        // Poly: B = ∇m × e; J_b = (∂_b ∇m) × e.
        let e = f0;
        let ijk = vec3<u32>(u32(f1.x), u32(f1.y), u32(f1.z));
        let v = cross(mono_dercol(ijk, vec3<u32>(0u), x), e);
        (*out)[0] = v.x;
        (*out)[1] = v.y;
        (*out)[2] = v.z;
        var j = mat3x3<f32>();
        for (var bb = 0u; bb < 3u; bb = bb + 1u) {
            var nb = vec3<u32>(0u);
            nb[bb] = 1u;
            let row = cross(mono_dercol(ijk, nb, x), e);
            for (var a = 0u; a < 3u; a = a + 1u) {
                j[bb][a] = row[a];
            }
        }
        (*out)[3] = j[0][0];
        (*out)[4] = j[1][1];
        (*out)[5] = j[2][2];
        (*out)[6] = SQRT2 * 0.5 * (j[0][1] + j[1][0]);
        (*out)[7] = SQRT2 * 0.5 * (j[0][2] + j[2][0]);
        (*out)[8] = SQRT2 * 0.5 * (j[1][2] + j[2][1]);
    }
}

@compute @workgroup_size(64)
fn ggram_eval_g(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = gid.x;
    if p >= P.n_points {
        return;
    }
    let x = vec3<f32>(pts[3u * p], pts[3u * p + 1u], pts[3u * p + 2u]);
    var vd: array<f32, 9>;
    for (var g = 0u; g < P.n_global; g = g + 1u) {
        gval_vd(globals[g], x, &vd);
        let at = (p * P.n_global + g) * 9u;
        for (var k = 0u; k < 9u; k = k + 1u) {
            gvals[at + k] = vd[k];
        }
    }
}
// END ENTRY ggram_eval_g

// ENTRY ggram_kk
@group(0) @binding(0) var<uniform> P: GgParams;
@group(0) @binding(1) var<storage, read> pts: array<f32>;
@group(0) @binding(2) var<storage, read> pw: array<f32>;
@group(0) @binding(3) var<storage, read> supps: array<f32>;
@group(0) @binding(4) var<storage, read> supp_base: array<u32>;
@group(0) @binding(5) var<storage, read> supp_off: array<u32>;
@group(0) @binding(6) var<storage, read> supp_list: array<u32>;
@group(0) @binding(7) var<storage, read> pairs_kk: array<u32>;
@group(0) @binding(8) var<storage, read_write> out_m: array<f32>;
@group(0) @binding(9) var<storage, read_write> out_s: array<f32>;

// One thread per support pair (s1,s2), s1 ≤ s2. Iterates the smaller support
// list and membership-tests the other; each accepted point is evaluated once
// for both supports, then the 9 mass and 9 strain axis-pairs accumulate in
// registers.
@compute @workgroup_size(64)
fn ggram_kk(@builtin(global_invocation_id) gid: vec3<u32>) {
    let rec = gid.x;
    if rec >= P.n_kk {
        return;
    }
    let s1 = pairs_kk[2u * rec];
    let s2 = pairs_kk[2u * rec + 1u];
    let c1 = vec3<f32>(supps[4u * s1], supps[4u * s1 + 1u], supps[4u * s1 + 2u]);
    let h1 = supps[4u * s1 + 3u];
    let c2 = vec3<f32>(supps[4u * s2], supps[4u * s2 + 1u], supps[4u * s2 + 2u]);
    let h2 = supps[4u * s2 + 3u];
    let n1 = supp_off[s1 + 1u] - supp_off[s1];
    let n2 = supp_off[s2 + 1u] - supp_off[s2];
    // Iterate the shorter list; test the longer's ball.
    var it = s1;
    var oc = c2;
    var oh = h2;
    if n2 < n1 {
        it = s2;
        oc = c1;
        oh = h1;
    }
    let lo = supp_off[it];
    let hi = supp_off[it + 1u];
    var acc: array<f32, 18>;
    for (var k = 0u; k < 18u; k = k + 1u) {
        acc[k] = 0.0;
    }
    var vd1: array<f32, 9>;
    var vd2: array<f32, 9>;
    let axes = array<vec3<f32>, 3>(
        vec3<f32>(1.0, 0.0, 0.0),
        vec3<f32>(0.0, 1.0, 0.0),
        vec3<f32>(0.0, 0.0, 1.0),
    );
    for (var t = lo; t < hi; t = t + 1u) {
        let p = supp_list[t];
        if s1 != s2 && !memtest(p, oc.x, oc.y, oc.z, oh) {
            continue;
        }
        let x = vec3<f32>(pts[3u * p], pts[3u * p + 1u], pts[3u * p + 2u]);
        let w = pw[p];
        for (var a1 = 0u; a1 < 3u; a1 = a1 + 1u) {
            supp_vd(c1.x, c1.y, c1.z, h1, axes[a1], x, &vd1);
            for (var a2 = 0u; a2 < 3u; a2 = a2 + 1u) {
                supp_vd(c2.x, c2.y, c2.z, h2, axes[a2], x, &vd2);
                let j = a1 * 3u + a2;
                acc[j] = acc[j] + w * (vd1[0] * vd2[0] + vd1[1] * vd2[1] + vd1[2] * vd2[2]);
                var ds = 0.0;
                for (var k = 0u; k < 6u; k = k + 1u) {
                    ds = ds + vd1[3u + k] * vd2[3u + k];
                }
                acc[9u + j] = acc[9u + j] + w * ds;
            }
        }
    }
    // Write the 9 mass and 9 strain entries mirrored across the diagonal.
    let nm = P.n_modes;
    for (var j = 0u; j < 9u; j = j + 1u) {
        let a1 = j / 3u;
        let a2 = j % 3u;
        let b1 = supp_base[s1] + a1;
        let b2 = supp_base[s2] + a2;
        out_m[b1 * nm + b2] = acc[j];
        out_m[b2 * nm + b1] = acc[j];
        if P.want_strain == 1u {
            out_s[b1 * nm + b2] = acc[9u + j];
            out_s[b2 * nm + b1] = acc[9u + j];
        }
    }
}
// END ENTRY ggram_kk
// ENTRY ggram_gk
@group(0) @binding(0) var<uniform> P: GgParams;
@group(0) @binding(1) var<storage, read> pts: array<f32>;
@group(0) @binding(2) var<storage, read> pw: array<f32>;
@group(0) @binding(3) var<storage, read> supps: array<f32>;
@group(0) @binding(4) var<storage, read> supp_base: array<u32>;
@group(0) @binding(5) var<storage, read> supp_off: array<u32>;
@group(0) @binding(6) var<storage, read> supp_list: array<u32>;
@group(0) @binding(7) var<storage, read> pairs_gk: array<u32>;
@group(0) @binding(8) var<storage, read> globals: array<u32>;
@group(0) @binding(9) var<storage, read> gvals: array<f32>;
@group(0) @binding(10) var<storage, read_write> out_m: array<f32>;
@group(0) @binding(11) var<storage, read_write> out_s: array<f32>;

// One thread per (support, global position) record. Iterates the support
// list and contracts the three kernel axes against the precomputed global
// values.
@compute @workgroup_size(64)
fn ggram_gk(@builtin(global_invocation_id) gid: vec3<u32>) {
    let rec = gid.x;
    if rec >= P.n_gk {
        return;
    }
    let s = pairs_gk[2u * rec];
    let g = pairs_gk[2u * rec + 1u];
    let gm = globals[g];
    let c1 = vec3<f32>(supps[4u * s], supps[4u * s + 1u], supps[4u * s + 2u]);
    let h1 = supps[4u * s + 3u];
    let lo = supp_off[s];
    let hi = supp_off[s + 1u];
    var acc: array<f32, 6>;
    for (var k = 0u; k < 6u; k = k + 1u) {
        acc[k] = 0.0;
    }
    var vd: array<f32, 9>;
    let axes = array<vec3<f32>, 3>(
        vec3<f32>(1.0, 0.0, 0.0),
        vec3<f32>(0.0, 1.0, 0.0),
        vec3<f32>(0.0, 0.0, 1.0),
    );
    for (var t = lo; t < hi; t = t + 1u) {
        let p = supp_list[t];
        let w = pw[p];
        let gv = (p * P.n_global + g) * 9u;
        for (var a = 0u; a < 3u; a = a + 1u) {
            supp_vd(c1.x, c1.y, c1.z, h1, axes[a],
                vec3<f32>(pts[3u * p], pts[3u * p + 1u], pts[3u * p + 2u]), &vd);
            acc[a] = acc[a] + w * (vd[0] * gvals[gv] + vd[1] * gvals[gv + 1u] + vd[2] * gvals[gv + 2u]);
            var ds = 0.0;
            for (var k = 0u; k < 6u; k = k + 1u) {
                ds = ds + vd[3u + k] * gvals[gv + 3u + k];
            }
            acc[3u + a] = acc[3u + a] + w * ds;
        }
    }
    let nm = P.n_modes;
    for (var a = 0u; a < 3u; a = a + 1u) {
        let b = supp_base[s] + a;
        out_m[b * nm + gm] = acc[a];
        out_m[gm * nm + b] = acc[a];
        if P.want_strain == 1u {
            out_s[b * nm + gm] = acc[3u + a];
            out_s[gm * nm + b] = acc[3u + a];
        }
    }
}
// END ENTRY ggram_gk

// ENTRY ggram_gg
@group(0) @binding(0) var<uniform> P: GgParams;
@group(0) @binding(1) var<storage, read> pw: array<f32>;
@group(0) @binding(2) var<storage, read> pairs_gg: array<u32>;
@group(0) @binding(3) var<storage, read> gvals: array<f32>;
@group(0) @binding(4) var<storage, read> globals: array<u32>;
@group(0) @binding(5) var<storage, read_write> out_m: array<f32>;
@group(0) @binding(6) var<storage, read_write> out_s: array<f32>;

// One thread per global×global pair — a dense contract over the whole
// quadrature against `gvals`.
@compute @workgroup_size(64)
fn ggram_gg(@builtin(global_invocation_id) gid: vec3<u32>) {
    let rec = gid.x;
    if rec >= P.n_gg {
        return;
    }
    // A record carries positions in `globals`; the mode index is globals[i].
    let b = pairs_gg[2u * rec];
    let c = pairs_gg[2u * rec + 1u];
    var am = 0.0;
    var as_ = 0.0;
    for (var p = 0u; p < P.n_points; p = p + 1u) {
        let w = pw[p];
        let ib = (p * P.n_global + b) * 9u;
        let ic = (p * P.n_global + c) * 9u;
        let vb = vec3<f32>(gvals[ib], gvals[ib + 1u], gvals[ib + 2u]);
        let vc = vec3<f32>(gvals[ic], gvals[ic + 1u], gvals[ic + 2u]);
        am = am + w * dot(vb, vc);
        var ds = 0.0;
        for (var a = 0u; a < 6u; a = a + 1u) {
            ds = ds + gvals[ib + 3u + a] * gvals[ic + 3u + a];
        }
        as_ = as_ + w * ds;
    }
    let nm = P.n_modes;
    let gm = globals[b];
    let gc = globals[c];
    out_m[gm * nm + gc] = am;
    out_m[gc * nm + gm] = am;
    if P.want_strain == 1u {
        out_s[gm * nm + gc] = as_;
        out_s[gc * nm + gm] = as_;
    }
}
// END ENTRY ggram_gg
