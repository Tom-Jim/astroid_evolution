// SHARED
struct ModeParams {
    n: u32,
    g: u32,
    dense: u32,
    pad: u32,
    w_hi: f32,
    w_lo: f32,
    cutoff_hi: f32,
    cutoff_lo: f32,
    reject_hi: f32,
    reject_lo: f32,
}



fn ld_axis(v: u32, axis: u32) -> ds {
return ds(dirs[v * 3u + axis].x, dirs[v * 3u + axis].y);
}

fn dir_dot(a: u32, b: u32) -> ds {
var s: dsacc = acc_zero();
s = acc_fma(s, ld_axis(a, 0u), ld_axis(b, 0u));
s = acc_fma(s, ld_axis(a, 1u), ld_axis(b, 1u));
s = acc_fma(s, ld_axis(a, 2u), ld_axis(b, 2u));
return acc_done(s);
}

fn positive(a: ds) -> bool {
return a.hi > 0.0 || (a.hi == 0.0 && a.lo > 0.0);
}

fn kernel_weight(a: u32, b: u32, cos_raw: f32, p: ModeParams) -> f32 {
if (a == b) {
return 1.0;
}
if (p.dense == 0u && cos_raw < p.reject_hi - 1.0e-5) {
return 0.0;
}
var c: f32 = clamp(cos_raw, -1.0, 1.0);
var ang: f32 = acos(c);
var cutoff: f32 = p.cutoff_hi + p.cutoff_lo;
if (!(ang < cutoff)) {
return 0.0;
}
var t: f32 = ang / (p.w_hi + p.w_lo);
var weight: f32 = exp(-0.5 * t * t);
return select(0.0, weight, weight > 0.0);
}

fn cosine(a: u32, b: u32) -> f32 {
var d: ds = dir_dot(a, b);
return d.hi + d.lo;
}

// ENTRY mode_c_norm
@group(0) @binding(0) var<storage, read> dirs: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> area: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> mass_rate: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read> bin_start: array<u32>;
@group(0) @binding(4) var<storage, read> bin_items: array<u32>;
@group(0) @binding(5) var<storage, read> caps: array<u32>;
@group(0) @binding(6) var<storage, read_write> norm: array<vec2<f32>>;
@group(0) @binding(7) var<storage, read> p: ModeParams;
@compute @workgroup_size(64)
fn mode_c_norm(@builtin(global_invocation_id) gid3: vec3<u32>) {
    let src = gid3.x;
if (src >= p.n) {
return;
}
if (!positive(ds(mass_rate[src].x, mass_rate[src].y))) {
norm[src] = vec2<f32>((ds_zero()).hi, (ds_zero()).lo);
return;
}
var sum: dsacc = acc_zero();
if (p.dense != 0u) {
for (var t: u32 = 0u; t < p.n; t = t + 1u) {
var weight: f32 = kernel_weight(src, t, cosine(src, t), p);
if (weight > 0.0) {
sum = acc_fma(sum, ds_f(weight), ds(area[t].x, area[t].y));
}
}
} else {
var x0: u32 = caps[src * 6u];
var x1: u32 = caps[src * 6u + 1u];
var y0: u32 = caps[src * 6u + 2u];
var y1: u32 = caps[src * 6u + 3u];
var z0: u32 = caps[src * 6u + 4u];
var z1: u32 = caps[src * 6u + 5u];
for (var x: u32 = x0; x <= x1; x = x + 1u) {
for (var y: u32 = y0; y <= y1; y = y + 1u) {
for (var z: u32 = z0; z <= z1; z = z + 1u) {
var c: u32 = (x * p.g + y) * p.g + z;
var end: u32 = bin_start[c + 1u];
for (var k: u32 = bin_start[c]; k < end; k = k + 1u) {
var t: u32 = bin_items[k];
var weight: f32 = kernel_weight(src, t, cosine(src, t), p);
if (weight > 0.0) {
sum = acc_fma(sum, ds_f(weight), ds(area[t].x, area[t].y));
}
}
}
}
}
}
norm[src] = vec2<f32>((acc_done(sum)).hi, (acc_done(sum)).lo);
}

// END ENTRY mode_c_norm
// ENTRY mode_c_deposit
@group(0) @binding(0) var<storage, read> dirs: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> mass_rate: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> norm: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read> bin_start: array<u32>;
@group(0) @binding(4) var<storage, read> bin_items: array<u32>;
@group(0) @binding(5) var<storage, read> caps: array<u32>;
@group(0) @binding(6) var<storage, read_write> j_dep: array<vec2<f32>>;
@group(0) @binding(7) var<storage, read> p: ModeParams;
@compute @workgroup_size(64)
fn mode_c_deposit(@builtin(global_invocation_id) gid3: vec3<u32>) {
    let dst = gid3.x;
if (dst >= p.n) {
return;
}
var sum: dsacc = acc_zero();
if (p.dense != 0u) {
for (var src: u32 = 0u; src < p.n; src = src + 1u) {
var rate: ds = ds(mass_rate[src].x, mass_rate[src].y);
var nrm: ds = ds(norm[src].x, norm[src].y);
if (!positive(rate) || !positive(nrm)) {
continue;
}
var weight: f32 = kernel_weight(src, dst, cosine(src, dst), p);
if (weight > 0.0) {
sum = acc_add(sum, ds_mul(ds_f(weight), ds_div(rate, nrm)));
}
}
} else {
var x0: u32 = caps[dst * 6u];
var x1: u32 = caps[dst * 6u + 1u];
var y0: u32 = caps[dst * 6u + 2u];
var y1: u32 = caps[dst * 6u + 3u];
var z0: u32 = caps[dst * 6u + 4u];
var z1: u32 = caps[dst * 6u + 5u];
for (var x: u32 = x0; x <= x1; x = x + 1u) {
for (var y: u32 = y0; y <= y1; y = y + 1u) {
for (var z: u32 = z0; z <= z1; z = z + 1u) {
var c: u32 = (x * p.g + y) * p.g + z;
var end: u32 = bin_start[c + 1u];
for (var k: u32 = bin_start[c]; k < end; k = k + 1u) {
var src: u32 = bin_items[k];
var rate: ds = ds(mass_rate[src].x, mass_rate[src].y);
var nrm: ds = ds(norm[src].x, norm[src].y);
if (!positive(rate) || !positive(nrm)) {
continue;
}
var weight: f32 = kernel_weight(src, dst, cosine(src, dst), p);
if (weight > 0.0) {
sum = acc_add(sum, ds_mul(ds_f(weight), ds_div(rate, nrm)));
}
}
}
}
}
}
j_dep[dst] = vec2<f32>((acc_done(sum)).hi, (acc_done(sum)).lo);
}

// END ENTRY mode_c_deposit
