// SHARED
struct GpuPoint {
    ox: f32,
    oy: f32,
    oz: f32,
    theta: f32,
    wx: f32,
    wy: f32,
    wz: f32,
    rho: f32,
}

struct GpuRay {
    dx: f32,
    dy: f32,
    dz: f32,
    weight: f32,
}

struct GpuKernel {
    cx: f32,
    cy: f32,
    cz: f32,
    inv_sigma2: f32,
    coeff: f32,
    alpha: f32,
}

struct GpuNode {
    x: f32,
    w: f32,
}

struct FieldParams {
    kernel_count: u32,
    node_count: u32,
    node_hi: u32,
    ray_count: u32,
    want_potential: u32,
    want_hessian: u32,
    bulk: f32,
    ell: f32,
    body_scale: f32,
    g: f32,
    merge_rel: f32,
    merge_abs: f32,
    origin_abs: f32,
    launch_offset: f32,
    point_count: u32,
}

struct MomentParams {
    kernel_count: u32,
    node_count: u32,
    ray_count: u32,
    max_panels: u32,
    bulk: f32,
    finest: f32,
    body_scale: f32,
    launch_offset: f32,
    merge_rel: f32,
    merge_abs: f32,
    origin_abs: f32,
    ox: f32,
    oy: f32,
    oz: f32,
    wx: f32,
    wy: f32,
    wz: f32,
    pad: f32,
}

struct Scalar {
    u: ds,
    g: ds,
    h: ds,
}

fn scalar_zero() -> Scalar {
return Scalar(ds_zero(), ds_zero(), ds_zero());
}

fn chord_ln(r0: f32, r1: f32, ell: f32, want_h: u32) -> ds {
if (want_h == 0u) {
return ds_zero();
}
if (r0 == 0.0) {
return ds_ln(ds_div(ds_from_float(r1), ds_from_float(ell)));
}
return ds_ln(ds_div(ds_from_float(r1), ds_from_float(r0)));
}

fn add_alpha_one(acc: ptr<function, Scalar>, b: f32, d: f32, a0: f32, coeff: f32, r0: f32, r1: f32, ln_ratio: ds, want_u: u32, want_h: u32) {
var db: ds = ds_from_float(b);
var dd: ds = ds_from_float(d);
var x0: ds = ds_div(ds_sub(ds_from_float(r0), db), dd);
var x1: ds = ds_div(ds_sub(ds_from_float(r1), db), dd);
var dat: ds = ds_atan2(ds_sub(x1, x0), ds_add(ds_from_float(1.0), ds_mul(x0, x1)));
var ig: ds = ds_div(dat, dd);
var iu: ds = ds_zero();
var ih: ds = ds_zero();
if (want_u != 0u || want_h != 0u) {
var half_ln: ds = ds_mul(
            ds_from_float(0.5),
            ds_ln(ds_div(
                ds_add(ds_from_float(1.0), ds_mul(x1, x1)),
                ds_add(ds_from_float(1.0), ds_mul(x0, x0)))));
var bd_dat: ds = ds_mul(ds_div(db, dd), dat);
iu = ds_add(half_ln, bd_dat);
if (want_h != 0u) {
ih = ds_div(ds_add(ds_sub(ln_ratio, half_ln), bd_dat), ds_from_float(a0));
}
}
var c: ds = ds_from_float(coeff);
(*acc).u = ds_add((*acc).u, ds_mul(c, iu));
(*acc).g = ds_add((*acc).g, ds_mul(c, ig));
(*acc).h = ds_add((*acc).h, ds_mul(c, ih));
}

fn log1p_f(x: f32) -> f32 {
if (abs(x) < 1.0e-4) {
return x - 0.5 * x * x;
}
return log(1.0 + x);
}

fn expm1_f(x: f32) -> f32 {
if (abs(x) < 1.0e-4) {
return x + 0.5 * x * x;
}
return exp(x) - 1.0;
}

fn add_general(acc: ptr<function, Scalar>, alpha: f32, b: f32, d: f32, a0: f32, coeff: f32, r0: f32, r1: f32, ln_ratio: ds, node_count: u32, node_hi: u32, want_u: u32, want_h: u32) {
if (!(a0 > 0.0) || !isfinite_f(a0)) {
return;
}
var span: f32 = r1 - r0;
if (!(span > 0.0) || node_count == 0u) {
return;
}
var nnode: u32 = node_count;
var base: u32 = 0u;
if (node_hi != 0u && span > 8.0 * d) {
nnode = node_hi;
base = node_count;
}
var half_span: f32 = 0.5 * span;
var mid: f32 = 0.5 * (r0 + r1);
var a0_pow: f32 = pow(a0, -alpha);
var reg_scale: f32 = 1.0e-14 * (abs(b) + d);
var iu: ds = ds_zero();
var ig: ds = ds_zero();
var ih: ds = ds_zero();
for (var n: u32 = 0u; n < nnode; n = n + 1u) {
var node: GpuNode = nodes[base + n];
var radius: f32 = mid + half_span * node.x;
var gw: f32 = half_span * node.w;
var delta: f32 = radius - b;
var shape: f32 = pow(delta * delta + d * d, -alpha);
var wd: ds = ds_from_float(gw);
var dens: ds = ds_from_float(shape);
iu = ds_add(iu, ds_mul(ds_mul(wd, dens), ds_from_float(radius)));
ig = ds_add(ig, ds_mul(wd, dens));
if (want_h != 0u) {
var regular: f32;
if (!(radius > reg_scale)) {
regular = (2.0 * alpha * b * a0_pow) / a0;
} else {
var arg: f32 = radius * (radius - 2.0 * b) / a0;
if (arg > -1.0) {
regular = a0_pow * expm1_f(-alpha * log1p_f(arg)) / radius;
} else {
regular = (shape - a0_pow) / radius;
}
}
ih = ds_add(ih, ds_mul(wd, ds_from_float(regular)));
}
}
if (want_h != 0u) {
ih = ds_add(ih, ds_mul(ds_from_float(a0_pow), ln_ratio));
}
var c: ds = ds_from_float(coeff);
if (want_u != 0u || want_h != 0u) {
(*acc).u = ds_add((*acc).u, ds_mul(c, iu));
}
(*acc).g = ds_add((*acc).g, ds_mul(c, ig));
if (want_h != 0u) {
(*acc).h = ds_add((*acc).h, ds_mul(c, ih));
}
}

fn integrate_chord(acc: ptr<function, Scalar>, r0: f32, r1: f32, world: vec3<f32>, unit_from_point: vec3<f32>, ell: f32, bulk: f32, kernel_count: u32, node_count: u32, node_hi: u32, want_u: u32, want_h: u32) {
if (!(r1 > r0)) {
return;
}
var ln_ratio: ds = chord_ln(r0, r1, ell, want_h);
if (bulk != 0.0) {
var r0d: ds = ds_from_float(r0);
var r1d: ds = ds_from_float(r1);
var dbulk: ds = ds_from_float(bulk);
var r2: ds = ds_sub(ds_mul(r1d, r1d), ds_mul(r0d, r0d));
(*acc).u = ds_add((*acc).u, ds_mul(ds_mul(ds_from_float(0.5), dbulk), r2));
(*acc).g = ds_add((*acc).g, ds_mul(dbulk, ds_sub(r1d, r0d)));
if (want_h != 0u) {
(*acc).h = ds_add((*acc).h, ds_mul(dbulk, ln_ratio));
}
}
for (var k: u32 = 0u; k < kernel_count; k = k + 1u) {
var src: GpuKernel = kernels[k];
if (src.coeff == 0.0) {
continue;
}
var p: vec3<f32> = world - vec3<f32>(src.cx, src.cy, src.cz);
var b: f32 = dot(p, unit_from_point);
var a0: f32 = src.inv_sigma2 + dot(p, p);
var perp: vec3<f32> = p - unit_from_point * b;
var d2: f32 = src.inv_sigma2 + dot(perp, perp);
var d: f32 = sqrt(d2);
if (!(d > 0.0) || !isfinite_f(d) || !isfinite_f(a0)) {
continue;
}
if (src.alpha == 1.0) {
add_alpha_one(acc, b, d, a0, src.coeff, r0, r1, ln_ratio, want_u, want_h);
} else {
add_general(acc, src.alpha, b, d, a0, src.coeff, r0, r1, ln_ratio, node_count, node_hi, want_u, want_h);
}
}
}

fn store_pair(slot: u32, value: ds) {
red_hi[slot] = value.hi;
red_lo[slot] = value.lo;
}

fn load_pair(slot: u32) -> ds {
return ds(red_hi[slot], red_lo[slot]);
}

fn reduce10(tid: u32) {
workgroupBarrier();
for (var s: u32 = 64u / 2u; s > 0u; s = s >> 1u) {
if (tid < s) {
for (var m: u32 = 0u; m < 10u; m = m + 1u) {
var slot: u32 = m * 64u + tid;
var other: u32 = m * 64u + tid + s;
var sum: ds = ds_add(load_pair(slot), load_pair(other));
store_pair(slot, sum);
}
ov[tid] = ov[tid] | ov[tid + s];
}
workgroupBarrier();
}
}

fn density_at(y: vec3<f32>, bulk: f32, kernel_count: u32) -> f32 {
var rho: f32 = bulk;
for (var k: u32 = 0u; k < kernel_count; k = k + 1u) {
var src: GpuKernel = kernels[k];
var d: vec3<f32> = y - vec3<f32>(src.cx, src.cy, src.cz);
var base: f32 = src.inv_sigma2 + dot(d, d);
rho = rho + (src.coeff * pow(base, -src.alpha));
}
return rho;
}

// ENTRY rtfp_field
@group(0) @binding(0) var<storage, read> points: array<GpuPoint>;
@group(0) @binding(1) var<storage, read> rays: array<GpuRay>;
@group(0) @binding(2) var<storage, read> kernels: array<GpuKernel>;
@group(0) @binding(3) var<storage, read> nodes: array<GpuNode>;
@group(0) @binding(4) var<storage, read> params: FieldParams;
@group(0) @binding(5) var<storage, read_write> out: array<f32>;
@group(0) @binding(6) var<storage, read_write> flags: array<u32>;
@compute @workgroup_size(64)
fn rtfp_field(@builtin(global_invocation_id) gid3: vec3<u32>) {
    let gid = gid3;
var pid: u32 = gid.x;
if (pid >= params.point_count) {
return;
}
var pt: GpuPoint = points[pid];
var origin_u: vec3<f32> = vec3<f32>(pt.ox, pt.oy, pt.oz);
var world: vec3<f32> = vec3<f32>(pt.wx, pt.wy, pt.wz);
var want_h: u32 = params.want_hessian;
var want_u: u32 = select(0u, 1u, (params.want_potential != 0u || want_h != 0u));
var part: array<ds, 10>;
for (var m: u32 = 0u; m < 10u; m = m + 1u) {
part[m] = ds_zero();
}
var overflow: bool = false;
var merged: array<f32, 128>;
for (var i: u32 = 0u; i < params.ray_count; i = i + 1u) {
var ray: GpuRay = rays[i];
if (!isfinite_f(ray.weight) || ray.weight == 0.0) {
continue;
}
var dir: vec3<f32> = vec3<f32>(ray.dx, ray.dy, ray.dz);
var len2: f32 = dot(dir, dir);
if (!(len2 > 0.0) || !isfinite_f(len2)) {
continue;
}
dir = dir * (inverseSqrt(len2));
var ray_overflow: bool = false;
var n: u32 = bvh_collect(origin_u, dir, params.body_scale, params.launch_offset, params.origin_abs, params.merge_abs, params.merge_rel, &merged, &ray_overflow);
if (ray_overflow) {
overflow = true;
continue;
}
var acc: Scalar = scalar_zero();
var open_r: f32 = select(-1.0, 0.0, (n & 1u) != 0u);
for (var h: u32 = 0u; h < n; h = h + 1u) {
var t: f32 = merged[h];
if (open_r >= 0.0) {
integrate_chord(&acc, open_r, t, world, -dir, params.ell, params.bulk, params.kernel_count, params.node_count, params.node_hi, want_u, want_h);
open_r = -1.0;
} else {
open_r = t;
}
}
var weight: ds = ds_from_float(ray.weight);
var wu: ds = ds_mul(weight, acc.u);
var wg: ds = ds_mul(weight, acc.g);
part[0] = ds_add(part[0], wu);
part[1] = ds_add(part[1], ds_mul(ds_from_float(dir.x), wg));
part[2] = ds_add(part[2], ds_mul(ds_from_float(dir.y), wg));
part[3] = ds_add(part[3], ds_mul(ds_from_float(dir.z), wg));
if (want_h != 0u) {
var wh: ds = ds_mul(weight, acc.h);
var txx: f32 = fma(3.0 * dir.x, dir.x, -1.0);
var tyy: f32 = fma(3.0 * dir.y, dir.y, -1.0);
var tzz: f32 = fma(3.0 * dir.z, dir.z, -1.0);
part[4] = ds_add(part[4], ds_mul(ds_from_float(txx), wh));
part[5] = ds_add(part[5], ds_mul(ds_from_float(tyy), wh));
part[6] = ds_add(part[6], ds_mul(ds_from_float(tzz), wh));
part[7] = ds_add(part[7], ds_mul(ds_from_float(3.0 * dir.x * dir.y), wh));
part[8] = ds_add(part[8], ds_mul(ds_from_float(3.0 * dir.x * dir.z), wh));
part[9] = ds_add(part[9], ds_mul(ds_from_float(3.0 * dir.y * dir.z), wh));
}
}
if (overflow) {
for (var m: u32 = 0u; m < 10u; m = m + 1u) {
part[m] = ds_zero();
}
}
var gscale: ds = ds_from_float(params.g);
for (var m: u32 = 0u; m < 10u; m = m + 1u) {
part[m] = ds_mul(part[m], gscale);
}
if (want_h != 0u && !overflow) {
var contact: ds = ds_mul(
            ds_mul(ds_from_float(-4.0 / 3.0), ds_pi()),
            ds_mul(gscale, ds_mul(ds_from_float(pt.theta), ds_from_float(pt.rho))));
for (var m: u32 = 4u; m < 7u; m = m + 1u) {
part[m] = ds_add(part[m], contact);
}
}
var base: u32 = pid * 20u;
for (var m: u32 = 0u; m < 10u; m = m + 1u) {
out[base + 2u * m] = part[m].hi;
out[base + 2u * m + 1u] = part[m].lo;
}
flags[pid] = select(0u, 1u, overflow);
}

// END ENTRY rtfp_field
// ENTRY rtfp_field_strided
var<workgroup> red_hi: array<f32, 640>;
var<workgroup> red_lo: array<f32, 640>;
var<workgroup> ov: array<u32, 64>;
@group(0) @binding(0) var<storage, read> points: array<GpuPoint>;
@group(0) @binding(1) var<storage, read> rays: array<GpuRay>;
@group(0) @binding(2) var<storage, read> kernels: array<GpuKernel>;
@group(0) @binding(3) var<storage, read> nodes: array<GpuNode>;
@group(0) @binding(4) var<storage, read> params: FieldParams;
@group(0) @binding(5) var<storage, read_write> out: array<f32>;
@group(0) @binding(6) var<storage, read_write> flags: array<u32>;
@compute @workgroup_size(64)
fn rtfp_field_strided(@builtin(local_invocation_id) lid3: vec3<u32>, @builtin(workgroup_id) wgid3: vec3<u32>) {
    let tid = lid3.x;
    let tgid = wgid3.x;
var pt: GpuPoint = points[tgid];
var origin_u: vec3<f32> = vec3<f32>(pt.ox, pt.oy, pt.oz);
var world: vec3<f32> = vec3<f32>(pt.wx, pt.wy, pt.wz);
var want_h: u32 = params.want_hessian;
var want_u: u32 = select(0u, 1u, (params.want_potential != 0u || want_h != 0u));
var part: array<ds, 10>;
for (var m: u32 = 0u; m < 10u; m = m + 1u) {
part[m] = ds_zero();
}
var overflow: bool = false;
var merged: array<f32, 128>;
for (var i: u32 = tid; i < params.ray_count; i = i + (64u)) {
var ray: GpuRay = rays[i];
if (!isfinite_f(ray.weight) || ray.weight == 0.0) {
continue;
}
var dir: vec3<f32> = vec3<f32>(ray.dx, ray.dy, ray.dz);
var len2: f32 = dot(dir, dir);
if (!(len2 > 0.0) || !isfinite_f(len2)) {
continue;
}
dir = dir * (inverseSqrt(len2));
var ray_overflow: bool = false;
var n: u32 = bvh_collect(origin_u, dir, params.body_scale, params.launch_offset, params.origin_abs, params.merge_abs, params.merge_rel, &merged, &ray_overflow);
if (ray_overflow) {
overflow = true;
continue;
}
var acc: Scalar = scalar_zero();
var open_r: f32 = select(-1.0, 0.0, (n & 1u) != 0u);
for (var h: u32 = 0u; h < n; h = h + 1u) {
var t: f32 = merged[h];
if (open_r >= 0.0) {
integrate_chord(&acc, open_r, t, world, -dir, params.ell, params.bulk, params.kernel_count, params.node_count, params.node_hi, want_u, want_h);
open_r = -1.0;
} else {
open_r = t;
}
}
var weight: ds = ds_from_float(ray.weight);
var wu: ds = ds_mul(weight, acc.u);
var wg: ds = ds_mul(weight, acc.g);
part[0] = ds_add(part[0], wu);
part[1] = ds_add(part[1], ds_mul(ds_from_float(dir.x), wg));
part[2] = ds_add(part[2], ds_mul(ds_from_float(dir.y), wg));
part[3] = ds_add(part[3], ds_mul(ds_from_float(dir.z), wg));
if (want_h != 0u) {
var wh: ds = ds_mul(weight, acc.h);
var txx: f32 = fma(3.0 * dir.x, dir.x, -1.0);
var tyy: f32 = fma(3.0 * dir.y, dir.y, -1.0);
var tzz: f32 = fma(3.0 * dir.z, dir.z, -1.0);
part[4] = ds_add(part[4], ds_mul(ds_from_float(txx), wh));
part[5] = ds_add(part[5], ds_mul(ds_from_float(tyy), wh));
part[6] = ds_add(part[6], ds_mul(ds_from_float(tzz), wh));
part[7] = ds_add(part[7], ds_mul(ds_from_float(3.0 * dir.x * dir.y), wh));
part[8] = ds_add(part[8], ds_mul(ds_from_float(3.0 * dir.x * dir.z), wh));
part[9] = ds_add(part[9], ds_mul(ds_from_float(3.0 * dir.y * dir.z), wh));
}
}
if (overflow) {
for (var m: u32 = 0u; m < 10u; m = m + 1u) {
part[m] = ds_zero();
}
}



for (var m: u32 = 0u; m < 10u; m = m + 1u) {
store_pair(m * 64u + tid, part[m]);
}
ov[tid] = select(0u, 1u, overflow);
reduce10(tid);
if (tid == 0u) {
var gscale: ds = ds_from_float(params.g);
for (var m: u32 = 0u; m < 10u; m = m + 1u) {
var slot: u32 = m * 64u;
store_pair(slot, ds_mul(load_pair(slot), gscale));
}
if (want_h != 0u) {
var contact: ds = ds_mul(
                ds_mul(ds_from_float(-4.0 / 3.0), ds_pi()),
                ds_mul(gscale, ds_mul(ds_from_float(pt.theta), ds_from_float(pt.rho))));
for (var m: u32 = 4u; m < 7u; m = m + 1u) {
var slot: u32 = m * 64u;
store_pair(slot, ds_add(load_pair(slot), contact));
}
}
var base: u32 = tgid * 20u;
for (var m: u32 = 0u; m < 10u; m = m + 1u) {
var value: ds = load_pair(m * 64u);
out[base + 2u * m] = value.hi;
out[base + 2u * m + 1u] = value.lo;
}
flags[tgid] = ov[0];
}
}

// END ENTRY rtfp_field_strided
// ENTRY rtfp_moments
var<workgroup> red_hi: array<f32, 640>;
var<workgroup> red_lo: array<f32, 640>;
var<workgroup> ov: array<u32, 64>;
@group(0) @binding(0) var<storage, read> rays: array<GpuRay>;
@group(0) @binding(1) var<storage, read> kernels: array<GpuKernel>;
@group(0) @binding(2) var<storage, read> nodes: array<GpuNode>;
@group(0) @binding(3) var<storage, read> params: MomentParams;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
@group(0) @binding(5) var<storage, read_write> flags: array<u32>;
@compute @workgroup_size(64)
fn rtfp_moments(@builtin(local_invocation_id) lid3: vec3<u32>) {
    let tid = lid3.x;
var origin_u: vec3<f32> = vec3<f32>(params.ox, params.oy, params.oz);
var world: vec3<f32> = vec3<f32>(params.wx, params.wy, params.wz);
var part: array<ds, 10>;
for (var m: u32 = 0u; m < 10u; m = m + 1u) {
part[m] = ds_zero();
}
var overflow: bool = false;
var merged: array<f32, 128>;
for (var i: u32 = tid; i < params.ray_count; i = i + (64u)) {
var ray: GpuRay = rays[i];
if (!isfinite_f(ray.weight) || ray.weight == 0.0) {
continue;
}
var dir: vec3<f32> = vec3<f32>(ray.dx, ray.dy, ray.dz);
var len2: f32 = dot(dir, dir);
if (!(len2 > 0.0) || !isfinite_f(len2)) {
continue;
}
dir = dir * (inverseSqrt(len2));
var ray_overflow: bool = false;
var n: u32 = bvh_collect(origin_u, dir, params.body_scale, params.launch_offset, params.origin_abs, params.merge_abs, params.merge_rel, &merged, &ray_overflow);
if (ray_overflow) {
overflow = true;
continue;
}
var open_r: f32 = select(-1.0, 0.0, (n & 1u) != 0u);
for (var h: u32 = 0u; h < n; h = h + 1u) {
var t: f32 = merged[h];
if (open_r < 0.0) {
open_r = t;
continue;
}
var r0: f32 = open_r;
var r1: f32 = t;
open_r = -1.0;
var span: f32 = r1 - r0;
if (!(span > 0.0)) {
continue;
}
var ratio: f32 = span / params.finest;
var panels: u32 = 1u;
if (!isfinite_f(ratio) || ratio > f32(params.max_panels)) {
panels = params.max_panels;
} else {
panels = max(u32(ceil(ratio)), 1u);
}
var step: f32 = span / f32(panels);
for (var p: u32 = 0u; p < panels; p = p + 1u) {
var lo: f32 = r0 + step * f32(p);
for (var q: u32 = 0u; q < params.node_count; q = q + 1u) {
var radius: f32 = lo + 0.5 * step * (nodes[q].x + 1.0);
var gw: f32 = 0.5 * step * nodes[q].w;
var y: vec3<f32> = world + dir * radius;
var rho: f32 = density_at(y, params.bulk, params.kernel_count);
var mass: f32 = ray.weight * rho * radius * radius * gw;
var dm: ds = ds_from_float(mass);
part[0] = ds_add(part[0], dm);
part[1] = ds_add(part[1], ds_mul(dm, ds_from_float(y.x)));
part[2] = ds_add(part[2], ds_mul(dm, ds_from_float(y.y)));
part[3] = ds_add(part[3], ds_mul(dm, ds_from_float(y.z)));
part[4] = ds_add(part[4], ds_mul(dm, ds_from_float(y.x * y.x)));
part[5] = ds_add(part[5], ds_mul(dm, ds_from_float(y.y * y.y)));
part[6] = ds_add(part[6], ds_mul(dm, ds_from_float(y.z * y.z)));
part[7] = ds_add(part[7], ds_mul(dm, ds_from_float(y.x * y.y)));
part[8] = ds_add(part[8], ds_mul(dm, ds_from_float(y.x * y.z)));
part[9] = ds_add(part[9], ds_mul(dm, ds_from_float(y.y * y.z)));
}
}
}
}
if (overflow) {
for (var m: u32 = 0u; m < 10u; m = m + 1u) {
part[m] = ds_zero();
}
}



for (var m: u32 = 0u; m < 10u; m = m + 1u) {
store_pair(m * 64u + tid, part[m]);
}
ov[tid] = select(0u, 1u, overflow);
reduce10(tid);
if (tid == 0u) {
for (var m: u32 = 0u; m < 10u; m = m + 1u) {
var value: ds = load_pair(m * 64u);
out[2u * m] = value.hi;
out[2u * m + 1u] = value.lo;
}
flags[0] = ov[0];
}
}

// END ENTRY rtfp_moments
