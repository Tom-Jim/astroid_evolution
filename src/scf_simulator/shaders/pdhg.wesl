// SHARED
struct Step {
    nx: u32,
    m: u32,
    n_eq: u32,
    n_nn: u32,
    n_soc: u32,
    n_trust: u32,
    avg_count: u32,
    tau_hi: f32,
    tau_lo: f32,
    sigma_hi: f32,
    sigma_lo: f32,
    n_long: u32,
}



fn mix_avg(mean: ds, sample: ds, count: u32) -> ds {
var inv: ds = ds_div(ds_f(1.0), ds_f(f32(count + 1u)));
return ds_addp(mean, ds_mul(inv, ds_subp(sample, mean)));
}

fn step_tau(s: Step) -> ds {
return ds(s.tau_hi, s.tau_lo);
}

fn step_sigma(s: Step) -> ds {
return ds(s.sigma_hi, s.sigma_lo);
}



// ENTRY pdhg_aty
@group(0) @binding(0) var<storage, read> col_ptr: array<u32>;
@group(0) @binding(1) var<storage, read> row_idx: array<u32>;
@group(0) @binding(2) var<storage, read> aval: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read> y: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read_write> aty: array<vec2<f32>>;
@group(0) @binding(5) var<storage, read> step: Step;
@compute @workgroup_size(64)
fn pdhg_aty(@builtin(global_invocation_id) gid3: vec3<u32>) {
    let tid = gid3.x;
if (tid >= step.nx) {
return;
}
var s: dsacc = acc_zero();
var end: u32 = col_ptr[tid + 1u];
for (var k: u32 = col_ptr[tid]; k < end; k = k + 1u) {
s = acc_fma(s, ds(aval[k].x, aval[k].y), ds(y[row_idx[k]].x, y[row_idx[k]].y));
}
aty[tid] = vec2<f32>((acc_done(s)).hi, (acc_done(s)).lo);
}

// END ENTRY pdhg_aty
// ENTRY pdhg_primal
@group(0) @binding(0) var<storage, read_write> x: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> aty: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> q: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read> pdiag: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read_write> xbar: array<vec2<f32>>;
@group(0) @binding(5) var<storage, read_write> xavg: array<vec2<f32>>;
@group(0) @binding(6) var<storage, read> tau_diag: array<vec2<f32>>;
@group(0) @binding(7) var<storage, read> step: Step;
@compute @workgroup_size(64)
fn pdhg_primal(@builtin(global_invocation_id) gid3: vec3<u32>) {
    let tid = gid3.x;
if (tid >= step.nx) {
return;
}
var tau: ds = ds_mul(ds(tau_diag[tid].x, tau_diag[tid].y), step_tau(step));
var x_old: ds = ds(x[tid].x, x[tid].y);
var grad: ds = ds_addp(ds(aty[tid].x, aty[tid].y), ds(q[tid].x, q[tid].y));
var num: ds = ds_subp(x_old, ds_mul(tau, grad));
var den: ds = ds_addp(ds_f(1.0), ds_mul(tau, ds(pdiag[tid].x, pdiag[tid].y)));
var x_new: ds = ds_div(num, den);
x[tid] = vec2<f32>((x_new).hi, (x_new).lo);
xbar[tid] = vec2<f32>((ds_subp(ds_scale(x_new, 2.0), x_old)).hi, (ds_subp(ds_scale(x_new, 2.0), x_old)).lo);
{ var mixed: ds = mix_avg(ds(xavg[tid].x, xavg[tid].y), x_new, step.avg_count); xavg[tid] = vec2<f32>(mixed.hi, mixed.lo); };
}

// END ENTRY pdhg_primal
// ENTRY pdhg_ax
@group(0) @binding(0) var<storage, read> row_ptr: array<u32>;
@group(0) @binding(1) var<storage, read> col_idx: array<u32>;
@group(0) @binding(2) var<storage, read> aval: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read> xbar: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read_write> ax: array<vec2<f32>>;
@group(0) @binding(5) var<storage, read> step: Step;
@compute @workgroup_size(64)
fn pdhg_ax(@builtin(global_invocation_id) gid3: vec3<u32>) {
    let tid = gid3.x;
if (tid >= step.m) {
return;
}
var begin: u32 = row_ptr[tid];
var end: u32 = row_ptr[tid + 1u];
if (end - begin > 32u) {
return;
}
var s: dsacc = acc_zero();
for (var k: u32 = begin; k < end; k = k + 1u) {
s = acc_fma(s, ds(aval[k].x, aval[k].y), ds(xbar[col_idx[k]].x, xbar[col_idx[k]].y));
}
ax[tid] = vec2<f32>((acc_done(s)).hi, (acc_done(s)).lo);
}

// END ENTRY pdhg_ax
// ENTRY pdhg_ax_long
var<workgroup> red_hi: array<f32, 64>;
var<workgroup> red_lo: array<f32, 64>;
@group(0) @binding(0) var<storage, read> row_ptr: array<u32>;
@group(0) @binding(1) var<storage, read> col_idx: array<u32>;
@group(0) @binding(2) var<storage, read> aval: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read> xbar: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read_write> ax: array<vec2<f32>>;
@group(0) @binding(5) var<storage, read> long_rows: array<u32>;
@group(0) @binding(6) var<storage, read> step: Step;
@compute @workgroup_size(64)
fn pdhg_ax_long(@builtin(local_invocation_id) lid3: vec3<u32>, @builtin(workgroup_id) wgid3: vec3<u32>) {
    let tg = wgid3.x;
    let lane = lid3.x;
    let lanes = 64u;


var row: u32 = long_rows[tg];
var begin: u32 = row_ptr[row];
var end: u32 = row_ptr[row + 1u];
var s: dsacc = acc_zero();
for (var k: u32 = begin + lane; k < end; k = k + (lanes)) {
s = acc_fma(s, ds(aval[k].x, aval[k].y), ds(xbar[col_idx[k]].x, xbar[col_idx[k]].y));
}
red_hi[lane] = s.hi;
red_lo[lane] = s.lo;
workgroupBarrier();
for (var stride: u32 = lanes >> 1u; stride > 0u; stride = stride >> (1u)) {
if (lane < stride) {
var part: dsacc = dsacc(red_hi[lane], red_lo[lane]);
part = acc_add(part, ds(red_hi[lane + stride], red_lo[lane + stride]));
red_hi[lane] = part.hi;
red_lo[lane] = part.lo;
}
workgroupBarrier();
}
if (lane == 0u) {
ax[row] = vec2<f32>((acc_done(dsacc(red_hi[0], red_lo[0]))).hi, (acc_done(dsacc(red_hi[0], red_lo[0]))).lo);
}
}

// END ENTRY pdhg_ax_long
// ENTRY pdhg_dual
@group(0) @binding(0) var<storage, read_write> y: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> ax: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> b: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> yavg: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read> sigma_diag: array<vec2<f32>>;
@group(0) @binding(5) var<storage, read> step: Step;
@compute @workgroup_size(64)
fn pdhg_dual(@builtin(global_invocation_id) gid3: vec3<u32>) {
    let tid = gid3.x;
var n_lin: u32 = step.n_eq + step.n_nn;
var n_jobs: u32 = n_lin + step.n_soc;
if (tid >= n_jobs) {
return;
}
if (tid < n_lin) {
var sigma: ds = ds_mul(ds(sigma_diag[tid].x, sigma_diag[tid].y), step_sigma(step));
var pred: ds = ds_addp(ds(y[tid].x, y[tid].y), ds_mul(sigma, ds_subp(ds(ax[tid].x, ax[tid].y), ds(b[tid].x, b[tid].y))));
if (tid >= step.n_eq) {
pred = ds_max0(pred);
}
y[tid] = vec2<f32>((pred).hi, (pred).lo);
{ var mixed: ds = mix_avg(ds(yavg[tid].x, yavg[tid].y), pred, step.avg_count); yavg[tid] = vec2<f32>(mixed.hi, mixed.lo); };
return;
}
var row: u32 = n_lin + (tid - n_lin) * 4u;
var sigma: ds = ds_mul(ds(sigma_diag[row].x, sigma_diag[row].y), step_sigma(step));
var t0: ds = ds_addp(ds(y[row].x, y[row].y), ds_mul(sigma, ds_subp(ds(ax[row].x, ax[row].y), ds(b[row].x, b[row].y))));
var t1: ds = ds_addp(ds(y[row + 1u].x, y[row + 1u].y), ds_mul(sigma, ds_subp(ds(ax[row + 1u].x, ax[row + 1u].y), ds(b[row + 1u].x, b[row + 1u].y))));
var t2: ds = ds_addp(ds(y[row + 2u].x, y[row + 2u].y), ds_mul(sigma, ds_subp(ds(ax[row + 2u].x, ax[row + 2u].y), ds(b[row + 2u].x, b[row + 2u].y))));
var t3: ds = ds_addp(ds(y[row + 3u].x, y[row + 3u].y), ds_mul(sigma, ds_subp(ds(ax[row + 3u].x, ax[row + 3u].y), ds(b[row + 3u].x, b[row + 3u].y))));
var nrm: ds = ds_sqrt(ds_addp(ds_addp(ds_sqr(t1), ds_sqr(t2)), ds_sqr(t3)));
if (!ds_gt(nrm, t0)) {
{ y[row] = vec2<f32>((t0).hi, (t0).lo); y[row + 1u] = vec2<f32>((t1).hi, (t1).lo); y[row + 2u] = vec2<f32>((t2).hi, (t2).lo); y[row + 3u] = vec2<f32>((t3).hi, (t3).lo); };
} else if (!ds_gt(nrm, ds_neg(t0))) {
var z: ds = ds_zero();
{ y[row] = vec2<f32>((z).hi, (z).lo); y[row + 1u] = vec2<f32>((z).hi, (z).lo); y[row + 2u] = vec2<f32>((z).hi, (z).lo); y[row + 3u] = vec2<f32>((z).hi, (z).lo); };
} else {
var a: ds = ds_div(ds_addp(nrm, t0), ds_scale(nrm, 2.0));
{ y[row] = vec2<f32>((ds_mul(a, nrm)).hi, (ds_mul(a, nrm)).lo); y[row + 1u] = vec2<f32>((ds_mul(a, t1)).hi, (ds_mul(a, t1)).lo); y[row + 2u] = vec2<f32>((ds_mul(a, t2)).hi, (ds_mul(a, t2)).lo); y[row + 3u] = vec2<f32>((ds_mul(a, t3)).hi, (ds_mul(a, t3)).lo); };
}
{ var mixed: ds = mix_avg(ds(yavg[row].x, yavg[row].y), ds(y[row].x, y[row].y), step.avg_count); yavg[row] = vec2<f32>(mixed.hi, mixed.lo); };
{ var mixed: ds = mix_avg(ds(yavg[row + 1u].x, yavg[row + 1u].y), ds(y[row + 1u].x, y[row + 1u].y), step.avg_count); yavg[row + 1u] = vec2<f32>(mixed.hi, mixed.lo); };
{ var mixed: ds = mix_avg(ds(yavg[row + 2u].x, yavg[row + 2u].y), ds(y[row + 2u].x, y[row + 2u].y), step.avg_count); yavg[row + 2u] = vec2<f32>(mixed.hi, mixed.lo); };
{ var mixed: ds = mix_avg(ds(yavg[row + 3u].x, yavg[row + 3u].y), ds(y[row + 3u].x, y[row + 3u].y), step.avg_count); yavg[row + 3u] = vec2<f32>(mixed.hi, mixed.lo); };
}

// END ENTRY pdhg_dual
// ENTRY pdhg_trust
var<workgroup> red_hi: array<f32, 64>;
var<workgroup> red_lo: array<f32, 64>;
@group(0) @binding(0) var<storage, read_write> y: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> ax: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> b: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> yavg: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read> sigma_diag: array<vec2<f32>>;
@group(0) @binding(5) var<storage, read> step: Step;
@compute @workgroup_size(64)
fn pdhg_trust(@builtin(local_invocation_id) lid3: vec3<u32>) {
    let lane = lid3.x;
    let lanes = 64u;


var row0: u32 = step.n_eq + step.n_nn + step.n_soc * 4u;
var sigma: ds = ds_mul(ds(sigma_diag[row0].x, sigma_diag[row0].y), step_sigma(step));
var t: ds = ds_addp(ds(y[row0].x, y[row0].y), ds_mul(sigma, ds_subp(ds(ax[row0].x, ax[row0].y), ds(b[row0].x, b[row0].y))));
if (lane == 0u) {
y[row0] = vec2<f32>((t).hi, (t).lo);
}
var nrm2: dsacc = acc_zero();
for (var i: u32 = 1u + lane; i < step.n_trust; i = i + (lanes)) {
var row: u32 = row0 + i;
var p: ds = ds_addp(ds(y[row].x, y[row].y), ds_mul(sigma, ds_subp(ds(ax[row].x, ax[row].y), ds(b[row].x, b[row].y))));
y[row] = vec2<f32>((p).hi, (p).lo);
nrm2 = acc_fma(nrm2, p, p);
}
red_hi[lane] = nrm2.hi;
red_lo[lane] = nrm2.lo;
workgroupBarrier();
for (var stride: u32 = lanes >> 1u; stride > 0u; stride = stride >> (1u)) {
if (lane < stride) {
var part: dsacc = dsacc(red_hi[lane], red_lo[lane]);
part = acc_add(part, ds(red_hi[lane + stride], red_lo[lane + stride]));
red_hi[lane] = part.hi;
red_lo[lane] = part.lo;
}
workgroupBarrier();
}
var nrm: ds = ds_sqrt(acc_done(dsacc(red_hi[0], red_lo[0])));
if (ds_gt(nrm, t)) {
if (!ds_gt(nrm, ds_neg(t))) {
for (var i: u32 = lane; i < step.n_trust; i = i + (lanes)) {
y[row0 + i] = vec2<f32>((ds_zero()).hi, (ds_zero()).lo);
}
} else {
var a: ds = ds_div(ds_addp(nrm, t), ds_scale(nrm, 2.0));
if (lane == 0u) {
y[row0] = vec2<f32>((ds_mul(a, nrm)).hi, (ds_mul(a, nrm)).lo);
}
for (var i: u32 = 1u + lane; i < step.n_trust; i = i + (lanes)) {
y[row0 + i] = vec2<f32>((ds_mul(a, ds(y[row0 + i].x, y[row0 + i].y))).hi, (ds_mul(a, ds(y[row0 + i].x, y[row0 + i].y))).lo);
}
}
}
if (lane == 0u) {
{ var mixed: ds = mix_avg(ds(yavg[row0].x, yavg[row0].y), ds(y[row0].x, y[row0].y), step.avg_count); yavg[row0] = vec2<f32>(mixed.hi, mixed.lo); };
}
for (var i: u32 = 1u + lane; i < step.n_trust; i = i + (lanes)) {
{ var mixed: ds = mix_avg(ds(yavg[row0 + i].x, yavg[row0 + i].y), ds(y[row0 + i].x, y[row0 + i].y), step.avg_count); yavg[row0 + i] = vec2<f32>(mixed.hi, mixed.lo); };
}
}

// END ENTRY pdhg_trust
