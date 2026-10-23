// #include "gpu/shader/ds.wesl"
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
    // 1/(avg_count + 1) as double-single, advanced on the device by `pdhg_tick`.
    inv_hi: f32,
    inv_lo: f32,
}

fn mix_avg(mean: ds, sample: ds, s: Step) -> ds {
    return ds_addp(mean, ds_mul(ds(s.inv_hi, s.inv_lo), ds_subp(sample, mean)));
}

fn step_tau(s: Step) -> ds {
    return ds(s.tau_hi, s.tau_lo);
}

fn step_sigma(s: Step) -> ds {
    return ds(s.sigma_hi, s.sigma_lo);
}

// ENTRY pdhg_primal
// x ← (x − τ(Aᵀy + q)) / (1 + τ P), x̄ ← 2x − x_old; Aᵀy by column (CSC).
// `cidx` holds the column pointers (nx + 1) followed by the row indices;
// `colc` holds q, P and the diagonal τ of each column.
@group(0) @binding(0)
var<storage, read> cidx: array<u32>;
@group(0) @binding(1)
var<storage, read> aval: array<vec2<f32>>;
@group(0) @binding(2)
var<storage, read> y: array<vec2<f32>>;
@group(0) @binding(3)
var<storage, read_write> x: array<vec2<f32>>;
@group(0) @binding(4)
var<storage, read> colc: array<vec2<f32>>;
@group(0) @binding(5)
var<storage, read_write> xbar: array<vec2<f32>>;
@group(0) @binding(6)
var<storage, read_write> xavg: array<vec2<f32>>;
@group(0) @binding(7)
var<storage, read> step: Step;
@compute @workgroup_size(64)
fn pdhg_primal(@builtin(global_invocation_id) gid3: vec3<u32>) {
    let tid = gid3.x;
    if tid >= step.nx {
        return;
    }
    let rows = step.nx + 1u;
    var s: dsacc = acc_zero();
    let end = cidx[tid + 1u];
    for(var k: u32 = cidx[tid]; k < end; k = k + 1u) {
        let yr = y[cidx[rows + k]];
        s = acc_fma(s, ds(aval[k].x, aval[k].y), ds(yr.x, yr.y));
    }
    let c = 3u * tid;
    let tau = ds_mul(ds(colc[c + 2u].x, colc[c + 2u].y), step_tau(step));
    let x_old = ds(x[tid].x, x[tid].y);
    let grad = ds_addp(acc_done(s), ds(colc[c].x, colc[c].y));
    let num = ds_subp(x_old, ds_mul(tau, grad));
    let den = ds_addp(ds_f(1.0), ds_mul(tau, ds(colc[c + 1u].x, colc[c + 1u].y)));
    let x_new = ds_div(num, den);
    x[tid] = vec2<f32>(x_new.hi, x_new.lo);
    let xb = ds_subp(ds_scale(x_new, 2.0), x_old);
    xbar[tid] = vec2<f32>(xb.hi, xb.lo);
    let mixed = mix_avg(ds(xavg[tid].x, xavg[tid].y), x_new, step);
    xavg[tid] = vec2<f32>(mixed.hi, mixed.lo);
}

// END ENTRY pdhg_primal
// ENTRY pdhg_ax
@group(0) @binding(0)
var<storage, read> row_ptr: array<u32>;
@group(0) @binding(1)
var<storage, read> col_idx: array<u32>;
@group(0) @binding(2)
var<storage, read> aval: array<vec2<f32>>;
@group(0) @binding(3)
var<storage, read> xbar: array<vec2<f32>>;
@group(0) @binding(4)
var<storage, read_write> ax: array<vec2<f32>>;
@group(0) @binding(5)
var<storage, read> step: Step;
@compute @workgroup_size(64)
fn pdhg_ax(@builtin(global_invocation_id) gid3: vec3<u32>) {
    let tid = gid3.x;
    if tid >= step.m {
        return;
    }
    var begin: u32 = row_ptr[tid];
    var end: u32 = row_ptr[tid + 1u];
    if end - begin > 32u {
        return;
    }
    var s: dsacc = acc_zero();
    for(var k: u32 = begin; k < end; k = k + 1u) {
        s = acc_fma(s, ds(aval[k].x, aval[k].y), ds(xbar[col_idx[k]].x, xbar[col_idx[k]].y));
    }
    ax[tid] = vec2<f32>((acc_done(s)).hi, (acc_done(s)).lo);
}

// END ENTRY pdhg_ax
// ENTRY pdhg_ax_long
// Rows longer than 32 entries (the mass row): one workgroup per chunk of at most
// LONG_CHUNK entries writes a partial; `pdhg_ax_long_sum` adds a row's chunks.
// chunks[3c..3c+3] = (row, begin, end).
var<workgroup> red_hi: array<f32, 256>;
var<workgroup> red_lo: array<f32, 256>;
@group(0) @binding(0)
var<storage, read> chunks: array<u32>;
@group(0) @binding(1)
var<storage, read> col_idx: array<u32>;
@group(0) @binding(2)
var<storage, read> aval: array<vec2<f32>>;
@group(0) @binding(3)
var<storage, read> xbar: array<vec2<f32>>;
@group(0) @binding(4)
var<storage, read_write> lpart: array<vec2<f32>>;
@compute @workgroup_size(256)
fn pdhg_ax_long(
    @builtin(local_invocation_id) lid3: vec3<u32>,
    @builtin(workgroup_id) wgid3: vec3<u32>,
) {
    let c = wgid3.x;
    let lane = lid3.x;
    let end = chunks[3u * c + 2u];
    var s: dsacc = acc_zero();
    for(var k = chunks[3u * c + 1u] + lane; k < end; k = k + 256u) {
        let xv = xbar[col_idx[k]];
        s = acc_fma(s, ds(aval[k].x, aval[k].y), ds(xv.x, xv.y));
    }
    red_hi[lane] = s.hi;
    red_lo[lane] = s.lo;
    workgroupBarrier();
    for(var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if lane < stride {
            let part = acc_add(
                    dsacc(red_hi[lane], red_lo[lane]),
                    ds(red_hi[lane + stride], red_lo[lane + stride]),
                );
            red_hi[lane] = part.hi;
            red_lo[lane] = part.lo;
        }
        workgroupBarrier();
    }
    if lane == 0u {
        lpart[c] = vec2<f32>(red_hi[0], red_lo[0]);
    }
}
// END ENTRY pdhg_ax_long
// ENTRY pdhg_ax_long_sum
// longs[3r..3r+3] = (row, first chunk, chunk count).
@group(0) @binding(0)
var<storage, read> longs: array<u32>;
@group(0) @binding(1)
var<storage, read> lsum: array<vec2<f32>>;
@group(0) @binding(2)
var<storage, read_write> axl: array<vec2<f32>>;
@group(0) @binding(3)
var<storage, read> lstep: Step;
@compute @workgroup_size(64)
fn pdhg_ax_long_sum(@builtin(global_invocation_id) gid3: vec3<u32>) {
    let r = gid3.x;
    if r >= lstep.n_long {
        return;
    }
    let first = longs[3u * r + 1u];
    var s: dsacc = acc_zero();
    for(var c = first; c < first + longs[3u * r + 2u]; c = c + 1u) {
        s = acc_add(s, ds(lsum[c].x, lsum[c].y));
    }
    let v = acc_done(s);
    axl[longs[3u * r]] = vec2<f32>(v.hi, v.lo);
}
// END ENTRY pdhg_ax_long_sum
// ENTRY pdhg_dual
@group(0) @binding(0)
var<storage, read_write> y: array<vec2<f32>>;
@group(0) @binding(1)
var<storage, read> ax: array<vec2<f32>>;
@group(0) @binding(2)
var<storage, read> b: array<vec2<f32>>;
@group(0) @binding(3)
var<storage, read_write> yavg: array<vec2<f32>>;
@group(0) @binding(4)
var<storage, read> sigma_diag: array<vec2<f32>>;
@group(0) @binding(5)
var<storage, read> step: Step;
@compute @workgroup_size(64)
fn pdhg_dual(@builtin(global_invocation_id) gid3: vec3<u32>) {
    let tid = gid3.x;
    var n_lin: u32 = step.n_eq + step.n_nn;
    var n_jobs: u32 = n_lin + step.n_soc;
    if tid >= n_jobs {
        return;
    }
    if tid < n_lin {
        var sigma: ds = ds_mul(ds(sigma_diag[tid].x, sigma_diag[tid].y), step_sigma(step));
        var pred: ds = ds_addp(
                ds(y[tid].x, y[tid].y),
                ds_mul(sigma, ds_subp(ds(ax[tid].x, ax[tid].y), ds(b[tid].x, b[tid].y))),
            );
        if tid >= step.n_eq {
            pred = ds_max0(pred);
        }
        y[tid] = vec2<f32>((pred).hi, (pred).lo);
        {
            var mixed: ds = mix_avg(ds(yavg[tid].x, yavg[tid].y), pred, step);
            yavg[tid] = vec2<f32>(mixed.hi, mixed.lo);
        }
        return;
    }
    var row: u32 = n_lin + (tid - n_lin) * 4u;
    var sigma: ds = ds_mul(ds(sigma_diag[row].x, sigma_diag[row].y), step_sigma(step));
    var t0: ds = ds_addp(
            ds(y[row].x, y[row].y),
            ds_mul(sigma, ds_subp(ds(ax[row].x, ax[row].y), ds(b[row].x, b[row].y))),
        );
    var t1: ds = ds_addp(
            ds(y[row + 1u].x, y[row + 1u].y),
            ds_mul(
                sigma,
                ds_subp(ds(ax[row + 1u].x, ax[row + 1u].y), ds(b[row + 1u].x, b[row + 1u].y)),
            ),
        );
    var t2: ds = ds_addp(
            ds(y[row + 2u].x, y[row + 2u].y),
            ds_mul(
                sigma,
                ds_subp(ds(ax[row + 2u].x, ax[row + 2u].y), ds(b[row + 2u].x, b[row + 2u].y)),
            ),
        );
    var t3: ds = ds_addp(
            ds(y[row + 3u].x, y[row + 3u].y),
            ds_mul(
                sigma,
                ds_subp(ds(ax[row + 3u].x, ax[row + 3u].y), ds(b[row + 3u].x, b[row + 3u].y)),
            ),
        );
    var nrm: ds = ds_sqrt(ds_addp(ds_addp(ds_sqr(t1), ds_sqr(t2)), ds_sqr(t3)));
    if !ds_gt(nrm, t0) {
        {
            y[row] = vec2<f32>((t0).hi, (t0).lo);
            y[row + 1u] = vec2<f32>((t1).hi, (t1).lo);
            y[row + 2u] = vec2<f32>((t2).hi, (t2).lo);
            y[row + 3u] = vec2<f32>((t3).hi, (t3).lo);
        }
    } else if !ds_gt(nrm, ds_neg(t0)) {
        var z: ds = ds_zero();
        {
            y[row] = vec2<f32>((z).hi, (z).lo);
            y[row + 1u] = vec2<f32>((z).hi, (z).lo);
            y[row + 2u] = vec2<f32>((z).hi, (z).lo);
            y[row + 3u] = vec2<f32>((z).hi, (z).lo);
        }
    } else {
        var a: ds = ds_div(ds_addp(nrm, t0), ds_scale(nrm, 2.0));
        {
            y[row] = vec2<f32>((ds_mul(a, nrm)).hi, (ds_mul(a, nrm)).lo);
            y[row + 1u] = vec2<f32>((ds_mul(a, t1)).hi, (ds_mul(a, t1)).lo);
            y[row + 2u] = vec2<f32>((ds_mul(a, t2)).hi, (ds_mul(a, t2)).lo);
            y[row + 3u] = vec2<f32>((ds_mul(a, t3)).hi, (ds_mul(a, t3)).lo);
        }
    }
    {
        var mixed: ds = mix_avg(ds(yavg[row].x, yavg[row].y), ds(y[row].x, y[row].y), step);
        yavg[row] = vec2<f32>(mixed.hi, mixed.lo);
    }
    {
        var mixed: ds = mix_avg(
                ds(yavg[row + 1u].x, yavg[row + 1u].y),
                ds(y[row + 1u].x, y[row + 1u].y),
                step,
            );
        yavg[row + 1u] = vec2<f32>(mixed.hi, mixed.lo);
    }
    {
        var mixed: ds = mix_avg(
                ds(yavg[row + 2u].x, yavg[row + 2u].y),
                ds(y[row + 2u].x, y[row + 2u].y),
                step,
            );
        yavg[row + 2u] = vec2<f32>(mixed.hi, mixed.lo);
    }
    {
        var mixed: ds = mix_avg(
                ds(yavg[row + 3u].x, yavg[row + 3u].y),
                ds(y[row + 3u].x, y[row + 3u].y),
                step,
            );
        yavg[row + 3u] = vec2<f32>(mixed.hi, mixed.lo);
    }
}

// END ENTRY pdhg_dual
// ENTRY pdhg_trust_pred
// Trailing second-order cone (t, w) of length n_trust, one thread per row: the
// dual prediction y + σ(Ax − b), and per-workgroup partial sums of |w|²; the head t
// is also kept after the partials, so the projection never reads a row another
// workgroup is rewriting.
var<workgroup> tred_hi: array<f32, 256>;
var<workgroup> tred_lo: array<f32, 256>;
@group(0) @binding(0)
var<storage, read_write> y: array<vec2<f32>>;
@group(0) @binding(1)
var<storage, read> ax: array<vec2<f32>>;
@group(0) @binding(2)
var<storage, read> b: array<vec2<f32>>;
@group(0) @binding(3)
var<storage, read> sigma_diag: array<vec2<f32>>;
@group(0) @binding(4)
var<storage, read_write> tpart: array<vec2<f32>>;
@group(0) @binding(5)
var<storage, read> step: Step;
@compute @workgroup_size(256)
fn pdhg_trust_pred(
    @builtin(global_invocation_id) gid3: vec3<u32>,
    @builtin(local_invocation_id) lid3: vec3<u32>,
    @builtin(workgroup_id) wgid3: vec3<u32>,
) {
    let i = gid3.x;
    let lane = lid3.x;
    let row0 = step.n_eq + step.n_nn + step.n_soc * 4u;
    var sq: dsacc = acc_zero();
    if i < step.n_trust {
        let sigma = ds_mul(ds(sigma_diag[row0].x, sigma_diag[row0].y), step_sigma(step));
        let row = row0 + i;
        let p = ds_addp(
                ds(y[row].x, y[row].y),
                ds_mul(sigma, ds_subp(ds(ax[row].x, ax[row].y), ds(b[row].x, b[row].y))),
            );
        y[row] = vec2<f32>(p.hi, p.lo);
        if i == 0u {
            tpart[(step.n_trust + 255u) / 256u] = vec2<f32>(p.hi, p.lo);
        } else {
            sq = acc_fma(sq, p, p);
        }
    }
    tred_hi[lane] = sq.hi;
    tred_lo[lane] = sq.lo;
    workgroupBarrier();
    for(var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if lane < stride {
            let part = acc_add(
                    dsacc(tred_hi[lane], tred_lo[lane]),
                    ds(tred_hi[lane + stride], tred_lo[lane + stride]),
                );
            tred_hi[lane] = part.hi;
            tred_lo[lane] = part.lo;
        }
        workgroupBarrier();
    }
    if lane == 0u {
        tpart[wgid3.x] = vec2<f32>(tred_hi[0], tred_lo[0]);
    }
}
// END ENTRY pdhg_trust_pred
// ENTRY pdhg_trust_proj
// Every workgroup reduces the partials to |w| and projects its rows onto the cone
// |w| ≤ t, then mixes them into the ergodic average.
var<workgroup> pred_hi: array<f32, 256>;
var<workgroup> pred_lo: array<f32, 256>;
@group(0) @binding(0)
var<storage, read_write> y: array<vec2<f32>>;
@group(0) @binding(1)
var<storage, read_write> yavg: array<vec2<f32>>;
@group(0) @binding(2)
var<storage, read> tpart: array<vec2<f32>>;
@group(0) @binding(3)
var<storage, read> step: Step;
@compute @workgroup_size(256)
fn pdhg_trust_proj(
    @builtin(global_invocation_id) gid3: vec3<u32>,
    @builtin(local_invocation_id) lid3: vec3<u32>,
) {
    let i = gid3.x;
    let lane = lid3.x;
    let parts = (step.n_trust + 255u) / 256u;
    var sq: dsacc = acc_zero();
    for(var k = lane; k < parts; k = k + 256u) {
        sq = acc_add(sq, ds(tpart[k].x, tpart[k].y));
    }
    pred_hi[lane] = sq.hi;
    pred_lo[lane] = sq.lo;
    workgroupBarrier();
    for(var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if lane < stride {
            let part = acc_add(
                    dsacc(pred_hi[lane], pred_lo[lane]),
                    ds(pred_hi[lane + stride], pred_lo[lane + stride]),
                );
            pred_hi[lane] = part.hi;
            pred_lo[lane] = part.lo;
        }
        workgroupBarrier();
    }
    if i >= step.n_trust {
        return;
    }
    let nrm = ds_sqrt(acc_done(dsacc(pred_hi[0], pred_lo[0])));
    let t = ds(tpart[parts].x, tpart[parts].y);
    let row = step.n_eq + step.n_nn + step.n_soc * 4u + i;
    var v = ds(y[row].x, y[row].y);
    if ds_gt(nrm, t) {
        if !ds_gt(nrm, ds_neg(t)) {
            v = ds_zero();
        } else {
            let a = ds_div(ds_addp(nrm, t), ds_scale(nrm, 2.0));
            if i == 0u {
                v = ds_mul(a, nrm);
            } else {
                v = ds_mul(a, v);
            }
        }
        y[row] = vec2<f32>(v.hi, v.lo);
    }
    let mixed = mix_avg(ds(yavg[row].x, yavg[row].y), v, step);
    yavg[row] = vec2<f32>(mixed.hi, mixed.lo);
}
// END ENTRY pdhg_trust_proj

// ENTRY pdhg_tick
// Advance the ergodic count after one iteration, keeping every pass of a batch on
// the device.
@group(0) @binding(0)
var<storage, read_write> tick: Step;
@compute @workgroup_size(1)
fn pdhg_tick() {
    let count = tick.avg_count + 1u;
    tick.avg_count = count;
    let inv = ds_div(ds_f(1.0), ds_f(f32(count + 1u)));
    tick.inv_hi = inv.hi;
    tick.inv_lo = inv.lo;
}
// END ENTRY pdhg_tick

// ENTRY pdhg_batch
// A whole batch in one workgroup, for a problem whose columns, rows, cones and long-row
// chunks fit the lanes and arrays below. Each phase is the arithmetic of pdhg_primal …
// pdhg_tick in the same order, so the iterates are the same bits; x̄, Ax and the
// partial sums stay in workgroup memory, and a barrier stands where a kernel ended.
// One dispatch replaces up to eight per iteration.
// `hdr`: iterations, nx, m, n_eq, n_nn, n_soc, n_trust, n_long, n_chunks; the offsets
// in `hdr` of the column pointers and row indices, row_ptr, col_idx, chunks and longs;
// the offsets in `cst` of the CSC values, (q, P, τ) per column, CSR values, b and σ.
@group(0) @binding(0)
var<storage, read> hdr: array<u32>;
@group(0) @binding(1)
var<storage, read> cst: array<vec2<f32>>;
@group(0) @binding(2)
var<storage, read_write> x: array<vec2<f32>>;
@group(0) @binding(3)
var<storage, read_write> xavg: array<vec2<f32>>;
@group(0) @binding(4)
var<storage, read_write> y: array<vec2<f32>>;
@group(0) @binding(5)
var<storage, read_write> yavg: array<vec2<f32>>;
@group(0) @binding(6)
var<storage, read_write> step: Step;
var<workgroup> bxbar: array<vec2<f32>, 256>;
var<workgroup> bax: array<vec2<f32>, 256>;
var<workgroup> bred_hi: array<f32, 256>;
var<workgroup> bred_lo: array<f32, 256>;
var<workgroup> blpart: array<vec2<f32>, 16>;
// Trust-cone partial of |w|² and the head t.
var<workgroup> btpart: array<vec2<f32>, 2>;
var<workgroup> bstep: Step;

fn cds(i: u32) -> ds {
    return ds(cst[i].x, cst[i].y);
}

// The tree of pdhg_ax_long and the trust kernels over bred_hi/bred_lo.
fn batch_tree(lane: u32) {
    workgroupBarrier();
    for(var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if lane < stride {
            let part = acc_add(
                    dsacc(bred_hi[lane], bred_lo[lane]),
                    ds(bred_hi[lane + stride], bred_lo[lane + stride]),
                );
            bred_hi[lane] = part.hi;
            bred_lo[lane] = part.lo;
        }
        workgroupBarrier();
    }
}

// y + σ(Ax − b) for one row.
fn batch_pred(row: u32, sigma: ds, o_b: u32) -> ds {
    return ds_addp(
        ds(y[row].x, y[row].y),
        ds_mul(sigma, ds_subp(ds(bax[row].x, bax[row].y), cds(o_b + row))),
    );
}

@compute @workgroup_size(256)
fn pdhg_batch(@builtin(local_invocation_id) lid3: vec3<u32>) {
    let lane = lid3.x;
    let iters = hdr[0];
    let nx = hdr[1];
    let m = hdr[2];
    let n_eq = hdr[3];
    let n_nn = hdr[4];
    let n_soc = hdr[5];
    let n_trust = hdr[6];
    let n_long = hdr[7];
    let n_chunks = hdr[8];
    let o_cidx = hdr[9];
    let o_rp = hdr[10];
    let o_col = hdr[11];
    let o_chunks = hdr[12];
    let o_longs = hdr[13];
    let o_cval = hdr[14];
    let o_colc = hdr[15];
    let o_rval = hdr[16];
    let o_b = hdr[17];
    let o_sig = hdr[18];
    let n_lin = n_eq + n_nn;
    let n_jobs = n_lin + n_soc;
    if lane == 0u {
        bstep = step;
    }
    workgroupBarrier();
    for(var it = 0u; it < iters; it = it + 1u) {
        let s = bstep;
        // pdhg_primal
        if lane < nx {
            let rows = nx + 1u;
            var acc: dsacc = acc_zero();
            let end = hdr[o_cidx + lane + 1u];
            for(var k: u32 = hdr[o_cidx + lane]; k < end; k = k + 1u) {
                let yr = y[hdr[o_cidx + rows + k]];
                acc = acc_fma(acc, cds(o_cval + k), ds(yr.x, yr.y));
            }
            let c = o_colc + 3u * lane;
            let tau = ds_mul(cds(c + 2u), step_tau(s));
            let x_old = ds(x[lane].x, x[lane].y);
            let grad = ds_addp(acc_done(acc), cds(c));
            let num = ds_subp(x_old, ds_mul(tau, grad));
            let den = ds_addp(ds_f(1.0), ds_mul(tau, cds(c + 1u)));
            let x_new = ds_div(num, den);
            x[lane] = vec2<f32>(x_new.hi, x_new.lo);
            let xb = ds_subp(ds_scale(x_new, 2.0), x_old);
            bxbar[lane] = vec2<f32>(xb.hi, xb.lo);
            let mixed = mix_avg(ds(xavg[lane].x, xavg[lane].y), x_new, s);
            xavg[lane] = vec2<f32>(mixed.hi, mixed.lo);
        }
        workgroupBarrier();
        // pdhg_ax
        if lane < m {
            let begin = hdr[o_rp + lane];
            let end = hdr[o_rp + lane + 1u];
            if end - begin <= 32u {
                var acc: dsacc = acc_zero();
                for(var k: u32 = begin; k < end; k = k + 1u) {
                    let xv = bxbar[hdr[o_col + k]];
                    acc = acc_fma(acc, cds(o_rval + k), ds(xv.x, xv.y));
                }
                bax[lane] = vec2<f32>((acc_done(acc)).hi, (acc_done(acc)).lo);
            }
        }
        // pdhg_ax_long, pdhg_ax_long_sum
        if n_long > 0u {
            for(var c = 0u; c < n_chunks; c = c + 1u) {
                let ch = o_chunks + 3u * c;
                let end = hdr[ch + 2u];
                var acc: dsacc = acc_zero();
                for(var k = hdr[ch + 1u] + lane; k < end; k = k + 256u) {
                    let xv = bxbar[hdr[o_col + k]];
                    acc = acc_fma(acc, cds(o_rval + k), ds(xv.x, xv.y));
                }
                workgroupBarrier();
                bred_hi[lane] = acc.hi;
                bred_lo[lane] = acc.lo;
                batch_tree(lane);
                if lane == 0u {
                    blpart[c] = vec2<f32>(bred_hi[0], bred_lo[0]);
                }
            }
            workgroupBarrier();
            if lane < n_long {
                let r = o_longs + 3u * lane;
                let first = hdr[r + 1u];
                var acc: dsacc = acc_zero();
                for(var c = first; c < first + hdr[r + 2u]; c = c + 1u) {
                    acc = acc_add(acc, ds(blpart[c].x, blpart[c].y));
                }
                let v = acc_done(acc);
                bax[hdr[r]] = vec2<f32>(v.hi, v.lo);
            }
        }
        workgroupBarrier();
        // pdhg_dual
        if lane < n_lin {
            let sigma = ds_mul(cds(o_sig + lane), step_sigma(s));
            var pred = batch_pred(lane, sigma, o_b);
            if lane >= n_eq {
                pred = ds_max0(pred);
            }
            y[lane] = vec2<f32>((pred).hi, (pred).lo);
            let mixed = mix_avg(ds(yavg[lane].x, yavg[lane].y), pred, s);
            yavg[lane] = vec2<f32>(mixed.hi, mixed.lo);
        } else if lane < n_jobs {
            let row = n_lin + (lane - n_lin) * 4u;
            let sigma = ds_mul(cds(o_sig + row), step_sigma(s));
            let t0 = batch_pred(row, sigma, o_b);
            let t1 = batch_pred(row + 1u, sigma, o_b);
            let t2 = batch_pred(row + 2u, sigma, o_b);
            let t3 = batch_pred(row + 3u, sigma, o_b);
            let nrm = ds_sqrt(ds_addp(ds_addp(ds_sqr(t1), ds_sqr(t2)), ds_sqr(t3)));
            if !ds_gt(nrm, t0) {
                y[row] = vec2<f32>((t0).hi, (t0).lo);
                y[row + 1u] = vec2<f32>((t1).hi, (t1).lo);
                y[row + 2u] = vec2<f32>((t2).hi, (t2).lo);
                y[row + 3u] = vec2<f32>((t3).hi, (t3).lo);
            } else if !ds_gt(nrm, ds_neg(t0)) {
                let z = ds_zero();
                y[row] = vec2<f32>((z).hi, (z).lo);
                y[row + 1u] = vec2<f32>((z).hi, (z).lo);
                y[row + 2u] = vec2<f32>((z).hi, (z).lo);
                y[row + 3u] = vec2<f32>((z).hi, (z).lo);
            } else {
                let a = ds_div(ds_addp(nrm, t0), ds_scale(nrm, 2.0));
                y[row] = vec2<f32>((ds_mul(a, nrm)).hi, (ds_mul(a, nrm)).lo);
                y[row + 1u] = vec2<f32>((ds_mul(a, t1)).hi, (ds_mul(a, t1)).lo);
                y[row + 2u] = vec2<f32>((ds_mul(a, t2)).hi, (ds_mul(a, t2)).lo);
                y[row + 3u] = vec2<f32>((ds_mul(a, t3)).hi, (ds_mul(a, t3)).lo);
            }
            for(var r = row; r < row + 4u; r = r + 1u) {
                let mixed = mix_avg(ds(yavg[r].x, yavg[r].y), ds(y[r].x, y[r].y), s);
                yavg[r] = vec2<f32>(mixed.hi, mixed.lo);
            }
        }
        // pdhg_trust_pred, pdhg_trust_proj (one workgroup: at most 256 rows)
        if n_trust > 0u {
            let row0 = n_lin + n_soc * 4u;
            var sq: dsacc = acc_zero();
            if lane < n_trust {
                let sigma = ds_mul(cds(o_sig + row0), step_sigma(s));
                let row = row0 + lane;
                let p = batch_pred(row, sigma, o_b);
                y[row] = vec2<f32>(p.hi, p.lo);
                if lane == 0u {
                    btpart[1] = vec2<f32>(p.hi, p.lo);
                } else {
                    sq = acc_fma(sq, p, p);
                }
            }
            workgroupBarrier();
            bred_hi[lane] = sq.hi;
            bred_lo[lane] = sq.lo;
            batch_tree(lane);
            if lane == 0u {
                btpart[0] = vec2<f32>(bred_hi[0], bred_lo[0]);
            }
            workgroupBarrier();
            var sum: dsacc = acc_zero();
            if lane == 0u {
                sum = acc_add(sum, ds(btpart[0].x, btpart[0].y));
            }
            bred_hi[lane] = sum.hi;
            bred_lo[lane] = sum.lo;
            batch_tree(lane);
            if lane < n_trust {
                let nrm = ds_sqrt(acc_done(dsacc(bred_hi[0], bred_lo[0])));
                let t = ds(btpart[1].x, btpart[1].y);
                let row = row0 + lane;
                var v = ds(y[row].x, y[row].y);
                if ds_gt(nrm, t) {
                    if !ds_gt(nrm, ds_neg(t)) {
                        v = ds_zero();
                    } else {
                        let a = ds_div(ds_addp(nrm, t), ds_scale(nrm, 2.0));
                        if lane == 0u {
                            v = ds_mul(a, nrm);
                        } else {
                            v = ds_mul(a, v);
                        }
                    }
                    y[row] = vec2<f32>(v.hi, v.lo);
                }
                let mixed = mix_avg(ds(yavg[row].x, yavg[row].y), v, s);
                yavg[row] = vec2<f32>(mixed.hi, mixed.lo);
            }
        }
        // pdhg_tick; the next primal reads y rows other lanes wrote.
        workgroupBarrier();
        if lane == 0u {
            let count = bstep.avg_count + 1u;
            bstep.avg_count = count;
            let inv = ds_div(ds_f(1.0), ds_f(f32(count + 1u)));
            bstep.inv_hi = inv.hi;
            bstep.inv_lo = inv.lo;
        }
        storageBarrier();
        workgroupBarrier();
    }
    if lane == 0u {
        step = bstep;
    }
}
// END ENTRY pdhg_batch
