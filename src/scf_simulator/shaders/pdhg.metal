// One Chambolle–Pock step for the diagonal-quadratic cone program assembled in
// `socp.rs`. Arithmetic is double-single. The saddle point is
//
//   min_z max_{y ∈ K*}  ½ zᵀ P z + qᵀ z + yᵀ (A z − b),
//
// with θ = 1 extrapolation. K* is free on equality rows, nonnegative on the
// linear inequalities, a second-order cone on each block of four rows, and,
// when n_trust > 0, one trailing second-order cone of that length.
//
//   z⁺ = (z − τ (q + Aᵀ y)) / (1 + τ P)
//   z̄ = 2 z⁺ − z
//   y⁺ = Π_{K*}(y + σ (A z̄ − b))
//
// τ and σ are the diagonal Pock–Chambolle steps, multiplied by the scalar
// factor in Step. Every row of one cone uses that cone's σ.

inline ds ld(device const float2* a, uint i) {
    float2 v = a[i];
    return ds{v.x, v.y};
}

inline void st(device float2* a, uint i, ds v) { a[i] = float2(v.hi, v.lo); }

struct Step {
    uint nx;
    uint m;
    uint n_eq;
    uint n_nn;
    uint n_soc;
    uint n_trust;
    uint avg_count;
    float tau_hi;
    float tau_lo;
    float sigma_hi;
    float sigma_lo;
    uint n_long;
};

inline ds step_tau(constant Step& s) { return ds{s.tau_hi, s.tau_lo}; }
inline ds step_sigma(constant Step& s) { return ds{s.sigma_hi, s.sigma_lo}; }

inline void accumulate(device float2* avg, uint i, ds sample, uint count) {
    ds mean = ld(avg, i);
    ds inv = ds_div(ds_f(1.0f), ds_f(float(count + 1u)));
    st(avg, i, ds_addp(mean, ds_mul(inv, ds_subp(sample, mean))));
}

kernel void pdhg_aty(
    device const uint* col_ptr [[buffer(0)]],
    device const uint* row_idx [[buffer(1)]],
    device const float2* aval [[buffer(2)]],
    device const float2* y [[buffer(3)]],
    device float2* aty [[buffer(4)]],
    constant Step& step [[buffer(5)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= step.nx) {
        return;
    }
    dsacc s = acc_zero();
    uint end = col_ptr[tid + 1u];
    for (uint k = col_ptr[tid]; k < end; ++k) {
        acc_fma(s, ld(aval, k), ld(y, row_idx[k]));
    }
    st(aty, tid, acc_done(s));
}

kernel void pdhg_primal(
    device float2* x [[buffer(0)]],
    device const float2* aty [[buffer(1)]],
    device const float2* q [[buffer(2)]],
    device const float2* pdiag [[buffer(3)]],
    device float2* xbar [[buffer(4)]],
    device float2* xavg [[buffer(5)]],
    device const float2* tau_diag [[buffer(6)]],
    constant Step& step [[buffer(7)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= step.nx) {
        return;
    }
    ds tau = ds_mul(ld(tau_diag, tid), step_tau(step));
    ds x_old = ld(x, tid);
    ds grad = ds_addp(ld(aty, tid), ld(q, tid));
    ds num = ds_subp(x_old, ds_mul(tau, grad));
    ds den = ds_addp(ds_f(1.0f), ds_mul(tau, ld(pdiag, tid)));
    ds x_new = ds_div(num, den);
    st(x, tid, x_new);
    st(xbar, tid, ds_subp(ds_scale(x_new, 2.0f), x_old));
    accumulate(xavg, tid, x_new, step.avg_count);
}

kernel void pdhg_ax(
    device const uint* row_ptr [[buffer(0)]],
    device const uint* col_idx [[buffer(1)]],
    device const float2* aval [[buffer(2)]],
    device const float2* xbar [[buffer(3)]],
    device float2* ax [[buffer(4)]],
    constant Step& step [[buffer(5)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= step.m) {
        return;
    }
    uint begin = row_ptr[tid];
    uint end = row_ptr[tid + 1u];
    // A row longer than one warp is the mass row. One thread walking it is
    // the whole kernel's runtime on a Ryugu-sized mobile set, so that row is
    // owned by pdhg_ax_long.
    if (end - begin > 32u) {
        return;
    }
    dsacc s = acc_zero();
    for (uint k = begin; k < end; ++k) {
        acc_fma(s, ld(aval, k), ld(xbar, col_idx[k]));
    }
    st(ax, tid, acc_done(s));
}

// One threadgroup per long row. The dot product is the same sum as pdhg_ax.
kernel void pdhg_ax_long(
    device const uint* row_ptr [[buffer(0)]],
    device const uint* col_idx [[buffer(1)]],
    device const float2* aval [[buffer(2)]],
    device const float2* xbar [[buffer(3)]],
    device float2* ax [[buffer(4)]],
    device const uint* long_rows [[buffer(5)]],
    constant Step& step [[buffer(6)]],
    uint tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint lanes [[threads_per_threadgroup]]) {
    threadgroup float red_hi[1024];
    threadgroup float red_lo[1024];
    uint row = long_rows[tg];
    uint begin = row_ptr[row];
    uint end = row_ptr[row + 1u];
    dsacc s = acc_zero();
    for (uint k = begin + lane; k < end; k += lanes) {
        acc_fma(s, ld(aval, k), ld(xbar, col_idx[k]));
    }
    red_hi[lane] = s.hi;
    red_lo[lane] = s.lo;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = lanes >> 1u; stride > 0u; stride >>= 1u) {
        if (lane < stride) {
            dsacc part = dsacc{red_hi[lane], red_lo[lane]};
            acc_add(part, ds{red_hi[lane + stride], red_lo[lane + stride]});
            red_hi[lane] = part.hi;
            red_lo[lane] = part.lo;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0u) {
        st(ax, row, acc_done(dsacc{red_hi[0], red_lo[0]}));
    }
}

inline void store4(device float2* y, uint row, ds a, ds b, ds c, ds d) {
    st(y, row, a);
    st(y, row + 1u, b);
    st(y, row + 2u, c);
    st(y, row + 3u, d);
}

kernel void pdhg_dual(
    device float2* y [[buffer(0)]],
    device const float2* ax [[buffer(1)]],
    device const float2* b [[buffer(2)]],
    device float2* yavg [[buffer(3)]],
    device const float2* sigma_diag [[buffer(4)]],
    constant Step& step [[buffer(5)]],
    uint tid [[thread_position_in_grid]]) {
    uint n_lin = step.n_eq + step.n_nn;
    // The trust cone is pdhg_trust. One thread for that cone serialises the step.
    uint n_jobs = n_lin + step.n_soc;
    if (tid >= n_jobs) {
        return;
    }
    if (tid < n_lin) {
        ds sigma = ds_mul(ld(sigma_diag, tid), step_sigma(step));
        ds pred = ds_addp(ld(y, tid), ds_mul(sigma, ds_subp(ld(ax, tid), ld(b, tid))));
        if (tid >= step.n_eq) {
            pred = ds_max0(pred);
        }
        st(y, tid, pred);
        accumulate(yavg, tid, pred, step.avg_count);
        return;
    }
    uint row = n_lin + (tid - n_lin) * 4u;
    ds sigma = ds_mul(ld(sigma_diag, row), step_sigma(step));
    ds t0 = ds_addp(ld(y, row), ds_mul(sigma, ds_subp(ld(ax, row), ld(b, row))));
    ds t1 = ds_addp(ld(y, row + 1u), ds_mul(sigma, ds_subp(ld(ax, row + 1u), ld(b, row + 1u))));
    ds t2 = ds_addp(ld(y, row + 2u), ds_mul(sigma, ds_subp(ld(ax, row + 2u), ld(b, row + 2u))));
    ds t3 = ds_addp(ld(y, row + 3u), ds_mul(sigma, ds_subp(ld(ax, row + 3u), ld(b, row + 3u))));
    ds nrm = ds_sqrt(ds_addp(ds_addp(ds_sqr(t1), ds_sqr(t2)), ds_sqr(t3)));
    if (!ds_gt(nrm, t0)) {
        store4(y, row, t0, t1, t2, t3);
    } else if (!ds_gt(nrm, ds_neg(t0))) {
        ds z = ds_zero();
        store4(y, row, z, z, z, z);
    } else {
        ds a = ds_div(ds_addp(nrm, t0), ds_scale(nrm, 2.0f));
        store4(y, row, ds_mul(a, nrm), ds_mul(a, t1), ds_mul(a, t2), ds_mul(a, t3));
    }
    accumulate(yavg, row, ld(y, row), step.avg_count);
    accumulate(yavg, row + 1u, ld(y, row + 1u), step.avg_count);
    accumulate(yavg, row + 2u, ld(y, row + 2u), step.avg_count);
    accumulate(yavg, row + 3u, ld(y, row + 3u), step.avg_count);
}

// Euclidean projection of the single trust cone. Every lane of one threadgroup
// owns a stride of the cone, instead of one thread walking every mobile vertex.
kernel void pdhg_trust(
    device float2* y [[buffer(0)]],
    device const float2* ax [[buffer(1)]],
    device const float2* b [[buffer(2)]],
    device float2* yavg [[buffer(3)]],
    device const float2* sigma_diag [[buffer(4)]],
    constant Step& step [[buffer(5)]],
    uint lane [[thread_index_in_threadgroup]],
    uint lanes [[threads_per_threadgroup]]) {
    threadgroup float red_hi[1024];
    threadgroup float red_lo[1024];
    uint row0 = step.n_eq + step.n_nn + step.n_soc * 4u;
    ds sigma = ds_mul(ld(sigma_diag, row0), step_sigma(step));
    ds t = ds_addp(ld(y, row0), ds_mul(sigma, ds_subp(ld(ax, row0), ld(b, row0))));
    if (lane == 0u) {
        st(y, row0, t);
    }
    dsacc nrm2 = acc_zero();
    for (uint i = 1u + lane; i < step.n_trust; i += lanes) {
        uint row = row0 + i;
        ds p = ds_addp(ld(y, row), ds_mul(sigma, ds_subp(ld(ax, row), ld(b, row))));
        st(y, row, p);
        acc_fma(nrm2, p, p);
    }
    red_hi[lane] = nrm2.hi;
    red_lo[lane] = nrm2.lo;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = lanes >> 1u; stride > 0u; stride >>= 1u) {
        if (lane < stride) {
            dsacc part = dsacc{red_hi[lane], red_lo[lane]};
            acc_add(part, ds{red_hi[lane + stride], red_lo[lane + stride]});
            red_hi[lane] = part.hi;
            red_lo[lane] = part.lo;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    ds nrm = ds_sqrt(acc_done(dsacc{red_hi[0], red_lo[0]}));
    if (ds_gt(nrm, t)) {
        if (!ds_gt(nrm, ds_neg(t))) {
            for (uint i = lane; i < step.n_trust; i += lanes) {
                st(y, row0 + i, ds_zero());
            }
        } else {
            ds a = ds_div(ds_addp(nrm, t), ds_scale(nrm, 2.0f));
            if (lane == 0u) {
                st(y, row0, ds_mul(a, nrm));
            }
            for (uint i = 1u + lane; i < step.n_trust; i += lanes) {
                st(y, row0 + i, ds_mul(a, ld(y, row0 + i)));
            }
        }
    }
    if (lane == 0u) {
        accumulate(yavg, row0, ld(y, row0), step.avg_count);
    }
    for (uint i = 1u + lane; i < step.n_trust; i += lanes) {
        accumulate(yavg, row0 + i, ld(y, row0 + i), step.avg_count);
    }
}
