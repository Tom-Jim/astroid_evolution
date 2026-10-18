// One Chambolle–Pock step for the diagonal-quadratic cone program assembled in
// `socp.rs`. Arithmetic is double-single. The saddle point is
//
//   min_z max_{y ∈ K*}  ½ zᵀ P z + qᵀ z + yᵀ (A z − b),
//
// with θ = 1 extrapolation. K* is free on equality rows, nonnegative on the
// linear inequalities, and a second-order cone on each block of four rows.
//
//   z⁺ = (z − τ (q + Aᵀ y)) / (1 + τ P)
//   z̄ = 2 z⁺ − z
//   y⁺ = Π_{K*}(y + σ (A z̄ − b))

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
    uint avg_count;
    float tau_hi;
    float tau_lo;
    float sigma_hi;
    float sigma_lo;
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
    constant Step& step [[buffer(6)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= step.nx) {
        return;
    }
    ds tau = step_tau(step);
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
    dsacc s = acc_zero();
    uint end = row_ptr[tid + 1u];
    for (uint k = row_ptr[tid]; k < end; ++k) {
        acc_fma(s, ld(aval, k), ld(xbar, col_idx[k]));
    }
    st(ax, tid, acc_done(s));
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
    constant Step& step [[buffer(4)]],
    uint tid [[thread_position_in_grid]]) {
    uint n_lin = step.n_eq + step.n_nn;
    if (tid >= n_lin + step.n_soc) {
        return;
    }
    ds sigma = step_sigma(step);
    if (tid < n_lin) {
        ds pred = ds_addp(ld(y, tid), ds_mul(sigma, ds_subp(ld(ax, tid), ld(b, tid))));
        if (tid >= step.n_eq) {
            pred = ds_max0(pred);
        }
        st(y, tid, pred);
        accumulate(yavg, tid, pred, step.avg_count);
        return;
    }
    uint row = n_lin + (tid - n_lin) * 4u;
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
