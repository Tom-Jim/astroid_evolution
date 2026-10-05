// One thread per vertex. Coefficients already carry the host spectral filter
// and, for m != 0, the sqrt(2) of the real harmonic.

struct Params {
    uint vertex_count;
    uint l_max;
    uint add_residual;
    uint pad1;
};

kernel void sh_eval(
    device const float* dirs [[buffer(0)]],
    constant Params& P [[buffer(1)]],
    constant float* coeff [[buffer(2)]],
    constant float* recur [[buffer(3)]],
    device float* radii [[buffer(4)]],
    device uint* stats [[buffer(5)]],
    device const float* residual [[buffer(6)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint tg [[threadgroup_position_in_grid]],
    uint sgs [[simdgroups_per_threadgroup]]) {
    uint bad = 0;
    if (gid < P.vertex_count) {
        device const float* d = dirs + gid * 6;
        ds x = ds{d[0], d[1]};
        ds y = ds{d[2], d[3]};
        ds z = ds{d[4], d[5]};
        const uint L = P.l_max;
        const uint pmm_base = (L + 1) * (L + 2) / 2 * 4;
        dsacc acc = acc_zero();
        ds cre = ds_f(1.0f);
        ds cim = ds_zero();
        for (uint m = 0; m <= L; ++m) {
            ds p1 = ds_zero();
            ds p2 = ds_zero();
            for (uint l = m; l <= L; ++l) {
                ds pl;
                if (l == m) {
                    pl = ds{recur[pmm_base + 2 * m], recur[pmm_base + 2 * m + 1]};
                } else {
                    constant float* ab = recur + (l * (l + 1) / 2 + m) * 4;
                    ds a = ds{ab[0], ab[1]};
                    ds zp = ds_mul(z, p1);
                    if (l > m + 1) {
                        ds b = ds{ab[2], ab[3]};
                        zp = ds_subp(zp, ds_mul(b, p2));
                    }
                    pl = ds_mul(a, zp);
                }
                if (m == 0) {
                    constant float* c = coeff + 2 * (l * l + l);
                    acc_fma(acc, ds{c[0], c[1]}, pl);
                } else {
                    constant float* cp = coeff + 2 * (l * l + l + m);
                    constant float* cm = coeff + 2 * (l * l + l - m);
                    acc_fma(acc, ds{cp[0], cp[1]}, ds_mul(pl, cre));
                    acc_fma(acc, ds{cm[0], cm[1]}, ds_mul(pl, cim));
                }
                p2 = p1;
                p1 = pl;
            }
            ds nre = ds_subp(ds_mul(cre, x), ds_mul(cim, y));
            ds nim = ds_addp(ds_mul(cre, y), ds_mul(cim, x));
            cre = nre;
            cim = nim;
        }
        ds r = acc_done(acc);
        if (P.add_residual != 0u) {
            device const float* h = residual + gid * 2;
            r = ds_addp(r, ds{h[0], h[1]});
        }
        radii[2 * gid] = r.hi;
        radii[2 * gid + 1] = r.lo;
        bool positive = r.hi > 0.0f || (r.hi == 0.0f && r.lo > 0.0f);
        bad = positive ? 0u : 1u;
    }
    uint reduced = simd_sum(bad);
    if (lane == 0) {
        stats[tg * sgs + sg] = reduced;
    }
}
