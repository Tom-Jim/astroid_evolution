// Rheology and friction-angle chain of mathstrict.md §4, §10.2.2, §10.2.3, §10.3.
//
// One thread owns one surface node. The thread
//   1. builds second-order Taylor jets (u, v chart on the unit sphere, node at the
//      origin) of the radius R and of seven mode sums by one pass over the real
//      spherical harmonics,
//   2. assembles the surface velocity trace (Proposition 10.2) as a Cartesian
//      vector jet, reads D_ab = sym(e_a . d_b v) (§10.2.3),
//   3. solves the frozen-viscosity fixed point for the surface shear rate with the
//      traction closure D_na = -n.T.e_a / (2 eta), and
//   4. evaluates I_c, mu_d, eta, mu_eff, phi, and the slope slack of §10.3 / §10.4.3.
// Every number is double-single.


struct Params {
    uint node_count;
    uint l_max;
    uint max_iter;
    uint proj_dynamic;
    uint pad0;
    uint pad1;
    float2 d_p;
    float2 rho_p;
    float2 mu_s;
    float2 mu_2;
    float2 i0;
    float2 coh;
    float2 p_c;
    float2 gamma_eps;
    float2 p_min;
    float2 rho_bulk;
    float2 h_s;
    float2 tide_coef;
    float2 mu_y;
    float2 tol;
    float2 eps_overhang;
};

constant uint NODE_WORDS = 38;
constant uint OUT_DS = 48;
constant uint MODE_DS = 9;

inline ds dsv(float2 v) { return ds{v.x, v.y}; }

inline ds dot3(thread const ds* a, thread const ds* b) {
    dsacc s = acc_zero();
    acc_fma(s, a[0], b[0]);
    acc_fma(s, a[1], b[1]);
    acc_fma(s, a[2], b[2]);
    return acc_done(s);
}

// ---------------------------------------------------------------------------
// Jets. Taylor coefficients in the order 1, u, v, uu, uv, vv (J2) or 1, u, v (J1).
// A derivative drops one order of validity; the code never reads an invalid slot.
// ---------------------------------------------------------------------------

struct J2 {
    ds c[6];
};

struct J1 {
    ds c[3];
};

inline J2 j2_zero() {
    J2 r;
    for (int i = 0; i < 6; ++i) {
        r.c[i] = ds_zero();
    }
    return r;
}

inline J1 j1_zero() {
    J1 r;
    for (int i = 0; i < 3; ++i) {
        r.c[i] = ds_zero();
    }
    return r;
}

inline J2 j2_const(ds a) {
    J2 r = j2_zero();
    r.c[0] = a;
    return r;
}

inline J2 j2_mul(thread const J2& a, thread const J2& b) {
    J2 r;
    r.c[0] = ds_mul(a.c[0], b.c[0]);
    dsacc s = acc_zero();
    acc_fma(s, a.c[0], b.c[1]);
    acc_fma(s, a.c[1], b.c[0]);
    r.c[1] = acc_done(s);
    s = acc_zero();
    acc_fma(s, a.c[0], b.c[2]);
    acc_fma(s, a.c[2], b.c[0]);
    r.c[2] = acc_done(s);
    s = acc_zero();
    acc_fma(s, a.c[0], b.c[3]);
    acc_fma(s, a.c[1], b.c[1]);
    acc_fma(s, a.c[3], b.c[0]);
    r.c[3] = acc_done(s);
    s = acc_zero();
    acc_fma(s, a.c[0], b.c[4]);
    acc_fma(s, a.c[1], b.c[2]);
    acc_fma(s, a.c[2], b.c[1]);
    acc_fma(s, a.c[4], b.c[0]);
    r.c[4] = acc_done(s);
    s = acc_zero();
    acc_fma(s, a.c[0], b.c[5]);
    acc_fma(s, a.c[2], b.c[2]);
    acc_fma(s, a.c[5], b.c[0]);
    r.c[5] = acc_done(s);
    return r;
}

inline J2 j2_scale(thread const J2& a, ds k) {
    J2 r;
    for (int i = 0; i < 6; ++i) {
        r.c[i] = ds_mul(a.c[i], k);
    }
    return r;
}

inline J2 j2_sub(thread const J2& a, thread const J2& b) {
    J2 r;
    for (int i = 0; i < 6; ++i) {
        r.c[i] = ds_sub(a.c[i], b.c[i]);
    }
    return r;
}

inline J2 j2_add(thread const J2& a, thread const J2& b) {
    J2 r;
    for (int i = 0; i < 6; ++i) {
        r.c[i] = ds_add(a.c[i], b.c[i]);
    }
    return r;
}

inline void j2_axpy(thread J2& acc, ds k, thread const J2& y) {
    for (int i = 0; i < 6; ++i) {
        dsacc s = acc_from(acc.c[i]);
        acc_fma(s, k, y.c[i]);
        acc.c[i] = acc_done(s);
    }
}

inline void j1_axpy(thread J1& acc, ds k, thread const J2& y) {
    for (int i = 0; i < 3; ++i) {
        dsacc s = acc_from(acc.c[i]);
        acc_fma(s, k, y.c[i]);
        acc.c[i] = acc_done(s);
    }
}

inline J1 j1_of(thread const J2& a) {
    J1 r;
    r.c[0] = a.c[0];
    r.c[1] = a.c[1];
    r.c[2] = a.c[2];
    return r;
}

inline J1 j1_du(thread const J2& a) {
    J1 r;
    r.c[0] = a.c[1];
    r.c[1] = ds_scale(a.c[3], 2.0f);
    r.c[2] = a.c[4];
    return r;
}

inline J1 j1_dv(thread const J2& a) {
    J1 r;
    r.c[0] = a.c[2];
    r.c[1] = a.c[4];
    r.c[2] = ds_scale(a.c[5], 2.0f);
    return r;
}

inline J1 j1_mul(thread const J1& a, thread const J1& b) {
    J1 r;
    r.c[0] = ds_mul(a.c[0], b.c[0]);
    dsacc s = acc_zero();
    acc_fma(s, a.c[0], b.c[1]);
    acc_fma(s, a.c[1], b.c[0]);
    r.c[1] = acc_done(s);
    s = acc_zero();
    acc_fma(s, a.c[0], b.c[2]);
    acc_fma(s, a.c[2], b.c[0]);
    r.c[2] = acc_done(s);
    return r;
}

inline J1 j1_add(thread const J1& a, thread const J1& b) {
    J1 r;
    for (int i = 0; i < 3; ++i) {
        r.c[i] = ds_add(a.c[i], b.c[i]);
    }
    return r;
}

inline J1 j1_sub(thread const J1& a, thread const J1& b) {
    J1 r;
    for (int i = 0; i < 3; ++i) {
        r.c[i] = ds_sub(a.c[i], b.c[i]);
    }
    return r;
}

inline J1 j1_scale(thread const J1& a, ds k) {
    J1 r;
    for (int i = 0; i < 3; ++i) {
        r.c[i] = ds_mul(a.c[i], k);
    }
    return r;
}

inline J1 j1_neg(thread const J1& a) {
    J1 r;
    for (int i = 0; i < 3; ++i) {
        r.c[i] = ds_neg(a.c[i]);
    }
    return r;
}

// 1/a through the first order: f0 = 1/a0, f1 = -a1 f0^2, f2 = -a2 f0^2.
inline J1 j1_recip(thread const J1& a) {
    J1 r;
    ds f0 = ds_recip(a.c[0]);
    ds f02 = ds_sqr(f0);
    r.c[0] = f0;
    r.c[1] = ds_neg(ds_mul(a.c[1], f02));
    r.c[2] = ds_neg(ds_mul(a.c[2], f02));
    return r;
}

struct V1 {
    J1 c[3];
};

inline J1 v1_dot(thread const V1& a, thread const V1& b) {
    J1 r = j1_mul(a.c[0], b.c[0]);
    r = j1_add(r, j1_mul(a.c[1], b.c[1]));
    return j1_add(r, j1_mul(a.c[2], b.c[2]));
}

// ---------------------------------------------------------------------------
// Stage 1: one pass over the real spherical harmonics.
// ---------------------------------------------------------------------------

struct Modes {
    J2 r;
    J2 z1;
    J2 z4;
    J2 z6;
    J1 z3;
    J1 z5;
    J1 z7;
    J1 p0;
    J1 lap;
};

inline void accumulate(
    thread Modes& m, constant float* table, uint idx, thread const J2& y) {
    constant float* c = table + idx * (MODE_DS * 2);
    j2_axpy(m.r, ds{c[0], c[1]}, y);
    j1_axpy(m.lap, ds{c[2], c[3]}, y);
    j2_axpy(m.z1, ds{c[4], c[5]}, y);
    j1_axpy(m.z3, ds{c[6], c[7]}, y);
    j2_axpy(m.z4, ds{c[8], c[9]}, y);
    j1_axpy(m.z5, ds{c[10], c[11]}, y);
    j2_axpy(m.z6, ds{c[12], c[13]}, y);
    j1_axpy(m.z7, ds{c[14], c[15]}, y);
    j1_axpy(m.p0, ds{c[16], c[17]}, y);
}

// ---------------------------------------------------------------------------
// Stage 3/4: closure.
// ---------------------------------------------------------------------------

struct Closure {
    ds mu_d;
    ds tau;
    ds inertial;
};

inline Closure dynamic_closure(constant Params& P, ds shear, ds p_star, ds p_tilde) {
    Closure c;
    ds d = dsv(P.d_p);
    ds scale = ds_sqrt(ds_div(p_star, dsv(P.rho_p)));
    ds df = ds_mul(d, shear);
    ds mu_s = dsv(P.mu_s);
    ds den = ds_add(ds_mul(dsv(P.i0), scale), df);
    if (den.hi > 0.0f) {
        c.mu_d = ds_add(mu_s, ds_mul(ds_subp(dsv(P.mu_2), mu_s), ds_div(df, den)));
        c.inertial = ds_div(df, scale);
    } else {
        c.mu_d = mu_s;
        c.inertial = ds_zero();
    }
    c.tau = ds_add(dsv(P.coh), ds_mul(c.mu_d, p_tilde));
    return c;
}

kernel void rheo_chain(
    device const float* nodes [[buffer(0)]],
    constant Params& P [[buffer(1)]],
    constant float* mode_table [[buffer(2)]],
    constant float* recur [[buffer(3)]],
    device float* out [[buffer(4)]],
    device uint* stats [[buffer(5)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint tg [[threadgroup_position_in_grid]],
    uint sgs [[simdgroups_per_threadgroup]]) {
    uint iters_used = 0;
    uint unconverged = 0;
    uint invalid = 0;

    if (gid < P.node_count) {
        device const float* nd = nodes + gid * NODE_WORDS;
#define RD(i) ds{nd[2 * (i)], nd[2 * (i) + 1]}
        ds w0[3] = {RD(0), RD(1), RD(2)};
        ds t1[3] = {RD(3), RD(4), RD(5)};
        ds t2[3] = {RD(6), RD(7), RD(8)};
        ds gv[3] = {RD(9), RD(10), RD(11)};
        ds hs[6] = {RD(12), RD(13), RD(14), RD(15), RD(16), RD(17)};
        ds pressure = RD(18);
#undef RD

        // omega(u, v) = u t1 + v t2 + sqrt(1 - u^2 - v^2) w0, exact through second order.
        J2 om[3];
        for (int i = 0; i < 3; ++i) {
            om[i].c[0] = w0[i];
            om[i].c[1] = t1[i];
            om[i].c[2] = t2[i];
            om[i].c[3] = ds_scale(ds_neg(w0[i]), 0.5f);
            om[i].c[4] = ds_zero();
            om[i].c[5] = om[i].c[3];
        }

        Modes M;
        M.r = j2_zero();
        M.z1 = j2_zero();
        M.z4 = j2_zero();
        M.z6 = j2_zero();
        M.z3 = j1_zero();
        M.z5 = j1_zero();
        M.z7 = j1_zero();
        M.p0 = j1_zero();
        M.lap = j1_zero();

        const uint L = P.l_max;
        const uint pmm_base = (L + 1) * (L + 2) / 2 * 4;
        J2 cre = j2_const(ds_f(1.0f));
        J2 cim = j2_zero();
        for (uint m = 0; m <= L; ++m) {
            J2 p1 = j2_zero();
            J2 p2 = j2_zero();
            for (uint l = m; l <= L; ++l) {
                J2 pl;
                if (l == m) {
                    pl = j2_const(ds{recur[pmm_base + 2 * m], recur[pmm_base + 2 * m + 1]});
                } else {
                    constant float* ab = recur + (l * (l + 1) / 2 + m) * 4;
                    J2 zp = j2_mul(om[2], p1);
                    if (l > m + 1) {
                        j2_axpy(zp, ds{-ab[2], -ab[3]}, p2);
                    }
                    pl = j2_scale(zp, ds{ab[0], ab[1]});
                }
                if (m == 0) {
                    accumulate(M, mode_table, l * l + l, pl);
                } else {
                    J2 yc = j2_mul(pl, cre);
                    J2 ys = j2_mul(pl, cim);
                    accumulate(M, mode_table, l * l + l + m, yc);
                    accumulate(M, mode_table, l * l + l - m, ys);
                }
                p2 = p1;
                p1 = pl;
            }
            // (x + i y)^(m+1)
            J2 nre = j2_sub(j2_mul(cre, om[0]), j2_mul(cim, om[1]));
            J2 nim = j2_add(j2_mul(cre, om[1]), j2_mul(cim, om[0]));
            cre = nre;
            cim = nim;
        }

        // -------------------------------------------------------------------
        // Stage 2: first-order jets of the surface geometry and velocity.
        // -------------------------------------------------------------------
        V1 wu;
        V1 wv;
        V1 wj;
        for (int i = 0; i < 3; ++i) {
            wu.c[i] = j1_du(om[i]);
            wv.c[i] = j1_dv(om[i]);
            wj.c[i] = j1_of(om[i]);
        }
        J1 guu = v1_dot(wu, wu);
        J1 guv = v1_dot(wu, wv);
        J1 gvv = v1_dot(wv, wv);
        J1 idet = j1_recip(j1_sub(j1_mul(guu, gvv), j1_mul(guv, guv)));
        J1 iuu = j1_mul(gvv, idet);
        J1 ivv = j1_mul(guu, idet);
        J1 iuv = j1_neg(j1_mul(guv, idet));

        // Surface gradient of a scalar jet f as f_a omega^a coefficients (a, b): grad f = a wu + b wv.
#define GRAD_AB(f, A, B)                                                                       \
    J1 A##_fu = j1_du(f);                                                                      \
    J1 A##_fv = j1_dv(f);                                                                      \
    J1 A = j1_add(j1_mul(iuu, A##_fu), j1_mul(iuv, A##_fv));                                   \
    J1 B = j1_add(j1_mul(iuv, A##_fu), j1_mul(ivv, A##_fv));

        J1 R1 = j1_of(M.r);
        J1 Ru = j1_du(M.r);
        J1 Rv = j1_dv(M.r);
        GRAD_AB(M.r, aR, bR)
        GRAD_AB(M.z1, aZ1, bZ1)
        GRAD_AB(M.z4, a4, b4)
        GRAD_AB(M.z6, a6, b6)
#undef GRAD_AB
        J1 Z1 = j1_of(M.z1);
        J1 Rinv = j1_recip(R1);
        J1 gr2 = j1_add(j1_mul(Ru, aR), j1_mul(Rv, bR));
        J1 dRdZ = j1_add(j1_mul(Ru, aZ1), j1_mul(Rv, bZ1));
        // v_r = R P0 - 2 grad R . grad Z1 - Z1 lap R - Z3 |grad R|^2 / R
        J1 vr = j1_mul(R1, M.p0);
        vr = j1_sub(vr, j1_scale(dRdZ, ds_f(2.0f)));
        vr = j1_sub(vr, j1_mul(Z1, M.lap));
        vr = j1_sub(vr, j1_mul(j1_mul(M.z3, gr2), Rinv));

        V1 gradR;
        V1 vs;
        V1 wt;
        for (int i = 0; i < 3; ++i) {
            gradR.c[i] = j1_add(j1_mul(aR, wu.c[i]), j1_mul(bR, wv.c[i]));
            J1 g4 = j1_add(j1_mul(a4, wu.c[i]), j1_mul(b4, wv.c[i]));
            J1 g6 = j1_add(j1_mul(a6, wu.c[i]), j1_mul(b6, wv.c[i]));
            // v_S = R grad Z4 + Z5 grad R ; W_T = R grad Z6 + Z7 grad R
            vs.c[i] = j1_add(j1_mul(R1, g4), j1_mul(M.z5, gradR.c[i]));
            wt.c[i] = j1_add(j1_mul(R1, g6), j1_mul(M.z7, gradR.c[i]));
        }
        V1 v;
        // v^T = W_T x e_r
        v.c[0] = j1_sub(j1_mul(wt.c[1], wj.c[2]), j1_mul(wt.c[2], wj.c[1]));
        v.c[1] = j1_sub(j1_mul(wt.c[2], wj.c[0]), j1_mul(wt.c[0], wj.c[2]));
        v.c[2] = j1_sub(j1_mul(wt.c[0], wj.c[1]), j1_mul(wt.c[1], wj.c[0]));
        for (int i = 0; i < 3; ++i) {
            v.c[i] = j1_add(v.c[i], vs.c[i]);
            v.c[i] = j1_add(v.c[i], j1_mul(vr, wj.c[i]));
        }

        // Values at the node.
        ds R0 = M.r.c[0];
        ds Ru0 = M.r.c[1];
        ds Rv0 = M.r.c[2];
        ds e1[3];
        ds e2[3];
        ds vv0[3];
        ds vu[3];
        ds vvv[3];
        for (int i = 0; i < 3; ++i) {
            dsacc s = acc_zero();
            acc_fma(s, Ru0, w0[i]);
            acc_fma(s, R0, t1[i]);
            e1[i] = acc_done(s);
            s = acc_zero();
            acc_fma(s, Rv0, w0[i]);
            acc_fma(s, R0, t2[i]);
            e2[i] = acc_done(s);
            vv0[i] = v.c[i].c[0];
            vu[i] = v.c[i].c[1];
            vvv[i] = v.c[i].c[2];
        }
        ds g11 = dot3(e1, e1);
        ds g12 = dot3(e1, e2);
        ds g22 = dot3(e2, e2);
        ds det;
        {
            dsacc s = acc_zero();
            acc_fma(s, g11, g22);
            acc_fnma(s, g12, g12);
            det = acc_done(s);
        }
        ds sqrt_g = ds_sqrt(det);
        ds idt = ds_recip(det);
        ds gi11 = ds_mul(g22, idt);
        ds gi22 = ds_mul(g11, idt);
        ds gi12 = ds_neg(ds_mul(g12, idt));

        ds d11 = dot3(e1, vu);
        ds d22 = dot3(e2, vvv);
        ds d12 = ds_scale(ds_add(dot3(e1, vvv), dot3(e2, vu)), 0.5f);

        ds m11, m12, m21, m22;
        {
            dsacc s = acc_zero();
            acc_fma(s, gi11, d11);
            acc_fma(s, gi12, d12);
            m11 = acc_done(s);
            s = acc_zero();
            acc_fma(s, gi11, d12);
            acc_fma(s, gi12, d22);
            m12 = acc_done(s);
            s = acc_zero();
            acc_fma(s, gi12, d11);
            acc_fma(s, gi22, d12);
            m21 = acc_done(s);
            s = acc_zero();
            acc_fma(s, gi12, d12);
            acc_fma(s, gi22, d22);
            m22 = acc_done(s);
        }
        ds tr_m = ds_add(m11, m22);
        ds tr_m2;
        {
            dsacc s = acc_zero();
            acc_fma(s, m11, m11);
            acc_fma(s, m22, m22);
            acc_fma(s, ds_scale(m12, 2.0f), m21);
            tr_m2 = acc_done(s);
        }
        // gamma^2 = 2 [ (tr M)^2 + tr(M^2) + 2 g^{ab} D_na D_nb ]
        ds s0 = ds_scale(ds_add(ds_sqr(tr_m), tr_m2), 2.0f);

        // Outward unit normal.
        ds nrm[3];
        {
            ds cx = ds_subp(ds_mul(e1[1], e2[2]), ds_mul(e1[2], e2[1]));
            ds cy = ds_subp(ds_mul(e1[2], e2[0]), ds_mul(e1[0], e2[2]));
            ds cz = ds_subp(ds_mul(e1[0], e2[1]), ds_mul(e1[1], e2[0]));
            ds inv = ds_recip(sqrt_g);
            nrm[0] = ds_mul(cx, inv);
            nrm[1] = ds_mul(cy, inv);
            nrm[2] = ds_mul(cz, inv);
        }

        // Tide: T = tide_coef * dev H. Traction t_a = n . T . e_a.
        ds trh3 = ds_div(ds_add(ds_add(hs[0], hs[1]), hs[2]), ds_f(3.0f));
        ds dh[3][3];
        dh[0][0] = ds_subp(hs[0], trh3);
        dh[1][1] = ds_subp(hs[1], trh3);
        dh[2][2] = ds_subp(hs[2], trh3);
        dh[0][1] = hs[3];
        dh[1][0] = hs[3];
        dh[0][2] = hs[4];
        dh[2][0] = hs[4];
        dh[1][2] = hs[5];
        dh[2][1] = hs[5];
        ds tcoef = dsv(P.tide_coef);
        ds hn[3];
        for (int i = 0; i < 3; ++i) {
            dsacc s = acc_zero();
            acc_fma(s, dh[i][0], nrm[0]);
            acc_fma(s, dh[i][1], nrm[1]);
            acc_fma(s, dh[i][2], nrm[2]);
            hn[i] = acc_done(s);
        }
        ds ta1 = ds_mul(tcoef, dot3(hn, e1));
        ds ta2 = ds_mul(tcoef, dot3(hn, e2));
        ds tg;
        {
            dsacc s = acc_zero();
            acc_fma(s, gi11, ds_sqr(ta1));
            acc_fma(s, gi22, ds_sqr(ta2));
            acc_fma(s, ds_scale(gi12, 2.0f), ds_mul(ta1, ta2));
            tg = acc_done(s);
        }
        ds frob2;
        {
            dsacc s = acc_zero();
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    acc_fma(s, dh[i][j], dh[i][j]);
                }
            }
            frob2 = acc_done(s);
        }
        ds tide_frob = ds_mul(ds_abs(tcoef), ds_sqrt(frob2));

        // Pressures.
        ds p_min = dsv(P.p_min);
        ds p_tilde = ds_max0(pressure);
        if (ds_gt(p_min, p_tilde)) {
            p_tilde = p_min;
        }
        ds p_star = ds_add(p_tilde, dsv(P.p_c));
        ds eps2 = ds_sqr(dsv(P.gamma_eps));

        // Fixed point for the frozen viscosity (§10.2.3):
        //   gamma^2 = S0 + tg (gamma^2 + eps^2) / tau(gamma)^2.
        ds g2 = s0;
        bool converged = true;
        if (tg.hi > 0.0f) {
            converged = false;
            ds tol = dsv(P.tol);
            for (uint it = 0; it < P.max_iter; ++it) {
                Closure cl = dynamic_closure(P, ds_sqrt(g2), p_star, p_tilde);
                ds ratio = ds_div(tg, ds_sqr(cl.tau));
                ds next = ds_add(s0, ds_mul(ratio, ds_add(g2, eps2)));
                iters_used = it + 1;
                if (!(next.hi == next.hi) || next.hi > 1.0e30f) {
                    g2 = next;
                    break;
                }
                ds diff = ds_abs(ds_subp(next, g2));
                ds scale = ds_add(next, eps2);
                g2 = next;
                if (!ds_gt(diff, ds_mul(tol, scale))) {
                    converged = true;
                    break;
                }
            }
        }
        if (!converged) {
            unconverged = 1;
        }
        ds shear = ds_sqrt(g2);
        Closure cl = dynamic_closure(P, shear, p_star, p_tilde);
        ds root = ds_sqrt(ds_add(g2, eps2));
        ds eta = ds_div(cl.tau, root);
        // D_na = -t_a / (2 eta)
        ds half_inv_eta = ds_scale(ds_div(root, cl.tau), 0.5f);
        ds dn1 = ds_neg(ds_mul(ta1, half_inv_eta));
        ds dn2 = ds_neg(ds_mul(ta2, half_inv_eta));

        // Friction angle (§10.3).
        ds gn = dot3(gv, nrm);
        ds gg = dot3(gv, gv);
        ds gnorm = ds_sqrt(gg);
        ds p_ref = ds_mul(ds_mul(dsv(P.rho_bulk), ds_abs(gn)), dsv(P.h_s));
        ds p_star_ref = ds_add(p_ref, dsv(P.p_c));
        if (ds_gt(p_min, p_star_ref)) {
            p_star_ref = p_min;
        }
        ds mu_proj = (P.proj_dynamic != 0) ? cl.mu_d : dsv(P.mu_y);
        ds num = ds_subp(
            ds_addp(dsv(P.coh), ds_mul(mu_proj, p_ref)),
            ds_mul(tide_frob, ds_mk(0.707106769f, 1.21016175e-08f)));
        ds mu_eff = ds_div(ds_max0(num), p_star_ref);
        ds phi = ds_atan_pos(mu_eff);
        ds mu2 = ds_sqr(mu_eff);
        ds sin2 = ds_div(mu2, ds_add(ds_f(1.0f), mu2));
        ds sin_phi = ds_sqrt(sin2);

        // Slope proxy of the SOCP row (§10.4.3) at d = 0.
        ds gxn[3] = {
            ds_subp(ds_mul(gv[1], nrm[2]), ds_mul(gv[2], nrm[1])),
            ds_subp(ds_mul(gv[2], nrm[0]), ds_mul(gv[0], nrm[2])),
            ds_subp(ds_mul(gv[0], nrm[1]), ds_mul(gv[1], nrm[0]))};
        ds sin_slope = ds_div(ds_sqrt(dot3(gxn, gxn)), gnorm);
        ds slack = ds_max0(ds_subp(sin_slope, sin_phi));
        ds overhang = ds_subp(ds_neg(ds_div(gn, gnorm)), dsv(P.eps_overhang));

        ds vr0 = vr.c[0];
        ds vn = dot3(vv0, nrm);

        device float* o = out + gid * (OUT_DS * 2);
#define WR(i, val)                                                                             \
    o[2 * (i)] = (val).hi;                                                                     \
    o[2 * (i) + 1] = (val).lo;
        WR(0, R0)
        WR(1, nrm[0])
        WR(2, nrm[1])
        WR(3, nrm[2])
        WR(4, sqrt_g)
        WR(5, vv0[0])
        WR(6, vv0[1])
        WR(7, vv0[2])
        WR(8, e1[0])
        WR(9, e1[1])
        WR(10, e1[2])
        WR(11, e2[0])
        WR(12, e2[1])
        WR(13, e2[2])
        WR(14, d11)
        WR(15, d12)
        WR(16, d22)
        WR(17, ds_neg(tr_m))
        WR(18, dn1)
        WR(19, dn2)
        WR(20, shear)
        WR(21, cl.inertial)
        WR(22, cl.mu_d)
        WR(23, eta)
        WR(24, cl.tau)
        WR(25, p_tilde)
        WR(26, p_star)
        WR(27, p_ref)
        WR(28, mu_proj)
        WR(29, mu_eff)
        WR(30, phi)
        WR(31, sin2)
        WR(32, sin_slope)
        WR(33, slack)
        WR(34, overhang)
        WR(35, tide_frob)
        WR(36, vr0)
        WR(37, vn)
        WR(38, s0)
        WR(39, tg)
        WR(40, ds_f(float(iters_used)))
        WR(41, ds_f(converged ? 1.0f : 0.0f))
        WR(42, sin_phi)
        WR(43, g11)
        WR(44, g12)
        WR(45, g22)
#undef WR
        if (!(R0.hi > 0.0f) || !(det.hi > 0.0f) || !(shear.hi == shear.hi)) {
            invalid = 1;
        }
    }

    // Simdgroup reductions: worst iteration count, failures, and invalid nodes.
    uint worst = simd_max(iters_used);
    uint bad = simd_sum(unconverged);
    uint inv = simd_sum(invalid);
    if (lane == 0) {
        uint s = (tg * sgs + sg) * 3;
        stats[s] = worst;
        stats[s + 1] = bad;
        stats[s + 2] = inv;
    }
}
