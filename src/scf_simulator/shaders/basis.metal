// Jacobi–spherical-harmonic basis (definition 8.5) in double-single arithmetic.
//
// P[Φ] = ∇(Φ + x·∇Φ) − x ΔΦ and T[Ψ] = ∇Ψ × x, with Φ = R² ψ(ζ) Y and
// Ψ = R ψ Y (lemma 5.5). R and Y depend on x only through x/|x|, so a jet
// computed on a ray scales onto every other radius:
// ∂^α f(ρ û) = ρ^{−|α|} ∂^α f(û). Order-0 pressure is the scalar recurrence:
// the value is constant along the ray and only ζ = r/R changes.

constexpr int jlen(int ord) {
    return ord <= 1 ? 4 : ord == 2 ? 10 : ord == 3 ? 20 : 35;
}

constexpr int ncoeff(int ord) {
    return ord <= 0 ? 1 : ord == 1 ? 4 : ord == 2 ? 10 : ord == 3 ? 20 : 35;
}

static_assert(jlen(0) == 4, "order 0 is padded");
static_assert(jlen(3) == 20, "order 3");
static_assert(ncoeff(4) == 35, "order 4");

template <int ORD>
struct Jt {
    ds c[jlen(ORD)];
};

struct BasisParams {
    uint n_rays;
    uint n_per;
    uint n_points;
    uint l_mode;
    uint l_surf;
    uint k_count;
    uint n_out;
    uint n_lift;
    uint n_comp;
    uint poly_stride;
    uint top_pow;
    uint want_v;
    uint want_d;
    uint want_lap;
    uint want_gd;
    uint gradients;
    uint nmul0, nmul1, nmul2, nmul3, nmul4;
    uint moff0, moff1, moff2, moff3, moff4;
    uint up_off, mono_off, deg_off, pair_off;
    float sqrt2_hi, sqrt2_lo;
    float pmm_hi, pmm_lo;
};

static_assert(sizeof(BasisParams) == 136, "BasisParams layout must match the Rust host");

template <int ORD>
inline constant uint* mul_tab(constant uint* tables, constant BasisParams& p) {
    if (ORD <= 0) return tables + p.moff0;
    if (ORD == 1) return tables + p.moff1;
    if (ORD == 2) return tables + p.moff2;
    if (ORD == 3) return tables + p.moff3;
    return tables + p.moff4;
}

template <int ORD>
inline uint mul_count(constant BasisParams& p) {
    if (ORD <= 0) return p.nmul0;
    if (ORD == 1) return p.nmul1;
    if (ORD == 2) return p.nmul2;
    if (ORD == 3) return p.nmul3;
    return p.nmul4;
}

inline uint lm_of(int l, int m) { return uint(l * l + l + m); }

inline uint vel_dof(int l, int m, uint tor, uint k, uint k_count) {
    int blk = l * l + l + m - 1;
    return uint(int(2u * k_count) * blk + int(tor ? k_count : 0u) + int(k));
}

inline ds coef_a(int l, int m) {
    float lf = float(l);
    float mf = float(m);
    return ds_sqrt(ds_div(ds_f(4.0f * lf * lf - 1.0f), ds_f(lf * lf - mf * mf)));
}

inline ds coef_b(int l, int m) {
    float lm = float(l) - 1.0f;
    float mf = float(m);
    return ds_sqrt(ds_div(ds_f(lm * lm - mf * mf), ds_f(4.0f * lm * lm - 1.0f)));
}

inline ds coeff_at(device const float2* coeff, int l, int m) {
    float2 c = coeff[lm_of(l, m)];
    return ds_mk(c.x, c.y);
}

template <int ORD>
inline void j_zero(thread Jt<ORD>& r) {
    for (int i = 0; i < jlen(ORD); ++i) r.c[i] = ds_zero();
}

template <int ORD>
inline void j_const(thread Jt<ORD>& r, ds v) {
    j_zero<ORD>(r);
    r.c[0] = v;
}

template <int ORD>
inline void j_var(thread Jt<ORD>& r, int axis, ds x0) {
    j_const<ORD>(r, x0);
    if (ORD >= 1) r.c[1 + axis] = ds_f(1.0f);
}

template <int ORD>
inline void j_mul(thread Jt<ORD>& r, thread const Jt<ORD>& a, thread const Jt<ORD>& b, constant uint* tables,
                  constant BasisParams& p) {
    dsacc acc[35];
    for (int i = 0; i < jlen(ORD); ++i) acc[i] = acc_zero();
    constant uint* tab = mul_tab<ORD>(tables, p);
    uint n = mul_count<ORD>(p);
    for (uint t = 0; t < n; ++t) {
        uint e = tab[t];
        acc_fma(acc[(e >> 16) & 255u], a.c[e & 255u], b.c[(e >> 8) & 255u]);
    }
    for (int i = 0; i < jlen(ORD); ++i) r.c[i] = acc_done(acc[i]);
}

template <int ORD>
inline void j_add(thread Jt<ORD>& r, thread const Jt<ORD>& a, thread const Jt<ORD>& b) {
    for (int i = 0; i < ncoeff(ORD); ++i) r.c[i] = ds_addp(a.c[i], b.c[i]);
    for (int i = ncoeff(ORD); i < jlen(ORD); ++i) r.c[i] = ds_zero();
}

template <int ORD>
inline void j_add_into(thread Jt<ORD>& r, thread const Jt<ORD>& a) {
    for (int i = 0; i < ncoeff(ORD); ++i) r.c[i] = ds_addp(r.c[i], a.c[i]);
}

template <int ORD>
inline void j_sub(thread Jt<ORD>& r, thread const Jt<ORD>& a, thread const Jt<ORD>& b) {
    for (int i = 0; i < ncoeff(ORD); ++i) r.c[i] = ds_subp(a.c[i], b.c[i]);
    for (int i = ncoeff(ORD); i < jlen(ORD); ++i) r.c[i] = ds_zero();
}

template <int ORD>
inline void j_add3(thread Jt<ORD>& r, thread const Jt<ORD>& a, thread const Jt<ORD>& b, thread const Jt<ORD>& c) {
    for (int i = 0; i < ncoeff(ORD); ++i) r.c[i] = ds_addp(ds_addp(a.c[i], b.c[i]), c.c[i]);
    for (int i = ncoeff(ORD); i < jlen(ORD); ++i) r.c[i] = ds_zero();
}

template <int ORD>
inline void j_scale(thread Jt<ORD>& r, thread const Jt<ORD>& a, ds s) {
    for (int i = 0; i < ncoeff(ORD); ++i) r.c[i] = ds_mul(a.c[i], s);
    for (int i = ncoeff(ORD); i < jlen(ORD); ++i) r.c[i] = ds_zero();
}

template <int ORD>
inline void j_axpy(thread Jt<ORD>& r, ds s, thread const Jt<ORD>& a) {
    for (int i = 0; i < ncoeff(ORD); ++i) r.c[i] = ds_addp(r.c[i], ds_mul(s, a.c[i]));
}

template <int FROM, int TO>
inline void j_trunc(thread Jt<TO>& r, thread const Jt<FROM>& a) {
    for (int i = 0; i < ncoeff(TO); ++i) r.c[i] = a.c[i];
    for (int i = ncoeff(TO); i < jlen(TO); ++i) r.c[i] = ds_zero();
}

template <int ORD>
inline void j_deriv(thread Jt<ORD - 1>& r, thread const Jt<ORD>& a, int axis, constant uint* tables,
                    constant BasisParams& p) {
    constant uint* up = tables + p.up_off;
    constant uint* mono = tables + p.mono_off;
    j_zero<ORD - 1>(r);
    for (int i = 0; i < ncoeff(ORD - 1); ++i) {
        r.c[i] = ds_mulf(float(mono[i * 3 + axis] + 1u), a.c[up[axis * 35 + i]]);
    }
}

template <int ORD>
inline void j_mul_x(thread Jt<ORD>& r, thread const Jt<ORD>& a, int axis, ds x0, constant uint* tables,
                    constant BasisParams& p) {
    for (int i = 0; i < ncoeff(ORD); ++i) r.c[i] = ds_mul(a.c[i], x0);
    for (int i = ncoeff(ORD); i < jlen(ORD); ++i) r.c[i] = ds_zero();
    constant uint* up = tables + p.up_off;
    constant uint* deg = tables + p.deg_off;
    for (int i = 0; i < ncoeff(ORD); ++i) {
        if (deg[i] < uint(ORD)) {
            uint dst = up[axis * 35 + i];
            r.c[dst] = ds_addp(r.c[dst], a.c[i]);
        }
    }
}

template <int ORD>
inline void j_compose(thread Jt<ORD>& r, thread const Jt<ORD>& self, thread const ds f[5], constant uint* tables,
                      constant BasisParams& p) {
    Jt<ORD> delta = self;
    delta.c[0] = ds_zero();
    ds fact = ds_f(1.0f);
    ds coef[5];
    for (int k = 0; k <= ORD; ++k) {
        if (k > 0) fact = ds_mul(fact, ds_f(float(k)));
        coef[k] = ds_div(f[k], fact);
    }
    j_const<ORD>(r, coef[ORD]);
    for (int k = int(ORD) - 1; k >= 0; --k) {
        Jt<ORD> tmp;
        j_mul<ORD>(tmp, r, delta, tables, p);
        tmp.c[0] = ds_addp(tmp.c[0], coef[k]);
        r = tmp;
    }
}

template <int ORD>
inline void j_recip(thread Jt<ORD>& r, thread const Jt<ORD>& self, constant uint* tables, constant BasisParams& p) {
    ds a = self.c[0];
    ds a2 = ds_sqr(a);
    ds a3 = ds_mul(a2, a);
    ds a4 = ds_sqr(a2);
    ds a5 = ds_mul(a4, a);
    ds f[5];
    f[0] = ds_recip(a);
    f[1] = ds_neg(ds_recip(a2));
    f[2] = ds_mul(ds_f(2.0f), ds_recip(a3));
    f[3] = ds_neg(ds_mul(ds_f(6.0f), ds_recip(a4)));
    f[4] = ds_mul(ds_f(24.0f), ds_recip(a5));
    j_compose<ORD>(r, self, f, tables, p);
}

template <int ORD>
inline void j_sqrt(thread Jt<ORD>& r, thread const Jt<ORD>& self, constant uint* tables, constant BasisParams& p) {
    ds a = self.c[0];
    ds s = ds_sqrt(a);
    ds as_ = ds_mul(a, s);
    ds a2 = ds_sqr(a);
    ds a2s = ds_mul(a2, s);
    ds a3s = ds_mul(ds_mul(a2, a), s);
    ds f[5];
    f[0] = s;
    f[1] = ds_div(ds_f(0.5f), s);
    f[2] = ds_neg(ds_div(ds_f(0.25f), as_));
    f[3] = ds_div(ds_f(0.375f), a2s);
    f[4] = ds_neg(ds_div(ds_f(0.9375f), a3s));
    j_compose<ORD>(r, self, f, tables, p);
}

template <int ORD>
inline void radial_jet(thread Jt<ORD>& radius, ds px, ds py, ds pz, constant uint* tables, constant BasisParams& p) {
    Jt<ORD> x, y, z, x2, y2, z2, r2;
    j_var<ORD>(x, 0, px);
    j_var<ORD>(y, 1, py);
    j_var<ORD>(z, 2, pz);
    j_mul<ORD>(x2, x, x, tables, p);
    j_mul<ORD>(y2, y, y, tables, p);
    j_mul<ORD>(z2, z, z, tables, p);
    j_add3<ORD>(r2, x2, y2, z2);
    j_sqrt<ORD>(radius, r2, tables, p);
}

template <int ORD>
inline void unit_frame(thread Jt<ORD> u[3], thread Jt<ORD>& radius, ds px, ds py, ds pz, constant uint* tables,
                       constant BasisParams& p) {
    Jt<ORD> x, y, z, inv;
    j_var<ORD>(x, 0, px);
    j_var<ORD>(y, 1, py);
    j_var<ORD>(z, 2, pz);
    radial_jet<ORD>(radius, px, py, pz, tables, p);
    j_recip<ORD>(inv, radius, tables, p);
    j_mul<ORD>(u[0], x, inv, tables, p);
    j_mul<ORD>(u[1], y, inv, tables, p);
    j_mul<ORD>(u[2], z, inv, tables, p);
}

inline void load_point(device const float2* points, uint i, thread ds& x, thread ds& y, thread ds& z) {
    uint b = 3u * i;
    x = ds_mk(points[b].x, points[b].y);
    y = ds_mk(points[b + 1].x, points[b + 1].y);
    z = ds_mk(points[b + 2].x, points[b + 2].y);
}

inline void store_ds(device float2* out, uint channel, uint n_points, uint point, ds v) {
    out[channel * n_points + point] = float2(v.hi, v.lo);
}

template <int ORD>
inline void cmul_u(thread Jt<ORD>& cre, thread Jt<ORD>& cim, thread const Jt<ORD> u[3], constant uint* tables,
                   constant BasisParams& p) {
    Jt<ORD> t1, t2, nre, nim;
    j_mul<ORD>(t1, cre, u[0], tables, p);
    j_mul<ORD>(t2, cim, u[1], tables, p);
    j_sub<ORD>(nre, t1, t2);
    j_mul<ORD>(t1, cre, u[1], tables, p);
    j_mul<ORD>(t2, cim, u[0], tables, p);
    j_add<ORD>(nim, t1, t2);
    cre = nre;
    cim = nim;
}

template <int ORD>
inline void legendre_p(thread Jt<ORD>& p, int l, int m, thread const Jt<ORD>& p1, thread const Jt<ORD>& p2,
                       thread const Jt<ORD>& uz, ds pmm, constant uint* tables, constant BasisParams& prm) {
    if (l == m) {
        j_const<ORD>(p, pmm);
        return;
    }
    ds a = coef_a(l, m);
    Jt<ORD> zp;
    j_mul<ORD>(zp, uz, p1, tables, prm);
    if (l == m + 1) {
        j_scale<ORD>(p, zp, a);
        return;
    }
    Jt<ORD> bp, d;
    j_scale<ORD>(bp, p2, coef_b(l, m));
    j_sub<ORD>(d, zp, bp);
    j_scale<ORD>(p, d, a);
}

template <int ORD>
inline void sh_accumulate(thread Jt<ORD>& rr, thread Jt<ORD>* saved, int l_save, int l_top, int l_surf,
                          thread const Jt<ORD> u[3], device const float2* coeff, ds sqrt2, ds pmm0,
                          constant uint* tables, constant BasisParams& prm) {
    j_const<ORD>(rr, ds_zero());
    Jt<ORD> cre, cim;
    j_const<ORD>(cre, ds_f(1.0f));
    j_const<ORD>(cim, ds_zero());
    ds pmm = pmm0;
    for (int m = 0; m <= l_top; ++m) {
        if (m > 0) {
            cmul_u<ORD>(cre, cim, u, tables, prm);
            pmm = ds_mul(pmm, ds_sqrt(ds_div(ds_f(float(2 * m + 1)), ds_f(float(2 * m)))));
        }
        Jt<ORD> p1, p2;
        j_const<ORD>(p1, ds_zero());
        j_const<ORD>(p2, ds_zero());
        for (int l = m; l <= l_top; ++l) {
            Jt<ORD> p, yp, ym, t;
            legendre_p<ORD>(p, l, m, p1, p2, u[2], pmm, tables, prm);
            if (m == 0) {
                yp = p;
            } else {
                j_mul<ORD>(t, p, cre, tables, prm);
                j_scale<ORD>(yp, t, sqrt2);
                j_mul<ORD>(t, p, cim, tables, prm);
                j_scale<ORD>(ym, t, sqrt2);
            }
            if (l <= l_surf) {
                ds cy = coeff_at(coeff, l, m);
                if (cy.hi != 0.0f || cy.lo != 0.0f) j_axpy<ORD>(rr, cy, yp);
                if (m > 0) {
                    ds cz = coeff_at(coeff, l, -m);
                    if (cz.hi != 0.0f || cz.lo != 0.0f) j_axpy<ORD>(rr, cz, ym);
                }
            }
            if (saved != nullptr && l >= 1 && l <= l_save) {
                saved[lm_of(l, m)] = yp;
                if (m > 0) saved[lm_of(l, -m)] = ym;
            }
            p2 = p1;
            p1 = p;
        }
    }
}

template <int ORD>
inline void j_psi(thread Jt<ORD>& s, device const float2* poly, uint len, thread Jt<ORD>* pows) {
    j_const<ORD>(s, ds_zero());
    for (uint j = 0; j < len; ++j) {
        ds a = ds_mk(poly[j].x, poly[j].y);
        if (a.hi != 0.0f || a.lo != 0.0f) j_axpy<ORD>(s, a, pows[j]);
    }
}

template <int ORD>
inline void fill_powers(thread Jt<ORD>* pows, thread const Jt<ORD>& zeta, uint top, constant uint* tables,
                        constant BasisParams& p) {
    j_const<ORD>(pows[0], ds_f(1.0f));
    for (uint n = 1; n <= top; ++n) j_mul<ORD>(pows[n], pows[n - 1], zeta, tables, p);
}

template <int ORD>
inline void poloidal(thread Jt<ORD - 2> out[3], thread const Jt<ORD>& phi, thread ds x0[3], constant uint* tables,
                     constant BasisParams& p) {
    Jt<ORD - 1> g[3];
    for (int a = 0; a < 3; ++a) j_deriv<ORD>(g[a], phi, a, tables, p);
    Jt<ORD - 1> e, mx;
    j_trunc<ORD, ORD - 1>(e, phi);
    for (int a = 0; a < 3; ++a) {
        j_mul_x<ORD - 1>(mx, g[a], a, x0[a], tables, p);
        j_add_into<ORD - 1>(e, mx);
    }
    Jt<ORD - 2> d0, d1, d2, lap;
    j_deriv<ORD - 1>(d0, g[0], 0, tables, p);
    j_deriv<ORD - 1>(d1, g[1], 1, tables, p);
    j_deriv<ORD - 1>(d2, g[2], 2, tables, p);
    j_add3<ORD - 2>(lap, d0, d1, d2);
    for (int a = 0; a < 3; ++a) {
        Jt<ORD - 2> de, lx;
        j_deriv<ORD - 1>(de, e, a, tables, p);
        j_mul_x<ORD - 2>(lx, lap, a, x0[a], tables, p);
        j_sub<ORD - 2>(out[a], de, lx);
    }
}

template <int ORD>
inline void toroidal(thread Jt<ORD - 2> out[3], thread const Jt<ORD>& psi, thread ds x0[3], constant uint* tables,
                     constant BasisParams& p) {
    Jt<ORD - 1> g[3];
    for (int a = 0; a < 3; ++a) j_deriv<ORD>(g[a], psi, a, tables, p);
    Jt<ORD - 1> a, b, cross;
    j_mul_x<ORD - 1>(a, g[1], 2, x0[2], tables, p);
    j_mul_x<ORD - 1>(b, g[2], 1, x0[1], tables, p);
    j_sub<ORD - 1>(cross, a, b);
    j_trunc<ORD - 1, ORD - 2>(out[0], cross);
    j_mul_x<ORD - 1>(a, g[2], 0, x0[0], tables, p);
    j_mul_x<ORD - 1>(b, g[0], 2, x0[2], tables, p);
    j_sub<ORD - 1>(cross, a, b);
    j_trunc<ORD - 1, ORD - 2>(out[1], cross);
    j_mul_x<ORD - 1>(a, g[0], 1, x0[1], tables, p);
    j_mul_x<ORD - 1>(b, g[1], 0, x0[0], tables, p);
    j_sub<ORD - 1>(cross, a, b);
    j_trunc<ORD - 1, ORD - 2>(out[2], cross);
}

template <int FORD>
inline void store_vd(thread const Jt<FORD> f[3], uint mode, uint point, device float2* out, constant BasisParams& p,
                     ds sqrt2) {
    uint n = p.n_points;
    uint local = 0;
    if (p.want_v) {
        for (int a = 0; a < 3; ++a) store_ds(out, mode * p.n_comp + local + uint(a), n, point, f[a].c[0]);
        local += 3u;
    }
    if (p.want_d) {
        ds g00 = f[0].c[1], g01 = f[0].c[2], g02 = f[0].c[3];
        ds g10 = f[1].c[1], g11 = f[1].c[2], g12 = f[1].c[3];
        ds g20 = f[2].c[1], g21 = f[2].c[2], g22 = f[2].c[3];
        ds d0 = ds_scale(ds_addp(g00, g00), 0.5f);
        ds d1 = ds_scale(ds_addp(g11, g11), 0.5f);
        ds d2 = ds_scale(ds_addp(g22, g22), 0.5f);
        ds d3 = ds_mul(sqrt2, ds_scale(ds_addp(g01, g10), 0.5f));
        ds d4 = ds_mul(sqrt2, ds_scale(ds_addp(g02, g20), 0.5f));
        ds d5 = ds_mul(sqrt2, ds_scale(ds_addp(g12, g21), 0.5f));
        ds d[6] = {d0, d1, d2, d3, d4, d5};
        for (uint a = 0; a < 6u; ++a) store_ds(out, mode * p.n_comp + local + a, n, point, d[a]);
    }
}

inline uint vd_channels(constant BasisParams& p) { return (p.want_v ? 3u : 0u) + (p.want_d ? 6u : 0u); }

template <bool HIGH, int FORD>
struct HighStore {
    static void apply(thread const Jt<FORD>*, uint, uint, device float2*, constant BasisParams&, constant uint*, ds) {}
};

template <int FORD>
struct HighStore<true, FORD> {
    static void apply(thread const Jt<FORD> f[3], uint mode, uint point, device float2* out, constant BasisParams& p,
                      constant uint* tables, ds sqrt2) {
        if (p.want_lap == 0u && p.want_gd == 0u) return;
        constant uint* pair = tables + p.pair_off;
        ds h[3][3][3];
        for (int comp = 0; comp < 3; ++comp) {
            for (int b = 0; b < 3; ++b) {
                for (int c = 0; c < 3; ++c) {
                    ds coef = f[comp].c[pair[b * 3 + c]];
                    h[comp][b][c] = (b == c) ? ds_scale(coef, 2.0f) : coef;
                }
            }
        }
        uint n = p.n_points;
        uint local = vd_channels(p);
        if (p.want_lap) {
            for (int a = 0; a < 3; ++a) {
                ds lap = ds_addp(ds_addp(h[a][0][0], h[a][1][1]), h[a][2][2]);
                store_ds(out, mode * p.n_comp + local + uint(a), n, point, lap);
            }
            local += 3u;
        }
        if (p.want_gd) {
            for (int k = 0; k < 3; ++k) {
                ds g0 = ds_scale(ds_addp(h[0][0][k], h[0][0][k]), 0.5f);
                ds g1 = ds_scale(ds_addp(h[1][1][k], h[1][1][k]), 0.5f);
                ds g2 = ds_scale(ds_addp(h[2][2][k], h[2][2][k]), 0.5f);
                ds g3 = ds_mul(sqrt2, ds_scale(ds_addp(h[0][1][k], h[1][0][k]), 0.5f));
                ds g4 = ds_mul(sqrt2, ds_scale(ds_addp(h[0][2][k], h[2][0][k]), 0.5f));
                ds g5 = ds_mul(sqrt2, ds_scale(ds_addp(h[1][2][k], h[2][1][k]), 0.5f));
                ds packed[6] = {g0, g1, g2, g3, g4, g5};
                for (uint c = 0; c < 6u; ++c) store_ds(out, mode * p.n_comp + local + c, n, point, packed[c]);
                local += 6u;
            }
        }
    }
};

template <int ORD>
inline void emit_mode(thread const Jt<ORD>& radial, thread const Jt<ORD>& ylm, uint tor, thread ds x0[3], int l, int m,
                      uint k, uint point, device float2* out, constant uint* tables, constant BasisParams& p, ds sqrt2) {
    Jt<ORD> phi;
    j_mul<ORD>(phi, radial, ylm, tables, p);
    Jt<ORD - 2> field[3];
    if (tor) {
        toroidal<ORD>(field, phi, x0, tables, p);
    } else {
        poloidal<ORD>(field, phi, x0, tables, p);
    }
    uint mode = vel_dof(l, m, tor, k, p.k_count);
    store_vd<ORD - 2>(field, mode, point, out, p, sqrt2);
    HighStore<(ORD >= 4), ORD - 2>::apply(field, mode, point, out, p, tables, sqrt2);
}

template <int ORD, int LMAX>
void eval_cached(uint ray, device const float2* points, constant BasisParams& prm, device const float2* coeff,
                 device const float2* polys, constant uint* tables, device float2* out) {
    if (ray >= prm.n_rays) return;
    ds sqrt2 = ds_mk(prm.sqrt2_hi, prm.sqrt2_lo);
    ds pmm0 = ds_mk(prm.pmm_hi, prm.pmm_lo);
    uint p0 = ray * prm.n_per;
    ds x0, y0, z0;
    load_point(points, p0, x0, y0, z0);
    Jt<ORD> u[3], radius0;
    unit_frame<ORD>(u, radius0, x0, y0, z0, tables, prm);
    int l_mode = int(prm.l_mode);
    int l_surf = int(prm.l_surf);
    int l_top = l_mode > l_surf ? l_mode : l_surf;
    Jt<ORD> saved[(LMAX + 1) * (LMAX + 1)];
    Jt<ORD> rr;
    sh_accumulate<ORD>(rr, saved, l_mode, l_top, l_surf, u, coeff, sqrt2, pmm0, tables, prm);
    ds rho0 = radius0.c[0];
    constant uint* deg = tables + prm.deg_off;

    for (uint s = 0; s < prm.n_per; ++s) {
        ds px, py, pz;
        load_point(points, p0 + s, px, py, pz);
        ds xyz[3] = {px, py, pz};
        Jt<ORD> rs = radius0;
        if (s != 0u) radial_jet<ORD>(rs, px, py, pz, tables, prm);
        ds rho = rs.c[0];
        ds factor[5];
        factor[0] = ds_f(1.0f);
        if (s != 0u) {
            ds ratio = ds_div(rho0, rho);
            for (int d = 1; d <= ORD; ++d) factor[d] = ds_mul(factor[d - 1], ratio);
        }
        Jt<ORD> rad_s = rr;
        if (s != 0u) {
            for (int i = 0; i < ncoeff(ORD); ++i) rad_s.c[i] = ds_mul(rr.c[i], factor[deg[i]]);
        }
        Jt<ORD> inv, zeta, r2;
        j_recip<ORD>(inv, rad_s, tables, prm);
        j_mul<ORD>(zeta, rs, inv, tables, prm);
        j_mul<ORD>(r2, rad_s, rad_s, tables, prm);
        Jt<ORD> pows[LMAX + 5];
        fill_powers<ORD>(pows, zeta, prm.top_pow, tables, prm);
        Jt<ORD> alk[LMAX * 4];
        Jt<ORD> blk[LMAX * 4];
        for (int l = 1; l <= l_mode; ++l) {
            for (uint k = 0; k < prm.k_count; ++k) {
                Jt<ORD> psi;
                device const float2* poly = polys + (uint(l) * prm.k_count + k) * prm.poly_stride;
                j_psi<ORD>(psi, poly, uint(l) + k + 1u, pows);
                uint slot = uint(l - 1) * prm.k_count + k;
                j_mul<ORD>(alk[slot], r2, psi, tables, prm);
                j_mul<ORD>(blk[slot], rad_s, psi, tables, prm);
            }
        }
        uint point = p0 + s;
        for (int l = 1; l <= l_mode; ++l) {
            for (int m = -l; m <= l; ++m) {
                Jt<ORD> ylm;
                if (s == 0u) {
                    ylm = saved[lm_of(l, m)];
                } else {
                    Jt<ORD> src = saved[lm_of(l, m)];
                    for (int i = 0; i < ncoeff(ORD); ++i) ylm.c[i] = ds_mul(src.c[i], factor[deg[i]]);
                    for (int i = ncoeff(ORD); i < jlen(ORD); ++i) ylm.c[i] = ds_zero();
                }
                for (uint k = 0; k < prm.k_count; ++k) {
                    uint slot = uint(l - 1) * prm.k_count + k;
                    emit_mode<ORD>(alk[slot], ylm, 0u, xyz, l, m, k, point, out, tables, prm, sqrt2);
                    emit_mode<ORD>(blk[slot], ylm, 1u, xyz, l, m, k, point, out, tables, prm, sqrt2);
                }
            }
        }
    }
}

template <int ORD>
void eval_point(uint gid, device const float2* points, constant BasisParams& prm, device const float2* coeff,
                device const float2* polys, constant uint* tables, device float2* out) {
    if (gid >= prm.n_points) return;
    ds sqrt2 = ds_mk(prm.sqrt2_hi, prm.sqrt2_lo);
    ds pmm0 = ds_mk(prm.pmm_hi, prm.pmm_lo);
    ds px, py, pz;
    load_point(points, gid, px, py, pz);
    ds xyz[3] = {px, py, pz};
    Jt<ORD> u[3], radius;
    unit_frame<ORD>(u, radius, px, py, pz, tables, prm);
    int l_mode = int(prm.l_mode);
    int l_surf = int(prm.l_surf);
    int l_top = l_mode > l_surf ? l_mode : l_surf;
    Jt<ORD> rr;
    sh_accumulate<ORD>(rr, nullptr, 0, l_top, l_surf, u, coeff, sqrt2, pmm0, tables, prm);
    Jt<ORD> inv, zeta, r2;
    j_recip<ORD>(inv, rr, tables, prm);
    j_mul<ORD>(zeta, radius, inv, tables, prm);
    j_mul<ORD>(r2, rr, rr, tables, prm);
    Jt<ORD> pows[13];
    fill_powers<ORD>(pows, zeta, prm.top_pow, tables, prm);

    Jt<ORD> cre, cim;
    j_const<ORD>(cre, ds_f(1.0f));
    j_const<ORD>(cim, ds_zero());
    ds pmm = pmm0;
    for (int m = 0; m <= l_mode; ++m) {
        if (m > 0) {
            cmul_u<ORD>(cre, cim, u, tables, prm);
            pmm = ds_mul(pmm, ds_sqrt(ds_div(ds_f(float(2 * m + 1)), ds_f(float(2 * m)))));
        }
        Jt<ORD> p1, p2;
        j_const<ORD>(p1, ds_zero());
        j_const<ORD>(p2, ds_zero());
        for (int l = m; l <= l_mode; ++l) {
            Jt<ORD> p, yp, ym, t;
            legendre_p<ORD>(p, l, m, p1, p2, u[2], pmm, tables, prm);
            if (m == 0) {
                yp = p;
            } else {
                j_mul<ORD>(t, p, cre, tables, prm);
                j_scale<ORD>(yp, t, sqrt2);
                j_mul<ORD>(t, p, cim, tables, prm);
                j_scale<ORD>(ym, t, sqrt2);
            }
            if (l >= 1) {
                for (uint k = 0; k < prm.k_count; ++k) {
                    Jt<ORD> psi, a, b;
                    device const float2* poly = polys + (uint(l) * prm.k_count + k) * prm.poly_stride;
                    j_psi<ORD>(psi, poly, uint(l) + k + 1u, pows);
                    j_mul<ORD>(a, r2, psi, tables, prm);
                    j_mul<ORD>(b, rr, psi, tables, prm);
                    emit_mode<ORD>(a, yp, 0u, xyz, l, m, k, gid, out, tables, prm, sqrt2);
                    emit_mode<ORD>(b, yp, 1u, xyz, l, m, k, gid, out, tables, prm, sqrt2);
                    if (m > 0) {
                        emit_mode<ORD>(a, ym, 0u, xyz, l, -m, k, gid, out, tables, prm, sqrt2);
                        emit_mode<ORD>(b, ym, 1u, xyz, l, -m, k, gid, out, tables, prm, sqrt2);
                    }
                }
            }
            p2 = p1;
            p1 = p;
        }
    }
}

template <int LMAX>
void eval_pres0(uint gid, device const float2* points, constant BasisParams& prm, device const float2* coeff,
                device const float2* polys, device float2* out) {
    if (gid >= prm.n_points) return;
    ds sqrt2 = ds_mk(prm.sqrt2_hi, prm.sqrt2_lo);
    ds pmm = ds_mk(prm.pmm_hi, prm.pmm_lo);
    ds x, y, z;
    load_point(points, gid, x, y, z);
    ds rho2 = ds_addp(ds_addp(ds_sqr(x), ds_sqr(y)), ds_sqr(z));
    ds rho = ds_sqrt(rho2);
    ds inv = ds_recip(rho);
    ds ux = ds_mul(x, inv);
    ds uy = ds_mul(y, inv);
    ds uz = ds_mul(z, inv);
    int l_mode = int(prm.l_mode);
    int l_surf = int(prm.l_surf);
    int l_top = l_mode > l_surf ? l_mode : l_surf;
    ds saved[(LMAX + 1) * (LMAX + 1)];
    ds rr = ds_zero();
    ds cre = ds_f(1.0f);
    ds cim = ds_zero();
    for (int m = 0; m <= l_top; ++m) {
        if (m > 0) {
            ds nre = ds_subp(ds_mul(cre, ux), ds_mul(cim, uy));
            ds nim = ds_addp(ds_mul(cre, uy), ds_mul(cim, ux));
            cre = nre;
            cim = nim;
            pmm = ds_mul(pmm, ds_sqrt(ds_div(ds_f(float(2 * m + 1)), ds_f(float(2 * m)))));
        }
        ds p1 = ds_zero();
        ds p2 = ds_zero();
        for (int l = m; l <= l_top; ++l) {
            ds assoc;
            if (l == m) {
                assoc = pmm;
            } else if (l == m + 1) {
                assoc = ds_mul(coef_a(l, m), ds_mul(uz, p1));
            } else {
                assoc = ds_mul(coef_a(l, m), ds_subp(ds_mul(uz, p1), ds_mul(coef_b(l, m), p2)));
            }
            ds yp = assoc;
            ds ym = ds_zero();
            if (m > 0) {
                yp = ds_mul(ds_mul(assoc, cre), sqrt2);
                ym = ds_mul(ds_mul(assoc, cim), sqrt2);
            }
            if (l <= l_surf) {
                rr = ds_addp(rr, ds_mul(coeff_at(coeff, l, m), yp));
                if (m > 0) rr = ds_addp(rr, ds_mul(coeff_at(coeff, l, -m), ym));
            }
            if (l <= l_mode && l <= LMAX) {
                saved[lm_of(l, m)] = yp;
                if (m > 0) saved[lm_of(l, -m)] = ym;
            }
            p2 = p1;
            p1 = assoc;
        }
    }
    ds zeta = ds_mul(rho, ds_recip(rr));
    ds one = ds_subp(ds_f(1.0f), zeta);
    ds zp[33];
    zp[0] = ds_f(1.0f);
    for (uint n = 1; n <= prm.top_pow && n < 33u; ++n) zp[n] = ds_mul(zp[n - 1], zeta);
    uint n = prm.n_points;
    uint np = prm.n_out;
    for (int l = 0; l <= l_mode; ++l) {
        for (int m = -l; m <= l; ++m) {
            uint lm = lm_of(l, m);
            ds ylm = saved[lm];
            store_ds(out, np + lm, n, gid, ds_mul(zp[uint(l)], ylm));
            for (uint k = 0; k < prm.k_count; ++k) {
                device const float2* poly = polys + (uint(l) * prm.k_count + k) * prm.poly_stride;
                ds psi = ds_zero();
                uint len = uint(l) + k + 1u;
                for (uint j = 0; j < len; ++j) {
                    ds a = ds_mk(poly[j].x, poly[j].y);
                    if (a.hi != 0.0f || a.lo != 0.0f) psi = ds_addp(psi, ds_mul(a, zp[j]));
                }
                ds phi = ds_mul(ds_mul(one, psi), ylm);
                store_ds(out, lm * prm.k_count + k, n, gid, phi);
            }
        }
    }
}

void eval_pres1(uint gid, device const float2* points, constant BasisParams& prm, device const float2* coeff,
                device const float2* polys, constant uint* tables, device float2* out) {
    if (gid >= prm.n_points) return;
    ds sqrt2 = ds_mk(prm.sqrt2_hi, prm.sqrt2_lo);
    ds pmm0 = ds_mk(prm.pmm_hi, prm.pmm_lo);
    ds px, py, pz;
    load_point(points, gid, px, py, pz);
    Jt<1> u[3], radius, rr;
    unit_frame<1>(u, radius, px, py, pz, tables, prm);
    int l_mode = int(prm.l_mode);
    int l_surf = int(prm.l_surf);
    int l_top = l_mode > l_surf ? l_mode : l_surf;
    sh_accumulate<1>(rr, nullptr, 0, l_top, l_surf, u, coeff, sqrt2, pmm0, tables, prm);
    Jt<1> inv, zeta, one;
    j_recip<1>(inv, rr, tables, prm);
    j_mul<1>(zeta, radius, inv, tables, prm);
    j_scale<1>(one, zeta, ds_f(-1.0f));
    one.c[0] = ds_addp(one.c[0], ds_f(1.0f));
    Jt<1> pows[33];
    fill_powers<1>(pows, zeta, prm.top_pow, tables, prm);
    Jt<1> cre, cim;
    j_const<1>(cre, ds_f(1.0f));
    j_const<1>(cim, ds_zero());
    ds pmm = pmm0;
    uint n = prm.n_points;
    uint np = prm.n_out;
    for (int m = 0; m <= l_mode; ++m) {
        if (m > 0) {
            cmul_u<1>(cre, cim, u, tables, prm);
            pmm = ds_mul(pmm, ds_sqrt(ds_div(ds_f(float(2 * m + 1)), ds_f(float(2 * m)))));
        }
        Jt<1> p1, p2;
        j_const<1>(p1, ds_zero());
        j_const<1>(p2, ds_zero());
        for (int l = m; l <= l_mode; ++l) {
            Jt<1> p, yp, ym, t;
            legendre_p<1>(p, l, m, p1, p2, u[2], pmm, tables, prm);
            if (m == 0) {
                yp = p;
            } else {
                j_mul<1>(t, p, cre, tables, prm);
                j_scale<1>(yp, t, sqrt2);
                j_mul<1>(t, p, cim, tables, prm);
                j_scale<1>(ym, t, sqrt2);
            }
            for (int pass = 0; pass < (m == 0 ? 1 : 2); ++pass) {
                int mm = pass == 0 ? m : -m;
                Jt<1> ylm = pass == 0 ? yp : ym;
                Jt<1> lift;
                j_mul<1>(lift, pows[uint(l)], ylm, tables, prm);
                uint lm = lm_of(l, mm);
                uint lift_ch = (np + lm) * 4u;
                store_ds(out, lift_ch, n, gid, lift.c[0]);
                store_ds(out, lift_ch + 1u, n, gid, lift.c[1]);
                store_ds(out, lift_ch + 2u, n, gid, lift.c[2]);
                store_ds(out, lift_ch + 3u, n, gid, lift.c[3]);
                for (uint k = 0; k < prm.k_count; ++k) {
                    Jt<1> psi, mid, phi;
                    device const float2* poly = polys + (uint(l) * prm.k_count + k) * prm.poly_stride;
                    j_psi<1>(psi, poly, uint(l) + k + 1u, pows);
                    j_mul<1>(mid, one, psi, tables, prm);
                    j_mul<1>(phi, mid, ylm, tables, prm);
                    uint ch = (lm * prm.k_count + k) * 4u;
                    store_ds(out, ch, n, gid, phi.c[0]);
                    store_ds(out, ch + 1u, n, gid, phi.c[1]);
                    store_ds(out, ch + 2u, n, gid, phi.c[2]);
                    store_ds(out, ch + 3u, n, gid, phi.c[3]);
                }
            }
            p2 = p1;
            p1 = p;
        }
    }
}

void eval_normal(uint gid, device const float2* points, constant BasisParams& prm, device const float2* coeff,
                 constant uint* tables, device float2* out) {
    if (gid >= prm.n_points) return;
    ds sqrt2 = ds_mk(prm.sqrt2_hi, prm.sqrt2_lo);
    ds pmm0 = ds_mk(prm.pmm_hi, prm.pmm_lo);
    ds px, py, pz;
    load_point(points, gid, px, py, pz);
    Jt<1> u[3], radius, rr;
    unit_frame<1>(u, radius, px, py, pz, tables, prm);
    int l_surf = int(prm.l_surf);
    sh_accumulate<1>(rr, nullptr, 0, l_surf, l_surf, u, coeff, sqrt2, pmm0, tables, prm);
    store_ds(out, 0u, prm.n_points, gid, rr.c[0]);
    store_ds(out, 1u, prm.n_points, gid, rr.c[1]);
    store_ds(out, 2u, prm.n_points, gid, rr.c[2]);
    store_ds(out, 3u, prm.n_points, gid, rr.c[3]);
}

kernel void basis_vel_o3_l2(uint gid [[thread_position_in_grid]], device const float2* points [[buffer(0)]],
                            constant BasisParams& params [[buffer(1)]], device const float2* coeff [[buffer(2)]],
                            device const float2* polys [[buffer(3)]], constant uint* tables [[buffer(4)]],
                            device float2* out [[buffer(5)]]) {
    eval_cached<3, 2>(gid, points, params, coeff, polys, tables, out);
}

kernel void basis_vel_o3_l4(uint gid [[thread_position_in_grid]], device const float2* points [[buffer(0)]],
                            constant BasisParams& params [[buffer(1)]], device const float2* coeff [[buffer(2)]],
                            device const float2* polys [[buffer(3)]], constant uint* tables [[buffer(4)]],
                            device float2* out [[buffer(5)]]) {
    eval_cached<3, 4>(gid, points, params, coeff, polys, tables, out);
}

kernel void basis_vel_o2(uint gid [[thread_position_in_grid]], device const float2* points [[buffer(0)]],
                         constant BasisParams& params [[buffer(1)]], device const float2* coeff [[buffer(2)]],
                         device const float2* polys [[buffer(3)]], constant uint* tables [[buffer(4)]],
                         device float2* out [[buffer(5)]]) {
    eval_point<2>(gid, points, params, coeff, polys, tables, out);
}

kernel void basis_vel_o3(uint gid [[thread_position_in_grid]], device const float2* points [[buffer(0)]],
                         constant BasisParams& params [[buffer(1)]], device const float2* coeff [[buffer(2)]],
                         device const float2* polys [[buffer(3)]], constant uint* tables [[buffer(4)]],
                         device float2* out [[buffer(5)]]) {
    eval_point<3>(gid, points, params, coeff, polys, tables, out);
}

kernel void basis_vel_o4(uint gid [[thread_position_in_grid]], device const float2* points [[buffer(0)]],
                         constant BasisParams& params [[buffer(1)]], device const float2* coeff [[buffer(2)]],
                         device const float2* polys [[buffer(3)]], constant uint* tables [[buffer(4)]],
                         device float2* out [[buffer(5)]]) {
    eval_point<4>(gid, points, params, coeff, polys, tables, out);
}

kernel void basis_pres0_l4(uint gid [[thread_position_in_grid]], device const float2* points [[buffer(0)]],
                           constant BasisParams& params [[buffer(1)]], device const float2* coeff [[buffer(2)]],
                           device const float2* polys [[buffer(3)]], device float2* out [[buffer(4)]]) {
    eval_pres0<4>(gid, points, params, coeff, polys, out);
}

kernel void basis_pres0_l24(uint gid [[thread_position_in_grid]], device const float2* points [[buffer(0)]],
                            constant BasisParams& params [[buffer(1)]], device const float2* coeff [[buffer(2)]],
                            device const float2* polys [[buffer(3)]], device float2* out [[buffer(4)]]) {
    eval_pres0<24>(gid, points, params, coeff, polys, out);
}

kernel void basis_pres1(uint gid [[thread_position_in_grid]], device const float2* points [[buffer(0)]],
                        constant BasisParams& params [[buffer(1)]], device const float2* coeff [[buffer(2)]],
                        device const float2* polys [[buffer(3)]], constant uint* tables [[buffer(4)]],
                        device float2* out [[buffer(5)]]) {
    eval_pres1(gid, points, params, coeff, polys, tables, out);
}

kernel void basis_normal(uint gid [[thread_position_in_grid]], device const float2* points [[buffer(0)]],
                         constant BasisParams& params [[buffer(1)]], device const float2* coeff [[buffer(2)]],
                         constant uint* tables [[buffer(3)]], device float2* out [[buffer(4)]]) {
    eval_normal(gid, points, params, coeff, tables, out);
}
