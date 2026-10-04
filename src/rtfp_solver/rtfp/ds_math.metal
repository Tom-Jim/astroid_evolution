#pragma clang fp contract(off)
#ifndef RTFP_DS_MATH_METAL
#define RTFP_DS_MATH_METAL

#include <metal_stdlib>
using namespace metal;

// Double-float: a value is hi + lo, with |lo| at most half an ulp of hi.
// Apple GPUs have no f64 unit; the radial integrals are accumulated here so a
// long sum of Cauchy kernels stays near 1e-14 relative instead of f32's 1e-7.
struct ds {
    float hi;
    float lo;
};

inline ds ds_new(float hi, float lo) {
    float s = hi + lo;
    return ds{s, lo - (s - hi)};
}

inline ds ds_zero() { return ds{0.0f, 0.0f}; }

inline ds ds_from_float(float h) { return ds{h, 0.0f}; }

inline ds ds_neg(ds a) { return ds{-a.hi, -a.lo}; }

// Knuth TwoSum.
inline ds ds_add(ds a, ds b) {
    float s = a.hi + b.hi;
    float v = s - a.hi;
    float e = (a.hi - (s - v)) + (b.hi - v) + a.lo + b.lo;
    float h = s + e;
    return ds{h, e - (h - s)};
}

inline ds ds_sub(ds a, ds b) { return ds_add(a, ds_neg(b)); }

// Dekker TwoProd. The residual of hi*hi is an exact hardware fma.
inline ds ds_mul(ds a, ds b) {
    float p = a.hi * b.hi;
    float e = metal::precise::fma(a.hi, b.hi, -p);
    e = metal::precise::fma(a.hi, b.lo, e);
    e = metal::precise::fma(a.lo, b.hi, e);
    e = metal::precise::fma(a.lo, b.lo, e);
    float h = p + e;
    return ds{h, e - (h - p)};
}

inline ds ds_abs(ds a) { return (a.hi < 0.0f || (a.hi == 0.0f && a.lo < 0.0f)) ? ds_neg(a) : a; }

// f32 splits of the constants the radial logs and the contact term need.
// hi is the f32 rounding; lo is the exact residual rounded back to f32.
inline ds ds_pi() { return ds{3.1415927410e+00f, -8.7422776573e-08f}; }
inline ds ds_half_pi() { return ds{1.5707963705e+00f, -4.3711388287e-08f}; }
inline ds ds_quarter_pi() { return ds{7.8539818525e-01f, -2.1855694143e-08f}; }
inline ds ds_ln2() { return ds{6.9314718246e-01f, -1.9046542121e-09f}; }
inline ds ds_inv_ln2() { return ds{1.4426950216e+00f, 1.9259630335e-08f}; }

inline ds ds_div(ds a, ds b) {
    float inv = 1.0f / b.hi;
    float q0 = a.hi * inv;
    ds r = ds_sub(a, ds_mul(ds_from_float(q0), b));
    float q1 = r.hi * inv;
    ds q = ds_add(ds_from_float(q0), ds_from_float(q1));
    r = ds_sub(a, ds_mul(q, b));
    float q2 = (r.hi + r.lo) * inv;
    return ds_add(q, ds_from_float(q2));
}

inline ds ds_sqrt(ds a) {
    if (!(a.hi > 0.0f)) {
        return ds_zero();
    }
    ds s = ds_from_float(precise::sqrt(a.hi));
    s = ds_mul(ds_add(s, ds_div(a, s)), ds_from_float(0.5f));
    s = ds_mul(ds_add(s, ds_div(a, s)), ds_from_float(0.5f));
    return s;
}

// exp via Cody–Waite reduction to [-ln2/2, ln2/2] and a Taylor polynomial.
inline ds ds_exp(ds x) {
    ds scaled = ds_mul(x, ds_inv_ln2());
    float nf = rint(scaled.hi + scaled.lo);
    ds r = ds_sub(x, ds_mul(ds_from_float(nf), ds_ln2()));
    ds sum = ds_from_float(1.0f);
    ds term = ds_from_float(1.0f);
    for (int k = 1; k <= 18; ++k) {
        term = ds_div(ds_mul(term, r), ds_from_float(float(k)));
        sum = ds_add(sum, term);
    }
    int n = int(nf);
    sum.hi = ldexp(sum.hi, n);
    sum.lo = ldexp(sum.lo, n);
    return sum;
}

// ln via frexp + two Halley steps. One step cubes a 1-ulp seed, which is
// already past the double-float ulp; the second step covers a bad seed.
inline ds ds_ln(ds a) {
    if (!(a.hi > 0.0f)) {
        return ds{-INFINITY, 0.0f};
    }
    int e = 0;
    float m = frexp(a.hi, e);
    ds md = ds_add(ds_from_float(m), ds_from_float(a.lo * ldexp(1.0f, -e)));
    if (md.hi < 0.7071067811865476f) {
        md = ds_add(md, md);
        e -= 1;
    }
    ds y = ds_from_float(precise::log(md.hi));
    for (int i = 0; i < 2; ++i) {
        ds ey = ds_exp(y);
        ds num = ds_sub(ey, md);
        ds den = ds_add(ey, md);
        y = ds_sub(y, ds_mul(ds_from_float(2.0f), ds_div(num, den)));
    }
    return ds_add(y, ds_mul(ds_from_float(float(e)), ds_ln2()));
}

inline ds ds_ln1p(ds z) {
    if (fabs(z.hi) < 0.125f) {
        ds sum = ds_zero();
        ds term = z;
        for (int k = 1; k <= 22; ++k) {
            ds piece = ds_div(term, ds_from_float(float(k)));
            sum = (k & 1) ? ds_add(sum, piece) : ds_sub(sum, piece);
            term = ds_mul(term, z);
        }
        return sum;
    }
    return ds_ln(ds_add(ds_from_float(1.0f), z));
}

inline ds ds_expm1(ds z) {
    if (fabs(z.hi) < 0.125f) {
        ds sum = ds_zero();
        ds term = ds_from_float(1.0f);
        for (int k = 1; k <= 22; ++k) {
            term = ds_div(ds_mul(term, z), ds_from_float(float(k)));
            sum = ds_add(sum, term);
        }
        return sum;
    }
    return ds_sub(ds_exp(z), ds_from_float(1.0f));
}

inline ds ds_atan_series(ds u) {
    ds u2 = ds_mul(u, u);
    ds term = u;
    ds sum = u;
    for (int k = 1; k <= 28; ++k) {
        term = ds_mul(term, u2);
        ds piece = ds_div(term, ds_from_float(float(2 * k + 1)));
        sum = (k & 1) ? ds_sub(sum, piece) : ds_add(sum, piece);
        if (piece.hi == 0.0f && piece.lo == 0.0f) {
            break;
        }
    }
    return sum;
}

inline ds ds_atan_unit(ds z) {
    if (z.hi > 0.41421356237f) {
        ds u = ds_div(ds_sub(z, ds_from_float(1.0f)), ds_add(z, ds_from_float(1.0f)));
        return ds_add(ds_quarter_pi(), ds_atan_series(u));
    }
    return ds_atan_series(z);
}

inline ds ds_atan_pos(ds z) {
    if (z.hi > 1.0f) {
        return ds_sub(ds_half_pi(), ds_atan_unit(ds_div(ds_from_float(1.0f), z)));
    }
    return ds_atan_unit(z);
}

inline ds ds_atan(ds z) {
    if (z.hi < 0.0f || (z.hi == 0.0f && z.lo < 0.0f)) {
        return ds_neg(ds_atan_pos(ds_neg(z)));
    }
    return ds_atan_pos(z);
}

// Same quadrant convention as f64 atan2, which the α = 1 chord uses for
// atan(x1) − atan(x0) when the two samples sit on opposite sides of the pole.
inline ds ds_atan2(ds y, ds x) {
    if (x.hi == 0.0f && y.hi == 0.0f) {
        return ds_zero();
    }
    if (x.hi > 0.0f) {
        return ds_atan(ds_div(y, x));
    }
    if (x.hi < 0.0f) {
        ds a = ds_atan(ds_div(y, x));
        return (y.hi >= 0.0f) ? ds_add(a, ds_pi()) : ds_sub(a, ds_pi());
    }
    return (y.hi >= 0.0f) ? ds_half_pi() : ds_neg(ds_half_pi());
}

inline ds ds_sinh(ds x) {
    ds e = ds_exp(x);
    ds em = ds_exp(ds_neg(x));
    return ds_mul(ds_sub(e, em), ds_from_float(0.5f));
}

inline ds ds_cosh(ds x) {
    ds e = ds_exp(x);
    ds em = ds_exp(ds_neg(x));
    return ds_mul(ds_add(e, em), ds_from_float(0.5f));
}

inline ds ds_asinh(ds z) {
    if (fabs(z.hi) > 1.0e4f) {
        ds a = ds_ln(ds_mul(ds_from_float(2.0f), ds_abs(z)));
        return (z.hi < 0.0f) ? ds_neg(a) : a;
    }
    ds s = ds_sqrt(ds_add(ds_mul(z, z), ds_from_float(1.0f)));
    return ds_ln(ds_add(z, s));
}

inline ds ds_pow(ds a, float p) {
    if (!(a.hi > 0.0f)) {
        return ds_zero();
    }
    return ds_exp(ds_mul(ds_ln(a), ds_from_float(p)));
}

#endif
