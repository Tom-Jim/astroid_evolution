#pragma clang fp contract(off)
#ifndef RHEO_DS_METAL
#define RHEO_DS_METAL

#include <metal_stdlib>
using namespace metal;

// Double-single arithmetic: a value is hi + lo with |lo| <= ulp(hi)/2.
//
// Apple GPUs have no f64 unit. Everything here is built from the native f32
// add, mul and fma. The library is compiled with MTLMathMode::Safe and
// `fp contract(off)`, so the compiler keeps every error-free transformation
// exactly as written.
//
// Cost model (f32 flops): add 11 (sloppy) / 22 (IEEE), mul 7, fma-accumulate 12,
// div 13, sqrt 9. The jet code stays in `dsacc` accumulators and normalizes
// once per output coefficient. The product lo*lo is dropped everywhere: it is
// 2^-48 relative.

struct ds {
    float hi;
    float lo;
};

inline ds ds_mk(float h, float l) { return ds{h, l}; }
inline ds ds_f(float h) { return ds{h, 0.0f}; }
inline ds ds_zero() { return ds{0.0f, 0.0f}; }
inline ds ds_neg(ds a) { return ds{-a.hi, -a.lo}; }
inline ds ds_abs(ds a) { return (a.hi < 0.0f || (a.hi == 0.0f && a.lo < 0.0f)) ? ds_neg(a) : a; }
inline float ds_float(ds a) { return a.hi + a.lo; }

// Knuth TwoSum: six flops, no ordering assumption.
inline float2 two_sum(float a, float b) {
    float s = a + b;
    float v = s - a;
    return float2(s, (a - (s - v)) + (b - v));
}

// Dekker Fast2Sum: three flops, requires |a| >= |b| or a == 0.
inline float2 quick_two_sum(float a, float b) {
    float s = a + b;
    return float2(s, b - (s - a));
}

// Sloppy add. The error is relative to |a| + |b|, which is what a long sum of
// same-scale terms needs.
inline ds ds_add(ds a, ds b) {
    float2 s = two_sum(a.hi, b.hi);
    float2 r = quick_two_sum(s.x, s.y + (a.lo + b.lo));
    return ds{r.x, r.y};
}

// IEEE-style add (Joldes–Muller–Popescu). Needed where operands cancel.
inline ds ds_addp(ds a, ds b) {
    float2 s = two_sum(a.hi, b.hi);
    float2 t = two_sum(a.lo, b.lo);
    float2 u = quick_two_sum(s.x, s.y + t.x);
    float2 w = quick_two_sum(u.x, t.y + u.y);
    return ds{w.x, w.y};
}

inline ds ds_sub(ds a, ds b) { return ds_add(a, ds_neg(b)); }
inline ds ds_subp(ds a, ds b) { return ds_addp(a, ds_neg(b)); }

// The residual of hi*hi is an exact hardware fma.
inline ds ds_mul(ds a, ds b) {
    float p = a.hi * b.hi;
    float e = fma(a.hi, b.hi, -p);
    e = fma(a.hi, b.lo, e);
    e = fma(a.lo, b.hi, e);
    float2 r = quick_two_sum(p, e);
    return ds{r.x, r.y};
}

inline ds ds_mulf(float a, ds b) {
    float p = a * b.hi;
    float e = fma(a, b.hi, -p);
    e = fma(a, b.lo, e);
    float2 r = quick_two_sum(p, e);
    return ds{r.x, r.y};
}

// Exact scalings by powers of two.
inline ds ds_scale(ds a, float pow2) { return ds{a.hi * pow2, a.lo * pow2}; }

inline ds ds_sqr(ds a) {
    float p = a.hi * a.hi;
    float e = fma(a.hi, a.hi, -p);
    e = fma(2.0f * a.hi, a.lo, e);
    float2 r = quick_two_sum(p, e);
    return ds{r.x, r.y};
}

// One Newton correction from the f32 seed leaves ~2^-46 relative error.
inline ds ds_recip(ds b) {
    float q0 = 1.0f / b.hi;
    float r = fma(-q0, b.hi, 1.0f) - q0 * b.lo;
    float2 q = quick_two_sum(q0, q0 * r);
    return ds{q.x, q.y};
}

inline ds ds_div(ds a, ds b) {
    float q0 = a.hi / b.hi;
    float r = fma(-q0, b.hi, a.hi) + (a.lo - q0 * b.lo);
    float2 q = quick_two_sum(q0, r / b.hi);
    return ds{q.x, q.y};
}

// Karp–Markstein: the residual of the f32 root is exact through fma.
inline ds ds_sqrt(ds a) {
    if (!(a.hi > 0.0f)) {
        return ds_zero();
    }
    float s0 = sqrt(a.hi);
    float r = fma(-s0, s0, a.hi) + a.lo;
    float2 s = quick_two_sum(s0, r / (2.0f * s0));
    return ds{s.x, s.y};
}

inline bool ds_gt(ds a, ds b) {
    ds d = ds_subp(a, b);
    return d.hi > 0.0f;
}

inline ds ds_max0(ds a) { return (a.hi > 0.0f || (a.hi == 0.0f && a.lo > 0.0f)) ? a : ds_zero(); }

// Fused accumulator: a long sum of products keeps its running high word exact
// (TwoSum) and collects every rounding error in one plain f32.
struct dsacc {
    float hi;
    float lo;
};

inline dsacc acc_zero() { return dsacc{0.0f, 0.0f}; }
inline dsacc acc_from(ds a) { return dsacc{a.hi, a.lo}; }

inline void acc_fma(thread dsacc& s, ds a, ds b) {
    float p = a.hi * b.hi;
    float e = fma(a.hi, b.hi, -p);
    e = fma(a.hi, b.lo, e);
    e = fma(a.lo, b.hi, e);
    float2 t = two_sum(s.hi, p);
    s.hi = t.x;
    s.lo += t.y + e;
}

inline void acc_fnma(thread dsacc& s, ds a, ds b) {
    float p = a.hi * b.hi;
    float e = fma(a.hi, b.hi, -p);
    e = fma(a.hi, b.lo, e);
    e = fma(a.lo, b.hi, e);
    float2 t = two_sum(s.hi, -p);
    s.hi = t.x;
    s.lo += t.y - e;
}

inline void acc_add(thread dsacc& s, ds a) {
    float2 t = two_sum(s.hi, a.hi);
    s.hi = t.x;
    s.lo += t.y + a.lo;
}

inline ds acc_done(dsacc s) {
    float2 r = two_sum(s.hi, s.lo);
    return ds{r.x, r.y};
}

// Constants split as hi = f32(x), lo = f32(x - hi).
inline ds ds_half_pi() { return ds{1.5707963705e+00f, -4.3711388287e-08f}; }

// arctan for z >= 0. A reciprocal fold maps z to [0, 1]; three half-angle
// steps u = z / (1 + sqrt(1 + z^2)) bring it below tan(pi/32) = 0.0985, so
// the Taylor series needs eight Horner terms instead of the 28 of a bare series.
inline ds ds_atan_pos(ds z) {
    bool inv = z.hi > 1.0f;
    if (inv) {
        z = ds_recip(z);
    }
    for (int i = 0; i < 3; ++i) {
        ds root = ds_sqrt(ds_add(ds_f(1.0f), ds_sqr(z)));
        z = ds_div(z, ds_add(ds_f(1.0f), root));
    }
    ds u2 = ds_sqr(z);
    ds h = ds{-6.66666701e-02f, 3.47693763e-09f};
    h = ds_add(ds_mul(h, u2), ds{7.69230798e-02f, -2.86560797e-09f});
    h = ds_add(ds_mul(h, u2), ds{-9.09090936e-02f, 2.70930212e-09f});
    h = ds_add(ds_mul(h, u2), ds{1.11111112e-01f, -8.27842295e-10f});
    h = ds_add(ds_mul(h, u2), ds{-1.42857149e-01f, 6.38621200e-09f});
    h = ds_add(ds_mul(h, u2), ds{2.00000003e-01f, -2.98023228e-09f});
    h = ds_add(ds_mul(h, u2), ds{-3.33333343e-01f, 9.93410776e-09f});
    h = ds_add(ds_mul(h, u2), ds_f(1.0f));
    ds a = ds_scale(ds_mul(h, z), 8.0f);
    return inv ? ds_subp(ds_half_pi(), a) : a;
}

#endif
