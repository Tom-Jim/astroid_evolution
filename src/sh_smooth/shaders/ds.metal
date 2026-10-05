#pragma clang fp contract(off)
#ifndef SH_DS_METAL
#define SH_DS_METAL

#include <metal_stdlib>
using namespace metal;

// Double-single arithmetic. Apple GPUs have no f64 unit. Compiled with
// MTLMathMode::Safe and fp contract(off), so every fma residual stays put.
// The product lo*lo is dropped: it is 2^-48 relative.

struct ds {
    float hi;
    float lo;
};

inline ds ds_f(float h) { return ds{h, 0.0f}; }
inline ds ds_zero() { return ds{0.0f, 0.0f}; }
inline ds ds_neg(ds a) { return ds{-a.hi, -a.lo}; }

inline float2 two_sum(float a, float b) {
    float s = a + b;
    float v = s - a;
    return float2(s, (a - (s - v)) + (b - v));
}

inline float2 quick_two_sum(float a, float b) {
    float s = a + b;
    return float2(s, b - (s - a));
}

inline ds ds_add(ds a, ds b) {
    float2 s = two_sum(a.hi, b.hi);
    float2 r = quick_two_sum(s.x, s.y + (a.lo + b.lo));
    return ds{r.x, r.y};
}

inline ds ds_addp(ds a, ds b) {
    float2 s = two_sum(a.hi, b.hi);
    float2 t = two_sum(a.lo, b.lo);
    float2 u = quick_two_sum(s.x, s.y + t.x);
    float2 w = quick_two_sum(u.x, t.y + u.y);
    return ds{w.x, w.y};
}

inline ds ds_sub(ds a, ds b) { return ds_add(a, ds_neg(b)); }
inline ds ds_subp(ds a, ds b) { return ds_addp(a, ds_neg(b)); }

inline ds ds_mul(ds a, ds b) {
    float p = a.hi * b.hi;
    float e = fma(a.hi, b.hi, -p);
    e = fma(a.hi, b.lo, e);
    e = fma(a.lo, b.hi, e);
    float2 r = quick_two_sum(p, e);
    return ds{r.x, r.y};
}

struct dsacc {
    float hi;
    float lo;
};

inline dsacc acc_zero() { return dsacc{0.0f, 0.0f}; }

inline void acc_fma(thread dsacc& s, ds a, ds b) {
    float p = a.hi * b.hi;
    float e = fma(a.hi, b.hi, -p);
    e = fma(a.hi, b.lo, e);
    e = fma(a.lo, b.hi, e);
    float2 t = two_sum(s.hi, p);
    s.hi = t.x;
    s.lo += t.y + e;
}

inline ds acc_done(dsacc s) {
    float2 r = two_sum(s.hi, s.lo);
    return ds{r.x, r.y};
}

#endif
