// Cap deposition of mode C. One thread owns one vertex.
//
// A kernel whose degree-32 Legendre truncation stays within 1e-3 relative L1
// is applied on the host by the zonal convolution in `mode_c_spectral`. This
// shader remains the reference walk for every narrower kernel, and the
// comparison sample that gates that truncation.
//
// `mode_c_norm` writes the source normalisation Σ weight·area over the cap.
// `mode_c_deposit` writes J_dep at the target. Both walks use the host cap
// box from `DirIndex::collect_within` and the same weight, so
// Σ J_dep dΓ = Σ J_ero dΓ up to double-single rounding.

struct ModeParams {
    uint n;
    uint g;
    uint dense;
    uint pad;
    float w_hi;
    float w_lo;
    float cutoff_hi;
    float cutoff_lo;
    float reject_hi;
    float reject_lo;
};

inline ds ld(device const float2* a, uint i) {
    float2 v = a[i];
    return ds{v.x, v.y};
}

inline void st(device float2* a, uint i, ds v) { a[i] = float2(v.hi, v.lo); }

inline ds ld_axis(device const float2* dirs, uint v, uint axis) {
    return ld(dirs, v * 3u + axis);
}

inline ds dir_dot(device const float2* dirs, uint a, uint b) {
    dsacc s = acc_zero();
    acc_fma(s, ld_axis(dirs, a, 0u), ld_axis(dirs, b, 0u));
    acc_fma(s, ld_axis(dirs, a, 1u), ld_axis(dirs, b, 1u));
    acc_fma(s, ld_axis(dirs, a, 2u), ld_axis(dirs, b, 2u));
    return acc_done(s);
}

inline bool positive(ds a) { return a.hi > 0.0f || (a.hi == 0.0f && a.lo > 0.0f); }

// Angle and Gaussian in native f32. The long sum of those weights stays in a
// double-single accumulator. A vertex paired with itself is angle 0.
inline float kernel_weight(uint a, uint b, float cos_raw, constant ModeParams& p) {
    if (a == b) {
        return 1.0f;
    }
    // Loosen the host reject by a few ulps so the f32 cosine cannot drop a
    // point the f64 `acos` test would keep.
    if (p.dense == 0u && cos_raw < p.reject_hi - 1.0e-5f) {
        return 0.0f;
    }
    float c = clamp(cos_raw, -1.0f, 1.0f);
    float ang = precise::acos(c);
    float cutoff = p.cutoff_hi + p.cutoff_lo;
    if (!(ang < cutoff)) {
        return 0.0f;
    }
    float t = ang / (p.w_hi + p.w_lo);
    float weight = precise::exp(-0.5f * t * t);
    return weight > 0.0f ? weight : 0.0f;
}

inline float cosine(device const float2* dirs, uint a, uint b) {
    ds d = dir_dot(dirs, a, b);
    return d.hi + d.lo;
}

kernel void mode_c_norm(
    device const float2* dirs [[buffer(0)]],
    device const float2* area [[buffer(1)]],
    device const float2* mass_rate [[buffer(2)]],
    device const uint* bin_start [[buffer(3)]],
    device const uint* bin_items [[buffer(4)]],
    device const uint* caps [[buffer(5)]],
    device float2* norm [[buffer(6)]],
    constant ModeParams& p [[buffer(7)]],
    uint src [[thread_position_in_grid]]) {
    if (src >= p.n) {
        return;
    }
    if (!positive(ld(mass_rate, src))) {
        st(norm, src, ds_zero());
        return;
    }
    dsacc sum = acc_zero();
    if (p.dense != 0u) {
        for (uint t = 0u; t < p.n; ++t) {
            float weight = kernel_weight(src, t, cosine(dirs, src, t), p);
            if (weight > 0.0f) {
                acc_fma(sum, ds_f(weight), ld(area, t));
            }
        }
    } else {
        uint x0 = caps[src * 6u];
        uint x1 = caps[src * 6u + 1u];
        uint y0 = caps[src * 6u + 2u];
        uint y1 = caps[src * 6u + 3u];
        uint z0 = caps[src * 6u + 4u];
        uint z1 = caps[src * 6u + 5u];
        for (uint x = x0; x <= x1; ++x) {
            for (uint y = y0; y <= y1; ++y) {
                for (uint z = z0; z <= z1; ++z) {
                    uint c = (x * p.g + y) * p.g + z;
                    uint end = bin_start[c + 1u];
                    for (uint k = bin_start[c]; k < end; ++k) {
                        uint t = bin_items[k];
                        float weight = kernel_weight(src, t, cosine(dirs, src, t), p);
                        if (weight > 0.0f) {
                            acc_fma(sum, ds_f(weight), ld(area, t));
                        }
                    }
                }
            }
        }
    }
    st(norm, src, acc_done(sum));
}

kernel void mode_c_deposit(
    device const float2* dirs [[buffer(0)]],
    device const float2* mass_rate [[buffer(1)]],
    device const float2* norm [[buffer(2)]],
    device const uint* bin_start [[buffer(3)]],
    device const uint* bin_items [[buffer(4)]],
    device const uint* caps [[buffer(5)]],
    device float2* j_dep [[buffer(6)]],
    constant ModeParams& p [[buffer(7)]],
    uint target [[thread_position_in_grid]]) {
    if (target >= p.n) {
        return;
    }
    dsacc sum = acc_zero();
    if (p.dense != 0u) {
        for (uint src = 0u; src < p.n; ++src) {
            ds rate = ld(mass_rate, src);
            ds nrm = ld(norm, src);
            if (!positive(rate) || !positive(nrm)) {
                continue;
            }
            float weight = kernel_weight(src, target, cosine(dirs, src, target), p);
            if (weight > 0.0f) {
                acc_add(sum, ds_mul(ds_f(weight), ds_div(rate, nrm)));
            }
        }
    } else {
        uint x0 = caps[target * 6u];
        uint x1 = caps[target * 6u + 1u];
        uint y0 = caps[target * 6u + 2u];
        uint y1 = caps[target * 6u + 3u];
        uint z0 = caps[target * 6u + 4u];
        uint z1 = caps[target * 6u + 5u];
        for (uint x = x0; x <= x1; ++x) {
            for (uint y = y0; y <= y1; ++y) {
                for (uint z = z0; z <= z1; ++z) {
                    uint c = (x * p.g + y) * p.g + z;
                    uint end = bin_start[c + 1u];
                    for (uint k = bin_start[c]; k < end; ++k) {
                        uint src = bin_items[k];
                        ds rate = ld(mass_rate, src);
                        ds nrm = ld(norm, src);
                        if (!positive(rate) || !positive(nrm)) {
                            continue;
                        }
                        float weight = kernel_weight(src, target, cosine(dirs, src, target), p);
                        if (weight > 0.0f) {
                            acc_add(sum, ds_mul(ds_f(weight), ds_div(rate, nrm)));
                        }
                    }
                }
            }
        }
    }
    st(j_dep, target, acc_done(sum));
}
