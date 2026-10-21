// Browser double-single. Desktop keeps shaders/ds.metal and does not compile this file.
// Tint deletes both the Knuth residual `(a - (s - v)) + (b - v)` and the product
// residual `fma(a, b, -a*b)`. Either one collapses this library to f32.
// `add_err` and `mul_err` are those residuals, computed from the significands.
// Accumulator updates return the new value because WGSL has no reference parameters.

struct ds {
    hi: f32,
    lo: f32,
}

struct dsacc {
    hi: f32,
    lo: f32,
}

fn ds_mk(h: f32, l: f32) -> ds { return ds(h, l); }
fn ds_f(h: f32) -> ds { return ds(h, 0.0); }
fn ds_zero() -> ds { return ds(0.0, 0.0); }
fn ds_neg(a: ds) -> ds { return ds(-a.hi, -a.lo); }
fn ds_abs(a: ds) -> ds {
    if (a.hi < 0.0 || (a.hi == 0.0 && a.lo < 0.0)) { return ds_neg(a); }
    return a;
}
fn ds_float(a: ds) -> f32 { return a.hi + a.lo; }

struct Fix {
    exp: i32,
    hi: i32,
    lo: u32,
}

fn unpack_to(x: f32, exp: i32) -> Fix {
    let u = bitcast<u32>(x);
    let sign = u >> 31u;
    let eb = i32((u >> 23u) & 0xffu);
    let frac = u & 0x7fffffu;
    var sig: u32 = 0u;
    var xe: i32 = -200;
    if (eb == 255) {
        return Fix(exp, 0, 0u);
    }
    if (eb == 0) {
        if (frac != 0u) {
            let k = countLeadingZeros(frac) - 8u;
            sig = frac << k;
            xe = -126 - i32(k);
        }
    } else {
        sig = frac | 0x800000u;
        xe = eb - 127;
    }
    if (sig == 0u) {
        return Fix(exp, 0, 0u);
    }
    let de = exp - xe;
    var hi: u32 = 0u;
    var lo: u32 = 0u;
    if (de <= 0) {
        let up = u32(-de);
        if (up >= 32u) {
            hi = 0x7fffffffu;
        } else {
            hi = sig << up;
        }
    } else if (de < 32) {
        hi = sig >> u32(de);
        lo = (sig & ((1u << u32(de)) - 1u)) << (32u - u32(de));
    } else if (de < 64) {
        lo = sig >> u32(de - 32);
    }
    var shi = i32(hi);
    if (sign != 0u) {
        if (lo != 0u) {
            lo = 0u - lo;
            shi = -shi - 1;
        } else {
            shi = -shi;
        }
    }
    return Fix(exp, shi, lo);
}

fn add_fix(a: Fix, b: Fix) -> Fix {
    let lo = a.lo + b.lo;
    var carry: i32 = 0;
    if (lo < a.lo) { carry = 1; }
    return Fix(a.exp, a.hi + b.hi + carry, lo);
}

fn pack_mag(sign: u32, bit0_exp: i32, mag: u32) -> f32 {
    if (mag == 0u) {
        return bitcast<f32>(sign << 31u);
    }
    let clz = countLeadingZeros(mag);
    let msb = 31 - i32(clz);
    let unbiased = bit0_exp + msb;
    let sig = (mag << clz) >> 8u;
    if (unbiased > 127) {
        return bitcast<f32>((sign << 31u) | 0x7f800000u);
    }
    if (unbiased < -126) {
        let sh = u32(-126 - unbiased);
        if (sh >= 24u) {
            return bitcast<f32>(sign << 31u);
        }
        return bitcast<f32>((sign << 31u) | ((sig >> sh) & 0x7fffffu));
    }
    return bitcast<f32>((sign << 31u) | (u32(unbiased + 127) << 23u) | (sig & 0x7fffffu));
}

fn sub_to_f32(exact: Fix, s: f32) -> f32 {
    let packed = unpack_to(s, exact.exp);
    let lo_wrap = exact.lo < packed.lo;
    let lo = exact.lo - packed.lo;
    var borrow: i32 = 0;
    if (lo_wrap) { borrow = 1; }
    let hi = exact.hi - packed.hi - borrow;
    if (hi == 0 && lo == 0u) {
        return 0.0;
    }
    var sign = 0u;
    var ah = hi;
    var al = lo;
    if (ah < 0) {
        sign = 1u;
        if (al != 0u) {
            al = 0u - al;
            ah = -ah - 1;
        } else {
            ah = -ah;
        }
    }
    let bit0 = exact.exp - 23 - 32;
    if (ah == 0) {
        return pack_mag(sign, bit0, al);
    }
    let top = countLeadingZeros(u32(ah));
    let unbiased = (exact.exp - 23) + (31 - i32(top));
    var sig = (u32(ah) << top) >> 8u;
    if (top > 8u) {
        sig = sig | (al >> (32u - (top - 8u)));
    }
    if (unbiased > 127) {
        return bitcast<f32>((sign << 31u) | 0x7f800000u);
    }
    // A product whose f32 rounding underflows leaves a subnormal residual in the
    // high limb. `u32` of a negative exponent field is not a small exponent.
    if (unbiased < -126) {
        let sh = -126 - unbiased;
        if (sh >= 24) {
            return bitcast<f32>(sign << 31u);
        }
        return bitcast<f32>((sign << 31u) | ((sig >> u32(sh)) & 0x7fffffu));
    }
    return bitcast<f32>((sign << 31u) | (u32(unbiased + 127) << 23u) | (sig & 0x7fffffu));
}

fn mul_u24(a: u32, b: u32) -> vec2<u32> {
    let a0 = a & 0xffffu;
    let a1 = a >> 16u;
    let b0 = b & 0xffffu;
    let b1 = b >> 16u;
    let p00 = a0 * b0;
    let p01 = a0 * b1;
    let p10 = a1 * b0;
    let p11 = a1 * b1;
    let mid = (p00 >> 16u) + (p01 & 0xffffu) + (p10 & 0xffffu);
    let lo = (p00 & 0xffffu) | ((mid & 0xffffu) << 16u);
    let hi = p11 + (p01 >> 16u) + (p10 >> 16u) + (mid >> 16u);
    return vec2<u32>(lo, hi);
}

// Exact residual of a f32 product: `a*b = fl(a*b) + mul_err(a, b)`.
fn mul_err(a: f32, b: f32) -> f32 {
    let p = a * b;
    let ea = (bitcast<u32>(a) >> 23u) & 0xffu;
    let eb = (bitcast<u32>(b) >> 23u) & 0xffu;
    if (ea == 0u || eb == 0u || ea == 0xffu || eb == 0xffu) {
        return 0.0;
    }
    let siga = (bitcast<u32>(a) & 0x7fffffu) | 0x800000u;
    let sigb = (bitcast<u32>(b) & 0x7fffffu) | 0x800000u;
    let prod = mul_u24(siga, sigb);
    let wide_lo = prod.x << 9u;
    let wide_hi = (prod.y << 9u) | (prod.x >> 23u);
    var shi = i32(wide_hi);
    var slo = wide_lo;
    if (((bitcast<u32>(a) ^ bitcast<u32>(b)) >> 31u) != 0u) {
        if (slo != 0u) {
            slo = 0u - slo;
            shi = -shi - 1;
        } else {
            shi = -shi;
        }
    }
    return sub_to_f32(Fix(i32(ea) + i32(eb) - 254, shi, slo), p);
}

fn add_err(a: f32, b: f32) -> vec2<f32> {
    let s = a + b;
    let ea_bits = (bitcast<u32>(a) >> 23u) & 0xffu;
    let eb_bits = (bitcast<u32>(b) >> 23u) & 0xffu;
    if (ea_bits == 0xffu || eb_bits == 0xffu || ea_bits == 0u || eb_bits == 0u) {
        return vec2<f32>(s, 0.0);
    }
    let ea = i32(ea_bits) - 127;
    let eb = i32(eb_bits) - 127;
    var exp = ea;
    if (eb > ea) { exp = eb; }
    let sum = add_fix(unpack_to(a, exp), unpack_to(b, exp));
    return vec2<f32>(s, sub_to_f32(sum, s));
}

fn two_sum(a: f32, b: f32) -> vec2<f32> { return add_err(a, b); }
fn quick_two_sum(a: f32, b: f32) -> vec2<f32> { return add_err(a, b); }

fn ds_add(a: ds, b: ds) -> ds {
    let s = two_sum(a.hi, b.hi);
    let r = quick_two_sum(s.x, s.y + (a.lo + b.lo));
    return ds(r.x, r.y);
}

fn ds_addp(a: ds, b: ds) -> ds {
    let s = two_sum(a.hi, b.hi);
    let t = two_sum(a.lo, b.lo);
    let u = quick_two_sum(s.x, s.y + t.x);
    let w = quick_two_sum(u.x, t.y + u.y);
    return ds(w.x, w.y);
}

fn ds_sub(a: ds, b: ds) -> ds { return ds_add(a, ds_neg(b)); }
fn ds_subp(a: ds, b: ds) -> ds { return ds_addp(a, ds_neg(b)); }

fn ds_mul(a: ds, b: ds) -> ds {
    let p = a.hi * b.hi;
    var e = mul_err(a.hi, b.hi);
    e = e + a.hi * b.lo;
    e = e + a.lo * b.hi;
    let r = quick_two_sum(p, e);
    return ds(r.x, r.y);
}

fn ds_mulf(a: f32, b: ds) -> ds {
    let p = a * b.hi;
    var e = mul_err(a, b.hi);
    e = e + a * b.lo;
    let r = quick_two_sum(p, e);
    return ds(r.x, r.y);
}

fn ds_scale(a: ds, pow2: f32) -> ds { return ds(a.hi * pow2, a.lo * pow2); }

fn ds_sqr(a: ds) -> ds {
    let p = a.hi * a.hi;
    var e = mul_err(a.hi, a.hi);
    e = e + 2.0 * a.hi * a.lo;
    let r = quick_two_sum(p, e);
    return ds(r.x, r.y);
}

fn ds_recip(b: ds) -> ds {
    let q0 = 1.0 / b.hi;
    let rem = ds_subp(ds_f(1.0), ds_mul(ds_f(q0), b));
    let corr = (rem.hi + rem.lo) * q0;
    return ds_add(ds_f(q0), ds_f(corr));
}

fn ds_div(a: ds, b: ds) -> ds {
    let q0 = a.hi / b.hi;
    let rem = ds_subp(a, ds_mul(ds_f(q0), b));
    let corr = (rem.hi + rem.lo) / b.hi;
    return ds_add(ds_f(q0), ds_f(corr));
}

fn ds_sqrt(a: ds) -> ds {
    if (!(a.hi > 0.0)) {
        return ds_zero();
    }
    let s0 = sqrt(a.hi);
    let rem = ds_subp(a, ds_sqr(ds_f(s0)));
    let corr = (rem.hi + rem.lo) / (2.0 * s0);
    return ds_add(ds_f(s0), ds_f(corr));
}

fn ds_gt(a: ds, b: ds) -> bool {
    let d = ds_subp(a, b);
    return d.hi > 0.0;
}

fn ds_max0(a: ds) -> ds {
    if (a.hi > 0.0 || (a.hi == 0.0 && a.lo > 0.0)) { return a; }
    return ds_zero();
}

fn acc_zero() -> dsacc { return dsacc(0.0, 0.0); }
fn acc_from(a: ds) -> dsacc { return dsacc(a.hi, a.lo); }

fn acc_fma(s: dsacc, a: ds, b: ds) -> dsacc {
    let p = a.hi * b.hi;
    var e = mul_err(a.hi, b.hi);
    e = e + a.hi * b.lo;
    e = e + a.lo * b.hi;
    let t = two_sum(s.hi, p);
    return dsacc(t.x, s.lo + t.y + e);
}

fn acc_fnma(s: dsacc, a: ds, b: ds) -> dsacc {
    let p = a.hi * b.hi;
    var e = mul_err(a.hi, b.hi);
    e = e + a.hi * b.lo;
    e = e + a.lo * b.hi;
    let t = two_sum(s.hi, -p);
    return dsacc(t.x, s.lo + t.y - e);
}

fn acc_add(s: dsacc, a: ds) -> dsacc {
    let t = two_sum(s.hi, a.hi);
    return dsacc(t.x, s.lo + t.y + a.lo);
}

fn acc_done(s: dsacc) -> ds {
    let r = two_sum(s.hi, s.lo);
    return ds(r.x, r.y);
}

fn ds_half_pi() -> ds { return ds(1.5707963705e+00, -4.3711388287e-08); }

fn ds_atan_pos(z_in: ds) -> ds {
    var z = z_in;
    let inv = z.hi > 1.0;
    if (inv) {
        z = ds_recip(z);
    }
    for (var i = 0; i < 3; i = i + 1) {
        let root = ds_sqrt(ds_add(ds_f(1.0), ds_sqr(z)));
        z = ds_div(z, ds_add(ds_f(1.0), root));
    }
    let u2 = ds_sqr(z);
    var h = ds(-6.66666701e-02, 3.47693763e-09);
    h = ds_add(ds_mul(h, u2), ds(7.69230798e-02, -2.86560797e-09));
    h = ds_add(ds_mul(h, u2), ds(-9.09090936e-02, 2.70930212e-09));
    h = ds_add(ds_mul(h, u2), ds(1.11111112e-01, -8.27842295e-10));
    h = ds_add(ds_mul(h, u2), ds(-1.42857149e-01, 6.38621200e-09));
    h = ds_add(ds_mul(h, u2), ds(2.00000003e-01, -2.98023228e-09));
    h = ds_add(ds_mul(h, u2), ds(-3.33333343e-01, 9.93410776e-09));
    h = ds_add(ds_mul(h, u2), ds_f(1.0));
    let a = ds_scale(ds_mul(h, z), 8.0);
    if (inv) {
        return ds_subp(ds_half_pi(), a);
    }
    return a;
}

// One f64 encoding for every WESL shader. Field used to keep a second copy.
fn ds_from_float(h: f32) -> ds { return ds_f(h); }

fn ds_pick(when_false: ds, when_true: ds, cond: bool) -> ds {
    if (cond) { return when_true; }
    return when_false;
}

fn ds_pi() -> ds { return ds(3.1415927410e+00, -8.7422776573e-08); }
fn ds_quarter_pi() -> ds { return ds(7.8539818525e-01, -2.1855694143e-08); }
fn ds_ln2() -> ds { return ds(6.9314718246e-01, -1.9046542121e-09); }
fn ds_inv_ln2() -> ds { return ds(1.4426950216e+00, 1.9259630335e-08); }

fn ds_exp(x: ds) -> ds {
    let scaled = ds_mul(x, ds_inv_ln2());
    let nf = round(scaled.hi + scaled.lo);
    let r = ds_sub(x, ds_mul(ds_f(nf), ds_ln2()));
    var sum = ds_f(1.0);
    var term = ds_f(1.0);
    for (var k = 1; k <= 18; k = k + 1) {
        term = ds_div(ds_mul(term, r), ds_f(f32(k)));
        sum = ds_add(sum, term);
    }
    let n = i32(nf);
    sum.hi = ldexp(sum.hi, n);
    sum.lo = ldexp(sum.lo, n);
    return sum;
}

fn ds_ln(a: ds) -> ds {
    if (!(a.hi > 0.0)) {
        // A constant infinity is rejected by Tint. Same bit pattern, not a const.
        var bits = 0xff800000u;
        return ds(bitcast<f32>(bits), 0.0);
    }
    let split = frexp(a.hi);
    let m = split.fract;
    var e = split.exp;
    var md = ds_add(ds_f(m), ds_f(a.lo * ldexp(1.0, -e)));
    if (md.hi < 0.7071067811865476) {
        md = ds_add(md, md);
        e = e - 1;
    }
    var y = ds_f(log(md.hi));
    for (var i = 0; i < 2; i = i + 1) {
        let ey = ds_exp(y);
        let num = ds_sub(ey, md);
        let den = ds_add(ey, md);
        y = ds_sub(y, ds_mul(ds_f(2.0), ds_div(num, den)));
    }
    return ds_add(y, ds_mul(ds_f(f32(e)), ds_ln2()));
}

fn ds_atan(z: ds) -> ds {
    if (z.hi < 0.0 || (z.hi == 0.0 && z.lo < 0.0)) {
        return ds_neg(ds_atan_pos(ds_neg(z)));
    }
    return ds_atan_pos(z);
}

fn ds_atan2(y: ds, x: ds) -> ds {
    if (x.hi == 0.0 && y.hi == 0.0) {
        return ds_zero();
    }
    if (x.hi > 0.0) {
        return ds_atan(ds_div(y, x));
    }
    if (x.hi < 0.0) {
        let a = ds_atan(ds_div(y, x));
        if (y.hi >= 0.0) {
            return ds_add(a, ds_pi());
        }
        return ds_sub(a, ds_pi());
    }
    if (y.hi >= 0.0) {
        return ds_half_pi();
    }
    return ds_neg(ds_half_pi());
}
