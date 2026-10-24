// #include "gpu/shader/ds.wesl"
// SHARED
// The ζ-power arrays of the velocity kernels hold 12 entries (VEL_POWS in basis_gpu.rs).
struct Jt_0 {
    c: array<ds, 4>,
}

struct Jt_1 {
    c: array<ds, 4>,
}

struct Jt_2 {
    c: array<ds, 10>,
}

struct Jt_3 {
    c: array<ds, 20>,
}

struct Jt_4 {
    c: array<ds, 35>,
}

struct BasisParams {
    n_rays: u32,
    n_per: u32,
    n_points: u32,
    l_mode: u32,
    l_surf: u32,
    k_count: u32,
    n_out: u32,
    n_lift: u32,
    n_comp: u32,
    poly_stride: u32,
    top_pow: u32,
    want_v: u32,
    want_d: u32,
    want_lap: u32,
    want_gd: u32,
    gradients: u32,
    nmul0: u32,
    nmul1: u32,
    nmul2: u32,
    nmul3: u32,
    nmul4: u32,
    moff0: u32,
    moff1: u32,
    moff2: u32,
    moff3: u32,
    moff4: u32,
    up_off: u32,
    mono_off: u32,
    deg_off: u32,
    pair_off: u32,
    sqrt2_hi: f32,
    sqrt2_lo: f32,
    pmm_hi: f32,
    pmm_lo: f32,
}

struct GemvParams {
    n_points: u32,
    n_modes: u32,
    n_comp: u32,
    pad: u32,
}

fn lm_of(l: i32, m: i32) -> u32 {
    return u32(l * l + l + m);
}

fn vel_dof(l: i32, m: i32, tor: u32, k: u32, k_count: u32) -> u32 {
    var blk: i32 = l * l + l + m - 1;
    return u32(i32(2u * k_count) * blk + i32(select(0u, k_count, tor != 0u)) + i32(k));
}

fn coef_a(l: i32, m: i32) -> ds {
    var lf: f32 = f32(l);
    var mf: f32 = f32(m);
    return ds_sqrt(ds_div(ds_f(4.0 * lf * lf - 1.0), ds_f(lf * lf - mf * mf)));
}

fn coef_b(l: i32, m: i32) -> ds {
    var lm: f32 = f32(l) - 1.0;
    var mf: f32 = f32(m);
    return ds_sqrt(ds_div(ds_f(lm * lm - mf * mf), ds_f(4.0 * lm * lm - 1.0)));
}

fn coeff_at(l: i32, m: i32) -> ds {
    var c: vec2<f32> = coeff[lm_of(l, m)];
    return ds_mk(c.x, c.y);
}

fn load_point(i: u32, x: ptr<function, ds>, y: ptr<function, ds>, z: ptr<function, ds>) {
    var b: u32 = 3u * i;
    *x = ds_mk(points[b].x, points[b].y);
    *y = ds_mk(points[b + 1].x, points[b + 1].y);
    *z = ds_mk(points[b + 2].x, points[b + 2].y);
}

fn store_ds(channel: u32, n_points: u32, point: u32, v: ds) {
    out[channel * n_points + point] = vec2<f32>(v.hi, v.lo);
}

fn store_contract(point: u32, n_comp: u32, acc: ptr<function, array<dsacc, 9>>) {
    for(var c: u32 = 0; c < n_comp; c = c + 1u) {
        var r: ds = acc_done((*acc)[c]);
        out[point * n_comp + c] = vec2<f32>(r.hi, r.lo);
    }
}

fn vd_channels(p: BasisParams) -> u32 {
    return select(0u, 3u, p.want_v != 0u) + select(0u, 6u, p.want_d != 0u);
}

fn j_zero_1(r: ptr<function, Jt_1>) {
    for(var i: i32 = 0; i < 4; i = i + 1) {
        (*r).c[i] = ds_zero();
    }
}

fn j_const_1(r: ptr<function, Jt_1>, v: ds) {
    j_zero_1(r);
    (*r).c[0] = v;
}

fn j_var_1(r: ptr<function, Jt_1>, axis: i32, x0: ds) {
    j_const_1(r, x0);
    if 1 >= 1 {
        (*r).c[1 + axis] = ds_f(1.0);
    }
}

fn mul_tab_1(p: BasisParams) -> u32 {
    if 1 <= 0 {
        return p.moff0;
    }
    if 1 == 1 {
        return p.moff1;
    }
    if 1 == 2 {
        return p.moff2;
    }
    if 1 == 3 {
        return p.moff3;
    }
    return p.moff4;
}

fn mul_count_1(p: BasisParams) -> u32 {
    if 1 <= 0 {
        return p.nmul0;
    }
    if 1 == 1 {
        return p.nmul1;
    }
    if 1 == 2 {
        return p.nmul2;
    }
    if 1 == 3 {
        return p.nmul3;
    }
    return p.nmul4;
}

fn j_mul_1(r: ptr<function, Jt_1>, a: Jt_1, b: Jt_1, p: BasisParams) {
    var acc: array<dsacc, 35>;
    for(var i: i32 = 0; i < 4; i = i + 1) {
        acc[i] = acc_zero();
    }
    let tab_base = mul_tab_1(p);
    var n: u32 = mul_count_1(p);
    for(var t: u32 = 0; t < n; t = t + 1u) {
        var e: u32 = tables[tab_base + t];
        acc[(e >> 16) & 255u] = acc_fma(acc[(e >> 16) & 255u], a.c[e & 255u], b.c[(e >> 8) & 255u]);
    }
    for(var i: i32 = 0; i < 4; i = i + 1) {
        (*r).c[i] = acc_done(acc[i]);
    }
}

fn j_add3_1(r: ptr<function, Jt_1>, a: Jt_1, b: Jt_1, c: Jt_1) {
    for(var i: i32 = 0; i < 4; i = i + 1) {
        (*r).c[i] = ds_addp(ds_addp(a.c[i], b.c[i]), c.c[i]);
    }
    for(var i: i32 = 4; i < 4; i = i + 1) {
        (*r).c[i] = ds_zero();
    }
}

fn j_compose_1(r: ptr<function, Jt_1>, jet: Jt_1, f: ptr<function, array<ds, 5>>, p: BasisParams) {
    var delta: Jt_1 = jet;
    delta.c[0] = ds_zero();
    var fact: ds = ds_f(1.0);
    var coef: array<ds, 5>;
    for(var k: i32 = 0; k <= 1; k = k + 1) {
        if k > 0 {
            fact = ds_mul(fact, ds_f(f32(k)));
        }
        coef[k] = ds_div((*f)[k], fact);
    }
    j_const_1(r, coef[1]);
    for(var k: i32 = i32(1) - 1; k >= 0; k = k - 1) {
        var tmp: Jt_1;
        j_mul_1(&tmp, *r, delta, p);
        tmp.c[0] = ds_addp(tmp.c[0], coef[k]);
        *r = tmp;
    }
}

fn j_sqrt_1(r: ptr<function, Jt_1>, jet: Jt_1, p: BasisParams) {
    var a: ds = jet.c[0];
    var s: ds = ds_sqrt(a);
    var as_: ds = ds_mul(a, s);
    var a2: ds = ds_sqr(a);
    var a2s: ds = ds_mul(a2, s);
    var a3s: ds = ds_mul(ds_mul(a2, a), s);
    var f: array<ds, 5>;
    f[0] = s;
    f[1] = ds_div(ds_f(0.5), s);
    f[2] = ds_neg(ds_div(ds_f(0.25), as_));
    f[3] = ds_div(ds_f(0.375), a2s);
    f[4] = ds_neg(ds_div(ds_f(0.9375), a3s));
    j_compose_1(r, jet, &f, p);
}

fn radial_jet_1(radius: ptr<function, Jt_1>, px: ds, py: ds, pz: ds, p: BasisParams) {
    var x: Jt_1;
    var y: Jt_1;
    var z: Jt_1;
    var x2: Jt_1;
    var y2: Jt_1;
    var z2: Jt_1;
    var r2: Jt_1;
    j_var_1(&x, 0, px);
    j_var_1(&y, 1, py);
    j_var_1(&z, 2, pz);
    j_mul_1(&x2, x, x, p);
    j_mul_1(&y2, y, y, p);
    j_mul_1(&z2, z, z, p);
    j_add3_1(&r2, x2, y2, z2);
    j_sqrt_1(radius, r2, p);
}

fn j_recip_1(r: ptr<function, Jt_1>, jet: Jt_1, p: BasisParams) {
    var a: ds = jet.c[0];
    var a2: ds = ds_sqr(a);
    var a3: ds = ds_mul(a2, a);
    var a4: ds = ds_sqr(a2);
    var a5: ds = ds_mul(a4, a);
    var f: array<ds, 5>;
    f[0] = ds_recip(a);
    f[1] = ds_neg(ds_recip(a2));
    f[2] = ds_mul(ds_f(2.0), ds_recip(a3));
    f[3] = ds_neg(ds_mul(ds_f(6.0), ds_recip(a4)));
    f[4] = ds_mul(ds_f(24.0), ds_recip(a5));
    j_compose_1(r, jet, &f, p);
}

fn unit_frame_1(
    u: ptr<function, array<Jt_1, 3>>,
    radius: ptr<function, Jt_1>,
    px: ds,
    py: ds,
    pz: ds,
    p: BasisParams,
) {
    var x: Jt_1;
    var y: Jt_1;
    var z: Jt_1;
    var inv: Jt_1;
    j_var_1(&x, 0, px);
    j_var_1(&y, 1, py);
    j_var_1(&z, 2, pz);
    radial_jet_1(radius, px, py, pz, p);
    j_recip_1(&inv, *radius, p);
    j_mul_1(&((*u)[0]), x, inv, p);
    j_mul_1(&((*u)[1]), y, inv, p);
    j_mul_1(&((*u)[2]), z, inv, p);
}

fn j_sub_1(r: ptr<function, Jt_1>, a: Jt_1, b: Jt_1) {
    for(var i: i32 = 0; i < 4; i = i + 1) {
        (*r).c[i] = ds_subp(a.c[i], b.c[i]);
    }
    for(var i: i32 = 4; i < 4; i = i + 1) {
        (*r).c[i] = ds_zero();
    }
}

fn j_add_1(r: ptr<function, Jt_1>, a: Jt_1, b: Jt_1) {
    for(var i: i32 = 0; i < 4; i = i + 1) {
        (*r).c[i] = ds_addp(a.c[i], b.c[i]);
    }
    for(var i: i32 = 4; i < 4; i = i + 1) {
        (*r).c[i] = ds_zero();
    }
}

fn cmul_u_1(
    cre: ptr<function, Jt_1>,
    cim: ptr<function, Jt_1>,
    u: ptr<function, array<Jt_1, 3>>,
    p: BasisParams,
) {
    var t1: Jt_1;
    var t2: Jt_1;
    var nre: Jt_1;
    var nim: Jt_1;
    j_mul_1(&(t1), *cre, (*u)[0], p);
    j_mul_1(&(t2), *cim, (*u)[1], p);
    j_sub_1(&nre, t1, t2);
    j_mul_1(&(t1), *cre, (*u)[1], p);
    j_mul_1(&(t2), *cim, (*u)[0], p);
    j_add_1(&nim, t1, t2);
    *cre = nre;
    *cim = nim;
}

fn j_scale_1(r: ptr<function, Jt_1>, a: Jt_1, s: ds) {
    for(var i: i32 = 0; i < 4; i = i + 1) {
        (*r).c[i] = ds_mul(a.c[i], s);
    }
    for(var i: i32 = 4; i < 4; i = i + 1) {
        (*r).c[i] = ds_zero();
    }
}

fn legendre_p_1(
    p: ptr<function, Jt_1>,
    l: i32,
    m: i32,
    p1: Jt_1,
    p2: Jt_1,
    uz: Jt_1,
    pmm: ds,
    prm: BasisParams,
) {
    if l == m {
        j_const_1(p, pmm);
        return;
    }
    var a: ds = coef_a(l, m);
    var zp: Jt_1;
    j_mul_1(&zp, uz, p1, prm);
    if l == m + 1 {
        j_scale_1(p, zp, a);
        return;
    }
    var bp: Jt_1;
    var d: Jt_1;
    j_scale_1(&bp, p2, coef_b(l, m));
    j_sub_1(&d, zp, bp);
    j_scale_1(p, d, a);
}

fn j_axpy_1(r: ptr<function, Jt_1>, s: ds, a: Jt_1) {
    for(var i: i32 = 0; i < 4; i = i + 1) {
        (*r).c[i] = ds_addp((*r).c[i], ds_mul(s, a.c[i]));
    }
}

fn sh_accumulate_1__saved(
    rr: ptr<function, Jt_1>,
    l_save: i32,
    l_top: i32,
    l_surf: i32,
    u: ptr<function, array<Jt_1, 3>>,
    sqrt2: ds,
    pmm0: ds,
    prm: BasisParams,
) {
    j_const_1(rr, ds_zero());
    var cre: Jt_1;
    var cim: Jt_1;
    j_const_1(&cre, ds_f(1.0));
    j_const_1(&cim, ds_zero());
    var pmm: ds = pmm0;
    for(var m: i32 = 0; m <= l_top; m = m + 1) {
        if m > 0 {
            cmul_u_1(&cre, &cim, u, prm);
            pmm = ds_mul(pmm, ds_sqrt(ds_div(ds_f(f32(2 * m + 1)), ds_f(f32(2 * m)))));
        }
        var p1: Jt_1;
        var p2: Jt_1;
        j_const_1(&p1, ds_zero());
        j_const_1(&p2, ds_zero());
        for(var l: i32 = m; l <= l_top; l = l + 1) {
            var p: Jt_1;
            var yp: Jt_1;
            var ym: Jt_1;
            var t: Jt_1;
            legendre_p_1(&(p), l, m, p1, p2, (*u)[2], pmm, prm);
            if m == 0 {
                yp = p;
            } else {
                j_mul_1(&t, p, cre, prm);
                j_scale_1(&yp, t, sqrt2);
                j_mul_1(&t, p, cim, prm);
                j_scale_1(&ym, t, sqrt2);
            }
            if l <= l_surf {
                var cy: ds = coeff_at(l, m);
                if cy.hi != 0.0 || cy.lo != 0.0 {
                    j_axpy_1(rr, cy, yp);
                }
                if m > 0 {
                    var cz: ds = coeff_at(l, -m);
                    if cz.hi != 0.0 || cz.lo != 0.0 {
                        j_axpy_1(rr, cz, ym);
                    }
                }
            }
            p2 = p1;
            p1 = p;
        }
    }
}

fn fill_powers_1(pows: ptr<function, array<Jt_1, 33>>, zeta: Jt_1, top: u32, p: BasisParams) {
    j_const_1(&((*pows)[0]), ds_f(1.0));
    for(var n: u32 = 1; n <= top; n = n + 1u) {
        j_mul_1(&((*pows)[n]), (*pows)[n - 1], zeta, p);
    }
}

fn j_psi_1(s: ptr<function, Jt_1>, base: u32, len: u32, pows: ptr<function, array<Jt_1, 33>>) {
    j_const_1(s, ds_zero());
    for(var j: u32 = 0; j < len; j = j + 1u) {
        var a: ds = ds_mk(polys[base + j].x, polys[base + j].y);
        if a.hi != 0.0 || a.lo != 0.0 {
            j_axpy_1(s, a, (*pows)[j]);
        }
    }
}

fn eval_pres1(gid: u32, prm: BasisParams) {
    if gid >= prm.n_points {
        return;
    }
    var sqrt2: ds = ds_mk(prm.sqrt2_hi, prm.sqrt2_lo);
    var pmm0: ds = ds_mk(prm.pmm_hi, prm.pmm_lo);
    var px: ds;
    var py: ds;
    var pz: ds;
    load_point(gid, &px, &py, &pz);
    var u: array<Jt_1, 3>;
    var radius: Jt_1;
    var rr: Jt_1;
    unit_frame_1(&(u), &(radius), px, py, pz, prm);
    var l_mode: i32 = i32(prm.l_mode);
    var l_surf: i32 = i32(prm.l_surf);
    var l_top: i32 = select(l_surf, l_mode, l_mode > l_surf);
    sh_accumulate_1__saved(&(rr), 0, l_top, l_surf, &(u), sqrt2, pmm0, prm);
    var inv: Jt_1;
    var zeta: Jt_1;
    var one: Jt_1;
    j_recip_1(&(inv), rr, prm);
    j_mul_1(&(zeta), radius, inv, prm);
    j_scale_1(&(one), zeta, ds_f(-1.0));
    one.c[0] = ds_addp(one.c[0], ds_f(1.0));
    var pows: array<Jt_1, 33>;
    fill_powers_1(&(pows), zeta, prm.top_pow, prm);
    var cre: Jt_1;
    var cim: Jt_1;
    j_const_1(&(cre), ds_f(1.0));
    j_const_1(&(cim), ds_zero());
    var pmm: ds = pmm0;
    var n: u32 = prm.n_points;
    var np: u32 = prm.n_out;
    for(var m: i32 = 0; m <= l_mode; m = m + 1) {
        if m > 0 {
            cmul_u_1(&(cre), &(cim), &(u), prm);
            pmm = ds_mul(pmm, ds_sqrt(ds_div(ds_f(f32(2 * m + 1)), ds_f(f32(2 * m)))));
        }
        var p1: Jt_1;
        var p2: Jt_1;
        j_const_1(&(p1), ds_zero());
        j_const_1(&(p2), ds_zero());
        for(var l: i32 = m; l <= l_mode; l = l + 1) {
            var p: Jt_1;
            var yp: Jt_1;
            var ym: Jt_1;
            var t: Jt_1;
            legendre_p_1(&(p), l, m, p1, p2, u[2], pmm, prm);
            if m == 0 {
                yp = p;
            } else {
                j_mul_1(&(t), p, cre, prm);
                j_scale_1(&(yp), t, sqrt2);
                j_mul_1(&(t), p, cim, prm);
                j_scale_1(&(ym), t, sqrt2);
            }
            for(var sweep: i32 = 0; sweep < select(2, 1, m == 0); sweep = sweep + 1) {
                var mm: i32 = select(-m, m, sweep == 0);
                var ylm: Jt_1 = ym;
                if sweep == 0 {
                    ylm = yp;
                }
                var lift: Jt_1;
                j_mul_1(&(lift), pows[u32(l)], ylm, prm);
                var lm: u32 = lm_of(l, mm);
                var lift_ch: u32 = (np + lm) * 4u;
                store_ds(lift_ch, n, gid, lift.c[0]);
                store_ds(lift_ch + 1u, n, gid, lift.c[1]);
                store_ds(lift_ch + 2u, n, gid, lift.c[2]);
                store_ds(lift_ch + 3u, n, gid, lift.c[3]);
                for(var k: u32 = 0; k < prm.k_count; k = k + 1u) {
                    var psi: Jt_1;
                    var mid: Jt_1;
                    var phi: Jt_1;
                    let poly_base = (u32(l) * prm.k_count + k) * prm.poly_stride;
                    j_psi_1(
                        &(psi),
                        poly_base,
                        select(u32(l) + k + 1u, 2u * k + 2u, l == 0),
                        &(pows),
                    );
                    j_mul_1(&(mid), one, psi, prm);
                    j_mul_1(&(phi), mid, ylm, prm);
                    var ch: u32 = (lm * prm.k_count + k) * 4u;
                    store_ds(ch, n, gid, phi.c[0]);
                    store_ds(ch + 1u, n, gid, phi.c[1]);
                    store_ds(ch + 2u, n, gid, phi.c[2]);
                    store_ds(ch + 3u, n, gid, phi.c[3]);
                }
            }
            p2 = p1;
            p1 = p;
        }
    }
}

fn j_zero_3(r: ptr<function, Jt_3>) {
    for(var i: i32 = 0; i < 20; i = i + 1) {
        (*r).c[i] = ds_zero();
    }
}

fn j_const_3(r: ptr<function, Jt_3>, v: ds) {
    j_zero_3(r);
    (*r).c[0] = v;
}

fn j_var_3(r: ptr<function, Jt_3>, axis: i32, x0: ds) {
    j_const_3(r, x0);
    if 3 >= 1 {
        (*r).c[1 + axis] = ds_f(1.0);
    }
}

fn mul_tab_3(p: BasisParams) -> u32 {
    if 3 <= 0 {
        return p.moff0;
    }
    if 3 == 1 {
        return p.moff1;
    }
    if 3 == 2 {
        return p.moff2;
    }
    if 3 == 3 {
        return p.moff3;
    }
    return p.moff4;
}

fn mul_count_3(p: BasisParams) -> u32 {
    if 3 <= 0 {
        return p.nmul0;
    }
    if 3 == 1 {
        return p.nmul1;
    }
    if 3 == 2 {
        return p.nmul2;
    }
    if 3 == 3 {
        return p.nmul3;
    }
    return p.nmul4;
}

fn j_mul_3(r: ptr<function, Jt_3>, a: Jt_3, b: Jt_3, p: BasisParams) {
    var acc: array<dsacc, 35>;
    for(var i: i32 = 0; i < 20; i = i + 1) {
        acc[i] = acc_zero();
    }
    let tab_base = mul_tab_3(p);
    var n: u32 = mul_count_3(p);
    for(var t: u32 = 0; t < n; t = t + 1u) {
        var e: u32 = tables[tab_base + t];
        acc[(e >> 16) & 255u] = acc_fma(acc[(e >> 16) & 255u], a.c[e & 255u], b.c[(e >> 8) & 255u]);
    }
    for(var i: i32 = 0; i < 20; i = i + 1) {
        (*r).c[i] = acc_done(acc[i]);
    }
}

fn j_add3_3(r: ptr<function, Jt_3>, a: Jt_3, b: Jt_3, c: Jt_3) {
    for(var i: i32 = 0; i < 20; i = i + 1) {
        (*r).c[i] = ds_addp(ds_addp(a.c[i], b.c[i]), c.c[i]);
    }
    for(var i: i32 = 20; i < 20; i = i + 1) {
        (*r).c[i] = ds_zero();
    }
}

fn j_compose_3(r: ptr<function, Jt_3>, jet: Jt_3, f: ptr<function, array<ds, 5>>, p: BasisParams) {
    var delta: Jt_3 = jet;
    delta.c[0] = ds_zero();
    var fact: ds = ds_f(1.0);
    var coef: array<ds, 5>;
    for(var k: i32 = 0; k <= 3; k = k + 1) {
        if k > 0 {
            fact = ds_mul(fact, ds_f(f32(k)));
        }
        coef[k] = ds_div((*f)[k], fact);
    }
    j_const_3(r, coef[3]);
    for(var k: i32 = i32(3) - 1; k >= 0; k = k - 1) {
        var tmp: Jt_3;
        j_mul_3(&tmp, *r, delta, p);
        tmp.c[0] = ds_addp(tmp.c[0], coef[k]);
        *r = tmp;
    }
}

fn j_sqrt_3(r: ptr<function, Jt_3>, jet: Jt_3, p: BasisParams) {
    var a: ds = jet.c[0];
    var s: ds = ds_sqrt(a);
    var as_: ds = ds_mul(a, s);
    var a2: ds = ds_sqr(a);
    var a2s: ds = ds_mul(a2, s);
    var a3s: ds = ds_mul(ds_mul(a2, a), s);
    var f: array<ds, 5>;
    f[0] = s;
    f[1] = ds_div(ds_f(0.5), s);
    f[2] = ds_neg(ds_div(ds_f(0.25), as_));
    f[3] = ds_div(ds_f(0.375), a2s);
    f[4] = ds_neg(ds_div(ds_f(0.9375), a3s));
    j_compose_3(r, jet, &f, p);
}

fn radial_jet_3(radius: ptr<function, Jt_3>, px: ds, py: ds, pz: ds, p: BasisParams) {
    var x: Jt_3;
    var y: Jt_3;
    var z: Jt_3;
    var x2: Jt_3;
    var y2: Jt_3;
    var z2: Jt_3;
    var r2: Jt_3;
    j_var_3(&x, 0, px);
    j_var_3(&y, 1, py);
    j_var_3(&z, 2, pz);
    j_mul_3(&x2, x, x, p);
    j_mul_3(&y2, y, y, p);
    j_mul_3(&z2, z, z, p);
    j_add3_3(&r2, x2, y2, z2);
    j_sqrt_3(radius, r2, p);
}

fn j_recip_3(r: ptr<function, Jt_3>, jet: Jt_3, p: BasisParams) {
    var a: ds = jet.c[0];
    var a2: ds = ds_sqr(a);
    var a3: ds = ds_mul(a2, a);
    var a4: ds = ds_sqr(a2);
    var a5: ds = ds_mul(a4, a);
    var f: array<ds, 5>;
    f[0] = ds_recip(a);
    f[1] = ds_neg(ds_recip(a2));
    f[2] = ds_mul(ds_f(2.0), ds_recip(a3));
    f[3] = ds_neg(ds_mul(ds_f(6.0), ds_recip(a4)));
    f[4] = ds_mul(ds_f(24.0), ds_recip(a5));
    j_compose_3(r, jet, &f, p);
}

fn unit_frame_3(
    u: ptr<function, array<Jt_3, 3>>,
    radius: ptr<function, Jt_3>,
    px: ds,
    py: ds,
    pz: ds,
    p: BasisParams,
) {
    var x: Jt_3;
    var y: Jt_3;
    var z: Jt_3;
    var inv: Jt_3;
    j_var_3(&x, 0, px);
    j_var_3(&y, 1, py);
    j_var_3(&z, 2, pz);
    radial_jet_3(radius, px, py, pz, p);
    j_recip_3(&inv, *radius, p);
    j_mul_3(&((*u)[0]), x, inv, p);
    j_mul_3(&((*u)[1]), y, inv, p);
    j_mul_3(&((*u)[2]), z, inv, p);
}

fn j_sub_3(r: ptr<function, Jt_3>, a: Jt_3, b: Jt_3) {
    for(var i: i32 = 0; i < 20; i = i + 1) {
        (*r).c[i] = ds_subp(a.c[i], b.c[i]);
    }
    for(var i: i32 = 20; i < 20; i = i + 1) {
        (*r).c[i] = ds_zero();
    }
}

fn j_add_3(r: ptr<function, Jt_3>, a: Jt_3, b: Jt_3) {
    for(var i: i32 = 0; i < 20; i = i + 1) {
        (*r).c[i] = ds_addp(a.c[i], b.c[i]);
    }
    for(var i: i32 = 20; i < 20; i = i + 1) {
        (*r).c[i] = ds_zero();
    }
}

fn cmul_u_3(
    cre: ptr<function, Jt_3>,
    cim: ptr<function, Jt_3>,
    u: ptr<function, array<Jt_3, 3>>,
    p: BasisParams,
) {
    var t1: Jt_3;
    var t2: Jt_3;
    var nre: Jt_3;
    var nim: Jt_3;
    j_mul_3(&(t1), *cre, (*u)[0], p);
    j_mul_3(&(t2), *cim, (*u)[1], p);
    j_sub_3(&nre, t1, t2);
    j_mul_3(&(t1), *cre, (*u)[1], p);
    j_mul_3(&(t2), *cim, (*u)[0], p);
    j_add_3(&nim, t1, t2);
    *cre = nre;
    *cim = nim;
}

fn j_scale_3(r: ptr<function, Jt_3>, a: Jt_3, s: ds) {
    for(var i: i32 = 0; i < 20; i = i + 1) {
        (*r).c[i] = ds_mul(a.c[i], s);
    }
    for(var i: i32 = 20; i < 20; i = i + 1) {
        (*r).c[i] = ds_zero();
    }
}

fn legendre_p_3(
    p: ptr<function, Jt_3>,
    l: i32,
    m: i32,
    p1: Jt_3,
    p2: Jt_3,
    uz: Jt_3,
    pmm: ds,
    prm: BasisParams,
) {
    if l == m {
        j_const_3(p, pmm);
        return;
    }
    var a: ds = coef_a(l, m);
    var zp: Jt_3;
    j_mul_3(&zp, uz, p1, prm);
    if l == m + 1 {
        j_scale_3(p, zp, a);
        return;
    }
    var bp: Jt_3;
    var d: Jt_3;
    j_scale_3(&bp, p2, coef_b(l, m));
    j_sub_3(&d, zp, bp);
    j_scale_3(p, d, a);
}

fn j_axpy_3(r: ptr<function, Jt_3>, s: ds, a: Jt_3) {
    for(var i: i32 = 0; i < 20; i = i + 1) {
        (*r).c[i] = ds_addp((*r).c[i], ds_mul(s, a.c[i]));
    }
}

fn sh_accumulate_3(
    rr: ptr<function, Jt_3>,
    saved: ptr<function, array<Jt_3, 25>>,
    l_save: i32,
    l_top: i32,
    l_surf: i32,
    u: ptr<function, array<Jt_3, 3>>,
    sqrt2: ds,
    pmm0: ds,
    prm: BasisParams,
) {
    j_const_3(rr, ds_zero());
    var cre: Jt_3;
    var cim: Jt_3;
    j_const_3(&cre, ds_f(1.0));
    j_const_3(&cim, ds_zero());
    var pmm: ds = pmm0;
    for(var m: i32 = 0; m <= l_top; m = m + 1) {
        if m > 0 {
            cmul_u_3(&cre, &cim, u, prm);
            pmm = ds_mul(pmm, ds_sqrt(ds_div(ds_f(f32(2 * m + 1)), ds_f(f32(2 * m)))));
        }
        var p1: Jt_3;
        var p2: Jt_3;
        j_const_3(&p1, ds_zero());
        j_const_3(&p2, ds_zero());
        for(var l: i32 = m; l <= l_top; l = l + 1) {
            var p: Jt_3;
            var yp: Jt_3;
            var ym: Jt_3;
            var t: Jt_3;
            legendre_p_3(&(p), l, m, p1, p2, (*u)[2], pmm, prm);
            if m == 0 {
                yp = p;
            } else {
                j_mul_3(&t, p, cre, prm);
                j_scale_3(&yp, t, sqrt2);
                j_mul_3(&t, p, cim, prm);
                j_scale_3(&ym, t, sqrt2);
            }
            if l <= l_surf {
                var cy: ds = coeff_at(l, m);
                if cy.hi != 0.0 || cy.lo != 0.0 {
                    j_axpy_3(rr, cy, yp);
                }
                if m > 0 {
                    var cz: ds = coeff_at(l, -m);
                    if cz.hi != 0.0 || cz.lo != 0.0 {
                        j_axpy_3(rr, cz, ym);
                    }
                }
            }
            if true && l >= 1 && l <= l_save {
                (*saved)[lm_of(l, m)] = yp;
                if m > 0 {
                    (*saved)[lm_of(l, -m)] = ym;
                }
            }
            p2 = p1;
            p1 = p;
        }
    }
}

fn fill_powers_3(pows: ptr<function, array<Jt_3, 12>>, zeta: Jt_3, top: u32, p: BasisParams) {
    j_const_3(&((*pows)[0]), ds_f(1.0));
    for(var n: u32 = 1; n <= top; n = n + 1u) {
        j_mul_3(&((*pows)[n]), (*pows)[n - 1], zeta, p);
    }
}

fn j_psi_3(s: ptr<function, Jt_3>, base: u32, len: u32, pows: ptr<function, array<Jt_3, 12>>) {
    j_const_3(s, ds_zero());
    for(var j: u32 = 0; j < len; j = j + 1u) {
        var a: ds = ds_mk(polys[base + j].x, polys[base + j].y);
        if a.hi != 0.0 || a.lo != 0.0 {
            j_axpy_3(s, a, (*pows)[j]);
        }
    }
}

fn high_apply_0_1(
    f: ptr<function, array<Jt_1, 3>>,
    mode: u32,
    point: u32,
    p: BasisParams,
    sqrt2: ds,
) {}

fn j_zero_2(r: ptr<function, Jt_2>) {
    for(var i: i32 = 0; i < 10; i = i + 1) {
        (*r).c[i] = ds_zero();
    }
}

fn j_deriv_3(r: ptr<function, Jt_2>, a: Jt_3, axis: i32, p: BasisParams) {
    let up_base = p.up_off;
    let mono_base = p.mono_off;
    j_zero_2(r);
    for(var i: i32 = 0; i < 10; i = i + 1) {
        (*r).c[i] = ds_mulf(
                f32(tables[mono_base + u32(i) * 3u + u32(axis)] + 1u),
                a.c[tables[up_base + u32(axis) * 35u + u32(i)]],
            );
    }
}

fn j_mul_x_2(r: ptr<function, Jt_2>, a: Jt_2, axis: i32, x0: ds, p: BasisParams) {
    for(var i: i32 = 0; i < 10; i = i + 1) {
        (*r).c[i] = ds_mul(a.c[i], x0);
    }
    for(var i: i32 = 10; i < 10; i = i + 1) {
        (*r).c[i] = ds_zero();
    }
    let up_base = p.up_off;
    let deg_base = p.deg_off;
    for(var i: i32 = 0; i < 10; i = i + 1) {
        if tables[deg_base + u32(i)] < u32(2) {
            var dst: u32 = tables[up_base + u32(axis) * 35u + u32(i)];
            (*r).c[dst] = ds_addp((*r).c[dst], a.c[i]);
        }
    }
}

fn j_sub_2(r: ptr<function, Jt_2>, a: Jt_2, b: Jt_2) {
    for(var i: i32 = 0; i < 10; i = i + 1) {
        (*r).c[i] = ds_subp(a.c[i], b.c[i]);
    }
    for(var i: i32 = 10; i < 10; i = i + 1) {
        (*r).c[i] = ds_zero();
    }
}

fn j_trunc_2_1(r: ptr<function, Jt_1>, a: Jt_2) {
    for(var i: i32 = 0; i < 4; i = i + 1) {
        (*r).c[i] = a.c[i];
    }
    for(var i: i32 = 4; i < 4; i = i + 1) {
        (*r).c[i] = ds_zero();
    }
}

fn toroidal_3(
    out: ptr<function, array<Jt_1, 3>>,
    psi: Jt_3,
    x0: ptr<function, array<ds, 3>>,
    p: BasisParams,
) {
    var g: array<Jt_2, 3>;
    for(var a: i32 = 0; a < 3; a = a + 1) {
        j_deriv_3(&(g[a]), psi, a, p);
    }
    var a: Jt_2;
    var b: Jt_2;
    var cross: Jt_2;
    j_mul_x_2(&(a), g[1], 2, (*x0)[2], p);
    j_mul_x_2(&(b), g[2], 1, (*x0)[1], p);
    j_sub_2(&cross, a, b);
    j_trunc_2_1(&((*out)[0]), cross);
    j_mul_x_2(&(a), g[2], 0, (*x0)[0], p);
    j_mul_x_2(&(b), g[0], 2, (*x0)[2], p);
    j_sub_2(&cross, a, b);
    j_trunc_2_1(&((*out)[1]), cross);
    j_mul_x_2(&(a), g[0], 1, (*x0)[1], p);
    j_mul_x_2(&(b), g[1], 0, (*x0)[0], p);
    j_sub_2(&cross, a, b);
    j_trunc_2_1(&((*out)[2]), cross);
}

fn j_trunc_3_2(r: ptr<function, Jt_2>, a: Jt_3) {
    for(var i: i32 = 0; i < 10; i = i + 1) {
        (*r).c[i] = a.c[i];
    }
    for(var i: i32 = 10; i < 10; i = i + 1) {
        (*r).c[i] = ds_zero();
    }
}

fn j_add_into_2(r: ptr<function, Jt_2>, a: Jt_2) {
    for(var i: i32 = 0; i < 10; i = i + 1) {
        (*r).c[i] = ds_addp((*r).c[i], a.c[i]);
    }
}

fn j_deriv_2(r: ptr<function, Jt_1>, a: Jt_2, axis: i32, p: BasisParams) {
    let up_base = p.up_off;
    let mono_base = p.mono_off;
    j_zero_1(r);
    for(var i: i32 = 0; i < 4; i = i + 1) {
        (*r).c[i] = ds_mulf(
                f32(tables[mono_base + u32(i) * 3u + u32(axis)] + 1u),
                a.c[tables[up_base + u32(axis) * 35u + u32(i)]],
            );
    }
}

fn j_mul_x_1(r: ptr<function, Jt_1>, a: Jt_1, axis: i32, x0: ds, p: BasisParams) {
    for(var i: i32 = 0; i < 4; i = i + 1) {
        (*r).c[i] = ds_mul(a.c[i], x0);
    }
    for(var i: i32 = 4; i < 4; i = i + 1) {
        (*r).c[i] = ds_zero();
    }
    let up_base = p.up_off;
    let deg_base = p.deg_off;
    for(var i: i32 = 0; i < 4; i = i + 1) {
        if tables[deg_base + u32(i)] < u32(1) {
            var dst: u32 = tables[up_base + u32(axis) * 35u + u32(i)];
            (*r).c[dst] = ds_addp((*r).c[dst], a.c[i]);
        }
    }
}

fn poloidal_3(
    out: ptr<function, array<Jt_1, 3>>,
    phi: Jt_3,
    x0: ptr<function, array<ds, 3>>,
    p: BasisParams,
) {
    var g: array<Jt_2, 3>;
    for(var a: i32 = 0; a < 3; a = a + 1) {
        j_deriv_3(&(g[a]), phi, a, p);
    }
    var e: Jt_2;
    var mx: Jt_2;
    j_trunc_3_2(&e, phi);
    for(var a: i32 = 0; a < 3; a = a + 1) {
        j_mul_x_2(&(mx), g[a], a, (*x0)[a], p);
        j_add_into_2(&e, mx);
    }
    var d0: Jt_1;
    var d1: Jt_1;
    var d2: Jt_1;
    var lap: Jt_1;
    j_deriv_2(&d0, g[0], 0, p);
    j_deriv_2(&d1, g[1], 1, p);
    j_deriv_2(&d2, g[2], 2, p);
    j_add3_1(&lap, d0, d1, d2);
    for(var a: i32 = 0; a < 3; a = a + 1) {
        var de: Jt_1;
        var lx: Jt_1;
        j_deriv_2(&de, e, a, p);
        j_mul_x_1(&(lx), lap, a, (*x0)[a], p);
        j_sub_1(&((*out)[a]), de, lx);
    }
}

fn pack_d_1(f: ptr<function, array<Jt_1, 3>>, d: ptr<function, array<ds, 6>>, sqrt2: ds) {
    var g00: ds = (*f)[0].c[1];
    var g01: ds = (*f)[0].c[2];
    var g02: ds = (*f)[0].c[3];
    var g10: ds = (*f)[1].c[1];
    var g11: ds = (*f)[1].c[2];
    var g12: ds = (*f)[1].c[3];
    var g20: ds = (*f)[2].c[1];
    var g21: ds = (*f)[2].c[2];
    var g22: ds = (*f)[2].c[3];
    (*d)[0] = ds_scale(ds_addp(g00, g00), 0.5);
    (*d)[1] = ds_scale(ds_addp(g11, g11), 0.5);
    (*d)[2] = ds_scale(ds_addp(g22, g22), 0.5);
    (*d)[3] = ds_mul(sqrt2, ds_scale(ds_addp(g01, g10), 0.5));
    (*d)[4] = ds_mul(sqrt2, ds_scale(ds_addp(g02, g20), 0.5));
    (*d)[5] = ds_mul(sqrt2, ds_scale(ds_addp(g12, g21), 0.5));
}

fn store_vd_1(f: ptr<function, array<Jt_1, 3>>, mode: u32, point: u32, p: BasisParams, sqrt2: ds) {
    var n: u32 = p.n_points;
    var local: u32 = 0;
    if p.want_v != 0u {
        for(var a: i32 = 0; a < 3; a = a + 1) {
            store_ds(mode * p.n_comp + local + u32(a), n, point, (*f)[a].c[0]);
        }
        local = local + (3u);
    }
    if p.want_d != 0u {
        var d: array<ds, 6>;
        pack_d_1(f, &d, sqrt2);
        for(var a: u32 = 0; a < 6u; a = a + 1u) {
            store_ds(mode * p.n_comp + local + a, n, point, d[a]);
        }
    }
}

fn emit_mode_3__acc(
    radial: Jt_3,
    ylm: Jt_3,
    tor: u32,
    x0: ptr<function, array<ds, 3>>,
    l: i32,
    m: i32,
    k: u32,
    point: u32,
    p: BasisParams,
    sqrt2: ds,
) {
    var phi: Jt_3;
    j_mul_3(&phi, radial, ylm, p);
    var field: array<Jt_1, 3>;
    if tor != 0u {
        toroidal_3(&field, phi, x0, p);
    } else {
        poloidal_3(&field, phi, x0, p);
    }
    var mode: u32 = vel_dof(l, m, tor, k, p.k_count);
    store_vd_1(&field, mode, point, p, sqrt2);
    high_apply_0_1(&field, mode, point, p, sqrt2);
}

fn eval_cached_3_2__coef(ray: u32, prm: BasisParams) {
    if ray >= prm.n_rays {
        return;
    }
    var sqrt2: ds = ds_mk(prm.sqrt2_hi, prm.sqrt2_lo);
    var pmm0: ds = ds_mk(prm.pmm_hi, prm.pmm_lo);
    var p0: u32 = ray * prm.n_per;
    var x0: ds;
    var y0: ds;
    var z0: ds;
    load_point(p0, &x0, &y0, &z0);
    var u: array<Jt_3, 3>;
    var radius0: Jt_3;
    unit_frame_3(&(u), &(radius0), x0, y0, z0, prm);
    var l_mode: i32 = i32(prm.l_mode);
    var l_surf: i32 = i32(prm.l_surf);
    var l_top: i32 = select(l_surf, l_mode, l_mode > l_surf);
    var saved: array<Jt_3, 25>;
    var rr: Jt_3;
    sh_accumulate_3(&rr, &saved, l_mode, l_top, l_surf, &(u), sqrt2, pmm0, prm);
    var rho0: ds = radius0.c[0];
    let deg_base = prm.deg_off;
    for(var s: u32 = 0; s < prm.n_per; s = s + 1u) {
        var px: ds;
        var py: ds;
        var pz: ds;
        load_point(p0 + s, &px, &py, &pz);
        var xyz: array<ds, 3> = array<ds, 3>(px, py, pz);
        var rs: Jt_3 = radius0;
        if s != 0u {
            radial_jet_3(&rs, px, py, pz, prm);
        }
        var rho: ds = rs.c[0];
        var factor: array<ds, 5>;
        factor[0] = ds_f(1.0);
        if s != 0u {
            var ratio: ds = ds_div(rho0, rho);
            for(var d: i32 = 1; d <= 3; d = d + 1) {
                factor[d] = ds_mul(factor[d - 1], ratio);
            }
        }
        var rad_s: Jt_3 = rr;
        if s != 0u {
            for(var i: i32 = 0; i < 20; i = i + 1) {
                rad_s.c[i] = ds_mul(rr.c[i], factor[tables[deg_base + u32(i)]]);
            }
        }
        var inv: Jt_3;
        var zeta: Jt_3;
        var r2: Jt_3;
        j_recip_3(&inv, rad_s, prm);
        j_mul_3(&zeta, rs, inv, prm);
        j_mul_3(&r2, rad_s, rad_s, prm);
        var pows: array<Jt_3, 12>;
        fill_powers_3(&pows, zeta, prm.top_pow, prm);
        var alk: array<Jt_3, 8>;
        var blk: array<Jt_3, 8>;
        for(var l: i32 = 1; l <= l_mode; l = l + 1) {
            for(var k: u32 = 0; k < prm.k_count; k = k + 1u) {
                var psi: Jt_3;
                let poly_base = (u32(l) * prm.k_count + k) * prm.poly_stride;
                j_psi_3(&psi, poly_base, u32(l) + k + 1u, &pows);
                var slot: u32 = u32(l - 1) * prm.k_count + k;
                j_mul_3(&(alk[slot]), r2, psi, prm);
                j_mul_3(&(blk[slot]), rad_s, psi, prm);
            }
        }
        var point: u32 = p0 + s;
        var acc: array<dsacc, 9>;
        for(var l: i32 = 1; l <= l_mode; l = l + 1) {
            for(var m: i32 = -l; m <= l; m = m + 1) {
                var ylm: Jt_3;
                if s == 0u {
                    ylm = saved[lm_of(l, m)];
                } else {
                    var src: Jt_3 = saved[lm_of(l, m)];
                    for(var i: i32 = 0; i < 20; i = i + 1) {
                        ylm.c[i] = ds_mul(src.c[i], factor[tables[deg_base + u32(i)]]);
                    }
                    for(var i: i32 = 20; i < 20; i = i + 1) {
                        ylm.c[i] = ds_zero();
                    }
                }
                for(var k: u32 = 0; k < prm.k_count; k = k + 1u) {
                    var slot: u32 = u32(l - 1) * prm.k_count + k;
                    emit_mode_3__acc(alk[slot], ylm, 0u, &xyz, l, m, k, point, prm, sqrt2);
                    emit_mode_3__acc(blk[slot], ylm, 1u, &xyz, l, m, k, point, prm, sqrt2);
                }
            }
        }
    }
}

fn eval_cached_3_4__coef(ray: u32, prm: BasisParams) {
    if ray >= prm.n_rays {
        return;
    }
    var sqrt2: ds = ds_mk(prm.sqrt2_hi, prm.sqrt2_lo);
    var pmm0: ds = ds_mk(prm.pmm_hi, prm.pmm_lo);
    var p0: u32 = ray * prm.n_per;
    var x0: ds;
    var y0: ds;
    var z0: ds;
    load_point(p0, &x0, &y0, &z0);
    var u: array<Jt_3, 3>;
    var radius0: Jt_3;
    unit_frame_3(&(u), &(radius0), x0, y0, z0, prm);
    var l_mode: i32 = i32(prm.l_mode);
    var l_surf: i32 = i32(prm.l_surf);
    var l_top: i32 = select(l_surf, l_mode, l_mode > l_surf);
    var saved: array<Jt_3, 25>;
    var rr: Jt_3;
    sh_accumulate_3(&rr, &saved, l_mode, l_top, l_surf, &(u), sqrt2, pmm0, prm);
    var rho0: ds = radius0.c[0];
    let deg_base = prm.deg_off;
    for(var s: u32 = 0; s < prm.n_per; s = s + 1u) {
        var px: ds;
        var py: ds;
        var pz: ds;
        load_point(p0 + s, &px, &py, &pz);
        var xyz: array<ds, 3> = array<ds, 3>(px, py, pz);
        var rs: Jt_3 = radius0;
        if s != 0u {
            radial_jet_3(&rs, px, py, pz, prm);
        }
        var rho: ds = rs.c[0];
        var factor: array<ds, 5>;
        factor[0] = ds_f(1.0);
        if s != 0u {
            var ratio: ds = ds_div(rho0, rho);
            for(var d: i32 = 1; d <= 3; d = d + 1) {
                factor[d] = ds_mul(factor[d - 1], ratio);
            }
        }
        var rad_s: Jt_3 = rr;
        if s != 0u {
            for(var i: i32 = 0; i < 20; i = i + 1) {
                rad_s.c[i] = ds_mul(rr.c[i], factor[tables[deg_base + u32(i)]]);
            }
        }
        var inv: Jt_3;
        var zeta: Jt_3;
        var r2: Jt_3;
        j_recip_3(&inv, rad_s, prm);
        j_mul_3(&zeta, rs, inv, prm);
        j_mul_3(&r2, rad_s, rad_s, prm);
        var pows: array<Jt_3, 12>;
        fill_powers_3(&pows, zeta, prm.top_pow, prm);
        var alk: array<Jt_3, 16>;
        var blk: array<Jt_3, 16>;
        for(var l: i32 = 1; l <= l_mode; l = l + 1) {
            for(var k: u32 = 0; k < prm.k_count; k = k + 1u) {
                var psi: Jt_3;
                let poly_base = (u32(l) * prm.k_count + k) * prm.poly_stride;
                j_psi_3(&psi, poly_base, u32(l) + k + 1u, &pows);
                var slot: u32 = u32(l - 1) * prm.k_count + k;
                j_mul_3(&(alk[slot]), r2, psi, prm);
                j_mul_3(&(blk[slot]), rad_s, psi, prm);
            }
        }
        var point: u32 = p0 + s;
        var acc: array<dsacc, 9>;
        for(var l: i32 = 1; l <= l_mode; l = l + 1) {
            for(var m: i32 = -l; m <= l; m = m + 1) {
                var ylm: Jt_3;
                if s == 0u {
                    ylm = saved[lm_of(l, m)];
                } else {
                    var src: Jt_3 = saved[lm_of(l, m)];
                    for(var i: i32 = 0; i < 20; i = i + 1) {
                        ylm.c[i] = ds_mul(src.c[i], factor[tables[deg_base + u32(i)]]);
                    }
                    for(var i: i32 = 20; i < 20; i = i + 1) {
                        ylm.c[i] = ds_zero();
                    }
                }
                for(var k: u32 = 0; k < prm.k_count; k = k + 1u) {
                    var slot: u32 = u32(l - 1) * prm.k_count + k;
                    emit_mode_3__acc(alk[slot], ylm, 0u, &xyz, l, m, k, point, prm, sqrt2);
                    emit_mode_3__acc(blk[slot], ylm, 1u, &xyz, l, m, k, point, prm, sqrt2);
                }
            }
        }
    }
}

fn j_const_2(r: ptr<function, Jt_2>, v: ds) {
    j_zero_2(r);
    (*r).c[0] = v;
}

fn j_var_2(r: ptr<function, Jt_2>, axis: i32, x0: ds) {
    j_const_2(r, x0);
    if 2 >= 1 {
        (*r).c[1 + axis] = ds_f(1.0);
    }
}

fn mul_tab_2(p: BasisParams) -> u32 {
    if 2 <= 0 {
        return p.moff0;
    }
    if 2 == 1 {
        return p.moff1;
    }
    if 2 == 2 {
        return p.moff2;
    }
    if 2 == 3 {
        return p.moff3;
    }
    return p.moff4;
}

fn mul_count_2(p: BasisParams) -> u32 {
    if 2 <= 0 {
        return p.nmul0;
    }
    if 2 == 1 {
        return p.nmul1;
    }
    if 2 == 2 {
        return p.nmul2;
    }
    if 2 == 3 {
        return p.nmul3;
    }
    return p.nmul4;
}

fn j_mul_2(r: ptr<function, Jt_2>, a: Jt_2, b: Jt_2, p: BasisParams) {
    var acc: array<dsacc, 35>;
    for(var i: i32 = 0; i < 10; i = i + 1) {
        acc[i] = acc_zero();
    }
    let tab_base = mul_tab_2(p);
    var n: u32 = mul_count_2(p);
    for(var t: u32 = 0; t < n; t = t + 1u) {
        var e: u32 = tables[tab_base + t];
        acc[(e >> 16) & 255u] = acc_fma(acc[(e >> 16) & 255u], a.c[e & 255u], b.c[(e >> 8) & 255u]);
    }
    for(var i: i32 = 0; i < 10; i = i + 1) {
        (*r).c[i] = acc_done(acc[i]);
    }
}

fn j_add3_2(r: ptr<function, Jt_2>, a: Jt_2, b: Jt_2, c: Jt_2) {
    for(var i: i32 = 0; i < 10; i = i + 1) {
        (*r).c[i] = ds_addp(ds_addp(a.c[i], b.c[i]), c.c[i]);
    }
    for(var i: i32 = 10; i < 10; i = i + 1) {
        (*r).c[i] = ds_zero();
    }
}

fn j_compose_2(r: ptr<function, Jt_2>, jet: Jt_2, f: ptr<function, array<ds, 5>>, p: BasisParams) {
    var delta: Jt_2 = jet;
    delta.c[0] = ds_zero();
    var fact: ds = ds_f(1.0);
    var coef: array<ds, 5>;
    for(var k: i32 = 0; k <= 2; k = k + 1) {
        if k > 0 {
            fact = ds_mul(fact, ds_f(f32(k)));
        }
        coef[k] = ds_div((*f)[k], fact);
    }
    j_const_2(r, coef[2]);
    for(var k: i32 = i32(2) - 1; k >= 0; k = k - 1) {
        var tmp: Jt_2;
        j_mul_2(&tmp, *r, delta, p);
        tmp.c[0] = ds_addp(tmp.c[0], coef[k]);
        *r = tmp;
    }
}

fn j_sqrt_2(r: ptr<function, Jt_2>, jet: Jt_2, p: BasisParams) {
    var a: ds = jet.c[0];
    var s: ds = ds_sqrt(a);
    var as_: ds = ds_mul(a, s);
    var a2: ds = ds_sqr(a);
    var a2s: ds = ds_mul(a2, s);
    var a3s: ds = ds_mul(ds_mul(a2, a), s);
    var f: array<ds, 5>;
    f[0] = s;
    f[1] = ds_div(ds_f(0.5), s);
    f[2] = ds_neg(ds_div(ds_f(0.25), as_));
    f[3] = ds_div(ds_f(0.375), a2s);
    f[4] = ds_neg(ds_div(ds_f(0.9375), a3s));
    j_compose_2(r, jet, &f, p);
}

fn radial_jet_2(radius: ptr<function, Jt_2>, px: ds, py: ds, pz: ds, p: BasisParams) {
    var x: Jt_2;
    var y: Jt_2;
    var z: Jt_2;
    var x2: Jt_2;
    var y2: Jt_2;
    var z2: Jt_2;
    var r2: Jt_2;
    j_var_2(&x, 0, px);
    j_var_2(&y, 1, py);
    j_var_2(&z, 2, pz);
    j_mul_2(&x2, x, x, p);
    j_mul_2(&y2, y, y, p);
    j_mul_2(&z2, z, z, p);
    j_add3_2(&r2, x2, y2, z2);
    j_sqrt_2(radius, r2, p);
}

fn j_recip_2(r: ptr<function, Jt_2>, jet: Jt_2, p: BasisParams) {
    var a: ds = jet.c[0];
    var a2: ds = ds_sqr(a);
    var a3: ds = ds_mul(a2, a);
    var a4: ds = ds_sqr(a2);
    var a5: ds = ds_mul(a4, a);
    var f: array<ds, 5>;
    f[0] = ds_recip(a);
    f[1] = ds_neg(ds_recip(a2));
    f[2] = ds_mul(ds_f(2.0), ds_recip(a3));
    f[3] = ds_neg(ds_mul(ds_f(6.0), ds_recip(a4)));
    f[4] = ds_mul(ds_f(24.0), ds_recip(a5));
    j_compose_2(r, jet, &f, p);
}

fn unit_frame_2(
    u: ptr<function, array<Jt_2, 3>>,
    radius: ptr<function, Jt_2>,
    px: ds,
    py: ds,
    pz: ds,
    p: BasisParams,
) {
    var x: Jt_2;
    var y: Jt_2;
    var z: Jt_2;
    var inv: Jt_2;
    j_var_2(&x, 0, px);
    j_var_2(&y, 1, py);
    j_var_2(&z, 2, pz);
    radial_jet_2(radius, px, py, pz, p);
    j_recip_2(&inv, *radius, p);
    j_mul_2(&((*u)[0]), x, inv, p);
    j_mul_2(&((*u)[1]), y, inv, p);
    j_mul_2(&((*u)[2]), z, inv, p);
}

fn j_add_2(r: ptr<function, Jt_2>, a: Jt_2, b: Jt_2) {
    for(var i: i32 = 0; i < 10; i = i + 1) {
        (*r).c[i] = ds_addp(a.c[i], b.c[i]);
    }
    for(var i: i32 = 10; i < 10; i = i + 1) {
        (*r).c[i] = ds_zero();
    }
}

fn cmul_u_2(
    cre: ptr<function, Jt_2>,
    cim: ptr<function, Jt_2>,
    u: ptr<function, array<Jt_2, 3>>,
    p: BasisParams,
) {
    var t1: Jt_2;
    var t2: Jt_2;
    var nre: Jt_2;
    var nim: Jt_2;
    j_mul_2(&(t1), *cre, (*u)[0], p);
    j_mul_2(&(t2), *cim, (*u)[1], p);
    j_sub_2(&nre, t1, t2);
    j_mul_2(&(t1), *cre, (*u)[1], p);
    j_mul_2(&(t2), *cim, (*u)[0], p);
    j_add_2(&nim, t1, t2);
    *cre = nre;
    *cim = nim;
}

fn j_scale_2(r: ptr<function, Jt_2>, a: Jt_2, s: ds) {
    for(var i: i32 = 0; i < 10; i = i + 1) {
        (*r).c[i] = ds_mul(a.c[i], s);
    }
    for(var i: i32 = 10; i < 10; i = i + 1) {
        (*r).c[i] = ds_zero();
    }
}

fn legendre_p_2(
    p: ptr<function, Jt_2>,
    l: i32,
    m: i32,
    p1: Jt_2,
    p2: Jt_2,
    uz: Jt_2,
    pmm: ds,
    prm: BasisParams,
) {
    if l == m {
        j_const_2(p, pmm);
        return;
    }
    var a: ds = coef_a(l, m);
    var zp: Jt_2;
    j_mul_2(&zp, uz, p1, prm);
    if l == m + 1 {
        j_scale_2(p, zp, a);
        return;
    }
    var bp: Jt_2;
    var d: Jt_2;
    j_scale_2(&bp, p2, coef_b(l, m));
    j_sub_2(&d, zp, bp);
    j_scale_2(p, d, a);
}

fn j_axpy_2(r: ptr<function, Jt_2>, s: ds, a: Jt_2) {
    for(var i: i32 = 0; i < 10; i = i + 1) {
        (*r).c[i] = ds_addp((*r).c[i], ds_mul(s, a.c[i]));
    }
}

fn sh_accumulate_2__saved(
    rr: ptr<function, Jt_2>,
    l_save: i32,
    l_top: i32,
    l_surf: i32,
    u: ptr<function, array<Jt_2, 3>>,
    sqrt2: ds,
    pmm0: ds,
    prm: BasisParams,
) {
    j_const_2(rr, ds_zero());
    var cre: Jt_2;
    var cim: Jt_2;
    j_const_2(&cre, ds_f(1.0));
    j_const_2(&cim, ds_zero());
    var pmm: ds = pmm0;
    for(var m: i32 = 0; m <= l_top; m = m + 1) {
        if m > 0 {
            cmul_u_2(&cre, &cim, u, prm);
            pmm = ds_mul(pmm, ds_sqrt(ds_div(ds_f(f32(2 * m + 1)), ds_f(f32(2 * m)))));
        }
        var p1: Jt_2;
        var p2: Jt_2;
        j_const_2(&p1, ds_zero());
        j_const_2(&p2, ds_zero());
        for(var l: i32 = m; l <= l_top; l = l + 1) {
            var p: Jt_2;
            var yp: Jt_2;
            var ym: Jt_2;
            var t: Jt_2;
            legendre_p_2(&(p), l, m, p1, p2, (*u)[2], pmm, prm);
            if m == 0 {
                yp = p;
            } else {
                j_mul_2(&t, p, cre, prm);
                j_scale_2(&yp, t, sqrt2);
                j_mul_2(&t, p, cim, prm);
                j_scale_2(&ym, t, sqrt2);
            }
            if l <= l_surf {
                var cy: ds = coeff_at(l, m);
                if cy.hi != 0.0 || cy.lo != 0.0 {
                    j_axpy_2(rr, cy, yp);
                }
                if m > 0 {
                    var cz: ds = coeff_at(l, -m);
                    if cz.hi != 0.0 || cz.lo != 0.0 {
                        j_axpy_2(rr, cz, ym);
                    }
                }
            }
            p2 = p1;
            p1 = p;
        }
    }
}

fn fill_powers_2(pows: ptr<function, array<Jt_2, 12>>, zeta: Jt_2, top: u32, p: BasisParams) {
    j_const_2(&((*pows)[0]), ds_f(1.0));
    for(var n: u32 = 1; n <= top; n = n + 1u) {
        j_mul_2(&((*pows)[n]), (*pows)[n - 1], zeta, p);
    }
}

fn j_psi_2(s: ptr<function, Jt_2>, base: u32, len: u32, pows: ptr<function, array<Jt_2, 12>>) {
    j_const_2(s, ds_zero());
    for(var j: u32 = 0; j < len; j = j + 1u) {
        var a: ds = ds_mk(polys[base + j].x, polys[base + j].y);
        if a.hi != 0.0 || a.lo != 0.0 {
            j_axpy_2(s, a, (*pows)[j]);
        }
    }
}

fn high_apply_0_0(
    f: ptr<function, array<Jt_0, 3>>,
    mode: u32,
    point: u32,
    p: BasisParams,
    sqrt2: ds,
) {}

fn j_trunc_1_0(r: ptr<function, Jt_0>, a: Jt_1) {
    for(var i: i32 = 0; i < 1; i = i + 1) {
        (*r).c[i] = a.c[i];
    }
    for(var i: i32 = 1; i < 4; i = i + 1) {
        (*r).c[i] = ds_zero();
    }
}

fn toroidal_2(
    out: ptr<function, array<Jt_0, 3>>,
    psi: Jt_2,
    x0: ptr<function, array<ds, 3>>,
    p: BasisParams,
) {
    var g: array<Jt_1, 3>;
    for(var a: i32 = 0; a < 3; a = a + 1) {
        j_deriv_2(&(g[a]), psi, a, p);
    }
    var a: Jt_1;
    var b: Jt_1;
    var cross: Jt_1;
    j_mul_x_1(&(a), g[1], 2, (*x0)[2], p);
    j_mul_x_1(&(b), g[2], 1, (*x0)[1], p);
    j_sub_1(&cross, a, b);
    j_trunc_1_0(&((*out)[0]), cross);
    j_mul_x_1(&(a), g[2], 0, (*x0)[0], p);
    j_mul_x_1(&(b), g[0], 2, (*x0)[2], p);
    j_sub_1(&cross, a, b);
    j_trunc_1_0(&((*out)[1]), cross);
    j_mul_x_1(&(a), g[0], 1, (*x0)[1], p);
    j_mul_x_1(&(b), g[1], 0, (*x0)[0], p);
    j_sub_1(&cross, a, b);
    j_trunc_1_0(&((*out)[2]), cross);
}

fn j_add_into_1(r: ptr<function, Jt_1>, a: Jt_1) {
    for(var i: i32 = 0; i < 4; i = i + 1) {
        (*r).c[i] = ds_addp((*r).c[i], a.c[i]);
    }
}

fn j_zero_0(r: ptr<function, Jt_0>) {
    for(var i: i32 = 0; i < 4; i = i + 1) {
        (*r).c[i] = ds_zero();
    }
}

fn j_deriv_1(r: ptr<function, Jt_0>, a: Jt_1, axis: i32, p: BasisParams) {
    let up_base = p.up_off;
    let mono_base = p.mono_off;
    j_zero_0(r);
    for(var i: i32 = 0; i < 1; i = i + 1) {
        (*r).c[i] = ds_mulf(
                f32(tables[mono_base + u32(i) * 3u + u32(axis)] + 1u),
                a.c[tables[up_base + u32(axis) * 35u + u32(i)]],
            );
    }
}

fn j_add3_0(r: ptr<function, Jt_0>, a: Jt_0, b: Jt_0, c: Jt_0) {
    for(var i: i32 = 0; i < 1; i = i + 1) {
        (*r).c[i] = ds_addp(ds_addp(a.c[i], b.c[i]), c.c[i]);
    }
    for(var i: i32 = 1; i < 4; i = i + 1) {
        (*r).c[i] = ds_zero();
    }
}

fn j_mul_x_0(r: ptr<function, Jt_0>, a: Jt_0, axis: i32, x0: ds, p: BasisParams) {
    for(var i: i32 = 0; i < 1; i = i + 1) {
        (*r).c[i] = ds_mul(a.c[i], x0);
    }
    for(var i: i32 = 1; i < 4; i = i + 1) {
        (*r).c[i] = ds_zero();
    }
    let up_base = p.up_off;
    let deg_base = p.deg_off;
    for(var i: i32 = 0; i < 1; i = i + 1) {
        if tables[deg_base + u32(i)] < u32(0) {
            var dst: u32 = tables[up_base + u32(axis) * 35u + u32(i)];
            (*r).c[dst] = ds_addp((*r).c[dst], a.c[i]);
        }
    }
}

fn j_sub_0(r: ptr<function, Jt_0>, a: Jt_0, b: Jt_0) {
    for(var i: i32 = 0; i < 1; i = i + 1) {
        (*r).c[i] = ds_subp(a.c[i], b.c[i]);
    }
    for(var i: i32 = 1; i < 4; i = i + 1) {
        (*r).c[i] = ds_zero();
    }
}

fn poloidal_2(
    out: ptr<function, array<Jt_0, 3>>,
    phi: Jt_2,
    x0: ptr<function, array<ds, 3>>,
    p: BasisParams,
) {
    var g: array<Jt_1, 3>;
    for(var a: i32 = 0; a < 3; a = a + 1) {
        j_deriv_2(&(g[a]), phi, a, p);
    }
    var e: Jt_1;
    var mx: Jt_1;
    j_trunc_2_1(&e, phi);
    for(var a: i32 = 0; a < 3; a = a + 1) {
        j_mul_x_1(&(mx), g[a], a, (*x0)[a], p);
        j_add_into_1(&e, mx);
    }
    var d0: Jt_0;
    var d1: Jt_0;
    var d2: Jt_0;
    var lap: Jt_0;
    j_deriv_1(&d0, g[0], 0, p);
    j_deriv_1(&d1, g[1], 1, p);
    j_deriv_1(&d2, g[2], 2, p);
    j_add3_0(&lap, d0, d1, d2);
    for(var a: i32 = 0; a < 3; a = a + 1) {
        var de: Jt_0;
        var lx: Jt_0;
        j_deriv_1(&de, e, a, p);
        j_mul_x_0(&(lx), lap, a, (*x0)[a], p);
        j_sub_0(&((*out)[a]), de, lx);
    }
}

fn pack_d_0(f: ptr<function, array<Jt_0, 3>>, d: ptr<function, array<ds, 6>>, sqrt2: ds) {
    var g00: ds = (*f)[0].c[1];
    var g01: ds = (*f)[0].c[2];
    var g02: ds = (*f)[0].c[3];
    var g10: ds = (*f)[1].c[1];
    var g11: ds = (*f)[1].c[2];
    var g12: ds = (*f)[1].c[3];
    var g20: ds = (*f)[2].c[1];
    var g21: ds = (*f)[2].c[2];
    var g22: ds = (*f)[2].c[3];
    (*d)[0] = ds_scale(ds_addp(g00, g00), 0.5);
    (*d)[1] = ds_scale(ds_addp(g11, g11), 0.5);
    (*d)[2] = ds_scale(ds_addp(g22, g22), 0.5);
    (*d)[3] = ds_mul(sqrt2, ds_scale(ds_addp(g01, g10), 0.5));
    (*d)[4] = ds_mul(sqrt2, ds_scale(ds_addp(g02, g20), 0.5));
    (*d)[5] = ds_mul(sqrt2, ds_scale(ds_addp(g12, g21), 0.5));
}

fn store_vd_0(f: ptr<function, array<Jt_0, 3>>, mode: u32, point: u32, p: BasisParams, sqrt2: ds) {
    var n: u32 = p.n_points;
    var local: u32 = 0;
    if p.want_v != 0u {
        for(var a: i32 = 0; a < 3; a = a + 1) {
            store_ds(mode * p.n_comp + local + u32(a), n, point, (*f)[a].c[0]);
        }
        local = local + (3u);
    }
    if p.want_d != 0u {
        var d: array<ds, 6>;
        pack_d_0(f, &d, sqrt2);
        for(var a: u32 = 0; a < 6u; a = a + 1u) {
            store_ds(mode * p.n_comp + local + a, n, point, d[a]);
        }
    }
}

fn emit_mode_2__acc(
    radial: Jt_2,
    ylm: Jt_2,
    tor: u32,
    x0: ptr<function, array<ds, 3>>,
    l: i32,
    m: i32,
    k: u32,
    point: u32,
    p: BasisParams,
    sqrt2: ds,
) {
    var phi: Jt_2;
    j_mul_2(&phi, radial, ylm, p);
    var field: array<Jt_0, 3>;
    if tor != 0u {
        toroidal_2(&field, phi, x0, p);
    } else {
        poloidal_2(&field, phi, x0, p);
    }
    var mode: u32 = vel_dof(l, m, tor, k, p.k_count);
    store_vd_0(&field, mode, point, p, sqrt2);
    high_apply_0_0(&field, mode, point, p, sqrt2);
}

fn eval_point_2__coef(gid: u32, prm: BasisParams) {
    if gid >= prm.n_points {
        return;
    }
    var sqrt2: ds = ds_mk(prm.sqrt2_hi, prm.sqrt2_lo);
    var pmm0: ds = ds_mk(prm.pmm_hi, prm.pmm_lo);
    var acc: array<dsacc, 9>;
    var px: ds;
    var py: ds;
    var pz: ds;
    load_point(gid, &px, &py, &pz);
    var xyz: array<ds, 3> = array<ds, 3>(px, py, pz);
    var u: array<Jt_2, 3>;
    var radius: Jt_2;
    unit_frame_2(&(u), &(radius), px, py, pz, prm);
    var l_mode: i32 = i32(prm.l_mode);
    var l_surf: i32 = i32(prm.l_surf);
    var l_top: i32 = select(l_surf, l_mode, l_mode > l_surf);
    var rr: Jt_2;
    sh_accumulate_2__saved(&rr, 0, l_top, l_surf, &(u), sqrt2, pmm0, prm);
    var inv: Jt_2;
    var zeta: Jt_2;
    var r2: Jt_2;
    j_recip_2(&inv, rr, prm);
    j_mul_2(&zeta, radius, inv, prm);
    j_mul_2(&r2, rr, rr, prm);
    var pows: array<Jt_2, 12>;
    fill_powers_2(&pows, zeta, prm.top_pow, prm);
    var cre: Jt_2;
    var cim: Jt_2;
    j_const_2(&cre, ds_f(1.0));
    j_const_2(&cim, ds_zero());
    var pmm: ds = pmm0;
    for(var m: i32 = 0; m <= l_mode; m = m + 1) {
        if m > 0 {
            cmul_u_2(&cre, &cim, &(u), prm);
            pmm = ds_mul(pmm, ds_sqrt(ds_div(ds_f(f32(2 * m + 1)), ds_f(f32(2 * m)))));
        }
        var p1: Jt_2;
        var p2: Jt_2;
        j_const_2(&p1, ds_zero());
        j_const_2(&p2, ds_zero());
        for(var l: i32 = m; l <= l_mode; l = l + 1) {
            var p: Jt_2;
            var yp: Jt_2;
            var ym: Jt_2;
            var t: Jt_2;
            legendre_p_2(&p, l, m, p1, p2, u[2], pmm, prm);
            if m == 0 {
                yp = p;
            } else {
                j_mul_2(&t, p, cre, prm);
                j_scale_2(&yp, t, sqrt2);
                j_mul_2(&t, p, cim, prm);
                j_scale_2(&ym, t, sqrt2);
            }
            if l >= 1 {
                for(var k: u32 = 0; k < prm.k_count; k = k + 1u) {
                    var psi: Jt_2;
                    var a: Jt_2;
                    var b: Jt_2;
                    let poly_base = (u32(l) * prm.k_count + k) * prm.poly_stride;
                    j_psi_2(&psi, poly_base, u32(l) + k + 1u, &pows);
                    j_mul_2(&a, r2, psi, prm);
                    j_mul_2(&b, rr, psi, prm);
                    emit_mode_2__acc(a, yp, 0u, &xyz, l, m, k, gid, prm, sqrt2);
                    emit_mode_2__acc(b, yp, 1u, &xyz, l, m, k, gid, prm, sqrt2);
                    if m > 0 {
                        emit_mode_2__acc(a, ym, 0u, &xyz, l, -m, k, gid, prm, sqrt2);
                        emit_mode_2__acc(b, ym, 1u, &xyz, l, -m, k, gid, prm, sqrt2);
                    }
                }
            }
            p2 = p1;
            p1 = p;
        }
    }
}

fn sh_accumulate_3__saved(
    rr: ptr<function, Jt_3>,
    l_save: i32,
    l_top: i32,
    l_surf: i32,
    u: ptr<function, array<Jt_3, 3>>,
    sqrt2: ds,
    pmm0: ds,
    prm: BasisParams,
) {
    j_const_3(rr, ds_zero());
    var cre: Jt_3;
    var cim: Jt_3;
    j_const_3(&cre, ds_f(1.0));
    j_const_3(&cim, ds_zero());
    var pmm: ds = pmm0;
    for(var m: i32 = 0; m <= l_top; m = m + 1) {
        if m > 0 {
            cmul_u_3(&cre, &cim, u, prm);
            pmm = ds_mul(pmm, ds_sqrt(ds_div(ds_f(f32(2 * m + 1)), ds_f(f32(2 * m)))));
        }
        var p1: Jt_3;
        var p2: Jt_3;
        j_const_3(&p1, ds_zero());
        j_const_3(&p2, ds_zero());
        for(var l: i32 = m; l <= l_top; l = l + 1) {
            var p: Jt_3;
            var yp: Jt_3;
            var ym: Jt_3;
            var t: Jt_3;
            legendre_p_3(&(p), l, m, p1, p2, (*u)[2], pmm, prm);
            if m == 0 {
                yp = p;
            } else {
                j_mul_3(&t, p, cre, prm);
                j_scale_3(&yp, t, sqrt2);
                j_mul_3(&t, p, cim, prm);
                j_scale_3(&ym, t, sqrt2);
            }
            if l <= l_surf {
                var cy: ds = coeff_at(l, m);
                if cy.hi != 0.0 || cy.lo != 0.0 {
                    j_axpy_3(rr, cy, yp);
                }
                if m > 0 {
                    var cz: ds = coeff_at(l, -m);
                    if cz.hi != 0.0 || cz.lo != 0.0 {
                        j_axpy_3(rr, cz, ym);
                    }
                }
            }
            p2 = p1;
            p1 = p;
        }
    }
}

fn eval_point_3__coef(gid: u32, prm: BasisParams) {
    if gid >= prm.n_points {
        return;
    }
    var sqrt2: ds = ds_mk(prm.sqrt2_hi, prm.sqrt2_lo);
    var pmm0: ds = ds_mk(prm.pmm_hi, prm.pmm_lo);
    var acc: array<dsacc, 9>;
    var px: ds;
    var py: ds;
    var pz: ds;
    load_point(gid, &px, &py, &pz);
    var xyz: array<ds, 3> = array<ds, 3>(px, py, pz);
    var u: array<Jt_3, 3>;
    var radius: Jt_3;
    unit_frame_3(&(u), &(radius), px, py, pz, prm);
    var l_mode: i32 = i32(prm.l_mode);
    var l_surf: i32 = i32(prm.l_surf);
    var l_top: i32 = select(l_surf, l_mode, l_mode > l_surf);
    var rr: Jt_3;
    sh_accumulate_3__saved(&rr, 0, l_top, l_surf, &(u), sqrt2, pmm0, prm);
    var inv: Jt_3;
    var zeta: Jt_3;
    var r2: Jt_3;
    j_recip_3(&inv, rr, prm);
    j_mul_3(&zeta, radius, inv, prm);
    j_mul_3(&r2, rr, rr, prm);
    var pows: array<Jt_3, 12>;
    fill_powers_3(&pows, zeta, prm.top_pow, prm);
    var cre: Jt_3;
    var cim: Jt_3;
    j_const_3(&cre, ds_f(1.0));
    j_const_3(&cim, ds_zero());
    var pmm: ds = pmm0;
    for(var m: i32 = 0; m <= l_mode; m = m + 1) {
        if m > 0 {
            cmul_u_3(&cre, &cim, &(u), prm);
            pmm = ds_mul(pmm, ds_sqrt(ds_div(ds_f(f32(2 * m + 1)), ds_f(f32(2 * m)))));
        }
        var p1: Jt_3;
        var p2: Jt_3;
        j_const_3(&p1, ds_zero());
        j_const_3(&p2, ds_zero());
        for(var l: i32 = m; l <= l_mode; l = l + 1) {
            var p: Jt_3;
            var yp: Jt_3;
            var ym: Jt_3;
            var t: Jt_3;
            legendre_p_3(&p, l, m, p1, p2, u[2], pmm, prm);
            if m == 0 {
                yp = p;
            } else {
                j_mul_3(&t, p, cre, prm);
                j_scale_3(&yp, t, sqrt2);
                j_mul_3(&t, p, cim, prm);
                j_scale_3(&ym, t, sqrt2);
            }
            if l >= 1 {
                for(var k: u32 = 0; k < prm.k_count; k = k + 1u) {
                    var psi: Jt_3;
                    var a: Jt_3;
                    var b: Jt_3;
                    let poly_base = (u32(l) * prm.k_count + k) * prm.poly_stride;
                    j_psi_3(&psi, poly_base, u32(l) + k + 1u, &pows);
                    j_mul_3(&a, r2, psi, prm);
                    j_mul_3(&b, rr, psi, prm);
                    emit_mode_3__acc(a, yp, 0u, &xyz, l, m, k, gid, prm, sqrt2);
                    emit_mode_3__acc(b, yp, 1u, &xyz, l, m, k, gid, prm, sqrt2);
                    if m > 0 {
                        emit_mode_3__acc(a, ym, 0u, &xyz, l, -m, k, gid, prm, sqrt2);
                        emit_mode_3__acc(b, ym, 1u, &xyz, l, -m, k, gid, prm, sqrt2);
                    }
                }
            }
            p2 = p1;
            p1 = p;
        }
    }
}

fn j_zero_4(r: ptr<function, Jt_4>) {
    for(var i: i32 = 0; i < 35; i = i + 1) {
        (*r).c[i] = ds_zero();
    }
}

fn j_const_4(r: ptr<function, Jt_4>, v: ds) {
    j_zero_4(r);
    (*r).c[0] = v;
}

fn j_var_4(r: ptr<function, Jt_4>, axis: i32, x0: ds) {
    j_const_4(r, x0);
    if 4 >= 1 {
        (*r).c[1 + axis] = ds_f(1.0);
    }
}

fn mul_tab_4(p: BasisParams) -> u32 {
    if 4 <= 0 {
        return p.moff0;
    }
    if 4 == 1 {
        return p.moff1;
    }
    if 4 == 2 {
        return p.moff2;
    }
    if 4 == 3 {
        return p.moff3;
    }
    return p.moff4;
}

fn mul_count_4(p: BasisParams) -> u32 {
    if 4 <= 0 {
        return p.nmul0;
    }
    if 4 == 1 {
        return p.nmul1;
    }
    if 4 == 2 {
        return p.nmul2;
    }
    if 4 == 3 {
        return p.nmul3;
    }
    return p.nmul4;
}

fn j_mul_4(r: ptr<function, Jt_4>, a: Jt_4, b: Jt_4, p: BasisParams) {
    var acc: array<dsacc, 35>;
    for(var i: i32 = 0; i < 35; i = i + 1) {
        acc[i] = acc_zero();
    }
    let tab_base = mul_tab_4(p);
    var n: u32 = mul_count_4(p);
    for(var t: u32 = 0; t < n; t = t + 1u) {
        var e: u32 = tables[tab_base + t];
        acc[(e >> 16) & 255u] = acc_fma(acc[(e >> 16) & 255u], a.c[e & 255u], b.c[(e >> 8) & 255u]);
    }
    for(var i: i32 = 0; i < 35; i = i + 1) {
        (*r).c[i] = acc_done(acc[i]);
    }
}

fn j_add3_4(r: ptr<function, Jt_4>, a: Jt_4, b: Jt_4, c: Jt_4) {
    for(var i: i32 = 0; i < 35; i = i + 1) {
        (*r).c[i] = ds_addp(ds_addp(a.c[i], b.c[i]), c.c[i]);
    }
    for(var i: i32 = 35; i < 35; i = i + 1) {
        (*r).c[i] = ds_zero();
    }
}

fn j_compose_4(r: ptr<function, Jt_4>, jet: Jt_4, f: ptr<function, array<ds, 5>>, p: BasisParams) {
    var delta: Jt_4 = jet;
    delta.c[0] = ds_zero();
    var fact: ds = ds_f(1.0);
    var coef: array<ds, 5>;
    for(var k: i32 = 0; k <= 4; k = k + 1) {
        if k > 0 {
            fact = ds_mul(fact, ds_f(f32(k)));
        }
        coef[k] = ds_div((*f)[k], fact);
    }
    j_const_4(r, coef[4]);
    for(var k: i32 = i32(4) - 1; k >= 0; k = k - 1) {
        var tmp: Jt_4;
        j_mul_4(&tmp, *r, delta, p);
        tmp.c[0] = ds_addp(tmp.c[0], coef[k]);
        *r = tmp;
    }
}

fn j_sqrt_4(r: ptr<function, Jt_4>, jet: Jt_4, p: BasisParams) {
    var a: ds = jet.c[0];
    var s: ds = ds_sqrt(a);
    var as_: ds = ds_mul(a, s);
    var a2: ds = ds_sqr(a);
    var a2s: ds = ds_mul(a2, s);
    var a3s: ds = ds_mul(ds_mul(a2, a), s);
    var f: array<ds, 5>;
    f[0] = s;
    f[1] = ds_div(ds_f(0.5), s);
    f[2] = ds_neg(ds_div(ds_f(0.25), as_));
    f[3] = ds_div(ds_f(0.375), a2s);
    f[4] = ds_neg(ds_div(ds_f(0.9375), a3s));
    j_compose_4(r, jet, &f, p);
}

fn radial_jet_4(radius: ptr<function, Jt_4>, px: ds, py: ds, pz: ds, p: BasisParams) {
    var x: Jt_4;
    var y: Jt_4;
    var z: Jt_4;
    var x2: Jt_4;
    var y2: Jt_4;
    var z2: Jt_4;
    var r2: Jt_4;
    j_var_4(&x, 0, px);
    j_var_4(&y, 1, py);
    j_var_4(&z, 2, pz);
    j_mul_4(&x2, x, x, p);
    j_mul_4(&y2, y, y, p);
    j_mul_4(&z2, z, z, p);
    j_add3_4(&r2, x2, y2, z2);
    j_sqrt_4(radius, r2, p);
}

fn j_recip_4(r: ptr<function, Jt_4>, jet: Jt_4, p: BasisParams) {
    var a: ds = jet.c[0];
    var a2: ds = ds_sqr(a);
    var a3: ds = ds_mul(a2, a);
    var a4: ds = ds_sqr(a2);
    var a5: ds = ds_mul(a4, a);
    var f: array<ds, 5>;
    f[0] = ds_recip(a);
    f[1] = ds_neg(ds_recip(a2));
    f[2] = ds_mul(ds_f(2.0), ds_recip(a3));
    f[3] = ds_neg(ds_mul(ds_f(6.0), ds_recip(a4)));
    f[4] = ds_mul(ds_f(24.0), ds_recip(a5));
    j_compose_4(r, jet, &f, p);
}

fn unit_frame_4(
    u: ptr<function, array<Jt_4, 3>>,
    radius: ptr<function, Jt_4>,
    px: ds,
    py: ds,
    pz: ds,
    p: BasisParams,
) {
    var x: Jt_4;
    var y: Jt_4;
    var z: Jt_4;
    var inv: Jt_4;
    j_var_4(&x, 0, px);
    j_var_4(&y, 1, py);
    j_var_4(&z, 2, pz);
    radial_jet_4(radius, px, py, pz, p);
    j_recip_4(&inv, *radius, p);
    j_mul_4(&((*u)[0]), x, inv, p);
    j_mul_4(&((*u)[1]), y, inv, p);
    j_mul_4(&((*u)[2]), z, inv, p);
}

fn j_sub_4(r: ptr<function, Jt_4>, a: Jt_4, b: Jt_4) {
    for(var i: i32 = 0; i < 35; i = i + 1) {
        (*r).c[i] = ds_subp(a.c[i], b.c[i]);
    }
    for(var i: i32 = 35; i < 35; i = i + 1) {
        (*r).c[i] = ds_zero();
    }
}

fn j_add_4(r: ptr<function, Jt_4>, a: Jt_4, b: Jt_4) {
    for(var i: i32 = 0; i < 35; i = i + 1) {
        (*r).c[i] = ds_addp(a.c[i], b.c[i]);
    }
    for(var i: i32 = 35; i < 35; i = i + 1) {
        (*r).c[i] = ds_zero();
    }
}

fn cmul_u_4(
    cre: ptr<function, Jt_4>,
    cim: ptr<function, Jt_4>,
    u: ptr<function, array<Jt_4, 3>>,
    p: BasisParams,
) {
    var t1: Jt_4;
    var t2: Jt_4;
    var nre: Jt_4;
    var nim: Jt_4;
    j_mul_4(&(t1), *cre, (*u)[0], p);
    j_mul_4(&(t2), *cim, (*u)[1], p);
    j_sub_4(&nre, t1, t2);
    j_mul_4(&(t1), *cre, (*u)[1], p);
    j_mul_4(&(t2), *cim, (*u)[0], p);
    j_add_4(&nim, t1, t2);
    *cre = nre;
    *cim = nim;
}

fn j_scale_4(r: ptr<function, Jt_4>, a: Jt_4, s: ds) {
    for(var i: i32 = 0; i < 35; i = i + 1) {
        (*r).c[i] = ds_mul(a.c[i], s);
    }
    for(var i: i32 = 35; i < 35; i = i + 1) {
        (*r).c[i] = ds_zero();
    }
}

fn legendre_p_4(
    p: ptr<function, Jt_4>,
    l: i32,
    m: i32,
    p1: Jt_4,
    p2: Jt_4,
    uz: Jt_4,
    pmm: ds,
    prm: BasisParams,
) {
    if l == m {
        j_const_4(p, pmm);
        return;
    }
    var a: ds = coef_a(l, m);
    var zp: Jt_4;
    j_mul_4(&zp, uz, p1, prm);
    if l == m + 1 {
        j_scale_4(p, zp, a);
        return;
    }
    var bp: Jt_4;
    var d: Jt_4;
    j_scale_4(&bp, p2, coef_b(l, m));
    j_sub_4(&d, zp, bp);
    j_scale_4(p, d, a);
}

fn j_axpy_4(r: ptr<function, Jt_4>, s: ds, a: Jt_4) {
    for(var i: i32 = 0; i < 35; i = i + 1) {
        (*r).c[i] = ds_addp((*r).c[i], ds_mul(s, a.c[i]));
    }
}

fn sh_accumulate_4__saved(
    rr: ptr<function, Jt_4>,
    l_save: i32,
    l_top: i32,
    l_surf: i32,
    u: ptr<function, array<Jt_4, 3>>,
    sqrt2: ds,
    pmm0: ds,
    prm: BasisParams,
) {
    j_const_4(rr, ds_zero());
    var cre: Jt_4;
    var cim: Jt_4;
    j_const_4(&cre, ds_f(1.0));
    j_const_4(&cim, ds_zero());
    var pmm: ds = pmm0;
    for(var m: i32 = 0; m <= l_top; m = m + 1) {
        if m > 0 {
            cmul_u_4(&cre, &cim, u, prm);
            pmm = ds_mul(pmm, ds_sqrt(ds_div(ds_f(f32(2 * m + 1)), ds_f(f32(2 * m)))));
        }
        var p1: Jt_4;
        var p2: Jt_4;
        j_const_4(&p1, ds_zero());
        j_const_4(&p2, ds_zero());
        for(var l: i32 = m; l <= l_top; l = l + 1) {
            var p: Jt_4;
            var yp: Jt_4;
            var ym: Jt_4;
            var t: Jt_4;
            legendre_p_4(&(p), l, m, p1, p2, (*u)[2], pmm, prm);
            if m == 0 {
                yp = p;
            } else {
                j_mul_4(&t, p, cre, prm);
                j_scale_4(&yp, t, sqrt2);
                j_mul_4(&t, p, cim, prm);
                j_scale_4(&ym, t, sqrt2);
            }
            if l <= l_surf {
                var cy: ds = coeff_at(l, m);
                if cy.hi != 0.0 || cy.lo != 0.0 {
                    j_axpy_4(rr, cy, yp);
                }
                if m > 0 {
                    var cz: ds = coeff_at(l, -m);
                    if cz.hi != 0.0 || cz.lo != 0.0 {
                        j_axpy_4(rr, cz, ym);
                    }
                }
            }
            p2 = p1;
            p1 = p;
        }
    }
}

fn fill_powers_4(pows: ptr<function, array<Jt_4, 12>>, zeta: Jt_4, top: u32, p: BasisParams) {
    j_const_4(&((*pows)[0]), ds_f(1.0));
    for(var n: u32 = 1; n <= top; n = n + 1u) {
        j_mul_4(&((*pows)[n]), (*pows)[n - 1], zeta, p);
    }
}

fn j_psi_4(s: ptr<function, Jt_4>, base: u32, len: u32, pows: ptr<function, array<Jt_4, 12>>) {
    j_const_4(s, ds_zero());
    for(var j: u32 = 0; j < len; j = j + 1u) {
        var a: ds = ds_mk(polys[base + j].x, polys[base + j].y);
        if a.hi != 0.0 || a.lo != 0.0 {
            j_axpy_4(s, a, (*pows)[j]);
        }
    }
}

fn high_apply_1_2(
    f: ptr<function, array<Jt_2, 3>>,
    mode: u32,
    point: u32,
    p: BasisParams,
    sqrt2: ds,
) {
    if p.want_lap == 0u && p.want_gd == 0u {
        return;
    }
    let pair_base = p.pair_off;
    var h: array<array<array<ds, 3>, 3>, 3>;
    for(var comp: i32 = 0; comp < 3; comp = comp + 1) {
        for(var b: i32 = 0; b < 3; b = b + 1) {
            for(var c: i32 = 0; c < 3; c = c + 1) {
                var coef: ds = (*f)[comp].c[tables[pair_base + u32(b) * 3u + u32(c)]];
                h[comp][b][c] = coef;
                if b == c {
                    h[comp][b][c] = ds_scale(coef, 2.0);
                }
            }
        }
    }
    var n: u32 = p.n_points;
    var local: u32 = vd_channels(p);
    if p.want_lap != 0u {
        for(var a: i32 = 0; a < 3; a = a + 1) {
            var lap: ds = ds_addp(ds_addp(h[a][0][0], h[a][1][1]), h[a][2][2]);
            store_ds(mode * p.n_comp + local + u32(a), n, point, lap);
        }
        local = local + (3u);
    }
    if p.want_gd != 0u {
        for(var k: i32 = 0; k < 3; k = k + 1) {
            var g0: ds = ds_scale(ds_addp(h[0][0][k], h[0][0][k]), 0.5);
            var g1: ds = ds_scale(ds_addp(h[1][1][k], h[1][1][k]), 0.5);
            var g2: ds = ds_scale(ds_addp(h[2][2][k], h[2][2][k]), 0.5);
            var g3: ds = ds_mul(sqrt2, ds_scale(ds_addp(h[0][1][k], h[1][0][k]), 0.5));
            var g4: ds = ds_mul(sqrt2, ds_scale(ds_addp(h[0][2][k], h[2][0][k]), 0.5));
            var g5: ds = ds_mul(sqrt2, ds_scale(ds_addp(h[1][2][k], h[2][1][k]), 0.5));
            var packed: array<ds, 6> = array<ds, 6>(g0, g1, g2, g3, g4, g5);
            for(var c: u32 = 0; c < 6u; c = c + 1u) {
                store_ds(mode * p.n_comp + local + c, n, point, packed[c]);
            }
            local = local + (6u);
        }
    }
}

fn j_deriv_4(r: ptr<function, Jt_3>, a: Jt_4, axis: i32, p: BasisParams) {
    let up_base = p.up_off;
    let mono_base = p.mono_off;
    j_zero_3(r);
    for(var i: i32 = 0; i < 20; i = i + 1) {
        (*r).c[i] = ds_mulf(
                f32(tables[mono_base + u32(i) * 3u + u32(axis)] + 1u),
                a.c[tables[up_base + u32(axis) * 35u + u32(i)]],
            );
    }
}

fn j_mul_x_3(r: ptr<function, Jt_3>, a: Jt_3, axis: i32, x0: ds, p: BasisParams) {
    for(var i: i32 = 0; i < 20; i = i + 1) {
        (*r).c[i] = ds_mul(a.c[i], x0);
    }
    for(var i: i32 = 20; i < 20; i = i + 1) {
        (*r).c[i] = ds_zero();
    }
    let up_base = p.up_off;
    let deg_base = p.deg_off;
    for(var i: i32 = 0; i < 20; i = i + 1) {
        if tables[deg_base + u32(i)] < u32(3) {
            var dst: u32 = tables[up_base + u32(axis) * 35u + u32(i)];
            (*r).c[dst] = ds_addp((*r).c[dst], a.c[i]);
        }
    }
}

fn toroidal_4(
    out: ptr<function, array<Jt_2, 3>>,
    psi: Jt_4,
    x0: ptr<function, array<ds, 3>>,
    p: BasisParams,
) {
    var g: array<Jt_3, 3>;
    for(var a: i32 = 0; a < 3; a = a + 1) {
        j_deriv_4(&(g[a]), psi, a, p);
    }
    var a: Jt_3;
    var b: Jt_3;
    var cross: Jt_3;
    j_mul_x_3(&(a), g[1], 2, (*x0)[2], p);
    j_mul_x_3(&(b), g[2], 1, (*x0)[1], p);
    j_sub_3(&cross, a, b);
    j_trunc_3_2(&((*out)[0]), cross);
    j_mul_x_3(&(a), g[2], 0, (*x0)[0], p);
    j_mul_x_3(&(b), g[0], 2, (*x0)[2], p);
    j_sub_3(&cross, a, b);
    j_trunc_3_2(&((*out)[1]), cross);
    j_mul_x_3(&(a), g[0], 1, (*x0)[1], p);
    j_mul_x_3(&(b), g[1], 0, (*x0)[0], p);
    j_sub_3(&cross, a, b);
    j_trunc_3_2(&((*out)[2]), cross);
}

fn j_trunc_4_3(r: ptr<function, Jt_3>, a: Jt_4) {
    for(var i: i32 = 0; i < 20; i = i + 1) {
        (*r).c[i] = a.c[i];
    }
    for(var i: i32 = 20; i < 20; i = i + 1) {
        (*r).c[i] = ds_zero();
    }
}

fn j_add_into_3(r: ptr<function, Jt_3>, a: Jt_3) {
    for(var i: i32 = 0; i < 20; i = i + 1) {
        (*r).c[i] = ds_addp((*r).c[i], a.c[i]);
    }
}

fn poloidal_4(
    out: ptr<function, array<Jt_2, 3>>,
    phi: Jt_4,
    x0: ptr<function, array<ds, 3>>,
    p: BasisParams,
) {
    var g: array<Jt_3, 3>;
    for(var a: i32 = 0; a < 3; a = a + 1) {
        j_deriv_4(&(g[a]), phi, a, p);
    }
    var e: Jt_3;
    var mx: Jt_3;
    j_trunc_4_3(&e, phi);
    for(var a: i32 = 0; a < 3; a = a + 1) {
        j_mul_x_3(&(mx), g[a], a, (*x0)[a], p);
        j_add_into_3(&e, mx);
    }
    var d0: Jt_2;
    var d1: Jt_2;
    var d2: Jt_2;
    var lap: Jt_2;
    j_deriv_3(&d0, g[0], 0, p);
    j_deriv_3(&d1, g[1], 1, p);
    j_deriv_3(&d2, g[2], 2, p);
    j_add3_2(&lap, d0, d1, d2);
    for(var a: i32 = 0; a < 3; a = a + 1) {
        var de: Jt_2;
        var lx: Jt_2;
        j_deriv_3(&de, e, a, p);
        j_mul_x_2(&(lx), lap, a, (*x0)[a], p);
        j_sub_2(&((*out)[a]), de, lx);
    }
}

fn pack_d_2(f: ptr<function, array<Jt_2, 3>>, d: ptr<function, array<ds, 6>>, sqrt2: ds) {
    var g00: ds = (*f)[0].c[1];
    var g01: ds = (*f)[0].c[2];
    var g02: ds = (*f)[0].c[3];
    var g10: ds = (*f)[1].c[1];
    var g11: ds = (*f)[1].c[2];
    var g12: ds = (*f)[1].c[3];
    var g20: ds = (*f)[2].c[1];
    var g21: ds = (*f)[2].c[2];
    var g22: ds = (*f)[2].c[3];
    (*d)[0] = ds_scale(ds_addp(g00, g00), 0.5);
    (*d)[1] = ds_scale(ds_addp(g11, g11), 0.5);
    (*d)[2] = ds_scale(ds_addp(g22, g22), 0.5);
    (*d)[3] = ds_mul(sqrt2, ds_scale(ds_addp(g01, g10), 0.5));
    (*d)[4] = ds_mul(sqrt2, ds_scale(ds_addp(g02, g20), 0.5));
    (*d)[5] = ds_mul(sqrt2, ds_scale(ds_addp(g12, g21), 0.5));
}

fn store_vd_2(f: ptr<function, array<Jt_2, 3>>, mode: u32, point: u32, p: BasisParams, sqrt2: ds) {
    var n: u32 = p.n_points;
    var local: u32 = 0;
    if p.want_v != 0u {
        for(var a: i32 = 0; a < 3; a = a + 1) {
            store_ds(mode * p.n_comp + local + u32(a), n, point, (*f)[a].c[0]);
        }
        local = local + (3u);
    }
    if p.want_d != 0u {
        var d: array<ds, 6>;
        pack_d_2(f, &d, sqrt2);
        for(var a: u32 = 0; a < 6u; a = a + 1u) {
            store_ds(mode * p.n_comp + local + a, n, point, d[a]);
        }
    }
}

fn emit_mode_4__acc(
    radial: Jt_4,
    ylm: Jt_4,
    tor: u32,
    x0: ptr<function, array<ds, 3>>,
    l: i32,
    m: i32,
    k: u32,
    point: u32,
    p: BasisParams,
    sqrt2: ds,
) {
    var phi: Jt_4;
    j_mul_4(&phi, radial, ylm, p);
    var field: array<Jt_2, 3>;
    if tor != 0u {
        toroidal_4(&field, phi, x0, p);
    } else {
        poloidal_4(&field, phi, x0, p);
    }
    var mode: u32 = vel_dof(l, m, tor, k, p.k_count);
    store_vd_2(&field, mode, point, p, sqrt2);
    high_apply_1_2(&field, mode, point, p, sqrt2);
}

fn eval_point_4__coef(gid: u32, prm: BasisParams) {
    if gid >= prm.n_points {
        return;
    }
    var sqrt2: ds = ds_mk(prm.sqrt2_hi, prm.sqrt2_lo);
    var pmm0: ds = ds_mk(prm.pmm_hi, prm.pmm_lo);
    var acc: array<dsacc, 9>;
    var px: ds;
    var py: ds;
    var pz: ds;
    load_point(gid, &px, &py, &pz);
    var xyz: array<ds, 3> = array<ds, 3>(px, py, pz);
    var u: array<Jt_4, 3>;
    var radius: Jt_4;
    unit_frame_4(&(u), &(radius), px, py, pz, prm);
    var l_mode: i32 = i32(prm.l_mode);
    var l_surf: i32 = i32(prm.l_surf);
    var l_top: i32 = select(l_surf, l_mode, l_mode > l_surf);
    var rr: Jt_4;
    sh_accumulate_4__saved(&rr, 0, l_top, l_surf, &(u), sqrt2, pmm0, prm);
    var inv: Jt_4;
    var zeta: Jt_4;
    var r2: Jt_4;
    j_recip_4(&inv, rr, prm);
    j_mul_4(&zeta, radius, inv, prm);
    j_mul_4(&r2, rr, rr, prm);
    var pows: array<Jt_4, 12>;
    fill_powers_4(&pows, zeta, prm.top_pow, prm);
    var cre: Jt_4;
    var cim: Jt_4;
    j_const_4(&cre, ds_f(1.0));
    j_const_4(&cim, ds_zero());
    var pmm: ds = pmm0;
    for(var m: i32 = 0; m <= l_mode; m = m + 1) {
        if m > 0 {
            cmul_u_4(&cre, &cim, &(u), prm);
            pmm = ds_mul(pmm, ds_sqrt(ds_div(ds_f(f32(2 * m + 1)), ds_f(f32(2 * m)))));
        }
        var p1: Jt_4;
        var p2: Jt_4;
        j_const_4(&p1, ds_zero());
        j_const_4(&p2, ds_zero());
        for(var l: i32 = m; l <= l_mode; l = l + 1) {
            var p: Jt_4;
            var yp: Jt_4;
            var ym: Jt_4;
            var t: Jt_4;
            legendre_p_4(&p, l, m, p1, p2, u[2], pmm, prm);
            if m == 0 {
                yp = p;
            } else {
                j_mul_4(&t, p, cre, prm);
                j_scale_4(&yp, t, sqrt2);
                j_mul_4(&t, p, cim, prm);
                j_scale_4(&ym, t, sqrt2);
            }
            if l >= 1 {
                for(var k: u32 = 0; k < prm.k_count; k = k + 1u) {
                    var psi: Jt_4;
                    var a: Jt_4;
                    var b: Jt_4;
                    let poly_base = (u32(l) * prm.k_count + k) * prm.poly_stride;
                    j_psi_4(&psi, poly_base, u32(l) + k + 1u, &pows);
                    j_mul_4(&a, r2, psi, prm);
                    j_mul_4(&b, rr, psi, prm);
                    emit_mode_4__acc(a, yp, 0u, &xyz, l, m, k, gid, prm, sqrt2);
                    emit_mode_4__acc(b, yp, 1u, &xyz, l, m, k, gid, prm, sqrt2);
                    if m > 0 {
                        emit_mode_4__acc(a, ym, 0u, &xyz, l, -m, k, gid, prm, sqrt2);
                        emit_mode_4__acc(b, ym, 1u, &xyz, l, -m, k, gid, prm, sqrt2);
                    }
                }
            }
            p2 = p1;
            p1 = p;
        }
    }
}

fn eval_pres0_4__coef(gid: u32, prm: BasisParams) {
    if gid >= prm.n_points {
        return;
    }
    var sqrt2: ds = ds_mk(prm.sqrt2_hi, prm.sqrt2_lo);
    var pmm: ds = ds_mk(prm.pmm_hi, prm.pmm_lo);
    var x: ds;
    var y: ds;
    var z: ds;
    load_point(gid, &x, &y, &z);
    var rho2: ds = ds_addp(ds_addp(ds_sqr(x), ds_sqr(y)), ds_sqr(z));
    var rho: ds = ds_sqrt(rho2);
    var inv: ds = ds_recip(rho);
    var ux: ds = ds_mul(x, inv);
    var uy: ds = ds_mul(y, inv);
    var uz: ds = ds_mul(z, inv);
    var l_mode: i32 = i32(prm.l_mode);
    var l_surf: i32 = i32(prm.l_surf);
    var l_top: i32 = select(l_surf, l_mode, l_mode > l_surf);
    var saved: array<ds, 625>;
    var rr: ds = ds_zero();
    var cre: ds = ds_f(1.0);
    var cim: ds = ds_zero();
    for(var m: i32 = 0; m <= l_top; m = m + 1) {
        if m > 0 {
            var nre: ds = ds_subp(ds_mul(cre, ux), ds_mul(cim, uy));
            var nim: ds = ds_addp(ds_mul(cre, uy), ds_mul(cim, ux));
            cre = nre;
            cim = nim;
            pmm = ds_mul(pmm, ds_sqrt(ds_div(ds_f(f32(2 * m + 1)), ds_f(f32(2 * m)))));
        }
        var p1: ds = ds_zero();
        var p2: ds = ds_zero();
        for(var l: i32 = m; l <= l_top; l = l + 1) {
            var assoc: ds;
            if l == m {
                assoc = pmm;
            } else if l == m + 1 {
                assoc = ds_mul(coef_a(l, m), ds_mul(uz, p1));
            } else {
                assoc = ds_mul(coef_a(l, m), ds_subp(ds_mul(uz, p1), ds_mul(coef_b(l, m), p2)));
            }
            var yp: ds = assoc;
            var ym: ds = ds_zero();
            if m > 0 {
                yp = ds_mul(ds_mul(assoc, cre), sqrt2);
                ym = ds_mul(ds_mul(assoc, cim), sqrt2);
            }
            if l <= l_surf {
                rr = ds_addp(rr, ds_mul(coeff_at(l, m), yp));
                if m > 0 {
                    rr = ds_addp(rr, ds_mul(coeff_at(l, -m), ym));
                }
            }
            if l <= l_mode && l <= 4 {
                saved[lm_of(l, m)] = yp;
                if m > 0 {
                    saved[lm_of(l, -m)] = ym;
                }
            }
            p2 = p1;
            p1 = assoc;
        }
    }
    var zeta: ds = ds_mul(rho, ds_recip(rr));
    var one: ds = ds_subp(ds_f(1.0), zeta);
    var zp: array<ds, 33>;
    zp[0] = ds_f(1.0);
    for(var n: u32 = 1; n <= prm.top_pow && n < 33u; n = n + 1u) {
        zp[n] = ds_mul(zp[n - 1], zeta);
    }
    var n: u32 = prm.n_points;
    var np: u32 = prm.n_out;
    var acc: dsacc = acc_zero();
    for(var l: i32 = 0; l <= l_mode; l = l + 1) {
        for(var m: i32 = -l; m <= l; m = m + 1) {
            var lm: u32 = lm_of(l, m);
            var ylm: ds = saved[lm];
            var lift: ds = ds_mul(zp[u32(l)], ylm);
            {
                store_ds(np + lm, n, gid, lift);
            }
            for(var k: u32 = 0; k < prm.k_count; k = k + 1u) {
                let poly_base = (u32(l) * prm.k_count + k) * prm.poly_stride;
                var psi: ds = ds_zero();
                var len: u32 = select(u32(l) + k + 1u, 2u * k + 2u, l == 0);
                for(var j: u32 = 0; j < len; j = j + 1u) {
                    var a: ds = ds_mk(polys[poly_base + j].x, polys[poly_base + j].y);
                    if a.hi != 0.0 || a.lo != 0.0 {
                        psi = ds_addp(psi, ds_mul(a, zp[j]));
                    }
                }
                var phi: ds = ds_mul(ds_mul(one, psi), ylm);
                {
                    store_ds(lm * prm.k_count + k, n, gid, phi);
                }
            }
        }
    }
}

fn eval_pres0_24__coef(gid: u32, prm: BasisParams) {
    if gid >= prm.n_points {
        return;
    }
    var sqrt2: ds = ds_mk(prm.sqrt2_hi, prm.sqrt2_lo);
    var pmm: ds = ds_mk(prm.pmm_hi, prm.pmm_lo);
    var x: ds;
    var y: ds;
    var z: ds;
    load_point(gid, &x, &y, &z);
    var rho2: ds = ds_addp(ds_addp(ds_sqr(x), ds_sqr(y)), ds_sqr(z));
    var rho: ds = ds_sqrt(rho2);
    var inv: ds = ds_recip(rho);
    var ux: ds = ds_mul(x, inv);
    var uy: ds = ds_mul(y, inv);
    var uz: ds = ds_mul(z, inv);
    var l_mode: i32 = i32(prm.l_mode);
    var l_surf: i32 = i32(prm.l_surf);
    var l_top: i32 = select(l_surf, l_mode, l_mode > l_surf);
    var saved: array<ds, 625>;
    var rr: ds = ds_zero();
    var cre: ds = ds_f(1.0);
    var cim: ds = ds_zero();
    for(var m: i32 = 0; m <= l_top; m = m + 1) {
        if m > 0 {
            var nre: ds = ds_subp(ds_mul(cre, ux), ds_mul(cim, uy));
            var nim: ds = ds_addp(ds_mul(cre, uy), ds_mul(cim, ux));
            cre = nre;
            cim = nim;
            pmm = ds_mul(pmm, ds_sqrt(ds_div(ds_f(f32(2 * m + 1)), ds_f(f32(2 * m)))));
        }
        var p1: ds = ds_zero();
        var p2: ds = ds_zero();
        for(var l: i32 = m; l <= l_top; l = l + 1) {
            var assoc: ds;
            if l == m {
                assoc = pmm;
            } else if l == m + 1 {
                assoc = ds_mul(coef_a(l, m), ds_mul(uz, p1));
            } else {
                assoc = ds_mul(coef_a(l, m), ds_subp(ds_mul(uz, p1), ds_mul(coef_b(l, m), p2)));
            }
            var yp: ds = assoc;
            var ym: ds = ds_zero();
            if m > 0 {
                yp = ds_mul(ds_mul(assoc, cre), sqrt2);
                ym = ds_mul(ds_mul(assoc, cim), sqrt2);
            }
            if l <= l_surf {
                rr = ds_addp(rr, ds_mul(coeff_at(l, m), yp));
                if m > 0 {
                    rr = ds_addp(rr, ds_mul(coeff_at(l, -m), ym));
                }
            }
            if l <= l_mode && l <= 24 {
                saved[lm_of(l, m)] = yp;
                if m > 0 {
                    saved[lm_of(l, -m)] = ym;
                }
            }
            p2 = p1;
            p1 = assoc;
        }
    }
    var zeta: ds = ds_mul(rho, ds_recip(rr));
    var one: ds = ds_subp(ds_f(1.0), zeta);
    var zp: array<ds, 33>;
    zp[0] = ds_f(1.0);
    for(var n: u32 = 1; n <= prm.top_pow && n < 33u; n = n + 1u) {
        zp[n] = ds_mul(zp[n - 1], zeta);
    }
    var n: u32 = prm.n_points;
    var np: u32 = prm.n_out;
    var acc: dsacc = acc_zero();
    for(var l: i32 = 0; l <= l_mode; l = l + 1) {
        for(var m: i32 = -l; m <= l; m = m + 1) {
            var lm: u32 = lm_of(l, m);
            var ylm: ds = saved[lm];
            var lift: ds = ds_mul(zp[u32(l)], ylm);
            {
                store_ds(np + lm, n, gid, lift);
            }
            for(var k: u32 = 0; k < prm.k_count; k = k + 1u) {
                let poly_base = (u32(l) * prm.k_count + k) * prm.poly_stride;
                var psi: ds = ds_zero();
                var len: u32 = select(u32(l) + k + 1u, 2u * k + 2u, l == 0);
                for(var j: u32 = 0; j < len; j = j + 1u) {
                    var a: ds = ds_mk(polys[poly_base + j].x, polys[poly_base + j].y);
                    if a.hi != 0.0 || a.lo != 0.0 {
                        psi = ds_addp(psi, ds_mul(a, zp[j]));
                    }
                }
                var phi: ds = ds_mul(ds_mul(one, psi), ylm);
                {
                    store_ds(lm * prm.k_count + k, n, gid, phi);
                }
            }
        }
    }
}

fn accumulate_vd_1(
    f: ptr<function, array<Jt_1, 3>>,
    acc: ptr<function, array<dsacc, 9>>,
    p: BasisParams,
    weight: ds,
    sqrt2: ds,
) {
    var local: u32 = 0;
    if p.want_v != 0u {
        for(var a: i32 = 0; a < 3; a = a + 1) {
            (*acc)[local + u32(a)] = acc_fma((*acc)[local + u32(a)], weight, (*f)[a].c[0]);
        }
        local = local + (3u);
    }
    if p.want_d != 0u {
        var d: array<ds, 6>;
        pack_d_1(f, &d, sqrt2);
        for(var a: u32 = 0; a < 6u; a = a + 1u) {
            (*acc)[local + a] = acc_fma((*acc)[local + a], weight, d[a]);
        }
    }
}

fn emit_mode_3(
    radial: Jt_3,
    ylm: Jt_3,
    tor: u32,
    x0: ptr<function, array<ds, 3>>,
    l: i32,
    m: i32,
    k: u32,
    point: u32,
    p: BasisParams,
    sqrt2: ds,
    acc: ptr<function, array<dsacc, 9>>,
) {
    var phi: Jt_3;
    j_mul_3(&phi, radial, ylm, p);
    var field: array<Jt_1, 3>;
    if tor != 0u {
        toroidal_3(&field, phi, x0, p);
    } else {
        poloidal_3(&field, phi, x0, p);
    }
    var mode: u32 = vel_dof(l, m, tor, k, p.k_count);
    {
        var w: vec2<f32> = coef[mode];
        accumulate_vd_1(&field, acc, p, ds_mk(w.x, w.y), sqrt2);
        return;
    }
    store_vd_1(&field, mode, point, p, sqrt2);
    high_apply_0_1(&field, mode, point, p, sqrt2);
}

fn eval_cached_3_2(ray: u32, prm: BasisParams) {
    if ray >= prm.n_rays {
        return;
    }
    var sqrt2: ds = ds_mk(prm.sqrt2_hi, prm.sqrt2_lo);
    var pmm0: ds = ds_mk(prm.pmm_hi, prm.pmm_lo);
    var p0: u32 = ray * prm.n_per;
    var x0: ds;
    var y0: ds;
    var z0: ds;
    load_point(p0, &x0, &y0, &z0);
    var u: array<Jt_3, 3>;
    var radius0: Jt_3;
    unit_frame_3(&(u), &(radius0), x0, y0, z0, prm);
    var l_mode: i32 = i32(prm.l_mode);
    var l_surf: i32 = i32(prm.l_surf);
    var l_top: i32 = select(l_surf, l_mode, l_mode > l_surf);
    var saved: array<Jt_3, 25>;
    var rr: Jt_3;
    sh_accumulate_3(&rr, &saved, l_mode, l_top, l_surf, &(u), sqrt2, pmm0, prm);
    var rho0: ds = radius0.c[0];
    let deg_base = prm.deg_off;
    for(var s: u32 = 0; s < prm.n_per; s = s + 1u) {
        var px: ds;
        var py: ds;
        var pz: ds;
        load_point(p0 + s, &px, &py, &pz);
        var xyz: array<ds, 3> = array<ds, 3>(px, py, pz);
        var rs: Jt_3 = radius0;
        if s != 0u {
            radial_jet_3(&rs, px, py, pz, prm);
        }
        var rho: ds = rs.c[0];
        var factor: array<ds, 5>;
        factor[0] = ds_f(1.0);
        if s != 0u {
            var ratio: ds = ds_div(rho0, rho);
            for(var d: i32 = 1; d <= 3; d = d + 1) {
                factor[d] = ds_mul(factor[d - 1], ratio);
            }
        }
        var rad_s: Jt_3 = rr;
        if s != 0u {
            for(var i: i32 = 0; i < 20; i = i + 1) {
                rad_s.c[i] = ds_mul(rr.c[i], factor[tables[deg_base + u32(i)]]);
            }
        }
        var inv: Jt_3;
        var zeta: Jt_3;
        var r2: Jt_3;
        j_recip_3(&inv, rad_s, prm);
        j_mul_3(&zeta, rs, inv, prm);
        j_mul_3(&r2, rad_s, rad_s, prm);
        var pows: array<Jt_3, 12>;
        fill_powers_3(&pows, zeta, prm.top_pow, prm);
        var alk: array<Jt_3, 8>;
        var blk: array<Jt_3, 8>;
        for(var l: i32 = 1; l <= l_mode; l = l + 1) {
            for(var k: u32 = 0; k < prm.k_count; k = k + 1u) {
                var psi: Jt_3;
                let poly_base = (u32(l) * prm.k_count + k) * prm.poly_stride;
                j_psi_3(&psi, poly_base, u32(l) + k + 1u, &pows);
                var slot: u32 = u32(l - 1) * prm.k_count + k;
                j_mul_3(&(alk[slot]), r2, psi, prm);
                j_mul_3(&(blk[slot]), rad_s, psi, prm);
            }
        }
        var point: u32 = p0 + s;
        var acc: array<dsacc, 9>;
        {
            for(var c: i32 = 0; c < 9; c = c + 1) {
                acc[c] = acc_zero();
            }
        }
        for(var l: i32 = 1; l <= l_mode; l = l + 1) {
            for(var m: i32 = -l; m <= l; m = m + 1) {
                var ylm: Jt_3;
                if s == 0u {
                    ylm = saved[lm_of(l, m)];
                } else {
                    var src: Jt_3 = saved[lm_of(l, m)];
                    for(var i: i32 = 0; i < 20; i = i + 1) {
                        ylm.c[i] = ds_mul(src.c[i], factor[tables[deg_base + u32(i)]]);
                    }
                    for(var i: i32 = 20; i < 20; i = i + 1) {
                        ylm.c[i] = ds_zero();
                    }
                }
                for(var k: u32 = 0; k < prm.k_count; k = k + 1u) {
                    var slot: u32 = u32(l - 1) * prm.k_count + k;
                    emit_mode_3(alk[slot], ylm, 0u, &xyz, l, m, k, point, prm, sqrt2, &acc);
                    emit_mode_3(blk[slot], ylm, 1u, &xyz, l, m, k, point, prm, sqrt2, &acc);
                }
            }
        }
        store_contract(point, prm.n_comp, &acc);
    }
}

fn eval_cached_3_4(ray: u32, prm: BasisParams) {
    if ray >= prm.n_rays {
        return;
    }
    var sqrt2: ds = ds_mk(prm.sqrt2_hi, prm.sqrt2_lo);
    var pmm0: ds = ds_mk(prm.pmm_hi, prm.pmm_lo);
    var p0: u32 = ray * prm.n_per;
    var x0: ds;
    var y0: ds;
    var z0: ds;
    load_point(p0, &x0, &y0, &z0);
    var u: array<Jt_3, 3>;
    var radius0: Jt_3;
    unit_frame_3(&(u), &(radius0), x0, y0, z0, prm);
    var l_mode: i32 = i32(prm.l_mode);
    var l_surf: i32 = i32(prm.l_surf);
    var l_top: i32 = select(l_surf, l_mode, l_mode > l_surf);
    var saved: array<Jt_3, 25>;
    var rr: Jt_3;
    sh_accumulate_3(&rr, &saved, l_mode, l_top, l_surf, &(u), sqrt2, pmm0, prm);
    var rho0: ds = radius0.c[0];
    let deg_base = prm.deg_off;
    for(var s: u32 = 0; s < prm.n_per; s = s + 1u) {
        var px: ds;
        var py: ds;
        var pz: ds;
        load_point(p0 + s, &px, &py, &pz);
        var xyz: array<ds, 3> = array<ds, 3>(px, py, pz);
        var rs: Jt_3 = radius0;
        if s != 0u {
            radial_jet_3(&rs, px, py, pz, prm);
        }
        var rho: ds = rs.c[0];
        var factor: array<ds, 5>;
        factor[0] = ds_f(1.0);
        if s != 0u {
            var ratio: ds = ds_div(rho0, rho);
            for(var d: i32 = 1; d <= 3; d = d + 1) {
                factor[d] = ds_mul(factor[d - 1], ratio);
            }
        }
        var rad_s: Jt_3 = rr;
        if s != 0u {
            for(var i: i32 = 0; i < 20; i = i + 1) {
                rad_s.c[i] = ds_mul(rr.c[i], factor[tables[deg_base + u32(i)]]);
            }
        }
        var inv: Jt_3;
        var zeta: Jt_3;
        var r2: Jt_3;
        j_recip_3(&inv, rad_s, prm);
        j_mul_3(&zeta, rs, inv, prm);
        j_mul_3(&r2, rad_s, rad_s, prm);
        var pows: array<Jt_3, 12>;
        fill_powers_3(&pows, zeta, prm.top_pow, prm);
        var alk: array<Jt_3, 16>;
        var blk: array<Jt_3, 16>;
        for(var l: i32 = 1; l <= l_mode; l = l + 1) {
            for(var k: u32 = 0; k < prm.k_count; k = k + 1u) {
                var psi: Jt_3;
                let poly_base = (u32(l) * prm.k_count + k) * prm.poly_stride;
                j_psi_3(&psi, poly_base, u32(l) + k + 1u, &pows);
                var slot: u32 = u32(l - 1) * prm.k_count + k;
                j_mul_3(&(alk[slot]), r2, psi, prm);
                j_mul_3(&(blk[slot]), rad_s, psi, prm);
            }
        }
        var point: u32 = p0 + s;
        var acc: array<dsacc, 9>;
        {
            for(var c: i32 = 0; c < 9; c = c + 1) {
                acc[c] = acc_zero();
            }
        }
        for(var l: i32 = 1; l <= l_mode; l = l + 1) {
            for(var m: i32 = -l; m <= l; m = m + 1) {
                var ylm: Jt_3;
                if s == 0u {
                    ylm = saved[lm_of(l, m)];
                } else {
                    var src: Jt_3 = saved[lm_of(l, m)];
                    for(var i: i32 = 0; i < 20; i = i + 1) {
                        ylm.c[i] = ds_mul(src.c[i], factor[tables[deg_base + u32(i)]]);
                    }
                    for(var i: i32 = 20; i < 20; i = i + 1) {
                        ylm.c[i] = ds_zero();
                    }
                }
                for(var k: u32 = 0; k < prm.k_count; k = k + 1u) {
                    var slot: u32 = u32(l - 1) * prm.k_count + k;
                    emit_mode_3(alk[slot], ylm, 0u, &xyz, l, m, k, point, prm, sqrt2, &acc);
                    emit_mode_3(blk[slot], ylm, 1u, &xyz, l, m, k, point, prm, sqrt2, &acc);
                }
            }
        }
        store_contract(point, prm.n_comp, &acc);
    }
}

fn accumulate_vd_0(
    f: ptr<function, array<Jt_0, 3>>,
    acc: ptr<function, array<dsacc, 9>>,
    p: BasisParams,
    weight: ds,
    sqrt2: ds,
) {
    var local: u32 = 0;
    if p.want_v != 0u {
        for(var a: i32 = 0; a < 3; a = a + 1) {
            (*acc)[local + u32(a)] = acc_fma((*acc)[local + u32(a)], weight, (*f)[a].c[0]);
        }
        local = local + (3u);
    }
    if p.want_d != 0u {
        var d: array<ds, 6>;
        pack_d_0(f, &d, sqrt2);
        for(var a: u32 = 0; a < 6u; a = a + 1u) {
            (*acc)[local + a] = acc_fma((*acc)[local + a], weight, d[a]);
        }
    }
}

fn emit_mode_2(
    radial: Jt_2,
    ylm: Jt_2,
    tor: u32,
    x0: ptr<function, array<ds, 3>>,
    l: i32,
    m: i32,
    k: u32,
    point: u32,
    p: BasisParams,
    sqrt2: ds,
    acc: ptr<function, array<dsacc, 9>>,
) {
    var phi: Jt_2;
    j_mul_2(&phi, radial, ylm, p);
    var field: array<Jt_0, 3>;
    if tor != 0u {
        toroidal_2(&field, phi, x0, p);
    } else {
        poloidal_2(&field, phi, x0, p);
    }
    var mode: u32 = vel_dof(l, m, tor, k, p.k_count);
    {
        var w: vec2<f32> = coef[mode];
        accumulate_vd_0(&field, acc, p, ds_mk(w.x, w.y), sqrt2);
        return;
    }
    store_vd_0(&field, mode, point, p, sqrt2);
    high_apply_0_0(&field, mode, point, p, sqrt2);
}

fn eval_point_2(gid: u32, prm: BasisParams) {
    if gid >= prm.n_points {
        return;
    }
    var sqrt2: ds = ds_mk(prm.sqrt2_hi, prm.sqrt2_lo);
    var pmm0: ds = ds_mk(prm.pmm_hi, prm.pmm_lo);
    var acc: array<dsacc, 9>;
    {
        for(var c: i32 = 0; c < 9; c = c + 1) {
            acc[c] = acc_zero();
        }
    }
    var px: ds;
    var py: ds;
    var pz: ds;
    load_point(gid, &px, &py, &pz);
    var xyz: array<ds, 3> = array<ds, 3>(px, py, pz);
    var u: array<Jt_2, 3>;
    var radius: Jt_2;
    unit_frame_2(&(u), &(radius), px, py, pz, prm);
    var l_mode: i32 = i32(prm.l_mode);
    var l_surf: i32 = i32(prm.l_surf);
    var l_top: i32 = select(l_surf, l_mode, l_mode > l_surf);
    var rr: Jt_2;
    sh_accumulate_2__saved(&rr, 0, l_top, l_surf, &(u), sqrt2, pmm0, prm);
    var inv: Jt_2;
    var zeta: Jt_2;
    var r2: Jt_2;
    j_recip_2(&inv, rr, prm);
    j_mul_2(&zeta, radius, inv, prm);
    j_mul_2(&r2, rr, rr, prm);
    var pows: array<Jt_2, 12>;
    fill_powers_2(&pows, zeta, prm.top_pow, prm);
    var cre: Jt_2;
    var cim: Jt_2;
    j_const_2(&cre, ds_f(1.0));
    j_const_2(&cim, ds_zero());
    var pmm: ds = pmm0;
    for(var m: i32 = 0; m <= l_mode; m = m + 1) {
        if m > 0 {
            cmul_u_2(&cre, &cim, &(u), prm);
            pmm = ds_mul(pmm, ds_sqrt(ds_div(ds_f(f32(2 * m + 1)), ds_f(f32(2 * m)))));
        }
        var p1: Jt_2;
        var p2: Jt_2;
        j_const_2(&p1, ds_zero());
        j_const_2(&p2, ds_zero());
        for(var l: i32 = m; l <= l_mode; l = l + 1) {
            var p: Jt_2;
            var yp: Jt_2;
            var ym: Jt_2;
            var t: Jt_2;
            legendre_p_2(&p, l, m, p1, p2, u[2], pmm, prm);
            if m == 0 {
                yp = p;
            } else {
                j_mul_2(&t, p, cre, prm);
                j_scale_2(&yp, t, sqrt2);
                j_mul_2(&t, p, cim, prm);
                j_scale_2(&ym, t, sqrt2);
            }
            if l >= 1 {
                for(var k: u32 = 0; k < prm.k_count; k = k + 1u) {
                    var psi: Jt_2;
                    var a: Jt_2;
                    var b: Jt_2;
                    let poly_base = (u32(l) * prm.k_count + k) * prm.poly_stride;
                    j_psi_2(&psi, poly_base, u32(l) + k + 1u, &pows);
                    j_mul_2(&a, r2, psi, prm);
                    j_mul_2(&b, rr, psi, prm);
                    emit_mode_2(a, yp, 0u, &xyz, l, m, k, gid, prm, sqrt2, &acc);
                    emit_mode_2(b, yp, 1u, &xyz, l, m, k, gid, prm, sqrt2, &acc);
                    if m > 0 {
                        emit_mode_2(a, ym, 0u, &xyz, l, -m, k, gid, prm, sqrt2, &acc);
                        emit_mode_2(b, ym, 1u, &xyz, l, -m, k, gid, prm, sqrt2, &acc);
                    }
                }
            }
            p2 = p1;
            p1 = p;
        }
    }
    store_contract(gid, prm.n_comp, &acc);
}

fn eval_point_3(gid: u32, prm: BasisParams) {
    if gid >= prm.n_points {
        return;
    }
    var sqrt2: ds = ds_mk(prm.sqrt2_hi, prm.sqrt2_lo);
    var pmm0: ds = ds_mk(prm.pmm_hi, prm.pmm_lo);
    var acc: array<dsacc, 9>;
    {
        for(var c: i32 = 0; c < 9; c = c + 1) {
            acc[c] = acc_zero();
        }
    }
    var px: ds;
    var py: ds;
    var pz: ds;
    load_point(gid, &px, &py, &pz);
    var xyz: array<ds, 3> = array<ds, 3>(px, py, pz);
    var u: array<Jt_3, 3>;
    var radius: Jt_3;
    unit_frame_3(&(u), &(radius), px, py, pz, prm);
    var l_mode: i32 = i32(prm.l_mode);
    var l_surf: i32 = i32(prm.l_surf);
    var l_top: i32 = select(l_surf, l_mode, l_mode > l_surf);
    var rr: Jt_3;
    sh_accumulate_3__saved(&rr, 0, l_top, l_surf, &(u), sqrt2, pmm0, prm);
    var inv: Jt_3;
    var zeta: Jt_3;
    var r2: Jt_3;
    j_recip_3(&inv, rr, prm);
    j_mul_3(&zeta, radius, inv, prm);
    j_mul_3(&r2, rr, rr, prm);
    var pows: array<Jt_3, 12>;
    fill_powers_3(&pows, zeta, prm.top_pow, prm);
    var cre: Jt_3;
    var cim: Jt_3;
    j_const_3(&cre, ds_f(1.0));
    j_const_3(&cim, ds_zero());
    var pmm: ds = pmm0;
    for(var m: i32 = 0; m <= l_mode; m = m + 1) {
        if m > 0 {
            cmul_u_3(&cre, &cim, &(u), prm);
            pmm = ds_mul(pmm, ds_sqrt(ds_div(ds_f(f32(2 * m + 1)), ds_f(f32(2 * m)))));
        }
        var p1: Jt_3;
        var p2: Jt_3;
        j_const_3(&p1, ds_zero());
        j_const_3(&p2, ds_zero());
        for(var l: i32 = m; l <= l_mode; l = l + 1) {
            var p: Jt_3;
            var yp: Jt_3;
            var ym: Jt_3;
            var t: Jt_3;
            legendre_p_3(&p, l, m, p1, p2, u[2], pmm, prm);
            if m == 0 {
                yp = p;
            } else {
                j_mul_3(&t, p, cre, prm);
                j_scale_3(&yp, t, sqrt2);
                j_mul_3(&t, p, cim, prm);
                j_scale_3(&ym, t, sqrt2);
            }
            if l >= 1 {
                for(var k: u32 = 0; k < prm.k_count; k = k + 1u) {
                    var psi: Jt_3;
                    var a: Jt_3;
                    var b: Jt_3;
                    let poly_base = (u32(l) * prm.k_count + k) * prm.poly_stride;
                    j_psi_3(&psi, poly_base, u32(l) + k + 1u, &pows);
                    j_mul_3(&a, r2, psi, prm);
                    j_mul_3(&b, rr, psi, prm);
                    emit_mode_3(a, yp, 0u, &xyz, l, m, k, gid, prm, sqrt2, &acc);
                    emit_mode_3(b, yp, 1u, &xyz, l, m, k, gid, prm, sqrt2, &acc);
                    if m > 0 {
                        emit_mode_3(a, ym, 0u, &xyz, l, -m, k, gid, prm, sqrt2, &acc);
                        emit_mode_3(b, ym, 1u, &xyz, l, -m, k, gid, prm, sqrt2, &acc);
                    }
                }
            }
            p2 = p1;
            p1 = p;
        }
    }
    store_contract(gid, prm.n_comp, &acc);
}

fn accumulate_vd_2(
    f: ptr<function, array<Jt_2, 3>>,
    acc: ptr<function, array<dsacc, 9>>,
    p: BasisParams,
    weight: ds,
    sqrt2: ds,
) {
    var local: u32 = 0;
    if p.want_v != 0u {
        for(var a: i32 = 0; a < 3; a = a + 1) {
            (*acc)[local + u32(a)] = acc_fma((*acc)[local + u32(a)], weight, (*f)[a].c[0]);
        }
        local = local + (3u);
    }
    if p.want_d != 0u {
        var d: array<ds, 6>;
        pack_d_2(f, &d, sqrt2);
        for(var a: u32 = 0; a < 6u; a = a + 1u) {
            (*acc)[local + a] = acc_fma((*acc)[local + a], weight, d[a]);
        }
    }
}

fn emit_mode_4(
    radial: Jt_4,
    ylm: Jt_4,
    tor: u32,
    x0: ptr<function, array<ds, 3>>,
    l: i32,
    m: i32,
    k: u32,
    point: u32,
    p: BasisParams,
    sqrt2: ds,
    acc: ptr<function, array<dsacc, 9>>,
) {
    var phi: Jt_4;
    j_mul_4(&phi, radial, ylm, p);
    var field: array<Jt_2, 3>;
    if tor != 0u {
        toroidal_4(&field, phi, x0, p);
    } else {
        poloidal_4(&field, phi, x0, p);
    }
    var mode: u32 = vel_dof(l, m, tor, k, p.k_count);
    {
        var w: vec2<f32> = coef[mode];
        accumulate_vd_2(&field, acc, p, ds_mk(w.x, w.y), sqrt2);
        return;
    }
    store_vd_2(&field, mode, point, p, sqrt2);
    high_apply_1_2(&field, mode, point, p, sqrt2);
}

fn eval_point_4(gid: u32, prm: BasisParams) {
    if gid >= prm.n_points {
        return;
    }
    var sqrt2: ds = ds_mk(prm.sqrt2_hi, prm.sqrt2_lo);
    var pmm0: ds = ds_mk(prm.pmm_hi, prm.pmm_lo);
    var acc: array<dsacc, 9>;
    {
        for(var c: i32 = 0; c < 9; c = c + 1) {
            acc[c] = acc_zero();
        }
    }
    var px: ds;
    var py: ds;
    var pz: ds;
    load_point(gid, &px, &py, &pz);
    var xyz: array<ds, 3> = array<ds, 3>(px, py, pz);
    var u: array<Jt_4, 3>;
    var radius: Jt_4;
    unit_frame_4(&(u), &(radius), px, py, pz, prm);
    var l_mode: i32 = i32(prm.l_mode);
    var l_surf: i32 = i32(prm.l_surf);
    var l_top: i32 = select(l_surf, l_mode, l_mode > l_surf);
    var rr: Jt_4;
    sh_accumulate_4__saved(&rr, 0, l_top, l_surf, &(u), sqrt2, pmm0, prm);
    var inv: Jt_4;
    var zeta: Jt_4;
    var r2: Jt_4;
    j_recip_4(&inv, rr, prm);
    j_mul_4(&zeta, radius, inv, prm);
    j_mul_4(&r2, rr, rr, prm);
    var pows: array<Jt_4, 12>;
    fill_powers_4(&pows, zeta, prm.top_pow, prm);
    var cre: Jt_4;
    var cim: Jt_4;
    j_const_4(&cre, ds_f(1.0));
    j_const_4(&cim, ds_zero());
    var pmm: ds = pmm0;
    for(var m: i32 = 0; m <= l_mode; m = m + 1) {
        if m > 0 {
            cmul_u_4(&cre, &cim, &(u), prm);
            pmm = ds_mul(pmm, ds_sqrt(ds_div(ds_f(f32(2 * m + 1)), ds_f(f32(2 * m)))));
        }
        var p1: Jt_4;
        var p2: Jt_4;
        j_const_4(&p1, ds_zero());
        j_const_4(&p2, ds_zero());
        for(var l: i32 = m; l <= l_mode; l = l + 1) {
            var p: Jt_4;
            var yp: Jt_4;
            var ym: Jt_4;
            var t: Jt_4;
            legendre_p_4(&p, l, m, p1, p2, u[2], pmm, prm);
            if m == 0 {
                yp = p;
            } else {
                j_mul_4(&t, p, cre, prm);
                j_scale_4(&yp, t, sqrt2);
                j_mul_4(&t, p, cim, prm);
                j_scale_4(&ym, t, sqrt2);
            }
            if l >= 1 {
                for(var k: u32 = 0; k < prm.k_count; k = k + 1u) {
                    var psi: Jt_4;
                    var a: Jt_4;
                    var b: Jt_4;
                    let poly_base = (u32(l) * prm.k_count + k) * prm.poly_stride;
                    j_psi_4(&psi, poly_base, u32(l) + k + 1u, &pows);
                    j_mul_4(&a, r2, psi, prm);
                    j_mul_4(&b, rr, psi, prm);
                    emit_mode_4(a, yp, 0u, &xyz, l, m, k, gid, prm, sqrt2, &acc);
                    emit_mode_4(b, yp, 1u, &xyz, l, m, k, gid, prm, sqrt2, &acc);
                    if m > 0 {
                        emit_mode_4(a, ym, 0u, &xyz, l, -m, k, gid, prm, sqrt2, &acc);
                        emit_mode_4(b, ym, 1u, &xyz, l, -m, k, gid, prm, sqrt2, &acc);
                    }
                }
            }
            p2 = p1;
            p1 = p;
        }
    }
    store_contract(gid, prm.n_comp, &acc);
}

fn eval_pres0_4(gid: u32, prm: BasisParams) {
    if gid >= prm.n_points {
        return;
    }
    var sqrt2: ds = ds_mk(prm.sqrt2_hi, prm.sqrt2_lo);
    var pmm: ds = ds_mk(prm.pmm_hi, prm.pmm_lo);
    var x: ds;
    var y: ds;
    var z: ds;
    load_point(gid, &x, &y, &z);
    var rho2: ds = ds_addp(ds_addp(ds_sqr(x), ds_sqr(y)), ds_sqr(z));
    var rho: ds = ds_sqrt(rho2);
    var inv: ds = ds_recip(rho);
    var ux: ds = ds_mul(x, inv);
    var uy: ds = ds_mul(y, inv);
    var uz: ds = ds_mul(z, inv);
    var l_mode: i32 = i32(prm.l_mode);
    var l_surf: i32 = i32(prm.l_surf);
    var l_top: i32 = select(l_surf, l_mode, l_mode > l_surf);
    var saved: array<ds, 625>;
    var rr: ds = ds_zero();
    var cre: ds = ds_f(1.0);
    var cim: ds = ds_zero();
    for(var m: i32 = 0; m <= l_top; m = m + 1) {
        if m > 0 {
            var nre: ds = ds_subp(ds_mul(cre, ux), ds_mul(cim, uy));
            var nim: ds = ds_addp(ds_mul(cre, uy), ds_mul(cim, ux));
            cre = nre;
            cim = nim;
            pmm = ds_mul(pmm, ds_sqrt(ds_div(ds_f(f32(2 * m + 1)), ds_f(f32(2 * m)))));
        }
        var p1: ds = ds_zero();
        var p2: ds = ds_zero();
        for(var l: i32 = m; l <= l_top; l = l + 1) {
            var assoc: ds;
            if l == m {
                assoc = pmm;
            } else if l == m + 1 {
                assoc = ds_mul(coef_a(l, m), ds_mul(uz, p1));
            } else {
                assoc = ds_mul(coef_a(l, m), ds_subp(ds_mul(uz, p1), ds_mul(coef_b(l, m), p2)));
            }
            var yp: ds = assoc;
            var ym: ds = ds_zero();
            if m > 0 {
                yp = ds_mul(ds_mul(assoc, cre), sqrt2);
                ym = ds_mul(ds_mul(assoc, cim), sqrt2);
            }
            if l <= l_surf {
                rr = ds_addp(rr, ds_mul(coeff_at(l, m), yp));
                if m > 0 {
                    rr = ds_addp(rr, ds_mul(coeff_at(l, -m), ym));
                }
            }
            if l <= l_mode && l <= 4 {
                saved[lm_of(l, m)] = yp;
                if m > 0 {
                    saved[lm_of(l, -m)] = ym;
                }
            }
            p2 = p1;
            p1 = assoc;
        }
    }
    var zeta: ds = ds_mul(rho, ds_recip(rr));
    var one: ds = ds_subp(ds_f(1.0), zeta);
    var zp: array<ds, 33>;
    zp[0] = ds_f(1.0);
    for(var n: u32 = 1; n <= prm.top_pow && n < 33u; n = n + 1u) {
        zp[n] = ds_mul(zp[n - 1], zeta);
    }
    var n: u32 = prm.n_points;
    var np: u32 = prm.n_out;
    var acc: dsacc = acc_zero();
    for(var l: i32 = 0; l <= l_mode; l = l + 1) {
        for(var m: i32 = -l; m <= l; m = m + 1) {
            var lm: u32 = lm_of(l, m);
            var ylm: ds = saved[lm];
            var lift: ds = ds_mul(zp[u32(l)], ylm);
            {
                var w: vec2<f32> = coef[np + lm];
                acc = acc_fma(acc, ds_mk(w.x, w.y), lift);
            }
            for(var k: u32 = 0; k < prm.k_count; k = k + 1u) {
                let poly_base = (u32(l) * prm.k_count + k) * prm.poly_stride;
                var psi: ds = ds_zero();
                var len: u32 = select(u32(l) + k + 1u, 2u * k + 2u, l == 0);
                for(var j: u32 = 0; j < len; j = j + 1u) {
                    var a: ds = ds_mk(polys[poly_base + j].x, polys[poly_base + j].y);
                    if a.hi != 0.0 || a.lo != 0.0 {
                        psi = ds_addp(psi, ds_mul(a, zp[j]));
                    }
                }
                var phi: ds = ds_mul(ds_mul(one, psi), ylm);
                {
                    var w: vec2<f32> = coef[lm * prm.k_count + k];
                    acc = acc_fma(acc, ds_mk(w.x, w.y), phi);
                }
            }
        }
    }
    {
        var r: ds = acc_done(acc);
        out[gid] = vec2<f32>(r.hi, r.lo);
    }
}

fn eval_pres0_24(gid: u32, prm: BasisParams) {
    if gid >= prm.n_points {
        return;
    }
    var sqrt2: ds = ds_mk(prm.sqrt2_hi, prm.sqrt2_lo);
    var pmm: ds = ds_mk(prm.pmm_hi, prm.pmm_lo);
    var x: ds;
    var y: ds;
    var z: ds;
    load_point(gid, &x, &y, &z);
    var rho2: ds = ds_addp(ds_addp(ds_sqr(x), ds_sqr(y)), ds_sqr(z));
    var rho: ds = ds_sqrt(rho2);
    var inv: ds = ds_recip(rho);
    var ux: ds = ds_mul(x, inv);
    var uy: ds = ds_mul(y, inv);
    var uz: ds = ds_mul(z, inv);
    var l_mode: i32 = i32(prm.l_mode);
    var l_surf: i32 = i32(prm.l_surf);
    var l_top: i32 = select(l_surf, l_mode, l_mode > l_surf);
    var saved: array<ds, 625>;
    var rr: ds = ds_zero();
    var cre: ds = ds_f(1.0);
    var cim: ds = ds_zero();
    for(var m: i32 = 0; m <= l_top; m = m + 1) {
        if m > 0 {
            var nre: ds = ds_subp(ds_mul(cre, ux), ds_mul(cim, uy));
            var nim: ds = ds_addp(ds_mul(cre, uy), ds_mul(cim, ux));
            cre = nre;
            cim = nim;
            pmm = ds_mul(pmm, ds_sqrt(ds_div(ds_f(f32(2 * m + 1)), ds_f(f32(2 * m)))));
        }
        var p1: ds = ds_zero();
        var p2: ds = ds_zero();
        for(var l: i32 = m; l <= l_top; l = l + 1) {
            var assoc: ds;
            if l == m {
                assoc = pmm;
            } else if l == m + 1 {
                assoc = ds_mul(coef_a(l, m), ds_mul(uz, p1));
            } else {
                assoc = ds_mul(coef_a(l, m), ds_subp(ds_mul(uz, p1), ds_mul(coef_b(l, m), p2)));
            }
            var yp: ds = assoc;
            var ym: ds = ds_zero();
            if m > 0 {
                yp = ds_mul(ds_mul(assoc, cre), sqrt2);
                ym = ds_mul(ds_mul(assoc, cim), sqrt2);
            }
            if l <= l_surf {
                rr = ds_addp(rr, ds_mul(coeff_at(l, m), yp));
                if m > 0 {
                    rr = ds_addp(rr, ds_mul(coeff_at(l, -m), ym));
                }
            }
            if l <= l_mode && l <= 24 {
                saved[lm_of(l, m)] = yp;
                if m > 0 {
                    saved[lm_of(l, -m)] = ym;
                }
            }
            p2 = p1;
            p1 = assoc;
        }
    }
    var zeta: ds = ds_mul(rho, ds_recip(rr));
    var one: ds = ds_subp(ds_f(1.0), zeta);
    var zp: array<ds, 33>;
    zp[0] = ds_f(1.0);
    for(var n: u32 = 1; n <= prm.top_pow && n < 33u; n = n + 1u) {
        zp[n] = ds_mul(zp[n - 1], zeta);
    }
    var n: u32 = prm.n_points;
    var np: u32 = prm.n_out;
    var acc: dsacc = acc_zero();
    for(var l: i32 = 0; l <= l_mode; l = l + 1) {
        for(var m: i32 = -l; m <= l; m = m + 1) {
            var lm: u32 = lm_of(l, m);
            var ylm: ds = saved[lm];
            var lift: ds = ds_mul(zp[u32(l)], ylm);
            {
                var w: vec2<f32> = coef[np + lm];
                acc = acc_fma(acc, ds_mk(w.x, w.y), lift);
            }
            for(var k: u32 = 0; k < prm.k_count; k = k + 1u) {
                let poly_base = (u32(l) * prm.k_count + k) * prm.poly_stride;
                var psi: ds = ds_zero();
                var len: u32 = select(u32(l) + k + 1u, 2u * k + 2u, l == 0);
                for(var j: u32 = 0; j < len; j = j + 1u) {
                    var a: ds = ds_mk(polys[poly_base + j].x, polys[poly_base + j].y);
                    if a.hi != 0.0 || a.lo != 0.0 {
                        psi = ds_addp(psi, ds_mul(a, zp[j]));
                    }
                }
                var phi: ds = ds_mul(ds_mul(one, psi), ylm);
                {
                    var w: vec2<f32> = coef[lm * prm.k_count + k];
                    acc = acc_fma(acc, ds_mk(w.x, w.y), phi);
                }
            }
        }
    }
    {
        var r: ds = acc_done(acc);
        out[gid] = vec2<f32>(r.hi, r.lo);
    }
}

// ENTRY basis_vel_o3_l2
@group(0) @binding(0)
var<storage, read> points: array<vec2<f32>>;
@group(0) @binding(1)
var<storage, read> params: BasisParams;
@group(0) @binding(2)
var<storage, read> coeff: array<vec2<f32>>;
@group(0) @binding(3)
var<storage, read> polys: array<vec2<f32>>;
@group(0) @binding(4)
var<storage, read> tables: array<u32>;
@group(0) @binding(5)
var<storage, read_write> out: array<vec2<f32>>;
@compute @workgroup_size(64)
fn basis_vel_o3_l2(@builtin(global_invocation_id) gid3: vec3<u32>) {
    let gid = gid3.x;
    eval_cached_3_2__coef(gid, params);
}

// END ENTRY basis_vel_o3_l2
// ENTRY basis_vel_o3_l4
@group(0) @binding(0)
var<storage, read> points: array<vec2<f32>>;
@group(0) @binding(1)
var<storage, read> params: BasisParams;
@group(0) @binding(2)
var<storage, read> coeff: array<vec2<f32>>;
@group(0) @binding(3)
var<storage, read> polys: array<vec2<f32>>;
@group(0) @binding(4)
var<storage, read> tables: array<u32>;
@group(0) @binding(5)
var<storage, read_write> out: array<vec2<f32>>;
@compute @workgroup_size(64)
fn basis_vel_o3_l4(@builtin(global_invocation_id) gid3: vec3<u32>) {
    let gid = gid3.x;
    eval_cached_3_4__coef(gid, params);
}

// END ENTRY basis_vel_o3_l4
// ENTRY basis_vel_o2
@group(0) @binding(0)
var<storage, read> points: array<vec2<f32>>;
@group(0) @binding(1)
var<storage, read> params: BasisParams;
@group(0) @binding(2)
var<storage, read> coeff: array<vec2<f32>>;
@group(0) @binding(3)
var<storage, read> polys: array<vec2<f32>>;
@group(0) @binding(4)
var<storage, read> tables: array<u32>;
@group(0) @binding(5)
var<storage, read_write> out: array<vec2<f32>>;
@compute @workgroup_size(64)
fn basis_vel_o2(@builtin(global_invocation_id) gid3: vec3<u32>) {
    let gid = gid3.x;
    eval_point_2__coef(gid, params);
}

// END ENTRY basis_vel_o2
// ENTRY basis_vel_o3
@group(0) @binding(0)
var<storage, read> points: array<vec2<f32>>;
@group(0) @binding(1)
var<storage, read> params: BasisParams;
@group(0) @binding(2)
var<storage, read> coeff: array<vec2<f32>>;
@group(0) @binding(3)
var<storage, read> polys: array<vec2<f32>>;
@group(0) @binding(4)
var<storage, read> tables: array<u32>;
@group(0) @binding(5)
var<storage, read_write> out: array<vec2<f32>>;
@compute @workgroup_size(64)
fn basis_vel_o3(@builtin(global_invocation_id) gid3: vec3<u32>) {
    let gid = gid3.x;
    eval_point_3__coef(gid, params);
}

// END ENTRY basis_vel_o3
// ENTRY basis_vel_o4
@group(0) @binding(0)
var<storage, read> points: array<vec2<f32>>;
@group(0) @binding(1)
var<storage, read> params: BasisParams;
@group(0) @binding(2)
var<storage, read> coeff: array<vec2<f32>>;
@group(0) @binding(3)
var<storage, read> polys: array<vec2<f32>>;
@group(0) @binding(4)
var<storage, read> tables: array<u32>;
@group(0) @binding(5)
var<storage, read_write> out: array<vec2<f32>>;
@compute @workgroup_size(64)
fn basis_vel_o4(@builtin(global_invocation_id) gid3: vec3<u32>) {
    let gid = gid3.x;
    eval_point_4__coef(gid, params);
}

// END ENTRY basis_vel_o4
// ENTRY basis_pres0_l4
@group(0) @binding(0)
var<storage, read> points: array<vec2<f32>>;
@group(0) @binding(1)
var<storage, read> params: BasisParams;
@group(0) @binding(2)
var<storage, read> coeff: array<vec2<f32>>;
@group(0) @binding(3)
var<storage, read> polys: array<vec2<f32>>;
@group(0) @binding(4)
var<storage, read_write> out: array<vec2<f32>>;
@compute @workgroup_size(64)
fn basis_pres0_l4(@builtin(global_invocation_id) gid3: vec3<u32>) {
    let gid = gid3.x;
    eval_pres0_4__coef(gid, params);
}

// END ENTRY basis_pres0_l4
// ENTRY basis_pres0_l24
@group(0) @binding(0)
var<storage, read> points: array<vec2<f32>>;
@group(0) @binding(1)
var<storage, read> params: BasisParams;
@group(0) @binding(2)
var<storage, read> coeff: array<vec2<f32>>;
@group(0) @binding(3)
var<storage, read> polys: array<vec2<f32>>;
@group(0) @binding(4)
var<storage, read_write> out: array<vec2<f32>>;
@compute @workgroup_size(64)
fn basis_pres0_l24(@builtin(global_invocation_id) gid3: vec3<u32>) {
    let gid = gid3.x;
    eval_pres0_24__coef(gid, params);
}

// END ENTRY basis_pres0_l24
// ENTRY basis_pres1
@group(0) @binding(0)
var<storage, read> points: array<vec2<f32>>;
@group(0) @binding(1)
var<storage, read> params: BasisParams;
@group(0) @binding(2)
var<storage, read> coeff: array<vec2<f32>>;
@group(0) @binding(3)
var<storage, read> polys: array<vec2<f32>>;
@group(0) @binding(4)
var<storage, read> tables: array<u32>;
@group(0) @binding(5)
var<storage, read_write> out: array<vec2<f32>>;
@compute @workgroup_size(64)
fn basis_pres1(@builtin(global_invocation_id) gid3: vec3<u32>) {
    let gid = gid3.x;
    eval_pres1(gid, params);
}

// END ENTRY basis_pres1
// ENTRY basis_vel_o3_l2_dot
@group(0) @binding(0)
var<storage, read> points: array<vec2<f32>>;
@group(0) @binding(1)
var<storage, read> params: BasisParams;
@group(0) @binding(2)
var<storage, read> coeff: array<vec2<f32>>;
@group(0) @binding(3)
var<storage, read> polys: array<vec2<f32>>;
@group(0) @binding(4)
var<storage, read> tables: array<u32>;
@group(0) @binding(5)
var<storage, read_write> out: array<vec2<f32>>;
@group(0) @binding(6)
var<storage, read> coef: array<vec2<f32>>;
@compute @workgroup_size(64)
fn basis_vel_o3_l2_dot(@builtin(global_invocation_id) gid3: vec3<u32>) {
    let gid = gid3.x;
    eval_cached_3_2(gid, params);
}

// END ENTRY basis_vel_o3_l2_dot
// ENTRY basis_vel_o3_l4_dot
@group(0) @binding(0)
var<storage, read> points: array<vec2<f32>>;
@group(0) @binding(1)
var<storage, read> params: BasisParams;
@group(0) @binding(2)
var<storage, read> coeff: array<vec2<f32>>;
@group(0) @binding(3)
var<storage, read> polys: array<vec2<f32>>;
@group(0) @binding(4)
var<storage, read> tables: array<u32>;
@group(0) @binding(5)
var<storage, read_write> out: array<vec2<f32>>;
@group(0) @binding(6)
var<storage, read> coef: array<vec2<f32>>;
@compute @workgroup_size(64)
fn basis_vel_o3_l4_dot(@builtin(global_invocation_id) gid3: vec3<u32>) {
    let gid = gid3.x;
    eval_cached_3_4(gid, params);
}

// END ENTRY basis_vel_o3_l4_dot
// ENTRY basis_vel_o2_dot
@group(0) @binding(0)
var<storage, read> points: array<vec2<f32>>;
@group(0) @binding(1)
var<storage, read> params: BasisParams;
@group(0) @binding(2)
var<storage, read> coeff: array<vec2<f32>>;
@group(0) @binding(3)
var<storage, read> polys: array<vec2<f32>>;
@group(0) @binding(4)
var<storage, read> tables: array<u32>;
@group(0) @binding(5)
var<storage, read_write> out: array<vec2<f32>>;
@group(0) @binding(6)
var<storage, read> coef: array<vec2<f32>>;
@compute @workgroup_size(64)
fn basis_vel_o2_dot(@builtin(global_invocation_id) gid3: vec3<u32>) {
    let gid = gid3.x;
    eval_point_2(gid, params);
}

// END ENTRY basis_vel_o2_dot
// ENTRY basis_vel_o3_dot
@group(0) @binding(0)
var<storage, read> points: array<vec2<f32>>;
@group(0) @binding(1)
var<storage, read> params: BasisParams;
@group(0) @binding(2)
var<storage, read> coeff: array<vec2<f32>>;
@group(0) @binding(3)
var<storage, read> polys: array<vec2<f32>>;
@group(0) @binding(4)
var<storage, read> tables: array<u32>;
@group(0) @binding(5)
var<storage, read_write> out: array<vec2<f32>>;
@group(0) @binding(6)
var<storage, read> coef: array<vec2<f32>>;
@compute @workgroup_size(64)
fn basis_vel_o3_dot(@builtin(global_invocation_id) gid3: vec3<u32>) {
    let gid = gid3.x;
    eval_point_3(gid, params);
}

// END ENTRY basis_vel_o3_dot
// ENTRY basis_vel_o4_dot
@group(0) @binding(0)
var<storage, read> points: array<vec2<f32>>;
@group(0) @binding(1)
var<storage, read> params: BasisParams;
@group(0) @binding(2)
var<storage, read> coeff: array<vec2<f32>>;
@group(0) @binding(3)
var<storage, read> polys: array<vec2<f32>>;
@group(0) @binding(4)
var<storage, read> tables: array<u32>;
@group(0) @binding(5)
var<storage, read_write> out: array<vec2<f32>>;
@group(0) @binding(6)
var<storage, read> coef: array<vec2<f32>>;
@compute @workgroup_size(64)
fn basis_vel_o4_dot(@builtin(global_invocation_id) gid3: vec3<u32>) {
    let gid = gid3.x;
    eval_point_4(gid, params);
}

// END ENTRY basis_vel_o4_dot
// ENTRY basis_pres0_l4_dot
@group(0) @binding(0)
var<storage, read> points: array<vec2<f32>>;
@group(0) @binding(1)
var<storage, read> params: BasisParams;
@group(0) @binding(2)
var<storage, read> coeff: array<vec2<f32>>;
@group(0) @binding(3)
var<storage, read> polys: array<vec2<f32>>;
@group(0) @binding(4)
var<storage, read> coef: array<vec2<f32>>;
@group(0) @binding(5)
var<storage, read_write> out: array<vec2<f32>>;
@compute @workgroup_size(64)
fn basis_pres0_l4_dot(@builtin(global_invocation_id) gid3: vec3<u32>) {
    let gid = gid3.x;
    eval_pres0_4(gid, params);
}

// END ENTRY basis_pres0_l4_dot
// ENTRY basis_pres0_l24_dot
@group(0) @binding(0)
var<storage, read> points: array<vec2<f32>>;
@group(0) @binding(1)
var<storage, read> params: BasisParams;
@group(0) @binding(2)
var<storage, read> coeff: array<vec2<f32>>;
@group(0) @binding(3)
var<storage, read> polys: array<vec2<f32>>;
@group(0) @binding(4)
var<storage, read> coef: array<vec2<f32>>;
@group(0) @binding(5)
var<storage, read_write> out: array<vec2<f32>>;
@compute @workgroup_size(64)
fn basis_pres0_l24_dot(@builtin(global_invocation_id) gid3: vec3<u32>) {
    let gid = gid3.x;
    eval_pres0_24(gid, params);
}

// END ENTRY basis_pres0_l24_dot
// ENTRY basis_gemv
@group(0) @binding(0)
var<storage, read> basis: array<vec2<f32>>;
@group(0) @binding(1)
var<storage, read> coef: array<vec2<f32>>;
@group(0) @binding(2)
var<storage, read_write> out: array<vec2<f32>>;
@group(0) @binding(3)
var<storage, read> p: GemvParams;
@compute @workgroup_size(64)
fn basis_gemv(@builtin(global_invocation_id) gid3: vec3<u32>) {
    let gid = gid3.x;
    if gid >= p.n_points {
        return;
    }
    for(var c: u32 = 0; c < p.n_comp; c = c + 1u) {
        var acc: dsacc = acc_zero();
        for(var j: u32 = 0; j < p.n_modes; j = j + 1u) {
            var b: vec2<f32> = basis[(j * p.n_comp + c) * p.n_points + gid];
            var w: vec2<f32> = coef[j];
            acc = acc_fma(acc, ds_mk(w.x, w.y), ds_mk(b.x, b.y));
        }
        var r: ds = acc_done(acc);
        out[gid * p.n_comp + c] = vec2<f32>(r.hi, r.lo);
    }
}

// END ENTRY basis_gemv
