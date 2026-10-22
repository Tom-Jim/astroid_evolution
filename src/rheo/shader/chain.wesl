// #include "gpu/shader/ds.wesl"
// SHARED
const NODE_WORDS: u32 = 38u;
const OUT_DS: u32 = 48u;
const MODE_DS: u32 = 9u;
struct Params {
    node_count: u32,
    l_max: u32,
    max_iter: u32,
    proj_dynamic: u32,
    pad0: u32,
    pad1: u32,
    d_p: array<f32, 2>,
    rho_p: array<f32, 2>,
    mu_s: array<f32, 2>,
    mu_2: array<f32, 2>,
    i0: array<f32, 2>,
    coh: array<f32, 2>,
    p_c: array<f32, 2>,
    gamma_eps: array<f32, 2>,
    p_min: array<f32, 2>,
    rho_bulk: array<f32, 2>,
    h_s: array<f32, 2>,
    tide_coef: array<f32, 2>,
    mu_y: array<f32, 2>,
    tol: array<f32, 2>,
    eps_overhang: array<f32, 2>,
}

struct J2 {
    c: array<ds, 6>,
}

struct J1 {
    c: array<ds, 3>,
}

struct V1 {
    c: array<J1, 3>,
}

struct Modes {
    r: J2,
    z1: J2,
    z4: J2,
    z6: J2,
    z3: J1,
    z5: J1,
    z7: J1,
    p0: J1,
    lap: J1,
}

struct Closure {
    mu_d: ds,
    tau: ds,
    inertial: ds,
}

fn dsv(v: array<f32, 2>) -> ds {
    return ds(v[0], v[1]);
}

fn dot3(a: ptr<function, array<ds, 3>>, b: ptr<function, array<ds, 3>>) -> ds {
    var s: dsacc = acc_zero();
    s = acc_fma(s, (*a)[0], (*b)[0]);
    s = acc_fma(s, (*a)[1], (*b)[1]);
    s = acc_fma(s, (*a)[2], (*b)[2]);
    return acc_done(s);
}

fn j2_zero() -> J2 {
    var r: J2;
    for(var i: i32 = 0; i < 6; i = i + 1) {
        r.c[i] = ds_zero();
    }
    return r;
}

fn j1_zero() -> J1 {
    var r: J1;
    for(var i: i32 = 0; i < 3; i = i + 1) {
        r.c[i] = ds_zero();
    }
    return r;
}

fn j2_const(a: ds) -> J2 {
    var r: J2 = j2_zero();
    r.c[0] = a;
    return r;
}

fn j2_mul(a: J2, b: J2) -> J2 {
    var r: J2;
    r.c[0] = ds_mul(a.c[0], b.c[0]);
    var s: dsacc = acc_zero();
    s = acc_fma(s, a.c[0], b.c[1]);
    s = acc_fma(s, a.c[1], b.c[0]);
    r.c[1] = acc_done(s);
    s = acc_zero();
    s = acc_fma(s, a.c[0], b.c[2]);
    s = acc_fma(s, a.c[2], b.c[0]);
    r.c[2] = acc_done(s);
    s = acc_zero();
    s = acc_fma(s, a.c[0], b.c[3]);
    s = acc_fma(s, a.c[1], b.c[1]);
    s = acc_fma(s, a.c[3], b.c[0]);
    r.c[3] = acc_done(s);
    s = acc_zero();
    s = acc_fma(s, a.c[0], b.c[4]);
    s = acc_fma(s, a.c[1], b.c[2]);
    s = acc_fma(s, a.c[2], b.c[1]);
    s = acc_fma(s, a.c[4], b.c[0]);
    r.c[4] = acc_done(s);
    s = acc_zero();
    s = acc_fma(s, a.c[0], b.c[5]);
    s = acc_fma(s, a.c[2], b.c[2]);
    s = acc_fma(s, a.c[5], b.c[0]);
    r.c[5] = acc_done(s);
    return r;
}

fn j2_scale(a: J2, k: ds) -> J2 {
    var r: J2;
    for(var i: i32 = 0; i < 6; i = i + 1) {
        r.c[i] = ds_mul(a.c[i], k);
    }
    return r;
}

fn j2_sub(a: J2, b: J2) -> J2 {
    var r: J2;
    for(var i: i32 = 0; i < 6; i = i + 1) {
        r.c[i] = ds_sub(a.c[i], b.c[i]);
    }
    return r;
}

fn j2_add(a: J2, b: J2) -> J2 {
    var r: J2;
    for(var i: i32 = 0; i < 6; i = i + 1) {
        r.c[i] = ds_add(a.c[i], b.c[i]);
    }
    return r;
}

fn j2_axpy(acc: ptr<function, J2>, k: ds, y: J2) {
    for(var i: i32 = 0; i < 6; i = i + 1) {
        var s: dsacc = acc_from((*acc).c[i]);
        s = acc_fma(s, k, y.c[i]);
        (*acc).c[i] = acc_done(s);
    }
}

fn j1_axpy(acc: ptr<function, J1>, k: ds, y: J2) {
    for(var i: i32 = 0; i < 3; i = i + 1) {
        var s: dsacc = acc_from((*acc).c[i]);
        s = acc_fma(s, k, y.c[i]);
        (*acc).c[i] = acc_done(s);
    }
}

fn j1_of(a: J2) -> J1 {
    var r: J1;
    r.c[0] = a.c[0];
    r.c[1] = a.c[1];
    r.c[2] = a.c[2];
    return r;
}

fn j1_du(a: J2) -> J1 {
    var r: J1;
    r.c[0] = a.c[1];
    r.c[1] = ds_scale(a.c[3], 2.0);
    r.c[2] = a.c[4];
    return r;
}

fn j1_dv(a: J2) -> J1 {
    var r: J1;
    r.c[0] = a.c[2];
    r.c[1] = a.c[4];
    r.c[2] = ds_scale(a.c[5], 2.0);
    return r;
}

fn j1_mul(a: J1, b: J1) -> J1 {
    var r: J1;
    r.c[0] = ds_mul(a.c[0], b.c[0]);
    var s: dsacc = acc_zero();
    s = acc_fma(s, a.c[0], b.c[1]);
    s = acc_fma(s, a.c[1], b.c[0]);
    r.c[1] = acc_done(s);
    s = acc_zero();
    s = acc_fma(s, a.c[0], b.c[2]);
    s = acc_fma(s, a.c[2], b.c[0]);
    r.c[2] = acc_done(s);
    return r;
}

fn j1_add(a: J1, b: J1) -> J1 {
    var r: J1;
    for(var i: i32 = 0; i < 3; i = i + 1) {
        r.c[i] = ds_add(a.c[i], b.c[i]);
    }
    return r;
}

fn j1_sub(a: J1, b: J1) -> J1 {
    var r: J1;
    for(var i: i32 = 0; i < 3; i = i + 1) {
        r.c[i] = ds_sub(a.c[i], b.c[i]);
    }
    return r;
}

fn j1_scale(a: J1, k: ds) -> J1 {
    var r: J1;
    for(var i: i32 = 0; i < 3; i = i + 1) {
        r.c[i] = ds_mul(a.c[i], k);
    }
    return r;
}

fn j1_neg(a: J1) -> J1 {
    var r: J1;
    for(var i: i32 = 0; i < 3; i = i + 1) {
        r.c[i] = ds_neg(a.c[i]);
    }
    return r;
}

fn j1_recip(a: J1) -> J1 {
    var r: J1;
    var f0: ds = ds_recip(a.c[0]);
    var f02: ds = ds_sqr(f0);
    r.c[0] = f0;
    r.c[1] = ds_neg(ds_mul(a.c[1], f02));
    r.c[2] = ds_neg(ds_mul(a.c[2], f02));
    return r;
}

fn v1_dot(a: V1, b: V1) -> J1 {
    var r: J1 = j1_mul(a.c[0], b.c[0]);
    r = j1_add(r, j1_mul(a.c[1], b.c[1]));
    return j1_add(r, j1_mul(a.c[2], b.c[2]));
}

fn accumulate(m: ptr<function, Modes>, idx: u32, y: J2) {
    let c_base = idx * (MODE_DS * 2);
    j2_axpy(&((*m).r), ds(mode_table[c_base + 0], mode_table[c_base + 1]), y);
    j1_axpy(&((*m).lap), ds(mode_table[c_base + 2], mode_table[c_base + 3]), y);
    j2_axpy(&((*m).z1), ds(mode_table[c_base + 4], mode_table[c_base + 5]), y);
    j1_axpy(&((*m).z3), ds(mode_table[c_base + 6], mode_table[c_base + 7]), y);
    j2_axpy(&((*m).z4), ds(mode_table[c_base + 8], mode_table[c_base + 9]), y);
    j1_axpy(&((*m).z5), ds(mode_table[c_base + 10], mode_table[c_base + 11]), y);
    j2_axpy(&((*m).z6), ds(mode_table[c_base + 12], mode_table[c_base + 13]), y);
    j1_axpy(&((*m).z7), ds(mode_table[c_base + 14], mode_table[c_base + 15]), y);
    j1_axpy(&((*m).p0), ds(mode_table[c_base + 16], mode_table[c_base + 17]), y);
}

fn dynamic_closure(P: Params, shear: ds, p_star: ds, p_tilde: ds) -> Closure {
    var c: Closure;
    var d: ds = dsv(P.d_p);
    var scale: ds = ds_sqrt(ds_div(p_star, dsv(P.rho_p)));
    var df: ds = ds_mul(d, shear);
    var mu_s: ds = dsv(P.mu_s);
    var den: ds = ds_add(ds_mul(dsv(P.i0), scale), df);
    if den.hi > 0.0 {
        c.mu_d = ds_add(mu_s, ds_mul(ds_subp(dsv(P.mu_2), mu_s), ds_div(df, den)));
        c.inertial = ds_div(df, scale);
    } else {
        c.mu_d = mu_s;
        c.inertial = ds_zero();
    }
    c.tau = ds_add(dsv(P.coh), ds_mul(c.mu_d, p_tilde));
    return c;
}

// ENTRY rheo_chain
@group(0) @binding(0)
var<storage, read> nodes: array<f32>;
@group(0) @binding(1)
var<storage, read> P: Params;
@group(0) @binding(2)
var<storage, read> mode_table: array<f32>;
@group(0) @binding(3)
var<storage, read> recur: array<f32>;
@group(0) @binding(4)
var<storage, read_write> out: array<f32>;
@group(0) @binding(5)
var<storage, read_write> stats: array<atomic<u32>>;
fn wr(gid: u32, i: u32, val: ds) {
    let base = gid * (OUT_DS * 2u) + 2u * i;
    out[base] = val.hi;
    out[base + 1u] = val.lo;
}

@compute @workgroup_size(64)
fn rheo_chain(
    @builtin(global_invocation_id) gid3: vec3<u32>,
    @builtin(workgroup_id) wgid3: vec3<u32>,
) {
    let gid = gid3.x;
    var iters_used: u32 = 0;
    var unconverged: u32 = 0;
    var invalid: u32 = 0;
    if gid < P.node_count {
        let nd_base = gid * NODE_WORDS;
        var w0: array<ds, 3> = array<ds, 3>(
                ds(nodes[nd_base + 2 * (0)], nodes[nd_base + 2 * (0) + 1]),
                ds(nodes[nd_base + 2 * (1)], nodes[nd_base + 2 * (1) + 1]),
                ds(nodes[nd_base + 2 * (2)], nodes[nd_base + 2 * (2) + 1]),
            );
        var t1: array<ds, 3> = array<ds, 3>(
                ds(nodes[nd_base + 2 * (3)], nodes[nd_base + 2 * (3) + 1]),
                ds(nodes[nd_base + 2 * (4)], nodes[nd_base + 2 * (4) + 1]),
                ds(nodes[nd_base + 2 * (5)], nodes[nd_base + 2 * (5) + 1]),
            );
        var t2: array<ds, 3> = array<ds, 3>(
                ds(nodes[nd_base + 2 * (6)], nodes[nd_base + 2 * (6) + 1]),
                ds(nodes[nd_base + 2 * (7)], nodes[nd_base + 2 * (7) + 1]),
                ds(nodes[nd_base + 2 * (8)], nodes[nd_base + 2 * (8) + 1]),
            );
        var gv: array<ds, 3> = array<ds, 3>(
                ds(nodes[nd_base + 2 * (9)], nodes[nd_base + 2 * (9) + 1]),
                ds(nodes[nd_base + 2 * (10)], nodes[nd_base + 2 * (10) + 1]),
                ds(nodes[nd_base + 2 * (11)], nodes[nd_base + 2 * (11) + 1]),
            );
        var hs: array<ds, 6> = array<ds, 6>(
                ds(nodes[nd_base + 2 * (12)], nodes[nd_base + 2 * (12) + 1]),
                ds(nodes[nd_base + 2 * (13)], nodes[nd_base + 2 * (13) + 1]),
                ds(nodes[nd_base + 2 * (14)], nodes[nd_base + 2 * (14) + 1]),
                ds(nodes[nd_base + 2 * (15)], nodes[nd_base + 2 * (15) + 1]),
                ds(nodes[nd_base + 2 * (16)], nodes[nd_base + 2 * (16) + 1]),
                ds(nodes[nd_base + 2 * (17)], nodes[nd_base + 2 * (17) + 1]),
            );
        var pressure: ds = ds(nodes[nd_base + 2 * (18)], nodes[nd_base + 2 * (18) + 1]);
        var om: array<J2, 3>;
        for(var i: i32 = 0; i < 3; i = i + 1) {
            om[i].c[0] = w0[i];
            om[i].c[1] = t1[i];
            om[i].c[2] = t2[i];
            om[i].c[3] = ds_scale(ds_neg(w0[i]), 0.5);
            om[i].c[4] = ds_zero();
            om[i].c[5] = om[i].c[3];
        }
        var M: Modes;
        M.r = j2_zero();
        M.z1 = j2_zero();
        M.z4 = j2_zero();
        M.z6 = j2_zero();
        M.z3 = j1_zero();
        M.z5 = j1_zero();
        M.z7 = j1_zero();
        M.p0 = j1_zero();
        M.lap = j1_zero();
        var L: u32 = P.l_max;
        var pmm_base: u32 = (L + 1) * (L + 2) / 2 * 4;
        var cre: J2 = j2_const(ds_f(1.0));
        var cim: J2 = j2_zero();
        for(var m: u32 = 0; m <= L; m = m + 1u) {
            var p1: J2 = j2_zero();
            var p2: J2 = j2_zero();
            for(var l: u32 = m; l <= L; l = l + 1u) {
                var pl: J2;
                if l == m {
                    pl = j2_const(ds(recur[pmm_base + 2 * m], recur[pmm_base + 2 * m + 1]));
                } else {
                    let ab_base = (l * (l + 1) / 2 + m) * 4;
                    var zp: J2 = j2_mul(om[2], p1);
                    if l > m + 1 {
                        j2_axpy(&zp, ds(-recur[ab_base + 2], -recur[ab_base + 3]), p2);
                    }
                    pl = j2_scale(zp, ds(recur[ab_base + 0], recur[ab_base + 1]));
                }
                if m == 0 {
                    accumulate(&M, l * l + l, pl);
                } else {
                    var yc: J2 = j2_mul(pl, cre);
                    var ys: J2 = j2_mul(pl, cim);
                    accumulate(&M, l * l + l + m, yc);
                    accumulate(&M, l * l + l - m, ys);
                }
                p2 = p1;
                p1 = pl;
            }
            var nre: J2 = j2_sub(j2_mul(cre, om[0]), j2_mul(cim, om[1]));
            var nim: J2 = j2_add(j2_mul(cre, om[1]), j2_mul(cim, om[0]));
            cre = nre;
            cim = nim;
        }
        var wu: V1;
        var wv: V1;
        var wj: V1;
        for(var i: i32 = 0; i < 3; i = i + 1) {
            wu.c[i] = j1_du(om[i]);
            wv.c[i] = j1_dv(om[i]);
            wj.c[i] = j1_of(om[i]);
        }
        var guu: J1 = v1_dot(wu, wu);
        var guv: J1 = v1_dot(wu, wv);
        var gvv: J1 = v1_dot(wv, wv);
        var idet: J1 = j1_recip(j1_sub(j1_mul(guu, gvv), j1_mul(guv, guv)));
        var iuu: J1 = j1_mul(gvv, idet);
        var ivv: J1 = j1_mul(guu, idet);
        var iuv: J1 = j1_neg(j1_mul(guv, idet));
        var R1: J1 = j1_of(M.r);
        var Ru: J1 = j1_du(M.r);
        var Rv: J1 = j1_dv(M.r);
        var aR_fu: J1 = j1_du(M.r);
        var aR_fv: J1 = j1_dv(M.r);
        var aR: J1 = j1_add(j1_mul(iuu, aR_fu), j1_mul(iuv, aR_fv));
        var bR: J1 = j1_add(j1_mul(iuv, aR_fu), j1_mul(ivv, aR_fv));
        var aZ1_fu: J1 = j1_du(M.z1);
        var aZ1_fv: J1 = j1_dv(M.z1);
        var aZ1: J1 = j1_add(j1_mul(iuu, aZ1_fu), j1_mul(iuv, aZ1_fv));
        var bZ1: J1 = j1_add(j1_mul(iuv, aZ1_fu), j1_mul(ivv, aZ1_fv));
        var a4_fu: J1 = j1_du(M.z4);
        var a4_fv: J1 = j1_dv(M.z4);
        var a4: J1 = j1_add(j1_mul(iuu, a4_fu), j1_mul(iuv, a4_fv));
        var b4: J1 = j1_add(j1_mul(iuv, a4_fu), j1_mul(ivv, a4_fv));
        var a6_fu: J1 = j1_du(M.z6);
        var a6_fv: J1 = j1_dv(M.z6);
        var a6: J1 = j1_add(j1_mul(iuu, a6_fu), j1_mul(iuv, a6_fv));
        var b6: J1 = j1_add(j1_mul(iuv, a6_fu), j1_mul(ivv, a6_fv));
        var Z1: J1 = j1_of(M.z1);
        var Rinv: J1 = j1_recip(R1);
        var gr2: J1 = j1_add(j1_mul(Ru, aR), j1_mul(Rv, bR));
        var dRdZ: J1 = j1_add(j1_mul(Ru, aZ1), j1_mul(Rv, bZ1));
        var vr: J1 = j1_mul(R1, M.p0);
        vr = j1_sub(vr, j1_scale(dRdZ, ds_f(2.0)));
        vr = j1_sub(vr, j1_mul(Z1, M.lap));
        vr = j1_sub(vr, j1_mul(j1_mul(M.z3, gr2), Rinv));
        var gradR: V1;
        var vs: V1;
        var wt: V1;
        for(var i: i32 = 0; i < 3; i = i + 1) {
            gradR.c[i] = j1_add(j1_mul(aR, wu.c[i]), j1_mul(bR, wv.c[i]));
            var g4: J1 = j1_add(j1_mul(a4, wu.c[i]), j1_mul(b4, wv.c[i]));
            var g6: J1 = j1_add(j1_mul(a6, wu.c[i]), j1_mul(b6, wv.c[i]));
            vs.c[i] = j1_add(j1_mul(R1, g4), j1_mul(M.z5, gradR.c[i]));
            wt.c[i] = j1_add(j1_mul(R1, g6), j1_mul(M.z7, gradR.c[i]));
        }
        var v: V1;
        v.c[0] = j1_sub(j1_mul(wt.c[1], wj.c[2]), j1_mul(wt.c[2], wj.c[1]));
        v.c[1] = j1_sub(j1_mul(wt.c[2], wj.c[0]), j1_mul(wt.c[0], wj.c[2]));
        v.c[2] = j1_sub(j1_mul(wt.c[0], wj.c[1]), j1_mul(wt.c[1], wj.c[0]));
        for(var i: i32 = 0; i < 3; i = i + 1) {
            v.c[i] = j1_add(v.c[i], vs.c[i]);
            v.c[i] = j1_add(v.c[i], j1_mul(vr, wj.c[i]));
        }
        var R0: ds = M.r.c[0];
        var Ru0: ds = M.r.c[1];
        var Rv0: ds = M.r.c[2];
        var e1: array<ds, 3>;
        var e2: array<ds, 3>;
        var vv0: array<ds, 3>;
        var vu: array<ds, 3>;
        var vvv: array<ds, 3>;
        for(var i: i32 = 0; i < 3; i = i + 1) {
            var s: dsacc = acc_zero();
            s = acc_fma(s, Ru0, w0[i]);
            s = acc_fma(s, R0, t1[i]);
            e1[i] = acc_done(s);
            s = acc_zero();
            s = acc_fma(s, Rv0, w0[i]);
            s = acc_fma(s, R0, t2[i]);
            e2[i] = acc_done(s);
            vv0[i] = v.c[i].c[0];
            vu[i] = v.c[i].c[1];
            vvv[i] = v.c[i].c[2];
        }
        var g11: ds = dot3(&e1, &e1);
        var g12: ds = dot3(&e1, &e2);
        var g22: ds = dot3(&e2, &e2);
        var det: ds;
        {
            var s: dsacc = acc_zero();
            s = acc_fma(s, g11, g22);
            s = acc_fnma(s, g12, g12);
            det = acc_done(s);
        }
        var sqrt_g: ds = ds_sqrt(det);
        var idt: ds = ds_recip(det);
        var gi11: ds = ds_mul(g22, idt);
        var gi22: ds = ds_mul(g11, idt);
        var gi12: ds = ds_neg(ds_mul(g12, idt));
        var d11: ds = dot3(&e1, &vu);
        var d22: ds = dot3(&e2, &vvv);
        var d12: ds = ds_scale(ds_add(dot3(&e1, &vvv), dot3(&e2, &vu)), 0.5);
        var m11: ds;
        var m12: ds;
        var m21: ds;
        var m22: ds;
        {
            var s: dsacc = acc_zero();
            s = acc_fma(s, gi11, d11);
            s = acc_fma(s, gi12, d12);
            m11 = acc_done(s);
            s = acc_zero();
            s = acc_fma(s, gi11, d12);
            s = acc_fma(s, gi12, d22);
            m12 = acc_done(s);
            s = acc_zero();
            s = acc_fma(s, gi12, d11);
            s = acc_fma(s, gi22, d12);
            m21 = acc_done(s);
            s = acc_zero();
            s = acc_fma(s, gi12, d12);
            s = acc_fma(s, gi22, d22);
            m22 = acc_done(s);
        }
        var tr_m: ds = ds_add(m11, m22);
        var tr_m2: ds;
        {
            var s: dsacc = acc_zero();
            s = acc_fma(s, m11, m11);
            s = acc_fma(s, m22, m22);
            s = acc_fma(s, ds_scale(m12, 2.0), m21);
            tr_m2 = acc_done(s);
        }
        var s0: ds = ds_scale(ds_add(ds_sqr(tr_m), tr_m2), 2.0);
        var nrm: array<ds, 3>;
        {
            var cx: ds = ds_subp(ds_mul(e1[1], e2[2]), ds_mul(e1[2], e2[1]));
            var cy: ds = ds_subp(ds_mul(e1[2], e2[0]), ds_mul(e1[0], e2[2]));
            var cz: ds = ds_subp(ds_mul(e1[0], e2[1]), ds_mul(e1[1], e2[0]));
            var inv: ds = ds_recip(sqrt_g);
            nrm[0] = ds_mul(cx, inv);
            nrm[1] = ds_mul(cy, inv);
            nrm[2] = ds_mul(cz, inv);
        }
        var trh3: ds = ds_div(ds_add(ds_add(hs[0], hs[1]), hs[2]), ds_f(3.0));
        var dh: array<array<ds, 3>, 3>;
        dh[0][0] = ds_subp(hs[0], trh3);
        dh[1][1] = ds_subp(hs[1], trh3);
        dh[2][2] = ds_subp(hs[2], trh3);
        dh[0][1] = hs[3];
        dh[1][0] = hs[3];
        dh[0][2] = hs[4];
        dh[2][0] = hs[4];
        dh[1][2] = hs[5];
        dh[2][1] = hs[5];
        var tcoef: ds = dsv(P.tide_coef);
        var hn: array<ds, 3>;
        for(var i: i32 = 0; i < 3; i = i + 1) {
            var s: dsacc = acc_zero();
            s = acc_fma(s, dh[i][0], nrm[0]);
            s = acc_fma(s, dh[i][1], nrm[1]);
            s = acc_fma(s, dh[i][2], nrm[2]);
            hn[i] = acc_done(s);
        }
        var ta1: ds = ds_mul(tcoef, dot3(&hn, &e1));
        var ta2: ds = ds_mul(tcoef, dot3(&hn, &e2));
        var tg: ds;
        {
            var s: dsacc = acc_zero();
            s = acc_fma(s, gi11, ds_sqr(ta1));
            s = acc_fma(s, gi22, ds_sqr(ta2));
            s = acc_fma(s, ds_scale(gi12, 2.0), ds_mul(ta1, ta2));
            tg = acc_done(s);
        }
        var frob2: ds;
        {
            var s: dsacc = acc_zero();
            for(var i: i32 = 0; i < 3; i = i + 1) {
                for(var j: i32 = 0; j < 3; j = j + 1) {
                    s = acc_fma(s, dh[i][j], dh[i][j]);
                }
            }
            frob2 = acc_done(s);
        }
        var tide_frob: ds = ds_mul(ds_abs(tcoef), ds_sqrt(frob2));
        var p_min: ds = dsv(P.p_min);
        var p_tilde: ds = ds_max0(pressure);
        if ds_gt(p_min, p_tilde) {
            p_tilde = p_min;
        }
        var p_star: ds = ds_add(p_tilde, dsv(P.p_c));
        var eps2: ds = ds_sqr(dsv(P.gamma_eps));
        var g2: ds = s0;
        var converged: bool = true;
        if tg.hi > 0.0 {
            converged = false;
            var tol: ds = dsv(P.tol);
            for(var it: u32 = 0; it < P.max_iter; it = it + 1u) {
                var cl: Closure = dynamic_closure(P, ds_sqrt(g2), p_star, p_tilde);
                var ratio: ds = ds_div(tg, ds_sqr(cl.tau));
                var next: ds = ds_add(s0, ds_mul(ratio, ds_add(g2, eps2)));
                iters_used = it + 1;
                if !(next.hi == next.hi) || next.hi > 1.0e30 {
                    g2 = next;
                    break;
                }
                var diff: ds = ds_abs(ds_subp(next, g2));
                var scale: ds = ds_add(next, eps2);
                g2 = next;
                if !ds_gt(diff, ds_mul(tol, scale)) {
                    converged = true;
                    break;
                }
            }
        }
        if !converged {
            unconverged = 1;
        }
        var shear: ds = ds_sqrt(g2);
        var cl: Closure = dynamic_closure(P, shear, p_star, p_tilde);
        var root: ds = ds_sqrt(ds_add(g2, eps2));
        var eta: ds = ds_div(cl.tau, root);
        var half_inv_eta: ds = ds_scale(ds_div(root, cl.tau), 0.5);
        var dn1: ds = ds_neg(ds_mul(ta1, half_inv_eta));
        var dn2: ds = ds_neg(ds_mul(ta2, half_inv_eta));
        var gn: ds = dot3(&gv, &nrm);
        var gg: ds = dot3(&gv, &gv);
        var gnorm: ds = ds_sqrt(gg);
        var p_ref: ds = ds_mul(ds_mul(dsv(P.rho_bulk), ds_abs(gn)), dsv(P.h_s));
        var p_star_ref: ds = ds_add(p_ref, dsv(P.p_c));
        if ds_gt(p_min, p_star_ref) {
            p_star_ref = p_min;
        }
        var mu_proj: ds = dsv(P.mu_y);
        if P.proj_dynamic != 0 {
            mu_proj = cl.mu_d;
        }
        var num: ds = ds_subp(
                ds_addp(dsv(P.coh), ds_mul(mu_proj, p_ref)),
                ds_mul(tide_frob, ds_mk(0.707106769, 1.21016175e-08)),
            );
        var mu_eff: ds = ds_div(ds_max0(num), p_star_ref);
        var phi: ds = ds_atan_pos(mu_eff);
        var mu2: ds = ds_sqr(mu_eff);
        var sin2: ds = ds_div(mu2, ds_add(ds_f(1.0), mu2));
        var sin_phi: ds = ds_sqrt(sin2);
        var gxn: array<ds, 3> = array<ds, 3>(
                ds_subp(ds_mul(gv[1], nrm[2]), ds_mul(gv[2], nrm[1])),
                ds_subp(ds_mul(gv[2], nrm[0]), ds_mul(gv[0], nrm[2])),
                ds_subp(ds_mul(gv[0], nrm[1]), ds_mul(gv[1], nrm[0])),
            );
        var sin_slope: ds = ds_div(ds_sqrt(dot3(&gxn, &gxn)), gnorm);
        var slack: ds = ds_max0(ds_subp(sin_slope, sin_phi));
        var overhang: ds = ds_subp(ds_neg(ds_div(gn, gnorm)), dsv(P.eps_overhang));
        var vr0: ds = vr.c[0];
        var vn: ds = dot3(&vv0, &nrm);
        let o_base = gid * (OUT_DS * 2);
        wr(gid, 0u, R0);
        wr(gid, 1u, nrm[0]);
        wr(gid, 2u, nrm[1]);
        wr(gid, 3u, nrm[2]);
        wr(gid, 4u, sqrt_g);
        wr(gid, 5u, vv0[0]);
        wr(gid, 6u, vv0[1]);
        wr(gid, 7u, vv0[2]);
        wr(gid, 8u, e1[0]);
        wr(gid, 9u, e1[1]);
        wr(gid, 10u, e1[2]);
        wr(gid, 11u, e2[0]);
        wr(gid, 12u, e2[1]);
        wr(gid, 13u, e2[2]);
        wr(gid, 14u, d11);
        wr(gid, 15u, d12);
        wr(gid, 16u, d22);
        wr(gid, 17u, ds_neg(tr_m));
        wr(gid, 18u, dn1);
        wr(gid, 19u, dn2);
        wr(gid, 20u, shear);
        wr(gid, 21u, cl.inertial);
        wr(gid, 22u, cl.mu_d);
        wr(gid, 23u, eta);
        wr(gid, 24u, cl.tau);
        wr(gid, 25u, p_tilde);
        wr(gid, 26u, p_star);
        wr(gid, 27u, p_ref);
        wr(gid, 28u, mu_proj);
        wr(gid, 29u, mu_eff);
        wr(gid, 30u, phi);
        wr(gid, 31u, sin2);
        wr(gid, 32u, sin_slope);
        wr(gid, 33u, slack);
        wr(gid, 34u, overhang);
        wr(gid, 35u, tide_frob);
        wr(gid, 36u, vr0);
        wr(gid, 37u, vn);
        wr(gid, 38u, s0);
        wr(gid, 39u, tg);
        wr(gid, 40u, ds_f(f32(iters_used)));
        wr(gid, 41u, ds_f(select(0.0, 1.0, converged)));
        wr(gid, 42u, sin_phi);
        wr(gid, 43u, g11);
        wr(gid, 44u, g12);
        wr(gid, 45u, g22);
        if !(R0.hi > 0.0) || !(det.hi > 0.0) || !(shear.hi == shear.hi) {
            invalid = 1;
        }
    }
    atomicMax(&stats[0], iters_used);
    atomicAdd(&stats[1], unconverged);
    atomicAdd(&stats[2], invalid);
}

// END ENTRY rheo_chain
