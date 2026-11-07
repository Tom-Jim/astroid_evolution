// The (G1) geometry-adapted divergence-free basis, evaluated on the GPU.
//
// Modes (packed 8 f32 each):
//   kind 0 — rigid:  B = a + w × x            (f0..f2 = a, f3..f5 = w)
//   kind 1 — kernel: B = ∇×(W(s) e) = k(s)·(r × e), r = x − c, s = |r|,
//                    Wendland C² profile W = (1 − s/h)⁴(4s/h + 1) giving
//                    k = −20q³/h², k′ = 60q²/h³, k″ = −120q/h⁴.
//                    (f0..f2 = c, f3..f5 = e, f6 = h)
//   kind 2 — flux:   B = −∇|x − z|⁻¹ = −r/r³  (f0..f2 = z)
//
// One thread per point walks all modes and writes the row blocks column-major
// like `VelRows<f32>`: out[b * (block_rows * n_points) + block_row * n_points + p].
// gd holds ∂_k D in the same scaled 6-order per k: row 18p + 6k + c.

struct GbParams {
    n_points: u32,
    n_modes: u32,
    want_v: u32,
    want_d: u32,
    want_lap: u32,
    want_gd: u32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<storage, read> gb_pts: array<f32>;
@group(0) @binding(1) var<uniform> gb_p: GbParams;
@group(0) @binding(2) var<storage, read> gb_modes: array<f32>;
@group(0) @binding(3) var<storage, read_write> out_v: array<f32>;
@group(0) @binding(4) var<storage, read_write> out_d: array<f32>;
@group(0) @binding(5) var<storage, read_write> out_lap: array<f32>;
@group(0) @binding(6) var<storage, read_write> out_gd: array<f32>;

const SQRT2: f32 = 1.4142135623730951;

fn axis(i: u32) -> vec3<f32> {
    var a = vec3<f32>(0.0);
    a[i] = 1.0;
    return a;
}

@compute @workgroup_size(64)
fn gbasis_rows(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = gid.x;
    if p >= gb_p.n_points {
        return;
    }
    let x = vec3<f32>(gb_pts[3u * p], gb_pts[3u * p + 1u], gb_pts[3u * p + 2u]);
    let nm = gb_p.n_modes;
    let np = gb_p.n_points;
    for (var b = 0u; b < nm; b = b + 1u) {
        let m = b * 8u;
        let kind = u32(gb_modes[m + 7u]);
        let f0 = vec3<f32>(gb_modes[m], gb_modes[m + 1u], gb_modes[m + 2u]);
        let f1 = vec3<f32>(gb_modes[m + 3u], gb_modes[m + 4u], gb_modes[m + 5u]);
        var val = vec3<f32>(0.0);
        var d6 = array<f32, 6>();
        var lap = vec3<f32>(0.0);
        var g18 = array<f32, 18>();
        if kind == 0u {
            val = f0 + cross(f1, x);
        } else if kind == 2u {
            let r = x - f0;
            let r2 = dot(r, r);
            if r2 > 1e-20 {
                let ir3 = 1.0 / (r2 * sqrt(r2));
                let ir5 = ir3 / r2;
                val = -r * ir3;
                var j = mat3x3<f32>();
                for (var i = 0u; i < 3u; i = i + 1u) {
                    for (var jj = 0u; jj < 3u; jj = jj + 1u) {
                        let dij = select(0.0, 1.0, i == jj);
                        j[i][jj] = -dij * ir3 + 3.0 * r[i] * r[jj] * ir5;
                    }
                }
                d6[0] = j[0][0];
                d6[1] = j[1][1];
                d6[2] = j[2][2];
                d6[3] = SQRT2 * 0.5 * (j[0][1] + j[1][0]);
                d6[4] = SQRT2 * 0.5 * (j[0][2] + j[2][0]);
                d6[5] = SQRT2 * 0.5 * (j[1][2] + j[2][1]);
                let ir7 = ir5 / r2;
                for (var kk = 0u; kk < 3u; kk = kk + 1u) {
                    var t = mat3x3<f32>();
                    for (var i = 0u; i < 3u; i = i + 1u) {
                        for (var jj = 0u; jj < 3u; jj = jj + 1u) {
                            let dki = select(0.0, 1.0, kk == i);
                            let dkj = select(0.0, 1.0, kk == jj);
                            let dij = select(0.0, 1.0, i == jj);
                            t[i][jj] = 3.0 * r[kk] * dij * ir5
                                + 3.0 * (dki * r[jj] + r[i] * dkj) * ir5
                                - 15.0 * r[kk] * r[i] * r[jj] * ir7;
                        }
                    }
                    g18[6u * kk] = t[0][0];
                    g18[6u * kk + 1u] = t[1][1];
                    g18[6u * kk + 2u] = t[2][2];
                    g18[6u * kk + 3u] = SQRT2 * 0.5 * (t[0][1] + t[1][0]);
                    g18[6u * kk + 4u] = SQRT2 * 0.5 * (t[0][2] + t[2][0]);
                    g18[6u * kk + 5u] = SQRT2 * 0.5 * (t[1][2] + t[2][1]);
                }
            }
        } else {
            let c = f0;
            let e = f1;
            let h = gb_modes[m + 6u];
            let r = x - c;
            let s2 = dot(r, r);
            if s2 < h * h && s2 > 1e-30 {
                let s = sqrt(s2);
                let q = 1.0 - s / h;
                let h2 = h * h;
                let k = -20.0 * q * q * q / h2;
                let dk = 60.0 * q * q / (h2 * h);
                let ddk = -120.0 * q / (h2 * h2);
                let rh = r / s;
                let rx_e = cross(r, e);
                val = k * rx_e;
                lap = (ddk + 2.0 * dk / s) * rx_e + 2.0 * dk * cross(rh, e);
                // jac[i][j] = ∂_i B_j  (transpose of the CPU layout; only the
                // symmetric part is consumed).
                var j = mat3x3<f32>();
                for (var i = 0u; i < 3u; i = i + 1u) {
                    let ci = cross(axis(i), e);
                    for (var jj = 0u; jj < 3u; jj = jj + 1u) {
                        j[i][jj] = dk * rh[i] * rx_e[jj] + k * ci[jj];
                    }
                }
                d6[0] = j[0][0];
                d6[1] = j[1][1];
                d6[2] = j[2][2];
                d6[3] = SQRT2 * 0.5 * (j[0][1] + j[1][0]);
                d6[4] = SQRT2 * 0.5 * (j[0][2] + j[2][0]);
                d6[5] = SQRT2 * 0.5 * (j[1][2] + j[2][1]);
                if gb_p.want_gd == 1u {
                    for (var kk = 0u; kk < 3u; kk = kk + 1u) {
                        let ck = cross(axis(kk), e);
                        var t = mat3x3<f32>();
                        for (var i = 0u; i < 3u; i = i + 1u) {
                            let ci = cross(axis(i), e);
                            let dki = select(0.0, 1.0, kk == i);
                            let pk = dki - rh[kk] * rh[i];
                            for (var jj = 0u; jj < 3u; jj = jj + 1u) {
                                t[i][jj] = ddk * rh[kk] * rh[i] * rx_e[jj]
                                    + (dk / s) * pk * rx_e[jj]
                                    + dk * rh[i] * ck[jj]
                                    + dk * rh[kk] * ci[jj];
                            }
                        }
                        g18[6u * kk] = t[0][0];
                        g18[6u * kk + 1u] = t[1][1];
                        g18[6u * kk + 2u] = t[2][2];
                        g18[6u * kk + 3u] = SQRT2 * 0.5 * (t[0][1] + t[1][0]);
                        g18[6u * kk + 4u] = SQRT2 * 0.5 * (t[0][2] + t[2][0]);
                        g18[6u * kk + 5u] = SQRT2 * 0.5 * (t[1][2] + t[2][1]);
                    }
                }
            }
        }
        if gb_p.want_v == 1u {
            for (var a = 0u; a < 3u; a = a + 1u) {
                out_v[b * 3u * np + 3u * p + a] = val[a];
            }
        }
        if gb_p.want_d == 1u {
            for (var a = 0u; a < 6u; a = a + 1u) {
                out_d[b * 6u * np + 6u * p + a] = d6[a];
            }
        }
        if gb_p.want_lap == 1u {
            for (var a = 0u; a < 3u; a = a + 1u) {
                out_lap[b * 3u * np + 3u * p + a] = lap[a];
            }
        }
        if gb_p.want_gd == 1u {
            for (var a = 0u; a < 18u; a = a + 1u) {
                out_gd[b * 18u * np + 18u * p + a] = g18[a];
            }
        }
    }
}
