// One axis-family sweep of the hyperbolic characteristic relaxation of
// §10.3.2 — the GPU operator of `hyperbolic::sweep_axis`. One thread per
// quadrature sample: the least-squares strain of `u_in`, the two-sign upwind
// interpolation of the `{P±, S_a±}` characteristics on the cell grid, the
// stationary-component exponential integrator, and the Jacobi write to
// `u_out`. The cell grid is a dense index volume over the CellBVH slot space;
// `grid[i]` is the dense cell id + 1 (0 = empty).
//
// The state is f32 on device; the host loop keeps the authoritative f64
// state and the convergence control — this operator only transports it.
// Function-space arrays are avoided in the hot path (dynamic indices spill
// to local memory on Metal): interpolation accumulates in vec4 registers
// and per-component reads go through constant-index select chains.

struct SwParams {
    n: u32,
    axis: u32,
    nx: u32,
    ny: u32,
    nz: u32,
    gx: i32,
    gy: i32,
    gz: i32,
    dt: f32,
    t_s: f32,
    kappa: f32,
    side: f32,
    ox: f32,
    oy: f32,
    oz: f32,
    pad: f32,
}

@group(0) @binding(0) var<uniform> P: SwParams;
@group(0) @binding(1) var<storage, read> pts: array<f32>;
@group(0) @binding(2) var<storage, read> u_in: array<f32>;
@group(0) @binding(3) var<storage, read_write> u_out: array<f32>;
@group(0) @binding(4) var<storage, read> nbr_off: array<u32>;
@group(0) @binding(5) var<storage, read> nbr_idx: array<u32>;
@group(0) @binding(6) var<storage, read> g_inv: array<f32>;
@group(0) @binding(7) var<storage, read> eta: array<f32>;
@group(0) @binding(8) var<storage, read> rho: array<f32>;
@group(0) @binding(9) var<storage, read> tide: array<f32>;
@group(0) @binding(10) var<storage, read> g_eff: array<f32>;
@group(0) @binding(11) var<storage, read> grid: array<u32>;
@group(0) @binding(12) var<storage, read> cell_off: array<u32>;
@group(0) @binding(13) var<storage, read> cell_mem: array<u32>;

const NDIM: u32 = 10u;

// The interpolated upwind state: `q0` = components 0..3, `q1` = 4..7,
// `q2` = 8..9; `ok` flags the IDW path (0 = nearest-member fallback).
struct Up {
    q0: vec4<f32>,
    q1: vec4<f32>,
    q2: vec2<f32>,
    ok: u32,
}

fn pt(i: u32) -> vec3<f32> {
    return vec3<f32>(pts[3u * i], pts[3u * i + 1u], pts[3u * i + 2u]);
}

fn slot_of(x: vec3<f32>) -> vec3<i32> {
    let q = vec3<f32>(P.ox, P.oy, P.oz);
    let s = floor((x - q) / P.side);
    return vec3<i32>(i32(s.x), i32(s.y), i32(s.z));
}

// `CellBvh::cell_of`: the dense grid cell holding `x`, −1 when the slot is
// empty or outside the index volume.
fn cell_of(x: vec3<f32>) -> i32 {
    let l = slot_of(x) - vec3<i32>(P.gx, P.gy, P.gz);
    if l.x < 0 || l.y < 0 || l.z < 0
        || u32(l.x) >= P.nx || u32(l.y) >= P.ny || u32(l.z) >= P.nz {
        return -1;
    }
    return i32(grid[(u32(l.x) * P.ny + u32(l.y)) * P.nz + u32(l.z)]) - 1;
}

// `RayView::interp`: IDW over the members of the 27-cell neighbourhood of
// the slot containing `x`, accumulated in vector registers.
fn interp(x: vec3<f32>) -> Up {
    var out: Up;
    out.q0 = vec4<f32>();
    out.q1 = vec4<f32>();
    out.q2 = vec2<f32>();
    out.ok = 0u;
    if cell_of(x) < 0 {
        return out;
    }
    var wsum = 0.0;
    let base = slot_of(x);
    for (var a = -1; a <= 1; a = a + 1) {
        for (var b = -1; b <= 1; b = b + 1) {
            for (var d = -1; d <= 1; d = d + 1) {
                let l = base + vec3<i32>(a, b, d) - vec3<i32>(P.gx, P.gy, P.gz);
                if l.x < 0 || l.y < 0 || l.z < 0
                    || u32(l.x) >= P.nx || u32(l.y) >= P.ny || u32(l.z) >= P.nz {
                    continue;
                }
                let g = grid[(u32(l.x) * P.ny + u32(l.y)) * P.nz + u32(l.z)];
                if g == 0u {
                    continue;
                }
                let c = g - 1u;
                for (var m = cell_off[c]; m < cell_off[c + 1u]; m = m + 1u) {
                    let j = cell_mem[m];
                    let b0 = NDIM * j;
                    let dx = distance(pt(j), x);
                    let w = 1.0 / (dx + 1e-12 * P.side);
                    out.q0 = out.q0 + w * vec4<f32>(
                        u_in[b0], u_in[b0 + 1u], u_in[b0 + 2u], u_in[b0 + 3u]);
                    out.q1 = out.q1 + w * vec4<f32>(
                        u_in[b0 + 4u], u_in[b0 + 5u], u_in[b0 + 6u], u_in[b0 + 7u]);
                    out.q2 = out.q2 + w * vec2<f32>(u_in[b0 + 8u], u_in[b0 + 9u]);
                    wsum = wsum + w;
                }
            }
        }
    }
    if wsum <= 0.0 {
        return out;
    }
    out.ok = 1u;
    out.q0 = out.q0 / wsum;
    out.q1 = out.q1 / wsum;
    out.q2 = out.q2 / wsum;
    return out;
}

// `RayView::nearest`: nearest member of the 27-cell neighbourhood of `x`'s
// own cell; the global nearest when the local search finds nothing.
fn nearest_idx(x: vec3<f32>) -> u32 {
    var best = 0u;
    var bd = 1e30;
    if cell_of(x) >= 0 {
        let base = slot_of(x);
        for (var a = -1; a <= 1; a = a + 1) {
            for (var b = -1; b <= 1; b = b + 1) {
                for (var d = -1; d <= 1; d = d + 1) {
                    let l = base + vec3<i32>(a, b, d) - vec3<i32>(P.gx, P.gy, P.gz);
                    if l.x < 0 || l.y < 0 || l.z < 0
                        || u32(l.x) >= P.nx || u32(l.y) >= P.ny || u32(l.z) >= P.nz {
                        continue;
                    }
                    let g = grid[(u32(l.x) * P.ny + u32(l.y)) * P.nz + u32(l.z)];
                    if g == 0u {
                        continue;
                    }
                    let c = g - 1u;
                    for (var m = cell_off[c]; m < cell_off[c + 1u]; m = m + 1u) {
                        let j = cell_mem[m];
                        let dd = distance(pt(j), x);
                        if dd < bd {
                            bd = dd;
                            best = j;
                        }
                    }
                }
            }
        }
        if bd < 1e30 {
            return best;
        }
    }
    for (var j = 0u; j < P.n; j = j + 1u) {
        let dd = distance(pt(j), x);
        if dd < bd {
            bd = dd;
            best = j;
        }
    }
    return best;
}

// `RayView::at`: the interpolated state, or the nearest sample's on exit.
fn at(x: vec3<f32>) -> Up {
    var out = interp(x);
    if out.ok == 0u {
        let j = nearest_idx(x);
        let b0 = NDIM * j;
        out.q0 = vec4<f32>(u_in[b0], u_in[b0 + 1u], u_in[b0 + 2u], u_in[b0 + 3u]);
        out.q1 = vec4<f32>(u_in[b0 + 4u], u_in[b0 + 5u], u_in[b0 + 6u], u_in[b0 + 7u]);
        out.q2 = vec2<f32>(u_in[b0 + 8u], u_in[b0 + 9u]);
    }
    return out;
}

// `RayView::strain`: the precomputed LSQ inverse over the fixed stencil.
// `g_inv` uploads row-major (nalgebra `from_row_slice`), so it is bound here
// as a mat3x3 with explicitly transposed columns. Returns the six-vector in
// two vec3s.
fn strain(i: u32) -> array<vec3<f32>, 2> {
    var out = array<vec3<f32>, 2>(vec3<f32>(), vec3<f32>());
    var zero = true;
    for (var k = 0u; k < 9u; k = k + 1u) {
        if g_inv[9u * i + k] != 0.0 {
            zero = false;
        }
    }
    if zero {
        return out;
    }
    let x0 = pt(i);
    var rhs = mat3x3<f32>();
    for (var t = nbr_off[i]; t < nbr_off[i + 1u]; t = t + 1u) {
        let j = nbr_idx[t];
        let dx = pt(j) - x0;
        let dv = vec3<f32>(
            u_in[NDIM * j] - u_in[NDIM * i],
            u_in[NDIM * j + 1u] - u_in[NDIM * i + 1u],
            u_in[NDIM * j + 2u] - u_in[NDIM * i + 2u],
        );
        let w = 1.0 / max(dot(dx, dx), 1e-30);
        for (var a = 0u; a < 3u; a = a + 1u) {
            for (var k = 0u; k < 3u; k = k + 1u) {
                rhs[k][a] = rhs[k][a] + w * dx[a] * dv[k];
            }
        }
    }
    let inv = mat3x3<f32>(
        vec3<f32>(g_inv[9u * i], g_inv[9u * i + 3u], g_inv[9u * i + 6u]),
        vec3<f32>(g_inv[9u * i + 1u], g_inv[9u * i + 4u], g_inv[9u * i + 7u]),
        vec3<f32>(g_inv[9u * i + 2u], g_inv[9u * i + 5u], g_inv[9u * i + 8u]),
    );
    let grad = inv * rhs; // grad[(r,c)] = ∂_r v_c, accessed as grad[c][r]
    let s2 = 0.7071067811865476;
    out[0] = vec3<f32>(grad[0][0], grad[1][1], grad[2][2]);
    out[1] = vec3<f32>(
        s2 * (grad[1][0] + grad[0][1]),
        s2 * (grad[2][0] + grad[0][2]),
        s2 * (grad[2][1] + grad[1][2]),
    );
    return out;
}

// The six-vector component by index (constant-index selects — no dynamic
// array addressing in the hot loop).
fn d_at(d: array<vec3<f32>, 2>, c: u32) -> f32 {
    if c < 3u {
        return d[0][i32(c)];
    }
    return d[1][i32(c) - 3];
}

// `pick3`: component `k` of a vec3 by select chain.
fn pick3(v: vec3<f32>, k: u32) -> f32 {
    return select(select(v.z, v.y, k == 1u), v.x, k == 0u);
}

fn shear_of_axis(a: u32) -> vec2<u32> {
    if a == 0u {
        return vec2<u32>(3u, 4u);
    }
    if a == 1u {
        return vec2<u32>(3u, 5u);
    }
    return vec2<u32>(4u, 5u);
}

@compute @workgroup_size(64)
fn relax_sweep(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= P.n {
        return;
    }
    let axis = P.axis;
    let j1 = (axis + 1u) % 3u;
    let j2 = (axis + 2u) % 3u;
    let sh = shear_of_axis(axis); // the two S_a· shear components
    let x = pt(i);
    let rr = max(rho[i], 1e-30);
    let cp = sqrt(2.0 * eta[i] / (rr * P.t_s) + 1.0 / (rr * P.kappa));
    let cs = sqrt(eta[i] / (rr * P.t_s));
    let ex = exp(-P.dt / P.t_s);
    let d_loc = strain(i);
    let b0 = NDIM * i;
    // Stationary (off-axis-family) stress components relax toward
    // `2ηD + T_tide` — the exponential integrator of §10.3.2. Owned slots
    // stay zero; they are filled by the characteristic amplitudes below.
    var stat = array<f32, 6>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    for (var c = 0u; c < 6u; c = c + 1u) {
        if c == axis || c == sh.x || c == sh.y {
            continue;
        }
        let src = 2.0 * eta[i] * d_at(d_loc, c) + tide[6u * i + c];
        stat[c] = ex * u_in[b0 + 3u + c] + (1.0 - ex) * src;
    }
    // Moving families at both signs: `v` and the owned stress from the
    // upwind characteristic amplitudes. `va/v1/v2` are the axis, j1, j2
    // velocity components; `sa` accumulates S_aa − p.
    let r_p = rr * cp;
    let r_s = rr * cs;
    var sig = 0.0;
    var va = 0.0;
    var v1 = 0.0;
    var v2 = 0.0;
    var sh_a = 0.0;
    var sh_b = 0.0;
    var eu = vec3<f32>(0.0, 0.0, 0.0);
    if axis == 0u {
        eu = vec3<f32>(1.0, 0.0, 0.0);
    } else if axis == 1u {
        eu = vec3<f32>(0.0, 1.0, 0.0);
    } else {
        eu = vec3<f32>(0.0, 0.0, 1.0);
    }
    let gi = vec3<f32>(g_eff[3u * i], g_eff[3u * i + 1u], g_eff[3u * i + 2u]);
    let gp = pick3(gi, axis);
    let g1 = pick3(gi, j1);
    let g2 = pick3(gi, j2);
    for (var si = 0; si < 2; si = si + 1) {
        let sign = select(-1.0, 1.0, si == 0);
        let up_p = at(x - sign * cp * P.dt * eu);
        let up_s = at(x - sign * cs * P.dt * eu);
        // P family on (v_a, σ̃ = S_aa − p). S components 0..2 sit at
        // buffer slots 3..5 → q0.w, q1.x, q1.y.
        let w_p = 0.5 * (pick3(up_p.q0.xyz, axis)
                + sign * (pick3(
                        vec3<f32>(up_p.q0.w, up_p.q1.x, up_p.q1.y), axis)
                    - up_p.q2.y) / r_p)
            + 0.5 * sign * P.dt * gp * rho[i] / r_p;
        va = va + w_p;
        sig = sig + sign * r_p * w_p;
        // S families on (v_j1, S_a j1) and (v_j2, S_a j2): the shear stress
        // components 3..5 sit at buffer slots 6..8 → q1.z, q1.w, q2.x.
        let w1 = 0.5 * (pick3(up_s.q0.xyz, j1)
                + sign * pick3(
                        vec3<f32>(up_s.q1.z, up_s.q1.w, up_s.q2.x), sh.x - 3u) / r_s)
            + 0.5 * sign * P.dt * g1 * rho[i] / r_s;
        v1 = v1 + w1;
        sh_a = sh_a + sign * r_s * w1;
        let w2 = 0.5 * (pick3(up_s.q0.xyz, j2)
                + sign * pick3(
                        vec3<f32>(up_s.q1.z, up_s.q1.w, up_s.q2.x), sh.y - 3u) / r_s)
            + 0.5 * sign * P.dt * g2 * rho[i] / r_s;
        v2 = v2 + w2;
        sh_b = sh_b + sign * r_s * w2;
    }
    let p_new = u_in[b0 + 9u];
    // Write the state slot: v | S | p. The owned stress slots get the
    // characteristic values; the rest take the stationary relax values.
    u_out[b0 + axis] = va;
    u_out[b0 + j1] = v1;
    u_out[b0 + j2] = v2;
    stat[axis] = sig + p_new;
    stat[sh.x] = sh_a;
    stat[sh.y] = sh_b;
    for (var c = 0u; c < 6u; c = c + 1u) {
        u_out[b0 + 3u + c] = stat[c];
    }
    u_out[b0 + 9u] = p_new;
}
