// Field of the uniform polyhedron with unit density (Werner and Scheeres 1997),
// the exact part of the density-difference split of MS §3.9:
//
//   U(x) = G ρ(x) Φ_V(x) + G Σ_q ω_q Σ_ℓ ∫ [ρ(x − R u) − ρ(x)] (R, u, T(u)/R) dR.
//
// Φ_V carries the whole singular structure at x (contact term, the logarithm at an
// edge, the face jump), so the ray part integrates a function that vanishes at
// R = 0. Unit frame, G = 1. One thread per point walks the face tree of
// `rtfp/tree.rs`: a node seen under h/|x − c| < θ contributes its octupole
// expansion of the four surface charges n_x, n_y, n_z, n·(y − c), with
// Φ_V = ½(P − (x − c)·S), ∇Φ_V = −S, ∇∇Φ_V = −∇S; the faces of a near leaf are
// summed in closed form. Partial sums are f32 and flushed to double-single.
//
// Face record (24 floats): vertices a, b, c; unit normal n; edge normals
// m_e = ê × n for the edges ab, bc, ca; edge lengths. Faces are in tree order.
// Output per point: double-single (Φ, ∇Φ, ∇∇Φ (xx, yy, zz, xy, xz, yz)), then the
// total solid angle Σ ω (4π inside, 0 outside; −tr ∇∇Φ_V for a far node).

// #include "gpu/shader/ds.wesl"

struct UniformParams {
    point_count: u32,
    node_count: u32,
    theta: f32,
    pad: u32,
}

@group(0) @binding(0)
var<storage, read> upoints: array<vec4<f32>>;
@group(0) @binding(1)
var<storage, read> own_face: array<u32>;
@group(0) @binding(2)
var<storage, read> faces: array<f32>;
@group(0) @binding(3)
var<storage, read> tnodes: array<f32>;
@group(0) @binding(4)
var<uniform> uparams: UniformParams;
@group(0) @binding(5)
var<storage, read_write> uout: array<f32>;

const FACE_WORDS: u32 = 24u;
const NODE_WORDS: u32 = 88u;
const UNIFORM_STRIDE: u32 = 21u;
const STACK: u32 = 64u;
const FLUSH: u32 = 8u;

// f32 partial sums: Φ, ∇Φ, diagonal and off-diagonal of ∇∇Φ, Σω.
struct Part {
    u: f32,
    g: vec3<f32>,
    h0: vec3<f32>,
    h1: vec3<f32>,
    omega: f32,
}

fn part_zero() -> Part {
    return Part(0.0, vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0), 0.0);
}

// ln((a + b + e)/(a + b − e)) with a + b − e = 2(ab + r₁·r₂)/(a + b + e), which keeps
// its digits near an edge line.
fn edge_log(r1: vec3<f32>, r2: vec3<f32>, a: f32, b: f32, e: f32) -> f32 {
    let s = a + b + e;
    let q = 2.0 * (a * b + dot(r1, r2));
    if !(q > 0.0) {
        return 0.0;
    }
    return log(s * s / q);
}

fn fvec(o: u32) -> vec3<f32> {
    return vec3<f32>(faces[o], faces[o + 1u], faces[o + 2u]);
}

// Closed-form term of face f. `own` drops ω for the face holding x: from its own
// plane the face subtends ±2π, and the mean of the one-sided Hessians takes 0.
fn face_term(acc: ptr<function, Part>, f: u32, x: vec3<f32>, own: bool) {
    let o = f * FACE_WORDS;
    let ra = fvec(o) - x;
    let rb = fvec(o + 3u) - x;
    let rc = fvec(o + 6u) - x;
    let n = fvec(o + 9u);
    let la = length(ra);
    let lb = length(rb);
    let lc = length(rc);
    // Signed solid angle (Van Oosterom and Strackee).
    var omega = 0.0;
    if !own {
        let num = dot(ra, cross(rb, rc));
        let den = la * lb * lc + dot(ra, rb) * lc + dot(rb, rc) * la + dot(rc, ra) * lb;
        // At a vertex of the face both vanish; the term is multiplied by h = 0.
        if num != 0.0 || den != 0.0 {
            omega = 2.0 * atan2(num, den);
        }
    }
    let h = dot(n, ra);
    var along = 0.0;
    var m_sum = vec3<f32>(0.0);
    let rv = array<vec3<f32>, 3>(ra, rb, rc);
    let lv = array<f32, 3>(la, lb, lc);
    for(var e = 0u; e < 3u; e = e + 1u) {
        let e2 = (e + 1u) % 3u;
        let m = fvec(o + 12u + 3u * e);
        let l = edge_log(rv[e], rv[e2], lv[e], lv[e2], faces[o + 21u + e]);
        along = along + dot(m, rv[e]) * l;
        m_sum = m_sum + m * l;
    }
    (*acc).u = (*acc).u + 0.5 * h * along - 0.5 * h * h * omega;
    (*acc).g = (*acc).g + n * (h * omega - along);
    // ∇∇Φ = Σ_f [ sym(n ⊗ Σ_e m_e L_e) − n nᵀ ω ].
    (*acc).h0 = (*acc).h0 + n * m_sum - n * n * omega;
    (*acc).h1 = (*acc).h1 + 0.5 * vec3<f32>(
            n.x * m_sum.y + n.y * m_sum.x,
            n.x * m_sum.z + n.z * m_sum.x,
            n.y * m_sum.z + n.z * m_sum.y,
        ) - vec3<f32>(n.x * n.y, n.x * n.z, n.y * n.z) * omega;
    (*acc).omega = (*acc).omega + omega;
}

fn nw(o: u32, k: u32, t: u32) -> f32 {
    return tnodes[o + k * 4u + t];
}

// Octupole expansion of node `o` (word offset) at r = x − c.
fn far_term(acc: ptr<function, Part>, o: u32, r: vec3<f32>) {
    let r2 = dot(r, r);
    let ir = inverseSqrt(r2);
    let ir2 = ir * ir;
    let ir3 = ir * ir2;
    let ir5 = ir3 * ir2;
    let ir7 = ir5 * ir2;
    let ir9 = ir7 * ir2;
    var phi = vec4<f32>(0.0);
    var grad = array<vec3<f32>, 4>();
    for(var t = 0u; t < 4u; t = t + 1u) {
        let q0 = nw(o, 2u, t);
        let q1 = vec3<f32>(nw(o, 3u, t), nw(o, 4u, t), nw(o, 5u, t));
        // Quadrupole (xx, yy, zz, xy, xz, yz) at words 6..11.
        let qxx = nw(o, 6u, t);
        let qyy = nw(o, 7u, t);
        let qzz = nw(o, 8u, t);
        let qxy = nw(o, 9u, t);
        let qxz = nw(o, 10u, t);
        let qyz = nw(o, 11u, t);
        let qr = vec3<f32>(
                qxx * r.x + qxy * r.y + qxz * r.z,
                qxy * r.x + qyy * r.y + qyz * r.z,
                qxz * r.x + qyz * r.y + qzz * r.z,
            );
        let a2 = dot(r, qr);
        let tr = qxx + qyy + qzz;
        // Octupole (xxx, yyy, zzz, xxy, xxz, xyy, yyz, xzz, yzz, xyz) at words 12..21:
        // w = M(·, r, r), v_k = M_iik, a₃ = M(r, r, r).
        var m: array<f32, 10>;
        for(var k = 0u; k < 10u; k = k + 1u) {
            m[k] = nw(o, 12u + k, t);
        }
        let xx = r.x * r.x;
        let yy = r.y * r.y;
        let zz = r.z * r.z;
        let xy2 = 2.0 * r.x * r.y;
        let xz2 = 2.0 * r.x * r.z;
        let yz2 = 2.0 * r.y * r.z;
        let w = vec3<f32>(
                m[0] * xx + m[5] * yy + m[7] * zz + m[3] * xy2 + m[4] * xz2 + m[9] * yz2,
                m[3] * xx + m[1] * yy + m[8] * zz + m[5] * xy2 + m[9] * xz2 + m[6] * yz2,
                m[4] * xx + m[6] * yy + m[2] * zz + m[9] * xy2 + m[7] * xz2 + m[8] * yz2,
            );
        let v = vec3<f32>(m[0] + m[5] + m[7], m[3] + m[1] + m[8], m[4] + m[6] + m[2]);
        let a3 = dot(r, w);
        let vr = dot(v, r);
        let quad = 3.0 * a2 - r2 * tr;
        let oct = 5.0 * a3 - 3.0 * r2 * vr;
        phi[t] = q0 * ir + dot(r, q1) * ir3 + 0.5 * quad * ir5 + 0.5 * oct * ir7;
        grad[t] =
            -r * (q0 * ir3) + q1 * ir3 - r * (3.0 * dot(r, q1) * ir5) + (3.0 * qr - r * tr) * ir5
            - r * (2.5 * quad * ir7)
            - (r * (105.0 * a3 - 45.0 * r2 * vr) - w * (45.0 * r2) + v * (9.0 * r2 * r2))
            * (ir9 / 6.0);
    }
    let s = phi.xyz;
    (*acc).u = (*acc).u + 0.5 * (phi.w - dot(r, s));
    (*acc).g = (*acc).g - s;
    let h0 = -vec3<f32>(grad[0].x, grad[1].y, grad[2].z);
    (*acc).h0 = (*acc).h0 + h0;
    (*acc).h1 = (*acc).h1
        - 0.5 * vec3<f32>(grad[0].y + grad[1].x, grad[0].z + grad[2].x, grad[1].z + grad[2].y);
    (*acc).omega = (*acc).omega - (h0.x + h0.y + h0.z);
}

fn flush(acc: ptr<function, array<ds, 10>>, p: Part) {
    (*acc)[0] = ds_add((*acc)[0], ds_f(p.u));
    (*acc)[1] = ds_add((*acc)[1], ds_f(p.g.x));
    (*acc)[2] = ds_add((*acc)[2], ds_f(p.g.y));
    (*acc)[3] = ds_add((*acc)[3], ds_f(p.g.z));
    (*acc)[4] = ds_add((*acc)[4], ds_f(p.h0.x));
    (*acc)[5] = ds_add((*acc)[5], ds_f(p.h0.y));
    (*acc)[6] = ds_add((*acc)[6], ds_f(p.h0.z));
    (*acc)[7] = ds_add((*acc)[7], ds_f(p.h1.x));
    (*acc)[8] = ds_add((*acc)[8], ds_f(p.h1.y));
    (*acc)[9] = ds_add((*acc)[9], ds_f(p.h1.z));
}

@compute @workgroup_size(64)
fn rtfp_uniform(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pid = gid.x;
    let p = uparams;
    if pid >= p.point_count {
        return;
    }
    let x = upoints[pid].xyz;
    let own = own_face[pid];
    var acc: array<ds, 10>;
    for(var m = 0u; m < 10u; m = m + 1u) {
        acc[m] = ds_zero();
    }
    var part = part_zero();
    var omega_total = 0.0;
    var pending = 0u;
    var stack: array<u32, STACK>;
    var top = 1u;
    stack[0] = 0u;
    while top > 0u {
        top = top - 1u;
        let o = stack[top] * NODE_WORDS;
        let r = x - vec3<f32>(tnodes[o], tnodes[o + 1u], tnodes[o + 2u]);
        let count = bitcast<u32>(tnodes[o + 5u]);
        if length(r) * p.theta > tnodes[o + 3u] {
            far_term(&part, o, r);
        } else if count > 0u {
            let first = bitcast<u32>(tnodes[o + 4u]);
            for(var f = first; f < first + count; f = f + 1u) {
                face_term(&part, f, x, f == own);
            }
        } else {
            stack[top] = bitcast<u32>(tnodes[o + 6u]);
            stack[top + 1u] = bitcast<u32>(tnodes[o + 7u]);
            top = top + 2u;
            continue;
        }
        pending = pending + 1u;
        if pending == FLUSH {
            flush(&acc, part);
            omega_total = omega_total + part.omega;
            part = part_zero();
            pending = 0u;
        }
    }
    flush(&acc, part);
    omega_total = omega_total + part.omega;
    let out_base = pid * UNIFORM_STRIDE;
    for(var m = 0u; m < 10u; m = m + 1u) {
        uout[out_base + 2u * m] = acc[m].hi;
        uout[out_base + 2u * m + 1u] = acc[m].lo;
    }
    uout[out_base + 20u] = omega_total;
}
