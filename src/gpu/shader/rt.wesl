// Ray casting against the body surface, shared by every program that traces it.
//
// `rt_hw` (desktop): a hardware ray query against the acceleration structure at
// binding 14 (Vulkan ray query, Metal `intersection_query`). The geometry is not
// opaque, so one query visits every crossing as a candidate.
// `!rt_hw` (browser, or a desktop GPU without ray tracing): a stack walk of the
// software BVH at bindings 14 and 15 with the watertight triangle test of Woop,
// Benthin and Wald (2013).
//
// `rt_closest` returns the nearest crossing with t ≥ t_min along origin + t dir,
// or a negative value. `rt_all` writes every crossing with t ≥ t_min, unsorted,
// and returns their number (more than 64 means the buffer overflowed).
// `rt_all_signed` writes (t, s) pairs instead, where s = sign(dir·n̂) of the
// crossed face: −1 enters the solid, +1 leaves it, 0 marks a degenerate
// (exactly grazing) report (MS §3.6, the orientation rule).

fn isfinite_f(x: f32) -> bool {
    return (bitcast<u32>(x) & 0x7fffffffu) < 0x7f800000u;
}

// Smallest float above `t`: the next `t_min` when stepping through every hit.
fn next_up(t: f32) -> f32 {
    if !isfinite_f(t) {
        return t;
    }
    if t == 0.0 {
        return bitcast<f32>(1u);
    }
    let bits = bitcast<u32>(t);
    if t > 0.0 {
        return bitcast<f32>(bits + 1u);
    }
    return bitcast<f32>(bits - 1u);
}

@if(rt_hw)
@group(0) @binding(14)
var rt_accel: acceleration_structure;

@if(rt_hw)
fn rt_closest(origin: vec3<f32>, dir: vec3<f32>, t_min: f32) -> f32 {
    var rq: ray_query;
    rayQueryInitialize(
        &rq,
        rt_accel,
        RayDesc(RAY_FLAG_FORCE_OPAQUE, 0xffu, t_min, 1.0e20, origin, dir),
    );
    while rayQueryProceed(&rq) {}
    let hit = rayQueryGetCommittedIntersection(&rq);
    if hit.kind == RAY_QUERY_INTERSECTION_TRIANGLE {
        return hit.t;
    }
    return -1.0;
}

// Metal: the candidate walk of an `intersection_query` does not terminate
// reliably on a wgpu-built instance structure, so crossings are stepped through
// with opaque nearest-hit queries from just past the previous one.
@if(rt_hw && metal)
fn rt_all(
    origin: vec3<f32>,
    dir: vec3<f32>,
    t_min: f32,
    out: ptr<function, array<f32, 64>>,
) -> u32 {
    var n = 0u;
    var t0 = t_min;
    loop {
        let t = rt_closest(origin, dir, t0);
        if !(t >= t0) {
            break;
        }
        if n < 64u {
            (*out)[n] = t;
        }
        n = n + 1u;
        if n > 64u {
            break;
        }
        t0 = next_up(t);
    }
    return min(n, 65u);
}

// The orientation sign of one hardware hit (MS §3.6): the front face of an
// outward-wound mesh is the side its outward normal points to, so a front hit
// has dir·n̂ < 0 (entering, s = −1) and a back hit s = +1 (leaving).
@if(rt_hw)
fn rt_sgn(front_face: bool) -> f32 {
    return select(1.0, -1.0, front_face);
}

@if(rt_hw && metal)
fn rt_all_signed(
    origin: vec3<f32>,
    dir: vec3<f32>,
    t_min: f32,
    out: ptr<function, array<vec2<f32>, 64>>,
) -> u32 {
    var n = 0u;
    var t0 = t_min;
    // The query object lives outside the stepping loop: Metal's
    // `intersection_query` cannot be assigned, only reset in place.
    var rq: ray_query;
    loop {
        rayQueryInitialize(
            &rq,
            rt_accel,
            RayDesc(RAY_FLAG_FORCE_OPAQUE, 0xffu, t0, 1.0e20, origin, dir),
        );
        while rayQueryProceed(&rq) {}
        let hit = rayQueryGetCommittedIntersection(&rq);
        if hit.kind != RAY_QUERY_INTERSECTION_TRIANGLE {
            break;
        }
        if n < 64u {
            (*out)[n] = vec2<f32>(hit.t, rt_sgn(hit.front_face));
        }
        n = n + 1u;
        if n > 64u {
            break;
        }
        t0 = next_up(hit.t);
    }
    return min(n, 65u);
}

@if(rt_hw && !metal)
fn rt_all(
    origin: vec3<f32>,
    dir: vec3<f32>,
    t_min: f32,
    out: ptr<function, array<f32, 64>>,
) -> u32 {
    var rq: ray_query;
    rayQueryInitialize(&rq, rt_accel, RayDesc(RAY_FLAG_NONE, 0xffu, t_min, 1.0e20, origin, dir));
    var n = 0u;
    while rayQueryProceed(&rq) {
        let c = rayQueryGetCandidateIntersection(&rq);
        if c.kind == RAY_QUERY_INTERSECTION_TRIANGLE {
            if n < 64u {
                (*out)[n] = c.t;
            }
            n = n + 1u;
        }
    }
    return min(n, 65u);
}

@if(rt_hw && !metal)
fn rt_all_signed(
    origin: vec3<f32>,
    dir: vec3<f32>,
    t_min: f32,
    out: ptr<function, array<vec2<f32>, 64>>,
) -> u32 {
    var rq: ray_query;
    rayQueryInitialize(&rq, rt_accel, RayDesc(RAY_FLAG_NONE, 0xffu, t_min, 1.0e20, origin, dir));
    var n = 0u;
    while rayQueryProceed(&rq) {
        let c = rayQueryGetCandidateIntersection(&rq);
        if c.kind == RAY_QUERY_INTERSECTION_TRIANGLE {
            if n < 64u {
                (*out)[n] = vec2<f32>(c.t, rt_sgn(c.front_face));
            }
            n = n + 1u;
        }
    }
    return min(n, 65u);
}

// Software BVH: a binary tree built on the CPU (PLOC, `obvhs`) whose node stores the
// boxes of both children, so one 64-byte load decides both. A child index with
// the top bit set is a triangle (`index & RT_INDEX`), otherwise a node. Triangles
// are stored in leaf order, 9 floats each. An empty child has an inverted box.
struct RtNode {
    a_lo: vec3<f32>,
    a: u32,
    a_hi: vec3<f32>,
    b: u32,
    b_lo: vec3<f32>,
    pad0: u32,
    b_hi: vec3<f32>,
    pad1: u32,
}

const RT_TRI: u32 = 0x80000000u;
const RT_INDEX: u32 = 0x7fffffffu;
const RT_STACK: u32 = 96u;
const RT_MISS: f32 = 3.0e38;

@if(!rt_hw)
@group(0) @binding(14)
var<storage, read> rt_bvh_nodes: array<RtNode>;
@if(!rt_hw)
@group(0) @binding(15)
var<storage, read> rt_bvh_tris: array<f32>;

// Per-ray constants of the watertight test (Woop, Benthin and Wald 2013).
struct RtRay {
    o: vec3<f32>,
    inv: vec3<f32>,
    d: vec3<f32>,
    kx: u32,
    ky: u32,
    kz: u32,
    sx: f32,
    sy: f32,
    sz: f32,
}

fn rt_ray(origin: vec3<f32>, dir: vec3<f32>) -> RtRay {
    let ad = abs(dir);
    var kz = 0u;
    if ad.y > ad.x && ad.y >= ad.z {
        kz = 1u;
    } else if ad.z > ad.x && ad.z > ad.y {
        kz = 2u;
    }
    var kx = (kz + 1u) % 3u;
    var ky = (kx + 1u) % 3u;
    // Keep the winding: swap when the dominant component is negative.
    if dir[kz] < 0.0 {
        let t = kx;
        kx = ky;
        ky = t;
    }
    let inv = vec3<f32>(
            select(1.0 / dir.x, 1.0e30, dir.x == 0.0),
            select(1.0 / dir.y, 1.0e30, dir.y == 0.0),
            select(1.0 / dir.z, 1.0e30, dir.z == 0.0),
        );
    let dz = dir[kz];
    return RtRay(origin, inv, dir, kx, ky, kz, dir[kx] / dz, dir[ky] / dz, 1.0 / dz);
}

// Slab test; the entry distance, or RT_MISS.
fn rt_box(r: RtRay, t_min: f32, t_max: f32, lo: vec3<f32>, hi: vec3<f32>) -> f32 {
    let t1 = (lo - r.o) * r.inv;
    let t2 = (hi - r.o) * r.inv;
    let tn = max(max(min(t1.x, t2.x), min(t1.y, t2.y)), max(min(t1.z, t2.z), t_min));
    let tf = min(min(max(t1.x, t2.x), max(t1.y, t2.y)), min(max(t1.z, t2.z), t_max));
    return select(RT_MISS, tn, tn <= tf);
}

@if(!rt_hw)
fn rt_tri(r: RtRay, index: u32, t_min: f32, t_max: f32) -> vec2<f32> {
    let base = index * 9u;
    let a = vec3<f32>(rt_bvh_tris[base], rt_bvh_tris[base + 1u], rt_bvh_tris[base + 2u]) - r.o;
    let b = vec3<f32>(rt_bvh_tris[base + 3u], rt_bvh_tris[base + 4u], rt_bvh_tris[base + 5u]) - r.o;
    let c = vec3<f32>(rt_bvh_tris[base + 6u], rt_bvh_tris[base + 7u], rt_bvh_tris[base + 8u]) - r.o;
    let ax = a[r.kx] - r.sx * a[r.kz];
    let ay = a[r.ky] - r.sy * a[r.kz];
    let bx = b[r.kx] - r.sx * b[r.kz];
    let by = b[r.ky] - r.sy * b[r.kz];
    let cx = c[r.kx] - r.sx * c[r.kz];
    let cy = c[r.ky] - r.sy * c[r.kz];
    let u = cx * by - cy * bx;
    let v = ax * cy - ay * cx;
    let w = bx * ay - by * ax;
    // Both orientations count: the surface is crossed from either side.
    if (u < 0.0 || v < 0.0 || w < 0.0) && (u > 0.0 || v > 0.0 || w > 0.0) {
        return vec2<f32>(-1.0, 0.0);
    }
    let det = u + v + w;
    if det == 0.0 {
        return vec2<f32>(-1.0, 0.0);
    }
    let t = (u * r.sz * a[r.kz] + v * r.sz * b[r.kz] + w * r.sz * c[r.kz]) / det;
    if t < t_min || t > t_max {
        return vec2<f32>(-1.0, 0.0);
    }
    // Orientation sign for MS §3.6: a, b, c hold the translated vertices, so
    // (b − a)×(c − a) is the face normal in input winding and its scalar
    // product with the ray direction is the crossing sign.
    let n = cross(b - a, c - a);
    let s = sign(dot(n, r.d));
    return vec2<f32>(t, s);
}

@if(!rt_hw)
fn rt_closest(origin: vec3<f32>, dir: vec3<f32>, t_min: f32) -> f32 {
    let r = rt_ray(origin, dir);
    // Near child first, far child pushed with its entry distance, so a popped
    // subtree behind the current hit is skipped. The stack holds one entry per level;
    // the host rejects a tree deeper than RT_STACK before upload.
    var stack: array<u32, 96>;
    var stack_t: array<f32, 96>;
    var sp = 0u;
    var best = -1.0;
    var limit = 1.0e20;
    var node = 0u;
    loop {
        let n = rt_bvh_nodes[node];
        var ta = rt_box(r, t_min, limit, n.a_lo, n.a_hi);
        var tb = rt_box(r, t_min, limit, n.b_lo, n.b_hi);
        if ta < RT_MISS && (n.a & RT_TRI) != 0u {
            let t = rt_tri(r, n.a & RT_INDEX, t_min, limit).x;
            if t >= 0.0 {
                best = t;
                limit = t;
            }
            ta = RT_MISS;
        }
        if tb < RT_MISS && (n.b & RT_TRI) != 0u {
            let t = rt_tri(r, n.b & RT_INDEX, t_min, limit).x;
            if t >= 0.0 {
                best = t;
                limit = t;
            }
            tb = RT_MISS;
        }
        let ha = ta <= limit;
        let hb = tb <= limit;
        if ha && hb {
            let a_first = ta <= tb;
            node = select(n.b, n.a, a_first);
            if sp < RT_STACK {
                stack[sp] = select(n.a, n.b, a_first);
                stack_t[sp] = max(ta, tb);
                sp = sp + 1u;
            }
        } else if ha {
            node = n.a;
        } else if hb {
            node = n.b;
        } else {
            // Pop, skipping subtrees entered beyond a later hit.
            var found = false;
            loop {
                if sp == 0u {
                    break;
                }
                sp = sp - 1u;
                if stack_t[sp] <= limit {
                    node = stack[sp];
                    found = true;
                    break;
                }
            }
            if !found {
                return best;
            }
        }
    }
    return best;
}

@if(!rt_hw)
fn rt_all(
    origin: vec3<f32>,
    dir: vec3<f32>,
    t_min: f32,
    out: ptr<function, array<f32, 64>>,
) -> u32 {
    let r = rt_ray(origin, dir);
    var stack: array<u32, 96>;
    var sp = 0u;
    var n = 0u;
    var node = 0u;
    loop {
        let nd = rt_bvh_nodes[node];
        var next = RT_TRI;
        for(var side = 0u; side < 2u; side = side + 1u) {
            let child = select(nd.b, nd.a, side == 0u);
            let lo = select(nd.b_lo, nd.a_lo, side == 0u);
            let hi = select(nd.b_hi, nd.a_hi, side == 0u);
            if rt_box(r, t_min, 1.0e20, lo, hi) >= RT_MISS {
                continue;
            }
            if (child & RT_TRI) != 0u {
                let t = rt_tri(r, child & RT_INDEX, t_min, 1.0e20).x;
                if t >= 0.0 {
                    if n < 64u {
                        (*out)[n] = t;
                    }
                    n = n + 1u;
                }
            } else if next == RT_TRI {
                next = child;
            } else if sp < RT_STACK {
                stack[sp] = child;
                sp = sp + 1u;
            }
        }
        if next != RT_TRI {
            node = next;
        } else if sp > 0u {
            sp = sp - 1u;
            node = stack[sp];
        } else {
            break;
        }
    }
    return min(n, 65u);
}

@if(!rt_hw)
fn rt_all_signed(
    origin: vec3<f32>,
    dir: vec3<f32>,
    t_min: f32,
    out: ptr<function, array<vec2<f32>, 64>>,
) -> u32 {
    let r = rt_ray(origin, dir);
    var stack: array<u32, 96>;
    var sp = 0u;
    var n = 0u;
    var node = 0u;
    loop {
        let nd = rt_bvh_nodes[node];
        var next = RT_TRI;
        for(var side = 0u; side < 2u; side = side + 1u) {
            let child = select(nd.b, nd.a, side == 0u);
            let lo = select(nd.b_lo, nd.a_lo, side == 0u);
            let hi = select(nd.b_hi, nd.a_hi, side == 0u);
            if rt_box(r, t_min, 1.0e20, lo, hi) >= RT_MISS {
                continue;
            }
            if (child & RT_TRI) != 0u {
                let hit = rt_tri(r, child & RT_INDEX, t_min, 1.0e20);
                if hit.x >= 0.0 {
                    if n < 64u {
                        (*out)[n] = hit;
                    }
                    n = n + 1u;
                }
            } else if next == RT_TRI {
                next = child;
            } else if sp < RT_STACK {
                stack[sp] = child;
                sp = sp + 1u;
            }
        }
        if next != RT_TRI {
            node = next;
        } else if sp > 0u {
            sp = sp - 1u;
            node = stack[sp];
        } else {
            break;
        }
    }
    return min(n, 65u);
}
