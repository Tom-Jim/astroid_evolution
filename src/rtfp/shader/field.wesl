// RTFP ray part of the potential, gradient and finite-part Hessian (MS §3.1, §3.8, §3.9).
//
// One thread per point integrates every direction of the Lebedev rule over the
// density difference ρ(y) − ρ(x); `points[i].w` is ρ(x). The difference vanishes at
// R = 0, so neither the contact term nor the logarithm of §3.9 arises here; both
// are in the uniform part ρ(x) Φ_V (`uniform.wesl`). Output per point, double-single
// pairs of (U, ∇U, H_xx, H_yy, H_zz, H_xy, H_xz, H_yz) in the unit frame without G,
// then an overflow flag.

// #include "rtfp/shader/chord.wesl"

@group(0) @binding(0)
var<storage, read> points: array<vec4<f32>>;
@group(0) @binding(1)
var<storage, read> rays: array<Ray>;
@group(0) @binding(2)
var<storage, read> kerns: array<Kern>;
@group(0) @binding(3)
var<storage, read> gl_nodes: array<Node>;
@group(0) @binding(4)
var<uniform> params: Chords;
@group(0) @binding(5)
var<storage, read_write> out: array<f32>;
@group(0) @binding(6)
var<storage, read> seg_counts: array<u32>;
@group(0) @binding(7)
var<storage, read> seg_pairs: array<f32>;

const FIELD_STRIDE: u32 = 21u;
const SEG_CAP: u32 = 8u;
const SEG_OVF: u32 = 0x40000000u;

@compute @workgroup_size(64)
fn rtfp_field(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pid = gid.x;
    let p = params;
    if pid >= p.item_count {
        return;
    }
    let x = points[pid].xyz;
    let bulk = p.bulk - points[pid].w;
    let want_h = p.want_h != 0u;
    var part: array<ds, 10>;
    for(var m = 0u; m < 10u; m = m + 1u) {
        part[m] = ds_zero();
    }
    var overflow = false;
    var dirty = 0u;
    var hits: array<vec2<f32>, 64>;
    var hits_p: array<f32, 64>;
    let seg_row = (p.seg_base + pid) * p.ray_count;
    for(var i = 0u; i < p.ray_count; i = i + 1u) {
        let ray = rays[i];
        let dir = ray.d;
        var acc = Radial(ds_zero(), ds_zero(), ds_zero());
        // The segment table 𝒯 (MS §3.15.1): intervals built once per geometry
        // are reused across every evaluation that binds them; a table row that
        // overflowed its store is retraced here, and a dirty row holds the
        // parity intervals the builder already wrote.
        var sc = 0u;
        if p.use_segs != 0u {
            sc = seg_counts[seg_row + i];
        }
        if p.use_segs != 0u && (sc & SEG_OVF) == 0u {
            let cnt = sc & 0x3fffffffu;
            for(var k = 0u; k < cnt; k = k + 1u) {
                let r0 = seg_pairs[(seg_row + i) * 2u * SEG_CAP + 2u * k];
                let r1 = seg_pairs[(seg_row + i) * 2u * SEG_CAP + 2u * k + 1u];
                integrate_segment(&acc, r0, r1, x, -dir, bulk, p);
            }
        } else {
        let n = collect_hits_o(x, dir, p, &hits);
        if n > RT_MAX_HITS {
            overflow = true;
            break;
        }
        // Occupied intervals from the orientation rule of MS §3.6: a crossing
        // with s > 0 leaves the solid and closes the open interval; s < 0
        // enters and opens one. A first crossing with s > 0 means the ray
        // started inside. Two same-sign crossings in a row, or an interval
        // left open, mean a lost or doubled report: the ray is dirty and falls
        // back to the parity construction, which stays correct as long as the
        // mesh is watertight (§3.6 step 4).
        var inside = n > 0u && hits[0].y > 0.5;
        var open_r = select(-1.0, 0.0, inside);
        var anomalies = 0u;
        for(var h = 0u; h < n; h = h + 1u) {
            let t = hits[h].x;
            let s = hits[h].y;
            if s > 0.5 {
                if inside {
                    integrate_segment(&acc, open_r, t, x, -dir, bulk, p);
                    inside = false;
                    open_r = -1.0;
                } else {
                    anomalies = anomalies + 1u;
                }
            } else if s < -0.5 {
                if !inside {
                    open_r = t;
                    inside = true;
                } else {
                    anomalies = anomalies + 1u;
                }
            }
        }
        if inside && n > 0u {
            anomalies = anomalies + 1u;
        }
        if anomalies > 0u {
            dirty = dirty + 1u;
            acc = Radial(ds_zero(), ds_zero(), ds_zero());
            let m = collect_hits(x, dir, p, &hits_p);
            if m > RT_MAX_HITS {
                overflow = true;
                break;
            }
            // An odd number of crossings: the ray starts inside (§3.6).
            var open_p = select(-1.0, 0.0, (m & 1u) != 0u);
            for(var h = 0u; h < m; h = h + 1u) {
                let t = hits_p[h];
                if open_p >= 0.0 {
                    integrate_segment(&acc, open_p, t, x, -dir, bulk, p);
                    open_p = -1.0;
                } else {
                    open_p = t;
                }
            }
        }
        }
        let w = ds_f(ray.weight);
        let wg = ds_mul(w, acc.g);
        part[0] = ds_add(part[0], ds_mul(w, acc.u));
        part[1] = ds_add(part[1], ds_mul(ds_f(dir.x), wg));
        part[2] = ds_add(part[2], ds_mul(ds_f(dir.y), wg));
        part[3] = ds_add(part[3], ds_mul(ds_f(dir.z), wg));
        if want_h {
            let wh = ds_mul(w, acc.h);
            part[4] = ds_add(part[4], ds_mul(ds_f(fma(3.0 * dir.x, dir.x, -1.0)), wh));
            part[5] = ds_add(part[5], ds_mul(ds_f(fma(3.0 * dir.y, dir.y, -1.0)), wh));
            part[6] = ds_add(part[6], ds_mul(ds_f(fma(3.0 * dir.z, dir.z, -1.0)), wh));
            part[7] = ds_add(part[7], ds_mul(ds_f(3.0 * dir.x * dir.y), wh));
            part[8] = ds_add(part[8], ds_mul(ds_f(3.0 * dir.x * dir.z), wh));
            part[9] = ds_add(part[9], ds_mul(ds_f(3.0 * dir.y * dir.z), wh));
        }
    }
    let base = pid * FIELD_STRIDE;
    for(var m = 0u; m < 10u; m = m + 1u) {
        out[base + 2u * m] = part[m].hi;
        out[base + 2u * m + 1u] = part[m].lo;
    }
    // Flag word: bit 0 is the overflow error, the upper bits count dirty rays
    // that fell back to the parity construction (MS §3.6 step 4).
    out[base + 20u] = f32(2u * dirty + select(0u, 1u, overflow));
}
