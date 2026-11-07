// The segment table 𝒯 of MS §3.15.1: for every (point m, direction q) pair the
// occupied intervals [R⁻, R⁺] of the ray, built once per geometry and reused
// across density updates and field evaluations.
//
// One thread per (point, ray) pair: id = m·ray_count + q. The intervals come
// from the orientation rule of §3.6 (`collect_hits_o`); a ray whose signed
// crossings do not alternate is rebuilt by the parity rule and marked dirty in
// the flag word. A ray needing more than SEG_CAP intervals marks overflow and
// is retraced inline by the field kernel.
//
// seg_counts[i] packs the interval count and flags:
//   bits 0..30  interval count (0 when the ray stays outside),
//   bit 30      overflow — the ray holds more segments than the table stores,
//   bit 31      dirty — the orientation rule saw an anomaly, parity was used.
// seg_pairs[i] holds SEG_CAP f32 pairs (R⁻, R⁺).

// #include "rtfp/shader/chord.wesl"

const SEG_CAP: u32 = 8u;
const SEG_OVF: u32 = 0x40000000u;
const SEG_DIRTY: u32 = 0x80000000u;

@group(0) @binding(0)
var<storage, read> seg_points: array<vec4<f32>>;
@group(0) @binding(1)
var<storage, read> seg_rays: array<Ray>;
@group(0) @binding(2)
var<uniform> seg_params: Chords;
@group(0) @binding(3)
var<storage, read_write> seg_counts: array<u32>;
@group(0) @binding(4)
var<storage, read_write> seg_pairs: array<f32>;

@compute @workgroup_size(64)
fn rtfp_segments(@builtin(global_invocation_id) gid: vec3<u32>) {
    let id = gid.x;
    let p = seg_params;
    if id >= p.item_count {
        return;
    }
    let pid = id / p.ray_count;
    let rid = id - pid * p.ray_count;
    let x = seg_points[pid].xyz;
    let dir = seg_rays[rid].d;

    var hits: array<vec2<f32>, 64>;
    let n = collect_hits_o(x, dir, p, &hits);
    if n > RT_MAX_HITS {
        seg_counts[id] = SEG_OVF | SEG_DIRTY;
        return;
    }

    var cnt = 0u;
    var anomalies = 0u;
    var inside = n > 0u && hits[0].y > 0.5;
    var open_r = select(-1.0, 0.0, inside);
    for(var h = 0u; h < n; h = h + 1u) {
        let t = hits[h].x;
        let s = hits[h].y;
        if s > 0.5 {
            if inside {
                if cnt < SEG_CAP {
                    seg_pairs[id * 2u * SEG_CAP + 2u * cnt] = open_r;
                    seg_pairs[id * 2u * SEG_CAP + 2u * cnt + 1u] = t;
                }
                cnt = cnt + 1u;
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

    if anomalies > 0u || cnt > SEG_CAP {
        // Rebuild the ray by the parity rule (§3.6 step 4): correct while the
        // mesh is watertight, and the dirty/overflow flags keep it visible.
        var raw: array<f32, 64>;
        let m = collect_hits(x, dir, p, &raw);
        cnt = 0u;
        if m <= RT_MAX_HITS {
            var open_p = select(-1.0, 0.0, (m & 1u) != 0u);
            for(var h = 0u; h < m; h = h + 1u) {
                let t = raw[h];
                if open_p >= 0.0 {
                    if cnt < SEG_CAP {
                        seg_pairs[id * 2u * SEG_CAP + 2u * cnt] = open_p;
                        seg_pairs[id * 2u * SEG_CAP + 2u * cnt + 1u] = t;
                    }
                    cnt = cnt + 1u;
                    open_p = -1.0;
                } else {
                    open_p = t;
                }
            }
        } else {
            cnt = SEG_CAP + 1u;
        }
        let flags = select(0u, SEG_DIRTY, anomalies > 0u) | select(0u, SEG_OVF, cnt > SEG_CAP);
        seg_counts[id] = min(cnt, 0x3fffffffu) | flags;
        return;
    }
    seg_counts[id] = cnt;
}
