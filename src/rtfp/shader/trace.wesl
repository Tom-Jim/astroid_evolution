// Every crossing of a batch of rays with the body surface, nearest first.
//
// Ray record: origin, t_min, direction, expect. With expect > 0 the first crossing
// must lie within rel_tol·expect + abs_tol of it, else the ray reports nothing.
// counts[i] is the number of crossings (more than max_hits marks an overflow);
// hit_t holds the first max_hits of them.

// #include "gpu/shader/rt.wesl"

struct TraceRay {
    o: vec3<f32>,
    t_min: f32,
    d: vec3<f32>,
    expect: f32,
}

struct TraceParams {
    ray_count: u32,
    max_hits: u32,
    rel_tol: f32,
    abs_tol: f32,
}

@group(0) @binding(0)
var<storage, read> trace_rays: array<TraceRay>;
@group(0) @binding(1)
var<uniform> trace_params: TraceParams;
@group(0) @binding(2)
var<storage, read_write> counts: array<u32>;
@group(0) @binding(3)
var<storage, read_write> hit_t: array<f32>;

@compute @workgroup_size(64)
fn trace(@builtin(global_invocation_id) gid: vec3<u32>) {
    let id = gid.x;
    let p = trace_params;
    if id >= p.ray_count {
        return;
    }
    let r = trace_rays[id];
    var raw: array<f32, 64>;
    var n = rt_all(r.o, r.d, r.t_min, &raw);
    if n <= 64u {
        for(var i = 1u; i < n; i = i + 1u) {
            let v = raw[i];
            var j = i;
            while j > 0u && raw[j - 1u] > v {
                raw[j] = raw[j - 1u];
                j = j - 1u;
            }
            raw[j] = v;
        }
        if n > 0u && r.expect > 0.0 && abs(raw[0] - r.expect) > p.rel_tol * r.expect + p.abs_tol {
            n = 0u;
        }
    }
    let keep = min(n, p.max_hits);
    for(var i = 0u; i < keep; i = i + 1u) {
        hit_t[id * p.max_hits + i] = raw[i];
    }
    counts[id] = n;
}
