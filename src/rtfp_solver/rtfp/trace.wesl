struct TraceRay {
    ox: f32,
    oy: f32,
    oz: f32,
    t_min: f32,
    dx: f32,
    dy: f32,
    dz: f32,
    expect: f32,
}

struct TraceParams {
    max_hits: u32,
    rel_tol: f32,
    abs_tol: f32,
}

@group(0) @binding(0) var<storage, read> rays: array<TraceRay>;
@group(0) @binding(1) var<storage, read_write> counts: array<u32>;
@group(0) @binding(2) var<storage, read_write> hit_t: array<f32>;
@group(0) @binding(3) var<storage, read> params: TraceParams;

@compute @workgroup_size(64)
fn trace(@builtin(global_invocation_id) gid: vec3<u32>) {
    let id = gid.x;
    if (id >= arrayLength(&rays)) {
        return;
    }
    let input = rays[id];
    let dir = vec3<f32>(input.dx, input.dy, input.dz);
    let origin = vec3<f32>(input.ox, input.oy, input.oz) + edge_shift(dir);
    var t_min = input.t_min;
    var n = 0u;
    loop {
        if (n >= params.max_hits) {
            break;
        }
        let t = bvh_closest(origin, dir, t_min);
        if (!(t >= t_min)) {
            break;
        }
        if (n == 0u && input.expect > 0.0 && abs(t - input.expect) > params.rel_tol * input.expect + params.abs_tol) {
            n = 0u;
            break;
        }
        hit_t[id * params.max_hits + n] = t;
        t_min = next_up(t);
        n = n + 1u;
    }
    counts[id] = n;
}
