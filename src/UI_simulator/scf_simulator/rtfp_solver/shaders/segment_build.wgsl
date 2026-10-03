// P4. Interior [0, R1], [R2, R3], … ; exterior [R1, R2], …
// A boundary sample (inside_flags == 2) is not forced into either pairing.

struct Globals {
    n_points: u32,
    n_dirs: u32,
}

@group(0) @binding(0) var<uniform> g: Globals;
@group(0) @binding(1) var<storage, read> hit_offsets: array<u32>;
@group(0) @binding(2) var<storage, read> hit_counts: array<u32>;
@group(0) @binding(3) var<storage, read> compressed_hits: array<f32>;
@group(0) @binding(4) var<storage, read> inside_flags: array<u32>;
@group(0) @binding(5) var<storage, read_write> segment_counts: array<u32>;
@group(0) @binding(6) var<storage, read_write> segments: array<vec2<f32>>;
@group(0) @binding(7) var<storage, read_write> segment_overflow: array<u32>;

const MAX_SEGMENTS: u32 = 256u;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ray_id = gid.x;
    if ray_id >= g.n_points * g.n_dirs {
        return;
    }
    let point_id = ray_id / g.n_dirs;
    let flag = inside_flags[point_id];
    if flag == 2u {
        segment_counts[ray_id] = 0u;
        segment_overflow[ray_id] = 4u;
        return;
    }
    let offset = hit_offsets[ray_id];
    let raw_count = hit_counts[ray_id];
    if raw_count > 256u {
        segment_overflow[ray_id] = 2u;
        segment_counts[ray_id] = 0u;
        return;
    }
    var unique_hits: array<f32, 256>;
    var count = 0u;
    for (var j = 0u; j < raw_count; j = j + 1u) {
        let t = compressed_hits[offset + j];
        if count == 0u || abs(t - unique_hits[count - 1u]) > max(1e-6, 8e-7 * abs(t)) {
            unique_hits[count] = t;
            count = count + 1u;
        }
    }
    var num_segs = 0u;
    let base_out = ray_id * MAX_SEGMENTS;
    var start = 0u;
    if flag == 1u {
        if count > 0u && unique_hits[0] > 1e-6 {
            segments[base_out] = vec2<f32>(0.0, unique_hits[0]);
            num_segs = 1u;
        }
        start = 1u;
    }
    var i = start;
    while i + 1u < count && num_segs < MAX_SEGMENTS {
        let r_enter = unique_hits[i];
        let r_out = unique_hits[i + 1u];
        if r_out > r_enter + 1e-6 {
            segments[base_out + num_segs] = vec2<f32>(r_enter, r_out);
            num_segs = num_segs + 1u;
        }
        i = i + 2u;
    }
    segment_counts[ray_id] = num_segs;
    segment_overflow[ray_id] = select(0u, 1u, num_segs >= MAX_SEGMENTS);
}
