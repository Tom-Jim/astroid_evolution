// Pass 4: segment_build.wgsl
// 依照 docs/software.md 第 12.1 节
// 根据 inside_at_zero 构造 [R-, R+] 实体交段

struct Globals {
    n_points: u32,
    n_dirs: u32,
    n_tris: u32,
    n_kernels: u32,
    t_min: f32,
    t_max: f32,
    g_si: f32,
    r0: f32,
    ell0: f32,
    bulk_density: f32,
    _pad: vec2<f32>,
}

@group(0) @binding(0) var<uniform> g: Globals;
@group(0) @binding(1) var<storage, read> hit_offsets: array<u32>;
@group(0) @binding(2) var<storage, read> hit_counts: array<u32>;
@group(0) @binding(3) var<storage, read> compressed_hits: array<f32>;
// 每个观测点的内外属性 (0 = 外部, 1 = 内部)
@group(0) @binding(4) var<storage, read> inside_flags: array<u32>;
// 每条射线的有效交段数量
@group(0) @binding(5) var<storage, read_write> segment_counts: array<u32>;
// 展开存储的交段: [ray_id * MAX_SEGMENTS + s] -> vec2<f32>(r_min, r_max)
@group(0) @binding(6) var<storage, read_write> segments: array<vec2<f32>>;
@group(0) @binding(7) var<storage, read_write> segment_overflow: array<u32>;

const MAX_SEGMENTS: u32 = 256u;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ray_id = gid.x;
    let total_rays = g.n_points * g.n_dirs;
    if (ray_id >= total_rays) {
        return;
    }

    let point_id = ray_id / g.n_dirs;
    let offset = hit_offsets[ray_id];
    let count = hit_counts[ray_id];
    let is_inside = inside_flags[point_id] != 0u;

    var num_segs = 0u;
    let base_out = ray_id * MAX_SEGMENTS;

    if (is_inside) {
        // 内部点: 首段从 0 开始至第一个穿出点
        if (count > 0u) {
            let r_exit = compressed_hits[offset];
            if (r_exit > 1e-6) {
                segments[base_out + num_segs] = vec2<f32>(0.0, r_exit);
                num_segs += 1u;
            }

            var i = 1u;
            while (i + 1u < count && num_segs < MAX_SEGMENTS) {
                let r_enter = compressed_hits[offset + i];
                let r_out = compressed_hits[offset + i + 1u];
                if (r_out > r_enter + 1e-6) {
                    segments[base_out + num_segs] = vec2<f32>(r_enter, r_out);
                    num_segs += 1u;
                }
                i += 2u;
            }
        }
    } else {
        // 外部点: 射线自外部射入并穿出
        var i = 0u;
        while (i + 1u < count && num_segs < MAX_SEGMENTS) {
            let r_enter = compressed_hits[offset + i];
            let r_out = compressed_hits[offset + i + 1u];
            if (r_out > r_enter + 1e-6) {
                segments[base_out + num_segs] = vec2<f32>(r_enter, r_out);
                num_segs += 1u;
            }
            i += 2u;
        }
    }

    segment_counts[ray_id] = num_segs;
    if ((count > 2u * MAX_SEGMENTS) || num_segs >= MAX_SEGMENTS) {
        segment_overflow[ray_id] = 1u;
    }
}
