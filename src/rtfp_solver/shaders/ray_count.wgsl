// P1. Prepend common/tri.wgsl. Counts hits with the same predicate P3 writes.

struct Globals {
    n_points: u32,
    n_dirs: u32,
    t_min: f32,
    t_max: f32,
}

@group(0) @binding(0) var<uniform> g: Globals;
@group(0) @binding(1) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> triangles: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read> bvh_min: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> bvh_max: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> bvh_links: array<vec4<u32>>;
@group(0) @binding(6) var<storage, read> points: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read> dirs: array<vec4<f32>>;
@group(0) @binding(8) var<storage, read_write> hit_counts: array<u32>;
@group(0) @binding(9) var<storage, read_write> ray_errors: array<u32>;

const STACK_SIZE: u32 = 32u;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ray_id = gid.x;
    let total_rays = g.n_points * g.n_dirs;
    if ray_id >= total_rays {
        return;
    }
    let orig = points[ray_id / g.n_dirs].xyz;
    let ray_dir = -dirs[ray_id % g.n_dirs].xyz;
    var stack: array<u32, 32>;
    var stack_ptr = 1u;
    stack[0] = 0u;
    var count = 0u;
    while stack_ptr > 0u {
        stack_ptr = stack_ptr - 1u;
        let node_idx = stack[stack_ptr];
        if !ray_slab_hit(orig, ray_dir, bvh_min[node_idx].xyz, bvh_max[node_idx].xyz) {
            continue;
        }
        let link = bvh_links[node_idx];
        let leaf_count = link.w;
        if leaf_count > 0u {
            let start = link.z;
            for (var i = 0u; i < leaf_count; i = i + 1u) {
                let tri = triangles[start + i];
                let t = ray_triangle_hit(
                    orig,
                    ray_dir,
                    positions[tri.x].xyz,
                    positions[tri.y].xyz,
                    positions[tri.z].xyz,
                    g.t_min,
                    g.t_max,
                );
                if t > 0.0 {
                    count = count + 1u;
                }
            }
        } else if stack_ptr + 2u <= STACK_SIZE {
            stack[stack_ptr] = link.x;
            stack_ptr = stack_ptr + 1u;
            stack[stack_ptr] = link.y;
            stack_ptr = stack_ptr + 1u;
        } else {
            ray_errors[ray_id] = 1u;
        }
    }
    if count > 256u {
        ray_errors[ray_id] = 2u;
    }
    hit_counts[ray_id] = count;
}
