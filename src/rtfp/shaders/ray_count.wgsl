// Pass 1: ray_count.wgsl
// 依照 docs/software.md 第 12.1 节
// 一个线程处理一条射线 (point_id, dir_id)
// 遍历扁平 BVH，输出 hit_counts[ray_id]

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
// 顶点 positions: vec4<f32>(x, y, z, 0.0)
@group(0) @binding(1) var<storage, read> positions: array<vec4<f32>>;
// 面片索引: vec4<u32>(i0, i1, i2, 0)
@group(0) @binding(2) var<storage, read> triangles: array<vec4<u32>>;
// BVH 包围盒: 2 行每个节点: bounds_min, bounds_max
@group(0) @binding(3) var<storage, read> bvh_bounds: array<vec4<f32>>;
// BVH 拓扑与叶片数据: vec4<u32>(left, right, start, count)
@group(0) @binding(4) var<storage, read> bvh_links: array<vec4<u32>>;
// 观测点: vec4<f32>(x, y, z, 0.0)
@group(0) @binding(5) var<storage, read> points: array<vec4<f32>>;
// 288 求积方向: vec4<f32>(u_x, u_y, u_z, omega)
@group(0) @binding(6) var<storage, read> dirs: array<vec4<f32>>;
// 输出命中计数: hit_counts[ray_id]
@group(0) @binding(7) var<storage, read_write> hit_counts: array<u32>;

const STACK_SIZE: u32 = 64u;

// Möller-Trumbore 射线-三角形求交测试
fn ray_triangle_hit(orig: vec3<f32>, dir: vec3<f32>, v0: vec3<f32>, v1: vec3<f32>, v2: vec3<f32>) -> f32 {
    let e1 = v1 - v0;
    let e2 = v2 - v0;
    let pvec = cross(dir, e2);
    let det = dot(e1, pvec);
    if (abs(det) < 1e-12) {
        return -1.0;
    }
    let inv_det = 1.0 / det;
    let tvec = orig - v0;
    let u = dot(tvec, pvec) * inv_det;
    if (u < 0.0 || u > 1.0) {
        return -1.0;
    }
    let qvec = cross(tvec, e1);
    let v = dot(dir, qvec) * inv_det;
    if (v < 0.0 || u + v > 1.0) {
        return -1.0;
    }
    let t = dot(e2, qvec) * inv_det;
    if (t > g.t_min && t < g.t_max) {
        return t;
    }
    return -1.0;
}

// 射线与 AABB slab 求交
fn ray_aabb_test(node_idx: u32, orig: vec3<f32>, inv_dir: vec3<f32>) -> bool {
    let bmin = bvh_bounds[2u * node_idx].xyz;
    let bmax = bvh_bounds[2u * node_idx + 1u].xyz;

    let tx1 = (bmin.x - orig.x) * inv_dir.x;
    let tx2 = (bmax.x - orig.x) * inv_dir.x;
    let tmin_x = min(tx1, tx2);
    let tmax_x = max(tx1, tx2);

    let ty1 = (bmin.y - orig.y) * inv_dir.y;
    let ty2 = (bmax.y - orig.y) * inv_dir.y;
    let tmin_y = min(ty1, ty2);
    let tmax_y = max(ty1, ty2);

    let tz1 = (bmin.z - orig.z) * inv_dir.z;
    let tz2 = (bmax.z - orig.z) * inv_dir.z;
    let tmin_z = min(tz1, tz2);
    let tmax_z = max(tz1, tz2);

    let t_enter = max(max(tmin_x, tmin_y), tmin_z);
    let t_exit = min(min(tmax_x, tmax_y), tmax_z);

    return t_exit >= 0.0 && t_enter <= t_exit;
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ray_id = gid.x;
    let total_rays = g.n_points * g.n_dirs;
    if (ray_id >= total_rays) {
        return;
    }

    let point_id = ray_id / g.n_dirs;
    let dir_id = ray_id % g.n_dirs;

    let orig = points[point_id].xyz;
    // 物理射线朝向源点方向 -u (y = x - R u)
    let u_dir = dirs[dir_id].xyz;
    let ray_dir = -u_dir;

    var inv_dir = vec3<f32>(1e12, 1e12, 1e12);
    if (abs(ray_dir.x) > 1e-12) { inv_dir.x = 1.0 / ray_dir.x; }
    if (abs(ray_dir.y) > 1e-12) { inv_dir.y = 1.0 / ray_dir.y; }
    if (abs(ray_dir.z) > 1e-12) { inv_dir.z = 1.0 / ray_dir.z; }

    var stack: array<u32, 64>;
    var stack_ptr = 0u;

    stack[0] = 0u; // 根节点
    stack_ptr = 1u;

    var count = 0u;

    while (stack_ptr > 0u) {
        stack_ptr -= 1u;
        let node_idx = stack[stack_ptr];

        if (!ray_aabb_test(node_idx, orig, inv_dir)) {
            continue;
        }

        let link = bvh_links[node_idx];
        let leaf_count = link.w;

        if (leaf_count > 0u) {
            let start = link.z;
            for (var i = 0u; i < leaf_count; i += 1u) {
                let tri = triangles[start + i];
                let v0 = positions[tri.x].xyz;
                let v1 = positions[tri.y].xyz;
                let v2 = positions[tri.z].xyz;
                let t = ray_triangle_hit(orig, ray_dir, v0, v1, v2);
                if (t > 0.0) {
                    count += 1u;
                }
            }
        } else {
            if (stack_ptr + 2u < STACK_SIZE) {
                stack[stack_ptr] = link.x;
                stack_ptr += 1u;
                stack[stack_ptr] = link.y;
                stack_ptr += 1u;
            }
        }
    }

    hit_counts[ray_id] = count;
}
