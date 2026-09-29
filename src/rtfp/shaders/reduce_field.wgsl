// Pass 6: reduce_field.wgsl
// 依照 docs/software.md 第 12.1 节
// 分层归约得到 U, g, H6, Tr(H)
// 避免全局原子浮点加法，采用 workgroup shared memory 并行归约

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
// per-ray 输入 (每个 ray 3 个 vec4)
@group(0) @binding(1) var<storage, read> ray_fields: array<vec4<f32>>;
// 归约输出: 每个 point 4 个 vec4
// [4*p + 0] = (U, g.x, g.y, g.z)
// [4*p + 1] = (H.xx, H.yy, H.zz, H.xy)
// [4*p + 2] = (H.xz, H.yz, trace, 0.0)
// [4*p + 3] = pad
@group(0) @binding(2) var<storage, read_write> output_fields: array<vec4<f32>>;

const WG_SIZE: u32 = 64u;

var<workgroup> s_u: array<f32, 64>;
var<workgroup> s_gx: array<f32, 64>;
var<workgroup> s_gy: array<f32, 64>;
var<workgroup> s_gz: array<f32, 64>;
var<workgroup> s_hxx: array<f32, 64>;
var<workgroup> s_hyy: array<f32, 64>;
var<workgroup> s_hzz: array<f32, 64>;
var<workgroup> s_hxy: array<f32, 64>;
var<workgroup> s_hxz: array<f32, 64>;
var<workgroup> s_hyz: array<f32, 64>;

@compute @workgroup_size(64, 1, 1)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let point_id = wid.x;
    if (point_id >= g.n_points) {
        return;
    }

    let thread_id = lid.x;
    let base_ray = point_id * g.n_dirs;

    var sum_u = 0.0;
    var sum_gx = 0.0;
    var sum_gy = 0.0;
    var sum_gz = 0.0;
    var sum_hxx = 0.0;
    var sum_hyy = 0.0;
    var sum_hzz = 0.0;
    var sum_hxy = 0.0;
    var sum_hxz = 0.0;
    var sum_hyz = 0.0;

    // 跨步累加所有方向
    var dir_idx = thread_id;
    while (dir_idx < g.n_dirs) {
        let ray_idx = base_ray + dir_idx;
        let r_base = 3u * ray_idx;
        let v0 = ray_fields[r_base + 0u];
        let v1 = ray_fields[r_base + 1u];
        let v2 = ray_fields[r_base + 2u];

        sum_u += v0.x;
        sum_gx += v0.y;
        sum_gy += v0.z;
        sum_gz += v0.w;

        sum_hxx += v1.x;
        sum_hyy += v1.y;
        sum_hzz += v1.z;
        sum_hxy += v1.w;

        sum_hxz += v2.x;
        sum_hyz += v2.y;

        dir_idx += WG_SIZE;
    }

    s_u[thread_id] = sum_u;
    s_gx[thread_id] = sum_gx;
    s_gy[thread_id] = sum_gy;
    s_gz[thread_id] = sum_gz;
    s_hxx[thread_id] = sum_hxx;
    s_hyy[thread_id] = sum_hyy;
    s_hzz[thread_id] = sum_hzz;
    s_hxy[thread_id] = sum_hxy;
    s_hxz[thread_id] = sum_hxz;
    s_hyz[thread_id] = sum_hyz;

    workgroupBarrier();

    // 树状并行归约
    for (var stride = WG_SIZE / 2u; stride > 0u; stride /= 2u) {
        if (thread_id < stride) {
            s_u[thread_id] += s_u[thread_id + stride];
            s_gx[thread_id] += s_gx[thread_id + stride];
            s_gy[thread_id] += s_gy[thread_id + stride];
            s_gz[thread_id] += s_gz[thread_id + stride];
            s_hxx[thread_id] += s_hxx[thread_id + stride];
            s_hyy[thread_id] += s_hyy[thread_id + stride];
            s_hzz[thread_id] += s_hzz[thread_id + stride];
            s_hxy[thread_id] += s_hxy[thread_id + stride];
            s_hxz[thread_id] += s_hxz[thread_id + stride];
            s_hyz[thread_id] += s_hyz[thread_id + stride];
        }
        workgroupBarrier();
    }

    if (thread_id == 0u) {
        let final_u = s_u[0];
        let final_g = vec3<f32>(s_gx[0], s_gy[0], s_gz[0]);
        let final_hxx = s_hxx[0];
        let final_hyy = s_hyy[0];
        let final_hzz = s_hzz[0];
        let final_hxy = s_hxy[0];
        let final_hxz = s_hxz[0];
        let final_hyz = s_hyz[0];
        let trace = final_hxx + final_hyy + final_hzz;

        let out_base = 4u * point_id;
        output_fields[out_base + 0u] = vec4<f32>(final_u, final_g.x, final_g.y, final_g.z);
        output_fields[out_base + 1u] = vec4<f32>(final_hxx, final_hyy, final_hzz, final_hxy);
        output_fields[out_base + 2u] = vec4<f32>(final_hxz, final_hyz, trace, 0.0);
        output_fields[out_base + 3u] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
}
