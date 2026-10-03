// P7. Sum direction partials, then add the contact term on the GPU.
// point_state.x = ϑ, point_state.y = ρ(x).
// H ← H − (4πG/3) ϑ ρ I, so tr(H) = −4πG ϑ ρ with no CPU readback.

struct Globals {
    n_points: u32,
    n_dirs: u32,
    g_si: f32,
    _pad: f32,
}

@group(0) @binding(0) var<uniform> g: Globals;
@group(0) @binding(1) var<storage, read> ray_fields: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> point_state: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> output_fields: array<vec4<f32>>;

const WG_SIZE: u32 = 64u;
const PI: f32 = 3.141592653589793;

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
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let point_id = wid.x;
    let thread_id = lid.x;
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
    if point_id < g.n_points {
        var dir_idx = thread_id;
        while dir_idx < g.n_dirs {
            let r_base = 3u * (point_id * g.n_dirs + dir_idx);
            let v0 = ray_fields[r_base + 0u];
            let v1 = ray_fields[r_base + 1u];
            let v2 = ray_fields[r_base + 2u];
            sum_u = sum_u + v0.x;
            sum_gx = sum_gx + v0.y;
            sum_gy = sum_gy + v0.z;
            sum_gz = sum_gz + v0.w;
            sum_hxx = sum_hxx + v1.x;
            sum_hyy = sum_hyy + v1.y;
            sum_hzz = sum_hzz + v1.z;
            sum_hxy = sum_hxy + v1.w;
            sum_hxz = sum_hxz + v2.x;
            sum_hyz = sum_hyz + v2.y;
            dir_idx = dir_idx + WG_SIZE;
        }
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
    for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride / 2u) {
        if thread_id < stride {
            s_u[thread_id] = s_u[thread_id] + s_u[thread_id + stride];
            s_gx[thread_id] = s_gx[thread_id] + s_gx[thread_id + stride];
            s_gy[thread_id] = s_gy[thread_id] + s_gy[thread_id + stride];
            s_gz[thread_id] = s_gz[thread_id] + s_gz[thread_id + stride];
            s_hxx[thread_id] = s_hxx[thread_id] + s_hxx[thread_id + stride];
            s_hyy[thread_id] = s_hyy[thread_id] + s_hyy[thread_id + stride];
            s_hzz[thread_id] = s_hzz[thread_id] + s_hzz[thread_id + stride];
            s_hxy[thread_id] = s_hxy[thread_id] + s_hxy[thread_id + stride];
            s_hxz[thread_id] = s_hxz[thread_id] + s_hxz[thread_id + stride];
            s_hyz[thread_id] = s_hyz[thread_id] + s_hyz[thread_id + stride];
        }
        workgroupBarrier();
    }
    if thread_id == 0u && point_id < g.n_points {
        let state = point_state[point_id];
        let contact = -(4.0 * PI * g.g_si / 3.0) * state.x * state.y;
        let hxx = s_hxx[0] + contact;
        let hyy = s_hyy[0] + contact;
        let hzz = s_hzz[0] + contact;
        let out_base = 4u * point_id;
        output_fields[out_base + 0u] = vec4<f32>(s_u[0], s_gx[0], s_gy[0], s_gz[0]);
        output_fields[out_base + 1u] = vec4<f32>(hxx, hyy, hzz, s_hxy[0]);
        output_fields[out_base + 2u] = vec4<f32>(s_hxz[0], s_hyz[0], hxx + hyy + hzz, 0.0);
        output_fields[out_base + 3u] = vec4<f32>(state.x, state.y, contact, 0.0);
    }
}
