// near_field.wgsl
// 依照 docs/software.md 第 7.4 节 与 docs/mathstrict.md 第 1.5 节
// 近场有限部分无发散差分积分: (rho(x - R*u) - rho(x)) / R

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

struct KernelGpu {
    center_sigma: vec4<f32>,
    weight_alpha: vec4<f32>,
}

@group(0) @binding(0) var<uniform> g: Globals;
@group(0) @binding(1) var<storage, read> points: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> dirs: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> kernels: array<KernelGpu>;
@group(0) @binding(4) var<storage, read_write> near_field_results: array<f32>;

fn eval_rho(pos: vec3<f32>) -> f32 {
    var sum = 0.0;
    for (var k = 0u; k < g.n_kernels; k += 1u) {
        let kern = kernels[k];
        let c = kern.center_sigma.xyz;
        let sigma = kern.center_sigma.w;
        let weight = kern.weight_alpha.x;
        let alpha = kern.weight_alpha.y;

        let diff = pos - c;
        let r2 = dot(diff, diff);
        let q = 1.0 + sigma * sigma * r2;
        sum += weight * pow(q, -alpha);
    }
    return sum + g.bulk_density;
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

    let x = points[point_id].xyz;
    let u = dirs[dir_id].xyz;

    let rho_x = eval_rho(x);

    // 采用 8 点求积在 (0, r0) 范围内积分
    let r0 = g.r0;
    var integral = 0.0;
    let steps = 8u;
    let dr = r0 / f32(steps);

    for (var i = 0u; i < steps; i += 1u) {
        let r = (f32(i) + 0.5) * dr;
        let y = x - r * u;
        let rho_y = eval_rho(y);
        integral += ((rho_y - rho_x) / r) * dr;
    }

    near_field_results[ray_id] = integral;
}
