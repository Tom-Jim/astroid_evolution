// P6. Near ball R < r0.
// Integrand (ρ(x − R u) − ρ(x)) / R. At R = 0 the limit is −∇ρ · u.
// Also writes ρ log(r0/ℓ0). The consumer multiplies both by G ω T(u).
// Contact −(4πG/3) ϑ ρ I is applied once per point in reduce_field.wgsl.

struct Globals {
    n_points: u32,
    n_dirs: u32,
    n_kernels: u32,
    g_si: f32,
    r0: f32,
    ell0: f32,
    bulk_density: f32,
    _pad: f32,
}

struct KernelGpu {
    center_sigma: vec4<f32>,
    weight_alpha: vec4<f32>,
}

@group(0) @binding(0) var<uniform> g: Globals;
@group(0) @binding(1) var<storage, read> points: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> dirs: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> kernels: array<KernelGpu>;
@group(0) @binding(4) var<storage, read> inside_flags: array<u32>;
@group(0) @binding(5) var<storage, read_write> near_field_results: array<vec2<f32>>;

fn rho_of(pos: vec3<f32>) -> f32 {
    var sum = g.bulk_density;
    for (var k = 0u; k < g.n_kernels; k = k + 1u) {
        let kern = kernels[k];
        let sigma = kern.center_sigma.w;
        let weight = kern.weight_alpha.x;
        let alpha = kern.weight_alpha.y;
        let d = pos - kern.center_sigma.xyz;
        let q = 1.0 + sigma * sigma * dot(d, d);
        sum = sum + weight * pow(q, -alpha);
    }
    return sum;
}

fn grad_rho(pos: vec3<f32>) -> vec3<f32> {
    var g_rho = vec3<f32>(0.0, 0.0, 0.0);
    for (var k = 0u; k < g.n_kernels; k = k + 1u) {
        let kern = kernels[k];
        let sigma = kern.center_sigma.w;
        let weight = kern.weight_alpha.x;
        let alpha = kern.weight_alpha.y;
        let d = pos - kern.center_sigma.xyz;
        let s2 = sigma * sigma;
        let q = 1.0 + s2 * dot(d, d);
        let factor = -2.0 * alpha * weight * s2 * pow(q, -alpha - 1.0);
        g_rho = g_rho + factor * d;
    }
    return g_rho;
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ray_id = gid.x;
    let total_rays = g.n_points * g.n_dirs;
    if ray_id >= total_rays {
        return;
    }
    let point_id = ray_id / g.n_dirs;
    let u = dirs[ray_id % g.n_dirs].xyz;
    if inside_flags[point_id] == 0u || g.r0 <= 0.0 {
        near_field_results[ray_id] = vec2<f32>(0.0, 0.0);
        return;
    }
    let x = points[point_id].xyz;
    let rho_x = rho_of(x);
    let slope = -dot(grad_rho(x), u);
    let steps = 16u;
    let dr = g.r0 / f32(steps);
    var integral = 0.0;
    for (var i = 0u; i < steps; i = i + 1u) {
        let r = (f32(i) + 0.5) * dr;
        var integrand = slope;
        if r > 1e-8 * g.r0 {
            integrand = (rho_of(x - r * u) - rho_x) / r;
        }
        integral = integral + integrand * dr;
    }
    let log_term = rho_x * log(g.r0 / g.ell0);
    near_field_results[ray_id] = vec2<f32>(integral, log_term);
}
