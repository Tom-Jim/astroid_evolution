struct CauchyKernel {
    center: vec3<f32>,
    weight: f32,
    sigma: f32,
    alpha: f32,
    _pad0: f32,
    _pad1: f32,
}
;
struct ScfParams {
    kernels: array<CauchyKernel, 4>,
    omega: f32,
    w_target: f32,
    friction_angle: f32,
    relax_factor: f32,
    g_const: f32,
    vertex_count: u32,
    dt: f32,
    time: f32,
}
;
struct VertexIn {
    position: vec3<f32>,
    normal: vec3<f32>,
    orig_dir: vec3<f32>,
    radius: f32,
}
;
struct VertexOut {
    position: vec3<f32>,
    normal: vec3<f32>,
    g_eff: vec3<f32>,
    radius: f32,
    w_eff: f32,
    slope_angle: f32,
    pad: vec2<f32>,
}
;
@group(0) @binding(0)
var<uniform> params: ScfParams;
@group(0) @binding(1)
var<storage, read> vertices_in: array<VertexIn>;
@group(0) @binding(2)
var<storage, read_write> vertices_out: array<VertexOut>;
fn compute_cauchy_field(x: vec3<f32>, k_idx: u32) -> vec4<f32> {
    let k = params.kernels[k_idx];
    let diff = x - k.center;
    let dist_sq = dot(diff, diff);
    let s2 = k.sigma * k.sigma;
    let denom = 1.0 + s2 * dist_sq;
    let factor = pow(denom, -k.alpha);
    let dist = max(sqrt(dist_sq), 1e-5);
    let u_contrib = params.g_const * k.weight * factor / dist;
    let grad_factor = -params.g_const * k.weight
        * (factor / (dist * dist * dist) + 2.0 * k.alpha * s2 * pow(denom, -k.alpha - 1.0) / dist);
    let grad_contrib = grad_factor * diff;
    return vec4<f32>(grad_contrib, u_contrib);
}
@compute @workgroup_size(64, 1, 1)
fn main_scf_update(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if idx >= params.vertex_count {
        return;
    }
    let v_in = vertices_in[idx];
    let x = v_in.position;
    let u_dir = normalize(v_in.orig_dir);
    let r_curr = v_in.radius;
    var total_u: f32 = 0.0;
    var grad_u = vec3<f32>(0.0);
    for(var k = 0u; k < 4u; k = k + 1u) {
        let field = compute_cauchy_field(x, k);
        grad_u += field.xyz;
        total_u += field.w;
    }
    let omega2 = params.omega * params.omega;
    let w_eff = total_u + 0.5 * omega2 * (x.x * x.x + x.y * x.y);
    let g_centrifugal = vec3<f32>(omega2 * x.x, omega2 * x.y, 0.0);
    let g_eff = grad_u + g_centrifugal;
    let g_norm = max(length(g_eff), 1e-6);
    let g_radial = dot(g_eff, u_dir);
    var delta_r_eq: f32 = 0.0;
    if abs(g_radial) > 1e-5 {
        delta_r_eq = (params.w_target - w_eff) / g_radial;
    }
    delta_r_eq = clamp(delta_r_eq, -0.3 * r_curr, 0.3 * r_curr);
    let r_eq = max(r_curr + delta_r_eq, 0.1);
    var normal = v_in.normal;
    let cross_gn = cross(g_eff, normal);
    let sin_slope = length(cross_gn) / g_norm;
    let slope_angle = asin(clamp(sin_slope, 0.0, 1.0));
    let sin_phi = sin(params.friction_angle);
    var r_projected = r_eq;
    if sin_slope > sin_phi {
        let slope_overshoot = sin_slope - sin_phi;
        r_projected = mix(r_eq, r_curr, clamp(slope_overshoot * 2.0, 0.0, 1.0));
    }
    let r_next = (1.0 - params.relax_factor) * r_curr + params.relax_factor * r_projected;
    let next_pos = r_next * u_dir;
    let next_normal = normalize(mix(normal, u_dir, 0.1));
    vertices_out[idx].position = next_pos;
    vertices_out[idx].normal = next_normal;
    vertices_out[idx].g_eff = g_eff;
    vertices_out[idx].radius = r_next;
    vertices_out[idx].w_eff = w_eff;
    vertices_out[idx].slope_angle = slope_angle;
    vertices_out[idx].pad = vec2<f32>(0.0, 0.0);
}
