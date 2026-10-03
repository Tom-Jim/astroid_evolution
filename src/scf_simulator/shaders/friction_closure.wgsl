struct RheologyUniforms {
    grain_diameter: f32,
    grain_density: f32,
    mu_s: f32,
    mu_2: f32,
    i_0: f32,
    cohesion_force: f32,
    mu_y: f32,
    cohesion_macro: f32,
    node_count: u32,
    _pad0: f32,
    _pad1: vec2<f32>,
}
;
struct RheologyNodeIn {
    q: vec4<f32>,
    dq_dtheta: vec4<f32>,
    dq_dphi: vec4<f32>,
    pressure: f32,
    packing_fraction: f32,
    coordination_z: f32,
    t_tide_norm: f32,
}
;
struct RheologyNodeOut {
    shear_rate: f32,
    inertial_number: f32,
    mu_dynamic: f32,
    friction_angle: f32,
    sin2_phi: f32,
    _pad: vec3<f32>,
}
;
@group(0) @binding(0)
var<uniform> params: RheologyUniforms;
@group(0) @binding(1)
var<storage, read> nodes_in: array<RheologyNodeIn>;
@group(0) @binding(2)
var<storage, read_write> nodes_out: array<RheologyNodeOut>;
@compute @workgroup_size(64, 1, 1)
fn main_friction_closure(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if idx >= params.node_count {
        return;
    }
    let node = nodes_in[idx];
    let grad_theta = 4.0 * node.dq_dtheta.y + 10.0 * node.dq_dtheta.z + 18.0 * node.dq_dtheta.w;
    let grad_phi = 4.0 * node.dq_dphi.y + 10.0 * node.dq_dphi.z + 18.0 * node.dq_dphi.w;
    let shear_rate = sqrt(grad_theta * grad_theta + grad_phi * grad_phi + 1e-12);
    let d = max(params.grain_diameter, 1e-4);
    let p_c = (node.coordination_z * node.packing_fraction * params.cohesion_force) / (d * d);
    let p_star = max(node.pressure + p_c, 1e-3);
    let vel_scale = sqrt(p_star / max(params.grain_density, 1.0));
    let i_c = (d * shear_rate) / max(vel_scale, 1e-6);
    let mu_d = params.mu_s + (params.mu_2 - params.mu_s) * i_c / (params.i_0 + i_c);
    let tide_term = node.t_tide_norm / 1.41421356;
    let mu_eff = max(
            (params.cohesion_macro + params.mu_y * node.pressure + tide_term) / p_star,
            0.0,
        );
    let phi = atan(mu_eff);
    let sin_phi = sin(phi);
    let sin2_phi = sin_phi * sin_phi;
    nodes_out[idx].shear_rate = shear_rate;
    nodes_out[idx].inertial_number = i_c;
    nodes_out[idx].mu_dynamic = mu_d;
    nodes_out[idx].friction_angle = phi;
    nodes_out[idx].sin2_phi = sin2_phi;
    nodes_out[idx]._pad = vec3<f32>(0.0);
}
