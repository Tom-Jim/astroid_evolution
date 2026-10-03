struct SourceParams {
    dt: f32,
    omega: f32,
    node_count: u32,
    _pad0: f32,
}
;
struct SourceNodeIn {
    position: vec3<f32>,
    g_eff: vec3<f32>,
    q: vec4<f32>,
    normal: vec3<f32>,
    radius: f32,
}
;
struct SourceNodeOut {
    source: vec4<f32>,
}
;
@group(0) @binding(0)
var<uniform> params: SourceParams;
@group(0) @binding(1)
var<storage, read> nodes_in: array<SourceNodeIn>;
@group(0) @binding(2)
var<storage, read_write> nodes_out: array<SourceNodeOut>;
@compute @workgroup_size(64, 1, 1)
fn main_modal_source(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if idx >= params.node_count {
        return;
    }
    let node = nodes_in[idx];
    let g_norm = length(node.g_eff);
    let r = max(node.radius, 0.1);
    let g_radial = dot(node.g_eff, normalize(node.position));
    let driving_force = g_radial / r;
    let s0 = 0.1 * driving_force * (node.q.x);
    let s1 = -0.05 * g_norm * (node.q.y);
    let s2 = -0.15 * g_norm * (node.q.z);
    let s3 = -0.30 * g_norm * (node.q.w);
    nodes_out[idx].source = vec4<f32>(s0, s1, s2, s3);
}
