struct TransportParams {
    dt: f32,
    node_count: u32,
    _pad0: f32,
    _pad1: f32,
}
;
struct ModalNodeIn {
    m: vec4<f32>,
    c_gamma: vec3<f32>,
    j_gamma: f32,
    source: vec4<f32>,
    j_map: f32,
    _pad: vec3<f32>,
}
;
struct ModalNodeOut {
    m_transported: vec4<f32>,
    q_physical: vec4<f32>,
}
;
@group(0) @binding(0)
var<uniform> params: TransportParams;
@group(0) @binding(1)
var<storage, read> nodes_in: array<ModalNodeIn>;
@group(0) @binding(2)
var<storage, read_write> nodes_out: array<ModalNodeOut>;
fn recover_physical_q(m: vec4<f32>) -> vec4<f32> {
    let q0 = 100800.0 * m.x - 415800.0 * m.y + 554400.0 * m.z - 240240.0 * m.w;
    let q1 = -415800.0 * m.x + 1742400.0 * m.y - 2352240.0 * m.z + 1029600.0 * m.w;
    let q2 = 554400.0 * m.x - 2352240.0 * m.y + 3207600.0 * m.z - 1415700.0 * m.w;
    let q3 = -240240.0 * m.x + 1029600.0 * m.y - 1415700.0 * m.z + 629200.0 * m.w;
    return vec4<f32>(q0, q1, q2, q3);
}
@compute @workgroup_size(64, 1, 1)
fn main_modal_transport(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if idx >= params.node_count {
        return;
    }
    let node = nodes_in[idx];
    let j_gamma_curr = max(node.j_gamma, 1e-6);
    let j_map = max(node.j_map, 1e-6);
    let m_adv = j_map * node.m;
    let m_src = params.dt * node.source;
    let m_next = m_adv + m_src;
    let q_next = recover_physical_q(m_next);
    nodes_out[idx].m_transported = m_next;
    nodes_out[idx].q_physical = q_next;
}
