struct UpdateParams {
    relax_factor: f32,
    node_count: u32,
    _pad0: f32,
    _pad1: f32,
}
;
struct SurfaceNodeIn {
    orig_dir: vec3<f32>,
    radius_curr: f32,
    radius_projected: f32,
    normal_curr: vec3<f32>,
}
;
struct SurfaceNodeOut {
    position_next: vec3<f32>,
    radius_next: f32,
    normal_next: vec3<f32>,
    _pad: f32,
}
;
@group(0) @binding(0)
var<uniform> params: UpdateParams;
@group(0) @binding(1)
var<storage, read> nodes_in: array<SurfaceNodeIn>;
@group(0) @binding(2)
var<storage, read_write> nodes_out: array<SurfaceNodeOut>;
@compute @workgroup_size(64, 1, 1)
fn main_surface_update(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if idx >= params.node_count {
        return;
    }
    let node = nodes_in[idx];
    let u_dir = normalize(node.orig_dir);
    let r_next = (1.0 - params.relax_factor) * node.radius_curr
        + params.relax_factor * node.radius_projected;
    let r_valid = max(r_next, 0.05);
    let pos_next = r_valid * u_dir;
    let n_next = normalize(mix(node.normal_curr, u_dir, 0.05));
    nodes_out[idx].position_next = pos_next;
    nodes_out[idx].radius_next = r_valid;
    nodes_out[idx].normal_next = n_next;
    nodes_out[idx]._pad = 0.0;
}
