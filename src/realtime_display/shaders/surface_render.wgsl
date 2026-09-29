// surface_render.wgsl
// 依照 docs/software.md 第 13 节
// 小行星表面光照渲染与多模态着色

struct DisplayUniforms {
    view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
    light_dir: vec4<f32>,
    base_color: vec4<f32>,
    field_params: vec4<f32>,
    debug_params: vec4<f32>,
    _reserved: array<vec4<f32>, 4>,
}

@group(0) @binding(0) var<uniform> u: DisplayUniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.world_pos = in.position;
    out.clip_position = u.view_proj * vec4<f32>(in.position, 1.0);
    out.normal = in.normal;
    out.color = in.color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let n = normalize(in.normal);
    let l = normalize(u.light_dir.xyz);
    let v = normalize(u.camera_pos.xyz - in.world_pos);
    let h = normalize(l + v);

    let ambient = 0.12;
    let diff = max(dot(n, l), 0.0);
    let spec = pow(max(dot(n, h), 0.0), 32.0) * 0.04;

    let lighting = ambient + diff * 0.85 + spec;
    let final_rgb = in.color.rgb * lighting;

    return vec4<f32>(final_rgb, in.color.a);
}
