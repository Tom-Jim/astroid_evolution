// debug_lines.wgsl
// 依照 docs/software.md 第 13 节
// 法向量、射线与误差指示线段渲染

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

struct LineVertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec4<f32>,
}

struct LineVertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
}

@vertex
fn vs_main(in: LineVertexInput) -> LineVertexOutput {
    var out: LineVertexOutput;
    out.clip_position = u.view_proj * vec4<f32>(in.position, 1.0);
    out.color = in.color;
    return out;
}

@fragment
fn fs_main(in: LineVertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}
