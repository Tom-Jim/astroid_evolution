// field_overlay.wgsl
// 依照 docs/software.md 第 13 节
// GPU 实时标量场伪彩色映射与等值线着色

struct DisplayUniforms {
    view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
    light_dir: vec4<f32>,
    base_color: vec4<f32>,
    field_params: vec4<f32>, // min_val, max_val, mode, colormap_type
    debug_params: vec4<f32>,
    _reserved: array<vec4<f32>, 4>,
}

@group(0) @binding(0) var<uniform> u: DisplayUniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) scalar_val: f32,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) normalized_scalar: f32,
}

fn colormap_turbo(val: f32) -> vec3<f32> {
    let x = clamp(val, 0.0, 1.0);
    let r = clamp(0.1357 + x * (4.61539 - x * (42.6603 - x * (132.131 - x * (152.55 - x * 61.386)))), 0.0, 1.0);
    let g = clamp(0.0914 + x * (2.19419 + x * (4.84297 - x * (14.185 - x * (4.2773 - x * 11.295)))), 0.0, 1.0);
    let b = clamp(0.1067 + x * (12.5925 - x * (60.1818 - x * (109.07 - x * (88.506 - x * 26.81)))), 0.0, 1.0);
    return vec3<f32>(r, g, b);
}

fn colormap_coolwarm(val: f32) -> vec3<f32> {
    let x = clamp(val, 0.0, 1.0);
    if (x < 0.5) {
        let f = x * 2.0;
        return vec3<f32>(0.23 + 0.77 * f, 0.30 + 0.70 * f, 0.75 + 0.25 * f);
    } else {
        let f = (x - 0.5) * 2.0;
        return vec3<f32>(1.0 - 0.30 * f, 1.0 - 0.85 * f, 1.0 - 0.85 * f);
    }
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.world_pos = in.position;
    out.clip_position = u.view_proj * vec4<f32>(in.position, 1.0);
    out.normal = in.normal;

    let min_v = u.field_params.x;
    let max_v = u.field_params.y;
    let range = max(max_v - min_v, 1e-9);
    out.normalized_scalar = clamp((in.scalar_val - min_v) / range, 0.0, 1.0);

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let cmap_type = u32(u.field_params.w);
    var rgb = colormap_turbo(in.normalized_scalar);
    if (cmap_type == 3u) {
        rgb = colormap_coolwarm(in.normalized_scalar);
    }

    // 简单等值线效果 (十条主轮廓线)
    let isoline = abs(fract(in.normalized_scalar * 10.0) - 0.5);
    if (isoline < 0.06) {
        rgb *= 0.6;
    }

    let n = normalize(in.normal);
    let l = normalize(u.light_dir.xyz);
    let diff = max(dot(n, l), 0.0) * 0.7 + 0.3;

    return vec4<f32>(rgb * diff, 1.0);
}
