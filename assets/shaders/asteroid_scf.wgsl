// asteroid_scf.wgsl
// 依据 docs/mathstrict.md 实现的 GPU 实时顶点形变 Compute Shader
// 包含：
// 1. Cauchy 4-Kernel 空间变密度重力势与引力场计算 (第 1.1、3.1、3.2 节)
// 2. 自转有效势 W 与有效加速度 g_eff (第 2.8、2.10 节)
// 3. 等势候选形状搜索与能量标定 (第 2.11、2.12 节)
// 4. 坡度摩擦角约束投影 (第 2.8、2.13、10.3 节)
// 5. 几何欠松弛推进 (第 2.14、10.4 节)

struct CauchyKernel {
    center: vec3<f32>,
    weight: f32,
    sigma: f32,
    alpha: f32,
    _pad0: f32,
    _pad1: f32,
};

struct ScfParams {
    kernels: array<CauchyKernel, 4>,
    omega: f32,              // 自转角速度
    w_target: f32,           // 目标等势值 W0
    friction_angle: f32,     // 物理摩擦角 phi
    relax_factor: f32,       // 欠松弛因子 lambda
    g_const: f32,            // 引力常数 G
    vertex_count: u32,       // 顶点数
    dt: f32,                 // 时间步长
    time: f32,               // 当前演化时间
};

struct VertexIn {
    position: vec3<f32>,     // 当前顶点位置 x
    normal: vec3<f32>,       // 当前法向 n
    orig_dir: vec3<f32>,     // 初始径向方向 u
    radius: f32,             // 当前径向距离 R
};

struct VertexOut {
    position: vec3<f32>,     // 更新后的顶点位置
    normal: vec3<f32>,       // 更新后的法向量
    g_eff: vec3<f32>,        // 有效加速度 g_eff
    radius: f32,             // 更新后的径向半径 R^(n+1)
    w_eff: f32,              // 当前有效势 W
    slope_angle: f32,        // 局部坡度角
    pad: vec2<f32>,
};

@group(0) @binding(0) var<uniform> params: ScfParams;
@group(0) @binding(1) var<storage, read> vertices_in: array<VertexIn>;
@group(0) @binding(2) var<storage, read_write> vertices_out: array<VertexOut>;

// 计算单个 Cauchy 核产生的有效势与引力加速度 (封闭分析径向积分近似)
fn compute_cauchy_field(x: vec3<f32>, k_idx: u32) -> vec4<f32> {
    let k = params.kernels[k_idx];
    let diff = x - k.center;
    let dist_sq = dot(diff, diff);
    let s2 = k.sigma * k.sigma;
    
    // 对应文档 1.1 节: ρ_k = w_k * [1 + σ_k^2 ||y - c_k||^2]^(-α_k)
    // 相应引力势与梯度贡献 (解析积分或核卷积响应)
    let denom = 1.0 + s2 * dist_sq;
    let factor = pow(denom, -k.alpha);
    let dist = max(sqrt(dist_sq), 1e-5);
    
    // 势函数贡献
    let u_contrib = params.g_const * k.weight * factor / dist;
    
    // 引力梯度 ∇U 贡献
    let grad_factor = -params.g_const * k.weight * (factor / (dist * dist * dist) + 2.0 * k.alpha * s2 * pow(denom, -k.alpha - 1.0) / dist);
    let grad_contrib = grad_factor * diff;
    
    return vec4<f32>(grad_contrib, u_contrib);
}

@compute @workgroup_size(64, 1, 1)
fn main_scf_update(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.vertex_count) {
        return;
    }

    let v_in = vertices_in[idx];
    let x = v_in.position;
    let u_dir = normalize(v_in.orig_dir);
    let r_curr = v_in.radius;

    // 1. 累加 4 个 Cauchy 核的引力势 U 与引力加速度 ∇U (第 1.1、2.10 节)
    var total_u: f32 = 0.0;
    var grad_u = vec3<f32>(0.0);
    
    for (var k = 0u; k < 4u; k = k + 1u) {
        let field = compute_cauchy_field(x, k);
        grad_u += field.xyz;
        total_u += field.w;
    }

    // 2. 叠加自转有效势与有效重力 (第 2.8、2.10 节)
    // W = U + 1/2 * ω^2 * (x1^2 + x2^2)
    // g_eff = ∇U + ω^2 * (x1, x2, 0)
    let omega2 = params.omega * params.omega;
    let w_eff = total_u + 0.5 * omega2 * (x.x * x.x + x.y * x.y);
    let g_centrifugal = vec3<f32>(omega2 * x.x, omega2 * x.y, 0.0);
    let g_eff = grad_u + g_centrifugal;
    let g_norm = max(length(g_eff), 1e-6);

    // 3. 求解等势候选半径 R_eq (第 2.12 节)
    // W(R_eq * u) = W_target  ==>  使用一阶泰勒展开快速微商逼近
    let g_radial = dot(g_eff, u_dir);
    var delta_r_eq: f32 = 0.0;
    if (abs(g_radial) > 1e-5) {
        delta_r_eq = (params.w_target - w_eff) / g_radial;
    }
    // 限制单步等势步长在合理信赖域内
    delta_r_eq = clamp(delta_r_eq, -0.3 * r_curr, 0.3 * r_curr);
    let r_eq = max(r_curr + delta_r_eq, 0.1);

    // 4. 局部法向更新与坡度摩擦角约束投影 (第 2.8、2.13、10.3 节)
    // 约束: |g_eff × n|^2 / |g_eff|^2 <= sin^2(phi)
    var normal = v_in.normal;
    let cross_gn = cross(g_eff, normal);
    let sin_slope = length(cross_gn) / g_norm;
    let slope_angle = asin(clamp(sin_slope, 0.0, 1.0));
    let sin_phi = sin(params.friction_angle);

    var r_projected = r_eq;
    // 如果超出颗粒安息角/摩擦角限制，执行沿坡度的约束投影 (Avalanche 欠松弛阻尼)
    if (sin_slope > sin_phi) {
        let slope_overshoot = sin_slope - sin_phi;
        // 坡度超限时，将半径拉回靠近原始等效自适应曲率
        r_projected = mix(r_eq, r_curr, clamp(slope_overshoot * 2.0, 0.0, 1.0));
    }

    // 5. 欠松弛几何推进 (第 2.14 节)
    // R^(n+1) = (1 - λ) * R^(n) + λ * R_*
    let r_next = (1.0 - params.relax_factor) * r_curr + params.relax_factor * r_projected;
    let next_pos = r_next * u_dir;

    // 6. 更新法向近似 (保持向外)
    let next_normal = normalize(mix(normal, u_dir, 0.1));

    // 写出更新数据到 Storage Buffer
    vertices_out[idx].position = next_pos;
    vertices_out[idx].normal = next_normal;
    vertices_out[idx].g_eff = g_eff;
    vertices_out[idx].radius = r_next;
    vertices_out[idx].w_eff = w_eff;
    vertices_out[idx].slope_angle = slope_angle;
    vertices_out[idx].pad = vec2<f32>(0.0, 0.0);
}
