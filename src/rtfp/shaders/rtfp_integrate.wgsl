// Pass 5: rtfp_integrate.wgsl
// 依照 docs/software.md 第 12.1 节
// 一个线程处理一条射线 (field_point_id, direction_id)
// 同时融合求解 I_U, I_g, I_H，输出 per-ray 场贡献 (3 个 vec4<f32>)

struct Globals {
    n_points: u32,
    n_dirs: u32,
    n_tris: u32,
    n_kernels: u32,
    t_min: f32,
    t_max: f32,
    g_si: f32,
    r0: f32,
    ell0: f32,
    bulk_density: f32,
    _pad: vec2<f32>,
}

struct KernelGpu {
    center_sigma: vec4<f32>, // center.xyz, sigma
    weight_alpha: vec4<f32>, // weight, alpha, pad0, pad1
}

@group(0) @binding(0) var<uniform> g: Globals;
@group(0) @binding(1) var<storage, read> points: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> dirs: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> kernels: array<KernelGpu>;
@group(0) @binding(4) var<storage, read> segment_counts: array<u32>;
@group(0) @binding(5) var<storage, read> segments: array<vec2<f32>>;
// 输出 per-ray 场数据: 每个 ray 连续占 3 个 vec4<f32>
// [3*ray_id + 0] = (U, g.x, g.y, g.z)
// [3*ray_id + 1] = (H.xx, H.yy, H.zz, H.xy)
// [3*ray_id + 2] = (H.xz, H.yz, 0.0, 0.0)
@group(0) @binding(6) var<storage, read_write> ray_fields: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read> near_field_results: array<f32>;

const MAX_SEGMENTS: u32 = 256u;

fn d_eval(r: f32, b: f32, d: f32) -> f32 {
    let diff = r - b;
    return max(diff * diff + d * d, 1e-30);
}

fn p_eval(r: f32, b: f32, d: f32) -> f32 {
    if (r == 0.0) {
        let a0 = max(b * b + d * d, 1e-30);
        return 0.5 * log(a0) - (b / d) * atan(b / d);
    }
    return 0.5 * log(d_eval(r, b, d)) + (b / d) * atan((r - b) / d);
}

fn q_eval(r: f32, b: f32, d: f32) -> f32 {
    if (r == 0.0) {
        return -(1.0 / d) * atan(b / d);
    }
    return (1.0 / d) * atan((r - b) / d);
}

fn l_eval(r: f32, a0: f32, b: f32, d: f32) -> f32 {
    let term = log(r) - 0.5 * log(d_eval(r, b, d)) + (b / d) * atan((r - b) / d);
    return term / a0;
}

// 零端点有限部分
fn l_fp_eval(r1: f32, a0: f32, b: f32, d: f32, ell0: f32) -> f32 {
    let dl = d_eval(r1, b, d);
    let term_log = log(r1 / ell0) - 0.5 * log(max(dl / a0, 1e-30));
    let term_atan = (b / d) * (atan((r1 - b) / d) + atan(b / d));
    return (term_log + term_atan) / a0;
}

const GL16_NODES: array<f32, 16> = array<f32, 16>(
    -0.989400935, -0.944575023, -0.865631202, -0.755404408,
    -0.617876244, -0.458016778, -0.281603551, -0.095012510,
     0.095012510,  0.281603551,  0.458016778,  0.617876244,
     0.755404408,  0.865631202,  0.944575023,  0.989400935
);

const GL16_WEIGHTS: array<f32, 16> = array<f32, 16>(
    0.027152459, 0.062253524, 0.095158512, 0.124628971,
    0.149595989, 0.169156519, 0.182603415, 0.189450610,
    0.189450610, 0.182603415, 0.169156519, 0.149595989,
    0.124628971, 0.095158512, 0.062253524, 0.027152459
);

fn fixed_quadrature_triplet(alpha: f32, b: f32, d: f32, r0: f32, r1: f32) -> vec3<f32> {
    let h = 0.5 * (r1 - r0);
    let mid = 0.5 * (r1 + r0);
    var sum_u = 0.0;
    var sum_g = 0.0;
    var sum_h = 0.0;
    for (var i = 0u; i < 16u; i += 1u) {
        let node = GL16_NODES[i];
        let weight = GL16_WEIGHTS[i];
        let r = mid + h * node;
        let d_val = d_eval(r, b, d);
        let d_pow = pow(d_val, -alpha);
        sum_u += weight * r * d_pow;
        sum_g += weight * d_pow;
        if (abs(r) > 1e-6) {
            sum_h += weight * (d_pow / r);
        }
    }
    return vec3<f32>(sum_u * h, sum_g * h, sum_h * h);
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ray_id = gid.x;
    let total_rays = g.n_points * g.n_dirs;
    if (ray_id >= total_rays) {
        return;
    }

    let point_id = ray_id / g.n_dirs;
    let dir_id = ray_id % g.n_dirs;

    let x = points[point_id].xyz;
    let u_entry = dirs[dir_id];
    let u = u_entry.xyz;
    let omega = u_entry.w;

    let num_segs = segment_counts[ray_id];
    let base_seg = ray_id * MAX_SEGMENTS;

    var total_u = 0.0;
    var total_g = vec3<f32>(0.0, 0.0, 0.0);
    var h_xx = 0.0;
    var h_yy = 0.0;
    var h_zz = 0.0;
    var h_xy = 0.0;
    var h_xz = 0.0;
    var h_yz = 0.0;

    let t_xx = 3.0 * u.x * u.x - 1.0;
    let t_yy = 3.0 * u.y * u.y - 1.0;
    let t_zz = 3.0 * u.z * u.z - 1.0;
    let t_xy = 3.0 * u.x * u.y;
    let t_xz = 3.0 * u.x * u.z;
    let t_yz = 3.0 * u.y * u.z;

    for (var k = 0u; k < g.n_kernels; k += 1u) {
        let kern = kernels[k];
        let c = kern.center_sigma.xyz;
        let sigma = kern.center_sigma.w;
        let weight = kern.weight_alpha.x;
        let alpha = kern.weight_alpha.y;

        let p = x - c;
        let p2 = dot(p, p);
        let b = dot(p, u);
        let inv_s2 = 1.0 / (sigma * sigma);
        let d2 = max(inv_s2 + p2 - b * b, 1e-20);
        let d = sqrt(d2);
        let a0 = b * b + d2;

        let s_factor = pow(sigma, -2.0 * alpha);
        let w_factor = weight * s_factor;

        var k_iu = 0.0;
        var k_ig = 0.0;
        var k_ih = 0.0;

        let is_alpha_one = abs(alpha - 1.0) < 1e-4;

        for (var s = 0u; s < num_segs; s += 1u) {
            let seg = segments[base_seg + s];
            let r0 = seg.x;
            let r1 = seg.y;

            if (r1 > r0) {
                if (is_alpha_one) {
                    k_iu += p_eval(r1, b, d) - p_eval(r0, b, d);
                    k_ig += q_eval(r1, b, d) - q_eval(r0, b, d);

                    if (r0 > 1e-5) {
                        k_ih += l_eval(r1, a0, b, d) - l_eval(r0, a0, b, d);
                    } else {
                        k_ih += l_fp_eval(r1, a0, b, d, g.ell0);
                    }
                } else {
                    let trip = fixed_quadrature_triplet(alpha, b, d, r0, r1);
                    k_iu += trip.x;
                    k_ig += trip.y;
                    k_ih += trip.z;
                }
            }
        }

        total_u += g.g_si * omega * w_factor * k_iu;

        let gw = -g.g_si * omega * w_factor * k_ig;
        total_g += gw * u;

        let hw = g.g_si * omega * w_factor * k_ih;
        h_xx += hw * t_xx;
        h_yy += hw * t_yy;
        h_zz += hw * t_zz;
        h_xy += hw * t_xy;
        h_xz += hw * t_xz;
        h_yz += hw * t_yz;
    }

    // 均匀背景密度的径向三重积分，所有交段共用同一射线几何。
    for (var s = 0u; s < num_segs; s += 1u) {
        let seg = segments[base_seg + s];
        let r0 = max(seg.x, 0.0);
        let r1 = seg.y;
        if (r1 <= r0) { continue; }
        let iu = 0.5 * (r1*r1 - r0*r0);
        let ig = r1 - r0;
        total_u += g.g_si * omega * g.bulk_density * iu;
        total_g -= g.g_si * omega * g.bulk_density * ig * u;
        if (r0 > 0.0) {
            let ih = log(r1 / r0);
            let hw = g.g_si * omega * g.bulk_density * ih;
            h_xx += hw*t_xx; h_yy += hw*t_yy; h_zz += hw*t_zz;
            h_xy += hw*t_xy; h_xz += hw*t_xz; h_yz += hw*t_yz;
        }
    }

    // near_field_results 是 (rho(x-Ru)-rho(x))/R 的径向积分，
    // 只进入 Hessian 的角向张量，绝不能加到势。
    let near_hw = g.g_si * omega * near_field_results[ray_id];
    h_xx += near_hw*t_xx; h_yy += near_hw*t_yy; h_zz += near_hw*t_zz;
    h_xy += near_hw*t_xy; h_xz += near_hw*t_xz; h_yz += near_hw*t_yz;
    let out_base = 3u * ray_id;
    ray_fields[out_base + 0u] = vec4<f32>(total_u, total_g.x, total_g.y, total_g.z);
    ray_fields[out_base + 1u] = vec4<f32>(h_xx, h_yy, h_zz, h_xy);
    ray_fields[out_base + 2u] = vec4<f32>(h_xz, h_yz, 0.0, 0.0);
}
