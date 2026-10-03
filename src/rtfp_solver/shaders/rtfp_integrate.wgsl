// P5. α = 1 uses the closed antiderivative. A segment that starts at R = 0
// contributes I_H only on [r0, R]. The ball R < r0 is the P6 difference plus
// ρ log(r0/ℓ0), both scaled by ℓ0 (MS §1.6), never by the near radius alone.
// Those two scalars are added into H. They are not added into U or ∇U:
// (ρ − ρx)/R is the Hessian integrand.

struct Globals {
    n_points: u32,
    n_dirs: u32,
    n_kernels: u32,
    g_si: f32,
    r0: f32,
    ell0: f32,
    bulk_density: f32,
    _pad: f32,
}

struct KernelGpu {
    center_sigma: vec4<f32>,
    weight_alpha: vec4<f32>,
}

@group(0) @binding(0) var<uniform> g: Globals;
@group(0) @binding(1) var<storage, read> points: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> dirs: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> kernels: array<KernelGpu>;
@group(0) @binding(4) var<storage, read> segment_counts: array<u32>;
@group(0) @binding(5) var<storage, read> segments: array<vec2<f32>>;
@group(0) @binding(6) var<storage, read> near_field_results: array<vec2<f32>>;
@group(0) @binding(7) var<storage, read_write> ray_fields: array<vec4<f32>>;

const MAX_SEGMENTS: u32 = 256u;

fn d_eval(r: f32, b: f32, d: f32) -> f32 {
    let diff = r - b;
    return max(diff * diff + d * d, 1e-30);
}

fn p_eval(r: f32, b: f32, d: f32) -> f32 {
    if r == 0.0 {
        let a0 = max(b * b + d * d, 1e-30);
        return 0.5 * log(a0) - (b / d) * atan(b / d);
    }
    return 0.5 * log(d_eval(r, b, d)) + (b / d) * atan((r - b) / d);
}

fn q_eval(r: f32, b: f32, d: f32) -> f32 {
    if r == 0.0 {
        return -(1.0 / d) * atan(b / d);
    }
    return (1.0 / d) * atan((r - b) / d);
}

fn l_eval(r: f32, a0: f32, b: f32, d: f32) -> f32 {
    let term = log(r) - 0.5 * log(d_eval(r, b, d)) + (b / d) * atan((r - b) / d);
    return term / a0;
}

fn l_fp_eval(r1: f32, a0: f32, b: f32, d: f32, ell0: f32) -> f32 {
    let dl = d_eval(r1, b, d);
    let term_log = log(r1 / ell0) - 0.5 * log(max(dl / a0, 1e-30));
    let term_atan = (b / d) * (atan((r1 - b) / d) + atan(b / d));
    return (term_log + term_atan) / a0;
}

fn ih_from_origin(r1: f32, a0: f32, b: f32, d: f32) -> f32 {
    if g.r0 > 0.0 {
        if r1 <= g.r0 {
            return 0.0;
        }
        return l_eval(r1, a0, b, d) - l_eval(g.r0, a0, b, d);
    }
    return l_fp_eval(r1, a0, b, d, g.ell0);
}

const GL16_NODES: array<f32, 16> = array<f32, 16>(
    -0.989400935, -0.944575023, -0.865631202, -0.755404408,
    -0.617876244, -0.458016778, -0.281603551, -0.095012510,
    0.095012510, 0.281603551, 0.458016778, 0.617876244,
    0.755404408, 0.865631202, 0.944575023, 0.989400935,
);
const GL16_WEIGHTS: array<f32, 16> = array<f32, 16>(
    0.027152459, 0.062253524, 0.095158512, 0.124628971,
    0.149595989, 0.169156519, 0.182603415, 0.189450610,
    0.189450610, 0.182603415, 0.169156519, 0.149595989,
    0.124628971, 0.095158512, 0.062253524, 0.027152459,
);

fn quad_span(alpha: f32, b: f32, d: f32, r_a: f32, r_b: f32) -> vec3<f32> {
    let h = 0.5 * (r_b - r_a);
    let mid = 0.5 * (r_b + r_a);
    var sum_u = 0.0;
    var sum_g = 0.0;
    var sum_h = 0.0;
    for (var i = 0u; i < 16u; i = i + 1u) {
        let r = mid + h * GL16_NODES[i];
        let diff = r - b;
        let d_pow = pow(max(diff * diff + d * d, 1e-30), -alpha);
        sum_u = sum_u + GL16_WEIGHTS[i] * r * d_pow;
        sum_g = sum_g + GL16_WEIGHTS[i] * d_pow;
        if r > 1e-8 {
            sum_h = sum_h + GL16_WEIGHTS[i] * d_pow / r;
        }
    }
    return vec3<f32>(sum_u * h, sum_g * h, sum_h * h);
}

fn accumulate_h(h: ptr<function, array<f32, 6>>, scale: f32, u: vec3<f32>) {
    (*h)[0] = (*h)[0] + scale * (3.0 * u.x * u.x - 1.0);
    (*h)[1] = (*h)[1] + scale * (3.0 * u.y * u.y - 1.0);
    (*h)[2] = (*h)[2] + scale * (3.0 * u.z * u.z - 1.0);
    (*h)[3] = (*h)[3] + scale * (3.0 * u.x * u.y);
    (*h)[4] = (*h)[4] + scale * (3.0 * u.x * u.z);
    (*h)[5] = (*h)[5] + scale * (3.0 * u.y * u.z);
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ray_id = gid.x;
    if ray_id >= g.n_points * g.n_dirs {
        return;
    }
    let point_id = ray_id / g.n_dirs;
    let u_entry = dirs[ray_id % g.n_dirs];
    let u = u_entry.xyz;
    let omega = u_entry.w;
    let x = points[point_id].xyz;
    let num_segs = segment_counts[ray_id];
    let base_seg = ray_id * MAX_SEGMENTS;
    var total_u = 0.0;
    var total_g = vec3<f32>(0.0, 0.0, 0.0);
    var h: array<f32, 6>;
    for (var c = 0u; c < 6u; c = c + 1u) {
        h[c] = 0.0;
    }

    for (var k = 0u; k < g.n_kernels; k = k + 1u) {
        let kern = kernels[k];
        let center = kern.center_sigma.xyz;
        let sigma = kern.center_sigma.w;
        let weight = kern.weight_alpha.x;
        let alpha = kern.weight_alpha.y;
        let p = x - center;
        let b = dot(p, u);
        let inv_s2 = 1.0 / (sigma * sigma);
        let d2 = max(inv_s2 + dot(p, p) - b * b, 1e-20);
        let d = sqrt(d2);
        let a0 = b * b + d2;
        let w_factor = weight * pow(sigma, -2.0 * alpha);
        var k_iu = 0.0;
        var k_ig = 0.0;
        var k_ih = 0.0;
        let alpha_one = abs(alpha - 1.0) < 1e-4;
        for (var s = 0u; s < num_segs; s = s + 1u) {
            let seg = segments[base_seg + s];
            let r_a = seg.x;
            let r_b = seg.y;
            if r_b <= r_a {
                continue;
            }
            if alpha_one {
                k_iu = k_iu + p_eval(r_b, b, d) - p_eval(r_a, b, d);
                k_ig = k_ig + q_eval(r_b, b, d) - q_eval(r_a, b, d);
                if r_a > 0.0 {
                    k_ih = k_ih + l_eval(r_b, a0, b, d) - l_eval(r_a, a0, b, d);
                } else {
                    k_ih = k_ih + ih_from_origin(r_b, a0, b, d);
                }
            } else {
                let far_a = select(r_a, g.r0, r_a == 0.0 && g.r0 > 0.0);
                if far_a < r_b {
                    let trip = quad_span(alpha, b, d, far_a, r_b);
                    k_iu = k_iu + trip.x;
                    k_ig = k_ig + trip.y;
                    k_ih = k_ih + trip.z;
                }
                if r_a == 0.0 && g.r0 > 0.0 {
                    let head = quad_span(alpha, b, d, 0.0, min(g.r0, r_b));
                    k_iu = k_iu + head.x;
                    k_ig = k_ig + head.y;
                }
            }
        }
        total_u = total_u + g.g_si * omega * w_factor * k_iu;
        total_g = total_g - g.g_si * omega * w_factor * k_ig * u;
        accumulate_h(&h, g.g_si * omega * w_factor * k_ih, u);
    }

    if g.bulk_density != 0.0 {
        for (var s = 0u; s < num_segs; s = s + 1u) {
            let seg = segments[base_seg + s];
            let r_a = max(seg.x, 0.0);
            let r_b = seg.y;
            if r_b <= r_a {
                continue;
            }
            total_u = total_u + g.g_si * omega * g.bulk_density * 0.5 * (r_b * r_b - r_a * r_a);
            total_g = total_g - g.g_si * omega * g.bulk_density * (r_b - r_a) * u;
            var ih = 0.0;
            if r_a > 0.0 {
                ih = log(r_b / r_a);
            } else if g.r0 > 0.0 && r_b > g.r0 {
                ih = log(r_b / g.r0);
            } else if g.r0 <= 0.0 {
                ih = log(r_b / g.ell0);
            }
            accumulate_h(&h, g.g_si * omega * g.bulk_density * ih, u);
        }
    }

    let near = near_field_results[ray_id];
    accumulate_h(&h, g.g_si * omega * (near.x + near.y), u);

    let out_base = 3u * ray_id;
    ray_fields[out_base + 0u] = vec4<f32>(total_u, total_g.x, total_g.y, total_g.z);
    ray_fields[out_base + 1u] = vec4<f32>(h[0], h[1], h[2], h[3]);
    ray_fields[out_base + 2u] = vec4<f32>(h[4], h[5], 0.0, 0.0);
}
