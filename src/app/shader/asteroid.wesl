// Field playback for the Tier 2 body.
// Layout follows Bevy 0.20 examples/shader/storage_buffer.wesl:
// the material bind group carries the uniform and the storage arrays.

import bevy_pbr::render::{
    mesh_functions,
    mesh_bindings::mesh,
    mesh_view_bindings::view,
    view_transformations::position_world_to_clip
};

struct PlaybackParams {
    tau: f32,
    time: f32,
    fade: f32,
    mode: u32,
    f_delta_p95: f32,
    slack_p95: f32,
    rho_s_p95: f32,
    slack_tol: f32,
    vertex_count: u32,
    particle_count: u32,
    streamline_steps: u32,
    has_field: u32,
    cut: vec4<f32>,
    interior: u32,
    core_count: u32,
    bulk: f32,
    rho_lo: f32,
    rho_hi: f32,
    sigma_p95: f32,
    glyph_count: u32,
    runout_count: u32,
    runout_samples: u32,
}

@group(constants::MATERIAL_BIND_GROUP) @binding(0)
var<uniform> material: PlaybackParams;
@group(constants::MATERIAL_BIND_GROUP) @binding(1)
var<storage, read> surface: array<vec4<f32>>;
@group(constants::MATERIAL_BIND_GROUP) @binding(2)
var<storage, read> stream: array<vec4<f32>>;
@group(constants::MATERIAL_BIND_GROUP) @binding(3)
var<storage, read> cores: array<vec4<f32>>;
@group(constants::MATERIAL_BIND_GROUP) @binding(4)
var<storage, read> runout: array<vec4<f32>>;

struct Vertex {
    @builtin(instance_index) instance_index: u32,
    @builtin(vertex_index) index: u32,
    @location(0) position: vec3<f32>,
    @if(VERTEX_NORMALS)
    @location(1) normal: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec4<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) extras: vec4<f32>,
    @location(3) flow: vec4<f32>,
    @location(4) marker: f32,
    @location(5) tension: f32,
}

// Ten rows per vertex: positions, normals, flow and ρ_s, slack, fluxes, then the largest
// principal stress σ₁ and its tangent direction (playback.rs `pack_surface`).
fn surface_base(vi: u32) -> u32 {
    return vi * 10u;
}

fn hidden(out_in: VertexOutput) -> VertexOutput {
    var out = out_in;
    out.world_position = vec4<f32>(0.0);
    out.world_normal = vec3<f32>(0.0, 1.0, 0.0);
    out.clip_position = vec4<f32>(0.0, 0.0, 2.0, 1.0);
    return out;
}

fn morphed_position(vi: u32, s: f32) -> vec3<f32> {
    let b = surface_base(vi);
    return mix(surface[b].xyz, surface[b + 1u].xyz, s);
}

fn morphed_normal(vi: u32, s: f32) -> vec3<f32> {
    let b = surface_base(vi);
    let n = mix(surface[b + 2u].xyz, surface[b + 3u].xyz, s);
    let len = length(n);
    if len > 1e-6 {
        return n / len;
    }
    return vec3<f32>(0.0, 1.0, 0.0);
}

fn sample_center(pid: u32, step: u32, s: f32) -> vec3<f32> {
    let base = (pid * material.streamline_steps + step) * 2u;
    let head = stream[base];
    let tail = stream[base + 1u];
    let weight = vec3<f32>(head.w, tail.x, tail.y);
    return weight.x * morphed_position(u32(head.x), s) + weight.y * morphed_position(u32(head.y), s)
        + weight.z * morphed_position(u32(head.z), s);
}

fn sample_vertex(pid: u32, step: u32) -> u32 {
    let base = (pid * material.streamline_steps + step) * 2u;
    return u32(stream[base].x);
}

fn placed(world_from_local: mat4x4<f32>, local: vec3<f32>) -> vec4<f32> {
    return mesh_functions::mesh_position_local_to_world(world_from_local, vec4<f32>(local, 1.0));
}

@vertex
fn vertex(vertex: Vertex) -> VertexOutput {
    var out: VertexOutput;
    out.extras = vec4<f32>(0.0);
    out.flow = vec4<f32>(0.0);
    out.marker = 0.0;
    out.tension = 0.0;
    let world_from_local = mesh_functions::get_world_from_local(vertex.instance_index);
    let tag = mesh_functions::get_tag(vertex.instance_index);
    let s = smoothstep(0.0, 1.0, clamp(material.tau, 0.0, 1.0));
    let fade = clamp(material.fade, 0.0, 1.0);

    if tag == 1u {
        out.marker = 1.0;
        let show = material.has_field == 1u && (material.mode == 3u || material.mode == 4u);
        if !show || material.streamline_steps < 2u {
            return hidden(out);
        }
        let pid = u32(vertex.position.z + 0.5);
        let steps = material.streamline_steps;
        let along = fract(material.time * 0.2) * f32(steps - 1u);
        let i0 = min(u32(floor(along)), steps - 1u);
        let i1 = min(i0 + 1u, steps - 1u);
        let frac = fract(along);
        let center = mix(sample_center(pid, i0, s), sample_center(pid, i1, s), frac);
        let nrm = morphed_normal(sample_vertex(pid, i0), s);
        let right = view.world_from_view[0].xyz;
        let up = view.world_from_view[1].xyz;
        let local = center + (right * vertex.position.x + up * vertex.position.y) * 0.008
            + nrm * 0.004;
        out.world_position = placed(world_from_local, local);
        out.clip_position = position_world_to_clip(out.world_position.xyz);
        out.world_normal = nrm;
        return out;
    }

    if tag == 3u {
        // Runout trail (§10.0.2): segment k of path p, a thin ribbon lifted off the surface.
        // `tension` carries the coordinate along the path, `extras.x` the airborne flag.
        out.marker = 3.0;
        let seg = u32(vertex.position.z + 0.5);
        let per = max(material.runout_samples, 2u) - 1u;
        let path = seg / per;
        let k = seg % per;
        if (material.mode != 3u && material.mode != 4u) || path >= material.runout_count
            || material.interior == 1u {
            return hidden(out);
        }
        let base = path * material.runout_samples;
        let a = runout[base + k];
        let b = runout[base + k + 1u];
        let d = b.xyz - a.xyz;
        if length(d) < 1e-6 {
            return hidden(out);
        }
        let along = normalize(d);
        let up = normalize(a.xyz + b.xyz);
        let across = normalize(cross(up, along));
        let t = vertex.position.x * 0.5 + 0.5;
        let local = mix(a.xyz, b.xyz, t) + across * (vertex.position.y * 0.0025) + up * 0.006;
        out.tension = (f32(k) + t) / f32(per);
        out.extras = vec4<f32>(max(a.w, b.w), 0.0, 0.0, 0.0);
        out.world_position = placed(world_from_local, local);
        out.clip_position = position_world_to_clip(out.world_position.xyz);
        out.world_normal = up;
        return out;
    }

    if tag == 2u {
        // Tension tick: a thin quad through vertex `vi` along the σ₁ direction, its
        // length growing with σ₁ / p95. Compression draws nothing.
        out.marker = 2.0;
        let gid = u32(vertex.position.z + 0.5);
        if material.has_field != 1u || material.mode != 5u || gid >= material.glyph_count
            || material.vertex_count == 0u {
            return hidden(out);
        }
        let vi = (gid * material.vertex_count) / material.glyph_count;
        let b = surface_base(vi);
        let earlier = surface[b + 8u];
        let later = surface[b + 9u];
        let sigma = mix(earlier.w, later.w, fade);
        let flip = select(1.0, -1.0, dot(earlier.xyz, later.xyz) < 0.0);
        let axis = mix(earlier.xyz * flip, later.xyz, fade);
        if sigma <= 0.0 || length(axis) < 1e-4 {
            return hidden(out);
        }
        let along = normalize(axis);
        let nrm = morphed_normal(vi, s);
        let across = normalize(cross(nrm, along));
        let t = clamp(sigma / max(material.sigma_p95, 1e-12), 0.0, 1.0);
        let half_length = 0.006 + 0.016 * t;
        let local = morphed_position(vi, s) + along * (vertex.position.x * half_length)
            + across * (vertex.position.y * 0.0015) + nrm * 0.004;
        out.tension = t;
        out.world_position = placed(world_from_local, local);
        out.clip_position = position_world_to_clip(out.world_position.xyz);
        out.world_normal = nrm;
        return out;
    }

    var local = vertex.position;
    @if(VERTEX_NORMALS)
    var nrm = vertex.normal;
    @else
    var nrm = vec3<f32>(0.0, 1.0, 0.0);
    if material.has_field == 1u && material.vertex_count > 0u {
        let first = mesh[vertex.instance_index].first_vertex_index;
        let vi = vertex.index - first;
        if vi < material.vertex_count {
            let b = surface_base(vi);
            local = morphed_position(vi, s);
            nrm = morphed_normal(vi, s);
            let f_delta = mix(surface[b].w, surface[b + 1u].w, fade);
            let movable = mix(surface[b + 2u].w, surface[b + 3u].w, fade);
            let rho = mix(surface[b + 4u].w, surface[b + 5u].w, fade);
            let slack_max = mix(surface[b + 6u].x, surface[b + 6u].z, fade);
            let slack_avg = mix(surface[b + 6u].y, surface[b + 6u].w, fade);
            let v_from = surface[b + 4u].xyz;
            let v_to = surface[b + 5u].xyz;
            out.extras = vec4<f32>(f_delta, movable, rho, slack_max);
            out.flow = vec4<f32>(mix(v_from, v_to, fade), slack_avg);
            out.tension = mix(surface[b + 8u].w, surface[b + 9u].w, fade);
        }
    }
    out.world_position = placed(world_from_local, local);
    out.clip_position = position_world_to_clip(out.world_position.xyz);
    out.world_normal = nrm;
    return out;
}

fn rock() -> vec3<f32> {
    return vec3<f32>(0.22, 0.23, 0.25);
}

fn shade(nrm: vec3<f32>, albedo: vec3<f32>) -> vec4<f32> {
    let to_light = normalize(vec3<f32>(-0.45, 1.0, -0.3));
    let ndl = clamp(dot(normalize(nrm), to_light), 0.0, 1.0);
    return vec4<f32>(albedo * (0.35 + 0.65 * ndl), 1.0);
}

// ρ(x) = ρ_b + Σ w (1 + dᵀ S d)^(−α), d = x − c (view units).
fn density(x: vec3<f32>) -> f32 {
    var rho = material.bulk;
    for(var k = 0u; k < material.core_count; k = k + 1u) {
        let a = cores[3u * k];
        let s = cores[3u * k + 1u];
        let o = cores[3u * k + 2u];
        let d = x - a.xyz;
        let q = 1.0 + s.x * d.x * d.x + s.y * d.y * d.y + s.z * d.z * d.z
            + 2.0 * (o.x * d.x * d.y + o.y * d.x * d.z + o.z * d.y * d.z);
        rho = rho + a.w * pow(q, -s.w);
    }
    return rho;
}

// Tension ramp: dark red → orange → pale yellow.
fn heat(t: f32) -> vec3<f32> {
    let u = clamp(t, 0.0, 1.0);
    if u < 0.5 {
        return mix(vec3<f32>(0.35, 0.04, 0.05), vec3<f32>(0.93, 0.36, 0.10), u * 2.0);
    }
    return mix(vec3<f32>(0.93, 0.36, 0.10), vec3<f32>(1.0, 0.93, 0.62), u * 2.0 - 1.0);
}

// Perceptually ordered ramp (dark indigo → teal → pale yellow).
fn ramp(t: f32) -> vec3<f32> {
    let c0 = vec3<f32>(0.18, 0.05, 0.33);
    let c1 = vec3<f32>(0.13, 0.45, 0.56);
    let c2 = vec3<f32>(0.37, 0.79, 0.38);
    let c3 = vec3<f32>(0.99, 0.91, 0.15);
    let u = clamp(t, 0.0, 1.0) * 3.0;
    if u < 1.0 {
        return mix(c0, c1, u);
    }
    if u < 2.0 {
        return mix(c1, c2, u - 1.0);
    }
    return mix(c2, c3, u - 2.0);
}

@fragment
fn fragment(in: VertexOutput, @builtin(front_facing) front: bool) -> @location(0) vec4<f32> {
    if in.marker > 2.5 {
        // Dashes flowing from the source to the arrival point; hops in amber.
        let dash = fract(in.tension * 6.0 - material.time * 0.6);
        if dash > 0.7 {
            discard;
        }
        let hue = select(vec3<f32>(0.55, 0.9, 1.0), vec3<f32>(1.0, 0.72, 0.28), in.extras.x > 0.5);
        return vec4<f32>(hue * (0.55 + 0.45 * in.tension), 1.0);
    }
    if in.marker > 1.5 {
        if material.interior == 1u {
            discard;
        }
        return vec4<f32>(mix(vec3<f32>(0.78, 0.86, 1.0), vec3<f32>(1.0), in.tension), 1.0);
    }
    if in.marker > 0.5 {
        if material.interior == 1u {
            discard;
        }
        return shade(in.world_normal, vec3<f32>(0.72, 0.58, 0.32));
    }
    if material.interior == 1u {
        // Cutaway: everything on the camera side of the plane goes; a back face
        // seen through the cut shows the density where its view ray crosses it.
        let n = material.cut.xyz;
        let side = dot(in.world_position.xyz, n) - material.cut.w;
        if side > 0.0 {
            discard;
        }
        if !front {
            let eye = view.world_position;
            let ray = in.world_position.xyz - eye;
            let t = (material.cut.w - dot(eye, n)) / dot(ray, n);
            let x = eye + ray * t;
            let rho = density(x);
            let u = (rho - material.rho_lo) / max(material.rho_hi - material.rho_lo, 1e-6);
            // Iso-density contours every tenth of the range.
            let band = abs(fract(u * 10.0) - 0.5);
            let line = 1.0 - smoothstep(0.0, 0.06, 0.5 - band);
            return vec4<f32>(mix(ramp(u), vec3<f32>(0.04), 0.35 * line), 1.0);
        }
    } else if !front {
        discard;
    }
    let f_delta = in.extras.x;
    let movable = in.extras.y;
    let rho = in.extras.z;
    let slack_max = in.extras.w;
    let flow = in.flow.xyz;
    let nrm = normalize(in.world_normal);
    var albedo = rock();
    let yield_t = clamp(f_delta / max(material.f_delta_p95, 1e-6), 0.0, 1.0);
    let yielded = movable > 0.5 && f_delta > 0.0;
    let hazard = slack_max > material.slack_tol;
    let rho_t = clamp(rho / max(material.rho_s_p95, 1e-6), 0.0, 1.0);
    let pulse = 0.5 + 0.5 * sin(material.time * 6.0);
    if material.mode == 1u || material.mode == 4u {
        if yielded {
            albedo = mix(albedo, vec3<f32>(0.95, 0.35, 0.08), yield_t);
        }
    }
    if material.mode == 2u || material.mode == 4u {
        if hazard {
            albedo = mix(albedo, vec3<f32>(1.0, 0.85, 0.15), pulse);
        }
    }
    if material.mode == 3u || material.mode == 4u {
        albedo = mix(albedo, vec3<f32>(0.55, 0.48, 0.32), rho_t);
        let tangent_flow = flow - nrm * dot(flow, nrm);
        let speed = length(tangent_flow);
        if speed > 1e-8 {
            let phase = dot(in.world_position.xyz, normalize(tangent_flow)) * 12.0
                - material.time * 1.5;
            let stripe = smoothstep(0.55, 0.75, fract(phase));
            albedo = mix(
                    albedo,
                    vec3<f32>(0.85, 0.75, 0.45),
                    stripe * rho_t * clamp(speed * 20.0, 0.0, 1.0),
                );
        }
    }
    if material.mode == 5u && in.tension > 0.0 {
        let t = clamp(in.tension / max(material.sigma_p95, 1e-12), 0.0, 1.0);
        albedo = mix(albedo, heat(t), 0.25 + 0.75 * t);
    }
    return shade(nrm, albedo);
}
