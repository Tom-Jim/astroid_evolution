//! One accepted Tier 2 step, packed on the worker for the viewer.
//!
//! Positions are `scale * ω̂ * R`. The radial morph between two snapshots is
//! `scale * ω̂ * mix(R_from, R_to, smoothstep(τ))`. `d_radial` stays a scalar.

use rayon::prelude::*;
use serde::{Deserialize, Deserializer, Serialize};

use crate::common::math::DVec3;
use crate::scf::evolve::{State, StepFields, StepRecord};

pub const PARTICLE_COUNT: usize = 4096;
pub const STREAMLINE_STEPS: usize = 8;
/// Direction ticks of the tension layer, spread evenly over the vertices.
pub const TENSION_GLYPHS: usize = 3072;

/// Scalar acceptance numbers of one geological step (§2.15, §10.4).
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct StepAudit {
    pub eps_momentum: f64,
    pub eps_lambda: f64,
    pub eps_picard: f64,
    pub picard_iterations: usize,
    pub tensile_fraction: f64,
    pub inverse_ekman: f64,
    pub mass_residual: f64,
    pub mobile_area_fraction: f64,
    pub core_mass_fraction: f64,
    pub slope_slack: f64,
    pub mass_slack: f64,
    pub n_mobile: usize,
    pub n_locked: usize,
    pub n_deposit: usize,
    pub socp_iterations: u32,
    pub trust: f64,
    pub trust_retries: usize,
    pub dt: f64,
    pub dt_cfl: f64,
    pub dt_c: f64,
    pub dt_char: f64,
    pub flux_gap: f64,
    pub max_locked_step: f64,
    pub max_deposit_miss: f64,
    pub rheo_unconverged: u32,
    /// Runout of the eroded packets (§2.8b): count, mass-weighted mean distance (m),
    /// mean change of |latitude| (rad, negative equatorward), packets that hopped.
    pub runout_traced: usize,
    pub runout_distance: f64,
    pub runout_dlat: f64,
    pub runout_hops: usize,
}

fn audit_of(record: &StepRecord) -> StepAudit {
    StepAudit {
        eps_momentum: record.eps_momentum,
        eps_lambda: record.eps_lambda,
        eps_picard: record.eps_picard,
        picard_iterations: record.picard_iterations,
        tensile_fraction: record.tensile_fraction,
        inverse_ekman: record.inverse_ekman,
        mass_residual: record.mass_residual,
        mobile_area_fraction: record.mobile_area_fraction,
        core_mass_fraction: record.core_mass_fraction,
        slope_slack: record.slope_slack,
        mass_slack: record.mass_slack,
        n_mobile: record.n_mobile,
        n_locked: record.n_locked,
        n_deposit: record.n_deposit,
        socp_iterations: record.socp_iterations,
        trust: record.trust,
        trust_retries: record.trust_retries,
        dt: record.dt,
        dt_cfl: record.dt_cfl,
        dt_c: record.dt_c,
        dt_char: record.dt_char,
        flux_gap: record.flux_gap,
        max_locked_step: record.max_locked_step,
        max_deposit_miss: record.max_deposit_miss,
        rheo_unconverged: record.rheo_unconverged,
        runout_traced: record.runout.traced,
        runout_distance: record.runout.mean_distance,
        runout_dlat: record.runout.mean_dlat,
        runout_hops: record.runout.hops,
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct VertexSample {
    pub f_delta: f32,
    pub movable: f32,
    pub v_sl: [f32; 3],
    pub rho_s: f32,
    pub j_net: f32,
    pub d_radial: f32,
    pub slack_max: f32,
    pub slack_avg: f32,
    /// Largest principal Cauchy stress (Pa, tension positive).
    pub sigma1: f32,
    /// Its direction projected on the tangent plane (unit), zero under compression.
    pub sigma_dir: [f32; 3],
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct StreamPoint {
    pub verts: [u32; 3],
    pub bary: [f32; 3],
}

/// Crosses the worker boundary of the browser build as postcard bytes (`web::wire`).
#[derive(Serialize, Deserialize)]
pub struct StepSnapshot {
    pub step: usize,
    pub time: f64,
    pub omega: f64,
    pub mass: f64,
    pub radius_min: f64,
    pub radius_max: f64,
    pub yielded_fraction: f64,
    pub max_slack: f64,
    pub rho_s_total: f64,
    pub max_displacement: f64,
    pub frame_applied: bool,
    pub wall_seconds: f64,
    pub flow_active: bool,
    /// Which bound set the step (`cfl`, `mode_c`, …).
    pub dt_bound: Tag,
    pub audit: StepAudit,
    pub slack_tol: f32,
    pub f_delta_p95: f32,
    pub slack_p95: f32,
    pub rho_s_p95: f32,
    /// 95th percentile of the positive `σ₁` (Pa), the scale of the tension layer.
    pub sigma_p95: f32,
    /// Largest `σ₁` on the surface (Pa).
    pub tension_max: f64,
    pub faces: Vec<[u32; 3]>,
    /// Final body, metres, Tier 2 frame. The headless OBJ writer uses this.
    pub vertices_m: Vec<[f64; 3]>,
    pub position_from: Vec<[f32; 3]>,
    pub position_to: Vec<[f32; 3]>,
    pub normal_from: Vec<[f32; 3]>,
    pub normal_to: Vec<[f32; 3]>,
    pub field: Vec<VertexSample>,
    /// `PARTICLE_COUNT` seeds, `STREAMLINE_STEPS` samples each.
    pub stream: Vec<StreamPoint>,
    /// Runout paths of this step (§2.8b), `runout::PATH_SAMPLES` points each, view units;
    /// `w = 1` while airborne.
    pub runout: Vec<[f32; 4]>,
    /// Interior density in view units: the matrix, and per core three rows
    /// `(c, w)`, `(S_xx, S_yy, S_zz, α)`, `(S_xy, S_xz, S_yz, 0)`.
    pub bulk: f32,
    pub cores: Vec<[f32; 4]>,
    /// Smallest and largest density on the vertices, the centre and the cores (kg/m³).
    pub rho_range: [f32; 2],
}

/// The interior density of `state` packed for the cutaway (view units `x_view = scale·x`).
fn interior(state: &State, scale: f64) -> (f32, Vec<[f32; 4]>, [f32; 2]) {
    let body = &state.body;
    let mut cores = Vec::with_capacity(3 * body.kernels.len());
    let inv2 = 1.0 / (scale * scale);
    for k in &body.kernels {
        let c = k.center * scale;
        let m = k.metric;
        cores.push([c.x as f32, c.y as f32, c.z as f32, k.weight as f32]);
        cores.push([
            (m[0] * inv2) as f32,
            (m[1] * inv2) as f32,
            (m[2] * inv2) as f32,
            k.alpha as f32,
        ]);
        cores.push([
            (m[3] * inv2) as f32,
            (m[4] * inv2) as f32,
            (m[5] * inv2) as f32,
            0.0,
        ]);
    }
    if cores.is_empty() {
        cores.push([0.0; 4]);
    }
    let rho = |x: DVec3| {
        body.bulk
            + body
                .kernels
                .iter()
                .map(|k| crate::common::density::kernel_rho(k, x))
                .sum::<f64>()
    };
    let (lo, hi) = body
        .positions()
        .par_iter()
        .map(|p| *p * 0.999)
        .chain(rayon::iter::once(DVec3::ZERO))
        .chain(body.kernels.par_iter().map(|k| k.center))
        .map(|x| {
            let r = rho(x);
            (r, r)
        })
        .reduce(
            || (f64::INFINITY, f64::NEG_INFINITY),
            |a, b| (a.0.min(b.0), a.1.max(b.1)),
        );
    (body.bulk as f32, cores, [lo as f32, hi as f32])
}

impl StepSnapshot {
    pub fn write_obj(&self, path: &std::path::Path) -> Result<(), String> {
        crate::common::obj::write(
            path,
            &format!(
                "# tier2 step={} t={:.6e} omega={:.6e} radius=[{:.6e},{:.6e}] m\n",
                self.step, self.time, self.omega, self.radius_min, self.radius_max
            ),
            &self.vertices_m,
            &self.faces,
        )
        .map_err(|e| e.to_string())
    }
}

pub fn smoothstep(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Cartesian mix the vertex shader uses: `mix(from, to, smoothstep(τ))`.
pub fn posed_positions(from: &[[f32; 3]], to: &[[f32; 3]], tau: f32) -> Vec<[f32; 3]> {
    let s = smoothstep(tau);
    let count = from.len().min(to.len());
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let a = from[i];
        let b = to[i];
        out.push([
            a[0] + (b[0] - a[0]) * s,
            a[1] + (b[1] - a[1]) * s,
            a[2] + (b[2] - a[2]) * s,
        ]);
    }
    out
}

/// Mixed normals, then normalized the same way as `morphed_normal` in the shader.
pub fn posed_normals(from: &[[f32; 3]], to: &[[f32; 3]], tau: f32) -> Vec<[f32; 3]> {
    posed_positions(from, to, tau)
        .into_iter()
        .map(|normal| {
            let len =
                (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
            if len > 1.0e-6 {
                [normal[0] / len, normal[1] / len, normal[2] / len]
            } else {
                [0.0, 1.0, 0.0]
            }
        })
        .collect()
}

/// `P = scale * ω̂ * mix(R_from, R_to, smoothstep(τ))`.
pub fn morphed_position(dir: [f64; 3], r_from: f64, r_to: f64, tau: f32, scale: f64) -> [f32; 3] {
    let radius = r_from + (r_to - r_from) * f64::from(smoothstep(tau));
    let span = scale * radius;
    [
        (dir[0] * span) as f32,
        (dir[1] * span) as f32,
        (dir[2] * span) as f32,
    ]
}

pub fn scatter_face_slack(
    vertex_count: usize,
    faces: &[[u32; 3]],
    slack: &[f64],
    area: &[f64],
) -> (Vec<f32>, Vec<f32>) {
    let mut slack_max = vec![0.0f32; vertex_count];
    let mut slack_sum = vec![0.0f64; vertex_count];
    let mut area_sum = vec![0.0f64; vertex_count];
    for (face, tri) in faces.iter().enumerate() {
        let value = slack.get(face).copied().unwrap_or(0.0);
        let weight = area.get(face).copied().unwrap_or(0.0).max(0.0);
        for vertex in tri {
            let index = *vertex as usize;
            if index >= vertex_count {
                continue;
            }
            slack_max[index] = slack_max[index].max(value as f32);
            slack_sum[index] += value * weight;
            area_sum[index] += weight;
        }
    }
    let slack_avg = slack_sum
        .iter()
        .zip(&area_sum)
        .zip(&slack_max)
        .map(|((sum, weight), max)| {
            if *weight > 0.0 {
                (sum / weight) as f32
            } else {
                *max
            }
        })
        .collect();
    (slack_max, slack_avg)
}

pub fn pack(
    record: &StepRecord,
    state: &State,
    fields: &StepFields,
    scale: f64,
    slack_tol: f32,
    wall_seconds: f64,
) -> StepSnapshot {
    let dirs = &state.body.topo.dirs;
    let faces = &state.body.topo.faces;
    let nv = dirs.len();
    let position_from = view_positions(dirs, &fields.radius_from, scale);
    let position_to = view_positions(dirs, &state.body.radius, scale);
    let normal_from = smooth_normals(&position_from, faces);
    let normal_to = smooth_normals(&position_to, faces);
    let geometry = state.body.face_geometry();
    let area: Vec<f64> = geometry.iter().map(|face| face.2).collect();
    let (slack_max, slack_avg) = scatter_face_slack(nv, faces, &fields.slack, &area);
    let tension = principal_tension(nv, faces, &fields.stress, &area, &normal_to);
    let mut field = Vec::with_capacity(nv);
    for i in 0..nv {
        let velocity = fields.v_sl.get(i).copied().unwrap_or(DVec3::ZERO);
        field.push(VertexSample {
            f_delta: fields.f_delta.get(i).copied().unwrap_or(0.0) as f32,
            movable: if fields.movable.get(i).copied().unwrap_or(false) {
                1.0
            } else {
                0.0
            },
            v_sl: [velocity.x as f32, velocity.y as f32, velocity.z as f32],
            rho_s: state.body.rho_s.get(i).copied().unwrap_or(0.0) as f32,
            j_net: (fields.j_dep.get(i).copied().unwrap_or(0.0)
                - fields.j_ero.get(i).copied().unwrap_or(0.0)) as f32,
            d_radial: fields.d_radial.get(i).copied().unwrap_or(0.0) as f32,
            slack_max: slack_max.get(i).copied().unwrap_or(0.0),
            slack_avg: slack_avg.get(i).copied().unwrap_or(0.0),
            sigma1: tension[i].0,
            sigma_dir: tension[i].1,
        });
    }
    let (mut radius_min, mut radius_max) = (f64::INFINITY, 0.0_f64);
    for radius in &state.body.radius {
        radius_min = radius_min.min(*radius);
        radius_max = radius_max.max(*radius);
    }
    if state.body.radius.is_empty() {
        radius_min = 0.0;
    }
    let neighbors = &state.body.topo.vertex_neighbors;
    let flow_active = fields.flow_active;
    let v_sl = &fields.v_sl;
    let stream = (0..PARTICLE_COUNT)
        .into_par_iter()
        .flat_map(|particle| {
            let seed = if nv == 0 {
                0
            } else {
                (particle * nv) / PARTICLE_COUNT
            };
            walk(dirs, neighbors, v_sl, flow_active, seed)
        })
        .collect();
    let (bulk, cores, rho_range) = interior(state, scale);
    StepSnapshot {
        bulk,
        cores,
        rho_range,
        step: record.step,
        time: record.t,
        omega: record.omega,
        mass: record.mass,
        radius_min,
        radius_max,
        yielded_fraction: record.yielded_fraction,
        max_slack: record.max_slack,
        rho_s_total: record.rho_s_total,
        max_displacement: record.max_displacement,
        frame_applied: record.frame_applied,
        wall_seconds,
        flow_active,
        dt_bound: Tag(record.dt_bound),
        audit: audit_of(record),
        slack_tol,
        f_delta_p95: percentile_95(field.iter().map(|sample| sample.f_delta.max(0.0))),
        slack_p95: percentile_95(field.iter().map(|sample| sample.slack_max)),
        rho_s_p95: percentile_95(field.iter().map(|sample| sample.rho_s)),
        sigma_p95: percentile_95(field.iter().map(|sample| sample.sigma1)),
        tension_max: field
            .iter()
            .map(|sample| f64::from(sample.sigma1))
            .reduce(f64::max)
            .unwrap_or(0.0),
        faces: faces.clone(),
        vertices_m: state
            .body
            .positions()
            .iter()
            .map(|p| p.to_array())
            .collect(),
        position_from,
        position_to,
        normal_from,
        normal_to,
        field,
        stream,
        runout: fields
            .runout
            .iter()
            .map(|p| {
                [
                    (p[0] * scale) as f32,
                    (p[1] * scale) as f32,
                    (p[2] * scale) as f32,
                    p[3] as f32,
                ]
            })
            .collect(),
    }
}

/// Body after `evolve::start`, before the first accepted step.
///
/// `position_from` and `position_to` are the same Tier 2 pose. Fields are zero
/// so the window can enter that frame while the first step is still running.
pub fn pack_initial(state: &State, mass: f64, scale: f64, slack_tol: f32) -> StepSnapshot {
    let nv = state.body.topo.n_vertices();
    let fields = StepFields {
        radius_from: state.body.radius.clone(),
        f_delta: vec![0.0; nv],
        movable: vec![false; nv],
        v_sl: Vec::new(),
        normal: Vec::new(),
        j_ero: Vec::new(),
        j_dep: Vec::new(),
        d_radial: Vec::new(),
        slack: Vec::new(),
        g_face: Vec::new(),
        stress: Vec::new(),
        runout: Vec::new(),
        flow_active: false,
    };
    let record = StepRecord {
        step: state.step,
        t: state.t,
        omega: state.omega,
        mass,
        ..StepRecord::default()
    };
    let mut snap = pack(&record, state, &fields, scale, slack_tol, 0.0);
    for sample in &mut snap.field {
        sample.rho_s = 0.0;
    }
    snap.rho_s_p95 = percentile_95(snap.field.iter().map(|sample| sample.rho_s));
    snap
}

fn view_positions(dirs: &[DVec3], radius: &[f64], scale: f64) -> Vec<[f32; 3]> {
    dirs.iter()
        .enumerate()
        .map(|(i, dir)| {
            let span = scale * radius.get(i).copied().unwrap_or(0.0);
            [
                (dir.x * span) as f32,
                (dir.y * span) as f32,
                (dir.z * span) as f32,
            ]
        })
        .collect()
}

pub fn smooth_normals(positions: &[[f32; 3]], faces: &[[u32; 3]]) -> Vec<[f32; 3]> {
    let count = positions.len();
    if count == 0 {
        return Vec::new();
    }
    let crosses: Vec<(u32, u32, u32, [f32; 3])> = faces
        .par_iter()
        .map(|tri| {
            let a = positions[tri[0] as usize];
            let b = positions[tri[1] as usize];
            let c = positions[tri[2] as usize];
            let ux = b[0] - a[0];
            let uy = b[1] - a[1];
            let uz = b[2] - a[2];
            let vx = c[0] - a[0];
            let vy = c[1] - a[1];
            let vz = c[2] - a[2];
            (
                tri[0],
                tri[1],
                tri[2],
                [uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx],
            )
        })
        .collect();
    let mut acc = vec![[0.0f32; 3]; count];
    for (i, j, k, normal) in crosses {
        for vertex in [i, j, k] {
            let slot = &mut acc[vertex as usize];
            slot[0] += normal[0];
            slot[1] += normal[1];
            slot[2] += normal[2];
        }
    }
    acc.par_iter_mut().for_each(|normal| {
        let len = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
        if len > 1.0e-12 {
            normal[0] /= len;
            normal[1] /= len;
            normal[2] /= len;
        } else {
            *normal = [0.0, 1.0, 0.0];
        }
    });
    acc
}

fn tangent(velocity: DVec3, dir: DVec3) -> DVec3 {
    let normal = dir.try_normalize().unwrap_or(dir);
    velocity - normal * velocity.dot(normal)
}

fn walk(
    dirs: &[DVec3],
    neighbors: &[Vec<usize>],
    v_sl: &[DVec3],
    flow_active: bool,
    seed: usize,
) -> [StreamPoint; STREAMLINE_STEPS] {
    let seed = if dirs.is_empty() {
        0
    } else {
        seed.min(dirs.len() - 1)
    };
    let stay = StreamPoint {
        verts: [seed as u32; 3],
        bary: [1.0, 0.0, 0.0],
    };
    let mut out = [stay; STREAMLINE_STEPS];
    if !flow_active || dirs.is_empty() {
        return out;
    }
    let mut current = seed;
    for slot in 0..STREAMLINE_STEPS {
        out[slot] = StreamPoint {
            verts: [current as u32; 3],
            bary: [1.0, 0.0, 0.0],
        };
        let dir = dirs[current];
        let flow = tangent(v_sl.get(current).copied().unwrap_or(DVec3::ZERO), dir);
        let mut best = current;
        let mut best_dot = 0.0;
        if let Some(ring) = neighbors.get(current) {
            for &next in ring {
                let score = flow.dot(dirs[next] - dir);
                if score > best_dot {
                    best_dot = score;
                    best = next;
                }
            }
        }
        if best == current || best_dot <= 0.0 {
            for rest in slot + 1..STREAMLINE_STEPS {
                out[rest] = out[slot];
            }
            break;
        }
        current = best;
    }
    out
}

/// Largest principal stress `σ₁` (tension positive) at each vertex, and its direction
/// projected on the tangent plane. Face tensors are averaged by area onto the vertices
/// first, so the eigenvectors are those of the averaged tensor.
fn principal_tension(
    nv: usize,
    faces: &[[u32; 3]],
    stress: &[[f64; 6]],
    area: &[f64],
    normals: &[[f32; 3]],
) -> Vec<(f32, [f32; 3])> {
    let mut sum = vec![[0.0f64; 6]; nv];
    let mut weight = vec![0.0f64; nv];
    for (face, tri) in faces.iter().enumerate() {
        let Some(t) = stress.get(face) else {
            continue;
        };
        let a = area.get(face).copied().unwrap_or(0.0).max(0.0);
        for &v in tri {
            let v = v as usize;
            if v < nv {
                for k in 0..6 {
                    sum[v][k] += a * t[k];
                }
                weight[v] += a;
            }
        }
    }
    (0..nv)
        .into_par_iter()
        .map(|i| {
            if weight[i] <= 0.0 {
                return (0.0, [0.0; 3]);
            }
            let t = sum[i].map(|x| x / weight[i]);
            let m = nalgebra::Matrix3::new(t[0], t[3], t[4], t[3], t[1], t[5], t[4], t[5], t[2]);
            let eig = m.symmetric_eigen();
            let k = eig.eigenvalues.imax();
            let sigma1 = eig.eigenvalues[k];
            if sigma1 <= 0.0 {
                return (sigma1 as f32, [0.0; 3]);
            }
            let v = eig.eigenvectors.column(k);
            let v = DVec3::new(v[0], v[1], v[2]);
            let n = normals.get(i).map_or(DVec3::ZERO, |n| {
                DVec3::new(f64::from(n[0]), f64::from(n[1]), f64::from(n[2]))
            });
            let along = (v - n * v.dot(n)).try_normalize().unwrap_or(DVec3::ZERO);
            (sigma1 as f32, [along.x as f32, along.y as f32, along.z as f32])
        })
        .collect()
}

/// Vertex neighbours from the faces (the ring `walk` steps through).
pub fn face_neighbors(nv: usize, faces: &[[u32; 3]]) -> Vec<Vec<usize>> {
    let mut neighbors = vec![Vec::with_capacity(6); nv];
    for face in faces {
        for k in 0..3 {
            let (a, b) = (face[k] as usize, face[(k + 1) % 3] as usize);
            if a < nv && b < nv {
                if !neighbors[a].contains(&b) {
                    neighbors[a].push(b);
                }
                if !neighbors[b].contains(&a) {
                    neighbors[b].push(a);
                }
            }
        }
    }
    neighbors
}

/// Streamlines of a stored frame, rebuilt from its view positions, vertex neighbours and
/// surface velocity: the same walk as `pack`.
pub fn streamlines(
    positions: &[[f32; 3]],
    neighbors: &[Vec<usize>],
    field: &[VertexSample],
    flow_active: bool,
) -> Vec<StreamPoint> {
    let nv = positions.len();
    if !flow_active {
        // Every seed stays put; `walk` would return the same.
        return (0..PARTICLE_COUNT)
            .flat_map(|particle| {
                let seed = if nv == 0 { 0 } else { (particle * nv) / PARTICLE_COUNT } as u32;
                [StreamPoint {
                    verts: [seed; 3],
                    bary: [1.0, 0.0, 0.0],
                }; STREAMLINE_STEPS]
            })
            .collect();
    }
    let dirs: Vec<DVec3> = positions
        .iter()
        .map(|p| {
            let p = DVec3::new(f64::from(p[0]), f64::from(p[1]), f64::from(p[2]));
            p.try_normalize().unwrap_or(DVec3::Z)
        })
        .collect();
    let v_sl: Vec<DVec3> = field
        .iter()
        .map(|s| DVec3::new(f64::from(s.v_sl[0]), f64::from(s.v_sl[1]), f64::from(s.v_sl[2])))
        .collect();
    (0..PARTICLE_COUNT)
        .into_par_iter()
        .flat_map_iter(|particle| {
            let seed = if nv == 0 { 0 } else { (particle * nv) / PARTICLE_COUNT };
            walk(&dirs, neighbors, &v_sl, true, seed)
        })
        .collect()
}

/// A Tier 2 label (`&'static str`) that crosses the worker boundary as a string. Reading
/// it back maps the strings Tier 2 uses to themselves and leaks anything else once, so
/// the snapshot needs no borrowed lifetime.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Tag(pub &'static str);

impl Serialize for Tag {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(self.0)
    }
}

impl<'de> Deserialize<'de> for Tag {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let text = <std::borrow::Cow<'de, str>>::deserialize(d)?;
        Ok(Tag(intern(&text)))
    }
}

fn intern(text: &str) -> &'static str {
    match text {
        "geometry" => "geometry",
        "stage-equilibrium" => "stage-equilibrium",
        "spin-only" => "spin-only",
        "cfl" => "cfl",
        "mode_c" => "mode_c",
        "characteristic" => "characteristic",
        "omega" => "omega",
        "dt_max" => "dt_max",
        "t_next" => "t_next",
        "core" => "core",
        "omega/max/t_next" => "omega/max/t_next",
        "" => "",
        other => Box::leak(other.to_string().into_boxed_str()),
    }
}

fn percentile_95(values: impl Iterator<Item = f32>) -> f32 {
    let mut kept: Vec<f32> = values
        .filter(|value| value.is_finite() && *value > 0.0)
        .collect();
    if kept.is_empty() {
        return 1.0;
    }
    let rank = ((kept.len() - 1) as f32 * 0.95) as usize;
    kept.select_nth_unstable_by(rank, |a, b| {
        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
    });
    kept[rank].max(1.0e-6)
}
