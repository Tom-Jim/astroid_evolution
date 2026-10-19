//! One accepted Tier 2 step, packed on the worker for the viewer.
//!
//! Positions are `scale * ω̂ * R`. The radial morph between two snapshots is
//! `scale * ω̂ * mix(R_from, R_to, smoothstep(τ))`. `d_radial` stays a scalar.

use rayon::prelude::*;

use crate::common::math::DVec3;
use crate::scf_simulator::evolve::{State, StepFields, StepRecord};

pub const PARTICLE_COUNT: usize = 4096;
pub const STREAMLINE_STEPS: usize = 8;

/// Scalar acceptance numbers of one geological step (§2.15, §10.4).
#[derive(Clone, Copy, Debug, Default)]
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
    }
}

#[derive(Clone, Copy, Debug)]
pub struct VertexSample {
    pub f_delta: f32,
    pub movable: f32,
    pub v_sl: [f32; 3],
    pub rho_s: f32,
    pub j_net: f32,
    pub d_radial: f32,
    pub slack_max: f32,
    pub slack_avg: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct StreamPoint {
    pub verts: [u32; 3],
    pub bary: [f32; 3],
}

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
    pub kind: &'static str,
    pub dt_bound: &'static str,
    pub audit: StepAudit,
    pub slack_tol: f32,
    pub f_delta_p95: f32,
    pub slack_p95: f32,
    pub rho_s_p95: f32,
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
}

impl StepSnapshot {
    pub fn write_obj(&self, path: &std::path::Path) -> Result<(), String> {
        write_wavefront(
            path,
            &format!(
                "# tier2 step={} t={:.6e} omega={:.6e} radius=[{:.6e},{:.6e}] m\n",
                self.step, self.time, self.omega, self.radius_min, self.radius_max
            ),
            &self.vertices_m,
            &self.faces,
        )
    }
}

pub fn write_wavefront(
    path: &std::path::Path,
    header: &str,
    vertices: &[[f64; 3]],
    faces: &[[u32; 3]],
) -> Result<(), String> {
    let mut text = header.to_string();
    for v in vertices {
        text.push_str(&format!("v {:.9} {:.9} {:.9}\n", v[0], v[1], v[2]));
    }
    for f in faces {
        text.push_str(&format!("f {} {} {}\n", f[0] + 1, f[1] + 1, f[2] + 1));
    }
    std::fs::write(path, text).map_err(|err| format!("{}: {err}", path.display()))
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
            let len = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2])
                .sqrt();
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
    StepSnapshot {
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
        kind: record.kind,
        dt_bound: record.dt_bound,
        audit: audit_of(record),
        slack_tol,
        f_delta_p95: percentile_95(field.iter().map(|sample| sample.f_delta.max(0.0))),
        slack_p95: percentile_95(field.iter().map(|sample| sample.slack_max)),
        rho_s_p95: percentile_95(field.iter().map(|sample| sample.rho_s)),
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::{
        morphed_position, pack, pack_initial, posed_normals, posed_positions, scatter_face_slack,
        smoothstep,
    };
    use crate::common::density::CauchyKernel;
    use crate::common::math::DVec3;
    use crate::scf_simulator::body::{Body, Topology};
    use crate::scf_simulator::evolve::{State, StepFields, StepRecord};

    fn record() -> StepRecord {
        StepRecord {
            step: 2,
            t: 3.0,
            advance: 1.0,
            omega: 0.2,
            kind: "spin-only",
            dt: 1.0,
            dt_bound: "omega",
            dt_omega: 1.0,
            quasi_static: true,
            core_mass_fraction: 1.0,
            mass: 5.0,
            min_radial: 1.0,
            ..StepRecord::default()
        }
    }

    #[test]
    fn morphed_radius_is_the_scaled_ray() {
        let dir = [0.0, 1.0, 0.0];
        let start = morphed_position(dir, 2.0, 4.0, 0.0, 0.5);
        assert!(start[0].abs() < 1.0e-6);
        assert!((start[1] - 1.0).abs() < 1.0e-5);
        let end = morphed_position(dir, 2.0, 4.0, 1.0, 0.5);
        assert!((end[1] - 2.0).abs() < 1.0e-5);
        assert!((smoothstep(0.5) - 0.5).abs() < 1.0e-6);
        let mid = morphed_position(dir, 2.0, 4.0, 0.5, 1.0);
        assert!((mid[1] - 3.0).abs() < 1.0e-5);
    }

    #[test]
    fn the_screen_pose_matches_the_shader_mix() {
        let from = [[0.0, 0.0, 1.0], [2.0, 0.0, 0.0]];
        let to = [[0.0, 0.0, 3.0], [6.0, 0.0, 0.0]];
        assert_eq!(posed_positions(&from, &to, 0.0), from);
        assert_eq!(posed_positions(&from, &to, 1.0), to);
        let s = smoothstep(0.25);
        let mid = posed_positions(&from, &to, 0.25);
        assert!((mid[0][2] - (1.0 + 2.0 * s)).abs() < 1.0e-6);
        assert!((mid[1][0] - (2.0 + 4.0 * s)).abs() < 1.0e-6);
        assert!((mid[0][0]).abs() < 1.0e-6);
        assert!((mid[0][1]).abs() < 1.0e-6);

        let from_n = [[0.0, 1.0, 0.0]];
        let to_n = [[0.0, 0.0, 1.0]];
        let start_n = posed_normals(&from_n, &to_n, 0.0);
        let end_n = posed_normals(&from_n, &to_n, 1.0);
        assert!((start_n[0][1] - 1.0).abs() < 1.0e-6);
        assert!((end_n[0][2] - 1.0).abs() < 1.0e-6);
        let mid_n = posed_normals(&from_n, &to_n, 0.5);
        let unit = (0.5f32).sqrt();
        assert!((mid_n[0][1] - unit).abs() < 1.0e-5);
        assert!((mid_n[0][2] - unit).abs() < 1.0e-5);
    }

    #[test]
    fn face_slack_scatters_by_the_incident_maximum_and_the_area_weight() {
        let faces = [[0, 1, 2], [0, 2, 3]];
        let (max, avg) = scatter_face_slack(4, &faces, &[2.0, 6.0], &[1.0, 3.0]);
        assert!((max[0] - 6.0).abs() < 1.0e-6);
        assert!((max[1] - 2.0).abs() < 1.0e-6);
        assert!((max[2] - 6.0).abs() < 1.0e-6);
        assert!((avg[0] - 5.0).abs() < 1.0e-5);
        assert!((avg[1] - 2.0).abs() < 1.0e-5);
        assert!((avg[3] - 6.0).abs() < 1.0e-5);
    }

    #[test]
    fn a_spin_step_snapshot_keeps_zero_radial_increment_and_stays_on_the_seed() {
        let topo = Arc::new(Topology::icosphere(0));
        let count = topo.n_vertices();
        let body = Body::new(
            topo.clone(),
            vec![2.0; count],
            1000.0,
            Vec::<CauchyKernel>::new(),
            vec![0.0; count],
        )
        .unwrap();
        let state = State {
            body,
            level: 0,
            t: 3.0,
            step: 2,
            omega: 0.2,
            frame_rotation: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            frame_translation: DVec3::ZERO,
            velocity: Vec::new(),
        };
        let fields = StepFields {
            radius_from: vec![1.0; count],
            f_delta: vec![0.0; count],
            movable: vec![false; count],
            v_sl: vec![DVec3::ZERO; count],
            normal: state.body.topo.dirs.clone(),
            j_ero: vec![0.0; count],
            j_dep: vec![0.0; count],
            d_radial: vec![0.0; count],
            slack: vec![0.0; state.body.topo.n_faces()],
            g_face: vec![DVec3::ZERO; state.body.topo.n_faces()],
            flow_active: false,
        };
        let snapshot = pack(&record(), &state, &fields, 0.5, 0.01, 1.5);
        assert_eq!(snapshot.field.len(), state.body.radius.len());
        assert!(snapshot.field.iter().all(|sample| sample.d_radial == 0.0));
        assert!(!snapshot.flow_active);
        for (shown, dir) in snapshot.position_to.iter().zip(&state.body.topo.dirs) {
            let expected = morphed_position([dir.x, dir.y, dir.z], 2.0, 2.0, 1.0, 0.5);
            assert!((shown[0] - expected[0]).abs() < 1.0e-5);
            assert!((shown[1] - expected[1]).abs() < 1.0e-5);
            assert!((shown[2] - expected[2]).abs() < 1.0e-5);
        }
        assert_eq!(
            snapshot.stream.len(),
            super::PARTICLE_COUNT * super::STREAMLINE_STEPS
        );
        for particle in 0..super::PARTICLE_COUNT {
            let seed = (particle * count) / super::PARTICLE_COUNT;
            for step in 0..super::STREAMLINE_STEPS {
                let point = snapshot.stream[particle * super::STREAMLINE_STEPS + step];
                assert_eq!(point.verts, [seed as u32; 3]);
            }
        }
    }

    #[test]
    fn the_initial_snapshot_is_a_still_tier2_pose() {
        let topo = Arc::new(Topology::icosphere(0));
        let count = topo.n_vertices();
        let body = Body::new(
            topo.clone(),
            vec![2.0; count],
            1000.0,
            Vec::<CauchyKernel>::new(),
            vec![0.0; count],
        )
        .unwrap();
        let state = State {
            body,
            level: 0,
            t: 0.0,
            step: 0,
            omega: 0.1,
            frame_rotation: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            frame_translation: DVec3::ZERO,
            velocity: Vec::new(),
        };
        let snapshot = pack_initial(&state, 9.0, 0.5, 0.01);
        assert_eq!(snapshot.position_from, snapshot.position_to);
        assert_eq!(snapshot.normal_from, snapshot.normal_to);
        assert_eq!((snapshot.step, snapshot.mass), (0, 9.0));
        assert!(snapshot.field.iter().all(|sample| sample.f_delta == 0.0));
        assert_eq!(
            snapshot.stream.len(),
            super::PARTICLE_COUNT * super::STREAMLINE_STEPS
        );
        let shown = posed_positions(&snapshot.position_from, &snapshot.position_to, 0.5);
        assert_eq!(shown, snapshot.position_to);
    }
}
