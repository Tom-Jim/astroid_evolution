//! Snapshot adoption and the per-frame `τ` advance.
//!
//! A new step retargets the morph at the position currently on screen and
//! crossfades the fields. After that, the main thread only writes the uniform.

use std::sync::mpsc::TryRecvError;

use bevy::prelude::*;
use bevy::render::storage::ShaderBuffer;

use crate::asteroid_app::material::{AsteroidMaterial, PlaybackParams};
use crate::asteroid_app::pipeline::WorkerMessage;
use crate::asteroid_app::render::{surface_mesh, write_surface};
use crate::asteroid_app::resources::{Evolution, FrameStats, Phase, SurfaceView};
use crate::asteroid_app::snapshot::{
    PARTICLE_COUNT, STREAMLINE_STEPS, StepSnapshot, StreamPoint, VertexSample, smoothstep,
};

#[derive(Clone, Copy, Debug, Resource)]
pub struct VisMode {
    pub mode: u32,
}

impl Default for VisMode {
    fn default() -> Self {
        Self { mode: 4 }
    }
}

pub fn vis_name(mode: u32) -> &'static str {
    match mode {
        1 => "yield",
        2 => "slope",
        3 => "regolith",
        _ => "combined",
    }
}

struct Keyframe {
    position: Vec<[f32; 3]>,
    normal: Vec<[f32; 3]>,
    field: Vec<VertexSample>,
}

#[derive(Resource)]
pub struct Playback {
    pub tau: f32,
    pub fade: f32,
    pub duration: f32,
    pub generation: u64,
    pub dirty: bool,
    pub slack_tol: f32,
    pub f_delta_p95: f32,
    pub slack_p95: f32,
    pub rho_s_p95: f32,
    pub vertex_count: u32,
    pub faces: Vec<[u32; 3]>,
    surface_gpu: Vec<[f32; 4]>,
    stream_gpu: Vec<[f32; 4]>,
    from: Keyframe,
    to: Keyframe,
}

impl Default for Playback {
    fn default() -> Self {
        Self {
            tau: 0.0,
            fade: 1.0,
            duration: 0.25,
            generation: 0,
            dirty: false,
            slack_tol: 0.0,
            f_delta_p95: 1.0,
            slack_p95: 1.0,
            rho_s_p95: 1.0,
            vertex_count: 0,
            faces: Vec::new(),
            surface_gpu: vec![[0.0; 4]],
            stream_gpu: vec![[0.0; 4], [0.0; 4]],
            from: Keyframe {
                position: Vec::new(),
                normal: Vec::new(),
                field: Vec::new(),
            },
            to: Keyframe {
                position: Vec::new(),
                normal: Vec::new(),
                field: Vec::new(),
            },
        }
    }
}

pub fn poll_snapshot(mut evolution: ResMut<Evolution>, mut playback: ResMut<Playback>) {
    let receiver = evolution.receiver.lock().unwrap().take();
    let mut frame = None;
    let mut closed = false;
    if let Some(receiver) = receiver.as_ref() {
        loop {
            match receiver.try_recv() {
                Ok(WorkerMessage::Status(text)) => evolution.status = text,
                Ok(WorkerMessage::Frame(snapshot)) => {
                    if evolution.phase == Phase::Preparing {
                        evolution.phase = Phase::Running;
                    }
                    frame = Some(snapshot);
                    break;
                }
                Ok(WorkerMessage::Finished(text)) => {
                    evolution.phase = Phase::Done;
                    evolution.status = text;
                }
                Ok(WorkerMessage::Failed(text)) => {
                    evolution.phase = Phase::Failed;
                    evolution.status = text;
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    closed = true;
                    break;
                }
            }
        }
    }
    *evolution.receiver.lock().unwrap() = receiver;
    if closed
        && matches!(
            evolution.phase,
            Phase::Preparing | Phase::Running | Phase::Paused
        )
    {
        evolution.phase = Phase::Failed;
        evolution.status = "worker stopped unexpectedly".into();
    }
    let Some(snapshot) = frame else {
        return;
    };
    evolution.stats = Some(FrameStats {
        step: snapshot.step,
        time: snapshot.time,
        omega: snapshot.omega,
        mass: snapshot.mass,
        radius_min: snapshot.radius_min,
        radius_max: snapshot.radius_max,
        yielded_fraction: snapshot.yielded_fraction,
        max_slack: snapshot.max_slack,
        rho_s_total: snapshot.rho_s_total,
        max_displacement: snapshot.max_displacement,
    });
    adopt(&mut playback, *snapshot);
}

fn adopt(play: &mut Playback, snap: StepSnapshot) {
    let same = play.generation > 0 && play.from.position.len() == snap.position_to.len();
    let from = if same {
        mix_key(&play.from, &play.to, smoothstep(play.tau), play.fade)
    } else {
        Keyframe {
            position: snap.position_from.clone(),
            normal: snap.normal_from.clone(),
            field: snap.field.clone(),
        }
    };
    let to = Keyframe {
        position: snap.position_to,
        normal: snap.normal_to,
        field: snap.field,
    };
    let first = play.generation == 0;
    play.tau = 0.0;
    play.fade = if first { 1.0 } else { 0.0 };
    play.duration = (snap.wall_seconds as f32).max(0.25);
    play.slack_tol = snap.slack_tol;
    play.f_delta_p95 = snap.f_delta_p95;
    play.slack_p95 = snap.slack_p95;
    play.rho_s_p95 = snap.rho_s_p95;
    play.vertex_count = from.position.len() as u32;
    play.surface_gpu = pack_surface(&from, &to);
    play.stream_gpu = pack_stream(&snap.stream);
    if play.faces.len() != snap.faces.len() {
        play.faces = snap.faces;
    }
    play.from = from;
    play.to = to;
    play.generation += 1;
    play.dirty = true;
}

pub fn advance_playback(
    time: Res<Time>,
    evolution: Res<Evolution>,
    mut playback: ResMut<Playback>,
) {
    if playback.generation == 0 || evolution.phase == Phase::Paused {
        return;
    }
    let dt = time.delta_secs();
    let duration = playback.duration.max(1.0e-3);
    playback.tau = (playback.tau + dt / duration).min(1.0);
    playback.fade = (playback.fade + dt / 0.4).min(1.0);
}

pub fn sync_material(
    mut playback: ResMut<Playback>,
    mode: Res<VisMode>,
    time: Res<Time>,
    mut materials: ResMut<Assets<AsteroidMaterial>>,
    mut buffers: ResMut<Assets<ShaderBuffer>>,
    mut meshes: ResMut<Assets<Mesh>>,
    surface: Query<&Mesh3d, With<SurfaceView>>,
    material_handle: Query<&MeshMaterial3d<AsteroidMaterial>, With<SurfaceView>>,
) {
    let Ok(material_id) = material_handle.single() else {
        return;
    };
    if playback.dirty {
        if let Ok(mesh_id) = surface.single() {
            if let Some(mut mesh) = meshes.get_mut(&mesh_id.0) {
                let count = mesh
                    .attribute(Mesh::ATTRIBUTE_POSITION)
                    .map(|values| values.len())
                    .unwrap_or(0);
                let positions = playback.from.position.clone();
                let normals = playback.from.normal.clone();
                if count == positions.len() {
                    write_surface(&mut mesh, positions, normals);
                } else if !playback.faces.is_empty() {
                    *mesh = surface_mesh(positions, normals, &playback.faces);
                }
            }
        }
        if let Some(material) = materials.get(&material_id.0) {
            if let Some(mut buffer) = buffers.get_mut(&material.surface) {
                buffer.clear();
                buffer.extend(playback.surface_gpu.iter().copied());
            }
            if let Some(mut buffer) = buffers.get_mut(&material.stream) {
                buffer.clear();
                buffer.extend(playback.stream_gpu.iter().copied());
            }
        }
        playback.dirty = false;
    }
    if let Some(mut material) = materials.get_mut(&material_id.0) {
        material.params = PlaybackParams {
            tau: playback.tau,
            time: time.elapsed_secs(),
            fade: playback.fade,
            mode: mode.mode,
            f_delta_p95: playback.f_delta_p95,
            slack_p95: playback.slack_p95,
            rho_s_p95: playback.rho_s_p95,
            slack_tol: playback.slack_tol,
            vertex_count: playback.vertex_count,
            particle_count: PARTICLE_COUNT as u32,
            streamline_steps: STREAMLINE_STEPS as u32,
            has_field: u32::from(playback.generation > 0 && playback.vertex_count > 0),
        };
    }
}

fn mix3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    ]
}

fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len > 1.0e-8 {
        [v[0] / len, v[1] / len, v[2] / len]
    } else {
        [0.0, 1.0, 0.0]
    }
}

fn mix_sample(a: VertexSample, b: VertexSample, t: f32) -> VertexSample {
    VertexSample {
        f_delta: a.f_delta + (b.f_delta - a.f_delta) * t,
        movable: a.movable + (b.movable - a.movable) * t,
        v_sl: mix3(a.v_sl, b.v_sl, t),
        rho_s: a.rho_s + (b.rho_s - a.rho_s) * t,
        j_net: a.j_net + (b.j_net - a.j_net) * t,
        d_radial: a.d_radial + (b.d_radial - a.d_radial) * t,
        slack_max: a.slack_max + (b.slack_max - a.slack_max) * t,
        slack_avg: a.slack_avg + (b.slack_avg - a.slack_avg) * t,
    }
}

fn mix_key(from: &Keyframe, to: &Keyframe, position_t: f32, field_t: f32) -> Keyframe {
    let count = from.position.len().min(to.position.len());
    let mut position = Vec::with_capacity(count);
    let mut normal = Vec::with_capacity(count);
    let mut field = Vec::with_capacity(count);
    for i in 0..count {
        position.push(mix3(from.position[i], to.position[i], position_t));
        normal.push(normalize3(mix3(from.normal[i], to.normal[i], position_t)));
        field.push(mix_sample(from.field[i], to.field[i], field_t));
    }
    Keyframe {
        position,
        normal,
        field,
    }
}

fn pack_surface(from: &Keyframe, to: &Keyframe) -> Vec<[f32; 4]> {
    let count = from
        .position
        .len()
        .min(to.position.len())
        .min(from.field.len())
        .min(to.field.len());
    if count == 0 {
        return vec![[0.0; 4]];
    }
    let mut out = Vec::with_capacity(count * 8);
    for i in 0..count {
        let a = from.field[i];
        let b = to.field[i];
        let p0 = from.position[i];
        let p1 = to.position[i];
        let n0 = from.normal[i];
        let n1 = to.normal[i];
        out.push([p0[0], p0[1], p0[2], a.f_delta]);
        out.push([p1[0], p1[1], p1[2], b.f_delta]);
        out.push([n0[0], n0[1], n0[2], a.movable]);
        out.push([n1[0], n1[1], n1[2], b.movable]);
        out.push([a.v_sl[0], a.v_sl[1], a.v_sl[2], a.rho_s]);
        out.push([b.v_sl[0], b.v_sl[1], b.v_sl[2], b.rho_s]);
        out.push([a.slack_max, a.slack_avg, b.slack_max, b.slack_avg]);
        out.push([a.j_net, b.j_net, a.d_radial, b.d_radial]);
    }
    out
}

fn pack_stream(points: &[StreamPoint]) -> Vec<[f32; 4]> {
    if points.is_empty() {
        return vec![[0.0; 4], [0.0; 4]];
    }
    let mut out = Vec::with_capacity(points.len() * 2);
    for point in points {
        out.push([
            point.verts[0] as f32,
            point.verts[1] as f32,
            point.verts[2] as f32,
            point.bary[0],
        ]);
        out.push([point.bary[1], point.bary[2], 0.0, 0.0]);
    }
    out
}
