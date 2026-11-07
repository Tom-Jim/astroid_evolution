//! Every step the viewer received, kept compactly so the timeline can return to any
//! of them. A frame stores the three vertex coordinates and the fourteen surface
//! channels, each quantized to 16 bits between its own extremes; the faces are shared
//! while the topology stays the same. Normals and
//! streamlines are rebuilt when a frame is shown. Over the memory budget, every other
//! stored frame is dropped (the newest is always kept), so coverage stays even.

use std::sync::Arc;

use bevy::prelude::*;

use crate::app::resources::FrameStats;
use crate::app::snapshot::{
    StepAudit, StepSnapshot, VertexSample, face_neighbors, smooth_normals, streamlines,
};

const CHANNELS: usize = 14;

#[cfg(not(target_arch = "wasm32"))]
const BUDGET: usize = 512 << 20;
#[cfg(target_arch = "wasm32")]
const BUDGET: usize = 256 << 20;

#[derive(serde::Serialize, serde::Deserialize)]
struct Quantized {
    lo: f32,
    hi: f32,
    values: Vec<u16>,
}

impl Quantized {
    fn new(values: impl Iterator<Item = f32> + Clone) -> Self {
        let (lo, hi) = values
            .clone()
            .filter(|v| v.is_finite())
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), v| {
                (lo.min(v), hi.max(v))
            });
        let (lo, hi) = if lo <= hi { (lo, hi) } else { (0.0, 0.0) };
        let span = hi - lo;
        let values = values
            .map(|v| {
                if span > 0.0 && v.is_finite() {
                    (((v - lo) / span) * 65535.0).round().clamp(0.0, 65535.0) as u16
                } else {
                    0
                }
            })
            .collect();
        Self { lo, hi, values }
    }

    fn get(&self, i: usize) -> f32 {
        self.lo + (self.hi - self.lo) * f32::from(self.values[i]) / 65535.0
    }
}

fn channels(s: &VertexSample) -> [f32; CHANNELS] {
    [
        s.f_delta,
        s.movable,
        s.v_sl[0],
        s.v_sl[1],
        s.v_sl[2],
        s.rho_s,
        s.j_net,
        s.d_normal,
        s.slack_max,
        s.slack_avg,
        s.sigma1,
        s.sigma_dir[0],
        s.sigma_dir[1],
        s.sigma_dir[2],
    ]
}

fn sample(c: [f32; CHANNELS]) -> VertexSample {
    VertexSample {
        f_delta: c[0],
        movable: c[1],
        v_sl: [c[2], c[3], c[4]],
        rho_s: c[5],
        j_net: c[6],
        d_normal: c[7],
        slack_max: c[8],
        slack_avg: c[9],
        sigma1: c[10],
        sigma_dir: [c[11], c[12], c[13]],
    }
}

struct Frame {
    stats: FrameStats,
    faces: Arc<Vec<[u32; 3]>>,
    /// Vertex rings of `faces`, built once per topology for the streamlines.
    neighbors: Arc<Vec<Vec<usize>>>,
    position: [Quantized; 3],
    field: Vec<Quantized>,
    bulk: f32,
    cores: Vec<[f32; 4]>,
    rho_range: [f32; 2],
    slack_tol: f32,
    f_delta_p95: f32,
    slack_p95: f32,
    rho_s_p95: f32,
    sigma_p95: f32,
    flow_active: bool,
    /// Runout paths, view units quantized to 16 bits on `[-RANGE, RANGE]`.
    runout: Vec<[u16; 4]>,
}

/// View-unit range of the quantized runout paths (bodies are scaled to radius 1).
const RANGE: f32 = 4.0;

impl Frame {
    /// Bytes owned by this frame alone (shared faces not counted).
    fn bytes(&self) -> usize {
        2 * self.position[0].values.len() * (3 + CHANNELS) + 16 * self.cores.len() + 8 * self.runout.len() + 256
    }
}

fn pack_point(p: [f32; 4]) -> [u16; 4] {
    let q = |v: f32| (((v / RANGE).clamp(-1.0, 1.0) * 0.5 + 0.5) * 65535.0).round() as u16;
    [q(p[0]), q(p[1]), q(p[2]), u16::from(p[3] > 0.5)]
}

fn unpack_point(p: [u16; 4]) -> [f32; 4] {
    let u = |v: u16| (f32::from(v) / 65535.0 * 2.0 - 1.0) * RANGE;
    [u(p[0]), u(p[1]), u(p[2]), f32::from(p[3])]
}

#[derive(Resource)]
pub struct History {
    frames: Vec<Frame>,
    bytes: usize,
    /// The frame on screen while the timeline is held back; `None` follows the run.
    pub view: Option<usize>,
}

impl Default for History {
    fn default() -> Self {
        Self {
            frames: Vec::new(),
            bytes: 0,
            view: None,
        }
    }
}

impl History {
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    pub fn clear(&mut self) {
        *self = Self::default();
    }

    pub fn stats(&self, index: usize) -> Option<FrameStats> {
        self.frames.get(index).map(|f| f.stats)
    }

    pub fn record(&mut self, snap: &StepSnapshot) {
        let last = self.frames.last();
        let (faces, neighbors) = match last {
            Some(f) if *f.faces == snap.faces => (f.faces.clone(), f.neighbors.clone()),
            _ => {
                let neighbors = face_neighbors(snap.position_to.len(), &snap.faces);
                self.bytes += 12 * snap.faces.len() + 8 * 6 * neighbors.len();
                (Arc::new(snap.faces.clone()), Arc::new(neighbors))
            }
        };
        let field = (0..CHANNELS)
            .map(|c| Quantized::new(snap.field.iter().map(move |s| channels(s)[c])))
            .collect();
        let frame = Frame {
            stats: FrameStats::of(snap),
            faces,
            neighbors,
            position: std::array::from_fn(|c| Quantized::new(snap.position_to.iter().map(move |p| p[c]))),
            field,
            bulk: snap.bulk,
            cores: snap.cores.clone(),
            rho_range: snap.rho_range,
            slack_tol: snap.slack_tol,
            f_delta_p95: snap.f_delta_p95,
            slack_p95: snap.slack_p95,
            rho_s_p95: snap.rho_s_p95,
            sigma_p95: snap.sigma_p95,
            flow_active: snap.flow_active,
            runout: snap.runout.iter().copied().map(pack_point).collect(),
        };
        self.bytes += frame.bytes();
        self.frames.push(frame);
        if self.bytes > BUDGET {
            self.thin();
        }
    }

    /// Keep frames 0, 2, 4, … and the newest; the held frame moves with its neighbours.
    fn thin(&mut self) {
        let newest = self.frames.len() - 1;
        let mut kept = Vec::with_capacity(self.frames.len() / 2 + 1);
        for (i, frame) in std::mem::take(&mut self.frames).into_iter().enumerate() {
            if i % 2 == 0 || i == newest {
                kept.push(frame);
            }
        }
        self.frames = kept;
        self.view = self.view.map(|v| (v / 2).min(self.frames.len() - 1));
        let mut faces = Vec::<*const Vec<[u32; 3]>>::new();
        self.bytes = 0;
        for f in &self.frames {
            self.bytes += f.bytes();
            if !faces.contains(&Arc::as_ptr(&f.faces)) {
                faces.push(Arc::as_ptr(&f.faces));
                self.bytes += 12 * f.faces.len() + 8 * 6 * f.neighbors.len();
            }
        }
    }
}

impl History {
    /// Move the oldest frame to the end of `dest` (playback of a cached run);
    /// `false` when none is left.
    pub fn move_first_to(&mut self, dest: &mut History) -> bool {
        if self.frames.is_empty() {
            return false;
        }
        let frame = self.frames.remove(0);
        if !dest.frames.last().is_some_and(|f| Arc::ptr_eq(&f.faces, &frame.faces)) {
            dest.bytes += 12 * frame.faces.len() + 8 * 6 * frame.neighbors.len();
        }
        dest.bytes += frame.bytes();
        dest.frames.push(frame);
        if dest.bytes > BUDGET {
            dest.thin();
        }
        true
    }
}

/// The stored timeline without the shared rings: what a cache file carries
/// (`app::cache`). Topologies are listed once; each frame names its own.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct Stored {
    topologies: Vec<Vec<[u32; 3]>>,
    frames: Vec<StoredFrame>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct StoredFrame {
    topology: u32,
    stats: FrameStats,
    position: [Quantized; 3],
    field: Vec<Quantized>,
    bulk: f32,
    cores: Vec<[f32; 4]>,
    rho_range: [f32; 2],
    slack_tol: f32,
    p95: [f32; 4],
    flow_active: bool,
    runout: Vec<[u16; 4]>,
}

impl Stored {
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }
}

impl History {
    /// The stored timeline, thinned evenly to at most `max_frames` (the newest is kept).
    pub fn to_stored(&self, max_frames: usize) -> Stored {
        let n = self.frames.len();
        let keep: Vec<usize> = if n <= max_frames.max(2) {
            (0..n).collect()
        } else {
            let mut k: Vec<usize> = (0..max_frames).map(|i| i * (n - 1) / (max_frames - 1)).collect();
            k.dedup();
            k
        };
        let mut topologies: Vec<Vec<[u32; 3]>> = Vec::new();
        let mut owners: Vec<*const Vec<[u32; 3]>> = Vec::new();
        let frames = keep
            .into_iter()
            .map(|i| {
                let f = &self.frames[i];
                let ptr = Arc::as_ptr(&f.faces);
                let topology = match owners.iter().position(|p| *p == ptr) {
                    Some(t) => t,
                    None => {
                        owners.push(ptr);
                        topologies.push(f.faces.to_vec());
                        topologies.len() - 1
                    }
                };
                let copy = |q: &Quantized| Quantized { lo: q.lo, hi: q.hi, values: q.values.clone() };
                StoredFrame {
                    topology: topology as u32,
                    stats: f.stats,
                    position: std::array::from_fn(|c| copy(&f.position[c])),
                    field: f.field.iter().map(copy).collect(),
                    bulk: f.bulk,
                    cores: f.cores.clone(),
                    rho_range: f.rho_range,
                    slack_tol: f.slack_tol,
                    p95: [f.f_delta_p95, f.slack_p95, f.rho_s_p95, f.sigma_p95],
                    flow_active: f.flow_active,
                    runout: f.runout.clone(),
                }
            })
            .collect();
        Stored { topologies, frames }
    }

    /// A timeline from its stored form; the rings are rebuilt once per topology.
    pub fn from_stored(stored: Stored) -> Result<Self, String> {
        let shared: Vec<(Arc<Vec<[u32; 3]>>, Arc<Vec<Vec<usize>>>)> = stored
            .topologies
            .into_iter()
            .map(|faces| {
                let nv = faces.iter().flatten().map(|v| *v as usize + 1).max().unwrap_or(0);
                let rings = face_neighbors(nv, &faces);
                (Arc::new(faces), Arc::new(rings))
            })
            .collect();
        let mut history = Self::default();
        for f in stored.frames {
            let (faces, neighbors) = shared
                .get(f.topology as usize)
                .cloned()
                .ok_or("cache: a frame names a missing topology")?;
            if f.field.len() != CHANNELS || f.position.iter().any(|p| p.values.len() != f.position[0].values.len()) {
                return Err("cache: a frame has the wrong channel layout".into());
            }
            let frame = Frame {
                stats: f.stats,
                faces,
                neighbors,
                position: f.position,
                field: f.field,
                bulk: f.bulk,
                cores: f.cores,
                rho_range: f.rho_range,
                slack_tol: f.slack_tol,
                f_delta_p95: f.p95[0],
                slack_p95: f.p95[1],
                rho_s_p95: f.p95[2],
                sigma_p95: f.p95[3],
                flow_active: f.flow_active,
                runout: f.runout,
            };
            history.bytes += frame.bytes();
            history.frames.push(frame);
        }
        Ok(history)
    }
}

impl History {
    /// Frame `index` as a still snapshot (both poses equal), ready for the playback.
    pub fn frame(&self, index: usize) -> Option<StepSnapshot> {
        let f = self.frames.get(index)?;
        let n = f.position[0].values.len();
        let position: Vec<[f32; 3]> =
            (0..n).map(|i| std::array::from_fn(|c| f.position[c].get(i))).collect();
        let normal = smooth_normals(&position, &f.faces);
        let field: Vec<VertexSample> = (0..n)
            .map(|i| sample(std::array::from_fn(|c| f.field[c].get(i))))
            .collect();
        let stream = streamlines(&position, &normal, &f.neighbors, &field, f.flow_active);
        let s = f.stats;
        Some(StepSnapshot {
            step: s.step,
            time: s.time,
            omega: s.omega,
            mass: s.mass,
            radius_min: s.radius_min,
            radius_max: s.radius_max,
            yielded_fraction: s.yielded_fraction,
            max_slack: s.max_slack,
            rho_s_total: s.rho_s_total,
            max_displacement: s.max_displacement,
            frame_applied: false,
            wall_seconds: 0.0,
            flow_active: f.flow_active,
            dt_bound: s.dt_bound,
            audit: StepAudit {
                eps_momentum: s.eps_momentum,
                eps_lambda: s.eps_lambda,
                eps_relax: s.eps_relax,
                sweeps: s.sweeps,
                tensile_fraction: s.tensile_fraction,
                inverse_ekman: s.inverse_ekman,
                mass_residual: s.mass_residual,
                mobile_area_fraction: s.mobile_area_fraction,
                dt: s.dt,
                dt_cfl: s.dt_cfl,
                runout_traced: s.runout_traced,
                runout_distance: s.runout_distance,
                runout_dlat: s.runout_dlat,
                runout_hops: s.runout_hops,
                ..StepAudit::default()
            },
            slack_tol: f.slack_tol,
            f_delta_p95: f.f_delta_p95,
            slack_p95: f.slack_p95,
            rho_s_p95: f.rho_s_p95,
            sigma_p95: f.sigma_p95,
            tension_max: s.tension_max,
            faces: f.faces.to_vec(),
            vertices_m: Vec::new(),
            position_from: position.clone(),
            position_to: position,
            normal_from: normal.clone(),
            normal_to: normal,
            field,
            stream,
            runout: f.runout.iter().copied().map(unpack_point).collect(),
            bulk: f.bulk,
            cores: f.cores.clone(),
            rho_range: f.rho_range,
        })
    }
}
