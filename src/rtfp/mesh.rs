//! The body surface on the GPU: a hardware acceleration structure or the software
//! BVH (`gpu::Accel`), in a frame centred on the bounding box and scaled to unit
//! bounding radius. A hit distance then carries a relative error of about `1e-7`
//! of the body size whatever the length unit (MS §3.4, §3.15.4).

use std::collections::HashMap;

use crate::common::error::SimError;
use crate::common::geometry::TriMesh;
use crate::common::math::DVec3;
use crate::gpu::{Accel, Job, Res, kernels};
use crate::rtfp::{chords, tree};

/// A first hit counts as the aimed-at surface within this fraction of the aimed
/// distance, plus the absolute floor, both in unit-radius coordinates.
const EXPECT_REL_TOL: f32 = 1.0e-3;
const EXPECT_ABS_TOL: f32 = 2.0e-6;
/// Hits nearer than this (unit-radius coordinates) belong to the launch point's own face.
pub(crate) const LAUNCH_OFFSET: f32 = 2.0e-6;
/// Crossings stored per ray; `rt_all` holds at most 64.
const MAX_HITS: u32 = 64;

#[derive(Clone, Copy, Debug)]
pub struct Frame {
    pub center: DVec3,
    pub scale: f64,
}

impl Frame {
    pub fn of(mesh: &TriMesh) -> Self {
        let first = mesh.vertices[0];
        let (lo, hi) = mesh
            .vertices
            .iter()
            .fold((first, first), |(lo, hi), v| (lo.min(*v), hi.max(*v)));
        let center = (lo + hi) * 0.5;
        let scale = mesh
            .vertices
            .iter()
            .map(|v| (*v - center).length())
            .fold(0.0_f64, f64::max);
        Self { center, scale }
    }

    pub fn to_unit(&self, x: DVec3) -> DVec3 {
        (x - self.center) / self.scale
    }
}

/// Rays packed for one launch: origin, t_min, direction, expected first hit.
pub struct RayBatch {
    frame: Frame,
    packed: Vec<[f32; 8]>,
}

impl RayBatch {
    pub fn len(&self) -> usize {
        self.packed.len()
    }

    pub fn is_empty(&self) -> bool {
        self.packed.is_empty()
    }

    /// Queue a ray. `dir` is a unit vector. `expect` is the distance at which the first
    /// crossing must occur for the ray to be reported; `None` reports every crossing.
    pub fn push(&mut self, origin: DVec3, dir: DVec3, expect: Option<f64>) {
        let o = self.frame.to_unit(origin);
        let expect = expect.map_or(0.0, |d| (d / self.frame.scale) as f32);
        self.packed.push([
            o.x as f32,
            o.y as f32,
            o.z as f32,
            LAUNCH_OFFSET,
            dir.x as f32,
            dir.y as f32,
            dir.z as f32,
            expect,
        ]);
    }
}

/// Crossing distances of a launch, ray by ray, nearest first. Each crossing also
/// carries the orientation sign `s = sign(dir·n̂)` of the crossed face (MS §3.6):
/// −1 entering the solid, +1 leaving it.
pub struct Hits {
    scale: f64,
    offsets: Vec<u32>,
    t: Vec<f32>,
    s: Vec<f32>,
}

impl Hits {
    pub fn len(&self) -> usize {
        self.offsets.len() - 1
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Distances along ray `i` in world units, appended to `out`.
    pub fn distances_into(&self, i: usize, out: &mut Vec<f64>) {
        let range = self.offsets[i] as usize..self.offsets[i + 1] as usize;
        out.extend(self.t[range].iter().map(|t| f64::from(*t) * self.scale));
    }

    /// `(distance, orientation sign)` pairs along ray `i` in world units,
    /// appended to `out` (MS §3.6, the orientation rule).
    pub fn crossings_into(&self, i: usize, out: &mut Vec<(f64, f32)>) {
        let range = self.offsets[i] as usize..self.offsets[i + 1] as usize;
        out.extend(
            self.t[range.clone()]
                .iter()
                .zip(&self.s[range])
                .map(|(t, s)| (f64::from(*t) * self.scale, *s)),
        );
    }

    pub fn count(&self, i: usize) -> usize {
        (self.offsets[i + 1] - self.offsets[i]) as usize
    }
}

/// A closed triangle mesh resident on the GPU for ray queries and for the
/// closed-form uniform field (`rtfp/shader/uniform.wesl`).
pub struct BodyMesh {
    pub(crate) accel: Accel,
    /// Face records of `uniform.wesl`, 24 floats each, unit frame, in tree order.
    pub(crate) faces: Res,
    /// Face tree of the uniform field (`rtfp/tree.rs`).
    pub(crate) nodes: Res,
    pub(crate) node_count: usize,
    /// Tree slot of each mesh face.
    slot: Vec<u32>,
    /// Face index by the bit pattern of its centroid `(a + b + c)/3`.
    centroids: HashMap<[u64; 3], u32>,
    pub frame: Frame,
}

impl std::fmt::Debug for BodyMesh {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BodyMesh")
            .field("frame", &self.frame)
            .finish()
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct TraceParams {
    ray_count: u32,
    max_hits: u32,
    rel_tol: f32,
    abs_tol: f32,
}

impl BodyMesh {
    pub fn build(mesh: &TriMesh) -> Result<Self, SimError> {
        if mesh.vertices.is_empty() || mesh.faces.is_empty() {
            return Err(SimError::geometry("empty mesh"));
        }
        let frame = Frame::of(mesh);
        let verts: Vec<f32> = mesh
            .vertices
            .iter()
            .flat_map(|v| frame.to_unit(*v).as_vec3().to_array())
            .collect();
        let tris: Vec<u32> = mesh.faces.iter().flatten().copied().collect();
        let accel = Accel::new(verts, tris).map_err(SimError::gpu)?;
        let unit: Vec<[DVec3; 3]> = (0..mesh.faces.len())
            .map(|f| mesh.triangle(f).map(|p| frame.to_unit(p)))
            .collect();
        let (tree, order) = tree::build(&unit);
        let mut slot = vec![0u32; order.len()];
        for (s, &f) in order.iter().enumerate() {
            slot[f as usize] = s as u32;
        }
        let centroids = (0..mesh.faces.len())
            .map(|f| {
                let [a, b, c] = mesh.triangle(f);
                (
                    ((a + b + c) * (1.0 / 3.0)).to_array().map(f64::to_bits),
                    f as u32,
                )
            })
            .collect();
        let mut records = Vec::with_capacity(mesh.faces.len() * 24);
        for &f in &order {
            let v = unit[f as usize];
            let n = (v[1] - v[0]).cross(v[2] - v[0]).normalize_or_zero();
            let mut rec = [0.0f32; 24];
            for (i, p) in v.iter().enumerate() {
                rec[3 * i..3 * i + 3].copy_from_slice(&p.as_vec3().to_array());
            }
            rec[9..12].copy_from_slice(&n.as_vec3().to_array());
            for e in 0..3 {
                let d = v[(e + 1) % 3] - v[e];
                let m = d.normalize_or_zero().cross(n);
                rec[12 + 3 * e..15 + 3 * e].copy_from_slice(&m.as_vec3().to_array());
                rec[21 + e] = d.length() as f32;
            }
            records.extend_from_slice(&rec);
        }
        let faces = Res::from_slice(&records).map_err(SimError::gpu)?;
        let nodes = Res::from_slice(&tree).map_err(SimError::gpu)?;
        Ok(Self {
            accel,
            faces,
            nodes,
            node_count: tree.len(),
            slot,
            centroids,
            frame,
        })
    }

    /// The tree slot of the face whose interior holds `x`: exact for a face
    /// centre, else a search.
    pub fn face_of(&self, mesh: &TriMesh, x: DVec3) -> Option<u32> {
        self.centroids
            .get(&x.to_array().map(f64::to_bits))
            .copied()
            .or_else(|| crate::rtfp::reference::werner::face_containing(mesh, x).map(|f| f as u32))
            .map(|f| self.slot[f as usize])
    }

    /// Bounding-sphere radius of the mesh.
    pub fn scale(&self) -> f64 {
        self.frame.scale
    }

    pub fn new_batch(&self) -> RayBatch {
        RayBatch {
            frame: self.frame,
            packed: Vec::new(),
        }
    }

    /// Every crossing of every ray of `batch`. With `first_only` a ray reports at most its first.
    pub fn trace(&self, batch: &RayBatch, first_only: bool) -> Result<Hits, SimError> {
        let n = batch.packed.len();
        let mut offsets = Vec::with_capacity(n + 1);
        offsets.push(0u32);
        let (mut t, mut s) = (Vec::new(), Vec::new());
        if n == 0 {
            return Ok(Hits {
                scale: self.frame.scale,
                offsets,
                t,
                s,
            });
        }
        let k = &kernels::TRACE;
        let mut counts_all = Vec::with_capacity(n);
        let mut slots_all = Vec::with_capacity(n * 2);
        let mut signs_all = Vec::with_capacity(n * 2);
        for chunk in batch
            .packed
            .chunks(k.wg[0] as usize * crate::gpu::MAX_GROUPS as usize)
        {
            let mut job = Job::new();
            let rays = job.input(chunk);
            let params = job.value(&TraceParams {
                ray_count: chunk.len() as u32,
                max_hits: MAX_HITS,
                rel_tol: EXPECT_REL_TOL,
                abs_tol: EXPECT_ABS_TOL,
            });
            let counts = job.zeros::<u32>(chunk.len());
            let hits = job.zeros::<f32>(chunk.len() * MAX_HITS as usize);
            let signs = job.zeros::<f32>(chunk.len() * MAX_HITS as usize);
            job.run(
                k,
                k.groups(chunk.len()),
                &[rays, params, counts, hits, signs, (&self.accel).into()],
            );
            let rc = job.read_all::<u32>(counts, chunk.len());
            let rh = job.read_all::<f32>(hits, chunk.len() * MAX_HITS as usize);
            let rs = job.read_all::<f32>(signs, chunk.len() * MAX_HITS as usize);
            let out = job.submit().map_err(SimError::gpu)?;
            counts_all.extend(out.get::<u32>(rc));
            slots_all.extend(out.get::<f32>(rh));
            signs_all.extend(out.get::<f32>(rs));
        }
        for (i, &c) in counts_all.iter().enumerate() {
            if c > MAX_HITS {
                return Err(SimError::geometry(format!(
                    "ray {i} crosses the surface more than {MAX_HITS} times"
                )));
            }
            let keep = if first_only { c.min(1) } else { c } as usize;
            let base = i * MAX_HITS as usize;
            t.extend_from_slice(&slots_all[base..base + keep]);
            s.extend_from_slice(&signs_all[base..base + keep]);
            offsets.push(t.len() as u32);
        }
        Ok(Hits {
            scale: self.frame.scale,
            offsets,
            t,
            s,
        })
    }

    /// Whether each point lies inside the solid: the majority parity of three rays.
    pub fn inside(&self, xs: &[DVec3]) -> Result<Vec<bool>, SimError> {
        const DIRS: [[f64; 3]; 3] = [
            [0.577_350_269_2, 0.577_350_269_2, 0.577_350_269_2],
            [-0.267_261_241_9, 0.534_522_483_8, 0.801_783_725_7],
            [0.742_781_352_7, -0.557_086_014_5, 0.371_390_676_4],
        ];
        let mut batch = self.new_batch();
        for x in xs {
            for d in DIRS {
                batch.push(*x, DVec3::from_array(d), None);
            }
        }
        let hits = self.trace(&batch, false)?;
        let mut xs_ = Vec::new();
        Ok((0..xs.len())
            .map(|p| {
                let votes = (0..3)
                    .filter(|&k| {
                        xs_.clear();
                        hits.crossings_into(3 * p + k, &mut xs_);
                        chords::starts_inside_oriented(&xs_, self.frame.scale)
                    })
                    .count();
                votes >= 2
            })
            .collect())
    }
}
