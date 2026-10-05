//! Wrap a coarse radial displacement onto a high-resolution mesh.
//!
//! Both meshes are star-shaped about one centre. At each coarse vertex
//!
//! ```text
//! ΔR(ω̂) = R_coarse(ω̂) − R_high(ω̂)
//! ```
//!
//! `R_high` is the ray hit of the detailed mesh. The field is the planar
//! interpolant on the coarse triangles, so it is exact at coarse vertices and
//! does not depend on either face count. A high-resolution vertex at direction
//! `ω̂_k` with radius `R_k` moves to `(R_k + ΔR(ω̂_k)) ω̂_k`. Its triangle list
//! is copied unchanged.

use serde::Serialize;

use super::error::ShError;
use super::math::Vec3;
use super::mesh::{Center, Mesh};

#[derive(Clone, Debug)]
pub struct FuseOptions {
    pub center: Center,
}

impl Default for FuseOptions {
    fn default() -> Self {
        Self {
            center: Center::Auto,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct FuseReport {
    pub high_vertices: usize,
    pub high_faces: usize,
    pub coarse_vertices: usize,
    pub coarse_faces: usize,
    pub output_vertices: usize,
    pub output_faces: usize,
    pub faces_preserved: bool,
    pub center: [f64; 3],
    pub delta_min: f64,
    pub delta_max: f64,
    pub delta_rms: f64,
    pub length_unit: String,
}

pub struct FuseOutput {
    pub mesh: Mesh,
    pub report: FuseReport,
}

pub fn fuse(high: &Mesh, coarse: &Mesh, options: &FuseOptions) -> Result<FuseOutput, ShError> {
    if high.faces.is_empty() || high.vertices.is_empty() {
        return Err(ShError::invalid_input(
            "high-resolution mesh has no triangles",
        ));
    }
    if coarse.faces.is_empty() || coarse.vertices.is_empty() {
        return Err(ShError::invalid_input("coarse mesh has no triangles"));
    }
    let center = options.center.resolve(high)?;
    let high_samples = high.radial_samples(center)?;
    let coarse_samples = coarse.radial_samples(center)?;
    let high_grid = TriGrid::build(&high.vertices, &high.faces)?;
    let mut node_delta = Vec::with_capacity(coarse.vertices.len());
    for i in 0..coarse.vertices.len() {
        let dir = Vec3::from_array(coarse_samples.dir[i]);
        let hit = high_grid.closest(&high.vertices, &high.faces, center, dir)?;
        let Some(hit) = hit else {
            return Err(ShError::geometry(format!(
                "high-resolution mesh does not cover coarse direction {i}"
            )));
        };
        node_delta.push(coarse_samples.radius[i] - hit.t);
    }
    let coarse_grid = TriGrid::build(&coarse.vertices, &coarse.faces)?;
    let mut vertices = Vec::with_capacity(high.vertices.len());
    let mut delta_min = f64::INFINITY;
    let mut delta_max = f64::NEG_INFINITY;
    let mut delta_sq = 0.0;
    for k in 0..high.vertices.len() {
        let dir = Vec3::from_array(high_samples.dir[k]);
        let hit = coarse_grid.closest(&coarse.vertices, &coarse.faces, center, dir)?;
        let Some(hit) = hit else {
            return Err(ShError::geometry(format!(
                "coarse mesh does not cover high-resolution direction {k}"
            )));
        };
        let delta = hit.w[0] * node_delta[hit.face[0] as usize]
            + hit.w[1] * node_delta[hit.face[1] as usize]
            + hit.w[2] * node_delta[hit.face[2] as usize];
        if !delta.is_finite() {
            return Err(ShError::geometry(format!(
                "interpolated displacement at vertex {k} is not finite"
            )));
        }
        let radius = high_samples.radius[k] + delta;
        if !(radius.is_finite() && radius > 0.0) {
            return Err(ShError::geometry(format!(
                "vertex {k} radius {radius} is not positive after adding ΔR {delta}"
            )));
        }
        delta_min = delta_min.min(delta);
        delta_max = delta_max.max(delta);
        delta_sq += delta * delta;
        let p = Vec3::from_array(center) + dir * radius;
        vertices.push(p.to_array());
    }
    let mesh = Mesh {
        vertices,
        faces: high.faces.clone(),
        length_unit: high.length_unit.clone(),
    };
    if mesh.faces != high.faces {
        return Err(ShError::geometry(
            "internal error: fused faces differ from the high-resolution mesh",
        ));
    }
    let n = high.vertices.len() as f64;
    Ok(FuseOutput {
        report: FuseReport {
            high_vertices: high.vertex_count(),
            high_faces: high.face_count(),
            coarse_vertices: coarse.vertex_count(),
            coarse_faces: coarse.face_count(),
            output_vertices: mesh.vertex_count(),
            output_faces: mesh.face_count(),
            faces_preserved: true,
            center,
            delta_min,
            delta_max,
            delta_rms: (delta_sq / n).sqrt(),
            length_unit: high.length_unit.clone(),
        },
        mesh,
    })
}

struct Hit {
    t: f64,
    face: [u32; 3],
    w: [f64; 3],
}

struct TriGrid {
    min: Vec3,
    cell: Vec3,
    n: [i32; 3],
    buckets: Vec<Vec<u32>>,
}

impl TriGrid {
    fn build(vertices: &[[f64; 3]], faces: &[[u32; 3]]) -> Result<Self, ShError> {
        let mut lo = Vec3::from_array(vertices[0]);
        let mut hi = lo;
        for v in vertices {
            let p = Vec3::from_array(*v);
            if !p.x.is_finite() || !p.y.is_finite() || !p.z.is_finite() {
                return Err(ShError::geometry("mesh vertex is not finite"));
            }
            lo.x = lo.x.min(p.x);
            lo.y = lo.y.min(p.y);
            lo.z = lo.z.min(p.z);
            hi.x = hi.x.max(p.x);
            hi.y = hi.y.max(p.y);
            hi.z = hi.z.max(p.z);
        }
        let ext = hi - lo;
        let scale = ext.x.max(ext.y).max(ext.z).max(1.0);
        let pad = scale * 1e-8 + 1e-8;
        lo = Vec3::new(lo.x - pad, lo.y - pad, lo.z - pad);
        hi = Vec3::new(hi.x + pad, hi.y + pad, hi.z + pad);
        let ext = hi - lo;
        let target = (faces.len() as f64).cbrt().clamp(4.0, 48.0);
        let n = [
            ((ext.x / scale * target).ceil() as i32).clamp(1, 48),
            ((ext.y / scale * target).ceil() as i32).clamp(1, 48),
            ((ext.z / scale * target).ceil() as i32).clamp(1, 48),
        ];
        let cell = Vec3::new(
            ext.x / n[0] as f64,
            ext.y / n[1] as f64,
            ext.z / n[2] as f64,
        );
        let mut buckets = vec![Vec::new(); (n[0] * n[1] * n[2]) as usize];
        for (f, tri) in faces.iter().enumerate() {
            let p = [
                Vec3::from_array(vertices[tri[0] as usize]),
                Vec3::from_array(vertices[tri[1] as usize]),
                Vec3::from_array(vertices[tri[2] as usize]),
            ];
            let mut bmin = p[0];
            let mut bmax = p[0];
            for q in &p[1..] {
                bmin.x = bmin.x.min(q.x);
                bmin.y = bmin.y.min(q.y);
                bmin.z = bmin.z.min(q.z);
                bmax.x = bmax.x.max(q.x);
                bmax.y = bmax.y.max(q.y);
                bmax.z = bmax.z.max(q.z);
            }
            let ix0 = cell_on(bmin.x, lo.x, cell.x, n[0], true);
            let iy0 = cell_on(bmin.y, lo.y, cell.y, n[1], true);
            let iz0 = cell_on(bmin.z, lo.z, cell.z, n[2], true);
            let ix1 = cell_on(bmax.x, lo.x, cell.x, n[0], false);
            let iy1 = cell_on(bmax.y, lo.y, cell.y, n[1], false);
            let iz1 = cell_on(bmax.z, lo.z, cell.z, n[2], false);
            for iz in iz0..=iz1 {
                for iy in iy0..=iy1 {
                    for ix in ix0..=ix1 {
                        buckets[bucket(n, ix, iy, iz)].push(f as u32);
                    }
                }
            }
        }
        Ok(Self {
            min: lo,
            cell,
            n,
            buckets,
        })
    }

    fn closest(
        &self,
        vertices: &[[f64; 3]],
        faces: &[[u32; 3]],
        center: [f64; 3],
        dir: Vec3,
    ) -> Result<Option<Hit>, ShError> {
        let origin = Vec3::from_array(center);
        if let Some(hit) = self.march(vertices, faces, origin, dir) {
            return Ok(Some(hit));
        }
        Ok(scan_all(vertices, faces, origin, dir))
    }

    fn march(
        &self,
        vertices: &[[f64; 3]],
        faces: &[[u32; 3]],
        origin: Vec3,
        dir: Vec3,
    ) -> Option<Hit> {
        let mut ix = ((origin.x - self.min.x) / self.cell.x).floor() as i32;
        let mut iy = ((origin.y - self.min.y) / self.cell.y).floor() as i32;
        let mut iz = ((origin.z - self.min.z) / self.cell.z).floor() as i32;
        if ix == self.n[0] {
            ix -= 1;
        }
        if iy == self.n[1] {
            iy -= 1;
        }
        if iz == self.n[2] {
            iz -= 1;
        }
        if ix < 0 || iy < 0 || iz < 0 || ix >= self.n[0] || iy >= self.n[1] || iz >= self.n[2] {
            return None;
        }
        let (step_x, mut t_max_x, t_delta_x) =
            axis_setup(origin.x, dir.x, self.min.x, self.cell.x, ix);
        let (step_y, mut t_max_y, t_delta_y) =
            axis_setup(origin.y, dir.y, self.min.y, self.cell.y, iy);
        let (step_z, mut t_max_z, t_delta_z) =
            axis_setup(origin.z, dir.z, self.min.z, self.cell.z, iz);
        let mut best: Option<Hit> = None;
        let limit = (self.n[0] + self.n[1] + self.n[2] + 8) as usize;
        for _ in 0..limit {
            for &f in &self.buckets[bucket(self.n, ix, iy, iz)] {
                let tri = faces[f as usize];
                let p0 = Vec3::from_array(vertices[tri[0] as usize]);
                let p1 = Vec3::from_array(vertices[tri[1] as usize]);
                let p2 = Vec3::from_array(vertices[tri[2] as usize]);
                if let Some((t, w)) = ray_triangle(origin, dir, p0, p1, p2) {
                    if best.as_ref().is_none_or(|h| t < h.t) {
                        best = Some(Hit { t, face: tri, w });
                    }
                }
            }
            let exit_t = t_max_x.min(t_max_y).min(t_max_z);
            if best.as_ref().is_some_and(|h| h.t <= exit_t) {
                break;
            }
            if t_max_x <= t_max_y && t_max_x <= t_max_z {
                ix += step_x;
                if ix < 0 || ix >= self.n[0] {
                    break;
                }
                t_max_x += t_delta_x;
            } else if t_max_y <= t_max_z {
                iy += step_y;
                if iy < 0 || iy >= self.n[1] {
                    break;
                }
                t_max_y += t_delta_y;
            } else {
                iz += step_z;
                if iz < 0 || iz >= self.n[2] {
                    break;
                }
                t_max_z += t_delta_z;
            }
        }
        best
    }
}

fn scan_all(vertices: &[[f64; 3]], faces: &[[u32; 3]], origin: Vec3, dir: Vec3) -> Option<Hit> {
    let mut best: Option<Hit> = None;
    for tri in faces {
        let p0 = Vec3::from_array(vertices[tri[0] as usize]);
        let p1 = Vec3::from_array(vertices[tri[1] as usize]);
        let p2 = Vec3::from_array(vertices[tri[2] as usize]);
        if let Some((t, w)) = ray_triangle(origin, dir, p0, p1, p2) {
            if best.as_ref().is_none_or(|h| t < h.t) {
                best = Some(Hit { t, face: *tri, w });
            }
        }
    }
    best
}

fn bucket(n: [i32; 3], ix: i32, iy: i32, iz: i32) -> usize {
    ((iz * n[1] + iy) * n[0] + ix) as usize
}

fn cell_on(x: f64, min: f64, cell: f64, n: i32, low: bool) -> i32 {
    let mut i = ((x - min) / cell).floor() as i32;
    if !low && (x - min) / cell - i as f64 == 0.0 && i > 0 {
        i -= 1;
    }
    i.clamp(0, n - 1)
}

fn axis_setup(origin: f64, dir: f64, min: f64, cell: f64, i: i32) -> (i32, f64, f64) {
    if dir > 0.0 {
        let step = 1;
        let delta = cell / dir;
        let mut tmax = (min + (i as f64 + 1.0) * cell - origin) / dir;
        if tmax < 1e-15 {
            tmax += delta;
        }
        (step, tmax, delta)
    } else if dir < 0.0 {
        let step = -1;
        let delta = cell / -dir;
        let mut tmax = (origin - (min + i as f64 * cell)) / -dir;
        if tmax < 1e-15 {
            tmax += delta;
        }
        (step, tmax, delta)
    } else {
        (0, f64::INFINITY, f64::INFINITY)
    }
}

/// Möller–Trumbore. `dir` is a unit vector. Returns `(t, barycentric)`.
fn ray_triangle(origin: Vec3, dir: Vec3, v0: Vec3, v1: Vec3, v2: Vec3) -> Option<(f64, [f64; 3])> {
    let e1 = v1 - v0;
    let e2 = v2 - v0;
    let p = dir.cross(e2);
    let det = e1.dot(p);
    if det.abs() < 1e-14 {
        return None;
    }
    let inv = 1.0 / det;
    let tvec = origin - v0;
    let u = tvec.dot(p) * inv;
    if !(-1e-7..=1.0 + 1e-7).contains(&u) {
        return None;
    }
    let q = tvec.cross(e1);
    let v = dir.dot(q) * inv;
    if v < -1e-7 || u + v > 1.0 + 1e-7 {
        return None;
    }
    let t = e2.dot(q) * inv;
    if !(t.is_finite() && t > 1e-12) {
        return None;
    }
    let mut w = [1.0 - u - v, u, v];
    for c in &mut w {
        if *c < 0.0 {
            *c = 0.0;
        }
    }
    let sum = w[0] + w[1] + w[2];
    if !(sum.is_finite() && sum > 0.0) {
        return None;
    }
    for c in &mut w {
        *c /= sum;
    }
    Some((t, w))
}
