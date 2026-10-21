//! Loose-layer density `ρ_s`: mode C erosion/deposition (§2.8) and conservative
//! semi-Lagrangian transport on `S²` (§9.2), with the Jacobian correction.
//!
//! `ρ_s` (kg m⁻²) lives on the surface vertices. The parameter velocity of a surface
//! material point is `c = v_S / R`, where `v_S` is the tangential part of the velocity
//! relative to the rigid core, restricted to the active set (`χ_A`).

use nalgebra::{Matrix3, Vector3};
use rayon::prelude::*;

use crate::common::math::DVec3;

use super::body::Topology;
use super::params::ModeCParams;

/// Surface parameter velocity and its first derivatives.
#[derive(Clone, Debug)]
pub struct SurfaceFlow {
    /// `c = χ_A v_S / R` per vertex (tangent to the unit sphere).
    pub c: Vec<DVec3>,
    /// Surface divergence `∇_{S²}·c`.
    pub div: Vec<f64>,
    /// `‖∇_{S²} c‖_∞`.
    pub grad_max: f64,
    /// Determinant of the tangential velocity gradient at each vertex.
    pub tangent_det: Vec<f64>,
}

fn to_vec(v: DVec3) -> Vector3<f64> {
    Vector3::new(v.x, v.y, v.z)
}

pub fn surface_flow(
    topo: &Topology,
    v_sl: &[DVec3],
    active: &[bool],
    radius: &[f64],
) -> SurfaceFlow {
    let nv = topo.n_vertices();
    let c: Vec<DVec3> = (0..nv)
        .map(|i| {
            if !active[i] {
                return DVec3::ZERO;
            }
            let w = topo.dirs[i];
            let vt = v_sl[i] - w * v_sl[i].dot(w);
            vt * (1.0 / radius[i])
        })
        .collect();
    let mut div = vec![0.0; nv];
    let mut tangent_det = vec![0.0; nv];
    let mut grad_max = 0.0_f64;
    for i in 0..nv {
        let w = to_vec(topo.dirs[i]);
        let proj = Matrix3::identity() - w * w.transpose();
        let mut ata = Matrix3::zeros();
        let mut rhs = Matrix3::zeros();
        for &j in &topo.vertex_neighbors[i] {
            let delta = proj * (to_vec(topo.dirs[j]) - w);
            let dc = proj * (to_vec(c[j]) - to_vec(c[i]));
            ata += delta * delta.transpose();
            rhs += dc * delta.transpose();
        }
        ata += w * w.transpose() * ata.trace().max(1e-300);
        if let Some(inv) = ata.try_inverse() {
            let g = rhs * inv;
            div[i] = (proj * g).trace();
            let t = topo.dirs[i];
            let e1v = t.cross(DVec3::new(1.0, 0.0, 0.0));
            let e1 = if e1v.length_squared() > 1e-20 {
                e1v.try_normalize().unwrap_or(DVec3::new(1.0, 0.0, 0.0))
            } else {
                t.cross(DVec3::new(0.0, 1.0, 0.0))
                    .try_normalize()
                    .unwrap_or(DVec3::new(1.0, 0.0, 0.0))
            };
            let e2 = t
                .cross(e1)
                .try_normalize()
                .unwrap_or(DVec3::new(0.0, 0.0, 1.0));
            let g1 = DVec3::new(g[(0, 0)], g[(1, 0)], g[(2, 0)]);
            let g2 = DVec3::new(g[(0, 1)], g[(1, 1)], g[(2, 1)]);
            tangent_det[i] = e1.dot(g1) * e2.dot(g2) - e1.dot(g2) * e2.dot(g1);
            grad_max = grad_max.max(g.norm());
        }
    }
    SurfaceFlow {
        c,
        div,
        grad_max,
        tangent_det,
    }
}

/// Barycentric coordinates of the direction `p` in the spherical triangle `abc`.
fn barycentric(a: DVec3, b: DVec3, c: DVec3, p: DVec3) -> Option<[f64; 3]> {
    let m = Matrix3::new(a.x, b.x, c.x, a.y, b.y, c.y, a.z, b.z, c.z);
    let inv = m.try_inverse()?;
    let l = inv * to_vec(p);
    let s = l.sum();
    if s <= 0.0 {
        return None;
    }
    Some([l[0] / s, l[1] / s, l[2] / s])
}

/// Cube bins of unit directions. Nearest neighbour is the maximum dot product.
/// An exact tie keeps the later index, matching `Iterator::max_by`.
pub struct DirIndex {
    g: usize,
    cell: f64,
    start: Vec<u32>,
    items: Vec<u32>,
}

impl DirIndex {
    pub fn build(dirs: &[DVec3]) -> Self {
        let g = {
            let guess = (dirs.len() as f64).sqrt() * 0.5;
            guess.round() as usize
        }
        .clamp(8, 48);
        let ncell = g * g * g;
        let mut counts = vec![0u32; ncell];
        let mut cells = Vec::with_capacity(dirs.len());
        for d in dirs {
            let c = cell_of(*d, g);
            counts[c] += 1;
            cells.push(c);
        }
        let mut start = vec![0u32; ncell + 1];
        for i in 0..ncell {
            start[i + 1] = start[i] + counts[i];
        }
        let mut cursor = start.clone();
        let mut items = vec![0u32; dirs.len()];
        for (i, c) in cells.into_iter().enumerate() {
            let slot = cursor[c] as usize;
            items[slot] = i as u32;
            cursor[c] += 1;
        }
        Self {
            g,
            cell: 2.0 / g as f64,
            start,
            items,
        }
    }

    pub fn nearest(&self, dirs: &[DVec3], p: DVec3) -> usize {
        if dirs.is_empty() {
            return 0;
        }
        let g = self.g;
        let (ix, iy, iz) = cell_xyz(p, g);
        let mut best_i = 0usize;
        let mut best_dot = f64::NEG_INFINITY;
        for rad in 0..g {
            let x0 = ix.saturating_sub(rad);
            let x1 = (ix + rad).min(g - 1);
            let y0 = iy.saturating_sub(rad);
            let y1 = (iy + rad).min(g - 1);
            let z0 = iz.saturating_sub(rad);
            let z1 = (iz + rad).min(g - 1);
            for x in x0..=x1 {
                for y in y0..=y1 {
                    for z in z0..=z1 {
                        let shell = rad == 0
                            || x == x0
                            || x == x1
                            || y == y0
                            || y == y1
                            || z == z0
                            || z == z1;
                        if !shell {
                            continue;
                        }
                        let c = (x * g + y) * g + z;
                        for &idx in &self.items[self.start[c] as usize..self.start[c + 1] as usize]
                        {
                            let i = idx as usize;
                            let dot = dirs[i].dot(p);
                            match dot.partial_cmp(&best_dot) {
                                Some(std::cmp::Ordering::Greater) => {
                                    best_dot = dot;
                                    best_i = i;
                                }
                                Some(std::cmp::Ordering::Equal) if i > best_i => best_i = i,
                                Some(_) => {}
                                None => panic!("direction search hit a non-finite dot product"),
                            }
                        }
                    }
                }
            }
            if best_dot.is_finite() {
                let lo = [
                    edge(x0, self.cell),
                    edge(y0, self.cell),
                    edge(z0, self.cell),
                ];
                let hi = [
                    edge(x1 + 1, self.cell),
                    edge(y1 + 1, self.cell),
                    edge(z1 + 1, self.cell),
                ];
                let chord2 = (2.0 - 2.0 * best_dot).max(0.0);
                if chord2 < dist_out_sq(p, lo, hi) * (1.0 - 1e-12) {
                    break;
                }
            }
        }
        best_i
    }

    /// Vertices that can lie inside the spherical cap of angular radius `alpha`
    /// about `origin`. The set is a superset: every unit direction with angle
    /// strictly below `alpha` is present, and some directions outside it may be too.
    #[cfg_attr(
        all(any(target_os = "macos", target_arch = "wasm32"), not(test)),
        allow(dead_code)
    )]
    fn collect_within(&self, origin: DVec3, alpha: f64, out: &mut Vec<usize>) {
        out.clear();
        if !alpha.is_finite() || alpha <= 0.0 {
            return;
        }
        if alpha >= std::f64::consts::PI {
            out.extend(self.items.iter().map(|i| *i as usize));
            return;
        }
        let b = cap_bounds(origin, alpha, self.g);
        let g = self.g;
        for x in b[0]..=b[1] {
            for y in b[2]..=b[3] {
                for z in b[4]..=b[5] {
                    let c = (x as usize * g + y as usize) * g + z as usize;
                    for &idx in &self.items[self.start[c] as usize..self.start[c + 1] as usize] {
                        out.push(idx as usize);
                    }
                }
            }
        }
    }
}

/// Inclusive cell box of the spherical cap, with one extra cell of slack.
///
/// The same box is uploaded to the Metal deposition kernels. An inverted
/// interval (`lo > hi`) is empty.
fn cap_bounds(origin: DVec3, alpha: f64, g: usize) -> [u32; 6] {
    if !alpha.is_finite() || alpha <= 0.0 || g == 0 {
        return [1, 0, 1, 0, 1, 0];
    }
    // One cell of slack covers the rounding gap between `acos` and the
    // Cartesian bounds of the cap, so a point the angle test would keep
    // cannot fall outside the visited bins.
    let slop = (alpha + 1.0e-6).min(std::f64::consts::PI);
    let comps = [origin.x, origin.y, origin.z];
    let mut out = [0u32; 6];
    for axis in 0..3 {
        let bearing = comps[axis].clamp(-1.0, 1.0).acos();
        let c_lo = (bearing + slop).min(std::f64::consts::PI).cos() - 1.0e-7;
        let c_hi = (bearing - slop).max(0.0).cos() + 1.0e-7;
        let i_lo = cell_axis(c_lo.clamp(-1.0, 1.0), g);
        let i_hi = cell_axis(c_hi.clamp(-1.0, 1.0), g);
        out[axis * 2] = i_lo.saturating_sub(1) as u32;
        out[axis * 2 + 1] = (i_hi + 1).min(g - 1) as u32;
    }
    out
}

fn edge(i: usize, cell: f64) -> f64 {
    -1.0 + i as f64 * cell
}

fn cell_xyz(p: DVec3, g: usize) -> (usize, usize, usize) {
    (cell_axis(p.x, g), cell_axis(p.y, g), cell_axis(p.z, g))
}

fn cell_axis(t: f64, g: usize) -> usize {
    let i = ((t + 1.0) * 0.5 * g as f64).floor() as isize;
    i.clamp(0, g as isize - 1) as usize
}

fn cell_of(p: DVec3, g: usize) -> usize {
    let (x, y, z) = cell_xyz(p, g);
    (x * g + y) * g + z
}

/// Distance squared from `p` to the nearest point of the domain cube that lies outside `lo..hi`.
fn dist_out_sq(p: DVec3, lo: [f64; 3], hi: [f64; 3]) -> f64 {
    const LIM: f64 = 1.0 - 1e-15;
    if lo[0] <= -LIM
        && hi[0] >= LIM
        && lo[1] <= -LIM
        && hi[1] >= LIM
        && lo[2] <= -LIM
        && hi[2] >= LIM
    {
        return f64::INFINITY;
    }
    let px = [p.x, p.y, p.z];
    let mut gap = f64::INFINITY;
    for a in 0..3 {
        if lo[a] > -1.0 {
            gap = gap.min(px[a] - lo[a]);
        }
        if hi[a] < 1.0 {
            gap = gap.min(hi[a] - px[a]);
        }
    }
    if gap <= 0.0 { 0.0 } else { gap * gap }
}

/// Nearest vertex and the ring of incident faces plus their neighbours.
///
/// `usize::MAX` is an empty neighbour slot. The ring is the same set
/// `interp_vertex_field` and the post-realignment resample both walk.
fn face_ring(topo: &Topology, index: &DirIndex, p: DVec3) -> (usize, Vec<usize>) {
    let nearest = index.nearest(&topo.dirs, p);
    let mut seen = topo.vertex_faces.get(nearest).cloned().unwrap_or_default();
    let first = seen.clone();
    for f in first {
        let Some(neighbors) = topo.face_neighbors.get(f) else {
            continue;
        };
        for &g in neighbors {
            if g != usize::MAX && !seen.contains(&g) {
                seen.push(g);
            }
        }
    }
    (nearest, seen)
}

fn weights_on(topo: &Topology, face: usize, p: DVec3) -> Option<([usize; 3], [f64; 3])> {
    let [i, j, k] = topo.faces.get(face)?.map(|v| v as usize);
    let w = barycentric(topo.dirs[i], topo.dirs[j], topo.dirs[k], p)?;
    if w.iter().all(|value| *value >= -1.0e-12) {
        Some(([i, j, k], w))
    } else {
        None
    }
}

/// Containing triangle of `p`, or the nearest vertex with weight 1.
pub(crate) fn locate_vertex(topo: &Topology, index: &DirIndex, p: DVec3) -> ([usize; 3], [f64; 3]) {
    let (nearest, faces) = face_ring(topo, index, p);
    for f in faces {
        if let Some(hit) = weights_on(topo, f, p) {
            return hit;
        }
    }
    ([nearest, nearest, nearest], [1.0, 0.0, 0.0])
}

/// Face that contains `p`, or the first face incident to the nearest vertex.
pub(crate) fn locate_face(topo: &Topology, index: &DirIndex, p: DVec3) -> usize {
    let (nearest, faces) = face_ring(topo, index, p);
    for f in faces {
        if weights_on(topo, f, p).is_some() {
            return f;
        }
    }
    topo.vertex_faces
        .get(nearest)
        .and_then(|faces| faces.first())
        .copied()
        .unwrap_or(0)
}

/// Barycentric interpolation of a vertex field at the direction `p`.
///
/// `index` is built once for `topo.dirs`. The face ring around the nearest vertex
/// is unchanged: the triangle that contains `p`, then its neighbours.
pub fn interp_vertex_field(topo: &Topology, index: &DirIndex, field: &[f64], p: DVec3) -> f64 {
    let (ids, w) = locate_vertex(topo, index, p);
    w[0] * field[ids[0]] + w[1] * field[ids[1]] + w[2] * field[ids[2]]
}

/// One semi-Lagrangian step of the vertex field `field` (`ρ_s`) with the area densities
/// `j_old`, `j_new` (area per solid angle before and after the geometric update).
///
/// `j_new,a ρ_a = J_a j_old(ω_d) ρ(ω_d)` with the tangent Jacobian, then one global
/// multiplier that restores `Σ j ρ`. The returned scalar is the relative size of that multiplier.
pub fn transport(
    topo: &Topology,
    flow: &SurfaceFlow,
    field: &[f64],
    j_old: &[f64],
    j_new: &[f64],
    dt: f64,
) -> Result<(Vec<f64>, f64), String> {
    let nv = topo.n_vertices();
    let mut out = vec![0.0; nv];
    for a in 0..nv {
        let w = topo.dirs[a];
        let foot = (w - flow.c[a] * dt).try_normalize().unwrap_or(w);
        let mut value = field[a] * j_old[a];
        // Candidate triangles: the rings of faces around `a`.
        let mut seen: Vec<usize> = topo.vertex_faces[a].clone();
        let first = seen.clone();
        for f in first {
            for &g in &topo.face_neighbors[f] {
                if !seen.contains(&g) {
                    seen.push(g);
                }
            }
        }
        let second = seen.clone();
        for f in second {
            for &g in &topo.face_neighbors[f] {
                if !seen.contains(&g) {
                    seen.push(g);
                }
            }
        }
        let mut found = false;
        let deposit = |f: usize, value: &mut f64, found: &mut bool| {
            let t = topo.faces[f];
            let [i, j, k] = t.map(|x| x as usize);
            if let Some(l) = barycentric(topo.dirs[i], topo.dirs[j], topo.dirs[k], foot) {
                if l.iter().all(|x| *x >= -1e-12) {
                    *value = l[0] * field[i] * j_old[i]
                        + l[1] * field[j] * j_old[j]
                        + l[2] * field[k] * j_old[k];
                    *found = true;
                }
            }
        };
        for f in seen {
            deposit(f, &mut value, &mut found);
            if found {
                break;
            }
        }
        if !found {
            // The two-ring search is only a locality optimisation. A closed spherical
            // mesh still covers the foot; scan the rest before declaring the step invalid.
            for f in 0..topo.n_faces() {
                deposit(f, &mut value, &mut found);
                if found {
                    break;
                }
            }
        }
        if !found {
            return Err(format!(
                "characteristic foot point not covered at vertex {a}"
            ));
        }
        let jac = (1.0 - dt * flow.div[a] + dt * dt * flow.tangent_det[a]).max(1e-12);
        out[a] = (jac * value / j_new[a]).max(0.0);
    }
    // The tangent Jacobian is the conservation factor of §9.2. It is a truncation of
    // det(Dω_d), so a single global multiplier restores Σ j ρ exactly. The returned
    // number is the relative size of that correction.
    let before: f64 = (0..nv)
        .map(|i| j_old[i] * topo.vertex_solid[i] * field[i])
        .sum();
    let after: f64 = (0..nv)
        .map(|i| j_new[i] * topo.vertex_solid[i] * out[i])
        .sum();
    let mut fix = 0.0;
    if after > 0.0 && before > 0.0 {
        let s = before / after;
        fix = (s - 1.0).abs();
        for v in &mut out {
            *v *= s;
        }
    }
    Ok((out, fix))
}

/// Erosion and deposition rates per vertex (kg m⁻² s⁻¹).
#[derive(Clone, Debug)]
pub struct ModeC {
    pub j_ero: Vec<f64>,
    pub j_dep: Vec<f64>,
    /// `max_i [F_δ]_+` over active vertices.
    pub max_excess: f64,
    pub eroded_rate: f64,
}

/// `J_ero = χ_A k_e ρ_s [F_δ]_+ / σ₀`; `J_dep(a) = ∫ K(a, a') J_ero(a') dΓ(a')` with
/// `∫ K(a, a') dΓ(a) = 1` for every source, so that `∫ J_dep dΓ = ∫ J_ero dΓ`.
///
/// Erosion is a per-vertex scale. Deposition is the normalised angular kernel
/// on the GPU: the Gaussian weight is native single precision, the cap sum is
/// double-single, and the cap box is the one from
/// [`DirIndex::collect_within`](DirIndex::collect_within).
pub fn mode_c(
    topo: &Topology,
    area: &[f64],
    rho_s: &[f64],
    movable: &[bool],
    f_delta: &[f64],
    prm: &ModeCParams,
) -> ModeC {
    let (j_ero, max_excess, eroded) = erosion(area, rho_s, movable, f_delta, prm);
    let j_dep = deposit(topo, area, &j_ero, prm);
    ModeC {
        j_ero,
        j_dep,
        max_excess,
        eroded_rate: eroded,
    }
}

/// Host reference. Tests compare the GPU field with this and with the dense loop.
#[cfg_attr(
    all(any(target_os = "macos", target_arch = "wasm32"), not(test)),
    allow(dead_code)
)]
fn mode_c_cpu(
    topo: &Topology,
    area: &[f64],
    rho_s: &[f64],
    movable: &[bool],
    f_delta: &[f64],
    prm: &ModeCParams,
) -> ModeC {
    let (j_ero, max_excess, eroded) = erosion(area, rho_s, movable, f_delta, prm);
    let j_dep = deposit_cpu(topo, area, &j_ero, prm);
    ModeC {
        j_ero,
        j_dep,
        max_excess,
        eroded_rate: eroded,
    }
}

fn erosion(
    area: &[f64],
    rho_s: &[f64],
    movable: &[bool],
    f_delta: &[f64],
    prm: &ModeCParams,
) -> (Vec<f64>, f64, f64) {
    let nv = area.len();
    let mut j_ero = vec![0.0; nv];
    let mut max_excess = 0.0_f64;
    let mut eroded = 0.0;
    for i in 0..nv {
        if movable[i] && rho_s[i] > 0.0 {
            let ex = f_delta[i].max(0.0);
            max_excess = max_excess.max(ex);
            j_ero[i] = prm.k_e * rho_s[i] * ex / prm.sigma_0;
            eroded += j_ero[i] * area[i];
        }
    }
    (j_ero, max_excess, eroded)
}

fn deposit(topo: &Topology, area: &[f64], j_ero: &[f64], prm: &ModeCParams) -> Vec<f64> {
    if let Some(beta) = super::mode_c_spectral::spectrum(prm.kernel_width) {
        let _span = super::timing::span("mode_c_spectral");
        if let Some(j_dep) = super::mode_c_spectral::deposit(&topo.dirs, area, j_ero, &beta) {
            return j_dep;
        }
    }
    #[cfg(any(target_os = "macos", target_arch = "wasm32"))]
    {
        deposit_gpu(topo, area, j_ero, prm).unwrap_or_else(|err| panic!("mode C GPU: {err}"))
    }
    #[cfg(not(any(target_os = "macos", target_arch = "wasm32")))]
    {
        let _span = super::timing::span("mode_c_kernel");
        deposit_cpu(topo, area, j_ero, prm)
    }
}

#[cfg(any(target_os = "macos", target_arch = "wasm32"))]
fn deposit_gpu(
    topo: &Topology,
    area: &[f64],
    j_ero: &[f64],
    prm: &ModeCParams,
) -> Result<Vec<f64>, String> {
    let _caps = super::timing::span("mode_c_caps");
    let nv = topo.n_vertices();
    let w = prm.kernel_width;
    let cutoff = 4.0 * w;
    let dense = cutoff >= std::f64::consts::PI;
    // `acos` is strictly decreasing, so `θ < 4w` fails once the cosine is safely
    // below `cos(4w)`. The margin keeps the reject off the rounding boundary;
    // every point that can pass still takes the original `acos` test.
    let cos_reject = if dense {
        f64::NEG_INFINITY
    } else {
        cutoff.cos() - 1.0e-8
    };
    let bins = DirIndex::build(&topo.dirs);
    let caps = if dense {
        Vec::new()
    } else {
        let mut caps = vec![0u32; nv * 6];
        caps.par_chunks_mut(6)
            .zip(topo.dirs.par_iter())
            .for_each(|(slot, dir)| {
                slot.copy_from_slice(&cap_bounds(*dir, cutoff, bins.g));
            });
        caps
    };
    let mass_rate: Vec<f64> = j_ero
        .iter()
        .zip(area)
        .map(|(j, a)| if *j > 0.0 { j * a } else { 0.0 })
        .collect();
    drop(_caps);
    let _kernel = super::timing::span("mode_c_kernel");
    super::mode_c_gpu::deposit(
        &topo.dirs,
        area,
        &mass_rate,
        bins.g,
        &bins.start,
        &bins.items,
        &caps,
        w,
        cutoff,
        cos_reject,
        dense,
    )
}

#[cfg_attr(
    all(any(target_os = "macos", target_arch = "wasm32"), not(test)),
    allow(dead_code)
)]
fn deposit_cpu(topo: &Topology, area: &[f64], j_ero: &[f64], prm: &ModeCParams) -> Vec<f64> {
    let nv = topo.n_vertices();
    let mut j_dep = vec![0.0; nv];
    let w = prm.kernel_width;
    let cutoff = 4.0 * w;
    let cos_reject = if cutoff < std::f64::consts::PI {
        cutoff.cos() - 1.0e-8
    } else {
        f64::NEG_INFINITY
    };
    let bins = DirIndex::build(&topo.dirs);
    let sources: Vec<usize> = (0..nv).filter(|&src| j_ero[src] > 0.0).collect();
    let batch = rayon::current_num_threads().max(1);
    for chunk in sources.chunks(batch) {
        let parts: Vec<(f64, Vec<(u32, f64)>)> = chunk
            .par_iter()
            .map(|&src| kernel_terms(&topo.dirs, &bins, src, area, w, cutoff, cos_reject))
            .collect();
        for (&src, (norm, terms)) in chunk.iter().zip(parts) {
            let mass_rate = j_ero[src] * area[src];
            for (t, weight) in terms {
                j_dep[t as usize] += mass_rate * weight / norm;
            }
        }
    }
    j_dep
}

/// Positive Gaussian weights of one source, in increasing target index.
///
/// Targets are the direction-bin superset of the cap `θ < cutoff`. A weight is
/// the same value the dense loop would store: clamp the dot, reject only when
/// the cosine is safely outside, otherwise `exp(-½(acos(c)/w)²)` when
/// `acos(c) < cutoff`.
#[cfg_attr(
    all(any(target_os = "macos", target_arch = "wasm32"), not(test)),
    allow(dead_code)
)]
fn kernel_terms(
    dirs: &[DVec3],
    bins: &DirIndex,
    src: usize,
    area: &[f64],
    w: f64,
    cutoff: f64,
    cos_reject: f64,
) -> (f64, Vec<(u32, f64)>) {
    let mut targets = Vec::new();
    bins.collect_within(dirs[src], cutoff, &mut targets);
    targets.sort_unstable();
    targets.dedup();
    let mut norm = 0.0;
    let mut terms = Vec::new();
    let origin = dirs[src];
    for t in targets {
        let cos_raw = origin.dot(dirs[t]);
        if cos_raw < cos_reject {
            continue;
        }
        let cos = cos_raw.clamp(-1.0, 1.0);
        let ang = cos.acos();
        if ang < cutoff {
            let weight = (-0.5 * (ang / w).powi(2)).exp();
            if weight > 0.0 {
                norm += weight * area[t];
                terms.push((t as u32, weight));
            }
        }
    }
    (norm, terms)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mode_c_conserves_mass_exactly() {
        let topo = Topology::icosphere(2);
        let nv = topo.n_vertices();
        let area: Vec<f64> = (0..nv).map(|i| topo.vertex_solid[i] * 100.0).collect();
        let rho_s = vec![5.0; nv];
        let movable: Vec<bool> = (0..nv).map(|i| topo.dirs[i].z > 0.3).collect();
        let f: Vec<f64> = (0..nv).map(|i| 3.0 * topo.dirs[i].z).collect();
        let prm = ModeCParams {
            k_e: 0.1,
            sigma_0: 10.0,
            kernel_width: 0.3,
        };
        let cpu = mode_c_cpu(&topo, &area, &rho_s, &movable, &f, &prm);
        let ero: f64 = (0..nv).map(|i| cpu.j_ero[i] * area[i]).sum();
        let dep: f64 = (0..nv).map(|i| cpu.j_dep[i] * area[i]).sum();
        assert!(ero > 0.0);
        assert!((ero - dep).abs() < 1e-12 * ero);
        let gpu = mode_c(&topo, &area, &rho_s, &movable, &f, &prm);
        assert_eq!(gpu.j_ero, cpu.j_ero);
        assert_eq!(gpu.max_excess, cpu.max_excess);
        assert_eq!(gpu.eroded_rate, cpu.eroded_rate);
        let gpu_dep: f64 = (0..nv).map(|i| gpu.j_dep[i] * area[i]).sum();
        assert!(
            (ero - gpu_dep).abs() < 1e-6 * ero,
            "deposition residual {}",
            (ero - gpu_dep).abs() / ero
        );
        assert_deposition(&gpu.j_dep, &cpu.j_dep, prm.kernel_width);
    }

    #[test]
    fn deposition_matches_the_direct_angular_kernel() {
        let topo = Topology::icosphere(3);
        let nv = topo.n_vertices();
        let area: Vec<f64> = (0..nv)
            .map(|i| topo.vertex_solid[i] * (1.0 + 0.01 * (i % 7) as f64))
            .collect();
        let rho_s: Vec<f64> = (0..nv)
            .map(|i| {
                if i % 3 == 0 {
                    0.0
                } else {
                    1.0 + (i % 5) as f64
                }
            })
            .collect();
        let movable: Vec<bool> = (0..nv).map(|i| i % 4 != 1).collect();
        let f: Vec<f64> = (0..nv).map(|i| topo.dirs[i].z * 2.0 - 0.2).collect();
        for kernel_width in [1.0e-4, 0.2, 0.3, 1.0] {
            let prm = ModeCParams {
                k_e: 0.1,
                sigma_0: 10.0,
                kernel_width,
            };
            let got = mode_c_cpu(&topo, &area, &rho_s, &movable, &f, &prm);
            let want = direct_mode_c(&topo, &area, &rho_s, &movable, &f, &prm);
            assert_eq!(got.j_ero, want.j_ero);
            assert_eq!(got.max_excess, want.max_excess);
            assert_eq!(got.eroded_rate, want.eroded_rate);
            assert_eq!(got.j_dep, want.j_dep, "kernel_width {kernel_width}");
            let gpu = mode_c(&topo, &area, &rho_s, &movable, &f, &prm);
            assert_eq!(gpu.j_ero, want.j_ero);
            assert_deposition(&gpu.j_dep, &want.j_dep, kernel_width);
        }
    }

    /// Cap kernels stay at the pointwise reference. A zonal truncation is
    /// accepted at the relative L1 stated for that kernel, not pointwise.
    fn assert_deposition(got: &[f64], want: &[f64], width: f64) {
        let l1_num: f64 = got.iter().zip(want).map(|(g, w)| (g - w).abs()).sum();
        let l1_den = want.iter().map(|w| w.abs()).sum::<f64>().max(1e-30);
        let l1 = l1_num / l1_den;
        if super::super::mode_c_spectral::spectrum(width).is_some() {
            assert!(
                l1 <= 1.0e-3,
                "spectral deposition L1 {l1:.3e} at kernel width {width}"
            );
            return;
        }
        assert_field_close(got, want, 1e-5);
    }

    fn assert_field_close(got: &[f64], want: &[f64], tol: f64) {
        assert_eq!(got.len(), want.len());
        let scale = want
            .iter()
            .map(|v| v.abs())
            .fold(0.0_f64, f64::max)
            .max(1e-30);
        let mut worst = 0.0_f64;
        let mut at = 0usize;
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            let err = (g - w).abs() / scale;
            if err > worst {
                worst = err;
                at = i;
            }
        }
        assert!(
            worst <= tol,
            "deposition[{at}] gpu {} cpu {} relative {worst:.3e} (tolerance {tol:.1e})",
            got[at],
            want[at]
        );
    }

    #[test]
    fn ryugu_zonal_sample_matches_the_cap_kernel() {
        use std::path::PathBuf;

        use crate::scf_simulator::frame::vertex_areas;
        use crate::scf_simulator::io::initial_body;
        use crate::scf_simulator::params::{ModeCParams, Project};

        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets/projects");
        let text = std::fs::read_to_string(dir.join("ryugu_cauchy.toml")).expect("ryugu project");
        let project = Project::parse(&text, dir).expect("parse");
        let body = initial_body(&project).expect("body");
        let area = vertex_areas(&body);
        let width = project.mode_c.kernel_width;
        let beta = super::super::mode_c_spectral::spectrum(width)
            .expect("Ryugu kernel width must pass the 1e-3 truncation gate");
        let nv = body.topo.n_vertices();
        let mut j_ero = vec![0.0; nv];
        let mut n_src = 0usize;
        for src in (0..nv).step_by((nv / 200).max(1)) {
            j_ero[src] = 1.0;
            n_src += 1;
            if n_src == 200 {
                break;
            }
        }
        let fast = super::super::mode_c_spectral::deposit(&body.topo.dirs, &area, &j_ero, &beta)
            .expect("source norms stay positive");
        let cap = deposit_cpu(
            &body.topo,
            &area,
            &j_ero,
            &ModeCParams {
                k_e: 1.0,
                sigma_0: 1.0,
                kernel_width: width,
            },
        );
        let num: f64 = fast.iter().zip(&cap).map(|(a, b)| (a - b).abs()).sum();
        let den = cap.iter().map(|v| v.abs()).sum::<f64>().max(1e-30);
        let l1 = num / den;
        let ero: f64 = j_ero.iter().zip(&area).map(|(j, a)| j * a).sum();
        let dep: f64 = fast.iter().zip(&area).map(|(j, a)| j * a).sum();
        assert!(
            (ero - dep).abs() <= 1.0e-8 * ero,
            "spectral mass gap {}",
            (ero - dep).abs() / ero
        );
        assert!(
            l1 <= 1.0e-3,
            "Ryugu zonal sample L1 {l1:.3e} on {n_src} sources"
        );
        eprintln!("ryugu zonal sample L1={l1:.3e} sources={n_src}");
    }

    #[test]
    fn cap_query_contains_every_vertex_inside_the_angle() {
        let topo = Topology::icosphere(4);
        let bins = DirIndex::build(&topo.dirs);
        let mut got = Vec::new();
        for alpha in [1.0e-4, 0.2, 0.8, 1.2, 3.2] {
            for src in (0..topo.n_vertices()).step_by(19) {
                bins.collect_within(topo.dirs[src], alpha, &mut got);
                got.sort_unstable();
                for (t, dir) in topo.dirs.iter().enumerate() {
                    let ang = topo.dirs[src].dot(*dir).clamp(-1.0, 1.0).acos();
                    if ang < alpha {
                        assert!(
                            got.binary_search(&t).is_ok(),
                            "missed vertex {t} from {src} at angle {ang} (cap {alpha})"
                        );
                    }
                }
            }
        }
    }

    /// The dense loop this kernel replaces. Kept so a faster path cannot drift.
    fn direct_mode_c(
        topo: &Topology,
        area: &[f64],
        rho_s: &[f64],
        movable: &[bool],
        f_delta: &[f64],
        prm: &ModeCParams,
    ) -> ModeC {
        let nv = topo.n_vertices();
        let mut j_ero = vec![0.0; nv];
        let mut max_excess = 0.0_f64;
        for i in 0..nv {
            if movable[i] && rho_s[i] > 0.0 {
                let ex = f_delta[i].max(0.0);
                max_excess = max_excess.max(ex);
                j_ero[i] = prm.k_e * rho_s[i] * ex / prm.sigma_0;
            }
        }
        let mut j_dep = vec![0.0; nv];
        let w = prm.kernel_width;
        let mut eroded = 0.0;
        let mut g = vec![0.0; nv];
        for src in 0..nv {
            if j_ero[src] <= 0.0 {
                continue;
            }
            let mass_rate = j_ero[src] * area[src];
            eroded += mass_rate;
            let mut norm = 0.0;
            for t in 0..nv {
                let cos = topo.dirs[src].dot(topo.dirs[t]).clamp(-1.0, 1.0);
                let ang = cos.acos();
                g[t] = if ang < 4.0 * w {
                    (-0.5 * (ang / w).powi(2)).exp()
                } else {
                    0.0
                };
                norm += g[t] * area[t];
            }
            for t in 0..nv {
                if g[t] > 0.0 {
                    j_dep[t] += mass_rate * g[t] / norm;
                }
            }
        }
        ModeC {
            j_ero,
            j_dep,
            max_excess,
            eroded_rate: eroded,
        }
    }

    #[test]
    fn transport_conserves_and_translates_a_bump() {
        let topo = Topology::icosphere(3);
        let nv = topo.n_vertices();
        // A rigid rotation about z at rate 0.05 rad per unit time.
        let flow = SurfaceFlow {
            c: topo
                .dirs
                .iter()
                .map(|d| DVec3::new(-d.y, d.x, 0.0) * 0.05)
                .collect(),
            div: vec![0.0; nv],
            grad_max: 0.05,
            tangent_det: vec![0.0; nv],
        };
        let field: Vec<f64> = topo
            .dirs
            .iter()
            .map(|d| (-(d.x - 1.0).powi(2) * 6.0 - d.y * d.y * 6.0 - d.z * d.z * 6.0).exp())
            .collect();
        let j = vec![1.0; nv];
        let (out, fix) = transport(&topo, &flow, &field, &j, &j, 1.0).unwrap();
        let mass = |f: &[f64]| (0..nv).map(|i| f[i] * topo.vertex_solid[i]).sum::<f64>();
        assert!((mass(&out) - mass(&field)).abs() < 1e-12 * mass(&field));
        assert!(fix < 0.1);
        // The bump centre moved from +x towards +y.
        let cy_before: f64 = (0..nv)
            .map(|i| field[i] * topo.dirs[i].y * topo.vertex_solid[i])
            .sum();
        let cy_after: f64 = (0..nv)
            .map(|i| out[i] * topo.dirs[i].y * topo.vertex_solid[i])
            .sum();
        assert!(cy_after > cy_before + 1e-3, "{cy_before} -> {cy_after}");
    }

    fn brute_nearest(dirs: &[DVec3], p: DVec3) -> usize {
        (0..dirs.len())
            .max_by(|a, b| dirs[*a].dot(p).partial_cmp(&dirs[*b].dot(p)).unwrap())
            .unwrap()
    }

    #[test]
    fn direction_grid_matches_the_linear_scan() {
        let topo = Topology::icosphere(4);
        let index = DirIndex::build(&topo.dirs);
        for d in &topo.dirs {
            assert_eq!(index.nearest(&topo.dirs, *d), brute_nearest(&topo.dirs, *d));
        }
        let golden = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
        for i in 0..40 {
            let z = 1.0 - 2.0 * (i as f64 + 0.5) / 40.0;
            let r = (1.0 - z * z).sqrt();
            let p = DVec3::new(
                r * (golden * i as f64).cos(),
                r * (golden * i as f64).sin(),
                z,
            );
            assert_eq!(index.nearest(&topo.dirs, p), brute_nearest(&topo.dirs, p));
        }
        let mut dirs = topo.dirs.clone();
        dirs.push(dirs[3]);
        let tied = DirIndex::build(&dirs);
        assert_eq!(tied.nearest(&dirs, dirs[3]), brute_nearest(&dirs, dirs[3]));
        assert_eq!(brute_nearest(&dirs, dirs[3]), dirs.len() - 1);
        let field: Vec<f64> = (0..topo.n_vertices())
            .map(|i| (i as f64 * 0.17).sin())
            .collect();
        let face = topo.faces[10];
        let mid = (topo.dirs[face[0] as usize]
            + topo.dirs[face[1] as usize]
            + topo.dirs[face[2] as usize])
            .try_normalize()
            .unwrap();
        let nearest = index.nearest(&topo.dirs, mid);
        assert_eq!(nearest, brute_nearest(&topo.dirs, mid));
        let value = interp_vertex_field(&topo, &index, &field, mid);
        let [i, j, k] = face.map(|v| v as usize);
        let span = [field[i], field[j], field[k], field[nearest]];
        assert!(value >= span.into_iter().fold(f64::INFINITY, f64::min) - 1e-12);
        assert!(value <= span.into_iter().fold(f64::NEG_INFINITY, f64::max) + 1e-12);
    }
}
