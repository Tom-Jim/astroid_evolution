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
use super::runout::{Arrival, moved_source};

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
    pub fn collect_within(&self, origin: DVec3, alpha: f64, out: &mut Vec<usize>) {
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
pub fn locate_vertex(topo: &Topology, index: &DirIndex, p: DVec3) -> ([usize; 3], [f64; 3]) {
    let (nearest, faces) = face_ring(topo, index, p);
    for f in faces {
        if let Some(hit) = weights_on(topo, f, p) {
            return hit;
        }
    }
    ([nearest, nearest, nearest], [1.0, 0.0, 0.0])
}

/// Face that contains `p`, or the first face incident to the nearest vertex.
pub fn locate_face(topo: &Topology, index: &DirIndex, p: DVec3) -> usize {
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
///
/// With `arrivals` (§2.8b, A-C3) each source's eroded mass is first moved to where its
/// packet came to rest; the spread is then the same.
pub fn mode_c(
    topo: &Topology,
    area: &[f64],
    rho_s: &[f64],
    movable: &[bool],
    f_delta: &[f64],
    prm: &ModeCParams,
    arrivals: Option<&[Arrival]>,
) -> ModeC {
    let (j_ero, max_excess, eroded) = erosion(area, rho_s, movable, f_delta, prm);
    let j_dep = deposit_from(topo, area, &j_ero, prm, arrivals);
    ModeC {
        j_ero,
        j_dep,
        max_excess,
        eroded_rate: eroded,
    }
}

/// Mode C fluxes averaged over a step of length `dt` (exact for the frozen erosion rate).
pub fn mode_c_over(
    topo: &Topology,
    area: &[f64],
    rho_s: &[f64],
    movable: &[bool],
    f_delta: &[f64],
    prm: &ModeCParams,
    arrivals: Option<&[Arrival]>,
    dt: f64,
) -> ModeC {
    let (j_ero, max_excess, eroded) = erosion_over(area, rho_s, movable, f_delta, prm, dt);
    let j_dep = deposit_from(topo, area, &j_ero, prm, arrivals);
    ModeC {
        j_ero,
        j_dep,
        max_excess,
        eroded_rate: eroded,
    }
}

pub fn erosion(
    area: &[f64],
    rho_s: &[f64],
    movable: &[bool],
    f_delta: &[f64],
    prm: &ModeCParams,
) -> (Vec<f64>, f64, f64) {
    erosion_over(area, rho_s, movable, f_delta, prm, 0.0)
}

/// Erosion rate averaged over a step `dt` (`dt = 0`: the instantaneous rate).
/// `J_ero = λ ρ_s` with `λ = k_e [F_δ]_+ / σ₀` frozen in the step is linear in
/// `ρ_s`, so the layer decays exactly as `ρ_s e^{−λ t}` and the step average is
/// `ρ_s (1 − e^{−λ dt}) / dt` (§10.4.1): positive for every `dt`.
pub fn erosion_over(
    area: &[f64],
    rho_s: &[f64],
    movable: &[bool],
    f_delta: &[f64],
    prm: &ModeCParams,
    dt: f64,
) -> (Vec<f64>, f64, f64) {
    let nv = area.len();
    let mut j_ero = vec![0.0; nv];
    let mut max_excess = 0.0_f64;
    let mut eroded = 0.0;
    for i in 0..nv {
        if movable[i] && rho_s[i] > 0.0 {
            let ex = f_delta[i].max(0.0);
            max_excess = max_excess.max(ex);
            let rate = prm.k_e * ex / prm.sigma_0;
            j_ero[i] = if dt > 0.0 {
                -rho_s[i] * (-rate * dt).exp_m1() / dt
            } else {
                rate * rho_s[i]
            };
            eroded += j_ero[i] * area[i];
        }
    }
    (j_ero, max_excess, eroded)
}

fn deposit_from(
    topo: &Topology,
    area: &[f64],
    j_ero: &[f64],
    prm: &ModeCParams,
    arrivals: Option<&[Arrival]>,
) -> Vec<f64> {
    match arrivals {
        Some(arrivals) => deposit(topo, area, &moved_source(area, j_ero, arrivals), prm),
        None => deposit(topo, area, j_ero, prm),
    }
}

pub fn deposit(topo: &Topology, area: &[f64], j_ero: &[f64], prm: &ModeCParams) -> Vec<f64> {
    if let Some(beta) = super::mode_c_spectral::spectrum(prm.kernel_width) {
        let _span = super::timing::span("mode_c_spectral");
        if let Some(j_dep) = super::mode_c_spectral::deposit(&topo.dirs, area, j_ero, &beta) {
            return j_dep;
        }
    }
    deposit_gpu(topo, area, j_ero, prm).unwrap_or_else(|err| panic!("mode C GPU: {err}"))
}

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
