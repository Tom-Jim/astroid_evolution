//! Loose-layer density `ρ_s` on the polyhedral surface (§2.8, §8.0 G5, §9.2): Mode C
//! erosion and deposition, the deposition spread `G_w` as a heat kernel on the mesh, and
//! conservative semi-Lagrangian transport along the tangential slip.
//!
//! `ρ_s` (kg m⁻²) lives on the vertices, as vertex masses `m_i = ρ_{s,i} A_i`.

use nalgebra::Matrix2;
use rayon::prelude::*;

use crate::common::math::DVec3;
use crate::rheo::model::local_frame;

use super::body::{Body, Topology};
use super::params::ModeCParams;
use super::runout::{Arrival, moved_source};
use super::surface::barycentric;

/// Tangential slip `v_T = χ_A (v_sl − (v_sl·N) N)` and its surface gradient.
#[derive(Clone, Debug)]
pub struct SurfaceFlow {
    pub v_t: Vec<DVec3>,
    /// `∇_Γ·v_T`, `det ∇_Γ v_T`, and `‖∇_Γ v_T‖_∞`.
    pub div: Vec<f64>,
    pub tangent_det: Vec<f64>,
    pub grad_max: f64,
}

/// `v_T` and `∇_Γ v_T`, fitted by least squares on the vertex ring in the tangent frame
/// of `N_i` (G5).
pub fn surface_flow(body: &Body, normals: &[DVec3], v_sl: &[DVec3], active: &[bool]) -> SurfaceFlow {
    let x = &body.positions;
    let v_t: Vec<DVec3> = (0..x.len())
        .map(|i| {
            if active[i] {
                v_sl[i] - normals[i] * v_sl[i].dot(normals[i])
            } else {
                DVec3::ZERO
            }
        })
        .collect();
    let fits: Vec<(f64, f64, f64)> = (0..x.len())
        .into_par_iter()
        .map(|i| {
            let (e1, e2) = local_frame(normals[i]);
            let (mut ata, mut atb) = (Matrix2::<f64>::zeros(), Matrix2::<f64>::zeros());
            for &j in &body.topo.vertex_neighbors[i] {
                let dx = x[j] - x[i];
                let dv = v_t[j] - v_t[i];
                let a = nalgebra::Vector2::new(dx.dot(e1), dx.dot(e2));
                let b = nalgebra::Vector2::new(dv.dot(e1), dv.dot(e2));
                ata += a * a.transpose();
                atb += b * a.transpose();
            }
            match ata.try_inverse() {
                Some(inv) => {
                    let g = atb * inv;
                    (g.trace(), g.determinant(), g.norm())
                }
                None => (0.0, 0.0, 0.0),
            }
        })
        .collect();
    SurfaceFlow {
        v_t,
        div: fits.iter().map(|f| f.0).collect(),
        tangent_det: fits.iter().map(|f| f.1).collect(),
        grad_max: fits.iter().map(|f| f.2).fold(0.0, f64::max),
    }
}

/// End of the straightest geodesic from vertex `a` with initial displacement `disp`,
/// straight within a face and unfolded across edges (G5): the face and barycentric weights.
/// A path through a vertex continues from that vertex in the face whose wedge contains
/// the direction.
pub fn geodesic(topo: &Topology, x: &[DVec3], a: usize, disp: DVec3) -> (usize, [f64; 3]) {
    let tri = |f: usize| topo.faces[f].map(|v| x[v as usize]);
    let normal = |f: usize| {
        let [a, b, c] = tri(f);
        (b - a).cross(c - a).normalize_or_zero()
    };
    let len = disp.length();
    if len == 0.0 {
        let f = topo.vertex_faces[a][0];
        let mut w = [0.0; 3];
        w[topo.faces[f].iter().position(|&v| v as usize == a).unwrap_or(0)] = 1.0;
        return (f, w);
    }
    let (mut d, mut left) = (disp / len, len);
    let (mut vertex, mut p) = (Some(a), x[a]);
    let (mut f, mut entry) = (topo.vertex_faces[a][0], usize::MAX);
    // `eps` guards the step; `snap` is the edge-parameter band within which a crossing is
    // taken at the nearer vertex (a path along an edge meets the far vertex there).
    let (eps, snap) = (1e-9, 1e-3);
    for _ in 0..4 * topo.n_faces() {
        if let Some(v) = vertex.take() {
            // The incident face whose wedge at `v` contains the projected direction.
            let mut best = (f, f64::NEG_INFINITY);
            for &g in &topo.vertex_faces[v] {
                let t = topo.faces[g].map(|k| k as usize);
                let k = t.iter().position(|&q| q == v).unwrap_or(0);
                let (b, c) = (x[t[(k + 1) % 3]] - x[v], x[t[(k + 2) % 3]] - x[v]);
                let n = b.cross(c).normalize_or_zero();
                let dp = d - n * d.dot(n);
                let (bb, bc, cc) = (b.dot(b), b.dot(c), c.dot(c));
                let det = bb * cc - bc * bc;
                let (db, dc) = (dp.dot(b), dp.dot(c));
                let (al, be) = ((db * cc - dc * bc) / det, (dc * bb - db * bc) / det);
                let score = al.min(be) / (al.abs() + be.abs()).max(1e-300);
                if score > best.1 {
                    best = (g, score);
                }
            }
            f = best.0;
            entry = usize::MAX;
            let n = normal(f);
            d = (d - n * d.dot(n)).normalize_or_zero();
        }
        let corners = tri(f);
        let n = normal(f);
        // Nearest exit through an edge other than the entry edge, inside the segment.
        let mut exit = (f64::INFINITY, usize::MAX, 0.0);
        for e in 0..3 {
            if e == entry {
                continue;
            }
            let (pa, pb) = (corners[e], corners[(e + 1) % 3]);
            let edge = pb - pa;
            let den = d.cross(edge).dot(n);
            if den.abs() < 1e-300 {
                continue;
            }
            let t = (pa - p).cross(edge).dot(n) / den;
            let sp = (pa - p).cross(d).dot(n) / den;
            if t > eps * len && t < exit.0 && (-snap..=1.0 + snap).contains(&sp) {
                exit = (t, e, sp);
            }
        }
        if exit.1 == usize::MAX || left <= exit.0 {
            let q = p + d * left.min(exit.0.min(left));
            let w = barycentric(q, corners[0], corners[1], corners[2]).map(|w| w.max(0.0));
            let s: f64 = w.iter().sum();
            return (f, w.map(|w| w / s.max(1e-300)));
        }
        let (t, e, sp) = exit;
        p += d * t;
        left -= t;
        let ids = topo.faces[f];
        if sp < snap || sp > 1.0 - snap {
            vertex = Some(ids[if sp < 0.5 { e } else { (e + 1) % 3 }] as usize);
            p = x[vertex.unwrap()];
            continue;
        }
        // Unfold across the edge: keep the components along it and across it.
        let g = topo.face_neighbors[f][e];
        let axis = (corners[(e + 1) % 3] - corners[e]).normalize_or_zero();
        let ng = normal(g);
        let mut across_g = ng.cross(axis);
        let centre_g = tri(g).iter().fold(DVec3::ZERO, |s, v| s + *v) / 3.0;
        if (centre_g - corners[e]).dot(across_g) < 0.0 {
            across_g = -across_g;
        }
        d = (axis * d.dot(axis) + across_g * d.dot(n.cross(axis)).abs()).normalize_or_zero();
        entry = (0..3).find(|&k| topo.face_neighbors[g][k] == f).unwrap_or(usize::MAX);
        f = g;
    }
    let c = tri(f);
    (f, barycentric(p, c[0], c[1], c[2]))
}

/// One semi-Lagrangian step of `ρ_s` (G5): foot point along `−τ v_T`, barycentric value,
/// the tangent Jacobian, then `ρ_s = m / A_new` and one global factor that restores
/// `Σ A ρ_s`. Returns the field and the relative size of that factor.
pub fn transport(
    body: &Body,
    flow: &SurfaceFlow,
    rho_s: &[f64],
    area_old: &[f64],
    area_new: &[f64],
    tau: f64,
) -> (Vec<f64>, f64) {
    let topo = &body.topo;
    let x = &body.positions;
    let mut out: Vec<f64> = (0..x.len())
        .into_par_iter()
        .map(|a| {
            let (f, w) = geodesic(topo, x, a, flow.v_t[a] * -tau);
            let t = topo.faces[f].map(|v| v as usize);
            let value = w[0] * rho_s[t[0]] + w[1] * rho_s[t[1]] + w[2] * rho_s[t[2]];
            let jac = (1.0 - tau * flow.div[a] + tau * tau * flow.tangent_det[a]).max(1e-12);
            (jac * value * area_old[a] / area_new[a]).max(0.0)
        })
        .collect();
    let before: f64 = rho_s.iter().zip(area_old).map(|(r, a)| r * a).sum();
    let after: f64 = out.iter().zip(area_new).map(|(r, a)| r * a).sum();
    let mut fix = 0.0;
    if after > 0.0 && before > 0.0 {
        let s = before / after;
        fix = (s - 1.0).abs();
        out.iter_mut().for_each(|v| *v *= s);
    }
    (out, fix)
}

/// The isotropic spread `G_w` of §2.8 on `Γ`: diffusion `∂_t u = Δ_Γ u` for `t = w²/2` (a
/// Gaussian of width `w` on a flat surface), in `SUBSTEPS` implicit steps of the
/// barycentric-dual finite-volume Laplacian. Its weights are positive, so the spread is
/// positive, and `Σ A u` is conserved by construction.
pub struct Spread {
    area: Vec<f64>,
    /// Symmetric edge weights `|dual edge| / |edge|` in compressed rows.
    start: Vec<usize>,
    cols: Vec<usize>,
    weights: Vec<f64>,
    tau: f64,
}

const SUBSTEPS: usize = 4;

impl Spread {
    pub fn new(body: &Body, area: &[f64], width: f64) -> Self {
        let x = &body.positions;
        let topo = &body.topo;
        let nv = x.len();
        let mut start = vec![0usize; nv + 1];
        let mut cols = Vec::new();
        let mut weights = Vec::new();
        for i in 0..nv {
            for &j in &topo.vertex_neighbors[i] {
                // The two faces on the edge (i, j); the dual edge joins their centroids
                // through the edge midpoint.
                let mid = (x[i] + x[j]) * 0.5;
                let dual: f64 = topo.vertex_faces[i]
                    .iter()
                    .filter(|&&f| topo.faces[f].contains(&(j as u32)))
                    .map(|&f| {
                        let c = topo.faces[f].iter().fold(DVec3::ZERO, |s, v| s + x[*v as usize]) / 3.0;
                        (c - mid).length()
                    })
                    .sum();
                cols.push(j);
                weights.push(dual / (x[j] - x[i]).length().max(1e-300));
            }
            start[i + 1] = cols.len();
        }
        Self {
            area: area.to_vec(),
            start,
            cols,
            weights,
            tau: 0.5 * width * width / SUBSTEPS as f64,
        }
    }

    /// `(A + τ L) u` with `L u_i = Σ_j w_ij (u_i − u_j)`.
    fn apply(&self, u: &[f64], out: &mut [f64]) {
        out.par_iter_mut().enumerate().for_each(|(i, o)| {
            let mut s = 0.0;
            for k in self.start[i]..self.start[i + 1] {
                s += self.weights[k] * (u[i] - u[self.cols[k]]);
            }
            *o = self.area[i] * u[i] + self.tau * s;
        });
    }

    /// Spread a density per unit area; `Σ A J_dep = Σ A j` to rounding.
    pub fn spread(&self, j: &[f64]) -> Vec<f64> {
        let n = j.len();
        if j.iter().all(|v| *v == 0.0) {
            return j.to_vec();
        }
        let diag: Vec<f64> = (0..n)
            .map(|i| {
                self.area[i] + self.tau * self.weights[self.start[i]..self.start[i + 1]].iter().sum::<f64>()
            })
            .collect();
        let mut u = j.to_vec();
        let (mut r, mut z, mut p, mut ap) = (vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        for _ in 0..SUBSTEPS {
            // Jacobi-preconditioned CG on (A + τL) u' = A u, warm-started at u.
            let b: Vec<f64> = (0..n).map(|i| self.area[i] * u[i]).collect();
            let bnorm = b.iter().map(|v| v * v).sum::<f64>().sqrt();
            self.apply(&u, &mut ap);
            for i in 0..n {
                r[i] = b[i] - ap[i];
                z[i] = r[i] / diag[i];
                p[i] = z[i];
            }
            let mut rz: f64 = r.iter().zip(&z).map(|(a, b)| a * b).sum();
            for _ in 0..1000 {
                if r.iter().map(|v| v * v).sum::<f64>().sqrt() <= 1e-12 * bnorm {
                    break;
                }
                self.apply(&p, &mut ap);
                let alpha = rz / p.iter().zip(&ap).map(|(a, b)| a * b).sum::<f64>();
                for i in 0..n {
                    u[i] += alpha * p[i];
                    r[i] -= alpha * ap[i];
                    z[i] = r[i] / diag[i];
                }
                let rz_new: f64 = r.iter().zip(&z).map(|(a, b)| a * b).sum();
                let beta = rz_new / rz;
                rz = rz_new;
                for i in 0..n {
                    p[i] = z[i] + beta * p[i];
                }
            }
        }
        // Exact conservation and positivity against the CG tolerance.
        u.iter_mut().for_each(|v| *v = v.max(0.0));
        let (before, after): (f64, f64) = (0..n)
            .map(|i| (self.area[i] * j[i], self.area[i] * u[i]))
            .fold((0.0, 0.0), |s, v| (s.0 + v.0, s.1 + v.1));
        if after > 0.0 {
            u.iter_mut().for_each(|v| *v *= before / after);
        }
        u
    }
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
/// Erosion is a per-vertex scale; deposition is the spread `G_w` of the eroded rate.
///
/// With `arrivals` (§2.8b, A-C3) each source's eroded mass is first moved to where its
/// packet came to rest; the spread is then the same.
pub fn mode_c(
    spread: &Spread,
    area: &[f64],
    rho_s: &[f64],
    movable: &[bool],
    f_delta: &[f64],
    prm: &ModeCParams,
    arrivals: Option<&[Arrival]>,
) -> ModeC {
    let (j_ero, max_excess, eroded) = erosion(area, rho_s, movable, f_delta, prm);
    let j_dep = deposit_from(spread, area, &j_ero, arrivals);
    ModeC {
        j_ero,
        j_dep,
        max_excess,
        eroded_rate: eroded,
    }
}

/// Mode C fluxes averaged over a step of length `dt` (exact for the frozen erosion rate).
pub fn mode_c_over(
    spread: &Spread,
    area: &[f64],
    rho_s: &[f64],
    movable: &[bool],
    f_delta: &[f64],
    prm: &ModeCParams,
    arrivals: Option<&[Arrival]>,
    dt: f64,
) -> ModeC {
    let (j_ero, max_excess, eroded) = erosion_over(area, rho_s, movable, f_delta, prm, dt);
    let j_dep = deposit_from(spread, area, &j_ero, arrivals);
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


fn deposit_from(spread: &Spread, area: &[f64], j_ero: &[f64], arrivals: Option<&[Arrival]>) -> Vec<f64> {
    match arrivals {
        Some(arrivals) => spread.spread(&moved_source(area, j_ero, arrivals)),
        None => spread.spread(j_ero),
    }
}
