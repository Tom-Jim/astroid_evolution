//! Loose-layer density `ρ_s`: mode C erosion/deposition (§2.8) and conservative
//! semi-Lagrangian transport on `S²` (§9.2), with the Jacobian correction.
//!
//! `ρ_s` (kg m⁻²) lives on the surface vertices. The parameter velocity of a surface
//! material point is `c = v_S / R`, where `v_S` is the tangential part of the velocity
//! relative to the rigid core, restricted to the active set (`χ_A`).

use nalgebra::{Matrix3, Vector3};

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
            grad_max = grad_max.max(g.norm());
        }
    }
    SurfaceFlow { c, div, grad_max }
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

/// Barycentric interpolation of a vertex field at the direction `p` (nearest-vertex ring search).
pub fn interp_vertex_field(topo: &Topology, field: &[f64], p: DVec3) -> f64 {
    let nearest = (0..topo.n_vertices())
        .max_by(|a, b| {
            topo.dirs[*a]
                .dot(p)
                .partial_cmp(&topo.dirs[*b].dot(p))
                .unwrap()
        })
        .unwrap();
    let mut seen: Vec<usize> = topo.vertex_faces[nearest].clone();
    let first = seen.clone();
    for f in first {
        for &g in &topo.face_neighbors[f] {
            if !seen.contains(&g) {
                seen.push(g);
            }
        }
    }
    for f in seen {
        let [i, j, k] = topo.faces[f].map(|x| x as usize);
        if let Some(l) = barycentric(topo.dirs[i], topo.dirs[j], topo.dirs[k], p) {
            if l.iter().all(|x| *x >= -1e-12) {
                return l[0] * field[i] + l[1] * field[j] + l[2] * field[k];
            }
        }
    }
    field[nearest]
}

/// One semi-Lagrangian step of the vertex field `field` (`ρ_s`) with the area densities
/// `j_old`, `j_new` (area per solid angle before and after the geometric update).
///
/// `j_new,a ρ_a = J_a j_old(ω_d) ρ(ω_d)` with `J_a = 1 − Δt ∇·c` to first order, followed by a
/// global multiplicative fixer that restores `Σ A ρ` exactly. The relative size of the
/// fixer is returned.
pub fn transport(
    topo: &Topology,
    flow: &SurfaceFlow,
    field: &[f64],
    j_old: &[f64],
    j_new: &[f64],
    dt: f64,
) -> (Vec<f64>, f64) {
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
        for f in seen {
            let t = topo.faces[f];
            let [i, j, k] = t.map(|x| x as usize);
            if let Some(l) = barycentric(topo.dirs[i], topo.dirs[j], topo.dirs[k], foot) {
                if l.iter().all(|x| *x >= -1e-12) {
                    value = l[0] * field[i] * j_old[i]
                        + l[1] * field[j] * j_old[j]
                        + l[2] * field[k] * j_old[k];
                    break;
                }
            }
        }
        let jac = (1.0 - dt * flow.div[a]).max(0.0);
        out[a] = (jac * value / j_new[a]).max(0.0);
    }
    // Conservative fixer on Σ j Ω ρ.
    let before: f64 = (0..nv).map(|i| j_old[i] * topo.vertex_solid[i] * field[i]).sum();
    let after: f64 = (0..nv).map(|i| j_new[i] * topo.vertex_solid[i] * out[i]).sum();
    let mut fix = 0.0;
    if after > 0.0 && before > 0.0 {
        let s = before / after;
        fix = (s - 1.0).abs();
        for v in &mut out {
            *v *= s;
        }
    }
    (out, fix)
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
/// `∫ K(a, a') dΓ(a) = 1` for every source, so that `∫ J_dep dΓ = ∫ J_ero dΓ` exactly.
pub fn mode_c(
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
        // The source itself always lies inside the kernel, so `norm > 0`.
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
        let m = mode_c(&topo, &area, &rho_s, &movable, &f, &prm);
        let ero: f64 = (0..nv).map(|i| m.j_ero[i] * area[i]).sum();
        let dep: f64 = (0..nv).map(|i| m.j_dep[i] * area[i]).sum();
        assert!(ero > 0.0);
        assert!((ero - dep).abs() < 1e-12 * ero);
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
        };
        let field: Vec<f64> = topo
            .dirs
            .iter()
            .map(|d| (-(d.x - 1.0).powi(2) * 6.0 - d.y * d.y * 6.0 - d.z * d.z * 6.0).exp())
            .collect();
        let j = vec![1.0; nv];
        let (out, fix) = transport(&topo, &flow, &field, &j, &j, 1.0);
        let mass = |f: &[f64]| (0..nv).map(|i| f[i] * topo.vertex_solid[i]).sum::<f64>();
        assert!((mass(&out) - mass(&field)).abs() < 1e-12 * mass(&field));
        assert!(fix < 0.1);
        // The bump centre moved from +x towards +y.
        let cy_before: f64 = (0..nv).map(|i| field[i] * topo.dirs[i].y * topo.vertex_solid[i]).sum();
        let cy_after: f64 = (0..nv).map(|i| out[i] * topo.dirs[i].y * topo.vertex_solid[i]).sum();
        assert!(cy_after > cy_before + 1e-3, "{cy_before} -> {cy_after}");
    }
}
