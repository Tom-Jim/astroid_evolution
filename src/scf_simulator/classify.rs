//! Yield classification, rigid core and sliding velocity (§2.8 B2–B4, B6).
//!
//! The single stress field of the Picard solution is evaluated on the classification grid
//! `X_Y` (vertex directions × radial Gauss–Jacobi nodes × the surface layer). `F_δ > 0`
//! marks yield; the unyielded points are connected through the graph `G_Y` and the heaviest
//! component is the rigid core. Everything else is mobile.

use disjoint_sets::UnionFind;
use nalgebra::{DMatrix, DVector};

use crate::common::math::DVec3;
use crate::rheo_friction::Material;

use super::basis::{Want, eval_pressure, eval_velocity_rays};
use super::body::Topology;
use super::harmonics::SurfaceFit;
use super::quad::gauss_jacobi_zeta2;
use super::stokes::{Operators, StokesSolution, eta_eval};

#[derive(Clone, Debug)]
pub struct Classification {
    /// `χ_Y`: the vertex is not part of the rigid core.
    pub movable: Vec<bool>,
    /// `F_δ` at the surface nodes.
    pub f_delta: Vec<f64>,
    /// Velocity relative to the rigid core, at the surface nodes.
    pub v_sl: Vec<DVec3>,
    /// Unit surface normal of the harmonic surface at the vertex directions.
    pub normal: Vec<DVec3>,
    /// Harmonic-surface radius at the vertex directions.
    pub radius: Vec<f64>,
    pub core_a: DVec3,
    pub core_theta: DVec3,
    pub core_degenerate: bool,
    pub core_empty: bool,
    pub yielded_fraction: f64,
    pub core_mass_fraction: f64,
    pub mobile_area_fraction: f64,
    /// `max |v_sl|` inside the core: regularised creep, a diagnostic only.
    pub core_creep: f64,
    /// `Φ_p = ∫ 1{F_δ>0} τ:D dV`.
    pub plastic_dissipation: f64,
    /// `Ek⁻¹ = max_Y ρ ω R² / η` over the yielded set.
    pub inverse_ekman: f64,
}

#[allow(clippy::too_many_arguments)]
pub fn classify(
    ops: &Operators,
    sol: &StokesSolution,
    surf: &SurfaceFit,
    topo: &Topology,
    material: &Material,
    delta_y: f64,
    n_radial: usize,
    rho_of: &dyn Fn(DVec3) -> f64,
    omega: f64,
    p_min: f64,
) -> Classification {
    let nv = topo.n_vertices();
    let (zeta, wr) = gauss_jacobi_zeta2(n_radial);
    let layers = n_radial + 1;
    // Grid X_Y: layer q < n_radial is interior, layer n_radial the surface.
    let (radius, normal): (Vec<f64>, Vec<DVec3>) = surf.normals(&topo.dirs).into_iter().unzip();
    let mut pts = Vec::with_capacity(nv * layers);
    let mut wgt = Vec::with_capacity(nv * layers);
    for i in 0..nv {
        for q in 0..layers {
            let z = if q < n_radial { zeta[q] } else { 1.0 };
            pts.push(topo.dirs[i] * (z * radius[i]));
            wgt.push(if q < n_radial {
                wr[q] * radius[i].powi(3) * topo.vertex_solid[i]
            } else {
                0.0
            });
        }
    }
    let u = DVector::from_column_slice(&sol.u);
    let rows = eval_velocity_rays(surf, &ops.lay, &pts, layers, Want::VD);
    let v = &rows.v * &u;
    let d = &rows.d * &u;
    let pres = eval_pressure(surf, &ops.play, &pts, false);
    let pvals = &pres.val * DVector::from_column_slice(&sol.pres_c)
        + &pres.lift_val * DVector::from_column_slice(&sol.lift);
    let np = pts.len();
    let mut fdelta = vec![0.0; np];
    let mut shear = vec![0.0; np];
    let mut eta = vec![0.0; np];
    for p in 0..np {
        let d6 = &d.as_slice()[6 * p..6 * p + 6];
        let sh = (2.0 * d6.iter().map(|x| x * x).sum::<f64>()).sqrt();
        let ptil = pvals[p].max(p_min);
        let ev = eta_eval(material, sh, ptil, sol.gamma_eps);
        let tau_y = material.spec.cohesion + material.mu_y * ptil;
        shear[p] = sh;
        eta[p] = ev.eta;
        fdelta[p] = ev.eta * sh - (1.0 - delta_y) * tau_y;
    }
    // Graph G_Y on the unyielded points.
    let virt = np;
    let mut dsu = UnionFind::new(np + 1);
    let ok = |p: usize| fdelta[p] <= 0.0;
    for i in 0..nv {
        for q in 0..layers {
            let p = i * layers + q;
            if !ok(p) {
                continue;
            }
            if q + 1 < layers && ok(p + 1) {
                dsu.union(p, p + 1);
            }
            for &j in &topo.vertex_neighbors[i] {
                let o = j * layers + q;
                if j > i && ok(o) {
                    dsu.union(p, o);
                }
            }
            if q == 0 {
                dsu.union(p, virt);
            }
        }
    }
    // Heaviest component: ρ w mass, ties broken by the number of ζ₁ points.
    let mut mass = std::collections::HashMap::<usize, (f64, usize)>::new();
    for i in 0..nv {
        for q in 0..layers {
            let p = i * layers + q;
            if !ok(p) {
                continue;
            }
            let root = dsu.find(p);
            let e = mass.entry(root).or_insert((0.0, 0));
            e.0 += wgt[p] * rho_of(pts[p]);
            if q == 0 {
                e.1 += 1;
            }
        }
    }
    let total_mass: f64 = (0..np).map(|p| wgt[p] * rho_of(pts[p])).sum();
    let core_root = mass
        .iter()
        .max_by(|a, b| a.1.0.total_cmp(&b.1.0).then(a.1.1.cmp(&b.1.1)))
        .map(|(r, _)| *r);
    let in_core = |p: usize| -> bool {
        match core_root {
            Some(r) => ok(p) && dsu.find(p) == r,
            None => false,
        }
    };
    let mut movable = vec![true; nv];
    let mut core_pts = Vec::new();
    for i in 0..nv {
        for q in 0..layers {
            let p = i * layers + q;
            if in_core(p) {
                core_pts.push(p);
                if q == layers - 1 {
                    movable[i] = false;
                }
            }
        }
    }
    // Least-squares rigid motion of the core: min Σ w ρ |v − a − θ × x|².
    let mut a_mat = DMatrix::<f64>::zeros(6, 6);
    let mut b_vec = DVector::<f64>::zeros(6);
    for &p in &core_pts {
        let w = wgt[p] * rho_of(pts[p]);
        if w <= 0.0 {
            continue;
        }
        let x = pts[p];
        // columns: a_x a_y a_z θ_x θ_y θ_z; (θ × x) = (θ_y z − θ_z y, θ_z x − θ_x z, θ_x y − θ_y x)
        let g = [
            [1.0, 0.0, 0.0, 0.0, x.z, -x.y],
            [0.0, 1.0, 0.0, -x.z, 0.0, x.x],
            [0.0, 0.0, 1.0, x.y, -x.x, 0.0],
        ];
        let vv = [v[3 * p], v[3 * p + 1], v[3 * p + 2]];
        for r in 0..3 {
            for c in 0..6 {
                b_vec[c] += w * g[r][c] * vv[r];
                for c2 in 0..6 {
                    a_mat[(c, c2)] += w * g[r][c] * g[r][c2];
                }
            }
        }
    }
    let mut degenerate = false;
    let mut motion = [0.0; 6];
    let scale = (0..6).map(|i| a_mat[(i, i)]).fold(0.0, f64::max);
    if scale > 0.0 && core_root.is_some() {
        // Collinear point sets leave the rotation about their axis undetermined: the
        // normal matrix is then singular and the core is declared degenerate.
        let eig = a_mat.clone().symmetric_eigen();
        let lo = eig
            .eigenvalues
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        if lo > 1e-12 * scale {
            if let Some(ch) = nalgebra::Cholesky::new(a_mat) {
                let s = ch.solve(&b_vec);
                for i in 0..6 {
                    motion[i] = s[i];
                }
            } else {
                degenerate = true;
            }
        } else {
            degenerate = true;
        }
    } else {
        degenerate = core_root.is_some();
    }
    let a_c = DVec3::new(motion[0], motion[1], motion[2]);
    let th = DVec3::new(motion[3], motion[4], motion[5]);
    let mut v_sl = Vec::with_capacity(nv);
    let mut core_creep = 0.0_f64;
    for i in 0..nv {
        let p = i * layers + layers - 1;
        let vi = DVec3::new(v[3 * p], v[3 * p + 1], v[3 * p + 2]);
        v_sl.push(vi - a_c - th.cross(pts[p]));
    }
    for &p in &core_pts {
        let vi = DVec3::new(v[3 * p], v[3 * p + 1], v[3 * p + 2]);
        core_creep = core_creep.max((vi - a_c - th.cross(pts[p])).length());
    }
    // Diagnostics.
    let mut yielded = 0.0;
    let mut diss = 0.0;
    let mut wtot = 0.0;
    let mut ek_inv = 0.0_f64;
    let r_max = radius.iter().copied().fold(0.0, f64::max);
    for p in 0..np {
        if wgt[p] > 0.0 {
            wtot += wgt[p];
            if fdelta[p] > 0.0 {
                yielded += wgt[p];
                diss += wgt[p] * eta[p] * shear[p] * shear[p];
                ek_inv = ek_inv.max(rho_of(pts[p]) * omega * r_max * r_max / eta[p]);
            }
        }
    }
    let core_mass = core_root.and_then(|r| mass.get(&r)).map_or(0.0, |m| m.0);
    let mobile_area: f64 = (0..nv)
        .filter(|&i| movable[i])
        .map(|i| topo.vertex_solid[i])
        .sum::<f64>();
    let f_delta: Vec<f64> = (0..nv).map(|i| fdelta[i * layers + layers - 1]).collect();
    Classification {
        movable,
        f_delta,
        v_sl,
        normal,
        radius,
        core_a: a_c,
        core_theta: th,
        core_degenerate: degenerate,
        core_empty: core_root.is_none(),
        yielded_fraction: yielded / wtot.max(1e-300),
        core_mass_fraction: core_mass / total_mass.max(1e-300),
        mobile_area_fraction: mobile_area / (4.0 * std::f64::consts::PI),
        core_creep,
        plastic_dissipation: diss,
        inverse_ekman: ek_inv,
    }
}
