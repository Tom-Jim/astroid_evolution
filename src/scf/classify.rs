//! Yield classification, rigid core and sliding velocity (§2.8 B2–B4, B6) on the
//! classification mesh `X_Y` of (G7).
//!
//! The single stress field of the converged solution is evaluated on `X_Y`: the volume
//! nodes, probes along the inward vertex normals, and the vertices. `F_δ > 0` marks
//! yield; the unyielded points are connected through the graph `G_Y` and the heaviest
//! component is the rigid core. Everything else is mobile.

use disjoint_sets::UnionFind;
use nalgebra::{DMatrix, DVector};
use rayon::prelude::*;

use crate::common::error::SimError;
use crate::common::math::DVec3;
use crate::rheo::Material;
use crate::rtfp::mesh::BodyMesh;

use super::basis::{Want, contract_pressure, contract_velocity_rays};
use super::body::Body;
use super::domain::{VolumeSet, depth};
use super::quad::gauss_jacobi_zeta2;
use super::stokes::{Local, MaterialField, Operators, StokesSolution, eta_eval};
use super::surface::{PointGrid, SurfaceIndex};

/// Volume nodes each point of `X_Y` is joined to, along segments inside `𝒱`.
const K_NEAR: usize = 3;

/// `X_Y` and its graph `G_Y` (G7). Points are vertex-major probes (`layers` per vertex,
/// deepest first, the vertex itself last), then the volume nodes.
pub struct ClassMesh {
    pub pts: Vec<DVec3>,
    layers: usize,
    /// Quadrature weight `w` of each point (zero on probes and vertices).
    weight: Vec<f64>,
    depth: Vec<f64>,
    edges: Vec<(u32, u32)>,
    grid: PointGrid,
}

impl ClassMesh {
    /// Probes `X_i − ξ_q t_i⁻ N_i`, with `t_i⁻` the distance to the next crossing of `Γ`
    /// along `−N_i` and `ξ_q = (1 − ζ_q)/2` for the Gauss–Jacobi nodes `ζ_q`: on a sphere
    /// these are the radial probes of (B3).
    pub fn build(
        body: &Body,
        normals: &[DVec3],
        vol: &VolumeSet,
        mesh: &BodyMesh,
        index: &SurfaceIndex,
        n_radial: usize,
    ) -> Result<Self, SimError> {
        let nv = body.positions.len();
        let layers = n_radial + 1;
        let x = &body.positions;
        let eps = 1e-7 * body.r_mean();
        let mut batch = mesh.new_batch();
        for i in 0..nv {
            batch.push(x[i] - normals[i] * eps, -normals[i], None);
        }
        let hits = mesh.trace(&batch, true)?;
        let mut buf = Vec::new();
        let (zeta, _) = gauss_jacobi_zeta2(n_radial);
        let mut pts = Vec::with_capacity(nv * layers + vol.pts.len());
        for i in 0..nv {
            buf.clear();
            hits.distances_into(i, &mut buf);
            let t = buf.first().copied().unwrap_or(2.0 * body.r_mean()) + eps;
            for z in &zeta {
                pts.push(x[i] - normals[i] * (0.5 * (1.0 - z) * t));
            }
            pts.push(x[i]);
        }
        let n_probe = pts.len();
        let (probe_depth, _) = depth(index, &pts);
        pts.extend_from_slice(&vol.pts);
        let mut weight = vec![0.0; n_probe];
        weight.extend_from_slice(&vol.w);
        let mut dep = probe_depth;
        dep.extend_from_slice(&vol.depth);

        let mut edges = Vec::new();
        for i in 0..nv {
            for q in 0..layers {
                let p = (i * layers + q) as u32;
                if q + 1 < layers {
                    edges.push((p, p + 1));
                }
                for &j in &body.topo.vertex_neighbors[i] {
                    if j > i {
                        edges.push((p, (j * layers + q) as u32));
                    }
                }
            }
        }
        // Each point to its K_NEAR nearest volume nodes, kept when the segment stays in 𝒱.
        let vgrid = PointGrid::build(&vol.pts);
        let near: Vec<Vec<(u32, u32)>> = (0..pts.len())
            .into_par_iter()
            .map(|p| {
                vgrid
                    .k_nearest(pts[p], K_NEAR, |k| n_probe + k == p)
                    .iter()
                    .map(|(_, j)| (p as u32, (n_probe + j) as u32))
                    .collect()
            })
            .collect();
        let candidates: Vec<(u32, u32)> = near.into_iter().flatten().collect();
        let mut batch = mesh.new_batch();
        for &(a, b) in &candidates {
            let (pa, pb) = (pts[a as usize], pts[b as usize]);
            batch.push(pa, (pb - pa).normalize_or_zero(), None);
        }
        let hits = mesh.trace(&batch, true)?;
        for (k, &(a, b)) in candidates.iter().enumerate() {
            buf.clear();
            hits.distances_into(k, &mut buf);
            let len = (pts[b as usize] - pts[a as usize]).length();
            if buf.first().is_none_or(|t| *t >= len * (1.0 - 1e-6)) {
                edges.push((a, b));
            }
        }
        Ok(Self {
            grid: PointGrid::build(&pts),
            pts,
            layers,
            weight,
            depth: dep,
            edges,
        })
    }
}

#[derive(Clone, Debug)]
pub struct Classification {
    /// `χ_Y`: the vertex is not part of the rigid core.
    pub movable: Vec<bool>,
    /// `F_δ` at the vertices.
    pub f_delta: Vec<f64>,
    /// Velocity relative to the rigid core, at the vertices.
    pub v_sl: Vec<DVec3>,
    /// Vertex normals `N_i` (G4).
    pub normal: Vec<DVec3>,
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
    /// `Ek⁻¹ = max_Y ρ ω R_*² / η` over the yielded set.
    pub inverse_ekman: f64,
    /// Membership of each point of `X_Y` in the rigid core.
    pub in_core: Vec<bool>,
    /// Plastic rate `γ̇_p = 1{F_δ>0} γ̇` at the volume nodes (§11.2).
    pub plastic_rate: Vec<f64>,
    /// Inertial number `I_c` at the volume nodes (for `ν_c(I_c)`, §11.3).
    pub inertial_vol: Vec<f64>,
    grid: PointGrid,
}

impl Classification {
    /// `χ_Y` at `x`, read at the nearest point of `X_Y` (§9.10.2).
    pub fn movable_near(&self, x: DVec3) -> bool {
        !self.in_core[self.grid.nearest(x)]
    }
}

#[allow(clippy::too_many_arguments)]
pub fn classify(
    ops: &Operators,
    sol: &StokesSolution,
    body: &Body,
    mesh: &BodyMesh,
    index: &SurfaceIndex,
    material: &Material,
    delta_y: f64,
    n_radial: usize,
    omega: f64,
    field: Option<&dyn MaterialField>,
) -> Result<Classification, SimError> {
    let nv = body.positions.len();
    let normal = body.vertex_normals();
    let cm = match ops.class_mesh.get() {
        Some(cm) => cm,
        None => {
            let cm = ClassMesh::build(body, &normal, &ops.vol, mesh, index, n_radial)?;
            ops.class_mesh.get_or_init(|| cm)
        }
    };
    let (pts, layers) = (&cm.pts, cm.layers);
    let np = pts.len();
    let n_probe = nv * layers;
    let (mut v, mut d) = (vec![0.0; 3 * np], vec![0.0; 6 * np]);
    let mut rows = vec![(0.0, 0.0, 0.0, 0.0); np];
    // `(1 − δ_Y) τ_y` of each evaluated point, the scale of the inheritance margin.
    let mut strength = vec![0.0; np];
    let ctx = Context { ops, sol, pts, depth: &cm.depth, material, delta_y, field };
    // Volume nodes: the operator rows are on the host already.
    {
        let _span = super::timing::span("classify_contract");
        let (vv, dv) = ops.vol_velocity(&sol.u);
        v[3 * n_probe..].copy_from_slice(&vv);
        d[6 * n_probe..].copy_from_slice(&dv);
    }
    let p_vol = ops.vol_pressure(&sol.pres_c, &sol.lift);
    let vol_idx: Vec<usize> = (n_probe..np).collect();
    ctx.finish(&vol_idx, &p_vol, &mut v, &mut d, &mut rows, &mut strength);
    // Probe lines (G7): the coarse vertices exactly; a probe of a midpoint vertex by
    // inheritance when both parent probes of its layer are clearly unyielded.
    let line = |i: usize| (i * layers..(i + 1) * layers).collect::<Vec<_>>();
    let hierarchy = midpoint_parents(&body.topo);
    let coarse: Vec<usize> = match &hierarchy {
        Some((n_c, _)) => (0..*n_c).flat_map(line).collect(),
        None => (0..nv).flat_map(line).collect(),
    };
    ctx.exact(&coarse, &mut v, &mut d, &mut rows, &mut strength)?;
    if let Some((n_c, parents)) = &hierarchy {
        // Parents are coarse vertices, all evaluated above. Each probe of a midpoint
        // vertex inherits when both parent probes of its layer are clearly unyielded.
        let clear = |p: usize| rows[p].2 <= -INHERIT_MARGIN * strength[p];
        let pairs: Vec<(usize, usize, usize)> = parents
            .iter()
            .enumerate()
            .flat_map(|(k, &[a, b])| (0..layers).map(move |q| ((n_c + k) * layers + q, a * layers + q, b * layers + q)))
            .collect();
        let (inherit, refine): (Vec<_>, Vec<_>) = pairs.into_iter().partition(|&(_, pa, pb)| clear(pa) && clear(pb));
        for (p, pa, pb) in inherit {
            (0..3).for_each(|c| v[3 * p + c] = 0.5 * (v[3 * pa + c] + v[3 * pb + c]));
            (0..6).for_each(|c| d[6 * p + c] = 0.5 * (d[6 * pa + c] + d[6 * pb + c]));
            let (ra, rb) = (rows[pa], rows[pb]);
            rows[p] = (0.5 * (ra.0 + rb.0), 0.5 * (ra.1 + rb.1), 0.5 * (ra.2 + rb.2), 0.5 * (ra.3 + rb.3));
        }
        let refine: Vec<usize> = refine.into_iter().map(|r| r.0).collect();
        ctx.exact(&refine, &mut v, &mut d, &mut rows, &mut strength)?;
    }
    let _graph = super::timing::span("classify_graph");
    let fdelta: Vec<f64> = rows.iter().map(|r| r.2).collect();
    let ok = |p: usize| fdelta[p] <= 0.0;
    let mut dsu = UnionFind::new(np);
    for &(a, b) in &cm.edges {
        if ok(a as usize) && ok(b as usize) {
            dsu.union(a as usize, b as usize);
        }
    }
    let rho: Vec<f64> = (0..np)
        .map(|p| if cm.weight[p] > 0.0 { ops.vol.rho[p - nv * layers] } else { 0.0 })
        .collect();
    // Heaviest unyielded component: ρ w mass, ties broken by the number of volume nodes.
    let mut mass = std::collections::HashMap::<usize, (f64, usize)>::new();
    for p in (0..np).filter(|&p| ok(p)) {
        let e = mass.entry(dsu.find(p)).or_insert((0.0, 0));
        e.0 += cm.weight[p] * rho[p];
        e.1 += usize::from(cm.weight[p] > 0.0);
    }
    let total_mass: f64 = (0..np).map(|p| cm.weight[p] * rho[p]).sum();
    let core_root = mass
        .iter()
        .max_by(|a, b| a.1.0.total_cmp(&b.1.0).then(a.1.1.cmp(&b.1.1)))
        .map(|(r, _)| *r);
    let in_core: Vec<bool> = (0..np)
        .map(|p| core_root.is_some_and(|r| ok(p) && dsu.find(p) == r))
        .collect();
    let surface = |i: usize| i * layers + layers - 1;
    let movable: Vec<bool> = (0..nv).map(|i| !in_core[surface(i)]).collect();
    let vel = |p: usize| DVec3::new(v[3 * p], v[3 * p + 1], v[3 * p + 2]);
    let (motion, degenerate) = rigid_fit(
        (0..np).filter(|&p| in_core[p] && cm.weight[p] > 0.0),
        |p| (pts[p], vel(p), cm.weight[p] * rho[p]),
    );
    let degenerate = degenerate && core_root.is_some();
    let (a_c, th) = motion;
    let v_sl: Vec<DVec3> = (0..nv)
        .map(|i| vel(surface(i)) - a_c - th.cross(pts[surface(i)]))
        .collect();
    let core_creep = (0..np)
        .filter(|&p| in_core[p])
        .map(|p| (vel(p) - a_c - th.cross(pts[p])).length())
        .fold(0.0, f64::max);
    // Diagnostics on the volume quadrature.
    let r_star = ops.surf.coeff[0] / (4.0 * std::f64::consts::PI).sqrt();
    let (mut yielded, mut diss, mut wtot, mut ek_inv) = (0.0, 0.0, 0.0, 0.0_f64);
    for p in (0..np).filter(|&p| cm.weight[p] > 0.0) {
        let (sh, eta, fd, _) = rows[p];
        wtot += cm.weight[p];
        if fd > 0.0 {
            yielded += cm.weight[p];
            diss += cm.weight[p] * eta * sh * sh;
            ek_inv = ek_inv.max(rho[p] * omega * r_star * r_star / eta);
        }
    }
    let area = body.vertex_areas();
    // A fold from +0: the empty `f64` sum is −0, which the records would print.
    let mobile_area = (0..nv).filter(|&i| movable[i]).map(|i| area[i]).fold(0.0, |a, b| a + b);
    let core_mass = core_root.and_then(|r| mass.get(&r)).map_or(0.0, |m| m.0);
    Ok(Classification {
        f_delta: (0..nv).map(|i| fdelta[surface(i)]).collect(),
        movable,
        v_sl,
        normal,
        core_a: a_c,
        core_theta: th,
        core_degenerate: degenerate,
        core_empty: core_root.is_none(),
        yielded_fraction: yielded / wtot.max(1e-300),
        core_mass_fraction: core_mass / total_mass.max(1e-300),
        mobile_area_fraction: mobile_area / area.iter().sum::<f64>().max(1e-300),
        core_creep,
        plastic_dissipation: diss,
        inverse_ekman: ek_inv,
        in_core,
        plastic_rate: (0..ops.vol.pts.len())
            .map(|q| {
                let (sh, _, fd, _) = rows[q + nv * layers];
                if fd > 0.0 { sh } else { 0.0 }
            })
            .collect(),
        inertial_vol: (0..ops.vol.pts.len()).map(|q| rows[q + nv * layers].3).collect(),
        grid: cm.grid.clone(),
    })
}

/// A probe of a midpoint vertex inherits when both parent probes of its layer are below
/// yield by this fraction of `(1 − δ_Y) τ_y` (§2.8 B3, G7).
const INHERIT_MARGIN: f64 = 0.05;

/// For a subdivided icosphere: the vertex count of the previous level and the two
/// parents of every later (midpoint) vertex, which are its neighbours of lower index.
pub fn midpoint_parents(topo: &super::body::Topology) -> Option<(usize, Vec<[usize; 2]>)> {
    let level = topo.icosphere.filter(|l| *l >= 1)?;
    let n_c = 10 * 4usize.pow(level as u32 - 1) + 2;
    let parents = (n_c..topo.n_vertices())
        .map(|i| {
            let low: Vec<usize> = topo.vertex_neighbors[i].iter().copied().filter(|j| *j < n_c).collect();
            (low.len() == 2).then(|| [low[0], low[1]])
        })
        .collect::<Option<Vec<_>>>()?;
    Some((n_c, parents))
}

/// What the evaluation of points of `X_Y` needs.
struct Context<'a> {
    ops: &'a Operators,
    sol: &'a StokesSolution,
    pts: &'a [DVec3],
    depth: &'a [f64],
    material: &'a Material,
    delta_y: f64,
    field: Option<&'a dyn MaterialField>,
}

impl Context<'_> {
    /// `v`, `D` and `p` at the points `idx` by one GPU contraction, then `finish`.
    #[allow(clippy::type_complexity)]
    fn exact(
        &self,
        idx: &[usize],
        v: &mut [f64],
        d: &mut [f64],
        rows: &mut [(f64, f64, f64, f64)],
        strength: &mut [f64],
    ) -> Result<(), SimError> {
        if idx.is_empty() {
            return Ok(());
        }
        let xs: Vec<DVec3> = idx.iter().map(|p| self.pts[*p]).collect();
        let (ops, sol) = (self.ops, self.sol);
        let field = {
            let _span = super::timing::span("classify_contract");
            contract_velocity_rays(&ops.surf, &ops.lay, &xs, 1, Want::VD, &sol.u)
        };
        let pvals = {
            let _span = super::timing::span("classify_pressure");
            contract_pressure(&ops.surf, &ops.play, &xs, &sol.pres_c, &sol.lift)
        };
        for (k, p) in idx.iter().enumerate() {
            v[3 * p..3 * p + 3].copy_from_slice(&field.v[3 * k..3 * k + 3]);
            d[6 * p..6 * p + 6].copy_from_slice(&field.d[6 * k..6 * k + 6]);
        }
        self.finish(idx, &pvals, v, d, rows, strength);
        Ok(())
    }

    /// The A-Ξ dilatational part (§11.4), the local strength and the closure at `idx`:
    /// `(γ̇, η, F_δ, I_c)` and `(1 − δ_Y) τ_y`.
    #[allow(clippy::type_complexity)]
    fn finish(
        &self,
        idx: &[usize],
        pvals: &[f64],
        v: &mut [f64],
        d: &mut [f64],
        rows: &mut [(f64, f64, f64, f64)],
        strength: &mut [f64],
    ) {
        let m = self.material;
        if let Some(st) = &self.sol.dilatancy {
            for p in idx {
                let x = self.pts[*p];
                let ve = st.field.velocity(x);
                let de = super::evolving::dilatation::deviator(st.field.strain(x));
                (0..3).for_each(|a| v[3 * p + a] += ve[a]);
                (0..6).for_each(|c| d[6 * p + c] += de[c]);
            }
        }
        let intact: Vec<Local> = idx
            .iter()
            .map(|p| Local { c: m.cohesion_at(self.depth[*p]), ..Local::default() })
            .collect();
        let local = match self.field {
            Some(f) => f.locals(m, &idx.iter().map(|p| self.pts[*p]).collect::<Vec<_>>(), &intact),
            None => intact,
        };
        let (p_min, eps, dy) = (m.spec.p_min, self.sol.gamma_eps, self.delta_y);
        let out: Vec<((f64, f64, f64, f64), f64)> = idx
            .par_iter()
            .zip(pvals.par_iter())
            .zip(local.par_iter())
            .map(|((p, pv), l)| {
                let d6 = &d[6 * p..6 * p + 6];
                let sh = (2.0 * d6.iter().map(|x| x * x).sum::<f64>()).sqrt();
                let ptil = pv.max(p_min);
                let ev = eta_eval(m, l, sh, ptil, eps);
                // τ_y = c + μ_y p̃, with μ_y(ν) = μ_s + ξ under A-Ξ (§11.3).
                let s = (1.0 - dy) * (l.c + (m.mu_y + l.xi) * ptil);
                ((sh, ev.eta, ev.eta * sh - s, ev.inertial), s)
            })
            .collect();
        for (p, (r, s)) in idx.iter().zip(out) {
            rows[*p] = r;
            strength[*p] = s;
        }
    }
}

/// Least-squares rigid motion `min Σ w |v − a − θ × x|²` over `points`; `true` when the
/// normal matrix is singular (collinear points leave a rotation undetermined).
fn rigid_fit(
    points: impl Iterator<Item = usize>,
    at: impl Fn(usize) -> (DVec3, DVec3, f64),
) -> ((DVec3, DVec3), bool) {
    let mut a_mat = DMatrix::<f64>::zeros(6, 6);
    let mut b_vec = DVector::<f64>::zeros(6);
    for p in points {
        let (x, v, w) = at(p);
        // Columns a_x a_y a_z θ_x θ_y θ_z of v = a + θ × x.
        let g = [
            [1.0, 0.0, 0.0, 0.0, x.z, -x.y],
            [0.0, 1.0, 0.0, -x.z, 0.0, x.x],
            [0.0, 0.0, 1.0, x.y, -x.x, 0.0],
        ];
        for r in 0..3 {
            for c in 0..6 {
                b_vec[c] += w * g[r][c] * v[r];
                for c2 in 0..6 {
                    a_mat[(c, c2)] += w * g[r][c] * g[r][c2];
                }
            }
        }
    }
    let scale = (0..6).map(|i| a_mat[(i, i)]).fold(0.0, f64::max);
    let lo = a_mat.clone().symmetric_eigen().eigenvalues.min();
    if !(scale > 0.0 && lo > 1e-12 * scale) {
        return ((DVec3::ZERO, DVec3::ZERO), true);
    }
    match nalgebra::Cholesky::new(a_mat) {
        Some(ch) => {
            let s = ch.solve(&b_vec);
            ((DVec3::new(s[0], s[1], s[2]), DVec3::new(s[3], s[4], s[5])), false)
        }
        None => ((DVec3::ZERO, DVec3::ZERO), true),
    }
}
