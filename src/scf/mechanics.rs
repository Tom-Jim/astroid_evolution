//! Steps 1–4 of §10.4.4 on a frozen geometry: forward gravity (Tier 1), quasi-static
//! Stokes–pressure Picard, yield classification, and the friction-angle chain
//! (`rheo_friction`, called once on the GPU at the face centres).

use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::{Arc, Mutex};

use crate::common::math::DVec3;
use crate::rheo::{ChainStats, NodeBatch, NodeResult, RheoGpu};

use super::body::Body;
use super::classify::{Classification, classify};
use super::forward::{Forward, Loading, deviator, effective_gravity, hessian_w};
use super::harmonics::SurfaceFit;
use super::params::Project;
use super::stokes::{Loads, Operators, StokesSolution, SurfaceSet, solve};
use crate::common::error::SimError;

pub struct Mechanics {
    pub omega: f64,
    pub surf: SurfaceFit,
    /// `max |R_SH − R| / R` over the vertices.
    pub fit_error: f64,
    pub ops: Arc<Operators>,
    pub sol: StokesSolution,
    pub class: Classification,
    /// `g_eff` at the face centres.
    pub g_face: Vec<DVec3>,
    /// Friction-angle chain at the face centres.
    pub rheo: Vec<NodeResult>,
    pub rheo_stats: ChainStats,
    /// Cauchy stress `σ = −p I + T_tide + 2η D` at the face centres (Pa,
    /// `(xx, yy, zz, xy, xz, yz)`).
    pub surface_stress: Vec<[f64; 6]>,
}

/// `n · T · n` of a `(xx, yy, zz, xy, xz, yz)` tensor.
pub fn quad_form(t: [f64; 6], n: DVec3) -> f64 {
    t[0] * n.x * n.x
        + t[1] * n.y * n.y
        + t[2] * n.z * n.z
        + 2.0 * (t[3] * n.x * n.y + t[4] * n.x * n.z + t[5] * n.y * n.z)
}

/// Cartesian `D` from its chart components: `D_ab = e_a·D·e_b`, `D_nα = n·D·e_α`,
/// `D_nn = n·D·n`. With the dual vectors `e^a = g^{ab} e_b`,
/// `D = D_ab e^a⊗e^b + D_nα (n⊗e^α + e^α⊗n) + D_nn n⊗n`. `metric` is `(g11, g12, g22)`.
pub fn cartesian_strain(
    e: [DVec3; 2],
    n: DVec3,
    metric: [f64; 3],
    d_ab: [f64; 3],
    d_na: [f64; 2],
    d_nn: f64,
) -> [[f64; 3]; 3] {
    let [g11, g12, g22] = metric;
    let det = g11 * g22 - g12 * g12;
    let mut d = [[0.0f64; 3]; 3];
    if !det.is_finite() || det.abs() < 1e-300 {
        return d;
    }
    let (i11, i12, i22) = (g22 / det, -g12 / det, g11 / det);
    let up = [e[0] * i11 + e[1] * i12, e[0] * i12 + e[1] * i22];
    let tangential = [[d_ab[0], d_ab[1]], [d_ab[1], d_ab[2]]];
    let mut add = |a: DVec3, b: DVec3, w: f64| {
        let (a, b) = (a.to_array(), b.to_array());
        for i in 0..3 {
            for j in 0..3 {
                d[i][j] += w * a[i] * b[j];
            }
        }
    };
    for a in 0..2 {
        for b in 0..2 {
            add(up[a], up[b], tangential[a][b]);
        }
        add(n, up[a], d_na[a]);
        add(up[a], n, d_na[a]);
    }
    add(n, n, d_nn);
    d
}

/// Cartesian `σ = −p I + T_tide + 2η D` at one face centre (Pa, `(xx, yy, zz, xy, xz, yz)`).
pub fn surface_cauchy(node: &NodeResult, pressure: f64, tide: [f64; 6]) -> [f64; 6] {
    let d = cartesian_strain(
        [node.e1, node.e2],
        node.normal,
        node.metric,
        node.d_ab,
        node.d_na,
        node.d_nn,
    );
    let two_eta = 2.0 * node.eta;
    [
        -pressure + tide[0] + two_eta * d[0][0],
        -pressure + tide[1] + two_eta * d[1][1],
        -pressure + tide[2] + two_eta * d[2][2],
        tide[3] + two_eta * d[0][1],
        tide[4] + two_eta * d[0][2],
        tide[5] + two_eta * d[1][2],
    ]
}

/// Harmonic fit of the radii (the Stokes domain and the `rheo_friction` surface).
pub fn fit_surface(project: &Project, body: &Body) -> Result<(SurfaceFit, f64), SimError> {
    let topo = &body.topo;
    let surf = SurfaceFit::fit(
        project.stokes.l_surface,
        &topo.dirs,
        &body.radius,
        &topo.vertex_solid,
    )?;
    let err = surf.max_relative_error(&topo.dirs, &body.radius);
    if err > project.geometry.sh_fit_tol {
        return Err(SimError::geometry(format!(
            "harmonic surface of degree {} misses the polyhedron by {err:.3e} > geometry.sh_fit_tol {:.3e}",
            project.stokes.l_surface, project.geometry.sh_fit_tol
        )));
    }
    Ok((surf, err))
}

/// Run the friction-angle chain at the face centres.
pub fn face_chain(
    gpu: &RheoGpu,
    project: &Project,
    surf: &SurfaceFit,
    velocity: &crate::rheo::VelocityField,
    dirs: &[DVec3],
    g_face: &[DVec3],
    hess_w: Vec<[f64; 6]>,
    pressure: Vec<f64>,
) -> Result<(Vec<NodeResult>, ChainStats), SimError> {
    let batch = NodeBatch {
        dirs: dirs.to_vec(),
        gravity: g_face.to_vec(),
        hessian_w: hess_w,
        pressure,
    };
    let res = gpu.evaluate(&project.material, &surf.to_rheo(), velocity, &batch)?;
    if res.stats.invalid > 0 {
        return Err(SimError::rheology_not_converged(format!(
            "{} rheology nodes were invalid",
            res.stats.invalid
        )));
    }
    Ok((res.nodes, res.stats))
}

/// The Stokes operators depend on the geometry and the density only (§10.4.2), not on
/// the spin or the loads. A spin-only step keeps both, so the last set is reused.
fn operators(
    project: &Project,
    body: &Body,
    surf: &SurfaceFit,
    density: &crate::common::density::CauchyDensity,
    dirs: &[DVec3],
) -> Result<Arc<Operators>, SimError> {
    static LAST: Mutex<Option<(u64, Arc<Operators>)>> = Mutex::new(None);
    let mut h = DefaultHasher::new();
    let bits = |h: &mut DefaultHasher, v: f64| v.to_bits().hash(h);
    body.radius.iter().for_each(|r| bits(&mut h, *r));
    bits(&mut h, body.bulk);
    for k in &body.kernels {
        k.center.to_array().iter().for_each(|v| bits(&mut h, *v));
        k.metric.iter().for_each(|v| bits(&mut h, *v));
        bits(&mut h, k.weight);
        bits(&mut h, k.alpha);
    }
    dirs.iter()
        .for_each(|d| d.to_array().iter().for_each(|v| bits(&mut h, *v)));
    format!("{:?}", project.stokes).hash(&mut h);
    let key = h.finish();
    let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((k, ops)) = last.as_ref() {
        if *k == key {
            return Ok(ops.clone());
        }
    }
    let _span = super::timing::span("stokes_operators");
    let surface = SurfaceSet::build(surf, dirs.to_vec(), body.topo.face_solid.clone());
    let ops = Arc::new(Operators::build(surf, density, surface, &project.stokes)?);
    *last = Some((key, ops.clone()));
    Ok(ops)
}

pub fn evaluate(
    project: &Project,
    gpu: &RheoGpu,
    body: &Body,
    fwd: &mut Forward,
    load: Loading,
    guess: Option<&[f64]>,
) -> Result<Mechanics, SimError> {
    let topo = body.topo.clone();
    let density = body.density()?;
    let material = &project.material;
    let (surf, fit_error) = {
        let _span = super::timing::span("surface_fit");
        fit_surface(project, body)?
    };

    fwd.eval_faces(body)?;
    let centres = fwd.face.points.clone();
    let dirs: Vec<DVec3> = centres
        .iter()
        .map(|c| c.try_normalize().unwrap_or(DVec3::new(0.0, 0.0, 1.0)))
        .collect();
    let ops = operators(project, body, &surf, &density, &dirs)?;

    // Interior fields at the Stokes quadrature points.
    let field = match ops.vol_field.get() {
        Some(field) => field,
        None => {
            let field = fwd.eval_field(&ops.vol.pts, true)?;
            ops.vol_field.get_or_init(|| field)
        }
    };
    let _loads = super::timing::span("effective_loads");
    let g_vol: Vec<DVec3> = ops
        .vol
        .pts
        .iter()
        .zip(&field.grad_u)
        .map(|(x, g)| effective_gravity(*g, *x, load))
        .collect();
    let tide: Vec<[f64; 6]> = field
        .hess_u
        .iter()
        .map(|h| {
            let d = deviator(hessian_w(*h, load));
            d.map(|v| material.tide_coef * v)
        })
        .collect();

    // Face-centre fields.
    let g_face: Vec<DVec3> = fwd
        .face
        .grad_u
        .iter()
        .zip(&centres)
        .map(|(g, x)| effective_gravity(*g, *x, load))
        .collect();
    let hess_w: Vec<[f64; 6]> = fwd
        .face
        .hess_u
        .iter()
        .map(|h| hessian_w(*h, load))
        .collect();
    let tide_nn: Vec<f64> = hess_w
        .iter()
        .zip(&ops.surface.normals)
        .map(|(h, n)| material.tide_coef * quad_form(deviator(*h), *n))
        .collect();
    drop(_loads);

    let ld = Loads {
        g_eff: &g_vol,
        tide: &tide,
        tide_nn: &tide_nn,
        g_const: project.g,
    };
    let sol = {
        let _span = super::timing::span("stokes");
        solve(&ops, &ld, &project.stokes, material, guess)?
    };

    let class = {
        let _span = super::timing::span("classify");
        classify(
            &ops,
            &sol,
            &surf,
            &topo,
            material,
            project.yielding.delta_y,
            project.stokes.n_radial,
            &|x| density.rho(x),
            load.omega,
            material.spec.p_min,
        )
    };

    let velocity = ops.lay.to_rheo(&sol.u);
    let tide_face: Vec<[f64; 6]> = hess_w
        .iter()
        .map(|h| deviator(*h).map(|v| material.tide_coef * v))
        .collect();
    let (rheo, rheo_stats) = {
        let _span = super::timing::span("rheology");
        face_chain(
            gpu,
            project,
            &surf,
            &velocity,
            &dirs,
            &g_face,
            hess_w,
            sol.p_surface.clone(),
        )?
    };
    let surface_stress = rheo
        .iter()
        .zip(&sol.p_surface)
        .zip(&tide_face)
        .map(|((node, p), tide)| surface_cauchy(node, *p, *tide))
        .collect();
    Ok(Mechanics {
        omega: load.omega,
        surf,
        fit_error,
        ops,
        sol,
        class,
        g_face,
        rheo,
        rheo_stats,
        surface_stress,
    })
}
