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
}

/// `n · T · n` of a `(xx, yy, zz, xy, xz, yz)` tensor.
pub fn quad_form(t: [f64; 6], n: DVec3) -> f64 {
    t[0] * n.x * n.x
        + t[1] * n.y * n.y
        + t[2] * n.z * n.z
        + 2.0 * (t[3] * n.x * n.y + t[4] * n.x * n.z + t[5] * n.y * n.z)
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
    })
}
