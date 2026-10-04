//! Steps 1–4 of §10.4.4 on a frozen geometry: forward gravity (Tier 1), quasi-static
//! Stokes–pressure Picard, yield classification, and the friction-angle chain
//! (`rheo_friction`, called once on the GPU at the face centres).

use crate::common::math::DVec3;
use crate::rheo_friction::{ChainStats, NodeBatch, NodeResult, RheoGpu};

use super::body::Body;
use super::classify::{Classification, classify};
use super::error::ScfError;
use super::forward::{Forward, deviator, effective_gravity, hessian_w};
use super::harmonics::SurfaceFit;
use super::params::Project;
use super::stokes::{Loads, Operators, StokesSolution, SurfaceSet, solve};

pub struct Mechanics {
    pub omega: f64,
    pub surf: SurfaceFit,
    /// `max |R_SH − R| / R` over the vertices.
    pub fit_error: f64,
    pub ops: Operators,
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
pub fn fit_surface(project: &Project, body: &Body) -> Result<(SurfaceFit, f64), ScfError> {
    let topo = &body.topo;
    let surf = SurfaceFit::fit(
        project.stokes.l_surface,
        &topo.dirs,
        &body.radius,
        &topo.vertex_solid,
    )?;
    let err = surf.max_relative_error(&topo.dirs, &body.radius);
    if err > project.geometry.sh_fit_tol {
        return Err(ScfError::geometry(format!(
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
    velocity: &crate::rheo_friction::VelocityField,
    dirs: &[DVec3],
    g_face: &[DVec3],
    hess_w: Vec<[f64; 6]>,
    pressure: Vec<f64>,
) -> Result<(Vec<NodeResult>, ChainStats), ScfError> {
    let batch = NodeBatch {
        dirs: dirs.to_vec(),
        gravity: g_face.to_vec(),
        hessian_w: hess_w,
        pressure,
    };
    let res = gpu.evaluate(&project.material, &surf.to_rheo(), velocity, &batch)?;
    if res.stats.invalid > 0 {
        return Err(ScfError::rheology_not_converged(format!(
            "{} rheology nodes were invalid",
            res.stats.invalid
        )));
    }
    Ok((res.nodes, res.stats))
}

pub fn evaluate(
    project: &Project,
    gpu: &RheoGpu,
    body: &Body,
    fwd: &mut Forward,
    omega: f64,
    guess: Option<&[f64]>,
) -> Result<Mechanics, ScfError> {
    let tm = std::time::Instant::now();
    let lap = |what: &str| if std::env::var_os("SCF_TIMING").is_some() { eprintln!("  [{:>8.3}s] {what}", tm.elapsed().as_secs_f64()); };
    let topo = body.topo.clone();
    let density = body.density()?;
    let material = &project.material;
    let (surf, fit_error) = fit_surface(project, body)?;

    lap("fit");
    fwd.eval_faces(body)?;
    lap("faces");
    let centres = fwd.face.points.clone();
    let dirs: Vec<DVec3> = centres
        .iter()
        .map(|c| c.try_normalize().unwrap_or(DVec3::new(0.0, 0.0, 1.0)))
        .collect();
    let surface = SurfaceSet::build(&surf, dirs.clone(), topo.face_solid.clone());
    let ops = Operators::build(&surf, &density, surface, &project.stokes)?;

    lap("operators");
    // Interior fields at the Stokes quadrature points.
    let field = fwd.eval_field(&ops.vol.pts, true)?;
    lap("field");
    let g_vol: Vec<DVec3> = ops
        .vol
        .pts
        .iter()
        .zip(&field.grad_u)
        .map(|(x, g)| effective_gravity(*g, *x, omega))
        .collect();
    let tide: Vec<[f64; 6]> = field
        .hess_u
        .iter()
        .map(|h| {
            let d = deviator(hessian_w(*h, omega));
            d.map(|v| material.tide_coef * v)
        })
        .collect();

    // Face-centre fields.
    let g_face: Vec<DVec3> = fwd
        .face
        .grad_u
        .iter()
        .zip(&centres)
        .map(|(g, x)| effective_gravity(*g, *x, omega))
        .collect();
    let hess_w: Vec<[f64; 6]> = fwd.face.hess_u.iter().map(|h| hessian_w(*h, omega)).collect();
    let tide_nn: Vec<f64> = hess_w
        .iter()
        .zip(&ops.surface.normals)
        .map(|(h, n)| material.tide_coef * quad_form(deviator(*h), *n))
        .collect();

    let ld = Loads {
        g_eff: &g_vol,
        tide: &tide,
        tide_nn: &tide_nn,
        g_const: project.g,
    };
    let sol = solve(&ops, &ld, &project.stokes, material, guess)?;

    lap("stokes");
    let class = classify(
        &ops,
        &sol,
        &surf,
        &topo,
        material,
        project.yielding.delta_y,
        project.stokes.n_radial,
        &|x| density.rho(x),
        omega,
        material.spec.p_min,
    );

    lap("classify");
    let velocity = ops.lay.to_rheo(&sol.u);
    let (rheo, rheo_stats) = face_chain(
        gpu,
        project,
        &surf,
        &velocity,
        &dirs,
        &g_face,
        hess_w,
        sol.p_surface.clone(),
    )?;
    lap("rheo");
    Ok(Mechanics {
        omega,
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
