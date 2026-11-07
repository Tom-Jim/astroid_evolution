//! Steps 1–4 of §10.3.4 on a frozen geometry: forward gravity (Tier 1), the quasi-static
//! Stokes–pressure solve, yield classification, and the friction-angle chain at the face
//! centres (§10.2), all on the general domain of hypothesis (G).

use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::Arc;

use rayon::prelude::*;

use crate::common::math::DVec3;
use crate::rheo::closure::{ClosureInput, closure};
use crate::rheo::model::local_frame;
use crate::rheo::{ChainStats, NodeResult};

use super::basis::{Want, contract_velocity_rays};
use super::body::Body;
use super::classify::{Classification, classify};
use super::domain::SurfacePoints;
use super::evolving::dilatation::{DilatationSpace, Source};
use super::evolving::markers::Nonlocal;
use super::forward::{Forward, Loading, deviator, effective_gravity, hessian_w};
use super::harmonics::SurfaceFit;
use super::params::Project;
use super::stokes::{Guess, Loads, MaterialField, Operators, StokesSolution, Strengths, solve};
use super::surface::SurfaceIndex;
use crate::common::error::SimError;
use crate::common::time::wall_seconds;

/// Strength of the friction chain at a face centre (Pa): the continuum cohesion `c`, the
/// shallow cohesion `c_Γ` of the repose proxy, and the static dilatancy shift `ξ`.
#[derive(Clone, Copy, Debug)]
pub struct FaceLaw {
    pub c: f64,
    pub shallow: f64,
    pub xi: f64,
}

impl FaceLaw {
    /// The surface value `c_s` of A-C2 for both cohesions and no shift.
    pub fn intact(m: &crate::rheo::Material) -> Self {
        let c = m.spec.cohesion_surface;
        Self { c, shallow: c, xi: 0.0 }
    }
}

/// What Chapter 11 contributes to one evaluation; the default is the reference closure.
#[derive(Clone, Copy, Default)]
pub struct Evolution<'a> {
    /// Local strengths from the markers (A-ϖ, A-Ξ).
    pub field: Option<&'a dyn MaterialField>,
    /// Surface damage `ϖ_Γ` per vertex (A-Θ).
    pub surface_damage: Option<&'a [f64]>,
}

pub struct Mechanics {
    pub omega: f64,
    /// The enclosing-ball fit `R ≡ R_*` of the basis (G1).
    pub surf: SurfaceFit,
    pub ops: Arc<Operators>,
    /// Closest-point and ray queries on `Γ` of this geometry.
    pub index: Arc<SurfaceIndex>,
    pub sol: StokesSolution,
    pub class: Classification,
    /// `g_eff` at the face centres.
    pub g_face: Vec<DVec3>,
    /// Friction-angle chain at the face centres, with the face laws it used.
    pub rheo: Vec<NodeResult>,
    pub face_law: Vec<FaceLaw>,
    /// Local strengths of the solve.
    pub strength: Strengths,
    pub rheo_stats: ChainStats,
    /// Cauchy stress `σ = −p I + T_tide + 2η D` at the face centres (Pa,
    /// `(xx, yy, zz, xy, xz, yz)`).
    pub surface_stress: Vec<[f64; 6]>,
    /// A-Λ: velocity coefficients of the ratchet `v_r` (§10.3.7), unmasked; empty for none.
    pub ratchet: Vec<f64>,
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

/// The friction chain of §10.2 at one face centre from the analytic strain `D` there
/// (§10.1): `M` is the tangential block of `D`, and the traction condition gives `D_na`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn face_node(
    m: &crate::rheo::Material,
    x: DVec3,
    n: DVec3,
    v: DVec3,
    d6: &[f64],
    g: DVec3,
    tide: [f64; 6],
    pressure: f64,
    law: FaceLaw,
    eps_v: f64,
) -> NodeResult {
    let s = std::f64::consts::FRAC_1_SQRT_2;
    let dm = [
        [d6[0], d6[3] * s, d6[4] * s],
        [d6[3] * s, d6[1], d6[5] * s],
        [d6[4] * s, d6[5] * s, d6[2]],
    ];
    let (e1, e2) = local_frame(n);
    let quad = |a: DVec3, b: DVec3| {
        let (a, b) = (a.to_array(), b.to_array());
        (0..3).map(|i| (0..3).map(|j| a[i] * dm[i][j] * b[j]).sum::<f64>()).sum::<f64>()
    };
    let d_ab = [quad(e1, e1), quad(e1, e2), quad(e2, e2)];
    let tr = d_ab[0] + d_ab[2];
    // §10.1.1 with `tr D = ε̇_v` (A-Ξ, §11.4): D_nn = ε̇_v − g^{ab} D_ab and
    // γ̇² = 2[D_nn² + D_ab D^ab + 2 D_na D^na − ε̇_v²/3]; the D_na part comes from the tide.
    let d_nn = eps_v - tr;
    let s0 = 2.0
        * (d_nn * d_nn + d_ab[0] * d_ab[0] + 2.0 * d_ab[1] * d_ab[1] + d_ab[2] * d_ab[2]
            - eps_v * eps_v / 3.0);
    let tm = sym(tide);
    let t = [e1, e2].map(|e| n.dot(tm * e));
    let tide_frob = (tide[..3].iter().map(|x| x * x).sum::<f64>()
        + 2.0 * tide[3..].iter().map(|x| x * x).sum::<f64>())
    .sqrt();
    let out = closure(
        m,
        &ClosureInput {
            s0,
            tg: t[0] * t[0] + t[1] * t[1],
            pressure,
            g_dot_n: g.dot(n),
            g_norm: g.length(),
            g_cross_n: g.cross(n).length(),
            tide_frob,
            cohesion: law.c,
            shallow_cohesion: law.shallow,
            xi: law.xi,
        },
    );
    let d_na = t.map(|ta| if out.eta > 0.0 { -ta / (2.0 * out.eta) } else { 0.0 });
    NodeResult {
        radius: x.length(),
        normal: n,
        area_density: 0.0,
        velocity: v,
        e1,
        e2,
        metric: [1.0, 0.0, 1.0],
        d_ab,
        d_nn,
        d_na,
        shear: out.shear,
        inertial: out.inertial,
        mu_d: out.mu_d,
        eta: out.eta,
        tau_eq: out.tau_eq,
        p_tilde: out.p_tilde,
        p_star: out.p_star,
        p_ref: out.p_ref,
        mu_proj: out.mu_proj,
        mu_eff: out.mu_eff,
        phi: out.phi,
        sin2_phi: out.sin2_phi,
        sin_phi: out.sin2_phi.sqrt(),
        sin_slope: out.sin_slope,
        slack: out.slack,
        overhang_margin: out.overhang_margin,
        tide_frob,
        v_r: v.dot(x.normalize_or_zero()),
        v_n: v.dot(n),
        s0,
        tg: t[0] * t[0] + t[1] * t[1],
        iterations: out.iterations,
        converged: out.converged,
    }
}

/// `(xx, yy, zz, xy, xz, yz)` as a matrix.
pub fn sym(t: [f64; 6]) -> crate::common::math::DMat3 {
    crate::common::math::sym_mat(t)
}

/// Run the friction chain at every face centre.
fn face_chain(
    project: &Project,
    ops: &Operators,
    sol: &StokesSolution,
    g_face: &[DVec3],
    tide_face: &[[f64; 6]],
    pressure: &[f64],
    laws: &[FaceLaw],
) -> Result<(Vec<NodeResult>, ChainStats), SimError> {
    let start = wall_seconds();
    let f = &ops.faces;
    let vd = contract_velocity_rays(&ops.surf, &ops.lay, &f.pts, 1, Want::VD, &sol.u);
    let nodes: Vec<NodeResult> = (0..f.pts.len())
        .into_par_iter()
        .map(|k| {
            // A-Ξ: the total velocity and strain, with ε̇_v = tr D (§11.4).
            let (de, e) = sol.d_eps(f.pts[k]);
            let v = DVec3::new(vd.v[3 * k], vd.v[3 * k + 1], vd.v[3 * k + 2]) + sol.v_eps(f.pts[k]);
            let d6: [f64; 6] = std::array::from_fn(|c| vd.d[6 * k + c] + de[c]);
            face_node(
                &project.material,
                f.pts[k],
                f.normals[k],
                v,
                &d6,
                g_face[k],
                tide_face[k],
                pressure[k],
                laws[k],
                e,
            )
        })
        .collect();
    let stats = ChainStats {
        max_iterations: nodes.iter().map(|n| n.iterations).max().unwrap_or(0),
        unconverged: nodes.iter().filter(|n| !n.converged).count() as u32,
        invalid: nodes.iter().filter(|n| !n.slack.is_finite()).count() as u32,
        seconds: wall_seconds() - start,
    };
    if stats.invalid > 0 {
        return Err(SimError::rheology_not_converged(format!(
            "{} rheology nodes were invalid",
            stats.invalid
        )));
    }
    Ok((nodes, stats))
}

/// `ε_R` of §8.1: a new basis scale is `R_ref = (1 + ε_R) R_*`, kept while
/// `R_* ≤ R_ref ≤ (1 + 2 ε_R) R_*`, so a growing body keeps it for a while.
const SCALE_SLACK: f64 = 0.02;

/// The last geometry of one run: a spin-only step takes it as it is, and the next
/// geometry its basis scale and node rows (§8.1). It belongs to the run, so a run never
/// depends on what an earlier run in the same process left behind.
#[derive(Default)]
pub struct GeometryCache(Option<(u64, Arc<SurfaceIndex>, Arc<Operators>)>);

/// The geometry of the step: the surface index and the Stokes operators. They depend on
/// the vertices and the density only, not on the spin, so a spin-only step reuses them.
pub fn geometry(
    last: &mut GeometryCache,
    project: &Project,
    body: &Body,
    fwd: &Forward,
) -> Result<(Arc<SurfaceIndex>, Arc<Operators>), SimError> {
    let mut h = DefaultHasher::new();
    let bits = |h: &mut DefaultHasher, v: f64| v.to_bits().hash(h);
    body.positions
        .iter()
        .for_each(|p| p.to_array().iter().for_each(|v| bits(&mut h, *v)));
    bits(&mut h, body.bulk);
    for k in &body.kernels {
        k.center.to_array().iter().for_each(|v| bits(&mut h, *v));
        k.metric.iter().for_each(|v| bits(&mut h, *v));
        bits(&mut h, k.weight);
        bits(&mut h, k.alpha);
    }
    format!("{:?}", project.stokes).hash(&mut h);
    let key = h.finish();
    if let Some((k, index, ops)) = last.0.as_ref() {
        if *k == key {
            return Ok((index.clone(), ops.clone()));
        }
    }
    // Only the volume rows and the basis scale of the stale geometry are kept; the rest
    // (its resident classification rows are GBs) is released before the new operators
    // are built. A geometry another holder still uses is not reused.
    let reuse = last
        .0
        .take()
        .and_then(|(_, _, ops)| Arc::try_unwrap(ops).ok())
        .map(Operators::into_reuse);
    let index = {
        let _span = super::timing::span("stokes_operators");
        Arc::new(SurfaceIndex::build(&body.positions, &body.topo.faces))
    };
    // §8.1: the Galerkin spaces do not depend on the scale of ζ, so the previous R_ref
    // stays while it still encloses Γ and is not too large for it.
    let r_star = body.r_star();
    let reuse = reuse.filter(|r| r_star <= r.r_ref && r.r_ref <= (1.0 + 2.0 * SCALE_SLACK) * r_star);
    let surf = SurfaceFit::sphere(reuse.as_ref().map_or((1.0 + SCALE_SLACK) * r_star, |r| r.r_ref));
    // §8.1 (G1): one flux mode per sealed cavity, centred at a cavity interior
    // point of the body mesh.
    let cavity_centres = body.mesh()?.cavity_interior_points();
    let ops = Arc::new(Operators::build(
        &surf,
        &body.density()?,
        fwd.body_mesh(),
        &index,
        SurfacePoints::faces(&body.face_geometry()),
        &project.stokes,
        reuse,
        &cavity_centres,
    )?);
    last.0 = Some((key, index.clone(), ops.clone()));
    Ok((index, ops))
}

/// The loads of one geometry under `load`: `g_eff` and `T_tide` at the volume nodes, the
/// boundary rule and the face centres, with the intact strength profile (§4.3a).
pub struct LoadSet {
    pub g_vol: Vec<DVec3>,
    pub tide: Vec<[f64; 6]>,
    pub g_bnd: Vec<DVec3>,
    pub tide_bnd: Vec<[f64; 6]>,
    pub g_face: Vec<DVec3>,
    pub tide_face: Vec<[f64; 6]>,
    pub tide_nn: Vec<f64>,
    pub strength: Strengths,
}

impl LoadSet {
    /// Needs the face fields of `fwd` on this geometry (`Forward::eval_faces`).
    pub fn of(project: &Project, ops: &Operators, fwd: &Forward, load: Loading) -> Result<Self, SimError> {
        let material = &project.material;
        let vol = match ops.vol_field.get() {
            Some(f) => f,
            None => {
                let f = fwd.eval_field(&ops.vol.pts, true)?;
                ops.vol_field.get_or_init(|| f)
            }
        };
        let bnd = match ops.bnd_field.get() {
            Some(f) => f,
            None => {
                let f = fwd.eval_surface(&ops.bnd.pts)?;
                ops.bnd_field.get_or_init(|| f)
            }
        };
        let _span = super::timing::span("effective_loads");
        let tide_of = |h: &[f64; 6]| deviator(hessian_w(*h, load)).map(|v| material.tide_coef * v);
        let g_of = |pts: &[DVec3], g: &[DVec3]| -> Vec<DVec3> {
            pts.iter().zip(g).map(|(x, g)| effective_gravity(*g, *x, load)).collect()
        };
        let tide_face: Vec<[f64; 6]> = fwd.face.hess_u.iter().map(tide_of).collect();
        Ok(Self {
            g_vol: g_of(&ops.vol.pts, &vol.grad_u),
            tide: vol.hess_u.iter().map(tide_of).collect(),
            g_bnd: g_of(&ops.bnd.pts, &bnd.grad_u),
            tide_bnd: bnd.hess_u.iter().map(tide_of).collect(),
            g_face: g_of(&ops.faces.pts, &fwd.face.grad_u),
            tide_nn: tide_face.iter().zip(&ops.faces.normals).map(|(t, n)| quad_form(*t, *n)).collect(),
            tide_face,
            strength: Strengths::profile(ops, material),
        })
    }

    pub fn loads(&self, g_const: f64) -> Loads<'_> {
        Loads {
            g_eff: &self.g_vol,
            tide: &self.tide,
            g_bnd: &self.g_bnd,
            tide_bnd: &self.tide_bnd,
            tide_nn: &self.tide_nn,
            g_const,
            strength: &self.strength,
            dilatancy: None,
        }
    }

    /// The cohesion `c` at the volume nodes.
    pub fn cohesion(&self) -> Vec<f64> {
        self.strength.vol.iter().map(|l| l.c).collect()
    }
}

/// The certified bracket of `L*` (§10.0.12.3) of the deep state `mech` under `load`, with the
/// strength `c + μ₂ p̃` of Proposition 10.0.12.1.
pub fn limit_bracket(project: &Project, mech: &Mechanics, fwd: &Forward, load: Loading) -> Result<super::stokes::limit::Bracket, SimError> {
    let mut set = LoadSet::of(project, &mech.ops, fwd, load)?;
    set.strength = mech.strength.clone();
    let m = &project.material;
    let data = super::stokes::limit::data(&mech.ops, &set.loads(project.g), &project.stokes, m)?;
    super::stokes::limit::bracket(&mech.ops, &data, &set.cohesion(), m.spec.mu_2, m.spec.p_min, 1e-3)
}

/// The mean bracket of `L*` (as [`limit_bracket`]) and the brackets of the tidal cycle `el`
/// about it (Proposition 10.3.7.5), sampled at `phases`.
pub fn limit_cycle(
    project: &Project,
    mech: &Mechanics,
    fwd: &Forward,
    load: Loading,
    el: &super::stokes::elastic::ElasticCycle,
    phases: usize,
) -> Result<(super::stokes::limit::Bracket, super::stokes::elastic::CycleBracket), SimError> {
    let mut set = LoadSet::of(project, &mech.ops, fwd, load)?;
    set.strength = mech.strength.clone();
    let m = &project.material;
    let data = super::stokes::limit::data(&mech.ops, &set.loads(project.g), &project.stokes, m)?;
    let cohesion = set.cohesion();
    let b = super::stokes::limit::bracket(&mech.ops, &data, &cohesion, m.spec.mu_2, m.spec.p_min, 1e-3)?;
    let c = super::stokes::elastic::cycle_bracket(&mech.ops, &b, &data.load, el, 1.0, &cohesion, m.spec.mu_2, &data.p0, m.spec.p_min, phases);
    Ok((b, c))
}

pub fn evaluate(
    cache: &mut GeometryCache,
    project: &Project,
    body: &Body,
    fwd: &mut Forward,
    load: Loading,
    guess: Option<&Guess>,
    ev: Evolution,
) -> Result<Mechanics, SimError> {
    let material = &project.material;
    let (index, ops) = geometry(cache, project, body, fwd)?;
    fwd.eval_faces(body)?;
    let mut set = LoadSet::of(project, &ops, fwd, load)?;
    // Local strengths: the intact profile, then the markers of Chapter 11.
    if let Some(f) = ev.field {
        set.strength = Strengths {
            vol: f.locals(material, &ops.vol.pts, &set.strength.vol),
            bnd: f.locals(material, &ops.bnd.pts, &set.strength.bnd),
            face: f.locals(material, &ops.faces.pts, &set.strength.face),
        };
    }
    let face_law = face_laws(material, body, &set.strength.face, ev);
    // A-Ξ: the dilatancy source on this geometry (§11.4).
    let source = match &project.dilatancy {
        Some(d) => {
            let r_star = body.r_star();
            let ell = project
                .markers
                .as_ref()
                .expect("A-Ξ needs the markers (checked by the project)")
                .nonlocal_length(r_star, project.stokes.l_velocity);
            Some(Source {
                space: DilatationSpace::build(&ops.vol.pts, &ops.vol.w, d.degree, d.regularization, r_star)?,
                nonlocal: Nonlocal::new(&ops.vol.pts, &ops.vol.w, ell, &index),
                params: d,
                delta_y: project.yielding.delta_y,
                pts: &ops.vol.pts,
                w: &ops.vol.w,
            })
        }
        None => None,
    };

    let ld = Loads {
        dilatancy: source.as_ref().map(|s| s as &dyn super::stokes::DilatancySource),
        ..set.loads(project.g)
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
            body,
            fwd.body_mesh(),
            &index,
            material,
            project.yielding.delta_y,
            project.stokes.n_radial,
            load.omega,
            ev.field,
        )?
    };
    let (rheo, rheo_stats) = {
        let _span = super::timing::span("rheology");
        face_chain(project, &ops, &sol, &set.g_face, &set.tide_face, &sol.p_surface, &face_law)?
    };
    let surface_stress = rheo
        .iter()
        .zip(&sol.p_surface)
        .zip(&set.tide_face)
        .map(|((node, p), tide)| surface_cauchy(node, *p, *tide))
        .collect();
    Ok(Mechanics {
        omega: load.omega,
        surf: ops.surf.clone(),
        ops,
        index,
        sol,
        class,
        g_face: set.g_face,
        rheo,
        rheo_stats,
        surface_stress,
        face_law,
        strength: set.strength,
        ratchet: Vec::new(),
    })
}

/// The face laws of the step: the continuum strength at the face centres, and the shallow
/// cohesion `c_Γ = (1 − ϖ)(1 − ϖ_Γ) c_s` as the mean of its vertex factors (§11.6).
fn face_laws(
    m: &crate::rheo::Material,
    body: &Body,
    face: &[super::stokes::Local],
    ev: Evolution,
) -> Vec<FaceLaw> {
    let c_s = m.spec.cohesion_surface;
    let nv = body.positions.len();
    // (1 − ϖ) at the vertices: the field's cohesion over the intact c_s.
    let bulk: Vec<f64> = match ev.field {
        Some(f) if c_s > 0.0 => {
            let intact = vec![super::stokes::Local { c: c_s, ..Default::default() }; nv];
            f.locals(m, &body.positions, &intact).iter().map(|l| l.c / c_s).collect()
        }
        _ => vec![1.0; nv],
    };
    let keep = |v: usize| bulk[v] * (1.0 - ev.surface_damage.map_or(0.0, |d| d[v]));
    body.topo
        .faces
        .iter()
        .zip(face)
        .map(|(t, l)| FaceLaw {
            c: l.c,
            shallow: c_s * t.iter().map(|v| keep(*v as usize)).sum::<f64>() / 3.0,
            xi: l.xi,
        })
        .collect()
}
