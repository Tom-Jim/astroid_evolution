//! Reconstruction of a pre-evolution state from a present-day one.
//!
//! The forward model `F` (the quasi-static evolution of §10.4 under an exogenous
//! schedule) maps an initial body `X₀` to a final one. Given the observed present
//! body `X_p`, the pre-evolution state is the fixed point of
//!
//!   X₀⁽ᵏ⁺¹⁾ = X₀⁽ᵏ⁾ + (X_p − F(X₀⁽ᵏ⁾)),
//!
//! applied to the ray radii, the loose-layer surface density and the core
//! parameters (centre, weight, metric). It converges when `F` is close to a
//! translation of its argument on the states visited, which is the regime of a
//! quasi-static history: the mobile layers move, the rigid interior does not.
//!
//! The present density comes from the measured mass and the degree-2 gravity:
//! a uniform matrix plus one central core, with `ρ_b` and the core weight
//! solved so that the mass and `C₂₀` match.

use super::body::Body;
use super::evolve::{self, RunOptions, State, StepFields, StepRecord};
use super::forward::Forward;
use super::forward::Moments;
use super::frame::undo_frame;
use super::params::Project;
use super::spin::SpinSchedule;
use crate::common::density::CauchyKernel;
use crate::common::error::SimError;
use crate::common::math::{DMat3, DVec3, sym_mat};

/// Unnormalized degree-2 coefficients at `r_ref`, in the principal frame:
/// `C₂₀ = −(C − (A + B)/2)/(M r²)`, `C₂₂ = (B − A)/(4 M r²)` with `A ≤ B ≤ C`.
#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct Degree2 {
    pub c20: f64,
    pub c22: f64,
}

pub fn degree2(m: &Moments, r_ref: f64) -> Degree2 {
    // Inertia about the centre of mass (parallel axis).
    let c = m.centroid;
    let shift = (DMat3::IDENTITY * c.length_squared()
        - DMat3::from_cols(c * c.x, c * c.y, c * c.z))
        * m.mass;
    let i = sym_mat(m.inertia) - shift;
    let e = nalgebra::Matrix3::from_fn(|r, k| i.col(k)[r]).symmetric_eigen();
    let mut ev = [e.eigenvalues[0], e.eigenvalues[1], e.eigenvalues[2]];
    ev.sort_by(f64::total_cmp);
    let mr2 = m.mass * r_ref * r_ref;
    Degree2 {
        c20: -(ev[2] - 0.5 * (ev[0] + ev[1])) / mr2,
        c22: (ev[1] - ev[0]) / (4.0 * mr2),
    }
}

/// Present-day density target.
#[derive(Clone, Copy, Debug)]
pub struct GravityTarget {
    pub mass: f64,
    pub c20: f64,
    pub r_ref: f64,
    /// Inverse width (1/m) and exponent of the central core.
    pub sigma: f64,
    pub alpha: f64,
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct CoreFit {
    pub bulk: f64,
    pub weight: f64,
    pub homogeneous: Degree2,
    pub fitted: Degree2,
    pub centre_density: f64,
}

/// Solve `ρ_b` and the weight `w` of a core at the centre of mass so that the mass
/// and `C₂₀` match `target`. The mass is linear in `(ρ_b, w)`; `C₂₀` is solved for
/// `w` by the secant method with `ρ_b` eliminated by the mass.
pub fn fit_core(body: &Body, g: f64, target: &GravityTarget) -> Result<(Body, CoreFit), SimError> {
    let with = |bulk: f64, w: f64, centre: DVec3| -> Result<Body, SimError> {
        let kernels = if w == 0.0 {
            Vec::new()
        } else {
            vec![CauchyKernel::try_new(
                centre,
                w,
                target.sigma,
                target.alpha,
            )?]
        };
        Body::new(
            body.topo.clone(),
            body.positions.clone(),
            bulk,
            kernels,
            body.rho_s.clone(),
        )
    };
    let uniform = Forward::bind(&with(1.0, 0.0, DVec3::ZERO)?, g)?.moments;
    let centre = uniform.centroid;
    // Mass of a unit-weight core (its sign does not matter for the integral).
    let probe = Forward::bind(&with(1.0, -0.5, centre)?, g)?.moments;
    let core_mass = (probe.mass - uniform.mass) / -0.5;
    let homogeneous = degree2(&uniform, target.r_ref);
    // At unit density `uniform.mass` is the volume, so `ρ_b = (M − w m_core)/V`.
    let c20_of = |w: f64| -> Result<(f64, Body), SimError> {
        let b = with((target.mass - w * core_mass) / uniform.mass, w, centre)?;
        let m = Forward::bind(&b, g)?.moments;
        Ok((degree2(&m, target.r_ref).c20, b))
    };
    let (mut w0, mut f0) = (0.0, homogeneous.c20 - target.c20);
    let mut w1 = -0.1 * target.mass / core_mass;
    let (mut f1, mut best) = {
        let (c, b) = c20_of(w1)?;
        (c - target.c20, b)
    };
    for _ in 0..12 {
        if (f1 - f0).abs() < 1e-15 || f1.abs() < 1e-6 * target.c20.abs() {
            break;
        }
        let w2 = w1 - f1 * (w1 - w0) / (f1 - f0);
        let (c, b) = c20_of(w2)?;
        (w0, f0, w1, f1, best) = (w1, f1, w2, c - target.c20, b);
    }
    let m = Forward::bind(&best, g)?.moments;
    let density = best.density()?;
    let fit = CoreFit {
        bulk: best.bulk,
        weight: w1,
        homogeneous,
        fitted: degree2(&m, target.r_ref),
        centre_density: density.rho(centre),
    };
    if !(fit.centre_density > 0.0) {
        return Err(SimError::model_not_closed(format!(
            "the core that matches C20 = {} leaves a non-positive central density {:.1}",
            target.c20, fit.centre_density
        )));
    }
    Ok((best, fit))
}

/// One fixed-point iteration of the reconstruction.
#[derive(Clone, Debug, serde::Serialize)]
pub struct InversionStep {
    pub iteration: usize,
    /// `‖R_p − R_final‖` (RMS and max, m) after this forward run.
    pub rms: f64,
    pub max: f64,
    /// Largest `|R₀ − R_p|` of the initial state that was run (m).
    pub departure: f64,
    pub steps: usize,
    pub geometry_steps: usize,
    pub seconds: f64,
}

pub struct Inversion {
    pub initial: Body,
    pub last_run: Vec<StepRecord>,
    pub steps: Vec<InversionStep>,
    pub converged: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct InversionOptions {
    pub max_iterations: usize,
    /// Stop when the RMS vertex misfit is below this (m).
    pub tol: f64,
    /// `β` of `X₀ ← X₀ + β (X_p − F(X₀))`, in `(0, 1]`.
    pub damping: f64,
}

/// Fixed point `X₀ ← X₀ + β (X_p − F(X₀))` on radii, loose layer and cores. The
/// observer receives every step of every forward run with the iteration index.
/// The returned `initial` is the iterate whose forward run misses `X_p` least.
pub fn invert(
    project: &Project,
    schedule: &SpinSchedule,
    present: &Body,
    opts: &InversionOptions,
    observer: &mut dyn FnMut(usize, &StepRecord, &State, &StepFields) -> Result<(), SimError>,
) -> Result<Inversion, SimError> {
    if !(opts.damping > 0.0 && opts.damping <= 1.0) {
        return Err(SimError::model_not_closed(format!(
            "inversion damping must lie in (0, 1], got {}",
            opts.damping
        )));
    }
    let n = present.positions.len();
    let mut initial = present.clone();
    let mut steps = Vec::new();
    let mut last_run = Vec::new();
    let mut converged = false;
    // Slope relaxation maps many shapes onto one, so the iteration need not
    // contract; the reconstruction is the iterate with the smallest misfit.
    let mut best: Option<(f64, Body, Vec<StepRecord>)> = None;
    for k in 0..opts.max_iterations {
        let clock = std::time::Instant::now();
        let fwd = Forward::bind(&initial, project.g)?;
        let state = State {
            level: project.body.level,
            body: initial.clone(),
            t: 0.0,
            step: 0,
            omega: 0.0,
            frame_rotation: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            frame_translation: DVec3::ZERO,
            velocity: Vec::new(),
            evolving: Default::default(),
        };
        let res = evolve::run(
            project,
            schedule,
            state,
            fwd,
            &RunOptions {
                max_steps: None,
                until: None,
                lag: true,
            },
            &mut |r, s, f| observer(k, r, s, f),
        )?;
        // A frame re-alignment during the run (§10.4.4 step 8b) leaves the final body in a
        // turned and shifted frame; the misfit and the update are taken in the frame of
        // the initial and present bodies.
        let fin_body = undo_frame(
            &res.state.body,
            res.state.frame_rotation,
            res.state.frame_translation,
        )?;
        let fin = &fin_body;
        if fin.positions.len() != n {
            return Err(SimError::geometry(
                "the forward run changed the vertex count",
            ));
        }
        // Vertex identity is kept by the normal updates and the rigid frame (G4, G8).
        let e: Vec<DVec3> = present
            .positions
            .iter()
            .zip(&fin.positions)
            .map(|(p, f)| *p - *f)
            .collect();
        let rms = (e.iter().map(|v| v.length_squared()).sum::<f64>() / n as f64).sqrt();
        let max = e.iter().fold(0.0f64, |m, v| m.max(v.length()));
        let departure = initial
            .positions
            .iter()
            .zip(&present.positions)
            .fold(0.0f64, |m, (a, b)| m.max((*a - *b).length()));
        steps.push(InversionStep {
            iteration: k,
            rms,
            max,
            departure,
            steps: res.records.len(),
            geometry_steps: res.records.iter().filter(|r| r.kind == "geometry").count(),
            seconds: clock.elapsed().as_secs_f64(),
        });
        if best.as_ref().is_none_or(|b| rms < b.0) {
            best = Some((rms, initial.clone(), res.records.clone()));
        }
        last_run = res.records;
        if rms <= opts.tol {
            converged = true;
            break;
        }
        if k + 1 == opts.max_iterations {
            break;
        }
        // Update every field by the present-minus-final misfit.
        let b = opts.damping;
        let positions: Vec<DVec3> = initial
            .positions
            .iter()
            .zip(&e)
            .map(|(x, d)| *x + *d * b)
            .collect();
        let rho_s: Vec<f64> = initial
            .rho_s
            .iter()
            .zip(present.rho_s.iter().zip(&fin.rho_s))
            .map(|(r, (p, f))| (r + b * (p - f)).max(0.0))
            .collect();
        let kernels = if fin.kernels.len() == initial.kernels.len() {
            initial
                .kernels
                .iter()
                .zip(present.kernels.iter().zip(&fin.kernels))
                .map(|(k0, (kp, kf))| {
                    let metric: [f64; 6] =
                        std::array::from_fn(|i| k0.metric[i] + b * (kp.metric[i] - kf.metric[i]));
                    CauchyKernel::try_anisotropic(
                        k0.center + (kp.center - kf.center) * b,
                        k0.weight + b * (kp.weight - kf.weight),
                        metric,
                        k0.alpha,
                    )
                    .unwrap_or(*k0)
                })
                .collect()
        } else {
            initial.kernels.clone()
        };
        initial = Body::new(initial.topo.clone(), positions, initial.bulk, kernels, rho_s)?;
    }
    if let Some((_, body, records)) = best {
        initial = body;
        last_run = records;
    }
    Ok(Inversion {
        initial,
        last_run,
        steps,
        converged,
    })
}
