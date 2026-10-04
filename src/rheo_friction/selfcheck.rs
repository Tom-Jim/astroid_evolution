//! Acceptance checks of the chain. Each check returns a residual and a limit;
//! `rheo-cli check` prints them and `tests/rheo_friction.rs` asserts them.

use std::f64::consts::PI;

use super::basis::{self, K, lm_index, sh_all};
use super::closure::{ClosureInput, closure};
use super::error::RheoError;
use super::gpu::{NodeResult, RheoGpu};
use super::model::{
    Environment, NodeBatch, StarSurface, VelocityField, chart_direction, fibonacci_directions,
};
use super::params::{Material, MaterialSpec, MuProjMode, SolverSpec};
use super::reference;
use crate::common::math::DVec3;

#[derive(Clone, Debug)]
pub struct Check {
    pub name: String,
    pub value: f64,
    pub limit: f64,
}

impl Check {
    pub fn passed(&self) -> bool {
        self.value.is_finite() && self.value <= self.limit
    }
}

fn check(name: &str, value: f64, limit: f64) -> Check {
    Check {
        name: name.to_string(),
        value,
        limit,
    }
}

/// Rubble-pile constants used by the checks (SI).
pub fn reference_material(c_t: f64, mode: &str) -> Result<Material, RheoError> {
    Material::new(
        MaterialSpec {
            nu0: 0.6,
            z0: 6.0,
            d_p: 0.05,
            rho_p: 2500.0,
            mu_s: 0.38,
            mu_2: 0.64,
            i_0: 0.279,
            cohesion: 25.0,
            c_c: 1.0,
            f_h: 1.0e-4,
            c_t,
            gamma_eps: 1.0e-9,
            p_min: 1.0e-6,
            h_s: 5.0,
            eps_overhang: 0.01,
            mu_proj_mode: mode.to_string(),
        },
        SolverSpec {
            max_iter: 200,
            tol: 1.0e-12,
        },
    )
}

pub fn lumpy_surface() -> StarSurface {
    StarSurface::from_entries(&[
        (0, 0, 500.0 * (4.0 * PI).sqrt()),
        (1, 1, 12.0),
        (2, 0, 35.0),
        (2, -1, -18.0),
        (2, 2, 14.0),
        (3, 1, 9.0),
        (4, -3, -6.0),
        (5, 2, 3.0),
    ])
    .expect("lumpy surface")
}

pub fn flow_field() -> VelocityField {
    VelocityField::from_entries(
        &[
            (1, 0, 1, 2.0e-6),
            (2, 1, 0, 3.0e-6),
            (2, 1, 2, -1.5e-6),
            (3, -2, 1, 2.5e-6),
            (3, 3, 3, 1.0e-6),
            (4, 0, 2, -2.0e-6),
        ],
        &[
            (1, 1, 2, 1.0e-6),
            (2, -2, 1, -2.0e-6),
            (3, 0, 0, 1.5e-6),
            (4, 2, 3, 1.0e-6),
        ],
    )
    .expect("flow field")
}

fn shape_env() -> Environment {
    Environment {
        gravity_k: 1.0e-5,
        omega: 2.0e-4,
        p_ext: 0.0,
    }
}

fn max_norm(values: impl Iterator<Item = f64>) -> f64 {
    values.fold(0.0_f64, |m, v| m.max(v.abs()))
}

/// Rigid translation (`l=1,k=0` poloidal) and rotation (`l=1,k=0` toroidal) on a sphere have no
/// surface shear: `c^P_{10} = c^T_{10} = 0`. The residual is the largest `gd R / |v|`.
pub fn rigid_motion(gpu: &RheoGpu) -> Result<Check, RheoError> {
    let material = reference_material(0.0, "dynamic")?;
    let surface = StarSurface::sphere(500.0);
    let vel = VelocityField::from_entries(
        &[(1, -1, 0, 1.3e-6), (1, 0, 0, -0.7e-6), (1, 1, 0, 2.1e-6)],
        &[(1, -1, 0, 0.4e-6), (1, 0, 0, 1.1e-6), (1, 1, 0, -0.9e-6)],
    )?;
    let nodes = shape_env().nodes(&surface, fibonacci_directions(512));
    let r = gpu.evaluate(&material, &surface, &vel, &nodes)?;
    let vmax = max_norm(r.nodes.iter().map(|n| n.velocity.length()));
    let gmax = max_norm(r.nodes.iter().map(|n| n.shear));
    Ok(check(
        "rigid motion: gd R / |v|",
        gmax * 500.0 / vmax,
        1.0e-9,
    ))
}

/// Radius from the DS spherical-harmonic recursion against the f64 sum.
pub fn radius_recursion(gpu: &RheoGpu) -> Result<Check, RheoError> {
    let material = reference_material(0.0, "dynamic")?;
    let mut entries = vec![(0, 0, 500.0 * (4.0 * PI).sqrt())];
    for l in 1..=24usize {
        for m in -(l as i32)..=(l as i32) {
            let v = ((l as f64 * 1.7 + m as f64 * 0.9).sin()) * 8.0 / (l * l) as f64;
            entries.push((l, m, v));
        }
    }
    let surface = StarSurface::from_entries(&entries)?;
    let nodes = shape_env().nodes(&surface, fibonacci_directions(777));
    let r = gpu.evaluate(&material, &surface, &VelocityField::zero(), &nodes)?;
    let worst = r
        .nodes
        .iter()
        .zip(&nodes.dirs)
        .map(|(n, d)| (n.radius - reference::radius(&surface, *d)).abs() / n.radius)
        .fold(0.0_f64, f64::max);
    Ok(check("radius recursion (l<=24): relative", worst, 1.0e-12))
}

/// Surface velocity trace of Proposition 10.2 against the definition `v_r = L^2 Phi / r`,
/// `v_S = r^-1 grad d_r(r Phi)`, `v^T = grad Psi x e_r` on a non-spherical surface.
pub fn velocity_trace(gpu: &RheoGpu) -> Result<Check, RheoError> {
    let material = reference_material(0.0, "dynamic")?;
    let (surface, vel) = (lumpy_surface(), flow_field());
    let nodes = shape_env().nodes(&surface, fibonacci_directions(24));
    let r = gpu.evaluate(&material, &surface, &vel, &nodes)?;
    let vmax = max_norm(r.nodes.iter().map(|n| n.velocity.length()));
    let worst = r
        .nodes
        .iter()
        .zip(&nodes.dirs)
        .map(|(n, d)| (n.velocity - reference::trace_velocity(&surface, &vel, *d)).length())
        .fold(0.0_f64, f64::max);
    Ok(check(
        "velocity trace vs potentials: relative",
        worst / vmax,
        1.0e-6,
    ))
}

/// `D_ab = sym(e_a . d_b v)` from the jets against a fourth-order stencil of the GPU's own
/// velocity values at neighbouring chart points.
pub fn strain_rate_stencil(gpu: &RheoGpu) -> Result<Check, RheoError> {
    let material = reference_material(0.0, "dynamic")?;
    let (surface, vel) = (lumpy_surface(), flow_field());
    let env = shape_env();
    let centres = fibonacci_directions(16);
    let h = 1.0e-3;
    let mut dirs = centres.clone();
    let offsets = [
        (1.0, 0.0),
        (-1.0, 0.0),
        (2.0, 0.0),
        (-2.0, 0.0),
        (0.0, 1.0),
        (0.0, -1.0),
        (0.0, 2.0),
        (0.0, -2.0),
    ];
    for c in &centres {
        for (a, b) in offsets {
            dirs.push(chart_direction(*c, a * h, b * h));
        }
    }
    let nodes = env.nodes(&surface, dirs);
    let r = gpu.evaluate(&material, &surface, &vel, &nodes)?;
    let n0 = centres.len();
    let mut worst = 0.0_f64;
    let mut scale = 0.0_f64;
    for (i, _) in centres.iter().enumerate() {
        let at = |k: usize| -> DVec3 { r.nodes[n0 + 8 * i + k].velocity };
        let d_u = (at(0) - at(1)) * (8.0 / (12.0 * h)) - (at(2) - at(3)) * (1.0 / (12.0 * h));
        let d_v = (at(4) - at(5)) * (8.0 / (12.0 * h)) - (at(6) - at(7)) * (1.0 / (12.0 * h));
        let n = &r.nodes[i];
        let d11 = n.e1.dot(d_u);
        let d22 = n.e2.dot(d_v);
        let d12 = 0.5 * (n.e1.dot(d_v) + n.e2.dot(d_u));
        for (g, f) in [(n.d_ab[0], d11), (n.d_ab[1], d12), (n.d_ab[2], d22)] {
            worst = worst.max((g - f).abs());
            scale = scale.max(g.abs());
        }
    }
    Ok(check(
        "D_ab jets vs velocity stencil: relative",
        worst / scale,
        1.0e-7,
    ))
}

/// Fixed reference sphere, poloidal flow: `gd^2 = 2 [D_rr^2 + D_SS : D_SS]` (Prop. 10.3), with the
/// spherical Hessian taken by an independent finite-difference stencil.
pub fn sphere_closed_form(gpu: &RheoGpu) -> Result<Check, RheoError> {
    let material = reference_material(0.0, "dynamic")?;
    let surface = StarSurface::sphere(500.0);
    let (l, m) = (3usize, 2i32);
    let q = [0.0, 1.4e-6, -0.9e-6, 0.6e-6];
    let mut poloidal = Vec::new();
    for (k, v) in q.iter().enumerate() {
        poloidal.push((l, m, k, *v));
    }
    let vel = VelocityField::from_entries(&poloidal, &[])?;
    let nodes = shape_env().nodes(&surface, fibonacci_directions(40));
    let r = gpu.evaluate(&material, &surface, &vel, &nodes)?;
    let idx = lm_index(l, m);
    let lam = basis::lambda(l);
    let a_sum: f64 = q.iter().sum();
    let s_sum: f64 = (0..K).map(|k| q[k] * basis::sigma(l, k)).sum();
    let a = a_sum + s_sum;
    let b = lam * a_sum;
    let mut worst = 0.0_f64;
    let mut scale = 0.0_f64;
    for (n, d) in r.nodes.iter().zip(&nodes.dirs) {
        let f = |w: DVec3| sh_all(l, w.to_array())[idx];
        let y = f(*d);
        let [h11, h12, h22] = reference::hessian(&f, *d);
        let hh = h11 * h11 + 2.0 * h12 * h12 + h22 * h22;
        let d_rr = lam * (s_sum - a_sum) * y;
        let d_ss = a * a * hh - 2.0 * a * b * lam * y * y + 2.0 * b * b * y * y;
        let g2 = 2.0 * (d_rr * d_rr + d_ss);
        worst = worst.max((n.shear - g2.sqrt()).abs());
        scale = scale.max(g2.sqrt());
    }
    Ok(check(
        "sphere closed form (Prop. 10.3): relative",
        worst / scale,
        1.0e-6,
    ))
}

fn closure_input(n: &NodeResult, gravity: DVec3, pressure: f64) -> ClosureInput {
    ClosureInput {
        s0: n.s0,
        tg: n.tg,
        pressure,
        g_dot_n: gravity.dot(n.normal),
        g_norm: gravity.length(),
        g_cross_n: gravity.cross(n.normal).length(),
        tide_frob: n.tide_frob,
    }
}

/// A tide strong enough that the frozen-viscosity fixed point needs several iterations.
fn tide_case(omega: f64) -> (Material, StarSurface, VelocityField, NodeBatch) {
    let material = reference_material(1.0e7, "dynamic").expect("material");
    let (surface, vel) = (lumpy_surface(), flow_field());
    let env = Environment {
        gravity_k: 1.0e-5,
        omega,
        p_ext: 0.0,
    };
    let nodes = env.nodes(&surface, fibonacci_directions(400));
    (material, surface, vel, nodes)
}

/// Every closure output of the DS kernel against the f64 reference fed with the kernel's own
/// `S0`, `t.g.t`, normal and tide norm.
pub fn closure_double_single(gpu: &RheoGpu) -> Result<Vec<Check>, RheoError> {
    let mut out = Vec::new();
    for (label, mode) in [("dynamic", "dynamic"), ("static", "static")] {
        let (_, surface, vel, nodes) = tide_case(1.0e-3);
        let material = reference_material(1.0e7, mode)?;
        let r = gpu.evaluate(&material, &surface, &vel, &nodes)?;
        let (mut shear, mut mu_eff, mut phi, mut eta, mut inertial, mut mu_d, mut slack) = (
            0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64,
        );
        let mut iterations = 0;
        for (i, n) in r.nodes.iter().enumerate() {
            let c = closure(
                &material,
                &closure_input(n, nodes.gravity[i], nodes.pressure[i]),
            );
            let rel = |a: f64, b: f64| (a - b).abs() / b.abs().max(1e-300);
            shear = shear.max(rel(n.shear, c.shear));
            mu_d = mu_d.max(rel(n.mu_d, c.mu_d));
            inertial = inertial.max(rel(n.inertial, c.inertial));
            eta = eta.max(rel(n.eta, c.eta));
            mu_eff = mu_eff.max(rel(n.mu_eff, c.mu_eff));
            phi = phi.max(rel(n.phi, c.phi));
            slack = slack.max((n.slack - c.slack).abs());
            iterations = iterations.max(n.iterations);
            if !n.converged || !c.converged {
                return Err(RheoError::rheology_not_converged("tide case must converge"));
            }
        }
        out.push(check(
            &format!("closure [{label}] shear: relative"),
            shear,
            1.0e-11,
        ));
        out.push(check(
            &format!("closure [{label}] mu_d: relative"),
            mu_d,
            1.0e-11,
        ));
        out.push(check(
            &format!("closure [{label}] I_c: relative"),
            inertial,
            1.0e-11,
        ));
        out.push(check(
            &format!("closure [{label}] eta: relative"),
            eta,
            1.0e-11,
        ));
        out.push(check(
            &format!("closure [{label}] mu_eff: relative"),
            mu_eff,
            1.0e-11,
        ));
        out.push(check(
            &format!("closure [{label}] phi: relative"),
            phi,
            1.0e-11,
        ));
        out.push(check(
            &format!("closure [{label}] slope slack: absolute"),
            slack,
            1.0e-11,
        ));
        if label == "dynamic" {
            out.push(check(
                "tide fixed point needs more than one pass (1 = yes)",
                if iterations > 1 { 0.0 } else { 1.0 },
                0.0,
            ));
        }
    }
    Ok(out)
}

/// `mu_proj = mu_s` exactly in static mode.
pub fn static_projection(gpu: &RheoGpu) -> Result<Check, RheoError> {
    let (_, surface, vel, nodes) = tide_case(1.0e-3);
    let material = reference_material(1.0e7, "static")?;
    debug_assert_eq!(material.mode, MuProjMode::Static);
    let r = gpu.evaluate(&material, &surface, &vel, &nodes)?;
    let worst = max_norm(r.nodes.iter().map(|n| n.mu_proj - material.mu_y));
    Ok(check("static mode: mu_proj - mu_s", worst, 1.0e-12))
}

/// A tide traction above the flow strength has no frozen-viscosity solution: the chain has to
/// report non-convergence instead of a number.
pub fn tide_beyond_strength(gpu: &RheoGpu) -> Result<Check, RheoError> {
    let (material, surface, vel, nodes) = tide_case(6.0e-3);
    let r = gpu.evaluate(&material, &surface, &vel, &nodes)?;
    let ok = r.stats.unconverged > 0 && r.nodes.iter().any(|n| !n.converged);
    Ok(check(
        "over-strength tide flagged non-converged (0 = flagged)",
        if ok { 0.0 } else { 1.0 },
        0.0,
    ))
}

/// Physical sanity of the friction angle chain: finite, `mu_d in [mu_s, mu_2)`, `phi = atan(mu_eff)`.
pub fn friction_ranges(gpu: &RheoGpu) -> Result<Check, RheoError> {
    let (material, surface, vel, nodes) = tide_case(1.0e-3);
    let r = gpu.evaluate(&material, &surface, &vel, &nodes)?;
    let s = &material.spec;
    let mut bad = 0.0_f64;
    for n in &r.nodes {
        let in_range = n.mu_d >= s.mu_s - 1e-12 && n.mu_d < s.mu_2;
        let angle = (n.phi - n.mu_eff.atan()).abs() / n.phi.abs().max(1e-300);
        let sin2 = (n.sin2_phi - n.phi.sin().powi(2)).abs();
        bad = bad
            .max(if in_range { 0.0 } else { 1.0 })
            .max(angle * 1e12)
            .max(sin2 * 1e12);
    }
    Ok(check(
        "friction ranges and phi = atan(mu_eff) (1e-12 units)",
        bad,
        1.0,
    ))
}

pub fn run_all(gpu: &RheoGpu) -> Result<Vec<Check>, RheoError> {
    let mut out = vec![
        rigid_motion(gpu)?,
        radius_recursion(gpu)?,
        velocity_trace(gpu)?,
        strain_rate_stencil(gpu)?,
        sphere_closed_form(gpu)?,
    ];
    out.extend(closure_double_single(gpu)?);
    out.push(static_projection(gpu)?);
    out.push(tide_beyond_strength(gpu)?);
    out.push(friction_ranges(gpu)?);
    Ok(out)
}
