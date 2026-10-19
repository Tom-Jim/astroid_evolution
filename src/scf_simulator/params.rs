//! Closed TOML project of Tier 2 (§10.4.4 step 0). Every constant is required: a missing or
//! out-of-range one is `ModelNotClosed` and the run is refused. There are no defaults.
//!
//! Units are SI. The spin schedule is *not* part of the project; it is a CLI input.

use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::common::density::CauchyKernel;
use crate::common::math::DVec3;
use crate::rheo_friction::{Material, MaterialSpec, SolverSpec};

use super::error::ScfError;

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BodySpec {
    /// Icosphere subdivision for `shape = "ellipsoid"` only (`20 * 4^level` faces).
    /// An imported mesh keeps every vertex and face; this field is not a resampling.
    pub level: usize,
    /// `"ellipsoid"` or `"mesh"`.
    pub shape: String,
    pub semi_axes: Option<[f64; 3]>,
    pub mesh: Option<String>,
    pub mesh_unit: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KernelSpec {
    pub center: [f64; 3],
    pub weight: f64,
    pub sigma: f64,
    pub alpha: f64,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DensitySpec {
    /// Constant part of `ρ_K` (kg m⁻³).
    pub bulk: f64,
    pub kernels: Vec<KernelSpec>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StokesParams {
    /// Degree of the harmonic surface `R_SH` shared by the Stokes basis and `rheo_friction`.
    pub l_surface: usize,
    /// `L_max` of the velocity basis (§8.2).
    pub l_velocity: usize,
    /// `L_p`, `K_p` of the pressure space (§8.7).
    pub l_pressure: usize,
    pub k_pressure: usize,
    /// Product angular rule: Gauss–Legendre points in `cos θ` and uniform points in `φ`.
    pub n_theta: usize,
    pub n_phi: usize,
    /// Radial Gauss–Jacobi nodes (weight `ζ²`).
    pub n_radial: usize,
    /// External pressure `p_ext` of the free surface.
    pub p_ext: f64,
    pub picard_max: usize,
    /// (a) Relative change of the velocity and of the viscosity between Picard sweeps.
    pub tol_picard: f64,
    /// (b) Relative weak momentum residual and relative rigid-body multiplier.
    pub tol_momentum: f64,
    /// Under-relaxation `θ_η` of the viscosity, in `(0, 1]`.
    pub theta_eta: f64,
    /// Number of `γ̇_ε` continuation levels (each ×10 above the target).
    pub continuation: usize,
    /// Largest accepted `η_hi / η_lo` is reported, not limited; this bounds the Cholesky condition.
    pub rigid_sigma_min: f64,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct YieldParams {
    /// Yield tolerance `δ_Y ∈ (0, 1)` of `F_δ` (§2.8 B2).
    pub delta_y: f64,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModeCParams {
    /// Erosion coefficient `k_e` (s⁻¹).
    pub k_e: f64,
    /// Yield-excess scale `σ₀` (Pa).
    pub sigma_0: f64,
    /// Angular width (rad) of the deposition kernel `K`.
    pub kernel_width: f64,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StepParams {
    pub theta_cfl: f64,
    pub theta_c: f64,
    pub theta_char: f64,
    pub eps_omega: f64,
    pub omega_ref: f64,
    pub dt_max: f64,
    /// Time-advance fraction `λ ∈ (0, 1]`.
    pub lambda: f64,
    /// Threshold `ε_in` of the quasi-static indicator `Ek⁻¹` (reported, step flagged).
    pub eps_inertia: f64,
    pub max_steps: usize,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeometryParams {
    /// Retained so existing project files still parse. Face transversality is
    /// reported on the step record; this value is not an acceptance threshold.
    pub c0: f64,
    /// Overhang margin `ε_n` of the SOCP row `g·n̂ ≤ −ε_n |g|`.
    pub eps_n: f64,
    /// Accepted relative mass residual `ε_M`.
    pub eps_mass: f64,
    /// Initial trust radius `Δ⁽⁰⁾` as a fraction of the mean radius.
    pub trust0: f64,
    /// Trust radius floor (fraction of the mean radius); below it the step fails.
    pub trust_min: f64,
    pub trust_shrink: f64,
    /// Elastic-slack penalty `ϱ`.
    pub varrho: f64,
    /// Largest accepted `max_i |R_SH(ω̂_i) − R_i| / R_i`.
    pub sh_fit_tol: f64,
    /// Frame thresholds `ε_frame`, `ε_deg` of step 8b.
    pub eps_frame: f64,
    pub eps_deg: f64,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReliefParams {
    /// Relaxation sweeps and the slack at which they stop.
    pub max_sweeps: usize,
    pub tol_slack: f64,
    /// Radius anomaly (fraction of the maximum radius) above which a vertex is a crater feature.
    pub radius_anomaly: f64,
    /// Curvature quantile above which the protection weight rises from 0 to 1.
    pub curvature_quantile: f64,
    /// Protection at or above which a vertex is hard-locked.
    pub lock_weight: f64,
    /// Optional per-vertex masks `0..1` (one value per line) of the crater floor, wall, rim.
    pub floor: Option<String>,
    pub wall: Option<String>,
    pub rim: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProjectFile {
    gravity: GravitySpec,
    body: BodySpec,
    density: DensitySpec,
    material: MaterialSpec,
    solver: SolverSpec,
    stokes: StokesParams,
    yielding: YieldParams,
    mode_c: ModeCParams,
    step: StepParams,
    geometry: GeometryParams,
    relief: ReliefParams,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GravitySpec {
    g: f64,
}

#[derive(Clone, Debug)]
pub struct Project {
    pub g: f64,
    pub body: BodySpec,
    pub density: DensitySpec,
    pub material: Material,
    pub stokes: StokesParams,
    pub yielding: YieldParams,
    pub mode_c: ModeCParams,
    pub step: StepParams,
    pub geometry: GeometryParams,
    pub relief: ReliefParams,
    pub base_dir: PathBuf,
}

fn positive(name: &str, v: f64) -> Result<(), ScfError> {
    if v.is_finite() && v > 0.0 {
        Ok(())
    } else {
        Err(ScfError::model_not_closed(format!(
            "{name} must be finite and positive, got {v}"
        )))
    }
}

fn non_negative(name: &str, v: f64) -> Result<(), ScfError> {
    if v.is_finite() && v >= 0.0 {
        Ok(())
    } else {
        Err(ScfError::model_not_closed(format!(
            "{name} must be finite and non-negative, got {v}"
        )))
    }
}

fn unit_interval(name: &str, v: f64) -> Result<(), ScfError> {
    if v.is_finite() && v > 0.0 && v < 1.0 {
        Ok(())
    } else {
        Err(ScfError::model_not_closed(format!(
            "{name} must lie in (0, 1), got {v}"
        )))
    }
}

impl Project {
    pub fn load(path: &Path) -> Result<Self, ScfError> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| ScfError::io(format!("cannot read {}: {e}", path.display())))?;
        let base = path.parent().map(Path::to_path_buf).unwrap_or_default();
        Self::parse(&text, base)
    }

    pub fn parse(text: &str, base_dir: PathBuf) -> Result<Self, ScfError> {
        let f: ProjectFile = toml::from_str(text)
            .map_err(|e| ScfError::model_not_closed(format!("project file: {e}")))?;
        let material = Material::new(f.material, f.solver)?;
        positive("gravity.g", f.gravity.g)?;
        let b = &f.body;
        if b.level > 6 {
            return Err(ScfError::model_not_closed(
                "body.level above 6 is not supported",
            ));
        }
        match b.shape.as_str() {
            "ellipsoid" => {
                let a = b
                    .semi_axes
                    .ok_or_else(|| ScfError::model_not_closed("body.semi_axes is required"))?;
                a.iter().try_for_each(|v| positive("body.semi_axes", *v))?;
            }
            "mesh" => {
                if b.mesh.is_none() || b.mesh_unit.is_none() {
                    return Err(ScfError::model_not_closed(
                        "body.mesh and body.mesh_unit are required for shape = \"mesh\"",
                    ));
                }
            }
            other => {
                return Err(ScfError::model_not_closed(format!(
                    "body.shape must be \"ellipsoid\" or \"mesh\", got \"{other}\""
                )));
            }
        }
        non_negative("density.bulk", f.density.bulk)?;
        f.density.kernels.iter().try_for_each(|k| {
            CauchyKernel::try_new(DVec3::from_array(k.center), k.weight, k.sigma, k.alpha)
                .map(|_| ())
                .map_err(ScfError::from)
        })?;
        let s = &f.stokes;
        if s.l_surface == 0 || s.l_surface > 24 {
            return Err(ScfError::model_not_closed(
                "stokes.l_surface must lie in 1..=24",
            ));
        }
        if s.l_velocity == 0 || s.l_velocity > 8 {
            return Err(ScfError::model_not_closed(
                "stokes.l_velocity must lie in 1..=8",
            ));
        }
        if s.k_pressure == 0 || s.k_pressure > 8 {
            return Err(ScfError::model_not_closed(
                "stokes.k_pressure must lie in 1..=8",
            ));
        }
        if s.n_theta < 2 || s.n_phi < 4 || s.n_radial < 2 {
            return Err(ScfError::model_not_closed(
                "stokes quadrature is too coarse",
            ));
        }
        if s.picard_max == 0 {
            return Err(ScfError::model_not_closed(
                "stokes.picard_max must be positive",
            ));
        }
        positive("stokes.tol_picard", s.tol_picard)?;
        positive("stokes.tol_momentum", s.tol_momentum)?;
        positive("stokes.rigid_sigma_min", s.rigid_sigma_min)?;
        if !(s.theta_eta.is_finite() && s.theta_eta > 0.0 && s.theta_eta <= 1.0) {
            return Err(ScfError::model_not_closed(
                "stokes.theta_eta must lie in (0, 1]",
            ));
        }
        if !s.p_ext.is_finite() {
            return Err(ScfError::model_not_closed("stokes.p_ext must be finite"));
        }
        unit_interval("yielding.delta_y", f.yielding.delta_y)?;
        non_negative("mode_c.k_e", f.mode_c.k_e)?;
        positive("mode_c.sigma_0", f.mode_c.sigma_0)?;
        positive("mode_c.kernel_width", f.mode_c.kernel_width)?;
        let t = &f.step;
        unit_interval("step.theta_cfl", t.theta_cfl)?;
        if t.theta_cfl > 0.5 {
            return Err(ScfError::model_not_closed(
                "step.theta_cfl must not exceed 0.5",
            ));
        }
        unit_interval("step.theta_c", t.theta_c)?;
        unit_interval("step.theta_char", t.theta_char)?;
        positive("step.eps_omega", t.eps_omega)?;
        positive("step.omega_ref", t.omega_ref)?;
        positive("step.dt_max", t.dt_max)?;
        positive("step.eps_inertia", t.eps_inertia)?;
        if !(t.lambda.is_finite() && t.lambda > 0.0 && t.lambda <= 1.0) {
            return Err(ScfError::model_not_closed("step.lambda must lie in (0, 1]"));
        }
        if t.max_steps == 0 {
            return Err(ScfError::model_not_closed(
                "step.max_steps must be positive",
            ));
        }
        let g = &f.geometry;
        unit_interval("geometry.c0", g.c0)?;
        unit_interval("geometry.eps_n", g.eps_n)?;
        positive("geometry.eps_mass", g.eps_mass)?;
        positive("geometry.trust0", g.trust0)?;
        positive("geometry.trust_min", g.trust_min)?;
        unit_interval("geometry.trust_shrink", g.trust_shrink)?;
        positive("geometry.varrho", g.varrho)?;
        positive("geometry.sh_fit_tol", g.sh_fit_tol)?;
        positive("geometry.eps_frame", g.eps_frame)?;
        positive("geometry.eps_deg", g.eps_deg)?;
        if g.trust_min >= g.trust0 {
            return Err(ScfError::model_not_closed(
                "geometry.trust_min must be below geometry.trust0",
            ));
        }
        let r = &f.relief;
        if r.max_sweeps == 0 {
            return Err(ScfError::model_not_closed(
                "relief.max_sweeps must be positive",
            ));
        }
        non_negative("relief.tol_slack", r.tol_slack)?;
        non_negative("relief.radius_anomaly", r.radius_anomaly)?;
        unit_interval("relief.curvature_quantile", r.curvature_quantile)?;
        positive("relief.lock_weight", r.lock_weight)?;
        let masks = [&r.floor, &r.wall, &r.rim];
        if masks.iter().any(|m| m.is_some()) && masks.iter().any(|m| m.is_none()) {
            return Err(ScfError::model_not_closed(
                "relief.floor, relief.wall and relief.rim must all be present or all absent",
            ));
        }
        Ok(Self {
            g: f.gravity.g,
            body: f.body,
            density: f.density,
            material,
            stokes: f.stokes,
            yielding: f.yielding,
            mode_c: f.mode_c,
            step: f.step,
            geometry: f.geometry,
            relief: f.relief,
            base_dir,
        })
    }

    pub fn kernels(&self) -> Vec<CauchyKernel> {
        self.density
            .kernels
            .iter()
            .map(|k| {
                CauchyKernel::try_new(DVec3::from_array(k.center), k.weight, k.sigma, k.alpha)
                    .expect("validated at load")
            })
            .collect()
    }

    pub fn resolve(&self, relative: &str) -> PathBuf {
        let p = Path::new(relative);
        if p.is_absolute() {
            p.to_path_buf()
        } else {
            self.base_dir.join(p)
        }
    }
}
