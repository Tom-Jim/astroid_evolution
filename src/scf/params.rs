//! Closed TOML project of Tier 2 (§10.3.4 step 0). Every constant is required: a missing or
//! out-of-range one is `ModelNotClosed` and the run is refused. There are no defaults.
//!
//! Units are SI. The spin schedule is *not* part of the project; it is a CLI input.

use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::common::density::CauchyKernel;
use crate::common::math::DVec3;
use crate::rheo::{Material, MaterialSpec, SolverSpec};

use super::cores::CoreParams;
use super::breakup::BreakupParams;
use super::cycle::CycleParams;
use super::orbit::OrbitParams;
use super::evolving::{DamageParams, DilatancyParams, MarkerParams, SpinLawParams, ThermalParams};
use crate::common::error::SimError;

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BodySpec {
    /// Icosphere subdivision for `shape = "ellipsoid"` and `"mesh_rays"` (`20 * 4^level` faces).
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
    /// Isotropic inverse width σ (1/length): `S = σ² I`.
    #[serde(default)]
    pub sigma: Option<f64>,
    /// Full metric `S`, `(xx, yy, zz, xy, xz, yz)` in 1/length². Exclusive with `sigma`.
    #[serde(default)]
    pub metric: Option<[f64; 6]>,
    pub alpha: f64,
}

impl KernelSpec {
    pub fn kernel(&self) -> Result<CauchyKernel, SimError> {
        let c = DVec3::from_array(self.center);
        match (self.sigma, self.metric) {
            (Some(s), None) => CauchyKernel::try_new(c, self.weight, s, self.alpha),
            (None, Some(m)) => CauchyKernel::try_anisotropic(c, self.weight, m, self.alpha),
            _ => {
                return Err(SimError::model_not_closed(
                    "a density kernel needs exactly one of `sigma` and `metric`",
                ));
            }
        }
    }
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
    /// `L_max` of the velocity basis (§8.1.3).
    pub l_velocity: usize,
    /// `L_p`, `K_p` of the pressure space (§8.1.3).
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
    /// Velocity solver at frozen pressure: `"primal-dual"` (§10.3.2) or `"picard"`, the
    /// fixed-viscosity map kept as a check.
    #[serde(default = "default_solver")]
    pub solver: String,
    /// `ε_pd` of criterion (a): relative velocity change and relative primal–dual gap.
    #[serde(default = "default_pd_tol")]
    pub pd_tol: f64,
    /// `k_max` of the primal–dual iteration.
    #[serde(default = "default_pd_max")]
    pub pd_max: usize,
    /// `N_p`: primal–dual iterations between pressure updates.
    #[serde(default = "default_pd_pressure_every")]
    pub pd_pressure_every: usize,
    /// Target number of points of the boundary rule of the Nitsche terms (G3).
    #[serde(default = "default_n_boundary")]
    pub n_boundary: usize,
    /// Refinement depth of cut angular cells of the volume rule (G2); 0 keeps the product rule.
    #[serde(default = "default_cut_depth")]
    pub cut_depth: usize,
}

fn default_n_boundary() -> usize {
    1024
}

fn default_cut_depth() -> usize {
    3
}

fn default_solver() -> String {
    "primal-dual".into()
}

fn default_pd_tol() -> f64 {
    1e-3
}

fn default_pd_max() -> usize {
    20_000
}

fn default_pd_pressure_every() -> usize {
    10
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct YieldParams {
    /// Yield tolerance `δ_Y ∈ (0, 1)` of `F_δ` (§10.0.2 B2).
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
    /// Rotating-frame runout of the eroded packets (§10.0.2, A-C3); `false` keeps the
    /// isotropic kernel of §10.0.2.
    pub runout: bool,
    /// `T_run` (s): a packet still moving then stops where it is.
    pub runout_time: f64,
    /// `ε_stop`: stopping speed as a fraction of `√(|g_eff| h_s)`.
    pub runout_stop: f64,
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
    /// Retained so existing project files still parse; the general route fits no surface.
    pub sh_fit_tol: f64,
    /// Frame thresholds `ε_frame`, `ε_deg` of step 8b.
    pub eps_frame: f64,
    pub eps_deg: f64,
    /// Clearance fraction `θ_g ∈ (0, ½)` of the rows `−θ_g t⁻ ≤ d ≤ θ_g t⁺` (G4).
    #[serde(default = "default_theta_g")]
    pub theta_g: f64,
}

fn default_theta_g() -> f64 {
    0.4
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
    cores: CoreParams,
    /// Chapter 11 switches: an absent section is the hypothesis off.
    #[serde(default)]
    spin_law: Option<SpinLawParams>,
    #[serde(default)]
    thermal: Option<ThermalParams>,
    #[serde(default)]
    markers: Option<MarkerParams>,
    #[serde(default)]
    damage: Option<DamageParams>,
    #[serde(default)]
    dilatancy: Option<DilatancyParams>,
    /// A-O (§10.3.6): mass-coupled orbital decay and loss through the Hill necks.
    #[serde(default)]
    orbit: Option<OrbitParams>,
    /// Chapter 12: breakup, fragments and the debris ring.
    #[serde(default)]
    breakup: Option<BreakupParams>,
    /// A-Λ (§10.3.7): eccentricity, forced libration and the tidal cycle.
    #[serde(default)]
    cycle: Option<CycleParams>,
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
    /// Lagrangian transport and refit of the density cores (§9.2).
    pub cores: CoreParams,
    /// A-L (§11.7): spin from the angular momentum; `None` keeps the exogenous table.
    pub spin_law: Option<SpinLawParams>,
    /// A-Θ (§11.6): thermal fatigue of the surface skin and the radiative torque.
    pub thermal: Option<ThermalParams>,
    /// The state markers of §11.1–11.2, required by A-ϖ and A-Ξ.
    pub markers: Option<MarkerParams>,
    /// A-ϖ (§11.5): damage of the cohesion.
    pub damage: Option<DamageParams>,
    /// A-Ξ (§11.3–11.4): rate-type dilatancy.
    pub dilatancy: Option<DilatancyParams>,
    /// A-O (§10.3.6): the orbit follows the mass, which leaves through the Hill necks.
    pub orbit: Option<OrbitParams>,
    /// Chapter 12: how a failed satellite breaks, its fragments move, and its debris spreads.
    pub breakup: Option<BreakupParams>,
    /// A-Λ (§10.3.7): the tidal cycle of an eccentric synchronous satellite.
    pub cycle: Option<CycleParams>,
    pub base_dir: PathBuf,
}

fn positive(name: &str, v: f64) -> Result<(), SimError> {
    if v.is_finite() && v > 0.0 {
        Ok(())
    } else {
        Err(SimError::model_not_closed(format!(
            "{name} must be finite and positive, got {v}"
        )))
    }
}

fn non_negative(name: &str, v: f64) -> Result<(), SimError> {
    if v.is_finite() && v >= 0.0 {
        Ok(())
    } else {
        Err(SimError::model_not_closed(format!(
            "{name} must be finite and non-negative, got {v}"
        )))
    }
}

fn unit_interval(name: &str, v: f64) -> Result<(), SimError> {
    if v.is_finite() && v > 0.0 && v < 1.0 {
        Ok(())
    } else {
        Err(SimError::model_not_closed(format!(
            "{name} must lie in (0, 1), got {v}"
        )))
    }
}

impl Project {
    pub fn load(path: &Path) -> Result<Self, SimError> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| SimError::io(format!("cannot read {}: {e}", path.display())))?;
        let base = path.parent().map(Path::to_path_buf).unwrap_or_default();
        Self::parse(&text, base)
    }

    pub fn parse(text: &str, base_dir: PathBuf) -> Result<Self, SimError> {
        let f: ProjectFile = toml::from_str(text)
            .map_err(|e| SimError::model_not_closed(format!("project file: {e}")))?;
        let mut material = Material::new(f.material, f.solver)?;
        f.cores.validate()?;
        if let Some(s) = &f.spin_law {
            s.validate()?;
            if s.torque == "radiative" && f.thermal.is_none() {
                return Err(SimError::model_not_closed(
                    "spin_law.torque = \"radiative\" needs the [thermal] section (A-Θ)",
                ));
            }
        }
        if let Some(t) = &f.thermal {
            t.validate()?;
        }
        if let Some(b) = &f.breakup {
            b.validate()?;
            if f.orbit.is_none() {
                return Err(SimError::model_not_closed("[breakup] needs the [orbit] section (A-O) for the primary"));
            }
        }
        if let Some(c) = &f.cycle {
            c.validate()?;
            if f.orbit.is_none() {
                return Err(SimError::model_not_closed("[cycle] (A-Λ) needs the [orbit] section (A-O)"));
            }
            if c.fatigue.is_some() && f.damage.is_none() {
                return Err(SimError::model_not_closed("cycle.fatigue damages the bonds: it needs [damage] (A-ϖ)"));
            }
        }
        if let Some(o) = &f.orbit {
            o.validate()?;
            if f.spin_law.is_some() {
                return Err(SimError::model_not_closed(
                    "[orbit] (A-O) keeps the spin synchronous; it excludes [spin_law] (A-L)",
                ));
            }
        }
        if let Some(d) = &f.damage {
            d.validate()?;
        }
        if let Some(d) = &f.dilatancy {
            d.validate(material.spec.mu_s)?;
            material.dilatancy = Some([d.k * d.d_nu, d.i_nu]);
        }
        match &f.markers {
            Some(m) => m.validate()?,
            None if f.damage.is_some() || f.dilatancy.is_some() => {
                return Err(SimError::model_not_closed(
                    "[damage] and [dilatancy] need the [markers] section (§11.1)",
                ));
            }
            None => {}
        }
        positive("gravity.g", f.gravity.g)?;
        let b = &f.body;
        if b.level > 6 {
            return Err(SimError::model_not_closed(
                "body.level above 6 is not supported",
            ));
        }
        match b.shape.as_str() {
            "ellipsoid" => {
                let a = b
                    .semi_axes
                    .ok_or_else(|| SimError::model_not_closed("body.semi_axes is required"))?;
                a.iter().try_for_each(|v| positive("body.semi_axes", *v))?;
            }
            "mesh" | "mesh_rays" => {
                if b.mesh.is_none() || b.mesh_unit.is_none() {
                    return Err(SimError::model_not_closed(format!(
                        "body.mesh and body.mesh_unit are required for shape = \"{}\"",
                        b.shape
                    )));
                }
            }
            other => {
                return Err(SimError::model_not_closed(format!(
                    "body.shape must be \"ellipsoid\", \"mesh\" or \"mesh_rays\", got \"{other}\""
                )));
            }
        }
        non_negative("density.bulk", f.density.bulk)?;
        f.density
            .kernels
            .iter()
            .try_for_each(|k| k.kernel().map(|_| ()))?;
        let s = &f.stokes;
        if s.l_surface == 0 || s.l_surface > 24 {
            return Err(SimError::model_not_closed(
                "stokes.l_surface must lie in 1..=24",
            ));
        }
        if s.l_velocity == 0 || s.l_velocity > 8 {
            return Err(SimError::model_not_closed(
                "stokes.l_velocity must lie in 1..=8",
            ));
        }
        if s.k_pressure == 0 || s.k_pressure > 8 {
            return Err(SimError::model_not_closed(
                "stokes.k_pressure must lie in 1..=8",
            ));
        }
        if s.n_theta < 2 || s.n_phi < 4 || s.n_radial < 2 {
            return Err(SimError::model_not_closed(
                "stokes quadrature is too coarse",
            ));
        }
        if s.picard_max == 0 {
            return Err(SimError::model_not_closed(
                "stokes.picard_max must be positive",
            ));
        }
        positive("stokes.tol_picard", s.tol_picard)?;
        positive("stokes.tol_momentum", s.tol_momentum)?;
        positive("stokes.rigid_sigma_min", s.rigid_sigma_min)?;
        if !(s.theta_eta.is_finite() && s.theta_eta > 0.0 && s.theta_eta <= 1.0) {
            return Err(SimError::model_not_closed(
                "stokes.theta_eta must lie in (0, 1]",
            ));
        }
        if !s.p_ext.is_finite() {
            return Err(SimError::model_not_closed("stokes.p_ext must be finite"));
        }
        unit_interval("yielding.delta_y", f.yielding.delta_y)?;
        non_negative("mode_c.k_e", f.mode_c.k_e)?;
        positive("mode_c.sigma_0", f.mode_c.sigma_0)?;
        positive("mode_c.kernel_width", f.mode_c.kernel_width)?;
        positive("mode_c.runout_time", f.mode_c.runout_time)?;
        unit_interval("mode_c.runout_stop", f.mode_c.runout_stop)?;
        let t = &f.step;
        unit_interval("step.theta_cfl", t.theta_cfl)?;
        if t.theta_cfl > 0.5 {
            return Err(SimError::model_not_closed(
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
            return Err(SimError::model_not_closed("step.lambda must lie in (0, 1]"));
        }
        if t.max_steps == 0 {
            return Err(SimError::model_not_closed(
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
            return Err(SimError::model_not_closed(
                "geometry.trust_min must be below geometry.trust0",
            ));
        }
        let r = &f.relief;
        if r.max_sweeps == 0 {
            return Err(SimError::model_not_closed(
                "relief.max_sweeps must be positive",
            ));
        }
        non_negative("relief.tol_slack", r.tol_slack)?;
        non_negative("relief.radius_anomaly", r.radius_anomaly)?;
        unit_interval("relief.curvature_quantile", r.curvature_quantile)?;
        positive("relief.lock_weight", r.lock_weight)?;
        let masks = [&r.floor, &r.wall, &r.rim];
        if masks.iter().any(|m| m.is_some()) && masks.iter().any(|m| m.is_none()) {
            return Err(SimError::model_not_closed(
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
            cores: f.cores,
            spin_law: f.spin_law,
            thermal: f.thermal,
            markers: f.markers,
            damage: f.damage,
            dilatancy: f.dilatancy,
            orbit: f.orbit,
            breakup: f.breakup,
            cycle: f.cycle,
            base_dir,
        })
    }

    pub fn kernels(&self) -> Vec<CauchyKernel> {
        self.density
            .kernels
            .iter()
            .map(|k| k.kernel().expect("validated at load"))
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
