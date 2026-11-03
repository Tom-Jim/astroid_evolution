//! Hypothesis A-Θ (§11.6): diurnal thermal cycling and fatigue of the surface skin.
//!
//! Every face is a one-dimensional half-space. At each orbital sample its surface
//! temperature is the exact periodic solution of `ε σ T|T|³ + Q_ω T = S` (Proposition
//! 11.10), solved by Newton on `N_φ` collocation phases. The diurnal range drives a
//! Basquin–Miner fatigue rate, which damages the shallow cohesion and produces loose
//! material (Proposition 11.11). The same solution gives the radiative torque of law L2.
//!
//! Shadow rays are cached as bits and retraced only where the geometry moved; face
//! temperatures are kept as a warm start and recomputed only where their insolation
//! changed or the spin left the configured tolerance.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::common::error::SimError;
use crate::common::math::DVec3;
use crate::scf::body::Body;
use crate::scf::surface::SurfaceIndex;

const SIGMA_SB: f64 = 5.670_374_419e-8;
const C_LIGHT: f64 = 299_792_458.0;
const AU: f64 = 1.495_978_707e11;
/// Nominal solar radius (m, IAU 2015 Resolution B3).
const R_SUN: f64 = 6.957e8;
/// Hemisphere directions of the occlusion indicator.
const SKY_SAMPLES: usize = 16;

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ThermalParams {
    /// Heliocentric orbit: semi-major axis (AU) and eccentricity.
    pub semi_major_axis_au: f64,
    pub eccentricity: f64,
    /// Pole: obliquity and longitude from perihelion in the orbit plane (degrees).
    pub obliquity_deg: f64,
    pub pole_longitude_deg: f64,
    /// Solar flux at 1 AU (W m⁻²).
    pub solar_flux: f64,
    pub albedo: f64,
    pub emissivity: f64,
    /// Thermal effusivity `Υ` (J m⁻² K⁻¹ s⁻¹ᐟ²) and volumetric heat capacity of the skin.
    pub effusivity: f64,
    pub heat_capacity: f64,
    /// The same pair for the loose cover, which sets the shielding depth `δ_1^reg`.
    pub cover_effusivity: f64,
    pub cover_heat_capacity: f64,
    /// `false` keeps the thermal solution for the torque only (`r_Θ ≡ 0`).
    pub fatigue: bool,
    /// Basquin–Miner: `N_Θ` cycles to failure at the range `ΔT_Θ` (K), exponent `m_Θ`.
    pub cycles_ref: f64,
    pub delta_t_ref: f64,
    pub exponent: f64,
    /// Fatigue-active depth `ℓ_Θ` (m) and healing rate `h_ϖ` (s⁻¹).
    pub active_depth: f64,
    pub healing: f64,
    /// `θ_Θ` of the step bound.
    pub theta_step: f64,
    pub n_orbit: usize,
    pub n_phase: usize,
    /// Visibility points per face: 1 (centre) or 3.
    pub n_vis: usize,
    /// Face temperatures are reused while `|Δω| / ω` stays below this (0 recomputes).
    pub omega_tolerance: f64,
    /// A satellite's primary (§11.6): its effective temperature (K) and Bond albedo. Its
    /// radius and distance come from the orbit of A-O.
    #[serde(default)]
    pub primary: Option<PrimaryLight>,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PrimaryLight {
    pub temperature: f64,
    pub albedo: f64,
}

/// The fraction of the Sun's disk (angular radius `a_s`) left visible by the primary's
/// (angular radius `a_p`) at the angular separation `theta`: two disks in the tangent plane.
pub fn eclipse_visible(a_s: f64, a_p: f64, theta: f64) -> f64 {
    let overlap = if theta >= a_s + a_p {
        0.0
    } else if theta <= (a_p - a_s).abs() {
        std::f64::consts::PI * a_s.min(a_p).powi(2)
    } else {
        let lens = |r: f64, q: f64| r * r * ((theta * theta + r * r - q * q) / (2.0 * theta * r)).clamp(-1.0, 1.0).acos();
        let k = (-theta + a_s + a_p) * (theta + a_s - a_p) * (theta - a_s + a_p) * (theta + a_s + a_p);
        lens(a_s, a_p) + lens(a_p, a_s) - 0.5 * k.max(0.0).sqrt()
    };
    1.0 - overlap / (std::f64::consts::PI * a_s * a_s)
}

/// The irradiance per unit `[n·e_x]_+` from the primary's heat and reflected sunlight
/// (§11.6), at solar flux `flux` with the Sun along `sun`; `(R_p/a)` is `ratio`.
pub fn primary_irradiance(p: &PrimaryLight, ratio: f64, flux: f64, sun: DVec3) -> f64 {
    let alpha = (-sun.x).clamp(-1.0, 1.0).acos();
    let phase = (alpha.sin() + (std::f64::consts::PI - alpha) * alpha.cos()) / std::f64::consts::PI;
    ratio * ratio * (SIGMA_SB * p.temperature.powi(4) + 2.0 / 3.0 * p.albedo * flux * phase)
}

impl ThermalParams {
    pub fn validate(&self) -> Result<(), SimError> {
        let pos = |name: &str, v: f64| {
            if v.is_finite() && v > 0.0 {
                Ok(())
            } else {
                Err(SimError::model_not_closed(format!("thermal.{name} must be positive, got {v}")))
            }
        };
        pos("semi_major_axis_au", self.semi_major_axis_au)?;
        pos("solar_flux", self.solar_flux)?;
        pos("emissivity", self.emissivity)?;
        pos("effusivity", self.effusivity)?;
        pos("heat_capacity", self.heat_capacity)?;
        pos("cover_effusivity", self.cover_effusivity)?;
        pos("cover_heat_capacity", self.cover_heat_capacity)?;
        pos("theta_step", self.theta_step)?;
        if self.fatigue {
            pos("cycles_ref", self.cycles_ref)?;
            pos("delta_t_ref", self.delta_t_ref)?;
            pos("exponent", self.exponent)?;
            pos("active_depth", self.active_depth)?;
        }
        let unit = |name: &str, v: f64| {
            if v.is_finite() && (0.0..1.0).contains(&v) {
                Ok(())
            } else {
                Err(SimError::model_not_closed(format!("thermal.{name} must lie in [0, 1), got {v}")))
            }
        };
        unit("eccentricity", self.eccentricity)?;
        unit("albedo", self.albedo)?;
        if let Some(p) = &self.primary {
            pos("primary.temperature", p.temperature)?;
            unit("primary.albedo", p.albedo)?;
        }
        if !(self.emissivity <= 1.0) {
            return Err(SimError::model_not_closed("thermal.emissivity must not exceed 1"));
        }
        if !(self.healing.is_finite() && self.healing >= 0.0)
            || !(self.omega_tolerance.is_finite() && self.omega_tolerance >= 0.0)
        {
            return Err(SimError::model_not_closed(
                "thermal.healing and thermal.omega_tolerance must be non-negative",
            ));
        }
        if self.n_orbit == 0 || !(8..=64).contains(&self.n_phase) || self.n_phase % 2 != 0 {
            return Err(SimError::model_not_closed(
                "thermal.n_orbit must be positive and thermal.n_phase even in 8..=64",
            ));
        }
        if self.n_vis != 1 && self.n_vis != 3 {
            return Err(SimError::model_not_closed("thermal.n_vis must be 1 or 3"));
        }
        Ok(())
    }

    /// Shielding depth `δ_1^reg = √(2 κ_reg / ω)` of the loose cover (m).
    fn cover_depth(&self, omega: f64) -> f64 {
        let kappa = (self.cover_effusivity / self.cover_heat_capacity).powi(2);
        (2.0 * kappa / omega.max(1e-30)).sqrt()
    }
}

/// Per-step diagnostics of §11.10 under A-Θ.
#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct ThermalReport {
    pub active: bool,
    pub faces_recomputed: usize,
    pub rays_traced: usize,
    pub max_delta_t: f64,
    pub min_temperature: f64,
    pub mean_rate: f64,
    pub max_rate: f64,
    /// Mass moved from the rock into `ρ_s` this step (kg).
    pub produced_mass: f64,
    pub max_surface_damage: f64,
    /// Diurnal skin depth over the smallest face size.
    pub skin_ratio: f64,
    pub mean_occlusion: f64,
    pub rotation_to_orbit: f64,
    /// Radiative axial torque of the resolved shape (N m).
    pub torque: f64,
}

impl ThermalReport {
    pub fn inactive(&self) -> bool {
        !self.active
    }
}

struct OrbitSample {
    flux: f64,
    /// Heliocentric distance (AU).
    r_au: f64,
    /// `cos δ`, `sin δ` of the subsolar latitude.
    cos_d: f64,
    sin_d: f64,
}

/// Mean anomalies equally spaced in time; Kepler's equation by Newton.
fn orbit_samples(p: &ThermalParams) -> Vec<OrbitSample> {
    let e = p.eccentricity;
    let (eps, lam) = (p.obliquity_deg.to_radians(), p.pole_longitude_deg.to_radians());
    (0..p.n_orbit)
        .map(|j| {
            let m = 2.0 * std::f64::consts::PI * (j as f64 + 0.5) / p.n_orbit as f64;
            let mut ea = m;
            for _ in 0..30 {
                let d = (ea - e * ea.sin() - m) / (1.0 - e * ea.cos());
                ea -= d;
                if d.abs() < 1e-14 {
                    break;
                }
            }
            let f = 2.0 * ((1.0 + e).sqrt() * (ea / 2.0).sin()).atan2((1.0 - e).sqrt() * (ea / 2.0).cos());
            let r_au = p.semi_major_axis_au * (1.0 - e * ea.cos());
            let sin_d = -(eps.sin() * (f - lam).cos());
            OrbitSample {
                flux: p.solar_flux / (r_au * r_au),
                r_au,
                cos_d: (1.0 - sin_d * sin_d).max(0.0).sqrt(),
                sin_d,
            }
        })
        .collect()
}

/// First row of the real circulant `Q_ω` for `ω = 1` (multiply by `√ω`).
fn circulant_row(n: usize, effusivity: f64) -> Vec<f64> {
    let half = n / 2;
    (0..n)
        .map(|d| {
            let theta = 2.0 * std::f64::consts::PI * d as f64 / n as f64;
            let mut c = 0.0;
            for k in 1..half {
                let a = effusivity * (k as f64 / 2.0).sqrt();
                c += 2.0 * a * ((k as f64 * theta).cos() - (k as f64 * theta).sin());
            }
            c += effusivity * (half as f64 / 2.0).sqrt() * (half as f64 * theta).cos();
            c / n as f64
        })
        .collect()
}

/// Newton on `G(T) = εσ T|T|³ + Q T − S` with dense LU; `t` holds the warm start.
fn solve_periodic(q: &[f64], es: f64, s: &[f64], t: &mut [f64]) {
    let n = s.len();
    let mut jac = vec![0.0; n * n];
    let mut g = vec![0.0; n];
    let residual = |t: &[f64], g: &mut [f64]| {
        let mut norm = 0.0f64;
        for m in 0..n {
            let mut qt = 0.0;
            for k in 0..n {
                qt += q[(m + n - k) % n] * t[k];
            }
            g[m] = es * t[m] * t[m].abs().powi(3) + qt - s[m];
            norm = norm.max(g[m].abs());
        }
        norm
    };
    let mut norm = residual(t, &mut g);
    let scale = s.iter().fold(0.0f64, |a, v| a.max(*v)).max(1e-12);
    for _ in 0..30 {
        if norm <= 1e-10 * scale {
            break;
        }
        for m in 0..n {
            for k in 0..n {
                jac[m * n + k] = q[(m + n - k) % n];
            }
            jac[m * n + m] += 4.0 * es * t[m].abs().powi(3);
        }
        let dx = lu_solve(&mut jac, &mut g.clone(), n);
        // Backtrack on the max-norm of G.
        let mut step = 1.0;
        let base = t.to_vec();
        loop {
            for m in 0..n {
                t[m] = base[m] - step * dx[m];
            }
            let next = residual(t, &mut g);
            if next < norm || step < 1e-3 {
                norm = next;
                break;
            }
            step *= 0.5;
        }
    }
}

/// Gaussian elimination with partial pivoting; `a` and `b` are overwritten.
fn lu_solve(a: &mut [f64], b: &mut [f64], n: usize) -> Vec<f64> {
    for c in 0..n {
        let p = (c..n).max_by(|x, y| a[x * n + c].abs().total_cmp(&a[y * n + c].abs())).unwrap_or(c);
        if p != c {
            for k in 0..n {
                a.swap(c * n + k, p * n + k);
            }
            b.swap(c, p);
        }
        let d = a[c * n + c];
        if d.abs() < 1e-300 {
            continue;
        }
        for r in c + 1..n {
            let f = a[r * n + c] / d;
            if f != 0.0 {
                for k in c..n {
                    a[r * n + k] -= f * a[c * n + k];
                }
                b[r] -= f * b[c];
            }
        }
    }
    let mut x = vec![0.0; n];
    for c in (0..n).rev() {
        let mut v = b[c];
        for k in c + 1..n {
            v -= a[c * n + k] * x[k];
        }
        let d = a[c * n + c];
        x[c] = if d.abs() < 1e-300 { 0.0 } else { v / d };
    }
    x
}

/// Cached geometry of the last evaluation.
struct Cache {
    positions: Vec<DVec3>,
    /// Shadow bits, `N_vis × N_orb × N_φ` per face (`1` = sunlit point).
    lit: Vec<u64>,
    words: usize,
    occlusion: Vec<f32>,
    /// Face temperatures `N_orb × N_φ` (K), the warm start of the next solve.
    temps: Vec<f32>,
    /// Spin of the stored temperatures; `fresh` marks the faces solved at it.
    omega: f64,
}

/// A-Θ for one run.
pub struct Thermal<'a> {
    p: &'a ThermalParams,
    orbit: Vec<OrbitSample>,
    q1: Vec<f64>,
    cache: Option<Cache>,
    /// `(R_p, a)` of a satellite's primary on the `+x` axis.
    primary: Option<(f64, f64)>,
}

/// One evaluation on the current shape.
pub struct ThermalOut {
    /// Fatigue rate `r_Θ` per vertex (s⁻¹), area-weighted from the faces.
    pub rate: Vec<f64>,
    pub report: ThermalReport,
}

impl<'a> Thermal<'a> {
    pub fn new(p: &'a ThermalParams) -> Self {
        Self {
            p,
            orbit: orbit_samples(p),
            q1: circulant_row(p.n_phase, p.effusivity),
            cache: None,
            primary: None,
        }
    }

    /// The primary's radius and distance for the next evaluation (§11.6). A change of
    /// distance re-solves every face, as a change of `ω` does.
    pub fn set_primary(&mut self, radius: f64, distance: f64) {
        if self.p.primary.is_some() {
            if self.primary.is_some_and(|(_, a)| a != distance) {
                if let Some(c) = self.cache.as_mut() {
                    c.omega = 0.0;
                }
            }
            self.primary = Some((radius, distance));
        }
    }

    fn sun(&self, j: usize, m: usize) -> DVec3 {
        let o = &self.orbit[j];
        let phi = 2.0 * std::f64::consts::PI * m as f64 / self.p.n_phase as f64;
        DVec3::new(o.cos_d * phi.cos(), -o.cos_d * phi.sin(), o.sin_d)
    }

    fn vis_points(&self, tri: [DVec3; 3], n: DVec3, lift: f64) -> Vec<DVec3> {
        let c = (tri[0] + tri[1] + tri[2]) / 3.0;
        let mut pts = vec![c];
        if self.p.n_vis == 3 {
            pts = (0..3).map(|k| tri[k] * (2.0 / 3.0) + (tri[(k + 1) % 3] + tri[(k + 2) % 3]) * (1.0 / 6.0)).collect();
        }
        pts.into_iter().map(|x| x + n * lift).collect()
    }

    /// Shadow bits and the occlusion indicator of face `f`.
    fn trace_face(&self, index: &SurfaceIndex, tri: [DVec3; 3], lift: f64, words: usize) -> (Vec<u64>, f32, usize) {
        let n = (tri[1] - tri[0]).cross(tri[2] - tri[0]).normalize_or_zero();
        let pts = self.vis_points(tri, n, lift);
        let dirs = self.p.n_orbit * self.p.n_phase;
        let mut bits = vec![0u64; words];
        let mut rays = 0;
        for (v, x) in pts.iter().enumerate() {
            for j in 0..self.p.n_orbit {
                for m in 0..self.p.n_phase {
                    let s = self.sun(j, m);
                    if n.dot(s) <= 0.0 {
                        continue;
                    }
                    rays += 1;
                    if index.first_hit(*x, s, 0.0, f64::INFINITY).is_none() {
                        let b = v * dirs + j * self.p.n_phase + m;
                        bits[b / 64] |= 1 << (b % 64);
                    }
                }
            }
        }
        let (e1, e2) = crate::rheo::model::local_frame(n);
        let golden = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
        let blocked = (0..SKY_SAMPLES)
            .filter(|&k| {
                let z = (k as f64 + 0.5) / SKY_SAMPLES as f64;
                let r = (1.0 - z * z).sqrt();
                let a = golden * k as f64;
                let d = e1 * (r * a.cos()) + e2 * (r * a.sin()) + n * z;
                index.first_hit(pts[0], d, 0.0, f64::INFINITY).is_some()
            })
            .count();
        (bits, blocked as f32 / SKY_SAMPLES as f32, rays + SKY_SAMPLES)
    }

    /// Thermal state of the current shape at spin `omega`, with `centroid` the centre of mass.
    pub fn evaluate(&mut self, body: &Body, index: &SurfaceIndex, omega: f64, centroid: DVec3) -> ThermalOut {
        let _span = crate::scf::timing::span("thermal");
        let p = self.p;
        let faces = &body.topo.faces;
        let x = &body.positions;
        let nf = faces.len();
        let (no, np) = (p.n_orbit, p.n_phase);
        let dirs = no * np;
        let words = (p.n_vis * dirs).div_ceil(64);
        let tri = |f: usize| faces[f].map(|v| x[v as usize]);
        let lift = 1e-6 * body.r_mean();

        // Faces whose shadows must be retraced: all on the first call or when most of the
        // body moved (a realignment); otherwise the moved faces and the faces whose cached
        // rays may cross a moved triangle (the dirty test of Proposition 3.15.2).
        let moved: Vec<bool> = match &self.cache {
            Some(c) => x.iter().zip(&c.positions).map(|(a, b)| a != b).collect(),
            None => vec![true; x.len()],
        };
        let n_moved = moved.iter().filter(|m| **m).count();
        let mut retrace = vec![false; nf];
        let full = self.cache.is_none() || n_moved * 4 > x.len();
        if full {
            retrace.iter_mut().for_each(|r| *r = true);
        } else if n_moved > 0 {
            let cache = self.cache.as_ref().expect("cache");
            let changed: Vec<usize> = (0..nf).filter(|&f| faces[f].iter().any(|v| moved[*v as usize])).collect();
            let (mut lo, mut hi) = (DVec3::splat(f64::INFINITY), DVec3::splat(f64::NEG_INFINITY));
            for &f in &changed {
                retrace[f] = true;
                for v in faces[f] {
                    for q in [x[v as usize], cache.positions[v as usize]] {
                        lo = lo.min(q);
                        hi = hi.max(q);
                    }
                }
            }
            let flags: Vec<bool> = (0..nf)
                .into_par_iter()
                .map(|f| {
                    if retrace[f] {
                        return true;
                    }
                    let t = tri(f);
                    let n = (t[1] - t[0]).cross(t[2] - t[0]).normalize_or_zero();
                    self.vis_points(t, n, lift).iter().any(|o| {
                        (0..no).any(|j| (0..np).any(|m| {
                            let s = self.sun(j, m);
                            n.dot(s) > 0.0 && ray_box(*o, s, lo, hi)
                        }))
                    })
                })
                .collect();
            retrace = flags;
        }

        let mut cache = self.cache.take().unwrap_or_else(|| Cache {
            positions: Vec::new(),
            lit: vec![0; nf * words],
            words,
            occlusion: vec![0.0; nf],
            temps: vec![0.0; nf * dirs],
            omega: 0.0,
        });
        let traced: Vec<(usize, Vec<u64>, f32, usize)> = (0..nf)
            .into_par_iter()
            .filter(|f| retrace[*f])
            .map(|f| {
                let (bits, occ, rays) = self.trace_face(index, tri(f), lift, words);
                (f, bits, occ, rays)
            })
            .collect();
        let mut rays_traced = 0;
        for (f, bits, occ, rays) in traced {
            cache.lit[f * words..(f + 1) * words].copy_from_slice(&bits);
            cache.occlusion[f] = occ;
            rays_traced += rays;
        }

        // Temperatures: faces with new insolation, or all when ω left the tolerance.
        let respin = cache.omega <= 0.0 || (omega / cache.omega - 1.0).abs() > p.omega_tolerance;
        let resolve: Vec<bool> = (0..nf).map(|f| respin || retrace[f]).collect();
        let es = p.emissivity * SIGMA_SB;
        let q: Vec<f64> = self.q1.iter().map(|c| c * omega.max(0.0).sqrt()).collect();
        let lit = |c: &Cache, f: usize, v: usize, j: usize, m: usize| {
            let b = v * dirs + j * np + m;
            (c.lit[f * c.words + b / 64] >> (b % 64)) & 1 == 1
        };
        let solved: Vec<(usize, Vec<f32>)> = (0..nf)
            .into_par_iter()
            .filter(|f| resolve[*f])
            .map(|f| {
                let t = tri(f);
                let n = (t[1] - t[0]).cross(t[2] - t[0]).normalize_or_zero();
                let mut out = vec![0f32; dirs];
                let mut s = vec![0.0; np];
                let mut temp = vec![0.0; np];
                for j in 0..no {
                    for m in 0..np {
                        let sun = self.sun(j, m);
                        let lit_frac = (0..p.n_vis).filter(|v| lit(&cache, f, *v, j, m)).count() as f64
                            / p.n_vis as f64;
                        let flux = self.orbit[j].flux;
                        s[m] = (1.0 - p.albedo) * flux * n.dot(sun).max(0.0) * lit_frac;
                        if let (Some(light), Some((r_p, a))) = (&p.primary, self.primary) {
                            let a_s = R_SUN / (self.orbit[j].r_au * AU);
                            let seen = eclipse_visible(a_s, (r_p / a).min(1.0).asin(), sun.x.clamp(-1.0, 1.0).acos());
                            s[m] = s[m] * seen
                                + (1.0 - p.albedo) * primary_irradiance(light, r_p / a, flux, sun) * n.x.max(0.0);
                        }
                    }
                    let warm = &cache.temps[f * dirs + j * np..f * dirs + (j + 1) * np];
                    if warm.iter().all(|v| *v > 0.0) {
                        temp.iter_mut().zip(warm).for_each(|(a, b)| *a = f64::from(*b));
                    } else {
                        let mean = s.iter().sum::<f64>() / np as f64;
                        temp.iter_mut().for_each(|v| *v = (mean / es).max(1e-6).powf(0.25));
                    }
                    solve_periodic(&q, es, &s, &mut temp);
                    for m in 0..np {
                        out[j * np + m] = temp[m] as f32;
                    }
                }
                (f, out)
            })
            .collect();
        let faces_recomputed = solved.len();
        for (f, t) in solved {
            cache.temps[f * dirs..(f + 1) * dirs].copy_from_slice(&t);
        }
        if respin {
            cache.omega = omega;
        }

        // Fatigue rate per face, the radiative torque, and the indicators.
        let mut report = ThermalReport {
            active: true,
            faces_recomputed,
            rays_traced,
            min_temperature: f64::INFINITY,
            rotation_to_orbit: 0.0,
            ..ThermalReport::default()
        };
        let mut face_rate = vec![0.0; nf];
        let mut torque = 0.0;
        let mut min_size = f64::INFINITY;
        for f in 0..nf {
            let t = tri(f);
            let cross = (t[1] - t[0]).cross(t[2] - t[0]);
            let area = 0.5 * cross.length();
            let n = cross.normalize_or_zero();
            let xf = (t[0] + t[1] + t[2]) / 3.0 - centroid;
            min_size = min_size.min(area.sqrt());
            let mut sum = 0.0;
            for j in 0..no {
                let row = &cache.temps[f * dirs + j * np..f * dirs + (j + 1) * np];
                let (lo, hi) = row.iter().fold((f64::INFINITY, 0.0f64), |(a, b), v| (a.min(f64::from(*v)), b.max(f64::from(*v))));
                report.max_delta_t = report.max_delta_t.max(hi - lo);
                report.min_temperature = report.min_temperature.min(lo);
                if p.fatigue {
                    sum += ((hi - lo) / p.delta_t_ref).powf(p.exponent);
                }
                for m in 0..np {
                    let sun = self.sun(j, m);
                    let lit_frac =
                        (0..p.n_vis).filter(|v| lit(&cache, f, *v, j, m)).count() as f64 / p.n_vis as f64;
                    let inc = area * self.orbit[j].flux * n.dot(sun).max(0.0) * lit_frac / C_LIGHT;
                    let emit = area * es * f64::from(row[m]).powi(4) / C_LIGHT;
                    let force = -(sun + n * (2.0 / 3.0 * p.albedo)) * inc - n * (2.0 / 3.0 * emit);
                    torque += xf.cross(force).z;
                }
            }
            if p.fatigue {
                face_rate[f] = omega / (2.0 * std::f64::consts::PI * p.cycles_ref) * sum / no as f64;
            }
            report.mean_occlusion += f64::from(cache.occlusion[f]) / nf as f64;
        }
        report.torque = torque / dirs as f64;
        let kappa = (p.effusivity / p.heat_capacity).powi(2);
        report.skin_ratio = (2.0 * kappa / omega.max(1e-30)).sqrt() / min_size.max(1e-300);
        let year = 2.0 * std::f64::consts::PI * ((p.semi_major_axis_au * AU).powi(3) / 1.327_124_400_18e20).sqrt();
        report.rotation_to_orbit = if omega > 0.0 { 2.0 * std::f64::consts::PI / omega / year } else { 0.0 };

        let mut rate = vec![0.0; x.len()];
        let mut weight = vec![0.0; x.len()];
        for f in 0..nf {
            let t = tri(f);
            let a = 0.5 * (t[1] - t[0]).cross(t[2] - t[0]).length();
            for v in faces[f] {
                rate[v as usize] += a * face_rate[f];
                weight[v as usize] += a;
            }
        }
        rate.iter_mut().zip(&weight).for_each(|(r, w)| *r /= w.max(1e-300));
        report.mean_rate = rate.iter().sum::<f64>() / rate.len().max(1) as f64;
        report.max_rate = rate.iter().copied().fold(0.0, f64::max);
        cache.positions = x.clone();
        self.cache = Some(cache);
        ThermalOut { rate, report }
    }

    /// Step bound `Δt_Θ = θ_Θ / max_i (r_Θ,i S_Θ,i + h_ϖ)` (§11.6).
    pub fn dt_bound(&self, rate: &[f64], rho_s: &[f64], rho_bulk: f64, omega: f64) -> f64 {
        let delta = self.p.cover_depth(omega);
        let worst = rate
            .iter()
            .zip(rho_s)
            .map(|(r, s)| r * shield(s / rho_bulk, self.p.exponent, delta) + self.p.healing)
            .fold(0.0, f64::max);
        if worst > 0.0 { self.p.theta_step / worst } else { f64::INFINITY }
    }

    /// Proposition 11.11 over `τ`: the surface damage and the shielded production of loose
    /// material, both from the cover at the step start. Returns the produced mass.
    pub fn advance(
        &self,
        tau: f64,
        omega: f64,
        rate: &[f64],
        area: &[f64],
        rho_bulk: f64,
        damage: &mut [f64],
        rho_s: &mut [f64],
    ) -> f64 {
        let p = self.p;
        let delta = p.cover_depth(omega);
        let mut produced = 0.0;
        for i in 0..damage.len() {
            let h = rho_s[i] / rho_bulk;
            let a = rate[i] * shield(h, p.exponent, delta);
            damage[i] = exact_damage(damage[i], a, p.healing, tau);
            if rate[i] > 0.0 && p.fatigue {
                let y = p.exponent * p.active_depth * rate[i] * tau / delta;
                let x = p.exponent * h / delta;
                // ln(e^x + y) without overflow for a thick cover.
                let h_new = delta / p.exponent * (x + (y * (-x).exp()).ln_1p());
                let s_new = rho_bulk * h_new;
                produced += (s_new - rho_s[i]) * area[i];
                rho_s[i] = s_new;
            }
        }
        produced
    }
}

/// `S_Θ(h) = exp(−m_Θ h / δ_1^reg)`.
fn shield(h: f64, m: f64, delta: f64) -> f64 {
    (-m * h.max(0.0) / delta.max(1e-300)).exp()
}

/// Proposition 11.9: `ϖ' = ϖ_∞ + (ϖ − ϖ_∞) e^{−(a+h)τ}`, `ϖ_∞ = a / (a + h)`.
pub fn exact_damage(w: f64, a: f64, h: f64, tau: f64) -> f64 {
    let k = a + h;
    if !(k > 0.0) {
        return w;
    }
    let inf = a / k;
    (inf + (w - inf) * (-k * tau).exp()).clamp(0.0, 1.0)
}

/// Slab test of the ray `o + t d`, `t > 0`, against an axis-aligned box.
fn ray_box(o: DVec3, d: DVec3, lo: DVec3, hi: DVec3) -> bool {
    let (mut t0, mut t1) = (0.0f64, f64::INFINITY);
    for k in 0..3 {
        let (oo, dd, l, h) = (o[k], d[k], lo[k], hi[k]);
        if dd.abs() < 1e-300 {
            if oo < l || oo > h {
                return false;
            }
            continue;
        }
        let (a, b) = ((l - oo) / dd, (h - oo) / dd);
        t0 = t0.max(a.min(b));
        t1 = t1.min(a.max(b));
        if t0 > t1 {
            return false;
        }
    }
    true
}
