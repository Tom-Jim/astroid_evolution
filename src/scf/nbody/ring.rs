//! The debris ring (§12.5): the surface density `Σ(R, t)` of an axisymmetric ring of
//! particles about the primary, spreading viscously between the primary and the fluid Roche
//! limit, by the conservative finite-volume form of `∂Σ/∂t = (3/R) ∂_R[√R ∂_R(νΣ√R)]`
//! with the viscosity of self-gravitating rings (Salmon et al. 2010, after Daisaka et al. 2001).

use serde::Serialize;

use crate::common::error::SimError;

/// Gravitational constant (m³ kg⁻¹ s⁻²).
const G: f64 = 6.6743e-11;
/// Toomre parameter below which self-gravity wakes form (Salmon et al. 2010).
const Q_WAKES: f64 = 2.0;

/// The ring: cells between the primary's surface and the fluid Roche limit, and the mass
/// that left it through each edge.
#[derive(Clone, Debug, Serialize)]
pub struct Ring {
    /// Cell centres (m), width (m), and the surface density in each (kg m⁻²).
    pub r: Vec<f64>,
    pub dr: f64,
    pub sigma: Vec<f64>,
    /// Mass fallen on the primary, and gone past the Roche limit to form moonlets (kg).
    pub to_primary: f64,
    pub to_moonlets: f64,
    gm: f64,
    particle_radius: f64,
    particle_density: f64,
}

impl Ring {
    /// An empty ring of `cells` from `inner` to the fluid Roche limit of particles of
    /// density `particle_density` about a primary `gm` of radius `inner`.
    pub fn new(gm: f64, inner: f64, cells: usize, particle_radius: f64, particle_density: f64) -> Self {
        let primary_density = 3.0 * gm / (4.0 * std::f64::consts::PI * G * inner.powi(3));
        let roche = 2.456 * inner * (primary_density / particle_density).cbrt();
        let outer = roche.max(inner * 1.01);
        let dr = (outer - inner) / cells as f64;
        Self {
            r: (0..cells).map(|i| inner + (i as f64 + 0.5) * dr).collect(),
            dr,
            sigma: vec![0.0; cells],
            to_primary: 0.0,
            to_moonlets: 0.0,
            gm,
            particle_radius,
            particle_density,
        }
    }

    pub fn inner(&self) -> f64 {
        self.r[0] - 0.5 * self.dr
    }

    pub fn roche(&self) -> f64 {
        self.r[self.r.len() - 1] + 0.5 * self.dr
    }

    fn area(&self, i: usize) -> f64 {
        2.0 * std::f64::consts::PI * self.r[i] * self.dr
    }

    pub fn mass(&self) -> f64 {
        (0..self.r.len()).map(|i| self.sigma[i] * self.area(i)).sum()
    }

    /// Lays `mass` down at the semi-major axis `a`: on the primary inside it, as moonlet
    /// material beyond the Roche limit, else in its cell.
    pub fn deposit(&mut self, a: f64, mass: f64) {
        if a < self.inner() {
            self.to_primary += mass;
        } else if a >= self.roche() {
            self.to_moonlets += mass;
        } else {
            let i = (((a - self.inner()) / self.dr) as usize).min(self.r.len() - 1);
            self.sigma[i] += mass / self.area(i);
        }
    }

    /// `ν = ν_trans + ν_coll + ν_grav` (Salmon et al. 2010, Eqs. 6–10) for particles of
    /// radius `r_p` and density `ρ_p`: with the optical depth `τ = 3Σ/(4 r_p ρ_p)`, the
    /// Hill parameter `r_h* = (2m_p/3M)^{1/3} R / (2 r_p)`, the velocity dispersion
    /// `σ_r = √(G m_p/r_p)` for `r_h* > 0.5` and `2 r_p Ω` below, and the Toomre parameter
    /// `Q = Ω σ_r / (3.36 G Σ)`.
    pub fn viscosity(&self, r: f64, sigma: f64) -> f64 {
        if !(sigma > 0.0) {
            return 0.0;
        }
        let rp = self.particle_radius;
        let omega = (self.gm / r.powi(3)).sqrt();
        let mp = 4.0 / 3.0 * std::f64::consts::PI * rp.powi(3) * self.particle_density;
        let tau = 3.0 * sigma / (4.0 * rp * self.particle_density);
        let rh = (2.0 * mp * G / (3.0 * self.gm)).cbrt() * r / (2.0 * rp);
        let disp = if rh > 0.5 { (G * mp / rp).sqrt() } else { 2.0 * rp * omega };
        let q = omega * disp / (3.36 * G * sigma);
        let coll = rp * rp * omega * tau;
        if q > Q_WAKES {
            disp * disp / (2.0 * omega) * 0.46 * tau / (1.0 + tau * tau) + coll
        } else {
            26.0 * rh.powi(5) * G * G * sigma * sigma / omega.powi(3) + coll
        }
    }

    /// Inward mass flux `6π √R ∂_R(νΣ√R)` (kg s⁻¹) at the faces, with an empty cell beyond
    /// each edge: matter crossing the inner edge falls on the primary, and matter crossing
    /// the Roche limit leaves to form moonlets.
    fn fluxes(&self) -> Vec<f64> {
        let n = self.r.len();
        let g: Vec<f64> = (0..n).map(|i| self.viscosity(self.r[i], self.sigma[i]) * self.sigma[i] * self.r[i].sqrt()).collect();
        (0..=n)
            .map(|f| {
                let rf = self.inner() + f as f64 * self.dr;
                let (lo, hi) = (if f == 0 { 0.0 } else { g[f - 1] }, if f == n { 0.0 } else { g[f] });
                6.0 * std::f64::consts::PI * rf.sqrt() * (hi - lo) / self.dr
            })
            .collect()
    }

    /// The largest stable explicit step: the diffusivity of `νΣ` is at most `3ν (1 + ∂ln ν/∂ln Σ)
    /// ≤ 9ν` with `ν ∝ Σ²` in the wake regime.
    fn stable_step(&self) -> f64 {
        let nu = (0..self.r.len()).map(|i| self.viscosity(self.r[i], self.sigma[i])).fold(0.0, f64::max);
        if nu > 0.0 { 0.2 * self.dr * self.dr / (9.0 * nu) } else { f64::INFINITY }
    }

    /// Advances the ring by `time` (s) in stable explicit steps; returns their number.
    pub fn evolve(&mut self, time: f64) -> Result<usize, SimError> {
        let (mut t, mut steps) = (0.0, 0usize);
        while t < time {
            let dt = self.stable_step().min(time - t);
            if !dt.is_finite() {
                break;
            }
            let flux = self.fluxes();
            let n = self.r.len();
            for i in 0..n {
                let dm = (flux[i + 1] - flux[i]) * dt;
                self.sigma[i] = (self.sigma[i] + dm / self.area(i)).max(0.0);
            }
            self.to_primary += flux[0] * dt;
            self.to_moonlets -= flux[n] * dt;
            t += dt;
            steps += 1;
            if steps > 100_000_000 {
                return Err(SimError::solver("ring: more than 10⁸ steps"));
            }
        }
        Ok(steps)
    }
}
