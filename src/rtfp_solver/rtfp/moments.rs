//! Mass, centroid and inertia of the whole density (bulk plus every Cauchy kernel).
//!
//! Rays leave the polyhedron's volume centroid on the same Lebedev–Laikov rule
//! as the field. Occupied intervals and the density quadrature run on the GPU.
//! The body need not be star-shaped about the centroid.

use super::{RtfpSolver, hw_rt::DeviceKernel, lebedev};
use crate::common::math::DVec3;
use crate::rtfp_solver::api::VolumeIntegrals;
use crate::rtfp_solver::error::RtfpError;
use crate::rtfp_solver::volume::uniform_integrals;

const MAX_PANELS: u32 = 64;

impl RtfpSolver {
    /// Volume, mass, centroid and origin inertia of the density on the bound mesh.
    pub fn density_integrals(&self) -> Result<VolumeIntegrals, RtfpError> {
        let geometric = uniform_integrals(&self.mesh, 1.0);
        if !(geometric.volume.is_finite() && geometric.volume > 0.0) {
            return Err(RtfpError::geometry(
                "closed-mesh volume is not a finite positive number",
            ));
        }
        if self.density.kernels.is_empty() {
            return Ok(uniform_integrals(&self.mesh, self.density.bulk));
        }
        let origin = geometric.centroid;
        let sphere = lebedev::rule();
        let kernels: Vec<DeviceKernel> = self
            .medium
            .kernels()
            .iter()
            .map(|k| DeviceKernel {
                center: k.center,
                inv_sigma2: k.inv_sigma2,
                coeff: k.coeff,
                alpha: k.alpha,
            })
            .collect();
        let finest = 1.0
            / self
                .density
                .kernels
                .iter()
                .map(|k| k.sigma)
                .fold(0.0_f64, f64::max);
        let total = self.hw.integrate_moments(
            origin,
            &sphere.dirs,
            &sphere.weights,
            self.medium.bulk(),
            &kernels,
            &self.quad[..self.quad_base],
            finest,
            MAX_PANELS,
        )?;
        let mass = total[0];
        if !(mass.is_finite() && mass > 0.0) {
            return Err(RtfpError::geometry(
                "Cauchy mass integral is not a finite positive mass",
            ));
        }
        let centroid = DVec3::new(total[1], total[2], total[3]) * (1.0 / mass);
        let (xx, yy, zz, xy, xz, yz) = (total[4], total[5], total[6], total[7], total[8], total[9]);
        Ok(VolumeIntegrals {
            volume: geometric.volume,
            mass,
            centroid,
            inertia: [yy + zz, xx + zz, xx + yy, -xy, -xz, -yz],
        })
    }
}
