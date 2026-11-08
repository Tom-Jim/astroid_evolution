//! Mass, centroid and inertia of the whole density (bulk plus every core).
//!
//! Rays leave the polyhedron's volume centroid on the Lebedev–Laikov rule of the
//! field. Occupied intervals and the radial moments run on the GPU
//! (`rtfp/shader/moments.wesl`); the body need not be star-shaped about the
//! centroid. The bulk part is closed per segment and each core uses the composite
//! rule of §3.3, so a narrow core is resolved whatever its width.

use crate::common::error::SimError;
use crate::common::math::DVec3;
use crate::gpu::{Job, kernels};
use crate::rtfp::RtfpSolver;
use crate::rtfp::api::VolumeIntegrals;
use crate::rtfp::lebedev;
use crate::rtfp::volume::uniform_integrals;

const STRIDE: usize = 21;

impl RtfpSolver {
    /// Volume, mass, centroid and origin inertia of the density on the bound mesh.
    pub fn density_integrals(&self) -> Result<VolumeIntegrals, SimError> {
        let geometric = uniform_integrals(&self.mesh, 1.0);
        if !(geometric.volume.is_finite() && geometric.volume > 0.0) {
            return Err(SimError::geometry(
                "closed-mesh volume is not a finite positive number",
            ));
        }
        if self.density.kernels.is_empty() {
            return Ok(uniform_integrals(&self.mesh, self.density.bulk));
        }
        // ρ = c + (ρ − c): the constant part exactly by the divergence theorem, the
        // rest along the rays. c is the mean density over the vertices, so the
        // integrand the angular rule sees is small where the chords end.
        let c_rho = self
            .mesh
            .vertices
            .iter()
            .map(|v| self.density.rho(*v))
            .sum::<f64>()
            / self.mesh.vertices.len() as f64;
        let exact = uniform_integrals(&self.mesh, c_rho);
        let frame = self.body.frame;
        let o = frame.to_unit(geometric.centroid);
        let kerns = self.unit_kernels();
        let n = lebedev::COUNT;
        let k = &kernels::RTFP_MOMENTS;
        let mut job = Job::new();
        let origin = job.value(&[o.x as f32, o.y as f32, o.z as f32, 0.0]);
        let cores = job.input(&kerns);
        let mut chord = self.chord_params(n as u32, false, false, false, 0);
        chord.bulk = (self.density.bulk - c_rho) as f32;
        let params = job.value(&chord);
        let res = job.zeros::<f32>(n * STRIDE);
        job.run(
            k,
            k.groups(n),
            &[
                origin,
                (&self.rays).into(),
                cores,
                (&self.nodes).into(),
                params,
                res,
                (&self.body.accel).into(),
            ],
        );
        let r = job.read_all::<f32>(res, n * STRIDE);
        let raw = job.submit().map_err(SimError::gpu)?.get::<f32>(r);
        let mut m = [0.0f64; 10];
        let mut anomalies = 0u32;
        for col in raw.chunks_exact(STRIDE) {
            let flags = col[20] as u32;
            if flags & 1 != 0 {
                return Err(SimError::geometry(
                    "a moment ray crosses the surface more than 64 times",
                ));
            }
            anomalies += flags >> 1;
            for (i, v) in m.iter_mut().enumerate() {
                *v += f64::from(col[2 * i]) + f64::from(col[2 * i + 1]);
            }
        }
        // Unit frame → world: y = c + L y_u, dV = L³ dV_u.
        let (c, l) = (frame.center, frame.scale);
        let l3 = l * l * l;
        let mass = l3 * m[0] + exact.mass;
        if !(mass.is_finite() && mass > 0.0) {
            return Err(SimError::geometry(
                "Cauchy mass integral is not a finite positive mass",
            ));
        }
        let m1 = DVec3::new(m[1], m[2], m[3]);
        let first = (c * m[0] + m1 * l) * l3 + exact.centroid * exact.mass;
        let centroid = first / mass;
        // ∫ρ y_a y_b = L³ [c_a c_b M0 + L (c_a M1_b + M1_a c_b) + L² M2_ab].
        let ca = c.to_array();
        let ma = m1.to_array();
        let second = |a: usize, b: usize, k: usize| {
            l3 * (ca[a] * ca[b] * m[0] + l * (ca[a] * ma[b] + ma[a] * ca[b]) + l * l * m[k])
        };
        let (xx, yy, zz) = (second(0, 0, 4), second(1, 1, 5), second(2, 2, 6));
        let (xy, xz, yz) = (second(0, 1, 7), second(0, 2, 8), second(1, 2, 9));
        let e = exact.inertia;
        Ok(VolumeIntegrals {
            volume: geometric.volume,
            mass,
            centroid,
            anomalies,
            inertia: [
                yy + zz + e[0],
                xx + zz + e[1],
                xx + yy + e[2],
                e[3] - xy,
                e[4] - xz,
                e[5] - yz,
            ],
        })
    }
}
