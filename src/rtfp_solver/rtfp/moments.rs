//! Mass, centroid and inertia of the whole density (bulk plus every Cauchy kernel).
//!
//! Rays leave the polyhedron's volume centroid through the same face-partition
//! patches as the field. Each ray's occupied intervals come from the hardware
//! crossings, so the body need not be star-shaped about the centroid.

use gauss_quad::GaussLegendre;
use rayon::prelude::*;
use std::num::NonZeroUsize;

use super::{RtfpSolver, chords, patches};
use crate::common::math::DVec3;
use crate::rtfp_solver::api::VolumeIntegrals;
use crate::rtfp_solver::error::RtfpError;
use crate::rtfp_solver::volume::uniform_integrals;

const NODES: usize = 16;
const MAX_PANELS: usize = 64;

#[derive(Clone, Copy, Default)]
struct Moments {
    mass: f64,
    first: [f64; 3],
    second: [f64; 6],
}

impl Moments {
    fn add(&mut self, y: DVec3, m: f64) {
        self.mass += m;
        self.first[0] += m * y.x;
        self.first[1] += m * y.y;
        self.first[2] += m * y.z;
        self.second[0] += m * y.x * y.x;
        self.second[1] += m * y.y * y.y;
        self.second[2] += m * y.z * y.z;
        self.second[3] += m * y.x * y.y;
        self.second[4] += m * y.x * y.z;
        self.second[5] += m * y.y * y.z;
    }

    fn merge(mut self, o: Moments) -> Moments {
        self.mass += o.mass;
        for i in 0..3 {
            self.first[i] += o.first[i];
        }
        for i in 0..6 {
            self.second[i] += o.second[i];
        }
        self
    }
}

impl RtfpSolver {
    /// Volume, mass, centroid and origin inertia of the density on the bound mesh.
    pub fn density_integrals(&self) -> Result<VolumeIntegrals, RtfpError> {
        let geometric = uniform_integrals(&self.mesh, 1.0);
        if !(geometric.volume.is_finite() && geometric.volume > 0.0) {
            return Err(RtfpError::geometry("closed-mesh volume is not a finite positive number"));
        }
        if self.density.kernels.is_empty() {
            return Ok(uniform_integrals(&self.mesh, self.density.bulk));
        }
        let origin = geometric.centroid;
        let leaves = patches::settle(&self.hw, &self.faces, &[origin], self.options.angular_ratio)?
            .pop()
            .unwrap_or_default();
        let samples = patches::flatten(origin, &leaves, &self.faces);
        let mut batch = self.hw.new_batch();
        for s in &samples {
            batch.push(origin, s.dir, None);
        }
        let hits = self.hw.trace(&batch, false)?;

        let rule = GaussLegendre::new(NonZeroUsize::new(NODES).expect("nonzero"));
        let finest = 1.0
            / self
                .density
                .kernels
                .iter()
                .map(|k| k.sigma)
                .fold(0.0_f64, f64::max);
        let scale = self.hw.scale();
        let total = (0..samples.len())
            .into_par_iter()
            .fold(
                || (Moments::default(), Vec::new(), Vec::new()),
                |(mut acc, mut dists, mut segs), i| {
                    dists.clear();
                    hits.distances_into(i, &mut dists);
                    chords::intervals(&dists, scale, &mut segs);
                    let s = samples[i];
                    for &(r0, r1) in &segs {
                        let panels = (((r1 - r0) / finest).ceil() as usize).clamp(1, MAX_PANELS);
                        let h = (r1 - r0) / panels as f64;
                        for p in 0..panels {
                            let lo = r0 + h * p as f64;
                            for &(node, weight) in rule.as_node_weight_pairs() {
                                let r = lo + 0.5 * h * (node + 1.0);
                                let y = origin + s.dir * r;
                                acc.add(y, s.weight * self.density.rho(y) * r * r * 0.5 * h * weight);
                            }
                        }
                    }
                    (acc, dists, segs)
                },
            )
            .map(|(acc, _, _)| acc)
            .reduce(Moments::default, Moments::merge);
        if !(total.mass.is_finite() && total.mass > 0.0) {
            return Err(RtfpError::geometry("Cauchy mass integral is not a finite positive mass"));
        }
        let [xx, yy, zz, xy, xz, yz] = total.second;
        Ok(VolumeIntegrals {
            volume: geometric.volume,
            mass: total.mass,
            centroid: DVec3::new(total.first[0], total.first[1], total.first[2]) * (1.0 / total.mass),
            inertia: [yy + zz, xx + zz, xx + yy, -xy, -xz, -yz],
        })
    }
}
