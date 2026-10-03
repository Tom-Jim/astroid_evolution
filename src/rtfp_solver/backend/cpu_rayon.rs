//! f64 ray integrator. One direction loop allocates nothing beyond the hit buffer.

use crate::common::density::CauchyDensity;
use crate::common::geometry::{FlattenedBvh, TriMesh};
use crate::common::math::{quadrupole, DVec3, Sym6};
use crate::common::par::map_collect;
use crate::rtfp_solver::api::{FieldBatch, FieldMask, PointRole, PointSet};
use crate::rtfp_solver::error::RtfpError;
use crate::rtfp_solver::geometry::ray_query::{distance_to_mesh, intersect_bvh, point_inside};
use crate::rtfp_solver::geometry::segments::{segments_from_hits, RaySegment};
use crate::rtfp_solver::density::cauchy_ray::{ray_factors, RayFactors};
use crate::rtfp_solver::radial::analytic_alpha1::{i_g, i_h, i_u};
use crate::rtfp_solver::radial::finite_part::i_h_finite_part;
use crate::rtfp_solver::radial::gauss_legendre::{
    quadrature_hessian_finite_part, quadrature_triplet,
};

#[derive(Clone, Debug)]
pub struct CpuRayonBackend {
    pub mesh: TriMesh,
    pub bvh: FlattenedBvh,
    pub density: CauchyDensity,
    pub dirs: Vec<(DVec3, f64)>,
    pub gl_nodes: Vec<f64>,
    pub gl_weights: Vec<f64>,
    pub g: f64,
    pub r0: f64,
    pub ell0: f64,
}

impl CpuRayonBackend {
    pub fn eval_points(
        &self,
        points: &PointSet,
        role: PointRole,
        want: FieldMask,
    ) -> Result<FieldBatch, RtfpError> {
        if role == PointRole::SurfaceVertex && want.hessian {
            return Err(RtfpError::model_not_closed(
                "MS §10.3: a vertex or edge Hessian must not be sampled; use SurfaceFace",
            ));
        }
        if points.x.len() != points.y.len() || points.y.len() != points.z.len() {
            return Err(RtfpError::model_not_closed(
                "PointSet SoA axes have different lengths",
            ));
        }
        let indexed: Vec<usize> = (0..points.len()).collect();
        let rows = map_collect(&indexed, |&i| {
            self.eval_one(points.point(i), role, want)
        });
        let mut batch = FieldBatch {
            potential: Vec::with_capacity(points.len()),
            gradient: Vec::with_capacity(points.len()),
            hessian: Vec::with_capacity(points.len()),
            theta: Vec::with_capacity(points.len()),
        };
        for row in rows {
            let (u, g, h, theta) = row?;
            batch.potential.push(u);
            batch.gradient.push(g);
            batch.hessian.push(h);
            batch.theta.push(theta);
        }
        Ok(batch)
    }

    pub fn eval_one(
        &self,
        x: DVec3,
        role: PointRole,
        want: FieldMask,
    ) -> Result<(f64, DVec3, Sym6, f64), RtfpError> {
        if !x.is_finite() {
            return Err(RtfpError::model_not_closed(
                "field point coordinates must be finite",
            ));
        }
        let inside = point_inside(&self.mesh, &self.bvh, x);
        let theta = match role {
            PointRole::SurfaceFace => 0.5,
            PointRole::SurfaceVertex => {
                if inside {
                    1.0
                } else {
                    0.0
                }
            }
            PointRole::VolumeQuad | PointRole::ClassifyGrid => {
                if inside {
                    1.0
                } else {
                    0.0
                }
            }
        };
        let dist = distance_to_mesh(&self.mesh, x);
        // MS §1.5: r0 must lie inside the tangent-cone ball. Exterior points keep every segment in the far field.
        let r0 = if inside {
            self.r0.min(dist * 0.5).max(1e-8 * self.ell0)
        } else {
            self.r0.min(dist * 0.5)
        };
        let mut potential = 0.0;
        let mut gradient = DVec3::ZERO;
        let mut hessian = [0.0; 6];
        for (u_arr, omega) in &self.dirs {
            let u = *u_arr;
            let hits = intersect_bvh(&self.mesh, &self.bvh, x, -u);
            let ts: Vec<f64> = hits.iter().map(|h| h.t).collect();
            let segs = segments_from_hits(&ts, inside)?;
            if want.potential || want.gradient || want.hessian {
                self.accumulate_direction(
                    x,
                    u,
                    *omega,
                    &segs,
                    want,
                    r0,
                    &mut potential,
                    &mut gradient,
                    &mut hessian,
                )?;
            }
        }
        if want.hessian && theta > 0.0 {
            let rho = self.density.rho(x);
            let contact = 4.0 * std::f64::consts::PI * self.g * theta * rho / 3.0;
            hessian[0] -= contact;
            hessian[1] -= contact;
            hessian[2] -= contact;
        }
        if !want.potential {
            potential = 0.0;
        }
        if !want.gradient {
            gradient = DVec3::ZERO;
        }
        if !want.hessian {
            hessian = [0.0; 6];
        }
        Ok((potential, gradient, hessian, theta))
    }

    fn accumulate_direction(
        &self,
        x: DVec3,
        u: DVec3,
        omega: f64,
        segs: &[RaySegment],
        want: FieldMask,
        r0: f64,
        potential: &mut f64,
        gradient: &mut DVec3,
        hessian: &mut Sym6,
    ) -> Result<(), RtfpError> {
        let t = quadrupole(u);
        for kernel in &self.density.kernels {
            let factor = ray_factors(x, u, kernel.center, kernel.sigma, kernel.weight, kernel.alpha);
            let (iu, ig, ih) = self.integrate_kernel(kernel.alpha, factor, segs, r0);
            let scale = self.g * omega * factor.coeff;
            if want.potential {
                *potential += scale * iu;
            }
            if want.gradient {
                *gradient = *gradient + u * (-scale * ig);
            }
            if want.hessian {
                let w = scale * ih;
                hessian[0] += w * t[0];
                hessian[1] += w * t[1];
                hessian[2] += w * t[2];
                hessian[3] += w * t[3];
                hessian[4] += w * t[4];
                hessian[5] += w * t[5];
            }
        }
        if self.density.bulk > 0.0 {
            let (iu, ig, ih) = integrate_bulk(segs, self.ell0);
            let scale = self.g * omega * self.density.bulk;
            if want.potential {
                *potential += scale * iu;
            }
            if want.gradient {
                *gradient = *gradient + u * (-scale * ig);
            }
            if want.hessian {
                let w = scale * ih;
                hessian[0] += w * t[0];
                hessian[1] += w * t[1];
                hessian[2] += w * t[2];
                hessian[3] += w * t[3];
                hessian[4] += w * t[4];
                hessian[5] += w * t[5];
            }
        }
        let _ = r0;
        Ok(())
    }

    fn integrate_kernel(
        &self,
        alpha: f64,
        factor: RayFactors,
        segs: &[RaySegment],
        _r0: f64,
    ) -> (f64, f64, f64) {
        let mut iu = 0.0;
        let mut ig = 0.0;
        let mut ih = 0.0;
        let alpha_one = (alpha - 1.0).abs() <= 1e-12;
        for seg in segs {
            if alpha_one {
                iu += i_u(factor.b, factor.d, factor.d2, seg.r_min, seg.r_max);
                ig += i_g(factor.b, factor.d, seg.r_min, seg.r_max);
                if seg.r_min == 0.0 {
                    ih += i_h_finite_part(
                        factor.a0,
                        factor.b,
                        factor.d,
                        factor.d2,
                        seg.r_max,
                        self.ell0,
                    );
                } else {
                    ih += i_h(
                        factor.a0,
                        factor.b,
                        factor.d,
                        factor.d2,
                        seg.r_min,
                        seg.r_max,
                    );
                }
            } else if seg.r_min == 0.0 {
                iu += quadrature_triplet(
                    alpha,
                    factor.b,
                    factor.d2,
                    0.0,
                    seg.r_max,
                    &self.gl_nodes,
                    &self.gl_weights,
                )
                .0;
                ig += quadrature_triplet(
                    alpha,
                    factor.b,
                    factor.d2,
                    0.0,
                    seg.r_max,
                    &self.gl_nodes,
                    &self.gl_weights,
                )
                .1;
                ih += quadrature_hessian_finite_part(
                    alpha,
                    factor.b,
                    factor.d2,
                    seg.r_max,
                    self.ell0,
                    &self.gl_nodes,
                    &self.gl_weights,
                );
            } else {
                let (a, b, c) = quadrature_triplet(
                    alpha,
                    factor.b,
                    factor.d2,
                    seg.r_min,
                    seg.r_max,
                    &self.gl_nodes,
                    &self.gl_weights,
                );
                iu += a;
                ig += b;
                ih += c;
            }
        }
        (iu, ig, ih)
    }
}

fn integrate_bulk(segs: &[RaySegment], ell0: f64) -> (f64, f64, f64) {
    let mut iu = 0.0;
    let mut ig = 0.0;
    let mut ih = 0.0;
    for seg in segs {
        iu += 0.5 * (seg.r_max * seg.r_max - seg.r_min * seg.r_min);
        ig += seg.r_max - seg.r_min;
        if seg.r_min == 0.0 {
            ih += (seg.r_max / ell0).ln();
        } else {
            ih += (seg.r_max / seg.r_min).ln();
        }
    }
    (iu, ig, ih)
}
