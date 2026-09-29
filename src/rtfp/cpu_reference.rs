//! CPU f64 严格参考求解器
//! 严格依照 docs/software.md 第 7.5 / 15 节 与 docs/mathstrict.md 第 1 节
//! 一次完成 U, g, H 共享计算，并支持外部经典与近场有限部分分支

use rayon::prelude::*;

use super::analytic::G_SI;
use super::bvh::{intersect_ray_bvh, is_point_inside, FlattenedBvh};
use super::finite_part::assemble_finite_part_hessian_with_background;
use super::mesh::Mesh;
use super::quadrature;
use super::radial_integrals::{
    alpha_one_gradient, alpha_one_hessian, alpha_one_potential, fixed_quadrature_triplet,
};
use super::ray_segments::{segments_from_hits, split_segments_near_far, RaySegment};
use super::tangent_cone::{classify_field_point, compute_tangent_cone};
use crate::model_calculation::contracts::{DensityField, DensityKernel, FieldBranch, FieldSample};

/// CPU 严格求解器配置
#[derive(Clone, Debug)]
pub struct CpuSolverOptions {
    /// 球面求积方向数 (默认 288)
    pub directions: usize,
    /// 近场半径 r0 (m)
    pub near_field_r0: f64,
    /// 有限部分标度 ell0 (m)
    pub ell0: f64,
}

impl Default for CpuSolverOptions {
    fn default() -> Self {
        Self {
            directions: 288,
            near_field_r0: 0.01,
            ell0: 1.0,
        }
    }
}

/// 独立的 CPU f64 严格参考求解器
pub struct CpuReferenceSolver<'a> {
    pub mesh: &'a Mesh,
    pub bvh: &'a FlattenedBvh,
    pub kernels: &'a [DensityKernel],
    pub bulk_density: f64,
    pub dirs: Vec<([f64; 3], f64)>,
    pub options: CpuSolverOptions,
}

impl<'a> CpuReferenceSolver<'a> {
    pub fn new(
        mesh: &'a Mesh,
        bvh: &'a FlattenedBvh,
        kernels: &'a [DensityKernel],
        options: CpuSolverOptions,
    ) -> Self {
        let dirs = quadrature::directions(options.directions);
        Self {
            mesh,
            bvh,
            kernels,
            bulk_density: 0.0,
            dirs,
            options,
        }
    }

    pub fn new_with_density_field(
        mesh: &'a Mesh, bvh: &'a FlattenedBvh, density: &'a DensityField,
        options: CpuSolverOptions,
    ) -> Self {
        let mut solver = Self::new(mesh, bvh, &density.kernels, options);
        solver.bulk_density = density.bulk_density;
        solver
    }

    /// 评估单个场点 x 处的势能 U、加速度 g 与梯度张量 H (一次计算共享几何与射线核)
    pub fn compute_field_at_point(&self, x: [f64; 3]) -> FieldSample {
        let inside = is_point_inside(self.bvh, self.mesh, x);
        let classified = classify_field_point(x, self.mesh, 1e-7);
        let branch = if inside && classified == FieldBranch::ExternalClassical {
            FieldBranch::InteriorFinitePart
        } else { classified };

        // 1. 遍历全部射线方向求交并收集交段
        let num_dirs = self.dirs.len();
        let mut segments_per_dir: Vec<Vec<RaySegment>> = Vec::with_capacity(num_dirs);

        for (u, _omega) in &self.dirs {
            let ray_dir = [-u[0], -u[1], -u[2]];
            let hits = intersect_ray_bvh(self.bvh, self.mesh, x, ray_dir);
            let segs = segments_from_hits(&hits, inside);
            segments_per_dir.push(segs);
        }

        match branch {
            FieldBranch::ExternalClassical => {
                self.compute_external_classical(x, &segments_per_dir)
            }
            _ => {
                self.compute_interior_or_boundary_finite_part(x, &segments_per_dir, branch)
            }
        }
    }

    /// 外部真空点经典 Newton/Feynman 求解分支 (Tr(H) = 0)
    fn compute_external_classical(
        &self,
        x: [f64; 3],
        segments_per_dir: &[Vec<RaySegment>],
    ) -> FieldSample {
        let mut u_total = 0.0f64;
        let mut g_total = [0.0f64; 3];
        let mut h_total = [0.0f64; 6];

        for (dir_idx, (u, omega)) in self.dirs.iter().enumerate() {
            let segs = &segments_per_dir[dir_idx];
            if segs.is_empty() {
                continue;
            }

            let mut dir_iu = 0.0f64;
            let mut dir_ig = 0.0f64;
            let mut dir_ih = 0.0f64;

            for k in self.kernels {
                let px = x[0] - k.center[0];
                let py = x[1] - k.center[1];
                let pz = x[2] - k.center[2];
                let p2 = px * px + py * py + pz * pz;
                let b = px * u[0] + py * u[1] + pz * u[2];
                let inv_s2 = 1.0 / (k.sigma * k.sigma);
                let d2 = (inv_s2 + p2 - b * b).max(1e-30);
                let d = d2.sqrt();
                let a0 = b * b + d2;

                let s_factor = k.sigma.powf(-2.0 * k.alpha);
                let weight_factor = k.weight * s_factor;

                let mut k_iu = 0.0f64;
                let mut k_ig = 0.0f64;
                let mut k_ih = 0.0f64;

                for seg in segs {
                    if (k.alpha - 1.0).abs() < 1e-9 {
                        k_iu += alpha_one_potential(b, d, seg.r_min, seg.r_max);
                        k_ig += alpha_one_gradient(b, d, seg.r_min, seg.r_max);
                        k_ih += alpha_one_hessian(a0, b, d, seg.r_min, seg.r_max);
                    } else {
                        let (iu, ig, ih) = fixed_quadrature_triplet(k.alpha, b, d, seg.r_min, seg.r_max);
                        k_iu += iu;
                        k_ig += ig;
                        k_ih += ih;
                    }
                }

                dir_iu += weight_factor * k_iu;
                dir_ig += weight_factor * k_ig;
                dir_ih += weight_factor * k_ih;
            }

            if self.bulk_density > 0.0 {
                for seg in segs {
                    let h = 0.5 * (seg.r_max - seg.r_min);
                    let mid = 0.5 * (seg.r_max + seg.r_min);
                    let mut iu = 0.0;
                    let mut ig = 0.0;
                    let mut ih = 0.0;
                    for j in 0..16 {
                        let z = super::radial_integrals::GAUSS_LEGENDRE_16_NODES[j];
                        let w = super::radial_integrals::GAUSS_LEGENDRE_16_WEIGHTS[j];
                        let r = mid + h*z;
                        if r > 0.0 { iu += w*r; ig += w; ih += w/r; }
                    }
                    dir_iu += self.bulk_density*iu*h;
                    dir_ig += self.bulk_density*ig*h;
                    dir_ih += self.bulk_density*ih*h;
                }
            }

            // 累加角向积分:
            // U = G sum omega I_U
            // g = -G sum omega u I_g
            // H = G sum omega T(u) I_H
            u_total += G_SI * omega * dir_iu;

            let g_weight = -G_SI * omega * dir_ig;
            g_total[0] += g_weight * u[0];
            g_total[1] += g_weight * u[1];
            g_total[2] += g_weight * u[2];

            let t_xx = 3.0 * u[0] * u[0] - 1.0;
            let t_yy = 3.0 * u[1] * u[1] - 1.0;
            let t_zz = 3.0 * u[2] * u[2] - 1.0;
            let t_xy = 3.0 * u[0] * u[1];
            let t_xz = 3.0 * u[0] * u[2];
            let t_yz = 3.0 * u[1] * u[2];

            let h_weight = G_SI * omega * dir_ih;
            h_total[0] += h_weight * t_xx;
            h_total[1] += h_weight * t_yy;
            h_total[2] += h_weight * t_zz;
            h_total[3] += h_weight * t_xy;
            h_total[4] += h_weight * t_xz;
            h_total[5] += h_weight * t_yz;
        }

        let trace = h_total[0] + h_total[1] + h_total[2];

        FieldSample {
            potential: u_total,
            gravity: g_total,
            hessian6: h_total,
            trace,
            branch: FieldBranch::ExternalClassical,
        }
    }

    /// 内部点与边界点球形有限部分求解分支 (近场/远场互补积分)
    fn compute_interior_or_boundary_finite_part(
        &self,
        x: [f64; 3],
        segments_per_dir: &[Vec<RaySegment>],
        branch: FieldBranch,
    ) -> FieldSample {
        let r0 = self.options.near_field_r0;
        let ell0 = self.options.ell0;

        // 1. 计算局部切锥与锥张量
        let cone = compute_tangent_cone(x, self.bvh, self.mesh, &self.dirs, 1e-4);

        // 2. 将射线交段分割为近场和远场
        let mut far_segments_per_dir: Vec<Vec<RaySegment>> = Vec::with_capacity(self.dirs.len());
        for segs in segments_per_dir {
            let (_near, far) = split_segments_near_far(segs, r0);
            far_segments_per_dir.push(far);
        }

        // 3. 组装球形有限部分 Hessian
        let h_fp = assemble_finite_part_hessian_with_background(
            x,
            &cone,
            &far_segments_per_dir,
            &self.dirs,
            self.kernels,
            r0,
            ell0,
            self.bulk_density,
        );

        // 4. 势能 U 与加速度 g 在包含原点的区间上求解
        let mut u_total = 0.0f64;
        let mut g_total = [0.0f64; 3];

        for (dir_idx, (u, omega)) in self.dirs.iter().enumerate() {
            let segs = &segments_per_dir[dir_idx];
            for k in self.kernels {
                let px = x[0] - k.center[0];
                let py = x[1] - k.center[1];
                let pz = x[2] - k.center[2];
                let p2 = px * px + py * py + pz * pz;
                let b = px * u[0] + py * u[1] + pz * u[2];
                let inv_s2 = 1.0 / (k.sigma * k.sigma);
                let d2 = (inv_s2 + p2 - b * b).max(1e-30);
                let d = d2.sqrt();

                let s_factor = k.sigma.powf(-2.0 * k.alpha);
                let weight_factor = k.weight * s_factor;

                for seg in segs {
                    let (iu, ig) = if (k.alpha - 1.0).abs() < 1e-9 {
                        (
                            alpha_one_potential(b, d, seg.r_min, seg.r_max),
                            alpha_one_gradient(b, d, seg.r_min, seg.r_max),
                        )
                    } else {
                        let (iu_q, ig_q, _ih_q) = fixed_quadrature_triplet(k.alpha, b, d, seg.r_min, seg.r_max);
                        (iu_q, ig_q)
                    };

                    u_total += G_SI * omega * weight_factor * iu;
                    let gw = -G_SI * omega * weight_factor * ig;
                    g_total[0] += gw * u[0];
                    g_total[1] += gw * u[1];
                    g_total[2] += gw * u[2];
                }
            }
            if self.bulk_density > 0.0 {
                for seg in segs {
                    let (iu, ig) = {
                        let h=0.5*(seg.r_max-seg.r_min); let mid=0.5*(seg.r_max+seg.r_min);
                        let mut a=0.0; let mut b=0.0;
                        for j in 0..16 { let z=super::radial_integrals::GAUSS_LEGENDRE_16_NODES[j]; let w=super::radial_integrals::GAUSS_LEGENDRE_16_WEIGHTS[j]; let r=mid+h*z; if r>0.0 {a+=w*r; b+=w;} }
                        (a*h,b*h)
                    };
                    u_total += G_SI*omega*self.bulk_density*iu;
                    let gw=-G_SI*omega*self.bulk_density*ig;
                    g_total[0]+=gw*u[0]; g_total[1]+=gw*u[1]; g_total[2]+=gw*u[2];
                }
            }
        }

        let trace = h_fp[0] + h_fp[1] + h_fp[2];

        FieldSample {
            potential: u_total,
            gravity: g_total,
            hessian6: h_fp,
            trace,
            branch,
        }
    }

    /// 批量并行计算场点
    pub fn compute_field_batch(&self, points: &[[f64; 3]]) -> Vec<FieldSample> {
        points
            .into_par_iter()
            .map(|&p| self.compute_field_at_point(p))
            .collect()
    }
}
