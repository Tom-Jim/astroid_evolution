//! RTFP (Ray-Traced Finite-Part) 独立重力计算模块
//! 依照 docs/mathstrict.md 与 ryugu_cauchy_gravity/src/rust/bin/rtfp_bake 实现
//! 输入模型 GLB 或点云表面，输出 1mm 高度的引力梯度张量 H 与引力加速度 g

pub mod analytic;
pub mod bvh;
pub mod cpu_reference;
pub mod density;
pub mod density_derivatives;
pub mod feynman;
pub mod finite_part;
pub mod geom;
pub mod geometry_validation;
pub mod gpu_dispatch;
pub mod mesh;
pub mod quadrature;
pub mod radial_integrals;
pub mod ray_backend;
pub mod ray_segments;
pub mod split;
pub mod tangent_cone;
pub mod tensor;
pub mod field_backend;
pub mod residual_channels;

use std::path::Path;
use rayon::prelude::*;

use analytic::{Face, G_SI};
use bvh::Bvh;
use density::Density;
use mesh::Mesh;
use tensor::Sym6;

const GEOM_EPS_M: f64 = 1e-9;

/// 1mm 高度观测面 RTFP 计算结果集
#[derive(Clone, Debug)]
pub struct RtfpResult {
    /// 1mm 高度观测点坐标 (米)
    pub points: Vec<[f64; 3]>,
    /// 对应顶点外法向量
    pub normals: Vec<[f64; 3]>,
    pub potentials: Vec<f64>,
    /// 1mm 高度处的引力梯度张量 H (6 独立分量 Sym6)
    pub hessians: Vec<Sym6>,
    /// 由 H 计算恢复的引力加速度矢量 g = [g_x, g_y, g_z] (m/s²)
    pub gravities: Vec<[f64; 3]>,
    /// 张量 Frobenius 范数 ||H||_F (s⁻²)
    pub h_frobenius: Vec<f64>,
    /// 引力加速度模长 ||g|| (m/s²)
    pub g_magnitudes: Vec<f64>,
    /// 张量迹 Tr(H) (自由空间无源校验 Tr(H) ≈ 0)
    pub traces: Vec<f64>,
    /// 观测面高度 (mm)
    pub standoff_mm: f64,
}

/// 独立的 RTFP 计算器模块
pub struct RtfpModule {
    pub mesh: Mesh,
    pub density: Density,
    pub bvh: Bvh,
    pub faces: Vec<Face>,
    pub dirs: Vec<([f64; 3], f64)>,
    pub volume: f64,
    pub centroid: [f64; 3],
    pub body_radius: f64,
}

impl RtfpModule {
    /// 从 GLB 模型文件加载并构建 RTFP 求解器
    pub fn from_glb(path: &Path, scale: f64, density: Density, directions: usize) -> Result<Self, String> {
        let mesh = Mesh::load_glb(path, scale)?;
        Self::new(mesh, density, directions)
    }

    /// 从表面点云构建并初始化 RTFP 求解器
    pub fn from_point_cloud(points: &[[f64; 3]], density: Density, directions: usize) -> Result<Self, String> {
        let mesh = Mesh::from_point_cloud(points, 48, 96)?;
        Self::new(mesh, density, directions)
    }

    /// 核心初始化：预计算面积分、构建扁平 BVH 加速结构与高精度球面求积方向
    pub fn new(mesh: Mesh, density: Density, directions: usize) -> Result<Self, String> {
        let faces = analytic::precompute(&mesh);
        let bvh = Bvh::build(&mesh);
        let dirs = quadrature::directions(directions);
        let (volume, centroid) = mesh.volume_and_centroid();

        let mut max_r = 0.0f64;
        for i in 0..mesh.vertex_count() {
            let p = mesh.vertex(i);
            let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
            if r > max_r {
                max_r = r;
            }
        }

        Ok(Self {
            mesh,
            density,
            bvh,
            faces,
            dirs,
            volume,
            centroid,
            body_radius: max_r,
        })
    }

    /// 计算单个场点 x 处的严格 RTFP 引力梯度张量 H(x)
    /// H(x) = ρ(x) · W(x) + G Σ_q ω_q T(u_q) Σ_k w_k R_k(u_q)
    pub fn evaluate_at_point(&self, x: [f64; 3]) -> Result<Sym6, String> {
        // 1. Werner 均匀多面体闭合解析表面积分 W(x)
        let mut hessian = analytic::hessian6(&self.faces, x);

        // 2. 乘以观测点局域 Cauchy 质量密度 ρ(x)
        let rho = self.density.rho(&x);
        for value in &mut hessian {
            *value *= rho;
        }

        // 3. 射线求交与平滑余项积分
        let t_max = self.body_radius * 4.0;
        let mut hits = Vec::new();

        // 判定内/外状态
        geom::bvh_crossings(
            &self.bvh,
            &self.mesh,
            x,
            [-1.0, 0.0, 0.0],
            GEOM_EPS_M,
            t_max,
            &mut hits,
        );
        let inside = hits.len() % 2 == 1;

        // 4. 球面积分 Σ_q ω_q T(u_q) Σ_k w_k R_k(u_q)
        let kernels = self.density.kernels();
        for (direction, weight) in &self.dirs {
            geom::bvh_crossings(
                &self.bvh,
                &self.mesh,
                x,
                [-direction[0], -direction[1], -direction[2]],
                GEOM_EPS_M,
                t_max,
                &mut hits,
            );

            let (intervals, overflow) = split::intervals(&hits, inside);
            if overflow {
                return Err(format!("RT-FP ray at {x:?} exceeded maximum visible intervals"));
            }

            let scalar = split::remainder_scalar(kernels, x, *direction, &intervals);
            tensor::add_tensor_term(&mut hessian, direction, G_SI * *weight * scalar);
        }

        Ok(hessian)
    }

    /// 计算模型表面 1mm (0.001 米) 高度观测面上的全场 H，并由 H 求解 g
    pub fn compute_surface_1mm(&self) -> Result<RtfpResult, String> {
        let standoff_mm = 1.0;
        let standoff_m = standoff_mm * 1e-3;

        let points = self.mesh.observation_points(standoff_m);
        let normals = self.mesh.vertex_normals();

        // 并行计算所有观测点处的 H 与 g
        let results: Vec<(f64, Sym6, [f64; 3], f64, f64, f64)> = points
            .par_iter()
            .map(|&p| -> Result<(f64, Sym6, [f64; 3], f64, f64, f64), String> {
                let h = self.evaluate_at_point(p)?;
                // g 必须由势的一阶导数直接得到，不能由 Hessian 反推。
                let g = self.gradient_at_point(p)?;
                let h_frob = tensor::frobenius(&h);
                let g_mag = (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt();
                let tr = tensor::trace(&h);
                let potential = self.potential_at_point(p)?;
                Ok((potential, h, g, h_frob, g_mag, tr))
            })
            .collect::<Result<Vec<_>, _>>()?;

        let mut hessians = Vec::with_capacity(results.len());
        let mut potentials = Vec::with_capacity(results.len());
        let mut gravities = Vec::with_capacity(results.len());
        let mut h_frobenius = Vec::with_capacity(results.len());
        let mut g_magnitudes = Vec::with_capacity(results.len());
        let mut traces = Vec::with_capacity(results.len());

        for (potential, h, g, hf, gm, tr) in results {
            potentials.push(potential);
            hessians.push(h);
            gravities.push(g);
            h_frobenius.push(hf);
            g_magnitudes.push(gm);
            traces.push(tr);
        }

        Ok(RtfpResult {
            points,
            normals,
            potentials,
            hessians,
            gravities,
            h_frobenius,
            g_magnitudes,
            traces,
            standoff_mm,
        })
    }

    fn potential_at_point(&self, x: [f64; 3]) -> Result<f64, String> {
        let mut total = 0.0;
        for (u, omega) in &self.dirs {
            let mut hits = Vec::new();
            geom::bvh_crossings(&self.bvh, &self.mesh, x, [-u[0],-u[1],-u[2]], GEOM_EPS_M, self.body_radius*4.0, &mut hits);
            let inside = hits.len() % 2 == 1;
            let (ivals, overflow) = split::intervals(&hits, inside);
            if overflow { return Err("势积分射线交段溢出".into()); }
            if self.density.bulk_density > 0.0 {
                let mut iu_total = 0.0;
                for slot in ivals.iter().flatten() {
                    let (r0,r1)=*slot;
                    let h=0.5*(r1-r0); let mid=0.5*(r1+r0);
                    for j in 0..16 { let z=radial_integrals::GAUSS_LEGENDRE_16_NODES[j]; let w=radial_integrals::GAUSS_LEGENDRE_16_WEIGHTS[j]; let r=mid+h*z; iu_total += w*r*h; }
                }
                total += analytic::G_SI * *omega * self.density.bulk_density * iu_total;
            }
            for slot in ivals.iter().flatten() {
                let (r0, r1) = *slot;
                for k in self.density.kernels() {
                    let p = [x[0]-k.c[0], x[1]-k.c[1], x[2]-k.c[2]];
                    let p2 = p[0]*p[0]+p[1]*p[1]+p[2]*p[2];
                    let b = p[0]*u[0]+p[1]*u[1]+p[2]*u[2];
                    let d = (1.0/(k.sigma*k.sigma)+p2-b*b).max(1e-30).sqrt();
                    let iu = if (k.alpha-1.0).abs() < 1e-9 {
                        radial_integrals::alpha_one_potential(b,d,r0,r1)
                    } else {
                        radial_integrals::fixed_quadrature_triplet(k.alpha,b,d,r0,r1).0
                    };
                    total += analytic::G_SI * *omega * k.w * k.sigma.powf(-2.0*k.alpha) * iu;
                }
            }
        }
        Ok(total)
    }

    fn gradient_at_point(&self, x: [f64; 3]) -> Result<[f64; 3], String> {
        let h = (self.body_radius * 1e-6).max(1e-5);
        let mut g = [0.0; 3];
        for axis in 0..3 {
            let mut xp=x; let mut xm=x; xp[axis]+=h; xm[axis]-=h;
            g[axis]=(self.potential_at_point(xp)?-self.potential_at_point(xm)?)/(2.0*h);
        }
        Ok(g)
    }
}
