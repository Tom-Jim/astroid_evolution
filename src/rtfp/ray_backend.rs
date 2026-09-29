//! 射线求交后端抽象 Trait 与 CPU 严格参考实现
//! 严格依照 docs/software.md 第 2 节 定义

use super::bvh::{intersect_ray_bvh, FlattenedBvh};
use super::mesh::Mesh;
use rayon::prelude::*;

/// 射线批次定义
#[derive(Clone, Debug)]
pub struct RayBatch {
    /// 射线原点集合
    pub origins: Vec<[f64; 3]>,
    /// 射线方向集合 (必须已单位归一化)
    pub directions: Vec<[f64; 3]>,
}

impl RayBatch {
    pub fn new(origins: Vec<[f64; 3]>, directions: Vec<[f64; 3]>) -> Result<Self, String> {
        if origins.len() != directions.len() {
            return Err(format!(
                "射线原点数量 ({}) 与方向数量 ({}) 不一致",
                origins.len(),
                directions.len()
            ));
        }
        if directions.iter().any(|d| {
            let n = (d[0]*d[0] + d[1]*d[1] + d[2]*d[2]).sqrt();
            !n.is_finite() || (n - 1.0).abs() > 1e-8
        }) {
            return Err("射线方向必须是有限单位向量".into());
        }
        Ok(Self { origins, directions })
    }

    pub fn len(&self) -> usize {
        self.origins.len()
    }

    pub fn is_empty(&self) -> bool {
        self.origins.is_empty()
    }
}

/// 射线碰撞距离缓冲区
#[derive(Clone, Debug, Default)]
pub struct RayHitBuffer {
    /// 每条射线的全部排序碰撞距离: [ray_id] -> [R_1, R_2, ..., R_L]
    pub hits_per_ray: Vec<Vec<f64>>,
}

/// 统一的射线计算后端 Trait (docs/software.md 第 2 节)
pub trait RayBackend: Send + Sync {
    fn intersect_batch(&self, rays: &RayBatch) -> RayHitBuffer;
}

/// CPU f64 严格校验后端 (基于 Flattened SoA BVH 与 Rayon 多线程求交)
pub struct CpuReferenceBackend<'a> {
    pub bvh: &'a FlattenedBvh,
    pub mesh: &'a Mesh,
}

impl<'a> CpuReferenceBackend<'a> {
    pub fn new(bvh: &'a FlattenedBvh, mesh: &'a Mesh) -> Self {
        Self { bvh, mesh }
    }
}

impl RayBackend for CpuReferenceBackend<'_> {
    fn intersect_batch(&self, rays: &RayBatch) -> RayHitBuffer {
        let n = rays.len();
        let hits_per_ray: Vec<Vec<f64>> = (0..n)
            .into_par_iter()
            .map(|i| {
                let orig = rays.origins[i];
                let dir = rays.directions[i];
                intersect_ray_bvh(self.bvh, self.mesh, orig, dir)
            })
            .collect();

        RayHitBuffer { hits_per_ray }
    }
}
