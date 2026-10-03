//! Surface samples on one asteroid entity. Every per-node array is SoA inside this component.
use crate::scf_simulator::surface::metric::{SurfaceMetric, compute_surface_metric};
#[derive(Clone, Debug)]
pub struct SurfaceState {
    pub theta: Vec<f64>,
    pub phi: Vec<f64>,
    pub radii: Vec<f64>,
    pub directions: Vec<[f64; 3]>,
    pub grad_r: Vec<[f64; 2]>,
    pub metrics: Vec<SurfaceMetric>,
    pub normals: Vec<[f64; 3]>,
}
impl SurfaceState {
    pub fn from_spherical_samples(
        theta: Vec<f64>,
        phi: Vec<f64>,
        radii: Vec<f64>,
        directions: Vec<[f64; 3]>,
        grad_r: Vec<[f64; 2]>,
    ) -> Result<Self, String> {
        if theta.len() != radii.len() || phi.len() != radii.len() {
            return Err("球面坐标数量不一致".into());
        }
        if radii.len() != directions.len() || radii.len() != grad_r.len() {
            return Err("表面节点数量不一致".into());
        }
        let metrics = theta
            .iter()
            .zip(&phi)
            .zip(radii.iter().zip(&grad_r))
            .map(|((t, p), (r, g))| compute_surface_metric(*t, *p, *r, g[0], g[1]))
            .collect::<Vec<_>>();
        let normals = metrics.iter().map(|m| m.normal).collect();
        Ok(Self {
            theta,
            phi,
            radii,
            directions,
            grad_r,
            metrics,
            normals,
        })
    }
    pub fn from_radial_samples(
        radii: Vec<f64>,
        directions: Vec<[f64; 3]>,
        grad_r: Vec<[f64; 2]>,
    ) -> Result<Self, String> {
        if radii.len() != directions.len() || radii.len() != grad_r.len() {
            return Err("表面半径、方向和梯度数量不一致".into());
        }
        let angles = directions
            .iter()
            .map(|u| {
                let l = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt().max(1e-30);
                ((u[2] / l).clamp(-1.0, 1.0).acos(), u[1].atan2(u[0]))
            })
            .collect::<Vec<_>>();
        let metrics: Vec<_> = radii
            .iter()
            .zip(&grad_r)
            .zip(&angles)
            .map(|((r, g), (t, p))| compute_surface_metric(*t, *p, *r, g[0], g[1]))
            .collect();
        let normals = metrics.iter().map(|m| m.normal).collect();
        Ok(Self {
            theta: angles.iter().map(|a| a.0).collect(),
            phi: angles.iter().map(|a| a.1).collect(),
            radii,
            directions,
            grad_r,
            metrics,
            normals,
        })
    }
}
