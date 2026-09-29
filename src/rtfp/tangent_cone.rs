//! 局部切锥与锥张量计算
//! 严格依照 docs/software.md 第 7.4 节 与 docs/mathstrict.md 第 1.3 节
//!
//! Gamma_x: 方向集合 u 满足 x - R u 属于 V (对充分小的 R > 0)
//! T_ij(u) = 3 u_i u_j - delta_ij
//! C_ij(x) = int_{Gamma_x} T_ij(u) dOmega
//! vartheta_x = (1 / 4 pi) int_{Gamma_x} 1 dOmega

use super::bvh::{is_point_inside, FlattenedBvh};
use super::mesh::Mesh;

/// 根据点到网格顶点/棱/面的距离返回有限部分分支。
/// 该分类只在已通过闭合网格验证后使用。
pub fn classify_field_point(x: [f64; 3], mesh: &Mesh, tolerance: f64) -> crate::model_calculation::contracts::FieldBranch {
    use crate::model_calculation::contracts::FieldBranch;
    if is_point_inside_mesh(x, mesh, tolerance) { return FieldBranch::InteriorFinitePart; }
    let mut vertex_hit = false;
    for i in 0..mesh.vertex_count() {
        let p = mesh.vertex(i);
        let d = ((x[0]-p[0]).powi(2)+(x[1]-p[1]).powi(2)+(x[2]-p[2]).powi(2)).sqrt();
        if d <= tolerance { vertex_hit = true; break; }
    }
    if vertex_hit { return FieldBranch::VertexFinitePart; }
    // 面/棱的精确最近点查询由几何后端负责；这里保守地将近表面点标记为边界。
    let mut near = false;
    for f in 0..mesh.face_count() {
        let [a,b,c] = mesh.face(f);
        let pa=mesh.vertex(a as usize); let pb=mesh.vertex(b as usize); let pc=mesh.vertex(c as usize);
        let n = super::mesh::cross(super::mesh::mesh_sub(pb, pa), super::mesh::mesh_sub(pc, pa));
        let nl = super::mesh::norm(n).max(1e-30);
        let d = ((x[0]-pa[0])*n[0]+(x[1]-pa[1])*n[1]+(x[2]-pa[2])*n[2]).abs()/nl;
        if d <= tolerance { near = true; break; }
    }
    if near { FieldBranch::BoundaryFinitePart } else { FieldBranch::ExternalClassical }
}

fn is_point_inside_mesh(x: [f64; 3], mesh: &Mesh, _tolerance: f64) -> bool {
    let mut count = 0usize;
    for f in 0..mesh.face_count() {
        let [i0,i1,i2] = mesh.face(f);
        let a=mesh.vertex(i0 as usize); let b=mesh.vertex(i1 as usize); let c=mesh.vertex(i2 as usize);
        let e1=super::mesh::mesh_sub(b,a); let e2=super::mesh::mesh_sub(c,a);
        let dir=[1.0,0.1234567,0.0712345];
        let p=super::mesh::cross(dir,e2); let det=super::mesh::dot(e1,p);
        if det.abs()<1e-14 { continue; }
        let inv=1.0/det; let t=super::mesh::mesh_sub(x,a); let u=super::mesh::dot(t,p)*inv;
        if u<0.0 || u>1.0 { continue; }
        let q=super::mesh::cross(t,e1); let v=super::mesh::dot(dir,q)*inv;
        if v>=0.0 && u+v<=1.0 && super::mesh::dot(e2,q)*inv>0.0 { count+=1; }
    }
    count%2==1
}

/// 局部切锥几何描述
#[derive(Clone, Debug)]
pub struct TangentCone {
    /// 立体角分数 vartheta_x in [0, 1] (内部点为 1, 外部点为 0, 平滑表面点为 0.5)
    pub solid_angle_fraction: f64,
    /// 切锥张量 C_ij(x) (6 独立分量: xx, yy, zz, xy, xz, yz)
    pub cone_tensor: [f64; 6],
    /// 射线属于 Gamma_x 的指示标志掩码 [dir_idx] -> bool
    pub in_cone_mask: Vec<bool>,
}

/// 计算场点 x 处的局部切锥 Gamma_x、立体角分数 vartheta_x 与锥张量 C_ij(x)
pub fn compute_tangent_cone(
    x: [f64; 3],
    bvh: &FlattenedBvh,
    mesh: &Mesh,
    dirs: &[([f64; 3], f64)],
    eps_probe: f64,
) -> TangentCone {
    let mut solid_angle_sum = 0.0f64;
    let mut c = [0.0f64; 6];
    let mut in_cone_mask = Vec::with_capacity(dirs.len());

    for (u, omega) in dirs {
        // 测试 x - eps * u 是否位于实体 V 内部
        let probe = [
            x[0] - eps_probe * u[0],
            x[1] - eps_probe * u[1],
            x[2] - eps_probe * u[2],
        ];

        let inside = is_point_inside(bvh, mesh, probe);
        in_cone_mask.push(inside);

        if inside {
            solid_angle_sum += omega;
            // T_ij(u) = 3 u_i u_j - delta_ij
            let t_xx = 3.0 * u[0] * u[0] - 1.0;
            let t_yy = 3.0 * u[1] * u[1] - 1.0;
            let t_zz = 3.0 * u[2] * u[2] - 1.0;
            let t_xy = 3.0 * u[0] * u[1];
            let t_xz = 3.0 * u[0] * u[2];
            let t_yz = 3.0 * u[1] * u[2];

            c[0] += omega * t_xx;
            c[1] += omega * t_yy;
            c[2] += omega * t_zz;
            c[3] += omega * t_xy;
            c[4] += omega * t_xz;
            c[5] += omega * t_yz;
        }
    }

    use std::f64::consts::PI;
    let vartheta = (solid_angle_sum / (4.0 * PI)).clamp(0.0, 1.0);

    TangentCone {
        solid_angle_fraction: vartheta,
        cone_tensor: c,
        in_cone_mask,
    }
}

/// 快速推断场点的立体角分数 vartheta 与锥张量 C_ij
pub fn estimate_ideal_tangent_cone(branch: crate::model_calculation::contracts::FieldBranch) -> (f64, [f64; 6]) {
    use crate::model_calculation::contracts::FieldBranch;
    match branch {
        FieldBranch::ExternalClassical => (0.0, [0.0; 6]),
        FieldBranch::InteriorFinitePart => (1.0, [0.0; 6]),
        FieldBranch::BoundaryFinitePart => (0.5, [0.0; 6]),
        FieldBranch::EdgeFinitePart => (0.25, [0.0; 6]),
        FieldBranch::VertexFinitePart => (0.125, [0.0; 6]),
    }
}
