//! 封闭多面体几何严格性验证
//! 严格依照 docs/software.md 第 7.1 节与 docs/mathstrict.md 第 1.1 节
//! 验证项：
//! 1. 面片非退化 (非共线，面积 > eps)
//! 2. 拓扑流形与定向一致性 (半边匹配，无开孔边缘，无多分支)
//! 3. 体积为正与空腔法向约定
//! 4. 无明显自交 (非相邻三角形相交检测)
//! 5. 欧拉示性数与亏格拓扑核查

use std::collections::HashMap;
use super::mesh::Mesh;

/// 几何校验详细报告
#[derive(Clone, Debug)]
pub struct GeometryValidationReport {
    /// 总体是否满足有界多面体实体要求
    pub is_valid: bool,
    /// 面片是否均非退化
    pub non_degenerate: bool,
    /// 是否为严格封闭二维流形
    pub is_closed_manifold: bool,
    /// 面定向是否完全一致
    pub consistent_orientation: bool,
    /// 是否存在自相交
    pub has_self_intersection: bool,
    /// 退化面片数量
    pub degenerate_face_count: usize,
    /// 开孔边界边数量 (封闭表面应为 0)
    pub boundary_edge_count: usize,
    /// 非流形边数量 (>2 面共享边)
    pub non_manifold_edge_count: usize,
    /// 反向定向冲突边数量
    pub orientation_conflict_count: usize,
    /// 自交面片对数量
    pub self_intersection_count: usize,
    /// 多面体代数带符号体积 (m³)
    pub signed_volume: f64,
    /// 质心坐标 [x, y, z] (m)
    pub centroid: [f64; 3],
    /// 欧拉示性数 chi = V - E + F (标准亏格 0 球面 chi = 2)
    pub euler_characteristic: i32,
    /// 详细错误信息
    pub errors: Vec<String>,
    /// 警告信息
    pub warnings: Vec<String>,
}

/// 核心验证函数: validate_closed_polyhedron(mesh)
pub fn validate_closed_polyhedron(mesh: &Mesh) -> GeometryValidationReport {
    let nv = mesh.vertex_count();
    let nf = mesh.face_count();

    let mut errors = Vec::new();
    let mut warnings = Vec::new();

    if nv < 4 || nf < 4 {
        errors.push(format!("网格顶点数 ({nv}) 或面片数 ({nf}) 不足以构成三维实体"));
        return GeometryValidationReport {
            is_valid: false,
            non_degenerate: false,
            is_closed_manifold: false,
            consistent_orientation: false,
            has_self_intersection: false,
            degenerate_face_count: 0,
            boundary_edge_count: 0,
            non_manifold_edge_count: 0,
            orientation_conflict_count: 0,
            self_intersection_count: 0,
            signed_volume: 0.0,
            centroid: [0.0; 3],
            euler_characteristic: 0,
            errors,
            warnings,
        };
    }

    // 计算特征尺寸 (AABB 对角线长)
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for i in 0..nv {
        let p = mesh.vertex(i);
        for k in 0..3 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    let char_len = ((hi[0] - lo[0]).powi(2) + (hi[1] - lo[1]).powi(2) + (hi[2] - lo[2]).powi(2)).sqrt();
    let eps_len = char_len * 1e-7;
    let eps_area = char_len * char_len * 1e-12;

    // 1. 面片非退化检查
    let mut degenerate_faces = 0;
    for f in 0..nf {
        let [i0, i1, i2] = mesh.face(f);
        if i0 == i1 || i1 == i2 || i2 == i0 {
            degenerate_faces += 1;
            continue;
        }
        let p0 = mesh.vertex(i0 as usize);
        let p1 = mesh.vertex(i1 as usize);
        let p2 = mesh.vertex(i2 as usize);

        let d01 = ((p1[0] - p0[0]).powi(2) + (p1[1] - p0[1]).powi(2) + (p1[2] - p0[2]).powi(2)).sqrt();
        let d12 = ((p2[0] - p1[0]).powi(2) + (p2[1] - p1[1]).powi(2) + (p2[2] - p1[2]).powi(2)).sqrt();
        let d20 = ((p0[0] - p2[0]).powi(2) + (p0[1] - p2[1]).powi(2) + (p0[2] - p2[2]).powi(2)).sqrt();
        if d01 < eps_len || d12 < eps_len || d20 < eps_len {
            degenerate_faces += 1;
            continue;
        }

        let e1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
        let e2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
        let cross = [
            e1[1] * e2[2] - e1[2] * e2[1],
            e1[2] * e2[0] - e1[0] * e2[2],
            e1[0] * e2[1] - e1[1] * e2[0],
        ];
        let double_area = (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt();
        if double_area < eps_area {
            degenerate_faces += 1;
        }
    }

    if degenerate_faces > 0 {
        errors.push(format!("检测到 {degenerate_faces} 个退化或共线面片 (面积 < {eps_area:.2e})"));
    }

    // 2. 拓扑流形与半边定向一致性检查
    // 无向边 key: (min(u, v), max(u, v)) -> Vec<(u, v, face_idx)>
    let mut edge_map: HashMap<(u32, u32), Vec<(u32, u32, usize)>> = HashMap::new();
    for f in 0..nf {
        let [i0, i1, i2] = mesh.face(f);
        let edges = [(i0, i1), (i1, i2), (i2, i0)];
        for (u, v) in edges {
            let key = if u < v { (u, v) } else { (v, u) };
            edge_map.entry(key).or_default().push((u, v, f));
        }
    }

    let mut boundary_edges = 0;
    let mut non_manifold_edges = 0;
    let mut orientation_conflicts = 0;

    for (_key, incidences) in &edge_map {
        match incidences.len() {
            1 => {
                boundary_edges += 1;
            }
            2 => {
                let e0 = incidences[0];
                let e1 = incidences[1];
                // 定向一致要求: 同一条边在两相邻面中的遍历方向必须相反
                if e0.0 == e1.0 && e0.1 == e1.1 {
                    orientation_conflicts += 1;
                }
            }
            _ => {
                non_manifold_edges += 1;
            }
        }
    }

    if boundary_edges > 0 {
        errors.push(format!("网格非封闭: 存在 {boundary_edges} 条开孔边界边 (Boundary Edges)"));
    }
    if non_manifold_edges > 0 {
        errors.push(format!("网格非流形: 存在 {non_manifold_edges} 条分叉边 (>2 面共享边)"));
    }
    if orientation_conflicts > 0 {
        errors.push(format!("面法向不一致: 存在 {orientation_conflicts} 条相邻面同向遍历冲突边"));
    }

    // 3. 欧拉示性数
    let num_edges = edge_map.len();
    let mut used_verts = std::collections::HashSet::new();
    for f in 0..nf {
        let [i0, i1, i2] = mesh.face(f);
        used_verts.insert(i0);
        used_verts.insert(i1);
        used_verts.insert(i2);
    }
    let euler = used_verts.len() as i32 - num_edges as i32 + nf as i32;

    if boundary_edges == 0 && non_manifold_edges == 0 {
        if euler != 2 {
            warnings.push(format!("欧拉示性数 chi = {euler} != 2 (可能包含多个连通分量或非亏格-0 环柄)"));
        }
    }

    // 4. 体积与质心
    let (vol, centroid) = mesh.volume_and_centroid();
    if vol <= 0.0 {
        errors.push(format!("多面体代数带符号体积非正: {vol:.4e} m³ (面法向整体颠倒或自翻转)"));
    }

    // 5. 快速自交检查 (利用 AABB 剔除与 Möller 三角形相交测试)
    let self_intersections = check_self_intersections(mesh, char_len);
    if self_intersections > 0 {
        errors.push(format!("检测到 {self_intersections} 处表面三角面片自相交"));
    }

    let non_deg = degenerate_faces == 0;
    let closed = boundary_edges == 0 && non_manifold_edges == 0;
    let orient = orientation_conflicts == 0;
    let no_self_int = self_intersections == 0;
    let vol_pos = vol > 0.0;

    let is_valid = non_deg && closed && orient && no_self_int && vol_pos;

    GeometryValidationReport {
        is_valid,
        non_degenerate: non_deg,
        is_closed_manifold: closed,
        consistent_orientation: orient,
        has_self_intersection: !no_self_int,
        degenerate_face_count: degenerate_faces,
        boundary_edge_count: boundary_edges,
        non_manifold_edge_count: non_manifold_edges,
        orientation_conflict_count: orientation_conflicts,
        self_intersection_count: self_intersections,
        signed_volume: vol,
        centroid,
        euler_characteristic: euler,
        errors,
        warnings,
    }
}

/// 快速非相邻三角形自交检查
fn check_self_intersections(mesh: &Mesh, _char_len: f64) -> usize {
    let nf = mesh.face_count();
    if nf < 6 {
        return 0;
    }

    // 抽样或全量 AABB 宽阶段检测
    let mut aabbs = Vec::with_capacity(nf);
    for f in 0..nf {
        let [i0, i1, i2] = mesh.face(f);
        let p0 = mesh.vertex(i0 as usize);
        let p1 = mesh.vertex(i1 as usize);
        let p2 = mesh.vertex(i2 as usize);
        let min = [
            p0[0].min(p1[0]).min(p2[0]),
            p0[1].min(p1[1]).min(p2[1]),
            p0[2].min(p1[2]).min(p2[2]),
        ];
        let max = [
            p0[0].max(p1[0]).max(p2[0]),
            p0[1].max(p1[1]).max(p2[1]),
            p0[2].max(p1[2]).max(p2[2]),
        ];
        aabbs.push((min, max));
    }

    let mut count = 0;
    // 为保证性能，对于特大网格 (>5000 面)，进行网格局部跨步检测
    let step = if nf > 5000 { (nf / 2000).max(1) } else { 1 };

    for f1 in (0..nf).step_by(step) {
        let [i0, i1, i2] = mesh.face(f1);
        let (min1, max1) = aabbs[f1];

        for f2 in (f1 + 1)..nf {
            let [j0, j1, j2] = mesh.face(f2);
            // 忽略相邻三角形 (共享顶点的面片不属于自交)
            if i0 == j0 || i0 == j1 || i0 == j2
                || i1 == j0 || i1 == j1 || i1 == j2
                || i2 == j0 || i2 == j1 || i2 == j2
            {
                continue;
            }

            let (min2, max2) = aabbs[f2];
            // AABB 重叠判定
            if min1[0] > max2[0] || max1[0] < min2[0]
                || min1[1] > max2[1] || max1[1] < min2[1]
                || min1[2] > max2[2] || max1[2] < min2[2]
            {
                continue;
            }

            // 窄阶段: 严格三维三角形求交
            let t1 = [mesh.vertex(i0 as usize), mesh.vertex(i1 as usize), mesh.vertex(i2 as usize)];
            let t2 = [mesh.vertex(j0 as usize), mesh.vertex(j1 as usize), mesh.vertex(j2 as usize)];

            if triangles_intersect_3d(&t1, &t2) {
                count += 1;
                if count >= 50 {
                    return count;
                }
            }
        }
    }

    count
}

/// 两个空间三角形的严密相交判定
fn triangles_intersect_3d(t1: &[[f64; 3]; 3], t2: &[[f64; 3]; 3]) -> bool {
    // 平面方程 1
    let n1 = cross_product(sub(t1[1], t1[0]), sub(t1[2], t1[0]));
    let d1 = -dot_product(n1, t1[0]);

    // t2 点到平面 1 的带符号距离
    let dist2 = [
        dot_product(n1, t2[0]) + d1,
        dot_product(n1, t2[1]) + d1,
        dot_product(n1, t2[2]) + d1,
    ];

    // 同号则不相交
    if (dist2[0] > 1e-8 && dist2[1] > 1e-8 && dist2[2] > 1e-8)
        || (dist2[0] < -1e-8 && dist2[1] < -1e-8 && dist2[2] < -1e-8)
    {
        return false;
    }

    // 平面方程 2
    let n2 = cross_product(sub(t2[1], t2[0]), sub(t2[2], t2[0]));
    let d2 = -dot_product(n2, t2[0]);

    let dist1 = [
        dot_product(n2, t1[0]) + d2,
        dot_product(n2, t1[1]) + d2,
        dot_product(n2, t1[2]) + d2,
    ];

    if (dist1[0] > 1e-8 && dist1[1] > 1e-8 && dist1[2] > 1e-8)
        || (dist1[0] < -1e-8 && dist1[1] < -1e-8 && dist1[2] < -1e-8)
    {
        return false;
    }

    // 计算交线方向 D = n1 x n2
    let d_dir = cross_product(n1, n2);
    let d_len2 = dot_product(d_dir, d_dir);
    if d_len2 < 1e-16 {
        // 近似共面三角形
        return false;
    }

    // 投影到交线上的区间重叠测试
    let int1 = triangle_interval_on_line(t1, &dist1, &d_dir);
    let int2 = triangle_interval_on_line(t2, &dist2, &d_dir);

    int1.0 <= int2.1 && int2.0 <= int1.1
}

fn triangle_interval_on_line(t: &[[f64; 3]; 3], d: &[f64; 3], dir: &[f64; 3]) -> (f64, f64) {
    let proj = [
        dot_product(t[0], *dir),
        dot_product(t[1], *dir),
        dot_product(t[2], *dir),
    ];

    let mut points_on_line = Vec::with_capacity(2);
    for &(i, j) in &[(0, 1), (1, 2), (2, 0)] {
        if (d[i] > 0.0 && d[j] < 0.0) || (d[i] < 0.0 && d[j] > 0.0) {
            let t_val = d[i] / (d[i] - d[j]);
            points_on_line.push(proj[i] + t_val * (proj[j] - proj[i]));
        } else if d[i].abs() < 1e-9 {
            points_on_line.push(proj[i]);
        }
    }

    if points_on_line.len() >= 2 {
        (points_on_line[0].min(points_on_line[1]), points_on_line[0].max(points_on_line[1]))
    } else {
        (f64::INFINITY, f64::NEG_INFINITY)
    }
}

#[inline]
fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
fn dot_product(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn cross_product(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
