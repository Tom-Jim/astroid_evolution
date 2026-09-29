//! 二叉 BVH 层次包围盒加速结构 (支持 CPU 遍历与 GPU WGSL 扁平 SoA 内存布局)
//! 严格依照 docs/software.md 第 7.1 节与 docs/mathstrict.md 第 1.4 节

use super::mesh::Mesh;

const LEAF_SIZE: usize = 4;
const MAX_DEPTH: u32 = 40;
const STACK_SIZE: usize = 64;

#[derive(Clone, Copy, Default, Debug)]
struct Node {
    min: [f32; 3],
    max: [f32; 3],
    left: u32,
    right: u32,
    start: u32,
    count: u32,
}

/// 扁平化 GPU/CPU 兼用的 SoA BVH 加速结构
#[derive(Clone, Debug, Default)]
pub struct FlattenedBvh {
    /// 兼容旧布局: 每个节点两行边界 (min, max)
    pub bounds: Vec<[f32; 4]>,
    /// 兼容旧布局: 每个节点元数据 (left, right, start, count)
    pub meta: Vec<[u32; 4]>,
    /// 兼容旧布局: 三角形排列索引
    pub order: Vec<u32>,
    /// 节点 AABB 最小边界 [min_x, min_y, min_z, 0.0]
    pub bounds_min: Vec<[f32; 4]>,
    /// 节点 AABB 最大边界 [max_x, max_y, max_z, 0.0]
    pub bounds_max: Vec<[f32; 4]>,
    /// 左子节点索引
    pub left: Vec<u32>,
    /// 右子节点索引
    pub right: Vec<u32>,
    /// 叶节点三角面片在 triangle_indices 中的起始偏移
    pub start: Vec<u32>,
    /// 叶节点所包含三角面片数量 (内部节点为 0)
    pub count: Vec<u32>,
    /// 排列后的三角形全局索引
    pub triangle_indices: Vec<u32>,
    /// 树最大深度
    pub depth: u32,
}

impl FlattenedBvh {
    pub fn node_count(&self) -> usize {
        self.bounds_min.len()
    }
}

/// 保持向后兼容的 Bvh 包装
pub type Bvh = FlattenedBvh;

impl Bvh {
    pub fn build(mesh: &Mesh) -> Self {
        build_flattened_bvh(mesh)
    }
}

/// 构造扁平化 SoA BVH 加速结构
pub fn build_flattened_bvh(mesh: &Mesh) -> FlattenedBvh {
    let nf = mesh.face_count();
    let mut order: Vec<u32> = (0..nf as u32).collect();
    let mut tris: Vec<([f32; 3], [f32; 3], [f32; 3])> = Vec::with_capacity(nf);

    for f in 0..nf {
        let [i0, i1, i2] = mesh.face(f);
        let c = [
            mesh.vertex(i0 as usize),
            mesh.vertex(i1 as usize),
            mesh.vertex(i2 as usize),
        ];
        let lo = [
            c[0][0].min(c[1][0]).min(c[2][0]) as f32,
            c[0][1].min(c[1][1]).min(c[2][1]) as f32,
            c[0][2].min(c[1][2]).min(c[2][2]) as f32,
        ];
        let hi = [
            c[0][0].max(c[1][0]).max(c[2][0]) as f32,
            c[0][1].max(c[1][1]).max(c[2][1]) as f32,
            c[0][2].max(c[1][2]).max(c[2][2]) as f32,
        ];
        let centroid = [
            (lo[0] + hi[0]) * 0.5,
            (lo[1] + hi[1]) * 0.5,
            (lo[2] + hi[2]) * 0.5,
        ];
        let pad = 1e-4f32;
        tris.push((
            [lo[0] - pad, lo[1] - pad, lo[2] - pad],
            [hi[0] + pad, hi[1] + pad, hi[2] + pad],
            centroid,
        ));
    }

    let mut builder = Builder {
        tris: &tris,
        nodes: Vec::with_capacity(2 * nf / LEAF_SIZE + 16),
        depth: 0,
    };
    builder.build(&mut order, 0, 0);

    let num_nodes = builder.nodes.len();
    let mut bounds = Vec::with_capacity(num_nodes * 2);
    let mut meta = Vec::with_capacity(num_nodes);
    let mut bounds_min = Vec::with_capacity(num_nodes);
    let mut bounds_max = Vec::with_capacity(num_nodes);
    let mut left = Vec::with_capacity(num_nodes);
    let mut right = Vec::with_capacity(num_nodes);
    let mut start = Vec::with_capacity(num_nodes);
    let mut count = Vec::with_capacity(num_nodes);

    for n in &builder.nodes {
        bounds.push([n.min[0], n.min[1], n.min[2], 0.0]);
        bounds.push([n.max[0], n.max[1], n.max[2], 0.0]);
        meta.push([n.left, n.right, n.start, n.count]);

        bounds_min.push([n.min[0], n.min[1], n.min[2], 0.0]);
        bounds_max.push([n.max[0], n.max[1], n.max[2], 0.0]);
        left.push(n.left);
        right.push(n.right);
        start.push(n.start);
        count.push(n.count);
    }

    FlattenedBvh {
        bounds,
        meta,
        order: order.clone(),
        bounds_min,
        bounds_max,
        left,
        right,
        start,
        count,
        triangle_indices: order,
        depth: builder.depth,
    }
}

struct Builder<'a> {
    tris: &'a [([f32; 3], [f32; 3], [f32; 3])],
    nodes: Vec<Node>,
    depth: u32,
}

impl Builder<'_> {
    fn build(&mut self, idx: &mut [u32], offset: usize, depth: u32) -> u32 {
        let me = self.nodes.len() as u32;
        self.nodes.push(Node::default());
        self.depth = self.depth.max(depth);

        let mut lo = [f32::INFINITY; 3];
        let mut hi = [f32::NEG_INFINITY; 3];
        let mut clo = [f32::INFINITY; 3];
        let mut chi = [f32::NEG_INFINITY; 3];

        for &t in idx.iter() {
            let (tlo, thi, tc) = self.tris[t as usize];
            for k in 0..3 {
                lo[k] = lo[k].min(tlo[k]);
                hi[k] = hi[k].max(thi[k]);
                clo[k] = clo[k].min(tc[k]);
                chi[k] = chi[k].max(tc[k]);
            }
        }

        self.nodes[me as usize].min = lo;
        self.nodes[me as usize].max = hi;

        if idx.len() <= LEAF_SIZE || depth >= MAX_DEPTH {
            let slot = &mut self.nodes[me as usize];
            slot.start = offset as u32;
            slot.count = idx.len() as u32;
            return me;
        }

        let mut axis = 0usize;
        let mut best = chi[0] - clo[0];
        for k in 1..3 {
            let e = chi[k] - clo[k];
            if e > best {
                best = e;
                axis = k;
            }
        }

        let mid = idx.len() / 2;
        let centroids = self.tris;
        idx.select_nth_unstable_by(mid, |a, b| {
            let ca = centroids[*a as usize].2[axis];
            let cb = centroids[*b as usize].2[axis];
            ca.partial_cmp(&cb).unwrap_or(std::cmp::Ordering::Equal)
        });

        let (left_slice, right_slice) = idx.split_at_mut(mid);
        let left = self.build(left_slice, offset, depth + 1);
        let right = self.build(right_slice, offset + mid, depth + 1);

        let slot = &mut self.nodes[me as usize];
        slot.left = left;
        slot.right = right;
        me
    }
}

/// 使用非递归显式栈遍历 BVH 求交，返回全部排序后的正碰撞距离:
/// 0 < R_1 < R_2 < ... < R_L
pub fn intersect_ray_bvh(
    bvh: &FlattenedBvh,
    mesh: &Mesh,
    origin: [f64; 3],
    direction: [f64; 3],
) -> Vec<f64> {
    if bvh.bounds_min.is_empty() {
        return Vec::new();
    }

    let orig_f = [origin[0] as f32, origin[1] as f32, origin[2] as f32];
    let dir_f = [direction[0] as f32, direction[1] as f32, direction[2] as f32];

    let inv_dir = [
        if dir_f[0].abs() > 1e-12 { 1.0 / dir_f[0] } else { 1e12 },
        if dir_f[1].abs() > 1e-12 { 1.0 / dir_f[1] } else { 1e12 },
        if dir_f[2].abs() > 1e-12 { 1.0 / dir_f[2] } else { 1e12 },
    ];

    let mut hits = Vec::with_capacity(8);
    let mut stack = [0u32; STACK_SIZE];
    let mut stack_ptr = 0usize;

    stack[0] = 0; // 根节点入栈
    stack_ptr += 1;

    while stack_ptr > 0 {
        stack_ptr -= 1;
        let node_idx = stack[stack_ptr] as usize;

        let bmin = bvh.bounds_min[node_idx];
        let bmax = bvh.bounds_max[node_idx];

        // 射线-AABB slab 求交测试
        let tx1 = (bmin[0] - orig_f[0]) * inv_dir[0];
        let tx2 = (bmax[0] - orig_f[0]) * inv_dir[0];
        let tmin_x = tx1.min(tx2);
        let tmax_x = tx1.max(tx2);

        let ty1 = (bmin[1] - orig_f[1]) * inv_dir[1];
        let ty2 = (bmax[1] - orig_f[1]) * inv_dir[1];
        let tmin_y = ty1.min(ty2);
        let tmax_y = ty1.max(ty2);

        let tz1 = (bmin[2] - orig_f[2]) * inv_dir[2];
        let tz2 = (bmax[2] - orig_f[2]) * inv_dir[2];
        let tmin_z = tz1.min(tz2);
        let tmax_z = tz1.max(tz2);

        let t_enter = tmin_x.max(tmin_y).max(tmin_z);
        let t_exit = tmax_x.min(tmax_y).min(tmax_z);

        if t_exit < 0.0 || t_enter > t_exit {
            continue; // 未命中该 AABB
        }

        let cnt = bvh.count[node_idx];
        if cnt > 0 {
            // 叶节点: 遍历所含三角形
            let st = bvh.start[node_idx] as usize;
            for i in 0..cnt as usize {
                let tri_idx = bvh.triangle_indices[st + i] as usize;
                let [i0, i1, i2] = mesh.face(tri_idx);
                let p0 = mesh.vertex(i0 as usize);
                let p1 = mesh.vertex(i1 as usize);
                let p2 = mesh.vertex(i2 as usize);

                if let Some(t) = ray_triangle_intersect_f64(origin, direction, p0, p1, p2) {
                    if t > 1e-9 {
                        hits.push(t);
                    }
                }
            }
        } else {
            // 内部节点: 将左右子节点压入栈 (固定大小显式栈)
            let l = bvh.left[node_idx];
            let r = bvh.right[node_idx];
            if stack_ptr + 2 < STACK_SIZE {
                stack[stack_ptr] = l;
                stack_ptr += 1;
                stack[stack_ptr] = r;
                stack_ptr += 1;
            }
        }
    }

    // 命中距离升序排序并去重
    hits.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    hits.dedup_by(|a, b| (*a - *b).abs() < 1e-7);
    hits
}

/// Möller–Trumbore f64 高精度射线-三角形相交测试
#[inline]
fn ray_triangle_intersect_f64(
    orig: [f64; 3],
    dir: [f64; 3],
    v0: [f64; 3],
    v1: [f64; 3],
    v2: [f64; 3],
) -> Option<f64> {
    let e1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
    let e2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];

    let pvec = [
        dir[1] * e2[2] - dir[2] * e2[1],
        dir[2] * e2[0] - dir[0] * e2[2],
        dir[0] * e2[1] - dir[1] * e2[0],
    ];

    let det = e1[0] * pvec[0] + e1[1] * pvec[1] + e1[2] * pvec[2];
    if det.abs() < 1e-15 {
        return None;
    }

    let inv_det = 1.0 / det;
    let tvec = [orig[0] - v0[0], orig[1] - v0[1], orig[2] - v0[2]];

    let u = (tvec[0] * pvec[0] + tvec[1] * pvec[1] + tvec[2] * pvec[2]) * inv_det;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }

    let qvec = [
        tvec[1] * e1[2] - tvec[2] * e1[1],
        tvec[2] * e1[0] - tvec[0] * e1[2],
        tvec[0] * e1[1] - tvec[1] * e1[0],
    ];

    let v = (dir[0] * qvec[0] + dir[1] * qvec[1] + dir[2] * qvec[2]) * inv_det;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }

    let t = (e2[0] * qvec[0] + e2[1] * qvec[1] + e2[2] * qvec[2]) * inv_det;
    if t > 0.0 {
        Some(t)
    } else {
        None
    }
}

/// 判定空间场点 x 是否位于闭合网格内部 (射线奇偶性 Jordan 规则)
pub fn is_point_inside(bvh: &FlattenedBvh, mesh: &Mesh, pt: [f64; 3]) -> bool {
    // 沿随机扰动方向发射射线，避免掠射共边
    let dir: [f64; 3] = [0.1234567, 0.7890123, 0.5987654];
    let len = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
    let norm_dir = [dir[0] / len, dir[1] / len, dir[2] / len];

    let hits = intersect_ray_bvh(bvh, mesh, pt, norm_dir);
    // 奇数次穿透表示在实体内部，偶数次表示在外部
    hits.len() % 2 == 1
}
