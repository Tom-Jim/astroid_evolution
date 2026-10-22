//! Face tree of the closed-form uniform field (MS §3.10, far-field multipoles).
//!
//! `Φ_V(x) = ∫_V dV/|x − y|` is a surface integral, `Φ_V = ½(P − x·S)` with
//! `S_i = ∮ n_i/|x − y| dS` and `P = ∮ (n·y)/|x − y| dS`; `∇Φ_V = −S` and
//! `∇∇Φ_V = −∇S`. Near faces are summed exactly (Werner–Scheeres); a cluster of
//! faces seen under a small angle uses the Cartesian multipole expansion of its
//! four charge densities `n_x dS, n_y dS, n_z dS, n·(y − c) dS` through octupole
//! order, with each triangle's own central moments included.

use rayon::prelude::*;

use crate::common::math::DVec3;

/// Faces per leaf.
const LEAF: usize = 4;

/// Default opening angle `h / |x − c|` below which a node uses its expansion.
pub const THETA: f64 = 0.1;

/// One node, as `TNode` in `rtfp/shader/uniform.wesl` (48 words).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TreeNode {
    /// Expansion centre and bounding radius.
    pub c: [f32; 3],
    pub h: f32,
    /// Leaf: first face slot and count. Inner: count 0, children `a`, `b`.
    pub first: u32,
    pub count: u32,
    pub a: u32,
    pub b: u32,
    /// Monopoles of the four charge types.
    pub q0: [f32; 4],
    /// Dipoles: x, y, z components, each over the four types.
    pub q1: [[f32; 4]; 3],
    /// Quadrupoles (xx, yy, zz, xy, xz, yz), each over the four types.
    pub q2: [[f32; 4]; 6],
    /// Octupoles in the order of `O3`, each over the four types.
    pub q3: [[f32; 4]; 10],
}

/// Index triples of the ten independent octupole components.
const O3: [[usize; 3]; 10] = [
    [0, 0, 0],
    [1, 1, 1],
    [2, 2, 2],
    [0, 0, 1],
    [0, 0, 2],
    [0, 1, 1],
    [1, 1, 2],
    [0, 2, 2],
    [1, 2, 2],
    [0, 1, 2],
];
/// Number of distinct permutations of each `O3` triple.
const O3_MULT: [f64; 10] = [1.0, 1.0, 1.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 6.0];
const O2: [[usize; 2]; 6] = [[0, 0], [1, 1], [2, 2], [0, 1], [0, 2], [1, 2]];

/// Per-face data for the moments: centroid, area times normal, and the
/// area second moment about the centroid.
struct FaceMom {
    g: DVec3,
    an: DVec3,
    /// `∫_f (y − g)^⊗2 dA` in the order of `O2`, and `∫_f (y − g)^⊗3 dA` in that of `O3`.
    m2: [f64; 6],
    m3: [f64; 10],
}

fn face_mom(v: [DVec3; 3]) -> FaceMom {
    let g = (v[0] + v[1] + v[2]) / 3.0;
    let an = (v[1] - v[0]).cross(v[2] - v[0]) * 0.5;
    let area = an.length();
    // Central moments through y − g = Σ λ_i e_i and ∫ λ^a λ^b λ^c dA = 2A a!b!c!/(a+b+c+2)!.
    let e = v.map(|p| (p - g).to_array());
    let l2 = |i: usize, j: usize| if i == j { area / 6.0 } else { area / 12.0 };
    let l3 = |i: usize, j: usize, k: usize| match (i == j, j == k, i == k) {
        (true, true, _) => area / 10.0,
        (false, false, false) => area / 60.0,
        _ => area / 30.0,
    };
    let m2 = O2.map(|[a, b]| {
        let mut s = 0.0;
        for i in 0..3 {
            for j in 0..3 {
                s += e[i][a] * e[j][b] * l2(i, j);
            }
        }
        s
    });
    let m3 = O3.map(|[a, b, c]| {
        let mut s = 0.0;
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    s += e[i][a] * e[j][b] * e[k][c] * l3(i, j, k);
                }
            }
        }
        s
    });
    FaceMom { g, an, m2, m3 }
}

impl FaceMom {
    fn m2_at(&self, a: usize, b: usize) -> f64 {
        self.m2[match (a.min(b), a.max(b)) {
            (0, 0) => 0,
            (1, 1) => 1,
            (2, 2) => 2,
            (0, 1) => 3,
            (0, 2) => 4,
            _ => 5,
        }]
    }
}

fn o3_index(a: usize, b: usize, c: usize) -> usize {
    let mut k = [a, b, c];
    k.sort_unstable();
    O3.iter().position(|o| *o == k).unwrap()
}

/// Moments of one node in f64 while the tree is built: the four charge types
/// about the node centre (type 3 is `n·(y − c)` with this node's `c`).
#[derive(Clone, Copy, Default)]
struct Moments {
    c: DVec3,
    h: f64,
    area: f64,
    q0: [f64; 4],
    q1: [[f64; 4]; 3],
    q2: [[f64; 4]; 6],
    q3: [[f64; 4]; 10],
}

impl Moments {
    /// Moments of a leaf about its area-weighted centroid.
    fn leaf(m: &[FaceMom], tris: &[[DVec3; 3]], ids: &[u32]) -> Self {
        let mut out = Moments::default();
        let mut cw = DVec3::ZERO;
        for &i in ids {
            let a = m[i as usize].an.length();
            cw += m[i as usize].g * a;
            out.area += a;
        }
        out.c = if out.area > 0.0 {
            cw / out.area
        } else {
            m[ids[0] as usize].g
        };
        let c = out.c;
        out.h = ids
            .iter()
            .flat_map(|&i| tris[i as usize])
            .map(|p| (p - c).length())
            .fold(0.0, f64::max);
        for &i in ids {
            let f = &m[i as usize];
            let d = f.g - c;
            let area = f.an.length();
            let unit = if area > 0.0 { f.an / area } else { DVec3::ZERO };
            let dv = d.to_array();
            // (d + ε)^⊗k over the face, ∫ ε dA = 0; every charge density is
            // constant on a planar face (n·(y − c) = n·d there).
            let pair = O2.map(|[a, b]| area * dv[a] * dv[b] + f.m2_at(a, b));
            let triple = O3.map(|[a, b, c]| {
                area * dv[a] * dv[b] * dv[c]
                    + dv[a] * f.m2_at(b, c)
                    + dv[b] * f.m2_at(a, c)
                    + dv[c] * f.m2_at(a, b)
                    + f.m3[o3_index(a, b, c)]
            });
            let sigma = [unit.x, unit.y, unit.z, unit.dot(d)];
            for t in 0..4 {
                out.q0[t] += sigma[t] * area;
                for k in 0..3 {
                    out.q1[k][t] += sigma[t] * area * dv[k];
                }
                for k in 0..6 {
                    out.q2[k][t] += sigma[t] * pair[k];
                }
                for k in 0..10 {
                    out.q3[k][t] += sigma[t] * triple[k];
                }
            }
        }
        out
    }

    /// Moments about `c` (multipole-to-multipole translation by s = c_old − c).
    fn shifted(&self, c: DVec3) -> Self {
        let s = (self.c - c).to_array();
        let mut out = *self;
        out.c = c;
        // Type 3 about the new centre: n·(y − c) = n·(y − c_old) + Σ_i s_i n_i.
        let mut q = *self;
        for k in 0..3 {
            q.q0[3] += s[k] * self.q0[k];
            for j in 0..3 {
                q.q1[j][3] += s[k] * self.q1[j][k];
            }
            for j in 0..6 {
                q.q2[j][3] += s[k] * self.q2[j][k];
            }
            for j in 0..10 {
                q.q3[j][3] += s[k] * self.q3[j][k];
            }
        }
        for t in 0..4 {
            let (m0, m1) = (q.q0[t], [q.q1[0][t], q.q1[1][t], q.q1[2][t]]);
            let m2 = |a: usize, b: usize| q.q2[o2_index(a, b)][t];
            out.q0[t] = m0;
            for a in 0..3 {
                out.q1[a][t] = m1[a] + s[a] * m0;
            }
            for (k, [a, b]) in O2.iter().enumerate() {
                out.q2[k][t] = q.q2[k][t] + s[*a] * m1[*b] + s[*b] * m1[*a] + s[*a] * s[*b] * m0;
            }
            for (k, [a, b, c]) in O3.iter().enumerate() {
                let (a, b, c) = (*a, *b, *c);
                out.q3[k][t] = q.q3[k][t]
                    + s[a] * m2(b, c)
                    + s[b] * m2(a, c)
                    + s[c] * m2(a, b)
                    + s[a] * s[b] * m1[c]
                    + s[a] * s[c] * m1[b]
                    + s[b] * s[c] * m1[a]
                    + s[a] * s[b] * s[c] * m0;
            }
        }
        out
    }

    fn merge(a: &Self, b: &Self) -> Self {
        let area = a.area + b.area;
        let c = if area > 0.0 {
            (a.c * a.area + b.c * b.area) / area
        } else {
            (a.c + b.c) * 0.5
        };
        let (sa, sb) = (a.shifted(c), b.shifted(c));
        let mut out = sa;
        out.area = area;
        out.h = (a.h + (a.c - c).length()).max(b.h + (b.c - c).length());
        for t in 0..4 {
            out.q0[t] += sb.q0[t];
            for k in 0..3 {
                out.q1[k][t] += sb.q1[k][t];
            }
            for k in 0..6 {
                out.q2[k][t] += sb.q2[k][t];
            }
            for k in 0..10 {
                out.q3[k][t] += sb.q3[k][t];
            }
        }
        out
    }

    fn node(&self, first: usize, count: usize, a: usize, b: usize) -> TreeNode {
        let to4 = |a: [f64; 4]| a.map(|v| v as f32);
        TreeNode {
            c: self.c.as_vec3().to_array(),
            h: self.h as f32,
            first: first as u32,
            count: count as u32,
            a: a as u32,
            b: b as u32,
            q0: to4(self.q0),
            q1: self.q1.map(to4),
            q2: self.q2.map(to4),
            q3: self.q3.map(to4),
        }
    }
}

fn o2_index(a: usize, b: usize) -> usize {
    match (a.min(b), a.max(b)) {
        (0, 0) => 0,
        (1, 1) => 1,
        (2, 2) => 2,
        (0, 1) => 3,
        (0, 2) => 4,
        _ => 5,
    }
}

/// Build the tree over unit-frame triangles. Returns the nodes (root first) and
/// the face order: `order[slot] = original face index`.
pub fn build(tris: &[[DVec3; 3]]) -> (Vec<TreeNode>, Vec<u32>) {
    let moms: Vec<FaceMom> = tris.par_iter().map(|t| face_mom(*t)).collect();
    let mut order: Vec<u32> = (0..tris.len() as u32).collect();
    let mut nodes = Vec::with_capacity(2 * tris.len() / LEAF + 1);
    nodes.push(bytemuck::Zeroable::zeroed());
    split(&moms, tris, &mut order, 0, tris.len(), 0, &mut nodes);
    (nodes, order)
}

/// Median split along the longest axis of the centroid box; children before the
/// parent's moments, which are translated up from them.
fn split(
    m: &[FaceMom],
    tris: &[[DVec3; 3]],
    order: &mut [u32],
    lo: usize,
    hi: usize,
    slot: usize,
    nodes: &mut Vec<TreeNode>,
) -> Moments {
    if hi - lo <= LEAF {
        let mo = Moments::leaf(m, tris, &order[lo..hi]);
        nodes[slot] = mo.node(lo, hi - lo, 0, 0);
        return mo;
    }
    let ids = &mut order[lo..hi];
    let (mut bmin, mut bmax) = (DVec3::splat(f64::MAX), DVec3::splat(f64::MIN));
    for &i in ids.iter() {
        bmin = bmin.min(m[i as usize].g);
        bmax = bmax.max(m[i as usize].g);
    }
    let ext = bmax - bmin;
    let axis = if ext.x >= ext.y && ext.x >= ext.z {
        0
    } else if ext.y >= ext.z {
        1
    } else {
        2
    };
    let mid = (hi - lo) / 2;
    ids.select_nth_unstable_by(mid, |a, b| {
        m[*a as usize].g[axis].total_cmp(&m[*b as usize].g[axis])
    });
    let a = nodes.len();
    nodes.push(bytemuck::Zeroable::zeroed());
    let b = nodes.len();
    nodes.push(bytemuck::Zeroable::zeroed());
    let ma = split(m, tris, order, lo, lo + mid, a, nodes);
    let mb = split(m, tris, order, lo + mid, hi, b, nodes);
    let mut mo = Moments::merge(&ma, &mb);
    // The exact radius: the merged bound h_a + |c_a − c| opens nodes needlessly.
    let c = mo.c;
    mo.h = order[lo..hi]
        .iter()
        .flat_map(|&i| tris[i as usize])
        .map(|p| (p - c).length_squared())
        .fold(0.0, f64::max)
        .sqrt();
    nodes[slot] = mo.node(lo, 0, a, b);
    mo
}

/// Whether a node is far enough from `x` for its expansion: `|x − c| > h / θ`.
#[inline]
pub fn is_far(n: &TreeNode, x: DVec3, theta: f64) -> bool {
    let c = DVec3::new(n.c[0].into(), n.c[1].into(), n.c[2].into());
    let r = (x - c).length();
    r * theta > f64::from(n.h)
}

/// CPU reference of `rtfp_uniform`'s traversal in f64: `(Φ_V, ∇Φ_V, ∇∇Φ_V, Σω)`
/// with `tris` in slot order and `own` the slot of the face containing `x`.
pub fn eval(
    nodes: &[TreeNode],
    tris: &[[DVec3; 3]],
    x: DVec3,
    theta: f64,
    own: Option<usize>,
) -> (f64, DVec3, [f64; 6], f64) {
    use crate::rtfp::reference::werner::{FaceTerm, face_term};
    let mut near = FaceTerm::default();
    let (mut u, mut g, mut h) = (0.0, DVec3::ZERO, [0.0; 6]);
    let mut stack = vec![0usize];
    while let Some(i) = stack.pop() {
        let n = &nodes[i];
        if is_far(n, x, theta) {
            let (fu, fg, fh) = far_field(n, x);
            u += fu;
            g += fg;
            for k in 0..6 {
                h[k] += fh[k];
            }
        } else if n.count > 0 {
            for f in n.first as usize..(n.first + n.count) as usize {
                near.add(&face_term(tris[f], x, own == Some(f)));
            }
        } else {
            stack.push(n.a as usize);
            stack.push(n.b as usize);
        }
    }
    let w = near.scaled(1.0);
    for k in 0..6 {
        h[k] += w.hessian[k];
    }
    // tr ∇∇Φ_V = −Σω face by face (n ⊥ m_e), so the far nodes add −tr of theirs.
    let omega = near.omega - (h[0] + h[1] + h[2] - w.hessian[0] - w.hessian[1] - w.hessian[2]);
    (u + w.potential, g + w.gradient, h, omega)
}

/// The distinct permutations of an index triple.
fn perms([i, j, k]: [usize; 3]) -> Vec<(usize, usize, usize)> {
    let mut all = vec![
        (i, j, k),
        (i, k, j),
        (j, i, k),
        (j, k, i),
        (k, i, j),
        (k, j, i),
    ];
    all.sort_unstable();
    all.dedup();
    all
}

/// The far-field expansion of one node at `x`: `(Φ, ∇Φ, ∇∇Φ)`.
pub fn far_field(n: &TreeNode, x: DVec3) -> (f64, DVec3, [f64; 6]) {
    let c = DVec3::new(n.c[0].into(), n.c[1].into(), n.c[2].into());
    let r = x - c;
    let r2 = r.length_squared();
    let rr = r2.sqrt();
    let mut phi = [0.0; 4];
    let mut grad = [DVec3::ZERO; 4];
    for t in 0..4 {
        let q0 = f64::from(n.q0[t]);
        let q1 = DVec3::new(n.q1[0][t].into(), n.q1[1][t].into(), n.q1[2][t].into());
        let q = |k: usize| f64::from(n.q2[k][t]);
        let qr = DVec3::new(
            q(0) * r.x + q(3) * r.y + q(4) * r.z,
            q(3) * r.x + q(1) * r.y + q(5) * r.z,
            q(4) * r.x + q(5) * r.y + q(2) * r.z,
        );
        let a = r.dot(qr);
        let tr = q(0) + q(1) + q(2);
        let (r3, r5, r7) = (rr * r2, rr * r2 * r2, rr * r2 * r2 * r2);
        // Octupole: a₃ = M(r,r,r), w = M(·,r,r), v_k = M_iik.
        let rv = r.to_array();
        let (mut a3, mut w, mut v) = (0.0, [0.0; 3], [0.0; 3]);
        for (k, [i, j, l]) in O3.iter().enumerate() {
            let m = f64::from(n.q3[k][t]);
            let mult = O3_MULT[k];
            a3 += mult * m * rv[*i] * rv[*j] * rv[*l];
            // Each distinct permutation (p, q, s) contributes m r_q r_s to w_p and, when p = q, m to v_s.
            for (p, q, s) in perms([*i, *j, *l]) {
                w[p] += m * rv[q] * rv[s];
                if p == q {
                    v[s] += m;
                }
            }
        }
        let (w, v) = (DVec3::from_array(w), DVec3::from_array(v));
        let r9 = r7 * r2;
        let vr = v.dot(r);
        phi[t] = q0 / rr
            + r.dot(q1) / r3
            + (3.0 * a - r2 * tr) / (2.0 * r5)
            + (5.0 * a3 - 3.0 * r2 * vr) / (2.0 * r7);
        grad[t] = -r * (q0 / r3) + q1 / r3 - r * (3.0 * r.dot(q1) / r5) + (qr * 3.0 - r * tr) / r5
            - r * (5.0 * (3.0 * a - r2 * tr) / (2.0 * r7))
            - (r * (105.0 * a3 - 45.0 * r2 * vr) - w * (45.0 * r2) + v * (9.0 * r2 * r2))
                / (6.0 * r9);
    }
    let s = DVec3::new(phi[0], phi[1], phi[2]);
    let pot = 0.5 * (phi[3] - r.dot(s));
    let h = [
        -grad[0].x,
        -grad[1].y,
        -grad[2].z,
        -0.5 * (grad[0].y + grad[1].x),
        -0.5 * (grad[0].z + grad[2].x),
        -0.5 * (grad[1].z + grad[2].y),
    ];
    (pot, -s, h)
}
