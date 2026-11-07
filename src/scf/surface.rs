//! CPU queries on the polyhedral surface `Γ` (§8.1, G6, G9): closest points, first
//! crossings of rays and segments, and barycentric coordinates.
//!
//! A bounding-volume hierarchy over the triangles, built once per geometry. The GPU
//! acceleration structure of `rtfp::mesh` serves batched ray traces; this index serves
//! the per-point queries of the runout, the surface transport, and the depth, which are
//! issued one at a time from parallel CPU loops.

use crate::common::math::DVec3;

/// Triangles per leaf.
const LEAF: usize = 4;

#[derive(Clone, Copy, Debug)]
struct Node {
    lo: DVec3,
    hi: DVec3,
    /// Leaf: first slot in `order` and the count. Inner: `count == 0`, children at
    /// `first` and `first + 1`.
    first: u32,
    count: u32,
}

/// The closest point of `Γ` to a query point.
#[derive(Clone, Copy, Debug)]
pub struct Closest {
    pub face: usize,
    pub point: DVec3,
    pub dist2: f64,
}

#[derive(Clone, Debug)]
pub struct SurfaceIndex {
    tris: Vec<[DVec3; 3]>,
    nodes: Vec<Node>,
    order: Vec<u32>,
}

impl SurfaceIndex {
    pub fn build(positions: &[DVec3], faces: &[[u32; 3]]) -> Self {
        let tris: Vec<[DVec3; 3]> = faces
            .iter()
            .map(|t| t.map(|i| positions[i as usize]))
            .collect();
        let centre: Vec<DVec3> = tris.iter().map(|t| (t[0] + t[1] + t[2]) / 3.0).collect();
        let mut order: Vec<u32> = (0..tris.len() as u32).collect();
        let mut nodes = Vec::with_capacity(2 * tris.len() / LEAF + 1);
        nodes.push(Node {
            lo: DVec3::ZERO,
            hi: DVec3::ZERO,
            first: 0,
            count: 0,
        });
        // Iterative build: (node, start, end) ranges of `order`.
        let mut stack = vec![(0usize, 0usize, order.len())];
        while let Some((at, start, end)) = stack.pop() {
            let (mut lo, mut hi) = (DVec3::splat(f64::INFINITY), DVec3::splat(f64::NEG_INFINITY));
            let (mut clo, mut chi) = (lo, hi);
            for &f in &order[start..end] {
                for p in tris[f as usize] {
                    lo = lo.min(p);
                    hi = hi.max(p);
                }
                clo = clo.min(centre[f as usize]);
                chi = chi.max(centre[f as usize]);
            }
            nodes[at].lo = lo;
            nodes[at].hi = hi;
            if end - start <= LEAF {
                nodes[at].first = start as u32;
                nodes[at].count = (end - start) as u32;
                continue;
            }
            let ext = chi - clo;
            let axis = if ext.x >= ext.y && ext.x >= ext.z {
                0
            } else if ext.y >= ext.z {
                1
            } else {
                2
            };
            let mid = start + (end - start) / 2;
            order[start..end].select_nth_unstable_by(mid - start, |a, b| {
                centre[*a as usize][axis].total_cmp(&centre[*b as usize][axis])
            });
            let left = nodes.len();
            nodes.push(nodes[at]);
            nodes.push(nodes[at]);
            nodes[at].first = left as u32;
            nodes[at].count = 0;
            stack.push((left, start, mid));
            stack.push((left + 1, mid, end));
        }
        Self { tris, nodes, order }
    }

    pub fn triangle(&self, face: usize) -> [DVec3; 3] {
        self.tris[face]
    }

    /// The closest point of `Γ` to `x`.
    pub fn closest(&self, x: DVec3) -> Closest {
        let mut best = Closest {
            face: 0,
            point: self.tris.first().map_or(DVec3::ZERO, |t| t[0]),
            dist2: f64::INFINITY,
        };
        let mut stack = vec![0usize];
        while let Some(at) = stack.pop() {
            let n = self.nodes[at];
            if box_dist2(n.lo, n.hi, x) >= best.dist2 {
                continue;
            }
            if n.count > 0 {
                for &f in &self.order[n.first as usize..(n.first + n.count) as usize] {
                    let [a, b, c] = self.tris[f as usize];
                    let p = closest_on_triangle(x, a, b, c);
                    let d2 = (p - x).length_squared();
                    if d2 < best.dist2 {
                        best = Closest {
                            face: f as usize,
                            point: p,
                            dist2: d2,
                        };
                    }
                }
            } else {
                let (l, r) = (n.first as usize, n.first as usize + 1);
                // Nearer child last, so it is popped first.
                let (dl, dr) = (
                    box_dist2(self.nodes[l].lo, self.nodes[l].hi, x),
                    box_dist2(self.nodes[r].lo, self.nodes[r].hi, x),
                );
                if dl < dr {
                    stack.push(r);
                    stack.push(l);
                } else {
                    stack.push(l);
                    stack.push(r);
                }
            }
        }
        best
    }

    /// The first crossing of `Γ` by `o + t d`, `t ∈ (t_min, t_max)`: `(t, face)`.
    pub fn first_hit(&self, o: DVec3, d: DVec3, t_min: f64, t_max: f64) -> Option<(f64, usize)> {
        let inv = DVec3::new(1.0 / d.x, 1.0 / d.y, 1.0 / d.z);
        let mut best: Option<(f64, usize)> = None;
        let mut limit = t_max;
        let mut stack = vec![0usize];
        while let Some(at) = stack.pop() {
            let n = self.nodes[at];
            if !slab(n.lo, n.hi, o, inv, t_min, limit) {
                continue;
            }
            if n.count > 0 {
                for &f in &self.order[n.first as usize..(n.first + n.count) as usize] {
                    let [a, b, c] = self.tris[f as usize];
                    if let Some(t) = ray_triangle(o, d, a, b, c) {
                        if t > t_min && t < limit {
                            limit = t;
                            best = Some((t, f as usize));
                        }
                    }
                }
            } else {
                stack.push(n.first as usize);
                stack.push(n.first as usize + 1);
            }
        }
        best
    }
}

impl SurfaceIndex {
    /// Whether `x` lies inside the solid: the parity of the crossings of a fixed ray
    /// (§3.6). A ray through an edge or a vertex is a null event for a generic direction.
    pub fn inside(&self, x: DVec3) -> bool {
        const DIR: DVec3 = DVec3::new(0.577_350_269_189_625_7, 0.534_522_483_824_848_7, 0.617_213_399_848_368_5);
        let (mut t, mut count) = (0.0, 0usize);
        while let Some((hit, _)) = self.first_hit(x, DIR, t, f64::INFINITY) {
            count += 1;
            t = hit * (1.0 + 1e-12) + 1e-300;
        }
        count % 2 == 1
    }
}

/// Nearest point of a fixed set: a uniform grid of cells holding the points.
#[derive(Clone, Debug)]
pub struct PointGrid {
    pts: Vec<DVec3>,
    lo: DVec3,
    cell: f64,
    dims: [usize; 3],
    start: Vec<u32>,
    items: Vec<u32>,
}

impl PointGrid {
    pub fn build(pts: &[DVec3]) -> Self {
        let (mut lo, mut hi) = (DVec3::splat(f64::INFINITY), DVec3::splat(f64::NEG_INFINITY));
        for p in pts {
            lo = lo.min(*p);
            hi = hi.max(*p);
        }
        let ext = (hi - lo).max(DVec3::splat(1e-12));
        // About two points per cell.
        let cell = (ext.x * ext.y * ext.z * 2.0 / pts.len().max(1) as f64).cbrt().max(1e-12);
        let dims = [0, 1, 2].map(|a| ((ext[a] / cell).ceil() as usize).max(1));
        let key = |p: DVec3| -> usize {
            let c = [0, 1, 2].map(|a| (((p[a] - lo[a]) / cell) as usize).min(dims[a] - 1));
            (c[2] * dims[1] + c[1]) * dims[0] + c[0]
        };
        let n = dims[0] * dims[1] * dims[2];
        let mut start = vec![0u32; n + 1];
        for p in pts {
            start[key(*p) + 1] += 1;
        }
        for i in 0..n {
            start[i + 1] += start[i];
        }
        let mut fill = start.clone();
        let mut items = vec![0u32; pts.len()];
        for (i, p) in pts.iter().enumerate() {
            let k = key(*p);
            items[fill[k] as usize] = i as u32;
            fill[k] += 1;
        }
        Self {
            pts: pts.to_vec(),
            lo,
            cell,
            dims,
            start,
            items,
        }
    }

    /// Every point within `r` of `x`, as `(|d|, index)` in no particular order.
    pub fn within(&self, x: DVec3, r: f64) -> Vec<(f64, usize)> {
        let lo = [0, 1, 2].map(|a| (((x[a] - r - self.lo[a]) / self.cell).floor() as i64).max(0));
        let hi = [0, 1, 2].map(|a| {
            (((x[a] + r - self.lo[a]) / self.cell).floor() as i64).min(self.dims[a] as i64 - 1)
        });
        let mut out = Vec::new();
        for k in lo[2]..=hi[2] {
            for j in lo[1]..=hi[1] {
                for i in lo[0]..=hi[0] {
                    let cell = ((k as usize * self.dims[1]) + j as usize) * self.dims[0] + i as usize;
                    for &p in &self.items[self.start[cell] as usize..self.start[cell + 1] as usize] {
                        let d = (self.pts[p as usize] - x).length();
                        if d < r {
                            out.push((d, p as usize));
                        }
                    }
                }
            }
        }
        out
    }

    /// Index of the point nearest to `x`.
    pub fn nearest(&self, x: DVec3) -> usize {
        self.k_nearest(x, 1, |_| false).first().map_or(0, |(_, p)| *p)
    }

    /// The `count` points nearest to `x` other than those `skip` rejects, as
    /// `(|d|², index)` in increasing distance, by growing shells of cells.
    pub fn k_nearest(&self, x: DVec3, count: usize, skip: impl Fn(usize) -> bool) -> Vec<(f64, usize)> {
        let c = [0, 1, 2].map(|a| {
            (((x[a] - self.lo[a]) / self.cell).floor() as i64).clamp(0, self.dims[a] as i64 - 1)
        });
        let mut best: Vec<(f64, usize)> = Vec::with_capacity(count + 1);
        let max_r = *self.dims.iter().max().unwrap() as i64;
        for r in 0..=max_r {
            for k in -r..=r {
                for j in -r..=r {
                    for i in -r..=r {
                        if i.abs().max(j.abs()).max(k.abs()) != r {
                            continue;
                        }
                        let q = [c[0] + i, c[1] + j, c[2] + k];
                        if (0..3).any(|a| q[a] < 0 || q[a] >= self.dims[a] as i64) {
                            continue;
                        }
                        let cell = ((q[2] as usize * self.dims[1]) + q[1] as usize) * self.dims[0]
                            + q[0] as usize;
                        for &p in &self.items[self.start[cell] as usize..self.start[cell + 1] as usize] {
                            let p = p as usize;
                            let d = (self.pts[p] - x).length_squared();
                            if (best.len() < count || d < best[count - 1].0) && !skip(p) {
                                let at = best.partition_point(|b| b.0 <= d);
                                best.insert(at, (d, p));
                                best.truncate(count);
                            }
                        }
                    }
                }
            }
            // Every unvisited cell is at least r cells away.
            if count == 0 || best.len() == count && best[count - 1].0.sqrt() <= r as f64 * self.cell {
                break;
            }
        }
        best
    }
}

fn box_dist2(lo: DVec3, hi: DVec3, x: DVec3) -> f64 {
    let d = (lo - x).max(DVec3::ZERO).max(x - hi);
    d.length_squared()
}

fn slab(lo: DVec3, hi: DVec3, o: DVec3, inv: DVec3, t0: f64, t1: f64) -> bool {
    let a = (lo - o) * inv;
    let b = (hi - o) * inv;
    let near = a.min(b);
    let far = a.max(b);
    let enter = near.x.max(near.y).max(near.z).max(t0);
    let exit = far.x.min(far.y).min(far.z).min(t1);
    // NaN (a zero direction component on the slab plane) keeps the node.
    !(enter > exit)
}

/// Möller–Trumbore: the ray parameter of the crossing with triangle `abc`, two-sided.
pub fn ray_triangle(o: DVec3, d: DVec3, a: DVec3, b: DVec3, c: DVec3) -> Option<f64> {
    let (e1, e2) = (b - a, c - a);
    let p = d.cross(e2);
    let det = e1.dot(p);
    if det.abs() < 1e-300 {
        return None;
    }
    let s = o - a;
    let u = s.dot(p) / det;
    if !(-1e-12..=1.0 + 1e-12).contains(&u) {
        return None;
    }
    let q = s.cross(e1);
    let v = d.dot(q) / det;
    if v < -1e-12 || u + v > 1.0 + 1e-12 {
        return None;
    }
    Some(e2.dot(q) / det)
}

/// Barycentric weights of `p` (projected on the plane of `abc`).
pub fn barycentric(p: DVec3, a: DVec3, b: DVec3, c: DVec3) -> [f64; 3] {
    let n = (b - a).cross(c - a);
    let area = n.length_squared().max(1e-300);
    let wa = (c - b).cross(p - b).dot(n) / area;
    let wb = (a - c).cross(p - c).dot(n) / area;
    [wa, wb, 1.0 - wa - wb]
}

/// The point of triangle `abc` closest to `p` (Voronoi regions of the vertices, edges, face).
pub fn closest_on_triangle(p: DVec3, a: DVec3, b: DVec3, c: DVec3) -> DVec3 {
    let (ab, ac, ap) = (b - a, c - a, p - a);
    let (d1, d2) = (ab.dot(ap), ac.dot(ap));
    if d1 <= 0.0 && d2 <= 0.0 {
        return a;
    }
    let bp = p - b;
    let (d3, d4) = (ab.dot(bp), ac.dot(bp));
    if d3 >= 0.0 && d4 <= d3 {
        return b;
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        return a + ab * (d1 / (d1 - d3));
    }
    let cp = p - c;
    let (d5, d6) = (ab.dot(cp), ac.dot(cp));
    if d6 >= 0.0 && d5 <= d6 {
        return c;
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        return a + ac * (d2 / (d2 - d6));
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && d4 - d3 >= 0.0 && d5 - d6 >= 0.0 {
        return b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)));
    }
    let denom = 1.0 / (va + vb + vc);
    a + ab * (vb * denom) + ac * (vc * denom)
}
