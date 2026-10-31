//! Labelled voxels of a body and the closed surface of one label by marching tetrahedra
//! (§12.2): each cube of voxel centres is cut into the six tetrahedra of its main diagonal,
//! so neighbouring cubes agree on every face and the surface is a closed 2-manifold.

use std::collections::HashMap;

use crate::common::math::DVec3;

/// A regular grid of voxel centres, padded by one empty layer, with a label per voxel
/// (`u32::MAX`: outside every block).
pub struct Voxels {
    pub lo: DVec3,
    pub h: f64,
    pub dims: [usize; 3],
    pub label: Vec<u32>,
}

pub const EMPTY: u32 = u32::MAX;

impl Voxels {
    /// The grid over `[lo, hi]` with `n` voxels along the longest side and the label
    /// `label(x)` at each centre.
    pub fn new(lo: DVec3, hi: DVec3, n: usize, label: impl Fn(DVec3) -> u32 + Sync) -> Self {
        use rayon::prelude::*;
        let h = (hi - lo).max_element() / n as f64;
        let lo = lo - DVec3::splat(1.5 * h);
        let dims = [0, 1, 2].map(|a| ((hi[a] - lo[a]) / h).ceil() as usize + 2);
        let total = dims[0] * dims[1] * dims[2];
        let label = (0..total)
            .into_par_iter()
            .map(|id| {
                let [i, j, k] = unflat(id, dims);
                let pad = i == 0 || j == 0 || k == 0 || i + 1 == dims[0] || j + 1 == dims[1] || k + 1 == dims[2];
                if pad { EMPTY } else { label(centre(lo, h, [i, j, k])) }
            })
            .collect();
        Self { lo, h, dims, label }
    }

    pub fn centre(&self, id: usize) -> DVec3 {
        centre(self.lo, self.h, unflat(id, self.dims))
    }

    fn neighbours(&self, id: usize) -> impl Iterator<Item = usize> + '_ {
        let c = unflat(id, self.dims);
        (0..6).filter_map(move |s| {
            let (a, d) = (s / 2, if s % 2 == 0 { -1i64 } else { 1 });
            let mut q = c.map(|v| v as i64);
            q[a] += d;
            (q[a] >= 0 && (q[a] as usize) < self.dims[a]).then(|| flat([q[0] as usize, q[1] as usize, q[2] as usize], self.dims))
        })
    }

    /// Keeps the largest face-connected component of every label and gives each smaller
    /// one to the label it touches most, until every label is connected.
    pub fn connect(&mut self) {
        loop {
            let mut comp = vec![u32::MAX; self.label.len()];
            let mut sizes: Vec<(u32, usize)> = Vec::new();
            for start in 0..self.label.len() {
                if self.label[start] == EMPTY || comp[start] != u32::MAX {
                    continue;
                }
                let id = sizes.len() as u32;
                let (mut stack, mut size) = (vec![start], 0usize);
                comp[start] = id;
                while let Some(v) = stack.pop() {
                    size += 1;
                    let nb: Vec<usize> = self.neighbours(v).collect();
                    for w in nb {
                        if comp[w] == u32::MAX && self.label[w] == self.label[start] {
                            comp[w] = id;
                            stack.push(w);
                        }
                    }
                }
                sizes.push((self.label[start], size));
            }
            let mut largest: HashMap<u32, (u32, usize)> = HashMap::new();
            for (c, &(l, n)) in sizes.iter().enumerate() {
                let e = largest.entry(l).or_insert((c as u32, n));
                if n > e.1 {
                    *e = (c as u32, n);
                }
            }
            let stray: Vec<bool> = (0..sizes.len()).map(|c| largest[&sizes[c].0].0 != c as u32).collect();
            if !stray.iter().any(|s| *s) {
                return;
            }
            let mut votes: HashMap<u32, HashMap<u32, usize>> = HashMap::new();
            for v in 0..self.label.len() {
                if self.label[v] == EMPTY || !stray[comp[v] as usize] {
                    continue;
                }
                for w in self.neighbours(v) {
                    if self.label[w] != EMPTY && comp[w] != comp[v] {
                        *votes.entry(comp[v]).or_default().entry(self.label[w]).or_default() += 1;
                    }
                }
            }
            for v in 0..self.label.len() {
                if self.label[v] == EMPTY || !stray[comp[v] as usize] {
                    continue;
                }
                // An island touching no other label leaves the body (it is dust of the cut).
                self.label[v] = votes
                    .get(&comp[v])
                    .and_then(|m| m.iter().max_by_key(|(l, n)| (**n, std::cmp::Reverse(**l))).map(|(l, _)| *l))
                    .unwrap_or(EMPTY);
            }
        }
    }

    /// The centre of the voxel labelled `which` farthest (in face steps) from every other
    /// voxel: a point deep inside the solid.
    pub fn deepest(&self, which: impl Fn(u32) -> bool) -> Option<DVec3> {
        let mut depth = vec![usize::MAX; self.label.len()];
        let mut front: Vec<usize> = (0..self.label.len()).filter(|&v| !which(self.label[v])).collect();
        front.iter().for_each(|&v| depth[v] = 0);
        let mut last = None;
        while !front.is_empty() {
            let mut next = Vec::new();
            for v in front {
                for w in self.neighbours(v) {
                    if depth[w] == usize::MAX {
                        depth[w] = depth[v] + 1;
                        next.push(w);
                        last = Some(w);
                    }
                }
            }
            front = next;
        }
        last.map(|v| self.centre(v))
    }

    /// The closed surface between the voxels labelled `which` and the rest, with vertices
    /// at the midpoints of the grid edges it crosses, oriented outward.
    pub fn surface(&self, which: impl Fn(u32) -> bool) -> (Vec<DVec3>, Vec<[u32; 3]>) {
        // Kuhn's six tetrahedra of a cube along its diagonal 0–7 (corner bits x, y, z).
        const TETS: [[usize; 4]; 6] = [[0, 1, 3, 7], [0, 1, 5, 7], [0, 2, 3, 7], [0, 2, 6, 7], [0, 4, 5, 7], [0, 4, 6, 7]];
        let d = self.dims;
        let inside = |id: usize| which(self.label[id]);
        let (mut verts, mut faces) = (Vec::new(), Vec::new());
        let mut at: HashMap<(usize, usize), u32> = HashMap::new();
        let mut vertex = |a: usize, b: usize, verts: &mut Vec<DVec3>| -> u32 {
            let key = if a < b { (a, b) } else { (b, a) };
            *at.entry(key).or_insert_with(|| {
                verts.push((self.centre(a) + self.centre(b)) * 0.5);
                (verts.len() - 1) as u32
            })
        };
        for k in 0..d[2] - 1 {
            for j in 0..d[1] - 1 {
                for i in 0..d[0] - 1 {
                    let corner: [usize; 8] = std::array::from_fn(|c| flat([i + (c & 1), j + ((c >> 1) & 1), k + (c >> 2)], d));
                    let flags: [bool; 8] = corner.map(inside);
                    if flags.iter().all(|f| *f) || flags.iter().all(|f| !*f) {
                        continue;
                    }
                    for t in TETS {
                        let ids = t.map(|c| corner[c]);
                        let (ins, outs): (Vec<usize>, Vec<usize>) = ids.iter().partition(|id| inside(**id));
                        let tris: Vec<[(usize, usize); 3]> = match ins.len() {
                            1 => vec![[(ins[0], outs[0]), (ins[0], outs[1]), (ins[0], outs[2])]],
                            3 => vec![[(ins[0], outs[0]), (ins[1], outs[0]), (ins[2], outs[0])]],
                            2 => vec![
                                [(ins[0], outs[0]), (ins[0], outs[1]), (ins[1], outs[1])],
                                [(ins[0], outs[0]), (ins[1], outs[1]), (ins[1], outs[0])],
                            ],
                            _ => continue,
                        };
                        let c_in = ins.iter().map(|v| self.centre(*v)).sum::<DVec3>() / ins.len() as f64;
                        let c_out = outs.iter().map(|v| self.centre(*v)).sum::<DVec3>() / outs.len() as f64;
                        for tri in tris {
                            let mut f = tri.map(|(a, b)| vertex(a, b, &mut verts));
                            let p = f.map(|v| verts[v as usize]);
                            if (p[1] - p[0]).cross(p[2] - p[0]).dot(c_out - c_in) < 0.0 {
                                f.swap(1, 2);
                            }
                            faces.push(f);
                        }
                    }
                }
            }
        }
        (verts, faces)
    }
}

fn centre(lo: DVec3, h: f64, c: [usize; 3]) -> DVec3 {
    lo + DVec3::new(c[0] as f64 + 0.5, c[1] as f64 + 0.5, c[2] as f64 + 0.5) * h
}

fn flat(c: [usize; 3], d: [usize; 3]) -> usize {
    (c[2] * d[1] + c[1]) * d[0] + c[0]
}

fn unflat(id: usize, d: [usize; 3]) -> [usize; 3] {
    [id % d[0], (id / d[0]) % d[1], id / (d[0] * d[1])]
}
