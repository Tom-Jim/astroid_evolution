//! The CellBVH of MS §8.1 (G2) and §10.3.2: a partition of `𝒱` into cubic
//! cells covering the volume nodes, with a point-to-cell lookup and the
//! ordered cell sequence of a ray. The ray solver walks it in §10.3.2 step (b);
//! its largest support diameter is `h_vol` of §11.2; the classification cells
//! of §9.2.2 are its cells.

use std::collections::HashMap;

use crate::common::math::DVec3;

/// Cubic cells over the volume nodes. Every node lies in exactly one cell;
/// a cell exists only when it holds at least one node.
#[derive(Clone, Debug)]
pub struct CellBvh {
    /// Cube edge length.
    pub side: f64,
    /// Corner of grid slot `(0,0,0)`: `min(node) − ε` along each axis, so no
    /// node sits on a cell boundary.
    pub origin: DVec3,
    /// Grid coordinates of each occupied cell, in dense order.
    pub coords: Vec<[i64; 3]>,
    /// Grid coordinates → dense cell index.
    index: HashMap<[i64; 3], u32>,
    /// Dense cell index → member node indices (CSR).
    offsets: Vec<u32>,
    members: Vec<u32>,
    /// Node index → dense cell index.
    cell_of_node: Vec<u32>,
}

impl CellBvh {
    /// Partition `points` into cubic cells of edge `side`.
    pub fn build(points: &[DVec3], side: f64) -> Self {
        assert!(side.is_finite() && side > 0.0, "cell edge must be positive");
        if points.is_empty() {
            return Self {
                side,
                origin: DVec3::ZERO,
                coords: Vec::new(),
                index: HashMap::new(),
                offsets: vec![0],
                members: Vec::new(),
                cell_of_node: Vec::new(),
            };
        }
        let lo = points
            .iter()
            .fold(points[0], |lo, p| lo.min(*p));
        // The nudge keeps nodes off cell walls, which would be ambiguous both
        // for `cell_of` and for the ray walk.
        let origin = lo - DVec3::splat(1.0e-9 * side);
        let mut index = HashMap::new();
        let mut coords = Vec::new();
        let mut cell_of_node = Vec::with_capacity(points.len());
        let mut members_by_cell: Vec<Vec<u32>> = Vec::new();
        for (i, p) in points.iter().enumerate() {
            let key = Self::slot_at(origin, side, *p);
            let cell = *index.entry(key).or_insert_with(|| {
                coords.push(key);
                members_by_cell.push(Vec::new());
                (coords.len() - 1) as u32
            });
            members_by_cell[cell as usize].push(i as u32);
            cell_of_node.push(cell);
        }
        let mut offsets = Vec::with_capacity(members_by_cell.len() + 1);
        offsets.push(0u32);
        let mut members = Vec::new();
        for list in &members_by_cell {
            members.extend_from_slice(list);
            offsets.push(members.len() as u32);
        }
        Self {
            side,
            origin,
            coords,
            index,
            offsets,
            members,
            cell_of_node,
        }
    }

    fn slot_at(origin: DVec3, side: f64, p: DVec3) -> [i64; 3] {
        let q = (p - origin) / side;
        [
            q.x.floor() as i64,
            q.y.floor() as i64,
            q.z.floor() as i64,
        ]
    }

    /// The grid slot `p` lies in, occupied or not.
    pub fn slot(&self, p: DVec3) -> [i64; 3] {
        Self::slot_at(self.origin, self.side, p)
    }

    /// The occupied cell containing `p`, if any.
    pub fn cell_of(&self, p: DVec3) -> Option<u32> {
        self.index.get(&self.slot(p)).copied()
    }

    /// The cell of volume node `i`.
    pub fn cell_of_node(&self, i: usize) -> u32 {
        self.cell_of_node[i]
    }

    /// Member volume nodes of cell `c`.
    pub fn members(&self, c: usize) -> &[u32] {
        &self.members[self.offsets[c] as usize..self.offsets[c + 1] as usize]
    }

    pub fn n_cells(&self) -> usize {
        self.coords.len()
    }

    /// Centre of cell `c`.
    pub fn centre(&self, c: usize) -> DVec3 {
        let k = self.coords[c];
        self.origin
            + DVec3::new(
                (k[0] as f64 + 0.5) * self.side,
                (k[1] as f64 + 0.5) * self.side,
                (k[2] as f64 + 0.5) * self.side,
            )
    }

    /// `h_vol` of §11.2: the largest support diameter among the cells — the
    /// space diagonal of a cubic cell.
    pub fn support_diameter(&self) -> f64 {
        self.side * 3.0_f64.sqrt()
    }

    /// The ordered cell sequence of the ray `x + t d` for `t ∈ [0, t_max]`:
    /// `(cell, t_enter, t_exit)` triples of occupied cells only
    /// (Amanatides–Woo traversal), as §10.3.2 step (b) consumes them.
    pub fn walk_ray(&self, o: DVec3, d: DVec3, t_max: f64) -> Vec<(u32, f64, f64)> {
        let mut out = Vec::new();
        if !(t_max > 0.0) || self.coords.is_empty() {
            return out;
        }
        let mut slot = self.slot(o);
        let q = (o - self.origin) / self.side;
        let (mut t, mut t_next) = (0.0, [0.0; 3]);
        let step = [
            if d.x > 0.0 {
                1i64
            } else if d.x < 0.0 {
                -1
            } else {
                0
            },
            if d.y > 0.0 {
                1i64
            } else if d.y < 0.0 {
                -1
            } else {
                0
            },
            if d.z > 0.0 {
                1i64
            } else if d.z < 0.0 {
                -1
            } else {
                0
            },
        ];
        let dd = [d.x, d.y, d.z];
        for a in 0..3 {
            if step[a] == 0 {
                t_next[a] = f64::INFINITY;
                continue;
            }
            let wall = if step[a] > 0 {
                slot[a] as f64 + 1.0
            } else {
                slot[a] as f64
            };
            t_next[a] = (wall - q[a]) * self.side / dd[a];
        }
        let t_delta = [
            if step[0] != 0 {
                self.side / d.x.abs()
            } else {
                f64::INFINITY
            },
            if step[1] != 0 {
                self.side / d.y.abs()
            } else {
                f64::INFINITY
            },
            if step[2] != 0 {
                self.side / d.z.abs()
            } else {
                f64::INFINITY
            },
        ];
        while t <= t_max {
            if let Some(&cell) = self.index.get(&slot) {
                out.push((cell, t, t_next[0].min(t_next[1]).min(t_next[2]).min(t_max)));
            }
            let axis = if t_next[0] <= t_next[1] && t_next[0] <= t_next[2] {
                0
            } else if t_next[1] <= t_next[2] {
                1
            } else {
                2
            };
            t = t_next[axis];
            slot[axis] += step[axis];
            t_next[axis] += t_delta[axis];
            if step[0] == 0 && step[1] == 0 && step[2] == 0 {
                break;
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cells_cover_every_node_and_report_ordered() {
        let pts: Vec<DVec3> = (0..200)
            .map(|i| DVec3::new(i as f64 * 0.37, (i as f64 * 0.11) % 1.3, 0.4))
            .collect();
        let bvh = CellBvh::build(&pts, 0.5);
        assert!(!bvh.index.is_empty());
        for (i, p) in pts.iter().enumerate() {
            let c = bvh.cell_of(*p).expect("every node has a cell");
            assert_eq!(c, bvh.cell_of_node(i));
            assert!(bvh.members(c as usize).contains(&(i as u32)));
        }
        assert!((bvh.support_diameter() - 0.5 * 3f64.sqrt()).abs() < 1e-15);
    }

    #[test]
    fn walk_ray_returns_occupied_cells_in_order() {
        // Nodes on a diagonal; the ray along +x sees their cells in order.
        let pts: Vec<DVec3> = (0..8)
            .map(|i| DVec3::new(0.25 + i as f64 * 0.5, 0.25, 0.25))
            .collect();
        let bvh = CellBvh::build(&pts, 0.5);
        let seq = bvh.walk_ray(DVec3::new(0.0, 0.25, 0.25), DVec3::new(1.0, 0.0, 0.0), 10.0);
        let cells: Vec<u32> = seq.iter().map(|e| e.0).collect();
        for w in cells.windows(2) {
            assert!(bvh.coords[w[0] as usize][0] < bvh.coords[w[1] as usize][0]);
        }
        assert_eq!(cells.len(), bvh.n_cells().min(9));
        // Intervals are ordered and contiguous on the ray.
        for w in seq.windows(2) {
            assert!(w[1].1 >= w[0].2 - 1e-12);
        }
    }
}
