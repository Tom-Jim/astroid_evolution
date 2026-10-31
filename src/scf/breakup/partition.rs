//! The piecewise-rigid fit of a collapse mechanism (§12.1): blocks that each move as one
//! rigid body, found by alternating the assignment of every node to the rigid motion that
//! fits it best and the least-squares fit of each block, with as many blocks as each one
//! more lowers the misfit by a fixed fraction.

use nalgebra::{Matrix6, Vector6};

use crate::common::math::DVec3;
use crate::scf::surface::PointGrid;

/// A rigid motion `v(x) = a + θ × (x − c)`.
#[derive(Clone, Copy, Debug)]
pub struct Rigid {
    pub a: DVec3,
    pub theta: DVec3,
    pub c: DVec3,
}

impl Rigid {
    pub fn at(&self, x: DVec3) -> DVec3 {
        self.a + self.theta.cross(x - self.c)
    }

    /// The weighted least-squares rigid motion of the points `idx` (`None` when they do not
    /// determine one).
    pub fn fit(x: &[DVec3], v: &[DVec3], m: &[f64], idx: &[usize]) -> Option<Self> {
        let mass: f64 = idx.iter().map(|&i| m[i]).sum();
        if !(mass > 0.0) {
            return None;
        }
        let c = idx.iter().map(|&i| x[i] * m[i]).sum::<DVec3>() / mass;
        // v ≈ a − [r]× θ with r = x − c: the rows (I | −[r]×).
        let (mut a, mut b) = (Matrix6::<f64>::zeros(), Vector6::<f64>::zeros());
        for &i in idx {
            let r = x[i] - c;
            let rows: [[f64; 6]; 3] = [
                [1.0, 0.0, 0.0, 0.0, r.z, -r.y],
                [0.0, 1.0, 0.0, -r.z, 0.0, r.x],
                [0.0, 0.0, 1.0, r.y, -r.x, 0.0],
            ];
            for (row, vi) in rows.iter().zip(v[i].to_array()) {
                for p in 0..6 {
                    b[p] += m[i] * row[p] * vi;
                    for q in 0..6 {
                        a[(p, q)] += m[i] * row[p] * row[q];
                    }
                }
            }
        }
        // A collinear set leaves the rotation about its line free: the smallest-norm fit.
        let s = a.svd(true, true).solve(&b, 1e-12 * a.norm()).ok()?;
        Some(Self { a: DVec3::new(s[0], s[1], s[2]), theta: DVec3::new(s[3], s[4], s[5]), c })
    }
}

/// Neighbours of every node among its `k` nearest: the graph that blocks must be connected on.
pub fn graph(x: &[DVec3], k: usize) -> Vec<Vec<usize>> {
    let grid = PointGrid::build(x);
    let mut nb: Vec<Vec<usize>> = x.iter().enumerate().map(|(i, p)| grid.k_nearest(*p, k, |j| j == i).into_iter().map(|(_, j)| j).collect()).collect();
    // Symmetric: an edge either way.
    for i in 0..nb.len() {
        for j in nb[i].clone() {
            if !nb[j].contains(&i) {
                nb[j].push(i);
            }
        }
    }
    nb
}

/// Options of [`partition`].
#[derive(Clone, Copy, Debug)]
pub struct PartitionOptions {
    pub max_blocks: usize,
    /// Fraction by which one more block must lower the misfit.
    pub gain: f64,
    /// Smallest block mass, as a fraction of the total.
    pub min_fraction: f64,
}

/// Block labels of the nodes `x` with mechanism velocities `v` and masses `m`, and the misfit
/// `Σ m |v − v_block|² / Σ m |v − v̄|²` of each accepted number of blocks.
pub fn partition(x: &[DVec3], v: &[DVec3], m: &[f64], nb: &[Vec<usize>], opt: &PartitionOptions) -> (Vec<usize>, Vec<f64>) {
    let n = x.len();
    let total: f64 = m.iter().sum();
    let mean = (0..n).map(|i| v[i] * m[i]).sum::<DVec3>() / total;
    let scale = (0..n).map(|i| m[i] * (v[i] - mean).length_squared()).sum::<f64>().max(1e-300);
    let mut labels = vec![0usize; n];
    let mut misfit = vec![fit_blocks(x, v, m, &mut labels, nb, 1, opt).map_or(1.0, |r| r / scale)];
    for k in 2..=opt.max_blocks {
        // The next block starts from the rigid motion around the worst-fitted node.
        let motions = motions(x, v, m, &labels, k - 1);
        let worst = (0..n)
            .max_by(|&a, &b| {
                let r = |i: usize| m[i] * (v[i] - motions[labels[i]].map_or(v[i], |q| q.at(x[i]))).length_squared();
                r(a).total_cmp(&r(b))
            })
            .unwrap_or(0);
        let mut trial = labels.clone();
        let near = PointGrid::build(x).k_nearest(x[worst], (n / (2 * k)).max(8), |_| false);
        for (_, i) in near {
            trial[i] = k - 1;
        }
        match fit_blocks(x, v, m, &mut trial, nb, k, opt) {
            Some(r) if r / scale <= (1.0 - opt.gain) * misfit[misfit.len() - 1] => {
                labels = trial;
                misfit.push(r / scale);
            }
            _ => break,
        }
    }
    (labels, misfit)
}

fn motions(x: &[DVec3], v: &[DVec3], m: &[f64], labels: &[usize], k: usize) -> Vec<Option<Rigid>> {
    (0..k)
        .map(|b| {
            let idx: Vec<usize> = (0..x.len()).filter(|&i| labels[i] == b).collect();
            Rigid::fit(x, v, m, &idx)
        })
        .collect()
}

/// Lloyd's iteration with `k` blocks from `labels`, each block kept connected on `nb` and
/// above the smallest mass; `None` when a block empties or stays too small.
fn fit_blocks(x: &[DVec3], v: &[DVec3], m: &[f64], labels: &mut [usize], nb: &[Vec<usize>], k: usize, opt: &PartitionOptions) -> Option<f64> {
    let n = x.len();
    let total: f64 = m.iter().sum();
    for _ in 0..100 {
        let fits = motions(x, v, m, labels, k);
        if fits.iter().any(|f| f.is_none()) {
            return None;
        }
        let fits: Vec<Rigid> = fits.into_iter().flatten().collect();
        let mut next: Vec<usize> = (0..n)
            .map(|i| (0..k).min_by(|&a, &b| (v[i] - fits[a].at(x[i])).length_squared().total_cmp(&(v[i] - fits[b].at(x[i])).length_squared())).unwrap_or(0))
            .collect();
        connect(&mut next, nb, k);
        let changed = next.iter().zip(labels.iter()).any(|(a, b)| a != b);
        labels.copy_from_slice(&next);
        if !changed {
            break;
        }
    }
    let mass: Vec<f64> = (0..k).map(|b| (0..n).filter(|&i| labels[i] == b).map(|i| m[i]).sum()).collect();
    if mass.iter().any(|w| *w < opt.min_fraction * total) {
        return None;
    }
    let fits: Vec<Rigid> = motions(x, v, m, labels, k).into_iter().collect::<Option<_>>()?;
    Some((0..n).map(|i| m[i] * (v[i] - fits[labels[i]].at(x[i])).length_squared()).sum())
}

/// Every block reduced to its largest connected component on `nb`; the other components
/// join the block they touch most.
fn connect(labels: &mut [usize], nb: &[Vec<usize>], k: usize) {
    let n = labels.len();
    let mut comp = vec![usize::MAX; n];
    let mut size = Vec::new();
    for s in 0..n {
        if comp[s] != usize::MAX {
            continue;
        }
        let id = size.len();
        let (mut stack, mut count) = (vec![s], 0usize);
        comp[s] = id;
        while let Some(p) = stack.pop() {
            count += 1;
            for &q in &nb[p] {
                if comp[q] == usize::MAX && labels[q] == labels[s] {
                    comp[q] = id;
                    stack.push(q);
                }
            }
        }
        size.push((labels[s], count));
    }
    let mut keep = vec![usize::MAX; k];
    for (c, &(l, s)) in size.iter().enumerate() {
        if keep[l] == usize::MAX || s > size[keep[l]].1 {
            keep[l] = c;
        }
    }
    for p in 0..n {
        if keep[labels[p]] == comp[p] {
            continue;
        }
        let mut votes = vec![0usize; k];
        for &q in &nb[p] {
            if keep[labels[q]] == comp[q] {
                votes[labels[q]] += 1;
            }
        }
        if let Some((best, _)) = votes.iter().enumerate().filter(|(_, c)| **c > 0).max_by_key(|(_, c)| **c) {
            labels[p] = best;
        }
    }
}
