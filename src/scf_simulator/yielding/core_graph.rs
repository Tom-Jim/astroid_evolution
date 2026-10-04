use disjoint_sets::UnionFind;

use crate::scf_simulator::error::ScfError;

/// Largest-mass connected component of the unyielded graph is the rigid core.
pub fn rigid_core(
    yielded: &[bool],
    neighbors: &[Vec<usize>],
    mass: &[f64],
) -> Result<Vec<bool>, ScfError> {
    let n = yielded.len();
    if !yielded.iter().any(|flag| !*flag) {
        return Err(ScfError::no_core("unyielded set is empty"));
    }
    let mut sets = UnionFind::new(n);
    for i in 0..n {
        if yielded[i] {
            continue;
        }
        for &j in &neighbors[i] {
            if j < n && !yielded[j] {
                sets.union(i, j);
            }
        }
    }
    let mut roots = Vec::new();
    let mut totals = Vec::new();
    for i in 0..n {
        if yielded[i] {
            continue;
        }
        let root = sets.find(i);
        if let Some(slot) = roots.iter().position(|r| *r == root) {
            totals[slot] += mass[i];
        } else {
            roots.push(root);
            totals.push(mass[i]);
        }
    }
    let best = totals
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| roots[i])
        .ok_or_else(|| ScfError::no_core("unyielded set is empty"))?;
    Ok((0..n)
        .map(|i| !yielded[i] && sets.find(i) == best)
        .collect())
}
