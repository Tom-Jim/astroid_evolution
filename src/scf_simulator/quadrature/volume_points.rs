use std::num::NonZeroUsize;

use gauss_quad::{FiniteAboveNegOneF64, GaussJacobi};

use crate::rtfp_solver::radial::gauss_legendre::lebedev_directions;

#[derive(Clone, Copy, Debug)]
pub struct VolumeNode {
    pub position: [f64; 3],
    pub weight: f64,
}

/// Angular Lebedev nodes times radial Gauss–Jacobi nodes on ζ ∈ (0, 1).
pub fn volume_points(directions: usize, radial_degree: usize) -> Vec<VolumeNode> {
    let dirs = lebedev_directions(directions);
    let degree = NonZeroUsize::new(radial_degree.max(2)).expect("degree");
    let zero = FiniteAboveNegOneF64::new(0.0).expect("jacobi exponent");
    let rule = GaussJacobi::new(degree, zero, zero);
    let mut nodes = Vec::with_capacity(dirs.len() * rule.degree());
    for &(direction, angular_weight) in &dirs {
        for (zeta, radial_weight) in rule.as_node_weight_pairs() {
            let s = 0.5 * (zeta + 1.0);
            let r2 = s * s;
            nodes.push(VolumeNode {
                position: [direction[0] * s, direction[1] * s, direction[2] * s],
                weight: angular_weight * radial_weight * 0.5 * r2,
            });
        }
    }
    nodes
}
