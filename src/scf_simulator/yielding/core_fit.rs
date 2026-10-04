use nalgebra::{DMatrix, DVector, Matrix3};

use crate::scf_simulator::error::ScfError;

#[derive(Clone, Copy, Debug)]
pub struct CoreMotion {
    pub translation: [f64; 3],
    pub rotation: [f64; 3],
}

/// Least-squares rigid motion of the core. A collinear core is `CoreDegenerate`
/// and is returned before any factorization can panic.
pub fn fit_core_motion(points: &[[f64; 3]], velocity: &[[f64; 3]]) -> Result<CoreMotion, ScfError> {
    let mut normal = DMatrix::<f64>::zeros(6, 6);
    let mut rhs = DVector::<f64>::zeros(6);
    for (x, v) in points.iter().zip(velocity) {
        let rows = [
            [1.0, 0.0, 0.0, 0.0, x[2], -x[1]],
            [0.0, 1.0, 0.0, -x[2], 0.0, x[0]],
            [0.0, 0.0, 1.0, x[1], -x[0], 0.0],
        ];
        for (row, component) in rows.iter().zip(v) {
            for a in 0..6 {
                rhs[a] += row[a] * component;
                for b in 0..6 {
                    normal[(a, b)] += row[a] * row[b];
                }
            }
        }
    }
    if core_is_collinear(points) {
        return Err(ScfError::core_degenerate("rigid core points are collinear"));
    }
    let solved = match normal.cholesky() {
        Some(factor) => factor.solve(&rhs),
        None => {
            return Err(ScfError::core_degenerate(
                "rigid core normal matrix is singular",
            ));
        }
    };
    if solved.iter().any(|value| !value.is_finite()) {
        return Err(ScfError::core_degenerate("rigid core motion is not finite"));
    }
    Ok(CoreMotion {
        translation: [solved[0], solved[1], solved[2]],
        rotation: [solved[3], solved[4], solved[5]],
    })
}

fn core_is_collinear(points: &[[f64; 3]]) -> bool {
    if points.len() < 3 {
        return true;
    }
    let mut mean = [0.0; 3];
    for point in points {
        for axis in 0..3 {
            mean[axis] += point[axis];
        }
    }
    let scale = points.len() as f64;
    for value in &mut mean {
        *value /= scale;
    }
    let mut covariance = Matrix3::<f64>::zeros();
    for point in points {
        for i in 0..3 {
            for j in 0..3 {
                covariance[(i, j)] += (point[i] - mean[i]) * (point[j] - mean[j]);
            }
        }
    }
    let mut eigenvalues = covariance.symmetric_eigen().eigenvalues.as_slice().to_vec();
    eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let largest = eigenvalues[2].abs().max(1e-30);
    eigenvalues[0].abs() <= 1e-8 * largest && eigenvalues[1].abs() <= 1e-8 * largest
}

pub fn slip_velocity(point: [f64; 3], velocity: [f64; 3], motion: CoreMotion) -> [f64; 3] {
    let w = motion.rotation;
    let cross = [
        w[1] * point[2] - w[2] * point[1],
        w[2] * point[0] - w[0] * point[2],
        w[0] * point[1] - w[1] * point[0],
    ];
    [
        velocity[0] - motion.translation[0] - cross[0],
        velocity[1] - motion.translation[1] - cross[1],
        velocity[2] - motion.translation[2] - cross[2],
    ]
}
