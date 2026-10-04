//! Reference `ℓ = 1` Jacobi mass blocks. The monomial Gram matrices of `ζ²…ζ⁵`
//! are not an assembly basis.

use crate::scf_simulator::basis::jacobi::{poloidal_mass, toroidal_mass, weighted_orthogonality};

/// Weighted `L²` Gram matrix `∫ ψ_1j ψ_1k ζ² dζ`, diagonal by orthogonality.
pub fn assemble_mass_matrix() -> [[f64; 4]; 4] {
    let mut m = [[0.0; 4]; 4];
    for k in 0..4 {
        m[k][k] = weighted_orthogonality(1, k, k);
    }
    m
}

pub fn assemble_mass_matrix_inverse() -> [[f64; 4]; 4] {
    let mut inv = [[0.0; 4]; 4];
    for k in 0..4 {
        inv[k][k] = 1.0 / weighted_orthogonality(1, k, k);
    }
    inv
}

/// Poloidal reference mass at unit density and radius.
pub fn assemble_poloidal_mass(l: usize, density: f64, radius: f64) -> [[f64; 4]; 4] {
    poloidal_mass(l, density, radius)
}

/// Toroidal reference mass, stored as a diagonal matrix.
pub fn assemble_toroidal_mass(l: usize, density: f64, radius: f64) -> [[f64; 4]; 4] {
    let diag = toroidal_mass(l, density, radius);
    let mut m = [[0.0; 4]; 4];
    for k in 0..4 {
        m[k][k] = diag[k];
    }
    m
}

pub fn assemble_k8_mass_matrix() -> [[f64; 8]; 8] {
    let block = assemble_mass_matrix();
    let mut m = [[0.0; 8]; 8];
    for i in 0..4 {
        for j in 0..4 {
            m[i][j] = block[i][j];
            m[i + 4][j + 4] = block[i][j];
        }
    }
    m
}

pub fn invert_k8_mass_matrix() -> Result<[[f64; 8]; 8], String> {
    let m = assemble_k8_mass_matrix();
    let mat = nalgebra::DMatrix::from_fn(8, 8, |i, j| m[i][j]);
    let inv = mat
        .try_inverse()
        .ok_or_else(|| "K=8 Jacobi mass matrix is singular".to_string())?;
    let mut out = [[0.0; 8]; 8];
    for i in 0..8 {
        for j in 0..8 {
            out[i][j] = inv[(i, j)];
        }
    }
    Ok(out)
}

pub fn invert_mass_matrix(m: &[[f64; 4]; 4]) -> Result<[[f64; 4]; 4], String> {
    let mat = nalgebra::Matrix4::new(
        m[0][0], m[0][1], m[0][2], m[0][3], m[1][0], m[1][1], m[1][2], m[1][3], m[2][0], m[2][1],
        m[2][2], m[2][3], m[3][0], m[3][1], m[3][2], m[3][3],
    );
    let inv = mat
        .try_inverse()
        .ok_or_else(|| "质量矩阵 M 不可逆".to_string())?;
    let mut res = [[0.0; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            res[i][j] = inv[(i, j)];
        }
    }
    Ok(res)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jacobi_mass_is_diagonal_and_inverts() {
        let m = assemble_mass_matrix();
        assert!((m[0][0] - 1.0 / 5.0).abs() < 1e-12);
        assert!((m[1][1] - 1.0 / 7.0).abs() < 1e-12);
        assert!((m[2][2] - 1.0 / 9.0).abs() < 1e-12);
        assert!((m[3][3] - 1.0 / 11.0).abs() < 1e-12);
        assert!(m[0][1].abs() < 1e-15);
        let inv = invert_mass_matrix(&m).unwrap();
        let closed = assemble_mass_matrix_inverse();
        for i in 0..4 {
            for j in 0..4 {
                assert!((inv[i][j] - closed[i][j]).abs() < 1e-10);
            }
        }
        let toroidal = assemble_toroidal_mass(1, 1.0, 1.0);
        assert!((toroidal[0][0] - 2.0 / 5.0).abs() < 1e-12);
        let poloidal = assemble_poloidal_mass(1, 1.0, 1.0);
        assert!(poloidal[0][0] > 0.0);
        assert!((poloidal[0][1] - poloidal[1][0]).abs() < 1e-8);
    }
}
