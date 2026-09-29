//! K=4 径向 Galerkin 结构矩阵装配 (M, K1, K2)
//! 严格依照 docs/software.md 第 10.1 节 与 docs/mathstrict.md 第 10.1 节
//!
//! 1. 质量矩阵 M_ij = 1 / (i + j + 5)
//! 2. 一阶导数刚度矩阵 K^(1)_ij = (i + 1)(j + 1) / (i + j + 3)
//! 3. 二阶导数双调和刚度矩阵 K^(2)_ij = i(i + 1)j(j + 1) / (i + j + 1)
//! 其中 i, j in {1, 2, 3, 4} (1-based index)

use num_rational::Ratio;

/// 装配精确有理数形式的质量矩阵 M
pub fn assemble_mass_matrix_rational() -> [[Ratio<i64>; 4]; 4] {
    let mut m = [[Ratio::new(0, 1); 4]; 4];
    for i in 1..=4 {
        for j in 1..=4 {
            m[i - 1][j - 1] = Ratio::new(1, (i + j + 5) as i64);
        }
    }
    m
}

/// 装配精确有理数形式的一阶导数刚度矩阵 K^(1)
pub fn assemble_first_stiffness_rational() -> [[Ratio<i64>; 4]; 4] {
    let mut k1 = [[Ratio::new(0, 1); 4]; 4];
    for i in 1..=4 {
        for j in 1..=4 {
            let num = ((i + 1) * (j + 1)) as i64;
            let den = (i + j + 3) as i64;
            k1[i - 1][j - 1] = Ratio::new(num, den);
        }
    }
    k1
}

/// 装配精确有理数形式的二阶导数刚度矩阵 K^(2)
pub fn assemble_second_stiffness_rational() -> [[Ratio<i64>; 4]; 4] {
    let mut k2 = [[Ratio::new(0, 1); 4]; 4];
    for i in 1..=4 {
        for j in 1..=4 {
            let num = (i * (i + 1) * j * (j + 1)) as i64;
            let den = (i + j + 1) as i64;
            k2[i - 1][j - 1] = Ratio::new(num, den);
        }
    }
    k2
}

/// 装配浮点数形式质量矩阵 M (4x4)
pub fn assemble_mass_matrix() -> [[f64; 4]; 4] {
    let r = assemble_mass_matrix_rational();
    let mut m = [[0.0; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            m[i][j] = *r[i][j].numer() as f64 / *r[i][j].denom() as f64;
        }
    }
    m
}

/// 装配浮点数形式一阶导数刚度矩阵 K^(1) (4x4)
pub fn assemble_first_stiffness() -> [[f64; 4]; 4] {
    let r = assemble_first_stiffness_rational();
    let mut k1 = [[0.0; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            k1[i][j] = *r[i][j].numer() as f64 / *r[i][j].denom() as f64;
        }
    }
    k1
}

/// 装配浮点数形式二阶导数刚度矩阵 K^(2) (4x4)
pub fn assemble_second_stiffness() -> [[f64; 4]; 4] {
    let r = assemble_second_stiffness_rational();
    let mut k2 = [[0.0; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            k2[i][j] = *r[i][j].numer() as f64 / *r[i][j].denom() as f64;
        }
    }
    k2
}

/// 装配精确有理数形式的质量矩阵逆矩阵 M^(-1)
///
/// 由于 M_ij = 1 / (i + j + 5) 为广义 Hilbert 矩阵，其解析逆矩阵具有完全精确的整数系数，
/// 直接使用解析逆可以彻底消除数值求逆在条件数 ~10^7 下造成的舍入精度损失。
pub fn assemble_mass_matrix_inverse_rational() -> [[Ratio<i64>; 4]; 4] {
    let inv_ints: [[i64; 4]; 4] = [
        [100800, -415800, 554400, -240240],
        [-415800, 1742400, -2352240, 1029600],
        [554400, -2352240, 3207600, -1415700],
        [-240240, 1029600, -1415700, 629200],
    ];
    let mut res = [[Ratio::new(0, 1); 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            res[i][j] = Ratio::new(inv_ints[i][j], 1);
        }
    }
    res
}

/// 装配浮点数形式质量矩阵逆矩阵 M^(-1) (4x4)
pub fn assemble_mass_matrix_inverse() -> [[f64; 4]; 4] {
    [
        [100800.0, -415800.0, 554400.0, -240240.0],
        [-415800.0, 1742400.0, -2352240.0, 1029600.0],
        [554400.0, -2352240.0, 3207600.0, -1415700.0],
        [-240240.0, 1029600.0, -1415700.0, 629200.0],
    ]
}

pub fn assemble_k8_mass_matrix() -> [[f64; 8]; 8] {
    let mut m=[[0.0;8];8]; for i in 1..=8 { for j in 1..=8 { m[i-1][j-1]=1.0/(i+j+5) as f64; }} m
}

pub fn assemble_k8_first_stiffness() -> [[f64; 8]; 8] {
    let mut k=[[0.0;8];8]; for i in 1..=8 { for j in 1..=8 { k[i-1][j-1]=((i+1)*(j+1)) as f64/(i+j+3) as f64; }} k
}

pub fn assemble_k8_second_stiffness() -> [[f64; 8]; 8] {
    let mut k=[[0.0;8];8]; for i in 1..=8 { for j in 1..=8 { k[i-1][j-1]=(i*(i+1)*j*(j+1)) as f64/(i+j+1) as f64; }} k
}

pub fn invert_k8_mass_matrix() -> Result<[[f64; 8]; 8], String> {
    let m=assemble_k8_mass_matrix(); let flat: Vec<f64>=m.iter().flat_map(|r| r.iter().copied()).collect();
    let inv=nalgebra::DMatrix::from_row_slice(8,8,&flat).try_inverse().ok_or_else(|| "K=8 质量矩阵不可逆".to_string())?;
    let mut out=[[0.0;8];8]; for i in 0..8 { for j in 0..8 { out[i][j]=inv[(i,j)]; }} Ok(out)
}

/// 计算质量矩阵 M 的精确逆矩阵 M^(-1)
///
/// 若为标准常数矩阵 M，直接返回零误差解析逆；对于任意输入则进行数值求逆。
pub fn invert_mass_matrix(m: &[[f64; 4]; 4]) -> Result<[[f64; 4]; 4], String> {
    let std_m = assemble_mass_matrix();
    let mut diff = 0.0;
    for i in 0..4 {
        for j in 0..4 {
            diff += (m[i][j] - std_m[i][j]).abs();
        }
    }
    if diff < 1e-12 {
        return Ok(assemble_mass_matrix_inverse());
    }

    let mat = nalgebra::Matrix4::new(
        m[0][0], m[0][1], m[0][2], m[0][3],
        m[1][0], m[1][1], m[1][2], m[1][3],
        m[2][0], m[2][1], m[2][2], m[2][3],
        m[3][0], m[3][1], m[3][2], m[3][3],
    );

    let inv = mat.try_inverse().ok_or_else(|| "质量矩阵 M 不可逆".to_string())?;

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
    fn test_rational_matrices_and_symmetry() {
        let m = assemble_mass_matrix_rational();
        let k1 = assemble_first_stiffness_rational();
        let k2 = assemble_second_stiffness_rational();

        // 验证对称性
        for i in 0..4 {
            for j in 0..4 {
                assert_eq!(m[i][j], m[j][i]);
                assert_eq!(k1[i][j], k1[j][i]);
                assert_eq!(k2[i][j], k2[j][i]);
            }
        }

        // 检验文档表格值 (docs/mathstrict.md 第 10.1 节):
        // M:
        assert_eq!(m[0][0], Ratio::new(1, 7));
        assert_eq!(m[0][3], Ratio::new(1, 10));
        assert_eq!(m[3][3], Ratio::new(1, 13));

        // K1:
        assert_eq!(k1[0][0], Ratio::new(4, 5));
        assert_eq!(k1[0][1], Ratio::new(1, 1));
        assert_eq!(k1[3][3], Ratio::new(25, 11));

        // K2:
        assert_eq!(k2[0][0], Ratio::new(4, 3));
        assert_eq!(k2[0][1], Ratio::new(3, 1));
        assert_eq!(k2[3][3], Ratio::new(400, 9));
    }

    #[test]
    fn test_mass_matrix_inversion() {
        // 1. 有理数严格零误差验证
        let m_rat = assemble_mass_matrix_rational();
        let inv_rat = assemble_mass_matrix_inverse_rational();
        for i in 0..4 {
            for j in 0..4 {
                let mut sum = Ratio::new(0, 1);
                for k in 0..4 {
                    sum = sum + m_rat[i][k] * inv_rat[k][j];
                }
                let expected = if i == j { Ratio::new(1, 1) } else { Ratio::new(0, 1) };
                assert_eq!(
                    sum, expected,
                    "有理数下 M * M^(-1) 不是精确单位阵: ({i}, {j}) = {sum}, expected {expected}"
                );
            }
        }

        // 2. 浮点数形式验证
        let m = assemble_mass_matrix();
        let inv = invert_mass_matrix(&m).unwrap();

        for i in 0..4 {
            for j in 0..4 {
                let mut sum = 0.0;
                for k in 0..4 {
                    sum += m[i][k] * inv[k][j];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (sum - expected).abs() < 1e-10,
                    "M * M^(-1) 不是单位阵: ({i}, {j}) = {sum}, expected {expected}"
                );
            }
        }
    }
}
