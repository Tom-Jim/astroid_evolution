/// Exact rational used for the K=4 Galerkin mass and stiffness entries.
#[derive(Clone, Copy, Debug, Eq)]
pub(crate) struct Exact {
    n: i128,
    d: i128,
}

fn gcd(mut a: i128, mut b: i128) -> i128 {
    a = a.abs();
    b = b.abs();
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a.max(1)
}

impl Exact {
    fn new(n: impl Into<i128>, d: impl Into<i128>) -> Self {
        let n = n.into();
        let d = d.into();
        if d < 0 {
            return Self::new(-n, -d);
        }
        let g = gcd(n, d);
        Self { n: n / g, d: d / g }
    }

    fn numer(&self) -> &i128 {
        &self.n
    }

    fn denom(&self) -> &i128 {
        &self.d
    }
}

impl std::fmt::Display for Exact {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}/{}", self.n, self.d)
    }
}

impl PartialEq for Exact {
    fn eq(&self, other: &Self) -> bool {
        self.n * other.d == other.n * self.d
    }
}

impl std::ops::Add for Exact {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::new(self.n * rhs.d + rhs.n * self.d, self.d * rhs.d)
    }
}

impl std::ops::Sub for Exact {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.n * rhs.d - rhs.n * self.d, self.d * rhs.d)
    }
}

impl std::ops::Mul for Exact {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self::new(self.n * rhs.n, self.d * rhs.d)
    }
}

impl std::ops::Div for Exact {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        Self::new(self.n * rhs.d, self.d * rhs.n)
    }
}

impl std::ops::DivAssign for Exact {
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl std::ops::SubAssign for Exact {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

pub(crate) fn assemble_mass_matrix_rational() -> [[Exact; 4]; 4] {
    let mut m = [[Exact::new(0, 1); 4]; 4];
    for i in 1..=4 {
        for j in 1..=4 {
            m[i - 1][j - 1] = Exact::new(1, (i + j + 5) as i64);
        }
    }
    m
}
pub(crate) fn assemble_first_stiffness_rational() -> [[Exact; 4]; 4] {
    let mut k1 = [[Exact::new(0, 1); 4]; 4];
    for i in 1..=4 {
        for j in 1..=4 {
            let num = ((i + 1) * (j + 1)) as i64;
            let den = (i + j + 3) as i64;
            k1[i - 1][j - 1] = Exact::new(num, den);
        }
    }
    k1
}
pub(crate) fn assemble_second_stiffness_rational() -> [[Exact; 4]; 4] {
    let mut k2 = [[Exact::new(0, 1); 4]; 4];
    for i in 1..=4 {
        for j in 1..=4 {
            let num = (i * (i + 1) * j * (j + 1)) as i64;
            let den = (i + j + 1) as i64;
            k2[i - 1][j - 1] = Exact::new(num, den);
        }
    }
    k2
}
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
pub(crate) fn assemble_mass_matrix_inverse_rational() -> [[Exact; 4]; 4] {
    let inv_ints: [[i64; 4]; 4] = [
        [100800, -415800, 554400, -240240],
        [-415800, 1742400, -2352240, 1029600],
        [554400, -2352240, 3207600, -1415700],
        [-240240, 1029600, -1415700, 629200],
    ];
    let mut res = [[Exact::new(0, 1); 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            res[i][j] = Exact::new(inv_ints[i][j], 1);
        }
    }
    res
}
pub fn assemble_mass_matrix_inverse() -> [[f64; 4]; 4] {
    [
        [100800.0, -415800.0, 554400.0, -240240.0],
        [-415800.0, 1742400.0, -2352240.0, 1029600.0],
        [554400.0, -2352240.0, 3207600.0, -1415700.0],
        [-240240.0, 1029600.0, -1415700.0, 629200.0],
    ]
}
pub fn assemble_k8_mass_matrix() -> [[f64; 8]; 8] {
    let mut m = [[0.0; 8]; 8];
    for i in 1..=8 {
        for j in 1..=8 {
            m[i - 1][j - 1] = 1.0 / (i + j + 5) as f64;
        }
    }
    m
}
pub fn assemble_k8_first_stiffness() -> [[f64; 8]; 8] {
    let mut k = [[0.0; 8]; 8];
    for i in 1..=8 {
        for j in 1..=8 {
            k[i - 1][j - 1] = ((i + 1) * (j + 1)) as f64 / (i + j + 3) as f64;
        }
    }
    k
}
pub fn assemble_k8_second_stiffness() -> [[f64; 8]; 8] {
    let mut k = [[0.0; 8]; 8];
    for i in 1..=8 {
        for j in 1..=8 {
            k[i - 1][j - 1] = (i * (i + 1) * j * (j + 1)) as f64 / (i + j + 1) as f64;
        }
    }
    k
}
pub fn invert_k8_mass_matrix() -> Result<[[f64; 8]; 8], String> {
    let mut aug = [[Exact::new(0, 1); 16]; 8];
    for i in 0..8 {
        for j in 0..8 {
            aug[i][j] = Exact::new(1, (i + j + 7) as i128);
            aug[i][8 + j] = if i == j {
                Exact::new(1, 1)
            } else {
                Exact::new(0, 1)
            };
        }
    }
    for col in 0..8 {
        let pivot = (col..8)
            .find(|r| *aug[*r][col].numer() != 0)
            .ok_or_else(|| "K=8 质量矩阵不可逆".to_string())?;
        if pivot != col {
            aug.swap(pivot, col);
        }
        let p = aug[col][col];
        for j in 0..16 {
            aug[col][j] /= p;
        }
        for r in 0..8 {
            if r != col {
                let f = aug[r][col];
                if *f.numer() != 0 {
                    for j in 0..16 {
                        let v = aug[col][j];
                        aug[r][j] -= f * v;
                    }
                }
            }
        }
    }
    let mut out = [[0.0; 8]; 8];
    for i in 0..8 {
        for j in 0..8 {
            out[i][j] = *aug[i][8 + j].numer() as f64 / *aug[i][8 + j].denom() as f64;
        }
    }
    Ok(out)
}
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
    fn test_rational_matrices_and_symmetry() {
        let m = assemble_mass_matrix_rational();
        let k1 = assemble_first_stiffness_rational();
        let k2 = assemble_second_stiffness_rational();
        for i in 0..4 {
            for j in 0..4 {
                assert_eq!(m[i][j], m[j][i]);
                assert_eq!(k1[i][j], k1[j][i]);
                assert_eq!(k2[i][j], k2[j][i]);
            }
        }
        assert_eq!(m[0][0], Exact::new(1, 7));
        assert_eq!(m[0][3], Exact::new(1, 10));
        assert_eq!(m[3][3], Exact::new(1, 13));
        assert_eq!(k1[0][0], Exact::new(4, 5));
        assert_eq!(k1[0][1], Exact::new(1, 1));
        assert_eq!(k1[3][3], Exact::new(25, 11));
        assert_eq!(k2[0][0], Exact::new(4, 3));
        assert_eq!(k2[0][1], Exact::new(3, 1));
        assert_eq!(k2[3][3], Exact::new(400, 9));
    }
    #[test]
    fn test_mass_matrix_inversion() {
        let m_rat = assemble_mass_matrix_rational();
        let inv_rat = assemble_mass_matrix_inverse_rational();
        for i in 0..4 {
            for j in 0..4 {
                let mut sum = Exact::new(0, 1);
                for k in 0..4 {
                    sum = sum + m_rat[i][k] * inv_rat[k][j];
                }
                let expected = if i == j {
                    Exact::new(1, 1)
                } else {
                    Exact::new(0, 1)
                };
                assert_eq!(
                    sum, expected,
                    "有理数下 M * M^(-1) 不是精确单位阵: ({i}, {j}) = {sum}, expected {expected}"
                );
            }
        }
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
