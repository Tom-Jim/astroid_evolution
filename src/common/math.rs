//! f64 vectors and the six-component symmetric tensor used by the Hessian.

use std::ops::{Add, Mul, Neg, Sub};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DVec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl DVec3 {
    pub const ZERO: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    #[inline]
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    #[inline]
    pub fn from_array(a: [f64; 3]) -> Self {
        Self::new(a[0], a[1], a[2])
    }

    #[inline]
    pub fn to_array(self) -> [f64; 3] {
        [self.x, self.y, self.z]
    }

    #[inline]
    pub fn dot(self, o: Self) -> f64 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }

    #[inline]
    pub fn cross(self, o: Self) -> Self {
        Self::new(
            self.y * o.z - self.z * o.y,
            self.z * o.x - self.x * o.z,
            self.x * o.y - self.y * o.x,
        )
    }

    #[inline]
    pub fn length_squared(self) -> f64 {
        self.dot(self)
    }

    #[inline]
    pub fn length(self) -> f64 {
        self.length_squared().sqrt()
    }

    /// Unit vector. A zero vector is rejected so a missing direction cannot become a silent zero.
    #[inline]
    pub fn try_normalize(self) -> Option<Self> {
        let len = self.length();
        if len > 0.0 && len.is_finite() {
            Some(self * (1.0 / len))
        } else {
            None
        }
    }

    #[inline]
    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }
}

impl Add for DVec3 {
    type Output = Self;
    #[inline]
    fn add(self, o: Self) -> Self {
        Self::new(self.x + o.x, self.y + o.y, self.z + o.z)
    }
}

impl Sub for DVec3 {
    type Output = Self;
    #[inline]
    fn sub(self, o: Self) -> Self {
        Self::new(self.x - o.x, self.y - o.y, self.z - o.z)
    }
}

impl Neg for DVec3 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self::new(-self.x, -self.y, -self.z)
    }
}

impl Mul<f64> for DVec3 {
    type Output = Self;
    #[inline]
    fn mul(self, s: f64) -> Self {
        Self::new(self.x * s, self.y * s, self.z * s)
    }
}

/// Row-major 3×3 matrix.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DMat3 {
    pub m: [[f64; 3]; 3],
}

impl DMat3 {
    pub const ZERO: Self = Self { m: [[0.0; 3]; 3] };

    #[inline]
    pub fn from_columns(c0: DVec3, c1: DVec3, c2: DVec3) -> Self {
        Self {
            m: [[c0.x, c1.x, c2.x], [c0.y, c1.y, c2.y], [c0.z, c1.z, c2.z]],
        }
    }

    #[inline]
    pub fn identity_scaled(s: f64) -> Self {
        Self {
            m: [[s, 0.0, 0.0], [0.0, s, 0.0], [0.0, 0.0, s]],
        }
    }

    #[inline]
    pub fn trace(self) -> f64 {
        self.m[0][0] + self.m[1][1] + self.m[2][2]
    }
}

/// Symmetric tensor packed as `(xx, yy, zz, xy, xz, yz)`.
pub type Sym6 = [f64; 6];

#[inline]
pub fn sym6_zero() -> Sym6 {
    [0.0; 6]
}

#[inline]
pub fn sym6_scale(h: Sym6, s: f64) -> Sym6 {
    [h[0] * s, h[1] * s, h[2] * s, h[3] * s, h[4] * s, h[5] * s]
}

#[inline]
pub fn sym6_add(mut a: Sym6, b: Sym6) -> Sym6 {
    for i in 0..6 {
        a[i] += b[i];
    }
    a
}

#[inline]
pub fn sym6_trace(h: Sym6) -> f64 {
    h[0] + h[1] + h[2]
}

/// `T(u) = 3 uuᵀ − I`, stored as six independent entries. Its trace is identically zero.
#[inline]
pub fn quadrupole(u: DVec3) -> Sym6 {
    [
        3.0 * u.x * u.x - 1.0,
        3.0 * u.y * u.y - 1.0,
        3.0 * u.z * u.z - 1.0,
        3.0 * u.x * u.y,
        3.0 * u.x * u.z,
        3.0 * u.y * u.z,
    ]
}

#[inline]
pub fn sym6_outer(n: DVec3) -> Sym6 {
    [
        n.x * n.x,
        n.y * n.y,
        n.z * n.z,
        n.x * n.y,
        n.x * n.z,
        n.y * n.z,
    ]
}

#[inline]
pub fn sym6_max_abs_diff(a: Sym6, b: Sym6) -> f64 {
    let mut m = 0.0_f64;
    for i in 0..6 {
        m = m.max((a[i] - b[i]).abs());
    }
    m
}
