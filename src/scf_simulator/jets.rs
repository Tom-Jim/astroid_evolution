//! Truncated Taylor polynomials in three variables, up to total degree 4.
//!
//! The velocity basis `V = ∇×∇×(Φ x)`, `W = ∇×(Ψ x)` of §8 is built from potentials
//! `Φ = R²(x̂) ψ_{ℓk}(r/R(x̂)) Y_{ℓm}(x̂)` whose angular dependence enters through
//! `x̂ = x/|x|`. Carrying the Taylor jet of `Φ` at a point gives `v`, `∇v` and `Δv`
//! exactly (to rounding), with no finite differences and no hand-expanded formulas
//! for the angular derivatives of `R`.
//!
//! A jet of order `ord` stores every Taylor coefficient of degree `<= ord`. Taking a
//! derivative lowers `ord` by one, so the result never claims a coefficient it does
//! not know.

use std::sync::OnceLock;

/// Monomials of degree `<= 4` in three variables.
pub const NC: usize = 35;
pub const MAX_ORD: usize = 4;

struct Tables {
    mono: [[u8; 3]; NC],
    deg: [u8; NC],
    /// Index of `α + e_a`, or `NC` when it exceeds degree 4.
    up: [[usize; NC]; 3],
    /// Product table per order: `(i, j, k)` with `mono[i] + mono[j] = mono[k]` and degree `<= ord`.
    mul: [Vec<(u8, u8, u8)>; MAX_ORD + 1],
    /// Index of `e_b + e_c`.
    pair: [[usize; 3]; 3],
}

fn tables() -> &'static Tables {
    static T: OnceLock<Tables> = OnceLock::new();
    T.get_or_init(|| {
        let mut mono = [[0u8; 3]; NC];
        let mut deg = [0u8; NC];
        let mut n = 0;
        for d in 0..=MAX_ORD as u8 {
            for a in (0..=d).rev() {
                for b in (0..=d - a).rev() {
                    mono[n] = [a, b, d - a - b];
                    deg[n] = d;
                    n += 1;
                }
            }
        }
        debug_assert_eq!(n, NC);
        let index = |m: [u8; 3]| -> usize {
            (0..NC).find(|&i| mono[i] == m).unwrap_or(NC)
        };
        let mut up = [[NC; NC]; 3];
        for axis in 0..3 {
            for i in 0..NC {
                let mut m = mono[i];
                m[axis] += 1;
                if m[0] + m[1] + m[2] <= MAX_ORD as u8 {
                    up[axis][i] = index(m);
                }
            }
        }
        let mut mul: [Vec<(u8, u8, u8)>; MAX_ORD + 1] = Default::default();
        for ord in 0..=MAX_ORD {
            for i in 0..NC {
                for j in 0..NC {
                    if (deg[i] + deg[j]) as usize <= ord {
                        let m = [
                            mono[i][0] + mono[j][0],
                            mono[i][1] + mono[j][1],
                            mono[i][2] + mono[j][2],
                        ];
                        mul[ord].push((i as u8, j as u8, index(m) as u8));
                    }
                }
            }
        }
        let mut pair = [[0usize; 3]; 3];
        for b in 0..3 {
            for c in 0..3 {
                let mut m = [0u8; 3];
                m[b] += 1;
                m[c] += 1;
                pair[b][c] = index(m);
            }
        }
        Tables {
            mono,
            deg,
            up,
            mul,
            pair,
        }
    })
}

/// Number of coefficients with degree `<= ord`.
const fn count(ord: usize) -> usize {
    match ord {
        0 => 1,
        1 => 4,
        2 => 10,
        3 => 20,
        _ => 35,
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Jet {
    pub c: [f64; NC],
    pub ord: usize,
}

impl Jet {
    pub fn constant(v: f64, ord: usize) -> Self {
        let mut c = [0.0; NC];
        c[0] = v;
        Self {
            c,
            ord: ord.min(MAX_ORD),
        }
    }

    /// `x0 + h_axis`.
    pub fn var(axis: usize, x0: f64, ord: usize) -> Self {
        let mut j = Self::constant(x0, ord);
        if j.ord >= 1 {
            j.c[1 + axis] = 1.0;
        }
        j
    }

    #[inline]
    pub fn val(&self) -> f64 {
        self.c[0]
    }

    /// First Taylor coefficients `(∂_x, ∂_y, ∂_z)`.
    #[inline]
    pub fn grad(&self) -> [f64; 3] {
        [self.c[1], self.c[2], self.c[3]]
    }

    pub fn add(&self, o: &Self) -> Self {
        let ord = self.ord.min(o.ord);
        let mut r = Self::constant(0.0, ord);
        for i in 0..count(ord) {
            r.c[i] = self.c[i] + o.c[i];
        }
        r
    }

    pub fn sub(&self, o: &Self) -> Self {
        let ord = self.ord.min(o.ord);
        let mut r = Self::constant(0.0, ord);
        for i in 0..count(ord) {
            r.c[i] = self.c[i] - o.c[i];
        }
        r
    }

    pub fn scale(&self, s: f64) -> Self {
        let mut r = *self;
        for i in 0..count(self.ord) {
            r.c[i] *= s;
        }
        r
    }

    pub fn add_scalar(&self, s: f64) -> Self {
        let mut r = *self;
        r.c[0] += s;
        r
    }

    /// `self += s * o` (orders must agree or `self` is truncated to the lower one).
    pub fn axpy(&mut self, s: f64, o: &Self) {
        self.ord = self.ord.min(o.ord);
        for i in 0..count(self.ord) {
            self.c[i] += s * o.c[i];
        }
    }

    pub fn mul(&self, o: &Self) -> Self {
        let ord = self.ord.min(o.ord);
        let mut r = Self::constant(0.0, ord);
        for &(i, j, k) in &tables().mul[ord] {
            r.c[k as usize] += self.c[i as usize] * o.c[j as usize];
        }
        r
    }

    /// `∂_axis`. The result has order `ord - 1`.
    pub fn deriv(&self, axis: usize) -> Self {
        assert!(self.ord >= 1, "a jet of order 0 has no derivative");
        let t = tables();
        let ord = self.ord - 1;
        let mut r = Self::constant(0.0, ord);
        for i in 0..count(ord) {
            let up = t.up[axis][i];
            r.c[i] = f64::from(t.mono[i][axis] + 1) * self.c[up];
        }
        r
    }

    /// `(x0 + h_axis) * self`, the product with a coordinate.
    pub fn mul_x(&self, axis: usize, x0: f64) -> Self {
        let t = tables();
        let mut r = self.scale(x0);
        for i in 0..count(self.ord) {
            if usize::from(t.deg[i]) < self.ord {
                r.c[t.up[axis][i]] += self.c[i];
            }
        }
        r
    }

    /// `f(self)` for a scalar function given by `f, f', f'', f''', f''''` at `self.val()`.
    pub fn compose(&self, f: &[f64; 5]) -> Self {
        let mut delta = *self;
        delta.c[0] = 0.0;
        let mut fact = 1.0;
        let mut coef = [0.0; 5];
        for k in 0..=MAX_ORD {
            if k > 0 {
                fact *= k as f64;
            }
            coef[k] = f[k] / fact;
        }
        let mut r = Self::constant(coef[self.ord], self.ord);
        for k in (0..self.ord).rev() {
            r = r.mul(&delta);
            r.c[0] += coef[k];
        }
        r
    }

    pub fn recip(&self) -> Self {
        let a = self.c[0];
        self.compose(&[
            1.0 / a,
            -1.0 / (a * a),
            2.0 / a.powi(3),
            -6.0 / a.powi(4),
            24.0 / a.powi(5),
        ])
    }

    pub fn sqrt(&self) -> Self {
        let a = self.c[0];
        let s = a.sqrt();
        self.compose(&[
            s,
            0.5 / s,
            -0.25 / (a * s),
            0.375 / (a * a * s),
            -0.937_5 / (a.powi(3) * s),
        ])
    }

    /// The same jet cut to a lower order.
    pub fn truncate(&self, ord: usize) -> Self {
        let mut r = *self;
        r.ord = r.ord.min(ord);
        for i in count(r.ord)..NC {
            r.c[i] = 0.0;
        }
        r
    }

    /// Second derivatives `∂_b ∂_c f` at the expansion point (order >= 2).
    pub fn hess(&self) -> [[f64; 3]; 3] {
        assert!(self.ord >= 2, "hess needs a jet of order 2");
        let t = tables();
        let mut out = [[0.0; 3]; 3];
        for b in 0..3 {
            for c in 0..3 {
                let factor = if b == c { 2.0 } else { 1.0 };
                out[b][c] = factor * self.c[t.pair[b][c]];
            }
        }
        out
    }

    /// Value of the Taylor polynomial at the offset `h`.
    pub fn eval(&self, h: [f64; 3]) -> f64 {
        let t = tables();
        let mut sum = 0.0;
        for i in 0..count(self.ord) {
            let m = t.mono[i];
            sum += self.c[i]
                * h[0].powi(i32::from(m[0]))
                * h[1].powi(i32::from(m[1]))
                * h[2].powi(i32::from(m[2]));
        }
        sum
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn product_and_derivative_follow_the_leibniz_rule() {
        let (x, y, z) = (
            Jet::var(0, 0.7, 4),
            Jet::var(1, -0.3, 4),
            Jet::var(2, 1.1, 4),
        );
        let f = x.mul(&y).mul(&z).add(&x.mul(&x));
        // f = x y z + x^2
        assert!((f.val() - (0.7 * -0.3 * 1.1 + 0.49)).abs() < 1e-14);
        let fx = f.deriv(0);
        assert!((fx.val() - (-0.3 * 1.1 + 1.4)).abs() < 1e-14);
        let fxx = fx.deriv(0);
        assert!((fxx.val() - 2.0).abs() < 1e-14);
        let fyz = f.deriv(1).deriv(2);
        assert!((fyz.val() - 0.7).abs() < 1e-14);
    }

    #[test]
    fn composition_matches_closed_forms() {
        let x = Jet::var(0, 2.0, 4);
        let s = x.sqrt();
        let h = 0.1;
        assert!((s.eval([h, 0.0, 0.0]) - (2.0 + h).sqrt()).abs() < 1e-7);
        let r = x.recip();
        assert!((r.eval([h, 0.0, 0.0]) - 1.0 / (2.0 + h)).abs() < 1e-6);
        let ms = x.mul_x(0, 2.0);
        // (2 + h)(2 + h) = 4 + 4h + h^2
        assert!((ms.eval([h, 0.0, 0.0]) - (2.0 + h) * (2.0 + h)).abs() < 1e-14);
    }
}
