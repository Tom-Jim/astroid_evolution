//! The divergence-free modal basis of §8 and the scalar pressure basis of §8.7,
//! evaluated pointwise through truncated Taylor jets.
//!
//! Velocity: `V_{lkm} = ∇×∇×(Φ x)`, `Φ = R² ψ_{lk}(ζ) Y_{lm}`;
//! `W_{lkm} = ∇×(Ψ x)`, `Ψ = R ψ_{lk}(ζ) Y_{lm}`; `ζ = r / R(ω̂)`, `ω̂ = x/r`.
//! `R(ω̂)` is the spherical-harmonic surface. Every derivative of the basis enters
//! analytically: `v` and `∇v` need jets of order 3, `Δv` and `∇∇v` order 4.

use nalgebra::DMatrix;
use rayon::prelude::*;

use crate::common::math::DVec3;
use crate::rheo_friction::basis::{K, lm_index, psi_poly};

use super::harmonics::{SurfaceFit, sh_jets};
use super::jets::Jet;

const S2: f64 = std::f64::consts::SQRT_2;

/// Velocity unknown layout: for every `(l, m)` with `l >= 1` eight consecutive entries,
/// poloidal `k = 0..K` then toroidal `k = 0..K`.
#[derive(Clone, Copy, Debug)]
pub struct VelLayout {
    pub l_max: usize,
    pub n_u: usize,
}

impl VelLayout {
    pub fn new(l_max: usize) -> Self {
        Self {
            l_max,
            n_u: 2 * K * l_max * (l_max + 2),
        }
    }

    #[inline]
    pub fn dof(&self, l: usize, m: i32, toroidal: bool, k: usize) -> usize {
        let blk = (l * l - 1) as i64 + l as i64 + i64::from(m);
        2 * K * blk as usize + if toroidal { K } else { 0 } + k
    }

    /// `(l, m, toroidal, k)` of every unknown, in index order.
    pub fn modes(&self) -> Vec<(usize, i32, bool, usize)> {
        let mut out = Vec::with_capacity(self.n_u);
        for l in 1..=self.l_max {
            for m in -(l as i32)..=(l as i32) {
                for tor in [false, true] {
                    for k in 0..K {
                        out.push((l, m, tor, k));
                    }
                }
            }
        }
        out
    }

    /// Coefficients in the layout of `rheo_friction::VelocityField`.
    pub fn to_rheo(&self, u: &[f64]) -> crate::rheo_friction::VelocityField {
        let n = (self.l_max + 1) * (self.l_max + 1);
        let mut q = vec![[0.0; K]; n];
        let mut s = vec![[0.0; K]; n];
        for l in 1..=self.l_max {
            for m in -(l as i32)..=(l as i32) {
                for k in 0..K {
                    q[lm_index(l, m)][k] = u[self.dof(l, m, false, k)];
                    s[lm_index(l, m)][k] = u[self.dof(l, m, true, k)];
                }
            }
        }
        crate::rheo_friction::VelocityField {
            l_max: self.l_max,
            q,
            s,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Want {
    pub v: bool,
    pub d: bool,
    pub lap: bool,
    pub gd: bool,
}

impl Want {
    pub const V: Self = Self {
        v: true,
        d: false,
        lap: false,
        gd: false,
    };
    pub const VD: Self = Self {
        v: true,
        d: true,
        lap: false,
        gd: false,
    };
    pub const ALL: Self = Self {
        v: true,
        d: true,
        lap: true,
        gd: true,
    };

    fn order(self) -> usize {
        if self.lap || self.gd {
            4
        } else if self.d {
            3
        } else {
            2
        }
    }
}

/// Velocity rows at a set of points. Row `3p + a` of `v`, `6p + c` of `d` (components
/// `xx, yy, zz, √2 xy, √2 xz, √2 yz`, so that `d·d = D:D`), `3p + a` of `lap`, and row
/// `18p + 6k + c` of `gd` (`∂_k D`).
pub struct VelRows {
    pub n_points: usize,
    pub v: DMatrix<f64>,
    pub d: DMatrix<f64>,
    pub lap: DMatrix<f64>,
    pub gd: DMatrix<f64>,
}

struct PointVel {
    v: Vec<[f64; 3]>,
    d: Vec<[f64; 6]>,
    lap: Vec<[f64; 3]>,
    gd: Vec<[f64; 18]>,
}

fn unit_jets(x0: DVec3, ord: usize) -> ([Jet; 3], Jet, [Jet; 3]) {
    let x = [
        Jet::var(0, x0.x, ord),
        Jet::var(1, x0.y, ord),
        Jet::var(2, x0.z, ord),
    ];
    let r2 = x[0].mul(&x[0]).add(&x[1].mul(&x[1])).add(&x[2].mul(&x[2]));
    let r = r2.sqrt();
    let inv = r.recip();
    let u = [x[0].mul(&inv), x[1].mul(&inv), x[2].mul(&inv)];
    (x, r, u)
}

/// Jets `ζ^n` for `n = 0..=top`.
fn zeta_powers(zeta: &Jet, top: usize) -> Vec<Jet> {
    let mut p = Vec::with_capacity(top + 1);
    p.push(Jet::constant(1.0, zeta.ord));
    for n in 1..=top {
        let next = p[n - 1].mul(zeta);
        p.push(next);
    }
    p
}

/// `Σ_j a_j ζ^j` with the Jacobi coefficients of `ψ_{lk}`.
fn psi_jet(poly: &[f64], pow: &[Jet]) -> Jet {
    let mut s = Jet::constant(0.0, pow[0].ord);
    for (j, a) in poly.iter().enumerate() {
        if *a != 0.0 {
            s.axpy(*a, &pow[j]);
        }
    }
    s
}

fn poloidal_field(phi: &Jet, x0: [f64; 3]) -> [Jet; 3] {
    let n = phi.ord;
    let g = [phi.deriv(0), phi.deriv(1), phi.deriv(2)];
    let mut e = phi.truncate(n - 1);
    for a in 0..3 {
        e = e.add(&g[a].mul_x(a, x0[a]));
    }
    let lap = g[0].deriv(0).add(&g[1].deriv(1)).add(&g[2].deriv(2));
    [0, 1, 2].map(|a| e.deriv(a).sub(&lap.mul_x(a, x0[a])))
}

fn toroidal_field(psi: &Jet, x0: [f64; 3]) -> [Jet; 3] {
    let g = [psi.deriv(0), psi.deriv(1), psi.deriv(2)];
    let vx = g[1].mul_x(2, x0[2]).sub(&g[2].mul_x(1, x0[1]));
    let vy = g[2].mul_x(0, x0[0]).sub(&g[0].mul_x(2, x0[2]));
    let vz = g[0].mul_x(1, x0[1]).sub(&g[1].mul_x(0, x0[0]));
    [vx, vy, vz].map(|j| j.truncate(psi.ord - 2))
}

fn pack(v: &[Jet; 3], want: Want, out: &mut PointVel) {
    out.v.push([v[0].val(), v[1].val(), v[2].val()]);
    if want.d || want.lap || want.gd {
        let gr = [v[0].grad(), v[1].grad(), v[2].grad()];
        let dd = |a: usize, b: usize| 0.5 * (gr[a][b] + gr[b][a]);
        out.d.push([
            dd(0, 0),
            dd(1, 1),
            dd(2, 2),
            S2 * dd(0, 1),
            S2 * dd(0, 2),
            S2 * dd(1, 2),
        ]);
    }
    if want.lap || want.gd {
        let h = [v[0].hess(), v[1].hess(), v[2].hess()];
        if want.lap {
            out.lap
                .push([0, 1, 2].map(|a| h[a][0][0] + h[a][1][1] + h[a][2][2]));
        }
        if want.gd {
            let mut g = [0.0; 18];
            for k in 0..3 {
                let dk = |i: usize, j: usize| 0.5 * (h[i][j][k] + h[j][i][k]);
                g[6 * k] = dk(0, 0);
                g[6 * k + 1] = dk(1, 1);
                g[6 * k + 2] = dk(2, 2);
                g[6 * k + 3] = S2 * dk(0, 1);
                g[6 * k + 4] = S2 * dk(0, 2);
                g[6 * k + 5] = S2 * dk(1, 2);
            }
            out.gd.push(g);
        }
    }
}

fn eval_point_velocity(
    surf: &SurfaceFit,
    lay: &VelLayout,
    polys: &[Vec<Vec<f64>>],
    x0: DVec3,
    want: Want,
) -> PointVel {
    let ord = want.order();
    let (x, r, u) = unit_jets(x0, ord);
    let l_top = lay.l_max.max(surf.l_max);
    let y = sh_jets(l_top, &u);
    let rr = surf.radius_jet(&y);
    let zeta = r.mul(&rr.recip());
    let r2 = rr.mul(&rr);
    let pow = zeta_powers(&zeta, lay.l_max + K);
    let xs = [x0.x, x0.y, x0.z];
    let _ = x;
    let mut out = PointVel {
        v: Vec::with_capacity(lay.n_u),
        d: Vec::new(),
        lap: Vec::new(),
        gd: Vec::new(),
    };
    // Per (l, k) radial factors; then per m.
    let mut a_lk: Vec<Vec<Jet>> = vec![Vec::new(); lay.l_max + 1];
    let mut b_lk: Vec<Vec<Jet>> = vec![Vec::new(); lay.l_max + 1];
    for l in 1..=lay.l_max {
        for k in 0..K {
            let psi = psi_jet(&polys[l][k], &pow);
            a_lk[l].push(r2.mul(&psi));
            b_lk[l].push(rr.mul(&psi));
        }
    }
    for l in 1..=lay.l_max {
        for m in -(l as i32)..=(l as i32) {
            let ylm = &y[lm_index(l, m)];
            for tor in [false, true] {
                for k in 0..K {
                    let field = if tor {
                        toroidal_field(&b_lk[l][k].mul(ylm), xs)
                    } else {
                        poloidal_field(&a_lk[l][k].mul(ylm), xs)
                    };
                    pack(&field, want, &mut out);
                }
            }
        }
    }
    out
}

/// Evaluate the velocity basis at `pts`.
pub fn eval_velocity(surf: &SurfaceFit, lay: &VelLayout, pts: &[DVec3], want: Want) -> VelRows {
    let np = pts.len();
    let n_u = lay.n_u;
    let mk = |on: bool, rows: usize| {
        if on {
            DMatrix::zeros(rows * np, n_u)
        } else {
            DMatrix::zeros(0, n_u)
        }
    };
    let mut rows = VelRows {
        n_points: np,
        v: mk(want.v, 3),
        d: mk(want.d || want.lap || want.gd, 6),
        lap: mk(want.lap, 3),
        gd: mk(want.gd, 18),
    };
    const CHUNK: usize = 128;
    let polys: Vec<Vec<Vec<f64>>> = (0..=lay.l_max)
        .map(|l| (0..K).map(|k| psi_poly(l.max(1), k)).collect())
        .collect();
    for (c, chunk) in pts.chunks(CHUNK).enumerate() {
        let eval: Vec<PointVel> = chunk
            .par_iter()
            .map(|p| eval_point_velocity(surf, lay, &polys, *p, want))
            .collect();
        for (i, pv) in eval.iter().enumerate() {
            let p = c * CHUNK + i;
            for j in 0..n_u {
                if want.v {
                    for a in 0..3 {
                        rows.v[(3 * p + a, j)] = pv.v[j][a];
                    }
                }
                if !pv.d.is_empty() {
                    for a in 0..6 {
                        rows.d[(6 * p + a, j)] = pv.d[j][a];
                    }
                }
                if want.lap {
                    for a in 0..3 {
                        rows.lap[(3 * p + a, j)] = pv.lap[j][a];
                    }
                }
                if want.gd {
                    for a in 0..18 {
                        rows.gd[(18 * p + a, j)] = pv.gd[j][a];
                    }
                }
            }
        }
    }
    rows
}

/// Scalar basis `(1 − ζ) ψ_{lk}(ζ) Y_{lm}`, `0 <= l <= l_p`, `k < k_p`, and the
/// boundary lift `ζ^l Y_{lm}`.
#[derive(Clone, Copy, Debug)]
pub struct PresLayout {
    pub l_max: usize,
    pub k_max: usize,
}

impl PresLayout {
    pub fn n_p(&self) -> usize {
        (self.l_max + 1) * (self.l_max + 1) * self.k_max
    }

    pub fn n_lift(&self) -> usize {
        (self.l_max + 1) * (self.l_max + 1)
    }

    #[inline]
    pub fn index(&self, l: usize, m: i32, k: usize) -> usize {
        lm_index(l, m) * self.k_max + k
    }
}

/// Values (`P × n`) and gradients (`3P × n`) of the interior basis and of the lift.
pub struct PresRows {
    pub val: DMatrix<f64>,
    pub grad: DMatrix<f64>,
    pub lift_val: DMatrix<f64>,
    pub lift_grad: DMatrix<f64>,
}

pub fn eval_pressure(
    surf: &SurfaceFit,
    lay: &PresLayout,
    pts: &[DVec3],
    gradients: bool,
) -> PresRows {
    let np = pts.len();
    let n = lay.n_p();
    let nl = lay.n_lift();
    let ord = if gradients { 1 } else { 0 };
    let polys: Vec<Vec<Vec<f64>>> = (0..=lay.l_max)
        .map(|l| (0..lay.k_max).map(|k| psi_poly(l, k)).collect())
        .collect();
    let per_point: Vec<(Vec<f64>, Vec<[f64; 3]>, Vec<f64>, Vec<[f64; 3]>)> = pts
        .par_iter()
        .map(|p| {
            let (_, r, u) = unit_jets(*p, ord);
            let l_top = lay.l_max.max(surf.l_max);
            let y = sh_jets(l_top, &u);
            let rr = surf.radius_jet(&y);
            let zeta = r.mul(&rr.recip());
            let pow = zeta_powers(&zeta, lay.l_max + lay.k_max);
            let one_minus = zeta.scale(-1.0).add_scalar(1.0);
            let mut val = vec![0.0; n];
            let mut grad = vec![[0.0; 3]; n];
            let mut lval = vec![0.0; nl];
            let mut lgrad = vec![[0.0; 3]; nl];
            for l in 0..=lay.l_max {
                for m in -(l as i32)..=(l as i32) {
                    let ylm = &y[lm_index(l, m)];
                    let lift = pow[l].mul(ylm);
                    lval[lm_index(l, m)] = lift.val();
                    if gradients {
                        lgrad[lm_index(l, m)] = lift.grad();
                    }
                    for k in 0..lay.k_max {
                        let psi = psi_jet(&polys[l][k], &pow);
                        let phi = one_minus.mul(&psi).mul(ylm);
                        val[lay.index(l, m, k)] = phi.val();
                        if gradients {
                            grad[lay.index(l, m, k)] = phi.grad();
                        }
                    }
                }
            }
            (val, grad, lval, lgrad)
        })
        .collect();
    let mut rows = PresRows {
        val: DMatrix::zeros(np, n),
        grad: DMatrix::zeros(if gradients { 3 * np } else { 0 }, n),
        lift_val: DMatrix::zeros(np, nl),
        lift_grad: DMatrix::zeros(if gradients { 3 * np } else { 0 }, nl),
    };
    for (p, (val, grad, lval, lgrad)) in per_point.iter().enumerate() {
        for j in 0..n {
            rows.val[(p, j)] = val[j];
            if gradients {
                for a in 0..3 {
                    rows.grad[(3 * p + a, j)] = grad[j][a];
                }
            }
        }
        for j in 0..nl {
            rows.lift_val[(p, j)] = lval[j];
            if gradients {
                for a in 0..3 {
                    rows.lift_grad[(3 * p + a, j)] = lgrad[j][a];
                }
            }
        }
    }
    rows
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn bumpy_surface() -> SurfaceFit {
        // R = 100 (1 + 0.08 Y20-like + 0.05 xy-like) fitted on a dense grid.
        let (dirs, w) = super::super::quad::sphere_grid(16, 32);
        let radii: Vec<f64> = dirs
            .iter()
            .map(|d| 100.0 * (1.0 + 0.08 * (3.0 * d.z * d.z - 1.0) + 0.05 * d.x * d.y))
            .collect();
        SurfaceFit::fit(4, &dirs, &radii, &w).unwrap()
    }

    #[test]
    fn velocity_modes_are_divergence_free_and_match_finite_differences() {
        let surf = bumpy_surface();
        let lay = VelLayout::new(3);
        let x0 = DVec3::new(31.0, -22.0, 40.0);
        let rows = eval_velocity(&surf, &lay, &[x0], Want::ALL);
        let h = 1e-3;
        let shift = |axis: usize, s: f64| {
            let mut p = x0.to_array();
            p[axis] += s;
            eval_velocity(&surf, &lay, &[DVec3::from_array(p)], Want::V)
        };
        let plus: Vec<_> = (0..3).map(|a| shift(a, h)).collect();
        let minus: Vec<_> = (0..3).map(|a| shift(a, -h)).collect();
        let mid = shift(0, 0.0);
        for j in 0..lay.n_u {
            let scale = (0..3).map(|a| rows.v[(a, j)].abs()).fold(1e-30, f64::max);
            let tr = rows.d[(0, j)] + rows.d[(1, j)] + rows.d[(2, j)];
            assert!(
                tr.abs() < 1e-8 * scale / 40.0 * 10.0 + 1e-12,
                "trace dof {j}: {tr}"
            );
            // gradient by central differences of the value rows
            let gr = |a: usize, b: usize| (plus[b].v[(a, j)] - minus[b].v[(a, j)]) / (2.0 * h);
            let dxx = gr(0, 0);
            let dxy = 0.5 * (gr(0, 1) + gr(1, 0));
            assert!(
                (dxx - rows.d[(0, j)]).abs() < 1e-5 * scale / 40.0,
                "Dxx dof {j}"
            );
            assert!(
                (S2 * dxy - rows.d[(3, j)]).abs() < 1e-5 * scale / 40.0,
                "Dxy dof {j}"
            );
            // Laplacian
            let mut lap = [0.0; 3];
            for a in 0..3 {
                for b in 0..3 {
                    lap[a] +=
                        (plus[b].v[(a, j)] - 2.0 * mid.v[(a, j)] + minus[b].v[(a, j)]) / (h * h);
                }
                assert!(
                    (lap[a] - rows.lap[(a, j)]).abs()
                        < 2e-4 * scale / 40.0 / 40.0 * 40.0 + 1e-6 * scale,
                    "Laplacian dof {j} comp {a}: {} vs {}",
                    lap[a],
                    rows.lap[(a, j)]
                );
            }
        }
    }

    #[test]
    fn l1_k0_modes_are_rigid_on_a_sphere() {
        let surf = SurfaceFit {
            l_max: 0,
            coeff: vec![100.0 * (4.0 * PI).sqrt()],
        };
        let lay = VelLayout::new(2);
        let x0 = DVec3::new(10.0, 20.0, -30.0);
        let rows = eval_velocity(&surf, &lay, &[x0], Want::VD);
        for m in -1..=1 {
            for tor in [false, true] {
                let j = lay.dof(1, m, tor, 0);
                for c in 0..6 {
                    assert!(rows.d[(c, j)].abs() < 1e-10, "mode {m} {tor} comp {c}");
                }
            }
        }
        // A genuinely deforming mode has non-zero strain.
        let j = lay.dof(2, 0, false, 0);
        assert!((0..6).map(|c| rows.d[(c, j)].abs()).sum::<f64>() > 1e-6);
    }

    #[test]
    fn pressure_basis_vanishes_on_the_surface() {
        let surf = bumpy_surface();
        let lay = PresLayout { l_max: 3, k_max: 3 };
        let w = DVec3::new(0.3, -0.5, 0.8).try_normalize().unwrap();
        let x = w * surf.radius(w);
        let rows = eval_pressure(&surf, &lay, &[x], true);
        assert!(rows.val.iter().all(|v| v.abs() < 1e-12));
        // The lift reproduces Y_lm there.
        let y = crate::rheo_friction::basis::sh_all(3, w.to_array());
        for l in 0..=3 {
            for m in -(l as i32)..=(l as i32) {
                assert!((rows.lift_val[(0, lm_index(l, m))] - y[lm_index(l, m)]).abs() < 1e-10);
            }
        }
    }
}
