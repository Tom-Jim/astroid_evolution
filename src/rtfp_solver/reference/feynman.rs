//! Feynman-parameter reference for a polyhedron with a generalized-Cauchy density.
//!
//! With `A(y) = σ⁻² + |y − c|²` and `p = α + ½`, the Feynman identity
//!
//! ```text
//! 1 / (|x−y| A^α) = C ∫₀¹ u^{-1/2} (1−u)^{α−1} (|y − m_u|² + q_u)^{-p} du,
//! C = Γ(α+½) / (Γ(½) Γ(α)),   m_u = u x + (1−u) c,
//! q_u = (1−u) σ⁻² + u (1−u) |x−c|²,
//! ```
//!
//! turns the Newton kernel into a spherically symmetric one about `m_u`.
//! Its volume integral over the polyhedron is a sum over faces (divergence
//! theorem with the regular field `(y−m) G_p(r)/r³`, `G_p(s) = ∫₀^s F_p r² dr`):
//!
//! ```text
//! Ψ_p     = ∫_V F_p dV   = Σ_f ∫_f (n·(y−m)) G_p(r)/r³ dA,    F_p = (|y−m|²+q)^{-p}
//! ∇_m Ψ_p = −∮ n F_p dS.
//! ```
//!
//! The gradient follows from `∂m/∂x = u I`, `∂q/∂x = 2u(1−u)(x−c)` and
//! `∂_q Ψ_p = −p Ψ_{p+1}`. The `u` integral is adaptive. Nothing here traces
//! rays and nothing is shared with the RTFP solver. The uniform bulk is the
//! exact Werner polyhedron.

use std::f64::consts::PI;
use std::num::NonZeroUsize;

use gauss_quad::GaussLegendre;
use rayon::prelude::*;

use super::werner;
use crate::common::density::{CauchyDensity, CauchyKernel};
use crate::common::geometry::TriMesh;
use crate::common::math::{DVec3, Sym6};
use crate::rtfp_solver::error::RtfpError;

/// A patch is split while its longest edge exceeds this fraction of `√(dist² + q)`.
const PATCH_RATIO: f64 = 0.25;
const PATCH_DEPTH: usize = 14;
/// Relative tolerance of the adaptive `u` integral.
const U_TOLERANCE: f64 = 1.0e-8;
const U_DEPTH: usize = 30;
const THETA_MAX: f64 = 80.0;
const THETA_PANEL: f64 = 0.5;

/// `K_p(w) = ∫₀^w t² (1+t²)^{-p} dt` and `K_{p+1}(w)`, tabulated in `θ = asinh w`.
struct RadialTable {
    p: f64,
    rule: GaussLegendre,
    /// Cumulative `(K_p, K_{p+1})` at each panel boundary.
    cumulative: Vec<(f64, f64)>,
}

impl RadialTable {
    fn new(p: f64) -> Self {
        let rule = GaussLegendre::new(NonZeroUsize::new(10).expect("nonzero"));
        let mut table = Self { p, rule, cumulative: vec![(0.0, 0.0)] };
        let panels = (THETA_MAX / THETA_PANEL) as usize;
        for j in 0..panels {
            let (a, b) = table.panel(j as f64 * THETA_PANEL, (j + 1) as f64 * THETA_PANEL);
            let last = table.cumulative[j];
            table.cumulative.push((last.0 + a, last.1 + b));
        }
        table
    }

    fn panel(&self, lo: f64, hi: f64) -> (f64, f64) {
        let (mut a, mut b) = (0.0, 0.0);
        let half = 0.5 * (hi - lo);
        for &(node, weight) in self.rule.as_node_weight_pairs() {
            let t = lo + half * (node + 1.0);
            let (s, c) = (t.sinh(), t.cosh());
            let base = s * s * c.powf(1.0 - 2.0 * self.p);
            a += half * weight * base;
            b += half * weight * base / (c * c);
        }
        (a, b)
    }

    /// `(K_p(w)/w³, K_{p+1}(w)/w³)`.
    fn scaled(&self, w: f64) -> (f64, f64) {
        if w < 1e-3 {
            let w2 = w * w;
            return (
                (1.0 - 0.6 * self.p * w2) / 3.0,
                (1.0 - 0.6 * (self.p + 1.0) * w2) / 3.0,
            );
        }
        let theta = w.asinh().min(THETA_MAX);
        let j = ((theta / THETA_PANEL) as usize).min(self.cumulative.len() - 2);
        let start = j as f64 * THETA_PANEL;
        let (a, b) = self.panel(start, theta);
        let base = self.cumulative[j];
        let w3 = w * w * w;
        ((base.0 + a) / w3, (base.1 + b) / w3)
    }
}

#[derive(Clone, Copy, Default)]
struct Pieces {
    psi: f64,
    psi_next: f64,
    surface: [f64; 3],
}

impl Pieces {
    fn add(&mut self, o: &Pieces) {
        self.psi += o.psi;
        self.psi_next += o.psi_next;
        for i in 0..3 {
            self.surface[i] += o.surface[i];
        }
    }
}

/// Triangle node rule: a Duffy map of `n × n` Gauss–Legendre points.
struct TriangleRule {
    nodes: Vec<(f64, f64, f64)>,
}

impl TriangleRule {
    fn new(n: usize) -> Self {
        let gl = GaussLegendre::new(NonZeroUsize::new(n).expect("nonzero"));
        let pairs: Vec<(f64, f64)> = gl
            .as_node_weight_pairs()
            .iter()
            .map(|&(x, w)| (0.5 * (x + 1.0), 0.5 * w))
            .collect();
        let mut nodes = Vec::new();
        for &(xi, wx) in &pairs {
            for &(eta, we) in &pairs {
                // y = a + xi (b − a) + xi eta (c − a); jacobian 2A xi.
                nodes.push((xi, eta, wx * we * xi));
            }
        }
        Self { nodes }
    }
}

pub struct Feynman<'a> {
    mesh: &'a TriMesh,
    density: &'a CauchyDensity,
    g: f64,
    triangle: TriangleRule,
}

impl<'a> Feynman<'a> {
    pub fn new(mesh: &'a TriMesh, density: &'a CauchyDensity, g: f64) -> Self {
        Self { mesh, density, g, triangle: TriangleRule::new(6) }
    }

    /// Body radius about the bounding-box centre.
    fn radius(&self) -> f64 {
        let (mut lo, mut hi) = (self.mesh.vertices[0], self.mesh.vertices[0]);
        for v in &self.mesh.vertices {
            lo = DVec3::new(lo.x.min(v.x), lo.y.min(v.y), lo.z.min(v.z));
            hi = DVec3::new(hi.x.max(v.x), hi.y.max(v.y), hi.z.max(v.z));
        }
        let center = (lo + hi) * 0.5;
        self.mesh.vertices.iter().map(|v| (*v - center).length()).fold(0.0, f64::max)
    }

    /// Face integrals at one Feynman parameter.
    fn pieces(&self, table: &RadialTable, m: DVec3, q: f64) -> Pieces {
        let p = table.p;
        let sqrt_q = q.sqrt();
        let q_pow = q.powf(-p);
        (0..self.mesh.faces.len())
            .into_par_iter()
            .map(|f| {
                let n = self.mesh.face_normal(f).expect("validated mesh");
                let mut out = Pieces::default();
                let mut stack = vec![(self.mesh.triangle(f), 0usize)];
                while let Some((tri, depth)) = stack.pop() {
                    let [a, b, c] = tri;
                    let centroid = (a + b + c) * (1.0 / 3.0);
                    let nearest = [centroid, a, b, c]
                        .iter()
                        .map(|y| (*y - m).length_squared())
                        .fold(f64::INFINITY, f64::min);
                    let diameter = (b - a).length().max((c - b).length()).max((a - c).length());
                    if depth < PATCH_DEPTH && diameter > PATCH_RATIO * (nearest + q).sqrt() {
                        let (ab, bc, ca) = ((a + b) * 0.5, (b + c) * 0.5, (c + a) * 0.5);
                        stack.push(([a, ab, ca], depth + 1));
                        stack.push(([ab, b, bc], depth + 1));
                        stack.push(([ca, bc, c], depth + 1));
                        stack.push(([ab, bc, ca], depth + 1));
                        continue;
                    }
                    let area2 = (b - a).cross(c - a).length();
                    for &(xi, eta, w) in &self.triangle.nodes {
                        let y = a + (b - a) * xi + (c - a) * (xi * eta);
                        let weight = w * area2;
                        let s = y - m;
                        let r2 = s.length_squared();
                        let f_q = r2 + q;
                        let f_p = f_q.powf(-p);
                        out.surface[0] += n.x * f_p * weight;
                        out.surface[1] += n.y * f_p * weight;
                        out.surface[2] += n.z * f_p * weight;
                        let sn = n.dot(s);
                        if sn != 0.0 {
                            let (k1, k2) = table.scaled(r2.sqrt() / sqrt_q);
                            out.psi += sn * q_pow * k1 * weight;
                            out.psi_next += sn * q_pow / q * k2 * weight;
                        }
                    }
                }
                out
            })
            .reduce(Pieces::default, |mut a, b| {
                a.add(&b);
                a
            })
    }

    /// `G (potential, gradient)` of one Cauchy kernel (weight included) over the polyhedron.
    pub fn kernel_field(&self, kernel: &CauchyKernel, x: DVec3) -> (f64, DVec3) {
        let alpha = kernel.alpha;
        let p = alpha + 0.5;
        let a2 = 1.0 / (kernel.sigma * kernel.sigma);
        let table = RadialTable::new(p);
        let offset = x - kernel.center;
        let dist2 = offset.length_squared();
        let k = 1.0 / alpha;

        let integrand = |tau: f64| -> [f64; 4] {
            let t = tau.powf(k);
            let s = 1.0 - t;
            let u = s * s;
            let one_minus_u = t * (2.0 - t);
            let m = kernel.center + offset * u;
            let q = one_minus_u * a2 + u * one_minus_u * dist2;
            let parts = self.pieces(&table, m, q);
            let weight = 2.0 * k * (2.0 - t).powf(alpha - 1.0);
            let c = 2.0 * p * u * one_minus_u * parts.psi_next;
            [
                weight * parts.psi,
                weight * (-u * parts.surface[0] - c * offset.x),
                weight * (-u * parts.surface[1] - c * offset.y),
                weight * (-u * parts.surface[2] - c * offset.z),
            ]
        };
        let v = adaptive(&integrand);
        let prefactor = libm::tgamma(p) / (PI.sqrt() * libm::tgamma(alpha))
            * kernel.weight
            * kernel.sigma.powf(-2.0 * alpha)
            * self.g;
        (prefactor * v[0], DVec3::new(v[1], v[2], v[3]) * prefactor)
    }

    /// Potential and gradient of the whole density: Werner bulk plus Feynman kernels.
    pub fn potential_gradient(&self, x: DVec3) -> (f64, DVec3) {
        let bulk = werner::uniform_polyhedron(self.mesh, x, self.g, self.density.bulk);
        let mut potential = bulk.potential;
        let mut gradient = bulk.gradient;
        for kernel in &self.density.kernels {
            let (u, g) = self.kernel_field(kernel, x);
            potential += u;
            gradient = gradient + g;
        }
        (potential, gradient)
    }

    fn kernel_gradient(&self, x: DVec3) -> DVec3 {
        let mut gradient = DVec3::ZERO;
        for kernel in &self.density.kernels {
            gradient = gradient + self.kernel_field(kernel, x).1;
        }
        gradient
    }

    /// Kernel part of the Hessian at a point off the surface, by central differences of the exact gradient.
    fn kernel_hessian_fd(&self, x: DVec3, h: f64) -> Sym6 {
        let axes = [DVec3::new(1.0, 0.0, 0.0), DVec3::new(0.0, 1.0, 0.0), DVec3::new(0.0, 0.0, 1.0)];
        let mut cols = [[0.0; 3]; 3];
        for (j, axis) in axes.iter().enumerate() {
            let d = (self.kernel_gradient(x + *axis * h) - self.kernel_gradient(x - *axis * h)) * (0.5 / h);
            cols[j] = d.to_array();
        }
        let sym = |i: usize, j: usize| 0.5 * (cols[j][i] + cols[i][j]);
        [sym(0, 0), sym(1, 1), sym(2, 2), sym(0, 1), sym(0, 2), sym(1, 2)]
    }

    /// Hessian at a point strictly inside or outside the body. The reference is a
    /// difference quotient, so the point must be farther from the surface than a few steps.
    pub fn hessian(&self, x: DVec3) -> Result<Sym6, RtfpError> {
        let radius = self.radius();
        let d = werner::distance_to_surface(self.mesh, x);
        if d < 1e-4 * radius {
            return Err(RtfpError::geometry(
                "Feynman Hessian reference needs a point farther from the surface than 1e-4 of the body radius",
            ));
        }
        let h = (1e-3 * radius).min(d / 8.0);
        let mut out = self.kernel_hessian_fd(x, h);
        let bulk = werner::uniform_polyhedron(self.mesh, x, self.g, self.density.bulk).hessian;
        for i in 0..6 {
            out[i] += bulk[i];
        }
        Ok(out)
    }

    /// Finite-part Hessian at the interior of face `face`: the mean of the one-sided limits.
    ///
    /// The kernel part is a quadratic extrapolation of the exterior Hessian to the face,
    /// reduced by `2π G ρ_k n⊗n`, half of the density-times-`n⊗n` jump across the surface.
    pub fn face_hessian(&self, x: DVec3, face: usize) -> Result<Sym6, RtfpError> {
        let n = self.mesh.face_normal(face).ok_or_else(|| RtfpError::geometry("degenerate face"))?;
        let e = 0.02 * self.radius();
        let at = |k: f64| self.kernel_hessian_fd(x + n * (k * e), e / 8.0);
        let (h1, h2, h3) = (at(1.0), at(2.0), at(3.0));
        let rho_kernels = self.density.rho(x) - self.density.bulk;
        let jump_half = 2.0 * PI * self.g * rho_kernels;
        let nn = crate::common::math::sym6_outer(n);
        let mut out = [0.0; 6];
        for i in 0..6 {
            out[i] = 3.0 * h1[i] - 3.0 * h2[i] + h3[i] - jump_half * nn[i];
        }
        let bulk = werner::uniform_polyhedron(self.mesh, x, self.g, self.density.bulk).hessian;
        for i in 0..6 {
            out[i] += bulk[i];
        }
        Ok(out)
    }
}

/// Adaptive Gauss–Legendre integral of a four-component function on `[0, 1]`.
fn adaptive(f: &dyn Fn(f64) -> [f64; 4]) -> [f64; 4] {
    let rule = GaussLegendre::new(NonZeroUsize::new(10).expect("nonzero"));
    let panel = |a: f64, b: f64| -> [f64; 4] {
        let half = 0.5 * (b - a);
        let mut acc = [0.0; 4];
        for &(node, weight) in rule.as_node_weight_pairs() {
            let v = f(a + half * (node + 1.0));
            for i in 0..4 {
                acc[i] += half * weight * v[i];
            }
        }
        acc
    };
    let (left, right) = (panel(0.0, 0.5), panel(0.5, 1.0));
    let estimate: [f64; 4] = std::array::from_fn(|i| left[i] + right[i]);
    let grad_scale = estimate[1].abs().max(estimate[2].abs()).max(estimate[3].abs());
    let scale = [
        estimate[0].abs().max(1e-300),
        estimate[1].abs().max(1e-3 * grad_scale).max(1e-300),
        estimate[2].abs().max(1e-3 * grad_scale).max(1e-300),
        estimate[3].abs().max(1e-3 * grad_scale).max(1e-300),
    ];
    let mut total = [0.0; 4];
    let mut stack = vec![(0.0, 0.5, left, 1usize), (0.5, 1.0, right, 1usize)];
    while let Some((a, b, whole, depth)) = stack.pop() {
        let mid = 0.5 * (a + b);
        let (l, r) = (panel(a, mid), panel(mid, b));
        let error = (0..4)
            .map(|i| (l[i] + r[i] - whole[i]).abs() / scale[i])
            .fold(0.0, f64::max);
        if error <= U_TOLERANCE || depth >= U_DEPTH {
            for i in 0..4 {
                total[i] += l[i] + r[i];
            }
        } else {
            stack.push((a, mid, l, depth + 1));
            stack.push((mid, b, r, depth + 1));
        }
    }
    total
}
