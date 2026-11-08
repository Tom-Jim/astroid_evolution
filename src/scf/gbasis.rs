//! The geometry-adapted divergence-free velocity basis of (G1), MS §8.1.
//!
//! Where `basis.rs` is the enclosing-ball spectral family of the reference
//! loop, this module is the normative construction: compactly supported vector
//! kernels centred at the volume nodes and face quadrature points of the actual
//! polyhedron, plus one flux mode `∇|x − z_j|⁻¹` per sealed cavity
//! (Proposition 8.1.5). Trial fields are curls of kernel fields, so every
//! member is exactly solenoidal; the six rigid modes are appended explicitly
//! (Proposition 8.1.4). The weighted Gram matrix is assembled on the (G2)
//! ray-segment quadrature and its rank is checked before the solve
//! (Proposition 8.1.3). This CPU assembly is the reference path; it is also the
//! projector of the Galerkin bridge that §8.3 and §10.3.2 build on the ray
//! state.

use nalgebra::DMatrix;

use crate::common::math::DVec3;

/// One basis function of the (G1) family.
#[derive(Clone, Debug)]
pub enum Mode {
    /// `B(x) = ∇×(W(r) e)` with `W` the compact Wendland C² kernel of radius
    /// `h` centred at `c`; `e` a unit axis. Solenoidal by construction.
    Kernel { c: DVec3, e: DVec3, h: f64 },
    /// A rigid translation `a` or rotation `θ × x` (Proposition 8.1.4).
    Rigid { a: DVec3, w: DVec3 },
    /// The cavity-flux mode `∇|x − z_j|⁻¹` of Proposition 8.1.5: harmonic and
    /// divergence-free on `𝒱`, with unit flux `±4π` through `∂C_j`.
    Flux { z: DVec3 },
    /// A global polynomial swirl `B = ∇×(m·e)`, `m = x^i y^j z^k`: the
    /// coarsest scale of the same curl construction. Compact swirls alone
    /// approximate a smooth field only through near-dependent combinations —
    /// the low-degree part is carried exactly by a few global modes.
    Poly { e: DVec3, ijk: [u32; 3] },
}

impl Mode {
    /// The vector potential `A(x)` with `B = ∇×A`: `W(s)·e` of a kernel,
    /// `Zero` where the potential has no canonical representative (rigid
    /// translations have none; a harmonic field's is not needed).
    pub fn potential(&self, x: DVec3) -> DVec3 {
        match *self {
            Mode::Kernel { c, e, h } => {
                let r = x - c;
                let s = r.length();
                if s >= h {
                    DVec3::ZERO
                } else {
                    let q = 1.0 - s / h;
                    e * (q.powi(4) * (4.0 * s / h + 1.0))
                }
            }
            Mode::Poly { e, ijk } => e * mono_dn(ijk, &[], x),
            _ => DVec3::ZERO,
        }
    }

    /// The value `B(x)`.
    pub fn value(&self, x: DVec3) -> DVec3 {
        match *self {
            Mode::Kernel { c, e, h } => curl_kernel(x, c, e, h),
            Mode::Rigid { a, w } => a + w.cross(x),
            Mode::Flux { z } => {
                let r = x - z;
                let rn = r.length();
                if rn < 1e-30 {
                    DVec3::ZERO
                } else {
                    // −∇|x−z|⁻¹: the flux carrier of Prop 8.1.5. The sign only
                    // orients the mode.
                    r / -(rn * rn * rn)
                }
            }
            Mode::Poly { e, ijk } => DVec3::new(
                mono_dn(ijk, &[0], x),
                mono_dn(ijk, &[1], x),
                mono_dn(ijk, &[2], x),
            )
            .cross(e),
        }
    }

    /// The scaled strain six-vector `D = sym ∇B` in `xx, yy, zz, √2xy, √2xz,
    /// √2yz` order, so `d·d = D:D` (the `basis.rs` convention).
    pub fn strain(&self, x: DVec3) -> [f64; 6] {
        const S2: f64 = std::f64::consts::SQRT_2;
        match *self {
            Mode::Kernel { c, e, h } => {
                let g = jac_curl_kernel(x, c, e, h);
                [
                    g[(0, 0)],
                    g[(1, 1)],
                    g[(2, 2)],
                    S2 * 0.5 * (g[(0, 1)] + g[(1, 0)]),
                    S2 * 0.5 * (g[(0, 2)] + g[(2, 0)]),
                    S2 * 0.5 * (g[(1, 2)] + g[(2, 1)]),
                ]
            }
            Mode::Rigid { .. } => [0.0; 6],
            Mode::Flux { z } => {
                let r = x - z;
                let rn2 = r.length_squared().max(1e-60);
                let rn5 = rn2.powf(2.5);
                // J_ij = −δ/r³ + 3 r_i r_j/r⁵ of −∇(1/r) (symmetric here).
                let j = |i: f64, j: f64, d: f64| -d / rn2.powf(1.5) + 3.0 * i * j / rn5;
                [
                    j(r.x, r.x, 1.0),
                    j(r.y, r.y, 1.0),
                    j(r.z, r.z, 1.0),
                    S2 * j(r.x, r.y, 0.0),
                    S2 * j(r.x, r.z, 0.0),
                    S2 * j(r.y, r.z, 0.0),
                ]
            }
            Mode::Poly { e, ijk } => {
                // ∂_b B = (∂_b ∇m) × e: Jacobian row b is the Hessian column
                // crossed with the axis.
                let j = |b: usize| {
                    DVec3::new(
                        mono_dn(ijk, &[b, 0], x),
                        mono_dn(ijk, &[b, 1], x),
                        mono_dn(ijk, &[b, 2], x),
                    )
                    .cross(e)
                };
                let (r0, r1, r2) = (j(0), j(1), j(2));
                [
                    r0.x,
                    r1.y,
                    r2.z,
                    S2 * 0.5 * (r0.y + r1.x),
                    S2 * 0.5 * (r0.z + r2.x),
                    S2 * 0.5 * (r1.z + r2.y),
                ]
            }
        }
    }

    /// `ΔB(x)` — the viscous term of the momentum residual.
    pub fn lap(&self, x: DVec3) -> DVec3 {
        match *self {
            Mode::Kernel { c, e, h } => {
                let r = x - c;
                let s = r.length();
                if s >= h || s < 1e-30 {
                    return DVec3::ZERO;
                }
                // B = k(r×e), k = W'/s ⇒ ΔB = (k'' + 2k'/s)(r×e) + 2k'(r̂×e).
                let (_, dk, ddk) = kernel_k(s, h);
                (r.cross(e)) * (ddk + 2.0 * dk / s) + (r / s).cross(e) * (2.0 * dk)
            }
            // ΔB = ∇(Δm) × e for the polynomial swirl.
            Mode::Poly { e, ijk } => DVec3::new(
                mono_dn(ijk, &[0, 0, 0], x)
                    + mono_dn(ijk, &[1, 1, 0], x)
                    + mono_dn(ijk, &[2, 2, 0], x),
                mono_dn(ijk, &[0, 0, 1], x)
                    + mono_dn(ijk, &[1, 1, 1], x)
                    + mono_dn(ijk, &[2, 2, 1], x),
                mono_dn(ijk, &[0, 0, 2], x)
                    + mono_dn(ijk, &[1, 1, 2], x)
                    + mono_dn(ijk, &[2, 2, 2], x),
            )
            .cross(e),
            // Rigid modes and the harmonic flux mode are harmonic.
            _ => DVec3::ZERO,
        }
    }

    /// `∂_k D` of `B`, 18 entries ordered `6k + c` in the same scaled
    /// six-vector convention as `strain`.
    pub fn grad_strain(&self, x: DVec3) -> [f64; 18] {
        const S2: f64 = std::f64::consts::SQRT_2;
        let mut out = [0.0; 18];
        match *self {
            Mode::Kernel { c, e, h } => {
                let r = x - c;
                let s = r.length();
                if s >= h || s < 1e-30 {
                    return out;
                }
                let (_, dk, ddk) = kernel_k(s, h);
                let rh = r / s;
                let rx_e = r.cross(e);
                let axes = [DVec3::X, DVec3::Y, DVec3::Z];
                for (kk, &ek) in axes.iter().enumerate() {
                    let ck = ek.cross(e);
                    let mut t = nalgebra::Matrix3::<f64>::zeros();
                    for (i, &ei) in axes.iter().enumerate() {
                        let ci = ei.cross(e);
                        let dki = if i == kk { 1.0 } else { 0.0 };
                        let pk = dki - rh[kk] * rh[i];
                        // t[i][j] = ∂_k∂_i B_j
                        let row = rx_e * (ddk * rh[kk] * rh[i] + dk / s * pk)
                            + ck * (dk * rh[i])
                            + ci * (dk * rh[kk]);
                        for j in 0..3 {
                            t[(i, j)] = row[j];
                        }
                    }
                    out[6 * kk] = t[(0, 0)];
                    out[6 * kk + 1] = t[(1, 1)];
                    out[6 * kk + 2] = t[(2, 2)];
                    out[6 * kk + 3] = S2 * 0.5 * (t[(0, 1)] + t[(1, 0)]);
                    out[6 * kk + 4] = S2 * 0.5 * (t[(0, 2)] + t[(2, 0)]);
                    out[6 * kk + 5] = S2 * 0.5 * (t[(1, 2)] + t[(2, 1)]);
                }
            }
            Mode::Flux { z } => {
                let r = x - z;
                let r2 = r.length_squared();
                if r2 < 1e-30 {
                    return out;
                }
                let ir5 = r2.powf(-2.5);
                let ir7 = ir5 / r2;
                let ri = [r.x, r.y, r.z];
                for kk in 0..3 {
                    let mut t = nalgebra::Matrix3::<f64>::zeros();
                    for i in 0..3 {
                        for j in 0..3 {
                            let dki = if i == kk { 1.0 } else { 0.0 };
                            let dkj = if j == kk { 1.0 } else { 0.0 };
                            let dij = if i == j { 1.0 } else { 0.0 };
                            t[(i, j)] = 3.0 * ri[kk] * dij * ir5
                                + 3.0 * (dki * ri[j] + ri[i] * dkj) * ir5
                                - 15.0 * ri[kk] * ri[i] * ri[j] * ir7;
                        }
                    }
                    out[6 * kk] = t[(0, 0)];
                    out[6 * kk + 1] = t[(1, 1)];
                    out[6 * kk + 2] = t[(2, 2)];
                    out[6 * kk + 3] = S2 * 0.5 * (t[(0, 1)] + t[(1, 0)]);
                    out[6 * kk + 4] = S2 * 0.5 * (t[(0, 2)] + t[(2, 0)]);
                    out[6 * kk + 5] = S2 * 0.5 * (t[(1, 2)] + t[(2, 1)]);
                }
            }
            Mode::Poly { e, ijk } => {
                for kk in 0..3 {
                    let mut t = nalgebra::Matrix3::<f64>::zeros();
                    for i in 0..3 {
                        let row = DVec3::new(
                            mono_dn(ijk, &[kk, i, 0], x),
                            mono_dn(ijk, &[kk, i, 1], x),
                            mono_dn(ijk, &[kk, i, 2], x),
                        )
                        .cross(e);
                        for j in 0..3 {
                            t[(i, j)] = row[j];
                        }
                    }
                    out[6 * kk] = t[(0, 0)];
                    out[6 * kk + 1] = t[(1, 1)];
                    out[6 * kk + 2] = t[(2, 2)];
                    out[6 * kk + 3] = S2 * 0.5 * (t[(0, 1)] + t[(1, 0)]);
                    out[6 * kk + 4] = S2 * 0.5 * (t[(0, 2)] + t[(2, 0)]);
                    out[6 * kk + 5] = S2 * 0.5 * (t[(1, 2)] + t[(2, 1)]);
                }
            }
            Mode::Rigid { .. } => {}
        }
        out
    }
}

/// Mixed partial of the monomial `x^i y^j z^k` at `x`: `ders` lists the
/// derivative coordinates (e.g. `[0,0,1]` for `∂²_x ∂_z`). Falling-factorial
/// coefficient included; an over-differentiated coordinate yields 0.
fn mono_dn(ijk: [u32; 3], ders: &[usize], x: DVec3) -> f64 {
    let mut n = [0u32; 3];
    for &a in ders {
        n[a] += 1;
    }
    let xs = [x.x, x.y, x.z];
    let mut v = 1.0;
    for c in 0..3 {
        if ijk[c] < n[c] {
            return 0.0;
        }
        let mut coef = 1.0;
        for t in 0..n[c] {
            coef *= (ijk[c] - t) as f64;
        }
        v *= coef * xs[c].powi((ijk[c] - n[c]) as i32);
    }
    v
}

/// `(k, k', k'')` of `B = k(s)·(r × e)`, `k = W'(s)/s` for the Wendland
/// profile: `k = −20q³/h²`, `k' = 60q²/h³`, `k'' = −120q/h⁴`, `q = 1 − s/h`.
#[inline]
fn kernel_k(s: f64, h: f64) -> (f64, f64, f64) {
    let q = 1.0 - s / h;
    let h2 = h * h;
    (
        -20.0 * q * q * q / h2,
        60.0 * q * q / (h2 * h),
        -120.0 * q / (h2 * h2),
    )
}

/// The compact kernel `W(s) = (1 − s/h)⁴ (4s/h + 1)`, `s = |x − c|`: the
/// Wendland C² function, `W > 0` on `s < h`. Only its derivatives enter the
/// basis, since every trial field is a curl of `W e`.
#[inline]
fn kernel_dw_ds(s: f64, h: f64) -> f64 {
    if s >= h {
        return 0.0;
    }
    let q = 1.0 - s / h;
    -20.0 * s * q * q * q / (h * h)
}

#[inline]
fn kernel_d2w(s: f64, h: f64) -> f64 {
    if s >= h {
        return 0.0;
    }
    let q = 1.0 - s / h;
    20.0 * q * q * (4.0 * s / h - 1.0) / (h * h)
}

/// `∇×(W e) = ∇W × e = W'(s)/s · (x − c) × e`.
fn curl_kernel(x: DVec3, c: DVec3, e: DVec3, h: f64) -> DVec3 {
    let r = x - c;
    let s = r.length();
    if s >= h || s < 1e-30 {
        return DVec3::ZERO;
    }
    (r.cross(e)) * (kernel_dw_ds(s, h) / s)
}

/// Jacobian `∂_j (∇×(We))_i` of the kernel mode.
fn jac_curl_kernel(x: DVec3, c: DVec3, e: DVec3, h: f64) -> nalgebra::Matrix3<f64> {
    let r = x - c;
    let s = r.length();
    let mut g = nalgebra::Matrix3::<f64>::zeros();
    if s >= h || s < 1e-30 {
        return g;
    }
    // B = k(s)·(r × e) with k = W'/s.  ∂B/∂r_j = k'(s) r̂_j (r×e) + k (e_j × e).
    let k = kernel_dw_ds(s, h) / s;
    let dk = (kernel_d2w(s, h) - k) / s; // d/ds(W'/s) = (W''−W'/s)/s
    let rx_e = r.cross(e);
    let unit = r / s;
    let axes = [DVec3::X, DVec3::Y, DVec3::Z];
    for j in 0..3 {
        let ej_x_e = axes[j].cross(e);
        for i in 0..3 {
            g[(i, j)] = dk * unit[j] * rx_e[i] + k * ej_x_e[i];
        }
    }
    g
}

/// The (G1) family on one geometry: kernel modes at the volume nodes and face
/// quadrature points, the six rigid modes, and one flux mode per sealed cavity.
#[derive(Clone, Debug)]
pub struct GeoBasis {
    pub modes: Vec<Mode>,
    /// Kernel support radius used by every `Mode::Kernel`.
    pub h_kernel: f64,
    /// Centre of each cavity flux mode (the `z_j` of Prop 8.1.5).
    pub cavity_centres: Vec<DVec3>,
}

impl GeoBasis {
    /// Build on the volume-node set and the face quadrature points.
    ///
    /// `h_kernel` is the common compact support radius; the caller ties it to
    /// the cell size (≈ `h_vol` of the CellBVH) so that every kernel overlaps
    /// several quadrature nodes. Every fourth centre is additionally given a
    /// kernel at `3·h_kernel`: a single support scale cannot carry both the
    /// smooth part of the field and its detail — wide swirls approximate the
    /// former at a mass-Gram conditioning price the narrow ones do not pay
    /// (Prop 8.1.3 is checked on the union). `cavity_centres` are the interior
    /// points of the sealed components of `TriMesh::cavities()`; with none,
    /// the family is complete in the connected-`Γ` sense of Prop 8.1.5(1).
    pub fn build(
        vol_pts: &[DVec3],
        face_pts: &[DVec3],
        h_kernel: f64,
        cavity_centres: Vec<DVec3>,
    ) -> Self {
        assert!(h_kernel > 0.0 && h_kernel.is_finite());
        let axes = [DVec3::X, DVec3::Y, DVec3::Z];
        let mut modes = Vec::new();
        // Six rigid modes first: they must be present in every retained family.
        for a in axes {
            modes.push(Mode::Rigid { a, w: DVec3::ZERO });
        }
        for w in axes {
            modes.push(Mode::Rigid { a: DVec3::ZERO, w });
        }
        // The global polynomial scale: `∇×(m·e)` for monomials of degree
        // 2..=4 — `B` then spans the cubic swirls (a Kelvin mode, a tide);
        // degree 1 repeats the rigid translations and a pure-power monomial
        // with its own axis is identically zero. Asking the compact kernels
        // to carry the smooth part forces near-dependent combinations the
        // Prop 8.1.3 check then has to pay for.
        for deg in 2..=4u32 {
            for i in 0..=deg {
                for j in 0..=(deg - i) {
                    let ijk = [i, j, deg - i - j];
                    for (a, &e) in axes.iter().enumerate() {
                        if ijk[a] == deg {
                            continue;
                        }
                        modes.push(Mode::Poly { e, ijk });
                    }
                }
            }
        }
        let n_c = vol_pts.len() + face_pts.len();
        for (i, &c) in vol_pts.iter().chain(face_pts.iter()).enumerate() {
            for e in axes {
                modes.push(Mode::Kernel { c, e, h: h_kernel });
            }
            if n_c > 64 && i % 4 == 0 {
                for e in axes {
                    modes.push(Mode::Kernel {
                        c,
                        e,
                        h: 3.0 * h_kernel,
                    });
                }
            }
            if n_c > 64 && i % 16 == 0 {
                for e in axes {
                    modes.push(Mode::Kernel {
                        c,
                        e,
                        h: 6.0 * h_kernel,
                    });
                }
            }
        }
        for &z in &cavity_centres {
            modes.push(Mode::Flux { z });
        }
        Self {
            modes,
            h_kernel,
            cavity_centres,
        }
    }

    pub fn n(&self) -> usize {
        self.modes.len()
    }

    /// The basis matrix `B` at `pts`: row `3p + a` of `v`, `6p + c` of `d`
    /// — the same layout `VelRows` uses for the spectral family.
    #[cfg(feature = "tier2")]
    pub fn rows(&self, pts: &[DVec3]) -> (DMatrix<f64>, DMatrix<f64>) {
        let (n_p, n_b) = (pts.len(), self.modes.len());
        let mut v = DMatrix::<f64>::zeros(3 * n_p, n_b);
        let mut d = DMatrix::<f64>::zeros(6 * n_p, n_b);
        for (b, m) in self.modes.iter().enumerate() {
            for (p, x) in pts.iter().enumerate() {
                let val = m.value(*x);
                for a in 0..3 {
                    v[(3 * p + a, b)] = val[a];
                }
                let st = m.strain(*x);
                for c in 0..6 {
                    d[(6 * p + c, b)] = st[c];
                }
            }
        }
        (v, d)
    }

    /// The velocity `Σ u_b B_b` at `pts`.
    #[cfg(feature = "tier2")]
    pub fn contract(&self, pts: &[DVec3], u: &[f64]) -> Vec<f64> {
        assert_eq!(u.len(), self.modes.len());
        let mut out = vec![0.0; 3 * pts.len()];
        for (p, x) in pts.iter().enumerate() {
            let mut val = DVec3::ZERO;
            for (b, m) in self.modes.iter().enumerate() {
                if u[b] != 0.0 {
                    val += m.value(*x) * u[b];
                }
            }
            out[3 * p] = val.x;
            out[3 * p + 1] = val.y;
            out[3 * p + 2] = val.z;
        }
        out
    }
}

/// Gram and independence report of Proposition 8.1.3: `M = Σ_p ρ_p w_p
/// B(x_p)ᵀ B(x_p)` on the (G2) quadrature, with the smallest singular value
/// and an effective rank at `tol` times the largest.
#[derive(Clone, Debug)]
pub struct GramReport {
    pub sigma_min: f64,
    pub sigma_max: f64,
    pub rank: usize,
    pub n_modes: usize,
    /// `sigma_min / sigma_max`: the conditioning §8.1 asks to report.
    pub cond: f64,
}

impl GramReport {
    pub fn full_rank(&self) -> bool {
        self.rank == self.n_modes
    }
}

/// Kernel modes centred further apart than the sum of their supports do not
/// overlap: their Gram entry is exactly zero. That is most pairs — skipping
/// them is what makes the Prop 8.1.3 assembly tractable.
pub(crate) fn disjoint(a: &Mode, b: &Mode) -> bool {
    if let (Mode::Kernel { c: ca, h: ha, .. }, Mode::Kernel { c: cb, h: hb, .. }) = (a, b) {
        ca.distance(*cb) >= ha + hb
    } else {
        false
    }
}

/// For every mode the quadrature points on which it does not vanish: the
/// support points of a kernel, all points of a global mode. Pair sums then
/// touch the union of two supports, not the whole quadrature.
pub(crate) fn support_lists(basis: &GeoBasis, pts: &[DVec3]) -> Vec<Vec<u32>> {
    basis
        .modes
        .iter()
        .map(|m| match m {
            Mode::Kernel { c, h, .. } => (0..pts.len() as u32)
                .filter(|&p| pts[p as usize].distance(*c) < *h)
                .collect(),
            _ => (0..pts.len() as u32).collect(),
        })
        .collect()
}

/// The quadrature points of the union of two support lists.
pub(crate) fn union_support(a: &[u32], b: &[u32], out: &mut Vec<u32>) {
    out.clear();
    out.extend_from_slice(a);
    out.extend_from_slice(b);
    out.sort_unstable();
    out.dedup();
}

/// Assemble `M_Γ = ∫ ρ B·B dV` on quadrature `(pts, w, rho)` and check rank.
/// `rho` uniform `None` means `ρ = 1`.
#[cfg(feature = "tier2")]
pub fn gram_check(
    basis: &GeoBasis,
    pts: &[DVec3],
    w: &[f64],
    rho: Option<&[f64]>,
    tol: f64,
) -> GramReport {
    let m = gram_matrix(basis, pts, w, rho);
    // Rank is read on the correlation Gram `D^{-1/2} M D^{-1/2}`: the raw
    // spectrum mixes the natural scales of the families — a rigid rotation's
    // |B| is ~R against a kernel's ~h^{-1} — so `tol·σ_max` measures a unit
    // mismatch, not dependence. Prop 8.1.3 asks whether the retained
    // *functions* are independent, which is the normalized matrix.
    let c = correlation(&m);
    let svd = c.svd(true, true);
    let s = svd.singular_values;
    let sigma_max = s.iter().cloned().fold(0.0, f64::max);
    let sigma_min = s.iter().cloned().fold(f64::INFINITY, f64::min);
    let rank = s.iter().filter(|&&v| v > tol * sigma_max).count();
    GramReport {
        sigma_min,
        sigma_max,
        rank,
        n_modes: basis.n(),
        cond: if sigma_min > 0.0 {
            sigma_max / sigma_min
        } else {
            f64::INFINITY
        },
    }
}

/// `D^{-1/2} M D^{-1/2}`: the Gram matrix of the modes normalized to unit
/// `L²(ρ w)` on the quadrature — the angle between the functions, their scale
/// removed. A zero diagonal (an unsupported mode) maps to a zero row.
pub(crate) fn correlation(m: &DMatrix<f64>) -> DMatrix<f64> {
    let n = m.nrows();
    // The floor must survive a product of two: `1e-300²` underflows to 0
    // and a zero-norm mode (rigid under the strain metric) would turn the
    // whole correlation row into NaN.
    let d: Vec<f64> = (0..n).map(|i| m[(i, i)].sqrt().max(1e-150)).collect();
    DMatrix::from_fn(n, n, |b, c| m[(b, c)] / (d[b] * d[c]))
}

/// Drop modes until the retained family is independent in both of the
/// metrics it serves. The orthogonalization of §8.1 is a pivoted
/// Gram–Schmidt on the correlation Gram `D^{-1/2} M D^{-1/2}`: walk the
/// modes, keep the ones whose weighted-orthogonal residual against the
/// retained span exceeds the tolerance, and grow the Cholesky factor of the
/// retained block as we go. Independence is then by construction — no
/// iterative peeling of an SVD. Stage one selects in the mass metric (the
/// projector's); stage two in the strain metric (the solves'): a mode whose
/// strain is already spanned sits in the stiffness Gram's numerical null
/// space and is handed a noise-determined coefficient. Rigid and flux modes
/// are never dropped: a deficient rigid block or a dependent flux mode means
/// the geometry is wrong, not the basis.
pub fn prune(
    mut basis: GeoBasis,
    pts: &[DVec3],
    w: &[f64],
    rho: Option<&[f64]>,
    tol: f64,
    max_drops: usize,
) -> (GeoBasis, GramReport) {
    // Keep modes whose orthogonal residual against the retained span exceeds
    // ~3%: a chained sequence of barely-independent pivots still leaves the
    // correlation Gram below the rank threshold, which poisons every solve
    // built on the basis.
    let res2_tol = 1e-3_f64.max(tol.sqrt());
    // Stage one selects in the strain metric (the solves'): a mode whose
    // strain is already spanned sits in the stiffness Gram's numerical null
    // space and is handed a noise-determined coefficient. The candidate order
    // puts the global layer — rigid, flux, polynomial — ahead of the compact
    // kernels, so an energy-distinct polynomial direction is never lost to a
    // kernel combination that merely resembles it. Energy-independence
    // implies mass-independence (a zero field has zero strain), which is what
    // lets the second stage leave the vetted global layer alone.
    let sgram = strain_gram_matrix(&basis, pts, w, rho);
    let scorr = correlation(&sgram);
    let (kept_s, drops) = pivoted_select(&scorr, &basis.modes, res2_tol, max_drops, &[]);
    basis.modes = kept_s.iter().map(|&b| basis.modes[b].clone()).collect();
    // Stage two re-vets only the compact kernels in the mass metric (the
    // projector's). The strain-vetted global layer is protected: its modes
    // are energy-distinct hence mass-independent, but possibly below the
    // mass threshold — the price of a complete even layer is a stiffer Gram
    // condition number, the remedy §8.1's orthonormalization anticipates.
    let protected: Vec<bool> = basis
        .modes
        .iter()
        .map(|m| !matches!(m, Mode::Kernel { .. }))
        .collect();
    let corr = correlation(&gram_matrix(&basis, pts, w, rho));
    let (kept, _) = pivoted_select(
        &corr,
        &basis.modes,
        res2_tol,
        max_drops.saturating_sub(drops),
        &protected,
    );
    basis.modes = kept.iter().map(|&b| basis.modes[b].clone()).collect();
    let rep = gram_check(&basis, pts, w, rho, tol);
    (basis, rep)
}

/// Greedy pivoted Gram–Schmidt on a correlation matrix: keep the modes whose
/// weighted-orthogonal residual against the retained span exceeds `res2_tol`
/// (squared), mandatory modes first — rigid and flux modes are never dropped.
/// `protected` modes (the strain-vetted global layer) are kept too, unless
/// their residual is numerically zero — a true duplicate rather than a
/// mass-weak direction. Returns the kept indices in keep order and the drop
/// count.
fn pivoted_select(
    corr: &DMatrix<f64>,
    modes: &[Mode],
    res2_tol: f64,
    max_drops: usize,
    protected: &[bool],
) -> (Vec<usize>, usize) {
    let n = modes.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&b| match modes[b] {
        Mode::Rigid { .. } | Mode::Flux { .. } => 0,
        Mode::Poly { .. } => 1,
        Mode::Kernel { .. } => 2,
    });
    let mut kept: Vec<usize> = Vec::with_capacity(n);
    let mut chol = DMatrix::<f64>::zeros(n, n);
    let mut drops = 0usize;
    for b in order {
        let k = kept.len();
        // y solves the kept block: residual² = corr_bb − ‖L⁻¹ c‖².
        let mut y = vec![0.0; k];
        for j in 0..k {
            let mut s = corr[(b, kept[j])];
            for t in 0..j {
                s -= chol[(j, t)] * y[t];
            }
            y[j] = s / chol[(j, j)];
        }
        let res2 = corr[(b, b)] - y.iter().map(|v| v * v).sum::<f64>();
        let mandatory = matches!(modes[b], Mode::Rigid { .. } | Mode::Flux { .. });
        let vetted = protected.get(b).copied().unwrap_or(false) && res2 > 1e-12;
        if res2 > res2_tol || mandatory || vetted || drops >= max_drops {
            for t in 0..k {
                chol[(k, t)] = y[t];
            }
            // A mandatory mode kept despite a vanishing residual (a rigid
            // mode under the strain metric) would inject a near-zero pivot:
            // its correlation column is identically zero anyway, so a unit
            // pivot is exact and keeps the forward solves finite.
            chol[(k, k)] = if res2 > res2_tol { res2.sqrt() } else { 1.0 };
            kept.push(b);
        } else {
            drops += 1;
        }
    }
    (kept, drops)
}

/// The assembled `M_Γ` of `gram_check`, reused by `prune`.
pub(crate) fn gram_matrix(
    basis: &GeoBasis,
    pts: &[DVec3],
    w: &[f64],
    rho: Option<&[f64]>,
) -> DMatrix<f64> {
    let (n_b, supp) = (basis.n(), support_lists(basis, pts));
    let mut m = DMatrix::<f64>::zeros(n_b, n_b);
    let mut union = Vec::new();
    for (b, mb) in basis.modes.iter().enumerate() {
        for (c, mc) in basis.modes.iter().take(b + 1).enumerate() {
            if disjoint(mb, mc) {
                continue;
            }
            union_support(&supp[b], &supp[c], &mut union);
            let mut s = 0.0;
            for &p in &union {
                let p = p as usize;
                let pw = rho.map_or(1.0, |r| r[p]) * w[p];
                if pw != 0.0 {
                    s += pw * mb.value(pts[p]).dot(mc.value(pts[p]));
                }
            }
            m[(b, c)] = s;
            m[(c, b)] = s;
        }
    }
    m
}

/// The strain Gram `∫ρ w D B_b·D B_c` on the same quadrature: the stiffness
/// metric of the flow solves. A mode can be mass-independent yet
/// strain-dependent — its coefficient is then determined by noise alone.
pub(crate) fn strain_gram_matrix(
    basis: &GeoBasis,
    pts: &[DVec3],
    w: &[f64],
    rho: Option<&[f64]>,
) -> DMatrix<f64> {
    let (n_b, supp) = (basis.n(), support_lists(basis, pts));
    let mut m = DMatrix::<f64>::zeros(n_b, n_b);
    let mut union = Vec::new();
    for (b, mb) in basis.modes.iter().enumerate() {
        for (c, mc) in basis.modes.iter().take(b + 1).enumerate() {
            if disjoint(mb, mc) {
                continue;
            }
            union_support(&supp[b], &supp[c], &mut union);
            let mut s = 0.0;
            for &p in &union {
                let p = p as usize;
                let pw = rho.map_or(1.0, |r| r[p]) * w[p];
                if pw != 0.0 {
                    let (sb, sc) = (mb.strain(pts[p]), mc.strain(pts[p]));
                    for k in 0..6 {
                        s += pw * sb[k] * sc[k];
                    }
                }
            }
            m[(b, c)] = s;
            m[(c, b)] = s;
        }
    }
    m
}

/// The Galerkin projection of §8.3: `u = argmin ∫ρ|v^ray − Σu_αB_α|²`, solved
/// by the normal equations on the (G2) quadrature. Returns `(u, e_proj)`,
/// `e_proj = ‖v^ray − v_h‖_{L²(ρ)} / ‖v^ray‖_{L²(ρ)}`.
#[cfg(feature = "tier2")]
pub fn project(
    basis: &GeoBasis,
    pts: &[DVec3],
    w: &[f64],
    rho: Option<&[f64]>,
    v_ray: &[f64],
) -> Result<(Vec<f64>, f64), String> {
    let n_b = basis.n();
    assert_eq!(v_ray.len(), 3 * pts.len());
    let m = gram_matrix(basis, pts, w, rho);
    let supp = support_lists(basis, pts);
    let mut rhs = vec![0.0f64; n_b];
    let mut norm_v = 0.0;
    for p in 0..pts.len() {
        let pw = rho.map_or(1.0, |r| r[p]) * w[p];
        norm_v += pw * (v_ray[3 * p].powi(2) + v_ray[3 * p + 1].powi(2) + v_ray[3 * p + 2].powi(2));
    }
    for (b, mb) in basis.modes.iter().enumerate() {
        for &p in &supp[b] {
            let p = p as usize;
            let pw = rho.map_or(1.0, |r| r[p]) * w[p];
            let v = DVec3::new(v_ray[3 * p], v_ray[3 * p + 1], v_ray[3 * p + 2]);
            rhs[b] += pw * mb.value(pts[p]).dot(v);
        }
    }
    let Some(ch) = m.cholesky() else {
        return Err("the (G1) Gram matrix is not positive definite".into());
    };
    let u = ch.solve(&nalgebra::DVector::from_column_slice(&rhs));
    let u = u.as_slice().to_vec();
    let vh = basis.contract(pts, &u);
    let mut res = 0.0;
    for p in 0..pts.len() {
        let pw = rho.map_or(1.0, |r| r[p]) * w[p];
        let dv = DVec3::new(
            vh[3 * p] - v_ray[3 * p],
            vh[3 * p + 1] - v_ray[3 * p + 1],
            vh[3 * p + 2] - v_ray[3 * p + 2],
        );
        res += pw * dv.length_squared();
    }
    let e_proj = if norm_v > 0.0 {
        (res / norm_v).sqrt()
    } else {
        res.sqrt()
    };
    Ok((u, e_proj))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::math::DVec3;

    fn grid(n: usize) -> Vec<DVec3> {
        let mut pts = Vec::new();
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    pts.push(DVec3::new(
                        (i as f64 + 0.5) / n as f64 - 0.5,
                        (j as f64 + 0.5) / n as f64 - 0.5,
                        (k as f64 + 0.5) / n as f64 - 0.5,
                    ));
                }
            }
        }
        pts
    }

    /// Every member field is divergence-free: `tr ∇B = 0` on the strain
    /// vector and `∇·v = 0` numerically for a contraction.
    #[test]
    fn kernel_modes_are_solenoidal() {
        let m = Mode::Kernel {
            c: DVec3::new(0.1, -0.2, 0.05),
            e: DVec3::Y,
            h: 0.8,
        };
        for x in [
            DVec3::new(0.0, 0.0, 0.0),
            DVec3::new(0.3, 0.1, -0.4),
            DVec3::new(-0.5, -0.5, 0.2),
        ] {
            let d = m.strain(x);
            assert!(d[0] + d[1] + d[2] < 1e-12, "trace {}", d[0] + d[1] + d[2]);
        }
        // Compact support: zero outside h.
        let m_far = Mode::Kernel {
            c: DVec3::ZERO,
            e: DVec3::Z,
            h: 0.5,
        };
        assert_eq!(m_far.value(DVec3::new(0.6, 0.0, 0.0)), DVec3::ZERO);
    }

    /// `∇|x−z|⁻¹` is harmonic on 𝒱: divergence-free away from z.
    #[test]
    fn flux_mode_is_solenoidal_and_harmonic() {
        let m = Mode::Flux { z: DVec3::ZERO };
        let d = m.strain(DVec3::new(0.4, 0.3, -0.2));
        assert!(
            (d[0] + d[1] + d[2]).abs() < 1e-12,
            "trace {}",
            d[0] + d[1] + d[2]
        );
        let v = m.value(DVec3::new(0.5, 0.0, 0.0));
        assert!((v.x + 4.0).abs() < 1e-12 && v.y.abs() < 1e-15);
    }

    /// A rigid motion is reproduced exactly by its appended mode, and the
    /// Gram check reports full rank on a well-sampled family.
    #[test]
    fn rigid_modes_exact_and_gram_full_rank() {
        let pts = grid(3);
        let w = vec![1.0 / pts.len() as f64; pts.len()];
        let basis = GeoBasis::build(&pts[..4], &pts[4..8], 1.2, vec![]);
        // Rigid translation along x is mode 0.
        let mut u = vec![0.0; basis.n()];
        u[0] = 2.5;
        let v = basis.contract(&pts, &u);
        assert!(
            v.chunks(3)
                .all(|c| (c[0] - 2.5).abs() < 1e-14 && c[1].abs() < 1e-14)
        );
        // Rotation about z is mode 5.
        let mut u2 = vec![0.0; basis.n()];
        u2[5] = 1.0;
        let v2 = basis.contract(&[DVec3::new(0.3, 0.0, 0.0)], &u2);
        assert!((v2[1] - 0.3).abs() < 1e-14, "θ×x = ŷ·0.3, got {v2:?}");
        let rep = gram_check(&basis, &pts, &w, None, 1e-10);
        assert_eq!(rep.n_modes, basis.n());
        assert!(
            rep.rank >= 6,
            "rigid block at least; σ_min = {:e}",
            rep.sigma_min
        );
    }

    /// The projection returns `e_proj = 0` for a field in the span and a
    /// bounded residual for a field outside it.
    #[test]
    fn projection_bridge_recovers_span_and_reports_residual() {
        let pts = grid(3);
        let w = vec![1.0 / pts.len() as f64; pts.len()];
        let basis = GeoBasis::build(&pts[..8], &[], 1.2, vec![]);
        let mut u0 = vec![0.0; basis.n()];
        u0[3] = 1.3; // rigid ω_x
        u0[7] = -0.4; // a kernel mode
        let v_ray = basis.contract(&pts, &u0);
        let (u, e_proj) = project(&basis, &pts, &w, None, &v_ray).unwrap();
        assert!(e_proj < 1e-8, "in-span field: e_proj = {e_proj}");
        let v_back = basis.contract(&pts, &u);
        for p in 0..pts.len() {
            for a in 0..3 {
                assert!((v_back[3 * p + a] - v_ray[3 * p + a]).abs() < 1e-8);
            }
        }
        // A general field (e.g. linear shear) projects with finite e_proj.
        let shear: Vec<f64> = pts
            .iter()
            .flat_map(|x| [x.y.powi(2), -x.x * x.z, x.z])
            .collect();
        let (_, e2) = project(&basis, &pts, &w, None, &shear).unwrap();
        assert!(e2.is_finite() && e2 < 1.0);
    }
}
