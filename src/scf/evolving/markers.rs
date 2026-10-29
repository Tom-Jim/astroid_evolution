//! Lagrangian state markers (§11.1) and the nonlocal plastic driving rate (§11.2).
//!
//! Markers carry the solid fraction `ν` (A-Ξ) and the damage `ϖ` (A-ϖ) with the material.
//! Values at evaluation points come from a visible, weighted linear moving least-squares
//! fit, clipped to the admissible range (Lemma 11.2).

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::common::error::SimError;
use crate::common::math::DVec3;
use crate::scf::surface::{PointGrid, SurfaceIndex};

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MarkerParams {
    /// Reconstruction radius `h_R` as a fraction of the mean radius.
    pub radius: f64,
    /// Coverage floor `n_min` and merge ceiling `n_max` (Definition 11.1).
    pub n_min: usize,
    pub n_max: usize,
    /// `c_ℓ ≥ 1` of the nonlocal length `ℓ_p = c_ℓ π R_* / L_max` (Definition 11.3).
    pub nonlocal: f64,
}

impl MarkerParams {
    /// `ℓ_p = c_ℓ π R_* / L_max`, with `L_max` the degree of the velocity basis.
    pub fn nonlocal_length(&self, r_star: f64, l_velocity: usize) -> f64 {
        self.nonlocal * std::f64::consts::PI * r_star / l_velocity as f64
    }

    pub fn validate(&self) -> Result<(), SimError> {
        if !(self.radius.is_finite() && self.radius > 0.0 && self.radius < 1.0) {
            return Err(SimError::model_not_closed("markers.radius must lie in (0, 1)"));
        }
        if self.n_min < 4 || self.n_max <= self.n_min {
            return Err(SimError::model_not_closed(
                "markers need n_min >= 4 and n_max > n_min",
            ));
        }
        if !(self.nonlocal.is_finite() && self.nonlocal >= 1.0) {
            return Err(SimError::model_not_closed("markers.nonlocal (c_ℓ) must be at least 1"));
        }
        Ok(())
    }
}

/// The marker set: positions and the two state fields.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Markers {
    pub pos: Vec<[f64; 3]>,
    pub nu: Vec<f64>,
    pub damage: Vec<f64>,
}

/// Admissible interval of a field (clip target of Lemma 11.2).
#[derive(Clone, Copy, Debug)]
pub struct Range {
    pub lo: f64,
    pub hi: f64,
}

/// Value and gradient of a reconstructed field.
#[derive(Clone, Copy, Debug, Default)]
pub struct Fit {
    pub value: f64,
    pub grad: DVec3,
}

impl Markers {
    pub fn len(&self) -> usize {
        self.pos.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pos.is_empty()
    }

    /// One marker per volume node, with the initial state (step 0 of Definition 11.1).
    pub fn seed(pts: &[DVec3], nu: f64, damage: f64) -> Self {
        Self {
            pos: pts.iter().map(|p| p.to_array()).collect(),
            nu: vec![nu; pts.len()],
            damage: vec![damage; pts.len()],
        }
    }

    pub fn points(&self) -> Vec<DVec3> {
        self.pos.iter().map(|p| DVec3::from_array(*p)).collect()
    }

    /// Rigid motion of step 8b: `y ← Q (y − x_cm)`.
    pub fn realign(&mut self, q: [[f64; 3]; 3], t: DVec3) {
        for p in &mut self.pos {
            let y = DVec3::from_array(*p) - t;
            *p = std::array::from_fn(|i| q[i][0] * y.x + q[i][1] * y.y + q[i][2] * y.z);
        }
    }
}

/// The reconstruction `R` of §11.1 on a fixed marker set and geometry.
pub struct Reconstruction<'a> {
    pts: Vec<DVec3>,
    grid: PointGrid,
    index: &'a SurfaceIndex,
    h: f64,
}

/// `w(s) = (1 − s²)⁴` on `[0, 1)`.
fn weight(s: f64) -> f64 {
    let a = (1.0 - s * s).max(0.0);
    a * a * a * a
}

/// The segment `x → y` stays inside the body: a ball of radius `depth` about `x` does,
/// otherwise one any-hit query.
fn visible(index: &SurfaceIndex, x: DVec3, depth: f64, y: DVec3) -> bool {
    let d = y - x;
    d.length() <= depth || index.first_hit(x, d, 1e-9, 1.0 - 1e-9).is_none()
}

impl<'a> Reconstruction<'a> {
    pub fn new(markers: &Markers, index: &'a SurfaceIndex, h: f64) -> Self {
        Self::on_points(markers.points(), index, h)
    }

    /// The same weights on any point set (the volume nodes of Definition 11.3).
    pub fn on_points(pts: Vec<DVec3>, index: &'a SurfaceIndex, h: f64) -> Self {
        Self {
            grid: PointGrid::build(&pts),
            pts,
            index,
            h,
        }
    }

    /// Visible Shepard means of the point fields `z` at `x`: weights `w(|x − y| / h)` over
    /// the visible points within `h`, and the nearest point's value where there is none.
    pub fn shepard_at(&self, x: &[DVec3], fields: &[&[f64]]) -> Vec<Vec<f64>> {
        let rows: Vec<Vec<f64>> = x
            .par_iter()
            .map(|p| {
                let sup = self.support(*p);
                let wsum: f64 = sup.iter().map(|s| s.1).sum();
                if wsum > 0.0 {
                    fields.iter().map(|z| sup.iter().map(|&(j, w)| w * z[j]).sum::<f64>() / wsum).collect()
                } else {
                    let j = self.grid.nearest(*p);
                    fields.iter().map(|z| z[j]).collect()
                }
            })
            .collect();
        (0..fields.len()).map(|k| rows.iter().map(|r| r[k]).collect()).collect()
    }

    /// Visible markers within `h_R` of `x`, with their weights.
    fn support(&self, x: DVec3) -> Vec<(usize, f64)> {
        let depth = self.index.closest(x).dist2.sqrt();
        self.grid
            .within(x, self.h)
            .into_iter()
            .filter(|(_, j)| visible(self.index, x, depth, self.pts[*j]))
            .map(|(d, j)| (j, weight(d / self.h)))
            .collect()
    }

    /// Linear MLS of the marker values `z` at every point, clipped to `range`; the
    /// Shepard mean (zero gradient) where the weighted design is singular. Also returns
    /// the number of points that fell back.
    pub fn fit(&self, pts: &[DVec3], fields: &[(&[f64], Range)]) -> (Vec<Vec<Fit>>, usize) {
        let rows: Vec<(Vec<Fit>, bool)> = pts
            .par_iter()
            .map(|x| {
                let sup = self.support(*x);
                if sup.is_empty() {
                    // No visible marker: the nearest one (always defined), constant.
                    let j = self.grid.nearest(*x);
                    return (
                        fields.iter().map(|(z, _)| Fit { value: z[j], grad: DVec3::ZERO }).collect(),
                        true,
                    );
                }
                // Normal equations of a + b·(y − x).
                let mut m = [[0.0f64; 4]; 4];
                for &(j, w) in &sup {
                    let r = self.pts[j] - *x;
                    let b = [1.0, r.x / self.h, r.y / self.h, r.z / self.h];
                    for a in 0..4 {
                        for c in 0..4 {
                            m[a][c] += w * b[a] * b[c];
                        }
                    }
                }
                let inv = invert4(&m);
                let fits = fields
                    .iter()
                    .map(|(z, range)| {
                        let wsum: f64 = sup.iter().map(|s| s.1).sum();
                        let shepard = sup.iter().map(|&(j, w)| w * z[j]).sum::<f64>() / wsum;
                        match &inv {
                            Some(inv) => {
                                let mut rhs = [0.0f64; 4];
                                for &(j, w) in &sup {
                                    let r = self.pts[j] - *x;
                                    let b = [1.0, r.x / self.h, r.y / self.h, r.z / self.h];
                                    for a in 0..4 {
                                        rhs[a] += w * b[a] * z[j];
                                    }
                                }
                                let s: [f64; 4] =
                                    std::array::from_fn(|a| (0..4).map(|c| inv[a][c] * rhs[c]).sum());
                                let inside = s[0] > range.lo && s[0] < range.hi;
                                Fit {
                                    value: s[0].clamp(range.lo, range.hi),
                                    grad: if inside { DVec3::new(s[1], s[2], s[3]) / self.h } else { DVec3::ZERO },
                                }
                            }
                            None => Fit { value: shepard.clamp(range.lo, range.hi), grad: DVec3::ZERO },
                        }
                    })
                    .collect();
                (fits, inv.is_none())
            })
            .collect();
        let fallbacks = rows.iter().filter(|r| r.1).count();
        let n = fields.len();
        let mut out = vec![Vec::with_capacity(pts.len()); n];
        for (fits, _) in rows {
            for (k, f) in fits.into_iter().enumerate() {
                out[k].push(f);
            }
        }
        (out, fallbacks)
    }

    /// Shepard mean of the visible markers within `2 h_R` (`None` when there is none).
    pub fn shepard(&self, x: DVec3, z: &[f64]) -> Option<f64> {
        let depth = self.index.closest(x).dist2.sqrt();
        let near: Vec<(usize, f64)> = self
            .grid
            .within(x, 2.0 * self.h)
            .into_iter()
            .filter(|(_, j)| visible(self.index, x, depth, self.pts[*j]))
            .map(|(d, j)| (j, weight(d / (2.0 * self.h))))
            .collect();
        let w: f64 = near.iter().map(|s| s.1).sum();
        (w > 0.0).then(|| near.iter().map(|&(j, a)| a * z[j]).sum::<f64>() / w)
    }

    /// Visible markers within `h_R` of `x`.
    pub fn coverage(&self, x: DVec3) -> usize {
        self.support(x).len()
    }
}

/// Inverse of a symmetric 4 × 4 matrix by Gauss–Jordan; `None` when it is singular
/// relative to its diagonal (fewer than four markers off a plane).
fn invert4(m: &[[f64; 4]; 4]) -> Option<[[f64; 4]; 4]> {
    let scale = (0..4).map(|i| m[i][i]).fold(0.0, f64::max);
    if !(scale > 0.0) {
        return None;
    }
    let mut a = *m;
    let mut inv = [[0.0; 4]; 4];
    for (i, row) in inv.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    for c in 0..4 {
        let p = (c..4).max_by(|x, y| a[*x][c].abs().total_cmp(&a[*y][c].abs()))?;
        if a[p][c].abs() < 1e-10 * scale {
            return None;
        }
        a.swap(c, p);
        inv.swap(c, p);
        let d = a[c][c];
        for k in 0..4 {
            a[c][k] /= d;
            inv[c][k] /= d;
        }
        for r in 0..4 {
            if r != c {
                let f = a[r][c];
                for k in 0..4 {
                    a[r][k] -= f * a[c][k];
                    inv[r][k] -= f * inv[c][k];
                }
            }
        }
    }
    Some(inv)
}

/// The driving rate of Definition 11.3 on one geometry: `γ̇_p^⋆` and `I_c` on the volume
/// nodes, where `F_δ` is known, carried to any point by the visible Shepard mean.
pub struct Driving<'a> {
    /// `γ̇_{p,q}^⋆` at the volume nodes.
    pub rate: Vec<f64>,
    inertial: Vec<f64>,
    rec: Reconstruction<'a>,
}

impl<'a> Driving<'a> {
    /// `plastic_rate` and `inertial` are `γ̇_p` and `I_c` at the nodes `pts` (weights `w`);
    /// `ell` is `ℓ_p` and `h` the reconstruction radius `h_R`.
    pub fn new(
        pts: &[DVec3],
        w: &[f64],
        plastic_rate: &[f64],
        inertial: &[f64],
        ell: f64,
        h: f64,
        index: &'a SurfaceIndex,
    ) -> Self {
        // A node's own plastic rate is positive exactly where its F_δ is (the mask).
        let mask: Vec<bool> = plastic_rate.iter().map(|r| *r > 0.0).collect();
        let rate = Nonlocal::new(pts, w, ell, index).driving(plastic_rate, w, pts, &mask);
        Self {
            rate,
            inertial: inertial.to_vec(),
            rec: Reconstruction::on_points(pts.to_vec(), index, h),
        }
    }

    /// `γ̇_p^⋆` and `I_c` at `x`.
    pub fn at(&self, x: &[DVec3]) -> (Vec<f64>, Vec<f64>) {
        if self.rate.iter().all(|r| *r == 0.0) {
            // Nothing yields: no state moves, and I_c enters only through a zero rate.
            return (vec![0.0; x.len()], vec![0.0; x.len()]);
        }
        let mut out = self.rec.shepard_at(x, &[&self.rate, &self.inertial]).into_iter();
        (out.next().expect("rate"), out.next().expect("I_c"))
    }
}

/// The nonlocal average of Definition 11.3 on a fixed geometry. The volume nodes are
/// aggregated into cubic cells of side at most `ℓ_p / 4` (weighted centroids and weights),
/// an `O((h_cell/ℓ_p)²)` quadrature of the same integral.
pub struct Nonlocal<'a> {
    ell: f64,
    index: &'a SurfaceIndex,
    /// Cell of every volume node, the cell weights and centroids.
    cell_of: Vec<usize>,
    weight: Vec<f64>,
    centre: Vec<DVec3>,
    grid: PointGrid,
}

impl<'a> Nonlocal<'a> {
    pub fn new(pts: &[DVec3], w: &[f64], ell: f64, index: &'a SurfaceIndex) -> Self {
        let side = ell / 4.0;
        let key = |p: DVec3| ((p / side).floor().as_ivec3()).to_array();
        let mut cells = std::collections::HashMap::<[i32; 3], usize>::new();
        let mut cell_of = Vec::with_capacity(pts.len());
        let (mut weight, mut centre) = (Vec::new(), Vec::new());
        for (p, wq) in pts.iter().zip(w) {
            let n = cells.len();
            let c = *cells.entry(key(*p)).or_insert(n);
            if c == weight.len() {
                weight.push(0.0);
                centre.push(DVec3::ZERO);
            }
            weight[c] += wq;
            centre[c] += *p * *wq;
            cell_of.push(c);
        }
        for (c, w) in centre.iter_mut().zip(&weight) {
            *c /= w.max(1e-300);
        }
        Self {
            ell,
            index,
            cell_of,
            grid: PointGrid::build(&centre),
            weight,
            centre,
        }
    }

    /// `γ̇_p^⋆ = 1{F_δ>0} γ̄̇_p` at `pts`; `yielded` gives the mask of each point and `rate`
    /// the local plastic rate at the volume nodes.
    pub fn driving(&self, rate: &[f64], w: &[f64], pts: &[DVec3], yielded: &[bool]) -> Vec<f64> {
        let mut num = vec![0.0; self.weight.len()];
        for (q, r) in rate.iter().enumerate() {
            num[self.cell_of[q]] += w[q] * r;
        }
        // Cell means of γ̇_p.
        let mean: Vec<f64> = num.iter().zip(&self.weight).map(|(n, w)| n / w.max(1e-300)).collect();
        if mean.iter().all(|m| *m == 0.0) {
            return vec![0.0; pts.len()];
        }
        pts.par_iter()
            .zip(yielded)
            .map(|(x, y)| {
                if !*y {
                    return 0.0;
                }
                let depth = self.index.closest(*x).dist2.sqrt();
                let (mut a, mut b) = (0.0, 0.0);
                for (d, c) in self.grid.within(*x, self.ell) {
                    let s = d / self.ell;
                    if !visible(self.index, *x, depth, self.centre[c]) {
                        continue;
                    }
                    let k = (1.0 - s * s).powi(2) * self.weight[c];
                    a += k * mean[c];
                    b += k;
                }
                if b > 0.0 { a / b } else { 0.0 }
            })
            .collect()
    }
}
