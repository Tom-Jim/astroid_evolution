//! Quasi-static Stokes–pressure solve of the problem of §10.3.2 on the general domain
//! of hypothesis (G) (§8.1).
//!
//! * velocity: the (G1) geometry-adapted family — compact divergence-free
//!   kernels on the volume and boundary nodes, the six rigid modes, and one
//!   flux mode per sealed cavity — evaluated on the GPU (`gbasis.wesl`) and
//!   integrated by the ray-segment rule (G2);
//! * pressure: the scalar Poisson problem with the free-surface datum `p_Γ` imposed by
//!   Nitsche's method (G3), whose matrix is factored once per geometry;
//! * the pair is closed by the characteristic pseudo-time relaxation of
//!   §10.3.2 (8.3-H, [`hyperbolic`]); the primal–dual iteration survives only
//!   as the `tier2` oracle ([`pd`]);
//! * the certified bracket of the limit load of §10.0.12.3 ([`limit`]).

pub mod elastic;
pub mod hyperbolic;
pub mod limit;
/// The primal–dual reference solver: test oracle only, compiled under `tier2`.
/// The file lives with the other test assets under `tests/support/`.
#[cfg(feature = "tier2")]
#[path = "../../../tests/support/pd_oracle.rs"]
mod pd;
mod pressure;

use std::f64::consts::PI;

use nalgebra::{Cholesky, DMatrix, DVector, Dyn, LU};

use crate::common::density::CauchyDensity;
use crate::common::math::DVec3;
use crate::rheo::Material;
use crate::rheo::closure::dynamic_friction;
use crate::rtfp::mesh::BodyMesh;

use super::basis::{ContractedVelocity, PresLayout, VelRows, Want, eval_pressure};
use super::domain::{SurfacePoints, VolumeSet};
use super::evolving::dilatation::Dilatational;
use super::gbasis::{self, GeoBasis, GramReport, Mode};
use super::harmonics::SurfaceFit;
use super::params::StokesParams;
use super::surface::SurfaceIndex;
use crate::common::error::SimError;
use crate::common::linalg::{self, Entry};

/// The primal–dual reference solver: test oracle only, compiled under `tier2`.
#[cfg(feature = "tier2")]
pub use pd::solve_pd;

/// Values and gradients of the pressure space `ℚ_h` at a point set: the interior family
/// `(1 − r/R_*) ψ_{lk} Y_{lm}` and the family `(r/R_*)^l Y_{lm}`, all free under (G3).
struct QRows<T: Entry = f64> {
    val: DMatrix<T>,
    grad: DMatrix<T>,
}

impl<T: Entry> QRows<T> {
    fn eval(surf: &SurfaceFit, play: &PresLayout, pts: &[DVec3]) -> Self {
        let r = eval_pressure(surf, play, pts, true);
        let a = play.n_p();
        // Interior columns, then the lift columns, stored as `T`.
        let join = |inner: &DMatrix<f64>, lift: &DMatrix<f64>| {
            DMatrix::from_fn(inner.nrows(), a + lift.ncols(), |i, j| {
                T::of(if j < a {
                    inner[(i, j)]
                } else {
                    lift[(i, j - a)]
                })
            })
        };
        Self {
            val: join(&r.val, &r.lift_val),
            grad: join(&r.grad, &r.lift_grad),
        }
    }

    /// `∂_n q` at each point.
    fn normal(&self, normals: &[DVec3]) -> DMatrix<f64> {
        let mut dn = DMatrix::zeros(normals.len(), self.val.ncols());
        for (p, n) in normals.iter().enumerate() {
            for c in 0..dn.ncols() {
                dn[(p, c)] = (0..3)
                    .map(|a| n[a] * self.grad[(3 * p + a, c)].into())
                    .sum();
            }
        }
        dn
    }
}

/// Where a point's basis rows come from when a geometry is rebuilt: point `i` of the
/// previous volume rule, or point `j` of the newly evaluated ones.
#[derive(Clone, Copy)]
enum Source {
    Old(usize),
    New(usize),
}

/// Relative distance `δ / R_ref` within which a node takes the rows of an old node: the
/// integrand of that node is then off by `O(L δ / R_ref)` (§8.1), below 1e-4 relative.
const REUSE_DISTANCE: f64 = 1.0e-5;

/// For every mode of `new`, the index of the old-family mode that is the
/// same function up to `O(δ)`: same kind and discrete parameters, kernel or
/// flux centre inside the reuse reach. `None` — the column is a fresh entry.
fn mode_map(new: &[Mode], old: &[Mode], reach: f64) -> Vec<Option<usize>> {
    let near = |a: &DVec3, b: &DVec3| a.distance(*b) <= reach;
    new.iter()
        .map(|m| {
            old.iter().position(|o| match (m, o) {
                (
                    Mode::Kernel { c, e, h },
                    Mode::Kernel {
                        c: oc,
                        e: oe,
                        h: oh,
                    },
                ) => e == oe && (*h - *oh).abs() <= 2.0e-2 * *h && near(c, oc),
                (Mode::Rigid { a, w }, Mode::Rigid { a: oa, w: ow }) => a == oa && w == ow,
                (Mode::Flux { z }, Mode::Flux { z: oz }) => near(z, oz),
                (Mode::Poly { e, ijk }, Mode::Poly { e: oe, ijk: oijk }) => e == oe && ijk == oijk,
                _ => false,
            })
        })
        .collect()
}

/// Each point of `pts` mapped to the nearest point of `old` within `reach`, or to the
/// list of points to evaluate.
fn reuse_map(pts: &[DVec3], old: &[DVec3], reach: f64) -> (Vec<Source>, Vec<DVec3>) {
    let cell = |p: &DVec3| (*p / reach).floor().as_i64vec3().to_array();
    let mut grid: std::collections::HashMap<[i64; 3], Vec<usize>> =
        std::collections::HashMap::new();
    for (i, p) in old.iter().enumerate() {
        grid.entry(cell(p)).or_default().push(i);
    }
    let mut fresh = Vec::new();
    let src = pts
        .iter()
        .map(|p| {
            let c = cell(p);
            let mut best: Option<(f64, usize)> = None;
            for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        for &i in grid
                            .get(&[c[0] + dx, c[1] + dy, c[2] + dz])
                            .into_iter()
                            .flatten()
                        {
                            let d = (old[i] - *p).length();
                            if d <= reach && best.is_none_or(|b| d < b.0) {
                                best = Some((d, i));
                            }
                        }
                    }
                }
            }
            match best {
                Some((_, i)) => Source::Old(i),
                None => {
                    fresh.push(*p);
                    Source::New(fresh.len() - 1)
                }
            }
        })
        .collect();
    (src, fresh)
}

/// Rows of `stride` per point, each taken from the old or the new block: one contiguous
/// copy per point and column, the columns in parallel.
fn gather<T: Entry + Send + Sync>(
    old: &DMatrix<T>,
    new: &DMatrix<T>,
    src: &[Source],
    stride: usize,
) -> DMatrix<T> {
    use rayon::prelude::*;
    let cols = old.ncols().max(new.ncols());
    let rows = stride * src.len();
    let mut out = DMatrix::from_element(rows, cols, T::of(0.0));
    if rows == 0 {
        return out;
    }
    let (no, nn) = (old.nrows(), new.nrows());
    out.as_mut_slice()
        .par_chunks_mut(rows)
        .enumerate()
        .for_each(|(c, col)| {
            for (p, s) in src.iter().enumerate() {
                let (from, at) = match *s {
                    Source::Old(i) => (&old.as_slice()[c * no..(c + 1) * no], stride * i),
                    Source::New(j) => (&new.as_slice()[c * nn..(c + 1) * nn], stride * j),
                };
                col[stride * p..stride * (p + 1)].copy_from_slice(&from[at..at + stride]);
            }
        });
    out
}

/// What the next geometry takes over from this one: the basis scale `R_ref` (§8.1), the
/// (G1) family itself, and the volume rows by point. The retained basis is a function of
/// the whole geometry — the thinned kernel centres and the pruned mode set — so the
/// velocity rows are only ever taken over wholesale, when the basis input is identical;
/// rows of another basis are other functions and are never gathered point-wise. The
/// pressure rows keep their point-wise reuse: they depend on `surf` alone.
pub struct Reuse {
    pub r_ref: f64,
    pts: Vec<DVec3>,
    bnd_pts: Vec<DVec3>,
    rho: Vec<f64>,
    cavity: Vec<DVec3>,
    geo: GeoBasis,
    geo_gram: GramReport,
    rows: VelRows<f32>,
    q_vol: QRows<f32>,
}

/// Scaled strain rows `D` at the face centres: the basis the friction chain
/// contracts every iteration.
pub struct FaceStrain {
    d: DMatrix<f32>,
}

impl FaceStrain {
    /// `D(Bu)` at each stored point, six scaled components, point-major.
    pub fn contract(&self, u: &[f64]) -> Result<Vec<f64>, String> {
        Ok(linalg::mul(&self.d, &DVector::from_column_slice(u))
            .as_slice()
            .to_vec())
    }
}

/// Geometry-dependent operators, built once per geometry and density.
pub struct Operators {
    /// The enclosing-ball fit `R ≡ R_*`: the basis scale and the pressure
    /// quadrature of (G2).
    pub surf: SurfaceFit,
    /// The retained (G1) mode count: `ops.rows` column count.
    pub n_u: usize,
    pub play: PresLayout,
    pub vol: VolumeSet,
    /// The boundary rule of the Nitsche terms.
    pub bnd: SurfacePoints,
    /// Face centres of `Γ`: the nodes of the friction chain and of `p_Γ`.
    pub faces: SurfacePoints,
    /// Volume rows in f32: the largest store of the solve (30 rows per point).
    pub rows: VelRows<f32>,
    bnd_rows: VelRows<f32>,
    bnd_rho: Vec<f64>,
    bnd_grho: Vec<DVec3>,
    q_vol: QRows<f32>,
    q_bnd: QRows,
    q_bnd_dn: DMatrix<f64>,
    nitsche: Cholesky<f64, Dyn>,
    /// `γ_N / h_N` and `h_N = R_*`; `C_N` of Proposition 8.1.6 is reported.
    gamma_over_h: f64,
    pub c_nitsche: f64,
    /// Strain basis at the face centres.
    pub surf_d: FaceStrain,
    pub rigid: DMatrix<f64>,
    /// Smallest singular value of the row-normalised rigid-body constraint matrix.
    pub rigid_sigma_min: f64,
    /// The classification mesh `X_Y` of (G7).
    pub class_mesh: std::sync::OnceLock<super::classify::ClassMesh>,
    /// Gravity at the volume and boundary points; independent of the spin.
    pub vol_field: std::sync::OnceLock<super::forward::PointFields>,
    pub bnd_field: std::sync::OnceLock<super::forward::PointFields>,
    /// The stiffness metric `G` at `η ≡ 1` and its range-space factorization:
    /// fixed by the geometry, so a spin-only step reuses them.
    pd_metric: std::sync::OnceLock<(DMatrix<f64>, RangeSolve)>,
    /// The (G1) geometry-adapted family: compact kernels on the volume nodes
    /// and face quadrature points, the six rigid modes, and one flux mode per
    /// sealed cavity. It is the Galerkin-projection target of §8.3 and the
    /// trial family of `stokes.basis = "geometry-adapted"`.
    pub geo: GeoBasis,
    /// The Prop 8.1.3 report on `geo`: σ_min, rank, conditioning of `M_Γ`,
    /// assembled on the (G2) volume quadrature after `gbasis::prune`.
    pub geo_gram: GramReport,
}

impl Operators {
    #[allow(clippy::too_many_arguments)]
    /// `v` and the scaled strain six-vector `D` at the volume nodes, from the host rows.
    pub fn vol_velocity(&self, u: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let u = DVector::from_column_slice(u);
        let v = linalg::mul(&self.rows.v, &u);
        let d = linalg::mul(&self.rows.d, &u);
        (v.as_slice().to_vec(), d.as_slice().to_vec())
    }

    /// The pressure `P c + lift` at the volume nodes, from the host rows.
    pub fn vol_pressure(&self, pres_c: &[f64], lift: &[f64]) -> Vec<f64> {
        let coeff = DVector::from_iterator(
            pres_c.len() + lift.len(),
            pres_c.iter().chain(lift).copied(),
        );
        linalg::mul(&self.q_vol.val, &coeff).as_slice().to_vec()
    }

    /// The rows and scale the next geometry can reuse; the rest is released.
    pub fn into_reuse(self) -> Reuse {
        Reuse {
            r_ref: self.surf.coeff[0] / (4.0 * PI).sqrt(),
            pts: self.vol.pts,
            rho: self.vol.rho,
            bnd_pts: self.bnd.pts,
            cavity: self.geo.cavity_centres.clone(),
            geo: self.geo,
            geo_gram: self.geo_gram,
            rows: self.rows,
            q_vol: self.q_vol,
        }
    }

    /// `prev` (same basis scale) supplies the rows of the volume nodes that moved by at
    /// most `REUSE_DISTANCE · R_ref` (the crossings of most rays barely move in a step).
    pub fn build(
        surf: &SurfaceFit,
        density: &CauchyDensity,
        mesh: &BodyMesh,
        index: &SurfaceIndex,
        faces: SurfacePoints,
        prm: &StokesParams,
        prev: Option<Reuse>,
        cavity_centres: &[DVec3],
    ) -> Result<Self, SimError> {
        let play = PresLayout {
            l_max: prm.l_pressure,
            k_max: prm.k_pressure,
        };
        let span = super::timing::span("operators_rule");
        let vol = VolumeSet::build(
            mesh,
            index,
            density,
            prm.n_theta,
            prm.n_phi,
            prm.n_radial,
            prm.cut_depth,
        )?;
        let face_triples: Vec<(DVec3, DVec3, f64)> = (0..faces.pts.len())
            .map(|f| (faces.pts[f], faces.normals[f], faces.w[f]))
            .collect();
        let bnd = SurfacePoints::clustered(&face_triples, prm.n_boundary);
        drop(span);
        // Rows of another basis scale are other functions: they are never taken over.
        let h_n = surf.coeff[0] / (4.0 * PI).sqrt();
        let prev = prev.filter(|old| (old.r_ref - h_n).abs() <= 1e-12 * h_n);
        // The (G1) family is a deterministic function of the point sets, the
        // density and the cavity list: an identical input retains identical
        // pruned modes, and only then are the old rows themselves rows of this
        // basis — anything else takes over entries mode-wise below.
        let same_input = prev.as_ref().is_some_and(|o| {
            o.pts == vol.pts
                && o.bnd_pts == bnd.pts
                && o.rho == vol.rho
                && o.cavity.as_slice() == cavity_centres
        });
        // (G1): the geometry-adapted family on this mesh. Kernel centres are
        // thinned so the Prop 8.1.3 Gram check stays a bounded dense solve;
        // `h_kernel` is the CellBVH support diameter `h_vol` of §11.2. Every
        // velocity row below is the GPU evaluation of this family.
        let (geo, geo_gram) = if same_input {
            let o = prev.as_ref().unwrap();
            (o.geo.clone(), o.geo_gram.clone())
        } else {
            let centres = thin(&vol.pts, &bnd.pts, MAX_GEO_CENTRES);
            let h = geo_support(&centres);
            let basis = GeoBasis::build(&centres, &[], h, cavity_centres.to_vec());
            gbasis::prune(
                basis,
                &vol.pts,
                &vol.w,
                Some(&vol.rho),
                1e-8,
                centres.len().max(8),
            )
        };
        if geo_gram.rank < 6 {
            return Err(SimError::solver(
                "the (G1) basis lost the rigid modes on this geometry",
            ));
        }
        let n_u = geo.n();
        let g_rows = |pts: &[DVec3]| -> Result<VelRows<f32>, SimError> {
            super::gbasis_gpu::rows(&geo, pts, Want::ALL).map_err(SimError::gpu)
        };
        let span = super::timing::span("operators_volume_rows");
        let (rows, q_vol) = match prev {
            Some(old) if same_input => (old.rows, old.q_vol),
            Some(old) => {
                let (src, fresh) = reuse_map(&vol.pts, &old.pts, REUSE_DISTANCE * old.r_ref);
                // The pressure family depends on `surf` alone: its rows are
                // gathered point-wise within the reuse reach either way.
                let q = QRows::<f32>::eval(surf, &play, &fresh);
                let _gather = super::timing::span("operators_volume_gather");
                let q_vol = QRows {
                    val: gather(&old.q_vol.val, &q.val, &src, 1),
                    grad: gather(&old.q_vol.grad, &q.grad, &src, 3),
                };
                // A node inside the reuse reach takes, for every mode that is
                // the same function up to `O(δ)` — same kind and discrete
                // parameters, centre inside the reach — the row entries of
                // its old node (§8.1); every other entry is the fresh one.
                let mm = mode_map(&geo.modes, &old.geo.modes, REUSE_DISTANCE * old.r_ref);
                let mut rows = g_rows(&vol.pts)?;
                for (i, s) in src.iter().enumerate() {
                    let Source::Old(j) = *s else { continue };
                    for (b, mk) in mm.iter().enumerate() {
                        let Some(k) = *mk else { continue };
                        for a in 0..3 {
                            rows.v[(3 * i + a, b)] = old.rows.v[(3 * j + a, k)];
                        }
                        for c in 0..6 {
                            rows.d[(6 * i + c, b)] = old.rows.d[(6 * j + c, k)];
                        }
                        for a in 0..3 {
                            rows.lap[(3 * i + a, b)] = old.rows.lap[(3 * j + a, k)];
                        }
                        for c in 0..18 {
                            rows.gd[(18 * i + c, b)] = old.rows.gd[(18 * j + c, k)];
                        }
                    }
                }
                (rows, q_vol)
            }
            None => (g_rows(&vol.pts)?, QRows::eval(surf, &play, &vol.pts)),
        };
        let rigid = rigid_matrix(&vol, &rows, n_u);
        let rigid_sigma_min = rigid_rank(&rigid, prm.rigid_sigma_min)?;
        drop(span);
        let span = super::timing::span("operators_boundary_rows");
        let bnd_rows = g_rows(&bnd.pts)?;
        let surf_d = FaceStrain {
            d: super::gbasis_gpu::rows(&geo, &faces.pts, Want::D)
                .map_err(SimError::gpu)?
                .d,
        };
        drop(span);
        let _span = super::timing::span("operators_nitsche");

        // Nitsche matrix a_N (G3) with h_N = R_* and γ_N = 4 C_N.
        let q_bnd = QRows::eval(surf, &play, &bnd.pts);
        let q_bnd_dn = q_bnd.normal(&bnd.normals);
        let stiff = linalg::weighted_gram(&q_vol.grad, 3, &vol.w);
        let weighted = |m: &DMatrix<f64>| {
            let mut out = m.clone();
            for (p, w) in bnd.w.iter().enumerate() {
                out.row_mut(p).scale_mut(*w);
            }
            out
        };
        let dn_w = weighted(&q_bnd_dn);
        let flux = dn_w.tr_mul(&q_bnd.val);
        let mass = weighted(&q_bnd.val).tr_mul(&q_bnd.val);
        let c_nitsche = nitsche_constant(&stiff, &(dn_w.tr_mul(&q_bnd_dn) * h_n));
        let gamma_over_h = (4.0 * c_nitsche).max(1.0) / h_n;
        let a_n = &stiff - &flux - flux.transpose() + mass * gamma_over_h;
        let nitsche = Cholesky::new(a_n).ok_or_else(|| {
            SimError::solver("the Nitsche pressure matrix is not positive definite")
        })?;
        Ok(Self {
            surf: surf.clone(),
            n_u,
            play,
            bnd_rho: bnd.pts.iter().map(|p| density.rho(*p)).collect(),
            bnd_grho: bnd.pts.iter().map(|p| density.grad(*p)).collect(),
            vol,
            bnd,
            faces,
            rows,
            bnd_rows,
            q_vol,
            q_bnd,
            q_bnd_dn,
            nitsche,
            gamma_over_h,
            c_nitsche,
            surf_d,
            rigid,
            rigid_sigma_min,
            class_mesh: std::sync::OnceLock::new(),
            vol_field: std::sync::OnceLock::new(),
            bnd_field: std::sync::OnceLock::new(),
            pd_metric: std::sync::OnceLock::new(),
            geo,
            geo_gram,
        })
    }

    /// The Galerkin projection bridge of §8.3: the (G1) coefficients and the
    /// reported `e_proj` of a sample field `v_ray` (point-major `3n` on the
    /// volume nodes). A mass-Gram solve on the GPU `v` rows.
    pub fn project_velocity(&self, v_ray: &[f64]) -> Result<(Vec<f64>, f64), SimError> {
        let s: Vec<f64> = self
            .vol
            .w
            .iter()
            .zip(&self.vol.rho)
            .map(|(w, r)| w * r)
            .collect();
        let gram = linalg::weighted_gram(&self.rows.v, 3, &s);
        let mut wr = vec![0.0; 3 * self.vol.pts.len()];
        for p in 0..self.vol.pts.len() {
            for a in 0..3 {
                wr[3 * p + a] = s[p] * v_ray[3 * p + a];
            }
        }
        let rhs = linalg::tr_mul(&self.rows.v, &DVector::from_column_slice(&wr));
        // Jacobi-preconditioned solve: the family mixes scales a rigid
        // rotation's Gram entry is ~R² against a kernel's — so the raw Gram
        // is stiff even when the functions are independent. Solving on the
        // correlation system `(D^{-1/2} M D^{-1/2})(D^{1/2}u) = D^{-1/2}b`
        // is the same projection with the scale removed.
        let n = gram.nrows();
        let d: Vec<f64> = (0..n).map(|i| gram[(i, i)].sqrt().max(1e-300)).collect();
        let corr = DMatrix::from_fn(n, n, |b, c| gram[(b, c)] / (d[b] * d[c]));
        let rhs_n = DVector::from_fn(n, |b, _| rhs[b] / d[b]);
        let ch = Cholesky::new(corr)
            .ok_or_else(|| SimError::solver("the velocity mass matrix is not SPD"))?;
        let un = ch.solve(&rhs_n);
        let u = DVector::from_fn(n, |b, _| un[b] / d[b]);
        let vh = linalg::mul(&self.rows.v, &u);
        let (mut num, mut den) = (0.0, 0.0);
        for p in 0..self.vol.pts.len() {
            for a in 0..3 {
                let pw = s[p];
                num += pw * (vh[3 * p + a] - v_ray[3 * p + a]).powi(2);
                den += pw * v_ray[3 * p + a].powi(2);
            }
        }
        Ok((u.as_slice().to_vec(), (num / den.max(1e-300)).sqrt()))
    }

    /// The `want` row blocks of the (G1) basis at `pts`.
    pub fn eval_velocity(&self, pts: &[DVec3], want: Want) -> Result<VelRows<f32>, SimError> {
        super::gbasis_gpu::rows(&self.geo, pts, want).map_err(SimError::gpu)
    }

    /// `Σ u_b B_b` at `pts`: the velocity and scaled strain of one coefficient
    /// vector of this geometry's (G1) basis.
    pub fn contract_velocity(&self, pts: &[DVec3], want: Want, u: &[f64]) -> ContractedVelocity {
        super::gbasis_gpu::contract(&self.geo, pts, want, u)
            .unwrap_or_else(|err| panic!("(G1) contraction GPU: {err}"))
    }
}

/// The kernel support of the (G1) family: about 1.4 mean kernel-centre
/// spacings, so every kernel overlaps its nearest neighbours and the Gram
/// matrix stays the local overlap graph of §8.1.
fn geo_support(centres: &[DVec3]) -> f64 {
    if centres.len() < 2 {
        return 1.0;
    }
    let (mut lo, mut hi) = (centres[0], centres[0]);
    for p in centres {
        lo = lo.min(*p);
        hi = hi.max(*p);
    }
    let d = hi - lo;
    let vol_box = (d.x * d.y * d.z).max(1e-30);
    1.4 * (vol_box / centres.len() as f64).cbrt()
}

/// Cap on kernel centres of the (G1) family per geometry: enough for the
/// approximation of §8.1 while keeping the Prop 8.1.3 Gram check a bounded
/// dense solve on the reference path.
const MAX_GEO_CENTRES: usize = 320;

/// Thin `vol` and `bnd` to at most `cap` centres, striding each list evenly so
/// the retained kernels still cover the whole body and its faces.
fn thin(vol: &[DVec3], bnd: &[DVec3], cap: usize) -> Vec<DVec3> {
    let total = vol.len() + bnd.len();
    if total <= cap {
        return vol.iter().chain(bnd.iter()).copied().collect();
    }
    let take_vol = cap * vol.len() / total;
    let take_bnd = cap - take_vol;
    let stride = |len: usize, take: usize| (len / take.max(1)).max(1);
    let sv = stride(vol.len(), take_vol);
    let sb = stride(bnd.len(), take_bnd);
    vol.iter()
        .step_by(sv)
        .take(take_vol)
        .chain(bnd.iter().step_by(sb).take(take_bnd))
        .copied()
        .collect()
}

/// Smallest singular value of the row-normalised `B_rb`; an error below `sigma_min`.
fn rigid_rank(rigid: &DMatrix<f64>, sigma_min: f64) -> Result<f64, SimError> {
    let mut norm = rigid.clone();
    for a in 0..6 {
        let n = norm.row(a).norm();
        if !(n.is_finite() && n > 0.0) {
            return Err(SimError::rigid_rank("a rigid-body constraint row vanishes"));
        }
        norm.row_mut(a).scale_mut(1.0 / n);
    }
    let smin = (&norm * norm.transpose())
        .symmetric_eigen()
        .eigenvalues
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min)
        .max(0.0)
        .sqrt();
    if smin < sigma_min {
        return Err(SimError::rigid_rank(format!(
            "rank of B_rb is deficient: smallest normalised singular value {smin:.3e}"
        )));
    }
    Ok(smin)
}

/// `C_N = max h_N ‖∂_n q‖²_Γ / ‖∇q‖²_𝒱` (Proposition 8.1.6): the largest eigenvalue of
/// `boundary` relative to `stiff`. Constants (`∇q = 0`, `∂_n q = 0`) are removed by a
/// relative shift of `stiff` that does not change the other eigenvalues materially.
fn nitsche_constant(stiff: &DMatrix<f64>, boundary: &DMatrix<f64>) -> f64 {
    let n = stiff.nrows();
    let shift = 1e-10 * stiff.trace().max(1e-300) / n as f64;
    let Some(l) = Cholesky::new(stiff + DMatrix::identity(n, n) * shift) else {
        return 0.0;
    };
    let li = l
        .l()
        .try_inverse()
        .unwrap_or_else(|| DMatrix::identity(n, n));
    let c = &li * boundary * li.transpose();
    c.symmetric_eigen()
        .eigenvalues
        .iter()
        .copied()
        .fold(0.0, f64::max)
}

/// `B_rb` of §A.3 on explicit rows — the A/B oracle path against another basis.
pub fn rigid_rows(vol: &VolumeSet, rows: &VelRows<f32>, n_u: usize) -> DMatrix<f64> {
    rigid_matrix(vol, rows, n_u)
}

/// `B_rb` of §A.3: rows `∫ρ B·e_a`, `∫ρ (x × B)·e_a`.
fn rigid_matrix(vol: &VolumeSet, rows: &VelRows<f32>, n_u: usize) -> DMatrix<f64> {
    let mut b = DMatrix::zeros(6, n_u);
    for (p, x) in vol.pts.iter().enumerate() {
        let wr = vol.w[p] * vol.rho[p];
        for j in 0..n_u {
            let [vx, vy, vz] = std::array::from_fn(|a| f64::from(rows.v[(3 * p + a, j)]));
            b[(0, j)] += wr * vx;
            b[(1, j)] += wr * vy;
            b[(2, j)] += wr * vz;
            b[(3, j)] += wr * (x.y * vz - x.z * vy);
            b[(4, j)] += wr * (x.z * vx - x.x * vz);
            b[(5, j)] += wr * (x.x * vy - x.y * vx);
        }
    }
    b
}

/// Local strength of the viscoplastic law at a point: the cohesion `c` (§4.3a, under A-ϖ
/// `(1 − ϖ) c`), the static dilatancy shift `ξ = K_Ξ (ν − ν_c⁰)` of the friction (A-Ξ,
/// §11.3), and their gradients for `∇η` (§8.1.3).
#[derive(Clone, Copy, Debug, Default)]
pub struct Local {
    pub c: f64,
    pub xi: f64,
    pub gc: DVec3,
    pub gxi: DVec3,
}

/// The local strengths at the volume nodes, the boundary rule and the face centres.
#[derive(Clone, Debug, Default)]
pub struct Strengths {
    pub vol: Vec<Local>,
    pub bnd: Vec<Local>,
    pub face: Vec<Local>,
}

impl Strengths {
    /// The intact profile of A-C2: `c(d)` and `∇c = c'(d) ∇d`, no dilatancy shift. The
    /// boundary and face values are the interior traces at depth 0, `∇d = −n`.
    pub fn profile(ops: &Operators, m: &Material) -> Self {
        let trace = |n: &DVec3| Local {
            c: m.cohesion_at(0.0),
            gc: *n * -m.cohesion_slope(0.0),
            ..Local::default()
        };
        Self {
            vol: (0..ops.vol.pts.len())
                .map(|p| Local {
                    c: m.cohesion_at(ops.vol.depth[p]),
                    gc: ops.vol.gdepth[p] * m.cohesion_slope(ops.vol.depth[p]),
                    ..Local::default()
                })
                .collect(),
            bnd: ops.bnd.normals.iter().map(trace).collect(),
            face: ops.faces.normals.iter().map(trace).collect(),
        }
    }
}

/// The evolving material of Chapter 11 seen by the solver: the local strengths at any
/// points, given the intact ones there.
pub trait MaterialField: Sync {
    fn locals(&self, m: &Material, pts: &[DVec3], intact: &[Local]) -> Vec<Local>;
}

/// The rate-type dilatancy of A-Ξ seen by the solver (§11.4): from the total shear rate,
/// `p̃` and the strengths at the volume nodes, the dilatational field of the next sweeps.
pub trait DilatancySource: Sync {
    fn field(
        &self,
        m: &Material,
        strength: &[Local],
        shear: &[f64],
        ptil: &[f64],
        eps: f64,
    ) -> DilatancyState;
}

/// One update of the dilatational field: `v_ε`, the projection defect `e_ε`, and the
/// largest driving rate `γ̇_p^⋆` (for `Δt_Ξ`).
#[derive(Clone, Debug)]
pub struct DilatancyState {
    pub field: Dilatational,
    pub defect: f64,
    pub driving_max: f64,
}

/// Viscosity of §4 and its partial derivatives, all analytic.
#[derive(Clone, Copy, Debug)]
pub struct EtaEval {
    pub eta: f64,
    pub d_shear: f64,
    pub d_p: f64,
    /// `∂η/∂c` and `∂η/∂ξ`.
    pub d_c: f64,
    pub d_xi: f64,
    pub mu_d: f64,
    pub inertial: f64,
}

/// `η = (c + μ_d p̃) / √(γ̇² + γ̇_ε²)`, `I_c = d_p γ̇ / √((p̃ + p_c)/ρ_p)`, with
/// `μ_d = μ_cs(I_c) + ξ + K_Ξ Δν_c I_c/(I_ν + I_c)` (the last two under A-Ξ only).
pub fn eta_eval(m: &Material, l: &Local, shear: f64, p_tilde: f64, eps: f64) -> EtaEval {
    let s = &m.spec;
    let p_star = p_tilde + m.p_c;
    let (mu_cs, inertial) = dynamic_friction(m, shear, p_star);
    let (extra, d_extra) = m.dilatancy_rate_term(inertial);
    let mu_d = mu_cs + l.xi + extra;
    let scale = (p_star / s.rho_p).sqrt();
    let df = s.d_p * shear;
    let den = s.i_0 * scale + df;
    let dmu = s.mu_2 - s.mu_s;
    // ∂I/∂γ̇ = d_p / scale, ∂I/∂p̃ = −I / (2 p_*).
    let (dmu_dg, dmu_dp) = if den > 0.0 {
        (
            dmu * s.d_p * s.i_0 * scale / (den * den) + d_extra * s.d_p / scale,
            -dmu * df * s.i_0 / (2.0 * scale * s.rho_p) / (den * den)
                - d_extra * inertial / (2.0 * p_star),
        )
    } else {
        (0.0, 0.0)
    };
    let big = (shear * shear + eps * eps).sqrt();
    let a = l.c + mu_d * p_tilde;
    EtaEval {
        eta: a / big,
        d_c: 1.0 / big,
        d_xi: p_tilde / big,
        d_shear: p_tilde * dmu_dg / big - a * shear / (big * big * big),
        d_p: (mu_d + p_tilde * dmu_dp) / big,
        mu_d,
        inertial,
    }
}

/// Loads of one geometric step, frozen for the solve.
pub struct Loads<'a> {
    /// `g_eff` and `T_tide` (`(xx, yy, zz, xy, xz, yz)`) at the volume points.
    pub g_eff: &'a [DVec3],
    /// `∇ × g_eff` at the volume points: zero for the gravitational and tidal
    /// loads of §10.0.12.1 (all gradients), `2 ω̂` of the libration part of a
    /// cycle's δg. The kernel loads are assembled by the curl identity
    /// `∫ρ g·∇×A = ∫(∇×(ρg))·A + ∮ρ g·(n̂×A)`, and `∇×(ρg) = ∇ρ×g + ρ(∇×g)`
    /// needs this slot to be exact on a non-conservative body force.
    pub g_curl: &'a [DVec3],
    pub tide: &'a [[f64; 6]],
    /// The same at the points of the boundary rule.
    pub g_bnd: &'a [DVec3],
    pub tide_bnd: &'a [[f64; 6]],
    /// `n · T_tide · n` at the face centres.
    pub tide_nn: &'a [f64],
    pub g_const: f64,
    /// Local strengths (§4.3a; Chapter 11).
    pub strength: &'a Strengths,
    /// A-Ξ; `None` keeps the velocity exactly divergence-free.
    pub dilatancy: Option<&'a dyn DilatancySource>,
}

#[derive(Clone, Debug)]
pub struct StokesSolution {
    pub u: Vec<f64>,
    pub lambda: [f64; 6],
    /// Pressure coefficients: the interior family, then the `(r/R_*)^l Y_{lm}` family.
    pub pres_c: Vec<f64>,
    pub lift: Vec<f64>,
    /// Final viscosity at the volume points.
    pub eta: Vec<f64>,
    /// Regularised pressure `p̃` at the volume points.
    pub p_tilde: Vec<f64>,
    /// `p_Γ = p_ext + n·τ·n` at the face centres.
    pub p_surface: Vec<f64>,
    pub sweeps: usize,
    pub eps_relax: f64,
    pub eps_momentum: f64,
    pub eps_lambda: f64,
    pub eps_rot: f64,
    pub eta_lo: f64,
    pub eta_hi: f64,
    pub tensile_fraction: f64,
    pub inverse_ekman: f64,
    pub gamma_eps: f64,
    /// A-Ξ: the converged dilatational field (the total velocity is `v_ε + Σ u B`).
    pub dilatancy: Option<DilatancyState>,
    /// `e_proj` of the §8.3 Galerkin bridge: `‖v^ray − v_h‖/‖v^ray‖_{L²(ρ)}`.
    /// `0` for solvers that produce `u` directly.
    pub e_proj: f64,
}

impl StokesSolution {
    /// `v_ε(x)` (zero without A-Ξ).
    pub fn v_eps(&self, x: DVec3) -> DVec3 {
        self.dilatancy
            .as_ref()
            .map_or(DVec3::ZERO, |d| d.field.velocity(x))
    }

    /// `D(v_ε)(x)` as the scaled six-vector, and `ε̇_{v,h}(x)` (zeros without A-Ξ).
    pub fn d_eps(&self, x: DVec3) -> ([f64; 6], f64) {
        self.dilatancy.as_ref().map_or(([0.0; 6], 0.0), |d| {
            (d.field.strain(x), d.field.divergence(x))
        })
    }
}

/// A warm start of the Stokes–pressure pair: the velocity and pressure coefficients of a
/// nearby solution. A part whose length does not fit the current spaces is ignored.
#[derive(Clone, Debug, Default)]
pub struct Guess {
    pub u: Vec<f64>,
    pub p: Vec<f64>,
}

impl StokesSolution {
    pub fn guess(&self) -> Guess {
        Guess {
            u: self.u.clone(),
            p: [self.pres_c.as_slice(), &self.lift].concat(),
        }
    }
}

pub(super) fn d6_dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// `D · v` from the scaled six-vector.
pub(super) fn d_apply(d: &[f64], v: DVec3) -> DVec3 {
    let s = std::f64::consts::FRAC_1_SQRT_2;
    let (xy, xz, yz) = (d[3] * s, d[4] * s, d[5] * s);
    DVec3::new(
        d[0] * v.x + xy * v.y + xz * v.z,
        xy * v.x + d[1] * v.y + yz * v.z,
        xz * v.x + yz * v.y + d[2] * v.z,
    )
}

pub(super) struct Fields {
    /// `γ̇` at the volume points.
    pub(super) shear: Vec<f64>,
    /// `D` at the volume points (six per point, scaled).
    pub(super) d: DVector<f64>,
}

/// `D'` of the total velocity and its shear rate at the volume nodes: the solenoidal part
/// of `u` plus the deviatoric shift `e^ε/√2` of A-Ξ (`None`: no shift).
pub(super) fn strain_fields_with(
    ops: &Operators,
    u: &DVector<f64>,
    shift: Option<&DVector<f64>>,
) -> Fields {
    let mut d = linalg::mul(&ops.rows.d, u);
    if let Some(e) = shift {
        d += e * std::f64::consts::FRAC_1_SQRT_2;
    }
    let np = ops.vol.pts.len();
    let shear = (0..np)
        .map(|p| {
            (2.0 * d6_dot(
                &d.as_slice()[6 * p..6 * p + 6],
                &d.as_slice()[6 * p..6 * p + 6],
            ))
            .sqrt()
        })
        .collect();
    Fields { shear, d }
}

pub(super) fn strain_fields(ops: &Operators, u: &DVector<f64>) -> Fields {
    let d = linalg::mul(&ops.rows.d, u);
    let np = ops.vol.pts.len();
    let shear = (0..np)
        .map(|p| {
            let dd = d6_dot(
                &d.as_slice()[6 * p..6 * p + 6],
                &d.as_slice()[6 * p..6 * p + 6],
            );
            (2.0 * dd).sqrt()
        })
        .collect();
    Fields { shear, d }
}

/// `G` and its range-space factorization, built on the first primal–dual solve of this
/// geometry.
pub(super) fn pd_metric(ops: &Operators) -> Result<&(DMatrix<f64>, RangeSolve), SimError> {
    if let Some(m) = ops.pd_metric.get() {
        return Ok(m);
    }
    let _span = super::timing::span("pd_metric");
    let metric = assemble_k(ops, &vec![1.0; ops.vol.pts.len()]);
    let mass = mass_diag(&ops.rows, &ops.vol);
    let damp = fine_damp(ops, &mass);
    let rs = factor_range(&ops.rigid, &metric, Some(&damp))?;
    Ok(ops.pd_metric.get_or_init(|| (metric, rs)))
}

pub(super) fn assemble_k(ops: &Operators, eta: &[f64]) -> DMatrix<f64> {
    // K = Σ_q 2 η_q w_q D_qᵀ D_q.
    let s: Vec<f64> = eta
        .iter()
        .zip(&ops.vol.w)
        .map(|(e, w)| 2.0 * e * w)
        .collect();
    linalg::weighted_gram(&ops.rows.d, 6, &s)
}

/// The diagonal of the weighted mass Gram `M_jj = ∫ ρ B_j·B_j`: the Jacobi
/// scale of the constrained systems and, weighted by `FINE_DAMP`, the price
/// the solve charges the compact kernel coefficients.
pub(super) fn mass_diag(rows: &VelRows<f32>, vol: &VolumeSet) -> Vec<f64> {
    let np = vol.pts.len();
    let n_u = rows.v.ncols();
    let mut m = vec![0.0; n_u];
    for p in 0..np {
        let wr = vol.w[p] * vol.rho[p];
        for j in 0..n_u {
            let s = (0..3)
                .map(|a| f64::from(rows.v[(3 * p + a, j)]).powi(2))
                .sum::<f64>();
            m[j] += wr * s;
        }
    }
    m
}

/// The mass-metric price `μ M_jj` a compact kernel mode pays in the range
/// solve; the global layer (rigid, flux, polynomial) is free. At `μ = 1e-3`
/// a fine-scale coefficient needs a load a thousand times its mass scale to
/// matter — exactly the discretization-noise floor the quadrature leaves.
const FINE_DAMP: f64 = 1e-3;

fn fine_damp(ops: &Operators, mass: &[f64]) -> DVector<f64> {
    DVector::from_fn(ops.geo.n(), |j, _| {
        if matches!(ops.geo.modes[j], gbasis::Mode::Kernel { .. }) {
            FINE_DAMP * mass[j]
        } else {
            0.0
        }
    })
}

pub(super) fn assemble_f(ops: &Operators, ld: &Loads) -> DVector<f64> {
    let mut f = assemble_f_on(&ops.rows, &ops.vol, ld);
    // Gravity part of the kernel modes by the continuum identity
    // ∫ρ g·∇×A = ∫(∇ρ×g + ρ∇×g)·A + ∮ρ g·(n̂×A). For the gradient loads
    // of §10.0.12.1 the interior kernels carry no gravity load at all; what
    // remains is a boundary quadrature of smooth fields. The direct product
    // quadrature of ∫ρ g·B resolves the kernel-scale field poorly and
    // fabricates a deviatoric load of pure quadrature noise. (The global
    // polynomial modes' volume quadrature is already exact at these degrees;
    // the by-parts form's boundary cancellation is the *less* accurate one.)
    let vol_s = gbasis::support_lists(&ops.geo, &ops.vol.pts);
    let bnd_s = gbasis::support_lists(&ops.geo, &ops.bnd.pts);
    for (j, m) in ops.geo.modes.iter().enumerate() {
        if !matches!(m, gbasis::Mode::Kernel { .. }) {
            continue;
        }
        let mut direct = 0.0;
        for &p in &vol_s[j] {
            let p = p as usize;
            direct += ops.vol.w[p] * ops.vol.rho[p] * ld.g_eff[p].dot(m.value(ops.vol.pts[p]));
        }
        let mut parts = 0.0;
        for &p in &vol_s[j] {
            let p = p as usize;
            let a = m.potential(ops.vol.pts[p]);
            parts += ops.vol.w[p]
                * (ops.vol.grho[p].cross(ld.g_eff[p]) + ops.vol.rho[p] * ld.g_curl[p]).dot(a);
        }
        for &q in &bnd_s[j] {
            let q = q as usize;
            let we = m.potential(ops.bnd.pts[q]);
            parts += ops.bnd.w[q] * ops.bnd_rho[q] * ld.g_bnd[q].dot(ops.bnd.normals[q].cross(we));
        }
        f[j] += parts - direct;
    }
    f
}

/// The load vector `ℓ_α = ∫ ρ g·B_α − T_tide : D B_α` on explicit rows: the A/B
/// oracle evaluates the same load on another basis family.
pub fn assemble_f_on(rows: &VelRows<f32>, vol: &VolumeSet, ld: &Loads) -> DVector<f64> {
    let np = vol.pts.len();
    let mut gv = DVector::zeros(3 * np);
    let mut tv = DVector::zeros(6 * np);
    for p in 0..np {
        let wr = vol.w[p] * vol.rho[p];
        gv[3 * p] = wr * ld.g_eff[p].x;
        gv[3 * p + 1] = wr * ld.g_eff[p].y;
        gv[3 * p + 2] = wr * ld.g_eff[p].z;
        let t = ld.tide[p];
        let s2 = std::f64::consts::SQRT_2;
        let w = vol.w[p];
        tv[6 * p] = w * t[0];
        tv[6 * p + 1] = w * t[1];
        tv[6 * p + 2] = w * t[2];
        tv[6 * p + 3] = w * s2 * t[3];
        tv[6 * p + 4] = w * s2 * t[4];
        tv[6 * p + 5] = w * s2 * t[5];
    }
    linalg::tr_mul(&rows.v, &gv) - linalg::tr_mul(&rows.d, &tv)
}

/// Magnitude scale of the load: the same sums with absolute values, so that a load that
/// cancels exactly (a uniform sphere in its own gravity) does not make relative residuals
/// meaningless.
pub(super) fn load_scale(ops: &Operators, ld: &Loads) -> f64 {
    let np = ops.vol.pts.len();
    let n_u = ops.n_u;
    let mut acc = DVector::<f64>::zeros(n_u);
    for p in 0..np {
        let wg = ops.vol.w[p] * ops.vol.rho[p] * ld.g_eff[p].length();
        let t = ld.tide[p];
        let wt = ops.vol.w[p] * t.iter().map(|x| x * x).sum::<f64>().sqrt();
        for j in 0..n_u {
            let vmag = (0..3)
                .map(|a| f64::from(ops.rows.v[(3 * p + a, j)]).powi(2))
                .sum::<f64>()
                .sqrt();
            let dmag = (0..6)
                .map(|c| f64::from(ops.rows.d[(6 * p + c, j)]).powi(2))
                .sum::<f64>()
                .sqrt();
            acc[j] += wg * vmag + wt * dmag;
        }
    }
    acc.norm().max(1e-300)
}

pub(super) struct RangeSolve {
    lu: LU<f64, Dyn, Dyn>,
    /// Jacobi scales: `u = D^{-1/2} û` solves the original system.
    pub(super) d: Vec<f64>,
    /// The factored KKT matrix, kept for iterative refinement of the solve.
    a: DMatrix<f64>,
    n: usize,
}

/// Factor the range-space saddle system of `min_u ½uᵀKu − fᵀu` s.t. `B_rb u = 0`.
/// The (G1) family contains rigid fields twice — the six explicit modes and
/// combinations of polynomial modes — so `K` has null directions the
/// `γ BᵀB` penalty could not pin without polluting the elastic block. The KKT
/// system `[Ks B'ᵀ; B' 0]` is regular exactly because `ker K ∩ ker B = {0}`:
/// every strain-free field is rigid (Korn) and every rigid field carries
/// momentum. Mass-diagonal Jacobi scaling removes the polynomial/kernel
/// scale disparity.
///
/// `damp`, when given, adds a diagonal regularizer `uᵀ(μ M_fine)u` to the
/// energy: a mass-metric price on the compact kernel coefficients. The
/// discrete system is singular at quadrature accuracy — near-strain-null
/// directions answer a discretization-level residual with O(1) fields — so
/// the delivered representative must be selected, not merely solved for.
/// Paying mass for fine-scale content picks the smoothest admissible field:
/// the global layer carries the smooth response while the kernels answer
/// only the residual that genuinely demands them. The penalty is added to
/// `K` unscaled, so its semantics do not depend on the Jacobi preconditioner.
pub(super) fn factor_range(
    rigid: &DMatrix<f64>,
    k: &DMatrix<f64>,
    damp: Option<&DVector<f64>>,
) -> Result<RangeSolve, SimError> {
    let n = k.nrows();
    let btb0 = rigid.transpose() * rigid;
    let d: Vec<f64> = (0..n)
        .map(|i| k[(i, i)].max(btb0[(i, i)]).max(1e-300).sqrt())
        .collect();
    let mut a = DMatrix::zeros(n + 6, n + 6);
    for b in 0..n {
        for c in 0..n {
            let mut s = k[(b, c)];
            if b == c {
                if let Some(dp) = damp {
                    s += dp[b];
                }
            }
            a[(b, c)] = s / (d[b] * d[c]);
        }
        for r in 0..6 {
            a[(b, n + r)] = rigid[(r, b)] / d[b];
            a[(n + r, b)] = rigid[(r, b)] / d[b];
        }
    }
    let lu = a.clone().lu();
    if !lu.is_invertible() {
        return Err(SimError::rigid_rank(
            "the scaled range-space KKT system is singular: a rigid-field direction \
             of the stiff Gram carries no momentum constraint",
        ));
    }
    Ok(RangeSolve { lu, d, a, n })
}

pub(super) fn solve_range(
    _rigid: &DMatrix<f64>,
    rs: &RangeSolve,
    f: &DVector<f64>,
) -> (DVector<f64>, DVector<f64>) {
    let n = rs.n;
    let mut rhs = DVector::zeros(n + 6);
    for i in 0..n {
        rhs[i] = f[i] / rs.d[i];
    }
    let mut sol = rs.lu.solve(&rhs).unwrap_or_else(|| DVector::zeros(n + 6));
    // Iterative refinement: the KKT conditioning leaves a residual of order
    // cond·eps; a few corrections recover it without a costlier factorization.
    for _ in 0..3 {
        let r = &rhs - &rs.a * &sol;
        if r.norm() <= 1e-12 * rhs.norm().max(1e-300) {
            break;
        }
        match rs.lu.solve(&r) {
            Some(c) => sol += c,
            None => break,
        }
    }
    let u = DVector::from_fn(n, |i, _| sol[i] / rs.d[i]);
    let lam = DVector::from_fn(6, |i, _| sol[n + i]);
    (u, lam)
}

impl RangeSolve {
    /// The metric projection onto the affine plane `B_rb u = −b` of §11.8:
    /// `v = u − K⁻¹ Bᵀ λ` with `B v = −b`, one scaled KKT solve with the
    /// right-hand side `[K_s û; −b]`.
    pub(super) fn project_onto(&self, u: &DVector<f64>, b: &DVector<f64>) -> DVector<f64> {
        let n = self.n;
        let us = DVector::from_fn(n, |i, _| u[i] * self.d[i]);
        let mut rhs = DVector::zeros(n + 6);
        for i in 0..n {
            rhs[i] = (0..n).map(|j| self.a[(i, j)] * us[j]).sum();
        }
        for r in 0..6 {
            rhs[n + r] = -b[r];
        }
        let sol = self.lu.solve(&rhs).unwrap_or_else(|| DVector::zeros(n + 6));
        DVector::from_fn(n, |i, _| sol[i] / self.d[i])
    }
}

/// The Stokes–pressure pair at fixed geometry (§10.3.2): the characteristic
/// pseudo-time relaxation (8.3-H), the sole rheology solver of the normative
/// loop. Non-convergence is reported as a bounded `RheologyNotConverged`
/// rather than propagated NaNs.
pub fn solve(
    ops: &Operators,
    ld: &Loads,
    prm: &StokesParams,
    m: &Material,
    guess: Option<&Guess>,
) -> Result<StokesSolution, SimError> {
    hyperbolic::solve_hyperbolic(ops, ld, prm, m, guess)
}
