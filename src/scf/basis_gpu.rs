//! GPU driver for the Jacobi–spherical-harmonic basis (`scf/shader/basis.wesl`).

use std::sync::OnceLock;

use rayon::prelude::*;

use nalgebra::DMatrix;

use crate::common::linalg::{self, Entry};
use crate::common::math::DVec3;
use crate::rheo::basis::{K, ds_pair, pack_dirs, pack_scalars, psi_poly};

use super::basis::{PresLayout, PresRows, VelLayout, VelRows, Want};
use super::harmonics::SurfaceFit;
use super::jets::packed_jet_tables;
use crate::gpu::{Arg, Res, kernel, once};

/// Output bytes per basis launch: larger batches fill the GPU (one thread per point), smaller
/// ones cap the transient copies (and, in the browser, the workers' heap high-water mark).
const BATCH_BYTES: usize = 32 * 1024 * 1024;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct BasisParams {
    n_rays: u32,
    n_per: u32,
    n_points: u32,
    l_mode: u32,
    l_surf: u32,
    k_count: u32,
    n_out: u32,
    n_lift: u32,
    n_comp: u32,
    poly_stride: u32,
    top_pow: u32,
    want_v: u32,
    want_d: u32,
    want_lap: u32,
    want_gd: u32,
    gradients: u32,
    nmul0: u32,
    nmul1: u32,
    nmul2: u32,
    nmul3: u32,
    nmul4: u32,
    moff0: u32,
    moff1: u32,
    moff2: u32,
    moff3: u32,
    moff4: u32,
    up_off: u32,
    mono_off: u32,
    deg_off: u32,
    pair_off: u32,
    sqrt2_hi: f32,
    sqrt2_lo: f32,
    pmm_hi: f32,
    pmm_lo: f32,
}

const _: () = assert!(std::mem::size_of::<BasisParams>() == 136);
const _: () = assert!(std::mem::align_of::<BasisParams>() == 4);

/// The monomial jet tables, resident for the whole run.
struct Core {
    tables: Res,
}

fn core() -> Result<&'static Core, String> {
    static SLOT: OnceLock<Result<Core, String>> = OnceLock::new();
    SLOT.get_or_init(|| {
        Ok(Core {
            tables: Res::from_slice(&packed_jet_tables().words)?,
        })
    })
    .as_ref()
    .map_err(Clone::clone)
}

fn host_buf(data: &[f32]) -> Result<Vec<f32>, String> {
    Ok(data.to_vec())
}

fn groups_of(n: usize) -> [u32; 3] {
    [n.div_ceil(64).max(1) as u32, 1, 1]
}

fn as_u32(n: usize, what: &str) -> Result<u32, String> {
    u32::try_from(n).map_err(|_| format!("{what} does not fit in a 32-bit Metal index"))
}

fn check_surface(surf: &SurfaceFit) -> Result<(), String> {
    let expect = (surf.l_max + 1) * (surf.l_max + 1);
    if surf.coeff.len() != expect {
        return Err(format!(
            "surface fit has {} coefficients, expected {expect} for degree {}",
            surf.coeff.len(),
            surf.l_max
        ));
    }
    if surf.l_max > 24 {
        return Err(format!(
            "surface degree {} exceeds the GPU limit 24",
            surf.l_max
        ));
    }
    Ok(())
}

fn blank_params() -> BasisParams {
    let tables = packed_jet_tables();
    let sqrt2 = ds_pair(std::f64::consts::SQRT_2);
    let pmm = ds_pair((1.0 / (4.0 * std::f64::consts::PI)).sqrt());
    BasisParams {
        n_rays: 0,
        n_per: 1,
        n_points: 0,
        l_mode: 0,
        l_surf: 0,
        k_count: 0,
        n_out: 0,
        n_lift: 0,
        n_comp: 0,
        poly_stride: 0,
        top_pow: 0,
        want_v: 0,
        want_d: 0,
        want_lap: 0,
        want_gd: 0,
        gradients: 0,
        nmul0: tables.mul_n[0],
        nmul1: tables.mul_n[1],
        nmul2: tables.mul_n[2],
        nmul3: tables.mul_n[3],
        nmul4: tables.mul_n[4],
        moff0: tables.mul_off[0],
        moff1: tables.mul_off[1],
        moff2: tables.mul_off[2],
        moff3: tables.mul_off[3],
        moff4: tables.mul_off[4],
        up_off: tables.up_off,
        mono_off: tables.mono_off,
        deg_off: tables.deg_off,
        pair_off: tables.pair_off,
        sqrt2_hi: sqrt2[0],
        sqrt2_lo: sqrt2[1],
        pmm_hi: pmm[0],
        pmm_lo: pmm[1],
    }
}

/// Jacobi coefficients, one `poly_stride` slot per `(l, k)`. `top` is the last power index.
/// Radial polynomial table: row `(l, k)` holds `ψ_{lk}` (§8.1). With `pressure`,
/// row `l = 0` holds instead `ψ_{0k}(ζ) = (1 + ζ) P_k^{(0,1/2)}(2ζ² − 1)` of §8.1.3, so
/// that `(1 − ζ) ψ_{0k}` is even in `r` and the pressure is smooth at the centre.
fn pack_polys(l_max: usize, k_count: usize, pressure: bool) -> Result<(Vec<f32>, u32), String> {
    let stride = if pressure {
        (l_max + k_count).max(2 * k_count)
    } else {
        l_max + k_count
    };
    let top = stride - 1;
    if top > 32 {
        return Err(format!(
            "radial degree l + k = {} exceeds the GPU power table",
            top
        ));
    }
    let mut words = vec![0.0f32; (l_max + 1) * k_count * stride * 2];
    for l in 0..=l_max {
        for k in 0..k_count {
            let poly = if pressure && l == 0 {
                pressure_poly0(k)
            } else {
                psi_poly(l, k)
            };
            debug_assert!(poly.len() <= stride);
            let base = (l * k_count + k) * stride;
            for (j, c) in poly.iter().enumerate() {
                let pair = ds_pair(*c);
                words[(base + j) * 2] = pair[0];
                words[(base + j) * 2 + 1] = pair[1];
            }
        }
    }
    Ok((words, as_u32(stride, "polynomial stride")?))
}

/// Ascending coefficients of `(1 + ζ) P_k^{(0,1/2)}(2ζ² − 1)`, with
/// `P_k^{(0,β)}(2u − 1) = Σ_s C(k, k−s) C(k+β, s) (u − 1)^s u^{k−s}` at `u = ζ²`.
fn pressure_poly0(k: usize) -> Vec<f64> {
    let binom = |n: f64, r: usize| (0..r).fold(1.0, |v, i| v * (n - i as f64) / (i + 1) as f64);
    let mut q = vec![0.0; k + 1];
    for s in 0..=k {
        let weight = binom(k as f64, k - s) * binom(k as f64 + 0.5, s);
        // (u − 1)^s u^(k−s)
        let mut poly = vec![1.0];
        for _ in 0..s {
            let mut next = vec![0.0; poly.len() + 1];
            for (i, c) in poly.iter().enumerate() {
                next[i] -= c;
                next[i + 1] += c;
            }
            poly = next;
        }
        for (i, c) in poly.iter().enumerate() {
            q[k - s + i] += weight * c;
        }
    }
    // u = ζ², then times (1 + ζ).
    let mut out = vec![0.0; 2 * k + 2];
    for (i, c) in q.iter().enumerate() {
        out[2 * i] += c;
        out[2 * i + 1] += c;
    }
    out
}

/// Size of the ζ-power arrays of the velocity jet kernels (`array<Jt_n, 12>` in
/// `basis.wesl`). Each GPU thread holds one; Metal reserves that private memory for every
/// thread in flight, so the arrays are sized to the velocity basis (l ≤ 8, K = 4), not
/// to the pressure table.
const VEL_POWS: usize = 12;

/// Last ζ power of the velocity kernels, checked against [`VEL_POWS`].
fn velocity_top_power(l_max: usize) -> Result<usize, String> {
    let top = l_max + K - 1;
    if top >= VEL_POWS {
        return Err(format!(
            "velocity radial degree l + k = {top} exceeds the GPU power table ({VEL_POWS})"
        ));
    }
    Ok(top)
}

/// Highest power of ζ the kernels read for `pack_polys(l_max, k, pressure)`.
fn top_power(l_max: usize, k_count: usize, pressure: bool) -> usize {
    if pressure {
        (l_max + k_count).max(2 * k_count) - 1
    } else {
        l_max + k_count - 1
    }
}

fn ds_at(words: &[f32], channel: usize, n: usize, point: usize) -> f64 {
    let i = (channel * n + point) * 2;
    f64::from(words[i]) + f64::from(words[i + 1])
}

fn write_at<T: Entry>(matrix: &mut DMatrix<T>, row: usize, col: usize, value: f64) {
    let nrows = matrix.nrows();
    debug_assert!(row < nrows && col < matrix.ncols());
    unsafe {
        *matrix.as_mut_ptr().add(col * nrows + row) = T::of(value);
    }
}

#[derive(Clone, Copy)]
enum Bind {
    /// Points, params, coefficients, polynomials, tables, output.
    Full,
    /// Pressure value kernel: no monomial tables.
    NoTables,
    /// Velocity contraction: mode coefficients at buffer 6, output at 5.
    FullDot,
    /// Pressure contraction: mode coefficients at buffer 4, output at 5.
    PresDot,
}

fn launch(
    core: &Core,
    entry: &str,
    points: &[f32],
    params: &BasisParams,
    coeff: &[f32],
    polys: &[f32],
    bind: Bind,
    n_f32: usize,
    threads: usize,
    coef: Option<&Vec<f32>>,
) -> Result<Words, String> {
    if threads == 0 || n_f32 == 0 {
        return Ok(Words(vec![0; n_f32 * 4]));
    }
    let params_b = bytemuck::bytes_of(params);
    let points_b: &[u8] = bytemuck::cast_slice(points);
    let coeff_b: &[u8] = bytemuck::cast_slice(coeff);
    let polys_b: &[u8] = bytemuck::cast_slice(polys);
    let coef_b: &[u8] = bytemuck::cast_slice(coef.map(Vec::as_slice).unwrap_or(&[0.0, 0.0]));
    let out = Arg::Out(n_f32 * 4);
    let mut args = vec![Arg::In(points_b), Arg::In(params_b), Arg::In(coeff_b)];
    match bind {
        Bind::Full => args.extend([Arg::In(polys_b), Arg::Res(&core.tables), out]),
        Bind::NoTables => args.extend([Arg::In(polys_b), out]),
        Bind::FullDot => args.extend([
            Arg::In(polys_b),
            Arg::Res(&core.tables),
            out,
            Arg::In(coef_b),
        ]),
        Bind::PresDot => args.extend([Arg::In(polys_b), Arg::In(coef_b), out]),
    }
    let mut back = once(kernel(entry)?, groups_of(threads), &args)?;
    Ok(Words(back.swap_remove(0)))
}

/// Read-back bytes of one launch, viewed as `f32` words.
struct Words(Vec<u8>);

impl Words {
    /// The words in place when the allocation is aligned for `f32` (the usual case),
    /// otherwise one copy.
    fn get(&self) -> std::borrow::Cow<'_, [f32]> {
        match bytemuck::try_cast_slice(&self.0) {
            Ok(w) => std::borrow::Cow::Borrowed(w),
            Err(_) => std::borrow::Cow::Owned(bytemuck::pod_collect_to_vec(&self.0)),
        }
    }
}

fn component_count(want: Want) -> usize {
    let mut n = 0;
    if want.v {
        n += 3;
    }
    if want.d || want.lap || want.gd {
        n += 6;
    }
    if want.lap {
        n += 3;
    }
    if want.gd {
        n += 18;
    }
    n
}

struct VelKernel {
    pipe: &'static str,
    dot: &'static str,
    grouped: bool,
    name_limit: u32,
}

fn velocity_kernel(ord: usize, l_max: usize, rays: bool) -> Result<VelKernel, String> {
    if l_max > 8 {
        return Err(format!("velocity degree {l_max} exceeds the GPU limit 8"));
    }
    // Low degrees use the ray-grouped kernels for every order up to 3: the
    // angular jets are shared by the samples of one ray, and the extra derivative
    // order costs less than recomputing them per sample.
    let grouped_ord = if rays { 0..=3 } else { 3..=3 };
    if grouped_ord.contains(&ord) && l_max <= 2 {
        Ok(VelKernel {
            pipe: "basis_vel_o3_l2",
            dot: "basis_vel_o3_l2_dot",
            grouped: true,
            name_limit: 6,
        })
    } else if grouped_ord.contains(&ord) && l_max <= 4 {
        Ok(VelKernel {
            pipe: "basis_vel_o3_l4",
            dot: "basis_vel_o3_l4_dot",
            grouped: true,
            name_limit: 8,
        })
    } else if ord <= 2 {
        Ok(VelKernel {
            pipe: "basis_vel_o2",
            dot: "basis_vel_o2_dot",
            grouped: false,
            name_limit: 12,
        })
    } else if ord == 3 {
        Ok(VelKernel {
            pipe: "basis_vel_o3",
            dot: "basis_vel_o3_dot",
            grouped: false,
            name_limit: 12,
        })
    } else if ord == 4 {
        Ok(VelKernel {
            pipe: "basis_vel_o4",
            dot: "basis_vel_o4_dot",
            grouped: false,
            name_limit: 12,
        })
    } else {
        Err(format!("velocity jet order {ord} is not supported"))
    }
}

pub(crate) fn velocity<T: Entry>(
    surf: &SurfaceFit,
    lay: &VelLayout,
    pts: &[DVec3],
    n_per: usize,
    want: Want,
) -> Result<VelRows<T>, String> {
    check_surface(surf)?;
    let np = pts.len();
    let n_u = lay.n_u;
    let mk = |on: bool, rows: usize| linalg::zeros(if on { rows * np } else { 0 }, n_u);
    let mut rows = VelRows {
        n_points: np,
        v: mk(want.v, 3),
        d: mk(want.d || want.lap || want.gd, 6),
        lap: mk(want.lap, 3),
        gd: mk(want.gd, 18),
    };
    if np == 0 || n_u == 0 {
        return Ok(rows);
    }
    if n_per == 0 || np % n_per != 0 {
        return Err(format!(
            "velocity ray width {n_per} does not divide {np} sample points"
        ));
    }
    let n_comp = component_count(want);
    if n_comp == 0 {
        return Ok(rows);
    }
    let kernel = velocity_kernel(want.order(), lay.l_max, true)?;
    let grouped = kernel.grouped;
    let n_per = if grouped { n_per } else { 1 };
    let n_rays = np / n_per;
    let top = velocity_top_power(lay.l_max)?;
    if top > kernel.name_limit as usize {
        return Err(format!(
            "velocity power index {top} exceeds the {} kernel",
            if grouped { "cached" } else { "point" }
        ));
    }
    let (polys, stride) = pack_polys(lay.l_max, K, false)?;
    let core = core()?;
    let coeff = host_buf(&pack_scalars(&surf.coeff))?;
    let poly_buffer = host_buf(&polys)?;
    let packed = pack_dirs(pts);
    let bytes_per_point = n_u * n_comp * 8;
    let rays_per_batch = (BATCH_BYTES / bytes_per_point.max(1) / n_per).clamp(1, n_rays);
    let mut params = blank_params();
    params.n_per = as_u32(n_per, "samples per ray")?;
    params.l_mode = as_u32(lay.l_max, "velocity degree")?;
    params.l_surf = as_u32(surf.l_max, "surface degree")?;
    params.k_count = K as u32;
    params.n_comp = as_u32(n_comp, "velocity components")?;
    params.poly_stride = stride;
    params.top_pow = as_u32(top, "power index")?;
    params.want_v = u32::from(want.v);
    params.want_d = u32::from(want.d || want.lap || want.gd);
    params.want_lap = u32::from(want.lap);
    params.want_gd = u32::from(want.gd);
    let mut ray0 = 0;
    while ray0 < n_rays {
        let batch_rays = rays_per_batch.min(n_rays - ray0);
        let batch_points = batch_rays * n_per;
        let point0 = ray0 * n_per;
        params.n_rays = as_u32(if grouped { batch_rays } else { batch_points }, "rays")?;
        params.n_points = as_u32(batch_points, "points")?;
        if !grouped {
            params.n_per = 1;
        }
        let word0 = point0 * 6;
        let words = launch(
            core,
            kernel.pipe,
            &packed[word0..word0 + batch_points * 6],
            &params,
            &coeff,
            &poly_buffer,
            Bind::Full,
            batch_points * n_u * n_comp * 2,
            if grouped { batch_rays } else { batch_points },
            None,
        )?;
        scatter_velocity(&mut rows, &words.get(), want, n_comp, point0, batch_points);
        ray0 += batch_rays;
    }
    Ok(rows)
}

/// Unpack one batch into the column-major rows, one column per task: each column is
/// written as one contiguous run instead of 30 strided writes per point and mode.
fn scatter_velocity<T: Entry>(
    rows: &mut VelRows<T>,
    words: &[f32],
    want: Want,
    n_comp: usize,
    point0: usize,
    n_batch: usize,
) {
    // Each matrix with its rows per point, at its offset in a point's record.
    let strain = want.d || want.lap || want.gd;
    let mut local = 0;
    let mut targets = Vec::with_capacity(4);
    for (m, per, on) in [
        (&mut rows.v, 3, want.v),
        (&mut rows.d, 6, strain),
        (&mut rows.lap, 3, want.lap),
        (&mut rows.gd, 18, want.gd),
    ] {
        if on {
            targets.push((m, per, local));
            local += per;
        }
    }
    debug_assert_eq!(local, n_comp);
    for (m, per, local) in targets {
        let nrows = m.nrows();
        m.as_mut_slice()
            .par_chunks_mut(nrows)
            .enumerate()
            .for_each(|(j, col)| {
                let dst = &mut col[per * point0..per * (point0 + n_batch)];
                for a in 0..per {
                    let channel = j * n_comp + local + a;
                    for p in 0..n_batch {
                        dst[per * p + a] = T::of(ds_at(words, channel, n_batch, p));
                    }
                }
            });
    }
}

pub(crate) fn pressure(
    surf: &SurfaceFit,
    lay: &PresLayout,
    pts: &[DVec3],
    gradients: bool,
) -> Result<PresRows, String> {
    check_surface(surf)?;
    let np = pts.len();
    let n = lay.n_p();
    let nl = lay.n_lift();
    let mut rows = PresRows {
        val: DMatrix::zeros(np, n),
        grad: DMatrix::zeros(if gradients { 3 * np } else { 0 }, n),
        lift_val: DMatrix::zeros(np, nl),
        lift_grad: DMatrix::zeros(if gradients { 3 * np } else { 0 }, nl),
    };
    if np == 0 || lay.k_max == 0 {
        return Ok(rows);
    }
    if lay.l_max > 24 {
        return Err(format!(
            "pressure degree {} exceeds the GPU limit 24",
            lay.l_max
        ));
    }
    let (polys, stride) = pack_polys(lay.l_max, lay.k_max, true)?;
    let top = top_power(lay.l_max, lay.k_max, true);
    let core = core()?;
    let coeff = host_buf(&pack_scalars(&surf.coeff))?;
    let poly_buffer = host_buf(&polys)?;
    let packed = pack_dirs(pts);
    let channels = if gradients { (n + nl) * 4 } else { n + nl };
    let per_batch = (BATCH_BYTES / (channels * 8).max(1)).clamp(1, np);
    let pipe = if gradients {
        "basis_pres1"
    } else if lay.l_max <= 4 {
        "basis_pres0_l4"
    } else {
        "basis_pres0_l24"
    };
    let bind = if gradients {
        Bind::Full
    } else {
        Bind::NoTables
    };
    let mut params = blank_params();
    params.n_per = 1;
    params.l_mode = as_u32(lay.l_max, "pressure degree")?;
    params.l_surf = as_u32(surf.l_max, "surface degree")?;
    params.k_count = as_u32(lay.k_max, "pressure radial count")?;
    params.n_out = as_u32(n, "pressure modes")?;
    params.n_lift = as_u32(nl, "lift modes")?;
    params.poly_stride = stride;
    params.top_pow = as_u32(top, "power index")?;
    params.gradients = u32::from(gradients);
    let mut point0 = 0;
    while point0 < np {
        let batch = per_batch.min(np - point0);
        params.n_rays = as_u32(batch, "points")?;
        params.n_points = params.n_rays;
        let word0 = point0 * 6;
        let words = launch(
            core,
            pipe,
            &packed[word0..word0 + batch * 6],
            &params,
            &coeff,
            &poly_buffer,
            bind,
            batch * channels * 2,
            batch,
            None,
        )?;
        if gradients {
            scatter_pressure_grad(&mut rows, &words.get(), n, nl, point0, batch);
        } else {
            scatter_pressure_value(&mut rows, &words.get(), n, nl, point0, batch);
        }
        point0 += batch;
    }
    Ok(rows)
}

fn scatter_pressure_value(
    rows: &mut PresRows,
    words: &[f32],
    n: usize,
    nl: usize,
    point0: usize,
    n_batch: usize,
) {
    for p in 0..n_batch {
        let gp = point0 + p;
        for j in 0..n {
            write_at(&mut rows.val, gp, j, ds_at(words, j, n_batch, p));
        }
        for lm in 0..nl {
            write_at(&mut rows.lift_val, gp, lm, ds_at(words, n + lm, n_batch, p));
        }
    }
}

fn scatter_pressure_grad(
    rows: &mut PresRows,
    words: &[f32],
    n: usize,
    nl: usize,
    point0: usize,
    n_batch: usize,
) {
    for p in 0..n_batch {
        let gp = point0 + p;
        for j in 0..n {
            let ch = j * 4;
            write_at(&mut rows.val, gp, j, ds_at(words, ch, n_batch, p));
            for a in 0..3 {
                write_at(
                    &mut rows.grad,
                    3 * gp + a,
                    j,
                    ds_at(words, ch + 1 + a, n_batch, p),
                );
            }
        }
        for lm in 0..nl {
            let ch = (n + lm) * 4;
            write_at(&mut rows.lift_val, gp, lm, ds_at(words, ch, n_batch, p));
            for a in 0..3 {
                write_at(
                    &mut rows.lift_grad,
                    3 * gp + a,
                    lm,
                    ds_at(words, ch + 1 + a, n_batch, p),
                );
            }
        }
    }
}

fn unpack_pairs(words: &[f32]) -> Vec<f64> {
    words
        .chunks_exact(2)
        .map(|p| f64::from(p[0]) + f64::from(p[1]))
        .collect()
}

pub(crate) struct ContractedVelocity {
    pub v: Vec<f64>,
    pub d: Vec<f64>,
}

/// `B u` for velocity and strain, accumulated in the kernel.
///
/// The basis matrix is not stored and is not copied back. `want` may ask for
/// `v` and `d` only; Laplacian rows stay on the matrix path used by the Stokes
/// operator.
pub(crate) fn contract_velocity(
    surf: &SurfaceFit,
    lay: &VelLayout,
    pts: &[DVec3],
    n_per: usize,
    want: Want,
    u: &[f64],
) -> Result<ContractedVelocity, String> {
    if want.lap || want.gd {
        return Err(
            "velocity contraction covers v and D; Laplacian rows stay on the operator matrix"
                .into(),
        );
    }
    check_surface(surf)?;
    let np = pts.len();
    let n_comp = component_count(want);
    let empty = ContractedVelocity {
        v: vec![0.0; if want.v { np * 3 } else { 0 }],
        d: vec![0.0; if want.d { np * 6 } else { 0 }],
    };
    if np == 0 || lay.n_u == 0 || n_comp == 0 {
        return Ok(empty);
    }
    if u.len() != lay.n_u {
        return Err(format!(
            "velocity contraction has {} coefficients, expected {}",
            u.len(),
            lay.n_u
        ));
    }
    if n_comp > 9 {
        return Err(format!(
            "velocity contraction has {n_comp} components; the kernel holds 9"
        ));
    }
    if n_per == 0 || np % n_per != 0 {
        return Err(format!(
            "velocity ray width {n_per} does not divide {np} sample points"
        ));
    }
    let kernel = velocity_kernel(want.order(), lay.l_max, true)?;
    let grouped = kernel.grouped;
    let n_per = if grouped { n_per } else { 1 };
    let n_rays = np / n_per;
    let (polys, stride) = pack_polys(lay.l_max, K, false)?;
    let top = velocity_top_power(lay.l_max)?;
    if top > kernel.name_limit as usize {
        return Err(format!(
            "velocity power index {top} exceeds the {} kernel",
            if grouped { "cached" } else { "point" }
        ));
    }
    let core = core()?;
    let coeff = host_buf(&pack_scalars(&surf.coeff))?;
    let poly_buffer = host_buf(&polys)?;
    let coef = host_buf(&pack_scalars(u))?;
    let packed = pack_dirs(pts);
    let mut params = blank_params();
    params.n_per = as_u32(n_per, "samples per ray")?;
    params.n_rays = as_u32(if grouped { n_rays } else { np }, "rays")?;
    params.n_points = as_u32(np, "points")?;
    params.l_mode = as_u32(lay.l_max, "velocity degree")?;
    params.l_surf = as_u32(surf.l_max, "surface degree")?;
    params.k_count = K as u32;
    params.n_comp = as_u32(n_comp, "velocity components")?;
    params.poly_stride = stride;
    params.top_pow = as_u32(top, "power index")?;
    params.want_v = u32::from(want.v);
    params.want_d = u32::from(want.d);
    if !grouped {
        params.n_per = 1;
    }
    let words = launch(
        core,
        kernel.dot,
        &packed,
        &params,
        &coeff,
        &poly_buffer,
        Bind::FullDot,
        np * n_comp * 2,
        if grouped { n_rays } else { np },
        Some(&coef),
    )?;
    let flat = unpack_pairs(&words.get());
    let mut v = vec![0.0; if want.v { np * 3 } else { 0 }];
    let mut d = vec![0.0; if want.d { np * 6 } else { 0 }];
    let nv = usize::from(want.v) * 3;
    for p in 0..np {
        let base = p * n_comp;
        if want.v {
            v[3 * p..3 * p + 3].copy_from_slice(&flat[base..base + 3]);
        }
        if want.d {
            d[6 * p..6 * p + 6].copy_from_slice(&flat[base + nv..base + nv + 6]);
        }
    }
    Ok(ContractedVelocity { v, d })
}

/// Face-centre strain basis kept on the GPU.
///
/// Picard changes `u` every sweep, so the basis has to stay, but it stays on
/// the device. One buffer for every point is about a gigabyte, which does not
/// fit the worker transfer, so the points are split and only the GEMV result
/// comes back. `contract` returns `6 * n_points` values.
pub(crate) struct GpuStrain {
    pieces: Vec<Res>,
    counts: Vec<usize>,
    n_points: usize,
    n_u: usize,
    n_comp: usize,
}

/// Velocity (`want.v`) and strain (`want.d`) basis rows at fixed points, resident on
/// the device; `contract` returns the selected components per point, `v` first.
pub(crate) fn resident_basis(
    surf: &SurfaceFit,
    lay: &VelLayout,
    pts: &[DVec3],
    want: Want,
) -> Result<GpuStrain, String> {
    if want.lap || want.gd || !(want.v || want.d) {
        return Err("a resident basis holds v and D rows only".into());
    }
    check_surface(surf)?;
    let np = pts.len();
    let n_comp = component_count(want);
    if np == 0 || lay.n_u == 0 {
        let _core = core()?;
        return Ok(GpuStrain {
            pieces: Vec::new(),
            counts: Vec::new(),
            n_points: np,
            n_u: lay.n_u,
            n_comp,
        });
    }
    let kernel = velocity_kernel(want.order(), lay.l_max, false)?;
    let (polys, stride) = pack_polys(lay.l_max, K, false)?;
    let top = velocity_top_power(lay.l_max)?;
    if top > kernel.name_limit as usize {
        return Err(format!(
            "velocity power index {top} exceeds the strain kernel"
        ));
    }
    let core = core()?;
    let coeff = host_buf(&pack_scalars(&surf.coeff))?;
    let poly_buffer = host_buf(&polys)?;
    let packed = pack_dirs(pts);
    let mut params = blank_params();
    params.n_per = 1;
    params.n_rays = as_u32(np, "rays")?;
    params.n_points = params.n_rays;
    params.l_mode = as_u32(lay.l_max, "velocity degree")?;
    params.l_surf = as_u32(surf.l_max, "surface degree")?;
    params.k_count = K as u32;
    params.n_comp = n_comp as u32;
    params.poly_stride = stride;
    params.top_pow = as_u32(top, "power index")?;
    params.want_v = u32::from(want.v);
    params.want_d = u32::from(want.d);
    let bytes_per = lay.n_u.saturating_mul(n_comp).saturating_mul(8).max(1);
    let piece = (128 * 1024 * 1024 / bytes_per).max(1);
    let mut resident = GpuStrain {
        pieces: Vec::new(),
        counts: Vec::new(),
        n_points: np,
        n_u: lay.n_u,
        n_comp,
    };
    let mut start = 0;
    while start < np {
        let n = piece.min(np - start);
        params.n_rays = as_u32(n, "rays")?;
        params.n_points = params.n_rays;
        let word0 = start * 6;
        let out = Res::zeroed((n * lay.n_u * n_comp * 8) as u64)?;
        let mut job = crate::gpu::Job::new();
        let args = [
            job.input(&packed[word0..word0 + n * 6]),
            job.value(&params),
            job.input(&coeff),
            job.input(&poly_buffer),
            (&core.tables).into(),
            (&out).into(),
        ];
        job.run(crate::gpu::kernel(kernel.pipe)?, groups_of(n), &args);
        job.submit()?;
        resident.pieces.push(out);
        resident.counts.push(n);
        start += n;
    }
    Ok(resident)
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GemvParams {
    n_points: u32,
    n_modes: u32,
    n_comp: u32,
    pad: u32,
}

impl GpuStrain {
    pub(crate) fn contract(&self, u: &[f64]) -> Result<Vec<f64>, String> {
        if self.n_points == 0 || self.n_u == 0 {
            return Ok(vec![0.0; self.n_points * self.n_comp]);
        }
        if u.len() != self.n_u {
            return Err(format!(
                "strain contraction has {} coefficients, expected {}",
                u.len(),
                self.n_u
            ));
        }
        let coef = host_buf(&pack_scalars(u))?;
        let mut flat = Vec::with_capacity(self.n_points * self.n_comp);
        for (piece, &n) in self.pieces.iter().zip(&self.counts) {
            let params = GemvParams {
                n_points: as_u32(n, "strain points")?,
                n_modes: as_u32(self.n_u, "strain modes")?,
                n_comp: as_u32(self.n_comp, "strain components")?,
                pad: 0,
            };
            let back = once(
                &crate::gpu::kernels::BASIS_GEMV,
                groups_of(n),
                &[
                    Arg::Res(piece),
                    Arg::In(bytemuck::cast_slice(&coef)),
                    Arg::Out(n * self.n_comp * 8),
                    Arg::In(bytemuck::bytes_of(&params)),
                ],
            )?;
            flat.extend(unpack_pairs(&bytemuck::pod_collect_to_vec::<u8, f32>(
                &back[0],
            )));
        }
        Ok(flat)
    }
}

pub(crate) fn contract_pressure(
    surf: &SurfaceFit,
    lay: &PresLayout,
    pts: &[DVec3],
    pres_c: &[f64],
    lift: &[f64],
) -> Result<Vec<f64>, String> {
    check_surface(surf)?;
    let np = pts.len();
    if np == 0 || lay.k_max == 0 {
        return Ok(vec![0.0; np]);
    }
    if lay.l_max > 24 {
        return Err(format!(
            "pressure degree {} exceeds the GPU limit 24",
            lay.l_max
        ));
    }
    let n = lay.n_p();
    let nl = lay.n_lift();
    if pres_c.len() != n || lift.len() != nl {
        return Err(format!(
            "pressure contraction expected {n} modes and {nl} lift coefficients, got {} and {}",
            pres_c.len(),
            lift.len()
        ));
    }
    let (polys, stride) = pack_polys(lay.l_max, lay.k_max, true)?;
    let top = top_power(lay.l_max, lay.k_max, true);
    let core = core()?;
    let coeff = host_buf(&pack_scalars(&surf.coeff))?;
    let poly_buffer = host_buf(&polys)?;
    let mut modes = pack_scalars(pres_c);
    modes.extend_from_slice(&pack_scalars(lift));
    let coef = host_buf(&modes)?;
    let packed = pack_dirs(pts);
    let pipe = if lay.l_max <= 4 {
        "basis_pres0_l4_dot"
    } else {
        "basis_pres0_l24_dot"
    };
    let mut params = blank_params();
    params.n_per = 1;
    params.n_rays = as_u32(np, "points")?;
    params.n_points = params.n_rays;
    params.l_mode = as_u32(lay.l_max, "pressure degree")?;
    params.l_surf = as_u32(surf.l_max, "surface degree")?;
    params.k_count = as_u32(lay.k_max, "pressure radial count")?;
    params.n_out = as_u32(n, "pressure modes")?;
    params.n_lift = as_u32(nl, "lift modes")?;
    params.poly_stride = stride;
    params.top_pow = as_u32(top, "power index")?;
    let words = launch(
        core,
        pipe,
        &packed,
        &params,
        &coeff,
        &poly_buffer,
        Bind::PresDot,
        np * 2,
        np,
        Some(&coef),
    )?;
    Ok(unpack_pairs(&words.get()))
}
