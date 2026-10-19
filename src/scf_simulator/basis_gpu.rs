//! Metal driver for the Jacobi–spherical-harmonic basis.
//!
//! Every GPU value is double-single, packed `[hi, lo]` with `hi = f32(x)` and
//! `lo = f32(x − hi)`. The kernels in `shaders/basis.metal` implement the same
//! operators as the Taylor jets: `P[Φ] = ∇(Φ + x·∇Φ) − x ΔΦ` and `T[Ψ] = ∇Ψ × x`.
//! Order-3 velocity along a ray reuses the direction jets; `R` and `Y` are
//! homogeneous of degree 0, so `∂^α f(ρ û) = ρ^{−|α|} ∂^α f(û)`.

use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::{Mutex, OnceLock};

use nalgebra::DMatrix;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLCompileOptions, MTLComputeCommandEncoder, MTLComputePipelineState,
    MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary, MTLMathFloatingPointFunctions,
    MTLMathMode, MTLResourceOptions, MTLSize,
};

use crate::common::math::DVec3;
use crate::rheo_friction::basis::{K, ds_pair, pack_dirs, pack_scalars, psi_poly};

use super::basis::{PresLayout, PresRows, VelLayout, VelRows, Want};
use super::harmonics::SurfaceFit;
use super::jets::packed_jet_tables;

const SOURCE: &str = concat!(
    include_str!("../rheo_friction/shaders/ds.metal"),
    "\n",
    include_str!("shaders/basis.metal"),
);

const BATCH_BYTES: usize = 64 * 1024 * 1024;

#[repr(C)]
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

type Device = ProtocolObject<dyn MTLDevice>;
type Queue = ProtocolObject<dyn MTLCommandQueue>;
type Pipeline = ProtocolObject<dyn MTLComputePipelineState>;
type Buffer = ProtocolObject<dyn MTLBuffer>;

struct Core {
    device: Retained<Device>,
    queue: Retained<Queue>,
    tables: Retained<Buffer>,
    vel_o3_l2: Retained<Pipeline>,
    vel_o3_l4: Retained<Pipeline>,
    vel_o2: Retained<Pipeline>,
    vel_o3: Retained<Pipeline>,
    vel_o4: Retained<Pipeline>,
    vel_o3_l2_dot: Retained<Pipeline>,
    vel_o3_l4_dot: Retained<Pipeline>,
    vel_o2_dot: Retained<Pipeline>,
    vel_o3_dot: Retained<Pipeline>,
    vel_o4_dot: Retained<Pipeline>,
    pres0_l4: Retained<Pipeline>,
    pres0_l24: Retained<Pipeline>,
    pres0_l4_dot: Retained<Pipeline>,
    pres0_l24_dot: Retained<Pipeline>,
    pres1: Retained<Pipeline>,
    normal: Retained<Pipeline>,
    gemv: Retained<Pipeline>,
    submit: Mutex<()>,
}

unsafe impl Send for Core {}
unsafe impl Sync for Core {}

fn core() -> Result<&'static Core, String> {
    static SLOT: OnceLock<Result<Core, String>> = OnceLock::new();
    match SLOT.get_or_init(init) {
        Ok(core) => Ok(core),
        Err(err) => Err(err.clone()),
    }
}

fn pipeline(
    device: &Device,
    library: &ProtocolObject<dyn MTLLibrary>,
    name: &str,
) -> Result<Retained<Pipeline>, String> {
    let function = library
        .newFunctionWithName(&NSString::from_str(name))
        .ok_or_else(|| format!("{name} kernel missing from the Metal library"))?;
    device
        .newComputePipelineStateWithFunction_error(&function)
        .map_err(|err| format!("Metal pipeline {name}: {err:?}"))
}

fn init() -> Result<Core, String> {
    let device = MTLCreateSystemDefaultDevice().ok_or("Metal device unavailable")?;
    let queue = device
        .newCommandQueue()
        .ok_or("Metal command queue unavailable")?;
    let options = MTLCompileOptions::new();
    options.setMathMode(MTLMathMode::Safe);
    options.setMathFloatingPointFunctions(MTLMathFloatingPointFunctions::Precise);
    let library = device
        .newLibraryWithSource_options_error(&NSString::from_str(SOURCE), Some(&options))
        .map_err(|err| format!("Metal shader: {err:?}"))?;
    let tables = packed_jet_tables();
    Ok(Core {
        tables: buffer_with(&device, &tables.words)?,
        vel_o3_l2: pipeline(&device, &library, "basis_vel_o3_l2")?,
        vel_o3_l4: pipeline(&device, &library, "basis_vel_o3_l4")?,
        vel_o2: pipeline(&device, &library, "basis_vel_o2")?,
        vel_o3: pipeline(&device, &library, "basis_vel_o3")?,
        vel_o4: pipeline(&device, &library, "basis_vel_o4")?,
        vel_o3_l2_dot: pipeline(&device, &library, "basis_vel_o3_l2_dot")?,
        vel_o3_l4_dot: pipeline(&device, &library, "basis_vel_o3_l4_dot")?,
        vel_o2_dot: pipeline(&device, &library, "basis_vel_o2_dot")?,
        vel_o3_dot: pipeline(&device, &library, "basis_vel_o3_dot")?,
        vel_o4_dot: pipeline(&device, &library, "basis_vel_o4_dot")?,
        pres0_l4: pipeline(&device, &library, "basis_pres0_l4")?,
        pres0_l24: pipeline(&device, &library, "basis_pres0_l24")?,
        pres0_l4_dot: pipeline(&device, &library, "basis_pres0_l4_dot")?,
        pres0_l24_dot: pipeline(&device, &library, "basis_pres0_l24_dot")?,
        pres1: pipeline(&device, &library, "basis_pres1")?,
        normal: pipeline(&device, &library, "basis_normal")?,
        gemv: pipeline(&device, &library, "basis_gemv")?,
        submit: Mutex::new(()),
        device,
        queue,
    })
}

fn buffer_with<T>(device: &Device, data: &[T]) -> Result<Retained<Buffer>, String> {
    let options = MTLResourceOptions::StorageModeShared;
    if data.is_empty() {
        return device
            .newBufferWithLength_options(4, options)
            .ok_or_else(|| "Metal buffer allocation failed".to_string());
    }
    let pointer = NonNull::new(data.as_ptr() as *mut c_void).unwrap();
    unsafe {
        device.newBufferWithBytes_length_options(pointer, std::mem::size_of_val(data), options)
    }
    .ok_or_else(|| "Metal buffer allocation failed".to_string())
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
fn pack_polys(l_max: usize, k_count: usize) -> Result<(Vec<f32>, u32), String> {
    let stride = l_max + k_count;
    let top = l_max + k_count - 1;
    if top > 32 {
        return Err(format!(
            "radial degree l + k = {} exceeds the GPU power table",
            top
        ));
    }
    let mut words = vec![0.0f32; (l_max + 1) * k_count * stride * 2];
    for l in 0..=l_max {
        for k in 0..k_count {
            let poly = psi_poly(l, k);
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

fn ds_at(words: &[f32], channel: usize, n: usize, point: usize) -> f64 {
    let i = (channel * n + point) * 2;
    f64::from(words[i]) + f64::from(words[i + 1])
}

fn write_at(matrix: &mut DMatrix<f64>, row: usize, col: usize, value: f64) {
    let nrows = matrix.nrows();
    debug_assert!(row < nrows && col < matrix.ncols());
    unsafe {
        *matrix.as_mut_ptr().add(col * nrows + row) = value;
    }
}

#[derive(Clone, Copy)]
enum Bind {
    /// Points, params, coefficients, polynomials, tables, output.
    Full,
    /// Pressure value kernel: no monomial tables.
    NoTables,
    /// Normal kernel: no polynomials.
    NoPolys,
    /// Velocity contraction: mode coefficients at buffer 6, output at 5.
    FullDot,
    /// Pressure contraction: mode coefficients at buffer 4, output at 5.
    PresDot,
}

fn launch(
    core: &Core,
    pipe: &Pipeline,
    points: &[f32],
    params: &BasisParams,
    coeff: &Buffer,
    polys: &Buffer,
    bind: Bind,
    n_f32: usize,
    threads: usize,
    coef: Option<&Buffer>,
) -> Result<Vec<f32>, String> {
    let out = launch_buffer(
        core, pipe, points, params, coeff, polys, bind, n_f32, threads, coef,
    )?;
    let words = unsafe { std::slice::from_raw_parts(out.contents().as_ptr() as *const f32, n_f32) };
    Ok(words.to_vec())
}

fn launch_buffer(
    core: &Core,
    pipe: &Pipeline,
    points: &[f32],
    params: &BasisParams,
    coeff: &Buffer,
    polys: &Buffer,
    bind: Bind,
    n_f32: usize,
    threads: usize,
    coef: Option<&Buffer>,
) -> Result<Retained<Buffer>, String> {
    if threads == 0 || n_f32 == 0 {
        return buffer_with(&core.device, &[] as &[f32]);
    }
    let point_buffer = buffer_with(&core.device, points)?;
    let out = core
        .device
        .newBufferWithLength_options(n_f32 * 4, MTLResourceOptions::StorageModeShared)
        .ok_or("Metal output buffer allocation failed")?;
    let width = pipe.threadExecutionWidth();
    let group = (pipe.maxTotalThreadsPerThreadgroup().min(width * 4) / width).max(1) * width;
    let groups = threads.div_ceil(group);
    let _guard = core.submit.lock().unwrap_or_else(|err| err.into_inner());
    let command = core
        .queue
        .commandBuffer()
        .ok_or("Metal command buffer unavailable")?;
    let encoder = command
        .computeCommandEncoder()
        .ok_or("Metal compute encoder unavailable")?;
    encoder.setComputePipelineState(pipe);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(point_buffer.as_ref()), 0, 0);
        encoder.setBytes_length_atIndex(
            NonNull::from(params).cast::<c_void>(),
            std::mem::size_of::<BasisParams>(),
            1,
        );
        encoder.setBuffer_offset_atIndex(Some(coeff.as_ref()), 0, 2);
        match bind {
            Bind::Full => {
                encoder.setBuffer_offset_atIndex(Some(polys.as_ref()), 0, 3);
                encoder.setBuffer_offset_atIndex(Some(core.tables.as_ref()), 0, 4);
                encoder.setBuffer_offset_atIndex(Some(out.as_ref()), 0, 5);
            }
            Bind::NoTables => {
                encoder.setBuffer_offset_atIndex(Some(polys.as_ref()), 0, 3);
                encoder.setBuffer_offset_atIndex(Some(out.as_ref()), 0, 4);
            }
            Bind::NoPolys => {
                encoder.setBuffer_offset_atIndex(Some(core.tables.as_ref()), 0, 3);
                encoder.setBuffer_offset_atIndex(Some(out.as_ref()), 0, 4);
            }
            Bind::FullDot => {
                let coef = coef.ok_or("velocity contraction is missing its coefficient buffer")?;
                encoder.setBuffer_offset_atIndex(Some(polys.as_ref()), 0, 3);
                encoder.setBuffer_offset_atIndex(Some(core.tables.as_ref()), 0, 4);
                encoder.setBuffer_offset_atIndex(Some(out.as_ref()), 0, 5);
                encoder.setBuffer_offset_atIndex(Some(coef.as_ref()), 0, 6);
            }
            Bind::PresDot => {
                let coef = coef.ok_or("pressure contraction is missing its coefficient buffer")?;
                encoder.setBuffer_offset_atIndex(Some(polys.as_ref()), 0, 3);
                encoder.setBuffer_offset_atIndex(Some(coef.as_ref()), 0, 4);
                encoder.setBuffer_offset_atIndex(Some(out.as_ref()), 0, 5);
            }
        }
    }
    encoder.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: groups,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: group,
            height: 1,
            depth: 1,
        },
    );
    encoder.endEncoding();
    command.commit();
    command.waitUntilCompleted();
    if command.status() != MTLCommandBufferStatus::Completed {
        return Err(format!("Metal command failed: {:?}", command.error()));
    }
    Ok(out)
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
    pipe: fn(&Core) -> &Pipeline,
    dot: fn(&Core) -> &Pipeline,
    grouped: bool,
    name_limit: u32,
}

fn velocity_kernel(ord: usize, l_max: usize) -> Result<VelKernel, String> {
    if l_max > 8 {
        return Err(format!("velocity degree {l_max} exceeds the GPU limit 8"));
    }
    if ord == 3 && l_max <= 2 {
        Ok(VelKernel {
            pipe: |c| &c.vel_o3_l2,
            dot: |c| &c.vel_o3_l2_dot,
            grouped: true,
            name_limit: 6,
        })
    } else if ord == 3 && l_max <= 4 {
        Ok(VelKernel {
            pipe: |c| &c.vel_o3_l4,
            dot: |c| &c.vel_o3_l4_dot,
            grouped: true,
            name_limit: 8,
        })
    } else if ord <= 2 {
        Ok(VelKernel {
            pipe: |c| &c.vel_o2,
            dot: |c| &c.vel_o2_dot,
            grouped: false,
            name_limit: 12,
        })
    } else if ord == 3 {
        Ok(VelKernel {
            pipe: |c| &c.vel_o3,
            dot: |c| &c.vel_o3_dot,
            grouped: false,
            name_limit: 12,
        })
    } else if ord == 4 {
        Ok(VelKernel {
            pipe: |c| &c.vel_o4,
            dot: |c| &c.vel_o4_dot,
            grouped: false,
            name_limit: 12,
        })
    } else {
        Err(format!("velocity jet order {ord} is not supported"))
    }
}

pub(crate) fn velocity(
    surf: &SurfaceFit,
    lay: &VelLayout,
    pts: &[DVec3],
    n_per: usize,
    want: Want,
) -> Result<VelRows, String> {
    check_surface(surf)?;
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
    let kernel = velocity_kernel(want.order(), lay.l_max)?;
    let grouped = kernel.grouped;
    let n_per = if grouped { n_per } else { 1 };
    let n_rays = np / n_per;
    let top = lay.l_max + K - 1;
    if top > kernel.name_limit as usize {
        return Err(format!(
            "velocity power index {top} exceeds the {} kernel",
            if grouped { "cached" } else { "point" }
        ));
    }
    let (polys, stride) = pack_polys(lay.l_max, K)?;
    let core = core()?;
    let coeff = buffer_with(&core.device, &pack_scalars(&surf.coeff))?;
    let poly_buffer = buffer_with(&core.device, &polys)?;
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
            (kernel.pipe)(core),
            &packed[word0..word0 + batch_points * 6],
            &params,
            &coeff,
            &poly_buffer,
            Bind::Full,
            batch_points * n_u * n_comp * 2,
            if grouped { batch_rays } else { batch_points },
            None,
        )?;
        scatter_velocity(&mut rows, &words, want, n_comp, n_u, point0, batch_points);
        ray0 += batch_rays;
    }
    Ok(rows)
}

fn scatter_velocity(
    rows: &mut VelRows,
    words: &[f32],
    want: Want,
    n_comp: usize,
    n_u: usize,
    point0: usize,
    n_batch: usize,
) {
    let mut cursor = 0;
    let local_v = cursor;
    if want.v {
        cursor += 3;
    }
    let local_d = cursor;
    if want.d || want.lap || want.gd {
        cursor += 6;
    }
    let local_lap = cursor;
    if want.lap {
        cursor += 3;
    }
    let local_gd = cursor;
    debug_assert_eq!(cursor + if want.gd { 18 } else { 0 }, n_comp);
    for p in 0..n_batch {
        let gp = point0 + p;
        for j in 0..n_u {
            let base = j * n_comp;
            if want.v {
                for a in 0..3 {
                    write_at(
                        &mut rows.v,
                        3 * gp + a,
                        j,
                        ds_at(words, base + local_v + a, n_batch, p),
                    );
                }
            }
            if want.d || want.lap || want.gd {
                for a in 0..6 {
                    write_at(
                        &mut rows.d,
                        6 * gp + a,
                        j,
                        ds_at(words, base + local_d + a, n_batch, p),
                    );
                }
            }
            if want.lap {
                for a in 0..3 {
                    write_at(
                        &mut rows.lap,
                        3 * gp + a,
                        j,
                        ds_at(words, base + local_lap + a, n_batch, p),
                    );
                }
            }
            if want.gd {
                for a in 0..18 {
                    write_at(
                        &mut rows.gd,
                        18 * gp + a,
                        j,
                        ds_at(words, base + local_gd + a, n_batch, p),
                    );
                }
            }
        }
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
    let (polys, stride) = pack_polys(lay.l_max, lay.k_max)?;
    let top = lay.l_max + lay.k_max - 1;
    let core = core()?;
    let coeff = buffer_with(&core.device, &pack_scalars(&surf.coeff))?;
    let poly_buffer = buffer_with(&core.device, &polys)?;
    let packed = pack_dirs(pts);
    let channels = if gradients { (n + nl) * 4 } else { n + nl };
    let per_batch = (BATCH_BYTES / (channels * 8).max(1)).clamp(1, np);
    let pipe = if gradients {
        &core.pres1
    } else if lay.l_max <= 4 {
        &core.pres0_l4
    } else {
        &core.pres0_l24
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
            scatter_pressure_grad(&mut rows, &words, n, nl, point0, batch);
        } else {
            scatter_pressure_value(&mut rows, &words, n, nl, point0, batch);
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

/// Radius and outward unit normal of `r = R(ω̂)` at each direction.
///
/// The kernel returns `(R, ∇R)` of the order-1 jet. The host finishes
/// `n = normalize(ω̂ − ∇R / R)` in f64, the same completion as `SurfaceFit::normal`.
pub(crate) fn normals(surf: &SurfaceFit, dirs: &[DVec3]) -> Result<Vec<(f64, DVec3)>, String> {
    check_surface(surf)?;
    if dirs.is_empty() {
        return Ok(Vec::new());
    }
    let core = core()?;
    let coeff = buffer_with(&core.device, &pack_scalars(&surf.coeff))?;
    let dummy = buffer_with(&core.device, &[] as &[f32])?;
    let packed = pack_dirs(dirs);
    let per_batch = (BATCH_BYTES / 32).clamp(1, dirs.len());
    let mut params = blank_params();
    params.n_per = 1;
    params.l_surf = as_u32(surf.l_max, "surface degree")?;
    params.l_mode = params.l_surf;
    let mut out = Vec::with_capacity(dirs.len());
    let mut point0 = 0;
    while point0 < dirs.len() {
        let batch = per_batch.min(dirs.len() - point0);
        params.n_rays = as_u32(batch, "directions")?;
        params.n_points = params.n_rays;
        let word0 = point0 * 6;
        let words = launch(
            core,
            &core.normal,
            &packed[word0..word0 + batch * 6],
            &params,
            &coeff,
            &dummy,
            Bind::NoPolys,
            batch * 4 * 2,
            batch,
            None,
        )?;
        for p in 0..batch {
            let r = ds_at(&words, 0, batch, p);
            let g = [
                ds_at(&words, 1, batch, p),
                ds_at(&words, 2, batch, p),
                ds_at(&words, 3, batch, p),
            ];
            let w = dirs[point0 + p];
            let n = if r.is_finite() && r != 0.0 {
                let k = 1.0 / r;
                DVec3::new(w.x - g[0] * k, w.y - g[1] * k, w.z - g[2] * k)
                    .try_normalize()
                    .unwrap_or(w)
            } else {
                w
            };
            out.push((r, n));
        }
        point0 += batch;
    }
    Ok(out)
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
    let kernel = velocity_kernel(want.order(), lay.l_max)?;
    let grouped = kernel.grouped;
    let n_per = if grouped { n_per } else { 1 };
    let n_rays = np / n_per;
    let (polys, stride) = pack_polys(lay.l_max, K)?;
    let top = lay.l_max + K - 1;
    if top > kernel.name_limit as usize {
        return Err(format!(
            "velocity power index {top} exceeds the {} kernel",
            if grouped { "cached" } else { "point" }
        ));
    }
    let core = core()?;
    let coeff = buffer_with(&core.device, &pack_scalars(&surf.coeff))?;
    let poly_buffer = buffer_with(&core.device, &polys)?;
    let coef = buffer_with(&core.device, &pack_scalars(u))?;
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
        (kernel.dot)(core),
        &packed,
        &params,
        &coeff,
        &poly_buffer,
        Bind::FullDot,
        np * n_comp * 2,
        if grouped { n_rays } else { np },
        Some(&coef),
    )?;
    let flat = unpack_pairs(&words);
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

/// Face-centre strain basis kept in the kernel's buffer.
///
/// Picard changes `u` every sweep, so the basis has to stay, but it stays on
/// the device. `contract` is one GEMV kernel and returns `6 * n_points` values.
pub(crate) struct GpuStrain {
    basis: Retained<Buffer>,
    n_points: usize,
    n_u: usize,
    n_comp: usize,
}

pub(crate) fn strain_basis(
    surf: &SurfaceFit,
    lay: &VelLayout,
    pts: &[DVec3],
) -> Result<GpuStrain, String> {
    let want = Want {
        v: false,
        d: true,
        lap: false,
        gd: false,
    };
    check_surface(surf)?;
    let np = pts.len();
    let n_comp = 6;
    if np == 0 || lay.n_u == 0 {
        let core = core()?;
        return Ok(GpuStrain {
            basis: buffer_with(&core.device, &[] as &[f32])?,
            n_points: np,
            n_u: lay.n_u,
            n_comp,
        });
    }
    let kernel = velocity_kernel(want.order(), lay.l_max)?;
    let (polys, stride) = pack_polys(lay.l_max, K)?;
    let top = lay.l_max + K - 1;
    if top > kernel.name_limit as usize {
        return Err(format!(
            "velocity power index {top} exceeds the strain kernel"
        ));
    }
    let core = core()?;
    let coeff = buffer_with(&core.device, &pack_scalars(&surf.coeff))?;
    let poly_buffer = buffer_with(&core.device, &polys)?;
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
    params.want_d = 1;
    let basis = launch_buffer(
        core,
        (kernel.pipe)(core),
        &packed,
        &params,
        &coeff,
        &poly_buffer,
        Bind::Full,
        np * lay.n_u * n_comp * 2,
        np,
        None,
    )?;
    Ok(GpuStrain {
        basis,
        n_points: np,
        n_u: lay.n_u,
        n_comp,
    })
}

#[repr(C)]
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
        let core = core()?;
        let coef = buffer_with(&core.device, &pack_scalars(u))?;
        let n_f32 = self.n_points * self.n_comp * 2;
        let out = core
            .device
            .newBufferWithLength_options(n_f32 * 4, MTLResourceOptions::StorageModeShared)
            .ok_or("Metal strain output allocation failed")?;
        let params = GemvParams {
            n_points: as_u32(self.n_points, "strain points")?,
            n_modes: as_u32(self.n_u, "strain modes")?,
            n_comp: as_u32(self.n_comp, "strain components")?,
            pad: 0,
        };
        let pipe = &core.gemv;
        let width = pipe.threadExecutionWidth();
        let group = (pipe.maxTotalThreadsPerThreadgroup().min(width * 4) / width).max(1) * width;
        let groups = self.n_points.div_ceil(group);
        let _guard = core.submit.lock().unwrap_or_else(|err| err.into_inner());
        let command = core
            .queue
            .commandBuffer()
            .ok_or("Metal command buffer unavailable")?;
        let encoder = command
            .computeCommandEncoder()
            .ok_or("Metal compute encoder unavailable")?;
        encoder.setComputePipelineState(pipe);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(self.basis.as_ref()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(coef.as_ref()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(out.as_ref()), 0, 2);
            encoder.setBytes_length_atIndex(
                NonNull::from(&params).cast::<c_void>(),
                std::mem::size_of::<GemvParams>(),
                3,
            );
        }
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: groups,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: group,
                height: 1,
                depth: 1,
            },
        );
        encoder.endEncoding();
        command.commit();
        command.waitUntilCompleted();
        if command.status() != MTLCommandBufferStatus::Completed {
            return Err(format!(
                "Metal strain product failed: {:?}",
                command.error()
            ));
        }
        let words =
            unsafe { std::slice::from_raw_parts(out.contents().as_ptr() as *const f32, n_f32) };
        Ok(unpack_pairs(words))
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
    let (polys, stride) = pack_polys(lay.l_max, lay.k_max)?;
    let top = lay.l_max + lay.k_max - 1;
    let core = core()?;
    let coeff = buffer_with(&core.device, &pack_scalars(&surf.coeff))?;
    let poly_buffer = buffer_with(&core.device, &polys)?;
    let mut modes = pack_scalars(pres_c);
    modes.extend_from_slice(&pack_scalars(lift));
    let coef = buffer_with(&core.device, &modes)?;
    let packed = pack_dirs(pts);
    let pipe = if lay.l_max <= 4 {
        &core.pres0_l4_dot
    } else {
        &core.pres0_l24_dot
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
    Ok(unpack_pairs(&words))
}
