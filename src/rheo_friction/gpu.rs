//! Metal driver of the double-single rheology chain. One thread per node; the
//! mode tables sit in the constant address space and are read uniformly by
//! every lane, the per-simdgroup statistics use `simd_max` / `simd_sum`.

use super::basis::{self, MODE_DS, ds_pair};
use super::error::RheoError;
use super::model::{NodeBatch, StarSurface, VelocityField, local_frame};
use super::params::{Material, MuProjMode};
use crate::common::math::DVec3;

/// DS values written per node.
pub const OUT_DS: usize = 48;
const NODE_DS: usize = 19;

#[derive(Clone, Debug)]
pub struct NodeResult {
    pub radius: f64,
    pub normal: DVec3,
    /// `sqrt(det g)`: surface area per unit solid angle.
    pub area_density: f64,
    pub velocity: DVec3,
    pub e1: DVec3,
    pub e2: DVec3,
    /// `g_ab` in the chart basis.
    pub metric: [f64; 3],
    /// `D_ab` in the chart basis `(D11, D12, D22)`.
    pub d_ab: [f64; 3],
    pub d_nn: f64,
    pub d_na: [f64; 2],
    pub shear: f64,
    pub inertial: f64,
    pub mu_d: f64,
    pub eta: f64,
    pub tau_eq: f64,
    pub p_tilde: f64,
    pub p_star: f64,
    pub p_ref: f64,
    pub mu_proj: f64,
    pub mu_eff: f64,
    pub phi: f64,
    pub sin2_phi: f64,
    pub sin_phi: f64,
    pub sin_slope: f64,
    pub slack: f64,
    pub overhang_margin: f64,
    pub tide_frob: f64,
    pub v_r: f64,
    pub v_n: f64,
    pub s0: f64,
    pub tg: f64,
    pub iterations: u32,
    pub converged: bool,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ChainStats {
    pub max_iterations: u32,
    pub unconverged: u32,
    pub invalid: u32,
    pub seconds: f64,
}

#[derive(Clone, Debug)]
pub struct ChainResult {
    pub nodes: Vec<NodeResult>,
    pub stats: ChainStats,
}

fn ds_value(words: &[f32], i: usize) -> f64 {
    f64::from(words[2 * i]) + f64::from(words[2 * i + 1])
}

fn vec3(words: &[f32], i: usize) -> DVec3 {
    DVec3::new(
        ds_value(words, i),
        ds_value(words, i + 1),
        ds_value(words, i + 2),
    )
}

fn unpack(words: &[f32]) -> NodeResult {
    NodeResult {
        radius: ds_value(words, 0),
        normal: vec3(words, 1),
        area_density: ds_value(words, 4),
        velocity: vec3(words, 5),
        e1: vec3(words, 8),
        e2: vec3(words, 11),
        metric: [
            ds_value(words, 43),
            ds_value(words, 44),
            ds_value(words, 45),
        ],
        d_ab: [
            ds_value(words, 14),
            ds_value(words, 15),
            ds_value(words, 16),
        ],
        d_nn: ds_value(words, 17),
        d_na: [ds_value(words, 18), ds_value(words, 19)],
        shear: ds_value(words, 20),
        inertial: ds_value(words, 21),
        mu_d: ds_value(words, 22),
        eta: ds_value(words, 23),
        tau_eq: ds_value(words, 24),
        p_tilde: ds_value(words, 25),
        p_star: ds_value(words, 26),
        p_ref: ds_value(words, 27),
        mu_proj: ds_value(words, 28),
        mu_eff: ds_value(words, 29),
        phi: ds_value(words, 30),
        sin2_phi: ds_value(words, 31),
        sin_slope: ds_value(words, 32),
        slack: ds_value(words, 33),
        overhang_margin: ds_value(words, 34),
        tide_frob: ds_value(words, 35),
        v_r: ds_value(words, 36),
        v_n: ds_value(words, 37),
        s0: ds_value(words, 38),
        tg: ds_value(words, 39),
        iterations: ds_value(words, 40) as u32,
        converged: ds_value(words, 41) > 0.5,
        sin_phi: ds_value(words, 42),
    }
}

fn pack_nodes(batch: &NodeBatch) -> Vec<f32> {
    let mut out = Vec::with_capacity(batch.len() * NODE_DS * 2);
    let mut push = |x: f64| out.extend_from_slice(&ds_pair(x));
    for i in 0..batch.len() {
        let w = batch.dirs[i]
            .try_normalize()
            .expect("directions validated as unit vectors");
        let (t1, t2) = local_frame(w);
        for v in [w, t1, t2, batch.gravity[i]] {
            push(v.x);
            push(v.y);
            push(v.z);
        }
        for h in batch.hessian_w[i] {
            push(h);
        }
        push(batch.pressure[i]);
    }
    out
}

#[repr(C)]
struct GpuParams {
    node_count: u32,
    l_max: u32,
    max_iter: u32,
    proj_dynamic: u32,
    pad0: u32,
    pad1: u32,
    d_p: [f32; 2],
    rho_p: [f32; 2],
    mu_s: [f32; 2],
    mu_2: [f32; 2],
    i0: [f32; 2],
    coh: [f32; 2],
    p_c: [f32; 2],
    gamma_eps: [f32; 2],
    p_min: [f32; 2],
    rho_bulk: [f32; 2],
    h_s: [f32; 2],
    tide_coef: [f32; 2],
    mu_y: [f32; 2],
    tol: [f32; 2],
    eps_overhang: [f32; 2],
}

fn gpu_params(m: &Material, nodes: usize, l_max: usize) -> GpuParams {
    let s = &m.spec;
    GpuParams {
        node_count: nodes as u32,
        l_max: l_max as u32,
        max_iter: m.max_iter,
        proj_dynamic: u32::from(m.mode == MuProjMode::Dynamic),
        pad0: 0,
        pad1: 0,
        d_p: ds_pair(s.d_p),
        rho_p: ds_pair(s.rho_p),
        mu_s: ds_pair(s.mu_s),
        mu_2: ds_pair(s.mu_2),
        i0: ds_pair(s.i_0),
        coh: ds_pair(s.cohesion),
        p_c: ds_pair(m.p_c),
        gamma_eps: ds_pair(s.gamma_eps),
        p_min: ds_pair(s.p_min),
        rho_bulk: ds_pair(m.rho_bulk),
        h_s: ds_pair(s.h_s),
        tide_coef: ds_pair(m.tide_coef),
        mu_y: ds_pair(m.mu_y),
        tol: ds_pair(m.tol),
        eps_overhang: ds_pair(s.eps_overhang),
    }
}

/// Host-side tables for one evaluation.
struct Tables {
    modes: Vec<f32>,
    recur: Vec<f32>,
    l_max: usize,
}

fn tables(surface: &StarSurface, velocity: &VelocityField) -> Result<Tables, RheoError> {
    let l_max = surface.l_max.max(velocity.l_max);
    let rows = basis::mode_sums(surface, velocity, l_max)?;
    debug_assert_eq!(rows.first().map(|r| r.len()), Some(MODE_DS));
    Ok(Tables {
        modes: basis::mode_table(&rows),
        recur: basis::recurrence_table(l_max),
        l_max,
    })
}

/// Handle to the GPU pipeline. Creating it compiles the Metal library once.
pub struct RheoGpu {
    #[cfg(target_os = "macos")]
    core: metal_rt::Core,
}

impl RheoGpu {
    pub fn new() -> Result<Self, RheoError> {
        #[cfg(target_os = "macos")]
        {
            Ok(Self {
                core: metal_rt::init().map_err(RheoError::gpu_unavailable)?,
            })
        }
        #[cfg(all(target_arch = "wasm32", not(target_os = "macos")))]
        {
            Ok(Self {})
        }
        #[cfg(not(any(target_os = "macos", target_arch = "wasm32")))]
        {
            Err(RheoError::gpu_unavailable(
                "the rheology chain runs on Metal; this platform has no Metal device",
            ))
        }
    }

    pub fn device_name(&self) -> String {
        #[cfg(target_os = "macos")]
        {
            self.core.name()
        }
        #[cfg(all(target_arch = "wasm32", not(target_os = "macos")))]
        {
            let _ = self;
            "WebGPU".to_string()
        }
        #[cfg(not(any(target_os = "macos", target_arch = "wasm32")))]
        {
            String::new()
        }
    }

    /// Run the whole chain on the nodes of `batch`.
    pub fn evaluate(
        &self,
        material: &Material,
        surface: &StarSurface,
        velocity: &VelocityField,
        batch: &NodeBatch,
    ) -> Result<ChainResult, RheoError> {
        batch.validate()?;
        if batch.is_empty() {
            return Ok(ChainResult {
                nodes: Vec::new(),
                stats: ChainStats::default(),
            });
        }
        let t = tables(surface, velocity)?;
        let packed = pack_nodes(batch);
        let params = gpu_params(material, batch.len(), t.l_max);
        #[cfg(target_os = "macos")]
        {
            let (words, stats) = metal_rt::launch(
                &self.core,
                &packed,
                &params,
                &t.modes,
                &t.recur,
                batch.len(),
            )
            .map_err(RheoError::gpu_unavailable)?;
            let nodes = words
                .chunks_exact(OUT_DS * 2)
                .map(unpack)
                .collect::<Vec<_>>();
            Ok(ChainResult { nodes, stats })
        }
        #[cfg(all(target_arch = "wasm32", not(target_os = "macos")))]
        {
            let started = crate::common::time::wall_seconds();
            let nodes_b = crate::browser::slice_bytes(&packed);
            let params_b = crate::browser::pod_bytes(&params);
            let modes_b = crate::browser::slice_bytes(&t.modes);
            let recur_b = crate::browser::slice_bytes(&t.recur);
            let out_len = batch.len() * OUT_DS * 8;
            let buffers = [
                crate::browser::GpuBuffer {
                    binding: 0,
                    bytes: &nodes_b,
                    zeros: 0,
                    readback: false,
                },
                crate::browser::GpuBuffer {
                    binding: 1,
                    bytes: &params_b,
                    zeros: 0,
                    readback: false,
                },
                crate::browser::GpuBuffer {
                    binding: 2,
                    bytes: &modes_b,
                    zeros: 0,
                    readback: false,
                },
                crate::browser::GpuBuffer {
                    binding: 3,
                    bytes: &recur_b,
                    zeros: 0,
                    readback: false,
                },
                crate::browser::GpuBuffer {
                    binding: 4,
                    bytes: &[],
                    zeros: out_len,
                    readback: true,
                },
                crate::browser::GpuBuffer {
                    binding: 5,
                    bytes: &[],
                    zeros: 12,
                    readback: true,
                },
            ];
            let back = crate::browser::dispatch(
                "rheo",
                "rheo_chain",
                crate::browser::groups64(batch.len()),
                &buffers,
            )
            .map_err(RheoError::gpu_unavailable)?;
            let words = crate::browser::f32_vec(back.first().map(Vec::as_slice).unwrap_or(&[]));
            let raw = back.get(1).map(Vec::as_slice).unwrap_or(&[]);
            let mut summary = ChainStats {
                seconds: crate::common::time::wall_seconds() - started,
                ..ChainStats::default()
            };
            if raw.len() >= 12 {
                summary.max_iterations = u32::from_le_bytes(raw[0..4].try_into().unwrap());
                summary.unconverged = u32::from_le_bytes(raw[4..8].try_into().unwrap());
                summary.invalid = u32::from_le_bytes(raw[8..12].try_into().unwrap());
            }
            let nodes = words
                .chunks_exact(OUT_DS * 2)
                .map(unpack)
                .collect::<Vec<_>>();
            Ok(ChainResult {
                nodes,
                stats: summary,
            })
        }
        #[cfg(not(any(target_os = "macos", target_arch = "wasm32")))]
        {
            let _ = (packed, params, t);
            Err(RheoError::gpu_unavailable("no Metal device"))
        }
    }
}

#[cfg(target_os = "macos")]
mod metal_rt {
    use std::ffi::c_void;
    use std::ptr::NonNull;
    use std::time::Instant;

    use objc2::rc::Retained;
    use objc2::runtime::ProtocolObject;
    use objc2_foundation::NSString;
    use objc2_metal::{
        MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
        MTLCompileOptions, MTLComputeCommandEncoder, MTLComputePipelineState,
        MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary, MTLMathFloatingPointFunctions,
        MTLMathMode, MTLResourceOptions, MTLSize,
    };

    use super::{ChainStats, GpuParams, OUT_DS};

    const SOURCE: &str = concat!(
        include_str!("shaders/ds.metal"),
        "\n",
        include_str!("shaders/chain.metal"),
    );

    type Device = ProtocolObject<dyn MTLDevice>;
    type Queue = ProtocolObject<dyn MTLCommandQueue>;
    type Pipeline = ProtocolObject<dyn MTLComputePipelineState>;
    type Buffer = ProtocolObject<dyn MTLBuffer>;

    pub struct Core {
        device: Retained<Device>,
        queue: Retained<Queue>,
        chain: Retained<Pipeline>,
        submit: std::sync::Mutex<()>,
    }

    // Metal devices, queues and pipeline states are thread safe; submission is serialised.
    unsafe impl Send for Core {}
    unsafe impl Sync for Core {}

    impl Core {
        pub fn name(&self) -> String {
            self.device.name().to_string()
        }
    }

    pub fn init() -> Result<Core, String> {
        let device = MTLCreateSystemDefaultDevice().ok_or("Metal device unavailable")?;
        let queue = device
            .newCommandQueue()
            .ok_or("Metal command queue unavailable")?;
        // Safe math keeps every error-free transformation of the DS code intact.
        let options = MTLCompileOptions::new();
        options.setMathMode(MTLMathMode::Safe);
        options.setMathFloatingPointFunctions(MTLMathFloatingPointFunctions::Precise);
        let library = device
            .newLibraryWithSource_options_error(&NSString::from_str(SOURCE), Some(&options))
            .map_err(|err| format!("Metal shader: {err:?}"))?;
        let function = library
            .newFunctionWithName(&NSString::from_str("rheo_chain"))
            .ok_or("rheo_chain kernel missing from the Metal library")?;
        let chain = device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|err| format!("Metal pipeline rheo_chain: {err:?}"))?;
        Ok(Core {
            device,
            queue,
            chain,
            submit: std::sync::Mutex::new(()),
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

    pub fn launch(
        core: &Core,
        nodes: &[f32],
        params: &GpuParams,
        modes: &[f32],
        recur: &[f32],
        count: usize,
    ) -> Result<(Vec<f32>, ChainStats), String> {
        let options = MTLResourceOptions::StorageModeShared;
        let node_buffer = buffer_with(&core.device, nodes)?;
        let mode_buffer = buffer_with(&core.device, modes)?;
        let recur_buffer = buffer_with(&core.device, recur)?;
        let out = core
            .device
            .newBufferWithLength_options(count * OUT_DS * 2 * 4, options)
            .ok_or("Metal output buffer allocation failed")?;
        let width = core.chain.threadExecutionWidth();
        let group =
            (core.chain.maxTotalThreadsPerThreadgroup().min(width * 4) / width).max(1) * width;
        let groups = count.div_ceil(group);
        let simd_groups = group / width;
        let stats = core
            .device
            .newBufferWithLength_options((groups * simd_groups * 3 * 4).max(4), options)
            .ok_or("Metal statistics buffer allocation failed")?;
        let _guard = core.submit.lock().unwrap_or_else(|err| err.into_inner());
        let started = Instant::now();
        let command = core
            .queue
            .commandBuffer()
            .ok_or("Metal command buffer unavailable")?;
        let encoder = command
            .computeCommandEncoder()
            .ok_or("Metal compute encoder unavailable")?;
        encoder.setComputePipelineState(&core.chain);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(node_buffer.as_ref()), 0, 0);
            encoder.setBytes_length_atIndex(
                NonNull::from(params).cast::<c_void>(),
                std::mem::size_of::<GpuParams>(),
                1,
            );
            encoder.setBuffer_offset_atIndex(Some(mode_buffer.as_ref()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(recur_buffer.as_ref()), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(out.as_ref()), 0, 4);
            encoder.setBuffer_offset_atIndex(Some(stats.as_ref()), 0, 5);
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
        let seconds = started.elapsed().as_secs_f64();
        let words = unsafe {
            std::slice::from_raw_parts(out.contents().as_ptr() as *const f32, count * OUT_DS * 2)
        }
        .to_vec();
        let raw = unsafe {
            std::slice::from_raw_parts(
                stats.contents().as_ptr() as *const u32,
                groups * simd_groups * 3,
            )
        };
        let mut summary = ChainStats {
            seconds,
            ..ChainStats::default()
        };
        for s in raw.chunks_exact(3) {
            summary.max_iterations = summary.max_iterations.max(s[0]);
            summary.unconverged += s[1];
            summary.invalid += s[2];
        }
        Ok((words, summary))
    }
}
