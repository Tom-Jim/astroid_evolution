//! Metal evaluation of the filtered expansion. One thread per vertex.

use super::error::ShError;
use super::harmonics::{self, check_l_max, coeff_count};

#[derive(Clone, Debug)]
pub struct EvalStats {
    pub radii: Vec<f64>,
    /// Vertices whose device radius is not strictly positive.
    pub nonpositive: u32,
    pub seconds: f64,
}

/// Handle to the GPU pipeline. Creating it compiles the Metal library once.
pub struct ShGpu {
    #[cfg(target_os = "macos")]
    core: metal_rt::Core,
}

impl ShGpu {
    pub fn new() -> Result<Self, ShError> {
        #[cfg(target_os = "macos")]
        {
            Ok(Self {
                core: metal_rt::init().map_err(ShError::gpu_unavailable)?,
            })
        }
        #[cfg(not(target_os = "macos"))]
        {
            Err(ShError::gpu_unavailable(
                "spherical-harmonic evaluation runs on Metal; this platform has no Metal device",
            ))
        }
    }

    pub fn device_name(&self) -> String {
        #[cfg(target_os = "macos")]
        {
            self.core.name()
        }
        #[cfg(not(target_os = "macos"))]
        {
            String::new()
        }
    }

    /// Evaluate `Σ c_lm Y_lm` at each direction. Directions are renormalised on the host.
    pub fn evaluate(
        &self,
        l_max: usize,
        coeff: &[f64],
        dirs: &[[f64; 3]],
    ) -> Result<EvalStats, ShError> {
        self.dispatch(l_max, coeff, dirs, None)
    }

    /// Device radius plus a per-vertex residual: `R_final = R_sh + residual`.
    pub fn evaluate_with_residual(
        &self,
        l_max: usize,
        coeff: &[f64],
        dirs: &[[f64; 3]],
        residual: &[f64],
    ) -> Result<EvalStats, ShError> {
        if residual.len() != dirs.len() {
            return Err(ShError::invalid_input(format!(
                "residual length {} does not match {} directions",
                residual.len(),
                dirs.len()
            )));
        }
        if residual.iter().any(|r| !r.is_finite()) {
            return Err(ShError::invalid_input("residuals must be finite"));
        }
        self.dispatch(l_max, coeff, dirs, Some(residual))
    }

    fn dispatch(
        &self,
        l_max: usize,
        coeff: &[f64],
        dirs: &[[f64; 3]],
        residual: Option<&[f64]>,
    ) -> Result<EvalStats, ShError> {
        check_l_max(l_max)?;
        let n = coeff_count(l_max);
        if coeff.len() != n {
            return Err(ShError::invalid_input(format!(
                "expected {n} coefficients at l_max {l_max}, got {}",
                coeff.len()
            )));
        }
        if coeff.iter().any(|c| !c.is_finite()) {
            return Err(ShError::invalid_input("coefficients must be finite"));
        }
        if dirs.is_empty() {
            return Ok(EvalStats {
                radii: Vec::new(),
                nonpositive: 0,
                seconds: 0.0,
            });
        }
        let mut unit = Vec::with_capacity(dirs.len());
        for (i, d) in dirs.iter().enumerate() {
            let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            if !(len.is_finite() && len > 0.0) || !d.iter().all(|v| v.is_finite()) {
                return Err(ShError::geometry(format!(
                    "direction {i} is not a finite nonzero vector"
                )));
            }
            unit.push([d[0] / len, d[1] / len, d[2] / len]);
        }
        let packed_dirs = harmonics::pack_directions(&unit);
        let packed_coeff = harmonics::pack_coefficients(l_max, coeff);
        let recur = harmonics::recurrence_table(l_max);
        let packed_residual = residual.map(harmonics::pack_scalars);
        #[cfg(target_os = "macos")]
        {
            let (words, nonpositive, seconds) = metal_rt::launch(
                &self.core,
                &packed_dirs,
                &packed_coeff,
                &recur,
                packed_residual.as_deref(),
                l_max,
                unit.len(),
            )
            .map_err(ShError::gpu_unavailable)?;
            let radii = words
                .chunks_exact(2)
                .map(|w| f64::from(w[0]) + f64::from(w[1]))
                .collect();
            Ok(EvalStats {
                radii,
                nonpositive,
                seconds,
            })
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = (packed_dirs, packed_coeff, recur, packed_residual);
            Err(ShError::gpu_unavailable("no Metal device"))
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

    const SOURCE: &str = concat!(
        include_str!("shaders/ds.metal"),
        "\n",
        include_str!("shaders/eval.metal"),
    );

    type Device = ProtocolObject<dyn MTLDevice>;
    type Queue = ProtocolObject<dyn MTLCommandQueue>;
    type Pipeline = ProtocolObject<dyn MTLComputePipelineState>;
    type Buffer = ProtocolObject<dyn MTLBuffer>;

    pub struct Core {
        device: Retained<Device>,
        queue: Retained<Queue>,
        pipeline: Retained<Pipeline>,
        submit: std::sync::Mutex<()>,
    }

    unsafe impl Send for Core {}
    unsafe impl Sync for Core {}

    impl Core {
        pub fn name(&self) -> String {
            self.device.name().to_string()
        }
    }

    #[repr(C)]
    struct GpuParams {
        vertex_count: u32,
        l_max: u32,
        add_residual: u32,
        pad1: u32,
    }

    pub fn init() -> Result<Core, String> {
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
        let function = library
            .newFunctionWithName(&NSString::from_str("sh_eval"))
            .ok_or("sh_eval kernel missing from the Metal library")?;
        let pipeline = device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|err| format!("Metal pipeline sh_eval: {err:?}"))?;
        Ok(Core {
            device,
            queue,
            pipeline,
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
        dirs: &[f32],
        coeff: &[f32],
        recur: &[f32],
        residual: Option<&[f32]>,
        l_max: usize,
        count: usize,
    ) -> Result<(Vec<f32>, u32, f64), String> {
        let options = MTLResourceOptions::StorageModeShared;
        let dir_buffer = buffer_with(&core.device, dirs)?;
        let coeff_buffer = buffer_with(&core.device, coeff)?;
        let recur_buffer = buffer_with(&core.device, recur)?;
        let residual_buffer = match residual {
            Some(words) => buffer_with(&core.device, words)?,
            None => buffer_with(&core.device, &[0.0f32, 0.0])?,
        };
        let out = core
            .device
            .newBufferWithLength_options(count * 2 * 4, options)
            .ok_or("Metal output buffer allocation failed")?;
        let params = GpuParams {
            vertex_count: count as u32,
            l_max: l_max as u32,
            add_residual: u32::from(residual.is_some()),
            pad1: 0,
        };
        let width = core.pipeline.threadExecutionWidth();
        let group =
            (core.pipeline.maxTotalThreadsPerThreadgroup().min(width * 4) / width).max(1) * width;
        let groups = count.div_ceil(group);
        let simd_groups = group / width;
        let stats = core
            .device
            .newBufferWithLength_options((groups * simd_groups * 4).max(4), options)
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
        encoder.setComputePipelineState(&core.pipeline);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(dir_buffer.as_ref()), 0, 0);
            encoder.setBytes_length_atIndex(
                NonNull::from(&params).cast::<c_void>(),
                std::mem::size_of::<GpuParams>(),
                1,
            );
            encoder.setBuffer_offset_atIndex(Some(coeff_buffer.as_ref()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(recur_buffer.as_ref()), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(out.as_ref()), 0, 4);
            encoder.setBuffer_offset_atIndex(Some(stats.as_ref()), 0, 5);
            encoder.setBuffer_offset_atIndex(Some(residual_buffer.as_ref()), 0, 6);
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
        let words =
            unsafe { std::slice::from_raw_parts(out.contents().as_ptr() as *const f32, count * 2) }
                .to_vec();
        let raw = unsafe {
            std::slice::from_raw_parts(
                stats.contents().as_ptr() as *const u32,
                groups * simd_groups,
            )
        };
        let nonpositive = raw.iter().sum();
        Ok((words, nonpositive, seconds))
    }
}
