//! Metal deposition for mode C.
//!
//! The host builds the same direction bins and cap boxes as the CPU kernel.
//! Two compute passes then normalise each source and scatter the deposition.
//! A missing device or a shader failure is returned. There is no CPU substitute
//! on macOS; the caller keeps the reference implementation for tests.

use crate::common::math::DVec3;

#[cfg(target_os = "macos")]
pub fn deposit(
    dirs: &[DVec3],
    area: &[f64],
    mass_rate: &[f64],
    bins_g: usize,
    bin_start: &[u32],
    bin_items: &[u32],
    caps: &[u32],
    kernel_width: f64,
    cutoff: f64,
    cos_reject: f64,
    dense: bool,
) -> Result<Vec<f64>, String> {
    let n = dirs.len();
    if area.len() != n || mass_rate.len() != n {
        return Err(format!(
            "mode C kernel: {} directions, {} areas, {} mass rates",
            n,
            area.len(),
            mass_rate.len()
        ));
    }
    if n == 0 {
        return Ok(Vec::new());
    }
    if !dense && caps.len() != n * 6 {
        return Err(format!(
            "mode C kernel: {} cap bounds for {n} vertices",
            caps.len()
        ));
    }
    if bins_g == 0 || bin_start.len() != bins_g * bins_g * bins_g + 1 || bin_items.len() != n {
        return Err(format!(
            "mode C kernel: bin grid {bins_g}, {} starts, {} items, {n} vertices",
            bin_start.len(),
            bin_items.len()
        ));
    }
    metal_rt::launch(
        dirs,
        area,
        mass_rate,
        bins_g,
        bin_start,
        bin_items,
        caps,
        kernel_width,
        cutoff,
        cos_reject,
        dense,
    )
}

#[cfg(not(target_os = "macos"))]
pub fn deposit(
    dirs: &[DVec3],
    area: &[f64],
    mass_rate: &[f64],
    bins_g: usize,
    bin_start: &[u32],
    bin_items: &[u32],
    caps: &[u32],
    kernel_width: f64,
    cutoff: f64,
    cos_reject: f64,
    dense: bool,
) -> Result<Vec<f64>, String> {
    let _ = (
        dirs,
        area,
        mass_rate,
        bins_g,
        bin_start,
        bin_items,
        caps,
        kernel_width,
        cutoff,
        cos_reject,
        dense,
    );
    Err("mode C deposition requires the Metal double-single kernels".into())
}

fn unpack_scalars(words: &[f32]) -> Vec<f64> {
    words
        .chunks_exact(2)
        .map(|p| f64::from(p[0]) + f64::from(p[1]))
        .collect()
}

#[cfg(target_os = "macos")]
mod metal_rt {
    use std::ffi::c_void;
    use std::ptr::NonNull;
    use std::sync::{Mutex, OnceLock};

    use objc2::rc::Retained;
    use objc2::runtime::ProtocolObject;
    use objc2_foundation::NSString;
    use objc2_metal::{
        MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
        MTLCompileOptions, MTLComputeCommandEncoder, MTLComputePipelineState,
        MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary, MTLMathFloatingPointFunctions,
        MTLMathMode, MTLResourceOptions, MTLSize,
    };

    use super::unpack_scalars;
    use crate::rheo_friction::basis::{pack_dirs, pack_scalars};
    use crate::common::math::DVec3;
    use crate::rheo_friction::basis::ds_pair;

    const SOURCE: &str = concat!(
        include_str!("../rheo_friction/shaders/ds.metal"),
        "\n",
        include_str!("shaders/mode_c.metal"),
    );

    #[repr(C)]
    struct ModeParams {
        n: u32,
        g: u32,
        dense: u32,
        pad: u32,
        w_hi: f32,
        w_lo: f32,
        cutoff_hi: f32,
        cutoff_lo: f32,
        reject_hi: f32,
        reject_lo: f32,
    }

    const _: () = assert!(std::mem::size_of::<ModeParams>() == 40);

    type Device = ProtocolObject<dyn MTLDevice>;
    type Queue = ProtocolObject<dyn MTLCommandQueue>;
    type Pipeline = ProtocolObject<dyn MTLComputePipelineState>;
    type Buffer = ProtocolObject<dyn MTLBuffer>;

    struct Core {
        device: Retained<Device>,
        queue: Retained<Queue>,
        norm: Retained<Pipeline>,
        deposit: Retained<Pipeline>,
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
            .map_err(|err| format!("mode C shader: {err:?}"))?;
        let (norm, deposit) = {
            let pipeline = |name: &str| -> Result<Retained<Pipeline>, String> {
                let function = library
                    .newFunctionWithName(&NSString::from_str(name))
                    .ok_or_else(|| format!("{name} kernel missing from the Metal library"))?;
                device
                    .newComputePipelineStateWithFunction_error(&function)
                    .map_err(|err| format!("Metal pipeline {name}: {err:?}"))
            };
            (
                pipeline("mode_c_norm")?,
                pipeline("mode_c_deposit")?,
            )
        };
        Ok(Core {
            device,
            queue,
            norm,
            deposit,
            submit: Mutex::new(()),
        })
    }

    fn buffer_with<T>(device: &Device, data: &[T]) -> Result<Retained<Buffer>, String> {
        let options = MTLResourceOptions::StorageModeShared;
        if data.is_empty() {
            return device
                .newBufferWithLength_options(16, options)
                .ok_or_else(|| "Metal buffer allocation failed".to_string());
        }
        let pointer = NonNull::new(data.as_ptr() as *mut c_void).unwrap();
        unsafe {
            device.newBufferWithBytes_length_options(pointer, std::mem::size_of_val(data), options)
        }
        .ok_or_else(|| "Metal buffer allocation failed".to_string())
    }

    fn pair(x: f64) -> (f32, f32) {
        let p = ds_pair(x);
        (p[0], p[1])
    }

    pub fn launch(
        dirs: &[DVec3],
        area: &[f64],
        mass_rate: &[f64],
        bins_g: usize,
        bin_start: &[u32],
        bin_items: &[u32],
        caps: &[u32],
        kernel_width: f64,
        cutoff: f64,
        cos_reject: f64,
        dense: bool,
    ) -> Result<Vec<f64>, String> {
        let core = core()?;
        let n = dirs.len();
        let (w_hi, w_lo) = pair(kernel_width);
        let (cutoff_hi, cutoff_lo) = pair(cutoff);
        let (reject_hi, reject_lo) = if dense {
            (0.0, 0.0)
        } else {
            pair(cos_reject)
        };
        let params = ModeParams {
            n: u32::try_from(n).map_err(|_| format!("mode C kernel: {n} vertices"))?,
            g: u32::try_from(bins_g).map_err(|_| format!("mode C kernel: bin grid {bins_g}"))?,
            dense: u32::from(dense),
            pad: 0,
            w_hi,
            w_lo,
            cutoff_hi,
            cutoff_lo,
            reject_hi,
            reject_lo,
        };
        let dir_buf = buffer_with(&core.device, &pack_dirs(dirs))?;
        let area_buf = buffer_with(&core.device, &pack_scalars(area))?;
        let rate_buf = buffer_with(&core.device, &pack_scalars(mass_rate))?;
        let start_buf = buffer_with(&core.device, bin_start)?;
        let item_buf = buffer_with(&core.device, bin_items)?;
        let cap_buf = buffer_with(&core.device, caps)?;
        let words = n * 2;
        let norm = core
            .device
            .newBufferWithLength_options(words * 4, MTLResourceOptions::StorageModeShared)
            .ok_or("Metal mode C norm allocation failed")?;
        let out = core
            .device
            .newBufferWithLength_options(words * 4, MTLResourceOptions::StorageModeShared)
            .ok_or("Metal mode C deposition allocation failed")?;
        let _guard = core.submit.lock().unwrap_or_else(|err| err.into_inner());
        let command = core
            .queue
            .commandBuffer()
            .ok_or("Metal command buffer unavailable")?;
        let encoder = command
            .computeCommandEncoder()
            .ok_or("Metal compute encoder unavailable")?;
        dispatch(
            &encoder,
            &core.norm,
            n,
            &[
                dir_buf.as_ref(),
                area_buf.as_ref(),
                rate_buf.as_ref(),
                start_buf.as_ref(),
                item_buf.as_ref(),
                cap_buf.as_ref(),
                norm.as_ref(),
            ],
            &params,
        );
        dispatch(
            &encoder,
            &core.deposit,
            n,
            &[
                dir_buf.as_ref(),
                rate_buf.as_ref(),
                norm.as_ref(),
                start_buf.as_ref(),
                item_buf.as_ref(),
                cap_buf.as_ref(),
                out.as_ref(),
            ],
            &params,
        );
        encoder.endEncoding();
        command.commit();
        command.waitUntilCompleted();
        if command.status() != MTLCommandBufferStatus::Completed {
            return Err(format!("mode C kernel failed: {:?}", command.error()));
        }
        let written =
            unsafe { std::slice::from_raw_parts(out.contents().as_ptr() as *const f32, words) };
        Ok(unpack_scalars(written))
    }

    fn dispatch(
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        pipe: &Pipeline,
        n: usize,
        buffers: &[&ProtocolObject<dyn MTLBuffer>],
        params: &ModeParams,
    ) {
        encoder.setComputePipelineState(pipe);
        unsafe {
            for (index, buf) in buffers.iter().enumerate() {
                encoder.setBuffer_offset_atIndex(Some(*buf), 0, index);
            }
            encoder.setBytes_length_atIndex(
                NonNull::from(params).cast::<c_void>(),
                std::mem::size_of::<ModeParams>(),
                buffers.len(),
            );
        }
        let width = pipe.threadExecutionWidth();
        let group = (pipe.maxTotalThreadsPerThreadgroup().min(width * 4) / width).max(1) * width;
        let groups = n.div_ceil(group);
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
    }
}
