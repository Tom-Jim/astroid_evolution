//! Double-single face frames for the §10.4.3 slope row.
//!
//! The production path is the Metal kernel. A missing device, a shader failure,
//! or a face index that does not exist is returned. There is no CPU substitute.

use crate::common::math::DVec3;

/// Unit normal `n̂` and the three columns `∂n̂/∂R_j` of one face.
#[derive(Clone, Copy, Debug)]
pub struct FaceFrame {
    pub n: DVec3,
    pub nj: [DVec3; 3],
}

/// `positions` are the vertices scaled by `1/R₀`. `dirs` are the fixed radial units.
pub fn face_frames(
    positions: &[DVec3],
    dirs: &[DVec3],
    faces: &[[u32; 3]],
) -> Result<Vec<FaceFrame>, String> {
    if positions.len() != dirs.len() {
        return Err(format!(
            "face-frame kernel: {} positions and {} directions",
            positions.len(),
            dirs.len()
        ));
    }
    let nv = positions.len();
    for (f, tri) in faces.iter().enumerate() {
        for v in tri {
            if *v as usize >= nv {
                return Err(format!(
                    "face {f} indexes vertex {v}, but the mesh has {nv} vertices"
                ));
            }
        }
    }
    if faces.is_empty() {
        return Ok(Vec::new());
    }
    #[cfg(target_os = "macos")]
    {
        metal_rt::launch(positions, dirs, faces)
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (positions, dirs, faces);
        Err("SOCP face frames require the Metal double-single kernel".into())
    }
}

fn unpack_frames(words: &[f32], nf: usize) -> Result<Vec<FaceFrame>, String> {
    if words.len() != nf * 24 {
        return Err(format!(
            "face-frame kernel returned {} floats, expected {}",
            words.len(),
            nf * 24
        ));
    }
    let mut out = Vec::with_capacity(nf);
    for f in 0..nf {
        let b = f * 24;
        let load = |k: usize| {
            let i = b + k * 6;
            DVec3::new(
                f64::from(words[i]) + f64::from(words[i + 1]),
                f64::from(words[i + 2]) + f64::from(words[i + 3]),
                f64::from(words[i + 4]) + f64::from(words[i + 5]),
            )
        };
        out.push(FaceFrame {
            n: load(0),
            nj: [load(1), load(2), load(3)],
        });
    }
    Ok(out)
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

    use super::{FaceFrame, unpack_frames};
    use crate::rheo_friction::basis::pack_dirs;
    use crate::common::math::DVec3;

    const SOURCE: &str = concat!(
        include_str!("../rheo_friction/shaders/ds.metal"),
        "\n",
        include_str!("shaders/socp_normal.metal"),
    );

    type Device = ProtocolObject<dyn MTLDevice>;
    type Queue = ProtocolObject<dyn MTLCommandQueue>;
    type Pipeline = ProtocolObject<dyn MTLComputePipelineState>;
    type Buffer = ProtocolObject<dyn MTLBuffer>;

    struct Core {
        device: Retained<Device>,
        queue: Retained<Queue>,
        frame: Retained<Pipeline>,
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
            .map_err(|err| format!("SOCP face-frame shader: {err:?}"))?;
        let function = library
            .newFunctionWithName(&NSString::from_str("socp_face_frame"))
            .ok_or("socp_face_frame kernel missing from the Metal library")?;
        let frame = device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|err| format!("Metal pipeline socp_face_frame: {err:?}"))?;
        Ok(Core {
            device,
            queue,
            frame,
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

    pub fn launch(
        positions: &[DVec3],
        dirs: &[DVec3],
        faces: &[[u32; 3]],
    ) -> Result<Vec<FaceFrame>, String> {
        let core = core()?;
        let nf = faces.len();
        let pos = pack_dirs(positions);
        let dir = pack_dirs(dirs);
        let index: Vec<u32> = faces.iter().flat_map(|t| *t).collect();
        let pos_buf = buffer_with(&core.device, &pos)?;
        let dir_buf = buffer_with(&core.device, &dir)?;
        let face_buf = buffer_with(&core.device, &index)?;
        let out = core
            .device
            .newBufferWithLength_options(nf * 24 * 4, MTLResourceOptions::StorageModeShared)
            .ok_or("Metal face-frame output allocation failed")?;
        let width = core.frame.threadExecutionWidth();
        let group =
            (core.frame.maxTotalThreadsPerThreadgroup().min(width * 4) / width).max(1) * width;
        let groups = nf.div_ceil(group);
        let nfaces = nf as u32;
        let _guard = core.submit.lock().unwrap_or_else(|err| err.into_inner());
        let command = core
            .queue
            .commandBuffer()
            .ok_or("Metal command buffer unavailable")?;
        let encoder = command
            .computeCommandEncoder()
            .ok_or("Metal compute encoder unavailable")?;
        encoder.setComputePipelineState(&core.frame);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(pos_buf.as_ref()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(dir_buf.as_ref()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(face_buf.as_ref()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(out.as_ref()), 0, 3);
            encoder.setBytes_length_atIndex(
                NonNull::from(&nfaces).cast::<c_void>(),
                std::mem::size_of::<u32>(),
                4,
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
                "SOCP face-frame kernel failed: {:?}",
                command.error()
            ));
        }
        let words =
            unsafe { std::slice::from_raw_parts(out.contents().as_ptr() as *const f32, nf * 24) };
        unpack_frames(words, nf)
    }
}
