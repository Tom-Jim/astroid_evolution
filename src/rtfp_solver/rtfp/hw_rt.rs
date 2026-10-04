//! Hardware ray tracing. Every surface query of Tier 1 runs on the GPU
//! ray-tracing units through a Metal primitive acceleration structure.
//!
//! The mesh is stored in `f32` in a frame centred on its bounding sphere and
//! scaled to unit radius, so a hit distance carries a relative error of about
//! `1e-7` of the body size whatever the length unit. All field arithmetic stays
//! in `f64` on the host; the GPU only reports crossing distances.

use std::sync::{Arc, OnceLock};

use crate::common::geometry::TriMesh;
use crate::common::math::DVec3;
use crate::rtfp_solver::error::RtfpError;
use crate::rtfp_solver::rtfp::chords;

/// Crossings recorded per ray in one launch. A ray that fills them is
/// relaunched from its last hit, so a chord is never truncated.
const SLOTS: usize = 12;
/// A first hit counts as the aimed-at surface within this fraction of the
/// aimed distance, plus the absolute floor, both in unit-radius coordinates.
const EXPECT_REL_TOL: f32 = 1.0e-3;
const EXPECT_ABS_TOL: f32 = 2.0e-6;
/// Hits nearer than this (unit-radius coordinates) belong to the launch point's own face and are skipped.
const LAUNCH_OFFSET: f32 = 2.0e-6;

#[derive(Clone, Copy, Debug)]
struct Frame {
    center: DVec3,
    scale: f64,
}

impl Frame {
    fn of(mesh: &TriMesh) -> Self {
        let first = mesh.vertices[0];
        let (mut lo, mut hi) = (first, first);
        for v in &mesh.vertices {
            lo = DVec3::new(lo.x.min(v.x), lo.y.min(v.y), lo.z.min(v.z));
            hi = DVec3::new(hi.x.max(v.x), hi.y.max(v.y), hi.z.max(v.z));
        }
        let center = (lo + hi) * 0.5;
        let scale = mesh
            .vertices
            .iter()
            .map(|v| (*v - center).length())
            .fold(0.0_f64, f64::max);
        Self { center, scale }
    }
}

/// Rays packed for one launch. Origins and directions are mapped into the unit-radius frame.
pub struct RayBatch {
    frame: Frame,
    packed: Vec<[f32; 8]>,
}

impl RayBatch {
    pub fn len(&self) -> usize {
        self.packed.len()
    }

    pub fn is_empty(&self) -> bool {
        self.packed.is_empty()
    }

    /// Move the rays of `other` to the end of this batch.
    pub fn append(&mut self, other: &mut RayBatch) {
        self.packed.append(&mut other.packed);
    }

    /// Queue a ray. `dir` is a unit vector. `expect` is the distance at which the first
    /// crossing must occur for the ray to be reported; `None` reports every crossing.
    pub fn push(&mut self, origin: DVec3, dir: DVec3, expect: Option<f64>) {
        let o = (origin - self.frame.center) * (1.0 / self.frame.scale);
        let expect = expect.map_or(0.0, |d| (d / self.frame.scale) as f32);
        self.packed.push([
            o.x as f32, o.y as f32, o.z as f32, LAUNCH_OFFSET, dir.x as f32, dir.y as f32, dir.z as f32,
            expect,
        ]);
    }
}

/// Crossing distances of a launch, ray by ray, nearest first.
pub struct Hits {
    scale: f64,
    offsets: Vec<u32>,
    t: Vec<f32>,
}

impl Hits {
    pub fn len(&self) -> usize {
        self.offsets.len() - 1
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Distances along ray `i` in world units, appended to `out`.
    pub fn distances_into(&self, i: usize, out: &mut Vec<f64>) {
        let range = self.offsets[i] as usize..self.offsets[i + 1] as usize;
        out.extend(self.t[range].iter().map(|t| *t as f64 * self.scale));
    }

    pub fn count(&self, i: usize) -> usize {
        (self.offsets[i + 1] - self.offsets[i]) as usize
    }
}

#[cfg(target_os = "macos")]
mod metal_rt {
    use std::ffi::c_void;
    use std::ptr::NonNull;

    use objc2::rc::Retained;
    use objc2::runtime::ProtocolObject;
    use objc2_foundation::{NSArray, NSString};
    use objc2_metal::{
        MTLAccelerationStructure, MTLAccelerationStructureCommandEncoder,
        MTLAccelerationStructureGeometryDescriptor,
        MTLAccelerationStructureTriangleGeometryDescriptor, MTLBuffer, MTLCommandBuffer,
        MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
        MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice, MTLGPUFamily,
        MTLIndexType, MTLLibrary, MTLPrimitiveAccelerationStructureDescriptor, MTLResourceOptions,
        MTLSize,
    };

    const SOURCE: &str = include_str!("trace.metal");

    #[link(name = "CoreGraphics", kind = "framework")]
    unsafe extern "C" {}

    type Device = ProtocolObject<dyn MTLDevice>;
    type Queue = ProtocolObject<dyn MTLCommandQueue>;
    type Pipeline = ProtocolObject<dyn MTLComputePipelineState>;
    type Accel = ProtocolObject<dyn MTLAccelerationStructure>;
    type Buffer = ProtocolObject<dyn MTLBuffer>;

    #[repr(C)]
    struct Params {
        max_hits: u32,
        rel_tol: f32,
        abs_tol: f32,
    }

    pub struct Core {
        device: Retained<Device>,
        queue: Retained<Queue>,
        pipeline: Retained<Pipeline>,
        submit: std::sync::Mutex<()>,
    }

    // Metal devices, queues and pipeline states are thread safe objects.
    // Command submission is serialized by `submit`.
    unsafe impl Send for Core {}
    unsafe impl Sync for Core {}

    pub fn init() -> Result<Core, String> {
        let device = MTLCreateSystemDefaultDevice().ok_or("Metal device unavailable")?;
        if !device.supportsRaytracing() {
            return Err("Metal device does not support ray tracing".into());
        }
        // Older Apple GPUs run Metal ray tracing in software. Apple9 and newer
        // have hardware traversal and intersection units.
        if !device.supportsFamily(MTLGPUFamily::Apple9)
            && !device.supportsFamily(MTLGPUFamily::Apple10)
        {
            return Err("RTFP requires an Apple9-or-newer hardware ray-tracing GPU".into());
        }
        let queue = device
            .newCommandQueue()
            .ok_or("Metal command queue unavailable")?;
        let library = device
            .newLibraryWithSource_options_error(&NSString::from_str(SOURCE), None)
            .map_err(|err| format!("Metal shader: {err:?}"))?;
        let function = library
            .newFunctionWithName(&NSString::from_str("trace"))
            .ok_or("trace kernel missing from the Metal library")?;
        let pipeline = device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|err| format!("Metal pipeline: {err:?}"))?;
        Ok(Core {
            device,
            queue,
            pipeline,
            submit: std::sync::Mutex::new(()),
        })
    }

    pub struct Built {
        accel: Retained<Accel>,
        // The structure refers to these buffers for its whole lifetime.
        _vertices: Retained<Buffer>,
        _indices: Retained<Buffer>,
    }

    // The acceleration structure is immutable once the build has completed.
    unsafe impl Send for Built {}
    unsafe impl Sync for Built {}

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

    fn finish(command: &ProtocolObject<dyn MTLCommandBuffer>) -> Result<(), String> {
        command.commit();
        command.waitUntilCompleted();
        if command.status() != MTLCommandBufferStatus::Completed {
            return Err(format!("Metal command failed: {:?}", command.error()));
        }
        Ok(())
    }

    pub fn build(core: &Core, vertices: &[f32], indices: &[u32]) -> Result<Built, String> {
        let options = MTLResourceOptions::StorageModeShared;
        let vertex_buffer = buffer_with(&core.device, vertices)?;
        let index_buffer = buffer_with(&core.device, indices)?;
        let geometry = MTLAccelerationStructureTriangleGeometryDescriptor::new();
        geometry.setVertexBuffer(Some(&vertex_buffer));
        geometry.setVertexStride(12);
        geometry.setIndexBuffer(Some(&index_buffer));
        geometry.setIndexType(MTLIndexType::UInt32);
        geometry.setTriangleCount(indices.len() / 3);
        geometry.setOpaque(true);
        let geometry: Retained<MTLAccelerationStructureGeometryDescriptor> = geometry.into_super();
        let descriptor = MTLPrimitiveAccelerationStructureDescriptor::new();
        descriptor.setGeometryDescriptors(Some(&NSArray::from_retained_slice(&[geometry])));
        let sizes = core
            .device
            .accelerationStructureSizesWithDescriptor(&descriptor);
        let accel = core
            .device
            .newAccelerationStructureWithSize(sizes.accelerationStructureSize)
            .ok_or("Metal acceleration structure allocation failed")?;
        let scratch = core
            .device
            .newBufferWithLength_options(sizes.buildScratchBufferSize.max(1), options)
            .ok_or("Metal scratch buffer allocation failed")?;
        let command = core
            .queue
            .commandBuffer()
            .ok_or("Metal command buffer unavailable")?;
        let encoder = command
            .accelerationStructureCommandEncoder()
            .ok_or("Metal acceleration-structure encoder unavailable")?;
        encoder.buildAccelerationStructure_descriptor_scratchBuffer_scratchBufferOffset(
            &accel,
            &descriptor,
            &scratch,
            0,
        );
        encoder.endEncoding();
        finish(&command)?;
        Ok(Built {
            accel,
            _vertices: vertex_buffer,
            _indices: index_buffer,
        })
    }

    /// Launch `rays` and return per-ray hit counts and `max_hits` distance slots per ray.
    pub fn trace(
        core: &Core,
        built: &Built,
        rays: &[[f32; 8]],
        max_hits: usize,
        rel_tol: f32,
        abs_tol: f32,
    ) -> Result<(Vec<u32>, Vec<f32>), String> {
        let options = MTLResourceOptions::StorageModeShared;
        let ray_buffer = buffer_with(&core.device, rays)?;
        let counts = core
            .device
            .newBufferWithLength_options(rays.len() * 4, options)
            .ok_or("Metal count buffer allocation failed")?;
        let hits = core
            .device
            .newBufferWithLength_options(rays.len() * max_hits * 4, options)
            .ok_or("Metal hit buffer allocation failed")?;
        let params = Params {
            max_hits: max_hits as u32,
            rel_tol,
            abs_tol,
        };
        let _guard = core.submit.lock().unwrap_or_else(|err| err.into_inner());
        let command = core
            .queue
            .commandBuffer()
            .ok_or("Metal command buffer unavailable")?;
        let encoder = command
            .computeCommandEncoder()
            .ok_or("Metal compute encoder unavailable")?;
        encoder.setComputePipelineState(&core.pipeline);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(ray_buffer.as_ref()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(counts.as_ref()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(hits.as_ref()), 0, 2);
            encoder.setBytes_length_atIndex(
                NonNull::from(&params).cast::<c_void>(),
                std::mem::size_of::<Params>(),
                3,
            );
            encoder.setAccelerationStructure_atBufferIndex(Some(built.accel.as_ref()), 4);
        }
        encoder.dispatchThreads_threadsPerThreadgroup(
            MTLSize {
                width: rays.len(),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 64,
                height: 1,
                depth: 1,
            },
        );
        encoder.endEncoding();
        finish(&command)?;
        let counts_ptr = counts.contents().as_ptr() as *const u32;
        let hits_ptr = hits.contents().as_ptr() as *const f32;
        Ok(unsafe {
            (
                std::slice::from_raw_parts(counts_ptr, rays.len()).to_vec(),
                std::slice::from_raw_parts(hits_ptr, rays.len() * max_hits).to_vec(),
            )
        })
    }
}

#[cfg(target_os = "macos")]
type MetalCore = metal_rt::Core;
#[cfg(not(target_os = "macos"))]
type MetalCore = ();

fn core() -> Result<Arc<MetalCore>, RtfpError> {
    static CORE: OnceLock<Result<Arc<MetalCore>, String>> = OnceLock::new();
    CORE.get_or_init(|| {
        #[cfg(target_os = "macos")]
        {
            metal_rt::init().map(Arc::new)
        }
        #[cfg(not(target_os = "macos"))]
        {
            Err("hardware ray tracing requires Metal".to_string())
        }
    })
    .clone()
    .map_err(RtfpError::gpu_unavailable)
}

/// A closed triangle mesh resident in a hardware ray-tracing acceleration structure.
pub struct HardwareMesh {
    #[cfg_attr(not(target_os = "macos"), allow(dead_code))]
    core: Arc<MetalCore>,
    #[cfg(target_os = "macos")]
    built: metal_rt::Built,
    frame: Frame,
}

impl std::fmt::Debug for HardwareMesh {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("HardwareMesh")
    }
}

impl HardwareMesh {
    pub fn build(mesh: &TriMesh) -> Result<Self, RtfpError> {
        let core = core()?;
        let frame = Frame::of(mesh);
        #[cfg(target_os = "macos")]
        {
            let vertices: Vec<f32> = mesh
                .vertices
                .iter()
                .flat_map(|v| {
                    let p = (*v - frame.center) * (1.0 / frame.scale);
                    [p.x as f32, p.y as f32, p.z as f32]
                })
                .collect();
            let indices: Vec<u32> = mesh.faces.iter().flatten().copied().collect();
            let built = metal_rt::build(&core, &vertices, &indices)
                .map_err(RtfpError::gpu_unavailable)?;
            Ok(Self { core, built, frame })
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = (core, frame);
            Err(RtfpError::gpu_unavailable(
                "hardware ray tracing requires Metal",
            ))
        }
    }

    /// Bounding-sphere radius of the mesh. Distances below `1e-6` of it are below the `f32` resolution.
    pub fn scale(&self) -> f64 {
        self.frame.scale
    }

    pub fn new_batch(&self) -> RayBatch {
        RayBatch {
            frame: self.frame,
            packed: Vec::new(),
        }
    }

    /// Launch every ray of `batch`. With `first_only` a ray reports at most its first crossing.
    pub fn trace(&self, batch: &RayBatch, first_only: bool) -> Result<Hits, RtfpError> {
        let slots = if first_only { 1 } else { SLOTS };
        let (counts, slot_t) = self.launch(&batch.packed, slots)?;
        let mut offsets = Vec::with_capacity(counts.len() + 1);
        let mut t = Vec::with_capacity(counts.len() * 2);
        let mut saturated = Vec::new();
        offsets.push(0u32);
        for (i, count) in counts.iter().enumerate() {
            let count = *count as usize;
            t.extend_from_slice(&slot_t[i * slots..i * slots + count]);
            offsets.push(t.len() as u32);
            if !first_only && count == SLOTS {
                saturated.push(i);
            }
        }
        let mut hits = Hits {
            scale: self.frame.scale,
            offsets,
            t,
        };
        if !saturated.is_empty() {
            hits = self.extend_saturated(batch, hits, saturated)?;
        }
        Ok(hits)
    }

    /// Continue every ray that filled its slots, from just past its last crossing.
    fn extend_saturated(
        &self,
        batch: &RayBatch,
        hits: Hits,
        mut pending: Vec<usize>,
    ) -> Result<Hits, RtfpError> {
        let mut extra: Vec<Vec<f32>> = vec![Vec::new(); hits.len()];
        let mut last: Vec<f32> = pending
            .iter()
            .map(|&i| hits.t[hits.offsets[i + 1] as usize - 1])
            .collect();
        while !pending.is_empty() {
            let rays: Vec<[f32; 8]> = pending
                .iter()
                .zip(&last)
                .map(|(&i, &t)| {
                    let mut ray = batch.packed[i];
                    ray[3] = f32::from_bits(t.to_bits() + 1);
                    ray[7] = 0.0;
                    ray
                })
                .collect();
            let (counts, slot_t) = self.launch(&rays, SLOTS)?;
            let mut next = Vec::new();
            let mut next_last = Vec::new();
            for (k, &i) in pending.iter().enumerate() {
                let count = counts[k] as usize;
                extra[i].extend_from_slice(&slot_t[k * SLOTS..k * SLOTS + count]);
                if count == SLOTS {
                    next.push(i);
                    next_last.push(slot_t[k * SLOTS + SLOTS - 1]);
                }
            }
            pending = next;
            last = next_last;
        }
        let mut offsets = Vec::with_capacity(hits.offsets.len());
        let mut t = Vec::with_capacity(hits.t.len());
        offsets.push(0u32);
        for i in 0..hits.len() {
            t.extend_from_slice(&hits.t[hits.offsets[i] as usize..hits.offsets[i + 1] as usize]);
            t.extend_from_slice(&extra[i]);
            offsets.push(t.len() as u32);
        }
        Ok(Hits {
            scale: hits.scale,
            offsets,
            t,
        })
    }

    #[cfg(target_os = "macos")]
    fn launch(&self, rays: &[[f32; 8]], slots: usize) -> Result<(Vec<u32>, Vec<f32>), RtfpError> {
        if rays.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }
        metal_rt::trace(
            &self.core,
            &self.built,
            rays,
            slots,
            EXPECT_REL_TOL,
            EXPECT_ABS_TOL,
        )
        .map_err(RtfpError::gpu_unavailable)
    }

    #[cfg(not(target_os = "macos"))]
    fn launch(&self, _rays: &[[f32; 8]], _slots: usize) -> Result<(Vec<u32>, Vec<f32>), RtfpError> {
        Err(RtfpError::gpu_unavailable(
            "hardware ray tracing requires Metal",
        ))
    }

    /// Whether each point lies inside the solid: the majority parity of three hardware rays.
    pub fn inside(&self, xs: &[DVec3]) -> Result<Vec<bool>, RtfpError> {
        const DIRS: [[f64; 3]; 3] = [
            [0.577_350_269_2, 0.577_350_269_2, 0.577_350_269_2],
            [-0.267_261_241_9, 0.534_522_483_8, 0.801_783_725_7],
            [0.742_781_352_7, -0.557_086_014_5, 0.371_390_676_4],
        ];
        let mut batch = self.new_batch();
        for x in xs {
            for d in DIRS {
                batch.push(*x, DVec3::from_array(d), None);
            }
        }
        let hits = self.trace(&batch, false)?;
        let mut dists = Vec::new();
        Ok((0..xs.len())
            .map(|p| {
                let votes = (0..3)
                    .filter(|&k| {
                        dists.clear();
                        hits.distances_into(3 * p + k, &mut dists);
                        chords::starts_inside(&dists, self.frame.scale)
                    })
                    .count();
                votes >= 2
            })
            .collect())
    }
}
