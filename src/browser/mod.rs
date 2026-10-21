//! Browser side of the viewer.
//!
//! The page and the evolution worker are two copies of this module. The page
//! owns the WebGPU device and the Bevy window. The worker runs `evolve::run`
//! and sends snapshots back. A compute call from the worker posts its bytes to
//! the page, then waits. The page is never blocked inside that wait, so the
//! canvas keeps drawing.

use std::sync::atomic::{AtomicBool, Ordering};

mod gpu;
mod host;
mod wire;

pub use gpu::{PdhgJob, PdhgOut, dispatch_resident, execute_gpu, keep_output, release_resident};
pub use host::{
    boot, deliver, prepare_viewer, set_pause_flag, set_quit_flag, spawn_evolution, worker_main,
};

static WORKER_MESH: AtomicBool = AtomicBool::new(false);

/// The evolution worker can post a compute call and wait. The page cannot:
/// WebGPU runs there, and the page must stay free to service that wait.
pub fn use_worker_mesh() {
    WORKER_MESH.store(true, Ordering::Relaxed);
}

pub fn worker_mesh() -> bool {
    WORKER_MESH.load(Ordering::Relaxed)
}

/// One storage buffer for a compute dispatch.
pub struct GpuBuffer<'a> {
    /// `@binding` index in the entry's shader.
    pub binding: u32,
    pub bytes: &'a [u8],
    /// Device bytes to allocate and zero. `bytes` is not copied across.
    pub zeros: usize,
    /// Copy this buffer back after the dispatch.
    pub readback: bool,
}

pub fn groups64(n: usize) -> [u32; 3] {
    [
        u32::try_from(n).unwrap_or(u32::MAX).div_ceil(64).max(1),
        1,
        1,
    ]
}

pub fn pod_bytes<T>(value: &T) -> Vec<u8> {
    unsafe {
        std::slice::from_raw_parts(
            std::ptr::from_ref(value).cast::<u8>(),
            std::mem::size_of::<T>(),
        )
        .to_vec()
    }
}

pub fn slice_bytes<T>(value: &[T]) -> Vec<u8> {
    unsafe {
        std::slice::from_raw_parts(value.as_ptr().cast::<u8>(), std::mem::size_of_val(value))
            .to_vec()
    }
}

pub fn f32_vec(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|word| f32::from_le_bytes(word.try_into().unwrap()))
        .collect()
}

/// Run one compute entry on the page's device. Called from the worker.
pub fn dispatch(
    shader: &str,
    entry: &str,
    groups: [u32; 3],
    buffers: &[GpuBuffer<'_>],
) -> Result<Vec<Vec<u8>>, String> {
    gpu::dispatch_single(shader, entry, groups, buffers)
}

/// Run a block of PDHG iterations without bringing the vectors back each kernel.
pub fn dispatch_pdhg(job: &gpu::PdhgJob) -> Result<gpu::PdhgOut, String> {
    gpu::dispatch_pdhg(job)
}

/// Build a GPU LBVH over packed triangles (`9` floats each) and return the nodes.
pub fn build_lbvh(tris: &[u8]) -> Result<Vec<u8>, String> {
    gpu::dispatch_lbvh(tris)
}

fn mesh_primitive_from_bytes(
    pos_count: u32,
    index_count: u32,
    positions: &[u8],
    indices: &[u8],
    nonfinite: u32,
    bad_index: u32,
) -> Result<(Vec<crate::common::math::DVec3>, Vec<[u32; 3]>), String> {
    if nonfinite != 0 {
        return Err("mesh vertex is not finite".into());
    }
    if bad_index != 0 {
        return Err("mesh face index is outside the vertex array".into());
    }
    let pos_bytes = pos_count as usize * 12;
    let idx_bytes = index_count as usize * 4;
    if positions.len() < pos_bytes || indices.len() < idx_bytes {
        return Err("mesh load readback is short".into());
    }
    let mut vertices = Vec::with_capacity(pos_count as usize);
    for chunk in positions[..pos_bytes].chunks_exact(12) {
        vertices.push(crate::common::math::DVec3::new(
            f32::from_le_bytes(chunk[0..4].try_into().unwrap()) as f64,
            f32::from_le_bytes(chunk[4..8].try_into().unwrap()) as f64,
            f32::from_le_bytes(chunk[8..12].try_into().unwrap()) as f64,
        ));
    }
    let mut faces = Vec::with_capacity(index_count as usize / 3);
    for chunk in indices[..idx_bytes].chunks_exact(12) {
        faces.push([
            u32::from_le_bytes(chunk[0..4].try_into().unwrap()),
            u32::from_le_bytes(chunk[4..8].try_into().unwrap()),
            u32::from_le_bytes(chunk[8..12].try_into().unwrap()),
        ]);
    }
    Ok((vertices, faces))
}

/// Decode one indexed triangle primitive on the GPU. The matrix is the glTF
/// world transform; length-unit scaling stays with the caller.
pub fn load_mesh_primitive(
    bin: &[u8],
    pos_count: u32,
    pos_offset: u32,
    pos_stride: u32,
    index_count: u32,
    index_offset: u32,
    index_stride: u32,
    index_width: u32,
    matrix: [f64; 16],
) -> Result<(Vec<crate::common::math::DVec3>, Vec<[u32; 3]>), String> {
    let matrix = matrix.map(|value| value as f32);
    let (positions, indices, nonfinite, bad_index) = gpu::dispatch_mesh_load(
        bin,
        pos_count,
        pos_offset,
        pos_stride,
        index_count,
        index_offset,
        index_stride,
        index_width,
        matrix,
    )?;
    mesh_primitive_from_bytes(
        pos_count,
        index_count,
        &positions,
        &indices,
        nonfinite,
        bad_index,
    )
}

/// Same decode as [`load_mesh_primitive`], run on the page's own device.
pub async fn load_mesh_primitive_local(
    bin: &[u8],
    pos_count: u32,
    pos_offset: u32,
    pos_stride: u32,
    index_count: u32,
    index_offset: u32,
    index_stride: u32,
    index_width: u32,
    matrix: [f64; 16],
) -> Result<(Vec<crate::common::math::DVec3>, Vec<[u32; 3]>), String> {
    let matrix = matrix.map(|value| value as f32);
    let (positions, indices, nonfinite, bad_index) = gpu::mesh_load_local(
        bin,
        pos_count,
        pos_offset,
        pos_stride,
        index_count,
        index_offset,
        index_stride,
        index_width,
        matrix,
    )
    .await?;
    mesh_primitive_from_bytes(
        pos_count,
        index_count,
        &positions,
        &indices,
        nonfinite,
        bad_index,
    )
}

/// Separate overlapping radial cones. Vertices are unchanged when nothing overlaps.
pub fn align_radial(
    vertices: &mut [crate::common::math::DVec3],
    faces: &[[u32; 3]],
) -> Result<(usize, usize, f64), String> {
    use crate::common::math::DVec3;
    let nvert = u32::try_from(vertices.len()).map_err(|_| "too many vertices for GPU alignment")?;
    let nface = u32::try_from(faces.len()).map_err(|_| "too many faces for GPU alignment")?;
    let packed_v = pack_ds_vertices(vertices);
    let mut packed_f = Vec::with_capacity(faces.len() * 12);
    for face in faces {
        for index in face {
            packed_f.extend_from_slice(&index.to_le_bytes());
        }
    }
    let (passes, bad, overflow, moved) =
        gpu::dispatch_mesh_align(&packed_v, &packed_f, nvert, nface)?;
    if overflow != 0 {
        return Err("radial alignment vertex degree exceeds the GPU neighbor table".into());
    }
    let mut max_offset = 0.0f64;
    if passes > 0 {
        if moved.len() < vertices.len() * 24 {
            return Err("mesh align readback is short".into());
        }
        for (index, vertex) in vertices.iter_mut().enumerate() {
            let base = index * 24;
            let next = DVec3::new(
                unpack_ds(&moved[base..base + 8]),
                unpack_ds(&moved[base + 8..base + 16]),
                unpack_ds(&moved[base + 16..base + 24]),
            );
            max_offset = max_offset.max((*vertex - next).length());
            *vertex = next;
        }
    }
    Ok((passes as usize, bad as usize, max_offset))
}

/// GPU broad-phase self-intersection. A hit uses the same error text as the CPU check.
pub fn reject_self_intersection(
    vertices: &[crate::common::math::DVec3],
    faces: &[[u32; 3]],
    scale: f64,
) -> Result<(), String> {
    if faces.len() < 2 {
        return Ok(());
    }
    let nvert =
        u32::try_from(vertices.len()).map_err(|_| "too many vertices for GPU intersection")?;
    let nface = u32::try_from(faces.len()).map_err(|_| "too many faces for GPU intersection")?;
    let packed_v = pack_ds_vertices(vertices);
    let mut packed_f = Vec::with_capacity(faces.len() * 12);
    for face in faces {
        for index in face {
            packed_f.extend_from_slice(&index.to_le_bytes());
        }
    }
    let (scale_hi, scale_lo) = if scale.is_finite() {
        let hi = scale as f32;
        (hi, (scale - f64::from(hi)) as f32)
    } else {
        (0.0, 0.0)
    };
    let (hit, lo, hi, gap) =
        gpu::dispatch_mesh_isect(&packed_v, &packed_f, nvert, nface, scale_hi, scale_lo)?;
    if hit != 0 {
        return Err(format!(
            "triangles {lo} and {hi} intersect (gap {gap:.4} m)"
        ));
    }
    Ok(())
}

fn pack_ds_vertices(vertices: &[crate::common::math::DVec3]) -> Vec<u8> {
    let mut packed = Vec::with_capacity(vertices.len() * 24);
    for vertex in vertices {
        for component in [vertex.x, vertex.y, vertex.z] {
            let hi = component as f32;
            let lo = (component - f64::from(hi)) as f32;
            packed.extend_from_slice(&hi.to_le_bytes());
            packed.extend_from_slice(&lo.to_le_bytes());
        }
    }
    packed
}

fn unpack_ds(bytes: &[u8]) -> f64 {
    let hi = f32::from_le_bytes(bytes[0..4].try_into().unwrap());
    let lo = f32::from_le_bytes(bytes[4..8].try_into().unwrap());
    f64::from(hi) + f64::from(lo)
}
