//! GPU LBVH resident on wasm. The mesh frame matches the Metal path: vertices
//! are stored in `f32`, centred on the bounding sphere and scaled to unit
//! radius. The hierarchy is built on the GPU. Field, moment and trace launches
//! use the same buffers. macOS does not compile this module.

use crate::common::geometry::TriMesh;
use crate::common::math::DVec3;
use crate::rtfp_solver::rtfp::chords::{MERGE_ABS, MERGE_REL, ORIGIN_ABS};

use super::{DeviceKernel, FieldJob, LAUNCH_OFFSET};

pub struct Resident {
    nodes: Vec<u8>,
    tris: Vec<u8>,
}

pub fn resident(mesh: &TriMesh, center: DVec3, scale: f64) -> Result<Resident, String> {
    let inv = 1.0 / scale;
    let mut tris = Vec::with_capacity(mesh.faces.len() * 9);
    for face in &mesh.faces {
        for corner in face {
            tris.extend_from_slice(&place(mesh.vertices[*corner as usize], center, inv));
        }
    }
    let nodes = if tris.is_empty() {
        leaf_bytes([0.0; 3], [0.0; 3], u32::MAX)
    } else if tris.len() == 9 {
        let (min, max) = expand_leaf(&tris);
        leaf_bytes(min, max, 0)
    } else {
        crate::browser::build_lbvh(&crate::browser::slice_bytes(&tris))?
    };
    Ok(Resident {
        nodes,
        tris: crate::browser::slice_bytes(&tris),
    })
}

fn leaf_bytes(min: [f32; 3], max: [f32; 3], prim: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity(32);
    for value in min {
        out.extend_from_slice(&value.to_le_bytes());
    }
    out.extend_from_slice(&prim.to_le_bytes());
    for value in max {
        out.extend_from_slice(&value.to_le_bytes());
    }
    out.extend_from_slice(&u32::MAX.to_le_bytes());
    out
}

fn expand_leaf(tris: &[f32]) -> ([f32; 3], [f32; 3]) {
    let mut min = [f32::MAX; 3];
    let mut max = [f32::MIN; 3];
    for value in tris.chunks_exact(3) {
        for axis in 0..3 {
            min[axis] = min[axis].min(value[axis]);
            max[axis] = max[axis].max(value[axis]);
        }
    }
    for axis in 0..3 {
        let pad = (max[axis] - min[axis]).abs().max(1.0e-6) * 1.0e-4 + 1.0e-6;
        min[axis] -= pad;
        max[axis] += pad;
    }
    (min, max)
}

fn place(v: DVec3, center: DVec3, inv: f64) -> [f32; 3] {
    let p = (v - center) * inv;
    [p.x as f32, p.y as f32, p.z as f32]
}

#[repr(C)]
struct GpuPoint {
    ox: f32,
    oy: f32,
    oz: f32,
    theta: f32,
    wx: f32,
    wy: f32,
    wz: f32,
    rho: f32,
}

#[repr(C)]
struct GpuRay {
    dx: f32,
    dy: f32,
    dz: f32,
    weight: f32,
}

#[repr(C)]
struct GpuKernel {
    cx: f32,
    cy: f32,
    cz: f32,
    inv_sigma2: f32,
    coeff: f32,
    alpha: f32,
}

#[repr(C)]
struct FieldParams {
    kernel_count: u32,
    node_count: u32,
    node_hi: u32,
    ray_count: u32,
    want_potential: u32,
    want_hessian: u32,
    bulk: f32,
    ell: f32,
    body_scale: f32,
    g: f32,
    merge_rel: f32,
    merge_abs: f32,
    origin_abs: f32,
    launch_offset: f32,
    point_count: u32,
}

#[repr(C)]
struct MomentParams {
    kernel_count: u32,
    node_count: u32,
    ray_count: u32,
    max_panels: u32,
    bulk: f32,
    finest: f32,
    body_scale: f32,
    launch_offset: f32,
    merge_rel: f32,
    merge_abs: f32,
    origin_abs: f32,
    ox: f32,
    oy: f32,
    oz: f32,
    wx: f32,
    wy: f32,
    wz: f32,
    pad: f32,
}

#[repr(C)]
struct TraceParams {
    max_hits: u32,
    rel_tol: f32,
    abs_tol: f32,
}

const _: () = assert!(std::mem::size_of::<FieldParams>() == 60);
const _: () = assert!(std::mem::size_of::<MomentParams>() == 72);
const _: () = assert!(std::mem::size_of::<TraceParams>() == 12);

fn pack_kernels(items: &[DeviceKernel]) -> Vec<GpuKernel> {
    items
        .iter()
        .map(|k| GpuKernel {
            cx: k.center.x as f32,
            cy: k.center.y as f32,
            cz: k.center.z as f32,
            inv_sigma2: k.inv_sigma2 as f32,
            coeff: k.coeff as f32,
            alpha: k.alpha as f32,
        })
        .collect()
}

fn column(words: &[f32]) -> Result<[f64; 10], String> {
    if words.len() < 20 {
        return Err("RTFP GPU integral is short".into());
    }
    let mut col = [0.0; 10];
    for c in 0..10 {
        let value = f64::from(words[c * 2]) + f64::from(words[c * 2 + 1]);
        if !value.is_finite() {
            return Err("RTFP GPU integral is not finite".into());
        }
        col[c] = value;
    }
    Ok(col)
}

pub fn field(
    resident: &Resident,
    center: DVec3,
    scale: f64,
    job: &FieldJob<'_>,
) -> Result<Vec<[f64; 10]>, String> {
    if !(scale.is_finite() && scale > 0.0) {
        return Err("mesh scale is not a positive finite length".into());
    }
    if job.points.is_empty() {
        return Ok(Vec::new());
    }
    let inv = 1.0 / scale;
    let gpu_points: Vec<GpuPoint> = job
        .points
        .iter()
        .zip(job.thetas)
        .zip(job.rhos)
        .map(|((p, &theta), &rho)| {
            let unit = (*p - center) * inv;
            GpuPoint {
                ox: unit.x as f32,
                oy: unit.y as f32,
                oz: unit.z as f32,
                theta: theta as f32,
                wx: p.x as f32,
                wy: p.y as f32,
                wz: p.z as f32,
                rho: rho as f32,
            }
        })
        .collect();
    let gpu_rays: Vec<GpuRay> = job
        .dirs
        .iter()
        .zip(job.weights)
        .map(|(d, &w)| GpuRay {
            dx: d.x as f32,
            dy: d.y as f32,
            dz: d.z as f32,
            weight: w as f32,
        })
        .collect();
    let gpu_kernels = pack_kernels(job.kernels);
    if job.node_base as usize > job.nodes.len() {
        return Err("Gauss node table is shorter than the base rule".into());
    }
    let params = FieldParams {
        kernel_count: gpu_kernels.len() as u32,
        node_count: job.node_base,
        node_hi: job.nodes.len() as u32 - job.node_base,
        ray_count: gpu_rays.len() as u32,
        want_potential: u32::from(job.want_potential),
        want_hessian: u32::from(job.want_hessian),
        bulk: job.bulk as f32,
        ell: scale as f32,
        body_scale: scale as f32,
        g: job.g as f32,
        merge_rel: MERGE_REL as f32,
        merge_abs: (MERGE_ABS * scale) as f32,
        origin_abs: (ORIGIN_ABS * scale) as f32,
        launch_offset: LAUNCH_OFFSET,
        point_count: u32::try_from(gpu_points.len())
            .map_err(|_| "too many field points for one launch".to_string())?,
    };
    let points_b = crate::browser::slice_bytes(&gpu_points);
    let rays_b = crate::browser::slice_bytes(&gpu_rays);
    let kernels_b = crate::browser::slice_bytes(&gpu_kernels);
    let nodes_b = crate::browser::slice_bytes(job.nodes);
    let params_b = crate::browser::pod_bytes(&params);
    let back = dispatch(
        "field",
        "rtfp_field",
        crate::browser::groups64(gpu_points.len()),
        &[
            (0, &points_b, 0, false),
            (1, &rays_b, 0, false),
            (2, &kernels_b, 0, false),
            (3, &nodes_b, 0, false),
            (4, &params_b, 0, false),
            (5, &[], gpu_points.len() * 80, true),
            (6, &[], gpu_points.len() * 4, true),
            (7, &resident.nodes, 0, false),
            (8, &resident.tris, 0, false),
        ],
    )?;
    let flags = crate::browser::f32_vec(&back[1]);
    if flags.iter().any(|flag| flag.to_bits() != 0) {
        return Err("a ray crossed the body more times than the register chord table holds".into());
    }
    let words = crate::browser::f32_vec(&back[0]);
    let mut columns = Vec::with_capacity(gpu_points.len());
    for p in 0..gpu_points.len() {
        columns.push(column(&words[p * 20..])?);
    }
    Ok(columns)
}

pub fn moments(
    resident: &Resident,
    center: DVec3,
    scale: f64,
    origin: DVec3,
    dirs: &[DVec3],
    weights: &[f64],
    bulk: f64,
    kernels: &[DeviceKernel],
    nodes: &[[f32; 2]],
    finest: f64,
    max_panels: u32,
) -> Result<[f64; 10], String> {
    if dirs.is_empty() {
        return Ok([0.0; 10]);
    }
    let inv = 1.0 / scale;
    let unit = (origin - center) * inv;
    let gpu_rays: Vec<GpuRay> = dirs
        .iter()
        .zip(weights)
        .map(|(d, &w)| GpuRay {
            dx: d.x as f32,
            dy: d.y as f32,
            dz: d.z as f32,
            weight: w as f32,
        })
        .collect();
    let gpu_kernels = pack_kernels(kernels);
    let params = MomentParams {
        kernel_count: gpu_kernels.len() as u32,
        node_count: nodes.len() as u32,
        ray_count: gpu_rays.len() as u32,
        max_panels,
        bulk: bulk as f32,
        finest: finest as f32,
        body_scale: scale as f32,
        launch_offset: LAUNCH_OFFSET,
        merge_rel: MERGE_REL as f32,
        merge_abs: (MERGE_ABS * scale) as f32,
        origin_abs: (ORIGIN_ABS * scale) as f32,
        ox: unit.x as f32,
        oy: unit.y as f32,
        oz: unit.z as f32,
        wx: origin.x as f32,
        wy: origin.y as f32,
        wz: origin.z as f32,
        pad: 0.0,
    };
    let rays_b = crate::browser::slice_bytes(&gpu_rays);
    let kernels_b = crate::browser::slice_bytes(&gpu_kernels);
    let nodes_b = crate::browser::slice_bytes(nodes);
    let params_b = crate::browser::pod_bytes(&params);
    let back = dispatch(
        "field",
        "rtfp_moments",
        [1, 1, 1],
        &[
            (0, &rays_b, 0, false),
            (1, &kernels_b, 0, false),
            (2, &nodes_b, 0, false),
            (3, &params_b, 0, false),
            (4, &[], 80, true),
            (5, &[], 4, true),
            (7, &resident.nodes, 0, false),
            (8, &resident.tris, 0, false),
        ],
    )?;
    if crate::browser::f32_vec(&back[1])
        .first()
        .is_some_and(|flag| flag.to_bits() != 0)
    {
        return Err("a ray crossed the body more times than the register chord table holds".into());
    }
    column(&crate::browser::f32_vec(&back[0]))
}

pub fn trace(
    resident: &Resident,
    rays: &[[f32; 8]],
    slots: usize,
    rel_tol: f32,
    abs_tol: f32,
) -> Result<(Vec<u32>, Vec<f32>), String> {
    if rays.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }
    let params = TraceParams {
        max_hits: slots as u32,
        rel_tol,
        abs_tol,
    };
    let rays_b = crate::browser::slice_bytes(rays);
    let params_b = crate::browser::pod_bytes(&params);
    let back = dispatch(
        "trace",
        "trace",
        crate::browser::groups64(rays.len()),
        &[
            (0, &rays_b, 0, false),
            (1, &[], rays.len() * 4, true),
            (2, &[], rays.len() * slots * 4, true),
            (3, &params_b, 0, false),
            (7, &resident.nodes, 0, false),
            (8, &resident.tris, 0, false),
        ],
    )?;
    let counts = back[0]
        .chunks_exact(4)
        .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
        .collect();
    Ok((counts, crate::browser::f32_vec(&back[1])))
}

fn dispatch(
    shader: &str,
    entry: &str,
    groups: [u32; 3],
    buffers: &[(u32, &[u8], usize, bool)],
) -> Result<Vec<Vec<u8>>, String> {
    let gpu: Vec<crate::browser::GpuBuffer<'_>> = buffers
        .iter()
        .map(
            |(binding, bytes, zeros, readback)| crate::browser::GpuBuffer {
                binding: *binding,
                bytes,
                zeros: *zeros,
                readback: *readback,
            },
        )
        .collect();
    crate::browser::dispatch(shader, entry, groups, &gpu)
}
