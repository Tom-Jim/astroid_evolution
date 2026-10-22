//! Double-single face frames for the §10.4.3 slope row.
//!
//! `scf/shader/socp_normal.wesl`. A shader failure or a face index that does not
//! exist is returned.

use crate::common::math::DVec3;
use crate::gpu::Arg;
use crate::rheo::basis::pack_dirs;

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
    let nf = faces.len();
    let index: Vec<u32> = faces.iter().copied().flatten().collect();
    let k = &crate::gpu::kernels::SOCP_FACE_FRAME;
    let back = crate::gpu::once(
        k,
        k.groups(nf),
        &[
            Arg::In(bytemuck::cast_slice(&pack_dirs(positions))),
            Arg::In(bytemuck::cast_slice(&pack_dirs(dirs))),
            Arg::In(bytemuck::cast_slice(&index)),
            Arg::Out(nf * 24 * 4),
            Arg::In(bytemuck::bytes_of(&(nf as u32))),
        ],
    )?;
    unpack_frames(&bytemuck::pod_collect_to_vec::<u8, f32>(&back[0]), nf)
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
