//! GPU evaluation of the (G1) geometry-adapted basis: the row blocks
//! `v / d / lap / gd` of `VelRows<f32>` at arbitrary point sets, and the
//! face-centre strain rows the friction chain contracts every iteration.
//!
//! Row storage matches the enclosing-ball convention (column-major
//! `DMatrix<f32>`, row `3p + a` of `v`, `6p + c` of `d`, `18p + 6k + c` of
//! `gd`); the kernel is `scf/shader/gbasis.wesl`.

use nalgebra::DMatrix;

use crate::common::math::DVec3;
use crate::gpu::{self, Arg};

use super::basis::{ContractedVelocity, VelRows, Want};
use super::gbasis::{GeoBasis, Mode};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GbParams {
    n_points: u32,
    n_modes: u32,
    want_v: u32,
    want_d: u32,
    want_lap: u32,
    want_gd: u32,
    pad0: u32,
    pad1: u32,
}

const _: () = assert!(std::mem::size_of::<GbParams>() == 32);

/// The modes packed as `8 f32` each: `f0..f2`, `f3..f5`, `h`, `kind`.
fn pack_modes(geo: &GeoBasis) -> Vec<f32> {
    let mut out = Vec::with_capacity(geo.modes.len() * 8);
    for m in &geo.modes {
        let (f0, f1, h, kind) = match *m {
            Mode::Rigid { a, w } => (a, w, 0.0, 0.0),
            Mode::Kernel { c, e, h } => (c, e, h as f32, 1.0),
            Mode::Flux { z } => (z, DVec3::ZERO, 0.0, 2.0),
            Mode::Poly { e, ijk } => (
                e,
                DVec3::new(ijk[0] as f64, ijk[1] as f64, ijk[2] as f64),
                0.0,
                3.0,
            ),
        };
        out.extend([
            f0.x as f32,
            f0.y as f32,
            f0.z as f32,
            f1.x as f32,
            f1.y as f32,
            f1.z as f32,
            h,
            kind,
        ]);
    }
    out
}

fn pack_pts(pts: &[DVec3]) -> Vec<f32> {
    let mut out = Vec::with_capacity(pts.len() * 3);
    for p in pts {
        out.extend([p.x as f32, p.y as f32, p.z as f32]);
    }
    out
}

fn ncols(geo: &GeoBasis) -> usize {
    geo.modes.len()
}

fn mat(rows: usize, cols: usize, data: Vec<f32>) -> DMatrix<f32> {
    DMatrix::from_vec(rows, cols, data)
}

fn empty(cols: usize) -> DMatrix<f32> {
    DMatrix::zeros(0, cols)
}

/// The `want` row blocks of `geo` at `pts`, in the `VelRows` layout.
///
/// Blocks not asked for come back empty (`0 × n_modes`); consumers only read
/// the blocks their `want` flagged, exactly as for the spectral family.
pub fn rows(geo: &GeoBasis, pts: &[DVec3], want: Want) -> Result<VelRows<f32>, String> {
    let (np, nm) = (pts.len(), ncols(geo));
    if np == 0 || nm == 0 {
        return Ok(VelRows {
            n_points: np,
            v: empty(nm),
            d: empty(nm),
            lap: empty(nm),
            gd: empty(nm),
        });
    }
    let params = GbParams {
        n_points: np as u32,
        n_modes: nm as u32,
        want_v: u32::from(want.v),
        want_d: u32::from(want.d),
        want_lap: u32::from(want.lap),
        want_gd: u32::from(want.gd),
        pad0: 0,
        pad1: 0,
    };
    let size = |block: usize, wanted: bool| {
        if wanted { block * np * nm * 4 } else { 4 }
    };
    let back = gpu::once(
        gpu::kernel("gbasis_rows")?,
        [np.div_ceil(64) as u32, 1, 1],
        &[
            Arg::In(bytemuck::cast_slice(&pack_pts(pts))),
            Arg::In(bytemuck::bytes_of(&params)),
            Arg::In(bytemuck::cast_slice(&pack_modes(geo))),
            Arg::Out(size(3, want.v)),
            Arg::Out(size(6, want.d)),
            Arg::Out(size(3, want.lap)),
            Arg::Out(size(18, want.gd)),
        ],
    )?;
    let grab = |i: usize, block: usize, wanted: bool| {
        if wanted {
            mat(block * np, nm, bytemuck::pod_collect_to_vec(&back[i]))
        } else {
            empty(nm)
        }
    };
    Ok(VelRows {
        n_points: np,
        v: grab(0, 3, want.v),
        d: grab(1, 6, want.d),
        lap: grab(2, 3, want.lap),
        gd: grab(3, 18, want.gd),
    })
}

/// `Σ u_b B_b` at `pts`: the velocity (`want.v`) and scaled strain (`want.d`)
/// of one coefficient vector, via rows + host multiply.
pub fn contract(
    geo: &GeoBasis,
    pts: &[DVec3],
    want: Want,
    u: &[f64],
) -> Result<ContractedVelocity, String> {
    let rows = rows(geo, pts, want)?;
    let u = nalgebra::DVector::from_column_slice(u);
    let v = if want.v {
        crate::common::linalg::mul(&rows.v, &u).as_slice().to_vec()
    } else {
        Vec::new()
    };
    let d = if want.d {
        crate::common::linalg::mul(&rows.d, &u).as_slice().to_vec()
    } else {
        Vec::new()
    };
    Ok(ContractedVelocity { v, d })
}
