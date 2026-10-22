//! GPU deposition for mode C (`scf/shader/mode_c.wesl`).
//!
//! The host builds the direction bins and cap boxes of the CPU kernel. Two
//! passes in one submission normalise each source and scatter the deposition;
//! the normalisation never leaves the device.

use crate::common::math::DVec3;
use crate::gpu::{Job, kernels};
use crate::rheo::basis::{ds_pair, pack_dirs, pack_scalars};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ModeParams {
    n: u32,
    g: u32,
    dense: u32,
    pad: u32,
    w: [f32; 2],
    cutoff: [f32; 2],
    reject: [f32; 2],
}

#[allow(clippy::too_many_arguments)]
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
    let params = ModeParams {
        n: u32::try_from(n).map_err(|_| format!("mode C kernel: {n} vertices"))?,
        g: u32::try_from(bins_g).map_err(|_| format!("mode C kernel: bin grid {bins_g}"))?,
        dense: u32::from(dense),
        pad: 0,
        w: ds_pair(kernel_width),
        cutoff: ds_pair(cutoff),
        reject: if dense {
            [0.0, 0.0]
        } else {
            ds_pair(cos_reject)
        },
    };
    let mut job = Job::new();
    let d = job.input(&pack_dirs(dirs));
    let a = job.input(&pack_scalars(area));
    let r = job.input(&pack_scalars(mass_rate));
    let s = job.input(bin_start);
    let i = job.input(bin_items);
    let c = job.input(caps);
    let norm = job.zeros::<f32>(2 * n);
    let out = job.zeros::<f32>(2 * n);
    let p = job.value(&params);
    let (kn, kd) = (&kernels::MODE_C_NORM, &kernels::MODE_C_DEPOSIT);
    job.run(kn, kn.groups(n), &[d, a, r, s, i, c, norm, p]);
    job.run(kd, kd.groups(n), &[d, r, norm, s, i, c, out, p]);
    let ro = job.read_all::<f32>(out, 2 * n);
    let words = job.submit()?.get::<f32>(ro);
    Ok(words
        .chunks_exact(2)
        .map(|w| f64::from(w[0]) + f64::from(w[1]))
        .collect())
}
