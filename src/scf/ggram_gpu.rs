//! Host side of the (G1) Gram assembly of `scf/shader/ggram.wesl`: the mass
//! Gram `∫ρw B_b·B_c` and the strain Gram `∫ρw D_b:D_c` of Proposition 8.1.3
//! in one submission.
//!
//! Where `gbasis::gram_matrix` / `strain_gram_matrix` scan all `n_b²` mode
//! pairs and sort the union of two support lists per surviving pair, the GPU
//! path builds per-support point lists once (sorted by construction), splits
//! the Gram into kernel×kernel, kernel×global and global×global blocks, and
//! lets each workgroup accumulate its output pair over the smaller support
//! list — no pair scan, no union sort.

use nalgebra::DMatrix;
use std::collections::HashMap;

use crate::common::math::DVec3;
use crate::gpu::{self, Job};

use super::gbasis::{GeoBasis, Mode};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GgParams {
    n_points: u32,
    n_modes: u32,
    n_supp: u32,
    n_global: u32,
    n_kk: u32,
    n_gk: u32,
    n_gg: u32,
    want_strain: u32,
}

const _: () = assert!(std::mem::size_of::<GgParams>() == 32);

fn groups_of(n: usize) -> [u32; 3] {
    [n.div_ceil(64).max(1) as u32, 1, 1]
}

/// The mode packing of `gbasis_gpu.rs` (8 f32 per mode, shared layout).
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
            f0.x as f32, f0.y as f32, f0.z as f32, f1.x as f32, f1.y as f32, f1.z as f32, h, kind,
        ]);
    }
    out
}

/// The unique kernel supports `(c, h)` of the basis, the first axis-mode of
/// each (the three axes of one support are consecutive in `GeoBasis::build`),
/// and the mode indices of every non-kernel mode.
struct Split {
    /// `(cx, cy, cz, h)` per unique support, in first-appearance order.
    supps: Vec<[f32; 4]>,
    /// First of the three consecutive axis modes of each support.
    base: Vec<u32>,
    /// Mode indices of rigid, poly and flux modes.
    globals: Vec<u32>,
}

fn split(geo: &GeoBasis) -> Result<Split, String> {
    let mut supps = Vec::new();
    let mut base = Vec::new();
    let mut globals = Vec::new();
    let mut seen: HashMap<(u64, u64, u64, u64), u32> = HashMap::new();
    for (b, m) in geo.modes.iter().enumerate() {
        let Mode::Kernel { c, h, .. } = *m else {
            globals.push(b as u32);
            continue;
        };
        let key = (c.x.to_bits(), c.y.to_bits(), c.z.to_bits(), h.to_bits());
        if seen.contains_key(&key) {
            continue;
        }
        // A new support: verify the axis-1 and axis-2 modes follow
        // consecutively with the same (c, h), as `GeoBasis::build` emits them.
        let follow = [(DVec3::X, 0usize), (DVec3::Y, 1), (DVec3::Z, 2)];
        for (ax, k) in follow {
            match geo.modes.get(b + k) {
                Some(Mode::Kernel { c: cc, e: ee, h: hh }) if *cc == c && *hh == h && *ee == ax => {}
                _ => return Err("(G1) kernel axes of a support are not consecutive".into()),
            }
        }
        seen.insert(key, supps.len() as u32);
        base.push(b as u32);
        supps.push([c.x as f32, c.y as f32, c.z as f32, h as f32]);
    }
    Ok(Split { supps, base, globals })
}

/// Mass and strain Gram of `geo` on the quadrature `(pts, w, rho)`.
///
/// `want_strain == false` skips the strain accumulators (the second `prune`
/// stage and `gram_check` need only the mass Gram). Returns `(mass, strain)`
/// as dense symmetric `n_b × n_b` matrices; `strain` is all zeros when not
/// asked for.
pub fn grams(
    geo: &GeoBasis,
    pts: &[DVec3],
    w: &[f64],
    rho: Option<&[f64]>,
    want_strain: bool,
) -> Result<(DMatrix<f64>, DMatrix<f64>), String> {
    let (np, nb) = (pts.len(), geo.modes.len());
    if np == 0 || nb == 0 {
        return Ok((DMatrix::zeros(nb, nb), DMatrix::zeros(nb, nb)));
    }
    if let Some(r) = rho {
        if r.len() != np {
            return Err(format!("rho has {} entries for {np} points", r.len()));
        }
    }
    let sp = split(geo)?;
    let (n_supp, n_g) = (sp.supps.len(), sp.globals.len());

    // Kernel×kernel support pairs: same support always overlaps, distinct ones
    // only when the support balls intersect (`disjoint` in gbasis).
    let mut kk: Vec<u32> = Vec::new();
    for s1 in 0..n_supp {
        for s2 in s1..n_supp {
            let (a, b) = (&sp.supps[s1], &sp.supps[s2]);
            let (dx, dy, dz) = (a[0] - b[0], a[1] - b[1], a[2] - b[2]);
            let rr = a[3] + b[3];
            if s1 == s2 || dx * dx + dy * dy + dz * dz < rr * rr {
                kk.push(s1 as u32);
                kk.push(s2 as u32);
            }
        }
    }
    // kernel×global: every (support, global position) pair.
    let mut gk: Vec<u32> = Vec::with_capacity(n_supp * n_g * 2);
    for s in 0..n_supp {
        for g in 0..n_g {
            gk.push(s as u32);
            gk.push(g as u32);
        }
    }
    // global×global: upper pairs of global positions.
    let mut gg: Vec<u32> = Vec::with_capacity(n_g * (n_g + 1));
    for b in 0..n_g {
        for c in 0..=b {
            gg.push(b as u32);
            gg.push(c as u32);
        }
    }

    let mut ptv = Vec::with_capacity(np * 3);
    let mut pw = Vec::with_capacity(np);
    for (p, x) in pts.iter().enumerate() {
        ptv.extend([x.x as f32, x.y as f32, x.z as f32]);
        pw.push((w[p] * rho.map_or(1.0, |r| r[p])) as f32);
    }
    let params = GgParams {
        n_points: np as u32,
        n_modes: nb as u32,
        n_supp: n_supp as u32,
        n_global: n_g as u32,
        n_kk: (kk.len() / 2) as u32,
        n_gk: (gk.len() / 2) as u32,
        n_gg: (gg.len() / 2) as u32,
        want_strain: u32::from(want_strain),
    };
    let gvals_len = np * n_g * 9;
    // Support lists in the worst case cover the whole quadrature.
    let supp_list_len = n_supp * np;

    let mut job = Job::new();
    let p = job.value(&params);
    let pts_b = job.input(&ptv);
    let pw_b = job.input(&pw);
    let modes_b = job.input(&pack_modes(geo));
    let supps_b = job.input(bytemuck::cast_slice::<[f32; 4], f32>(&sp.supps));
    let base_b = job.input(&sp.base);
    let off_b = job.zeros::<u32>(n_supp.max(1) + 1);
    let list_b = job.zeros::<u32>(supp_list_len.max(1));
    let kk_b = job.input(&kk);
    let gk_b = job.input(&gk);
    let gg_b = job.input(&gg);
    let globals_b = job.input(&sp.globals);
    let gvals_b = job.zeros::<f32>(gvals_len.max(1));
    let m_b = job.zeros::<f32>((nb * nb).max(1));
    let s_b = job.zeros::<f32>((nb * nb).max(1));
    let cnt_b = job.zeros::<u32>(n_supp.max(1));

    if n_supp > 0 {
        job.run(
            gpu::kernel("ggram_supp_count")?,
            groups_of(n_supp),
            &[p, pts_b, supps_b, cnt_b],
        );
        job.run(gpu::kernel("ggram_supp_prefix")?, [1, 1, 1], &[p, cnt_b, off_b]);
        job.run(
            gpu::kernel("ggram_supp_fill")?,
            groups_of(n_supp),
            &[p, pts_b, supps_b, off_b, list_b],
        );
    }
    if n_g > 0 {
        job.run(
            gpu::kernel("ggram_eval_g")?,
            groups_of(np),
            &[p, pts_b, modes_b, globals_b, gvals_b],
        );
    }
    if params.n_kk > 0 {
        job.run(
            gpu::kernel("ggram_kk")?,
            [params.n_kk, 1, 1],
            &[p, pts_b, pw_b, supps_b, base_b, off_b, list_b, kk_b, m_b, s_b],
        );
    }
    if params.n_gk > 0 {
        job.run(
            gpu::kernel("ggram_gk")?,
            groups_of(params.n_gk as usize),
            &[p, pts_b, pw_b, supps_b, base_b, off_b, list_b, gk_b, globals_b, gvals_b, m_b, s_b],
        );
    }
    if params.n_gg > 0 {
        job.run(
            gpu::kernel("ggram_gg")?,
            groups_of(params.n_gg as usize),
            &[p, pw_b, gg_b, gvals_b, globals_b, m_b, s_b],
        );
    }
    let rm = job.read_all::<f32>(m_b, nb * nb);
    let rs = job.read_all::<f32>(s_b, nb * nb);
    let back = job.submit()?;
    let mm = back.get::<f32>(rm);
    let ss = back.get::<f32>(rs);
    let mass = DMatrix::from_fn(nb, nb, |r, c| mm[r * nb + c] as f64);
    let strain = DMatrix::from_fn(nb, nb, |r, c| ss[r * nb + c] as f64);
    Ok((mass, strain))
}
