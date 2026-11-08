//! Host side of the axis sweep operator `relax_sweep` of
//! `scf/shader/relax.wesl` — the GPU half of `hyperbolic::sweep_axis`.
//!
//! The pattern is the fused-operator one of `ggram`: the geometry (points,
//! neighbour CSR, LSQ inverses, dense cell index, member CSR) lives in `Res`
//! buffers as an implementation detail of the operator, and each relax sweep
//! is one `Job` — upload of the f64 state as f32, the three axis dispatches
//! ping-ponging between `u_a`/`u_b`, and the read-back at the boundary. All
//! convergence control (traction fix, AC correction, rigid projection,
//! norms, Δτ rollbacks) stays on the host in f64.

use crate::common::error::SimError;
use crate::common::math::DVec3;
use crate::gpu::{self, Job, Res};
use crate::scf::cell_bvh::CellBvh;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SwParams {
    n: u32,
    axis: u32,
    nx: u32,
    ny: u32,
    nz: u32,
    gx: i32,
    gy: i32,
    gz: i32,
    dt: f32,
    t_s: f32,
    kappa: f32,
    side: f32,
    ox: f32,
    oy: f32,
    oz: f32,
    pad: f32,
}

const _: () = assert!(std::mem::size_of::<SwParams>() == 64);

const NDIM: usize = 10;

fn groups_of(n: usize) -> [u32; 3] {
    [n.div_ceil(64).max(1) as u32, 1, 1]
}

/// The device-resident geometry of one `Ray` — everything `sweep_axis`
/// reads except the state and the per-pass `η`, `T_tide`, `g_eff` fields.
/// Device arrays are stored in **cell order** (samples sorted by their
/// `CellBvh` cell): the 27-cell neighbourhood of a backtrace point then
/// covers a handful of contiguous spans of `u_in`, and the gather passes of
/// `interp`/`strain` coalesce instead of hitting ~80 scattered addresses
/// per call. `perm[old] = new` maps host sample indices into device order.
pub struct SweepGpu {
    perm: Vec<u32>,
    n: usize,
    dims: [u32; 3],
    gmin: [i32; 3],
    origin: [f32; 3],
    side: f32,
    pts: Res,
    nbr_off: Res,
    nbr_idx: Res,
    g_inv: Res,
    eta: Res,
    rho: Res,
    tide: Res,
    g_eff: Res,
    grid: Res,
    cell_off: Res,
    cell_mem: Res,
    u_a: Res,
    u_b: Res,
    /// Which of `u_a`/`u_b` held the input of the last dispatch — the result
    /// sits in the other one.
    cur_is_a: bool,
}

impl SweepGpu {
    /// Upload the fixed geometry of a `Ray`. `None` when no GPU is present —
    /// the caller then keeps the CPU path.
    pub fn build(
        pts: &[DVec3],
        nbr: &[Vec<u32>],
        g_inv: &[f64],
        bvh: &CellBvh,
        rho: &[f64],
    ) -> Option<Self> {
        let n = pts.len();
        if n == 0 {
            return None;
        }
        if gpu::gpu().is_err() {
            return None;
        }
        let ok = |r: Result<Res, String>| r.ok();
        // Cell-ordered permutation of the samples: `order[new] = old`,
        // `perm[old] = new`.
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by_key(|&i| bvh.cell_of_node(i) as usize);
        let mut perm = vec![0u32; n];
        for (new, &old) in order.iter().enumerate() {
            perm[old] = new as u32;
        }
        // Neighbour stencil to CSR, in device order with device indices.
        let mut nbr_off = Vec::with_capacity(n + 1);
        let mut nbr_idx = Vec::new();
        nbr_off.push(0u32);
        for &old in &order {
            nbr_idx.extend(nbr[old].iter().map(|&j| perm[j as usize]));
            nbr_off.push(nbr_idx.len() as u32);
        }
        // The dense slot→cell index volume of `CellBvh::index`.
        let gmin = [
            bvh.coords.iter().map(|c| c[0]).min()? as i32,
            bvh.coords.iter().map(|c| c[1]).min()? as i32,
            bvh.coords.iter().map(|c| c[2]).min()? as i32,
        ];
        let gmax = [
            bvh.coords.iter().map(|c| c[0]).max()? as i32,
            bvh.coords.iter().map(|c| c[1]).max()? as i32,
            bvh.coords.iter().map(|c| c[2]).max()? as i32,
        ];
        let dims = [
            (gmax[0] - gmin[0] + 1) as u32,
            (gmax[1] - gmin[1] + 1) as u32,
            (gmax[2] - gmin[2] + 1) as u32,
        ];
        let mut grid = vec![0u32; (dims[0] * dims[1] * dims[2]) as usize];
        for (c, k) in bvh.coords.iter().enumerate() {
            let l = [
                (k[0] as i32 - gmin[0]) as u32,
                (k[1] as i32 - gmin[1]) as u32,
                (k[2] as i32 - gmin[2]) as u32,
            ];
            grid[((l[0] * dims[1] + l[1]) * dims[2] + l[2]) as usize] = c as u32 + 1;
        }
        // Cell members in device order — a cell's members are exactly the
        // contiguous block the cell order assigns it.
        let mut cell_off = Vec::with_capacity(bvh.n_cells() + 1);
        let mut cell_mem = Vec::new();
        cell_off.push(0u32);
        for c in 0..bvh.n_cells() {
            let mut ms: Vec<u32> = bvh.members(c).iter().map(|&j| perm[j as usize]).collect();
            ms.sort_unstable();
            cell_mem.extend_from_slice(&ms);
            cell_off.push(cell_mem.len() as u32);
        }
        // Per-sample blocks of `w` f64s → device order as f32.
        let perm_f32_n = |v: &[f64], w: usize| -> Vec<f32> {
            let mut out = vec![0f32; v.len()];
            for (new, &old) in order.iter().enumerate() {
                for k in 0..w {
                    out[w * new + k] = v[w * old + k] as f32;
                }
            }
            out
        };
        let mut pf = vec![0f32; 3 * n];
        for (new, &old) in order.iter().enumerate() {
            let p = pts[old];
            pf[3 * new] = p.x as f32;
            pf[3 * new + 1] = p.y as f32;
            pf[3 * new + 2] = p.z as f32;
        }
        Some(Self {
            perm,
            n,
            dims,
            gmin,
            origin: [bvh.origin.x as f32, bvh.origin.y as f32, bvh.origin.z as f32],
            side: bvh.side as f32,
            pts: ok(Res::from_slice(&pf))?,
            nbr_off: ok(Res::from_slice(&nbr_off))?,
            nbr_idx: ok(Res::from_slice(&nbr_idx))?,
            g_inv: ok(Res::from_slice(&perm_f32_n(g_inv, 9)))?,
            eta: ok(Res::zeroed((4 * n) as u64))?,
            rho: ok(Res::from_slice(&perm_f32_n(rho, 1)))?,
            tide: ok(Res::zeroed((4 * 6 * n) as u64))?,
            g_eff: ok(Res::zeroed((4 * 3 * n) as u64))?,
            grid: ok(Res::from_slice(&grid))?,
            cell_off: ok(Res::from_slice(&cell_off))?,
            cell_mem: ok(Res::from_slice(&cell_mem))?,
            u_a: ok(Res::zeroed((4 * NDIM * n) as u64))?,
            u_b: ok(Res::zeroed((4 * NDIM * n) as u64))?,
            cur_is_a: true,
        })
    }

    /// One axis sweep: `u_in → u_out` for the frozen fields already resident.
    fn axis(&mut self, job: &mut Job, axis: u32, dt: f64, t_s: f64, kappa: f64) {
        let kernel = match gpu::kernel("relax_sweep") {
            Ok(k) => k,
            Err(_) => return,
        };
        let p = SwParams {
            n: self.n as u32,
            axis,
            nx: self.dims[0],
            ny: self.dims[1],
            nz: self.dims[2],
            gx: self.gmin[0],
            gy: self.gmin[1],
            gz: self.gmin[2],
            dt: dt as f32,
            t_s: t_s as f32,
            kappa: kappa as f32,
            side: self.side,
            ox: self.origin[0],
            oy: self.origin[1],
            oz: self.origin[2],
            pad: 0.0,
        };
        let pb = job.value(&p);
        let (src, dst) = if self.cur_is_a {
            (&self.u_a, &self.u_b)
        } else {
            (&self.u_b, &self.u_a)
        };
        job.run(
            kernel,
            groups_of(self.n),
            &[
                pb,
                (&self.pts).into(),
                src.into(),
                dst.into(),
                (&self.nbr_off).into(),
                (&self.nbr_idx).into(),
                (&self.g_inv).into(),
                (&self.eta).into(),
                (&self.rho).into(),
                (&self.tide).into(),
                (&self.g_eff).into(),
                (&self.grid).into(),
                (&self.cell_off).into(),
                (&self.cell_mem).into(),
            ],
        );
        self.cur_is_a = !self.cur_is_a;
    }

    /// The per-relax fields `η`, `T_tide`, `g_eff` — one small submission at
    /// the top of `relax`, not per sweep.
    pub fn set_fields(
        &mut self,
        eta: &[f64],
        tide: &[[f64; 6]],
        g_eff: &[DVec3],
    ) -> Result<(), SimError> {
        let mut job = Job::new();
        // Field arrays upload in device (cell) order.
        let mut ef = vec![0f32; self.n];
        let mut tf = vec![0f32; 6 * self.n];
        let mut gf = vec![0f32; 3 * self.n];
        for (i, &p) in self.perm.iter().enumerate() {
            let d = p as usize;
            ef[d] = eta[i] as f32;
            for k in 0..6 {
                tf[6 * d + k] = tide[i][k] as f32;
            }
            gf[3 * d] = g_eff[i].x as f32;
            gf[3 * d + 1] = g_eff[i].y as f32;
            gf[3 * d + 2] = g_eff[i].z as f32;
        }
        job.write(&self.eta, 0, &ef);
        job.write(&self.tide, 0, &tf);
        job.write(&self.g_eff, 0, &gf);
        job.submit().map_err(SimError::gpu)?;
        Ok(())
    }

    /// One full relax sweep on device: upload `u` (f64 → f32), run the three
    /// axis families ping-ponging `u_a`/`u_b`, read the result back.
    pub fn sweep(
        &mut self,
        u: &[f64],
        dt: f64,
        t_s: f64,
        kappa: f64,
    ) -> Result<Vec<f64>, SimError> {
        let mut job = Job::new();
        // State upload in device order.
        let mut uf = vec![0f32; NDIM * self.n];
        for (i, &p) in self.perm.iter().enumerate() {
            let d = NDIM * p as usize;
            for k in 0..NDIM {
                uf[d + k] = u[NDIM * i + k] as f32;
            }
        }
        job.write(&self.u_a, 0, &uf);
        self.cur_is_a = true;
        for axis in 0..3u32 {
            self.axis(&mut job, axis, dt, t_s, kappa);
        }
        let out = if self.cur_is_a { &self.u_a } else { &self.u_b };
        let rd = job.read_all::<f32>(out, NDIM * self.n);
        let outs = job.submit().map_err(SimError::gpu)?;
        let raw: Vec<f32> = outs.get(rd);
        // …and back to host sample order.
        let mut u2 = vec![0f64; NDIM * self.n];
        for (i, &p) in self.perm.iter().enumerate() {
            let s = NDIM * p as usize;
            for k in 0..NDIM {
                u2[NDIM * i + k] = raw[s + k] as f64;
            }
        }
        Ok(u2)
    }
}
