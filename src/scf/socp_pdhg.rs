//! Metal PDHG for the cone program assembled in `socp.rs`.
//!
//! The program is a diagonal convex quadratic over a product of an equality
//! (zero) cone, a nonnegative cone, and four-dimensional second-order cones.
//! A trailing second-order cone is accepted for tests; the geometric trust
//! region is a box in the nonnegative block. The objective is divided by the
//! slope penalty, and each displacement column is scaled so its curvature is
//! one. A mass slack whose kilogram residual cannot be met inside the box is
//! removed by `socp.rs` and replaced by the exact linear penalty; otherwise
//! its column is scaled so the equality entry and the travel in the scaled
//! variable are comparable. Ruiz balances the cones and the box and does not
//! let the mass row spend the curvature. A primal step whose linear term
//! exceeds 10 includes that term in the denominator, so the first prox step
//! stays O(1). The iteration is Chambolle–Pock with diagonal step lengths, one
//! length per cone, on the GPU in double-single arithmetic. A point is
//! accepted when the original KKT residuals meet `KKT_TOL`. Stationarity is
//! the worst relative residual of one variable.

use crate::common::time::wall_seconds;

mod cpu;

/// Row residual, block stationarity, duality gap, and dual-cone feasibility.
pub const KKT_TOL: f64 = 1.0e-3;
/// Residual checks stay on this cadence. A longer batch changes when the
/// step-size shrink fires, and that schedule is what closes the dual residual.
const CHECK_ITERS: u32 = 48;
const MAX_ITERS: u32 = 40_000;
/// The stationarity phase stops at `MAX_ITERS`. Only a gap that is still
/// falling may continue, and only up to twice that budget.
const GAP_ITERS_CAP: u32 = 80_000;
/// Twenty residual checks. The gap extension ends when a window this long
/// does not cut the gap by five percent.
const GAP_WINDOW: u32 = 960;
/// Batches one submission may run past the batch whose residuals are checked. A
/// browser round trip costs milliseconds, a native one a fraction of that.
#[cfg(target_arch = "wasm32")]
const MAX_AHEAD: usize = 16;
#[cfg(not(target_arch = "wasm32"))]
const MAX_AHEAD: usize = 8;

/// Extra iterations `k (gap/tol − 1) · 2`, capped at `GAP_ITERS_CAP`.
fn gap_extension(iter: u32, gap: f64) -> u32 {
    let ratio = (gap / KKT_TOL - 1.0).max(0.0);
    let extra = iter as f64 * ratio * 2.0;
    let extra = if extra.is_finite() {
        extra.ceil() as u32
    } else {
        0
    };
    iter.saturating_add(extra.max(CHECK_ITERS))
        .min(GAP_ITERS_CAP)
}
const MAX_STEP_GROWS: u32 = 8;
/// Entries per workgroup of a long row (`pdhg_ax_long`).
const LONG_CHUNK: u32 = 4096;

pub struct Problem {
    pub p_diag: Vec<f64>,
    pub q: Vec<f64>,
    pub rows: Vec<usize>,
    pub cols: Vec<usize>,
    pub vals: Vec<f64>,
    pub b: Vec<f64>,
    pub n_eq: usize,
    pub n_nn: usize,
    pub n_soc: usize,
    /// Length of the single trailing second-order cone. Zero when it is absent.
    pub n_trust: usize,
    /// Original-space saddle point. `None` starts at the origin.
    pub warm: Option<WarmStart>,
}

/// Primal `x` and dual `y` before column/row equilibration.
pub struct WarmStart {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
}

pub struct Solution {
    pub x: Vec<f64>,
    /// Original-space multipliers of the accepted point.
    pub y: Vec<f64>,
    pub iterations: u32,
    pub seconds: f64,
    /// Row, dual, and gap residuals of the accepted point. Tests read them;
    /// the solver already refuses a point that misses `KKT_TOL`.
    pub primal: f64,
    pub dual: f64,
    pub gap: f64,
    /// Acceptance metric of the uploaded start: max of row, dual, gap, dual-cone.
    pub initial_metric: f64,
}

#[derive(Debug)]
pub enum PdhgError {
    Gpu(String),
    Failed(String),
}

#[derive(Clone)]
struct Csr {
    m: usize,
    n: usize,
    row_ptr: Vec<u32>,
    col: Vec<u32>,
    val: Vec<f64>,
}

struct Scale {
    /// `x = C z` after column equilibration.
    c: Vec<f64>,
    /// `y = E y' / α` after row equilibration and the objective scale `α`.
    e: Vec<f64>,
    alpha: f64,
}

struct Residuals {
    rp: f64,
    rd: f64,
    gap: f64,
    dy: f64,
    rp_inf: f64,
    /// `x · (P x + q + Aᵀ y)`, the stationarity pairing inside the gap.
    stationarity: f64,
    /// `y · (b − A x)`, the complementarity pairing inside the gap.
    complementarity: f64,
}

impl Residuals {
    fn metric(&self) -> f64 {
        // The mass row is one entry of the primal residual. The 2-norm hides it
        // once `b` is long, so the acceptance norm is the largest row violation.
        self.rp_inf.max(self.rd).max(self.gap).max(self.dy)
    }

    /// Sum of the four residuals. The max alone will not leave a point that is
    /// stationary and complementary but still violates a cone: every step that
    /// reduces the row also spends some stationarity, and the max goes up.
    fn progress(&self) -> f64 {
        self.rp_inf + self.rd + self.gap + self.dy
    }

    fn accepted(&self) -> bool {
        self.rp_inf <= KKT_TOL && self.rd <= KKT_TOL && self.gap <= KKT_TOL && self.dy <= KKT_TOL
    }
}

struct Candidate {
    z: Vec<f64>,
    y: Vec<f64>,
    x: Vec<f64>,
    y_orig: Vec<f64>,
    residuals: Residuals,
}

/// The CPU in f64 by default: faster than the GPU kernels on every cone program of
/// the bundled bodies (a few hundred to several thousand rows), more precise than
/// their double-single, and free of GPU round trips (milliseconds each in the browser).
fn on_cpu() -> bool {
    std::env::var("AE_PDHG").as_deref() != Ok("gpu")
}

/// Where the batches run: the GPU kernels in double-single, or the same iteration on
/// the CPU in f64.
enum Engine {
    Gpu(gpu::Device),
    Cpu(cpu::Device),
}

/// The iterates after a batch, as the engine holds them.
#[derive(Clone)]
enum Snapshot {
    Gpu(gpu::Words),
    Cpu(cpu::State),
}

impl Snapshot {
    /// `(x, y, x_avg, y_avg)`.
    fn iterates(&self) -> [Vec<f64>; 4] {
        match self {
            Self::Gpu(words) => words.each_ref().map(|w| gpu::undo(w)),
            Self::Cpu(state) => state.clone(),
        }
    }
}

impl Engine {
    #[allow(clippy::too_many_arguments)]
    fn new(
        a: &Csr,
        p: &[f64],
        q: &[f64],
        b: &[f64],
        tau: &[f64],
        sigma: &[f64],
        n_eq: usize,
        n_nn: usize,
        n_soc: usize,
        n_trust: usize,
    ) -> Result<Self, PdhgError> {
        Ok(if on_cpu() {
            Self::Cpu(cpu::Device::new(a, p, q, b, tau, sigma, n_eq, n_nn, n_soc, n_trust))
        } else {
            Self::Gpu(gpu::Device::new(a, p, q, b, tau, sigma, n_eq, n_nn, n_soc, n_trust)?)
        })
    }

    fn run(&mut self, plan: &[(u32, u32)], scale: f64) -> Result<Vec<Snapshot>, PdhgError> {
        Ok(match self {
            Self::Gpu(d) => d.run(plan, scale)?.into_iter().map(Snapshot::Gpu).collect(),
            Self::Cpu(c) => c.run(plan, scale).into_iter().map(Snapshot::Cpu).collect(),
        })
    }

    fn restore(&mut self, snap: &Snapshot) -> Result<(), PdhgError> {
        match (self, snap) {
            (Self::Gpu(d), Snapshot::Gpu(words)) => d.restore(words),
            (Self::Cpu(c), Snapshot::Cpu(state)) => {
                c.restore(state);
                Ok(())
            }
            _ => Err(PdhgError::Gpu("PDHG snapshot from another engine".into())),
        }
    }

    fn write_state(&mut self, x: &[f64], y: &[f64], xavg: &[f64], yavg: &[f64]) -> Result<(), PdhgError> {
        match self {
            Self::Gpu(d) => d.write_state(x, y, xavg, yavg),
            Self::Cpu(c) => {
                c.write_state(x, y, xavg, yavg);
                Ok(())
            }
        }
    }

    /// Batches worth running ahead of the residual check: a GPU submission costs a
    /// round trip, a CPU batch only its iterations.
    fn ahead_limit(&self) -> usize {
        match self {
            Self::Gpu(_) => MAX_AHEAD,
            Self::Cpu(_) => 1,
        }
    }
}

/// A batch the device ran before the loop asked for it, and the parameters it ran with.
struct Planned {
    iters: u32,
    avg_count: u32,
    scale: f64,
    snap: Snapshot,
}

/// Batches run ahead of the residual check. The loop takes them in order while each
/// is the batch it would have asked for (same length, average count and step scale,
/// no rewritten state); otherwise the rest are dropped, the device goes back to the
/// last batch taken, and the run continues from there. The iterates are therefore
/// those of one batch per submission.
#[derive(Default)]
struct Ahead {
    queue: std::collections::VecDeque<Planned>,
    /// The batch taken last, while the device has run further.
    taken: Option<Snapshot>,
    len: usize,
}

impl Ahead {
    fn next(
        &mut self,
        device: &mut Engine,
        iters: u32,
        avg_count: u32,
        scale: f64,
        iter: u32,
        limit: u32,
    ) -> Result<Snapshot, PdhgError> {
        let hit = self.queue.front().is_some_and(|p| {
            p.iters == iters && p.avg_count == avg_count && p.scale.to_bits() == scale.to_bits()
        });
        if !hit {
            if self.queue.is_empty() {
                self.len = (self.len * 2).clamp(1, device.ahead_limit());
            } else {
                if let Some(snap) = &self.taken {
                    device.restore(snap)?;
                }
                self.queue.clear();
                self.len = 1;
            }
            let mut plan = Vec::with_capacity(self.len);
            let (mut at, mut count) = (iter, avg_count);
            while plan.len() < self.len && at < limit {
                let b = CHECK_ITERS.min(limit - at);
                plan.push((b, count));
                at += b;
                count += b;
            }
            for (&(iters, avg_count), snap) in plan.iter().zip(device.run(&plan, scale)?) {
                self.queue.push_back(Planned {
                    iters,
                    avg_count,
                    scale,
                    snap,
                });
            }
        }
        let planned = self
            .queue
            .pop_front()
            .ok_or_else(|| PdhgError::Gpu("PDHG planned no batch".into()))?;
        self.taken = (!self.queue.is_empty()).then(|| planned.snap.clone());
        Ok(planned.snap)
    }

    /// The loop wrote the device state: nothing run ahead applies any more.
    fn rewritten(&mut self) {
        self.queue.clear();
        self.taken = None;
    }
}

pub fn solve(problem: Problem) -> Result<Solution, PdhgError> {
    let started = wall_seconds();
    let n = problem.p_diag.len();
    let m = problem.b.len();
    if problem.q.len() != n {
        return Err(PdhgError::Failed(
            "PDHG quadratic linear term has the wrong length".into(),
        ));
    }
    if m != problem.n_eq + problem.n_nn + 4 * problem.n_soc + problem.n_trust || n == 0 || m == 0 {
        return Err(PdhgError::Failed(format!(
            "PDHG cone layout is inconsistent: n={n}, m={m}, eq={}, nn={}, soc={}, trust={}",
            problem.n_eq, problem.n_nn, problem.n_soc, problem.n_trust
        )));
    }
    let original = Csr::from_triplets(m, n, &problem.rows, &problem.cols, &problem.vals)?;
    let p0 = problem.p_diag;
    let q0 = problem.q;
    let b0 = problem.b;
    let mut scaled = original.clone();
    let mut p = p0.clone();
    let mut q = q0.clone();
    let mut b = b0.clone();
    let scale = equilibrate(
        &mut scaled,
        &mut p,
        &mut q,
        &mut b,
        problem.n_eq,
        problem.n_nn,
        problem.n_soc,
        problem.n_trust,
    )?;
    // A single Lipschitz step τ = σ = 0.9/‖A‖ gives every row the slowest
    // row's step. Diagonal Pock–Chambolle steps keep the same cone projection
    // and the same minimizer; a row or column with a small sum takes a longer
    // step. The global factor starts at 1 and shrinks only if a step overflows
    // double-single.
    let (tau_diag, sigma_diag) = diagonal_steps(
        &scaled,
        &q,
        problem.n_eq,
        problem.n_nn,
        problem.n_soc,
        problem.n_trust,
    );
    let mut step_scale = 1.0_f64;

    let mut device = Engine::new(
        &scaled,
        &p,
        &q,
        &b,
        &tau_diag,
        &sigma_diag,
        problem.n_eq,
        problem.n_nn,
        problem.n_soc,
        problem.n_trust,
    )?;
    let (z0, y_scaled, mut avg_count) = scaled_start(&problem.warm, &scale, n, m);
    let (x0, y0) = recover(&z0, &y_scaled, &scale);
    let r0 = residuals(
        &original,
        &p0,
        &q0,
        &b0,
        &x0,
        &y0,
        problem.n_eq,
        problem.n_nn,
        problem.n_soc,
        problem.n_trust,
    );
    if r0.accepted() {
        return Ok(Solution {
            x: x0,
            y: y0,
            iterations: 0,
            seconds: wall_seconds() - started,
            primal: r0.rp,
            dual: r0.rd,
            gap: r0.gap,
            initial_metric: r0.metric(),
        });
    }
    // A warm saddle is already on the device only after this upload. The
    // average counts that point as its first sample, and x̄ starts at z so the
    // first dual step multiplies A z rather than A·0.
    if avg_count > 0 {
        device.write_state(&z0, &y_scaled, &z0, &y_scaled)?;
    }
    let initial_metric = r0.metric();

    let mut best = Candidate {
        z: z0,
        y: y_scaled,
        x: x0,
        y_orig: y0,
        residuals: r0,
    };
    let mut iter = 0u32;
    let mut grows = 0u32;
    let mut stall = 0u32;
    let mut backoff = 0u32;
    let mut refines = 0u32;
    let mut held_step = false;
    let mut limit = MAX_ITERS;
    let mut ahead = Ahead::default();
    let mut gap_phase = false;
    let mut gap_anchor = f64::INFINITY;
    let mut gap_anchor_iter = 0u32;
    let log = std::env::var_os("PDHG_LOG").is_some();
    if log {
        let l_f = scaled.val.iter().map(|v| v * v).sum::<f64>().sqrt();
        let l_pow = op_norm(&scaled);
        let tau_max = tau_diag.iter().copied().fold(0.0_f64, f64::max);
        let tau_min = tau_diag.iter().copied().fold(f64::MAX, f64::min);
        eprintln!(
            "pdhg n={} m={} nnz={} L={l_pow:.3e} (frobenius {l_f:.3e}) diagonal τ=[{tau_min:.3e}, {tau_max:.3e}]",
            scaled.n,
            scaled.m,
            scaled.val.len()
        );
    }

    while iter < limit {
        let batch = CHECK_ITERS.min(limit - iter);
        let words = {
            let _span = super::timing::span("pdhg_gpu");
            ahead.next(&mut device, batch, avg_count, step_scale, iter, limit)?
        };
        let [z, y, za, ya] = words.iterates();
        iter += batch;
        avg_count += batch;
        if !(finite(&z) && finite(&y) && finite(&za) && finite(&ya)) {
            grows += 1;
            if grows > MAX_STEP_GROWS {
                break;
            }
            step_scale *= 0.5;
            device.write_state(&best.z, &best.y, &best.z, &best.y)?;
            ahead.rewritten();
            avg_count = 1;
            stall = 0;
            if log {
                eprintln!("pdhg non-finite step; scale={step_scale:.3e}");
            }
            continue;
        }
        let before = best.residuals.progress();
        let _residual = super::timing::span("pdhg_residual");
        consider(
            &mut best,
            &z,
            &y,
            &original,
            &p0,
            &q0,
            &b0,
            &scale,
            problem.n_eq,
            problem.n_nn,
            problem.n_soc,
            problem.n_trust,
        );
        consider(
            &mut best,
            &za,
            &ya,
            &original,
            &p0,
            &q0,
            &b0,
            &scale,
            problem.n_eq,
            problem.n_nn,
            problem.n_soc,
            problem.n_trust,
        );
        // A stall stays on the last-iterate trajectory and keeps the ergodic
        // average. Clearing it would throw away the only sequence whose gap is
        // guaranteed to fall. A step-size restart below is what rewrites the
        // point and starts the average over.
        if best.residuals.progress() < before * 0.95 {
            // Early stalls shrink the step while the best point is still the
            // origin. A later feasible point then cannot polish a 1e-2 dual
            // residual. Give that point the original step, twice at most.
            if before > 0.0
                && best.residuals.progress() < before * 0.1
                && step_scale < 1.0
                && refines < 2
                && !gap_phase
            {
                refines += 1;
                step_scale = 1.0;
                backoff = 0;
                device.write_state(&best.z, &best.y, &best.z, &best.y)?;
                ahead.rewritten();
                avg_count = 1;
                if log {
                    eprintln!("pdhg metric dropped; restore step");
                }
            }
            stall = 0;
        } else {
            stall = stall.saturating_add(1);
        }
        drop(_residual);
        if log {
            let r = &best.residuals;
            let last = residuals(
                &original,
                &p0,
                &q0,
                &b0,
                &recover(&z, &y, &scale).0,
                &recover(&z, &y, &scale).1,
                problem.n_eq,
                problem.n_nn,
                problem.n_soc,
                problem.n_trust,
            );
            eprintln!(
                "pdhg iter {iter} best rp={:.3e} rd={:.3e} gap={:.3e} sum={:.3e} | last rp={:.3e} rd={:.3e} gap={:.3e} sum={:.3e}",
                r.rp,
                r.rd,
                r.gap,
                r.progress(),
                last.rp_inf,
                last.rd,
                last.gap,
                last.progress()
            );
        }
        if best.residuals.accepted() {
            break;
        }
        // The row, the block dual and the dual cone are the local KKT
        // conditions. Once they pass, the normalised gap is an ergodic
        // certificate and a halved step or a cleared average only delays it.
        let locals_ok = best.residuals.rp_inf <= KKT_TOL
            && best.residuals.rd <= KKT_TOL
            && best.residuals.dy <= KKT_TOL;
        if locals_ok && best.residuals.gap > KKT_TOL {
            let gap = best.residuals.gap;
            if !gap_phase {
                gap_phase = true;
                gap_anchor = gap;
                gap_anchor_iter = iter;
                backoff = 3;
                held_step = true;
                if step_scale < 1.0 {
                    step_scale = 1.0;
                }
                limit = gap_extension(iter, gap);
            } else if gap < gap_anchor * 0.95 {
                gap_anchor = gap;
                gap_anchor_iter = iter;
                limit = gap_extension(iter, gap);
            } else if iter.saturating_sub(gap_anchor_iter) >= GAP_WINDOW {
                break;
            }
        }
        // A stall only shrinks the step. Jumping back to the best point throws
        // away a last iterate that is still closing the cones, and clearing the
        // average throws away the gap certificate. The average restarts only
        // when the point itself is rewritten (a non-finite step).
        // Near the tolerance a halved step only stretches the same contraction.
        // Freeze the full step once the f64 metric is inside fifty tolerances.
        if !held_step && best.residuals.metric() <= 50.0 * KKT_TOL {
            held_step = true;
            backoff = 3;
            if step_scale < 1.0 {
                step_scale = 1.0;
                stall = 0;
                device.write_state(&best.z, &best.y, &best.z, &best.y)?;
                ahead.rewritten();
                avg_count = 1;
            }
        }
        if stall >= 8 && backoff < 3 {
            backoff += 1;
            step_scale *= 0.5;
            stall = 0;
            if log {
                eprintln!("pdhg residual stall; scale={step_scale:.3e}");
            }
        }
    }

    // Ruiz can crush the displacement curvature while it restores the mass
    // row. Reintroducing that curvature by a global factor on P and q does not
    // change the minimizer. f64 then polishes the stationarity the GPU stopped
    // feeling.
    let locals_ok = best.residuals.rp_inf <= KKT_TOL
        && best.residuals.rd <= KKT_TOL
        && best.residuals.dy <= KKT_TOL;
    // Curvature polish repairs a block dual residual. It does not close a
    // complementarity gap, and it would spend the extension budget twice.
    if !best.residuals.accepted() && !locals_ok {
        let mut z = best.z.clone();
        let mut y = best.y.clone();
        {
            let _span = super::timing::span("pdhg_polish");
            curvature_polish(
                &scaled,
                &p,
                &q,
                &b,
                &tau_diag,
                &sigma_diag,
                &mut z,
                &mut y,
                problem.n_eq,
                problem.n_nn,
                problem.n_soc,
                problem.n_trust,
                4_096,
            );
        }
        consider(
            &mut best,
            &z,
            &y,
            &original,
            &p0,
            &q0,
            &b0,
            &scale,
            problem.n_eq,
            problem.n_nn,
            problem.n_soc,
            problem.n_trust,
        );
        if std::env::var_os("PDHG_LOG").is_some() {
            let r = &best.residuals;
            eprintln!(
                "pdhg final lift rp={:.3e} rd={:.3e} gap={:.3e} accepted={}",
                r.rp_inf,
                r.rd,
                r.gap,
                r.accepted()
            );
        }
    }

    // The GPU certifies the scaled gradient. A vertex with curvature ~1e5
    // turns a scaled residual of 1e-5 into an original force of a few tenths,
    // which misses the per-variable test even though the implied radius change
    // is under a millimetre. One equality-aware Newton step removes that force
    // without moving the mass row.
    if !best.residuals.accepted() {
        if let Some((x2, y2)) = displacement_newton(
            &original,
            &p0,
            &q0,
            &b0,
            &best.x,
            &best.y_orig,
            problem.n_eq,
            problem.n_nn,
            problem.n_soc,
            problem.n_trust,
        ) {
            let z2: Vec<f64> = x2.iter().zip(&scale.c).map(|(x, c)| x / c).collect();
            let y2s: Vec<f64> = y2
                .iter()
                .zip(&scale.e)
                .map(|(y, e)| y * scale.alpha / e)
                .collect();
            consider(
                &mut best,
                &z2,
                &y2s,
                &original,
                &p0,
                &q0,
                &b0,
                &scale,
                problem.n_eq,
                problem.n_nn,
                problem.n_soc,
                problem.n_trust,
            );
        }
    }
    if !best.residuals.accepted() {
        if let Some((x2, y2)) = balance_active_multipliers(
            &original,
            &p0,
            &q0,
            &b0,
            &best.x,
            &best.y_orig,
            problem.n_eq,
            problem.n_nn,
            problem.n_soc,
            problem.n_trust,
        ) {
            let z2: Vec<f64> = x2.iter().zip(&scale.c).map(|(x, c)| x / c).collect();
            let y2s: Vec<f64> = y2
                .iter()
                .zip(&scale.e)
                .map(|(y, e)| y * scale.alpha / e)
                .collect();
            consider(
                &mut best,
                &z2,
                &y2s,
                &original,
                &p0,
                &q0,
                &b0,
                &scale,
                problem.n_eq,
                problem.n_nn,
                problem.n_soc,
                problem.n_trust,
            );
        }
    }
    if !best.residuals.accepted() {
        if let Some((x2, y2)) = close_box_gap(
            &original,
            &p0,
            &q0,
            &b0,
            &best.x,
            &best.y_orig,
            problem.n_eq,
            problem.n_nn,
            problem.n_soc,
            problem.n_trust,
        ) {
            let z2: Vec<f64> = x2.iter().zip(&scale.c).map(|(x, c)| x / c).collect();
            let y2s: Vec<f64> = y2
                .iter()
                .zip(&scale.e)
                .map(|(y, e)| y * scale.alpha / e)
                .collect();
            consider(
                &mut best,
                &z2,
                &y2s,
                &original,
                &p0,
                &q0,
                &b0,
                &scale,
                problem.n_eq,
                problem.n_nn,
                problem.n_soc,
                problem.n_trust,
            );
        }
    }

    let r = &best.residuals;
    if !r.accepted() {
        if std::env::var_os("PDHG_LOG").is_some() {
            let mut aty = vec![0.0; original.n];
            spmv_t(&original, &best.y_orig, &mut aty);
            let mut worst: Vec<(f64, usize)> = (0..original.n)
                .map(|i| {
                    let g = p0[i] * best.x[i] + q0[i] + aty[i];
                    let scale = if p0[i] > 0.0 {
                        1.0 + q0[i].abs() + (p0[i] * best.x[i]).abs() + p0[i] * 1.0e-2
                    } else {
                        1.0 + q0[i].abs()
                    };
                    (g.abs() / scale, i)
                })
                .collect();
            worst.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            for (rd, i) in worst.into_iter().take(4) {
                let g = p0[i] * best.x[i] + q0[i] + aty[i];
                eprintln!(
                    "  rd={rd:.3e} i={i} g={g:.3e} x={:.3e} P={:.3e} q={:.3e}",
                    best.x[i], p0[i], q0[i]
                );
            }
        }
        let mut aty = vec![0.0; original.n];
        spmv_t(&original, &best.y_orig, &mut aty);
        let mut worst_i = 0usize;
        let mut worst_rd = 0.0_f64;
        for i in 0..original.n {
            let g = p0[i] * best.x[i] + q0[i] + aty[i];
            let scale = if p0[i] > 0.0 {
                1.0 + q0[i].abs() + (p0[i] * best.x[i]).abs() + p0[i] * 1.0e-2
            } else {
                1.0 + q0[i].abs()
            };
            let rd_i = g.abs() / scale;
            if rd_i > worst_rd {
                worst_rd = rd_i;
                worst_i = i;
            }
        }
        let g = p0[worst_i] * best.x[worst_i] + q0[worst_i] + aty[worst_i];
        let (comp_eq, comp_hi, comp_lo, comp_other) = complementarity_parts(
            &original,
            &b0,
            &best.x,
            &best.y_orig,
            problem.n_eq,
            problem.n_nn,
        );
        let pairing = pairing_note(
            &original,
            &b0,
            &best.x,
            &best.y_orig,
            problem.n_eq,
            problem.n_nn,
        );
        return Err(PdhgError::Failed(format!(
            "Metal-PDHG stopped after {iter} iterations at row {:.3e}, dual {:.3e}, gap {:.3e} (stationarity {:.3e}, complementarity {:.3e}), dual-cone {:.3e} (tolerance {KKT_TOL:.0e}); worst i={worst_i} rd={worst_rd:.3e} g={g:.3e} x={:.3e} P={:.3e} q={:.3e}; comp eq={comp_eq:.3e} trust_hi={comp_hi:.3e} trust_lo={comp_lo:.3e} other={comp_other:.3e}; {pairing}",
            r.rp_inf,
            r.rd,
            r.gap,
            r.stationarity,
            r.complementarity,
            r.dy,
            best.x[worst_i],
            p0[worst_i],
            q0[worst_i]
        )));
    }
    Ok(Solution {
        x: best.x,
        y: best.y_orig,
        iterations: iter,
        seconds: wall_seconds() - started,
        primal: r.rp,
        dual: r.rd,
        gap: r.gap,
        initial_metric,
    })
}

/// Map an original-space saddle into the equilibrated variables.
///
/// `x = C z` and `y = E y' / α`, so `z = x / c` and `y' = y · α / e`.
/// A missing or mismatched seed stays at the origin and does not count as a
/// sample of the average.
fn scaled_start(
    warm: &Option<WarmStart>,
    scale: &Scale,
    n: usize,
    m: usize,
) -> (Vec<f64>, Vec<f64>, u32) {
    let Some(warm) = warm else {
        return (vec![0.0; n], vec![0.0; m], 0);
    };
    if warm.x.len() != n || warm.y.len() != m || !finite(&warm.x) || !finite(&warm.y) {
        return (vec![0.0; n], vec![0.0; m], 0);
    }
    let z: Vec<f64> = warm.x.iter().zip(&scale.c).map(|(x, c)| x / c).collect();
    let y: Vec<f64> = warm
        .y
        .iter()
        .zip(&scale.e)
        .map(|(y, e)| y * scale.alpha / e)
        .collect();
    if !finite(&z) || !finite(&y) {
        return (vec![0.0; n], vec![0.0; m], 0);
    }
    (z, y, 1)
}

/// Pock–Chambolle diagonal steps, α = 1, with one step length per cone.
///
/// A smaller dual step on every row of a cone is still inside the
/// convergence condition, and it keeps the prox equal to the Euclidean
/// projection onto that cone. Rows with no entries (the trust radius) take
/// the cone's step.
fn diagonal_steps(
    a: &Csr,
    q: &[f64],
    n_eq: usize,
    n_nn: usize,
    n_soc: usize,
    n_trust: usize,
) -> (Vec<f64>, Vec<f64>) {
    let mut col_sum = vec![0.0; a.n];
    let mut row_sum = vec![0.0; a.m];
    for i in 0..a.m {
        for k in a.row_ptr[i] as usize..a.row_ptr[i + 1] as usize {
            let v = a.val[k].abs();
            row_sum[i] += v;
            col_sum[a.col[k] as usize] += v;
        }
    }
    let step = |sum: f64| {
        if sum > 0.0 {
            (0.9 / sum).clamp(1.0e-10, 1.0e6)
        } else {
            1.0
        }
    };
    // A mass slack with no curvature takes a step τ|q| unless |q| is in the
    // denominator. The same applies to a displacement whose linear term is the
    // exact mass penalty ϱ M folded in: τ from A alone would jump by |q|.
    // Slope slacks sit near |q| ~ 1 and keep the step from A.
    let tau: Vec<f64> = col_sum
        .iter()
        .zip(q)
        .map(|(sum, qj)| {
            let extra = if qj.abs() > 10.0 { qj.abs() } else { 0.0 };
            step(sum + extra)
        })
        .collect();
    let mut sigma = vec![1.0; a.m];
    let nn_end = n_eq + n_nn;
    for i in 0..nn_end {
        sigma[i] = step(row_sum[i]);
    }
    for cone in 0..n_soc {
        let begin = nn_end + 4 * cone;
        let mag = row_sum[begin..begin + 4]
            .iter()
            .copied()
            .fold(0.0, f64::max);
        let s = step(mag);
        for value in &mut sigma[begin..begin + 4] {
            *value = s;
        }
    }
    if n_trust > 0 {
        let begin = nn_end + 4 * n_soc;
        let mag = row_sum[begin..begin + n_trust]
            .iter()
            .copied()
            .fold(0.0, f64::max);
        let s = step(mag);
        for value in &mut sigma[begin..begin + n_trust] {
            *value = s;
        }
    }
    (tau, sigma)
}

fn curvature_polish(
    scaled: &Csr,
    p: &[f64],
    q: &[f64],
    b: &[f64],
    tau: &[f64],
    sigma: &[f64],
    z: &mut [f64],
    y: &mut [f64],
    n_eq: usize,
    n_nn: usize,
    n_soc: usize,
    n_trust: usize,
    steps: u32,
) {
    let mut p_b = p.to_vec();
    let mut q_b = q.to_vec();
    let mut log_p = 0.0_f64;
    let mut n_p = 0usize;
    for &pj in p {
        if pj > 0.0 {
            log_p += pj.ln();
            n_p += 1;
        }
    }
    let p_geo = if n_p > 0 {
        (log_p / n_p as f64).exp()
    } else {
        1.0
    };
    let beta = if p_geo < 0.1 {
        (0.1 / p_geo).min(1.0e8)
    } else {
        1.0
    };
    if std::env::var_os("PDHG_LOG").is_some() {
        eprintln!("pdhg curvature beta={beta:.3e} Pgeo={p_geo:.3e}");
    }
    if beta > 1.0 {
        for v in p_b.iter_mut().chain(q_b.iter_mut()) {
            *v *= beta;
        }
        for v in y.iter_mut() {
            *v *= beta;
        }
    }
    polish_f64(
        scaled, &p_b, &q_b, b, tau, sigma, 1.0, z, y, n_eq, n_nn, n_soc, n_trust, steps,
    );
    if beta > 1.0 {
        for v in y.iter_mut() {
            *v /= beta;
        }
    }
}

fn polish_f64(
    a: &Csr,
    p: &[f64],
    q: &[f64],
    b: &[f64],
    tau: &[f64],
    sigma: &[f64],
    step_scale: f64,
    z: &mut [f64],
    y: &mut [f64],
    n_eq: usize,
    n_nn: usize,
    n_soc: usize,
    n_trust: usize,
    steps: u32,
) {
    let mut aty = vec![0.0; z.len()];
    let mut zbar = vec![0.0; z.len()];
    let mut ax = vec![0.0; y.len()];
    for _ in 0..steps {
        spmv_t(a, y, &mut aty);
        for j in 0..z.len() {
            let old = z[j];
            let step = tau[j] * step_scale;
            let num = old - step * (q[j] + aty[j]);
            let den = 1.0 + step * p[j];
            z[j] = num / den;
            zbar[j] = 2.0 * z[j] - old;
        }
        spmv(a, &zbar, &mut ax);
        for i in 0..y.len() {
            y[i] += sigma[i] * step_scale * (ax[i] - b[i]);
        }
        let projected = project_dual(y, n_eq, n_nn, n_soc, n_trust);
        y.copy_from_slice(&projected);
    }
}

fn finite(v: &[f64]) -> bool {
    v.iter().all(|x| x.is_finite())
}

fn recover(z: &[f64], y: &[f64], scale: &Scale) -> (Vec<f64>, Vec<f64>) {
    let x = z.iter().zip(&scale.c).map(|(z, c)| z * c).collect();
    let alpha = scale.alpha;
    let y = y.iter().zip(&scale.e).map(|(y, e)| y * e / alpha).collect();
    (x, y)
}

fn consider(
    best: &mut Candidate,
    z: &[f64],
    y: &[f64],
    original: &Csr,
    p0: &[f64],
    q0: &[f64],
    b0: &[f64],
    scale: &Scale,
    n_eq: usize,
    n_nn: usize,
    n_soc: usize,
    n_trust: usize,
) {
    let (x, y_orig) = recover(z, y, scale);
    if !finite(&x) || !finite(&y_orig) {
        return;
    }
    let residuals = residuals(
        original, p0, q0, b0, &x, &y_orig, n_eq, n_nn, n_soc, n_trust,
    );
    let take = if residuals.accepted() {
        !best.residuals.accepted() || residuals.metric() < best.residuals.metric()
    } else {
        !best.residuals.accepted() && residuals.progress() < best.residuals.progress()
    };
    if take {
        *best = Candidate {
            z: z.to_vec(),
            y: y.to_vec(),
            x,
            y_orig,
            residuals,
        };
    }
}

fn op_norm(a: &Csr) -> f64 {
    if a.n == 0 || a.m == 0 || a.val.is_empty() {
        return 0.0;
    }
    let mut v = vec![1.0 / (a.n as f64).sqrt(); a.n];
    let mut av = vec![0.0; a.m];
    let mut atv = vec![0.0; a.n];
    let mut norm = 0.0;
    for _ in 0..40 {
        spmv(a, &v, &mut av);
        norm = av.iter().map(|x| x * x).sum::<f64>().sqrt();
        spmv_t(a, &av, &mut atv);
        let nrm = atv.iter().map(|x| x * x).sum::<f64>().sqrt();
        if nrm == 0.0 {
            return 0.0;
        }
        for (slot, value) in v.iter_mut().zip(&atv) {
            *slot = value / nrm;
        }
    }
    norm
}

fn equilibrate(
    a: &mut Csr,
    p: &mut [f64],
    q: &mut [f64],
    b: &mut [f64],
    n_eq: usize,
    n_nn: usize,
    n_soc: usize,
    n_trust: usize,
) -> Result<Scale, PdhgError> {
    let mut c = vec![1.0; a.n];
    let mut e = vec![1.0; a.m];
    let nn_end = n_eq + n_nn;
    // Divide the objective by the slope penalty so slope slacks have a unit
    // cost, then scale each displacement column so its curvature is 1. The
    // mass slack is reduced to a unit cost and the equality row is scaled only
    // until its largest entry is 1, which keeps ϱ|s_M|. Ruiz then balances the
    // cones and the box, but it does not see the equality row: folding that
    // row in spends the curvature to cancel a column that is already capped.
    let mut slope_level = 0.0_f64;
    for j in 0..a.n {
        if p[j] == 0.0 && q[j] > 1.0 {
            if slope_level == 0.0 || q[j] < slope_level {
                slope_level = q[j];
            }
        }
    }
    let alpha = if slope_level > 1.0 {
        1.0 / slope_level
    } else {
        1.0
    };
    if alpha != 1.0 {
        for v in p.iter_mut().chain(q.iter_mut()) {
            *v *= alpha;
        }
    }
    // Slope slacks are now O(1). A mass slack is the penalty that is still
    // orders of magnitude larger (ϱ M before this division, M after it).
    // One pass: scaling each quadratic column by its own factor is the same
    // product, and a scan per column is quadratic in the Ryugu matrix.
    let mut scol = vec![1.0; a.n];
    for j in 0..a.n {
        if p[j] > 0.0 {
            scol[j] = 1.0 / p[j].sqrt();
        }
    }
    for k in 0..a.val.len() {
        a.val[k] *= scol[a.col[k] as usize];
    }
    for j in 0..a.n {
        let s = scol[j];
        if s != 1.0 {
            p[j] *= s * s;
            q[j] *= s;
            c[j] *= s;
        }
    }
    let mut mass_col = vec![false; a.n];
    if slope_level > 1.0 {
        for j in 0..a.n {
            if p[j] == 0.0 && q[j] > 100.0 {
                mass_col[j] = true;
            }
        }
    }
    if mass_col.iter().any(|flag| *flag) {
        // q' * z' equals the kilogram slack. A residual of s kilograms therefore
        // needs either a huge multiplier or a long travel in z. Balance the two
        // so each is about sqrt(s) steps: q' = sqrt(20 s), and the equality row
        // is scaled until the mass column is ±1.
        let q_ref = (0..a.n).find(|&j| mass_col[j]).map(|j| q[j]).unwrap_or(1.0);
        let mut s_hat = 0.0_f64;
        for i in 0..n_eq {
            s_hat = s_hat.max(b[i].abs() * q_ref);
        }
        let c_sigma = if s_hat > 1.0e-3 * q_ref {
            (20.0 * s_hat).sqrt() / q_ref
        } else {
            1.0 / q_ref
        };
        for j in 0..a.n {
            if mass_col[j] {
                scale_col(a, p, q, &mut c, j, c_sigma);
            }
        }
        for i in n_eq..nn_end {
            if let Some((j, val)) = sole_entry(a, i) {
                if mass_col[j] && val != 0.0 {
                    scale_row(a, b, &mut e, i, 1.0 / val.abs());
                }
            }
        }
        for i in 0..n_eq {
            let mass = row_mask_max(a, i, &mass_col);
            let mut other = 0.0_f64;
            for k in a.row_ptr[i] as usize..a.row_ptr[i + 1] as usize {
                if !mass_col[a.col[k] as usize] {
                    other = other.max(a.val[k].abs());
                }
            }
            if s_hat > 1.0e-3 * q_ref && mass > 0.0 {
                // The slack has to travel. Make its column ±1 even if that
                // leaves displacement entries larger than 1.
                scale_row(a, b, &mut e, i, 1.0 / mass);
            } else {
                let largest = mass.max(other);
                if largest > 0.0 {
                    scale_row(a, b, &mut e, i, 1.0 / largest);
                }
            }
        }
    }
    for _ in 0..10 {
        let mut col_max = vec![0.0_f64; a.n];
        // Equality rows are already capped. Letting them set the column scale
        // spends the displacement curvature on the mass coupling.
        for i in n_eq..a.m {
            for k in a.row_ptr[i] as usize..a.row_ptr[i + 1] as usize {
                col_max[a.col[k] as usize] = col_max[a.col[k] as usize].max(a.val[k].abs());
            }
        }
        let mut scol = vec![1.0; a.n];
        for j in 0..a.n {
            if mass_col[j] {
                scol[j] = 1.0;
                continue;
            }
            // |q| is not part of the column norm. Folding it in shrinks a penalty
            // column inside a cone whose other entries are already O(1), and the
            // row scale then cannot restore that entry. The linear term stays in
            // the proximal map.
            let magnitude = col_max[j].max(p[j].max(0.0).sqrt());
            if magnitude > 1.0e-8 {
                scol[j] = 1.0 / magnitude.sqrt();
            }
        }
        for k in 0..a.val.len() {
            a.val[k] *= scol[a.col[k] as usize];
        }
        for j in 0..a.n {
            let s = scol[j];
            p[j] *= s * s;
            q[j] *= s;
            c[j] *= s;
        }
        for i in n_eq..nn_end {
            scale_row(a, b, &mut e, i, row_scale(row_max(a, i)));
        }
        for cone in 0..n_soc {
            let begin = nn_end + 4 * cone;
            let mut magnitude = 0.0_f64;
            for i in begin..begin + 4 {
                magnitude = magnitude.max(row_max(a, i));
            }
            let s = row_scale(magnitude);
            for i in begin..begin + 4 {
                scale_row(a, b, &mut e, i, s);
            }
        }
        if n_trust > 0 {
            let begin = nn_end + 4 * n_soc;
            let mut magnitude = 0.0_f64;
            for i in begin..begin + n_trust {
                magnitude = magnitude.max(row_max(a, i));
            }
            let s = row_scale(magnitude);
            for i in begin..begin + n_trust {
                scale_row(a, b, &mut e, i, s);
            }
        }
    }
    let finite_scale = alpha.is_finite()
        && alpha > 0.0
        && c.iter().chain(e.iter()).all(|v| v.is_finite() && *v > 0.0)
        && p.iter()
            .chain(q.iter())
            .chain(b.iter())
            .chain(a.val.iter())
            .all(|v| v.is_finite());
    if !finite_scale {
        return Err(PdhgError::Failed(
            "PDHG equilibration produced a non-finite scaling".into(),
        ));
    }
    if std::env::var_os("PDHG_LOG").is_some() {
        let mut q_max = 0.0_f64;
        let mut q_at = 0usize;
        for (j, &qj) in q.iter().enumerate() {
            if qj.abs() > q_max {
                q_max = qj.abs();
                q_at = j;
            }
        }
        let mut p_med = 0.0_f64;
        let mut n_p = 0usize;
        for &pj in p.iter() {
            if pj > 0.0 {
                p_med += pj.ln();
                n_p += 1;
            }
        }
        let p_geo = if n_p > 0 {
            (p_med / n_p as f64).exp()
        } else {
            0.0
        };
        let mut mass_a = 0.0_f64;
        let mut bound_a = 0.0_f64;
        if n_eq > 0 {
            for k in a.row_ptr[0] as usize..a.row_ptr[1] as usize {
                if a.col[k] as usize == q_at {
                    mass_a = a.val[k];
                }
            }
        }
        for i in n_eq..nn_end {
            for k in a.row_ptr[i] as usize..a.row_ptr[i + 1] as usize {
                if a.col[k] as usize == q_at {
                    bound_a = a.val[k];
                }
            }
        }
        eprintln!(
            "pdhg scale q_max={q_max:.3e} at {q_at} c={:.3e} Pgeo={p_geo:.3e} e_mass={:.3e} A_mass={mass_a:.3e} A_bound={bound_a:.3e}",
            c[q_at],
            if n_eq > 0 { e[0] } else { 0.0 }
        );
    }
    Ok(Scale { c, e, alpha })
}

fn scale_col(a: &mut Csr, p: &mut [f64], q: &mut [f64], c: &mut [f64], j: usize, s: f64) {
    if s == 1.0 {
        return;
    }
    for k in 0..a.val.len() {
        if a.col[k] as usize == j {
            a.val[k] *= s;
        }
    }
    p[j] *= s * s;
    q[j] *= s;
    c[j] *= s;
}

fn sole_entry(a: &Csr, i: usize) -> Option<(usize, f64)> {
    let begin = a.row_ptr[i] as usize;
    let end = a.row_ptr[i + 1] as usize;
    if end == begin + 1 {
        Some((a.col[begin] as usize, a.val[begin]))
    } else {
        None
    }
}

fn row_mask_max(a: &Csr, i: usize, mask: &[bool]) -> f64 {
    let mut magnitude = 0.0_f64;
    for k in a.row_ptr[i] as usize..a.row_ptr[i + 1] as usize {
        if mask[a.col[k] as usize] {
            magnitude = magnitude.max(a.val[k].abs());
        }
    }
    magnitude
}

fn row_max(a: &Csr, i: usize) -> f64 {
    let mut magnitude = 0.0_f64;
    for k in a.row_ptr[i] as usize..a.row_ptr[i + 1] as usize {
        magnitude = magnitude.max(a.val[k].abs());
    }
    magnitude
}

fn row_scale(magnitude: f64) -> f64 {
    // A row whose entries are numerical dust (a degenerate overhang Jacobian)
    // must not be amplified. Ten Ruiz passes would otherwise turn a 1e-16
    // coefficient into an O(1) row and a 1e14 multiplier.
    if magnitude > 1.0e-8 {
        1.0 / magnitude.sqrt()
    } else {
        1.0
    }
}

fn scale_row(a: &mut Csr, b: &mut [f64], e: &mut [f64], i: usize, s: f64) {
    if s == 1.0 {
        return;
    }
    b[i] *= s;
    e[i] *= s;
    for k in a.row_ptr[i] as usize..a.row_ptr[i + 1] as usize {
        a.val[k] *= s;
    }
}

/// Active trust rows are sole ±1 entries. Set that multiplier so the
/// displacement gradient vanishes, and keep the update only when the other
/// f64 residuals stay inside the acceptance tolerance.
fn balance_active_multipliers(
    a: &Csr,
    p: &[f64],
    q: &[f64],
    b: &[f64],
    x: &[f64],
    y: &[f64],
    n_eq: usize,
    n_nn: usize,
    n_soc: usize,
    n_trust: usize,
) -> Option<(Vec<f64>, Vec<f64>)> {
    let mut aty = vec![0.0; a.n];
    spmv_t(a, y, &mut aty);
    let nn_end = n_eq + n_nn;
    let mut tight = vec![(usize::MAX, f64::INFINITY, 0.0_f64); a.n];
    for i in n_eq..nn_end {
        let Some((j, aval)) = sole_entry(a, i) else {
            continue;
        };
        if aval == 0.0 || p[j] <= 0.0 {
            continue;
        }
        let slack = (b[i] - aval * x[j]).abs();
        if slack < tight[j].1 {
            tight[j] = (i, slack, aval);
        }
    }
    let mut y2 = y.to_vec();
    for j in 0..a.n {
        let (i, slack, aval) = tight[j];
        if i == usize::MAX || slack > 1.0e-6 || aval == 0.0 {
            continue;
        }
        let g = p[j] * x[j] + q[j] + aty[j];
        let g_without = g - aval * y[i];
        y2[i] = (-g_without / aval).max(0.0);
    }
    let old = residuals(a, p, q, b, x, y, n_eq, n_nn, n_soc, n_trust);
    let mut kept: Option<Vec<f64>> = None;
    let mut kept_rd = old.rd;
    for alpha in [1.0_f64, 0.5, 0.25, 0.1, 0.03] {
        let y_try: Vec<f64> = y
            .iter()
            .zip(&y2)
            .map(|(old, new)| old + alpha * (new - old))
            .collect();
        let new = residuals(a, p, q, b, x, &y_try, n_eq, n_nn, n_soc, n_trust);
        if new.rp_inf <= KKT_TOL && new.gap <= KKT_TOL && new.dy <= KKT_TOL && new.rd < kept_rd {
            kept_rd = new.rd;
            let done = new.rd <= KKT_TOL;
            kept = Some(y_try);
            if done {
                break;
            }
        }
    }
    kept.map(|y_new| (x.to_vec(), y_new))
}

/// One Newton step on the quadratic columns.
///
/// `dx = -(g - A_eqᵀ λ) / P` and `y_eq -= λ`, with `λ` chosen so the equality
/// rows do not move. A column whose step would leave a bound, or whose step is
/// larger than `1e-4` in `x = d/R₀`, is left where the iteration put it.
/// Returns the updated point only when the row, gap, and dual cone stay inside
/// tolerance and the block stationarity improves.
fn displacement_newton(
    a: &Csr,
    p: &[f64],
    q: &[f64],
    b: &[f64],
    x: &[f64],
    y: &[f64],
    n_eq: usize,
    n_nn: usize,
    n_soc: usize,
    n_trust: usize,
) -> Option<(Vec<f64>, Vec<f64>)> {
    if n_eq > 4 || x.is_empty() {
        return None;
    }
    let mut aty = vec![0.0; a.n];
    spmv_t(a, y, &mut aty);
    let g: Vec<f64> = (0..a.n).map(|j| p[j] * x[j] + q[j] + aty[j]).collect();
    let mut free = vec![false; a.n];
    for j in 0..a.n {
        // Up to one trust radius in x = d/R₀. A shorter cap left the columns
        // that still carry the dual residual out of the correction.
        if p[j] > 0.0 && g[j].abs() / p[j] <= 2.0e-2 {
            free[j] = true;
        }
    }
    let nn_end = n_eq + n_nn;
    let mut by_col: Vec<Vec<(usize, f64)>> = vec![Vec::new(); a.n];
    for i in n_eq..nn_end {
        if let Some((j, aval)) = sole_entry(a, i) {
            if aval != 0.0 {
                by_col[j].push((i, aval));
            }
        }
    }
    for j in 0..a.n {
        if !free[j] {
            continue;
        }
        for &(i, aval) in &by_col[j] {
            let slack = b[i] - aval * x[j];
            if slack.abs() <= 1.0e-8 {
                free[j] = false;
                break;
            }
        }
    }
    let mut gram = vec![0.0; n_eq * n_eq];
    let mut rhs = vec![0.0; n_eq];
    for i in 0..n_eq {
        for k in a.row_ptr[i] as usize..a.row_ptr[i + 1] as usize {
            let j = a.col[k] as usize;
            if !free[j] {
                continue;
            }
            let aij = a.val[k];
            rhs[i] += aij * g[j] / p[j];
            gram[i * n_eq + i] += aij * aij / p[j];
        }
    }
    // Off-diagonal Gram entries, for more than one equality.
    for i in 0..n_eq {
        for r in (i + 1)..n_eq {
            let mut acc = 0.0;
            for k in a.row_ptr[i] as usize..a.row_ptr[i + 1] as usize {
                let j = a.col[k] as usize;
                if !free[j] {
                    continue;
                }
                for t in a.row_ptr[r] as usize..a.row_ptr[r + 1] as usize {
                    if a.col[t] as usize == j {
                        acc += a.val[k] * a.val[t] / p[j];
                    }
                }
            }
            gram[i * n_eq + r] = acc;
            gram[r * n_eq + i] = acc;
        }
    }
    let lambda = solve_gram(&gram, &rhs, n_eq)?;
    let mut eq_col = vec![0.0; a.n];
    for i in 0..n_eq {
        for k in a.row_ptr[i] as usize..a.row_ptr[i + 1] as usize {
            eq_col[a.col[k] as usize] += lambda[i] * a.val[k];
        }
    }
    let dx: Vec<f64> = (0..a.n)
        .map(|j| {
            if free[j] {
                -(g[j] - eq_col[j]) / p[j]
            } else {
                0.0
            }
        })
        .collect();
    let old = residuals(a, p, q, b, x, y, n_eq, n_nn, n_soc, n_trust);
    let mut kept: Option<(Vec<f64>, Vec<f64>)> = None;
    let mut kept_rd = old.rd;
    for alpha in [1.0_f64, 0.5, 0.25, 0.125, 0.0625, 0.03, 0.01] {
        let mut x2 = x.to_vec();
        for j in 0..a.n {
            if !free[j] {
                continue;
            }
            let trial = x[j] + alpha * dx[j];
            let mut blocked = false;
            for &(i, aval) in &by_col[j] {
                if aval * trial > b[i] + 1.0e-9 {
                    blocked = true;
                    break;
                }
            }
            if !blocked {
                x2[j] = trial;
            }
        }
        let mut y2 = y.to_vec();
        for i in 0..n_eq {
            y2[i] -= alpha * lambda[i];
        }
        let new = residuals(a, p, q, b, &x2, &y2, n_eq, n_nn, n_soc, n_trust);
        if new.rp_inf <= KKT_TOL && new.gap <= KKT_TOL && new.dy <= KKT_TOL && new.rd < kept_rd {
            kept_rd = new.rd;
            let done = new.rd <= KKT_TOL;
            kept = Some((x2, y2));
            if done {
                break;
            }
        }
    }
    if kept.is_none() && std::env::var_os("PDHG_LOG").is_some() {
        eprintln!("pdhg newton reject rd={:.3e} (was {:.3e})", kept_rd, old.rd);
    }
    kept
}

struct TrustPair {
    hi: usize,
    lo: usize,
    j: usize,
    aval_hi: f64,
    aval_lo: f64,
    x_hi: f64,
    x_lo: f64,
}

fn trust_pairs(a: &Csr, b: &[f64], n_eq: usize, n_nn: usize) -> Vec<TrustPair> {
    let nn_end = n_eq + n_nn;
    let mut hi_at = vec![usize::MAX; a.n];
    let mut lo_at = vec![usize::MAX; a.n];
    for i in n_eq..nn_end {
        let Some((j, aval)) = sole_entry(a, i) else {
            continue;
        };
        if aval > 0.0 {
            hi_at[j] = i;
        } else if aval < 0.0 {
            lo_at[j] = i;
        }
    }
    let mut pairs = Vec::new();
    for j in 0..a.n {
        let hi = hi_at[j];
        let lo = lo_at[j];
        if hi == usize::MAX || lo == usize::MAX {
            continue;
        }
        if (b[hi] - b[lo]).abs() > 1.0e-8 * (1.0 + b[hi].abs()) {
            continue;
        }
        let Some((_, aval_hi)) = sole_entry(a, hi) else {
            continue;
        };
        let Some((_, aval_lo)) = sole_entry(a, lo) else {
            continue;
        };
        if aval_hi == 0.0 || aval_lo == 0.0 {
            continue;
        }
        pairs.push(TrustPair {
            hi,
            lo,
            j,
            aval_hi,
            aval_lo,
            x_hi: b[hi] / aval_hi,
            x_lo: b[lo] / aval_lo,
        });
    }
    pairs
}

fn boxed_interior(x: f64, pair: &TrustPair) -> bool {
    let width = (pair.x_hi - pair.x_lo).abs().max(1.0);
    x < pair.x_hi - 1.0e-6 * width && x > pair.x_lo + 1.0e-6 * width
}

/// Cancel the shared part of a trust pair. `y_hi − y_lo` is unchanged, so the
/// stationarity gradient does not move, and the complementarity pairing falls.
fn reduce_trust_overlap(y: &mut [f64], pairs: &[TrustPair]) {
    for pair in pairs {
        let share = y[pair.hi].min(y[pair.lo]).max(0.0);
        if share > 0.0 {
            y[pair.hi] -= share;
            y[pair.lo] -= share;
        }
    }
}

fn lay_trust_multipliers(
    a: &Csr,
    p: &[f64],
    q: &[f64],
    x: &[f64],
    y: &mut [f64],
    pairs: &[TrustPair],
) {
    let mut aty = vec![0.0; a.n];
    spmv_t(a, y, &mut aty);
    for pair in pairs {
        if p[pair.j] <= 0.0 {
            continue;
        }
        let contrib = pair.aval_hi * y[pair.hi] + pair.aval_lo * y[pair.lo];
        let base = p[pair.j] * x[pair.j] + q[pair.j] + aty[pair.j] - contrib;
        let width = (pair.x_hi - pair.x_lo).abs().max(1.0);
        let at_hi = (x[pair.j] - pair.x_hi).abs() <= 1.0e-8 * width;
        let at_lo = (x[pair.j] - pair.x_lo).abs() <= 1.0e-8 * width;
        if at_hi && !at_lo && base * pair.aval_hi < 0.0 {
            y[pair.hi] = (-base / pair.aval_hi).max(0.0);
            y[pair.lo] = 0.0;
        } else if at_lo && !at_hi && base * pair.aval_lo < 0.0 {
            y[pair.lo] = (-base / pair.aval_lo).max(0.0);
            y[pair.hi] = 0.0;
        } else {
            y[pair.hi] = 0.0;
            y[pair.lo] = 0.0;
        }
    }
}

/// Equality-constrained Newton on the diagonal columns, clamped to the trust
/// box.
///
/// The step solves `P dx + Aᵀ dλ = −g` and `A dx = b − Ax` together, so the
/// equality residual and the stationarity residual leave at the same time.
/// A column that would cross a wall is pinned there and removed from the free
/// set; the multiplier is updated only on a round that stays inside the box.
/// Slope-cone multipliers are not touched.
fn equality_box_newton(
    a: &Csr,
    p: &[f64],
    q: &[f64],
    b: &[f64],
    x: &[f64],
    y: &[f64],
    pairs: &[TrustPair],
    n_eq: usize,
) -> (Vec<f64>, Vec<f64>) {
    let mut x2 = x.to_vec();
    let mut y2 = y.to_vec();
    if n_eq == 0 || n_eq > 4 {
        return (x2, y2);
    }
    let mut bound = vec![false; a.n];
    for _ in 0..8 {
        for pair in pairs {
            if p[pair.j] > 0.0 && !bound[pair.j] && boxed_interior(x2[pair.j], pair) {
                y2[pair.hi] = 0.0;
                y2[pair.lo] = 0.0;
            }
        }
        let mut aty = vec![0.0; a.n];
        spmv_t(a, &y2, &mut aty);
        let g: Vec<f64> = (0..a.n).map(|j| p[j] * x2[j] + q[j] + aty[j]).collect();
        let mut free = vec![false; a.n];
        let mut any = false;
        for pair in pairs {
            if p[pair.j] > 0.0 && !bound[pair.j] && boxed_interior(x2[pair.j], pair) {
                free[pair.j] = true;
                any = true;
            }
        }
        if !any {
            break;
        }
        let mut gram = vec![0.0; n_eq * n_eq];
        let mut rhs = vec![0.0; n_eq];
        for i in 0..n_eq {
            for k in a.row_ptr[i] as usize..a.row_ptr[i + 1] as usize {
                let j = a.col[k] as usize;
                if !free[j] || p[j] <= 0.0 {
                    continue;
                }
                let aij = a.val[k];
                rhs[i] += aij * g[j] / p[j];
                gram[i * n_eq + i] += aij * aij / p[j];
            }
        }
        let mut ax = vec![0.0; a.m];
        spmv(a, &x2, &mut ax);
        for i in 0..n_eq {
            rhs[i] += b[i] - ax[i];
        }
        for i in 0..n_eq {
            for r in (i + 1)..n_eq {
                let mut acc = 0.0;
                for k in a.row_ptr[i] as usize..a.row_ptr[i + 1] as usize {
                    let j = a.col[k] as usize;
                    if !free[j] || p[j] <= 0.0 {
                        continue;
                    }
                    for t in a.row_ptr[r] as usize..a.row_ptr[r + 1] as usize {
                        if a.col[t] as usize == j {
                            acc += a.val[k] * a.val[t] / p[j];
                        }
                    }
                }
                gram[i * n_eq + r] = acc;
                gram[r * n_eq + i] = acc;
            }
        }
        let Some(lambda) = solve_gram(&gram, &rhs, n_eq) else {
            break;
        };
        let mut eq_col = vec![0.0; a.n];
        for i in 0..n_eq {
            for k in a.row_ptr[i] as usize..a.row_ptr[i + 1] as usize {
                eq_col[a.col[k] as usize] += lambda[i] * a.val[k];
            }
        }
        let mut pinned = Vec::new();
        for pair in pairs {
            if !free[pair.j] {
                continue;
            }
            let dx = -(g[pair.j] - eq_col[pair.j]) / p[pair.j];
            let trial = x2[pair.j] + dx;
            let (lo, hi) = if pair.x_lo <= pair.x_hi {
                (pair.x_lo, pair.x_hi)
            } else {
                (pair.x_hi, pair.x_lo)
            };
            if trial < lo || trial > hi {
                pinned.push((pair.j, trial.clamp(lo, hi)));
            }
        }
        if !pinned.is_empty() {
            for (j, wall) in pinned {
                bound[j] = true;
                x2[j] = wall;
            }
            continue;
        }
        for pair in pairs {
            if !free[pair.j] {
                continue;
            }
            x2[pair.j] -= (g[pair.j] - eq_col[pair.j]) / p[pair.j];
        }
        for i in 0..n_eq {
            y2[i] -= lambda[i];
        }
        break;
    }
    lay_trust_multipliers(a, p, q, &x2, &mut y2, pairs);
    (x2, y2)
}

/// Put the mass pair on one nonnegative slack, keeping `σ+ − σ−` fixed.
///
/// A shared positive part cancels in the equality and only adds penalty. A
/// negative side is infeasible; folding it into the other slack restores the
/// bound without changing the mass row.
fn fold_mass_slacks(a: &Csr, p: &[f64], x: &mut [f64], n_eq: usize) {
    if n_eq != 1 {
        return;
    }
    let mut plus = None;
    let mut minus = None;
    for k in a.row_ptr[0] as usize..a.row_ptr[1] as usize {
        let j = a.col[k] as usize;
        if p[j] != 0.0 {
            continue;
        }
        let v = a.val[k];
        if (v - 1.0).abs() <= 1.0e-12 {
            plus = Some(j);
        } else if (v + 1.0).abs() <= 1.0e-12 {
            minus = Some(j);
        }
    }
    let (Some(jp), Some(jm)) = (plus, minus) else {
        return;
    };
    let s = x[jp] - x[jm];
    if s >= 0.0 {
        x[jp] = s;
        x[jm] = 0.0;
    } else {
        x[jp] = 0.0;
        x[jm] = -s;
    }
}

/// A linear column sitting within `1e-4` of its bound is put on the bound.
///
/// The mass-slack multiplier is `ϱ M`. A slack of `1e-8` then contributes an
/// `O(ϱ M · 1e-8)` pairing to the gap while the row residual is already under
/// the tolerance. The displacement gradient does not depend on that column.
fn snap_linear_bounds(
    a: &Csr,
    p: &[f64],
    b: &[f64],
    x: &[f64],
    n_eq: usize,
    n_nn: usize,
) -> Vec<f64> {
    let mut x2 = x.to_vec();
    let nn_end = n_eq + n_nn;
    for i in n_eq..nn_end {
        let Some((j, aval)) = sole_entry(a, i) else {
            continue;
        };
        if p[j] != 0.0 || aval == 0.0 {
            continue;
        }
        let slack = b[i] - aval * x2[j];
        if slack.abs() <= 1.0e-4 {
            x2[j] = b[i] / aval;
        }
    }
    x2
}

/// Remove a complementarity gap that lives on the trust box.
///
/// Overlap reduction keeps `Aᵀy` and is valid with slope cones present. The
/// equality Newton runs only when there is no slope cone, because that is the
/// diagonal box problem the multipliers were standing in for. The point is
/// returned only when the four f64 residuals meet `KKT_TOL`.
fn close_box_gap(
    a: &Csr,
    p: &[f64],
    q: &[f64],
    b: &[f64],
    x: &[f64],
    y: &[f64],
    n_eq: usize,
    n_nn: usize,
    n_soc: usize,
    n_trust: usize,
) -> Option<(Vec<f64>, Vec<f64>)> {
    let mut x_snap = snap_linear_bounds(a, p, b, x, n_eq, n_nn);
    fold_mass_slacks(a, p, &mut x_snap, n_eq);
    let pairs = trust_pairs(a, b, n_eq, n_nn);
    let mut y2 = y.to_vec();
    reduce_trust_overlap(&mut y2, &pairs);
    let snapped = residuals(a, p, q, b, &x_snap, &y2, n_eq, n_nn, n_soc, n_trust);
    if snapped.accepted() {
        return Some((x_snap, y2));
    }
    if pairs.is_empty() || n_soc != 0 {
        return None;
    }
    let (x2, y_newton) = equality_box_newton(a, p, q, b, &x_snap, &y2, &pairs, n_eq);
    let new = residuals(a, p, q, b, &x2, &y_newton, n_eq, n_nn, n_soc, n_trust);
    if new.accepted() {
        Some((x2, y_newton))
    } else {
        if std::env::var_os("PDHG_LOG").is_some() {
            eprintln!(
                "box-gap reject rp={:.3e} rd={:.3e} gap={:.3e} dy={:.3e} comp={:.3e} (snap rp={:.3e} rd={:.3e} gap={:.3e})",
                new.rp_inf,
                new.rd,
                new.gap,
                new.dy,
                new.complementarity,
                snapped.rp_inf,
                snapped.rd,
                snapped.gap
            );
        }
        None
    }
}

/// Split `y · (b − Ax)` into the mass equality, the two trust-box sides, and
/// every other row. A trust pair is the sole `+1` and `−1` rows of one column
/// that share the same right-hand side.
fn complementarity_parts(
    a: &Csr,
    b: &[f64],
    x: &[f64],
    y: &[f64],
    n_eq: usize,
    n_nn: usize,
) -> (f64, f64, f64, f64) {
    let mut ax = vec![0.0; a.m];
    spmv(a, x, &mut ax);
    let nn_end = n_eq + n_nn;
    let mut hi = vec![usize::MAX; a.n];
    let mut lo = vec![usize::MAX; a.n];
    for i in n_eq..nn_end {
        let Some((j, aval)) = sole_entry(a, i) else {
            continue;
        };
        if aval > 0.0 {
            hi[j] = i;
        } else if aval < 0.0 {
            lo[j] = i;
        }
    }
    let mut side = vec![0u8; a.m];
    for j in 0..a.n {
        let h = hi[j];
        let l = lo[j];
        if h == usize::MAX || l == usize::MAX {
            continue;
        }
        if (b[h] - b[l]).abs() <= 1.0e-8 * (1.0 + b[h].abs()) {
            side[h] = 1;
            side[l] = 2;
        }
    }
    let mut eq = 0.0;
    let mut trust_hi = 0.0;
    let mut trust_lo = 0.0;
    let mut other = 0.0;
    for i in 0..a.m {
        let product = y[i] * (b[i] - ax[i]);
        if i < n_eq {
            eq += product;
        } else if side[i] == 1 {
            trust_hi += product;
        } else if side[i] == 2 {
            trust_lo += product;
        } else {
            other += product;
        }
    }
    (eq, trust_hi, trust_lo, other)
}

/// Equality multiplier and the largest non-trust pairing, so a gap failure
/// names the row that actually carries it.
fn pairing_note(a: &Csr, b: &[f64], x: &[f64], y: &[f64], n_eq: usize, n_nn: usize) -> String {
    let mut ax = vec![0.0; a.m];
    spmv(a, x, &mut ax);
    let (eq_y, eq_s) = if n_eq > 0 {
        (y[0], b[0] - ax[0])
    } else {
        (0.0, 0.0)
    };
    let nn_end = n_eq + n_nn;
    let mut best = 0.0_f64;
    let mut best_i = 0usize;
    for i in n_eq..a.m {
        let product = y[i] * (b[i] - ax[i]);
        if product.abs() > best.abs() {
            best = product;
            best_i = i;
        }
    }
    let nnz = if a.m > 0 {
        a.row_ptr[best_i + 1] - a.row_ptr[best_i]
    } else {
        0
    };
    format!(
        "eq_y={eq_y:.3e} eq_s={eq_s:.3e} nn={nn_end} top_row={best_i} top={best:.3e} y={:.3e} s={:.3e} nnz={nnz}",
        y.get(best_i).copied().unwrap_or(0.0),
        if a.m > 0 { b[best_i] - ax[best_i] } else { 0.0 }
    )
}

fn solve_gram(gram: &[f64], rhs: &[f64], n: usize) -> Option<Vec<f64>> {
    if n == 0 {
        return Some(Vec::new());
    }
    let mut m = gram.to_vec();
    let mut r = rhs.to_vec();
    for k in 0..n {
        let mut pivot = k;
        for i in (k + 1)..n {
            if m[i * n + k].abs() > m[pivot * n + k].abs() {
                pivot = i;
            }
        }
        if m[pivot * n + k].abs() < 1.0e-18 {
            return Some(vec![0.0; n]);
        }
        if pivot != k {
            for j in k..n {
                m.swap(k * n + j, pivot * n + j);
            }
            r.swap(k, pivot);
        }
        let diag = m[k * n + k];
        for i in (k + 1)..n {
            let f = m[i * n + k] / diag;
            for j in k..n {
                m[i * n + j] -= f * m[k * n + j];
            }
            r[i] -= f * r[k];
        }
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut acc = r[i];
        for j in (i + 1)..n {
            acc -= m[i * n + j] * x[j];
        }
        x[i] = acc / m[i * n + i];
    }
    if x.iter().all(|v| v.is_finite()) {
        Some(x)
    } else {
        None
    }
}

fn residuals(
    a: &Csr,
    p: &[f64],
    q: &[f64],
    b: &[f64],
    x: &[f64],
    y: &[f64],
    n_eq: usize,
    n_nn: usize,
    n_soc: usize,
    n_trust: usize,
) -> Residuals {
    let mut ax = vec![0.0; a.m];
    spmv(a, x, &mut ax);
    let slack: Vec<f64> = b.iter().zip(&ax).map(|(b, ax)| b - ax).collect();
    let proj = primal_project(&slack, n_eq, n_nn, n_soc, n_trust);
    let mut rp2 = 0.0;
    let mut rp_inf = 0.0_f64;
    for (v, s) in slack.iter().zip(&proj) {
        let d = v - s;
        rp2 += d * d;
        rp_inf = rp_inf.max(d.abs());
    }
    let bnorm = b.iter().map(|v| v * v).sum::<f64>().sqrt();
    let rp = rp2.sqrt() / (1.0 + bnorm);

    let y_proj = project_dual(y, n_eq, n_nn, n_soc, n_trust);
    let mut dy2 = 0.0;
    for (v, s) in y.iter().zip(&y_proj) {
        let d = v - s;
        dy2 += d * d;
    }
    let ynorm = y.iter().map(|v| v * v).sum::<f64>().sqrt();
    let dy = dy2.sqrt() / (1.0 + ynorm);

    let mut aty = vec![0.0; a.n];
    spmv_t(a, y, &mut aty);
    let mut rd_block = 0.0_f64;
    let mut quad = 0.0;
    let mut lin = 0.0;
    let mut stationarity = 0.0;
    for i in 0..a.n {
        let g = p[i] * x[i] + q[i] + aty[i];
        // Quadratic columns are the free displacements. Linear columns are the
        // slope and mass slacks. A single ‖g‖/(1+‖q‖) lets ϱ M hide a bad
        // displacement, so each variable is judged on its own. For a
        // displacement, |P x| is ~0 when the node has not moved, and the
        // absolute force 1e-3 is smaller than a scaled iteration can certify
        // once the vertex stiffness is ~1e5: the same point is a fraction of
        // a millimetre from equilibrium, and cancelling the force outright
        // leaves the slope cone by a few times 1e-3. The extra P·10⁻² is the
        // force of a one-percent-of-radius step, the order of the trust region.
        let scale = if p[i] > 0.0 {
            1.0 + q[i].abs() + (p[i] * x[i]).abs() + p[i] * 1.0e-2
        } else {
            1.0 + q[i].abs()
        };
        rd_block = rd_block.max(g.abs() / scale);
        quad += p[i] * x[i] * x[i];
        lin += q[i] * x[i];
        stationarity += x[i] * g;
    }
    let rd = rd_block;
    let by = b.iter().zip(y).map(|(b, y)| b * y).sum::<f64>();
    let complementarity: f64 = slack.iter().zip(y).map(|(s, y)| y * s).sum();
    let gap_num = quad + lin + by;
    let primal = 0.5 * quad + lin;
    let dual = -0.5 * quad - by;
    let gap = gap_num.abs() / (1.0 + primal.abs() + dual.abs());
    Residuals {
        rp,
        rd,
        gap,
        dy,
        rp_inf,
        stationarity,
        complementarity,
    }
}

fn primal_project(v: &[f64], n_eq: usize, n_nn: usize, n_soc: usize, n_trust: usize) -> Vec<f64> {
    let mut s = vec![0.0; v.len()];
    let nn_end = n_eq + n_nn;
    for i in n_eq..nn_end {
        s[i] = v[i].max(0.0);
    }
    project_cones(&mut s, v, nn_end, n_soc, n_trust);
    s
}

pub(crate) fn project_dual(
    v: &[f64],
    n_eq: usize,
    n_nn: usize,
    n_soc: usize,
    n_trust: usize,
) -> Vec<f64> {
    let mut s = v.to_vec();
    let nn_end = n_eq + n_nn;
    for i in n_eq..nn_end {
        s[i] = s[i].max(0.0);
    }
    project_cones(&mut s, v, nn_end, n_soc, n_trust);
    s
}

fn project_cones(s: &mut [f64], v: &[f64], nn_end: usize, n_soc: usize, n_trust: usize) {
    for cone in 0..n_soc {
        project_soc(s, v, nn_end + 4 * cone, 4);
    }
    if n_trust > 0 {
        project_soc(s, v, nn_end + 4 * n_soc, n_trust);
    }
}

fn project_soc(s: &mut [f64], v: &[f64], begin: usize, len: usize) {
    let t = v[begin];
    let mut nrm2 = 0.0;
    for i in 1..len {
        let x = v[begin + i];
        nrm2 += x * x;
    }
    let nrm = nrm2.sqrt();
    if nrm <= t {
        s[begin..begin + len].copy_from_slice(&v[begin..begin + len]);
    } else if nrm <= -t {
        for value in &mut s[begin..begin + len] {
            *value = 0.0;
        }
    } else {
        let a = 0.5 * (nrm + t) / nrm;
        s[begin] = a * nrm;
        for i in 1..len {
            s[begin + i] = a * v[begin + i];
        }
    }
}

fn spmv(a: &Csr, x: &[f64], out: &mut [f64]) {
    for i in 0..a.m {
        let mut sum = 0.0;
        for k in a.row_ptr[i] as usize..a.row_ptr[i + 1] as usize {
            sum += a.val[k] * x[a.col[k] as usize];
        }
        out[i] = sum;
    }
}

fn spmv_t(a: &Csr, y: &[f64], out: &mut [f64]) {
    out.fill(0.0);
    for i in 0..a.m {
        let yi = y[i];
        if yi == 0.0 {
            continue;
        }
        for k in a.row_ptr[i] as usize..a.row_ptr[i + 1] as usize {
            out[a.col[k] as usize] += a.val[k] * yi;
        }
    }
}

impl Csr {
    fn from_triplets(
        m: usize,
        n: usize,
        rows: &[usize],
        cols: &[usize],
        vals: &[f64],
    ) -> Result<Self, PdhgError> {
        if rows.len() != cols.len() || cols.len() != vals.len() {
            return Err(PdhgError::Failed(
                "PDHG constraint triplets have different lengths".into(),
            ));
        }
        let mut trip = Vec::with_capacity(vals.len());
        for i in 0..vals.len() {
            if rows[i] >= m || cols[i] >= n {
                return Err(PdhgError::Failed(format!(
                    "PDHG triplet ({}, {}) is outside a {m}×{n} matrix",
                    rows[i], cols[i]
                )));
            }
            if !vals[i].is_finite() {
                return Err(PdhgError::Failed(
                    "PDHG constraint matrix has a non-finite entry".into(),
                ));
            }
            if vals[i] != 0.0 {
                trip.push((rows[i] as u32, cols[i] as u32, vals[i]));
            }
        }
        trip.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
        let mut merged: Vec<(u32, u32, f64)> = Vec::with_capacity(trip.len());
        for (r, c, v) in trip {
            if let Some(last) = merged.last_mut()
                && last.0 == r
                && last.1 == c
            {
                last.2 += v;
                continue;
            }
            merged.push((r, c, v));
        }
        merged.retain(|t| t.2 != 0.0 && t.2.is_finite());
        let mut row_ptr = vec![0u32; m + 1];
        for t in &merged {
            row_ptr[t.0 as usize + 1] += 1;
        }
        for i in 0..m {
            row_ptr[i + 1] += row_ptr[i];
        }
        let col = merged.iter().map(|t| t.1).collect();
        let val = merged.iter().map(|t| t.2).collect();
        Ok(Self {
            m,
            n,
            row_ptr,
            col,
            val,
        })
    }
}

/// Device side of the iteration (`scf/shader/pdhg.wesl`): the matrix in CSR and
/// CSC form and the iterates stay resident; one submission runs a block of
/// iterations and reads the iterates back.
mod gpu {
    use super::{Csr, PdhgError};
    use crate::gpu::{Job, Res, kernels};

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Step {
        nx: u32,
        m: u32,
        n_eq: u32,
        n_nn: u32,
        n_soc: u32,
        n_trust: u32,
        avg_count: u32,
        tau: [f32; 2],
        sigma: [f32; 2],
        n_long: u32,
        inv: [f32; 2],
    }

    pub(super) fn ds(values: &[f64]) -> Vec<[f32; 2]> {
        values
            .iter()
            .map(|&x| crate::common::math::ds_pair(x))
            .collect()
    }

    pub(super) fn undo(words: &[[f32; 2]]) -> Vec<f64> {
        words
            .iter()
            .map(|w| f64::from(w[0]) + f64::from(w[1]))
            .collect()
    }

    pub(super) struct Csc {
        pub(super) ptr: Vec<u32>,
        pub(super) row: Vec<u32>,
        pub(super) val: Vec<f64>,
    }

    pub(super) fn transpose(a: &Csr) -> Csc {
        let mut ptr = vec![0u32; a.n + 1];
        for &c in &a.col {
            ptr[c as usize + 1] += 1;
        }
        for j in 0..a.n {
            ptr[j + 1] += ptr[j];
        }
        let mut row = vec![0u32; a.val.len()];
        let mut val = vec![0.0; a.val.len()];
        let mut cursor = ptr.clone();
        for i in 0..a.m {
            for k in a.row_ptr[i] as usize..a.row_ptr[i + 1] as usize {
                let j = a.col[k] as usize;
                let p = cursor[j] as usize;
                cursor[j] += 1;
                row[p] = i as u32;
                val[p] = a.val[k];
            }
        }
        Csc { ptr, row, val }
    }

    /// Device words of `(x, y, x_avg, y_avg)`.
    pub type Words = [Vec<[f32; 2]>; 4];

    pub struct Device {
        nx: usize,
        m: usize,
        counts: [u32; 4],
        n_long: u32,
        n_chunks: u32,
        /// (row, begin, end) per chunk of a long row, and (row, first chunk, count) per long row.
        chunks: Res,
        longs: Res,
        /// Chunk partials of the long rows; trust-cone partials and its head.
        lpart: Res,
        tpart: Res,
        /// Column pointers then row indices, and the values.
        csc: [Res; 2],
        csr: [Res; 3],
        /// x, xbar, xavg, y, yavg, ax, (q, P, τ) per column, b, σ, step.
        state: [Res; 10],
        /// `pdhg_batch` header and indices, and its constants, when the problem fits one
        /// workgroup.
        batch: Option<[Res; 2]>,
    }

    /// Words of the `pdhg_batch` header, before the index arrays.
    const BATCH_HEADER: usize = 19;

    /// Whether one 256-lane workgroup holds every column, row, cone job and long-row
    /// chunk (the array sizes of `pdhg_batch`). `AE_PDHG_SPLIT=1` keeps one kernel per
    /// phase.
    pub(super) fn fits_batch(nx: usize, m: usize, n_jobs: usize, n_trust: usize, n_chunks: usize) -> bool {
        std::env::var_os("AE_PDHG_SPLIT").is_none()
            && nx <= 256
            && m <= 256
            && n_jobs <= 256
            && n_trust <= 256
            && n_chunks <= 16
    }

    impl Device {
        #[allow(clippy::too_many_arguments)]
        pub fn new(
            a: &Csr,
            p: &[f64],
            q: &[f64],
            b: &[f64],
            tau: &[f64],
            sigma: &[f64],
            n_eq: usize,
            n_nn: usize,
            n_soc: usize,
            n_trust: usize,
        ) -> Result<Self, PdhgError> {
            let e = PdhgError::Gpu;
            let csc = transpose(a);
            let long_rows: Vec<u32> = (0..a.m)
                .filter(|&i| a.row_ptr[i + 1] - a.row_ptr[i] > 32)
                .map(|i| i as u32)
                .collect();
            let (mut chunks, mut longs) = (Vec::new(), Vec::new());
            for &r in &long_rows {
                let (begin, end) = (a.row_ptr[r as usize], a.row_ptr[r as usize + 1]);
                longs.extend([
                    r,
                    (chunks.len() / 3) as u32,
                    (end - begin).div_ceil(super::LONG_CHUNK),
                ]);
                for c in (begin..end).step_by(super::LONG_CHUNK as usize) {
                    chunks.extend([r, c, (c + super::LONG_CHUNK).min(end)]);
                }
            }
            let n_chunks = (chunks.len() / 3) as u32;
            let zeros = |n: usize| Res::zeroed((n * 8) as u64).map_err(e);
            let colc: Vec<f64> = (0..a.n).flat_map(|j| [q[j], p[j], tau[j]]).collect();
            let batch = if fits_batch(a.n, a.m, n_eq + n_nn + n_soc, n_trust, n_chunks as usize)
            {
                let mut hdr = vec![0u32; BATCH_HEADER];
                hdr[1..9].copy_from_slice(&[
                    a.n as u32,
                    a.m as u32,
                    n_eq as u32,
                    n_nn as u32,
                    n_soc as u32,
                    n_trust as u32,
                    long_rows.len() as u32,
                    n_chunks,
                ]);
                // The row indices follow the column pointers, as in `csc[0]`.
                for (slot, part) in [
                    (Some(9), &csc.ptr),
                    (None, &csc.row),
                    (Some(10), &a.row_ptr),
                    (Some(11), &a.col),
                    (Some(12), &chunks),
                    (Some(13), &longs),
                ] {
                    if let Some(slot) = slot {
                        hdr[slot] = hdr.len() as u32;
                    }
                    hdr.extend_from_slice(part);
                }
                let mut cst = Vec::new();
                let values: [&[f64]; 5] = [&csc.val, &colc, &a.val, b, sigma];
                for (slot, part) in values.into_iter().enumerate() {
                    hdr[14 + slot] = cst.len() as u32;
                    cst.extend(ds(part));
                }
                Some([
                    Res::from_slice(&hdr).map_err(e)?,
                    Res::from_slice(&cst).map_err(e)?,
                ])
            } else {
                None
            };
            Ok(Self {
                nx: a.n,
                m: a.m,
                counts: [n_eq as u32, n_nn as u32, n_soc as u32, n_trust as u32],
                n_long: long_rows.len() as u32,
                n_chunks,
                chunks: Res::from_slice(&chunks).map_err(e)?,
                longs: Res::from_slice(&longs).map_err(e)?,
                lpart: zeros(n_chunks as usize + 1)?,
                tpart: zeros(n_trust.div_ceil(256) + 2)?,
                csc: [
                    Res::from_slice(&[csc.ptr, csc.row].concat()).map_err(e)?,
                    Res::from_slice(&ds(&csc.val)).map_err(e)?,
                ],
                csr: [
                    Res::from_slice(&a.row_ptr).map_err(e)?,
                    Res::from_slice(&a.col).map_err(e)?,
                    Res::from_slice(&ds(&a.val)).map_err(e)?,
                ],
                state: [
                    zeros(a.n)?,
                    zeros(a.n)?,
                    zeros(a.n)?,
                    zeros(a.m)?,
                    zeros(a.m)?,
                    zeros(a.m)?,
                    Res::from_slice(&ds(&colc)).map_err(e)?,
                    Res::from_slice(&ds(b)).map_err(e)?,
                    Res::from_slice(&ds(sigma)).map_err(e)?,
                    Res::zeroed(std::mem::size_of::<Step>() as u64).map_err(e)?,
                ],
                batch,
            })
        }

        pub fn write_state(
            &self,
            x: &[f64],
            y: &[f64],
            xavg: &[f64],
            yavg: &[f64],
        ) -> Result<(), PdhgError> {
            let mut job = Job::new();
            let st = &self.state;
            for (r, v) in [
                (&st[0], x),
                (&st[1], x),
                (&st[2], xavg),
                (&st[3], y),
                (&st[4], yavg),
            ] {
                if !v.is_empty() {
                    job.write(r, 0, &ds(v));
                }
            }
            job.submit().map(|_| ()).map_err(PdhgError::Gpu)
        }

        /// Batches of `(iterations, average count)` back to back at one step scale in
        /// one submission, and the device words of `(x, y, x_avg, y_avg)` after each.
        pub fn run(&self, plan: &[(u32, u32)], scale: f64) -> Result<Vec<Words>, PdhgError> {
            let st = &self.state;
            let [x, _, xavg, y, yavg, ..] = st.each_ref();
            let mut job = Job::new();
            let mut reads = Vec::with_capacity(plan.len());
            for &(iters, avg_count) in plan {
                self.encode(&mut job, iters, scale, scale, avg_count);
                reads.push(
                    [(x, self.nx), (y, self.m), (xavg, self.nx), (yavg, self.m)]
                        .map(|(r, n)| job.read(r, 0, (n * 8) as u64)),
                );
            }
            let out = job.submit().map_err(PdhgError::Gpu)?;
            Ok(reads
                .into_iter()
                .map(|batch| batch.map(|i| out.get::<[f32; 2]>(i)))
                .collect())
        }

        /// Put back the iterates of an earlier batch, word for word. Nothing else
        /// carries over between batches: x̄ and Ax are rewritten before they are read,
        /// and the step block is written with every batch.
        pub fn restore(&self, words: &Words) -> Result<(), PdhgError> {
            let st = &self.state;
            let mut job = Job::new();
            for (r, w) in [(&st[0], &words[0]), (&st[3], &words[1]), (&st[2], &words[2]), (&st[4], &words[3])] {
                if !w.is_empty() {
                    job.write(r, 0, w);
                }
            }
            job.submit().map(|_| ()).map_err(PdhgError::Gpu)
        }

        /// Record `iters` iterations.
        fn encode(&self, job: &mut Job, iters: u32, tau: f64, sigma: f64, avg_count: u32) {
            let [n_eq, n_nn, n_soc, n_trust] = self.counts;
            let n_jobs = (n_eq + n_nn + n_soc) as usize;
            let st = &self.state;
            let [x, xbar, xavg, y, yavg, ax, colc, b, sdiag, step] = st.each_ref();
            let [ci, cv] = self.csc.each_ref();
            let [rp, rc, rv] = self.csr.each_ref();
            let s = Step {
                nx: self.nx as u32,
                m: self.m as u32,
                n_eq,
                n_nn,
                n_soc,
                n_trust,
                avg_count,
                tau: ds(&[tau])[0],
                sigma: ds(&[sigma])[0],
                n_long: self.n_long,
                inv: ds(&[1.0 / (f64::from(avg_count) + 1.0)])[0],
            };
            job.write(step, 0, std::slice::from_ref(&s));
            if let Some([hdr, cst]) = &self.batch {
                job.write(hdr, 0, &[iters]);
                job.run(
                    &kernels::PDHG_BATCH,
                    [1, 1, 1],
                    &[
                        hdr.into(),
                        cst.into(),
                        x.into(),
                        xavg.into(),
                        y.into(),
                        yavg.into(),
                        step.into(),
                    ],
                );
            }
            // Without the fused kernel, one dispatch per phase and iteration.
            let split = if self.batch.is_some() { 0 } else { iters };
            for _ in 0..split {
                if self.nx > 0 {
                    let k = &kernels::PDHG_PRIMAL;
                    job.run(
                        k,
                        k.groups(self.nx),
                        &[
                            ci.into(),
                            cv.into(),
                            y.into(),
                            x.into(),
                            colc.into(),
                            xbar.into(),
                            xavg.into(),
                            step.into(),
                        ],
                    );
                }
                if self.m > 0 {
                    let k = &kernels::PDHG_AX;
                    job.run(
                        k,
                        k.groups(self.m),
                        &[
                            rp.into(),
                            rc.into(),
                            rv.into(),
                            xbar.into(),
                            ax.into(),
                            step.into(),
                        ],
                    );
                }
                if self.n_long > 0 {
                    let k = &kernels::PDHG_AX_LONG;
                    job.run(
                        k,
                        [self.n_chunks, 1, 1],
                        &[
                            (&self.chunks).into(),
                            rc.into(),
                            rv.into(),
                            xbar.into(),
                            (&self.lpart).into(),
                        ],
                    );
                    let k = &kernels::PDHG_AX_LONG_SUM;
                    job.run(
                        k,
                        k.groups(self.n_long as usize),
                        &[
                            (&self.longs).into(),
                            (&self.lpart).into(),
                            ax.into(),
                            step.into(),
                        ],
                    );
                }
                if n_jobs > 0 {
                    let k = &kernels::PDHG_DUAL;
                    job.run(
                        k,
                        k.groups(n_jobs),
                        &[
                            y.into(),
                            ax.into(),
                            b.into(),
                            yavg.into(),
                            sdiag.into(),
                            step.into(),
                        ],
                    );
                }
                if n_trust > 0 {
                    let k = &kernels::PDHG_TRUST_PRED;
                    job.run(
                        k,
                        k.groups(n_trust as usize),
                        &[
                            y.into(),
                            ax.into(),
                            b.into(),
                            sdiag.into(),
                            (&self.tpart).into(),
                            step.into(),
                        ],
                    );
                    let k = &kernels::PDHG_TRUST_PROJ;
                    job.run(
                        k,
                        k.groups(n_trust as usize),
                        &[y.into(), yavg.into(), (&self.tpart).into(), step.into()],
                    );
                }
                let k = &kernels::PDHG_TICK;
                job.run(k, [1, 1, 1], &[step.into()]);
            }
        }
    }
}
