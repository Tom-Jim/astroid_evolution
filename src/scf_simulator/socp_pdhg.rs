//! Metal PDHG for the cone program assembled in `socp.rs`.
//!
//! The program is unchanged: a diagonal convex quadratic over a product of an
//! equality (zero) cone, a nonnegative cone, and four-dimensional second-order
//! cones. Column/row equilibration is a diagonal change of variables that
//! preserves each cone and the minimizer. The Chambolle–Pock iteration itself
//! runs on the GPU in double-single arithmetic. A point is accepted when the
//! original, unscaled KKT residuals are at most `KKT_TOL`.

use std::time::Instant;

/// Relative primal residual, dual residual, duality gap, and dual-cone
/// feasibility. This is the loosened PDHG tolerance.
const KKT_TOL: f64 = 1.0e-3;
const CHECK_ITERS: u32 = 48;
const MAX_ITERS: u32 = 40_000;
const MAX_STEP_GROWS: u32 = 8;

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
}

pub struct Solution {
    pub x: Vec<f64>,
    pub iterations: u32,
    pub seconds: f64,
    pub primal: f64,
    pub dual: f64,
    pub gap: f64,
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
    /// `y = E y' / α` after row equilibration and objective scaling.
    e: Vec<f64>,
    alpha: f64,
}

struct Residuals {
    rp: f64,
    rd: f64,
    gap: f64,
    dy: f64,
    rp_inf: f64,
    rd_inf: f64,
}

impl Residuals {
    fn metric(&self) -> f64 {
        // The mass row is one entry of the primal residual. The 2-norm hides it
        // once `b` is long, so the acceptance norm is the largest row violation.
        self.rp_inf.max(self.rd).max(self.gap).max(self.dy)
    }

    fn accepted(&self) -> bool {
        self.rp_inf <= KKT_TOL && self.rd <= KKT_TOL && self.gap <= KKT_TOL && self.dy <= KKT_TOL
    }
}

struct Candidate {
    z: Vec<f64>,
    y: Vec<f64>,
    x: Vec<f64>,
    residuals: Residuals,
}

pub fn solve(problem: Problem) -> Result<Solution, PdhgError> {
    let started = Instant::now();
    let n = problem.p_diag.len();
    let m = problem.b.len();
    if problem.q.len() != n {
        return Err(PdhgError::Failed(
            "PDHG quadratic linear term has the wrong length".into(),
        ));
    }
    if m != problem.n_eq + problem.n_nn + 4 * problem.n_soc || n == 0 || m == 0 {
        return Err(PdhgError::Failed(format!(
            "PDHG cone layout is inconsistent: n={n}, m={m}, eq={}, nn={}, soc={}",
            problem.n_eq, problem.n_nn, problem.n_soc
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
    )?;
    // ‖A‖₂ ≤ ‖A‖_F. The power estimate is tighter; the factor 1.25 covers a
    // short power iteration, and a later step that exceeds it grows `l_used`.
    let l_f = scaled.val.iter().map(|v| v * v).sum::<f64>().sqrt();
    let l_pow = op_norm(&scaled);
    let mut l_used = if l_pow > 0.0 { l_pow * 1.25 } else { l_f };
    if l_f > 0.0 {
        l_used = l_used.min(l_f);
    }
    l_used = l_used.max(1.0e-12);
    let (mut tau, mut sigma) = steps_from(l_used, 1.0);

    let device = gpu::Device::new(
        &scaled,
        &p,
        &q,
        &b,
        problem.n_eq,
        problem.n_nn,
        problem.n_soc,
    )?;
    let zero_z = vec![0.0; n];
    let zero_y = vec![0.0; m];
    let (x0, y0) = recover(&zero_z, &zero_y, &scale);
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
    );
    if r0.accepted() {
        return Ok(Solution {
            x: x0,
            iterations: 0,
            seconds: started.elapsed().as_secs_f64(),
            primal: r0.rp,
            dual: r0.rd,
            gap: r0.gap,
        });
    }

    let mut best = Candidate {
        z: zero_z,
        y: zero_y,
        x: x0,
        residuals: r0,
    };
    let mut avg_count = 0u32;
    let mut iter = 0u32;
    let mut grows = 0u32;
    let mut stall = 0u32;
    let mut prev_z: Option<Vec<f64>> = None;
    let log = std::env::var_os("PDHG_LOG").is_some();
    if log {
        eprintln!("pdhg L={l_used:.3e} (power {l_pow:.3e}, frobenius {l_f:.3e}) τ={tau:.3e}");
    }

    while iter < MAX_ITERS {
        let batch = CHECK_ITERS.min(MAX_ITERS - iter);
        device.step(batch, tau, sigma, avg_count)?;
        iter += batch;
        avg_count += batch;
        let z = device.read_x();
        let y = device.read_y();
        let za = device.read_xavg();
        let ya = device.read_yavg();
        if !(finite(&z) && finite(&y) && finite(&za) && finite(&ya)) {
            grows += 1;
            if grows > MAX_STEP_GROWS {
                break;
            }
            l_used *= 2.0;
            (tau, sigma) = steps_from(l_used, 1.0);
            device.write_state(&best.z, &best.y, &best.z, &best.y);
            avg_count = 1;
            prev_z = None;
            stall = 0;
            continue;
        }
        if let Some(prev) = &prev_z {
            let seen = observed_norm(&scaled, prev, &z);
            if seen > l_used * 1.01 {
                grows += 1;
                if grows > MAX_STEP_GROWS {
                    break;
                }
                l_used = (seen * 1.25).max(l_used);
                (tau, sigma) = steps_from(l_used, 1.0);
                device.write_state(&best.z, &best.y, &best.z, &best.y);
                avg_count = 1;
                prev_z = None;
                stall = 0;
                if log {
                    eprintln!("pdhg step norm {seen:.3e} exceeds the estimate; τ={tau:.3e}");
                }
                continue;
            }
        }
        prev_z = Some(z.clone());
        let before = best.residuals.metric();
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
        );
        if best.residuals.metric() < before * 0.95 {
            stall = 0;
        } else {
            stall += 1;
        }
        if log {
            let r = &best.residuals;
            eprintln!(
                "pdhg iter {iter} best rp={:.3e} rd={:.3e} gap={:.3e} dy={:.3e} inf=({:.3e},{:.3e})",
                r.rp, r.rd, r.gap, r.dy, r.rp_inf, r.rd_inf
            );
        }
        if best.residuals.accepted() {
            break;
        }
        // The running average has to forget the transient. Restarting it
        // leaves the primal-dual state on its trajectory.
        if stall >= 8 {
            device.write_avg(&z, &y);
            avg_count = 1;
            stall = 0;
        }
    }

    let r = &best.residuals;
    if !r.accepted() {
        return Err(PdhgError::Failed(format!(
            "Metal-PDHG stopped after {iter} iterations at row {:.3e}, dual {:.3e}, gap {:.3e}, dual-cone {:.3e} (tolerance {KKT_TOL:.0e})",
            r.rp_inf, r.rd, r.gap, r.dy
        )));
    }
    Ok(Solution {
        x: best.x,
        iterations: iter,
        seconds: started.elapsed().as_secs_f64(),
        primal: r.rp,
        dual: r.rd,
        gap: r.gap,
    })
}

fn steps_from(l_used: f64, balance: f64) -> (f64, f64) {
    let prod = (0.9 / l_used).powi(2);
    let balance = if balance.is_finite() && balance > 0.0 {
        balance.clamp(1.0e-6, 1.0e6)
    } else {
        1.0
    };
    let mut tau = (prod * balance).sqrt();
    let mut sigma = (prod / balance).sqrt();
    // Smaller than the safe step is still contractive. The cap keeps the
    // double-single factor from leaving the f32 exponent range.
    tau = tau.min(10.0).max(1.0e-10);
    sigma = sigma.min(10.0).max(1.0e-10);
    (tau, sigma)
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
) {
    let (x, y_orig) = recover(z, y, scale);
    if !finite(&x) || !finite(&y_orig) {
        return;
    }
    let residuals = residuals(original, p0, q0, b0, &x, &y_orig, n_eq, n_nn, n_soc);
    if residuals.metric() < best.residuals.metric() {
        *best = Candidate {
            z: z.to_vec(),
            y: y.to_vec(),
            x,
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

fn observed_norm(a: &Csr, x0: &[f64], x1: &[f64]) -> f64 {
    let mut dx_norm2 = 0.0;
    let dx: Vec<f64> = x0
        .iter()
        .zip(x1)
        .map(|(a, b)| {
            let d = b - a;
            dx_norm2 += d * d;
            d
        })
        .collect();
    if dx_norm2 < 1.0e-30 {
        return 0.0;
    }
    let mut adx = vec![0.0; a.m];
    spmv(a, &dx, &mut adx);
    adx.iter().map(|v| v * v).sum::<f64>().sqrt() / dx_norm2.sqrt()
}

fn equilibrate(
    a: &mut Csr,
    p: &mut [f64],
    q: &mut [f64],
    b: &mut [f64],
    n_eq: usize,
    n_nn: usize,
    n_soc: usize,
) -> Result<Scale, PdhgError> {
    let mut c = vec![1.0; a.n];
    let mut e = vec![1.0; a.m];
    let nn_end = n_eq + n_nn;
    for _ in 0..10 {
        let mut col_max = vec![0.0_f64; a.n];
        for (&col, &val) in a.col.iter().zip(a.val.iter()) {
            col_max[col as usize] = col_max[col as usize].max(val.abs());
        }
        let mut scol = vec![1.0; a.n];
        for j in 0..a.n {
            let magnitude = col_max[j].max(p[j].max(0.0).sqrt());
            if magnitude > 0.0 {
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
        for i in 0..nn_end {
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
    }
    let mut magnitude = 0.0_f64;
    for v in p.iter().chain(q.iter()) {
        magnitude = magnitude.max(v.abs());
    }
    let alpha = if magnitude > 0.0 {
        1.0 / magnitude
    } else {
        1.0
    };
    for v in p.iter_mut().chain(q.iter_mut()) {
        *v *= alpha;
    }
    let finite_scale = c.iter().chain(e.iter()).all(|v| v.is_finite() && *v > 0.0)
        && alpha.is_finite()
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
    Ok(Scale { c, e, alpha })
}

fn row_max(a: &Csr, i: usize) -> f64 {
    let mut magnitude = 0.0_f64;
    for k in a.row_ptr[i] as usize..a.row_ptr[i + 1] as usize {
        magnitude = magnitude.max(a.val[k].abs());
    }
    magnitude
}

fn row_scale(magnitude: f64) -> f64 {
    if magnitude > 0.0 {
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
) -> Residuals {
    let mut ax = vec![0.0; a.m];
    spmv(a, x, &mut ax);
    let slack: Vec<f64> = b.iter().zip(&ax).map(|(b, ax)| b - ax).collect();
    let proj = primal_project(&slack, n_eq, n_nn, n_soc);
    let mut rp2 = 0.0;
    let mut rp_inf = 0.0_f64;
    for (v, s) in slack.iter().zip(&proj) {
        let d = v - s;
        rp2 += d * d;
        rp_inf = rp_inf.max(d.abs());
    }
    let bnorm = b.iter().map(|v| v * v).sum::<f64>().sqrt();
    let rp = rp2.sqrt() / (1.0 + bnorm);

    let y_proj = dual_project(y, n_eq, n_nn, n_soc);
    let mut dy2 = 0.0;
    for (v, s) in y.iter().zip(&y_proj) {
        let d = v - s;
        dy2 += d * d;
    }
    let ynorm = y.iter().map(|v| v * v).sum::<f64>().sqrt();
    let dy = dy2.sqrt() / (1.0 + ynorm);

    let mut aty = vec![0.0; a.n];
    spmv_t(a, y, &mut aty);
    let mut rd2 = 0.0;
    let mut rd_inf = 0.0_f64;
    let mut quad = 0.0;
    let mut lin = 0.0;
    for i in 0..a.n {
        let g = p[i] * x[i] + q[i] + aty[i];
        rd2 += g * g;
        rd_inf = rd_inf.max(g.abs());
        quad += p[i] * x[i] * x[i];
        lin += q[i] * x[i];
    }
    let qnorm = q.iter().map(|v| v * v).sum::<f64>().sqrt();
    let rd = rd2.sqrt() / (1.0 + qnorm);
    let by = b.iter().zip(y).map(|(b, y)| b * y).sum::<f64>();
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
        rd_inf,
    }
}

fn primal_project(v: &[f64], n_eq: usize, n_nn: usize, n_soc: usize) -> Vec<f64> {
    let mut s = vec![0.0; v.len()];
    let nn_end = n_eq + n_nn;
    for i in n_eq..nn_end {
        s[i] = v[i].max(0.0);
    }
    project_socs(&mut s, v, nn_end, n_soc);
    s
}

fn dual_project(v: &[f64], n_eq: usize, n_nn: usize, n_soc: usize) -> Vec<f64> {
    let mut s = v.to_vec();
    let nn_end = n_eq + n_nn;
    for i in n_eq..nn_end {
        s[i] = s[i].max(0.0);
    }
    project_socs(&mut s, v, nn_end, n_soc);
    s
}

fn project_socs(s: &mut [f64], v: &[f64], nn_end: usize, n_soc: usize) {
    for cone in 0..n_soc {
        let i = nn_end + 4 * cone;
        let t = v[i];
        let x = v[i + 1];
        let y = v[i + 2];
        let z = v[i + 3];
        let nrm = (x * x + y * y + z * z).sqrt();
        if nrm <= t {
            s[i] = t;
            s[i + 1] = x;
            s[i + 2] = y;
            s[i + 3] = z;
        } else if nrm <= -t {
            s[i] = 0.0;
            s[i + 1] = 0.0;
            s[i + 2] = 0.0;
            s[i + 3] = 0.0;
        } else {
            let a = 0.5 * (nrm + t) / nrm;
            s[i] = a * nrm;
            s[i + 1] = a * x;
            s[i + 2] = a * y;
            s[i + 3] = a * z;
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

#[cfg(target_os = "macos")]
mod gpu {
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

    use super::{Csr, PdhgError};

    const SOURCE: &str = concat!(
        include_str!("../rheo_friction/shaders/ds.metal"),
        "\n",
        include_str!("shaders/pdhg.metal"),
    );

    #[repr(C)]
    struct Step {
        nx: u32,
        m: u32,
        n_eq: u32,
        n_nn: u32,
        n_soc: u32,
        avg_count: u32,
        tau_hi: f32,
        tau_lo: f32,
        sigma_hi: f32,
        sigma_lo: f32,
    }

    const _: () = assert!(std::mem::size_of::<Step>() == 40);
    const _: () = assert!(std::mem::align_of::<Step>() == 4);

    type DeviceP = ProtocolObject<dyn MTLDevice>;
    type Queue = ProtocolObject<dyn MTLCommandQueue>;
    type Pipeline = ProtocolObject<dyn MTLComputePipelineState>;
    type Buffer = ProtocolObject<dyn MTLBuffer>;
    type Encoder = ProtocolObject<dyn MTLComputeCommandEncoder>;

    struct Core {
        device: Retained<DeviceP>,
        queue: Retained<Queue>,
        aty: Retained<Pipeline>,
        primal: Retained<Pipeline>,
        ax: Retained<Pipeline>,
        dual: Retained<Pipeline>,
        submit: Mutex<()>,
    }

    unsafe impl Send for Core {}
    unsafe impl Sync for Core {}

    fn core() -> Result<&'static Core, PdhgError> {
        static SLOT: OnceLock<Result<Core, String>> = OnceLock::new();
        match SLOT.get_or_init(init) {
            Ok(core) => Ok(core),
            Err(err) => Err(PdhgError::Gpu(err.clone())),
        }
    }

    fn init() -> Result<Core, String> {
        let gpu = MTLCreateSystemDefaultDevice().ok_or("Metal device unavailable")?;
        let queue = gpu
            .newCommandQueue()
            .ok_or("Metal command queue unavailable")?;
        let options = MTLCompileOptions::new();
        options.setMathMode(MTLMathMode::Safe);
        options.setMathFloatingPointFunctions(MTLMathFloatingPointFunctions::Precise);
        let library = gpu
            .newLibraryWithSource_options_error(&NSString::from_str(SOURCE), Some(&options))
            .map_err(|err| format!("PDHG shader: {err:?}"))?;
        let (aty, primal, ax, dual) = {
            let pipeline = |name: &str| -> Result<Retained<Pipeline>, String> {
                let function = library
                    .newFunctionWithName(&NSString::from_str(name))
                    .ok_or_else(|| format!("{name} kernel missing from the Metal library"))?;
                gpu.newComputePipelineStateWithFunction_error(&function)
                    .map_err(|err| format!("Metal pipeline {name}: {err:?}"))
            };
            (
                pipeline("pdhg_aty")?,
                pipeline("pdhg_primal")?,
                pipeline("pdhg_ax")?,
                pipeline("pdhg_dual")?,
            )
        };
        Ok(Core {
            device: gpu,
            queue,
            aty,
            primal,
            ax,
            dual,
            submit: Mutex::new(()),
        })
    }

    fn bytes_buffer(device: &DeviceP, bytes: &[u8]) -> Result<Retained<Buffer>, PdhgError> {
        let options = MTLResourceOptions::StorageModeShared;
        if bytes.is_empty() {
            return device
                .newBufferWithLength_options(16, options)
                .ok_or_else(|| PdhgError::Gpu("Metal buffer allocation failed".into()));
        }
        let pointer = NonNull::new(bytes.as_ptr() as *mut c_void).unwrap();
        unsafe { device.newBufferWithBytes_length_options(pointer, bytes.len(), options) }
            .ok_or_else(|| PdhgError::Gpu("Metal buffer allocation failed".into()))
    }

    fn u32_buffer(device: &DeviceP, data: &[u32]) -> Result<Retained<Buffer>, PdhgError> {
        let bytes = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
        };
        bytes_buffer(device, bytes)
    }

    fn ds_words(data: &[f64]) -> Result<Vec<f32>, PdhgError> {
        let mut words = Vec::with_capacity(data.len() * 2);
        for &x in data {
            if !x.is_finite() {
                return Err(PdhgError::Failed(
                    "PDHG uploaded a non-finite coefficient".into(),
                ));
            }
            let hi = x as f32;
            words.push(hi);
            words.push((x - f64::from(hi)) as f32);
        }
        Ok(words)
    }

    fn ds_buffer(device: &DeviceP, data: &[f64]) -> Result<Retained<Buffer>, PdhgError> {
        let words = ds_words(data)?;
        let bytes = unsafe {
            std::slice::from_raw_parts(
                words.as_ptr() as *const u8,
                std::mem::size_of_val(words.as_slice()),
            )
        };
        bytes_buffer(device, bytes)
    }

    fn zero_buffer(device: &DeviceP, n: usize) -> Result<Retained<Buffer>, PdhgError> {
        ds_buffer(device, &vec![0.0; n.max(1)])
    }

    fn write_ds(buf: &Buffer, data: &[f64]) {
        let ptr = buf.contents().as_ptr() as *mut f32;
        for (i, &x) in data.iter().enumerate() {
            let hi = x as f32;
            unsafe {
                *ptr.add(2 * i) = hi;
                *ptr.add(2 * i + 1) = (x - f64::from(hi)) as f32;
            }
        }
    }

    fn read_ds(buf: &Buffer, n: usize) -> Vec<f64> {
        let words =
            unsafe { std::slice::from_raw_parts(buf.contents().as_ptr() as *const f32, n * 2) };
        words
            .chunks(2)
            .map(|c| f64::from(c[0]) + f64::from(c[1]))
            .collect()
    }

    struct Csc {
        ptr: Vec<u32>,
        row: Vec<u32>,
        val: Vec<f64>,
    }

    fn transpose(a: &Csr) -> Csc {
        let mut counts = vec![0u32; a.n];
        for &c in &a.col {
            counts[c as usize] += 1;
        }
        let mut ptr = vec![0u32; a.n + 1];
        for j in 0..a.n {
            ptr[j + 1] = ptr[j] + counts[j];
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

    pub struct Device {
        core: &'static Core,
        nx: usize,
        m: usize,
        n_eq: u32,
        n_nn: u32,
        n_soc: u32,
        x: Retained<Buffer>,
        xbar: Retained<Buffer>,
        xavg: Retained<Buffer>,
        y: Retained<Buffer>,
        yavg: Retained<Buffer>,
        aty: Retained<Buffer>,
        ax: Retained<Buffer>,
        q: Retained<Buffer>,
        pdiag: Retained<Buffer>,
        b: Retained<Buffer>,
        csr_ptr: Retained<Buffer>,
        csr_col: Retained<Buffer>,
        csr_val: Retained<Buffer>,
        csc_ptr: Retained<Buffer>,
        csc_row: Retained<Buffer>,
        csc_val: Retained<Buffer>,
    }

    impl Device {
        pub fn new(
            a: &Csr,
            p: &[f64],
            q: &[f64],
            b: &[f64],
            n_eq: usize,
            n_nn: usize,
            n_soc: usize,
        ) -> Result<Self, PdhgError> {
            let core = core()?;
            let csc = transpose(a);
            let device = &core.device;
            Ok(Self {
                core,
                nx: a.n,
                m: a.m,
                n_eq: n_eq as u32,
                n_nn: n_nn as u32,
                n_soc: n_soc as u32,
                x: zero_buffer(device, a.n)?,
                xbar: zero_buffer(device, a.n)?,
                xavg: zero_buffer(device, a.n)?,
                y: zero_buffer(device, a.m)?,
                yavg: zero_buffer(device, a.m)?,
                aty: zero_buffer(device, a.n)?,
                ax: zero_buffer(device, a.m)?,
                q: ds_buffer(device, q)?,
                pdiag: ds_buffer(device, p)?,
                b: ds_buffer(device, b)?,
                csr_ptr: u32_buffer(device, &a.row_ptr)?,
                csr_col: u32_buffer(device, &a.col)?,
                csr_val: ds_buffer(device, &a.val)?,
                csc_ptr: u32_buffer(device, &csc.ptr)?,
                csc_row: u32_buffer(device, &csc.row)?,
                csc_val: ds_buffer(device, &csc.val)?,
            })
        }

        pub fn write_state(&self, x: &[f64], y: &[f64], xavg: &[f64], yavg: &[f64]) {
            write_ds(&self.x, x);
            write_ds(&self.y, y);
            write_ds(&self.xavg, xavg);
            write_ds(&self.yavg, yavg);
        }

        pub fn write_avg(&self, x: &[f64], y: &[f64]) {
            write_ds(&self.xavg, x);
            write_ds(&self.yavg, y);
        }

        pub fn read_x(&self) -> Vec<f64> {
            read_ds(&self.x, self.nx)
        }

        pub fn read_y(&self) -> Vec<f64> {
            read_ds(&self.y, self.m)
        }

        pub fn read_xavg(&self) -> Vec<f64> {
            read_ds(&self.xavg, self.nx)
        }

        pub fn read_yavg(&self) -> Vec<f64> {
            read_ds(&self.yavg, self.m)
        }

        pub fn step(
            &self,
            iters: u32,
            tau: f64,
            sigma: f64,
            mut avg_count: u32,
        ) -> Result<(), PdhgError> {
            let _guard = self
                .core
                .submit
                .lock()
                .unwrap_or_else(|err| err.into_inner());
            let (tau_hi, tau_lo) = split_ds(tau);
            let (sigma_hi, sigma_lo) = split_ds(sigma);
            let command = self
                .core
                .queue
                .commandBuffer()
                .ok_or_else(|| PdhgError::Gpu("Metal command buffer unavailable".into()))?;
            let encoder = command
                .computeCommandEncoder()
                .ok_or_else(|| PdhgError::Gpu("Metal compute encoder unavailable".into()))?;
            for _ in 0..iters {
                let step = Step {
                    nx: self.nx as u32,
                    m: self.m as u32,
                    n_eq: self.n_eq,
                    n_nn: self.n_nn,
                    n_soc: self.n_soc,
                    avg_count,
                    tau_hi,
                    tau_lo,
                    sigma_hi,
                    sigma_lo,
                };
                self.dispatch_aty(&encoder, &step);
                self.dispatch_primal(&encoder, &step);
                self.dispatch_ax(&encoder, &step);
                self.dispatch_dual(&encoder, &step);
                avg_count = avg_count.saturating_add(1);
            }
            encoder.endEncoding();
            command.commit();
            command.waitUntilCompleted();
            if command.status() != MTLCommandBufferStatus::Completed {
                return Err(PdhgError::Gpu(format!(
                    "PDHG kernel failed: {:?}",
                    command.error()
                )));
            }
            Ok(())
        }

        fn dispatch_aty(&self, enc: &Encoder, step: &Step) {
            enc.setComputePipelineState(&self.core.aty);
            set_buf(enc, &self.csc_ptr, 0);
            set_buf(enc, &self.csc_row, 1);
            set_buf(enc, &self.csc_val, 2);
            set_buf(enc, &self.y, 3);
            set_buf(enc, &self.aty, 4);
            set_step(enc, step, 5);
            dispatch(enc, &self.core.aty, self.nx);
        }

        fn dispatch_primal(&self, enc: &Encoder, step: &Step) {
            enc.setComputePipelineState(&self.core.primal);
            set_buf(enc, &self.x, 0);
            set_buf(enc, &self.aty, 1);
            set_buf(enc, &self.q, 2);
            set_buf(enc, &self.pdiag, 3);
            set_buf(enc, &self.xbar, 4);
            set_buf(enc, &self.xavg, 5);
            set_step(enc, step, 6);
            dispatch(enc, &self.core.primal, self.nx);
        }

        fn dispatch_ax(&self, enc: &Encoder, step: &Step) {
            enc.setComputePipelineState(&self.core.ax);
            set_buf(enc, &self.csr_ptr, 0);
            set_buf(enc, &self.csr_col, 1);
            set_buf(enc, &self.csr_val, 2);
            set_buf(enc, &self.xbar, 3);
            set_buf(enc, &self.ax, 4);
            set_step(enc, step, 5);
            dispatch(enc, &self.core.ax, self.m);
        }

        fn dispatch_dual(&self, enc: &Encoder, step: &Step) {
            enc.setComputePipelineState(&self.core.dual);
            set_buf(enc, &self.y, 0);
            set_buf(enc, &self.ax, 1);
            set_buf(enc, &self.b, 2);
            set_buf(enc, &self.yavg, 3);
            set_step(enc, step, 4);
            dispatch(
                enc,
                &self.core.dual,
                (self.n_eq + self.n_nn + self.n_soc) as usize,
            );
        }
    }

    fn split_ds(x: f64) -> (f32, f32) {
        let hi = x as f32;
        (hi, (x - f64::from(hi)) as f32)
    }

    fn set_buf(enc: &Encoder, buf: &Buffer, index: usize) {
        unsafe {
            enc.setBuffer_offset_atIndex(Some(buf.as_ref()), 0, index);
        }
    }

    fn set_step(enc: &Encoder, step: &Step, index: usize) {
        unsafe {
            enc.setBytes_length_atIndex(
                NonNull::from(step).cast::<c_void>(),
                std::mem::size_of::<Step>(),
                index,
            );
        }
    }

    fn dispatch(enc: &Encoder, pipe: &Pipeline, n: usize) {
        if n == 0 {
            return;
        }
        let width = pipe.threadExecutionWidth();
        let group = (pipe.maxTotalThreadsPerThreadgroup().min(width * 8) / width).max(1) * width;
        let groups = n.div_ceil(group);
        enc.dispatchThreadgroups_threadsPerThreadgroup(
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
    }
}

#[cfg(not(target_os = "macos"))]
mod gpu {
    use super::{Csr, PdhgError};

    pub struct Device;

    impl Device {
        pub fn new(
            _: &Csr,
            _: &[f64],
            _: &[f64],
            _: &[f64],
            _: usize,
            _: usize,
            _: usize,
        ) -> Result<Self, PdhgError> {
            Err(PdhgError::Gpu(
                "Metal-PDHG requires the Metal double-single kernels".into(),
            ))
        }

        pub fn write_state(&self, _: &[f64], _: &[f64], _: &[f64], _: &[f64]) {}
        pub fn write_avg(&self, _: &[f64], _: &[f64]) {}
        pub fn read_x(&self) -> Vec<f64> {
            Vec::new()
        }
        pub fn read_y(&self) -> Vec<f64> {
            Vec::new()
        }
        pub fn read_xavg(&self) -> Vec<f64> {
            Vec::new()
        }
        pub fn read_yavg(&self) -> Vec<f64> {
            Vec::new()
        }
        pub fn step(&self, _: u32, _: f64, _: f64, _: u32) -> Result<(), PdhgError> {
            Err(PdhgError::Gpu(
                "Metal-PDHG requires the Metal double-single kernels".into(),
            ))
        }
    }
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;

    fn assert_kkt(sol: &Solution) {
        assert!(
            sol.primal <= KKT_TOL && sol.dual <= KKT_TOL && sol.gap <= KKT_TOL,
            "rp {:.3e} rd {:.3e} gap {:.3e}",
            sol.primal,
            sol.dual,
            sol.gap
        );
    }

    #[test]
    fn bound_quadratic_stops_at_the_wall() {
        // min ½(x−5)² s.t. x ≤ 1. The minimizer is the bound x = 1.
        let sol = solve(Problem {
            p_diag: vec![1.0],
            q: vec![-5.0],
            rows: vec![0],
            cols: vec![0],
            vals: vec![1.0],
            b: vec![1.0],
            n_eq: 0,
            n_nn: 1,
            n_soc: 0,
        })
        .expect("bound QP");
        assert_kkt(&sol);
        assert!((sol.x[0] - 1.0).abs() <= 2.0e-3, "x = {}", sol.x[0]);
    }

    #[test]
    fn equality_quadratic_hits_the_hyperplane() {
        // min ½(x−1)²+½(y−2)² s.t. x+y = 1. The minimizer is (0, 1).
        let sol = solve(Problem {
            p_diag: vec![1.0, 1.0],
            q: vec![-1.0, -2.0],
            rows: vec![0, 0],
            cols: vec![0, 1],
            vals: vec![1.0, 1.0],
            b: vec![1.0],
            n_eq: 1,
            n_nn: 0,
            n_soc: 0,
        })
        .expect("equality QP");
        assert_kkt(&sol);
        assert!(sol.x[0].abs() <= 2.0e-3, "x = {}", sol.x[0]);
        assert!((sol.x[1] - 1.0).abs() <= 2.0e-3, "y = {}", sol.x[1]);
    }

    #[test]
    fn second_order_cone_moves_toward_the_centre() {
        // min ½(x−3)²+½(y−4)²+0.1 t s.t. ‖(x,y)‖ ≤ t.
        // The ray through (3, 4) gives r = 4.9, so (x, y, t) = (2.94, 3.92, 4.9).
        let sol = solve(Problem {
            p_diag: vec![1.0, 1.0, 0.0],
            q: vec![-3.0, -4.0, 0.1],
            rows: vec![0, 1, 2],
            cols: vec![2, 0, 1],
            vals: vec![-1.0, -1.0, -1.0],
            b: vec![0.0, 0.0, 0.0, 0.0],
            n_eq: 0,
            n_nn: 0,
            n_soc: 1,
        })
        .expect("SOC");
        assert_kkt(&sol);
        let expect = [2.94, 3.92, 4.9];
        for (got, want) in sol.x.iter().zip(expect) {
            assert!(
                (got - want).abs() <= 2.0e-3,
                "x = {:?} want {expect:?}",
                sol.x
            );
        }
    }
}
