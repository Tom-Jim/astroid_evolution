//! The convex step of §2.13 and §10.4.3, solved with Metal-PDHG.
//!
//! The boxed program is in metres and kilograms:
//!
//! ```text
//! min  ½ Σ_{i∈A} w_i (d_i − d_pred,i)² + ϱ (Σ t_i + |s_M|)
//! s.t. d_i = 0 on L,  d_i = d_pred,i on D
//!      ‖g × n̂(d)‖ / |g| ≤ sin φ + t,   t ≥ 0          (slope, face centre)
//!      g · n̂(d) ≤ −ε_n |g|                            (anti-overhang)
//!      Dℳ d + s_M = M − ℳ
//!      |d_i| ≤ Δ for i in A
//! ```
//!
//! The solver unknown of a displacement is `x = d / R₀`, so the stored
//! quadratic is `½ w R₀² (x − d_pred/R₀)²`. The mass row is divided by `M`
//! and the slack unknown is the relative mass `σ = s_M / M`, so its column
//! is `±1` and its linear coefficient is `ϱ M`. That product is `ϱ |s_M|`.
//! The trust region is the box `|d_i| ≤ Δ` on the free nodes (§2.13: the
//! linearization error is per node). Slope cones stay on face centres, where
//! §10.3 evaluates the Hessian. Star-shapedness `n · e_r ≥ c_0` is an outer
//! acceptance test, not a row of this program. `n̂(d)` is the first-order
//! expansion from the double-single Metal kernel.

use rayon::prelude::*;

use crate::common::density::CauchyDensity;
use crate::common::math::DVec3;

use super::body::Body;
use super::socp_gpu::{FaceFrame, face_frames};
use crate::common::error::SimError;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// Locked: `d = 0`.
    Locked,
    /// Pure deposition: `d = d_pred`.
    Deposit,
    /// Mobile: free in the trust region.
    Mobile,
}

pub struct SocpInput<'a> {
    pub body: &'a Body,
    pub kind: &'a [Kind],
    /// Predicted displacement per vertex (metres).
    pub d_pred: &'a [f64],
    /// Fidelity weights per vertex (dimensionless, around 1).
    pub weight: &'a [f64],
    /// `g_eff` at the face centres.
    pub gravity: &'a [DVec3],
    /// `sin φ` per face.
    pub sin_phi: &'a [f64],
    /// Whether the slope row of the face is imposed.
    pub slope_row: &'a [bool],
    /// `∂ℳ/∂R_i` (kg per metre).
    pub mass_jac: &'a [f64],
    /// `M − ℳ(R)` (kg).
    pub mass_residual: f64,
    /// Mass `M` for scaling (kg).
    pub mass: f64,
    /// Trust radius `Δ` (metres).
    pub trust: f64,
    pub eps_n: f64,
    pub varrho: f64,
    /// Saddle point of a previous assembly, indexed by vertex and face.
    pub warm: Option<&'a SocpSeed>,
}

#[derive(Clone, Debug)]
pub struct SocpOutput {
    pub d: Vec<f64>,
    pub status: String,
    pub iterations: u32,
    /// `Σ t_f` of the slope slacks (dimensionless).
    pub slope_slack: f64,
    pub mass_slack: f64,
    /// Overhang rows dropped because the face already violates them.
    pub dropped_overhang: usize,
    pub solve_seconds: f64,
    /// Accepted saddle point, ready for the next assembly.
    pub seed: SocpSeed,
    /// Acceptance metric of the uploaded start, before any iteration.
    pub initial_metric: f64,
    /// Quadratic coefficient of each vertex in the dimensionless unknown. Not free: 0.
    pub vertex_p: Vec<f64>,
    /// Linear coefficient of each vertex in that same unknown. Not free: 0.
    pub vertex_q: Vec<f64>,
    /// Linear coefficient of each face's slope slack. No slack: 0.
    pub slope_cost: Vec<f64>,
    /// Linear coefficient of `σ⁺`. The unknown is `s_M / M`, so this is `ϱ M`.
    pub mass_cost: f64,
    /// Unscaled mass-row column of `σ⁺`. The relative-mass unknown makes this `+1`.
    pub mass_col_plus: f64,
    /// Unscaled mass-row column of `σ⁻`.
    pub mass_col_minus: f64,
    pub n_nn: usize,
    pub n_soc: usize,
    /// Trailing trust cone length. The box `|d_i| ≤ Δ` is in the nonnegative block, so this is 0.
    pub n_trust: usize,
}

impl SocpOutput {
    /// No geometric correction: the cone program could not be solved down to the
    /// trust-radius floor, and the step keeps its geometry (§10.4.3).
    pub fn unresolved(nv: usize, nf: usize) -> Self {
        Self {
            d: vec![0.0; nv],
            status: "Unresolved".into(),
            iterations: 0,
            slope_slack: 0.0,
            mass_slack: 0.0,
            dropped_overhang: 0,
            solve_seconds: 0.0,
            seed: SocpSeed::blank(nv, nf),
            initial_metric: f64::INFINITY,
            vertex_p: vec![0.0; nv],
            vertex_q: vec![0.0; nv],
            slope_cost: vec![0.0; nf],
            mass_cost: 0.0,
            mass_col_plus: 0.0,
            mass_col_minus: 0.0,
            n_nn: 0,
            n_soc: 0,
            n_trust: 0,
        }
    }
}

/// Accepted primal-dual point, stored by mesh identity rather than by row number.
///
/// A later step may drop or add mobile vertices and slope faces. Missing
/// identities stay at zero; identities that disappeared are not copied.
#[derive(Clone, Debug)]
pub struct SocpSeed {
    pub d: Vec<f64>,
    pub t: Vec<f64>,
    pub s_plus: f64,
    pub s_minus: f64,
    pub y_mass: f64,
    pub y_t: Vec<f64>,
    pub y_s_plus: f64,
    pub y_s_minus: f64,
    /// Multiplier of `d_i ≤ Δ`. Locked vertices stay at 0.
    pub y_trust_hi: Vec<f64>,
    /// Multiplier of `−d_i ≤ Δ`. Locked vertices stay at 0.
    pub y_trust_lo: Vec<f64>,
    pub y_overhang: Vec<f64>,
    pub y_soc: Vec<[f64; 4]>,
}

impl SocpSeed {
    pub(crate) fn blank(nv: usize, nf: usize) -> Self {
        Self {
            d: vec![0.0; nv],
            t: vec![0.0; nf],
            s_plus: 0.0,
            s_minus: 0.0,
            y_mass: 0.0,
            y_t: vec![0.0; nf],
            y_s_plus: 0.0,
            y_s_minus: 0.0,
            y_trust_hi: vec![0.0; nv],
            y_trust_lo: vec![0.0; nv],
            y_overhang: vec![0.0; nf],
            y_soc: vec![[0.0; 4]; nf],
        }
    }
}

#[derive(Clone, Copy)]
enum ColId {
    Mobile(usize),
    Slack(usize),
    MassPlus,
    MassMinus,
}

#[derive(Clone, Copy)]
enum RowId {
    Mass,
    SlackNonneg(usize),
    MassPlus,
    MassMinus,
    Overhang(usize),
    Soc(usize, u8),
    TrustHi(usize),
    TrustLo(usize),
}

/// `∂ℳ/∂R_i = ω̂_i · Σ_{f ∋ i} n_f ∫_f ρ φ_i dS` (edge-midpoint rule, exact for `ρ` linear).
pub fn mass_jacobian(body: &Body, density: &CauchyDensity) -> Vec<f64> {
    let p = body.positions();
    let geo = body.face_geometry();
    let faces = &body.topo.faces;
    let dirs = &body.topo.dirs;
    let nv = body.topo.n_vertices();
    let nf = faces.len();
    let nthreads = rayon::current_num_threads().max(1);
    let partial: Vec<Vec<f64>> = (0..nthreads)
        .into_par_iter()
        .map(|thread| {
            let mut j = vec![0.0; nv];
            let mut f = thread;
            while f < nf {
                let [a, b, c] = faces[f].map(|x| x as usize);
                let (_, n, area) = geo[f];
                let rho = |u: usize, v: usize| density.rho((p[u] + p[v]) * 0.5);
                let (mab, mac, mbc) = (rho(a, b), rho(a, c), rho(b, c));
                let share = area / 3.0;
                let flux = [
                    share * 0.5 * (mab + mac),
                    share * 0.5 * (mab + mbc),
                    share * 0.5 * (mac + mbc),
                ];
                for (k, v) in [a, b, c].into_iter().enumerate() {
                    j[v] += dirs[v].dot(n) * flux[k];
                }
                f += nthreads;
            }
            j
        })
        .collect();
    partial
        .into_iter()
        .reduce(|mut acc, part| {
            for (a, b) in acc.iter_mut().zip(part) {
                *a += b;
            }
            acc
        })
        .unwrap_or_else(|| vec![0.0; nv])
}

fn require_frame(frames: &[Option<FaceFrame>], f: usize) -> Result<&FaceFrame, SimError> {
    let frame = frames[f]
        .as_ref()
        .ok_or_else(|| SimError::gpu(format!("face {f}: slope frame was not evaluated")))?;
    if !frame.n.is_finite() || (frame.n.length() - 1.0).abs() > 1.0e-4 {
        return Err(SimError::gpu(format!(
            "face {f}: double-single normal is not a finite unit vector (|n| = {})",
            frame.n.length()
        )));
    }
    for (k, column) in frame.nj.iter().enumerate() {
        if !column.is_finite() {
            return Err(SimError::gpu(format!(
                "face {f}: double-single normal sensitivity column {k} is not finite"
            )));
        }
    }
    Ok(frame)
}

#[derive(Default)]
struct Block {
    trip: Vec<(usize, usize, f64)>,
    b: Vec<f64>,
}

impl Block {
    fn row(&mut self, b: f64) -> usize {
        self.b.push(b);
        self.b.len() - 1
    }

    fn put(&mut self, row: usize, col: usize, v: f64) {
        if v != 0.0 {
            self.trip.push((row, col, v));
        }
    }
}

pub fn solve(inp: &SocpInput) -> Result<SocpOutput, SimError> {
    let mut assemble = super::timing::span("socp_assemble");
    let body = inp.body;
    let topo = &body.topo;
    let nv = topo.n_vertices();
    let nf = topo.n_faces();
    let r0 = body.radius.iter().sum::<f64>() / nv as f64;
    let mscale = inp.mass;
    // Unknown numbering: mobile displacements, slope slacks, mass slacks.
    let mut col_of = vec![usize::MAX; nv];
    let mut na = 0usize;
    for i in 0..nv {
        if inp.kind[i] == Kind::Mobile {
            col_of[i] = na;
            na += 1;
        }
    }
    let p = body.positions();
    let ps: Vec<DVec3> = p.iter().map(|x| *x * (1.0 / r0)).collect();
    // Slope cones are the only rows that use n̂ and ∂n̂/∂R.
    let active: Vec<usize> = (0..nf).filter(|&f| inp.slope_row[f]).collect();
    let active_faces: Vec<[u32; 3]> = active.iter().map(|&f| topo.faces[f]).collect();
    assemble.pause();
    let computed = {
        let _span = super::timing::span("socp_face_frames");
        face_frames(&ps, &topo.dirs, &active_faces).map_err(SimError::gpu)?
    };
    assemble.resume();
    if computed.len() != active.len() {
        return Err(SimError::gpu(format!(
            "face-frame kernel returned {} frames for {} slope faces",
            computed.len(),
            active.len()
        )));
    }
    let mut frames = vec![None; nf];
    for (slot, &f) in active.iter().enumerate() {
        frames[f] = Some(computed[slot]);
    }
    // §10.4.3 fixes d = d_pred on the deposition set. The trust ball
    // constrains only the free variables, so this value is not clipped.
    let delta = inp.trust / r0;
    let known: Vec<f64> = (0..nv)
        .map(|i| match inp.kind[i] {
            Kind::Deposit => inp.d_pred[i] / r0,
            _ => 0.0,
        })
        .collect();
    let mut t_col = vec![usize::MAX; nf];
    let mut nt = 0usize;
    for f in 0..nf {
        if !inp.slope_row[f] {
            continue;
        }
        let frame = require_frame(&frames, f)?;
        let gnorm = inp.gravity[f].length();
        let overhang_ok =
            gnorm > 1e-14 && frame.n.dot(inp.gravity[f] * (1.0 / gnorm)) <= -inp.eps_n;
        if overhang_ok
            && topo.faces[f]
                .iter()
                .any(|v| inp.kind[*v as usize] == Kind::Mobile)
        {
            t_col[f] = na + nt;
            nt += 1;
        }
    }
    let sp = na + nt;
    let sm = sp + 1;
    let nx = sm + 1;
    let mut col_ids = vec![ColId::MassPlus; nx];
    for i in 0..nv {
        if col_of[i] != usize::MAX {
            col_ids[col_of[i]] = ColId::Mobile(i);
        }
    }
    for f in 0..nf {
        if t_col[f] != usize::MAX {
            col_ids[t_col[f]] = ColId::Slack(f);
        }
    }
    col_ids[sp] = ColId::MassPlus;
    col_ids[sm] = ColId::MassMinus;

    // §10.4.3: ½ Σ w_i (d_i − d_pred,i)² + ϱ (Σ t_i + |s_M|).
    // x = d/R₀, so the quadratic in x is ½ (w R₀²) x² − (w R₀ d_pred) x.
    // The mass row is divided by M and the unknown is σ = s_M / M, so each
    // equality column is ±1 and the linear coefficient is ϱ M. t is the
    // dimensionless slope excess, so its coefficient stays ϱ.
    // Weights are the given w_i; they are not divided by Σ w.
    let mut p_diag = vec![0.0; nx];
    let mut q = vec![0.0; nx];
    let mut vertex_p = vec![0.0; nv];
    let mut vertex_q = vec![0.0; nv];
    let mut slope_cost = vec![0.0; nf];
    for i in 0..nv {
        if col_of[i] != usize::MAX {
            let w = inp.weight[i];
            let p = w * r0 * r0;
            let qi = -w * r0 * inp.d_pred[i];
            p_diag[col_of[i]] = p;
            q[col_of[i]] = qi;
            {
                vertex_p[i] = p;
                vertex_q[i] = qi;
            }
        }
    }
    for f in 0..nf {
        if t_col[f] != usize::MAX {
            q[t_col[f]] = inp.varrho;
            {
                slope_cost[f] = inp.varrho;
            }
        }
    }
    let mass_penalty = inp.varrho * mscale;
    q[sp] = mass_penalty;
    q[sm] = mass_penalty;

    // Constraints.
    let mut eq = Block::default();
    let mut nn = Block::default();
    let mut soc: Vec<Block> = Vec::new();
    let mut nn_ids: Vec<RowId> = Vec::new();
    let mut soc_faces: Vec<usize> = Vec::new();

    // Mass row.
    let mut known_mass = 0.0;
    for i in 0..nv {
        if col_of[i] == usize::MAX {
            known_mass += inp.mass_jac[i] * known[i] * r0 / mscale;
        }
    }
    let r = eq.row(inp.mass_residual / mscale - known_mass);
    for i in 0..nv {
        if col_of[i] != usize::MAX {
            eq.put(r, col_of[i], inp.mass_jac[i] * r0 / mscale);
        }
    }
    eq.put(r, sp, 1.0);
    eq.put(r, sm, -1.0);

    // t, s± ≥ 0.
    for c in na..nx {
        nn.row(0.0);
        nn.put(nn.b.len() - 1, c, -1.0);
        nn_ids.push(match col_ids[c] {
            ColId::Slack(f) => RowId::SlackNonneg(f),
            ColId::MassPlus => RowId::MassPlus,
            ColId::MassMinus => RowId::MassMinus,
            ColId::Mobile(_) => unreachable!("slack columns are the slope and mass slacks"),
        });
    }

    let mut dropped = 0usize;
    for f in 0..nf {
        let tri = topo.faces[f].map(|x| x as usize);
        if !tri.iter().any(|v| col_of[*v] != usize::MAX) || !inp.slope_row[f] {
            continue;
        }
        let frame = require_frame(&frames, f)?;
        let (n, nj) = (frame.n, frame.nj);
        // Constant part of n̂(d) from the known displacements.
        let mut n0 = n;
        for k in 0..3 {
            if col_of[tri[k]] == usize::MAX {
                n0 = n0 + nj[k] * known[tri[k]];
            }
        }
        let g = inp.gravity[f];
        let gn = g.length();
        if gn <= 1e-14 {
            // A zero effective gravity direction has no defined slope or overhang
            // constraint (§2.8); do not divide by zero or manufacture a NaN row.
            continue;
        }
        let ghat = g * (1.0 / gn);
        // Anti-overhang, only where the current face already satisfies it.
        // A face that already violates the row is dropped before the SOCP (§2.13).
        // n·e_r ≥ c_0 is an outer check, not a row of this program.
        if n.dot(ghat) <= -inp.eps_n {
            let r = nn.row(-inp.eps_n - n0.dot(ghat));
            nn_ids.push(RowId::Overhang(f));
            for k in 0..3 {
                if col_of[tri[k]] != usize::MAX {
                    nn.put(r, col_of[tri[k]], nj[k].dot(ghat));
                }
            }
        } else {
            dropped += 1;
        }
        // Slope SOC.
        if t_col[f] != usize::MAX {
            let mut blk = Block::default();
            let r0w = blk.row(inp.sin_phi[f]);
            blk.put(r0w, t_col[f], -1.0);
            let gxn0 = ghat.cross(n0);
            let comps = [gxn0.x, gxn0.y, gxn0.z];
            for (c, val) in comps.iter().enumerate() {
                let r = blk.row(*val);
                for k in 0..3 {
                    if col_of[tri[k]] != usize::MAX {
                        let s = ghat.cross(nj[k]);
                        let sv = [s.x, s.y, s.z][c];
                        blk.put(r, col_of[tri[k]], -sv);
                    }
                }
            }
            soc.push(blk);
            soc_faces.push(f);
        }
    }

    // |d_i| ≤ Δ on the free nodes, written |x| ≤ Δ/R₀. Two nonnegative rows
    // per mobile vertex. A Euclidean ball of the same radius would also force
    // the patch RMS down by √N, which §2.13's per-node linearization does not ask for.
    for i in 0..nv {
        let j = col_of[i];
        if j == usize::MAX {
            continue;
        }
        let hi = nn.row(delta);
        nn.put(hi, j, 1.0);
        nn_ids.push(RowId::TrustHi(i));
        let lo = nn.row(delta);
        nn.put(lo, j, -1.0);
        nn_ids.push(RowId::TrustLo(i));
    }

    // Assemble. Cone order: one equality, the nonnegative block (slacks,
    // anti-overhang, and the trust box), then one second-order cone of
    // length 4 per slope face.
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    let mut b = Vec::new();
    let mut offset = 0usize;
    let mut push = |blk: &Block,
                    rows: &mut Vec<usize>,
                    cols: &mut Vec<usize>,
                    vals: &mut Vec<f64>,
                    b: &mut Vec<f64>| {
        for (r, c, v) in &blk.trip {
            rows.push(offset + r);
            cols.push(*c);
            vals.push(*v);
        }
        b.extend_from_slice(&blk.b);
        offset += blk.b.len();
    };
    let n_eq = eq.b.len();
    let n_nn = nn.b.len();
    let n_soc = soc.len();
    let n_trust = 0;
    push(&eq, &mut rows, &mut cols, &mut vals, &mut b);
    push(&nn, &mut rows, &mut cols, &mut vals, &mut b);
    for blk in &soc {
        push(blk, &mut rows, &mut cols, &mut vals, &mut b);
    }
    let mut row_ids = Vec::with_capacity(b.len());
    row_ids.push(RowId::Mass);
    row_ids.extend(nn_ids);
    for f in soc_faces {
        for k in 0..4 {
            row_ids.push(RowId::Soc(f, k as u8));
        }
    }
    debug_assert_eq!(row_ids.len(), b.len());
    let warm = inp
        .warm
        .and_then(|seed| scatter_seed(seed, &col_ids, &row_ids, r0, n_eq, n_nn, n_soc, n_trust));
    drop(assemble);
    // Inside the box, a·x cannot leave [-a_span, a_span]. When the mass target
    // sits outside that interval, σ = s_M/M cannot change sign, and ϱ|s_M| is
    // the linear function ϱ M |b − a·x|. Folding it into q removes the mass
    // columns, which otherwise sit ϱ M away from the slope penalty and stall
    // the saddle. The assembled ±1 columns are what the output still reports.
    let b_mass = inp.mass_residual / mscale - known_mass;
    let mut a_coeff = vec![0.0; nx];
    let mut a_span = 0.0_f64;
    for &(r, c, v) in &eq.trip {
        if r == 0 {
            a_coeff[c] = v;
            if c < na {
                a_span += v.abs() * delta;
            }
        }
    }
    let attempt = |fold: bool| -> Result<Option<_>, SimError> {
        // Below that threshold the mass target is met inside the box, and ϱ M is so
        // much larger than any multiplier the slope rows can carry that the exact
        // penalty keeps σ = 0. The mass row is then a hard equality without the σ±
        // columns, which would otherwise sit ϱ M away from the slope penalty.
        if fold && std::env::var_os("PDHG_LOG").is_some() {
            eprintln!("socp fold mass b={b_mass:.3e} span={a_span:.3e} mobile={na}");
        }
        let drop_row: Vec<bool> = row_ids
            .iter()
            .map(|id| match id {
                RowId::Mass => fold,
                RowId::MassPlus | RowId::MassMinus => true,
                _ => false,
            })
            .collect();
        let solved = {
            let _span = super::timing::span("socp_pdhg");
            let mut q_f = q[..sp].to_vec();
            let p_f = p_diag[..sp].to_vec();
            if fold {
                let sign = b_mass.signum();
                for j in 0..na {
                    q_f[j] -= sign * mass_penalty * a_coeff[j];
                }
            }
            let new_row = new_row_of(&drop_row);
            let mut rows_f = Vec::new();
            let mut cols_f = Vec::new();
            let mut vals_f = Vec::new();
            for k in 0..rows.len() {
                if drop_row[rows[k]] || cols[k] >= sp {
                    continue;
                }
                rows_f.push(new_row[rows[k]]);
                cols_f.push(cols[k]);
                vals_f.push(vals[k]);
            }
            let keep = |v: &[f64]| -> Vec<f64> {
                v.iter()
                    .enumerate()
                    .filter(|(i, _)| !drop_row[*i])
                    .map(|(_, v)| *v)
                    .collect()
            };
            let b_f = keep(&b);
            let n_eq_f = if fold { 0 } else { n_eq };
            let n_nn_f = n_nn - 2;
            let warm_f = if fold {
                Some(mass_penalty_start(
                    &p_f, &q_f, &rows_f, &cols_f, &vals_f, &b_f, n_nn_f, n_soc, delta, inp.varrho,
                ))
            } else {
                warm.as_ref().map(|w| super::socp_pdhg::WarmStart {
                    x: w.x[..sp].to_vec(),
                    y: keep(&w.y),
                })
            };
            let problem = super::socp_pdhg::Problem {
                p_diag: p_f,
                q: q_f,
                rows: rows_f,
                cols: cols_f,
                vals: vals_f,
                b: b_f,
                n_eq: n_eq_f,
                n_nn: n_nn_f,
                n_soc,
                n_trust,
                warm: warm_f,
            };
            super::socp_pdhg::solve(problem).map_err(|err| match err {
                super::socp_pdhg::PdhgError::Gpu(detail) => SimError::gpu(detail),
                super::socp_pdhg::PdhgError::Failed(detail) => {
                    SimError::solver(format!("{detail} (mobile {na}, slope cones {nt})"))
                }
            })?
        };
        let _recover = super::timing::span("socp_recover");
        let adot: f64 = (0..na).map(|j| a_coeff[j] * solved.x[j]).sum();
        let sigma = if fold { b_mass - adot } else { 0.0 };
        // The fold is the exact penalty only while σ keeps the sign of the
        // target. A sign change means the box can meet the mass row, which is then
        // solved as the hard equality instead.
        if fold && sigma * b_mass < 0.0 {
            return Ok(None);
        }
        let mut x_full = vec![0.0; nx];
        x_full[..solved.x.len()].copy_from_slice(&solved.x);
        if sigma >= 0.0 {
            x_full[sp] = sigma;
        } else {
            x_full[sm] = -sigma;
        }
        let mut y_full = vec![0.0; row_ids.len()];
        for (i, &r) in new_row_of(&drop_row).iter().enumerate() {
            if r != usize::MAX && r < solved.y.len() {
                y_full[i] = solved.y[r];
            }
        }
        Ok(Some((solved, x_full, y_full, sigma)))
    };
    let first = b_mass.abs() > 1.0e-4;
    let (solved, x_full, y_full, sigma) = match attempt(first)? {
        Some(done) => done,
        None => attempt(false)?.ok_or_else(|| SimError::solver("hard mass row did not solve"))?,
    };
    let mut d = vec![0.0; nv];
    for i in 0..nv {
        d[i] = if col_of[i] != usize::MAX {
            x_full[col_of[i]] * r0
        } else {
            known[i] * r0
        };
    }
    let slope_slack = (0..nf)
        .filter(|f| t_col[*f] != usize::MAX)
        .map(|f| x_full[t_col[f]])
        .sum();
    let seed = pack_seed(nv, nf, r0, &col_ids, &row_ids, &x_full, &y_full);
    let initial_metric = solved.initial_metric;
    Ok(SocpOutput {
        d,
        status: "Solved".to_string(),
        iterations: solved.iterations,
        slope_slack,
        mass_slack: mscale * sigma,
        dropped_overhang: dropped,
        solve_seconds: solved.seconds,
        seed,
        initial_metric,
        vertex_p,
        vertex_q,
        slope_cost,
        mass_cost: mass_penalty,
        mass_col_plus: eq
            .trip
            .iter()
            .find(|t| t.1 == sp)
            .map(|t| t.2)
            .unwrap_or(0.0),
        mass_col_minus: eq
            .trip
            .iter()
            .find(|t| t.1 == sm)
            .map(|t| t.2)
            .unwrap_or(0.0),
        n_nn,
        n_soc,
        n_trust,
    })
}

/// Position of each kept row in the reduced program (`usize::MAX`: dropped).
fn new_row_of(drop_row: &[bool]) -> Vec<usize> {
    let mut n = 0usize;
    drop_row
        .iter()
        .map(|d| {
            if *d {
                usize::MAX
            } else {
                n += 1;
                n - 1
            }
        })
        .collect()
}

/// Box-corner start for a mass penalty that has been folded into `q`.
///
/// Every free node sits on the bound the linear term points at, each slope
/// slack opens just enough to meet its cone, and the bound multiplier cancels
/// `P x + q`. The cone multipliers are `ϱ` and are negligible next to `ϱ M`.
fn mass_penalty_start(
    p: &[f64],
    q: &[f64],
    rows: &[usize],
    cols: &[usize],
    vals: &[f64],
    b: &[f64],
    n_nn: usize,
    n_soc: usize,
    delta: f64,
    varrho: f64,
) -> super::socp_pdhg::WarmStart {
    let n = q.len();
    let m = b.len();
    let mut x = vec![0.0; n];
    for j in 0..n {
        if p[j] > 0.0 && q[j] != 0.0 {
            x[j] = -q[j].signum() * delta;
        }
    }
    let mut ax = vec![0.0; m];
    let accum = |ax: &mut [f64], x: &[f64]| {
        ax.fill(0.0);
        for k in 0..vals.len() {
            ax[rows[k]] += vals[k] * x[cols[k]];
        }
    };
    accum(&mut ax, &x);
    for cone in 0..n_soc {
        let base = n_nn + 4 * cone;
        if base + 3 >= m {
            break;
        }
        let mut t_col = None;
        let mut t_coeff = 0.0;
        for k in 0..vals.len() {
            if rows[k] == base && p[cols[k]] == 0.0 {
                t_col = Some(cols[k]);
                t_coeff = vals[k];
            }
        }
        let Some(t_col) = t_col else { continue };
        if t_coeff == 0.0 {
            continue;
        }
        let s1 = b[base + 1] - ax[base + 1];
        let s2 = b[base + 2] - ax[base + 2];
        let s3 = b[base + 3] - ax[base + 3];
        let spatial = (s1.hypot(s2)).hypot(s3);
        let s0 = b[base] - ax[base];
        // Adding t changes s0 by −t_coeff * t. t_coeff is −1, so s0 increases by t.
        let grow = -t_coeff;
        if grow <= 0.0 {
            continue;
        }
        let t = ((spatial - s0) / grow).max(0.0);
        x[t_col] = t;
    }
    accum(&mut ax, &x);
    let mut y = vec![0.0; m];
    for i in 0..n_nn {
        let mut entry = None;
        let mut count = 0usize;
        for k in 0..vals.len() {
            if rows[k] == i {
                count += 1;
                entry = Some((cols[k], vals[k]));
            }
        }
        let Some((j, aval)) = entry else { continue };
        if count != 1 || aval == 0.0 {
            continue;
        }
        let slack = b[i] - aval * x[j];
        if slack > 1.0e-8 * (1.0 + b[i].abs()) {
            continue;
        }
        let mult = -(p[j] * x[j] + q[j]) / aval;
        if mult >= 0.0 {
            y[i] = mult;
        }
    }
    for cone in 0..n_soc {
        let base = n_nn + 4 * cone;
        if base + 3 >= m {
            break;
        }
        let s0 = b[base] - ax[base];
        let s1 = b[base + 1] - ax[base + 1];
        let s2 = b[base + 2] - ax[base + 2];
        let s3 = b[base + 3] - ax[base + 3];
        let spatial = (s1.hypot(s2)).hypot(s3);
        if s0 < 1.0e-8 || spatial < 1.0e-8 {
            continue;
        }
        // Dual on the boundary, parallel to the tight primal cone, with the
        // leading multiplier equal to the slack's linear cost.
        let scale = varrho / s0;
        y[base] = varrho;
        y[base + 1] = scale * s1;
        y[base + 2] = scale * s2;
        y[base + 3] = scale * s3;
    }
    super::socp_pdhg::WarmStart { x, y }
}

fn scatter_seed(
    seed: &SocpSeed,
    col_ids: &[ColId],
    row_ids: &[RowId],
    r0: f64,
    n_eq: usize,
    n_nn: usize,
    n_soc: usize,
    n_trust: usize,
) -> Option<super::socp_pdhg::WarmStart> {
    let nv = seed.d.len();
    let nf = seed.t.len();
    if seed.y_t.len() != nf
        || seed.y_trust_hi.len() != nv
        || seed.y_trust_lo.len() != nv
        || seed.y_overhang.len() != nf
        || seed.y_soc.len() != nf
        || !r0.is_finite()
        || r0 == 0.0
    {
        return None;
    }
    let mut x = vec![0.0; col_ids.len()];
    for (j, id) in col_ids.iter().enumerate() {
        x[j] = match *id {
            ColId::Mobile(i) => *seed.d.get(i)?,
            ColId::Slack(f) => *seed.t.get(f)?,
            ColId::MassPlus => seed.s_plus,
            ColId::MassMinus => seed.s_minus,
        };
        if matches!(*id, ColId::Mobile(_)) {
            x[j] /= r0;
        }
    }
    let mut y = vec![0.0; row_ids.len()];
    for (i, id) in row_ids.iter().enumerate() {
        y[i] = match *id {
            RowId::Mass => seed.y_mass,
            RowId::SlackNonneg(f) => *seed.y_t.get(f)?,
            RowId::MassPlus => seed.y_s_plus,
            RowId::MassMinus => seed.y_s_minus,
            RowId::Overhang(f) => *seed.y_overhang.get(f)?,
            RowId::Soc(f, k) => *seed.y_soc.get(f)?.get(k as usize)?,
            RowId::TrustHi(v) => *seed.y_trust_hi.get(v)?,
            RowId::TrustLo(v) => *seed.y_trust_lo.get(v)?,
        };
    }
    if !x.iter().chain(y.iter()).all(|v| v.is_finite()) {
        return None;
    }
    y = super::socp_pdhg::project_dual(&y, n_eq, n_nn, n_soc, n_trust);
    Some(super::socp_pdhg::WarmStart { x, y })
}

fn pack_seed(
    nv: usize,
    nf: usize,
    r0: f64,
    col_ids: &[ColId],
    row_ids: &[RowId],
    x: &[f64],
    y: &[f64],
) -> SocpSeed {
    let mut seed = SocpSeed::blank(nv, nf);
    for (j, id) in col_ids.iter().enumerate() {
        let value = x.get(j).copied().unwrap_or(0.0);
        match *id {
            ColId::Mobile(i) if i < nv => seed.d[i] = value * r0,
            ColId::Slack(f) if f < nf => seed.t[f] = value,
            ColId::MassPlus => seed.s_plus = value,
            ColId::MassMinus => seed.s_minus = value,
            _ => {}
        }
    }
    for (i, id) in row_ids.iter().enumerate() {
        let value = y.get(i).copied().unwrap_or(0.0);
        match *id {
            RowId::Mass => seed.y_mass = value,
            RowId::SlackNonneg(f) if f < nf => seed.y_t[f] = value,
            RowId::MassPlus => seed.y_s_plus = value,
            RowId::MassMinus => seed.y_s_minus = value,
            RowId::Overhang(f) if f < nf => seed.y_overhang[f] = value,
            RowId::Soc(f, k) if f < nf => seed.y_soc[f][k as usize] = value,
            RowId::TrustHi(v) if v < nv => seed.y_trust_hi[v] = value,
            RowId::TrustLo(v) if v < nv => seed.y_trust_lo[v] = value,
            _ => {}
        }
    }
    seed
}
