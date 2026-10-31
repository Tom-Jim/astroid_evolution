//! The convex step of §2.13 and §10.4.3, solved by the interior-point method (`socp_ipm`).
//!
//! The boxed program is in metres and kilograms:
//!
//! ```text
//! min  ½ Σ_{i∈A} w_i (d_i − d_pred,i)² + ϱ (Σ t_i + |s_M|)
//! s.t. d_i = 0 on L,  d_i = d_pred,i on D
//!      ‖g × n̂(d)‖ / |g| ≤ sin φ + t,   t ≥ 0          (slope, face centre)
//!      g · n̂(d) ≤ −ε_n |g|                            (anti-overhang)
//!      Dℳ d + s_M = M − ℳ
//!      |d_i| ≤ Δ, lower_i ≤ d_i ≤ upper_i for i in A (clearance, loose mass)
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
//! expansion of `face_frame`.

use rayon::prelude::*;

use crate::common::density::CauchyDensity;
use crate::common::math::DVec3;

use super::body::Body;
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
    /// Bounds on `d_i` (metres) per vertex inside the box `|d_i| ≤ Δ`, or empty: the
    /// clearance rows `−θ_g t⁻ ≤ d ≤ θ_g t⁺` of (G4) and the loose-mass bound of a
    /// promoted vertex (§2.8c). `±∞` where there is none.
    pub lower: &'a [f64],
    pub upper: &'a [f64],
    pub eps_n: f64,
    pub varrho: f64,
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
}

impl SocpOutput {
    /// No geometric correction: the cone program could not be solved down to the
    /// trust-radius floor, and the step keeps its geometry (§10.4.3).
    pub fn unresolved(nv: usize) -> Self {
        Self {
            d: vec![0.0; nv],
            status: "Unresolved".into(),
            iterations: 0,
            slope_slack: 0.0,
            mass_slack: 0.0,
            dropped_overhang: 0,
            solve_seconds: 0.0,
        }
    }
}

/// `(Dℳ)_i = ∂ℳ/∂d_i = N_i · Σ_{f ∋ i} n_f ∫_f ρ φ_i dS` for the normal increment
/// `X_i ← X_i + d_i N_i` (G4; edge-midpoint rule, exact for `ρ` linear, and
/// `ρ |A_i|` for constant `ρ`).
pub fn mass_jacobian(body: &Body, density: &CauchyDensity) -> Vec<f64> {
    let p = body.positions();
    let geo = body.face_geometry();
    let faces = &body.topo.faces;
    let normals = body.vertex_normals();
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
                    j[v] += normals[v].dot(n) * flux[k];
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

/// Unit normal `n̂` of a face and the columns `∂n̂/∂x_k`, `x_k = d_k / R₀`, for the
/// normal increments of its three vertices.
#[derive(Clone, Copy, Debug)]
pub struct FaceFrame {
    pub n: DVec3,
    pub nj: [DVec3; 3],
}

/// `u = (b − a) × (c − a)` has `∂u/∂X_a · v = v × (b − c)` (cyclically), and
/// `∂n̂ = (I − n̂ n̂ᵀ) ∂u / |u|`. `ps` are the vertices over `R₀`.
pub fn face_frame(ps: &[DVec3], normals: &[DVec3], tri: [u32; 3]) -> FaceFrame {
    let [a, b, c] = tri.map(|v| ps[v as usize]);
    let u = (b - a).cross(c - a);
    let len = u.length().max(1e-300);
    let n = u / len;
    let opposite = [b - c, c - a, a - b];
    let nj = std::array::from_fn(|k| {
        let du = normals[tri[k] as usize].cross(opposite[k]) / len;
        du - n * n.dot(du)
    });
    FaceFrame { n, nj }
}

fn require_frame(frames: &[Option<FaceFrame>], f: usize) -> Result<&FaceFrame, SimError> {
    frames[f]
        .as_ref()
        .ok_or_else(|| SimError::solver(format!("face {f}: slope frame was not evaluated")))
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
    let assemble = super::timing::span("socp_assemble");
    let body = inp.body;
    let topo = &body.topo;
    let nv = topo.n_vertices();
    let nf = topo.n_faces();
    let r0 = body.r_mean();
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
    // The slope cones and the anti-overhang rows use n̂ and ∂n̂/∂R. Every face with a
    // free vertex carries the anti-overhang row (§2.8c), so its frame is needed even
    // when it has no slope row.
    let has_free = |f: usize| {
        topo.faces[f]
            .iter()
            .any(|v| inp.kind[*v as usize] == Kind::Mobile)
    };
    let active: Vec<usize> = (0..nf)
        .filter(|&f| inp.slope_row[f] || has_free(f))
        .collect();
    let normals = body.vertex_normals();
    let computed: Vec<FaceFrame> = active
        .par_iter()
        .map(|&f| face_frame(&ps, &normals, topo.faces[f]))
        .collect();
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
        if overhang_ok && has_free(f) {
            t_col[f] = na + nt;
            nt += 1;
        }
    }
    let sp = na + nt;
    // Nothing is free: the fixed deposition is the step, and the mass row books its rest in s_M.
    if sp == 0 {
        let d: Vec<f64> = known.iter().map(|k| k * r0).collect();
        let booked: f64 = d.iter().zip(inp.mass_jac).map(|(d, j)| d * j).sum();
        return Ok(SocpOutput {
            d,
            status: "Solved".into(),
            iterations: 0,
            slope_slack: 0.0,
            mass_slack: inp.mass_residual - booked,
            dropped_overhang: 0,
            solve_seconds: 0.0,
        });
    }
    let sm = sp + 1;
    let nx = sm + 1;

    // §10.4.3: ½ Σ w_i (d_i − d_pred,i)² + ϱ (Σ t_i + |s_M|).
    // x = d/R₀, so the quadratic in x is ½ (w R₀²) x² − (w R₀ d_pred) x.
    // The mass row is divided by M and the unknown is σ = s_M / M, so each
    // equality column is ±1 and the linear coefficient is ϱ M. t is the
    // dimensionless slope excess, so its coefficient stays ϱ.
    // Weights are the given w_i; they are not divided by Σ w.
    let mut p_diag = vec![0.0; nx];
    let mut q = vec![0.0; nx];
    for i in 0..nv {
        if col_of[i] != usize::MAX {
            let w = inp.weight[i];
            p_diag[col_of[i]] = w * r0 * r0;
            q[col_of[i]] = -w * r0 * inp.d_pred[i];
        }
    }
    for f in 0..nf {
        if t_col[f] != usize::MAX {
            q[t_col[f]] = inp.varrho;
        }
    }
    let mass_penalty = inp.varrho * mscale;
    q[sp] = mass_penalty;
    q[sm] = mass_penalty;

    // Constraints.
    let mut eq = Block::default();
    let mut nn = Block::default();
    let mut soc: Vec<Block> = Vec::new();

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

    // t, s± ≥ 0: rows nt and nt + 1 of this block are the mass slacks.
    for c in na..nx {
        nn.row(0.0);
        nn.put(nn.b.len() - 1, c, -1.0);
    }

    let mut dropped = 0usize;
    for f in 0..nf {
        let tri = topo.faces[f].map(|x| x as usize);
        if !tri.iter().any(|v| col_of[*v] != usize::MAX) {
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
        }
    }

    // |d_i| ≤ Δ on the free nodes, written |x| ≤ Δ/R₀. Two nonnegative rows
    // per mobile vertex. A Euclidean ball of the same radius would also force
    // the patch RMS down by √N, which §2.13's per-node linearization does not ask for.
    // `x_box[j]` keeps the interval of free column j for the fold test below.
    let mut x_box = vec![(0.0, 0.0); na];
    for i in 0..nv {
        let j = col_of[i];
        if j == usize::MAX {
            continue;
        }
        let bound = |b: Option<&f64>, sign: f64| match b {
            Some(v) if v.is_finite() => (sign * v / r0).clamp(-delta, delta),
            _ => delta,
        };
        let (u, l) = (bound(inp.upper.get(i), 1.0), bound(inp.lower.get(i), -1.0));
        x_box[j] = (-l, u);
        let hi = nn.row(u);
        nn.put(hi, j, 1.0);
        let lo = nn.row(l);
        nn.put(lo, j, -1.0);
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
    push(&eq, &mut rows, &mut cols, &mut vals, &mut b);
    push(&nn, &mut rows, &mut cols, &mut vals, &mut b);
    for blk in &soc {
        push(blk, &mut rows, &mut cols, &mut vals, &mut b);
    }
    drop(assemble);
    // Inside the bounds, a·x cannot leave [reach.0, reach.1]. When the mass target
    // sits outside that interval, σ = s_M/M cannot change sign, and ϱ|s_M| is
    // the linear function ϱ M |b − a·x|. Folding it into q removes the mass
    // columns, which otherwise sit ϱ M away from the slope penalty and stall
    // the saddle. The assembled ±1 columns are what the output still reports.
    // A reachable target keeps the hard equality: the exact penalty holds σ = 0.
    let b_mass = inp.mass_residual / mscale - known_mass;
    let mut a_coeff = vec![0.0; nx];
    let mut reach = (0.0_f64, 0.0_f64);
    for &(r, c, v) in &eq.trip {
        if r == 0 {
            a_coeff[c] = v;
            if c < na {
                let (lo, hi) = x_box[c];
                reach.0 += (v * lo).min(v * hi);
                reach.1 += (v * lo).max(v * hi);
            }
        }
    }
    // The σ± columns never reach the solver: folded, ϱ M |b − a·x| is in q; otherwise
    // the mass row is a hard equality. Their nonnegativity rows (nt and nt + 1 of the
    // nonnegative block) go with them, and the mass row itself when folded.
    let mass_slack_rows = [n_eq + nt, n_eq + nt + 1];
    let attempt = |fold: bool| -> Result<Option<(super::socp_ipm::Solution, f64)>, SimError> {
        let drop_row: Vec<bool> = (0..b.len())
            .map(|r| (fold && r == 0) || mass_slack_rows.contains(&r))
            .collect();
        let solved = {
            let _span = super::timing::span("socp_ipm");
            let mut q_f = q[..sp].to_vec();
            if fold {
                let sign = b_mass.signum();
                for j in 0..na {
                    q_f[j] -= sign * mass_penalty * a_coeff[j];
                }
            }
            let new_row = new_row_of(&drop_row);
            let (mut rows_f, mut cols_f, mut vals_f) = (Vec::new(), Vec::new(), Vec::new());
            for k in 0..rows.len() {
                if drop_row[rows[k]] || cols[k] >= sp {
                    continue;
                }
                rows_f.push(new_row[rows[k]]);
                cols_f.push(cols[k]);
                vals_f.push(vals[k]);
            }
            let b_f = b.iter().zip(&drop_row).filter(|(_, d)| !**d).map(|(v, _)| *v).collect();
            super::socp_ipm::solve(super::socp_ipm::Problem {
                p_diag: p_diag[..sp].to_vec(),
                q: q_f,
                rows: rows_f,
                cols: cols_f,
                vals: vals_f,
                b: b_f,
                n_eq: if fold { 0 } else { n_eq },
                n_nn: n_nn - 2,
                n_soc,
            })
            .map_err(|err| SimError::solver(format!("{} (mobile {na}, slope cones {nt})", err.detail)))?
        };
        let adot: f64 = (0..na).map(|j| a_coeff[j] * solved.x[j]).sum();
        let sigma = if fold { b_mass - adot } else { 0.0 };
        // The fold is the exact penalty only while σ keeps the sign of the
        // target. A sign change means the box can meet the mass row, which is then
        // solved as the hard equality instead.
        if fold && sigma * b_mass < 0.0 {
            return Ok(None);
        }
        Ok(Some((solved, sigma)))
    };
    let first = b_mass < reach.0 || b_mass > reach.1;
    let (solved, sigma) = match attempt(first)? {
        Some(done) => done,
        None => attempt(false)?.ok_or_else(|| SimError::solver("hard mass row did not solve"))?,
    };
    let mut d = vec![0.0; nv];
    for i in 0..nv {
        d[i] = if col_of[i] != usize::MAX {
            solved.x[col_of[i]] * r0
        } else {
            known[i] * r0
        };
    }
    let slope_slack = (0..nf)
        .filter(|f| t_col[*f] != usize::MAX)
        .map(|f| solved.x[t_col[f]])
        .sum();
    Ok(SocpOutput {
        d,
        status: solved.status,
        iterations: solved.iterations,
        slope_slack,
        mass_slack: mscale * sigma,
        dropped_overhang: dropped,
        solve_seconds: solved.seconds,
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
