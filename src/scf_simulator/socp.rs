//! The convex step problem of §10.4.3, solved with Clarabel.
//!
//! The vertex displacements `d_i` along the fixed directions `ω̂_i` are the unknowns of the
//! mobile set `A`; vertices of the locked set `L` do not move and those of the pure-deposition
//! set `D` move by their prediction. The problem is posed in lengths scaled by `R₀` and masses
//! scaled by `M`:
//!
//! ```text
//! min ½ Σ_A w_i (d_i − d_pred,i)² + ϱ (Σ_f t_f + s⁺ + s⁻)
//! s.t. Σ J_i d_i + s⁺ − s⁻ = M − ℳ(R)                         (linearised mass)
//!      ‖g_f × n̂_f(d)‖ / |g_f| ≤ sin φ_f + t_f,  t_f ≥ 0          (slope SOC, face based)
//!      g_f · n̂_f(d) ≤ −ε_n |g_f|                                  (no overhang)
//!      n̂_f(d) · ê_f ≥ c₀                                          (linearised star-shape)
//!      |d_i| ≤ Δ  (mobile only)                                   (trust region)
//! ```
//!
//! `n̂_f(d)` is the first-order expansion of the face normal. The nonlinear acceptance test
//! (valid mesh, mass residual) belongs to the caller, which also shrinks `Δ` on failure.

use clarabel::algebra::CscMatrix;
use clarabel::solver::{
    DefaultSettingsBuilder, DefaultSolver, IPSolver, SolverStatus, SupportedConeT,
};

use crate::common::density::CauchyDensity;
use crate::common::math::DVec3;

use super::body::Body;
use super::error::ScfError;

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
    pub c0: f64,
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

/// `∂ℳ/∂R_i = ω̂_i · Σ_{f ∋ i} n_f ∫_f ρ φ_i dS` (edge-midpoint rule, exact for `ρ` linear).
pub fn mass_jacobian(body: &Body, density: &CauchyDensity) -> Vec<f64> {
    let p = body.positions();
    let geo = body.face_geometry();
    let mut j = vec![0.0; body.topo.n_vertices()];
    for (f, t) in body.topo.faces.iter().enumerate() {
        let [a, b, c] = t.map(|x| x as usize);
        let (_, n, area) = geo[f];
        let mid = |u: usize, v: usize| density.rho((p[u] + p[v]) * 0.5);
        let (mab, mac, mbc) = (mid(a, b), mid(a, c), mid(b, c));
        let share = area / 3.0;
        let flux = [
            share * 0.5 * (mab + mac),
            share * 0.5 * (mab + mbc),
            share * 0.5 * (mac + mbc),
        ];
        for (k, v) in [a, b, c].into_iter().enumerate() {
            j[v] += body.topo.dirs[v].dot(n) * flux[k];
        }
    }
    j
}

/// `N_j = (I − n nᵀ)(∂A/∂R_j)/|A|` for the three vertices of the face (columns of `∂n̂/∂R`).
pub fn normal_sensitivity(pa: DVec3, pb: DVec3, pc: DVec3, w: [DVec3; 3]) -> (DVec3, [DVec3; 3]) {
    let area_vec = (pb - pa).cross(pc - pa);
    let len = area_vec.length();
    let n = area_vec * (1.0 / len);
    let da = [
        w[0].cross(pb - pc),
        w[1].cross(pc - pa),
        w[2].cross(pa - pb),
    ];
    let proj = |v: DVec3| (v - n * v.dot(n)) * (1.0 / len);
    (n, [proj(da[0]), proj(da[1]), proj(da[2])])
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

pub fn solve(inp: &SocpInput) -> Result<SocpOutput, ScfError> {
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
    // Known (scaled) displacement of non-mobile vertices.
    let known: Vec<f64> = (0..nv)
        .map(|i| match inp.kind[i] {
            Kind::Deposit => inp.d_pred[i] / r0,
            _ => 0.0,
        })
        .collect();
    let mut t_col = vec![usize::MAX; nf];
    let mut nt = 0usize;
    for f in 0..nf {
        if inp.slope_row[f] && topo.faces[f].iter().any(|v| inp.kind[*v as usize] == Kind::Mobile) {
            t_col[f] = na + nt;
            nt += 1;
        }
    }
    let sp = na + nt;
    let sm = sp + 1;
    let nx = sm + 1;

    // Objective.
    let mut p_diag = vec![0.0; nx];
    let mut q = vec![0.0; nx];
    let wsum: f64 = (0..nv).filter(|i| inp.kind[*i] == Kind::Mobile).map(|i| inp.weight[i]).sum();
    for i in 0..nv {
        if col_of[i] != usize::MAX {
            let w = inp.weight[i] / wsum.max(1e-300);
            p_diag[col_of[i]] = w;
            q[col_of[i]] = -w * inp.d_pred[i] / r0;
        }
    }
    for f in 0..nf {
        if t_col[f] != usize::MAX {
            q[t_col[f]] = inp.varrho / nf as f64;
        }
    }
    q[sp] = inp.varrho;
    q[sm] = inp.varrho;

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

    // t, s± ≥ 0.
    for c in (na..nx).collect::<Vec<_>>() {
        let r = nn.row(0.0);
        nn.put(r, c, -1.0);
    }
    // Trust region (box on the mobile displacements).
    let delta = inp.trust / r0;
    for i in 0..nv {
        if col_of[i] != usize::MAX {
            let r1 = nn.row(delta);
            nn.put(r1, col_of[i], 1.0);
            let r2 = nn.row(delta);
            nn.put(r2, col_of[i], -1.0);
        }
    }

    let mut dropped = 0usize;
    for f in 0..nf {
        let tri = topo.faces[f].map(|x| x as usize);
        if !tri.iter().any(|v| col_of[*v] != usize::MAX) {
            continue;
        }
        let w = tri.map(|v| topo.dirs[v]);
        let (n, nj) = normal_sensitivity(ps[tri[0]], ps[tri[1]], ps[tri[2]], w);
        // Constant part of n̂(d) from the known displacements.
        let mut n0 = n;
        for k in 0..3 {
            if col_of[tri[k]] == usize::MAX {
                n0 = n0 + nj[k] * known[tri[k]];
            }
        }
        let g = inp.gravity[f];
        let gn = g.length();
        let ghat = g * (1.0 / gn);
        let e_f = topo.face_dirs[f];

        // Star-shape: n̂·ê ≥ c0  ->  −Σ (N_j·ê) d_j ≤ n0·ê − c0.
        {
            let r = nn.row(n0.dot(e_f) - inp.c0);
            for k in 0..3 {
                if col_of[tri[k]] != usize::MAX {
                    nn.put(r, col_of[tri[k]], -nj[k].dot(e_f));
                }
            }
        }
        // No overhang: ĝ·n̂ ≤ −ε_n, imposed only where the face is not already violating it.
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

    // Assemble.
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    let mut b = Vec::new();
    let mut cones = Vec::new();
    let mut offset = 0usize;
    let mut push = |blk: &Block, rows: &mut Vec<usize>, cols: &mut Vec<usize>, vals: &mut Vec<f64>, b: &mut Vec<f64>| {
        for (r, c, v) in &blk.trip {
            rows.push(offset + r);
            cols.push(*c);
            vals.push(*v);
        }
        b.extend_from_slice(&blk.b);
        offset += blk.b.len();
    };
    push(&eq, &mut rows, &mut cols, &mut vals, &mut b);
    cones.push(SupportedConeT::ZeroConeT(eq.b.len()));
    push(&nn, &mut rows, &mut cols, &mut vals, &mut b);
    cones.push(SupportedConeT::NonnegativeConeT(nn.b.len()));
    for blk in &soc {
        push(blk, &mut rows, &mut cols, &mut vals, &mut b);
        cones.push(SupportedConeT::SecondOrderConeT(4));
    }
    let m = b.len();
    let a_mat = CscMatrix::new_from_triplets(m, nx, rows, cols, vals);
    let p_idx: Vec<usize> = (0..nx).collect();
    let p_mat = CscMatrix::new_from_triplets(nx, nx, p_idx.clone(), p_idx, p_diag);
    let settings = DefaultSettingsBuilder::default()
        .verbose(false)
        .max_iter(200)
        .build()
        .map_err(|e| ScfError::solver(format!("Clarabel settings: {e}")))?;
    let mut solver = DefaultSolver::new(&p_mat, &q, &a_mat, &b, &cones, settings)
        .map_err(|e| ScfError::solver(format!("Clarabel setup: {e}")))?;
    solver.solve();
    let sol = &solver.solution;
    let status = format!("{:?}", sol.status);
    if !matches!(sol.status, SolverStatus::Solved | SolverStatus::AlmostSolved) {
        return Err(ScfError::solver(format!("step SOCP ended with status {status}")));
    }
    let mut d = vec![0.0; nv];
    for i in 0..nv {
        d[i] = if col_of[i] != usize::MAX {
            sol.x[col_of[i]] * r0
        } else {
            known[i] * r0
        };
    }
    let slope_slack = (0..nf)
        .filter(|f| t_col[*f] != usize::MAX)
        .map(|f| sol.x[t_col[f]])
        .sum();
    Ok(SocpOutput {
        d,
        status,
        iterations: sol.iterations,
        slope_slack,
        mass_slack: sol.x[sp] + sol.x[sm],
        dropped_overhang: dropped,
        solve_seconds: sol.solve_time,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use super::super::body::Topology;

    fn max_slope(body: &Body, g: &[DVec3]) -> f64 {
        body.face_geometry()
            .iter()
            .zip(g)
            .map(|((_, n, _), g)| g.try_normalize().unwrap().cross(*n).length())
            .fold(0.0, f64::max)
    }

    #[test]
    fn spike_is_relaxed_with_mass_conserved() {
        let topo = Arc::new(Topology::icosphere(2));
        let nv = topo.n_vertices();
        let mut radius = vec![1.0; nv];
        radius[0] = 1.25;
        let body = Body::new(topo.clone(), radius, 1000.0, vec![], vec![0.0; nv]).unwrap();
        let dens = body.density().unwrap();
        let geo = body.face_geometry();
        let gravity: Vec<DVec3> = geo.iter().map(|(c, _, _)| c.try_normalize().unwrap() * -9.8).collect();
        let nf = topo.n_faces();
        let jac = mass_jacobian(&body, &dens);
        let inp = SocpInput {
            body: &body,
            kind: &vec![Kind::Mobile; nv],
            d_pred: &vec![0.0; nv],
            weight: &vec![1.0; nv],
            gravity: &gravity,
            sin_phi: &vec![0.35; nf],
            slope_row: &vec![true; nf],
            mass_jac: &jac,
            mass_residual: 0.0,
            mass: 4.0e3,
            trust: 0.1,
            c0: 0.3,
            eps_n: 0.01,
            varrho: 1e3,
        };
        let out = solve(&inp).unwrap();
        let mut next = body.clone();
        for i in 0..nv {
            next.radius[i] += out.d[i];
        }
        let before = max_slope(&body, &gravity);
        let after = max_slope(&next, &gravity);
        assert!(out.d[0] < -0.01, "spike moved by {}", out.d[0]);
        assert!(after < before, "{before} -> {after}");
        let dm: f64 = (0..nv).map(|i| jac[i] * out.d[i]).sum();
        assert!(dm.abs() < 1e-6 * 4.0e3 * 3.0, "mass change {dm}");
    }
}
