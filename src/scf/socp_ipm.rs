//! The cone program of §10.0.8 by the interior-point method (Clarabel, a primal–dual
//! homogeneous-embedding conic solver with a sparse quasi-definite LDLᵀ).
//!
//! The program is `min ½ xᵀ P x + qᵀ x` subject to `A x + s = b`, `s ∈ K`, with `P`
//! diagonal and `K` the zero cone (`n_eq` rows), the nonnegative cone (`n_nn` rows) and
//! `n_soc` second-order cones of length 4, in that order.

use clarabel::algebra::CscMatrix;
use clarabel::solver::{
    DefaultSettings, DefaultSolver, IPSolver, NonnegativeConeT, SecondOrderConeT, SolverStatus,
    SupportedConeT, ZeroConeT,
};

use crate::common::error::SimError;

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
    /// `Solved`, or `AlmostSolved` at the reduced tolerances.
    pub status: String,
    pub iterations: u32,
    pub seconds: f64,
}

/// Iteration cap: an interior-point method converges in tens of iterations; reaching
/// the cap is a failed solve, which the outer level answers by halving `Δ` (§10.0.8).
const MAX_ITER: u32 = 200;

pub fn solve(problem: Problem) -> Result<Solution, SimError> {
    let n = problem.p_diag.len();
    let m = problem.b.len();
    if problem.q.len() != n || m != problem.n_eq + problem.n_nn + 4 * problem.n_soc || n == 0 {
        return Err(SimError::solver(format!(
            "cone layout is inconsistent: n={n}, m={m}, eq={}, nn={}, soc={}",
            problem.n_eq, problem.n_nn, problem.n_soc
        )));
    }
    // Scaling the objective leaves the minimizer unchanged. The folded mass penalty
    // puts linear costs of order ϱM against a quadratic of order wR₀²; at unit scale
    // the interior point no longer mistakes that spread for an unbounded direction.
    let scale = problem.q.iter().chain(&problem.p_diag).fold(0.0_f64, |m, v| m.max(v.abs())).max(1e-300);
    let p_diag: Vec<f64> = problem.p_diag.iter().map(|v| v / scale).collect();
    let q: Vec<f64> = problem.q.iter().map(|v| v / scale).collect();
    let diag: Vec<usize> = (0..n).collect();
    let p = CscMatrix::new_from_triplets(n, n, diag.clone(), diag, p_diag);
    let a = CscMatrix::new_from_triplets(m, n, problem.rows, problem.cols, problem.vals);
    let mut cones: Vec<SupportedConeT<f64>> = Vec::with_capacity(2 + problem.n_soc);
    if problem.n_eq > 0 {
        cones.push(ZeroConeT(problem.n_eq));
    }
    if problem.n_nn > 0 {
        cones.push(NonnegativeConeT(problem.n_nn));
    }
    cones.extend((0..problem.n_soc).map(|_| SecondOrderConeT(4)));
    let settings = DefaultSettings {
        verbose: false,
        max_iter: MAX_ITER,
        ..DefaultSettings::default()
    };
    let mut solver = DefaultSolver::new(&p, &q, &a, &problem.b, &cones, settings)
        .map_err(|e| SimError::solver(format!("cone program: {e:?}")))?;
    solver.solve();
    let sol = &solver.solution;
    match sol.status {
        // `AlmostSolved` meets the reduced tolerances (feasibility 1e-4, gap 5e-5),
        // inside the acceptance bound of §10.0.8 item 5.
        SolverStatus::Solved | SolverStatus::AlmostSolved => Ok(Solution {
            x: sol.x.clone(),
            status: format!("{:?}", sol.status),
            iterations: sol.iterations,
            seconds: sol.solve_time,
        }),
        status => Err(SimError::solver(format!(
            "interior point: {status:?} after {} iterations (primal residual {:.3e}, dual {:.3e})",
            sol.iterations, sol.r_prim, sol.r_dual
        ))),
    }
}
