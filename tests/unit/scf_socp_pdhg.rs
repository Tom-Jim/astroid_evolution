//! Unit tests of `scf::socp_pdhg`, moved out of `src/scf/socp_pdhg.rs`.

mod ctx {
    pub use astroid_evolution::common::time::wall_seconds;
}
use ctx::*;

use astroid_evolution::scf::socp_pdhg::*;

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
        n_trust: 0,
        warm: None,
    })
    .expect("bound QP");
    assert_kkt(&sol);
    assert!((sol.x[0] - 1.0).abs() <= 2.0e-3, "x = {}", sol.x[0]);
    let again = solve(Problem {
        p_diag: vec![1.0],
        q: vec![-5.0],
        rows: vec![0],
        cols: vec![0],
        vals: vec![1.0],
        b: vec![1.0],
        n_eq: 0,
        n_nn: 1,
        n_soc: 0,
        n_trust: 0,
        warm: Some(WarmStart {
            x: sol.x.clone(),
            y: sol.y.clone(),
        }),
    })
    .expect("warm bound QP");
    assert_eq!(again.iterations, 0);
    assert!((again.x[0] - sol.x[0]).abs() <= 1e-12);
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
        n_trust: 0,
        warm: None,
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
        n_trust: 0,
        warm: None,
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

#[test]
fn a_stalled_quadratic_still_converges() {
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
        n_trust: 0,
        warm: None,
    })
    .expect("bound QP");
    assert_kkt(&sol);
}

#[test]
fn trust_ball_stops_on_the_sphere() {
    // min ½(x−3)²+½(y−4)² s.t. ‖(x, y)‖ ≤ 1. The minimizer is (0.6, 0.8).
    let sol = solve(Problem {
        p_diag: vec![1.0, 1.0],
        q: vec![-3.0, -4.0],
        rows: vec![1, 2],
        cols: vec![0, 1],
        vals: vec![-1.0, -1.0],
        b: vec![1.0, 0.0, 0.0],
        n_eq: 0,
        n_nn: 0,
        n_soc: 0,
        n_trust: 3,
        warm: None,
    })
    .expect("trust ball");
    assert_kkt(&sol);
    assert!((sol.x[0] - 0.6).abs() <= 2.0e-3, "x = {}", sol.x[0]);
    assert!((sol.x[1] - 0.8).abs() <= 2.0e-3, "y = {}", sol.x[1]);
}

#[test]
fn a_long_mass_row_and_trust_cone_finish() {
    // min ½‖x‖² s.t. Σ x_i = 1 and ‖x‖ ≤ 10. The mass row and the trust
    // cone are both longer than one GPU thread used to walk.
    let n = 4096usize;
    let mut rows = Vec::with_capacity(n * 2);
    let mut cols = Vec::with_capacity(n * 2);
    let mut vals = Vec::with_capacity(n * 2);
    for j in 0..n {
        rows.push(0);
        cols.push(j);
        vals.push(1.0);
        rows.push(2 + j);
        cols.push(j);
        vals.push(-1.0);
    }
    let mut b = vec![0.0; n + 2];
    b[0] = 1.0;
    b[1] = 10.0;
    let started = std::time::Instant::now();
    let sol = solve(Problem {
        p_diag: vec![1.0; n],
        q: vec![0.0; n],
        rows,
        cols,
        vals,
        b,
        n_eq: 1,
        n_nn: 0,
        n_soc: 0,
        n_trust: n + 1,
        warm: None,
    })
    .expect("long row");
    assert!(
        started.elapsed().as_secs_f64() < 15.0,
        "long reduction took {:.1}s",
        started.elapsed().as_secs_f64()
    );
    assert_kkt(&sol);
    let mean = 1.0 / n as f64;
    for (i, x) in sol.x.iter().enumerate() {
        assert!((x - mean).abs() <= 2.0e-3, "x[{i}] = {x}");
    }
}
