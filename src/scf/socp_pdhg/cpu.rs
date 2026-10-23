//! The PDHG iteration of `pdhg.wesl` on the CPU in f64.
//!
//! The same primal, dual, cone and ergodic-average steps in the same order as the
//! GPU kernels, carried in f64 instead of double-single: 53 bits where the GPU keeps
//! about 48, so the iterates agree with the GPU's to its precision and are no less
//! accurate. A cone program the size of a surface patch then needs no GPU round
//! trip, which the browser pays in milliseconds per batch.

use super::Csr;
use super::gpu::{Csc, transpose};

/// `(x, y, x_avg, y_avg)`.
pub type State = [Vec<f64>; 4];

pub struct Device {
    counts: [usize; 4],
    csc: Csc,
    a: Csr,
    /// (q, P, τ) per column.
    colc: Vec<[f64; 3]>,
    b: Vec<f64>,
    sigma: Vec<f64>,
    x: Vec<f64>,
    xavg: Vec<f64>,
    y: Vec<f64>,
    yavg: Vec<f64>,
    xbar: Vec<f64>,
    ax: Vec<f64>,
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
    ) -> Self {
        Self {
            counts: [n_eq, n_nn, n_soc, n_trust],
            csc: transpose(a),
            a: a.clone(),
            colc: (0..a.n).map(|j| [q[j], p[j], tau[j]]).collect(),
            b: b.to_vec(),
            sigma: sigma.to_vec(),
            x: vec![0.0; a.n],
            xavg: vec![0.0; a.n],
            y: vec![0.0; a.m],
            yavg: vec![0.0; a.m],
            xbar: vec![0.0; a.n],
            ax: vec![0.0; a.m],
        }
    }

    /// The point `write_state` puts on the GPU (x̄ is rewritten before it is read).
    pub fn write_state(&mut self, x: &[f64], y: &[f64], xavg: &[f64], yavg: &[f64]) {
        for (dst, v) in [
            (&mut self.x, x),
            (&mut self.xavg, xavg),
            (&mut self.y, y),
            (&mut self.yavg, yavg),
        ] {
            if !v.is_empty() {
                dst.copy_from_slice(v);
            }
        }
    }

    pub fn restore(&mut self, state: &State) {
        self.write_state(&state[0], &state[1], &state[2], &state[3]);
    }

    /// Batches of `(iterations, average count)` at one step scale, and the iterates
    /// after each.
    pub fn run(&mut self, plan: &[(u32, u32)], scale: f64) -> Vec<State> {
        plan.iter()
            .map(|&(iters, avg_count)| {
                for k in 0..iters {
                    // pdhg_tick: the sample after `avg_count + k` enters with weight
                    // 1/(count + 1).
                    self.iterate(scale, 1.0 / (f64::from(avg_count + k) + 1.0));
                }
                [
                    self.x.clone(),
                    self.y.clone(),
                    self.xavg.clone(),
                    self.yavg.clone(),
                ]
            })
            .collect()
    }

    fn iterate(&mut self, scale: f64, inv: f64) {
        let [n_eq, n_nn, n_soc, n_trust] = self.counts;
        let n_lin = n_eq + n_nn;
        let mix = |mean: &mut f64, sample: f64| *mean += inv * (sample - *mean);
        // pdhg_primal: x ← (x − τ(Aᵀy + q)) / (1 + τP), x̄ ← 2x − x_old.
        for j in 0..self.a.n {
            let mut aty = 0.0;
            for k in self.csc.ptr[j] as usize..self.csc.ptr[j + 1] as usize {
                aty += self.csc.val[k] * self.y[self.csc.row[k] as usize];
            }
            let [q, p, tau_j] = self.colc[j];
            let tau = tau_j * scale;
            let x_old = self.x[j];
            let x_new = (x_old - tau * (aty + q)) / (1.0 + tau * p);
            self.x[j] = x_new;
            self.xbar[j] = 2.0 * x_new - x_old;
            mix(&mut self.xavg[j], x_new);
        }
        // pdhg_ax (long rows included: one sum per row)
        for i in 0..self.a.m {
            let mut s = 0.0;
            for k in self.a.row_ptr[i] as usize..self.a.row_ptr[i + 1] as usize {
                s += self.a.val[k] * self.xbar[self.a.col[k] as usize];
            }
            self.ax[i] = s;
        }
        let pred = |y: &[f64], row: usize, sigma: f64| y[row] + sigma * (self.ax[row] - self.b[row]);
        // pdhg_dual: equality and non-negative rows.
        for i in 0..n_lin {
            let mut p = pred(&self.y, i, self.sigma[i] * scale);
            if i >= n_eq {
                p = p.max(0.0);
            }
            self.y[i] = p;
            mix(&mut self.yavg[i], p);
        }
        // Second-order cones of four rows (t, w).
        for job in 0..n_soc {
            let row = n_lin + 4 * job;
            let sigma = self.sigma[row] * scale;
            let t: [f64; 4] = std::array::from_fn(|r| pred(&self.y, row + r, sigma));
            let nrm = (t[1] * t[1] + t[2] * t[2] + t[3] * t[3]).sqrt();
            let out = if nrm <= t[0] {
                t
            } else if nrm <= -t[0] {
                [0.0; 4]
            } else {
                let a = (nrm + t[0]) / (2.0 * nrm);
                [a * nrm, a * t[1], a * t[2], a * t[3]]
            };
            for r in 0..4 {
                self.y[row + r] = out[r];
                mix(&mut self.yavg[row + r], out[r]);
            }
        }
        // pdhg_trust_pred, pdhg_trust_proj: the trailing cone (t, w).
        if n_trust > 0 {
            let row0 = n_lin + 4 * n_soc;
            let sigma = self.sigma[row0] * scale;
            let mut sq = 0.0;
            for i in 0..n_trust {
                let p = pred(&self.y, row0 + i, sigma);
                self.y[row0 + i] = p;
                if i > 0 {
                    sq += p * p;
                }
            }
            let head = self.y[row0];
            let nrm = sq.sqrt();
            for i in 0..n_trust {
                let row = row0 + i;
                let mut v = self.y[row];
                if nrm > head {
                    v = if nrm <= -head {
                        0.0
                    } else {
                        let a = (nrm + head) / (2.0 * nrm);
                        if i == 0 { a * nrm } else { a * v }
                    };
                    self.y[row] = v;
                }
                mix(&mut self.yavg[row], v);
            }
        }
    }
}
