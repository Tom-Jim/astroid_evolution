//! GPU evaluation of the filtered expansion (`shsmooth/shader/eval.wesl`). One thread per vertex.

use super::harmonics::{self, check_l_max, coeff_count};
use crate::common::error::SimError;

#[derive(Clone, Debug)]
pub struct EvalStats {
    pub radii: Vec<f64>,
    /// Vertices whose device radius is not strictly positive.
    pub nonpositive: u32,
    pub seconds: f64,
}

/// Handle to the GPU evaluation (`shsmooth/shader/eval.wesl`).
pub struct ShGpu;

impl ShGpu {
    pub fn new() -> Result<Self, SimError> {
        crate::gpu::gpu().map_err(SimError::gpu)?;
        Ok(Self)
    }

    pub fn device_name(&self) -> String {
        crate::gpu::gpu()
            .map(|g| g.adapter.clone())
            .unwrap_or_default()
    }

    /// Evaluate `Σ c_lm Y_lm` at each direction. Directions are renormalised on the host.
    pub fn evaluate(
        &self,
        l_max: usize,
        coeff: &[f64],
        dirs: &[[f64; 3]],
    ) -> Result<EvalStats, SimError> {
        self.dispatch(l_max, coeff, dirs, None)
    }

    /// Device radius plus a per-vertex residual: `R_final = R_sh + residual`.
    pub fn evaluate_with_residual(
        &self,
        l_max: usize,
        coeff: &[f64],
        dirs: &[[f64; 3]],
        residual: &[f64],
    ) -> Result<EvalStats, SimError> {
        if residual.len() != dirs.len() {
            return Err(SimError::invalid_input(format!(
                "residual length {} does not match {} directions",
                residual.len(),
                dirs.len()
            )));
        }
        if residual.iter().any(|r| !r.is_finite()) {
            return Err(SimError::invalid_input("residuals must be finite"));
        }
        self.dispatch(l_max, coeff, dirs, Some(residual))
    }

    fn dispatch(
        &self,
        l_max: usize,
        coeff: &[f64],
        dirs: &[[f64; 3]],
        residual: Option<&[f64]>,
    ) -> Result<EvalStats, SimError> {
        check_l_max(l_max)?;
        let n = coeff_count(l_max);
        if coeff.len() != n {
            return Err(SimError::invalid_input(format!(
                "expected {n} coefficients at l_max {l_max}, got {}",
                coeff.len()
            )));
        }
        if coeff.iter().any(|c| !c.is_finite()) {
            return Err(SimError::invalid_input("coefficients must be finite"));
        }
        if dirs.is_empty() {
            return Ok(EvalStats {
                radii: Vec::new(),
                nonpositive: 0,
                seconds: 0.0,
            });
        }
        let mut unit = Vec::with_capacity(dirs.len());
        for (i, d) in dirs.iter().enumerate() {
            let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            if !(len.is_finite() && len > 0.0) || !d.iter().all(|v| v.is_finite()) {
                return Err(SimError::geometry(format!(
                    "direction {i} is not a finite nonzero vector"
                )));
            }
            unit.push([d[0] / len, d[1] / len, d[2] / len]);
        }
        let packed_dirs = harmonics::pack_directions(&unit);
        let packed_coeff = harmonics::pack_coefficients(l_max, coeff);
        let recur = harmonics::recurrence_table(l_max);
        let packed_residual = residual.map(harmonics::pack_scalars);
        let started = crate::common::time::wall_seconds();
        let k = &crate::gpu::kernels::SH_EVAL;
        let n = unit.len();
        let mut job = crate::gpu::Job::new();
        let d = job.input(&packed_dirs);
        let params = job.value(&[
            n as u32,
            l_max as u32,
            u32::from(packed_residual.is_some()),
            0,
        ]);
        let c = job.input(&packed_coeff);
        let r = job.input(&recur);
        let radii = job.zeros::<f32>(2 * n);
        let stats = job.zeros::<u32>(1);
        let res = job.input(packed_residual.as_deref().unwrap_or(&[0.0f32, 0.0]));
        job.run(k, k.groups(n), &[d, params, c, r, radii, stats, res]);
        let rr = job.read_all::<f32>(radii, 2 * n);
        let rs = job.read_all::<u32>(stats, 1);
        let out = job.submit().map_err(SimError::gpu)?;
        let radii = out
            .get::<f32>(rr)
            .chunks_exact(2)
            .map(|w| f64::from(w[0]) + f64::from(w[1]))
            .collect();
        Ok(EvalStats {
            radii,
            nonpositive: out.get::<u32>(rs)[0],
            seconds: crate::common::time::wall_seconds() - started,
        })
    }
}
