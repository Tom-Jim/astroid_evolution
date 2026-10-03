pub fn apply_under_relaxation(r_current: &[f64], r_star: &[f64], lambda: f64) -> Vec<f64> {
    assert_eq!(r_current.len(), r_star.len());
    let factor = lambda.clamp(0.01, 1.0);
    r_current
        .iter()
        .zip(r_star.iter())
        .map(|(&r_curr, &r_st)| (1.0 - factor) * r_curr + factor * r_st)
        .collect()
}
#[derive(Debug, Clone)]
pub struct AdaptiveRelaxationController {
    pub lambda: f64,
    pub lambda_min: f64,
    pub lambda_max: f64,
    pub prev_residual: f64,
}
impl AdaptiveRelaxationController {
    pub fn new(initial_lambda: f64, lambda_min: f64, lambda_max: f64) -> Self {
        Self {
            lambda: initial_lambda.clamp(lambda_min, lambda_max),
            lambda_min,
            lambda_max,
            prev_residual: f64::INFINITY,
        }
    }
    pub fn update(&mut self, current_residual: f64) -> f64 {
        if current_residual < self.prev_residual {
            self.lambda = (self.lambda * 1.05).min(self.lambda_max);
        } else {
            self.lambda = (self.lambda * 0.7).max(self.lambda_min);
        }
        self.prev_residual = current_residual;
        self.lambda
    }
}
#[derive(Debug, Clone, Default)]
pub struct AitkenAccelerator {
    prev_r: Option<Vec<f64>>,
    prev_diff: Option<Vec<f64>>,
}
impl AitkenAccelerator {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn reset(&mut self) {
        self.prev_r = None;
        self.prev_diff = None;
    }
    pub fn accelerate(
        &mut self,
        r_curr: &[f64],
        r_base_next: &[f64],
        trust_radius: f64,
    ) -> Vec<f64> {
        let n = r_curr.len();
        if r_base_next.len() != n {
            return r_base_next.to_vec();
        }
        let curr_diff: Vec<f64> = (0..n).map(|i| r_base_next[i] - r_curr[i]).collect();
        if let (Some(prev_r), Some(prev_diff)) = (&self.prev_r, &self.prev_diff) {
            if prev_r.len() == n && prev_diff.len() == n {
                let dist_from_prev = (0..n)
                    .map(|i| (r_curr[i] - prev_r[i]).powi(2))
                    .sum::<f64>()
                    .sqrt();
                let r_norm = (0..n)
                    .map(|i| r_curr[i].powi(2))
                    .sum::<f64>()
                    .sqrt()
                    .max(1.0);
                if dist_from_prev / r_norm < 0.05 {
                    let dot_num: f64 = (0..n).map(|i| curr_diff[i] * prev_diff[i]).sum();
                    let dot_den: f64 = (0..n).map(|i| prev_diff[i] * prev_diff[i]).sum();
                    if dot_den > 1e-12 {
                        let rho = (dot_num / dot_den).clamp(-0.5, 0.98);
                        if rho > 0.25 && rho < 0.95 {
                            let factor = (0.55 * rho / (1.0 - rho)).clamp(0.0, 3.5);
                            let mut accelerated = Vec::with_capacity(n);
                            let mut max_shift = 0.0f64;
                            for i in 0..n {
                                let shift = factor * curr_diff[i];
                                max_shift = max_shift.max(shift.abs());
                                accelerated.push(r_base_next[i] + shift);
                            }
                            if max_shift < trust_radius {
                                self.prev_r = Some(accelerated.clone());
                                self.prev_diff = Some(curr_diff);
                                return accelerated;
                            }
                        }
                    }
                }
            }
        }
        self.prev_r = Some(r_base_next.to_vec());
        self.prev_diff = Some(curr_diff);
        r_base_next.to_vec()
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_under_relaxation() {
        let r_curr = vec![100.0, 200.0];
        let r_star = vec![120.0, 180.0];
        let lambda = 0.25;
        let r_next = apply_under_relaxation(&r_curr, &r_star, lambda);
        assert!((r_next[0] - 105.0).abs() < 1e-12);
        assert!((r_next[1] - 195.0).abs() < 1e-12);
    }
    #[test]
    fn test_adaptive_controller() {
        let mut controller = AdaptiveRelaxationController::new(0.3, 0.05, 0.8);
        assert_eq!(controller.lambda, 0.3);
        controller.update(1.0);
        controller.update(0.8);
        assert!(controller.lambda > 0.3);
        let before = controller.lambda;
        controller.update(0.9);
        assert!(controller.lambda < before);
    }
}
