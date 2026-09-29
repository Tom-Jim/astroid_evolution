//! 欠松弛迭代与自适应阻尼
//! 严格依照 docs/software.md 第 11 节 与 docs/mathstrict.md 第 10.4 节
//!
//! R^(n+1) = (1 - lambda) * R^(n) + lambda * R_*^(n)

/// 执行经典欠松弛半径更新
///
/// 参数：
/// - r_current: 当前迭代步半径序列 R^(n)
/// - r_star: QP 投影候选半径序列 R_*^(n)
/// - lambda: 松弛阻尼因子 (0 < lambda <= 1.0)
pub fn apply_under_relaxation(
    r_current: &[f64],
    r_star: &[f64],
    lambda: f64,
) -> Vec<f64> {
    assert_eq!(r_current.len(), r_star.len());
    let factor = lambda.clamp(0.01, 1.0);

    r_current
        .iter()
        .zip(r_star.iter())
        .map(|(&r_curr, &r_st)| (1.0 - factor) * r_curr + factor * r_st)
        .collect()
}

/// 自适应欠松弛控制器
#[derive(Debug, Clone)]
pub struct AdaptiveRelaxationController {
    /// 当前松弛因子 lambda
    pub lambda: f64,
    /// 最小允许松弛因子
    pub lambda_min: f64,
    /// 最大允许松弛因子
    pub lambda_max: f64,
    /// 上一步几何残差
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

    /// 根据残差变化自适应调整松弛因子
    pub fn update(&mut self, current_residual: f64) -> f64 {
        if current_residual < self.prev_residual {
            // 残差单调下降：平稳放大松弛因子以加速收敛
            self.lambda = (self.lambda * 1.05).min(self.lambda_max);
        } else {
            // 残差反弹/振荡：回退并施加更强阻尼
            self.lambda = (self.lambda * 0.7).max(self.lambda_min);
        }
        self.prev_residual = current_residual;
        self.lambda
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
        // r0 = 0.75 * 100 + 0.25 * 120 = 75 + 30 = 105.0
        // r1 = 0.75 * 200 + 0.25 * 180 = 150 + 45 = 195.0
        assert!((r_next[0] - 105.0).abs() < 1e-12);
        assert!((r_next[1] - 195.0).abs() < 1e-12);
    }

    #[test]
    fn test_adaptive_controller() {
        let mut controller = AdaptiveRelaxationController::new(0.3, 0.05, 0.8);
        assert_eq!(controller.lambda, 0.3);

        // 残差改善
        controller.update(1.0);
        controller.update(0.8);
        assert!(controller.lambda > 0.3);

        // 残差变差
        let before = controller.lambda;
        controller.update(0.9);
        assert!(controller.lambda < before);
    }
}
