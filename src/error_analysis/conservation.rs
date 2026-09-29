//! 质量、动量与外部调和性的守恒检查。

pub fn normalized_delta(before: f64, after: f64) -> f64 {
    (after - before).abs() / before.abs().max(1e-30)
}

