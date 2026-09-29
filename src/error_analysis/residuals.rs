//! 残差通道的统一标量诊断。

pub fn l2(values: &[f64]) -> f64 {
    values.iter().map(|v| v * v).sum::<f64>().sqrt()
}

pub fn linf(values: &[f64]) -> f64 {
    values.iter().map(|v| v.abs()).fold(0.0, f64::max)
}

