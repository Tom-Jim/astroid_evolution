//! 径向核积分模块
//! 包含 alpha=1 解析原函数 (P, Q, L)、有限部分下限与任意 alpha 的固定高斯求积三元组
//! 严格依照 docs/software.md 第 7.3 节 与 docs/mathstrict.md 第 1.6 / 3.2 节

/// 16 点 Gauss-Legendre 求积节点与权重 (归一化区间 [-1, 1])
pub const GAUSS_LEGENDRE_16_NODES: [f64; 16] = [
    -0.9894009349916499, -0.9445750230732326, -0.8656312023878318, -0.7554044083550030,
    -0.6178762444026438, -0.4580167776572274, -0.2816035507797087, -0.0950125098376374,
     0.0950125098376374,  0.2816035507797087,  0.4580167776572274,  0.6178762444026438,
     0.7554044083550030,  0.8656312023878318,  0.9445750230732326,  0.9894009349916499,
];

pub const GAUSS_LEGENDRE_16_WEIGHTS: [f64; 16] = [
    0.0271524594117541, 0.0622535239386479, 0.0951585116824928, 0.1246289712555339,
    0.1495959888165767, 0.1691565193950025, 0.1826034150449236, 0.1894506104550685,
    0.1894506104550685, 0.1826034150449236, 0.1691565193950025, 0.1495959888165767,
    0.1246289712555339, 0.0951585116824928, 0.0622535239386479, 0.0271524594117541,
];

/// D(R) = (R - b)^2 + d^2
#[inline]
pub fn d_eval(r: f64, b: f64, d: f64) -> f64 {
    let diff = r - b;
    (diff * diff + d * d).max(1e-300)
}

/// alpha=1 时势能径向积分原函数 P(R) = 0.5 * ln(D(R)) + (b/d) * atan((R-b)/d)
#[inline]
pub fn p_eval(r: f64, b: f64, d: f64) -> f64 {
    if r == 0.0 {
        let a0 = (b * b + d * d).max(1e-300);
        0.5 * a0.ln() - (b / d) * (b / d).atan()
    } else {
        0.5 * d_eval(r, b, d).ln() + (b / d) * ((r - b) / d).atan()
    }
}

/// alpha=1 时梯度径向积分原函数 Q(R) = (1/d) * atan((R-b)/d)
#[inline]
pub fn q_eval(r: f64, b: f64, d: f64) -> f64 {
    if r == 0.0 {
        -(1.0 / d) * (b / d).atan()
    } else {
        (1.0 / d) * ((r - b) / d).atan()
    }
}

/// alpha=1 时 Hessian 径向积分原函数 L(R)
/// L(R) = (1/A0) [ ln(R) - 0.5 * ln(D(R)) + (b/d) * atan((R-b)/d) ]
#[inline]
pub fn l_eval(r: f64, a0: f64, b: f64, d: f64) -> f64 {
    let term = r.ln() - 0.5 * d_eval(r, b, d).ln() + (b / d) * ((r - b) / d).atan();
    term / a0
}

/// alpha=1 时普通非零交段 [r0, r1] 上的势能径向积分 I_U = P(r1) - P(r0)
pub fn alpha_one_potential(b: f64, d: f64, r0: f64, r1: f64) -> f64 {
    p_eval(r1, b, d) - p_eval(r0, b, d)
}

/// alpha=1 时普通非零交段 [r0, r1] 上的梯度径向积分 I_g = Q(r1) - Q(r0)
pub fn alpha_one_gradient(b: f64, d: f64, r0: f64, r1: f64) -> f64 {
    q_eval(r1, b, d) - q_eval(r0, b, d)
}

/// alpha=1 时普通外部交段 [r0, r1] (r0 > 0) 上的 Hessian 径向积分 I_H = L(r1) - L(r0)
pub fn alpha_one_hessian(a0: f64, b: f64, d: f64, r0: f64, r1: f64) -> f64 {
    l_eval(r1, a0, b, d) - l_eval(r0, a0, b, d)
}

/// alpha=1 时以 r0=0 为端点的有限部分径向积分 J_{ell0}^{FP}(0, L)
/// 严格依照 docs/mathstrict.md 第 1.6 节 公式 (630-646 行):
/// J_{ell0}^{FP}(0, L) = (1/A0) [ ln(L / ell0) - 0.5 * ln(D(L)/A0) + (b/d) ( atan((L-b)/d) + atan(b/d) ) ]
pub fn alpha_one_hessian_finite_part(a0: f64, b: f64, d: f64, l: f64, ell0: f64) -> f64 {
    let dl = d_eval(l, b, d);
    let ratio_dl = (dl / a0).max(1e-300);
    let term_log = (l / ell0).ln() - 0.5 * ratio_dl.ln();
    let term_atan = (b / d) * (((l - b) / d).atan() + (b / d).atan());
    (term_log + term_atan) / a0
}

/// 任意 alpha (包含 alpha != 1) 的 16 点 Gauss-Legendre 固定求积三元组 (I_U, I_g, I_H)
/// I_U = int_{r0}^{r1} R D(R)^(-alpha) dR
/// I_g = int_{r0}^{r1} D(R)^(-alpha) dR
/// I_H = int_{r0}^{r1} (D(R)^(-alpha) / R) dR
pub fn fixed_quadrature_triplet(
    alpha: f64,
    b: f64,
    d: f64,
    r0: f64,
    r1: f64,
) -> (f64, f64, f64) {
    let h = 0.5 * (r1 - r0);
    let mid = 0.5 * (r1 + r0);

    let mut sum_u = 0.0f64;
    let mut sum_g = 0.0f64;
    let mut sum_h = 0.0f64;

    for i in 0..16 {
        let node = GAUSS_LEGENDRE_16_NODES[i];
        let weight = GAUSS_LEGENDRE_16_WEIGHTS[i];
        let r = mid + h * node;
        let d_val = d_eval(r, b, d);
        let d_pow = d_val.powf(-alpha);

        sum_u += weight * r * d_pow;
        sum_g += weight * d_pow;
        if r.abs() > 1e-12 {
            sum_h += weight * (d_pow / r);
        }
    }

    (sum_u * h, sum_g * h, sum_h * h)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alpha_one_analytic_vs_quadrature() {
        let b = 15.0;
        let d = 20.0;
        let a0 = b * b + d * d;
        let r0 = 5.0;
        let r1 = 35.0;

        let iu_ana = alpha_one_potential(b, d, r0, r1);
        let ig_ana = alpha_one_gradient(b, d, r0, r1);
        let ih_ana = alpha_one_hessian(a0, b, d, r0, r1);

        let (iu_num, ig_num, ih_num) = fixed_quadrature_triplet(1.0, b, d, r0, r1);

        assert!((iu_ana - iu_num).abs() < 1e-10, "I_U 解析解与求积误差过大: ana {iu_ana}, num {iu_num}");
        assert!((ig_ana - ig_num).abs() < 1e-10, "I_g 解析解与求积误差过大: ana {ig_ana}, num {ig_num}");
        assert!((ih_ana - ih_num).abs() < 1e-10, "I_H 解析解与求积误差过大: ana {ih_ana}, num {ih_num}");
    }
}
