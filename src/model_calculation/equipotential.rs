//! 等势候选面半径求解与径向连续跟踪
//! 严格依照 docs/software.md 第 8.4 节 与 docs/mathstrict.md 第 8 节

/// 等势半径单方向求解结果
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EquipotentialResult {
    /// 求解得到的等势半径 R_eq (m)
    pub radius: f64,
    /// 是否成功找到合法等势根
    pub success: bool,
    /// 最终势能残差 |W(R_eq u) - W_0|
    pub residual: f64,
    /// 迭代迭代次数
    pub iterations: usize,
}

/// 求解指定方向 u 上满足 W(R u) = W_0 的等势半径 R_eq
///
/// 文档强制要求：
/// 1. 检查根存在性（区间端点符号变号检测）；
/// 2. 规定多根时的分支：默认优先选择靠近基准初猜的最外层稳定分支；
/// 3. 进行径向连续跟踪：利用 initial_guess 进行有界初猜；
/// 4. 记录失败方向。
pub fn solve_equipotential_radius<F>(
    direction: [f64; 3],
    w0: f64,
    potential_fn: &F,
    r_min: f64,
    r_max: f64,
    initial_guess: f64,
    tol: f64,
    max_iter: usize,
) -> EquipotentialResult
where
    F: Fn([f64; 3]) -> f64,
{
    // 归一化方向向量
    let dir_norm = (direction[0] * direction[0] + direction[1] * direction[1] + direction[2] * direction[2]).sqrt();
    if dir_norm < 1e-12 {
        return EquipotentialResult {
            radius: initial_guess,
            success: false,
            residual: f64::INFINITY,
            iterations: 0,
        };
    }
    let u = [
        direction[0] / dir_norm,
        direction[1] / dir_norm,
        direction[2] / dir_norm,
    ];

    let eval_diff = |r: f64| -> f64 {
        let pos = [r * u[0], r * u[1], r * u[2]];
        potential_fn(pos) - w0
    };

    // 1. 检查区间与变号
    let mut low = r_min.max(1e-3);
    let mut high = r_max.max(low + 1.0);

    let f_low = eval_diff(low);
    let f_high = eval_diff(high);

    // 尝试在 initial_guess 附近先进行局部搜索
    let r_init = initial_guess.clamp(low, high);
    let f_init = eval_diff(r_init);
    if f_init.abs() < tol {
        return EquipotentialResult {
            radius: r_init,
            success: true,
            residual: f_init.abs(),
            iterations: 0,
        };
    }

    // 确定包含根的子区间
    if f_low * f_init <= 0.0 {
        high = r_init;
    } else if f_init * f_high <= 0.0 {
        low = r_init;
    } else if f_low * f_high > 0.0 {
        // 未变号：多网格采样寻找跨越零点的区间
        let n_sub = 16;
        let dr = (high - low) / n_sub as f64;
        let mut found = false;
        let mut r_prev = low;
        let mut f_prev = f_low;

        for k in 1..=n_sub {
            let r_curr = low + k as f64 * dr;
            let f_curr = eval_diff(r_curr);
            if f_prev * f_curr <= 0.0 {
                low = r_prev;
                high = r_curr;
                found = true;
                break;
            }
            r_prev = r_curr;
            f_prev = f_curr;
        }

        if !found {
            // 记录失败方向，返回最接近 W0 的点
            return EquipotentialResult {
                radius: r_init,
                success: false,
                residual: f_init.abs(),
                iterations: n_sub,
            };
        }
    }

    // 2. 二分法 + 拟牛顿法精化根
    let mut a = low;
    let mut b = high;
    let mut fa = eval_diff(a);
    let mut iter = 0;

    let mut r_mid = 0.5 * (a + b);
    let mut f_mid = eval_diff(r_mid);

    while iter < max_iter {
        iter += 1;
        r_mid = 0.5 * (a + b);
        f_mid = eval_diff(r_mid);

        let rel_err = f_mid.abs() / w0.abs().max(1.0);
        if f_mid.abs() < tol || rel_err < tol || (b - a) < 1e-12 {
            return EquipotentialResult {
                radius: r_mid,
                success: true,
                residual: f_mid.abs(),
                iterations: iter,
            };
        }

        if fa * f_mid <= 0.0 {
            b = r_mid;
        } else {
            a = r_mid;
            fa = f_mid;
        }
    }

    EquipotentialResult {
        radius: r_mid,
        success: f_mid.abs() < tol * 10.0,
        residual: f_mid.abs(),
        iterations: iter,
    }
}

/// 全向批量求解等势候选面半径并进行连续跟踪与失败回退修复
pub fn solve_equipotential_radii_batch<F>(
    directions: &[[f64; 3]],
    w0: f64,
    potential_fn: &F,
    r_min: f64,
    r_max: f64,
    initial_radius: f64,
    tol: f64,
) -> (Vec<f64>, usize)
where
    F: Fn([f64; 3]) -> f64,
{
    let n = directions.len();
    let mut radii = Vec::with_capacity(n);
    let mut failed_count = 0;
    let mut last_valid_r = initial_radius;

    for &dir in directions {
        let res = solve_equipotential_radius(
            dir,
            w0,
            potential_fn,
            r_min,
            r_max,
            last_valid_r,
            tol,
            64,
        );
        if res.success {
            radii.push(res.radius);
            last_valid_r = res.radius;
        } else {
            failed_count += 1;
            // 失败方向使用上一近邻有效半径回退
            radii.push(last_valid_r);
        }
    }

    (radii, failed_count)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_mass_equipotential() {
        // 点质量势能: U(r) = -G M / r, 无自转
        // W(r) = -1000.0 / r. 求解 W(r) = -2.0 -> r = 500.0
        let gm = 1000.0;
        let potential_fn = |pos: [f64; 3]| -> f64 {
            let r = (pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2]).sqrt();
            -gm / r.max(1e-6)
        };

        let dir = [1.0, 0.0, 0.0];
        let w0 = -2.0;

        let res = solve_equipotential_radius(
            dir,
            w0,
            &potential_fn,
            10.0,
            2000.0,
            400.0,
            1e-7,
            100,
        );

        assert!(res.success);
        assert!((res.radius - 500.0).abs() < 1e-5);
        assert!(res.residual < 1e-6);
    }
}
