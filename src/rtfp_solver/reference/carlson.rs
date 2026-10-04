//! Carlson symmetric elliptic integrals `R_F` and `R_D` by the duplication
//! theorem (Carlson 1995; Press et al., Numerical Recipes §6.12).

const RF_ERRTOL: f64 = 0.0025;
const RD_ERRTOL: f64 = 0.0015;

/// `R_F(x, y, z) = ½ ∫₀^∞ dt / √((t+x)(t+y)(t+z))`, for `x, y, z ≥ 0` with at most one zero.
pub fn rf(x: f64, y: f64, z: f64) -> f64 {
    let (mut xt, mut yt, mut zt) = (x, y, z);
    loop {
        let (sx, sy, sz) = (xt.sqrt(), yt.sqrt(), zt.sqrt());
        let lambda = sx * (sy + sz) + sy * sz;
        xt = 0.25 * (xt + lambda);
        yt = 0.25 * (yt + lambda);
        zt = 0.25 * (zt + lambda);
        let ave = (xt + yt + zt) / 3.0;
        let (dx, dy, dz) = ((ave - xt) / ave, (ave - yt) / ave, (ave - zt) / ave);
        if dx.abs().max(dy.abs()).max(dz.abs()) <= RF_ERRTOL {
            let e2 = dx * dy - dz * dz;
            let e3 = dx * dy * dz;
            let series = 1.0 + (e2 / 24.0 - 0.1 - 3.0 / 44.0 * e3) * e2 + e3 / 14.0;
            return series / ave.sqrt();
        }
    }
}

/// `R_D(x, y, z) = (3/2) ∫₀^∞ dt / ((t+z) √((t+x)(t+y)(t+z)))`, for `x, y ≥ 0` with at most one zero, and `z > 0`.
pub fn rd(x: f64, y: f64, z: f64) -> f64 {
    const C1: f64 = 3.0 / 14.0;
    const C2: f64 = 1.0 / 6.0;
    const C3: f64 = 9.0 / 22.0;
    const C4: f64 = 3.0 / 26.0;
    const C5: f64 = 0.25 * C3;
    const C6: f64 = 1.5 * C4;
    let (mut xt, mut yt, mut zt) = (x, y, z);
    let mut sum = 0.0;
    let mut fac = 1.0;
    loop {
        let (sx, sy, sz) = (xt.sqrt(), yt.sqrt(), zt.sqrt());
        let lambda = sx * (sy + sz) + sy * sz;
        sum += fac / (sz * (zt + lambda));
        fac *= 0.25;
        xt = 0.25 * (xt + lambda);
        yt = 0.25 * (yt + lambda);
        zt = 0.25 * (zt + lambda);
        let ave = 0.2 * (xt + yt + 3.0 * zt);
        let (dx, dy, dz) = ((ave - xt) / ave, (ave - yt) / ave, (ave - zt) / ave);
        if dx.abs().max(dy.abs()).max(dz.abs()) <= RD_ERRTOL {
            let ea = dx * dy;
            let eb = dz * dz;
            let ec = ea - eb;
            let ed = ea - 6.0 * eb;
            let ee = ed + ec + ec;
            return 3.0 * sum
                + fac
                    * (1.0
                        + ed * (-C1 + C5 * ed - C6 * dz * ee)
                        + dz * (C2 * ee + dz * (-C3 * ec + dz * C4 * ea)))
                    / (ave * ave.sqrt());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rf_matches_published_values() {
        assert!((rf(1.0, 2.0, 4.0) - 0.685_085_816_6).abs() < 1e-9);
        assert!((rf(0.0, 1.0, 2.0) - 1.311_028_777_1).abs() < 1e-9);
        assert!((rf(3.0, 3.0, 3.0) - 3.0_f64.sqrt().recip()).abs() < 1e-15);
    }

    #[test]
    fn rd_matches_published_values() {
        assert!((rd(0.0, 2.0, 1.0) - 1.797_210_352_1).abs() < 1e-9);
        assert!((rd(2.0, 2.0, 2.0) - 2.0_f64.powf(-1.5)).abs() < 1e-15);
    }

    #[test]
    fn rf_and_rd_obey_the_homogeneity_degree() {
        let (x, y, z) = (0.7, 1.9, 3.1);
        assert!((rf(4.0 * x, 4.0 * y, 4.0 * z) - 0.5 * rf(x, y, z)).abs() < 1e-14);
        assert!((rd(4.0 * x, 4.0 * y, 4.0 * z) - 0.125 * rd(x, y, z)).abs() < 1e-14);
    }
}
