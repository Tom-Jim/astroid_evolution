//! Lebedev–Laikov nodes for the angular integral (MS §3.14).
//!
//! The 302-point rule of Lebedev and Laikov (degree 29), tabulated in
//! `lebedev_table.rs`. The tabulated weights sum to one; the factor `4π` is
//! applied here so the GPU kernel receives `ω_q` directly.

use std::sync::OnceLock;

use crate::common::math::DVec3;

/// Production rule. `320` face centres times this count stays under `10^5` rays.
pub const COUNT: usize = 302;

pub struct Rule {
    pub dirs: Vec<DVec3>,
    pub weights: Vec<f64>,
}

pub fn rule() -> &'static Rule {
    static RULE: OnceLock<Rule> = OnceLock::new();
    RULE.get_or_init(|| {
        let (x, y, z, w) = grid();
        let dirs = x
            .into_iter()
            .zip(y)
            .zip(z)
            .map(|((x, y), z)| DVec3::new(x, y, z))
            .collect();
        let weights = w
            .into_iter()
            .map(|w| w * 4.0 * std::f64::consts::PI)
            .collect();
        Rule { dirs, weights }
    })
}

fn grid() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    use super::lebedev_table::{W, X, Y, Z};
    (X.to_vec(), Y.to_vec(), Z.to_vec(), W.to_vec())
}
