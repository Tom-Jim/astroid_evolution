use super::assemble::{assemble_k8_mass_matrix, invert_k8_mass_matrix};
pub type K8State = [f64; 8];
pub fn project_k8(rhs: &K8State) -> Result<K8State, String> {
    let inv = invert_k8_mass_matrix()?;
    let mut out = [0.0; 8];
    for i in 0..8 {
        out[i] = (0..8).map(|j| inv[i][j] * rhs[j]).sum();
    }
    Ok(out)
}
pub fn reconstruct_k8(coefficients: &K8State, basis: &[f64; 8]) -> f64 {
    coefficients.iter().zip(basis).map(|(a, b)| a * b).sum()
}
pub fn k8_condition_proxy() -> f64 {
    let m = assemble_k8_mass_matrix();
    let max_row = m
        .iter()
        .map(|r| r.iter().map(|x| x.abs()).sum::<f64>())
        .fold(0.0, f64::max);
    max_row / m[7][7].abs().max(1e-30)
}
pub fn transport_k8(
    current: &[K8State],
    jacobians: &[f64],
    dt: f64,
    source: Option<&[K8State]>,
) -> Result<Vec<K8State>, String> {
    if current.len() != jacobians.len() || source.map_or(false, |s| s.len() != current.len()) {
        return Err("K=8 输运输入长度不一致".into());
    }
    if !dt.is_finite()
        || dt < 0.0
        || current.iter().flatten().any(|x| !x.is_finite())
        || jacobians.iter().any(|x| !x.is_finite() || *x <= 0.0)
    {
        return Err("K=8 物理输运状态非法".into());
    }
    Ok(current
        .iter()
        .enumerate()
        .map(|(i, q)| {
            std::array::from_fn(|k| jacobians[i] * q[k] + source.map_or(0.0, |s| dt * s[i][k]))
        })
        .collect())
}
pub fn advance_k8(
    current: &[K8State],
    jacobians: &[f64],
    dt: f64,
    source: Option<&[K8State]>,
) -> Result<Vec<K8State>, String> {
    if current.len() != jacobians.len() || source.is_some_and(|s| s.len() != current.len()) {
        return Err("K=8 物理模态输入长度不一致".into());
    }
    if !dt.is_finite()
        || dt < 0.0
        || current.iter().flatten().any(|x| !x.is_finite())
        || jacobians.iter().any(|x| !x.is_finite() || *x <= 0.0)
        || source.is_some_and(|s| s.iter().flatten().any(|x| !x.is_finite()))
    {
        return Err("K=8 输运需要有限状态、正 Jacobian 和非负 dt".into());
    }
    let source_q = if dt > 0.0 {
        source
            .map(|s| s.iter().map(project_k8).collect::<Result<Vec<_>, _>>())
            .transpose()?
    } else {
        None
    };
    Ok(current
        .iter()
        .enumerate()
        .map(|(i, q)| {
            std::array::from_fn(|k| {
                jacobians[i] * q[k] + source_q.as_ref().map_or(0.0, |s| dt * s[i][k])
            })
        })
        .collect())
}
#[derive(Clone, Copy, Debug)]
pub struct K8RheologyPoint {
    pub pressure: f64,
    pub packing_fraction: f64,
    pub coordination_number: f64,
    pub grain_diameter: f64,
    pub grain_density: f64,
    pub cohesion_force: f64,
    pub cohesion_macro: f64,
    pub mu_s: f64,
    pub mu_2: f64,
    pub i_0: f64,
    pub mu_y: f64,
    pub f_rad: f64,
    pub tide_norm: f64,
}
#[derive(Clone, Copy, Debug)]
pub struct K8RheologyOutput {
    pub mu_dynamic: f64,
    pub mu_effective: f64,
    pub friction_angle: f64,
    pub inertial_number: f64,
    pub p_star: f64,
}
pub fn close_k8_rheology_physical(p: K8RheologyPoint) -> Result<K8RheologyOutput, String> {
    if ![
        p.pressure,
        p.packing_fraction,
        p.coordination_number,
        p.grain_diameter,
        p.grain_density,
        p.f_rad,
        p.tide_norm,
    ]
    .iter()
    .all(|x| x.is_finite())
        || p.pressure < 0.0
        || p.grain_diameter <= 0.0
        || p.grain_density <= 0.0
    {
        return Err("K=8 流变状态非法".into());
    }
    let pc = p.coordination_number.max(0.0) * p.packing_fraction.max(0.0) * p.cohesion_force
        / (p.grain_diameter * p.grain_diameter);
    let ps = (p.pressure + pc).max(1e-12);
    let ic = p.grain_diameter * p.f_rad.max(0.0) / (ps / p.grain_density).sqrt();
    let mu_d = p.mu_s + (p.mu_2 - p.mu_s) * ic / (p.i_0.max(1e-12) + ic);
    let mu_eff =
        (p.cohesion_macro + p.mu_y * p.pressure + p.tide_norm / std::f64::consts::SQRT_2) / ps;
    Ok(K8RheologyOutput {
        mu_dynamic: mu_d,
        mu_effective: mu_eff.max(0.0),
        friction_angle: mu_eff.max(0.0).atan(),
        inertial_number: ic,
        p_star: ps,
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn physical_advance_preserves_constant_modal_state_at_unit_jacobian() {
        let q = vec![[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]];
        let out = advance_k8(&q, &[1.0], 0.1, None).unwrap();
        assert!((out[0][0] - 1.0).abs() < 1e-8);
        assert!(out[0][1..].iter().all(|x| x.abs() < 1e-8));
    }
    #[test]
    fn repeated_identity_does_not_excitate_high_modes() {
        let initial = vec![[1.0, -0.2, 0.3, 0.04, 0.0, 0.0, 0.0, 0.0]];
        let mut q = initial.clone();
        for _ in 0..1000 {
            q = advance_k8(&q, &[1.0], 0.001, None).unwrap();
        }
        assert_eq!(q, initial);
        assert!(advance_k8(&q, &[1.0], f64::NAN, None).is_err());
    }
    #[test]
    fn physical_rheology_reports_mu_i_chain() {
        let out = close_k8_rheology_physical(K8RheologyPoint {
            pressure: 1000.0,
            packing_fraction: 0.6,
            coordination_number: 6.0,
            grain_diameter: 0.01,
            grain_density: 2500.0,
            cohesion_force: 0.0,
            cohesion_macro: 0.0,
            mu_s: 0.38,
            mu_2: 0.64,
            i_0: 0.279,
            mu_y: 0.38,
            f_rad: 0.2,
            tide_norm: 0.0,
        })
        .unwrap();
        assert!(out.mu_dynamic > 0.38 && out.mu_dynamic < 0.64);
        assert!((out.friction_angle - 0.38_f64.atan()).abs() < 0.1);
    }
}
