pub fn evaluate_mass_at_radii<D>(
    radii: &[f64],
    directions: &[[f64; 3]],
    weights: &[f64],
    density_fn: &D,
) -> f64
where
    D: Fn([f64; 3]) -> f64,
{
    let n = radii.len();
    assert_eq!(directions.len(), n);
    assert_eq!(weights.len(), n);
    let mut total_mass = 0.0;
    for i in 0..n {
        let r_i = radii[i].max(0.0);
        let u = directions[i];
        let w = weights[i];
        let n_steps = 16;
        let dr = r_i / n_steps as f64;
        let mut rad_int = 0.0;
        for k in 0..=n_steps {
            let r_k = k as f64 * dr;
            let pos = [r_k * u[0], r_k * u[1], r_k * u[2]];
            let rho = density_fn(pos);
            let val = rho * r_k * r_k;
            let weight = if k == 0 || k == n_steps {
                1.0
            } else if k % 2 == 1 {
                4.0
            } else {
                2.0
            };
            rad_int += weight * val;
        }
        rad_int *= dr / 3.0;
        total_mass += w * rad_int;
    }
    total_mass
}
/// ∂M/∂R_i = ω_i ρ_K(R_i u_i) R_i² for the star-ray mass.
pub fn surface_mass_jacobian(solid_angle: f64, rho_k: f64, radius: f64) -> f64 {
    solid_angle * rho_k * radius * radius
}

/// Flux thickness counted in the Cauchy density: Δh* = (ρ_bulk / ρ_K) Δh.
pub fn flux_density_weight(rho_bulk: f64, rho_k: f64) -> Result<f64, String> {
    if !(rho_bulk.is_finite() && rho_bulk > 0.0) || !(rho_k.is_finite() && rho_k > 0.0) {
        return Err("rho_bulk and rho_K must be finite and positive".into());
    }
    Ok(rho_bulk / rho_k)
}

pub fn solid_angle_weights(
    vertices: &[crate::common::math::DVec3],
    faces: &[[u32; 3]],
) -> Result<Vec<f64>, String> {
    let mut weights = vec![0.0; vertices.len()];
    for face in faces {
        let ids = [face[0] as usize, face[1] as usize, face[2] as usize];
        let corner = |index: usize| {
            vertices[index]
                .try_normalize()
                .ok_or_else(|| "solid angle needs a vertex off the origin".to_string())
        };
        let a = corner(ids[0])?;
        let b = corner(ids[1])?;
        let c = corner(ids[2])?;
        let triple = a.dot(b.cross(c));
        let denom = 1.0 + a.dot(b) + b.dot(c) + c.dot(a);
        if !triple.is_finite() || !denom.is_finite() || denom.abs() <= 1e-14 {
            return Err("a face solid angle is undefined".into());
        }
        let omega = 2.0 * (triple / denom).atan();
        if !omega.is_finite() {
            return Err("a face solid angle is not finite".into());
        }
        for id in ids {
            weights[id] += omega / 3.0;
        }
    }
    if weights.iter().any(|w| !w.is_finite() || *w <= 0.0) {
        return Err("a vertex solid angle is not positive".into());
    }
    Ok(weights)
}

pub fn evaluate_mass_gradient<D>(
    radii: &[f64],
    directions: &[[f64; 3]],
    weights: &[f64],
    density_fn: &D,
) -> Vec<f64>
where
    D: Fn([f64; 3]) -> f64,
{
    let n = radii.len();
    let mut grad = Vec::with_capacity(n);
    for i in 0..n {
        let r_i = radii[i].max(0.0);
        let u = directions[i];
        let pos = [r_i * u[0], r_i * u[1], r_i * u[2]];
        let rho = density_fn(pos);
        let d_m_i = weights[i] * rho * r_i * r_i;
        grad.push(d_m_i);
    }
    grad
}
#[derive(Debug, Clone)]
pub struct LinearizedMassConstraint {
    pub coefficients: Vec<f64>,
    pub target_val: f64,
    pub current_mass: f64,
}
pub fn linearize_mass_constraint<D>(
    current_radii: &[f64],
    directions: &[[f64; 3]],
    weights: &[f64],
    density_fn: &D,
    target_mass: f64,
) -> LinearizedMassConstraint
where
    D: Fn([f64; 3]) -> f64,
{
    let current_mass = evaluate_mass_at_radii(current_radii, directions, weights, density_fn);
    let grad = evaluate_mass_gradient(current_radii, directions, weights, density_fn);
    let mut sum_a_r = 0.0;
    for (i, &r_i) in current_radii.iter().enumerate() {
        sum_a_r += grad[i] * r_i;
    }
    let b = target_mass - current_mass + sum_a_r;
    LinearizedMassConstraint {
        coefficients: grad,
        target_val: b,
        current_mass,
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;
    #[test]
    fn test_mass_gradient_accuracy() {
        let r0 = 100.0;
        let rho0 = 2500.0;
        let density_fn = |_pos: [f64; 3]| -> f64 { rho0 };
        let dirs = vec![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let weights = vec![2.0 * PI, 2.0 * PI];
        let radii = vec![r0, r0];
        let grad = evaluate_mass_gradient(&radii, &dirs, &weights, &density_fn);
        let expected_grad_0 = 2.0 * PI * rho0 * r0 * r0;
        assert!((grad[0] - expected_grad_0).abs() < 1e-6);
        let eps = 1e-4;
        let mut radii_perturbed = radii.clone();
        radii_perturbed[0] += eps;
        let m1 = evaluate_mass_at_radii(&radii_perturbed, &dirs, &weights, &density_fn);
        let m0 = evaluate_mass_at_radii(&radii, &dirs, &weights, &density_fn);
        let num_grad = (m1 - m0) / eps;
        assert!((grad[0] - num_grad).abs() / grad[0] < 1e-3);
    }
}
