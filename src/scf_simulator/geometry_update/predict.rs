use crate::common::math::DVec3;
use crate::scf_simulator::geometry_update::accept::STAR_MIN;
use crate::scf_simulator::geometry_update::mass::flux_density_weight;

/// Locked nodes stay at zero. Deposit nodes keep the predictor. The active set is the rest.
pub fn classify_nodes(
    predicted: &[f64],
    locked: &[bool],
    deposit: &[bool],
) -> (Vec<bool>, Vec<f64>) {
    let active: Vec<bool> = predicted
        .iter()
        .enumerate()
        .map(|(i, _)| !locked[i] && !deposit[i])
        .collect();
    let mut seed = predicted.to_vec();
    for (i, flag) in locked.iter().enumerate() {
        if *flag {
            seed[i] = 0.0;
        }
    }
    (active, seed)
}

/// Clarabel is not used here. Movable nodes belong to the frozen incremental SOCP.
pub fn solve_geometry(predicted: &[f64], active: &[bool], _trust: f64) -> Result<Vec<f64>, String> {
    if active.iter().any(|flag| *flag) {
        return Err("movable nodes are free variables of the frozen incremental SOCP".into());
    }
    Ok(predicted.to_vec())
}

/// Core with no flux is locked. Core with flux is deposit. Every yielded node is movable.
pub fn partition_nodes(
    in_core: &[bool],
    net_flux: &[f64],
) -> Result<(Vec<bool>, Vec<bool>, Vec<bool>), String> {
    if in_core.len() != net_flux.len() {
        return Err("node class lengths differ".into());
    }
    let n = in_core.len();
    let mut locked = vec![false; n];
    let mut deposit = vec![false; n];
    let mut movable = vec![false; n];
    for i in 0..n {
        if !net_flux[i].is_finite() {
            return Err("a node flux is not finite".into());
        }
        if in_core[i] && net_flux[i] == 0.0 {
            locked[i] = true;
        } else if in_core[i] {
            deposit[i] = true;
        } else {
            movable[i] = true;
        }
    }
    Ok((locked, deposit, movable))
}

/// `d_pred` from the slip term and `(ρ_bulk/ρ_K) Δh`. Locked nodes return `+0.0`.
pub fn predicted_radial_increment(
    movable: bool,
    deposit: bool,
    normal: DVec3,
    radial: DVec3,
    dt: f64,
    slip: DVec3,
    net_mass_rate: f64,
    area: f64,
    rho_bulk: f64,
    rho_k: f64,
) -> Result<f64, String> {
    if !movable && !deposit {
        return Ok(0.0);
    }
    let normal = normal
        .try_normalize()
        .ok_or("a surface normal is degenerate")?;
    let radial = radial
        .try_normalize()
        .ok_or("a radial direction is degenerate")?;
    let radial_margin = normal.dot(radial);
    if !radial_margin.is_finite() || radial_margin < STAR_MIN {
        return Err("a grazing node cannot take a radial increment".into());
    }
    let mut thickness = 0.0;
    if net_mass_rate != 0.0 {
        if !dt.is_finite() || !(area.is_finite() && area > 0.0) {
            return Err("a flux increment needs a finite step and a positive area".into());
        }
        let weight = flux_density_weight(rho_bulk, rho_k)?;
        let delta_h = dt * net_mass_rate / (area * rho_bulk);
        thickness = weight * delta_h;
    }
    let slip_term = if movable && slip.length_squared() > 0.0 {
        if !dt.is_finite() || !slip.is_finite() {
            return Err("slip increment is not finite".into());
        }
        dt * slip.dot(normal)
    } else {
        0.0
    };
    if slip_term == 0.0 && thickness == 0.0 {
        return Ok(0.0);
    }
    let increment = (slip_term + thickness) / radial_margin;
    if !increment.is_finite() {
        return Err("radial increment is not finite".into());
    }
    Ok(increment)
}

/// Normal displacement of a general triangle mesh. The radial chart is not used.
pub fn predicted_normal_increment(
    movable: bool,
    deposit: bool,
    normal: DVec3,
    dt: f64,
    slip: DVec3,
    net_mass_rate: f64,
    area: f64,
    rho_bulk: f64,
    rho_k: f64,
) -> Result<f64, String> {
    if !movable && !deposit {
        return Ok(0.0);
    }
    let normal = normal
        .try_normalize()
        .ok_or("a surface normal is degenerate")?;
    let mut thickness = 0.0;
    if net_mass_rate != 0.0 {
        if !dt.is_finite() || !(area.is_finite() && area > 0.0) {
            return Err("a flux increment needs a finite step and a positive area".into());
        }
        let weight = flux_density_weight(rho_bulk, rho_k)?;
        let delta_h = dt * net_mass_rate / (area * rho_bulk);
        thickness = weight * delta_h;
    }
    let slip_term = if movable && slip.length_squared() > 0.0 {
        if !dt.is_finite() || !slip.is_finite() {
            return Err("slip increment is not finite".into());
        }
        dt * slip.dot(normal)
    } else {
        0.0
    };
    let increment = slip_term + thickness;
    if !increment.is_finite() {
        return Err("normal increment is not finite".into());
    }
    Ok(increment)
}
