//! Host fit, macro ridge rollback, then a device evaluation.
//!
//! `detail = R_orig - R_macro`. The ancient coefficients attenuate `Y_2^0` and `Y_4^0`.
//! An equatorial cosine then lowers the macro radius. Equatorial detail is attenuated
//! so the sharp ridge is not pasted back. The device writes
//! `R_final = R_ancient_flattened + filtered_detail`.

use serde::Serialize;

use super::error::ShError;
use super::filter::Filter;
use super::fit::{self, Spectrum};
use super::gpu::ShGpu;
use super::harmonics::{degree_power, lm_index};
use super::mesh::{Center, Mesh};

#[derive(Clone, Debug)]
pub struct SmoothOptions {
    pub l_max: usize,
    pub filter: Filter,
    pub center: Center,
    pub ridge: f64,
    /// `0.3` keeps 30% of the `Y_2^0` oblateness (70% attenuation).
    pub y20_gain: f64,
    /// `0` removes `Y_4^0`. `1` leaves the filtered coefficient.
    pub y40_gain: f64,
    /// Metres (mesh units) subtracted from the macro radius at the equator.
    pub equator_drop: f64,
    /// Latitude, in degrees, at which the equatorial drop falls to zero.
    pub drop_latitude_deg: f64,
    /// `|latitude|` below this keeps `detail_gain`; above the blend, detail is unchanged.
    pub detail_latitude_deg: f64,
    /// Scale on high-frequency detail inside the equatorial belt. `0.2` cuts the sharp ridge.
    pub detail_gain: f64,
}

impl Default for SmoothOptions {
    fn default() -> Self {
        Self {
            l_max: 8,
            filter: Filter::Heat { kappa: 0.02 },
            center: Center::Auto,
            ridge: 1.0e-12,
            y20_gain: 1.0,
            y40_gain: 1.0,
            equator_drop: 0.0,
            drop_latitude_deg: 25.0,
            detail_latitude_deg: 15.0,
            detail_gain: 1.0,
        }
    }
}

impl SmoothOptions {
    /// Ridge-free ancient body: `Y20`×0.3, `Y40`×0, equator −45, equatorial detail ×0.2.
    ///
    /// The 45 m cosine is full on the equator and zero at ±25°. Detail is ×0.2 for
    /// `|lat| ≤ 15°` and returns to 1 by ±25°, so polar and mid-latitude craters stay.
    pub fn ancient_ryugu() -> Self {
        Self {
            filter: Filter::Heat { kappa: 0.0 },
            y20_gain: 0.3,
            y40_gain: 0.0,
            equator_drop: 45.0,
            detail_gain: 0.2,
            ..Self::default()
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct SmoothReport {
    pub device: String,
    pub input_vertices: usize,
    pub output_vertices: usize,
    pub input_faces: usize,
    pub output_faces: usize,
    pub faces_preserved: bool,
    pub l_max: usize,
    pub filter: String,
    pub center: [f64; 3],
    pub fit_power: Vec<f64>,
    pub filtered_power: Vec<f64>,
    pub y20_gain: f64,
    pub y20_macro: f64,
    pub y20_ancient: f64,
    pub y40_gain: f64,
    pub y40_macro: f64,
    pub y40_ancient: f64,
    pub equator_drop: f64,
    pub detail_gain: f64,
    pub residual_rms: f64,
    pub residual_abs_max: f64,
    pub pole_radius_in: f64,
    pub pole_radius_out: f64,
    pub equator_radius_in: f64,
    pub equator_radius_out: f64,
    pub max_relative_change: f64,
    pub nonpositive: u32,
    pub fit_seconds: f64,
    pub eval_seconds: f64,
    pub length_unit: String,
}

pub struct SmoothOutput {
    pub mesh: Mesh,
    pub report: SmoothReport,
    pub filtered: Spectrum,
}

pub fn smooth(mesh: &Mesh, options: &SmoothOptions, gpu: &ShGpu) -> Result<SmoothOutput, ShError> {
    options.filter.validate(options.l_max)?;
    validate_rollback(options)?;
    let center = options.center.resolve(mesh)?;
    let samples = mesh.radial_samples(center)?;
    let fit_started = std::time::Instant::now();
    let spectrum = fit::fit(
        options.l_max,
        &samples.dir,
        &samples.radius,
        &samples.weight,
        options.ridge,
    )?;
    let fit_seconds = fit_started.elapsed().as_secs_f64();
    let macro_eval = gpu.evaluate(options.l_max, &spectrum.coeff, &samples.dir)?;
    if macro_eval.radii.len() != samples.radius.len() {
        return Err(ShError::geometry(
            "device returned a different vertex count for the macro radius",
        ));
    }
    let residual: Vec<f64> = samples
        .radius
        .iter()
        .zip(&macro_eval.radii)
        .map(|(orig, macro_r)| orig - macro_r)
        .collect();
    if residual.iter().any(|r| !r.is_finite()) {
        return Err(ShError::geometry("high-frequency residual is not finite"));
    }
    let mut filtered_detail = Vec::with_capacity(residual.len());
    let mut transfer = Vec::with_capacity(residual.len());
    for (i, &detail) in residual.iter().enumerate() {
        let lat = latitude_deg(samples.dir[i]);
        let kept = detail * detail_belt_gain(lat, options.detail_latitude_deg, options.detail_gain);
        let drop = options.equator_drop * equator_window(lat, options.drop_latitude_deg);
        filtered_detail.push(kept);
        // Device adds this to the ancient harmonic radius:
        // R_ancient_flattened + filtered_detail = R_sh - drop + kept.
        transfer.push(kept - drop);
    }
    let ancient_coeff = options.filter.ancient(
        options.l_max,
        &spectrum.coeff,
        options.y20_gain,
        options.y40_gain,
    )?;
    let ancient_eval =
        gpu.evaluate_with_residual(options.l_max, &ancient_coeff, &samples.dir, &transfer)?;
    let eval = ancient_eval;
    let eval_seconds = macro_eval.seconds + eval.seconds;
    if eval.radii.len() != mesh.vertices.len() {
        return Err(ShError::geometry(
            "device returned a different vertex count",
        ));
    }
    let host_bad = eval
        .radii
        .iter()
        .filter(|r| !(r.is_finite() && **r > 0.0))
        .count() as u32;
    let nonpositive = eval.nonpositive.max(host_bad);
    if nonpositive > 0 {
        return Err(ShError::geometry(format!(
            "{nonpositive} vertices have a non-positive filtered radius; the face list was not rewritten"
        )));
    }
    let mut max_relative_change = 0.0_f64;
    let mut vertices = Vec::with_capacity(mesh.vertices.len());
    for i in 0..mesh.vertices.len() {
        let old = samples.radius[i];
        let new = eval.radii[i];
        max_relative_change = max_relative_change.max((new - old).abs() / old);
        let d = samples.dir[i];
        vertices.push([
            center[0] + new * d[0],
            center[1] + new * d[1],
            center[2] + new * d[2],
        ]);
    }
    let out = Mesh {
        vertices,
        faces: mesh.faces.clone(),
        length_unit: mesh.length_unit.clone(),
    };
    if out.faces != mesh.faces || out.face_count() != mesh.face_count() {
        return Err(ShError::geometry(
            "internal error: output faces differ from the input",
        ));
    }
    let filtered = Spectrum {
        l_max: options.l_max,
        coeff: ancient_coeff,
    };
    let n = filtered_detail.len() as f64;
    let residual_rms = (filtered_detail.iter().map(|r| r * r).sum::<f64>() / n).sqrt();
    let residual_abs_max = filtered_detail.iter().fold(0.0_f64, |m, r| m.max(r.abs()));
    let zonal = |coeff: &[f64], degree: usize| {
        if options.l_max >= degree {
            coeff[lm_index(degree, 0)]
        } else {
            0.0
        }
    };
    let (pole, equator) = pole_and_equator(&samples.dir);
    let report = SmoothReport {
        device: gpu.device_name(),
        input_vertices: mesh.vertex_count(),
        output_vertices: out.vertex_count(),
        input_faces: mesh.face_count(),
        output_faces: out.face_count(),
        faces_preserved: out.faces == mesh.faces,
        l_max: options.l_max,
        filter: options.filter.label(),
        center,
        fit_power: degree_power(options.l_max, &spectrum.coeff),
        filtered_power: degree_power(options.l_max, &filtered.coeff),
        y20_gain: options.y20_gain,
        y20_macro: zonal(&spectrum.coeff, 2),
        y20_ancient: zonal(&filtered.coeff, 2),
        y40_gain: options.y40_gain,
        y40_macro: zonal(&spectrum.coeff, 4),
        y40_ancient: zonal(&filtered.coeff, 4),
        equator_drop: options.equator_drop,
        detail_gain: options.detail_gain,
        residual_rms,
        residual_abs_max,
        pole_radius_in: samples.radius[pole],
        pole_radius_out: eval.radii[pole],
        equator_radius_in: samples.radius[equator],
        equator_radius_out: eval.radii[equator],
        max_relative_change,
        nonpositive,
        fit_seconds,
        eval_seconds,
        length_unit: mesh.length_unit.clone(),
    };
    Ok(SmoothOutput {
        mesh: out,
        report,
        filtered,
    })
}

fn validate_rollback(options: &SmoothOptions) -> Result<(), ShError> {
    for (name, gain) in [
        ("Y20", options.y20_gain),
        ("Y40", options.y40_gain),
        ("equatorial detail", options.detail_gain),
    ] {
        if !(gain.is_finite() && (0.0..=1.0).contains(&gain)) {
            return Err(ShError::invalid_input(format!(
                "{name} gain must be a finite number in [0, 1]"
            )));
        }
    }
    if !(options.equator_drop.is_finite() && options.equator_drop >= 0.0) {
        return Err(ShError::invalid_input(
            "equator drop must be a finite number >= 0",
        ));
    }
    for (name, lat) in [
        ("drop latitude", options.drop_latitude_deg),
        ("detail latitude", options.detail_latitude_deg),
    ] {
        if !(lat.is_finite() && lat > 0.0 && lat < 90.0) {
            return Err(ShError::invalid_input(format!(
                "{name} must be a finite number in (0, 90) degrees"
            )));
        }
    }
    Ok(())
}

fn latitude_deg(dir: [f64; 3]) -> f64 {
    dir[2].clamp(-1.0, 1.0).asin().to_degrees()
}

/// 1 on the equator, 0 at `±half_width_deg` and beyond. Raised cosine, so the
/// flank meets mid-latitudes with zero slope.
fn equator_window(lat_deg: f64, half_width_deg: f64) -> f64 {
    let a = lat_deg.abs();
    if a >= half_width_deg {
        0.0
    } else {
        0.5 * (1.0 + (std::f64::consts::PI * a / half_width_deg).cos())
    }
}

/// `gain` for `|lat| ≤ belt`, 1 for `|lat| ≥ belt + 10°`, cosine between.
fn detail_belt_gain(lat_deg: f64, belt_deg: f64, gain: f64) -> f64 {
    let a = lat_deg.abs();
    let outer = belt_deg + 10.0;
    if a <= belt_deg {
        gain
    } else if a >= outer {
        1.0
    } else {
        let t = (a - belt_deg) / (outer - belt_deg);
        let blend = 0.5 * (1.0 - (std::f64::consts::PI * t).cos());
        gain + (1.0 - gain) * blend
    }
}

fn pole_and_equator(dirs: &[[f64; 3]]) -> (usize, usize) {
    let mut pole = 0usize;
    let mut equator = 0usize;
    for i in 1..dirs.len() {
        if dirs[i][2].abs() > dirs[pole][2].abs() {
            pole = i;
        }
        if dirs[i][2].abs() < dirs[equator][2].abs() {
            equator = i;
        }
    }
    (pole, equator)
}
