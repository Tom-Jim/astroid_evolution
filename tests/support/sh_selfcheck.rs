//! Acceptance checks. `sh-cli check` prints them; `tests/sh_smooth.rs` asserts them.

use std::f64::consts::PI;

use astroid_evolution::common::error::SimError;
use astroid_evolution::shsmooth::eval;
use astroid_evolution::shsmooth::filter::Filter;
use astroid_evolution::shsmooth::fit;
use astroid_evolution::shsmooth::gpu::ShGpu;
use astroid_evolution::shsmooth::harmonics::{self, fibonacci_directions, lm_index, sh_all};
use astroid_evolution::shsmooth::mesh::{Center, Mesh};
use astroid_evolution::shsmooth::smooth::{self, SmoothOptions};

#[derive(Clone, Debug)]
pub struct Check {
    pub name: String,
    pub value: f64,
    pub limit: f64,
}

impl Check {
    pub fn passed(&self) -> bool {
        self.value.is_finite() && self.value <= self.limit
    }
}

fn check(name: &str, value: f64, limit: f64) -> Check {
    Check {
        name: name.to_string(),
        value,
        limit,
    }
}

fn unit(d: [f64; 3]) -> [f64; 3] {
    let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    [d[0] / len, d[1] / len, d[2] / len]
}

pub fn closed_form() -> Check {
    let d = unit([0.3, -0.4, 0.8]);
    let y = sh_all(2, d);
    let y00 = (1.0 / (4.0 * PI)).sqrt();
    let y10 = (3.0 / (4.0 * PI)).sqrt() * d[2];
    let y11 = (3.0 / (4.0 * PI)).sqrt() * d[0];
    let y1m = (3.0 / (4.0 * PI)).sqrt() * d[1];
    let y20 = (5.0 / (16.0 * PI)).sqrt() * (3.0 * d[2] * d[2] - 1.0);
    let err = (y[lm_index(0, 0)] - y00)
        .abs()
        .max((y[lm_index(1, 0)] - y10).abs())
        .max((y[lm_index(1, 1)] - y11).abs())
        .max((y[lm_index(1, -1)] - y1m).abs())
        .max((y[lm_index(2, 0)] - y20).abs());
    check("closed-form Y_lm through degree 2", err, 1e-12)
}

pub fn orthonormality() -> Check {
    let n = 4000;
    let dirs = fibonacci_directions(n);
    let w = 4.0 * PI / n as f64;
    let l_max = 2;
    let ys: Vec<Vec<f64>> = dirs.iter().copied().map(|d| sh_all(l_max, d)).collect();
    let m = ys[0].len();
    let mut err = 0.0_f64;
    for a in 0..m {
        for b in a..m {
            let mut s = 0.0;
            for y in &ys {
                s += w * y[a] * y[b];
            }
            let expect = if a == b { 1.0 } else { 0.0 };
            err = err.max((s - expect).abs());
        }
    }
    check("Fibonacci quadrature of Y_lm, l <= 2", err, 2e-3)
}

pub fn coefficient_recovery() -> Result<Check, SimError> {
    let l_max = 3;
    let mut coeff = vec![0.0; harmonics::coeff_count(l_max)];
    coeff[lm_index(0, 0)] = 2.0;
    coeff[lm_index(1, 0)] = -0.15;
    coeff[lm_index(2, 2)] = 0.08;
    coeff[lm_index(2, -1)] = 0.05;
    coeff[lm_index(3, -3)] = -0.02;
    let dirs = fibonacci_directions(800);
    let radii: Vec<f64> = dirs
        .iter()
        .copied()
        .map(|d| eval::radius(l_max, &coeff, d))
        .collect();
    let weights = vec![1.0; dirs.len()];
    let got = fit::fit(l_max, &dirs, &radii, &weights, 1e-14)?;
    let mut err = 0.0_f64;
    for (a, b) in coeff.iter().zip(&got.coeff) {
        err = err.max((a - b).abs());
    }
    Ok(check(
        "band-limited least squares recovers coefficients",
        err,
        1e-7,
    ))
}

pub fn filter_gains() -> Check {
    let heat = Filter::Heat { kappa: 0.0 };
    let trunc = Filter::Truncate { l_cut: 2 };
    let cosine = Filter::Cosine { l0: 2.0, l1: 4.0 };
    let gauss = Filter::Gaussian { sigma: 2.0 };
    let mut err = 0.0_f64;
    err = err.max((heat.gain(5) - 1.0).abs());
    err = err.max((trunc.gain(2) - 1.0).abs());
    err = err.max(trunc.gain(3).abs());
    err = err.max((cosine.gain(2) - 1.0).abs());
    err = err.max((cosine.gain(3) - 0.5).abs());
    err = err.max(cosine.gain(4).abs());
    err = err.max((gauss.gain(0) - 1.0).abs());
    err = err.max((gauss.gain(2) - (-1.0_f64).exp()).abs());
    let mut coeff = vec![0.0; harmonics::coeff_count(4)];
    coeff[lm_index(4, 1)] = 3.0;
    let filtered = Filter::Truncate { l_cut: 2 }.apply(4, &coeff);
    err = err.max(filtered[lm_index(4, 1)].abs());
    check("spectral gains", err, 1e-15)
}

fn lumpy(l_max: usize) -> Vec<f64> {
    let mut c = vec![0.0; harmonics::coeff_count(l_max)];
    c[0] = 1.5 * (4.0 * PI).sqrt();
    let specs = [
        (1, 0, 0.05),
        (2, 1, -0.04),
        (2, -2, 0.03),
        (3, 2, 0.02),
        (4, -3, -0.015),
        (5, 1, 0.01),
        (6, 4, -0.008),
    ];
    for (l, m, v) in specs {
        if l <= l_max {
            c[lm_index(l, m)] = v;
        }
    }
    c
}

pub fn gpu_matches_reference(gpu: &ShGpu) -> Result<Check, SimError> {
    let dirs = fibonacci_directions(128);
    let mut err = 0.0_f64;
    for l_max in 0..=6 {
        let coeff = lumpy(l_max);
        let got = gpu.evaluate(l_max, &coeff, &dirs)?;
        if got.nonpositive != 0 {
            return Ok(check(
                "device radius stays positive on the lumpy expansion",
                got.nonpositive as f64,
                0.0,
            ));
        }
        for (dir, &radius) in dirs.iter().zip(&got.radii) {
            let reference = eval::radius(l_max, &coeff, *dir);
            err = err.max((radius - reference).abs() / reference.abs());
        }
    }
    Ok(check(
        "device double-single radius matches f64, l <= 6",
        err,
        1e-5,
    ))
}

fn ripple(theta: f64, phi: f64) -> f64 {
    1.0 + 0.08 * theta.sin().powi(6) * (6.0 * phi).cos()
}

fn radius_deviation(mesh: &Mesh) -> f64 {
    let radii: Vec<f64> = mesh
        .vertices
        .iter()
        .map(|v| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt())
        .collect();
    let mean = radii.iter().sum::<f64>() / radii.len() as f64;
    radii.iter().fold(0.0, |m, r| m.max((r - mean).abs()))
}

fn max_relative(before: &Mesh, after: &Mesh) -> f64 {
    before
        .vertices
        .iter()
        .zip(&after.vertices)
        .map(|(a, b)| {
            let ra = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
            let rb = (b[0] * b[0] + b[1] * b[1] + b[2] * b[2]).sqrt();
            (ra - rb).abs() / ra
        })
        .fold(0.0, f64::max)
}

pub fn faces_and_identity(gpu: &ShGpu) -> Result<Vec<Check>, SimError> {
    let mut mesh = Mesh::from_radial(10, ripple)?;
    mesh.length_unit = "m".to_string();
    let options = SmoothOptions {
        l_max: 6,
        filter: Filter::Heat { kappa: 0.0 },
        center: Center::Origin,
        ridge: 1e-12,
        ..SmoothOptions::default()
    };
    let out = smooth::smooth(&mesh, &options, gpu)?;
    let faces_differ = f64::from(out.mesh.faces != mesh.faces)
        + f64::from(out.mesh.face_count() != mesh.face_count())
        + f64::from(out.mesh.vertex_count() != mesh.vertex_count())
        + f64::from(out.mesh.face_lines() != mesh.face_lines());
    let dir = std::env::temp_dir();
    let tag = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let input = dir.join(format!("sh_smooth_in_{tag}.obj"));
    let output = dir.join(format!("sh_smooth_out_{tag}.obj"));
    mesh.write_obj(&input, "")?;
    out.mesh.write_obj(&output, &options.filter.label())?;
    let in_text = std::fs::read_to_string(&input).unwrap_or_default();
    let out_text = std::fs::read_to_string(&output).unwrap_or_default();
    let in_faces: Vec<_> = in_text.lines().filter(|l| l.starts_with("f ")).collect();
    let out_faces: Vec<_> = out_text.lines().filter(|l| l.starts_with("f ")).collect();
    let text_differ = f64::from(in_faces != out_faces);
    let _ = std::fs::remove_file(&input);
    let _ = std::fs::remove_file(&output);
    let back = Mesh::parse_obj(&out_text)?;
    let parsed_differ = f64::from(back.faces != mesh.faces);
    Ok(vec![
        check(
            "output face list and OBJ face lines match the input",
            faces_differ + text_differ + parsed_differ,
            0.0,
        ),
        check(
            "identity filter reproduces a degree-6 radius",
            max_relative(&mesh, &out.mesh),
            1e-4,
        ),
    ])
}

pub fn heat_attenuates(gpu: &ShGpu) -> Result<Check, SimError> {
    let mesh = Mesh::from_radial(10, ripple)?;
    let before = radius_deviation(&mesh);
    let options = SmoothOptions {
        l_max: 8,
        filter: Filter::Heat { kappa: 0.25 },
        center: Center::Origin,
        ridge: 1e-12,
        ..SmoothOptions::default()
    };
    let out = smooth::smooth(&mesh, &options, gpu)?;
    let after = radius_deviation(&out.mesh);
    let ratio = after / before;
    Ok(check(
        "heat kernel attenuates the degree-6 ripple",
        ratio,
        0.02,
    ))
}

pub fn truncate_projects_to_mean(gpu: &ShGpu) -> Result<Check, SimError> {
    let mesh = Mesh::from_radial(10, ripple)?;
    let options = SmoothOptions {
        l_max: 6,
        filter: Filter::Truncate { l_cut: 0 },
        center: Center::Origin,
        ridge: 1e-12,
        ..SmoothOptions::default()
    };
    let out = smooth::smooth(&mesh, &options, gpu)?;
    let err = out
        .mesh
        .vertices
        .iter()
        .map(|v| ((v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt() - 1.0).abs())
        .fold(0.0, f64::max);
    let faces_ok = out.mesh.faces == mesh.faces;
    Ok(check(
        "degree-0 truncation leaves the mean radius and the faces",
        err + f64::from(!faces_ok),
        1e-4,
    ))
}

pub fn sphere_is_fixed(gpu: &ShGpu) -> Result<Check, SimError> {
    let mesh = Mesh::from_radial(8, |_, _| 2.0)?;
    let options = SmoothOptions {
        l_max: 4,
        filter: Filter::Heat { kappa: 0.05 },
        center: Center::Origin,
        ridge: 1e-12,
        ..SmoothOptions::default()
    };
    let out = smooth::smooth(&mesh, &options, gpu)?;
    Ok(check(
        "a sphere is unchanged by the heat filter",
        max_relative(&mesh, &out.mesh),
        1e-5,
    ))
}

fn vertex_radius(v: &[f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn neighbor_mean_radius(mesh: &Mesh, index: usize) -> f64 {
    let mut seen = vec![false; mesh.vertices.len()];
    let mut sum = 0.0_f64;
    let mut count = 0.0_f64;
    for face in &mesh.faces {
        if !face.contains(&(index as u32)) {
            continue;
        }
        for &corner in face {
            let corner = corner as usize;
            if corner != index && !seen[corner] {
                seen[corner] = true;
                sum += vertex_radius(&mesh.vertices[corner]);
                count += 1.0;
            }
        }
    }
    sum / count
}

fn punch_crater(mesh: &mut Mesh, depth: f64) -> usize {
    let crater = mesh
        .vertices
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a[0].total_cmp(&b[0]))
        .map(|(i, _)| i)
        .expect("mesh has a vertex");
    let radius = vertex_radius(&mesh.vertices[crater]);
    let scale = (radius - depth) / radius;
    for coord in &mut mesh.vertices[crater] {
        *coord *= scale;
    }
    crater
}

pub fn gpu_adds_residual(gpu: &ShGpu) -> Result<Check, SimError> {
    let l_max = 4;
    let coeff = lumpy(l_max);
    let dirs = fibonacci_directions(96);
    let residual: Vec<f64> = (0..dirs.len())
        .map(|i| 0.02 * ((i % 7) as f64) - 0.05)
        .collect();
    let got = gpu.evaluate_with_residual(l_max, &coeff, &dirs, &residual)?;
    let mut err = 0.0_f64;
    for (i, dir) in dirs.iter().enumerate() {
        let reference = eval::radius(l_max, &coeff, *dir) + residual[i];
        err = err.max((got.radii[i] - reference).abs() / reference.abs());
    }
    Ok(check("device adds the high-frequency residual", err, 1e-5))
}

pub fn crater_survives_the_macro_filter(gpu: &ShGpu) -> Result<Check, SimError> {
    let mut mesh = Mesh::from_radial(16, |_, _| 1.4)?;
    let crater = punch_crater(&mut mesh, 0.1);
    let depth_in = neighbor_mean_radius(&mesh, crater) - vertex_radius(&mesh.vertices[crater]);
    let options = SmoothOptions {
        l_max: 8,
        filter: Filter::Heat { kappa: 0.25 },
        center: Center::Origin,
        ridge: 1e-12,
        ..SmoothOptions::default()
    };
    let out = smooth::smooth(&mesh, &options, gpu)?;
    let depth_out =
        neighbor_mean_radius(&out.mesh, crater) - vertex_radius(&out.mesh.vertices[crater]);
    let lost = (depth_in - depth_out).max(0.0) / depth_in;
    let faces = f64::from(out.mesh.faces != mesh.faces);
    Ok(check(
        "heat on the macro surface keeps the crater residual",
        lost + faces,
        0.25,
    ))
}

pub fn equatorial_ridge_attenuation_keeps_the_crater(gpu: &ShGpu) -> Result<Vec<Check>, SimError> {
    let mut mesh = Mesh::from_radial(16, |theta, _| {
        let z = theta.cos();
        let p4 = (35.0 * z.powi(4) - 30.0 * z * z + 3.0) / 8.0;
        1.4 + 0.12 * p4
    })?;
    let crater = punch_crater(&mut mesh, 0.1);
    let depth_in = neighbor_mean_radius(&mesh, crater) - vertex_radius(&mesh.vertices[crater]);
    let pole_in = vertex_radius(&mesh.vertices[0]);
    let options = SmoothOptions {
        l_max: 8,
        filter: Filter::Heat { kappa: 0.0 },
        center: Center::Origin,
        ridge: 1e-12,
        y40_gain: 0.0,
        ..SmoothOptions::default()
    };
    let out = smooth::smooth(&mesh, &options, gpu)?;
    let depth_out =
        neighbor_mean_radius(&out.mesh, crater) - vertex_radius(&out.mesh.vertices[crater]);
    let pole_shift = (vertex_radius(&out.mesh.vertices[0]) - pole_in).abs();
    Ok(vec![
        check(
            "Y40 removal keeps the crater depth",
            (depth_in - depth_out).abs() / depth_in,
            0.25,
        ),
        check(
            "Y40 removal shifts the pole",
            (0.02 - pole_shift).max(0.0),
            0.0,
        ),
        check(
            "ancient Y40 coefficient is removed",
            out.report.y40_ancient.abs(),
            1e-15,
        ),
        check(
            "macro fit still saw the equatorial ridge",
            (0.02 - out.report.y40_macro.abs()).max(0.0),
            0.0,
        ),
    ])
}

fn oblate_ryugu(theta: f64, _phi: f64) -> f64 {
    let z = theta.cos();
    440.0 + 62.0 * (1.0 - z * z)
}

pub fn y20_attenuation_cuts_oblateness(gpu: &ShGpu) -> Result<Check, SimError> {
    let mesh = Mesh::from_radial(16, oblate_ryugu)?;
    let options = SmoothOptions {
        l_max: 4,
        filter: Filter::Heat { kappa: 0.0 },
        center: Center::Origin,
        y20_gain: 0.3,
        ..SmoothOptions::default()
    };
    let out = smooth::smooth(&mesh, &options, gpu)?;
    let gap_in = out.report.equator_radius_in - out.report.pole_radius_in;
    let gap_out = out.report.equator_radius_out - out.report.pole_radius_out;
    Ok(check(
        "Y20 gain 0.3 leaves 30% of the pole-equator gap",
        (gap_out / gap_in - 0.3).abs(),
        0.05,
    ))
}

pub fn equator_drop_lowers_only_the_belt(gpu: &ShGpu) -> Result<Vec<Check>, SimError> {
    let mesh = Mesh::from_radial(16, |_, _| 500.0)?;
    let options = SmoothOptions {
        l_max: 4,
        filter: Filter::Heat { kappa: 0.0 },
        center: Center::Origin,
        equator_drop: 45.0,
        ..SmoothOptions::default()
    };
    let out = smooth::smooth(&mesh, &options, gpu)?;
    Ok(vec![
        check(
            "45 m cosine leaves the pole",
            (out.report.pole_radius_out - 500.0).abs(),
            0.05,
        ),
        check(
            "45 m cosine lowers the equator to 455 m",
            (out.report.equator_radius_out - 455.0).abs(),
            0.05,
        ),
        check(
            "equator drop keeps the face list",
            f64::from(out.mesh.faces != mesh.faces),
            0.0,
        ),
    ])
}

pub fn equatorial_detail_is_cut_and_polar_craters_stay(
    gpu: &ShGpu,
) -> Result<Vec<Check>, SimError> {
    let mut mesh = Mesh::from_radial(20, |_, _| 500.0)?;
    let equator = mesh
        .vertices
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a[0].total_cmp(&b[0]))
        .map(|(i, _)| i)
        .expect("equatorial vertex");
    let pole = 0usize;
    let eq_r = vertex_radius(&mesh.vertices[equator]);
    let pole_r = vertex_radius(&mesh.vertices[pole]);
    let eq_scale = (eq_r + 30.0) / eq_r;
    for coord in &mut mesh.vertices[equator] {
        *coord *= eq_scale;
    }
    let pole_scale = (pole_r - 20.0) / pole_r;
    for coord in &mut mesh.vertices[pole] {
        *coord *= pole_scale;
    }
    let spike_in = vertex_radius(&mesh.vertices[equator]) - neighbor_mean_radius(&mesh, equator);
    let crater_in = neighbor_mean_radius(&mesh, pole) - vertex_radius(&mesh.vertices[pole]);
    let options = SmoothOptions {
        l_max: 6,
        filter: Filter::Heat { kappa: 0.0 },
        center: Center::Origin,
        detail_gain: 0.2,
        ..SmoothOptions::default()
    };
    let out = smooth::smooth(&mesh, &options, gpu)?;
    let spike_out =
        vertex_radius(&out.mesh.vertices[equator]) - neighbor_mean_radius(&out.mesh, equator);
    let crater_out =
        neighbor_mean_radius(&out.mesh, pole) - vertex_radius(&out.mesh.vertices[pole]);
    let spike_ratio = spike_out / spike_in;
    let crater_ratio = crater_out / crater_in;
    Ok(vec![
        check(
            "equatorial sharp ridge is not returned whole",
            (spike_ratio - 0.55).max(0.0),
            0.0,
        ),
        check(
            "polar crater detail stays",
            (crater_ratio - 1.0).abs(),
            0.05,
        ),
    ])
}

pub fn ancient_rollback_lowers_the_equator(gpu: &ShGpu) -> Result<Vec<Check>, SimError> {
    let mesh = Mesh::from_radial(16, oblate_ryugu)?;
    let options = SmoothOptions {
        center: Center::Origin,
        ..SmoothOptions::ancient_ryugu()
    };
    let out = smooth::smooth(&mesh, &options, gpu)?;
    let y20_ratio = if out.report.y20_macro.abs() > 1e-8 {
        out.report.y20_ancient / out.report.y20_macro
    } else {
        0.0
    };
    Ok(vec![
        check(
            "ancient preset scales Y20 by 0.3",
            (y20_ratio - 0.3).abs(),
            1e-12,
        ),
        check(
            "ancient preset removes Y40",
            out.report.y40_ancient.abs(),
            1e-12,
        ),
        check(
            "ancient equator is well below 502 m",
            (out.report.equator_radius_out - 470.0).max(0.0),
            0.0,
        ),
        check(
            "ancient rollback keeps the face list",
            f64::from(out.mesh.faces != mesh.faces),
            0.0,
        ),
    ])
}
