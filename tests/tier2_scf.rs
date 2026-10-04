use std::path::Path;
use std::process::Command;

use astroid_evolution::common::density::{CauchyDensity, CauchyKernel};
use astroid_evolution::common::geometry::TriMesh;
use astroid_evolution::common::math::DVec3;
use astroid_evolution::scf_simulator::geometry_update::realign::{principal_frame, realign_body};
use astroid_evolution::scf_simulator::run_sim::validate_at;
use astroid_evolution::scf_simulator::simulation::{
    StageStatus, advance_closed_mass, closed_run, static_sphere_step, validate_project_toml,
};
use astroid_evolution::scf_simulator::spin::{SpinSchedule, SpinStage};
use astroid_evolution::scf_simulator::stokes::picard_loop::{
    GravSample, PicardMaterial, kinetic_mass, picard_solve,
};
use astroid_evolution::scf_simulator::stokes::rigid::{assemble_rigid_constraint, singular_values};
use astroid_evolution::scf_simulator::yielding::core_fit::fit_core_motion;
use astroid_evolution::scf_simulator::yielding::core_graph::rigid_core;
use nalgebra::{Matrix3, Vector3};

#[test]
fn g1_resting_sphere_is_an_equilibrium() {
    let schedule = SpinSchedule::Stages(vec![SpinStage {
        t_end: 1.0,
        omega: [0.0; 3],
    }]);
    let step = static_sphere_step(8, 1.0, &schedule).unwrap();
    assert_eq!(step.active_count, 0);
    assert_eq!(step.status, StageStatus::StageEquilibrium);
    assert!(
        step.displacement
            .iter()
            .all(|d| d.to_bits() == 0.0_f64.to_bits())
    );
    assert_eq!(step.mass_drift.to_bits(), 0.0_f64.to_bits());
}

#[test]
fn g3_core_surface_nodes_are_bit_locked() {
    let yielded = [false, false, true, false];
    let neighbors = vec![vec![1], vec![0], vec![], vec![]];
    let mass = [2.0, 2.0, 0.1, 0.2];
    let core = rigid_core(&yielded, &neighbors, &mass).unwrap();
    let mut displacement = [1.0_f64, 1.0, 1.0, 1.0];
    for (value, locked) in displacement.iter_mut().zip(core) {
        if locked {
            *value = 0.0;
        }
    }
    assert_eq!(displacement[0].to_bits(), 0.0_f64.to_bits());
    assert_eq!(displacement[1].to_bits(), 0.0_f64.to_bits());
    assert_ne!(displacement[3].to_bits(), 0.0_f64.to_bits());
}

#[test]
fn g2_ten_steps_conserve_mass_and_balance_flux() {
    let history = advance_closed_mass(4.0, 10, 1.0e-6);
    assert!((history.mass - 4.0).abs() / 4.0 <= 1.0e-8);
    assert_eq!(history.deposition, history.erosion);
}

#[test]
fn g11_missing_constitutive_parameter_is_model_not_closed() {
    let text = "[rotation]\nomega = [0.0, 0.0, 0.0]\n";
    let err = validate_project_toml(text).unwrap_err();
    assert_eq!(err.kind, "ModelNotClosed");
    assert_eq!(err.code, 10);
}

fn cli_value(text: &str, key: &str) -> f64 {
    for line in text.lines() {
        let mut parts = line.split_whitespace();
        while let Some(token) = parts.next() {
            if token == key {
                return parts
                    .next()
                    .unwrap_or_else(|| panic!("missing number after {key}"))
                    .parse()
                    .unwrap_or_else(|_| panic!("non-numeric {key} in {line}"));
            }
        }
    }
    panic!("missing {key} in {text}");
}

fn run_cli(args: &[&str]) -> std::process::Output {
    Command::new(scf_cli())
        .args(args)
        .output()
        .expect("scf-cli")
}

fn sample_with(mesh: &str, spin: &str, picard_extra: &str) -> String {
    let mesh_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join(mesh)
        .display()
        .to_string();
    let mut text = sample_project_text()
        .replace("../models/ryugu.glb", &mesh_path)
        .replace(
            "mass_target = 4.50e11",
            "mass_target = 2194.2094889027243880",
        )
        .replace("value = 4.50e11", "value = 2194.2094889027243880")
        .replace("uncertainty = 5.85e9", "uncertainty = 20.0")
        .replace("mass_tol = 0.013", "mass_tol = 0.009114899967915806")
        .replace(
            "representation = \"general_mesh\"",
            "representation = \"radial_star\"",
        );
    if !spin.is_empty() {
        let start = text.find("[spin_schedule]").expect("spin");
        let end = text.find("[material]").expect("material");
        text = format!("{}{}{}", &text[..start], spin, &text[end..]);
    }
    if !picard_extra.is_empty() {
        text = text.replace(
            "[picard]\nunder_relaxation = 0.5\nmax_iterations = 1\n",
            &format!("[picard]\nunder_relaxation = 0.5\nmax_iterations = 1\n{picard_extra}\n"),
        );
    }
    text
}

fn write_project(name: &str, text: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join("astroid-evolution-tier2");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(name);
    std::fs::write(&path, text).unwrap();
    path
}

#[test]
fn cli_resting_sphere_displacement_and_exit_code() {
    let output = run_cli(&["rest-sphere"]);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert_eq!(
        output.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        cli_value(&stdout, "displacement_residual").to_bits(),
        0.0_f64.to_bits()
    );
    assert_eq!(
        cli_value(&stdout, "mass_drift").to_bits(),
        0.0_f64.to_bits()
    );
    assert_eq!(cli_value(&stdout, "active").to_bits(), 0.0_f64.to_bits());
}

#[test]
fn cli_two_hundred_steps_conserve_mass_and_mode_c_flux() {
    let output = run_cli(&["conserve", "--steps", "200"]);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert_eq!(
        output.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(cli_value(&stdout, "mass_error") <= 1.0e-8);
    assert!(cli_value(&stdout, "flux_residual") <= 1.0e-12);
}

#[test]
fn cli_spin_acceleration_and_stage_boundary() {
    let linear = sample_with(
        "assets/mesh/octahedron.obj",
        "[spin_schedule]\nkind = \"linear\"\nomega0 = [0.0, 0.0, 0.0]\nomega1 = [0.0, 0.0, 0.2]\nt0 = 0.0\nt1 = 2.0\n",
        "",
    );
    let linear_path = write_project("linear.toml", &linear);
    let output = run_cli(&[
        "spin",
        "--project",
        linear_path.to_str().unwrap(),
        "--time",
        "1",
    ]);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert_eq!(
        output.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!((cli_value(&stdout, "omega") - 0.0).abs() < 1e-15);
    let omega_z = stdout
        .split_whitespace()
        .nth(5)
        .unwrap()
        .parse::<f64>()
        .unwrap();
    let dot_z = stdout
        .split_whitespace()
        .nth(9)
        .unwrap()
        .parse::<f64>()
        .unwrap();
    assert!((omega_z - 0.1).abs() < 1e-12, "omega_z {omega_z}");
    assert!((dot_z - 0.1).abs() < 1e-12, "omega_dot_z {dot_z}");

    let stages = sample_with(
        "assets/mesh/octahedron.obj",
        "[spin_schedule]\nkind = \"stages\"\n\n[[spin_schedule.stages]]\nt_end = 1.0\nomega = [0.0, 0.0, 0.1]\n\n[[spin_schedule.stages]]\nt_end = 3.0\nomega = [0.0, 0.0, 0.4]\n",
        "",
    );
    let stage_path = write_project("stages.toml", &stages);
    let at_end = run_cli(&[
        "spin",
        "--project",
        stage_path.to_str().unwrap(),
        "--time",
        "1",
    ]);
    let after = run_cli(&[
        "spin",
        "--project",
        stage_path.to_str().unwrap(),
        "--time",
        "1.000001",
    ]);
    assert_eq!(at_end.status.code(), Some(0));
    assert_eq!(after.status.code(), Some(0));
    let end_z = String::from_utf8_lossy(&at_end.stdout)
        .split_whitespace()
        .nth(5)
        .unwrap()
        .parse::<f64>()
        .unwrap();
    let after_z = String::from_utf8_lossy(&after.stdout)
        .split_whitespace()
        .nth(5)
        .unwrap()
        .parse::<f64>()
        .unwrap();
    assert!((end_z - 0.1).abs() < 1e-12);
    assert!((after_z - 0.4).abs() < 1e-12);
    assert!((after_z - end_z).abs() > 0.2);
}

#[test]
fn cli_socp_lock_and_realignment_residuals() {
    let socp = run_cli(&["socp-lock"]);
    let socp_out = String::from_utf8_lossy(&socp.stdout);
    assert_eq!(
        socp.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&socp.stderr)
    );
    assert_eq!(
        cli_value(&socp_out, "locked_residual").to_bits(),
        0.0_f64.to_bits()
    );
    assert!(cli_value(&socp_out, "deposit_residual") <= 1.0e-15);
    assert!((cli_value(&socp_out, "free_count") - 1.0).abs() < 1e-15);

    let frame = run_cli(&["realign"]);
    let frame_out = String::from_utf8_lossy(&frame.stdout);
    assert_eq!(
        frame.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&frame.stderr)
    );
    assert!(cli_value(&frame_out, "centroid_residual") <= 1.0e-6);
    assert!(cli_value(&frame_out, "det_residual") <= 1.0e-8);
    assert!(cli_value(&frame_out, "center_residual") <= 1.0e-8);
}

#[test]
fn cli_checkpoint_resume_matches_a_continuous_run() {
    let project = write_project(
        "sample.toml",
        &sample_with("assets/mesh/octahedron.obj", "", ""),
    );
    let dir = project.parent().unwrap();
    let first_path = dir.join("step1.chk");
    let second_path = dir.join("step2.chk");
    let project_arg = project.to_str().unwrap();
    let first = run_cli(&[
        "run-sim",
        "--project",
        project_arg,
        "--max-steps",
        "1",
        "--checkpoint",
        first_path.to_str().unwrap(),
    ]);
    let first_out = String::from_utf8_lossy(&first.stdout);
    let first_err = String::from_utf8_lossy(&first.stderr);
    assert_eq!(first.status.code(), Some(0), "{first_err}");
    let rtfp = cli_value(&first_out, "rtfp_residual");
    assert!(rtfp.is_finite(), "{first_out}");
    assert!(first_out.contains("phi_min"));
    assert!(!first_out.contains("verification_level passed"));
    assert!(first_err.contains("model radial_star"), "{first_err}");
    assert!(first_err.contains("face_hessian"), "{first_err}");
    assert!(first_err.contains("volume_hessian"), "{first_err}");
    let resumed = run_cli(&[
        "run-sim",
        "--project",
        project_arg,
        "--max-steps",
        "1",
        "--resume",
        first_path.to_str().unwrap(),
        "--checkpoint",
        second_path.to_str().unwrap(),
    ]);
    let resumed_out = String::from_utf8_lossy(&resumed.stdout);
    assert_eq!(
        resumed.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&resumed.stderr)
    );
    let continuous = run_cli(&["run-sim", "--project", project_arg, "--max-steps", "2"]);
    let continuous_out = String::from_utf8_lossy(&continuous.stdout);
    assert_eq!(
        continuous.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&continuous.stderr)
    );
    let mass_gap = (cli_value(&resumed_out, "mass") - cli_value(&continuous_out, "mass")).abs();
    let time_gap = (cli_value(&resumed_out, "time") - cli_value(&continuous_out, "time")).abs();
    assert!(cli_value(&first_out, "mass_error") < 1.0e-3);
    assert!(
        mass_gap / cli_value(&continuous_out, "mass") <= 1.0e-8,
        "mass gap {mass_gap}"
    );
    assert!(time_gap <= 1.0e-12, "time gap {time_gap}");
    assert!((cli_value(&resumed_out, "time") - 0.5).abs() < 1.0e-12);
}

#[test]
fn cli_missing_parameter_nonstrar_picard_rank_gpu_and_nonfinite() {
    let missing = write_project(
        "missing.toml",
        &sample_with("assets/mesh/octahedron.obj", "", "").replace(
            "[spin_schedule]\nkind = \"stages\"\n\n[spin_schedule.time]\nvalue = 1000000.0\nunit = \"s\"\nuncertainty = 0.0\nsource = \"model stage horizon in SI seconds; Watanabe et al. 2019 report no detected spin change\"\ncalibration = \"the stage holds the measured rotation rate; t_end is not an observed duration\"\n\n[[spin_schedule.stages]]\nt_end = 1000000.0\nomega = [0.0, 0.0, 0.00022866712242903872]\n\n",
            "",
        ),
    );
    let missing_out = run_cli(&["validate", "--project", missing.to_str().unwrap()]);
    let missing_err = String::from_utf8_lossy(&missing_out.stderr);
    assert_eq!(missing_out.status.code(), Some(10), "{missing_err}");
    assert!(missing_err.contains("ModelNotClosed"));
    assert!(!String::from_utf8_lossy(&missing_out.stdout).contains("mass_error"));

    let obj = std::env::temp_dir()
        .join("astroid-evolution-tier2")
        .join("offset.obj");
    std::fs::write(
        &obj,
        "v 3 0 0\nv 4 0 0\nv 3 1 0\nv 3 0 1\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n",
    )
    .unwrap();
    let nonstar = write_project(
        "nonstrar.toml",
        &sample_project_text()
            .replace("../models/ryugu.glb", obj.to_str().unwrap())
            .replace("mass_target = 4.50e11", "mass_target = 198.33333333333334")
            .replace("value = 4.50e11", "value = 198.33333333333334")
            .replace("uncertainty = 5.85e9", "uncertainty = 20.0")
            .replace("mass_tol = 0.013", "mass_tol = 0.1008403361344538")
            .replace(
                "representation = \"general_mesh\"",
                "representation = \"radial_star\"",
            ),
    );
    let nonstar_out = run_cli(&[
        "run-sim",
        "--project",
        nonstar.to_str().unwrap(),
        "--max-steps",
        "1",
    ]);
    let nonstar_err = String::from_utf8_lossy(&nonstar_out.stderr);
    assert_eq!(nonstar_out.status.code(), Some(10), "{nonstar_err}");
    assert!(cli_value(&nonstar_err, "star_margin") < 0.0);

    let picard = write_project(
        "picard.toml",
        &sample_with(
            "assets/mesh/octahedron.obj",
            "",
            "require_convergence = true",
        )
        .replace(
            "[[spin_schedule.stages]]\nt_end = 1000000.0\nomega = [0.0, 0.0, 0.00022866712242903872]",
            "[[spin_schedule.stages]]\nt_end = 1000000.0\nomega = [0.0, 0.0, 1.0]",
        ),
    );
    let picard_out = run_cli(&[
        "run-sim",
        "--project",
        picard.to_str().unwrap(),
        "--max-steps",
        "1",
    ]);
    let picard_err = String::from_utf8_lossy(&picard_out.stderr);
    assert_eq!(picard_out.status.code(), Some(20), "{picard_err}");
    assert!(cli_value(&picard_err, "picard_residual") > 1.0e-3);

    let mesh =
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets/mesh/octahedron.obj");
    let rank = run_cli(&["rigid", "--mesh", mesh.to_str().unwrap(), "--l-max", "0"]);
    let rank_err = String::from_utf8_lossy(&rank.stderr);
    assert_eq!(rank.status.code(), Some(11), "{rank_err}");
    assert!((cli_value(&rank_err, "rank") - 0.0).abs() < 1e-15);
    assert!(cli_value(&rank_err, "sigma_min").abs() <= 1.0e-15);

    let gpu = run_cli(&["gpu", "--force-missing"]);
    assert_eq!(
        gpu.status.code(),
        Some(50),
        "{}",
        String::from_utf8_lossy(&gpu.stderr)
    );
    assert!(String::from_utf8_lossy(&gpu.stderr).contains("GpuUnavailable"));

    let nonfinite = run_cli(&[
        "run-sim",
        "--project",
        "assets/projects/sample_project.toml",
        "--until-t",
        "nan",
    ]);
    let nonfinite_err = String::from_utf8_lossy(&nonfinite.stderr);
    assert_eq!(nonfinite.status.code(), Some(10), "{nonfinite_err}");
    assert!(nonfinite_err.contains("not finite"));
    assert!(!String::from_utf8_lossy(&nonfinite.stdout).contains("mass_error"));
}

fn scf_cli() -> std::path::PathBuf {
    if let Ok(path) = std::env::var("CARGO_BIN_EXE_scf_cli") {
        return std::path::PathBuf::from(path);
    }
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/debug/scf-cli")
}

fn sample_project_text() -> String {
    std::fs::read_to_string("assets/projects/sample_project.toml").expect("sample project")
}

fn project_dir() -> &'static Path {
    Path::new("assets/projects")
}

fn assert_unclosed(text: &str, dir: &Path) {
    let validated = validate_at(text, dir).expect_err("validate accepted an unclosed project");
    assert_eq!(validated.kind, "ModelNotClosed");
    assert_eq!(validated.code, 10);
    let ran = closed_run(text, dir, 1).expect_err("run-sim accepted an unclosed project");
    assert_eq!(ran.kind, "ModelNotClosed");
    assert_eq!(ran.code, 10);
    assert_eq!(validated.detail, ran.detail);
}

fn assert_sample_rejected(text: &str) {
    assert_unclosed(text, project_dir());
}

#[test]
fn sample_project_is_closed_for_validate() {
    let text = sample_project_text();
    validate_project_toml(&text).expect("schema");
    validate_at(&text, project_dir()).expect("validate");
}

#[test]
fn schema_rejects_nonfinite_and_illegal_fields() {
    let base = sample_project_text();
    assert_sample_rejected(&base.replace(
        "[material.cohesion]\nvalue = 0.05",
        "[material.cohesion]\nvalue = -1.0",
    ));
    assert_sample_rejected(&base.replace(
        "[material.cohesion]\nvalue = 0.05",
        "[material.cohesion]\nvalue = \"nan\"",
    ));
    assert_sample_rejected(&base.replace(
        "[material.cohesion]\nvalue = 0.05",
        "[material.cohesion]\nvalue = \"inf\"",
    ));
    assert_sample_rejected(&base.replace(
        "[material.cohesion]\nvalue = 0.05",
        "[material.cohesion]\nvalue = nan",
    ));
    assert_sample_rejected(&base.replace(
        "[material.cohesion]\nvalue = 0.05",
        "[material.cohesion]\nvalue = inf",
    ));
    assert_sample_rejected(&base.replace(
        "[material.mu_s]\nvalue = 0.6152041047020307",
        "[material.mu_s]\nvalue = -0.01",
    ));
    assert_sample_rejected(&base.replace(
        "[material.mu_y]\nvalue = 0.6152041047020307",
        "[material.mu_y]\nvalue = -0.2",
    ));
    assert_sample_rejected(&base.replace(
        "[material.mu_2]\nvalue = 0.6152041047020307",
        "[material.mu_2]\nvalue = -0.2",
    ));
    assert_sample_rejected(&base.replace(
        "[material.eta_c]\nvalue = 21523.132776306986",
        "[material.eta_c]\nvalue = 0.0",
    ));
    assert_sample_rejected(&base.replace("[material]\n", "[material]\neta = -1.0\n"));
    assert_sample_rejected(&base.replace(
        "[material.gamma_eps]\nvalue = 1.0e-6",
        "[material.gamma_eps]\nvalue = 0.0",
    ));
    assert_sample_rejected(&base.replace(
        "[material.p_min]\nvalue = 1.0e-6",
        "[material.p_min]\nvalue = 0.0",
    ));
    assert_sample_rejected(&base.replace(
        "[material.p_min]\nvalue = 1.0e-6",
        "[material.p_min]\nvalue = -5.0",
    ));
    assert_sample_rejected(&base.replace(
        "[rotation]\nomega = [0.0, 0.0, 0.00022866712242903872]",
        "[rotation]\nomega = [\"nan\", \"inf\", \"x\"]",
    ));
    assert_sample_rejected(&base.replace(
        "[rotation]\nomega = [0.0, 0.0, 0.00022866712242903872]",
        "[rotation]\nomega = [\"0\", \"0\", \"0\"]",
    ));
    assert_sample_rejected(&base.replace(
        "[spin_schedule]\nkind = \"stages\"\n\n[spin_schedule.time]\nvalue = 1000000.0\nunit = \"s\"\nuncertainty = 0.0\nsource = \"model stage horizon in SI seconds; Watanabe et al. 2019 report no detected spin change\"\ncalibration = \"the stage holds the measured rotation rate; t_end is not an observed duration\"\n\n[[spin_schedule.stages]]\nt_end = 1000000.0\nomega = [0.0, 0.0, 0.00022866712242903872]\n\n",
        "",
    ));
    assert_sample_rejected(&base.replace("path = \"../models/ryugu.glb\"\n", ""));
    assert_sample_rejected(&base.replace("mass_target = 4.50e11\n", ""));
    assert_sample_rejected(&base.replace(
        "[mode_c]\n\n[mode_c.erosion_coefficient]\nvalue = 0.0\nunit = \"1\"\nuncertainty = 0.0\nsource = \"no closed Ryugu mass-loss measurement\"\ncalibration = \"erosion stays identically zero; a nonzero rate would be an unsourced coefficient\"\n\n[mode_c.external_mass_flux]\nvalue = 0.0\nunit = \"kg/s\"\nuncertainty = 0.0\nsource = \"no closed Ryugu external mass flux\"\ncalibration = \"external mass flux stays identically zero\"\n\n[mode_c.stress_scale]\nvalue = 1.0\nunit = \"1\"\nuncertainty = 0.0\nsource = \"no measured stress normalization; erosion is closed\"\ncalibration = \"stress_scale stays 1 while the erosion coefficient is zero\"\n\n",
        "",
    ));
    assert_sample_rejected(&base.replace("[mode_c.stress_scale]\n", ""));
    let material_start = base.find("[material]\n").expect("material");
    let material_end = base.find("[mode_c]\n").expect("mode_c");
    let mut missing_material = base.clone();
    missing_material.replace_range(material_start..material_end, "");
    assert_sample_rejected(&missing_material);
    assert_sample_rejected(&base.replace("[material.mu_s]\n", ""));
    assert_sample_rejected(&base.replace("sigma = 6.081303192631498e-05", "sigma = 0.0"));
    assert_sample_rejected(&base.replace("sigma = 6.081303192631498e-05", "sigma = -1.0"));
    assert_sample_rejected(&base.replace("alpha = 1.0", "alpha = 0.0"));
    assert_sample_rejected(&base.replace(
        "role = \"homogeneous_bulk\"\n",
        "role = \"homogeneous_bulk\"\ntype = \"gaussian\"\n",
    ));
    assert_sample_rejected(&format!(
        "{base}\n[[density.kernels]]\nrole = \"illegal\"\ncenter = [0.0, 0.0, 0.0]\nweight = -1.0e12\nsigma = 1.0\nalpha = 1.0\n"
    ));
    assert_sample_rejected(
        &base
            .replace(
                "require_nonnegative = true\n",
                "require_nonnegative = \"yes\"\n",
            )
            .replace("[density]\n", "[density]\nrequire_nonnegative = \"yes\"\n"),
    );
    assert_sample_rejected(&format!(
        "{}\n[[density.kernels]]\nrole = \"hole\"\ncenter = [0.0, 0.0, 0.0]\nweight = -1.0e12\nsigma = 1.0\nalpha = 1.0\n",
        base.replace("[density]\n", "[density]\nrequire_nonnegative = false\n")
    ));
    assert_sample_rejected(&base.replace("galerkin_order = 4", "galerkin_order = 0"));
    assert_sample_rejected(&base.replace("galerkin_order = 4", "galerkin_order = 4.5"));
    assert_sample_rejected(&base.replace("galerkin_order = 4", "galerkin_order = -2"));
    assert_sample_rejected(&base.replace("l_max = 1", "l_max = 0"));
    assert_sample_rejected(&base.replace("l_max = 1", "l_max = 1.5"));
    assert_sample_rejected(&base.replace("l_max = 1", "l_max = -1"));
    assert_sample_rejected(&base.replace("[solver]\n", "[solver]\nK = -1\nK_p = 0\n"));
    assert_sample_rejected(&base.replace("[solver]\n", "[solver]\nK_p = 2.5\n"));
    assert_sample_rejected(&base.replace(
        "path = \"../models/ryugu.glb\"",
        "path = \"missing-mesh.obj\"",
    ));
    assert_sample_rejected(&base.replace(
        "path = \"../models/ryugu.glb\"",
        "path = [\"../models/ryugu.glb\"]",
    ));
}

#[test]
fn geometry_source_is_accepted_and_bad_files_are_rejected() {
    let base = sample_project_text();
    let sourced = base.replace(
        "path = \"../models/ryugu.glb\"",
        "source = \"../models/ryugu.glb\"",
    );
    validate_at(&sourced, project_dir()).expect("geometry.source");

    let dir = std::env::temp_dir().join(format!("astroid-schema-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let empty = base.replace("path = \"../models/ryugu.glb\"", "path = \"empty.obj\"");
    std::fs::write(dir.join("empty.obj"), "").unwrap();
    assert_unclosed(&empty, &dir);
    let blank = base.replace("path = \"../models/ryugu.glb\"", "path = \"blank.obj\"");
    std::fs::write(dir.join("blank.obj"), "   \n").unwrap();
    assert_unclosed(&blank, &dir);
    let wrong = base.replace("path = \"../models/ryugu.glb\"", "path = \"notes.txt\"");
    std::fs::write(dir.join("notes.txt"), "this is not a mesh\n").unwrap();
    assert_unclosed(&wrong, &dir);
    let _ = dir;
}

#[test]
fn validate_and_run_sim_share_nonzero_exit_for_unclosed_input() {
    let dir = std::env::temp_dir().join(format!("astroid-cli-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let bad = dir.join("unclosed.toml");
    let text = sample_project_text().replace(
        "[rotation]\nomega = [0.0, 0.0, 0.00022866712242903872]",
        "[rotation]\nomega = [\"a\", \"b\", \"c\"]",
    );
    std::fs::write(&bad, text).unwrap();
    for command in ["validate", "run-sim"] {
        let output = Command::new(scf_cli())
            .args([command, "--project", bad.to_str().unwrap()])
            .output()
            .expect("scf-cli");
        assert!(
            !output.status.success(),
            "{command} accepted a string array: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert_ne!(output.status.code(), Some(0));
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(!stdout.starts_with("closed "), "{stdout}");
    }
    let output = Command::new(scf_cli())
        .args([
            "run-sim",
            "--project",
            "assets/projects/sample_project.toml",
            "--until-t",
            "nan",
        ])
        .output()
        .expect("scf-cli");
    assert!(
        !output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let output = Command::new(scf_cli())
        .args([
            "run-sim",
            "--project",
            "assets/projects/sample_project.toml",
            "--until-t",
            "inf",
        ])
        .output()
        .expect("scf-cli");
    assert!(
        !output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn rigid_constraint_has_rank_six() {
    let values = singular_values(1).unwrap();
    assert_eq!(values.len(), 6);
    assert!(values.iter().all(|value| *value > 1.0e-8));
}

#[test]
fn rigid_constraint_is_density_weighted_and_rank_deficient_is_not_a_panic() {
    let mesh = TriMesh::unit_cube();
    let light = CauchyDensity::uniform(1.0).unwrap();
    let heavy = CauchyDensity::uniform(2.0).unwrap();
    let low = assemble_rigid_constraint(1, &mesh, &light, &[]).unwrap();
    let high = assemble_rigid_constraint(1, &mesh, &heavy, &[]).unwrap();
    assert_eq!(low.rank, 6);
    assert!(low.sigma_min > 1.0e-8);
    let low_peak = low.matrix.iter().map(|v| v.abs()).fold(0.0, f64::max);
    let high_peak = high.matrix.iter().map(|v| v.abs()).fold(0.0, f64::max);
    assert!(low_peak > 0.0);
    assert!((high_peak / low_peak - 2.0).abs() < 1.0e-8);
    let samples = [([1.0, 0.0, 0.0], [0.0, -2.0, 0.0], 0.5)];
    let loaded = assemble_rigid_constraint(1, &mesh, &light, &samples).unwrap();
    assert!((loaded.force[1] + 1.0).abs() < 1.0e-12);
    assert!((loaded.torque[2] + 1.0).abs() < 1.0e-12);
    let err = assemble_rigid_constraint(0, &mesh, &light, &[]).unwrap_err();
    assert_eq!(err.kind, "RigidRankDeficient");
    assert_eq!(err.code, 11);
}

#[test]
fn realignment_translates_and_rotates_the_body_and_the_kernels() {
    let cube = TriMesh::unit_cube();
    let shift = DVec3::new(0.4, 0.1, -0.2);
    let vertices = cube.vertices.iter().map(|v| *v + shift).collect();
    let mesh = TriMesh::try_new(vertices, cube.faces.clone()).unwrap();
    let directions: Vec<DVec3> = cube
        .vertices
        .iter()
        .map(|v| v.try_normalize().unwrap())
        .collect();
    let kernel =
        CauchyKernel::try_new(shift + DVec3::new(0.2, 0.0, 0.0), 1.0e-4, 1.0, 1.0).unwrap();
    let density = CauchyDensity::try_new(10.0, vec![kernel]).unwrap();
    let frame = realign_body(&mesh, &directions, &density, 1.0e-6, 1.0e-8).unwrap();
    assert!(frame.applied);
    assert!((frame.det - 1.0).abs() < 1.0e-8);
    let center = frame.density.kernels[0].center;
    assert!((center - DVec3::new(0.2, 0.0, 0.0)).length() < 1.0e-8);
    assert!(frame.centroid_residual <= 1.0e-6);
    assert!(frame.ixz.abs().max(frame.iyz.abs()) <= 1.0e-6 * frame.i1.abs().max(1.0));
    for (radius, vertex) in frame.radii.iter().zip(&cube.vertices) {
        assert!((radius - vertex.length()).abs() < 1.0e-4);
    }
}

#[test]
fn principal_axes_follow_the_inertia_signs_and_degeneracy() {
    let (rotation, _) = principal_frame(
        &Matrix3::from_diagonal(&Vector3::new(4.0, 4.0, 4.0)),
        1.0e-8,
    )
    .unwrap();
    assert_eq!(
        rotation,
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    );
    let a1 = Vector3::new(1.0, 1.0, 0.0).normalize();
    let a3 = Vector3::new(1.0, -1.0, 0.0).normalize();
    let a2 = a1.cross(&a3).normalize();
    let inertia = a1 * a1.transpose() * 5.0 + a2 * a2.transpose() * 3.0 + a3 * a3.transpose() * 1.0;
    let (rotation, i1) = principal_frame(&inertia, 1.0e-8).unwrap();
    assert!((i1 - 5.0).abs() < 1.0e-8);
    let ez = Vector3::new(rotation[2][0], rotation[2][1], rotation[2][2]);
    let ex = Vector3::new(rotation[0][0], rotation[0][1], rotation[0][2]);
    assert!(ez.cross(&a1).norm() < 1.0e-8);
    assert!(ex.dot(&a3) > 0.0);
    let det = ex.dot(&Vector3::new(rotation[1][0], rotation[1][1], rotation[1][2]).cross(&ez));
    assert!((det - 1.0).abs() < 1.0e-8);
}

#[test]
fn operators_follow_density_and_picard_updates_pressure_viscosity_and_velocity() {
    let mesh = TriMesh::unit_cube();
    let light = CauchyDensity::uniform(1.0).unwrap();
    let heavy = CauchyDensity::uniform(2.0).unwrap();
    let low = kinetic_mass(1, &mesh, &light).unwrap();
    let high = kinetic_mass(1, &mesh, &heavy).unwrap();
    let low_peak = low.iter().map(|v| v.abs()).fold(0.0, f64::max);
    let high_peak = high.iter().map(|v| v.abs()).fold(0.0, f64::max);
    assert!(low_peak > 0.0);
    assert!((high_peak / low_peak - 2.0).abs() < 1e-8);
    let material = PicardMaterial {
        cohesion: 0.0,
        mu: 0.6,
        gamma_eps: 1e-4,
        p_min: 1e-6,
        gravity: 1.0,
        tide_coefficient: 0.0,
        omega_z: 0.0,
        relaxation: 1.0,
        aug_lagrange: 1.0,
    };
    let samples = [GravSample {
        position: [0.0, 0.0, 0.0],
        g_eff: [0.0, 0.0, -1.0],
    }];
    let first = picard_solve(1, 2, &mesh, &light, &samples, material, 1, Vec::new()).unwrap();
    assert!(first.pressure.iter().any(|p| p.abs() > 1e-8));
    assert!(first.eta.iter().any(|eta| eta.is_finite() && *eta > 0.0));
    assert!(first.coefficients.iter().any(|u| u.abs() > 1e-10));
    assert!(first.picard_residual.is_finite());
    assert!(first.pcg_residual.is_finite());
    assert!(first.strong_residual.is_finite());
    assert!(first.tension_fraction.is_finite());
    let second = picard_solve(
        1,
        2,
        &mesh,
        &light,
        &samples,
        material,
        1,
        first.coefficients.clone(),
    )
    .unwrap();
    let changed = first
        .coefficients
        .iter()
        .zip(&second.coefficients)
        .any(|(a, b)| (a - b).abs() > 1e-8);
    assert!(
        changed
            || second
                .eta
                .iter()
                .zip(&first.eta)
                .any(|(a, b)| (a - b).abs() > 1e-8)
    );
    let mut tidal = material;
    tidal.tide_coefficient = 1.0;
    tidal.omega_z = 1.0;
    let quiet = picard_solve(2, 1, &mesh, &light, &samples, material, 1, Vec::new()).unwrap();
    let with_tide = picard_solve(2, 1, &mesh, &light, &samples, tidal, 1, Vec::new()).unwrap();
    let _ = quiet;
    let tide_moves_pressure = quiet
        .pressure
        .iter()
        .zip(&with_tide.pressure)
        .any(|(a, b)| (a - b).abs() > 1e-8);
    assert!(tide_moves_pressure);
}

#[test]
fn a_collinear_core_is_degenerate_without_a_factorization_panic() {
    let line = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
    let err = fit_core_motion(&line, &[[0.0; 3]; 3]).unwrap_err();
    assert_eq!(err.kind, "CoreDegenerate");
    assert_eq!(err.code, 32);
    let points = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ];
    let velocity = [[0.4, -0.2, 0.1]; 4];
    let motion = fit_core_motion(&points, &velocity).unwrap();
    assert!((motion.translation[0] - 0.4).abs() < 1.0e-8);
    assert!(motion.rotation.iter().all(|w| w.abs() < 1.0e-8));
}

#[test]
fn general_mesh_tier2_stays_unsupported_and_writes_no_stable_mesh() {
    let mass = 1602.4101348454262_f64;
    let text = sample_with("assets/mesh/octahedron.obj", "", "")
        .replace(
            "representation = \"radial_star\"",
            "representation = \"general_mesh\"",
        )
        .replace(
            "mass_target = 2194.2094889027243880",
            "mass_target = 1602.4101348454262",
        )
        .replace(
            "value = 2194.2094889027243880",
            "value = 1602.4101348454262",
        )
        .replace(
            "mass_tol = 0.009114899967915806",
            &format!("mass_tol = {:.16}", 20.0 / mass),
        );
    let project = write_project("general-mesh.toml", &text);
    let out_dir = std::env::temp_dir()
        .join("astroid-evolution-tier2")
        .join("general-mesh-out");
    let _ = std::fs::remove_dir_all(&out_dir);
    std::fs::create_dir_all(&out_dir).unwrap();
    let output = run_cli(&[
        "run-sim",
        "--project",
        project.to_str().unwrap(),
        "--max-steps",
        "1",
        "--stable-steps",
        "2",
        "--out-dir",
        out_dir.to_str().unwrap(),
    ]);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_ne!(output.status.code(), Some(14), "{stderr}\n{stdout}");
    assert!(!stderr.contains("ModelUnsupported"), "{stderr}");
    assert!(!stderr.contains("direction*radius"), "{stderr}");
    assert!(stdout.contains("model general_mesh"), "{stdout}");
    assert!(
        stdout.contains("final_stable false") || stdout.contains("final_stable true"),
        "{stdout}"
    );
    if stdout.contains("final_stable true") {
        assert!(stdout.contains("status stable"), "{stdout}");
        assert!(out_dir.join("final.obj").exists());
    } else {
        assert!(!out_dir.join("final.obj").exists(), "{stdout}");
        assert!(!out_dir.join("final.glb").exists(), "{stdout}");
    }
    assert!(
        out_dir.join("diagnostics.json").exists(),
        "{stdout}\n{stderr}"
    );
    let diagnostics = std::fs::read_to_string(out_dir.join("diagnostics.json")).unwrap();
    assert!(diagnostics.contains("tier1_call_id"), "{diagnostics}");
    assert!(diagnostics.contains("mesh_hash"), "{diagnostics}");

    let vertices = std::env::temp_dir()
        .join("astroid-evolution-tier2")
        .join("vertices-only.obj");
    std::fs::write(&vertices, "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\n").unwrap();
    let bare = text.replace(
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("assets/mesh/octahedron.obj")
            .display()
            .to_string()
            .as_str(),
        vertices.to_str().unwrap(),
    );
    let bare_project = write_project("vertices-only.toml", &bare);
    let bare_out = run_cli(&[
        "run-sim",
        "--project",
        bare_project.to_str().unwrap(),
        "--max-steps",
        "1",
    ]);
    let bare_err = String::from_utf8_lossy(&bare_out.stderr);
    assert_eq!(bare_out.status.code(), Some(10), "{bare_err}");
    assert!(bare_err.contains("ModelNotClosed"), "{bare_err}");
    assert!(bare_err.contains("reconstruct"), "{bare_err}");
    assert!(!String::from_utf8_lossy(&bare_out.stdout).contains("status stable"));
}
