//! RTFP command line.
//!
//! `eval-field` runs the ray-traced finite-part solver on the hardware ray
//! tracer. The Feynman, Carlson and Werner benchmarks run only when
//! `--compare` names one of them.

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use astroid_evolution::common::density::CauchyDensity;
use astroid_evolution::common::geometry::TriMesh;
use astroid_evolution::common::math::{DVec3, Sym6};
use astroid_evolution::rtfp_solver::api::{FieldBatch, FieldMask, PointRole, PointSet};
use astroid_evolution::rtfp_solver::io::aevb::{Comparison, FieldRecord, write_field};
use astroid_evolution::rtfp_solver::io::{
    DensityInput, ImportRequest, MeshDocument, Reconstruct, assert_nonnegative, density_hash,
    density_to_kg_m3, face_centroids, load_density, load_mesh_file, positivity_points,
    vertex_points, write_obj,
};
use astroid_evolution::rtfp_solver::reference::ellipsoid::Ellipsoid;
use astroid_evolution::rtfp_solver::reference::feynman::Feynman;
use astroid_evolution::rtfp_solver::reference::werner;
use astroid_evolution::rtfp_solver::{RtfpError, RtfpOptions, RtfpSolver};
use clap::{Parser, Subcommand, ValueEnum};

#[derive(Parser)]
#[command(name = "rtfp-cli")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Read GLB, glTF, OBJ, or a vertex list and write a checked triangle mesh.
    ImportMesh {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long)]
        length_unit: String,
        #[arg(long, value_enum)]
        reconstruct: Option<ReconstructArg>,
    },
    /// Evaluate the field on the hardware ray tracer and write an `.aevb` record.
    EvalField {
        #[command(flatten)]
        mesh: MeshArgs,
        #[command(flatten)]
        density: DensityArgs,
        #[command(flatten)]
        samples: SampleArgs,
        /// Components to compute: any of `u`, `grad`, `hess`, comma separated.
        #[arg(long)]
        want: String,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value_t = 6.67430e-11)]
        g: f64,
        /// A patch is split while its edge exceeds this fraction of its distance from the point.
        #[arg(long)]
        angular_ratio: Option<f64>,
        /// Also run a high-precision benchmark and report the relative error.
        #[arg(long, value_enum)]
        compare: Option<CompareArg>,
        /// Semi-axes `a,b,c` of the ellipsoid the mesh samples, for `--compare carlson`
        /// (Cauchy kernels need a sphere with kernels at its centre).
        #[arg(long, value_delimiter = ',', num_args = 3)]
        ellipsoid: Option<Vec<f64>>,
    },
    /// Volume, mass, centroid and inertia of the density on the mesh.
    Integrals {
        #[command(flatten)]
        mesh: MeshArgs,
        #[command(flatten)]
        density: DensityArgs,
        #[arg(long)]
        angular_ratio: Option<f64>,
    },
}

#[derive(clap::Args)]
struct MeshArgs {
    #[arg(long)]
    mesh: PathBuf,
    #[arg(long)]
    length_unit: Option<String>,
}

#[derive(clap::Args)]
struct DensityArgs {
    #[arg(long)]
    density: Option<PathBuf>,
    #[arg(long)]
    uniform: Option<f64>,
    #[arg(long)]
    density_unit: Option<String>,
}

#[derive(clap::Args)]
struct SampleArgs {
    #[arg(long)]
    surface_faces: bool,
    #[arg(long)]
    surface_vertices: bool,
    #[arg(long)]
    points: Option<PathBuf>,
    #[arg(long)]
    max_points: Option<usize>,
}

#[derive(Clone, Copy, ValueEnum)]
enum ReconstructArg {
    #[value(name = "convex_hull", alias = "convex-hull")]
    ConvexHull,
    #[value(name = "radial_raycast", alias = "radial-raycast")]
    RadialRaycast,
}

#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum CompareArg {
    /// Exact uniform polyhedron.
    Werner,
    /// Feynman-parameter evaluation of the Cauchy kernels.
    Feynman,
    /// Exact homogeneous ellipsoid through Carlson integrals.
    Carlson,
}

fn main() -> ExitCode {
    let result = match Cli::parse().command {
        Command::ImportMesh { input, out, length_unit, reconstruct } => {
            import_mesh(&input, &out, &length_unit, reconstruct)
        }
        Command::EvalField {
            mesh,
            density,
            samples,
            want,
            out,
            g,
            angular_ratio,
            compare,
            ellipsoid,
        } => eval_field(&mesh, &density, &samples, &want, &out, g, angular_ratio, compare, ellipsoid),
        Command::Integrals { mesh, density, angular_ratio } => integrals(&mesh, &density, angular_ratio),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("{err}");
            ExitCode::from(err.code() as u8)
        }
    }
}

fn import_mesh(
    input: &Path,
    out: &Path,
    length_unit: &str,
    reconstruct: Option<ReconstructArg>,
) -> Result<(), RtfpError> {
    let document = load_mesh_file(ImportRequest {
        path: input,
        length_unit: Some(length_unit),
        reconstruct: reconstruct.map(|value| match value {
            ReconstructArg::ConvexHull => Reconstruct::ConvexHull,
            ReconstructArg::RadialRaycast => Reconstruct::RadialRaycast,
        }),
    })?;
    write_obj(out, &document)?;
    println!("vertices {}", document.mesh.vertices.len());
    println!("faces {}", document.mesh.faces.len());
    println!("welded_positions {}", document.welded_positions);
    println!("dropped_degenerate {}", document.dropped_degenerate);
    println!("source_length_unit {}", document.source_length_unit);
    println!("length_unit {}", document.length_unit);
    println!("mesh_hash {}", document.mesh_hash);
    println!("out {}", out.display());
    Ok(())
}

fn options(g: f64, angular_ratio: Option<f64>) -> RtfpOptions {
    let mut options = RtfpOptions { g, ..RtfpOptions::default() };
    if let Some(ratio) = angular_ratio {
        options.angular_ratio = ratio;
    }
    options
}

#[allow(clippy::too_many_arguments)]
fn eval_field(
    mesh_args: &MeshArgs,
    density_args: &DensityArgs,
    samples: &SampleArgs,
    want: &str,
    out: &Path,
    g: f64,
    angular_ratio: Option<f64>,
    compare: Option<CompareArg>,
    ellipsoid: Option<Vec<f64>>,
) -> Result<(), RtfpError> {
    let want = parse_want(want)?;
    let document = open_mesh(mesh_args)?;
    let density = open_density(density_args)?;
    let (role, role_name, points) = sample_points(&document.mesh, samples)?;
    let probes = positivity_points(&document.mesh, &points)?;
    assert_nonnegative(&density.density, &probes)?;

    let solver = RtfpSolver::bind(options(g, angular_ratio), document.mesh.clone(), density.density.clone())?;
    let start = Instant::now();
    let batch = solver.eval(&PointSet::from_points(&points), role, want)?;
    let seconds = start.elapsed().as_secs_f64();
    if !field_is_finite(&batch) {
        return Err(RtfpError::geometry("the evaluated field contains a non-finite value"));
    }

    let comparison = match compare {
        Some(kind) => Some(run_comparison(kind, &document.mesh, &density.density, &points, role, want, &batch, g, ellipsoid)?),
        None => None,
    };

    write_field(
        out,
        &FieldRecord {
            role: role_name.into(),
            g,
            density_model: density.model.clone(),
            density_unit: density.density_unit.clone(),
            density_hash: density.hash.clone(),
            mesh_hash: document.mesh_hash.clone(),
            mesh_length_unit: document.length_unit.clone(),
            faces: document.mesh.faces.len(),
            angular_ratio: solver.options().angular_ratio,
            want_potential: want.potential,
            want_gradient: want.gradient,
            want_hessian: want.hessian,
            points: points.clone(),
            potential: batch.potential.clone(),
            gradient: batch.gradient.clone(),
            hessian: batch.hessian.clone(),
            theta: batch.theta.clone(),
            solid_angle: batch.solid_angle.clone(),
            comparison,
        },
    )?;
    println!("role {role_name}");
    println!("points {}", points.len());
    println!("faces {}", document.mesh.faces.len());
    println!("seconds {seconds:.3}");
    println!("out {}", out.display());
    Ok(())
}

/// Relative errors against the requested benchmark, per point and as maxima.
#[allow(clippy::too_many_arguments)]
fn run_comparison(
    kind: CompareArg,
    mesh: &TriMesh,
    density: &CauchyDensity,
    points: &[DVec3],
    role: PointRole,
    want: FieldMask,
    batch: &FieldBatch,
    g: f64,
    ellipsoid: Option<Vec<f64>>,
) -> Result<Comparison, RtfpError> {
    let uniform = density.kernels.is_empty();
    let body = match (kind, ellipsoid) {
        (CompareArg::Carlson, Some(axes)) if axes.len() == 3 => Some(Ellipsoid::new(DVec3::ZERO, [axes[0], axes[1], axes[2]])),
        (CompareArg::Carlson, _) => {
            return Err(RtfpError::model_not_closed("--compare carlson needs --ellipsoid a,b,c"));
        }
        _ => None,
    };
    if kind == CompareArg::Werner && !uniform {
        return Err(RtfpError::model_not_closed(
            "--compare werner is exact for a uniform density; use --compare feynman or carlson for kernels",
        ));
    }
    let feynman = (kind == CompareArg::Feynman).then(|| Feynman::new(mesh, density, g));
    let mut per_point = Vec::with_capacity(points.len());
    let (mut max_u, mut max_g, mut max_h): (Option<f64>, Option<f64>, Option<f64>) = (None, None, None);
    let bump = |slot: &mut Option<f64>, value: Option<f64>| {
        if let Some(v) = value {
            *slot = Some(slot.map_or(v, |m| m.max(v)));
        }
    };
    for (i, x) in points.iter().enumerate() {
        let (u, grad, hess): (f64, DVec3, Option<Sym6>) = match kind {
            CompareArg::Werner => {
                let w = werner::uniform_polyhedron(mesh, *x, g, density.bulk);
                (w.potential, w.gradient, Some(w.hessian))
            }
            CompareArg::Carlson => {
                let f = body
                    .expect("ellipsoid was checked")
                    .cauchy_field(*x, g, density)
                    .map_err(RtfpError::model_not_closed)?;
                (f.potential, f.gradient, (role == PointRole::Field).then_some(f.hessian))
            }
            CompareArg::Feynman => {
                let reference = feynman.as_ref().expect("feynman reference");
                let (u, grad) = reference.potential_gradient(*x);
                let hess = if !want.hessian {
                    None
                } else {
                    match role {
                        PointRole::Field => Some(reference.hessian(*x)?),
                        PointRole::SurfaceFace => {
                            let face = werner::face_containing(mesh, *x)
                                .ok_or_else(|| RtfpError::geometry("face point is not inside a face"))?;
                            Some(reference.face_hessian(*x, face)?)
                        }
                        PointRole::SurfaceVertex => None,
                    }
                };
                (u, grad, hess)
            }
        };
        let eu = want.potential.then(|| (batch.potential[i] - u).abs() / u.abs().max(1e-300));
        let eg = want.gradient.then(|| (batch.gradient[i] - grad).length() / grad.length().max(1e-300));
        let eh = hess.filter(|_| want.hessian).map(|h| {
            let scale = h.iter().fold(0.0_f64, |m, v| m.max(v.abs())).max(1e-300);
            (0..6).fold(0.0_f64, |m, c| m.max((batch.hessian[i][c] - h[c]).abs())) / scale
        });
        bump(&mut max_u, eu);
        bump(&mut max_g, eg);
        bump(&mut max_h, eh);
        per_point.push([eu, eg, eh]);
    }
    let name = match kind {
        CompareArg::Werner => "werner",
        CompareArg::Feynman => "feynman",
        CompareArg::Carlson => "carlson",
    };
    let show = |v: Option<f64>| v.map_or_else(|| "none".to_string(), |v| format!("{v:.3e}"));
    println!("reference {name}");
    println!("max_error_u {}", show(max_u));
    println!("max_error_grad {}", show(max_g));
    println!("max_error_hess {}", show(max_h));
    Ok(Comparison {
        reference: name.into(),
        max_error_u: max_u,
        max_error_grad: max_g,
        max_error_hess: max_h,
        per_point,
    })
}

fn integrals(mesh_args: &MeshArgs, density_args: &DensityArgs, angular_ratio: Option<f64>) -> Result<(), RtfpError> {
    let document = open_mesh(mesh_args)?;
    let density = open_density(density_args)?;
    let solver = RtfpSolver::bind(options(RtfpOptions::default().g, angular_ratio), document.mesh.clone(), density.density.clone())?;
    let got = solver.density_integrals()?;
    println!("volume {:.16e}", got.volume);
    println!("mass {:.16e}", got.mass);
    println!("centroid {:.16e} {:.16e} {:.16e}", got.centroid.x, got.centroid.y, got.centroid.z);
    println!(
        "inertia {:.16e} {:.16e} {:.16e} {:.16e} {:.16e} {:.16e}",
        got.inertia[0], got.inertia[1], got.inertia[2], got.inertia[3], got.inertia[4], got.inertia[5]
    );
    println!("density_model {}", density.model);
    println!("density_hash {}", density.hash);
    println!("mesh_hash {}", document.mesh_hash);
    Ok(())
}

fn sample_points(
    mesh: &TriMesh,
    samples: &SampleArgs,
) -> Result<(PointRole, &'static str, Vec<DVec3>), RtfpError> {
    let selected = [samples.surface_faces, samples.surface_vertices, samples.points.is_some()]
        .iter()
        .filter(|flag| **flag)
        .count();
    if selected != 1 {
        return Err(RtfpError::model_not_closed(
            "eval-field requires exactly one of --surface-faces, --surface-vertices, or --points",
        ));
    }
    if samples.surface_faces {
        Ok((PointRole::SurfaceFace, "surface_face", face_centroids(mesh, samples.max_points)))
    } else if samples.surface_vertices {
        Ok((PointRole::SurfaceVertex, "surface_vertex", vertex_points(mesh, samples.max_points)))
    } else {
        let path = samples.points.as_ref().expect("selected");
        Ok((PointRole::Field, "field", read_points(path, samples.max_points)?))
    }
}

fn read_points(path: &Path, max_points: Option<usize>) -> Result<Vec<DVec3>, RtfpError> {
    let text = std::fs::read_to_string(path).map_err(|err| {
        RtfpError::model_not_closed(format!("cannot read points {}: {err}", path.display()))
    })?;
    let mut points = Vec::new();
    for (line_number, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let coords: Vec<f64> = line
            .split_whitespace()
            .map(|token| token.parse::<f64>().unwrap_or(f64::NAN))
            .collect();
        if coords.len() < 3 || coords.iter().any(|value| !value.is_finite()) {
            return Err(RtfpError::model_not_closed(format!(
                "point line {} is not three finite coordinates",
                line_number + 1
            )));
        }
        points.push(DVec3::new(coords[0], coords[1], coords[2]));
        if max_points.is_some_and(|limit| points.len() >= limit) {
            break;
        }
    }
    if points.is_empty() {
        return Err(RtfpError::model_not_closed("point file has no coordinates"));
    }
    Ok(points)
}

fn open_mesh(args: &MeshArgs) -> Result<MeshDocument, RtfpError> {
    load_mesh_file(ImportRequest {
        path: &args.mesh,
        length_unit: args.length_unit.as_deref(),
        reconstruct: None,
    })
}

fn open_density(args: &DensityArgs) -> Result<DensityInput, RtfpError> {
    match (&args.density, args.uniform, &args.density_unit) {
        (Some(_), Some(_), _) => Err(RtfpError::model_not_closed(
            "pass either --density or --uniform, not both",
        )),
        (None, None, _) => Err(RtfpError::model_not_closed(
            "density is missing; pass --density or --uniform and --density-unit. The value is not defaulted to 1",
        )),
        (Some(path), None, _) => load_density(path),
        (None, Some(_), None) => Err(RtfpError::model_not_closed("uniform density requires --density-unit")),
        (None, Some(value), Some(unit)) => {
            let density = CauchyDensity::uniform(value * density_to_kg_m3(unit)?)?;
            let mut input = DensityInput {
                model: "uniform".into(),
                density_unit: unit.clone(),
                length_unit: "m".into(),
                hash: String::new(),
                density,
            };
            input.hash = density_hash(&input);
            Ok(input)
        }
    }
}

fn parse_want(text: &str) -> Result<FieldMask, RtfpError> {
    let mut mask = FieldMask { potential: false, gradient: false, hessian: false };
    for part in text.split(',') {
        match part.trim() {
            "u" => mask.potential = true,
            "grad" => mask.gradient = true,
            "hess" => mask.hessian = true,
            other => {
                return Err(RtfpError::model_not_closed(format!(
                    "unknown field component '{other}'; use u, grad or hess"
                )));
            }
        }
    }
    Ok(mask)
}

fn field_is_finite(batch: &FieldBatch) -> bool {
    batch.potential.iter().all(|v| v.is_finite())
        && batch.gradient.iter().all(|v| v.is_finite())
        && batch.hessian.iter().all(|h| h.iter().all(|v| v.is_finite()))
        && batch.theta.iter().all(|v| v.is_finite())
}
