use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;
use std::process::ExitCode;

use astroid_evolution::common::error::SimError;
use astroid_evolution::shsmooth::ShGpu;
use astroid_evolution::shsmooth::filter::Filter;
use astroid_evolution::shsmooth::fuse::{self, FuseOptions};
use astroid_evolution::shsmooth::harmonics;
use astroid_evolution::shsmooth::mesh::{Center, Mesh};
use astroid_evolution::shsmooth::smooth::{self, SmoothOptions};

#[derive(Parser)]
#[command(
    name = "sh-cli",
    about = "Spherical-harmonic mesh smoothing (host fit and filter, Metal evaluation)"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Fit, filter and evaluate. The output OBJ keeps the input face list.
    Smooth {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value_t = 8)]
        l_max: usize,
        #[arg(long, value_enum, default_value_t = FilterName::Heat)]
        filter: FilterName,
        /// Heat-kernel time. Gain of degree l is exp(-kappa*l*(l+1)).
        #[arg(long, default_value_t = 0.02)]
        kappa: f64,
        /// Highest degree kept by the truncate filter. Defaults to l_max.
        #[arg(long)]
        l_cut: Option<usize>,
        /// Cosine taper: gain 1 at l0, gain 0 at l1.
        #[arg(long)]
        l0: Option<f64>,
        #[arg(long)]
        l1: Option<f64>,
        /// Gaussian width in degree.
        #[arg(long)]
        sigma: Option<f64>,
        /// Ancient Ryugu rollback: Y20×0.3, Y40×0, equator −45 m, equatorial detail ×0.2.
        #[arg(long)]
        ancient: bool,
        /// Scale on Y20 after the spectral filter. 0.3 attenuates the oblateness by 70%.
        #[arg(long)]
        y20_scale: Option<f64>,
        /// Scale on Y40 after the spectral filter. 0 removes it, 1 keeps it.
        #[arg(long)]
        y40_scale: Option<f64>,
        /// Metres subtracted from the macro equator, fading to 0 at --drop-lat.
        #[arg(long)]
        equator_drop: Option<f64>,
        /// Latitude in degrees where the equatorial drop reaches zero.
        #[arg(long, default_value_t = 25.0)]
        drop_lat: f64,
        /// Equatorial belt, in degrees, inside which detail is scaled by --detail-gain.
        #[arg(long, default_value_t = 15.0)]
        detail_lat: f64,
        /// Detail scale inside the equatorial belt. 0.2 cuts the sharp ridge; poles stay at 1.
        #[arg(long)]
        detail_gain: Option<f64>,
        /// `auto`, `origin`, or `x,y,z`.
        #[arg(long, default_value = "auto")]
        center: String,
        #[arg(long, default_value_t = 1e-12)]
        ridge: f64,
        /// JSON report. The mesh itself is always written to --out.
        #[arg(long)]
        report: Option<PathBuf>,
    },
    /// Add the coarse radial displacement onto the high-resolution mesh.
    ///
    /// At each coarse vertex, ΔR = R_coarse − R_high along that direction.
    /// Every high-resolution vertex keeps its radius detail and adds the
    /// interpolated ΔR. Face counts are whatever the two inputs have.
    Fuse {
        /// Detailed mesh. Its faces are copied onto the output.
        #[arg(long)]
        high: PathBuf,
        /// Coarse mesh that carries the macroscopic radius.
        #[arg(long)]
        coarse: PathBuf,
        #[arg(long)]
        out: PathBuf,
        /// `auto`, `origin`, or `x,y,z`. Resolved on the high-resolution mesh.
        #[arg(long, default_value = "auto")]
        center: String,
        /// JSON report. The mesh itself is always written to --out.
        #[arg(long)]
        report: Option<PathBuf>,
    },
    /// Check that the mesh is a triangle mesh and star-shaped about the center.
    Validate {
        #[arg(long)]
        input: PathBuf,
        #[arg(long, default_value = "auto")]
        center: String,
    },
    /// Print the fitted power of each degree. No device evaluation.
    Spectrum {
        #[arg(long)]
        input: PathBuf,
        #[arg(long, default_value_t = 8)]
        l_max: usize,
        #[arg(long, default_value = "auto")]
        center: String,
        #[arg(long, default_value_t = 1e-12)]
        ridge: f64,
    },
    /// Report the Metal device the evaluator would use.
    Gpu,
}

#[derive(Clone, Copy, ValueEnum)]
enum FilterName {
    Heat,
    Truncate,
    Cosine,
    Gaussian,
}

impl std::fmt::Display for FilterName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            FilterName::Heat => "heat",
            FilterName::Truncate => "truncate",
            FilterName::Cosine => "cosine",
            FilterName::Gaussian => "gaussian",
        })
    }
}

fn parse_center(text: &str) -> Result<Center, SimError> {
    match text {
        "auto" => Ok(Center::Auto),
        "origin" => Ok(Center::Origin),
        other => {
            let parts: Vec<&str> = other.split(',').collect();
            if parts.len() != 3 {
                return Err(SimError::invalid_input(
                    "center must be auto, origin, or x,y,z",
                ));
            }
            let mut c = [0.0; 3];
            for i in 0..3 {
                c[i] = parts[i].trim().parse::<f64>().map_err(|_| {
                    SimError::invalid_input(format!("bad center component '{}'", parts[i]))
                })?;
            }
            Ok(Center::At(c))
        }
    }
}

fn build_filter(
    name: FilterName,
    l_max: usize,
    kappa: f64,
    l_cut: Option<usize>,
    l0: Option<f64>,
    l1: Option<f64>,
    sigma: Option<f64>,
) -> Result<Filter, SimError> {
    let filter = match name {
        FilterName::Heat => Filter::Heat { kappa },
        FilterName::Truncate => Filter::Truncate {
            l_cut: l_cut.unwrap_or(l_max),
        },
        FilterName::Cosine => Filter::Cosine {
            l0: l0.unwrap_or((l_max / 2) as f64),
            l1: l1.unwrap_or(l_max.max(1) as f64),
        },
        FilterName::Gaussian => Filter::Gaussian {
            sigma: sigma.unwrap_or(((l_max / 2).max(1)) as f64),
        },
    };
    filter.validate(l_max)?;
    Ok(filter)
}

fn run(cli: Cli) -> Result<(), SimError> {
    match cli.command {
        Command::Smooth {
            input,
            out,
            l_max,
            filter,
            kappa,
            l_cut,
            l0,
            l1,
            sigma,
            ancient,
            y20_scale,
            y40_scale,
            equator_drop,
            drop_lat,
            detail_lat,
            detail_gain,
            center,
            ridge,
            report,
        } => {
            let mesh = Mesh::read_obj(&input)?;
            let preset = SmoothOptions::ancient_ryugu();
            let plain = SmoothOptions::default();
            let pick = |explicit: Option<f64>, ancient_value: f64, plain_value: f64| {
                explicit.unwrap_or(if ancient { ancient_value } else { plain_value })
            };
            let options = SmoothOptions {
                l_max,
                filter: build_filter(filter, l_max, kappa, l_cut, l0, l1, sigma)?,
                center: parse_center(&center)?,
                ridge,
                y20_gain: pick(y20_scale, preset.y20_gain, plain.y20_gain),
                y40_gain: pick(y40_scale, preset.y40_gain, plain.y40_gain),
                equator_drop: pick(equator_drop, preset.equator_drop, plain.equator_drop),
                drop_latitude_deg: drop_lat,
                detail_latitude_deg: detail_lat,
                detail_gain: pick(detail_gain, preset.detail_gain, plain.detail_gain),
            };
            let gpu = ShGpu::new()?;
            let smoothed = smooth::smooth(&mesh, &options, &gpu)?;
            smoothed.mesh.write_obj(&out, &smoothed.report.filter)?;
            println!(
                "{} faces -> {} faces, {} vertices, {}, Y20 {:.3e}->{:.3e}, Y40 {:.3e}->{:.3e}, equator drop {:.1}, detail ×{:.2}, pole {:.2}->{:.2}, equator {:.2}->{:.2}, eval {:.4} s on {}",
                smoothed.report.input_faces,
                smoothed.report.output_faces,
                smoothed.report.output_vertices,
                smoothed.report.filter,
                smoothed.report.y20_macro,
                smoothed.report.y20_ancient,
                smoothed.report.y40_macro,
                smoothed.report.y40_ancient,
                smoothed.report.equator_drop,
                smoothed.report.detail_gain,
                smoothed.report.pole_radius_in,
                smoothed.report.pole_radius_out,
                smoothed.report.equator_radius_in,
                smoothed.report.equator_radius_out,
                smoothed.report.eval_seconds,
                smoothed.report.device
            );
            if let Some(path) = report {
                let text = serde_json::to_string_pretty(&smoothed.report).unwrap_or_default();
                std::fs::write(&path, text).map_err(|err| {
                    SimError::invalid_input(format!("cannot write {}: {err}", path.display()))
                })?;
            }
        }
        Command::Fuse {
            high,
            coarse,
            out,
            center,
            report,
        } => {
            let high = Mesh::read_obj(&high)?;
            let coarse = Mesh::read_obj(&coarse)?;
            let fused = fuse::fuse(
                &high,
                &coarse,
                &FuseOptions {
                    center: parse_center(&center)?,
                },
            )?;
            fused.mesh.write_obj(
                &out,
                &format!(
                    "fuse:dR=[{:.3e},{:.3e}]",
                    fused.report.delta_min, fused.report.delta_max
                ),
            )?;
            println!(
                "{} faces + coarse {} faces -> {} faces, {} vertices, ΔR [{:.3e}, {:.3e}] rms {:.3e}",
                fused.report.high_faces,
                fused.report.coarse_faces,
                fused.report.output_faces,
                fused.report.output_vertices,
                fused.report.delta_min,
                fused.report.delta_max,
                fused.report.delta_rms
            );
            if let Some(path) = report {
                let text = serde_json::to_string_pretty(&fused.report).unwrap_or_default();
                std::fs::write(&path, text).map_err(|err| {
                    SimError::invalid_input(format!("cannot write {}: {err}", path.display()))
                })?;
            }
        }
        Command::Validate { input, center } => {
            let mesh = Mesh::read_obj(&input)?;
            let center = parse_center(&center)?.resolve(&mesh)?;
            let samples = mesh.radial_samples(center)?;
            let (mut rmin, mut rmax) = (f64::MAX, 0.0_f64);
            for r in &samples.radius {
                rmin = rmin.min(*r);
                rmax = rmax.max(*r);
            }
            println!(
                "closed {}  vertices {}  faces {}  center {:.6} {:.6} {:.6}  radius [{:.6}, {:.6}]  unit {}",
                input.display(),
                mesh.vertex_count(),
                mesh.face_count(),
                center[0],
                center[1],
                center[2],
                rmin,
                rmax,
                mesh.length_unit
            );
        }
        Command::Spectrum {
            input,
            l_max,
            center,
            ridge,
        } => {
            let mesh = Mesh::read_obj(&input)?;
            let center = parse_center(&center)?.resolve(&mesh)?;
            let samples = mesh.radial_samples(center)?;
            let spectrum = astroid_evolution::shsmooth::fit::fit(
                l_max,
                &samples.dir,
                &samples.radius,
                &samples.weight,
                ridge,
            )?;
            let power = harmonics::degree_power(l_max, &spectrum.coeff);
            let report = serde_json::json!({
                "vertices": mesh.vertex_count(),
                "faces": mesh.face_count(),
                "l_max": l_max,
                "center": center,
                "power": power,
            });
            println!(
                "{}",
                serde_json::to_string_pretty(&report).unwrap_or_default()
            );
        }
        Command::Gpu => {
            let gpu = ShGpu::new()?;
            println!(
                "spherical-harmonic evaluation ready on {}",
                gpu.device_name()
            );
        }
    }
    Ok(())
}

fn main() -> ExitCode {
    SimError::exit(run(Cli::parse()))
}
