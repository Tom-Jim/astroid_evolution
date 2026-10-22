//! Unit tests of `scf::transport`, moved out of `src/scf/transport.rs`.

mod ctx {
    pub use astroid_evolution::common::math::DVec3;
    pub use astroid_evolution::scf::body::Topology;
    pub use astroid_evolution::scf::params::ModeCParams;
    pub use nalgebra::{Matrix3, Vector3};
    pub use rayon::prelude::*;
}
use ctx::*;

use astroid_evolution::scf::transport::*;

/// The step-averaged erosion integrates `dρ_s/dt = −λ ρ_s` exactly, so the layer
/// left after any step is `ρ_s e^{−λ dt} ≥ 0`, and short steps recover the rate.
#[test]
fn step_averaged_erosion_is_exact_and_positive() {
    let prm = ModeCParams {
        k_e: 1e-3,
        sigma_0: 1.0,
        kernel_width: 0.2,
    };
    let (area, rho_s, movable, f) = (
        vec![1.0; 3],
        vec![9.0, 0.5, 0.0],
        vec![true; 3],
        vec![0.4, 0.4, 0.4],
    );
    let lambda = prm.k_e * 0.4 / prm.sigma_0;
    for dt in [1e-3, 1e2, 1e4, 1e8] {
        let (j, _, _) = erosion_over(&area, &rho_s, &movable, &f, &prm, dt);
        for i in 0..3 {
            let left = rho_s[i] - dt * j[i];
            assert!(
                left >= 0.0
                    && (left - rho_s[i] * (-lambda * dt).exp()).abs() <= 1e-12 * rho_s[i].max(1.0),
                "{dt} {i}"
            );
        }
    }
    let (j0, _, _) = erosion_over(&area, &rho_s, &movable, &f, &prm, 0.0);
    let (j1, _, _) = erosion_over(&area, &rho_s, &movable, &f, &prm, 1e-6);
    assert!((j0[0] - j1[0]).abs() < 1e-9 * j0[0]);
}

#[test]
fn mode_c_conserves_mass_exactly() {
    let topo = Topology::icosphere(2);
    let nv = topo.n_vertices();
    let area: Vec<f64> = (0..nv).map(|i| topo.vertex_solid[i] * 100.0).collect();
    let rho_s = vec![5.0; nv];
    let movable: Vec<bool> = (0..nv).map(|i| topo.dirs[i].z > 0.3).collect();
    let f: Vec<f64> = (0..nv).map(|i| 3.0 * topo.dirs[i].z).collect();
    let prm = ModeCParams {
        k_e: 0.1,
        sigma_0: 10.0,
        kernel_width: 0.3,
    };
    let cpu = mode_c_cpu(&topo, &area, &rho_s, &movable, &f, &prm);
    let ero: f64 = (0..nv).map(|i| cpu.j_ero[i] * area[i]).sum();
    let dep: f64 = (0..nv).map(|i| cpu.j_dep[i] * area[i]).sum();
    assert!(ero > 0.0);
    assert!((ero - dep).abs() < 1e-12 * ero);
    let gpu = mode_c(&topo, &area, &rho_s, &movable, &f, &prm);
    assert_eq!(gpu.j_ero, cpu.j_ero);
    assert_eq!(gpu.max_excess, cpu.max_excess);
    assert_eq!(gpu.eroded_rate, cpu.eroded_rate);
    let gpu_dep: f64 = (0..nv).map(|i| gpu.j_dep[i] * area[i]).sum();
    assert!(
        (ero - gpu_dep).abs() < 1e-6 * ero,
        "deposition residual {}",
        (ero - gpu_dep).abs() / ero
    );
    assert_deposition(&gpu.j_dep, &cpu.j_dep, prm.kernel_width);
}

#[test]
fn deposition_matches_the_direct_angular_kernel() {
    let topo = Topology::icosphere(3);
    let nv = topo.n_vertices();
    let area: Vec<f64> = (0..nv)
        .map(|i| topo.vertex_solid[i] * (1.0 + 0.01 * (i % 7) as f64))
        .collect();
    let rho_s: Vec<f64> = (0..nv)
        .map(|i| {
            if i % 3 == 0 {
                0.0
            } else {
                1.0 + (i % 5) as f64
            }
        })
        .collect();
    let movable: Vec<bool> = (0..nv).map(|i| i % 4 != 1).collect();
    let f: Vec<f64> = (0..nv).map(|i| topo.dirs[i].z * 2.0 - 0.2).collect();
    for kernel_width in [1.0e-4, 0.2, 0.3, 1.0] {
        let prm = ModeCParams {
            k_e: 0.1,
            sigma_0: 10.0,
            kernel_width,
        };
        let got = mode_c_cpu(&topo, &area, &rho_s, &movable, &f, &prm);
        let want = direct_mode_c(&topo, &area, &rho_s, &movable, &f, &prm);
        assert_field_close(&got.j_ero, &want.j_ero, 1e-14);
        assert!((got.max_excess - want.max_excess).abs() <= 1e-14 * want.max_excess.abs());
        assert!((got.eroded_rate - want.eroded_rate).abs() <= 1e-14 * want.eroded_rate.abs());
        assert_deposition(&got.j_dep, &want.j_dep, kernel_width);
        let gpu = mode_c(&topo, &area, &rho_s, &movable, &f, &prm);
        assert_field_close(&gpu.j_ero, &want.j_ero, 1e-14);
        assert_deposition(&gpu.j_dep, &want.j_dep, kernel_width);
    }
}

/// Cap kernels stay at the pointwise reference. A zonal truncation is
/// accepted at the relative L1 stated for that kernel, not pointwise.
fn assert_deposition(got: &[f64], want: &[f64], width: f64) {
    let l1_num: f64 = got.iter().zip(want).map(|(g, w)| (g - w).abs()).sum();
    let l1_den = want.iter().map(|w| w.abs()).sum::<f64>().max(1e-30);
    let l1 = l1_num / l1_den;
    if astroid_evolution::scf::mode_c_spectral::spectrum(width).is_some() {
        assert!(
            l1 <= 1.0e-3,
            "spectral deposition L1 {l1:.3e} at kernel width {width}"
        );
        return;
    }
    assert_field_close(got, want, 1e-5);
}

fn assert_field_close(got: &[f64], want: &[f64], tol: f64) {
    assert_eq!(got.len(), want.len());
    let scale = want
        .iter()
        .map(|v| v.abs())
        .fold(0.0_f64, f64::max)
        .max(1e-30);
    let mut worst = 0.0_f64;
    let mut at = 0usize;
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let err = (g - w).abs() / scale;
        if err > worst {
            worst = err;
            at = i;
        }
    }
    assert!(
        worst <= tol,
        "deposition[{at}] gpu {} cpu {} relative {worst:.3e} (tolerance {tol:.1e})",
        got[at],
        want[at]
    );
}

#[test]
fn ryugu_zonal_sample_matches_the_cap_kernel() {
    use std::path::PathBuf;

    use astroid_evolution::scf::frame::vertex_areas;
    use astroid_evolution::scf::io::initial_body;
    use astroid_evolution::scf::params::{ModeCParams, Project};

    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("dist/.stage/assets/projects");
    let text = std::fs::read_to_string(dir.join("ryugu.toml")).expect("ryugu project");
    let project = Project::parse(&text, dir).expect("parse");
    let body = initial_body(&project).expect("body");
    let area = vertex_areas(&body);
    let width = project.mode_c.kernel_width;
    let beta = astroid_evolution::scf::mode_c_spectral::spectrum(width)
        .expect("Ryugu kernel width must pass the 1e-3 truncation gate");
    let nv = body.topo.n_vertices();
    let mut j_ero = vec![0.0; nv];
    let mut n_src = 0usize;
    for src in (0..nv).step_by((nv / 200).max(1)) {
        j_ero[src] = 1.0;
        n_src += 1;
        if n_src == 200 {
            break;
        }
    }
    let fast =
        astroid_evolution::scf::mode_c_spectral::deposit(&body.topo.dirs, &area, &j_ero, &beta)
            .expect("source norms stay positive");
    let cap = deposit_cpu(
        &body.topo,
        &area,
        &j_ero,
        &ModeCParams {
            k_e: 1.0,
            sigma_0: 1.0,
            kernel_width: width,
        },
    );
    let num: f64 = fast.iter().zip(&cap).map(|(a, b)| (a - b).abs()).sum();
    let den = cap.iter().map(|v| v.abs()).sum::<f64>().max(1e-30);
    let l1 = num / den;
    let ero: f64 = j_ero.iter().zip(&area).map(|(j, a)| j * a).sum();
    let dep: f64 = fast.iter().zip(&area).map(|(j, a)| j * a).sum();
    assert!(
        (ero - dep).abs() <= 1.0e-8 * ero,
        "spectral mass gap {}",
        (ero - dep).abs() / ero
    );
    assert!(
        l1 <= 1.0e-3,
        "Ryugu zonal sample L1 {l1:.3e} on {n_src} sources"
    );
    eprintln!("ryugu zonal sample L1={l1:.3e} sources={n_src}");
}

#[test]
fn cap_query_contains_every_vertex_inside_the_angle() {
    let topo = Topology::icosphere(4);
    let bins = DirIndex::build(&topo.dirs);
    let mut got = Vec::new();
    for alpha in [1.0e-4, 0.2, 0.8, 1.2, 3.2] {
        for src in (0..topo.n_vertices()).step_by(19) {
            bins.collect_within(topo.dirs[src], alpha, &mut got);
            got.sort_unstable();
            for (t, dir) in topo.dirs.iter().enumerate() {
                let ang = topo.dirs[src].dot(*dir).clamp(-1.0, 1.0).acos();
                if ang < alpha {
                    assert!(
                        got.binary_search(&t).is_ok(),
                        "missed vertex {t} from {src} at angle {ang} (cap {alpha})"
                    );
                }
            }
        }
    }
}

/// The dense loop this kernel replaces. Kept so a faster path cannot drift.
fn direct_mode_c(
    topo: &Topology,
    area: &[f64],
    rho_s: &[f64],
    movable: &[bool],
    f_delta: &[f64],
    prm: &ModeCParams,
) -> ModeC {
    let nv = topo.n_vertices();
    let mut j_ero = vec![0.0; nv];
    let mut max_excess = 0.0_f64;
    for i in 0..nv {
        if movable[i] && rho_s[i] > 0.0 {
            let ex = f_delta[i].max(0.0);
            max_excess = max_excess.max(ex);
            j_ero[i] = prm.k_e * rho_s[i] * ex / prm.sigma_0;
        }
    }
    let mut j_dep = vec![0.0; nv];
    let w = prm.kernel_width;
    let mut eroded = 0.0;
    let mut g = vec![0.0; nv];
    for src in 0..nv {
        if j_ero[src] <= 0.0 {
            continue;
        }
        let mass_rate = j_ero[src] * area[src];
        eroded += mass_rate;
        let mut norm = 0.0;
        for t in 0..nv {
            let cos = topo.dirs[src].dot(topo.dirs[t]).clamp(-1.0, 1.0);
            let ang = cos.acos();
            g[t] = if ang < 4.0 * w {
                (-0.5 * (ang / w).powi(2)).exp()
            } else {
                0.0
            };
            norm += g[t] * area[t];
        }
        for t in 0..nv {
            if g[t] > 0.0 {
                j_dep[t] += mass_rate * g[t] / norm;
            }
        }
    }
    ModeC {
        j_ero,
        j_dep,
        max_excess,
        eroded_rate: eroded,
    }
}

#[test]
fn transport_conserves_and_translates_a_bump() {
    let topo = Topology::icosphere(3);
    let nv = topo.n_vertices();
    // A rigid rotation about z at rate 0.05 rad per unit time.
    let flow = SurfaceFlow {
        c: topo
            .dirs
            .iter()
            .map(|d| DVec3::new(-d.y, d.x, 0.0) * 0.05)
            .collect(),
        div: vec![0.0; nv],
        grad_max: 0.05,
        tangent_det: vec![0.0; nv],
    };
    let field: Vec<f64> = topo
        .dirs
        .iter()
        .map(|d| (-(d.x - 1.0).powi(2) * 6.0 - d.y * d.y * 6.0 - d.z * d.z * 6.0).exp())
        .collect();
    let j = vec![1.0; nv];
    let (out, fix) = transport(&topo, &flow, &field, &j, &j, 1.0).unwrap();
    let mass = |f: &[f64]| (0..nv).map(|i| f[i] * topo.vertex_solid[i]).sum::<f64>();
    assert!((mass(&out) - mass(&field)).abs() < 1e-12 * mass(&field));
    assert!(fix < 0.1);
    // The bump centre moved from +x towards +y.
    let cy_before: f64 = (0..nv)
        .map(|i| field[i] * topo.dirs[i].y * topo.vertex_solid[i])
        .sum();
    let cy_after: f64 = (0..nv)
        .map(|i| out[i] * topo.dirs[i].y * topo.vertex_solid[i])
        .sum();
    assert!(cy_after > cy_before + 1e-3, "{cy_before} -> {cy_after}");
}

fn brute_nearest(dirs: &[DVec3], p: DVec3) -> usize {
    (0..dirs.len())
        .max_by(|a, b| dirs[*a].dot(p).partial_cmp(&dirs[*b].dot(p)).unwrap())
        .unwrap()
}

#[test]
fn direction_grid_matches_the_linear_scan() {
    let topo = Topology::icosphere(4);
    let index = DirIndex::build(&topo.dirs);
    for d in &topo.dirs {
        assert_eq!(index.nearest(&topo.dirs, *d), brute_nearest(&topo.dirs, *d));
    }
    let golden = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
    for i in 0..40 {
        let z = 1.0 - 2.0 * (i as f64 + 0.5) / 40.0;
        let r = (1.0 - z * z).sqrt();
        let p = DVec3::new(
            r * (golden * i as f64).cos(),
            r * (golden * i as f64).sin(),
            z,
        );
        assert_eq!(index.nearest(&topo.dirs, p), brute_nearest(&topo.dirs, p));
    }
    let mut dirs = topo.dirs.clone();
    dirs.push(dirs[3]);
    let tied = DirIndex::build(&dirs);
    assert_eq!(tied.nearest(&dirs, dirs[3]), brute_nearest(&dirs, dirs[3]));
    assert_eq!(brute_nearest(&dirs, dirs[3]), dirs.len() - 1);
    let field: Vec<f64> = (0..topo.n_vertices())
        .map(|i| (i as f64 * 0.17).sin())
        .collect();
    let face = topo.faces[10];
    let mid =
        (topo.dirs[face[0] as usize] + topo.dirs[face[1] as usize] + topo.dirs[face[2] as usize])
            .try_normalize()
            .unwrap();
    let nearest = index.nearest(&topo.dirs, mid);
    assert_eq!(nearest, brute_nearest(&topo.dirs, mid));
    let value = interp_vertex_field(&topo, &index, &field, mid);
    let [i, j, k] = face.map(|v| v as usize);
    let span = [field[i], field[j], field[k], field[nearest]];
    assert!(value >= span.into_iter().fold(f64::INFINITY, f64::min) - 1e-12);
    assert!(value <= span.into_iter().fold(f64::NEG_INFINITY, f64::max) + 1e-12);
}

// CPU references formerly kept in the source under cfg(test).
fn mode_c_cpu(
    topo: &Topology,
    area: &[f64],
    rho_s: &[f64],
    movable: &[bool],
    f_delta: &[f64],
    prm: &ModeCParams,
) -> ModeC {
    let (j_ero, max_excess, eroded) = erosion(area, rho_s, movable, f_delta, prm);
    let j_dep = deposit_cpu(topo, area, &j_ero, prm);
    ModeC {
        j_ero,
        j_dep,
        max_excess,
        eroded_rate: eroded,
    }
}

fn deposit_cpu(topo: &Topology, area: &[f64], j_ero: &[f64], prm: &ModeCParams) -> Vec<f64> {
    let nv = topo.n_vertices();
    let mut j_dep = vec![0.0; nv];
    let w = prm.kernel_width;
    let cutoff = 4.0 * w;
    let cos_reject = if cutoff < std::f64::consts::PI {
        cutoff.cos() - 1.0e-8
    } else {
        f64::NEG_INFINITY
    };
    let bins = DirIndex::build(&topo.dirs);
    let sources: Vec<usize> = (0..nv).filter(|&src| j_ero[src] > 0.0).collect();
    let batch = rayon::current_num_threads().max(1);
    for chunk in sources.chunks(batch) {
        let parts: Vec<(f64, Vec<(u32, f64)>)> = chunk
            .par_iter()
            .map(|&src| kernel_terms(&topo.dirs, &bins, src, area, w, cutoff, cos_reject))
            .collect();
        for (&src, (norm, terms)) in chunk.iter().zip(parts) {
            let mass_rate = j_ero[src] * area[src];
            for (t, weight) in terms {
                j_dep[t as usize] += mass_rate * weight / norm;
            }
        }
    }
    j_dep
}

/// Positive Gaussian weights of one source, in increasing target index.
///
/// Targets are the direction-bin superset of the cap `θ < cutoff`. A weight is
/// the same value the dense loop would store: clamp the dot, reject only when
/// the cosine is safely outside, otherwise `exp(-½(acos(c)/w)²)` when
/// `acos(c) < cutoff`.
fn kernel_terms(
    dirs: &[DVec3],
    bins: &DirIndex,
    src: usize,
    area: &[f64],
    w: f64,
    cutoff: f64,
    cos_reject: f64,
) -> (f64, Vec<(u32, f64)>) {
    let mut targets = Vec::new();
    bins.collect_within(dirs[src], cutoff, &mut targets);
    targets.sort_unstable();
    targets.dedup();
    let mut norm = 0.0;
    let mut terms = Vec::new();
    let origin = dirs[src];
    for t in targets {
        let cos_raw = origin.dot(dirs[t]);
        if cos_raw < cos_reject {
            continue;
        }
        let cos = cos_raw.clamp(-1.0, 1.0);
        let ang = cos.acos();
        if ang < cutoff {
            let weight = (-0.5 * (ang / w).powi(2)).exp();
            if weight > 0.0 {
                norm += weight * area[t];
                terms.push((t as u32, weight));
            }
        }
    }
    (norm, terms)
}
