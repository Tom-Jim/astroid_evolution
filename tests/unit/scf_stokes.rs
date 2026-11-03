//! Unit tests of `scf::stokes` on the general domain of hypothesis (G).

mod ctx {
    pub use astroid_evolution::common::density::CauchyDensity;
    pub use astroid_evolution::common::error::SimError;
    pub use astroid_evolution::common::math::DVec3;
    pub use astroid_evolution::rheo::Material;
    pub use astroid_evolution::rheo::basis::sh_all;
    pub use astroid_evolution::rheo::closure::dynamic_friction;
    pub use astroid_evolution::scf::basis::{
        PresLayout, PresRows, SurfaceStrain, VelLayout, VelRows, Want, eval_pressure,
        eval_velocity_rays, surface_strain,
    };
    pub use astroid_evolution::scf::harmonics::SurfaceFit;
    pub use astroid_evolution::scf::params::StokesParams;
    pub use astroid_evolution::scf::quad::{gauss_jacobi_zeta2, sphere_grid};
    pub use nalgebra::{Cholesky, DMatrix, DVector, Dyn};
    pub use std::f64::consts::PI;
}
use ctx::*;

use std::sync::Arc;

use astroid_evolution::rheo::{MaterialSpec, SolverSpec};
use astroid_evolution::rtfp::mesh::BodyMesh;
use astroid_evolution::scf::body::{Body, Topology};
use astroid_evolution::scf::domain::SurfacePoints;
use astroid_evolution::scf::stokes::*;
use astroid_evolution::scf::surface::SurfaceIndex;

pub fn material() -> Material {
    Material::new(
        MaterialSpec {
            nu0: 0.6,
            z0: 6.0,
            d_p: 0.05,
            rho_p: 2500.0,
            mu_s: 0.38,
            mu_2: 0.64,
            i_0: 0.279,
            cohesion: 25.0,
            cohesion_surface: 25.0,
            cohesion_depth: 1.0,
            c_c: 1.0,
            f_h: 1.0e-4,
            c_t: 1.0,
            gamma_eps: 1.0e-9,
            p_min: 1.0e-6,
            h_s: 5.0,
            eps_overhang: 0.01,
            mu_proj_mode: "dynamic".into(),
        },
        SolverSpec {
            max_iter: 100,
            tol: 1.0e-12,
        },
    )
    .unwrap()
}

pub fn params() -> StokesParams {
    StokesParams {
        l_surface: 4,
        l_velocity: 3,
        l_pressure: 4,
        k_pressure: 4,
        n_theta: 10,
        n_phi: 20,
        n_radial: 5,
        p_ext: 0.0,
        picard_max: 60,
        tol_picard: 1e-8,
        tol_momentum: 1e-6,
        theta_eta: 1.0,
        continuation: 2,
        rigid_sigma_min: 1e-6,
        solver: "primal-dual".into(),
        pd_tol: 1e-8,
        pd_max: 400_000,
        pd_pressure_every: 10,
        n_boundary: 1024,
        cut_depth: 3,
    }
}

/// The Stokes operators of a uniform polyhedral sphere of radius `r0` (§8.0).
pub fn sphere_ops(r0: f64, rho: f64, prm: &StokesParams) -> Operators {
    let (_, dirs) = Topology::icosphere(4);
    let positions: Vec<DVec3> = dirs.iter().map(|d| *d * r0).collect();
    ops_on(&positions, rho, prm, None, None)
}

/// The operators of the level-4 icosphere body with vertices `positions`, at the basis
/// scale `r_ref` (default `R_*`), taking node rows from `prev` (§8.1).
fn ops_on(positions: &[DVec3], rho: f64, prm: &StokesParams, r_ref: Option<f64>, prev: Option<Reuse>) -> Operators {
    let (topo, _) = Topology::icosphere(4);
    let nv = positions.len();
    let body = Body::new(Arc::new(topo), positions.to_vec(), rho, vec![], vec![0.0; nv]).unwrap();
    let mesh = BodyMesh::build(&body.mesh().unwrap()).unwrap();
    let index = SurfaceIndex::build(&body.positions, &body.topo.faces);
    let faces = SurfacePoints::faces(&body.face_geometry());
    let surf = SurfaceFit::sphere(r_ref.unwrap_or_else(|| body.r_star()));
    Operators::build(&surf, &body.density().unwrap(), &mesh, &index, faces, prm, prev).unwrap()
}

/// Largest entry difference of two row blocks, relative to the largest entry of `b`.
fn rel_diff(a: &DMatrix<f32>, b: &DMatrix<f32>) -> f64 {
    assert_eq!(a.shape(), b.shape());
    let scale = b.iter().fold(0.0_f32, |m, x| m.max(x.abs())) as f64;
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs() as f64).fold(0.0, f64::max) / scale
}

fn rows_diff(a: &Operators, b: &Operators) -> f64 {
    [(&a.rows.v, &b.rows.v), (&a.rows.d, &b.rows.d), (&a.rows.lap, &b.rows.lap), (&a.rows.gd, &b.rows.gd)]
        .into_iter()
        .map(|(x, y)| rel_diff(x, y))
        .fold(0.0, f64::max)
}

/// The pressure values `P c + lift` of both operators for one fixed coefficient vector.
fn pressure_diff(a: &Operators, b: &Operators) -> f64 {
    let fill = |n: usize, seed: usize| -> Vec<f64> { (0..n).map(|k| ((k * 7919 + seed) % 101) as f64 / 101.0 - 0.5).collect() };
    let (c, lift) = (fill(a.play.n_p(), 0), fill(a.play.n_lift(), 13));
    let (pa, pb) = (a.vol_pressure(&c, &lift), b.vol_pressure(&c, &lift));
    let scale = pb.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
    pa.iter().zip(&pb).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max) / scale
}

/// Node rows taken over from the previous geometry (§8.1) are those of the new nodes:
/// exactly on the same body, to `O(L δ / R_ref)` on a body whose nodes moved by
/// `δ ≤ δ_R R_ref`, and never across a larger move or another basis scale.
#[test]
fn reused_node_rows_are_the_rows_of_the_new_nodes() {
    let (r0, rho) = (1000.0, 2000.0);
    let prm = params();
    let (_, dirs) = Topology::icosphere(4);
    let at = |amp: f64| -> Vec<DVec3> {
        dirs.iter().enumerate().map(|(i, d)| *d * (r0 * (1.0 + amp * ((i * 37 % 17) as f64 / 17.0)))).collect()
    };
    let base = at(0.0);
    let r_ref = 1.02 * r0;
    let fresh = |p: &[DVec3]| ops_on(p, rho, &prm, Some(r_ref), None);
    let reused = |p: &[DVec3]| ops_on(p, rho, &prm, Some(r_ref), Some(fresh(&base).into_reuse()));

    // The same body: every node is an old node, and the rows are bit for bit the fresh ones.
    let (a, b) = (reused(&base), fresh(&base));
    assert_eq!(a.vol.pts, b.vol.pts);
    assert_eq!(rows_diff(&a, &b), 0.0);
    assert_eq!(pressure_diff(&a, &b), 0.0);

    // Nodes moved by about 3·10⁻⁶ R: inside the reuse distance, so the rows are the old
    // nodes' (they differ from a fresh build), and off by O(L δ / R) only.
    let moved = at(3.0e-6);
    let (a, b) = (reused(&moved), fresh(&moved));
    assert_eq!(a.vol.pts, b.vol.pts);
    let (dv, dp) = (rows_diff(&a, &b), pressure_diff(&a, &b));
    assert!(dv > 0.0 && dp > 0.0, "no node row was reused ({dv:e}, {dp:e})");
    assert!(dv < 1.0e-4 && dp < 1.0e-4, "reused rows off by {dv:.2e} (velocity), {dp:.2e} (pressure)");

    // Nodes moved by 10⁻³ R are new nodes: the rows are the fresh ones.
    let far = at(1.0e-3);
    let (a, b) = (reused(&far), fresh(&far));
    assert_eq!(rows_diff(&a, &b), 0.0);
    assert_eq!(pressure_diff(&a, &b), 0.0);

    // Rows of another basis scale are other functions and are not taken over.
    let other = ops_on(&base, rho, &prm, Some(1.03 * r0), Some(fresh(&base).into_reuse()));
    let own = ops_on(&base, rho, &prm, Some(1.03 * r0), None);
    assert_eq!(rows_diff(&other, &own), 0.0);
}

/// A uniform sphere in its own gravity is at rest, and the Nitsche pressure (G3) is the
/// hydrostatic `ρ k (R² − r²) / 2` up to the polyhedral approximation of the sphere.
#[test]
fn uniform_sphere_is_hydrostatic_and_at_rest() {
    let (r0, rho, g) = (1000.0, 2000.0, 6.6743e-11);
    let k = 4.0 * PI / 3.0 * g * rho;
    let prm = params();
    let ops = sphere_ops(r0, rho, &prm);
    let g_eff: Vec<DVec3> = ops.vol.pts.iter().map(|x| *x * -k).collect();
    let g_bnd: Vec<DVec3> = ops.bnd.pts.iter().map(|x| *x * -k).collect();
    let tide = vec![[0.0; 6]; ops.vol.pts.len()];
    let tide_bnd = vec![[0.0; 6]; ops.bnd.pts.len()];
    let tide_nn = vec![0.0; ops.faces.pts.len()];
    let strength = Strengths::profile(&ops, &material());
    let ld = Loads {
        g_eff: &g_eff,
        tide: &tide,
        g_bnd: &g_bnd,
        tide_bnd: &tide_bnd,
        tide_nn: &tide_nn,
        g_const: g,
        strength: &strength,
        dilatancy: None,
    };
    let sol = solve(&ops, &ld, &prm, &material(), None).unwrap();
    let umax = sol.u.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    assert!(umax < 1e-12, "velocity {umax}");
    let centre = 0.5 * rho * k * r0 * r0;
    for (p, x) in ops.vol.pts.iter().enumerate() {
        let exact = 0.5 * rho * k * (r0 * r0 - x.length_squared());
        assert!((sol.p_tilde[p] - exact).abs() < 5e-3 * centre, "p at r={}: {} vs {exact}", x.length(), sol.p_tilde[p]);
    }
    assert!(sol.eps_rot < 1e-2, "rot {}", sol.eps_rot);
}

/// The loads of a uniform sphere (`∇U = −k x`) in the Hill field of strength `s = n²`:
/// `g_eff = −k x + s (3x, 0, −z)` and `T_tide = C dev H_W` with `H_W = −k I + s diag(3, 0, −1)`.
struct Hill {
    g_vol: Vec<DVec3>,
    tide: Vec<[f64; 6]>,
    g_bnd: Vec<DVec3>,
    tide_bnd: Vec<[f64; 6]>,
    tide_nn: Vec<f64>,
    strength: Strengths,
}

impl Hill {
    fn new(ops: &Operators, m: &Material, k: f64, s: f64) -> Self {
        let g = |x: &DVec3| *x * -k + DVec3::new(3.0 * x.x, 0.0, -x.z) * s;
        // dev diag(3, 0, −1) = diag(7, −2, −5)/3; the isotropic −k I has no deviator.
        let t = [7.0, -2.0, -5.0, 0.0, 0.0, 0.0].map(|v| m.tide_coef * s * v / 3.0);
        let nn = |n: &DVec3| t[0] * n.x * n.x + t[1] * n.y * n.y + t[2] * n.z * n.z;
        Self {
            g_vol: ops.vol.pts.iter().map(g).collect(),
            tide: vec![t; ops.vol.pts.len()],
            g_bnd: ops.bnd.pts.iter().map(g).collect(),
            tide_bnd: vec![t; ops.bnd.pts.len()],
            tide_nn: ops.faces.normals.iter().map(nn).collect(),
            strength: Strengths::profile(ops, m),
        }
    }

    fn loads(&self, g_const: f64) -> Loads<'_> {
        Loads {
            g_eff: &self.g_vol,
            tide: &self.tide,
            g_bnd: &self.g_bnd,
            tide_bnd: &self.tide_bnd,
            tide_nn: &self.tide_nn,
            g_const,
            strength: &self.strength,
            dilatancy: None,
        }
    }
}

/// The certified bracket of `L*` (§2.17.3) on a uniform sphere of radius 1 km in a Hill
/// tide. Its exact properties:
/// - `lo ≤ hi`, closing as the reweighting proceeds;
/// - without a tide the load has no deviatoric part and `L*` is the small discretization
///   residue of the polyhedral sphere;
/// - `L*` is homogeneous of degree 1 in the load and, without cohesion, of degree −1 in `μ`;
/// - it agrees with the primal–dual solve: certified `L* < 1` where the solve converges,
///   certified `L* > 1` where it reports a limit-load collapse.
#[test]
fn the_limit_load_bracket_is_certified_and_agrees_with_the_collapse_test() {
    use astroid_evolution::scf::stokes::limit::{LimitData, bracket, data};
    let (r0, rho, g) = (1000.0, 2000.0, 6.6743e-11);
    let k = 4.0 * PI / 3.0 * g * rho;
    let mut prm = params();
    prm.pd_max = 40_000;
    let m = material();
    let ops = sphere_ops(r0, rho, &prm);
    let at = |s: f64| {
        let h = Hill::new(&ops, &m, k, s);
        let d = data(&ops, &h.loads(g), &prm, &m).unwrap();
        (h, d)
    };
    let cohesion: Vec<f64> = Strengths::profile(&ops, &m).vol.iter().map(|l| l.c).collect();
    let (mu, p_min) = (m.spec.mu_2, m.spec.p_min);

    let (_, rest) = at(0.0);
    let b = bracket(&ops, &rest, &cohesion, mu, p_min, 1e-3).unwrap();
    assert!(b.lo <= b.hi && b.hi < 1e-2, "no tide: L* in [{}, {}]", b.lo, b.hi);

    let scan: Vec<(f64, f64, f64)> = [0.01, 0.1, 0.3, 1.0, 3.0]
        .iter()
        .map(|f| {
            let b = bracket(&ops, &at(f * k).1, &cohesion, mu, p_min, 1e-3).unwrap();
            assert!(b.lo <= b.hi * (1.0 + 1e-12), "s = {f} k: lo {} > hi {}", b.lo, b.hi);
            (*f, b.lo, b.hi)
        })
        .collect();
    eprintln!("L* scan (s/k, lo, hi): {scan:?}");
    let (low, high) = (scan[0], scan[scan.len() - 1]);
    assert!(low.2 < 1.0 && high.1 > 1.0, "the scan does not cross 1: {scan:?}");

    // Homogeneity in the load, and in 1/μ without cohesion.
    let (_, mid) = at(0.3 * k);
    let double = LimitData { load: &mid.load * 2.0, p0: mid.p0.clone() };
    let (b1, b2) = (
        bracket(&ops, &mid, &cohesion, mu, p_min, 1e-3).unwrap(),
        bracket(&ops, &double, &cohesion, mu, p_min, 1e-3).unwrap(),
    );
    assert!((b2.lo - 2.0 * b1.lo).abs() <= 1e-9 * b2.lo && (b2.hi - 2.0 * b1.hi).abs() <= 1e-9 * b2.hi);
    let none = vec![0.0; cohesion.len()];
    let (f1, f2) = (
        bracket(&ops, &mid, &none, mu, p_min, 1e-3).unwrap(),
        bracket(&ops, &mid, &none, 2.0 * mu, p_min, 1e-3).unwrap(),
    );
    assert!((2.0 * f2.lo - f1.lo).abs() <= 1e-9 * f1.lo && (2.0 * f2.hi - f1.hi).abs() <= 1e-9 * f1.hi);

    // The primal–dual solve: a state where the bracket certifies one, none where it
    // certifies failure.
    let (h, _) = at(low.0 * k);
    assert!(solve(&ops, &h.loads(g), &prm, &m, None).is_ok(), "no state at a certified L* < 1");
    let (h, _) = at(high.0 * k);
    let err = solve(&ops, &h.loads(g), &prm, &m, None).err().expect("a state past a certified L* > 1");
    assert_eq!(err.kind, "LimitLoadCollapse", "{err}");
}

#[test]
fn two_layer_cohesion_profile() {
    let mut m = material();
    m.spec.cohesion = 6.0;
    m.spec.cohesion_surface = 0.5;
    m.spec.cohesion_depth = 3.0;
    assert!((m.cohesion_at(0.0) - 0.5).abs() < 1e-15);
    assert!((m.cohesion_at(1e3) - 6.0).abs() < 1e-12);
    for d in [0.1, 1.0, 3.0, 10.0] {
        let h = 1e-6;
        let fd = (m.cohesion_at(d + h) - m.cohesion_at(d - h)) / (2.0 * h);
        assert!((fd - m.cohesion_slope(d)).abs() < 1e-7, "slope at {d}");
        assert!(m.cohesion_at(d) > m.cohesion_at(d * 0.5));
    }
    // A homogeneous profile gives the single-layer viscosity of §4.3.
    let one = material();
    let local = |c: f64| Local { c, ..Local::default() };
    let a = eta_eval(&one, &local(one.cohesion_at(4.0)), 1e-6, 30.0, 1e-9).eta;
    let b = eta_eval(&one, &local(one.spec.cohesion), 1e-6, 30.0, 1e-9).eta;
    assert_eq!(a, b);
}

/// The depth is the distance to the surface (G6), not the radial gap: the foot
/// `x − d ∇d` lies on `r = R(ω̂)`, the gap is normal there (`∇d = −n`), `|∇d| = 1`,
/// and `d` never exceeds the radial gap `R(x̂) − |x|`.
#[test]
fn chart_strain_rebuilds_the_cartesian_tensor() {
    use astroid_evolution::common::math::DVec3;
    use astroid_evolution::scf::mechanics::cartesian_strain;
    // A symmetric tensor and a skewed chart: e1, e2 neither unit nor orthogonal.
    let d = [[0.3, -0.2, 0.5], [-0.2, -0.7, 0.1], [0.5, 0.1, 0.4]];
    let apply = |v: DVec3| {
        let v = v.to_array();
        DVec3::new(
            d[0][0] * v[0] + d[0][1] * v[1] + d[0][2] * v[2],
            d[1][0] * v[0] + d[1][1] * v[1] + d[1][2] * v[2],
            d[2][0] * v[0] + d[2][1] * v[1] + d[2][2] * v[2],
        )
    };
    let e1 = DVec3::new(1.3, 0.2, -0.1);
    let e2 = DVec3::new(0.4, 0.9, 0.3);
    let n = e1.cross(e2).normalize();
    let rebuilt = cartesian_strain(
        [e1, e2],
        n,
        [e1.dot(e1), e1.dot(e2), e2.dot(e2)],
        [e1.dot(apply(e1)), e1.dot(apply(e2)), e2.dot(apply(e2))],
        [n.dot(apply(e1)), n.dot(apply(e2))],
        n.dot(apply(n)),
    );
    for i in 0..3 {
        for j in 0..3 {
            assert!((rebuilt[i][j] - d[i][j]).abs() < 1e-12, "{i}{j}: {}", rebuilt[i][j]);
        }
    }
}

/// Proposition 10.4.7.4 against Kelvin's incompressible sphere: under the body force
/// `ρ∇V`, `V = ½ xᵀGx` with `G` symmetric and traceless, the traction-free solution is
/// `u = α[(5/42) r² G x − (2/21) x V] + β G x` with `α = −21ρ/19`, `β = 4ρR²/19` (unit
/// modulus), and the pressure `q = −(2/19) ρ V`. Without physical libration both components
/// of the cycle are such loads (the tide's strength and, through `f − M`, its direction), and
/// the discrete cycle reproduces both.
#[test]
fn the_elastic_cycle_of_a_sphere_is_kelvins() {
    use astroid_evolution::scf::cycle::{Cycle, Libration};
    use astroid_evolution::scf::stokes::elastic::elastic_cycle;
    let (r0, rho) = (1000.0, 2000.0);
    let prm = params();
    let m = material();
    let ops = sphere_ops(r0, rho, &prm);
    let cyc = Cycle::new(1.0e-3, 0.01, Libration { chi: 0.0, gamma_s: 0.0, gamma_c: 0.0 });
    assert!(cyc.euler == [0.0, 0.0]);
    let el = elastic_cycle(&ops, &cyc, &m, &Strengths::profile(&ops, &m), 6.6743e-11).unwrap();
    // The polyhedral sphere has the volume of a sphere of radius R.
    let vol: f64 = ops.vol.w.iter().sum();
    let r = (3.0 * vol / (4.0 * PI)).cbrt();
    let (alpha, beta) = (-21.0 * rho / 19.0, 4.0 * rho * r * r / 19.0);
    let s2 = std::f64::consts::SQRT_2;
    for k in 0..2 {
        let gm = cyc.h[k].to_cols_array_2d();
        let (mut err_s, mut max_s, mut err_p, mut max_p) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
        for (i, x) in ops.vol.pts.iter().enumerate() {
            let gx = cyc.h[k] * *x;
            let (v, r2) = (0.5 * x.dot(gx), x.length_squared());
            let grad = |a: usize, b: usize| -> f64 {
                let d = f64::from(u8::from(a == b));
                alpha * (5.0 / 42.0 * (2.0 * gx[a] * x[b] + r2 * gm[b][a]) - 2.0 / 21.0 * (d * v + x[a] * gx[b])) + beta * gm[b][a]
            };
            let tau = |a: usize, b: usize| grad(a, b) + grad(b, a);
            let want = [tau(0, 0), tau(1, 1), tau(2, 2), s2 * tau(0, 1), s2 * tau(0, 2), s2 * tau(1, 2)].map(|t| t / s2);
            let got = el.sigma[k][i];
            err_s = err_s.max((0..6).map(|c| (got[c] - want[c]).abs()).fold(0.0, f64::max));
            max_s = max_s.max(want.iter().map(|w| w.abs()).fold(0.0, f64::max));
            err_p = err_p.max((el.dp[k][i] + 2.0 / 19.0 * rho * v).abs());
            max_p = max_p.max((2.0 / 19.0 * rho * v).abs());
        }
        eprintln!("Kelvin sphere, component {k}: stress error {:.2e}, pressure error {:.2e}", err_s / max_s, err_p / max_p);
        // The level-4 icosphere is not the sphere: about 1e-4 in the stress and 5e-3 in the
        // pressure, whose boundary datum sits on the facets.
        assert!(err_s < 1e-3 * max_s, "component {k}: stress {err_s:e} of {max_s:e}");
        assert!(err_p < 1e-2 * max_p, "component {k}: pressure {err_p:e} of {max_p:e}");
    }
}

/// Proposition 10.4.7.5 on the Hill sphere: without a cycle the brackets are the mean one;
/// with one, the peak bound lies below the shakedown bound, the shakedown bracket holds,
/// and the bounds grow with the cycle.
#[test]
fn the_cycle_brackets_are_ordered_and_reduce_to_the_mean() {
    use astroid_evolution::scf::cycle::{Cycle, Libration};
    use astroid_evolution::scf::stokes::elastic::{cycle_bracket, elastic_cycle};
    use astroid_evolution::scf::stokes::limit::{bracket, data};
    let (r0, rho, g) = (1000.0, 2000.0, 6.6743e-11);
    let k = 4.0 * PI / 3.0 * g * rho;
    let prm = params();
    let m = material();
    let ops = sphere_ops(r0, rho, &prm);
    let h = Hill::new(&ops, &m, k, 0.3 * k);
    let d = data(&ops, &h.loads(g), &prm, &m).unwrap();
    let cohesion: Vec<f64> = h.strength.vol.iter().map(|l| l.c).collect();
    let (mu, p_min) = (m.spec.mu_2, m.spec.p_min);
    let b = bracket(&ops, &d, &cohesion, mu, p_min, 1e-4).unwrap();
    let cyc = Cycle::new((0.3 * k).sqrt(), 1.0, Libration::new(1.0, 0.3, 100.0));
    let el = elastic_cycle(&ops, &cyc, &m, &h.strength, g).unwrap();
    let at = |scale: f64| cycle_bracket(&ops, &b, &d.load, &el, scale, &cohesion, mu, &d.p0, p_min, 32);
    let none = at(0.0);
    assert!((none.peak_lo - b.lo).abs() <= 1e-12 * b.lo && (none.shakedown_lo - b.lo).abs() <= 1e-12 * b.lo);
    assert!((none.shakedown_hi - b.hi).abs() <= 1e-9 * b.hi, "{} vs {}", none.shakedown_hi, b.hi);
    let mut last = none;
    for scale in [0.01, 0.03, 0.1] {
        let c = at(scale);
        assert!(c.peak_lo <= c.shakedown_lo * (1.0 + 1e-12), "{c:?}");
        assert!(c.shakedown_lo <= c.shakedown_hi, "the shakedown bracket fails: {c:?}");
        assert!(c.shakedown_lo > last.shakedown_lo && c.shakedown_hi > last.shakedown_hi, "{c:?} after {last:?}");
        last = c;
    }
}

/// Algorithm 10.4.7.6 on the Hill sphere: a cycle inside the strength is elastic; one beyond
/// it is plastic, and its plastic strains scale exactly as load over modulus (Proposition
/// 10.4.7.7(4)).
#[test]
fn the_steady_cycle_scales_as_load_over_modulus() {
    use astroid_evolution::scf::cycle::{Cycle, Libration};
    use astroid_evolution::scf::stokes::elastic::{CycleOptions, elastic_cycle, steady_cycle};
    use astroid_evolution::scf::stokes::limit::{bracket, data};
    let (r0, rho, g) = (1000.0, 2000.0, 6.6743e-11);
    let k = 4.0 * PI / 3.0 * g * rho;
    let prm = params();
    let m = material();
    let ops = sphere_ops(r0, rho, &prm);
    let h = Hill::new(&ops, &m, k, 0.3 * k);
    let d = data(&ops, &h.loads(g), &prm, &m).unwrap();
    let cohesion: Vec<f64> = h.strength.vol.iter().map(|l| l.c).collect();
    let (mu, p_min) = (m.spec.mu_2, m.spec.p_min);
    // The mean: an equilibrated static field at 90 % of the strength where it is largest.
    let b = bracket(&ops, &d, &cohesion, mu, p_min, 1e-4).unwrap();
    let creep: Vec<[f64; 6]> = b.stress.iter().map(|s| s.map(|v| 0.9 * v / b.hi)).collect();
    let trigger = vec![true; ops.vol.pts.len()];
    let run = |cycle_scale: f64, mu_e: f64, stress: f64| {
        let cyc = Cycle::new((cycle_scale * k).sqrt(), 1.0, Libration::new(1.0, 0.3, 100.0));
        let mut el = elastic_cycle(&ops, &cyc, &m, &h.strength, g).unwrap();
        el.sigma.iter_mut().flatten().for_each(|s| *s = s.map(|v| v * stress));
        el.dp.iter_mut().flatten().for_each(|p| *p *= stress);
        let mean: Vec<[f64; 6]> = creep.iter().map(|s| s.map(|v| v * stress)).collect();
        let p: Vec<f64> = d.p0.iter().map(|p| p * stress).collect();
        let c: Vec<f64> = cohesion.iter().map(|c| c * stress).collect();
        let opt = CycleOptions { rigidity: mu_e, phases: 16, cycles_max: 40, tol: 1e-6, tol_cycle: 1e-6 };
        steady_cycle(&ops, &el, &mean, &p, &c, mu, p_min, &trigger, &opt).unwrap()
    };
    let small = run(1e-4, 1e8, 1.0);
    assert_eq!(small.cycles, 0, "a cycle far inside the strength is elastic");
    assert!(small.ratchet.iter().all(|v| *v == 0.0) && small.utilization.iter().all(|u| *u < 1.0));
    let base = run(0.3, 1e8, 1.0);
    assert!(base.cycles > 0 && base.range.iter().any(|r| *r > 0.0), "a strong cycle must be plastic");
    let stiff = run(0.3, 2e8, 1.0);
    let loaded = run(0.3, 1e8, 3.0);
    let rel = |a: &[f64], b: &[f64], f: f64| {
        let scale = a.iter().map(|v| v.abs()).fold(0.0, f64::max).max(1e-300);
        a.iter().zip(b).map(|(x, y)| (x * f - y).abs()).fold(0.0, f64::max) / scale
    };
    assert!(rel(&base.range, &stiff.range, 0.5) < 1e-9, "range ∝ 1/μ_e");
    assert!(rel(&base.range, &loaded.range, 3.0) < 1e-9, "range ∝ load");
    assert!(rel(&base.ratchet, &stiff.ratchet, 0.5) < 1e-9 || base.ratchet.iter().all(|v| *v == 0.0));
    assert!(rel(&base.ratchet, &loaded.ratchet, 3.0) < 1e-9 || base.ratchet.iter().all(|v| *v == 0.0));
    assert_eq!((base.cycles, stiff.cycles, loaded.cycles), (base.cycles, base.cycles, base.cycles));
}
