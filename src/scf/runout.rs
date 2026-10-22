//! Rotating-frame runout of eroded regolith (§2.8b, hypothesis A-C3).
//!
//! A packet eroded at a vertex slides over the frozen polyhedron in the frozen effective
//! gravity, with the depth-averaged `μ(I)` basal friction and the Coriolis acceleration
//! of the body frame `Ω = ω e_z`. Where the normal reaction vanishes it leaves on a
//! ballistic hop and lands inelastically. Its arrival point carries the eroded mass to
//! the deposition spread of §2.8, so the mass balance is unchanged.

use rayon::prelude::*;

use crate::common::math::DVec3;
use crate::rheo::Material;
use crate::rheo::closure::dynamic_friction;

use super::body::Topology;
use super::forward::Loading;
use super::transport::{DirIndex, locate_face, locate_vertex};

/// Substep fraction `θ_r` (§2.8b).
pub const THETA: f64 = 0.2;
/// Escape radius as a multiple of the largest surface radius.
const ESCAPE: f64 = 3.0;
/// Above this multiple of the largest radius a packet climbing under outward effective
/// gravity has escaped.
const EARLY: f64 = 1.05;
/// Below this multiple the face under a flying packet is looked up every substep.
const LOOKUP: f64 = 1.02;
/// Path samples kept per packet for the viewer.
pub const PATH_SAMPLES: usize = 12;
/// Hard cap on substeps; a packet that reaches it is counted as timed out.
const MAX_SUBSTEPS: usize = 100_000;

/// The frozen surface and loads of one step.
pub struct RunoutField<'a> {
    pub topo: &'a Topology,
    pub index: &'a DirIndex,
    /// Vertex positions (m).
    pub positions: &'a [DVec3],
    /// `g_eff` at the face centres (step 1).
    pub g_face: &'a [DVec3],
    pub load: Loading,
    pub material: &'a Material,
    /// `T_run` (s) and `ε_stop`.
    pub time_limit: f64,
    pub stop: f64,
    /// Substep fraction `θ_r` (`THETA` in production).
    pub theta: f64,
}

/// Where one packet came to rest, and how.
#[derive(Clone, Debug)]
pub struct Arrival {
    pub source: usize,
    /// Vertices and barycentric weights of the arrival point.
    pub ids: [usize; 3],
    pub weights: [f64; 3],
    /// Surface distance travelled (m) and the change of |latitude| (rad).
    pub distance: f64,
    pub dlat: f64,
    pub hops: usize,
    pub max_hop: f64,
    pub escaped: bool,
    pub timed_out: bool,
    /// `PATH_SAMPLES` points along the path (m); `w = 1` while airborne.
    pub path: Vec<[f64; 4]>,
}

/// Per-step summary (mass-weighted where stated).
#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
pub struct RunoutReport {
    pub traced: usize,
    pub mean_distance: f64,
    pub max_distance: f64,
    /// Mass-weighted mean change of |latitude| (rad); negative is equatorward.
    pub mean_dlat: f64,
    pub hops: usize,
    pub max_hop: f64,
    pub escaped: usize,
    pub timed_out: usize,
}

struct Faces {
    normal: Vec<DVec3>,
    /// Area-weighted vertex normals: the smooth surface that `κ_v` is measured on.
    smooth: Vec<DVec3>,
    /// A vertex of the face, for the ray–plane intersection.
    anchor: Vec<DVec3>,
    edge: Vec<f64>,
    /// Self-gravity at the face centre: `g_eff` minus the centrifugal and tidal terms.
    grav: Vec<DVec3>,
}

impl RunoutField<'_> {
    fn faces(&self) -> Faces {
        let nf = self.topo.faces.len();
        let mut smooth = vec![DVec3::ZERO; self.positions.len()];
        let mut f = Faces {
            smooth: Vec::new(),
            normal: Vec::with_capacity(nf),
            anchor: Vec::with_capacity(nf),
            edge: Vec::with_capacity(nf),
            grav: Vec::with_capacity(nf),
        };
        for (k, tri) in self.topo.faces.iter().enumerate() {
            let [a, b, c] = tri.map(|v| self.positions[v as usize]);
            let cross = (b - a).cross(c - a);
            let mut n = cross.try_normalize().unwrap_or(DVec3::Z);
            let centre = (a + b + c) / 3.0;
            if n.dot(centre) < 0.0 {
                n = -n;
            }
            for v in tri {
                smooth[*v as usize] += n * cross.length();
            }
            f.normal.push(n);
            f.anchor.push(a);
            f.edge.push(((b - a).length() + (c - b).length() + (a - c).length()) / 3.0);
            let g = self.g_face.get(k).copied().unwrap_or(DVec3::ZERO);
            f.grav.push(g - frame_terms(centre, self.load));
        }
        f.smooth = smooth.into_iter().map(|n| n.try_normalize().unwrap_or(DVec3::Z)).collect();
        f
    }

    /// Smooth normal at `x` on face `f` (barycentric blend of the vertex normals).
    fn smooth_normal(&self, faces: &Faces, f: usize, x: DVec3) -> DVec3 {
        let [i, j, k] = self.topo.faces[f].map(|v| v as usize);
        let (a, b, c) = (self.positions[i], self.positions[j], self.positions[k]);
        let n = (b - a).cross(c - a);
        let area = n.length_squared().max(1e-300);
        let wa = (c - b).cross(x - b).dot(n) / area;
        let wb = (a - c).cross(x - c).dot(n) / area;
        let wc = 1.0 - wa - wb;
        (faces.smooth[i] * wa + faces.smooth[j] * wb + faces.smooth[k] * wc)
            .try_normalize()
            .unwrap_or(faces.normal[f])
    }

    /// Trace every source vertex; `sources` are the vertices with `J_ero > 0`.
    pub fn trace(&self, sources: &[usize]) -> Vec<Arrival> {
        let faces = self.faces();
        let r_max = self
            .positions
            .iter()
            .map(|p| p.length())
            .fold(0.0, f64::max);
        sources
            .par_iter()
            .map(|&i| self.packet(&faces, r_max, i, DVec3::ZERO))
            .collect()
    }

    /// One packet from vertex `source` with the initial tangential velocity `v0`.
    pub fn launch(&self, source: usize, v0: DVec3) -> Arrival {
        let faces = self.faces();
        let r_max = self.positions.iter().map(|p| p.length()).fold(0.0, f64::max);
        self.packet(&faces, r_max, source, v0)
    }

    /// The surface point and face in direction `dir` (ray–face intersection).
    fn on_surface(&self, faces: &Faces, dir: DVec3) -> (usize, DVec3) {
        let f = locate_face(self.topo, self.index, dir);
        let n = faces.normal[f];
        let den = n.dot(dir);
        let r = if den > 1e-12 {
            n.dot(faces.anchor[f]) / den
        } else {
            faces.anchor[f].length()
        };
        (f, dir * r)
    }

    fn packet(&self, faces: &Faces, r_max: f64, source: usize, v0: DVec3) -> Arrival {
        let m = self.material;
        let s = &m.spec;
        let omega = DVec3::new(0.0, 0.0, self.load.omega);
        let start = self.positions[source];
        let start_dir = start.try_normalize().unwrap_or(DVec3::Z);
        let (mut face, mut x) = self.on_surface(faces, start_dir);
        // Surface point below the packet (equal to x while sliding).
        let mut foot = x;
        let mut foot_radius = x.length();
        let mut v = v0;
        let (mut t, mut kappa) = (0.0, 0.0);
        let mut airborne = false;
        let (mut distance, mut hops, mut max_hop) = (0.0, 0usize, 0.0_f64);
        let mut end = End::TimedOut;
        let mut trail = Trail::new(x);
        for _ in 0..MAX_SUBSTEPS {
            if t >= self.time_limit {
                break;
            }
            let edge = faces.edge[face].max(1e-9);
            if !airborne {
                let n = faces.normal[face];
                let g = self.g_face[face];
                let g_t = g - n * g.dot(n);
                let speed = v.length();
                let normal = -g.dot(n) + 2.0 * omega.cross(v).dot(n) - kappa * speed * speed;
                if normal <= 0.0 {
                    airborne = true;
                    hops += 1;
                    continue;
                }
                let v_stop = self.stop * (g.length() * s.h_s).sqrt();
                if speed <= v_stop && g_t.length() <= s.mu_s * normal {
                    end = End::Arrested;
                    break;
                }
                let accel = g_t.length() + friction(m, speed, normal) * normal;
                let tau = substep(self.theta, edge, speed, accel, self.load.omega)
                    .min(self.time_limit - t);
                // Drive, the exact tangential Coriolis turn, then friction without reversal.
                let v_old = v;
                v += g_t * tau;
                let angle = -2.0 * omega.dot(n) * tau;
                v = v * angle.cos() + n.cross(v) * angle.sin();
                let sp = v.length();
                let drop = friction(m, sp, normal) * normal * tau;
                v = if sp > drop { v * ((sp - drop) / sp) } else { DVec3::ZERO };
                // Back onto Γ along the ray, and onto the new tangent plane at the same speed.
                // Trapezoidal drift: exact for a constant deceleration.
                let dir = (x + (v_old + v) * (0.5 * tau)).try_normalize().unwrap_or(start_dir);
                let (next_face, next) = self.on_surface(faces, dir);
                let step = next - x;
                let len = step.length();
                let n_next = faces.normal[next_face];
                // κ_v on the smooth surface: facet edges would otherwise read as kinks.
                let (s0, s1) = (
                    self.smooth_normal(faces, face, x),
                    self.smooth_normal(faces, next_face, next),
                );
                kappa = if len > 1e-12 { (s1 - s0).dot(step) / (len * len) } else { 0.0 };
                let sp = v.length();
                v = (v - n_next * v.dot(n_next)).try_normalize().map_or(DVec3::ZERO, |u| u * sp);
                distance += len;
                x = next;
                foot = next;
                face = next_face;
                t += tau;
            } else {
                // Above the highest point of Γ the face below is kept, not looked up.
                let high = x.length() > LOOKUP * r_max;
                if !high {
                    let dir = x.try_normalize().unwrap_or(start_dir);
                    let (below, ground) = self.on_surface(faces, dir);
                    face = below;
                    foot_radius = ground.length();
                }
                let r = x.length().max(1e-9);
                let g = faces.grav[face] * (foot_radius * foot_radius / (r * r)) + frame_terms(x, self.load);
                // Outward, under outward effective gravity, well above the body: beyond the
                // synchronous radius the outward pull grows with height, so it does not return.
                let up = x / r;
                if r > EARLY * r_max && v.dot(up) > 0.0 && g.dot(up) >= 0.0 {
                    end = End::Escaped;
                    break;
                }
                let tau = substep(self.theta, edge, v.length(), g.length(), self.load.omega)
                    .min(self.time_limit - t);
                // Exact Coriolis turn about e_z, then gravity, then the trapezoidal drift.
                let angle = -2.0 * self.load.omega * tau;
                let (c, sn) = (angle.cos(), angle.sin());
                let v_old = v;
                v = DVec3::new(c * v.x - sn * v.y, sn * v.x + c * v.y, v.z) + g * tau;
                x += (v_old + v) * (0.5 * tau);
                t += tau;
                if x.length() > ESCAPE * r_max {
                    end = End::Escaped;
                    break;
                }
                if x.length() <= LOOKUP * r_max {
                    let dir = x.try_normalize().unwrap_or(start_dir);
                    let (landing, ground) = self.on_surface(faces, dir);
                    distance += (ground - foot).length();
                    foot = ground;
                    face = landing;
                    foot_radius = ground.length();
                    let height = x.length() - foot_radius;
                    max_hop = max_hop.max(height);
                    if height <= 0.0 {
                        // Perfectly inelastic in the normal direction (e_n = 0).
                        let n = faces.normal[landing];
                        v -= n * v.dot(n);
                        x = ground;
                        kappa = 0.0;
                        airborne = false;
                    }
                } else {
                    max_hop = max_hop.max(x.length() - foot_radius);
                }
            }
            trail.push([x.x, x.y, x.z, f64::from(u8::from(airborne))]);
        }
        if matches!(end, End::Escaped) {
            x = start;
            distance = 0.0;
        }
        let end_dir = x.try_normalize().unwrap_or(start_dir);
        let (ids, weights) = locate_vertex(self.topo, self.index, end_dir);
        let lat = |p: DVec3| (p.z / p.length().max(1e-300)).clamp(-1.0, 1.0).asin().abs();
        Arrival {
            source,
            ids,
            weights,
            distance,
            dlat: lat(x) - lat(start),
            hops,
            max_hop,
            escaped: matches!(end, End::Escaped),
            timed_out: matches!(end, End::TimedOut),
            path: trail.samples(),
        }
    }
}

enum End {
    Arrested,
    Escaped,
    TimedOut,
}

/// `τ = θ_r min(ℓ/|v|, √(ℓ/|a|), |v|/|a|, 1/(2ω))`: the third bound resolves a
/// decelerating packet (absent at rest), the last the Coriolis turn.
fn substep(theta: f64, edge: f64, speed: f64, accel: f64, omega: f64) -> f64 {
    let a = accel.max(1e-300);
    let mut tau = (edge / speed.max(1e-300)).min((edge / a).sqrt());
    if speed > 0.0 {
        tau = tau.min(speed / a);
    }
    if omega > 0.0 {
        tau = tau.min(0.5 / omega);
    }
    theta * tau
}

/// Path points with bounded memory: past `CAP` points every other one is dropped and the
/// stride doubles, so any path keeps an even record of at most `CAP` points.
struct Trail {
    points: Vec<[f64; 4]>,
    last: [f64; 4],
    stride: usize,
    count: usize,
}

const CAP: usize = 256;

impl Trail {
    fn new(x: DVec3) -> Self {
        let p = [x.x, x.y, x.z, 0.0];
        Self {
            points: vec![p],
            last: p,
            stride: 1,
            count: 0,
        }
    }

    fn push(&mut self, p: [f64; 4]) {
        self.last = p;
        self.count += 1;
        if self.count % self.stride != 0 {
            return;
        }
        self.points.push(p);
        if self.points.len() > CAP {
            let mut i = 0;
            self.points.retain(|_| {
                i += 1;
                i % 2 == 1
            });
            self.stride *= 2;
        }
    }

    /// `PATH_SAMPLES` evenly spaced points, ending at the newest position.
    fn samples(&self) -> Vec<[f64; 4]> {
        let mut all = self.points.clone();
        if all.last() != Some(&self.last) {
            all.push(self.last);
        }
        let n = all.len();
        (0..PATH_SAMPLES)
            .map(|k| all[(k * (n - 1)) / (PATH_SAMPLES - 1)])
            .collect()
    }
}

/// Centrifugal and tidal parts of `g_eff` at `x` (`effective_gravity` without `∇U`).
fn frame_terms(x: DVec3, l: Loading) -> DVec3 {
    let o2 = l.omega * l.omega;
    DVec3::new(o2 * x.x, o2 * x.y, 0.0) + DVec3::new(2.0 * x.x, -x.y, -x.z) * l.tide
}

/// Depth-averaged basal `μ(I)` of a Bagnold layer of thickness `h_s` (§2.8b): basal shear
/// rate `5|v|/(2h_s)`, basal pressure `ρ_bulk N h_s`, plus `p_c` in the inertial number.
fn friction(m: &Material, speed: f64, normal: f64) -> f64 {
    let s = &m.spec;
    let shear = 2.5 * speed / s.h_s;
    let p_star = m.rho_bulk * normal.max(0.0) * s.h_s + m.p_c;
    dynamic_friction(m, shear, p_star).0
}

/// Move each source's eroded mass rate `J_ero A` to its arrival point (barycentric split)
/// and return the source density per vertex; the sum `Σ J A` is unchanged.
pub fn moved_source(area: &[f64], j_ero: &[f64], arrivals: &[Arrival]) -> Vec<f64> {
    let nv = area.len();
    let mut mass = vec![0.0; nv];
    let mut moved = vec![false; nv];
    for a in arrivals {
        let rate = j_ero.get(a.source).copied().unwrap_or(0.0) * area[a.source];
        if rate <= 0.0 {
            continue;
        }
        moved[a.source] = true;
        for k in 0..3 {
            mass[a.ids[k]] += rate * a.weights[k];
        }
    }
    for i in 0..nv {
        if !moved[i] {
            mass[i] += j_ero[i] * area[i];
        }
    }
    mass.iter()
        .zip(area)
        .map(|(m, a)| if *a > 0.0 { m / a } else { 0.0 })
        .collect()
}

/// Mass-weighted summary of one step's runout; `weights[k]` is the mass of packet `k`.
pub fn report(weights: &[f64], arrivals: &[Arrival]) -> RunoutReport {
    let mut r = RunoutReport {
        traced: arrivals.len(),
        ..RunoutReport::default()
    };
    let mut total = 0.0;
    for (a, &w) in arrivals.iter().zip(weights) {
        total += w;
        r.mean_distance += w * a.distance;
        r.mean_dlat += w * a.dlat;
        r.max_distance = r.max_distance.max(a.distance);
        r.hops += usize::from(a.hops > 0);
        r.max_hop = r.max_hop.max(a.max_hop);
        r.escaped += usize::from(a.escaped);
        r.timed_out += usize::from(a.timed_out);
    }
    if total > 0.0 {
        r.mean_distance /= total;
        r.mean_dlat /= total;
    }
    r
}

/// Mass of each eroded packet over a step of length `dt`: `J_ero A dt`.
pub fn eroded_masses(area: &[f64], j_ero: &[f64], arrivals: &[Arrival], dt: f64) -> Vec<f64> {
    arrivals
        .iter()
        .map(|a| j_ero.get(a.source).copied().unwrap_or(0.0) * area[a.source] * dt)
        .collect()
}

/// The relief of one cone solution, routed (§2.8b, second part).
pub struct Routing {
    /// Radial deposit `δ` per vertex.
    pub delta: Vec<f64>,
    pub arrivals: Vec<Arrival>,
    /// Mass `m_i` of each routed packet, in the order of `arrivals`.
    pub masses: Vec<f64>,
}

/// Route the mass that the cone removed from the slope vertices: `removed[i] = r_i`
/// (radial, zero off the slope set), `jac[i] = (DM)_i`. The packets run out, their mass
/// is spread by the kernel of §2.8, and `δ_j = (K m)_j A_j / (DM)_j`, so that
/// `Σ (DM)_j δ_j = Σ m_i`.
pub fn route_relief(
    field: &RunoutField,
    area: &[f64],
    jac: &[f64],
    removed: &[f64],
    prm: &super::params::ModeCParams,
) -> Routing {
    let nv = area.len();
    let sources: Vec<usize> = (0..nv).filter(|&i| removed[i] > 0.0 && jac[i] > 0.0).collect();
    let arrivals = field.trace(&sources);
    let masses: Vec<f64> = arrivals.iter().map(|a| jac[a.source] * removed[a.source]).collect();
    // Mass per area at the sources, carried to the arrival points, then spread.
    let mut per_area = vec![0.0; nv];
    for (a, m) in arrivals.iter().zip(&masses) {
        per_area[a.source] = m / area[a.source];
    }
    let moved = moved_source(area, &per_area, &arrivals);
    let spread = super::transport::deposit(field.topo, area, &moved, prm);
    let delta = (0..nv)
        .map(|j| if jac[j] > 0.0 { spread[j] * area[j] / jac[j] } else { 0.0 })
        .collect();
    Routing {
        delta,
        arrivals,
        masses,
    }
}

/// The `count` heaviest paths (by packet mass), `PATH_SAMPLES` points each.
pub fn paths(weights: &[f64], arrivals: &[Arrival], count: usize) -> Vec<[f64; 4]> {
    let mut order: Vec<(f64, &Arrival)> = weights
        .iter()
        .copied()
        .zip(arrivals)
        .filter(|(_, a)| !a.escaped)
        .collect();
    order.sort_by(|a, b| b.0.total_cmp(&a.0));
    order
        .into_iter()
        .take(count)
        .flat_map(|(_, a)| a.path.iter().copied())
        .collect()
}
