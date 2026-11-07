//! Rotating-frame runout of eroded regolith (§10.0.2, hypothesis A-C3).
//!
//! A packet eroded at a vertex slides over the frozen polyhedron in the frozen effective
//! gravity, with the depth-averaged `μ(I)` basal friction and the Coriolis acceleration
//! of the body frame `Ω = ω e_z`. Where the normal reaction vanishes it leaves on a
//! ballistic hop and lands inelastically. Its arrival point carries the eroded mass to
//! the deposition spread of §10.0.2, so the mass balance is unchanged.

use rayon::prelude::*;

use crate::common::error::SimError;
use crate::common::math::{DMat3, DVec3, Sym6, sym_mat};
use crate::rheo::Material;
use crate::rheo::closure::dynamic_friction;

use super::body::Topology;
use super::forward::Loading;
use super::surface::{SurfaceIndex, barycentric};

/// Substep fraction `θ_r` (§10.0.2).
pub const THETA: f64 = 0.2;
/// Escape radius as a multiple of the largest surface radius.
const ESCAPE: f64 = 3.0;
/// Above this multiple of the largest radius a packet climbing under outward effective
/// gravity has escaped.
const EARLY: f64 = 1.05;
/// Above this fraction of the mean radius a flight takes the exterior self-gravity
/// instead of the face expansion, whose error grows as `(h/R̄)²` (§10.0.2).
pub const FAR: f64 = 0.1;
/// A far flight uses the linear expansion `g₀ + H₀ (x − x₀)` of an exterior value at `x₀`
/// while `|x − x₀| ≤ REUSE h`: the same `O((|x − x₀|/h)²)` budget as the face expansion.
const REUSE: f64 = 0.1;
/// A flight that needs an exterior value asks for a whole row of them along its predicted
/// arc, `PREFETCH · REUSE h` apart, at most `PREFETCH_MAX` (one batch instead of one call
/// per value: each exterior call has a fixed cost).
const PREFETCH: f64 = 0.5;
const PREFETCH_MAX: usize = 256;
/// Path samples kept per packet for the viewer.
pub const PATH_SAMPLES: usize = 12;
/// Hard cap on substeps; a packet that reaches it is counted as timed out.
const MAX_SUBSTEPS: usize = 100_000;

/// The frozen surface and loads of one step.
pub struct RunoutField<'a> {
    pub topo: &'a Topology,
    /// Closest-point and segment queries on `Γ` (G9).
    pub index: &'a SurfaceIndex,
    /// Vertex positions (m).
    pub positions: &'a [DVec3],
    /// `g_eff` at the face centres (step 1).
    pub g_face: &'a [DVec3],
    /// Finite-part Hessian `Hess U` of the self-potential at the face centres; empty for
    /// none. With `rho_face` (`ρ_K` there) and `grav_const` it gives the exterior one-sided
    /// Hessian `H⁺ = H^FP + 2πGρ_K n nᵀ` of the flight expansion (Proposition 2.7.1(5)).
    pub hess_face: &'a [Sym6],
    pub rho_face: &'a [f64],
    pub grav_const: f64,
    pub load: Loading,
    pub material: &'a Material,
    /// A-O (§10.3.6): the necks of the Hill region. A packet beyond a neck plane has left
    /// and its mass is lost; without them escape is the heuristic count of §10.0.2.
    pub necks: Option<&'a super::orbit::Necks>,
    /// A-Λ (§10.3.7): the tidal cycle; a packet flies in the field of its launch phase,
    /// spread over the packets by a low-discrepancy sequence, and the necks follow it.
    pub cycle: Option<&'a super::cycle::Cycle>,
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
    /// Escaped through a neck (A-O): the mass leaves the body and is not deposited.
    pub lost: bool,
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
    centre: Vec<DVec3>,
    /// Self-gravity at the face centre: `g_eff` minus the centrifugal and tidal terms.
    grav: Vec<DVec3>,
    /// Exterior one-sided Hessian `H⁺` of the self-potential at the face centre.
    hess: Vec<DMat3>,
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
            centre: Vec::with_capacity(nf),
            grav: Vec::with_capacity(nf),
            hess: Vec::with_capacity(nf),
        };
        for (k, tri) in self.topo.faces.iter().enumerate() {
            let [a, b, c] = tri.map(|v| self.positions[v as usize]);
            let cross = (b - a).cross(c - a);
            // Faces are consistently oriented (G): the cross product is outward.
            let n = cross.try_normalize().unwrap_or(DVec3::Z);
            let centre = (a + b + c) / 3.0;
            for v in tri {
                smooth[*v as usize] += n * cross.length();
            }
            f.normal.push(n);
            f.anchor.push(a);
            f.edge.push(((b - a).length() + (c - b).length() + (a - c).length()) / 3.0);
            let g = self.g_face.get(k).copied().unwrap_or(DVec3::ZERO);
            f.grav.push(g - frame_terms(centre, self.load));
            f.centre.push(centre);
            let h = match (self.hess_face.get(k), self.rho_face.get(k)) {
                (Some(h), Some(rho)) => {
                    let jump = 2.0 * std::f64::consts::PI * self.grav_const * rho;
                    sym_mat(*h) + DMat3::from_cols(n * n.x, n * n.y, n * n.z) * jump
                }
                (Some(h), None) => sym_mat(*h),
                _ => DMat3::ZERO,
            };
            f.hess.push(h);
        }
        f.smooth = smooth.into_iter().map(|n| n.try_normalize().unwrap_or(DVec3::Z)).collect();
        f
    }

    /// Smooth normal at `x` on face `f` (barycentric blend of the vertex normals).
    fn smooth_normal(&self, faces: &Faces, f: usize, x: DVec3) -> DVec3 {
        let [i, j, k] = self.topo.faces[f].map(|v| v as usize);
        let w = barycentric(x, self.positions[i], self.positions[j], self.positions[k]);
        (faces.smooth[i] * w[0] + faces.smooth[j] * w[1] + faces.smooth[k] * w[2])
            .try_normalize()
            .unwrap_or(faces.normal[f])
    }

    /// Trace every source vertex; `sources` are the vertices with `J_ero > 0`. Flights take
    /// their self-gravity from the face expansion at every height.
    pub fn trace(&self, sources: &[usize]) -> Vec<Arrival> {
        self.trace_with(sources, None)
            .unwrap_or_else(|_| unreachable!("no exterior evaluation was requested"))
    }

    /// Trace every source vertex. A flight higher than `FAR R̄` above `Γ` takes its
    /// self-gravity from `exterior` (§10.0.2). The packets run in parallel until each is done
    /// or waits for such a value; the waiting points go to `exterior` as one batch, and the
    /// round repeats.
    pub fn trace_with(
        &self,
        sources: &[usize],
        exterior: Option<&Exterior<'_>>,
    ) -> Result<Vec<Arrival>, SimError> {
        let packets = sources.iter().map(|&i| Packet::new(self, i, DVec3::ZERO)).collect();
        self.drive(packets, exterior)
    }

    /// One packet from vertex `source` with the initial velocity `v0`; an outward normal
    /// component starts it in flight (an ejection), a tangential one sliding.
    pub fn launch(&self, source: usize, v0: DVec3) -> Arrival {
        self.launch_with(source, v0, None)
            .unwrap_or_else(|_| unreachable!("no exterior evaluation was requested"))
    }

    /// `launch`, with far flights on the exterior self-gravity as in `trace_with`.
    pub fn launch_with(
        &self,
        source: usize,
        v0: DVec3,
        exterior: Option<&Exterior<'_>>,
    ) -> Result<Arrival, SimError> {
        let mut done = self.drive(vec![Packet::new(self, source, v0)], exterior)?;
        Ok(done.remove(0))
    }

    /// Runs `packets` to their ends, sending the exterior requests of each round in one batch.
    fn drive(&self, mut packets: Vec<Packet>, exterior: Option<&Exterior<'_>>) -> Result<Vec<Arrival>, SimError> {
        let faces = self.faces();
        let scale = Scale::of(self.positions);
        let far = exterior.is_some();
        let mut waiting: Vec<Vec<DVec3>> = packets
            .par_iter_mut()
            .map(|p| p.run(self, &faces, scale, far))
            .collect();
        if let Some(exterior) = exterior {
            // Each round sends every waiting flight's prefetch row in one batch.
            while waiting.iter().any(|w| !w.is_empty()) {
                let points: Vec<DVec3> = waiting.iter().flatten().copied().collect();
                let g = exterior(&points)?;
                if g.len() != points.len() {
                    return Err(SimError::solver(format!(
                        "exterior gravity returned {} values for {} points",
                        g.len(),
                        points.len()
                    )));
                }
                let mut values = g.into_iter();
                for (p, w) in packets.iter_mut().zip(&waiting) {
                    p.anchors.extend(w.iter().map(|x| {
                        let (g, h) = values.next().expect("one value per point");
                        (*x, g, h)
                    }));
                }
                waiting = packets
                    .par_iter_mut()
                    .zip(&waiting)
                    .map(|(p, w)| if w.is_empty() { Vec::new() } else { p.run(self, &faces, scale, far) })
                    .collect();
            }
        }
        Ok(packets.into_iter().map(|p| p.finish(self)).collect())
    }

    /// `π(x)`, the closest point of `Γ`, and its face (G9).
    fn ground(&self, x: DVec3) -> (usize, DVec3) {
        let c = self.index.closest(x);
        (c.face, c.point)
    }

    /// Self-gravity at `x` from the face expansion `g(c_f) + H⁺_f (x − c_f)` about the
    /// closest face (§10.0.2), and the height `|x − π(x)|` above `Γ`.
    fn expansion(&self, faces: &Faces, f: usize, foot: DVec3, x: DVec3) -> (DVec3, f64) {
        let g = faces.grav[f] + faces.hess[f] * (x - faces.centre[f]);
        (g, (x - foot).length())
    }
}

/// Self-gravity and its Hessian at a batch of points outside the body (the exterior
/// route of §3.11).
pub type Exterior<'a> = dyn Fn(&[DVec3]) -> Result<Vec<(DVec3, DMat3)>, SimError> + 'a;

/// Length scales of the frozen surface.
#[derive(Clone, Copy)]
struct Scale {
    /// Largest and mean vertex distance from the frame origin.
    r_max: f64,
    r_bar: f64,
}

impl Scale {
    fn of(positions: &[DVec3]) -> Self {
        let n = positions.len().max(1) as f64;
        Self {
            r_max: positions.iter().map(|p| p.length()).fold(0.0, f64::max),
            r_bar: positions.iter().map(|p| p.length()).sum::<f64>() / n,
        }
    }
}

/// The state of one packet between substeps.
#[derive(Clone)]
struct Packet {
    source: usize,
    start: DVec3,
    face: usize,
    x: DVec3,
    /// Surface point below the packet (equal to x while sliding).
    foot: DVec3,
    v: DVec3,
    t: f64,
    kappa: f64,
    airborne: bool,
    distance: f64,
    hops: usize,
    max_hop: f64,
    substeps: usize,
    /// Exterior values `(x₀, g₀, H₀)` received for this packet's far flights.
    anchors: Vec<(DVec3, DVec3, DMat3)>,
    /// `(g_n, τ_n)` of the last flight substep, whose velocity still lacks the end-point
    /// half of its kick (velocity Verlet).
    kick: Option<(DVec3, f64)>,
    end: Option<End>,
    trail: Trail,
    /// Orbital phase at launch (A-Λ).
    phase: f64,
}

impl Packet {
    fn new(field: &RunoutField, source: usize, v0: DVec3) -> Self {
        let start = field.positions[source];
        let (face, x) = field.ground(start);
        // Outward against the vertex normal: a tangential `v0` slides on any facet.
        let normal: DVec3 = field.topo.vertex_faces[source]
            .iter()
            .map(|&f| {
                let [a, b, c] = field.topo.faces[f].map(|v| field.positions[v as usize]);
                (b - a).cross(c - a)
            })
            .sum();
        let airborne = v0.dot(normal) > 1e-9 * v0.length() * normal.length();
        Self {
            source,
            start,
            face,
            x,
            foot: x,
            v: v0,
            t: 0.0,
            kappa: 0.0,
            airborne,
            distance: 0.0,
            hops: usize::from(airborne),
            max_hop: 0.0,
            substeps: 0,
            anchors: Vec::new(),
            kick: None,
            end: None,
            trail: Trail::new(x),
            // The golden-ratio sequence spreads the launch phases uniformly.
            phase: 2.0 * std::f64::consts::PI * ((source as f64 + 1.0) * 0.618_033_988_749_894_9).fract(),
        }
    }

    /// Substeps until the packet stops (empty), or, with `far`, until a flight above
    /// `FAR R̄` has no exterior value within `REUSE h`; it then returns the points of its
    /// prefetch row, the first being the current position.
    fn run(&mut self, field: &RunoutField, faces: &Faces, scale: Scale, far: bool) -> Vec<DVec3> {
        self.run_inner(field, faces, scale, far, None)
    }

    /// `run`, or with `row` the prediction of `prefetch`: missing exterior values are
    /// estimated and their points recorded instead of ending the run.
    fn run_inner(
        &mut self,
        field: &RunoutField,
        faces: &Faces,
        scale: Scale,
        far: bool,
        mut row: Option<&mut Vec<DVec3>>,
    ) -> Vec<DVec3> {
        let m = field.material;
        let s = &m.spec;
        let r_max = scale.r_max;
        while self.end.is_none() {
            if self.substeps >= MAX_SUBSTEPS || self.t >= field.time_limit {
                self.end = Some(End::TimedOut);
                break;
            }
            let edge = faces.edge[self.face].max(1e-9);
            let (spin, dg, phase) = self.cyclic(field, self.x);
            let omega = DVec3::new(0.0, 0.0, spin);
            if !self.airborne {
                self.substeps += 1;
                let n = faces.normal[self.face];
                let g = field.g_face[self.face] + dg;
                let g_t = g - n * g.dot(n);
                let speed = self.v.length();
                let normal =
                    -g.dot(n) + 2.0 * omega.cross(self.v).dot(n) - self.kappa * speed * speed;
                if normal <= 0.0 {
                    self.airborne = true;
                    self.hops += 1;
                    continue;
                }
                let v_stop = field.stop * (g.length() * s.h_s).sqrt();
                if speed <= v_stop && g_t.length() <= s.mu_s * normal {
                    self.end = Some(End::Arrested);
                    break;
                }
                let accel = g_t.length() + friction(m, speed, normal) * normal;
                let tau = substep(field.theta, edge, speed, accel, field.load.omega)
                    .min(field.time_limit - self.t);
                // Drive, the exact tangential Coriolis turn, then friction without reversal.
                let v_old = self.v;
                let mut v = self.v + g_t * tau;
                let angle = -2.0 * omega.dot(n) * tau;
                v = v * angle.cos() + n.cross(v) * angle.sin();
                let sp = v.length();
                let drop = friction(m, sp, normal) * normal * tau;
                v = if sp > drop { v * ((sp - drop) / sp) } else { DVec3::ZERO };
                // Back onto Γ at the closest point, and onto the new tangent plane at the same
                // speed. Trapezoidal drift: exact for a constant deceleration.
                let (next_face, next) = field.ground(self.x + (v_old + v) * (0.5 * tau));
                let step = next - self.x;
                let len = step.length();
                let n_next = faces.normal[next_face];
                // κ_v on the smooth surface: facet edges would otherwise read as kinks.
                let (s0, s1) = (
                    field.smooth_normal(faces, self.face, self.x),
                    field.smooth_normal(faces, next_face, next),
                );
                self.kappa = if len > 1e-12 { (s1 - s0).dot(step) / (len * len) } else { 0.0 };
                let sp = v.length();
                self.v = (v - n_next * v.dot(n_next))
                    .try_normalize()
                    .map_or(DVec3::ZERO, |u| u * sp);
                self.distance += len;
                self.x = next;
                self.foot = next;
                self.face = next_face;
                self.t += tau;
            } else {
                let x = self.x;
                let (below, ground) = field.ground(x);
                self.face = below;
                // Near Γ the expansion about the closest face, higher up the exterior field.
                let (near, height) = field.expansion(faces, below, ground, x);
                let g_self = if far && height > FAR * scale.r_bar {
                    match (self.nearest_anchor(x, REUSE * height), row.as_deref_mut()) {
                        (Some((x0, g0, h0)), _) => g0 + h0 * (x - x0),
                        (None, None) => return self.prefetch(field, faces, scale),
                        (None, Some(row)) => {
                            if row.last().is_none_or(|p| (x - *p).length() > PREFETCH * REUSE * height) {
                                row.push(x);
                                if row.len() >= PREFETCH_MAX {
                                    return Vec::new();
                                }
                            }
                            self.nearest_anchor(x, height).map_or(near, |(x0, g0, h0)| g0 + h0 * (x - x0))
                        }
                    }
                } else {
                    near
                };
                self.substeps += 1;
                let g = g_self + frame_terms(x, field.load) + dg;
                // Velocity Verlet: the last kick takes the mean of the gravity at both ends.
                if let Some((g0, tau0)) = self.kick.take() {
                    self.v += (g - g0) * (0.5 * tau0);
                }
                // Outward, under outward effective gravity, well above the body: beyond the
                // synchronous radius the outward pull grows with height, so it does not return.
                let r = x.length().max(1e-9);
                let up = x / r;
                if field.necks.is_none() && r > EARLY * r_max && self.v.dot(up) > 0.0 && g.dot(up) >= 0.0 {
                    self.end = Some(End::Escaped);
                    break;
                }
                // In flight the length scale is the height above Γ: the field varies on it,
                // and a step of θ_r h cannot pass the ground before the landing test.
                let ell = edge.max(height);
                let tau = substep(field.theta, ell, self.v.length(), g.length(), field.load.omega)
                    .min(field.time_limit - self.t);
                // Exact Coriolis turn about e_z, then gravity at the start point, then the
                // trapezoidal drift `v τ + g τ²/2`; the next substep completes the kick.
                let angle = -2.0 * spin * tau;
                let (c, sn) = (angle.cos(), angle.sin());
                let v_old = self.v;
                let v = self.v;
                self.v = DVec3::new(c * v.x - sn * v.y, sn * v.x + c * v.y, v.z) + g * tau;
                let drift = (v_old + self.v) * (0.5 * tau);
                self.t += tau;
                self.max_hop = self.max_hop.max(height);
                // Landing at the first crossing of the drift segment with Γ (G9).
                if let Some((t, landing)) = field.index.first_hit(x, drift, 1e-6, 1.0) {
                    let point = x + drift * t;
                    self.distance += (point - self.foot).length();
                    self.foot = point;
                    self.face = landing;
                    // Perfectly inelastic in the normal direction (e_n = 0).
                    let n = faces.normal[landing];
                    self.v -= n * self.v.dot(n);
                    self.x = point;
                    self.kappa = 0.0;
                    self.airborne = false;
                } else {
                    self.kick = Some((g, tau));
                    self.x += drift;
                    let beyond = match (field.necks, field.cycle) {
                        (Some(n), Some(c)) => c.necks_at(n, phase).beyond(self.x),
                        (Some(n), None) => n.beyond(self.x),
                        _ => false,
                    };
                    if self.x.length() > ESCAPE * r_max || beyond {
                        self.end = Some(End::Escaped);
                        break;
                    }
                }
            }
            let x = self.x;
            self.trail.push([x.x, x.y, x.z, f64::from(u8::from(self.airborne))]);
        }
        Vec::new()
    }

    /// The spin rate, the cyclic body force at `x` and the orbital phase at the packet's
    /// time (A-Λ); `(n, 0, 0)` without the cycle.
    fn cyclic(&self, field: &RunoutField, x: DVec3) -> (f64, DVec3, f64) {
        match field.cycle {
            Some(c) => {
                let m = self.phase + c.n * self.t;
                (c.spin(m), c.delta_g(x, m), m)
            }
            None => (field.load.omega, DVec3::ZERO, 0.0),
        }
    }

    /// The received exterior value nearest to `x`, if it lies within `reach`.
    fn nearest_anchor(&self, x: DVec3, reach: f64) -> Option<(DVec3, DVec3, DMat3)> {
        self.anchors
            .iter()
            .map(|a| ((a.0 - x).length_squared(), a))
            .filter(|(d2, _)| *d2 <= reach * reach)
            .min_by(|a, b| a.0.total_cmp(&b.0))
            .map(|(_, a)| *a)
    }

    /// The prefetch row of a far flight: the current position and the points where the
    /// rest of the packet's run, predicted by a copy of it with estimated exterior values
    /// (the nearest received one within `h`, else the face expansion), needs one, at least
    /// `PREFETCH · REUSE h` apart. The prediction only places the points; a miss asks again.
    fn prefetch(&self, field: &RunoutField, faces: &Faces, scale: Scale) -> Vec<DVec3> {
        let mut row = vec![self.x];
        self.clone().run_inner(field, faces, scale, true, Some(&mut row));
        row
    }

    fn finish(self, field: &RunoutField) -> Arrival {
        let escaped = matches!(self.end, Some(End::Escaped));
        let (x, distance) = if escaped { (self.start, 0.0) } else { (self.x, self.distance) };
        let (face, foot) = field.ground(x);
        let [a, b, c] = field.topo.faces[face].map(|v| v as usize);
        let w = barycentric(foot, field.positions[a], field.positions[b], field.positions[c])
            .map(|w| w.max(0.0));
        let sum: f64 = w.iter().sum();
        let (ids, weights) = ([a, b, c], w.map(|w| w / sum.max(1e-300)));
        let lat = |p: DVec3| (p.z / p.length().max(1e-300)).clamp(-1.0, 1.0).asin().abs();
        Arrival {
            source: self.source,
            ids,
            weights,
            distance,
            dlat: lat(x) - lat(self.start),
            hops: self.hops,
            max_hop: self.max_hop,
            escaped,
            lost: escaped && field.necks.is_some(),
            timed_out: matches!(self.end, Some(End::TimedOut)),
            path: self.trail.samples(),
        }
    }
}

#[derive(Clone, Copy)]
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
#[derive(Clone)]
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

/// Depth-averaged basal `μ(I)` of a Bagnold layer of thickness `h_s` (§10.0.2): basal shear
/// rate `5|v|/(2h_s)`, basal pressure `ρ_bulk N h_s`, plus `p_c` in the inertial number.
fn friction(m: &Material, speed: f64, normal: f64) -> f64 {
    let s = &m.spec;
    let shear = 2.5 * speed / s.h_s;
    let p_star = m.rho_bulk * normal.max(0.0) * s.h_s + m.p_c;
    dynamic_friction(m, shear, p_star).0
}

/// Move each source's eroded mass rate `J_ero A` to its arrival point (barycentric split)
/// and return the source density per vertex; the sum `Σ J A` is unchanged except for the
/// packets lost through a neck (A-O), whose mass leaves.
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
        if a.lost {
            continue;
        }
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

/// The relief of one cone solution, routed (§10.0.2, second part).
pub struct Routing {
    /// Radial deposit `δ` per vertex.
    pub delta: Vec<f64>,
    pub arrivals: Vec<Arrival>,
    /// Mass `m_i` of each routed packet, in the order of `arrivals`.
    pub masses: Vec<f64>,
}

/// Route the mass that the cone removed from the slope vertices: `removed[i] = r_i`
/// (radial, zero off the slope set), `jac[i] = (DM)_i`. The packets run out, their mass
/// is spread by the kernel of §10.0.2, and `δ_j = (K m)_j A_j / (DM)_j`, so that
/// `Σ (DM)_j δ_j = Σ m_i`.
pub fn route_relief(
    field: &RunoutField,
    area: &[f64],
    jac: &[f64],
    removed: &[f64],
    spread: &super::transport::Spread,
    exterior: Option<&Exterior<'_>>,
) -> Result<Routing, SimError> {
    let nv = area.len();
    let sources: Vec<usize> = (0..nv).filter(|&i| removed[i] > 0.0 && jac[i] > 0.0).collect();
    let arrivals = field.trace_with(&sources, exterior)?;
    let masses: Vec<f64> = arrivals.iter().map(|a| jac[a.source] * removed[a.source]).collect();
    // Mass per area at the sources, carried to the arrival points, then spread.
    let mut per_area = vec![0.0; nv];
    for (a, m) in arrivals.iter().zip(&masses) {
        per_area[a.source] = m / area[a.source];
    }
    let moved = moved_source(area, &per_area, &arrivals);
    let spread = spread.spread(&moved);
    let delta = (0..nv)
        .map(|j| if jac[j] > 0.0 { spread[j] * area[j] / jac[j] } else { 0.0 })
        .collect();
    Ok(Routing {
        delta,
        arrivals,
        masses,
    })
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
