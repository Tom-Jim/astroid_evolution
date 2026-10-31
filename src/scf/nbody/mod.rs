//! The fragments of a breakup about their primary (§12.3–12.5): rigid bodies under the
//! primary's gravity and their own, integrated by a second-order symplectic splitting;
//! fragments that touch merge, fragments whose certified limit load exceeds 1 break again,
//! and pieces below the debris size join a viscous ring.

pub mod gravity;
pub mod ring;

use std::collections::HashSet;

use serde::Serialize;

use crate::common::error::SimError;
use crate::common::math::{DMat3, DVec3};

use super::body::Body;
use super::breakup::{BreakupParams, EMPTY, Piece, Voxels, bounds, moved_kernel, piece, split};
use super::domain::volume_rule;
use super::forward::{Forward, Loading};
use super::orbit::certify;
use super::params::Project;
use super::surface::SurfaceIndex;

pub use gravity::{G, Primary};
pub use ring::Ring;

/// The volume rule of the near-field force integrals: `θ × φ × radial` nodes.
const NODES: (usize, usize, usize) = (12, 24, 4);

/// A rigid fragment: a body in its principal frame, and its state in the frame of the primary.
pub struct Fragment {
    pub id: usize,
    pub body: Body,
    pub fwd: Forward,
    index: SurfaceIndex,
    /// Volume nodes (body frame) and their masses, for the near-field forces.
    nodes: Vec<DVec3>,
    node_mass: Vec<f64>,
    pub mass: f64,
    /// Principal moments about the centre of mass.
    pub inertia: DVec3,
    /// Largest vertex distance from the centre of mass, and the volume-equivalent diameter.
    pub radius: f64,
    pub diameter: f64,
    /// Centre of mass, velocity, rotation from the body frame, and spin angular momentum.
    pub x: DVec3,
    pub v: DVec3,
    pub rot: DMat3,
    pub l: DVec3,
    /// Tidal strength `n²` up to which the fragment is certified to stand (`L* < 1`).
    stable_n2: f64,
}

impl Fragment {
    /// The fragment of `piece` placed at `x`, `rot`, moving with `v` and spinning at `omega`.
    pub fn new(id: usize, piece: Piece, x: DVec3, v: DVec3, rot: DMat3, omega: DVec3) -> Result<Self, SimError> {
        let Piece { body, fwd, .. } = piece;
        let (s, n) = (NODES.0, NODES.1);
        let (nodes, w, ..) = volume_rule(fwd.body_mesh(), s, n, NODES.2, 3)?;
        let density = body.density()?;
        let node_mass = nodes.iter().zip(&w).map(|(p, w)| density.rho(*p) * w).collect();
        let i = fwd.moments.inertia;
        let inertia = DVec3::new(i[0], i[1], i[2]);
        let mass = fwd.moments.mass;
        let diameter = 2.0 * (3.0 * fwd.moments.volume / (4.0 * std::f64::consts::PI)).cbrt();
        let mut f = Self {
            id,
            index: SurfaceIndex::build(&body.positions, &body.topo.faces),
            radius: body.r_star(),
            body,
            fwd,
            nodes,
            node_mass,
            mass,
            inertia,
            diameter,
            x,
            v,
            rot,
            l: DVec3::ZERO,
            stable_n2: 0.0,
        };
        f.l = f.inertia_world() * omega;
        Ok(f)
    }

    pub fn inertia_world(&self) -> DMat3 {
        self.rot * DMat3::from_diagonal(self.inertia) * self.rot.transpose()
    }

    pub fn omega(&self) -> DVec3 {
        self.rot * (self.rot.transpose() * self.l / self.inertia)
    }

    /// Whether any vertex of one lies inside the other.
    fn touches(&self, other: &Fragment) -> bool {
        let into = |a: &Fragment, b: &Fragment| {
            a.body.positions.iter().any(|p| b.index.inside(b.rot.transpose() * (a.x + a.rot * *p - b.x)))
        };
        (self.x - other.x).length() < self.radius + other.radius && (into(self, other) || into(other, self))
    }

    /// Orbital elements about the primary: semi-major axis (infinite when unbound),
    /// eccentricity and periapsis.
    pub fn elements(&self, gm: f64) -> (f64, f64, f64) {
        let r = self.x.length();
        let energy = 0.5 * self.v.length_squared() - gm / r;
        let h = self.x.cross(self.v).length();
        if energy >= 0.0 {
            return (f64::INFINITY, 1.0, h * h / (2.0 * gm));
        }
        let a = -gm / (2.0 * energy);
        let e = (1.0 + 2.0 * energy * h * h / (gm * gm)).max(0.0).sqrt();
        (a, e, a * (1.0 - e))
    }
}

/// A fragment as recorded.
#[derive(Clone, Debug, Serialize)]
pub struct FragmentState {
    pub id: usize,
    pub mass: f64,
    pub diameter: f64,
    pub x: [f64; 3],
    pub v: [f64; 3],
    pub a: f64,
    pub e: f64,
    pub spin: f64,
}

/// The system at one time: fragments, ledgers, and the conserved quantities.
#[derive(Clone, Debug, Serialize)]
pub struct Frame {
    pub t: f64,
    pub fragments: Vec<FragmentState>,
    pub ring: f64,
    pub to_primary: f64,
    pub to_moonlets: f64,
    pub escaped: f64,
    pub momentum: [f64; 3],
    pub angular_momentum: [f64; 3],
    pub energy: f64,
}

/// A breakup, merger, or a fragment leaving the dynamics.
#[derive(Clone, Debug, Serialize)]
pub struct Event {
    pub t: f64,
    pub kind: &'static str,
    pub ids: Vec<usize>,
    pub mass: f64,
    pub detail: String,
}

/// What a run produced.
pub struct Outcome {
    pub frames: Vec<Frame>,
    pub events: Vec<Event>,
    pub fragments: Vec<Fragment>,
    pub ring: Ring,
    pub escaped: f64,
}

/// The fragments' dynamics and the ring.
pub struct System<'a> {
    project: &'a Project,
    prm: &'a BreakupParams,
    pub primary: Primary,
    pub fragments: Vec<Fragment>,
    /// Pairs born touching, which merge only after they have once come apart.
    born: HashSet<(usize, usize)>,
    pub ring: Ring,
    pub escaped: f64,
    pub events: Vec<Event>,
    next_id: usize,
    pub t: f64,
}

impl<'a> System<'a> {
    pub fn new(project: &'a Project, prm: &'a BreakupParams, primary: Primary) -> Self {
        let ring = Ring::new(primary.gm, primary.radius, prm.ring_cells, prm.ring_particle_radius, project.material.spec.rho_p);
        Self { project, prm, primary, fragments: Vec::new(), born: HashSet::new(), ring, escaped: 0.0, events: Vec::new(), next_id: 0, t: 0.0 }
    }

    /// Adds the pieces of a cut of a body at `x`, `rot`, moving with `v` and spinning at
    /// `omega`: each keeps the parent's rigid motion at its centre.
    pub fn add_pieces(&mut self, pieces: Vec<Piece>, x: DVec3, v: DVec3, rot: DMat3, omega: DVec3) -> Result<Vec<usize>, SimError> {
        let mut ids = Vec::new();
        for p in pieces {
            let arm = rot * p.com;
            let (px, pv, prot) = (x + arm, v + omega.cross(arm), rot * p.rot);
            let id = self.next_id;
            self.next_id += 1;
            self.fragments.push(Fragment::new(id, p, px, pv, prot, omega)?);
            ids.push(id);
        }
        for &a in &ids {
            for &b in &ids {
                if a < b {
                    self.born.insert((a, b));
                }
            }
        }
        Ok(ids)
    }

    fn event(&mut self, kind: &'static str, ids: Vec<usize>, mass: f64, detail: String) {
        self.events.push(Event { t: self.t, kind, ids, mass, detail });
    }

    pub fn frame(&self) -> Result<Frame, SimError> {
        let (_, mutual) = gravity::forces(&self.primary, &self.fragments)?;
        let (mut p, mut l, mut e) = (DVec3::ZERO, DVec3::ZERO, mutual);
        for f in &self.fragments {
            p += f.v * f.mass;
            l += f.x.cross(f.v) * f.mass + f.l;
            e += 0.5 * f.mass * f.v.length_squared() + 0.5 * f.omega().dot(f.l);
        }
        Ok(Frame {
            t: self.t,
            fragments: self
                .fragments
                .iter()
                .map(|f| {
                    let (a, e, _) = f.elements(self.primary.gm);
                    FragmentState { id: f.id, mass: f.mass, diameter: f.diameter, x: f.x.to_array(), v: f.v.to_array(), a, e, spin: f.omega().length() }
                })
                .collect(),
            ring: self.ring.mass(),
            to_primary: self.ring.to_primary,
            to_moonlets: self.ring.to_moonlets,
            escaped: self.escaped,
            momentum: p.to_array(),
            angular_momentum: l.to_array(),
            energy: e,
        })
    }

    /// The substep: an orbit over `steps_per_orbit`, and a fiftieth of the mutual free-fall
    /// time of every close pair and of every spin period.
    fn step_size(&self) -> f64 {
        let mut dt = f64::INFINITY;
        for f in &self.fragments {
            let r = f.x.length();
            dt = dt.min(2.0 * std::f64::consts::PI * (r.powi(3) / self.primary.gm).sqrt() / self.prm.steps_per_orbit as f64);
            let w = f.omega().length();
            if w > 0.0 {
                dt = dt.min(0.02 * 2.0 * std::f64::consts::PI / w);
            }
        }
        for (j, a) in self.fragments.iter().enumerate() {
            for b in &self.fragments[j + 1..] {
                let d = (a.x - b.x).length();
                if d < gravity::NEAR * (a.radius + b.radius) {
                    dt = dt.min(0.02 * (d.powi(3) / (G * (a.mass + b.mass))).sqrt());
                }
            }
        }
        dt
    }

    /// One kick–drift–kick step of `dt`: translation by velocity Verlet, rotation by the
    /// splitting of the free rigid body into its principal-axis flows (§12.3).
    fn step(&mut self, dt: f64) -> Result<(), SimError> {
        let (f0, _) = gravity::forces(&self.primary, &self.fragments)?;
        for (f, (force, torque)) in self.fragments.iter_mut().zip(&f0) {
            f.v += *force * (0.5 * dt / f.mass);
            f.l += *torque * (0.5 * dt);
            f.x += f.v * dt;
            free_rotation(&mut f.rot, f.l, f.inertia, dt);
        }
        let (f1, _) = gravity::forces(&self.primary, &self.fragments)?;
        for (f, (force, torque)) in self.fragments.iter_mut().zip(&f1) {
            f.v += *force * (0.5 * dt / f.mass);
            f.l += *torque * (0.5 * dt);
        }
        self.t += dt;
        Ok(())
    }

    /// One substep of `dt` and the mergers it brings.
    pub fn advance(&mut self, dt: f64) -> Result<(), SimError> {
        {
            let _span = super::timing::span("nbody_step");
            self.step(dt)?;
        }
        self.contacts()
    }

    /// Mergers of touching pairs that approach, and the release of pairs born touching.
    fn contacts(&mut self) -> Result<(), SimError> {
        let mut j = 0;
        while j < self.fragments.len() {
            let mut k = j + 1;
            let mut merged = false;
            while k < self.fragments.len() {
                let (a, b) = (&self.fragments[j], &self.fragments[k]);
                let key = (a.id.min(b.id), a.id.max(b.id));
                let touching = a.touches(b);
                if !touching {
                    self.born.remove(&key);
                } else if !self.born.contains(&key) && (b.v - a.v).dot(b.x - a.x) < 0.0 {
                    let b = self.fragments.remove(k);
                    let a = self.fragments.remove(j);
                    let c = self.merge(a, b)?;
                    self.fragments.insert(j, c);
                    merged = true;
                    break;
                }
                k += 1;
            }
            if !merged {
                j += 1;
            }
        }
        Ok(())
    }

    /// The perfectly inelastic merger of `a` and `b`: the union of their solids, with mass,
    /// momentum and angular momentum conserved (§12.4).
    fn merge(&mut self, a: Fragment, b: Fragment) -> Result<Fragment, SimError> {
        let mass = a.mass + b.mass;
        let x = (a.x * a.mass + b.x * b.mass) / mass;
        let v = (a.v * a.mass + b.v * b.mass) / mass;
        let l = a.l + b.l + (a.x - x).cross(a.v - v) * a.mass + (b.x - x).cross(b.v - v) * b.mass;
        let world = |f: &Fragment| f.body.positions.iter().map(|p| f.x - x + f.rot * *p).collect::<Vec<_>>();
        let pts: Vec<DVec3> = world(&a).into_iter().chain(world(&b)).collect();
        let (lo, hi) = bounds(&pts);
        let inside = |f: &Fragment, p: DVec3| f.index.inside(f.rot.transpose() * (p + x - f.x));
        let mut vox = Voxels::new(lo, hi, self.prm.voxels, |p| if inside(&a, p) || inside(&b, p) { 0 } else { EMPTY });
        vox.connect();
        let kernels = a
            .body
            .kernels
            .iter()
            .map(|kn| moved_kernel(kn, a.rot, a.x - x))
            .chain(b.body.kernels.iter().map(|kn| moved_kernel(kn, b.rot, b.x - x)))
            .collect::<Result<Vec<_>, _>>()?;
        let bulk = (a.body.bulk * a.mass + b.body.bulk * b.mass) / mass;
        let big = if a.mass >= b.mass { &a } else { &b };
        let area: f64 = big.body.face_geometry().iter().map(|f| f.2).sum();
        let level = big.body.topo.icosphere.unwrap_or(self.project.body.level);
        // The larger part's edge length, allowing one level more than its own (the union's
        // area is up to four times as large at the next level).
        let inner = vox.deepest(|lab| lab == 0).ok_or_else(|| SimError::geometry("merger: an empty union"))?;
        let p = piece(self.project, &vox.surface(|lab| lab == 0), inner, kernels, bulk, mass, level + 1, area * 4.0)?;
        let (com, rot) = (p.com, p.rot);
        let id = self.next_id;
        self.next_id += 1;
        let mut c = Fragment::new(id, p, x + com, v, rot, DVec3::ZERO)?;
        // The geometric centre may differ from the parts' by a voxel: momentum and angular
        // momentum are those of the parts about their centre of mass.
        c.x = x;
        c.l = l;
        let ke = |f: &Fragment| 0.5 * f.mass * f.v.length_squared() + 0.5 * f.omega().dot(f.l);
        let lost = ke(&a) + ke(&b) - ke(&c);
        self.event("merge", vec![a.id, b.id, id], mass, format!("dissipated {lost:.4e} J"));
        Ok(c)
    }

    /// Fragments that hit the primary, leave its gravity, or are below the debris size leave
    /// the dynamics; the rest are checked against their limit load when the tide has grown
    /// since their last certificate, and break along their mechanism when it exceeds 1.
    fn cascade(&mut self) -> Result<(), SimError> {
        let gm = self.primary.gm;
        let mut k = 0;
        while k < self.fragments.len() {
            let f = &self.fragments[k];
            let (a, _, peri) = f.elements(gm);
            if f.x.length() < self.primary.radius + f.radius || peri < self.primary.radius {
                let f = self.fragments.remove(k);
                self.ring.to_primary += f.mass;
                self.event("impact", vec![f.id], f.mass, String::new());
                continue;
            }
            if !a.is_finite() {
                let f = self.fragments.remove(k);
                self.escaped += f.mass;
                self.event("escape", vec![f.id], f.mass, String::new());
                continue;
            }
            if f.diameter < self.prm.debris_diameter {
                let f = self.fragments.remove(k);
                self.ring.deposit(a, f.mass);
                self.event("debris", vec![f.id], f.mass, format!("a = {a:.4e} m"));
                continue;
            }
            let n2 = gm / f.x.length().powi(3);
            if n2 <= f.stable_n2 * 1.05 {
                k += 1;
                continue;
            }
            // Synchronous and aligned with the tide (long axis toward the primary): the
            // largest tidal load the fragment's orbit can put on it.
            let mut fwd = Forward::bind(&f.body, self.project.g)?;
            let (bracket, ops) = certify(self.project, &f.body, &mut fwd, Loading { omega: n2.sqrt(), tide: n2 })?;
            if bracket.lo > 1.0 {
                let f = self.fragments.remove(k);
                let range = format!("L* in [{:.4}, {:.4}]", bracket.lo, bracket.hi);
                match split(self.project, self.prm, &f.body, &fwd, &ops, &bracket.u)? {
                    Some(cut) => {
                        let omega = f.omega();
                        let ids = self.add_pieces(cut.pieces, f.x, f.v, f.rot, omega)?;
                        let detail = format!("{range}, misfit {:?}, mass error {:.2e}", cut.misfit, cut.mass_error);
                        self.event("split", std::iter::once(f.id).chain(ids).collect(), f.mass, detail);
                    }
                    // A collapse that is no set of rigid blocks is a granular flow: the
                    // fragment disperses into the ring.
                    None => {
                        self.ring.deposit(a, f.mass);
                        self.event("disperse", vec![f.id], f.mass, format!("{range}, no rigid blocks"));
                    }
                }
                continue;
            }
            // Standing, or undecided at the bracket's width: checked again once the tide grows.
            self.fragments[k].stable_n2 = n2;
            k += 1;
        }
        Ok(())
    }

    /// The dynamic phase for `prm.dynamic_time`, then the ring for `prm.ring_time`;
    /// `observer` sees every recorded frame.
    pub fn run(mut self, observer: &mut dyn FnMut(&Frame, &[Event]) -> Result<(), SimError>) -> Result<Outcome, SimError> {
        let mut frames = Vec::new();
        let (mut next_out, mut next_check) = (0.0, 0.0);
        let end = self.t + self.prm.dynamic_time;
        while self.t < end && !self.fragments.is_empty() {
            if self.t >= next_check {
                let _span = super::timing::span("nbody_cascade");
                self.cascade()?;
                let r = self.fragments.iter().map(|f| f.x.length()).fold(f64::INFINITY, f64::min);
                next_check = self.t + self.prm.check_every * 2.0 * std::f64::consts::PI * (r.powi(3) / self.primary.gm).sqrt();
            }
            if self.t >= next_out {
                let frame = self.frame()?;
                observer(&frame, &self.events)?;
                frames.push(frame);
                next_out = self.t + self.prm.output_every;
            }
            if self.fragments.is_empty() {
                break;
            }
            let dt = self.step_size().min(end - self.t);
            self.advance(dt)?;
        }
        // Debris still in orbit at the end of the dynamic phase stays as fragments; the ring
        // spreads on its own time scale.
        {
            let _span = super::timing::span("ring");
            self.ring.evolve(self.prm.ring_time)?;
        }
        self.t = end + self.prm.ring_time;
        let frame = self.frame()?;
        observer(&frame, &self.events)?;
        frames.push(frame);
        Ok(Outcome { frames, events: self.events, fragments: self.fragments, ring: self.ring, escaped: self.escaped })
    }
}

/// The free rigid body over `dt` at angular momentum `l` (fixed in space): the symmetric
/// splitting `x(h/2) y(h/2) z(h) y(h/2) x(h/2)` of the principal-axis flows, each an exact
/// rotation (Dullweber, Leimkuhler & McLachlan 1997).
pub fn free_rotation(rot: &mut DMat3, l: DVec3, inertia: DVec3, dt: f64) {
    let mut pi = rot.transpose() * l;
    for (axis, frac) in [(0, 0.5), (1, 0.5), (2, 1.0), (1, 0.5), (0, 0.5)] {
        let phi = pi[axis] / inertia[axis] * dt * frac;
        let r = match axis {
            0 => DMat3::from_rotation_x(phi),
            1 => DMat3::from_rotation_y(phi),
            _ => DMat3::from_rotation_z(phi),
        };
        *rot = *rot * r;
        pi = r.transpose() * pi;
    }
}
