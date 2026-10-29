//! The marker-borne material state of A-ϖ (§11.5) and A-Ξ (§11.3) as the solver sees
//! it, and its evolution between steps: transport, exact state update, maintenance.

use std::sync::atomic::{AtomicUsize, Ordering};

use serde::{Deserialize, Serialize};

use crate::common::error::SimError;
use crate::common::math::DVec3;
use crate::rheo::Material;
use crate::scf::stokes::{Local, MaterialField};
use crate::scf::surface::SurfaceIndex;

use super::dilatancy::DilatancyParams;
use super::markers::{Markers, Range, Reconstruction};
use super::thermal::exact_damage;

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DamageParams {
    /// `γ_ϖ`: plastic strain scale of bond breakage.
    pub strain: f64,
    /// `h_ϖ` (s⁻¹), healing rate.
    pub healing: f64,
    /// `ϖ⁽⁰⁾` and `ϖ_dep` (a deposit carries no bonds: 1).
    pub initial: f64,
    pub deposit: f64,
    /// `θ_ϖ ≤ 1` of the step bound `Δt_ϖ`.
    pub theta_step: f64,
}

impl DamageParams {
    pub fn validate(&self) -> Result<(), SimError> {
        let unit = |v: f64| v.is_finite() && (0.0..=1.0).contains(&v);
        if !(self.strain.is_finite() && self.strain > 0.0)
            || !(self.healing.is_finite() && self.healing >= 0.0)
            || !unit(self.initial)
            || !unit(self.deposit)
            || !(self.theta_step > 0.0 && self.theta_step <= 1.0)
        {
            return Err(SimError::model_not_closed(
                "damage needs strain > 0, healing >= 0, initial and deposit in [0, 1], theta_step in (0, 1]",
            ));
        }
        Ok(())
    }
}

/// Per-step diagnostics of §11.10 for the markers, A-ϖ and A-Ξ.
#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct MaterialReport {
    pub active: bool,
    pub markers: usize,
    pub inserted: usize,
    pub deleted: usize,
    pub merged: usize,
    pub fallbacks: usize,
    /// `ℓ_p / ℓ_h`.
    pub ell_ratio: f64,
    pub max_damage: f64,
    pub mean_damage: f64,
    pub nu_min: f64,
    pub nu_max: f64,
    pub max_xi: f64,
    /// A-Ξ: projection defect `e_ε` of `χ_h` and the step bound `Δt_Ξ`.
    pub dilatation_defect: f64,
    pub dt_xi: f64,
    /// A-ϖ: the step bound `Δt_ϖ`.
    pub dt_damage: f64,
}

impl MaterialReport {
    pub fn inactive(&self) -> bool {
        !self.active
    }
}

/// The local strengths that the markers give the solver (`MaterialField`).
pub struct StateField<'a> {
    rec: Reconstruction<'a>,
    markers: &'a Markers,
    damage: bool,
    /// `K_Ξ`, `ν_c⁰` and the admissible range of `ν` (A-Ξ).
    dilatancy: Option<(f64, f64, Range)>,
    pub fallbacks: AtomicUsize,
}

impl<'a> StateField<'a> {
    pub fn new(
        markers: &'a Markers,
        index: &'a SurfaceIndex,
        h: f64,
        damage: bool,
        dilatancy: Option<&DilatancyParams>,
    ) -> Self {
        Self {
            rec: Reconstruction::new(markers, index, h),
            markers,
            damage,
            dilatancy: dilatancy.map(|d| (d.k, d.nu_c0, d.range())),
            fallbacks: AtomicUsize::new(0),
        }
    }

    pub fn reconstruction(&self) -> &Reconstruction<'a> {
        &self.rec
    }

    #[allow(clippy::type_complexity)]
    fn fits(&self, pts: &[DVec3]) -> (Vec<(f64, DVec3)>, Vec<(f64, DVec3)>) {
        let mut fields: Vec<(&[f64], Range)> = Vec::new();
        if self.damage {
            fields.push((&self.markers.damage, Range { lo: 0.0, hi: 1.0 }));
        }
        if let Some((_, _, range)) = self.dilatancy {
            fields.push((&self.markers.nu, range));
        }
        let (fit, fallbacks) = self.rec.fit(pts, &fields);
        self.fallbacks.fetch_add(fallbacks, Ordering::Relaxed);
        let mut it = fit.into_iter();
        let damage = if self.damage {
            it.next().expect("damage fit").into_iter().map(|f| (f.value, f.grad)).collect()
        } else {
            vec![(0.0, DVec3::ZERO); pts.len()]
        };
        let nu = match self.dilatancy {
            Some(_) => it.next().expect("ν fit").into_iter().map(|f| (f.value, f.grad)).collect(),
            None => vec![(self.dilatancy.map_or(0.0, |d| d.1), DVec3::ZERO); pts.len()],
        };
        (damage, nu)
    }
}

impl MaterialField for StateField<'_> {
    /// `c = (1 − ϖ) c_intact`, `∇c = (1 − ϖ) ∇c_intact − c_intact ∇ϖ` (Proposition 11.9),
    /// `ξ = K_Ξ (ν − ν_c⁰)`, `∇ξ = K_Ξ ∇ν` (A-Ξ).
    fn locals(&self, _m: &Material, pts: &[DVec3], intact: &[Local]) -> Vec<Local> {
        let (damage, nu) = self.fits(pts);
        intact
            .iter()
            .zip(damage)
            .zip(nu)
            .map(|((l, (w, gw)), (n, gn))| {
                let (k, nu_c0) = self.dilatancy.map_or((0.0, 0.0), |d| (d.0, d.1));
                Local {
                    c: (1.0 - w) * l.c,
                    gc: l.gc * (1.0 - w) - gw * l.c,
                    xi: if self.dilatancy.is_some() { k * (n - nu_c0) } else { 0.0 },
                    gxi: gn * k,
                }
            })
            .collect()
    }
}

/// `Δt_ϖ` (§11.5): `θ_ϖ` over the largest `|(1 − Rϖ) γ̇_p^⋆ / γ_ϖ − h_ϖ Rϖ|` on the volume
/// nodes `pts` (driving rate `rate` there), `∞` when it vanishes. Without healing only the
/// driven nodes contribute.
pub fn dt_bound(d: &DamageParams, markers: &Markers, rec: &Reconstruction, rate: &[f64], pts: &[DVec3]) -> f64 {
    let at: Vec<usize> = (0..pts.len()).filter(|q| rate[*q] > 0.0 || d.healing > 0.0).collect();
    if at.is_empty() {
        return f64::INFINITY;
    }
    let x: Vec<DVec3> = at.iter().map(|q| pts[*q]).collect();
    let (fit, _) = rec.fit(&x, &[(&markers.damage, Range { lo: 0.0, hi: 1.0 })]);
    let worst = fit[0]
        .iter()
        .zip(&at)
        .map(|(f, q)| ((1.0 - f.value) * rate[*q] / d.strain - d.healing * f.value).abs())
        .fold(0.0, f64::max);
    if worst > 0.0 { d.theta_step / worst } else { f64::INFINITY }
}

/// One step of the marker state (Definition 11.1, items 2–3): RK2 transport along
/// `v_adv` and the exact state update at the midpoints with the frozen rates.
/// `rate(pts)` gives `γ̇_p^⋆` and `ν_c(I_c)` at points.
pub fn advance(
    markers: &mut Markers,
    tau: f64,
    v_adv: &dyn Fn(&[DVec3]) -> Vec<DVec3>,
    rate: &dyn Fn(&[DVec3]) -> (Vec<f64>, Vec<f64>),
    damage: Option<&DamageParams>,
    dilatancy: Option<&DilatancyParams>,
) {
    if markers.is_empty() || tau <= 0.0 {
        return;
    }
    let y = markers.points();
    let v0 = v_adv(&y);
    let mid: Vec<DVec3> = y.iter().zip(&v0).map(|(p, v)| *p + *v * (0.5 * tau)).collect();
    let v1 = v_adv(&mid);
    let (gp, nu_c) = rate(&mid);
    for j in 0..y.len() {
        markers.pos[j] = (y[j] + v1[j] * tau).to_array();
        if let Some(d) = damage {
            markers.damage[j] = exact_damage(markers.damage[j], gp[j] / d.strain, d.healing, tau);
        }
        if let Some(d) = dilatancy {
            markers.nu[j] = d.exact_nu(markers.nu[j], nu_c[j], tau * gp[j]);
        }
    }
}

/// Maintenance on the settled geometry (Definition 11.1, item 5): delete the markers
/// outside the body, insert where a volume node is under-covered, merge clusters.
/// `deposit_at(x)` decides whether a new point carries the deposit state.
#[allow(clippy::too_many_arguments)]
pub fn maintain(
    markers: &mut Markers,
    index: &SurfaceIndex,
    nodes: &[DVec3],
    h: f64,
    n_min: usize,
    n_max: usize,
    deposit_state: (f64, f64),
    deposit_at: &dyn Fn(DVec3) -> bool,
    report: &mut MaterialReport,
) {
    // Deletion.
    let keep: Vec<bool> = markers.points().iter().map(|p| index.inside(*p)).collect();
    report.deleted = keep.iter().filter(|k| !**k).count();
    if report.deleted > 0 {
        let mut k = keep.iter();
        markers.pos.retain(|_| *k.next().expect("flag"));
        let mut k = keep.iter();
        markers.nu.retain(|_| *k.next().expect("flag"));
        let mut k = keep.iter();
        markers.damage.retain(|_| *k.next().expect("flag"));
    }
    // Coverage at the volume nodes.
    let added: Vec<(DVec3, f64, f64)> = {
        let rec = Reconstruction::new(markers, index, h);
        nodes
            .iter()
            .filter(|x| rec.coverage(**x) < n_min)
            .map(|x| {
                if deposit_at(*x) {
                    (*x, deposit_state.0, deposit_state.1)
                } else {
                    let nu = rec.shepard(*x, &markers.nu);
                    let dm = rec.shepard(*x, &markers.damage);
                    match (nu, dm) {
                        (Some(n), Some(d)) => (*x, n, d),
                        _ => (*x, deposit_state.0, deposit_state.1),
                    }
                }
            })
            .collect()
    };
    report.inserted = added.len();
    for (x, nu, d) in added {
        markers.pos.push(x.to_array());
        markers.nu.push(nu);
        markers.damage.push(d);
    }
    // Merging: more than n_max markers within h/4 become their mean.
    let pts = markers.points();
    let grid = crate::scf::surface::PointGrid::build(&pts);
    let mut gone = vec![false; pts.len()];
    let mut merged = Markers::default();
    for j in 0..pts.len() {
        if gone[j] {
            continue;
        }
        let near: Vec<usize> = grid
            .within(pts[j], 0.25 * h)
            .into_iter()
            .map(|(_, i)| i)
            .filter(|i| !gone[*i])
            .collect();
        if near.len() > n_max {
            let n = near.len() as f64;
            let mean = |f: &dyn Fn(usize) -> f64| near.iter().map(|i| f(*i)).sum::<f64>() / n;
            let c = near.iter().fold(DVec3::ZERO, |s, i| s + pts[*i]) / n;
            merged.pos.push(c.to_array());
            merged.nu.push(mean(&|i| markers.nu[i]));
            merged.damage.push(mean(&|i| markers.damage[i]));
            near.iter().for_each(|i| gone[*i] = true);
            report.merged += near.len();
        }
    }
    if report.merged > 0 {
        let mut out = Markers::default();
        for j in 0..pts.len() {
            if !gone[j] {
                out.pos.push(markers.pos[j]);
                out.nu.push(markers.nu[j]);
                out.damage.push(markers.damage[j]);
            }
        }
        out.pos.extend(merged.pos);
        out.nu.extend(merged.nu);
        out.damage.extend(merged.damage);
        *markers = out;
    }
    report.markers = markers.len();
}
