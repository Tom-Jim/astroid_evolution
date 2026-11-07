//! Occupied radial intervals of one ray (MS §3.6).
//!
//! The crossings of a closed surface alternate between entering and leaving the
//! solid, and the ray ends outside it. Counting the crossings therefore fixes
//! the occupancy at the start of the ray, for an interior point, an exterior
//! point, and a boundary point alike. An odd count means the ray starts inside:
//! the intervals are `[0, R1], [R2, R3], …`. An even count gives
//! `[R1, R2], [R3, R4], …`.

/// Relative and absolute (in units of the body radius) distance below which
/// two crossings are one geometric point: an edge or vertex reported by both
/// triangles that share it.
pub(crate) const MERGE_REL: f64 = 1.0e-6;
pub(crate) const MERGE_ABS: f64 = 1.0e-6;

/// Crossings nearer than this fraction of the body radius belong to the launch
/// point itself, not to the body.
pub(crate) const ORIGIN_ABS: f64 = 1.0e-6;

/// Crossings of one ray with coincident reports merged and launch-point hits dropped.
fn merged(hits: &[f64], scale: f64) -> impl Iterator<Item = f64> + '_ {
    let mut last = f64::NEG_INFINITY;
    hits.iter().copied().filter(move |&t| {
        if !(t.is_finite() && t > ORIGIN_ABS * scale)
            || t - last <= MERGE_ABS * scale + MERGE_REL * t
        {
            return false;
        }
        last = t;
        true
    })
}

/// Parity of the crossings of one ray: true when the ray starts inside the solid.
pub fn starts_inside(hits: &[f64], scale: f64) -> bool {
    merged(hits, scale).count() % 2 == 1
}

/// Radial intervals `(R⁻, R⁺)` for the sorted crossing distances `hits` of one
/// ray. `scale` is the body radius. The intervals are written to `out`; a first
/// interval that begins at `0.0` marks a ray that starts inside the solid.
pub fn intervals(hits: &[f64], scale: f64, out: &mut Vec<(f64, f64)>) {
    out.clear();
    let mut open = if starts_inside(hits, scale) {
        Some(0.0)
    } else {
        None
    };
    for t in merged(hits, scale) {
        match open.take() {
            Some(start) => out.push((start, t)),
            None => open = Some(t),
        }
    }
}

/// Crossings of one ray with their orientation signs `(t, s = sign(dir·n̂))`:
/// reports closer than the geometric merge tolerance are one vertex/edge
/// report group whose net sign decides whether occupancy changes (MS §3.6).
/// A group whose signs cancel — a tangency or an exact vertex hit — is dropped.
fn merged_signed(hits: &[(f64, f32)], scale: f64) -> Vec<(f64, f32)> {
    let mut out: Vec<(f64, f32)> = Vec::with_capacity(hits.len());
    let mut last = f64::NEG_INFINITY;
    let mut acc = 0.0f64;
    for &(t, s) in hits {
        if !(t.is_finite() && t > ORIGIN_ABS * scale) {
            continue;
        }
        if !out.is_empty() && t - last <= MERGE_ABS * scale + MERGE_REL * t {
            acc += f64::from(s);
            continue;
        }
        if let Some(prev) = out.last_mut() {
            if acc.abs() < 0.5 {
                out.pop();
            } else {
                prev.1 = acc.signum() as f32;
            }
        }
        out.push((t, 0.0));
        acc = f64::from(s);
        last = t;
    }
    if let Some(prev) = out.last_mut() {
        if acc.abs() < 0.5 {
            out.pop();
        } else {
            prev.1 = acc.signum() as f32;
        }
    }
    out
}

/// Occupancy at the start of the ray from the orientation rule: the ray starts
/// inside iff its first crossing leaves the solid (`s > 0`). Falls back to the
/// parity when the crossing list carries no sign information.
pub fn starts_inside_oriented(hits: &[(f64, f32)], scale: f64) -> bool {
    match merged_signed(hits, scale).first() {
        Some(&(_, s)) if s > 0.5 => true,
        Some(&(_, s)) if s < -0.5 => false,
        _ => {
            let t: Vec<f64> = hits.iter().map(|h| h.0).collect();
            starts_inside(&t, scale)
        }
    }
}

/// The orientation-rule construction of MS §3.6: a crossing with `s < 0` opens
/// an occupied interval, `s > 0` closes it. Returns the number of anomalies —
/// same-sign crossings in a row, or an interval left open — which mark the ray
/// dirty; the caller then repeats it with `intervals` and counts it.
pub fn intervals_oriented(
    hits: &[(f64, f32)],
    scale: f64,
    out: &mut Vec<(f64, f64)>,
) -> u32 {
    out.clear();
    let merged = merged_signed(hits, scale);
    let mut inside = merged.first().is_some_and(|&(_, s)| s > 0.5);
    let mut open = inside.then_some(0.0);
    let mut anomalies = 0u32;
    for &(t, s) in &merged {
        if s > 0.5 {
            if inside {
                out.push((open.unwrap_or(0.0), t));
                inside = false;
                open = None;
            } else {
                anomalies += 1;
            }
        } else if s < -0.5 {
            if inside {
                anomalies += 1;
            } else {
                open = Some(t);
                inside = true;
            }
        }
    }
    if inside && !merged.is_empty() {
        // An interval left open means a lost exit crossing: drop it and mark
        // the ray dirty.
        anomalies += 1;
    }
    anomalies
}
