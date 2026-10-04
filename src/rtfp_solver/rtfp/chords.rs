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

#[cfg(test)]
mod tests {
    use super::*;

    fn run(hits: &[f64]) -> Vec<(f64, f64)> {
        let mut out = Vec::new();
        intervals(hits, 1.0, &mut out);
        out
    }

    #[test]
    fn odd_count_starts_inside() {
        assert_eq!(run(&[2.0]), vec![(0.0, 2.0)]);
        assert_eq!(run(&[1.0, 2.0, 3.0]), vec![(0.0, 1.0), (2.0, 3.0)]);
        assert!(starts_inside(&[2.0], 1.0));
    }

    #[test]
    fn even_count_starts_outside() {
        assert!(run(&[]).is_empty());
        assert_eq!(run(&[1.0, 3.0]), vec![(1.0, 3.0)]);
        assert_eq!(run(&[1.0, 2.0, 3.0, 4.0]), vec![(1.0, 2.0), (3.0, 4.0)]);
        assert!(!starts_inside(&[1.0, 3.0], 1.0));
    }

    #[test]
    fn an_edge_crossing_reported_twice_is_one_crossing() {
        assert_eq!(run(&[1.0, 1.0 + 1.0e-9, 3.0]), vec![(1.0, 3.0)]);
    }

    #[test]
    fn a_hit_at_the_launch_point_is_dropped() {
        assert_eq!(run(&[1.0e-9, 2.0]), vec![(0.0, 2.0)]);
        assert_eq!(run(&[1.0e-9, 1.0, 2.0]), vec![(1.0, 2.0)]);
    }
}
