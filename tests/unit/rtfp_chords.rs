//! Unit tests of `rtfp::chords`, moved out of `src/rtfp/chords.rs`.

use astroid_evolution::rtfp::chords::*;

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

fn run_o(hits: &[(f64, f32)]) -> (Vec<(f64, f64)>, u32) {
    let mut out = Vec::new();
    let anomalies = intervals_oriented(hits, 1.0, &mut out);
    (out, anomalies)
}

#[test]
fn oriented_alternating_signs_give_the_parity_intervals() {
    // Outside start: enter, leave, enter, leave.
    assert_eq!(
        run_o(&[(1.0, -1.0), (3.0, 1.0), (4.0, -1.0), (6.0, 1.0)]),
        (vec![(1.0, 3.0), (4.0, 6.0)], 0)
    );
    // Inside start: the first crossing leaves the solid.
    assert_eq!(
        run_o(&[(2.0, 1.0), (3.0, -1.0), (5.0, 1.0)]),
        (vec![(0.0, 2.0), (3.0, 5.0)], 0)
    );
    assert!(starts_inside_oriented(&[(2.0, 1.0)], 1.0));
    assert!(!starts_inside_oriented(&[(1.0, -1.0)], 1.0));
}

#[test]
fn oriented_tangency_reports_cancel_and_drop() {
    // A tangency reports an entering and a leaving crossing at the same point;
    // their signs cancel and no occupancy change survives.
    assert_eq!(
        run_o(&[(1.0, -1.0), (1.0 + 1.0e-9, 1.0), (3.0, -1.0), (5.0, 1.0)]),
        (vec![(3.0, 5.0)], 0)
    );
}

#[test]
fn oriented_anomalies_mark_the_ray_dirty() {
    // Two entering crossings in a row cannot occur on a watertight surface.
    let (_, anomalies) = run_o(&[(1.0, -1.0), (2.0, -1.0), (3.0, 1.0)]);
    assert!(anomalies > 0);
    // An interval left open is a lost exit crossing.
    let (_, anomalies) = run_o(&[(1.0, -1.0), (2.0, 1.0), (3.0, -1.0)]);
    assert!(anomalies > 0);
}

#[test]
fn signless_crossings_fall_back_to_parity() {
    assert!(starts_inside_oriented(&[(1.0, 0.0), (3.0, 0.0), (5.0, 0.0)], 1.0));
    assert!(!starts_inside_oriented(&[(1.0, 0.0), (3.0, 0.0)], 1.0));
}
