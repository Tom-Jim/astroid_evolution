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
