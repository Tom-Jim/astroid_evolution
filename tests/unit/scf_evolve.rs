//! Unit tests of `scf::evolve`.

#[test]
fn slope_promotion_is_local_to_the_supercritical_faces() {
    use astroid_evolution::scf::settle::slope_promotion;
    // Two faces sharing the edge 1–2; face 0 is supercritical, face 1 is not.
    let faces = [[0u32, 1, 2], [1, 2, 3]];
    let slack = [0.05, 0.0];
    let tol = 1e-3;
    // Nothing yields: every loose vertex of face 0 is promoted, vertex 3 is not.
    let (movable, promoted) =
        slope_promotion(&faces, &[false; 4], &slack, &[1.0, 1.0, 1.0, 1.0], tol);
    assert_eq!(movable, vec![true, true, true, false]);
    assert_eq!(promoted, vec![true, true, true, false]);
    // A vertex without loose material stays locked.
    let (movable, _) = slope_promotion(&faces, &[false; 4], &slack, &[1.0, 0.0, 1.0, 1.0], tol);
    assert_eq!(movable, vec![true, false, true, false]);
    // A yielded vertex elsewhere (3) does not switch the promotion off, and a yielded
    // vertex of the supercritical face (0) is movable without being promoted.
    let (movable, promoted) =
        slope_promotion(&faces, &[true, false, false, true], &slack, &[1.0; 4], tol);
    assert_eq!(movable, vec![true, true, true, true]);
    assert_eq!(promoted, vec![false, true, true, false]);
    // Below the tolerance nothing is promoted.
    let (movable, promoted) = slope_promotion(&faces, &[false; 4], &[1e-4, 0.0], &[1.0; 4], tol);
    assert!(movable.iter().all(|m| !*m) && promoted.iter().all(|p| !*p));
}
