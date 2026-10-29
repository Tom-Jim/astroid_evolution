//! Unit tests of `scf::classify`.

use astroid_evolution::scf::body::Topology;
use astroid_evolution::scf::classify::midpoint_parents;

/// The probes that inherit (G7) are those of the midpoint vertices: past the vertices of
/// the previous level, each vertex of a subdivided icosphere is the midpoint of the edge
/// between its two parents, and the coarse vertices come first and unchanged.
#[test]
fn midpoint_parents_are_the_ends_of_the_split_edge() {
    for level in 1..=5 {
        let (topo, dirs) = Topology::icosphere(level);
        let (coarse, coarse_dirs) = Topology::icosphere(level - 1);
        let (n_c, parents) = midpoint_parents(&topo).expect("a subdivided icosphere");
        assert_eq!(n_c, coarse.n_vertices(), "level {level}");
        assert_eq!(&dirs[..n_c], &coarse_dirs[..], "level {level}: coarse vertices moved");
        assert_eq!(parents.len(), topo.n_vertices() - n_c);
        for (k, &[a, b]) in parents.iter().enumerate() {
            assert!(coarse.vertex_neighbors[a].contains(&b), "level {level}: {a}–{b} is no coarse edge");
            let mid = (dirs[a] + dirs[b]).normalize();
            assert!((dirs[n_c + k] - mid).length() < 1e-12, "level {level}: vertex {} is not the midpoint", n_c + k);
        }
    }
    // The base icosahedron and a mesh not known to be subdivided have no parents.
    assert!(midpoint_parents(&Topology::icosphere(0).0).is_none());
    let (topo, dirs) = Topology::icosphere(3);
    assert!(midpoint_parents(&Topology::new(dirs.len(), topo.faces.clone())).is_none());
}
