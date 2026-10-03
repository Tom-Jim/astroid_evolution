//! Closed triangle meshes and a flattened median-split BVH.

pub mod bvh;
pub mod star_mesh;
pub mod tri_mesh;

pub use bvh::FlattenedBvh;
pub use star_mesh::StarMesh;
pub use tri_mesh::TriMesh;
