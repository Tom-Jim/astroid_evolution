//! Quasi-static Stokes simulator. Pure functions and SoA components. No trait dispatch.

pub mod accept;
pub mod basis;
pub mod checkpoint;
pub mod error;
pub mod general_mesh;
pub mod geometry_update;
pub mod history_review;
pub mod mesh_exec;
pub mod mesh_io;
pub mod mesh_loop;
pub mod pressure;
pub mod quadrature;
pub mod reserved;
pub mod run_sim;
pub mod simulation;
pub mod spin;
pub mod state;
pub mod stepping;
pub mod stokes;
pub mod surface;
pub mod yielding;

pub use error::ScfError;

pub use state::SurfaceState;
