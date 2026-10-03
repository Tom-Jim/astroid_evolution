//! Quasi-static Stokes simulator. Pure functions and SoA components. No trait dispatch.

pub mod accept;
pub mod basis;
pub mod geometry_update;
pub mod reserved;
pub mod rheology;
pub mod state;
pub mod stokes;
pub mod surface;
pub mod yielding;

pub use state::SurfaceState;
