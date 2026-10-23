//! Unit tests of the library modules. Each file mirrors one source file.
#![allow(unused_imports)]

#[cfg(feature = "tier3")]
mod app_snapshot;
mod common_density;
mod common_geometry_tri_mesh;
#[cfg(feature = "tier1")]
mod io_memfs;
#[cfg(feature = "rheo")]
mod rheo_basis;
#[cfg(feature = "rheo")]
mod rheo_closure;
#[cfg(feature = "rheo")]
mod rheo_project;
#[cfg(feature = "tier1")]
mod rtfp_chords;
#[cfg(feature = "tier1")]
mod rtfp_lebedev;
#[cfg(feature = "tier1")]
mod rtfp_tree;
#[cfg(feature = "tier2")]
mod scf_basis;
#[cfg(feature = "tier2")]
mod scf_body;
#[cfg(feature = "tier2")]
mod scf_cores;
#[cfg(feature = "tier2")]
mod scf_evolve;
#[cfg(feature = "tier2")]
mod scf_frame;
#[cfg(feature = "tier2")]
mod scf_harmonics;
#[cfg(feature = "tier2")]
mod scf_jets;
#[cfg(feature = "tier2")]
mod scf_mode_c_spectral;
#[cfg(feature = "tier2")]
mod scf_quad;
#[cfg(feature = "tier2")]
mod scf_relax;
#[cfg(feature = "tier2")]
mod scf_runout;
#[cfg(feature = "tier2")]
mod scf_socp;
#[cfg(feature = "tier2")]
mod scf_socp_pdhg;
#[cfg(feature = "tier2")]
mod scf_spin;
#[cfg(feature = "tier2")]
mod scf_stokes;
#[cfg(feature = "tier2")]
mod scf_transport;
