//! Unit tests of `rheo::project`, moved out of `src/rheo/project.rs`.

mod ctx {
    pub use astroid_evolution::common::error::SimError;
    pub use astroid_evolution::rheo::basis::K;
    pub use astroid_evolution::rheo::gpu::{ChainResult, RheoGpu};
    pub use astroid_evolution::rheo::model::{
        Environment, NodeBatch, StarSurface, VelocityField, fibonacci_directions,
    };
    pub use astroid_evolution::rheo::params::{Material, MaterialSpec, SolverSpec};
    pub use serde::Deserialize;
    pub use serde_json::{Value, json};
    pub use std::f64::consts::PI;
}
use ctx::*;

use astroid_evolution::rheo::project::*;

const GOOD: &str = r#"
[material]
nu0 = 0.6
z0 = 6.0
d_p = 0.05
rho_p = 2500.0
mu_s = 0.38
mu_2 = 0.64
i_0 = 0.279
cohesion = 25.0
cohesion_surface = 25.0
cohesion_depth = 1.0
c_c = 1.0
f_h = 1.0e-4
c_t = 1.0
gamma_eps = 1.0e-9
p_min = 1.0e-6
h_s = 5.0
eps_overhang = 0.01
mu_proj_mode = "dynamic"
[solver]
max_iter = 100
tol = 1.0e-12
[surface]
mean_radius = 500.0
modes = [[2, 0, 30.0]]
[velocity]
poloidal = [[2, 1, 2, 1.0e-6]]
toroidal = []
[environment]
gravity_k = 1.0e-5
omega = 1.0e-4
p_ext = 0.0
[nodes]
count = 100
"#;

#[test]
fn closed_project_parses() {
    let p = Project::parse(GOOD).unwrap();
    assert_eq!(p.node_count, 100);
}

#[test]
fn missing_constant_is_model_not_closed() {
    let text = GOOD.replace("i_0 = 0.279\n", "");
    assert_eq!(Project::parse(&text).err().unwrap().kind, "ModelNotClosed");
}

#[test]
fn mode_switch_must_be_explicit_and_valid() {
    let text = GOOD.replace("\"dynamic\"", "\"sometimes\"");
    assert_eq!(Project::parse(&text).err().unwrap().kind, "ModelNotClosed");
}
