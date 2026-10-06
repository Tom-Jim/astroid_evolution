//! Project-to-body construction, snapshots and exports.

use std::path::Path;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::common::density::CauchyKernel;
use crate::common::geometry::TriMesh;
use crate::common::math::DVec3;
use crate::rtfp_solver::io::{ImportRequest, load_mesh_file};

use super::body::{Body, Topology};
use super::error::ScfError;
use super::params::Project;

const FORMAT: &str = "scf-snapshot-v2";

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KernelRecord {
    pub center: [f64; 3],
    pub weight: f64,
    pub sigma: f64,
    pub alpha: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Snapshot {
    pub format: String,
    pub level: usize,
    pub bulk: f64,
    pub kernels: Vec<KernelRecord>,
    pub radius: Vec<f64>,
    pub rho_s: Vec<f64>,
    pub time: f64,
    pub step: usize,
    pub omega: f64,
    /// Accumulated frame map `x_snapshot = Rᵀ x_initial-frame…`: rows `(e_x', e_y', e_z')`.
    pub frame_rotation: [[f64; 3]; 3],
    pub frame_translation: [f64; 3],
    /// Last velocity coefficients (`u⁰` of the next step when the frame is unchanged).
    pub velocity: Vec<f64>,
}

impl Snapshot {
    pub fn from_body(
        body: &Body,
        level: usize,
        time: f64,
        step: usize,
        omega: f64,
        frame_rotation: [[f64; 3]; 3],
        frame_translation: DVec3,
        velocity: Vec<f64>,
    ) -> Self {
        Self {
            format: FORMAT.into(),
            level,
            bulk: body.bulk,
            kernels: body
                .kernels
                .iter()
                .map(|k| KernelRecord {
                    center: k.center.to_array(),
                    weight: k.weight,
                    sigma: k.sigma,
                    alpha: k.alpha,
                })
                .collect(),
            radius: body.radius.clone(),
            rho_s: body.rho_s.clone(),
            time,
            step,
            omega,
            frame_rotation,
            frame_translation: frame_translation.to_array(),
            velocity,
        }
    }

    pub fn to_body(&self) -> Result<Body, ScfError> {
        let topo = Arc::new(Topology::icosphere(self.level));
        let kernels = self
            .kernels
            .iter()
            .map(|k| {
                CauchyKernel::try_new(DVec3::from_array(k.center), k.weight, k.sigma, k.alpha)
                    .map_err(ScfError::from)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Body::new(
            topo,
            self.radius.clone(),
            self.bulk,
            kernels,
            self.rho_s.clone(),
        )
    }
}

pub fn write_snapshot(path: &Path, snap: &Snapshot) -> Result<(), ScfError> {
    ensure_parent(path)?;
    let text = serde_json::to_string(snap).map_err(|e| ScfError::io(e.to_string()))?;
    std::fs::write(path, text).map_err(|e| ScfError::io(format!("{}: {e}", path.display())))
}

pub fn read_snapshot(path: &Path) -> Result<Snapshot, ScfError> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| ScfError::io(format!("{}: {e}", path.display())))?;
    let snap: Snapshot = serde_json::from_str(&text)
        .map_err(|e| ScfError::io(format!("{}: {e}", path.display())))?;
    if snap.format != FORMAT {
        return Err(ScfError::io(format!(
            "{} is a '{}' container; expected '{FORMAT}'",
            path.display(),
            snap.format
        )));
    }
    Ok(snap)
}

fn ensure_parent(path: &Path) -> Result<(), ScfError> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .map_err(|e| ScfError::io(format!("{}: {e}", parent.display())))?;
        }
    }
    Ok(())
}

/// `ρ_s⁰ = ρ_bulk h_s` and the initial star-shaped polyhedron of the project.
pub fn initial_body(project: &Project) -> Result<Body, ScfError> {
    let topo = Arc::new(Topology::icosphere(project.body.level));
    let nv = topo.n_vertices();
    let radius = match project.body.shape.as_str() {
        "ellipsoid" => Body::ellipsoid(topo.clone(), project.body.semi_axes.expect("validated")),
        _ => {
            let path = project.resolve(project.body.mesh.as_deref().expect("validated"));
            let doc = load_mesh_file(ImportRequest {
                path: &path,
                length_unit: project.body.mesh_unit.as_deref(),
                reconstruct: None,
            })?;
            let n = doc.mesh.vertices.len() as f64;
            let mut origin = DVec3::ZERO;
            for v in &doc.mesh.vertices {
                origin = origin + *v;
            }
            let origin = origin * (1.0 / n);
            let shifted: Vec<DVec3> = doc.mesh.vertices.iter().map(|v| *v - origin).collect();
            let mesh = TriMesh::try_new(shifted, doc.mesh.faces.clone())?;
            Body::radii_from_mesh(&mesh, DVec3::ZERO, &topo.dirs)?
        }
    };
    let rho_bulk = project.material.spec.rho_p * project.material.spec.nu0;
    let rho_s = vec![rho_bulk * project.material.spec.h_s; nv];
    Body::new(topo, radius, project.density.bulk, project.kernels(), rho_s)
}

pub fn write_obj(path: &Path, body: &Body) -> Result<(), ScfError> {
    ensure_parent(path)?;
    let mut text = String::new();
    for p in body.positions() {
        text.push_str(&format!("v {:.9e} {:.9e} {:.9e}\n", p.x, p.y, p.z));
    }
    for t in &body.topo.faces {
        text.push_str(&format!("f {} {} {}\n", t[0] + 1, t[1] + 1, t[2] + 1));
    }
    std::fs::write(path, text).map_err(|e| ScfError::io(format!("{}: {e}", path.display())))
}

pub fn write_vertices_csv(path: &Path, body: &Body) -> Result<(), ScfError> {
    ensure_parent(path)?;
    let mut text = String::from("i,x,y,z,radius,rho_s\n");
    for (i, p) in body.positions().iter().enumerate() {
        text.push_str(&format!(
            "{i},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e}\n",
            p.x, p.y, p.z, body.radius[i], body.rho_s[i]
        ));
    }
    std::fs::write(path, text).map_err(|e| ScfError::io(format!("{}: {e}", path.display())))
}
