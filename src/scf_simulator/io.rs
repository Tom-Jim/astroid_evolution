//! Project-to-body construction, snapshots and exports.

use std::path::Path;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::common::density::CauchyKernel;
use crate::common::math::DVec3;
use crate::rtfp_solver::io::{ImportRequest, load_mesh_file};

use super::body::{Body, Topology};
use super::error::ScfError;
use super::params::Project;

const FORMAT: &str = "scf-snapshot-v3";

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
    /// Ellipsoid subdivision. An imported mesh stores its own directions and faces;
    /// this value is not used to rebuild that mesh.
    pub level: usize,
    /// Unit directions of the star nodes, one per vertex.
    pub dirs: Vec<[f64; 3]>,
    /// Imported or ellipsoid triangles. Vertex order matches `dirs` and `radius`.
    pub faces: Vec<[u32; 3]>,
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
            dirs: body.topo.dirs.iter().map(|d| d.to_array()).collect(),
            faces: body.topo.faces.clone(),
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
        if self.dirs.len() != self.radius.len() || self.dirs.len() != self.rho_s.len() {
            return Err(ScfError::io(
                "snapshot directions, radii and ρ_s differ in length",
            ));
        }
        let topo = Arc::new(Topology::new(
            self.dirs.iter().copied().map(DVec3::from_array).collect(),
            self.faces.clone(),
        ));
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
    match project.body.shape.as_str() {
        "ellipsoid" => {
            let topo = Arc::new(Topology::icosphere(project.body.level));
            let radius = Body::ellipsoid(topo.clone(), project.body.semi_axes.expect("validated"));
            assemble(project, topo, radius, project.kernels())
        }
        _ => {
            let path = project.resolve(project.body.mesh.as_deref().expect("validated"));
            let doc = load_mesh_file(ImportRequest {
                path: &path,
                length_unit: project.body.mesh_unit.as_deref(),
                reconstruct: None,
            })?;
            mesh_body(project, &doc.mesh.vertices, doc.mesh.faces.clone()).map(|(body, _)| body)
        }
    }
}

/// Star body of an imported triangle mesh.
///
/// Vertices and faces are the mesh itself, centred on the vertex mean. Radial cones
/// that overlap are separated tangentially at constant radius. Cauchy centres
/// move by the same translation, so the density stays attached to the polyhedron.
/// The returned origin is that vertex mean in the input frame (§10.4.4 step 8b maps
/// the evolved body back by this shift). Updating `radius[i]` later moves vertex `i`.
pub fn mesh_body(
    project: &Project,
    vertices: &[DVec3],
    faces: Vec<[u32; 3]>,
) -> Result<(Body, DVec3), ScfError> {
    let (topo, radius, origin) = Topology::star_from_mesh(vertices, faces)?;
    let kernels = project
        .kernels()
        .into_iter()
        .map(|k| {
            CauchyKernel::try_new(k.center - origin, k.weight, k.sigma, k.alpha)
                .map_err(ScfError::from)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let body = assemble(project, Arc::new(topo), radius, kernels)?;
    Ok((body, origin))
}

fn assemble(
    project: &Project,
    topo: Arc<Topology>,
    radius: Vec<f64>,
    kernels: Vec<CauchyKernel>,
) -> Result<Body, ScfError> {
    let rho_bulk = project.material.spec.rho_p * project.material.spec.nu0;
    let rho_s = vec![rho_bulk * project.material.spec.h_s; topo.n_vertices()];
    Body::new(topo, radius, project.density.bulk, kernels, rho_s)
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
