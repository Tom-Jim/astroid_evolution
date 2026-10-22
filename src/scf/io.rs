//! Project-to-body construction, snapshots and exports.

use std::path::Path;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::common::density::CauchyKernel;
use crate::common::math::DVec3;
use crate::io::{ImportRequest, load_mesh_file};

use super::body::{Body, Topology};
use super::evolve::State;
use super::params::Project;
use crate::common::error::SimError;

const FORMAT: &str = "scf-snapshot-v4";
const IDENTITY: [[f64; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KernelRecord {
    pub center: [f64; 3],
    pub weight: f64,
    /// Metric `S`, `(xx, yy, zz, xy, xz, yz)`.
    pub metric: [f64; 6],
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
                    metric: k.metric,
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

    /// A body at `t = 0` in its own frame, with no spin and no velocity.
    pub fn at_rest(body: &Body, level: usize) -> Self {
        Self::from_body(body, level, 0.0, 0, 0.0, IDENTITY, DVec3::ZERO, vec![])
    }

    /// Everything an evolution needs to resume: body, clock, spin, frame and velocity.
    pub fn from_state(state: &State) -> Self {
        Self::from_body(
            &state.body,
            state.level,
            state.t,
            state.step,
            state.omega,
            state.frame_rotation,
            state.frame_translation,
            state.velocity.clone(),
        )
    }

    pub fn to_state(&self) -> Result<State, SimError> {
        Ok(State {
            body: self.to_body()?,
            level: self.level,
            t: self.time,
            step: self.step,
            omega: self.omega,
            frame_rotation: self.frame_rotation,
            frame_translation: DVec3::from_array(self.frame_translation),
            velocity: self.velocity.clone(),
        })
    }

    pub fn to_body(&self) -> Result<Body, SimError> {
        if self.dirs.len() != self.radius.len() || self.dirs.len() != self.rho_s.len() {
            return Err(SimError::io(
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
                CauchyKernel::try_anisotropic(
                    DVec3::from_array(k.center),
                    k.weight,
                    k.metric,
                    k.alpha,
                )
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

pub fn write_snapshot(path: &Path, snap: &Snapshot) -> Result<(), SimError> {
    ensure_parent(path)?;
    let text = serde_json::to_string(snap)?;
    std::fs::write(path, text).map_err(|e| SimError::io(format!("{}: {e}", path.display())))
}

pub fn read_snapshot(path: &Path) -> Result<Snapshot, SimError> {
    // The page and its workers read installed bytes; native builds read the disk.
    let text = crate::io::memfs::read_to_string(path).map_err(SimError::io)?;
    let snap: Snapshot = serde_json::from_str(&text)
        .map_err(|e| SimError::io(format!("{}: {e}", path.display())))?;
    if snap.format != FORMAT {
        return Err(SimError::io(format!(
            "{} is a '{}' container; expected '{FORMAT}'",
            path.display(),
            snap.format
        )));
    }
    Ok(snap)
}

fn ensure_parent(path: &Path) -> Result<(), SimError> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .map_err(|e| SimError::io(format!("{}: {e}", parent.display())))?;
        }
    }
    Ok(())
}

/// `ρ_s⁰ = ρ_bulk h_s` and the initial star-shaped polyhedron of the project.
pub fn initial_body(project: &Project) -> Result<Body, SimError> {
    let _span = super::timing::span("mesh_load");
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
) -> Result<(Body, DVec3), SimError> {
    let (topo, radius, origin) = match project.body.shape.as_str() {
        "mesh_rays" => Topology::star_from_rays(project.body.level, vertices, &faces)?,
        _ => Topology::star_from_mesh(vertices, faces)?,
    };
    let kernels = project
        .kernels()
        .into_iter()
        .map(|k| CauchyKernel::try_anisotropic(k.center - origin, k.weight, k.metric, k.alpha))
        .collect::<Result<Vec<_>, _>>()?;
    let body = assemble(project, Arc::new(topo), radius, kernels)?;
    Ok((body, origin))
}

fn assemble(
    project: &Project,
    topo: Arc<Topology>,
    radius: Vec<f64>,
    kernels: Vec<CauchyKernel>,
) -> Result<Body, SimError> {
    let rho_bulk = project.material.spec.rho_p * project.material.spec.nu0;
    let rho_s = vec![rho_bulk * project.material.spec.h_s; topo.n_vertices()];
    Body::new(topo, radius, project.density.bulk, kernels, rho_s)
}

pub fn write_obj(path: &Path, body: &Body) -> Result<(), SimError> {
    let positions = body.positions().into_iter().map(|p| p.to_array());
    crate::common::obj::write(path, "", positions, &body.topo.faces)
}

pub fn write_vertices_csv(path: &Path, body: &Body) -> Result<(), SimError> {
    ensure_parent(path)?;
    let mut text = String::from("i,x,y,z,radius,rho_s\n");
    for (i, p) in body.positions().iter().enumerate() {
        text.push_str(&format!(
            "{i},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e}\n",
            p.x, p.y, p.z, body.radius[i], body.rho_s[i]
        ));
    }
    std::fs::write(path, text).map_err(|e| SimError::io(format!("{}: {e}", path.display())))
}
