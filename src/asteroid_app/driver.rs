use std::path::PathBuf;

use bevy::prelude::*;

use crate::asteroid_app::components::{StarMesh, StepRecord, SurfaceFields};
use crate::asteroid_app::resources::{StepControl, StepPhase};
use crate::asteroid_app::PublishBody;
use crate::common::math::DVec3;
use crate::rheo_friction::RheoGpu;
use crate::scf_simulator::evolve::{compose_frame, start};
use crate::scf_simulator::frame::realign;
use crate::scf_simulator::io::{initial_body, write_obj, write_snapshot, Snapshot};
use crate::scf_simulator::params::Project;
use crate::scf_simulator::relax::relax;

#[derive(Resource)]
pub struct ScfActive;

#[derive(Resource)]
struct ProjectSource {
    text: String,
    dir: PathBuf,
    omega: f64,
    out: Option<PathBuf>,
    obj: Option<PathBuf>,
}

/// Runs the same closed relaxation as `scf-cli relax` and publishes the actual project
/// surface. The transition component keeps the change visible in Bevy.
pub struct ScfPlugin {
    pub project_text: String,
    pub project_dir: PathBuf,
    pub omega: f64,
    pub out: Option<PathBuf>,
    pub obj: Option<PathBuf>,
}

impl Plugin for ScfPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(ScfActive)
            .insert_resource(ProjectSource {
                text: self.project_text.clone(),
                dir: self.project_dir.clone(),
                omega: self.omega,
                out: self.out.clone(),
                obj: self.obj.clone(),
            })
            .insert_resource(StepControl::Run)
            .add_systems(Startup, publish_relaxed_body.in_set(PublishBody))
            .add_systems(Update, animate_relaxation);
    }
}

#[derive(Component)]
struct RelaxTransition {
    from: Vec<f64>,
    to: Vec<f64>,
    elapsed: f32,
}

fn publish_relaxed_body(mut commands: Commands, source: Res<ProjectSource>) {
    let project = Project::parse(&source.text, source.dir.clone()).expect("closed project");
    let initial = initial_body(&project).expect("initial project body");
    let (state, forward) = start(&project, initial.clone()).expect("initial SCF state");
    let gpu = RheoGpu::new().expect("rheology GPU");
    let (relaxed, _) = relax(&project, &gpu, &state.body, source.omega).expect("relaxation");
    let (relaxed, frame_report, _) =
        realign(&relaxed, None, project.g, &project.geometry, false).expect("frame alignment");

    let mut final_state = state;
    final_state.body = relaxed.clone();
    final_state.omega = source.omega;
    let (rotation, translation) = compose_frame(
        final_state.frame_rotation,
        final_state.frame_translation,
        &frame_report,
    );
    final_state.frame_rotation = rotation;
    final_state.frame_translation = translation;
    let snapshot = Snapshot::from_body(
        &final_state.body,
        project.body.level,
        final_state.t,
        final_state.step,
        final_state.omega,
        final_state.frame_rotation,
        final_state.frame_translation,
        final_state.velocity.clone(),
    );
    if let Some(path) = &source.out {
        write_snapshot(path, &snapshot).expect("write relaxed snapshot");
    }
    if let Some(path) = &source.obj {
        write_obj(path, &final_state.body).expect("write relaxed OBJ");
    }

    let from = initial.radius.clone();
    let to = relaxed.radius.clone();
    let mesh = star_mesh(&initial);
    let fields = fields_for(&from, &to, &initial.topo.dirs);
    let record = StepRecord {
        step: 0,
        time: final_state.t,
        omega: [0.0, 0.0, source.omega],
        yielded: to.iter().zip(&from).any(|(a, b)| (a - b).abs() > 1.0e-12),
        mobile_fraction: 0.0,
        mass: forward.moments.mass,
        mass_error: 0.0,
    };
    commands.spawn((
        mesh,
        fields,
        record,
        RelaxTransition {
            from,
            to,
            elapsed: 0.0,
        },
    ));
    println!(
        "tier3 relax omega={:.9e} frame_applied={} output={}",
        source.omega,
        frame_report.applied,
        source
            .out
            .as_deref()
            .map_or_else(|| "none".into(), |p| p.display().to_string())
    );
}

fn animate_relaxation(
    time: Res<Time>,
    mut phase: ResMut<StepPhase>,
    mut bodies: Query<(&mut StarMesh, &mut RelaxTransition)>,
) {
    let Ok((mut mesh, mut transition)) = bodies.single_mut() else {
        return;
    };
    transition.elapsed = (transition.elapsed + time.delta_secs() / 2.0).min(1.0);
    let t = transition.elapsed;
    for (radius, (from, to)) in mesh
        .radii
        .iter_mut()
        .zip(transition.from.iter().zip(&transition.to))
    {
        *radius = from + (to - from) * f64::from(t);
    }
    mesh.revision = mesh.revision.saturating_add(1);
    *phase = if t < 1.0 {
        StepPhase::Stepping
    } else {
        StepPhase::Idle
    };
}

fn star_mesh(body: &crate::scf_simulator::body::Body) -> StarMesh {
    StarMesh {
        directions: body.topo.dirs.iter().map(|d| d.to_array()).collect(),
        radii: body.radius.clone(),
        faces: body.topo.faces.clone(),
        revision: 1,
    }
}

fn fields_for(from: &[f64], to: &[f64], dirs: &[DVec3]) -> SurfaceFields {
    let scale = from
        .iter()
        .zip(to)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max)
        .max(1.0e-12);
    let normals: Vec<[f64; 3]> = dirs.iter().map(|d| d.to_array()).collect();
    SurfaceFields {
        yield_criterion: from.iter().zip(to).map(|(a, b)| (b - a) / scale).collect(),
        pressure: vec![0.0; from.len()],
        viscosity: vec![1.0; from.len()],
        shear_rate: vec![0.0; from.len()],
        friction_angle: vec![0.0; from.len()],
        surface_density: vec![1.0; from.len()],
        slide_speed: vec![0.0; from.len()],
        slide_velocity: vec![[0.0; 3]; from.len()],
        gravity: normals.iter().map(|n| [-n[0], -n[1], -n[2]]).collect(),
        normals,
        mobile: from
            .iter()
            .zip(to)
            .map(|(a, b)| (b - a).abs() > 1.0e-12)
            .collect(),
    }
}
