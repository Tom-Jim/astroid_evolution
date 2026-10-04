use bevy::prelude::*;

use crate::asteroid_app::PublishBody;
use crate::asteroid_app::components::{StarMesh, StepRecord, SurfaceFields};
use crate::asteroid_app::events::ParamPatch;
use crate::asteroid_app::resources::{StepControl, StepPhase};
use crate::common::geometry::TriMesh;
use crate::scf_simulator::simulation::{ClosedRun, closed_run};

#[derive(Resource)]
pub struct ScfActive;

#[derive(Resource)]
struct ProjectSource {
    text: String,
    dir: std::path::PathBuf,
}

pub struct ScfPlugin {
    pub project_text: String,
    pub project_dir: std::path::PathBuf,
    pub max_steps: u32,
    pub exit_when_done: bool,
}

impl Plugin for ScfPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(ScfActive)
            .insert_resource(ProjectSource {
                text: self.project_text.clone(),
                dir: self.project_dir.clone(),
            })
            .insert_resource(if self.exit_when_done {
                StepControl::Run
            } else {
                StepControl::Pause
            });
        if self.exit_when_done {
            app.insert_resource(ExitAfter(self.max_steps));
        }
        app.add_systems(Startup, publish_initial_body.in_set(PublishBody))
            .add_systems(Update, advance_closed_steps);
    }
}

#[derive(Resource)]
struct ExitAfter(u32);

fn publish_initial_body(mut commands: Commands, source: Res<ProjectSource>) {
    let report =
        closed_run(&source.text, &source.dir, 0).expect("project closed before the first step");
    let (mesh, fields) = resting_sphere();
    commands.spawn((mesh, fields, record_from(report)));
}

fn advance_closed_steps(
    mut phase: ResMut<StepPhase>,
    mut control: ResMut<StepControl>,
    source: Res<ProjectSource>,
    exit_after: Option<Res<ExitAfter>>,
    mut patches: MessageReader<ParamPatch>,
    mut bodies: Query<(&mut StarMesh, &mut StepRecord)>,
    mut exits: MessageWriter<AppExit>,
) {
    if *phase != StepPhase::Idle {
        return;
    }
    for patch in patches.read() {
        let _ = (patch.key.as_str(), patch.value);
    }
    let step_now = *control == StepControl::Run || *control == StepControl::StepOnce;
    if !step_now {
        return;
    }
    *phase = StepPhase::Stepping;
    let Ok((mut mesh, mut record)) = bodies.single_mut() else {
        *phase = StepPhase::Idle;
        return;
    };
    if exit_after
        .as_ref()
        .is_some_and(|limit| record.step >= u64::from(limit.0))
    {
        println!(
            "steps {} time {} omega {} {} {} mass {} mass_error {}",
            record.step,
            record.time,
            record.omega[0],
            record.omega[1],
            record.omega[2],
            record.mass,
            record.mass_error
        );
        *phase = StepPhase::Idle;
        exits.write(AppExit::Success);
        return;
    }
    let next = record.step as u32 + 1;
    let report = closed_run(&source.text, &source.dir, next).expect("closed project");
    record.step = u64::from(report.steps);
    record.time = report.time;
    record.omega = report.omega;
    record.mass = report.mass;
    record.mass_error = report.mass_error;
    mesh.revision = mesh.revision.saturating_add(1);
    if *control == StepControl::StepOnce {
        *control = StepControl::Pause;
    }
    let finished = exit_after
        .as_ref()
        .is_some_and(|limit| record.step >= u64::from(limit.0));
    if finished {
        println!(
            "steps {} time {} omega {} {} {} mass {} mass_error {}",
            record.step,
            record.time,
            record.omega[0],
            record.omega[1],
            record.omega[2],
            record.mass,
            record.mass_error
        );
    }
    *phase = StepPhase::Idle;
    if finished {
        exits.write(AppExit::Success);
    }
}

pub fn record_from(report: ClosedRun) -> StepRecord {
    StepRecord {
        step: u64::from(report.steps),
        time: report.time,
        omega: report.omega,
        yielded: false,
        mobile_fraction: 0.0,
        mass: report.mass,
        mass_error: report.mass_error,
    }
}

pub fn resting_sphere() -> (StarMesh, SurfaceFields) {
    let tri = TriMesh::uv_sphere(4, 1.0);
    let mut directions = Vec::with_capacity(tri.vertices.len());
    let mut radii = Vec::with_capacity(tri.vertices.len());
    for vertex in &tri.vertices {
        let radius = vertex.length();
        let unit = vertex.try_normalize().expect("sphere vertex");
        directions.push(unit.to_array());
        radii.push(radius);
    }
    let count = radii.len();
    let fields = SurfaceFields {
        yield_criterion: vec![-0.25; count],
        pressure: vec![1.0; count],
        viscosity: vec![1.0e6; count],
        shear_rate: vec![0.0; count],
        friction_angle: vec![0.6; count],
        surface_density: vec![1.0; count],
        slide_speed: vec![0.0; count],
        slide_velocity: vec![[0.0; 3]; count],
        gravity: directions
            .iter()
            .map(|direction| [-direction[0], -direction[1], -direction[2]])
            .collect(),
        normals: directions.clone(),
        mobile: vec![false; count],
    };
    (
        StarMesh {
            directions,
            radii,
            faces: tri.faces.clone(),
            revision: 1,
        },
        fields,
    )
}
