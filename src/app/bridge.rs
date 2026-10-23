//! The control layer. React panels, in the browser page or in the Tauri webview, drive
//! the viewer with JSON commands and read one JSON status object back. Bevy draws the
//! body; the panels hold no 3D code.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Mutex;

use bevy::camera_controller::pan_orbit_camera::prelude::*;
use bevy::input::ButtonState;
use bevy::input::mouse::{MouseButtonInput, MouseScrollUnit, MouseWheel};
use bevy::input::touch::TouchPhase;
use bevy::pbr::wireframe::{Wireframe, WireframeColor};
use bevy::prelude::*;
use bevy::window::{CursorMoved, PrimaryWindow, WindowEvent};
use serde::{Deserialize, Serialize};

use crate::app::pipeline::EvolutionPool;
use crate::app::history::History;
use crate::app::playback::{self, Playback, VisMode};
use crate::app::render::imported_mesh;
use crate::app::resources::{
    AppConfig, Evolution, FrameStats, Phase, Scene, SurfaceView, display_imported,
};
use crate::app::ui;

#[derive(Deserialize, Debug, Clone, PartialEq)]
#[serde(tag = "cmd", rename_all = "snake_case")]
pub enum Command {
    Pause {
        paused: bool,
    },
    Restart(RunRequest),
    /// 1 yield, 2 slope, 3 regolith, 4 combined, 5 principal tension.
    Layer {
        mode: u32,
    },
    /// Density cutaway through the centre, facing the camera; `offset` in view units.
    Interior {
        on: bool,
        #[serde(default)]
        offset: f32,
    },
    Wireframe {
        on: bool,
    },
    ResetView,
    Screenshot,
    /// Hold the view on stored step `index`; `None` follows the run again.
    Seek {
        #[serde(default)]
        index: Option<usize>,
    },
    /// Pointer in logical window pixels. `button` uses DOM numbering (0 left, 1 middle,
    /// 2 right). Only the Tauri shell sends these: its webview covers the window.
    Pointer {
        x: f32,
        y: f32,
        #[serde(default)]
        button: Option<u8>,
        #[serde(default)]
        down: Option<bool>,
    },
    /// Wheel in DOM pixels (positive scrolls down).
    Wheel {
        dy: f32,
    },
}

/// A new run. Paths are relative to the data root (`DataRoot`).
#[derive(Deserialize, Debug, Clone, PartialEq, Default)]
pub struct RunRequest {
    pub project: String,
    pub spin: String,
    /// `section.key` values that replace the project's (the parameter sliders).
    #[serde(default)]
    pub overrides: BTreeMap<String, f64>,
    #[serde(default)]
    pub snapshot: Option<String>,
    #[serde(default)]
    pub max_steps: Option<usize>,
}

static INBOX: Mutex<Vec<Command>> = Mutex::new(Vec::new());
static POINTER: Mutex<Vec<Command>> = Mutex::new(Vec::new());
type Sink = Box<dyn Fn(&str) + Send + Sync>;
static SINK: Mutex<Option<Sink>> = Mutex::new(None);

/// Queue one JSON command for the next frame.
pub fn send(json: &str) -> Result<(), String> {
    let command: Command = serde_json::from_str(json).map_err(|e| format!("command: {e}"))?;
    let inbox = match command {
        Command::Pointer { .. } | Command::Wheel { .. } => &POINTER,
        _ => &INBOX,
    };
    inbox.lock().unwrap().push(command);
    Ok(())
}

/// Receiver of the status JSON (an event to the webview, a call into the page).
pub fn set_sink(sink: impl Fn(&str) + Send + Sync + 'static) {
    *SINK.lock().unwrap() = Some(Box::new(sink));
}

/// Directory `RunRequest` paths are relative to: the bundle's resources, `dist/.stage`,
/// or the page root.
#[derive(Resource, Clone, Default)]
pub struct DataRoot(pub PathBuf);

/// The last failed command, shown by the panels until the next one succeeds.
#[derive(Resource, Default)]
pub struct Notice(pub Option<String>);

/// Project keys the panels read back (and may override).
pub const PARAMS: &[&str] = &[
    "body.level",
    "material.mu_s",
    "material.mu_2",
    "material.cohesion",
    "material.cohesion_surface",
    "material.cohesion_depth",
    "material.h_s",
    "step.dt_max",
];

const MYR: f64 = 3.15576e13;

#[derive(Serialize)]
pub struct Orbit {
    pub a_km: f64,
    pub period_h: f64,
}

#[derive(Serialize)]
pub struct Status {
    pub phase: Phase,
    pub status: String,
    pub notice: Option<String>,
    pub project: String,
    pub mesh: String,
    pub mesh_unit: String,
    pub spin: String,
    pub snapshot: Option<String>,
    pub params: BTreeMap<String, f64>,
    pub stats: Option<FrameStats>,
    pub time_myr: Option<f64>,
    /// End of the spin schedule.
    pub t_end_myr: Option<f64>,
    pub period_h: Option<f64>,
    pub orbit: Option<Orbit>,
    /// Steps stored for the timeline, and the one held on screen (`None`: live).
    pub frames: usize,
    pub view: Option<usize>,
    /// Newest step and its time while an earlier one is held.
    pub live_step: Option<usize>,
    pub live_time_myr: Option<f64>,
    pub layer: u32,
    pub interior: bool,
    pub cut: f32,
    pub wireframe: bool,
    pub rho_range: [f32; 2],
    /// Metres per view unit.
    pub view_m: f64,
}

/// Everything the panels show, from the viewer resources.
pub fn status(
    config: &AppConfig,
    scene: &Scene,
    evolution: &Evolution,
    history: &History,
    mode: &VisMode,
    rho_range: [f32; 2],
    wireframe: bool,
) -> Status {
    let schedule = crate::scf::spin::SpinSchedule::parse(&config.spin).ok();
    // The held step, if any, drives every readout.
    let stats = history
        .view
        .and_then(|i| history.stats(i))
        .or(evolution.stats);
    let orbit = stats.zip(schedule.as_ref()).and_then(|(s, schedule)| {
        let tide = schedule.tide(s.time);
        (tide > 0.0).then(|| Orbit {
            a_km: schedule.semi_major_axis(s.time) / 1e3,
            period_h: std::f64::consts::TAU / tide.sqrt() / 3600.0,
        })
    });
    let table: Option<toml::Table> = config.project_text.parse().ok();
    let params = PARAMS
        .iter()
        .filter_map(|path| {
            let (section, key) = path.split_once('.')?;
            let value = table.as_ref()?.get(section)?.get(key)?;
            let number = value
                .as_float()
                .or_else(|| value.as_integer().map(|i| i as f64))?;
            Some((path.to_string(), number))
        })
        .collect();
    Status {
        phase: evolution.phase,
        status: evolution.status.clone(),
        notice: None,
        project: config.project_path.clone(),
        mesh: config.mesh_path.clone(),
        mesh_unit: config.mesh_unit.clone(),
        spin: config.spin.clone(),
        snapshot: config.snapshot.as_ref().map(|p| p.display().to_string()),
        params,
        stats,
        time_myr: stats.map(|s| s.time / MYR),
        t_end_myr: schedule
            .as_ref()
            .map(|s| s.t_end() / MYR)
            .filter(|t| t.is_finite()),
        period_h: stats.map(|s| std::f64::consts::TAU / s.omega / 3600.0),
        orbit,
        frames: history.len(),
        view: history.view,
        live_step: evolution.stats.map(|s| s.step),
        live_time_myr: evolution.stats.map(|s| s.time / MYR),
        layer: mode.mode,
        interior: mode.interior,
        cut: mode.cut_offset,
        wireframe,
        rho_range,
        view_m: if scene.scale > 0.0 {
            1.0 / scene.scale
        } else {
            0.0
        },
    }
}

/// Send the status when it changed, at most ten times a second, and every two seconds
/// regardless (a webview that loaded late still gets one).
pub fn publish_status(
    time: Res<Time>,
    config: Res<AppConfig>,
    scene: Res<Scene>,
    evolution: Res<Evolution>,
    playback: Res<Playback>,
    history: Res<History>,
    mode: Res<VisMode>,
    notice: Res<Notice>,
    views: Query<Has<Wireframe>, With<SurfaceView>>,
    mut last: Local<(f32, f32, String)>,
) {
    let now = time.elapsed_secs();
    if now - last.0 < 0.1 {
        return;
    }
    last.0 = now;
    let mut out = status(
        &config,
        &scene,
        &evolution,
        &history,
        &mode,
        playback.rho_range,
        views.iter().any(|w| w),
    );
    out.notice = notice.0.clone();
    let Ok(json) = serde_json::to_string(&out) else {
        return;
    };
    if json != last.2 || now - last.1 > 2.0 {
        if let Some(sink) = SINK.lock().unwrap().as_ref() {
            sink(&json);
        }
        last.1 = now;
        last.2 = json;
    }
}

/// Apply the queued panel commands.
#[allow(clippy::too_many_arguments)]
pub fn apply_commands(
    mut commands: Commands,
    root: Res<DataRoot>,
    mut config: ResMut<AppConfig>,
    mut scene: ResMut<Scene>,
    pool: Res<EvolutionPool>,
    mut evolution: ResMut<Evolution>,
    mut playback: ResMut<Playback>,
    mut history: ResMut<History>,
    mut mode: ResMut<VisMode>,
    mut notice: ResMut<Notice>,
    shots: Res<ui::ShotDir>,
    mut meshes: ResMut<Assets<Mesh>>,
    views: Query<(Entity, &Mesh3d), With<SurfaceView>>,
    mut cameras: Query<(&mut Transform, &mut PanOrbitCamera)>,
) {
    let mut queued = std::mem::take(&mut *INBOX.lock().unwrap());
    // A drag sends many seeks; only the last one of a frame is worth decoding.
    if let Some(last) = queued.iter().rposition(|c| matches!(c, Command::Seek { .. })) {
        let mut i = 0;
        queued.retain(|c| {
            i += 1;
            !matches!(c, Command::Seek { .. }) || i - 1 == last
        });
    }
    for command in queued {
        match command {
            Command::Pause { paused } => match (evolution.phase, paused) {
                (Phase::Running, true) => {
                    evolution.control.set_paused(true);
                    evolution.phase = Phase::Paused;
                }
                (Phase::Paused, false) => {
                    evolution.control.set_paused(false);
                    evolution.phase = Phase::Running;
                }
                _ => {}
            },
            Command::Restart(request) => {
                match restart(&root.0, &request) {
                    Ok((next, next_scene)) => {
                        // The old worker stops at its next step gate; its channel is dropped.
                        evolution.control.request_quit();
                        *evolution = Evolution::default();
                        *playback = Playback::default();
                        history.clear();
                        let positions = display_imported(
                            &next_scene.vertices,
                            next_scene.center,
                            next_scene.scale,
                        );
                        for (_, mesh) in &views {
                            if let Some(mut slot) = meshes.get_mut(&mesh.0) {
                                *slot = imported_mesh(positions.clone(), &next_scene.faces);
                            }
                        }
                        *config = next;
                        *scene = next_scene;
                        ui::start_worker(&mut evolution, &config, scene.scale, &pool);
                        notice.0 = None;
                    }
                    Err(err) => notice.0 = Some(err),
                }
            }
            Command::Layer { mode: layer } => mode.mode = layer.clamp(1, 5),
            Command::Interior { on, offset } => {
                mode.interior = on;
                mode.cut_offset = offset.clamp(-1.0, 1.0);
            }
            Command::Wireframe { on } => {
                for (entity, _) in &views {
                    if on {
                        commands
                            .entity(entity)
                            .insert((Wireframe, WireframeColor { color: ui::WIRE_COLOR }));
                    } else {
                        commands
                            .entity(entity)
                            .remove::<(Wireframe, WireframeColor)>();
                    }
                }
            }
            Command::ResetView => {
                for (mut transform, mut camera) in &mut cameras {
                    *transform = ui::home_transform();
                    camera.last_anchor_depth = -f64::from(ui::START_RADIUS);
                }
            }
            Command::Screenshot => ui::save_screenshot(&mut commands, &shots, "asteroid"),
            Command::Seek { index } => {
                let newest = history.len().checked_sub(1);
                // Seeking to the newest step is the same as following the run.
                let held = index.filter(|i| Some(*i) != newest);
                history.view = held;
                if let Some(snap) = held.or(newest).and_then(|i| history.frame(i)) {
                    playback::show(&mut playback, snap);
                }
            }
            Command::Pointer { .. } | Command::Wheel { .. } => {}
        }
    }
}

fn restart(root: &std::path::Path, request: &RunRequest) -> Result<(AppConfig, Scene), String> {
    use crate::io::memfs::relative_inside;
    let snapshot = match &request.snapshot {
        Some(s) => Some(root.join(relative_inside(s)?)),
        None => None,
    };
    let mut next = AppConfig::load(
        &root.join(relative_inside(&request.project)?),
        &request.spin,
        request.max_steps,
        snapshot,
    )?;
    next.apply_overrides(&request.overrides)?;
    let scene = crate::app::scene_of(&next)?;
    Ok((next, scene))
}

/// The Tauri webview covers the window, so pointer input reaches Bevy as commands. They
/// become the same messages a winit window writes; picking and the camera read them.
pub fn forward_pointer(
    window: Query<Entity, With<PrimaryWindow>>,
    mut last: Local<Option<Vec2>>,
    mut moved: MessageWriter<CursorMoved>,
    mut buttons: MessageWriter<MouseButtonInput>,
    mut wheels: MessageWriter<MouseWheel>,
    mut events: MessageWriter<WindowEvent>,
) {
    let queued = std::mem::take(&mut *POINTER.lock().unwrap());
    let Ok(window) = window.single() else {
        return;
    };
    for command in queued {
        match command {
            Command::Pointer { x, y, button, down } => {
                let position = Vec2::new(x, y);
                let event = CursorMoved {
                    window,
                    position,
                    delta: last.map(|p| position - p),
                };
                *last = Some(position);
                moved.write(event.clone());
                events.write(WindowEvent::CursorMoved(event));
                if let (Some(button), Some(down)) = (button, down) {
                    let event = MouseButtonInput {
                        button: match button {
                            0 => MouseButton::Left,
                            1 => MouseButton::Middle,
                            2 => MouseButton::Right,
                            n => MouseButton::Other(u16::from(n)),
                        },
                        state: if down {
                            ButtonState::Pressed
                        } else {
                            ButtonState::Released
                        },
                        window,
                    };
                    buttons.write(event);
                    events.write(WindowEvent::MouseButtonInput(event));
                }
            }
            Command::Wheel { dy } => {
                let event = MouseWheel {
                    unit: MouseScrollUnit::Pixel,
                    x: 0.0,
                    y: -dy,
                    window,
                    phase: TouchPhase::Moved,
                };
                wheels.write(event);
                events.write(WindowEvent::MouseWheel(event));
            }
            _ => {}
        }
    }
}

