//! Desktop shell. Tauri owns the window and the React control layer; Bevy renders
//! the body natively into the same window, under the transparent webview, and
//! the evolution runs on native compute.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::rc::Rc;

use astroid_evolution::app::bridge::{self, DataRoot};
use astroid_evolution::app::resources::{AppConfig, DEFAULT_PROJECT, DATA_DIR, DEFAULT_SPIN};
use astroid_evolution::app::ui::ShotDir;
use astroid_evolution::app::AsteroidPlugin;
use bevy::app::PluginsState;
use bevy::prelude::*;
use bevy::tasks::tick_global_task_pools_on_main_thread;
use bevy::window::{
    PrimaryWindow, RawHandleWrapper, RawHandleWrapperHolder, WindowResized, WindowResolution,
    WindowScaleFactorChanged, WindowWrapper,
};
use tauri::{Emitter, Manager, RunEvent, WindowEvent};

/// The bundled project, or the repository copy when run from the source tree.
fn data_root(app: &tauri::App) -> PathBuf {
    let bundled = app.path().resource_dir().ok();
    let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("../dist/.stage");
    bundled
        .filter(|d| d.join("assets/projects/ryugu.toml").exists())
        .unwrap_or(source)
}

/// One JSON command from the React panels (`app::bridge::Command`).
#[tauri::command]
fn ui_command(json: String) -> Result<(), String> {
    bridge::send(&json)
}

fn main() {
    let mut tauri_app = tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![ui_command])
        .build(tauri::generate_context!())
        .expect("the Tauri application");
    // Configured windows appear once the event loop has started.
    let window = loop {
        if let Some(w) = tauri_app.get_webview_window("main") {
            break w;
        }
        #[allow(deprecated)]
        tauri_app.run_iteration(|_, _| {});
    };
    let root = data_root(&tauri_app);
    let project = root.join(DEFAULT_PROJECT.strip_prefix(DATA_DIR).unwrap_or(DEFAULT_PROJECT));
    let config = AppConfig::load(&project, DEFAULT_SPIN, None, None)
        .unwrap_or_else(|e| panic!("{}: {e}", project.display()));
    let shots = tauri_app
        .path()
        .picture_dir()
        .map(|d| d.join("Asteroid Evolution"))
        .unwrap_or_else(|_| root.join("output/screenshots"));
    let events = tauri_app.handle().clone();
    bridge::set_sink(move |json| {
        let _ = events.emit("status", json);
    });

    let size = window.inner_size().expect("window size");
    let scale = window.scale_factor().expect("scale factor") as f32;
    let mut app = App::new();
    app.insert_resource(DataRoot(root))
        .insert_resource(ShotDir(shots))
        .add_plugins(bevy::window::WindowPlugin {
        primary_window: Some(Window {
            title: "Asteroid Evolution".into(),
            resolution: WindowResolution::new(size.width, size.height)
                .with_scale_factor_override(scale),
            ..default()
        }),
        ..default()
    });
    // Bevy draws into the Tauri window, under the webview: the primary window carries
    // the handle of the drawing surface.
    let handle = RawHandleWrapper::new(&WindowWrapper::new(surface::under(&window)))
        .expect("a native window handle");
    let world = app.world_mut();
    let (entity, holder) = world
        .query_filtered::<(Entity, &RawHandleWrapperHolder), With<PrimaryWindow>>()
        .single(world)
        .map(|(e, h)| (e, h.clone()))
        .expect("the primary window");
    *holder.0.lock().unwrap() = Some(handle.clone());
    world.entity_mut(entity).insert(handle);
    app.add_plugins(DefaultPlugins.build().disable::<bevy::window::WindowPlugin>())
        .add_plugins(AsteroidPlugin {
            config,
            windowed: true,
        })
        .set_runner(move |app| run(app, tauri_app));
    app.run();
}

/// The view Bevy draws into. wgpu attaches its layer on top of the view it is given,
/// so on macOS Bevy gets a view of its own, ordered below the webview; elsewhere the
/// webview is a child of the window and Bevy draws into the window itself.
mod surface {
    #[cfg(target_os = "macos")]
    pub fn under(window: &tauri::WebviewWindow) -> View {
        use objc2::MainThreadMarker;
        use objc2_app_kit::{NSAutoresizingMaskOptions, NSView, NSWindowOrderingMode};
        let mtm = MainThreadMarker::new().expect("the main thread");
        let content = window.ns_view().expect("the content view") as *const NSView;
        // SAFETY: Tauri returns the window's live content view.
        let content = unsafe { &*content };
        let view = NSView::initWithFrame(mtm.alloc(), content.bounds());
        view.setAutoresizingMask(
            NSAutoresizingMaskOptions::ViewWidthSizable | NSAutoresizingMaskOptions::ViewHeightSizable,
        );
        content.addSubview_positioned_relativeTo(&view, NSWindowOrderingMode::Below, None);
        View(view)
    }

    #[cfg(target_os = "macos")]
    pub struct View(objc2::rc::Retained<objc2_app_kit::NSView>);

    // SAFETY: the handle is only used to create the surface; AppKit calls stay on the
    // main thread, where the Tauri loop runs Bevy.
    #[cfg(target_os = "macos")]
    unsafe impl Send for View {}
    #[cfg(target_os = "macos")]
    unsafe impl Sync for View {}

    #[cfg(target_os = "macos")]
    impl raw_window_handle::HasWindowHandle for View {
        fn window_handle(
            &self,
        ) -> Result<raw_window_handle::WindowHandle<'_>, raw_window_handle::HandleError> {
            use raw_window_handle::{AppKitWindowHandle, RawWindowHandle, WindowHandle};
            let raw = RawWindowHandle::AppKit(AppKitWindowHandle::new(
                std::ptr::NonNull::from(&*self.0).cast(),
            ));
            // SAFETY: the view lives as long as `self`.
            Ok(unsafe { WindowHandle::borrow_raw(raw) })
        }
    }

    #[cfg(target_os = "macos")]
    impl raw_window_handle::HasDisplayHandle for View {
        fn display_handle(
            &self,
        ) -> Result<raw_window_handle::DisplayHandle<'_>, raw_window_handle::HandleError> {
            Ok(raw_window_handle::DisplayHandle::appkit())
        }
    }

    #[cfg(not(target_os = "macos"))]
    pub fn under(window: &tauri::WebviewWindow) -> tauri::WebviewWindow {
        window.clone()
    }
}

/// One Tauri event iteration, then one Bevy frame, until the window closes.
fn run(mut app: App, mut tauri_app: tauri::App) -> AppExit {
    while app.plugins_state() == PluginsState::Adding {
        tick_global_task_pools_on_main_thread();
    }
    app.finish();
    app.cleanup();
    // Window events of one iteration: (physical size, new scale factor).
    let pending = Rc::new(RefCell::new((None, None)));
    loop {
        let sink = pending.clone();
        // `run_iteration` warns about busy loops; this loop is paced by the
        // vsync-blocked present of every Bevy frame.
        #[allow(deprecated)]
        tauri_app.run_iteration(move |_, event| {
            if let RunEvent::WindowEvent { event, .. } = event {
                let mut p = sink.borrow_mut();
                match event {
                    WindowEvent::Resized(s) => p.0 = Some(s),
                    WindowEvent::ScaleFactorChanged {
                        scale_factor,
                        new_inner_size,
                        ..
                    } => *p = (Some(new_inner_size), Some(scale_factor as f32)),
                    _ => {}
                }
            }
        });
        let (resized, rescaled) = std::mem::take(&mut *pending.borrow_mut());
        if tauri_app.webview_windows().is_empty() {
            tauri_app.cleanup_before_exit();
            return AppExit::Success;
        }
        if let Some(size) = resized {
            resize(&mut app, size, rescaled);
        }
        app.update();
        if let Some(exit) = app.should_exit() {
            tauri_app.cleanup_before_exit();
            return exit;
        }
    }
}

fn resize(app: &mut App, size: tauri::PhysicalSize<u32>, factor: Option<f32>) {
    let world = app.world_mut();
    let mut windows = world.query_filtered::<(Entity, &mut Window), With<PrimaryWindow>>();
    let Ok((entity, mut window)) = windows.single_mut(world) else {
        return;
    };
    window.resolution.set_physical_resolution(size.width, size.height);
    if let Some(f) = factor {
        window.resolution.set_scale_factor_override(Some(f));
    }
    let (width, height) = (window.width(), window.height());
    if let Some(f) = factor {
        world.write_message(WindowScaleFactorChanged {
            window: entity,
            scale_factor: f as f64,
        });
    }
    world.write_message(WindowResized {
        window: entity,
        width,
        height,
    });
}
