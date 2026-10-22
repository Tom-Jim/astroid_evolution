//! Browser side of the viewer.
//!
//! The page and the evolution worker are two copies of this module. The page
//! owns the WebGPU device and the Bevy window. The worker runs `evolve::run`
//! and sends snapshots back. A compute call from the worker posts its bytes to
//! the page, then waits. The page is never blocked inside that wait, so the
//! canvas keeps drawing.

use std::sync::atomic::{AtomicBool, Ordering};

mod host;
mod wire;

pub use host::{
    boot, call_gpu, deliver, prepare_viewer, set_pause_flag, set_quit_flag, spawn_evolution,
    worker_main,
};

static WORKER_MESH: AtomicBool = AtomicBool::new(false);

/// The evolution worker can post a compute call and wait. The page cannot:
/// WebGPU runs there, and the page must stay free to service that wait.
pub fn use_worker_mesh() {
    WORKER_MESH.store(true, Ordering::Relaxed);
}

pub fn worker_mesh() -> bool {
    WORKER_MESH.load(Ordering::Relaxed)
}

thread_local! {
    static PAGE: std::cell::RefCell<Option<crate::gpu::Exec>> = const { std::cell::RefCell::new(None) };
}

async fn page_exec() -> Result<crate::gpu::Exec, String> {
    let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
    desc.backends = wgpu::Backends::BROWSER_WEBGPU;
    let adapter = wgpu::Instance::new(desc)
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        })
        .await
        .map_err(|e| format!("WebGPU adapter: {e}"))?;
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("astroid_evolution compute"),
            required_limits: adapter.limits(),
            ..Default::default()
        })
        .await
        .map_err(|e| format!("WebGPU device: {e}"))?;
    Ok(crate::gpu::Exec::from_device(
        device,
        queue,
        false,
        "WebGPU (software BVH)".into(),
    ))
}

/// Run one serialized compute job from the worker on the page's device.
pub async fn execute_gpu(request: &[u8]) -> Result<Vec<u8>, String> {
    let mut exec = match PAGE.with(|p| p.borrow_mut().take()) {
        Some(e) => e,
        None => page_exec().await?,
    };
    let reply = crate::gpu::serve(&mut exec, request).await;
    PAGE.with(|p| *p.borrow_mut() = Some(exec));
    Ok(reply)
}
