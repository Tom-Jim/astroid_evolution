//! WebGPU 运行上下文初始化与 CPU Fallback 计算调度分发
//! 严格依照 docs/software.md 第 14、15.7 节 与 ryugu_cauchy_gravity 实现

use crate::wasm_browser_display::capability_probe::{probe_webgpu_support, FallbackReason};
use std::sync::Arc;

/// 求解后端分发模式
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolverBackendMode {
    /// WebGPU / GPU 硬件加速管线
    WebGpuHardware,
    /// CPU 多线程/单线程参考求解器 (Fallback)
    CpuFallback(FallbackReason),
}

/// WebGPU 运行时核心上下文
pub struct WebGpuContext {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

/// 初始化 WebGPU 设备；若不可用则返回相应的 FallbackReason
pub async fn init_webgpu_or_fallback() -> Result<WebGpuContext, FallbackReason> {
    let probe = probe_webgpu_support();
    if !probe.supported {
        return Err(probe.fallback_reason.unwrap_or(FallbackReason::NoWebGpu));
    }

    let instance = wgpu::Instance::default();

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
            ..Default::default()
        })
        .await
        .map_err(|_| FallbackReason::AdapterNotFound)?;

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("AstroidEvolution WebGPU Device"),
            ..Default::default()
        })
        .await
        .map_err(|_| FallbackReason::DeviceLost)?;

    Ok(WebGpuContext {
        instance,
        adapter,
        device,
        queue,
    })
}

/// 求解执行调度器：支持自动在 WebGPU 与 CPU 之间平滑降级
pub struct ComputeDispatcher {
    pub mode: SolverBackendMode,
    pub gpu_ctx: Option<Arc<WebGpuContext>>,
}

impl ComputeDispatcher {
    /// 自动创建调度器并探针硬件
    pub async fn create_auto() -> Self {
        match init_webgpu_or_fallback().await {
            Ok(ctx) => Self {
                mode: SolverBackendMode::WebGpuHardware,
                gpu_ctx: Some(Arc::new(ctx)),
            },
            Err(reason) => Self {
                mode: SolverBackendMode::CpuFallback(reason),
                gpu_ctx: None,
            },
        }
    }

    /// 强制使用 CPU 参考解模式
    pub fn create_cpu_only() -> Self {
        Self {
            mode: SolverBackendMode::CpuFallback(FallbackReason::UserForcedCpu),
            gpu_ctx: None,
        }
    }

    /// 是否正在运行于 GPU 模式
    pub fn is_gpu_active(&self) -> bool {
        matches!(self.mode, SolverBackendMode::WebGpuHardware)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_dispatcher_cpu_fallback() {
        let dispatcher = ComputeDispatcher::create_cpu_only();
        assert!(!dispatcher.is_gpu_active());
        assert_eq!(
            dispatcher.mode,
            SolverBackendMode::CpuFallback(FallbackReason::UserForcedCpu)
        );
    }
}
