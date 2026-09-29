//! WASM 浏览器前端主入口与 JavaScript 会话交互控制器
//! 严格依照 docs/software.md 第 14、15.7 节 与 ryugu_cauchy_gravity 实现

use crate::wasm_browser_display::capability_probe::{probe_webgpu_support, WebGpuCapabilityReport};
use serde::{Deserialize, Serialize};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

/// 注册全局 Panic 钩子，将 Rust panic 栈回溯输出至浏览器控制台
pub fn init_panic_hook() {
    #[cfg(target_arch = "wasm32")]
    {
        console_error_panic_hook::set_once();
    }
}

/// 浏览器前端控制指令
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ViewerCommand {
    Run,
    Pause,
    Step,
    Reset,
    ToggleWireframe,
    SetColorMode(ColorMode),
}

/// 表面着色模式
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ColorMode {
    EffectivePotential,
    GravityMagnitude,
    SlopeAngle,
    ShearRate,
}

/// 浏览器交互会话状态控制器
#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
pub struct WasmSession {
    is_running: bool,
    current_step: usize,
    capability: WebGpuCapabilityReport,
    selected_model: String,
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
impl WasmSession {
    /// 创建并初始化会话
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen(constructor))]
    pub fn new() -> Self {
        init_panic_hook();
        let capability = probe_webgpu_support();

        Self {
            is_running: false,
            current_step: 0,
            capability,
            selected_model: "Ryugu".to_string(),
        }
    }

    /// 获取硬件能力与 WebGPU 探测结果的 JSON 字符串
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen(js_name = getCapabilityJson))]
    pub fn get_capability_json(&self) -> String {
        let reason_str = match self.capability.fallback_reason {
            Some(reason) => format!("\"{:?}\"", reason),
            None => "null".to_string(),
        };
        format!(
            r#"{{"supported":{},"adapter_info":"{}","fallback_reason":{},"max_compute_workgroup_size_x":{},"max_storage_buffer_binding_size":{},"max_buffer_size":{},"supports_shader_f16":{},"supports_timestamp_query":{}}}"#,
            self.capability.supported,
            self.capability.adapter_info,
            reason_str,
            self.capability.max_compute_workgroup_size_x,
            self.capability.max_storage_buffer_binding_size,
            self.capability.max_buffer_size,
            self.capability.supports_shader_f16,
            self.capability.supports_timestamp_query
        )
    }

    /// 启动仿真计算
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen(js_name = run))]
    pub fn run(&mut self) {
        self.is_running = true;
    }

    /// 暂停仿真计算
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen(js_name = pause))]
    pub fn pause(&mut self) {
        self.is_running = false;
    }

    /// 单步推进仿真
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen(js_name = step))]
    pub fn step(&mut self) -> usize {
        self.current_step += 1;
        self.current_step
    }

    /// 重置会话
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen(js_name = reset))]
    pub fn reset(&mut self) {
        self.is_running = false;
        self.current_step = 0;
    }

    /// 选择激活的小行星模型
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen(js_name = selectModel))]
    pub fn select_model(&mut self, model_name: &str) {
        self.selected_model = model_name.to_string();
    }

    /// 当前是否处于运行状态
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen(js_name = isRunning))]
    pub fn is_running(&self) -> bool {
        self.is_running
    }

    /// 获取当前迭代步数
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen(js_name = currentStep))]
    pub fn current_step(&self) -> usize {
        self.current_step
    }

    /// 获取当前选中的小行星模型名称
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen(js_name = getSelectedModel))]
    pub fn selected_model(&self) -> String {
        self.selected_model.clone()
    }

}

impl WasmSession {
    /// 获取硬件能力探测报告引用 (Rust 内部/测试调用)
    pub fn capability(&self) -> &WebGpuCapabilityReport {
        &self.capability
    }
}

impl Default for WasmSession {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wasm_session_controls() {
        let mut session = WasmSession::new();
        assert!(!session.is_running());
        assert_eq!(session.current_step(), 0);

        session.run();
        assert!(session.is_running());

        session.step();
        session.step();
        assert_eq!(session.current_step(), 2);

        session.pause();
        assert!(!session.is_running());

        session.reset();
        assert_eq!(session.current_step(), 0);

        let cap_json = session.get_capability_json();
        assert!(cap_json.contains("supported"));
    }
}
