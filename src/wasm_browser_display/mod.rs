//! WASM 浏览器显示与 WebGPU 并行计算模块
//! 包含 WebGPU 能力探测、浏览器资源加载、WGSL 兼容性检查、CPU fallback 与前端交互会话

pub mod asset_fetch;
pub mod canvas;
pub mod capability_probe;
pub mod readback;
pub mod wasm_entry;
pub mod web_gpu;

pub use wasm_entry::WasmSession;
