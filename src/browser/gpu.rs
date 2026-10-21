//! WebGPU compute for the worker.
//!
//! The worker encodes a request and blocks in JavaScript. This file also runs
//! on the page, where `execute_gpu` performs the request. Persistent PDHG
//! buffers stay on the device for the whole iteration block.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};

use super::GpuBuffer;
use super::host::call_gpu;

const MAGIC: u32 = 0x3147_5341;
const KIND_SINGLE: u32 = 1;
const KIND_PDHG: u32 = 2;
const KIND_LBVH: u32 = 3;
const KIND_MESH: u32 = 4;
const MESH_LOAD: u32 = 1;
const MESH_ALIGN: u32 = 2;
const MESH_ISECT: u32 = 3;
const FLAG_READ: u32 = 1;
const FLAG_ALLOC: u32 = 2;
const FLAG_KEEP: u32 = 4;
const FLAG_RESIDENT: u32 = 8;
const KIND_DROP: u32 = 5;

struct Cursor<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Cursor<'a> {
    fn rest(&self) -> usize {
        self.bytes.len().saturating_sub(self.at)
    }

    fn u32(&mut self) -> Result<u32, String> {
        if self.rest() < 4 {
            return Err("gpu request ended early".into());
        }
        let value = u32::from_le_bytes(self.bytes[self.at..self.at + 4].try_into().unwrap());
        self.at += 4;
        Ok(value)
    }

    fn bytes(&mut self, len: usize) -> Result<&'a [u8], String> {
        if self.rest() < len {
            return Err("gpu request ended early".into());
        }
        let out = &self.bytes[self.at..self.at + len];
        self.at += len;
        let pad = (4 - (len % 4)) % 4;
        if self.rest() < pad {
            return Err("gpu request ended early".into());
        }
        self.at += pad;
        Ok(out)
    }
}

fn push_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_bytes(out: &mut Vec<u8>, bytes: &[u8]) {
    push_u32(out, bytes.len() as u32);
    out.extend_from_slice(bytes);
    while out.len() % 4 != 0 {
        out.push(0);
    }
}

pub fn dispatch_single(
    shader: &str,
    entry: &str,
    groups: [u32; 3],
    buffers: &[GpuBuffer<'_>],
) -> Result<Vec<Vec<u8>>, String> {
    let mut payload = Vec::new();
    push_u32(&mut payload, MAGIC);
    push_u32(&mut payload, KIND_SINGLE);
    push_bytes(&mut payload, shader.as_bytes());
    push_bytes(&mut payload, entry.as_bytes());
    push_u32(&mut payload, groups[0]);
    push_u32(&mut payload, groups[1]);
    push_u32(&mut payload, groups[2]);
    push_u32(&mut payload, buffers.len() as u32);
    for buffer in buffers {
        push_u32(&mut payload, buffer.binding);
        if buffer.zeros > 0 {
            let len = u32::try_from(buffer.zeros)
                .map_err(|_| "gpu buffer is larger than 4 GiB".to_string())?;
            push_u32(&mut payload, u32::from(buffer.readback) | FLAG_ALLOC);
            push_u32(&mut payload, len);
        } else {
            push_u32(&mut payload, u32::from(buffer.readback));
            push_bytes(&mut payload, buffer.bytes);
        }
    }
    decode_reads(&call_gpu(&payload)?)
}

/// Run one dispatch and leave the zeroed output buffer on the device.
/// The returned id is not a copy of that buffer.
pub fn keep_output(
    shader: &str,
    entry: &str,
    groups: [u32; 3],
    buffers: &[GpuBuffer<'_>],
    out_binding: u32,
    out_len: usize,
) -> Result<u32, String> {
    let len = u32::try_from(out_len).map_err(|_| "gpu buffer is larger than 4 GiB".to_string())?;
    let mut payload = Vec::new();
    push_u32(&mut payload, MAGIC);
    push_u32(&mut payload, KIND_SINGLE);
    push_bytes(&mut payload, shader.as_bytes());
    push_bytes(&mut payload, entry.as_bytes());
    push_u32(&mut payload, groups[0]);
    push_u32(&mut payload, groups[1]);
    push_u32(&mut payload, groups[2]);
    push_u32(&mut payload, buffers.len() as u32 + 1);
    for buffer in buffers {
        push_u32(&mut payload, buffer.binding);
        if buffer.zeros > 0 {
            let n = u32::try_from(buffer.zeros)
                .map_err(|_| "gpu buffer is larger than 4 GiB".to_string())?;
            push_u32(&mut payload, u32::from(buffer.readback) | FLAG_ALLOC);
            push_u32(&mut payload, n);
        } else {
            push_u32(&mut payload, u32::from(buffer.readback));
            push_bytes(&mut payload, buffer.bytes);
        }
    }
    push_u32(&mut payload, out_binding);
    push_u32(&mut payload, FLAG_READ | FLAG_ALLOC | FLAG_KEEP);
    push_u32(&mut payload, len);
    let reads = decode_reads(&call_gpu(&payload)?)?;
    let id_bytes = reads.first().ok_or("kept buffer returned nothing")?;
    if id_bytes.len() < 4 {
        return Err("kept buffer id is short".into());
    }
    Ok(u32::from_le_bytes(id_bytes[..4].try_into().unwrap()))
}

/// One dispatch that binds a buffer previously returned by [`keep_output`].
pub fn dispatch_resident(
    shader: &str,
    entry: &str,
    groups: [u32; 3],
    resident_binding: u32,
    resident: u32,
    buffers: &[GpuBuffer<'_>],
) -> Result<Vec<Vec<u8>>, String> {
    let mut payload = Vec::new();
    push_u32(&mut payload, MAGIC);
    push_u32(&mut payload, KIND_SINGLE);
    push_bytes(&mut payload, shader.as_bytes());
    push_bytes(&mut payload, entry.as_bytes());
    push_u32(&mut payload, groups[0]);
    push_u32(&mut payload, groups[1]);
    push_u32(&mut payload, groups[2]);
    push_u32(&mut payload, buffers.len() as u32 + 1);
    push_u32(&mut payload, resident_binding);
    push_u32(&mut payload, FLAG_RESIDENT);
    push_u32(&mut payload, resident);
    for buffer in buffers {
        push_u32(&mut payload, buffer.binding);
        if buffer.zeros > 0 {
            let n = u32::try_from(buffer.zeros)
                .map_err(|_| "gpu buffer is larger than 4 GiB".to_string())?;
            push_u32(&mut payload, u32::from(buffer.readback) | FLAG_ALLOC);
            push_u32(&mut payload, n);
        } else {
            push_u32(&mut payload, u32::from(buffer.readback));
            push_bytes(&mut payload, buffer.bytes);
        }
    }
    decode_reads(&call_gpu(&payload)?)
}

pub fn release_resident(id: u32) -> Result<(), String> {
    if id == 0 {
        return Ok(());
    }
    let mut payload = Vec::new();
    push_u32(&mut payload, MAGIC);
    push_u32(&mut payload, KIND_DROP);
    push_u32(&mut payload, id);
    call_gpu(&payload)?;
    Ok(())
}

pub struct PdhgJob {
    pub iters: u32,
    pub avg_count: u32,
    pub step: [u8; 48],
    pub csc_ptr: Vec<u8>,
    pub csc_row: Vec<u8>,
    pub csc_val: Vec<u8>,
    pub csr_ptr: Vec<u8>,
    pub csr_col: Vec<u8>,
    pub csr_val: Vec<u8>,
    pub long_rows: Vec<u8>,
    pub x: Vec<u8>,
    pub xbar: Vec<u8>,
    pub xavg: Vec<u8>,
    pub y: Vec<u8>,
    pub yavg: Vec<u8>,
    pub aty: Vec<u8>,
    pub ax: Vec<u8>,
    pub q: Vec<u8>,
    pub pdiag: Vec<u8>,
    pub b: Vec<u8>,
    pub tau_diag: Vec<u8>,
    pub sigma_diag: Vec<u8>,
    pub nx: u32,
    pub m: u32,
    pub n_long: u32,
    pub n_jobs: u32,
    pub n_trust: u32,
}

pub struct PdhgOut {
    pub x: Vec<u8>,
    pub xbar: Vec<u8>,
    pub xavg: Vec<u8>,
    pub y: Vec<u8>,
    pub yavg: Vec<u8>,
}

pub fn dispatch_pdhg(job: &PdhgJob) -> Result<PdhgOut, String> {
    let mut payload = Vec::new();
    push_u32(&mut payload, MAGIC);
    push_u32(&mut payload, KIND_PDHG);
    push_u32(&mut payload, job.iters);
    push_u32(&mut payload, job.avg_count);
    push_u32(&mut payload, job.nx);
    push_u32(&mut payload, job.m);
    push_u32(&mut payload, job.n_long);
    push_u32(&mut payload, job.n_jobs);
    push_u32(&mut payload, job.n_trust);
    payload.extend_from_slice(&job.step);
    for bytes in [
        job.csc_ptr.as_slice(),
        job.csc_row.as_slice(),
        job.csc_val.as_slice(),
        job.csr_ptr.as_slice(),
        job.csr_col.as_slice(),
        job.csr_val.as_slice(),
        job.long_rows.as_slice(),
        job.x.as_slice(),
        job.xbar.as_slice(),
        job.xavg.as_slice(),
        job.y.as_slice(),
        job.yavg.as_slice(),
        job.aty.as_slice(),
        job.ax.as_slice(),
        job.q.as_slice(),
        job.pdiag.as_slice(),
        job.b.as_slice(),
        job.tau_diag.as_slice(),
        job.sigma_diag.as_slice(),
    ] {
        push_bytes(&mut payload, bytes);
    }
    let mut reads = decode_reads(&call_gpu(&payload)?)?;
    if reads.len() != 5 {
        return Err(format!("PDHG returned {} buffers", reads.len()));
    }
    Ok(PdhgOut {
        x: reads.remove(0),
        xbar: reads.remove(0),
        xavg: reads.remove(0),
        y: reads.remove(0),
        yavg: reads.remove(0),
    })
}

fn decode_reads(bytes: &[u8]) -> Result<Vec<Vec<u8>>, String> {
    let mut cursor = Cursor { bytes, at: 0 };
    let n = cursor.u32()? as usize;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let len = cursor.u32()? as usize;
        out.push(cursor.bytes(len)?.to_vec());
    }
    Ok(out)
}

fn slice_entry(source: &str, entry: &str) -> String {
    let marker = format!("// ENTRY {entry}\n");
    let Some(at) = source.find(&marker) else {
        return source.to_string();
    };
    let shared_end = source.find("// ENTRY").unwrap_or(at);
    let rest = &source[at + marker.len()..];
    let end = format!("// END ENTRY {entry}");
    let body = rest.split(&end).next().unwrap_or(rest);
    format!("{}\n{body}", reachable_prelude(&source[..shared_end], body))
}

/// naga validates every function in a module, including ones the entry never
/// calls. Helpers name the globals of the kernels that use them, so a module
/// keeps only the functions reachable from its entry.
fn reachable_prelude(shared: &str, entry: &str) -> String {
    let fns = shader_functions(shared);
    let names: HashSet<&str> = fns.iter().map(|item| item.name.as_str()).collect();
    let mut needed: HashSet<String> = HashSet::new();
    let mut stack = shader_calls(entry, &names);
    while let Some(name) = stack.pop() {
        if !needed.insert(name.clone()) {
            continue;
        }
        let Some(span) = fns.iter().find(|item| item.name == name) else {
            continue;
        };
        stack.extend(shader_calls(&shared[span.start..span.end], &names));
    }
    let mut out = String::new();
    let mut cursor = 0;
    for item in &fns {
        out.push_str(&shared[cursor..item.start]);
        if needed.contains(item.name.as_str()) {
            out.push_str(&shared[item.start..item.end]);
        }
        cursor = item.end;
    }
    out.push_str(&shared[cursor..]);
    out
}

struct ShaderFn {
    name: String,
    start: usize,
    end: usize,
}

fn shader_functions(source: &str) -> Vec<ShaderFn> {
    let bytes = source.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i + 3 < bytes.len() {
        if shader_fn_at(source, i) {
            let name_start = i + 3;
            let mut name_end = name_start;
            while name_end < bytes.len() && shader_ident(bytes[name_end]) {
                name_end += 1;
            }
            if name_end > name_start {
                if let Some(end) = shader_function_end(source, name_end) {
                    out.push(ShaderFn {
                        name: source[name_start..name_end].to_string(),
                        start: i,
                        end,
                    });
                    i = end;
                    continue;
                }
            }
        }
        i += 1;
    }
    out
}

fn shader_fn_at(source: &str, index: usize) -> bool {
    if !source[index..].starts_with("fn ") {
        return false;
    }
    index == 0 || {
        let before = source.as_bytes()[index - 1];
        !shader_ident(before)
    }
}

fn shader_ident(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_'
}

fn shader_function_end(source: &str, from: usize) -> Option<usize> {
    let brace = source[from..].find('{')? + from;
    let mut depth = 0i32;
    for (offset, ch) in source[brace..].char_indices() {
        match ch {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(brace + offset + ch.len_utf8());
                }
            }
            _ => {}
        }
    }
    None
}

fn shader_calls(text: &str, names: &HashSet<&str>) -> Vec<String> {
    let bytes = text.as_bytes();
    let mut found = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        if shader_ident(bytes[i]) && (i == 0 || !shader_ident(bytes[i - 1])) {
            let start = i;
            i += 1;
            while i < bytes.len() && shader_ident(bytes[i]) {
                i += 1;
            }
            let name = &text[start..i];
            let mut next = i;
            while next < bytes.len() && bytes[next].is_ascii_whitespace() {
                next += 1;
            }
            let defined_here = start >= 3 && text.as_bytes()[start - 3..start] == *b"fn ";
            if next < bytes.len() && bytes[next] == b'(' && names.contains(name) && !defined_here {
                found.push(name.to_string());
            }
            continue;
        }
        i += 1;
    }
    found
}

fn shader_wesl(name: &str) -> Result<&'static str, String> {
    let source = match name {
        "rheo" => concat!(
            include_str!("../rheo_friction/shaders/ds.wesl"),
            "\n",
            include_str!("../rheo_friction/shaders/chain.wesl"),
        ),
        "socp" => concat!(
            include_str!("../rheo_friction/shaders/ds.wesl"),
            "\n",
            include_str!("../scf_simulator/shaders/socp_normal.wesl"),
        ),
        "pdhg" => concat!(
            include_str!("../rheo_friction/shaders/ds.wesl"),
            "\n",
            include_str!("../scf_simulator/shaders/pdhg.wesl"),
        ),
        "mode_c" => concat!(
            include_str!("../rheo_friction/shaders/ds.wesl"),
            "\n",
            include_str!("../scf_simulator/shaders/mode_c.wesl"),
        ),
        "basis" => concat!(
            include_str!("../rheo_friction/shaders/ds.wesl"),
            "\n",
            include_str!("../scf_simulator/shaders/basis.wesl"),
        ),
        "trace" => concat!(
            include_str!("../rtfp_solver/rtfp/bvh_lib.wesl"),
            "\n",
            include_str!("../rtfp_solver/rtfp/trace.wesl"),
        ),
        "lbvh" => include_str!("../rtfp_solver/rtfp/lbvh.wesl"),
        "mesh" => concat!(
            include_str!("../rheo_friction/shaders/ds.wesl"),
            "\n",
            include_str!("../rtfp_solver/rtfp/mesh_gpu.wesl"),
        ),
        "field" => concat!(
            include_str!("../rheo_friction/shaders/ds.wesl"),
            "\n",
            include_str!("../rtfp_solver/rtfp/bvh_lib.wesl"),
            "\n",
            include_str!("../rtfp_solver/rtfp/field_kernel.wesl"),
        ),
        other => return Err(format!("unknown shader {other}")),
    };
    Ok(source)
}

fn compile_wesl(source: &str) -> Result<String, String> {
    let mut resolver = wesl::resolver::VirtualResolver::new();
    let path: wesl::syntax::ModulePath = "main"
        .parse()
        .map_err(|err| format!("shader path: {err:?}"))?;
    resolver.add_module(path.clone(), std::borrow::Cow::Owned(source.to_string()));
    let compiled = wesl::compile(&path, &wesl::CompileOptions::default(), &resolver)
        .map_err(|err| format!("WESL: {err}"))?;
    Ok(compiled.to_string())
}

struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    modules: HashMap<(String, String), wgpu::ShaderModule>,
    pipelines: HashMap<(String, String), wgpu::ComputePipeline>,
    /// Buffers that stay on the device. The web backend does not free them on drop,
    /// so removal goes through `DropBuffer`.
    residents: HashMap<u32, DropBuffer>,
    next_resident: u32,
}

thread_local! {
    static GPU: RefCell<Option<Gpu>> = const { RefCell::new(None) };
    static GPU_FAULT: RefCell<Option<String>> = const { RefCell::new(None) };
    static GPU_LOST: RefCell<Option<String>> = const { RefCell::new(None) };
}

fn note_gpu_fault(err: wgpu::Error) {
    GPU_FAULT.with(|slot| *slot.borrow_mut() = Some(err.to_string()));
}

fn take_gpu_fault() -> Option<String> {
    GPU_FAULT.with(|slot| slot.borrow_mut().take())
}

fn note_gpu_lost(reason: wgpu::DeviceLostReason, message: String) {
    GPU_LOST.with(|slot| *slot.borrow_mut() = Some(format!("{reason:?}: {message}")));
}

fn take_gpu_lost() -> Option<String> {
    GPU_LOST.with(|slot| slot.borrow_mut().take())
}

struct ErrorScopes {
    validation: wgpu::ErrorScopeGuard,
    memory: wgpu::ErrorScopeGuard,
    internal: wgpu::ErrorScopeGuard,
}

fn open_scopes(device: &wgpu::Device) -> ErrorScopes {
    ErrorScopes {
        internal: device.push_error_scope(wgpu::ErrorFilter::Internal),
        memory: device.push_error_scope(wgpu::ErrorFilter::OutOfMemory),
        validation: device.push_error_scope(wgpu::ErrorFilter::Validation),
    }
}

async fn close_scopes(scopes: ErrorScopes) -> Result<(), String> {
    let mut errors = Vec::new();
    for scope in [scopes.validation, scopes.memory, scopes.internal] {
        if let Some(err) = scope.pop().await {
            errors.push(err.to_string());
        }
    }
    if let Some(prior) = take_gpu_fault() {
        errors.push(prior);
    }
    if let Some(lost) = take_gpu_lost() {
        errors.push(lost);
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors.join(" | "))
    }
}

async fn gpu() -> Result<(), String> {
    let ready = GPU.with(|slot| slot.borrow().is_some());
    if ready {
        return Ok(());
    }
    let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
    desc.backends = wgpu::Backends::BROWSER_WEBGPU;
    let instance = wgpu::Instance::new(desc);
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
        .map_err(|err| format!("WebGPU adapter: {err}"))?;
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            required_limits: adapter.limits(),
            ..Default::default()
        })
        .await
        .map_err(|err| format!("WebGPU device: {err}"))?;
    device.on_uncaptured_error(std::sync::Arc::new(note_gpu_fault));
    device.set_device_lost_callback(note_gpu_lost);
    GPU.with(|slot| {
        *slot.borrow_mut() = Some(Gpu {
            device,
            queue,
            modules: HashMap::new(),
            pipelines: HashMap::new(),
            residents: HashMap::new(),
            next_resident: 1,
        });
    });
    Ok(())
}

fn js_text(err: &wasm_bindgen::JsValue) -> String {
    if let Some(text) = err.as_string() {
        return text;
    }
    if let Ok(message) = js_sys::Reflect::get(err, &wasm_bindgen::JsValue::from_str("message")) {
        if let Some(text) = message.as_string() {
            if !text.is_empty() {
                return text;
            }
        }
    }
    format!("{err:?}")
}

fn js_set(
    target: &wasm_bindgen::JsValue,
    key: &str,
    value: &wasm_bindgen::JsValue,
) -> Result<(), String> {
    js_sys::Reflect::set(target, &wasm_bindgen::JsValue::from_str(key), value)
        .map_err(|err| js_text(&err))?;
    Ok(())
}

/// Dawn drops the device when `createComputePipeline` compiles a large kernel
/// synchronously. The async entry compiles the same WGSL, and the following
/// synchronous create is then a cache hit.
async fn warm_compute_pipeline(
    device: &wgpu::Device,
    wgsl: &str,
    entry: &str,
) -> Result<(), String> {
    use wasm_bindgen::JsCast;
    let gpu = device.as_webgpu().ok_or("WebGPU device is not ready")?;
    let device_js: &wasm_bindgen::JsValue = gpu.as_ref();
    let module_desc = js_sys::Object::new();
    js_set(&module_desc, "code", &wasm_bindgen::JsValue::from_str(wgsl))?;
    let create_module = js_sys::Reflect::get(
        device_js,
        &wasm_bindgen::JsValue::from_str("createShaderModule"),
    )
    .map_err(|err| js_text(&err))?
    .dyn_into::<js_sys::Function>()
    .map_err(|_| "WebGPU createShaderModule is missing".to_string())?;
    let module = create_module
        .call1(device_js, &module_desc)
        .map_err(|err| format!("shader module: {}", js_text(&err)))?;
    let stage = js_sys::Object::new();
    js_set(&stage, "module", &module)?;
    js_set(
        &stage,
        "entryPoint",
        &wasm_bindgen::JsValue::from_str(entry),
    )?;
    let desc = js_sys::Object::new();
    js_set(&desc, "layout", &wasm_bindgen::JsValue::from_str("auto"))?;
    js_set(&desc, "compute", &stage)?;
    let create = js_sys::Reflect::get(
        device_js,
        &wasm_bindgen::JsValue::from_str("createComputePipelineAsync"),
    )
    .map_err(|err| js_text(&err))?
    .dyn_into::<js_sys::Function>()
    .map_err(|_| "WebGPU createComputePipelineAsync is missing".to_string())?;
    let promise = create
        .call1(device_js, &desc)
        .map_err(|err| format!("WebGPU pipeline: {}", js_text(&err)))?
        .dyn_into::<js_sys::Promise>()
        .map_err(|_| "WebGPU pipeline did not return a promise".to_string())?;
    wasm_bindgen_futures::JsFuture::from(promise)
        .await
        .map_err(|err| format!("WebGPU pipeline: {}", js_text(&err)))?;
    Ok(())
}

async fn pipeline(
    shader: &str,
    entry: &str,
) -> Result<(wgpu::Device, wgpu::Queue, wgpu::ComputePipeline), String> {
    let cached = GPU.with(|slot| -> Result<Option<_>, String> {
        let gpu = slot.borrow();
        let gpu = gpu.as_ref().ok_or("WebGPU device is not ready")?;
        let key = (shader.to_string(), entry.to_string());
        Ok(gpu
            .pipelines
            .get(&key)
            .map(|found| (gpu.device.clone(), gpu.queue.clone(), found.clone())))
    })?;
    if let Some(hit) = cached {
        return Ok(hit);
    }
    let wesl = slice_entry(shader_wesl(shader)?, entry);
    let wgsl = compile_wesl(&wesl)?;
    let device = GPU.with(|slot| {
        slot.borrow()
            .as_ref()
            .map(|gpu| gpu.device.clone())
            .ok_or("WebGPU device is not ready".to_string())
    })?;
    warm_compute_pipeline(&device, &wgsl, entry).await?;
    GPU.with(|slot| {
        let mut gpu = slot.borrow_mut();
        let gpu = gpu.as_mut().ok_or("WebGPU device is not ready")?;
        let key = (shader.to_string(), entry.to_string());
        if let Some(found) = gpu.pipelines.get(&key) {
            return Ok((gpu.device.clone(), gpu.queue.clone(), found.clone()));
        }
        let module = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(entry),
                source: wgpu::ShaderSource::Wgsl(wgsl.into()),
            });
        gpu.modules.insert(key.clone(), module.clone());
        let pipeline = gpu
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: None,
                module: &module,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        gpu.pipelines.insert(key, pipeline.clone());
        Ok((gpu.device.clone(), gpu.queue.clone(), pipeline))
    })
}

/// wgpu 30's web backend does not free a GpuBuffer when the Rust value drops.
struct DropBuffer(wgpu::Buffer);

impl Drop for DropBuffer {
    fn drop(&mut self) {
        self.0.destroy();
    }
}

impl std::ops::Deref for DropBuffer {
    type Target = wgpu::Buffer;
    fn deref(&self) -> &wgpu::Buffer {
        &self.0
    }
}

fn alloc_raw(device: &wgpu::Device, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: size.max(16).next_multiple_of(4),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

fn storage(device: &wgpu::Device, bytes: &[u8]) -> DropBuffer {
    DropBuffer(alloc_raw(device, bytes.len() as u64))
}

fn upload(queue: &wgpu::Queue, buffer: &wgpu::Buffer, bytes: &[u8]) {
    if bytes.is_empty() {
        queue.write_buffer(buffer, 0, &[0u8; 16]);
        return;
    }
    queue.write_buffer(buffer, 0, bytes);
}

async fn readback(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    len: usize,
) -> Result<Vec<u8>, String> {
    let size = buffer.size();
    let staging = DropBuffer(device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    }));
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
    let scopes = open_scopes(device);
    queue.submit([encoder.finish()]);
    close_scopes(scopes)
        .await
        .map_err(|err| format!("WebGPU readback: {err}"))?;
    let (tx, rx) = futures_channel::oneshot::channel();
    staging
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
    let _ = device.poll(wgpu::PollType::Poll);
    rx.await
        .map_err(|_| "WebGPU map dropped".to_string())?
        .map_err(|err| {
            let extra = [take_gpu_fault(), take_gpu_lost()]
                .into_iter()
                .flatten()
                .collect::<Vec<_>>()
                .join(" | ");
            format!("WebGPU map: {err} {extra}")
        })?;
    let mapped = staging
        .slice(..)
        .get_mapped_range()
        .map_err(|err| format!("WebGPU mapped range: {err}"))?;
    let take = len.min(mapped.len());
    let out = mapped[..take].to_vec();
    drop(mapped);
    staging.unmap();
    Ok(out)
}

enum Hold {
    Owned(DropBuffer),
    Resident(u32),
}

struct Live {
    binding: u32,
    hold: Hold,
    len: usize,
    readback: bool,
    keep: bool,
}

struct InBuf {
    binding: u32,
    bytes: Vec<u8>,
    zeros: usize,
    readback: bool,
    keep: bool,
    resident: u32,
}

async fn dispatch_buffers(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline: &wgpu::ComputePipeline,
    groups: [u32; 3],
    inputs: &[InBuf],
) -> Result<Vec<Vec<u8>>, String> {
    let mut live = Vec::new();
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    for input in inputs {
        if input.resident != 0 {
            live.push(Live {
                binding: input.binding,
                hold: Hold::Resident(input.resident),
                len: 0,
                readback: false,
                keep: false,
            });
        } else if input.zeros > 0 {
            let buffer = alloc_buffer(device, input.zeros as u64);
            encoder.clear_buffer(&buffer, 0, None);
            live.push(Live {
                binding: input.binding,
                hold: Hold::Owned(buffer),
                len: input.zeros,
                readback: input.readback,
                keep: input.keep,
            });
        } else {
            let buffer = storage(device, &input.bytes);
            upload(queue, &buffer, &input.bytes);
            live.push(Live {
                binding: input.binding,
                hold: Hold::Owned(buffer),
                len: input.bytes.len(),
                readback: input.readback,
                keep: false,
            });
        }
    }
    let mut kept = Vec::new();
    if groups[0] > 0 && groups[1] > 0 && groups[2] > 0 {
        let bind = GPU.with(|slot| {
            let gpu = slot.borrow();
            let gpu = gpu.as_ref().ok_or("WebGPU device is not ready")?;
            let mut entries = Vec::with_capacity(live.len());
            for item in &live {
                let resource = match &item.hold {
                    Hold::Owned(buffer) => buffer.as_entire_binding(),
                    Hold::Resident(id) => gpu
                        .residents
                        .get(id)
                        .ok_or_else(|| format!("GPU buffer {id} is not on the device"))?
                        .as_entire_binding(),
                };
                entries.push(wgpu::BindGroupEntry {
                    binding: item.binding,
                    resource,
                });
            }
            let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipeline.get_bind_group_layout(0),
                entries: &entries,
            });
            Ok::<_, String>(bind)
        })?;
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(groups[0], groups[1], groups[2]);
        }
        kept.push(bind);
    }
    let scopes = open_scopes(device);
    queue.submit([encoder.finish()]);
    let _kept = kept;
    close_scopes(scopes)
        .await
        .map_err(|err| format!("WebGPU dispatch: {err}"))?;
    let mut out = Vec::new();
    let mut kept_bufs: Vec<(usize, DropBuffer)> = Vec::new();
    for item in live {
        if item.keep {
            let Hold::Owned(buffer) = item.hold else {
                return Err("a resident buffer cannot be kept again".into());
            };
            kept_bufs.push((out.len(), buffer));
            out.push(Vec::new());
        } else if item.readback {
            let Hold::Owned(buffer) = &item.hold else {
                return Err("a resident buffer is not copied back".into());
            };
            out.push(readback(device, queue, buffer, item.len).await?);
        }
    }
    if !kept_bufs.is_empty() {
        GPU.with(|slot| {
            let mut gpu = slot.borrow_mut();
            let gpu = gpu.as_mut().ok_or("WebGPU device is not ready")?;
            for (index, buffer) in kept_bufs {
                let id = gpu.next_resident;
                gpu.next_resident = gpu.next_resident.wrapping_add(1).max(1);
                gpu.residents.insert(id, buffer);
                out[index] = id.to_le_bytes().to_vec();
            }
            Ok::<(), String>(())
        })?;
    }
    Ok(out)
}

fn groups_for(n: u32) -> [u32; 3] {
    [n.div_ceil(64).max(1), 1, 1]
}

async fn run_single(bytes: &[u8]) -> Result<Vec<u8>, String> {
    let mut cursor = Cursor { bytes, at: 8 };
    let shader = std::str::from_utf8({
        let len = cursor.u32()? as usize;
        cursor.bytes(len)?
    })
    .map_err(|err| err.to_string())?
    .to_string();
    let entry = std::str::from_utf8({
        let len = cursor.u32()? as usize;
        cursor.bytes(len)?
    })
    .map_err(|err| err.to_string())?
    .to_string();
    let groups = [cursor.u32()?, cursor.u32()?, cursor.u32()?];
    let n = cursor.u32()? as usize;
    let mut inputs = Vec::with_capacity(n);
    for _ in 0..n {
        let binding = cursor.u32()?;
        let flags = cursor.u32()?;
        if flags & FLAG_RESIDENT != 0 {
            inputs.push(InBuf {
                binding,
                bytes: Vec::new(),
                zeros: 0,
                readback: false,
                keep: false,
                resident: cursor.u32()?,
            });
        } else {
            let len = cursor.u32()? as usize;
            if flags & FLAG_ALLOC != 0 {
                inputs.push(InBuf {
                    binding,
                    bytes: Vec::new(),
                    zeros: len,
                    readback: flags & FLAG_READ != 0,
                    keep: flags & FLAG_KEEP != 0,
                    resident: 0,
                });
            } else {
                inputs.push(InBuf {
                    binding,
                    bytes: cursor.bytes(len)?.to_vec(),
                    zeros: 0,
                    readback: flags & FLAG_READ != 0,
                    keep: false,
                    resident: 0,
                });
            }
        }
    }
    gpu().await?;
    let device_for_scope = GPU.with(|slot| {
        slot.borrow()
            .as_ref()
            .map(|gpu| gpu.device.clone())
            .ok_or("WebGPU device is not ready")
    })?;
    let scopes = open_scopes(&device_for_scope);
    let (device, queue, pipeline) = pipeline(&shader, &entry).await?;
    close_scopes(scopes)
        .await
        .map_err(|err| format!("WebGPU pipeline: {err}"))?;
    let reads = dispatch_buffers(&device, &queue, &pipeline, groups, &inputs).await?;
    Ok(encode_reads(&reads))
}

fn encode_reads(reads: &[Vec<u8>]) -> Vec<u8> {
    let mut out = Vec::new();
    push_u32(&mut out, reads.len() as u32);
    for bytes in reads {
        push_bytes(&mut out, bytes);
    }
    out
}

async fn run_pdhg(bytes: &[u8]) -> Result<Vec<u8>, String> {
    let mut cursor = Cursor { bytes, at: 8 };
    let iters = cursor.u32()?;
    let mut avg_count = cursor.u32()?;
    let nx = cursor.u32()?;
    let m = cursor.u32()?;
    let n_long = cursor.u32()?;
    let n_jobs = cursor.u32()?;
    let n_trust = cursor.u32()?;
    if cursor.rest() < 48 {
        return Err("PDHG step header is short".into());
    }
    let mut step = [0u8; 48];
    step.copy_from_slice(&cursor.bytes[cursor.at..cursor.at + 48]);
    cursor.at += 48;
    let mut blobs = Vec::with_capacity(19);
    for _ in 0..19 {
        let len = cursor.u32()? as usize;
        blobs.push(cursor.bytes(len)?.to_vec());
    }
    let (device, queue, _pipeline) = pipeline("pdhg", "pdhg_aty").await?;
    let mut buffers = Vec::new();
    let mut clear = Vec::new();
    for (index, blob) in blobs.iter().enumerate() {
        // 12 aty and 13 ax are scratch. An empty blob means zero them on the device.
        if blob.is_empty() && matches!(index, 12 | 13) {
            let count = if index == 12 {
                u64::from(nx)
            } else {
                u64::from(m)
            };
            clear.push(buffers.len());
            buffers.push(alloc_buffer(&device, count.saturating_mul(8)));
        } else {
            let buffer = storage(&device, blob);
            upload(&queue, &buffer, blob);
            buffers.push(buffer);
        }
    }
    // 0..6 matrix, 7 x, 8 xbar, 9 xavg, 10 y, 11 yavg, 12 aty, 13 ax,
    // 14 q, 15 pdiag, 16 b, 17 tau, 18 sigma.
    let step_buffer = storage(&device, &step);
    for _ in 0..iters {
        step[24..28].copy_from_slice(&avg_count.to_le_bytes());
        upload(&queue, &step_buffer, &step);
        record_pdhg(
            &device,
            &queue,
            &buffers,
            &step_buffer,
            &clear,
            nx,
            m,
            n_long,
            n_jobs,
            n_trust,
        )
        .await?;
        avg_count = avg_count.saturating_add(1);
        clear.clear();
    }
    let mut reads = Vec::new();
    for index in [7, 8, 9, 10, 11] {
        reads.push(readback(&device, &queue, &buffers[index], blobs[index].len()).await?);
    }
    Ok(encode_reads(&reads))
}

pub fn dispatch_lbvh(tris: &[u8]) -> Result<Vec<u8>, String> {
    if tris.len() % 36 != 0 {
        return Err("LBVH triangles are not packed as 9 floats".into());
    }
    let mut payload = Vec::new();
    push_u32(&mut payload, MAGIC);
    push_u32(&mut payload, KIND_LBVH);
    push_bytes(&mut payload, tris);
    let mut reads = decode_reads(&call_gpu(&payload)?)?;
    if reads.len() != 1 {
        return Err(format!("LBVH returned {} buffers", reads.len()));
    }
    Ok(reads.remove(0))
}

async fn run_lbvh(bytes: &[u8]) -> Result<Vec<u8>, String> {
    let mut cursor = Cursor { bytes, at: 8 };
    let len = cursor.u32()? as usize;
    let tris = cursor.bytes(len)?.to_vec();
    if tris.len() % 36 != 0 {
        return Err("LBVH triangles are not packed as 9 floats".into());
    }
    let n = u32::try_from(tris.len() / 36)
        .map_err(|_| "too many triangles for one LBVH".to_string())?;
    if n < 2 {
        return Err("LBVH build was asked for fewer than two triangles".into());
    }
    let groups = n.div_ceil(256);
    if groups > 65535 {
        return Err("LBVH triangle count exceeds one compute dispatch".into());
    }
    let nodes_len = (u64::from(n) * 2 - 1) * 32;
    if nodes_len > 80 * 1024 * 1024 {
        return Err("LBVH node buffer does not fit the worker transfer".into());
    }
    let (device, queue, _) = pipeline("lbvh", "lbvh_bounds").await?;
    let tris_b = alloc_buffer(&device, tris.len() as u64);
    upload(&queue, &tris_b, &tris);
    let aabb = alloc_buffer(&device, u64::from(n) * 24);
    let centroid = alloc_buffer(&device, u64::from(n) * 12);
    let partial = alloc_buffer(&device, u64::from(groups) * 24);
    let scene = alloc_buffer(&device, 24);
    let lo_a = alloc_buffer(&device, u64::from(n) * 4);
    let hi_a = alloc_buffer(&device, u64::from(n) * 4);
    let idx_a = alloc_buffer(&device, u64::from(n) * 4);
    let lo_b = alloc_buffer(&device, u64::from(n) * 4);
    let hi_b = alloc_buffer(&device, u64::from(n) * 4);
    let idx_b = alloc_buffer(&device, u64::from(n) * 4);
    let hist = alloc_buffer(&device, u64::from(groups) * 256 * 4);
    let bin_total = alloc_buffer(&device, 256 * 4);
    let bin_start = alloc_buffer(&device, 256 * 4);
    let tile_offset = alloc_buffer(&device, u64::from(groups) * 256 * 4);
    let parent = alloc_buffer(&device, (u64::from(n) * 2 - 1) * 4);
    let nodes = alloc_buffer(&device, nodes_len);
    let root = alloc_buffer(&device, 4);
    let params = alloc_buffer(&device, 16);
    upload(&queue, &params, &lbv_params(n, groups, 0));
    let mut pass_params = Vec::with_capacity(8);
    for pass in 0..8 {
        let buffer = alloc_buffer(&device, 16);
        upload(&queue, &buffer, &lbv_params(n, groups, pass));
        pass_params.push(buffer);
    }

    let mut steps: Vec<(&str, Vec<&DropBuffer>, u32)> = vec![
        (
            "lbvh_bounds",
            vec![&tris_b, &aabb, &centroid, &params],
            groups,
        ),
        ("lbvh_reduce", vec![&centroid, &partial, &params], groups),
        ("lbvh_scene", vec![&partial, &scene, &params], 1),
        (
            "lbvh_morton",
            vec![&centroid, &scene, &lo_a, &hi_a, &idx_a, &params],
            groups,
        ),
    ];
    let mut src_is_a = true;
    for pass in 0..8 {
        let (lo, hi, idx, lo_o, hi_o, idx_o) = if src_is_a {
            (&lo_a, &hi_a, &idx_a, &lo_b, &hi_b, &idx_b)
        } else {
            (&lo_b, &hi_b, &idx_b, &lo_a, &hi_a, &idx_a)
        };
        let step_params = &pass_params[pass];
        steps.push(("lbvh_hist", vec![lo, hi, &hist, step_params], groups));
        steps.push(("lbvh_bin_total", vec![&hist, &bin_total, step_params], 1));
        steps.push(("lbvh_bin_scan", vec![&bin_total, &bin_start], 1));
        steps.push((
            "lbvh_tile_offset",
            vec![&hist, &bin_start, &tile_offset, step_params],
            1,
        ));
        steps.push((
            "lbvh_scatter",
            vec![lo, hi, idx, lo_o, hi_o, idx_o, &tile_offset, step_params],
            groups,
        ));
        src_is_a = !src_is_a;
    }
    let internal = (n - 1).div_ceil(256).max(1);
    let parent_groups = (n * 2 - 1).div_ceil(256).max(1);
    let word_groups = ((n * 2 - 1) * 8).div_ceil(256).max(1);
    steps.extend([
        ("lbvh_zero", vec![&nodes, &params], word_groups),
        ("lbvh_clear_parent", vec![&parent, &params], parent_groups),
        ("lbvh_leaves", vec![&aabb, &idx_a, &nodes, &params], groups),
        (
            "lbvh_karras",
            vec![&lo_a, &hi_a, &nodes, &parent, &params],
            internal,
        ),
        ("lbvh_init_aabb", vec![&nodes, &params], internal),
        ("lbvh_root", vec![&parent, &root, &params], 1),
        ("lbvh_fit", vec![&nodes, &parent, &params], groups),
        ("lbvh_convert", vec![&nodes, &params], internal),
        ("lbvh_swap_root", vec![&nodes, &root], 1),
        ("lbvh_patch", vec![&nodes, &root, &params], internal),
    ]);

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    let mut kept = Vec::new();
    for (entry, buffers, count) in steps {
        let (_, _, pipe) = pipeline("lbvh", entry).await?;
        let entries: Vec<wgpu::BindGroupEntry> = buffers
            .iter()
            .enumerate()
            .map(|(slot, buffer)| wgpu::BindGroupEntry {
                binding: slot as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect();
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(entry),
            layout: &pipe.get_bind_group_layout(0),
            entries: &entries,
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&pipe);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(count, 1, 1);
        }
        kept.push(bind);
    }
    let _kept = kept;
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    queue.submit([encoder.finish()]);
    if let Some(err) = scope.pop().await {
        let prior = take_gpu_fault().unwrap_or_default();
        return Err(format!("LBVH: {err} {prior}"));
    }
    if let Some(prior) = take_gpu_fault() {
        return Err(format!("LBVH: {prior}"));
    }
    let nodes_bytes = readback(&device, &queue, &nodes, nodes_len as usize).await?;
    Ok(encode_reads(&[nodes_bytes]))
}

fn lbv_params(n: u32, groups: u32, pass: u32) -> [u8; 16] {
    let mut out = [0u8; 16];
    out[0..4].copy_from_slice(&n.to_le_bytes());
    out[4..8].copy_from_slice(&groups.to_le_bytes());
    out[8..12].copy_from_slice(&pass.to_le_bytes());
    out
}

fn groups256(n: u32) -> Result<u32, String> {
    let groups = n.div_ceil(256).max(1);
    if groups > 65535 {
        return Err("mesh compute dispatch exceeds one dimension".into());
    }
    Ok(groups)
}

async fn record_compute(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    entry: &str,
    buffers: &[&wgpu::Buffer],
    groups: u32,
    kept: &mut Vec<wgpu::BindGroup>,
) -> Result<(), String> {
    let (_, _, pipe) = pipeline("mesh", entry).await?;
    let entries: Vec<wgpu::BindGroupEntry> = buffers
        .iter()
        .enumerate()
        .map(|(slot, buffer)| wgpu::BindGroupEntry {
            binding: slot as u32,
            resource: buffer.as_entire_binding(),
        })
        .collect();
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(entry),
        layout: &pipe.get_bind_group_layout(0),
        entries: &entries,
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&pipe);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(groups, 1, 1);
    }
    kept.push(bind);
    Ok(())
}

async fn finish_compute(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    encoder: wgpu::CommandEncoder,
    what: &str,
) -> Result<(), String> {
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    queue.submit([encoder.finish()]);
    if let Some(err) = scope.pop().await {
        let prior = take_gpu_fault().unwrap_or_default();
        return Err(format!("{what}: {err} {prior}"));
    }
    if let Some(prior) = take_gpu_fault() {
        return Err(format!("{what}: {prior}"));
    }
    Ok(())
}

fn mesh_param_bytes(words: [u32; 12]) -> [u8; 48] {
    let mut out = [0u8; 48];
    for (index, word) in words.into_iter().enumerate() {
        out[index * 4..index * 4 + 4].copy_from_slice(&word.to_le_bytes());
    }
    out
}

fn alloc_buffer(device: &wgpu::Device, size: u64) -> DropBuffer {
    DropBuffer(alloc_raw(device, size))
}

async fn record_pdhg(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buffers: &[DropBuffer],
    step: &wgpu::Buffer,
    clear: &[usize],
    nx: u32,
    m: u32,
    n_long: u32,
    n_jobs: u32,
    n_trust: u32,
) -> Result<(), String> {
    let passes: [(&str, Vec<usize>, [u32; 3]); 6] = [
        ("pdhg_aty", vec![0, 1, 2, 10, 12, 19], groups_for(nx)),
        (
            "pdhg_primal",
            vec![7, 12, 14, 15, 8, 9, 17, 19],
            groups_for(nx),
        ),
        ("pdhg_ax", vec![3, 4, 5, 8, 13, 19], groups_for(m)),
        (
            "pdhg_ax_long",
            // `step` is declared and unused. Tint drops it from the auto layout.
            vec![3, 4, 5, 8, 13, 6],
            [n_long.max(1), 1, 1],
        ),
        (
            "pdhg_dual",
            vec![10, 13, 16, 11, 18, 19],
            groups_for(n_jobs),
        ),
        ("pdhg_trust", vec![10, 13, 16, 11, 18, 19], [1, 1, 1]),
    ];
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    for &index in clear {
        encoder.clear_buffer(&buffers[index], 0, None);
    }
    let mut kept_groups = Vec::new();
    for (entry, bindings, groups) in passes {
        if entry == "pdhg_ax_long" && n_long == 0 {
            continue;
        }
        if entry == "pdhg_trust" && n_trust == 0 {
            continue;
        }
        if entry != "pdhg_ax_long"
            && entry != "pdhg_trust"
            && groups[0] == 1
            && matches!(entry, "pdhg_aty" | "pdhg_primal" | "pdhg_ax" | "pdhg_dual")
            && binding_count(entry, nx, m, n_jobs) == 0
        {
            continue;
        }
        let (_, _, pipeline) = pipeline("pdhg", entry).await?;
        let mut entries = Vec::new();
        for (slot, binding) in bindings.iter().enumerate() {
            let resource = if *binding == 19 {
                step.as_entire_binding()
            } else {
                buffers[*binding].as_entire_binding()
            };
            entries.push(wgpu::BindGroupEntry {
                binding: slot as u32,
                resource,
            });
        }
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(entry),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(groups[0], groups[1], groups[2]);
        }
        kept_groups.push(bind);
    }
    let _kept = kept_groups;
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    queue.submit([encoder.finish()]);
    if let Some(err) = scope.pop().await {
        return Err(format!("PDHG: {err}"));
    }
    Ok(())
}

fn binding_count(entry: &str, nx: u32, m: u32, n_jobs: u32) -> u32 {
    match entry {
        "pdhg_aty" | "pdhg_primal" => nx,
        "pdhg_ax" => m,
        "pdhg_dual" => n_jobs,
        _ => 1,
    }
}

fn mesh_load_payload(
    bin: &[u8],
    pos_count: u32,
    pos_offset: u32,
    pos_stride: u32,
    index_count: u32,
    index_offset: u32,
    index_stride: u32,
    index_width: u32,
    matrix: [f32; 16],
) -> Vec<u8> {
    let mut payload = Vec::new();
    push_u32(&mut payload, MAGIC);
    push_u32(&mut payload, KIND_MESH);
    push_u32(&mut payload, MESH_LOAD);
    for word in [
        pos_count,
        pos_offset,
        pos_stride,
        index_count,
        index_offset,
        index_stride,
        index_width,
    ] {
        push_u32(&mut payload, word);
    }
    let mut matrix_bytes = [0u8; 64];
    for (index, value) in matrix.into_iter().enumerate() {
        matrix_bytes[index * 4..index * 4 + 4].copy_from_slice(&value.to_le_bytes());
    }
    payload.extend_from_slice(&matrix_bytes);
    push_bytes(&mut payload, bin);
    payload
}

fn decode_mesh_load(bytes: &[u8]) -> Result<(Vec<u8>, Vec<u8>, u32, u32), String> {
    let mut reads = decode_reads(bytes)?;
    if reads.len() != 3 {
        return Err(format!("mesh load returned {} buffers", reads.len()));
    }
    let indices = reads.pop().unwrap();
    let positions = reads.pop().unwrap();
    let bad = reads.pop().unwrap();
    if bad.len() < 8 {
        return Err("mesh load status is short".into());
    }
    let nonfinite = u32::from_le_bytes(bad[0..4].try_into().unwrap());
    let bad_index = u32::from_le_bytes(bad[4..8].try_into().unwrap());
    Ok((positions, indices, nonfinite, bad_index))
}

pub fn dispatch_mesh_load(
    bin: &[u8],
    pos_count: u32,
    pos_offset: u32,
    pos_stride: u32,
    index_count: u32,
    index_offset: u32,
    index_stride: u32,
    index_width: u32,
    matrix: [f32; 16],
) -> Result<(Vec<u8>, Vec<u8>, u32, u32), String> {
    decode_mesh_load(&call_gpu(&mesh_load_payload(
        bin,
        pos_count,
        pos_offset,
        pos_stride,
        index_count,
        index_offset,
        index_stride,
        index_width,
        matrix,
    ))?)
}

/// Page-local mesh decode. The page owns the device, so this awaits it directly
/// instead of posting through the worker gate.
pub async fn mesh_load_local(
    bin: &[u8],
    pos_count: u32,
    pos_offset: u32,
    pos_stride: u32,
    index_count: u32,
    index_offset: u32,
    index_stride: u32,
    index_width: u32,
    matrix: [f32; 16],
) -> Result<(Vec<u8>, Vec<u8>, u32, u32), String> {
    decode_mesh_load(
        &execute_gpu(&mesh_load_payload(
            bin,
            pos_count,
            pos_offset,
            pos_stride,
            index_count,
            index_offset,
            index_stride,
            index_width,
            matrix,
        ))
        .await?,
    )
}

pub fn dispatch_mesh_align(
    vertices: &[u8],
    faces: &[u8],
    nvert: u32,
    nface: u32,
) -> Result<(u32, u32, u32, Vec<u8>), String> {
    let mut payload = Vec::new();
    push_u32(&mut payload, MAGIC);
    push_u32(&mut payload, KIND_MESH);
    push_u32(&mut payload, MESH_ALIGN);
    push_u32(&mut payload, nvert);
    push_u32(&mut payload, nface);
    push_bytes(&mut payload, vertices);
    push_bytes(&mut payload, faces);
    let mut reads = decode_reads(&call_gpu(&payload)?)?;
    if reads.len() != 2 {
        return Err(format!("mesh align returned {} buffers", reads.len()));
    }
    let moved = reads.pop().unwrap();
    let header = reads.pop().unwrap();
    if header.len() < 12 {
        return Err("mesh align status is short".into());
    }
    let passes = u32::from_le_bytes(header[0..4].try_into().unwrap());
    let bad = u32::from_le_bytes(header[4..8].try_into().unwrap());
    let overflow = u32::from_le_bytes(header[8..12].try_into().unwrap());
    Ok((passes, bad, overflow, moved))
}

pub fn dispatch_mesh_isect(
    vertices: &[u8],
    faces: &[u8],
    nvert: u32,
    nface: u32,
    scale_hi: f32,
    scale_lo: f32,
) -> Result<(u32, u32, u32, f32), String> {
    let mut payload = Vec::new();
    push_u32(&mut payload, MAGIC);
    push_u32(&mut payload, KIND_MESH);
    push_u32(&mut payload, MESH_ISECT);
    push_u32(&mut payload, nvert);
    push_u32(&mut payload, nface);
    push_u32(&mut payload, scale_hi.to_bits());
    push_u32(&mut payload, scale_lo.to_bits());
    push_bytes(&mut payload, vertices);
    push_bytes(&mut payload, faces);
    let reads = decode_reads(&call_gpu(&payload)?)?;
    if reads.len() != 1 || reads[0].len() < 16 {
        return Err("mesh intersection status is short".into());
    }
    let hit = u32::from_le_bytes(reads[0][0..4].try_into().unwrap());
    let lo = u32::from_le_bytes(reads[0][4..8].try_into().unwrap());
    let hi = u32::from_le_bytes(reads[0][8..12].try_into().unwrap());
    let gap = f32::from_le_bytes(reads[0][12..16].try_into().unwrap());
    Ok((hit, lo, hi, gap))
}

async fn run_mesh(bytes: &[u8]) -> Result<Vec<u8>, String> {
    if bytes.len() < 12 {
        return Err("mesh request is short".into());
    }
    let op = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
    match op {
        MESH_LOAD => run_mesh_load(bytes).await,
        MESH_ALIGN => run_mesh_align(bytes).await,
        MESH_ISECT => run_mesh_isect(bytes).await,
        other => Err(format!("unknown mesh request {other}")),
    }
}

async fn run_mesh_load(bytes: &[u8]) -> Result<Vec<u8>, String> {
    let mut cursor = Cursor { bytes, at: 12 };
    let pos_count = cursor.u32()?;
    let pos_offset = cursor.u32()?;
    let pos_stride = cursor.u32()?;
    let index_count = cursor.u32()?;
    let index_offset = cursor.u32()?;
    let index_stride = cursor.u32()?;
    let index_width = cursor.u32()?;
    if cursor.rest() < 64 {
        return Err("mesh load matrix is short".into());
    }
    let mut matrix = [0u8; 64];
    matrix.copy_from_slice(&cursor.bytes[cursor.at..cursor.at + 64]);
    cursor.at += 64;
    let bin_len = cursor.u32()? as usize;
    let bin = cursor.bytes(bin_len)?;
    let mut padded = bin.to_vec();
    padded.extend_from_slice(&[0u8; 8]);
    let mut header = [0u8; 32];
    for (index, word) in [
        pos_count,
        pos_offset,
        pos_stride,
        index_count,
        index_offset,
        index_stride,
        index_width,
        0,
    ]
    .into_iter()
    .enumerate()
    {
        header[index * 4..index * 4 + 4].copy_from_slice(&word.to_le_bytes());
    }
    let (device, queue, _) = pipeline("mesh", "mesh_load_pos").await?;
    let bin_b = alloc_buffer(&device, padded.len() as u64);
    let params = alloc_buffer(&device, 32);
    let matrix_b = alloc_buffer(&device, 64);
    let positions = alloc_buffer(&device, u64::from(pos_count) * 12);
    let indices = alloc_buffer(&device, u64::from(index_count) * 4);
    let bad = alloc_buffer(&device, 8);
    upload(&queue, &bin_b, &padded);
    upload(&queue, &params, &header);
    upload(&queue, &matrix_b, &matrix);
    upload(&queue, &bad, &[0u8; 8]);
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    let mut kept = Vec::new();
    record_compute(
        &device,
        &mut encoder,
        "mesh_load_pos",
        &[&bin_b, &params, &matrix_b, &positions, &bad],
        groups256(pos_count)?,
        &mut kept,
    )
    .await?;
    record_compute(
        &device,
        &mut encoder,
        "mesh_load_idx",
        &[&bin_b, &params, &indices, &bad],
        groups256(index_count)?,
        &mut kept,
    )
    .await?;
    let _kept = kept;
    finish_compute(&device, &queue, encoder, "mesh load").await?;
    let pos_bytes = readback(&device, &queue, &positions, pos_count as usize * 12).await?;
    let idx_bytes = readback(&device, &queue, &indices, index_count as usize * 4).await?;
    let bad_bytes = readback(&device, &queue, &bad, 8).await?;
    Ok(encode_reads(&[bad_bytes, pos_bytes, idx_bytes]))
}

async fn run_mesh_align(bytes: &[u8]) -> Result<Vec<u8>, String> {
    let mut cursor = Cursor { bytes, at: 12 };
    let nvert = cursor.u32()?;
    let nface = cursor.u32()?;
    let vert_len = cursor.u32()? as usize;
    let vertices = cursor.bytes(vert_len)?.to_vec();
    let face_len = cursor.u32()? as usize;
    let faces = cursor.bytes(face_len)?.to_vec();
    if nvert < 4
        || nface < 4
        || vertices.len() < nvert as usize * 24
        || faces.len() < nface as usize * 12
    {
        return Err("mesh align inputs are short".into());
    }
    let vert_groups = groups256(nvert)?;
    let face_groups = groups256(nface)?;
    let params_bytes = mesh_param_bytes([
        nvert,
        nface,
        0,
        0,
        0,
        0,
        vert_groups,
        face_groups,
        0,
        0,
        0,
        0,
    ]);
    let (device, queue, _) = pipeline("mesh", "mesh_zero_counts").await?;
    let pos = alloc_buffer(&device, vertices.len() as u64);
    let face_b = alloc_buffer(&device, faces.len() as u64);
    let params = alloc_buffer(&device, 48);
    let counts = alloc_buffer(&device, u64::from(nvert) * 4);
    let offsets = alloc_buffer(&device, u64::from(nvert + 1) * 4);
    let cursors = alloc_buffer(&device, u64::from(nvert) * 4);
    let face_ids = alloc_buffer(&device, u64::from(nface) * 12);
    let flags = alloc_buffer(&device, u64::from(nvert) * 4);
    let bad = alloc_buffer(&device, 8);
    upload(&queue, &pos, &vertices);
    upload(&queue, &face_b, &faces);
    upload(&queue, &params, &params_bytes);
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    let mut kept = Vec::new();
    record_compute(
        &device,
        &mut encoder,
        "mesh_zero_counts",
        &[&counts, &params],
        vert_groups,
        &mut kept,
    )
    .await?;
    record_compute(
        &device,
        &mut encoder,
        "mesh_align_count",
        &[&face_b, &counts, &params],
        face_groups,
        &mut kept,
    )
    .await?;
    record_compute(
        &device,
        &mut encoder,
        "mesh_align_scan",
        &[&counts, &offsets, &cursors, &params],
        1,
        &mut kept,
    )
    .await?;
    record_compute(
        &device,
        &mut encoder,
        "mesh_align_fill",
        &[&face_b, &cursors, &face_ids, &params],
        face_groups,
        &mut kept,
    )
    .await?;
    finish_compute(&device, &queue, encoder, "mesh align").await?;
    let _kept = kept;
    let mut passes = 0u32;
    let mut bad_count = 0u32;
    let mut overflow = 0u32;
    for _ in 0..120 {
        upload(&queue, &bad, &[0u8; 8]);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        let mut kept = Vec::new();
        record_compute(
            &device,
            &mut encoder,
            "mesh_zero_flags",
            &[&flags, &params],
            vert_groups,
            &mut kept,
        )
        .await?;
        record_compute(
            &device,
            &mut encoder,
            "mesh_align_test",
            &[&face_b, &pos, &offsets, &face_ids, &flags, &bad, &params],
            vert_groups,
            &mut kept,
        )
        .await?;
        finish_compute(&device, &queue, encoder, "mesh align").await?;
        let _kept = kept;
        let status = readback(&device, &queue, &bad, 8).await?;
        bad_count = u32::from_le_bytes(status[0..4].try_into().unwrap());
        overflow = u32::from_le_bytes(status[4..8].try_into().unwrap());
        if overflow != 0 {
            break;
        }
        if bad_count == 0 {
            break;
        }
        let chunk = 4096u32;
        let mut start = 0u32;
        while start < nvert {
            let count = chunk.min(nvert - start);
            upload(
                &queue,
                &params,
                &mesh_param_bytes([
                    nvert,
                    nface,
                    start,
                    count,
                    0,
                    0,
                    vert_groups,
                    face_groups,
                    0,
                    0,
                    0,
                    0,
                ]),
            );
            let mut encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            let mut kept = Vec::new();
            record_compute(
                &device,
                &mut encoder,
                "mesh_align_relax",
                &[&pos, &flags, &offsets, &face_ids, &face_b, &bad, &params],
                1,
                &mut kept,
            )
            .await?;
            finish_compute(&device, &queue, encoder, "mesh align").await?;
            let _kept = kept;
            start += count;
        }
        upload(&queue, &params, &params_bytes);
        let status = readback(&device, &queue, &bad, 8).await?;
        overflow = u32::from_le_bytes(status[4..8].try_into().unwrap());
        passes += 1;
        if overflow != 0 {
            break;
        }
    }
    let moved = if passes == 0 {
        Vec::new()
    } else {
        readback(&device, &queue, &pos, nvert as usize * 24).await?
    };
    let mut header = [0u8; 12];
    header[0..4].copy_from_slice(&passes.to_le_bytes());
    header[4..8].copy_from_slice(&bad_count.to_le_bytes());
    header[8..12].copy_from_slice(&overflow.to_le_bytes());
    Ok(encode_reads(&[header.to_vec(), moved]))
}

async fn run_mesh_isect(bytes: &[u8]) -> Result<Vec<u8>, String> {
    let mut cursor = Cursor { bytes, at: 12 };
    let nvert = cursor.u32()?;
    let nface = cursor.u32()?;
    let scale_bits = cursor.u32()?;
    let scale_lo_bits = cursor.u32()?;
    let vert_len = cursor.u32()? as usize;
    let vertices = cursor.bytes(vert_len)?.to_vec();
    let face_len = cursor.u32()? as usize;
    let faces = cursor.bytes(face_len)?.to_vec();
    if nface < 2 {
        return Ok(encode_reads(&[[0u8; 16].to_vec()]));
    }
    let cells = (nface as f64).cbrt().ceil().clamp(4.0, 128.0) as u32;
    let side = cells + 1;
    let vert_groups = groups256(nvert)?;
    let face_groups = groups256(nface)?;
    let cell_n = side.saturating_mul(side).saturating_mul(side);
    let params_bytes = mesh_param_bytes([
        nvert,
        nface,
        cells,
        0,
        side,
        scale_bits,
        vert_groups,
        face_groups,
        0,
        0,
        0,
        scale_lo_bits,
    ]);
    let (device, queue, _) = pipeline("mesh", "mesh_vert_partial").await?;
    let pos = alloc_buffer(&device, vertices.len() as u64);
    let face_b = alloc_buffer(&device, faces.len() as u64);
    let params = alloc_buffer(&device, 48);
    let partial = alloc_buffer(&device, u64::from(vert_groups) * 24);
    let scene = alloc_buffer(&device, 32);
    let tri_cells = alloc_buffer(&device, u64::from(nface) * 24);
    let cover = alloc_buffer(&device, 4);
    upload(&queue, &pos, &vertices);
    upload(&queue, &face_b, &faces);
    upload(&queue, &params, &params_bytes);
    upload(&queue, &cover, &[0u8; 4]);
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    let mut kept = Vec::new();
    record_compute(
        &device,
        &mut encoder,
        "mesh_vert_partial",
        &[&pos, &partial, &params],
        vert_groups,
        &mut kept,
    )
    .await?;
    record_compute(
        &device,
        &mut encoder,
        "mesh_scene",
        &[&partial, &scene, &params],
        1,
        &mut kept,
    )
    .await?;
    record_compute(
        &device,
        &mut encoder,
        "mesh_tri_cell",
        &[&face_b, &pos, &scene, &tri_cells, &cover, &params],
        face_groups,
        &mut kept,
    )
    .await?;
    finish_compute(&device, &queue, encoder, "mesh intersection").await?;
    let _kept = kept;
    let scene_bytes = readback(&device, &queue, &scene, 32).await?;
    let cover_bytes = readback(&device, &queue, &cover, 4).await?;
    let cover_n = u32::from_le_bytes(cover_bytes[0..4].try_into().unwrap());
    if cover_n == 0 || cover_n > 16_000_000 {
        return Err(format!("self-intersection grid covers {cover_n} cells"));
    }
    let origin = [
        f32::from_le_bytes(scene_bytes[0..4].try_into().unwrap()).to_bits(),
        f32::from_le_bytes(scene_bytes[4..8].try_into().unwrap()).to_bits(),
        f32::from_le_bytes(scene_bytes[8..12].try_into().unwrap()).to_bits(),
    ];
    let params_bytes = mesh_param_bytes([
        nvert,
        nface,
        cells,
        cover_n,
        side,
        scale_bits,
        vert_groups,
        face_groups,
        origin[0],
        origin[1],
        origin[2],
        scale_lo_bits,
    ]);
    upload(&queue, &params, &params_bytes);
    let cell_count = alloc_buffer(&device, u64::from(cell_n) * 4);
    let cell_start = alloc_buffer(&device, u64::from(cell_n) * 4);
    let cursor = alloc_buffer(&device, u64::from(cell_n) * 4);
    let ids = alloc_buffer(&device, u64::from(cover_n) * 4);
    let hits = alloc_buffer(&device, u64::from(nface) * 8);
    let best = alloc_buffer(&device, 16);
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    let mut kept = Vec::new();
    record_compute(
        &device,
        &mut encoder,
        "mesh_zero_cells",
        &[&cell_count, &params],
        groups256(cell_n)?,
        &mut kept,
    )
    .await?;
    record_compute(
        &device,
        &mut encoder,
        "mesh_cell_count",
        &[&tri_cells, &cell_count, &params],
        face_groups,
        &mut kept,
    )
    .await?;
    record_compute(
        &device,
        &mut encoder,
        "mesh_cell_scan",
        &[&cell_count, &cell_start, &cursor, &params],
        1,
        &mut kept,
    )
    .await?;
    record_compute(
        &device,
        &mut encoder,
        "mesh_cell_fill",
        &[&tri_cells, &cursor, &ids, &params],
        face_groups,
        &mut kept,
    )
    .await?;
    record_compute(
        &device,
        &mut encoder,
        "mesh_isect",
        &[
            &face_b,
            &pos,
            &tri_cells,
            &cell_start,
            &cell_count,
            &ids,
            &hits,
            &params,
        ],
        face_groups,
        &mut kept,
    )
    .await?;
    record_compute(
        &device,
        &mut encoder,
        "mesh_best",
        &[&hits, &best, &params],
        1,
        &mut kept,
    )
    .await?;
    finish_compute(&device, &queue, encoder, "mesh intersection").await?;
    let _kept = kept;
    let best_bytes = readback(&device, &queue, &best, 16).await?;
    Ok(encode_reads(&[best_bytes]))
}

/// Page entry. `request` is the worker's payload.
pub async fn execute_gpu(request: &[u8]) -> Result<Vec<u8>, String> {
    if request.len() < 8 {
        return Err("empty gpu request".into());
    }
    let magic = u32::from_le_bytes(request[0..4].try_into().unwrap());
    let kind = u32::from_le_bytes(request[4..8].try_into().unwrap());
    if magic != MAGIC {
        return Err("gpu request has the wrong header".into());
    }
    gpu().await?;
    match kind {
        KIND_SINGLE => run_single(request).await,
        KIND_PDHG => run_pdhg(request).await,
        KIND_LBVH => run_lbvh(request).await,
        KIND_MESH => run_mesh(request).await,
        KIND_DROP => {
            if request.len() < 12 {
                return Err("gpu buffer release is short".into());
            }
            let id = u32::from_le_bytes(request[8..12].try_into().unwrap());
            GPU.with(|slot| {
                if let Some(gpu) = slot.borrow_mut().as_mut() {
                    gpu.residents.remove(&id);
                }
            });
            Ok(Vec::new())
        }
        other => Err(format!("unknown gpu request {other}")),
    }
}
