//! wgpu executor of [`Op`] lists: the desktop device, and the page device in the browser.

use std::collections::HashMap;

use super::kernels::ALL;
use super::{BindKind, Code, Kernel, Op, Ref, Variant};

enum AccelObj {
    Hw {
        tlas: wgpu::Tlas,
        _blas: wgpu::Blas,
        _bufs: [wgpu::Buffer; 2],
    },
    Sw {
        nodes: wgpu::Buffer,
        tris: wgpu::Buffer,
    },
}

pub struct Exec {
    device: wgpu::Device,
    queue: wgpu::Queue,
    hw_rt: bool,
    pipes: Vec<Option<(wgpu::ComputePipeline, wgpu::BindGroupLayout)>>,
    res: HashMap<u64, wgpu::Buffer>,
    accels: HashMap<u64, AccelObj>,
    /// Bind groups whose arguments are all resident, by kernel and arguments.
    groups: HashMap<(usize, Vec<Ref>), wgpu::BindGroup>,
    fault: std::sync::Arc<std::sync::Mutex<Option<String>>>,
    /// Job-local and read-back buffers kept between jobs, by (capacity, usage).
    pool: HashMap<(u64, u32), Vec<wgpu::Buffer>>,
    pub adapter: String,
}

/// Buffers a size class keeps for reuse; more are destroyed.
const POOL_PER_CLASS: usize = 4;
const READBACK: wgpu::BufferUsages =
    wgpu::BufferUsages::MAP_READ.union(wgpu::BufferUsages::COPY_DST);

/// A job-local buffer from the pool and the bytes the job uses of it.
struct Local {
    buf: wgpu::Buffer,
    size: u64,
}

const STORAGE: wgpu::BufferUsages = wgpu::BufferUsages::STORAGE
    .union(wgpu::BufferUsages::UNIFORM)
    .union(wgpu::BufferUsages::COPY_SRC)
    .union(wgpu::BufferUsages::COPY_DST);

/// Buffer sizes are rounded to 16 bytes: a uniform binding needs it, and a
/// zero-sized binding is invalid.
fn padded(size: u64) -> u64 {
    size.max(16).div_ceil(16) * 16
}

impl Exec {
    /// The desktop device: Vulkan on Linux and Windows, Metal on macOS.
    #[cfg(not(target_arch = "wasm32"))]
    pub async fn native(want_hw_rt: bool) -> Result<Self, String> {
        let backends = if cfg!(target_os = "macos") {
            wgpu::Backends::METAL
        } else {
            wgpu::Backends::VULKAN
        };
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .map_err(|e| format!("no Vulkan/Metal adapter: {e}"))?;
        let have = adapter.features();
        if !have.contains(wgpu::Features::PASSTHROUGH_SHADERS) {
            return Err("the adapter does not accept passthrough shaders".into());
        }
        let hw_rt = want_hw_rt && have.contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY);
        let mut features = wgpu::Features::PASSTHROUGH_SHADERS;
        if hw_rt {
            features |= wgpu::Features::EXPERIMENTAL_RAY_QUERY;
        }
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("astroid_evolution"),
                required_features: features,
                required_limits: adapter.limits(),
                experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
                ..Default::default()
            })
            .await
            .map_err(|e| format!("device: {e}"))?;
        let info = adapter.get_info();
        Ok(Self::from_device(
            device,
            queue,
            hw_rt,
            format!(
                "{} ({:?}, {})",
                info.name,
                info.backend,
                if hw_rt { "ray queries" } else { "software BVH" }
            ),
        ))
    }

    pub fn from_device(
        device: wgpu::Device,
        queue: wgpu::Queue,
        hw_rt: bool,
        adapter: String,
    ) -> Self {
        // An error outside a scope is reported by the next job instead of aborting.
        let fault = std::sync::Arc::new(std::sync::Mutex::new(None::<String>));
        let sink = fault.clone();
        device.on_uncaptured_error(std::sync::Arc::new(move |e: wgpu::Error| {
            *sink.lock().unwrap_or_else(|p| p.into_inner()) = Some(e.to_string());
        }));
        Self {
            fault,
            device,
            queue,
            hw_rt,
            pipes: (0..ALL.len()).map(|_| None).collect(),
            res: HashMap::new(),
            accels: HashMap::new(),
            groups: HashMap::new(),
            pool: HashMap::new(),
            adapter,
        }
    }

    pub fn hw_rt(&self) -> bool {
        self.hw_rt
    }

    fn variant(&self, k: &Kernel) -> &'static Variant {
        let k: &'static Kernel = ALL[k.id];
        if self.hw_rt && k.traces { &k.hw } else { &k.sw }
    }

    async fn pipeline(&mut self, id: usize) -> Result<(), String> {
        if self.pipes[id].is_some() {
            return Ok(());
        }
        let k = ALL[id];
        let v = self.variant(k);
        #[cfg(target_arch = "wasm32")]
        if let Code::Wgsl(src) = v.code {
            warm(&self.device, src, k.name).await?;
        }
        let passthrough = !matches!(v.code, Code::Wgsl(_));
        let entries: Vec<wgpu::BindGroupLayoutEntry> = v
            .binds
            .iter()
            .enumerate()
            .map(|(dense, b)| wgpu::BindGroupLayoutEntry {
                binding: if passthrough { dense as u32 } else { b.binding },
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: match b.kind {
                    BindKind::Accel => wgpu::BindingType::AccelerationStructure {
                        vertex_return: false,
                    },
                    BindKind::Uniform => wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    kind => wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage {
                            read_only: kind != BindKind::Write,
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                },
                count: None,
            })
            .collect();
        let bgl = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(k.name),
                entries: &entries,
            });
        let layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(k.name),
                bind_group_layouts: &[Some(&bgl)],
                immediate_size: 0,
            });
        let entry_name = match v.code {
            Code::Msl(_, f) => f,
            _ => k.name,
        };
        let scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = match v.code {
            Code::Wgsl(src) => self
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(k.name),
                    source: wgpu::ShaderSource::Wgsl(src.into()),
                }),
            code => {
                let entry = [wgpu::PassthroughShaderEntryPoint {
                    name: entry_name.into(),
                    workgroup_size: (k.wg[0], k.wg[1], k.wg[2]),
                }];
                let mut desc = wgpu::ShaderModuleDescriptorPassthrough {
                    label: Some(k.name),
                    entry_points: std::borrow::Cow::Borrowed(&entry),
                    ..Default::default()
                };
                match code {
                    Code::Spirv(bytes) => desc.spirv = Some(wgpu::util::make_spirv_raw(bytes)),
                    Code::Msl(src, _) => desc.msl = Some(src.into()),
                    Code::Wgsl(_) => unreachable!(),
                }
                // SAFETY: the module was generated and validated by naga in build.rs
                // for exactly this binding layout.
                unsafe { self.device.create_shader_module_passthrough(desc) }
            }
        };
        let pipe = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(k.name),
                layout: Some(&layout),
                module: &module,
                entry_point: Some(entry_name),
                compilation_options: Default::default(),
                cache: None,
            });
        drop(module);
        if let Some(e) = scope.pop().await {
            return Err(format!("pipeline {}: {e}", k.name));
        }
        self.pipes[id] = Some((pipe, bgl));
        Ok(())
    }

    /// A buffer of at least `size` bytes: reused from the pool, or created at the
    /// next power of two so that later jobs of a similar size reuse it.
    fn take(&mut self, size: u64, usage: wgpu::BufferUsages) -> wgpu::Buffer {
        let cap = padded(size).next_power_of_two().max(256);
        if let Some(b) = self.pool.get_mut(&(cap, usage.bits())).and_then(Vec::pop) {
            return b;
        }
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pooled"),
            size: cap,
            usage,
            mapped_at_creation: false,
        })
    }

    fn give(&mut self, buf: wgpu::Buffer) {
        let class = self.pool.entry((buf.size(), buf.usage().bits())).or_default();
        if class.len() < POOL_PER_CLASS {
            class.push(buf);
        } else {
            buf.destroy();
        }
    }

    fn storage(&self, size: u64, label: &str) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: padded(size),
            usage: STORAGE,
            mapped_at_creation: false,
        })
    }

    fn filled(&self, data: &[u8], usage: wgpu::BufferUsages, label: &str) -> wgpu::Buffer {
        let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: padded(data.len() as u64),
            usage,
            mapped_at_creation: true,
        });
        buf.slice(..data.len() as u64)
            .get_mapped_range_mut()
            .expect("a buffer mapped at creation")
            .copy_from_slice(data);
        buf.unmap();
        buf
    }

    fn build_accel(
        &mut self,
        enc: &mut wgpu::CommandEncoder,
        id: u64,
        verts: &[f32],
        tris: &[u32],
        bvh: Option<(Vec<u8>, Vec<u8>)>,
    ) -> Result<(), String> {
        if let Some((nodes, t)) = bvh {
            let nodes = self.filled(&nodes, STORAGE, "bvh nodes");
            let tris = self.filled(&t, STORAGE, "bvh triangles");
            self.accels.insert(id, AccelObj::Sw { nodes, tris });
            return Ok(());
        }
        if !self.hw_rt {
            return Err("software ray tracing needs the host-built BVH".into());
        }
        let vb = self.filled(
            bytemuck::cast_slice(verts),
            wgpu::BufferUsages::BLAS_INPUT,
            "blas vertices",
        );
        let ib = self.filled(
            bytemuck::cast_slice(tris),
            wgpu::BufferUsages::BLAS_INPUT,
            "blas indices",
        );
        let size = wgpu::BlasTriangleGeometrySizeDescriptor {
            vertex_format: wgpu::VertexFormat::Float32x3,
            vertex_count: (verts.len() / 3) as u32,
            index_format: Some(wgpu::IndexFormat::Uint32),
            index_count: Some(tris.len() as u32),
            // Not opaque: one query visits every crossing as a candidate. No
            // duplicates: each triangle is reported once.
            flags: wgpu::AccelerationStructureGeometryFlags::NO_DUPLICATE_ANY_HIT_INVOCATION,
        };
        let blas = self.device.create_blas(
            &wgpu::CreateBlasDescriptor {
                label: Some("body"),
                flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE
                    | wgpu::AccelerationStructureFlags::ALLOW_UPDATE,
                update_mode: wgpu::AccelerationStructureUpdateMode::Build,
            },
            wgpu::BlasGeometrySizeDescriptors::Triangles {
                descriptors: vec![size.clone()],
            },
        );
        let mut tlas = self.device.create_tlas(&wgpu::CreateTlasDescriptor {
            label: Some("body"),
            max_instances: 1,
            flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: wgpu::AccelerationStructureUpdateMode::Build,
        });
        tlas[0] = Some(wgpu::TlasInstance::new(
            &blas,
            [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            0,
            0xff,
        ));
        enc.build_acceleration_structures(
            std::iter::once(&wgpu::BlasBuildEntry {
                blas: &blas,
                geometry: wgpu::BlasGeometries::TriangleGeometries(vec![
                    wgpu::BlasTriangleGeometry {
                        size: &size,
                        vertex_buffer: &vb,
                        first_vertex: 0,
                        vertex_stride: 12,
                        index_buffer: Some(&ib),
                        first_index: Some(0),
                        transform_buffer: None,
                        transform_buffer_offset: None,
                    },
                ]),
            }),
            std::iter::once(&tlas),
        );
        self.accels.insert(
            id,
            AccelObj::Hw {
                tlas,
                _blas: blas,
                _bufs: [vb, ib],
            },
        );
        Ok(())
    }

    /// Run one job. Resolves once every read-back is mapped.
    pub async fn run(&mut self, ops: Vec<Op>) -> Result<Vec<Vec<u8>>, String> {
        // Validation and out-of-memory errors reach `fault` through the
        // uncaptured-error handler, so a job pushes no error scopes.
        let mut locals: HashMap<u32, Local> = HashMap::new();
        let mut staging: Vec<(wgpu::Buffer, u64)> = Vec::new();
        for op in &ops {
            if let Op::Run { kernel, .. } = op {
                self.pipeline(*kernel).await?;
            }
        }
        let mut enc = self.device.create_command_encoder(&Default::default());
        let mut pass: Option<wgpu::ComputePass<'static>> = None;
        for op in ops {
            if !matches!(op, Op::Run { .. }) {
                pass = None;
            }
            match op {
                Op::NewRes { id, size, data } => {
                    let buf = match data {
                        Some(d) => self.filled(&d, STORAGE, "resident"),
                        None => self.storage(size, "resident"),
                    };
                    self.res.insert(id, buf);
                }
                Op::FreeRes { id } => {
                    // The browser backend does not free a buffer on drop.
                    if let Some(b) = self.res.remove(&id) {
                        b.destroy();
                    }
                    self.groups
                        .retain(|(_, args), _| !args.contains(&Ref::Res(id)));
                }
                Op::NewAccel {
                    id,
                    verts,
                    tris,
                    bvh,
                } => self.build_accel(&mut enc, id, &verts, &tris, bvh)?,
                Op::FreeAccel { id } => {
                    self.accels.remove(&id);
                    self.groups
                        .retain(|(_, args), _| !args.contains(&Ref::Accel(id)));
                }
                Op::Local { id, size, data } => {
                    let buf = self.take(size, STORAGE);
                    match data {
                        Some(mut d) => {
                            d.resize(d.len().div_ceil(4) * 4, 0);
                            self.queue.write_buffer(&buf, 0, &d);
                        }
                        None => enc.clear_buffer(&buf, 0, Some(padded(size))),
                    }
                    locals.insert(id, Local { buf, size: padded(size) });
                }
                Op::Write { dst, offset, data } => {
                    let src = self.filled(&data, wgpu::BufferUsages::COPY_SRC, "write");
                    let dst = buffer(&locals, &self.res, dst)?;
                    enc.copy_buffer_to_buffer(&src, 0, dst, offset, Some(data.len() as u64));
                }
                Op::Copy {
                    src,
                    src_offset,
                    dst,
                    dst_offset,
                    size,
                } => {
                    let s = buffer(&locals, &self.res, src)?;
                    let d = buffer(&locals, &self.res, dst)?;
                    enc.copy_buffer_to_buffer(s, src_offset, d, dst_offset, Some(size));
                }
                Op::Read { src, offset, size } => {
                    let st = self.take(size.max(4).div_ceil(4) * 4, READBACK);
                    let s = buffer(&locals, &self.res, src)?;
                    if size > 0 {
                        enc.copy_buffer_to_buffer(s, offset, &st, 0, Some(size.div_ceil(4) * 4));
                    }
                    staging.push((st, size));
                }
                Op::Run {
                    kernel,
                    groups,
                    args,
                } => {
                    let resident = args.iter().all(|a| !matches!(a, Ref::Local(_)));
                    let key = (kernel, args);
                    let bg = match self.groups.get(&key) {
                        Some(bg) => bg.clone(),
                        None => {
                            let bg = self.bind_group(kernel, &key.1, &locals)?;
                            if resident {
                                self.groups.insert(key, bg.clone());
                            }
                            bg
                        }
                    };
                    // Consecutive dispatches share one pass; wgpu orders them by usage.
                    let pass = pass.get_or_insert_with(|| {
                        enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: None,
                            timestamp_writes: None,
                        })
                        .forget_lifetime()
                    });
                    pass.set_pipeline(&self.pipes[kernel].as_ref().unwrap().0);
                    pass.set_bind_group(0, &bg, &[]);
                    pass.dispatch_workgroups(groups[0], groups[1], groups[2]);
                }
            }
        }
        drop(pass);
        self.queue.submit([enc.finish()]);
        let mut waits = Vec::with_capacity(staging.len());
        for (st, _) in &staging {
            let (tx, rx) = futures_channel::oneshot::channel();
            st.slice(..).map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
            waits.push(rx);
        }
        #[cfg(not(target_arch = "wasm32"))]
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| format!("device poll: {e}"))?;
        if let Some(e) = self.fault.lock().unwrap_or_else(|p| p.into_inner()).take() {
            return Err(format!("gpu: {e}"));
        }
        let mut out = Vec::with_capacity(staging.len());
        for ((st, size), rx) in staging.into_iter().zip(waits) {
            rx.await
                .map_err(|_| "gpu read-back was cancelled".to_string())?
                .map_err(|e| format!("gpu read-back: {e}"))?;
            let view = st
                .slice(..)
                .get_mapped_range()
                .map_err(|e| format!("gpu read-back: {e}"))?;
            out.push(view[..size as usize].to_vec());
            drop(view);
            st.unmap();
            self.give(st);
        }
        for (_, l) in locals {
            self.give(l.buf);
        }
        Ok(out)
    }
}

fn buffer<'a>(
    locals: &'a HashMap<u32, Local>,
    res: &'a HashMap<u64, wgpu::Buffer>,
    r: Ref,
) -> Result<&'a wgpu::Buffer, String> {
    match r {
        Ref::Local(id) => locals
            .get(&id)
            .map(|l| &l.buf)
            .ok_or_else(|| format!("no local buffer {id}")),
        Ref::Res(id) => res
            .get(&id)
            .ok_or_else(|| format!("no resident buffer {id}")),
        Ref::Accel(_) => Err("a mesh is not a buffer".into()),
    }
}

impl Exec {
    /// The bind group of `kernel` over `args` (BVH arguments bind nodes and triangles).
    fn bind_group(
        &self,
        kernel: usize,
        args: &[Ref],
        locals: &HashMap<u32, Local>,
    ) -> Result<wgpu::BindGroup, String> {
        let k = ALL[kernel];
        let v = self.variant(k);
        let passthrough = !matches!(v.code, Code::Wgsl(_));
        let mut entries = Vec::with_capacity(v.binds.len());
        let mut next = 0usize;
        let mut bvh_second = false;
        for (dense, b) in v.binds.iter().enumerate() {
            let binding = if passthrough { dense as u32 } else { b.binding };
            let arg = if b.kind == BindKind::Bvh && bvh_second {
                args[next - 1]
            } else {
                let a = *args
                    .get(next)
                    .ok_or_else(|| format!("{}: missing argument {next}", k.name))?;
                next += 1;
                a
            };
            let resource = match (b.kind, arg) {
                (BindKind::Accel, Ref::Accel(id)) => match self.accels.get(&id) {
                    Some(AccelObj::Hw { tlas, .. }) => tlas.as_binding(),
                    _ => return Err(format!("{}: no hardware acceleration structure", k.name)),
                },
                (BindKind::Bvh, Ref::Accel(id)) => match self.accels.get(&id) {
                    Some(AccelObj::Sw { nodes, tris }) => {
                        if bvh_second {
                            tris.as_entire_binding()
                        } else {
                            nodes.as_entire_binding()
                        }
                    }
                    _ => return Err(format!("{}: no software BVH", k.name)),
                },
                (BindKind::Accel | BindKind::Bvh, _) => {
                    return Err(format!("{}: argument {} must be a mesh", k.name, next - 1));
                }
                // A pooled local is larger than the job's data: bind its own bytes.
                (_, Ref::Local(id)) => {
                    let l = locals
                        .get(&id)
                        .ok_or_else(|| format!("no local buffer {id}"))?;
                    wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &l.buf,
                        offset: 0,
                        size: std::num::NonZeroU64::new(l.size),
                    })
                }
                (_, r) => buffer(locals, &self.res, r)?.as_entire_binding(),
            };
            bvh_second = b.kind == BindKind::Bvh && !bvh_second;
            entries.push(wgpu::BindGroupEntry { binding, resource });
        }
        if next != args.len() {
            return Err(format!(
                "{}: {} arguments for {next} bindings",
                k.name,
                args.len()
            ));
        }
        let (_, bgl) = self.pipes[kernel].as_ref().unwrap();
        Ok(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(k.name),
            layout: bgl,
            entries: &entries,
        }))
    }
}

/// Dawn may lose the device when `createComputePipeline` compiles a large kernel
/// synchronously. Compiling the same module through `createComputePipelineAsync`
/// first makes the synchronous call a cache hit.
#[cfg(target_arch = "wasm32")]
async fn warm(device: &wgpu::Device, wgsl: &str, entry: &str) -> Result<(), String> {
    use wasm_bindgen::{JsCast, JsValue};
    let text = |e: JsValue| e.as_string().unwrap_or_else(|| format!("{e:?}"));
    let set = |o: &js_sys::Object, k: &str, v: &JsValue| {
        js_sys::Reflect::set(o, &JsValue::from_str(k), v)
            .map(|_| ())
            .map_err(text)
    };
    let call = |name: &str, target: &JsValue, arg: &JsValue| -> Result<JsValue, String> {
        js_sys::Reflect::get(target, &JsValue::from_str(name))
            .map_err(text)?
            .dyn_into::<js_sys::Function>()
            .map_err(|_| format!("WebGPU {name} is missing"))?
            .call1(target, arg)
            .map_err(text)
    };
    let gpu = device.as_webgpu().ok_or("WebGPU device is not ready")?;
    let dev: &JsValue = gpu.as_ref();
    let md = js_sys::Object::new();
    set(&md, "code", &JsValue::from_str(wgsl))?;
    let module = call("createShaderModule", dev, &md)?;
    let stage = js_sys::Object::new();
    set(&stage, "module", &module)?;
    set(&stage, "entryPoint", &JsValue::from_str(entry))?;
    let desc = js_sys::Object::new();
    set(&desc, "layout", &JsValue::from_str("auto"))?;
    set(&desc, "compute", &stage)?;
    let promise = call("createComputePipelineAsync", dev, &desc)?
        .dyn_into::<js_sys::Promise>()
        .map_err(|_| "createComputePipelineAsync did not return a promise".to_string())?;
    wasm_bindgen_futures::JsFuture::from(promise)
        .await
        .map(|_| ())
        .map_err(|e| format!("pipeline {entry}: {}", text(e)))
}
