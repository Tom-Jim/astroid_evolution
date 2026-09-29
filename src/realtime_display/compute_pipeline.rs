//! 实时 GPU 顶点与法向计算管道 (Compute Pass)
//! 依照 docs/software.md 第 13 节
//! Compute Shader 直接修改 R(theta, phi)、surface_position 与 surface_normal

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::rtfp::gpu_dispatch::GpuContext;

const SURFACE_DEFORM_SHADER: &str = r#"
// surface_deform.wgsl
// GPU 实时半径重塑与网格顶点更新

struct DeformGlobals {
    vertex_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> g: DeformGlobals;
// 基础球面方向: vec4<f32>(u_x, u_y, u_z, 0.0)
@group(0) @binding(1) var<storage, read> directions: array<vec4<f32>>;
// 径向模态极半径: array<f32>
@group(0) @binding(2) var<storage, read> radii: array<f32>;
// 输出更新后的顶点坐标: vec4<f32>(p_x, p_y, p_z, 1.0)
@group(0) @binding(3) var<storage, read_write> updated_positions: array<vec4<f32>>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= g.vertex_count) {
        return;
    }

    let u = directions[idx].xyz;
    let r = radii[idx];
    let pos = u * r;

    updated_positions[idx] = vec4<f32>(pos, 1.0);
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DeformGlobals {
    vertex_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// GPU 表面形状动态更新器
pub struct GpuSurfaceUpdater {
    pub pipeline: wgpu::ComputePipeline,
}

impl GpuSurfaceUpdater {
    pub fn new(ctx: &GpuContext) -> Result<Self, String> {
        let module = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("surface_deform_module"),
            source: wgpu::ShaderSource::Wgsl(SURFACE_DEFORM_SHADER.into()),
        });

        let pipeline = ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("surface_deform_pipeline"),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Ok(Self { pipeline })
    }

    /// 在 GPU 上执行实时顶点更新，根据极半径 R 与基方向 u 产生新三维坐标集合
    pub fn update_surface_positions_gpu(
        &self,
        ctx: &GpuContext,
        directions: &[[f32; 3]],
        radii: &[f32],
    ) -> Result<Vec<[f32; 3]>, String> {
        let n = directions.len();
        if n == 0 || n != radii.len() {
            return Err("方向数与半径数不匹配".to_string());
        }

        let dev = &ctx.device;
        let queue = &ctx.queue;

        let globals = DeformGlobals {
            vertex_count: n as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };

        let dirs_data: Vec<[f32; 4]> = directions.iter().map(|d| [d[0], d[1], d[2], 0.0]).collect();

        let globals_buf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("deform_globals"),
            contents: bytemuck::bytes_of(&globals),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let dirs_buf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("deform_dirs"),
            contents: bytemuck::cast_slice(&dirs_data),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let radii_buf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("deform_radii"),
            contents: bytemuck::cast_slice(radii),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let out_buf = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("deform_out"),
            size: (n * 4 * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let readback_buf = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("deform_readback"),
            size: (n * 4 * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("deform_bg"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: globals_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: dirs_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: radii_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: out_buf.as_entire_binding() },
            ],
        });

        let mut encoder = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("deform_encoder"),
        });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("deform_cpass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            let wgs = ((n as u32) + 63) / 64;
            cpass.dispatch_workgroups(wgs, 1, 1);
        }

        encoder.copy_buffer_to_buffer(
            &out_buf,
            0,
            &readback_buf,
            0,
            (n * 4 * std::mem::size_of::<f32>()) as u64,
        );

        queue.submit(Some(encoder.finish()));

        // 回读数据
        let slice = readback_buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            tx.send(res).unwrap();
        });

        dev.poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| format!("Deform GPU poll 失败: {e:?}"))?;
        rx.recv()
            .map_err(|e| format!("接收失败: {e}"))?
            .map_err(|e| format!("映射失败: {e:?}"))?;

        let view = slice.get_mapped_range().map_err(|e| format!("{e:?}"))?;
        let raw: &[[f32; 4]] = bytemuck::cast_slice(&view);

        let positions: Vec<[f32; 3]> = raw.iter().map(|p| [p[0], p[1], p[2]]).collect();

        drop(view);
        readback_buf.unmap();

        Ok(positions)
    }
}
