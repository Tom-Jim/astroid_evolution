//! wgpu / WGSL 6 阶段融合 GPU 调度管道
//! 严格依照 docs/software.md 第 2 / 12 / 15 节
//!
//! 调度流程:
//!   Pass 1: ray_count.wgsl     (光线穿透计数)
//!   Pass 2: prefix_sum.wgsl    (前缀和全局紧凑偏移)
//!   Pass 3: ray_write.wgsl     (BVH 遍历与交点排序写入)
//!   Pass 4: segment_build.wgsl (根据 inside_at_zero 构造实体交段)
//!   Pass 5: rtfp_integrate.wgsl(融合求解 I_U, I_g, I_H)
//!   Pass 6: reduce_field.wgsl  (分层 workgroup 树状归约得到 U, g, H6, Tr)

use std::sync::Arc;
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use super::bvh::FlattenedBvh;
use super::mesh::Mesh;
use super::quadrature;
use super::ray_backend::{RayBackend, RayBatch, RayHitBuffer};
use crate::model_calculation::contracts::{DensityKernel, FieldBranch, FieldSample, KernelGpu};

const RAY_COUNT_SHADER: &str = include_str!("shaders/ray_count.wgsl");
const PREFIX_SUM_SHADER: &str = include_str!("shaders/prefix_sum.wgsl");
const RAY_WRITE_SHADER: &str = include_str!("shaders/ray_write.wgsl");
const SEGMENT_BUILD_SHADER: &str = include_str!("shaders/segment_build.wgsl");
const RTFP_INTEGRATE_SHADER: &str = include_str!("shaders/rtfp_integrate.wgsl");
const REDUCE_FIELD_SHADER: &str = include_str!("shaders/reduce_field.wgsl");
const NEAR_FIELD_SHADER: &str = include_str!("shaders/near_field.wgsl");

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuGlobals {
    n_points: u32,
    n_dirs: u32,
    n_tris: u32,
    n_kernels: u32,
    t_min: f32,
    t_max: f32,
    g_si: f32,
    r0: f32,
    ell0: f32,
    bulk_density: f32,
    _pad: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PrefixGlobals {
    count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// GPU 运行时上下文与预编译管线
pub struct GpuContext {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub ray_count_pipeline: wgpu::ComputePipeline,
    pub prefix_sum_pipeline: wgpu::ComputePipeline,
    pub ray_write_pipeline: wgpu::ComputePipeline,
    pub segment_build_pipeline: wgpu::ComputePipeline,
    pub rtfp_integrate_pipeline: wgpu::ComputePipeline,
    pub reduce_field_pipeline: wgpu::ComputePipeline,
    pub near_field_pipeline: wgpu::ComputePipeline,
}

impl GpuContext {
    /// 初始化 wgpu 设备并编译全部 6 个计算着色器管线
    pub fn new() -> Result<Self, String> {
        Self::new_with_fallback(false)
    }

    pub fn new_with_fallback(force_fallback_adapter: bool) -> Result<Self, String> {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter,
            compatible_surface: None,
            ..Default::default()
        }))
        .map_err(|e| format!("未找到适用的 GPU 适配器 (wgpu adapter): {e:?}"))?;

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("rtfp_gpu_context"),
            ..Default::default()
        }))
        .map_err(|e| format!("请求 wgpu 设备失败: {e}"))?;

        let device = Arc::new(device);
        let queue = Arc::new(queue);

        // 编译 WGSL 着色器模块
        let ray_count_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ray_count_module"),
            source: wgpu::ShaderSource::Wgsl(RAY_COUNT_SHADER.into()),
        });
        let prefix_sum_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("prefix_sum_module"),
            source: wgpu::ShaderSource::Wgsl(PREFIX_SUM_SHADER.into()),
        });
        let ray_write_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ray_write_module"),
            source: wgpu::ShaderSource::Wgsl(RAY_WRITE_SHADER.into()),
        });
        let segment_build_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("segment_build_module"),
            source: wgpu::ShaderSource::Wgsl(SEGMENT_BUILD_SHADER.into()),
        });
        let rtfp_integrate_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rtfp_integrate_module"),
            source: wgpu::ShaderSource::Wgsl(RTFP_INTEGRATE_SHADER.into()),
        });
        let reduce_field_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("reduce_field_module"),
            source: wgpu::ShaderSource::Wgsl(REDUCE_FIELD_SHADER.into()),
        });

        // 创建计算管线
        let ray_count_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ray_count_pipeline"),
            layout: None,
            module: &ray_count_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let prefix_sum_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("prefix_sum_pipeline"),
            layout: None,
            module: &prefix_sum_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let ray_write_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ray_write_pipeline"),
            layout: None,
            module: &ray_write_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let segment_build_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("segment_build_pipeline"),
            layout: None,
            module: &segment_build_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let rtfp_integrate_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("rtfp_integrate_pipeline"),
            layout: None,
            module: &rtfp_integrate_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let reduce_field_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("reduce_field_pipeline"),
            layout: None,
            module: &reduce_field_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let near_field_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("near_field_module"),
            source: wgpu::ShaderSource::Wgsl(NEAR_FIELD_SHADER.into()),
        });

        let near_field_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("near_field_pipeline"),
            layout: None,
            module: &near_field_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Ok(Self {
            device,
            queue,
            ray_count_pipeline,
            prefix_sum_pipeline,
            ray_write_pipeline,
            segment_build_pipeline,
            rtfp_integrate_pipeline,
            reduce_field_pipeline,
            near_field_pipeline,
        })
    }
}

/// 默认可移植的 GPU BVH WGSL 计算后端 (docs/software.md 第 2 节)
pub struct GpuBvhWgslBackend<'a> {
    pub context: &'a GpuContext,
    pub mesh: &'a Mesh,
    pub bvh: &'a FlattenedBvh,
    pub kernels: &'a [DensityKernel],
    pub dirs: Vec<([f64; 3], f64)>,
    pub near_field_r0: f32,
    pub bulk_density: f32,
    pub finite_part_scale: f32,
}

impl<'a> GpuBvhWgslBackend<'a> {
    pub fn new(
        context: &'a GpuContext,
        mesh: &'a Mesh,
        bvh: &'a FlattenedBvh,
        kernels: &'a [DensityKernel],
        directions_count: usize,
    ) -> Self {
        let dirs = quadrature::directions(directions_count);
        Self {
            context,
            mesh,
            bvh,
            kernels,
            dirs,
            near_field_r0: 0.01,
            bulk_density: 0.0,
            finite_part_scale: 1.0,
        }
    }

    pub fn with_finite_part_scales(mut self, near_field_r0: f32, finite_part_scale: f32) -> Self {
        self.near_field_r0 = near_field_r0;
        self.finite_part_scale = finite_part_scale;
        self
    }

    pub fn with_bulk_density(mut self, bulk_density: f32) -> Self {
        self.bulk_density = bulk_density;
        self
    }

    /// 自动判定点在内部还是外部，并执行 6-Pass GPU 计算
    pub fn compute_field(&self, points: &[[f64; 3]]) -> Result<Vec<FieldSample>, String> {
        let inside_flags: Vec<bool> = points
            .iter()
            .map(|&p| super::bvh::is_point_inside(self.bvh, self.mesh, p))
            .collect();
        self.compute_field_batch(points, &inside_flags)
    }

    /// 执行完整的 GPU 6-Pass 计算，得到所有观测点的势能 U, 加速度 g, Hessian H6 与 Tr(H)
    pub fn compute_field_batch(
        &self,
        points: &[[f64; 3]],
        inside_flags: &[bool],
    ) -> Result<Vec<FieldSample>, String> {
        let n_points = points.len();
        if n_points == 0 {
            return Ok(Vec::new());
        }
        if inside_flags.len() != n_points {
            return Err("inside_flags 数量必须与场点数量一致".into());
        }

        let n_dirs = self.dirs.len();
        let total_rays = n_points * n_dirs;

        let dev = &self.context.device;
        let queue = &self.context.queue;

        // 1. 构建上传数据
        let globals = GpuGlobals {
            n_points: n_points as u32,
            n_dirs: n_dirs as u32,
            n_tris: self.mesh.face_count() as u32,
            n_kernels: self.kernels.len() as u32,
            t_min: 1e-4,
            t_max: 1e9,
            g_si: super::analytic::G_SI as f32,
            r0: self.near_field_r0,
            ell0: self.finite_part_scale,
            bulk_density: self.bulk_density,
            _pad: [0.0; 2],
        };

        // 顶点 SoA (vec4<f32>)
        let nv = self.mesh.vertex_count();
        let mut pos_buf_data = Vec::with_capacity(nv);
        for i in 0..nv {
            let p = self.mesh.vertex(i);
            pos_buf_data.push([p[0] as f32, p[1] as f32, p[2] as f32, 0.0f32]);
        }

        // 面片三角形 (vec4<u32>)
        let nf = self.mesh.face_count();
        let mut tri_buf_data = Vec::with_capacity(nf);
        for f in 0..nf {
            let tri_idx = self.bvh.triangle_indices[f] as usize;
            let [i0, i1, i2] = self.mesh.face(tri_idx);
            tri_buf_data.push([i0, i1, i2, 0u32]);
        }

        // BVH 包围盒 (bounds_min, bounds_max)
        let num_nodes = self.bvh.node_count();
        let mut bvh_bounds_data = Vec::with_capacity(num_nodes * 2);
        let mut bvh_links_data = Vec::with_capacity(num_nodes);
        for i in 0..num_nodes {
            bvh_bounds_data.push(self.bvh.bounds_min[i]);
            bvh_bounds_data.push(self.bvh.bounds_max[i]);
            bvh_links_data.push([
                self.bvh.left[i],
                self.bvh.right[i],
                self.bvh.start[i],
                self.bvh.count[i],
            ]);
        }

        // 观测点 (vec4<f32>)
        let pts_data: Vec<[f32; 4]> = points
            .iter()
            .map(|p| [p[0] as f32, p[1] as f32, p[2] as f32, 0.0f32])
            .collect();

        // 方向集合 (vec4<f32>: u.xyz, weight)
        let dirs_data: Vec<[f32; 4]> = self
            .dirs
            .iter()
            .map(|(u, w)| [u[0] as f32, u[1] as f32, u[2] as f32, *w as f32])
            .collect();

        // Cauchy 混合核 (KernelGpu)
        let kernels_data: Vec<KernelGpu> = self.kernels.iter().map(|k| k.to_gpu()).collect();

        // 内部标志位 (u32)
        let inside_data: Vec<u32> = inside_flags
            .iter()
            .map(|&b| if b { 1u32 } else { 0u32 })
            .collect();

        // 2. 创建 GPU 缓冲区
        let globals_buf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("globals_buf"),
            contents: bytemuck::bytes_of(&globals),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let pos_buf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pos_buf"),
            contents: bytemuck::cast_slice(&pos_buf_data),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let tri_buf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tri_buf"),
            contents: bytemuck::cast_slice(&tri_buf_data),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let bvh_bounds_buf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bvh_bounds_buf"),
            contents: bytemuck::cast_slice(&bvh_bounds_data),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let bvh_links_buf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bvh_links_buf"),
            contents: bytemuck::cast_slice(&bvh_links_data),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let points_buf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("points_buf"),
            contents: bytemuck::cast_slice(&pts_data),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let dirs_buf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dirs_buf"),
            contents: bytemuck::cast_slice(&dirs_data),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let kernels_buf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("kernels_buf"),
            contents: bytemuck::cast_slice(&kernels_data),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let inside_buf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("inside_buf"),
            contents: bytemuck::cast_slice(&inside_data),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let near_field_results_buf = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("near_field_results_buf"),
            size: (total_rays * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // 过程与输出缓冲区
        let hit_counts_buf = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hit_counts_buf"),
            size: (total_rays * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let hit_offsets_buf = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hit_offsets_buf"),
            size: (total_rays * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let overflow_flags_buf = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("overflow_flags_buf"),
            size: (total_rays * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let segment_overflow_buf = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("segment_overflow_buf"),
            size: (total_rays * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let total_hits_buf = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("total_hits_buf"),
            size: std::mem::size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // 严格模式预留较大交点池；shader 仍会报告超限，不能静默截断。
        let max_compressed_hits = total_rays * 256;
        let compressed_hits_buf = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("compressed_hits_buf"),
            size: (max_compressed_hits * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let segment_counts_buf = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("segment_counts_buf"),
            size: (total_rays * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // 每条射线 256 个 vec2<f32>，与 segment_build/rtfp_integrate 一致
        let segments_buf = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("segments_buf"),
            size: (total_rays * 256 * 2 * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // 每条射线 3 个 vec4<f32> (48 字节)
        let ray_fields_buf = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ray_fields_buf"),
            size: (total_rays * 3 * 4 * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // 每个观测点 4 个 vec4<f32> (64 字节)
        let output_fields_buf = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("output_fields_buf"),
            size: (n_points * 4 * 4 * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let readback_buf = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback_buf"),
            size: ((n_points * 4 * 4 * std::mem::size_of::<f32>()) + (total_rays * std::mem::size_of::<u32>())) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // 3. 构造 BindGroups
        let pass1_bg = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pass1_bg"),
            layout: &self.context.ray_count_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: globals_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: pos_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: tri_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: bvh_bounds_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: bvh_links_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: points_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: dirs_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: hit_counts_buf.as_entire_binding() },
            ],
        });

        let prefix_globals = PrefixGlobals {
            count: total_rays as u32,
            _pad0: 0, _pad1: 0, _pad2: 0,
        };
        let prefix_globals_buf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("prefix_globals_buf"),
            contents: bytemuck::bytes_of(&prefix_globals),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let pass2_bg = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pass2_bg"),
            layout: &self.context.prefix_sum_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: prefix_globals_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: hit_counts_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: hit_offsets_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: total_hits_buf.as_entire_binding() },
            ],
        });

        let pass3_bg = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pass3_bg"),
            layout: &self.context.ray_write_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: globals_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: pos_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: tri_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: bvh_bounds_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: bvh_links_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: points_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: dirs_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: hit_offsets_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: compressed_hits_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 9, resource: overflow_flags_buf.as_entire_binding() },
            ],
        });

        let pass4_bg = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pass4_bg"),
            layout: &self.context.segment_build_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: globals_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: hit_offsets_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: hit_counts_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: compressed_hits_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: inside_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: segment_counts_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: segments_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: segment_overflow_buf.as_entire_binding() },
            ],
        });

        let pass5_bg = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pass5_bg"),
            layout: &self.context.rtfp_integrate_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: globals_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: points_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: dirs_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: kernels_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: segment_counts_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: segments_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: ray_fields_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: near_field_results_buf.as_entire_binding() },
            ],
        });

        let near_field_bg = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("near_field_bg"),
            layout: &self.context.near_field_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: globals_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: points_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: dirs_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: kernels_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: near_field_results_buf.as_entire_binding() },
            ],
        });

        let pass6_bg = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pass6_bg"),
            layout: &self.context.reduce_field_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: globals_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: ray_fields_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: output_fields_buf.as_entire_binding() },
            ],
        });

        // 4. 提交 Compute Passes
        let mut encoder = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("rtfp_gpu_encoder"),
        });
        encoder.clear_buffer(&overflow_flags_buf, 0, None);
        encoder.clear_buffer(&segment_overflow_buf, 0, None);

        let num_workgroups_rays = ((total_rays as u32) + 63) / 64;

        // Pass 1: ray_count
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("pass1_ray_count"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.context.ray_count_pipeline);
            cpass.set_bind_group(0, &pass1_bg, &[]);
            cpass.dispatch_workgroups(num_workgroups_rays, 1, 1);
        }

        // Pass 2: prefix_sum
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("pass2_prefix_sum"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.context.prefix_sum_pipeline);
            cpass.set_bind_group(0, &pass2_bg, &[]);
            cpass.dispatch_workgroups(((total_rays as u32) + 255) / 256, 1, 1);
        }

        // Pass 3: ray_write
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("pass3_ray_write"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.context.ray_write_pipeline);
            cpass.set_bind_group(0, &pass3_bg, &[]);
            cpass.dispatch_workgroups(num_workgroups_rays, 1, 1);
        }

        // Pass 4: segment_build
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("pass4_segment_build"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.context.segment_build_pipeline);
            cpass.set_bind_group(0, &pass4_bg, &[]);
            cpass.dispatch_workgroups(num_workgroups_rays, 1, 1);
        }

        // Pass 5: rtfp_integrate
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("near_field"), timestamp_writes: None });
            cpass.set_pipeline(&self.context.near_field_pipeline);
            cpass.set_bind_group(0, &near_field_bg, &[]);
            cpass.dispatch_workgroups(num_workgroups_rays, 1, 1);
        }
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("pass5_rtfp_integrate"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.context.rtfp_integrate_pipeline);
            cpass.set_bind_group(0, &pass5_bg, &[]);
            cpass.dispatch_workgroups(num_workgroups_rays, 1, 1);
        }

        // Pass 6: reduce_field (每个点 1 个 workgroup)
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("pass6_reduce_field"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.context.reduce_field_pipeline);
            cpass.set_bind_group(0, &pass6_bg, &[]);
            cpass.dispatch_workgroups(n_points as u32, 1, 1);
        }

        // 拷贝输出缓冲区至可读取缓冲区
        encoder.copy_buffer_to_buffer(
            &output_fields_buf,
            0,
            &readback_buf,
            0,
            (n_points * 4 * 4 * std::mem::size_of::<f32>()) as u64,
        );
        encoder.copy_buffer_to_buffer(
            &segment_overflow_buf, 0, &readback_buf,
            ((n_points * 4 * 4 * std::mem::size_of::<f32>()) + (total_rays * std::mem::size_of::<u32>())) as u64,
            (total_rays * std::mem::size_of::<u32>()) as u64,
        );
        encoder.copy_buffer_to_buffer(
            &overflow_flags_buf,
            0,
            &readback_buf,
            (n_points * 4 * 4 * std::mem::size_of::<f32>()) as u64,
            (total_rays * std::mem::size_of::<u32>()) as u64,
        );

        queue.submit(Some(encoder.finish()));

        // 5. 回读输出数据
        let slice = readback_buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });

        let poll_type = wgpu::PollType::wait_indefinitely();
        dev.poll(poll_type).map_err(|e| format!("GPU poll 失败: {e:?}"))?;
        rx.recv()
            .map_err(|e| format!("GPU 通道接收失败: {e}"))?
            .map_err(|e| format!("GPU 映射失败: {e:?}"))?;

        let view = slice.get_mapped_range().map_err(|e| format!("获取映射范围失败: {e:?}"))?;
        let field_bytes = n_points * 4 * 4 * std::mem::size_of::<f32>();
        let raw_floats: &[[f32; 4]] = bytemuck::cast_slice(&view[..field_bytes]);
        let overflow_bytes = &view[field_bytes..field_bytes + total_rays * std::mem::size_of::<u32>()];
        let overflow: &[u32] = bytemuck::cast_slice(overflow_bytes);
        if overflow.iter().any(|v| *v != 0) {
            drop(view);
            readback_buf.unmap();
            return Err("GPU 射线交点池溢出，结果被拒绝".into());
        }
        let segment_offset = field_bytes + total_rays * std::mem::size_of::<u32>();
        let segment_bytes = &view[segment_offset..segment_offset + total_rays * std::mem::size_of::<u32>()];
        let segment_overflow: &[u32] = bytemuck::cast_slice(segment_bytes);
        if segment_overflow.iter().any(|v| *v != 0) {
            drop(view); readback_buf.unmap();
            return Err("GPU 交段数量超过容量，结果被拒绝".into());
        }

        let mut samples = Vec::with_capacity(n_points);
        for p in 0..n_points {
            let row0 = raw_floats[4 * p + 0];
            let row1 = raw_floats[4 * p + 1];
            let row2 = raw_floats[4 * p + 2];

            let u_val = row0[0] as f64;
            let g_val = [row0[1] as f64, row0[2] as f64, row0[3] as f64];
            let h_val = [
                row1[0] as f64,
                row1[1] as f64,
                row1[2] as f64,
                row1[3] as f64,
                row2[0] as f64,
                row2[1] as f64,
            ];
            let tr = row2[2] as f64;

            let branch = if inside_flags[p] {
                FieldBranch::InteriorFinitePart
            } else {
                FieldBranch::ExternalClassical
            };

            samples.push(FieldSample {
                potential: u_val,
                gravity: g_val,
                hessian6: h_val,
                trace: tr,
                branch,
            });
        }

        drop(view);
        readback_buf.unmap();

        Ok(samples)
    }
}

impl RayBackend for GpuBvhWgslBackend<'_> {
    fn intersect_batch(&self, rays: &RayBatch) -> RayHitBuffer {
        // 利用既有 BVH 求交逻辑构建 RayHitBuffer
        let n = rays.len();
        let mut hits_per_ray = Vec::with_capacity(n);
        for i in 0..n {
            let orig = rays.origins[i];
            let dir = rays.directions[i];
            let hits = super::bvh::intersect_ray_bvh(self.bvh, self.mesh, orig, dir);
            hits_per_ray.push(hits);
        }
        RayHitBuffer { hits_per_ray }
    }
}
