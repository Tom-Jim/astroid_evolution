# 小行星地质演化仿真器：软件工程设计说明书

> 版本：v2（完全重构）。
> 数学与物理的**唯一真理源**是 `docs/mathstrict.md`（下称"数学规范"）。本文只规定"如何把数学规范落成 Rust/WGSL 软件"，不修改、不妥协、不质疑其任何公式与约定。
> 本文以"章.节"（如 §8.4′、§10.4.4、附录 A）引用数学规范，凡出现 `[MS §x]` 即指数学规范中的对应位置。
> 本文**不迁就** `src/` 中任何历史实现；现有目录按本文第 2 章整体重建。
> **架构范式（v2.1）：Bevy ECS 为绝对骨架。** 全系统是 Bevy 插件（Plugin）、调度（Schedule / SystemSet / State）、组件（Component）、资源（Resource）与事件（Event）的组合；**不存在** OOP/Trait 派发（无 `Simulator<F,O>`、`FieldProvider`、`SimulationDriver`、`Observer`、`RayBackend`、`FarField`、`AssetSource/Sink` trait），**不存在**后台 worker 线程与双缓冲胶水层。无头 CLI 是纯 Bevy Headless `App`；GUI 在同一 `App`/`World` 上挂载渲染与 egui 插件，直接共享状态与 `Query`。网格与点集在**单个 Component 内**以连续 SoA 存储，**禁止**把顶点/场点拆成单独 Entity。本次重构不触及任何数学内容。

**数学规范中被本文视为不可更改的核心约定（实现红线）**

| 编号 | 约定 | 出处 |
|---|---|---|
| C1 | 全系统只有**一个应力场** $\boldsymbol\sigma=-p\mathbf I+\mathbf T_{\rm tide}+2\eta(\tilde p,\mathbf D)\mathbf D$；不存在弹性应变、位移自由度、硬化变量、返回映射 | MS §2.8 |
| C2 | 自转 $\omega(t)$ **外生给定**（`spin_schedule` 必须显式给定：`stages`/`linear`/`table`），与流动、惯量、角动量无任何反馈 | MS §10.4.1 |
| C3 | 每个地质步求解**准静态 Stokes 隐式平衡**：速度空间精确无散的极向—环向双势 Galerkin（$K=4$），系统 (8.4′)；压力由**独立标量 Poisson** 给出；二者经 Picard 迭代闭合 | MS §8.4′、§8.7、§10.4.2 |
| C4 | 屈服判据 $F_\delta=\|\boldsymbol\tau_{\rm flow}\|_{\rm eq}-(1-\delta_Y)\tau_y>0$；未屈服连通域中质量最大者为**刚性核**，核上表面节点**死区锁定**（$d_i=0$ 硬约束） | MS §2.8 B2–B4 |
| C5 | 几何校正是以**当前几何 $R^{(n)}$ 为基准**的 Clarabel 弹性 SOCP：目标不含 $R_{\rm eq}$、$W_0$ 或任何等势面项；内层绝不缩小信赖域，仅外层几何验收失败（倒悬/自交/星形性/质量）才缩小 | MS §2.13、§10.4.3 |
| C6 | 模式 A（流体质量能级 $W_0$、等势面 $R_{\rm eq}$）**只保留接口**，不进入闭环，**不得删除** | MS §2.11–2.12 |
| C7 | 模式 B/C 准静态主线**不输运**模态动量 $\mathbf m$；仅对表面面密度 $\rho_s$ 做单场半拉格朗日输运；动态自由表面（第 9 章其余部分）不在本软件范围内 | MS §9 适用范围 |
| C8 | 缺任一必需模型参数 → `ModelNotClosed`，拒绝运行；不存在隐式默认值或回退到流体等势目标 | MS §10.4.4 步骤 0 |
| C9 | Track A（全局 PCG + 值域空间法）为生产基准；Track B（块对角 $8\times8$）仅作预条件子，或带后验残差 $\varepsilon_B$ 的显式近似 | MS 附录 A.4 |

---

# 1. 系统总体架构与三层划分

## 1.1 三层解耦与依赖方向

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│ Tier 3  asteroid_app   (AsteroidPlugin：Bevy 渲染 + egui 插件)               │
│   3D 渲染 · 地形形变 · 应力/流速覆层 · 时间轴 · 参数面板 · 快照回放          │
│   仅"驱动与展示"，不含任何物理公式                                           │
└───────────────▲──────────────────────────────────────────────────────────────┘
                │  同一 World：Query / Res（只读）· ParamPatch 事件 · .snap 文件
┌───────────────┴──────────────────────────────────────────────────────────────┐
│ Tier 2  scf_simulator  (ScfPlugin，headless App，bin: scf-cli)               │
│   外生自转表 → [RTFP 正演] → Stokes–Poisson Picard → 屈服分类/刚性核         │
│   → 摩擦角闭合 → 模式 C 通量 → 地质步长 → Clarabel SOCP → 几何更新+重对齐    │
│   快照序列化 · 主循环 · 断点续算                                             │
└───────────────▲──────────────────────────────────────────────────────────────┘
                │  FieldRequest 事件 + FieldBatch 组件  /  .field.bin / 管道
┌───────────────┴──────────────────────────────────────────────────────────────┐
│ Tier 1  rtfp_solver    (RtfpPlugin，headless App，bin: rtfp-cli)             │
│   多面体 BVH · 射线求交 · Cauchy 密度及导数 · Feynman 参数化 · 有限部分       │
│   输出 U, ∇U, ∇²U (及质量/体积分)；WGSL 射线计算管线 + CPU 参考后端          │
└───────────────▲──────────────────────────────────────────────────────────────┘
                │
┌───────────────┴──────────────────────────────────────────────────────────────┐
│ common  (纯数学与数据定义：向量/张量、单位制、StarMesh、BVH 数据、TOML schema、 │
│         二进制容器、AssetIo 资源、GpuContext 资源、并行抽象；共享 Component 定义)│
└──────────────────────────────────────────────────────────────────────────────┘
```

**依赖规则（强制，由 CI 脚本 `tools/check_layering.sh` 校验）**

1. 依赖只能自上而下：`asteroid_app → scf_simulator → rtfp_solver → common`；禁止反向与跨层循环。
2. `common` 除 `wgpu` 与 Bevy 的无渲染子集（`bevy_ecs`/`bevy_app`/`bevy_tasks`/`bevy_state`，仅用于 `derive(Component/Resource/Event)` 与插件注册）外不依赖任何上层，不含任何物理求解逻辑，只含数据定义与基础设施。
3. Tier 1/2 以 `bevy`（`default-features = false`，仅无渲染子集：`bevy_app`/`bevy_ecs`/`bevy_state`/`bevy_tasks`/`bevy_time`/`bevy_log`）构成 **Headless App**；`rtfp_solver` 不得引用 `clarabel`、`bevy_render`、`bevy_winit`、`egui`；`scf_simulator` 不得引用 `bevy_render`、`bevy_winit`、`egui`、`winit`。渲染/窗口/egui 只允许出现在 `asteroid_app`。
4. 三层之间的**运行时耦合只有一个窄接口：共享的 ECS `World`**。Tier 2 通过 `FieldRequest` 事件与 `FieldBatch` 组件（§1.3）获取场数据；Tier 3 通过对 Tier 2 组件/资源的**只读** `Query`/`Res` 展示状态，并通过 `ParamPatch` 事件（§1.4、§3.7）提交修改。**不存在 trait 对象、泛型静态分派的 Simulator、后台 worker 线程与双缓冲。**
5. 目录名不使用 `core`：`crate::core` 会与标准库 `core` 在 `use core::…` 上产生歧义，故共享层命名为 `common`。

## 1.2 各层职责边界

| 层 | 输入 | 输出 | 专职 | 明确不做 |
|---|---|---|---|---|
| Tier 1 `rtfp_solver` | `StarMesh`/任意闭合三角网格、Cauchy 密度核、场点集、自转角速度（仅用于合成 $W$，可选）| $U,\nabla U,\nabla^2U$ 缓冲区；体积/质量/质心/惯量张量；诊断标志 | 引力正演、有效势 $W=U+\tfrac12\omega^2(x^2+y^2)$、$\mathbf g_{\rm eff}$、$H_W$ | 不含流变、Stokes、几何更新、时间推进 |
| Tier 2 `scf_simulator` | 项目 TOML、初始 `StarMesh`、Tier 1 场数据 | 时间序列快照 `.snap`、诊断日志 | 准静态 Stokes 双势 Galerkin、Poisson 压力、Picard、屈服分类、摩擦角闭合、模式 C、Clarabel SOCP、几何重建、参考系重对齐 | 不含任何窗口/渲染；不含动态自由表面（C7）|
| Tier 3 `asteroid_app` | 项目 TOML 或 `.snap` 序列 | 交互式 3D 显示、参数调整后的重启请求 | 渲染、交互、播放、参数面板 | 不含物理公式；不改写 Tier 2 状态（只能在步边界提交"参数补丁"）|

## 1.3 层间接口一：`FieldRequest` 事件 / `FieldBatch` 组件（Tier 1 → Tier 2）

Tier 1 是 `RtfpPlugin`：它注册 `FieldSet` 系统集，消费 `FieldRequest` 事件，把结果写回**点集实体**上的 `FieldBatch` 组件。每个 `PointRole` 对应**一个**点集实体（`PointSet` 组件内为连续 SoA；**一个实体 = 一类点集，绝不是一个场点**），Tier 2 的系统用 `Query<(&PointRole, &PointSet, &mut FieldBatch)>` 直接读取。

```rust
/// 一次正演请求使用的点集语义（决定 Hessian 是否取有限部分/单侧规则）
#[derive(Component, Clone, Copy)]
pub enum PointRole {
    SurfaceVertex,   // g_i = ∇U + ω² x_⊥（节点），只要 U,∇U
    SurfaceFace,     // 面心 Hessian（MS §10.3：顶点 Hessian 不得用于 T_tide）
    VolumeQuad,      // 体内 Stokes 求积点：只要 ∇U（经典收敛，无需 Hessian）
    ClassifyGrid,    // X_Y：体内点 ∇U,∇²U（经典收敛）
}

#[derive(Component)] pub struct PointSet  { pub x: Vec<f64>, pub y: Vec<f64>, pub z: Vec<f64> }  // SoA，f64，SI
#[derive(Component)] pub struct FieldWant(pub FieldMask);                                          // U | GRAD | HESS
#[derive(Component)] pub struct FieldBatch { /* §3.3：u / grad[3] / hess[6] / flags，SoA */ }

/// 事件：Tier 2 → Tier 1
pub enum FieldRequest {
    Bind,                                              // 绑定当前 StarMesh 与 CauchyDensity（重建 BVH）；每个地质步一次
    Eval      { entity: Entity },                      // 对某个点集实体求值，结果写回其 FieldBatch
    Integrals,                                         // 体积、质量 𝓜(V,ρ_K)、质心、惯量张量（步骤 8a/8b 与质量 DM 所需）→ VolumeIntegrals 资源
    RayRadii  { origin: DVec3, dirs: Entity },         // 沿固定方向 ω̂_i 自给定原点射线求交（重对齐 8b(v) 与 Stokes 的 R 采样）→ RayRadii 组件
}
/// 事件：Tier 1 → Tier 2（完成通知；错误为 RtfpError 事件，由 Tier 2 汇入 StepStatus）
pub enum FieldReady { Bound, Evaluated(Entity), Integrals, RayRadii }
```

后端选择（`rtfp.backend`）是**资源 `RtfpBackendKind` + 系统集 `run_if`**，而非 trait：`FieldSet::Gpu`（P1–P7 管线）与 `FieldSet::CpuRayon`（同算法 f64）两组系统消费同一 `FieldRequest` 并写同一 `FieldBatch`；生产路径无虚调用。

## 1.4 层间接口二：共享 `World`（Tier 2 → Tier 3）

Tier 3 不持有仿真器句柄，也不接收快照拷贝；它与 Tier 2 **运行在同一个 `App` / `World`** 中：

- **状态读取**：渲染与 egui 系统以只读 `Query<&StarMesh>`、`Query<&SurfaceFields>`、`Res<StepRecord>` 等直接读取 R、顶点、$\rho_s$、$\eta$、$\tilde p$、$F_\delta$、$\chi_Y$、$\mathbf v_{\rm sl}$ …（零拷贝，无 `SnapshotView`、无双缓冲）。Tier 3 系统对 Tier 2 数据**只允许** `Res`/`Query<&T>`，由系统签名在编译期保证（U1）。
- **推进**：`StepPhase`（Bevy `States`）状态机在 `Update` 调度中协作式推进（§1.6）；`Res<StepControl>`（`Run | Pause | StepOnce`）由 UI 修改。
- **修改**：`ParamPatch` 事件，仅在 `StepPhase::Idle`（步边界）由 `apply_patch` 系统消费（见 §3.7）。
- **保存**：`SaveSnapshot` 事件 → `snapshot` 系统经 `AssetIo` 资源写 `.aevb`（§7.2）。

UI 可以**纯回放** `.snap` 文件而不加入 `ScfPlugin`：回放插件把 `.snap` 反序列化为同一批 Tier 2 输出组件/资源，渲染系统无需区分来源。

## 1.5 数据流向（生产路径）

```text
project.toml ─┐
mesh/pointcloud/glb ─► [common::geometry::StarMesh::from_*] ─► R^(0), 固定节点方向 {ω̂_i}
                                        │
   对 n = 0,1,2,…, t_n < T_end:        ▼
   spin_schedule ─► ω_n ─► FieldRequest::Bind(mesh_n)
        ├─ Eval(SurfaceVertex)  → g_i, ∇U_i
        ├─ Eval(SurfaceFace)    → Hess U(面心) → T_tide(面心→节点面积平均)
        ├─ Eval(VolumeQuad)     → ∇U(X_vol)
        └─ Eval(ClassifyGrid)   → ∇U, ∇²U (X_Y)
                │
                ▼  Picard(k):  P1 Poisson(p) → P2 黏度 η → P3 Stokes (8.4′) → P4 收敛判据(a)
        (u^(n), p^(n), λ^(n), η^(n)) ─► 判据(b) 力学平衡验收
                │
                ▼  F_δ → 图 G_Y 连通分量 → 刚性核 U_core → χ_Y → v_sl
        摩擦链(10.3) → φ_i ;  模式C → J_ero,J_dep ;  Δt_n ;  d_pred ; L/D/A 分类
                │
                ▼  Clarabel SOCP (10.4.3) → d_* ─► R^(n+1)=R^(n)+λ d_*  → 几何验收(失败→缩信赖域)
                │
                ▼  [触发时] 质心/主轴重对齐 (8b) → ρ_s 半拉格朗日输运 → 写快照/日志 → n+1
```

上图每个方框是一个 `StepPhase`（对应一个 `ScfSet` 系统集），箭头是 `NextState<StepPhase>` 转移；见 §1.6。

## 1.6 ECS 调度总览（Plugins / Schedules / SystemSets / States）

**插件组合**（每个 bin 只是 `App::new()` + 下表插件，无其他胶水）

| bin | 插件组合 | 运行器 |
|---|---|---|
| `rtfp-cli` | `MinimalPlugins` + `CommonPlugin` + `RtfpPlugin` + `RtfpCliPlugin`（子命令 → 发出 `FieldRequest`，完成后写出并发 `AppExit`）| `ScheduleRunnerPlugin`（循环，直至 `AppExit`）|
| `scf-cli` | `MinimalPlugins` + `CommonPlugin` + `RtfpPlugin` + `ScfPlugin` + `ScfCliPlugin` | 同上 |
| `asteroid-app` | `DefaultPlugins`（精简：`bevy_render`/`bevy_winit`/`bevy_window`）+ `EguiPlugin` + `CommonPlugin` + `RtfpPlugin` + `ScfPlugin` + `AsteroidPlugin`（回放模式不加 `ScfPlugin`/`RtfpPlugin`，改加 `PlaybackPlugin`）| winit 事件循环 |

`diagnostics` feature 额外提供 `RtfpDiagnosticsPlugin` / `ScfDiagnosticsPlugin`（§6），未启用时不存在。

**核心资源 / 组件 / 事件**

| 类别 | 项 | 说明 |
|---|---|---|
| Resource | `ProjectConfig`、`UnitSystem`、`GpuContext`、`AssetIo`、`SpinSchedule`、`StepControl`、`StepRecord`、`PicardState`、`TrustRegion`、`VolumeIntegrals` | 全局唯一量；`ProjectConfig` 由 `config::validate` 校验后插入（失败 → `ModelNotClosed`，C8）|
| Component（单一 `Asteroid` 实体）| `StarMesh`（SoA）、`SurfaceFields`（SoA：$\rho_s$、$\eta$、$\tilde p$、$F_\delta$、$\chi_Y$、$\mathbf v_{\rm sl}$ …）、`ModalState`（$\mathbf u^0,p^0$）、`KernelCenters`、`FrameState`（$Q_{\rm cum}$）| **所有逐节点数据都是该实体上某组件内的连续 `Vec`**，禁止每顶点一个 Entity |
| Component（点集实体，每个 `PointRole` 一个）| `PointRole`、`PointSet`、`FieldWant`、`FieldBatch` | §1.3 |
| Event | `FieldRequest`/`FieldReady`、`ParamPatch`、`SaveSnapshot`、`PicardIterDone`、`StepEnded`、`StepFailed(StepStatus)` | `StepFailed` 由 `exit_on_status` 系统映射为 `AppExit::Error(code)`（§3.6）|
| State | `StepPhase`：`Idle → Spin → Bind → Forward → Picard → Yield → Friction → ModeC → Timestep → Socp → Accept → Realign → Transport → Output → Idle` | 与 MS §10.4.4 步骤 0–9 一一对应；`Picard` 相位内自循环（P1→P4）；`Accept` 失败回到 `Socp`（外层缩小信赖域，不推进时间）|

**调度**：所有物理系统位于 `Update`，分属 `ScfSet::{Spin, Bind, Forward, Picard, Yield, Friction, ModeC, Timestep, Socp, Accept, Realign, Transport, Output}`，每个系统集 `.run_if(in_state(StepPhase::…))`；集间 `.chain()` 保证顺序，集内系统由 Bevy 多线程执行器按 `Query`/`Res` 访问冲突**自动并行**；系统内的数据并行由 `common::par`（rayon）负责。重型相位（Stokes PCG、Clarabel、GPU 回读）以 Bevy `AsyncComputeTaskPool` 任务实现：系统 `spawn` 一个任务并把 `Task<CommandQueue>` 存为组件，`poll_task` 系统在完成时直接把结果写入 `World` 并 `NextState` 转移——结果**就地**进入 ECS，不存在 worker 线程所有权、消息通道或双缓冲。`StepControl::mode` 选择：`Cooperative`（GUI 默认，每帧推进、不阻塞渲染）或 `Blocking`（CLI 默认，同一 tick 内 `block_on` 完成当前相位；`DeterministicReduce` 回归测试亦用此模式）。

---

# 2. 源码目录结构规范（`src/`）

## 2.1 Cargo 组织

单包（package `astroid_evolution`），按 feature 与 bin 切分三层；层边界与未来拆分为 workspace 的 crate 边界一一对应（拆分时不改任何 `use` 语义）。

```toml
[features]
default      = ["tier1"]
tier1        = []                                   # rtfp_solver + common（Bevy 无渲染子集，Headless App）
tier2        = ["tier1", "dep:clarabel"]            # scf_simulator
tier3        = ["tier2", "bevy/bevy_render", "bevy/bevy_winit", "bevy/bevy_window", "dep:bevy_egui"]   # 渲染/窗口/egui 仅此处开启
diagnostics  = []                                   # 全部验证/基准/误差预算代码（§6）
wasm         = []                                   # 仅开启 wasm 适配层（§7）

[[bin]] name = "rtfp-cli"      path = "src/bin/rtfp_cli.rs"      required-features = ["tier1"]
[[bin]] name = "scf-cli"       path = "src/bin/scf_cli.rs"       required-features = ["tier2"]
[[bin]] name = "asteroid-app"  path = "src/bin/asteroid_app.rs"  required-features = ["tier3"]
```

依赖（与当前项目一致，仅调整归属）：`bevy`（`default-features = false`，所有层共用的 ECS 骨架：`bevy_app`/`bevy_ecs`/`bevy_state`/`bevy_tasks`/`bevy_time`/`bevy_log`）、`wgpu`、`rayon`、`nalgebra`（CPU f64 权威数值）、`glam`（仅用于 GPU 打包与 Bevy 边界）、`bytemuck`、`clarabel`（tier2）、`bevy_egui`（tier3；渲染/窗口 feature 见上，**wgpu 必须与 Bevy 内置版本一致**，不得出现两份 wgpu）、`serde`+`toml`、`gltf`（`.glb` 导入）。新增 `clap`（CLI）、`thiserror`（错误类型）、`crc32fast`（容器校验）。移除 `parry3d`、`num-rational`（BVH/求交自研，见 §4.1）；`pollster` 仅在 `src/bin/*` 与 `common::native` 内使用。

## 2.2 目录树

```text
astroid_evolution/
├── Cargo.toml
├── assets/                          # 示例资产（模型 .glb、点云、density、project）
├── docs/{mathstrict.md, software.md}
├── tools/check_layering.sh          # 依赖方向/禁用 std::fs 检查
├── tests/                           # 集成与验收测试（§5），按层命名
│   ├── tier1_*.rs  tier2_*.rs  tier3_*.rs
│   ├── data/                        # 极小固定样例（icosphere、微型 Cauchy 核）
│   └── verify_closure.py            # 仅抽象线性代数恒等式（MS §9.4 注记），不构成物理验证
└── src/
    ├── lib.rs                       # pub mod common; #[cfg(tier1)] rtfp_solver; …（每层导出一个 Plugin）
    │
    ├── common/                      # ───────── 共享底层（纯数据/基础设施）─────────
    │   ├── mod.rs                   # CommonPlugin：注册 UnitSystem/GpuContext/AssetIo 资源与通用事件
    │   ├── units.rs                 # UnitSystem：SI ↔ 无量纲内部单位（L0=R_ref, ρ0, G）
    │   ├── math.rs                  # DVec3/DMat3、对称张量打包(6)、dev、‖·‖_eq、安全 normalize
    │   ├── geometry/
    │   │   ├── star_mesh.rs         # StarMesh（Component，连续 SoA）：固定节点方向 ω̂_i + 半径 R_i + 固定拓扑三角形
    │   │   ├── icosphere.rs         # 节点方向生成器（细分球面）
    │   │   ├── tri_mesh.rs          # 任意闭合三角网格（输入/输出）、定向/闭合/自交检查
    │   │   ├── bvh.rs               # 扁平 BVH 数据结构与 CPU 构建（SAH/binned），上传布局
    │   │   └── quad_grids.rs        # 球面求积网格 (Gauss–Legendre×等距φ)、Lebedev 表
    │   ├── density.rs               # CauchyDensity { kernels: Vec<Kernel{c,w,σ,α}> }，ρ_K、∇ρ_K
    │   ├── config/                  # TOML schema（serde）与严格校验（ModelNotClosed）
    │   │   ├── project.rs  spin.rs  material.rs  numerics.rs  output.rs  validate.rs
    │   ├── io/
    │   │   ├── asset.rs             # AssetIo（Resource，enum：Fs/Mem/Fetch）+ AssetRead/AssetWrite 事件与完成系统（异步，无 trait）
    │   │   ├── container.rs         # 二进制容器 .aevb 读写（§3.4）
    │   │   ├── formats.rs           # xyz 点云、.glb、.mesh/.field/.snap 编解码
    │   │   └── native_fs.rs         # #[cfg(not(wasm32))]：唯一允许 std::fs 的位置
    │   ├── gpu/
    │   │   ├── context.rs           # GpuContext（Resource）：async 获取 adapter/device；from_external()（取自 Bevy RenderDevice）
    │   │   ├── buffers.rs           # 带标签的 Buffer 池、分块上传、async 回读
    │   │   ├── pod.rs               # #[repr(C)] bytemuck Pod 结构（与 WGSL struct 一一对应）
    │   │   ├── scan.rs              # 通用 GPU 前缀和（Blelloch，三遍）
    │   │   └── shader_compose.rs    # WGSL 片段拼接（无 include 的替代）、override 常量注入
    │   ├── par.rs                   # 并行抽象：native=rayon；wasm 无线程=串行（§7）
    │   ├── harmonics/
    │   │   ├── real_sh.rs           # 实球谐 Y_ℓm 及其 ∇_S²、Δ_S² ，Gaunt 带宽表
    │   │   ├── sht.rs               # 球谐分析/综合（rayon；GEMM 形式）
    │   │   └── jacobi.rs            # 移位 Jacobi ψ_ℓk、Gauss–Jacobi 节点、σ_ℓk/κ_ℓk 表（MS §10.2.1）
    │   ├── events.rs                # 通用事件：PicardIterDone、StepEnded、StepFailed（§6 诊断系统的只读订阅点）
    │   └── error.rs
    │
    ├── rtfp_solver/                 # ───────── Tier 1 ─────────
    │   ├── mod.rs                   # RtfpPlugin、FieldSet 系统集、RtfpBackendKind（Resource）
    │   ├── api.rs                   # PointRole/PointSet/FieldBatch(Component)、FieldRequest/FieldReady(Event)、VolumeIntegrals(Resource)
    │   ├── geometry/
    │   │   ├── ray_query.rs         # 射线—三角形求交（Möller–Trumbore 稳健版）+ BVH 遍历 CPU 版
    │   │   ├── segments.rs          # 内外分类与交段构建 [R0,R1] 规则（MS §3.6）
    │   │   └── inside_test.rs       # 奇偶性内外判定 + 边界点内/外侧极限约定
    │   ├── density/
    │   │   ├── cauchy_ray.rs        # 沿射线的 b_k、d_k² 、ρ、∂_Rρ 解析式（MS §1.4–1.6）
    │   │   └── derivatives.rs       # ∇ρ、∇²ρ（有限部分减法所需）
    │   ├── radial/
    │   │   ├── analytic_alpha1.rs   # α=1 解析原函数 I_U,I_g,I_H（MS §3.2）
    │   │   ├── gauss_jacobi.rs      # 一般 α 的固定节点求积表（MS §3.3）
    │   │   └── finite_part.rs       # 近场球形有限部分、log(r0/ℓ0)、接触项 −(4πG/3)ϑρI（MS §3.9）
    │   ├── field/
    │   │   ├── fused.rs             # 融合 U/∇U/H 累积（T(u)=3uuᵀ−I，6 分量）的 CPU 实现
    │   │   ├── effective.rs         # W=U+½ω²(x²+y²)，g_eff，H_W 合成
    │   │   └── far_field.rs         # 远场多极系统集 FarFieldSet（v1 为占位：run_if 关闭，直接法回退）
    │   ├── backend/
    │   │   ├── gpu.rs               # FieldSet::Gpu 系统：管线编排、分块、描述符布局
    │   │   ├── cpu_rayon.rs         # FieldSet::CpuRayon 系统：同算法 f64，rayon 并行（无 GPU 环境/参考靶）
    │   │   └── dispatch.rs          # 混合调度 d<d_near / 中间 / d>d_far（MS §3.11）
    │   ├── shaders/                 # ★ Tier 1 WGSL
    │   │   ├── common/{types.wgsl, bvh.wgsl, tri.wgsl, cauchy.wgsl, radial.wgsl}
    │   │   ├── ray_count.wgsl       # 阶段 A：统计交点数
    │   │   ├── prefix_sum.wgsl      # 阶段 B：分块 Blelloch 扫描 + 块偏移回加
    │   │   ├── ray_write.wgsl       # 阶段 C：重发射线，写入压缩交点数组并按 t 排序
    │   │   ├── segment_build.wgsl   # 奇偶/内外状态 → 交段 [R−,R+]
    │   │   ├── rtfp_integrate.wgsl  # 融合 kernel：径向积分 + 张量累积 + workgroup 归约
    │   │   ├── near_field.wgsl      # 近场球形有限部分（R<r0 的差分项 + 解析 log 项）
    │   │   ├── reduce_field.wgsl    # 跨 workgroup/方向批的最终归约
    │   │   └── far_field.wgsl       # (预留) 远场多极
    │   ├── volume.rs                # 体积/质量/质心/惯量（散度定理 + 径向 Gauss–Jacobi）
    │   ├── ray_radii.rs             # R(ω̂) 射线采样系统（响应 FieldRequest::RayRadii，供 Tier 2 复用）
    │   ├── mesh_import.rs           # 点云/.glb → StarMesh（径向射线重采样；要求星形）
    │   ├── cli.rs                   # RtfpCliPlugin：rtfp-cli 子命令 → FieldRequest，完成后 AppExit
    │   └── diagnostics/             # #[cfg(feature="diagnostics")]（§6）
    │       ├── cpu_reference.rs     # 高精度 CPU 参考（自适应求积，f64）
    │       ├── feynman_track.rs     # Feynman 参数化独立通道 + Carlson 判据（MS §1.7–1.9, §2.2）
    │       ├── polyhedron_exact.rs  # 均匀密度多面体解析（Werner–Scheeres 类）独立参考
    │       ├── sphere_analytic.rs   # 均匀球/同心层球解析解
    │       ├── laplace_checks.rs    # tr H=−4πGρ、跳跃 H⁺−H⁻=4πGρ nnᵀ、梯度连续性
    │       ├── gpu_roundoff.rs      # f32 舍入/归约误差评估（对比 f64 后端）
    │       └── bench.rs
    │
    ├── scf_simulator/               # ───────── Tier 2 ─────────
    │   ├── mod.rs                   # ScfPlugin：注册资源/事件、StepPhase 状态、ScfSet 系统集排布（以下各子模块均为系统 + 纯函数，由此挂入）
    │   ├── state.rs                 # Component/Resource 定义（原 SimState 拆分，§3.3）：几何、ρ_s、诊断量、u^0/p̃^0（热启动）、t_n、Q_cum、c_k
    │   ├── spin.rs                  # SpinSchedule{Stages,Linear,Table}：ω(t)、ω̇、t_next
    │   ├── basis/
    │   │   ├── layout.rs            # DOF 排序：球谐优先、8 个 (X,k) 连续；N_u=8L(L+2)
    │   │   ├── radial_traces.rs     # σ_ℓk、κ_ℓk、c^P_ℓk、c^T_ℓk 表（10.2.1/10.2.4）
    │   │   └── geometry_repr.rs     # R_h：带限 SH 重构 R、∇R、ΔR（Stokes 用，见 §4.7）
    │   ├── quadrature/
    │   │   ├── volume_points.rs     # X_vol：角向球面求积 × 径向 Gauss–Jacobi（权 ζ^{2ℓ+2}）
    │   │   └── classify_grid.rs     # X_Y：N_r+1 径向层(含 ζ=1) × 表面顶点方向；图 G_Y 邻接
    │   ├── stokes/
    │   │   ├── assemble.rs          # K(η)、F、B_rb（CPU 参考装配 + GPU 装配调度）
    │   │   ├── strain_rate.rs       # 解析 D(B_α) （引理 5.5）与 D(u) 合成
    │   │   ├── solve.rs             # 值域空间法：K_γ 的 7 次 PCG + 6×6 Schur（MS §8.4′）
    │   │   ├── precond_b1.rs        # 块对角 8×8 Cholesky 预条件子 K_{γ,B1}^{-1}（rayon）
    │   │   ├── rigid.rs             # B_rb 行、秩/最小奇异值报告、可解性相容检查（A.3）
    │   │   └── track_b.rs           # Track B 显式近似 + ε_B（仅配置开启时）
    │   ├── pressure/
    │   │   ├── poisson.rs           # MS §8.7：刚度 K_p、提升 p_b、右端 f、PCG
    │   │   └── rhs_fields.rs        # ∇·T_tide、∇·τ_flow=ηΔv+2D∇η、ρ_K g_eff 的体内取值
    │   ├── rheology/
    │   │   ├── closure.rs           # η(p̃,D)（Bingham/Papanastasiou 二选一）、μ_d、I_c、p_c、p̃
    │   │   ├── tide_stress.rs       # T_tide=C_T ρ_p ν0 d_p² dev H_W
    │   │   └── picard.rs            # P1–P4 循环；10γ_ε 连续化重试；θ_η 松弛
    │   ├── yielding/
    │   │   ├── yield_fn.rs          # F_δ、γ̇_Y（诊断）
    │   │   ├── core_graph.rs        # G_Y 并查集连通分量、质量最大分量、虚拟中心结点
    │   │   ├── core_fit.rs          # 6×6 最小二乘 (a_c,θ_c)、v_sl、核退化处理
    │   │   └── diag_plastic.rs      # ε̄_p、Φ_p（只读诊断，不反馈，B6）
    │   ├── surface/
    │   │   ├── metric.rs            # n、n·e_r、j、Dn Jacobian、对偶面积、曲率 K_ab、H（CPU 参考）
    │   │   ├── trace.rs             # 命题 10.2 表面速度迹；10.2.3 曲面剪切率 γ̇_Γ；δ_na
    │   │   ├── friction.rs          # 10.3：p_ref、p_{*,phys/ref}、μ_d、μ_eff、φ_i、sin²φ
    │   │   └── mode_c.rs           # J_ero、J_dep（归一化核 K）、ρ_s 半拉格朗日输运(9.2)
    │   ├── stepping/
    │   │   ├── timestep.rs          # Δt_n = min{CFL,C,char,ω,max,next}（10.4.1）
    │   │   └── stage.rs             # 阶段平衡 2.15(c)、跳步逻辑、终止判据
    │   ├── geometry_update/
    │   │   ├── predict.rs           # d_pred（B4）、节点分类 L/D/A
    │   │   ├── socp.rs              # Clarabel 问题构造/求解/解包（10.4.3）
    │   │   ├── accept.rs            # 外层非线性验收；信赖域缩小；R^(n+1)
    │   │   ├── mass.rs              # 𝓜(R,ρ)、D𝓜 节点权（对偶立体角×ρ R²）
    │   │   └── realign.rs           # 8b：质心、惯量、主轴 Q、核中心同步、射线重采样
    │   ├── accept/
    │   │   ├── step_checks.rs       # 每步硬验收 (a)(b)(c)、坡度、质量；O(N) 廉价量
    │   │   └── status.rs            # StepStatus（Event）→ AppExit 退出码映射
    │   ├── snapshot.rs              # .snap 编解码、续算（resume）：SaveSnapshot 事件系统 / 从快照重建 World 状态
    │   ├── schedule.rs              # 主循环（10.4.4 步骤 0–9）= StepPhase 状态机 + ScfSet 链 + ComputeTask 轮询 + exit_on_status
    │   ├── reserved/
    │   │   └── fluid_equilibrium.rs # 模式 A 接口（W0 求根、R_eq 求解）：保留、不接入闭环（C6）
    │   ├── shaders/                 # ★ Tier 2 WGSL
    │   │   ├── common/{types.wgsl, sh.wgsl, jacobi.wgsl}
    │   │   ├── surface_metric.wgsl      # 法向、n·e_r、j、Dn、对偶面积、曲率
    │   │   ├── radius_sample.wgsl       # 由 SH 系数在求积方向合成 R,∇R,ΔR
    │   │   ├── strain_eval.wgsl         # D(u) 与 v 在体内/分类点的合成（GEMV）
    │   │   ├── stokes_assemble_k.wgsl   # K(η)=Dᵀ diag(2ηw) D 分块 tile 装配（df32 累积）
    │   │   ├── stokes_load.wgsl         # F_α：ρ_K g_eff·B_α 与 T_tide:D(B_α) 求积
    │   │   ├── poisson_assemble.wgsl    # 压力刚度 K_p 与右端装配
    │   │   ├── surface_trace.wgsl       # 表面速度迹(10.2) + 曲面剪切率 γ̇_Γ + 摩擦链
    │   │   ├── yield_eval.wgsl          # τ_flow、F_δ、η 在 X_Y/X_vol 点上的逐点计算
    │   │   ├── mode_c.wgsl              # J_ero 与 ρ_s 半拉格朗日反向追踪+重心插值
    │   │   └── socp_prep.wgsl           # SOCP 系数：Dn 稀疏条目、g×n̂ 仿射项、质量行 D𝓜
    │   └── diagnostics/             # #[cfg(feature="diagnostics")]（§6）
    │       ├── cpu_verify.rs        # --verify-cpu：f64 CPU 重算 K、Poisson、Picard 并逐项比对
    │       ├── error_budget.rs      # --error-budget：E_total^qs 各分量（MS §9.8）
    │       ├── k8_experiment.rs     # E_{4→8}、L_max 加密、PCG 迭代数与 η 对比度
    │       ├── analytic_sphere.rs   # 球对称解析/制造解验证
    │       ├── energy_checks.rs     # 质量/刚体动量/耗散率 Φ_p 恒等式
    │       ├── socp_verify.rs       # KKT/约束违反/等价 QP 对拍
    │       └── bench.rs
    │
    ├── asteroid_app/                # ───────── Tier 3 ─────────
    │   ├── mod.rs                   # AsteroidPlugin（Bevy Plugin 聚合；与 ScfPlugin 同 App/World，直接 Query 其组件）
    │   ├── resources.rs  events.rs  # 仅 Tier 3 自有的 Resource/Event（视图选项、覆层选择、相机）
    │   ├── render/
    │   │   ├── surface_mesh.rs      # 表面网格（固定拓扑，顶点缓冲每步更新半径）
    │   │   ├── terrain_deform.rs    # GPU 顶点着色形变（R_i 缓冲 → 位置/法向）
    │   │   ├── field_overlay.rs     # 标量覆层 (F_δ, p̃, η, γ̇, φ, ρ_s, χ_Y, |v_sl|)
    │   │   ├── vector_overlay.rs    # 矢量覆层 (v_sl 切向、g_eff 、法向)
    │   │   ├── interior_slice.rs    # 内部切片：X_Y 上的 F_δ/η/p̃ 截面
    │   │   └── debug_lines.rs
    │   ├── ui/
    │   │   ├── panels.rs            # egui：参数面板（只读 TOML 派生，补丁式修改）
    │   │   ├── timeline.rs          # 时间轴：自转表 ω(t) 曲线、当前 t_n、播放/暂停/回放
    │   │   ├── diagnostics_view.rs  # 步状态、Picard 迭代数、可动面积分数、质量残差曲线
    │   │   └── camera.rs            # 自由视角（轨道/飞行）
    │   ├── shaders/                 # ★ Tier 3 WGSL（渲染）
    │   │   ├── surface_deform.wgsl  # 顶点：x = R_i ω̂_i 、法向重算
    │   │   ├── surface_render.wgsl  # 光照 + 覆层混合
    │   │   ├── field_overlay.wgsl   # 色图映射（线性/对数/发散色图）
    │   │   ├── vector_glyph.wgsl    # 箭头/流线
    │   │   └── debug_lines.wgsl
    │   └── playback.rs              # PlaybackPlugin：.snap 序列回放（无仿真器模式，写入同一批输出组件）
    │
    ├── bin/{rtfp_cli.rs, scf_cli.rs, asteroid_app.rs}   # 入口：仅解析参数 + 组装 App（§1.6 表）+ app.run()
    └── wasm/                        # #[cfg(target_arch="wasm32")]（§7，v1 仅骨架）
        ├── entry.rs  asset_fetch.rs  canvas.rs  capability.rs
```

## 2.3 WGSL 组织规则

1. 每层的着色器放在**本层** `shaders/` 下，用 `include_str!` 编入；共享片段放 `shaders/common/`，由 `common::gpu::shader_compose` 在**创建管线时**拼接（WGSL 无 `#include`）。
2. 所有跨 CPU/GPU 的结构体在 `common/gpu/pod.rs` 与对应 `types.wgsl` 中**成对定义**，并有单元测试校验 `size_of`/`offset_of` 与 WGSL 声明一致（用 `naga` 反射或手写常量表）。
3. 可调常量（workgroup 大小、$K=4$、直接/解析分支）使用 WGSL `override` 常量，管线创建时注入，避免运行期分支。
4. 任何 Tier 1/2 的 WGSL 不得依赖 Tier 3；Tier 3 的渲染着色器不做物理计算，只读取 Tier 2 写入 ECS 组件/GPU 缓冲资源的数据。

---

# 3. 数据契约与通信协议

## 3.1 单位与数值约定

- **外部（TOML/CLI/文件）一律 SI**，且字段名带单位后缀或有 `[units]` 声明；`common::units::UnitSystem` 是单位转换的唯一入口（不得在其他位置出现隐式 km→m 等换算）。
- **CPU 权威状态为 f64**（SI）。GPU 内核运行在**无量纲内部单位**（$L_0=R_{\rm ref}$ 平均半径，$\rho_0$，$G$ 归一），输入输出在边界处缩放，以保证 f32 动态范围。
- GPU 为 f32（WGSL 无 f64）。f32 精度策略 `PrecisionPolicy`：求积累加用 pairwise/Kahan 或 double-f32（df32）；跨块归约在 CPU f64 完成；MS §10.3 要求的 $p_{\min}$、$\mu_{\rm eff}$ 分母等敏感判断必须在 f64 复核（见 §4.5）。
- 实球谐正交归一；DOF 顺序"球谐指标优先、8 个 $(X,k)$ 连续"（MS §9.1），$\mathbf S=\mathbf S_1\otimes\mathbf I_8$。

## 3.2 项目 TOML Schema（Tier 2 全量；Tier 1 仅读 `[geometry]/[density]/[rtfp]/[gpu]/[units]`）

所有标 **必填** 的字段缺失 → `ScfError::ModelNotClosed { missing: [...] }`（C8）。无任何隐式默认值（括号内为样例推荐值，非默认）。

```toml
[project]
name = "ryugu_scf"

[units]                                    # 必填
length = "m"                               # 文件内坐标/半径单位
density = "kg/m3"
time = "s"

[geometry]                                 # 必填
source = "point_cloud"                     # point_cloud | glb | mesh_bin
path = "../point_cloud/sample_asteroid.xyz"
node_directions = { kind = "icosphere", subdivisions = 5 }   # 固定节点方向 ω̂_i（拓扑永久固定）
reconstruct = "radial_raycast"             # 以原点(初始质心)为中心的射线重采样；要求星形
star_min_cos = 0.2                         # c_0：n·e_r ≥ c_0 > 0（MS A.2）
closed_surface_required = true

[density]                                  # 必填；ρ_K = Σ w_k [1+σ_k²|x−c_k|²]^{−α_k}
kernels_from = "../density/cauchy_elliptic.toml"   # 或内联 [[density.kernels]]
require_nonnegative = true                 # 在体内/表面采样验证 ρ_K ≥ 0
mass_target = 4.5e11                       # M（kg）；或 mean_density（二选一，不可并存）

[spin_schedule]                            # 必填（C2）：三选一
kind = "stages"                            # stages | linear | table
stages = [ { omega = 2.8e-4, duration = 3.0e11 }, { omega = 4.0e-4, duration = 5.0e11 } ]
# linear: omega0 = ..., domega_dt = ..., t_end = ...
# table : t = [...], omega = [...]（t 严格递增，分段线性）

[material]                                 # 必填：MS 附录 A.1 / §4 / §10.3 全部常量
nu0 = 0.6                # 密实度 ν0
Z0 = 6.0                 # 配位数
rho_p = 2650.0           # 颗粒密度 → rho_bulk = rho_p*nu0
d_p = 0.05               # 颗粒直径
cohesion_c = 0.0         # c(ν0,Z0)
mu_s = 0.38              # μ_s(ν0,Z0)  （μ_y ≡ μ_s）
mu_2 = 0.64
I_0 = 0.279
C_c = 1.0                # 内聚压力系数 → p_c = C_c Z0 ν0 F_H / d_p²
F_H = 0.0
C_T = 1.0                # 潮汐背应力系数
regularization = "bingham"   # bingham | papanastasiou（全局二选一，不得混用）
gamma_eps = 1.0e-9       # γ̇_ε
delta_Y = 0.05           # δ_Y ∈ (0,1)
p_min = 1.0e-6           # p_min > 0
mu_proj_mode = "dynamic" # dynamic | static （必须显式；不改变屈服分类）
h_s = 1.0                # 活动表层厚度

[mode_c]                                   # 必填（模式 C 参数）
k_e = 1.0e-6             # 侵蚀系数 (s^-1)
sigma_0 = 1.0e2          # 屈服超额尺度 (Pa)
deposition_kernel = { kind = "normalized_surface_kernel", radius = 0.1 }  # 无量纲弧长半径（以 R_ref）
J_ext = 0.0              # 外部物料源（须显式计入质量预算）

[galerkin]                                 # 必填
K = 4                    # 锁定；其他值 → 拒绝（K=8 仅可由 diagnostics 的 k8_experiment 内部使用）
L_max = 16
L_p = 16
K_p = 4
radial_quad = 12         # N_r（Gauss–Jacobi 节点数）
angular_quad = { kind = "gauss_legendre_x_uniform_phi", exact_degree = 40 }
geometry_repr = "sh_smoothed"    # sh_smoothed | facet_exact（§4.7）；L_geo ≥ L_max
L_geo = 24

[stokes]
track = "A"              # A（基准）| B1（需 eps_B_threshold；超限自动回退 A）
pcg = { tol = 1e-10, max_iter = 500 }
assembly_precision = "gpu_df32"  # gpu_df32 | cpu_f64
gamma_AL = "auto"        # = ‖diag K‖∞ / ‖diag BᵀB‖∞
newton = false           # 缺省 Picard

[picard]
eps_pic = 1e-6  eps_u = 1e-30  eps_p = 1e-30
k_max = 60
theta_eta = 1.0
continuation = { factors = [10.0, 3.0, 1.0] }   # 10γ_ε→…→γ_ε 逐级
[accept]                                   # 每步硬验收阈值（MS 2.15）
eps_mom = 1e-6  eps_lambda_tol = 1e-6  eps_F = 1e-30
eps_slope = 1e-4  eps_M = 1e-8
eps_in = 1e-2            # Ek^{-1} 阈值（准静态假设）
res_rot_warn = 1e-2  delta_na_warn = 1e-2  m_spur_tol = 1e-6

[time]
T_end_from_spin = true   # T_end 取自 spin_schedule
theta_CFL = 0.5  theta_C = 0.5  theta_ch = 0.5
eps_omega = 0.05  omega_ref = 1e-5  dt_max = 1.0e10
lambda = 1.0             # 欠松弛 λ ∈ (0,1]

[socp]
weight = "dual_area"     # w_i>0：对偶面积 / uniform
varrho = 1.0e3           # 精确罚因子 ϱ
eps_n = 1e-3             # 防倒悬裕度 ε_n ∈ (0,1)
trust = { radius0 = 1e-3, norm = "linf", shrink = 0.5, min = 1e-8 }   # 单位：R_ref
solver = { max_iter = 200, tol_gap = 1e-9, tol_feas = 1e-9 }
[frame]
eps_frame = 1e-3  eps_deg = 1e-6          # 重对齐触发/近简并阈值

[rtfp]                                     # Tier 1 参数
backend = "gpu"                            # gpu | cpu_rayon
directions = { kind = "lebedev", count = 302 }
radial_quadrature = 16                     # 一般 α 的 Gauss–Jacobi 节点数
near_field_radius = 0.01                   # r0（无量纲）
finite_part_scale = 1.0                    # ℓ0 > 0
d_near = 0.02  d_far = 4.0                 # 混合调度阈值（无量纲）；d_far=inf 关闭远场
G = 6.67430e-11

[gpu]
power_preference = "high_performance"
max_storage_binding_mb = 256
prefer_subgroups = true                    # 仅用于已验证的归约替换

[output]
dir = "out/ryugu"
snapshot_every = 50                        # 地质步数
snapshot_fields = ["radius","rho_s","eta","p","F_delta","chi_Y","v_sl"]   # 精简列表可省空间
log = "series.jsonl"
```

**校验清单（`config::validate`，Tier 1 子集标 ★）**

1. ★ $\sigma_k>0$、$\alpha_k>0$、点云非空、闭合、定向一致、无自交；★ $\ell_0>0$、$r_0>0$、`directions.count` 在 Lebedev 表中。
2. $\rho_K\ge0$ 在体内/表面/X_Y 采样上成立（`require_nonnegative`）。
3. $K=4$；$L_p\le L_{\max}$、$K_p\le K$；$\delta_Y\in(0,1)$；$p_{\min}>0$；$\gamma_\varepsilon>0$；$\lambda\in(0,1]$；$\theta_{\rm CFL}\le0.5$；$\theta_C,\theta_{\rm ch}<1$。
4. `spin_schedule`：stages 的 $T_j>0$；linear 要求 $\omega(t)\ge0$；table 要求 $t_k$ 严格递增。
5. `mass_target` 与 `mean_density` 恰好给一个；初始 $|\mathcal M(V^{(0)})-M|/M$ 超容差则报告并拒绝。
6. `star_min_cos` $>0$ 且初始星形性成立（所有三角形 $n\cdot e_r\ge c_0$）。
7. `regularization` 全局唯一；`mu_proj_mode` 必须出现。
8. 无外部源时 $\int J_{\rm dep}=\int J_{\rm ero}$ 在核归一化下按构造成立（单元测试覆盖）；`J_ext≠0` 时质量目标按 MS §2.8 更新。

## 3.3 几何与场数据结构

```rust
/// 星形表面：节点方向与拓扑在整个仿真中固定，只有半径 R_i 演化。
/// 整张网格是 Asteroid 实体上的**一个 Component**，内部全部为连续 SoA（禁止每顶点一个 Entity）
#[derive(Component)]
pub struct StarMesh {
    pub dirs: [Vec<f64>; 3],       // ω̂_i ∈ S²，单位向量分量 (x,y,z)   (各 N_S)
    pub radius: Vec<f64>,          // R_i > 0                (N_S)
    pub tris: [Vec<u32>; 3],       // 固定拓扑，外向定向，三个顶点索引列      (各 N_T = 2N_S − 4)
    pub nbrs: CsrAdjacency,        // 顶点 1-环（SOCP 的 Dn 稀疏结构、图 G_Y 邻接）
}
// 顶点位置 x_i = R_i ω̂_i；质心重对齐后 dirs 与 radius 同步重采样（§4.6）。

#[derive(Component)]
pub struct PointSet { pub x: Vec<f64>, pub y: Vec<f64>, pub z: Vec<f64> }   // SoA
#[derive(Component)]
pub struct FieldBatch {
    pub u: Option<Vec<f64>>,               // U
    pub grad: Option<[Vec<f64>; 3]>,       // ∇U
    pub hess: Option<[Vec<f64>; 6]>,       // Hxx,Hyy,Hzz,Hxy,Hxz,Hyz
    pub flags: Vec<u8>,                    // bit0 inside; bit1 on_surface; bit2 near_field; bit3 fp_used
}
#[derive(Resource)]
pub struct VolumeIntegrals { pub volume: f64, pub mass: f64, pub com: DVec3, pub inertia: DMat3 }
```

Tier 2 的主状态（原 `SimState`）拆分为 `Asteroid` 实体上的**组件**与全局**资源**；其并集可完整序列化为 `.snap`，因而可**从任意快照重放任一地质步**（以快照重建 `World` 中对应组件/资源即可），这是 §6 离线误差预算的基础：

| 字段 | ECS 载体 | 内容 |
|---|---|---|
| `t_n`, `n`, `omega_n` | Resource `StepClock` | 地质时间、步号、冻结自转 |
| `mesh: StarMesh` | Component `StarMesh` | $R^{(n)}$ |
| `kernel_centers` | Component `KernelCenters`（SoA：`[Vec<f64>;3]`）| 当前体坐标系下的 Cauchy 核中心 $\mathbf c_k$（随重对齐同步旋转/平移）|
| `rho_s: Vec<f64>` | Component `SurfaceFields.rho_s` | 表层面密度 $\rho_s$（节点）|
| `eps_p_bar: Vec<f64>` | Component `YieldDiagnostics` | $\bar\varepsilon_p$（X_Y 点，只读诊断，B6）|
| `u0: Vec<f64>`, `p0: Vec<f64>` | Component `ModalState` | 热启动：上一步收敛的 $\mathbf u$、压力系数（重对齐后 `u0 = 0`）|
| `Q_cum: DMat3` | Component `FrameState` | 累计重对齐旋转（输出到原始惯性系用）|
| `last: StepRecord` | Resource `StepRecord` | 上一步的 Picard 迭代数、$\varepsilon_{\rm mom}^{\rm weak}$、$\varepsilon_\lambda$、$\Delta t_n$、可动面积分数、$s_{M*}$、受拉比例、$\mathrm{Ek}^{-1}$、$\varepsilon_{\rm Euler}$ 等 |

## 3.4 二进制容器 `.aevb`（网格/场/快照共用，支持文件与管道）

小端；头部 64 字节对齐；数据段 64 字节对齐（便于 `bytemuck::cast_slice` 与 mmap）。

```text
offset  size  field
0       8     magic   "AEVBIN\0\1"            (最后字节=格式主版本)
8       4     kind    1=mesh 2=points 3=field 4=snapshot 5=series_index
12      4     flags   bit0=f32 bit1=f64 bit2=crc32 present
16      8     meta_len  (UTF-8 TOML 元数据长度)
24      4     n_sections
28      4     reserved
32      …     meta (TOML：单位、坐标系说明、生成命令行、项目哈希、版本、verification_level)
…       …     section table: [name[16], dtype u32, ndim u32, shape[4] u64, offset u64, byte_len u64, crc32 u32] × n
…       …     data sections (64B aligned)
```

**`kind=field`（Tier 1 输出）** 的 section：`points.{x,y,z}`(f64)、`U`(f64)、`grad.{x,y,z}`、`hess.{xx,yy,zz,xy,xz,yz}`、`flags`(u8)。meta 含：`role`、`unit_system`、`backend`、`precision`、`directions`、`r0`、`ell0`、`omega`（若合成了有效量）、网格哈希、密度哈希。
**`kind=snapshot`（Tier 2 输出）**：meta 含 `t_n, n, omega_n, verification_level="production"|"certified"`；sections 为 `StarMesh`（dirs/radius/tris）、`rho_s`、`kernel_centers`、`u0/p0`、选定场（`eta, p_tilde, F_delta, chi_Y, v_sl` 于 X_Y 或表面节点）、`Q_cum`。

**管道**：`-` 表示 stdin/stdout；流式读取时容器头自描述，因此
`rtfp-cli eval-field … --out - | scf-cli …` 与 `cat x.field.bin | …` 均成立。纯文本辅助输出（`series.jsonl`，每步一行 JSON）供绘图与回归比对。

**GPU 缓冲区格式**（Tier 1，全部 `#[repr(C)] + bytemuck::Pod`；SoA 避免非合并访存）

| 缓冲 | 布局 | 绑定 |
|---|---|---|
| `tri_v0/v1/v2` | 各 `array<vec4<f32>>`（xyz + 预计算 ‖面法向‖ 的倒数） | storage RO |
| `tri_normal` | `array<vec4<f32>>`（xyz + 面积） | storage RO |
| `bvh_nodes` | `array<BvhNode>` 32B：`min:vec3<f32>, left_or_first:u32, max:vec3<f32>, count_flags:u32` | storage RO |
| `kernels` | `array<Kernel>` 32B：`c:vec3<f32>, w:f32, sigma:f32, alpha:f32, mode:u32(0=α=1解析,1=GJ), _pad` | storage RO |
| `dirs` | `array<vec4<f32>>`（xyz + 求积权 $w_q$） | storage RO |
| `points` | `array<vec4<f32>>`（xyz + 点类别/标志）| storage RO |
| `ray_count` | `array<u32>`，长度 $N_x N_\Omega$ | storage RW |
| `ray_offset` | `array<u32>`，exclusive scan 结果 | storage RW |
| `hits_t` | `array<f32>` 压缩交点距离（每射线内升序）| storage RW |
| `segments` | `array<vec2<f32>>` $[R^-,R^+]$，及 `seg_offset` | storage RW |
| `partials` | `array<FieldPartial>`（$U$, $\nabla U$(3), $H$(6) 共 10×f32，40B→按 48B 对齐）| storage RW |
| `field_out` | 同 `FieldPartial`，每点一条 | storage RW → 回读 |
| `params` | uniform：`n_points, n_dirs, n_kernels, r0, ell0, G, flags, chunk_offset` | uniform |

分块：一次分派的 $N_x^{\rm chunk}$ 由 `max_storage_binding_size` 与 `ray_count`/`hits_t` 容量联合反解（预估 $L$ 取 BVH 预扫描的 99 分位 +溢出重试标志），保证任何缓冲不超限。

## 3.5 CLI 子命令规范

所有 CLI 共通：`--project <toml>`；`--log-level`；`--threads N`（rayon）；`--device auto|<name>`；退出码见 §3.6。长耗时命令支持 `--resume`。每个 CLI 是一个纯 Bevy Headless `App`（§1.6 表）：`clap` 解析结果写入 `ProjectConfig` 等资源，子命令对应的 `*CliPlugin` 发出事件并在完成时发 `AppExit`；CLI 参数规范本身不因此改变。

### 3.5.1 `rtfp-cli`（Tier 1，独立可用）

```text
rtfp-cli import-mesh   --input <xyz|glb|mesh.aevb> --dirs-subdiv 5 --out <mesh.aevb>
rtfp-cli eval-field    --project P.toml --mesh <mesh.aevb|-> --points <points.aevb|--surface-vertices|--surface-faces
                                                        |--grid xmin,xmax,nx,…>
                       [--role vertex|face|volume|classify] [--want u,grad,hess]
                       [--omega W]            # 给定则同时输出 W、g_eff、H_W
                       [--backend gpu|cpu_rayon] --out <field.aevb|-> [--format aevb|csv]
rtfp-cli integrals     --project P.toml --mesh M --out <json|toml>      # 体积、质量、质心、惯量
rtfp-cli ray-radii     --mesh M --origin x,y,z --dirs <file> --out …
rtfp-cli info          <file.aevb>                                       # 打印头/元数据
# 以下仅在 --features diagnostics 构建中存在（§6）：
rtfp-cli verify        --project P.toml --mesh M [--verify-cpu] [--analytic-sphere]
                       [--polyhedron-exact] [--feynman] [--laplace-checks] [--roundoff]
rtfp-cli bench         --project P.toml --mesh M --points N --repeat R
```

示例：`cargo run --features tier1 --bin rtfp-cli -- eval-field --project assets/projects/sample_project.toml --mesh out/mesh.aevb --surface-faces --want grad,hess --omega 2.8e-4 --out out/face.field.aevb`

### 3.5.2 `scf-cli`（Tier 2，纯无头）

```text
scf-cli run-sim   --project P.toml [--out-dir D] [--resume <snap>] [--max-steps N] [--until-t T]
                  [--backend gpu|cpu_rayon] [--snapshot-every K] [--progress]
scf-cli resume    --snapshot S.aevb [--project P.toml]
scf-cli inspect   <snap|series.jsonl> [--fields …]            # 打印摘要/导出 CSV
scf-cli export    --snapshot S --format {obj,ply,glb,csv} --out F
scf-cli validate  --project P.toml                             # 仅模型闭合性/配置检查，不求解
# 以下仅在 --features diagnostics 中存在：
scf-cli run-sim   … --verify-cpu [--verify-every K]            # 每 K 步 CPU 对拍（默认不存在）
scf-cli error-budget --snapshots D --steps 0,100,200 [--k8] [--lmax-refine] [--out report.toml]
scf-cli benchmark --project P.toml [--stage rtfp|stokes|poisson|socp|all]
scf-cli verify    --suite {basis,stokes,poisson,sphere,socp,realign,all}
```

`run-sim` 的输出目录：`snap_{n:06}.aevb`（按 `snapshot_every`）、`final.aevb`、`series.jsonl`（每步一行：$n,t_n,\omega_n,\Delta t_n$，Picard 迭代数，$\varepsilon_{\rm mom}^{\rm weak},\varepsilon_\lambda$，可动面积分数，质量残差，$s_{M*}$，$\|d_*\|_\infty$，受拉比例，$\mathrm{Ek}^{-1}$，$\varepsilon_{\rm Euler}$，$\mathrm{rank}\,\mathbf B_{\rm rb}$，$\sigma_{\min}(\mathbf B_{\rm rb})$，PCG 迭代数，$\eta_{\rm hi}/\eta_{\rm lo}$ …）。

### 3.5.3 `asteroid-app`（Tier 3）

```text
asteroid-app [--project P.toml | --playback <dir|snap>] [--backend gpu] [--headless-test]
```

## 3.6 状态码与退出码

| `StepStatus` / 退出码 | 含义 | 对应数学条款 |
|---|---|---|
| 0 `Ok` / `StageEquilibrium` / `Finished` | 正常；阶段平衡（$\mathcal A=\varnothing$）为一等状态 | §2.15(c) |
| 10 `ModelNotClosed` | 缺参数，拒绝运行 | §10.4.4 步骤 0 |
| 11 `GeometryInvalid` | 初始非星形/非闭合/自交/质量不符 | A.2 |
| 20 `RheologyNotConverged` | Picard 经连续化与松弛重试仍失败 | §2.15(a)、§10.4.2 |
| 21 `MomentumResidual` | $\varepsilon_{\rm mom}^{\rm weak}$ 超限 | §2.15(b) |
| 22 `LoadsIncompatible` | 合力/合力矩不相容（重对齐后仍超限）| A.3 |
| 23 `RigidRankDeficient` | $\mathrm{rank}\,\mathbf B_{\rm rb}<6$ | A.3 |
| 24 `QuasiStaticInvalid` | $\mathrm{Ek}^{-1}>\varepsilon_{\rm in}$（标记，是否终止由配置决定）| §2.8 B1 |
| 30 `SocpFailed` | Clarabel 非最优且外层无法恢复 | §10.4.3 |
| 31 `TrustRadiusExhausted` | 缩小到下限仍倒悬/自交 | §2.13 |
| 32 `CoreDegenerate` / `NoCore` | $\mathcal U=\varnothing$（整体失稳）| §2.8 B3 |
| 40 `TransportInvalid` | 特征线折叠（$\Delta t\|\nabla\mathbf c\|\ge1$）| §9.2 |
| 50 `GpuUnavailable` / `GpuDeviceLost` | 设备问题 | — |
| 3 `AuditFailed`（仅 diagnostics）| 验证/误差预算阈值未过 | — |

`StepStatus` 以 `StepFailed` 事件发出，`exit_on_status` 系统将其映射为 `AppExit::Error(code)`（GUI 中改为在诊断面板显示并转入 `StepControl::Pause`）。

**严禁**在非零状态下写出 `verification_level` 为通过的快照；`StageEquilibrium` 与"阶段时长耗尽仍有可动集（未平衡）"是**物理结果**，不是失败，须带可动面积分数输出。

## 3.7 Tier 3 的参数补丁语义

`ParamPatch`（Event）由 `apply_patch` 系统在 `StepPhase::Idle` 消费，只允许修改**不使已有历史失效**的量（可视化选项、`snapshot_every`、`dt_max`、`theta_*`、`spin_schedule` 的未来段）；修改材料常量、几何、密度、`L_max`、`K` 等改变物理模型的参数时，UI 必须把它转换为"以当前快照为初值的新运行"，并在日志中标记 `branch_from = snapshot hash`。补丁仅在地质步边界应用。

---

# 4. 计算管线与加速设计

## 4.0 总原则

- **CPU/GPU 分工**：体积/曲面的**逐点密集计算**（求交、径向积分、度量、迹、求积装配、逐点本构）下沉 WGSL；**稀疏/串行相关/小规模稠密**环节（PCG、$8\times8$ 块分解、$6\times6$ 法方程、球谐变换、并查集、Clarabel）留在 CPU，用 `rayon` 并行。
- **一次上传、多次分派、一次回读**：每个地质步内，网格/BVH 只上传一次；同一批次的多个点集共享 BVH 与核参数；GPU 到 CPU 的回读只发生在：场数据、装配矩阵、$D(\mathbf u)$/$\eta$ 逐点量、SOCP 系数。
- **批处理与命令缓冲**：同一阶段的多个 compute pass 写入同一 `CommandEncoder`，单次 `queue.submit`；回读用 `map_async` + 阶段末同步。

## 4.1 Tier 1：RTFP WGSL Compute Pipeline

**阶段 0（每个地质步，CPU）**：由 `StarMesh` 生成三角形 SoA；构建扁平 BVH（CPU f32 化存储，构建用 f64，`rayon` 并行 binned-SAH）；上传 `tri_*`、`bvh_nodes`、`kernels`、`dirs`（`queue.write_buffer`）。无硬件 RT 依赖；硬件 RT 为可选后端（独立系统集 `FieldSet::HwRt`，以 `run_if(capability)` 门控，`FieldRequest`/`FieldBatch` 接口不变）。

**阶段 1–5（GPU，对每个点块）**，与 MS §3.6–3.8 逐条对应：

| Pass | WGSL | dispatch | 线程映射 | 输入→输出 | 要点 |
|---|---|---|---|---|---|
| P1 | `ray_count.wgsl` | $\lceil N_xN_\Omega/64\rceil$ | 1 invocation = 1 射线 | 点、方向、BVH → `ray_count[]` | 射线 $r(t)=x-t\,u_q$；只数交点；栈式 BVH 遍历，栈深 32（`var<private>`）|
| P2 | `prefix_sum.wgsl` | 3 遍 | workgroup 内 Blelloch，块和再扫描 | `ray_count`→`ray_offset` | 通用 `common::gpu::scan` 复用 |
| P3 | `ray_write.wgsl` | 同 P1 | 同 P1 | → `hits_t[]` | 重放射线，写入后按 $t$ 插入排序（每射线交点数小）；与 P1 使用**逐位相同**的求交代码以避免计数/写入不一致 |
| P4 | `segment_build.wgsl` | 同 P1 | 同 P1 | `hits_t`+内外标志 → `segments[]` | 交点数奇偶判定 $x\in\operatorname{int}V$：内部 $[0,R_1],[R_2,R_3],…$；外部 $[R_1,R_2],…$；边界点走近场分支（不套配对规则，MS §3.6）|
| P5 | `rtfp_integrate.wgsl` | $N_x$ 个 workgroup（一个点批次）| 1 invocation = 1 方向，循环该射线全部交段与核 $k$ | segments、kernels → `partials` | **融合 kernel**：$\alpha_k=1$ 走解析原函数（`override` 常量选择分支，无运行期发散）；一般 $\alpha$ 走固定节点 Gauss–Jacobi；同时累加 $U$、$\nabla U$、$H$（六分量，$T(u)=3uu^\top-I$）；workgroup 内共享内存分层归约；`prefer_subgroups` 时在已验证范围内用 subgroup 规约替换 |
| P6 | `near_field.wgsl` | 仅 `near` 标志点 | 同 P5 | 同 P5 | $R<r_0$ 部分：密度差分项 $\int[\rho(x-Ru)-\rho(x)]/R\,dR$ 固定求积，$R=0$ 节点取极限 $-\nabla\rho\cdot u$（绝不做 $0/0$）；显式加 $G\rho C(x)\log(r_0/\ell_0)$ 与接触项 $-\tfrac{4\pi G}{3}\vartheta_x\rho I$；**GPU 不计算含 $\log\varepsilon$ 的发散项** |
| P7 | `reduce_field.wgsl` | $N_x$ | 1 invocation = 1 点 | `partials`→`field_out` | 跨方向批最终归约；写 10×f32 与标志 |
| P8 | `far_field.wgsl`（预留）| $d>d_{\rm far}$ 的点 | — | — | v1 提供 `FarFieldSet` 占位系统集（`run_if` 恒假）与"直接法回退"；多极实现要求先补齐 MS §3.10 所述的多重指标/归一化/平移算子/余项界，未补齐前不得启用 |

混合调度（`backend/dispatch.rs`，MS §3.11）：以 BVH 上的点到几何最短距离 $d(x,V)$ 将点集划分为 $d<d_{\rm near}$（RTFP+有限部分）、$[d_{\rm near},d_{\rm far}]$（RTFP 全射线）、$d>d_{\rm far}$（远场）三个子批，分别分派。

**点集角色决定规则（`PointRole`）**

| 角色 | 需要 | 数学要求 | 备注 |
|---|---|---|---|
| `SurfaceVertex` | $U,\nabla U$ | $g_i=\nabla U+\omega^2(x,y,0)$ | 梯度跨表面连续，无需单侧规则 |
| `SurfaceFace` | $\nabla^2U$（面心）| 面片内点 $C_{ij}=0$，$H^{\rm FP}$ 与 $\ell_0$ 无关；单侧极限平均 $=H^{\rm FP}$ | **顶点/棱 Hessian 禁止进入 $T_{\rm tide}$**（MS §10.3）；节点值取入射面心的面积加权平均（工程选择，写入元数据）|
| `VolumeQuad` | $\nabla U$ | 内部点经典收敛，无需有限部分 Hessian | 步骤 1 必须在 $\mathcal X_{\rm vol}$ 上额外计算（MS §8.7）|
| `ClassifyGrid` | $\nabla U,\nabla^2U$ | 体内点经典收敛 | $\zeta=1$ 层点复用 `SurfaceFace` 面平均 |

**CPU 后端 `FieldSet::CpuRayon`**：同一算法的 f64 实现，`par_chunks` 按点并行，方向循环内无分配；用途：无 GPU 环境、GPU 结果的对拍靶、`--verify-cpu`。它是 **Tier 1 生产可选后端**（`rtfp.backend = "cpu_rayon"`），与 `diagnostics/` 中**独立实现**的参考（Feynman 通道、多面体精确、自适应求积）是两件事。

**Tier 1 CPU Rayon 热点**：BVH 构建、点云/网格预处理（法向、去重、射线重采样）、`integrals`（体积/惯量求积）、CPU 后端本体、GPU 回读后的 f64 转换与分块汇总。

## 4.2 Tier 2：每个地质步的 GPU/CPU 分工总表

| 阶段（MS §10.4.4）| 内容 | 位置 | 说明 |
|---|---|---|---|
| 1 正演 | 四类点集场数据 | Tier 1（GPU）| `FieldRequest::Eval` ×4（四个点集实体），共用同一次 `FieldRequest::Bind` |
| 1′ 几何量 | $n_i$、$n\cdot e_r$、$j$、$Dn$ Jacobian、对偶面积、曲率 | **GPU** `surface_metric.wgsl` | 逐顶点 1-环；固定拓扑，邻接 CSR 常驻 |
| 1″ $R_h,\nabla R_h,\Delta R_h$ | 求积方向上的星形函数 | **GPU** `radius_sample.wgsl` | SH 综合（$N_{\rm quad}\times N_{\ell m}$ GEMV）；SH 分析在 CPU rayon（$N_S\to N_{\ell m}$，一次/步）|
| 2-P1 | Poisson 右端、刚度装配 | **GPU** `poisson_assemble.wgsl` | $\nabla\cdot\tau_{\rm flow}=\eta\Delta v+2D\nabla\eta$ 需逐点 $\Delta v$、$\nabla\eta$：`strain_eval.wgsl` |
| 2-P1 | Poisson 求解 | **CPU rayon** PCG | $\mathbb Q_h^0$ 的块对角预条件；$(L_p+1)^2K_p$ 量级的 SPD 系统 |
| 2-P2 | 逐点 $\dot\gamma,I_c,\mu_d,\eta$ | **GPU** `yield_eval.wgsl` | f32 计算，$p_{\min}$ 等敏感分母 f64 复核（§4.5）|
| 2-P3 | $\mathbf K(\eta)$、$\mathbf F$ 装配 | **GPU** `stokes_assemble_k/load.wgsl` | tile-GEMM 形式 $\mathbf K=\mathbf D^\top\mathrm{diag}(2\eta w)\mathbf D$，df32 累积，回读后转 f64 |
| 2-P3 | $\mathbf B_{\rm rb}$（$6\times N_u$）| **CPU rayon** | 轻量，$f64$；同时给秩/最小奇异值 |
| 2-P3 | 值域空间法 7 次 PCG、$6\times6$ Schur | **CPU rayon** | 稠密 $\mathbf K_\gamma$ matvec 行并行；预条件子 $\mathbf K_{\gamma,B1}^{-1}$ 每 $(\ell,m)$ 一个 $8\times8$ Cholesky，块并行 |
| 2-P4 | 判据 (a)(b)、$\varepsilon_\lambda$ | CPU | 一次 matvec 级别 |
| 3 屈服分类 | $F_\delta$、$\tau_{\rm flow}$、$\chi_Y$ 逐点 | **GPU** `yield_eval.wgsl` | $\mathcal X_Y$ 上 |
| 3 刚性核 | 图 $\mathsf G_Y$ 连通分量、质量最大分量 | CPU（串行并查集，$O(N)$）| 邻接固定，可预建；质量权 GPU 预算 |
| 3 核拟合 | $6\times6$ 法方程 | CPU rayon reduce | |
| 4 摩擦链 | $\mathbf u\to$ 迹 $\to\dot\gamma_\Gamma\to I_c\to\mu_d\to\mu_{\rm eff}\to\phi_i$ | **GPU** `surface_trace.wgsl` | 仅 $\mathcal S_{\rm slope}$ 节点 |
| 5 模式 C | $J_{\rm ero}$、核 $K$ 归一化沉积、$\rho_s$ 半拉格朗日 | **GPU** `mode_c.wgsl` | 反向特征线（中点法，MS §9.2）+ 重心插值 + Jacobian 修正；沉积核的全局归一化为一次 reduction |
| 6 步长 | $\Delta t_n$ | CPU（取 GPU 回读的极值）| |
| 7 SOCP | 系数 + Clarabel | 系数 **GPU** `socp_prep.wgsl`；求解 **CPU**（单线程为主，必要时 Clarabel 多线程特性）| 变量仅限可动集 $\mathcal A$（$\mathcal L,\mathcal D$ 在预处理中代入消去）|
| 8a/8b | 几何更新、重对齐 | CPU rayon + Tier 1 `ray_radii` | 惯量/质心为 Tier 1 `integrals`；射线重采样复用 Tier 1 |
| 9 | 输出 | CPU | 异步写快照（`AsyncComputeTaskPool` 任务），不阻塞下一步 |

上表每一行对应一个 `StepPhase`/`ScfSet`（§1.6）；GPU 回读与重型 CPU 求解（PCG、Clarabel）作为 `ComputeTask` 组件由 `poll_task` 系统收口。

## 4.3 Stokes（8.4′）求解管线细节

**求积结构**：$\mathcal X_{\rm vol}$ = 角向 $N_\Omega^{\rm q}$ 点（Gauss–Legendre×等距 $\varphi$，对 $2L_{\max}$ 次多项式精确，MS §9.4）× 径向 $N_r$ 个 Gauss–Jacobi 点（权 $\zeta^{2\ell+2}$）。径向节点按 $\ell$ 分组复用（按 $\ell$ 的 Jacobi 权不同；实现取 $\max$-阶权一组节点 + 多项式精确度校核，单测验证 $\tilde N_\ell,\tilde P_\ell$ 与解析一致）。

**GPU 装配（`stokes_assemble_k.wgsl`）**：按 $\alpha,\beta$ 的 $(16\times16)$ tile 分派；每个 tile 循环求积点块，在共享内存中即时求 $\mathbf D(\mathbf B_\alpha)$（引理 5.5 的解析式，**逐点解析、不预存六分量大矩阵**），乘 $2\eta_q w_q$ 累积，使显存为 $O(N_u^2)$ 而非 $O(N_qN_u)$。累积使用 df32；对称性只算上三角。
**载荷** $F_\alpha=\int\rho_K\mathbf g_{\rm eff}\cdot\mathbf B_\alpha-\int\mathbf T_{\rm tide}:\mathbf D(\mathbf B_\alpha)+\oint\mathbf t_{\rm ext}\cdot\mathbf B_\alpha$，同一内核循环完成。
**带状稀疏**：$R_h$ 与 $\rho_K$ 带限时（Gaunt 条件，MS 命题 8.9）块 $(\ell,m;\ell',m')$ 在 $|\ell-\ell'|>L_w$ 时置零；$N_u\le2500$（$L_{\max}\le16$）时仍用稠密存储，超过则 CSR 块稀疏。
**求解**：`gamma_AL` 自动取 $\|\mathrm{diag}\,\mathbf K\|_\infty/\|\mathrm{diag}\,\mathbf B^\top\mathbf B\|_\infty$；对 $\mathbf K_\gamma$ 作 7 次右端 PCG（$\mathbf F$ 与 $\mathbf B_{\rm rb}^\top$ 的 6 列；**多右端同批 PCG**，一次 matvec 同时推进 7 列以共享内存带宽）；得 $\mathbf y_0,\mathbf Y$，解 $6\times6$ SPD $(\mathbf B_{\rm rb}\mathbf Y)\boldsymbol\lambda=\mathbf B_{\rm rb}\mathbf y_0$，$\mathbf u=\mathbf y_0-\mathbf Y\boldsymbol\lambda$。
**必报量**：PCG 迭代数、$\eta_{\rm hi}/\eta_{\rm lo}$、$\mathrm{rank}\,\mathbf B_{\rm rb}$ 与 $\sigma_{\min}$（MS §8.9）；$\mathrm{rank}<6$ → `RigidRankDeficient`。
**不变量**：$\ell=0$ 绝不进入任何矩阵（A.3）；$\mathbf K$ 仅装配 $\ell\ge1$。

## 4.4 压力 Poisson（§8.7）

基 $\mathbb Q_h^0=\mathrm{span}\{(1-\zeta)\psi_{\ell k}Y_{\ell m}\}$，$\psi_{0k}=P_k^{(0,2)}$；刚度 $(\mathbf K_p)_{ab}=\int\nabla\phi_a\cdot\nabla\phi_b$ SPD，**无自由常数**（Dirichlet 数据 $p_\Gamma$ 经提升 $p_b=\sum\hat p_{\ell m}\zeta^\ell Y_{\ell m}$，$\hat p_{\ell m}$ 由 $p_\Gamma$ 节点值 SH 分析得到）。右端体内数据：$\rho_K\mathbf g_{\rm eff}$（Tier 1 `VolumeQuad` 梯度）、$\nabla\cdot\mathbf T_{\rm tide}=-\tfrac{8\pi}{3}GC_T\rho_p\nu_0d_p^2\nabla\rho_K$（解析，无需三阶导数）、$\nabla\cdot\boldsymbol\tau_{\rm flow}=\eta\Delta\mathbf v+2\mathbf D\nabla\eta$（用 Picard 上一轮 $\mathbf u^k$；$\nabla\eta$ 由链式法则，$\nabla\tilde p=\mathbf 1_{\{p>p_{\min}\}}\nabla p$）。**第一个地质步** $\mathbf u^0=\mathbf 0$ ⇒ $\boldsymbol\tau_{\rm flow}=\mathbf 0$，$p$ 为含潮汐的广义静水压。受拉截断 $\tilde p=\max([p]_+,p_{\min})$ 后，**受拉比例 $\|[p]_-\|$ 必报**，且不得声称 $\tilde p$ 仍满足 Poisson（B5）。

## 4.5 精度策略（GPU f32 与 f64 复核）

| 量 | GPU 内部 | CPU 复核/最终 |
|---|---|---|
| RTFP 累积 $U,\nabla U,H$ | f32，pairwise/df32 | 块汇总 f64；`--verify-cpu` 可对拍 |
| $\mathbf K(\eta)$、$\mathbf F$ 装配 | df32 累积 | 回读后 f64 参与 PCG（所有迭代向量 f64）|
| $\eta,\mu_d,I_c$ 逐点 | f32 | 当 $p_{*}$ 或 $\dot\gamma$ 低于 $\epsilon_{\rm f32}$ 阈值（接近 $p_{\min}$ 或 $\dot\gamma\to0$）的点标记，回读后 CPU f64 重算这些点（"危险点复核"，常为极少数）|
| $\sin^2\phi,\mu_{\rm eff}$ 的分母 | f32 | $p_{*,\rm ref}$ 取 $\max(\cdot,p_{\min})$ 并以 f64 计算 |
| Clarabel 系数与结果 | — | f64 |

## 4.6 几何更新与重对齐的实现要点（步骤 7–9）

1. **预测**：$d_{{\rm pred},i}=\big[\chi_{Y,i}\Delta t_n\mathbf v_{{\rm sl},i}\cdot n_i+(\rho_{\rm bulk}/\rho_K(\mathbf x_i))\Delta h_{C,i}\big]/(n_i\cdot e_r)$；要求 $n\cdot e_r\ge c_0$，否则该节点不得用径向参数化（报 `GeometryInvalid`）。分类 $\mathcal L,\mathcal D,\mathcal A$ 由 $\chi_Y$ 与 $\Delta h_C$ 唯一决定。
2. **SOCP 构造（`socp.rs`）**：自由变量 $d_{\mathcal A}\in\mathbb R^{|\mathcal A|}$、$t_i\ge0$（$i\in\mathcal S_{\rm slope}$）、$s_M=s_M^+-s_M^-$。
   - 目标：$\tfrac12\sum_{\mathcal A}w_i(d_i-d_{{\rm pred},i})^2+\varrho(\sum t_i+s_M^++s_M^-)$（常数项忽略）；
   - 锥：每个 $i\in\mathcal S_{\rm slope}$ 一个 4 维二阶锥 $\big((\sin\phi_i+t_i);\ (g_i\times\hat n_i(d))/|g_i|\big)\in\mathcal Q^4$，其中 $\hat n_i(d)=n_i+Dn_i\,d$ 是 $d$ 的仿射函数，**$Dn_i$ 为归一化单位法向的 Jacobian**（只涉及 $i$ 的 1-环，稀疏）；
   - 线性不等式：$g_i\cdot\hat n_i(d)\le-\epsilon_n|g_i|$（无松弛，若 $d=0$ 违反 → 外层预处理沿 $-g$ 修复或剔除该点出 $\mathcal S_{\rm slope}$）；
   - 质量等式：$D\mathcal M\,d+s_M=M-\mathcal M(R^{(n)},\rho^{(n)})$，$D\mathcal M_i=\Omega_i\rho_K(R_i\hat\omega_i)R_i^2$（$\Omega_i$ 为顶点对偶立体角；通量增量用 $\Delta h_C^*$）；
   - 信赖域：$\|d_{\mathcal A}\|\le\Delta^{(n)}$（`linf` 盒或 `l2` 锥，配置给定）；
   - $\mathcal L$ 代入 $d=0$，$\mathcal D$ 代入 $d=d_{\rm pred}$，从变量中消去；**$\mathcal A=\varnothing$ 时不构造问题、不调用 Clarabel**，$d_*=d_{\rm pred}$（测试用调用计数器断言）。
3. **外层验收**：用 $R^{(n+1)}=R^{(n)}+\lambda d_*$ 重建几何；检查正体积、无自交（BVH 自查询，rayon）、$n\cdot e_r\ge c_0$、$|\mathcal M-M|\le\varepsilon_M$。失败 → 缩小 $\Delta^{(n)}$（`shrink`）重解步骤 7，**不推进时间**；达下限 → `TrustRadiusExhausted`。**Clarabel 内层绝不缩小信赖域**；非零 $t_*$/$s_{M*}$ 如实报告，不据此判定张量屈服。
4. **重对齐 8b**（`realign.rs`）：仅在触发条件满足时执行（避免反复重采样的数值扩散），按 MS 步骤 8b(i)–(vii) 逐条实现：质心→平移（含核中心 $\mathbf c_k$）→惯量特征分解→$e_z'=a_1$、$e_x'=a_3$ 的**符号与近简并规则**（$I_1-I_2\le\varepsilon_{\rm deg}I_1$ 等）→旋转 $Q$（$\det=+1$，核中心同步旋转）→沿**固定节点方向** $\hat\omega_i$ 对新多面体做射线重采样（复用 Tier 1 `ray_radii`）→标量重心插值（$\rho_s$，诊断 $\bar\varepsilon_p$）→复核残差 $|x_{\rm cm}|\le\varepsilon_{\rm frame}R_{\rm ref}$、$|I_{xz}|,|I_{yz}|\le\varepsilon_{\rm frame}I_1$。重对齐后 `u0 = 0`；$\omega$ 不变，新标架 $\boldsymbol\Omega=\omega e_z'$。
5. **$\rho_s$ 输运**：反向特征线 $\hat\omega_d=\mathcal N[\hat\omega_a-\Delta t\,\mathbf c(\hat\omega_m,t^{n+1/2})]$（需 $t^n$ 与 $t^{n+1/2}$ 速度；若只有 $t^n$ 则按一阶使用并如实声明，MS §9.2），$\mathbf c=\chi_A\mathbf v_{{\rm sl},S}/R$；Jacobian 修正；特征线单值性校验 $\Delta t\|\nabla\mathbf c\|<1$（由 $\Delta t_{\rm char}$ 保证）；正性由 $\Delta t_C$ 保证。

## 4.7 几何表示 $R_h$（Stokes 求积用）与多面体（RTFP 用）

数学规范以 $R\in C^3$ 为 Stokes 弱式前提（命题 8.4、10.2 含 $\Delta R$），而多面体仅 Lipschitz，棱上 $\Delta_{S^2}R$ 含 δ 型奇异。软件对此的**纯数值表示**选择（不改变任何方程）：

- `geometry_repr = "sh_smoothed"`（推荐）：对节点半径 $R_i$ 作带限球谐分析（$L_{\rm geo}\ge L_{\max}$）得 $R_h$，在求积方向解析给出 $R_h,\nabla R_h,\Delta R_h$；RTFP 与质量积分使用**多面体**本身。二者差 $\|R_h-R_{\rm poly}\|_\infty$ 作为诊断 `geom_repr_gap` 每步记录（廉价），超限告警。
- `facet_exact`：用平面片解析 $R=\mathbf n\!\cdot\!\mathbf x_0/\mathbf n\!\cdot\!\hat\omega$ 及其梯度（$\Delta R=0$，棱上 δ 项被忽略），仅用于与多面体严格一致的对拍，不作生产默认。

## 4.8 Rayon 并行热点汇总

| 位置 | 并行策略 |
|---|---|
| BVH 构建、自交检测 | 分治 + `par_iter`（任务粒度阈值避免过细）|
| 球谐分析/综合 | 按 $(\ell,m)$ 或按求积点分块的 GEMM，`par_chunks_mut`；Legendre 递推以 $\ell$ 为外层，缓存 $P_\ell^m$ 表 |
| PCG：matvec | 稠密 $\mathbf K_\gamma$ 按行块 `par_chunks`；内积/范数用 `par_iter().sum()`（确定性模式下固定分块树归约，保证位级可复现）|
| 预条件子 | 每 $(\ell,m)$ 一个 $8\times8$ Cholesky/solve，`par_iter` 于块；预条件在每步 $\eta$ 更新后重建 |
| 屈服图 | 邻接固定；$\chi_Y$/分量标记并行，连通分量并查集串行（$O(N\alpha(N))$，非瓶颈）|
| 核拟合、惯量、质心 | `par_iter().map().reduce()` 累积 $6\times6$ / $3\times3$ |
| 点云预处理（法向、去重、射线重采样）| `par_iter` |
| 快照压缩/写出 | `AsyncComputeTaskPool` 任务 |
| 系统间并行 | Bevy 多线程执行器按 `Query`/`Res` 访问冲突自动调度；系统内数据并行仅经 `par.rs`，二者不得互相嵌套阻塞 |

**确定性**：`par.rs` 提供 `DeterministicReduce`（固定分块树）开关；回归测试与 `--verify-cpu` 默认开启，生产默认关闭以取最大吞吐。

---

# 5. 分阶段实施与验收测试路线图

三阶段**严格由底向上**，每阶段有"独立 CLI + 单元测试 + 集成验收门槛"三件套，**前一阶段验收门槛未全部通过，不得开始下一阶段**。所有门槛均为自动化测试（CI 可运行；GPU 用例在无 GPU 环境自动降级为 `cpu_rayon` 并标注）。

## 5.1 阶段一：Tier 1 `rtfp_solver`

### 5.1.1 开发顺序

| 步 | 内容 | 产出 |
|---|---|---|
| 1.1 | `common`：units、math、config 校验、`.aevb` 容器、`AssetIo` 资源、`par`、`GpuContext` 资源（async）、`CommonPlugin`| 基础设施单测 |
| 1.2 | `StarMesh`/icosphere/tri_mesh（闭合、定向、自交检查）、点云/glb 导入与射线重采样 | `import-mesh` CLI |
| 1.3 | BVH（CPU 构建）、射线求交 CPU 版、内外判定、交段构建 | 几何单测 |
| 1.4 | Cauchy 密度及导数、α=1 解析径向积分、Gauss–Jacobi 通用 α | 径向单测 |
| 1.5 | `FieldSet::CpuRayon`（f64）：融合 $U,\nabla U,H$、近场有限部分、体积/惯量 | `eval-field --backend cpu_rayon` |
| 1.6 | WGSL 管线 P1–P7、分块、`FieldSet::Gpu` | `eval-field --backend gpu` |
| 1.7 | 混合调度与 `FarFieldSet` 占位；`RtfpPlugin` 封装（`FieldRequest`/`FieldBatch` 契约）；`ray-radii`/`integrals` | 插件接口冻结 |
| 1.8 | diagnostics：独立参考（多面体精确、球解析、Feynman/Carlson、Laplace 检查、舍入评估）| `verify` 子命令 |

### 5.1.2 单元测试指标

| 测试 | 指标 |
|---|---|
| 单位/容器 | 往返无损（bit-exact）；CRC 篡改被检出；GPU 与 Rust Pod 结构 `size/offset` 与 WGSL 一致 |
| 射线求交 | 随机射线 vs 暴力求交：计数/距离一致（容差 $10^{-6}$ 相对）；P1 与 P3 计数一致（防写入越界）|
| 交段 | 凸体、非凸体、含空腔体、外部点、内部点：区间总长 = 暴力积分；边界点走近场分支且无 NaN |
| 径向积分 | α=1 解析 vs 高阶求积 $\le10^{-12}$；一般 α Gauss–Jacobi 随节点数指数收敛 |
| 球（解析）| 均匀球（σ→极小的单核近似常数）与同心层球：CPU f64 后端 $U$ 相对 $\le10^{-8}$、$\nabla U\le10^{-7}$（网格细分极限外推），$\mathrm{tr}H$ 满足 $-4\pi G\rho$（内）/$0$（外）$\le10^{-6}$ |
| 多面体精确（均匀密度）| 与 Werner–Scheeres 类精确公式：CPU f64 后端 $U\le10^{-9}$、$\nabla U\le10^{-8}$、$H\le10^{-7}$（场点距面 $\ge0.05L_0$）|
| 有限部分 | 表面/近表面点 $H$ 对 $\ell_0$ 的依赖仅在棱/顶点（面心点与 $\ell_0$ 无关）；$H^+-H^-=4\pi G\rho\,\mathbf n\mathbf n^\top$ 误差 $\le10^{-6}$（f64）|
| 梯度/势连续性 | 穿越表面的 $U,\nabla U$ 连续 |
| 非均匀 Cauchy | GPU vs CPU f64：$U\le10^{-5}$，$\nabla U\le10^{-4}$，$H\le10^{-3}$（相对场尺度 $\|\cdot\|_\infty$ 归一）；近表面 $\le5\times10^{-3}$ |
| 方向数收敛 | $N_\Omega=302\to590\to1202$ 误差单调下降 |
| 混合调度 | 分批结果与全射线结果在阈值处连续（差 $\le$ 各自误差界）|
| 体积分 | 体积/质量/质心/惯量 vs 解析多面体公式 $\le10^{-10}$（f64）|

### 5.1.3 集成验收门槛（Tier 1 → Tier 2 的放行条件）

1. `rtfp-cli eval-field` 在样例 `ryugu` 级网格（$\ge10^5$ 三角形）上：GPU 与 `cpu_rayon` 一致性满足上表；无 NaN/Inf；`flags` 分类与几何一致。
2. `rtfp-cli verify --all`（diagnostics 构建）全部通过。
3. `FieldRequest`/`FieldBatch` 四类 `PointRole` 数据契约冻结并有回归样例（`tests/data`）。
4. 性能（**记录不作硬门槛**，但需出报告）：给出 $10^5$ 场点 × 302 方向 × $10^5$ 三角形的 GPU/CPU 用时与 GPU 对 CPU 加速比。
5. 无 `std::fs` 出现在 `rtfp_solver/` 与 `common/`（除 `native_fs.rs`）；`tools/check_layering.sh` 通过。

## 5.2 阶段二：Tier 2 `scf_simulator`

### 5.2.1 开发顺序

| 步 | 内容 | 对应 MS |
|---|---|---|
| 2.1 | `spin.rs`（stages/linear/table，$\omega,\dot\omega,t_{\rm next}$）与步长公式 `timestep.rs` | §10.4.1 |
| 2.2 | Jacobi 基、$\sigma_{\ell k},\kappa_{\ell k}$ 表、实球谐、Gauss–Jacobi、SH 变换 | §8.1、§10.2.1 |
| 2.3 | 双势基的解析 $\mathbf D(\mathbf B_\alpha)$、质量/刚体约束矩阵 $\mathbf B_{\rm rb}$（CPU f64 参考装配）| §5、§8.3、§8.5 |
| 2.4 | $\mathbf K(\eta)$、$\mathbf F$ 装配 + 值域空间法 + B1 预条件子（先 CPU 版）| §8.4′ |
| 2.5 | Poisson 压力 | §8.7 |
| 2.6 | 流变闭合与 Picard（含连续化重试）| §4、§10.4.2 |
| 2.7 | 屈服函数、图 $\mathsf G_Y$、刚性核、核拟合、$\mathbf v_{\rm sl}$ | §2.8 B2–B4 |
| 2.8 | 表面迹、曲面剪切率、摩擦角链 | §10.2–10.3 |
| 2.9 | 模式 C 与 $\rho_s$ 输运 | §2.8、§9.2 |
| 2.10 | `d_pred`、分类、Clarabel SOCP、外层验收 | §2.13、§10.4.3 |
| 2.11 | 参考系重对齐 | §10.4.4 步骤 8b |
| 2.12 | `ScfPlugin` 的 `StepPhase` 状态机/主循环、快照、断点续算、`scf-cli run-sim` | §10.4.4 |
| 2.13 | **GPU 下沉**（逐核替换 CPU 参考，每替换一个核即与 CPU 对拍通过后才并入）：`surface_metric` → `radius_sample` → `strain_eval/yield_eval` → `poisson_assemble` → `stokes_assemble_*` → `surface_trace` → `mode_c` → `socp_prep` | §4.2 |
| 2.14 | diagnostics：`--verify-cpu`、`--error-budget`、$K=8$ 实验、解析/制造解 | §6 |

先以**纯 CPU f64 参考实现整条链路跑通**（步 2.1–2.12），再做 GPU 下沉（2.13）——保证每个 WGSL 核都有可直接对拍的权威 CPU 实现。

### 5.2.2 单元测试指标

| 模块 | 测试 | 指标 |
|---|---|---|
| Jacobi 基 | 正交 $\int\psi_{\ell j}\psi_{\ell k}\zeta^2=\delta_{jk}/(2k+2\ell+3)$；$\psi(1)=1$、$\psi'(1)=\sigma_{\ell k}$、$\psi''(1)=\kappa_{\ell k}$（命题 8.3、§10.2.1 表）| $\le10^{-13}$ |
| 质量块 | $R\equiv R_0$、$\rho\equiv\rho_0$：$\mathbf M^T_\ell$ 对角、$\mathbf M^P_\ell=\rho_0R_0^5\Lambda(\Lambda\tilde N+\tilde P)$，$\tilde N_1,\tilde P_1$ 与 §10.1 解析矩阵一致 | $\le10^{-12}$ |
| 刚体约束 | 命题 8.11：$C_{\rm P}=\tfrac{4\sqrt{3\pi}}3\rho_0R_0^4$、$C_{\rm L}$ 仅 $k=0$ 非零；秩 6 | $\le10^{-12}$ |
| 无散/迹 | 速度场数值散度 $\nabla\cdot\mathbf v=0$（有限差分）；命题 10.2 三式对非球形 $R$ 与有限差分对拍 | $\le10^{-9}$ |
| 球面特例 | 命题 10.3：$D_{rS}=0\Leftrightarrow\sum c^P q=0$、$\sum c^Ts=0$；$\ell=1,k=0$ 刚体模式 $c=0$ | 精确 |
| 曲面剪切率 | 一般公式在 $R\equiv R_0$ 极限退化为固定球面公式；$\mathrm{tr}\,\mathbf D=0$ | $\le10^{-10}$ |
| $\mathbf K$ 装配 | 常数 $\eta$：对称正定（刚体约束补空间）；制造解 $\mathbf u^*$ → $\mathbf F=\mathbf K\mathbf u^*$ → 求解还原 | $\|\mathbf u-\mathbf u^*\|/\|\mathbf u^*\|\le10^{-10}$ |
| 引理 8.12 | $\mathbf K_\gamma$ 正定；以 $\mathbf K_\gamma$ 替换 $\mathbf K$ 解不变 | $\le10^{-10}$ |
| 值域空间法 | 与直接求解 KKT 稠密系统对拍；PCG 迭代数报告 | $\le10^{-9}$ |
| Poisson | 均匀自引力球（无自转）：$p(r)=\tfrac{2\pi}{3}G\rho_0^2(R_0^2-r^2)$；Dirichlet 提升正确性 | 相对 $\le10^{-6}$（受 Tier 1 梯度精度支配，用解析重力注入隔离：$\le10^{-10}$）|
| 压力—速度耦合 | 均匀球、无自转、$\mathbf u^0=\mathbf 0$：Picard 一轮收敛，$\mathbf u=\mathbf 0$，$\mathbf r_{\rm rot}\le10^{-8}$ | — |
| 流变闭合 | $\eta$ 的上下界 $\eta_{\rm lo}\le\eta\le\eta_{\rm hi}$；$\mu_d$ 单调、值域 $[\mu_s,\mu_2)$；$p_*\to0$ 与 $\dot\gamma=0$ 规则；Bingham/Papanastasiou 不混用（配置互斥测试）| 精确 |
| 屈服分类 | 刚体平移/转动 $F_\delta<0$ 与速度大小无关；$\mu_d=\mu_s$ 时 $F_\delta>0\iff\dot\gamma>\dot\gamma_Y$ | 精确 |
| 刚性核 | 构造"被剪切带隔开的滑块"：滑块 $\chi_Y=1$ 而其表面点自身未屈服；质量最大分量并列规则；$\mathcal U=\varnothing$ → `NoCore`；共线核退化 | 精确 |
| 核拟合 | 刚体速度场 → 精确恢复 $(\mathbf a_c,\boldsymbol\theta_c)$，$\mathbf v_{\rm sl}\equiv0$ | $\le10^{-12}$ |
| 步长 | 各项限制逐一触发；$\mathcal A=\varnothing$ 时 CFL 为 $+\infty$；`stages` 边界对齐 | 精确 |
| 自转表 | stages/linear/table 的 $\omega(t)$、$T_{\rm end}$、$t_{\rm next}$；$\omega<0$ 被拒 | 精确 |
| 模式 C | 无外源：$\int J_{\rm dep}j\,d\Omega=\int J_{\rm ero}j\,d\Omega$；$\rho_s\ge0$（步长条件下）；$\rho_{\rm bulk}/\rho_K$ 权使质量行零漂移 | $\le10^{-12}$（f64）|
| 半拉格朗日 | 常速度转动：$\int j\rho_s$ 守恒；二阶格式阶数测试（中点法）| 守恒 $\le10^{-10}$ |
| SOCP | ① $d_{\rm pred}$ 可行且无约束活跃 ⇒ $d_*=d_{\rm pred}$；② 质量行满足 $|\hat{\mathcal M}(d_*)-M|\le$ tol 或 $s_M$ 显式非零；③ 锁死 $d=0$ 位级；④ $\mathcal A=\varnothing$ 零次 Clarabel 调用；⑤ 与等价 QP（线性化范数）对拍；⑥ KKT 残差 | tol $10^{-8}$ |
| 重对齐 | 随机旋转/平移的椭球 → 触发后 $|x_{\rm cm}|,|I_{xz}|,|I_{yz}|\le\varepsilon_{\rm frame}$；$\det Q=+1$；核中心同步（密度场不变：重对齐前后 $\rho_K$ 在对应点相等）；近简并规则；$\omega$ 不改变 | 精确/ $10^{-10}$ |
| 快照 | 写—读—续算比特一致（CPU 确定性模式）；续算 $N$ 步 = 连续 $N$ 步 | 位级 |

### 5.2.3 集成验收门槛（Tier 2 → Tier 3 的放行条件）

| # | 场景 | 通过标准 |
|---|---|---|
| G1 | **静止球**：均匀球、不自转、`stages` 单阶段 | 第 1 步 $\mathcal A=\varnothing$，`StageEquilibrium`，$d_*\equiv0$ 位级，质量偏差 $=0$ |
| G2 | **质量守恒**：任意含流动场景连续运行 $\ge200$ 步 | $|\mathcal M-M|/M\le\varepsilon_M$ 每步成立；无外源时 $\sum J_{\rm dep}=\sum J_{\rm ero}$ |
| G3 | **死区锁定**：任意步 | 锁死集 $d_i=0$ **位级**；无 $\bar\varepsilon_p$ 等诊断量影响几何（对照实验：关闭诊断，几何位级相同）|
| G4 | **加速自转**：球/椭球 + `linear` 或 `stages` 升速 | 赤道可动集非空；形状演化；刚体动量/角动量约束 $\varepsilon_\lambda\le$ tol；每步 2.15(a)(b) 成立；重对齐按触发条件执行且残差复核通过 |
| G5 | **阶段平衡**：`stages` 阶段末 | 若 $\mathcal A=\varnothing$ → 平衡；否则带可动面积分数输出"未平衡"，非失败 |
| G6 | **GPU—CPU 对拍**（`--verify-cpu`，diagnostics 构建）：同一初始条件前 $N_v$ 步 | $\mathbf u$ 相对差 $\le10^{-4}$，$p\le10^{-4}$，$\chi_Y$ 分类一致率 $\ge99.9\%$（临界点除外，计数入报告），$d_*$ 相对差 $\le10^{-3}$ |
| G7 | **收敛性**（`--error-budget --k8`）：$E_{4\to8}$、$L_{\max}$ 加密 | 出报告；$E_{4\to8}$ 超配置阈值标记"分辨率不足"，不改写通过判定（MS §8.9）|
| G8 | **误差预算**（`error-budget`）| $E^{\rm qs}_{\rm total}$ 各分量按 MS §9.8 无量纲求和，单列 $E_{\rm proj}$、$|s_{M*}|/M$ 等；报告完整 |
| G9 | **确定性**：同输入、CPU 确定性模式两次运行 | 快照位级一致；GPU 模式差 $\le$ G6 容差 |
| G10 | **断点续算**：中途终止 → `resume` | 与不中断运行一致（CPU 确定性位级）|
| G11 | **失败路径**：缺参数、非星形、Picard 不收敛（人为 $k_{\max}=1$）、$\mathrm{rank}<6$ | 返回表 3.6 的对应错误码，且不写"通过"快照 |
| G12 | 层次：`tools/check_layering.sh` 通过；`run-sim` 生产构建（无 `diagnostics`）二进制中不含验证代码符号 | 脚本检查 |

## 5.3 阶段三：Tier 3 `asteroid_app`

### 5.3.1 开发顺序

| 步 | 内容 |
|---|---|
| 3.1 | 在 Tier 1/2 的同一 Bevy `App` 上挂载 `AsteroidPlugin`；复用 `GpuContext::from_external`（共享 Bevy 的 `RenderDevice/Queue`，避免第二个 wgpu 实例）|
| 3.2 | `ScfPlugin` 直接挂入同一 `World`：`StepPhase` 协作式推进（`StepControl::Cooperative`）+ `ComputeTask` 异步重型相位；渲染系统直接 `Query` 输出组件；暂停/单步经 `StepControl`（无 worker 线程、无双缓冲）|
| 3.3 | 表面网格渲染（固定拓扑，顶点缓冲 $R_i$ 每步更新）与自由视角相机 |
| 3.4 | `surface_deform.wgsl` GPU 形变 + 法向重算 |
| 3.5 | 标量/矢量覆层、色图、内部切片 |
| 3.6 | egui：参数面板（补丁语义，§3.7）、时间轴（$\omega(t)$ 曲线、播放/暂停/步进/回放）、诊断视图 |
| 3.7 | `.snap` 回放模式（不依赖仿真器）|
| 3.8 | 性能与稳定性打磨 |

### 5.3.2 验收门槛

| # | 标准 |
|---|---|
| U1 | 渲染不物理：Tier 3 系统对 Tier 2 组件/资源仅有 `Res`/`Query<&T>` 只读访问，删除 `ScfPlugin` 后仍可回放（编译期检查：回放模块不 `use scf_simulator::schedule`，且不出现 `ResMut`/`Query<&mut …>` 指向 Tier 2 类型）|
| U2 | 仿真由 `StepPhase` 协作式推进（重型相位为 `ComputeTask`），UI 帧时间与仿真步耗时解耦：$2\times10^4$ 顶点网格下 UI $\ge60$ FPS（中端独显/Apple M 系列），仿真步不阻塞渲染 |
| U3 | 渲染所见量 = Tier 2 输出组件量（`StarMesh`、`SurfaceFields`…）（截图回归：同一快照渲染图像与基线 SSIM $\ge0.99$）|
| U4 | 参数补丁只在步边界生效；改变物理模型参数 → 产生新分支运行并有日志标记 |
| U5 | 覆层色图的数值范围、对数/线性切换、发散色图在 0 处居中单测 |
| U6 | `--headless-test`：无窗口启动、加载样例、推进 $N$ 步、断言快照与 `scf-cli` 同输入输出一致 |
| U7 | 回放数 GB 快照序列：内存占用有界（流式加载）|

---

# 6. 基准测试与调试隔离机制

## 6.1 设计原则

1. **生产路径 = 数学规范规定的最少计算**。`scf-cli run-sim` 默认只做 MS 要求的**廉价硬验收**（§6.2 白名单）；任何昂贵或需独立重算的检查**绝不**进入主循环。
2. **隔离在编译期与配置期双重完成**：
   - 编译期：全部验证/基准/误差预算代码位于 `diagnostics/` 模块并由 `feature = "diagnostics"` 门控；默认构建不包含这些代码，也不暴露对应 CLI 标志（对未知标志 `clap` 直接报错）。
   - 配置期：即使在 diagnostics 构建中，也必须**显式传入独立标志**才触发；标志不出现 → 与生产构建执行相同路径。
3. **零开销抽象（诊断插件 + 只读系统）**：

```rust
// 主循环只发出 O(1) 的轻量事件（无状态拷贝）：
//   PicardIterDone { k, info: PicardInfo }     StepEnded { n }
// 诊断系统只读订阅，参数签名中不含任何 ResMut / Query<&mut _> / Commands 对物理状态的写访问：
fn cpu_verify_on_step_end(
    mut ended: EventReader<StepEnded>,
    mesh: Query<&StarMesh>, fields: Query<&SurfaceFields>,
    rec: Res<StepRecord>, cfg: Res<VerifyConfig>, /* … 全部只读 */
) { /* … */ }

pub struct ScfDiagnosticsPlugin;   // #[cfg(feature = "diagnostics")]：Plugin::build 注册上面的系统，并 .after(ScfSet::Output)
```

   诊断系统（`CpuVerify*`、`ErrorBudget*`）仅存在于 `diagnostics/`，由 CLI 在 diagnostics 构建且显式标志时才 `add_plugins`；生产构建既无代码也无注册，诊断事件无订阅者，开销为零。**诊断系统不得改变状态**（只读系统参数，由签名与 `tools/check_layering.sh` 的签名检查共同保证），故诊断开启与否数值路径相同。

## 6.2 生产硬验收白名单（始终开启，$O(N)$ 或一次 matvec）

这些是数学规范**强制每步判定/报告**的量，且都是主循环已产出量的廉价归约：

| 量 | 来源 | 动作 |
|---|---|---|
| Picard 判据 (a)、$\varepsilon_{\rm mom}^{\rm weak}$、$\varepsilon_\lambda$ | 一次 $\mathbf K\mathbf u$ matvec（Picard 末轮已有）| 失败→状态码 |
| $\mathrm{rank}\,\mathbf B_{\rm rb}$、$\sigma_{\min}(\mathbf B_{\rm rb})$ | $6\times6$ | 失败→状态码 |
| 合力/合力矩相容 | 求积和 | 失败→状态码 |
| 质量残差、正体积、无自交、星形性 | 几何验收（SOCP 外层必需）| 触发信赖域缩小 |
| 可动面积分数、受拉比例、$\mathrm{Ek}^{-1}$、$\varepsilon_{\rm Euler}$、$s_{M*}$、$\|\mathbf m^{\rm spur}\|$ 类 | 归约 | 写日志 |
| $\mathbf r_{\rm rot}$、$\delta_{na}$ | 已有 $\nabla\cdot\tau_{\rm flow}$ 与表面迹 | 写日志，超阈值标记"分辨率不足" |
| $\mathrm{PCG}$ 迭代数、$\eta_{\rm hi}/\eta_{\rm lo}$ | 求解器计数 | 写日志 |
| `geom_repr_gap` | 采样比较 | 写日志 |

**判定准则**：成本不超过"一次已有矩阵的 matvec 或一次逐点归约"、且不需要第二套独立实现 ⇒ 白名单；否则 ⇒ 显式诊断。

## 6.3 显式触发的诊断/基准（仅 `--features diagnostics` 且显式标志）

| 触发 | 内容 | 对应数学规范 |
|---|---|---|
| `rtfp-cli verify --verify-cpu` | GPU 结果 vs 高精度 CPU f64 参考（自适应求积）| §1、§3 |
| `rtfp-cli verify --analytic-sphere` / `--polyhedron-exact` | 球解析、多面体精确公式对拍 | — |
| `rtfp-cli verify --feynman` | Feynman 参数化 + Carlson 独立通道与射线通道互校 | §1.7–1.9、§2.2 |
| `rtfp-cli verify --laplace-checks` | $\mathrm{tr}H$、跳跃条件、梯度连续 | §1.2、§2.7 |
| `rtfp-cli verify --roundoff` | f32 vs f64 舍入/归约误差统计 | §3.14 |
| `scf-cli run-sim --verify-cpu [--verify-every K]` | 每 $K$ 步在 CPU f64 上重算 $\mathbf K$、Poisson、Picard 并比对（**默认不存在**）| §8.9 |
| `scf-cli error-budget` | $E^{\rm qs}_{\rm total}$：$\varepsilon_{\rm mom}^{\rm weak},\varepsilon_\lambda,\varepsilon_{\rm pic}^{\rm achieved}$、$\mathbf r_{\rm rot}$、$E_{\rm frame}$（细网格射线重采样参考）、$\mathbf u_B-\mathbf u$（Track B）、$\delta_{na}$；单列 $E_{\rm proj}$ 等 | §9.8 |
| `scf-cli error-budget --k8 [--lmax-refine]` | $E_{4\to8}=\|\mathbf u_8-\mathbf u_4\|/\|\mathbf u_8\|$、$L_{\max}$ 收敛；$K=8$ 仅在此实验内启用（`k8_experiment.rs`）| §8.9、§9.8 |
| `scf-cli verify --suite …` | 解析/制造解、能量与耗散率恒等式、SOCP KKT 对拍 | §8、§10 |
| `* benchmark` / `rtfp-cli bench` | 分阶段计时、吞吐、GPU 占用、PCG 收敛曲线 | — |

**离线重放原则**：因 `.snap` 含完整可重启状态（含热启动 $\mathbf u^0,p^0$），`error-budget` 通过**加载生产运行产生的快照，在选定步重放该步**来计算昂贵量，不需要也不允许修改生产循环。生产快照元数据标 `verification_level = "production"`；经 `error-budget`/`verify` 全部通过的运行可生成 `certified` 报告（独立文件，指向快照哈希）——满足数学规范"每次运行须报告收敛性"的要求，而不拖慢生产。

## 6.4 基准测试体系

- **组件级基准**（`criterion`，`diagnostics` 下）：BVH 构建、射线计数/写入、融合积分 kernel、前缀和、SH 变换、$\mathbf K$ 装配、PCG、Poisson、Clarabel 调用（按 $|\mathcal A|$ 缩放）。
- **端到端基准**：固定样例（`tests/data/bench_*.toml`）的每步耗时分解表（Tier 1 / Stokes / Poisson / 屈服 / SOCP / 重对齐 / IO）。
- **回归基线**：CI 保存 `series.jsonl` 的关键标量（Picard 迭代数、PCG 迭代数、质量残差、$\Delta t_n$ 序列）；偏离超过 5% 报警。性能基线仅趋势告警，不作硬门槛。
- **GPU 计时**：`wgpu::Features::TIMESTAMP_QUERY` 可用时按 pass 记录，仅 diagnostics 构建启用。

---

# 7. 跨平台与 WASM 扩展性设计

## 7.1 当前目标（v1）

原生本地优先：macOS（Metal）、Linux（Vulkan）、Windows（DirectX 12）。`wgpu` 后端由平台自动选择；`rayon` 全线程。任何仅原生可用的能力（线程、文件系统、阻塞回读）必须被隔离在 `cfg(not(target_arch = "wasm32"))` 或 `native` 模块内。

## 7.2 抽象层约束（硬性规则，CI 检查）

| 约束 | 规则 | 检查方式 |
|---|---|---|
| 异步设备获取 | `GpuContext::new(...)` 为 `async fn`，内部 `Instance::request_adapter().await` / `request_device().await`；`pollster::block_on` **只出现在 `src/bin/*` 与 `common::native`** | grep/CI |
| 异步回读 | `common::gpu::buffers::read_async()`：`map_async` + 通知；原生端等待用 `device.poll(Wait)`，wasm 端 `await` 回调（由浏览器驱动）| 单测（原生）|
| 文件 IO | 核心计算层（`common` 除 `io/native_fs.rs`、`rtfp_solver`、`scf_simulator`、`asteroid_app`）**禁止** `std::fs`/`std::path::Path::exists` 等；一律经 `AssetIo` 资源（`AssetRead`/`AssetWrite` 事件，完成后发 `AssetReady`/`AssetWritten`）| `tools/check_layering.sh` 对 `std::fs` 的 grep 门禁 |
| 实现 | `AssetIo` 为 enum 资源：读 `Fs`（原生）、`Mem`（测试/管道）、`Fetch`（wasm，`web-sys` fetch）；写 `Fs`、`Mem`、（未来）`BlobDownload`/`IndexedDb`；以 `match` 分派，无 trait | |
| 并行 | 仅通过 `common::par`：native=`rayon`；wasm（无共享内存线程）=串行实现，API 相同（`par_for_each`、`par_map_reduce`、`par_chunks_mut`）；未来可接 `wasm-bindgen-rayon` 而不改调用点 | 禁止业务代码直接 `use rayon::prelude::*`（`par.rs` 内除外）|
| 时间 | 不用 `std::time::Instant` 于核心（wasm 不可用）；统一 `common::time::now()` 抽象 | |
| 随机数 | 种子显式；不依赖 OS 熵（`getrandom` 仅在 wasm 特性下配置 `wasm_js`）| |
| 阻塞 | 主循环为 `StepPhase` 协作式状态机（`Cooperative` 每 tick 推进一个相位并让出事件循环）；`Blocking` 模式（`block_on` 当前相位）仅 `cfg(not(wasm32))`；wasm 端只用 `Cooperative` + `ComputeTask` 轮询 | |
| 内存 | GPU 缓冲上限从 `adapter.limits()` 读取（WebGPU 默认限额更低：`maxStorageBufferBindingSize` 128 MiB）；分块逻辑已按此设计，禁止硬编码大缓冲 | 小限额模拟测试 |
| 精度 | f32 GPU 策略在 WebGPU 与原生相同；无 f64 依赖 GPU 特性 | |
| 特性探测 | `subgroups`、`TIMESTAMP_QUERY`、硬件 RT 均为**可选特性**，缺失时走通用路径（`common::gpu::capability`）| |
| 单线程 Clarabel | wasm 下 Clarabel 单线程运行；SOCP 规模仅 $|\mathcal A|$ 级，可接受 | |

## 7.3 WASM 适配层范围（未来增量，v1 只留骨架）

`src/wasm/` 仅包含：
1. `entry.rs`：`#[wasm_bindgen(start)]`、`console_error_panic_hook`；
2. `asset_fetch.rs`：`AssetIo::Fetch` 的 fetch 实现；`BlobDownload` 写出；
3. `canvas.rs`：Bevy `WindowPlugin` 绑定 `HtmlCanvasElement`（Tier 3）；
4. `capability.rs`：`navigator.gpu` 探测与 limits 报告。

**增加上述适配层即可编译到浏览器 WebGPU**：核心（Tier 1/2）无需任何改动。`Cargo.toml` 已含的 `wasm-bindgen`、`web-sys`（`Gpu`、`HtmlCanvasElement`、`Request`、`Response` …）、`js-sys`、`wasm-bindgen-futures`、`console_error_panic_hook`、`getrandom/wasm_js` 保持。浏览器端目标分层：Tier 1（`eval-field` 等价 API）可先行；Tier 2 视内存/线程条件分期；Tier 3 复用 Bevy 的 wasm 路径。

## 7.4 可移植性检查清单（每个 PR 的合并前自查）

- [ ] 新增 IO 是否经 `AssetIo`？
- [ ] 是否把逐顶点/逐场点数据拆成了单独 Entity，或引入了 trait 派发/后台线程/双缓冲？
- [ ] 是否引入 `std::thread`/阻塞等待到 `common` 以外？
- [ ] WGSL 是否使用了非 WebGPU 标准特性（需 `enable` 指令的必须在 capability 内探测）？
- [ ] 缓冲区是否尊重 `limits`？
- [ ] 新增并行是否经 `par.rs`？

---

# 附录 A：数学规范 → 模块追溯表

| 数学规范条款 | 实现位置 |
|---|---|
| §1.1 密度 $\rho_K$、势 | `common/density.rs` |
| §1.3–1.6 切锥/有限部分/径向原函数 | `rtfp_solver/radial/*`、`density/cauchy_ray.rs` |
| §1.7–1.9 Feynman/Carlson | `rtfp_solver/diagnostics/feynman_track.rs`（独立通道，仅验证）|
| §2.4–2.7 梯度/Hessian/内部与表面有限部分 | `rtfp_solver/field/*`、`near_field.wgsl` |
| §2.8 B1–B6、模式 C | `scf_simulator/{pressure,rheology,yielding,surface/mode_c}` |
| §2.11–2.12 模式 A 接口 | `scf_simulator/reserved/fluid_equilibrium.rs`（保留，C6）|
| §2.13–2.14 SOCP / 欠松弛 | `geometry_update/{predict,socp,accept}` |
| §2.15 收敛判据 (a)(b)(c) | `rheology/picard.rs`、`accept/step_checks.rs`、`stepping/stage.rs` |
| §3 GPU 射线管线 | `rtfp_solver/{backend,shaders}` |
| §4 联合流变 | `rheology/closure.rs`、`tide_stress.rs` |
| §5–7 弱式/双势/变密度（准静态限制）| `basis/*`、`stokes/strain_rate.rs` |
| §8.1–8.3、8.5 基/质量/刚体约束 | `common/harmonics/jacobi.rs`、`stokes/rigid.rs` |
| §8.4′ 准静态系统、引理 8.12、值域空间法 | `stokes/{assemble,solve,precond_b1}.rs` |
| §8.7 压力 Poisson | `pressure/*` |
| §8.9 误差报告 | 白名单（§6.2）+ `diagnostics/error_budget.rs` |
| §9.0、§9.2 曲面几何、半拉格朗日 | `surface/{metric,mode_c}.rs`、`mode_c.wgsl` |
| §9.5 / A.4 Track B | `stokes/{precond_b1,track_b}.rs` |
| §9.8 误差预算 | `diagnostics/error_budget.rs` |
| §10.2 表面迹/曲面剪切率 | `surface/trace.rs`、`surface_trace.wgsl` |
| §10.3 摩擦角闭合 | `surface/friction.rs` |
| §10.4.1 自转表、步长 | `spin.rs`、`stepping/timestep.rs` |
| §10.4.2 Picard | `rheology/picard.rs` |
| §10.4.3 SOCP | `geometry_update/socp.rs` |
| §10.4.4 主循环与步骤 8b | `schedule.rs`（`StepPhase`/`ScfSet`）、`geometry_update/realign.rs` |
| 附录 A.3 刚体约束/可解性/重对齐 | `stokes/rigid.rs`、`realign.rs`、`accept/step_checks.rs` |
| 附录 A.5 PO 登记 | 未证项在日志/报告中以"假设"标注（如 PO-11 Picard 收敛性：失败按 2.15(a) 判，不隐藏）|

# 附录 B：风险与未决事项（需在编码中显式跟踪）

| 项 | 风险 | 处置 |
|---|---|---|
| df32 装配精度 | $\eta$ 对比度大时 $\mathbf K$ 条件数高，f32 误差可能影响 Picard | `assembly_precision = cpu_f64` 回退开关；G6 对拍监控；必要时 $\mathbf K$ 装配切分为 GPU 部分和 + CPU 补偿 |
| Gaunt 带宽与稠密存储 | $L_{\max}$ 大时 $N_u^2$ 显存 | 块稀疏 CSR（§4.3）|
| $R_h$ 与多面体不一致 | 棱处 $\Delta R$ 奇异 | `sh_smoothed` + `geom_repr_gap` 监控（§4.7）|
| 面心 Hessian → 节点 | 面积加权平均为工程选择 | 写入元数据，MS 未规定其他方式时保持此约定并在文档标注 |
| 远场多极 | MS §3.10 明确未完成定义 | 先占位；补齐定义后再实现 |
| PO-9/11/12 | 非线性收敛、屈服正则化极限未证 | 失败如实报告，不得输出通过状态 |
| Bevy 与 wgpu 版本 | 版本不一致将出现双份 wgpu | 以 Bevy 导出的 wgpu 类型为准，锁定版本 |
| 动态自由表面模式 | 不在范围 | 保留 MS 第 9 章仅作理论基准，不实现 `m` 输运 |
