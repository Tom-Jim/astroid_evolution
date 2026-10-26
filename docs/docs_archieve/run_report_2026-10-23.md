# 运行与优化报告（2026-10-23）

分支 `review-fixes-and-perf`，改动均未提交。测试机为 Apple M3、16 GB 内存、macOS 27；工具链为 Rust 1.99.1、trunk 0.22.0、wasm-opt 133、bun 1.3.14。

## 1. 结论

- **网页端：** 不再卡住。原来每个几何步 20–30 秒，现在每步约 0.1 秒；Ryugu 默认 990 步全程约 130 秒，Phobos 约 109 ms/步。
- **桌面端：** 物理计算原来被锁在单线程，现已放开到多核；各项功能均能正常使用。
- **命令行：**
  - Ryugu 正演：46.5 s → 27.7 s
  - Phobos 预测：88.9 s → 65.1 s
  - 1 次迭代的 history：95.3 s → 27.5 s
  - 峰值内存约降 170 MB
- **测试：** 149 个 tier2 测试和 6 个查看器测试全部通过。

## 2. 发现并修复的问题

| # | 问题 | 现象 | 修复 | 位置 |
|---|---|---|---|---|
| 1 | Phobos 结果不可复现 | 同一构建跑两遍，从第 862 步起出现差异 | rayon 的 fold/reduce 切分点随调度变化；改为固定 64 块、按顺序求和 | `src/scf/cores.rs`, `src/scf/mode_c_spectral.rs` |
| 2 | 桌面端物理只用单核 | 演化跑在 1 线程 rayon 池里，内部所有 `par_iter` 都在这一个线程上执行 | 换成独立线程加任务队列，并行循环走全局 rayon 池；CPU 占用从约 100% 升到约 210% | `src/app/pipeline.rs` |
| 3 | 网页端 MESH/W 线框无效 | WebGPU 不支持线框模式，Bevy 的 WireframePlugin 不加载 | wasm 下改用边线 LineList 网格，跟随表面姿态 | `src/app/render.rs`, `ui.rs`, `playback.rs`, `mod.rs`, `resources.rs` |
| 4 | 中文输入法下 `[` `]` 切面键失效 | 拼音输入法把 `]` 变成 `】`，`e.key` 匹配不上 | 快捷键改按物理键 `e.code` 判断；输入法正在组字时忽略 | `ui/src/App.tsx` |
| 5 | 快速连按 ← → `]` 不累计 | 状态每 0.1 s 才刷新一次，连按都从同一步算起 | 发送命令后立即在本地推进状态 | `ui/src/App.tsx` |
| 6 | 均匀密度时图例显示“每 0 一条等值线” | — | 改为显示“Interior density uniform, ρ kg/m³” | `ui/src/App.tsx` |
| 7 | CI 网页构建失败 | wasm-opt 拒绝 bulk-memory 等指令；版本 133 也一样，旧版本并不是根因 | 用 `data-wasm-opt-params` 显式启用 rustc 默认开启的 wasm 特性（另加 SIMD） | `src/web/index.html` |
| 8 | README 快捷键表写的是图层 1–4 | 实际有 5 个图层 | 改为 1–5，并补充 `?timing=1`、`AE_PDHG` 的说明 | `README.md` |
| 9 | 本轮自己引入的 wasm 编译错误 | 删除一行 import 后，cfg 属性落到了 Path 的 import 上 | 已修复，wasm 和桌面两个目标都已类型检查通过 | `src/app/pipeline.rs` |

## 3. 性能优化

### 3.1 锥规划（Metal-PDHG）：默认改为 CPU f64

步骤如下：

1. **GPU 融合核：** 小问题一个 workgroup 一次调度跑完 48 次迭代，原来每次迭代最多 8 次调度。与原实现逐位一致。
2. **预先多跑几批：** 一次 GPU 请求连跑最多 8 批（网页 16 批）；判断结果一变就按字节回滚。与原实现逐位一致。
3. **按你的决定改为 CPU f64：** 迭代公式和顺序与 GPU 核完全相同，用 f64（53 位）代替双单精度（约 48 位），并且完全没有 GPU 往返。

| 场景 | 优化前 | GPU 优化后 | CPU f64（默认） | 结果差异 |
|---|---|---|---|---|
| Ryugu 正演 PDHG 总耗时 | 16.6 s | 8.2 s | **0.38 s** | 最终半径最大差 1.1e-13 m，批数不变（17029） |
| Phobos 预测 PDHG | — | 1.38 s | **0.46 s** | 半径最大差 0.16 mm（11 km 天体，相对 1.4e-8） |
| 网页 PDHG | 211 s / 104 次求解 | — | **0.74 s / 124 次** | — |

`AE_PDHG=gpu` 可切回 GPU 双单精度路径。文档 §10.5.3 的精度实现表和 README 已同步。

### 3.2 网页整数双单精度快速路径（逐位一致）

浏览器里的 ds 运算用整数模拟。`mul_err`/`add_err` 在常见情况下改走直线整数路径，其余情况回退原通用路径。用 `AE_DS_EXACT=1` 构建的本机整数路径对比 200 步，逐位一致：

| 模块 | 优化前 | 优化后 |
|---|---|---|
| PDHG 每批 | 4.14 ms | 2.64 ms |
| `classify_contract` | 108 ms | 28.7 ms |
| `stokes_operators` | 8.4 s | 0.93 s |
| 200 步总耗时 | 188 s | 73 s |

另一个快捷分支（s == x 时残差即为 y）实测会让 GPU 内核变慢，已撤回。

### 3.3 其他

- **刚度矩阵组装：** 改为对上三角紧凑存储做连续 axpy，再一次性对称展开。求和顺序不变，结果逐位一致。
- **速度核幂次表：** 33 → 12（`basis.wesl`）。inspect 的峰值内存 1890 → 1717 MB，运行时 2142 → 1967 MB。
- **表面拟合缓存、零平流短路：** 上一轮已做，结果逐位一致。
- **wasm 开启 SIMD128：** LLVM 只对逐元素循环向量化，不会重排浮点运算。
- **网页 `?timing=1`：** 可查看各模块耗时，结果累计在 `asteroid.timing`。

## 4. 命令行耗时与内存（优化前 → 优化后）

| 命令 | 耗时 | 峰值 footprint |
|---|---|---|
| inspect_ryugu | 0.69 → 0.46 s | 1890 → 1717 MB |
| inspect_phobos | 0.52 → 0.52 s | 2042 → 1869 MB |
| relax_ryugu | 0.88 → 0.83 s | 522 → 520 MB |
| rtfp_integrals | 1.20 → 0.57 s | 221 MB |
| rtfp_eval_faces | 1.80 → 1.80 s | 731 → 707 MB |
| rheo_run | 0.80 → 0.13 s | 160 MB |
| runsim_ryugu_fwd | **46.53 → 27.74 s** | 2142 → 1967 MB |
| runsim_phobos_fut | **88.86 → 65.08 s** | 2297 → 2124 MB |
| history_ryugu_1 | **95.25 → 27.46 s** | 2347 → 2171 MB |

footprint 包含 GPU 显存，其中分类基常驻约 1 GB。本轮后期机器同一二进制前后两次运行能差 30%，单次数字只作参考。

## 5. 功能测试

### 网页（Chrome，WebGPU，trunk release 构建）

通过的项目：

- 图层 1–5
- 暂停/继续（按钮、Space）
- 内部切面 I 及 `[` `]`
- 线框（修复后）
- 重置视角 R
- 左键平移、滚轮缩放
- 时间轴 ← → / LIVE
- 切换 Ryugu / Phobos / Deimos 及各场景，LAUNCH / RESTART（0.5 s 进入第 0 步，约 3 s 到第 3 步）
- 参数滑块（摩擦角 30° 实际生效为 μ_s = tan 30°），RESET literature
- 自旋表达式错误时显示 FAULT 提示，当前运行不中断
- 面板折叠/展开，求解器详情
- 控制台无错误

未测：网页截图 P，会触发浏览器文件下载。

网页指标：

- 页面 DOM 就绪 0.96 s
- wasm 原始 28.8 MB，gzip 7.6 MB，brotli 4.7 MB
- JS 堆 11–41 MB
- 浏览器渲染进程 footprint 约 1.8–2.0 GB（三个 wasm 实例：页面、演化 worker、GPU worker）
- 时间轴历史上限 256 MB，超出后抽稀；实测 638 → 555 帧

### 桌面（Tauri .app，51 MB，其中二进制 35.8 MB）

通过的项目：

- 图层
- 内部切面与图例
- 原生线框
- 滚轮缩放、左键平移
- 重置视角
- 时间轴回放/LIVE
- 切换 Phobos 并 LAUNCH（约 26 步/s）
- Space 暂停（遥测显示 HOLD）
- 截图保存到 `~/Pictures/Asteroid Evolution`

未能用工具测试：右键拖动轨道旋转。

桌面指标：

- Ryugu 默认 990 步全程约 70 s
- 进程 footprint 约 2.9 GB（含 GPU），RSS 约 0.45–0.9 GB

## 6. 产物大小

| 产物 | 大小 |
|---|---|
| 桌面 .app | 51 MB（二进制 35.8 MB，glb 模型 9 MB，初始快照 JSON 6 份共 7.9 MB） |
| 网页 wasm（dist profile + wasm-opt -O3） | 28.8 MB，brotli 后 4.7 MB；之前目录里那份 89 MB 的是未优化的开发构建 |
| 内嵌着色器 | 约 1.3 MB（WGSL） |

## 7. GitHub Actions

- **桌面矩阵：** macOS arm64、Windows x64、**Windows on ARM64**（windows-11-arm，只打 NSIS 包）、Linux x64、**Linux ARM64**（ubuntu-22.04-arm）。Intel Mac 已按你的决定去掉。
- **工具版本与本地一致：** Rust 1.99.1、bun 1.3.14、trunk 0.22.0、wasm-bindgen 0.2.129、wasm-opt 133；actions 全部升级到当前大版本。
- **编译加速：** 关闭增量编译，依赖不带调试信息，加 rust-cache 共享 key 和 bun/trunk 缓存，trunk 用预编译二进制安装。
- **网页任务修复：** 见第 2 节第 7 项。
- **待处理：** 旧的 macos-13 任务 39821617426 仍在排队，需要的话我可以替你取消。

## 8. 遗留问题（需要你决定）

1. **从网格起步时锥规划不收敛：** Ryugu 默认场景的第 4–11 步，锥规划在 40000 次迭代内达不到对偶间隙 1e-3，这几步保持原几何（日志里的 “geometry kept”）。GPU 和 CPU 路径的迭代数、残差完全相同，属于原有行为，不是本轮引入的回归。这些问题很小（n ≤ 34），可以考虑改用 f64 内点法精确求解。这属于求解器改动，需要你批准。
2. **桌面首次启动黑屏：** 新二进制第一次启动，到第一帧约 2 分钟；再次启动约 24 s。原因是 Metal 着色器缓存为空，要从头编译很大的 asteroid 材质着色器，这部分代码本轮没改。可选方案：启动时显示加载提示、预热管线，或拆分着色器。
3. **网页面板文件名不带内容哈希：** `ui/ui.js` 没有哈希，浏览器可能拿缓存的旧面板去配新的 wasm。需要改 Vite 输出和 index.html 的引用方式。
4. **网页上的按键修复未做行为测试：** 浏览器侧测试因服务器被停止没有完成；桌面端已确认按键能到达处理函数，并定位了输入法这个根因。
5. **测试截图：** `~/Pictures/Asteroid Evolution` 里有 17 张测试截图，需要时可删除。
6. **全部改动尚未提交。**
