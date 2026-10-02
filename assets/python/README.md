# Asteroid Tools (Python)

该目录专门存放小行星几何建模、数据转换与可视化相关的 Python 实用脚本，统一使用 `uv` 进行环境与依赖包管理。

## 环境初始化与依赖安装

进入当前目录，使用 `uv` 自动同步虚拟环境与依赖：

```bash
cd assets/python
uv sync
```

或直接免激活运行任何脚本：

```bash
uv run python <script_name>.py
```

## 核心脚本说明

### 1. `surface_xyz_to_glb.py`
将小行星表面演化计算导出的顶点云 (`output/evolved_surface.xyz`) 重建为封闭水密的三维三角面元网格，并导出为标准的 glTF 2.0 二进制文件 (`.glb`)。

- **算法机制**：基于球面投影 Delaunay / 3D 凸包拓扑，100% 保持所有 2562 个顶点的绝对坐标精度不变，并严格校准面片顶点环绕序使法向一致朝外。
- **基本用法**：
  ```bash
  # 默认读取 ../output/evolved_surface.xyz 并导出至 ../output/evolved_surface.glb
  uv run python surface_xyz_to_glb.py

  # 导出高程/隆起伪彩色模式
  uv run python surface_xyz_to_glb.py --color-mode elevation -o ../output/evolved_surface_elevation.glb

  # 指定自定义输入输出路径
  uv run python surface_xyz_to_glb.py -i /path/to/points.xyz -o /path/to/model.glb
  ```

### 2. `generate_progenitor_ryugu.py`
生成初始龙宫母体形状与非轴对称采样点云。
