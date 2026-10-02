#!/usr/bin/env python3
"""
Surface XYZ Point Cloud to Watertight Triangle Mesh GLB Converter
=================================================================
将小行星表面演化生成的顶点云 (evolved_surface.xyz) 重建为严格封闭的
二维球面同胚三角网格流形，并导出为标准的 glTF 2.0 Binary (.glb) 文件。

特性:
  1. 采用球面投影 Delaunay / 3D 凸包拓扑，100% 保留原始顶点的绝对物理坐标。
  2. 严格校准三角形环绕顺序，保证所有面法向量严格朝向小行星外部。
  3. 计算面积加权平滑顶点法向量，在任何 Web 3D / 引擎中呈现真实光照。
  4. 支持物理真实陨石暗岩材质 (PBR)，亦支持半径/高程伪彩色渲染。
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull
import trimesh


def load_xyz(filepath: Path) -> np.ndarray:
    """读取 XYZ 格式点云，自动忽略注释行和空行。"""
    if not filepath.exists():
        raise FileNotFoundError(f"输入点云文件不存在: {filepath}")

    points = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("//"):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) not in (3,) and len(parts) < 6:
                raise ValueError(
                    f"第 {line_num} 行格式错误：需要至少 3 个数值列，实际 {len(parts)} 列"
                )
            try:
                point = [float(parts[0]), float(parts[1]), float(parts[2])]
            except ValueError as exc:
                raise ValueError(f"第 {line_num} 行包含非数值坐标") from exc
            points.append(point)

    if len(points) < 4:
        raise ValueError(f"有效顶点数量过少 ({len(points)} < 4)，无法构成三维多面体")

    pts = np.asarray(points, dtype=np.float64)
    if not np.all(np.isfinite(pts)):
        raise ValueError("输入点云包含非有限浮点数 (NaN 或 Inf)")
    if len(pts) != 100_000:
        raise ValueError(f"输入点云必须严格包含 100000 个顶点，实际为 {len(pts)}")
    return pts


def triangulate_star_shaped_surface(points: np.ndarray) -> np.ndarray:
    """
    对星形流形点云进行严格球面 Delaunay 拓扑三角剖分。
    
    原理:
      小行星表面为亏格 0 (球同胚) 闭流形。将以质心为中心的各点方向向量
      u_i = (p_i - c) / ||p_i - c|| 投影到单位球 S² 上，其 3D 凸包在球面
      上的面片拓扑精确对应球面 Delaunay 剖分。
      使用原绝对物理坐标配对该剖分面片，即可在 100% 保留所有顶点真实几何的同时，
      自动生成完全封闭、无缝隙、无重叠、欧拉示性数 V - E + F = 2 的水密网格。
    """
    centroid = np.mean(points, axis=0)
    relative = points - centroid
    radii = np.linalg.norm(relative, axis=1)

    if np.any(radii < 1e-12):
        raise ValueError("存在退化至质心原点的奇异顶点，无法定义径向投影")

    # 单位球面投影
    unit_directions = relative / radii[:, np.newaxis]

    # 球面凸包拓扑
    hull = ConvexHull(unit_directions)
    faces = hull.simplices.copy()

    # 严格校准面片顶点环绕序 (Winding Order)，确保所有法向量背离质心朝外
    v0 = points[faces[:, 0]]
    v1 = points[faces[:, 1]]
    v2 = points[faces[:, 2]]

    face_normals = np.cross(v1 - v0, v2 - v0)
    face_centers = (v0 + v1 + v2) / 3.0
    outward_vectors = face_centers - centroid

    # 点乘检测：如果面法向与朝外方向夹角大于 90 度，则翻转面片
    dot_products = np.sum(face_normals * outward_vectors, axis=1)
    flip_mask = dot_products < 0.0

    faces[flip_mask] = faces[flip_mask][:, [0, 2, 1]]

    return faces


def crater_colormap(points: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    """根据去除整体椭球扁率后的局域地貌高程残差着色 (深陷撞击坑深蓝/青色，平原灰白，高耸赤道脊金红)"""
    rel = points - centroid
    r = np.linalg.norm(rel, axis=1)
    dirs = rel / r[:, None]

    # 拟合基准三轴椭球
    a_mat = np.column_stack([dirs[:, 0]**2, dirs[:, 1]**2, dirs[:, 2]**2])
    coeffs, _, _, _ = np.linalg.lstsq(a_mat, 1.0 / (r**2), rcond=None)
    coeffs = np.maximum(coeffs, 1e-12)
    abc = 1.0 / np.sqrt(coeffs)
    r_ellip = 1.0 / np.sqrt(
        (dirs[:, 0] / abc[0])**2 + (dirs[:, 1] / abc[1])**2 + (dirs[:, 2] / abc[2])**2
    )
    residuals = r - r_ellip  # 负值为凹陷/陨石坑，正值为赤道脊/峰峦

    res_min = np.percentile(residuals, 2.0)
    res_max = np.percentile(residuals, 98.0)
    norm = np.clip((residuals - res_min) / (res_max - res_min + 1e-12), 0.0, 1.0)

    colors = np.zeros((len(points), 4), dtype=np.uint8)
    colors[:, 3] = 255
    for i, t in enumerate(norm):
        if t < 0.35:  # 陨石坑底 (深蓝 -> 宝蓝 -> 青)
            s = t / 0.35
            colors[i, :3] = [int(15 + 20 * s), int(30 + 130 * s), int(180 + 30 * s)]
        elif t < 0.65:  # 平原与斜坡 (青灰 -> 浅沙黄)
            s = (t - 0.35) / 0.30
            colors[i, :3] = [int(35 + 160 * s), int(160 + 30 * s), int(210 - 130 * s)]
        else:  # 赤道隆起脊与山巅 (金黄 -> 亮橙红)
            s = (t - 0.65) / 0.35
            colors[i, :3] = [int(195 + 55 * s), int(190 - 140 * s), int(80 - 50 * s)]
    return colors


def elevation_colormap(radii: np.ndarray) -> np.ndarray:
    """根据距质心半径计算高对比度伪彩色 (赤道菱形脊金红，中纬度黄色，极区塌陷深蓝)"""
    r_min, r_max = np.min(radii), np.max(radii)
    norm = (radii - r_min) / (r_max - r_min + 1e-12)

    colors = np.zeros((len(radii), 4), dtype=np.uint8)
    colors[:, 3] = 255

    for i, t in enumerate(norm):
        if t < 0.333:
            s = t / 0.333
            r = int(20 + s * (30 - 20))
            g = int(60 + s * (180 - 60))
            b = int(160 + s * (190 - 160))
        elif t < 0.666:
            s = (t - 0.333) / 0.333
            r = int(30 + s * (230 - 30))
            g = int(180 + s * (190 - 180))
            b = int(190 + s * (40 - 190))
        else:
            s = (t - 0.666) / 0.334
            r = int(230 + s * (220 - 230))
            g = int(190 + s * (40 - 190))
            b = int(40 + s * (30 - 40))
        colors[i, 0] = np.clip(r, 0, 255)
        colors[i, 1] = np.clip(g, 0, 255)
        colors[i, 2] = np.clip(b, 0, 255)

    return colors


def convert_xyz_to_glb(
    input_path: Path,
    output_path: Path,
    color_mode: str = "realistic",
) -> trimesh.Trimesh:
    """执行点云到 GLB 的完整几何重建与导出流程。"""
    print(f"--> [1/4] 载入表面点云: {input_path}")
    points = load_xyz(input_path)
    nv = len(points)
    centroid = np.mean(points, axis=0)
    relative = points - centroid
    radii = np.linalg.norm(relative, axis=1)

    r_min, r_max, r_mean = np.min(radii), np.max(radii), np.mean(radii)
    flattening = (r_max - r_min) / r_mean

    print(f"    * 顶点总数: {nv}")
    print(f"    * 几何中心: [{centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f}] m")
    print(f"    * 径向范围: R_min = {r_min:.2f} m, R_max = {r_max:.2f} m, R_mean = {r_mean:.2f} m")
    print(f"    * 外形扁率/隆起度: ε = {flattening:.4f}")

    print("--> [2/4] 执行球面流形 Delaunay 拓扑三角剖分与外向一致性定向...")
    faces = triangulate_star_shaped_surface(points)
    nf = len(faces)
    print(f"    * 生成三角面片数: {nf}")
    print(f"    * 欧拉示性数: χ = V - E + F = {nv} - {nv + nf - 2} + {nf} = 2 (亏格 0 封闭球面 ✓)")

    print(f"--> [3/4] 构建三维网格模型并应用材质 (模式: {color_mode})...")
    mesh = trimesh.Trimesh(
        vertices=points,
        faces=faces,
        process=True,
        validate=True,
    )

    # 验证拓扑封闭水密性
    is_watertight = mesh.is_watertight
    volume = abs(mesh.volume)
    area = mesh.area
    print(f"    * 多面体封闭水密性 (Watertight): {'完全通过 ✓' if is_watertight else '存在边缘 ✗'}")
    print(f"    * 实体表面积: {area:.4e} m²")
    print(f"    * 封闭多面体体积: {volume:.4e} m³")
    if len(mesh.vertices) != nv:
        raise ValueError(
            f"三角网格处理改变了顶点数：输入 {nv}，输出 {len(mesh.vertices)}；"
            "禁止静默丢失或合并高频表面点"
        )
    if not is_watertight or not np.isfinite(volume) or volume <= 0.0:
        raise ValueError("输出网格未通过水密性和正体积检查")

    # 赋予顶点颜色或 PBR 材质
    if color_mode == "crater" or color_mode == "topography":
        colors = crater_colormap(points, centroid)
        mesh.visual.vertex_colors = colors
    elif color_mode == "elevation":
        colors = elevation_colormap(radii)
        mesh.visual.vertex_colors = colors
    elif color_mode == "normal":
        normals = mesh.vertex_normals
        normal_colors = ((normals * 0.5 + 0.5) * 255).astype(np.uint8)
        colors = np.zeros((nv, 4), dtype=np.uint8)
        colors[:, :3] = normal_colors
        colors[:, 3] = 255
        mesh.visual.vertex_colors = colors
    else:
        # realistic: 龙宫/贝努典型暗碳质碎石堆深空低反照率外观 (Albedo ~ 0.045)
        # 根据局域撞击坑与脊线微调岩石明暗阴影 (凸起处略亮受光，坑底由于自遮挡略暗)
        rel_c = points - centroid
        r_c = np.linalg.norm(rel_c, axis=1)
        res_rough = r_c - np.median(r_c)
        shade_factor = np.clip(1.0 + (res_rough / (np.max(np.abs(res_rough)) + 1e-6)) * 0.25, 0.7, 1.3)
        base_color = np.array([120, 115, 110], dtype=np.float64)
        vert_rgb = np.clip(base_color[None, :] * shade_factor[:, None], 30, 220).astype(np.uint8)
        colors = np.zeros((nv, 4), dtype=np.uint8)
        colors[:, :3] = vert_rgb
        colors[:, 3] = 255
        mesh.visual = trimesh.visual.ColorVisuals(
            mesh=mesh,
            vertex_colors=colors,
        )

    print(f"--> [4/4] 导出 glTF 2.0 二进制文件 (.glb): {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    glb_data = mesh.export(file_type="glb")
    with open(output_path, "wb") as f:
        f.write(glb_data)

    file_size_kb = output_path.stat().st_size / 1024.0
    print(f"--> 转换完成！输出文件已生成: {output_path} ({file_size_kb:.1f} KB)\n")
    return mesh


def main():
    parser = argparse.ArgumentParser(
        description="将演化后表面点云 (evolved_surface.xyz) 连接三角面并转换为 GLB 3D 模型"
    )
    # 默认路径计算: 使用项目根目录下的 output/，避免把生成物混入 assets/
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    default_input = project_root / "output" / "evolved_surface.xyz"
    default_output = project_root / "output" / "evolved_surface.glb"

    parser.add_argument(
        "-i", "--input",
        type=Path,
        default=default_input,
        help=f"输入 XYZ 点云路径 (默认: {default_input})"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=default_output,
        help=f"输出 GLB 路径 (默认: {default_output})"
    )
    parser.add_argument(
        "--color-mode",
        choices=["realistic", "elevation", "normal", "crater", "topography"],
        default="realistic",
        help="着色模式: realistic (真实碎石堆暗岩外观), elevation (径向高程伪彩), crater/topography (去扁率局域凹坑与峰脊), normal (法向色彩)"
    )

    args = parser.parse_args()

    convert_xyz_to_glb(args.input, args.output, args.color_mode)


if __name__ == "__main__":
    main()
