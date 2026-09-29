//! 点云数据读取与预处理
//! 支持 .xyz, .csv, .pts, .txt 格式，支持法向量解析与单位转换

use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

use super::unit_system::UnitSystem;

/// 点云数据集合
#[derive(Clone, Debug, Default)]
pub struct PointCloud {
    /// 点坐标 (米)
    pub points: Vec<[f64; 3]>,
    /// 可选法向量 (已归一化)
    pub normals: Option<Vec<[f64; 3]>>,
}

impl PointCloud {
    pub fn new(points: Vec<[f64; 3]>, normals: Option<Vec<[f64; 3]>>) -> Result<Self, String> {
        if points.is_empty() {
            return Err("点云数据为空".to_string());
        }
        for (i, p) in points.iter().enumerate() {
            if !p[0].is_finite() || !p[1].is_finite() || !p[2].is_finite() {
                return Err(format!("点云索引 #{i} 包含非有限浮点数 (NaN/Inf): [{:?}]", p));
            }
        }
        if let Some(ref nrms) = normals {
            if nrms.len() != points.len() {
                return Err(format!(
                    "法向量数量 ({}) 与顶点数量 ({}) 不一致",
                    nrms.len(),
                    points.len()
                ));
            }
        }
        Ok(Self { points, normals })
    }

    /// 从文件加载点云，并应用指定的单位系统转换至米 (m)
    pub fn load_from_file(path: &Path, units: &UnitSystem) -> Result<Self, String> {
        let file = File::open(path)
            .map_err(|e| format!("无法打开点云文件 '{}': {e}", path.display()))?;
        let reader = BufReader::new(file);

        let mut points = Vec::new();
        let mut normals = Vec::new();
        let mut has_normals = false;

        for (line_idx, line_res) in reader.lines().enumerate() {
            let line = line_res.map_err(|e| format!("读取点云第 {} 行出错: {e}", line_idx + 1))?;
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') || trimmed.starts_with("//") {
                continue;
            }

            // 支持逗号或空白字符分隔
            let tokens: Vec<&str> = trimmed
                .split(|c: char| c == ',' || c.is_whitespace())
                .filter(|s| !s.is_empty())
                .collect();

            if tokens.len() < 3 {
                continue;
            }

            let x: f64 = tokens[0]
                .parse()
                .map_err(|_| format!("第 {} 行坐标 x 解析失败: '{}'", line_idx + 1, tokens[0]))?;
            let y: f64 = tokens[1]
                .parse()
                .map_err(|_| format!("第 {} 行坐标 y 解析失败: '{}'", line_idx + 1, tokens[1]))?;
            let z: f64 = tokens[2]
                .parse()
                .map_err(|_| format!("第 {} 行坐标 z 解析失败: '{}'", line_idx + 1, tokens[2]))?;

            // 转换至标准米 (m)
            let p_si = units.coords_to_meters([x, y, z]);
            points.push(p_si);

            // 如果包含法向量 (tokens >= 6)
            if tokens.len() >= 6 {
                if points.len() == 1 {
                    has_normals = true;
                }
                if has_normals {
                    let nx: f64 = tokens[3].parse().unwrap_or(0.0);
                    let ny: f64 = tokens[4].parse().unwrap_or(0.0);
                    let nz: f64 = tokens[5].parse().unwrap_or(0.0);
                    let len = (nx * nx + ny * ny + nz * nz).sqrt();
                    if len > 1e-12 {
                        normals.push([nx / len, ny / len, nz / len]);
                    } else {
                        normals.push([0.0, 1.0, 0.0]);
                    }
                }
            }
        }

        if points.is_empty() {
            return Err(format!("点云文件 '{}' 未包含任何有效数据点", path.display()));
        }

        let nrms = if has_normals && normals.len() == points.len() {
            Some(normals)
        } else {
            None
        };

        Ok(Self { points, normals: nrms })
    }

    /// 计算点云包围盒 AABB: (min, max)
    pub fn bounding_box(&self) -> ([f64; 3], [f64; 3]) {
        let mut min = [f64::INFINITY; 3];
        let mut max = [f64::NEG_INFINITY; 3];
        for p in &self.points {
            for k in 0..3 {
                min[k] = min[k].min(p[k]);
                max[k] = max[k].max(p[k]);
            }
        }
        (min, max)
    }

    /// 计算点云几何中心
    pub fn centroid(&self) -> [f64; 3] {
        let n = self.points.len() as f64;
        let mut sum = [0.0; 3];
        for p in &self.points {
            sum[0] += p[0];
            sum[1] += p[1];
            sum[2] += p[2];
        }
        [sum[0] / n, sum[1] / n, sum[2] / n]
    }

    /// 估算点云平均点间距 (使用局部随机抽样计算最近邻距离)
    pub fn estimate_average_spacing(&self) -> f64 {
        let n = self.points.len();
        if n < 2 {
            return 1.0;
        }

        let sample_count = n.min(200);
        let step = (n / sample_count).max(1);
        let mut sum_dist = 0.0;
        let mut count = 0;

        for i in (0..n).step_by(step) {
            let pi = self.points[i];
            let mut min_d2 = f64::INFINITY;
            for j in 0..n {
                if i == j {
                    continue;
                }
                let pj = self.points[j];
                let dx = pi[0] - pj[0];
                let dy = pi[1] - pj[1];
                let dz = pi[2] - pj[2];
                let d2 = dx * dx + dy * dy + dz * dz;
                if d2 < min_d2 {
                    min_d2 = d2;
                }
            }
            if min_d2.is_finite() {
                sum_dist += min_d2.sqrt();
                count += 1;
            }
        }

        if count > 0 {
            sum_dist / count as f64
        } else {
            1.0
        }
    }

    /// 将点云输出保存至 .xyz 文件
    pub fn save_to_xyz(&self, path: &Path) -> Result<(), String> {
        let mut file = File::create(path)
            .map_err(|e| format!("创建文件 '{}' 失败: {e}", path.display()))?;
        for (i, p) in self.points.iter().enumerate() {
            if let Some(ref nrms) = self.normals {
                let n = nrms[i];
                writeln!(file, "{:.6} {:.6} {:.6} {:.6} {:.6} {:.6}", p[0], p[1], p[2], n[0], n[1], n[2])
                    .map_err(|e| format!("写入数据失败: {e}"))?;
            } else {
                writeln!(file, "{:.6} {:.6} {:.6}", p[0], p[1], p[2])
                    .map_err(|e| format!("写入数据失败: {e}"))?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_cloud_basic() {
        let pts = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let pc = PointCloud::new(pts, None).unwrap();
        assert_eq!(pc.points.len(), 3);
        let (min, max) = pc.bounding_box();
        assert_eq!(min, [0.0, 0.0, 0.0]);
        assert_eq!(max, [1.0, 1.0, 0.0]);
        assert_eq!(pc.centroid(), [1.0 / 3.0, 1.0 / 3.0, 0.0]);
    }
}
