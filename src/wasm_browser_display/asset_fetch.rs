//! 浏览器资源异步加载与格式解析
//! 严格依照 docs/software.md 第 14、15.7 节 与 ryugu_cauchy_gravity 实现
//!
//! 支持通过 HTTP Fetch 加载远程模型/配置，或直接接收来自文件拖放 (Drag & Drop) 的字节流。

use serde::{Deserialize, Serialize};

/// 浏览器资源加载类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AssetType {
    /// 3D 三角网格模型 (.glb / .gltf)
    GlbModel,
    /// 密度核与求解配置 (.toml / .json)
    ConfigToml,
    /// 散乱激光雷达点云 (.txt / .xyz)
    PointCloud,
    /// WGSL 计算着色器 (.wgsl)
    WgslShader,
}

/// 判定 URL 或文件名的资源类型
pub fn detect_asset_type(filename_or_url: &str) -> AssetType {
    let lower = filename_or_url.to_lowercase();
    if lower.ends_with(".glb") || lower.ends_with(".gltf") {
        AssetType::GlbModel
    } else if lower.ends_with(".toml") || lower.ends_with(".json") {
        AssetType::ConfigToml
    } else if lower.ends_with(".wgsl") {
        AssetType::WgslShader
    } else {
        AssetType::PointCloud
    }
}

/// 异步加载资源原始字节数据
pub async fn fetch_asset_bytes(url: &str) -> Result<Vec<u8>, String> {
    #[cfg(target_arch = "wasm32")]
    {
        use wasm_bindgen::JsCast;
        use wasm_bindgen_futures::JsFuture;

        let window = web_sys::window().ok_or_else(|| "Window 对象不存在".to_string())?;
        let resp_value = JsFuture::from(window.fetch_with_str(url))
            .await
            .map_err(|e| format!("Fetch 请求失败 {url}: {e:?}"))?;

        let resp: web_sys::Response = resp_value
            .dyn_into()
            .map_err(|_| "Response 类型转换失败".to_string())?;

        if !resp.ok() {
            return Err(format!("HTTP 状态码错误: {}", resp.status()));
        }

        let array_buffer_promise = resp
            .array_buffer()
            .map_err(|e| format!("ArrayBuffer 获取失败: {e:?}"))?;

        let array_buffer = JsFuture::from(array_buffer_promise)
            .await
            .map_err(|e| format!("ArrayBuffer 解析失败: {e:?}"))?;

        let uint8_array = js_sys::Uint8Array::new(&array_buffer);
        Ok(uint8_array.to_vec())
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        // 原生文件系统读取支持
        std::fs::read(url).map_err(|e| format!("本地文件读取失败 {url}: {e}"))
    }
}

/// 异步加载资源文本内容 (如 TOML / WGSL)
pub async fn fetch_asset_text(url: &str) -> Result<String, String> {
    let bytes = fetch_asset_bytes(url).await?;
    String::from_utf8(bytes).map_err(|e| format!("UTF-8 字符解码失败: {e}"))
}

/// 解析浏览器拖放 (Drag & Drop) 或上传的散乱点云文本/字节流
pub fn parse_dropped_xyz_point_cloud(bytes: &[u8]) -> Result<Vec<[f64; 3]>, String> {
    let text = std::str::from_utf8(bytes).map_err(|e| format!("无效 UTF-8 编码: {e}"))?;
    let mut points = Vec::new();

    for (line_idx, line) in text.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') || trimmed.starts_with("//") {
            continue;
        }

        let parts: Vec<&str> = trimmed
            .split(|c: char| c.is_whitespace() || c == ',')
            .filter(|s| !s.is_empty())
            .collect();

        if parts.len() < 3 {
            return Err(format!("第 {} 行坐标分量不足 3 个: '{}'", line_idx + 1, line));
        }

        let x: f64 = parts[0].parse().map_err(|e| format!("第 {} 行 X 分量解析失败: {e}", line_idx + 1))?;
        let y: f64 = parts[1].parse().map_err(|e| format!("第 {} 行 Y 分量解析失败: {e}", line_idx + 1))?;
        let z: f64 = parts[2].parse().map_err(|e| format!("第 {} 行 Z 分量解析失败: {e}", line_idx + 1))?;

        points.push([x, y, z]);
    }

    if points.is_empty() {
        return Err("点云数据为空".to_string());
    }

    Ok(points)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_asset_type() {
        assert_eq!(detect_asset_type("ryugu.glb"), AssetType::GlbModel);
        assert_eq!(detect_asset_type("models/phobos.gltf"), AssetType::GlbModel);
        assert_eq!(detect_asset_type("cauchy_elliptic.toml"), AssetType::ConfigToml);
        assert_eq!(detect_asset_type("shader.wgsl"), AssetType::WgslShader);
        assert_eq!(detect_asset_type("points.xyz"), AssetType::PointCloud);
    }

    #[test]
    fn test_parse_dropped_xyz_point_cloud() {
        let raw = b"# Header comment\n1.0 2.0 3.0\n4.5, 5.5, 6.5\n// Another comment\n-1.0 -2.0 -3.0\n";
        let pts = parse_dropped_xyz_point_cloud(raw).expect("parse failed");
        assert_eq!(pts.len(), 3);
        assert_eq!(pts[0], [1.0, 2.0, 3.0]);
        assert_eq!(pts[1], [4.5, 5.5, 6.5]);
        assert_eq!(pts[2], [-1.0, -2.0, -3.0]);

        assert!(parse_dropped_xyz_point_cloud(b"# only comments\n").is_err());
        assert!(parse_dropped_xyz_point_cloud(b"invalid data string").is_err());
    }
}

