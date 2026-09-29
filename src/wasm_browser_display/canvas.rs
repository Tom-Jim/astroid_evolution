//! 浏览器 HTML5 Canvas 与视口交互管理
//! 严格依照 docs/software.md 第 13、14 节 与 ryugu_cauchy_gravity 实现

use serde::{Deserialize, Serialize};

/// Canvas 尺寸与 DPI 配置
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CanvasViewport {
    /// 物理像素宽度
    pub width: u32,
    /// 物理像素高度
    pub height: u32,
    /// 屏幕设备像素比 (Retina HiDPI)
    pub device_pixel_ratio: f64,
}


impl CanvasViewport {
    /// 从 CSS 逻辑像素与设备像素比构建物理视口
    pub fn from_css_size(css_width: f64, css_height: f64, device_pixel_ratio: f64) -> Self {
        let dpr = device_pixel_ratio.max(0.1);
        let width = (css_width * dpr).round() as u32;
        let height = (css_height * dpr).round() as u32;
        Self {
            width: width.max(1),
            height: height.max(1),
            device_pixel_ratio: dpr,
        }
    }

    /// 获取宽高比 Aspect Ratio (width / height)
    pub fn aspect_ratio(&self) -> f64 {
        if self.height == 0 {
            1.0
        } else {
            self.width as f64 / self.height as f64
        }
    }

    /// 获取 CSS 逻辑宽度
    pub fn css_width(&self) -> f64 {
        self.width as f64 / self.device_pixel_ratio
    }

    /// 获取 CSS 逻辑高度
    pub fn css_height(&self) -> f64 {
        self.height as f64 / self.device_pixel_ratio
    }
}

impl Default for CanvasViewport {
    fn default() -> Self {
        Self {
            width: 1280,
            height: 720,
            device_pixel_ratio: 1.0,
        }
    }
}

/// 获取当前视口配置 (WASM 下读取 DOM，原生下返回默认)
pub fn get_current_viewport(canvas_id: &str) -> CanvasViewport {
    #[cfg(target_arch = "wasm32")]
    {
        use wasm_bindgen::JsCast;

        let window = match web_sys::window() {
            Some(w) => w,
            None => return CanvasViewport::default(),
        };

        let dpr = window.device_pixel_ratio();
        let document = match window.document() {
            Some(d) => d,
            None => return CanvasViewport::default(),
        };

        if let Some(canvas_elem) = document.get_element_by_id(canvas_id) {
            if let Ok(canvas) = canvas_elem.dyn_into::<web_sys::HtmlCanvasElement>() {
                let css_width = canvas.client_width() as f64;
                let css_height = canvas.client_height() as f64;
                let width = (css_width * dpr).round() as u32;
                let height = (css_height * dpr).round() as u32;
                return CanvasViewport {
                    width: width.max(100),
                    height: height.max(100),
                    device_pixel_ratio: dpr,
                };
            }
        }

        CanvasViewport {
            width: 1280,
            height: 720,
            device_pixel_ratio: dpr,
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = canvas_id;
        CanvasViewport::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_canvas_viewport_default() {
        let vp = get_current_viewport("bevy");
        assert!(vp.width >= 100);
        assert!(vp.height >= 100);
        assert!(vp.device_pixel_ratio >= 1.0);
    }
}
