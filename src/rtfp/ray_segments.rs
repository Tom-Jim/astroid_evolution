//! 射线交段生成与近场/远场分解
//! 严格依照 docs/software.md 第 7.1 节与 docs/mathstrict.md 第 1.4-1.5 节

/// 射线交段 [R_minus, R_plus]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RaySegment {
    pub r_min: f64,
    pub r_max: f64,
}

impl RaySegment {
    #[inline]
    pub fn new(r_min: f64, r_max: f64) -> Self {
        Self { r_min, r_max }
    }

    #[inline]
    pub fn length(&self) -> f64 {
        (self.r_max - self.r_min).max(0.0)
    }
}

/// 根据光线碰撞距离序列与起点内外属性构造严格实体交段
///
/// 外部点 (inside_at_zero = false):
///   线段序列为: [R_1, R_2], [R_3, R_4], ...
/// 内部点 (inside_at_zero = true):
///   线段序列为: [0, R_1], [R_2, R_3], [R_4, R_5], ...
///
/// 严禁仅做无条件的简单两两配对，必须保持非凸多面体的拓扑正确性。
pub fn segments_from_hits(hits: &[f64], inside_at_zero: bool) -> Vec<RaySegment> {
    if hits.is_empty() {
        return Vec::new();
    }

    // 过滤重复交点 (去重 epsilon 阈值，消除共边/共顶点射线微小抖动)
    let mut clean_hits = Vec::with_capacity(hits.len());
    for &h in hits {
        if h <= 1e-12 {
            continue;
        }
        if let Some(&last) = clean_hits.last() {
            let diff: f64 = h - last;
            if diff.abs() > 1e-7 {
                clean_hits.push(h);
            }
        } else {
            clean_hits.push(h);
        }
    }

    let mut segments = Vec::new();

    if inside_at_zero {
        // 内部点: 首段从 0 开始至第一个穿出点 R_1
        if !clean_hits.is_empty() {
            segments.push(RaySegment::new(0.0, clean_hits[0]));
            // 之后的段两两配对: [R_2, R_3], [R_4, R_5], ...
            let mut i = 1;
            while i + 1 < clean_hits.len() {
                let r_enter = clean_hits[i];
                let r_exit = clean_hits[i + 1];
                if r_exit > r_enter + 1e-9 {
                    segments.push(RaySegment::new(r_enter, r_exit));
                }
                i += 2;
            }
        }
    } else {
        // 外部点: 射线自真空射入实体 [R_1, R_2], [R_3, R_4], ...
        let mut i = 0;
        while i + 1 < clean_hits.len() {
            let r_enter = clean_hits[i];
            let r_exit = clean_hits[i + 1];
            if r_exit > r_enter + 1e-9 {
                segments.push(RaySegment::new(r_enter, r_exit));
            }
            i += 2;
        }
    }

    segments
}

/// 按照切锥近场半径 r_0 将交段分解为近场交段与远场交段 (docs/mathstrict.md 第 1.5 节)
/// E_x^near = E_x \cap (0, r_0)
/// E_x^far  = E_x \cap [r_0, \infty)
pub fn split_segments_near_far(
    segments: &[RaySegment],
    r0: f64,
) -> (Vec<RaySegment>, Vec<RaySegment>) {
    let mut near = Vec::new();
    let mut far = Vec::new();

    for seg in segments {
        if seg.r_max <= r0 {
            // 完全位于近场
            near.push(*seg);
        } else if seg.r_min >= r0 {
            // 完全位于远场
            far.push(*seg);
        } else {
            // 跨越 r0 端点，严格划分为互不重叠的两段
            near.push(RaySegment::new(seg.r_min, r0));
            far.push(RaySegment::new(r0, seg.r_max));
        }
    }

    (near, far)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_external_segments() {
        let hits = vec![10.0, 20.0, 30.0, 40.0];
        let segs = segments_from_hits(&hits, false);
        assert_eq!(segs.len(), 2);
        assert_eq!(segs[0], RaySegment::new(10.0, 20.0));
        assert_eq!(segs[1], RaySegment::new(30.0, 40.0));
    }

    #[test]
    fn test_internal_segments() {
        let hits = vec![15.0, 30.0, 45.0];
        let segs = segments_from_hits(&hits, true);
        assert_eq!(segs.len(), 2);
        assert_eq!(segs[0], RaySegment::new(0.0, 15.0));
        assert_eq!(segs[1], RaySegment::new(30.0, 45.0));
    }

    #[test]
    fn test_near_far_split() {
        let segs = vec![RaySegment::new(0.0, 50.0)];
        let (near, far) = split_segments_near_far(&segs, 10.0);
        assert_eq!(near, vec![RaySegment::new(0.0, 10.0)]);
        assert_eq!(far, vec![RaySegment::new(10.0, 50.0)]);
    }
}
