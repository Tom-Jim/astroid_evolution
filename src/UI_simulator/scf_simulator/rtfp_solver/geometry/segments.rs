use crate::rtfp_solver::error::RtfpError;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RaySegment {
    pub r_min: f64,
    pub r_max: f64,
}

/// Pair crossings using the occupancy at `R = 0`.
///
/// Interior: `[0, R1], [R2, R3], …`. Exterior: `[R1, R2], [R3, R4], …`.
/// Duplicate distances from a shared edge or vertex collapse to one crossing.
/// A boundary point is not accepted here; the caller must use the finite-part branch.
pub fn segments_from_hits(hits: &[f64], inside: bool) -> Result<Vec<RaySegment>, RtfpError> {
    let mut clean: Vec<f64> = Vec::with_capacity(hits.len());
    for &t in hits {
        if !(t.is_finite() && t > 0.0) {
            continue;
        }
        if let Some(&last) = clean.last() {
            let tol = (1e-9 * t).max(1e-12);
            if (t - last).abs() > tol {
                clean.push(t);
            }
        } else {
            clean.push(t);
        }
    }
    if inside == (clean.len() % 2 == 0) {
        return Err(RtfpError::model_not_closed(format!(
            "ray parity disagrees with the inside test: inside={inside}, crossings={}",
            clean.len()
        )));
    }
    let mut segs = Vec::new();
    if inside {
        if clean.is_empty() {
            return Err(RtfpError::model_not_closed(
                "interior ray produced no exit intersection",
            ));
        }
        segs.push(RaySegment {
            r_min: 0.0,
            r_max: clean[0],
        });
        let mut i = 1;
        while i + 1 < clean.len() {
            segs.push(RaySegment {
                r_min: clean[i],
                r_max: clean[i + 1],
            });
            i += 2;
        }
    } else {
        let mut i = 0;
        while i + 1 < clean.len() {
            segs.push(RaySegment {
                r_min: clean[i],
                r_max: clean[i + 1],
            });
            i += 2;
        }
    }
    Ok(segs)
}
