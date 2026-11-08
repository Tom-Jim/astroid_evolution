//! The segment table `𝒯 = {(m, q, ℓ, R⁻, R⁺, flags)}` of MS §3.15.1: the
//! occupied ray intervals of one point set, built once per geometry by
//! `rtfp/shader/segments.wesl` and kept device-resident, so a density update
//! (`W3`) or another field evaluation of the same point set no longer traces
//! the surface (Proposition 3.15.1).

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use crate::common::error::SimError;
use crate::gpu::{Job, Res, kernels};
use crate::rtfp::api::PointSet;
use crate::rtfp::lebedev;
use crate::rtfp::solver::RtfpSolver;

/// Intervals kept per (point, ray) row; a ray needing more marks overflow and
/// is retraced inline by the field kernel. Mirrors `SEG_CAP` of
/// `rtfp/shader/segments.wesl`.
pub(crate) const SEG_CAP: usize = 8;

/// The table is built only when its store stays below this many (point, ray)
/// rows; beyond it the field kernel keeps tracing every evaluation.
const SEG_TABLE_MAX_ROWS: usize = 3_000_000;

/// One point set's table: interval counts and `(R⁻, R⁺)` pairs, both row-major
/// over `(m·ray_count + q)`, device-resident.
pub(crate) struct SegmentTable {
    pub counts: Res,
    pub pairs: Res,
    /// Rows whose orientation-rule build saw an anomaly and fell back to the
    /// parity construction (MS §3.6 step 4).
    pub dirty: u32,
    /// Rows that needed more than `SEG_CAP` intervals.
    pub overflow: u32,
    /// Total table rows, `n_points · ray_count`.
    pub rows: usize,
}

/// A cheap identity of a point set: its size and every coordinate's bits.
/// Two evaluations of the same coordinates hit the same table.
pub(crate) fn point_key(points: &PointSet) -> u64 {
    let mut h = DefaultHasher::new();
    points.len().hash(&mut h);
    for c in [&points.x, &points.y, &points.z] {
        for v in c {
            v.to_bits().hash(&mut h);
        }
    }
    h.finish()
}

impl SegmentTable {
    /// Whether a table may be built for `n_points`: bounded store and at least
    /// one ray to put in it.
    pub(crate) fn fits(n_points: usize) -> bool {
        n_points > 0 && n_points * lebedev::COUNT <= SEG_TABLE_MAX_ROWS
    }

    /// Build `𝒯` for the whole point set: one `rtfp_segments` dispatch over
    /// `n_points · ray_count` threads. The count buffer is read back once to
    /// tally the dirty and overflow flags.
    pub(crate) fn build(solver: &RtfpSolver, points: &PointSet) -> Result<Self, SimError> {
        let n = points.len();
        let r = lebedev::COUNT;
        let rows = n * r;
        let frame = solver.body.frame;
        let pts: Vec<[f32; 4]> = (0..n)
            .map(|i| {
                let u = frame.to_unit(points.point(i));
                [u.x as f32, u.y as f32, u.z as f32, 0.0]
            })
            .collect();
        let gpu = SimError::gpu;
        let counts = Res::zeroed((rows * 4) as u64).map_err(&gpu)?;
        let pairs = Res::zeroed((rows * 2 * SEG_CAP * 4) as u64).map_err(&gpu)?;
        let mut job = Job::new();
        let points_res = job.input(&pts);
        let params = job.value(&solver.chord_params(rows as u32, false, false, false, 0, 0));
        job.run(
            &kernels::RTFP_SEGMENTS,
            kernels::RTFP_SEGMENTS.groups(rows),
            &[
                points_res,
                (&solver.rays).into(),
                params,
                (&counts).into(),
                (&pairs).into(),
                (&solver.body.accel).into(),
            ],
        );
        let rc = job.read_all::<u32>(&counts, rows);
        let out = job.submit().map_err(gpu)?;
        let mut dirty = 0;
        let mut overflow = 0;
        for &c in out.get::<u32>(rc).iter() {
            dirty += (c >> 31) & 1;
            overflow += (c >> 30) & 1;
        }
        Ok(Self {
            counts,
            pairs,
            dirty,
            overflow,
            rows,
        })
    }
}
