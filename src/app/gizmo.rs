//! The spin axis and the sense of rotation of the body frame, drawn as gizmos.
//!
//! Tier 2 keeps the body in its principal-axis frame with the spin about `+z`
//! (hypothesis A-S), `Ω = ω e_z`, so the axis is the frame's `z` axis. The ring turns in
//! the sense of `Ω` once every `TURN_SECONDS` of wall time; the body's own period, hours,
//! is in the telemetry.

use std::f32::consts::TAU;

use bevy::prelude::*;

use crate::app::playback::{Playback, VisMode};

const AXIS_COLOR: Color = Color::srgb(0.35, 0.85, 1.0);
const RING_COLOR: Color = Color::srgba(0.35, 0.85, 1.0, 0.55);
/// Wall-clock seconds per drawn turn.
const TURN_SECONDS: f32 = 6.0;
/// Axis and ring sizes in units of the largest body radius on screen.
const AXIS_REACH: f32 = 1.45;
const RING_RADIUS: f32 = 1.2;
const RING_SEGMENTS: usize = 96;
const ARROWS: usize = 3;
/// Arc covered by each rotation arrow (rad).
const ARROW_ARC: f32 = 0.45;

/// Draw the axis through the body with its arrow at `+Ω`, and the equatorial ring with
/// arrows that circulate in the sense of rotation. Nothing is drawn before the first
/// Tier 2 frame: the startup mesh is still in the frame of the imported file.
pub fn draw_spin(mut gizmos: Gizmos, play: Res<Playback>, mode: Res<VisMode>, time: Res<Time>) {
    if !mode.spin_axis || play.generation == 0 || !(play.reach > 0.0) {
        return;
    }
    let r = play.reach;
    let sense = if play.omega < 0.0 { -1.0 } else { 1.0 };
    let tip = Vec3::Z * (sense * AXIS_REACH * r);
    gizmos.line(-tip, tip * 0.8, AXIS_COLOR);
    gizmos.arrow(tip * 0.8, tip, AXIS_COLOR).with_tip_length(0.12 * r);

    let ring = RING_RADIUS * r;
    let at = |a: f32| Vec3::new(a.cos(), a.sin(), 0.0) * ring;
    gizmos.linestrip((0..=RING_SEGMENTS).map(|k| at(TAU * k as f32 / RING_SEGMENTS as f32)), RING_COLOR);
    let phase = sense * TAU * (time.elapsed_secs() / TURN_SECONDS).fract();
    for k in 0..ARROWS {
        let a0 = phase + TAU * k as f32 / ARROWS as f32;
        let a1 = a0 + sense * ARROW_ARC;
        gizmos.arrow(at(a0), at(a1), AXIS_COLOR).with_tip_length(0.08 * r);
    }
}
