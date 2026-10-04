use bevy::prelude::*;

use crate::asteroid_app::components::StepRecord;

#[derive(Component)]
pub struct DiagnosticsHud;

pub fn spawn_diagnostics(mut commands: Commands) {
    commands.spawn((
        DiagnosticsHud,
        Text::new("step 0"),
        TextFont::from_font_size(16.0),
        TextColor(Color::srgb(0.92, 0.94, 0.96)),
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(12.0),
            left: Val::Px(12.0),
            ..default()
        },
    ));
}

pub fn refresh_diagnostics(
    record: Query<&StepRecord>,
    mut label: Query<&mut Text, With<DiagnosticsHud>>,
    time: Res<Time>,
) {
    let Ok(record) = record.single() else {
        return;
    };
    let Ok(mut text) = label.single_mut() else {
        return;
    };
    let fps = if time.delta_secs() > 0.0 {
        1.0 / time.delta_secs()
    } else {
        0.0
    };
    text.0 = format!(
        "step {step}   t {t:.3}\nomega [{ox:.4}, {oy:.4}, {oz:.4}]\nyielded {yielded}   mobile {mobile:.3}\nmass error {error:.3e}   fps {fps:.0}",
        step = record.step,
        t = record.time,
        ox = record.omega[0],
        oy = record.omega[1],
        oz = record.omega[2],
        yielded = record.yielded,
        mobile = record.mobile_fraction,
        error = record.mass_error,
        fps = fps,
    );
}
