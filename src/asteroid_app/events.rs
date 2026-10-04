use bevy::prelude::*;

/// Consumed only while [`crate::asteroid_app::resources::StepPhase::Idle`].
#[derive(Message, Clone, Debug)]
pub struct ParamPatch {
    pub key: String,
    pub value: f64,
}
