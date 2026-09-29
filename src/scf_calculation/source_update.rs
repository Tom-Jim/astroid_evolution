#[derive(Clone, Copy, Debug, Default)]
pub struct ModalSource {
    pub values: [f64; 4],
}

pub fn apply_modal_source(m: [f64; 4], source: ModalSource, dt: f64) -> [f64; 4] {
    [m[0] + dt*source.values[0], m[1] + dt*source.values[1],
     m[2] + dt*source.values[2], m[3] + dt*source.values[3]]
}
