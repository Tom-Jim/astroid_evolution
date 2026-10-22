//! Text field record. The extension is `.aevb`; the contents are line oriented so the
//! header fields can be audited without a separate decoder.

use crate::common::error::SimError;
use crate::common::math::{DVec3, Sym6};
use std::fmt::Write as _;
use std::path::Path;

/// Largest relative differences from the benchmark, present only when a comparison ran.
pub struct Comparison {
    pub reference: String,
    pub max_error_u: Option<f64>,
    pub max_error_grad: Option<f64>,
    pub max_error_hess: Option<f64>,
    /// Per point `(u, grad, hess)` relative errors; `None` where the benchmark has no value.
    pub per_point: Vec<[Option<f64>; 3]>,
}

pub struct FieldRecord {
    pub role: String,
    pub g: f64,
    pub density_model: String,
    pub density_unit: String,
    pub density_hash: String,
    pub mesh_hash: String,
    pub mesh_length_unit: String,
    pub faces: usize,
    /// Gauss–Legendre nodes per part of the radial rule.
    pub radial_nodes: usize,
    pub want_potential: bool,
    pub want_gradient: bool,
    pub want_hessian: bool,
    pub points: Vec<DVec3>,
    pub potential: Vec<f64>,
    pub gradient: Vec<DVec3>,
    pub hessian: Vec<Sym6>,
    pub theta: Vec<f64>,
    pub solid_angle: Vec<f64>,
    pub comparison: Option<Comparison>,
}

fn optional(value: Option<f64>) -> String {
    value.map_or_else(|| "none".to_string(), |v| format!("{v:.6e}"))
}

pub fn write_field(path: &Path, record: &FieldRecord) -> Result<(), SimError> {
    let mut text = String::new();
    let mut line = |key: &str, value: String| {
        let _ = writeln!(text, "{key} {value}");
    };
    line("aevb", "2".into());
    line(
        "backend",
        crate::gpu::gpu().map_or_else(|e| e, |g| g.adapter.clone()),
    );
    line("role", record.role.clone());
    line("g", format!("{:.16e}", record.g));
    line("density_model", record.density_model.clone());
    line("density_unit", record.density_unit.clone());
    line("density_hash", record.density_hash.clone());
    line("mesh_hash", record.mesh_hash.clone());
    line("mesh_length_unit", record.mesh_length_unit.clone());
    line("faces", record.faces.to_string());
    line(
        "angular_rule",
        format!("lebedev{}", crate::rtfp::lebedev::COUNT),
    );
    line("radial_nodes", record.radial_nodes.to_string());
    line("points", record.points.len().to_string());
    line("want_u", u8::from(record.want_potential).to_string());
    line("want_grad", u8::from(record.want_gradient).to_string());
    line("want_hess", u8::from(record.want_hessian).to_string());
    match &record.comparison {
        Some(c) => {
            line("reference", c.reference.clone());
            line("max_error_u", optional(c.max_error_u));
            line("max_error_grad", optional(c.max_error_grad));
            line("max_error_hess", optional(c.max_error_hess));
        }
        None => line("reference", "none".into()),
    }
    line(
        "columns",
        "x y z U gx gy gz Hxx Hyy Hzz Hxy Hxz Hyz theta solid_angle err_u err_grad err_hess".into(),
    );
    for i in 0..record.points.len() {
        let (p, g, h) = (record.points[i], record.gradient[i], record.hessian[i]);
        let errors = record
            .comparison
            .as_ref()
            .map_or([None; 3], |c| c.per_point[i]);
        line(
            "point",
            format!(
                "{:.16e} {:.16e} {:.16e} {:.16e} {:.16e} {:.16e} {:.16e} {:.16e} {:.16e} {:.16e} {:.16e} {:.16e} {:.16e} {:.16e} {:.16e} {} {} {}",
                p.x,
                p.y,
                p.z,
                record.potential[i],
                g.x,
                g.y,
                g.z,
                h[0],
                h[1],
                h[2],
                h[3],
                h[4],
                h[5],
                record.theta[i],
                record.solid_angle[i],
                optional(errors[0]),
                optional(errors[1]),
                optional(errors[2]),
            ),
        );
    }
    std::fs::write(path, text).map_err(|err| {
        SimError::model_not_closed(format!("cannot write {}: {err}", path.display()))
    })
}
