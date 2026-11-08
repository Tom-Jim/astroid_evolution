//! The one failure type of every module. Each CLI exits with `code()`.

use std::fmt;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SimError {
    pub kind: &'static str,
    pub detail: String,
}

macro_rules! kinds {
    ($($(#[$doc:meta])* $name:ident => $kind:literal, $code:literal;)*) => {
        impl SimError {
            $(
                $(#[$doc])*
                pub fn $name(detail: impl Into<String>) -> Self {
                    Self { kind: $kind, detail: detail.into() }
                }
            )*

            pub fn code(&self) -> i32 {
                match self.kind {
                    $($kind => $code,)*
                    _ => 1,
                }
            }
        }
    };
}

kinds! {
    /// A required constant or input is missing or out of range (§10.3.4 step 0).
    /// Callers must not substitute a default.
    model_not_closed => "ModelNotClosed", 10;
    /// The body is not an admissible polyhedral solid of hypothesis (G).
    geometry => "GeometryInvalid", 11;
    invalid_input => "InvalidInput", 12;
    /// `B_rb` lost rank, or the load is incompatible with the rigid modes (§A.3).
    rigid_rank => "RigidRankDeficient", 13;
    /// The characteristic-ray relaxation, the weak momentum balance, or the
    /// friction chain did not converge (§10.0.10 a, b).
    rheology_not_converged => "RheologyNotConverged", 20;
    /// The load exceeds the limit load at the frozen pressure (`L* > 1`, §10.0.12.1): no
    /// quasi-static state exists; the certified bracket showed the certificate.
    limit_load_collapse => "LimitLoadCollapse", 21;
    /// The outer geometry acceptance failed at the trust-radius floor (§10.3.3).
    trust_exhausted => "TrustRadiusExhausted", 23;
    /// The configured `step.max_steps` limit was reached before `T_end` (§10.3.4 step 9).
    step_limit_exhausted => "StepLimitExhausted", 24;
    solver => "SolverFailure", 30;
    io => "Io", 31;
    gpu => "GpuUnavailable", 50;
}

impl fmt::Display for SimError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({}): {}", self.kind, self.code(), self.detail)
    }
}

impl std::error::Error for SimError {}

impl SimError {
    /// Exit status of a CLI run: success, or the message on stderr and `code()`.
    pub fn exit(result: Result<(), Self>) -> std::process::ExitCode {
        match result {
            Ok(()) => std::process::ExitCode::SUCCESS,
            Err(err) => {
                eprintln!("{err}");
                std::process::ExitCode::from(err.code().clamp(1, 255) as u8)
            }
        }
    }
}

impl From<std::io::Error> for SimError {
    fn from(e: std::io::Error) -> Self {
        Self::io(e.to_string())
    }
}

impl From<serde_json::Error> for SimError {
    fn from(e: serde_json::Error) -> Self {
        Self::io(e.to_string())
    }
}
