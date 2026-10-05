use std::fmt;

/// Failure of the smoother. `sh-cli` exits with `code()`.
#[derive(Clone, Debug)]
pub struct ShError {
    pub kind: &'static str,
    pub detail: String,
}

impl ShError {
    pub fn model_not_closed(detail: impl Into<String>) -> Self {
        Self {
            kind: "ModelNotClosed",
            detail: detail.into(),
        }
    }

    pub fn geometry(detail: impl Into<String>) -> Self {
        Self {
            kind: "GeometryInvalid",
            detail: detail.into(),
        }
    }

    pub fn invalid_input(detail: impl Into<String>) -> Self {
        Self {
            kind: "InvalidInput",
            detail: detail.into(),
        }
    }

    pub fn gpu_unavailable(detail: impl Into<String>) -> Self {
        Self {
            kind: "GpuUnavailable",
            detail: detail.into(),
        }
    }

    pub fn code(&self) -> i32 {
        match self.kind {
            "ModelNotClosed" => 10,
            "GeometryInvalid" => 11,
            "InvalidInput" => 12,
            "GpuUnavailable" => 50,
            _ => 1,
        }
    }
}

impl fmt::Display for ShError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({}): {}", self.kind, self.code(), self.detail)
    }
}

impl std::error::Error for ShError {}
