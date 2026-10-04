use std::fmt;

#[derive(Clone, Debug)]
pub struct RtfpError {
    pub kind: &'static str,
    pub detail: String,
}

impl RtfpError {
    pub fn model_not_closed(detail: impl Into<String>) -> Self {
        Self {
            kind: "ModelNotClosed",
            detail: detail.into(),
        }
    }

    pub fn gpu_unavailable(detail: impl Into<String>) -> Self {
        Self {
            kind: "GpuUnavailable",
            detail: detail.into(),
        }
    }

    pub fn geometry(detail: impl Into<String>) -> Self {
        Self {
            kind: "GeometryInvalid",
            detail: detail.into(),
        }
    }

    pub fn unsupported(detail: impl Into<String>) -> Self {
        Self {
            kind: "Unsupported",
            detail: detail.into(),
        }
    }
}

impl RtfpError {
    pub fn code(&self) -> i32 {
        match self.kind {
            "GpuUnavailable" => 50,
            "GeometryInvalid" => 11,
            "Unsupported" => 12,
            _ => 10,
        }
    }
}

impl fmt::Display for RtfpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({}): {}", self.kind, self.code(), self.detail)
    }
}

impl std::error::Error for RtfpError {}

impl From<crate::common::error::CommonError> for RtfpError {
    fn from(value: crate::common::error::CommonError) -> Self {
        Self {
            kind: value.kind,
            detail: value.detail,
        }
    }
}
