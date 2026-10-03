use std::fmt;

/// Missing or non-finite model data. Callers must not substitute a default.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CommonError {
    pub kind: &'static str,
    pub detail: String,
}

impl CommonError {
    pub fn model_not_closed(detail: impl Into<String>) -> Self {
        Self {
            kind: "ModelNotClosed",
            detail: detail.into(),
        }
    }
}

impl fmt::Display for CommonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.kind, self.detail)
    }
}

impl std::error::Error for CommonError {}
