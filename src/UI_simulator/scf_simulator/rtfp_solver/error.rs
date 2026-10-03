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

    pub fn geometry(detail: impl Into<String>) -> Self {
        Self {
            kind: "GeometryInvalid",
            detail: detail.into(),
        }
    }
}

impl fmt::Display for RtfpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.kind, self.detail)
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
