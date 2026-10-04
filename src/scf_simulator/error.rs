#[derive(Clone, Debug)]
pub struct ScfError {
    pub kind: &'static str,
    pub code: i32,
    pub detail: String,
}

impl ScfError {
    pub fn model_not_closed(detail: impl Into<String>) -> Self {
        Self {
            kind: "ModelNotClosed",
            code: 10,
            detail: detail.into(),
        }
    }

    pub fn rigid_rank_deficient(detail: impl Into<String>) -> Self {
        Self {
            kind: "RigidRankDeficient",
            code: 11,
            detail: detail.into(),
        }
    }

    pub fn no_core(detail: impl Into<String>) -> Self {
        Self {
            kind: "NoCore",
            code: 12,
            detail: detail.into(),
        }
    }

    pub fn core_degenerate(detail: impl Into<String>) -> Self {
        Self {
            kind: "CoreDegenerate",
            code: 32,
            detail: detail.into(),
        }
    }

    pub fn rheology_not_converged(detail: impl Into<String>) -> Self {
        Self {
            kind: "RheologyNotConverged",
            code: 20,
            detail: detail.into(),
        }
    }

    pub fn trust_radius_exhausted(detail: impl Into<String>) -> Self {
        Self {
            kind: "TrustRadiusExhausted",
            code: 23,
            detail: detail.into(),
        }
    }

    pub fn model_unsupported(detail: impl Into<String>) -> Self {
        Self {
            kind: "ModelUnsupported",
            code: 14,
            detail: detail.into(),
        }
    }

    pub fn geometry_invalid(detail: impl Into<String>) -> Self {
        Self {
            kind: "GeometryInvalid",
            code: 13,
            detail: detail.into(),
        }
    }

    pub fn picard_not_converged(detail: impl Into<String>) -> Self {
        Self {
            kind: "PicardNotConverged",
            code: 21,
            detail: detail.into(),
        }
    }

    pub fn tier1_error_unknown(detail: impl Into<String>) -> Self {
        Self {
            kind: "Tier1ErrorUnknown",
            code: 22,
            detail: detail.into(),
        }
    }
}

impl std::fmt::Display for ScfError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} ({}): {}", self.kind, self.code, self.detail)
    }
}

impl std::error::Error for ScfError {}
