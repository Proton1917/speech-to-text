#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum TranscriptMode {
    #[default]
    Quality,
    Raw,
}

impl TranscriptMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Quality => "quality",
            Self::Raw => "raw",
        }
    }

    pub const fn editing_policy(self) -> &'static str {
        match self {
            Self::Quality => "faithful_readability_cleanup",
            Self::Raw => "verbatim",
        }
    }

    pub const fn output_extension(self) -> &'static str {
        match self {
            Self::Quality => "md",
            Self::Raw => "raw.md",
        }
    }

    pub const fn title(self) -> &'static str {
        match self {
            Self::Quality => "高质量转写稿",
            Self::Raw => "原始逐字稿",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modes_have_distinct_contracts_and_outputs() {
        assert_eq!(TranscriptMode::default(), TranscriptMode::Quality);
        assert_eq!(TranscriptMode::Quality.as_str(), "quality");
        assert_eq!(TranscriptMode::Quality.output_extension(), "md");
        assert_eq!(TranscriptMode::Raw.as_str(), "raw");
        assert_eq!(TranscriptMode::Raw.output_extension(), "raw.md");
        assert_ne!(
            TranscriptMode::Quality.editing_policy(),
            TranscriptMode::Raw.editing_policy()
        );
    }
}
