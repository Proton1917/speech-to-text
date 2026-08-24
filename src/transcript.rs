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
            Self::Quality => "protected_host_readability_cleanup_on_primary_asr",
            Self::Raw => "unpolished_primary_asr_not_verbatim_guaranteed",
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
            Self::Quality => "事实保护可读性清稿",
            Self::Raw => "单路 ASR 原始输出稿",
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
        assert_eq!(
            TranscriptMode::Quality.editing_policy(),
            "protected_host_readability_cleanup_on_primary_asr"
        );
        assert_eq!(
            TranscriptMode::Raw.editing_policy(),
            "unpolished_primary_asr_not_verbatim_guaranteed"
        );
        assert_eq!(TranscriptMode::Quality.title(), "事实保护可读性清稿");
        assert_eq!(TranscriptMode::Raw.title(), "单路 ASR 原始输出稿");
    }
}
