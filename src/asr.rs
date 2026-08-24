use anyhow::{Context, Result, bail};

use crate::chinese::normalize_to_simplified;
use crate::speaker::{LocalSpeakerTurn, LocalTranscript};

/// Hard ceiling for the user-facing comparison diagnostic. The transcript
/// contents themselves remain available through [`AsrTextComparison`].
pub const MAX_DIFFERENCE_SUMMARY_CHARS: usize = 320;

/// Locally validated ASR source text and its deterministic display projection.
///
/// Construction is intentionally restricted to [`validate_and_normalize_text`]
/// so downstream code cannot accidentally treat unchecked model output as an
/// authoritative primary transcript.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NormalizedAsrText {
    source_text: String,
    text: String,
}

impl NormalizedAsrText {
    /// Returns the validated model text before the display-only OpenCC projection.
    pub fn source_as_str(&self) -> &str {
        &self.source_text
    }

    /// Returns the validated text after the display-only OpenCC projection.
    pub fn as_str(&self) -> &str {
        &self.text
    }

    pub fn into_string(self) -> String {
        self.text
    }

    pub fn canonical_content(&self) -> String {
        canonical_content(&self.text)
    }

    fn source_canonical_content(&self) -> String {
        canonical_content(&self.source_text)
    }
}

/// The only two claims made by the local primary/verifier comparison.
///
/// `ExactConsensus` means that both independently validated source outputs have
/// the same canonical content before the display-only OpenCC projection. It is
/// deliberately not a ground-truth or accuracy verification claim.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AsrComparisonStatus {
    ExactConsensus,
    Disagreement,
}

impl AsrComparisonStatus {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ExactConsensus => "exact_consensus",
            Self::Disagreement => "disagreement",
        }
    }
}

/// Local comparison result between a primary ASR output and a quality verifier
/// output. Exactness is decided from their validated source text; the primary
/// display projection remains authoritative even when the verifier disagrees.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AsrTextComparison {
    pub status: AsrComparisonStatus,
    pub primary: NormalizedAsrText,
    pub quality_verifier: NormalizedAsrText,
    pub difference_summary: Option<String>,
}

impl AsrTextComparison {
    pub fn is_exact_consensus(&self) -> bool {
        self.status == AsrComparisonStatus::ExactConsensus
    }
}

/// Validates untrusted ASR text and applies the embedded OpenCC t2s converter.
///
/// The byte limit is enforced both before and after normalization. Whitespace
/// is preserved in the returned text; it is ignored only by
/// [`canonical_content`] when two outputs are compared.
pub fn validate_and_normalize_text(text: &str, max_bytes: usize) -> Result<NormalizedAsrText> {
    if max_bytes == 0 {
        bail!("ASR 正文字节上限必须大于 0");
    }
    validate_text_boundary(text, max_bytes, "ASR 原始正文")?;

    let normalized = normalize_to_simplified(text).context("无法将 ASR 正文归一化为简体中文")?;
    validate_text_boundary(&normalized, max_bytes, "ASR 归一化正文")?;
    if canonical_content(&normalized).is_empty() {
        bail!("ASR 正文不包含可比较的内容字符");
    }

    Ok(NormalizedAsrText {
        source_text: text.to_owned(),
        text: normalized,
    })
}

/// Produces a conservative comparison between primary and quality-verifier
/// text. Consensus is based only on pre-OpenCC source canonical equality and
/// must not be presented as ground-truth verification.
pub fn compare_primary_and_quality_verifier(
    primary: &str,
    quality_verifier: &str,
    max_bytes_per_text: usize,
) -> Result<AsrTextComparison> {
    let primary =
        validate_and_normalize_text(primary, max_bytes_per_text).context("primary ASR 正文无效")?;
    let quality_verifier = validate_and_normalize_text(quality_verifier, max_bytes_per_text)
        .context("quality verifier ASR 正文无效")?;
    Ok(compare_normalized_texts(primary, quality_verifier))
}

/// Compares two already validated ASR values without changing either one.
pub fn compare_normalized_texts(
    primary: NormalizedAsrText,
    quality_verifier: NormalizedAsrText,
) -> AsrTextComparison {
    // OpenCC is a display projection, not an injective semantic transform. Distinct source
    // characters can collapse to the same Simplified Chinese output, so cross-ASR consensus must
    // be established before that projection. The normalized display text remains authoritative
    // for Markdown rendering and speaker-turn restoration.
    let primary_canonical = primary.source_canonical_content();
    let verifier_canonical = quality_verifier.source_canonical_content();
    if primary_canonical == verifier_canonical {
        AsrTextComparison {
            status: AsrComparisonStatus::ExactConsensus,
            primary,
            quality_verifier,
            difference_summary: None,
        }
    } else {
        AsrTextComparison {
            status: AsrComparisonStatus::Disagreement,
            difference_summary: Some(bounded_difference_summary(
                &primary_canonical,
                &verifier_canonical,
            )),
            primary,
            quality_verifier,
        }
    }
}

/// Removes only Unicode/ASCII whitespace and an explicit set of presentation
/// punctuation. All other characters are retained exactly: no case folding,
/// Unicode compatibility folding, number-word conversion, or token rewriting
/// is performed.
///
/// Separators that form part of a numeric representation (for example the dot
/// in `3.14`, comma in `1,000`, or colon in `12:30`) are retained. Minus and
/// dash characters are never ignored.
pub fn canonical_content(text: &str) -> String {
    let characters = text.chars().collect::<Vec<_>>();
    characters
        .iter()
        .copied()
        .enumerate()
        .filter_map(|(index, character)| {
            if character.is_whitespace() {
                is_semantic_whitespace(&characters, index).then_some(' ')
            } else if is_ignorable_presentation_punctuation(&characters, index) {
                None
            } else {
                Some(character)
            }
        })
        .collect()
}

/// Creates the safe speaker fallback for an authoritative primary transcript.
/// The normalized primary text is copied byte-for-byte into one UNKNOWN turn;
/// no verifier text, canonical text, or inferred identity may replace it.
pub fn build_primary_fallback_transcript(
    primary: &NormalizedAsrText,
    duration_ms: u64,
) -> Result<LocalTranscript> {
    if duration_ms == 0 {
        bail!("无法为 0 ms 音频构造 ASR fallback turn");
    }
    Ok(LocalTranscript {
        has_speech: true,
        turns: vec![LocalSpeakerTurn {
            local_speaker_id: "UNKNOWN".to_owned(),
            start_ms: 0,
            end_ms: duration_ms,
            text: primary.as_str().to_owned(),
            clean_reference: false,
        }],
        activity_ranges: None,
    })
}

/// Validates that an alignment model changed only segmentation and speaker
/// labels, never the authoritative primary ASR words.
///
/// Turn text is concatenated in the supplied order, locally validated and t2s
/// normalized, and then compared with the primary canonical content. A bounded
/// diagnostic is returned on disagreement.
pub fn validate_aligned_turns_against_primary(
    turns: &[LocalSpeakerTurn],
    primary: &NormalizedAsrText,
    max_joined_bytes: usize,
) -> Result<()> {
    let primary_canonical = primary.canonical_content();
    let (_, aligned_canonical) = aligned_turn_canonical_counts(turns, max_joined_bytes)?;
    if aligned_canonical != primary_canonical {
        bail!(
            "对齐模型改写了 authoritative ASR 正文：{}",
            bounded_difference_summary(&primary_canonical, &aligned_canonical)
        );
    }
    Ok(())
}

/// Restores the authoritative primary bytes into already validated alignment
/// turns while retaining every speaker label, timestamp and reference flag.
///
/// Each alignment turn contributes only its canonical character count. Primary
/// slices are then cut at those cumulative canonical boundaries. Ignorable
/// punctuation or whitespace between two content characters stays with the
/// preceding turn; leading material stays with the first turn and all trailing
/// material stays with the final turn. No turn is changed unless the complete
/// redistribution has first been validated successfully.
pub fn restore_primary_text_to_aligned_turns(
    turns: &mut [LocalSpeakerTurn],
    primary: &NormalizedAsrText,
    max_joined_bytes: usize,
) -> Result<()> {
    let primary_canonical = primary.canonical_content();
    let (canonical_counts, aligned_canonical) =
        aligned_turn_canonical_counts(turns, max_joined_bytes)?;
    if aligned_canonical != primary_canonical {
        bail!(
            "对齐模型改写了 authoritative ASR 正文：{}",
            bounded_difference_summary(&primary_canonical, &aligned_canonical)
        );
    }

    let canonical_starts = canonical_character_byte_starts(primary.as_str());
    let expected_canonical_chars = canonical_counts.iter().try_fold(0_usize, |total, count| {
        total
            .checked_add(*count)
            .context("对齐 turn canonical 字符计数溢出")
    })?;
    if canonical_starts.len() != expected_canonical_chars {
        bail!(
            "primary canonical 字符数与对齐 turn 配额不一致：{} != {}",
            canonical_starts.len(),
            expected_canonical_chars
        );
    }

    let mut restored = Vec::with_capacity(turns.len());
    let mut slice_start = 0_usize;
    let mut consumed_canonical = 0_usize;
    for (index, count) in canonical_counts.iter().copied().enumerate() {
        consumed_canonical = consumed_canonical
            .checked_add(count)
            .context("对齐 turn canonical 累计边界溢出")?;
        let slice_end = if index + 1 == canonical_counts.len() {
            primary.as_str().len()
        } else {
            *canonical_starts
                .get(consumed_canonical)
                .context("无法在 primary 正文中定位下一 turn 的 canonical 边界")?
        };
        restored.push(primary.as_str()[slice_start..slice_end].to_owned());
        slice_start = slice_end;
    }

    if restored.concat() != primary.as_str() {
        bail!("内部错误：恢复后的 turn 正文未逐字重建 primary ASR");
    }
    for (turn, text) in turns.iter_mut().zip(restored) {
        turn.text = text;
    }
    Ok(())
}

/// Convenience wrapper for callers that already hold a local transcript.
pub fn validate_aligned_transcript_against_primary(
    transcript: &LocalTranscript,
    primary: &NormalizedAsrText,
    max_joined_bytes: usize,
) -> Result<()> {
    validate_aligned_turns_against_primary(&transcript.turns, primary, max_joined_bytes)
}

/// Convenience wrapper that restores primary text into a local transcript in
/// place after the alignment-only invariant has been proven.
pub fn restore_primary_text_to_aligned_transcript(
    transcript: &mut LocalTranscript,
    primary: &NormalizedAsrText,
    max_joined_bytes: usize,
) -> Result<()> {
    restore_primary_text_to_aligned_turns(&mut transcript.turns, primary, max_joined_bytes)
}

fn validate_text_boundary(text: &str, max_bytes: usize, label: &str) -> Result<()> {
    if text.trim().is_empty() {
        bail!("{label}不能为空或仅包含空白");
    }
    if text.len() > max_bytes {
        bail!("{label}超过字节上限：{} > {}", text.len(), max_bytes);
    }
    if text.contains('\0') {
        bail!("{label}包含 NUL 字符");
    }
    if text.contains('\u{fffd}') {
        bail!("{label}包含 Unicode replacement character");
    }
    Ok(())
}

fn is_ignorable_presentation_punctuation(characters: &[char], index: usize) -> bool {
    let character = characters[index];
    if matches!(
        character,
        ',' | '，' | '.' | ':' | '：' | '!' | '！' | '?' | '？' | ';' | '；' | '\'' | '’'
    ) && is_semantic_separator(characters, index)
    {
        return false;
    }
    matches!(
        character,
        ',' | '，'
            | '.'
            | '。'
            | '!'
            | '！'
            | '?'
            | '？'
            | ';'
            | '；'
            | ':'
            | '：'
            | '、'
            | '…'
            | '"'
            | '\''
            | '“'
            | '”'
            | '‘'
            | '’'
    )
}

fn is_semantic_separator(characters: &[char], index: usize) -> bool {
    let separator = characters[index];
    let previous = index.checked_sub(1).map(|previous| characters[previous]);
    let next = characters.get(index + 1).copied();

    // A leading decimal marker has no left token neighbor, but deleting it
    // changes the value (`.5%` becomes `5%`). Whitespace or an opening
    // delimiter before the marker is intentionally allowed; a following
    // numeric character is the semantic evidence.
    if separator == '.' && next.is_some_and(char::is_numeric) {
        return true;
    }

    // Consecutive ASCII colons carry structure in IPv6 addresses and scoped
    // identifiers. Immediate-neighbor logic cannot classify either colon in
    // `2001:db8::1` or `C::foo`. Treating every member of a colon run as
    // semantic is both conservative and constant-time even for hostile input.
    if separator == ':' && is_semantic_ascii_colon_run(characters, index) {
        return true;
    }

    let (Some(previous), Some(next)) = (previous, next) else {
        return false;
    };
    if previous.is_numeric() && next.is_numeric() {
        return true;
    }
    // Apostrophes inside non-Han Unicode alphabetic words are lexical, not
    // presentation punctuation. This covers names such as `d’Ávila` without
    // retaining straight or curly quotes between ordinary Chinese text.
    if matches!(separator, '\'' | '’')
        && previous.is_alphabetic()
        && next.is_alphabetic()
        && !is_han(previous)
        && !is_han(next)
    {
        return true;
    }
    if matches!(
        separator,
        ',' | '，' | '.' | ':' | '：' | '!' | '！' | '?' | '？' | ';' | '；' | '\'' | '’'
    ) && is_ascii_token_neighbor(previous)
        && is_ascii_token_neighbor(next)
    {
        return true;
    }
    separator == ':'
        && previous.is_ascii_alphanumeric()
        && characters.get(index + 1) == Some(&'/')
        && characters.get(index + 2) == Some(&'/')
}

fn is_semantic_ascii_colon_run(characters: &[char], index: usize) -> bool {
    (index > 0 && characters[index - 1] == ':')
        || characters.get(index + 1).is_some_and(|next| *next == ':')
}

fn is_ascii_token_neighbor(character: char) -> bool {
    character.is_ascii_alphanumeric()
        || matches!(
            character,
            '/' | '@' | '_' | '-' | '+' | '=' | '%' | '&' | '#' | '~'
        )
}

fn is_semantic_whitespace(characters: &[char], index: usize) -> bool {
    if !characters[index].is_whitespace() || (index > 0 && characters[index - 1].is_whitespace()) {
        return false;
    }
    let previous = (0..index).rev().find_map(|neighbor_index| {
        let character = characters[neighbor_index];
        (!character.is_whitespace()
            && !is_ignorable_presentation_punctuation(characters, neighbor_index))
        .then_some(character)
    });
    let next = (index + 1..characters.len()).find_map(|neighbor_index| {
        let character = characters[neighbor_index];
        (!character.is_whitespace()
            && !is_ignorable_presentation_punctuation(characters, neighbor_index))
        .then_some(character)
    });
    match (previous, next) {
        (Some(previous), Some(next)) => {
            previous.is_alphanumeric()
                && next.is_alphanumeric()
                && !(is_han(previous) && is_han(next))
        }
        _ => false,
    }
}

fn is_han(character: char) -> bool {
    matches!(
        character as u32,
        0x3400..=0x4DBF
            | 0x4E00..=0x9FFF
            | 0xF900..=0xFAFF
            | 0x20000..=0x2EBEF
            | 0x30000..=0x323AF
    )
}

fn aligned_turn_canonical_counts(
    turns: &[LocalSpeakerTurn],
    max_joined_bytes: usize,
) -> Result<(Vec<usize>, String)> {
    if max_joined_bytes == 0 {
        bail!("对齐正文的字节上限必须大于 0");
    }
    if turns.is_empty() {
        bail!("对齐模型没有返回任何 turn");
    }

    let joined_bytes = turns.iter().try_fold(0_usize, |total, turn| {
        total
            .checked_add(turn.text.len())
            .context("对齐模型 turn 正文字节计数溢出")
    })?;
    if joined_bytes > max_joined_bytes {
        bail!(
            "对齐模型 turn 拼接正文超过字节上限：{} > {}",
            joined_bytes,
            max_joined_bytes
        );
    }

    let mut raw_joined = String::with_capacity(joined_bytes);
    let mut character_owners = Vec::new();
    for (turn_index, turn) in turns.iter().enumerate() {
        validate_text_boundary(&turn.text, max_joined_bytes, "对齐模型 turn 正文")
            .with_context(|| format!("对齐模型 turn[{turn_index}] 正文无效"))?;
        raw_joined.push_str(&turn.text);
        for _ in turn.text.chars() {
            character_owners.push(turn_index);
        }
    }
    // Normalize the complete sequence once. This makes OpenCC/kana behavior
    // independent of the model's turn boundaries.
    let normalized = validate_and_normalize_text(&raw_joined, max_joined_bytes)
        .context("对齐模型拼接正文无效")?;
    let joined_characters = normalized.as_str().chars().collect::<Vec<_>>();
    if joined_characters.len() != character_owners.len() {
        bail!("对齐模型整段归一化改变了字符数量，无法安全恢复 turn 边界");
    }

    // Canonicalization must see the concatenated text so punctuation whose
    // meaning depends on both neighbors (such as a decimal point at a turn
    // boundary) is classified exactly as it is in the complete transcript.
    let mut canonical_counts = vec![0_usize; turns.len()];
    let mut joined_canonical = String::new();
    for (index, character) in joined_characters.iter().copied().enumerate() {
        if character.is_whitespace() {
            if !is_semantic_whitespace(&joined_characters, index) {
                continue;
            }
            let owner = character_owners[index];
            canonical_counts[owner] = canonical_counts[owner]
                .checked_add(1)
                .context("对齐模型 turn canonical 字符计数溢出")?;
            joined_canonical.push(' ');
            continue;
        }
        if is_ignorable_presentation_punctuation(&joined_characters, index) {
            continue;
        }
        let owner = character_owners[index];
        canonical_counts[owner] = canonical_counts[owner]
            .checked_add(1)
            .context("对齐模型 turn canonical 字符计数溢出")?;
        joined_canonical.push(character);
    }
    if let Some((index, _)) = canonical_counts
        .iter()
        .enumerate()
        .find(|(_, count)| **count == 0)
    {
        bail!("对齐模型 turn[{index}] canonical 正文为空");
    }
    Ok((canonical_counts, joined_canonical))
}

fn canonical_character_byte_starts(text: &str) -> Vec<usize> {
    let characters = text.chars().collect::<Vec<_>>();
    text.char_indices()
        .enumerate()
        .filter_map(|(index, (byte_start, character))| {
            if character.is_whitespace() {
                is_semantic_whitespace(&characters, index).then_some(byte_start)
            } else {
                (!is_ignorable_presentation_punctuation(&characters, index)).then_some(byte_start)
            }
        })
        .collect()
}

fn bounded_difference_summary(primary: &str, verifier: &str) -> String {
    let primary_chars = primary.chars().collect::<Vec<_>>();
    let verifier_chars = verifier.chars().collect::<Vec<_>>();
    let first_difference = primary_chars
        .iter()
        .zip(&verifier_chars)
        .position(|(left, right)| left != right)
        .unwrap_or_else(|| primary_chars.len().min(verifier_chars.len()));

    let primary_context = difference_context(&primary_chars, first_difference);
    let verifier_context = difference_context(&verifier_chars, first_difference);
    let primary_context =
        serde_json::to_string(&primary_context).unwrap_or_else(|_| "\"<unavailable>\"".to_owned());
    let verifier_context =
        serde_json::to_string(&verifier_context).unwrap_or_else(|_| "\"<unavailable>\"".to_owned());
    let summary = format!(
        "canonical disagreement at char {first_difference}; primary_chars={}; verifier_chars={}; primary_context={primary_context}; verifier_context={verifier_context}",
        primary_chars.len(),
        verifier_chars.len()
    );
    bounded_chars(&summary, MAX_DIFFERENCE_SUMMARY_CHARS)
}

fn difference_context(characters: &[char], difference: usize) -> String {
    const BEFORE: usize = 16;
    const AFTER: usize = 32;
    let start = difference.saturating_sub(BEFORE);
    let end = characters.len().min(difference.saturating_add(AFTER));
    let mut context = characters[start..end].iter().collect::<String>();
    if start > 0 {
        context.insert(0, '…');
    }
    if end < characters.len() {
        context.push('…');
    }
    context
}

fn bounded_chars(text: &str, maximum: usize) -> String {
    if text.chars().count() <= maximum {
        return text.to_owned();
    }
    if maximum == 0 {
        return String::new();
    }
    let mut bounded = text
        .chars()
        .take(maximum.saturating_sub(1))
        .collect::<String>();
    bounded.push('…');
    bounded
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::media::MediaChunk;
    use crate::speaker::parse_local_transcript;

    fn turn(text: &str, start_ms: u64, end_ms: u64) -> LocalSpeakerTurn {
        LocalSpeakerTurn {
            local_speaker_id: "L1".to_owned(),
            start_ms,
            end_ms,
            text: text.to_owned(),
            clean_reference: true,
        }
    }

    #[test]
    fn validates_bounds_and_forbidden_characters_without_trimming_primary() {
        let value = validate_and_normalize_text("  繁體字。\n", 64).unwrap();
        assert_eq!(value.as_str(), "  繁体字。\n");
        assert!(validate_and_normalize_text("   \n", 64).is_err());
        assert!(validate_and_normalize_text("正文", 0).is_err());
        assert!(validate_and_normalize_text("正文", 5).is_err());
        assert!(validate_and_normalize_text("正\0文", 64).is_err());
        assert!(validate_and_normalize_text("正\u{fffd}文", 64).is_err());
        assert!(validate_and_normalize_text("……", 64).is_err());
    }

    #[test]
    fn punctuation_and_unicode_whitespace_can_reach_exact_consensus() {
        let comparison = compare_primary_and_quality_verifier(
            "你好，世界！这是测试。",
            "你 好　世界 这是测试",
            128,
        )
        .unwrap();
        assert_eq!(comparison.status.as_str(), "exact_consensus");
        assert!(comparison.is_exact_consensus());
        assert!(comparison.difference_summary.is_none());
    }

    #[test]
    fn simplified_and_traditional_outputs_are_not_exact_source_consensus() {
        let comparison =
            compare_primary_and_quality_verifier("我們瞭解這個項目。", "我们了解这个项目", 128)
                .unwrap();
        assert_eq!(comparison.status, AsrComparisonStatus::Disagreement);
        assert!(comparison.difference_summary.is_some());
        assert_eq!(comparison.primary.source_as_str(), "我們瞭解這個項目。");
        assert_eq!(
            comparison.quality_verifier.source_as_str(),
            "我们了解这个项目"
        );
        assert_eq!(comparison.primary.as_str(), "我们了解这个项目。");
        assert_eq!(comparison.quality_verifier.as_str(), "我们了解这个项目");
    }

    #[test]
    fn opencc_many_to_one_projection_cannot_create_false_exact_consensus() {
        let comparison = compare_primary_and_quality_verifier("臺積電", "颱積電", 128).unwrap();
        assert_eq!(comparison.primary.as_str(), "台积电");
        assert_eq!(comparison.quality_verifier.as_str(), "台积电");
        assert_ne!(
            comparison.primary.source_as_str(),
            comparison.quality_verifier.source_as_str()
        );
        assert_eq!(comparison.status, AsrComparisonStatus::Disagreement);
        let summary = comparison.difference_summary.unwrap();
        assert!(summary.contains("臺積電"));
        assert!(summary.contains("颱積電"));
    }

    #[test]
    fn number_words_and_arabic_digits_are_not_equivalent() {
        let comparison = compare_primary_and_quality_verifier("四十二", "40", 64).unwrap();
        assert_eq!(comparison.status, AsrComparisonStatus::Disagreement);
        assert!(comparison.difference_summary.is_some());
    }

    #[test]
    fn alpha_seven_and_alpha_twelve_are_not_equivalent() {
        let comparison =
            compare_primary_and_quality_verifier("阿尔法七", "阿尔法十二", 64).unwrap();
        assert_eq!(comparison.status, AsrComparisonStatus::Disagreement);
    }

    #[test]
    fn missing_negation_is_a_disagreement() {
        let comparison =
            compare_primary_and_quality_verifier("我不同意这个方案。", "我同意这个方案。", 64)
                .unwrap();
        assert_eq!(comparison.status, AsrComparisonStatus::Disagreement);
    }

    #[test]
    fn numeric_format_letters_and_minus_are_preserved() {
        for (primary, verifier) in [
            ("3.14", "314"),
            ("1,000", "1000"),
            ("12:30", "1230"),
            ("API", "Api"),
            ("-40", "40"),
            ("−40", "40"),
        ] {
            let comparison = compare_primary_and_quality_verifier(primary, verifier, 64).unwrap();
            assert_eq!(
                comparison.status,
                AsrComparisonStatus::Disagreement,
                "{primary:?} and {verifier:?} must remain distinct"
            );
        }
    }

    #[test]
    fn leading_fractions_and_colon_runs_cannot_create_false_exact_consensus() {
        for (primary, verifier) in [
            (".5%", "5%"),
            ("2001:db8::1", "2001:db81"),
            ("C::foo", "Cfoo"),
            ("::1", "1"),
            ("fe80::", "fe80"),
        ] {
            let comparison = compare_primary_and_quality_verifier(primary, verifier, 128).unwrap();
            assert_eq!(
                comparison.status,
                AsrComparisonStatus::Disagreement,
                "semantic separator must be preserved: {primary:?} vs {verifier:?}"
            );
            assert_ne!(canonical_content(primary), canonical_content(verifier));
        }

        assert_eq!(canonical_content(".5%"), ".5%");
        assert_eq!(canonical_content("2001:db8::1"), "2001:db8::1");
        assert_eq!(canonical_content("C::foo"), "C::foo");
    }

    #[test]
    fn unicode_word_apostrophes_are_semantic_but_chinese_quotes_are_presentation() {
        for (primary, verifier) in [
            ("d’Ávila", "dÁvila"),
            ("d'Ávila", "dÁvila"),
            ("О’Ніл", "ОНіл"),
        ] {
            let comparison = compare_primary_and_quality_verifier(primary, verifier, 128).unwrap();
            assert_eq!(
                comparison.status,
                AsrComparisonStatus::Disagreement,
                "word-internal apostrophe must be preserved: {primary:?} vs {verifier:?}"
            );
        }

        for quoted in ["他说‘你好’然后走了。", "他说'你好'然后走了。"] {
            let comparison =
                compare_primary_and_quality_verifier(quoted, "他说你好然后走了", 128).unwrap();
            assert_eq!(
                comparison.status,
                AsrComparisonStatus::ExactConsensus,
                "ordinary Chinese quotes must remain presentation punctuation: {quoted:?}"
            );
        }
    }

    #[test]
    fn numeric_separator_at_turn_boundary_remains_semantic() {
        let primary = validate_and_normalize_text("1.2", 64).unwrap();
        let mut turns = vec![turn("1", 0, 1_000), turn(".2", 1_000, 2_000)];
        validate_aligned_turns_against_primary(&turns, &primary, 64).unwrap();
        restore_primary_text_to_aligned_turns(&mut turns, &primary, 64).unwrap();
        assert_eq!(turns[0].text, "1");
        assert_eq!(turns[1].text, ".2");
    }

    #[test]
    fn url_email_and_identifier_separators_are_semantic() {
        for (primary, verifier) in [
            ("john.smith@example.com", "johnsmith@examplecom"),
            ("https://api.openai.com/v1?q=x", "https//apiopenaicom/v1q=x"),
            (
                "https://example.com/a;jsessionid=1",
                "https://example.com/ajsessionid=1",
            ),
            ("release:v1.2", "releasev12"),
            ("API：v1", "APIv1"),
            ("O’Neil", "ONeil"),
        ] {
            let comparison = compare_primary_and_quality_verifier(primary, verifier, 256).unwrap();
            assert_eq!(
                comparison.status,
                AsrComparisonStatus::Disagreement,
                "token punctuation must be preserved: {primary:?} vs {verifier:?}"
            );
        }
        let sentence =
            compare_primary_and_quality_verifier("Hello, world.", "Hello world", 128).unwrap();
        assert_eq!(sentence.status, AsrComparisonStatus::ExactConsensus);
    }

    #[test]
    fn unicode_word_boundaries_are_semantic_but_han_spacing_is_presentation() {
        for (primary, verifier) in [
            ("not able", "notable"),
            ("now here", "nowhere"),
            ("Open AI", "OpenAI"),
            ("1 000", "1000"),
            ("не возможно", "невозможно"),
            ("한 국", "한국"),
            ("غير ممكن", "غيرممكن"),
            ("これ は", "これは"),
        ] {
            assert_eq!(
                compare_primary_and_quality_verifier(primary, verifier, 128)
                    .unwrap()
                    .status,
                AsrComparisonStatus::Disagreement
            );
        }
        assert_eq!(
            canonical_content("Open   AI"),
            canonical_content("Open\nAI")
        );
        assert_eq!(canonical_content("你 好"), canonical_content("你好"));
    }

    #[test]
    fn json_parse_then_restore_preserves_ascii_word_boundary_bytes() {
        let chunk = MediaChunk {
            source_path: "canonical.flac".into(),
            audio_start_ms: 0,
            start_ms: 0,
            end_ms: 2_000,
            lineage: "ascii-boundary".to_owned(),
        };
        let mut transcript = parse_local_transcript(
            r#"{"audio_status":"speech","target_complete":true,"processed_through_ms":2000,"turns":[{"local_speaker_id":"L1","start_ms":0,"end_ms":1000,"text":"hello ","clean_reference":true},{"local_speaker_id":"L2","start_ms":1000,"end_ms":2000,"text":"world","clean_reference":true}]}"#,
            &chunk,
            16,
        )
        .unwrap();
        assert_eq!(transcript.turns[0].text, "hello ");

        let primary = validate_and_normalize_text("hello world", 128).unwrap();
        restore_primary_text_to_aligned_transcript(&mut transcript, &primary, 128).unwrap();
        assert_eq!(
            transcript
                .turns
                .iter()
                .map(|turn| turn.text.as_str())
                .collect::<String>(),
            primary.as_str()
        );
    }

    #[test]
    fn json_parse_then_restore_preserves_mixed_kana_and_source_han_bytes() {
        let chunk = MediaChunk {
            source_path: "canonical.flac".into(),
            audio_start_ms: 0,
            start_ms: 0,
            end_ms: 3_000,
            lineage: "mixed-kana".to_owned(),
        };
        let mut transcript = parse_local_transcript(
            r#"{"audio_status":"speech","target_complete":true,"processed_through_ms":3000,"turns":[{"local_speaker_id":"L1","start_ms":0,"end_ms":1000,"text":"我們確認 ","clean_reference":true},{"local_speaker_id":"L2","start_ms":1000,"end_ms":2000,"text":"これはテストです ","clean_reference":true},{"local_speaker_id":"L1","start_ms":2000,"end_ms":3000,"text":"團隊通過","clean_reference":true}]}"#,
            &chunk,
            16,
        )
        .unwrap();
        assert_eq!(transcript.turns[0].text, "我們確認 ");
        assert_eq!(transcript.turns[1].text, "これはテストです ");
        assert_eq!(transcript.turns[2].text, "團隊通過");

        let primary =
            validate_and_normalize_text("我們確認 これはテストです 團隊通過", 256).unwrap();
        restore_primary_text_to_aligned_transcript(&mut transcript, &primary, 256).unwrap();
        assert_eq!(
            transcript
                .turns
                .iter()
                .map(|turn| turn.text.as_str())
                .collect::<String>(),
            primary.as_str()
        );
    }

    #[test]
    fn kana_aware_normalization_is_independent_of_turn_boundaries() {
        let primary = validate_and_normalize_text("我們確認これはテストです。", 256).unwrap();
        let mut turns = vec![
            turn("我們確認", 0, 1_000),
            turn("これはテストです。", 1_000, 3_000),
        ];
        restore_primary_text_to_aligned_turns(&mut turns, &primary, 256).unwrap();
        assert_eq!(turns[0].text, "我們確認");
        assert_eq!(turns[1].text, "これはテストです。");
        assert_eq!(
            turns
                .iter()
                .map(|turn| turn.text.as_str())
                .collect::<String>(),
            primary.as_str()
        );
    }

    #[test]
    fn disagreement_summary_is_bounded() {
        let primary = format!("{}甲", "相同内容".repeat(100));
        let verifier = format!("{}乙", "相同内容".repeat(100));
        let comparison = compare_primary_and_quality_verifier(&primary, &verifier, 4_096).unwrap();
        let summary = comparison.difference_summary.unwrap();
        assert!(summary.chars().count() <= MAX_DIFFERENCE_SUMMARY_CHARS);
        assert!(summary.contains("canonical disagreement"));
    }

    #[test]
    fn fallback_covers_duration_and_preserves_normalized_primary_exactly() {
        let primary = validate_and_normalize_text("  我們確認。\n", 128).unwrap();
        let transcript = build_primary_fallback_transcript(&primary, 42_000).unwrap();
        assert!(transcript.has_speech);
        assert_eq!(transcript.turns.len(), 1);
        let fallback = &transcript.turns[0];
        assert_eq!(fallback.local_speaker_id, "UNKNOWN");
        assert_eq!(fallback.start_ms, 0);
        assert_eq!(fallback.end_ms, 42_000);
        assert_eq!(fallback.text, "  我们确认。\n");
        assert!(!fallback.clean_reference);
        assert!(transcript.activity_ranges.is_none());
        assert!(build_primary_fallback_transcript(&primary, 0).is_err());
    }

    #[test]
    fn aligned_turns_may_change_punctuation_but_not_authoritative_content() {
        let primary = validate_and_normalize_text("我们不接受四十二个方案。", 128).unwrap();
        let equivalent = vec![
            turn("我們不接受，", 0, 1_000),
            turn("四十二個方案！", 1_000, 2_000),
        ];
        validate_aligned_turns_against_primary(&equivalent, &primary, 128).unwrap();

        let missing_negation = vec![turn("我们接受四十二个方案。", 0, 2_000)];
        assert!(validate_aligned_turns_against_primary(&missing_negation, &primary, 128).is_err());

        let changed_number = vec![turn("我们不接受40个方案。", 0, 2_000)];
        assert!(validate_aligned_turns_against_primary(&changed_number, &primary, 128).is_err());
    }

    #[test]
    fn restores_primary_bytes_while_preserving_alignment_fields() {
        let primary = validate_and_normalize_text("  我们不接受，四十二个方案。\n", 128).unwrap();
        let mut turns = vec![
            LocalSpeakerTurn {
                local_speaker_id: "L2".to_owned(),
                start_ms: 100,
                end_ms: 1_000,
                text: "我们不接受".to_owned(),
                clean_reference: true,
            },
            LocalSpeakerTurn {
                local_speaker_id: "L1".to_owned(),
                start_ms: 1_000,
                end_ms: 2_500,
                text: "四十二个方案！".to_owned(),
                clean_reference: false,
            },
        ];

        restore_primary_text_to_aligned_turns(&mut turns, &primary, 128).unwrap();
        assert_eq!(turns[0].text, "  我们不接受，");
        assert_eq!(turns[1].text, "四十二个方案。\n");
        assert_eq!(
            turns
                .iter()
                .map(|turn| turn.text.as_str())
                .collect::<String>(),
            primary.as_str()
        );
        assert_eq!(turns[0].local_speaker_id, "L2");
        assert_eq!(turns[0].start_ms, 100);
        assert_eq!(turns[0].end_ms, 1_000);
        assert!(turns[0].clean_reference);
        assert_eq!(turns[1].local_speaker_id, "L1");
        assert!(!turns[1].clean_reference);
    }

    #[test]
    fn restore_rejects_empty_canonical_turn_and_semantic_changes_atomically() {
        let primary = validate_and_normalize_text("我们不同意四十二项。", 128).unwrap();

        let mut empty = vec![
            turn("我们不同意四十二项。", 0, 1_000),
            turn("，！", 1_000, 2_000),
        ];
        let before_empty = empty
            .iter()
            .map(|turn| turn.text.clone())
            .collect::<Vec<_>>();
        assert!(restore_primary_text_to_aligned_turns(&mut empty, &primary, 128).is_err());
        assert_eq!(
            empty
                .iter()
                .map(|turn| turn.text.clone())
                .collect::<Vec<_>>(),
            before_empty
        );

        for changed in ["我们同意四十二项。", "我们不同意40项。"] {
            let mut turns = vec![turn(changed, 0, 2_000)];
            let before = turns[0].text.clone();
            assert!(restore_primary_text_to_aligned_turns(&mut turns, &primary, 128).is_err());
            assert_eq!(turns[0].text, before);
        }
    }
}
