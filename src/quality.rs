use crate::speaker::LocalTranscript;

pub const CODE_ACOUSTIC_COVERAGE_WARNING: &str = "acoustic_coverage_warning";
pub const CODE_QUALITY_BOOTSTRAP: &str = "quality_bootstrap";
pub const CODE_INHERITED_ADAPTIVE_SPLIT: &str = "inherited_adaptive_split";
pub const CODE_CJK_INTERNAL_WHITESPACE: &str = "cjk_internal_whitespace";
pub const CODE_MECHANICAL_REPETITION: &str = "mechanical_repetition";
pub const CODE_HIGH_FILLER_DENSITY: &str = "high_filler_density";
pub const CODE_UNCLEAR_MARKER: &str = "unclear_marker";
pub const CODE_REPLACEMENT_CHARACTER: &str = "replacement_character";
pub const CODE_EXCESSIVE_UNFINISHED_TURNS: &str = "excessive_unfinished_turns";

/// Non-text signals inherited from the Stage A processing path.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct QualitySignals {
    pub acoustic_warning: bool,
    pub inherited_split: bool,
}

/// Counts deterministic, meaning-preserving presentation repairs.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct NormalizationStats {
    pub changed_turns: usize,
    pub punctuation_replacements: usize,
    pub spaces_removed: usize,
}

/// Stable quality reason codes, ordered independently of transcript contents.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct QualityGate {
    pub codes: Vec<String>,
}

impl QualityGate {
    pub fn should_escalate(&self) -> bool {
        !self.codes.is_empty()
    }
}

/// Applies only deterministic typography repairs; it never adds or removes words.
pub fn normalize_quality_text(text: &str) -> String {
    normalize_quality_text_with_stats(text).0
}

/// Normalizes every quality-mode turn in place without touching timing or speaker data.
pub fn normalize_quality_transcript(transcript: &mut LocalTranscript) -> NormalizationStats {
    let mut total = NormalizationStats::default();
    for turn in &mut transcript.turns {
        let (normalized, stats) = normalize_quality_text_with_stats(&turn.text);
        if normalized != turn.text {
            turn.text = normalized;
            total.changed_turns = total.changed_turns.saturating_add(1);
        }
        total.punctuation_replacements = total
            .punctuation_replacements
            .saturating_add(stats.punctuation_replacements);
        total.spaces_removed = total.spaces_removed.saturating_add(stats.spaces_removed);
    }
    total
}

/// Applies a conservative disfluency cleanup only after an audio-backed quality review.
/// The rules cover explicit connector/response duplication and stutter-prone pronouns; they do
/// not collapse general Chinese reduplication such as `好好学习` or `非常非常重要`.
pub fn cleanup_reviewed_quality_transcript(transcript: &mut LocalTranscript) -> usize {
    let mut changed_turns = 0_usize;
    for turn in &mut transcript.turns {
        if contains_protected_literal(&turn.text) || !is_cleanup_worthy_turn(&turn.text) {
            continue;
        }
        let cleaned = cleanup_reviewed_quality_text(&turn.text);
        if cleaned != turn.text {
            turn.text = cleaned;
            changed_turns = changed_turns.saturating_add(1);
        }
    }
    changed_turns
}

fn contains_protected_literal(text: &str) -> bool {
    text.contains("://")
        || text.contains("www.")
        || text.contains('@')
        || text.contains('`')
        || text.contains("原样")
        || text.contains("逐字")
        || text.chars().any(|character| {
            matches!(
                character,
                '“' | '”'
                    | '‘'
                    | '’'
                    | '"'
                    | '\''
                    | '《'
                    | '》'
                    | '〈'
                    | '〉'
                    | '「'
                    | '」'
                    | '『'
                    | '』'
                    | '('
                    | ')'
                    | '（'
                    | '）'
                    | '['
                    | ']'
                    | '【'
                    | '】'
                    | '〔'
                    | '〕'
                    | '［'
                    | '］'
                    | '{'
                    | '}'
                    | '｛'
                    | '｝'
                    | '<'
                    | '>'
            )
        })
}

fn cleanup_reviewed_quality_text(text: &str) -> String {
    const PHRASES: [&str; 10] = [
        "不好意思",
        "我懂了",
        "但是",
        "然后",
        "所以",
        "因为",
        "就是",
        "你说",
        "好的",
        "对的",
    ];
    let mut cleaned = text.to_owned();
    for phrase in PHRASES {
        for separator in ["", "，", ",", "、", " "] {
            let repeated = format!("{phrase}{separator}{phrase}");
            while cleaned.contains(&repeated) {
                cleaned = cleaned.replace(&repeated, phrase);
            }
        }
    }

    let characters = cleaned.chars().collect::<Vec<_>>();
    let mut output = String::with_capacity(cleaned.len());
    let mut index = 0_usize;
    while index < characters.len() {
        let character = characters[index];
        let mut end = index + 1;
        while end < characters.len() && characters[end] == character {
            end += 1;
        }
        let run = end - index;
        let collapse = matches!(
            character,
            '我' | '你' | '他' | '她' | '它' | '这' | '那' | '都' | '就'
        ) || matches!(character, '嗯' | '哦' | '呃')
            || (character == '好' && run >= 3);
        if collapse {
            output.push(character);
        } else {
            output.extend(&characters[index..end]);
        }
        index = end;
    }
    normalize_quality_text(output.trim())
}

fn is_cleanup_worthy_turn(text: &str) -> bool {
    if has_mechanical_repetition(text) {
        return true;
    }
    if has_character_run(text, '嗯', 2)
        || has_character_run(text, '哦', 2)
        || has_character_run(text, '呃', 2)
        || has_character_run(text, '好', 3)
    {
        return true;
    }
    const SAFE_PHRASES: [&str; 10] = [
        "不好意思",
        "我懂了",
        "但是",
        "然后",
        "所以",
        "因为",
        "就是",
        "你说",
        "好的",
        "对的",
    ];
    let mut signals = 0_usize;
    for phrase in SAFE_PHRASES {
        for separator in ["", "，", ",", "、", " "] {
            signals = signals.saturating_add(
                text.matches(&format!("{phrase}{separator}{phrase}"))
                    .count(),
            );
        }
    }
    for character in ['我', '你', '他', '她', '它', '这', '那', '都', '就'] {
        signals = signals.saturating_add(text.matches(&format!("{character}{character}")).count());
    }
    signals >= 2
}

fn has_character_run(text: &str, target: char, minimum: usize) -> bool {
    let mut run = 0_usize;
    for character in text.chars() {
        if character == target {
            run += 1;
            if run >= minimum {
                return true;
            }
        } else {
            run = 0;
        }
    }
    false
}

/// Evaluates whether a Lite transcript needs the stronger quality review path.
///
/// Codes always use the order declared here so metadata and tests remain stable.
pub fn evaluate_quality(transcript: &LocalTranscript, signals: QualitySignals) -> QualityGate {
    let mut codes = Vec::new();
    if signals.acoustic_warning {
        codes.push(CODE_ACOUSTIC_COVERAGE_WARNING.to_owned());
    }
    if signals.inherited_split {
        codes.push(CODE_INHERITED_ADAPTIVE_SPLIT.to_owned());
    }
    if has_systemic_cjk_internal_whitespace(transcript) {
        codes.push(CODE_CJK_INTERNAL_WHITESPACE.to_owned());
    }
    if transcript
        .turns
        .iter()
        .any(|turn| has_mechanical_repetition(&turn.text))
        || has_distributed_double_stutters(transcript)
    {
        codes.push(CODE_MECHANICAL_REPETITION.to_owned());
    }
    if has_high_filler_density(transcript) {
        codes.push(CODE_HIGH_FILLER_DENSITY.to_owned());
    }
    if transcript
        .turns
        .iter()
        .any(|turn| has_unclear_marker(&turn.text))
    {
        codes.push(CODE_UNCLEAR_MARKER.to_owned());
    }
    if transcript
        .turns
        .iter()
        .any(|turn| turn.text.contains('\u{fffd}'))
    {
        codes.push(CODE_REPLACEMENT_CHARACTER.to_owned());
    }
    if has_excessive_unfinished_turns(transcript) {
        codes.push(CODE_EXCESSIVE_UNFINISHED_TURNS.to_owned());
    }
    QualityGate { codes }
}

fn normalize_quality_text_with_stats(text: &str) -> (String, NormalizationStats) {
    let source = text.chars().collect::<Vec<_>>();
    let mut converted = Vec::with_capacity(source.len());
    let mut stats = NormalizationStats::default();

    for (index, character) in source.iter().copied().enumerate() {
        let replacement = fullwidth_punctuation(character).filter(|_| {
            has_chinese_punctuation_context(&source, index)
                && !is_preserved_ascii_structure(&source, index, character)
        });
        if let Some(replacement) = replacement {
            converted.push(replacement);
            stats.punctuation_replacements = stats.punctuation_replacements.saturating_add(1);
        } else {
            converted.push(character);
        }
    }

    let mut output = String::with_capacity(text.len());
    let mut index = 0;
    while index < converted.len() {
        if is_horizontal_whitespace(converted[index]) {
            let start = index;
            while index < converted.len() && is_horizontal_whitespace(converted[index]) {
                index += 1;
            }
            let previous = output.chars().next_back();
            let next = converted.get(index).copied();
            if !is_inside_inline_code(&converted, start)
                && (previous.is_some_and(is_chinese_punctuation)
                    || next.is_some_and(is_chinese_punctuation))
            {
                stats.spaces_removed = stats.spaces_removed.saturating_add(index - start);
            } else {
                output.extend(&converted[start..index]);
            }
        } else {
            output.push(converted[index]);
            index += 1;
        }
    }
    (output, stats)
}

fn fullwidth_punctuation(character: char) -> Option<char> {
    match character {
        ',' => Some('，'),
        '?' => Some('？'),
        '!' => Some('！'),
        ';' => Some('；'),
        ':' => Some('：'),
        _ => None,
    }
}

fn has_chinese_punctuation_context(characters: &[char], index: usize) -> bool {
    horizontal_neighbor_before(characters, index).is_some_and(is_compact_chinese)
        || horizontal_neighbor_after(characters, index).is_some_and(is_compact_chinese)
}

fn horizontal_neighbor_before(characters: &[char], index: usize) -> Option<char> {
    characters[..index]
        .iter()
        .rev()
        .copied()
        .take_while(|character| *character != '\n' && *character != '\r')
        .find(|character| !is_horizontal_whitespace(*character))
}

fn horizontal_neighbor_after(characters: &[char], index: usize) -> Option<char> {
    characters[index.saturating_add(1)..]
        .iter()
        .copied()
        .take_while(|character| *character != '\n' && *character != '\r')
        .find(|character| !is_horizontal_whitespace(*character))
}

fn is_preserved_ascii_structure(characters: &[char], index: usize, punctuation: char) -> bool {
    let previous = horizontal_neighbor_before(characters, index);
    let next = horizontal_neighbor_after(characters, index);
    if is_inside_inline_code(characters, index) || token_is_url_or_email(characters, index) {
        return true;
    }
    if matches!((punctuation, previous, next), (',', Some(a), Some(b)) if a.is_ascii_digit() && b.is_ascii_digit())
        || matches!((punctuation, previous, next), (':', Some(a), Some(b)) if a.is_ascii_digit() && b.is_ascii_digit())
    {
        return true;
    }
    if matches!(punctuation, ':' | ';')
        && (previous.is_some_and(|value| value.is_ascii_alphanumeric())
            || next.is_some_and(|value| value.is_ascii_alphanumeric()))
    {
        return true;
    }
    if punctuation == '?' {
        let token_start = characters[..index]
            .iter()
            .rposition(|character| character.is_whitespace())
            .map_or(0, |position| position + 1);
        let left = characters[token_start..index].iter().collect::<String>();
        if left.contains("://") || left.starts_with("www.") {
            return true;
        }
    }
    if punctuation == ';' {
        let token_start = characters[..index]
            .iter()
            .rposition(|character| character.is_whitespace() || is_compact_chinese(*character))
            .map_or(0, |position| position + 1);
        let left = &characters[token_start..index];
        if left.first() == Some(&'&')
            && left[1..]
                .iter()
                .all(|character| character.is_ascii_alphanumeric() || *character == '#')
        {
            return true;
        }
    }
    false
}

fn is_inside_inline_code(characters: &[char], index: usize) -> bool {
    let mut active_delimiter = None::<usize>;
    let mut cursor = 0_usize;
    while cursor < index {
        if characters[cursor] != '`' {
            cursor += 1;
            continue;
        }
        let start = cursor;
        while cursor < index && characters[cursor] == '`' {
            cursor += 1;
        }
        let run = cursor - start;
        if active_delimiter == Some(run) {
            active_delimiter = None;
        } else if active_delimiter.is_none() {
            active_delimiter = Some(run);
        }
    }
    active_delimiter.is_some()
}

fn token_is_url_or_email(characters: &[char], index: usize) -> bool {
    let start = characters[..index]
        .iter()
        .rposition(|character| character.is_whitespace())
        .map_or(0, |position| position + 1);
    let end = characters[index..]
        .iter()
        .position(|character| character.is_whitespace())
        .map_or(characters.len(), |position| index + position);
    let token = characters[start..end].iter().collect::<String>();
    token.contains("://") || token.starts_with("www.") || token.contains('@')
}

fn is_horizontal_whitespace(character: char) -> bool {
    character == ' '
}

fn is_compact_chinese(character: char) -> bool {
    is_han(character) || is_chinese_punctuation(character)
}

fn is_han(character: char) -> bool {
    matches!(
        character as u32,
        0x3400..=0x4dbf
            | 0x4e00..=0x9fff
            | 0xf900..=0xfaff
            | 0x20000..=0x2fa1f
            | 0x30000..=0x323af
    )
}

fn is_chinese_punctuation(character: char) -> bool {
    matches!(
        character,
        '，' | '。'
            | '！'
            | '？'
            | '；'
            | '：'
            | '、'
            | '（'
            | '）'
            | '《'
            | '》'
            | '〈'
            | '〉'
            | '“'
            | '”'
            | '‘'
            | '’'
            | '【'
            | '】'
            | '〔'
            | '〕'
            | '［'
            | '］'
            | '｛'
            | '｝'
            | '—'
            | '…'
            | '·'
    )
}

fn has_systemic_cjk_internal_whitespace(transcript: &LocalTranscript) -> bool {
    transcript
        .turns
        .iter()
        .any(|turn| count_suspicious_cjk_internal_whitespace(&turn.text) >= 2)
}

fn count_suspicious_cjk_internal_whitespace(text: &str) -> usize {
    let characters = text.chars().collect::<Vec<_>>();
    let mut count = 0_usize;
    let mut index = 0_usize;
    while index < characters.len() {
        if characters[index] != ' ' {
            index += 1;
            continue;
        }
        let start = index;
        while index < characters.len() && characters[index] == ' ' {
            index += 1;
        }
        let left_han_run = characters[..start]
            .iter()
            .rev()
            .take_while(|character| is_han(**character))
            .count();
        let right_han_run = characters[index..]
            .iter()
            .take_while(|character| is_han(**character))
            .count();
        if !is_inside_inline_code(&characters, start)
            && left_han_run > 0
            && right_han_run > 0
            && (left_han_run == 1 || right_han_run == 1)
        {
            count = count.saturating_add(1);
        }
    }
    count
}

fn has_mechanical_repetition(text: &str) -> bool {
    let compact = text
        .chars()
        .filter(|character| !character.is_whitespace() && !is_repetition_separator(*character))
        .map(|character| {
            if character.is_ascii_alphabetic() {
                character.to_ascii_lowercase()
            } else {
                character
            }
        })
        .collect::<Vec<_>>();
    if compact.len() < 4 {
        return false;
    }

    let mut run = 1_usize;
    for pair in compact.windows(2) {
        if pair[0] == pair[1] && !pair[0].is_numeric() {
            run += 1;
            let threshold = if is_stutter_prone_function_char(pair[0]) {
                3
            } else {
                4
            };
            if run >= threshold {
                return true;
            }
        } else {
            run = 1;
        }
    }

    for start in 0..compact.len() {
        let maximum_width = 12.min((compact.len() - start) / 3);
        for width in 2..=maximum_width {
            let first = &compact[start..start + width];
            if first.iter().all(|character| character.is_numeric())
                || first.iter().all(|character| *character == first[0])
                || (width <= 2
                    && first
                        .iter()
                        .all(|character| character.is_ascii_alphanumeric()))
            {
                continue;
            }
            if first == &compact[start + width..start + width * 2]
                && first == &compact[start + width * 2..start + width * 3]
            {
                return true;
            }
        }
    }
    false
}

fn has_distributed_double_stutters(transcript: &LocalTranscript) -> bool {
    let mut total = 0_usize;
    for turn in &transcript.turns {
        let count = count_double_stutters(&turn.text);
        if count >= 2 {
            return true;
        }
        total = total.saturating_add(count);
    }
    total >= 3
}

fn count_double_stutters(text: &str) -> usize {
    const SINGLE_CHAR_PATTERNS: [&str; 15] = [
        "我", "你", "他", "她", "它", "这", "那", "是", "的", "了", "都", "就", "可", "会", "要",
    ];
    const PHRASE_PATTERNS: [&str; 14] = [
        "但是",
        "然后",
        "所以",
        "因为",
        "就是",
        "这个",
        "那个",
        "可以",
        "我们",
        "你说",
        "我懂了",
        "好的",
        "对的",
        "不好意思",
    ];
    let mut count = SINGLE_CHAR_PATTERNS
        .iter()
        .map(|pattern| text.matches(&format!("{pattern}{pattern}")).count())
        .sum::<usize>();
    for pattern in PHRASE_PATTERNS {
        for separator in ["", "，", ",", "、", " "] {
            count = count.saturating_add(
                text.matches(&format!("{pattern}{separator}{pattern}"))
                    .count(),
            );
        }
    }
    count
}

fn is_stutter_prone_function_char(character: char) -> bool {
    matches!(
        character,
        '我' | '你' | '他' | '她' | '它' | '这' | '那' | '是' | '的' | '了' | '都' | '就'
    )
}

fn is_repetition_separator(character: char) -> bool {
    matches!(
        character,
        ',' | '，' | '.' | '。' | '!' | '！' | '?' | '？' | ';' | '；' | ':' | '：' | '、' | '…'
    )
}

#[derive(Clone, Copy, Debug, Default)]
struct FillerMetrics {
    count: usize,
    characters: usize,
    visible_characters: usize,
}

impl FillerMetrics {
    fn add(self, other: Self) -> Self {
        Self {
            count: self.count.saturating_add(other.count),
            characters: self.characters.saturating_add(other.characters),
            visible_characters: self
                .visible_characters
                .saturating_add(other.visible_characters),
        }
    }

    fn is_high(self, aggregate: bool) -> bool {
        if self.visible_characters < 8 {
            return false;
        }
        if aggregate {
            (self.count >= 6
                && self.characters.saturating_mul(100) >= self.visible_characters.saturating_mul(8))
                || (self.count >= 12
                    && self.characters.saturating_mul(100)
                        >= self.visible_characters.saturating_mul(4))
        } else {
            self.count >= 3
                && self.characters.saturating_mul(100) >= self.visible_characters.saturating_mul(20)
        }
    }
}

fn has_high_filler_density(transcript: &LocalTranscript) -> bool {
    let mut aggregate = FillerMetrics::default();
    for turn in &transcript.turns {
        let metrics = filler_metrics(&turn.text);
        if metrics.is_high(false) {
            return true;
        }
        aggregate = aggregate.add(metrics);
    }
    aggregate.is_high(true)
}

fn filler_metrics(text: &str) -> FillerMetrics {
    const SOFT_FILLERS: [&str; 4] = ["怎么说呢", "就是说", "那什么", "这么说"];
    let characters = text.chars().collect::<Vec<_>>();
    let mut metrics = FillerMetrics {
        visible_characters: characters
            .iter()
            .filter(|character| character.is_alphanumeric())
            .count(),
        ..FillerMetrics::default()
    };

    for (index, character) in characters.iter().copied().enumerate() {
        let strong = matches!(character, '嗯' | '呃' | '啊' | '呐' | '唔')
            || (character == '额'
                && !characters
                    .get(index.wrapping_sub(1))
                    .is_some_and(|neighbor| is_han(*neighbor))
                && !characters
                    .get(index + 1)
                    .is_some_and(|neighbor| is_han(*neighbor)));
        if strong {
            metrics.count = metrics.count.saturating_add(1);
            metrics.characters = metrics.characters.saturating_add(1);
        }
    }

    let mut byte_index = 0;
    while byte_index < text.len() {
        let remaining = &text[byte_index..];
        if let Some(marker) = SOFT_FILLERS
            .iter()
            .find(|marker| remaining.starts_with(**marker))
        {
            metrics.count = metrics.count.saturating_add(1);
            metrics.characters = metrics.characters.saturating_add(marker.chars().count());
            byte_index += marker.len();
        } else {
            byte_index += remaining.chars().next().map_or(1, char::len_utf8);
        }
    }
    metrics
}

fn has_unclear_marker(text: &str) -> bool {
    const MARKERS: [&str; 8] = [
        "[听不清]",
        "［听不清］",
        "【听不清】",
        "（听不清）",
        "[无法听清]",
        "[无法辨认]",
        "[听不懂]",
        "[inaudible]",
    ];
    let lowercase = text.to_ascii_lowercase();
    MARKERS.iter().any(|marker| lowercase.contains(marker))
}

fn has_excessive_unfinished_turns(transcript: &LocalTranscript) -> bool {
    let eligible = transcript
        .turns
        .iter()
        .filter(|turn| turn.text.chars().any(char::is_alphanumeric))
        .count();
    if eligible < 3 {
        return false;
    }
    let unfinished = transcript
        .turns
        .iter()
        .filter(|turn| is_unfinished(&turn.text))
        .count();
    unfinished >= 3 && unfinished.saturating_mul(100) >= eligible.saturating_mul(30)
}

fn is_unfinished(text: &str) -> bool {
    let trimmed = text.trim_end();
    if trimmed.is_empty() {
        return false;
    }
    if trimmed.chars().next_back().is_some_and(|character| {
        matches!(
            character,
            ',' | '，' | '、' | ':' | '：' | ';' | '；' | '-' | '—' | '…'
        )
    }) {
        return true;
    }
    const DANGLING_ENDINGS: [&str; 12] = [
        "因为", "所以", "但是", "然后", "如果", "的话", "就是", "比如", "包括", "以及", "而且",
        "并且",
    ];
    DANGLING_ENDINGS
        .iter()
        .any(|ending| trimmed.ends_with(ending))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::speaker::LocalSpeakerTurn;

    fn transcript(texts: &[&str]) -> LocalTranscript {
        LocalTranscript {
            has_speech: !texts.is_empty(),
            turns: texts
                .iter()
                .enumerate()
                .map(|(index, text)| LocalSpeakerTurn {
                    local_speaker_id: "L1".to_owned(),
                    start_ms: index as u64 * 1_000,
                    end_ms: index as u64 * 1_000 + 900,
                    text: (*text).to_owned(),
                    clean_reference: false,
                })
                .collect(),
            activity_ranges: None,
        }
    }

    #[test]
    fn normalizes_chinese_punctuation_and_spacing_without_damaging_ascii() {
        let input = "你 好 , 世界 ! 版本 3.7 Flash, API: https://example.com?q=中文 , 金额 1,234.56。Hello, world!";
        assert_eq!(
            normalize_quality_text(input),
            "你 好，世界！版本 3.7 Flash, API: https://example.com?q=中文，金额 1,234.56。Hello, world!"
        );
    }

    #[test]
    fn normalization_is_idempotent_and_preserves_newlines() {
        let normalized = normalize_quality_text("第一行 , 好\n第二行 : 也好");
        assert_eq!(normalized, "第一行，好\n第二行：也好");
        assert_eq!(normalize_quality_text(&normalized), normalized);
    }

    #[test]
    fn normalization_preserves_urls_email_inline_code_and_structural_tabs() {
        for input in [
            "https://x.example/search?q=北京,上海&next=甲:乙",
            "user@example.com",
            "`键:value;中文`",
            "``打印 甲:乙；再输出``",
            "甲\t乙",
            "甲\u{3000}乙",
            "甲\u{00a0}乙",
        ] {
            assert_eq!(normalize_quality_text(input), input);
        }
    }

    #[test]
    fn normalizes_transcript_in_place_and_reports_repairs() {
        let mut value = transcript(&["你 好 , 世界", "English, text"]);
        let stats = normalize_quality_transcript(&mut value);
        assert_eq!(value.turns[0].text, "你 好，世界");
        assert_eq!(value.turns[1].text, "English, text");
        assert_eq!(stats.changed_turns, 1);
        assert_eq!(stats.punctuation_replacements, 1);
        assert_eq!(stats.spaces_removed, 2);
    }

    #[test]
    fn cjk_spaces_are_preserved_and_only_systemic_spacing_triggers_review() {
        let names = transcript(&["参与者：张三 李四 王五 赵六。", "``打印 甲 乙``"]);
        let mut normalized = names.clone();
        normalize_quality_transcript(&mut normalized);
        assert_eq!(normalized.turns[0].text, names.turns[0].text);
        assert_eq!(normalized.turns[1].text, names.turns[1].text);
        assert_eq!(
            evaluate_quality(&normalized, QualitySignals::default()),
            QualityGate::default()
        );

        let spaced = transcript(&["中 药 文 化需要正确识别。"]);
        assert_eq!(
            evaluate_quality(&spaced, QualitySignals::default()).codes,
            vec![CODE_CJK_INTERNAL_WHITESPACE.to_owned()]
        );
    }

    #[test]
    fn catches_xiaotiao_style_mechanical_repetition() {
        assert!(has_mechanical_repetition("这个都都都都可以改。"));
        assert!(has_mechanical_repetition("我懂了，我懂了，我懂了。"));
        assert!(has_mechanical_repetition(
            "Sorry, sorry, SORRY, sorry, Sorry."
        ));
    }

    #[test]
    fn reviewed_cleanup_removes_known_disfluencies_but_preserves_real_reduplication() {
        let mut value = transcript(&[
            "因为但是但是这里有逻辑，你说，你说应该怎么办？",
            "我我我明白了。好的好的。嗯嗯。好好好。",
            "请好好学习，这一点非常非常重要，人人都要认真。",
        ]);
        assert_eq!(cleanup_reviewed_quality_transcript(&mut value), 2);
        assert_eq!(value.turns[0].text, "因为但是这里有逻辑，你说应该怎么办？");
        assert_eq!(value.turns[1].text, "我明白了。好的。嗯。好。");
        assert_eq!(
            value.turns[2].text,
            "请好好学习，这一点非常非常重要，人人都要认真。"
        );

        let mut acknowledgements = transcript(&["嗯嗯，我知道了。", "好好好，我们继续。"]);
        assert_eq!(
            cleanup_reviewed_quality_transcript(&mut acknowledgements),
            2
        );
        assert_eq!(acknowledgements.turns[0].text, "嗯，我知道了。");
        assert_eq!(acknowledgements.turns[1].text, "好，我们继续。");
    }

    #[test]
    fn reviewed_cleanup_never_rewrites_literal_or_address_spans() {
        let texts = [
            "`我我我` 是测试代码。",
            "访问 https://x.example/我我我",
            "访问 www.example.com/我我我",
            "标题叫《我们我们》。",
            "请把“因为，因为”原样写两遍。",
            "请原样保留我我我三个字。",
            "型号【我我我】，代码（好好好）。",
            "邮箱是 我我我@example.com。",
        ];
        let mut value = transcript(&texts);
        assert_eq!(cleanup_reviewed_quality_transcript(&mut value), 0);
        for (turn, expected) in value.turns.iter().zip(texts) {
            assert_eq!(turn.text, expected);
        }

        let mut mixed = transcript(&["我我我刚才说错了。", "公司名称是我们我们科技。"]);
        assert_eq!(cleanup_reviewed_quality_transcript(&mut mixed), 1);
        assert_eq!(mixed.turns[0].text, "我刚才说错了。");
        assert_eq!(mixed.turns[1].text, "公司名称是我们我们科技。");
    }

    #[test]
    fn repetition_gate_avoids_numbers_and_two_real_mentions() {
        assert!(!has_mechanical_repetition(
            "编号是 20202020，金额为 1111 元。"
        ));
        assert!(!has_mechanical_repetition(
            "全角编号 １１１１，代码 ABABAB。"
        ));
        assert!(!has_mechanical_repetition("研究研究型学习方法。"));
        assert!(has_mechanical_repetition("我我我刚才说错了。"));
    }

    #[test]
    fn catches_dense_fillers_but_not_normal_connectives() {
        let noisy = transcript(&["嗯，呃，啊，唔，怎么说呢，那什么，我觉得是这样。"]);
        assert!(has_high_filler_density(&noisy));

        let normal = transcript(&[
            "我们先核对实验数据，然后检查图表。",
            "结果确认后，就是按原计划提交。",
            "这就是第一点，这就是第二点，这就是第三点。",
        ]);
        assert!(!has_high_filler_density(&normal));
    }

    #[test]
    fn catches_distributed_double_stutters_without_rejecting_one_emphasis() {
        let noisy = transcript(&[
            "因为但是但是这里有逻辑。",
            "你说你说应该怎么办？",
            "这个可以可以继续调整。",
        ]);
        assert!(has_distributed_double_stutters(&noisy));

        let emphasis = transcript(&["这个方案非常非常重要。", "谢谢老师。"]);
        assert!(!has_distributed_double_stutters(&emphasis));

        let natural_reduplication = transcript(&[
            "谢谢老师，我们慢慢调整。",
            "大家想想办法，逐步推进。",
            "我们提交，交给老师；结论保留，留给后续。",
        ]);
        assert!(!has_distributed_double_stutters(&natural_reduplication));
    }

    #[test]
    fn gate_emits_stable_codes_for_text_and_pipeline_signals() {
        let value = transcript(&[
            "嗯，呃，那个，就是，就是说，然后然后然后。[听不清]�，",
            "我懂了，我懂了，我懂了，",
            "这个部分因为，",
        ]);
        let gate = evaluate_quality(
            &value,
            QualitySignals {
                acoustic_warning: true,
                inherited_split: true,
            },
        );
        assert_eq!(
            gate.codes,
            vec![
                CODE_ACOUSTIC_COVERAGE_WARNING.to_owned(),
                CODE_INHERITED_ADAPTIVE_SPLIT.to_owned(),
                CODE_MECHANICAL_REPETITION.to_owned(),
                CODE_HIGH_FILLER_DENSITY.to_owned(),
                CODE_UNCLEAR_MARKER.to_owned(),
                CODE_REPLACEMENT_CHARACTER.to_owned(),
                CODE_EXCESSIVE_UNFINISHED_TURNS.to_owned(),
            ]
        );
        assert!(gate.should_escalate());
    }

    #[test]
    fn clean_mixed_language_transcript_does_not_escalate() {
        let value = transcript(&[
            "我们在 2026 年完成第一阶段，然后开始第二阶段。",
            "API: https://example.com?q=中文，模型为 Gemini 3.7 Flash。",
            "金额为 1,234.56 元，结论已经复核。",
        ]);
        assert_eq!(
            evaluate_quality(&value, QualitySignals::default()),
            QualityGate::default()
        );
    }
}
