use std::error::Error;
use std::fmt;

pub const CODE_CHINESE_PUNCTUATION_NORMALIZED: &str =
    "quality_cleanup_chinese_punctuation_normalized";
pub const CODE_POSSIBLE_DISFLUENCY_PRESERVED: &str =
    "quality_cleanup_signal_possible_disfluency_preserved";
pub const CODE_CLEANUP_REVERTED: &str = "quality_cleanup_reverted";

/// Counts deterministic presentation edits. No lexical character may be
/// inserted, substituted or removed by this module.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CleanupStats {
    pub punctuation_replacements: usize,
    pub punctuation_spaces_removed: usize,
}

impl CleanupStats {
    pub fn operation_count(self) -> usize {
        self.punctuation_replacements
            .saturating_add(self.punctuation_spaces_removed)
    }

    pub fn is_empty(self) -> bool {
        self == Self::default()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CleanupResult {
    pub text: String,
    pub stats: CleanupStats,
    pub codes: Vec<&'static str>,
    pub rejection: Option<CleanupRejection>,
}

impl CleanupResult {
    pub fn changed(&self) -> bool {
        !self.stats.is_empty() && self.rejection.is_none()
    }

    pub fn reverted(&self) -> bool {
        self.rejection.is_some()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CleanupRejection {
    UnauthorizedCharacterChange,
    ProtectedTextChanged,
    PathologicalExpansion,
}

impl fmt::Display for CleanupRejection {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::UnauthorizedCharacterChange => "清稿包含未授权的字符变化",
            Self::ProtectedTextChanged => "清稿试图修改受保护文本",
            Self::PathologicalExpansion => "清稿结果发生非预期膨胀",
        })
    }
}

impl Error for CleanupRejection {}

#[derive(Clone, Copy, Debug)]
struct TrackedCharacter {
    original: char,
    current: char,
    protected: bool,
    remove_space: bool,
    replace_punctuation: bool,
}

/// Applies presentation-only cleanup. Spoken words, fillers, repetitions,
/// names, numbers, negation and conditions are immutable. Potential
/// disfluencies are reported and intentionally preserved until an
/// audio/timestamp-backed deletion contract exists.
pub fn cleanup_quality_text(text: &str) -> CleanupResult {
    match try_cleanup_quality_text(text) {
        Ok(result) => result,
        Err(rejection) => CleanupResult {
            text: text.to_owned(),
            stats: CleanupStats::default(),
            codes: vec![CODE_CLEANUP_REVERTED],
            rejection: Some(rejection),
        },
    }
}

pub fn try_cleanup_quality_text(text: &str) -> Result<CleanupResult, CleanupRejection> {
    let source = text.chars().collect::<Vec<_>>();
    let protected = protected_mask(&source);
    let mut tracked = source
        .iter()
        .copied()
        .zip(protected)
        .map(|(character, protected)| TrackedCharacter {
            original: character,
            current: character,
            protected,
            remove_space: false,
            replace_punctuation: false,
        })
        .collect::<Vec<_>>();
    let mut stats = CleanupStats::default();

    for index in 0..tracked.len() {
        if tracked[index].protected {
            continue;
        }
        let Some(replacement) = fullwidth_punctuation(tracked[index].original) else {
            continue;
        };
        if has_chinese_context(&tracked, index)
            && !has_structural_space_run_around(&tracked, index)
            && !is_preserved_ascii_structure(&tracked, index, tracked[index].original)
        {
            tracked[index].current = replacement;
            tracked[index].replace_punctuation = true;
            stats.punctuation_replacements = stats.punctuation_replacements.saturating_add(1);
        }
    }

    let mut index = 0_usize;
    while index < tracked.len() {
        if tracked[index].original != ' ' || tracked[index].protected {
            index += 1;
            continue;
        }
        let start = index;
        while index < tracked.len() && tracked[index].original == ' ' && !tracked[index].protected {
            index += 1;
        }
        if index - start != 1 {
            continue;
        }
        let left = active_before(&tracked, start);
        let right = active_after(&tracked, index);
        if left
            .map(|position| is_chinese_punctuation(tracked[position].current))
            .unwrap_or(false)
            || right
                .map(|position| is_chinese_punctuation(tracked[position].current))
                .unwrap_or(false)
        {
            tracked[start].remove_space = true;
            stats.punctuation_spaces_removed = stats.punctuation_spaces_removed.saturating_add(1);
        }
    }

    let output = tracked
        .iter()
        .filter(|item| !item.remove_space)
        .map(|item| item.current)
        .collect::<String>();
    audit_cleanup(text, &tracked, &output)?;

    let mut codes = Vec::with_capacity(2);
    if !stats.is_empty() {
        codes.push(CODE_CHINESE_PUNCTUATION_NORMALIZED);
    }
    if has_possible_disfluency(text) {
        codes.push(CODE_POSSIBLE_DISFLUENCY_PRESERVED);
    }
    Ok(CleanupResult {
        text: output,
        stats,
        codes,
        rejection: None,
    })
}

fn has_structural_space_run_around(tracked: &[TrackedCharacter], index: usize) -> bool {
    if index
        .checked_sub(1)
        .and_then(|position| tracked.get(position))
        .is_some_and(|item| item.original.is_whitespace() && item.original != ' ')
        || tracked
            .get(index + 1)
            .is_some_and(|item| item.original.is_whitespace() && item.original != ' ')
    {
        return true;
    }
    let left_spaces = tracked[..index]
        .iter()
        .rev()
        .take_while(|item| item.original == ' ')
        .count();
    let right_spaces = tracked[index + 1..]
        .iter()
        .take_while(|item| item.original == ' ')
        .count();
    left_spaces > 1 || right_spaces > 1
}

fn audit_cleanup(
    input: &str,
    tracked: &[TrackedCharacter],
    output: &str,
) -> Result<(), CleanupRejection> {
    if output.chars().count() > input.chars().count()
        || output.len() > input.len().saturating_add(tracked.len().saturating_mul(2))
    {
        return Err(CleanupRejection::PathologicalExpansion);
    }
    for item in tracked {
        if item.protected
            && (item.current != item.original || item.remove_space || item.replace_punctuation)
        {
            return Err(CleanupRejection::ProtectedTextChanged);
        }
        if item.remove_space && item.original != ' ' {
            return Err(CleanupRejection::UnauthorizedCharacterChange);
        }
        if item.replace_punctuation && fullwidth_punctuation(item.original) != Some(item.current) {
            return Err(CleanupRejection::UnauthorizedCharacterChange);
        }
        if !item.replace_punctuation && item.current != item.original {
            return Err(CleanupRejection::UnauthorizedCharacterChange);
        }
    }
    if lexical_projection(input) != lexical_projection(output) {
        return Err(CleanupRejection::UnauthorizedCharacterChange);
    }
    Ok(())
}

fn lexical_projection(text: &str) -> String {
    text.chars()
        .filter_map(|character| {
            if character == ' ' {
                None
            } else {
                Some(match character {
                    '，' => ',',
                    '？' => '?',
                    '！' => '!',
                    '；' => ';',
                    '：' => ':',
                    other => other,
                })
            }
        })
        .collect()
}

fn protected_mask(characters: &[char]) -> Vec<bool> {
    let mut protected = vec![false; characters.len()];
    protect_inline_code(characters, &mut protected);
    protect_paired_literals(characters, &mut protected);
    protect_address_tokens(characters, &mut protected);
    protected
}

fn protect_inline_code(characters: &[char], protected: &mut [bool]) {
    let mut index = 0_usize;
    while index < characters.len() {
        if characters[index] != '`' {
            index += 1;
            continue;
        }
        let start = index;
        while index < characters.len() && characters[index] == '`' {
            index += 1;
        }
        let delimiter_len = index - start;
        let mut cursor = index;
        let mut end = characters.len();
        while cursor < characters.len() {
            if characters[cursor] != '`' {
                cursor += 1;
                continue;
            }
            let run_start = cursor;
            while cursor < characters.len() && characters[cursor] == '`' {
                cursor += 1;
            }
            if cursor - run_start == delimiter_len {
                end = cursor;
                break;
            }
        }
        mark_protected(protected, start, end);
        index = end;
    }
}

fn protect_paired_literals(characters: &[char], protected: &mut [bool]) {
    for (open, close) in [
        ('"', '"'),
        ('“', '”'),
        ('‘', '’'),
        ('《', '》'),
        ('〈', '〉'),
        ('「', '」'),
        ('『', '』'),
        ('(', ')'),
        ('（', '）'),
        ('[', ']'),
        ('【', '】'),
        ('{', '}'),
        ('｛', '｝'),
    ] {
        let mut index = 0_usize;
        while index < characters.len() {
            let Some(relative_start) = characters[index..].iter().position(|value| *value == open)
            else {
                break;
            };
            let start = index + relative_start;
            let end = characters[start + 1..]
                .iter()
                .position(|value| *value == close)
                .map_or(characters.len(), |relative| start + 2 + relative);
            mark_protected(protected, start, end);
            index = end;
        }
    }
}

fn protect_address_tokens(characters: &[char], protected: &mut [bool]) {
    let mut index = 0_usize;
    while index < characters.len() {
        if characters[index].is_whitespace() {
            index += 1;
            continue;
        }
        let start = index;
        while index < characters.len() && !characters[index].is_whitespace() {
            index += 1;
        }
        let token = characters[start..index].iter().collect::<String>();
        if token.contains("://")
            || token.to_ascii_lowercase().starts_with("www.")
            || token.contains('@')
        {
            mark_protected(protected, start, index);
        }
    }
}

fn mark_protected(mask: &mut [bool], start: usize, end: usize) {
    for value in mask.get_mut(start..end).unwrap_or_default() {
        *value = true;
    }
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

fn has_chinese_context(tracked: &[TrackedCharacter], index: usize) -> bool {
    active_before(tracked, index)
        .map(|position| is_compact_chinese(tracked[position].current))
        .unwrap_or(false)
        || active_after(tracked, index + 1)
            .map(|position| is_compact_chinese(tracked[position].current))
            .unwrap_or(false)
}

fn is_preserved_ascii_structure(
    tracked: &[TrackedCharacter],
    index: usize,
    punctuation: char,
) -> bool {
    let immediate_left = index
        .checked_sub(1)
        .and_then(|position| tracked.get(position));
    let immediate_right = tracked.get(index + 1);
    let adjacent_ascii = immediate_left
        .zip(immediate_right)
        .is_some_and(|(left, right)| {
            left.original.is_ascii_alphanumeric() && right.original.is_ascii_alphanumeric()
        });
    let adjacent_numbers = immediate_left
        .zip(immediate_right)
        .is_some_and(|(left, right)| left.original.is_numeric() && right.original.is_numeric());
    adjacent_ascii || (matches!(punctuation, ',' | ':' | ';') && adjacent_numbers)
}

fn active_before(tracked: &[TrackedCharacter], index: usize) -> Option<usize> {
    (0..index)
        .rev()
        .find(|position| tracked[*position].original != ' ')
}

fn active_after(tracked: &[TrackedCharacter], index: usize) -> Option<usize> {
    (index..tracked.len()).find(|position| tracked[*position].original != ' ')
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
        '，' | '。' | '！' | '？' | '；' | '：' | '、' | '”' | '’' | '》' | '）' | '】'
    )
}

fn has_possible_disfluency(text: &str) -> bool {
    let compact = text
        .chars()
        .filter(|character| !character.is_whitespace())
        .collect::<String>();
    compact.contains('嗯')
        || compact.contains('呃')
        || [
            "我我",
            "你你",
            "然后然后",
            "但是但是",
            "所以所以",
            "Sorry,Sorry",
            "sorry,sorry",
        ]
        .iter()
        .any(|pattern| compact.contains(pattern))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_chinese_punctuation_without_touching_lexical_text() {
        let result = cleanup_quality_text("嗯,我我确认 ! 预算是42.5万元;如果没批准就不提交。");
        assert_eq!(
            result.text,
            "嗯，我我确认！预算是42.5万元；如果没批准就不提交。"
        );
        assert!(result.changed());
        assert!(result.codes.contains(&CODE_CHINESE_PUNCTUATION_NORMALIZED));
        assert!(result.codes.contains(&CODE_POSSIBLE_DISFLUENCY_PRESERVED));
    }

    #[test]
    fn preserves_every_spoken_word_including_possible_fillers_and_names() {
        for input in [
            "项目代号是：然后然后",
            "姓名为：你你",
            "额，是财务术语。",
            "嗯，是汉语中的应答词。",
            "我我我文化传媒有限公司",
            "如果没批准就不提交",
        ] {
            assert_eq!(cleanup_quality_text(input).text, input);
        }
    }

    #[test]
    fn protects_urls_emails_code_and_quoted_spans() {
        for input in [
            "https://x.example/a?q=甲,乙",
            "user@example.com",
            "`键:value;中文,原样`",
            "他说“API:v1,不要改”。",
        ] {
            assert_eq!(cleanup_quality_text(input).text, input);
        }
    }

    #[test]
    fn preserves_ascii_and_numeric_separators() {
        let result = cleanup_quality_text("版本API:v1, 数值1,000与12:30, 中文,结束");
        assert_eq!(result.text, "版本API:v1，数值1,000与12:30，中文，结束");
    }

    #[test]
    fn structural_whitespace_is_not_collapsed() {
        for input in ["甲  ,  乙", "甲\t,乙", "甲\n,乙", "甲\u{3000},乙"] {
            assert_eq!(cleanup_quality_text(input).text, input);
        }
    }

    #[test]
    fn cleanup_is_idempotent() {
        let first = cleanup_quality_text("甲 , 乙 !");
        let second = cleanup_quality_text(&first.text);
        assert_eq!(second.text, first.text);
        assert!(!second.changed());
    }

    #[test]
    fn audit_rejects_a_lexical_substitution() {
        let input = "预算是42万元";
        let mut tracked = input
            .chars()
            .map(|character| TrackedCharacter {
                original: character,
                current: character,
                protected: false,
                remove_space: false,
                replace_punctuation: false,
            })
            .collect::<Vec<_>>();
        tracked[0].current = '费';
        let output = tracked.iter().map(|item| item.current).collect::<String>();
        assert_eq!(
            audit_cleanup(input, &tracked, &output),
            Err(CleanupRejection::UnauthorizedCharacterChange)
        );
    }
}
