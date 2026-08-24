use std::ops::Range;
use std::sync::{Mutex, OnceLock};

use anyhow::{Result, anyhow, bail};
use opencc_fmmseg::OpenCC;

const MAX_PROTECTED_SPANS: usize = 4_096;

const FACT_LABELS: &[&str] = &[
    "项目代号",
    "项目代號",
    "項目代号",
    "項目代號",
    "代号",
    "代號",
    "名称",
    "名稱",
    "姓名",
    "名字",
    "公司名",
    "品牌",
    "机构",
    "機構",
    "学校",
    "學校",
    "型号",
    "型號",
    "编号",
    "編號",
    "账号",
    "帐号",
    "帳号",
    "账號",
    "帳號",
];

const QUOTE_PAIRS: &[(char, char)] = &[
    ('“', '”'),
    ('‘', '’'),
    ('「', '」'),
    ('『', '』'),
    ('《', '》'),
    ('〈', '〉'),
    ('"', '"'),
];

static T2S_CONVERTER: OnceLock<std::result::Result<Mutex<OpenCC>, String>> = OnceLock::new();

/// Converts ordinary Chinese prose to Simplified Chinese while keeping
/// bounded, syntactically identifiable fact spans byte-for-byte unchanged.
/// Japanese text containing kana is preserved as a whole because Han glyphs
/// alone do not identify the spoken language.
pub fn normalize_to_simplified(text: &str) -> Result<String> {
    if text.is_empty() || !text.chars().any(is_han) || text.chars().any(is_kana) {
        return Ok(text.to_owned());
    }

    let protected = collect_protected_spans(text)?;
    let converter = t2s_converter()?;
    let converter = converter
        .lock()
        .map_err(|_| anyhow!("内置 OpenCC t2s 转换器锁已损坏"))?;
    let mut normalized = String::with_capacity(text.len());
    let mut cursor = 0;

    for span in protected {
        append_converted(&converter, &text[cursor..span.start], &mut normalized)?;
        normalized.push_str(&text[span.clone()]);
        cursor = span.end;
    }
    append_converted(&converter, &text[cursor..], &mut normalized)?;

    if normalized.is_empty() && !text.is_empty() {
        bail!("OpenCC t2s 返回了异常空文本");
    }
    Ok(normalized)
}

pub fn ensure_simplified_converter() -> Result<()> {
    t2s_converter().map(|_| ())
}

fn t2s_converter() -> Result<&'static Mutex<OpenCC>> {
    let converter = T2S_CONVERTER.get_or_init(|| {
        OpenCC::clear_last_error();
        let mut converter = OpenCC::new();
        converter.set_parallel(false);
        match OpenCC::get_last_error() {
            Some(error) => Err(error),
            None => Ok(Mutex::new(converter)),
        }
    });
    converter
        .as_ref()
        .map_err(|error| anyhow!("无法初始化内置 OpenCC t2s 词典：{error}"))
}

fn append_converted(converter: &OpenCC, text: &str, output: &mut String) -> Result<()> {
    if text.is_empty() {
        return Ok(());
    }
    if !text.chars().any(is_han) {
        output.push_str(text);
        return Ok(());
    }

    OpenCC::clear_last_error();
    let converted = converter.t2s(text, false);
    if let Some(error) = OpenCC::get_last_error() {
        bail!("OpenCC t2s 转换失败：{error}");
    }
    if converted.is_empty() {
        bail!("OpenCC t2s 返回了异常空片段");
    }
    output.push_str(&converted);
    Ok(())
}

fn collect_protected_spans(text: &str) -> Result<Vec<Range<usize>>> {
    let mut spans = Vec::new();
    protect_inline_code(text, &mut spans)?;
    protect_quoted_text(text, &mut spans)?;
    protect_urls(text, &mut spans)?;
    protect_emails(text, &mut spans)?;
    protect_fact_values(text, &mut spans)?;
    protect_explicit_character_designations(text, &mut spans)?;
    merge_protected_spans(text, spans)
}

fn protect_inline_code(text: &str, spans: &mut Vec<Range<usize>>) -> Result<()> {
    let bytes = text.as_bytes();
    let mut cursor = 0;
    while cursor < bytes.len() {
        if bytes[cursor] != b'`' {
            cursor += 1;
            continue;
        }

        let opening_start = cursor;
        while cursor < bytes.len() && bytes[cursor] == b'`' {
            cursor += 1;
        }
        let delimiter_len = cursor - opening_start;
        let mut search = cursor;
        let mut closing_end = None;
        while search < bytes.len() {
            if bytes[search] != b'`' {
                search += 1;
                continue;
            }
            let run_start = search;
            while search < bytes.len() && bytes[search] == b'`' {
                search += 1;
            }
            if search - run_start == delimiter_len {
                closing_end = Some(search);
                break;
            }
        }

        let end = closing_end.unwrap_or(bytes.len());
        push_protected_span(spans, opening_start..end)?;
        cursor = end;
    }
    Ok(())
}

fn protect_quoted_text(text: &str, spans: &mut Vec<Range<usize>>) -> Result<()> {
    for &(opening, closing) in QUOTE_PAIRS {
        let mut cursor = 0;
        while cursor < text.len() {
            let Some(opening_offset) = text[cursor..].find(opening) else {
                break;
            };
            let content_start = cursor + opening_offset + opening.len_utf8();
            let Some(closing_offset) = text[content_start..].find(closing) else {
                break;
            };
            let content_end = content_start + closing_offset;
            push_protected_span(spans, content_start..content_end)?;
            cursor = content_end + closing.len_utf8();
        }
    }
    Ok(())
}

fn protect_urls(text: &str, spans: &mut Vec<Range<usize>>) -> Result<()> {
    for prefix in ["https://", "http://", "www."] {
        for (start, _) in text.match_indices(prefix) {
            let mut end = start + prefix.len();
            let content_start = end;
            for (offset, character) in text[content_start..].char_indices() {
                if is_url_terminator(character) {
                    break;
                }
                end = content_start + offset + character.len_utf8();
            }
            if end > start + prefix.len() {
                push_protected_span(spans, start..end)?;
            }
        }
    }
    Ok(())
}

fn protect_emails(text: &str, spans: &mut Vec<Range<usize>>) -> Result<()> {
    let bytes = text.as_bytes();
    for (at, _) in text.match_indices('@') {
        let mut start = at;
        while start > 0 && is_email_local_byte(bytes[start - 1]) {
            start -= 1;
        }
        let mut end = at + 1;
        while end < bytes.len() && is_email_domain_byte(bytes[end]) {
            end += 1;
        }
        if start < at && end > at + 1 && bytes[at + 1..end].iter().any(u8::is_ascii_alphanumeric) {
            push_protected_span(spans, start..end)?;
        }
    }
    Ok(())
}

fn protect_fact_values(text: &str, spans: &mut Vec<Range<usize>>) -> Result<()> {
    for label in FACT_LABELS {
        for (label_start, _) in text.match_indices(label) {
            let after_label = label_start + label.len();
            if let Some(value_span) = fact_value_span(text, after_label) {
                push_protected_span(spans, value_span)?;
            }
        }
    }
    Ok(())
}

fn fact_value_span(text: &str, after_label: usize) -> Option<Range<usize>> {
    let mut value_start = skip_whitespace(text, after_label);
    for connector in [
        "称为", "称為", "稱为", "稱為", "：", ":", "=", "是", "为", "為", "叫",
    ] {
        if text[value_start..].starts_with(connector) {
            value_start += connector.len();
            value_start = skip_whitespace(text, value_start);
            break;
        }
    }

    for (offset, character) in text[value_start..].char_indices() {
        if is_fact_value_terminator(character) {
            let value_end = value_start + offset;
            return (value_start < value_end).then_some(value_start..value_end);
        }
    }
    (value_start < text.len()).then_some(value_start..text.len())
}

fn skip_whitespace(text: &str, cursor: usize) -> usize {
    for (offset, character) in text[cursor..].char_indices() {
        if !character.is_whitespace() {
            return cursor + offset;
        }
    }
    text.len()
}

fn protect_explicit_character_designations(
    text: &str,
    spans: &mut Vec<Range<usize>>,
) -> Result<()> {
    let characters = text.char_indices().collect::<Vec<_>>();
    for index in 0..characters.len() {
        if characters[index].1 != '的' || index + 1 >= characters.len() {
            continue;
        }
        let target_index = index + 1;
        let target = characters[target_index].1;
        if !is_han(target)
            || characters
                .get(target_index + 1)
                .is_some_and(|(_, character)| !is_designation_boundary(*character))
        {
            continue;
        }

        let mut clause_start = index;
        while clause_start > 0 && !is_designation_boundary(characters[clause_start - 1].1) {
            clause_start -= 1;
        }
        if !characters[clause_start..index]
            .iter()
            .any(|(_, character)| *character == target)
        {
            continue;
        }

        for &(start, character) in &characters[clause_start..index] {
            if character == target {
                push_protected_span(spans, start..start + character.len_utf8())?;
            }
        }
        let target_start = characters[target_index].0;
        push_protected_span(spans, target_start..target_start + target.len_utf8())?;
    }
    Ok(())
}

fn push_protected_span(spans: &mut Vec<Range<usize>>, span: Range<usize>) -> Result<()> {
    if span.start >= span.end {
        return Ok(());
    }
    if spans.len() >= MAX_PROTECTED_SPANS {
        bail!("受保护事实片段超过上限 {MAX_PROTECTED_SPANS}");
    }
    spans.push(span);
    Ok(())
}

fn merge_protected_spans(text: &str, mut spans: Vec<Range<usize>>) -> Result<Vec<Range<usize>>> {
    spans.sort_unstable_by_key(|span| (span.start, span.end));
    let mut merged: Vec<Range<usize>> = Vec::with_capacity(spans.len());
    for span in spans {
        if span.end > text.len()
            || !text.is_char_boundary(span.start)
            || !text.is_char_boundary(span.end)
        {
            bail!("受保护事实片段边界无效");
        }
        if let Some(previous) = merged.last_mut()
            && span.start <= previous.end
        {
            previous.end = previous.end.max(span.end);
            continue;
        }
        merged.push(span);
    }
    if merged.len() > MAX_PROTECTED_SPANS {
        bail!("合并后的受保护事实片段超过上限 {MAX_PROTECTED_SPANS}");
    }
    Ok(merged)
}

fn is_url_terminator(character: char) -> bool {
    character.is_whitespace()
        || matches!(
            character,
            ',' | '，'
                | '。'
                | ';'
                | '；'
                | '！'
                | '？'
                | '、'
                | '<'
                | '>'
                | '“'
                | '”'
                | '‘'
                | '’'
                | '「'
                | '」'
                | '『'
                | '』'
                | '《'
                | '》'
                | '〈'
                | '〉'
                | '"'
                | '\''
                | '`'
                | ')'
                | '）'
                | ']'
                | '】'
        )
}

fn is_email_local_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric()
        || matches!(
            byte,
            b'.' | b'!'
                | b'#'
                | b'$'
                | b'%'
                | b'&'
                | b'\''
                | b'*'
                | b'+'
                | b'-'
                | b'/'
                | b'='
                | b'?'
                | b'^'
                | b'_'
                | b'`'
                | b'{'
                | b'|'
                | b'}'
                | b'~'
        )
}

fn is_email_domain_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'-' | b'_')
}

fn is_fact_value_terminator(character: char) -> bool {
    matches!(
        character,
        ',' | '，' | '.' | '。' | ';' | '；' | '!' | '！' | '?' | '？' | '、' | '\n' | '\r' | '\t'
    )
}

fn is_designation_boundary(character: char) -> bool {
    character.is_whitespace()
        || is_fact_value_terminator(character)
        || matches!(
            character,
            ':' | '：'
                | '“'
                | '”'
                | '‘'
                | '’'
                | '「'
                | '」'
                | '『'
                | '』'
                | '《'
                | '》'
                | '〈'
                | '〉'
                | '('
                | ')'
                | '（'
                | '）'
                | '['
                | ']'
                | '【'
                | '】'
        )
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

fn is_kana(character: char) -> bool {
    let scalar = character as u32;
    scalar != 0x30FB
        && matches!(
            scalar,
            0x3040..=0x30FF
                | 0x31F0..=0x31FF
                | 0xFF65..=0xFF9F
                | 0x1AFF0..=0x1AFFF
                | 0x1B000..=0x1B0FF
                | 0x1B100..=0x1B12F
                | 0x1B130..=0x1B16F
        )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ordinary_traditional_prose_is_fully_normalized() {
        ensure_simplified_converter().unwrap();
        assert_eq!(
            normalize_to_simplified("我們瞭解這個項目，並準備後續資料。").unwrap(),
            "我们了解这个项目，并准备后续资料。"
        );
    }

    #[test]
    fn fact_label_and_explicit_character_designation_preserve_source_glyphs() {
        assert_eq!(
            normalize_to_simplified("項目代號是乾紅，乾坤的乾").unwrap(),
            "项目代号是乾紅，乾坤的乾"
        );
    }

    #[test]
    fn every_supported_fact_label_protects_its_value_only() {
        let source = "名稱為臺積電；姓名是趙乾；公司名是髮藝；品牌是乾紅；機構是臺研院；學校是颱風學校；型號是髮-42；編號是臺-7；帳號是乾User。後續瞭解。";
        let expected = "名称为臺積電；姓名是趙乾；公司名是髮藝；品牌是乾紅；机构是臺研院；学校是颱風學校；型号是髮-42；编号是臺-7；帐号是乾User。后续了解。";
        assert_eq!(normalize_to_simplified(source).unwrap(), expected);
    }

    #[test]
    fn paired_quotes_and_book_titles_protect_inner_text() {
        let source = "我們讀《臺積電與乾紅》，也聽到“髮型品牌”，然後瞭解。";
        let expected = "我们读《臺積電與乾紅》，也听到“髮型品牌”，然后了解。";
        assert_eq!(normalize_to_simplified(source).unwrap(), expected);
    }

    #[test]
    fn inline_code_urls_and_emails_are_unchanged() {
        let source = "我們執行 `echo 臺積電`，訪問 https://example.com/臺積電?x=1，寄到 A.B+tag@example.com，然後瞭解。";
        let expected = "我们执行 `echo 臺積電`，访问 https://example.com/臺積電?x=1，寄到 A.B+tag@example.com，然后了解。";
        assert_eq!(normalize_to_simplified(source).unwrap(), expected);
    }

    #[test]
    fn unlabelled_ambiguous_text_uses_normal_opencc() {
        assert_eq!(
            normalize_to_simplified("臺積電與颱積電正在發展髮型產品。").unwrap(),
            "台积电与台积电正在发展发型产品。"
        );
    }

    #[test]
    fn simplified_and_non_chinese_text_are_stable() {
        assert_eq!(
            normalize_to_simplified("我们团队正在进行这个项目。OpenRouter 3.7").unwrap(),
            "我们团队正在进行这个项目。OpenRouter 3.7"
        );
        assert_eq!(normalize_to_simplified("hello 123").unwrap(), "hello 123");
    }

    #[test]
    fn katakana_middle_dot_does_not_disable_chinese_normalization() {
        assert_eq!(
            normalize_to_simplified("我們・團隊瞭解這個項目。").unwrap(),
            "我们・团队了解这个项目。"
        );
    }

    #[test]
    fn japanese_with_kana_is_not_rewritten_as_chinese() {
        let japanese = "これは繁體字の説明です。";
        assert_eq!(normalize_to_simplified(japanese).unwrap(), japanese);
    }

    #[test]
    fn supplementary_and_extended_kana_preserve_the_whole_text() {
        for kana in ['\u{1aff0}', '\u{1b000}', '\u{1b11f}', '\u{1b150}'] {
            let mixed = format!("我們{kana}團隊瞭解。");
            assert_eq!(normalize_to_simplified(&mixed).unwrap(), mixed);
        }
    }

    #[test]
    fn protected_projection_is_idempotent() {
        let source = "項目代號是乾紅，乾坤的乾；帳號是乾User；我們讀《臺積電》，然後瞭解。";
        let once = normalize_to_simplified(source).unwrap();
        let twice = normalize_to_simplified(&once).unwrap();
        assert_eq!(twice, once);
    }
}
