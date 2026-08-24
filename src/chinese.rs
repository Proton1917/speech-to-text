use std::sync::OnceLock;

use anyhow::{Result, anyhow, bail};
use opencc_fmmseg::OpenCC;

static T2S_CONVERTER: OnceLock<std::result::Result<OpenCC, String>> = OnceLock::new();

/// Deterministically normalizes Chinese model output to Simplified Chinese.
/// Japanese turns containing kana are preserved because OpenCC cannot infer
/// the spoken language from Han characters alone.
pub fn normalize_to_simplified(text: &str) -> Result<String> {
    if text.is_empty() || !text.chars().any(is_han) || text.chars().any(is_kana) {
        return Ok(text.to_owned());
    }

    let converter = t2s_converter()?;
    let normalized = converter.t2s(text, false);
    if normalized.is_empty() && !text.is_empty() {
        bail!("OpenCC t2s 返回了异常空文本");
    }
    Ok(normalized)
}

pub fn ensure_simplified_converter() -> Result<()> {
    t2s_converter().map(|_| ())
}

fn t2s_converter() -> Result<&'static OpenCC> {
    let converter = T2S_CONVERTER.get_or_init(|| {
        OpenCC::clear_last_error();
        let mut converter = OpenCC::new();
        converter.set_parallel(false);
        match OpenCC::get_last_error() {
            Some(error) => Err(error),
            None => Ok(converter),
        }
    });
    let converter = converter
        .as_ref()
        .map_err(|error| anyhow!("无法初始化内置 OpenCC t2s 词典：{error}"))?;
    Ok(converter)
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
    matches!(
        character as u32,
        0x3040..=0x30FF | 0x31F0..=0x31FF | 0xFF65..=0xFF9F
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mixed_traditional_transcript_is_normalized_to_zh_hans() {
        ensure_simplified_converter().unwrap();
        let source = "瞭解，但是我們團隊希望對於這個項目進行結合，傳統中藥文化與年輕群體。";
        let normalized = normalize_to_simplified(source).unwrap();
        assert_eq!(
            normalized,
            "了解，但是我们团队希望对于这个项目进行结合，传统中药文化与年轻群体。"
        );
    }

    #[test]
    fn observed_transcript_vocabulary_is_normalized() {
        let source = "那麼我們團隊在2025年成立，將傳統中藥文化與年輕群體進行結合，並在門店進行研發，然後到第五頁。";
        assert_eq!(
            normalize_to_simplified(source).unwrap(),
            "那么我们团队在2025年成立，将传统中药文化与年轻群体进行结合，并在门店进行研发，然后到第五页。"
        );
    }

    #[test]
    fn simplified_and_non_chinese_text_are_stable() {
        assert_eq!(
            normalize_to_simplified("我们团队正在进行这个项目。OpenRouter 3.5").unwrap(),
            "我们团队正在进行这个项目。OpenRouter 3.5"
        );
        assert_eq!(normalize_to_simplified("hello 123").unwrap(), "hello 123");
    }

    #[test]
    fn japanese_with_kana_is_not_rewritten_as_chinese() {
        let japanese = "これは繁體字の説明です。";
        assert_eq!(normalize_to_simplified(japanese).unwrap(), japanese);
    }
}
