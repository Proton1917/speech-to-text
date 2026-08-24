use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use unicode_normalization::{is_nfc, UnicodeNormalization};

#[derive(Clone, Debug, Eq, PartialEq)]
struct Turn {
    speaker: String,
    text: String,
}

#[derive(Clone, Debug)]
struct ExpectedTerm {
    category: String,
    id: String,
    variants: Vec<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum AlignmentOp {
    None,
    Match,
    DeleteReference,
    InsertHypothesis,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CharacterAlignmentOp {
    None,
    Diagonal,
    DeleteReference,
    InsertHypothesis,
}

const MAX_TURN_ALIGNMENT_CER: f64 = 0.60;

#[derive(Debug)]
struct SpeakerScore {
    correct: usize,
    denominator: usize,
    aligned_pairs: usize,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("spt-bench: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let mut arguments = env::args().skip(1);
    let Some(command) = arguments.next() else {
        print_help();
        return Ok(());
    };
    if command == "--help" || command == "-h" || command == "help" {
        print_help();
        return Ok(());
    }
    if command != "evaluate" {
        return Err(format!("未知命令 {command:?}；当前只支持 evaluate"));
    }

    let mut case_directory = None::<PathBuf>;
    let mut transcript_path = None::<PathBuf>;
    let mut run_path = None::<PathBuf>;
    let mut report_path = None::<PathBuf>;
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--case" => case_directory = Some(next_path(&mut arguments, "--case")?),
            "--transcript" => transcript_path = Some(next_path(&mut arguments, "--transcript")?),
            "--run" => run_path = Some(next_path(&mut arguments, "--run")?),
            "--report" => report_path = Some(next_path(&mut arguments, "--report")?),
            "--help" | "-h" => {
                print_help();
                return Ok(());
            }
            _ => return Err(format!("未知参数 {argument:?}")),
        }
    }

    let case_directory = case_directory.ok_or_else(|| "缺少 --case DIR".to_owned())?;
    let report = evaluate(
        &case_directory,
        transcript_path.as_deref(),
        run_path.as_deref(),
    )?;
    if let Some(path) = report_path {
        write_atomic(&path, &report)?;
    } else {
        print!("{report}");
    }
    Ok(())
}

fn next_path(
    arguments: &mut impl Iterator<Item = String>,
    option: &str,
) -> Result<PathBuf, String> {
    arguments
        .next()
        .map(PathBuf::from)
        .ok_or_else(|| format!("{option} 缺少路径参数"))
}

fn print_help() {
    println!(
        "spt-bench：离线评估一份 spt Markdown 转写\n\n\
用法：\n  spt-bench evaluate --case CASE_DIR [--transcript FILE] [--run RUN_TSV] [--report REPORT_TSV]\n\n\
评测器本身不执行 spt、不访问网络，也不读取 OPENROUTER_API_KEY。失败的运行可只提供 --run；\n\
此时失败、耗时、调用和成本字段仍会进入报告，内容准确率指标记为 NA。"
    );
}

fn evaluate(
    case_directory: &Path,
    transcript_path: Option<&Path>,
    run_path: Option<&Path>,
) -> Result<String, String> {
    let case_metadata = read_key_value_tsv(&case_directory.join("case.tsv"))?;
    let case_id = required_field(&case_metadata, "case_id", "case.tsv")?.to_owned();
    let language = required_field(&case_metadata, "language", "case.tsv")?;
    if !language.starts_with("zh-") {
        return Err(format!(
            "当前文字规范化合同只支持中文用例（language=zh-*），实际为 {language:?}"
        ));
    }
    let unicode_normalization =
        required_field(&case_metadata, "unicode_normalization", "case.tsv")?;
    if unicode_normalization != "NFC" {
        return Err(format!(
            "case.tsv 的 unicode_normalization 必须是 NFC，实际为 {unicode_normalization:?}"
        ));
    }
    for (key, value) in &case_metadata {
        ensure_declared_nfc(&format!("case.tsv 字段 {key:?}"), value)?;
    }
    let reference_turns = read_turns(&case_directory.join("turns.tsv"))?;
    if reference_turns.is_empty() {
        return Err("turns.tsv 至少需要一个参考 turn".to_owned());
    }
    let expected_terms = read_terms(&case_directory.join("terms.tsv"))?;
    for turn in &reference_turns {
        ensure_declared_nfc("turns.tsv 参考说话人", &turn.speaker)?;
        ensure_declared_nfc("turns.tsv 参考文字", &turn.text)?;
    }
    for term in &expected_terms {
        ensure_declared_nfc("terms.tsv category", &term.category)?;
        ensure_declared_nfc("terms.tsv id", &term.id)?;
        for variant in &term.variants {
            ensure_declared_nfc("terms.tsv 允许变体", variant)?;
        }
    }
    let run_metadata = match run_path {
        Some(path) => read_key_value_tsv(path)?,
        None => BTreeMap::new(),
    };

    let transcript = match transcript_path {
        Some(path) => Some(
            fs::read_to_string(path)
                .map_err(|error| format!("无法读取转写 {}：{error}", path.display()))?,
        ),
        None => None,
    };
    let (front_matter, hypothesis_turns) = match transcript.as_deref() {
        Some(content) => parse_spt_markdown(content),
        None => (BTreeMap::new(), Vec::new()),
    };

    let default_status = if transcript.is_some() {
        "success"
    } else {
        "failure"
    };
    let run_status = run_metadata
        .get("status")
        .map(String::as_str)
        .unwrap_or(default_status);
    if run_status == "success" && transcript.is_none() {
        return Err("run.tsv 声明 status=success，但没有提供 --transcript".to_owned());
    }
    let failed = run_status != "success";

    let mut report = Vec::<(String, String)>::new();
    push_field(&mut report, "benchmark_schema", "1");
    push_field(&mut report, "case_id", &case_id);
    push_field(&mut report, "language", language);
    push_field(&mut report, "unicode_normalization", unicode_normalization);
    push_field(&mut report, "run_id", field_or_na(&run_metadata, "run_id"));
    push_field(&mut report, "run_status", run_status);
    push_field(&mut report, "failed", if failed { "true" } else { "false" });
    push_field(
        &mut report,
        "failure_kind",
        field_or_na(&run_metadata, "failure_kind"),
    );
    push_field(
        &mut report,
        "exit_code",
        field_or_na(&run_metadata, "exit_code"),
    );
    push_field(
        &mut report,
        "elapsed_ms",
        field_or_na(&run_metadata, "elapsed_ms"),
    );
    push_field(
        &mut report,
        "http_attempts",
        field_or_na(&run_metadata, "http_attempts"),
    );
    push_field(
        &mut report,
        "accounted_model_responses",
        first_field(
            &run_metadata,
            &front_matter,
            "model_responses",
            "accounted_model_responses",
        ),
    );
    push_field(
        &mut report,
        "reported_accounted_cost_usd",
        first_field(
            &run_metadata,
            &front_matter,
            "cost_usd",
            "reported_accounted_cost_usd",
        ),
    );
    push_field(
        &mut report,
        "model_requested",
        first_field(&run_metadata, &front_matter, "model", "model_requested"),
    );
    push_field(
        &mut report,
        "quality_model_requested",
        first_field(
            &run_metadata,
            &front_matter,
            "quality_model",
            "quality_review_model_requested",
        ),
    );
    push_field(
        &mut report,
        "provider_requested",
        first_field(
            &run_metadata,
            &front_matter,
            "provider",
            "asr_provider_expected",
        ),
    );
    push_field(
        &mut report,
        "transcript_mode",
        run_metadata
            .get("mode")
            .or_else(|| front_matter.get("transcript_mode"))
            .map(String::as_str)
            .unwrap_or("NA"),
    );

    let reference_text = reference_turns
        .iter()
        .map(|turn| turn.text.as_str())
        .collect::<String>();
    let normalized_reference = normalize_text(&reference_text);
    ensure_nonempty_normalized_reference(&normalized_reference)?;
    let exact_reference = normalize_exact_text(&reference_text);
    validate_term_locations(&expected_terms, &exact_reference)?;
    push_field(
        &mut report,
        "reference_characters",
        &normalized_reference.chars().count().to_string(),
    );
    push_field(
        &mut report,
        "reference_turns",
        &reference_turns.len().to_string(),
    );

    if transcript.is_some() {
        let hypothesis_text = hypothesis_turns
            .iter()
            .map(|turn| turn.text.as_str())
            .collect::<String>();
        let normalized_hypothesis = normalize_text(&hypothesis_text);
        let exact_hypothesis = normalize_exact_text(&hypothesis_text);
        let reference_characters = normalized_reference.chars().collect::<Vec<_>>();
        let hypothesis_characters = normalized_hypothesis.chars().collect::<Vec<_>>();
        let character_errors = levenshtein(&reference_characters, &hypothesis_characters);
        let cer = character_errors as f64 / reference_characters.len() as f64;
        push_field(
            &mut report,
            "hypothesis_characters",
            &hypothesis_characters.len().to_string(),
        );
        push_field(
            &mut report,
            "character_errors",
            &character_errors.to_string(),
        );
        push_field(&mut report, "cer", &format_ratio(cer));

        append_term_metrics(
            &mut report,
            &expected_terms,
            &exact_reference,
            &exact_hypothesis,
        )?;

        let speaker_score = score_speakers(&reference_turns, &hypothesis_turns)?;
        push_field(
            &mut report,
            "hypothesis_turns",
            &hypothesis_turns.len().to_string(),
        );
        push_field(
            &mut report,
            "speaker_aligned_turn_pairs",
            &speaker_score.aligned_pairs.to_string(),
        );
        push_field(
            &mut report,
            "speaker_turn_alignment_max_cer",
            &format_ratio(MAX_TURN_ALIGNMENT_CER),
        );
        push_field(
            &mut report,
            "speaker_turn_correct",
            &speaker_score.correct.to_string(),
        );
        push_field(
            &mut report,
            "speaker_turn_denominator",
            &speaker_score.denominator.to_string(),
        );
        let speaker_accuracy =
            speaker_score.correct as f64 / speaker_score.denominator.max(1) as f64;
        push_field(
            &mut report,
            "speaker_permutation_invariant_turn_accuracy",
            &format_ratio(speaker_accuracy),
        );
        push_field(
            &mut report,
            "speaker_turn_error_rate",
            &format_ratio(1.0 - speaker_accuracy),
        );
    } else {
        push_field(&mut report, "hypothesis_characters", "NA");
        push_field(&mut report, "character_errors", "NA");
        push_field(&mut report, "cer", "NA");
        append_unavailable_term_metrics(&mut report, &expected_terms);
        push_field(&mut report, "hypothesis_turns", "NA");
        push_field(&mut report, "speaker_aligned_turn_pairs", "NA");
        push_field(
            &mut report,
            "speaker_turn_alignment_max_cer",
            &format_ratio(MAX_TURN_ALIGNMENT_CER),
        );
        push_field(&mut report, "speaker_turn_correct", "NA");
        push_field(
            &mut report,
            "speaker_turn_denominator",
            &reference_turns.len().to_string(),
        );
        push_field(
            &mut report,
            "speaker_permutation_invariant_turn_accuracy",
            "NA",
        );
        push_field(&mut report, "speaker_turn_error_rate", "NA");
    }

    let mut output = String::from("metric\tvalue\n");
    for (key, value) in report {
        output.push_str(&sanitize_tsv(&key));
        output.push('\t');
        output.push_str(&sanitize_tsv(&value));
        output.push('\n');
    }
    Ok(output)
}

fn push_field(report: &mut Vec<(String, String)>, key: &str, value: &str) {
    report.push((key.to_owned(), value.to_owned()));
}

fn field_or_na<'a>(fields: &'a BTreeMap<String, String>, key: &str) -> &'a str {
    fields.get(key).map(String::as_str).unwrap_or("NA")
}

fn first_field<'a>(
    preferred: &'a BTreeMap<String, String>,
    fallback: &'a BTreeMap<String, String>,
    preferred_key: &str,
    fallback_key: &str,
) -> &'a str {
    preferred
        .get(preferred_key)
        .or_else(|| fallback.get(fallback_key))
        .map(String::as_str)
        .unwrap_or("NA")
}

fn append_term_metrics(
    report: &mut Vec<(String, String)>,
    terms: &[ExpectedTerm],
    exact_reference: &str,
    exact_hypothesis: &str,
) -> Result<(), String> {
    let reference_characters = exact_reference.chars().collect::<Vec<_>>();
    let hypothesis_characters = exact_hypothesis.chars().collect::<Vec<_>>();
    let (gap_starts, gap_ends) =
        character_alignment_gaps(&reference_characters, &hypothesis_characters);
    let categories = required_and_present_categories(terms);
    let mut macro_total = 0.0;
    let mut macro_categories = 0usize;
    for category in categories {
        let category_terms = terms
            .iter()
            .filter(|term| term.category == category)
            .collect::<Vec<_>>();
        let mut missed = Vec::new();
        let mut hits = 0usize;
        for term in &category_terms {
            let hit = exact_term_hit(
                term,
                &reference_characters,
                &hypothesis_characters,
                &gap_starts,
                &gap_ends,
            )?;
            if hit {
                hits += 1;
            } else {
                missed.push(term.id.clone());
            }
        }
        let key = metric_key(&category);
        push_field(report, &format!("{key}_exact_hits"), &hits.to_string());
        push_field(
            report,
            &format!("{key}_exact_total"),
            &category_terms.len().to_string(),
        );
        if category_terms.is_empty() {
            push_field(report, &format!("{key}_exact_recall"), "NA");
        } else {
            let recall = hits as f64 / category_terms.len() as f64;
            push_field(
                report,
                &format!("{key}_exact_recall"),
                &format_ratio(recall),
            );
            macro_total += recall;
            macro_categories += 1;
        }
        let missed_ids = if missed.is_empty() {
            "[]".to_owned()
        } else {
            missed.join(",")
        };
        push_field(report, &format!("{key}_missed_ids"), &missed_ids);
    }
    let macro_recall = if macro_categories == 0 {
        "NA".to_owned()
    } else {
        format_ratio(macro_total / macro_categories as f64)
    };
    push_field(report, "term_category_macro_exact_recall", &macro_recall);
    Ok(())
}

fn append_unavailable_term_metrics(report: &mut Vec<(String, String)>, terms: &[ExpectedTerm]) {
    for category in required_and_present_categories(terms) {
        let key = metric_key(&category);
        let total = terms
            .iter()
            .filter(|term| term.category == category)
            .count();
        push_field(report, &format!("{key}_exact_hits"), "NA");
        push_field(report, &format!("{key}_exact_total"), &total.to_string());
        push_field(report, &format!("{key}_exact_recall"), "NA");
        push_field(report, &format!("{key}_missed_ids"), "NA");
    }
    push_field(report, "term_category_macro_exact_recall", "NA");
}

fn required_and_present_categories(terms: &[ExpectedTerm]) -> BTreeSet<String> {
    let mut categories = ["number", "proper_name", "negation", "condition"]
        .into_iter()
        .map(str::to_owned)
        .collect::<BTreeSet<_>>();
    categories.extend(terms.iter().map(|term| term.category.clone()));
    categories
}

fn metric_key(category: &str) -> String {
    let mut key = String::new();
    let mut previous_underscore = false;
    for character in category.chars() {
        if character.is_ascii_alphanumeric() {
            key.push(character.to_ascii_lowercase());
            previous_underscore = false;
        } else if !previous_underscore {
            key.push('_');
            previous_underscore = true;
        }
    }
    key.trim_matches('_').to_owned()
}

fn read_key_value_tsv(path: &Path) -> Result<BTreeMap<String, String>, String> {
    let content = fs::read_to_string(path)
        .map_err(|error| format!("无法读取 {}：{error}", path.display()))?;
    let mut fields = BTreeMap::new();
    for (line_index, line) in content.lines().enumerate() {
        let line = line.trim_end_matches('\r');
        if line.trim().is_empty() || line.trim_start().starts_with('#') {
            continue;
        }
        let (key, value) = line
            .split_once('\t')
            .ok_or_else(|| format!("{}:{} 必须是 key<TAB>value", path.display(), line_index + 1))?;
        let key = key.trim();
        let value = value.trim();
        if key.is_empty() || value.is_empty() {
            return Err(format!(
                "{}:{} 的 key/value 不能为空",
                path.display(),
                line_index + 1
            ));
        }
        if fields.insert(key.to_owned(), value.to_owned()).is_some() {
            return Err(format!("{} 中存在重复 key {key:?}", path.display()));
        }
    }
    Ok(fields)
}

fn required_field<'a>(
    fields: &'a BTreeMap<String, String>,
    key: &str,
    source: &str,
) -> Result<&'a str, String> {
    fields
        .get(key)
        .map(String::as_str)
        .ok_or_else(|| format!("{source} 缺少 {key}"))
}

fn read_turns(path: &Path) -> Result<Vec<Turn>, String> {
    let content = fs::read_to_string(path)
        .map_err(|error| format!("无法读取 {}：{error}", path.display()))?;
    let mut turns = Vec::new();
    for (line_index, line) in content.lines().enumerate() {
        let line = line.trim_end_matches('\r');
        if line.trim().is_empty() || line.trim_start().starts_with('#') {
            continue;
        }
        let (speaker, text) = line.split_once('\t').ok_or_else(|| {
            format!(
                "{}:{} 必须是 speaker<TAB>text",
                path.display(),
                line_index + 1
            )
        })?;
        if speaker.trim().is_empty() || text.trim().is_empty() {
            return Err(format!(
                "{}:{} 的 speaker/text 不能为空",
                path.display(),
                line_index + 1
            ));
        }
        turns.push(Turn {
            speaker: speaker.trim().to_owned(),
            text: text.trim().to_owned(),
        });
    }
    Ok(turns)
}

fn read_terms(path: &Path) -> Result<Vec<ExpectedTerm>, String> {
    let content = fs::read_to_string(path)
        .map_err(|error| format!("无法读取 {}：{error}", path.display()))?;
    let mut terms = Vec::new();
    let mut identifiers = BTreeSet::new();
    for (line_index, line) in content.lines().enumerate() {
        let line = line.trim_end_matches('\r');
        if line.trim().is_empty() || line.trim_start().starts_with('#') {
            continue;
        }
        let fields = line.splitn(3, '\t').collect::<Vec<_>>();
        if fields.len() != 3 {
            return Err(format!(
                "{}:{} 必须是 category<TAB>id<TAB>variant1|variant2",
                path.display(),
                line_index + 1
            ));
        }
        let category = fields[0].trim();
        let id = fields[1].trim();
        let variants = fields[2]
            .split('|')
            .map(str::trim)
            .filter(|variant| !variant.is_empty())
            .map(str::to_owned)
            .collect::<Vec<_>>();
        if category.is_empty() || id.is_empty() || variants.is_empty() {
            return Err(format!(
                "{}:{} 的 category/id/variants 不能为空",
                path.display(),
                line_index + 1
            ));
        }
        let unique_id = format!("{category}\t{id}");
        if !identifiers.insert(unique_id) {
            return Err(format!(
                "{}:{} 存在重复 category/id",
                path.display(),
                line_index + 1
            ));
        }
        terms.push(ExpectedTerm {
            category: category.to_owned(),
            id: id.to_owned(),
            variants,
        });
    }
    Ok(terms)
}

fn parse_spt_markdown(content: &str) -> (BTreeMap<String, String>, Vec<Turn>) {
    let mut front_matter = BTreeMap::new();
    let mut body_lines = Vec::new();
    let mut lines = content.lines();
    let first = lines
        .next()
        .unwrap_or_default()
        .trim_start_matches('\u{feff}');
    if first.trim() == "---" {
        let mut closed = false;
        for line in &mut lines {
            if line.trim() == "---" {
                closed = true;
                break;
            }
            if let Some((key, value)) = line.split_once(':') {
                front_matter.insert(key.trim().to_owned(), unquote(value.trim()));
            }
        }
        if closed {
            body_lines.extend(lines.map(str::to_owned));
        }
    } else {
        body_lines.push(first.to_owned());
        body_lines.extend(lines.map(str::to_owned));
    }

    let mut turns = Vec::<Turn>::new();
    for line in body_lines {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let plain = unescape_markdown(trimmed);
        if let Some((speaker, text)) = split_speaker_turn(&plain) {
            turns.push(Turn { speaker, text });
        } else if let Some(previous) = turns.last_mut() {
            previous.text.push(' ');
            previous.text.push_str(plain.trim());
        } else {
            turns.push(Turn {
                speaker: "UNKNOWN".to_owned(),
                text: plain,
            });
        }
    }
    (front_matter, turns)
}

fn unquote(value: &str) -> String {
    if value.len() >= 2 && value.starts_with('"') && value.ends_with('"') {
        value[1..value.len() - 1]
            .replace("\\\"", "\"")
            .replace("\\\\", "\\")
    } else {
        value.to_owned()
    }
}

fn split_speaker_turn(line: &str) -> Option<(String, String)> {
    let separator = line
        .char_indices()
        .find(|(_, character)| *character == '：' || *character == ':')?;
    let speaker = line[..separator.0].trim();
    if !valid_hypothesis_speaker(speaker) {
        return None;
    }
    let text_start = separator.0 + separator.1.len_utf8();
    let text = line[text_start..].trim();
    Some((speaker.to_owned(), text.to_owned()))
}

fn valid_hypothesis_speaker(speaker: &str) -> bool {
    speaker == "UNKNOWN"
        || speaker
            .strip_prefix('S')
            .is_some_and(|number| !number.is_empty() && number.chars().all(|c| c.is_ascii_digit()))
}

fn unescape_markdown(value: &str) -> String {
    let value = value
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">");
    let mut unescaped = String::with_capacity(value.len());
    let mut characters = value.chars();
    while let Some(character) = characters.next() {
        if character == '\\' {
            if let Some(next) = characters.next() {
                unescaped.push(next);
            }
        } else {
            unescaped.push(character);
        }
    }
    unescaped
}

fn normalize_text(value: &str) -> String {
    let mut output = String::new();
    for character in value.nfc() {
        let normalized = normalize_width(character);
        if normalized.is_alphanumeric() {
            output.extend(normalized.to_lowercase());
        }
    }
    output.nfc().collect()
}

fn normalize_exact_text(value: &str) -> String {
    let mut output = String::new();
    for character in value.nfc() {
        let normalized = normalize_width(character);
        if !normalized.is_whitespace() {
            output.extend(normalized.to_lowercase());
        }
    }
    output.nfc().collect()
}

fn normalize_width(character: char) -> char {
    if ('\u{ff01}'..='\u{ff5e}').contains(&character) {
        char::from_u32(character as u32 - 0xfee0).unwrap_or(character)
    } else if character == '\u{3000}' {
        ' '
    } else {
        character
    }
}

fn ensure_declared_nfc(source: &str, value: &str) -> Result<(), String> {
    if !is_nfc(value) {
        return Err(format!(
            "{source} 与 case.tsv 声明的 NFC 不一致；请先将内容规范化为 Unicode NFC"
        ));
    }
    Ok(())
}

fn ensure_nonempty_normalized_reference(normalized_reference: &str) -> Result<(), String> {
    if normalized_reference.is_empty() {
        Err("参考文字删除空白、标点和符号后为空，CER 没有合法分母".to_owned())
    } else {
        Ok(())
    }
}

fn validate_term_locations(terms: &[ExpectedTerm], exact_reference: &str) -> Result<(), String> {
    let reference = exact_reference.chars().collect::<Vec<_>>();
    for term in terms {
        let located = term.variants.iter().any(|variant| {
            let normalized = normalize_exact_text(variant).chars().collect::<Vec<_>>();
            !normalized.is_empty() && !find_occurrences(&reference, &normalized).is_empty()
        });
        if !located {
            return Err(format!(
                "terms.tsv 的 {}/{} 没有任何允许变体出现在参考文字中",
                term.category, term.id
            ));
        }
    }
    Ok(())
}

fn exact_term_hit(
    term: &ExpectedTerm,
    reference: &[char],
    hypothesis: &[char],
    gap_starts: &[usize],
    gap_ends: &[usize],
) -> Result<bool, String> {
    let accepted = term
        .variants
        .iter()
        .map(|variant| normalize_exact_text(variant).chars().collect::<Vec<_>>())
        .filter(|variant| !variant.is_empty())
        .collect::<Vec<_>>();
    if accepted.is_empty() {
        return Err(format!(
            "terms.tsv 的 {}/{} 归一化后没有有效变体",
            term.category, term.id
        ));
    }
    for reference_variant in &accepted {
        for start in find_occurrences(reference, reference_variant) {
            let end = start + reference_variant.len();
            let hypothesis_start = gap_starts[start];
            let hypothesis_end = gap_ends[end];
            if hypothesis_start <= hypothesis_end
                && accepted
                    .iter()
                    .any(|variant| hypothesis[hypothesis_start..hypothesis_end] == variant[..])
            {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

fn find_occurrences(haystack: &[char], needle: &[char]) -> Vec<usize> {
    if needle.is_empty() || needle.len() > haystack.len() {
        return Vec::new();
    }
    (0..=haystack.len() - needle.len())
        .filter(|&start| haystack[start..start + needle.len()] == needle[..])
        .collect()
}

fn character_alignment_gaps(reference: &[char], hypothesis: &[char]) -> (Vec<usize>, Vec<usize>) {
    let rows = reference.len() + 1;
    let columns = hypothesis.len() + 1;
    let mut costs = vec![0usize; rows * columns];
    let mut operations = vec![CharacterAlignmentOp::None; rows * columns];
    for reference_index in 1..rows {
        costs[reference_index * columns] = reference_index;
        operations[reference_index * columns] = CharacterAlignmentOp::DeleteReference;
    }
    for hypothesis_index in 1..columns {
        costs[hypothesis_index] = hypothesis_index;
        operations[hypothesis_index] = CharacterAlignmentOp::InsertHypothesis;
    }
    for reference_index in 1..rows {
        for hypothesis_index in 1..columns {
            let same = reference[reference_index - 1] == hypothesis[hypothesis_index - 1];
            let diagonal =
                costs[(reference_index - 1) * columns + hypothesis_index - 1] + usize::from(!same);
            let delete = costs[(reference_index - 1) * columns + hypothesis_index] + 1;
            let insert = costs[reference_index * columns + hypothesis_index - 1] + 1;
            let minimum = diagonal.min(delete).min(insert);
            let cell = reference_index * columns + hypothesis_index;
            costs[cell] = minimum;
            operations[cell] = if same && diagonal == minimum {
                CharacterAlignmentOp::Diagonal
            } else if insert == minimum {
                CharacterAlignmentOp::InsertHypothesis
            } else if delete == minimum {
                CharacterAlignmentOp::DeleteReference
            } else {
                CharacterAlignmentOp::Diagonal
            };
        }
    }

    let mut reference_index = reference.len();
    let mut hypothesis_index = hypothesis.len();
    let mut reversed = Vec::new();
    while reference_index > 0 || hypothesis_index > 0 {
        let operation = operations[reference_index * columns + hypothesis_index];
        reversed.push(operation);
        match operation {
            CharacterAlignmentOp::Diagonal => {
                reference_index -= 1;
                hypothesis_index -= 1;
            }
            CharacterAlignmentOp::DeleteReference => reference_index -= 1,
            CharacterAlignmentOp::InsertHypothesis => hypothesis_index -= 1,
            CharacterAlignmentOp::None => break,
        }
    }
    reversed.reverse();

    let mut gap_starts = vec![0usize; reference.len() + 1];
    let mut gap_ends = vec![0usize; reference.len() + 1];
    let mut operation_index = 0usize;
    let mut hypothesis_position = 0usize;
    for boundary in 0..=reference.len() {
        gap_starts[boundary] = hypothesis_position;
        while operations_at(&reversed, operation_index) == CharacterAlignmentOp::InsertHypothesis {
            hypothesis_position += 1;
            operation_index += 1;
        }
        gap_ends[boundary] = hypothesis_position;
        if boundary < reference.len() {
            match operations_at(&reversed, operation_index) {
                CharacterAlignmentOp::Diagonal => hypothesis_position += 1,
                CharacterAlignmentOp::DeleteReference => {}
                unexpected => debug_assert!(
                    false,
                    "character alignment expected reference-consuming op, got {unexpected:?}"
                ),
            }
            operation_index += 1;
        }
    }
    debug_assert_eq!(hypothesis_position, hypothesis.len());
    (gap_starts, gap_ends)
}

fn operations_at(operations: &[CharacterAlignmentOp], index: usize) -> CharacterAlignmentOp {
    operations
        .get(index)
        .copied()
        .unwrap_or(CharacterAlignmentOp::None)
}

fn levenshtein<T: Eq>(reference: &[T], hypothesis: &[T]) -> usize {
    if reference.is_empty() {
        return hypothesis.len();
    }
    if hypothesis.is_empty() {
        return reference.len();
    }
    let mut previous = (0..=hypothesis.len()).collect::<Vec<_>>();
    let mut current = vec![0usize; hypothesis.len() + 1];
    for (reference_index, reference_item) in reference.iter().enumerate() {
        current[0] = reference_index + 1;
        for (hypothesis_index, hypothesis_item) in hypothesis.iter().enumerate() {
            let substitution =
                previous[hypothesis_index] + usize::from(reference_item != hypothesis_item);
            let deletion = previous[hypothesis_index + 1] + 1;
            let insertion = current[hypothesis_index] + 1;
            current[hypothesis_index + 1] = substitution.min(deletion).min(insertion);
        }
        std::mem::swap(&mut previous, &mut current);
    }
    previous[hypothesis.len()]
}

fn score_speakers(reference: &[Turn], hypothesis: &[Turn]) -> Result<SpeakerScore, String> {
    let aligned_pairs = align_turns(reference, hypothesis);
    let reference_speakers = reference
        .iter()
        .map(|turn| turn.speaker.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let hypothesis_speakers = hypothesis
        .iter()
        .filter(|turn| turn.speaker != "UNKNOWN")
        .map(|turn| turn.speaker.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let reference_indices = reference_speakers
        .iter()
        .enumerate()
        .map(|(index, speaker)| (speaker.as_str(), index))
        .collect::<BTreeMap<_, _>>();
    let hypothesis_indices = hypothesis_speakers
        .iter()
        .enumerate()
        .map(|(index, speaker)| (speaker.as_str(), index))
        .collect::<BTreeMap<_, _>>();
    let mut weights = vec![vec![0usize; reference_speakers.len()]; hypothesis_speakers.len()];
    for &(reference_index, hypothesis_index) in &aligned_pairs {
        let hypothesis_speaker = hypothesis[hypothesis_index].speaker.as_str();
        let Some(&predicted_index) = hypothesis_indices.get(hypothesis_speaker) else {
            continue;
        };
        let reference_speaker = reference[reference_index].speaker.as_str();
        let expected_index = reference_indices[reference_speaker];
        weights[predicted_index][expected_index] += 1;
    }
    let correct = maximum_one_to_one_weight(&weights, reference_speakers.len());
    Ok(SpeakerScore {
        correct,
        denominator: reference.len().max(hypothesis.len()),
        aligned_pairs: aligned_pairs.len(),
    })
}

fn align_turns(reference: &[Turn], hypothesis: &[Turn]) -> Vec<(usize, usize)> {
    let rows = reference.len() + 1;
    let columns = hypothesis.len() + 1;
    let mut costs = vec![0.0f64; rows * columns];
    let mut operations = vec![AlignmentOp::None; rows * columns];
    for reference_index in 1..rows {
        costs[reference_index * columns] = reference_index as f64;
        operations[reference_index * columns] = AlignmentOp::DeleteReference;
    }
    for hypothesis_index in 1..columns {
        costs[hypothesis_index] = hypothesis_index as f64;
        operations[hypothesis_index] = AlignmentOp::InsertHypothesis;
    }
    for reference_index in 1..rows {
        for hypothesis_index in 1..columns {
            let match_cost = costs[(reference_index - 1) * columns + hypothesis_index - 1]
                + turn_substitution_cost(
                    &reference[reference_index - 1].text,
                    &hypothesis[hypothesis_index - 1].text,
                );
            let delete_cost = costs[(reference_index - 1) * columns + hypothesis_index] + 1.0;
            let insert_cost = costs[reference_index * columns + hypothesis_index - 1] + 1.0;
            let cell = reference_index * columns + hypothesis_index;
            if match_cost <= delete_cost && match_cost <= insert_cost {
                costs[cell] = match_cost;
                operations[cell] = AlignmentOp::Match;
            } else if delete_cost <= insert_cost {
                costs[cell] = delete_cost;
                operations[cell] = AlignmentOp::DeleteReference;
            } else {
                costs[cell] = insert_cost;
                operations[cell] = AlignmentOp::InsertHypothesis;
            }
        }
    }
    let mut reference_index = reference.len();
    let mut hypothesis_index = hypothesis.len();
    let mut pairs = Vec::new();
    while reference_index > 0 || hypothesis_index > 0 {
        match operations[reference_index * columns + hypothesis_index] {
            AlignmentOp::Match => {
                pairs.push((reference_index - 1, hypothesis_index - 1));
                reference_index -= 1;
                hypothesis_index -= 1;
            }
            AlignmentOp::DeleteReference => reference_index -= 1,
            AlignmentOp::InsertHypothesis => hypothesis_index -= 1,
            AlignmentOp::None => break,
        }
    }
    pairs.reverse();
    pairs
}

fn turn_substitution_cost(reference: &str, hypothesis: &str) -> f64 {
    let reference = normalize_text(reference).chars().collect::<Vec<_>>();
    let hypothesis = normalize_text(hypothesis).chars().collect::<Vec<_>>();
    let denominator = reference.len().max(hypothesis.len());
    if denominator == 0 {
        2.000_001
    } else {
        let cer = levenshtein(&reference, &hypothesis) as f64 / denominator as f64;
        if cer <= MAX_TURN_ALIGNMENT_CER {
            cer
        } else {
            2.000_001
        }
    }
}

fn maximum_one_to_one_weight(weights: &[Vec<usize>], reference_speakers: usize) -> usize {
    if weights.is_empty() || reference_speakers == 0 {
        return 0;
    }
    let size = weights.len().max(reference_speakers);
    let maximum_weight = weights
        .iter()
        .flat_map(|row| row.iter().copied())
        .max()
        .unwrap_or(0) as i64;
    let mut row_potential = vec![0i64; size + 1];
    let mut column_potential = vec![0i64; size + 1];
    let mut row_for_column = vec![0usize; size + 1];
    let mut predecessor = vec![0usize; size + 1];

    for row in 1..=size {
        row_for_column[0] = row;
        let mut current_column = 0usize;
        let mut minimum_reduced_cost = vec![i64::MAX; size + 1];
        let mut used = vec![false; size + 1];
        loop {
            used[current_column] = true;
            let current_row = row_for_column[current_column];
            let mut delta = i64::MAX;
            let mut next_column = 0usize;
            for column in 1..=size {
                if used[column] {
                    continue;
                }
                let weight = weights
                    .get(current_row - 1)
                    .and_then(|values| values.get(column - 1))
                    .copied()
                    .unwrap_or(0) as i64;
                let reduced_cost =
                    maximum_weight - weight - row_potential[current_row] - column_potential[column];
                if reduced_cost < minimum_reduced_cost[column] {
                    minimum_reduced_cost[column] = reduced_cost;
                    predecessor[column] = current_column;
                }
                if minimum_reduced_cost[column] < delta {
                    delta = minimum_reduced_cost[column];
                    next_column = column;
                }
            }
            for column in 0..=size {
                if used[column] {
                    row_potential[row_for_column[column]] += delta;
                    column_potential[column] -= delta;
                } else {
                    minimum_reduced_cost[column] -= delta;
                }
            }
            current_column = next_column;
            if row_for_column[current_column] == 0 {
                break;
            }
        }
        loop {
            let previous_column = predecessor[current_column];
            row_for_column[current_column] = row_for_column[previous_column];
            current_column = previous_column;
            if current_column == 0 {
                break;
            }
        }
    }

    (1..=size)
        .filter_map(|column| {
            let row = row_for_column[column];
            weights
                .get(row.checked_sub(1)?)
                .and_then(|values| values.get(column - 1))
                .copied()
        })
        .sum()
}

fn format_ratio(value: f64) -> String {
    format!("{value:.6}")
}

fn sanitize_tsv(value: &str) -> String {
    value
        .chars()
        .map(|character| match character {
            '\t' | '\n' | '\r' => ' ',
            _ => character,
        })
        .collect()
}

fn write_atomic(path: &Path, content: &str) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("无法创建报告目录 {}：{error}", parent.display()))?;
    }
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| format!("报告路径缺少有效文件名：{}", path.display()))?;
    let temporary = path.with_file_name(format!(".{file_name}.{}.tmp", std::process::id()));
    fs::write(&temporary, content)
        .map_err(|error| format!("无法写入临时报告 {}：{error}", temporary.display()))?;
    fs::rename(&temporary, path).map_err(|error| {
        let _ = fs::remove_file(&temporary);
        format!("无法提交报告 {}：{error}", path.display())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn turn(speaker: &str, text: &str) -> Turn {
        Turn {
            speaker: speaker.to_owned(),
            text: text.to_owned(),
        }
    }

    fn term(category: &str, id: &str, variants: &[&str]) -> ExpectedTerm {
        ExpectedTerm {
            category: category.to_owned(),
            id: id.to_owned(),
            variants: variants
                .iter()
                .map(|variant| (*variant).to_owned())
                .collect(),
        }
    }

    fn exact_hit(term: &ExpectedTerm, reference: &str, hypothesis: &str) -> bool {
        let reference = normalize_exact_text(reference).chars().collect::<Vec<_>>();
        let hypothesis = normalize_exact_text(hypothesis).chars().collect::<Vec<_>>();
        let (gap_starts, gap_ends) = character_alignment_gaps(&reference, &hypothesis);
        exact_term_hit(term, &reference, &hypothesis, &gap_starts, &gap_ends).unwrap()
    }

    #[test]
    fn markdown_parser_extracts_front_matter_and_speaker_turns() {
        let markdown = r#"---
transcript_mode: "raw"
accounted_model_responses: 2
---

# meeting 原始逐字稿

## 00:00:00–00:00:22

S1：预算是四十二万元\。

S2：测试环境周五之前就绪\。

S1：项目代号是阿尔法七号\。
"#;
        let (front_matter, turns) = parse_spt_markdown(markdown);
        assert_eq!(front_matter["transcript_mode"], "raw");
        assert_eq!(front_matter["accounted_model_responses"], "2");
        assert_eq!(turns.len(), 3);
        assert_eq!(turns[0], turn("S1", "预算是四十二万元。"));
        assert_eq!(turns[2].speaker, "S1");
    }

    #[test]
    fn cer_exposes_plausible_but_wrong_numbers_and_names() {
        let reference = normalize_text("预算是四十二万元。项目代号是阿尔法七号。");
        let hypothesis = normalize_text("预算是40万元。项目在号12法7号。");
        let errors = levenshtein(
            &reference.chars().collect::<Vec<_>>(),
            &hypothesis.chars().collect::<Vec<_>>(),
        );
        assert!(errors >= 5);
    }

    #[test]
    fn speaker_score_ignores_global_label_names() {
        let reference = vec![
            turn("A", "甲发言"),
            turn("B", "乙发言"),
            turn("A", "甲再发言"),
        ];
        let hypothesis = vec![
            turn("S2", "甲发言"),
            turn("S1", "乙发言"),
            turn("S2", "甲再发言"),
        ];
        let score = score_speakers(&reference, &hypothesis).unwrap();
        assert_eq!(score.correct, 3);
        assert_eq!(score.denominator, 3);
    }

    #[test]
    fn speaker_score_catches_cross_assignment_for_same_voice() {
        let reference = vec![
            turn("A", "甲发言"),
            turn("B", "乙发言"),
            turn("A", "甲再发言"),
        ];
        let hypothesis = vec![
            turn("S2", "甲发言"),
            turn("S1", "乙发言"),
            turn("S1", "甲再发言"),
        ];
        let score = score_speakers(&reference, &hypothesis).unwrap();
        assert_eq!(score.correct, 2);
        assert_eq!(score.denominator, 3);
    }

    #[test]
    fn extra_hypothesis_turn_penalizes_speaker_accuracy() {
        let reference = vec![turn("A", "甲"), turn("B", "乙")];
        let hypothesis = vec![turn("S1", "甲"), turn("S2", "乙"), turn("S1", "多余")];
        let score = score_speakers(&reference, &hypothesis).unwrap();
        assert_eq!(score.correct, 2);
        assert_eq!(score.denominator, 3);
    }

    #[test]
    fn hallucinated_turn_does_not_steal_exact_speaker_alignment() {
        let reference = vec![turn("A", "甲"), turn("B", "乙")];
        let hypothesis = vec![turn("S9", "多余"), turn("S8", "甲")];
        let score = score_speakers(&reference, &hypothesis).unwrap();
        assert_eq!(score.correct, 1);
        assert_eq!(score.aligned_pairs, 1);
        assert_eq!(score.denominator, 2);
    }

    #[test]
    fn exact_terms_use_reference_span_not_unbounded_substring() {
        let budget = term("number", "budget", &["四十二万元", "42万元"]);
        assert!(exact_hit(&budget, "预算是四十二万元。", "预算是42万元。"));
        assert!(!exact_hit(&budget, "预算是四十二万元。", "预算是142万元。"));

        let name = term("proper_name", "person", &["李雷"]);
        assert!(!exact_hit(&name, "李雷发言。", "李雷鸣发言。"));
    }

    #[test]
    fn exact_number_preserves_sign_decimal_point_and_percent() {
        let rate = term("number", "rate", &["-3.5%"]);
        assert!(exact_hit(&rate, "变化率是-3.5%。", "变化率是-3.5%。"));
        assert!(!exact_hit(&rate, "变化率是-3.5%。", "变化率是3.5%。"));
    }

    #[test]
    fn hungarian_speaker_mapping_supports_thirty_two_speakers() {
        let reference = (1..=32)
            .map(|index| turn(&format!("R{index}"), &format!("第{index}人发言")))
            .collect::<Vec<_>>();
        let hypothesis = (1..=32)
            .map(|index| turn(&format!("S{}", 33 - index), &format!("第{index}人发言")))
            .collect::<Vec<_>>();
        let score = score_speakers(&reference, &hypothesis).unwrap();
        assert_eq!(score.correct, 32);
        assert_eq!(score.denominator, 32);
    }

    #[test]
    fn reference_contract_rejects_text_that_is_not_declared_nfc() {
        assert!(ensure_declared_nfc("test", "ǎ").is_ok());
        assert!(ensure_declared_nfc("test", "a\u{030c}").is_err());
        assert!(ensure_declared_nfc("test", "が").is_ok());
        assert!(ensure_declared_nfc("test", "か\u{3099}").is_err());
        assert!(ensure_declared_nfc("test", "가").is_ok());
        assert!(ensure_declared_nfc("test", "\u{1100}\u{1161}").is_err());
    }

    #[test]
    fn nfc_normalization_equates_decomposed_japanese_and_hangul_hypotheses() {
        assert_eq!(normalize_text("か\u{3099}"), normalize_text("が"));
        assert_eq!(
            normalize_exact_text("か\u{3099}"),
            normalize_exact_text("が")
        );
        assert_eq!(normalize_text("\u{1100}\u{1161}"), normalize_text("가"));
        assert_eq!(
            normalize_exact_text("\u{1100}\u{1161}"),
            normalize_exact_text("가")
        );
    }

    #[test]
    fn punctuation_only_reference_has_no_cer_denominator() {
        let normalized = normalize_text("……？！");
        assert!(ensure_nonempty_normalized_reference(&normalized).is_err());
    }
}
