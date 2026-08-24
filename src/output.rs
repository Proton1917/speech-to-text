use std::collections::{BTreeSet, hash_map::DefaultHasher};
use std::fs;
use std::hash::{Hash, Hasher};
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use fs2::FileExt;
use tempfile::NamedTempFile;

use crate::config::Config;
use crate::media::{AudioInfo, ImageInfo};
use crate::openrouter::Completion;
use crate::security::{secure_directory, secure_file};

#[derive(Clone, Debug)]
pub struct TranscriptPart {
    pub start_ms: u64,
    pub end_ms: u64,
    pub completion: Completion,
    pub auxiliary_completions: Vec<Completion>,
    pub speaker_ids: Vec<String>,
    pub turn_count: usize,
    pub acoustic_coverage_warning: bool,
}

pub struct AtomicOutput {
    path: PathBuf,
    parent: PathBuf,
    force: bool,
    temporary: NamedTempFile,
    _lock: OutputLock,
}

struct OutputLock(fs::File);

impl Drop for OutputLock {
    fn drop(&mut self) {
        let _ = FileExt::unlock(&self.0);
    }
}

impl AtomicOutput {
    pub fn begin(path: &Path, force: bool) -> Result<Self> {
        let lock = acquire_output_lock(path)?;
        ensure_output_available(path, force)?;
        let parent = output_parent(path);
        let parent_metadata = fs::metadata(parent)
            .with_context(|| format!("无法读取输出目录 {}", parent.display()))?;
        if !parent_metadata.is_dir() {
            bail!("输出位置的父路径不是目录：{}", parent.display());
        }
        let temporary = NamedTempFile::new_in(parent)
            .with_context(|| format!("无法在 {} 创建临时输出", parent.display()))?;
        secure_file(temporary.path())?;
        Ok(Self {
            path: path.to_owned(),
            parent: parent.to_owned(),
            force,
            temporary,
            _lock: lock,
        })
    }

    pub fn commit(mut self, content: &str) -> Result<()> {
        self.temporary
            .write_all(content.as_bytes())
            .context("无法写入临时文字稿")?;
        self.temporary.flush().context("无法刷新临时文字稿")?;
        self.temporary
            .as_file()
            .sync_all()
            .context("无法同步临时文字稿")?;
        ensure_output_available(&self.path, self.force)?;

        if self.force {
            self.temporary
                .persist(&self.path)
                .map_err(|error| error.error)
                .with_context(|| format!("无法替换输出文件 {}", self.path.display()))?;
        } else {
            self.temporary
                .persist_noclobber(&self.path)
                .map_err(|error| error.error)
                .with_context(|| format!("输出文件已存在或无法创建：{}", self.path.display()))?;
        }
        sync_directory(&self.parent)?;
        Ok(())
    }
}

pub fn render_transcript(
    input: &Path,
    config: &Config,
    info: &AudioInfo,
    parts: &[TranscriptPart],
) -> Result<String> {
    if parts.is_empty() {
        bail!("没有可写入的转写片段");
    }
    let file_name = input
        .file_name()
        .and_then(|name| name.to_str())
        .context("源文件名不是有效 UTF-8")?;
    let title = input
        .file_stem()
        .and_then(|name| name.to_str())
        .context("源文件缺少主文件名")?;

    let completions = parts
        .iter()
        .flat_map(|part| std::iter::once(&part.completion).chain(part.auxiliary_completions.iter()))
        .collect::<Vec<_>>();
    let models = completions
        .iter()
        .map(|completion| completion.model.as_str())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>()
        .join(", ");
    let providers = completions
        .iter()
        .map(|completion| completion.provider.as_str())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>()
        .join(", ");
    let prompt_tokens = completions
        .iter()
        .map(|completion| completion.prompt_tokens)
        .sum::<u64>();
    let completion_tokens = completions
        .iter()
        .map(|completion| completion.completion_tokens)
        .sum::<u64>();
    let reasoning_tokens = completions
        .iter()
        .map(|completion| completion.reasoning_tokens)
        .sum::<u64>();
    let visible_output_tokens = completion_tokens.saturating_sub(reasoning_tokens);
    let cost = completions
        .iter()
        .map(|completion| completion.cost)
        .sum::<f64>();
    let all_speaker_ids = parts
        .iter()
        .flat_map(|part| part.speaker_ids.iter().cloned())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let unresolved_speaker_present = all_speaker_ids.iter().any(|speaker| speaker == "UNKNOWN");
    let mut speaker_ids = all_speaker_ids
        .into_iter()
        .filter(|speaker| speaker != "UNKNOWN")
        .collect::<Vec<_>>();
    speaker_ids.sort_by_key(|speaker| {
        speaker
            .strip_prefix('S')
            .and_then(|number| number.parse::<usize>().ok())
            .unwrap_or(usize::MAX)
    });
    let speaker_alignment_status = if unresolved_speaker_present {
        "unresolved_labels_present"
    } else {
        "structurally_resolved"
    };
    let acoustic_coverage_warning_present = parts.iter().any(|part| part.acoustic_coverage_warning);
    let usage_reported_for_all_requests = completions
        .iter()
        .all(|completion| completion.usage_reported);
    let reasoning_tokens_reported_for_all_requests = completions
        .iter()
        .all(|completion| completion.reasoning_tokens_reported);

    let mut markdown = String::new();
    markdown.push_str("---\n");
    markdown.push_str(&format!("source: {}\n", yaml_string(file_name)?));
    markdown.push_str(&format!(
        "model_requested: {}\n",
        yaml_string(&config.model)?
    ));
    markdown.push_str(&format!(
        "provider_requested: {}\n",
        yaml_string(&config.provider)?
    ));
    markdown.push_str(&format!(
        "provider_privacy_mode: {}\n",
        yaml_string(if config.uses_any_provider() {
            "any_explicit_privacy_downgrade"
        } else {
            "fixed_zdr_data_collection_denied"
        })?
    ));
    markdown.push_str(&format!(
        "model_reported_or_requested: {}\n",
        yaml_string(&models)?
    ));
    markdown.push_str(&format!(
        "provider_reported_or_requested: {}\n",
        yaml_string(&providers)?
    ));
    markdown.push_str(&format!(
        "duration_seconds: {:.3}\n",
        info.duration_ms as f64 / 1_000.0
    ));
    markdown.push_str(&format!("source_codec: {}\n", yaml_string(&info.codec)?));
    markdown.push_str(&format!(
        "source_container: {}\n",
        yaml_string(&info.container)?
    ));
    markdown.push_str("chinese_script: \"zh-Hans\"\n");
    markdown.push_str("chinese_normalization: \"opencc-t2s\"\n");
    markdown.push_str(&format!("segments: {}\n", parts.len()));
    markdown.push_str(&format!(
        "accepted_model_responses: {}\n",
        completions.len()
    ));
    markdown.push_str(&format!(
        "speaker_turns: {}\n",
        parts.iter().map(|part| part.turn_count).sum::<usize>()
    ));
    markdown.push_str("speaker_tracking: \"openrouter-two-stage-reference-harness-v2\"\n");
    markdown.push_str("speaker_identity_scope: \"whole_job\"\n");
    markdown.push_str("speaker_identity_guarantee: \"best_effort\"\n");
    markdown.push_str("speaker_id_assignment: \"host_managed\"\n");
    markdown.push_str("speaker_names_inferred: false\n");
    markdown.push_str(&format!(
        "speaker_alignment_status: \"{speaker_alignment_status}\"\n"
    ));
    markdown.push_str(&format!(
        "acoustic_coverage_status: \"{}\"\n",
        if acoustic_coverage_warning_present {
            "ffmpeg_energy_advisory_warning"
        } else {
            "ffmpeg_energy_advisory_passed"
        }
    ));
    markdown.push_str(&format!(
        "speaker_overlap_seconds: {}\n",
        config.overlap_seconds
    ));
    markdown.push_str(&format!(
        "unresolved_speaker_present: {unresolved_speaker_present}\n"
    ));
    markdown.push_str(&format!("speaker_count: {}\n", speaker_ids.len()));
    markdown.push_str(&format!(
        "speaker_ids: {}\n",
        serde_json::to_string(&speaker_ids).context("无法序列化说话人列表")?
    ));
    markdown.push_str(&format!(
        "usage_reported_for_all_accepted_responses: {usage_reported_for_all_requests}\n"
    ));
    markdown.push_str(&format!(
        "reasoning_tokens_reported_for_all_accepted_responses: {reasoning_tokens_reported_for_all_requests}\n"
    ));
    markdown.push_str(&format!("reported_prompt_tokens: {prompt_tokens}\n"));
    markdown.push_str(&format!(
        "reported_completion_tokens: {completion_tokens}\n"
    ));
    markdown.push_str(&format!("reported_reasoning_tokens: {reasoning_tokens}\n"));
    markdown.push_str(&format!(
        "reported_visible_output_tokens: {visible_output_tokens}\n"
    ));
    markdown.push_str(&format!("reported_accepted_cost_usd: {cost:.9}\n"));
    markdown.push_str("---\n\n");
    markdown.push_str(&format!("# {} 转写稿\n\n", escape_markdown_text(title)));

    for part in parts {
        markdown.push_str(&format!(
            "## {}–{}\n\n",
            format_timestamp(part.start_ms),
            format_timestamp(part.end_ms)
        ));
        markdown.push_str(part.completion.text.trim());
        markdown.push_str("\n\n");
    }
    Ok(markdown)
}

pub fn render_ocr(
    input: &Path,
    config: &Config,
    info: &ImageInfo,
    completion: &Completion,
) -> Result<String> {
    let file_name = input
        .file_name()
        .and_then(|name| name.to_str())
        .context("源文件名不是有效 UTF-8")?;
    let title = input
        .file_stem()
        .and_then(|name| name.to_str())
        .context("源文件缺少主文件名")?;
    Ok(format!(
        "---\nsource: {}\nmodel_requested: {}\nprovider_requested: {}\nprovider_privacy_mode: {}\nmodel_reported_or_requested: {}\nprovider_reported_or_requested: {}\nsource_codec: {}\nsource_container: {}\nsource_width: {}\nsource_height: {}\nusage_reported: {}\nreasoning_tokens_reported: {}\nreported_prompt_tokens: {}\nreported_completion_tokens: {}\nreported_reasoning_tokens: {}\nreported_visible_output_tokens: {}\nreported_accepted_cost_usd: {:.9}\n---\n\n# {} OCR\n\n{}\n",
        yaml_string(file_name)?,
        yaml_string(&config.model)?,
        yaml_string(&config.provider)?,
        yaml_string(if config.uses_any_provider() {
            "any_explicit_privacy_downgrade"
        } else {
            "fixed_zdr_data_collection_denied"
        })?,
        yaml_string(&completion.model)?,
        yaml_string(&completion.provider)?,
        yaml_string(&info.codec)?,
        yaml_string(&info.container)?,
        info.width,
        info.height,
        completion.usage_reported,
        completion.reasoning_tokens_reported,
        completion.prompt_tokens,
        completion.completion_tokens,
        completion.reasoning_tokens,
        completion.visible_output_tokens(),
        completion.cost,
        escape_markdown_text(title),
        escape_markdown_text(completion.text.trim()),
    ))
}

pub fn ocr_output_path(input: &Path) -> Result<PathBuf> {
    let canonical = fs::canonicalize(input)
        .with_context(|| format!("无法解析 OCR 输入真实路径 {}", input.display()))?;
    let stem = canonical
        .file_stem()
        .and_then(|name| name.to_str())
        .context("OCR 源文件缺少有效主文件名")?;
    Ok(canonical.with_file_name(format!("{stem}.ocr.md")))
}

pub fn write_private_atomic(path: &Path, content: &str, force: bool) -> Result<()> {
    AtomicOutput::begin(path, force)?.commit(content)
}

pub fn ensure_output_available(path: &Path, force: bool) -> Result<()> {
    match fs::symlink_metadata(path) {
        Ok(metadata) => {
            if metadata.file_type().is_symlink() || !metadata.file_type().is_file() {
                bail!("输出目标不是普通文件，拒绝覆盖：{}", path.display());
            }
            if !force {
                bail!("输出文件已存在：{}；如需替换请使用 --force", path.display());
            }
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => {
            return Err(error).with_context(|| format!("无法检查输出目标 {}", path.display()));
        }
    }
    Ok(())
}

pub fn format_timestamp(milliseconds: u64) -> String {
    let total_seconds = milliseconds / 1_000;
    let hours = total_seconds / 3_600;
    let minutes = (total_seconds % 3_600) / 60;
    let seconds = total_seconds % 60;
    if milliseconds.is_multiple_of(1_000) {
        format!("{hours:02}:{minutes:02}:{seconds:02}")
    } else {
        format!(
            "{hours:02}:{minutes:02}:{seconds:02}.{:03}",
            milliseconds % 1_000
        )
    }
}

fn yaml_string(value: &str) -> Result<String> {
    serde_json::to_string(value).context("无法转义 Markdown 元数据")
}

pub fn escape_markdown_text(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
    for character in value.chars() {
        match character {
            '&' => escaped.push_str("&amp;"),
            '<' => escaped.push_str("&lt;"),
            '>' => escaped.push_str("&gt;"),
            '\\' | '`' | '*' | '_' | '{' | '}' | '[' | ']' | '(' | ')' | '#' | '+' | '-' | '.'
            | '!' | '|' | '~' => {
                escaped.push('\\');
                escaped.push(character);
            }
            _ => escaped.push(character),
        }
    }
    escaped
}

fn output_parent(path: &Path) -> &Path {
    path.parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
}

fn acquire_output_lock(path: &Path) -> Result<OutputLock> {
    let parent = output_parent(path);
    let canonical_parent = fs::canonicalize(parent)
        .with_context(|| format!("无法解析输出目录 {}", parent.display()))?;
    let file_name = path.file_name().context("输出路径缺少文件名")?;
    let normalized_target = canonical_parent.join(file_name);
    let mut hasher = DefaultHasher::new();
    normalized_target.hash(&mut hasher);

    let cache_root = dirs::cache_dir()
        .or_else(|| dirs::home_dir().map(|home| home.join(".cache")))
        .context("无法确定输出锁目录")?;
    let lock_directory = cache_root.join("spt/output-locks");
    fs::create_dir_all(&lock_directory)
        .with_context(|| format!("无法创建输出锁目录 {}", lock_directory.display()))?;
    secure_directory(&lock_directory)?;
    let lock_path = lock_directory.join(format!("{:016x}.lock", hasher.finish()));
    let file = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(&lock_path)
        .with_context(|| format!("无法打开输出锁 {}", lock_path.display()))?;
    secure_file(&lock_path)?;
    match file.try_lock_exclusive() {
        Ok(()) => Ok(OutputLock(file)),
        Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
            bail!("另一个 spt 任务正在处理同一输出：{}", path.display())
        }
        Err(error) => Err(error).with_context(|| format!("无法获取输出锁 {}", lock_path.display())),
    }
}

#[cfg(unix)]
fn sync_directory(path: &Path) -> Result<()> {
    fs::File::open(path)
        .and_then(|directory| directory.sync_all())
        .with_context(|| format!("无法同步输出目录 {}", path.display()))
}

#[cfg(not(unix))]
fn sync_directory(_path: &Path) -> Result<()> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_long_and_fractional_timestamps() {
        assert_eq!(format_timestamp(0), "00:00:00");
        assert_eq!(format_timestamp(3_661_000), "01:01:01");
        assert_eq!(format_timestamp(61_234), "00:01:01.234");
    }

    #[test]
    fn atomic_writer_refuses_existing_file_without_force() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("result.md");
        fs::write(&path, "old").unwrap();
        assert!(write_private_atomic(&path, "new", false).is_err());
        assert_eq!(fs::read_to_string(&path).unwrap(), "old");
        write_private_atomic(&path, "new", true).unwrap();
        assert_eq!(fs::read_to_string(&path).unwrap(), "new");
    }

    #[test]
    fn ocr_output_uses_distinct_suffix() {
        let directory = tempfile::tempdir().unwrap();
        let source = directory.path().join("scan.page.png");
        fs::write(&source, b"fixture").unwrap();
        assert_eq!(
            ocr_output_path(&source).unwrap(),
            fs::canonicalize(directory.path())
                .unwrap()
                .join("scan.page.ocr.md")
        );
    }

    #[test]
    fn relative_output_uses_current_directory() {
        assert_eq!(output_parent(Path::new("meeting.md")), Path::new("."));
        assert_eq!(
            output_parent(Path::new("notes/meeting.md")),
            Path::new("notes")
        );
    }

    #[test]
    fn markdown_escape_blocks_html_and_remote_links() {
        let escaped = escape_markdown_text("<img src=\"https://evil\"> [打开](https://evil)");
        assert!(!escaped.contains("<img"));
        assert!(!escaped.contains("[打开]("));
        assert!(escaped.contains("&lt;img"));
        assert!(escaped.contains("\\[打开\\]\\("));
    }

    #[test]
    fn transcript_metadata_records_deterministic_zh_hans_normalization() {
        let part = TranscriptPart {
            start_ms: 0,
            end_ms: 1_000,
            completion: Completion {
                text: "S1：我们".into(),
                model: "test/model".into(),
                provider: "test/provider".into(),
                prompt_tokens: 1,
                completion_tokens: 1,
                reasoning_tokens: 0,
                cost: 0.0,
                usage_reported: true,
                reasoning_tokens_reported: true,
            },
            auxiliary_completions: Vec::new(),
            speaker_ids: vec!["S1".into()],
            turn_count: 1,
            acoustic_coverage_warning: false,
        };
        let markdown = render_transcript(
            Path::new("meeting.wav"),
            &Config::default(),
            &AudioInfo {
                duration_ms: 1_000,
                codec: "pcm_s16le".into(),
                container: "wav".into(),
            },
            &[part],
        )
        .unwrap();
        assert!(markdown.contains("chinese_script: \"zh-Hans\""));
        assert!(markdown.contains("chinese_normalization: \"opencc-t2s\""));
    }
}
