use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
#[cfg(test)]
use std::process;
#[cfg(test)]
use std::sync::OnceLock;

use anyhow::{Context, Result, bail};
use fs2::FileExt;
use tempfile::NamedTempFile;

use crate::config::Config;
use crate::media::{AudioInfo, ImageInfo};
use crate::openrouter::{Completion, CompletionOrigin};
use crate::security::secure_file;
use crate::transcript::TranscriptMode;

#[derive(Clone, Debug)]
pub struct TranscriptPart {
    pub start_ms: u64,
    pub end_ms: u64,
    pub completion: Completion,
    pub auxiliary_completions: Vec<Completion>,
    pub speaker_ids: Vec<String>,
    pub turn_count: usize,
    pub acoustic_coverage_warning: bool,
    pub quality_reviewed: bool,
    pub quality_review_advisory: bool,
    pub quality_cleanup_turns: usize,
    pub quality_trigger_codes: Vec<String>,
    pub quality_residual_advisory_codes: Vec<String>,
}

pub struct AtomicOutput {
    path: PathBuf,
    parent: PathBuf,
    force: bool,
    temporary: NamedTempFile,
    _lock: OutputLock,
}

struct OutputLock {
    files: Vec<fs::File>,
    _directories: Vec<fs::File>,
    #[cfg(test)]
    shard_path: PathBuf,
}

impl Drop for OutputLock {
    fn drop(&mut self) {
        for file in self.files.iter().rev() {
            let _ = FileExt::unlock(file);
        }
    }
}

impl AtomicOutput {
    pub fn begin(path: &Path, force: bool) -> Result<Self> {
        let lock = acquire_output_lock(path)?;
        Self::begin_with_lock(path, force, lock)
    }

    #[cfg(test)]
    fn begin_in(path: &Path, force: bool, lock_directory: &Path) -> Result<Self> {
        let lock = acquire_output_lock_in(path, lock_directory, false)?;
        Self::begin_with_lock(path, force, lock)
    }

    fn begin_with_lock(path: &Path, force: bool, lock: OutputLock) -> Result<Self> {
        ensure_output_available(path, force)?;
        let parent = output_parent(path);
        let parent_metadata = fs::metadata(parent)
            .with_context(|| format!("无法读取输出目录 {}", parent.display()))?;
        if !parent_metadata.is_dir() {
            bail!("输出位置的父路径不是目录：{}", parent.display());
        }
        validate_output_parent_security(parent)?;
        let temporary = NamedTempFile::new_in(parent)
            .with_context(|| format!("无法在 {} 创建临时输出", parent.display()))?;
        if !path_matches_open_file(temporary.path(), temporary.as_file(), false) {
            bail!("输出临时文件路径与已打开句柄不一致");
        }
        secure_file(temporary.path())?;
        if !path_matches_open_file(temporary.path(), temporary.as_file(), false) {
            bail!("输出临时文件在设置权限期间被替换");
        }
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
        if !path_matches_open_file(self.temporary.path(), self.temporary.as_file(), false) {
            bail!("输出临时文件在提交前被替换");
        }
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

#[cfg(unix)]
fn validate_output_parent_security(path: &Path) -> Result<()> {
    use std::os::unix::fs::MetadataExt;

    unsafe extern "C" {
        fn geteuid() -> u32;
    }

    let directory = open_directory_handle(path)
        .with_context(|| format!("无法打开输出父目录句柄 {}", path.display()))?;
    if !path_matches_open_file(path, &directory, true) {
        bail!("输出父目录路径与句柄 identity 不一致：{}", path.display());
    }
    let metadata = directory
        .metadata()
        .with_context(|| format!("无法读取输出父目录句柄 {}", path.display()))?;
    let effective_uid = unsafe { geteuid() };
    if metadata.uid() != effective_uid {
        bail!("输出父目录必须由当前用户拥有：{}", path.display());
    }
    if metadata.mode() & 0o022 != 0 {
        bail!("输出父目录不能允许组或其他用户写入：{}", path.display());
    }
    #[cfg(target_os = "macos")]
    if macos_handle_acl_grants_write(&directory)? {
        bail!("输出父目录的扩展 ACL 不能授予写权限：{}", path.display());
    }
    if let Some(namespace) = path.parent() {
        let namespace_handle = open_directory_handle(namespace)
            .with_context(|| format!("无法打开输出父目录命名空间 {}", namespace.display()))?;
        if !path_matches_open_file(namespace, &namespace_handle, true) {
            bail!(
                "输出父目录命名空间 identity 不一致：{}",
                namespace.display()
            );
        }
        let namespace_metadata = namespace_handle
            .metadata()
            .with_context(|| format!("无法读取输出父目录命名空间 {}", namespace.display()))?;
        let writable_by_others = namespace_metadata.mode() & 0o022 != 0;
        let sticky = namespace_metadata.mode() & 0o1000 != 0;
        if writable_by_others && !sticky {
            bail!(
                "输出父目录的上级不能允许其他用户无 sticky 保护地改名：{}",
                namespace.display()
            );
        }
        #[cfg(target_os = "macos")]
        if macos_handle_acl_grants_write(&namespace_handle)? {
            bail!(
                "输出父目录上级的扩展 ACL 不能授予写权限：{}",
                namespace.display()
            );
        }
    }
    Ok(())
}

#[cfg(not(unix))]
fn validate_output_parent_security(path: &Path) -> Result<()> {
    crate::security::validate_private_directory(path)
}

pub fn render_transcript(
    input: &Path,
    config: &Config,
    info: &AudioInfo,
    parts: &[TranscriptPart],
    mode: TranscriptMode,
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
    let overlay_model_requested = match mode {
        TranscriptMode::Quality => config.effective_quality_review_model(),
        TranscriptMode::Raw => config.model.as_str(),
    };

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
    let stt_completions = completions
        .iter()
        .copied()
        .filter(|completion| completion.origin == CompletionOrigin::Stt)
        .collect::<Vec<_>>();
    let chat_completions = completions
        .iter()
        .copied()
        .filter(|completion| completion.origin == CompletionOrigin::Chat)
        .collect::<Vec<_>>();
    let stt_models_reported = stt_completions
        .iter()
        .filter(|completion| completion.model_reported_by_api)
        .map(|completion| completion.model.as_str())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>()
        .join(", ");
    let stt_providers_reported = stt_completions
        .iter()
        .filter(|completion| completion.provider_reported_by_api)
        .map(|completion| completion.provider.as_str())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>()
        .join(", ");
    let chat_models_reported = chat_completions
        .iter()
        .filter(|completion| completion.model_reported_by_api)
        .map(|completion| completion.model.as_str())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>()
        .join(", ");
    let chat_providers_reported = chat_completions
        .iter()
        .filter(|completion| completion.provider_reported_by_api)
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
    let visible_output_tokens = completions
        .iter()
        .map(|completion| completion.visible_output_tokens())
        .sum::<u64>();
    let cost = completions
        .iter()
        .map(|completion| completion.cost)
        .sum::<f64>();
    let mut responses_by_model = BTreeMap::<String, u64>::new();
    let mut cost_by_model = BTreeMap::<String, f64>::new();
    for completion in completions
        .iter()
        .filter(|completion| completion.model_reported_by_api)
    {
        *responses_by_model
            .entry(completion.model.clone())
            .or_default() += 1;
        *cost_by_model.entry(completion.model.clone()).or_default() += completion.cost;
    }
    let responses_without_model_report = completions
        .iter()
        .filter(|completion| !completion.model_reported_by_api)
        .count();
    let cost_without_model_report = completions
        .iter()
        .filter(|completion| !completion.model_reported_by_api)
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
    let speaker_label_assignment_status = if unresolved_speaker_present {
        "unknown_labels_present"
    } else {
        "no_unknown_labels_present"
    };
    let acoustic_coverage_warning_present = parts.iter().any(|part| part.acoustic_coverage_warning);
    let quality_review_segments = parts.iter().filter(|part| part.quality_reviewed).count();
    let quality_review_advisory_present = parts.iter().any(|part| part.quality_review_advisory);
    let quality_cleanup_turns = parts
        .iter()
        .map(|part| part.quality_cleanup_turns)
        .sum::<usize>();
    let quality_cleanup_reverted_present = parts.iter().any(|part| {
        part.quality_residual_advisory_codes
            .iter()
            .any(|code| code == "quality_cleanup_reverted")
    });
    let quality_trigger_codes = parts
        .iter()
        .flat_map(|part| part.quality_trigger_codes.iter().cloned())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let quality_residual_advisory_codes = parts
        .iter()
        .flat_map(|part| part.quality_residual_advisory_codes.iter().cloned())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let asr_crosscheck_signal_codes = quality_trigger_codes
        .iter()
        .filter(|code| code.starts_with("asr_crosscheck_"))
        .cloned()
        .collect::<Vec<_>>();
    let legacy_quality_review_trigger_codes = quality_trigger_codes
        .iter()
        .filter(|code| {
            !code.starts_with("asr_crosscheck_") && !code.starts_with("quality_cleanup_")
        })
        .cloned()
        .collect::<Vec<_>>();
    let quality_cleanup_codes = quality_trigger_codes
        .iter()
        .filter(|code| {
            code.starts_with("quality_cleanup_") && !code.starts_with("quality_cleanup_signal_")
        })
        .cloned()
        .collect::<Vec<_>>();
    let quality_cleanup_signal_codes = quality_trigger_codes
        .iter()
        .filter(|code| code.starts_with("quality_cleanup_signal_"))
        .cloned()
        .collect::<Vec<_>>();
    let asr_crosscheck_advisory_codes = quality_residual_advisory_codes
        .iter()
        .filter(|code| code.starts_with("asr_crosscheck_"))
        .cloned()
        .collect::<Vec<_>>();
    let quality_cleanup_advisory_codes = quality_residual_advisory_codes
        .iter()
        .filter(|code| code.starts_with("quality_cleanup_"))
        .cloned()
        .collect::<Vec<_>>();
    let quality_content_filter_fallback_present = quality_residual_advisory_codes
        .iter()
        .any(|code| code == "quality_review_content_filter");
    let all_quality_parts_have_exact_asr_consensus = mode == TranscriptMode::Quality
        && parts.iter().all(|part| {
            part.quality_trigger_codes
                .iter()
                .any(|code| code == "asr_crosscheck_exact_consensus_not_ground_truth")
        });
    let asr_crosscheck_skipped_segments = if mode == TranscriptMode::Quality {
        parts
            .iter()
            .filter(|part| {
                part.quality_trigger_codes
                    .iter()
                    .any(|code| code == "asr_crosscheck_skipped_cost_bounded")
            })
            .count()
    } else {
        0
    };
    let asr_crosscheck_sampled_segments = if mode == TranscriptMode::Quality {
        parts.len().saturating_sub(asr_crosscheck_skipped_segments)
    } else {
        0
    };
    let all_sampled_parts_have_exact_asr_consensus = mode == TranscriptMode::Quality
        && asr_crosscheck_sampled_segments > 0
        && parts.iter().all(|part| {
            part.quality_trigger_codes.iter().any(|code| {
                matches!(
                    code.as_str(),
                    "asr_crosscheck_exact_consensus_not_ground_truth"
                        | "asr_crosscheck_skipped_cost_bounded"
                )
            })
        });
    let asr_crosscheck_disagreement_present = quality_residual_advisory_codes
        .iter()
        .any(|code| code == "asr_crosscheck_disagreement");
    let asr_crosscheck_unavailable_present = quality_residual_advisory_codes
        .iter()
        .any(|code| code == "asr_crosscheck_unavailable");
    let asr_crosscheck_status = match mode {
        TranscriptMode::Raw => "not_applicable",
        TranscriptMode::Quality if asr_crosscheck_disagreement_present => {
            "disagreement_requires_review"
        }
        TranscriptMode::Quality if asr_crosscheck_unavailable_present => {
            "unavailable_requires_review"
        }
        TranscriptMode::Quality if all_quality_parts_have_exact_asr_consensus => {
            "exact_consensus_not_ground_truth"
        }
        TranscriptMode::Quality
            if asr_crosscheck_skipped_segments > 0
                && all_sampled_parts_have_exact_asr_consensus =>
        {
            "sampled_exact_consensus_not_ground_truth"
        }
        TranscriptMode::Quality => "unavailable_requires_review",
    };
    let asr_crosscheck_requires_review = matches!(
        asr_crosscheck_status,
        "disagreement_requires_review" | "unavailable_requires_review"
    );
    let quality_bootstrap_segments = parts
        .iter()
        .filter(|part| {
            part.quality_trigger_codes
                .iter()
                .any(|code| code == "quality_bootstrap")
        })
        .count();
    let quality_review_status = match mode {
        TranscriptMode::Raw => "not_applicable",
        TranscriptMode::Quality if quality_content_filter_fallback_present => {
            "filtered_fallback_requires_review"
        }
        TranscriptMode::Quality
            if quality_review_segments > 0 && quality_review_advisory_present =>
        {
            "completed_with_advisory"
        }
        TranscriptMode::Quality if quality_review_segments > 0 => "completed",
        TranscriptMode::Quality => "not_used_dedicated_stt_pipeline",
    };
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
        yaml_string(&config.asr_model)?
    ));
    markdown.push_str(&format!(
        "base_model_requested: {}\n",
        yaml_string(&config.asr_model)?
    ));
    markdown.push_str(&format!(
        "asr_model_requested: {}\n",
        yaml_string(&config.asr_model)?
    ));
    match mode {
        TranscriptMode::Quality => {
            markdown.push_str(&format!(
                "quality_asr_model: {}\n",
                yaml_string(config.effective_quality_asr_model())?
            ));
            markdown.push_str(&format!(
                "quality_asr_provider: {}\n",
                yaml_string(&config.quality_asr_provider)?
            ));
            markdown.push_str(&format!(
                "quality_asr_provider_configured: {}\n",
                yaml_string(&config.quality_asr_provider)?
            ));
            markdown.push_str(&format!(
                "quality_asr_provider_routing_mode: {}\n",
                yaml_string(stt_provider_routing_mode(&config.quality_asr_provider))?
            ));
            markdown.push_str(&format!(
                "quality_asr_provider_expected: {}\n",
                provider_expected_value(&config.quality_asr_provider)?
            ));
            markdown.push_str(&format!(
                "quality_asr_provider_privacy_mode: {}\n",
                yaml_string(stt_provider_privacy_mode(&config.quality_asr_provider))?
            ));
        }
        TranscriptMode::Raw => {
            markdown.push_str("quality_asr_model: null\n");
            markdown.push_str("quality_asr_provider: null\n");
            markdown.push_str("quality_asr_provider_configured: null\n");
            markdown.push_str("quality_asr_provider_routing_mode: null\n");
            markdown.push_str("quality_asr_provider_expected: null\n");
            markdown.push_str("quality_asr_provider_privacy_mode: null\n");
        }
    }
    markdown.push_str(&format!(
        "multimodal_overlay_model_configured: {}\n",
        yaml_string(overlay_model_requested)?
    ));
    markdown.push_str(&format!(
        "multimodal_overlay_used: {}\n",
        !chat_completions.is_empty()
    ));
    markdown.push_str(&format!(
        "quality_overlay_model_requested: {}\n",
        match mode {
            TranscriptMode::Quality => yaml_string(overlay_model_requested)?,
            TranscriptMode::Raw => "null".to_owned(),
        }
    ));
    markdown.push_str(&format!(
        "multimodal_overlay_provider_configured: {}\n",
        yaml_string(&config.provider)?
    ));
    markdown.push_str(&format!(
        "multimodal_overlay_provider_routing_mode: {}\n",
        yaml_string(multimodal_provider_routing_mode(config))?
    ));
    markdown.push_str(&format!(
        "multimodal_overlay_provider_expected: {}\n",
        provider_expected_value(&config.provider)?
    ));
    markdown.push_str(
        "chat_provider_report_semantics: \"api_display_vendor_not_exact_endpoint_tag_proof\"\n",
    );
    markdown.push_str(&format!(
        "multimodal_overlay_provider_privacy_mode: {}\n",
        yaml_string(multimodal_provider_privacy_mode(config))?
    ));
    if mode == TranscriptMode::Quality
        && (quality_review_segments > 0 || quality_content_filter_fallback_present)
    {
        markdown.push_str(&format!(
            "quality_review_model_requested: {}\n",
            yaml_string(config.effective_quality_review_model())?
        ));
    } else {
        markdown.push_str("quality_review_model_requested: null\n");
    }
    markdown.push_str(&format!(
        "asr_provider_configured: {}\n",
        yaml_string(&config.asr_provider)?
    ));
    markdown.push_str(&format!(
        "asr_provider_routing_mode: {}\n",
        yaml_string(stt_provider_routing_mode(&config.asr_provider))?
    ));
    markdown.push_str(&format!(
        "asr_provider_expected: {}\n",
        provider_expected_value(&config.asr_provider)?
    ));
    markdown.push_str(&format!(
        "provider_privacy_mode: {}\n",
        yaml_string(stt_provider_privacy_mode(&config.asr_provider))?
    ));
    markdown.push_str(&format!(
        "asr_provider_privacy_mode: {}\n",
        yaml_string(stt_provider_privacy_mode(&config.asr_provider))?
    ));
    markdown.push_str(&format!(
        "accounted_response_models: {}\n",
        yaml_string(&models)?
    ));
    markdown.push_str(&format!(
        "accounted_response_providers: {}\n",
        yaml_string(&providers)?
    ));
    markdown.push_str(
        "accounted_route_label_policy: \"api_reported_when_valid_else_configured_or_explicit_unreported_sentinel\"\n",
    );
    markdown.push_str(&format!(
        "stt_models_reported_by_api: {}\n",
        yaml_string(&stt_models_reported)?
    ));
    markdown.push_str(&format!(
        "stt_providers_reported_by_api: {}\n",
        yaml_string(&stt_providers_reported)?
    ));
    markdown.push_str(&format!(
        "stt_model_reported_for_all_accounted_responses: {}\n",
        !stt_completions.is_empty()
            && stt_completions
                .iter()
                .all(|completion| completion.model_reported_by_api)
    ));
    markdown.push_str(&format!(
        "stt_provider_reported_for_all_accounted_responses: {}\n",
        !stt_completions.is_empty()
            && stt_completions
                .iter()
                .all(|completion| completion.provider_reported_by_api)
    ));
    markdown.push_str(&format!(
        "chat_models_reported_by_api: {}\n",
        yaml_string(&chat_models_reported)?
    ));
    markdown.push_str(&format!(
        "chat_providers_reported_by_api: {}\n",
        yaml_string(&chat_providers_reported)?
    ));
    markdown.push_str(&format!(
        "chat_model_reported_for_all_accounted_responses: {}\n",
        !chat_completions.is_empty()
            && chat_completions
                .iter()
                .all(|completion| completion.model_reported_by_api)
    ));
    markdown.push_str(&format!(
        "chat_provider_reported_for_all_accounted_responses: {}\n",
        !chat_completions.is_empty()
            && chat_completions
                .iter()
                .all(|completion| completion.provider_reported_by_api)
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
    markdown.push_str(&format!("transcript_mode: \"{}\"\n", mode.as_str()));
    markdown.push_str(&format!(
        "transcript_editing: \"{}\"\n",
        mode.editing_policy()
    ));
    markdown.push_str(
        "chinese_script: \"zh-Hans_with_source_fact_spans_or_source_when_kana_present\"\n",
    );
    markdown
        .push_str("chinese_normalization: \"opencc-t2s-fact-span-protected-kana-preserving\"\n");
    markdown.push_str(
        "transcription_strategy: \"dedicated-stt-cost-bounded-crosscheck-frozen-text-turn-overlay-v2\"\n",
    );
    markdown.push_str(
        "text_authority: \"primary_dedicated_stt_canonical_content_frozen_presentation_rendered\"\n",
    );
    markdown.push_str(
        "alignment_text_policy: \"segmentation_and_labels_only_canonical_fact_characters_immutable\"\n",
    );
    markdown.push_str(
        "stage_b_text_policy: \"identity_mapping_only_canonical_fact_characters_immutable\"\n",
    );
    markdown.push_str("alignment_may_modify_canonical_fact_characters: false\n");
    markdown.push_str("stage_b_may_modify_canonical_fact_characters: false\n");
    markdown.push_str("primary_asr_provider_bytes_restored_before_markdown_render: false\n");
    markdown.push_str("primary_asr_display_projection_slices_restored: true\n");
    markdown.push_str(&format!(
        "markdown_presentation_policy: \"{}\"\n",
        match mode {
            TranscriptMode::Quality => {
                "host_protected_readability_cleanup_speaker_labels_and_markdown_escape"
            }
            TranscriptMode::Raw => {
                "host_display_projection_speaker_labels_and_markdown_escape_no_quality_cleanup"
            }
        }
    ));
    markdown.push_str(&format!(
        "root_target_seconds: {}\n",
        config.effective_asr_chunk_seconds()
    ));
    markdown.push_str("transcript_accuracy_verification: \"not_measured\"\n");
    markdown.push_str("transcript_accuracy_basis: \"no_ground_truth_comparison_performed\"\n");
    markdown.push_str(&format!(
        "asr_crosscheck_status: \"{asr_crosscheck_status}\"\n"
    ));
    markdown.push_str(&format!(
        "asr_crosscheck_coverage: \"{}\"\n",
        if mode == TranscriptMode::Raw {
            "not_applicable"
        } else if asr_crosscheck_skipped_segments == 0 {
            "all_root_targets_checked"
        } else {
            "first_and_every_fifth_root_target_cost_bounded"
        }
    ));
    markdown.push_str(&format!(
        "asr_crosscheck_sampled_segments: {asr_crosscheck_sampled_segments}\n"
    ));
    markdown.push_str(&format!(
        "asr_crosscheck_skipped_segments: {asr_crosscheck_skipped_segments}\n"
    ));
    markdown.push_str(&format!(
        "asr_crosscheck_signals: {}\n",
        serde_json::to_string(&asr_crosscheck_signal_codes)
            .context("无法序列化 ASR 交叉检查信号")?
    ));
    markdown.push_str(&format!(
        "asr_crosscheck_advisories: {}\n",
        serde_json::to_string(&asr_crosscheck_advisory_codes)
            .context("无法序列化 ASR 交叉检查告警")?
    ));
    markdown.push_str(
        "quality_gate_scope: \"legacy_surface_gate_not_used_by_dedicated_stt_pipeline\"\n",
    );
    markdown.push_str("surface_quality_gate_version: \"rust-quality-signals-v1\"\n");
    markdown.push_str("quality_gate_version: \"rust-quality-signals-v1\"\n");
    markdown.push_str(&format!(
        "quality_review_status: \"{quality_review_status}\"\n"
    ));
    let legacy_surface_quality_gate_status = match mode {
        TranscriptMode::Raw => "not_applicable",
        TranscriptMode::Quality => "not_used",
    };
    markdown.push_str(&format!(
        "surface_quality_gate_status: \"{legacy_surface_quality_gate_status}\"\n"
    ));
    // Compatibility alias retained for existing readers. ASR comparison codes
    // are reported separately and must never be presented as surface-gate data.
    markdown.push_str(&format!(
        "quality_gate_status: \"{legacy_surface_quality_gate_status}\"\n"
    ));
    markdown.push_str(&format!(
        "quality_review_segments: {quality_review_segments}\n"
    ));
    markdown.push_str(&format!(
        "quality_bootstrap_segments: {quality_bootstrap_segments}\n"
    ));
    markdown.push_str(&format!(
        "quality_host_cleanup_turns: {quality_cleanup_turns}\n"
    ));
    markdown.push_str(&format!(
        "quality_host_cleanup_status: \"{}\"\n",
        match mode {
            TranscriptMode::Raw => "not_applicable",
            TranscriptMode::Quality if quality_cleanup_reverted_present => {
                "completed_with_reverted_turns_requires_review"
            }
            TranscriptMode::Quality if quality_cleanup_turns > 0 => "applied",
            TranscriptMode::Quality => "no_allowlisted_edits_needed",
        }
    ));
    markdown.push_str(&format!(
        "quality_host_cleanup_operations: {}\n",
        serde_json::to_string(&quality_cleanup_codes).context("无法序列化主机清稿操作")?
    ));
    markdown.push_str(&format!(
        "quality_host_cleanup_signals: {}\n",
        serde_json::to_string(&quality_cleanup_signal_codes).context("无法序列化主机清稿信号")?
    ));
    markdown.push_str(&format!(
        "quality_host_cleanup_advisories: {}\n",
        serde_json::to_string(&quality_cleanup_advisory_codes).context("无法序列化主机清稿告警")?
    ));
    markdown.push_str(&format!(
        "quality_review_triggers: {}\n",
        serde_json::to_string(&legacy_quality_review_trigger_codes)
            .context("无法序列化质量复核触发原因")?
    ));
    let quality_residual_advisories = serde_json::to_string(&quality_residual_advisory_codes)
        .context("无法序列化质量复核残留告警")?;
    markdown.push_str("surface_quality_residual_advisories: []\n");
    // Broad v0.4 compatibility alias. Canonical ASR comparison consumers should
    // use `asr_crosscheck_advisories`; this is not a surface-gate result.
    markdown.push_str(&format!(
        "quality_residual_advisories: {quality_residual_advisories}\n"
    ));
    markdown.push_str(&format!(
        "final_transcript_segments_from_primary_asr: {}\n",
        parts.len()
    ));
    markdown.push_str(&format!(
        "final_transcript_segments_from_base: {}\n",
        parts.len().saturating_sub(quality_review_segments)
    ));
    markdown.push_str(&format!(
        "final_transcript_segments_from_review: {quality_review_segments}\n"
    ));
    markdown.push_str(&format!("segments: {}\n", parts.len()));
    markdown.push_str(&format!(
        "accounted_model_responses: {}\n",
        completions.len()
    ));
    markdown.push_str(&format!(
        "speaker_turns: {}\n",
        parts.iter().map(|part| part.turn_count).sum::<usize>()
    ));
    markdown.push_str("speaker_tracking: \"openrouter-per-turn-reference-harness-v3\"\n");
    markdown.push_str("speaker_identity_scope: \"whole_job\"\n");
    markdown.push_str("speaker_identity_guarantee: \"best_effort\"\n");
    markdown.push_str("speaker_identity_accuracy: \"not_measured\"\n");
    markdown.push_str("speaker_id_assignment: \"host_managed\"\n");
    markdown.push_str("speaker_names_inferred: false\n");
    markdown.push_str(&format!(
        "speaker_label_assignment_status: \"{speaker_label_assignment_status}\"\n"
    ));
    markdown.push_str(&format!(
        "speaker_all_turns_have_non_unknown_labels: {}\n",
        !unresolved_speaker_present
    ));
    markdown.push_str("speaker_alignment_status: \"not_verified\"\n");
    markdown.push_str(&format!(
        "acoustic_coverage_status: \"{}\"\n",
        if acoustic_coverage_warning_present {
            "ffmpeg_energy_advisory_warning"
        } else {
            "ffmpeg_energy_advisory_passed"
        }
    ));
    markdown.push_str(&format!(
        "speaker_boundary_context_seconds: {}\n",
        config.overlap_seconds
    ));
    markdown.push_str(&format!(
        "unresolved_speaker_present: {unresolved_speaker_present}\n"
    ));
    markdown.push_str(&format!(
        "assigned_speaker_id_count: {}\n",
        speaker_ids.len()
    ));
    markdown.push_str(&format!(
        "speaker_ids: {}\n",
        serde_json::to_string(&speaker_ids).context("无法序列化说话人列表")?
    ));
    markdown.push_str("stt_token_counts_may_be_unavailable: true\n");
    markdown.push_str(&format!(
        "usage_reported_for_all_accounted_responses: {usage_reported_for_all_requests}\n"
    ));
    markdown.push_str(&format!(
        "reasoning_tokens_reported_for_all_accounted_responses: {reasoning_tokens_reported_for_all_requests}\n"
    ));
    markdown.push_str(&format!("reported_prompt_tokens: {prompt_tokens}\n"));
    markdown.push_str(&format!(
        "reported_completion_tokens: {completion_tokens}\n"
    ));
    markdown.push_str(&format!("reported_reasoning_tokens: {reasoning_tokens}\n"));
    markdown.push_str(&format!(
        "reported_visible_output_tokens: {visible_output_tokens}\n"
    ));
    markdown.push_str(&format!("reported_accounted_cost_usd: {cost:.9}\n"));
    markdown.push_str(
        "reported_model_bucket_policy: \"only_api_reported_model_values_no_requested_fallback\"\n",
    );
    markdown.push_str(&format!(
        "reported_responses_by_model: {}\n",
        serde_json::to_string(&responses_by_model).context("无法序列化按模型响应数")?
    ));
    markdown.push_str(&format!(
        "reported_cost_usd_by_model: {}\n",
        serde_json::to_string(&cost_by_model).context("无法序列化按模型费用")?
    ));
    markdown.push_str(&format!(
        "responses_without_model_reported_by_api: {responses_without_model_report}\n"
    ));
    markdown.push_str(&format!(
        "reported_cost_usd_without_model_reported_by_api: {cost_without_model_report:.9}\n"
    ));
    markdown.push_str("---\n\n");
    let title_qualifier = if mode == TranscriptMode::Quality
        && (quality_review_advisory_present || asr_crosscheck_requires_review)
    {
        "（准确率未验证，含需复核片段）"
    } else {
        "（准确率未验证）"
    };
    markdown.push_str(&format!(
        "# {} {}{}\n\n",
        escape_markdown_text(title),
        mode.title(),
        title_qualifier,
    ));

    for part in parts {
        let review_label = if part.quality_residual_advisory_codes.is_empty() {
            String::new()
        } else {
            format!(
                "（需复核：{}）",
                part.quality_residual_advisory_codes.join(",")
            )
        };
        markdown.push_str(&format!(
            "## {}–{}{}\n\n",
            format_timestamp(part.start_ms),
            format_timestamp(part.end_ms),
            review_label,
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
    rejected_accounting: &[Completion],
) -> Result<String> {
    if completion.origin != CompletionOrigin::Chat {
        bail!("OCR 结果必须来自 Chat Image 响应");
    }
    if rejected_accounting
        .iter()
        .any(|rejected| rejected.origin != CompletionOrigin::Chat || !rejected.text.is_empty())
    {
        bail!("OCR rejected accounting 必须是空正文的 Chat 响应占位");
    }
    let file_name = input
        .file_name()
        .and_then(|name| name.to_str())
        .context("源文件名不是有效 UTF-8")?;
    let title = input
        .file_stem()
        .and_then(|name| name.to_str())
        .context("源文件缺少主文件名")?;
    let accounted = std::iter::once(completion)
        .chain(rejected_accounting.iter())
        .collect::<Vec<_>>();
    let prompt_tokens = accounted
        .iter()
        .map(|completion| completion.prompt_tokens)
        .sum::<u64>();
    let completion_tokens = accounted
        .iter()
        .map(|completion| completion.completion_tokens)
        .sum::<u64>();
    let reasoning_tokens = accounted
        .iter()
        .map(|completion| completion.reasoning_tokens)
        .sum::<u64>();
    let visible_output_tokens = accounted
        .iter()
        .map(|completion| completion.visible_output_tokens())
        .sum::<u64>();
    let rejected_cost = rejected_accounting
        .iter()
        .map(|completion| completion.cost)
        .sum::<f64>();
    let accounted_cost = completion.cost + rejected_cost;
    let usage_reported_for_all_accounted_responses =
        accounted.iter().all(|completion| completion.usage_reported);
    let reasoning_tokens_reported_for_all_accounted_responses = accounted
        .iter()
        .all(|completion| completion.reasoning_tokens_reported);
    let mut responses_by_model = BTreeMap::<String, u64>::new();
    let mut cost_by_model = BTreeMap::<String, f64>::new();
    let mut responses_without_model_report = 0_u64;
    let mut cost_without_model_report = 0_f64;
    for accounted_completion in &accounted {
        if accounted_completion.model_reported_by_api {
            *responses_by_model
                .entry(accounted_completion.model.clone())
                .or_default() += 1;
            *cost_by_model
                .entry(accounted_completion.model.clone())
                .or_default() += accounted_completion.cost;
        } else {
            responses_without_model_report = responses_without_model_report.saturating_add(1);
            cost_without_model_report += accounted_completion.cost;
        }
    }
    Ok(format!(
        "---\nsource: {}\nmodel_configured: {}\nprovider_configured: {}\nprovider_privacy_mode: {}\nmodel_reported_by_api: {}\nprovider_reported_by_api: {}\nsource_codec: {}\nsource_container: {}\nsource_width: {}\nsource_height: {}\naccounted_model_responses: {}\nrejected_accounted_responses: {}\nusage_reported: {}\nreasoning_tokens_reported: {}\nusage_reported_for_all_accounted_responses: {}\nreasoning_tokens_reported_for_all_accounted_responses: {}\nreported_prompt_tokens: {}\nreported_completion_tokens: {}\nreported_reasoning_tokens: {}\nreported_visible_output_tokens: {}\nreported_accepted_cost_usd: {:.9}\nreported_rejected_cost_usd: {:.9}\nreported_accounted_cost_usd: {:.9}\nreported_model_bucket_policy: \"only_api_reported_model_values_no_requested_fallback\"\nreported_responses_by_model: {}\nreported_cost_usd_by_model: {}\nresponses_without_model_reported_by_api: {}\nreported_cost_usd_without_model_reported_by_api: {:.9}\n---\n\n# {} OCR\n\n{}\n",
        yaml_string(file_name)?,
        yaml_string(&config.model)?,
        yaml_string(&config.provider)?,
        yaml_string(if config.uses_any_provider() {
            "any_explicit_privacy_downgrade"
        } else {
            "fixed_zdr_data_collection_denied"
        })?,
        if completion.model_reported_by_api {
            yaml_string(&completion.model)?
        } else {
            "null".to_owned()
        },
        if completion.provider_reported_by_api {
            yaml_string(&completion.provider)?
        } else {
            "null".to_owned()
        },
        yaml_string(&info.codec)?,
        yaml_string(&info.container)?,
        info.width,
        info.height,
        accounted.len(),
        rejected_accounting.len(),
        completion.usage_reported,
        completion.reasoning_tokens_reported,
        usage_reported_for_all_accounted_responses,
        reasoning_tokens_reported_for_all_accounted_responses,
        prompt_tokens,
        completion_tokens,
        reasoning_tokens,
        visible_output_tokens,
        completion.cost,
        rejected_cost,
        accounted_cost,
        serde_json::to_string(&responses_by_model).context("无法序列化 OCR 按模型响应数")?,
        serde_json::to_string(&cost_by_model).context("无法序列化 OCR 按模型费用")?,
        responses_without_model_report,
        cost_without_model_report,
        escape_markdown_text(title),
        escape_markdown_text(completion.text.trim()),
    ))
}

pub fn ocr_output_path(input: &Path) -> Result<PathBuf> {
    let parent = output_parent(input);
    let canonical_parent = fs::canonicalize(parent)
        .with_context(|| format!("无法解析 OCR 输入目录 {}", parent.display()))?;
    let stem = input
        .file_stem()
        .and_then(|name| name.to_str())
        .context("OCR 源文件缺少有效主文件名")?;
    Ok(canonical_parent.join(format!("{stem}.ocr.md")))
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

fn stt_provider_privacy_mode(provider: &str) -> &'static str {
    if provider.eq_ignore_ascii_case("any") {
        "any_explicit_privacy_downgrade"
    } else {
        "catalog_unique_zdr_preflight_no_request_level_pin"
    }
}

fn stt_provider_routing_mode(provider: &str) -> &'static str {
    if provider.eq_ignore_ascii_case("any") {
        "automatic_unpinned"
    } else {
        "catalog_unique_fixed_expected_no_request_level_pin"
    }
}

fn provider_expected_value(provider: &str) -> Result<String> {
    if provider.eq_ignore_ascii_case("any") {
        Ok("null".to_owned())
    } else {
        yaml_string(provider)
    }
}

fn multimodal_provider_privacy_mode(config: &Config) -> &'static str {
    if config.uses_any_provider() {
        "any_explicit_privacy_downgrade"
    } else {
        "fixed_zdr_data_collection_denied"
    }
}

fn multimodal_provider_routing_mode(config: &Config) -> &'static str {
    if config.uses_any_provider() {
        "automatic_unpinned"
    } else {
        "fixed_provider_only"
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

const OUTPUT_LOCK_SHARDS: u64 = 4_096;
const FNV1A64_OFFSET_BASIS: u64 = 0xcbf29ce484222325;
const FNV1A64_PRIME: u64 = 0x100000001b3;

struct PreparedLockDirectory {
    path: PathBuf,
    state_root_parent: fs::File,
    state_root: fs::File,
    directory: fs::File,
}

struct CanonicalOutputIdentity {
    normalized_target: PathBuf,
    bytes: Vec<u8>,
    parent_path: PathBuf,
    parent: fs::File,
}

fn acquire_output_lock(path: &Path) -> Result<OutputLock> {
    let (lock_directory, secure_existing_state_root) = output_lock_location()?;
    #[cfg(not(test))]
    {
        let default_directory = default_output_lock_directory()?;
        if lock_directory != default_directory {
            // Every process first enters the immutable per-user namespace.
            // A custom state root adds a second lock but can never replace the
            // shared default lock, so two SPT_STATE_DIR values still contend.
            let mut default_lock =
                acquire_output_lock_in_with_policy(path, &default_directory, true, true)?;
            let mut custom_lock = acquire_output_lock_in_with_policy(
                path,
                &lock_directory,
                false,
                secure_existing_state_root,
            )?;
            default_lock.files.append(&mut custom_lock.files);
            default_lock
                ._directories
                .append(&mut custom_lock._directories);
            return Ok(default_lock);
        }
    }
    acquire_output_lock_in_with_policy(path, &lock_directory, true, secure_existing_state_root)
}

#[cfg(test)]
fn acquire_output_lock_in(
    path: &Path,
    lock_directory: &Path,
    include_legacy_lock: bool,
) -> Result<OutputLock> {
    acquire_output_lock_in_with_policy(path, lock_directory, include_legacy_lock, true)
}

fn acquire_output_lock_in_with_policy(
    path: &Path,
    lock_directory: &Path,
    include_legacy_lock: bool,
    secure_existing_state_root: bool,
) -> Result<OutputLock> {
    let identity = canonical_output_identity(path)?;
    let hash = fnv1a64(&identity.bytes);
    let prepared = prepare_output_lock_directory(lock_directory, secure_existing_state_root)?;
    let shard_path = prepared.path.join(shard_file_name(hash));
    let mut files = Vec::with_capacity(2);

    if include_legacy_lock
        && let Some(legacy) = acquire_existing_legacy_lock(&identity.normalized_target, path)?
    {
        files.push(legacy);
    }

    let shard = open_persistent_shard(&shard_path)?;
    try_lock_output_file(&shard, &shard_path, path, false)?;
    if !path_matches_open_file(&identity.parent_path, &identity.parent, true)
        || !path_matches_open_file(&prepared.path, &prepared.directory, true)
        || !path_matches_open_file(&shard_path, &shard, false)
    {
        bail!(
            "输出锁目录或 shard 在获取期间被替换：{}",
            shard_path.display()
        );
    }
    files.push(shard);

    Ok(OutputLock {
        files,
        _directories: vec![
            identity.parent,
            prepared.state_root_parent,
            prepared.state_root,
            prepared.directory,
        ],
        #[cfg(test)]
        shard_path,
    })
}

fn canonical_output_identity(path: &Path) -> Result<CanonicalOutputIdentity> {
    let normalized_target = normalized_output_target(path)?;
    let parent_path = normalized_target
        .parent()
        .context("输出路径缺少 canonical parent")?
        .to_owned();
    let file_name = normalized_target
        .file_name()
        .context("输出路径缺少文件名")?;
    let parent = open_directory_handle(&parent_path)
        .with_context(|| format!("无法打开输出目录句柄 {}", parent_path.display()))?;
    if !path_matches_open_file(&parent_path, &parent, true) {
        bail!(
            "输出目录路径与句柄 identity 不一致：{}",
            parent_path.display()
        );
    }
    let bytes = output_identity_bytes(&parent, file_name)?;
    if !path_matches_open_file(&parent_path, &parent, true) {
        bail!(
            "输出目录在 identity 计算期间被替换：{}",
            parent_path.display()
        );
    }
    Ok(CanonicalOutputIdentity {
        normalized_target,
        bytes,
        parent_path,
        parent,
    })
}

fn normalized_output_target(path: &Path) -> Result<PathBuf> {
    let parent = output_parent(path);
    let canonical_parent = fs::canonicalize(parent)
        .with_context(|| format!("无法解析输出目录 {}", parent.display()))?;
    let file_name = path.file_name().context("输出路径缺少文件名")?;
    Ok(canonical_parent.join(file_name))
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = FNV1A64_OFFSET_BASIS;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(FNV1A64_PRIME);
    }
    hash
}

fn shard_file_name(hash: u64) -> String {
    format!("shard-{:03x}.lock", hash & (OUTPUT_LOCK_SHARDS - 1))
}

fn output_lock_location() -> Result<(PathBuf, bool)> {
    #[cfg(test)]
    {
        static TEST_STATE_ROOT: OnceLock<PathBuf> = OnceLock::new();
        let state_root = TEST_STATE_ROOT.get_or_init(|| {
            tempfile::Builder::new()
                .prefix(&format!("spt-test-state-{}-", process::id()))
                .tempdir_in(env::temp_dir())
                .expect("无法创建隔离的 spt 测试状态目录")
                .keep()
        });
        Ok((state_root.join("output-locks"), true))
    }
    #[cfg(not(test))]
    {
        if let Some(path) = env::var_os("SPT_STATE_DIR") {
            if path.is_empty() {
                bail!("SPT_STATE_DIR 不能为空");
            }
            #[cfg(windows)]
            bail!("Windows 当前不允许覆盖 SPT_STATE_DIR；请使用默认私有状态目录");
            #[cfg(not(windows))]
            return Ok((lock_directory_for_state_root(PathBuf::from(path))?, false));
        }
        Ok((default_output_lock_directory()?, true))
    }
}

#[cfg(not(test))]
fn default_output_lock_directory() -> Result<PathBuf> {
    let home = dirs::home_dir().context("无法确定 spt 状态目录；可设置 SPT_STATE_DIR")?;
    lock_directory_for_state_root(home.join(".spt"))
}

#[cfg(test)]
fn output_lock_directory() -> Result<PathBuf> {
    output_lock_location().map(|(directory, _)| directory)
}

fn lock_directory_for_state_root(root: PathBuf) -> Result<PathBuf> {
    if !root.is_absolute() {
        bail!("SPT_STATE_DIR 必须是绝对路径");
    }
    if root.parent().is_none() {
        bail!("SPT_STATE_DIR 不能是文件系统根目录");
    }
    Ok(root.join("output-locks"))
}

fn prepare_output_lock_directory(
    lock_directory: &Path,
    secure_existing_state_root: bool,
) -> Result<PreparedLockDirectory> {
    if !lock_directory.is_absolute() {
        bail!("输出锁目录必须是绝对路径：{}", lock_directory.display());
    }
    let state_root_path = lock_directory
        .parent()
        .context("输出锁目录缺少状态根目录")?;
    let state_root_parent_path = state_root_path
        .parent()
        .context("spt 状态根目录缺少父目录")?;
    let state_root_parent = open_directory_handle(state_root_parent_path)
        .with_context(|| format!("无法打开状态根父目录 {}", state_root_parent_path.display()))?;
    if !path_matches_open_file(state_root_parent_path, &state_root_parent, true) {
        bail!(
            "状态根父目录路径与句柄 identity 不一致：{}",
            state_root_parent_path.display()
        );
    }
    if !secure_existing_state_root {
        validate_custom_state_root_parent(&state_root_parent, state_root_parent_path)?;
    }
    let state_root = ensure_private_real_directory(state_root_path, secure_existing_state_root)?;
    if !secure_existing_state_root {
        validate_existing_custom_state_root(&state_root, state_root_path)?;
    }
    let directory = ensure_private_real_directory(lock_directory, true)?;
    if !path_matches_open_file(state_root_path, &state_root, true)
        || !path_matches_open_file(state_root_parent_path, &state_root_parent, true)
        || !path_matches_open_file(lock_directory, &directory, true)
    {
        bail!("spt 状态目录在准备期间被替换");
    }
    let canonical = fs::canonicalize(lock_directory)
        .with_context(|| format!("无法解析输出锁目录 {}", lock_directory.display()))?;
    if !path_matches_open_file(&canonical, &directory, true) {
        bail!(
            "输出锁目录 canonical identity 不一致：{}",
            canonical.display()
        );
    }
    Ok(PreparedLockDirectory {
        path: canonical,
        state_root_parent,
        state_root,
        directory,
    })
}

fn ensure_private_real_directory(path: &Path, secure_existing: bool) -> Result<fs::File> {
    let created = match fs::symlink_metadata(path) {
        Ok(metadata) => {
            validate_real_directory(path, &metadata)?;
            false
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            let parent = path.parent().context("spt 状态路径缺少父目录")?;
            let parent_metadata = fs::symlink_metadata(parent)
                .with_context(|| format!("无法读取 spt 状态父目录 {}", parent.display()))?;
            validate_real_directory(parent, &parent_metadata)?;
            match fs::create_dir(path) {
                Ok(()) => true,
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => false,
                Err(error) => {
                    return Err(error)
                        .with_context(|| format!("无法创建 spt 状态目录 {}", path.display()));
                }
            }
        }
        Err(error) => {
            return Err(error).with_context(|| format!("无法检查 spt 状态目录 {}", path.display()));
        }
    };

    let metadata = fs::symlink_metadata(path)
        .with_context(|| format!("无法复核 spt 状态目录 {}", path.display()))?;
    validate_real_directory(path, &metadata)?;
    let directory = open_directory_handle(path)
        .with_context(|| format!("无法打开 spt 状态目录 {}", path.display()))?;
    if !directory.metadata().is_ok_and(|metadata| metadata.is_dir())
        || !path_matches_open_file(path, &directory, true)
    {
        bail!("spt 状态目录句柄与路径不一致：{}", path.display());
    }
    if created || secure_existing {
        set_private_permissions(&directory, 0o700)
            .with_context(|| format!("无法设置私有状态目录权限 {}", path.display()))?;
        #[cfg(windows)]
        crate::security::secure_directory(path)
            .with_context(|| format!("无法设置 Windows 私有状态目录 DACL {}", path.display()))?;
    }
    if !path_matches_open_file(path, &directory, true) {
        bail!("spt 状态目录在设置权限期间被替换：{}", path.display());
    }
    Ok(directory)
}

fn validate_real_directory(path: &Path, metadata: &fs::Metadata) -> Result<()> {
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        bail!("spt 状态路径不是实体目录：{}", path.display());
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;

        const FILE_ATTRIBUTE_REPARSE_POINT: u32 = 0x00000400;
        if metadata.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0 {
            bail!(
                "spt 状态路径不能是 Windows reparse point：{}",
                path.display()
            );
        }
    }
    Ok(())
}

#[cfg(unix)]
fn validate_custom_state_root_parent(file: &fs::File, path: &Path) -> Result<()> {
    use std::os::unix::fs::MetadataExt;

    let metadata = file
        .metadata()
        .with_context(|| format!("无法读取自定义状态根父目录 {}", path.display()))?;
    let writable_by_others = metadata.mode() & 0o022 != 0;
    let sticky = metadata.mode() & 0o1000 != 0;
    if writable_by_others && !sticky {
        bail!(
            "SPT_STATE_DIR 的父目录不能允许其他用户无 sticky 保护地改名：{}",
            path.display()
        );
    }
    #[cfg(target_os = "macos")]
    if macos_handle_acl_grants_write(file)? {
        bail!(
            "SPT_STATE_DIR 父目录的扩展 ACL 不能授予写权限：{}",
            path.display()
        );
    }
    Ok(())
}

#[cfg(not(unix))]
fn validate_custom_state_root_parent(_file: &fs::File, _path: &Path) -> Result<()> {
    Ok(())
}

#[cfg(unix)]
fn validate_existing_custom_state_root(file: &fs::File, path: &Path) -> Result<()> {
    use std::os::unix::fs::MetadataExt;

    unsafe extern "C" {
        fn geteuid() -> u32;
    }

    let metadata = file
        .metadata()
        .with_context(|| format!("无法读取自定义状态根目录 {}", path.display()))?;
    let effective_uid = unsafe { geteuid() };
    if metadata.uid() != effective_uid {
        bail!("SPT_STATE_DIR 必须由当前用户拥有：{}", path.display());
    }
    if metadata.mode() & 0o022 != 0 {
        bail!("SPT_STATE_DIR 不能允许组或其他用户写入：{}", path.display());
    }
    #[cfg(target_os = "macos")]
    if macos_handle_acl_grants_write(file)? {
        bail!(
            "SPT_STATE_DIR 的扩展 ACL 不能授予写权限：{}",
            path.display()
        );
    }
    Ok(())
}

#[cfg(not(unix))]
fn validate_existing_custom_state_root(_file: &fs::File, _path: &Path) -> Result<()> {
    Ok(())
}

#[cfg(target_os = "macos")]
fn macos_handle_acl_grants_write(file: &fs::File) -> Result<bool> {
    use std::ffi::c_void;
    use std::os::unix::io::AsRawFd;
    use std::ptr;

    const ACL_TYPE_EXTENDED: i32 = 0x0000_0100;
    const ACL_FIRST_ENTRY: i32 = 0;
    const ACL_NEXT_ENTRY: i32 = -1;
    const ACL_EXTENDED_ALLOW: i32 = 1;
    const ACL_EXTENDED_DENY: i32 = 2;
    const ACL_MAX_ENTRIES: usize = 128;
    const ACL_WRITE_DATA: u64 = 1 << 2;
    const ACL_DELETE: u64 = 1 << 4;
    const ACL_APPEND_DATA: u64 = 1 << 5;
    const ACL_DELETE_CHILD: u64 = 1 << 6;
    const ACL_WRITE_ATTRIBUTES: u64 = 1 << 8;
    const ACL_WRITE_EXTATTRIBUTES: u64 = 1 << 10;
    const ACL_WRITE_SECURITY: u64 = 1 << 12;
    const ACL_CHANGE_OWNER: u64 = 1 << 13;
    const WRITE_PERMISSIONS: u64 = ACL_WRITE_DATA
        | ACL_DELETE
        | ACL_APPEND_DATA
        | ACL_DELETE_CHILD
        | ACL_WRITE_ATTRIBUTES
        | ACL_WRITE_EXTATTRIBUTES
        | ACL_WRITE_SECURITY
        | ACL_CHANGE_OWNER;

    unsafe extern "C" {
        fn acl_get_fd_np(file_descriptor: i32, acl_type: i32) -> *mut c_void;
        fn acl_get_entry(acl: *mut c_void, entry_id: i32, entry: *mut *mut c_void) -> i32;
        fn acl_get_tag_type(entry: *mut c_void, tag_type: *mut i32) -> i32;
        fn acl_get_permset_mask_np(entry: *mut c_void, mask: *mut u64) -> i32;
        fn acl_free(value: *mut c_void) -> i32;
    }

    let acl = unsafe { acl_get_fd_np(file.as_raw_fd(), ACL_TYPE_EXTENDED) };
    if acl.is_null() {
        let error = std::io::Error::last_os_error();
        if error.kind() == std::io::ErrorKind::NotFound {
            return Ok(false);
        }
        return Err(error).context("无法读取自定义状态根目录 ACL");
    }

    let inspection = (|| -> Result<bool> {
        for index in 0..=ACL_MAX_ENTRIES {
            let mut entry = ptr::null_mut();
            let entry_id = if index == 0 {
                ACL_FIRST_ENTRY
            } else {
                ACL_NEXT_ENTRY
            };
            let result = unsafe { acl_get_entry(acl, entry_id, &mut entry) };
            if result < 0 {
                let error = std::io::Error::last_os_error();
                if error.raw_os_error() == Some(libc::EINVAL) {
                    return Ok(false);
                }
                return Err(error).context("无法遍历 macOS 扩展 ACL");
            }
            if index == ACL_MAX_ENTRIES {
                bail!("macOS 扩展 ACL 条目超过系统安全上限");
            }
            if entry.is_null() {
                bail!("macOS 扩展 ACL 返回了空条目");
            }

            let mut tag_type = 0_i32;
            if unsafe { acl_get_tag_type(entry, &mut tag_type) } < 0 {
                return Err(std::io::Error::last_os_error())
                    .context("无法读取 macOS 扩展 ACL 条目类型");
            }
            match tag_type {
                ACL_EXTENDED_DENY => continue,
                ACL_EXTENDED_ALLOW => {
                    let mut permissions = 0_u64;
                    if unsafe { acl_get_permset_mask_np(entry, &mut permissions) } < 0 {
                        return Err(std::io::Error::last_os_error())
                            .context("无法读取 macOS 扩展 ACL 权限");
                    }
                    if permissions & WRITE_PERMISSIONS != 0 {
                        return Ok(true);
                    }
                }
                _ => bail!("macOS 扩展 ACL 包含无法识别的条目类型"),
            }
        }
        unreachable!("有界 ACL 遍历必须返回")
    })();
    let _ = unsafe { acl_free(acl) };
    inspection
}

fn open_persistent_shard(path: &Path) -> Result<fs::File> {
    reject_existing_symlink_or_non_file(path)?;
    let file = open_lock_file_handle(path, true)
        .with_context(|| format!("无法打开输出锁 shard {}", path.display()))?;
    let metadata = file
        .metadata()
        .with_context(|| format!("无法读取输出锁 shard 句柄 {}", path.display()))?;
    if !metadata.is_file() || metadata.len() != 0 || !path_matches_open_file(path, &file, false) {
        bail!("输出锁 shard 不是匹配的零字节普通文件：{}", path.display());
    }
    set_private_permissions(&file, 0o600)
        .with_context(|| format!("无法设置输出锁 shard 权限 {}", path.display()))?;
    #[cfg(windows)]
    crate::security::secure_file(path)
        .with_context(|| format!("无法设置 Windows 输出锁 shard DACL {}", path.display()))?;
    if !path_matches_open_file(path, &file, false) {
        bail!("输出锁 shard 在设置权限期间被替换：{}", path.display());
    }
    Ok(file)
}

fn reject_existing_symlink_or_non_file(path: &Path) -> Result<()> {
    match fs::symlink_metadata(path) {
        Ok(metadata) => {
            if metadata.file_type().is_symlink() || !metadata.is_file() {
                bail!("输出锁 shard 路径不是普通文件：{}", path.display());
            }
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => {
            return Err(error).with_context(|| format!("无法检查输出锁 shard {}", path.display()));
        }
    }
    Ok(())
}

fn try_lock_output_file(
    file: &fs::File,
    lock_path: &Path,
    output_path: &Path,
    legacy: bool,
) -> Result<()> {
    match file.try_lock_exclusive() {
        Ok(()) => Ok(()),
        Err(error) if lock_would_block(&error) => {
            if legacy {
                bail!(
                    "旧版 spt 任务仍在处理同一输出：{}；请等待旧任务结束",
                    output_path.display()
                );
            }
            bail!(
                "另一个 spt 任务正在处理同一输出或共享锁分片：{}",
                output_path.display()
            )
        }
        Err(error) => Err(error).with_context(|| format!("无法获取输出锁 {}", lock_path.display())),
    }
}

fn lock_would_block(error: &std::io::Error) -> bool {
    if error.kind() == std::io::ErrorKind::WouldBlock {
        return true;
    }
    #[cfg(windows)]
    {
        return error.raw_os_error() == Some(33);
    }
    #[cfg(not(windows))]
    false
}

#[cfg(not(test))]
fn acquire_existing_legacy_lock(
    normalized_target: &Path,
    output_path: &Path,
) -> Result<Option<fs::File>> {
    let Some(cache_root) =
        dirs::cache_dir().or_else(|| dirs::home_dir().map(|home| home.join(".cache")))
    else {
        return Ok(None);
    };
    let legacy_directory = cache_root.join("spt/output-locks");
    acquire_existing_legacy_lock_from_directory(normalized_target, output_path, &legacy_directory)
}

fn acquire_existing_legacy_lock_from_directory(
    normalized_target: &Path,
    output_path: &Path,
    legacy_directory: &Path,
) -> Result<Option<fs::File>> {
    let directory_metadata = match fs::symlink_metadata(legacy_directory) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => {
            return Err(error)
                .with_context(|| format!("无法检查旧版输出锁目录 {}", legacy_directory.display()));
        }
    };
    validate_real_directory(legacy_directory, &directory_metadata)?;

    let legacy_path = legacy_directory.join(legacy_lock_file_name(normalized_target));
    let metadata = match fs::symlink_metadata(&legacy_path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => {
            return Err(error)
                .with_context(|| format!("无法检查旧版输出锁 {}", legacy_path.display()));
        }
    };
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        bail!("旧版输出锁不是普通文件：{}", legacy_path.display());
    }
    let file = open_lock_file_handle(&legacy_path, false)
        .with_context(|| format!("无法打开旧版输出锁 {}", legacy_path.display()))?;
    if !file.metadata().is_ok_and(|metadata| metadata.is_file())
        || !path_matches_open_file(&legacy_path, &file, false)
    {
        bail!("旧版输出锁路径在打开期间被替换：{}", legacy_path.display());
    }
    try_lock_output_file(&file, &legacy_path, output_path, true)?;
    if !path_matches_open_file(&legacy_path, &file, false) {
        bail!("旧版输出锁路径在获取期间被替换：{}", legacy_path.display());
    }
    Ok(Some(file))
}

fn legacy_lock_file_name(normalized_target: &Path) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    normalized_target.hash(&mut hasher);
    format!("{:016x}.lock", hasher.finish())
}

#[cfg(test)]
fn acquire_existing_legacy_lock(
    _normalized_target: &Path,
    _output_path: &Path,
) -> Result<Option<fs::File>> {
    Ok(None)
}

#[cfg(target_os = "macos")]
fn set_private_permissions(file: &fs::File, mode: u32) -> std::io::Result<()> {
    use std::ffi::c_void;
    use std::os::unix::fs::PermissionsExt;
    use std::os::unix::io::AsRawFd;

    const ACL_TYPE_EXTENDED: i32 = 0x0000_0100;

    unsafe extern "C" {
        fn acl_init(count: i32) -> *mut c_void;
        fn acl_set_fd_np(file_descriptor: i32, acl: *mut c_void, acl_type: i32) -> i32;
        fn acl_free(value: *mut c_void) -> i32;
    }

    let acl = unsafe { acl_init(0) };
    if acl.is_null() {
        return Err(std::io::Error::last_os_error());
    }
    let result = unsafe { acl_set_fd_np(file.as_raw_fd(), acl, ACL_TYPE_EXTENDED) };
    let error = if result == 0 {
        None
    } else {
        Some(std::io::Error::last_os_error())
    };
    let _ = unsafe { acl_free(acl) };
    if let Some(error) = error {
        return Err(error);
    }

    file.set_permissions(fs::Permissions::from_mode(mode))
}

#[cfg(all(unix, not(target_os = "macos")))]
fn set_private_permissions(file: &fs::File, mode: u32) -> std::io::Result<()> {
    use std::os::unix::fs::PermissionsExt;

    file.set_permissions(fs::Permissions::from_mode(mode))
}

#[cfg(not(unix))]
fn set_private_permissions(_file: &fs::File, _mode: u32) -> std::io::Result<()> {
    Ok(())
}

#[cfg(not(windows))]
fn open_directory_handle(path: &Path) -> std::io::Result<fs::File> {
    fs::File::open(path)
}

#[cfg(windows)]
fn open_directory_handle(path: &Path) -> std::io::Result<fs::File> {
    use std::os::windows::fs::OpenOptionsExt;

    const FILE_FLAG_BACKUP_SEMANTICS: u32 = 0x02000000;
    const FILE_FLAG_OPEN_REPARSE_POINT: u32 = 0x00200000;
    const FILE_SHARE_READ: u32 = 0x00000001;
    const FILE_SHARE_WRITE: u32 = 0x00000002;
    let mut options = fs::OpenOptions::new();
    options
        .access_mode(0)
        .share_mode(FILE_SHARE_READ | FILE_SHARE_WRITE)
        .custom_flags(FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT)
        .open(path)
}

#[cfg(not(windows))]
fn open_lock_file_handle(path: &Path, create: bool) -> std::io::Result<fs::File> {
    fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(create)
        .truncate(false)
        .open(path)
}

#[cfg(windows)]
fn open_lock_file_handle(path: &Path, create: bool) -> std::io::Result<fs::File> {
    use std::os::windows::fs::OpenOptionsExt;

    const FILE_FLAG_OPEN_REPARSE_POINT: u32 = 0x00200000;
    const FILE_SHARE_READ: u32 = 0x00000001;
    const FILE_SHARE_WRITE: u32 = 0x00000002;
    let mut options = fs::OpenOptions::new();
    options
        .read(true)
        .write(true)
        .create(create)
        .truncate(false)
        .share_mode(FILE_SHARE_READ | FILE_SHARE_WRITE)
        .custom_flags(FILE_FLAG_OPEN_REPARSE_POINT)
        .open(path)
}

#[cfg(unix)]
fn path_matches_open_file(path: &Path, file: &fs::File, expect_directory: bool) -> bool {
    use std::os::unix::fs::MetadataExt;

    let Ok(path_metadata) = fs::symlink_metadata(path) else {
        return false;
    };
    if path_metadata.file_type().is_symlink()
        || path_metadata.is_dir() != expect_directory
        || path_metadata.is_file() == expect_directory
    {
        return false;
    }
    let Ok(file_metadata) = file.metadata() else {
        return false;
    };
    file_metadata.is_dir() == expect_directory
        && file_metadata.is_file() != expect_directory
        && path_metadata.dev() == file_metadata.dev()
        && path_metadata.ino() == file_metadata.ino()
}

#[cfg(windows)]
fn path_matches_open_file(path: &Path, file: &fs::File, expect_directory: bool) -> bool {
    use std::os::windows::fs::{FileTypeExt, MetadataExt};

    const FILE_ATTRIBUTE_REPARSE_POINT: u32 = 0x00000400;

    let Ok(path_metadata) = fs::symlink_metadata(path) else {
        return false;
    };
    let file_type = path_metadata.file_type();
    if file_type.is_symlink()
        || file_type.is_symlink_dir()
        || file_type.is_symlink_file()
        || path_metadata.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0
        || path_metadata.is_dir() != expect_directory
        || path_metadata.is_file() == expect_directory
    {
        return false;
    }
    let Ok(reopened) = open_identity_handle(path, expect_directory) else {
        return false;
    };
    matches!(
        (windows_file_identity(file), windows_file_identity(&reopened)),
        (Some(left), Some(right)) if left == right
    )
}

#[cfg(not(any(unix, windows)))]
fn path_matches_open_file(path: &Path, file: &fs::File, expect_directory: bool) -> bool {
    let Ok(path_metadata) = fs::symlink_metadata(path) else {
        return false;
    };
    let Ok(file_metadata) = file.metadata() else {
        return false;
    };
    !path_metadata.file_type().is_symlink()
        && path_metadata.is_dir() == expect_directory
        && file_metadata.is_dir() == expect_directory
}

#[cfg(windows)]
fn open_identity_handle(path: &Path, directory: bool) -> std::io::Result<fs::File> {
    use std::os::windows::fs::OpenOptionsExt;

    const FILE_FLAG_BACKUP_SEMANTICS: u32 = 0x02000000;
    const FILE_FLAG_OPEN_REPARSE_POINT: u32 = 0x00200000;
    const FILE_SHARE_READ: u32 = 0x00000001;
    const FILE_SHARE_WRITE: u32 = 0x00000002;
    let flags = FILE_FLAG_OPEN_REPARSE_POINT
        | if directory {
            FILE_FLAG_BACKUP_SEMANTICS
        } else {
            0
        };
    let mut options = fs::OpenOptions::new();
    options
        .access_mode(0)
        .share_mode(FILE_SHARE_READ | FILE_SHARE_WRITE)
        .custom_flags(flags)
        .open(path)
}

#[cfg(windows)]
fn windows_file_identity(file: &fs::File) -> Option<(u32, u64)> {
    use std::ffi::c_void;
    use std::mem::MaybeUninit;
    use std::os::windows::io::AsRawHandle;

    #[repr(C)]
    struct FileTime {
        low: u32,
        high: u32,
    }

    #[repr(C)]
    struct ByHandleFileInformation {
        file_attributes: u32,
        creation_time: FileTime,
        last_access_time: FileTime,
        last_write_time: FileTime,
        volume_serial_number: u32,
        file_size_high: u32,
        file_size_low: u32,
        number_of_links: u32,
        file_index_high: u32,
        file_index_low: u32,
    }

    #[link(name = "Kernel32")]
    unsafe extern "system" {
        #[link_name = "GetFileInformationByHandle"]
        fn get_file_information_by_handle(
            file: *mut c_void,
            information: *mut ByHandleFileInformation,
        ) -> i32;
    }

    let mut information = MaybeUninit::<ByHandleFileInformation>::uninit();
    let succeeded = unsafe {
        get_file_information_by_handle(
            file.as_raw_handle().cast::<c_void>(),
            information.as_mut_ptr(),
        )
    };
    if succeeded == 0 {
        return None;
    }
    let information = unsafe { information.assume_init() };
    let file_index =
        (u64::from(information.file_index_high) << 32) | u64::from(information.file_index_low);
    Some((information.volume_serial_number, file_index))
}

fn output_identity_bytes(parent: &fs::File, file_name: &std::ffi::OsStr) -> Result<Vec<u8>> {
    let parent_identity = parent_directory_identity_bytes(parent)?;
    let file_name_identity = platform_output_file_name_identity_bytes(file_name)?;
    let mut identity = b"spt-output-identity-v2\0".to_vec();
    identity.extend_from_slice(&(parent_identity.len() as u64).to_le_bytes());
    identity.extend_from_slice(&parent_identity);
    identity.extend_from_slice(&(file_name_identity.len() as u64).to_le_bytes());
    identity.extend_from_slice(&file_name_identity);
    Ok(identity)
}

#[cfg(unix)]
fn parent_directory_identity_bytes(parent: &fs::File) -> Result<Vec<u8>> {
    use std::os::unix::fs::MetadataExt;

    let metadata = parent
        .metadata()
        .context("无法读取输出父目录句柄 identity")?;
    if !metadata.is_dir() {
        bail!("输出父目录句柄不是目录");
    }
    let mut identity = Vec::with_capacity(16);
    identity.extend_from_slice(&metadata.dev().to_le_bytes());
    identity.extend_from_slice(&metadata.ino().to_le_bytes());
    Ok(identity)
}

#[cfg(windows)]
fn parent_directory_identity_bytes(parent: &fs::File) -> Result<Vec<u8>> {
    let (volume, file_index) =
        windows_file_identity(parent).context("无法读取 Windows 输出父目录句柄 identity")?;
    let mut identity = Vec::with_capacity(12);
    identity.extend_from_slice(&volume.to_le_bytes());
    identity.extend_from_slice(&file_index.to_le_bytes());
    Ok(identity)
}

#[cfg(not(any(unix, windows)))]
fn parent_directory_identity_bytes(_parent: &fs::File) -> Result<Vec<u8>> {
    // Unknown platforms conservatively serialize all output directories.
    Ok(b"unknown-platform-output-parent".to_vec())
}

fn platform_output_file_name_identity_bytes(_file_name: &std::ffi::OsStr) -> Result<Vec<u8>> {
    // Directory-level locking is the only table-free identity that cannot miss
    // HFS+ ignorable characters, NTFS short names, or case-insensitive Unix
    // mounts such as vfat, exFAT, ntfs3, and CIFS nocase.
    Ok(Vec::new())
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
        let state = tempfile::tempdir().unwrap();
        let lock_directory = state.path().join("state/output-locks");
        fs::create_dir(lock_directory.parent().unwrap()).unwrap();
        let path = directory.path().join("result.md");
        fs::write(&path, "old").unwrap();
        assert!(AtomicOutput::begin_in(&path, false, &lock_directory).is_err());
        assert_eq!(fs::read_to_string(&path).unwrap(), "old");
        AtomicOutput::begin_in(&path, true, &lock_directory)
            .unwrap()
            .commit("new")
            .unwrap();
        assert_eq!(fs::read_to_string(&path).unwrap(), "new");
    }

    #[test]
    fn atomic_writer_rejects_a_replaced_named_temporary_file() {
        let directory = tempfile::tempdir().unwrap();
        let output = directory.path().join("result.md");
        let transaction = AtomicOutput::begin(&output, false).unwrap();
        let temporary_path = transaction.temporary.path().to_owned();
        let moved = directory.path().join("moved-temp");
        fs::rename(&temporary_path, &moved).unwrap();
        fs::write(&temporary_path, "attacker replacement").unwrap();

        assert!(transaction.commit("trusted transcript").is_err());
        assert!(!output.exists());
    }

    #[cfg(unix)]
    #[test]
    fn output_parent_must_be_private_and_owned_by_the_current_user() {
        use std::os::unix::fs::PermissionsExt;

        let directory = tempfile::tempdir().unwrap();
        fs::set_permissions(directory.path(), fs::Permissions::from_mode(0o777)).unwrap();
        let output = directory.path().join("result.md");
        assert!(AtomicOutput::begin(&output, false).is_err());
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn macos_deny_only_acl_does_not_grant_write() {
        let directory = tempfile::tempdir().unwrap();
        let status = std::process::Command::new("/bin/chmod")
            .env_remove("OPENROUTER_API_KEY")
            .args(["+a", "everyone deny delete"])
            .arg(directory.path())
            .status()
            .unwrap();
        assert!(status.success());
        let handle = fs::File::open(directory.path()).unwrap();
        assert!(!macos_handle_acl_grants_write(&handle).unwrap());
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn macos_write_allow_acl_is_rejected() {
        let directory = tempfile::tempdir().unwrap();
        let status = std::process::Command::new("/bin/chmod")
            .env_remove("OPENROUTER_API_KEY")
            .args(["+a", "everyone allow add_file"])
            .arg(directory.path())
            .status()
            .unwrap();
        assert!(status.success());
        let handle = fs::File::open(directory.path()).unwrap();
        assert!(macos_handle_acl_grants_write(&handle).unwrap());
    }

    #[test]
    fn output_lock_shard_persists_and_is_reusable_after_drop() {
        let output_directory = tempfile::tempdir().unwrap();
        let state_directory = tempfile::tempdir().unwrap();
        let output = output_directory.path().join("lock-cleanup.md");
        let lock_directory = state_directory.path().join("state/output-locks");
        fs::create_dir(lock_directory.parent().unwrap()).unwrap();

        let first = acquire_output_lock_in(&output, &lock_directory, false).unwrap();
        let shard_path = first.shard_path.clone();
        assert!(shard_path.exists());
        assert_eq!(fs::metadata(&shard_path).unwrap().len(), 0);
        let contention = acquire_output_lock_in(&output, &lock_directory, false)
            .err()
            .unwrap();
        assert!(contention.to_string().contains("共享锁分片"));

        drop(first);
        assert!(shard_path.exists());
        assert_eq!(fs::metadata(&shard_path).unwrap().len(), 0);
        let second = acquire_output_lock_in(&output, &lock_directory, false).unwrap();
        assert_eq!(second.shard_path, shard_path);
    }

    #[test]
    fn state_root_must_be_absolute() {
        assert!(lock_directory_for_state_root(PathBuf::from("relative-state")).is_err());
        #[cfg(unix)]
        assert!(lock_directory_for_state_root(PathBuf::from("/")).is_err());
        let absolute = env::temp_dir().join("spt-absolute-state");
        assert_eq!(
            lock_directory_for_state_root(absolute.clone()).unwrap(),
            absolute.join("output-locks")
        );
    }

    #[test]
    fn fnv_and_shard_names_are_stable_and_bounded() {
        assert_eq!(fnv1a64(b""), 0xcbf29ce484222325);
        assert_eq!(fnv1a64(b"a"), 0xaf63dc4c8601ec8c);
        assert_eq!(fnv1a64(b"hello"), 0xa430d84680aabd0b);
        assert_eq!(shard_file_name(0), "shard-000.lock");
        assert_eq!(shard_file_name(4_095), "shard-fff.lock");
        assert_eq!(shard_file_name(4_096), "shard-000.lock");
    }

    #[test]
    fn existing_v04_lock_is_held_without_creating_missing_legacy_state() {
        let root = tempfile::tempdir().unwrap();
        let output_directory = root.path().join("output");
        let legacy_directory = root.path().join("legacy/output-locks");
        fs::create_dir(&output_directory).unwrap();
        fs::create_dir_all(&legacy_directory).unwrap();
        let output = output_directory.join("meeting.md");
        let normalized_target = normalized_output_target(&output).unwrap();
        let legacy_path = legacy_directory.join(legacy_lock_file_name(&normalized_target));
        let old_process_lock = open_lock_file_handle(&legacy_path, true).unwrap();
        old_process_lock.try_lock_exclusive().unwrap();

        let error = acquire_existing_legacy_lock_from_directory(
            &normalized_target,
            &output,
            &legacy_directory,
        )
        .err()
        .unwrap();
        assert!(error.to_string().contains("旧版 spt 任务"));

        FileExt::unlock(&old_process_lock).unwrap();
        drop(old_process_lock);
        let compatibility_lock = acquire_existing_legacy_lock_from_directory(
            &normalized_target,
            &output,
            &legacy_directory,
        )
        .unwrap()
        .unwrap();
        drop(compatibility_lock);

        let missing_directory = root.path().join("missing/output-locks");
        assert!(
            acquire_existing_legacy_lock_from_directory(
                &normalized_target,
                &output,
                &missing_directory,
            )
            .unwrap()
            .is_none()
        );
        assert!(!missing_directory.exists());
    }

    #[test]
    fn test_state_directory_is_scoped_to_the_process() {
        let directory = output_lock_directory().unwrap();
        assert!(directory.is_absolute());
        assert!(
            directory
                .to_string_lossy()
                .contains(&process::id().to_string())
        );
        assert_eq!(directory.file_name().unwrap(), "output-locks");
        let state_root = directory.parent().unwrap();
        if state_root.exists() {
            fs::remove_dir_all(state_root).unwrap();
        }
    }

    #[cfg(windows)]
    #[test]
    fn windows_file_name_identity_is_conservatively_directory_level() {
        use std::ffi::OsStr;

        assert!(
            platform_output_file_name_identity_bytes(OsStr::new("LongFileName.md"))
                .unwrap()
                .is_empty()
        );
        assert!(
            platform_output_file_name_identity_bytes(OsStr::new("LONGFI~1.MD"))
                .unwrap()
                .is_empty()
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn macos_identity_is_directory_level_and_collapses_firmlink_aliases() {
        let root = tempfile::tempdir().unwrap();
        let first = canonical_output_identity(&root.path().join("Meeting.md")).unwrap();
        let second = canonical_output_identity(&root.path().join("完全不同.md")).unwrap();
        assert_eq!(
            first.bytes, second.bytes,
            "macOS 同目录输出必须保守共享一个 shard"
        );

        let canonical_parent = fs::canonicalize(".").unwrap();
        let data_alias = Path::new("/System/Volumes/Data").join(
            canonical_parent
                .strip_prefix("/")
                .expect("macOS canonical path 应为绝对路径"),
        );
        if data_alias.is_dir() {
            let direct = canonical_output_identity(&canonical_parent.join("meeting.md")).unwrap();
            let aliased = canonical_output_identity(&data_alias.join("meeting.md")).unwrap();
            assert_eq!(direct.bytes, aliased.bytes);

            let state = tempfile::tempdir().unwrap();
            let lock_directory = state.path().join("state/output-locks");
            fs::create_dir(lock_directory.parent().unwrap()).unwrap();
            let first_lock = acquire_output_lock_in(
                &canonical_parent.join("__spt_firmlink_lock_test__.md"),
                &lock_directory,
                false,
            )
            .unwrap();
            let contention = acquire_output_lock_in(
                &data_alias.join("__spt_firmlink_lock_test__.md"),
                &lock_directory,
                false,
            )
            .err()
            .unwrap();
            assert!(contention.to_string().contains("共享锁分片"));
            drop(first_lock);
        }
    }

    #[test]
    fn every_platform_uses_directory_level_output_identity() {
        let root = tempfile::tempdir().unwrap();
        assert_eq!(
            canonical_output_identity(&root.path().join("Meeting.md"))
                .unwrap()
                .bytes,
            canonical_output_identity(&root.path().join("meeting.md"))
                .unwrap()
                .bytes
        );
    }

    #[cfg(unix)]
    #[test]
    fn state_root_lock_directory_and_shard_symlinks_are_rejected() {
        use std::os::unix::fs::symlink;

        let root = tempfile::tempdir().unwrap();
        let victim_directory = root.path().join("victim-directory");
        fs::create_dir(&victim_directory).unwrap();

        let state_link = root.path().join("state-link");
        symlink(&victim_directory, &state_link).unwrap();
        assert!(prepare_output_lock_directory(&state_link.join("output-locks"), true).is_err());

        let real_state = root.path().join("real-state");
        fs::create_dir(&real_state).unwrap();
        let output_locks_link = real_state.join("output-locks");
        symlink(&victim_directory, &output_locks_link).unwrap();
        assert!(prepare_output_lock_directory(&output_locks_link, true).is_err());

        let safe_state = root.path().join("safe-state");
        let safe_locks = safe_state.join("output-locks");
        let prepared = prepare_output_lock_directory(&safe_locks, true).unwrap();
        let output_directory = root.path().join("output");
        fs::create_dir(&output_directory).unwrap();
        let output = output_directory.join("meeting.md");
        let identity = canonical_output_identity(&output).unwrap();
        let hash = fnv1a64(&identity.bytes);
        let shard_path = prepared.path.join(shard_file_name(hash));
        let victim_file = root.path().join("victim-file");
        fs::write(&victim_file, b"do not touch").unwrap();
        symlink(&victim_file, &shard_path).unwrap();
        assert!(acquire_output_lock_in(&output, &safe_locks, false).is_err());
        assert_eq!(fs::read(&victim_file).unwrap(), b"do not touch");
    }

    #[cfg(unix)]
    #[test]
    fn state_and_shard_permissions_are_set_through_open_handles() {
        use std::os::unix::fs::PermissionsExt;

        let root = tempfile::tempdir().unwrap();
        let output_directory = root.path().join("output");
        let state_root = root.path().join("state");
        let lock_directory = state_root.join("output-locks");
        fs::create_dir(&output_directory).unwrap();
        let output = output_directory.join("meeting.md");
        let lock = acquire_output_lock_in(&output, &lock_directory, false).unwrap();

        assert_eq!(
            fs::metadata(&state_root).unwrap().permissions().mode() & 0o777,
            0o700
        );
        assert_eq!(
            fs::metadata(&lock_directory).unwrap().permissions().mode() & 0o777,
            0o700
        );
        assert_eq!(
            fs::metadata(&lock.shard_path).unwrap().permissions().mode() & 0o777,
            0o600
        );
    }

    #[cfg(unix)]
    #[test]
    fn existing_custom_state_root_mode_is_preserved() {
        use std::os::unix::fs::PermissionsExt;

        let root = tempfile::tempdir().unwrap();
        let output_directory = root.path().join("output");
        let custom_state_root = root.path().join("shared-state-root");
        let lock_directory = custom_state_root.join("output-locks");
        fs::create_dir(&output_directory).unwrap();
        fs::create_dir(&custom_state_root).unwrap();
        fs::set_permissions(&custom_state_root, fs::Permissions::from_mode(0o750)).unwrap();
        let output = output_directory.join("meeting.md");

        let lock =
            acquire_output_lock_in_with_policy(&output, &lock_directory, false, false).unwrap();
        assert_eq!(
            fs::metadata(&custom_state_root)
                .unwrap()
                .permissions()
                .mode()
                & 0o777,
            0o750
        );
        assert_eq!(
            fs::metadata(&lock_directory).unwrap().permissions().mode() & 0o777,
            0o700
        );
        assert_eq!(
            fs::metadata(&lock.shard_path).unwrap().permissions().mode() & 0o777,
            0o600
        );
    }

    #[cfg(unix)]
    #[test]
    fn writable_custom_state_root_is_rejected_without_chmod() {
        use std::os::unix::fs::PermissionsExt;

        let root = tempfile::tempdir().unwrap();
        let output_directory = root.path().join("output");
        let custom_state_root = root.path().join("writable-state-root");
        let lock_directory = custom_state_root.join("output-locks");
        fs::create_dir(&output_directory).unwrap();
        fs::create_dir(&custom_state_root).unwrap();
        fs::set_permissions(&custom_state_root, fs::Permissions::from_mode(0o770)).unwrap();
        let output = output_directory.join("meeting.md");

        let error = acquire_output_lock_in_with_policy(&output, &lock_directory, false, false)
            .err()
            .unwrap();
        assert!(error.to_string().contains("不能允许组或其他用户写入"));
        assert_eq!(
            fs::metadata(&custom_state_root)
                .unwrap()
                .permissions()
                .mode()
                & 0o777,
            0o770
        );
        assert!(!lock_directory.exists());
    }

    #[cfg(unix)]
    #[test]
    fn custom_state_root_in_nonsticky_writable_parent_is_rejected() {
        use std::os::unix::fs::PermissionsExt;

        let root = tempfile::tempdir().unwrap();
        let output_directory = root.path().join("output");
        let writable_parent = root.path().join("writable-parent");
        let custom_state_root = writable_parent.join("private-state-root");
        let lock_directory = custom_state_root.join("output-locks");
        fs::create_dir(&output_directory).unwrap();
        fs::create_dir(&writable_parent).unwrap();
        fs::create_dir(&custom_state_root).unwrap();
        fs::set_permissions(&writable_parent, fs::Permissions::from_mode(0o777)).unwrap();
        fs::set_permissions(&custom_state_root, fs::Permissions::from_mode(0o700)).unwrap();
        let output = output_directory.join("meeting.md");

        let error = acquire_output_lock_in_with_policy(&output, &lock_directory, false, false)
            .err()
            .unwrap();
        assert!(error.to_string().contains("无 sticky 保护"));
        assert!(!lock_directory.exists());
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
    fn ocr_metadata_accounts_for_accepted_and_rejected_responses_once() {
        let accepted = Completion {
            origin: CompletionOrigin::Chat,
            text: "识别正文".into(),
            model: "accepted/model".into(),
            provider: "accepted-provider".into(),
            model_reported_by_api: true,
            provider_reported_by_api: true,
            prompt_tokens: 10,
            completion_tokens: 8,
            reasoning_tokens: 2,
            cost: 0.1,
            usage_reported: true,
            reasoning_tokens_reported: true,
        };
        let rejected = Completion {
            origin: CompletionOrigin::Chat,
            text: String::new(),
            model: "rejected/model".into(),
            provider: "rejected-provider".into(),
            model_reported_by_api: false,
            provider_reported_by_api: true,
            prompt_tokens: 3,
            completion_tokens: 4,
            reasoning_tokens: 1,
            cost: 0.2,
            usage_reported: true,
            reasoning_tokens_reported: true,
        };
        let markdown = render_ocr(
            Path::new("scan.png"),
            &Config::default(),
            &ImageInfo {
                codec: "mjpeg".into(),
                container: "jpeg_pipe".into(),
                width: 640,
                height: 480,
            },
            &accepted,
            &[rejected],
        )
        .unwrap();

        assert!(markdown.contains("accounted_model_responses: 2"));
        assert!(markdown.contains("rejected_accounted_responses: 1"));
        assert!(markdown.contains("reported_prompt_tokens: 13"));
        assert!(markdown.contains("reported_completion_tokens: 12"));
        assert!(markdown.contains("reported_reasoning_tokens: 3"));
        assert!(markdown.contains("reported_visible_output_tokens: 9"));
        assert!(markdown.contains("reported_accepted_cost_usd: 0.100000000"));
        assert!(markdown.contains("reported_rejected_cost_usd: 0.200000000"));
        assert!(markdown.contains("reported_accounted_cost_usd: 0.300000000"));
        assert!(markdown.contains(
            "reported_model_bucket_policy: \"only_api_reported_model_values_no_requested_fallback\""
        ));
        assert!(markdown.contains("reported_responses_by_model: {\"accepted/model\":1}"));
        assert!(markdown.contains("reported_cost_usd_by_model: {\"accepted/model\":0.1}"));
        assert!(markdown.contains("responses_without_model_reported_by_api: 1"));
        assert!(markdown.contains("reported_cost_usd_without_model_reported_by_api: 0.200000000"));
        assert!(!markdown.contains("rejected/model"));
        assert!(markdown.ends_with("# scan OCR\n\n识别正文\n"));
    }

    #[test]
    fn ocr_rejected_accounting_cannot_carry_rendered_text() {
        let mut rejected = Completion {
            origin: CompletionOrigin::Chat,
            text: "不得渲染".into(),
            model: "rejected/model".into(),
            provider: "rejected-provider".into(),
            model_reported_by_api: true,
            provider_reported_by_api: true,
            prompt_tokens: 1,
            completion_tokens: 1,
            reasoning_tokens: 0,
            cost: 0.1,
            usage_reported: true,
            reasoning_tokens_reported: false,
        };
        let accepted = rejected.clone();
        rejected.text = "untrusted rejected text".into();
        let error = render_ocr(
            Path::new("scan.png"),
            &Config::default(),
            &ImageInfo {
                codec: "mjpeg".into(),
                container: "jpeg_pipe".into(),
                width: 1,
                height: 1,
            },
            &accepted,
            &[rejected],
        )
        .unwrap_err();
        assert!(error.to_string().contains("空正文"));
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

    fn dedicated_stt_part() -> TranscriptPart {
        TranscriptPart {
            start_ms: 0,
            end_ms: 1_000,
            completion: Completion {
                origin: CompletionOrigin::Stt,
                text: "S1：我们".into(),
                model: "test/model".into(),
                provider: "test/provider".into(),
                model_reported_by_api: true,
                provider_reported_by_api: true,
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
            quality_reviewed: false,
            quality_review_advisory: false,
            quality_cleanup_turns: 0,
            quality_trigger_codes: vec!["asr_crosscheck_exact_consensus_not_ground_truth".into()],
            quality_residual_advisory_codes: Vec::new(),
        }
    }

    fn render_part(part: &TranscriptPart, config: &Config, mode: TranscriptMode) -> String {
        render_transcript(
            Path::new("meeting.wav"),
            config,
            &AudioInfo {
                duration_ms: 1_000,
                codec: "pcm_s16le".into(),
                container: "wav".into(),
            },
            std::slice::from_ref(part),
            mode,
        )
        .unwrap()
    }

    #[test]
    fn quality_metadata_describes_dedicated_stt_and_exact_crosscheck_honestly() {
        let part = dedicated_stt_part();
        let markdown = render_part(&part, &Config::default(), TranscriptMode::Quality);

        assert!(markdown.contains("transcript_mode: \"quality\""));
        assert!(
            markdown.contains(
                "transcript_editing: \"protected_host_readability_cleanup_on_primary_asr\""
            )
        );
        assert!(markdown.contains("model_requested: \"qwen/qwen3-asr-1.7b\""));
        assert!(markdown.contains("base_model_requested: \"qwen/qwen3-asr-1.7b\""));
        assert!(markdown.contains("asr_model_requested: \"qwen/qwen3-asr-1.7b\""));
        assert!(markdown.contains("asr_provider_expected: \"deepinfra\""));
        assert!(markdown.contains("quality_asr_model: \"fish-audio/transcribe-1\""));
        assert!(markdown.contains("quality_asr_provider: \"fish-audio\""));
        assert!(markdown.contains("quality_asr_provider_expected: \"fish-audio\""));
        assert!(
            markdown.contains("multimodal_overlay_model_configured: \"google/gemini-3.7-flash\"")
        );
        assert!(markdown.contains("multimodal_overlay_used: false"));
        assert!(markdown.contains("quality_overlay_model_requested: \"google/gemini-3.7-flash\""));
        assert!(
            markdown.contains("multimodal_overlay_provider_configured: \"google-vertex/global\"")
        );
        assert!(markdown.contains("stt_providers_reported_by_api: \"test/provider\""));
        assert!(markdown.contains("stt_provider_reported_for_all_accounted_responses: true"));
        assert!(markdown.contains(
            "provider_privacy_mode: \"catalog_unique_zdr_preflight_no_request_level_pin\""
        ));
        assert!(markdown.contains(
            "asr_provider_privacy_mode: \"catalog_unique_zdr_preflight_no_request_level_pin\""
        ));
        assert!(markdown.contains(
            "quality_asr_provider_privacy_mode: \"catalog_unique_zdr_preflight_no_request_level_pin\""
        ));
        assert!(markdown.contains("quality_review_model_requested: null"));
        assert!(markdown.contains(
            "transcription_strategy: \"dedicated-stt-cost-bounded-crosscheck-frozen-text-turn-overlay-v2\""
        ));
        assert!(markdown.contains(
            "text_authority: \"primary_dedicated_stt_canonical_content_frozen_presentation_rendered\""
        ));
        assert!(markdown.contains(
            "alignment_text_policy: \"segmentation_and_labels_only_canonical_fact_characters_immutable\""
        ));
        assert!(markdown.contains(
            "stage_b_text_policy: \"identity_mapping_only_canonical_fact_characters_immutable\""
        ));
        assert!(markdown.contains("alignment_may_modify_canonical_fact_characters: false"));
        assert!(markdown.contains("stage_b_may_modify_canonical_fact_characters: false"));
        assert!(
            markdown.contains("primary_asr_provider_bytes_restored_before_markdown_render: false")
        );
        assert!(markdown.contains("primary_asr_display_projection_slices_restored: true"));
        assert!(markdown.contains(
            "markdown_presentation_policy: \"host_protected_readability_cleanup_speaker_labels_and_markdown_escape\""
        ));
        assert!(markdown.contains("root_target_seconds: 120"));
        assert!(markdown.contains("transcript_accuracy_verification: \"not_measured\""));
        assert!(
            markdown
                .contains("transcript_accuracy_basis: \"no_ground_truth_comparison_performed\"")
        );
        assert!(markdown.contains("asr_crosscheck_status: \"exact_consensus_not_ground_truth\""));
        assert!(markdown.contains("asr_crosscheck_coverage: \"all_root_targets_checked\""));
        assert!(markdown.contains("asr_crosscheck_sampled_segments: 1"));
        assert!(markdown.contains("asr_crosscheck_skipped_segments: 0"));
        assert!(markdown.contains(
            "asr_crosscheck_signals: [\"asr_crosscheck_exact_consensus_not_ground_truth\"]"
        ));
        assert!(markdown.contains("asr_crosscheck_advisories: []"));
        assert!(markdown.contains("quality_review_triggers: []"));
        assert!(markdown.contains("quality_review_status: \"not_used_dedicated_stt_pipeline\""));
        assert!(markdown.contains("quality_review_segments: 0"));
        assert!(markdown.contains(
            "quality_gate_scope: \"legacy_surface_gate_not_used_by_dedicated_stt_pipeline\""
        ));
        assert!(markdown.contains("surface_quality_gate_status: \"not_used\""));
        assert!(markdown.contains("quality_gate_status: \"not_used\""));
        assert!(markdown.contains("surface_quality_residual_advisories: []"));
        assert!(markdown.contains("quality_residual_advisories: []"));
        assert!(markdown.contains("final_transcript_segments_from_primary_asr: 1"));
        assert!(markdown.contains("# meeting 事实保护可读性清稿（准确率未验证）"));
        assert!(markdown.contains(
            "chinese_script: \"zh-Hans_with_source_fact_spans_or_source_when_kana_present\""
        ));
        assert!(
            markdown.contains(
                "chinese_normalization: \"opencc-t2s-fact-span-protected-kana-preserving\""
            )
        );
        assert!(
            markdown.contains("speaker_tracking: \"openrouter-per-turn-reference-harness-v3\"")
        );
        assert!(
            markdown.contains("speaker_label_assignment_status: \"no_unknown_labels_present\"")
        );
        assert!(markdown.contains("speaker_all_turns_have_non_unknown_labels: true"));
        assert!(markdown.contains("stt_token_counts_may_be_unavailable: true"));
        assert!(markdown.contains("speaker_identity_accuracy: \"not_measured\""));
        assert!(markdown.contains("speaker_alignment_status: \"not_verified\""));
    }

    #[test]
    fn sampled_crosscheck_is_reported_as_cost_bounded_not_unavailable() {
        let checked = dedicated_stt_part();
        let mut skipped = dedicated_stt_part();
        skipped.start_ms = 1_000;
        skipped.end_ms = 2_000;
        skipped.quality_trigger_codes = vec!["asr_crosscheck_skipped_cost_bounded".into()];
        let markdown = render_transcript(
            Path::new("meeting.wav"),
            &Config::default(),
            &AudioInfo {
                duration_ms: 2_000,
                codec: "pcm_s16le".into(),
                container: "wav".into(),
            },
            &[checked, skipped],
            TranscriptMode::Quality,
        )
        .unwrap();

        assert!(
            markdown
                .contains("asr_crosscheck_status: \"sampled_exact_consensus_not_ground_truth\"")
        );
        assert!(markdown.contains(
            "asr_crosscheck_coverage: \"first_and_every_fifth_root_target_cost_bounded\""
        ));
        assert!(markdown.contains("asr_crosscheck_sampled_segments: 1"));
        assert!(markdown.contains("asr_crosscheck_skipped_segments: 1"));
        assert!(!markdown.contains("unavailable_requires_review"));
    }

    #[test]
    fn reported_route_values_are_separated_by_api_origin_and_completeness() {
        let mut part = dedicated_stt_part();
        part.completion.model_reported_by_api = false;
        part.completion.provider = "unreported".into();
        part.completion.provider_reported_by_api = false;
        part.completion.cost = 0.004;
        part.auxiliary_completions.push(Completion {
            origin: CompletionOrigin::Chat,
            text: String::new(),
            model: "google/gemini-3.7-flash".into(),
            provider: "Google".into(),
            model_reported_by_api: true,
            provider_reported_by_api: true,
            prompt_tokens: 1,
            completion_tokens: 1,
            reasoning_tokens: 0,
            cost: 0.0,
            usage_reported: true,
            reasoning_tokens_reported: true,
        });
        let markdown = render_part(&part, &Config::default(), TranscriptMode::Quality);

        assert!(markdown.contains("accounted_response_providers: \"Google, unreported\""));
        assert!(markdown.contains("stt_providers_reported_by_api: \"\""));
        assert!(markdown.contains("stt_provider_reported_for_all_accounted_responses: false"));
        assert!(markdown.contains("chat_providers_reported_by_api: \"Google\""));
        assert!(markdown.contains("chat_provider_reported_for_all_accounted_responses: true"));
        assert!(markdown.contains(
            "reported_model_bucket_policy: \"only_api_reported_model_values_no_requested_fallback\""
        ));
        assert!(markdown.contains("reported_responses_by_model: {\"google/gemini-3.7-flash\":1}"));
        assert!(!markdown.contains("reported_responses_by_model: {\"test/model\""));
        assert!(markdown.contains("responses_without_model_reported_by_api: 1"));
        assert!(markdown.contains("reported_cost_usd_without_model_reported_by_api: 0.004000000"));
        assert!(markdown.contains("multimodal_overlay_used: true"));
    }

    #[test]
    fn automatic_provider_sentinels_are_configured_but_never_expected_endpoints() {
        let config = Config {
            provider: "any".into(),
            asr_provider: "any".into(),
            quality_asr_provider: "any".into(),
            ..Config::default()
        };
        let markdown = render_part(&dedicated_stt_part(), &config, TranscriptMode::Quality);

        assert!(markdown.contains("asr_provider_configured: \"any\""));
        assert!(markdown.contains("asr_provider_routing_mode: \"automatic_unpinned\""));
        assert!(markdown.contains("asr_provider_expected: null"));
        assert!(markdown.contains("quality_asr_provider_configured: \"any\""));
        assert!(markdown.contains("quality_asr_provider_routing_mode: \"automatic_unpinned\""));
        assert!(markdown.contains("quality_asr_provider_expected: null"));
        assert!(markdown.contains("multimodal_overlay_provider_configured: \"any\""));
        assert!(
            markdown.contains("multimodal_overlay_provider_routing_mode: \"automatic_unpinned\"")
        );
        assert!(markdown.contains("multimodal_overlay_provider_expected: null"));
    }

    #[test]
    fn host_cleanup_metadata_is_separate_from_legacy_review_triggers() {
        let mut part = dedicated_stt_part();
        part.quality_cleanup_turns = 1;
        part.quality_trigger_codes
            .push("quality_cleanup_chinese_punctuation_normalized".into());
        let markdown = render_part(&part, &Config::default(), TranscriptMode::Quality);

        assert!(markdown.contains("quality_host_cleanup_turns: 1"));
        assert!(markdown.contains("quality_host_cleanup_status: \"applied\""));
        assert!(markdown.contains(
            "quality_host_cleanup_operations: [\"quality_cleanup_chinese_punctuation_normalized\"]"
        ));
        assert!(markdown.contains("quality_host_cleanup_advisories: []"));
        assert!(markdown.contains("quality_review_triggers: []"));
    }

    #[test]
    fn speaker_assignment_completeness_remains_separate_from_accuracy() {
        let mut unresolved_part = dedicated_stt_part();
        unresolved_part.speaker_ids = vec!["UNKNOWN".into()];
        let unresolved_markdown = render_part(
            &unresolved_part,
            &Config::default(),
            TranscriptMode::Quality,
        );
        assert!(
            unresolved_markdown
                .contains("speaker_label_assignment_status: \"unknown_labels_present\"")
        );
        assert!(unresolved_markdown.contains("speaker_all_turns_have_non_unknown_labels: false"));
        assert!(unresolved_markdown.contains("speaker_alignment_status: \"not_verified\""));
    }

    #[test]
    fn crosscheck_disagreement_and_unavailability_require_review_without_surface_gate_claims() {
        let mut disagreement_part = dedicated_stt_part();
        disagreement_part.quality_trigger_codes.clear();
        disagreement_part.quality_review_advisory = true;
        disagreement_part.quality_residual_advisory_codes =
            vec!["asr_crosscheck_disagreement".into()];
        let disagreement_markdown = render_part(
            &disagreement_part,
            &Config::default(),
            TranscriptMode::Quality,
        );
        assert!(
            disagreement_markdown
                .contains("asr_crosscheck_status: \"disagreement_requires_review\"")
        );
        assert!(
            disagreement_markdown
                .contains("asr_crosscheck_advisories: [\"asr_crosscheck_disagreement\"]")
        );
        assert!(
            disagreement_markdown
                .contains("quality_review_status: \"not_used_dedicated_stt_pipeline\"")
        );
        assert!(disagreement_markdown.contains("surface_quality_gate_status: \"not_used\""));
        assert!(disagreement_markdown.contains("surface_quality_residual_advisories: []"));
        assert!(
            disagreement_markdown
                .contains("quality_residual_advisories: [\"asr_crosscheck_disagreement\"]")
        );
        assert!(
            disagreement_markdown
                .contains("# meeting 事实保护可读性清稿（准确率未验证，含需复核片段）")
        );
        assert!(
            disagreement_markdown
                .contains("## 00:00:00–00:00:01（需复核：asr_crosscheck_disagreement）")
        );

        let mut unavailable_part = dedicated_stt_part();
        unavailable_part.quality_trigger_codes.clear();
        unavailable_part.quality_review_advisory = true;
        unavailable_part.quality_residual_advisory_codes =
            vec!["asr_crosscheck_unavailable".into()];
        let unavailable_markdown = render_part(
            &unavailable_part,
            &Config::default(),
            TranscriptMode::Quality,
        );
        assert!(
            unavailable_markdown.contains("asr_crosscheck_status: \"unavailable_requires_review\"")
        );
        assert!(
            unavailable_markdown
                .contains("asr_crosscheck_advisories: [\"asr_crosscheck_unavailable\"]")
        );
        assert!(unavailable_markdown.contains("surface_quality_gate_status: \"not_used\""));
    }

    #[test]
    fn raw_metadata_uses_only_primary_asr_and_does_not_claim_verbatim_accuracy() {
        let mut raw_part = dedicated_stt_part();
        raw_part.quality_review_advisory = false;
        raw_part.quality_trigger_codes.clear();
        raw_part.quality_residual_advisory_codes.clear();
        let raw_markdown = render_part(&raw_part, &Config::default(), TranscriptMode::Raw);
        assert!(raw_markdown.contains("transcript_mode: \"raw\""));
        assert!(
            raw_markdown
                .contains("transcript_editing: \"unpolished_primary_asr_not_verbatim_guaranteed\"")
        );
        assert!(raw_markdown.contains("quality_asr_model: null"));
        assert!(raw_markdown.contains("quality_asr_provider: null"));
        assert!(raw_markdown.contains("quality_asr_provider_expected: null"));
        assert!(raw_markdown.contains("quality_review_model_requested: null"));
        assert!(
            raw_markdown
                .contains("multimodal_overlay_model_configured: \"google/gemini-3.7-flash\"")
        );
        assert!(raw_markdown.contains("multimodal_overlay_used: false"));
        assert!(raw_markdown.contains("quality_overlay_model_requested: null"));
        assert!(raw_markdown.contains("quality_review_status: \"not_applicable\""));
        assert!(raw_markdown.contains("asr_crosscheck_status: \"not_applicable\""));
        assert!(raw_markdown.contains("surface_quality_gate_status: \"not_applicable\""));
        assert!(raw_markdown.contains("quality_gate_status: \"not_applicable\""));
        assert!(raw_markdown.contains("root_target_seconds: 120"));
        assert!(raw_markdown.contains("# meeting 单路 ASR 原始输出稿（准确率未验证）"));
    }

    #[test]
    fn zero_stt_token_counters_are_not_claimed_as_provider_reported_zero() {
        let mut part = dedicated_stt_part();
        part.completion.prompt_tokens = 0;
        part.completion.completion_tokens = 0;
        part.completion.usage_reported = false;
        part.completion.reasoning_tokens_reported = false;
        let markdown = render_part(&part, &Config::default(), TranscriptMode::Raw);
        assert!(markdown.contains("stt_token_counts_may_be_unavailable: true"));
        assert!(markdown.contains("usage_reported_for_all_accounted_responses: false"));
        assert!(markdown.contains("reported_prompt_tokens: 0"));
        assert!(markdown.contains("reported_completion_tokens: 0"));
    }

    #[test]
    fn any_stt_provider_is_reported_as_an_explicit_privacy_downgrade() {
        let config = Config {
            asr_provider: "any".into(),
            quality_asr_provider: "any".into(),
            provider: "any".into(),
            ..Config::default()
        };
        let markdown = render_part(&dedicated_stt_part(), &config, TranscriptMode::Quality);
        assert!(markdown.contains("provider_privacy_mode: \"any_explicit_privacy_downgrade\""));
        assert!(markdown.contains("asr_provider_privacy_mode: \"any_explicit_privacy_downgrade\""));
        assert!(
            markdown
                .contains("quality_asr_provider_privacy_mode: \"any_explicit_privacy_downgrade\"")
        );
        assert!(markdown.contains(
            "multimodal_overlay_provider_privacy_mode: \"any_explicit_privacy_downgrade\""
        ));
    }

    #[test]
    fn legacy_review_metadata_is_preserved_only_when_a_part_was_actually_reviewed() {
        let mut legacy_part = dedicated_stt_part();
        legacy_part.quality_reviewed = true;
        let markdown = render_part(&legacy_part, &Config::default(), TranscriptMode::Quality);
        assert!(markdown.contains("quality_review_model_requested: \"google/gemini-3.7-flash\""));
        assert!(markdown.contains("quality_review_status: \"completed\""));
        assert!(markdown.contains("quality_review_segments: 1"));
    }
}
