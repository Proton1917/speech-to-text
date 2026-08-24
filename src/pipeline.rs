use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use async_recursion::async_recursion;
use futures::{StreamExt, TryStreamExt, stream};
use tokio::sync::Semaphore;

use crate::config::Config;
use crate::media::{
    MediaChunk, canonicalize_audio, ensure_media_tools, markdown_output_path, normalize_image,
    prepare_audio_chunks, split_audio_chunk, validate_audio, validate_image,
};
use crate::openrouter::{CompletionResult, OpenRouterClient, looks_repetitive};
use crate::output::{AtomicOutput, TranscriptPart, ocr_output_path, render_ocr, render_transcript};

#[derive(Clone)]
struct TranscriptionContext {
    client: OpenRouterClient,
    config: Config,
    media_semaphore: Arc<Semaphore>,
}

pub async fn transcribe(input: &Path, config: &Config, force: bool) -> Result<PathBuf> {
    config.validate()?;
    ensure_media_tools()?;
    let mut info = validate_audio(input)?;
    let output = markdown_output_path(input)?;
    let output_transaction = AtomicOutput::begin(&output, force)?;
    let client = OpenRouterClient::from_environment(config.clone(), true)?;
    client.validate_selection("audio").await?;
    let workspace = private_workspace("spt-audio-")?;

    eprintln!(
        "正在生成 32 kHz 单声道无损母版：{} / {}",
        info.codec, info.container
    );
    let (canonical_path, canonical_info) =
        canonicalize_audio(input, workspace.path(), config.max_temp_bytes).await?;
    info.duration_ms = canonical_info.duration_ms;
    eprintln!(
        "正在从无损时间轴切分音频：{:.1} 分钟",
        info.duration_ms as f64 / 60_000.0,
    );
    let chunks = prepare_audio_chunks(&canonical_path, &info, config, workspace.path()).await?;
    eprintln!(
        "将处理 {} 个初始片段，最多并发 {} 个请求",
        chunks.len(),
        config.parallel_requests
    );

    let context = Arc::new(TranscriptionContext {
        client,
        config: config.clone(),
        media_semaphore: Arc::new(Semaphore::new(config.parallel_requests.min(2))),
    });
    let nested = stream::iter(chunks)
        .map(|chunk| process_chunk(Arc::clone(&context), chunk, 0))
        .buffer_unordered(config.parallel_requests)
        .try_collect::<Vec<_>>()
        .await?;
    let mut parts = nested.into_iter().flatten().collect::<Vec<_>>();
    parts.sort_by_key(|part| (part.start_ms, part.end_ms));
    validate_timeline(&parts, info.duration_ms)?;

    let markdown = render_transcript(input, config, &info, &parts)?;
    output_transaction.commit(&markdown)?;
    Ok(output)
}

pub async fn ocr(input: &Path, config: &Config, force: bool) -> Result<PathBuf> {
    config.validate()?;
    ensure_media_tools()?;
    let info = validate_image(input)?;
    let output = ocr_output_path(input)?;
    let output_transaction = AtomicOutput::begin(&output, force)?;
    let client = OpenRouterClient::from_environment(config.clone(), true)?;
    client.validate_selection("image").await?;
    let workspace = private_workspace("spt-ocr-")?;
    let normalized = normalize_image(input, workspace.path()).await?;
    eprintln!("正在识别图片文字：{} / {}", info.codec, info.container);
    let completion = match client.recognize_image(&normalized).await? {
        CompletionResult::Complete(completion) => completion,
        CompletionResult::NeedsSplit { reason } => {
            bail!("OCR 输出达到模型长度边界：{reason}；请裁成多张图片后重试")
        }
    };
    if looks_repetitive(&completion.text, 0) {
        bail!("OCR 返回内容存在明显循环，已拒绝生成正式文档");
    }
    let markdown = render_ocr(input, config, &info, &completion)?;
    output_transaction.commit(&markdown)?;
    Ok(output)
}

#[async_recursion]
async fn process_chunk(
    context: Arc<TranscriptionContext>,
    chunk: MediaChunk,
    adaptive_depth: u8,
) -> Result<Vec<TranscriptPart>> {
    eprintln!(
        "请求转写 {}–{}",
        crate::output::format_timestamp(chunk.start_ms),
        crate::output::format_timestamp(chunk.end_ms)
    );
    let result = context
        .client
        .transcribe_audio(&chunk.path, chunk.duration_ms())
        .await
        .with_context(|| {
            format!(
                "片段 {}–{} 转写失败",
                crate::output::format_timestamp(chunk.start_ms),
                crate::output::format_timestamp(chunk.end_ms)
            )
        })?;
    tokio::fs::remove_file(&chunk.path)
        .await
        .with_context(|| format!("无法清理临时音频片段 {}", chunk.path.display()))?;

    let split_reason = match &result {
        CompletionResult::NeedsSplit { reason } => Some(reason.clone()),
        CompletionResult::Complete(completion)
            if looks_repetitive(&completion.text, chunk.duration_ms()) =>
        {
            Some("模型输出出现病理性高密度循环".into())
        }
        CompletionResult::Complete(_) => None,
    };

    if let Some(reason) = split_reason {
        if adaptive_depth >= context.config.max_adaptive_depth {
            bail!(
                "片段 {}–{} 达到自适应切分深度上限 {}：{reason}",
                crate::output::format_timestamp(chunk.start_ms),
                crate::output::format_timestamp(chunk.end_ms),
                context.config.max_adaptive_depth
            );
        }
        let min_pair_ms = context.config.min_chunk_seconds * 2 * 1_000;
        if chunk.duration_ms() < min_pair_ms {
            bail!(
                "片段 {}–{} 已达到最小切分时长，仍无法得到完整可靠输出：{reason}",
                crate::output::format_timestamp(chunk.start_ms),
                crate::output::format_timestamp(chunk.end_ms)
            );
        }
        eprintln!(
            "片段 {}–{} 触发自适应二分：{reason}",
            crate::output::format_timestamp(chunk.start_ms),
            crate::output::format_timestamp(chunk.end_ms)
        );
        let (left, right) = {
            let _permit = context
                .media_semaphore
                .acquire()
                .await
                .context("媒体并发控制器已经关闭")?;
            let children = split_audio_chunk(&chunk).await.with_context(|| {
                format!(
                    "片段 {}–{} 自适应二分失败",
                    crate::output::format_timestamp(chunk.start_ms),
                    crate::output::format_timestamp(chunk.end_ms)
                )
            })?;
            let work_dir = chunk.path.parent().context("临时片段没有父目录")?;
            ensure_workspace_budget(work_dir, context.config.max_temp_bytes)?;
            children
        };
        let next_depth = adaptive_depth + 1;
        let (mut parts, right_parts) = tokio::try_join!(
            process_chunk(Arc::clone(&context), left, next_depth),
            process_chunk(Arc::clone(&context), right, next_depth)
        )?;
        parts.extend(right_parts);
        return Ok(parts);
    }

    let CompletionResult::Complete(completion) = result else {
        unreachable!("NeedsSplit 已在上方处理")
    };
    eprintln!(
        "完成 {}–{}（可见输出 {} tokens，总 completion {} tokens）",
        crate::output::format_timestamp(chunk.start_ms),
        crate::output::format_timestamp(chunk.end_ms),
        completion.visible_output_tokens(),
        completion.completion_tokens
    );
    Ok(vec![TranscriptPart {
        start_ms: chunk.start_ms,
        end_ms: chunk.end_ms,
        completion,
    }])
}

fn validate_timeline(parts: &[TranscriptPart], duration_ms: u64) -> Result<()> {
    let Some(first) = parts.first() else {
        bail!("转写结果为空");
    };
    if first.start_ms != 0 {
        bail!("转写片段没有从音频起点开始");
    }
    let mut expected_start = 0;
    for part in parts {
        if part.start_ms != expected_start || part.end_ms <= part.start_ms {
            bail!("转写片段时间线不连续");
        }
        expected_start = part.end_ms;
    }
    if expected_start != duration_ms {
        bail!("转写片段没有完整覆盖源音频");
    }
    Ok(())
}

fn private_workspace(prefix: &str) -> Result<tempfile::TempDir> {
    let mut builder = tempfile::Builder::new();
    builder.prefix(prefix);
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        builder.permissions(std::fs::Permissions::from_mode(0o700));
    }
    builder.tempdir().context("无法创建安全临时目录")
}

fn ensure_workspace_budget(work_dir: &Path, max_bytes: u64) -> Result<()> {
    let mut total = 0_u64;
    for entry in std::fs::read_dir(work_dir)
        .with_context(|| format!("无法读取临时工作目录 {}", work_dir.display()))?
    {
        let entry = entry.context("无法读取临时工作目录项")?;
        let metadata = entry
            .metadata()
            .with_context(|| format!("无法读取临时文件大小 {}", entry.path().display()))?;
        if metadata.is_file() {
            total = total.saturating_add(metadata.len());
        }
    }
    if total > max_bytes {
        bail!(
            "临时媒体已达到 max_temp_bytes 安全上限（当前 {:.2} GiB）",
            total as f64 / 1024_f64.powi(3)
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::openrouter::Completion;

    fn part(start_ms: u64, end_ms: u64) -> TranscriptPart {
        TranscriptPart {
            start_ms,
            end_ms,
            completion: Completion {
                text: "测试".into(),
                model: "test/model".into(),
                provider: "test/provider".into(),
                prompt_tokens: 1,
                completion_tokens: 1,
                reasoning_tokens: 0,
                cost: 0.0,
                usage_reported: true,
                reasoning_tokens_reported: true,
            },
        }
    }

    #[test]
    fn timeline_must_be_contiguous_and_complete() {
        assert!(validate_timeline(&[part(0, 1_000), part(1_000, 2_000)], 2_000).is_ok());
        assert!(validate_timeline(&[part(0, 1_000), part(1_001, 2_000)], 2_000).is_err());
        assert!(validate_timeline(&[part(100, 2_000)], 2_000).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn private_workspace_is_owner_only() {
        use std::os::unix::fs::PermissionsExt;
        let workspace = private_workspace("spt-test-").unwrap();
        let mode = std::fs::metadata(workspace.path())
            .unwrap()
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(mode, 0o700);
    }
}
