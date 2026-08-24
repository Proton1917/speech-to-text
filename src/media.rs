use std::ffi::{OsStr, OsString};
use std::fs;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use tokio::io::{AsyncRead, AsyncReadExt};
use tokio::process::Command as TokioCommand;

use crate::config::Config;
use crate::security::secure_file;

const AUDIO_EXTENSIONS: &[&str] = &[
    "aac", "aif", "aiff", "caf", "flac", "m4a", "m4b", "mp3", "oga", "ogg", "opus", "wav", "webm",
    "wma",
];
const IMAGE_EXTENSIONS: &[&str] = &["jpeg", "jpg", "png", "webp"];
const PROBE_STDOUT_LIMIT: usize = 1024 * 1024;
const CHILD_STDERR_LIMIT: usize = 256 * 1024;
const PROBE_TIMEOUT: Duration = Duration::from_secs(10);
const FFMPEG_TIMEOUT: Duration = Duration::from_secs(3_600);

#[derive(Clone, Debug)]
pub struct AudioInfo {
    pub duration_ms: u64,
    pub codec: String,
    pub container: String,
}

#[derive(Clone, Debug)]
pub struct ImageInfo {
    pub codec: String,
    pub container: String,
    pub width: u64,
    pub height: u64,
}

#[derive(Clone, Debug)]
pub struct MediaChunk {
    pub source_path: PathBuf,
    pub audio_start_ms: u64,
    pub start_ms: u64,
    pub end_ms: u64,
    pub lineage: String,
}

impl MediaChunk {
    pub fn duration_ms(&self) -> u64 {
        self.end_ms.saturating_sub(self.start_ms)
    }

    pub fn audio_duration_ms(&self) -> u64 {
        self.end_ms.saturating_sub(self.audio_start_ms)
    }

    pub fn context_ms(&self) -> u64 {
        self.start_ms.saturating_sub(self.audio_start_ms)
    }
}

#[derive(Clone, Debug)]
pub struct SpeakerReferenceRange {
    pub speaker_id: String,
    pub start_ms: u64,
    pub end_ms: u64,
}

#[derive(Clone, Debug)]
pub struct PacketReferenceWindow {
    pub speaker_id: String,
    pub start_ms: u64,
    pub end_ms: u64,
}

#[derive(Clone, Debug)]
pub struct SpeakerPacket {
    pub path: PathBuf,
    pub references: Vec<PacketReferenceWindow>,
    pub boundary_context: Option<PacketReferenceWindow>,
    pub candidates: Vec<PacketReferenceWindow>,
    pub total_duration_ms: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NonSilentRange {
    pub start_ms: u64,
    pub end_ms: u64,
}

#[derive(Debug, Deserialize)]
struct ProbeOutput {
    #[serde(default)]
    streams: Vec<ProbeStream>,
    #[serde(default)]
    format: ProbeFormat,
}

#[derive(Debug, Deserialize)]
struct ProbeStream {
    codec_type: Option<String>,
    codec_name: Option<String>,
    duration: Option<String>,
    duration_ts: Option<i64>,
    time_base: Option<String>,
    width: Option<u64>,
    height: Option<u64>,
    nb_frames: Option<String>,
    #[serde(default)]
    disposition: ProbeDisposition,
}

#[derive(Debug, Default, Deserialize)]
struct ProbeDisposition {
    #[serde(default)]
    attached_pic: u8,
}

#[derive(Debug, Default, Deserialize)]
struct ProbeFormat {
    duration: Option<String>,
    format_name: Option<String>,
}

pub fn ensure_media_tools() -> Result<()> {
    ensure_tool("ffprobe")?;
    ensure_tool("ffmpeg")?;
    Ok(())
}

pub async fn ensure_media_tools_async() -> Result<()> {
    tokio::task::spawn_blocking(ensure_media_tools)
        .await
        .context("媒体工具检查任务异常终止")?
}

pub fn validate_audio(path: &Path) -> Result<AudioInfo> {
    validate_common(path, AUDIO_EXTENSIONS, "音频")?;
    let probe = probe(path)?;
    let audio_streams = probe
        .streams
        .iter()
        .filter(|stream| stream.codec_type.as_deref() == Some("audio"))
        .collect::<Vec<_>>();
    if audio_streams.is_empty() {
        bail!("文件不包含可识别的音轨：{}", path.display());
    }
    if audio_streams.len() != 1 {
        bail!(
            "首版只接受恰好一条音轨，当前文件包含 {} 条",
            audio_streams.len()
        );
    }
    let stream = audio_streams[0];
    if probe.streams.iter().any(|stream| {
        stream.codec_type.as_deref() == Some("video") && stream.disposition.attached_pic == 0
    }) {
        bail!("默认转写入口只接受纯音频文件，不接受含视频流的容器");
    }

    let duration = exact_stream_duration(stream)
        .or_else(|| {
            stream
                .duration
                .as_deref()
                .and_then(|value| value.parse::<f64>().ok())
        })
        .or_else(|| {
            probe
                .format
                .duration
                .as_deref()
                .and_then(|value| value.parse::<f64>().ok())
        })
        .unwrap_or(0.0);
    if !duration.is_finite() || duration < 0.0 {
        bail!("音频时长字段无效");
    }
    let duration_ms = (duration * 1_000.0).round() as u64;

    Ok(AudioInfo {
        duration_ms,
        codec: stream
            .codec_name
            .clone()
            .unwrap_or_else(|| "unknown".into()),
        container: probe.format.format_name.unwrap_or_else(|| "unknown".into()),
    })
}

pub async fn validate_audio_async(path: &Path) -> Result<AudioInfo> {
    let path = path.to_owned();
    tokio::task::spawn_blocking(move || validate_audio(&path))
        .await
        .context("音频探测任务异常终止")?
}

pub fn validate_image(path: &Path) -> Result<ImageInfo> {
    validate_common(path, IMAGE_EXTENSIONS, "图片")?;
    let size = fs::metadata(path)
        .with_context(|| format!("无法读取图片信息：{}", path.display()))?
        .len();
    if size > 64 * 1024 * 1024 {
        bail!("OCR 图片超过 64 MiB 安全上限");
    }
    reject_animated_image(path)?;
    let probe = probe(path)?;
    let stream = probe
        .streams
        .iter()
        .find(|stream| stream.codec_type.as_deref() == Some("video"))
        .with_context(|| format!("文件不包含可识别的图像：{}", path.display()))?;
    let width = stream.width.context("无法读取图片宽度")?;
    let height = stream.height.context("无法读取图片高度")?;
    if width == 0 || height == 0 || width > 32_768 || height > 32_768 {
        bail!("图片尺寸无效或单边超过 32768 像素");
    }
    if width.saturating_mul(height) > 100_000_000 {
        bail!("图片总像素超过 1 亿安全上限");
    }
    if stream
        .nb_frames
        .as_deref()
        .and_then(|value| value.parse::<u64>().ok())
        .is_some_and(|frames| frames > 1)
    {
        bail!("OCR 首版只接受单帧图片");
    }

    Ok(ImageInfo {
        codec: stream
            .codec_name
            .clone()
            .unwrap_or_else(|| "unknown".into()),
        container: probe.format.format_name.unwrap_or_else(|| "unknown".into()),
        width,
        height,
    })
}

pub async fn validate_image_async(path: &Path) -> Result<ImageInfo> {
    let path = path.to_owned();
    tokio::task::spawn_blocking(move || validate_image(&path))
        .await
        .context("图片探测任务异常终止")?
}

fn reject_animated_image(path: &Path) -> Result<()> {
    let extension = path
        .extension()
        .and_then(OsStr::to_str)
        .map(str::to_ascii_lowercase)
        .unwrap_or_default();
    let animated = match extension.as_str() {
        "png" => png_is_animated(path)?,
        "webp" => webp_is_animated(path)?,
        _ => false,
    };
    if animated {
        bail!("OCR 首版不接受动画 PNG/WebP，请先导出单张静态图片");
    }
    Ok(())
}

fn png_is_animated(path: &Path) -> Result<bool> {
    let mut file = fs::File::open(path)
        .with_context(|| format!("无法检查 PNG 动画标记：{}", path.display()))?;
    let mut signature = [0_u8; 8];
    if file.read_exact(&mut signature).is_err() || signature != *b"\x89PNG\r\n\x1a\n" {
        return Ok(false);
    }
    loop {
        let mut length = [0_u8; 4];
        if file.read_exact(&mut length).is_err() {
            return Ok(false);
        }
        let length = u32::from_be_bytes(length) as u64;
        let mut kind = [0_u8; 4];
        file.read_exact(&mut kind).context("PNG chunk 头部不完整")?;
        if &kind == b"acTL" {
            return Ok(true);
        }
        if &kind == b"IDAT" || &kind == b"IEND" {
            return Ok(false);
        }
        file.seek(SeekFrom::Current((length + 4) as i64))
            .context("PNG chunk 长度无效")?;
    }
}

fn webp_is_animated(path: &Path) -> Result<bool> {
    let mut file = fs::File::open(path)
        .with_context(|| format!("无法检查 WebP 动画标记：{}", path.display()))?;
    let mut header = [0_u8; 12];
    if file.read_exact(&mut header).is_err() || &header[..4] != b"RIFF" || &header[8..] != b"WEBP" {
        return Ok(false);
    }
    loop {
        let mut kind = [0_u8; 4];
        if file.read_exact(&mut kind).is_err() {
            return Ok(false);
        }
        let mut length = [0_u8; 4];
        file.read_exact(&mut length)
            .context("WebP chunk 头部不完整")?;
        if &kind == b"ANIM" || &kind == b"ANMF" {
            return Ok(true);
        }
        let length = u32::from_le_bytes(length) as u64;
        let padded = length + (length % 2);
        file.seek(SeekFrom::Current(padded as i64))
            .context("WebP chunk 长度无效")?;
    }
}

pub fn markdown_output_path(input: &Path) -> Result<PathBuf> {
    validate_file_name(input)?;
    let canonical = fs::canonicalize(input)
        .with_context(|| format!("无法解析输入文件真实路径 {}", input.display()))?;
    Ok(canonical.with_extension("md"))
}

pub async fn canonicalize_audio(
    input: &Path,
    work_dir: &Path,
    max_temp_bytes: u64,
) -> Result<(PathBuf, AudioInfo)> {
    let output = work_dir.join("canonical.flac");
    let available = fs2::available_space(work_dir)
        .with_context(|| format!("无法读取临时目录可用空间 {}", work_dir.display()))?;
    let reserve = 256 * 1024 * 1024_u64;
    if available <= reserve + 64 * 1024 * 1024 {
        bail!("临时目录空间不足，无法安全生成无损母版");
    }
    let effective_cap = max_temp_bytes.min(available - reserve);
    let max_file_size = effective_cap.to_string();
    run_ffmpeg([
        OsStr::new("-hide_banner"),
        OsStr::new("-loglevel"),
        OsStr::new("error"),
        OsStr::new("-nostdin"),
        OsStr::new("-y"),
        OsStr::new("-i"),
        input.as_os_str(),
        OsStr::new("-map"),
        OsStr::new("0:a:0"),
        OsStr::new("-vn"),
        OsStr::new("-map_metadata"),
        OsStr::new("-1"),
        OsStr::new("-map_chapters"),
        OsStr::new("-1"),
        OsStr::new("-ac"),
        OsStr::new("1"),
        OsStr::new("-ar"),
        OsStr::new("32000"),
        OsStr::new("-sample_fmt"),
        OsStr::new("s16"),
        OsStr::new("-codec:a"),
        OsStr::new("flac"),
        OsStr::new("-compression_level"),
        OsStr::new("5"),
        OsStr::new("-fs"),
        OsStr::new(&max_file_size),
        output.as_os_str(),
    ])
    .await
    .context("无法生成无损音频母版")?;
    secure_file(&output)?;
    let canonical_size = fs::metadata(&output)
        .with_context(|| format!("无法读取无损母版大小 {}", output.display()))?
        .len();
    if canonical_size >= effective_cap.saturating_sub(1024 * 1024) {
        bail!("无损音频母版达到临时空间安全上限，已拒绝截断处理");
    }
    let info = validate_audio_async(&output)
        .await
        .context("无损音频母版校验失败")?;
    if info.duration_ms < 250 {
        bail!("音频时长必须至少为 0.25 秒");
    }
    Ok((output, info))
}

pub fn prepare_audio_chunks(
    input: &Path,
    info: &AudioInfo,
    config: &Config,
) -> Result<Vec<MediaChunk>> {
    let boundaries = plan_chunk_boundaries(
        info.duration_ms,
        config.chunk_seconds * 1_000,
        config.min_chunk_seconds * 1_000,
    );

    let canonical_size = fs::metadata(input)
        .with_context(|| format!("无法读取无损母版大小 {}", input.display()))?
        .len();
    let sample_with_silence_seconds = config
        .speaker_reference_seconds
        .saturating_add(config.speaker_reference_silence_seconds);
    let alignment_packet_seconds = config
        .overlap_seconds
        .saturating_add(config.speaker_reference_silence_seconds)
        .saturating_add(
            sample_with_silence_seconds
                .saturating_mul(config.max_speakers as u64)
                .saturating_mul(3),
        );
    let maximum_packet_ms = config
        .chunk_seconds
        .max(alignment_packet_seconds)
        .saturating_mul(1_000);
    let estimated_packet_size = maximum_packet_ms.saturating_mul(9);
    let projected_size = canonical_size.saturating_add(estimated_packet_size);
    if projected_size > config.max_temp_bytes {
        bail!(
            "预计临时空间 {:.2} GiB 超过 max_temp_bytes {:.2} GiB",
            projected_size as f64 / 1024_f64.powi(3),
            config.max_temp_bytes as f64 / 1024_f64.powi(3)
        );
    }
    let work_dir = input.parent().context("无损母版没有父目录")?;
    let available = fs2::available_space(work_dir)
        .with_context(|| format!("无法读取临时目录可用空间 {}", work_dir.display()))?;
    let additional_needed = estimated_packet_size.saturating_add(256 * 1024 * 1024);
    if available < additional_needed {
        bail!(
            "临时目录可用空间不足：需要约 {:.2} GiB，当前 {:.2} GiB",
            additional_needed as f64 / 1024_f64.powi(3),
            available as f64 / 1024_f64.powi(3)
        );
    }

    let overlap_ms = config.overlap_seconds * 1_000;
    let mut chunks = Vec::with_capacity(boundaries.len() - 1);
    for (index, window) in boundaries.windows(2).enumerate() {
        chunks.push(MediaChunk {
            source_path: input.to_owned(),
            audio_start_ms: window[0].saturating_sub(overlap_ms),
            start_ms: window[0],
            end_ms: window[1],
            lineage: format!("{index:06}"),
        });
    }
    Ok(chunks)
}

fn plan_chunk_boundaries(duration_ms: u64, chunk_ms: u64, minimum_tail_ms: u64) -> Vec<u64> {
    let mut boundaries = vec![0_u64];
    let mut boundary = chunk_ms;
    while boundary < duration_ms {
        let remaining = duration_ms - boundary;
        if remaining < minimum_tail_ms {
            let adjusted = duration_ms - minimum_tail_ms;
            if adjusted > *boundaries.last().unwrap_or(&0) {
                boundaries.push(adjusted);
            }
            break;
        }
        boundaries.push(boundary);
        boundary = boundary.saturating_add(chunk_ms);
    }
    boundaries.push(duration_ms);
    boundaries
}

pub fn split_audio_chunk(chunk: &MediaChunk, overlap_ms: u64) -> Result<(MediaChunk, MediaChunk)> {
    if chunk.duration_ms() < 2_000 {
        bail!("音频片段太短，无法继续二分");
    }
    let left_duration_ms = chunk.duration_ms() / 2;
    let midpoint_ms = chunk.start_ms + left_duration_ms;

    Ok((
        MediaChunk {
            source_path: chunk.source_path.clone(),
            audio_start_ms: chunk.start_ms.saturating_sub(overlap_ms),
            start_ms: chunk.start_ms,
            end_ms: midpoint_ms,
            lineage: format!("{}L", chunk.lineage),
        },
        MediaChunk {
            source_path: chunk.source_path.clone(),
            audio_start_ms: midpoint_ms.saturating_sub(overlap_ms),
            start_ms: midpoint_ms,
            end_ms: chunk.end_ms,
            lineage: format!("{}R", chunk.lineage),
        },
    ))
}

pub async fn build_exact_target_audio(chunk: &MediaChunk, output: &Path) -> Result<()> {
    encode_audio_range(
        &chunk.source_path,
        chunk.start_ms,
        chunk.duration_ms(),
        output,
    )
    .await
    .context("无法生成 exact TARGET 音频")?;
    validate_encoded_chunk(output, chunk.duration_ms()).await
}

pub async fn detect_non_silent_ranges(
    input: &Path,
    duration_ms: u64,
) -> Result<Vec<NonSilentRange>> {
    let stderr = run_ffmpeg_capture_stderr([
        OsStr::new("-hide_banner"),
        OsStr::new("-nostdin"),
        OsStr::new("-nostats"),
        OsStr::new("-loglevel"),
        OsStr::new("info"),
        OsStr::new("-i"),
        input.as_os_str(),
        OsStr::new("-af"),
        OsStr::new("silencedetect=noise=-30dB:d=1.0"),
        OsStr::new("-f"),
        OsStr::new("null"),
        OsStr::new("-"),
    ])
    .await
    .context("无法执行 TARGET 声学覆盖检查")?;
    Ok(parse_non_silent_ranges(&stderr, duration_ms))
}

pub async fn build_speaker_packet(
    chunk: &MediaChunk,
    references: &[SpeakerReferenceRange],
    candidates: &[SpeakerReferenceRange],
    silence_ms: u64,
    output: &Path,
) -> Result<SpeakerPacket> {
    if candidates.is_empty() {
        bail!("身份映射 packet 至少需要一个本片候选声音");
    }
    let mut reference_windows = Vec::with_capacity(references.len());
    let mut candidate_windows = Vec::with_capacity(candidates.len());
    let mut cursor_ms = 0_u64;
    let mut args = vec![
        OsString::from("-hide_banner"),
        OsString::from("-loglevel"),
        OsString::from("error"),
        OsString::from("-nostdin"),
        OsString::from("-y"),
    ];
    let mut input_count = 0_usize;
    for reference in references {
        append_packet_range(
            &mut args,
            &chunk.source_path,
            reference,
            &mut reference_windows,
            &mut cursor_ms,
            &mut input_count,
        )?;
        append_packet_silence(&mut args, silence_ms, &mut cursor_ms, &mut input_count);
    }

    let boundary_context = if chunk.context_ms() > 0 {
        let context = SpeakerReferenceRange {
            speaker_id: "BOUNDARY_CONTEXT".to_owned(),
            start_ms: chunk.audio_start_ms,
            end_ms: chunk.start_ms,
        };
        let mut windows = Vec::with_capacity(1);
        append_packet_range(
            &mut args,
            &chunk.source_path,
            &context,
            &mut windows,
            &mut cursor_ms,
            &mut input_count,
        )?;
        append_packet_silence(&mut args, silence_ms, &mut cursor_ms, &mut input_count);
        windows.pop()
    } else {
        None
    };

    for candidate in candidates {
        append_packet_range(
            &mut args,
            &chunk.source_path,
            candidate,
            &mut candidate_windows,
            &mut cursor_ms,
            &mut input_count,
        )?;
        append_packet_silence(&mut args, silence_ms, &mut cursor_ms, &mut input_count);
    }

    let concat_inputs = (0..input_count)
        .map(|index| format!("[{index}:a]"))
        .collect::<String>();
    let filter = format!("{concat_inputs}concat=n={input_count}:v=0:a=1[packet]");
    args.extend([
        OsString::from("-filter_complex"),
        OsString::from(filter),
        OsString::from("-map"),
        OsString::from("[packet]"),
        OsString::from("-map_metadata"),
        OsString::from("-1"),
        OsString::from("-map_chapters"),
        OsString::from("-1"),
        OsString::from("-ac"),
        OsString::from("1"),
        OsString::from("-ar"),
        OsString::from("32000"),
        OsString::from("-codec:a"),
        OsString::from("libmp3lame"),
        OsString::from("-b:a"),
        OsString::from("64k"),
        output.as_os_str().to_owned(),
    ]);
    run_ffmpeg(args)
        .await
        .context("无法生成 SpeakerHarness 身份映射音频包")?;

    validate_encoded_chunk(output, cursor_ms).await?;
    Ok(SpeakerPacket {
        path: output.to_owned(),
        references: reference_windows,
        boundary_context,
        candidates: candidate_windows,
        total_duration_ms: cursor_ms,
    })
}

fn append_packet_range(
    args: &mut Vec<OsString>,
    source_path: &Path,
    range: &SpeakerReferenceRange,
    windows: &mut Vec<PacketReferenceWindow>,
    cursor_ms: &mut u64,
    input_count: &mut usize,
) -> Result<()> {
    if range.end_ms <= range.start_ms {
        bail!("声音样本 {} 的音频范围无效", range.speaker_id);
    }
    let duration_ms = range.end_ms - range.start_ms;
    args.extend([
        OsString::from("-ss"),
        OsString::from(seconds_arg(range.start_ms)),
        OsString::from("-t"),
        OsString::from(seconds_arg(duration_ms)),
        OsString::from("-i"),
        source_path.as_os_str().to_owned(),
    ]);
    windows.push(PacketReferenceWindow {
        speaker_id: range.speaker_id.clone(),
        start_ms: *cursor_ms,
        end_ms: cursor_ms.saturating_add(duration_ms),
    });
    *cursor_ms = cursor_ms.saturating_add(duration_ms);
    *input_count += 1;
    Ok(())
}

fn append_packet_silence(
    args: &mut Vec<OsString>,
    silence_ms: u64,
    cursor_ms: &mut u64,
    input_count: &mut usize,
) {
    args.extend([
        OsString::from("-f"),
        OsString::from("lavfi"),
        OsString::from("-t"),
        OsString::from(seconds_arg(silence_ms)),
        OsString::from("-i"),
        OsString::from("anullsrc=r=32000:cl=mono"),
    ]);
    *cursor_ms = cursor_ms.saturating_add(silence_ms);
    *input_count += 1;
}

pub async fn normalize_image(input: &Path, work_dir: &Path) -> Result<PathBuf> {
    let output = work_dir.join("ocr-input.jpg");
    run_ffmpeg([
        OsStr::new("-hide_banner"),
        OsStr::new("-loglevel"),
        OsStr::new("error"),
        OsStr::new("-nostdin"),
        OsStr::new("-y"),
        OsStr::new("-i"),
        input.as_os_str(),
        OsStr::new("-filter_complex"),
        OsStr::new(
            "[0:v:0]scale='min(4096,iw)':'min(4096,ih)':force_original_aspect_ratio=decrease,format=rgba,split=2[background][foreground];[background]drawbox=color=white:t=fill[white];[white][foreground]overlay=format=auto,format=rgb24[output]",
        ),
        OsStr::new("-map"),
        OsStr::new("[output]"),
        OsStr::new("-frames:v"),
        OsStr::new("1"),
        OsStr::new("-map_metadata"),
        OsStr::new("-1"),
        OsStr::new("-q:v"),
        OsStr::new("2"),
        output.as_os_str(),
    ])
    .await
    .context("OCR 图片标准化失败")?;
    secure_file(&output)?;
    Ok(output)
}

async fn encode_audio_range(
    input: &Path,
    offset_ms: u64,
    duration_ms: u64,
    output: &Path,
) -> Result<()> {
    let offset = seconds_arg(offset_ms);
    let duration = seconds_arg(duration_ms);
    run_ffmpeg([
        OsStr::new("-hide_banner"),
        OsStr::new("-loglevel"),
        OsStr::new("error"),
        OsStr::new("-nostdin"),
        OsStr::new("-y"),
        OsStr::new("-ss"),
        OsStr::new(&offset),
        OsStr::new("-i"),
        input.as_os_str(),
        OsStr::new("-t"),
        OsStr::new(&duration),
        OsStr::new("-map"),
        OsStr::new("0:a:0"),
        OsStr::new("-vn"),
        OsStr::new("-map_metadata"),
        OsStr::new("-1"),
        OsStr::new("-map_chapters"),
        OsStr::new("-1"),
        OsStr::new("-ac"),
        OsStr::new("1"),
        OsStr::new("-ar"),
        OsStr::new("32000"),
        OsStr::new("-codec:a"),
        OsStr::new("libmp3lame"),
        OsStr::new("-b:a"),
        OsStr::new("64k"),
        output.as_os_str(),
    ])
    .await
    .with_context(|| format!("无法二分临时音频片段 {}", input.display()))
}

async fn validate_encoded_chunk(path: &Path, expected_duration_ms: u64) -> Result<()> {
    secure_file(path)?;
    let actual = validate_audio_async(path)
        .await
        .with_context(|| format!("生成的临时音频无效：{}", path.display()))?;
    if actual.duration_ms.abs_diff(expected_duration_ms) > 80 {
        bail!(
            "临时音频时长异常：预计 {:.3} 秒，实际 {:.3} 秒",
            expected_duration_ms as f64 / 1_000.0,
            actual.duration_ms as f64 / 1_000.0
        );
    }
    Ok(())
}

fn exact_stream_duration(stream: &ProbeStream) -> Option<f64> {
    let ticks = stream.duration_ts?;
    if ticks <= 0 {
        return None;
    }
    let (numerator, denominator) = stream.time_base.as_deref()?.split_once('/')?;
    let numerator = numerator.parse::<f64>().ok()?;
    let denominator = denominator.parse::<f64>().ok()?;
    if denominator <= 0.0 {
        return None;
    }
    Some(ticks as f64 * numerator / denominator)
}

fn validate_common(path: &Path, allowlist: &[&str], label: &str) -> Result<()> {
    validate_file_name(path)?;
    let metadata = fs::symlink_metadata(path)
        .with_context(|| format!("无法读取{label}文件：{}", path.display()))?;
    if metadata.file_type().is_symlink() {
        bail!("为避免输出位置歧义，不接受符号链接：{}", path.display());
    }
    if !metadata.file_type().is_file() {
        bail!("路径不是普通文件：{}", path.display());
    }
    if metadata.len() == 0 {
        bail!("文件为空：{}", path.display());
    }

    let extension = path
        .extension()
        .and_then(OsStr::to_str)
        .map(str::to_ascii_lowercase)
        .with_context(|| format!("{label}文件必须带有合法扩展名"))?;
    if !allowlist.contains(&extension.as_str()) {
        bail!(
            "不支持的{label}格式 .{extension}；允许：{}",
            allowlist.join(", ")
        );
    }
    Ok(())
}

fn validate_file_name(path: &Path) -> Result<()> {
    let file_name = path
        .file_name()
        .and_then(OsStr::to_str)
        .context("文件名必须是有效的 UTF-8 文本")?;
    if file_name.is_empty() || file_name.chars().any(char::is_control) {
        bail!("文件名为空或包含控制字符");
    }
    let stem = path
        .file_stem()
        .and_then(OsStr::to_str)
        .context("文件名必须包含非空主文件名")?;
    if stem.is_empty() || stem == "." || stem == ".." {
        bail!("文件名必须包含非空主文件名");
    }
    Ok(())
}

fn probe(path: &Path) -> Result<ProbeOutput> {
    let mut child = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-show_entries",
            "format=duration,format_name:stream=codec_type,codec_name,duration,duration_ts,time_base,width,height,nb_frames:stream_disposition=attached_pic",
            "-of",
            "json",
            "-i",
        ])
        .arg(path)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| format!("无法启动 ffprobe 检查 {}", path.display()))?;
    let stdout = child.stdout.take().context("无法读取 ffprobe stdout")?;
    let stderr = child.stderr.take().context("无法读取 ffprobe stderr")?;
    let overflow = Arc::new(AtomicBool::new(false));
    let stdout_overflow = Arc::clone(&overflow);
    let stderr_overflow = Arc::clone(&overflow);
    let stdout_reader =
        thread::spawn(move || read_bounded_sync(stdout, PROBE_STDOUT_LIMIT, stdout_overflow));
    let stderr_reader =
        thread::spawn(move || read_bounded_sync(stderr, CHILD_STDERR_LIMIT, stderr_overflow));
    let deadline = Instant::now() + PROBE_TIMEOUT;
    let mut timed_out = false;
    let status = loop {
        if overflow.load(Ordering::Relaxed) {
            break None;
        }
        match child.try_wait() {
            Ok(Some(status)) => break Some(status),
            Ok(None) => {}
            Err(error) => {
                let _ = child.kill();
                let _ = child.wait();
                let _ = stdout_reader.join();
                let _ = stderr_reader.join();
                return Err(error).context("无法等待 ffprobe");
            }
        }
        if Instant::now() >= deadline {
            timed_out = true;
            break None;
        }
        thread::sleep(Duration::from_millis(10));
    };
    if status.is_none() {
        let _ = child.kill();
        let _ = child.wait();
    }
    let stdout = stdout_reader
        .join()
        .map_err(|_| anyhow::anyhow!("ffprobe stdout 读取线程异常"))??;
    let stderr = stderr_reader
        .join()
        .map_err(|_| anyhow::anyhow!("ffprobe stderr 读取线程异常"))??;
    if timed_out {
        bail!("ffprobe 运行超过 10 秒，已终止");
    }
    if overflow.load(Ordering::Relaxed) {
        bail!("ffprobe 输出超过安全上限，已终止");
    }
    let status = status.context("ffprobe 未返回退出状态")?;
    if !status.success() {
        let detail = bounded_stderr(&stderr);
        bail!("文件内容不是合法媒体或已经损坏：{detail}");
    }
    serde_json::from_slice(&stdout).context("无法解析 ffprobe 输出")
}

fn ensure_tool(tool: &str) -> Result<()> {
    let mut child = Command::new(tool)
        .arg("-version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .with_context(|| format!("未找到 {tool}；请先安装 FFmpeg"))?;
    let deadline = Instant::now() + Duration::from_secs(5);
    let status = loop {
        if let Some(status) = child
            .try_wait()
            .with_context(|| format!("无法等待 {tool}"))?
        {
            break status;
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            bail!("{tool} 版本检查超时");
        }
        thread::sleep(Duration::from_millis(10));
    };
    if !status.success() {
        bail!("{tool} 无法正常运行");
    }
    Ok(())
}

async fn run_ffmpeg<I, S>(args: I) -> Result<()>
where
    I: IntoIterator<Item = S>,
    S: AsRef<OsStr>,
{
    run_ffmpeg_capture_stderr(args).await.map(|_| ())
}

async fn run_ffmpeg_capture_stderr<I, S>(args: I) -> Result<Vec<u8>>
where
    I: IntoIterator<Item = S>,
    S: AsRef<OsStr>,
{
    let mut command = TokioCommand::new("ffmpeg");
    command
        .args(args)
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    let mut child = command.spawn().context("无法启动 ffmpeg")?;
    let stderr = child.stderr.take().context("无法读取 ffmpeg stderr")?;
    let outcome = tokio::time::timeout(FFMPEG_TIMEOUT, async {
        let (status, stderr) = tokio::try_join!(
            async { child.wait().await.context("无法等待 ffmpeg") },
            read_bounded_async(stderr, CHILD_STDERR_LIMIT),
        )?;
        Ok::<_, anyhow::Error>((status, stderr))
    })
    .await;
    let (status, stderr) = match outcome {
        Ok(Ok(outcome)) => outcome,
        Ok(Err(error)) => {
            let _ = child.kill().await;
            let _ = child.wait().await;
            return Err(error);
        }
        Err(_) => {
            let _ = child.kill().await;
            let _ = child.wait().await;
            bail!("FFmpeg 运行超过 1 小时，已终止");
        }
    };
    if !status.success() {
        bail!("FFmpeg 执行失败：{}", bounded_stderr(&stderr));
    }
    Ok(stderr)
}

fn parse_non_silent_ranges(stderr: &[u8], duration_ms: u64) -> Vec<NonSilentRange> {
    let diagnostics = String::from_utf8_lossy(stderr);
    let mut silence_start_ms = None;
    let mut silence_ranges = Vec::<NonSilentRange>::new();
    for line in diagnostics.lines() {
        if let Some(seconds) = parse_silence_marker(line, "silence_start:") {
            silence_start_ms = Some(seconds_to_milliseconds(seconds, duration_ms));
        }
        if let Some(seconds) = parse_silence_marker(line, "silence_end:") {
            let end_ms = seconds_to_milliseconds(seconds, duration_ms);
            let start_ms = silence_start_ms.take().unwrap_or(0).min(end_ms);
            silence_ranges.push(NonSilentRange { start_ms, end_ms });
        }
    }
    if let Some(start_ms) = silence_start_ms {
        silence_ranges.push(NonSilentRange {
            start_ms,
            end_ms: duration_ms,
        });
    }
    silence_ranges.sort_by_key(|range| (range.start_ms, range.end_ms));

    let mut merged_silence = Vec::<NonSilentRange>::new();
    for range in silence_ranges {
        if range.end_ms <= range.start_ms {
            continue;
        }
        if let Some(previous) = merged_silence.last_mut()
            && range.start_ms <= previous.end_ms
        {
            previous.end_ms = previous.end_ms.max(range.end_ms);
        } else {
            merged_silence.push(range);
        }
    }

    let mut cursor_ms = 0_u64;
    let mut non_silent = Vec::new();
    for silence in merged_silence {
        if silence.start_ms > cursor_ms {
            non_silent.push(NonSilentRange {
                start_ms: cursor_ms,
                end_ms: silence.start_ms,
            });
        }
        cursor_ms = cursor_ms.max(silence.end_ms);
    }
    if cursor_ms < duration_ms {
        non_silent.push(NonSilentRange {
            start_ms: cursor_ms,
            end_ms: duration_ms,
        });
    }
    non_silent
}

fn parse_silence_marker(line: &str, marker: &str) -> Option<f64> {
    let value = line.split_once(marker)?.1.split_whitespace().next()?;
    value
        .parse::<f64>()
        .ok()
        .filter(|seconds| seconds.is_finite() && *seconds >= 0.0)
}

fn seconds_to_milliseconds(seconds: f64, duration_ms: u64) -> u64 {
    (seconds * 1_000.0).round().clamp(0.0, duration_ms as f64) as u64
}

fn read_bounded_sync<R: Read>(
    mut reader: R,
    limit: usize,
    overflow: Arc<AtomicBool>,
) -> std::io::Result<Vec<u8>> {
    let mut output = Vec::with_capacity(limit.min(8 * 1024));
    let mut buffer = [0_u8; 8 * 1024];
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        if output.len().saturating_add(read) > limit {
            overflow.store(true, Ordering::Relaxed);
            break;
        }
        output.extend_from_slice(&buffer[..read]);
    }
    Ok(output)
}

async fn read_bounded_async<R: AsyncRead + Unpin>(mut reader: R, limit: usize) -> Result<Vec<u8>> {
    let mut output = Vec::with_capacity(limit.min(8 * 1024));
    let mut buffer = [0_u8; 8 * 1024];
    loop {
        let read = reader
            .read(&mut buffer)
            .await
            .context("无法读取子进程输出")?;
        if read == 0 {
            break;
        }
        if output.len().saturating_add(read) > limit {
            bail!("子进程诊断输出超过 256 KiB 安全上限");
        }
        output.extend_from_slice(&buffer[..read]);
    }
    Ok(output)
}

fn seconds_arg(milliseconds: u64) -> String {
    format!("{}.{:03}", milliseconds / 1_000, milliseconds % 1_000)
}

fn bounded_stderr(bytes: &[u8]) -> String {
    let text = String::from_utf8_lossy(bytes);
    let trimmed = text.trim();
    let mut chars = trimmed.chars();
    let prefix = chars.by_ref().take(2_000).collect::<String>();
    if chars.next().is_some() {
        format!("{prefix}…")
    } else {
        prefix
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn seconds_are_stable_and_millisecond_precise() {
        assert_eq!(seconds_arg(0), "0.000");
        assert_eq!(seconds_arg(61_234), "61.234");
    }

    #[test]
    fn silence_diagnostics_are_complemented_into_activity_ranges() {
        let diagnostics = b"[silencedetect] silence_start: 2.0\n\
[silencedetect] silence_end: 4.5 | silence_duration: 2.5\n\
[silencedetect] silence_start: 8.0\n";
        assert_eq!(
            parse_non_silent_ranges(diagnostics, 10_000),
            vec![
                NonSilentRange {
                    start_ms: 0,
                    end_ms: 2_000,
                },
                NonSilentRange {
                    start_ms: 4_500,
                    end_ms: 8_000,
                },
            ]
        );
    }

    #[test]
    fn renamed_text_is_not_accepted_as_audio() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("notes.mp3");
        fs::File::create(&path)
            .unwrap()
            .write_all(b"not audio")
            .unwrap();
        assert!(validate_audio(&path).is_err());
    }

    #[test]
    fn arbitrary_extension_is_rejected_before_probe() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("notes.txt");
        fs::write(&path, b"hello").unwrap();
        let error = validate_audio(&path).unwrap_err().to_string();
        assert!(error.contains("不支持的音频格式"));
    }

    #[test]
    fn short_tail_is_shifted_to_minimum_size_without_tiny_segments() {
        assert_eq!(
            plan_chunk_boundaries(30_001, 30_000, 10_000),
            vec![0, 20_001, 30_001]
        );
        assert_eq!(
            plan_chunk_boundaries(60_001, 30_000, 10_000),
            vec![0, 30_000, 50_001, 60_001]
        );
        assert_eq!(
            plan_chunk_boundaries(620_000, 300_000, 30_000),
            vec![0, 300_000, 590_000, 620_000]
        );
        assert_eq!(
            plan_chunk_boundaries(920_000, 900_000, 30_000),
            vec![0, 890_000, 920_000]
        );
        assert_eq!(
            plan_chunk_boundaries(900_500, 900_000, 30_000),
            vec![0, 870_500, 900_500]
        );
    }

    #[test]
    fn exact_duration_uses_stream_ticks() {
        let stream = ProbeStream {
            codec_type: Some("audio".into()),
            codec_name: Some("flac".into()),
            duration: None,
            duration_ts: Some(1_920_000),
            time_base: Some("1/32000".into()),
            width: None,
            height: None,
            nb_frames: None,
            disposition: ProbeDisposition::default(),
        };
        assert_eq!(exact_stream_duration(&stream), Some(60.0));
    }

    #[test]
    fn adaptive_split_keeps_disjoint_targets_and_leading_context() {
        let parent = MediaChunk {
            source_path: "canonical.flac".into(),
            audio_start_ms: 870_000,
            start_ms: 900_000,
            end_ms: 1_800_000,
            lineage: "000001".into(),
        };
        let (left, right) = split_audio_chunk(&parent, 30_000).unwrap();
        assert_eq!((left.start_ms, left.end_ms), (900_000, 1_350_000));
        assert_eq!((right.start_ms, right.end_ms), (1_350_000, 1_800_000));
        assert_eq!(left.audio_start_ms, 870_000);
        assert_eq!(right.audio_start_ms, 1_320_000);
        assert_eq!(left.context_ms(), 30_000);
        assert_eq!(right.context_ms(), 30_000);
    }

    #[tokio::test]
    async fn speaker_packet_manifest_matches_encoded_audio() {
        let directory = tempfile::tempdir().unwrap();
        let canonical = directory.path().join("canonical.flac");
        let status = Command::new("ffmpeg")
            .args([
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:sample_rate=32000:duration=20",
                "-ac",
                "1",
                "-codec:a",
                "flac",
            ])
            .arg(&canonical)
            .status()
            .unwrap();
        assert!(status.success());
        let chunk = MediaChunk {
            source_path: canonical,
            audio_start_ms: 5_000,
            start_ms: 10_000,
            end_ms: 20_000,
            lineage: "packet-test".into(),
        };
        let output = directory.path().join("packet.mp3");
        let packet = build_speaker_packet(
            &chunk,
            &[SpeakerReferenceRange {
                speaker_id: "S1".into(),
                start_ms: 0,
                end_ms: 3_000,
            }],
            &[SpeakerReferenceRange {
                speaker_id: "L1".into(),
                start_ms: 12_000,
                end_ms: 15_000,
            }],
            1_000,
            &output,
        )
        .await
        .unwrap();
        assert_eq!(packet.references[0].start_ms, 0);
        assert_eq!(packet.references[0].end_ms, 3_000);
        assert_eq!(packet.boundary_context.as_ref().unwrap().start_ms, 4_000);
        assert_eq!(packet.boundary_context.as_ref().unwrap().end_ms, 9_000);
        assert_eq!(packet.candidates[0].start_ms, 10_000);
        assert_eq!(packet.candidates[0].end_ms, 13_000);
        assert_eq!(packet.total_duration_ms, 14_000);

        let exact_target = directory.path().join("exact-target.mp3");
        build_exact_target_audio(&chunk, &exact_target)
            .await
            .unwrap();
        let exact_info = validate_audio(&exact_target).unwrap();
        assert_eq!(exact_info.duration_ms, 10_000);

        let first_chunk = MediaChunk {
            source_path: chunk.source_path.clone(),
            audio_start_ms: 0,
            start_ms: 0,
            end_ms: 10_000,
            lineage: "first-identity".into(),
        };
        let first_packet_path = directory.path().join("first-identity.mp3");
        let first_packet = build_speaker_packet(
            &first_chunk,
            &[],
            &[SpeakerReferenceRange {
                speaker_id: "L1".into(),
                start_ms: 0,
                end_ms: 3_000,
            }],
            1_000,
            &first_packet_path,
        )
        .await
        .unwrap();
        assert!(first_packet.references.is_empty());
        assert!(first_packet.boundary_context.is_none());
        assert_eq!(first_packet.candidates[0].start_ms, 0);
        assert_eq!(first_packet.candidates[0].end_ms, 3_000);
        assert_eq!(first_packet.total_duration_ms, 4_000);
    }
}
