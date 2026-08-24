use std::ffi::OsStr;
use std::fs;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use tokio::process::Command as TokioCommand;

use crate::config::Config;

const AUDIO_EXTENSIONS: &[&str] = &[
    "aac", "aif", "aiff", "caf", "flac", "m4a", "m4b", "mp3", "oga", "ogg", "opus", "wav", "webm",
    "wma",
];
const IMAGE_EXTENSIONS: &[&str] = &["jpeg", "jpg", "png", "webp"];

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
    pub path: PathBuf,
    pub source_path: PathBuf,
    pub start_ms: u64,
    pub end_ms: u64,
    pub lineage: String,
}

impl MediaChunk {
    pub fn duration_ms(&self) -> u64 {
        self.end_ms.saturating_sub(self.start_ms)
    }
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
    Ok(input.with_extension("md"))
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
    secure_temporary_media(&output)?;
    let canonical_size = fs::metadata(&output)
        .with_context(|| format!("无法读取无损母版大小 {}", output.display()))?
        .len();
    if canonical_size >= effective_cap.saturating_sub(1024 * 1024) {
        bail!("无损音频母版达到临时空间安全上限，已拒绝截断处理");
    }
    let info = validate_audio(&output).context("无损音频母版校验失败")?;
    if info.duration_ms < 250 {
        bail!("音频时长必须至少为 0.25 秒");
    }
    Ok((output, info))
}

pub async fn prepare_audio_chunks(
    input: &Path,
    info: &AudioInfo,
    config: &Config,
    work_dir: &Path,
) -> Result<Vec<MediaChunk>> {
    let boundaries = plan_chunk_boundaries(
        info.duration_ms,
        config.chunk_seconds * 1_000,
        config.min_chunk_seconds * 1_000,
    );

    let canonical_size = fs::metadata(input)
        .with_context(|| format!("无法读取无损母版大小 {}", input.display()))?
        .len();
    let estimated_mp3_size = info.duration_ms.saturating_mul(9);
    let projected_size = canonical_size.saturating_add(estimated_mp3_size);
    if projected_size > config.max_temp_bytes {
        bail!(
            "预计临时空间 {:.2} GiB 超过 max_temp_bytes {:.2} GiB",
            projected_size as f64 / 1024_f64.powi(3),
            config.max_temp_bytes as f64 / 1024_f64.powi(3)
        );
    }
    let available = fs2::available_space(work_dir)
        .with_context(|| format!("无法读取临时目录可用空间 {}", work_dir.display()))?;
    let additional_needed = estimated_mp3_size.saturating_add(256 * 1024 * 1024);
    if available < additional_needed {
        bail!(
            "临时目录可用空间不足：需要约 {:.2} GiB，当前 {:.2} GiB",
            additional_needed as f64 / 1024_f64.powi(3),
            available as f64 / 1024_f64.powi(3)
        );
    }

    let mut chunks = Vec::with_capacity(boundaries.len() - 1);
    let mut used_bytes = canonical_size;
    for (index, window) in boundaries.windows(2).enumerate() {
        let path = work_dir.join(format!("chunk_{index:06}.mp3"));
        let duration_ms = window[1] - window[0];
        encode_audio_range(input, window[0], duration_ms, &path)
            .await
            .with_context(|| format!("无法生成第 {} 个音频片段", index + 1))?;
        validate_encoded_chunk(&path, duration_ms)?;
        used_bytes = used_bytes.saturating_add(
            fs::metadata(&path)
                .with_context(|| format!("无法读取临时片段大小 {}", path.display()))?
                .len(),
        );
        if used_bytes > config.max_temp_bytes {
            bail!("临时媒体超过 max_temp_bytes 安全上限");
        }
        chunks.push(MediaChunk {
            path,
            source_path: input.to_owned(),
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
        if duration_ms - boundary < minimum_tail_ms {
            break;
        }
        boundaries.push(boundary);
        boundary = boundary.saturating_add(chunk_ms);
    }
    boundaries.push(duration_ms);
    boundaries
}

pub async fn split_audio_chunk(chunk: &MediaChunk) -> Result<(MediaChunk, MediaChunk)> {
    if chunk.duration_ms() < 2_000 {
        bail!("音频片段太短，无法继续二分");
    }
    let left_duration_ms = chunk.duration_ms() / 2;
    let right_duration_ms = chunk.duration_ms() - left_duration_ms;
    let midpoint_ms = chunk.start_ms + left_duration_ms;
    let parent = chunk.path.parent().context("临时音频片段没有父目录")?;
    let left_path = parent.join(format!("adaptive_{}L.mp3", chunk.lineage));
    let right_path = parent.join(format!("adaptive_{}R.mp3", chunk.lineage));

    encode_audio_range(
        &chunk.source_path,
        chunk.start_ms,
        left_duration_ms,
        &left_path,
    )
    .await?;
    validate_encoded_chunk(&left_path, left_duration_ms)?;
    encode_audio_range(
        &chunk.source_path,
        midpoint_ms,
        right_duration_ms,
        &right_path,
    )
    .await?;
    validate_encoded_chunk(&right_path, right_duration_ms)?;

    Ok((
        MediaChunk {
            path: left_path,
            source_path: chunk.source_path.clone(),
            start_ms: chunk.start_ms,
            end_ms: midpoint_ms,
            lineage: format!("{}L", chunk.lineage),
        },
        MediaChunk {
            path: right_path,
            source_path: chunk.source_path.clone(),
            start_ms: midpoint_ms,
            end_ms: chunk.end_ms,
            lineage: format!("{}R", chunk.lineage),
        },
    ))
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
    secure_temporary_media(&output)?;
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

fn validate_encoded_chunk(path: &Path, expected_duration_ms: u64) -> Result<()> {
    secure_temporary_media(path)?;
    let actual =
        validate_audio(path).with_context(|| format!("生成的临时音频无效：{}", path.display()))?;
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

#[cfg(unix)]
fn secure_temporary_media(path: &Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(path, fs::Permissions::from_mode(0o600))
        .with_context(|| format!("无法设置临时媒体权限 {}", path.display()))
}

#[cfg(not(unix))]
fn secure_temporary_media(_path: &Path) -> Result<()> {
    Ok(())
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
    let output = Command::new("ffprobe")
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
        .output()
        .with_context(|| format!("无法启动 ffprobe 检查 {}", path.display()))?;
    if !output.status.success() {
        let detail = bounded_stderr(&output.stderr);
        bail!("文件内容不是合法媒体或已经损坏：{detail}");
    }
    serde_json::from_slice(&output.stdout).context("无法解析 ffprobe 输出")
}

fn ensure_tool(tool: &str) -> Result<()> {
    let status = Command::new(tool)
        .arg("-version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .with_context(|| format!("未找到 {tool}；请先安装 FFmpeg"))?;
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
    let mut command = TokioCommand::new("ffmpeg");
    command.args(args).kill_on_drop(true);
    let output = tokio::time::timeout(std::time::Duration::from_secs(3_600), command.output())
        .await
        .context("FFmpeg 运行超过 1 小时，已终止")?
        .context("无法启动 ffmpeg")?;
    if !output.status.success() {
        bail!("FFmpeg 执行失败：{}", bounded_stderr(&output.stderr));
    }
    Ok(())
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
    fn short_tail_is_merged_instead_of_becoming_empty_segment() {
        assert_eq!(
            plan_chunk_boundaries(30_001, 30_000, 10_000),
            vec![0, 30_001]
        );
        assert_eq!(
            plan_chunk_boundaries(60_001, 30_000, 10_000),
            vec![0, 30_000, 60_001]
        );
        assert_eq!(
            plan_chunk_boundaries(620_000, 300_000, 30_000),
            vec![0, 300_000, 620_000]
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
}
