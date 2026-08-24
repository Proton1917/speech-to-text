use std::env;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use fs2::FileExt;
use serde::{Deserialize, Serialize};
use tempfile::NamedTempFile;

use crate::security::{secure_directory, secure_file};

pub const DEFAULT_MODEL: &str = "google/gemini-3.5-flash-lite";
pub const DEFAULT_PROVIDER: &str = "google-vertex/global";
pub const ANY_PROVIDER: &str = "any";

pub struct ConfigLock {
    _file: fs::File,
}

impl Drop for ConfigLock {
    fn drop(&mut self) {
        let _ = FileExt::unlock(&self._file);
    }
}

impl ConfigLock {
    pub fn acquire(path: &Path) -> Result<Self> {
        let parent = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        let parent_existed = parent.exists();
        fs::create_dir_all(parent)
            .with_context(|| format!("无法创建配置目录 {}", parent.display()))?;
        if !parent_existed {
            secure_directory(parent)?;
        }
        let lock_path = parent.join(".config.lock");
        let file = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&lock_path)
            .with_context(|| format!("无法打开配置锁 {}", lock_path.display()))?;
        secure_file(&lock_path)?;
        match file.try_lock_exclusive() {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                bail!("另一个 spt 进程正在更新配置，请稍后重试")
            }
            Err(error) => {
                return Err(error)
                    .with_context(|| format!("无法获取配置锁 {}", lock_path.display()));
            }
        }
        Ok(Self { _file: file })
    }
}

const fn default_schema_version() -> u32 {
    2
}

fn default_model() -> String {
    DEFAULT_MODEL.to_owned()
}

fn default_provider() -> String {
    DEFAULT_PROVIDER.to_owned()
}

const fn default_chunk_seconds() -> u64 {
    900
}

const fn default_overlap_seconds() -> u64 {
    30
}

const fn default_min_chunk_seconds() -> u64 {
    30
}

const fn default_max_output_tokens() -> u32 {
    16_000
}

const fn default_split_output_tokens() -> u32 {
    12_000
}

const fn default_parallel_requests() -> usize {
    1
}

const fn default_retries() -> u32 {
    5
}

const fn default_max_adaptive_depth() -> u8 {
    4
}

const fn default_max_http_attempts() -> u32 {
    1_000
}

const fn default_max_temp_bytes() -> u64 {
    20 * 1024 * 1024 * 1024
}

const fn default_max_speakers() -> usize {
    16
}

const fn default_speaker_reference_seconds() -> u64 {
    6
}

const fn default_speaker_reference_silence_seconds() -> u64 {
    1
}

const fn default_speaker_context_chars() -> usize {
    4_000
}

const fn default_max_transcript_bytes() -> u64 {
    64 * 1024 * 1024
}

const fn default_max_total_turns() -> u64 {
    100_000
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub schema_version: u32,
    pub model: String,
    pub provider: String,
    pub chunk_seconds: u64,
    pub overlap_seconds: u64,
    pub min_chunk_seconds: u64,
    pub max_output_tokens: u32,
    pub split_output_tokens: u32,
    pub parallel_requests: usize,
    pub retries: u32,
    pub max_adaptive_depth: u8,
    pub max_http_attempts: u32,
    pub max_temp_bytes: u64,
    pub max_speakers: usize,
    pub speaker_reference_seconds: u64,
    pub speaker_reference_silence_seconds: u64,
    pub speaker_context_chars: usize,
    pub max_transcript_bytes: u64,
    pub max_total_turns: u64,
}

#[derive(Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct LegacyConfigV1 {
    schema_version: u32,
    model: String,
    provider: String,
    chunk_seconds: u64,
    min_chunk_seconds: u64,
    max_output_tokens: u32,
    split_output_tokens: u32,
    parallel_requests: usize,
    retries: u32,
    max_adaptive_depth: u8,
    max_http_attempts: u32,
    max_temp_bytes: u64,
}

impl Default for LegacyConfigV1 {
    fn default() -> Self {
        Self {
            schema_version: 1,
            model: default_model(),
            provider: default_provider(),
            chunk_seconds: 300,
            min_chunk_seconds: default_min_chunk_seconds(),
            max_output_tokens: 6_000,
            split_output_tokens: 5_000,
            parallel_requests: 3,
            retries: default_retries(),
            max_adaptive_depth: default_max_adaptive_depth(),
            max_http_attempts: default_max_http_attempts(),
            max_temp_bytes: default_max_temp_bytes(),
        }
    }
}

impl From<LegacyConfigV1> for Config {
    fn from(legacy: LegacyConfigV1) -> Self {
        Self {
            schema_version: legacy.schema_version,
            model: legacy.model,
            provider: legacy.provider,
            chunk_seconds: legacy.chunk_seconds,
            min_chunk_seconds: legacy.min_chunk_seconds,
            max_output_tokens: legacy.max_output_tokens,
            split_output_tokens: legacy.split_output_tokens,
            parallel_requests: legacy.parallel_requests,
            retries: legacy.retries,
            max_adaptive_depth: legacy.max_adaptive_depth,
            max_http_attempts: legacy.max_http_attempts,
            max_temp_bytes: legacy.max_temp_bytes,
            ..Config::default()
        }
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            schema_version: default_schema_version(),
            model: default_model(),
            provider: default_provider(),
            chunk_seconds: default_chunk_seconds(),
            overlap_seconds: default_overlap_seconds(),
            min_chunk_seconds: default_min_chunk_seconds(),
            max_output_tokens: default_max_output_tokens(),
            split_output_tokens: default_split_output_tokens(),
            parallel_requests: default_parallel_requests(),
            retries: default_retries(),
            max_adaptive_depth: default_max_adaptive_depth(),
            max_http_attempts: default_max_http_attempts(),
            max_temp_bytes: default_max_temp_bytes(),
            max_speakers: default_max_speakers(),
            speaker_reference_seconds: default_speaker_reference_seconds(),
            speaker_reference_silence_seconds: default_speaker_reference_silence_seconds(),
            speaker_context_chars: default_speaker_context_chars(),
            max_transcript_bytes: default_max_transcript_bytes(),
            max_total_turns: default_max_total_turns(),
        }
    }
}

impl Config {
    pub fn load() -> Result<(Self, PathBuf, bool, bool)> {
        let path = config_path()?;
        let metadata = match fs::symlink_metadata(&path) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Ok((Self::default(), path, false, false));
            }
            Err(error) => {
                return Err(error)
                    .with_context(|| format!("无法读取配置文件元数据 {}", path.display()));
            }
        };
        if metadata.file_type().is_symlink() || !metadata.file_type().is_file() {
            bail!("配置路径不是普通文件：{}", path.display());
        }

        let raw = fs::read_to_string(&path)
            .with_context(|| format!("无法读取配置文件 {}", path.display()))?;
        let (config, migrated) =
            decode_config(&raw).with_context(|| format!("配置文件格式无效：{}", path.display()))?;
        Ok((config, path, true, migrated))
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        self.validate()?;
        let parent = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        let parent_existed = parent.exists();
        fs::create_dir_all(parent)
            .with_context(|| format!("无法创建配置目录 {}", parent.display()))?;
        if !parent_existed {
            secure_directory(parent)?;
        }

        let encoded = toml::to_string_pretty(self).context("无法序列化配置")?;
        let mut temporary = NamedTempFile::new_in(parent)
            .with_context(|| format!("无法在 {} 创建临时配置", parent.display()))?;
        secure_file(temporary.path())?;
        temporary
            .write_all(encoded.as_bytes())
            .context("无法写入临时配置")?;
        temporary.flush().context("无法刷新临时配置")?;
        temporary.as_file().sync_all().context("无法同步临时配置")?;
        temporary
            .persist(path)
            .map_err(|error| error.error)
            .with_context(|| format!("无法保存配置文件 {}", path.display()))?;
        sync_directory(parent)?;
        Ok(())
    }

    pub fn validate(&self) -> Result<()> {
        if self.schema_version != default_schema_version() {
            bail!(
                "不支持的配置 schema_version={}；当前仅支持 2",
                self.schema_version
            );
        }
        validate_model_id(&self.model)?;
        validate_provider_id(&self.provider)?;
        if !(30..=900).contains(&self.chunk_seconds) {
            bail!("SpeakerHarness 的 chunk_seconds 必须在 30 到 900 之间");
        }
        if !(5..=30).contains(&self.overlap_seconds) || self.overlap_seconds >= self.chunk_seconds {
            bail!("SpeakerHarness 要求 overlap_seconds 在 5 到 30 之间，且小于 chunk_seconds");
        }
        if !(10..=300).contains(&self.min_chunk_seconds) {
            bail!("min_chunk_seconds 必须在 10 到 300 之间");
        }
        if self.min_chunk_seconds >= self.chunk_seconds {
            bail!("min_chunk_seconds 必须小于 chunk_seconds");
        }
        if self.min_chunk_seconds.saturating_mul(2) > self.chunk_seconds {
            bail!("2 * min_chunk_seconds 必须小于等于 chunk_seconds");
        }
        if !(256..=65_536).contains(&self.max_output_tokens) {
            bail!("max_output_tokens 必须在 256 到 65536 之间");
        }
        if self.split_output_tokens < 128 || self.split_output_tokens >= self.max_output_tokens {
            bail!("split_output_tokens 必须至少为 128，且小于 max_output_tokens");
        }
        if self.parallel_requests != 1 {
            bail!("SpeakerHarness 要求 parallel_requests=1，以保证全局说话人状态顺序传递");
        }
        if !(1..=10).contains(&self.retries) {
            bail!("retries 必须在 1 到 10 之间");
        }
        if !(1..=6).contains(&self.max_adaptive_depth) {
            bail!("max_adaptive_depth 必须在 1 到 6 之间");
        }
        if !(1..=10_000).contains(&self.max_http_attempts) {
            bail!("max_http_attempts 必须在 1 到 10000 之间");
        }
        if !(64 * 1024 * 1024..=100 * 1024 * 1024 * 1024).contains(&self.max_temp_bytes) {
            bail!("max_temp_bytes 必须在 64 MiB 到 100 GiB 之间");
        }
        if !(1..=32).contains(&self.max_speakers) {
            bail!("max_speakers 必须在 1 到 32 之间");
        }
        if !(2..=10).contains(&self.speaker_reference_seconds) {
            bail!("speaker_reference_seconds 必须在 2 到 10 之间");
        }
        if !(1..=5).contains(&self.speaker_reference_silence_seconds) {
            bail!("speaker_reference_silence_seconds 必须在 1 到 5 之间");
        }
        if !(500..=20_000).contains(&self.speaker_context_chars) {
            bail!("speaker_context_chars 必须在 500 到 20000 之间");
        }
        if !(1024 * 1024..=1024 * 1024 * 1024).contains(&self.max_transcript_bytes) {
            bail!("max_transcript_bytes 必须在 1 MiB 到 1 GiB 之间");
        }
        if !(1_000..=1_000_000).contains(&self.max_total_turns) {
            bail!("max_total_turns 必须在 1000 到 1000000 之间");
        }
        Ok(())
    }

    pub fn uses_any_provider(&self) -> bool {
        self.provider.eq_ignore_ascii_case(ANY_PROVIDER)
    }

    fn migrate_v1(&mut self) {
        self.schema_version = 2;
        if self.chunk_seconds == 300 {
            self.chunk_seconds = default_chunk_seconds();
        } else {
            self.chunk_seconds = self.chunk_seconds.min(default_chunk_seconds());
        }
        self.overlap_seconds = default_overlap_seconds()
            .min((self.chunk_seconds / 6).max(5))
            .min(self.chunk_seconds.saturating_sub(1));
        self.min_chunk_seconds = self.min_chunk_seconds.clamp(10, 300);
        if self.min_chunk_seconds.saturating_mul(2) > self.chunk_seconds {
            self.min_chunk_seconds = self.chunk_seconds / 2;
        }
        if self.max_output_tokens == 6_000 && self.split_output_tokens == 5_000 {
            self.max_output_tokens = default_max_output_tokens();
            self.split_output_tokens = default_split_output_tokens();
        }
        self.parallel_requests = default_parallel_requests();
    }

    fn normalize_v2_overlap_bound(&mut self) -> bool {
        let maximum = 30_u64.min(self.chunk_seconds.saturating_sub(1));
        if self.overlap_seconds > maximum && maximum >= 5 {
            self.overlap_seconds = maximum;
            true
        } else {
            false
        }
    }
}

fn decode_config(raw: &str) -> Result<(Config, bool)> {
    let document: toml::Value = toml::from_str(raw).context("无法解析 TOML")?;
    let version = match document.get("schema_version") {
        Some(value) => value
            .as_integer()
            .and_then(|version| u32::try_from(version).ok())
            .context("schema_version 必须是非负整数")?,
        None => 1,
    };
    let (mut config, mut migrated) = match version {
        1 => {
            let legacy: LegacyConfigV1 = toml::from_str(raw).context("无法解析 v1 配置")?;
            (Config::from(legacy), true)
        }
        2 => (
            toml::from_str::<Config>(raw).context("无法解析 v2 配置")?,
            false,
        ),
        version => bail!("不支持的配置 schema_version={version}；当前仅支持 1 和 2"),
    };
    if migrated {
        config.migrate_v1();
    } else if config.normalize_v2_overlap_bound() {
        migrated = true;
    }
    config.validate()?;
    Ok((config, migrated))
}

pub fn config_path() -> Result<PathBuf> {
    if let Some(path) = env::var_os("SPT_CONFIG_PATH") {
        if path.is_empty() {
            bail!("SPT_CONFIG_PATH 不能为空");
        }
        let path = PathBuf::from(path);
        validate_config_path(&path)?;
        return Ok(path);
    }

    if let Some(xdg) = env::var_os("XDG_CONFIG_HOME")
        && !xdg.is_empty()
    {
        let path = PathBuf::from(xdg).join("spt/config.toml");
        validate_config_path(&path)?;
        return Ok(path);
    }

    let home = dirs::home_dir().context("无法确定用户主目录；可设置 SPT_CONFIG_PATH")?;
    let path = home.join(".config/spt/config.toml");
    validate_config_path(&path)?;
    Ok(path)
}

fn validate_config_path(path: &Path) -> Result<()> {
    if path.file_name().and_then(|name| name.to_str()) == Some(".config.lock") {
        bail!("配置文件不能使用保留名称 .config.lock");
    }
    Ok(())
}

pub fn validate_model_id(value: &str) -> Result<()> {
    validate_route_id("模型代号", value, false)?;
    if !value.contains('/') {
        bail!("模型代号必须是 OpenRouter 的 provider/model 形式");
    }
    Ok(())
}

pub fn validate_provider_id(value: &str) -> Result<()> {
    if value.eq_ignore_ascii_case(ANY_PROVIDER) {
        return Ok(());
    }
    validate_route_id("provider 代号", value, false)
}

fn validate_route_id(label: &str, value: &str, allow_empty: bool) -> Result<()> {
    if !allow_empty && value.is_empty() {
        bail!("{label}不能为空");
    }
    if value.len() > 256 {
        bail!("{label}过长");
    }
    if !value.bytes().all(|byte| {
        byte.is_ascii_alphanumeric()
            || matches!(byte, b'-' | b'_' | b'.' | b'/' | b':' | b'@' | b'~')
    }) {
        bail!("{label}包含非法字符");
    }
    if value
        .split('/')
        .any(|segment| segment.is_empty() || segment == "." || segment == "..")
    {
        bail!("{label}包含非法路径片段");
    }
    Ok(())
}

#[cfg(unix)]
fn sync_directory(path: &Path) -> Result<()> {
    fs::File::open(path)
        .and_then(|directory| directory.sync_all())
        .with_context(|| format!("无法同步配置目录 {}", path.display()))
}

#[cfg(not(unix))]
fn sync_directory(_path: &Path) -> Result<()> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_valid() {
        let config = Config::default();
        config.validate().unwrap();
        assert_eq!(config.schema_version, 2);
        assert_eq!(config.chunk_seconds, 900);
        assert_eq!(config.overlap_seconds, 30);
        assert_eq!(config.parallel_requests, 1);
    }

    #[test]
    fn model_requires_openrouter_slug() {
        assert!(validate_model_id(DEFAULT_MODEL).is_ok());
        assert!(validate_model_id("gemini").is_err());
        assert!(validate_model_id("google/gemini 3").is_err());
        assert!(validate_model_id("google/gemini?key=x").is_err());
        assert!(validate_model_id("google/../models").is_err());
    }

    #[test]
    fn provider_accepts_pinned_or_any() {
        assert!(validate_provider_id(DEFAULT_PROVIDER).is_ok());
        assert!(validate_provider_id(ANY_PROVIDER).is_ok());
        assert!(validate_provider_id("bad provider").is_err());
    }

    #[test]
    fn toml_round_trip_preserves_values() {
        let config = Config::default();
        let encoded = toml::to_string(&config).unwrap();
        let decoded: Config = toml::from_str(&encoded).unwrap();
        assert_eq!(decoded.model, DEFAULT_MODEL);
        assert_eq!(decoded.provider, DEFAULT_PROVIDER);
    }

    #[test]
    fn v1_defaults_migrate_to_speaker_harness_defaults() {
        let mut config = Config {
            schema_version: 1,
            chunk_seconds: 300,
            max_output_tokens: 6_000,
            split_output_tokens: 5_000,
            parallel_requests: 3,
            ..Config::default()
        };
        config.migrate_v1();
        assert_eq!(config.schema_version, 2);
        assert_eq!(config.chunk_seconds, 900);
        assert_eq!(config.max_output_tokens, 16_000);
        assert_eq!(config.split_output_tokens, 12_000);
        assert_eq!(config.parallel_requests, 1);
        config.validate().unwrap();
    }

    #[test]
    fn v1_custom_token_pair_is_not_split_during_migration() {
        let mut config = Config {
            schema_version: 1,
            max_output_tokens: 8_000,
            split_output_tokens: 5_000,
            ..Config::default()
        };
        config.migrate_v1();
        assert_eq!(config.max_output_tokens, 8_000);
        assert_eq!(config.split_output_tokens, 5_000);
        config.validate().unwrap();
    }

    #[test]
    fn v1_long_custom_chunk_is_clamped_to_harness_limit() {
        let (config, migrated) =
            decode_config("schema_version = 1\nchunk_seconds = 1200\n").unwrap();
        assert!(migrated);
        assert_eq!(config.chunk_seconds, 900);
        assert_eq!(config.min_chunk_seconds, 30);
        config.validate().unwrap();
    }

    #[test]
    fn v1_custom_chunk_preserves_chunk_and_reduces_minimum_for_bisection() {
        let (config, migrated) =
            decode_config("schema_version = 1\nchunk_seconds = 500\nmin_chunk_seconds = 300\n")
                .unwrap();
        assert!(migrated);
        assert_eq!(config.chunk_seconds, 500);
        assert_eq!(config.min_chunk_seconds, 250);
        config.validate().unwrap();
    }

    #[test]
    fn v1_minimum_chunk_receives_a_valid_short_identity_context() {
        let (config, migrated) =
            decode_config("schema_version = 1\nchunk_seconds = 30\nmin_chunk_seconds = 10\n")
                .unwrap();
        assert!(migrated);
        assert_eq!(config.chunk_seconds, 30);
        assert_eq!(config.overlap_seconds, 5);
        config.validate().unwrap();
    }

    #[test]
    fn earlier_v2_long_overlap_is_clamped_and_marked_for_save() {
        let (config, migrated) =
            decode_config("schema_version = 2\noverlap_seconds = 120\n").unwrap();
        assert!(migrated);
        assert_eq!(config.overlap_seconds, 30);
        config.validate().unwrap();
    }

    #[test]
    fn harness_requires_overlap_and_sequential_requests() {
        let config = Config {
            overlap_seconds: 0,
            ..Config::default()
        };
        assert!(config.validate().is_err());
        let config = Config {
            parallel_requests: 2,
            ..Config::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn config_path_rejects_lock_file_name() {
        assert!(validate_config_path(Path::new("/tmp/.config.lock")).is_err());
        assert!(validate_config_path(Path::new("/tmp/config.toml")).is_ok());
    }

    #[test]
    fn missing_schema_version_uses_v1_defaults_before_migration() {
        let (config, migrated) = decode_config("parallel_requests = 3\n").unwrap();
        assert!(migrated);
        assert_eq!(config.schema_version, 2);
        assert_eq!(config.chunk_seconds, 900);
        assert_eq!(config.max_output_tokens, 16_000);
        assert_eq!(config.split_output_tokens, 12_000);
        assert_eq!(config.parallel_requests, 1);
    }
}
