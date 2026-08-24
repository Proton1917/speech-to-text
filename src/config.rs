use std::env;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use fs2::FileExt;
use serde::{Deserialize, Serialize};
use tempfile::NamedTempFile;

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
        file.lock_exclusive()
            .with_context(|| format!("无法获取配置锁 {}", lock_path.display()))?;
        Ok(Self { _file: file })
    }
}

const fn default_schema_version() -> u32 {
    1
}

fn default_model() -> String {
    DEFAULT_MODEL.to_owned()
}

fn default_provider() -> String {
    DEFAULT_PROVIDER.to_owned()
}

const fn default_chunk_seconds() -> u64 {
    300
}

const fn default_min_chunk_seconds() -> u64 {
    30
}

const fn default_max_output_tokens() -> u32 {
    6_000
}

const fn default_split_output_tokens() -> u32 {
    5_000
}

const fn default_parallel_requests() -> usize {
    3
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

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub schema_version: u32,
    pub model: String,
    pub provider: String,
    pub chunk_seconds: u64,
    pub min_chunk_seconds: u64,
    pub max_output_tokens: u32,
    pub split_output_tokens: u32,
    pub parallel_requests: usize,
    pub retries: u32,
    pub max_adaptive_depth: u8,
    pub max_http_attempts: u32,
    pub max_temp_bytes: u64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            schema_version: default_schema_version(),
            model: default_model(),
            provider: default_provider(),
            chunk_seconds: default_chunk_seconds(),
            min_chunk_seconds: default_min_chunk_seconds(),
            max_output_tokens: default_max_output_tokens(),
            split_output_tokens: default_split_output_tokens(),
            parallel_requests: default_parallel_requests(),
            retries: default_retries(),
            max_adaptive_depth: default_max_adaptive_depth(),
            max_http_attempts: default_max_http_attempts(),
            max_temp_bytes: default_max_temp_bytes(),
        }
    }
}

impl Config {
    pub fn load() -> Result<(Self, PathBuf, bool)> {
        let path = config_path()?;
        let metadata = match fs::symlink_metadata(&path) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Ok((Self::default(), path, false));
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
        let config: Self = toml::from_str(&raw)
            .with_context(|| format!("配置文件格式无效：{}", path.display()))?;
        config.validate()?;
        Ok((config, path, true))
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
                "不支持的配置 schema_version={}；当前仅支持 1",
                self.schema_version
            );
        }
        validate_model_id(&self.model)?;
        validate_provider_id(&self.provider)?;
        if !(30..=1_800).contains(&self.chunk_seconds) {
            bail!("chunk_seconds 必须在 30 到 1800 之间");
        }
        if !(10..=300).contains(&self.min_chunk_seconds) {
            bail!("min_chunk_seconds 必须在 10 到 300 之间");
        }
        if self.min_chunk_seconds >= self.chunk_seconds {
            bail!("min_chunk_seconds 必须小于 chunk_seconds");
        }
        if !(256..=65_536).contains(&self.max_output_tokens) {
            bail!("max_output_tokens 必须在 256 到 65536 之间");
        }
        if self.split_output_tokens < 128 || self.split_output_tokens >= self.max_output_tokens {
            bail!("split_output_tokens 必须至少为 128，且小于 max_output_tokens");
        }
        if !(1..=8).contains(&self.parallel_requests) {
            bail!("parallel_requests 必须在 1 到 8 之间");
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
        Ok(())
    }

    pub fn uses_any_provider(&self) -> bool {
        self.provider.eq_ignore_ascii_case(ANY_PROVIDER)
    }
}

pub fn config_path() -> Result<PathBuf> {
    if let Some(path) = env::var_os("SPT_CONFIG_PATH") {
        if path.is_empty() {
            bail!("SPT_CONFIG_PATH 不能为空");
        }
        return Ok(PathBuf::from(path));
    }

    if let Some(xdg) = env::var_os("XDG_CONFIG_HOME")
        && !xdg.is_empty()
    {
        return Ok(PathBuf::from(xdg).join("spt/config.toml"));
    }

    let home = dirs::home_dir().context("无法确定用户主目录；可设置 SPT_CONFIG_PATH")?;
    Ok(home.join(".config/spt/config.toml"))
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
fn secure_directory(path: &Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(path, fs::Permissions::from_mode(0o700))
        .with_context(|| format!("无法设置配置目录权限 {}", path.display()))
}

#[cfg(not(unix))]
fn secure_directory(_path: &Path) -> Result<()> {
    Ok(())
}

#[cfg(unix)]
fn secure_file(path: &Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(path, fs::Permissions::from_mode(0o600))
        .with_context(|| format!("无法设置配置文件权限 {}", path.display()))
}

#[cfg(not(unix))]
fn secure_file(_path: &Path) -> Result<()> {
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
        Config::default().validate().unwrap();
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
}
