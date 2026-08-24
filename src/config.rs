use std::env;
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use fs2::FileExt;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use tempfile::NamedTempFile;

use crate::security::{secure_directory, secure_file};

pub const DEFAULT_MODEL: &str = "google/gemini-3.7-flash";
pub const DEFAULT_QUALITY_REVIEW_MODEL: &str = "google/gemini-3.7-flash";
pub const DEFAULT_ASR_MODEL: &str = "qwen/qwen3-asr-1.7b";
pub const DEFAULT_QUALITY_ASR_MODEL: &str = "fish-audio/transcribe-1";
pub const DEFAULT_PROVIDER: &str = "google-vertex/global";
pub const DEFAULT_ASR_PROVIDER: &str = "deepinfra";
pub const DEFAULT_QUALITY_ASR_PROVIDER: &str = "fish-audio";
pub const ANY_PROVIDER: &str = "any";

const LEGACY_DEFAULT_MODEL: &str = "google/gemini-3.5-flash-lite";
const MAX_CONFIG_BYTES: u64 = 1024 * 1024;

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
        let parent = config_parent(path);
        prepare_config_directory(parent)?;
        let lock_path = parent.join(".config.lock");
        validate_lock_path_before_open(&lock_path)?;

        let mut options = fs::OpenOptions::new();
        options.read(true).write(true).create(true).truncate(false);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options
                .mode(0o600)
                .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC);
        }
        #[cfg(windows)]
        {
            use std::os::windows::fs::OpenOptionsExt;

            const FILE_FLAG_OPEN_REPARSE_POINT: u32 = 0x00200000;
            const FILE_SHARE_READ: u32 = 0x00000001;
            const FILE_SHARE_WRITE: u32 = 0x00000002;
            options
                .share_mode(FILE_SHARE_READ | FILE_SHARE_WRITE)
                .custom_flags(FILE_FLAG_OPEN_REPARSE_POINT);
        }
        let file = options
            .open(&lock_path)
            .with_context(|| format!("无法打开配置锁 {}", lock_path.display()))?;
        verify_open_lock_identity(&lock_path, &file)?;
        secure_open_lock_file(&lock_path, &file)?;
        verify_open_lock_identity(&lock_path, &file)?;
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
        verify_open_lock_identity(&lock_path, &file)?;
        Ok(Self { _file: file })
    }
}

fn config_parent(path: &Path) -> &Path {
    path.parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
}

fn prepare_config_directory(parent: &Path) -> Result<()> {
    validate_config_directory_boundary(parent)?;
    let parent_existed = match fs::symlink_metadata(parent) {
        Ok(_) => true,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => false,
        Err(error) => {
            return Err(error)
                .with_context(|| format!("无法读取配置目录元数据 {}", parent.display()));
        }
    };

    fs::create_dir_all(parent).with_context(|| format!("无法创建配置目录 {}", parent.display()))?;
    checked_config_directory_metadata(parent)?;
    if !parent_existed {
        secure_directory(parent)?;
    }
    #[cfg(windows)]
    secure_directory(parent)?;

    #[cfg(unix)]
    {
        let directory = fs::File::open(parent)
            .with_context(|| format!("无法打开配置目录 {}", parent.display()))?;
        verify_open_directory_identity(parent, &directory)?;
        if !parent_existed {
            secure_open_directory(parent, &directory)?;
            verify_open_directory_identity(parent, &directory)?;
        }
    }

    let final_metadata = checked_config_directory_metadata(parent)?;
    validate_private_directory_mode(parent, &final_metadata)
}

fn validate_config_directory_boundary(parent: &Path) -> Result<()> {
    let mut boundary = if parent.is_absolute() {
        parent.to_owned()
    } else {
        env::current_dir().context("无法确定当前目录")?.join(parent)
    };

    loop {
        match fs::symlink_metadata(&boundary) {
            Ok(metadata) => {
                validate_config_directory_kind(&boundary, &metadata)?;
                return validate_private_directory_mode(&boundary, &metadata);
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                if !boundary.pop() {
                    bail!("无法确定配置目录的安全创建边界：{}", parent.display());
                }
            }
            Err(error) => {
                return Err(error)
                    .with_context(|| format!("无法读取配置目录元数据 {}", boundary.display()));
            }
        }
    }
}

fn checked_config_directory_metadata(path: &Path) -> Result<fs::Metadata> {
    let metadata = fs::symlink_metadata(path)
        .with_context(|| format!("无法读取配置目录元数据 {}", path.display()))?;
    validate_config_directory_kind(path, &metadata)?;
    Ok(metadata)
}

fn validate_config_directory_kind(path: &Path, metadata: &fs::Metadata) -> Result<()> {
    if metadata.file_type().is_symlink() {
        bail!("配置目录不能是符号链接：{}", path.display());
    }
    if !metadata.file_type().is_dir() {
        bail!("配置目录不是目录：{}", path.display());
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;

        const FILE_ATTRIBUTE_REPARSE_POINT: u32 = 0x00000400;
        if metadata.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0 {
            bail!("配置目录不能是 Windows reparse point：{}", path.display());
        }
    }
    Ok(())
}

#[cfg(unix)]
fn validate_private_directory_mode(path: &Path, metadata: &fs::Metadata) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;

    if metadata.permissions().mode() & 0o022 != 0 {
        bail!("配置目录不能允许组或其他用户写入：{}", path.display());
    }
    Ok(())
}

#[cfg(not(unix))]
fn validate_private_directory_mode(_path: &Path, _metadata: &fs::Metadata) -> Result<()> {
    Ok(())
}

#[cfg(unix)]
fn verify_open_directory_identity(path: &Path, directory: &fs::File) -> Result<()> {
    use std::os::unix::fs::MetadataExt;

    let path_metadata = checked_config_directory_metadata(path)?;
    let handle_metadata = directory
        .metadata()
        .with_context(|| format!("无法读取已打开配置目录的元数据 {}", path.display()))?;
    if !handle_metadata.file_type().is_dir()
        || path_metadata.dev() != handle_metadata.dev()
        || path_metadata.ino() != handle_metadata.ino()
    {
        bail!("配置目录在打开期间发生变化：{}", path.display());
    }
    Ok(())
}

#[cfg(unix)]
fn secure_open_directory(path: &Path, directory: &fs::File) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;

    directory
        .set_permissions(fs::Permissions::from_mode(0o700))
        .with_context(|| format!("无法通过句柄设置配置目录权限 {}", path.display()))
}

fn validate_lock_path_before_open(path: &Path) -> Result<()> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_symlink() => {
            bail!("配置锁不能是符号链接：{}", path.display())
        }
        Ok(metadata) if !metadata.file_type().is_file() => {
            bail!("配置锁不是普通文件：{}", path.display())
        }
        Ok(_metadata) => {
            #[cfg(windows)]
            {
                use std::os::windows::fs::MetadataExt;

                const FILE_ATTRIBUTE_REPARSE_POINT: u32 = 0x00000400;
                if _metadata.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0 {
                    bail!("配置锁不能是 Windows reparse point：{}", path.display());
                }
            }
            Ok(())
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => {
            Err(error).with_context(|| format!("无法读取配置锁元数据 {}", path.display()))
        }
    }
}

fn verify_open_lock_identity(path: &Path, file: &fs::File) -> Result<()> {
    let path_metadata = fs::symlink_metadata(path)
        .with_context(|| format!("无法读取配置锁元数据 {}", path.display()))?;
    if path_metadata.file_type().is_symlink() || !path_metadata.file_type().is_file() {
        bail!("配置锁路径在打开期间变为非普通文件：{}", path.display());
    }
    let handle_metadata = file
        .metadata()
        .with_context(|| format!("无法读取已打开配置锁的元数据 {}", path.display()))?;
    if !handle_metadata.file_type().is_file() {
        bail!("已打开的配置锁不是普通文件：{}", path.display());
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;

        const FILE_ATTRIBUTE_REPARSE_POINT: u32 = 0x00000400;
        if path_metadata.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0
            || handle_metadata.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0
        {
            bail!("配置锁不能是 Windows reparse point：{}", path.display());
        }
    }

    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;

        if path_metadata.dev() != handle_metadata.dev()
            || path_metadata.ino() != handle_metadata.ino()
        {
            bail!("配置锁路径与已打开句柄不一致：{}", path.display());
        }
        if handle_metadata.nlink() != 1 {
            bail!("配置锁不能有多个硬链接：{}", path.display());
        }
    }
    Ok(())
}

#[cfg(unix)]
fn secure_open_lock_file(path: &Path, file: &fs::File) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;

    file.set_permissions(fs::Permissions::from_mode(0o600))
        .with_context(|| format!("无法通过句柄设置配置锁权限 {}", path.display()))?;
    let mode = file
        .metadata()
        .with_context(|| format!("无法验证配置锁权限 {}", path.display()))?
        .permissions()
        .mode()
        & 0o777;
    if mode != 0o600 {
        bail!("配置锁权限不是 0600：{}", path.display());
    }
    Ok(())
}

#[cfg(windows)]
fn secure_open_lock_file(path: &Path, _file: &fs::File) -> Result<()> {
    secure_file(path)
}

#[cfg(not(any(unix, windows)))]
fn secure_open_lock_file(_path: &Path, _file: &fs::File) -> Result<()> {
    Ok(())
}

const fn default_schema_version() -> u32 {
    4
}

fn default_model() -> String {
    DEFAULT_MODEL.to_owned()
}

fn default_quality_review_model() -> String {
    DEFAULT_QUALITY_REVIEW_MODEL.to_owned()
}

fn default_asr_model() -> String {
    DEFAULT_ASR_MODEL.to_owned()
}

fn default_quality_asr_model() -> String {
    DEFAULT_QUALITY_ASR_MODEL.to_owned()
}

fn default_provider() -> String {
    DEFAULT_PROVIDER.to_owned()
}

fn default_asr_provider() -> String {
    DEFAULT_ASR_PROVIDER.to_owned()
}

fn default_quality_asr_provider() -> String {
    DEFAULT_QUALITY_ASR_PROVIDER.to_owned()
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

const ASR_MAX_CHUNK_SECONDS: u64 = 120;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub schema_version: u32,
    pub model: String,
    pub quality_review_model: String,
    pub asr_model: String,
    pub quality_asr_model: String,
    pub provider: String,
    pub asr_provider: String,
    pub quality_asr_provider: String,
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
            model: LEGACY_DEFAULT_MODEL.to_owned(),
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
        let quality_review_model = legacy.model.clone();
        Self {
            schema_version: legacy.schema_version,
            model: legacy.model,
            quality_review_model,
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
            quality_review_model: default_quality_review_model(),
            asr_model: default_asr_model(),
            quality_asr_model: default_quality_asr_model(),
            provider: default_provider(),
            asr_provider: default_asr_provider(),
            quality_asr_provider: default_quality_asr_provider(),
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
        let mut file = match open_config_file_nofollow(&path) {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Ok((Self::default(), path, false, false));
            }
            Err(error) => {
                return Err(error)
                    .with_context(|| format!("无法读取配置文件元数据 {}", path.display()));
            }
        };
        let raw = read_open_config_bounded(&mut file, &path)?;
        let (config, migrated) =
            decode_config(&raw).with_context(|| format!("配置文件格式无效：{}", path.display()))?;
        Ok((config, path, true, migrated))
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        self.validate()?;
        let parent = config_parent(path);
        prepare_config_directory(parent)?;

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
                "不支持的配置 schema_version={}；当前仅支持 4",
                self.schema_version
            );
        }
        validate_model_id(&self.model)?;
        validate_model_id(&self.quality_review_model)?;
        validate_model_id(&self.asr_model)?;
        validate_model_id(&self.quality_asr_model)?;
        validate_provider_id(&self.provider)?;
        validate_provider_id(&self.asr_provider)?;
        validate_provider_id(&self.quality_asr_provider)?;
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

    /// Returns the configured multimodal quality-overlay route. Dedicated transcript text comes
    /// from `quality_asr_model`; this route handles the quality bootstrap/review overlay.
    pub fn effective_quality_review_model(&self) -> &str {
        &self.quality_review_model
    }

    pub fn effective_quality_asr_model(&self) -> &str {
        &self.quality_asr_model
    }

    pub fn effective_asr_chunk_seconds(&self) -> u64 {
        self.chunk_seconds.min(ASR_MAX_CHUNK_SECONDS)
    }

    pub fn effective_asr_min_chunk_seconds(&self) -> u64 {
        self.min_chunk_seconds
            .min(self.effective_asr_chunk_seconds() / 2)
    }

    pub fn effective_quality_chunk_seconds(&self) -> u64 {
        self.effective_asr_chunk_seconds()
    }

    pub fn effective_quality_min_chunk_seconds(&self) -> u64 {
        self.effective_asr_min_chunk_seconds()
    }

    fn migrate_v1(&mut self) {
        self.schema_version = default_schema_version();
        if self.model == LEGACY_DEFAULT_MODEL {
            self.model = default_model();
        }
        self.quality_review_model = self.model.clone();
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

    fn normalize_v2_overlap_bound(&mut self) {
        let maximum = 30_u64.min(self.chunk_seconds.saturating_sub(1));
        if self.overlap_seconds > maximum && maximum >= 5 {
            self.overlap_seconds = maximum;
        }
    }
}

fn parse_toml_safely<T>(raw: &str, diagnostic: &'static str) -> Result<T>
where
    T: DeserializeOwned,
{
    // toml::de::Error retains and renders the complete source line. Discard it at this boundary so
    // a misplaced credential can never enter anyhow's Display or Debug error chain.
    toml::from_str(raw).map_err(|_| anyhow!(diagnostic))
}

fn read_open_config_bounded(file: &mut fs::File, path: &Path) -> Result<String> {
    let metadata = file
        .metadata()
        .with_context(|| format!("无法读取配置文件句柄元数据 {}", path.display()))?;
    if !metadata.file_type().is_file() {
        bail!("配置路径不是普通文件：{}", path.display());
    }
    if metadata.len() > MAX_CONFIG_BYTES {
        bail!("配置文件超过 1 MiB 安全上限：{}", path.display());
    }
    let mut raw = String::new();
    Read::by_ref(file)
        .take(MAX_CONFIG_BYTES + 1)
        .read_to_string(&mut raw)
        .with_context(|| format!("无法读取 UTF-8 配置文件 {}", path.display()))?;
    if raw.len() as u64 > MAX_CONFIG_BYTES {
        bail!("配置文件在读取期间超过 1 MiB 安全上限：{}", path.display());
    }
    if file
        .metadata()
        .with_context(|| format!("无法复核配置文件句柄 {}", path.display()))?
        .len()
        != metadata.len()
    {
        bail!("配置文件在读取期间发生变化：{}", path.display());
    }
    Ok(raw)
}

fn decode_config(raw: &str) -> Result<(Config, bool)> {
    let document: toml::Value = parse_toml_safely(raw, "无法解析 TOML 配置（内容已隐藏）")?;
    let version = match document.get("schema_version") {
        Some(value) => value
            .as_integer()
            .and_then(|version| u32::try_from(version).ok())
            .context("schema_version 必须是非负整数")?,
        None => 1,
    };
    let (mut config, migrated) = match version {
        1 => {
            let legacy: LegacyConfigV1 = parse_toml_safely(raw, "无法解析 v1 配置（内容已隐藏）")?;
            (Config::from(legacy), true)
        }
        2 => {
            let mut config = parse_toml_safely::<Config>(raw, "无法解析 v2 配置（内容已隐藏）")?;
            config.schema_version = default_schema_version();
            if config.model == LEGACY_DEFAULT_MODEL {
                config.model = default_model();
            }
            config.quality_review_model = config.model.clone();
            config.asr_model = default_asr_model();
            config.quality_asr_model = default_quality_asr_model();
            config.asr_provider = default_asr_provider();
            config.quality_asr_provider = default_quality_asr_provider();
            (config, true)
        }
        3 => {
            let mut config = parse_toml_safely::<Config>(raw, "无法解析 v3 配置（内容已隐藏）")?;
            config.schema_version = default_schema_version();
            if config.model == LEGACY_DEFAULT_MODEL
                && config.quality_review_model == DEFAULT_QUALITY_REVIEW_MODEL
            {
                config.model = default_model();
            }
            config.asr_model = default_asr_model();
            config.quality_asr_model = default_quality_asr_model();
            config.asr_provider = default_asr_provider();
            config.quality_asr_provider = default_quality_asr_provider();
            (config, true)
        }
        4 => (
            parse_toml_safely::<Config>(raw, "无法解析 v4 配置（内容已隐藏）")?,
            false,
        ),
        version => bail!("不支持的配置 schema_version={version}；当前仅支持 1、2、3 和 4"),
    };
    if version == 1 {
        config.migrate_v1();
    }
    if version == 2 {
        config.normalize_v2_overlap_bound();
    }
    config.validate()?;
    Ok((config, migrated))
}

pub fn config_path() -> Result<PathBuf> {
    if let Some(path) = env::var_os("SPT_CONFIG_PATH") {
        if path.is_empty() {
            bail!("SPT_CONFIG_PATH 不能为空");
        }
        let path = canonicalize_existing_config_parent(PathBuf::from(path))?;
        validate_config_path(&path)?;
        return Ok(path);
    }

    if let Some(xdg) = env::var_os("XDG_CONFIG_HOME")
        && !xdg.is_empty()
    {
        let path = canonicalize_existing_config_parent(PathBuf::from(xdg).join("spt/config.toml"))?;
        validate_config_path(&path)?;
        return Ok(path);
    }

    let home = dirs::home_dir().context("无法确定用户主目录；可设置 SPT_CONFIG_PATH")?;
    let path = canonicalize_existing_config_parent(home.join(".config/spt/config.toml"))?;
    validate_config_path(&path)?;
    Ok(path)
}

fn canonicalize_existing_config_parent(path: PathBuf) -> Result<PathBuf> {
    let parent = config_parent(&path);
    match fs::canonicalize(parent) {
        Ok(canonical_parent) => {
            Ok(canonical_parent.join(path.file_name().context("配置路径缺少文件名")?))
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(path),
        Err(error) => {
            Err(error).with_context(|| format!("无法解析配置目录真实路径 {}", parent.display()))
        }
    }
}

fn validate_config_path(path: &Path) -> Result<()> {
    if path
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| {
            name.trim_end_matches(['.', ' '])
                .eq_ignore_ascii_case(".config.lock")
        })
    {
        bail!("配置文件不能使用保留名称 .config.lock");
    }
    validate_config_directory_boundary(config_parent(path))
}

#[cfg(target_os = "macos")]
fn open_config_file_nofollow(path: &Path) -> std::io::Result<fs::File> {
    use std::os::unix::fs::OpenOptionsExt;

    const O_NOFOLLOW_ANY: i32 = 0x2000_0000;
    fs::OpenOptions::new()
        .read(true)
        .custom_flags(O_NOFOLLOW_ANY | libc::O_CLOEXEC)
        .open(path)
}

#[cfg(all(unix, not(target_os = "macos")))]
fn open_config_file_nofollow(path: &Path) -> std::io::Result<fs::File> {
    use std::os::unix::fs::OpenOptionsExt;

    fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
        .open(path)
}

#[cfg(windows)]
fn open_config_file_nofollow(path: &Path) -> std::io::Result<fs::File> {
    use std::os::windows::fs::{MetadataExt, OpenOptionsExt};

    const FILE_ATTRIBUTE_REPARSE_POINT: u32 = 0x00000400;
    const FILE_FLAG_OPEN_REPARSE_POINT: u32 = 0x00200000;
    let file = fs::OpenOptions::new()
        .read(true)
        .custom_flags(FILE_FLAG_OPEN_REPARSE_POINT)
        .open(path)?;
    if file.metadata()?.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "config path is a Windows reparse point",
        ));
    }
    Ok(file)
}

#[cfg(not(any(unix, windows)))]
fn open_config_file_nofollow(path: &Path) -> std::io::Result<fs::File> {
    let metadata = fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "config path is a symbolic link",
        ));
    }
    fs::File::open(path)
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

    fn assert_error_chain_is_redacted(error: &anyhow::Error, secret: &str) {
        let mut renderings = vec![
            format!("{error:#}"),
            format!("{error:?}"),
            format!("{error:#?}"),
        ];
        for cause in error.chain() {
            renderings.push(format!("{cause}"));
            renderings.push(format!("{cause:?}"));
        }
        for rendered in renderings {
            assert!(!rendered.contains(secret), "错误泄露了伪密钥：{rendered}");
            assert!(
                !rendered.contains("OPENROUTER_API_KEY ="),
                "错误泄露了原始配置行：{rendered}"
            );
        }
    }

    #[test]
    fn defaults_are_valid() {
        let config = Config::default();
        config.validate().unwrap();
        assert_eq!(config.schema_version, 4);
        assert_eq!(config.model, DEFAULT_MODEL);
        assert_eq!(config.quality_review_model, DEFAULT_QUALITY_REVIEW_MODEL);
        assert_eq!(config.model, config.quality_review_model);
        assert_eq!(config.asr_model, DEFAULT_ASR_MODEL);
        assert_eq!(config.quality_asr_model, DEFAULT_QUALITY_ASR_MODEL);
        assert_eq!(config.asr_provider, DEFAULT_ASR_PROVIDER);
        assert_eq!(config.quality_asr_provider, DEFAULT_QUALITY_ASR_PROVIDER);
        assert_eq!(config.chunk_seconds, 900);
        assert_eq!(config.overlap_seconds, 30);
        assert_eq!(config.parallel_requests, 1);
        assert_eq!(
            config.effective_quality_review_model(),
            DEFAULT_QUALITY_REVIEW_MODEL
        );
        assert_eq!(
            config.effective_quality_asr_model(),
            DEFAULT_QUALITY_ASR_MODEL
        );
        assert_eq!(config.effective_asr_chunk_seconds(), 120);
        assert_eq!(config.effective_asr_min_chunk_seconds(), 30);
        assert_eq!(config.effective_quality_chunk_seconds(), 120);
        assert_eq!(config.effective_quality_min_chunk_seconds(), 30);
    }

    #[test]
    fn explicit_model_override_remains_authoritative_for_quality_review() {
        let config = Config {
            model: "anthropic/claude-sonnet-4.5".to_owned(),
            quality_review_model: "anthropic/claude-sonnet-4.5".to_owned(),
            ..Config::default()
        };
        assert_eq!(
            config.effective_quality_review_model(),
            "anthropic/claude-sonnet-4.5"
        );
    }

    #[test]
    fn asr_window_is_hard_capped_and_clamps_runtime_split_minimum() {
        let smaller = Config {
            chunk_seconds: 90,
            min_chunk_seconds: 30,
            ..Config::default()
        };
        assert_eq!(smaller.effective_asr_chunk_seconds(), 90);
        assert_eq!(smaller.effective_asr_min_chunk_seconds(), 30);

        let large_minimum = Config {
            chunk_seconds: 900,
            min_chunk_seconds: 200,
            ..Config::default()
        };
        assert_eq!(large_minimum.effective_asr_chunk_seconds(), 120);
        assert_eq!(large_minimum.effective_asr_min_chunk_seconds(), 60);
        assert_eq!(large_minimum.effective_quality_chunk_seconds(), 120);
        assert_eq!(large_minimum.effective_quality_min_chunk_seconds(), 60);
    }

    #[test]
    fn model_requires_openrouter_slug() {
        assert!(validate_model_id(DEFAULT_MODEL).is_ok());
        assert!(validate_model_id(DEFAULT_ASR_MODEL).is_ok());
        assert!(validate_model_id(DEFAULT_QUALITY_ASR_MODEL).is_ok());
        assert!(validate_model_id("gemini").is_err());
        assert!(validate_model_id("google/gemini 3").is_err());
        assert!(validate_model_id("google/gemini?key=x").is_err());
        assert!(validate_model_id("google/../models").is_err());
    }

    #[test]
    fn provider_accepts_pinned_or_any() {
        assert!(validate_provider_id(DEFAULT_PROVIDER).is_ok());
        assert!(validate_provider_id(DEFAULT_ASR_PROVIDER).is_ok());
        assert!(validate_provider_id(DEFAULT_QUALITY_ASR_PROVIDER).is_ok());
        assert!(validate_provider_id(ANY_PROVIDER).is_ok());
        assert!(validate_provider_id("bad provider").is_err());

        let automatic_asr = Config {
            asr_provider: ANY_PROVIDER.to_owned(),
            quality_asr_provider: ANY_PROVIDER.to_owned(),
            ..Config::default()
        };
        automatic_asr.validate().unwrap();
    }

    #[test]
    fn toml_round_trip_preserves_values() {
        let config = Config::default();
        let encoded = toml::to_string(&config).unwrap();
        let decoded: Config = toml::from_str(&encoded).unwrap();
        assert_eq!(decoded.schema_version, 4);
        assert_eq!(decoded.model, DEFAULT_MODEL);
        assert_eq!(decoded.quality_review_model, DEFAULT_QUALITY_REVIEW_MODEL);
        assert_eq!(decoded.asr_model, DEFAULT_ASR_MODEL);
        assert_eq!(decoded.quality_asr_model, DEFAULT_QUALITY_ASR_MODEL);
        assert_eq!(decoded.provider, DEFAULT_PROVIDER);
        assert_eq!(decoded.asr_provider, DEFAULT_ASR_PROVIDER);
        assert_eq!(decoded.quality_asr_provider, DEFAULT_QUALITY_ASR_PROVIDER);
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
        assert_eq!(config.schema_version, 4);
        assert_eq!(config.chunk_seconds, 900);
        assert_eq!(config.max_output_tokens, 16_000);
        assert_eq!(config.split_output_tokens, 12_000);
        assert_eq!(config.parallel_requests, 1);
        assert_eq!(config.model, DEFAULT_MODEL);
        assert_eq!(config.quality_review_model, DEFAULT_QUALITY_REVIEW_MODEL);
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
    fn v1_migration_preserves_multimodal_route_and_adds_asr_defaults() {
        let (config, migrated) = decode_config(
            "schema_version = 1\nmodel = \"anthropic/claude-sonnet-4.5\"\nprovider = \"any\"\n",
        )
        .unwrap();
        assert!(migrated);
        assert_eq!(config.schema_version, 4);
        assert_eq!(config.model, "anthropic/claude-sonnet-4.5");
        assert_eq!(config.quality_review_model, "anthropic/claude-sonnet-4.5");
        assert_eq!(config.provider, ANY_PROVIDER);
        assert_eq!(config.asr_model, DEFAULT_ASR_MODEL);
        assert_eq!(config.quality_asr_model, DEFAULT_QUALITY_ASR_MODEL);
        assert_eq!(config.asr_provider, DEFAULT_ASR_PROVIDER);
        assert_eq!(config.quality_asr_provider, DEFAULT_QUALITY_ASR_PROVIDER);
        config.validate().unwrap();
    }

    #[test]
    fn v1_legacy_default_lite_migrates_to_37_overlay() {
        let (config, migrated) = decode_config(&format!(
            "schema_version = 1\nmodel = \"{LEGACY_DEFAULT_MODEL}\"\n"
        ))
        .unwrap();
        assert!(migrated);
        assert_eq!(config.schema_version, 4);
        assert_eq!(config.model, DEFAULT_MODEL);
        assert_eq!(config.quality_review_model, DEFAULT_QUALITY_REVIEW_MODEL);
        assert_eq!(config.model, config.quality_review_model);
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
        assert_eq!(config.schema_version, 4);
        assert_eq!(config.quality_review_model, DEFAULT_MODEL);
        assert_eq!(config.asr_model, DEFAULT_ASR_MODEL);
        assert_eq!(config.quality_asr_model, DEFAULT_QUALITY_ASR_MODEL);
        assert_eq!(config.asr_provider, DEFAULT_ASR_PROVIDER);
        assert_eq!(config.quality_asr_provider, DEFAULT_QUALITY_ASR_PROVIDER);
        config.validate().unwrap();
    }

    #[test]
    fn v4_long_overlap_is_rejected_instead_of_silently_migrated() {
        let error =
            decode_config("schema_version = 4\nchunk_seconds = 900\noverlap_seconds = 31\n")
                .unwrap_err();
        assert!(format!("{error:#}").contains("overlap_seconds"));
    }

    #[test]
    fn v2_legacy_default_lite_migrates_to_37_overlay() {
        let (config, migrated) = decode_config(&format!(
            "schema_version = 2\nmodel = \"{LEGACY_DEFAULT_MODEL}\"\n"
        ))
        .unwrap();
        assert!(migrated);
        assert_eq!(config.schema_version, 4);
        assert_eq!(config.model, DEFAULT_MODEL);
        assert_eq!(config.quality_review_model, DEFAULT_QUALITY_REVIEW_MODEL);
        assert_eq!(config.model, config.quality_review_model);
    }

    #[test]
    fn v2_custom_model_migration_does_not_silently_add_another_model() {
        let (config, migrated) = decode_config(
            "schema_version = 2\nmodel = \"anthropic/claude-sonnet-4.5\"\nprovider = \"any\"\n",
        )
        .unwrap();
        assert!(migrated);
        assert_eq!(config.schema_version, 4);
        assert_eq!(config.model, "anthropic/claude-sonnet-4.5");
        assert_eq!(config.quality_review_model, "anthropic/claude-sonnet-4.5");
        assert_eq!(config.provider, ANY_PROVIDER);
        assert_eq!(config.asr_model, DEFAULT_ASR_MODEL);
        assert_eq!(config.quality_asr_model, DEFAULT_QUALITY_ASR_MODEL);
        assert_eq!(config.asr_provider, DEFAULT_ASR_PROVIDER);
        assert_eq!(config.quality_asr_provider, DEFAULT_QUALITY_ASR_PROVIDER);
    }

    #[test]
    fn v3_custom_migration_preserves_multimodal_routes_and_adds_asr_defaults() {
        let (config, migrated) = decode_config(
            "schema_version = 3\nmodel = \"anthropic/claude-sonnet-4.5\"\nquality_review_model = \"google/gemini-3.7-flash\"\nprovider = \"any\"\nchunk_seconds = 480\n",
        )
        .unwrap();
        assert!(migrated);
        assert_eq!(config.schema_version, 4);
        assert_eq!(config.model, "anthropic/claude-sonnet-4.5");
        assert_eq!(config.quality_review_model, "google/gemini-3.7-flash");
        assert_eq!(config.provider, ANY_PROVIDER);
        assert_eq!(config.chunk_seconds, 480);
        assert_eq!(config.asr_model, DEFAULT_ASR_MODEL);
        assert_eq!(config.quality_asr_model, DEFAULT_QUALITY_ASR_MODEL);
        assert_eq!(config.asr_provider, DEFAULT_ASR_PROVIDER);
        assert_eq!(config.quality_asr_provider, DEFAULT_QUALITY_ASR_PROVIDER);
        config.validate().unwrap();
    }

    #[test]
    fn v3_official_lite_37_pair_migrates_both_overlays_to_37() {
        let (config, migrated) = decode_config(&format!(
            "schema_version = 3\nmodel = \"{LEGACY_DEFAULT_MODEL}\"\nquality_review_model = \"{DEFAULT_QUALITY_REVIEW_MODEL}\"\nprovider = \"google-vertex/global\"\n"
        ))
        .unwrap();
        assert!(migrated);
        assert_eq!(config.schema_version, 4);
        assert_eq!(config.model, DEFAULT_MODEL);
        assert_eq!(config.quality_review_model, DEFAULT_QUALITY_REVIEW_MODEL);
        assert_eq!(config.model, config.quality_review_model);
        assert_eq!(config.provider, DEFAULT_PROVIDER);
    }

    #[test]
    fn v3_explicit_single_lite_overlay_is_preserved_as_custom() {
        let (config, migrated) = decode_config(&format!(
            "schema_version = 3\nmodel = \"{LEGACY_DEFAULT_MODEL}\"\nquality_review_model = \"{LEGACY_DEFAULT_MODEL}\"\nprovider = \"any\"\n"
        ))
        .unwrap();
        assert!(migrated);
        assert_eq!(config.schema_version, 4);
        assert_eq!(config.model, LEGACY_DEFAULT_MODEL);
        assert_eq!(config.quality_review_model, LEGACY_DEFAULT_MODEL);
        assert_eq!(config.provider, ANY_PROVIDER);
    }

    #[test]
    fn v4_decode_preserves_explicit_asr_routes_without_migration() {
        let (config, migrated) = decode_config(
            "schema_version = 4\nasr_model = \"custom/raw-asr\"\nquality_asr_model = \"custom/quality-asr\"\nasr_provider = \"custom-raw\"\nquality_asr_provider = \"custom-quality\"\n",
        )
        .unwrap();
        assert!(!migrated);
        assert_eq!(config.schema_version, 4);
        assert_eq!(config.asr_model, "custom/raw-asr");
        assert_eq!(config.quality_asr_model, "custom/quality-asr");
        assert_eq!(config.asr_provider, "custom-raw");
        assert_eq!(config.quality_asr_provider, "custom-quality");
        config.validate().unwrap();
    }

    #[test]
    fn invalid_asr_route_ids_are_rejected() {
        let invalid_raw = Config {
            asr_model: "invalid asr".to_owned(),
            ..Config::default()
        };
        assert!(invalid_raw.validate().is_err());

        let invalid_quality = Config {
            quality_asr_model: "quality-asr-without-provider".to_owned(),
            ..Config::default()
        };
        assert!(invalid_quality.validate().is_err());

        let invalid_provider = Config {
            asr_provider: "invalid provider".to_owned(),
            ..Config::default()
        };
        assert!(invalid_provider.validate().is_err());

        let invalid_quality_provider = Config {
            quality_asr_provider: "invalid quality provider".to_owned(),
            ..Config::default()
        };
        assert!(invalid_quality_provider.validate().is_err());
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
        let directory = tempfile::tempdir().unwrap();
        assert!(validate_config_path(&directory.path().join(".config.lock")).is_err());
        assert!(validate_config_path(&directory.path().join(".CONFIG.LOCK")).is_err());
        assert!(validate_config_path(&directory.path().join(".config.lock. ")).is_err());
        assert!(validate_config_path(&directory.path().join("config.toml")).is_ok());
    }

    #[test]
    fn config_reader_rejects_files_over_one_mibibyte() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("config.toml");
        fs::write(&path, vec![b'x'; MAX_CONFIG_BYTES as usize + 1]).unwrap();
        let canonical = fs::canonicalize(&path).unwrap();
        let mut file = open_config_file_nofollow(&canonical).unwrap();
        assert!(read_open_config_bounded(&mut file, &canonical).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn config_reader_does_not_follow_a_terminal_symlink() {
        use std::os::unix::fs::symlink;

        let directory = tempfile::tempdir().unwrap();
        let target = directory.path().join("target.toml");
        fs::write(&target, "schema_version = 4\n").unwrap();
        let link = directory.path().join("config.toml");
        symlink(&target, &link).unwrap();
        let canonical_parent = fs::canonicalize(directory.path()).unwrap();
        assert!(open_config_file_nofollow(&canonical_parent.join("config.toml")).is_err());
    }

    #[test]
    fn toml_unknown_field_error_never_retains_secret_source_line() {
        let secret = "TEST-SECRET-FAKE-CONFIG-UNKNOWN-FIELD";
        let raw = format!("schema_version = 4\nOPENROUTER_API_KEY = \"{secret}\"\n");
        let error = decode_config(&raw)
            .context("配置文件格式无效：/safe/config.toml")
            .unwrap_err();
        assert!(format!("{error:#}").contains("无法解析 v4 配置"));
        assert_error_chain_is_redacted(&error, secret);
    }

    #[test]
    fn malformed_toml_error_never_retains_secret_source_line() {
        let secret = "TEST-SECRET-FAKE-CONFIG-MALFORMED";
        let raw = format!("schema_version = 4\nOPENROUTER_API_KEY = \"{secret}\n");
        let error = decode_config(&raw)
            .context("配置文件格式无效：/safe/config.toml")
            .unwrap_err();
        assert!(format!("{error:#}").contains("无法解析 TOML 配置"));
        assert_error_chain_is_redacted(&error, secret);
    }

    #[cfg(unix)]
    #[test]
    fn config_lock_rejects_existing_symlink_without_changing_target_permissions() {
        use std::os::unix::fs::{PermissionsExt, symlink};

        let directory = tempfile::tempdir().unwrap();
        let target = directory.path().join("lock-target");
        fs::write(&target, b"do not touch").unwrap();
        fs::set_permissions(&target, fs::Permissions::from_mode(0o644)).unwrap();
        symlink(&target, directory.path().join(".config.lock")).unwrap();

        let error = ConfigLock::acquire(&directory.path().join("config.toml"))
            .err()
            .expect("symlink lock must be rejected");
        assert!(format!("{error:#}").contains("符号链接"));
        assert!(
            fs::symlink_metadata(directory.path().join(".config.lock"))
                .unwrap()
                .file_type()
                .is_symlink()
        );
        assert_eq!(
            fs::metadata(target).unwrap().permissions().mode() & 0o777,
            0o644
        );
    }

    #[cfg(unix)]
    #[test]
    fn open_lock_identity_rejects_replaced_path() {
        let directory = tempfile::tempdir().unwrap();
        let lock_path = directory.path().join(".config.lock");
        let replacement = directory.path().join("replacement");
        fs::write(&lock_path, b"original").unwrap();
        let file = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&lock_path)
            .unwrap();
        fs::write(&replacement, b"replacement").unwrap();
        fs::rename(&replacement, &lock_path).unwrap();

        let error = verify_open_lock_identity(&lock_path, &file).unwrap_err();
        assert!(format!("{error:#}").contains("不一致"));
    }

    #[cfg(unix)]
    #[test]
    fn config_lock_sets_permissions_through_open_handle() {
        use std::os::unix::fs::PermissionsExt;

        let directory = tempfile::tempdir().unwrap();
        let lock_path = directory.path().join(".config.lock");
        fs::write(&lock_path, b"").unwrap();
        fs::set_permissions(&lock_path, fs::Permissions::from_mode(0o666)).unwrap();

        let lock = ConfigLock::acquire(&directory.path().join("config.toml")).unwrap();
        assert_eq!(
            lock._file.metadata().unwrap().permissions().mode() & 0o777,
            0o600
        );
    }

    #[cfg(unix)]
    #[test]
    fn config_parent_rejects_symlink_non_directory_and_world_writable_boundary() {
        use std::os::unix::fs::{PermissionsExt, symlink};

        let root = tempfile::tempdir().unwrap();
        let real = root.path().join("real");
        fs::create_dir(&real).unwrap();
        let linked = root.path().join("linked");
        symlink(&real, &linked).unwrap();
        assert!(validate_config_path(&linked.join("config.toml")).is_err());

        let not_directory = root.path().join("not-directory");
        fs::write(&not_directory, b"").unwrap();
        assert!(validate_config_path(&not_directory.join("config.toml")).is_err());

        let shared = root.path().join("shared");
        fs::create_dir(&shared).unwrap();
        fs::set_permissions(&shared, fs::Permissions::from_mode(0o777)).unwrap();
        let error = validate_config_path(&shared.join("config.toml")).unwrap_err();
        assert!(format!("{error:#}").contains("组或其他用户写入"));
    }

    #[test]
    fn missing_schema_version_uses_v1_defaults_before_migration() {
        let (config, migrated) = decode_config("parallel_requests = 3\n").unwrap();
        assert!(migrated);
        assert_eq!(config.schema_version, 4);
        assert_eq!(config.chunk_seconds, 900);
        assert_eq!(config.max_output_tokens, 16_000);
        assert_eq!(config.split_output_tokens, 12_000);
        assert_eq!(config.parallel_requests, 1);
        assert_eq!(config.model, DEFAULT_MODEL);
        assert_eq!(config.quality_review_model, DEFAULT_MODEL);
        assert_eq!(config.asr_model, DEFAULT_ASR_MODEL);
        assert_eq!(config.quality_asr_model, DEFAULT_QUALITY_ASR_MODEL);
        assert_eq!(config.asr_provider, DEFAULT_ASR_PROVIDER);
        assert_eq!(config.quality_asr_provider, DEFAULT_QUALITY_ASR_PROVIDER);
    }
}
