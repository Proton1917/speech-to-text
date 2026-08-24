# spt

`spt` 是一个以 OpenRouter 多模态模型为核心的 Rust 语音转文字 CLI。给出合法的本地音频路径后，它会在原文件旁生成同名 Markdown 文字稿：

```bash
spt "/path/to/会议录音.m4a"
# 输出：/path/to/会议录音.md
```

当前默认模型与 provider：

```text
model    = google/gemini-3.5-flash-lite
provider = google-vertex/global
```

截至 2026-08-23，OpenRouter 模型目录确认该模型支持 `audio` 和 `image` 输入；`google-vertex/global` 是它的有效 endpoint tag。

## 工作方式

```text
本地路径
  → 扩展名 allowlist + 普通文件/文件名校验
  → FFprobe 检查真实媒体内容和流类型
  → FFmpeg 一次性解码成单声道 32 kHz 无损 FLAC 母版
  → 规划连续、无重叠的 15 分钟 TARGET
  → 阶段 A 只把 exact TARGET 交给模型，生成正文、时间和片内 L1/L2
  → FFmpeg 能量覆盖提示发现明显空洞；异常时重听一次并记录 advisory
  → 阶段 B 把历史 S 参考、边界上下文和本片 L 候选合成短 MP3
  → 同一个模型只返回 L→S/NEW/UNKNOWN；Rust 持有并更新全局 S1/S2/S3
  → 全部片段成功后按时间顺序合并
  → 在源文件同目录以 `0600` 权限原子写入 Markdown
```

程序不会下载本地 diarization 模型，也不会把数小时音频整体读入内存。SpeakerHarness 只在任务内保存无损母版上的短参考范围，每次临时合成后上传给当前 OpenRouter provider，请求结束立即删除。临时目录为 `0700`、媒体为 `0600`，临时空间、HTTP 尝试次数和自适应深度都有硬上限。Ctrl-C 会协作式取消网络与 FFmpeg 并清理临时内容。阶段 A 正文无法可靠完成时整项失败；阶段 B 身份对齐失败时不会丢正文，而是显式降为 `UNKNOWN`。

## 安装

要求：

- Rust 1.90 或更新版本
- FFmpeg（同时需要 `ffmpeg` 和 `ffprobe`）
- OpenRouter API Key

macOS 可安装 FFmpeg：

```bash
brew install ffmpeg
```

构建并安装 `spt`：

```bash
cargo install --path . --locked
```

设置 API Key。Key 只从进程环境读取，不会写进配置或日志：

```bash
export OPENROUTER_API_KEY="你的 OpenRouter API Key"
```

## CLI

### 查看指令介绍

直接运行以下任一命令即可查看内置中文指令表、常用示例、输出规则和安全说明：

```bash
spt
spt --help
spt help
```

查看指定功能：

```bash
spt help audio
spt help ocr
spt help models
spt help providers
spt help config
```

`spt help` 在配置加载前处理，不会访问 OpenRouter，也不会修改当前模型或 provider。

### 转写音频

```bash
spt "会议录音.m4a"
spt --force "会议录音.m4a"
```

默认拒绝覆盖已存在的 `会议录音.md`。只有明确传入 `--force` 才会在完整处理成功后原子替换。

支持的输入扩展名大小写不敏感：

```text
.aac .aif .aiff .caf .flac .m4a .m4b .mp3
.oga .ogg .opus .wav .webm .wma
```

扩展名只是第一道校验；FFprobe 还必须确认文件确实包含音轨。默认入口拒绝目录、空文件、符号链接、伪装成音频的文本，以及包含非封面视频流的容器。

### 持久选择模型

```bash
spt --model google/gemini-3.5-flash-lite
```

OpenRouter 模型代号含 `/`，因此它是 `--model` 的参数值。正确写法是 `spt --model <MODEL_ID>`，不是动态选项 `spt --<MODEL_ID>`。

列出当前 OpenRouter 目录中声明支持音频输入的模型：

```bash
spt models
spt models gemini
```

自定义模型必须实际支持音频输入；OCR 还要求图片输入能力。

### 持久选择 provider

严格固定一个 provider endpoint：

```bash
spt --provider google-vertex/global
```

允许 OpenRouter 任意路由：

```bash
spt --provider any
```

`any` 只是一种本地持久配置状态。发请求时，程序会完全省略 OpenRouter 的 `provider` 字段，不会把字符串 `"any"` 发送给 API。

除 `any` 外，配置值必须精确匹配当前模型公开的 `endpoints[].tag`。base provider slug（如 `google-vertex`）可能匹配多个 endpoints，因此会被拒绝；先用 `spt providers` 查看可选的完整 tag。

固定 endpoint 还必须出现在 OpenRouter 的 ZDR 目录中。正式请求会附带 `data_collection=deny` 和 `zdr=true`；不满足零数据保留条件时，会在发送录音和参考声音前失败。`any` 为用户明确授权的隐私降级模式，仍按原契约完全省略 provider 字段。

列出当前模型或指定模型的 provider endpoints：

```bash
spt providers
spt providers google/gemini-3.5-flash-lite
```

模型与 provider 也可一次保存并立即用于当前文件：

```bash
spt \
  --model google/gemini-3.5-flash-lite \
  --provider google-vertex/global \
  "会议录音.m4a"
```

### 查看配置

```bash
spt config
```

默认配置文件为：

```text
~/.config/spt/config.toml
```

可用 `SPT_CONFIG_PATH` 指定其他位置。配置采用进程锁、原子替换和目录同步，Unix 下文件权限为 `0600`；其中不保存 API Key。相同输出也有跨进程锁，第二个任务会在付费请求前拒绝重复处理。

主要参数：

```toml
schema_version = 2
model = "google/gemini-3.5-flash-lite"
provider = "google-vertex/global" # 或 "any"
chunk_seconds = 900
overlap_seconds = 30
min_chunk_seconds = 30
max_output_tokens = 16000
split_output_tokens = 12000
parallel_requests = 1
retries = 5
max_adaptive_depth = 4
max_http_attempts = 1000
max_temp_bytes = 21474836480
max_speakers = 16
speaker_reference_seconds = 6
speaker_reference_silence_seconds = 1
speaker_context_chars = 4000
max_transcript_bytes = 67108864
max_total_turns = 100000
```

通常只需通过 CLI 修改 `model` 和 `provider`。其他参数用于针对特定录音密度和限流条件调整，不应把 `split_output_tokens` 设置到 `max_output_tokens` 以上。

v0.2 首次运行会把 schema v1 迁移为 v2。旧版标准值 `300/6000/5000/3` 会作为一次明确的行为升级改为 `900/16000/12000/1`；这不是“完全保留旧行为”的无损迁移，而是启用 SpeakerHarness 所必需的版本迁移。

### OCR

OCR 是独立入口，不会让默认音频命令接受任意文件：

```bash
spt ocr "扫描件.png"
# 输出：扫描件.ocr.md
```

当前支持单张静态 `.png`、`.jpg`、`.jpeg`、`.webp`。动画 PNG/WebP 会被拒绝；最长边不超过 32768 像素、总像素不超过 1 亿、文件不超过 64 MiB。发送前先在不放大小图的前提下缩至最长边 4096 像素以内，再把透明背景合成白色并编码为 JPEG。首版不处理 PDF 或多页文档。

## SpeakerHarness 与全局说话人

每个录音只维护一份任务内 SpeakerHarness：

```text
第一段 TARGET 00:00–15:00
  → 阶段 A 只听 00:00–15:00，使用局部 L1/L2
  → 阶段 B0 比较本片短候选，合并可能被过分割的同一声音
  → 模型返回 NEW1/NEW2，Rust 再原子建立全局 S1/S2
  → 从清晰单人发言保存最多约 6 秒的母版范围

第二段 TARGET 15:00–30:00
  → 阶段 A 只听 exact TARGET，冻结完整正文和新的 L1/L2
  → 阶段 B 在同一个短 packet 中听 S1/S2 参考与 L1/L2 候选
  → 只返回 L1→S2、L2→S1 等映射，不能改正文或时间
  → 新声音返回 NEW1，再由 Rust 分配为 S3
```

OpenRouter 请求本身没有隐式记忆。所谓“记住”是 Rust 明确保存全局编号、短参考范围和上一段有界尾文，并在下一轮重新注入。阶段 A 的尾文只辅助术语连续，不参与身份判断；阶段 B 才比较历史 `S#` 与本片 `L#` 的声音。模型不能直接创建最终 S 编号；无法可靠匹配时必须输出 `UNKNOWN`，不会强行归到某个已有说话人。

阶段 A 的 exact TARGET，以及阶段 B 的历史参考、最多 30 秒边界上下文和局部候选，全部直接从同一份无损母版取样，不做递归 MP3 重编码。阶段 B 只发送一个短合成 packet，因此不依赖 provider 对多音频附件的兼容行为。边界上下文和参考窗口从数据结构上就没有正文输出权。

阶段 A 还使用 FFmpeg `silencedetect` 做保守的本地能量覆盖提示：模型返回的 turns 中有长连续活动空洞、整体覆盖明显过低，或声明 `no_speech` 但本地存在持续能量时，会要求重新听一次。FFmpeg 无法区分人声、掌声、音乐、引擎声和环境噪声，所以第二次仍冲突时不会仅凭能量拒绝结构合法的正文，而是在 front matter 记录 `ffmpeg_energy_advisory_warning`，不把能量检测冒充 VAD。它不下载模型、不生成文字，也不能证明逐字 100% 无误。

该模式属于 `reference-assisted speaker matching`，可以维持整项任务的编号一致性，但不是生物声纹鉴定。极其相似的声音、变声、电话信道变化、极短插话或多人抢话仍可能得到 `UNKNOWN` 或误匹配；正式稿会在 front matter 中明确标记 `best_effort`，不会声称身份已验证。

## 长音频与 Token 边界

模型返回前无法精确知道转写会产生多少 Token。因此 `spt` 使用两层策略：

1. 请求前按 15 分钟 TARGET 主动切分，避免把完整长音频一次送入模型。
2. 单段返回 `finish_reason=length`、HTTP 413、明确的上下文超限错误，或可靠统计的可见输出达到 `split_output_tokens` 时，把这一段从无损母版的时间中点二分后重新转写。隐藏 reasoning tokens 不计入可见输出阈值。

如果阶段 A 已经缩短到 `min_chunk_seconds` 附近或达到 `max_adaptive_depth`，仍无法通过长度、循环或结构门禁，整项任务会失败，不会拼入截断或非法文字。所有 FFmpeg 能量覆盖冲突都只属于上述 advisory；没有真正 speech VAD 时不把非静音当作必须存在文字的证据。HTTP 408/409/429/500/502/503/504/524/529 和可识别的临时 provider 错误只做有限重试，并优先遵守数值型 `Retry-After`；不会偷偷更换用户固定的 provider。

TARGET 按连续、无重叠的无损时间轴覆盖全部采样。阶段 A 的时间坐标始终从 exact TARGET 的 `0` 开始，切点处的半句话也必须按实际可听后缀/前缀转写；身份阶段才听前 30 秒边界上下文。自适应二分严格先完成左半段的正文与身份状态，再处理右半段，不会产生两个并发编号空间。首版不做全文模糊去重，以免误删真实重复内容。

尾段不足 30 秒时会向前移动最后一个边界，让尾段至少保留 30 秒，而不是生成几秒钟的小请求；每个 TARGET 都不会超过配置的 15 分钟硬上限。

## 输出格式

文字稿以 Markdown front matter 记录：

- 源文件名、真实 codec/container 和音频时长；
- 请求模型/provider，以及 API 有报告时的模型/provider；缺失时明确回落为请求值；
- 分段数、被接受的模型响应数、API 已报告的 Token 使用量、reasoning tokens 和费用；
- 每段的确定性音频边界。
- 全局 speaker IDs、30 秒 overlap、对齐状态，以及 `best_effort` 身份保证边界。
- FFmpeg 能量覆盖状态；`advisory_warning` 只表示能量与 `no_speech` 冲突，不能当作语音判定。

时间标题表示本地无损母版的切分边界，不是模型猜测的逐句时间戳。字段使用 `reported_*` 前缀；`usage_reported_for_all_accepted_responses` 会说明已接受响应的统计是否完整。`reported_accepted_cost_usd` 统计最终正文响应和成功身份映射响应；语义重试、失败映射或自适应二分前被丢弃的响应可能已经产生额外费用。

## 开发验证

```bash
cargo fmt --check
cargo clippy --locked --all-targets -- -D warnings
cargo test --locked --all-targets
cargo build --locked --release
```

OpenRouter 是付费外部服务，默认单元测试不会发送真实 API 请求。

## 安全边界

- 媒体路径始终作为独立参数传给 FFmpeg，不经过 shell；文件名中的空格、中文或 shell 元字符不会被执行。
- 模型输出只作为不可信文本写入 Markdown，不参与路径构造，也不会触发命令或工具调用。
- 音频和图片提示中明确把媒体内的指令视为待转写/待识别内容，避免媒体内提示注入改变任务。
- 两阶段都使用 strict JSON Schema；Rust 再检查 TARGET 时间、局部标签、能量覆盖提示、L→S 映射、全局 ID、新说话人数和实际有声参考范围。阶段 A 正文冻结后，阶段 B 没有改写或删除正文的接口。
- 说话人参考只保存母版绝对时间范围；临时 packet 和 Base64 不写日志、不进入 Markdown、不持久保存。
- 参考声音会随每个后续 packet 再次发送给当前 provider；使用 `--provider any` 表示用户接受由 OpenRouter 自动选择承载这些参考声音的 provider。
- 请求采用 HTTPS；API Key 只进入敏感 `Authorization` header。
- 固定 provider 时先用实时 catalog 校验完整 endpoint tag，再使用 `provider.only`、关闭 fallback，并要求请求参数受该 endpoint 支持；只有用户执行 `spt --provider any` 才授权全局自动路由。
- Rust 负责 CLI、配置、网络、并发、状态与写盘；媒体解析和编码隔离在受控 FFmpeg 子进程中。
