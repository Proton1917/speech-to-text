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
  → 按无损采样时间轴切成单代 64 kbps MP3，默认每段 5 分钟
  → 最多并发 3 个 OpenRouter 请求
  → 遇到输出 Token 阈值、finish_reason=length 或明显循环时递归二分
  → 全部片段成功后按时间顺序合并
  → 在源文件同目录以 `0600` 权限原子写入 Markdown
```

程序不会把数小时音频整体读入内存。请求并发会在读取与 Base64 编码之前限流；临时目录为 `0700`、媒体为 `0600`，临时空间、HTTP 尝试次数和自适应深度都有硬上限，已处理片段会及时删除。Ctrl-C 会协作式取消网络与 FFmpeg 并清理临时内容。任何片段失败时都不会生成一份看似完整的正式稿。

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
schema_version = 1
model = "google/gemini-3.5-flash-lite"
provider = "google-vertex/global" # 或 "any"
chunk_seconds = 300
min_chunk_seconds = 30
max_output_tokens = 6000
split_output_tokens = 5000
parallel_requests = 3
retries = 5
max_adaptive_depth = 4
max_http_attempts = 1000
max_temp_bytes = 21474836480
```

通常只需通过 CLI 修改 `model` 和 `provider`。其他参数用于针对特定录音密度和限流条件调整，不应把 `split_output_tokens` 设置到 `max_output_tokens` 以上。

### OCR

OCR 是独立入口，不会让默认音频命令接受任意文件：

```bash
spt ocr "扫描件.png"
# 输出：扫描件.ocr.md
```

当前支持单张静态 `.png`、`.jpg`、`.jpeg`、`.webp`。动画 PNG/WebP 会被拒绝；最长边不超过 32768 像素、总像素不超过 1 亿、文件不超过 64 MiB。发送前先在不放大小图的前提下缩至最长边 4096 像素以内，再把透明背景合成白色并编码为 JPEG。首版不处理 PDF 或多页文档。

## 长音频与 Token 边界

模型返回前无法精确知道转写会产生多少 Token。因此 `spt` 使用两层策略：

1. 请求前按 5 分钟主动切分，避免把完整长音频一次送入模型。
2. 单段返回 `finish_reason=length`、HTTP 413、明确的上下文超限错误，或可靠统计的可见输出达到 `split_output_tokens` 时，把这一段从无损母版的时间中点二分后重新转写。隐藏 reasoning tokens 不计入可见输出阈值。

如果已经缩短到 `min_chunk_seconds` 附近或达到 `max_adaptive_depth`，仍无法完整返回，整项任务会失败，不会拼入截断文字。HTTP 408/409/429/500/502/503/504/524/529 和可识别的临时 provider 错误只做有限重试，并优先遵守数值型 `Retry-After`；不会偷偷更换用户固定的 provider。

片段按连续、无重叠的无损时间轴覆盖全部采样；时间边界不丢音频，但模型对刚好跨边界的词句仍可能识别不一致。首版不做模糊重叠去重，以免误删真实重复内容。

## 输出格式

文字稿以 Markdown front matter 记录：

- 源文件名、真实 codec/container 和音频时长；
- 请求模型/provider，以及 API 有报告时的模型/provider；缺失时明确回落为请求值；
- 分段数、API 已报告的 Token 使用量、reasoning tokens 和已接收片段费用；
- 每段的确定性音频边界。

时间标题表示本地无损母版的切分边界，不是模型猜测的逐句时间戳。字段使用 `reported_*` 前缀；`usage_reported_for_all_segments` 会说明统计是否完整。`reported_accepted_cost_usd` 只统计最终被采纳且 API 报告用量的片段；自适应二分前被丢弃的异常响应可能已经产生额外费用。

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
- 请求采用 HTTPS；API Key 只进入敏感 `Authorization` header。
- 固定 provider 时先用实时 catalog 校验完整 endpoint tag，再使用 `provider.only`、关闭 fallback，并要求请求参数受该 endpoint 支持；只有用户执行 `spt --provider any` 才授权全局自动路由。
- Rust 负责 CLI、配置、网络、并发、状态与写盘；媒体解析和编码隔离在受控 FFmpeg 子进程中。
