# spt

`spt` 是一个以 OpenRouter 多模态模型为核心的 Rust 语音转文字 CLI。默认生成清理口语冗余、但不总结或改变事实的高质量 Markdown；显式使用 `--raw` 才生成原始逐字稿：

```bash
spt "/path/to/会议录音.m4a"
# 输出：/path/to/会议录音.md（高质量版）

spt --raw "/path/to/会议录音.m4a"
# 输出：/path/to/会议录音.raw.md（原始逐字版）
```

当前默认后端路由与 provider：

```text
基础转写 / raw / OCR / 说话人映射 = google/gemini-3.5-flash-lite
可疑 quality 片段独立重听          = google/gemini-3.7-flash
provider                            = google-vertex/global
```

截至 2026-08-24，OpenRouter 实时目录确认两个模型都支持 `audio`、`response_format` 和 `structured_outputs`，`google-vertex/global` 同时是两者的 ZDR endpoint。quality 的首个最长 5 分钟 TARGET 直接由 3.7 建立可靠术语与说话人起点；后续 Lite TARGET 只有被 Rust 门禁判定可疑时才升级，不会让整段音频无条件双跑。

## 工作方式

```text
本地路径
  → 扩展名 allowlist + 普通文件/文件名校验
  → FFprobe 检查真实媒体内容和流类型
  → FFmpeg 一次性解码成单声道 32 kHz 无损 FLAC 母版
  → raw 规划最长 15 分钟 TARGET；quality 规划最长 5 分钟 TARGET
  → quality 首个 TARGET 由 3.7 处理；后续阶段 A 由基础模型生成正文、时间和片内 L1/L2
  → Rust 使用内置 OpenCC t2s 将中文正文确定性归一化为 zh-Hans
  → quality：Rust 规范中文标点周边排版，并检测系统性汉字空格、重复、填充词、听不清和声学告警
  → quality 可疑：只将当前 TARGET 升级给 Gemini 3.7 Flash 独立重听
  → raw：跳过质量升级，始终由基础模型保留原始口语
  → 阶段 B 把历史 S 参考、边界上下文和本片 L 候选合成短 MP3
  → 基础模型只返回 L→S/NEW/UNKNOWN；Rust 持有并更新全局 S1/S2/S3
  → 全部片段成功后按时间顺序合并
  → 在源文件同目录以 `0600` 权限原子写入 Markdown
```

程序不会下载本地 diarization 或转写模型，也不会把数小时音频整体读入内存。Lite 与 3.7 路由共享同一个 HTTPS client、串行信号量和任务级 HTTP 次数预算，不能通过换模型绕过成本上限。简繁转换使用嵌入二进制的 OpenCC 文字词典，不增加 OpenRouter 调用。SpeakerHarness 只在任务内保存无损母版上的短参考范围，每次临时合成后上传给当前 OpenRouter provider，请求结束立即删除。临时目录为 `0700`、媒体为 `0600`，临时空间、HTTP 尝试次数和自适应深度都有硬上限。Ctrl-C 会协作式取消网络与 FFmpeg 并清理临时内容。

## 安装

推荐通过项目官方 Homebrew tap 安装：

```bash
brew install Proton1917/tap/spt
```

升级与卸载：

```bash
brew upgrade Proton1917/tap/spt
brew uninstall spt
```

Formula 从 GitHub 的不可变版本源码构建，自动安装运行时依赖 FFmpeg；构建、测试和安装过程都不需要 OpenRouter API Key。Key 仍只在实际转写时从当前进程的 `OPENROUTER_API_KEY` 环境变量读取，不会进入 Homebrew Formula、Cellar、GitHub Release 或配置文件。

也可以从源码手动安装：

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
# 默认：高质量可读稿
spt "会议录音.m4a"

# 原始：保留语气词、卡顿、重复和自我修正
spt --raw "会议录音.m4a"

spt --force "会议录音.m4a"
spt --raw --force "会议录音.m4a"
```

默认 quality 模式的首个最长 5 分钟 TARGET 直接使用 3.7，避免用 Lite 草稿建立错误术语；后续 TARGET 先使用 Lite。Rust 会确定性转换中文上下文的半角标点并清理标点两侧普通空格，但不会删除可能分隔姓名、列表或代码的汉字间空格；系统性异常空格会作为信号交给 3.7。门禁还检测机械重复、高密度强填充词、听不清标记、编码损坏、过多未完句和 FFmpeg 声学覆盖告警。没有发现信号时采用 Lite；命中可疑信号时，让 Gemini 3.7 Flash 对同一 TARGET 独立重听。复核后，Rust 只折叠已知连接词/代词结巴和纯应答连写，保留“好好学习、非常非常重要、人人”等正常叠词，并记录 `quality_host_cleanup_turns`。3.7 仍必须完整保留事实、数字、专名、观点、否定、条件、不确定性和任务要求，不总结、不改变立场、不重排内容。

Rust 门禁不会根据词频猜测公司名、中药名或技术名。表面通顺但实际听错的专名仍可能需要术语表或人工复核；经过 3.7 后仍存在声学/文本 advisory 时，标题和 front matter 会明确标记“含需复核片段”，不会再把警告隐藏在无条件“高质量”名称下面。

`--raw` 模式保留语气词、口头禅、结巴、卡顿、重复、自我修正、错误开头和不完整句，只补充必要标点。高质量版与原始版使用独立输出路径，可以同时存在：

```text
会议录音.md       quality / faithful_readability_cleanup
会议录音.raw.md   raw / verbatim
```

两种模式都默认拒绝覆盖各自已有的目标文件。只有明确传入 `--force`，才会在该模式完整处理成功后原子替换对应输出。

支持的输入扩展名大小写不敏感：

```text
.aac .aif .aiff .caf .flac .m4a .m4b .mp3
.oga .ogg .opus .wav .webm .wma
```

扩展名只是第一道校验；FFprobe 还必须确认文件确实包含音轨。默认入口拒绝目录、空文件、符号链接、伪装成音频的文本，以及包含非封面视频流的容器。

### 持久选择模型

```bash
spt --model google/gemini-3.5-flash-lite
spt --quality-model google/gemini-3.7-flash
```

OpenRouter 模型代号含 `/`，因此它是 `--model` 的参数值。正确写法是 `spt --model <MODEL_ID>`，不是动态选项 `spt --<MODEL_ID>`。

默认基础模型为 Gemini 3.5 Flash Lite，quality 模型为 Gemini 3.7 Flash。显式执行 `--model` 后，该自定义模型同时覆盖基础转写与质量复核，保持“设置后一直使用该模型”的原有合同；如需再次拆分两路，可使用 `--quality-model`。`spt config` 会同时显示两者。

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
schema_version = 3
model = "google/gemini-3.5-flash-lite"
quality_review_model = "google/gemini-3.7-flash"
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

通常只需通过 CLI 修改 `model`、`quality_review_model` 和 `provider`。`spt --model X` 同时设置两路，`spt --quality-model Y` 只调整质量路由。quality 的有效根 TARGET 上限还会由后端显示为 `effective_quality_chunk_seconds`，默认 300 秒；raw 继续使用 `chunk_seconds=900`。其他参数用于针对特定录音密度和限流条件调整，不应把 `split_output_tokens` 设置到 `max_output_tokens` 以上。

v0.4 首次运行会把 schema v1/v2 原子迁移为 v3。为了避免旧用户在不知情时把音频发送给另一模型，任何旧配置都先令 `quality_review_model=model`；升级用户可再显式执行 `spt --quality-model google/gemini-3.7-flash` 启用默认双路。旧版 v1 的标准值 `300/6000/5000/3` 仍会升级为 SpeakerHarness 所需的 `900/16000/12000/1`。

### OCR

OCR 是独立入口，不会让默认音频命令接受任意文件：

```bash
spt ocr "扫描件.png"
# 输出：扫描件.ocr.md
```

当前支持单张静态 `.png`、`.jpg`、`.jpeg`、`.webp`。动画 PNG/WebP 会被拒绝；最长边不超过 32768 像素、总像素不超过 1 亿、文件不超过 64 MiB。发送前先在不放大小图的前提下缩至最长边 4096 像素以内，再把透明背景合成白色并编码为 JPEG。首版不处理 PDF 或多页文档。

## 简体中文保证

模型提示仍要求输出简体中文，但模型指令不是硬约束。每个 Stage A turn 通过 JSON/时间校验后，Rust 会在进入 SpeakerHarness 状态、`previous_tail` 和 Markdown 之前执行内置 OpenCC `t2s` 转换。因此模型偶发返回的繁体或简繁混合文本会被确定性写成 `zh-Hans`，不需要第二次模型改写，也不会增加费用。

含有日文假名的 turn 会保持原文，避免把日语汉字误当成中文转换。OCR 继续忠实保留图片原文，不执行简繁转换。

## SpeakerHarness 与全局说话人

每个录音只维护一份任务内 SpeakerHarness：

```text
第一段 TARGET 00:00–15:00
  → 阶段 A 只听 00:00–15:00，使用局部 L1/L2
  → 阶段 B0 比较本片短候选，合并可能被过分割的同一声音
  → 模型返回 NEW1/NEW2，Rust 再原子建立全局 S1/S2
  → 从清晰单人发言保存最多约 6 秒的母版范围

第二段 TARGET 15:00–30:00
  → 阶段 A 只听 exact TARGET；quality 门禁可在冻结前让 3.7 独立重听
  → 阶段 B 在同一个短 packet 中听 S1/S2 参考与 L1/L2 候选
  → 只返回 L1→S2、L2→S1 等映射，不能改正文或时间
  → 新声音返回 NEW1，再由 Rust 分配为 S3
```

OpenRouter 请求本身没有隐式记忆。所谓“记住”是 Rust 明确保存全局编号、短参考范围和上一段有界尾文，并在下一轮重新注入。阶段 A 的尾文只辅助术语连续，不参与身份判断；阶段 B 才比较历史 `S#` 与本片 `L#` 的声音。模型不能直接创建最终 S 编号；无法可靠匹配时必须输出 `UNKNOWN`，不会强行归到某个已有说话人。

阶段 A 的 exact TARGET，以及阶段 B 的历史参考、最多 30 秒边界上下文和局部候选，全部直接从同一份无损母版取样，不做递归 MP3 重编码。阶段 B 只发送一个短合成 packet，因此不依赖 provider 对多音频附件的兼容行为。边界上下文和参考窗口从数据结构上就没有正文输出权。

阶段 A 还使用 FFmpeg `silencedetect` 做保守的本地能量覆盖提示：Lite 重听后仍有长连续活动空洞、整体覆盖明显过低，或声明 `no_speech` 但本地存在持续能量时，quality 会升级 3.7；raw 继续记录 advisory。FFmpeg 无法区分人声、掌声、音乐、引擎声和环境噪声，所以 3.7 后仍冲突时不会仅凭能量拒绝结构合法的正文，而是标记 `completed_with_advisory` 和“含需复核片段”，不把能量检测冒充 VAD。

该模式属于 `reference-assisted speaker matching`，可以维持整项任务的编号一致性，但不是生物声纹鉴定。极其相似的声音、变声、电话信道变化、极短插话或多人抢话仍可能得到 `UNKNOWN` 或误匹配；正式稿会在 front matter 中明确标记 `best_effort`，不会声称身份已验证。

## 长音频与 Token 边界

模型返回前无法精确知道转写会产生多少 Token。因此 `spt` 使用两层策略：

1. raw 请求前按最长 15 分钟 TARGET 主动切分；quality 使用最长 5 分钟根 TARGET，使首段 3.7 bootstrap 和后续按需升级保持可控。
2. 单段返回 `finish_reason=length`、HTTP 413、明确的上下文超限错误，或可靠统计的可见输出达到 `split_output_tokens` 时，把这一段从无损母版的时间中点二分后重新转写。隐藏 reasoning tokens 不计入可见输出阈值。

如果阶段 A 已经缩短到 `min_chunk_seconds` 附近或达到 `max_adaptive_depth`，仍无法通过长度、循环或结构门禁，整项任务会失败，不会拼入截断或非法文字。所有 FFmpeg 能量覆盖冲突都只属于上述 advisory；没有真正 speech VAD 时不把非静音当作必须存在文字的证据。HTTP 408/409/429/500/502/503/504/524/529 和可识别的临时 provider 错误只做有限重试，并优先遵守数值型 `Retry-After`；不会偷偷更换用户固定的 provider。

TARGET 按连续、无重叠的无损时间轴覆盖全部采样。阶段 A 的时间坐标始终从 exact TARGET 的 `0` 开始，切点处的半句话也必须按实际可听后缀/前缀转写；身份阶段才听前 30 秒边界上下文。自适应二分严格先完成左半段的正文与身份状态，再处理右半段，不会产生两个并发编号空间。首版不做全文模糊去重，以免误删真实重复内容。

尾段不足有效 `min_chunk_seconds` 时会向前移动最后一个边界，而不是生成几秒钟的小请求；raw TARGET 不超过 15 分钟，quality TARGET 不超过 5 分钟各自的硬上限。

## 输出格式

文字稿以 Markdown front matter 记录：

- 源文件名、真实 codec/container 和音频时长；
- `transcript_mode` 与 `transcript_editing`，明确区分默认高质量稿和原始逐字稿；
- `transcription_strategy`、基础模型、quality 模型、bootstrap/复核片段数、初始触发原因、复核后残留 advisory 及最终正文来源；
- 请求模型/provider，以及 API 有报告时的模型/provider；缺失时明确回落为请求值；
- 分段数、被接受的模型响应数、API 已报告的 Token 使用量、reasoning tokens 和费用；
- 每段的确定性音频边界。
- 全局 speaker IDs、30 秒 overlap、对齐状态，以及 `best_effort` 身份保证边界。
- FFmpeg 能量覆盖状态；`advisory_warning` 只表示能量与 `no_speech` 冲突，不能当作语音判定。
- `chinese_script: zh-Hans` 与 `chinese_normalization: opencc-t2s`，记录确定性简体归一化。

时间标题表示本地无损母版的切分边界，不是模型猜测的逐句时间戳。字段使用 `reported_*` 前缀；`accounted_model_responses` 和 `reported_accounted_cost_usd` 汇总所有被任务账本保留的完整响应，包括最终正文、触发质量升级的 Lite 草稿、可归属到后续叶片的 split 前响应和成功身份映射。HTTP/结构重试中没有可用 usage 的失败响应仍可能产生目录无法报告的额外费用；`reported_responses_by_model` 与 `reported_cost_usd_by_model` 给出模型级拆分。

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
- Lite、3.7 与身份阶段都使用 strict JSON Schema；Rust 再检查 TARGET 时间、局部标签、质量信号、能量覆盖提示、L→S 映射、全局 ID、新说话人数和实际有声参考范围。3.7 复核发生在正文冻结和身份对齐之前；阶段 B 没有改写或删除正文的接口。
- 说话人参考只保存母版绝对时间范围；临时 packet 和 Base64 不写日志、不进入 Markdown、不持久保存。
- 参考声音会随每个后续 packet 再次发送给当前 provider；使用 `--provider any` 表示用户接受由 OpenRouter 自动选择承载这些参考声音的 provider。
- 请求采用 HTTPS；API Key 只进入敏感 `Authorization` header。
- 固定 provider 时先用实时 catalog 校验完整 endpoint tag，再使用 `provider.only`、关闭 fallback，并要求请求参数受该 endpoint 支持；只有用户执行 `spt --provider any` 才授权全局自动路由。
- Rust 负责 CLI、配置、网络、并发、状态与写盘；媒体解析和编码隔离在受控 FFmpeg 子进程中。
