# spt

spt 是一个以 OpenRouter 专用语音转写接口为正文来源的 Rust CLI。给出本地音频路径后，它会在音频旁生成 Markdown；图片 OCR 仍作为独立子命令提供。

当前发布版本为 0.5.0。本仓库只包含后端和 CLI，前端尚未开始。默认链路已经从“让通用多模态模型同时猜正文和说话人”改为“专用 STT 返回 provider source，Rust 冻结事实片段受保护的 OpenCC 展示投影，再做受约束的 turn 与说话人 overlay”。

## 快速开始

通过 Homebrew tap 安装 0.5.0：

~~~bash
brew install Proton1917/tap/spt
~~~

已经安装旧版本时，用同一 tap 升级：

~~~bash
brew upgrade Proton1917/tap/spt
~~~

也可以在仓库中从源码构建并直接运行产物：

~~~bash
cargo build --locked --release
export OPENROUTER_API_KEY="your-key"
./target/release/spt "/path/to/会议录音.m4a"
~~~

默认生成 /path/to/会议录音.md。默认 quality 只做 presentation-only 标点与邻接空格清理，并以首段加每第 5 个根 TARGET 的方式抽检第二路 ASR。需要每段都核验时使用：

~~~bash
./target/release/spt --verify-all "/path/to/会议录音.m4a"
~~~

未经第二路 ASR 交叉检查和主机清稿的原始模式：

~~~bash
./target/release/spt --raw "/path/to/会议录音.m4a"
~~~

它生成 /path/to/会议录音.raw.md。

--raw 跳过第二路 ASR 核对和 quality 清稿。语气词、结巴与停顿能否保留仍取决于底层 ASR。两种模式都明确写入：

~~~yaml
transcript_accuracy_verification: "not_measured"
~~~

## 运行架构

通用音频 LLM 负责分 turn 和比较声线。Rust 冻结的是经校验、事实片段受保护的 OpenCC 展示投影，不是 provider 原始响应字节；overlay 只能返回与这份展示正文 canonical 一致的 turn 结构，因此不能把“四十二万元”改成“40 万元”。

~~~text
本地音频
  → 文件名、扩展名、真实音频流和资源上限校验
  → 32 kHz 单声道无损 canonical FLAC
  → 连续、无重叠、最长 120 秒的 exact TARGET
  → Primary STT：Qwen3 ASR 1.7B，生成正文 authority
  → quality 默认：首段及每第 5 个根 TARGET 用 Fish Audio Transcribe 1 独立抽检
  → quality --verify-all：每个根 TARGET 都运行 Fish
  → Primary 非空时，Gemini 3.7 Flash：只返回 turn、时间和局部声音标签
  → Rust：把每个 turn.text 恢复为经校验、事实片段保护和 OpenCC 展示归一化后的 Primary 冻结切片
  → quality only：Rust 只做事实字符不变的标点/邻接空格清理，并标记可能的口吃；raw 完全跳过
  → SpeakerHarness：按逐 turn 短声音包映射为 S1、S2、NEW 或 UNKNOWN
  → 原子写入 Markdown
~~~

| 组件 | 输入 | 允许输出 | 不允许做的事 |
| --- | --- | --- | --- |
| Primary STT | exact TARGET 音频 | provider 响应正文、可选 usage | 不负责全局说话人身份 |
| Quality STT | 被抽检或 --verify-all 的 exact TARGET | 第二份独立文本、可选 usage | 不能覆盖 Primary 文本 |
| Turn overlay | 音频 + JSON 转义后的 Primary 文本 | turn 边界、时间、局部标签 | 不能改 canonical 事实字符 |
| SpeakerHarness Stage B | 历史短参考 + 本片逐 turn 候选 | T# → S#/NEW#/UNKNOWN | 不能生成或修改正文 |
| Rust host | 所有结构化结果 | 校验、编号、冻结 OpenCC 展示投影、presentation-only 清稿、渲染、记账 | 不根据词频猜专名或事实 |

## 默认模型和 provider

截至 2026-08-24，默认配置为：

| 角色 | 模型 | endpoint 预期值 | 用途 |
| --- | --- | --- | --- |
| Primary STT | qwen/qwen3-asr-1.7b | deepinfra | quality/raw 正文 |
| Quality STT | fish-audio/transcribe-1 | fish-audio | 仅 quality canonical 交叉检查 |
| Turn 与说话人 overlay | google/gemini-3.7-flash | google-vertex/global | turn 边界和逐 turn 声音比较 |
| OCR | google/gemini-3.7-flash | google-vertex/global | 单图 OCR |

专用 STT 使用 OpenRouter 的 POST /api/v1/audio/transcriptions：

- [OpenRouter Speech-to-Text](https://openrouter.ai/docs/guides/overview/multimodal/stt)
- [STT API Reference](https://openrouter.ai/docs/api/api-reference/stt/create-transcription)
- [STT 模型目录](https://openrouter.ai/api/v1/models?output_modalities=transcription&limit=1000)
- [ZDR endpoint 目录](https://openrouter.ai/api/v1/endpoints/zdr)

### STT provider 的真实安全边界

Chat API 支持 provider.only、allow_fallbacks=false、data_collection=deny 和 zdr=true。STT OpenAPI 当前只正式公开 provider-specific options，没有请求级 provider.only。

因此，STT fixed 模式采用以下合同：

1. 在任何付费请求之前查询实时模型目录；
2. 要求该 STT 模型只有一个公开 endpoint；
3. 要求唯一 endpoint 的 tag 与配置完全一致；
4. 要求同一模型和 tag 同时存在于实时 ZDR 目录；
5. 任一条件不成立时拒绝上传音频。

输出将该边界记录为：

~~~yaml
asr_provider_privacy_mode: "catalog_unique_zdr_preflight_no_request_level_pin"
~~~

这项保证止于目录预检。目录校验与请求之间存在很短的时效窗口；更严格的部署还应在 OpenRouter 账号或 guardrail 层启用 ZDR。

asr_provider=any、quality_asr_provider=any 或多模态 provider=any 都是显式隐私降级。

固定 STT 配置中的 deepinfra、fish-audio 等值是目录预检所要求的 endpoint 预期值，不是 STT 请求携带的 pin，也不能冒充响应已经确认的实际 endpoint。只有 STT 响应明确返回的 provider 才能记为 reported provider；未报告时保持 unreported，不用 expected 值补成 actual。自动路由同样不会把字符串 any 冒充实际 endpoint。

## quality 与 raw

### 默认 quality

~~~bash
spt "meeting.m4a"
~~~

quality 的执行顺序是：

1. 每个 TARGET 都运行 Primary STT；
2. 默认只在第 1、6、11……个根 TARGET 运行 Quality STT，也就是首段后约每 10 分钟抽检一次；`--verify-all` 改为每段核验；
3. 仅当 Primary 正文非空时运行 Gemini 3.7 turn overlay，最多两次结构尝试；
4. 仅当 overlay 产生可采样候选时运行一次可选 SpeakerHarness Stage B；
5. Rust 逐 turn 执行 presentation-only 清稿：只规范中文语境标点和标点旁单空格；`嗯/呃`、重复、专名、数字、否定、条件及其他所有 spoken words 原样保留，疑似口吃只写 signal，不自动删字；任何越权变化整 turn 回退 Primary。

Primary 返回空正文时，只有本段被抽检或使用 `--verify-all` 才运行第二路 STT；该结果只生成复核告警，不回填正文。turn overlay、Stage B 和清稿均跳过，Gemini 也不会生成正文或说话人标签。

两路 STT 使用 OpenCC 处理前的 provider source canonical 比较，避免 `臺/颱→台` 之类多对一转换制造假共识。比较可以忽略展示性空白和句读，但不会把以下内容视为等价：

- 四十二 与 40；
- 阿尔法七号 与 阿尔法十二号；
- 有否定词与无否定词；
- 3.14 与 314；
- -40 与 40；
- 大小写不同的字母标识符。

状态包括：

~~~text
exact_consensus_not_ground_truth
sampled_exact_consensus_not_ground_truth
disagreement_requires_review
unavailable_requires_review
~~~

前两种只确认被核验 TARGET 的两个独立 STT canonical 内容一致；第二种还明确存在因成本策略未核验的 TARGET。真人真值比较仍需要带参考文字的录音基准。出现分歧或第二路失败时，Primary STT 仍是正文来源，受影响片段会在标题和 front matter 中标记需复核。

~~~bash
spt --verify-all "meeting.m4a"
~~~

`--verify-all` 不改变正文 authority，只把 Fish 交叉核验覆盖从抽样提升到全部根 TARGET，因此费用更高。

### raw

~~~bash
spt --raw "meeting.m4a"
~~~

raw 不调用 Fish Audio，也不运行双 ASR 核对或 quality 清稿。Primary 正文非空时，它与 quality 一样使用 Gemini 3.7 做 turn/说话人 overlay；Primary 为空时跳过 overlay 和 Stage B。raw 的成本通常低于 quality，两种模式采用相同的 speaker 策略。

## 说话人处理

Turn overlay 只按真实换声切分，不应按逗号、句号或条件从句拆 turn。Primary provider source 仅在当前 TARGET 处理期间保留，用于跨 ASR 比较和生成展示投影，不会作为长期状态持久化。展示正文使用内置 OpenCC t2s，源字形保护范围限于：

- 由项目代号、代号、名称、姓名、公司名、品牌、机构、学校、型号、编号、账号等事实标签引出的值；
- 成对引号或书名号中的内容；
- inline code、URL 和 email；
- “乾坤的乾”这类显式指定字形。

无标签专名仍按普通 OpenCC t2s 转换，当前保护规则不能保证保留所有专名歧义字。模型返回后，Rust 验证全文 canonical 内容，并把每个 turn 替换回这份事实片段受保护的 Primary 展示切片。模型回显的改字不会进入结果；展示切片也不等于 provider 响应的原始字节。

Stage B 使用 host-owned T1/T2/T3...，不再把整个局部 L1/L2 一次映射为全局身份。A→B→A 可以表示为：

~~~text
T1 → NEW1
T2 → NEW2
T3 → NEW1

Rust allocation:
S1 → S2 → S1
~~~

身份包的硬上限：

- 最多 48 个 turn candidate；
- candidate 有声音频合计最多 120 秒；
- candidate 单段 1–10 秒；
- 历史 reference 最多 32 个；
- reference 单段 2–10 秒；
- 边界上下文最多 30 秒；
- 样本间静音最多 5 秒。

1 秒短 turn 可以与已有 S# 比较，也可以加入一个包含长 clean anchor 的 NEW# 组；全短 NEW# 组不能创建新 S。短 turn、重叠 turn 和未实际送入 Stage B 的 turn 不会成为持久 reference；无法可靠处理时输出 UNKNOWN。

后续 TARGET 出现新人时，Stage B 可以返回 NEW#，Rust 再分配新的全局 S 编号。历史参考只在当前任务内存在，不保存为跨文件声纹。

~~~yaml
speaker_label_assignment_status: "no_unknown_labels_present"
speaker_alignment_status: "not_verified"
speaker_identity_accuracy: "not_measured"
~~~

第一项只说明有没有 UNKNOWN，不表示分对。当前系统不是生物声纹认证，也没有真人录音 DER 证据。如果两个人的语音和正文已经被 overlay 合并进同一个 turn，Stage B 不能凭空把该 turn 拆开。

## CLI

~~~text
spt <AUDIO_PATH>
spt --verify-all <AUDIO_PATH>
spt --raw <AUDIO_PATH>
spt --force <AUDIO_PATH>

spt --asr-model <MODEL_ID>
spt --quality-asr-model <MODEL_ID>
spt --asr-provider <ENDPOINT_TAG|any>
spt --quality-asr-provider <ENDPOINT_TAG|any>

spt --model <MULTIMODAL_MODEL_ID>
spt --quality-model <MULTIMODAL_MODEL_ID>
spt --provider <ENDPOINT_TAG|any>

spt asr-models [SEARCH]
spt asr-providers [MODEL_ID]
spt models [SEARCH]
spt providers [MODEL_ID]
spt config
spt ocr <IMAGE_PATH>
spt help [COMMAND]
~~~

模型选项会持久保存：

- --asr-model：Primary STT；
- --quality-asr-model：quality 的第二路 STT；
- --verify-all：仅对当前任务生效，不持久化；把第二路 STT 从成本有界抽检改为每段核验；
- --model：同时设置 raw/quality turn overlay，并作为 OCR 模型；
- --quality-model：只覆盖 quality overlay；
- --provider：保存 Chat overlay endpoint；传入 `any` 允许 OpenRouter 自动路由，属于显式隐私降级；
- --asr-provider：保存 Primary STT endpoint 预期值；传入 `any` 接受自动路由，属于显式隐私降级；
- --quality-asr-provider：保存 Quality STT endpoint 预期值；传入 `any` 接受自动路由，属于显式隐私降级。

quality 要求两条 STT 路由可证明独立：model 与 provider 完全相同会拒绝；model 相同且任一路由为 `any` 也会因实际 endpoint 可能重合而拒绝；同一 model 的两个明确不同 fixed provider 可以继续接受各自 live catalog 校验。raw 不运行第二路 STT，因此不受这项独立性限制。

配置命令不显示 API Key。裸 spt、spt --help 和 spt help ... 离线执行。

## 配置 schema v4

配置路径按以下优先级解析：

1. 非空 SPT_CONFIG_PATH 指定的路径；
2. 非空 XDG_CONFIG_HOME 下的 XDG_CONFIG_HOME/spt/config.toml；
3. ~/.config/spt/config.toml。

首次执行音频转写、OCR、目录查询、`spt config` 或持久设置等非帮助操作时，如果配置不存在，程序会在上述路径原子写入默认 v4 配置，并在同目录创建零字节 `.config.lock`。裸 `spt`、`spt --help` 和 `spt help ...` 不读取或创建配置；已有 v1-v3 配置会在取得该锁后原子迁移为 v4。

已存在的配置父目录会先解析为真实路径，再校验其类型和权限边界；尚未完整存在的目录从最近的已存在祖先开始校验，创建边界中的 symlink/reparse point、非目录和不安全的 group/world-writable 目录会被拒绝。配置文件和 `.config.lock` 本身均以不跟随 symlink/reparse point 的方式打开，并通过已打开句柄复核 identity。TOML 解析失败只返回脱敏类别，不回显原始配置行，避免用户误把 Key 写进配置时泄漏到 stderr。

~~~toml
schema_version = 4

asr_model = "qwen/qwen3-asr-1.7b"
quality_asr_model = "fish-audio/transcribe-1"
asr_provider = "deepinfra"
quality_asr_provider = "fish-audio"

model = "google/gemini-3.7-flash"
quality_review_model = "google/gemini-3.7-flash"
provider = "google-vertex/global"

chunk_seconds = 900
min_chunk_seconds = 30
overlap_seconds = 30
parallel_requests = 1
retries = 5
max_http_attempts = 1000
max_temp_bytes = 21474836480
max_speakers = 16
speaker_reference_seconds = 6
speaker_reference_silence_seconds = 1
max_transcript_bytes = 67108864
max_total_turns = 100000
~~~

chunk_seconds 为兼容配置上限；专用 STT 的实际根 TARGET 仍硬限制为 120 秒。speaker_context_chars 是旧 schema 的兼容字段，v0.5 运行时不再使用 previous-tail 文本记忆。

旧配置会在取得同目录 .config.lock 后原子迁移到 v4，具体规则如下：

- v1：保留自定义 model/provider，并把同一个自定义 model 复制到 raw 与 quality 两条 overlay；旧官方默认 Lite 升级为两条 3.7 overlay。补入默认 Primary/Quality STT 路由。旧默认 chunk_seconds=300 升为 900，其他值最高收紧到 900；overlap_seconds 根据迁移后的 chunk 取 5–30 秒有效值；min_chunk_seconds 收紧到有效范围且不超过 chunk 的一半；旧默认 6000/5000 token 对升级为 16000/12000，其他合法自定义 token 对保留；parallel_requests 固定为 1。
- v2：保留自定义 model/provider，并把该 model 复制到两条 overlay；旧官方默认 Lite 升级为两条 3.7 overlay。补入默认 Primary/Quality STT 路由，超过 30 秒的旧 overlap 会收紧到有效上限。
- v3：保留已有的 model、quality_review_model 和 provider 双模型配置；只有旧官方 Lite + 3.7 这一精确组合会升级为两条 3.7 overlay。Lite + Lite 和其他自定义双模型组合保持原值，并补入默认 Primary/Quality STT 路由。

迁移不会读取、保存或打印 API Key。

## 输入、输出与覆盖规则

音频允许扩展名：

~~~text
aac aif aiff caf flac m4a m4b mp3 oga ogg opus wav webm wma
~~~

校验同时要求：

- 普通非空文件；
- 不是符号链接；
- 扩展名在 allowlist；
- 扩展名对应的 FFmpeg demuxer 在打开前被精确限制，伪装成音频的 concat/playlist 不能再读取嵌套文件或网络；
- FFprobe 确认存在真实音频流；
- 不含真实视频流。

输入目录项只打开一次：Unix 使用 `O_NOFOLLOW`，Windows 使用不跟随 reparse point 的句柄；确认是普通文件后，从该固定句柄有界复制到任务私有工作区。后续 FFprobe、FFmpeg 和 OpenRouter 上传只读取固定副本，避免共享可写目录中的路径替换把另一份文件送往外部 API。canonical FLAC 还会使用 `-xerror` 完整解码，并在具有可靠时长的容器上核对源时长；raw ADTS AAC 因 ffprobe 时长估算不可靠，以无错误解码后的 canonical 时长为准。

OCR 只接受 png、jpg、jpeg、webp 单图。

~~~text
meeting.m4a  → meeting.md
meeting.m4a  → meeting.raw.md  # --raw
scan.png     → scan.ocr.md      # ocr
~~~

默认不覆盖已有结果。--force 也只在完整任务成功后原子替换；网络、模型、结构、预算或写盘失败均保留旧文件。

## 安全与资源边界

- OPENROUTER_API_KEY 只从进程环境读取，不写入配置、日志、Markdown、Release 或 bottle；
- 所有模型输出都按不可信数据处理，不用于命令、路径、模型选择或 provider 选择；
- FFmpeg 参数通过 Command 参数数组传递，不拼接 shell；
- 请求媒体、响应、临时磁盘、turn 数、说话人数、HTTP attempts 和 transcript bytes 均有上限；
- content_filter、SAFETY 和 policy violation 不重试；错误报告实际请求次数；
- 16 MiB 响应超限和 32 MiB 请求媒体超限不会无界分配或重复付费；
- 所有音频请求顺序执行，确保 SpeakerHarness 状态按时间提交。

## 文件、状态目录和清理

正式运行只会长期保留：

| 路径 | 内容 | 增长行为 |
| --- | --- | --- |
| 音频旁的 .md / .raw.md / .ocr.md | 用户结果 | 每个源文件按模式最多一个，默认不覆盖 |
| 当前配置路径 | 持久配置 | 单文件原子替换 |
| 当前配置目录/.config.lock | 跨进程配置更新锁 | 固定一个零字节文件，不随任务增长 |
| 状态根目录/output-locks/ 下的 shard lock | 跨进程输出事务锁 | 按需创建，最多 4096 个零字节文件 |

输出锁使用稳定 FNV-1a 64-bit 的低 12 位映射到固定 4096 个 shard。所有平台都以已打开父目录的文件系统 identity 作为 key，同一目录内的 spt 输出会保守串行，从而覆盖 APFS/HFS+ 大小写、Unicode、firmlink，NTFS short-name，以及 Linux 上 vfat/exFAT/ntfs3/CIFS nocase 等文件系统 alias。锁文件长期保留但总数有硬上限；任务结束或进程崩溃时由操作系统释放文件锁，不依赖删除锁文件。不同目录若落到同一 shard，也只会保守串行，不会错误覆盖结果。默认状态根目录为 ~/.spt；Unix 上的绝对 `SPT_STATE_DIR` 会额外取得自定义 namespace 的锁，但不能替代每个进程都必须取得的默认 ~/.spt 锁，因此不同自定义值仍会互斥。Windows 为避免无法验证的共享 DACL，暂不允许覆盖状态根。程序不会擅自 chmod 已存在的 Unix 自定义状态根，只会保护自己创建的目录和专属 output-locks。该锁不再写入 ~/Library/Caches 或 ~/.cache，也不会按输出路径无限生成文件。

在迁移期，v0.5 若发现已存在的 v0.4 cache lock，会同时尝试持有旧锁；新安装不会重新创建旧 cache。旧版 `DefaultHasher` 跨工具链不保证完全稳定，因此这是 best-effort 兼容，不能反向修改正在运行的旧二进制。

固定输入副本、canonical FLAC、exact TARGET 和 speaker packet 位于系统临时目录，任务成功、普通错误或 Ctrl-C 后由 tempfile 清理。`SIGKILL`、进程 abort、断电或系统崩溃无法运行 Rust `Drop`，可能留下当前用户私有的 `spt-audio-*` / `spt-ocr-*` 目录；系统临时目录通常会自行回收，但这不是 spt 的强保证。确认没有 spt 任务运行后，可按 owner、目录前缀和修改时间人工检查并删除旧目录，不应使用指向整个临时根的递归删除命令。仓库自己的 target/、benchmarks/.build/、benchmarks/results/、public-fixtures/ 和合成音频属于开发产物，不是最终用户运行缓存。

## 基准与已验证结果

默认单元测试不访问 OpenRouter。真实 API 运行必须显式设置：

~~~bash
SPT_BENCH_ALLOW_PAID=1 ./benchmarks/scripts/run-spt.sh \
  --case ./benchmarks/fixtures/synthetic-zh-aba \
  --mode quality \
  --spt ./target/release/spt
~~~

离线评测器计算 CER、数字/专名/否定/条件 exact recall、permutation-invariant speaker turn accuracy、失败、时延、已入账模型响应数和 provider 已报告成本。

2026-08-24 的单次合成基线使用 macOS say 的 Tingting → Meijia → Tingting，源音频 11.624 秒。结果记录在 [v0.5.0 synthetic baseline](benchmarks/baselines/v0.5.0-synthetic-zh-aba.tsv)：

| 模式 | CER | 四类事实召回 | Speaker turn accuracy | 耗时 | 已入账模型响应数 | provider 已报告成本 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 默认 quality | 0.000000 | 全部 1.000000 | 1.000000 | 17.665 s | 4 | $0.003259811 |
| 3.7 raw | 0.000000 | 全部 1.000000 | 1.000000 | 17.406 s | 3 | $0.001917623 |
| Lite raw 对照 | 0.000000 | 全部 1.000000 | 0.333333 | 16.470 s | 3 | $0.000925890 |

这三个配置各运行一次，只证明该合成音频上的行为。它们不证明真人会议 CER、真实房间噪声鲁棒性、稳定时延、稳定成本或 DER。

仓库还提供手动生成的公开真人短样本：

~~~bash
./benchmarks/scripts/generate-ascend-zh-aba.sh
~~~

它固定使用 [CAiRE/ASCEND](https://huggingface.co/datasets/CAiRE/ASCEND) revision `737e9800ae31be9932ba8464c80366559bd28424` 的 test rows 400、904、401，按 CC BY-SA 4.0 许可与论文归属下载两个真人说话人的三段短语音，再插入两段 0.8 秒静音拼成 14.8 秒 A→B→A。生成器会核对 SHA、许可、row ID、speaker、参考文本、时长和媒体参数；音频与真值写入被 Git 忽略的 `benchmarks/public-fixtures/`。这是人工拼接的真人短样本，不是自然连续对话，更不是多人会议；没有重叠、远场噪声、插话或后段新人，不能用于生产级结论。用户或私有真人录音仍应放在被 Git 忽略的 `benchmarks/private/`，并确认处理授权。

v0.5.0 的预发布 live 快照对同一份 ASCEND fixture 各运行一次，结果如下；运行发生在最终 provenance 与本地安全加固之前，基线文件已明确记录 base commit、dirty 状态和 snapshot boundary，不能冒充最终 release commit 的逐字复现。该用例没有数字/专名/否定/条件 terms，因此四类事实召回为 `NA`：

| 模式 | CER | Speaker turn accuracy | 耗时 | 已入账模型响应数 | provider 已报告成本 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 默认 quality | 0.000000 | 1.000000 | 93.986 s | 4 | $0.003443490 |
| raw | 0.000000 | 1.000000 | 16.203 s | 3 | $0.001943490 |

两份输出顺序均为 `S1 → S2 → S1`。quality 的两路 STT source canonical 一致，本段只有一个根 TARGET，所以默认抽检覆盖等于全量覆盖；presentation-only 清稿没有改变任何 turn，`quality_host_cleanup_turns=0`。quality 本次明显更慢，属于一笔实时路由样本，不应解释为稳定时延分布。

“已入账模型响应数”包括写入结果账本的已接受响应，以及虽被正文或结构门禁拒绝、但客户端已收到可解析 usage 的响应；它不等于 HTTP attempt 总数。“provider 已报告成本”只累加这些响应 usage 中实际返回的 cost，不推测缺失值，也不是 OpenRouter 最终账单。缺失 usage、网络层失败或上游未报告的收费只能由 OpenRouter 账单确认。

~~~bash
./benchmarks/scripts/test.sh
~~~

## 已知边界

- 没有真人会议录音基准；公开 ASCEND fixture 只是人工拼接的 14.8 秒真人短样本，当前不能称为生产级会议纪要系统；
- 默认只抽检首段及每第 5 个根 TARGET；未抽检片段不会伪装成已核验，`--verify-all` 才覆盖全部根 TARGET；
- 双 ASR 一致不是 ground truth；
- OpenRouter STT 没有统一的跨 provider glossary 字段，顶层 prompt 会被接受但忽略；v0.5 仅使用 provider 明确公开的字段；
- 多 endpoint STT 模型在 fixed 模式下会被拒绝，因为请求级 provider pin 尚无公开合同；
- 说话人输出是任务内 best-effort，不是声纹认证；
- 抢话、重叠语音、极短新声、同一 turn 内合并了两个人的情况可能输出 UNKNOWN 或无法拆分；
- quality 不做 LLM 全文改写，也不凭文字静态删除 filler/重复，只执行 Rust presentation-only 清稿并记录疑似口吃 signal；它无法安全自动修正听错的专名、数字或整句语义，清稿门禁失败时保留 Primary 并标记需复核；
- --raw 不做第二路核对或 quality 清稿，但仍执行主机侧安全校验和事实片段保护的 OpenCC 展示归一化；它不是 provider 原始响应字节，也不保证保留每一个非词汇声音；
- reqwest 使用静态 WebPKI 根和环境代理变量，不自动读取 macOS Keychain/Linux 系统 CA，也不依赖仅存在于系统设置中的代理；企业 TLS inspection 或仅系统代理环境需要显式配置可用的标准环境变量/网络边界；
- 正常退出会清理临时媒体，无法对 `SIGKILL`、断电或进程崩溃承诺即时回收。
- 输出事务以稳定父目录 identity 选锁，但最终跨平台提交仍使用路径名原子 rename；若同一系统用户在任务进行中主动 rename 并重建输出父目录，当前版本不承诺把提交锚定到旧目录句柄。

## 本地构建与验证

源码构建要求 Rust 1.90+ 和 FFmpeg：

~~~bash
cargo build --locked --release
~~~

完整离线门禁：

~~~bash
cargo fmt --check
cargo clippy --locked --all-targets -- -D warnings
cargo test --locked --all-targets
cargo build --locked --release
./benchmarks/scripts/test.sh
~~~

最终产物是单一 CLI 二进制，不需要 Python、Node 或动态 OpenSSL。它不能准确宣传为“100% 纯 Rust 构建”：`ring` 与 `zstd-sys` 在源码构建时包含少量 C/汇编 build steps，因此从源码构建需要可用的本地 C toolchain；Homebrew bottle 安装不会把 Rust、LLVM 或这些构建依赖留在用户系统中。FFmpeg 是唯一明确的外部运行时边界，用于探测、无损 canonicalization、切片、能量提示和短 speaker packet。

## License

程序代码采用 MIT License。`benchmarks/scripts/generate-ascend-zh-aba.sh` 获取并改编的 CAiRE/ASCEND 来源材料采用 CC-BY-SA-4.0；归属、固定 revision、改编方式和再分发边界见 [benchmarks/README.md](benchmarks/README.md)。
