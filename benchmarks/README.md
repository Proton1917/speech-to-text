# spt 转写基准

这套基准测量 `spt` 输出中的文字错误、敏感事实短语和说话人编号一致性。离线评测器
`spt-bench` 读取人工真值与已经生成的 Markdown，不执行转写，也不访问 OpenRouter。仓库当前
提供一个 macOS 合成用例、一个可手动联网生成的 ASCEND 真人短样本，以及 Git 忽略的私有用例
入口。公开或私有音频均不进入 Git。

当前交付状态：离线评测器、A→B→A 合成脚本、显式付费运行脚本和私有用例模板均已实现。
`benchmarks/scripts/test.sh` 负责 Rust 单元测试、CLI/NFC 样例、API Key 进程隔离回归和 Shell
语法检查。

## 文件与职责

| 路径 | 内容 | Git 状态 |
| --- | --- | --- |
| `src/spt-bench.rs` | 使用 `unicode-normalization` 的离线指标计算器 | 跟踪 |
| `fixtures/synthetic-zh-aba/` | 合成用例真值、敏感短语和生成参数入口 | 跟踪 TSV，忽略音频 |
| `scripts/generate-synthetic-zh-aba.sh` | 使用两个 macOS 中文声音生成 A→B→A 音频 | 跟踪脚本 |
| `scripts/generate-ascend-zh-aba.sh` | 从固定 ASCEND revision 验证并下载三段真人语音，生成 A→B→A fixture | 跟踪脚本，手动联网 |
| `public-fixtures/ascend-zh-aba/` | ASCEND 拼接音频、真值与生成记录 | 整个目录忽略 |
| `scripts/run-spt.sh` | 隔离执行一次 `spt`，记录运行信息并调用评测器 | 跟踪脚本，忽略结果 |
| `templates/private-case/` | 私有用例的三个 TSV 模板 | 跟踪 |
| `private/`、`private-fixtures.tsv` | 本地录音、真值与路径清单 | 忽略 |
| `results/<case>/<run>/` | 每次运行的音频副本、日志、转写、运行记录和指标 | 忽略 |

评测器使用独立的 `benchmarks/Cargo.toml` 与锁文件构建，需要 Rust/Cargo。合成音频还需要
macOS `say`、`ffmpeg` 和 `ffprobe`。实际转写默认使用仓库的 `target/release/spt`、
`OPENROUTER_API_KEY` 和每次 run 新建的隔离默认配置；可用 `--spt PATH` / `SPT_BENCH_SPT`
显式换二进制，用 `SPT_BENCH_CONFIG` 复制一份受控配置。ASCEND 真人短样本生成器需要 `curl`、`jq`、`ffmpeg`、
`ffprobe` 和 Cargo，并会访问 Hugging Face 的公开 Hub API 与 Dataset Viewer。

## 用例契约

每个用例目录包含三个 UTF-8 TSV 文件。字段之间必须使用制表符，注释行以 `#` 开头。
`case_id` 只能包含 ASCII 字母、数字、点、下划线和连字符，且不能是 `.` 或 `..`。

文字输入采用标准 Unicode NFC，`language` 当前限于 `zh-*`。`case.tsv` 必须声明
`unicode_normalization=NFC`；评测器会用 Unicode normalization 数据验证 `case.tsv`、参考 turn
和事实变体确实已是 NFC，而不是手工枚举 combining-mark 范围。非 NFC 真值会被拒绝；模型输出
则先真正规范化到 NFC 再计分，因此分解日文浊音和分解 Hangul 与对应预组字符按规范等价。
CER 归一化后为空的参考也会被拒绝，因为此时不存在合法分母。

`case.tsv` 保存用例身份与音频入口：

```text
case_id<TAB>synthetic-zh-aba
language<TAB>zh-CN
unicode_normalization<TAB>NFC
audio_path<TAB>audio.m4a
provenance<TAB>macos-say-generated-not-committed
speaker_sequence<TAB>A,B,A
```

`turns.tsv` 每行是一段按时间排序的人工真值。参考说话人使用匿名且稳定的 `A/B/C`，不使用
`spt` 输出的 `S1/S2`：

```text
A<TAB>预算是四十二万元。
B<TAB>测试环境周五之前就绪。
A<TAB>项目代号是阿尔法七号。如果预算没有批准，就不要上线。
```

`terms.tsv` 每行定义一个必须命中的事实短语。同一含义允许多种明确列出的等价写法：

```text
number<TAB>budget_amount<TAB>四十二万元|42万元|420000元
proper_name<TAB>project_code<TAB>阿尔法七号|阿尔法7号
negation<TAB>no_launch<TAB>不要上线
condition<TAB>budget_not_approved<TAB>如果预算没有批准|如果预算未批准
```

允许变体由真值作者逐项决定，其中至少一个变体必须真实出现在参考文字中。评测器先在参考文字
中定位该项的字符 span，再通过整篇字符对齐取得假设中的对应 span，最后要求该 span 与某个允许
变体完全相等。`42万元` 不会命中 `142万元`，`李雷` 不会命中 `李雷鸣`；负号、小数点、百分号
等符号也参与 exact 比较。

## 生成 A→B→A 合成音频

在 macOS 上运行：

```bash
./benchmarks/scripts/generate-synthetic-zh-aba.sh
```

脚本优先选择 `Tingting` 作为 A、`Meijia` 作为 B；缺少这些声音时，选择本机前两个不同的
中文声音。可以显式指定声音、语速和 turn 间静音：

```bash
SPT_BENCH_VOICE_A='Tingting' \
SPT_BENCH_VOICE_B='Meijia' \
SPT_BENCH_SAY_RATE=155 \
SPT_BENCH_PAUSE_SECONDS=0.75 \
./benchmarks/scripts/generate-synthetic-zh-aba.sh --force
```

输出为 `fixtures/synthetic-zh-aba/audio.m4a` 和 `generation.tsv`，两者均被 Git 忽略。
`generation.tsv` 记录实际声音、语速、静音和总时长。不同 macOS 版本或声音资源可能产生不同
波形，因此模型比较必须复用同一份已生成的 `audio.m4a`。

## 生成 ASCEND 真人 A→B→A 短样本

`generate-ascend-zh-aba.sh` 是手动联网生成器，不属于默认测试。它只下载三段已固定真值的公开
真人语音，不调用 `spt`，也不产生 OpenRouter 费用：

```bash
./benchmarks/scripts/generate-ascend-zh-aba.sh
```

生成器先验证数据集对象，再下载音频。固定输入如下：

| 顺序 | ASCEND test row | 数据集 id | 匿名角色 | 原 speaker | 标注时长 | 文字 |
| --- | ---: | --- | --- | ---: | ---: | --- |
| 1 | 400 | `00400` | A | 3 | 5.72 秒 | 就你要申请去交换你并不需要去说哦我要去哪一个department交换是直接选school |
| 2 | 904 | `00904` | B | 17 | 4.82 秒 | 就是你是在大学的时候哪一个阶段才萌生 |
| 3 | 401 | `00401` | A | 3 | 2.66 秒 | 我要去这个university做交换 |

数据对象与许可：

- 数据集：[CAiRE/ASCEND](https://huggingface.co/datasets/CAiRE/ASCEND)，即 *ASCEND: A
  Spontaneous Chinese-English Dataset for Code-switching in Multi-turn Conversation*；
- 固定 revision：`737e9800ae31be9932ba8464c80366559bd28424`；脚本要求当前公开 metadata 的
  SHA 和每个 Dataset Viewer 音频 URL 都绑定此 revision，否则拒绝下载；
- 论文：[Lovenia et al., ASCEND, LREC 2022](https://arxiv.org/abs/2112.06223)；
- 原数据许可：[Creative Commons Attribution-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-sa/4.0/)
  （CC-BY-SA-4.0）。本脚本把三段原语音重排，并插入两段各 0.8 秒静音后无损编码为 FLAC；
  该本地改编音频继续按 CC-BY-SA-4.0 使用，分发时必须保留归属、许可和修改说明。

脚本从 Dataset Viewer 每次动态读取短期音频 URL。在任何下载发生前，它会依次验证：

1. Hub metadata 的数据集身份、公开状态、revision SHA 和许可证；
2. 三行的 row index、`id`、`original_speaker_id`、transcription、duration、language 和音频类型；
3. 音频 URL 的 host、数据集、revision、config、split 与 row 路径；
4. 下载文件的大小、16 kHz 单声道音频结构和媒体时长；
5. 拼接结果时长为约 14.80 秒，且生成的 `case.tsv`、`turns.tsv` 和空 `terms.tsv` 满足 NFC
   benchmark 契约。

输出位于 `benchmarks/public-fixtures/ascend-zh-aba/`，其中 `generation.tsv` 保留作品标题、
Lovenia 等作者归属、论文、来源、许可链接和本次改编说明；整个目录由 Git 忽略。再次生成时使用
`--force`；完整新结果验证成功后才会替换旧目录。

这个对象是三段真实单人语音的人工拼接，不是原数据中的自然连续对话，也不是会议录音。它没有
重叠说话、远场噪声、多人插话或后段新增说话人，只能补充验证真实声线下的短文本和 A→B→A
标签一致性，不能据此宣称真人会议 CER、DER 或生产可用性。

## 构建并离线评估

构建命令把单个二进制写入被忽略的 `.build/`：

```bash
./benchmarks/scripts/build.sh
```

对已有 `spt` Markdown 计算指标：

```bash
./benchmarks/.build/spt-bench evaluate \
  --case ./benchmarks/fixtures/synthetic-zh-aba \
  --transcript /absolute/path/to/transcript.md \
  --run /absolute/path/to/run.tsv \
  --report /absolute/path/to/report.tsv
```

`--run` 可以省略；调用数与成本会从 Markdown front matter 的
`accounted_model_responses` 和 `reported_accounted_cost_usd` 读取。转写失败且没有 Markdown 时，
只提供 `--run`。评测器会保留失败、耗时、调用和成本字段，并把文字与说话人指标写成 `NA`。

最小 `run.tsv` 如下：

```text
run_id<TAB>manual-001
status<TAB>failure
exit_code<TAB>1
elapsed_ms<TAB>27800
http_attempts<TAB>1
failure_kind<TAB>content_filter
mode<TAB>quality
model<TAB>qwen/qwen3-asr-1.7b
quality_model<TAB>fish-audio/transcribe-1
provider<TAB>deepinfra
model_responses<TAB>NA
cost_usd<TAB>NA
```

字段 `model_responses` 和 `cost_usd` 优先于 Markdown front matter，便于导入外部请求账单。
`run-spt.sh` 当前将 `http_attempts` 写为 `NA`，因为 v0.5.0 Markdown 没有暴露实际 HTTP
尝试次数。

## 显式执行一次真实 OpenRouter 转写

付费运行由 `SPT_BENCH_ALLOW_PAID=1` 解锁。脚本会把音频复制到独立结果目录，默认调用
`../target/release/spt`，并在该 run 内新建 `config.toml` 与 state 目录；不会读取、迁移或修改用户的真实 spt 配置，也不把 API Key 写入任何文件。要测试自定义路由，先把不含 Key 的配置路径放入 `SPT_BENCH_CONFIG`。脚本读取 Key 后立即从
自身环境删除。manifest 入口通过同一 Bash 进程中的非导出变量调用 runner；Key 只在实际 `spt`
转写子进程中做单命令注入，`dirname`、`awk`、构建器、配置读取和评测器都收不到 Key。

```bash
export OPENROUTER_API_KEY='已配置的 OpenRouter API Key'

SPT_BENCH_ALLOW_PAID=1 \
./benchmarks/scripts/run-spt.sh \
  --case ./benchmarks/fixtures/synthetic-zh-aba \
  --mode quality \
  --spt ./target/release/spt
```

原始模式把 `--mode quality` 改为 `--mode raw`；每个根 TARGET 都运行第二路 STT 的对照使用
`--mode verify-all`。一次运行产生：

```text
benchmarks/results/<case-id>/<UTC-time>-<pid>/
  input.<audio-extension>
  input.md 或 input.raw.md
  stdout.log
  stderr.log
  spt-config.txt
  spt-config.stderr.log
  spt-version.txt
  config.toml
  run.tsv
  report.tsv
```

`spt-config.txt` 是转写前读取的隔离配置快照；`run.tsv` 固定记录 mode、Primary/Quality STT、
两个 STT provider、overlay model/provider、二进制版本、Git commit 与 dirty 状态，所以失败运行也保留实验条件。显式 `--audio` 的相对路径按 `--case` 目录解析，
避免从调用者工作目录误取同名文件。

进程退出码、毫秒耗时和可识别的 `content_filter`/超时错误来自本次运行。调用数与成本来自
成功 Markdown 中的任务账本；没有 usage 的失败请求可能已收费，所以该成本只能解释为
`spt` 已报告并纳入账本的金额。

## 私有录音

从模板建立一个本地用例：

```bash
cp -R ./benchmarks/templates/private-case ./benchmarks/private/meeting-001
```

填写 `case.tsv`、`turns.tsv`、`terms.tsv`，再把音频放到该目录或保存在其他本地位置。复制
`private-fixtures.example.tsv` 为 `private-fixtures.tsv`，每行写入：

```text
case_id<TAB>case_dir<TAB>audio_path
```

路径相对 manifest 所在目录解析，也可以使用绝对路径。按清单运行某个用例：

```bash
SPT_BENCH_ALLOW_PAID=1 \
./benchmarks/scripts/run-manifest-case.sh meeting-001 quality
```

`private/`、`private-fixtures.tsv` 与 `results/` 均被忽略。运行脚本仍会把所选音频上传给当前
OpenRouter provider；只有取得录音处理授权后才应执行。

## 指标口径

| 指标 | 计算方法 | 解释 |
| --- | --- | --- |
| `cer` | 对参考与假设文本删除空白、标点和符号，统一全角 ASCII 与英文大小写后，计算 Unicode 字符 Levenshtein 距离 ÷ 参考字符数 | `0` 为逐字符一致；插入很多时可大于 `1` |
| `number_exact_recall` | 在参考 span 对应的假设 span 上整段匹配允许的数字写法 | 保留正负号、小数点、百分号和数值边界 |
| `proper_name_exact_recall` | 在参考 span 对应的假设 span 上整段匹配允许的专名 | 不做无边界子串、近音或模糊匹配 |
| `negation_exact_recall` | 在参考 span 对应的假设 span 上整段匹配完整否定短语 | 检查否定语义是否保留 |
| `condition_exact_recall` | 在参考 span 对应的假设 span 上整段匹配完整条件短语 | 检查条件前提是否保留 |
| `speaker_permutation_invariant_turn_accuracy` | 仅对 turn CER `≤0.60` 的文本做单调对齐，再用匈牙利算法寻找预测标签到参考说话人的最大权重一一映射；正确数 ÷ `max(参考 turn 数, 假设 turn 数)` | 无关或幻觉 turn 不参与身份得分；整体换名不扣分，同一真实声音跨 turn 改号会扣分；支持 32 人用例 |
| `speaker_turn_error_rate` | `1 - speaker_permutation_invariant_turn_accuracy` | 这是 turn 级错误率 |
| `failed`、`elapsed_ms`、`http_attempts`、`accounted_model_responses`、`reported_accounted_cost_usd` | 来自 `run.tsv`，调用与成本可回落到 Markdown front matter | 用于比较失败率、时延和已报告费用 |

`spt` Markdown 当前只保存 TARGET 边界，没有每个 speaker turn 的绝对时间。评测器因此不计算
DER；turn 级指标不冒充带时间权重的 diarization error rate。若以后输出可信的逐 turn 时间戳，
可以在保持现有 turn 指标的同时另加 DER。

## 对照实验

模型、provider、模式和音频是比较时的控制变量。每个候选设置使用同一音频与同一用例目录，
运行结果写入不同的 `<run-id>` 目录。比较时至少同时查看：

- `failed` 与 `failure_kind`；
- `cer`；
- 四类 exact recall；
- 说话人 turn 准确率；
- `elapsed_ms`、调用数和已报告成本。

流畅程度不作为替代指标。一次合成用例只检验已写入真值的句子和两个系统声音；ASCEND fixture
也只是三个干净真人短片段的人工拼接。两者都不能证明真实会议录音已经达到生产质量。新增获授权
的自然会议用例后，应保留每个用例的逐项报告，再按固定用例集统计失败率、加权 CER 和有时间轴
真值支持的 DER。

经过人工核对、准备随版本发布的指标快照保存在 `benchmarks/baselines/`。基线文件不包含音频、
API Key 或完整 provider 响应，并且必须记录样本数和适用边界；一次运行不能被描述为稳定性能。

## 验证与清理

默认测试不联网、不读取 API Key，也不调用 `spt`；它只静态检查 ASCEND 生成器，不执行下载：

```bash
./benchmarks/scripts/test.sh
```

每次付费运行创建一个独立结果目录，历史不会被自动覆盖。检查完后按明确的用例和运行 ID 删除：

```bash
rm -rf ./benchmarks/results/<case-id>/<run-id>
```

合成音频和评测器构建产物可以分别删除：

```bash
rm ./benchmarks/fixtures/synthetic-zh-aba/audio.m4a
rm -rf ./benchmarks/.build
```

ASCEND 公开短样本可以按其确定目录单独删除，之后可由手动联网生成器恢复：

```bash
rm -rf ./benchmarks/public-fixtures/ascend-zh-aba
```

这些路径都由本仓库生成且处于 `benchmarks/` 下；私有原始录音的清理由录音所有者单独处理。
