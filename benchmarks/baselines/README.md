# 已记录基线

本目录保存可审查的指标快照，不保存音频、API Key、完整请求或 provider 响应。

`v0.5.0-synthetic-zh-aba.tsv` 使用 `fixtures/synthetic-zh-aba` 的同一份本机生成音频，比较默认 quality、3.7 raw overlay 和 Lite raw overlay。每个配置只运行一次，因此它只证明这一个合成用例在该次运行中的行为，不能推断真人会议 CER、说话人 DER、稳定时延或稳定成本。

`v0.5.0-ascend-zh-aba.tsv` 使用 `scripts/generate-ascend-zh-aba.sh` 从固定 revision 的 CAiRE/ASCEND 下载两个真人说话人的三段短语音，再人工拼成 A→B→A。它比较默认 quality 与 raw，每种模式同样只有一次运行。该样本是真人声音，但不是自然连续对话、房间会议或带重叠/插话/后段新人的录音，因此只能补充验证短文本和标签一致性，不能替代真人会议验收。

复现步骤：

1. 运行 `../scripts/generate-synthetic-zh-aba.sh` 生成被 Git 忽略的音频。
2. 通过 `SPT_BENCH_ALLOW_PAID=1 ../scripts/run-spt.sh ...` 分别运行目标配置。
3. 将每次 `report.tsv` 中的指标复制到新的、带版本号的基线文件。
4. 核对模型、provider、代码提交、音频生成参数和 `sample_count` 后再比较。

只有同一音频、同一代码、同一模式以及除被比较变量外完全相同的配置，才可称为受控 A/B。若代码或 prompt 同时变化，应记录为一次新的整体基线，不应把差异单独归因于模型。
