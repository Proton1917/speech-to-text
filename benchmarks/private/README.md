# 本地私有基准

此目录只保留这份说明。真实录音、人工真值和任何可识别信息都放在
`benchmarks/private/<case-id>/`；该路径已由上级 `.gitignore` 整体忽略。

从 `benchmarks/templates/private-case/` 复制三个 TSV 模板后再填写。不要把 API Key、
真实姓名或未获授权的音频路径写入被 Git 跟踪的文件。

真值与允许变体必须保存为标准 Unicode NFC，并在 `case.tsv` 声明
`unicode_normalization=NFC`。离线评测器使用 Unicode normalization 数据验证声明与内容；非 NFC
参考会被拒绝，模型输出则在计分前统一为 NFC。
