use std::env;
use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::{Result, bail};
use clap::{CommandFactory, Parser, Subcommand};

use spt::config::{ANY_PROVIDER, Config, ConfigLock, validate_model_id, validate_provider_id};
use spt::openrouter::OpenRouterClient;
use spt::transcript::TranscriptMode;

#[derive(Debug, Parser)]
#[command(
    name = "spt",
    version,
    about = "安全、可持久配置的 OpenRouter 语音转文字 CLI",
    long_about = "spt 使用 OpenRouter 专用 speech-to-text 模型生成 provider source；Rust 校验后冻结事实片段受保护的 OpenCC 展示投影，再由受约束的音频多模态模型切分 turn，并由 Rust SpeakerHarness 维护跨片段说话人编号。\n\n直接给出合法音频路径后，会在源文件同一目录生成同名 Markdown 文字稿。专用 STT TARGET 最长 120 秒，以规避上游处理超时；模型和 provider 预期值会持久保存，直到再次修改。",
    arg_required_else_help = false,
    disable_help_subcommand = true,
    after_help = "常用指令：\n  spt <AUDIO_PATH>                         成本有界的 quality 清稿，输出同名 .md\n  spt --verify-all <AUDIO_PATH>            每个 TARGET 都运行第二路 ASR 核验\n  spt --raw <AUDIO_PATH>                   单 ASR 未清稿，输出同名 .raw.md\n  spt --force <AUDIO_PATH>                 完整成功后原子替换已有目标稿件\n  spt --asr-model <MODEL_ID>               持久设置正文专用 STT 模型\n  spt --quality-asr-model <MODEL_ID>       持久设置 quality 交叉检查 STT 模型\n  spt --asr-provider <TAG|any>             保存正文 STT endpoint；any 为隐私降级\n  spt --quality-asr-provider <TAG|any>     保存检查 STT endpoint；any 为隐私降级\n  spt --model <MODEL_ID>                   同时设置 raw/quality overlay 模型\n  spt --quality-model <MODEL_ID>           仅设置 quality overlay 模型\n  spt --provider <ENDPOINT_TAG|any>        设置 overlay provider；any 为隐私降级\n  spt asr-models [SEARCH]                  列出专用 speech-to-text 模型\n  spt asr-providers [MODEL_ID]             列出专用 STT 模型 endpoints\n  spt models [SEARCH]                      列出 Chat Audio 多模态模型\n  spt providers [MODEL_ID]                 列出多模态模型 endpoints\n  spt config                               查看生效配置，不显示 API Key\n  spt ocr <IMAGE_PATH>                     OCR 单张图片，生成 *.ocr.md\n  spt help [COMMAND]                       显示完整介绍或指定子命令帮助\n\n示例：\n  spt \"会议录音.m4a\"\n  spt --verify-all \"会议录音.m4a\"\n  spt --raw \"会议录音.m4a\"\n  spt --asr-model qwen/qwen3-asr-1.7b --asr-provider deepinfra\n  spt --quality-asr-model fish-audio/transcribe-1 --quality-asr-provider fish-audio\n  spt --model google/gemini-3.7-flash --provider google-vertex/global\n\n说明：\n  - 默认正文来自 OpenRouter 专用 STT；Rust 冻结经校验、事实片段受保护的 OpenCC 展示投影，而非 provider 原始响应字节。\n  - quality 对首段及此后约每 10 分钟抽检独立 STT，--verify-all 改为全量核验；一致仍不等于真值。空 Primary 的第二路结果只记录复核告警，不回填正文。\n  - 多模态模型只负责把已冻结的展示正文切成 turn 和比较短声音样本，不得改写事实正文。\n  - 默认 quality 只做事实字符不变的主机标点清理；疑似 filler/口吃只标 signal，不自动删字；--raw 跳过清稿和第二路 ASR。\n  - STT API 暂不支持 Chat API 的 provider.only；固定模式只接受目录中唯一且为 ZDR 的 endpoint。\n  - OPENROUTER_API_KEY 只从环境变量读取，不会写入配置。\n  - 中文正文只保护事实标签值、成对引号/书名号内容、inline code、URL/email 和显式指定字形；无标签专名仍按普通 OpenCC t2s 转换。\n  - 首次执行非帮助操作时会原子写入默认配置并创建同目录 .config.lock；v1-v3 配置会在锁内迁移为 v4。\n  - 配置路径优先级为 SPT_CONFIG_PATH、XDG_CONFIG_HOME/spt/config.toml、~/.config/spt/config.toml。\n  - 默认不覆盖已有输出；只有 --force 会在完整结果就绪后原子替换。"
)]
struct Cli {
    /// 要转写的本地音频文件
    #[arg(value_name = "AUDIO_PATH")]
    audio_path: Option<PathBuf>,

    /// 保存模型并同时覆盖 raw/quality turn overlay 与 OCR 路由
    #[arg(long, global = true, value_name = "MODEL_ID")]
    model: Option<String>,

    /// 单独保存 quality turn overlay 模型；--model 会同时覆盖两条 overlay
    #[arg(long, global = true, value_name = "MODEL_ID")]
    quality_model: Option<String>,

    /// 保存正文专用 speech-to-text 模型
    #[arg(long, global = true, value_name = "MODEL_ID")]
    asr_model: Option<String>,

    /// 保存 quality 模式的独立 STT 交叉检查模型
    #[arg(long, global = true, value_name = "MODEL_ID")]
    quality_asr_model: Option<String>,

    /// 保存完整 provider endpoint tag；any 允许 OpenRouter 任意路由，属于显式隐私降级
    #[arg(long, global = true, value_name = "PROVIDER_ID|any")]
    provider: Option<String>,

    /// 保存正文 STT 的目录 endpoint 预期值；any 接受自动路由，属于显式隐私降级
    #[arg(long, global = true, value_name = "PROVIDER_ID|any")]
    asr_provider: Option<String>,

    /// 保存 quality STT 的目录 endpoint 预期值；any 接受自动路由，属于显式隐私降级
    #[arg(long, global = true, value_name = "PROVIDER_ID|any")]
    quality_asr_provider: Option<String>,

    /// 原子替换已经存在的 Markdown 输出
    #[arg(long, global = true)]
    force: bool,

    /// 输出未经 spt 清稿的单路 ASR 结果；默认只抽检第二路 ASR
    #[arg(long)]
    raw: bool,

    /// quality 模式对每个 TARGET 运行第二路 STT；默认只做成本有界抽检
    #[arg(long, conflicts_with = "raw")]
    verify_all: bool,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// 从单张本地图片提取文字，生成 *.ocr.md
    Ocr {
        /// 要识别的图片文件
        #[arg(value_name = "IMAGE_PATH")]
        image_path: PathBuf,
    },
    /// 列出当前支持音频输入的 OpenRouter 模型
    Models {
        /// 按模型代号或名称过滤
        #[arg(value_name = "SEARCH")]
        search: Option<String>,
    },
    /// 列出 OpenRouter 专用 speech-to-text 模型
    AsrModels {
        /// 按模型代号或名称过滤
        #[arg(value_name = "SEARCH")]
        search: Option<String>,
    },
    /// 列出指定模型可用的 OpenRouter provider endpoints
    Providers {
        /// 默认使用当前已保存的模型
        #[arg(id = "provider_target_model", value_name = "MODEL_ID")]
        target_model: Option<String>,
    },
    /// 列出指定专用 STT 模型当前公开的 provider endpoints
    AsrProviders {
        /// 默认使用当前已保存的正文 STT 模型
        #[arg(id = "asr_provider_target_model", value_name = "MODEL_ID")]
        target_model: Option<String>,
    },
    /// 显示生效配置和配置文件位置，不显示 API Key
    Config,
    /// 显示完整中文指令介绍，或查看指定子命令帮助
    Help {
        /// 可选：audio、ocr、asr-models、asr-providers、models、providers、config 或 help
        #[arg(value_name = "COMMAND")]
        command: Option<String>,
    },
}

#[tokio::main]
async fn main() -> ExitCode {
    let result = tokio::select! {
        result = run() => Some(result),
        signal = tokio::signal::ctrl_c() => {
            match signal {
                Ok(()) => eprintln!("已取消：正在终止网络请求和 FFmpeg，并清理临时文件"),
                Err(error) => eprintln!("错误：无法监听中断信号：{error}"),
            }
            None
        }
    };
    let Some(result) = result else {
        return ExitCode::from(130);
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("错误：{error:#}");
            ExitCode::FAILURE
        }
    }
}

async fn run() -> Result<()> {
    let cli = Cli::parse();
    if cli.audio_path.is_none()
        && cli.command.is_none()
        && cli.model.is_none()
        && cli.quality_model.is_none()
        && cli.asr_model.is_none()
        && cli.quality_asr_model.is_none()
        && cli.provider.is_none()
        && cli.asr_provider.is_none()
        && cli.quality_asr_provider.is_none()
        && !cli.force
        && !cli.raw
        && !cli.verify_all
    {
        print_command_help(None)?;
        return Ok(());
    }
    if let Some(Commands::Help { command }) = cli.command.as_ref() {
        if cli.audio_path.is_some()
            || cli.model.is_some()
            || cli.quality_model.is_some()
            || cli.asr_model.is_some()
            || cli.quality_asr_model.is_some()
            || cli.provider.is_some()
            || cli.asr_provider.is_some()
            || cli.quality_asr_provider.is_some()
            || cli.force
            || cli.raw
            || cli.verify_all
        {
            bail!("spt help 不能与音频路径、配置选项、--force、--raw 或 --verify-all 同时使用");
        }
        print_command_help(command.as_deref())?;
        return Ok(());
    }
    validate_operation_scope(&cli)?;
    let (mut config, config_path, config_existed, config_migrated) = Config::load()?;
    let loaded_config = config.clone();
    let overrides = ConfigOverrides {
        model: cli.model.as_deref(),
        quality_model: cli.quality_model.as_deref(),
        asr_model: cli.asr_model.as_deref(),
        quality_asr_model: cli.quality_asr_model.as_deref(),
        provider: cli.provider.as_deref(),
        asr_provider: cli.asr_provider.as_deref(),
        quality_asr_provider: cli.quality_asr_provider.as_deref(),
    };
    let changed = apply_config_overrides(&mut config, overrides)?;
    config.validate()?;
    if changed {
        let client = OpenRouterClient::from_environment(config.clone(), false)?;
        client.validate_selection("audio").await?;
        if config.effective_quality_review_model() != config.model {
            client
                .routed_to_model(config.effective_quality_review_model())?
                .validate_selection("audio")
                .await?;
        }
        client
            .validate_stt_selection(&config.asr_model, &config.asr_provider)
            .await?;
        if config.effective_quality_asr_model() != config.asr_model
            || config.quality_asr_provider != config.asr_provider
        {
            client
                .validate_stt_selection(
                    config.effective_quality_asr_model(),
                    &config.quality_asr_provider,
                )
                .await?;
        }
    }
    if changed || !config_existed || config_migrated {
        let config_lock = ConfigLock::acquire(&config_path)?;
        let (mut latest, latest_path, latest_existed, latest_migrated) = Config::load()?;
        if changed && latest != loaded_config {
            bail!("配置在网络校验期间被其他进程修改，请重新执行本次设置命令");
        }
        apply_config_overrides(&mut latest, overrides)?;
        latest.validate()?;
        if changed || !latest_existed || latest_migrated {
            latest.save(&latest_path)?;
        }
        config = latest;
        drop(config_lock);
        if changed {
            eprintln!("配置已保存：{}", config_path.display());
        } else if config_migrated {
            eprintln!(
                "配置已迁移到专用 STT + 多模态 overlay schema v4：{}",
                config_path.display()
            );
        }
    }

    if let Some(command) = cli.command {
        if cli.audio_path.is_some() {
            bail!("音频路径不能与子命令同时使用");
        }
        match command {
            Commands::Ocr { image_path } => {
                let output = spt::pipeline::ocr(&image_path, &config, cli.force).await?;
                println!("{}", output.display());
            }
            Commands::Models { search } => {
                reject_force(cli.force)?;
                let client = OpenRouterClient::from_environment(config, false)?;
                let models = client.list_audio_models(search.as_deref()).await?;
                if models.is_empty() {
                    println!("没有找到匹配的音频多模态模型");
                } else {
                    for model in models {
                        println!(
                            "{}\t{}\tcontext={}\tmax_output={}",
                            model.id, model.name, model.context_length, model.max_completion_tokens
                        );
                    }
                }
            }
            Commands::AsrModels { search } => {
                reject_force(cli.force)?;
                let client = OpenRouterClient::from_environment(config, false)?;
                let models = client.list_stt_models(search.as_deref()).await?;
                if models.is_empty() {
                    println!("没有找到匹配的专用 speech-to-text 模型");
                } else {
                    for model in models {
                        println!(
                            "{}\t{}\tcontext={}\tmax_output={}",
                            model.id, model.name, model.context_length, model.max_completion_tokens
                        );
                    }
                }
            }
            Commands::Providers { target_model } => {
                reject_force(cli.force)?;
                let model = target_model.unwrap_or_else(|| config.model.clone());
                validate_model_id(&model)?;
                let client = OpenRouterClient::from_environment(config, false)?;
                let providers = client.list_providers(&model).await?;
                if providers.is_empty() {
                    println!("模型 {model} 当前没有公开 provider endpoint");
                } else {
                    for provider in providers {
                        println!(
                            "{}\t{}\tcontext={}\tmax_output={}",
                            provider.tag,
                            provider.name,
                            provider.context_length,
                            provider.max_completion_tokens
                        );
                    }
                }
            }
            Commands::AsrProviders { target_model } => {
                reject_force(cli.force)?;
                let model = target_model.unwrap_or_else(|| config.asr_model.clone());
                validate_model_id(&model)?;
                let client = OpenRouterClient::from_environment(config, false)?;
                let providers = client.list_providers(&model).await?;
                if providers.is_empty() {
                    println!("专用 STT 模型 {model} 当前没有公开 provider endpoint");
                } else {
                    for provider in providers {
                        println!(
                            "{}\t{}\tcontext={}\tmax_output={}",
                            provider.tag,
                            provider.name,
                            provider.context_length,
                            provider.max_completion_tokens
                        );
                    }
                }
            }
            Commands::Config => {
                reject_force(cli.force)?;
                print_config(&config, &config_path);
            }
            Commands::Help { .. } => unreachable!("help 已在配置加载前处理"),
        }
        return Ok(());
    }

    if let Some(audio_path) = cli.audio_path {
        let mode = if cli.raw {
            TranscriptMode::Raw
        } else {
            TranscriptMode::Quality
        };
        let output =
            spt::pipeline::transcribe(&audio_path, &config, cli.force, mode, cli.verify_all)
                .await?;
        println!("{}", output.display());
        return Ok(());
    }

    if changed {
        println!("model={}", config.model);
        println!("quality_review_model={}", config.quality_review_model);
        println!("asr_model={}", config.asr_model);
        println!("quality_asr_model={}", config.quality_asr_model);
        println!("provider={}", config.provider);
        println!("asr_provider={}", config.asr_provider);
        println!("quality_asr_provider={}", config.quality_asr_provider);
        return Ok(());
    }

    Cli::command().print_help()?;
    println!();
    Ok(())
}

fn validate_operation_scope(cli: &Cli) -> Result<()> {
    if cli.raw && cli.command.is_some() {
        bail!("--raw 只适用于音频转写，不能与子命令同时使用");
    }
    if cli.raw && cli.audio_path.is_none() {
        bail!("--raw 需要同时提供音频路径");
    }
    if cli.verify_all && cli.command.is_some() {
        bail!("--verify-all 只适用于音频转写，不能与子命令同时使用");
    }
    if cli.verify_all && cli.audio_path.is_none() {
        bail!("--verify-all 需要同时提供音频路径");
    }
    if cli.force
        && cli.audio_path.is_none()
        && !matches!(cli.command.as_ref(), Some(Commands::Ocr { .. }))
    {
        bail!("--force 只适用于音频转写或 OCR 输出");
    }
    Ok(())
}

fn print_command_help(command: Option<&str>) -> Result<()> {
    let mut root = Cli::command();
    match command {
        None => root.print_long_help()?,
        Some(name) => {
            let available =
                "ocr、asr-models、asr-providers、models、providers、config、help、audio";
            let guide = command_topic_guide(name)
                .ok_or_else(|| anyhow::anyhow!("未知帮助主题 {name:?}；可选：{available}"))?;
            println!("{guide}");
            return Ok(());
        }
    }
    println!();
    Ok(())
}

fn command_topic_guide(name: &str) -> Option<&'static str> {
    match name {
        "audio" | "transcribe" | "转写" => Some(
            "音频转写\n\n用法：\n  spt <AUDIO_PATH>\n  spt --verify-all <AUDIO_PATH>\n  spt --raw <AUDIO_PATH>\n  spt --force <AUDIO_PATH>\n\n输出：\n  默认生成 <AUDIO_STEM>.md：Primary STT 生成 provider source；Rust 校验后冻结事实片段受保护的 OpenCC 展示投影，再做 presentation-only 标点清理。第二路 STT 默认检查首段及此后每第 5 个根 TARGET（约每 10 分钟）。\n  --verify-all 对每个根 TARGET 运行第二路 STT，费用更高，但交叉核验覆盖完整。Primary 为空时，第二路结果只记录复核告警，不会回填正文。\n  --raw 生成 <AUDIO_STEM>.raw.md：只运行主 STT，不做第二路交叉检查或 quality 清稿；它仍不承诺底层 ASR 保留每个非词汇声音。\n  通用多模态模型只能切 turn 和比较短声音样本，不能改写事实正文。两种输出可以同时存在，默认均不覆盖已有文件。任何输出都明确标记准确率未测量。\n  冻结正文是 OpenCC 展示投影，不是 provider 原始响应字节。源字形保护仅覆盖事实标签值、成对引号/书名号内容、inline code、URL/email 和显式指定字形；无标签专名仍按普通 OpenCC t2s 转换。\n\n支持格式：\n  aac, aif, aiff, caf, flac, m4a, m4b, mp3, oga, ogg, opus, wav, webm, wma\n\n示例：\n  spt \"/path/to/会议录音.m4a\"\n  spt --verify-all \"/path/to/会议录音.m4a\"\n  spt --raw \"/path/to/会议录音.m4a\"",
        ),
        "ocr" => Some(
            "图片 OCR\n\n用法：\n  spt ocr <IMAGE_PATH>\n  spt ocr --force <IMAGE_PATH>\n\n输出：\n  在图片旁生成 <IMAGE_STEM>.ocr.md。\n\n支持格式：\n  png, jpg, jpeg, webp\n\n示例：\n  spt ocr \"/path/to/扫描件.png\"",
        ),
        "models" => Some(
            "Chat Audio 模型目录\n\n用法：\n  spt models [SEARCH]\n\n作用：\n  查询 OpenRouter 当前声明支持音频输入的通用多模态模型；这些模型用于 turn/说话人 overlay，不是默认正文 STT。\n\n示例：\n  spt models gemini\n  spt --model google/gemini-3.7-flash",
        ),
        "asr-models" | "asr_models" => Some(
            "专用 STT 模型目录\n\n用法：\n  spt asr-models [SEARCH]\n\n作用：\n  查询 OpenRouter /audio/transcriptions 的 audio→transcription 模型。固定 provider 模式仍会在付费前要求目录唯一 endpoint 与 ZDR。\n\n示例：\n  spt asr-models qwen\n  spt --asr-model qwen/qwen3-asr-1.7b",
        ),
        "providers" => Some(
            "Chat Audio Provider 目录\n\n用法：\n  spt providers [MODEL_ID]\n\n作用：\n  列出通用多模态模型的 endpoint tag；省略 MODEL_ID 时使用当前已保存模型。spt --provider any 允许 OpenRouter 自动路由，属于显式隐私降级。\n\n示例：\n  spt providers google/gemini-3.7-flash\n  spt --provider google-vertex/global",
        ),
        "asr-providers" | "asr_providers" => Some(
            "专用 STT Provider 目录\n\n用法：\n  spt asr-providers [MODEL_ID]\n\n作用：\n  列出专用 STT 模型的 endpoint。STT OpenAPI 当前没有 provider.only；固定模式只接受目录中唯一且为 ZDR 的 endpoint。spt --asr-provider any 和 spt --quality-asr-provider any 接受自动路由，均属于显式隐私降级。\n\n示例：\n  spt asr-providers qwen/qwen3-asr-1.7b\n  spt --asr-provider deepinfra",
        ),
        "config" => Some(
            "查看配置\n\n用法：\n  spt config\n\n作用：\n  显示专用 STT、quality STT、多模态 overlay、各自 provider 及资源预算。只显示 OPENROUTER_API_KEY 是否已设置，绝不显示 Key 内容。\n  首次执行非帮助操作时，spt 会原子写入默认配置并创建同目录 .config.lock；已有 v1-v3 配置会在锁内迁移为 v4。\n  配置路径依次取非空 SPT_CONFIG_PATH、非空 XDG_CONFIG_HOME 下的 spt/config.toml、~/.config/spt/config.toml。\n\n持久修改：\n  spt --asr-model <MODEL_ID>\n  spt --quality-asr-model <MODEL_ID>\n  spt --asr-provider <ENDPOINT_TAG|any>    # any 为显式隐私降级\n  spt --quality-asr-provider <ENDPOINT_TAG|any> # any 为显式隐私降级\n  spt --model <MULTIMODAL_MODEL_ID>\n  spt --quality-model <MULTIMODAL_MODEL_ID>\n  spt --provider <ENDPOINT_TAG|any>        # any 为显式隐私降级",
        ),
        "help" => Some(
            "指令帮助\n\n用法：\n  spt\n  spt --help\n  spt help [COMMAND]\n\n可选主题：\n  audio, ocr, asr-models, asr-providers, models, providers, config, help\n\n帮助命令离线执行，不读取或修改 spt 配置。",
        ),
        _ => None,
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct ConfigOverrides<'a> {
    model: Option<&'a str>,
    quality_model: Option<&'a str>,
    asr_model: Option<&'a str>,
    quality_asr_model: Option<&'a str>,
    provider: Option<&'a str>,
    asr_provider: Option<&'a str>,
    quality_asr_provider: Option<&'a str>,
}

fn apply_config_overrides(config: &mut Config, overrides: ConfigOverrides<'_>) -> Result<bool> {
    let mut changed = false;
    if let Some(model) = overrides.model {
        validate_model_id(model)?;
        config.model = model.to_owned();
        config.quality_review_model = model.to_owned();
        changed = true;
    }
    if let Some(quality_model) = overrides.quality_model {
        validate_model_id(quality_model)?;
        config.quality_review_model = quality_model.to_owned();
        changed = true;
    }
    if let Some(asr_model) = overrides.asr_model {
        validate_model_id(asr_model)?;
        config.asr_model = asr_model.to_owned();
        changed = true;
    }
    if let Some(quality_asr_model) = overrides.quality_asr_model {
        validate_model_id(quality_asr_model)?;
        config.quality_asr_model = quality_asr_model.to_owned();
        changed = true;
    }
    if let Some(provider) = overrides.provider {
        let normalized = if provider.eq_ignore_ascii_case(ANY_PROVIDER) {
            ANY_PROVIDER
        } else {
            provider
        };
        validate_provider_id(normalized)?;
        config.provider = normalized.to_owned();
        changed = true;
    }
    if let Some(asr_provider) = overrides.asr_provider {
        let normalized = if asr_provider.eq_ignore_ascii_case(ANY_PROVIDER) {
            ANY_PROVIDER
        } else {
            asr_provider
        };
        validate_provider_id(normalized)?;
        config.asr_provider = normalized.to_owned();
        changed = true;
    }
    if let Some(quality_asr_provider) = overrides.quality_asr_provider {
        let normalized = if quality_asr_provider.eq_ignore_ascii_case(ANY_PROVIDER) {
            ANY_PROVIDER
        } else {
            quality_asr_provider
        };
        validate_provider_id(normalized)?;
        config.quality_asr_provider = normalized.to_owned();
        changed = true;
    }
    Ok(changed)
}

fn reject_force(force: bool) -> Result<()> {
    if force {
        bail!("--force 只适用于音频转写或 OCR 输出");
    }
    Ok(())
}

fn print_config(config: &Config, path: &std::path::Path) {
    let key_status = if env::var("OPENROUTER_API_KEY").is_ok_and(|value| !value.trim().is_empty()) {
        "已设置"
    } else {
        "未设置"
    };
    println!("config={}", path.display());
    println!("schema_version={}", config.schema_version);
    println!("model={}", config.model);
    println!("quality_review_model={}", config.quality_review_model);
    println!("asr_model={}", config.asr_model);
    println!("quality_asr_model={}", config.quality_asr_model);
    println!(
        "effective_quality_review_model={}",
        config.effective_quality_review_model()
    );
    println!("provider={}", config.provider);
    println!("asr_provider={}", config.asr_provider);
    println!("quality_asr_provider={}", config.quality_asr_provider);
    println!("chunk_seconds={}", config.chunk_seconds);
    println!(
        "effective_quality_chunk_seconds={}",
        config.effective_quality_chunk_seconds()
    );
    println!(
        "effective_asr_chunk_seconds={}",
        config.effective_asr_chunk_seconds()
    );
    println!(
        "effective_asr_min_chunk_seconds={}",
        config.effective_asr_min_chunk_seconds()
    );
    println!(
        "effective_quality_min_chunk_seconds={}",
        config.effective_quality_min_chunk_seconds()
    );
    println!("overlap_seconds={}", config.overlap_seconds);
    println!("min_chunk_seconds={}", config.min_chunk_seconds);
    println!("max_output_tokens={}", config.max_output_tokens);
    println!("split_output_tokens={}", config.split_output_tokens);
    println!("parallel_requests={}", config.parallel_requests);
    println!("retries={}", config.retries);
    println!("max_adaptive_depth={}", config.max_adaptive_depth);
    println!("max_http_attempts={}", config.max_http_attempts);
    println!("max_temp_bytes={}", config.max_temp_bytes);
    println!("max_speakers={}", config.max_speakers);
    println!(
        "speaker_reference_seconds={}",
        config.speaker_reference_seconds
    );
    println!(
        "speaker_reference_silence_seconds={}",
        config.speaker_reference_silence_seconds
    );
    println!("speaker_context_chars={}", config.speaker_context_chars);
    println!("max_transcript_bytes={}", config.max_transcript_bytes);
    println!("max_total_turns={}", config.max_total_turns);
    println!("effective_transcription_parallel_requests=1");
    println!("OPENROUTER_API_KEY={key_status}");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn root_help_contains_the_command_guide() {
        let help = Cli::command().render_long_help().to_string();
        assert!(help.contains("常用指令"));
        assert!(help.contains("spt <AUDIO_PATH>"));
        assert!(help.contains("spt --model <MODEL_ID>"));
        assert!(help.contains("spt --asr-model <MODEL_ID>"));
        assert!(help.contains("spt --quality-asr-model <MODEL_ID>"));
        assert!(help.contains("spt --quality-model <MODEL_ID>"));
        assert!(help.contains("google/gemini-3.7-flash"));
        assert!(help.contains("spt --raw <AUDIO_PATH>"));
        assert!(help.contains("spt --verify-all <AUDIO_PATH>"));
        assert!(help.contains("spt help [COMMAND]"));
        assert!(help.contains("any 为隐私降级"));
        assert!(help.contains("OPENROUTER_API_KEY"));
        assert!(help.contains("OpenCC 展示投影"));
        assert!(help.contains("事实标签值、成对引号/书名号内容、inline code、URL/email"));
        assert!(help.contains("无标签专名仍按普通 OpenCC t2s 转换"));
        assert!(help.contains("空 Primary 的第二路结果只记录复核告警，不回填正文"));
        assert!(help.contains("首次执行非帮助操作时会原子写入默认配置"));
        assert!(help.contains("XDG_CONFIG_HOME/spt/config.toml"));
    }

    #[test]
    fn explicit_help_command_accepts_a_topic() {
        let cli = Cli::try_parse_from(["spt", "help", "ocr"]).unwrap();
        assert!(matches!(
            cli.command,
            Some(Commands::Help {
                command: Some(ref command)
            }) if command == "ocr"
        ));
    }

    #[test]
    fn bare_spt_parses_as_the_offline_guide_entrypoint() {
        let cli = Cli::try_parse_from(["spt"]).unwrap();
        assert!(cli.audio_path.is_none());
        assert!(cli.command.is_none());
        assert!(cli.model.is_none());
        assert!(cli.quality_model.is_none());
        assert!(cli.asr_model.is_none());
        assert!(cli.quality_asr_model.is_none());
        assert!(cli.provider.is_none());
        assert!(cli.asr_provider.is_none());
        assert!(cli.quality_asr_provider.is_none());
        assert!(!cli.force);
        assert!(!cli.raw);
        assert!(!cli.verify_all);
    }

    #[test]
    fn raw_flag_selects_an_audio_only_mode() {
        let cli = Cli::try_parse_from(["spt", "--raw", "meeting.m4a"]).unwrap();
        assert!(cli.raw);
        assert!(!cli.verify_all);
        assert_eq!(cli.audio_path, Some(PathBuf::from("meeting.m4a")));
        assert!(cli.command.is_none());
        validate_operation_scope(&cli).unwrap();
    }

    #[test]
    fn raw_flag_is_rejected_without_audio_or_with_a_subcommand() {
        let no_audio = Cli::try_parse_from(["spt", "--raw"]).unwrap();
        assert!(validate_operation_scope(&no_audio).is_err());

        let subcommand = Cli::try_parse_from(["spt", "--raw", "config"]).unwrap();
        assert!(validate_operation_scope(&subcommand).is_err());
    }

    #[test]
    fn verify_all_is_audio_only_and_conflicts_with_raw() {
        let cli = Cli::try_parse_from(["spt", "--verify-all", "meeting.m4a"]).unwrap();
        assert!(cli.verify_all);
        assert!(!cli.raw);
        validate_operation_scope(&cli).unwrap();

        let no_audio = Cli::try_parse_from(["spt", "--verify-all"]).unwrap();
        assert!(validate_operation_scope(&no_audio).is_err());
        assert!(Cli::try_parse_from(["spt", "--raw", "--verify-all", "meeting.m4a"]).is_err());
    }

    #[test]
    fn force_without_an_output_operation_is_rejected_before_config_loading() {
        let force_only = Cli::try_parse_from(["spt", "--force"]).unwrap();
        assert!(validate_operation_scope(&force_only).is_err());
        let config = Cli::try_parse_from(["spt", "--force", "config"]).unwrap();
        assert!(validate_operation_scope(&config).is_err());
        let ocr = Cli::try_parse_from(["spt", "--force", "ocr", "scan.png"]).unwrap();
        assert!(validate_operation_scope(&ocr).is_ok());
    }

    #[test]
    fn providers_positional_model_does_not_mutate_the_global_model_option() {
        let cli = Cli::try_parse_from(["spt", "providers", "google/gemini-3.7-flash"]).unwrap();
        assert!(cli.model.is_none());
        assert!(cli.quality_model.is_none());
        assert!(matches!(
            cli.command,
            Some(Commands::Providers {
                target_model: Some(ref model)
            }) if model == "google/gemini-3.7-flash"
        ));
    }

    #[test]
    fn model_override_controls_both_routes_and_quality_override_can_split_them() {
        let mut config = Config::default();
        apply_config_overrides(
            &mut config,
            ConfigOverrides {
                model: Some("google/gemini-3.5-flash-lite"),
                ..ConfigOverrides::default()
            },
        )
        .unwrap();
        assert_eq!(config.model, "google/gemini-3.5-flash-lite");
        assert_eq!(config.quality_review_model, "google/gemini-3.5-flash-lite");

        apply_config_overrides(
            &mut config,
            ConfigOverrides {
                quality_model: Some("google/gemini-3.7-flash"),
                ..ConfigOverrides::default()
            },
        )
        .unwrap();
        assert_eq!(config.model, "google/gemini-3.5-flash-lite");
        assert_eq!(config.quality_review_model, "google/gemini-3.7-flash");
    }

    #[test]
    fn dedicated_asr_overrides_are_independent_and_persistent() {
        let mut config = Config::default();
        apply_config_overrides(
            &mut config,
            ConfigOverrides {
                asr_model: Some("qwen/qwen3-asr-0.6b"),
                quality_asr_model: Some("fish-audio/transcribe-1"),
                asr_provider: Some("deepinfra"),
                quality_asr_provider: Some("fish-audio"),
                ..ConfigOverrides::default()
            },
        )
        .unwrap();
        assert_eq!(config.asr_model, "qwen/qwen3-asr-0.6b");
        assert_eq!(config.quality_asr_model, "fish-audio/transcribe-1");
        assert_eq!(config.asr_provider, "deepinfra");
        assert_eq!(config.quality_asr_provider, "fish-audio");
    }

    #[test]
    fn every_documented_help_topic_has_a_chinese_guide() {
        for topic in [
            "audio",
            "ocr",
            "models",
            "asr-models",
            "providers",
            "asr-providers",
            "config",
            "help",
        ] {
            let guide = command_topic_guide(topic).unwrap();
            assert!(guide.contains("用法："), "topic={topic}");
            assert!(guide.contains("spt"), "topic={topic}");
        }
        assert!(command_topic_guide("unknown").is_none());

        let audio = command_topic_guide("audio").unwrap();
        assert!(audio.contains("第二路结果只记录复核告警，不会回填正文"));
        assert!(audio.contains("不是 provider 原始响应字节"));
        assert!(audio.contains("无标签专名仍按普通 OpenCC t2s 转换"));

        let config = command_topic_guide("config").unwrap();
        assert!(config.contains("原子写入默认配置并创建同目录 .config.lock"));
        assert!(config.contains("已有 v1-v3 配置会在锁内迁移为 v4"));
        assert!(config.contains("SPT_CONFIG_PATH"));
        assert!(config.contains("XDG_CONFIG_HOME"));

        let providers = command_topic_guide("providers").unwrap();
        assert!(providers.contains("--provider any"));
        assert!(providers.contains("显式隐私降级"));

        let asr_providers = command_topic_guide("asr-providers").unwrap();
        assert!(asr_providers.contains("--asr-provider any"));
        assert!(asr_providers.contains("--quality-asr-provider any"));
        assert!(asr_providers.contains("显式隐私降级"));
    }
}
