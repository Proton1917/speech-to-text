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
    long_about = "spt 使用 OpenRouter 音频多模态模型转写本地录音，并由 Rust SpeakerHarness 维护跨片段说话人编号。\n\n直接给出合法音频路径后，会在源文件同一目录生成同名 Markdown 文字稿。quality 根片段最长 5 分钟，raw 最长 15 分钟；正文过长时从无损母版继续二分。模型和 provider 会持久保存，直到再次修改。",
    arg_required_else_help = false,
    disable_help_subcommand = true,
    after_help = "常用指令：\n  spt <AUDIO_PATH>                         默认高质量稿，输出同名 .md\n  spt --raw <AUDIO_PATH>                   原始逐字稿，输出同名 .raw.md\n  spt --force <AUDIO_PATH>                 完整成功后原子替换已有目标稿件\n  spt --model <MODEL_ID>                   持久设置并统一覆盖两个模型\n  spt --quality-model <MODEL_ID>           单独设置质量复核模型\n  spt --provider <ENDPOINT_TAG|any>        持久设置精确 provider 或自动路由\n  spt models [SEARCH]                      列出支持音频的模型\n  spt providers [MODEL_ID]                 列出模型可用的 provider endpoints\n  spt config                               查看生效配置，不显示 API Key\n  spt ocr <IMAGE_PATH>                     OCR 单张图片，生成 *.ocr.md\n  spt help [COMMAND]                       显示完整介绍或指定子命令帮助\n\n示例：\n  spt \"会议录音.m4a\"\n  spt --raw \"会议录音.m4a\"\n  spt --model google/gemini-3.5-flash-lite\n  spt --quality-model google/gemini-3.7-flash\n  spt --provider google-vertex/global\n\n说明：\n  - 默认 quality 的首个 TARGET 用 Gemini 3.7 Flash 建立可靠起点；后续由基础模型转写，只有可疑片段再升级 3.7。\n  - --raw 始终使用基础模型，保留语气词、卡顿、重复、自我修正和不完整句。\n  - 显式 --model 会统一覆盖基础与质量复核模型；--quality-model 可再单独调整复核模型。\n  - OPENROUTER_API_KEY 只从环境变量读取，不会写入配置。\n  - 中文转写会在写盘前由内置 OpenCC 确定性归一化为 zh-Hans。\n  - 默认不覆盖已有输出；只有 --force 会在完整结果就绪后原子替换。\n  - provider=any 是显式隐私降级；固定 provider 会要求 ZDR 并关闭 fallback。"
)]
struct Cli {
    /// 要转写的本地音频文件
    #[arg(value_name = "AUDIO_PATH")]
    audio_path: Option<PathBuf>,

    /// 保存模型并同时覆盖基础与质量复核路由
    #[arg(long, global = true, value_name = "MODEL_ID")]
    model: Option<String>,

    /// 单独保存质量复核模型；--model 会同时覆盖基础与复核模型
    #[arg(long, global = true, value_name = "MODEL_ID")]
    quality_model: Option<String>,

    /// 保存完整 provider endpoint tag；传入 any 表示允许 OpenRouter 任意路由
    #[arg(long, global = true, value_name = "PROVIDER_ID|any")]
    provider: Option<String>,

    /// 原子替换已经存在的 Markdown 输出
    #[arg(long, global = true)]
    force: bool,

    /// 输出原始逐字稿到 *.raw.md；默认输出清理口语冗余的高质量稿
    #[arg(long)]
    raw: bool,

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
    /// 列出指定模型可用的 OpenRouter provider endpoints
    Providers {
        /// 默认使用当前已保存的模型
        #[arg(id = "provider_target_model", value_name = "MODEL_ID")]
        target_model: Option<String>,
    },
    /// 显示生效配置和配置文件位置，不显示 API Key
    Config,
    /// 显示完整中文指令介绍，或查看指定子命令帮助
    Help {
        /// 可选：ocr、models、providers、config 或 help
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
        && cli.provider.is_none()
        && !cli.force
        && !cli.raw
    {
        print_command_help(None)?;
        return Ok(());
    }
    if let Some(Commands::Help { command }) = cli.command.as_ref() {
        if cli.audio_path.is_some()
            || cli.model.is_some()
            || cli.quality_model.is_some()
            || cli.provider.is_some()
            || cli.force
            || cli.raw
        {
            bail!("spt help 不能与音频路径、配置选项、--force 或 --raw 同时使用");
        }
        print_command_help(command.as_deref())?;
        return Ok(());
    }
    validate_raw_scope(&cli)?;
    let (mut config, config_path, config_existed, config_migrated) = Config::load()?;
    let loaded_config = config.clone();
    let changed = apply_config_overrides(
        &mut config,
        cli.model.as_deref(),
        cli.quality_model.as_deref(),
        cli.provider.as_deref(),
    )?;
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
    }
    if changed || !config_existed || config_migrated {
        let config_lock = ConfigLock::acquire(&config_path)?;
        let (mut latest, latest_path, latest_existed, latest_migrated) = Config::load()?;
        if changed && latest != loaded_config {
            bail!("配置在网络校验期间被其他进程修改，请重新执行本次设置命令");
        }
        apply_config_overrides(
            &mut latest,
            cli.model.as_deref(),
            cli.quality_model.as_deref(),
            cli.provider.as_deref(),
        )?;
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
                "配置已迁移到双模型质量路由 schema v3：{}",
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
        let output = spt::pipeline::transcribe(&audio_path, &config, cli.force, mode).await?;
        println!("{}", output.display());
        return Ok(());
    }

    if changed {
        println!("model={}", config.model);
        println!("quality_review_model={}", config.quality_review_model);
        println!("provider={}", config.provider);
        return Ok(());
    }

    Cli::command().print_help()?;
    println!();
    Ok(())
}

fn validate_raw_scope(cli: &Cli) -> Result<()> {
    if cli.raw && cli.command.is_some() {
        bail!("--raw 只适用于音频转写，不能与子命令同时使用");
    }
    if cli.raw && cli.audio_path.is_none() {
        bail!("--raw 需要同时提供音频路径");
    }
    Ok(())
}

fn print_command_help(command: Option<&str>) -> Result<()> {
    let mut root = Cli::command();
    match command {
        None => root.print_long_help()?,
        Some(name) => {
            let available = "ocr、models、providers、config、help、audio";
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
            "音频转写\n\n用法：\n  spt <AUDIO_PATH>\n  spt --raw <AUDIO_PATH>\n  spt --force <AUDIO_PATH>\n\n输出：\n  默认生成 <AUDIO_STEM>.md 高质量稿：首个 TARGET 使用 Gemini 3.7 Flash，后续基础模型片段只有被 Rust 门禁判为可疑时才升级 3.7。\n  --raw 生成 <AUDIO_STEM>.raw.md 原始逐字稿，只使用基础模型并保留语气词、卡顿、重复、自我修正和不完整句。\n  两种输出可以同时存在，默认均不覆盖已有文件。\n  中文正文在写盘前由内置 OpenCC t2s 归一化为 zh-Hans。\n\n支持格式：\n  aac, aif, aiff, caf, flac, m4a, m4b, mp3, oga, ogg, opus, wav, webm, wma\n\n示例：\n  spt \"/path/to/会议录音.m4a\"\n  spt --raw \"/path/to/会议录音.m4a\"",
        ),
        "ocr" => Some(
            "图片 OCR\n\n用法：\n  spt ocr <IMAGE_PATH>\n  spt ocr --force <IMAGE_PATH>\n\n输出：\n  在图片旁生成 <IMAGE_STEM>.ocr.md。\n\n支持格式：\n  png, jpg, jpeg, webp\n\n示例：\n  spt ocr \"/path/to/扫描件.png\"",
        ),
        "models" => Some(
            "模型目录\n\n用法：\n  spt models [SEARCH]\n\n作用：\n  查询 OpenRouter 当前声明支持音频输入的模型；SEARCH 可按模型代号或名称过滤。\n\n示例：\n  spt models\n  spt models gemini\n  spt --model google/gemini-3.5-flash-lite",
        ),
        "providers" => Some(
            "Provider 目录\n\n用法：\n  spt providers [MODEL_ID]\n\n作用：\n  列出指定模型的完整 endpoint tag；省略 MODEL_ID 时使用当前已保存模型。\n\n示例：\n  spt providers\n  spt providers google/gemini-3.5-flash-lite\n  spt --provider google-vertex/global\n  spt --provider any",
        ),
        "config" => Some(
            "查看配置\n\n用法：\n  spt config\n\n作用：\n  显示配置文件位置、基础模型、质量复核模型、provider、切分和安全预算等生效值。\n  只显示 OPENROUTER_API_KEY 是否已设置，绝不显示 Key 内容。\n\n持久修改：\n  spt --model <MODEL_ID>\n  spt --quality-model <MODEL_ID>\n  spt --provider <ENDPOINT_TAG|any>",
        ),
        "help" => Some(
            "指令帮助\n\n用法：\n  spt\n  spt --help\n  spt help [COMMAND]\n\n可选主题：\n  audio, ocr, models, providers, config, help\n\n帮助命令离线执行，不读取或修改 spt 配置。",
        ),
        _ => None,
    }
}

fn apply_config_overrides(
    config: &mut Config,
    model: Option<&str>,
    quality_model: Option<&str>,
    provider: Option<&str>,
) -> Result<bool> {
    let mut changed = false;
    if let Some(model) = model {
        validate_model_id(model)?;
        config.model = model.to_owned();
        config.quality_review_model = model.to_owned();
        changed = true;
    }
    if let Some(quality_model) = quality_model {
        validate_model_id(quality_model)?;
        config.quality_review_model = quality_model.to_owned();
        changed = true;
    }
    if let Some(provider) = provider {
        let normalized = if provider.eq_ignore_ascii_case(ANY_PROVIDER) {
            ANY_PROVIDER
        } else {
            provider
        };
        validate_provider_id(normalized)?;
        config.provider = normalized.to_owned();
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
    println!(
        "effective_quality_review_model={}",
        config.effective_quality_review_model()
    );
    println!("provider={}", config.provider);
    println!("chunk_seconds={}", config.chunk_seconds);
    println!(
        "effective_quality_chunk_seconds={}",
        config.effective_quality_chunk_seconds()
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
        assert!(help.contains("spt --quality-model <MODEL_ID>"));
        assert!(help.contains("spt --raw <AUDIO_PATH>"));
        assert!(help.contains("spt help [COMMAND]"));
        assert!(help.contains("OPENROUTER_API_KEY"));
        assert!(help.contains("zh-Hans"));
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
        assert!(cli.provider.is_none());
        assert!(!cli.force);
        assert!(!cli.raw);
    }

    #[test]
    fn raw_flag_selects_an_audio_only_mode() {
        let cli = Cli::try_parse_from(["spt", "--raw", "meeting.m4a"]).unwrap();
        assert!(cli.raw);
        assert_eq!(cli.audio_path, Some(PathBuf::from("meeting.m4a")));
        assert!(cli.command.is_none());
        validate_raw_scope(&cli).unwrap();
    }

    #[test]
    fn raw_flag_is_rejected_without_audio_or_with_a_subcommand() {
        let no_audio = Cli::try_parse_from(["spt", "--raw"]).unwrap();
        assert!(validate_raw_scope(&no_audio).is_err());

        let subcommand = Cli::try_parse_from(["spt", "--raw", "config"]).unwrap();
        assert!(validate_raw_scope(&subcommand).is_err());
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
            Some("google/gemini-3.5-flash-lite"),
            None,
            None,
        )
        .unwrap();
        assert_eq!(config.model, "google/gemini-3.5-flash-lite");
        assert_eq!(config.quality_review_model, "google/gemini-3.5-flash-lite");

        apply_config_overrides(&mut config, None, Some("google/gemini-3.7-flash"), None).unwrap();
        assert_eq!(config.model, "google/gemini-3.5-flash-lite");
        assert_eq!(config.quality_review_model, "google/gemini-3.7-flash");
    }

    #[test]
    fn every_documented_help_topic_has_a_chinese_guide() {
        for topic in ["audio", "ocr", "models", "providers", "config", "help"] {
            let guide = command_topic_guide(topic).unwrap();
            assert!(guide.contains("用法："), "topic={topic}");
            assert!(guide.contains("spt"), "topic={topic}");
        }
        assert!(command_topic_guide("unknown").is_none());
    }
}
