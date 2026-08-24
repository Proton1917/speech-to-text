use std::env;
use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::{Result, bail};
use clap::{CommandFactory, Parser, Subcommand};

use spt::config::{ANY_PROVIDER, Config, ConfigLock, validate_model_id, validate_provider_id};
use spt::openrouter::OpenRouterClient;

#[derive(Debug, Parser)]
#[command(
    name = "spt",
    version,
    about = "安全、可持久配置的 OpenRouter 语音转文字 CLI",
    long_about = "给出合法的本地音频路径，在同一目录生成 Markdown 文字稿。长音频会先按时长切分；模型输出达到 Token 边界时会继续递归二分。",
    arg_required_else_help = true
)]
struct Cli {
    /// 要转写的本地音频文件
    #[arg(value_name = "AUDIO_PATH")]
    audio_path: Option<PathBuf>,

    /// 保存并使用 OpenRouter 模型代号，例如 google/gemini-3.5-flash-lite
    #[arg(long, global = true, value_name = "MODEL_ID")]
    model: Option<String>,

    /// 保存完整 provider endpoint tag；传入 any 表示允许 OpenRouter 任意路由
    #[arg(long, global = true, value_name = "PROVIDER_ID|any")]
    provider: Option<String>,

    /// 原子替换已经存在的 Markdown 输出
    #[arg(long, global = true)]
    force: bool,

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
        #[arg(value_name = "MODEL_ID")]
        model: Option<String>,
    },
    /// 显示生效配置和配置文件位置，不显示 API Key
    Config,
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
    let (mut config, config_path, config_existed, config_migrated) = Config::load()?;
    let changed =
        apply_config_overrides(&mut config, cli.model.as_deref(), cli.provider.as_deref())?;
    config.validate()?;
    if changed {
        OpenRouterClient::from_environment(config.clone(), false)?
            .validate_selection("audio")
            .await?;
    }
    if changed || !config_existed || config_migrated {
        let config_lock = ConfigLock::acquire(&config_path)?;
        let (mut latest, latest_path, latest_existed, latest_migrated) = Config::load()?;
        apply_config_overrides(&mut latest, cli.model.as_deref(), cli.provider.as_deref())?;
        latest.validate()?;
        if changed && (latest.model != config.model || latest.provider != config.provider) {
            bail!("配置在网络校验期间被其他进程修改，请重新执行本次设置命令");
        }
        if changed || !latest_existed || latest_migrated {
            latest.save(&latest_path)?;
        }
        config = latest;
        drop(config_lock);
        if changed {
            eprintln!("配置已保存：{}", config_path.display());
        } else if config_migrated {
            eprintln!(
                "配置已迁移到 SpeakerHarness schema v2：{}",
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
            Commands::Providers { model } => {
                reject_force(cli.force)?;
                let model = model.unwrap_or_else(|| config.model.clone());
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
        }
        return Ok(());
    }

    if let Some(audio_path) = cli.audio_path {
        let output = spt::pipeline::transcribe(&audio_path, &config, cli.force).await?;
        println!("{}", output.display());
        return Ok(());
    }

    if changed {
        println!("model={}", config.model);
        println!("provider={}", config.provider);
        return Ok(());
    }

    Cli::command().print_help()?;
    println!();
    Ok(())
}

fn apply_config_overrides(
    config: &mut Config,
    model: Option<&str>,
    provider: Option<&str>,
) -> Result<bool> {
    let mut changed = false;
    if let Some(model) = model {
        validate_model_id(model)?;
        config.model = model.to_owned();
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
    println!("provider={}", config.provider);
    println!("chunk_seconds={}", config.chunk_seconds);
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
