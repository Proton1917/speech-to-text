# spt Project Guide

## Scope

This repository contains the Rust backend/CLI for `spt`, an OpenRouter-powered speech-to-text project. The frontend has not been started. Keep backend changes focused on local media validation, OpenRouter request orchestration, transcript generation, OCR, configuration, and CLI behavior.

## User-facing contract

- `spt <AUDIO_PATH>` writes `<AUDIO_STEM>.md` beside the source file.
- `spt --model <OPENROUTER_MODEL_ID>` persistently changes the model.
- `spt --provider <ENDPOINT_TAG>` persists one exact endpoint verified against the selected model's live catalog.
- `spt --provider any` authorizes OpenRouter automatic routing; the API request must omit the `provider` field.
- `spt ocr <IMAGE_PATH>` writes `<IMAGE_STEM>.ocr.md`.
- Existing output is never replaced without `--force`, and replacement happens only after the complete result is ready.
- The API key is read only from `OPENROUTER_API_KEY`; never persist or log it.

## Current external defaults

Verified against the live OpenRouter catalog on 2026-08-23:

```text
model    = google/gemini-3.5-flash-lite
provider = google-vertex/global
endpoint = https://openrouter.ai/api/v1/chat/completions
```

The default model accepts audio and image input. Audio is sent as raw Base64 through `input_audio`; OCR uses a JPEG data URL. Re-check catalog and endpoint behavior before changing these defaults because OpenRouter routing can drift.

## Architecture

- `src/main.rs`: Clap interface, persistent settings, catalog commands, process exit.
- `src/config.rs`: schema/defaults, route ID validation, private atomic TOML storage.
- `src/media.rs`: filename/allowlist checks, FFprobe validation, lossless canonical FLAC generation, and one-generation MP3 ranges.
- `src/openrouter.rs`: HTTPS requests, secret header handling, retry classification, response parsing, catalog queries.
- `src/pipeline.rs`: bounded concurrency, adaptive recursive audio splitting, timeline validation, OCR orchestration.
- `src/output.rs`: Markdown metadata/rendering and private atomic output.

FFmpeg is intentionally a subprocess boundary. Never interpolate paths into a shell command; continue passing every argument directly to `Command`/`tokio::process::Command`.

## Safety invariants

- Default transcription accepts only allowlisted, non-empty regular files whose contents contain an audio stream. Reject symlinks, fake extensions, and media with a real video stream.
- OCR remains an explicit subcommand and accepts only validated single-image formats; do not silently treat arbitrary files or PDFs as images.
- Bound request media size, HTTP attempts, adaptive depth, active roots, FFmpeg work and temporary disk before they can accumulate.
- Fixed provider mode requires an exact live `endpoints[].tag`, uses `provider.only`, disables fallback, and does not silently switch after an error.
- Require a complete non-empty model response. Length/context overflow or clear looping triggers source-audio bisection; reaching the minimum duration without a reliable result fails the whole job.
- Model output is untrusted text. Never use it for commands, paths, tool calls, or provider/model selection.
- Derive all initial and adaptive ranges directly from the same lossless canonical source; never recursively re-encode MP3.
- Hold an output transaction and cross-process target lock before paid work. Write the final document only after every part succeeds; preserve an existing result on any processing failure.

## Validation

After code changes, run:

```bash
cargo fmt --check
cargo clippy --locked --all-targets -- -D warnings
cargo test --locked --all-targets
cargo build --locked --release
```

Do not make paid OpenRouter calls in default tests. A live smoke test must be deliberate, tiny, and use the configured environment key without printing it.

## Project log

- 2026-08-23: Created backend v0.1 from an empty directory. Added the Rust CLI, locked private model/provider configuration, strict audio/image validation, lossless canonical audio, five-minute bounded transcription, Token/length/loop-driven recursive bisection from the lossless source, crash-safe Markdown provenance, OCR, OpenRouter discovery/error routing, resource budgets, cancellation, documentation, tests, and release workflow. Frontend work remains out of scope for this milestone.
