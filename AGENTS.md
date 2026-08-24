# spt Project Guide

## Scope

This repository contains the Rust backend/CLI for `spt`, an OpenRouter-powered speech-to-text project. The frontend has not been started. Keep backend changes focused on local media validation, OpenRouter request orchestration, transcript generation, OCR, configuration, and CLI behavior.

## User-facing contract

- `spt <AUDIO_PATH>` defaults to a faithful readability-cleaned quality transcript and writes `<AUDIO_STEM>.md` beside the source file.
- `spt --raw <AUDIO_PATH>` preserves fillers, stutters, repetitions, self-corrections and incomplete speech, writing `<AUDIO_STEM>.raw.md` so both modes can coexist.
- Default quality uses a cost-bounded cascade: the first TARGET is at most five minutes and uses Gemini 3.7 Flash as a reliable bootstrap; later TARGETs use Gemini 3.5 Flash Lite and upgrade only when flagged by the Rust quality gate. Raw never invokes the review route.
- `spt --model <OPENROUTER_MODEL_ID>` persistently overrides both the base and effective quality-review model, preserving the single-model override contract.
- `spt --quality-model <OPENROUTER_MODEL_ID>` persistently changes only the quality bootstrap/review model.
- `spt --provider <ENDPOINT_TAG>` persists one exact endpoint verified against the selected model's live catalog.
- `spt --provider any` authorizes OpenRouter automatic routing; the API request must omit the `provider` field.
- `spt ocr <IMAGE_PATH>` writes `<IMAGE_STEM>.ocr.md`.
- `spt`, `spt --help`, and `spt help [COMMAND]` expose the built-in Chinese command guide without network or configuration mutation.
- Existing output is never replaced without `--force`, and replacement happens only after the complete result is ready.
- The API key is read only from `OPENROUTER_API_KEY`; never persist or log it.

## Current external defaults

Verified against the live OpenRouter catalog on 2026-08-24:

```text
base model           = google/gemini-3.5-flash-lite
quality review model = google/gemini-3.7-flash
provider             = google-vertex/global
endpoint             = https://openrouter.ai/api/v1/chat/completions
```

Both models accept audio plus strict structured output on the fixed ZDR endpoint. Audio is sent as raw Base64 through `input_audio`; OCR uses the base model with a JPEG data URL. Re-check catalog and endpoint behavior before changing these defaults because OpenRouter routing can drift.

## Architecture

- `src/main.rs`: Clap interface, persistent settings, catalog commands, process exit.
- `src/chinese.rs`: embedded OpenCC t2s normalization for validated Stage A Chinese text; preserves turns containing Japanese kana.
- `src/config.rs`: schema/defaults, route ID validation, private atomic TOML storage.
- `src/media.rs`: filename/allowlist checks, canonical FLAC, exact TARGET audio, FFmpeg activity ranges, and short one-file identity packets.
- `src/openrouter.rs`: HTTPS requests, secret header handling, retry classification, response parsing, catalog queries.
- `src/quality.rs`: deterministic Chinese typography normalization and stable signals that decide whether a quality TARGET needs the stronger review route.
- `src/speaker.rs`: strict local-turn and identity-mapping schemas, global S-ID allocation, bounded tail state, and clean reference ranges.
- `src/transcript.rs`: explicit quality/raw mode contract, output suffixes, editing provenance, and titles.
- `src/pipeline.rs`: base/review/identity orchestration, acoustic and quality gates, left-before-right adaptive splitting, and OCR.
- `src/output.rs`: Markdown metadata/rendering and private atomic output.

FFmpeg is intentionally a subprocess boundary. Never interpolate paths into a shell command; continue passing every argument directly to `Command`/`tokio::process::Command`.

## Safety invariants

- Default transcription accepts only allowlisted, non-empty regular files whose contents contain an audio stream. Reject symlinks, fake extensions, and media with a real video stream.
- OCR remains an explicit subcommand and accepts only validated single-image formats; do not silently treat arbitrary files or PDFs as images.
- Speaker-aware requests are always sequential. Raw TARGETs are at most 15 minutes; quality root TARGETs are at most five minutes. Stage A hears only exact TARGET; Stage B may use at most 30 seconds of boundary context but cannot emit text.
- Bound request media size, HTTP attempts, adaptive depth, speaker count, reference duration, FFmpeg work and temporary disk before they can accumulate.
- Fixed provider mode requires an exact live ZDR `endpoints[].tag`, uses `provider.only`, `data_collection=deny`, `zdr=true`, disables fallback, and does not silently switch after an error. `any` must continue omitting the provider field and is an explicit privacy downgrade.
- Base and quality-review routes must share the same HTTPS transport, semaphore and task-level HTTP-attempt budget. Never construct an independent review budget.
- Stage A is the only transcript authority. Length/context overflow, looping, or invalid structure triggers source-audio bisection. Every FFmpeg energy-coverage mismatch retries once and then records an advisory; raw energy must never become a hard speech gate because it is not VAD.
- Quality mode may remove only non-semantic spoken disfluency and repair clearly supported punctuation/wording; it must preserve every fact, number, name, position, negation, condition, uncertainty and task without summary, reordering or cross-turn merging. Raw mode is verbatim apart from necessary punctuation.
- Host typography normalization must preserve URL/email/inline-code tokens, tabs, non-ASCII structural whitespace and ambiguous Han-Han spaces. Systemic Han spacing is a review signal, never a license to merge names or list fields.
- Quality bootstrap/review must happen before SpeakerHarness identity alignment. Apart from the bounded first bootstrap, only suspicious quality TARGETs use the stronger model; raw must never call it. Persist initial trigger codes separately from residual advisory codes, mark affected time ranges, and account for accepted base/review/alignment costs once.
- A structurally valid label assignment is not verified identity. Use `all_labels_assigned`, `UNKNOWN`, and the existing best-effort guarantee without claiming speaker correctness.
- Normalize every validated Stage A turn to Simplified Chinese with embedded OpenCC t2s before duplicate checks, SpeakerHarness state, previous-tail context, or Markdown rendering. OCR must preserve source script.
- Model output is untrusted text. Never use it for commands, paths, tool calls, or provider/model selection.
- Derive all initial and adaptive ranges directly from the same lossless canonical source; never recursively re-encode MP3.
- Never let model-provided temporary labels become final IDs directly. Rust owns NEW-to-S allocation; uncertain voices remain UNKNOWN.
- Stage B receives only short historical S references, boundary context and local L candidates. It may only return L-to-S/NEW/UNKNOWN mappings; failure degrades labels to UNKNOWN and must never discard or rewrite accepted Stage A text.
- References are task-local source ranges, not persistent voiceprints. The output must call alignment best-effort, never verified identity.
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

- 2026-08-23: Created backend v0.1 from an empty directory with safe media validation, canonical audio, bounded OpenRouter transcription, OCR, atomic output and resource budgets.
- 2026-08-23: Upgraded to v0.2 two-stage SpeakerHarness: 15-minute exact TARGET transcription, FFmpeg activity coverage, short reference/candidate identity packets, host-owned global IDs, sequential state transfer and v1-to-v2 migration. This replaced the initial single composite transcript packet after a real cross-boundary E2E exposed deterministic omitted speech. Frontend remains out of scope.
- 2026-08-24: Released v0.2.1 with a built-in Chinese command guide for bare `spt`, `--help`, `help`, per-command help topics, examples, output behavior, persistent model/provider settings, and security notes.
- 2026-08-24: Released v0.2.2 with deterministic embedded OpenCC t2s normalization before transcript state/output, preventing per-chunk Traditional Chinese drift without another model call.
- 2026-08-24: Released v0.3.0 with quality mode as the default `<stem>.md` contract and explicit `--raw` verbatim output at `<stem>.raw.md`, with independent locks, atomic replacement and provenance.
- 2026-08-24: Added v0.4.0 adaptive quality review after a real transcript showed that a prompt-only quality label was insufficient. A bounded first 3.7 TARGET establishes the job; later Lite TARGETs are upgraded only on stable Rust signals. A 90-second E2E corrected key terms such as 脱敏/客户 while costing less than the Lite-only baseline; deterministic reviewed-text cleanup now removes the remaining known connector/pronoun stutters without changing general reduplication. Added schema v3 dual-model pins, shared budgets, trigger-versus-residual provenance, URL-safe typography repair, honest advisory headings, and fixed `spt providers MODEL_ID` plus config compare-and-swap bugs.
- 2026-08-24: Published immutable `v0.4.0` and the official `Proton1917/homebrew-tap` Formula. The Formula builds the locked Rust source, depends on FFmpeg at runtime, deletes `OPENROUTER_API_KEY` in its offline test, and passed Homebrew style, strict online audit, source installation, linkage and `brew test` on Apple Silicon.
- 2026-08-24: Added an `arm64_tahoe` relocatable Homebrew bottle for v0.4.0 after a clean-machine install exposed seven unnecessary source-build dependencies. Normal Apple Silicon Tahoe installs now pour the 3 MB compressed bottle and do not install Rust, LLVM, Python, Z3, libgit2 or pkgconf; FFmpeg remains an explicit runtime dependency. The Formula caveat only explains the required runtime `OPENROUTER_API_KEY`, its non-persistence boundary, and the `spt --help` entrypoint.
