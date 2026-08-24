# spt Project Guide

## Scope

This repository contains the Rust backend and CLI for spt. The frontend has not started. Keep changes focused on validated local media, OpenRouter dedicated STT and Chat Audio orchestration, frozen transcript text, speaker overlays, OCR, configuration, Markdown output, benchmarks, and packaging.

## User-facing contract

- spt AUDIO_PATH writes AUDIO_STEM.md beside the source.
- Default quality uses Primary STT on every root target, samples an independent Quality STT on the first and every fifth root target, and applies only presentation punctuation and adjacent-space cleanup. Agreement is not ground-truth verification.
- spt --verify-all AUDIO_PATH runs the independent Quality STT on every root target for the current task; it is not persisted and cannot be combined with --raw.
- spt --raw AUDIO_PATH writes AUDIO_STEM.raw.md, skips the second STT and quality cleanup, and does not claim provider-level verbatim disfluency preservation.
- Validated Primary provider source is retained only while the current TARGET is processed for cross-ASR evidence and display projection. A fact-span-protected OpenCC display projection is the transcript authority rendered to Markdown; it is not byte-identical provider output. Chat Audio may segment turns and label voices but cannot change canonical fact characters.
- spt --asr-model and --asr-provider persist the Primary STT route. asr-provider=any is an explicit privacy downgrade.
- spt --quality-asr-model and --quality-asr-provider persist the quality cross-check route. quality-asr-provider=any is an explicit privacy downgrade.
- spt --model sets both raw and quality overlay models; --quality-model may override only the quality overlay.
- spt --provider persists the Chat Audio endpoint. any is an explicit privacy downgrade.
- spt ocr IMAGE_PATH writes IMAGE_STEM.ocr.md.
- Bare spt, --help, and help COMMAND are offline and do not mutate configuration.
- The first non-help operation atomically writes the default v4 configuration and creates its sibling .config.lock when configuration is absent. Existing v1-v3 files migrate under that lock.
- Existing output is never replaced without --force; replacement occurs only after the complete result is ready.
- OPENROUTER_API_KEY is read only from the process environment and must never be persisted or logged.

## Current defaults

Verified against the live OpenRouter catalog and a controlled synthetic fixture on 2026-08-24:

~~~text
Primary STT model       = qwen/qwen3-asr-1.7b
Primary STT endpoint    = deepinfra
Quality STT model       = fish-audio/transcribe-1
Quality STT endpoint    = fish-audio
Raw/quality overlay     = google/gemini-3.7-flash
Overlay endpoint        = google-vertex/global
STT API                 = https://openrouter.ai/api/v1/audio/transcriptions
Chat API                = https://openrouter.ai/api/v1/chat/completions
~~~

The STT OpenAPI does not expose Chat provider.only. Fixed STT mode is therefore accepted only when the live catalog contains exactly one endpoint, its tag exactly matches configuration, and the same model/tag is present in the live ZDR catalog. Describe this as catalog-unique ZDR preflight, never as request-level pinning.

## Architecture

- src/main.rs: Clap interface, schema-v4 route settings, catalog commands, process exit.
- src/asr.rs: Primary/Quality STT validation, pre-OpenCC source comparison, frozen display-text restoration and UNKNOWN fallback.
- src/chinese.rs: embedded OpenCC t2s with fact-span and Japanese-kana preservation.
- src/cleanup.rs: quality-only presentation punctuation cleanup; spoken disfluencies are signaled but preserved, with lexical projection audit and whole-turn fallback.
- src/config.rs: defaults, v1-v3 migration, strict v4 validation, route ID validation, private atomic TOML storage.
- src/media.rs: no-follow fixed input snapshots, demuxer/protocol validation, canonical FLAC duration checks, exact 120-second targets, FFmpeg activity ranges and bounded speaker packets.
- src/openrouter.rs: bounded HTTPS, secret headers, retry classification, dedicated STT schemas, Chat schemas, live catalogs and route checks.
- src/speaker.rs: strict turn schemas, per-turn T-to-S/NEW/UNKNOWN mapping, global S allocation, short candidate and long reference rules.
- src/pipeline.rs: Primary STT, sampled or verify-all Quality STT, quality-only host cleanup, frozen turn overlay, SpeakerHarness, OCR and shared request budgets.
- src/output.rs: honest front matter, atomic Markdown output and bounded ~/.spt output-lock shards.
- src/transcript.rs: quality/raw output names and editing contracts.
- benchmarks/: offline CER/fact/speaker metrics, paid-run guard, private fixtures and versioned baseline snapshots.

FFmpeg is an intentional subprocess boundary. Pass paths as Command arguments; never interpolate them into a shell command.

## Safety invariants

- Open user media once without following Unix symlinks or Windows reparse points, verify the handle is a regular file, copy it with a hard byte limit into a private task workspace, and use only that fixed copy afterward.
- Accept only allowlisted, non-empty regular audio files with a real audio stream. Restrict FFprobe/FFmpeg demuxers by extension and protocol before input open; reject symlinks, fake extensions, concat/playlists, nested resources and real video streams.
- OCR is explicit and accepts only validated png, jpg, jpeg or webp images.
- Derive every target and reference from the same lossless canonical source; never recursively encode MP3.
- Dedicated STT targets are continuous, non-overlapping and at most 120 seconds because upstream STT processing has a shorter timeout boundary than Chat Audio.
- Validate Primary STT, Quality STT and the selected Chat overlay against live catalogs after local preparation and immediately before the first paid request.
- Fixed Chat mode uses provider.only, allow_fallbacks=false, require_parameters=true, data_collection=deny and zdr=true.
- STT fixed mode must not pretend to send unsupported provider routing fields. Reject multiple-endpoint models until the public STT contract supports a real pin.
- A fixed STT provider value is a catalog-unique expected endpoint, not an actual endpoint claim. Only a provider explicitly reported by the STT response may be recorded as reported/actual.
- All clients share one authenticated HTTPS transport, semaphore and task-level HTTP-attempt ledger.
- content_filter, SAFETY and policy errors are never retried. Error text reports the actual number of attempts.
- Bound request media, response bytes, temporary disk, HTTP attempts, transcript bytes, turns, speakers, references and identity packet media before accumulation.
- Validate Primary source, retain it only for the current TARGET, and create a fact-span-protected OpenCC display projection before freezing. Protected source glyphs are limited to recognized fact-label values, paired quotation/book-title contents, inline code, URLs/emails and explicit character designations. Unlabelled proper nouns follow normal OpenCC t2s conversion. Reject NUL, replacement characters and pathological repetition.
- An empty Primary transcript skips turn overlay, Stage B and cleanup. Quality runs its independent STT only when that root target is sampled or --verify-all is active; the second result emits an advisory and never backfills text. Raw stops the TARGET after Primary.
- Quality comparison operates on pre-OpenCC provider source. It may ignore presentation whitespace and sentence punctuation but must preserve digits, numeric separators, signs, letters, names, negation and conditions.
- Quality disagreement or verifier failure retains Primary text and emits a review advisory; it never lets the verifier overwrite text.
- Quality cleanup runs only after frozen turns exist. It may replace only presentation punctuation and remove only adjacent ordinary spaces; every spoken lexical character, including fillers and repetitions, remains immutable and possible disfluencies are signals only. It must revert the entire turn on any failed lexical audit. Raw never calls it.
- Turn overlay receives Primary display text only as JSON-escaped untrusted data. Rust must validate canonical equality and restore the fact-span-protected Primary display slices before accepting turns; do not describe them as original provider bytes.
- Overlay failure, filtering, invalid structure or semantic mutation degrades to one full-duration UNKNOWN turn while preserving Primary text.
- Stage B receives only short historical references, boundary context and host-owned T candidates. It cannot emit text.
- Stage B mappings must cover every candidate T exactly once. Rust owns NEW-to-S allocation and commits speaker state only after complete validation.
- Candidate turns may be 1-10 seconds. A NEW group may create S only when it contains a clean, non-overlapping reference-eligible sample of at least 2 seconds. Short-only NEW groups stay UNKNOWN.
- Unsampled, overlapping or unreliable turns stay UNKNOWN. They never inherit a Stage-A local label.
- References are task-local source ranges, not persistent voiceprints. Output must say not_verified and not_measured, never all_labels_assigned.
- Hold an output transaction and cross-process lock before paid work. Map stable parent-directory file identity with FNV-1a into 4096 persistent ~/.spt/output-locks shards; all platforms intentionally serialize spt outputs in one directory to close case, Unicode, firmlink, short-name and nocase-mount aliases. Every Unix process always acquires the default ~/.spt shard even when it also uses an absolute custom SPT_STATE_DIR; Windows rejects custom roots. The OS releases locks on process exit, and an existing Unix custom state root must not be chmodded by spt.
- Resolve an existing configuration parent to its canonical directory before validation. For a not-yet-complete parent, reject symlink/reparse-point creation boundaries. Open the terminal configuration file and .config.lock without following symlinks/reparse points.
- Usage and cost metadata cover only received responses with provider-reported usage. They are not HTTP-attempt totals or a final billing statement.
- Model output must never control commands, paths, model selection or provider selection.

## Validation and evidence

After code changes run:

~~~bash
cargo fmt --check
cargo clippy --locked --all-targets -- -D warnings
cargo test --locked --all-targets
cargo build --locked --release
./benchmarks/scripts/test.sh
~~~

Default tests must not make paid requests. A live test must be tiny, deliberate, use an isolated SPT_CONFIG_PATH, never print the Key, and record metrics rather than relying on prose quality.

The tracked synthetic baseline is benchmarks/baselines/v0.5.0-synthetic-zh-aba.tsv. It contains one run per configuration and cannot be used as a real-audio production claim. The manual CAiRE/ASCEND generator creates an ignored 14.8-second public human A-B-A fixture with pinned revision and CC-BY-SA attribution; it is an artificial splice, not a meeting benchmark. Private recordings belong only in ignored benchmarks/private fixtures after authorization.

## Project log

- 2026-08-23: v0.1 created safe media validation, canonical audio, bounded OpenRouter calls, OCR and atomic output.
- 2026-08-23: v0.2 separated exact-target transcript generation from short SpeakerHarness identity packets.
- 2026-08-24: v0.2.1-v0.2.2 added the Chinese CLI guide and deterministic zh-Hans normalization.
- 2026-08-24: v0.3.0 added independent quality and raw output paths.
- 2026-08-24: v0.4.0 added the Gemini Lite/3.7 surface-gated cascade, schema v3 and Homebrew source/bottle delivery.
- 2026-08-24: v0.5.0 replaced Gemini text authority with dedicated STT, added schema v4, sampled/verify-all cross-ASR evidence, fact-protected OpenCC display text, presentation-only cleanup, per-turn SpeakerHarness mapping, fixed no-follow input snapshots, canonical-duration checks, honest route/cost provenance, OCR rejected-response accounting and bounded ~/.spt lock shards. Release gates pass with 212 library tests, 11 CLI tests, 12 benchmark tests, strict Clippy, release build and Windows cross-build/link. One-run synthetic and pinned ASCEND human-splice A-B-A snapshots both recorded CER 0 and S1-S2-S1 on their narrow fixtures; neither is a meeting/DER claim, and real-meeting acceptance remains open.
