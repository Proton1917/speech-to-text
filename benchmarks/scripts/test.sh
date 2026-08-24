#!/usr/bin/env bash
set -euo pipefail
unset OPENROUTER_API_KEY

BENCHMARK_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
TEMPORARY_DIRECTORY=$(mktemp -d "${TMPDIR:-/tmp}/spt-bench-test.XXXXXX")
KEY_ISOLATION_CASE_ID="spt-key-isolation-$$"
cleanup() {
  rm -rf -- "$TEMPORARY_DIRECTORY"
  rm -rf -- "$BENCHMARK_ROOT/results/$KEY_ISOLATION_CASE_ID"
}
trap cleanup EXIT

command -v cargo >/dev/null 2>&1 || {
  printf '缺少 cargo，无法测试离线评测器。\n' >&2
  exit 127
}
if command -v rustfmt >/dev/null 2>&1; then
  cargo fmt --manifest-path "$BENCHMARK_ROOT/Cargo.toml" -- --check
fi

CARGO_TARGET_DIR="$TEMPORARY_DIRECTORY/cargo-target" RUSTFLAGS='-D warnings' cargo test \
  --locked \
  --manifest-path "$BENCHMARK_ROOT/Cargo.toml"
CARGO_TARGET_DIR="$TEMPORARY_DIRECTORY/cargo-target" RUSTFLAGS='-D warnings' cargo build \
  --locked \
  --manifest-path "$BENCHMARK_ROOT/Cargo.toml"
BENCHMARK_BINARY="$TEMPORARY_DIRECTORY/cargo-target/debug/spt-bench"

printf '%s\n' \
  '---' \
  'transcript_mode: "raw"' \
  'accounted_model_responses: 2' \
  'reported_accounted_cost_usd: 0.001700600' \
  '---' \
  '' \
  '# 合成测试 原始逐字稿' \
  '' \
  '## 00:00:00–00:00:22' \
  '' \
  'S2：预算是四十二万元。' \
  '' \
  'S1：测试环境周五之前就绪。' \
  '' \
  'S2：项目代号是阿尔法七号。如果预算没有批准，就不要上线。' \
  >"$TEMPORARY_DIRECTORY/transcript.md"

"$BENCHMARK_BINARY" evaluate \
  --case "$BENCHMARK_ROOT/fixtures/synthetic-zh-aba" \
  --transcript "$TEMPORARY_DIRECTORY/transcript.md" \
  --report "$TEMPORARY_DIRECTORY/report.tsv"

grep -F $'cer\t0.000000' "$TEMPORARY_DIRECTORY/report.tsv" >/dev/null
grep -F $'number_exact_recall\t1.000000' "$TEMPORARY_DIRECTORY/report.tsv" >/dev/null
grep -F $'proper_name_exact_recall\t1.000000' "$TEMPORARY_DIRECTORY/report.tsv" >/dev/null
grep -F $'speaker_permutation_invariant_turn_accuracy\t1.000000' \
  "$TEMPORARY_DIRECTORY/report.tsv" >/dev/null

NFC_CASE_DIRECTORY="$TEMPORARY_DIRECTORY/nfc-case"
mkdir -p "$NFC_CASE_DIRECTORY"
printf 'case_id\tnfc-equivalence\nlanguage\tzh-CN\nunicode_normalization\tNFC\naudio_path\tunused.wav\n' \
  >"$NFC_CASE_DIRECTORY/case.tsv"
printf 'A\tが가\n' >"$NFC_CASE_DIRECTORY/turns.tsv"
printf '%s\n' '# no exact terms' >"$NFC_CASE_DIRECTORY/terms.tsv"
printf '%s\n' 'S1：が가' >"$TEMPORARY_DIRECTORY/nfc-hypothesis.md"
"$BENCHMARK_BINARY" evaluate \
  --case "$NFC_CASE_DIRECTORY" \
  --transcript "$TEMPORARY_DIRECTORY/nfc-hypothesis.md" \
  --report "$TEMPORARY_DIRECTORY/nfc-report.tsv"
grep -F $'cer\t0.000000' "$TEMPORARY_DIRECTORY/nfc-report.tsv" >/dev/null

printf 'A\tが\n' >"$NFC_CASE_DIRECTORY/turns.tsv"
if "$BENCHMARK_BINARY" evaluate \
  --case "$NFC_CASE_DIRECTORY" \
  --transcript "$TEMPORARY_DIRECTORY/nfc-hypothesis.md" \
  >"$TEMPORARY_DIRECTORY/non-nfc-reference.stdout" \
  2>"$TEMPORARY_DIRECTORY/non-nfc-reference.stderr"; then
  printf '%s\n' '非 NFC 参考文字不应通过 case 契约。' >&2
  exit 1
fi
grep -F '与 case.tsv 声明的 NFC 不一致' \
  "$TEMPORARY_DIRECTORY/non-nfc-reference.stderr" >/dev/null

printf 'case_id\tlegacy-normalization\nlanguage\tzh-CN\nunicode_normalization\tNFC-no-combining-marks\naudio_path\tunused.wav\n' \
  >"$NFC_CASE_DIRECTORY/case.tsv"
printf 'A\t测试\n' >"$NFC_CASE_DIRECTORY/turns.tsv"
if "$BENCHMARK_BINARY" evaluate \
  --case "$NFC_CASE_DIRECTORY" \
  >"$TEMPORARY_DIRECTORY/legacy-normalization.stdout" \
  2>"$TEMPORARY_DIRECTORY/legacy-normalization.stderr"; then
  printf '%s\n' '非标准 Unicode normalization 声明不应通过 case 契约。' >&2
  exit 1
fi
grep -F 'unicode_normalization 必须是 NFC' \
  "$TEMPORARY_DIRECTORY/legacy-normalization.stderr" >/dev/null

assert_key_capture_precedes_helpers() {
  local script=$1
  local capture_line unexport_line unset_line helper_line
  capture_line=$(awk '/^SPT_BENCH_CAPTURED_API_KEY=/{ print NR; exit }' "$script")
  unexport_line=$(awk '/^export -n SPT_BENCH_CAPTURED_API_KEY$/{ print NR; exit }' "$script")
  unset_line=$(awk '/^unset OPENROUTER_API_KEY$/{ print NR; exit }' "$script")
  helper_line=$(awk '/^BENCHMARK_ROOT=/{ print NR; exit }' "$script")
  [[ -n "$capture_line" && -n "$unexport_line" && -n "$unset_line" && -n "$helper_line" ]]
  ((capture_line < unexport_line && unexport_line < unset_line && unset_line < helper_line))
}

assert_key_capture_precedes_helpers "$BENCHMARK_ROOT/scripts/run-spt.sh"
assert_key_capture_precedes_helpers "$BENCHMARK_ROOT/scripts/run-manifest-case.sh"
for helper_script in "$BENCHMARK_ROOT"/scripts/*.sh; do
  helper_unset_line=$(awk '/^unset OPENROUTER_API_KEY$/{ print NR; exit }' "$helper_script")
  helper_setup_line=$(awk '/^BENCHMARK_ROOT=/{ print NR; exit }' "$helper_script")
  [[ -n "$helper_unset_line" && -n "$helper_setup_line" ]]
  ((helper_unset_line < helper_setup_line))
done
if grep -Eq '^[[:space:]]*export[[:space:]]+OPENROUTER_API_KEY' \
  "$BENCHMARK_ROOT/scripts/run-spt.sh" "$BENCHMARK_ROOT/scripts/run-manifest-case.sh"; then
  printf '%s\n' 'runner 不得向中间 shell 或辅助进程 export OPENROUTER_API_KEY。' >&2
  exit 1
fi
[[ $(grep -Ec '^[[:space:]]+OPENROUTER_API_KEY=\$benchmark_openrouter_api_key[[:space:]]+\\$' \
  "$BENCHMARK_ROOT/scripts/run-spt.sh") -eq 1 ]]
grep -F 'source "$BENCHMARK_ROOT/scripts/run-spt.sh"' \
  "$BENCHMARK_ROOT/scripts/run-manifest-case.sh" >/dev/null

REAL_AWK=$(command -v awk)
REAL_DIRNAME=$(command -v dirname)
GUARD_DIRECTORY="$TEMPORARY_DIRECTORY/guard-bin"
mkdir -p "$GUARD_DIRECTORY"
write_environment_guard() {
  local path=$1
  local real_command=$2
  {
    printf '%s\n' '#!/usr/bin/env bash' 'set -euo pipefail'
    printf '%s\n' \
      'for private_name in OPENROUTER_API_KEY SPT_BENCH_CAPTURED_API_KEY benchmark_openrouter_api_key BENCHMARK_OPENROUTER_API_KEY MANIFEST_OPENROUTER_API_KEY; do' \
      '  if declare -p "$private_name" >/dev/null 2>&1; then' \
      '    printf "密钥变量泄漏给辅助进程 %s：%s\\n" "$0" "$private_name" >&2' \
      '    exit 97' \
      '  fi' \
      'done'
    printf 'exec %q "$@"\n' "$real_command"
  } >"$path"
  chmod 700 "$path"
}
write_environment_guard "$GUARD_DIRECTORY/awk" "$REAL_AWK"
write_environment_guard "$GUARD_DIRECTORY/dirname" "$REAL_DIRNAME"

KEY_CASE_DIRECTORY="$TEMPORARY_DIRECTORY/key-case"
mkdir -p "$KEY_CASE_DIRECTORY"
printf 'case_id\t%s\nlanguage\tzh-CN\nunicode_normalization\tNFC\naudio_path\taudio.wav\n' \
  "$KEY_ISOLATION_CASE_ID" >"$KEY_CASE_DIRECTORY/case.tsv"
printf 'A\t密钥隔离测试。\n' >"$KEY_CASE_DIRECTORY/turns.tsv"
printf '%s\n' '# no exact terms' >"$KEY_CASE_DIRECTORY/terms.tsv"
: >"$KEY_CASE_DIRECTORY/audio.wav"
printf '%s\t%s\t%s\n' \
  "$KEY_ISOLATION_CASE_ID" "$KEY_CASE_DIRECTORY" "$KEY_CASE_DIRECTORY/audio.wav" \
  >"$TEMPORARY_DIRECTORY/private-fixtures.tsv"

FAKE_SPT="$TEMPORARY_DIRECTORY/fake-spt"
FAKE_SPT_EVENTS="$TEMPORARY_DIRECTORY/fake-spt-events.tsv"
KEY_SENTINEL='spt-test-key-never-write-this-value'
{
  printf '%s\n' '#!/usr/bin/env bash' 'set -euo pipefail'
  printf 'expected_key=%q\nevent_log=%q\n' "$KEY_SENTINEL" "$FAKE_SPT_EVENTS"
  printf '%s\n' \
    'for private_name in SPT_BENCH_CAPTURED_API_KEY benchmark_openrouter_api_key BENCHMARK_OPENROUTER_API_KEY MANIFEST_OPENROUTER_API_KEY; do' \
    '  if declare -p "$private_name" >/dev/null 2>&1; then' \
    '    printf "内部密钥变量泄漏给 spt 子进程：%s\\n" "$private_name" >&2' \
    '    exit 98' \
    '  fi' \
    'done' \
    'if [[ ${1:-} == --version ]]; then' \
    '  if declare -p OPENROUTER_API_KEY >/dev/null 2>&1; then' \
    '    printf "%s\\n" "spt --version 不应收到 OPENROUTER_API_KEY" >&2' \
    '    exit 99' \
    '  fi' \
    '  printf "%s\\n" "version_clean" >>"$event_log"' \
    '  printf "%s\\n" "spt 0.5.0-test"' \
    '  exit 0' \
    'fi' \
    'if [[ ${1:-} == config ]]; then' \
    '  if declare -p OPENROUTER_API_KEY >/dev/null 2>&1; then' \
    '    printf "%s\\n" "spt config 不应收到 OPENROUTER_API_KEY" >&2' \
    '    exit 99' \
    '  fi' \
    '  printf "%s\\n" "config_clean" >>"$event_log"' \
    '  printf "%s\\n" "asr_model=qwen/qwen3-asr-1.7b" "quality_asr_model=fish-audio/transcribe-1" "asr_provider=deepinfra" "quality_asr_provider=fish-audio" "effective_quality_review_model=google/gemini-3.7-flash" "provider=google-vertex/global"' \
    '  exit 0' \
    'fi' \
    '[[ ${OPENROUTER_API_KEY:-} == "$expected_key" ]] || {' \
    '  printf "%s\\n" "实际 spt 转写子进程没有收到唯一注入的 Key" >&2' \
    '  exit 100' \
    '}' \
    'printf "%s\\n" "transcription_key_only" >>"$event_log"' \
    'audio_path=${!#}' \
    'output_path=${audio_path%.*}.md' \
    '[[ ${1:-} == --raw ]] && output_path=${audio_path%.*}.raw.md' \
    'printf "%s\\n" "---" "transcript_mode: raw" "---" "" "S1：密钥隔离测试。" >"$output_path"'
} >"$FAKE_SPT"
chmod 700 "$FAKE_SPT"

OPENROUTER_API_KEY=$KEY_SENTINEL \
SPT_BENCH_ALLOW_PAID=1 \
SPT_BENCH_SPT=$FAKE_SPT \
PATH="$GUARD_DIRECTORY:$PATH" \
CARGO_NET_OFFLINE=true \
  "$BENCHMARK_ROOT/scripts/run-manifest-case.sh" \
  "$KEY_ISOLATION_CASE_ID" raw "$TEMPORARY_DIRECTORY/private-fixtures.tsv" \
  >"$TEMPORARY_DIRECTORY/key-runner.stdout" \
  2>"$TEMPORARY_DIRECTORY/key-runner.stderr"

OPENROUTER_API_KEY=$KEY_SENTINEL \
SPT_BENCH_ALLOW_PAID=1 \
PATH="$GUARD_DIRECTORY:$PATH" \
CARGO_NET_OFFLINE=true \
  "$BENCHMARK_ROOT/scripts/run-spt.sh" \
  --case "$KEY_CASE_DIRECTORY" \
  --audio "$KEY_CASE_DIRECTORY/audio.wav" \
  --mode raw \
  --spt "$FAKE_SPT" \
  >"$TEMPORARY_DIRECTORY/direct-key-runner.stdout" \
  2>"$TEMPORARY_DIRECTORY/direct-key-runner.stderr"
[[ $(grep -Fxc 'version_clean' "$FAKE_SPT_EVENTS") -eq 2 ]]
[[ $(grep -Fxc 'config_clean' "$FAKE_SPT_EVENTS") -eq 2 ]]
[[ $(grep -Fxc 'transcription_key_only' "$FAKE_SPT_EVENTS") -eq 2 ]]
[[ $(wc -l <"$FAKE_SPT_EVENTS") -eq 6 ]]
if grep -RIlF -- "$KEY_SENTINEL" \
  "$BENCHMARK_ROOT/results/$KEY_ISOLATION_CASE_ID" \
  "$TEMPORARY_DIRECTORY/key-runner.stdout" \
  "$TEMPORARY_DIRECTORY/key-runner.stderr" \
  "$TEMPORARY_DIRECTORY/direct-key-runner.stdout" \
  "$TEMPORARY_DIRECTORY/direct-key-runner.stderr" >/dev/null; then
  printf '%s\n' 'runner 将 API Key 打印或写入了结果目录。' >&2
  exit 1
fi

ASCEND_GENERATOR="$BENCHMARK_ROOT/scripts/generate-ascend-zh-aba.sh"
ASCEND_UNSET_LINE=$(awk '/^unset OPENROUTER_API_KEY$/{ print NR; exit }' "$ASCEND_GENERATOR")
ASCEND_HELPER_LINE=$(awk '/^BENCHMARK_ROOT=/{ print NR; exit }' "$ASCEND_GENERATOR")
[[ -n "$ASCEND_UNSET_LINE" && -n "$ASCEND_HELPER_LINE" ]]
((ASCEND_UNSET_LINE < ASCEND_HELPER_LINE))
if grep -Eq '^[[:space:]]*(export[[:space:]]+)?OPENROUTER_API_KEY=' "$ASCEND_GENERATOR"; then
  printf '%s\n' 'ASCEND 生成器不得保存或重新导出 OPENROUTER_API_KEY。' >&2
  exit 1
fi
if grep -Eq '^[[:space:]]*(curl|jq|ffmpeg|ffprobe)([[:space:]]|$)' "$ASCEND_GENERATOR"; then
  printf '%s\n' 'ASCEND 生成器的网络和媒体子进程必须显式移除 OPENROUTER_API_KEY。' >&2
  exit 1
fi
grep -F "DATASET_REVISION='737e9800ae31be9932ba8464c80366559bd28424'" \
  "$ASCEND_GENERATOR" >/dev/null
grep -F "DATASET_LICENSE='cc-by-sa-4.0'" "$ASCEND_GENERATOR" >/dev/null
grep -F 'https://creativecommons.org/licenses/by-sa/4.0/' "$ASCEND_GENERATOR" >/dev/null
grep -F 'ROW_INDEXES=(400 904 401)' "$ASCEND_GENERATOR" >/dev/null
grep -F 'ROW_IDS=(00400 00904 00401)' "$ASCEND_GENERATOR" >/dev/null
grep -F 'ROW_SPEAKERS=(3 17 3)' "$ASCEND_GENERATOR" >/dev/null
grep -F 'ROW_DURATIONS=(5.72 4.82 2.66)' "$ASCEND_GENERATOR" >/dev/null
grep -F "PAUSE_SECONDS='0.8'" "$ASCEND_GENERATOR" >/dev/null
grep -F 'https://datasets-server.huggingface.co/rows?dataset=CAiRE%2FASCEND' \
  "$ASCEND_GENERATOR" >/dev/null
grep -F '就你要申请去交换你并不需要去说哦我要去哪一个department交换是直接选school' \
  "$ASCEND_GENERATOR" >/dev/null
grep -F '就是你是在大学的时候哪一个阶段才萌生' "$ASCEND_GENERATOR" >/dev/null
grep -F '我要去这个university做交换' "$ASCEND_GENERATOR" >/dev/null
grep -Fx '/public-fixtures/' "$BENCHMARK_ROOT/.gitignore" >/dev/null

for script in "$BENCHMARK_ROOT"/scripts/*.sh; do
  bash -n "$script"
done

printf 'benchmark tests passed\n'
