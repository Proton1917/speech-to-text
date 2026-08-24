#!/usr/bin/env bash
set -euo pipefail
umask 077
SPT_BENCH_CAPTURED_API_KEY=${SPT_BENCH_CAPTURED_API_KEY-${OPENROUTER_API_KEY:-}}
export -n SPT_BENCH_CAPTURED_API_KEY
unset OPENROUTER_API_KEY

spt_bench_run() {
local benchmark_openrouter_api_key=$SPT_BENCH_CAPTURED_API_KEY
export -n benchmark_openrouter_api_key
unset SPT_BENCH_CAPTURED_API_KEY OPENROUTER_API_KEY

BENCHMARK_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
CASE_DIRECTORY=''
AUDIO_PATH=''
MODE=quality
SPT_BINARY=${SPT_BENCH_SPT:-}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --case)
      [[ $# -ge 2 ]] || { printf '%s\n' '--case 缺少路径。' >&2; exit 64; }
      CASE_DIRECTORY=$2
      shift 2
      ;;
    --audio)
      [[ $# -ge 2 ]] || { printf '%s\n' '--audio 缺少路径。' >&2; exit 64; }
      AUDIO_PATH=$2
      shift 2
      ;;
    --mode)
      [[ $# -ge 2 ]] || { printf '%s\n' '--mode 缺少 quality/raw/verify-all。' >&2; exit 64; }
      MODE=$2
      shift 2
      ;;
    --spt)
      [[ $# -ge 2 ]] || { printf '%s\n' '--spt 缺少可执行文件。' >&2; exit 64; }
      SPT_BINARY=$2
      shift 2
      ;;
    -h|--help)
      printf '用法：%s --case CASE_DIR [--audio AUDIO] [--mode quality|raw|verify-all] [--spt PATH]\n' "$0"
      printf '必须显式设置 SPT_BENCH_ALLOW_PAID=1；每次运行都会把音频上传到当前 spt provider。\n'
      exit 0
      ;;
    *)
      printf '未知参数：%s\n' "$1" >&2
      exit 64
      ;;
  esac
done

[[ -n "$CASE_DIRECTORY" ]] || { printf '%s\n' '缺少 --case CASE_DIR。' >&2; exit 64; }
[[ "$MODE" == quality || "$MODE" == raw || "$MODE" == verify-all ]] || {
  printf '无效模式：%s（只支持 quality/raw/verify-all）。\n' "$MODE" >&2
  exit 64
}
[[ "${SPT_BENCH_ALLOW_PAID:-}" == 1 ]] || {
  printf '%s\n' '拒绝运行：这一步会上传音频并可能产生 OpenRouter 费用。' >&2
  printf '%s\n' '确认后显式设置：SPT_BENCH_ALLOW_PAID=1' >&2
  exit 77
}
[[ -n "$benchmark_openrouter_api_key" ]] || {
  printf '%s\n' '缺少 OPENROUTER_API_KEY。' >&2
  exit 78
}

CASE_DIRECTORY=$(cd -- "$CASE_DIRECTORY" && pwd)
[[ -f "$CASE_DIRECTORY/case.tsv" && -f "$CASE_DIRECTORY/turns.tsv" && -f "$CASE_DIRECTORY/terms.tsv" ]] || {
  printf 'case 目录缺少 case.tsv、turns.tsv 或 terms.tsv：%s\n' "$CASE_DIRECTORY" >&2
  exit 66
}

case_value() {
  local key=$1
  awk -F '\t' -v wanted="$key" '$1 == wanted { print $2; exit }' "$CASE_DIRECTORY/case.tsv"
}

CASE_ID=$(case_value case_id)
[[ -n "$CASE_ID" && "$CASE_ID" != . && "$CASE_ID" != .. && "$CASE_ID" =~ ^[A-Za-z0-9._-]+$ ]] || {
  printf 'case_id 为空或包含不安全字符：%s\n' "$CASE_ID" >&2
  exit 65
}
if [[ -z "$AUDIO_PATH" ]]; then
  AUDIO_PATH=$(case_value audio_path)
  [[ -n "$AUDIO_PATH" ]] || { printf '%s\n' 'case.tsv 缺少 audio_path。' >&2; exit 65; }
fi
[[ "$AUDIO_PATH" == /* ]] || AUDIO_PATH="$CASE_DIRECTORY/$AUDIO_PATH"
[[ -f "$AUDIO_PATH" ]] || { printf '找不到音频：%s\n' "$AUDIO_PATH" >&2; exit 66; }

if [[ -z "$SPT_BINARY" ]]; then
  SPT_BINARY="$BENCHMARK_ROOT/../target/release/spt"
fi
if [[ "$SPT_BINARY" == */* ]]; then
  [[ -x "$SPT_BINARY" ]] || { printf 'spt 不可执行：%s\n' "$SPT_BINARY" >&2; exit 126; }
else
  SPT_BINARY=$(command -v "$SPT_BINARY") || { printf '%s\n' 'PATH 中找不到 spt。' >&2; exit 127; }
fi

RUN_ID=$(date -u '+%Y%m%dT%H%M%SZ')-$$
RESULT_DIRECTORY="$BENCHMARK_ROOT/results/$CASE_ID/$RUN_ID"
mkdir -p "$RESULT_DIRECTORY"
RUN_CONFIG_PATH="$RESULT_DIRECTORY/config.toml"
RUN_STATE_DIRECTORY="$RESULT_DIRECTORY/state"
if [[ -n "${SPT_BENCH_CONFIG:-}" ]]; then
  [[ -f "$SPT_BENCH_CONFIG" ]] || { printf '找不到 SPT_BENCH_CONFIG：%s\n' "$SPT_BENCH_CONFIG" >&2; exit 66; }
  cp -- "$SPT_BENCH_CONFIG" "$RUN_CONFIG_PATH"
fi

AUDIO_BASENAME=$(basename -- "$AUDIO_PATH")
AUDIO_EXTENSION=${AUDIO_BASENAME##*.}
[[ "$AUDIO_EXTENSION" =~ ^[A-Za-z0-9]+$ ]] || {
  printf '音频扩展名不可用于隔离运行：%s\n' "$AUDIO_EXTENSION" >&2
  exit 65
}
RUN_AUDIO="$RESULT_DIRECTORY/input.$AUDIO_EXTENSION"
cp -- "$AUDIO_PATH" "$RUN_AUDIO"
EVALUATOR=$(env -u OPENROUTER_API_KEY "$BENCHMARK_ROOT/scripts/build.sh")
env -u OPENROUTER_API_KEY SPT_CONFIG_PATH="$RUN_CONFIG_PATH" SPT_STATE_DIR="$RUN_STATE_DIRECTORY" \
  "$SPT_BINARY" --version >"$RESULT_DIRECTORY/spt-version.txt"
env -u OPENROUTER_API_KEY SPT_CONFIG_PATH="$RUN_CONFIG_PATH" SPT_STATE_DIR="$RUN_STATE_DIRECTORY" \
  "$SPT_BINARY" config \
  >"$RESULT_DIRECTORY/spt-config.txt" 2>"$RESULT_DIRECTORY/spt-config.stderr.log"
CONFIG_MODEL=$(awk -F '=' '$1 == "asr_model" { print substr($0, index($0, "=") + 1); exit }' \
  "$RESULT_DIRECTORY/spt-config.txt")
CONFIG_QUALITY_MODEL=$(awk -F '=' '$1 == "quality_asr_model" { print substr($0, index($0, "=") + 1); exit }' \
  "$RESULT_DIRECTORY/spt-config.txt")
CONFIG_PROVIDER=$(awk -F '=' '$1 == "asr_provider" { print substr($0, index($0, "=") + 1); exit }' \
  "$RESULT_DIRECTORY/spt-config.txt")
CONFIG_QUALITY_PROVIDER=$(awk -F '=' '$1 == "quality_asr_provider" { print substr($0, index($0, "=") + 1); exit }' \
  "$RESULT_DIRECTORY/spt-config.txt")
CONFIG_OVERLAY_MODEL=$(awk -F '=' '$1 == "effective_quality_review_model" { print substr($0, index($0, "=") + 1); exit }' \
  "$RESULT_DIRECTORY/spt-config.txt")
CONFIG_OVERLAY_PROVIDER=$(awk -F '=' '$1 == "provider" { print substr($0, index($0, "=") + 1); exit }' \
  "$RESULT_DIRECTORY/spt-config.txt")
SPT_VERSION=$(head -n 1 "$RESULT_DIRECTORY/spt-version.txt")
GIT_COMMIT=$(git -C "$BENCHMARK_ROOT/.." rev-parse HEAD 2>/dev/null || printf 'not-a-git-worktree')
if git -C "$BENCHMARK_ROOT/.." diff --quiet --ignore-submodules HEAD -- 2>/dev/null && \
   [[ -z "$(git -C "$BENCHMARK_ROOT/.." status --porcelain --untracked-files=normal 2>/dev/null)" ]]; then
  GIT_DIRTY=false
else
  GIT_DIRTY=true
fi
[[ -n "$CONFIG_MODEL" && -n "$CONFIG_QUALITY_MODEL" && -n "$CONFIG_PROVIDER" && \
   -n "$CONFIG_QUALITY_PROVIDER" && -n "$CONFIG_OVERLAY_MODEL" && -n "$CONFIG_OVERLAY_PROVIDER" ]] || {
  printf '%s\n' '无法从隔离 spt config 读取完整路由。' >&2
  exit 65
}

now_ms() {
  if command -v perl >/dev/null 2>&1; then
    perl -MTime::HiRes=time -e 'printf "%.0f\n", time() * 1000'
  else
    printf '%s000\n' "$(date '+%s')"
  fi
}

START_MS=$(now_ms)
SPT_TRANSCRIBE_ARGS=()
[[ "$MODE" == raw ]] && SPT_TRANSCRIBE_ARGS+=(--raw)
[[ "$MODE" == verify-all ]] && SPT_TRANSCRIBE_ARGS+=(--verify-all)
set +e
SPT_CONFIG_PATH="$RUN_CONFIG_PATH" SPT_STATE_DIR="$RUN_STATE_DIRECTORY" \
  OPENROUTER_API_KEY=$benchmark_openrouter_api_key \
  "$SPT_BINARY" "${SPT_TRANSCRIBE_ARGS[@]}" "$RUN_AUDIO" \
  >"$RESULT_DIRECTORY/stdout.log" 2>"$RESULT_DIRECTORY/stderr.log"
EXIT_CODE=$?
set -e
unset benchmark_openrouter_api_key
END_MS=$(now_ms)
ELAPSED_MS=$((END_MS - START_MS))

if [[ $EXIT_CODE -eq 0 ]]; then
  STATUS=success
  FAILURE_KIND=none
else
  STATUS=failure
  if grep -Eiq 'content[_ -]?filter|SAFETY' "$RESULT_DIRECTORY/stderr.log"; then
    FAILURE_KIND=content_filter
  elif grep -Eiq 'timeout|超时' "$RESULT_DIRECTORY/stderr.log"; then
    FAILURE_KIND=timeout
  else
    FAILURE_KIND=process_error
  fi
fi

printf 'run_id\t%s\nstatus\t%s\nexit_code\t%s\nelapsed_ms\t%s\nhttp_attempts\tNA\nfailure_kind\t%s\nmode\t%s\nmodel\t%s\nquality_model\t%s\nprovider\t%s\nquality_provider\t%s\noverlay_model\t%s\noverlay_provider\t%s\nspt_version\t%s\ngit_commit\t%s\ngit_dirty\t%s\n' \
  "$RUN_ID" "$STATUS" "$EXIT_CODE" "$ELAPSED_MS" "$FAILURE_KIND" \
  "$MODE" "$CONFIG_MODEL" "$CONFIG_QUALITY_MODEL" "$CONFIG_PROVIDER" \
  "$CONFIG_QUALITY_PROVIDER" "$CONFIG_OVERLAY_MODEL" "$CONFIG_OVERLAY_PROVIDER" \
  "$SPT_VERSION" "$GIT_COMMIT" "$GIT_DIRTY" \
  >"$RESULT_DIRECTORY/run.tsv"

TRANSCRIPT_PATH="$RESULT_DIRECTORY/input.md"
[[ "$MODE" == raw ]] && TRANSCRIPT_PATH="$RESULT_DIRECTORY/input.raw.md"
EVALUATE_ARGUMENTS=(
  evaluate
  --case "$CASE_DIRECTORY"
  --run "$RESULT_DIRECTORY/run.tsv"
  --report "$RESULT_DIRECTORY/report.tsv"
)
if [[ -f "$TRANSCRIPT_PATH" ]]; then
  EVALUATE_ARGUMENTS+=(--transcript "$TRANSCRIPT_PATH")
fi
env -u OPENROUTER_API_KEY "$EVALUATOR" "${EVALUATE_ARGUMENTS[@]}"

printf '运行目录：%s\n' "$RESULT_DIRECTORY"
printf '状态：%s；耗时：%s ms；报告：%s\n' \
  "$STATUS" "$ELAPSED_MS" "$RESULT_DIRECTORY/report.tsv"
exit "$EXIT_CODE"
}

if [[ ${BASH_SOURCE[0]} == "$0" ]]; then
  spt_bench_run "$@"
fi
