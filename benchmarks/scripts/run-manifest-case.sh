#!/usr/bin/env bash
set -euo pipefail
SPT_BENCH_CAPTURED_API_KEY=${OPENROUTER_API_KEY:-}
export -n SPT_BENCH_CAPTURED_API_KEY
unset OPENROUTER_API_KEY

BENCHMARK_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
MANIFEST="$BENCHMARK_ROOT/private-fixtures.tsv"
MODE=quality

if [[ $# -lt 1 || $# -gt 3 ]]; then
  printf '用法：%s CASE_ID [quality|raw] [MANIFEST_TSV]\n' "$0" >&2
  exit 64
fi
CASE_ID=$1
[[ $# -ge 2 ]] && MODE=$2
[[ $# -ge 3 ]] && MANIFEST=$3
[[ "$CASE_ID" != . && "$CASE_ID" != .. && "$CASE_ID" =~ ^[A-Za-z0-9._-]+$ ]] || {
  printf '无效 case id：%s\n' "$CASE_ID" >&2
  exit 64
}
[[ -f "$MANIFEST" ]] || {
  printf '找不到私有 manifest：%s\n先复制 private-fixtures.example.tsv。\n' "$MANIFEST" >&2
  exit 66
}

MATCH_COUNT=$(awk -F '\t' -v wanted="$CASE_ID" '$1 == wanted { count++ } END { print count + 0 }' "$MANIFEST")
[[ "$MATCH_COUNT" -eq 1 ]] || {
  printf 'manifest 中 case_id=%s 应恰好出现一次，实际为 %s 次。\n' "$CASE_ID" "$MATCH_COUNT" >&2
  exit 65
}
RECORD=$(awk -F '\t' -v wanted="$CASE_ID" '$1 == wanted { print $1 "\t" $2 "\t" $3; exit }' "$MANIFEST")
IFS=$'\t' read -r _ CASE_DIRECTORY AUDIO_PATH <<<"$RECORD"
[[ -n "$CASE_DIRECTORY" && -n "$AUDIO_PATH" ]] || {
  printf 'manifest 的 %s 记录缺少 case_dir 或 audio_path。\n' "$CASE_ID" >&2
  exit 65
}

MANIFEST_DIRECTORY=$(cd -- "$(dirname -- "$MANIFEST")" && pwd)
[[ "$CASE_DIRECTORY" == /* ]] || CASE_DIRECTORY="$MANIFEST_DIRECTORY/$CASE_DIRECTORY"
[[ "$AUDIO_PATH" == /* ]] || AUDIO_PATH="$MANIFEST_DIRECTORY/$AUDIO_PATH"

source "$BENCHMARK_ROOT/scripts/run-spt.sh"
spt_bench_run \
  --case "$CASE_DIRECTORY" \
  --audio "$AUDIO_PATH" \
  --mode "$MODE"
