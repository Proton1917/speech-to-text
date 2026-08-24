#!/usr/bin/env bash
set -euo pipefail
unset OPENROUTER_API_KEY

BENCHMARK_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
CASE_DIRECTORY="$BENCHMARK_ROOT/fixtures/synthetic-zh-aba"
OUTPUT="$CASE_DIRECTORY/audio.m4a"
FORCE=0
if [[ "${1:-}" == "--force" ]]; then
  FORCE=1
elif [[ $# -ne 0 ]]; then
  printf '用法：%s [--force]\n' "$0" >&2
  exit 64
fi

[[ "$(uname -s)" == "Darwin" ]] || {
  printf '此合成脚本需要 macOS 的 say。可在其他系统手工准备同一 turns.tsv 对应的音频。\n' >&2
  exit 69
}
for command_name in say ffmpeg ffprobe awk sed; do
  command -v "$command_name" >/dev/null 2>&1 || {
    printf '缺少命令：%s\n' "$command_name" >&2
    exit 127
  }
done
if [[ -e "$OUTPUT" && $FORCE -ne 1 ]]; then
  printf '音频已存在：%s；如需重新生成请显式使用 --force。\n' "$OUTPUT" >&2
  exit 73
fi

AVAILABLE_VOICES=()
while IFS= read -r voice_line; do
  if [[ "$voice_line" =~ ^(.+)[[:space:]]+zh_(CN|TW|HK)[[:space:]] ]]; then
    voice_name="${BASH_REMATCH[1]}"
    voice_name=$(printf '%s' "$voice_name" | sed 's/[[:space:]]*$//')
    AVAILABLE_VOICES+=("$voice_name")
  fi
done < <(say -v '?' 2>/dev/null)

voice_available() {
  local requested=$1
  local candidate
  for candidate in "${AVAILABLE_VOICES[@]}"; do
    [[ "$candidate" == "$requested" ]] && return 0
  done
  return 1
}

select_fallback_voice() {
  local excluded=${1:-}
  local candidate
  for candidate in "${AVAILABLE_VOICES[@]}"; do
    if [[ "$candidate" != "$excluded" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

if [[ -n "${SPT_BENCH_VOICE_A:-}" ]]; then
  VOICE_A=$SPT_BENCH_VOICE_A
  voice_available "$VOICE_A" || {
    printf '找不到 SPT_BENCH_VOICE_A=%s。\n' "$VOICE_A" >&2
    exit 69
  }
elif voice_available Tingting; then
  VOICE_A=Tingting
else
  VOICE_A=$(select_fallback_voice) || {
    printf '至少需要两个已安装的中文 say 声音。\n' >&2
    exit 69
  }
fi

if [[ -n "${SPT_BENCH_VOICE_B:-}" ]]; then
  VOICE_B=$SPT_BENCH_VOICE_B
  voice_available "$VOICE_B" || {
    printf '找不到 SPT_BENCH_VOICE_B=%s。\n' "$VOICE_B" >&2
    exit 69
  }
elif voice_available Meijia && [[ "$VOICE_A" != Meijia ]]; then
  VOICE_B=Meijia
else
  VOICE_B=$(select_fallback_voice "$VOICE_A") || {
    printf '至少需要两个不同的已安装中文 say 声音。\n' >&2
    exit 69
  }
fi
[[ "$VOICE_A" != "$VOICE_B" ]] || {
  printf 'VOICE_A 与 VOICE_B 必须不同。\n' >&2
  exit 64
}

SAY_RATE=${SPT_BENCH_SAY_RATE:-155}
PAUSE_SECONDS=${SPT_BENCH_PAUSE_SECONDS:-0.75}
[[ "$SAY_RATE" =~ ^[0-9]+$ ]] || {
  printf 'SPT_BENCH_SAY_RATE 必须是正整数。\n' >&2
  exit 64
}
[[ "$PAUSE_SECONDS" =~ ^[0-9]+([.][0-9]+)?$ ]] || {
  printf 'SPT_BENCH_PAUSE_SECONDS 必须是非负秒数。\n' >&2
  exit 64
}

TEMPORARY_DIRECTORY=$(mktemp -d "${TMPDIR:-/tmp}/spt-synthetic-aba.XXXXXX")
trap 'rm -rf -- "$TEMPORARY_DIRECTORY"' EXIT

say -v "$VOICE_A" -r "$SAY_RATE" -o "$TEMPORARY_DIRECTORY/a1.aiff" \
  '预算是四十二万元。'
say -v "$VOICE_B" -r "$SAY_RATE" -o "$TEMPORARY_DIRECTORY/b1.aiff" \
  '测试环境周五之前就绪。'
say -v "$VOICE_A" -r "$SAY_RATE" -o "$TEMPORARY_DIRECTORY/a2.aiff" \
  '项目代号是阿尔法七号。如果预算没有批准，就不要上线。'

ffmpeg -hide_banner -loglevel error -y \
  -i "$TEMPORARY_DIRECTORY/a1.aiff" \
  -f lavfi -i "anullsrc=r=24000:cl=mono:d=$PAUSE_SECONDS" \
  -i "$TEMPORARY_DIRECTORY/b1.aiff" \
  -f lavfi -i "anullsrc=r=24000:cl=mono:d=$PAUSE_SECONDS" \
  -i "$TEMPORARY_DIRECTORY/a2.aiff" \
  -filter_complex \
  '[0:a]aresample=24000,aformat=sample_fmts=fltp:channel_layouts=mono[a0];[1:a]aformat=sample_fmts=fltp:sample_rates=24000:channel_layouts=mono[s0];[2:a]aresample=24000,aformat=sample_fmts=fltp:channel_layouts=mono[a1];[3:a]aformat=sample_fmts=fltp:sample_rates=24000:channel_layouts=mono[s1];[4:a]aresample=24000,aformat=sample_fmts=fltp:channel_layouts=mono[a2];[a0][s0][a1][s1][a2]concat=n=5:v=0:a=1[out]' \
  -map '[out]' -c:a aac -b:a 96k -movflags +faststart \
  "$TEMPORARY_DIRECTORY/audio.m4a"

DURATION_SECONDS=$(ffprobe -v error -show_entries format=duration \
  -of default=noprint_wrappers=1:nokey=1 "$TEMPORARY_DIRECTORY/audio.m4a")
DURATION_MS=$(awk -v seconds="$DURATION_SECONDS" 'BEGIN { printf "%.0f", seconds * 1000 }')

mv -f -- "$TEMPORARY_DIRECTORY/audio.m4a" "$OUTPUT"
printf 'case_id\tsynthetic-zh-aba\nvoice_a\t%s\nvoice_b\t%s\nsay_rate\t%s\npause_seconds\t%s\nduration_ms\t%s\n' \
  "$VOICE_A" "$VOICE_B" "$SAY_RATE" "$PAUSE_SECONDS" "$DURATION_MS" \
  >"$CASE_DIRECTORY/generation.tsv"

printf '已生成（Git 忽略）：%s\n' "$OUTPUT"
printf 'A=%s，B=%s，顺序=A→B→A，时长=%s ms\n' \
  "$VOICE_A" "$VOICE_B" "$DURATION_MS"
