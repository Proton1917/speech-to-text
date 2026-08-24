#!/usr/bin/env bash
set -euo pipefail
umask 077
unset OPENROUTER_API_KEY

BENCHMARK_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
OUTPUT_ROOT="$BENCHMARK_ROOT/public-fixtures"
OUTPUT_DIRECTORY="$OUTPUT_ROOT/ascend-zh-aba"
DATASET_ID='CAiRE/ASCEND'
DATASET_REVISION='737e9800ae31be9932ba8464c80366559bd28424'
DATASET_CONFIG='main'
DATASET_SPLIT='test'
DATASET_LICENSE='cc-by-sa-4.0'
PAUSE_SECONDS='0.8'
EXPECTED_OUTPUT_DURATION='14.80'
FORCE=0

case "${1:-}" in
  '') ;;
  --force) FORCE=1 ;;
  -h|--help)
    printf '用法：%s [--force]\n' "$0"
    printf '%s\n' '手动联网下载 CAiRE/ASCEND 的三个公开 test 样本并生成真人 A→B→A fixture。'
    exit 0
    ;;
  *)
    printf '用法：%s [--force]\n' "$0" >&2
    exit 64
    ;;
esac

if [[ -e "$OUTPUT_DIRECTORY" && $FORCE -ne 1 ]]; then
  printf '输出已存在：%s；如需重新生成请显式使用 --force。\n' "$OUTPUT_DIRECTORY" >&2
  exit 73
fi

for command_name in curl jq ffmpeg ffprobe cargo; do
  command -v "$command_name" >/dev/null 2>&1 || {
    printf '缺少命令：%s\n' "$command_name" >&2
    exit 127
  }
done

TEMPORARY_DIRECTORY=$(mktemp -d "${TMPDIR:-/tmp}/spt-ascend-aba.XXXXXX")
STAGING_DIRECTORY=''
BACKUP_DIRECTORY=''
cleanup() {
  rm -rf -- "$TEMPORARY_DIRECTORY"
  if [[ -n "$STAGING_DIRECTORY" && -e "$STAGING_DIRECTORY" ]]; then
    rm -rf -- "$STAGING_DIRECTORY"
  fi
  if [[ -n "$BACKUP_DIRECTORY" && -e "$BACKUP_DIRECTORY" && ! -e "$OUTPUT_DIRECTORY" ]]; then
    mv -- "$BACKUP_DIRECTORY" "$OUTPUT_DIRECTORY"
  fi
}
trap cleanup EXIT

fetch_json() {
  local url=$1
  local destination=$2
  env -u OPENROUTER_API_KEY curl \
    --proto '=https' \
    --tlsv1.2 \
    --fail \
    --silent \
    --show-error \
    --location \
    --connect-timeout 15 \
    --max-time 60 \
    --retry 2 \
    --max-filesize 1048576 \
    --output "$destination" \
    "$url"
}

download_audio() {
  local url=$1
  local destination=$2
  env -u OPENROUTER_API_KEY curl \
    --proto '=https' \
    --tlsv1.2 \
    --fail \
    --silent \
    --show-error \
    --location \
    --connect-timeout 15 \
    --max-time 120 \
    --retry 2 \
    --max-filesize 16777216 \
    --output "$destination" \
    "$url"
  local byte_count
  byte_count=$(wc -c <"$destination")
  [[ "$byte_count" -gt 44 && "$byte_count" -le 16777216 ]] || {
    printf '下载音频大小异常：%s bytes\n' "$byte_count" >&2
    exit 65
  }
}

METADATA_URL='https://huggingface.co/api/datasets/CAiRE/ASCEND'
fetch_json "$METADATA_URL" "$TEMPORARY_DIRECTORY/metadata.json"
env -u OPENROUTER_API_KEY jq -e \
  --arg dataset "$DATASET_ID" \
  --arg revision "$DATASET_REVISION" \
  --arg license "$DATASET_LICENSE" \
  '
    .id == $dataset and
    .sha == $revision and
    .private == false and
    .gated == false and
    .disabled == false and
    ((.cardData.license // []) |
      if type == "array" then . else [.] end |
      map(ascii_downcase) |
      index($license) != null)
  ' "$TEMPORARY_DIRECTORY/metadata.json" >/dev/null || {
  printf '%s\n' \
    'CAiRE/ASCEND 当前 metadata 的 SHA、公开状态或 CC-BY-SA-4.0 许可证与已审核契约不一致；拒绝下载。' >&2
  exit 65
}

ROW_INDEXES=(400 904 401)
ROW_IDS=(00400 00904 00401)
ROW_SPEAKERS=(3 17 3)
ROW_DURATIONS=(5.72 4.82 2.66)
ROW_LANGUAGES=(mixed zh mixed)
ROW_TRANSCRIPTS=(
  '就你要申请去交换你并不需要去说哦我要去哪一个department交换是直接选school'
  '就是你是在大学的时候哪一个阶段才萌生'
  '我要去这个university做交换'
)
AUDIO_URLS=()

for position in 0 1 2; do
  row_index=${ROW_INDEXES[$position]}
  row_id=${ROW_IDS[$position]}
  row_speaker=${ROW_SPEAKERS[$position]}
  row_duration=${ROW_DURATIONS[$position]}
  row_language=${ROW_LANGUAGES[$position]}
  row_transcript=${ROW_TRANSCRIPTS[$position]}
  row_json="$TEMPORARY_DIRECTORY/row-$row_index.json"
  row_url="https://datasets-server.huggingface.co/rows?dataset=CAiRE%2FASCEND&config=$DATASET_CONFIG&split=$DATASET_SPLIT&offset=$row_index&length=1"
  fetch_json "$row_url" "$row_json"

  audio_prefix="https://datasets-server.huggingface.co/cached-assets/CAiRE/ASCEND/--/$DATASET_REVISION/--/$DATASET_CONFIG/$DATASET_SPLIT/$row_index/audio/audio.wav?"
  env -u OPENROUTER_API_KEY jq -e \
    --argjson row_index "$row_index" \
    --arg id "$row_id" \
    --argjson speaker "$row_speaker" \
    --arg transcript "$row_transcript" \
    --argjson duration "$row_duration" \
    --arg language "$row_language" \
    --arg audio_prefix "$audio_prefix" \
    '
      (.rows | length == 1) and
      .rows[0].row_idx == $row_index and
      .rows[0].row.id == $id and
      .rows[0].row.original_speaker_id == $speaker and
      .rows[0].row.transcription == $transcript and
      .rows[0].row.duration >= ($duration - 0.001) and
      .rows[0].row.duration <= ($duration + 0.001) and
      .rows[0].row.language == $language and
      (.rows[0].truncated_cells | length == 0) and
      (.rows[0].row.audio | length == 1) and
      .rows[0].row.audio[0].type == "audio/wav" and
      (.rows[0].row.audio[0].src | type == "string" and startswith($audio_prefix))
    ' "$row_json" >/dev/null || {
    printf 'Dataset Viewer 的 test row %s 与已审核 id/speaker/transcript/duration/audio 契约不一致；拒绝下载。\n' \
      "$row_index" >&2
    exit 65
  }
  AUDIO_URLS[$position]=$(env -u OPENROUTER_API_KEY jq -er '.rows[0].row.audio[0].src' "$row_json")
done

# 三行 metadata 全部通过后才开始下载动态 signed URL。
for position in 0 1 2; do
  row_index=${ROW_INDEXES[$position]}
  audio_path="$TEMPORARY_DIRECTORY/row-$row_index.wav"
  probe_path="$TEMPORARY_DIRECTORY/row-$row_index.probe.json"
  download_audio "${AUDIO_URLS[$position]}" "$audio_path"
  env -u OPENROUTER_API_KEY ffprobe \
    -v error \
    -show_entries format=duration:stream=codec_type,sample_rate,channels \
    -of json \
    "$audio_path" >"$probe_path"
  env -u OPENROUTER_API_KEY jq -e \
    --argjson duration "${ROW_DURATIONS[$position]}" \
    '
      (.streams | length == 1) and
      .streams[0].codec_type == "audio" and
      (.streams[0].sample_rate | tonumber) == 16000 and
      .streams[0].channels == 1 and
      (.format.duration | tonumber) >= ($duration - 0.05) and
      (.format.duration | tonumber) <= ($duration + 0.05)
    ' "$probe_path" >/dev/null || {
    printf '下载的 test row %s 不是预期的 16 kHz 单声道 WAV，或媒体时长不匹配。\n' \
      "$row_index" >&2
    exit 65
  }
done

mkdir -p "$OUTPUT_ROOT"
STAGING_DIRECTORY=$(mktemp -d "$OUTPUT_ROOT/.ascend-zh-aba.XXXXXX")

env -u OPENROUTER_API_KEY ffmpeg -hide_banner -loglevel error -y \
  -i "$TEMPORARY_DIRECTORY/row-400.wav" \
  -f lavfi -i "anullsrc=r=16000:cl=mono:d=$PAUSE_SECONDS" \
  -i "$TEMPORARY_DIRECTORY/row-904.wav" \
  -f lavfi -i "anullsrc=r=16000:cl=mono:d=$PAUSE_SECONDS" \
  -i "$TEMPORARY_DIRECTORY/row-401.wav" \
  -filter_complex \
  '[0:a]aresample=16000,aformat=sample_fmts=s16:channel_layouts=mono[a0];[1:a]aformat=sample_fmts=s16:sample_rates=16000:channel_layouts=mono[s0];[2:a]aresample=16000,aformat=sample_fmts=s16:channel_layouts=mono[a1];[3:a]aformat=sample_fmts=s16:sample_rates=16000:channel_layouts=mono[s1];[4:a]aresample=16000,aformat=sample_fmts=s16:channel_layouts=mono[a2];[a0][s0][a1][s1][a2]concat=n=5:v=0:a=1[out]' \
  -map '[out]' \
  -c:a flac \
  -compression_level 8 \
  "$STAGING_DIRECTORY/audio.flac"

OUTPUT_DURATION=$(env -u OPENROUTER_API_KEY ffprobe \
  -v error \
  -show_entries format=duration \
  -of default=noprint_wrappers=1:nokey=1 \
  "$STAGING_DIRECTORY/audio.flac")
env -u OPENROUTER_API_KEY jq -en \
  --argjson actual "$OUTPUT_DURATION" \
  --argjson expected "$EXPECTED_OUTPUT_DURATION" \
  '$actual >= ($expected - 0.05) and $actual <= ($expected + 0.05)' >/dev/null || {
  printf '拼接音频时长异常：实际 %s 秒，预期 %s 秒。\n' \
    "$OUTPUT_DURATION" "$EXPECTED_OUTPUT_DURATION" >&2
  exit 65
}

printf '%s\n' \
  '# key<TAB>value' \
  $'case_id\tascend-zh-aba' \
  $'language\tzh-CN-mixed-en' \
  $'unicode_normalization\tNFC' \
  $'audio_path\taudio.flac' \
  $'provenance\tCAiRE-ASCEND-test-rows-400-904-401' \
  $'speaker_sequence\tA,B,A' \
  $'dataset\tCAiRE/ASCEND' \
  $'dataset_revision\t737e9800ae31be9932ba8464c80366559bd28424' \
  $'dataset_config\tmain' \
  $'dataset_split\ttest' \
  $'dataset_rows\t400,904,401' \
  $'license\tCC-BY-SA-4.0' \
  $'license_url\thttps://creativecommons.org/licenses/by-sa/4.0/' \
  $'pause_seconds\t0.8' \
  >"$STAGING_DIRECTORY/case.tsv"

printf '%s\n' \
  '# speaker<TAB>reference text' \
  $'A\t就你要申请去交换你并不需要去说哦我要去哪一个department交换是直接选school' \
  $'B\t就是你是在大学的时候哪一个阶段才萌生' \
  $'A\t我要去这个university做交换' \
  >"$STAGING_DIRECTORY/turns.tsv"

printf '%s\n' '# intentionally empty: this fixture measures CER and A→B→A speaker consistency' \
  >"$STAGING_DIRECTORY/terms.tsv"

printf 'dataset\t%s\ntitle\tASCEND: A Spontaneous Chinese-English Dataset for Code-switching in Multi-turn Conversation\nattribution\tHoly Lovenia et al., LREC 2022\ndataset_revision\t%s\ndataset_config\t%s\ndataset_split\t%s\ndataset_rows\t400,904,401\nlicense\tCC-BY-SA-4.0\nlicense_url\thttps://creativecommons.org/licenses/by-sa/4.0/\nsource\thttps://huggingface.co/datasets/CAiRE/ASCEND\npaper\thttps://arxiv.org/abs/2112.06223\nadaptation\ttest rows 400,904,401 reordered as A-B-A with two 0.8-second silences and lossless FLAC encoding\noutput_duration_seconds\t%s\n' \
  "$DATASET_ID" "$DATASET_REVISION" "$DATASET_CONFIG" "$DATASET_SPLIT" "$OUTPUT_DURATION" \
  >"$STAGING_DIRECTORY/generation.tsv"

EVALUATOR=$(env -u OPENROUTER_API_KEY "$BENCHMARK_ROOT/scripts/build.sh")
env -u OPENROUTER_API_KEY "$EVALUATOR" evaluate \
  --case "$STAGING_DIRECTORY" \
  --report "$TEMPORARY_DIRECTORY/fixture-validation.tsv"
grep -F $'unicode_normalization\tNFC' "$TEMPORARY_DIRECTORY/fixture-validation.tsv" >/dev/null
grep -F $'reference_turns\t3' "$TEMPORARY_DIRECTORY/fixture-validation.tsv" >/dev/null

if [[ -e "$OUTPUT_DIRECTORY" ]]; then
  BACKUP_DIRECTORY="$OUTPUT_ROOT/.ascend-zh-aba.backup.$$"
  [[ ! -e "$BACKUP_DIRECTORY" ]] || {
    printf '备份路径已存在：%s\n' "$BACKUP_DIRECTORY" >&2
    exit 73
  }
  mv -- "$OUTPUT_DIRECTORY" "$BACKUP_DIRECTORY"
fi
mv -- "$STAGING_DIRECTORY" "$OUTPUT_DIRECTORY"
STAGING_DIRECTORY=''
if [[ -n "$BACKUP_DIRECTORY" ]]; then
  rm -rf -- "$BACKUP_DIRECTORY"
  BACKUP_DIRECTORY=''
fi

printf '已生成公开真人短样本（Git 忽略）：%s\n' "$OUTPUT_DIRECTORY/audio.flac"
printf '来源：%s @ %s，test rows 400→904→401，CC-BY-SA-4.0。\n' \
  "$DATASET_ID" "$DATASET_REVISION"
printf '%s\n' \
  '这是三段真实单人语音的人工 A→B→A 拼接，不是自然连续会议，也不能单独证明生产质量。'
