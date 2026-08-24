#!/usr/bin/env bash
set -euo pipefail
unset OPENROUTER_API_KEY

BENCHMARK_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
BUILD_DIRECTORY="$BENCHMARK_ROOT/.build"
OUTPUT="$BUILD_DIRECTORY/spt-bench"
TEMPORARY="$BUILD_DIRECTORY/.spt-bench.$$.tmp"

command -v cargo >/dev/null 2>&1 || {
  printf '缺少 cargo，无法构建离线评测器。\n' >&2
  exit 127
}

mkdir -p "$BUILD_DIRECTORY"
trap 'rm -f -- "$TEMPORARY"' EXIT
CARGO_TARGET_DIR="$BUILD_DIRECTORY/cargo-target" cargo build \
  --locked \
  --release \
  --manifest-path "$BENCHMARK_ROOT/Cargo.toml"
cp -- "$BUILD_DIRECTORY/cargo-target/release/spt-bench" "$TEMPORARY"
mv -f -- "$TEMPORARY" "$OUTPUT"
trap - EXIT
printf '%s\n' "$OUTPUT"
