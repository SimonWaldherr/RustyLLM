#!/usr/bin/env bash
# Runs one decoder family in isolation so its cooldown can be sized independently.

set -euo pipefail

FAMILY="${FAMILY:-}"
MODEL_ROOT="${MODEL_ROOT:-${HOME:-}/.cache/lm-studio/models}"
INITIAL_COOLDOWN_SECS="${INITIAL_COOLDOWN_SECS:-60}"

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

resolve_model() {
  local explicit="$1"
  local pattern="$2"
  local path
  if [ -n "$explicit" ]; then
    [ -f "$explicit" ] || die "model does not exist: $explicit"
    printf '%s\n' "$explicit"
    return 0
  fi
  while IFS= read -r path; do
    printf '%s\n' "$path"
    return 0
  done < <(find "$MODEL_ROOT" -type f -name "$pattern" -print 2>/dev/null | sort)
  die "could not find $pattern below $MODEL_ROOT"
}

case "$FAMILY" in
  ministral)
    label="Ministral 3 3B Q4_K_M"
    model=$(resolve_model "${MINISTRAL_MODEL:-}" 'Ministral-3-3B-Instruct-2512-Q4_K_M.gguf')
    ;;
  gemma)
    label="Gemma 4 12B QAT Q4_0"
    model=$(resolve_model "${GEMMA_MODEL:-}" 'gemma-4-12B-it-QAT-Q4_0.gguf')
    ;;
  qwen)
    label="Qwen 3.8 27B Q4_K_M"
    model=$(resolve_model "${QWEN_MODEL:-}" 'Qwen3.8-27B-Q4_K_M.gguf')
    ;;
  llama)
    label="Llama 3.1 8B Q4_K_M"
    model=$(resolve_model "${LLAMA_MODEL:-}" 'Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf')
    ;;
  *)
    die "set FAMILY to ministral, gemma, qwen, or llama"
    ;;
esac

export MODEL_SPECS="$label|$model"
export REPORT="${REPORT:-BENCHMARK_${FAMILY^^}_CONTROLLED.md}"
export RAW_DIR="${RAW_DIR:-.bench_raw/families-controlled/$FAMILY}"
export REPORT_TITLE="${REPORT_TITLE:-RustyLLM vs llama.cpp — $label controlled}"
export PROTOCOL_NOTE="${PROTOCOL_NOTE:-This family was run in isolation after an initial ${INITIAL_COOLDOWN_SECS}-second cooling interval; its per-process cooldown is recorded above.}"

case "$INITIAL_COOLDOWN_SECS" in
  ''|*[!0-9]*) die "INITIAL_COOLDOWN_SECS must be a non-negative integer" ;;
esac
if [ "$INITIAL_COOLDOWN_SECS" != "0" ]; then
  printf 'Initial cooldown: %s seconds\n' "$INITIAL_COOLDOWN_SECS"
  remaining="$INITIAL_COOLDOWN_SECS"
  while [ "$remaining" -gt 0 ]; do
    if [ "$remaining" -gt 60 ]; then chunk=60; else chunk="$remaining"; fi
    sleep "$chunk"
    remaining=$((remaining - chunk))
  done
fi

exec ./bench_reference.sh
