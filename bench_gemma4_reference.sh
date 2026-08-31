#!/usr/bin/env bash
# Reproducible Gemma 4 comparison using the generic external-reference harness.

set -euo pipefail

MODEL_ROOT="${MODEL_ROOT:-${HOME:-}/.cache/lm-studio/models}"
GEMMA4_12B_MODEL="${GEMMA4_12B_MODEL:-}"
GEMMA4_26B_MODEL="${GEMMA4_26B_MODEL:-}"
GEMMA4_E2B_MODEL="${GEMMA4_E2B_MODEL:-}"

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
  [ -d "$MODEL_ROOT" ] || die "MODEL_ROOT does not exist: $MODEL_ROOT"
  while IFS= read -r path; do
    printf '%s\n' "$path"
    return 0
  done < <(find "$MODEL_ROOT" -type f -name "$pattern" -print 2>/dev/null | sort)
  die "could not find $pattern below $MODEL_ROOT; set the corresponding GEMMA4_*_MODEL variable"
}

gemma4_12b_path=$(resolve_model "$GEMMA4_12B_MODEL" 'gemma-4-12B-it-QAT-Q4_0.gguf')
gemma4_26b_path=$(resolve_model "$GEMMA4_26B_MODEL" 'gemma-4-26B-A4B-it-Q4_K_M.gguf')
gemma4_e2b_path=$(resolve_model "$GEMMA4_E2B_MODEL" 'gemma-4-E2B-it-Q4_K_M.gguf')

MODEL_SPECS=$(printf '%s\n%s\n%s\n' \
  "Gemma 4 12B QAT Q4_0|$gemma4_12b_path" \
  "Gemma 4 26B-A4B Q4_K_M|$gemma4_26b_path" \
  "Gemma 4 E2B Q4_K_M|$gemma4_e2b_path")

export MODEL_SPECS
export REPORT="${REPORT:-BENCHMARK_GEMMA4.md}"
export RAW_DIR="${RAW_DIR:-.bench_raw/gemma4-reference}"
export REPORT_TITLE="${REPORT_TITLE:-RustyLLM vs external GGUF reference — Gemma 4}"
export PROTOCOL_NOTE="${PROTOCOL_NOTE:-The suite covers the dense 12B QAT model and the sparse 26B-A4B and E2B variants. RustyLLM auto-selected Metal for 12B and CPU for 26B-A4B/E2B; explicit Metal checks on the sparse variants were slower. Decode throughput is the primary cross-engine metric; short-prompt prefill can differ when prompt rendering or one-time GPU initialization differs.}"

exec ./bench_reference.sh
