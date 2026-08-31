#!/usr/bin/env bash
# Controlled five-family benchmark. Decoder and encoder metrics stay separate.

set -euo pipefail

MODEL_ROOT="${MODEL_ROOT:-${HOME:-}/.cache/lm-studio/models}"
MINISTRAL_MODEL="${MINISTRAL_MODEL:-}"
GEMMA_MODEL="${GEMMA_MODEL:-}"
QWEN_MODEL="${QWEN_MODEL:-}"
LLAMA_MODEL="${LLAMA_MODEL:-}"

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
  die "could not find $pattern below $MODEL_ROOT"
}

ministral_path=$(resolve_model "$MINISTRAL_MODEL" 'Ministral-3-3B-Instruct-2512-Q4_K_M.gguf')
gemma_path=$(resolve_model "$GEMMA_MODEL" 'gemma-4-12B-it-QAT-Q4_0.gguf')
qwen_path=$(resolve_model "$QWEN_MODEL" 'Qwen3.8-27B-Q4_K_M.gguf')
llama_path=$(resolve_model "$LLAMA_MODEL" 'Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf')

MODEL_SPECS=$(printf '%s\n%s\n%s\n%s\n' \
  "Ministral 3 3B Q4_K_M|$ministral_path" \
  "Gemma 4 12B QAT Q4_0|$gemma_path" \
  "Qwen 3.8 27B Q4_K_M|$qwen_path" \
  "Llama 3.1 8B Q4_K_M|$llama_path")

export MODEL_SPECS
export REPORT="${REPORT:-BENCHMARK_FAMILIES.md}"
export RAW_DIR="${RAW_DIR:-.bench_raw/families-reference}"
export REPORT_TITLE="${REPORT_TITLE:-RustyLLM vs llama.cpp — decoder families}"
export PROTOCOL_NOTE="${PROTOCOL_NOTE:-Nomic is encoder-only and is measured separately with embedding throughput; it is not mixed into decoder tokens/s. Qwen uses the locally available Qwen 3.8 27B text GGUF.}"

exec ./bench_reference.sh
