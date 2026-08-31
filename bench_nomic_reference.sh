#!/usr/bin/env bash
# Controlled end-to-end embedding benchmark for the local Nomic GGUF.

set -euo pipefail

RUSTY_BIN="${RUSTY_BIN:-./target/release-max/rusty-llm}"
LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-/opt/homebrew/bin/llama-server}"
MODEL_ROOT="${MODEL_ROOT:-${HOME:-}/.cache/lm-studio/models}"
NOMIC_MODEL="${NOMIC_MODEL:-}"
RUNS="${RUNS:-3}"
WARMUPS="${WARMUPS:-2}"
REQUESTS="${REQUESTS:-10}"
THREADS="${THREADS:-12}"
COOLDOWN_SECS="${COOLDOWN_SECS:-10}"
RUSTY_PORT="${RUSTY_PORT:-18271}"
LLAMA_PORT="${LLAMA_PORT:-18272}"
RAW_DIR="${RAW_DIR:-.bench_raw/nomic-reference}"
REPORT="${REPORT:-BENCHMARK_NOMIC.md}"
INPUT_TEXT="${INPUT_TEXT:-search_document: Memory bandwidth determines how quickly model weights can reach the processor during local inference. Quantized weights reduce traffic, while batching improves arithmetic intensity. A reliable comparison uses the same model file, tokenizer, input, pooling rule, normalization, thread count, and warm server state. Repeated measurements and alternating engine order reduce random and thermal bias. The benchmark must distinguish encoder throughput from autoregressive decode throughput because they exercise different execution paths. Long embedding inputs expose matrix multiplication efficiency and scheduler overhead more clearly than very short requests. Local HTTP latency, tokenization, encoder execution, mean pooling, normalization, and response serialization are all included in this end-to-end measurement. Thermal state is captured before and after every server run. Results are retained as raw per-request timings so later changes can be checked against the same protocol.}"

RESULTS_TSV="$RAW_DIR/results.tsv"
THERMAL_TSV="$RAW_DIR/thermal.tsv"
ENV_TSV="$RAW_DIR/environment.tsv"

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

resolve_model() {
  local path
  if [ -n "$NOMIC_MODEL" ]; then
    [ -f "$NOMIC_MODEL" ] || die "model does not exist: $NOMIC_MODEL"
    printf '%s\n' "$NOMIC_MODEL"
    return 0
  fi
  [ -d "$MODEL_ROOT" ] || die "MODEL_ROOT does not exist: $MODEL_ROOT"
  while IFS= read -r path; do
    printf '%s\n' "$path"
    return 0
  done < <(find "$MODEL_ROOT" -type f -name 'nomic-embed-text-v1.5.Q4_K_M.gguf' -print 2>/dev/null | sort)
  die "could not find nomic-embed-text-v1.5.Q4_K_M.gguf below $MODEL_ROOT"
}

cooldown() {
  if [ "$COOLDOWN_SECS" != "0" ]; then
    sleep "$COOLDOWN_SECS"
  fi
}

capture_thermal() {
  local engine="$1"
  local run="$2"
  local phase="$3"
  local state="unavailable"
  if command -v pmset >/dev/null 2>&1; then
    state=$(pmset -g therm 2>&1 || true)
    state=${state//$'\n'/; }
    state=${state//$'\t'/ }
    [ -n "$state" ] || state="unavailable"
  fi
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$engine" "$run" "$phase" "$state" \
    >> "$THERMAL_TSV"
}

wait_ready() {
  local url="$1"
  local pid="$2"
  local attempt
  for attempt in $(seq 1 120); do
    if curl -sS --fail --max-time 1 "$url/health" >/dev/null 2>&1; then
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      return 1
    fi
    sleep 1
  done
  return 1
}

stop_server() {
  local pid="$1"
  if kill -0 "$pid" 2>/dev/null; then
    kill "$pid"
    wait "$pid" 2>/dev/null || true
  fi
}

median_file() {
  LC_ALL=C sort -n "$1" | LC_ALL=C awk '
    { values[++count] = $1 }
    END {
      if (!count) exit 1
      if (count % 2) printf "%.9f", values[(count + 1) / 2]
      else printf "%.9f", (values[count / 2] + values[count / 2 + 1]) / 2
    }
  '
}

mean_for() {
  LC_ALL=C awk -F '\t' -v engine="$1" -v column="$2" '
    $1 == engine { sum += $column; count++ }
    END { if (count) printf "%.6f", sum / count }
  ' "$RESULTS_TSV"
}

sd_for() {
  LC_ALL=C awk -F '\t' -v engine="$1" -v column="$2" '
    $1 == engine { values[++count] = $column; sum += $column }
    END {
      if (!count) exit
      mean = sum / count
      for (i = 1; i <= count; i++) squared += (values[i] - mean) ^ 2
      printf "%.6f", sqrt(squared / count)
    }
  ' "$RESULTS_TSV"
}

run_engine() {
  local engine="$1"
  local run="$2"
  local order="$3"
  local model="$4"
  local port log_file endpoint pid response times_file latency token_count dimension
  local median_latency throughput request_index warmup_index

  log_file="$RAW_DIR/${engine//[^A-Za-z0-9]/_}.run${run}.server.log"
  response="$RAW_DIR/${engine//[^A-Za-z0-9]/_}.run${run}.response.json"
  times_file="$RAW_DIR/${engine//[^A-Za-z0-9]/_}.run${run}.latencies.txt"
  : > "$times_file"

  printf '[Nomic][%s/%s][%s/2] %s\n' "$run" "$RUNS" "$order" "$engine"
  capture_thermal "$engine" "$run" "before"
  if [ "$engine" = "RustyLLM" ]; then
    port="$RUSTY_PORT"
    endpoint="http://127.0.0.1:$port"
    RUSTY_LLM_METAL=0 "$RUSTY_BIN" --model "$model" --backend cpu \
      --threads "$THREADS" --threads-batch "$THREADS" \
      --serve "127.0.0.1:$port" > "$log_file" 2>&1 &
  else
    port="$LLAMA_PORT"
    endpoint="http://127.0.0.1:$port"
    "$LLAMA_SERVER_BIN" -m "$model" --host 127.0.0.1 --port "$port" \
      --embedding --pooling mean --embd-normalize 2 -ngl 0 \
      -t "$THREADS" -tb "$THREADS" > "$log_file" 2>&1 &
  fi
  pid=$!

  if ! wait_ready "$endpoint" "$pid"; then
    stop_server "$pid"
    die "$engine server did not become ready; see $log_file"
  fi

  for warmup_index in $(seq 1 "$WARMUPS"); do
    curl -sS --fail -o "$response" -H 'Content-Type: application/json' \
      -d "$REQUEST_BODY" "$endpoint/v1/embeddings" >/dev/null
  done

  for request_index in $(seq 1 "$REQUESTS"); do
    latency=$(curl -sS --fail -o "$response" -w '%{time_total}' \
      -H 'Content-Type: application/json' -d "$REQUEST_BODY" \
      "$endpoint/v1/embeddings")
    printf '%s\n' "$latency" >> "$times_file"
  done

  token_count=$(jq -r '.usage.prompt_tokens // .usage.total_tokens // empty' "$response")
  dimension=$(jq -r '.data[0].embedding | length' "$response")
  if [ -z "$token_count" ] || [ "$token_count" = "0" ] || [ "$dimension" = "0" ]; then
    stop_server "$pid"
    die "$engine returned an invalid embedding response; see $response"
  fi
  median_latency=$(median_file "$times_file")
  throughput=$(LC_ALL=C awk -v tokens="$token_count" -v seconds="$median_latency" \
    'BEGIN { printf "%.6f", tokens / seconds }')
  stop_server "$pid"
  capture_thermal "$engine" "$run" "after"

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$engine" "$run" "$order" "$median_latency" "$throughput" \
    "$token_count" "$dimension" "$times_file" >> "$RESULTS_TSV"
  printf '  median: %.2f ms, %.1f tokens/s (%s tokens)\n' \
    "$(LC_ALL=C awk -v seconds="$median_latency" 'BEGIN { print seconds * 1000 }')" \
    "$throughput" "$token_count"
}

write_report() {
  local model="$1"
  local rusty_latency rusty_latency_sd rusty_rate rusty_rate_sd
  local llama_latency llama_latency_sd llama_rate llama_rate_sd delta winner thermal_note
  local agreement_json agreement_cosine agreement_dimensions agreement_tokens
  rusty_latency=$(mean_for RustyLLM 4)
  rusty_latency_sd=$(sd_for RustyLLM 4)
  rusty_rate=$(mean_for RustyLLM 5)
  rusty_rate_sd=$(sd_for RustyLLM 5)
  llama_latency=$(mean_for llama.cpp 4)
  llama_latency_sd=$(sd_for llama.cpp 4)
  llama_rate=$(mean_for llama.cpp 5)
  llama_rate_sd=$(sd_for llama.cpp 5)
  delta=$(LC_ALL=C awk -v rusty="$rusty_rate" -v other="$llama_rate" \
    'BEGIN { printf "%+.1f", (rusty / other - 1) * 100 }')
  winner=$(LC_ALL=C awk -v rusty="$rusty_rate" -v other="$llama_rate" \
    'BEGIN { print (rusty >= other) ? "RustyLLM" : "llama.cpp" }')
  agreement_json=$(jq -s '
    .[0].data[0].embedding as $a
    | .[1].data[0].embedding as $b
    | ([range(0; $a | length) as $i | $a[$i] * $b[$i]] | add) as $dot
    | ([$a[] | . * .] | add | sqrt) as $na
    | ([$b[] | . * .] | add | sqrt) as $nb
    | {
        dimensions: [$a | length, $b | length],
        tokens: [.[0].usage.prompt_tokens, .[1].usage.prompt_tokens],
        cosine: ($dot / ($na * $nb))
      }
  ' "$RAW_DIR/RustyLLM.run${RUNS}.response.json" \
    "$RAW_DIR/llama_cpp.run${RUNS}.response.json")
  agreement_cosine=$(jq -r '.cosine' <<< "$agreement_json")
  agreement_dimensions=$(jq -r '.dimensions | unique | if length == 1 then .[0] else join("/") end' <<< "$agreement_json")
  agreement_tokens=$(jq -r '.tokens | unique | if length == 1 then .[0] else join("/") end' <<< "$agreement_json")
  thermal_note="No macOS thermal warning was reported during the captured checkpoints."
  if rg -qi 'CPU_Speed_Limit[^0-9]+([0-9]|[1-9][0-9])([^0-9]|$)|thermal warning level[^0-9]+[1-9]|performance warning level[^0-9]+[1-9]' "$THERMAL_TSV"; then
    thermal_note="At least one thermal checkpoint indicated a possible limit; interpret the comparison cautiously."
  fi

  {
    printf '# RustyLLM vs llama.cpp — Nomic embeddings\n\n'
    printf 'Generated: `%s`  \n' "$(date '+%Y-%m-%d %H:%M:%S %Z')"
    printf 'Model: `%s`  \n' "$model"
    printf 'Rust revision: `%s + working tree changes`  \n' "$(git rev-parse --short HEAD 2>/dev/null || printf unknown)"
    printf 'llama.cpp: `%s`\n\n' "$($LLAMA_SERVER_BIN --version 2>&1 | head -n 1)"
    printf '## Result\n\n'
    printf '| Workload | RustyLLM | llama.cpp | RustyLLM delta | Faster |\n'
    printf '|---|---:|---:|---:|---|\n'
    printf '| Nomic embed, end-to-end | %.1f ± %.1f tokens/s | %.1f ± %.1f tokens/s | %s%% | %s |\n\n' \
      "$rusty_rate" "$rusty_rate_sd" "$llama_rate" "$llama_rate_sd" "$delta" "$winner"
    printf '| Engine | Median HTTP latency per fresh server run |\n'
    printf '|---|---:|\n'
    printf '| RustyLLM | %.2f ± %.2f ms |\n' \
      "$(LC_ALL=C awk -v v="$rusty_latency" 'BEGIN { print v * 1000 }')" \
      "$(LC_ALL=C awk -v v="$rusty_latency_sd" 'BEGIN { print v * 1000 }')"
    printf '| llama.cpp | %.2f ± %.2f ms |\n\n' \
      "$(LC_ALL=C awk -v v="$llama_latency" 'BEGIN { print v * 1000 }')" \
      "$(LC_ALL=C awk -v v="$llama_latency_sd" 'BEGIN { print v * 1000 }')"
    printf 'Response check: both engines reported `%s` prompt tokens and `%s` dimensions; the final-run embedding cosine was `%.6f`.\n\n' \
      "$agreement_tokens" "$agreement_dimensions" "$agreement_cosine"
    printf '## Protocol\n\n'
    printf -- '- Same GGUF, exact request body, mean pooling, L2 normalization, CPU backend, and `%s` threads.\n' "$THREADS"
    printf -- '- `%s` fresh server runs per engine, alternating order; `%s` warm-ups and `%s` measured requests per server.\n' "$RUNS" "$WARMUPS" "$REQUESTS"
    printf -- '- Reported rate is prompt tokens divided by the median end-to-end loopback HTTP latency for each fresh server run; table values are mean ± population SD across runs.\n'
    printf -- '- Latency includes tokenization, encoder forward, pooling, normalization, JSON serialization, and loopback HTTP. Model loading and warm-up are excluded.\n'
    printf -- '- Nomic is an encoder-only workload and is intentionally not compared with autoregressive decode tokens/s.\n'
    printf -- '- %s\n\n' "$thermal_note"
    printf 'Raw evidence: `%s`, `%s`.\n' "$RESULTS_TSV" "$THERMAL_TSV"
  } > "$REPORT"
}

require_cmd curl
require_cmd jq
require_cmd rg
[ -x "$RUSTY_BIN" ] || die "RustyLLM binary not found: $RUSTY_BIN"
[ -x "$LLAMA_SERVER_BIN" ] || die "llama server binary not found: $LLAMA_SERVER_BIN"
case "$RUNS:$WARMUPS:$REQUESTS:$THREADS" in
  *[!0-9:]*|0:*|*:0:*|*:*:0:*|*:*:*:0) die "runs, warmups, requests, and threads must be positive integers" ;;
esac

model=$(resolve_model)
mkdir -p "$RAW_DIR"
printf 'engine\trun\torder\tmedian_latency_s\ttokens_per_s\tprompt_tokens\tdimension\tlatencies_file\n' > "$RESULTS_TSV"
printf 'timestamp\tengine\trun\tphase\tthermal_state\n' > "$THERMAL_TSV"
REQUEST_BODY=$(jq -cn --arg input "$INPUT_TEXT" '{model:"nomic-embed", input:$input}')
export REQUEST_BODY

{
  printf 'generated_at\t%s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')"
  printf 'model\t%s\n' "$model"
  printf 'runs\t%s\n' "$RUNS"
  printf 'warmups\t%s\n' "$WARMUPS"
  printf 'requests\t%s\n' "$REQUESTS"
  printf 'threads\t%s\n' "$THREADS"
  printf 'cooldown_seconds\t%s\n' "$COOLDOWN_SECS"
  printf 'input_sha256\t%s\n' "$(printf '%s' "$INPUT_TEXT" | shasum -a 256 | awk '{print $1}')"
} > "$ENV_TSV"

for run in $(seq 1 "$RUNS"); do
  if [ $((run % 2)) -eq 1 ]; then
    run_engine RustyLLM "$run" 1 "$model"
    cooldown
    run_engine llama.cpp "$run" 2 "$model"
  else
    run_engine llama.cpp "$run" 1 "$model"
    cooldown
    run_engine RustyLLM "$run" 2 "$model"
  fi
  if [ "$run" -lt "$RUNS" ]; then
    cooldown
  fi
done

write_report "$model"
printf 'Wrote %s\n' "$REPORT"
