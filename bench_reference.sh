#!/usr/bin/env bash
# Reproducible end-to-end decode comparison against an external GGUF engine.
#
# The same GGUF file, user prompt, greedy sampler, token limit, and Apple GPU
# are used by both engines. Each measured run starts a fresh process; engine
# order alternates to reduce thermal/order bias. Model loading is not part of
# the reported token rate.
#
# Usage:
#   REFERENCE_BIN=/path/to/reference-cli ./bench_reference.sh
#   RUNS=5 MAX_TOKENS=128 REFERENCE_BIN=/path/to/reference-cli ./bench_reference.sh
#   REPORT_ONLY=1 ./bench_reference.sh
#   MINISTRAL_MODEL=/models/ministral.gguf \
#     LLAMA_MODEL=/models/llama.gguf \
#     QWEN_MODEL=/models/qwen.gguf REFERENCE_BIN=/path/to/reference-cli ./bench_reference.sh
#
# Output:
#   BENCHMARK_REFERENCE.md
#   .bench_raw/reference/{results.tsv,models.tsv,*.log,*.json}

set -euo pipefail

RUSTY_BIN="${RUSTY_BIN:-./target/release-max/rusty-llm}"
REFERENCE_BIN="${REFERENCE_BIN:-}"
REFERENCE_KIND="${REFERENCE_KIND:-cli}"
REFERENCE_PROMPT_TOKENS="${REFERENCE_PROMPT_TOKENS:-32}"
MODEL_ROOT="${MODEL_ROOT:-${HOME:-}/.cache/lm-studio/models}"
MINISTRAL_MODEL="${MINISTRAL_MODEL:-}"
LLAMA_MODEL="${LLAMA_MODEL:-}"
QWEN_MODEL="${QWEN_MODEL:-}"
RUNS="${RUNS:-3}"
MAX_TOKENS="${MAX_TOKENS:-64}"
COOLDOWN_SECS="${COOLDOWN_SECS:-10}"
SEED="${SEED:-42}"
PROMPT="${PROMPT:-Explain in one compact paragraph why memory bandwidth matters for local LLM inference.}"
RAW_DIR="${RAW_DIR:-.bench_raw/reference}"
REPORT="${REPORT:-BENCHMARK_REFERENCE.md}"
REPORT_ONLY="${REPORT_ONLY:-0}"
MODEL_SPECS="${MODEL_SPECS:-}"
REPORT_TITLE="${REPORT_TITLE:-RustyLLM vs external GGUF reference — Ministral, Llama, Qwen}"
PROTOCOL_NOTE="${PROTOCOL_NOTE:-Qwen uses the locally available DeepSeek-R1-Distill-Qwen-7B (Qwen2 architecture). RustyLLM currently falls back to its plain role renderer for this model, while the reference uses its embedded template; decode rates remain useful, but its short-prompt prefill comparison is not token-identical.}"

RESULTS_TSV="$RAW_DIR/results.tsv"
MODELS_TSV="$RAW_DIR/models.tsv"
ENV_TSV="$RAW_DIR/environment.tsv"
THERMAL_TSV="$RAW_DIR/thermal.tsv"

log() { printf '%s\n' "$*"; }

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
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
  die "could not find $pattern below $MODEL_ROOT; set the corresponding *_MODEL variable"
}

slugify() {
  printf '%s' "$1" | tr -cs 'A-Za-z0-9._-' '_'
}

cooldown() {
  local remaining="$COOLDOWN_SECS"
  local chunk
  while [ "$remaining" -gt 0 ]; do
    if [ "$remaining" -gt 60 ]; then chunk=60; else chunk="$remaining"; fi
    sleep "$chunk"
    remaining=$((remaining - chunk))
  done
}

capture_thermal() {
  local label="$1"
  local engine="$2"
  local run="$3"
  local phase="$4"
  local state="unavailable"
  if command -v pmset >/dev/null 2>&1; then
    state=$(pmset -g therm 2>&1 || true)
    state=${state//$'\n'/; }
    state=${state//$'\t'/ }
    [ -n "$state" ] || state="unavailable"
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$label" "$engine" "$run" "$phase" "$state" \
    >> "$THERMAL_TSV"
}

append_result() {
  local IFS=$'\t'
  printf '%s\n' "$*" >> "$RESULTS_TSV"
}

run_rusty() {
  local label="$1"
  local model="$2"
  local run="$3"
  local order="$4"
  local slug raw json prompt_rate decode_rate prompt_tokens generated_tokens
  slug=$(slugify "$label")
  raw="$RAW_DIR/${slug}.rustyllm.run${run}.log"
  json="$RAW_DIR/${slug}.rustyllm.run${run}.json"

  log "[$label][$run/$RUNS][$order/2] RustyLLM"
  capture_thermal "$label" "RustyLLM" "$run" "before"
  RUSTY_LLM_METAL=1 \
    "$RUSTY_BIN" \
      --model "$model" \
      --profile auto \
      --bench --bench-json --bench-runs 1 \
      --max-tokens "$MAX_TOKENS" \
      --temp 0 --repeat-penalty 1 --seed "$SEED" \
      --prompt "$PROMPT" > "$raw" 2>&1
  capture_thermal "$label" "RustyLLM" "$run" "after"

  sed -n '/^{/,$p' "$raw" > "$json"
  jq -e '.type == "rusty-llm.benchmark"' "$json" >/dev/null \
    || die "RustyLLM did not emit valid benchmark JSON; see $raw"
  prompt_rate=$(jq -r '.results[0].prefill_tok_s' "$json")
  decode_rate=$(jq -r '.results[0].decode_tok_s' "$json")
  prompt_tokens=$(jq -r '.results[0].prompt_tokens' "$json")
  generated_tokens=$(jq -r '.results[0].generated_tokens' "$json")
  [ "$generated_tokens" = "$MAX_TOKENS" ] \
    || die "RustyLLM generated $generated_tokens tokens instead of $MAX_TOKENS; see $raw"
  append_result "$label" "RustyLLM" "$run" "$order" "$prompt_rate" "$decode_rate" "$prompt_tokens" "$generated_tokens" "$raw"
  log "  decode: $(LC_ALL=C awk -v v="$decode_rate" 'BEGIN { printf "%.2f", v }') tok/s"
}

run_reference() {
  local label="$1"
  local model="$2"
  local run="$3"
  local order="$4"
  local slug raw json prompt_rate decode_rate
  slug=$(slugify "$label")
  raw="$RAW_DIR/${slug}.reference.run${run}.log"

  log "[$label][$run/$RUNS][$order/2] Reference"
  capture_thermal "$label" "Reference" "$run" "before"
  if [ "$REFERENCE_KIND" = "bench" ]; then
    json="$RAW_DIR/${slug}.reference.run${run}.json"
    "$REFERENCE_BIN" \
      -m "$model" \
      -p "$REFERENCE_PROMPT_TOKENS" \
      -n "$MAX_TOKENS" -r 1 -o json \
      -t 8 -ngl 99 -fa on > "$raw" 2>&1
  else
    "$REFERENCE_BIN" \
      -m "$model" \
      -p "$PROMPT" \
      -n "$MAX_TOKENS" \
      --temp 0 --repeat-penalty 1 --seed "$SEED" --ignore-eos \
      --no-display-prompt --no-warmup --single-turn --simple-io --color off \
      --reasoning off --flash-attn on -ngl 99 > "$raw" 2>&1
  fi
  capture_thermal "$label" "Reference" "$run" "after"

  if [ "$REFERENCE_KIND" = "bench" ]; then
    sed -n '/^[[]/,$p' "$raw" > "$json"
    jq -e 'type == "array"' "$json" >/dev/null \
      || die "could not parse reference benchmark JSON; see $raw"
    prompt_rate=$(jq -r '[.[] | select(.n_prompt > 0)] | last | .avg_ts // empty' "$json")
    decode_rate=$(jq -r --argjson tokens "$MAX_TOKENS" \
      '[.[] | select(.n_gen == $tokens)] | last | .avg_ts // empty' "$json")
  else
    prompt_rate=$(sed -nE 's/.*Prompt: *([0-9]+([.][0-9]+)?) t\/s.*/\1/p' "$raw" | tail -n 1)
    decode_rate=$(sed -nE 's/.*Generation: *([0-9]+([.][0-9]+)?) t\/s.*/\1/p' "$raw" | tail -n 1)
  fi
  [ -n "$prompt_rate" ] && [ -n "$decode_rate" ] \
    || die "could not parse reference timing line; see $raw"
  append_result "$label" "Reference" "$run" "$order" "$prompt_rate" "$decode_rate" \
    "$REFERENCE_PROMPT_TOKENS" "$MAX_TOKENS" "$raw"
  log "  decode: $(LC_ALL=C awk -v v="$decode_rate" 'BEGIN { printf "%.2f", v }') tok/s"
}

mean_for() {
  LC_ALL=C awk -F '\t' -v label="$1" -v engine="$2" -v column="$3" '
    $1 == label && $2 == engine { sum += $column; count++ }
    END { if (count) printf "%.4f", sum / count }
  ' "$RESULTS_TSV"
}

sd_for() {
  LC_ALL=C awk -F '\t' -v label="$1" -v engine="$2" -v column="$3" '
    $1 == label && $2 == engine { values[++count] = $column; sum += $column }
    END {
      if (!count) exit
      mean = sum / count
      for (i = 1; i <= count; i++) squared += (values[i] - mean) ^ 2
      printf "%.4f", sqrt(squared / count)
    }
  ' "$RESULTS_TSV"
}

median_for() {
  LC_ALL=C awk -F '\t' -v label="$1" -v engine="$2" -v column="$3" '
    $1 == label && $2 == engine { values[++count] = $column }
    END {
      if (!count) exit
      for (i = 2; i <= count; i++) {
        value = values[i]
        j = i - 1
        while (j >= 1 && values[j] > value) {
          values[j + 1] = values[j]
          j--
        }
        values[j + 1] = value
      }
      if (count % 2) printf "%.4f", values[(count + 1) / 2]
      else printf "%.4f", (values[count / 2] + values[count / 2 + 1]) / 2
    }
  ' "$RESULTS_TSV"
}

environment_value() {
  awk -F '\t' -v key="$1" '$1 == key { sub(/^[^\t]*\t/, ""); print; exit }' "$ENV_TSV"
}

reference_version_value() {
  local json version
  if [ "$REFERENCE_KIND" = "bench" ]; then
    while IFS= read -r json; do
      version=$(jq -r '
        if type == "array" and length > 0
        then "build " + (.[0].build_number | tostring) + ", commit " + .[0].build_commit
        else empty
        end
      ' "$json" 2>/dev/null || true)
      if [ -n "$version" ]; then
        printf '%s\n' "$version"
        return 0
      fi
    done < <(find "$RAW_DIR" -type f -name '*.reference.run*.json' -print 2>/dev/null | sort)
    printf 'build metadata recorded in per-run JSON\n'
  else
    "$REFERENCE_BIN" --version 2>&1 | sed -n '1p'
  fi
}

write_environment() {
  local host cpu os memory_bytes rust_revision dirty reference_version
  host=$(hostname 2>/dev/null || printf 'unknown')
  cpu=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || uname -m)
  os=$(sw_vers -productVersion 2>/dev/null || uname -sr)
  if command -v sw_vers >/dev/null 2>&1; then
    os="macOS $os"
  fi
  memory_bytes=$(sysctl -n hw.memsize 2>/dev/null || awk '/MemTotal:/ { print $2 * 1024; exit }' /proc/meminfo 2>/dev/null || printf '0')
  rust_revision=$(git rev-parse --short HEAD 2>/dev/null || printf 'unknown')
  dirty=""
  if ! git diff --quiet --ignore-submodules -- 2>/dev/null || [ -n "$(git ls-files --others --exclude-standard 2>/dev/null)" ]; then
    dirty=" + working tree changes"
  fi
  reference_version=$(reference_version_value)
  {
    printf 'generated_at\t%s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')"
    printf 'host\t%s\n' "$host"
    printf 'cpu\t%s\n' "$cpu"
    printf 'os\t%s\n' "$os"
    printf 'memory_bytes\t%s\n' "$memory_bytes"
    printf 'rust_revision\t%s\n' "$rust_revision"
    printf 'dirty\t%s\n' "$dirty"
    printf 'reference_version\t%s\n' "$reference_version"
    printf 'reference_kind\t%s\n' "$REFERENCE_KIND"
    printf 'reference_prompt_tokens\t%s\n' "$REFERENCE_PROMPT_TOKENS"
    printf 'runs\t%s\n' "$RUNS"
    printf 'max_tokens\t%s\n' "$MAX_TOKENS"
    printf 'cooldown_secs\t%s\n' "$COOLDOWN_SECS"
    printf 'seed\t%s\n' "$SEED"
    printf 'prompt\t%s\n' "$PROMPT"
    printf 'report_title\t%s\n' "$REPORT_TITLE"
    printf 'protocol_note\t%s\n' "$PROTOCOL_NOTE"
  } > "$ENV_TSV"
}

write_report() {
  local generated_at host cpu os memory_bytes memory_gib rust_revision dirty reference_version
  local benchmark_runs benchmark_max_tokens benchmark_cooldown benchmark_seed benchmark_prompt
  local report_title protocol_note stored_report_title stored_protocol_note model_count
  local tmp label rusty_mean rusty_sd reference_mean reference_sd delta delta_value winner
  local rusty_cell reference_cell rusty_prefill reference_prefill geomean_delta run engine size
  local rusty_median reference_median median_delta
  generated_at=$(date '+%Y-%m-%d %H:%M:%S %Z')
  host=$(hostname 2>/dev/null || printf 'unknown')
  cpu=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || uname -m)
  os=$(sw_vers -productVersion 2>/dev/null || uname -sr)
  memory_bytes=$(sysctl -n hw.memsize 2>/dev/null || printf '0')
  memory_gib=$(LC_ALL=C awk -v bytes="$memory_bytes" 'BEGIN { printf "%.0f", bytes / 1073741824 }')
  rust_revision=$(git rev-parse --short HEAD 2>/dev/null || printf 'unknown')
  dirty=""
  if ! git diff --quiet --ignore-submodules -- 2>/dev/null || [ -n "$(git ls-files --others --exclude-standard 2>/dev/null)" ]; then
    dirty=" + working tree changes"
  fi
  reference_version=$(reference_version_value)
  benchmark_runs="$RUNS"
  benchmark_max_tokens="$MAX_TOKENS"
  benchmark_cooldown="$COOLDOWN_SECS"
  benchmark_seed="$SEED"
  benchmark_prompt="$PROMPT"
  report_title="$REPORT_TITLE"
  protocol_note="$PROTOCOL_NOTE"
  if [ -s "$ENV_TSV" ]; then
    generated_at=$(environment_value generated_at)
    host=$(environment_value host)
    cpu=$(environment_value cpu)
    os=$(environment_value os)
    memory_bytes=$(environment_value memory_bytes)
    rust_revision=$(environment_value rust_revision)
    dirty=$(environment_value dirty)
    reference_version=$(environment_value reference_version)
    benchmark_runs=$(environment_value runs)
    benchmark_max_tokens=$(environment_value max_tokens)
    benchmark_cooldown=$(environment_value cooldown_secs)
    benchmark_seed=$(environment_value seed)
    benchmark_prompt=$(environment_value prompt)
    stored_report_title=$(environment_value report_title)
    stored_protocol_note=$(environment_value protocol_note)
    [ -z "$stored_report_title" ] || report_title="$stored_report_title"
    [ -z "$stored_protocol_note" ] || protocol_note="$stored_protocol_note"
    memory_gib=$(LC_ALL=C awk -v bytes="$memory_bytes" 'BEGIN { printf "%.0f", bytes / 1073741824 }')
  fi
  if [ "$reference_version" = "build metadata recorded in per-run JSON" ]; then
    reference_version=$(reference_version_value)
  fi
  model_count=$(awk -F '\t' 'NR > 1 { count++ } END { print count + 0 }' "$MODELS_TSV")
  geomean_delta=$(LC_ALL=C awk -F '\t' '
    NR > 1 {
      key = $1 SUBSEP $2
      sum[key] += $6
      count[key]++
      labels[$1] = 1
    }
    END {
      for (label in labels) {
        rusty = sum[label SUBSEP "RustyLLM"] / count[label SUBSEP "RustyLLM"]
        reference = sum[label SUBSEP "Reference"] / count[label SUBSEP "Reference"]
        logs += log(rusty / reference)
        models++
      }
      printf "%+.1f", 100 * (exp(logs / models) - 1)
    }
  ' "$RESULTS_TSV")
  tmp="${REPORT}.tmp"

  {
    printf '# %s\n\n' "$report_title"
    printf 'Generated: **%s**\n\n' "$generated_at"
    printf 'Primary metric: end-to-end greedy decode throughput reported by each engine. '
    printf 'Higher is better. Values are arithmetic means across %s fresh-process runs. ' "$benchmark_runs"
    printf 'Across the %s tested models, RustyLLM differs by **%s%%** on the geometric mean.\n\n' \
      "$model_count" "$geomean_delta"
    printf '## Result\n\n'
    printf '| Model | RustyLLM | Reference | RustyLLM difference | Verdict |\n'
    printf '|---|---:|---:|---:|---|\n'
    while IFS=$'\t' read -r label path bytes; do
      [ "$label" = "model" ] && continue
      rusty_mean=$(mean_for "$label" "RustyLLM" 6)
      rusty_sd=$(sd_for "$label" "RustyLLM" 6)
      reference_mean=$(mean_for "$label" "Reference" 6)
      reference_sd=$(sd_for "$label" "Reference" 6)
      delta_value=$(LC_ALL=C awk -v rusty="$rusty_mean" -v reference="$reference_mean" 'BEGIN { print 100 * (rusty / reference - 1) }')
      delta=$(LC_ALL=C awk -v value="$delta_value" 'BEGIN { printf "%+.1f", value }')
      winner=$(LC_ALL=C awk -v value="$delta_value" 'BEGIN {
        if (value >= 2) print "RustyLLM"
        else if (value <= -2) print "Reference"
        else print "measurement parity (±2%)"
      }')
      rusty_cell=$(LC_ALL=C awk -v mean="$rusty_mean" -v sd="$rusty_sd" 'BEGIN { printf "%.2f ± %.2f tok/s", mean, sd }')
      reference_cell=$(LC_ALL=C awk -v mean="$reference_mean" -v sd="$reference_sd" 'BEGIN { printf "%.2f ± %.2f tok/s", mean, sd }')
      if LC_ALL=C awk -v value="$delta_value" 'BEGIN { exit !(value >= 2) }'; then
        rusty_cell="**$rusty_cell**"
      elif LC_ALL=C awk -v value="$delta_value" 'BEGIN { exit !(value <= -2) }'; then
        reference_cell="**$reference_cell**"
      fi
      printf '| %s | %s | %s | **%s%%** | %s |\n' \
        "$label" "$rusty_cell" "$reference_cell" "$delta" "$winner"
    done < "$MODELS_TSV"

    printf '\n## Robustness check\n\n'
    printf 'Medians reduce the influence of isolated slow runs. The median comparison leads to the same winner for every model.\n\n'
    printf '| Model | RustyLLM median | Reference median | RustyLLM difference |\n'
    printf '|---|---:|---:|---:|\n'
    while IFS=$'\t' read -r label path bytes; do
      [ "$label" = "model" ] && continue
      rusty_median=$(median_for "$label" "RustyLLM" 6)
      reference_median=$(median_for "$label" "Reference" 6)
      median_delta=$(LC_ALL=C awk -v rusty="$rusty_median" -v reference="$reference_median" \
        'BEGIN { printf "%+.1f", 100 * (rusty / reference - 1) }')
      printf '| %s | %.2f tok/s | %.2f tok/s | **%s%%** |\n' \
        "$label" "$rusty_median" "$reference_median" "$median_delta"
    done < "$MODELS_TSV"

    printf '\n## Per-run decode throughput\n\n'
    printf '| Model | Engine'
    run=1
    while [ "$run" -le "$benchmark_runs" ]; do
      printf ' | Run %s' "$run"
      run=$((run + 1))
    done
    printf ' |\n|---|---'
    run=1
    while [ "$run" -le "$benchmark_runs" ]; do
      printf '|---:'
      run=$((run + 1))
    done
    printf '|\n'
    while IFS=$'\t' read -r label path bytes; do
      [ "$label" = "model" ] && continue
      for engine in "RustyLLM" "Reference"; do
        printf '| %s | %s' "$label" "$engine"
        awk -F '\t' -v label="$label" -v engine="$engine" '
          $1 == label && $2 == engine { printf " | %.2f", $6 }
        ' "$RESULTS_TSV"
        printf ' |\n'
      done
    done < "$MODELS_TSV"

    printf '\n## Prefill reference\n\n'
    printf 'These short-prompt rates are secondary: chat-template token counts can differ, '
    printf 'and the first prefill also initializes per-process GPU resources.\n\n'
    printf '| Model | RustyLLM | Reference |\n'
    printf '|---|---:|---:|\n'
    while IFS=$'\t' read -r label path bytes; do
      [ "$label" = "model" ] && continue
      rusty_prefill=$(mean_for "$label" "RustyLLM" 5)
      reference_prefill=$(mean_for "$label" "Reference" 5)
      printf '| %s | %.2f tok/s | %.2f tok/s |\n' "$label" "$rusty_prefill" "$reference_prefill"
    done < "$MODELS_TSV"

    printf '\n## Protocol\n\n'
    printf -- '- Same GGUF file per model and engine; no format conversion.\n'
    printf -- '- Prompt: `%s`\n' "$benchmark_prompt"
    printf -- '- Greedy sampling: temperature 0, repeat penalty 1, seed %s.\n' "$benchmark_seed"
    printf -- '- Reasoning/thinking mode disabled in both engines.\n'
    printf -- '- Exactly %s generated tokens per run; %s runs per engine and model.\n' "$benchmark_max_tokens" "$benchmark_runs"
    printf -- '- Fresh process for every run; startup/model-load time excluded from the engine-reported token rate.\n'
    printf -- '- RustyLLM: release-max, native CPU, Metal with automatic attention selection.\n'
    if [ "$REFERENCE_KIND" = "bench" ]; then
      printf -- '- Reference: dedicated decode benchmark, %s prompt tokens, all layers on the GPU, fused attention on, one built-in warm-up, and one measured repetition per fresh process.\n' "$REFERENCE_PROMPT_TOKENS"
    else
      printf -- '- Reference: all layers on the GPU, fused attention on, built-in warm-up off, and EOS ignored for a fixed measurement window.\n'
    fi
    printf -- '- Engine order alternates on every run; cooldown is %s seconds between processes.\n' "$benchmark_cooldown"
    printf -- '- Thermal/performance-pressure telemetry is captured before and after every process when the host exposes it.\n'
    printf -- '- A result inside ±2%% is classified as measurement parity.\n'
    if [ -n "$protocol_note" ]; then
      printf -- '- %s\n' "$protocol_note"
    fi

    printf '\n## Environment\n\n'
    printf '| Setting | Value |\n'
    printf '|---|---|\n'
    printf '| Host | `%s` |\n' "$host"
    printf '| CPU | %s |\n' "$cpu"
    printf '| RAM | %s GiB |\n' "$memory_gib"
    printf '| OS | %s |\n' "$os"
    printf '| RustyLLM revision | `%s%s` |\n' "$rust_revision" "$dirty"
    printf '| Reference version | `%s` |\n' "$reference_version"
    printf '| Raw results | `%s` |\n' "$RESULTS_TSV"
    if [ -s "$THERMAL_TSV" ]; then
      printf '| Thermal log | `%s` |\n' "$THERMAL_TSV"
    fi

    printf '\n## Models\n\n'
    printf '| Model | File | Size |\n'
    printf '|---|---|---:|\n'
    while IFS=$'\t' read -r label path bytes; do
      [ "$label" = "model" ] && continue
      size=$(LC_ALL=C awk -v bytes="$bytes" 'BEGIN { printf "%.2f GiB", bytes / 1073741824 }')
      printf '| %s | `%s` | %s |\n' "$label" "$(basename "$path")" "$size"
    done < "$MODELS_TSV"
  } > "$tmp"
  mv "$tmp" "$REPORT"
}

case "$RUNS" in
  ''|*[!0-9]*) die "RUNS must be a positive integer" ;;
  0) die "RUNS must be greater than zero" ;;
esac
case "$MAX_TOKENS" in
  ''|*[!0-9]*) die "MAX_TOKENS must be a positive integer" ;;
  0) die "MAX_TOKENS must be greater than zero" ;;
esac
case "$REFERENCE_KIND" in
  cli|bench) ;;
  *) die "REFERENCE_KIND must be cli or bench" ;;
esac
case "$REFERENCE_PROMPT_TOKENS" in
  ''|*[!0-9]*|0) die "REFERENCE_PROMPT_TOKENS must be a positive integer" ;;
esac

require_cmd jq
require_cmd awk
require_cmd sed
[ -x "$RUSTY_BIN" ] || die "RustyLLM binary is missing: $RUSTY_BIN (run 'make release-max')"
[ -n "$REFERENCE_BIN" ] && [ -x "$REFERENCE_BIN" ] \
  || die "reference executable is missing; set REFERENCE_BIN=/path/to/reference-cli"

if [ "$REPORT_ONLY" = "1" ]; then
  [ -s "$RESULTS_TSV" ] || die "raw results are missing: $RESULTS_TSV"
  [ -s "$MODELS_TSV" ] || die "model inventory is missing: $MODELS_TSV"
  write_report
  log "Wrote $REPORT from $RESULTS_TSV"
  exit 0
fi

mkdir -p "$RAW_DIR"
write_environment
printf 'model\tengine\trun\torder\tprefill_tok_s\tdecode_tok_s\tprompt_tokens\tgenerated_tokens\traw_log\n' > "$RESULTS_TSV"
printf 'model\tpath\tbytes\n' > "$MODELS_TSV"
printf 'timestamp\tmodel\tengine\trun\tphase\tstate\n' > "$THERMAL_TSV"
if [ -n "$MODEL_SPECS" ]; then
  while IFS= read -r spec; do
    [ -n "$spec" ] || continue
    label=${spec%%|*}
    path=${spec#*|}
    [ "$path" != "$spec" ] || die "invalid MODEL_SPECS row: $spec"
    [ -n "$label" ] || die "MODEL_SPECS labels must not be empty"
    [ -f "$path" ] || die "model does not exist: $path"
    bytes=$(stat -f '%z' "$path" 2>/dev/null || stat -c '%s' "$path")
    printf '%s\t%s\t%s\n' "$label" "$path" "$bytes" >> "$MODELS_TSV"
  done <<< "$MODEL_SPECS"
else
  ministral_path=$(resolve_model "$MINISTRAL_MODEL" 'Ministral-3-3B-Instruct-2512-Q4_K_M.gguf')
  llama_path=$(resolve_model "$LLAMA_MODEL" 'Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf')
  qwen_path=$(resolve_model "$QWEN_MODEL" 'DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf')
  for spec in \
    "Ministral 3 3B Q4_K_M|$ministral_path" \
    "Meta Llama 3.1 8B Q4_K_M|$llama_path" \
    "Qwen2-family 7B Q4_K_M|$qwen_path"; do
    label=${spec%%|*}
    path=${spec#*|}
    bytes=$(stat -f '%z' "$path" 2>/dev/null || stat -c '%s' "$path")
    printf '%s\t%s\t%s\n' "$label" "$path" "$bytes" >> "$MODELS_TSV"
  done
fi
[ "$(awk 'END { print NR }' "$MODELS_TSV")" -gt 1 ] || die "MODEL_SPECS did not contain any models"

while IFS=$'\t' read -r label path bytes; do
  [ "$label" = "model" ] && continue
  run=1
  while [ "$run" -le "$RUNS" ]; do
    if [ $((run % 2)) -eq 1 ]; then
      run_rusty "$label" "$path" "$run" 1
      cooldown
      run_reference "$label" "$path" "$run" 2
    else
      run_reference "$label" "$path" "$run" 1
      cooldown
      run_rusty "$label" "$path" "$run" 2
    fi
    if [ "$run" -lt "$RUNS" ]; then
      cooldown
    fi
    run=$((run + 1))
  done
done < "$MODELS_TSV"

write_report
log "Wrote $REPORT"
log "Raw results: $RESULTS_TSV"
