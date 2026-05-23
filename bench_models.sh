#!/usr/bin/env bash
# RustyLLM — sequential model compatibility & speed benchmark
#
# Usage:
#   ./bench_models.sh                   # scan ~/.cache/lm-studio/models
#   MODEL_DIR=/path/to/models ./bench_models.sh
#
# Output:
#   BENCHMARK.md          updated in-place (created if absent)
#   README.md             a link line is added once if not already present
#   raw JSON per model    stored next to this script as bench_raw_<name>.json
#
set -euo pipefail

BINARY="${BINARY:-./target/release/rusty-llm}"
MODEL_DIR="${MODEL_DIR:-$HOME/.cache/lm-studio/models}"
WAIT_SECS="${WAIT_SECS:-8}"
BENCH_RUNS="${BENCH_RUNS:-2}"
MAX_TOKENS="${MAX_TOKENS:-64}"
TIMEOUT_SECS="${TIMEOUT_SECS:-600}"
README="${README:-README.md}"
BENCHMARK_MD="${BENCHMARK_MD:-BENCHMARK.md}"
RAW_DIR="${RAW_DIR:-.bench_raw}"

mkdir -p "$RAW_DIR"

# ── helpers ───────────────────────────────────────────────────────────────────
log() { printf '%s\n' "$1"; }
extract_json() {
  # Extract the outermost JSON object from mixed stdout/stderr output
  python3 -c "
import sys, json
raw = sys.stdin.read()
start = raw.find('{')
if start < 0: print('{}'); sys.exit(0)
depth = 0
for i, c in enumerate(raw[start:], start):
    if c == '{': depth += 1
    elif c == '}':
        depth -= 1
        if depth == 0: print(raw[start:i+1]); sys.exit(0)
print('{}')
" 2>/dev/null || echo "{}"
}

# ── collect models ─────────────────────────────────────────────────────────────
mapfile -t MODELS < <(find "$MODEL_DIR" -name "*.gguf" | grep -iv mmproj | sort)
TOTAL=${#MODELS[@]}

log "RustyLLM Model Benchmark — $(date)"
log "  Binary : $BINARY"
log "  Models : $TOTAL  (from $MODEL_DIR)"
log "  Runs   : ${BENCH_RUNS} × ${MAX_TOKENS} tokens each"
log "  Pause  : ${WAIT_SECS}s between models"
log "================================================================"

# ── per-model loop ─────────────────────────────────────────────────────────────
# Results stored as parallel arrays (bash 3 compat via indexed arrays)
declare -a R_FILE R_NAME R_ARCH R_STATUS R_MB R_LOAD R_DECODE R_PREFILL R_NOTE

idx=0
for MODEL_PATH in "${MODELS[@]}"; do
  MODEL_FILE=$(basename "$MODEL_PATH")
  SLUG="${MODEL_FILE%.gguf}"
  log ""
  log "[$((idx+1))/$TOTAL] ── $MODEL_FILE"

  # ── inspect ──────────────────────────────────────────────────────────────
  # Run separately from extract_json so a non-zero exit doesn't abort via pipefail
  INSPECT_RAW=$(timeout 30 "$BINARY" "$MODEL_PATH" --inspect 2>&1) || true
  INSPECT_JSON=$(printf '%s' "$INSPECT_RAW" | extract_json)
  echo "$INSPECT_JSON" > "$RAW_DIR/${SLUG}_inspect.json"

  STATUS=$(echo "$INSPECT_JSON"  | jq -r '.status              // "unknown"' 2>/dev/null || echo "unknown")
  ARCH=$(echo "$INSPECT_JSON"    | jq -r '.model.architecture  // "unknown"' 2>/dev/null || echo "unknown")
  MODEL_NAME=$(echo "$INSPECT_JSON" | jq -r '.model.name       // ""'        2>/dev/null || echo "")
  FILE_MB=$(echo "$INSPECT_JSON" | jq -r '(.file_size_bytes // 0) / 1048576 | floor' 2>/dev/null || echo "0")
  UNSUPPORTED=$(echo "$INSPECT_JSON" | jq -r '.gguf.unsupported_tensor_count // 0' 2>/dev/null || echo "0")
  [ -z "$MODEL_NAME" ] && MODEL_NAME="$MODEL_FILE"

  log "  arch=$ARCH  status=$STATUS  size=${FILE_MB}MB  unsupported_tensors=$UNSUPPORTED"

  R_FILE[$idx]="$MODEL_FILE"
  R_NAME[$idx]="$MODEL_NAME"
  R_ARCH[$idx]="$ARCH"
  R_MB[$idx]="$FILE_MB"

  if [ "$STATUS" != "supported" ]; then
    log "  ⚠  Not supported — skipping benchmark"
    NOTE=$(echo "$INSPECT_JSON" | jq -r '
      if ((.gguf.unsupported_layouts // []) | length) > 0 then
        (.gguf.unsupported_layouts[0])
      elif ((.gguf.missing_tensor_examples // []) | length) > 0 then
        "missing tensor: " + (.gguf.missing_tensor_examples[0])
      elif ((.gguf.unsupported_tensor_examples // []) | length) > 0 then
        "unsupported tensor: " + (.gguf.unsupported_tensor_examples[0])
      elif (.model.supported_architecture // false) == false then
        "unsupported architecture"
      else
        "not loadable"
      end
    ' 2>/dev/null || echo "not loadable")
    R_STATUS[$idx]="unsupported"
    R_LOAD[$idx]="—"
    R_DECODE[$idx]="—"
    R_PREFILL[$idx]="—"
    R_NOTE[$idx]="${NOTE:0:120}"
    idx=$((idx+1))
    log "  Waiting ${WAIT_SECS}s…"
    sleep "$WAIT_SECS"
    continue
  fi

  # ── benchmark ────────────────────────────────────────────────────────────
  log "  Running ${BENCH_RUNS} runs × ${MAX_TOKENS} tokens…"
  BENCH_RAW=$(timeout "$TIMEOUT_SECS" "$BINARY" "$MODEL_PATH" \
    --bench-json --bench-runs "$BENCH_RUNS" --max-tokens "$MAX_TOKENS" 2>&1) \
    && BENCH_EXIT=0 || BENCH_EXIT=$?

  BENCH_JSON=$(echo "$BENCH_RAW" | extract_json)
  echo "$BENCH_JSON" > "$RAW_DIR/${SLUG}_bench.json"

  if [ "$BENCH_EXIT" -ne 0 ] || [ "$BENCH_JSON" = "{}" ]; then
    ERR=$(echo "$BENCH_RAW" | grep -i "error\|panic\|failed\|unsupported" | head -1 \
          | tr '\t' ' ' || echo "bench failed (exit $BENCH_EXIT)")
    log "  ✗  Benchmark failed: $ERR"
    R_STATUS[$idx]="bench_failed"
    R_LOAD[$idx]="—"
    R_DECODE[$idx]="—"
    R_PREFILL[$idx]="—"
    R_NOTE[$idx]="${ERR:0:80}"
  else
    LOAD_MS=$(echo "$BENCH_JSON"  | jq -r '.load_ms                          // 0' 2>/dev/null || echo "0")
    DECODE=$(echo "$BENCH_JSON"   | jq -r '.summary.aggregate_decode_tok_s   // 0' 2>/dev/null || echo "0")
    PREFILL=$(echo "$BENCH_JSON"  | jq -r '.summary.aggregate_prefill_tok_s  // 0' 2>/dev/null || echo "0")
    DF=$(printf "%.1f" "$DECODE"  2>/dev/null || echo "$DECODE")
    PF=$(printf "%.1f" "$PREFILL" 2>/dev/null || echo "$PREFILL")
    log "  ✓  load=${LOAD_MS}ms  decode=${DF} tok/s  prefill=${PF} tok/s"
    R_STATUS[$idx]="supported"
    R_LOAD[$idx]="$LOAD_MS"
    R_DECODE[$idx]="$DECODE"
    R_PREFILL[$idx]="$PREFILL"
    R_NOTE[$idx]=""
  fi

  idx=$((idx+1))
  log "  Waiting ${WAIT_SECS}s to free memory…"
  sleep "$WAIT_SECS"
done

# ── system info ───────────────────────────────────────────────────────────────
CPU=$(sysctl -n machdep.cpu.brand_string 2>/dev/null \
      || grep -m1 "^model name" /proc/cpuinfo 2>/dev/null | cut -d: -f2 | xargs \
      || uname -m)
CORES=$(sysctl -n hw.logicalcpu 2>/dev/null || nproc 2>/dev/null || echo "?")
RAM_GB=$(( $(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1073741824 ))
OS_NAME=$(sw_vers -productName 2>/dev/null || uname -s)
OS_VER=$(sw_vers -productVersion 2>/dev/null || uname -r)
RUST_VER=$(rustc --version 2>/dev/null || echo "unknown")
# Detect SIMD and Metal from a quick probe run (inspect is cheap — no weights loaded)
PROBE=$("$BINARY" --help 2>&1 | head -5 || true)
SIMD=$(echo "$PROBE" | grep -o "SIMD:.*" | head -1 | xargs || echo "unknown")
METAL_STATUS="disabled (RUSTY_LLM_METAL not set)"
if [ -n "${RUSTY_LLM_METAL:-}" ]; then
  METAL_STATUS="enabled (RUSTY_LLM_METAL=1)"
fi
RUN_DATE=$(date "+%Y-%m-%d %H:%M %Z")

# ── build BENCHMARK.md ────────────────────────────────────────────────────────
log ""
log "Writing $BENCHMARK_MD …"

{
cat <<HEADER
# RustyLLM Benchmark Results

> **Prompt:** *"Explain local LLM inference performance in one concise paragraph."*
> **Runs:** ${BENCH_RUNS} runs × ${MAX_TOKENS} tokens per model, CPU-only (no GPU)
> **Date:** ${RUN_DATE}

## Hardware

| | |
|---|---|
| **CPU** | ${CPU} |
| **Logical cores** | ${CORES} |
| **RAM** | ${RAM_GB} GB |
| **OS** | ${OS_NAME} ${OS_VER} |
| **Rust** | ${RUST_VER} |
| **SIMD** | ${SIMD} |
| **Metal GPU** | ${METAL_STATUS} |

## Model Results

| # | Model file | Architecture | Compat | Size (MB) | Load (ms) | Decode (tok/s) | Prefill (tok/s) |
|--:|-----------|:---:|:---:|---:|---:|---:|---:|
HEADER

for ((i=0; i<idx; i++)); do
  case "${R_STATUS[$i]}" in
    supported)    ICON="✅" ;;
    bench_failed) ICON="⚠️" ;;
    *)            ICON="❌" ;;
  esac

  if [[ "${R_DECODE[$i]}" =~ ^[0-9] ]]; then
    DF=$(printf "%.1f" "${R_DECODE[$i]}" 2>/dev/null || echo "${R_DECODE[$i]}")
    PF=$(printf "%.1f" "${R_PREFILL[$i]}" 2>/dev/null || echo "${R_PREFILL[$i]}")
  else
    DF="${R_DECODE[$i]}"
    PF="${R_PREFILL[$i]}"
  fi

  NOTE=""
  [ -n "${R_NOTE[$i]:-}" ] && NOTE=" <!-- ${R_NOTE[$i]} -->"

  printf "| %d | \`%s\` | %s | %s | %s | %s | %s | %s |%s\n" \
    "$((i+1))" "${R_FILE[$i]}" "${R_ARCH[$i]}" "$ICON" \
    "${R_MB[$i]}" "${R_LOAD[$i]}" "$DF" "$PF" "$NOTE"
done

echo ""
echo "## Speed Ranking"
echo ""
echo "Decode throughput (tok/s), supported models only, fastest first."
echo ""
echo "| Rank | Model file | Architecture | Decode (tok/s) | Prefill (tok/s) | Size (MB) |"
echo "|:---:|-----------|:---:|---:|---:|---:|"

# Build temp file for sorting
TMP_RANK=$(mktemp)
for ((i=0; i<idx; i++)); do
  [[ "${R_DECODE[$i]}" =~ ^[0-9] ]] || continue
  printf "%s\t%s\t%s\t%s\t%s\n" \
    "${R_DECODE[$i]}" "${R_FILE[$i]}" "${R_ARCH[$i]}" "${R_PREFILL[$i]}" "${R_MB[$i]}" \
    >> "$TMP_RANK"
done
sort -t$'\t' -k1 -rn "$TMP_RANK" | awk -F'\t' '{
  rank++
  df = sprintf("%.1f", $1)
  pf = sprintf("%.1f", $4)
  printf "| %d | `%s` | %s | %s | %s | %s |\n", rank, $2, $3, df, pf, $5
}'
rm -f "$TMP_RANK"

cat <<FOOTER

---

*Re-run \`bench_models.sh\` to refresh. Raw JSON per model is stored in \`.bench_raw/\`.*
*Numbers reflect single-socket CPU inference; results vary with thermal state and OS load.*
*Ollama and LM Studio are typically faster due to GPU offload and llama.cpp kernels.*
FOOTER
} > "$BENCHMARK_MD"

log "✓  $BENCHMARK_MD written."

# ── ensure README.md has a link ───────────────────────────────────────────────
LINK_LINE="→ **[Benchmark results](BENCHMARK.md)** — compatibility and speed for all tested models."

if [ -f "$README" ]; then
  if grep -qF "BENCHMARK.md" "$README"; then
    log "✓  README.md already links to BENCHMARK.md — no change needed."
  else
    # Insert after the first paragraph (after the first blank line following content)
    python3 - "$README" "$LINK_LINE" <<'PYEOF'
import sys
path, link = sys.argv[1], sys.argv[2]
with open(path) as f:
    lines = f.readlines()

# Find first blank line that follows at least one non-blank, non-heading line
insert_at = len(lines)
saw_content = False
for i, line in enumerate(lines):
    stripped = line.strip()
    if stripped and not stripped.startswith('#') and not stripped.startswith('[!['):
        saw_content = True
    if saw_content and stripped == '':
        insert_at = i + 1
        break

lines.insert(insert_at, '\n')
lines.insert(insert_at, link + '\n')
lines.insert(insert_at, '\n')
with open(path, 'w') as f:
    f.writelines(lines)
PYEOF
    log "✓  README.md: link to BENCHMARK.md inserted."
  fi
else
  log "⚠  README.md not found — skipping link insertion."
fi

log ""
log "================================================================"
log "Done."
log "  BENCHMARK.md : $BENCHMARK_MD"
log "  Raw JSON     : $RAW_DIR/"
log "================================================================"
