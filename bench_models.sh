#!/usr/bin/env bash
# RustyLLM - model compatibility and backend speed benchmark.
#
# Usage:
#   ./bench_models.sh
#   BENCH_PROFILES=cpu ./bench_models.sh
#   BENCH_PROFILES="cpu metal" MODEL_DIR=/path/to/models ./bench_models.sh
#   RUSTY_LLM_MODEL_DIR=/path/to/models ./bench_models.sh
#   REPORT_ONLY=1 ./bench_models.sh
#   FIND_MODEL_DIR_ONLY=1 ./bench_models.sh
#   MODEL_FILTER=phi MODEL_LIMIT=2 MAX_TOKENS=16 WAIT_SECS=0 ./bench_models.sh
#
# Output:
#   BENCHMARK.md              readable benchmark report
#   .bench_raw/<profile>/     inspect/bench JSON for each model
#
set -euo pipefail

BINARY="${BINARY:-./target/release/rusty-llm}"
MODEL_DIR="${MODEL_DIR:-}"
BENCH_PROFILES="${BENCH_PROFILES:-cpu metal}"
MODEL_FILTER="${MODEL_FILTER:-}"
MODEL_LIMIT="${MODEL_LIMIT:-0}"
WAIT_SECS="${WAIT_SECS:-8}"
BENCH_RUNS="${BENCH_RUNS:-2}"
MAX_TOKENS="${MAX_TOKENS:-64}"
PROMPT="${PROMPT:-Explain local LLM inference performance in one concise paragraph.}"
TIMEOUT_SECS="${TIMEOUT_SECS:-600}"
README="${README:-README.md}"
BENCHMARK_MD="${BENCHMARK_MD:-BENCHMARK.md}"
RAW_DIR="${RAW_DIR:-.bench_raw}"
REPORT_ONLY="${REPORT_ONLY:-0}"
FIND_MODEL_DIR_ONLY="${FIND_MODEL_DIR_ONLY:-0}"

RESULTS_TSV="$RAW_DIR/results.tsv"
PROFILES_TSV="$RAW_DIR/profiles.tsv"

log() { printf '%s\n' "$1"; }

format_decimal() {
  LC_ALL=C awk -v value="$1" '
    BEGIN {
      if (value == "" || value == "null" || value == "—") {
        print value
      } else {
        printf "%.1f", value + 0
      }
    }
  '
}

is_gguf_file() {
  [ -f "$1" ] || return 1
  [ "$(dd if="$1" bs=4 count=1 2>/dev/null || true)" = "GGUF" ]
}

find_model_files() {
  [ -d "$1" ] || return 0
  find "$1" \( -iname "*.gguf" -o -path "*/blobs/sha256-*" \) -type f -print 2>/dev/null | sort
}

has_model_files() {
  [ -d "$1" ] || return 1
  local path
  while IFS= read -r path; do
    case "$(basename "$path")" in
      *mmproj*|*MMProj*|*MMPROJ*) continue ;;
    esac
    case "$path" in
      *.gguf|*.GGUF) return 0 ;;
      *) is_gguf_file "$path" && return 0 ;;
    esac
  done < <(find_model_files "$1")
  return 1
}

model_dir_candidates() {
  if [ -n "${RUSTY_LLM_MODEL_DIR:-}" ]; then
    printf '%s\n' "$RUSTY_LLM_MODEL_DIR"
  fi
  if [ -n "${OLLAMA_MODELS:-}" ]; then
    printf '%s\n' "$OLLAMA_MODELS"
  fi

  if [ -n "${HOME:-}" ]; then
    # LM Studio
    printf '%s\n' "$HOME/.cache/lm-studio/models/lmstudio-community"
    printf '%s\n' "$HOME/.lmstudio/models/lmstudio-community"
    printf '%s\n' "$HOME/Library/Application Support/LM Studio/models"
    printf '%s\n' "$HOME/.cache/lm-studio/models"
    printf '%s\n' "$HOME/.lmstudio/models"
    # Ollama stores GGUF payloads as content-addressed blobs.
    printf '%s\n' "$HOME/.ollama/models"
    # GPT4All / Nomic stores downloadable GGUF models in user cache folders.
    printf '%s\n' "$HOME/Library/Application Support/nomic.ai/GPT4All"
    printf '%s\n' "$HOME/.cache/nomic.ai/GPT4All"
    # Jan commonly keeps local model files under its app data/cache trees.
    printf '%s\n' "$HOME/jan/models"
    printf '%s\n' "$HOME/.cache/jan/models"
    printf '%s\n' "$HOME/Library/Application Support/Jan/models"
    # Plain project-local/user model folders are useful for source checkouts.
    printf '%s\n' "$HOME/models"
  fi
  if [ -n "${USERPROFILE:-}" ]; then
    printf '%s\n' "$USERPROFILE/.ollama/models"
  fi
  if [ -n "${LOCALAPPDATA:-}" ]; then
    printf '%s\n' "$LOCALAPPDATA/LM Studio/models"
    printf '%s\n' "$LOCALAPPDATA/Ollama/models"
    printf '%s\n' "$LOCALAPPDATA/nomic.ai/GPT4All"
    printf '%s\n' "$LOCALAPPDATA/Jan/models"
  fi
  printf '%s\n' "/usr/share/ollama/.ollama/models"
  printf '%s\n' "/usr/share/ollama/models"
  printf '%s\n' "/usr/local/share/ollama/.ollama/models"
  printf '%s\n' "/usr/local/share/ollama/models"
  printf '%s\n' "/var/lib/ollama/models"
  printf '%s\n' "/var/lib/ollama/.ollama/models"
  printf '%s\n' "./models"
}

resolve_model_dir() {
  if [ -n "$MODEL_DIR" ]; then
    if [ -d "$MODEL_DIR" ]; then
      printf '%s\n' "$MODEL_DIR"
      return 0
    fi
    printf 'error: MODEL_DIR does not exist: %s\n' "$MODEL_DIR" >&2
    return 1
  fi

  local candidate
  local seen=""
  while IFS= read -r candidate; do
    [ -n "$candidate" ] || continue
    case "$seen" in
      *"
$candidate
"*) continue ;;
    esac
    seen="${seen}
$candidate
"
    if has_model_files "$candidate"; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done < <(model_dir_candidates)

  printf 'error: no GGUF text models found in known model directories.\n' >&2
  printf '       Set MODEL_DIR=/path/to/models or RUSTY_LLM_MODEL_DIR=/path/to/models.\n' >&2
  printf '       Checked:\n' >&2
  while IFS= read -r candidate; do
    [ -n "$candidate" ] && printf '       - %s\n' "$candidate" >&2
  done < <(model_dir_candidates)
  return 1
}

model_dir_from_results() {
  [ -s "$RESULTS_TSV" ] || return 1
  awk -F '\t' 'NF >= 5 && $5 != "" { sub("/[^/]*$", "", $5); print $5; exit }' "$RESULTS_TSV"
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    printf 'error: required command not found: %s\n' "$1" >&2
    exit 1
  fi
}

clean_field() {
  printf '%s' "$1" | tr '\t\r\n' '   '
}

append_tsv() {
  local IFS=$'\t'
  printf '%s\n' "$*" >> "$RESULTS_TSV"
}

append_profile_tsv() {
  local IFS=$'\t'
  printf '%s\n' "$*" >> "$PROFILES_TSV"
}

slugify() {
  printf '%s' "${1%.gguf}" | tr -cs 'A-Za-z0-9._-' '_'
}

extract_json() {
  # Extract the outermost JSON object from mixed stdout/stderr output.
  python3 -c "
import sys
raw = sys.stdin.read()
start = raw.find('{')
if start < 0:
    print('{}')
    sys.exit(0)
depth = 0
in_string = False
escape = False
for i, c in enumerate(raw[start:], start):
    if in_string:
        if escape:
            escape = False
        elif c == '\\\\':
            escape = True
        elif c == '\"':
            in_string = False
        continue
    if c == '\"':
        in_string = True
    elif c == '{':
        depth += 1
    elif c == '}':
        depth -= 1
        if depth == 0:
            print(raw[start:i + 1])
            sys.exit(0)
print('{}')
" 2>/dev/null || echo "{}"
}

note_from_inspect() {
  jq -r '
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
  ' 2>/dev/null || echo "not loadable"
}

profile_config() {
  local key
  key=$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')
  case "$key" in
    cpu|off|0)
      printf 'cpu\tCPU\t0\n'
      ;;
    metal|gpu|on|1)
      printf 'metal\tMetal GPU\t1\n'
      ;;
    *)
      printf 'error: unknown BENCH_PROFILES entry: %s\n' "$1" >&2
      exit 1
      ;;
  esac
}

if [ "$FIND_MODEL_DIR_ONLY" = "1" ]; then
  resolve_model_dir
  exit 0
fi

require_cmd python3
require_cmd jq
require_cmd timeout

mkdir -p "$RAW_DIR"

if [ "$REPORT_ONLY" = "1" ] && [ -z "$MODEL_DIR" ]; then
  MODEL_DIR=$(resolve_model_dir 2>/dev/null || model_dir_from_results || true)
  [ -n "$MODEL_DIR" ] || MODEL_DIR="unknown"
else
  MODEL_DIR=$(resolve_model_dir)
fi

if [ ! -x "$BINARY" ]; then
  log "Binary not found or not executable: $BINARY"
  log "Building release binary first..."
  cargo build --release
fi

PROFILE_WORDS=${BENCH_PROFILES//,/ }
read -r -a PROFILE_INPUTS <<< "$PROFILE_WORDS"
RUN_DATE=$(date "+%Y-%m-%d %H:%M %Z")

if [ "$REPORT_ONLY" = "1" ]; then
  if [ ! -s "$RESULTS_TSV" ] || [ ! -s "$PROFILES_TSV" ]; then
    log "REPORT_ONLY=1 requires existing raw TSV files:"
    log "  $RESULTS_TSV"
    log "  $PROFILES_TSV"
    exit 1
  fi
  log "RustyLLM Model Benchmark - $(date)"
  log "  Binary   : $BINARY"
  log "  Models   : report-only (from $RESULTS_TSV)"
  log "  Profiles : report-only (from $PROFILES_TSV)"
  log "================================================================"
else
  : > "$RESULTS_TSV"
  : > "$PROFILES_TSV"

# Collect text GGUF models. Projector files are skipped because they are not
# standalone language models. Ollama blob stores are included when the blob
# itself has a GGUF file signature.
declare -a MODELS
MODELS=()
while IFS= read -r model_path; do
  model_file=$(basename "$model_path")
  case "$model_file" in
    *mmproj*|*MMProj*|*MMPROJ*) continue ;;
  esac
  case "$model_path" in
    *.gguf|*.GGUF) ;;
    *) is_gguf_file "$model_path" || continue ;;
  esac
  if [ -n "$MODEL_FILTER" ] && ! printf '%s\n' "$model_path" | grep -qi "$MODEL_FILTER"; then
    continue
  fi
  MODELS+=("$model_path")
  if [ "$MODEL_LIMIT" -gt 0 ] && [ "${#MODELS[@]}" -ge "$MODEL_LIMIT" ]; then
    break
  fi
done < <(find_model_files "$MODEL_DIR")

TOTAL=${#MODELS[@]}
if [ "$TOTAL" -eq 0 ]; then
  log "No GGUF text models found in $MODEL_DIR"
  exit 1
fi

log "RustyLLM Model Benchmark - $(date)"
log "  Binary   : $BINARY"
log "  Models   : $TOTAL  (from $MODEL_DIR)"
log "  Profiles : ${PROFILE_INPUTS[*]}"
log "  Runs     : ${BENCH_RUNS} x ${MAX_TOKENS} tokens each"
log "  Pause    : ${WAIT_SECS}s between models"
log "================================================================"

for profile_input in "${PROFILE_INPUTS[@]}"; do
  IFS=$'\t' read -r PROFILE_KEY PROFILE_LABEL METAL_ENV < <(profile_config "$profile_input")
  RAW_PROFILE_DIR="$RAW_DIR/$PROFILE_KEY"
  mkdir -p "$RAW_PROFILE_DIR"
  PROFILE_RUNTIME="not observed"

  log ""
  log "Profile: $PROFILE_LABEL (RUSTY_LLM_METAL=$METAL_ENV)"
  log "----------------------------------------------------------------"

  idx=0
  for MODEL_PATH in "${MODELS[@]}"; do
    MODEL_FILE=$(basename "$MODEL_PATH")
    RAW_BASE=$(printf '%02d_%s' "$((idx + 1))" "$(slugify "$MODEL_FILE")")
    log ""
    log "[$((idx + 1))/$TOTAL][$PROFILE_KEY] $MODEL_FILE"

    INSPECT_RAW=$(RUSTY_LLM_METAL="$METAL_ENV" timeout 30 "$BINARY" "$MODEL_PATH" --inspect 2>&1) || true
    INSPECT_JSON=$(printf '%s' "$INSPECT_RAW" | extract_json)
    printf '%s\n' "$INSPECT_JSON" > "$RAW_PROFILE_DIR/${RAW_BASE}_inspect.json"

    METAL_LINE=$(printf '%s\n' "$INSPECT_RAW" | grep -m1 '^Metal:' || true)
    if [ -n "$METAL_LINE" ]; then
      PROFILE_RUNTIME="$METAL_LINE"
    fi

    STATUS=$(printf '%s' "$INSPECT_JSON" | jq -r '.status // "unknown"' 2>/dev/null || echo "unknown")
    ARCH=$(printf '%s' "$INSPECT_JSON" | jq -r '.model.architecture // "unknown"' 2>/dev/null || echo "unknown")
    MODEL_NAME=$(printf '%s' "$INSPECT_JSON" | jq -r '.model.name // ""' 2>/dev/null || echo "")
    FILE_MB=$(printf '%s' "$INSPECT_JSON" | jq -r '(.file_size_bytes // 0) / 1048576 | floor' 2>/dev/null || echo "0")
    UNSUPPORTED=$(printf '%s' "$INSPECT_JSON" | jq -r '.gguf.unsupported_tensor_count // 0' 2>/dev/null || echo "0")
    [ -z "$MODEL_NAME" ] || [ "$MODEL_NAME" = "null" ] && MODEL_NAME="$MODEL_FILE"

    log "  arch=$ARCH  status=$STATUS  size=${FILE_MB}MB  unsupported_tensors=$UNSUPPORTED"

    if [ "$STATUS" != "supported" ]; then
      NOTE=$(printf '%s' "$INSPECT_JSON" | note_from_inspect)
      [ -z "$NOTE" ] || [ "$NOTE" = "null" ] && NOTE="not loadable"
      log "  skip: $NOTE"
      append_tsv \
        "$PROFILE_KEY" "$PROFILE_LABEL" "$METAL_ENV" "$((idx + 1))" \
        "$(clean_field "$MODEL_PATH")" "$(clean_field "$MODEL_FILE")" \
        "$(clean_field "$MODEL_NAME")" "$(clean_field "$ARCH")" \
        "$(clean_field "$STATUS")" "$FILE_MB" "—" "—" "—" "$(clean_field "$NOTE")"
      idx=$((idx + 1))
      log "  Waiting ${WAIT_SECS}s..."
      sleep "$WAIT_SECS"
      continue
    fi

    log "  Running ${BENCH_RUNS} runs x ${MAX_TOKENS} tokens..."
    BENCH_RAW=$(RUSTY_LLM_METAL="$METAL_ENV" timeout "$TIMEOUT_SECS" "$BINARY" "$MODEL_PATH" \
      --bench-json --bench-runs "$BENCH_RUNS" --max-tokens "$MAX_TOKENS" --prompt "$PROMPT" 2>&1) \
      && BENCH_EXIT=0 || BENCH_EXIT=$?

    BENCH_JSON=$(printf '%s' "$BENCH_RAW" | extract_json)
    printf '%s\n' "$BENCH_JSON" > "$RAW_PROFILE_DIR/${RAW_BASE}_bench.json"

    if [ "$BENCH_EXIT" -ne 0 ] || [ "$BENCH_JSON" = "{}" ]; then
      ERR=$(printf '%s\n' "$BENCH_RAW" | { grep -i "error\|panic\|failed\|unsupported" || true; } | head -1 | tr '\t' ' ')
      [ -z "$ERR" ] && ERR="bench failed (exit $BENCH_EXIT)"
      log "  failed: $ERR"
      append_tsv \
        "$PROFILE_KEY" "$PROFILE_LABEL" "$METAL_ENV" "$((idx + 1))" \
        "$(clean_field "$MODEL_PATH")" "$(clean_field "$MODEL_FILE")" \
        "$(clean_field "$MODEL_NAME")" "$(clean_field "$ARCH")" \
        "bench_failed" "$FILE_MB" "—" "—" "—" "$(clean_field "$ERR")"
    else
      LOAD_MS=$(printf '%s' "$BENCH_JSON" | jq -r '.load_ms // 0' 2>/dev/null || echo "0")
      DECODE=$(printf '%s' "$BENCH_JSON" | jq -r '.summary.aggregate_decode_tok_s // 0' 2>/dev/null || echo "0")
      PREFILL=$(printf '%s' "$BENCH_JSON" | jq -r '.summary.aggregate_prefill_tok_s // 0' 2>/dev/null || echo "0")
      DF=$(format_decimal "$DECODE")
      PF=$(format_decimal "$PREFILL")
      log "  ok: load=${LOAD_MS}ms  decode=${DF} tok/s  prefill=${PF} tok/s"
      append_tsv \
        "$PROFILE_KEY" "$PROFILE_LABEL" "$METAL_ENV" "$((idx + 1))" \
        "$(clean_field "$MODEL_PATH")" "$(clean_field "$MODEL_FILE")" \
        "$(clean_field "$MODEL_NAME")" "$(clean_field "$ARCH")" \
        "supported" "$FILE_MB" "$LOAD_MS" "$DECODE" "$PREFILL" ""
    fi

    idx=$((idx + 1))
    log "  Waiting ${WAIT_SECS}s to free memory..."
    sleep "$WAIT_SECS"
  done

  append_profile_tsv \
    "$PROFILE_KEY" "$PROFILE_LABEL" "$METAL_ENV" \
    "$(clean_field "$RAW_PROFILE_DIR")" "$(clean_field "$PROFILE_RUNTIME")"
done
fi

HW_INFO=$(system_profiler SPHardwareDataType 2>/dev/null || true)
CPU=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || true)
if [ -z "$CPU" ]; then
  CPU=$(printf '%s\n' "$HW_INFO" | awk -F': ' '/Chip:/ { print $2; exit }')
fi
if [ -z "$CPU" ]; then
  CPU=$(grep -m1 "^model name" /proc/cpuinfo 2>/dev/null | cut -d: -f2 | xargs || true)
fi
if [ -z "$CPU" ]; then
  case "$(uname -m)" in
    arm64|aarch64) CPU="Apple Silicon ($(uname -m))" ;;
    *) CPU="$(uname -m)" ;;
  esac
fi

CORES=$(sysctl -n hw.logicalcpu 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || nproc 2>/dev/null || echo "?")

RAM_BYTES=$(sysctl -n hw.memsize 2>/dev/null || echo 0)
if [ "$RAM_BYTES" -gt 0 ] 2>/dev/null; then
  RAM_DISPLAY="$(( RAM_BYTES / 1073741824 )) GB"
else
  RAM_DISPLAY=$(printf '%s\n' "$HW_INFO" | awk -F': ' '/Memory:/ { print $2; exit }')
  [ -z "$RAM_DISPLAY" ] && RAM_DISPLAY="unknown"
fi
OS_NAME=$(sw_vers -productName 2>/dev/null || uname -s)
OS_VER=$(sw_vers -productVersion 2>/dev/null || uname -r)
RUST_VER=$(rustc --version 2>/dev/null || echo "unknown")
MACHINE=$(uname -m)
case "$MACHINE" in
  arm64|aarch64) SIMD="ARM NEON (native)" ;;
  x86_64) SIMD="x86 runtime detection" ;;
  *) SIMD="runtime dependent" ;;
esac

log ""
log "Writing $BENCHMARK_MD ..."

RESULTS_TSV="$RESULTS_TSV" \
PROFILES_TSV="$PROFILES_TSV" \
MODEL_DIR="$MODEL_DIR" \
RUN_DATE="$RUN_DATE" \
BENCH_RUNS="$BENCH_RUNS" \
MAX_TOKENS="$MAX_TOKENS" \
WAIT_SECS="$WAIT_SECS" \
PROMPT="$PROMPT" \
CPU="$CPU" \
CORES="$CORES" \
RAM_DISPLAY="$RAM_DISPLAY" \
OS_NAME="$OS_NAME" \
OS_VER="$OS_VER" \
RUST_VER="$RUST_VER" \
SIMD="$SIMD" \
python3 <<'PYEOF' > "$BENCHMARK_MD"
import csv
import os
import statistics
from collections import defaultdict


def read_tsv(path, fields):
    rows = []
    with open(path, newline="") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if not row:
                continue
            row += [""] * (len(fields) - len(row))
            rows.append(dict(zip(fields, row)))
    return rows


def md(text):
    return str(text).replace("|", "\\|").replace("\n", " ").strip()


def code(text):
    return "`" + md(text).replace("`", "\\`") + "`"


def as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt_speed(value):
    number = as_float(value)
    if number is None:
        return "—"
    return f"{number:.1f}"


def fmt_int(value):
    number = as_float(value)
    if number is None:
        return "—"
    return str(int(number))


def status_text(status):
    if status == "supported":
        return "ok"
    if status == "bench_failed":
        return "failed"
    if status == "partially-supported":
        return "partial"
    return "skip"


def short_note(note, limit=92):
    note = md(note)
    if len(note) <= limit:
        return note
    return note[: limit - 1].rstrip() + "…"


result_fields = [
    "profile",
    "profile_label",
    "metal_env",
    "index",
    "path",
    "file",
    "name",
    "arch",
    "status",
    "mb",
    "load",
    "decode",
    "prefill",
    "note",
]
profile_fields = ["profile", "profile_label", "metal_env", "raw_dir", "runtime"]

results = read_tsv(os.environ["RESULTS_TSV"], result_fields)
profiles = read_tsv(os.environ["PROFILES_TSV"], profile_fields)

for row in results:
    row["index_num"] = int(row["index"] or 0)

profile_order = [row["profile"] for row in profiles]
rows_by_profile = defaultdict(list)
rows_by_key = {}
model_order = {}
for row in results:
    rows_by_profile[row["profile"]].append(row)
    rows_by_key[(row["profile"], row["path"])] = row
    model_order.setdefault(row["path"], row)

print("# RustyLLM Benchmark Results")
print()
print(f"Updated: **{md(os.environ['RUN_DATE'])}**")
print()
print(
    "This report compares the CPU path with the optional Apple Metal GPU path. "
    "Metal here means GPU acceleration through RustyLLM's Metal kernels; it is not a CoreML, ANE, or NPU backend."
)
print()
print("## Run Configuration")
print()
print("| Setting | Value |")
print("|---|---|")
print(f"| Model directory | {code(os.environ['MODEL_DIR'])} |")
print(f"| Prompt | {md(os.environ['PROMPT'])} |")
print(f"| Runs | {md(os.environ['BENCH_RUNS'])} x {md(os.environ['MAX_TOKENS'])} generated tokens per model |")
print(f"| Pause | {md(os.environ['WAIT_SECS'])} seconds between models |")
print(f"| Raw JSON | {code('.bench_raw/<profile>/')} |")
print()
print("## Hardware")
print()
print("| Component | Value |")
print("|---|---|")
print(f"| CPU | {md(os.environ['CPU'])} |")
print(f"| Logical cores | {md(os.environ['CORES'])} |")
print(f"| RAM | {md(os.environ['RAM_DISPLAY'])} |")
print(f"| OS | {md(os.environ['OS_NAME'])} {md(os.environ['OS_VER'])} |")
print(f"| Rust | {md(os.environ['RUST_VER'])} |")
print(f"| SIMD | {md(os.environ['SIMD'])} |")
print()
print("## Backend Profiles")
print()
print("| Profile | Env | Runtime report | Raw JSON |")
print("|---|---|---|---|")
for profile in profiles:
    print(
        f"| {md(profile['profile_label'])} | "
        f"{code('RUSTY_LLM_METAL=' + profile['metal_env'])} | "
        f"{md(profile['runtime']) or 'not observed'} | "
        f"{code(profile['raw_dir'] + '/')} |"
    )
print()
print("## Summary")
print()
print("| Profile | Ok | Failed | Skipped/partial | Best decode | Median decode |")
print("|---|---:|---:|---:|---:|---:|")
for profile in profiles:
    rows = rows_by_profile[profile["profile"]]
    ok = [row for row in rows if row["status"] == "supported"]
    failed = [row for row in rows if row["status"] == "bench_failed"]
    skipped = len(rows) - len(ok) - len(failed)
    speeds = [as_float(row["decode"]) for row in ok if as_float(row["decode"]) is not None]
    best = max(speeds) if speeds else None
    median = statistics.median(speeds) if speeds else None
    print(
        f"| {md(profile['profile_label'])} | {len(ok)} | {len(failed)} | {skipped} | "
        f"{fmt_speed(best)} | {fmt_speed(median)} |"
    )
print()

if "cpu" in profile_order and "metal" in profile_order:
    print("## CPU vs Metal")
    print()
    print("Each speed cell is `decode / prefill` in tokens per second. Speedup uses decode throughput.")
    print()
    print("| # | Model | Arch | Size | CPU | Metal | Metal/CPU | Result |")
    print("|---:|---|:---:|---:|---:|---:|---:|---|")
    for path, first in sorted(model_order.items(), key=lambda item: item[1]["index_num"]):
        cpu = rows_by_key.get(("cpu", path))
        metal = rows_by_key.get(("metal", path))
        if not cpu or not metal:
            continue

        def speed_pair(row):
            if row["status"] != "supported":
                return status_text(row["status"])
            return f"{fmt_speed(row['decode'])} / {fmt_speed(row['prefill'])}"

        cpu_decode = as_float(cpu["decode"]) if cpu["status"] == "supported" else None
        metal_decode = as_float(metal["decode"]) if metal["status"] == "supported" else None
        if cpu_decode and metal_decode and cpu_decode > 0:
            ratio = metal_decode / cpu_decode
            speedup = f"{ratio:.2f}x"
            if ratio > 1.05:
                result = "Metal faster"
            elif ratio < 0.95:
                result = "CPU faster"
            else:
                result = "similar"
        else:
            speedup = "—"
            notes = []
            for row in (cpu, metal):
                if row["status"] != "supported" and row["note"]:
                    notes.append(short_note(row["note"], 44))
            result = "; ".join(dict.fromkeys(notes)) or "not comparable"

        print(
            f"| {first['index_num']} | {code(first['file'])} | {md(first['arch'])} | "
            f"{fmt_int(first['mb'])} | {speed_pair(cpu)} | {speed_pair(metal)} | "
            f"{speedup} | {md(result)} |"
        )
    print()

issues = [row for row in results if row["status"] != "supported"]
if issues:
    print("## Support Issues")
    print()
    print("| Profile | Model | Arch | Status | Reason |")
    print("|---|---|:---:|---|---|")
    for row in issues:
        print(
            f"| {md(row['profile_label'])} | {code(row['file'])} | {md(row['arch'])} | "
            f"{md(status_text(row['status']))} | {md(short_note(row['note']))} |"
        )
    print()

print("## Profile Details")
print()
for profile in profiles:
    rows = sorted(rows_by_profile[profile["profile"]], key=lambda row: row["index_num"])
    print(f"### {md(profile['profile_label'])}")
    print()
    print("| # | Model | Arch | Status | Size | Load | Decode | Prefill | Note |")
    print("|---:|---|:---:|---|---:|---:|---:|---:|---|")
    for row in rows:
        print(
            f"| {row['index_num']} | {code(row['file'])} | {md(row['arch'])} | "
            f"{md(status_text(row['status']))} | {fmt_int(row['mb'])} | "
            f"{fmt_int(row['load'])} | {fmt_speed(row['decode'])} | "
            f"{fmt_speed(row['prefill'])} | {md(short_note(row['note']))} |"
        )
    print()

    ranking = [
        row for row in rows
        if row["status"] == "supported" and as_float(row["decode"]) is not None
    ]
    ranking.sort(key=lambda row: as_float(row["decode"]) or 0.0, reverse=True)
    if ranking:
        print(f"### {md(profile['profile_label'])} Decode Ranking")
        print()
        print("| Rank | Model | Decode | Prefill | Load |")
        print("|---:|---|---:|---:|---:|")
        for rank, row in enumerate(ranking, 1):
            print(
                f"| {rank} | {code(row['file'])} | {fmt_speed(row['decode'])} | "
                f"{fmt_speed(row['prefill'])} | {fmt_int(row['load'])} |"
            )
        print()

print("---")
print()
print("Re-run `bench_models.sh` to refresh this report. Use `BENCH_PROFILES=cpu` or `BENCH_PROFILES=metal` for a single backend run.")
PYEOF

log "Benchmark report written: $BENCHMARK_MD"

LINK_LINE="-> **[Benchmark results](BENCHMARK.md)** - CPU and Metal compatibility/speed for tested models."
if [ -f "$README" ]; then
  if grep -qF "BENCHMARK.md" "$README"; then
    log "README.md already links to BENCHMARK.md."
  else
    python3 - "$README" "$LINK_LINE" <<'PYEOF'
import sys

path, link = sys.argv[1], sys.argv[2]
with open(path) as handle:
    lines = handle.readlines()

insert_at = len(lines)
saw_content = False
for i, line in enumerate(lines):
    stripped = line.strip()
    if stripped and not stripped.startswith("#") and not stripped.startswith("[!["):
        saw_content = True
    if saw_content and stripped == "":
        insert_at = i + 1
        break

lines.insert(insert_at, "\n")
lines.insert(insert_at, link + "\n")
lines.insert(insert_at, "\n")
with open(path, "w") as handle:
    handle.writelines(lines)
PYEOF
    log "README.md link inserted."
  fi
else
  log "README.md not found; link insertion skipped."
fi

log ""
log "================================================================"
log "Done."
log "  BENCHMARK.md : $BENCHMARK_MD"
log "  Raw JSON     : $RAW_DIR/<profile>/"
log "================================================================"
