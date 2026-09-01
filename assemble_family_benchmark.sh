#!/usr/bin/env bash
# Combines isolated, thermally controlled family runs into one decoder report.

set -euo pipefail

SOURCE_ROOT="${SOURCE_ROOT:-.bench_raw/families-final}"
RAW_DIR="${RAW_DIR:-.bench_raw/families-final/combined}"
REPORT="${REPORT:-BENCHMARK_FAMILIES.md}"
REFERENCE_BIN="${REFERENCE_BIN:-/opt/homebrew/bin/llama-bench}"

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

mkdir -p "$RAW_DIR"
printf 'model\tengine\trun\torder\tprefill_tok_s\tdecode_tok_s\tprompt_tokens\tgenerated_tokens\traw_log\n' > "$RAW_DIR/results.tsv"
printf 'model\tpath\tbytes\n' > "$RAW_DIR/models.tsv"
printf 'timestamp\tmodel\tengine\trun\tphase\tstate\n' > "$RAW_DIR/thermal.tsv"

for family in ministral gemma qwen llama; do
  family_dir="$SOURCE_ROOT/$family"
  [ -s "$family_dir/results.tsv" ] || die "missing $family_dir/results.tsv"
  [ -s "$family_dir/models.tsv" ] || die "missing $family_dir/models.tsv"
  [ -s "$family_dir/thermal.tsv" ] || die "missing $family_dir/thermal.tsv"
  sed '1d' "$family_dir/results.tsv" >> "$RAW_DIR/results.tsv"
  sed '1d' "$family_dir/models.tsv" >> "$RAW_DIR/models.tsv"
  sed '1d' "$family_dir/thermal.tsv" >> "$RAW_DIR/thermal.tsv"
done

first_env="$SOURCE_ROOT/ministral/environment.tsv"
[ -s "$first_env" ] || die "missing $first_env"
generated_at=$(awk -F '\t' '$1 == "generated_at" { sub(/^[^\t]*\t/, ""); print; exit }' "$first_env")
host=$(awk -F '\t' '$1 == "host" { sub(/^[^\t]*\t/, ""); print; exit }' "$first_env")
cpu=$(awk -F '\t' '$1 == "cpu" { sub(/^[^\t]*\t/, ""); print; exit }' "$first_env")
os=$(awk -F '\t' '$1 == "os" { sub(/^[^\t]*\t/, ""); print; exit }' "$first_env")
memory_bytes=$(awk -F '\t' '$1 == "memory_bytes" { print $2; exit }' "$first_env")
rust_revision=$(git rev-parse --short HEAD 2>/dev/null || printf unknown)
reference_json=""
while IFS= read -r candidate; do
  reference_json="$candidate"
  break
done < <(find "$SOURCE_ROOT" -type f -name '*.reference.run*.json' -print 2>/dev/null | sort)
[ -n "$reference_json" ] || die "missing reference benchmark JSON"
reference_version=$(jq -r '"build " + (.[0].build_number | tostring) + ", commit " + .[0].build_commit' "$reference_json")

{
  printf 'generated_at\t%s\n' "$generated_at"
  printf 'host\t%s\n' "$host"
  printf 'cpu\t%s\n' "$cpu"
  printf 'os\t%s\n' "$os"
  printf 'memory_bytes\t%s\n' "$memory_bytes"
  printf 'rust_revision\t%s\n' "$rust_revision"
  printf 'dirty\t + working tree changes\n'
  printf 'reference_version\t%s\n' "$reference_version"
  printf 'reference_kind\tbench\n'
  printf 'reference_prompt_tokens\t32\n'
  printf 'runs\t3\n'
  printf 'max_tokens\t64\n'
  printf 'cooldown_secs\tfamily-specific 45–120\n'
  printf 'seed\t42\n'
  printf 'prompt\tExplain in one compact paragraph why memory bandwidth matters for local LLM inference.\n'
  printf 'report_title\tRustyLLM vs llama.cpp — thermally controlled decoder families\n'
  printf 'protocol_note\tEach family ran in isolation with its own initial and per-process cooling interval. See the family reports and raw environment files for exact values. Nomic is reported separately because it is an encoder workload.\n'
} > "$RAW_DIR/environment.tsv"

REPORT_ONLY=1 \
RAW_DIR="$RAW_DIR" \
REPORT="$REPORT" \
REFERENCE_BIN="$REFERENCE_BIN" \
REFERENCE_KIND=bench \
REFERENCE_PROMPT_TOKENS=32 \
./bench_reference.sh
