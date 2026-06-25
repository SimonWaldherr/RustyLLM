APP        ?= rusty-llm
CARGO      ?= cargo
MODEL_DIR  ?= $(shell FIND_MODEL_DIR_ONLY=1 ./bench_models.sh 2>/dev/null || printf '%s\n' "$(HOME)/.lmstudio/models/lmstudio-community")
MODEL      ?=
PROMPT     ?= Wer war Albert Einstein?
SYNONYM_PROMPT ?= Nenne ein Synonym für Synonym und antworte nur mit diesem einen Wort.
NATO_PROMPT ?= Output exactly the 26 NATO phonetic alphabet code words from A to Z, one word per line. No letters, numbers, punctuation, parentheses, or explanation.
MAX_TOKENS ?= 32
TEMP       ?= 0
TOP_P      ?= 0.9
TOP_K      ?= 40
BENCH_RUNS ?= 3
BENCH_PROFILES ?= cpu metal
KERNEL_BENCH_RUNS ?= 25
KERNEL_BENCH_LAYER ?= 0
PROFILE    ?= auto
ADDR       ?= 127.0.0.1:8080
SERVE_ADDR ?= $(ADDR)
CHAT       ?= 1
TLS_CERT   ?= cert.pem
TLS_KEY    ?= key.pem
WASM_OUT   ?= demo/wasm/pkg
WASM_TARGET ?= wasm32-unknown-unknown
WASM_BINDGEN ?= wasm-bindgen
WASM_BINDGEN_VERSION ?= 0.2.100
WASM_OPT ?= wasm-opt
WASM_OPT_FLAGS ?= -Oz
RUSTFLAGS  ?= -C target-cpu=native

BIN        := ./target/release/$(APP)
CHAT_FLAG  := $(if $(filter 1 true yes on,$(CHAT)),--chat,)
_MODEL_ARG := $(if $(MODEL),--model "$(MODEL)",)
_RUN_ARGS  := --model-dir "$(MODEL_DIR)" $(_MODEL_ARG) --profile "$(PROFILE)" --prompt "$(PROMPT)" --max-tokens "$(MAX_TOKENS)" --temp "$(TEMP)" --top-p "$(TOP_P)" --top-k "$(TOP_K)"

.PHONY: all build release release-max run repl serve serve-metal serve-ultra https find-model-dir list-models inspect list-tensors bench cargo-bench bench-model bench-model-metal bench-model-ultra bench-models benchmark-report synonym-bench nato-bench nato-bench-metal kernel-bench kernel-bench-metal kernel-bench-ultra fmt test vet check wasm clean help

all: check release

build:
	$(CARGO) build

release:
	RUSTFLAGS="$(RUSTFLAGS)" $(CARGO) build --release

release-max:
	RUSTFLAGS="$(RUSTFLAGS)" $(CARGO) build --profile release-max

run: release
	$(BIN) $(_RUN_ARGS)

repl: release
	$(BIN) --model-dir "$(MODEL_DIR)" $(_MODEL_ARG) --repl

serve: release
	$(BIN) --model-dir "$(MODEL_DIR)" $(_MODEL_ARG) --serve "$(SERVE_ADDR)" $(CHAT_FLAG)

serve-metal: release
	RUSTY_LLM_METAL=1 $(BIN) --model-dir "$(MODEL_DIR)" $(_MODEL_ARG) --serve "$(SERVE_ADDR)" $(CHAT_FLAG)

serve-ultra: release
	RUSTY_LLM_METAL=1 $(BIN) --model-dir "$(MODEL_DIR)" $(_MODEL_ARG) --profile mistral-ultra --serve "$(SERVE_ADDR)" $(CHAT_FLAG)

https: release
	$(BIN) --model-dir "$(MODEL_DIR)" $(_MODEL_ARG) --serve "$(SERVE_ADDR)" --tls-cert "$(TLS_CERT)" --tls-key "$(TLS_KEY)" $(CHAT_FLAG)

find-model-dir:
	@FIND_MODEL_DIR_ONLY=1 ./bench_models.sh

list-models: release
	$(BIN) --model-dir "$(MODEL_DIR)" --list-models

inspect: release
	$(BIN) --model-dir "$(MODEL_DIR)" $(_MODEL_ARG) --inspect

list-tensors: release
	$(BIN) --model-dir "$(MODEL_DIR)" $(_MODEL_ARG) --list-tensors

bench: bench-model

cargo-bench:
	$(CARGO) bench

bench-model: release
	$(BIN) --model-dir "$(MODEL_DIR)" $(_MODEL_ARG) \
		--profile "$(PROFILE)" --prompt "$(PROMPT)" --max-tokens "$(MAX_TOKENS)" --temp "$(TEMP)" \
		--bench --bench-json --bench-runs "$(BENCH_RUNS)"

bench-model-metal: release
	RUSTY_LLM_METAL=1 $(BIN) --model-dir "$(MODEL_DIR)" $(_MODEL_ARG) \
		--profile "$(PROFILE)" --prompt "$(PROMPT)" --max-tokens "$(MAX_TOKENS)" --temp "$(TEMP)" \
		--bench --bench-json --bench-runs "$(BENCH_RUNS)"

bench-model-ultra: release
	RUSTY_LLM_METAL=1 $(BIN) --model-dir "$(MODEL_DIR)" $(_MODEL_ARG) \
		--profile mistral-ultra --prompt "$(PROMPT)" --max-tokens "$(MAX_TOKENS)" --temp "$(TEMP)" \
		--bench --bench-json --bench-runs "$(BENCH_RUNS)"

bench-models: release
	BENCH_PROFILES="$(BENCH_PROFILES)" ./bench_models.sh

benchmark-report:
	REPORT_ONLY=1 ./bench_models.sh

synonym-bench: release
	$(BIN) --model-dir "$(MODEL_DIR)" $(_MODEL_ARG) \
		--prompt "$(SYNONYM_PROMPT)" --max-tokens "8" --temp "0" \
		--top-p "$(TOP_P)" --top-k "$(TOP_K)" --bench --bench-json --bench-runs "$(BENCH_RUNS)"

nato-bench: release
	$(BIN) --model-dir "$(MODEL_DIR)" $(_MODEL_ARG) \
		--prompt "$(NATO_PROMPT)" --max-tokens "128" --temp "0" \
		--top-p "$(TOP_P)" --top-k "$(TOP_K)" --repeat-penalty "1" --bench --bench-json --bench-runs "$(BENCH_RUNS)"

nato-bench-metal: release
	RUSTY_LLM_METAL=1 $(BIN) --model-dir "$(MODEL_DIR)" $(_MODEL_ARG) \
		--prompt "$(NATO_PROMPT)" --max-tokens "128" --temp "0" \
		--top-p "$(TOP_P)" --top-k "$(TOP_K)" --repeat-penalty "1" --bench --bench-json --bench-runs "$(BENCH_RUNS)"

kernel-bench: release
	$(BIN) --model-dir "$(MODEL_DIR)" $(_MODEL_ARG) \
		--profile "$(PROFILE)" --kernel-bench-json --kernel-bench-runs "$(KERNEL_BENCH_RUNS)" --kernel-bench-layer "$(KERNEL_BENCH_LAYER)"

kernel-bench-metal: release
	RUSTY_LLM_METAL=1 $(BIN) --model-dir "$(MODEL_DIR)" $(_MODEL_ARG) \
		--profile "$(PROFILE)" --kernel-bench-json --kernel-bench-runs "$(KERNEL_BENCH_RUNS)" --kernel-bench-layer "$(KERNEL_BENCH_LAYER)"

kernel-bench-ultra: release
	RUSTY_LLM_METAL=1 $(BIN) --model-dir "$(MODEL_DIR)" $(_MODEL_ARG) \
		--profile mistral-ultra --kernel-bench-json --kernel-bench-runs "$(KERNEL_BENCH_RUNS)" --kernel-bench-layer "$(KERNEL_BENCH_LAYER)"

fmt:
	$(CARGO) fmt

test:
	$(CARGO) test

vet:
	$(CARGO) clippy --all-targets -- -D warnings

check: fmt test vet

wasm:
	@rustup target list --installed | grep -qx "$(WASM_TARGET)" || rustup target add $(WASM_TARGET)
	$(CARGO) build --lib --release --target $(WASM_TARGET) --no-default-features --features wasm
	@if ! command -v $(WASM_BINDGEN) >/dev/null 2>&1 || \
		[ "$$($(WASM_BINDGEN) --version | sed 's/^wasm-bindgen //')" != "$(WASM_BINDGEN_VERSION)" ]; then \
		$(CARGO) install wasm-bindgen-cli --version "$(WASM_BINDGEN_VERSION)" --locked --force; \
	fi
	rm -rf "$(WASM_OUT)"
	mkdir -p "$(WASM_OUT)"
	$(WASM_BINDGEN) --target web --out-dir "$(WASM_OUT)" --out-name rusty_llm "target/$(WASM_TARGET)/release/rusty_llm.wasm"
	@if command -v $(WASM_OPT) >/dev/null 2>&1; then \
		$(WASM_OPT) $(WASM_OPT_FLAGS) -o "$(WASM_OUT)/rusty_llm_bg.wasm" "$(WASM_OUT)/rusty_llm_bg.wasm"; \
	else \
		printf "Skipping wasm-opt; install binaryen for optional size optimization.\n"; \
	fi

clean:
	$(CARGO) clean
	rm -rf demo/wasm/pkg

help:
	@printf "Targets:\n"
	@printf "  make all                             Run check and release build\n"
	@printf "  make build                           Build debug binary\n"
	@printf "  make release                         Build optimized native binary with faster ThinLTO\n"
	@printf "  make release-max                     Build slower FatLTO binary for final benchmarking\n"
	@printf "  make run MODEL=... PROMPT='...'      Generate from a one-shot prompt\n"
	@printf "  make repl MODEL=...                  Start interactive REPL mode\n"
	@printf "  make serve MODEL=... CHAT=1          Start HTTP API / optional web UI\n"
	@printf "  make serve-metal MODEL=...           Start server with RUSTY_LLM_METAL=1\n"
	@printf "  make serve-ultra MODEL=...           Start Mistral Ultra server with aggressive Metal routing\n"
	@printf "  make https MODEL=...                 Start HTTPS API with TLS_CERT/TLS_KEY\n"
	@printf "  make find-model-dir                  Print the auto-detected GGUF model directory\n"
	@printf "  make list-models                     List GGUFs in MODEL_DIR\n"
	@printf "  make inspect MODEL=...               Inspect GGUF metadata and compatibility\n"
	@printf "  make list-tensors MODEL=...          Print tensor inventory\n"
	@printf "  make bench MODEL=...                 Run generation benchmark with tokens/sec JSON\n"
	@printf "  make cargo-bench                     Run Rust benchmark harness\n"
	@printf "  make bench-model MODEL=...           Run CLI generation benchmark JSON with per-run output\n"
	@printf "  make bench-model-metal MODEL=...     Run generation benchmark with RUSTY_LLM_METAL=1\n"
	@printf "  make bench-model-ultra MODEL=...     Run Mistral Ultra benchmark with aggressive Metal routing\n"
	@printf "  make bench-models                    Refresh BENCHMARK.md across discovered models\n"
	@printf "  make benchmark-report                Rebuild BENCHMARK.md from existing .bench_raw TSV files\n"
	@printf "  make synonym-bench MODEL=...         Run fixed one-word synonym prompt benchmark\n"
	@printf "  make nato-bench MODEL=...            Run fixed NATO alphabet prompt benchmark\n"
	@printf "  make nato-bench-metal MODEL=...      Run NATO benchmark with RUSTY_LLM_METAL=1\n"
	@printf "  make kernel-bench MODEL=...          Run isolated kernel benchmark JSON\n"
	@printf "  make kernel-bench-metal MODEL=...    Run isolated kernel benchmark with RUSTY_LLM_METAL=1\n"
	@printf "  make kernel-bench-ultra MODEL=...    Run isolated Mistral Ultra kernel benchmark\n"
	@printf "  make fmt/test/vet/check              Format, test, lint, or all three\n"
	@printf "  make wasm                            Build stable web wasm package\n"
	@printf "  make clean                           Remove build artifacts\n"
	@printf "\nVariables:\n"
	@printf "  MODEL_DIR=%s\n" "$(MODEL_DIR)"
	@printf "  MODEL=%s\n" "$(MODEL)"
	@printf "  PROMPT=%s\n" "$(PROMPT)"
	@printf "  SYNONYM_PROMPT=%s\n" "$(SYNONYM_PROMPT)"
	@printf "  NATO_PROMPT=%s\n" "$(NATO_PROMPT)"
	@printf "  MAX_TOKENS=%s TEMP=%s TOP_P=%s TOP_K=%s\n" "$(MAX_TOKENS)" "$(TEMP)" "$(TOP_P)" "$(TOP_K)"
	@printf "  BENCH_RUNS=%s BENCH_PROFILES=%s PROFILE=%s SERVE_ADDR=%s CHAT=%s\n" "$(BENCH_RUNS)" "$(BENCH_PROFILES)" "$(PROFILE)" "$(SERVE_ADDR)" "$(CHAT)"
	@printf "  KERNEL_BENCH_RUNS=%s KERNEL_BENCH_LAYER=%s\n" "$(KERNEL_BENCH_RUNS)" "$(KERNEL_BENCH_LAYER)"
	@printf "  WASM_OUT=%s WASM_TARGET=%s WASM_BINDGEN_VERSION=%s\n" "$(WASM_OUT)" "$(WASM_TARGET)" "$(WASM_BINDGEN_VERSION)"
