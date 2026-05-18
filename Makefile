APP := rusty-llm
MODEL ?= model.gguf
PROMPT ?=
ADDR ?= 127.0.0.1:8080
CHAT ?= 0
TLS_CERT ?= cert.pem
TLS_KEY ?= key.pem
WASM_OUT ?= demo/wasm/pkg
BENCH_RUNS ?= 3
CHAT_FLAG := $(if $(filter 1 true yes on,$(CHAT)),--chat,)

.PHONY: build release run repl serve https bench wasm clean help

help:
	@printf '%s\n' "rusty-llm make targets - TLDR"
	@printf '%s\n' ""
	@printf '%s\n' "USAGE"
	@printf '%s\n' "  make <target> [MODEL=/path/to/model.gguf] [PROMPT='...'] [ADDR=host:port]"
	@printf '%s\n' ""
	@printf '%s\n' "COMMON TARGETS"
	@printf '%s\n' "  make build                 Build debug binary"
	@printf '%s\n' "  make release               Build optimized binary for the native CPU"
	@printf '%s\n' "  make run MODEL=... PROMPT='Hello'"
	@printf '%s\n' "                             Generate from a one-shot prompt"
	@printf '%s\n' "  make repl MODEL=...        Start interactive REPL mode"
	@printf '%s\n' "  make serve MODEL=...       Start HTTP API at ADDR (default: $(ADDR))"
	@printf '%s\n' "  make serve MODEL=... CHAT=1"
	@printf '%s\n' "                             Start HTTP API with /chat enabled"
	@printf '%s\n' "  make https MODEL=...       Start HTTPS API with TLS_CERT/TLS_KEY"
	@printf '%s\n' "  make bench MODEL=...       Run generation benchmark"
	@printf '%s\n' "  make wasm                  Build the wasm demo package"
	@printf '%s\n' "  make clean                 Remove Cargo and wasm build artifacts"
	@printf '%s\n' ""
	@printf '%s\n' "VARIABLES"
	@printf '%s\n' "  MODEL=$(MODEL)"
	@printf '%s\n' "  PROMPT=$(PROMPT)"
	@printf '%s\n' "  ADDR=$(ADDR)"
	@printf '%s\n' "  CHAT=$(CHAT)"
	@printf '%s\n' "  BENCH_RUNS=$(BENCH_RUNS)"
	@printf '%s\n' "  TLS_CERT=$(TLS_CERT)"
	@printf '%s\n' "  TLS_KEY=$(TLS_KEY)"
	@printf '%s\n' "  WASM_OUT=$(WASM_OUT)"
	@printf '%s\n' ""
	@printf '%s\n' "USEFUL ENV FLAGS"
	@printf '%s\n' "  RUSTY_LLM_METAL=1          Enable experimental macOS Metal Q4_K path"
	@printf '%s\n' "  RUSTY_LLM_FAST_ATTN=1      Enable fast attention path when available"
	@printf '%s\n' "  RUSTY_LLM_MODEL_DIR=...    Default directory for model discovery"
	@printf '%s\n' ""
	@printf '%s\n' "EXAMPLES"
	@printf '%s\n' "  make run MODEL=~/models/model.gguf PROMPT='Wer war Albert Einstein?'"
	@printf '%s\n' "  make repl MODEL=~/models/model.gguf"
	@printf '%s\n' "  make serve MODEL=~/models/model.gguf ADDR=127.0.0.1:8080 CHAT=1"
	@printf '%s\n' "  make bench MODEL=~/models/model.gguf BENCH_RUNS=5 PROMPT='Explain SIMD briefly'"

build:
	cargo build

release:
	RUSTFLAGS="-C target-cpu=native" cargo build --release

run: release
	./target/release/$(APP) "$(MODEL)" --prompt "$(PROMPT)"

repl: release
	./target/release/$(APP) "$(MODEL)" --repl

serve: release
	./target/release/$(APP) "$(MODEL)" --serve "$(ADDR)" $(CHAT_FLAG)

https: release
	./target/release/$(APP) "$(MODEL)" --serve "$(ADDR)" --tls-cert "$(TLS_CERT)" --tls-key "$(TLS_KEY)" $(CHAT_FLAG)

bench: release
	./target/release/$(APP) "$(MODEL)" --bench --bench-runs "$(BENCH_RUNS)" --prompt "$(PROMPT)"

wasm:
	rustup target add wasm32-unknown-unknown
	wasm-pack build --target web --features wasm --out-dir $(WASM_OUT)

clean:
	cargo clean
	rm -rf demo/wasm/pkg
