APP := rusty-llm
MODEL ?= model.gguf
ADDR ?= 127.0.0.1:8080
TLS_CERT ?= cert.pem
TLS_KEY ?= key.pem
WASM_OUT ?= demo/wasm/pkg
BENCH_RUNS ?= 3

.PHONY: build release run repl serve https bench wasm clean help

help:
	@echo "Targets:"
	@echo "  make build                 Build debug binary"
	@echo "  make release               Build optimized binary"
	@echo "  make run MODEL=... PROMPT='Hello'"
	@echo "  make repl MODEL=...        Start REPL mode"
	@echo "  make serve MODEL=...       Start HTTP endpoint"
	@echo "  make https MODEL=...       Start HTTPS endpoint"
	@echo "  make bench MODEL=...       Run generation benchmark"
	@echo "  make wasm                  Build the wasm demo package"

build:
	cargo build

release:
	RUSTFLAGS="-C target-cpu=native" cargo build --release

run: release
	./target/release/$(APP) "$(MODEL)" --prompt "$(PROMPT)"

repl: release
	./target/release/$(APP) "$(MODEL)" --repl

serve: release
	./target/release/$(APP) "$(MODEL)" --serve "$(ADDR)"

https: release
	./target/release/$(APP) "$(MODEL)" --serve "$(ADDR)" --tls-cert "$(TLS_CERT)" --tls-key "$(TLS_KEY)"

bench: release
	./target/release/$(APP) "$(MODEL)" --bench --bench-runs "$(BENCH_RUNS)" --prompt "$(PROMPT)"

wasm:
	rustup target add wasm32-unknown-unknown
	wasm-pack build --target web --features wasm --out-dir $(WASM_OUT)

clean:
	cargo clean
	rm -rf demo/wasm/pkg
