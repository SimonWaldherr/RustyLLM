# RustyLLM

[![DOI](https://zenodo.org/badge/1201177634.svg)](https://doi.org/10.5281/zenodo.20772661)
[![CI](https://github.com/SimonWaldherr/RustyLLM/actions/workflows/ci.yml/badge.svg)](https://github.com/SimonWaldherr/RustyLLM/actions/workflows/ci.yml)
[![License](https://img.shields.io/github/license/SimonWaldherr/RustyLLM)](LICENSE)
[![GitHub release](https://img.shields.io/github/v/release/SimonWaldherr/RustyLLM)](https://github.com/SimonWaldherr/RustyLLM/releases)
[![Crates.io](https://img.shields.io/crates/v/rusty-llm)](https://crates.io/crates/rusty-llm)
[![docs.rs](https://img.shields.io/docsrs/rusty-llm)](https://docs.rs/rusty-llm)
[![Rust](https://img.shields.io/badge/rust-1.95%2B-orange.svg)](https://www.rust-lang.org/)

→ **[Benchmark results](BENCHMARK.md)** — compatibility and speed for 14 tested models on Apple M2 Max.

RustyLLM is an educational GGUF inference runner for developers who want to
understand how a local language-model runtime works. You do not need AI
experience to read the project: the code is organized as ordinary file parsing,
arrays, math kernels, state management, HTTP routing, and optional browser/WASM
experiments.

At a high level, RustyLLM reads a `.gguf` model file, converts input text into
integer token IDs, repeatedly runs those IDs through model weights to predict
the next token, and converts the chosen output tokens back into text. The
project is deliberately small enough to read end to end, while still showing the
complete path from model-file parsing to a command-line tool and a minimal
OpenAI-compatible HTTP API.

The runner loads model weights directly from disk, keeps quantized tensors in
memory-mapped storage on native targets, and exposes the same core through a
CLI, a small HTTP server, LM Studio-compatible aliases, Ollama-compatible routes,
and a Rust library API.

RustyLLM is best treated as learning-oriented infrastructure: practical enough
to run small local models, intentionally dependency-light, and transparent enough
for studying how local inference systems are assembled without first adopting a
large production runtime.

If the AI terms are new, start with
[AI inference for non-AI developers](docs/AI_FOR_DEVELOPERS.md). It explains the
core vocabulary used by the codebase before the module-by-module architecture
guide.

## Learning Path

If you are reading the code to understand local inference, use this order:

1. `src/gguf.rs` parses the GGUF container, tensor directory, and metadata.
2. `src/tokenizer.rs` turns text into token IDs and decodes generated tokens.
3. `src/simd.rs` implements scalar, NEON, AVX2/FMA, and quantized math kernels.
4. `src/model.rs` loads tensors and runs transformer forward passes.
5. `src/runtime.rs` wraps the model with generation, chat templates, embeddings,
   benchmark helpers, and optional session reuse.
6. `src/server.rs` maps the runner onto the native, OpenAI-compatible,
   LM Studio-compatible, and Ollama-compatible HTTP routes.

Additional documentation:

- [AI inference for non-AI developers](docs/AI_FOR_DEVELOPERS.md) explains the
  vocabulary and mental model used by the project.
- [Architecture guide](docs/ARCHITECTURE.md) explains the inference pipeline and
  module responsibilities.
- [MTP usage guide](docs/MTP.md) explains greedy assistant-based speculative
  decoding, benchmark comparison, and troubleshooting.
- [Function reference](docs/FUNCTION_REFERENCE.md) documents every non-test Rust
  function under `src/`.

## Highlights

- Native GGUF loading with zero-copy memory mapping on macOS and Linux.
- GGUF metadata inspection, model discovery, model selection, and tensor listing.
- Tokenizer support for SentencePiece-style and GPT-2-style metadata.
- Quantized inference paths for `Q8_0`, `Q4_0`, `Q4_K`, `Q6_K`, and `MXFP4`
  tensors.
- SIMD kernels for Apple Silicon NEON and x86_64 AVX2/FMA, with scalar fallback.
- Metal acceleration for Q4_K/Q6_K matrix-vector work on macOS, enabled by
  default when the Objective-C shim builds and the GPU backend is available.
  Set `RUSTY_LLM_METAL=0` to force the CPU path.
- One-shot generation, interactive REPL mode, benchmark mode, JSON benchmark
  output, and append-only chat history logging.
- Prompt-selected `SKILL.md` loading via `--skills-dir`, with per-session
  de-duplication so long chats do not inject the same skill repeatedly.
- OpenAI-compatible `/v1/models`, `/v1/completions`, `/v1/chat/completions`,
  `/v1/responses`, and `/v1/embeddings` routes.
- LM Studio-style `/api/v0/*` aliases and Ollama-style `/api/*` compatibility
  routes.
- Server-Sent Events streaming for OpenAI-compatible completions and chat
  completions, plus Responses API streaming and Ollama-style NDJSON streaming.
- OpenAPI 3.1 document at `/openapi.json` with Swagger UI at `/docs`.
- Model Context Protocol stdio mode with `generate`, `chat`, `embed`, and
  `models` tools.
- Text embeddings via `Runner::embed`, mean-pooled over the last transformer
  layer and L2-normalized for cosine similarity.
- Minimal browser chat UI served from `/chat`, an expert UI from
  `/chat?expert`, and a GGUF explorer from `/explorer`.
- Library API for embedding RustyLLM in other Rust applications.
- `wasm32-unknown-unknown` check support for the no-default-features WASM build.

## Supported Model Families

RustyLLM accepts GGUF files whose `general.architecture` metadata matches one of
the supported architecture identifiers:

`llama`, `llama2`, `llama3`, `mistral`, `mistral3`, `mixtral`, `ministral`,
`qwen2`, `qwen3`, `gpt-oss`, `gemma`, `gemma2`, `gemma3`, `gemma4`,
`gemma4n`, `gemma4-assistant`, `granite`, `granite3`, `granite4`,
`deepseek`, `deepseek-v2`, `deepseek2`, `nemotron`, `hermes`, `phi`, `phi2`,
`phi3`, `phi4`, `falcon`, `falcon3`, `stablelm`, `starcoder2`, `command-r`,
`cohere`, `internlm2`, `olmo`, `olmo2`, `exaone`, `solar`, `yi`, `arctic`,
`nomic-bert`, `nomic-embed`, and
`text-embedding-nomic-embed-text`.

Support still depends on the tensors present in a specific GGUF file. Use
`--inspect` before loading an unfamiliar model to verify architecture, tensor
types, tokenizer metadata, and API compatibility.

Gemma-family GGUFs use the dedicated Gemma loader and native
`<start_of_turn>` chat formatting when the tokenizer template exposes it.
Q4_0 QAT GGUFs such as `google/gemma-4-12B-it-qat-q4_0-gguf` are supported by
the same path and use fused CPU Q/K/V and Gate/Up projection jobs when Metal is
not selected for those projections.

## Requirements

- Rust 1.95 or newer. The repository pins `1.95.0` in
  [rust-toolchain.toml](rust-toolchain.toml).
- A GGUF model file. The runner does not download models.
- macOS or Linux for native memory-mapped execution.
- Optional for WebAssembly experiments: `wasm-pack` and the
  `wasm32-unknown-unknown` target.
- Optional for macOS Metal experiments: Xcode command line tools with `xcrun`,
  `clang`, and `ar`.

## Build

```bash
cargo build --release
```

The release binary is written to:

```text
target/release/rusty-llm
```

For local performance work, build for the native CPU:

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

The Makefile wraps the common commands:

```bash
make help
make release
make run MODEL=/path/to/model.gguf PROMPT="Explain GGUF in one paragraph"
make repl MODEL=/path/to/model.gguf
make serve MODEL=/path/to/model.gguf ADDR=127.0.0.1:8080 CHAT=1
make bench MODEL=/path/to/model.gguf BENCH_RUNS=5 PROMPT="Explain SIMD briefly"
```

## Quick Start

Run one prompt:

```bash
./target/release/rusty-llm ./models/model.gguf \
  --prompt "Explain rotary embeddings in two sentences." \
  --max-tokens 128
```

Start a chat REPL:

```bash
./target/release/rusty-llm ./models/model.gguf --repl
```

Start the HTTP API:

```bash
./target/release/rusty-llm ./models/model.gguf --serve 127.0.0.1:8080
```

Start the HTTP API with the built-in chat UI:

```bash
./target/release/rusty-llm ./models/model.gguf --serve 127.0.0.1:8080 --chat
```

Then open:

- `http://127.0.0.1:8080/chat`
- `http://127.0.0.1:8080/chat?expert`
- `http://127.0.0.1:8080/explorer`

The explorer shows GGUF metadata, tokenizer output, token-embedding vectors,
nearest vocabulary neighbors, the tensor directory, and the model catalog
discovered from the configured `--model-dir`.

The Chat and Expert views expose a model-independent Thinking toggle. When
enabled, RustyLLM first asks the loaded model to rewrite the latest user prompt
with a compact meta-prompt, then uses that rewritten prompt for the final answer.

Optional Skills can be enabled by pointing RustyLLM at a directory tree that
contains `SKILL.md` files:

```bash
./target/release/rusty-llm ./models/model.gguf \
  --skills-dir ./skills/default \
  --repl
```

RustyLLM indexes skill names and descriptions, loads only the matching
`SKILL.md` bodies for each prompt, and remembers loaded skill paths inside the
REPL or server session. The repository includes self-contained example Skills
under `skills/default`:

- `rust-code-review`
- `local-llm-troubleshooting`
- `skill-authoring`
- `german-technical-writing`

RustyLLM does not execute skill scripts or lazily read `references/` files; keep
Skills self-contained unless the user explicitly provides additional context.

When serving an exact `.gguf` file path, RustyLLM skips the recursive startup
catalog scan so the server can start loading the requested model immediately.

## Model Discovery

The general CLI form is:

```bash
rusty-llm [model.gguf|model-name|model-dir] [options]
```

You can pass an exact `.gguf` file:

```bash
rusty-llm ./models/model.gguf --prompt "Hello"
```

You can also select a model from a directory:

```bash
rusty-llm --model-dir ./models --list-models
rusty-llm --model-dir ./models --model phi-4 --prompt "Write a Rust enum example"
```

When `--model` is an exact `.gguf` file name or a path relative to
`--model-dir`, RustyLLM resolves it with a lightweight file scan before falling
back to full metadata discovery.

If no model directory is provided, RustyLLM uses:

1. `RUSTY_LLM_MODEL_DIR`, when set and non-empty.
2. the default local LM Studio community model cache.

For benchmark automation, `bench_models.sh` and the Makefile's default
`MODEL_DIR` scan additional common LM Studio, Ollama, GPT4All, Jan, `~/models`,
and project-local model paths.

Model selection is intentionally lenient: `--model` can match repository names,
file names, relative IDs, or GGUF metadata names. If a selector matches multiple
models, RustyLLM prints the matching choices and asks for a more specific value.

Projector files such as `mmproj-*.gguf` are ignored for text model selection.

## CLI Reference

```text
rusty-llm [model.gguf|model-name|model-dir] [options]
```

Model and inspection options:

- `--model <name>` selects a GGUF from `--model-dir`.
- `--model-dir <path>` recursively scans a directory for `.gguf` files.
- `--list-models` lists discovered models and exits.
- `--inspect` prints a JSON compatibility report without loading weights.
- `--list-tensors` loads the model and prints tensor names, dtypes, and shapes.
- `--verbose` or `-v` prints startup timing details, including model
  resolution, mmap open, GGUF parsing, tokenizer build, and weight setup.

Execution modes:

- `--prompt <text>` or `-p <text>` runs one-shot generation.
- `--repl` starts an interactive chat session.
- `--serve <addr>` starts the HTTP(S) server, for example `127.0.0.1:8080`.
- `--mcp` starts a Model Context Protocol stdio server for the loaded model.
- `--chat` enables the built-in web UIs at `/chat`, `/chat?expert`, and
  `/explorer`.
- `--embed` embeds `--prompt` and prints the embedding vector.
- `--bench` runs a non-streaming generation benchmark.
- `--bench-json` runs benchmark mode and emits a machine-readable JSON report.

Generation options:

- `--max-tokens <N>` or `-n <N>` sets the maximum number of generated tokens.
- `--temp <F>` or `-t <F>` sets temperature; `0` uses greedy decoding.
- `--top-p <F>` sets nucleus sampling in the range `(0, 1]`.
- `--top-k <N>` sets top-k filtering.
- `--repeat-penalty <F>` applies a repetition penalty to recent tokens.
- `--seed <N>` sets the RNG seed. `0` uses the default time-based behavior.
- `--system-prompt <text>` overrides the default chat system prompt.
- `--thinking` rewrites each prompt with the built-in Thinking meta-prompt
  before answering.
- `--thinking-prompt <text>` overrides the Thinking meta-prompt.
- `--thinking-max-tokens <N>` caps the internal Thinking rewrite.
- `--skills-dir <path>` enables prompt-selected Skills from a directory tree of
  `SKILL.md` files. Use `--skills-dir skills/default` to try the bundled
  examples.
- `--max-skills <N>` limits new Skills loaded for one prompt. The default is
  `3`.
- `--skill-max-bytes <N>` caps the loaded bytes per `SKILL.md`. The default is
  `16384`.
- `--stop <text>` stops generation when the text appears. The flag can be
  repeated.
- `--threads <N>` overrides the SIMD worker thread count.
- `--profile <name>` selects runtime planning: `auto`, `mistral`,
  `mistral-ultra`, or `gemma`. `mistral-ultra` is an experimental aggressive
  Metal mode for Mistral/Ministral-style GGUFs; it lowers Metal dispatch
  thresholds for Q4_K/Q6_K projections and attention scans, with native SIMD
  fallback for kernels that still run on CPU.
- `--mtp-assistant <path>` loads a smaller assistant GGUF for greedy
  speculative decoding.
- `--mtp-tokens <N>` sets the maximum speculative draft tokens.
- `--mtp-min-accept-rate <F>` disables MTP when the acceptance rate drops below
  this threshold. The default is `0.5`.
- `--no-mtp-adaptive` keeps the MTP draft length fixed instead of adapting it.
- `--no-speculative` disables MTP/speculative decoding.

Server options:

- `--tls-cert <path>` enables HTTPS with a PEM certificate.
- `--tls-key <path>` enables HTTPS with a PEM private key.
- `--max-connections <N>` caps concurrent server connections. The default is
  `max(16, available_threads * 8)`.
- `--chat-history <path>` or `--chat-log <path>` appends CLI and server turns to
  a JSON file.

## CLI Examples

One-shot generation:

```bash
rusty-llm ./models/model.gguf \
  --prompt "Name three practical uses for local embeddings." \
  --max-tokens 96 \
  --temp 0.7 \
  --top-p 0.9
```

Read a prompt from stdin:

```bash
printf "Summarize grouped-query attention." | rusty-llm ./models/model.gguf
```

Stop at a custom delimiter:

```bash
rusty-llm ./models/model.gguf \
  --prompt "Name three fruits:" \
  --stop "\n" \
  --max-tokens 32
```

Use the LM Studio community cache:

```bash
rusty-llm --list-models
rusty-llm --model phi-4 --prompt "Write a concise Rust trait example"
```

Run a local HTTPS server:

```bash
rusty-llm ./models/model.gguf \
  --serve 127.0.0.1:8443 \
  --tls-cert cert.pem \
  --tls-key key.pem
```

Write chat history:

```bash
rusty-llm ./models/model.gguf \
  --repl \
  --chat-history ./runs/chat-history.json
```

## HTTP API

Start the server:

```bash
rusty-llm ./models/model.gguf --serve 127.0.0.1:8080
```

Health and metadata routes:

- `GET /`, `GET /health`, `GET /healthz`, `GET /ready`
- `GET /api/version`
- `GET /openapi.json`, `GET /swagger.json`, `GET /docs`
- `GET /v1/models`
- `GET /api/v0/models`
- `GET /api/tags`
- `GET /api/explorer/model` (loaded model metadata, tensor inventory, and
  discovered model catalog)

Generation and embedding routes:

- `POST /generate`
- `POST /v1/completions`
- `POST /v1/chat/completions`
- `POST /v1/responses`
- `POST /v1/embeddings`
- `POST /api/v0/completions`
- `POST /api/v0/chat/completions`
- `POST /api/v0/embeddings`
- `POST /api/generate`
- `POST /api/chat`
- `POST /api/embeddings`
- `POST /api/embed`
- `POST /api/explorer/tokenize`
- `POST /api/explorer/vector`
- `POST /api/explorer/neighbors`

All `POST` routes require `Content-Type: application/json`. Requests are bounded
by header and body limits and a per-connection I/O timeout. CORS headers are
included on responses.

### Native `/generate`

Prompt input:

```bash
curl -X POST http://127.0.0.1:8080/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "Summarize grouped-query attention in two sentences.",
    "max_tokens": 80,
    "temp": 0.7,
    "top_p": 0.9,
    "top_k": 40,
    "repeat_penalty": 1.1,
    "stop": ["</s>", "\n\n"]
  }'
```

Chat input:

```bash
curl -X POST http://127.0.0.1:8080/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [
      {"role": "system", "content": "You are concise."},
      {"role": "user", "content": "What is GGUF?"}
    ],
    "max_tokens": 64
  }'
```

Response:

```json
{
  "text": "...",
  "prompt_tokens": 123,
  "generated_tokens": 64,
  "prefill_ms": 42,
  "decode_ms": 180,
  "total_ms": 223
}
```

### OpenAI-Compatible Chat

Non-streaming:

```bash
curl -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "llama3",
    "messages": [
      {"role": "system", "content": "You are concise."},
      {"role": "user", "content": "What is GGUF?"}
    ],
    "max_tokens": 64,
    "temperature": 0.7,
    "stop": ["</answer>"]
  }'
```

Streaming SSE:

```bash
curl -N -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "llama3",
    "messages": [{"role": "user", "content": "Tell me a joke."}],
    "max_completion_tokens": 128,
    "stream": true
  }'
```

Each chunk is emitted as:

```text
data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":...,"model":"llama3","choices":[{"index":0,"delta":{"content":"..."},"finish_reason":null}]}
```

The final event is:

```text
data: [DONE]
```

`max_completion_tokens` is accepted as an alias for `max_tokens`.

### OpenAI-Compatible Completions

```bash
curl -X POST http://127.0.0.1:8080/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "local-model",
    "prompt": "Complete this sentence: Rust is",
    "max_tokens": 48,
    "temperature": 0.5
  }'
```

Streaming is also supported on `/v1/completions` and `/api/v0/completions` with
`"stream": true`.

### OpenAI-Compatible Responses

```bash
curl -X POST http://127.0.0.1:8080/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "local-model",
    "instructions": "You are concise.",
    "input": "Explain GGUF in one sentence.",
    "max_output_tokens": 64
  }'
```

Streaming is supported with `"stream": true` and emits Responses-style SSE
events ending in `data: [DONE]`.

RustyLLM accepts `response_format`, `text.format`, `tools`, and `tool_choice`
fields for OpenAI client compatibility. JSON response formats add a prompt-level
instruction to produce valid JSON; RustyLLM does not yet enforce JSON schemas
token by token, and tool definitions are not executed by the model server.

### Multimodal Message Format

RustyLLM accepts OpenAI-style multimodal `content` arrays on chat routes:

```bash
curl -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "llama3",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "Describe what you see:"},
          {"type": "image_url", "image_url": {"url": "https://example.com/photo.jpg"}}
        ]
      }
    ],
    "max_tokens": 128
  }'
```

Image references are converted into text placeholders such as
`[image: https://example.com/photo.jpg]` or `[image: base64 data]`. RustyLLM does
not currently run a vision encoder.

### Embeddings

OpenAI-compatible single input:

```bash
curl -X POST http://127.0.0.1:8080/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "nomic-embed",
    "input": "The quick brown fox jumps over the lazy dog"
  }'
```

OpenAI-compatible batch input:

```bash
curl -X POST http://127.0.0.1:8080/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "nomic-embed",
    "input": ["First sentence.", "Second sentence."]
  }'
```

Response shape:

```json
{
  "object": "list",
  "data": [
    {"object": "embedding", "embedding": [0.012, -0.034], "index": 0}
  ],
  "model": "nomic-embed",
  "usage": {"prompt_tokens": 9, "total_tokens": 9}
}
```

Ollama-style embeddings:

```bash
curl -X POST http://127.0.0.1:8080/api/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model": "nomic-embed", "prompt": "The quick brown fox"}'
```

### Ollama-Compatible Routes

List tags:

```bash
curl http://127.0.0.1:8080/api/tags
```

Generate:

```bash
curl -X POST http://127.0.0.1:8080/api/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "local",
    "prompt": "Why are GGUF models convenient?",
    "options": {
      "num_predict": 80,
      "temperature": 0.7,
      "top_p": 0.9,
      "top_k": 40,
      "repeat_penalty": 1.1
    }
  }'
```

Chat:

```bash
curl -X POST http://127.0.0.1:8080/api/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "local",
    "messages": [{"role": "user", "content": "What is memory mapping?"}]
  }'
```

Ollama `stream: true` requests on `/api/generate` and `/api/chat` return
newline-delimited JSON chunks with a final `done: true` object:

```bash
curl -N -X POST http://127.0.0.1:8080/api/generate \
  -H 'Content-Type: application/json' \
  -d '{"model": "local", "prompt": "Say hello.", "stream": true}'
```

### OpenAPI and Swagger

The server exposes a machine-readable OpenAPI document and a browser docs page:

- `GET /openapi.json`
- `GET /swagger.json`
- `GET /docs`

`/docs` loads Swagger UI from a CDN and uses `/openapi.json` as its spec source.

### Model Context Protocol

Start a stdio MCP server for the loaded model:

```bash
rusty-llm ./models/model.gguf --mcp
```

The MCP server exposes `generate`, `chat`, `embed`, and `models` tools. It uses
newline-delimited JSON-RPC on stdin/stdout and writes boot logs to stderr.

## Benchmarking

Run a text benchmark:

```bash
rusty-llm ./models/model.gguf \
  --bench \
  --bench-runs 5 \
  --max-tokens 64 \
  --threads 8 \
  --prompt "Explain local LLM inference performance in one concise paragraph."
```

Emit JSON for scripts or CI artifacts:

```bash
rusty-llm ./models/model.gguf \
  --bench-json \
  --bench-runs 5 \
  --max-tokens 64 \
  --prompt "Explain SIMD briefly" > benchmark.json
```

Benchmark output includes prompt tokens, generated tokens, prefill time, decode
time, wall time, and aggregate throughput. Use the same model, prompt,
temperature, seed, thread count, and build flags when comparing changes.

Ollama and LM Studio are often faster on macOS because they use heavily tuned
llama.cpp kernels and GPU paths. RustyLLM benchmark numbers are most useful for
tracking RustyLLM changes against itself.

## Inspection and Utilities

Inspect compatibility without loading model weights:

```bash
rusty-llm ./models/model.gguf --inspect
```

List tensors through the main binary:

```bash
rusty-llm ./models/model.gguf --list-tensors
```

Run utility binaries:

```bash
cargo run --release --bin list_tensors -- ./models/model.gguf
cargo run --release --bin analyze_gguf -- ./models/model.gguf
```

`analyze_gguf` is currently focused on Gemma-style layer structure analysis.

## Embedding Demo

The embedding demo computes embedding vectors and compares cosine similarity:

```bash
cargo run --release --bin embedding_demo -- \
  ./models/embed.gguf \
  "Albert Einstein was a physicist." \
  "Einstein developed the theory of relativity." \
  "A banana is a tropical fruit."
```

You can also call the CLI embedding mode directly:

```bash
rusty-llm ./models/embed.gguf --embed --prompt "The quick brown fox"
```

## Library Usage

```rust
use rusty_llm::runtime::{GenerationOptions, Runner};

fn main() -> Result<(), String> {
    let (runner, _) = Runner::from_path("./models/model.gguf")?;

    let result = runner.generate("Hello", &GenerationOptions::default())?;
    println!("{}", result.text);

    let emb = runner.embed("The quick brown fox")?;
    println!("dim={} tokens={}", emb.embedding.len(), emb.token_count);

    Ok(())
}
```

Chat generation:

```rust
use rusty_llm::runtime::{ChatMessage, GenerationOptions, Runner};

fn main() -> Result<(), String> {
    let (runner, _) = Runner::from_path("./models/model.gguf")?;
    let messages = vec![
        ChatMessage::user("Explain GGUF in one sentence."),
    ];
    let result = runner.generate_chat(&messages, &GenerationOptions::default())?;
    println!("{}", result.text);
    Ok(())
}
```

Cosine similarity:

```rust
use rusty_llm::runtime::{cosine_similarity, Runner};

fn main() -> Result<(), String> {
    let (runner, _) = Runner::from_path("./models/embed.gguf")?;
    let a = runner.embed("Einstein developed relativity.")?;
    let b = runner.embed("Relativity was developed by Einstein.")?;
    println!("{:.4}", cosine_similarity(&a.embedding, &b.embedding)?);
    Ok(())
}
```

## Features and Build Profiles

Default Cargo features:

- `full`: enables the default native application feature set.
- `cli`: builds the CLI binaries and enables JSON helpers for command-line
  tools.
- `server`: enables the HTTP server.
- `tls`: enables HTTPS serving through `rustls`.
- `metal`: compiles the optional macOS Metal backend when Xcode command line
  tools are available. When compiled and the GPU backend is available, it is
  used by default at runtime (set `RUSTY_LLM_METAL=0` to opt out).

Mistral Ultra mode:

```bash
RUSTY_LLM_METAL=1 rusty-llm --model-dir ./models --model Ministral \
  --profile mistral-ultra --prompt "Explain metal inference briefly."
```

For repeatable checks, use `make bench-model-ultra MODEL=...` or
`make kernel-bench-ultra MODEL=...`. Tune the aggressive routing thresholds with
`RUSTY_LLM_METAL_ULTRA_Q4K_MIN_ROWS`, `RUSTY_LLM_METAL_ULTRA_Q6K_MIN_ROWS`, and
`RUSTY_LLM_METAL_ULTRA_ATTENTION_MIN_TOKENS`; all default to `512`. Metal
matvec and attention calls prefer Shared/NoCopy host buffers by default to avoid
extra Rust/Metal memcpy traffic; set `RUSTY_LLM_METAL_NOCOPY=0` to benchmark the
older copy-buffer path. Mistral-style Q4_K/Q4_K/Q6_K FFN blocks are fused into
one Metal command buffer by default; set `RUSTY_LLM_METAL_FUSED_FFN=0` to compare
against the older Gate/Up, CPU SwiGLU, Down sequence.

Optional feature:

- `wasm`: enables the wasm-bindgen interface and is intended for
  `wasm32-unknown-unknown` builds without default native features.

Examples:

```bash
cargo build --release --features full
cargo check --no-default-features --features cli,server,tls
cargo check --no-default-features --features wasm --target wasm32-unknown-unknown --lib
make wasm
```

The release profile uses `opt-level = 3`, fat LTO, one codegen unit, stripping,
and `panic = "abort"`. The bench profile mirrors the release optimizer while
keeping line-table debug info for profiler output. For smaller WebAssembly
artifacts, `cargo build --profile wasm-release --no-default-features --features wasm --target wasm32-unknown-unknown --lib`
uses size-oriented optimization.

## GitHub Pages WASM Demo

The browser demo lives in `demo/wasm/index.html`. Generated wasm-bindgen output
is intentionally ignored via `demo/wasm/pkg/` and should not be committed to the
main branch.

The `Deploy WASM demo` GitHub Actions workflow builds the WASM package in CI,
assembles a temporary Pages artifact, and deploys it with GitHub Pages. To use
it, configure the repository's Pages source to **GitHub Actions** in the GitHub
repository settings. The deployed page contains:

- `index.html` from `demo/wasm/index.html`
- generated `pkg/rusty_llm.js`
- generated `pkg/rusty_llm_bg.wasm`
- generated TypeScript declaration files

No generated WASM binaries are written back to the repository branch.

## Environment Variables

- `RUSTY_LLM_MODEL_DIR`: default directory used by model discovery.
- `RUSTY_LLM_FAST_ATTN`: enables the approximate fast attention path when set.
- `RUSTY_LLM_METAL`: controls the macOS Metal Q4_0/Q8_0/Q4_K/Q6_K and
  long-context attention GPU paths. When the binary was built with the `metal`
  feature and the backend compiled and is available, Metal is used by default.
  Set `RUSTY_LLM_METAL=0` to force the CPU path; `RUSTY_LLM_METAL=1` keeps it
  explicit.

## Development

Useful checks:

```bash
cargo fmt --check
cargo clippy --all-targets --features full -- -D warnings
cargo test --features full
cargo check --no-default-features --features wasm --target wasm32-unknown-unknown --lib
```

The CI workflow runs the full native check set and the no-default-features WASM
library check on Ubuntu. Local GitHub Actions runs are supported with `act`:

```bash
act pull_request
```

The repository includes `.actrc` runner mappings and skips GitHub-hosted-only
deployment/cache steps when `ACT=true`.

Focused embedding tests:

```bash
cargo test runtime::tests
```

## Notes and Limitations

- Native builds use memory mapping; WASM builds load GGUF bytes from memory.
- Generation calls are serialized inside a `Runner` to protect shared inference
  state.
- The HTTP parser is intentionally small and expects HTTP/1.1 requests with
  `Content-Length` for JSON `POST` bodies.
- Server requests have bounded header and body sizes, per-connection timeouts,
  and a configurable concurrency cap.
- Some GGUF chat templates are mapped into internal prompt renderers;
  unsupported templates fall back to a plain `System/User/Assistant` transcript.
- SSE responses do not include `Content-Length`; the stream ends with
  `data: [DONE]` and the socket closes.
- Embeddings are mean-pooled over input token positions and L2-normalized.
- Unknown model IDs sent to the API are accepted and mapped to the loaded model,
  which helps existing OpenAI, LM Studio, and RAG clients work without knowing
  the exact local model name.
- Multimodal request bodies are accepted for API compatibility, but images are
  represented as text placeholders rather than processed by a vision encoder.

## Alternatives

RustyLLM is intentionally small and learning-oriented. If you need production
throughput, GPU offloading, a polished GUI, or broader model and quantization
coverage, one of the following projects is likely a better fit.

| Project | Language | Focus |
|---|---|---|
| [llama.cpp](https://github.com/ggerganov/llama.cpp) | C/C++ | Reference implementation for GGUF inference; the origin of the GGUF format, quantization schemes, and most SIMD/GPU kernels used across the ecosystem. Highest raw throughput for CPU and GPU inference. |
| [Ollama](https://ollama.com) | Go + llama.cpp | User-friendly CLI and REST API wrapping llama.cpp; pulls models automatically and exposes the same `/api/` routes that RustyLLM emulates. Best choice when you want a local model running in one command. |
| [LM Studio](https://lmstudio.ai) | Electron + llama.cpp | Desktop GUI for discovering, downloading, and chatting with local GGUF models; includes an OpenAI-compatible local server. Best for non-developers or when a visual interface matters. |
| [mistral.rs](https://github.com/EricLBuehler/mistral.rs) | Rust | Production-grade Rust inference engine with CUDA/Metal GPU support, speculative decoding, vision models, and a Python/HTTP API. The Rust alternative to RustyLLM for real workloads. |
| [candle](https://github.com/huggingface/candle) | Rust | Hugging Face's minimalist Rust ML framework. Runs many model families from Safetensors or GGUF; designed as a library rather than a standalone runner. |
| [llamafile](https://github.com/Mozilla-Ocho/llamafile) | C/C++ | Packages a model and the llama.cpp runtime into a single cross-platform executable. Useful when you want to distribute a self-contained model binary. |
| [GPT4All](https://gpt4all.io) | C++ + Qt | Cross-platform desktop application with a chat UI and a local model store; targets end users rather than developers. |
| [koboldcpp](https://github.com/LostRuins/koboldcpp) | Python + llama.cpp | llama.cpp frontend focused on creative writing and role-play; includes a web UI and KoboldAI-compatible API routes. |
