# RustyLLM

RustyLLM is a small Rust inference runner for GGUF language models. It focuses on local execution with memory-mapped weights, quantized matvec kernels, and a minimal CLI/HTTP surface.

## Features

- Loads GGUF models directly from disk with zero-copy memory mapping on native targets.
- Supports GGUF architectures including `llama`/`llama2`/`llama3`, `mistral`/`mixtral`/`ministral`, `qwen2`/`qwen3`, `gpt-oss`, `gemma`/`gemma2`/`gemma4`, `deepseek` variants, `nemotron`, `hermes`, `phi`/`phi2`/`phi3`/`phi4`, `falcon`/`falcon3`, `stablelm`, `starcoder2`, `command-r`/`cohere`, `internlm2`, `olmo`/`olmo2`, `exaone`, `solar`, `yi`, `arctic`, and `nomic` embeddings.
- Handles SentencePiece and GPT-2 style tokenizers from GGUF metadata.
- Runs quantized inference paths for `Q8_0`, `Q4_0`, `Q4_K`, `Q6_K`, and `MXFP4` tensors.
- Uses native SIMD backends on Apple Silicon and AVX2/FMA-capable x86_64 systems, with scalar fallback elsewhere.
- Exposes three entry points: one-shot generation, interactive REPL, and a small HTTP(S) API.
- **Text embedding** (`Runner::embed`) for RAG retrieval: mean-pools the last transformer layer, L2-normalises, and returns a dense vector.
- **Multimodal message format** (OpenAI vision API): accepts `content` arrays with `text` and `image_url` parts; image references are described as `[image: ...]` placeholders.
- **Stop sequences**: generation halts as soon as any configured stop string appears in the output.
- **Streaming SSE**: `/v1/chat/completions` and `/v1/completions` honour `"stream": true` and emit Server-Sent Events.
- **`/v1/embeddings`** endpoint (OpenAI-compatible) for RAG pipelines.
- `max_completion_tokens` alias for `max_tokens` (OpenAI spec ≥ 2024-10).
- Lenient model-ID resolution: unknown model names are silently mapped to the loaded model, so RAG pipelines can send any model name.
- Builds as both a library and a CLI binary.

## Build

```bash
cargo build --release
```

The binary will be available at `target/release/rusty-llm`.

For local performance testing on Apple Silicon, prefer the release binary and
explicit thread counts:

```bash
cargo build --release
./target/release/rusty-llm --model phi-4 --bench --bench-runs 3 --max-tokens 64 --threads 12
```

Ollama and LM Studio are usually faster on macOS because they run llama.cpp with
Metal GPU acceleration and heavily tuned kernels. RustyLLM currently runs CPU
inference only, so the benchmark numbers are most useful for tracking RustyLLM
changes against itself.

## CLI Usage

```bash
cargo run --release --bin rusty-llm -- /path/to/model.gguf --prompt "Write a short poem"
```

General form:

```bash
rusty-llm [model.gguf|model-name|model-dir] [options]
```

Options:

- `--model <name>`: select a GGUF from `--model-dir` by repository, file name, or GGUF metadata name.
- `--model-dir <path>`: directory to recursively scan for GGUF files. Defaults to `$RUSTY_LLM_MODEL_DIR` or the LM Studio community cache under `~/.cache/lm-studio/models/lmstudio-community`.
- `--list-models`: list GGUF files in `--model-dir` and exit.
- `--prompt <text>`: generate from a single prompt.
- `--repl`: start an interactive chat session.
- `--serve <addr>`: start the HTTP(S) server, for example `127.0.0.1:8080`.
- `--tls-cert <path>`: PEM certificate for HTTPS.
- `--tls-key <path>`: PEM private key for HTTPS.
- `--max-connections <N>`: max concurrent server connections (default: `max(16, threads*8)`).
- `--max-tokens <N>`: maximum generated tokens.
- `--temp <F>`: temperature, where `0` switches to greedy decoding.
- `--top-p <F>`: nucleus sampling threshold.
- `--top-k <N>`: top-k filtering limit.
- `--repeat-penalty <F>`: repetition penalty applied to recent tokens.
- `--seed <N>`: deterministic RNG seed.
- `--threads <N>`: override worker thread count for SIMD kernels.
- `--system-prompt <text>`: override the default chat system prompt.
- `--stop <text>`: stop generation when this string appears in the output (can be repeated).
- `--embed`: run the prompt through the model and print the L2-normalised embedding vector instead of generating text.
- `--bench`: run a non-streaming generation benchmark and print per-run throughput.
- `--bench-runs <N>`: number of benchmark runs (default: `3`).
- `--list-tensors`: print the GGUF tensor inventory and exit.

Examples:

```bash
# One-shot generation
rusty-llm ./models/model.gguf --prompt "Explain rotary embeddings" --max-tokens 128

# List models downloaded by LM Studio
rusty-llm --list-models

# Select by repository or file substring from the LM Studio cache
rusty-llm --model phi-4 --prompt "Write a Rust enum example"

# Stop generation at a custom delimiter
rusty-llm ./models/model.gguf --prompt "Name three fruits:" --stop "\n" --max-tokens 32

# Embed text for RAG retrieval
rusty-llm ./models/embed.gguf --embed --prompt "The quick brown fox"

# Benchmark decode throughput
rusty-llm --model phi-4 --bench --bench-runs 5 --max-tokens 64 --threads 8

# Local LM Studio Ministral smoke test
./target/release/rusty-llm "$HOME/.cache/lm-studio/models/lmstudio-community/Ministral-3-14B-Reasoning-2512-GGUF/Ministral-3-14B-Reasoning-2512-Q4_K_M.gguf" \
  --prompt "Wer war Albert Einstein?"

# Same benchmark through the Makefile release build
make bench MODEL=/path/to/model.gguf BENCH_RUNS=5 PROMPT="Explain SIMD briefly"

# Interactive mode
rusty-llm ./models/model.gguf --repl

# Inspect tensor names and dtypes
rusty-llm ./models/model.gguf --list-tensors
```

Embedding demo (cosine similarity check with three texts):

```bash
cargo run --release --bin embedding_demo -- \
  "$HOME/.cache/lm-studio/models/lmstudio-community/Ministral-3-14B-Reasoning-2512-GGUF/Ministral-3-14B-Reasoning-2512-Q4_K_M.gguf" \
  "Albert Einstein was a physicist." \
  "Einstein developed the theory of relativity." \
  "A banana is a tropical fruit."
```

Embedding-focused tests:

```bash
cargo test runtime::tests
```

## HTTP API

Start the server:

```bash
rusty-llm ./models/model.gguf --serve 127.0.0.1:8080
```

Available routes:

- `GET /health`
- `POST /generate`
- `GET /v1/models` (OpenAI-compatible)
- `POST /v1/completions` (OpenAI-compatible, streaming supported)
- `POST /v1/chat/completions` (OpenAI-compatible, streaming and multimodal content supported)
- `POST /v1/embeddings` (OpenAI-compatible, for RAG)

### Prompt generation

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

### Chat generation

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

Response shape:

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

### OpenAI-compatible chat (non-streaming)

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

### OpenAI-compatible chat (streaming SSE)

```bash
curl -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "llama3",
    "messages": [{"role": "user", "content": "Tell me a joke."}],
    "max_tokens": 128,
    "stream": true
  }'
```

Each SSE chunk has the form:

```
data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":...,"model":"llama3","choices":[{"index":0,"delta":{"content":"..."},"finish_reason":null}]}
```

The final event is `data: [DONE]`.

### Multimodal messages

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

Image URL references are described as `[image: <url>]` in the prompt.  A full
vision encoder pipeline can extend this by processing the image bytes before
sending the request.

### Embeddings (RAG)

```bash
curl -X POST http://127.0.0.1:8080/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "nomic-embed",
    "input": "The quick brown fox jumps over the lazy dog"
  }'
```

Batch embedding:

```bash
curl -X POST http://127.0.0.1:8080/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "nomic-embed",
    "input": ["First sentence.", "Second sentence."]
  }'
```

Response shape (OpenAI-compatible):

```json
{
  "object": "list",
  "data": [
    {"object": "embedding", "embedding": [0.012, -0.034, ...], "index": 0}
  ],
  "model": "nomic-embed",
  "usage": {"prompt_tokens": 9, "total_tokens": 9}
}
```

## Library Usage

```rust
use rusty_llm::runtime::{GenerationOptions, Runner};

fn main() -> Result<(), String> {
    let (runner, _) = Runner::from_path("./models/model.gguf")?;

    // Text generation
    let result = runner.generate("Hello", &GenerationOptions::default())?;
    println!("{}", result.text);

    // Embedding (for RAG)
    let emb = runner.embed("The quick brown fox")?;
    println!("dim={} tokens={}", emb.embedding.len(), emb.token_count);

    Ok(())
}
```

## Notes

- Native builds use memory mapping; WASM builds load from in-memory GGUF bytes.
- The HTTP parser expects HTTP/1.1 with `Content-Length` and `application/json` on `POST /generate`.
- Server requests are bounded (header/body size limits, per-connection timeouts, and concurrency cap).
- Some GGUF chat templates are mapped into internal prompt renderers; unsupported templates fall back to a plain `System/User/Assistant` transcript.
- SSE streaming does not include `Content-Length`; the stream ends when the socket closes.
- Embeddings are mean-pooled across all input token positions and L2-normalised, making them suitable for cosine similarity comparisons.
- Unknown model IDs sent to the API are silently accepted and mapped to the loaded model, so existing RAG pipelines don't need to know the exact model name.
