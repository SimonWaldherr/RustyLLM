# RustyLLM

RustyLLM is a small Rust inference runner for GGUF language models. It focuses on local execution with memory-mapped weights, quantized matvec kernels, and a minimal CLI/HTTP surface.

## Features

- Loads GGUF models directly from disk with zero-copy memory mapping on native targets.
- Supports GGUF architectures including `llama`/`llama2`/`llama3`, `mistral`/`mixtral`/`ministral`, `qwen2`, `gpt-oss`, `gemma`/`gemma2`/`gemma4`, `deepseek` variants, `nemotron`, `hermes`, `phi` variants, and `nomic` embeddings.
- Handles SentencePiece and GPT-2 style tokenizers from GGUF metadata.
- Runs quantized inference paths for `Q8_0`, `Q4_0`, `Q4_K`, `Q6_K`, and `MXFP4` tensors.
- Uses native SIMD backends on Apple Silicon and AVX2/FMA-capable x86_64 systems, with scalar fallback elsewhere.
- Exposes three entry points: one-shot generation, interactive REPL, and a small HTTP(S) API.
- Builds as both a library and a CLI binary.

## Build

```bash
cargo build --release
```

The binary will be available at `target/release/rusty-llm`.

## CLI Usage

```bash
cargo run --release -- /path/to/model.gguf --prompt "Write a short poem"
```

General form:

```bash
rusty-llm <model.gguf> [options]
```

Options:

- `--prompt <text>`: generate from a single prompt.
- `--repl`: start an interactive chat session.
- `--serve <addr>`: start the HTTP(S) server, for example `127.0.0.1:8080`.
- `--tls-cert <path>`: PEM certificate for HTTPS.
- `--tls-key <path>`: PEM private key for HTTPS.
- `--max-tokens <N>`: maximum generated tokens.
- `--temp <F>`: temperature, where `0` switches to greedy decoding.
- `--top-p <F>`: nucleus sampling threshold.
- `--top-k <N>`: top-k filtering limit.
- `--repeat-penalty <F>`: repetition penalty applied to recent tokens.
- `--seed <N>`: deterministic RNG seed.
- `--threads <N>`: override worker thread count for SIMD kernels.
- `--system-prompt <text>`: override the default chat system prompt.
- `--list-tensors`: print the GGUF tensor inventory and exit.

Examples:

```bash
# One-shot generation
rusty-llm ./models/model.gguf --prompt "Explain rotary embeddings" --max-tokens 128

# Interactive mode
rusty-llm ./models/model.gguf --repl

# Inspect tensor names and dtypes
rusty-llm ./models/model.gguf --list-tensors
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
- `POST /v1/completions` (OpenAI-compatible)
- `POST /v1/chat/completions` (OpenAI-compatible)

Prompt request example:

```bash
curl -X POST http://127.0.0.1:8080/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "Summarize grouped-query attention in two sentences.",
    "max_tokens": 80,
    "temp": 0.7,
    "top_p": 0.9,
    "top_k": 40,
    "repeat_penalty": 1.1
  }'
```

Chat request example:

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

OpenAI chat completion example:

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
    "temperature": 0.7
  }'
```

## Library Usage

```rust
use rusty_llm::runtime::{GenerationOptions, Runner};

fn main() -> Result<(), String> {
    let (runner, _) = Runner::from_path("./models/model.gguf")?;
    let result = runner.generate("Hello", &GenerationOptions::default())?;
    println!("{}", result.text);
    Ok(())
}
```

## Notes

- Native builds use memory mapping; WASM builds load from in-memory GGUF bytes.
- The HTTP parser is intentionally minimal and expects `Content-Length` requests.
- Some GGUF chat templates are mapped into internal prompt renderers; unsupported templates fall back to a plain `System/User/Assistant` transcript.
