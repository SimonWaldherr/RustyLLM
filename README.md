# RustyLLM

RustyLLM is a small Rust inference runner for GGUF language models. It focuses on local execution with memory-mapped weights, quantized matvec kernels, and a minimal CLI/HTTP surface.

## Features

- Loads GGUF models directly from disk with zero-copy memory mapping on native targets.
- Supports common decoder GGUF architectures including `llama`, `mistral`, `mistral3`, `qwen2`, `qwen3`, `phi3`, `granite`, `deepseek2`, `gemma4`, and `gpt-oss`.
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
- `--list-tensors`: print the GGUF tensor inventory and exit.

Examples:

```bash
# One-shot generation
rusty-llm ./models/model.gguf --prompt "Explain rotary embeddings" --max-tokens 128

# List models downloaded by LM Studio
rusty-llm --list-models

# Select by repository or file substring from the LM Studio cache
rusty-llm --model phi-4 --prompt "Write a Rust enum example"

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
- The HTTP parser expects HTTP/1.1 with `Content-Length` and `application/json` on `POST /generate`.
- Server requests are bounded (header/body size limits, per-connection timeouts, and concurrency cap).
- Some GGUF chat templates are mapped into internal prompt renderers; unsupported templates fall back to a plain `System/User/Assistant` transcript.
