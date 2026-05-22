# RustyLLM Architecture

RustyLLM is organized as a compact inference stack rather than a framework. The
goal is to make each inference stage visible and editable without hiding the
core mechanics behind large dependencies.

If terms such as token, tensor, embedding, or quantization are unfamiliar, read
[AI inference for non-AI developers](AI_FOR_DEVELOPERS.md) first. This document
uses those terms to describe how the modules fit together.

## Pipeline

```text
GGUF bytes
  -> gguf::GGUFFile
  -> tokenizer::Tokenizer + model::Config + model weights
  -> runtime::Runner
  -> prompt tokens
  -> model::forward*_into
  -> sampling::sample_with_scratch
  -> decoded text, embeddings, or HTTP responses
```

## Module Map

- `gguf.rs` parses the GGUF file header, metadata values, tensor descriptors,
  tensor offsets, and quantization layout sizes.
- `mmap.rs` maps native model files into memory so large quantized tensors can
  be referenced without eagerly copying the whole file.
- `tokenizer.rs` builds SentencePiece-style or GPT-2-style tokenizers from GGUF
  metadata, including byte fallback and special token IDs.
- `simd.rs` contains the educational kernel layer: plain scalar operations,
  thread-parallel matvec dispatch, quantized dot products, dequantization
  helpers, and architecture-specific acceleration hooks.
- `metal.rs` is an optional macOS acceleration shim for selected Q4_K and Q6_K
  matvec workloads. It remains optional so the CPU path stays understandable.
- `model.rs` converts GGUF tensors into typed weight structures and implements
  forward passes for the supported architecture families.
- `sampling.rs` applies repetition penalty, temperature, top-k, top-p, and
  pseudo-random token selection.
- `runtime.rs` provides the user-facing `Runner`: loading, generation,
  streaming generation, chat rendering, embeddings, session-aware generation,
  benchmark helpers, and WASM bindings.
- `session.rs` keeps bounded KV-cache sessions for HTTP chat reuse.
- `server.rs` implements the built-in HTTP server and compatibility routes.
- `catalog.rs` scans local GGUF model directories and selects models by path,
  repository name, filename, or metadata name.
- `main.rs` is the CLI entry point and connects all runtime modes.

## Educational Boundaries

RustyLLM intentionally keeps several choices explicit:

- GGUF parsing is local code, not delegated to a model runtime.
- Tensor kernels are normal Rust functions with specialized fast paths nearby.
- Quantized formats expose their byte layout in `simd.rs`.
- Chat rendering and stop handling live in `runtime.rs` rather than a hidden
  prompt framework.
- The HTTP server uses the standard library plus optional TLS support, making
  request parsing and route behavior easy to inspect.

This makes RustyLLM useful for experiments, teaching, benchmarking individual
kernel choices, and understanding inference plumbing. It is not intended to be
a replacement for high-throughput production serving systems.
