# RustyLLM Benchmark Results

Updated: **2026-07-16 22:27 CEST**

This report compares the CPU path with the optional Apple Metal GPU path. Metal here means GPU acceleration through RustyLLM's Metal kernels; it is not a CoreML, ANE, or NPU backend.

## Run Configuration

| Setting | Value |
|---|---|
| Model directory | `/Users/simonwaldherr/.cache/lm-studio/models/lmstudio-community` |
| Prompt | Explain local LLM inference performance in one concise paragraph. |
| Runs | 2 x 64 generated tokens per model |
| Pause | 8 seconds between models |
| Raw JSON | `.bench_raw/<profile>/` |

## Hardware

| Component | Value |
|---|---|
| CPU | Apple M2 Max |
| Logical cores | 12 |
| RAM | 32 GB |
| OS | macOS 26.5.1 |
| Rust | rustc 1.95.0 (59807616e 2026-04-14) |
| SIMD | ARM NEON (native) |

## Backend Profiles

| Profile | Env | Runtime report | Raw JSON |
|---|---|---|---|
| CPU | `RUSTY_LLM_METAL=0` | Metal: disabled by RUSTY_LLM_METAL | `.bench_raw/cpu/` |
| Metal GPU | `RUSTY_LLM_METAL=1` | Metal: Q4_K matvec enabled, Q6_K output matvec enabled | `.bench_raw/metal/` |

## Embedding Supplement — 2026-07-16

Warm end-to-end `/v1/embeddings` measurements for
`nomic-embed-text-v1.5.Q4_K_M.gguf` (80.2 MiB, 768 dimensions). The optimized
release build used `-C target-cpu=native`, 12 workers, two warm-ups, and ten
measured requests per cell. Latency includes local loopback HTTP, tokenization,
encoder forward, pooling, and response serialization. The before values are the
previous `af6af2f` measurement; current values use the token/row-batched
K-quant encoder scheduler. Each current value is the median; throughput uses
the unrounded latency. Because the earlier baseline used a different sample
count, the reported speedups are directional rather than a strict A/B result.

| Input tokens | CPU before | CPU current | Metal-enabled profile (CPU encoder) | CPU speedup |
|---:|---:|---:|---:|---:|
| 16 | 64.3 ms / 248.7 tok/s | **20.1 ms / 795 tok/s** | **19.9 ms / 803 tok/s** | 3.19x |
| 222 | 997.8 ms / 222.5 tok/s | **222.4 ms / 998 tok/s** | **223.8 ms / 992 tok/s** | 4.49x |

`RUSTY_LLM_METAL=1` was enabled for the last column, but these encoder requests
deliberately submit no Metal work once the batch path is selected: Q5_K QKV has
no suitable Metal batch kernel, and synchronous per-token GPU dispatches were
slower. This is therefore a Metal-enabled runtime-profile measurement, not a
GPU-throughput claim; both profiles are within about 1% end-to-end.

## Summary

| Profile | Ok | Failed | Skipped/partial | Best decode | Median decode |
|---|---:|---:|---:|---:|---:|
| CPU | 11 | 0 | 3 | 32.4 | 17.5 |
| Metal GPU | 11 | 0 | 3 | 48.4 | 19.7 |

## CPU vs Metal

Each speed cell is `decode / prefill` in tokens per second. Speedup uses decode throughput.

| # | Model | Arch | Size | CPU | Metal | Metal/CPU | Result |
|---:|---|:---:|---:|---:|---:|---:|---|
| 1 | `DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf` | deepseek2 | 9884 | partial | partial | — | DeepSeek2 MLA attention tensors are present… |
| 2 | `DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf` | qwen2 | 4466 | 17.5 / 19.1 | 28.3 / 19.9 | 1.62x | Metal faster |
| 3 | `gemma-4-12B-it-QAT-Q4_0.gguf` | gemma4 | 6652 | 5.7 / 6.0 | 8.0 / 7.2 | 1.39x | Metal faster |
| 4 | `gemma-4-26B-A4B-it-Q4_K_M.gguf` | gemma4 | 16017 | 23.8 / 30.7 | 19.7 / 20.2 | 0.83x | CPU faster |
| 5 | `gemma-4-E2B-it-Q4_K_M.gguf` | gemma4 | 3269 | 27.6 / 29.9 | 22.0 / 23.2 | 0.80x | CPU faster |
| 6 | `gpt-oss-20b-MXFP4.gguf` | gpt-oss | 11548 | 2.7 / 2.7 | 2.5 / 2.5 | 0.93x | CPU faster |
| 7 | `granite-3.1-8b-instruct-Q4_K_M.gguf` | granite | 4713 | 16.3 / 16.5 | 23.8 / 17.4 | 1.47x | Metal faster |
| 8 | `granite-embedding-278m-multilingual-Q4_K_M.gguf` | bert | 208 | skip | skip | — | unsupported architecture |
| 9 | `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` | llama | 4692 | 18.0 / 19.2 | 26.2 / 21.0 | 1.46x | Metal faster |
| 10 | `Ministral-3-14B-Reasoning-2512-Q4_K_M.gguf` | mistral3 | 7857 | 11.9 / 12.3 | 15.8 / 10.7 | 1.33x | Metal faster |
| 11 | `Ministral-3-3B-Instruct-2512-Q4_K_M.gguf` | mistral3 | 2047 | 32.4 / 35.7 | 48.4 / 42.2 | 1.49x | Metal faster |
| 12 | `NVIDIA-Nemotron-3-Nano-4B-Q4_K_M.gguf` | nemotron_h | 2705 | skip | skip | — | unsupported architecture |
| 13 | `Phi-3.1-mini-128k-instruct-Q4_K_M.gguf` | phi3 | 2282 | 23.0 / 24.4 | 16.6 / 15.7 | 0.72x | CPU faster |
| 14 | `phi-4-Q4_K_M.gguf` | phi3 | 8633 | 9.0 / 9.0 | 7.8 / 6.3 | 0.87x | CPU faster |

## Support Issues

| Profile | Model | Arch | Status | Reason |
|---|---|:---:|---|---|
| CPU | `DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf` | deepseek2 | partial | DeepSeek2 MLA attention tensors are present, but the runtime does not yet implement MLA |
| CPU | `granite-embedding-278m-multilingual-Q4_K_M.gguf` | bert | skip | unsupported architecture |
| CPU | `NVIDIA-Nemotron-3-Nano-4B-Q4_K_M.gguf` | nemotron_h | skip | unsupported architecture |
| Metal GPU | `DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf` | deepseek2 | partial | DeepSeek2 MLA attention tensors are present, but the runtime does not yet implement MLA |
| Metal GPU | `granite-embedding-278m-multilingual-Q4_K_M.gguf` | bert | skip | unsupported architecture |
| Metal GPU | `NVIDIA-Nemotron-3-Nano-4B-Q4_K_M.gguf` | nemotron_h | skip | unsupported architecture |

## Profile Details

### CPU

| # | Model | Arch | Status | Size | Load | Decode | Prefill | Note |
|---:|---|:---:|---|---:|---:|---:|---:|---|
| 1 | `DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf` | deepseek2 | partial | 9884 | — | — | — | DeepSeek2 MLA attention tensors are present, but the runtime does not yet implement MLA |
| 2 | `DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf` | qwen2 | ok | 4466 | 345 | 17.5 | 19.1 |  |
| 3 | `gemma-4-12B-it-QAT-Q4_0.gguf` | gemma4 | ok | 6652 | 590 | 5.7 | 6.0 |  |
| 4 | `gemma-4-26B-A4B-it-Q4_K_M.gguf` | gemma4 | ok | 16017 | 976 | 23.8 | 30.7 |  |
| 5 | `gemma-4-E2B-it-Q4_K_M.gguf` | gemma4 | ok | 3269 | 400 | 27.6 | 29.9 |  |
| 6 | `gpt-oss-20b-MXFP4.gguf` | gpt-oss | ok | 11548 | 705 | 2.7 | 2.7 |  |
| 7 | `granite-3.1-8b-instruct-Q4_K_M.gguf` | granite | ok | 4713 | 233 | 16.3 | 16.5 |  |
| 8 | `granite-embedding-278m-multilingual-Q4_K_M.gguf` | bert | skip | 208 | — | — | — | unsupported architecture |
| 9 | `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` | llama | ok | 4692 | 327 | 18.0 | 19.2 |  |
| 10 | `Ministral-3-14B-Reasoning-2512-Q4_K_M.gguf` | mistral3 | ok | 7857 | 468 | 11.9 | 12.3 |  |
| 11 | `Ministral-3-3B-Instruct-2512-Q4_K_M.gguf` | mistral3 | ok | 2047 | 213 | 32.4 | 35.7 |  |
| 12 | `NVIDIA-Nemotron-3-Nano-4B-Q4_K_M.gguf` | nemotron_h | skip | 2705 | — | — | — | unsupported architecture |
| 13 | `Phi-3.1-mini-128k-instruct-Q4_K_M.gguf` | phi3 | ok | 2282 | 74 | 23.0 | 24.4 |  |
| 14 | `phi-4-Q4_K_M.gguf` | phi3 | ok | 8633 | 433 | 9.0 | 9.0 |  |

### CPU Decode Ranking

| Rank | Model | Decode | Prefill | Load |
|---:|---|---:|---:|---:|
| 1 | `Ministral-3-3B-Instruct-2512-Q4_K_M.gguf` | 32.4 | 35.7 | 213 |
| 2 | `gemma-4-E2B-it-Q4_K_M.gguf` | 27.6 | 29.9 | 400 |
| 3 | `gemma-4-26B-A4B-it-Q4_K_M.gguf` | 23.8 | 30.7 | 976 |
| 4 | `Phi-3.1-mini-128k-instruct-Q4_K_M.gguf` | 23.0 | 24.4 | 74 |
| 5 | `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` | 18.0 | 19.2 | 327 |
| 6 | `DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf` | 17.5 | 19.1 | 345 |
| 7 | `granite-3.1-8b-instruct-Q4_K_M.gguf` | 16.3 | 16.5 | 233 |
| 8 | `Ministral-3-14B-Reasoning-2512-Q4_K_M.gguf` | 11.9 | 12.3 | 468 |
| 9 | `phi-4-Q4_K_M.gguf` | 9.0 | 9.0 | 433 |
| 10 | `gemma-4-12B-it-QAT-Q4_0.gguf` | 5.7 | 6.0 | 590 |
| 11 | `gpt-oss-20b-MXFP4.gguf` | 2.7 | 2.7 | 705 |

### Metal GPU

| # | Model | Arch | Status | Size | Load | Decode | Prefill | Note |
|---:|---|:---:|---|---:|---:|---:|---:|---|
| 1 | `DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf` | deepseek2 | partial | 9884 | — | — | — | DeepSeek2 MLA attention tensors are present, but the runtime does not yet implement MLA |
| 2 | `DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf` | qwen2 | ok | 4466 | 275 | 28.3 | 19.9 |  |
| 3 | `gemma-4-12B-it-QAT-Q4_0.gguf` | gemma4 | ok | 6652 | 528 | 8.0 | 7.2 |  |
| 4 | `gemma-4-26B-A4B-it-Q4_K_M.gguf` | gemma4 | ok | 16017 | 900 | 19.7 | 20.2 |  |
| 5 | `gemma-4-E2B-it-Q4_K_M.gguf` | gemma4 | ok | 3269 | 352 | 22.0 | 23.2 |  |
| 6 | `gpt-oss-20b-MXFP4.gguf` | gpt-oss | ok | 11548 | 659 | 2.5 | 2.5 |  |
| 7 | `granite-3.1-8b-instruct-Q4_K_M.gguf` | granite | ok | 4713 | 234 | 23.8 | 17.4 |  |
| 8 | `granite-embedding-278m-multilingual-Q4_K_M.gguf` | bert | skip | 208 | — | — | — | unsupported architecture |
| 9 | `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` | llama | ok | 4692 | 299 | 26.2 | 21.0 |  |
| 10 | `Ministral-3-14B-Reasoning-2512-Q4_K_M.gguf` | mistral3 | ok | 7857 | 434 | 15.8 | 10.7 |  |
| 11 | `Ministral-3-3B-Instruct-2512-Q4_K_M.gguf` | mistral3 | ok | 2047 | 167 | 48.4 | 42.2 |  |
| 12 | `NVIDIA-Nemotron-3-Nano-4B-Q4_K_M.gguf` | nemotron_h | skip | 2705 | — | — | — | unsupported architecture |
| 13 | `Phi-3.1-mini-128k-instruct-Q4_K_M.gguf` | phi3 | ok | 2282 | 63 | 16.6 | 15.7 |  |
| 14 | `phi-4-Q4_K_M.gguf` | phi3 | ok | 8633 | 394 | 7.8 | 6.3 |  |

### Metal GPU Decode Ranking

| Rank | Model | Decode | Prefill | Load |
|---:|---|---:|---:|---:|
| 1 | `Ministral-3-3B-Instruct-2512-Q4_K_M.gguf` | 48.4 | 42.2 | 167 |
| 2 | `DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf` | 28.3 | 19.9 | 275 |
| 3 | `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` | 26.2 | 21.0 | 299 |
| 4 | `granite-3.1-8b-instruct-Q4_K_M.gguf` | 23.8 | 17.4 | 234 |
| 5 | `gemma-4-E2B-it-Q4_K_M.gguf` | 22.0 | 23.2 | 352 |
| 6 | `gemma-4-26B-A4B-it-Q4_K_M.gguf` | 19.7 | 20.2 | 900 |
| 7 | `Phi-3.1-mini-128k-instruct-Q4_K_M.gguf` | 16.6 | 15.7 | 63 |
| 8 | `Ministral-3-14B-Reasoning-2512-Q4_K_M.gguf` | 15.8 | 10.7 | 434 |
| 9 | `gemma-4-12B-it-QAT-Q4_0.gguf` | 8.0 | 7.2 | 528 |
| 10 | `phi-4-Q4_K_M.gguf` | 7.8 | 6.3 | 394 |
| 11 | `gpt-oss-20b-MXFP4.gguf` | 2.5 | 2.5 | 659 |

---

Re-run `bench_models.sh` to refresh this report. Use `BENCH_PROFILES=cpu` or `BENCH_PROFILES=metal` for a single backend run.
