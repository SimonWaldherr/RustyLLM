# RustyLLM Benchmark Results

Updated: **2026-06-09 22:01 CEST**

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

## Summary

| Profile | Ok | Failed | Skipped/partial | Best decode | Median decode |
|---|---:|---:|---:|---:|---:|
| CPU | 11 | 0 | 3 | 12.7 | 5.0 |
| Metal GPU | 11 | 0 | 3 | 23.7 | 13.2 |

## CPU vs Metal

Each speed cell is `decode / prefill` in tokens per second. Speedup uses decode throughput.

| # | Model | Arch | Size | CPU | Metal | Metal/CPU | Result |
|---:|---|:---:|---:|---:|---:|---:|---|
| 1 | `DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf` | deepseek2 | 9884 | partial | partial | — | DeepSeek2 MLA attention tensors are present… |
| 2 | `DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf` | qwen2 | 4466 | 5.4 / 5.7 | 14.8 / 12.9 | 2.75x | Metal faster |
| 3 | `gemma-4-12B-it-QAT-Q4_0.gguf` | gemma4 | 6652 | 4.3 / 4.8 | 8.0 / 7.6 | 1.86x | Metal faster |
| 4 | `gemma-4-26B-A4B-it-Q4_K_M.gguf` | gemma4 | 16017 | 12.1 / 17.6 | 18.3 / 20.6 | 1.52x | Metal faster |
| 5 | `gemma-4-E2B-it-Q4_K_M.gguf` | gemma4 | 3269 | 12.7 / 15.5 | 16.0 / 17.1 | 1.26x | Metal faster |
| 6 | `gpt-oss-20b-MXFP4.gguf` | gpt-oss | 11548 | 2.7 / 2.8 | 2.6 / 2.5 | 0.93x | CPU faster |
| 7 | `granite-3.1-8b-instruct-Q4_K_M.gguf` | granite | 4713 | 4.6 / 4.6 | 13.2 / 11.2 | 2.89x | Metal faster |
| 8 | `granite-embedding-278m-multilingual-Q4_K_M.gguf` | bert | 208 | skip | skip | — | unsupported architecture |
| 9 | `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` | llama | 4692 | 5.0 / 5.4 | 15.1 / 13.7 | 3.03x | Metal faster |
| 10 | `Ministral-3-14B-Reasoning-2512-Q4_K_M.gguf` | mistral3 | 7857 | 3.0 / 3.2 | 9.8 / 8.2 | 3.27x | Metal faster |
| 11 | `Ministral-3-3B-Instruct-2512-Q4_K_M.gguf` | mistral3 | 2047 | 10.1 / 11.7 | 23.7 / 22.3 | 2.33x | Metal faster |
| 12 | `NVIDIA-Nemotron-3-Nano-4B-Q4_K_M.gguf` | nemotron_h | 2705 | skip | skip | — | unsupported architecture |
| 13 | `Phi-3.1-mini-128k-instruct-Q4_K_M.gguf` | phi3 | 2282 | 9.7 / 10.1 | 13.0 / 13.0 | 1.34x | Metal faster |
| 14 | `phi-4-Q4_K_M.gguf` | phi3 | 8633 | 2.8 / 2.9 | 6.1 / 5.5 | 2.19x | Metal faster |

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
| 2 | `DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf` | qwen2 | ok | 4466 | 270 | 5.4 | 5.7 |  |
| 3 | `gemma-4-12B-it-QAT-Q4_0.gguf` | gemma4 | ok | 6652 | 542 | 4.3 | 4.8 |  |
| 4 | `gemma-4-26B-A4B-it-Q4_K_M.gguf` | gemma4 | ok | 16017 | 956 | 12.1 | 17.6 |  |
| 5 | `gemma-4-E2B-it-Q4_K_M.gguf` | gemma4 | ok | 3269 | 403 | 12.7 | 15.5 |  |
| 6 | `gpt-oss-20b-MXFP4.gguf` | gpt-oss | ok | 11548 | 698 | 2.7 | 2.8 |  |
| 7 | `granite-3.1-8b-instruct-Q4_K_M.gguf` | granite | ok | 4713 | 242 | 4.6 | 4.6 |  |
| 8 | `granite-embedding-278m-multilingual-Q4_K_M.gguf` | bert | skip | 208 | — | — | — | unsupported architecture |
| 9 | `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` | llama | ok | 4692 | 324 | 5.0 | 5.4 |  |
| 10 | `Ministral-3-14B-Reasoning-2512-Q4_K_M.gguf` | mistral3 | ok | 7857 | 470 | 3.0 | 3.2 |  |
| 11 | `Ministral-3-3B-Instruct-2512-Q4_K_M.gguf` | mistral3 | ok | 2047 | 184 | 10.1 | 11.7 |  |
| 12 | `NVIDIA-Nemotron-3-Nano-4B-Q4_K_M.gguf` | nemotron_h | skip | 2705 | — | — | — | unsupported architecture |
| 13 | `Phi-3.1-mini-128k-instruct-Q4_K_M.gguf` | phi3 | ok | 2282 | 77 | 9.7 | 10.1 |  |
| 14 | `phi-4-Q4_K_M.gguf` | phi3 | ok | 8633 | 428 | 2.8 | 2.9 |  |

### CPU Decode Ranking

| Rank | Model | Decode | Prefill | Load |
|---:|---|---:|---:|---:|
| 1 | `gemma-4-E2B-it-Q4_K_M.gguf` | 12.7 | 15.5 | 403 |
| 2 | `gemma-4-26B-A4B-it-Q4_K_M.gguf` | 12.1 | 17.6 | 956 |
| 3 | `Ministral-3-3B-Instruct-2512-Q4_K_M.gguf` | 10.1 | 11.7 | 184 |
| 4 | `Phi-3.1-mini-128k-instruct-Q4_K_M.gguf` | 9.7 | 10.1 | 77 |
| 5 | `DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf` | 5.4 | 5.7 | 270 |
| 6 | `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` | 5.0 | 5.4 | 324 |
| 7 | `granite-3.1-8b-instruct-Q4_K_M.gguf` | 4.6 | 4.6 | 242 |
| 8 | `gemma-4-12B-it-QAT-Q4_0.gguf` | 4.3 | 4.8 | 542 |
| 9 | `Ministral-3-14B-Reasoning-2512-Q4_K_M.gguf` | 3.0 | 3.2 | 470 |
| 10 | `phi-4-Q4_K_M.gguf` | 2.8 | 2.9 | 428 |
| 11 | `gpt-oss-20b-MXFP4.gguf` | 2.7 | 2.8 | 698 |

### Metal GPU

| # | Model | Arch | Status | Size | Load | Decode | Prefill | Note |
|---:|---|:---:|---|---:|---:|---:|---:|---|
| 1 | `DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf` | deepseek2 | partial | 9884 | — | — | — | DeepSeek2 MLA attention tensors are present, but the runtime does not yet implement MLA |
| 2 | `DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf` | qwen2 | ok | 4466 | 273 | 14.8 | 12.9 |  |
| 3 | `gemma-4-12B-it-QAT-Q4_0.gguf` | gemma4 | ok | 6652 | 529 | 8.0 | 7.6 |  |
| 4 | `gemma-4-26B-A4B-it-Q4_K_M.gguf` | gemma4 | ok | 16017 | 899 | 18.3 | 20.6 |  |
| 5 | `gemma-4-E2B-it-Q4_K_M.gguf` | gemma4 | ok | 3269 | 395 | 16.0 | 17.1 |  |
| 6 | `gpt-oss-20b-MXFP4.gguf` | gpt-oss | ok | 11548 | 666 | 2.6 | 2.5 |  |
| 7 | `granite-3.1-8b-instruct-Q4_K_M.gguf` | granite | ok | 4713 | 222 | 13.2 | 11.2 |  |
| 8 | `granite-embedding-278m-multilingual-Q4_K_M.gguf` | bert | skip | 208 | — | — | — | unsupported architecture |
| 9 | `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` | llama | ok | 4692 | 302 | 15.1 | 13.7 |  |
| 10 | `Ministral-3-14B-Reasoning-2512-Q4_K_M.gguf` | mistral3 | ok | 7857 | 436 | 9.8 | 8.2 |  |
| 11 | `Ministral-3-3B-Instruct-2512-Q4_K_M.gguf` | mistral3 | ok | 2047 | 191 | 23.7 | 22.3 |  |
| 12 | `NVIDIA-Nemotron-3-Nano-4B-Q4_K_M.gguf` | nemotron_h | skip | 2705 | — | — | — | unsupported architecture |
| 13 | `Phi-3.1-mini-128k-instruct-Q4_K_M.gguf` | phi3 | ok | 2282 | 64 | 13.0 | 13.0 |  |
| 14 | `phi-4-Q4_K_M.gguf` | phi3 | ok | 8633 | 391 | 6.1 | 5.5 |  |

### Metal GPU Decode Ranking

| Rank | Model | Decode | Prefill | Load |
|---:|---|---:|---:|---:|
| 1 | `Ministral-3-3B-Instruct-2512-Q4_K_M.gguf` | 23.7 | 22.3 | 191 |
| 2 | `gemma-4-26B-A4B-it-Q4_K_M.gguf` | 18.3 | 20.6 | 899 |
| 3 | `gemma-4-E2B-it-Q4_K_M.gguf` | 16.0 | 17.1 | 395 |
| 4 | `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` | 15.1 | 13.7 | 302 |
| 5 | `DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf` | 14.8 | 12.9 | 273 |
| 6 | `granite-3.1-8b-instruct-Q4_K_M.gguf` | 13.2 | 11.2 | 222 |
| 7 | `Phi-3.1-mini-128k-instruct-Q4_K_M.gguf` | 13.0 | 13.0 | 64 |
| 8 | `Ministral-3-14B-Reasoning-2512-Q4_K_M.gguf` | 9.8 | 8.2 | 436 |
| 9 | `gemma-4-12B-it-QAT-Q4_0.gguf` | 8.0 | 7.6 | 529 |
| 10 | `phi-4-Q4_K_M.gguf` | 6.1 | 5.5 | 391 |
| 11 | `gpt-oss-20b-MXFP4.gguf` | 2.6 | 2.5 | 666 |

---

Re-run `bench_models.sh` to refresh this report. Use `BENCH_PROFILES=cpu` or `BENCH_PROFILES=metal` for a single backend run.
