# RustyLLM Benchmark Results

Updated: **2026-05-23 23:05 CEST**

This report compares the CPU path with the optional Apple Metal GPU path. Metal here means GPU acceleration through RustyLLM's Metal kernels; it is not a CoreML, ANE, or NPU backend.

## Run Configuration

| Setting | Value |
|---|---|
| Model directory | `/Users/simonwaldherr/.cache/lm-studio/models` |
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
| OS | macOS 26.5 |
| Rust | rustc 1.85.0 (4d91de4e4 2025-02-17) |
| SIMD | ARM NEON (native) |

## Backend Profiles

| Profile | Env | Runtime report | Raw JSON |
|---|---|---|---|
| CPU | `RUSTY_LLM_METAL=0` | Metal: disabled by RUSTY_LLM_METAL | `.bench_raw/cpu/` |
| Metal GPU | `RUSTY_LLM_METAL=1` | Metal: Q4_K matvec enabled, Q6_K output matvec enabled | `.bench_raw/metal/` |

## Summary

| Profile | Ok | Failed | Skipped/partial | Best decode | Median decode |
|---|---:|---:|---:|---:|---:|
| CPU | 16 | 0 | 3 | 14.5 | 5.0 |
| Metal GPU | 16 | 0 | 3 | 17.9 | 10.3 |

## CPU vs Metal

Each speed cell is `decode / prefill` in tokens per second. Speedup uses decode throughput.

| # | Model | Arch | Size | CPU | Metal | Metal/CPU | Result |
|---:|---|:---:|---:|---:|---:|---:|---|
| 1 | `Hermes-3-Llama-3.2-3B.Q4_K_M.gguf` | llama | 1925 | 10.6 / 12.4 | 16.7 / 16.3 | 1.57x | Metal faster |
| 2 | `deepseek-math-7b-instruct.Q4_0.gguf` | llama | 3814 | 6.8 / 7.8 | 6.6 / 7.1 | 0.97x | similar |
| 3 | `DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf` | deepseek4_mtp_support | 3631 | skip | skip | — | unsupported architecture |
| 4 | `MiniMaxAI_MiniMax-M2.7-imatrix.gguf` | unknown | 469 | skip | skip | — | unsupported architecture |
| 5 | `openai_gpt-oss-20b-MXFP4.gguf` | gpt-oss | 11548 | 2.6 / 2.6 | 2.6 / 2.7 | 1.01x | similar |
| 6 | `stable-code-instruct-3b-Q4_0.gguf` | stablelm | 1534 | 14.5 / 15.6 | 14.3 / 15.2 | 0.98x | similar |
| 7 | `internlm2_5-20b-chat-q4_0.gguf` | internlm2 | 10798 | 2.5 / 2.6 | 2.5 / 2.5 | 1.00x | similar |
| 8 | `DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf` | deepseek2 | 9884 | partial | partial | — | DeepSeek2 MLA attention tensors are present… |
| 9 | `DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf` | qwen2 | 4466 | 5.2 / 5.6 | 12.3 / 10.9 | 2.36x | Metal faster |
| 10 | `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` | llama | 4692 | 4.8 / 5.2 | 12.9 / 11.6 | 2.66x | Metal faster |
| 11 | `Ministral-3-14B-Reasoning-2512-Q4_K_M.gguf` | mistral3 | 7857 | 2.7 / 3.0 | 8.9 / 3.8 | 3.25x | Metal faster |
| 12 | `Ministral-3-3B-Instruct-2512-Q4_K_M.gguf` | mistral3 | 2047 | 9.7 / 9.1 | 17.9 / 18.1 | 1.84x | Metal faster |
| 13 | `Phi-3.1-mini-128k-instruct-Q4_K_M.gguf` | phi3 | 2282 | 9.0 / 9.4 | 12.3 / 12.2 | 1.37x | Metal faster |
| 14 | `gemma-4-26B-A4B-it-Q4_K_M.gguf` | gemma4 | 16017 | 11.3 / 12.0 | 13.2 / 10.0 | 1.16x | Metal faster |
| 15 | `gpt-oss-20b-MXFP4.gguf` | gpt-oss | 11548 | 2.7 / 2.6 | 2.6 / 2.7 | 0.99x | similar |
| 16 | `granite-3.1-8b-instruct-Q4_K_M.gguf` | granite | 4713 | 4.4 / 4.6 | 11.6 / 9.8 | 2.63x | Metal faster |
| 17 | `phi-4-Q4_K_M.gguf` | phi3 | 8633 | 2.6 / 2.7 | 6.9 / 3.4 | 2.63x | Metal faster |
| 18 | `Teuken-7B-instruct-commercial-v0.4.Q4_K_S.gguf` | llama | 4480 | 3.0 / 3.3 | 3.7 / 3.6 | 1.24x | Metal faster |
| 19 | `llava-v1.5-7b-Q4_0.gguf` | llama | 3648 | 6.9 / 7.4 | 6.9 / 7.1 | 1.00x | similar |

## Support Issues

| Profile | Model | Arch | Status | Reason |
|---|---|:---:|---|---|
| CPU | `DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf` | deepseek4_mtp_support | skip | unsupported architecture |
| CPU | `MiniMaxAI_MiniMax-M2.7-imatrix.gguf` | unknown | skip | unsupported architecture |
| CPU | `DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf` | deepseek2 | partial | DeepSeek2 MLA attention tensors are present, but the runtime does not yet implement MLA |
| Metal GPU | `DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf` | deepseek4_mtp_support | skip | unsupported architecture |
| Metal GPU | `MiniMaxAI_MiniMax-M2.7-imatrix.gguf` | unknown | skip | unsupported architecture |
| Metal GPU | `DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf` | deepseek2 | partial | DeepSeek2 MLA attention tensors are present, but the runtime does not yet implement MLA |

## Profile Details

### CPU

| # | Model | Arch | Status | Size | Load | Decode | Prefill | Note |
|---:|---|:---:|---|---:|---:|---:|---:|---|
| 1 | `Hermes-3-Llama-3.2-3B.Q4_K_M.gguf` | llama | ok | 1925 | 152 | 10.6 | 12.4 |  |
| 2 | `deepseek-math-7b-instruct.Q4_0.gguf` | llama | ok | 3814 | 151 | 6.8 | 7.8 |  |
| 3 | `DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf` | deepseek4_mtp_support | skip | 3631 | — | — | — | unsupported architecture |
| 4 | `MiniMaxAI_MiniMax-M2.7-imatrix.gguf` | unknown | skip | 469 | — | — | — | unsupported architecture |
| 5 | `openai_gpt-oss-20b-MXFP4.gguf` | gpt-oss | ok | 11548 | 1133 | 2.6 | 2.6 |  |
| 6 | `stable-code-instruct-3b-Q4_0.gguf` | stablelm | ok | 1534 | 71 | 14.5 | 15.6 |  |
| 7 | `internlm2_5-20b-chat-q4_0.gguf` | internlm2 | ok | 10798 | 599 | 2.5 | 2.6 |  |
| 8 | `DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf` | deepseek2 | partial | 9884 | — | — | — | DeepSeek2 MLA attention tensors are present, but the runtime does not yet implement MLA |
| 9 | `DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf` | qwen2 | ok | 4466 | 276 | 5.2 | 5.6 |  |
| 10 | `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` | llama | ok | 4692 | 312 | 4.8 | 5.2 |  |
| 11 | `Ministral-3-14B-Reasoning-2512-Q4_K_M.gguf` | mistral3 | ok | 7857 | 537 | 2.7 | 3.0 |  |
| 12 | `Ministral-3-3B-Instruct-2512-Q4_K_M.gguf` | mistral3 | ok | 2047 | 222 | 9.7 | 9.1 |  |
| 13 | `Phi-3.1-mini-128k-instruct-Q4_K_M.gguf` | phi3 | ok | 2282 | 101 | 9.0 | 9.4 |  |
| 14 | `gemma-4-26B-A4B-it-Q4_K_M.gguf` | gemma4 | ok | 16017 | 7751 | 11.3 | 12.0 |  |
| 15 | `gpt-oss-20b-MXFP4.gguf` | gpt-oss | ok | 11548 | 1261 | 2.7 | 2.6 |  |
| 16 | `granite-3.1-8b-instruct-Q4_K_M.gguf` | granite | ok | 4713 | 277 | 4.4 | 4.6 |  |
| 17 | `phi-4-Q4_K_M.gguf` | phi3 | ok | 8633 | 471 | 2.6 | 2.7 |  |
| 18 | `Teuken-7B-instruct-commercial-v0.4.Q4_K_S.gguf` | llama | ok | 4480 | 286 | 3.0 | 3.3 |  |
| 19 | `llava-v1.5-7b-Q4_0.gguf` | llama | ok | 3648 | 184 | 6.9 | 7.4 |  |

### CPU Decode Ranking

| Rank | Model | Decode | Prefill | Load |
|---:|---|---:|---:|---:|
| 1 | `stable-code-instruct-3b-Q4_0.gguf` | 14.5 | 15.6 | 71 |
| 2 | `gemma-4-26B-A4B-it-Q4_K_M.gguf` | 11.3 | 12.0 | 7751 |
| 3 | `Hermes-3-Llama-3.2-3B.Q4_K_M.gguf` | 10.6 | 12.4 | 152 |
| 4 | `Ministral-3-3B-Instruct-2512-Q4_K_M.gguf` | 9.7 | 9.1 | 222 |
| 5 | `Phi-3.1-mini-128k-instruct-Q4_K_M.gguf` | 9.0 | 9.4 | 101 |
| 6 | `llava-v1.5-7b-Q4_0.gguf` | 6.9 | 7.4 | 184 |
| 7 | `deepseek-math-7b-instruct.Q4_0.gguf` | 6.8 | 7.8 | 151 |
| 8 | `DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf` | 5.2 | 5.6 | 276 |
| 9 | `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` | 4.8 | 5.2 | 312 |
| 10 | `granite-3.1-8b-instruct-Q4_K_M.gguf` | 4.4 | 4.6 | 277 |
| 11 | `Teuken-7B-instruct-commercial-v0.4.Q4_K_S.gguf` | 3.0 | 3.3 | 286 |
| 12 | `Ministral-3-14B-Reasoning-2512-Q4_K_M.gguf` | 2.7 | 3.0 | 537 |
| 13 | `gpt-oss-20b-MXFP4.gguf` | 2.7 | 2.6 | 1261 |
| 14 | `phi-4-Q4_K_M.gguf` | 2.6 | 2.7 | 471 |
| 15 | `openai_gpt-oss-20b-MXFP4.gguf` | 2.6 | 2.6 | 1133 |
| 16 | `internlm2_5-20b-chat-q4_0.gguf` | 2.5 | 2.6 | 599 |

### Metal GPU

| # | Model | Arch | Status | Size | Load | Decode | Prefill | Note |
|---:|---|:---:|---|---:|---:|---:|---:|---|
| 1 | `Hermes-3-Llama-3.2-3B.Q4_K_M.gguf` | llama | ok | 1925 | 166 | 16.7 | 16.3 |  |
| 2 | `deepseek-math-7b-instruct.Q4_0.gguf` | llama | ok | 3814 | 259 | 6.6 | 7.1 |  |
| 3 | `DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf` | deepseek4_mtp_support | skip | 3631 | — | — | — | unsupported architecture |
| 4 | `MiniMaxAI_MiniMax-M2.7-imatrix.gguf` | unknown | skip | 469 | — | — | — | unsupported architecture |
| 5 | `openai_gpt-oss-20b-MXFP4.gguf` | gpt-oss | ok | 11548 | 798 | 2.6 | 2.7 |  |
| 6 | `stable-code-instruct-3b-Q4_0.gguf` | stablelm | ok | 1534 | 89 | 14.3 | 15.2 |  |
| 7 | `internlm2_5-20b-chat-q4_0.gguf` | internlm2 | ok | 10798 | 553 | 2.5 | 2.5 |  |
| 8 | `DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf` | deepseek2 | partial | 9884 | — | — | — | DeepSeek2 MLA attention tensors are present, but the runtime does not yet implement MLA |
| 9 | `DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf` | qwen2 | ok | 4466 | 320 | 12.3 | 10.9 |  |
| 10 | `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` | llama | ok | 4692 | 335 | 12.9 | 11.6 |  |
| 11 | `Ministral-3-14B-Reasoning-2512-Q4_K_M.gguf` | mistral3 | ok | 7857 | 502 | 8.9 | 3.8 |  |
| 12 | `Ministral-3-3B-Instruct-2512-Q4_K_M.gguf` | mistral3 | ok | 2047 | 141 | 17.9 | 18.1 |  |
| 13 | `Phi-3.1-mini-128k-instruct-Q4_K_M.gguf` | phi3 | ok | 2282 | 68 | 12.3 | 12.2 |  |
| 14 | `gemma-4-26B-A4B-it-Q4_K_M.gguf` | gemma4 | ok | 16017 | 7093 | 13.2 | 10.0 |  |
| 15 | `gpt-oss-20b-MXFP4.gguf` | gpt-oss | ok | 11548 | 631 | 2.6 | 2.7 |  |
| 16 | `granite-3.1-8b-instruct-Q4_K_M.gguf` | granite | ok | 4713 | 231 | 11.6 | 9.8 |  |
| 17 | `phi-4-Q4_K_M.gguf` | phi3 | ok | 8633 | 397 | 6.9 | 3.4 |  |
| 18 | `Teuken-7B-instruct-commercial-v0.4.Q4_K_S.gguf` | llama | ok | 4480 | 228 | 3.7 | 3.6 |  |
| 19 | `llava-v1.5-7b-Q4_0.gguf` | llama | ok | 3648 | 163 | 6.9 | 7.1 |  |

### Metal GPU Decode Ranking

| Rank | Model | Decode | Prefill | Load |
|---:|---|---:|---:|---:|
| 1 | `Ministral-3-3B-Instruct-2512-Q4_K_M.gguf` | 17.9 | 18.1 | 141 |
| 2 | `Hermes-3-Llama-3.2-3B.Q4_K_M.gguf` | 16.7 | 16.3 | 166 |
| 3 | `stable-code-instruct-3b-Q4_0.gguf` | 14.3 | 15.2 | 89 |
| 4 | `gemma-4-26B-A4B-it-Q4_K_M.gguf` | 13.2 | 10.0 | 7093 |
| 5 | `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` | 12.9 | 11.6 | 335 |
| 6 | `Phi-3.1-mini-128k-instruct-Q4_K_M.gguf` | 12.3 | 12.2 | 68 |
| 7 | `DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf` | 12.3 | 10.9 | 320 |
| 8 | `granite-3.1-8b-instruct-Q4_K_M.gguf` | 11.6 | 9.8 | 231 |
| 9 | `Ministral-3-14B-Reasoning-2512-Q4_K_M.gguf` | 8.9 | 3.8 | 502 |
| 10 | `phi-4-Q4_K_M.gguf` | 6.9 | 3.4 | 397 |
| 11 | `llava-v1.5-7b-Q4_0.gguf` | 6.9 | 7.1 | 163 |
| 12 | `deepseek-math-7b-instruct.Q4_0.gguf` | 6.6 | 7.1 | 259 |
| 13 | `Teuken-7B-instruct-commercial-v0.4.Q4_K_S.gguf` | 3.7 | 3.6 | 228 |
| 14 | `gpt-oss-20b-MXFP4.gguf` | 2.6 | 2.7 | 631 |
| 15 | `openai_gpt-oss-20b-MXFP4.gguf` | 2.6 | 2.7 | 798 |
| 16 | `internlm2_5-20b-chat-q4_0.gguf` | 2.5 | 2.5 | 553 |

---

Re-run `bench_models.sh` to refresh this report. Use `BENCH_PROFILES=cpu` or `BENCH_PROFILES=metal` for a single backend run.
