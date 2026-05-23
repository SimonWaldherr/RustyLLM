# RustyLLM Benchmark Results

Updated: **2026-05-23 18:58 CEST**

This report compares the CPU path with the optional Apple Metal GPU path. Metal here means GPU acceleration through RustyLLM's Metal kernels; it is not a CoreML, ANE, or NPU backend.

## Run Configuration

| Setting | Value |
|---|---|
| Model directory | `/Users/simonwaldherr/.cache/lm-studio/models` |
| Prompt | Explain local LLM inference performance in one concise paragraph. |
| Runs | 2 x 64 generated tokens per model |
| Pause | 2 seconds between models |
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
| CPU | 16 | 0 | 3 | 14.0 | 4.6 |
| Metal GPU | 16 | 0 | 3 | 17.6 | 8.3 |

## CPU vs Metal

Each speed cell is `decode / prefill` in tokens per second. Speedup uses decode throughput.

| # | Model | Arch | Size | CPU | Metal | Metal/CPU | Result |
|---:|---|:---:|---:|---:|---:|---:|---|
| 1 | `Hermes-3-Llama-3.2-3B.Q4_K_M.gguf` | llama | 1925 | 9.9 / 12.1 | 16.8 / 16.0 | 1.69x | Metal faster |
| 2 | `deepseek-math-7b-instruct.Q4_0.gguf` | llama | 3814 | 5.9 / 6.1 | 7.3 / 7.7 | 1.24x | Metal faster |
| 3 | `DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf` | deepseek4_mtp_support | 3631 | skip | skip | — | unsupported architecture |
| 4 | `MiniMaxAI_MiniMax-M2.7-imatrix.gguf` | unknown | 469 | skip | skip | — | unsupported architecture |
| 5 | `openai_gpt-oss-20b-MXFP4.gguf` | gpt-oss | 11548 | 2.6 / 2.5 | 2.3 / 2.5 | 0.89x | CPU faster |
| 6 | `stable-code-instruct-3b-Q4_0.gguf` | stablelm | 1534 | 14.0 / 14.8 | 14.4 / 15.1 | 1.03x | similar |
| 7 | `internlm2_5-20b-chat-q4_0.gguf` | internlm2 | 10798 | 2.1 / 1.6 | 2.5 / 2.4 | 1.22x | Metal faster |
| 8 | `DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf` | deepseek2 | 9884 | partial | partial | — | DeepSeek2 MLA attention tensors are present… |
| 9 | `DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf` | qwen2 | 4466 | 4.7 / 5.2 | 12.7 / 11.1 | 2.71x | Metal faster |
| 10 | `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` | llama | 4692 | 4.4 / 4.5 | 13.8 / 12.5 | 3.12x | Metal faster |
| 11 | `Ministral-3-14B-Reasoning-2512-Q4_K_M.gguf` | mistral3 | 7857 | 2.9 / 3.0 | 9.2 / 2.7 | 3.17x | Metal faster |
| 12 | `Ministral-3-3B-Instruct-2512-Q4_K_M.gguf` | mistral3 | 2047 | 10.0 / 11.4 | 17.6 / 16.8 | 1.76x | Metal faster |
| 13 | `Phi-3.1-mini-128k-instruct-Q4_K_M.gguf` | phi3 | 2282 | 6.2 / 6.5 | 7.2 / 7.0 | 1.16x | Metal faster |
| 14 | `gemma-4-26B-A4B-it-Q4_K_M.gguf` | gemma4 | 16017 | 11.8 / 13.3 | 13.3 / 10.1 | 1.12x | Metal faster |
| 15 | `gpt-oss-20b-MXFP4.gguf` | gpt-oss | 11548 | 2.5 / 2.5 | 2.5 / 2.5 | 1.00x | similar |
| 16 | `granite-3.1-8b-instruct-Q4_K_M.gguf` | granite | 4713 | 4.5 / 4.6 | 11.8 / 10.1 | 2.64x | Metal faster |
| 17 | `phi-4-Q4_K_M.gguf` | phi3 | 8633 | 2.2 / 2.2 | 4.0 / 2.1 | 1.88x | Metal faster |
| 18 | `Teuken-7B-instruct-commercial-v0.4.Q4_K_S.gguf` | llama | 4480 | 3.0 / 3.2 | 3.7 / 3.6 | 1.25x | Metal faster |
| 19 | `llava-v1.5-7b-Q4_0.gguf` | llama | 3648 | 6.9 / 7.2 | 7.2 / 7.3 | 1.04x | similar |

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
| 1 | `Hermes-3-Llama-3.2-3B.Q4_K_M.gguf` | llama | ok | 1925 | 159 | 9.9 | 12.1 |  |
| 2 | `deepseek-math-7b-instruct.Q4_0.gguf` | llama | ok | 3814 | 243 | 5.9 | 6.1 |  |
| 3 | `DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf` | deepseek4_mtp_support | skip | 3631 | — | — | — | unsupported architecture |
| 4 | `MiniMaxAI_MiniMax-M2.7-imatrix.gguf` | unknown | skip | 469 | — | — | — | unsupported architecture |
| 5 | `openai_gpt-oss-20b-MXFP4.gguf` | gpt-oss | ok | 11548 | 2053 | 2.6 | 2.5 |  |
| 6 | `stable-code-instruct-3b-Q4_0.gguf` | stablelm | ok | 1534 | 89 | 14.0 | 14.8 |  |
| 7 | `internlm2_5-20b-chat-q4_0.gguf` | internlm2 | ok | 10798 | 2846 | 2.1 | 1.6 |  |
| 8 | `DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf` | deepseek2 | partial | 9884 | — | — | — | DeepSeek2 MLA attention tensors are present, but the runtime does not yet implement MLA |
| 9 | `DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf` | qwen2 | ok | 4466 | 352 | 4.7 | 5.2 |  |
| 10 | `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` | llama | ok | 4692 | 372 | 4.4 | 4.5 |  |
| 11 | `Ministral-3-14B-Reasoning-2512-Q4_K_M.gguf` | mistral3 | ok | 7857 | 493 | 2.9 | 3.0 |  |
| 12 | `Ministral-3-3B-Instruct-2512-Q4_K_M.gguf` | mistral3 | ok | 2047 | 166 | 10.0 | 11.4 |  |
| 13 | `Phi-3.1-mini-128k-instruct-Q4_K_M.gguf` | phi3 | ok | 2282 | 84 | 6.2 | 6.5 |  |
| 14 | `gemma-4-26B-A4B-it-Q4_K_M.gguf` | gemma4 | ok | 16017 | 8245 | 11.8 | 13.3 |  |
| 15 | `gpt-oss-20b-MXFP4.gguf` | gpt-oss | ok | 11548 | 3335 | 2.5 | 2.5 |  |
| 16 | `granite-3.1-8b-instruct-Q4_K_M.gguf` | granite | ok | 4713 | 261 | 4.5 | 4.6 |  |
| 17 | `phi-4-Q4_K_M.gguf` | phi3 | ok | 8633 | 486 | 2.2 | 2.2 |  |
| 18 | `Teuken-7B-instruct-commercial-v0.4.Q4_K_S.gguf` | llama | ok | 4480 | 308 | 3.0 | 3.2 |  |
| 19 | `llava-v1.5-7b-Q4_0.gguf` | llama | ok | 3648 | 185 | 6.9 | 7.2 |  |

### CPU Decode Ranking

| Rank | Model | Decode | Prefill | Load |
|---:|---|---:|---:|---:|
| 1 | `stable-code-instruct-3b-Q4_0.gguf` | 14.0 | 14.8 | 89 |
| 2 | `gemma-4-26B-A4B-it-Q4_K_M.gguf` | 11.8 | 13.3 | 8245 |
| 3 | `Ministral-3-3B-Instruct-2512-Q4_K_M.gguf` | 10.0 | 11.4 | 166 |
| 4 | `Hermes-3-Llama-3.2-3B.Q4_K_M.gguf` | 9.9 | 12.1 | 159 |
| 5 | `llava-v1.5-7b-Q4_0.gguf` | 6.9 | 7.2 | 185 |
| 6 | `Phi-3.1-mini-128k-instruct-Q4_K_M.gguf` | 6.2 | 6.5 | 84 |
| 7 | `deepseek-math-7b-instruct.Q4_0.gguf` | 5.9 | 6.1 | 243 |
| 8 | `DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf` | 4.7 | 5.2 | 352 |
| 9 | `granite-3.1-8b-instruct-Q4_K_M.gguf` | 4.5 | 4.6 | 261 |
| 10 | `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` | 4.4 | 4.5 | 372 |
| 11 | `Teuken-7B-instruct-commercial-v0.4.Q4_K_S.gguf` | 3.0 | 3.2 | 308 |
| 12 | `Ministral-3-14B-Reasoning-2512-Q4_K_M.gguf` | 2.9 | 3.0 | 493 |
| 13 | `openai_gpt-oss-20b-MXFP4.gguf` | 2.6 | 2.5 | 2053 |
| 14 | `gpt-oss-20b-MXFP4.gguf` | 2.5 | 2.5 | 3335 |
| 15 | `phi-4-Q4_K_M.gguf` | 2.2 | 2.2 | 486 |
| 16 | `internlm2_5-20b-chat-q4_0.gguf` | 2.1 | 1.6 | 2846 |

### Metal GPU

| # | Model | Arch | Status | Size | Load | Decode | Prefill | Note |
|---:|---|:---:|---|---:|---:|---:|---:|---|
| 1 | `Hermes-3-Llama-3.2-3B.Q4_K_M.gguf` | llama | ok | 1925 | 151 | 16.8 | 16.0 |  |
| 2 | `deepseek-math-7b-instruct.Q4_0.gguf` | llama | ok | 3814 | 236 | 7.3 | 7.7 |  |
| 3 | `DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf` | deepseek4_mtp_support | skip | 3631 | — | — | — | unsupported architecture |
| 4 | `MiniMaxAI_MiniMax-M2.7-imatrix.gguf` | unknown | skip | 469 | — | — | — | unsupported architecture |
| 5 | `openai_gpt-oss-20b-MXFP4.gguf` | gpt-oss | ok | 11548 | 1150 | 2.3 | 2.5 |  |
| 6 | `stable-code-instruct-3b-Q4_0.gguf` | stablelm | ok | 1534 | 74 | 14.4 | 15.1 |  |
| 7 | `internlm2_5-20b-chat-q4_0.gguf` | internlm2 | ok | 10798 | 2626 | 2.5 | 2.4 |  |
| 8 | `DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf` | deepseek2 | partial | 9884 | — | — | — | DeepSeek2 MLA attention tensors are present, but the runtime does not yet implement MLA |
| 9 | `DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf` | qwen2 | ok | 4466 | 293 | 12.7 | 11.1 |  |
| 10 | `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` | llama | ok | 4692 | 288 | 13.8 | 12.5 |  |
| 11 | `Ministral-3-14B-Reasoning-2512-Q4_K_M.gguf` | mistral3 | ok | 7857 | 516 | 9.2 | 2.7 |  |
| 12 | `Ministral-3-3B-Instruct-2512-Q4_K_M.gguf` | mistral3 | ok | 2047 | 163 | 17.6 | 16.8 |  |
| 13 | `Phi-3.1-mini-128k-instruct-Q4_K_M.gguf` | phi3 | ok | 2282 | 98 | 7.2 | 7.0 |  |
| 14 | `gemma-4-26B-A4B-it-Q4_K_M.gguf` | gemma4 | ok | 16017 | 8124 | 13.3 | 10.1 |  |
| 15 | `gpt-oss-20b-MXFP4.gguf` | gpt-oss | ok | 11548 | 2344 | 2.5 | 2.5 |  |
| 16 | `granite-3.1-8b-instruct-Q4_K_M.gguf` | granite | ok | 4713 | 228 | 11.8 | 10.1 |  |
| 17 | `phi-4-Q4_K_M.gguf` | phi3 | ok | 8633 | 407 | 4.0 | 2.1 |  |
| 18 | `Teuken-7B-instruct-commercial-v0.4.Q4_K_S.gguf` | llama | ok | 4480 | 224 | 3.7 | 3.6 |  |
| 19 | `llava-v1.5-7b-Q4_0.gguf` | llama | ok | 3648 | 152 | 7.2 | 7.3 |  |

### Metal GPU Decode Ranking

| Rank | Model | Decode | Prefill | Load |
|---:|---|---:|---:|---:|
| 1 | `Ministral-3-3B-Instruct-2512-Q4_K_M.gguf` | 17.6 | 16.8 | 163 |
| 2 | `Hermes-3-Llama-3.2-3B.Q4_K_M.gguf` | 16.8 | 16.0 | 151 |
| 3 | `stable-code-instruct-3b-Q4_0.gguf` | 14.4 | 15.1 | 74 |
| 4 | `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` | 13.8 | 12.5 | 288 |
| 5 | `gemma-4-26B-A4B-it-Q4_K_M.gguf` | 13.3 | 10.1 | 8124 |
| 6 | `DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf` | 12.7 | 11.1 | 293 |
| 7 | `granite-3.1-8b-instruct-Q4_K_M.gguf` | 11.8 | 10.1 | 228 |
| 8 | `Ministral-3-14B-Reasoning-2512-Q4_K_M.gguf` | 9.2 | 2.7 | 516 |
| 9 | `deepseek-math-7b-instruct.Q4_0.gguf` | 7.3 | 7.7 | 236 |
| 10 | `Phi-3.1-mini-128k-instruct-Q4_K_M.gguf` | 7.2 | 7.0 | 98 |
| 11 | `llava-v1.5-7b-Q4_0.gguf` | 7.2 | 7.3 | 152 |
| 12 | `phi-4-Q4_K_M.gguf` | 4.0 | 2.1 | 407 |
| 13 | `Teuken-7B-instruct-commercial-v0.4.Q4_K_S.gguf` | 3.7 | 3.6 | 224 |
| 14 | `gpt-oss-20b-MXFP4.gguf` | 2.5 | 2.5 | 2344 |
| 15 | `internlm2_5-20b-chat-q4_0.gguf` | 2.5 | 2.4 | 2626 |
| 16 | `openai_gpt-oss-20b-MXFP4.gguf` | 2.3 | 2.5 | 1150 |

---

Re-run `bench_models.sh` to refresh this report. Use `BENCH_PROFILES=cpu` or `BENCH_PROFILES=metal` for a single backend run.
