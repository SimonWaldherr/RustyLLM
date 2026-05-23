# RustyLLM Benchmark Results

> **Prompt:** *"Explain local LLM inference performance in one concise paragraph."*
> **Runs:** 2 runs × 64 tokens per model, CPU-only (no GPU)
> **Date:** 2026-05-23 17:37 CEST

## Hardware

| | |
|---|---|
| **CPU** | Apple M2 Max |
| **Logical cores** | 12 |
| **RAM** | 32 GB |
| **OS** | macOS 26.5 |
| **Rust** | rustc 1.85.0 (4d91de4e4 2025-02-17) |
| **SIMD** | unknown |
| **Metal GPU** | disabled (RUSTY_LLM_METAL not set) |

## Model Results

| # | Model file | Architecture | Compat | Size (MB) | Load (ms) | Decode (tok/s) | Prefill (tok/s) |
|--:|-----------|:---:|:---:|---:|---:|---:|---:|
| 1 | `Hermes-3-Llama-3.2-3B.Q4_K_M.gguf` | llama | ✅ | 1925 | 169 | 9.7 | 12.0 |
| 2 | `deepseek-math-7b-instruct.Q4_0.gguf` | llama | ✅ | 3814 | 245 | 6.4 | 7.2 |
| 3 | `DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf` | unknown | ❌ | 0 | — | — | — | <!-- arch not recognized -->
| 4 | `MiniMaxAI_MiniMax-M2.7-imatrix.gguf` | unknown | ❌ | 0 | — | — | — | <!-- arch not recognized -->
| 5 | `openai_gpt-oss-20b-MXFP4.gguf` | gpt-oss | ✅ | 11548 | 4969 | 2.4 | 2.4 |
| 6 | `stable-code-instruct-3b-Q4_0.gguf` | stablelm | ✅ | 1534 | 87 | 12.6 | 14.2 |
| 7 | `internlm2_5-20b-chat-q4_0.gguf` | internlm2 | ✅ | 10798 | 4322 | 1.4 | 1.4 |
| 8 | `DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf` | deepseek2 | ⚠️ | 9884 | — | — | — | <!-- thread 'main' panicked at src/model.rs:643:28: -->
| 9 | `DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf` | qwen2 | ✅ | 4466 | 317 | 4.1 | 5.1 |
| 10 | `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` | llama | ✅ | 4692 | 380 | 4.6 | 4.6 |
| 11 | `Ministral-3-14B-Reasoning-2512-Q4_K_M.gguf` | mistral3 | ✅ | 7857 | 544 | 3.0 | 3.1 |
| 12 | `Ministral-3-3B-Instruct-2512-Q4_K_M.gguf` | mistral3 | ✅ | 2047 | 184 | 10.1 | 11.6 |
| 13 | `Phi-3.1-mini-128k-instruct-Q4_K_M.gguf` | phi3 | ❌ | 2282 | — | — | — | <!-- arch not recognized -->
| 14 | `gemma-4-26B-A4B-it-Q4_K_M.gguf` | gemma4 | ✅ | 16017 | 7846 | 13.1 | 14.3 |
| 15 | `gpt-oss-20b-MXFP4.gguf` | gpt-oss | ✅ | 11548 | 3285 | 2.6 | 2.6 |
| 16 | `granite-3.1-8b-instruct-Q4_K_M.gguf` | granite | ✅ | 4713 | 0 | 0.0 | 0.0 |
| 17 | `phi-4-Q4_K_M.gguf` | phi3 | ❌ | 8633 | — | — | — | <!-- arch not recognized -->
| 18 | `Teuken-7B-instruct-commercial-v0.4.Q4_K_S.gguf` | llama | ❌ | 4480 | — | — | — | <!-- arch not recognized -->
| 19 | `llava-v1.5-7b-Q4_0.gguf` | llama | ✅ | 3648 | 226 | 6.5 | 6.9 |

## Speed Ranking

Decode throughput (tok/s), supported models only, fastest first.

| Rank | Model file | Architecture | Decode (tok/s) | Prefill (tok/s) | Size (MB) |
|:---:|-----------|:---:|---:|---:|---:|
| 1 | `gemma-4-26B-A4B-it-Q4_K_M.gguf` | gemma4 | 13.1 | 14.3 | 16017 |
| 2 | `stable-code-instruct-3b-Q4_0.gguf` | stablelm | 12.6 | 14.2 | 1534 |
| 3 | `Ministral-3-3B-Instruct-2512-Q4_K_M.gguf` | mistral3 | 10.1 | 11.6 | 2047 |
| 4 | `Hermes-3-Llama-3.2-3B.Q4_K_M.gguf` | llama | 9.7 | 12.0 | 1925 |
| 5 | `llava-v1.5-7b-Q4_0.gguf` | llama | 6.5 | 6.9 | 3648 |
| 6 | `deepseek-math-7b-instruct.Q4_0.gguf` | llama | 6.4 | 7.2 | 3814 |
| 7 | `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` | llama | 4.6 | 4.6 | 4692 |
| 8 | `DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf` | qwen2 | 4.1 | 5.1 | 4466 |
| 9 | `Ministral-3-14B-Reasoning-2512-Q4_K_M.gguf` | mistral3 | 3.0 | 3.1 | 7857 |
| 10 | `gpt-oss-20b-MXFP4.gguf` | gpt-oss | 2.6 | 2.6 | 11548 |
| 11 | `openai_gpt-oss-20b-MXFP4.gguf` | gpt-oss | 2.4 | 2.4 | 11548 |
| 12 | `internlm2_5-20b-chat-q4_0.gguf` | internlm2 | 1.4 | 1.4 | 10798 |
| 13 | `granite-3.1-8b-instruct-Q4_K_M.gguf` | granite | 0.0 | 0.0 | 4713 |

---

*Re-run `bench_models.sh` to refresh. Raw JSON per model is stored in `.bench_raw/`.*
*Numbers reflect single-socket CPU inference; results vary with thermal state and OS load.*
*Ollama and LM Studio are typically faster due to GPU offload and llama.cpp kernels.*
