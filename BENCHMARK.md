# RustyLLM Benchmark Results

Updated: **2026-07-02 (Ministral focus section); 2026-06-23 23:33 CEST (multi-model tables below)**

This report compares the CPU path with the optional Apple Metal GPU path. Metal here means GPU acceleration through RustyLLM's Metal kernels; it is not a CoreML, ANE, or NPU backend.

## Focus: Ministral 3 3B Instruct 2512

Focused tuning run for `Ministral-3-3B-Instruct-2512-Q4_K_M.gguf` on Apple M2 Max, 12 logical cores, 32 GB RAM, macOS 26.5.1, rustc 1.95.0. The benchmark used 6 runs with up to 128 generated tokens, `--temp 0`, `--seed 42`, and the prompt `Explain local LLM inference performance in one concise paragraph.`, with an 15s cooldown between profiles (M2 Max shows real ±15-30% run-to-run throughput swings from thermal/clock scaling; single un-averaged runs are not reliable for backend comparisons).

**Note on the CPU number:** the CPU path has gotten substantially faster since this doc was first written (SIMD work landed in commits after 2026-06-23), so the old CPU baseline of 9.3 tok/s below is stale — re-measured today it's **26.2 tok/s**, which changes the Metal/CPU speedup story a lot for this model. The multi-model tables further down have *not* been re-verified and likely understate CPU throughput for the same reason.

| Profile | Extra env / args | Decode tok/s | Prefill tok/s | Load | Result |
|---|---|---:|---:|---:|---|
| CPU | `RUSTY_LLM_METAL=0` | 26.2 | 31.6 | 284 ms | baseline (re-measured; was 9.3 in the 2026-06-23 run) |
| Metal standard | `RUSTY_LLM_METAL=1` | 27.9 | 28.0 | 257 ms | best previously-known-good path; only ~1.07x over CPU now |
| **Metal resident decoder** | `RUSTY_LLM_METAL=1` (auto-enabled for CLI/REPL/`--bench`) | **42.9** | **49.1** | 273 ms | new, fastest — see below |
| Metal ultra *(not re-verified since 2026-06-23)* | `RUSTY_LLM_METAL=1 --profile mistral-ultra` | 25.4 | 26.9 | 258 ms | slower/less stable |
| Metal post-FFN fusion *(not re-verified; flag is now opt-in, see note)* | `RUSTY_LLM_METAL=1 RUSTY_LLM_METAL_POST_FFN=1` | 23.2 | 23.2 | 205 ms | slower |
| Metal fast attention approx *(not re-verified since 2026-06-23)* | `RUSTY_LLM_METAL=1 RUSTY_LLM_FAST_ATTN=1` | 20.6 | 19.6 | 288 ms | slower |

Optimized operating point today: plain **`RUSTY_LLM_METAL=1`** now gets the resident decoder automatically outside server mode, a **1.64x** decode speedup over the re-measured CPU baseline (1.54x over standard Metal). Leave `mistral-ultra` and `RUSTY_LLM_FAST_ATTN` disabled.

`RUSTY_LLM_METAL_POST_FFN` was found to default to *enabled* (`env_flag(...) != Some(false)`) despite being documented as experimental and measured slower here — inconsistent with the otherwise-equivalent `RUSTY_LLM_METAL_NOCOPY`, which correctly defaults off. Fixed in [`metal.rs`](src/metal.rs) so it now requires an explicit `RUSTY_LLM_METAL_POST_FFN=1` to enable.

### New: GPU-resident single-command-buffer decoder

The resident decoder runs an entire token's forward pass (embedding → all layers → final norm → logits) as **one Metal command buffer** with **one `waitUntilCompleted`**, keeping the KV cache and all intermediates GPU-resident instead of round-tripping per matvec/attention op. This removes most of the CPU↔GPU synchronization overhead that otherwise limits the standard path.

- Verified byte-identical greedy-decode output against the standard Metal path across multiple prompts/seeds/lengths (temp 0).
- It keeps a single static GPU-resident KV cache and working buffer set indexed only by token position, so it's only safe for one exclusive conversation at a time — a lock prevents data races, but two interleaved conversations would still silently overwrite each other's KV slots. Because of that it **auto-enables for CLI/REPL/`--bench`/`--mcp`** (all single- or sequential-conversation) but **auto-disables under `--serve`** (concurrent multi-session HTTP API, per `session.rs`/`server.rs`). `RUSTY_LLM_METAL_RESIDENT=0|1` always overrides the auto-detection either way.
- Requires: no sliding window, ≤200 layers, `dim`/`hidden_dim` multiples of 256, `head_dim`/`value_dim` ≤ 256, GQA-compatible head counts, and Q4_K/Q6_K weights throughout. Falls back to the standard path automatically if any of this doesn't hold.
- Falls back safely (never reuses stale GPU buffers) if a different model's shape/pointer fingerprint is detected in the same process.

Isolated layer-0 kernel timing with Metal standard shows the main decode cost is projection-heavy: fused Q/K/V takes 0.61 ms, the fused FFN block takes 0.98 ms, and the tied-vocabulary output projection takes 2.32 ms per token. The resident decoder collapses all of this (across all 26 layers) plus RoPE/attention/KV-cache writes into one submission per token, which is why its prefill also nearly doubles (49.1 vs 28.0 tok/s) — prefill here is a sequence of single-token forward passes (23 prompt tokens), so per-call dispatch overhead matters just as much as in decode. There is no smaller compatible local MTP draft model available in the tested model directory, so speculative decoding was not enabled.

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
| CPU | 11 | 0 | 3 | 12.9 | 5.0 |
| Metal GPU | 11 | 0 | 3 | 27.7 | 15.1 |

## CPU vs Metal

*(Table below is from the 2026-06-23 multi-model sweep and has not been re-verified. The Ministral 3B CPU number in it, 10.1 tok/s, is now known stale — see the Ministral focus section above, which re-measured 26.2 tok/s on 2026-07-02. CPU-path SIMD work landed after this sweep, so other models' `Metal/CPU` speedup ratios here are likely overstated too; re-run `bench_models.sh` for current numbers.)*

Each speed cell is `decode / prefill` in tokens per second. Speedup uses decode throughput.

| # | Model | Arch | Size | CPU | Metal | Metal/CPU | Result |
|---:|---|:---:|---:|---:|---:|---:|---|
| 1 | `DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf` | deepseek2 | 9884 | partial | partial | — | DeepSeek2 MLA attention tensors are present… |
| 2 | `DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf` | qwen2 | 4466 | 5.3 / 5.8 | 18.6 / 14.8 | 3.49x | Metal faster |
| 3 | `gemma-4-12B-it-QAT-Q4_0.gguf` | gemma4 | 6652 | 4.3 / 4.8 | 7.6 / 7.2 | 1.79x | Metal faster |
| 4 | `gemma-4-26B-A4B-it-Q4_K_M.gguf` | gemma4 | 16017 | 12.2 / 15.1 | 18.5 / 14.1 | 1.51x | Metal faster |
| 5 | `gemma-4-E2B-it-Q4_K_M.gguf` | gemma4 | 3269 | 12.9 / 15.6 | 18.2 / 18.7 | 1.41x | Metal faster |
| 6 | `gpt-oss-20b-MXFP4.gguf` | gpt-oss | 11548 | 2.7 / 2.8 | 2.5 / 2.5 | 0.91x | CPU faster |
| 7 | `granite-3.1-8b-instruct-Q4_K_M.gguf` | granite | 4713 | 4.5 / 4.7 | 15.1 / 11.6 | 3.32x | Metal faster |
| 8 | `granite-embedding-278m-multilingual-Q4_K_M.gguf` | bert | 208 | skip | skip | — | unsupported architecture |
| 9 | `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` | llama | 4692 | 5.0 / 5.4 | 18.6 / 15.1 | 3.73x | Metal faster |
| 10 | `Ministral-3-14B-Reasoning-2512-Q4_K_M.gguf` | mistral3 | 7857 | 3.0 / 3.2 | 11.2 / 6.8 | 3.71x | Metal faster |
| 11 | `Ministral-3-3B-Instruct-2512-Q4_K_M.gguf` | mistral3 | 2047 | 10.1 / 11.6 | 27.7 / 24.5 | 2.75x | Metal faster |
| 12 | `NVIDIA-Nemotron-3-Nano-4B-Q4_K_M.gguf` | nemotron_h | 2705 | skip | skip | — | unsupported architecture |
| 13 | `Phi-3.1-mini-128k-instruct-Q4_K_M.gguf` | phi3 | 2282 | 8.5 / 9.0 | 13.6 / 14.0 | 1.61x | Metal faster |
| 14 | `phi-4-Q4_K_M.gguf` | phi3 | 8633 | 2.6 / 2.6 | 7.5 / 5.7 | 2.85x | Metal faster |

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
| 2 | `DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf` | qwen2 | ok | 4466 | 263 | 5.3 | 5.8 |  |
| 3 | `gemma-4-12B-it-QAT-Q4_0.gguf` | gemma4 | ok | 6652 | 590 | 4.3 | 4.8 |  |
| 4 | `gemma-4-26B-A4B-it-Q4_K_M.gguf` | gemma4 | ok | 16017 | 6537 | 12.2 | 15.1 |  |
| 5 | `gemma-4-E2B-it-Q4_K_M.gguf` | gemma4 | ok | 3269 | 421 | 12.9 | 15.6 |  |
| 6 | `gpt-oss-20b-MXFP4.gguf` | gpt-oss | ok | 11548 | 756 | 2.7 | 2.8 |  |
| 7 | `granite-3.1-8b-instruct-Q4_K_M.gguf` | granite | ok | 4713 | 255 | 4.5 | 4.7 |  |
| 8 | `granite-embedding-278m-multilingual-Q4_K_M.gguf` | bert | skip | 208 | — | — | — | unsupported architecture |
| 9 | `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` | llama | ok | 4692 | 334 | 5.0 | 5.4 |  |
| 10 | `Ministral-3-14B-Reasoning-2512-Q4_K_M.gguf` | mistral3 | ok | 7857 | 507 | 3.0 | 3.2 |  |
| 11 | `Ministral-3-3B-Instruct-2512-Q4_K_M.gguf` | mistral3 | ok | 2047 | 202 | 10.1 | 11.6 |  |
| 12 | `NVIDIA-Nemotron-3-Nano-4B-Q4_K_M.gguf` | nemotron_h | skip | 2705 | — | — | — | unsupported architecture |
| 13 | `Phi-3.1-mini-128k-instruct-Q4_K_M.gguf` | phi3 | ok | 2282 | 91 | 8.5 | 9.0 |  |
| 14 | `phi-4-Q4_K_M.gguf` | phi3 | ok | 8633 | 519 | 2.6 | 2.6 |  |

### CPU Decode Ranking

| Rank | Model | Decode | Prefill | Load |
|---:|---|---:|---:|---:|
| 1 | `gemma-4-E2B-it-Q4_K_M.gguf` | 12.9 | 15.6 | 421 |
| 2 | `gemma-4-26B-A4B-it-Q4_K_M.gguf` | 12.2 | 15.1 | 6537 |
| 3 | `Ministral-3-3B-Instruct-2512-Q4_K_M.gguf` | 10.1 | 11.6 | 202 |
| 4 | `Phi-3.1-mini-128k-instruct-Q4_K_M.gguf` | 8.5 | 9.0 | 91 |
| 5 | `DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf` | 5.3 | 5.8 | 263 |
| 6 | `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` | 5.0 | 5.4 | 334 |
| 7 | `granite-3.1-8b-instruct-Q4_K_M.gguf` | 4.5 | 4.7 | 255 |
| 8 | `gemma-4-12B-it-QAT-Q4_0.gguf` | 4.3 | 4.8 | 590 |
| 9 | `Ministral-3-14B-Reasoning-2512-Q4_K_M.gguf` | 3.0 | 3.2 | 507 |
| 10 | `gpt-oss-20b-MXFP4.gguf` | 2.7 | 2.8 | 756 |
| 11 | `phi-4-Q4_K_M.gguf` | 2.6 | 2.6 | 519 |

### Metal GPU

| # | Model | Arch | Status | Size | Load | Decode | Prefill | Note |
|---:|---|:---:|---|---:|---:|---:|---:|---|
| 1 | `DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf` | deepseek2 | partial | 9884 | — | — | — | DeepSeek2 MLA attention tensors are present, but the runtime does not yet implement MLA |
| 2 | `DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf` | qwen2 | ok | 4466 | 286 | 18.6 | 14.8 |  |
| 3 | `gemma-4-12B-it-QAT-Q4_0.gguf` | gemma4 | ok | 6652 | 555 | 7.6 | 7.2 |  |
| 4 | `gemma-4-26B-A4B-it-Q4_K_M.gguf` | gemma4 | ok | 16017 | 7153 | 18.5 | 14.1 |  |
| 5 | `gemma-4-E2B-it-Q4_K_M.gguf` | gemma4 | ok | 3269 | 424 | 18.2 | 18.7 |  |
| 6 | `gpt-oss-20b-MXFP4.gguf` | gpt-oss | ok | 11548 | 740 | 2.5 | 2.5 |  |
| 7 | `granite-3.1-8b-instruct-Q4_K_M.gguf` | granite | ok | 4713 | 265 | 15.1 | 11.6 |  |
| 8 | `granite-embedding-278m-multilingual-Q4_K_M.gguf` | bert | skip | 208 | — | — | — | unsupported architecture |
| 9 | `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` | llama | ok | 4692 | 365 | 18.6 | 15.1 |  |
| 10 | `Ministral-3-14B-Reasoning-2512-Q4_K_M.gguf` | mistral3 | ok | 7857 | 525 | 11.2 | 6.8 |  |
| 11 | `Ministral-3-3B-Instruct-2512-Q4_K_M.gguf` | mistral3 | ok | 2047 | 220 | 27.7 | 24.5 |  |
| 12 | `NVIDIA-Nemotron-3-Nano-4B-Q4_K_M.gguf` | nemotron_h | skip | 2705 | — | — | — | unsupported architecture |
| 13 | `Phi-3.1-mini-128k-instruct-Q4_K_M.gguf` | phi3 | ok | 2282 | 84 | 13.6 | 14.0 |  |
| 14 | `phi-4-Q4_K_M.gguf` | phi3 | ok | 8633 | 426 | 7.5 | 5.7 |  |

### Metal GPU Decode Ranking

| Rank | Model | Decode | Prefill | Load |
|---:|---|---:|---:|---:|
| 1 | `Ministral-3-3B-Instruct-2512-Q4_K_M.gguf` | 27.7 | 24.5 | 220 |
| 2 | `DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf` | 18.6 | 14.8 | 286 |
| 3 | `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` | 18.6 | 15.1 | 365 |
| 4 | `gemma-4-26B-A4B-it-Q4_K_M.gguf` | 18.5 | 14.1 | 7153 |
| 5 | `gemma-4-E2B-it-Q4_K_M.gguf` | 18.2 | 18.7 | 424 |
| 6 | `granite-3.1-8b-instruct-Q4_K_M.gguf` | 15.1 | 11.6 | 265 |
| 7 | `Phi-3.1-mini-128k-instruct-Q4_K_M.gguf` | 13.6 | 14.0 | 84 |
| 8 | `Ministral-3-14B-Reasoning-2512-Q4_K_M.gguf` | 11.2 | 6.8 | 525 |
| 9 | `gemma-4-12B-it-QAT-Q4_0.gguf` | 7.6 | 7.2 | 555 |
| 10 | `phi-4-Q4_K_M.gguf` | 7.5 | 5.7 | 426 |
| 11 | `gpt-oss-20b-MXFP4.gguf` | 2.5 | 2.5 | 740 |

---

Re-run `bench_models.sh` to refresh this report. Use `BENCH_PROFILES=cpu` or `BENCH_PROFILES=metal` for a single backend run.
