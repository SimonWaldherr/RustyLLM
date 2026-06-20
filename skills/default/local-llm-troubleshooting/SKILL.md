---
name: local-llm-troubleshooting
description: Use this skill when diagnosing local LLM startup speed, prompt latency, GGUF model loading, tokenizer/chat-template issues, Metal or CPU backend behavior, model selection problems, context-window limits, sampling settings, or API compatibility for OpenAI, LM Studio, Ollama, and MCP routes.
---

# Local LLM Troubleshooting

## Triage Order

1. Separate startup time from first-token latency and decode throughput.
2. Identify the model path resolution mode: exact `.gguf` path, directory scan, metadata catalog lookup, or remote/client alias.
3. Check backend selection: Metal enabled/disabled, CPU SIMD path, thread count, and runtime profile.
4. Check prompt rendering: system prompt, chat template, Thinking mode, Skills mode, stop sequences, and context length.
5. Check sampling: temperature, top-p, top-k, repeat penalty, seed, and max tokens.

## Useful Measurements

- startup: process start, model resolution, mmap open, GGUF parse, tokenizer build, weight setup
- prefill: rendered prompt tokens and prompt tokens per second
- decode: generated tokens and tokens per second
- session reuse: cached prompt tokens versus evaluated prompt tokens

## Output

Give the likely cause first, then a short measurement plan, then the smallest command or option change that can confirm it.

## Gotchas

- A direct `.gguf` path can be much faster than `--model-dir --model` if the latter triggers recursive catalog discovery.
- Longer system prompts, Thinking rewrites, and loaded Skills increase prefill work even when model loading is unchanged.
- Session cache helps repeated chat turns, but only when clients send a stable `conversation_id` with prompt caching enabled.
- Do not compare decode speed across prompts unless prompt length, max tokens, and sampling settings are comparable.
