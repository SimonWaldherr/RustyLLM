# MTP Usage Guide

RustyLLM implements greedy speculative decoding with exact target-model
verification. Qwen hybrid GGUFs can use their embedded one-step draft head;
Ministral, LLaMA, Qwen, and Soofi can also use a compatible smaller assistant
GGUF. On the CPU paths for these four architecture families, the target
evaluates the complete draft as one token-major micro-batch. Accepted tokens
are emitted unchanged; on the first mismatch, the target token replaces the
draft and the suffix is discarded. A backend without this batched verifier
keeps normal decoding instead of paying draft overhead without a batch benefit.

This mode is useful only when the assistant is much faster than the target and
the two models agree often enough. If the acceptance rate is low, speculative
decoding can be slower than normal decoding.

## Requirements

- Build a release binary first:

```bash
cargo build --release
```

- When using an external assistant, target and assistant need compatible tokenization:
  - same vocabulary size
  - same BOS token ID
  - same EOS token ID
  - assistant context large enough for the prompt
- Use greedy decoding with `--temp 0`. RustyLLM disables MTP for non-greedy
  sampling because the current verifier requires exact target argmax equality.
- Prefer a smaller assistant from the same model family. For example, a 3B
  assistant can be tested against a 14B target, but only keep it if benchmarks
  show a real speedup.

An embedded Qwen draft head needs no `--mtp-assistant`. Set `--mtp-tokens 2`
or greater; the native head predicts one additional token after the target's
first greedy token. It is opt-in: without `--mtp-tokens`, loading a model with
an embedded head does not run a calibration probe.

## Basic Command

Embedded Qwen head:

```bash
./target/release/rusty-llm /path/to/qwen.gguf \
  --mtp-tokens 2 \
  --temp 0 \
  --max-tokens 128 \
  --prompt "Explain speculative decoding in one paragraph."
```

External assistant:

```bash
./target/release/rusty-llm /path/to/target.gguf \
  --mtp-assistant /path/to/assistant.gguf \
  --mtp-tokens 4 \
  --temp 0 \
  --max-tokens 128 \
  --prompt "Explain speculative decoding in one paragraph."
```

At startup, RustyLLM prints the active optimization summary. A working MTP setup
contains an item similar to:

```text
mtp=greedy assistant=/path/to/assistant.gguf draft_tokens=4 adaptive=true min_accept_rate=0.50
```

If MTP is inactive, the summary prints:

```text
mtp=off
```

For an embedded head the summary contains `mtp=greedy embedded`.

## Benchmark Before Keeping It

Always compare the target without MTP against the same target with MTP.

Baseline:

```bash
RUSTY_LLM_METAL=0 ./target/release/rusty-llm /path/to/target.gguf \
  --bench \
  --bench-runs 5 \
  --max-tokens 128 \
  --temp 0 \
  --no-speculative \
  --prompt "Write a concise explanation of KV cache reuse."
```

MTP run:

```bash
RUSTY_LLM_METAL=0 ./target/release/rusty-llm /path/to/target.gguf \
  --bench \
  --bench-runs 5 \
  --max-tokens 128 \
  --temp 0 \
  --mtp-assistant /path/to/assistant.gguf \
  --mtp-tokens 4 \
  --prompt "Write a concise explanation of KV cache reuse."
```

Use `--bench-json` instead of `--bench` when you want machine-readable output:

```bash
RUSTY_LLM_METAL=0 ./target/release/rusty-llm /path/to/target.gguf \
  --bench-json \
  --bench-runs 5 \
  --max-tokens 128 \
  --temp 0 \
  --mtp-assistant /path/to/assistant.gguf \
  --mtp-tokens 4 \
  --prompt "Write a concise explanation of KV cache reuse."
```

The benchmark output includes MTP fields when a draft path is active:

- `accept_rate`: accepted draft tokens divided by drafted tokens
- `draft_tok_s`: assistant draft throughput
- `eff_tok_s`: effective target generation throughput
- `drafted`, `accepted`, `rejected`: token counts used to judge whether MTP
  helps
- `verification_ms`, `verification_batches`: target batch cost
- `recovery_ms`, `recovered_target_steps`: state repair after rejected batches

For the embedded head, `drafted` counts only the token predicted by that head.
The first token comes directly from target logits and is not counted as an
accepted draft.

RustyLLM disables MTP automatically when acceptance remains below the selected
threshold or measured draft overhead exceeds the conservative estimate of
saved target work. Rejected recurrent batches provide an in-situ baseline for
this timing guard, and a non-productive embedded head is disabled after a very
small probe. Still compare effective throughput on normal prompts;
`--no-speculative` remains the explicit off switch.

RustyLLM also prints `recommendation=disable` in the normal generation stats
when the final acceptance rate is below the configured threshold.

## Choosing `--mtp-tokens`

Start conservatively:

```bash
--mtp-tokens 2
```

Then test:

```bash
--mtp-tokens 4
--mtp-tokens 8
```

Higher values are only useful when the assistant and target agree for several
tokens in a row. RustyLLM starts with a conservative draft limit and adapts it
up or down during generation. If a batch diverges, tokens after the first
divergence are discarded.

Use a custom threshold when a target/assistant pair needs a stricter or looser
cutoff:

```bash
--mtp-min-accept-rate 0.6
```

Disable adaptive draft sizing only for controlled experiments:

```bash
--no-mtp-adaptive
```

Keeping adaptation enabled is the safer default for normal use.

## CPU, Metal, and NPU/GPU Comparison

MTP interacts with the active backend. Test CPU and Metal separately on macOS:

```bash
RUSTY_LLM_METAL=0 ./target/release/rusty-llm /path/to/target.gguf \
  --bench-json --bench-runs 5 --temp 0 \
  --mtp-assistant /path/to/assistant.gguf

RUSTY_LLM_METAL=1 ./target/release/rusty-llm /path/to/target.gguf \
  --bench-json --bench-runs 5 --temp 0 \
  --mtp-assistant /path/to/assistant.gguf
```

The faster backend for normal decoding is not always the faster backend for MTP,
because MTP runs both assistant and target work.

## Troubleshooting

`MTP assistant tokenizer mismatch`

The target and assistant have different vocabulary metadata. Use a closer
assistant from the same model family or release line.

`MTP assistant BOS/EOS mismatch`

The tokenizer special token IDs differ. Do not use that assistant with this
target.

`MTP: disabled for non-greedy sampling`

Set `--temp 0`. Current MTP verification is greedy-only.

`MTP: disabled because prompt needs ... tokens`

The prompt is too long for the assistant context after runtime context caps.
Use a shorter prompt, a larger-context assistant, or adjust `--max-context`.

Low `accept_rate`

The assistant is predicting tokens the target does not accept. Reduce
`--mtp-tokens`, try a closer assistant, or disable MTP.

Slower than baseline

This is expected for some target/assistant pairs. Keep the normal path with
`--no-speculative` unless MTP improves measured throughput.

Assistant is close to target size

RustyLLM prints a startup warning when the assistant is at least 75 percent of
the target file size. In that case, the assistant is usually too expensive for
MTP to pay off.

## Practical Notes

- MTP is most likely to help large target models with a much smaller but closely
  related assistant.
- It is less likely to help small 3B targets because the target itself is
  already fast and assistant overhead can dominate.
- Ministral 3B is better treated as an assistant candidate than as a target for
  MTP.
- For deterministic performance measurements, use the same prompt, same
  `--max-tokens`, same `--temp 0`, same backend flag, and at least several
  benchmark runs.
