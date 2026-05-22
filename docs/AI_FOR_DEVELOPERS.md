# AI Inference for Non-AI Developers

RustyLLM can be understood as a data-processing program. It reads a model file,
turns text into numbers, repeatedly applies matrix math to those numbers, and
turns the selected output numbers back into text.

You do not need machine-learning background to follow the code. The important
idea is that a trained model is mostly a large collection of numeric arrays. At
runtime, RustyLLM does not train the model or change what it knows. It only
loads those arrays and uses them to compute the next likely token.

## The Short Version

For a text prompt such as:

```text
Explain Rust ownership briefly.
```

RustyLLM does this:

1. Loads a `.gguf` file that contains model metadata, tokenizer data, and model
   weights.
2. Converts the prompt text into token IDs, which are integers.
3. Runs the token IDs through the model weights to produce logits, which are
   scores for possible next tokens.
4. Chooses one next token with greedy decoding or sampling.
5. Appends that token to the context and repeats the process.
6. Decodes the generated token IDs back into text.

That loop is inference.

## Vocabulary

- Model: the loaded numeric data and metadata used to predict text. In RustyLLM,
  the model usually comes from one `.gguf` file.
- Inference: running an already-trained model to get an output. This project is
  for inference, not training.
- Token: a small piece of text represented by an integer ID. A token can be a
  word, part of a word, punctuation, whitespace, or a byte fallback.
- Tokenizer: the component that converts text to token IDs and token IDs back to
  text.
- Vocabulary: the table of all token IDs the model understands.
- Weight: a numeric array loaded from the model file. Weights are the fixed data
  learned during training.
- Tensor: a multi-dimensional numeric array. In this codebase, tensors are
  usually model weights such as matrices and vectors.
- Matrix-vector multiply: the main math operation used during inference. A
  matrix from the model is multiplied by the current activation vector to create
  the next activation vector.
- Logits: raw output scores for each possible next token. Higher usually means
  more likely.
- Sampling: the process of choosing the next token from logits. Temperature,
  top-k, and top-p change how predictable or varied the choice is.
- Embedding: a vector representation of text. Similar text should produce
  vectors with high cosine similarity.
- Context: the tokens currently visible to the model, including the prompt and
  already generated tokens.
- KV cache: saved attention state from previous tokens. It avoids recomputing
  all previous context for every new token.
- Quantization: storing weights in fewer bits than normal `f32` floats. This
  makes files smaller and inference cheaper, but the code must decode or operate
  on compact formats such as `Q4_K` or `Q8_0`.
- GGUF: the model-file format parsed by RustyLLM. It contains metadata, tokenizer
  data, tensor descriptions, and raw tensor bytes.

## How This Maps to the Code

- `src/gguf.rs` is the file-format reader. It is similar to writing a parser for
  a binary asset format.
- `src/mmap.rs` maps the model file into memory. This lets RustyLLM reference
  large model bytes without copying the whole file.
- `src/tokenizer.rs` is text serialization for models. It maps user strings to
  integer IDs and output IDs back to strings.
- `src/model.rs` is the model executor. It loads named tensors and implements
  the per-token forward pass.
- `src/simd.rs` is the math layer. It contains normal Rust implementations plus
  faster CPU-specific versions for repeated vector operations.
- `src/sampling.rs` decides which token to emit from the model's output scores.
- `src/runtime.rs` is the ergonomic API. It combines loading, tokenization,
  forward passes, sampling, chat formatting, embeddings, and optional cache
  reuse.
- `src/server.rs` wraps the same runtime in HTTP routes.
- `src/main.rs` wraps the same runtime in a CLI.

## The Generation Loop in Plain Terms

The core runtime loop looks conceptually like this:

```text
tokens = tokenizer.encode(prompt)
cache = empty key/value cache

for each token we want to generate:
    logits = model.forward(last_token, cache)
    next_token = sampler.choose(logits)
    tokens.push(next_token)
    print tokenizer.decode(next_token)
```

The real implementation has more detail: prompt prefill, stop strings, chat
templates, reusable buffers, quantized weights, and model-family differences.
But the loop above is the mental model.

## What RustyLLM Does Not Do

- It does not train models.
- It does not download models.
- It does not provide distributed serving.
- It does not hide model execution behind a large external runtime.
- It does not make every model architecture work automatically; it supports the
  families implemented in `src/model.rs` and checked in `src/runtime.rs`.

## Reading Order

If you have systems or backend experience but no AI background, read in this
order:

1. `src/gguf.rs`: understand what data is loaded.
2. `src/tokenizer.rs`: understand why text becomes integers.
3. `src/sampling.rs`: understand how output scores become one selected token.
4. `src/runtime.rs`: understand the generation loop from a user's perspective.
5. `src/model.rs`: understand how the model computes logits.
6. `src/simd.rs`: understand how the hot math paths are optimized.
7. `src/server.rs` and `src/main.rs`: understand how the runtime is exposed.

The lower you go in that list, the more AI-specific math and model-family
details you will see.
