# Rust Function Reference

This reference documents every non-test Rust function and method under `src/`.
It excludes `extern` declarations and unit-test helpers. The descriptions are
intended as a reading map for the implementation, not as a stability guarantee
for private functions.

For the AI vocabulary used in these descriptions, see
[AI inference for non-AI developers](AI_FOR_DEVELOPERS.md).

## `src/lib.rs`

No functions are defined here. The library root publishes modules and re-exports
the high-level runtime API.

## `src/catalog.rs`

- `ModelEntry::status` returns the display status for a discovered model:
  `projector`, `supported`, or `unsupported`.
- `default_model_dir` resolves the default model search directory from
  `RUSTY_LLM_MODEL_DIR`, `$HOME`, or a relative LM Studio cache fallback.
- `discover_models` recursively scans a directory for GGUF files and returns
  inspected, sorted model entries.
- `resolve_model_path` accepts a path, directory, or model selector and resolves
  it to one GGUF file.
- `select_model` selects one supported text model from discovered entries using
  a lenient selector.
- `print_model_list` prints a compact table of discovered models.
- `choose_from_directory` chooses one supported model from a directory or reports
  an ambiguity.
- `collect_gguf_files` recursively gathers `.gguf` files.
- `inspect_model` reads GGUF metadata and builds a `ModelEntry`.
- `matching_entries` finds entries matching a selector across ID, repository,
  filename, and metadata name.
- `format_ambiguous` formats a selector ambiguity error.
- `format_model_choices` renders model choices for CLI errors.
- `truncate` shortens display strings without splitting Unicode scalar values.

## `src/gguf.rs`

- `GGMLType::from` converts raw GGUF tensor type IDs into `GGMLType` variants.
- `GGMLType::block_bytes` returns the encoded byte width of one quantization
  block.
- `GGMLType::block_size` returns how many scalar values one quantization block
  represents.
- `GGMLType::data_size` computes the tensor byte size for a number of elements.
- `MetaValue::as_u32` reads integer-like metadata as `u32`.
- `MetaValue::as_f32` reads numeric metadata as `f32`.
- `MetaValue::as_str` reads string metadata.
- `MetaValue::as_string_array` reads string-array metadata.
- `MetaValue::as_f32_array` reads float-array metadata.
- `TensorInfo::numel` returns the tensor element count from its dimensions.
- `Cursor::new` creates a little-endian parser over a byte slice.
- `Cursor::read_u8`, `read_u16`, `read_u32`, `read_i32`, `read_u64`,
  `read_i64`, `read_f32`, and `read_f64` read primitive values.
- `Cursor::read_string` reads a GGUF length-prefixed UTF-8 string.
- `Cursor::read_bool` reads a GGUF boolean.
- `Cursor::read_value` reads one typed GGUF metadata value.
- `GGUFFile::parse` parses a GGUF file and prints summary metadata.
- `GGUFFile::parse_quiet` parses a GGUF file without printing.
- `GGUFFile::parse_inner` implements shared GGUF header, metadata, tensor, and
  offset parsing.
- `GGUFFile::get_u32` returns a `u32` metadata value or a default.
- `GGUFFile::get_f32` returns an `f32` metadata value or a default.
- `GGUFFile::get_str` returns a string metadata value when present.

## `src/mmap.rs`

- `MmapFile::open` opens and memory-maps a model file on native targets.
- `MmapFile::as_slice` returns the mapped bytes.
- `MmapFile::len` returns the mapping length.
- `MmapFile::is_empty` reports whether the mapping has zero length.
- `MmapFile::drop` unmaps native memory when the wrapper is dropped.

## `src/tokenizer.rs`

- `Tokenizer::from_metadata` builds a tokenizer from GGUF tokenizer metadata.
- `Tokenizer::encode` encodes text and applies the configured BOS token policy.
- `Tokenizer::encode_without_bos` encodes text without adding BOS.
- `Tokenizer::decode_raw` returns the raw vocabulary string for a token ID.
- `Tokenizer::decode_token` decodes one token into user-facing text.
- `Tokenizer::vocab_size` returns the vocabulary size.
- `Tokenizer::adds_bos_token` reports whether text encoding prepends BOS.
- `Tokenizer::special_id` looks up a special token by literal text.
- `Tokenizer::encode_sentencepiece` runs the SentencePiece-style BPE path.
- `Tokenizer::encode_gpt2_bpe` runs reversible byte-level GPT-2 BPE.
- `Tokenizer::encode_from_pieces` maps tokenizer pieces to IDs with byte
  fallback.
- `Tokenizer::decode_gpt2_bytes` reverses GPT-2 byte-level token text.
- `pretokenize_gpt2` splits text into GPT-2-compatible pre-token pieces.
- `build_byte_maps` builds the reversible GPT-2 byte-to-Unicode maps.

## `src/sampling.rs`

- `Rng::new` creates the deterministic xorshift PRNG used by sampling.
- `Rng::next_f32` returns a pseudo-random float in `[0, 1)`.
- `SamplerConfig::default` provides conservative sampling defaults.
- `sample` samples one token using a temporary candidate buffer.
- `sample_with_scratch` samples one token while reusing caller-owned scratch
  storage.
- `sample_top_k` applies top-k, optional top-p truncation, softmax, and random
  selection over candidates.
- `bubble_up_last` keeps the bounded candidate list sorted by descending logit.
- `argmax_token` returns the highest-logit token without finite filtering.
- `argmax_finite_token` returns the highest finite-logit token.

## `src/simd.rs`

- `has_avx2_fma` detects x86_64 AVX2/FMA support.
- `f16_to_f32` converts IEEE-754 half precision bits to `f32`.
- `set_num_threads` sets the global worker count for parallel kernels.
- `num_threads` returns the configured worker count.
- `f16_lookup` returns the lazy half-to-float lookup table.
- `parallel_matvec_f32`, `parallel_matvec_u8`, and `parallel_matvec` dispatch
  matrix-vector work across the worker pool.
- `Q4KMatvec3Job::work_items` reports how many rows a fused Q4_K triple-matvec
  job contains.
- `clipped_range` maps a global worker range onto one output slice.
- `WorkerJob::workers` returns the worker count requested by a queued job.
- `WorkerPool::new` creates the reusable worker pool.
- `WorkerPool::run`, `run_q4k_matvec3`, and `run_job` execute queued matvec
  jobs and wait for completion.
- `worker_loop` is the background loop run by each worker thread.
- `worker_pool` returns the process-wide worker pool.
- `f16_to_f32_soft` is the portable half-to-float conversion fallback.
- `dot_f32` computes a float dot product.
- `axpy_f32` computes `out += alpha * x`.
- `scale_f32` scales a vector in place.
- `scale_add_f32` computes `out = out * scale + add`.
- `axpy_f32_scalar`, `scale_f32_scalar`, and `scale_add_f32_scalar` are scalar
  fallbacks for vector update kernels.
- `dot_q8_0_f32`, `dot_q4_0_f32`, `dot_q4_k_f32`, `dot_q6_k_f32`, and
  `dot_mxfp4_f32` compute quantized-row dot products against an `f32` vector.
- `matvec_f32`, `matvec_q8_0`, `matvec_q4_0`, `matvec_q4_k`,
  `matvec_q6_k`, and `matvec_mxfp4` allocate and return matrix-vector results.
- `matvec_f32_into`, `matvec_q8_0_into`, `matvec_q4_0_into`,
  `matvec_q4_k_into`, `matvec_q4_k3_into`, `matvec_q4_k2_into`,
  `matvec_q6_k_into`, and `matvec_mxfp4_into` write matrix-vector results into
  caller-owned output buffers.
- `matvec_quant2_into` and `matvec_quant3_into` fuse two or three supported
  quantized projections that share the same input vector.
- `dequant_row_q8_0`, `dequant_row_q4_0`, `dequant_row_q4_k`,
  `dequant_row_q6_k`, and `dequant_row_mxfp4` decode one quantized row to
  `f32` values.
- `dot_f32_scalar`, `dot_q8_0_f32_scalar`, `dot_q4_0_f32_scalar`,
  `dot_q4_k_f32_scalar`, `dot_q6_k_f32_scalar`, and
  `dot_mxfp4_f32_scalar` are portable scalar dot-product implementations.
- `get_scale_min_k4` extracts scale and minimum metadata for a Q4_K sub-block.
- `mxfp4_nibble_to_f32` maps MXFP4 nibble values to floats.

## `src/metal.rs`

- `available` reports whether the Metal shim is compiled and available.
- `enabled` checks whether Metal use was requested and is available.
- `requested` reads the `RUSTY_LLM_METAL` environment flag.
- `q6k_enabled` checks the `RUSTY_LLM_METAL_Q6K` environment flag.
- `q4k_matvec_into`, `q6k_matvec_into`, `q4k_matvec2_into`, and
  `q4k_matvec3_into` try selected Metal matrix-vector kernels and fall back by
  returning `false`.
- `q4k_single_should_use_metal` avoids dispatching very small Q4_K workloads to
  Metal.
- `q4k_matvec_raw`, `q4k_matvec2_raw`, `q6k_matvec_raw`, and
  `q4k_matvec3_raw` call the Objective-C Metal shim on macOS or return `false`
  on unsupported builds.
- `env_flag` reads optional boolean environment flags.
- `parse_env_flag` interprets truthy and falsey environment values.

## `src/model.rs`

- `Config::from_gguf` derives transformer dimensions and architecture settings
  from GGUF metadata.
- `RawTensorData::clone`, `owned`, `view`, and `as_slice` manage owned or
  borrowed tensor bytes.
- `Weight::matvec`, `matvec_into`, `row`, `row_into`, and `row_f32` run typed
  tensor access for dense and quantized weights.
- `try_quant_matvec3_into` and `try_quant_matvec2_into` run optional fused
  quantized projections when feature/target gates allow them.
- `ExpertWeight::matvec_expert` and `matvec_expert_into` run one expert slice
  from a mixture-of-experts tensor.
- `KVCache::new` allocates per-layer key/value cache storage.
- `build_rope_inv_freq` builds rotary embedding inverse frequencies.
- `build_rope_inv_freq_gpt_oss` builds GPT-OSS rotary frequencies and scaling.
- `DecodeBuffer::new` allocates reusable temporary buffers for decoding.
- `load_weight` loads one named tensor from GGUF metadata and bytes.
- `load_f32_vec` loads a required float vector tensor.
- `load_optional_f32_vec` loads an optional float vector tensor.
- `load_expert_weight` loads one expert tensor by trying naming variants.
- `load_model`, `load_gpt_oss_model`, and `load_gemma4_model` load supported
  model-family weight structures.
- `rms_norm_into` applies RMS normalization into a caller-owned output buffer.
- `apply_rope` applies standard rotary embeddings to query/key vectors.
- `fast_attn_enabled` checks the fast attention approximation flag.
- `exp_attn` selects exact or approximate exponentiation for attention.
- `fast_exp_approx` approximates `exp` for optional faster attention.
- `online_attention_with_sink` computes attention with an additional sink term.
- `online_attention` computes streaming softmax attention over cached keys and
  values.
- `silu` applies the SiLU activation.
- `swiglu_gpt_oss` applies the GPT-OSS SwiGLU variant.
- `apply_rope_gpt_oss` applies the GPT-OSS rotary embedding variant.
- `softmax_selected_into` normalizes selected expert-router logits.
- `forward_gpt_oss`, `forward`, and `forward_gemma4` allocate and return logits
  for one token.
- `forward_gpt_oss_into`, `forward_into`, and `forward_gemma4_into` write
  logits into caller-owned buffers.
- `find_alternative` searches alternate tensor names while loading Gemma 4
  weights.
- `validate_shape` checks expected tensor dimensions while loading Gemma 4
  weights.
- `forward_hidden`, `forward_hidden_gpt_oss`, and `forward_hidden_gemma4`
  return final hidden states for embedding-style workloads.

## `src/runtime.rs`

- `ChatMessage::user` and `ChatMessage::assistant` construct common chat
  messages.
- `GenerationOptions::default` supplies generation defaults.
- `GenerationOptions::validate` checks bounds for generation settings.
- `mean_pool_in_place` mean-pools hidden states across token positions.
- `l2_normalize_in_place` normalizes a vector for cosine similarity.
- `cosine_similarity` computes cosine similarity with validation.
- `architecture_supported` checks whether a GGUF architecture is loadable.
- `is_gemma_arch` identifies Gemma-family architecture strings that share the
  Gemma loader and runtime profile.
- `Runner::from_gguf_bytes` builds a runner from in-memory GGUF bytes.
- `Runner::from_path` memory-maps and loads a runner from a GGUF file path.
- `Runner::architecture`, `model_name`, `tokenizer`, `gguf`, and `config`
  expose loaded model metadata.
- `Runner::kernel_benchmark` benchmarks representative matrix-vector kernels.
- `Runner::generate` produces one non-streaming completion.
- `Runner::generate_chat` renders chat messages and produces a response.
- `Runner::generate_stream` produces tokens and calls a callback as text
  becomes available.
- `Runner::generate_chat_stream` streams a chat response.
- `Runner::forward_token_into` runs one token through the active model family.
- `Runner::forward_hidden_token` returns one token hidden state for embeddings.
- `Runner::embed` returns a mean-pooled, L2-normalized embedding.
- `Runner::is_stop_token` checks built-in stop tokens.
- `Runner::render_messages`, `render_plain_messages`,
  `render_gpt_oss_messages`, `render_gemma_turn_messages`, and
  `render_header_chat_messages` convert chat messages into prompt token IDs.
- `Runner::chat_template_kind` classifies supported chat template styles.
- `Runner::longest_common_prefix` finds reusable KV-cache prefix length.
- `Runner::kv_dims` returns cache dimensions for the active model family.
- `Runner::new_session` creates a reusable session for chat generation.
- `Runner::generate_chat_with_session` generates a chat response with KV-cache
  reuse.
- `measure_matvec` times one matrix-vector operation repeatedly.
- `measure_kernel` is the generic timing loop used by kernel benchmarks.
- `weight_shape` returns a benchmarkable matrix shape for a weight.
- `weight_dtype` returns a display dtype for a weight.
- `deterministic_bench_vector` builds reproducible benchmark input data.
- `WasmRunner::new`, `model_name`, `generate`, `embed`, and
  `cosine_similarity` expose the runner to WASM builds.

## `src/session.rs`

- `Session::new` creates a session from a KV cache and decode buffer.
- `Session::reset` clears reusable session state.
- `SessionStore::new` creates a bounded LRU-style session store.
- `SessionStore::max_cached_tokens` returns the per-session cache-token limit.
- `SessionStore::get_or_create` returns an existing session or inserts a new
  one.
- `SessionStore::delete` removes a session by ID.
- `SessionStore::len` returns the number of stored sessions.
- `SessionStore::is_empty` reports whether the store has no sessions.

## `src/server.rs`

- `ServeOptions::is_tls` reports whether both TLS certificate and key paths are
  configured.
- `ApiMessageContent::into_text` flattens OpenAI text/image content into text.
- `StopSpec::into_vec` normalizes stop strings.
- `make_cache_stats` builds cache telemetry for HTTP responses.
- `ActiveConnectionGuard::drop` releases one active connection slot.
- `HttpError::bad_request`, `payload_too_large`, and `request_timeout` build
  HTTP parsing errors.
- `serve` starts the HTTP or HTTPS server loop.
- `try_acquire_connection_slot` enforces the configured connection limit.
- `handle_plain_connection` and `handle_tls_connection` wrap accepted sockets.
- `handle_connection` reads one request, routes it, and writes a response.
- `chat_ui_route` maps UI paths to built-in UI assets.
- `is_streaming_request` detects streaming completion/chat requests.
- `route_streaming_request` writes Server-Sent Events for OpenAI-compatible
  streaming requests.
- `route_request` dispatches non-streaming HTTP routes.
- `generate_with_optional_session` runs chat generation with optional session
  reuse.
- `route_generate`, `route_openai_completion`, `route_openai_chat`,
  `route_embeddings`, `route_ollama_tags`, `route_ollama_generate`,
  `route_ollama_chat`, and `route_ollama_embeddings` implement API endpoints.
- `apply_ollama_options` maps Ollama generation options to RustyLLM options.
- `parse_ollama_messages` converts Ollama messages to runtime chat messages.
- `dominant_quantization` reports the dominant loaded tensor format.
- `append_chat_history` appends JSONL-style chat history entries.
- `history_message_json` serializes a chat message for history.
- `json_response` serializes successful JSON responses.
- `is_json_content_type` accepts JSON content types with optional parameters.
- `apply_generation_overrides` applies request-specific generation options.
- `parse_api_messages` converts OpenAI-style messages to runtime messages.
- `resolve_model` resolves optional requested model IDs for responses.
- `advertised_model_ids` returns model IDs and compatibility aliases.
- `model_aliases_for_arch` returns architecture-specific model aliases.
- `unix_timestamp` returns the current Unix timestamp.
- `iso_timestamp` returns a compact timestamp string.
- `json_error` formats JSON error bodies.
- `read_http_request` parses one bounded HTTP request.
- `chat_ui_html` and `expert_chat_ui_html` return embedded HTML assets.
- `write_http_response` writes a normal text/html response.
- `write_http_response_with_content_type` writes a response with explicit
  content type and CORS headers.
- `load_tls_config` loads TLS certificate and private-key PEM files.

## `src/main.rs`

- `print_usage` prints CLI help.
- `parse_arg` parses the value following a typed CLI flag.
- `set_model_selector` records the positional or flag-selected model.
- `main` runs the CLI and prints fatal errors.
- `run` parses CLI arguments and dispatches the selected mode.
- `inspect_model_file` prints a JSON compatibility report for a GGUF file.
- `run_benchmark` runs a text-generation benchmark.
- `run_benchmark_json` runs the benchmark and prints JSON.
- `run_kernel_benchmark` runs kernel-level benchmarks.
- `run_repl` runs the interactive chat loop.
- `append_cli_history` appends CLI chat turns to a history file.
- `chat_message_json` serializes a chat message for CLI history.
- `unix_timestamp` returns the current Unix timestamp.
- `atty_is_stdin` detects whether stdin is a terminal.

## `src/bin/analyze_gguf.rs`

- `main` loads a GGUF file, prints metadata, and optionally compares tensor
  blocks.
- `compare_blocks` compares selected quantized blocks between two tensors.
- `format_dims` formats tensor dimensions for display.

## `src/bin/embedding_demo.rs`

- `usage` prints command usage.
- `main` embeds two input strings and prints cosine similarity.

## `src/bin/list_tensors.rs`

- `main` loads a GGUF file and prints tensor names, dtypes, and dimensions.
