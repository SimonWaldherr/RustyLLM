use crate::gguf::{GGMLType, GGUFFile};
use crate::model::{self, Config, DecodeBuffer, GptOssWeights, KVCache, ModelWeights, Weight};
use crate::sampling::{self, SamplerConfig};
use crate::tokenizer::Tokenizer;
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Embedding result returned by [`Runner::embed`].
#[derive(Clone, Debug)]
pub struct EmbeddingResult {
    /// L2-normalized embedding vector of dimension `config.dim`.
    pub embedding: Vec<f32>,
    /// Number of tokens in the input prompt.
    pub token_count: usize,
}

#[derive(Clone, Debug)]
pub enum ChatRole {
    System,
    User,
    Assistant,
}

#[derive(Clone, Debug)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
}

impl ChatMessage {
    /// Constructs a user chat message.
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::User,
            content: content.into(),
        }
    }

    /// Constructs an assistant chat message.
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::Assistant,
            content: content.into(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct GenerationOptions {
    pub max_tokens: usize,
    pub sampler: SamplerConfig,
    pub seed: u64,
    pub system_prompt: String,
    /// Stop generation when any of these strings appears in the output.
    /// The matched sequence is not included in the returned text.
    pub stop_sequences: Vec<String>,
}

impl Default for GenerationOptions {
    /// provides conservative sampling defaults.
    fn default() -> Self {
        Self {
            max_tokens: 256,
            sampler: SamplerConfig::default(),
            seed: 0,
            system_prompt: String::from("You are a helpful assistant."),
            stop_sequences: Vec::new(),
        }
    }
}

impl GenerationOptions {
    /// Validates generation option ranges before decoding.
    pub fn validate(&self) -> Result<(), String> {
        if self.max_tokens == 0 {
            return Err(String::from("max_tokens must be greater than 0."));
        }
        if !self.sampler.temperature.is_finite() || self.sampler.temperature < 0.0 {
            return Err(String::from("temperature must be a finite number >= 0."));
        }
        if !self.sampler.top_p.is_finite() || self.sampler.top_p <= 0.0 || self.sampler.top_p > 1.0
        {
            return Err(String::from("top_p must be in the range (0, 1]."));
        }
        if !self.sampler.repeat_penalty.is_finite() || self.sampler.repeat_penalty <= 0.0 {
            return Err(String::from("repeat_penalty must be a finite number > 0."));
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct GenerationStats {
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub prefill_time: Duration,
    pub decode_time: Duration,
    pub total_time: Duration,
    /// Number of prompt tokens that were served from the KV cache without
    /// re-evaluation.  Always 0 for stateless (non-session) generation.
    pub cached_tokens: usize,
}

#[derive(Clone, Debug)]
pub struct GenerationResult {
    pub text: String,
    pub stats: GenerationStats,
}

#[derive(Clone, Debug)]
pub struct LoadInfo {
    pub file_size_bytes: usize,
    pub load_time: Duration,
}

#[derive(Clone, Debug)]
pub struct KernelBenchRow {
    pub name: String,
    pub dtype: String,
    pub rows: usize,
    pub cols: usize,
    pub runs: usize,
    pub avg_ms: f64,
    pub total_ms: f64,
}

const RECENT_TOKEN_LIMIT: usize = 64;

#[inline]
/// Mean-pools consecutive token vectors in place.
fn mean_pool_in_place(values: &mut [f32], sample_count: usize) {
    if sample_count == 0 {
        return;
    }
    let scale = 1.0 / sample_count as f32;
    for value in values.iter_mut() {
        *value *= scale;
    }
}

/// Copies the trailing tokens used by the repetition-penalty sampler.
fn recent_token_tail(tokens: &[u32]) -> Vec<u32> {
    let start = tokens.len().saturating_sub(RECENT_TOKEN_LIMIT);
    tokens[start..].to_vec()
}

/// Appends one token while keeping the repetition-penalty window bounded.
fn push_recent_token(recent: &mut Vec<u32>, token: u32) {
    if recent.len() == RECENT_TOKEN_LIMIT {
        recent.copy_within(1.., 0);
        *recent.last_mut().expect("recent token buffer is non-empty") = token;
    } else {
        recent.push(token);
    }
}

#[inline]
/// Normalizes a vector to unit length when possible.
fn l2_normalize_in_place(values: &mut [f32]) {
    let norm: f32 = values.iter().map(|&v| v * v).sum::<f32>().sqrt();
    if norm > 1e-8 {
        let inv = 1.0 / norm;
        for value in values.iter_mut() {
            *value *= inv;
        }
    }
}

/// Compute cosine similarity between two embeddings.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> Result<f32, String> {
    if a.len() != b.len() {
        return Err(format!(
            "cosine_similarity: dimension mismatch ({} vs {})",
            a.len(),
            b.len()
        ));
    }
    if a.is_empty() {
        return Err(String::from("cosine_similarity: empty vectors"));
    }
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom <= 1e-12 {
        return Err(String::from(
            "cosine_similarity: zero-norm vector encountered",
        ));
    }
    Ok(dot / denom)
}

enum LoadedWeights {
    Standard(ModelWeights),
    GptOss(GptOssWeights),
    Gemma4(crate::model::Gemma4Weights),
}

pub struct Runner {
    gguf: GGUFFile,
    arch: String,
    tok: Tokenizer,
    config: Config,
    weights: LoadedWeights,
    /// Serialises concurrent generation calls.
    /// The worker pool's job slot is single-entry; two simultaneous
    /// forward passes would race on it and produce corrupted output.
    generation_lock: Mutex<()>,
    #[allow(dead_code)]
    #[cfg(not(target_family = "wasm"))]
    mapped_model: Option<crate::mmap::MmapFile>,
}

/// Reports whether the architecture string maps to a supported loader.
pub fn architecture_supported(arch: &str) -> bool {
    matches!(
        arch,
        "llama"
            | "llama2"
            | "llama3"
            | "mistral"
            | "mistral3"
            | "mixtral"
            | "ministral"
            | "qwen2"
            | "qwen3"
            | "gpt-oss"
            | "gemma"
            | "gemma2"
            | "gemma4"
            | "granite"
            | "granite3"
            | "granite4"
            | "deepseek"
            | "deepseek-v2"
            | "deepseek2"
            | "nemotron"
            | "hermes"
            | "phi"
            | "phi2"
            | "phi3"
            | "phi4"
            | "falcon"
            | "falcon3"
            | "stablelm"
            | "starcoder2"
            | "command-r"
            | "cohere"
            | "internlm2"
            | "olmo"
            | "olmo2"
            | "exaone"
            | "solar"
            | "yi"
            | "arctic"
            | "nomic-bert"
            | "nomic-embed"
            | "text-embedding-nomic-embed-text"
    )
}

/// Checks that every tensor in `gguf` uses a quantization type that the
/// inference kernels support.  Returns a descriptive error for any tensor
/// whose dtype would cause a panic inside `load_weight`.
fn validate_tensor_dtypes(gguf: &GGUFFile) -> Result<(), String> {
    for tensor in &gguf.tensors {
        match tensor.dtype {
            GGMLType::F32
            | GGMLType::F16
            | GGMLType::Q4_0
            | GGMLType::Q4_1
            | GGMLType::Q5_0
            | GGMLType::Q5_1
            | GGMLType::Q8_0
            | GGMLType::Q8_1
            | GGMLType::Q4_K
            | GGMLType::Q6_K
            | GGMLType::MXFP4 => {}
            unsupported => {
                return Err(format!(
                    "Tensor '{}' uses unsupported quantization type {:?}. \
                     Please re-quantize the model using a supported format: \
                     F16, Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, Q4_K, or Q6_K.",
                    tensor.name, unsupported
                ));
            }
        }
    }
    Ok(())
}

impl Runner {
    /// Loads a runner from GGUF bytes already present in memory.
    pub fn from_gguf_bytes(data: &[u8]) -> Result<Self, String> {
        let gguf = GGUFFile::parse(data)?;
        let arch = gguf
            .get_str("general.architecture")
            .unwrap_or("llama")
            .to_string();

        if !architecture_supported(&arch) {
            return Err(format!(
                "Unsupported architecture: {}. Please ensure you are using a supported GGUF architecture.",
                arch
            ));
        }

        validate_tensor_dtypes(&gguf)?;

        let tok = Tokenizer::from_metadata(&gguf.metadata);
        let (config, weights) = match arch.as_str() {
            "gpt-oss" => {
                let (config, weights) = model::load_gpt_oss_model(data, &gguf, false);
                (config, LoadedWeights::GptOss(weights))
            }
            "gemma" | "gemma2" | "gemma4" => {
                let (config, weights) = model::load_gemma4_model(data, &gguf, false);
                (config, LoadedWeights::Gemma4(weights))
            }
            // Platzhalter für DeepSeek, weitere Loader können analog ergänzt werden
            _ => {
                let (config, weights) = model::load_model(data, &gguf, false);
                (config, LoadedWeights::Standard(weights))
            }
        };

        Ok(Self {
            gguf,
            arch,
            tok,
            config,
            weights,
            generation_lock: Mutex::new(()),
            #[cfg(not(target_family = "wasm"))]
            mapped_model: None,
        })
    }

    #[cfg(not(target_family = "wasm"))]
    /// Loads a runner by memory-mapping a GGUF file path.
    pub fn from_path(path: &str) -> Result<(Self, LoadInfo), String> {
        let t0 = Instant::now();
        let mmap = crate::mmap::MmapFile::open(path)
            .map_err(|err| format!("Failed to open model: {}", err))?;
        let file_size_bytes = mmap.len();
        let gguf = GGUFFile::parse(mmap.as_slice())?;
        let arch = gguf
            .get_str("general.architecture")
            .unwrap_or("llama")
            .to_string();

        if !architecture_supported(&arch) {
            return Err(format!(
                "Unsupported architecture: {}. Please ensure you are using a supported GGUF architecture.",
                arch
            ));
        }

        validate_tensor_dtypes(&gguf)?;

        let tok = Tokenizer::from_metadata(&gguf.metadata);
        let (config, weights) = match arch.as_str() {
            "gpt-oss" => {
                let (config, weights) = model::load_gpt_oss_model(mmap.as_slice(), &gguf, true);
                (config, LoadedWeights::GptOss(weights))
            }
            "gemma" | "gemma2" | "gemma4" => {
                let (config, weights) = model::load_gemma4_model(mmap.as_slice(), &gguf, true);
                (config, LoadedWeights::Gemma4(weights))
            }
            // Platzhalter für DeepSeek, weitere Loader können analog ergänzt werden
            _ => {
                let (config, weights) = model::load_model(mmap.as_slice(), &gguf, true);
                (config, LoadedWeights::Standard(weights))
            }
        };

        let runner = Self {
            gguf,
            arch,
            tok,
            config,
            weights,
            generation_lock: Mutex::new(()),
            mapped_model: Some(mmap),
        };
        let load_time = t0.elapsed();
        Ok((
            runner,
            LoadInfo {
                file_size_bytes,
                load_time,
            },
        ))
    }

    /// Returns the loaded GGUF architecture string.
    pub fn architecture(&self) -> &str {
        &self.arch
    }

    /// Returns the optional model name from GGUF metadata.
    pub fn model_name(&self) -> Option<&str> {
        self.gguf.get_str("general.name")
    }

    /// Returns the tokenizer used by this runner.
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tok
    }

    /// Returns the parsed GGUF metadata and tensor directory.
    pub fn gguf(&self) -> &GGUFFile {
        &self.gguf
    }

    /// Returns the model configuration used for inference.
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Benchmarks representative projection kernels for the loaded model.
    pub fn kernel_benchmark(
        &self,
        runs: usize,
        requested_layer: usize,
    ) -> Result<(usize, Vec<KernelBenchRow>), String> {
        if runs == 0 {
            return Err(String::from("--kernel-bench-runs must be greater than 0."));
        }
        let LoadedWeights::Standard(weights) = &self.weights else {
            return Err(String::from(
                "--kernel-bench currently supports standard transformer weights only.",
            ));
        };
        let layer = requested_layer.min(weights.layers.len().saturating_sub(1));
        let mut rows = Vec::new();
        let mut row_out = Vec::new();
        rows.push(measure_kernel(
            "token_embd.row",
            &weights.token_embd,
            self.config.vocab_size,
            self.config.dim,
            runs,
            || {
                row_out = weights.token_embd.row(0, self.config.dim);
                std::hint::black_box(row_out.len());
            },
        ));
        if weights.layers.is_empty() {
            return Ok((layer, rows));
        }
        let layer_weights = &weights.layers[layer];
        let dim_input = deterministic_bench_vector(self.config.dim);
        let attn_out_input =
            deterministic_bench_vector(self.config.n_heads * self.config.value_dim);
        let hidden_input = deterministic_bench_vector(self.config.hidden_dim);
        let mut out = Vec::new();
        rows.push(measure_matvec(
            "attn_q",
            &layer_weights.wq,
            &dim_input,
            runs,
            &mut out,
        ));
        rows.push(measure_matvec(
            "attn_k",
            &layer_weights.wk,
            &dim_input,
            runs,
            &mut out,
        ));
        rows.push(measure_matvec(
            "attn_v",
            &layer_weights.wv,
            &dim_input,
            runs,
            &mut out,
        ));
        rows.push(measure_matvec(
            "attn_output",
            &layer_weights.wo,
            &attn_out_input,
            runs,
            &mut out,
        ));
        rows.push(measure_matvec(
            "ffn_gate",
            &layer_weights.w1,
            &dim_input,
            runs,
            &mut out,
        ));
        rows.push(measure_matvec(
            "ffn_up",
            &layer_weights.w3,
            &dim_input,
            runs,
            &mut out,
        ));
        rows.push(measure_matvec(
            "ffn_down",
            &layer_weights.w2,
            &hidden_input,
            runs,
            &mut out,
        ));
        rows.push(measure_matvec(
            "output",
            &weights.output,
            &dim_input,
            runs,
            &mut out,
        ));
        Ok((layer, rows))
    }

    /// Generates a complete response for a plain prompt.
    pub fn generate(
        &self,
        prompt: &str,
        options: &GenerationOptions,
    ) -> Result<GenerationResult, String> {
        let messages = [ChatMessage::user(prompt)];
        self.generate_chat(&messages, options)
    }

    /// Generates a complete response for chat messages.
    pub fn generate_chat(
        &self,
        messages: &[ChatMessage],
        options: &GenerationOptions,
    ) -> Result<GenerationResult, String> {
        self.generate_chat_stream(messages, options, |_| {})
    }

    /// Generates a prompt response while streaming decoded text chunks.
    pub fn generate_stream<F>(
        &self,
        prompt: &str,
        options: &GenerationOptions,
        on_token: F,
    ) -> Result<GenerationResult, String>
    where
        F: FnMut(&str),
    {
        let messages = [ChatMessage::user(prompt)];
        self.generate_chat_stream(&messages, options, on_token)
    }

    /// Generates a chat response while streaming decoded text chunks.
    pub fn generate_chat_stream<F>(
        &self,
        messages: &[ChatMessage],
        options: &GenerationOptions,
        mut on_token: F,
    ) -> Result<GenerationResult, String>
    where
        F: FnMut(&str),
    {
        let _guard = self
            .generation_lock
            .lock()
            .expect("generation lock poisoned");
        options.validate()?;

        if messages.is_empty() {
            return Err(String::from("No prompt provided."));
        }

        let total_start = Instant::now();
        let tokens = self.render_messages(messages, &options.system_prompt);
        if tokens.is_empty() {
            return Err(String::from("Prompt rendered to zero tokens."));
        }

        let cache_len = std::cmp::min(
            self.config.max_seq_len,
            tokens.len() + options.max_tokens + 1,
        );

        // For architectures with per-layer layouts (Gemma-4), compute the
        // maximum per-layer head/value sizes so we can allocate buffers and
        // the KV cache with safe upper bounds.
        let (kv_k_dim, kv_v_dim, max_head_dim, max_n_kv_heads, max_value_dim) = self.kv_dims();

        let mut cache = KVCache::new(self.config.n_layers, kv_k_dim, kv_v_dim, cache_len);
        let mut buf = DecodeBuffer::new(&self.config, max_head_dim, max_n_kv_heads, max_value_dim);
        let mut rng = if options.seed == 0 {
            let t = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64;
            sampling::Rng::new(t)
        } else {
            sampling::Rng::new(options.seed)
        };

        // Prefill runs the full prompt through the model once to seed the KV
        // cache; incremental decoding starts from the last prompt logits.
        let t_prefill = Instant::now();
        let mut logits = Vec::new();
        let last_prompt_pos = tokens.len() - 1;
        for (pos, &tok_id) in tokens.iter().enumerate() {
            if pos == last_prompt_pos {
                self.forward_token_into(&mut cache, &mut buf, tok_id, pos, &mut logits);
            } else {
                let _ = self.forward_hidden_token(&mut cache, &mut buf, tok_id, pos);
            }
        }
        let prefill_time = t_prefill.elapsed();

        // Decode advances one sampled token at a time while reusing the cache.
        let t_decode = Instant::now();
        let mut output = String::new();
        let mut generated_tokens = 0usize;
        let mut pos = tokens.len();
        let mut recent = recent_token_tail(&tokens);

        // Pre-compute the longest stop sequence length so we only scan a small
        // trailing window of `output` on each token instead of the full string.
        let max_stop_len = options
            .stop_sequences
            .iter()
            .map(|s| s.len())
            .max()
            .unwrap_or(0);

        'decode: for _ in 0..options.max_tokens {
            let token = sampling::sample_with_scratch(
                &mut logits,
                &options.sampler,
                &mut rng,
                recent.as_slice(),
                &mut buf.sampler_candidates,
            );

            if self.is_stop_token(token) {
                break;
            }

            let text = self.tok.decode_token(token);
            output.push_str(&text);

            // Check stop sequences only within a trailing window equal to the
            // longest stop sequence length plus the current token length.  This
            // keeps the scan O(max_stop_len) per token instead of O(output_len).
            if max_stop_len > 0 {
                let window_start = output.len().saturating_sub(max_stop_len + text.len());
                let window = &output[window_start..];
                for stop in &options.stop_sequences {
                    if let Some(rel_idx) = window.find(stop.as_str()) {
                        // Map the relative index back to an absolute offset.
                        output.truncate(window_start + rel_idx);
                        break 'decode;
                    }
                }
            }

            on_token(&text);

            generated_tokens += 1;
            push_recent_token(&mut recent, token);

            if generated_tokens >= options.max_tokens || pos >= cache_len {
                break;
            }

            self.forward_token_into(&mut cache, &mut buf, token, pos, &mut logits);
            pos += 1;
        }

        let decode_time = t_decode.elapsed();
        Ok(GenerationResult {
            text: output,
            stats: GenerationStats {
                prompt_tokens: tokens.len(),
                generated_tokens,
                prefill_time,
                decode_time,
                total_time: total_start.elapsed(),
                cached_tokens: 0,
            },
        })
    }

    /// Dispatches one token through the active model-family forward pass.
    fn forward_token_into(
        &self,
        cache: &mut KVCache,
        buf: &mut DecodeBuffer,
        token: u32,
        pos: usize,
        logits: &mut Vec<f32>,
    ) {
        match &self.weights {
            LoadedWeights::GptOss(weights) => {
                model::forward_gpt_oss_into(&self.config, weights, cache, buf, token, pos, logits)
            }
            LoadedWeights::Gemma4(weights) => {
                model::forward_gemma4_into(&self.config, weights, cache, buf, token, pos, logits)
            }
            LoadedWeights::Standard(weights) => {
                model::forward_into(&self.config, weights, cache, buf, token, pos, logits)
            }
        }
    }

    /// Like `forward_token` but returns the normalized hidden state (dim-sized)
    /// before the output projection.  Used for embedding generation.
    fn forward_hidden_token<'a>(
        &self,
        cache: &mut KVCache,
        buf: &'a mut DecodeBuffer,
        token: u32,
        pos: usize,
    ) -> &'a [f32] {
        match &self.weights {
            LoadedWeights::GptOss(weights) => {
                model::forward_hidden_gpt_oss(&self.config, weights, cache, buf, token, pos)
            }
            LoadedWeights::Gemma4(weights) => {
                model::forward_hidden_gemma4(&self.config, weights, cache, buf, token, pos)
            }
            LoadedWeights::Standard(weights) => {
                model::forward_hidden(&self.config, weights, cache, buf, token, pos)
            }
        }
    }

    /// Embed `text` as a dense vector.
    ///
    /// Runs the full prefill pass, mean-pools the per-token hidden states
    /// across all input positions, then L2-normalises the result.  The
    /// returned vector has dimension `config.dim` and is suitable for cosine
    /// similarity comparisons (RAG retrieval, semantic search, etc.).
    pub fn embed(&self, text: &str) -> Result<EmbeddingResult, String> {
        let _guard = self
            .generation_lock
            .lock()
            .expect("generation lock poisoned");
        let tokens = self.tok.encode(text);
        if tokens.is_empty() {
            return Err(String::from("embed: input tokenised to zero tokens"));
        }

        let cache_len = std::cmp::min(self.config.max_seq_len, tokens.len() + 1);

        let (kv_k_dim, kv_v_dim, max_head_dim, max_n_kv_heads, max_value_dim) = self.kv_dims();

        let mut cache = KVCache::new(self.config.n_layers, kv_k_dim, kv_v_dim, cache_len);
        let mut buf = DecodeBuffer::new(&self.config, max_head_dim, max_n_kv_heads, max_value_dim);

        let dim = self.config.dim;
        let mut sum = vec![0.0f32; dim];
        let token_count = tokens.len();

        for (pos, &tok_id) in tokens.iter().enumerate() {
            let h = self.forward_hidden_token(&mut cache, &mut buf, tok_id, pos);
            for (i, &v) in h.iter().enumerate() {
                sum[i] += v;
            }
        }

        // Mean pool across all token positions, then L2 normalise so cosine
        // similarity is equivalent to a dot product.
        mean_pool_in_place(&mut sum, token_count);
        l2_normalize_in_place(&mut sum);

        Ok(EmbeddingResult {
            embedding: sum,
            token_count,
        })
    }

    /// Checks whether a token is a built-in end-of-generation token.
    fn is_stop_token(&self, token: u32) -> bool {
        if self.arch == "gpt-oss" {
            token == self.tok.eos_id || token == 200002 || token == 200007
        } else {
            token == self.tok.eos_id
        }
    }

    /// Chooses a chat-template renderer and tokenizes the rendered messages.
    fn render_messages(&self, messages: &[ChatMessage], system_prompt: &str) -> Vec<u32> {
        // Prefer architecture-specific chat formatting when the tokenizer
        // metadata exposes one we know how to mirror.
        if self.arch == "gpt-oss" {
            return self.render_gpt_oss_messages(messages, system_prompt);
        }
        if matches!(self.chat_template_kind(), Some("header-chat")) {
            if let Some(tokens) = self.render_header_chat_messages(messages, system_prompt) {
                return tokens;
            }
        }
        self.render_plain_messages(messages, system_prompt)
    }

    /// Renders messages with a simple role-prefixed chat format.
    fn render_plain_messages(&self, messages: &[ChatMessage], system_prompt: &str) -> Vec<u32> {
        let mut prompt = String::new();
        if !system_prompt.trim().is_empty() {
            prompt.push_str("System: ");
            prompt.push_str(system_prompt.trim());
            prompt.push_str("\n\n");
        }

        for message in messages {
            let label = match message.role {
                ChatRole::System => "System",
                ChatRole::User => "User",
                ChatRole::Assistant => "Assistant",
            };
            prompt.push_str(label);
            prompt.push_str(": ");
            prompt.push_str(message.content.trim());
            prompt.push_str("\n\n");
        }
        prompt.push_str("Assistant:");
        self.tok.encode(&prompt)
    }

    /// Renders messages using GPT-OSS harmony-style tokens.
    fn render_gpt_oss_messages(&self, messages: &[ChatMessage], system_prompt: &str) -> Vec<u32> {
        let start = self.tok.special_id("<|start|>").unwrap_or(200006);
        let channel = self.tok.special_id("<|channel|>").unwrap_or(200005);
        let message = self.tok.special_id("<|message|>").unwrap_or(200008);
        let end = self.tok.special_id("<|end|>").unwrap_or(200007);
        let user = self
            .tok
            .special_id("user")
            .unwrap_or_else(|| self.tok.encode_without_bos("user")[0]);
        let assistant = self
            .tok
            .special_id("assistant")
            .unwrap_or_else(|| self.tok.encode_without_bos("assistant")[0]);
        let system = self
            .tok
            .special_id("system")
            .unwrap_or_else(|| self.tok.encode_without_bos("system")[0]);
        let final_tok = self
            .tok
            .special_id("final")
            .unwrap_or_else(|| self.tok.encode_without_bos("final")[0]);

        let mut tokens = Vec::new();

        if !system_prompt.trim().is_empty()
            && !messages.iter().any(|m| matches!(m.role, ChatRole::System))
        {
            tokens.push(start);
            tokens.push(system);
            tokens.push(message);
            tokens.extend(self.tok.encode_without_bos(system_prompt));
            tokens.push(end);
        }

        for item in messages {
            let role_id = match item.role {
                ChatRole::System => system,
                ChatRole::User => user,
                ChatRole::Assistant => assistant,
            };
            tokens.push(start);
            tokens.push(role_id);
            tokens.push(message);
            tokens.extend(self.tok.encode_without_bos(&item.content));
            tokens.push(end);
        }

        // Leave the final assistant message open so generation continues in
        // the assistant/final channel rather than closing the turn.
        tokens.push(start);
        tokens.push(assistant);
        tokens.push(channel);
        tokens.push(final_tok);
        tokens.push(message);
        tokens
    }

    /// Renders messages using header-delimited chat templates.
    fn render_header_chat_messages(
        &self,
        messages: &[ChatMessage],
        system_prompt: &str,
    ) -> Option<Vec<u32>> {
        let bot = self.tok.special_id("<|begin_of_text|>")?;
        let start_header = self.tok.special_id("<|start_header_id|>")?;
        let end_header = self.tok.special_id("<|end_header_id|>")?;
        let eot = self.tok.special_id("<|eot_id|>")?;
        let system = self.tok.special_id("system")?;
        let user = self.tok.special_id("user")?;
        let assistant = self.tok.special_id("assistant")?;

        let mut tokens = Vec::new();
        let push_header = |role_id: u32, out: &mut Vec<u32>| {
            out.push(start_header);
            out.push(role_id);
            out.push(end_header);
            out.extend(self.tok.encode_without_bos("\n\n"));
        };

        tokens.push(bot);
        if !system_prompt.trim().is_empty()
            && !messages.iter().any(|m| matches!(m.role, ChatRole::System))
        {
            push_header(system, &mut tokens);
            tokens.extend(self.tok.encode_without_bos(system_prompt));
            tokens.push(eot);
        }

        for message in messages {
            let role_id = match message.role {
                ChatRole::System => system,
                ChatRole::User => user,
                ChatRole::Assistant => assistant,
            };
            push_header(role_id, &mut tokens);
            tokens.extend(self.tok.encode_without_bos(&message.content));
            tokens.push(eot);
        }

        // Emit only the assistant header so the model predicts the reply body.
        push_header(assistant, &mut tokens);
        Some(tokens)
    }

    /// Classifies the loaded tokenizer chat template.
    fn chat_template_kind(&self) -> Option<&'static str> {
        let template = self
            .gguf
            .metadata
            .get("tokenizer.chat_template")?
            .as_str()?;
        if template.contains("<|start_header_id|>") && template.contains("<|eot_id|>") {
            Some("header-chat")
        } else {
            None
        }
    }

    // ─── Session-based generation ────────────────────────────────────────────

    /// Compute the Longest Common Prefix length between two token sequences.
    #[cfg_attr(target_family = "wasm", allow(dead_code))]
    fn longest_common_prefix(a: &[u32], b: &[u32]) -> usize {
        a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
    }

    /// Compute the dimension parameters needed to allocate a KV cache and
    /// decode buffer for this model.
    fn kv_dims(&self) -> (usize, usize, usize, usize, usize) {
        match &self.weights {
            LoadedWeights::Gemma4(w) => {
                let mut max_hd = self.config.head_dim;
                let mut max_kv_heads = self.config.n_kv_heads;
                let mut max_val = self.config.value_dim;
                for l in w.layers.iter() {
                    if l.head_dim > max_hd {
                        max_hd = l.head_dim;
                    }
                    if l.n_kv_heads > max_kv_heads {
                        max_kv_heads = l.n_kv_heads;
                    }
                    if l.value_dim > max_val {
                        max_val = l.value_dim;
                    }
                }
                (
                    max_kv_heads * max_hd,
                    max_kv_heads * max_val,
                    max_hd,
                    max_kv_heads,
                    max_val,
                )
            }
            _ => (
                self.config.n_kv_heads * self.config.head_dim,
                self.config.kv_dim,
                self.config.head_dim,
                self.config.n_kv_heads,
                self.config.value_dim,
            ),
        }
    }

    /// Create a blank [`crate::session::Session`] pre-allocated for this model.
    ///
    /// `max_cached_tokens` caps the KV-cache length at the given token count
    /// (clamped to `config.max_seq_len`).
    #[cfg(all(not(target_family = "wasm"), feature = "server"))]
    /// Creates a server session with KV cache and decode buffers for this runner.
    pub fn new_session(&self, max_cached_tokens: usize) -> crate::session::Session {
        let cap = max_cached_tokens.min(self.config.max_seq_len).max(1);
        let (kv_k_dim, kv_v_dim, max_head_dim, max_n_kv_heads, max_value_dim) = self.kv_dims();
        let kv_cache = KVCache::new(self.config.n_layers, kv_k_dim, kv_v_dim, cap);
        let decode_buf =
            DecodeBuffer::new(&self.config, max_head_dim, max_n_kv_heads, max_value_dim);
        crate::session::Session::new(kv_cache, decode_buf)
    }

    /// Multi-turn generation that reuses a persistent [`crate::session::Session`].
    ///
    /// On each call the full chat prompt is rendered and tokenised.  A
    /// Longest-Common-Prefix comparison against the session's cached token
    /// sequence determines how much of the KV cache can be reused:
    ///
    /// * Only the differing suffix is prefilled.
    /// * Each generated token is forwarded into the same KV cache before the
    ///   next sampling step.
    /// * Logits and cached tokens are persisted in the session for the next
    ///   turn.
    ///
    /// Falls back to a full prefill when the new prompt exceeds the session's
    /// KV-cache capacity.
    #[cfg(all(not(target_family = "wasm"), feature = "server"))]
    /// Generates chat text while reusing a session KV cache when possible.
    pub fn generate_chat_with_session<F>(
        &self,
        messages: &[ChatMessage],
        options: &GenerationOptions,
        session: &mut crate::session::Session,
        mut on_token: F,
    ) -> Result<GenerationResult, String>
    where
        F: FnMut(&str),
    {
        let _guard = self
            .generation_lock
            .lock()
            .expect("generation lock poisoned");
        options.validate()?;

        if messages.is_empty() {
            return Err(String::from("No prompt provided."));
        }

        let total_start = Instant::now();
        session.last_used = total_start;

        let tokens = self.render_messages(messages, &options.system_prompt);
        if tokens.is_empty() {
            return Err(String::from("Prompt rendered to zero tokens."));
        }

        let cache_limit = session.kv_cache.max_len;

        // If the prompt alone already exceeds the KV-cache capacity, reset the
        // session so we fall back to a clean full prefill.
        if tokens.len() >= cache_limit {
            session.reset();
        }

        let prefix_len = Self::longest_common_prefix(&tokens, &session.cached_tokens);
        let reused = prefix_len;
        let new_prompt_tokens = tokens.len() - prefix_len;
        session.cached_tokens_served += reused;
        session.evaluated_tokens += new_prompt_tokens;

        // ── Prefill ──────────────────────────────────────────────────────────
        let t_prefill = Instant::now();
        let mut logits = Vec::new();
        let last_prompt_pos = tokens.len() - 1;

        if prefix_len == tokens.len() {
            // All prompt tokens are already in the cache; reuse last logits.
            if !session.last_logits.is_empty() {
                logits = std::mem::take(&mut session.last_logits);
            } else {
                // No saved logits (e.g. session was just created or reset).
                // Re-forward the last prompt token to obtain them.
                self.forward_token_into(
                    &mut session.kv_cache,
                    &mut session.decode_buf,
                    tokens[last_prompt_pos],
                    last_prompt_pos,
                    &mut logits,
                );
            }
        } else {
            // Prefill only the suffix that differs from the cached prefix.
            for (idx, &tok_id) in tokens[prefix_len..].iter().enumerate() {
                let pos = prefix_len + idx;
                if pos == last_prompt_pos {
                    self.forward_token_into(
                        &mut session.kv_cache,
                        &mut session.decode_buf,
                        tok_id,
                        pos,
                        &mut logits,
                    );
                } else {
                    let _ = self.forward_hidden_token(
                        &mut session.kv_cache,
                        &mut session.decode_buf,
                        tok_id,
                        pos,
                    );
                }
            }
        }
        let prefill_time = t_prefill.elapsed();

        // Capture values needed for the decode phase before transferring
        // ownership of `tokens` into the session.
        let prompt_len = tokens.len();
        // Update the session's token list to the current prompt.
        // Generated tokens are appended below during the decode loop.
        session.cached_tokens = tokens;

        // ── Decode ───────────────────────────────────────────────────────────
        let t_decode = Instant::now();
        let mut output = String::new();
        let mut generated_tokens = 0usize;
        let mut pos = prompt_len;

        // Initialise the repeat-penalty window from only the prompt tail.  This
        // keeps sampling work bounded for long prompts and matches the window
        // used after generation starts.
        let mut recent = recent_token_tail(&session.cached_tokens);

        let max_stop_len = options
            .stop_sequences
            .iter()
            .map(|s| s.len())
            .max()
            .unwrap_or(0);

        let mut rng = if options.seed == 0 {
            let t = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64;
            sampling::Rng::new(t)
        } else {
            sampling::Rng::new(options.seed)
        };

        'decode: for _ in 0..options.max_tokens {
            let token = sampling::sample_with_scratch(
                &mut logits,
                &options.sampler,
                &mut rng,
                recent.as_slice(),
                &mut session.decode_buf.sampler_candidates,
            );

            if self.is_stop_token(token) {
                break;
            }

            let text = self.tok.decode_token(token);
            output.push_str(&text);

            if max_stop_len > 0 {
                let window_start = output.len().saturating_sub(max_stop_len + text.len());
                let window = &output[window_start..];
                for stop in &options.stop_sequences {
                    if let Some(rel_idx) = window.find(stop.as_str()) {
                        output.truncate(window_start + rel_idx);
                        break 'decode;
                    }
                }
            }

            on_token(&text);

            generated_tokens += 1;
            session.cached_tokens.push(token);
            push_recent_token(&mut recent, token);

            if pos >= cache_limit - 1 {
                break;
            }

            self.forward_token_into(
                &mut session.kv_cache,
                &mut session.decode_buf,
                token,
                pos,
                &mut logits,
            );
            pos += 1;
        }

        // Save logits as the starting point for the next turn's sampling.
        session.last_logits = logits;
        session.recent = recent;
        session.evaluated_tokens += generated_tokens;

        let decode_time = t_decode.elapsed();
        Ok(GenerationResult {
            text: output,
            stats: GenerationStats {
                prompt_tokens: prompt_len,
                generated_tokens,
                prefill_time,
                decode_time,
                total_time: total_start.elapsed(),
                cached_tokens: reused,
            },
        })
    }
}

/// Measures one matrix-vector kernel and returns timing statistics.
fn measure_matvec(
    name: &str,
    weight: &Weight,
    x: &[f32],
    runs: usize,
    out: &mut Vec<f32>,
) -> KernelBenchRow {
    let (rows, cols) = weight_shape(weight, x.len());
    measure_kernel(name, weight, rows, cols, runs, || {
        weight.matvec_into(x, out);
        std::hint::black_box(out.len());
    })
}

/// Runs a benchmark closure repeatedly and summarizes timing.
fn measure_kernel<F>(
    name: &str,
    weight: &Weight,
    rows: usize,
    cols: usize,
    runs: usize,
    mut body: F,
) -> KernelBenchRow
where
    F: FnMut(),
{
    body();
    let start = Instant::now();
    for _ in 0..runs {
        body();
    }
    let total_ms = start.elapsed().as_secs_f64() * 1000.0;
    KernelBenchRow {
        name: name.to_string(),
        dtype: weight_dtype(weight),
        rows,
        cols,
        runs,
        avg_ms: total_ms / runs as f64,
        total_ms,
    }
}

/// Infers a matrix shape for benchmark display.
fn weight_shape(weight: &Weight, input_cols: usize) -> (usize, usize) {
    match weight {
        Weight::F32(data) => {
            let cols = input_cols.max(1);
            (data.len() / cols, input_cols)
        }
        Weight::Quantized { rows, cols, .. } => (*rows, *cols),
    }
}

/// Returns a display string for a weight data type.
fn weight_dtype(weight: &Weight) -> String {
    match weight {
        Weight::F32(_) => String::from("F32"),
        Weight::Quantized { dtype, .. } => format!("{:?}", dtype),
    }
}

/// Builds deterministic input data for repeatable kernel benchmarks.
fn deterministic_bench_vector(n: usize) -> Vec<f32> {
    let mut out = vec![0.0; n];
    for (i, value) in out.iter_mut().enumerate() {
        *value = ((i % 251) as f32 - 125.0) / 125.0;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{
        RECENT_TOKEN_LIMIT, cosine_similarity, l2_normalize_in_place, mean_pool_in_place,
        push_recent_token, recent_token_tail,
    };

    #[test]
    /// Verifies that mean pooling divides accumulated token vectors by sample count.
    fn mean_pool_scales_by_sample_count() {
        let mut v = vec![6.0f32, -3.0, 9.0];
        mean_pool_in_place(&mut v, 3);
        assert_eq!(v, vec![2.0, -1.0, 3.0]);
    }

    #[test]
    /// Verifies that L2 normalization produces a unit-length vector.
    fn l2_normalize_produces_unit_vector() {
        let mut v = vec![3.0f32, 4.0];
        l2_normalize_in_place(&mut v);
        let norm = (v[0] * v[0] + v[1] * v[1]).sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
        assert!((v[0] - 0.6).abs() < 1e-6);
        assert!((v[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    /// Verifies that zero vectors remain stable during L2 normalization.
    fn l2_normalize_keeps_zero_vector_stable() {
        let mut v = vec![0.0f32, 0.0, 0.0];
        l2_normalize_in_place(&mut v);
        assert_eq!(v, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    /// Verifies cosine similarity for identical and orthogonal vectors.
    fn cosine_similarity_for_identical_and_orthogonal_vectors() {
        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![1.0f32, 2.0, 3.0];
        let c = vec![1.0f32, -2.0, 1.0];
        let sim_ab = cosine_similarity(&a, &b).unwrap();
        let sim_ac = cosine_similarity(&a, &c).unwrap();
        assert!((sim_ab - 1.0).abs() < 1e-6);
        assert!(sim_ac.abs() < 1e-6);
    }

    #[test]
    /// Verifies cosine similarity input validation for bad lengths and values.
    fn cosine_similarity_rejects_invalid_inputs() {
        assert!(cosine_similarity(&[], &[]).is_err());
        assert!(cosine_similarity(&[1.0], &[1.0, 2.0]).is_err());
        assert!(cosine_similarity(&[0.0, 0.0], &[1.0, 2.0]).is_err());
    }

    #[test]
    /// Verifies that identical token sequences return their full length as prefix.
    fn longest_common_prefix_identical_sequences() {
        use super::Runner;
        let a = vec![1u32, 2, 3, 4, 5];
        assert_eq!(Runner::longest_common_prefix(&a, &a), 5);
    }

    #[test]
    /// Verifies prefix length when token sequences diverge after shared tokens.
    fn longest_common_prefix_partial_match() {
        use super::Runner;
        let a = vec![1u32, 2, 3, 4, 5];
        let b = vec![1u32, 2, 3, 7, 8, 9];
        assert_eq!(Runner::longest_common_prefix(&a, &b), 3);
    }

    #[test]
    /// Verifies that unrelated token sequences have no shared prefix.
    fn longest_common_prefix_no_match() {
        use super::Runner;
        let a = vec![1u32, 2, 3];
        let b = vec![9u32, 8, 7];
        assert_eq!(Runner::longest_common_prefix(&a, &b), 0);
    }

    #[test]
    /// Verifies prefix handling when one or both token sequences are empty.
    fn longest_common_prefix_empty_inputs() {
        use super::Runner;
        assert_eq!(Runner::longest_common_prefix(&[], &[1u32, 2]), 0);
        assert_eq!(Runner::longest_common_prefix(&[1u32, 2], &[]), 0);
    }

    #[test]
    /// Verifies that the repeat-penalty window starts with only the prompt tail.
    fn recent_token_tail_keeps_only_bounded_suffix() {
        let tokens: Vec<u32> = (0..100).collect();
        let recent = recent_token_tail(&tokens);
        assert_eq!(recent.len(), RECENT_TOKEN_LIMIT);
        assert_eq!(recent[0], 36);
        assert_eq!(recent[RECENT_TOKEN_LIMIT - 1], 99);
    }

    #[test]
    /// Verifies that appending recent tokens preserves the fixed sampling window.
    fn push_recent_token_preserves_bounded_window() {
        let mut recent: Vec<u32> = (0..RECENT_TOKEN_LIMIT as u32).collect();
        push_recent_token(&mut recent, 999);
        assert_eq!(recent.len(), RECENT_TOKEN_LIMIT);
        assert_eq!(recent[0], 1);
        assert_eq!(recent[RECENT_TOKEN_LIMIT - 1], 999);
    }
}

#[cfg(target_family = "wasm")]
use wasm_bindgen::prelude::*;

#[cfg(target_family = "wasm")]
#[wasm_bindgen]
pub struct WasmRunner {
    inner: Runner,
}

#[cfg(target_family = "wasm")]
#[wasm_bindgen]
impl WasmRunner {
    #[wasm_bindgen(constructor)]
    /// Creates a WASM runner from GGUF bytes supplied by JavaScript.
    pub fn new(model_bytes: &[u8]) -> Result<WasmRunner, JsValue> {
        let inner = Runner::from_gguf_bytes(model_bytes).map_err(|err| JsValue::from_str(&err))?;
        Ok(Self { inner })
    }

    #[wasm_bindgen(getter)]
    /// Returns the optional model name from GGUF metadata.
    pub fn model_name(&self) -> String {
        self.inner.model_name().unwrap_or("unknown").to_string()
    }

    /// Generates a complete response for a plain prompt.
    pub fn generate(&self, prompt: &str, max_tokens: usize, temp: f32) -> Result<String, JsValue> {
        let mut options = GenerationOptions::default();
        options.max_tokens = max_tokens.max(1);
        options.sampler.temperature = temp;
        self.inner
            .generate(prompt, &options)
            .map(|result| result.text)
            .map_err(|err| JsValue::from_str(&err))
    }

    /// Returns an L2-normalised embedding vector as a `Float32Array`.
    /// Suitable for cosine similarity / RAG retrieval directly in JS.
    pub fn embed(&self, text: &str) -> Result<js_sys::Float32Array, JsValue> {
        self.inner
            .embed(text)
            .map(|r| js_sys::Float32Array::from(r.embedding.as_slice()))
            .map_err(|err| JsValue::from_str(&err))
    }

    /// Cosine similarity between two `Float32Array` embeddings produced by `embed()`.
    /// Returns a value in [-1, 1]; higher means more similar.
    pub fn cosine_similarity(
        a: &js_sys::Float32Array,
        b: &js_sys::Float32Array,
    ) -> Result<f32, JsValue> {
        let a_vec: Vec<f32> = a.to_vec();
        let b_vec: Vec<f32> = b.to_vec();
        crate::runtime::cosine_similarity(&a_vec, &b_vec).map_err(|err| JsValue::from_str(&err))
    }
}
