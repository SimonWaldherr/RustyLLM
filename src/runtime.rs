use crate::gguf::GGUFFile;
use crate::model::{self, Config, DecodeBuffer, GptOssWeights, KVCache, ModelWeights};
use crate::sampling::{self, SamplerConfig};
use crate::tokenizer::Tokenizer;
use std::collections::VecDeque;
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
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::User,
            content: content.into(),
        }
    }

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

#[inline]
fn mean_pool_in_place(values: &mut [f32], sample_count: usize) {
    if sample_count == 0 {
        return;
    }
    let scale = 1.0 / sample_count as f32;
    for value in values.iter_mut() {
        *value *= scale;
    }
}

#[inline]
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
    #[allow(dead_code)]
    #[cfg(not(target_family = "wasm"))]
    mapped_model: Option<crate::mmap::MmapFile>,
}

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

impl Runner {
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
            #[cfg(not(target_family = "wasm"))]
            mapped_model: None,
        })
    }

    #[cfg(not(target_family = "wasm"))]
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

    pub fn architecture(&self) -> &str {
        &self.arch
    }

    pub fn model_name(&self) -> Option<&str> {
        self.gguf.get_str("general.name")
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tok
    }

    pub fn gguf(&self) -> &GGUFFile {
        &self.gguf
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    pub fn generate(
        &self,
        prompt: &str,
        options: &GenerationOptions,
    ) -> Result<GenerationResult, String> {
        let messages = [ChatMessage::user(prompt)];
        self.generate_chat(&messages, options)
    }

    pub fn generate_chat(
        &self,
        messages: &[ChatMessage],
        options: &GenerationOptions,
    ) -> Result<GenerationResult, String> {
        self.generate_chat_stream(messages, options, |_| {})
    }

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

    pub fn generate_chat_stream<F>(
        &self,
        messages: &[ChatMessage],
        options: &GenerationOptions,
        mut on_token: F,
    ) -> Result<GenerationResult, String>
    where
        F: FnMut(&str),
    {
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
        let (kv_k_dim, kv_v_dim, max_head_dim, max_n_kv_heads, max_value_dim) = match &self.weights
        {
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
                let kk = max_kv_heads * max_hd;
                let vv = max_kv_heads * max_val;
                (kk, vv, max_hd, max_kv_heads, max_val)
            }
            _ => (
                self.config.n_kv_heads * self.config.head_dim,
                self.config.kv_dim,
                self.config.head_dim,
                self.config.n_kv_heads,
                self.config.value_dim,
            ),
        };

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
                logits = self.forward_token(&mut cache, &mut buf, tok_id, pos);
            } else {
                let _ = self.forward_hidden_token(&mut cache, &mut buf, tok_id, pos);
            }
        }
        let prefill_time = t_prefill.elapsed();

        // Decode advances one sampled token at a time while reusing the cache.
        let t_decode = Instant::now();
        let mut output = String::new();
        let mut generated = Vec::new();
        let mut pos = tokens.len();
        let mut recent: VecDeque<u32> = tokens.iter().copied().collect();

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
                recent.make_contiguous(),
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

            generated.push(token);
            recent.push_back(token);
            if recent.len() > 64 {
                recent.pop_front();
            }

            if generated.len() >= options.max_tokens || pos >= cache_len {
                break;
            }

            logits = self.forward_token(&mut cache, &mut buf, token, pos);
            pos += 1;
        }

        let decode_time = t_decode.elapsed();
        Ok(GenerationResult {
            text: output,
            stats: GenerationStats {
                prompt_tokens: tokens.len(),
                generated_tokens: generated.len(),
                prefill_time,
                decode_time,
                total_time: total_start.elapsed(),
            },
        })
    }

    fn forward_token(
        &self,
        cache: &mut KVCache,
        buf: &mut DecodeBuffer,
        token: u32,
        pos: usize,
    ) -> Vec<f32> {
        match &self.weights {
            LoadedWeights::GptOss(weights) => {
                model::forward_gpt_oss(&self.config, weights, cache, buf, token, pos)
            }
            LoadedWeights::Gemma4(weights) => {
                model::forward_gemma4(&self.config, weights, cache, buf, token, pos)
            }
            LoadedWeights::Standard(weights) => {
                model::forward(&self.config, weights, cache, buf, token, pos)
            }
        }
    }

    /// Like `forward_token` but returns the normalized hidden state (dim-sized)
    /// before the output projection.  Used for embedding generation.
    fn forward_hidden_token(
        &self,
        cache: &mut KVCache,
        buf: &mut DecodeBuffer,
        token: u32,
        pos: usize,
    ) -> Vec<f32> {
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
        let tokens = self.tok.encode(text);
        if tokens.is_empty() {
            return Err(String::from("embed: input tokenised to zero tokens"));
        }

        let cache_len = std::cmp::min(self.config.max_seq_len, tokens.len() + 1);

        let (kv_k_dim, kv_v_dim, max_head_dim, max_n_kv_heads, max_value_dim) = match &self.weights
        {
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
                let kk = max_kv_heads * max_hd;
                let vv = max_kv_heads * max_val;
                (kk, vv, max_hd, max_kv_heads, max_val)
            }
            _ => (
                self.config.n_kv_heads * self.config.head_dim,
                self.config.kv_dim,
                self.config.head_dim,
                self.config.n_kv_heads,
                self.config.value_dim,
            ),
        };

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

    fn is_stop_token(&self, token: u32) -> bool {
        if self.arch == "gpt-oss" {
            token == self.tok.eos_id || token == 200002 || token == 200007
        } else {
            token == self.tok.eos_id
        }
    }

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
}

#[cfg(test)]
mod tests {
    use super::{cosine_similarity, l2_normalize_in_place, mean_pool_in_place};

    #[test]
    fn mean_pool_scales_by_sample_count() {
        let mut v = vec![6.0f32, -3.0, 9.0];
        mean_pool_in_place(&mut v, 3);
        assert_eq!(v, vec![2.0, -1.0, 3.0]);
    }

    #[test]
    fn l2_normalize_produces_unit_vector() {
        let mut v = vec![3.0f32, 4.0];
        l2_normalize_in_place(&mut v);
        let norm = (v[0] * v[0] + v[1] * v[1]).sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
        assert!((v[0] - 0.6).abs() < 1e-6);
        assert!((v[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn l2_normalize_keeps_zero_vector_stable() {
        let mut v = vec![0.0f32, 0.0, 0.0];
        l2_normalize_in_place(&mut v);
        assert_eq!(v, vec![0.0, 0.0, 0.0]);
    }

    #[test]
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
    fn cosine_similarity_rejects_invalid_inputs() {
        assert!(cosine_similarity(&[], &[]).is_err());
        assert!(cosine_similarity(&[1.0], &[1.0, 2.0]).is_err());
        assert!(cosine_similarity(&[0.0, 0.0], &[1.0, 2.0]).is_err());
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
    pub fn new(model_bytes: &[u8]) -> Result<WasmRunner, JsValue> {
        let inner = Runner::from_gguf_bytes(model_bytes).map_err(|err| JsValue::from_str(&err))?;
        Ok(Self { inner })
    }

    #[wasm_bindgen(getter)]
    pub fn model_name(&self) -> String {
        self.inner.model_name().unwrap_or("unknown").to_string()
    }

    pub fn generate(&self, prompt: &str, max_tokens: usize, temp: f32) -> Result<String, JsValue> {
        let mut options = GenerationOptions::default();
        options.max_tokens = max_tokens.max(1);
        options.sampler.temperature = temp;
        self.inner
            .generate(prompt, &options)
            .map(|result| result.text)
            .map_err(|err| JsValue::from_str(&err))
    }
}
