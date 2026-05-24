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
    pub speculative: SpeculativeConfig,
    pub runtime: RuntimeOptConfig,
    /// Stop generation when any of these strings appears in the output.
    /// The matched sequence is not included in the returned text.
    pub stop_sequences: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct SpeculativeConfig {
    pub enabled: bool,
    pub assistant_path: Option<String>,
    pub max_draft_tokens: usize,
    pub adaptive: bool,
    pub min_accept_rate: f32,
}

impl Default for SpeculativeConfig {
    /// Uses conservative speculative decoding defaults.
    fn default() -> Self {
        Self {
            enabled: true,
            assistant_path: None,
            max_draft_tokens: 4,
            adaptive: true,
            min_accept_rate: 0.5,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvCacheDType {
    Auto,
    F32,
    Bf16,
    Q8,
}

impl KvCacheDType {
    /// Parses a CLI/API KV-cache dtype name.
    pub fn parse(value: &str) -> Option<Self> {
        match value.to_ascii_lowercase().as_str() {
            "auto" => Some(Self::Auto),
            "f32" | "float32" => Some(Self::F32),
            "bf16" | "bfloat16" => Some(Self::Bf16),
            "q8" | "q8_0" | "q8-kv" => Some(Self::Q8),
            _ => None,
        }
    }

    /// Returns the stable display name used in boot logs.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::F32 => "f32",
            Self::Bf16 => "bf16",
            Self::Q8 => "q8",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RuntimeProfile {
    Auto,
    Mistral,
    Gemma,
}

impl RuntimeProfile {
    /// Parses a runtime optimization profile name.
    pub fn parse(value: &str) -> Option<Self> {
        match value.to_ascii_lowercase().as_str() {
            "auto" => Some(Self::Auto),
            "mistral" | "ministral" => Some(Self::Mistral),
            "gemma" | "gemma4" | "gemma4-assistant" => Some(Self::Gemma),
            _ => None,
        }
    }

    /// Returns the stable display name used in boot logs.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Mistral => "mistral",
            Self::Gemma => "gemma",
        }
    }
}

#[derive(Clone, Debug)]
pub struct RuntimeOptConfig {
    pub kv_cache_dtype: KvCacheDType,
    pub flash_attention: bool,
    pub sliding_window_size: Option<usize>,
    pub max_context: Option<usize>,
    pub profile: RuntimeProfile,
}

impl Default for RuntimeOptConfig {
    /// Uses runtime options that preserve existing model behavior.
    fn default() -> Self {
        Self {
            kv_cache_dtype: KvCacheDType::Auto,
            flash_attention: true,
            sliding_window_size: None,
            max_context: None,
            profile: RuntimeProfile::Auto,
        }
    }
}

impl Default for GenerationOptions {
    /// provides conservative sampling defaults.
    fn default() -> Self {
        Self {
            max_tokens: 256,
            sampler: SamplerConfig::default(),
            seed: 0,
            system_prompt: String::from("You are a helpful assistant."),
            speculative: SpeculativeConfig::default(),
            runtime: RuntimeOptConfig::default(),
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
        if self.speculative.max_draft_tokens == 0 && self.speculative.assistant_path.is_some() {
            return Err(String::from("--mtp-tokens must be greater than 0."));
        }
        if !self.speculative.min_accept_rate.is_finite()
            || self.speculative.min_accept_rate < 0.0
            || self.speculative.min_accept_rate > 1.0
        {
            return Err(String::from(
                "speculative min_accept_rate must be in the range [0, 1].",
            ));
        }
        if matches!(self.runtime.max_context, Some(0)) {
            return Err(String::from("--max-context must be greater than 0."));
        }
        if matches!(self.runtime.sliding_window_size, Some(0)) {
            return Err(String::from("--sliding-window must be greater than 0."));
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
    pub speculative: Option<SpeculativeStats>,
}

#[derive(Clone, Debug, Default)]
pub struct SpeculativeStats {
    pub drafted_tokens: usize,
    pub accepted_tokens: usize,
    pub rejected_tokens: usize,
    pub draft_time: Duration,
    pub disabled: bool,
}

impl SpeculativeStats {
    /// Returns accepted / drafted tokens.
    pub fn accept_rate(&self) -> f32 {
        if self.drafted_tokens == 0 {
            0.0
        } else {
            self.accepted_tokens as f32 / self.drafted_tokens as f32
        }
    }

    /// Returns assistant draft throughput.
    pub fn draft_tok_s(&self) -> f32 {
        self.drafted_tokens as f32 / self.draft_time.as_secs_f32().max(0.001)
    }
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

/// Returns the highest finite-logit token ID, falling back to 0.
fn argmax_finite_token(logits: &[f32]) -> u32 {
    let mut best_idx = 0usize;
    let mut best = f32::NEG_INFINITY;
    for (idx, &value) in logits.iter().enumerate() {
        if value.is_finite() && value > best {
            best = value;
            best_idx = idx;
        }
    }
    best_idx as u32
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

struct SpeculativeState<'a> {
    assistant: &'a Runner,
    cache: KVCache,
    buf: DecodeBuffer,
    logits: Vec<f32>,
    draft_limit: usize,
    max_draft_tokens: usize,
    adaptive: bool,
    min_accept_rate: f32,
    stats: SpeculativeStats,
    enabled: bool,
}

enum DecodeFlow {
    Continue,
    Stop,
    Fallback,
}

pub struct Runner {
    gguf: GGUFFile,
    arch: String,
    tok: Tokenizer,
    config: Config,
    weights: LoadedWeights,
    speculative_assistant: Option<Box<Runner>>,
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
            | "gemma4-assistant"
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

#[derive(Clone, Debug)]
pub struct CompatibilityReport {
    pub supported_architecture: bool,
    pub unsupported_tensor_types: Vec<String>,
    pub missing_tensors: Vec<String>,
    pub unsupported_layouts: Vec<String>,
}

impl CompatibilityReport {
    /// Returns true when the model can be loaded by the current runtime.
    pub fn is_supported(&self) -> bool {
        self.supported_architecture
            && self.unsupported_tensor_types.is_empty()
            && self.missing_tensors.is_empty()
            && self.unsupported_layouts.is_empty()
    }

    /// Returns a compact status string for CLI/catalog reporting.
    pub fn status(&self) -> &'static str {
        if self.is_supported() {
            "supported"
        } else if self.supported_architecture
            && (!self.unsupported_tensor_types.is_empty()
                || !self.missing_tensors.is_empty()
                || !self.unsupported_layouts.is_empty())
        {
            "partially-supported"
        } else {
            "unsupported"
        }
    }

    fn first_error(&self, arch: &str) -> String {
        if !self.supported_architecture {
            return format!(
                "Unsupported architecture: {}. Please ensure you are using a supported GGUF architecture.",
                arch
            );
        }
        if let Some(name) = self.unsupported_tensor_types.first() {
            return format!(
                "Tensor '{}' uses an unsupported quantization type. Please re-quantize the model using F16, Q8_0/Q8_1, Q4_0/Q4_1, Q5_0/Q5_1, Q4_K, Q5_K, Q6_K, or MXFP4.",
                name
            );
        }
        if let Some(name) = self.missing_tensors.first() {
            return format!("Model layout is not supported: missing tensor '{}'.", name);
        }
        if let Some(reason) = self.unsupported_layouts.first() {
            return format!("Model layout is not supported: {}.", reason);
        }
        String::from("Model is not supported by this runtime.")
    }
}

/// Builds a compatibility report for the parsed GGUF metadata and tensor layout.
pub fn compatibility_report(gguf: &GGUFFile) -> CompatibilityReport {
    let arch = gguf.get_str("general.architecture").unwrap_or("unknown");
    let mut report = CompatibilityReport {
        supported_architecture: architecture_supported(arch),
        unsupported_tensor_types: Vec::new(),
        missing_tensors: Vec::new(),
        unsupported_layouts: Vec::new(),
    };

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
            | GGMLType::Q5_K
            | GGMLType::Q6_K
            | GGMLType::MXFP4 => {}
            _ => report.unsupported_tensor_types.push(tensor.name.clone()),
        }
    }

    if report.supported_architecture {
        validate_tensor_layout(gguf, arch, &mut report);
    }

    report
}

fn validate_tensor_layout(gguf: &GGUFFile, arch: &str, report: &mut CompatibilityReport) {
    let has = |name: &str| gguf.tensors.iter().any(|tensor| tensor.name == name);
    let config = Config::from_gguf(gguf);

    match arch {
        "gpt-oss" | "gemma" | "gemma2" | "gemma4" | "gemma4-assistant" => return,
        "deepseek2" | "deepseek-v2" => {
            if has("blk.0.attn_kv_a_mqa.weight") || has("blk.0.attn_kv_b.weight") {
                report.unsupported_layouts.push(String::from(
                    "DeepSeek2 MLA attention tensors are present, but the runtime does not yet implement MLA",
                ));
                return;
            }
        }
        _ => {}
    }

    if config.dim == 0 || config.n_layers == 0 || config.n_heads == 0 || config.n_kv_heads == 0 {
        report
            .unsupported_layouts
            .push(String::from("missing required transformer metadata"));
        return;
    }

    for name in ["token_embd.weight", "output_norm.weight"] {
        if !has(name) {
            report.missing_tensors.push(name.to_string());
        }
    }

    let q_rows = config.n_heads * config.head_dim;
    let k_rows = config.n_kv_heads * config.head_dim;
    let v_rows = config.n_kv_heads * config.value_dim;

    for layer in 0..config.n_layers {
        let prefix = format!("blk.{}.", layer);
        for suffix in [
            "attn_norm.weight",
            "attn_output.weight",
            "ffn_norm.weight",
            "ffn_down.weight",
        ] {
            let name = format!("{}{}", prefix, suffix);
            if !has(&name) {
                report.missing_tensors.push(name);
            }
        }

        let q = format!("{}attn_q.weight", prefix);
        let k = format!("{}attn_k.weight", prefix);
        let v = format!("{}attn_v.weight", prefix);
        let qkv = format!("{}attn_qkv.weight", prefix);
        if !(has(&qkv) || has(&q) && has(&k) && has(&v)) {
            report.missing_tensors.push(format!(
                "{}attn_q/attn_k/attn_v.weight or attn_qkv.weight",
                prefix
            ));
        } else if has(&qkv) {
            validate_row_count(gguf, &qkv, q_rows + k_rows + v_rows, report);
        }

        let gate = format!("{}ffn_gate.weight", prefix);
        let up = format!("{}ffn_up.weight", prefix);
        if !has(&gate) {
            if !has(&up) {
                report.missing_tensors.push(gate);
            } else {
                validate_row_count(gguf, &up, config.hidden_dim * 2, report);
            }
        }
    }
}

fn validate_row_count(
    gguf: &GGUFFile,
    name: &str,
    required_rows: usize,
    report: &mut CompatibilityReport,
) {
    if let Some(tensor) = gguf.tensors.iter().find(|tensor| tensor.name == name) {
        let rows = tensor.dims.get(1).copied().unwrap_or(0) as usize;
        if rows < required_rows {
            report.unsupported_layouts.push(format!(
                "{} has {} rows, expected at least {}",
                name, rows, required_rows
            ));
        }
    }
}

/// Checks that every tensor in `gguf` uses a quantization type that the
/// inference kernels support.  Returns a descriptive error for any tensor
/// whose dtype would cause a panic inside `load_weight`.
#[allow(dead_code)]
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
            | GGMLType::Q5_K
            | GGMLType::Q6_K
            | GGMLType::MXFP4 => {}
            unsupported => {
                return Err(format!(
                    "Tensor '{}' uses unsupported quantization type {:?}. \
	                     Please re-quantize the model using a supported format: \
	                     F16, Q8_0, Q8_1, Q4_0, Q4_1, Q5_0, Q5_1, Q4_K, Q5_K, Q6_K, or MXFP4.",
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

        let compatibility = compatibility_report(&gguf);
        if !compatibility.is_supported() {
            return Err(compatibility.first_error(&arch));
        }

        let tok = Tokenizer::from_metadata(&gguf.metadata);
        let (config, weights) = match arch.as_str() {
            "gpt-oss" => {
                let (config, weights) = model::load_gpt_oss_model(data, &gguf, false);
                (config, LoadedWeights::GptOss(weights))
            }
            "gemma" | "gemma2" | "gemma4" | "gemma4-assistant" => {
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
            speculative_assistant: None,
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

        let compatibility = compatibility_report(&gguf);
        if !compatibility.is_supported() {
            return Err(compatibility.first_error(&arch));
        }

        let tok = Tokenizer::from_metadata(&gguf.metadata);
        let (config, weights) = match arch.as_str() {
            "gpt-oss" => {
                let (config, weights) = model::load_gpt_oss_model(mmap.as_slice(), &gguf, true);
                (config, LoadedWeights::GptOss(weights))
            }
            "gemma" | "gemma2" | "gemma4" | "gemma4-assistant" => {
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
            speculative_assistant: None,
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

    /// Attaches a verified assistant checkpoint for greedy speculative decoding.
    pub fn attach_speculative_assistant(&mut self, mut assistant: Runner) -> Result<(), String> {
        assistant.speculative_assistant = None;
        self.verify_speculative_assistant(&assistant)?;
        self.speculative_assistant = Some(Box::new(assistant));
        Ok(())
    }

    /// Returns whether a speculative assistant checkpoint is available.
    pub fn has_speculative_assistant(&self) -> bool {
        self.speculative_assistant.is_some()
    }

    /// Describes runtime optimizations that can affect generation behavior.
    pub fn optimization_summary(&self, options: &GenerationOptions) -> Vec<String> {
        let mut items = Vec::new();
        items.push(format!("profile={}", options.runtime.profile.as_str()));
        items.push(match options.runtime.kv_cache_dtype {
            KvCacheDType::Auto => String::from("kv-cache=f32(auto)"),
            KvCacheDType::F32 => String::from("kv-cache=f32"),
            requested => format!(
                "kv-cache=f32(requested {} pending-bench)",
                requested.as_str()
            ),
        });
        if options.runtime.flash_attention {
            items.push(String::from("flash-attn=online-softmax"));
        } else {
            items.push(String::from("flash-attn=off"));
        }
        if let Some(window) = self.effective_sliding_window(options) {
            items.push(format!("sliding-window={} tokens", window));
        }
        items.push(format!(
            "max-context={} tokens",
            self.effective_max_context(options)
        ));
        if options.speculative.enabled && self.speculative_assistant.is_some() {
            items.push(format!(
                "mtp=greedy assistant={} draft_tokens={} adaptive={} min_accept_rate={:.2}",
                options
                    .speculative
                    .assistant_path
                    .as_deref()
                    .unwrap_or("attached"),
                options.speculative.max_draft_tokens,
                options.speculative.adaptive,
                options.speculative.min_accept_rate
            ));
        } else {
            items.push(String::from("mtp=off"));
        }
        items
    }

    /// Emits conservative runtime warnings for known long-context pitfalls.
    pub fn optimization_warnings(&self, options: &GenerationOptions) -> Vec<String> {
        let mut warnings = Vec::new();
        let profile = self.effective_profile(options);
        if profile == RuntimeProfile::Mistral && self.config.max_seq_len >= 131_072 {
            warnings.push(format!(
                "Ministral context metadata advertises {} tokens; default runtime cap is {} unless --max-context overrides it.",
                self.config.max_seq_len,
                self.effective_max_context(options)
            ));
        }
        if options.speculative.enabled
            && self.speculative_assistant.is_some()
            && options.sampler.temperature != 0.0
        {
            warnings.push(String::from(
                "MTP assistant is loaded, but MTP is greedy-only and will stay disabled unless --temp 0 is used.",
            ));
        }
        if matches!(
            options.runtime.kv_cache_dtype,
            KvCacheDType::Bf16 | KvCacheDType::Q8
        ) {
            warnings.push(String::from(
                "Requested compressed KV cache is not activated yet; f32 KV is kept until BF16/Q8 improves measured throughput.",
            ));
        } else if options.runtime.kv_cache_dtype == KvCacheDType::Auto
            && self.effective_max_context(options) > 16_384
        {
            warnings.push(String::from(
                "Compressed KV cache requested/eligible, but f32 KV remains active until a benchmark proves BF16/Q8 is faster on this backend.",
            ));
        }
        warnings
    }

    fn verify_speculative_assistant(&self, assistant: &Runner) -> Result<(), String> {
        if self.tok.vocab_size() != assistant.tok.vocab_size() {
            return Err(format!(
                "MTP assistant tokenizer mismatch: target vocab={} assistant vocab={}.",
                self.tok.vocab_size(),
                assistant.tok.vocab_size()
            ));
        }
        if self.tok.bos_id != assistant.tok.bos_id || self.tok.eos_id != assistant.tok.eos_id {
            return Err(format!(
                "MTP assistant BOS/EOS mismatch: target BOS/EOS={}/{} assistant BOS/EOS={}/{}.",
                self.tok.bos_id, self.tok.eos_id, assistant.tok.bos_id, assistant.tok.eos_id
            ));
        }
        if assistant.config.max_seq_len < 2 {
            return Err(String::from(
                "MTP assistant context is too small for speculative decoding.",
            ));
        }
        // Detect DeepSeek/Gemma-style "NextN" MTP heads, which are *not* standalone
        // draft models: they share the target's hidden state via a `nextn.*`
        // projection and typically ship without per-layer attn_k / attn_v weights.
        // Running them through the standard speculative decoder produces gibberish
        // drafts (acceptance rate ~0), so refuse to attach with a clear message.
        let has_nextn = assistant
            .gguf
            .tensors
            .iter()
            .any(|t| t.name.starts_with("nextn.") || t.name.contains(".nextn."));
        let has_any_attn_k = assistant
            .gguf
            .tensors
            .iter()
            .any(|t| t.name.ends_with(".attn_k.weight"));
        if has_nextn || !has_any_attn_k {
            return Err(format!(
                "MTP assistant '{}' is a NextN/MTP-head checkpoint (nextn.* projection{}), \
                 not a standalone draft model. RustyLLM's speculative decoder needs a \
                 separately-trained small LM with full attn_k/attn_v weights. \
                 Use a small standalone model (e.g. a 1B/3B sibling) as --mtp-assistant instead.",
                assistant
                    .model_name()
                    .unwrap_or(assistant.architecture()),
                if has_any_attn_k {
                    ""
                } else {
                    ", missing attn_k weights"
                }
            ));
        }
        Ok(())
    }

    fn effective_profile(&self, options: &GenerationOptions) -> RuntimeProfile {
        if options.runtime.profile != RuntimeProfile::Auto {
            return options.runtime.profile;
        }
        match self.arch.as_str() {
            "mistral" | "mistral3" | "ministral" => RuntimeProfile::Mistral,
            "gemma" | "gemma2" | "gemma4" | "gemma4-assistant" => RuntimeProfile::Gemma,
            _ => RuntimeProfile::Auto,
        }
    }

    fn effective_max_context(&self, options: &GenerationOptions) -> usize {
        if let Some(max_context) = options.runtime.max_context {
            return max_context.min(self.config.max_seq_len).max(1);
        }
        if self.effective_profile(options) == RuntimeProfile::Mistral {
            return self.config.max_seq_len.clamp(1, 8192);
        }
        self.config.max_seq_len.max(1)
    }

    fn effective_sliding_window(&self, options: &GenerationOptions) -> Option<usize> {
        options
            .runtime
            .sliding_window_size
            .or_else(|| (self.config.sliding_window > 0).then_some(self.config.sliding_window))
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

        let runtime_context = self.effective_max_context(options);
        let cache_len = std::cmp::min(runtime_context, tokens.len() + options.max_tokens + 1);

        // For architectures with per-layer layouts (Gemma-4), compute the
        // maximum per-layer head/value sizes so we can allocate buffers and
        // the KV cache with safe upper bounds.
        let (kv_k_dim, kv_v_dim, max_head_dim, max_n_kv_heads, max_value_dim) = self.kv_dims();

        let mut cache = KVCache::new(self.config.n_layers, kv_k_dim, kv_v_dim, cache_len);
        cache.sliding_window = self.effective_sliding_window(options);
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

        let mut speculative = self.prepare_speculative_state(options, &tokens, cache_len);

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

        'decode: while generated_tokens < options.max_tokens {
            if let Some(state) = speculative.as_mut() {
                if options.sampler.temperature == 0.0
                    && state.enabled
                    && options.speculative.enabled
                    && pos < cache_len
                {
                    let outcome = self.verify_batch(
                        state,
                        &mut cache,
                        &mut buf,
                        &mut logits,
                        &mut output,
                        &mut on_token,
                        &options.stop_sequences,
                        max_stop_len,
                        &mut recent,
                        &mut generated_tokens,
                        &mut pos,
                        cache_len,
                        options.max_tokens,
                    );
                    match outcome {
                        DecodeFlow::Continue => continue,
                        DecodeFlow::Stop => break 'decode,
                        DecodeFlow::Fallback => {}
                    }
                }
            }

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
        let speculative_stats = speculative.map(|state| state.stats);
        Ok(GenerationResult {
            text: output,
            stats: GenerationStats {
                prompt_tokens: tokens.len(),
                generated_tokens,
                prefill_time,
                decode_time,
                total_time: total_start.elapsed(),
                cached_tokens: 0,
                speculative: speculative_stats,
            },
        })
    }

    fn prepare_speculative_state<'a>(
        &'a self,
        options: &GenerationOptions,
        tokens: &[u32],
        target_cache_len: usize,
    ) -> Option<SpeculativeState<'a>> {
        if !options.speculative.enabled || options.speculative.max_draft_tokens == 0 {
            return None;
        }
        let assistant = self.speculative_assistant.as_deref()?;
        if options.sampler.temperature != 0.0 {
            eprintln!("MTP: disabled for non-greedy sampling; set --temp 0 for Greedy-MTP.");
            return None;
        }
        let assistant_cache_len = target_cache_len.min(assistant.effective_max_context(options));
        if tokens.len() >= assistant_cache_len {
            eprintln!(
                "MTP: disabled because prompt needs {} tokens but assistant context is {}.",
                tokens.len(),
                assistant_cache_len
            );
            return None;
        }

        let (kv_k_dim, kv_v_dim, max_head_dim, max_n_kv_heads, max_value_dim) = assistant.kv_dims();
        let mut cache = KVCache::new(
            assistant.config.n_layers,
            kv_k_dim,
            kv_v_dim,
            assistant_cache_len,
        );
        cache.sliding_window = assistant.effective_sliding_window(options);
        let mut buf = DecodeBuffer::new(
            &assistant.config,
            max_head_dim,
            max_n_kv_heads,
            max_value_dim,
        );
        let mut logits = Vec::new();
        let last_prompt_pos = tokens.len() - 1;
        let t_draft = Instant::now();
        for (pos, &tok_id) in tokens.iter().enumerate() {
            if pos == last_prompt_pos {
                assistant.forward_token_into(&mut cache, &mut buf, tok_id, pos, &mut logits);
            } else {
                let _ = assistant.forward_hidden_token(&mut cache, &mut buf, tok_id, pos);
            }
        }

        Some(SpeculativeState {
            assistant,
            cache,
            buf,
            logits,
            draft_limit: options.speculative.max_draft_tokens.clamp(1, 2),
            max_draft_tokens: options.speculative.max_draft_tokens,
            adaptive: options.speculative.adaptive,
            min_accept_rate: options.speculative.min_accept_rate,
            stats: SpeculativeStats {
                draft_time: t_draft.elapsed(),
                ..SpeculativeStats::default()
            },
            enabled: true,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn verify_batch<F>(
        &self,
        state: &mut SpeculativeState<'_>,
        cache: &mut KVCache,
        buf: &mut DecodeBuffer,
        logits: &mut Vec<f32>,
        output: &mut String,
        on_token: &mut F,
        stop_sequences: &[String],
        max_stop_len: usize,
        recent: &mut Vec<u32>,
        generated_tokens: &mut usize,
        pos: &mut usize,
        cache_len: usize,
        max_tokens: usize,
    ) -> DecodeFlow
    where
        F: FnMut(&str),
    {
        let limit = state
            .draft_limit
            .min(max_tokens.saturating_sub(*generated_tokens))
            .min(cache_len.saturating_sub(*pos))
            .min(state.cache.max_len.saturating_sub(*pos));
        if limit == 0 {
            return DecodeFlow::Fallback;
        }

        let mut draft = Vec::with_capacity(limit);
        let t_draft = Instant::now();
        for i in 0..limit {
            let token = argmax_finite_token(&state.logits);
            draft.push(token);
            if self.is_stop_token(token) || i + 1 == limit {
                break;
            }
            state.assistant.forward_token_into(
                &mut state.cache,
                &mut state.buf,
                token,
                *pos + i,
                &mut state.logits,
            );
        }
        state.stats.draft_time += t_draft.elapsed();
        state.stats.drafted_tokens += draft.len();

        if draft.is_empty() {
            return DecodeFlow::Fallback;
        }

        let mut accepted_in_batch = 0usize;
        for (idx, &draft_token) in draft.iter().enumerate() {
            let target_token = argmax_finite_token(logits);
            let accepted = target_token == draft_token;
            let token = if accepted {
                state.stats.accepted_tokens += 1;
                accepted_in_batch += 1;
                draft_token
            } else {
                state.stats.rejected_tokens += draft.len() - idx;
                target_token
            };

            if self.is_stop_token(token) {
                return DecodeFlow::Stop;
            }

            if self.emit_token_text(token, output, on_token, stop_sequences, max_stop_len) {
                return DecodeFlow::Stop;
            }

            *generated_tokens += 1;
            push_recent_token(recent, token);

            if *generated_tokens >= max_tokens || *pos >= cache_len {
                return DecodeFlow::Stop;
            }

            self.forward_token_into(cache, buf, token, *pos, logits);

            if !accepted {
                if *pos < state.cache.max_len {
                    state.assistant.forward_token_into(
                        &mut state.cache,
                        &mut state.buf,
                        token,
                        *pos,
                        &mut state.logits,
                    );
                }
                *pos += 1;
                break;
            }

            *pos += 1;
        }

        if state.stats.drafted_tokens >= 16 && state.stats.accept_rate() < state.min_accept_rate {
            state.enabled = false;
            state.stats.disabled = true;
        } else if state.adaptive {
            if accepted_in_batch == draft.len() {
                state.draft_limit = (state.draft_limit + 1).min(state.max_draft_tokens);
            } else {
                state.draft_limit = state.draft_limit.saturating_sub(1).max(1);
            }
        }

        DecodeFlow::Continue
    }

    fn emit_token_text<F>(
        &self,
        token: u32,
        output: &mut String,
        on_token: &mut F,
        stop_sequences: &[String],
        max_stop_len: usize,
    ) -> bool
    where
        F: FnMut(&str),
    {
        let text = self.tok.decode_token(token);
        output.push_str(&text);

        if max_stop_len > 0 {
            let window_start = output.len().saturating_sub(max_stop_len + text.len());
            let window = &output[window_start..];
            for stop in stop_sequences {
                if let Some(rel_idx) = window.find(stop.as_str()) {
                    output.truncate(window_start + rel_idx);
                    return true;
                }
            }
        }

        on_token(&text);
        false
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

        session.kv_cache.sliding_window = self.effective_sliding_window(options);
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
                speculative: None,
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
