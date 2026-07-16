use crate::gguf::{GGMLType, GGUFFile};
use crate::model::{
    self, Config, DecodeBuffer, ExpertWeight, GptOssWeights, KVCache, ModelWeights, Weight,
    apply_rope_qk, online_attention, rms_norm_into,
};
use crate::sampling::{self, SamplerConfig};
use crate::tokenizer::Tokenizer;
use std::cmp::Ordering;
#[cfg(not(target_family = "wasm"))]
use std::collections::HashSet;
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

/// Nearest vocabulary-token result from token-embedding similarity search.
#[derive(Clone, Debug)]
pub struct TokenNeighbor {
    pub id: u32,
    pub score: f32,
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
    pub thinking: ThinkingConfig,
    pub skills: SkillConfig,
    pub speculative: SpeculativeConfig,
    pub runtime: RuntimeOptConfig,
    /// Stop generation when any of these strings appears in the output.
    /// The matched sequence is not included in the returned text.
    pub stop_sequences: Vec<String>,
}

pub const DEFAULT_THINKING_SYSTEM_PROMPT: &str = "Formuliere den folgenden Prompt in eigenen Worten, möglichst kompakt und mit passender Fachterminologie. Identifiziere Sprache und Stil des Original-Prompts. Erkenne, ob der Prompt eine Ausgabe in einem bestimmten Stil fordert; falls kein Stil gefordert wird, orientiere dich am Prompt. Gib ausschließlich den umformulierten Prompt aus, ohne Erklärung, Vorrede oder Markdown.";

#[derive(Clone, Debug)]
pub struct ThinkingConfig {
    pub enabled: bool,
    pub system_prompt: String,
    pub max_tokens: usize,
}

impl Default for ThinkingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            system_prompt: DEFAULT_THINKING_SYSTEM_PROMPT.to_string(),
            max_tokens: 192,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SkillConfig {
    pub directory: Option<String>,
    pub max_skills: usize,
    pub max_bytes_per_skill: usize,
}

impl SkillConfig {
    pub fn is_enabled(&self) -> bool {
        self.directory
            .as_deref()
            .map(|path| !path.trim().is_empty())
            .unwrap_or(false)
    }
}

impl Default for SkillConfig {
    fn default() -> Self {
        Self {
            directory: None,
            max_skills: 3,
            max_bytes_per_skill: 16 * 1024,
        }
    }
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
    MistralUltra,
    Gemma,
}

impl RuntimeProfile {
    /// Parses a runtime optimization profile name.
    pub fn parse(value: &str) -> Option<Self> {
        match value.to_ascii_lowercase().as_str() {
            "auto" => Some(Self::Auto),
            "mistral" | "ministral" => Some(Self::Mistral),
            "ultra" | "mistral-ultra" | "ministral-ultra" | "mistral_ultra" | "ministral_ultra" => {
                Some(Self::MistralUltra)
            }
            "gemma" | "gemma2" | "gemma3" | "gemma4" | "gemma4n" | "gemma4-assistant" => {
                Some(Self::Gemma)
            }
            _ => None,
        }
    }

    /// Returns the stable display name used in boot logs.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Mistral => "mistral",
            Self::MistralUltra => "mistral-ultra",
            Self::Gemma => "gemma",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendPolicy {
    Auto,
    Cpu,
    Metal,
    MetalUltra,
}

impl BackendPolicy {
    /// Parses a runtime backend dispatch policy name.
    pub fn parse(value: &str) -> Option<Self> {
        match value.to_ascii_lowercase().as_str() {
            "auto" => Some(Self::Auto),
            "cpu" | "native" => Some(Self::Cpu),
            "metal" | "gpu" => Some(Self::Metal),
            "metal-ultra" | "metal_ultra" | "ultra" => Some(Self::MetalUltra),
            _ => None,
        }
    }

    /// Returns the stable display name used in boot logs.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Cpu => "cpu",
            Self::Metal => "metal",
            Self::MetalUltra => "metal-ultra",
        }
    }
}

#[derive(Clone, Debug)]
pub struct RuntimeOptConfig {
    pub kv_cache_dtype: KvCacheDType,
    pub flash_attention: bool,
    pub sliding_window_size: Option<usize>,
    pub max_context: Option<usize>,
    pub batch_threads: Option<usize>,
    pub prefill_ubatch: Option<usize>,
    pub auto_batch_threads: bool,
    pub worker_poll_spins: Option<usize>,
    pub profile: RuntimeProfile,
    pub backend_policy: BackendPolicy,
    pub lock_model: bool,
    pub cpu_affinity: bool,
}

impl Default for RuntimeOptConfig {
    /// Uses runtime options that preserve existing model behavior.
    fn default() -> Self {
        Self {
            kv_cache_dtype: KvCacheDType::Auto,
            flash_attention: true,
            sliding_window_size: None,
            max_context: None,
            batch_threads: None,
            prefill_ubatch: None,
            auto_batch_threads: true,
            worker_poll_spins: None,
            profile: RuntimeProfile::Auto,
            backend_policy: BackendPolicy::Auto,
            lock_model: false,
            cpu_affinity: false,
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
            thinking: ThinkingConfig::default(),
            skills: SkillConfig::default(),
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
        if self.thinking.enabled && self.thinking.max_tokens == 0 {
            return Err(String::from("thinking max_tokens must be greater than 0."));
        }
        if self.thinking.enabled && self.thinking.system_prompt.trim().is_empty() {
            return Err(String::from("thinking prompt must not be empty."));
        }
        if self.skills.is_enabled() {
            if self.skills.max_skills == 0 {
                return Err(String::from("--max-skills must be greater than 0."));
            }
            if self.skills.max_bytes_per_skill == 0 {
                return Err(String::from("--skill-max-bytes must be greater than 0."));
            }
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
        if matches!(self.runtime.batch_threads, Some(0)) {
            return Err(String::from("--threads-batch must be greater than 0."));
        }
        if matches!(self.runtime.prefill_ubatch, Some(0)) {
            return Err(String::from("--ubatch must be greater than 0."));
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
    pub mmap_time: Duration,
    pub parse_time: Duration,
    pub compatibility_time: Duration,
    pub tokenizer_time: Duration,
    pub weights_time: Duration,
    pub runner_time: Duration,
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

struct SimdThreadGuard {
    previous: Option<usize>,
}

impl SimdThreadGuard {
    fn set_temporarily(target: Option<usize>) -> Self {
        let Some(target) = target else {
            return Self { previous: None };
        };
        let current = crate::simd::num_threads();
        if target == current {
            return Self { previous: None };
        }
        crate::simd::set_num_threads(target);
        Self {
            previous: Some(current),
        }
    }
}

impl Drop for SimdThreadGuard {
    fn drop(&mut self) {
        if let Some(previous) = self.previous {
            crate::simd::set_num_threads(previous);
        }
    }
}

#[derive(Clone, Copy)]
enum RuntimePhase {
    Prefill,
    Decode,
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

/// Returns a trailing byte offset adjusted to a valid UTF-8 boundary.
fn trailing_char_boundary_start(text: &str, max_bytes: usize) -> usize {
    let mut start = text.len().saturating_sub(max_bytes);
    while start > 0 && !text.is_char_boundary(start) {
        start -= 1;
    }
    start
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

fn clean_thinking_prompt(text: &str) -> String {
    let mut value = text.trim().to_string();
    if let Some(stripped) = value.strip_prefix("```") {
        value = stripped.trim().to_string();
        if let Some((_, rest)) = value.split_once('\n') {
            value = rest.trim().to_string();
        }
        if let Some(stripped) = value.strip_suffix("```") {
            value = stripped.trim().to_string();
        }
    }

    for prefix in [
        "Prompt:",
        "Umformulierter Prompt:",
        "Rewritten prompt:",
        "Reformulated prompt:",
    ] {
        if let Some(stripped) = value.strip_prefix(prefix) {
            value = stripped.trim().to_string();
            break;
        }
    }

    let quoted = (value.starts_with('"') && value.ends_with('"'))
        || (value.starts_with('\'') && value.ends_with('\''));
    if quoted && value.len() >= 2 {
        value = value[1..value.len() - 1].trim().to_string();
    }
    value
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
    /// BERT-style encoder (nomic-bert) — embedding-only, no decode path.
    NomicBert(crate::model::NomicBertWeights),
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
    if is_gemma_arch(arch) {
        return true;
    }
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
            | "nomic-bert-moe"
            | "nomic-embed"
            | "text-embedding-nomic-embed-text"
            | "bert"
    )
}

/// Reports whether an architecture uses the Gemma-family dense decoder path.
/// Reports whether batched prompt prefill is enabled (default on; set
/// `RUSTY_LLM_BATCH_PREFILL=0` to force the per-token path, e.g. for A/B
/// measurement).
#[cfg(not(target_family = "wasm"))]
fn batch_prefill_enabled() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        !matches!(
            std::env::var("RUSTY_LLM_BATCH_PREFILL").as_deref(),
            Ok("0") | Ok("false") | Ok("off")
        )
    })
}

pub(crate) fn is_gemma_arch(arch: &str) -> bool {
    matches!(
        arch,
        "gemma" | "gemma2" | "gemma3" | "gemma4" | "gemma4n" | "gemma4-assistant"
    )
}

/// Reports whether an architecture uses the nomic-bert / BERT encoder path.
pub(crate) fn is_nomic_bert_arch(arch: &str) -> bool {
    matches!(
        arch,
        "nomic-bert"
            | "nomic-bert-moe"
            | "nomic-embed"
            | "text-embedding-nomic-embed-text"
            | "bert"
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
                "Tensor '{}' uses an unsupported quantization type. Please re-quantize the model using F16/BF16, Q8_0/Q8_1, Q4_0/Q4_1, Q5_0/Q5_1, Q4_K, Q5_K, Q6_K, or MXFP4.",
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
            | GGMLType::BF16
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
        "gpt-oss" => return,
        _ if is_gemma_arch(arch) => {
            validate_gemma_layout(gguf, &config, report);
            return;
        }
        _ if is_nomic_bert_arch(arch) => {
            validate_nomic_bert_layout(gguf, &config, report);
            return;
        }
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

fn validate_gemma_layout(gguf: &GGUFFile, config: &Config, report: &mut CompatibilityReport) {
    let has = |name: &str| gguf.tensors.iter().any(|tensor| tensor.name == name);
    let has_layer_weight = |layer: usize, canonical: &str, subs: &[&str]| {
        let canonical_name = format!("blk.{}.{}", layer, canonical);
        if has(&canonical_name) {
            return true;
        }
        let prefix = format!("blk.{}.", layer);
        gguf.tensors.iter().any(|tensor| {
            tensor.name.starts_with(&prefix)
                && tensor.name.ends_with(".weight")
                && subs.iter().all(|needle| tensor.name.contains(needle))
        })
    };

    if config.dim == 0 || config.n_layers == 0 || config.n_heads == 0 || config.n_kv_heads == 0 {
        report
            .unsupported_layouts
            .push(String::from("missing required Gemma transformer metadata"));
        return;
    }

    for name in ["token_embd.weight", "output_norm.weight"] {
        if !has(name) {
            report.missing_tensors.push(name.to_string());
        }
    }

    for layer in 0..config.n_layers {
        for suffix in ["attn_norm.weight", "ffn_norm.weight"] {
            let name = format!("blk.{}.{}", layer, suffix);
            if !has(&name) {
                report.missing_tensors.push(name);
            }
        }
        for (canonical, subs) in [
            ("attn_q.weight", &["attn", "q"][..]),
            ("attn_k.weight", &["attn", "k"][..]),
            ("attn_output.weight", &["attn", "output"][..]),
            ("ffn_gate.weight", &["ffn", "gate"][..]),
            ("ffn_up.weight", &["ffn", "up"][..]),
            ("ffn_down.weight", &["ffn", "down"][..]),
        ] {
            if !has_layer_weight(layer, canonical, subs) {
                report
                    .missing_tensors
                    .push(format!("blk.{}.{}", layer, canonical));
            }
        }
    }
}

/// Validates the tensor layout of a nomic-bert / BERT encoder. Unlike the
/// decoder path this requires no output head or output_norm; instead it
/// expects LayerNorm weight+bias pairs and either a fused `attn_qkv` or split
/// q/k/v. `ffn_gate` is required for the SwiGLU nomic-bert arch and optional
/// for plain BERT (which uses a GELU FFN).
fn validate_nomic_bert_layout(gguf: &GGUFFile, config: &Config, report: &mut CompatibilityReport) {
    let has = |name: &str| gguf.tensors.iter().any(|tensor| tensor.name == name);

    if config.dim == 0 || config.n_layers == 0 || config.n_heads == 0 {
        report.unsupported_layouts.push(String::from(
            "missing required nomic-bert transformer metadata",
        ));
        return;
    }

    for name in [
        "token_embd.weight",
        "token_embd_norm.weight",
        "token_embd_norm.bias",
    ] {
        if !has(name) {
            report.missing_tensors.push(name.to_string());
        }
    }

    let arch = gguf.get_str("general.architecture").unwrap_or("nomic-bert");
    let require_gate = arch == "nomic-bert";

    for layer in 0..config.n_layers {
        let prefix = format!("blk.{}.", layer);
        let q = format!("{}attn_q.weight", prefix);
        let k = format!("{}attn_k.weight", prefix);
        let v = format!("{}attn_v.weight", prefix);
        let qkv = format!("{}attn_qkv.weight", prefix);
        if !(has(&qkv) || (has(&q) && has(&k) && has(&v))) {
            report.missing_tensors.push(format!(
                "{}attn_qkv.weight or attn_q/attn_k/attn_v.weight",
                prefix
            ));
        }
        for suffix in [
            "attn_output.weight",
            "attn_output_norm.weight",
            "attn_output_norm.bias",
            "ffn_up.weight",
            "ffn_down.weight",
            "layer_output_norm.weight",
            "layer_output_norm.bias",
        ] {
            let name = format!("{}{}", prefix, suffix);
            if !has(&name) {
                report.missing_tensors.push(name);
            }
        }
        if require_gate {
            let gate = format!("{}ffn_gate.weight", prefix);
            if !has(&gate) {
                report.missing_tensors.push(gate);
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
            | GGMLType::BF16
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
	                     F16/BF16, Q8_0, Q8_1, Q4_0, Q4_1, Q5_0, Q5_1, Q4_K, Q5_K, Q6_K, or MXFP4.",
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
            arch if is_gemma_arch(arch) => {
                let (config, weights) = model::load_gemma4_model(data, &gguf, false);
                (config, LoadedWeights::Gemma4(weights))
            }
            arch if is_nomic_bert_arch(arch) => {
                let (config, weights) = model::load_nomic_bert_model(data, &gguf, false);
                (config, LoadedWeights::NomicBert(weights))
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
        Self::from_path_with_options(path, &GenerationOptions::default())
    }

    #[cfg(not(target_family = "wasm"))]
    /// Loads a runner by memory-mapping a GGUF file path with runtime load options.
    pub fn from_path_with_options(
        path: &str,
        options: &GenerationOptions,
    ) -> Result<(Self, LoadInfo), String> {
        let t0 = Instant::now();
        let t_mmap = Instant::now();
        let mut mmap = crate::mmap::MmapFile::open(path)
            .map_err(|err| format!("Failed to open model: {}", err))?;
        if options.runtime.lock_model {
            if let Err(err) = mmap.lock_in_memory() {
                eprintln!(
                    "Warning: --mlock requested, but model pages could not be locked: {}",
                    err
                );
            }
        }
        let mmap_time = t_mmap.elapsed();
        let file_size_bytes = mmap.len();
        let t_parse = Instant::now();
        let gguf = GGUFFile::parse_quiet(mmap.as_slice())?;
        let parse_time = t_parse.elapsed();
        let arch = gguf
            .get_str("general.architecture")
            .unwrap_or("llama")
            .to_string();

        let t_compatibility = Instant::now();
        let compatibility = compatibility_report(&gguf);
        if !compatibility.is_supported() {
            return Err(compatibility.first_error(&arch));
        }
        let compatibility_time = t_compatibility.elapsed();

        let t_tokenizer = Instant::now();
        let tok = Tokenizer::from_metadata(&gguf.metadata);
        let tokenizer_time = t_tokenizer.elapsed();
        let t_weights = Instant::now();
        let (config, weights) = match arch.as_str() {
            "gpt-oss" => {
                let (config, weights) = model::load_gpt_oss_model(mmap.as_slice(), &gguf, true);
                (config, LoadedWeights::GptOss(weights))
            }
            arch if is_gemma_arch(arch) => {
                let (config, weights) = model::load_gemma4_model(mmap.as_slice(), &gguf, true);
                (config, LoadedWeights::Gemma4(weights))
            }
            arch if is_nomic_bert_arch(arch) => {
                let (config, weights) = model::load_nomic_bert_model(mmap.as_slice(), &gguf, true);
                (config, LoadedWeights::NomicBert(weights))
            }
            // Platzhalter für DeepSeek, weitere Loader können analog ergänzt werden
            _ => {
                let (config, weights) = model::load_model(mmap.as_slice(), &gguf, true);
                (config, LoadedWeights::Standard(weights))
            }
        };
        let weights_time = t_weights.elapsed();

        let t_runner = Instant::now();
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
        let runner_time = t_runner.elapsed();
        let load_time = t0.elapsed();
        Ok((
            runner,
            LoadInfo {
                file_size_bytes,
                load_time,
                mmap_time,
                parse_time,
                compatibility_time,
                tokenizer_time,
                weights_time,
                runner_time,
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

    /// Returns the number of rows available in the loaded token embedding matrix.
    pub fn token_embedding_count(&self) -> usize {
        let dim = self.config.dim;
        match self.token_embedding_weight() {
            Weight::F32(data) if dim > 0 => data.len() / dim,
            Weight::F32(_) => 0,
            Weight::Quantized { rows, .. } => *rows,
        }
    }

    /// Returns one raw token-embedding row as f32 values.
    pub fn token_embedding(&self, token_id: u32) -> Result<Vec<f32>, String> {
        let mut out = Vec::new();
        self.token_embedding_into(token_id, &mut out)?;
        Ok(out)
    }

    /// Mean-pools token-embedding rows for `text` in token-embedding space.
    ///
    /// This is intentionally different from [`Runner::embed`]: it does not run
    /// transformer layers. It stays in the static `token_embd.weight` space so
    /// vocabulary-token nearest-neighbor search compares like with like.
    pub fn token_embedding_query(&self, text: &str) -> Result<(Vec<f32>, Vec<u32>), String> {
        let tokens = self.tok.encode_without_bos(text);
        if tokens.is_empty() {
            return Err(String::from(
                "token_embedding_query: input tokenised to zero tokens",
            ));
        }

        let mut sum = vec![0.0f32; self.config.dim];
        let mut row = Vec::new();
        let mut used = Vec::new();
        for token in tokens {
            if self.token_embedding_into(token, &mut row).is_ok() {
                for (dst, value) in sum.iter_mut().zip(row.iter()) {
                    *dst += *value;
                }
                used.push(token);
            }
        }
        if used.is_empty() {
            return Err(String::from(
                "token_embedding_query: no input token has an embedding row",
            ));
        }

        mean_pool_in_place(&mut sum, used.len());
        l2_normalize_in_place(&mut sum);
        Ok((sum, used))
    }

    /// Finds the closest vocabulary tokens in static token-embedding space.
    pub fn nearest_token_embeddings(
        &self,
        query: &[f32],
        limit: usize,
        include_special: bool,
    ) -> Result<Vec<TokenNeighbor>, String> {
        if query.len() != self.config.dim {
            return Err(format!(
                "nearest_token_embeddings: dimension mismatch ({} vs {})",
                query.len(),
                self.config.dim
            ));
        }

        let mut query_norm = 0.0f32;
        for value in query {
            query_norm += value * value;
        }
        query_norm = query_norm.sqrt();
        if query_norm <= 1e-12 {
            return Err(String::from(
                "nearest_token_embeddings: zero-norm query vector",
            ));
        }

        let row_count = self.token_embedding_count().min(self.tok.vocab_size());
        let mut row = Vec::with_capacity(self.config.dim);
        let mut neighbors = Vec::with_capacity(row_count.min(limit.max(1) * 4));
        for token_id in 0..row_count {
            if !include_special && self.skip_explorer_token(token_id as u32) {
                continue;
            }
            self.token_embedding_into(token_id as u32, &mut row)?;

            let mut dot = 0.0f32;
            let mut row_norm = 0.0f32;
            for (a, b) in query.iter().zip(row.iter()) {
                dot += a * b;
                row_norm += b * b;
            }
            row_norm = row_norm.sqrt();
            if row_norm <= 1e-12 {
                continue;
            }
            neighbors.push(TokenNeighbor {
                id: token_id as u32,
                score: dot / (query_norm * row_norm),
            });
        }

        neighbors.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
        neighbors.truncate(limit.clamp(1, 100));
        Ok(neighbors)
    }

    fn token_embedding_weight(&self) -> &Weight {
        match &self.weights {
            LoadedWeights::Standard(weights) => &weights.token_embd,
            LoadedWeights::GptOss(weights) => &weights.token_embd,
            LoadedWeights::Gemma4(weights) => &weights.token_embd,
            LoadedWeights::NomicBert(weights) => &weights.token_embd,
        }
    }

    fn token_embedding_into(&self, token_id: u32, out: &mut Vec<f32>) -> Result<(), String> {
        let row_count = self.token_embedding_count();
        if token_id as usize >= row_count {
            return Err(format!(
                "token_embedding: token id {} is out of range for {} embedding rows",
                token_id, row_count
            ));
        }
        self.token_embedding_weight()
            .row_into(token_id as usize, self.config.dim, out);
        Ok(())
    }

    fn skip_explorer_token(&self, token_id: u32) -> bool {
        if token_id == self.tok.bos_id || token_id == self.tok.eos_id {
            return true;
        }
        let raw = self.tok.raw_token(token_id).unwrap_or("");
        let decoded = self.tok.decode_token(token_id);
        if decoded.trim().is_empty() {
            return true;
        }
        if raw.starts_with("<0x") {
            return true;
        }
        (raw.starts_with('<') && raw.ends_with('>')) || (raw.starts_with('[') && raw.ends_with(']'))
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
        let backend = self.effective_backend_policy(options);
        if options.runtime.backend_policy == BackendPolicy::Auto && backend != BackendPolicy::Auto {
            items.push(format!("backend={} (auto)", backend.as_str()));
        } else {
            items.push(format!("backend={}", backend.as_str()));
        }
        let chat_template = if self.arch == "gpt-oss" {
            "gpt-oss"
        } else {
            self.chat_template_kind().unwrap_or("plain")
        };
        items.push(format!("chat-template={}", chat_template));
        items.push(match options.runtime.kv_cache_dtype {
            KvCacheDType::Auto => String::from("kv-cache=f32(auto)"),
            KvCacheDType::F32 => String::from("kv-cache=f32"),
            requested => format!(
                "kv-cache=f32(requested {} pending-bench)",
                requested.as_str()
            ),
        });
        if options.runtime.flash_attention {
            items.push(String::from("flash-attn=online-softmax+linear-kv"));
        } else {
            items.push(String::from("flash-attn=off"));
        }
        if let Some(window) = self.effective_sliding_window(options) {
            items.push(format!("sliding-window={} tokens", window));
        }
        if let Some(threads) = options.runtime.batch_threads {
            items.push(format!("threads-batch={}", threads));
        } else if options.runtime.auto_batch_threads {
            items.push(String::from("threads-batch=auto"));
        }
        if let Some(spins) = options.runtime.worker_poll_spins {
            items.push(format!("worker-poll={}", spins));
        } else {
            items.push(format!("worker-poll={}", crate::simd::worker_poll_spins()));
        }
        let ubatch = options
            .runtime
            .prefill_ubatch
            .map(|value| value.to_string())
            .unwrap_or_else(|| String::from("auto"));
        items.push(format!("ubatch={}", ubatch));
        if options.runtime.lock_model {
            items.push(String::from("mlock=best-effort"));
        }
        if options.runtime.cpu_affinity {
            items.push(String::from("cpu-affinity=best-effort"));
        }
        items.push(format!(
            "max-context={} tokens",
            self.effective_max_context(options)
        ));
        if options.thinking.enabled {
            items.push(format!(
                "thinking=on max_tokens={}",
                options.thinking.max_tokens
            ));
        } else {
            items.push(String::from("thinking=off"));
        }
        if let Some(directory) = options.skills.directory.as_deref() {
            if !directory.trim().is_empty() {
                items.push(format!(
                    "skills=on dir={} max={}",
                    directory, options.skills.max_skills
                ));
            } else {
                items.push(String::from("skills=off"));
            }
        } else {
            items.push(String::from("skills=off"));
        }
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
        if matches!(
            profile,
            RuntimeProfile::Mistral | RuntimeProfile::MistralUltra
        ) && self.config.max_seq_len >= 131_072
        {
            warnings.push(format!(
                "Ministral context metadata advertises {} tokens; default runtime cap is {} unless --max-context overrides it.",
                self.config.max_seq_len,
                self.effective_max_context(options)
            ));
        }
        if profile == RuntimeProfile::MistralUltra && !crate::metal::enabled() {
            warnings.push(String::from(
                "Mistral ultra profile requested, but Metal is unavailable or disabled; falling back to native SIMD CPU kernels.",
            ));
        } else if profile == RuntimeProfile::MistralUltra && self.arch == "mistral3" {
            warnings.push(String::from(
                "Mistral ultra uses aggressive Metal routing; use --backend metal or --backend cpu for A/B checks if it regresses on this machine.",
            ));
        }
        if matches!(
            self.effective_backend_policy(options),
            BackendPolicy::Metal | BackendPolicy::MetalUltra
        ) && !crate::metal::enabled()
        {
            warnings.push(String::from(
                "Metal backend policy was requested, but Metal is unavailable or disabled; native CPU kernels will run.",
            ));
        }
        if self
            .gguf
            .metadata
            .get("tokenizer.chat_template")
            .and_then(|value| value.as_str())
            .is_some()
            && self.chat_template_kind().is_none()
            && self.arch != "gpt-oss"
        {
            warnings.push(String::from(
                "Tokenizer has a chat template, but RustyLLM does not recognize it yet; falling back to plain role-prefixed chat formatting.",
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
                assistant.model_name().unwrap_or(assistant.architecture()),
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
            arch if is_gemma_arch(arch) => RuntimeProfile::Gemma,
            _ => RuntimeProfile::Auto,
        }
    }

    fn effective_max_context(&self, options: &GenerationOptions) -> usize {
        if let Some(max_context) = options.runtime.max_context {
            return max_context.min(self.config.max_seq_len).max(1);
        }
        if matches!(
            self.effective_profile(options),
            RuntimeProfile::Mistral | RuntimeProfile::MistralUltra
        ) {
            return self.config.max_seq_len.clamp(1, 8192);
        }
        self.config.max_seq_len.max(1)
    }

    fn effective_backend_policy(&self, options: &GenerationOptions) -> BackendPolicy {
        if options.runtime.backend_policy != BackendPolicy::Auto {
            return options.runtime.backend_policy;
        }
        if self.effective_profile(options) == RuntimeProfile::MistralUltra {
            return BackendPolicy::MetalUltra;
        }
        // Small Gemma models are dispatch-bound on the per-op Metal path and
        // measured SLOWER than CPU (gemma-4-E2B 22.0 vs 27.6 t/s, 26B-A4B 19.7
        // vs 23.8; see BENCHMARK.md), since gemma4 has no resident decoder or
        // fused FFN. Default them to CPU unless RUSTY_LLM_GEMMA_METAL=1.
        if is_gemma_arch(&self.arch)
            && self.config.dim < 3072
            && crate::metal::enabled()
            && std::env::var("RUSTY_LLM_GEMMA_METAL").as_deref() != Ok("1")
        {
            return BackendPolicy::Cpu;
        }
        if self.arch == "mistral3" && crate::metal::enabled() {
            BackendPolicy::Metal
        } else {
            BackendPolicy::Auto
        }
    }

    fn scoped_backend_dispatch(
        &self,
        options: &GenerationOptions,
        _phase: RuntimePhase,
    ) -> crate::metal::DispatchPolicyGuard {
        match self.effective_backend_policy(options) {
            BackendPolicy::Cpu => crate::metal::scoped_dispatch_policy(true, false),
            BackendPolicy::MetalUltra => crate::metal::scoped_dispatch_policy(false, true),
            BackendPolicy::Auto | BackendPolicy::Metal => {
                crate::metal::scoped_dispatch_policy(false, false)
            }
        }
    }

    fn effective_sliding_window(&self, options: &GenerationOptions) -> Option<usize> {
        options
            .runtime
            .sliding_window_size
            .or_else(|| (self.config.sliding_window > 0).then_some(self.config.sliding_window))
    }

    fn effective_prefill_ubatch(&self, options: &GenerationOptions, prompt_tokens: usize) -> usize {
        let tokens = prompt_tokens.max(1);
        options
            .runtime
            .prefill_ubatch
            .unwrap_or({
                if tokens >= 1024 {
                    256
                } else if tokens >= 256 {
                    128
                } else {
                    64
                }
            })
            .clamp(1, tokens)
    }

    fn effective_prefill_threads(
        &self,
        options: &GenerationOptions,
        prompt_tokens: usize,
    ) -> Option<usize> {
        if let Some(threads) = options.runtime.batch_threads {
            return Some(threads);
        }
        if !options.runtime.auto_batch_threads || prompt_tokens < 32 {
            return None;
        }
        // Widen for long prompts, but never past one thread per physical
        // core: SMT siblings contend in the vector units and measurably slow
        // the bandwidth-bound matvec kernels (prefill at 12 logical threads
        // benchmarked BELOW the 6-physical-core decode default on a 6C/12T
        // i7). On chips without SMT (Apple Silicon) this is a no-op.
        let target = crate::simd::available_threads().min(crate::simd::physical_threads());
        let current = crate::simd::num_threads();
        (target > current).then_some(target)
    }

    /// Benchmarks representative projection kernels for the loaded model.
    pub fn kernel_benchmark(
        &self,
        runs: usize,
        requested_layer: usize,
    ) -> Result<(usize, Vec<KernelBenchRow>), String> {
        self.kernel_benchmark_with_options(runs, requested_layer, &GenerationOptions::default())
    }

    /// Benchmarks representative projection kernels under the provided runtime options.
    pub fn kernel_benchmark_with_options(
        &self,
        runs: usize,
        requested_layer: usize,
        options: &GenerationOptions,
    ) -> Result<(usize, Vec<KernelBenchRow>), String> {
        if runs == 0 {
            return Err(String::from("--kernel-bench-runs must be greater than 0."));
        }
        let _backend_guard = self.scoped_backend_dispatch(options, RuntimePhase::Decode);
        match &self.weights {
            LoadedWeights::Standard(weights) => {
                self.standard_kernel_benchmark(weights, runs, requested_layer)
            }
            LoadedWeights::GptOss(weights) => {
                self.gpt_oss_kernel_benchmark(weights, runs, requested_layer)
            }
            LoadedWeights::Gemma4(_) => Err(String::from(
                "--kernel-bench currently supports standard transformer and gpt-oss MoE weights only.",
            )),
            LoadedWeights::NomicBert(_) => Err(String::from(
                "--kernel-bench does not support nomic-bert encoder weights.",
            )),
        }
    }

    fn standard_kernel_benchmark(
        &self,
        weights: &ModelWeights,
        runs: usize,
        requested_layer: usize,
    ) -> Result<(usize, Vec<KernelBenchRow>), String> {
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
        let attn_q = measure_matvec("attn_q", &layer_weights.wq, &dim_input, runs, &mut out);
        rows.push(attn_q.clone());
        let attn_k = measure_matvec("attn_k", &layer_weights.wk, &dim_input, runs, &mut out);
        rows.push(attn_k.clone());
        let attn_v = measure_matvec("attn_v", &layer_weights.wv, &dim_input, runs, &mut out);
        rows.push(attn_v.clone());
        let attn_qkv_fused = measure_fused_kquant3(
            "attn_qkv.fused",
            &layer_weights.wq,
            &layer_weights.wk,
            &layer_weights.wv,
            &dim_input,
            runs,
        );
        if let Some(row) = &attn_qkv_fused {
            rows.push(row.clone());
        }
        let attn_output = measure_matvec(
            "attn_output",
            &layer_weights.wo,
            &attn_out_input,
            runs,
            &mut out,
        );
        rows.push(attn_output.clone());
        let ffn_gate = measure_matvec("ffn_gate", &layer_weights.w1, &dim_input, runs, &mut out);
        rows.push(ffn_gate.clone());
        let ffn_up = measure_matvec("ffn_up", &layer_weights.w3, &dim_input, runs, &mut out);
        rows.push(ffn_up.clone());
        let ffn_gate_up_fused = measure_fused_kquant2(
            "ffn_gate_up.fused",
            &layer_weights.w1,
            &layer_weights.w3,
            &dim_input,
            runs,
        );
        if let Some(row) = &ffn_gate_up_fused {
            rows.push(row.clone());
        }
        let ffn_down = measure_matvec("ffn_down", &layer_weights.w2, &hidden_input, runs, &mut out);
        rows.push(ffn_down.clone());
        let ffn_block_fused = measure_mistral_ffn_block(
            "ffn_block.fused",
            &layer_weights.w1,
            &layer_weights.w3,
            &layer_weights.w2,
            &dim_input,
            runs,
        );
        if let Some(row) = &ffn_block_fused {
            rows.push(row.clone());
        }
        let output = measure_matvec("output", &weights.output, &dim_input, runs, &mut out);
        rows.push(output.clone());

        rows.push(measure_rms_norm(
            "attn_norm.rms",
            &dim_input,
            &layer_weights.attn_norm,
            self.config.rms_norm_eps,
            runs,
        ));
        rows.push(measure_rope_qk(
            "attn_rope.qk",
            self.config.n_heads * self.config.head_dim,
            self.config.n_kv_heads * self.config.head_dim,
            self.config.head_dim,
            self.config.n_heads,
            self.config.n_kv_heads,
            &deterministic_rope_inv_freq(
                self.config.rope_theta,
                self.config.head_dim,
                self.config.rope_scaling_factor,
                self.config.rope_original_context_length,
            ),
            runs,
        ));
        let mut attn_ctx_lens = vec![
            32usize.min(self.config.max_seq_len),
            64usize.min(self.config.max_seq_len),
            128usize.min(self.config.max_seq_len),
            256usize.min(self.config.max_seq_len),
            512usize.min(self.config.max_seq_len),
            1024usize.min(self.config.max_seq_len),
            2048usize.min(self.config.max_seq_len),
            4096usize.min(self.config.max_seq_len),
            8192usize.min(self.config.max_seq_len),
        ];
        attn_ctx_lens.sort_unstable();
        attn_ctx_lens.dedup();
        for ctx_len in attn_ctx_lens.into_iter().filter(|ctx| *ctx > 0) {
            rows.push(measure_attention_scan(
                &format!("attn_scan.ctx{}", ctx_len),
                self.config.n_heads,
                self.config.n_kv_heads,
                self.config.kv_mul,
                self.config.head_dim,
                self.config.value_dim,
                ctx_len,
                runs,
            ));
        }
        rows.push(measure_rms_norm(
            "ffn_norm.rms",
            &dim_input,
            &layer_weights.ffn_norm,
            self.config.rms_norm_eps,
            runs,
        ));
        rows.push(measure_ffn_activation(
            "ffn_activation.silu_mul",
            self.config.hidden_dim,
            runs,
        ));

        let attn_qkv_ms = attn_qkv_fused
            .as_ref()
            .map(|row| row.avg_ms)
            .unwrap_or(attn_q.avg_ms + attn_k.avg_ms + attn_v.avg_ms);
        let ffn_gate_up_ms = ffn_gate_up_fused
            .as_ref()
            .map(|row| row.avg_ms)
            .unwrap_or(ffn_gate.avg_ms + ffn_up.avg_ms);
        let ffn_block_ms = ffn_block_fused
            .as_ref()
            .map(|row| row.avg_ms)
            .unwrap_or(ffn_gate_up_ms + ffn_down.avg_ms);
        let layer_projection_ms = attn_qkv_ms + attn_output.avg_ms + ffn_block_ms;
        rows.push(estimated_kernel_row(
            "layer_projections.estimated",
            self.config.dim,
            runs,
            layer_projection_ms,
        ));
        rows.push(estimated_kernel_row(
            "decode_projections.estimated",
            self.config.dim,
            runs,
            layer_projection_ms * self.config.n_layers as f64 + output.avg_ms,
        ));
        rows.push(estimated_kernel_row(
            "decode_output_projection.estimated",
            self.config.dim,
            runs,
            output.avg_ms,
        ));
        Ok((layer, rows))
    }

    fn gpt_oss_kernel_benchmark(
        &self,
        weights: &GptOssWeights,
        runs: usize,
        requested_layer: usize,
    ) -> Result<(usize, Vec<KernelBenchRow>), String> {
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

        let attn_q = measure_matvec("attn_q", &layer_weights.wq, &dim_input, runs, &mut out);
        rows.push(attn_q.clone());
        let attn_k = measure_matvec("attn_k", &layer_weights.wk, &dim_input, runs, &mut out);
        rows.push(attn_k.clone());
        let attn_v = measure_matvec("attn_v", &layer_weights.wv, &dim_input, runs, &mut out);
        rows.push(attn_v.clone());
        let attn_qkv_fused = measure_fused_kquant3(
            "attn_qkv.fused",
            &layer_weights.wq,
            &layer_weights.wk,
            &layer_weights.wv,
            &dim_input,
            runs,
        );
        if let Some(row) = &attn_qkv_fused {
            rows.push(row.clone());
        }

        let attn_output = measure_matvec(
            "attn_output",
            &layer_weights.wo,
            &attn_out_input,
            runs,
            &mut out,
        );
        rows.push(attn_output.clone());
        let router = measure_matvec(
            "router_gate",
            &layer_weights.gate_inp,
            &dim_input,
            runs,
            &mut out,
        );
        rows.push(router.clone());

        let expert_idx = 0usize;
        let expert_gate = measure_expert_matvec(
            "expert0_gate",
            &layer_weights.gate_exps,
            expert_idx,
            &dim_input,
            runs,
            &mut out,
        );
        rows.push(expert_gate.clone());
        let expert_up = measure_expert_matvec(
            "expert0_up",
            &layer_weights.up_exps,
            expert_idx,
            &dim_input,
            runs,
            &mut out,
        );
        rows.push(expert_up.clone());
        let expert_gate_up_fused = measure_expert_matvec_pair(
            "expert0_gate_up.fused",
            &layer_weights.gate_exps,
            &layer_weights.up_exps,
            expert_idx,
            &dim_input,
            runs,
        );
        if let Some(row) = &expert_gate_up_fused {
            rows.push(row.clone());
        }
        let expert_down = measure_expert_matvec(
            "expert0_down",
            &layer_weights.down_exps,
            expert_idx,
            &hidden_input,
            runs,
            &mut out,
        );
        rows.push(expert_down.clone());

        let output = measure_matvec("output", &weights.output, &dim_input, runs, &mut out);
        rows.push(output.clone());

        rows.push(measure_rms_norm(
            "attn_norm.rms",
            &dim_input,
            &layer_weights.attn_norm,
            self.config.rms_norm_eps,
            runs,
        ));
        rows.push(measure_rope_qk(
            "attn_rope.qk",
            self.config.n_heads * self.config.head_dim,
            self.config.n_kv_heads * self.config.head_dim,
            self.config.head_dim,
            self.config.n_heads,
            self.config.n_kv_heads,
            &deterministic_rope_inv_freq(
                self.config.rope_theta,
                self.config.head_dim,
                self.config.rope_scaling_factor,
                self.config.rope_original_context_length,
            ),
            runs,
        ));

        let mut attn_ctx_lens = vec![
            32usize.min(self.config.max_seq_len),
            64usize.min(self.config.max_seq_len),
            128usize.min(self.config.max_seq_len),
            256usize.min(self.config.max_seq_len),
            512usize.min(self.config.max_seq_len),
            1024usize.min(self.config.max_seq_len),
            2048usize.min(self.config.max_seq_len),
            4096usize.min(self.config.max_seq_len),
            8192usize.min(self.config.max_seq_len),
        ];
        attn_ctx_lens.sort_unstable();
        attn_ctx_lens.dedup();
        for ctx_len in attn_ctx_lens.into_iter().filter(|ctx| *ctx > 0) {
            rows.push(measure_attention_scan(
                &format!("attn_scan.ctx{}", ctx_len),
                self.config.n_heads,
                self.config.n_kv_heads,
                self.config.kv_mul,
                self.config.head_dim,
                self.config.value_dim,
                ctx_len,
                runs,
            ));
        }

        rows.push(measure_rms_norm(
            "ffn_norm.rms",
            &dim_input,
            &layer_weights.post_attn_norm,
            self.config.rms_norm_eps,
            runs,
        ));
        rows.push(measure_ffn_activation(
            "expert_activation.swiglu",
            self.config.hidden_dim,
            runs,
        ));

        let attn_qkv_ms = attn_qkv_fused
            .as_ref()
            .map(|row| row.avg_ms)
            .unwrap_or(attn_q.avg_ms + attn_k.avg_ms + attn_v.avg_ms);
        let expert_gate_up_ms = expert_gate_up_fused
            .as_ref()
            .map(|row| row.avg_ms)
            .unwrap_or(expert_gate.avg_ms + expert_up.avg_ms);
        let expert_triplet_ms = expert_gate_up_ms + expert_down.avg_ms;
        rows.push(estimated_kernel_row(
            "expert0_triplet.estimated",
            self.config.dim,
            runs,
            expert_triplet_ms,
        ));
        let routed_expert_ms = expert_triplet_ms * self.config.expert_used_count.max(1) as f64;
        rows.push(estimated_kernel_row(
            "routed_experts.estimated",
            self.config.dim,
            runs,
            routed_expert_ms,
        ));
        let layer_projection_ms =
            attn_qkv_ms + attn_output.avg_ms + router.avg_ms + routed_expert_ms;
        rows.push(estimated_kernel_row(
            "layer_projections.estimated",
            self.config.dim,
            runs,
            layer_projection_ms,
        ));
        rows.push(estimated_kernel_row(
            "decode_projections.estimated",
            self.config.dim,
            runs,
            layer_projection_ms * self.config.n_layers as f64 + output.avg_ms,
        ));
        rows.push(estimated_kernel_row(
            "decode_output_projection.estimated",
            self.config.dim,
            runs,
            output.avg_ms,
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

    #[cfg(not(target_family = "wasm"))]
    fn options_with_skill_context(
        &self,
        messages: &[ChatMessage],
        options: &GenerationOptions,
        loaded_skills: &mut HashSet<String>,
    ) -> Result<GenerationOptions, String> {
        let bundle =
            crate::skills::prepare_skill_context(&options.skills, messages, loaded_skills)?;
        if bundle.system_prompt_suffix.is_empty() {
            return Ok(options.clone());
        }

        for path in &bundle.loaded_paths {
            loaded_skills.insert(path.clone());
        }

        let mut generation = options.clone();
        generation.system_prompt =
            crate::skills::append_skill_context(&generation.system_prompt, &bundle);
        Ok(generation)
    }

    /// Generates a complete response for a plain prompt with prompt-selected skills.
    #[cfg(not(target_family = "wasm"))]
    pub fn generate_with_skill_memory(
        &self,
        prompt: &str,
        options: &GenerationOptions,
        loaded_skills: &mut HashSet<String>,
    ) -> Result<GenerationResult, String> {
        let messages = [ChatMessage::user(prompt)];
        self.generate_chat_stream_with_skill_memory(&messages, options, loaded_skills, |_| {})
    }

    /// Generates a chat response with prompt-selected skills.
    #[cfg(not(target_family = "wasm"))]
    pub fn generate_chat_with_skill_memory(
        &self,
        messages: &[ChatMessage],
        options: &GenerationOptions,
        loaded_skills: &mut HashSet<String>,
    ) -> Result<GenerationResult, String> {
        self.generate_chat_stream_with_skill_memory(messages, options, loaded_skills, |_| {})
    }

    /// Generates a prompt response while streaming decoded chunks and loading each matching skill once.
    #[cfg(not(target_family = "wasm"))]
    pub fn generate_stream_with_skill_memory<F>(
        &self,
        prompt: &str,
        options: &GenerationOptions,
        loaded_skills: &mut HashSet<String>,
        on_token: F,
    ) -> Result<GenerationResult, String>
    where
        F: FnMut(&str),
    {
        let messages = [ChatMessage::user(prompt)];
        self.generate_chat_stream_with_skill_memory(&messages, options, loaded_skills, on_token)
    }

    /// Generates chat text while loading only prompt-relevant `SKILL.md` files once per memory set.
    #[cfg(not(target_family = "wasm"))]
    pub fn generate_chat_stream_with_skill_memory<F>(
        &self,
        messages: &[ChatMessage],
        options: &GenerationOptions,
        loaded_skills: &mut HashSet<String>,
        on_token: F,
    ) -> Result<GenerationResult, String>
    where
        F: FnMut(&str),
    {
        let generation = self.options_with_skill_context(messages, options, loaded_skills)?;
        self.generate_chat_stream(messages, &generation, on_token)
    }

    /// Generates a chat response while streaming decoded text chunks.
    pub fn generate_chat_stream<F>(
        &self,
        messages: &[ChatMessage],
        options: &GenerationOptions,
        on_token: F,
    ) -> Result<GenerationResult, String>
    where
        F: FnMut(&str),
    {
        options.validate()?;
        if matches!(self.weights, LoadedWeights::NomicBert(_)) {
            return Err(String::from(
                "nomic-bert is an encoder-only embedding model; use the embeddings endpoints (/v1/embeddings) instead of text generation.",
            ));
        }
        if options.thinking.enabled {
            let prepared_messages = self.apply_thinking_to_messages(messages, options)?;
            let mut generation = options.clone();
            generation.thinking.enabled = false;
            return self.generate_chat_stream(&prepared_messages, &generation, on_token);
        }

        let _guard = self
            .generation_lock
            .lock()
            .expect("generation lock poisoned");
        let mut on_token = on_token;
        let _backend_guard = self.scoped_backend_dispatch(options, RuntimePhase::Decode);

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

        let sliding_window = self.effective_sliding_window(options);
        let mut cache = KVCache::with_sliding_window(
            self.config.n_layers,
            kv_k_dim,
            kv_v_dim,
            cache_len,
            sliding_window,
        );
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
        let mut logits = Vec::with_capacity(self.config.vocab_size);
        self.prefill_prompt_tokens(&mut cache, &mut buf, &tokens, 0, &mut logits, options);
        let prefill_time = t_prefill.elapsed();

        let mut speculative = self.prepare_speculative_state(options, &tokens, cache_len);

        // Decode advances one sampled token at a time while reusing the cache.
        let t_decode = Instant::now();
        let mut output = String::with_capacity(options.max_tokens.saturating_mul(4));
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
                let window_start = trailing_char_boundary_start(&output, max_stop_len + text.len());
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

    fn prefill_prompt_tokens(
        &self,
        cache: &mut KVCache,
        buf: &mut DecodeBuffer,
        tokens: &[u32],
        start_pos: usize,
        logits: &mut Vec<f32>,
        options: &GenerationOptions,
    ) {
        if tokens.is_empty() {
            return;
        }
        let _backend_guard = self.scoped_backend_dispatch(options, RuntimePhase::Prefill);
        let _thread_guard =
            SimdThreadGuard::set_temporarily(self.effective_prefill_threads(options, tokens.len()));
        let last_idx = tokens.len() - 1;
        let ubatch = self.effective_prefill_ubatch(options, tokens.len());

        // Batched prefill runs every weight matrix once per chunk instead of
        // once per token (see model::forward_prefill_batch). It is gated on
        // Metal being fully disabled so macOS keeps its per-token fused-Metal
        // and GPU-resident prefill paths untouched; the batchability check is
        // hoisted out of the chunk loop so unsupported models skip straight
        // to the per-token fallback.
        #[cfg(not(target_family = "wasm"))]
        let mut batch_state = match &self.weights {
            LoadedWeights::Standard(weights)
                if !crate::metal::enabled()
                    && tokens.len() > 2
                    && batch_prefill_enabled()
                    && model::standard_prefill_batchable(weights) =>
            {
                Some((weights, model::PrefillBatchBuffer::new(&self.config)))
            }
            _ => None,
        };

        for (chunk_idx, chunk) in tokens[..last_idx].chunks(ubatch).enumerate() {
            let chunk_start = chunk_idx * ubatch;
            #[cfg(not(target_family = "wasm"))]
            if let Some((weights, batch_buf)) = batch_state.as_mut() {
                if chunk.len() > 1
                    && model::forward_prefill_batch(
                        &self.config,
                        weights,
                        cache,
                        batch_buf,
                        chunk,
                        start_pos + chunk_start,
                    )
                {
                    continue;
                }
            }
            for (idx, &tok_id) in chunk.iter().enumerate() {
                self.forward_prefill_token(cache, buf, tok_id, start_pos + chunk_start + idx);
            }
        }

        self.forward_token_into(cache, buf, tokens[last_idx], start_pos + last_idx, logits);
    }

    /// Runs the model-independent Thinking pre-pass over the latest user turn.
    fn apply_thinking_to_messages(
        &self,
        messages: &[ChatMessage],
        options: &GenerationOptions,
    ) -> Result<Vec<ChatMessage>, String> {
        if messages.is_empty() {
            return Err(String::from("No prompt provided."));
        }

        let Some(target_idx) = messages
            .iter()
            .rposition(|message| matches!(message.role, ChatRole::User))
        else {
            return Ok(messages.to_vec());
        };

        let original = messages[target_idx].content.trim();
        if original.is_empty() {
            return Ok(messages.to_vec());
        }

        let mut thinking_options = options.clone();
        thinking_options.thinking.enabled = false;
        thinking_options.max_tokens = options.thinking.max_tokens;
        thinking_options.sampler.temperature = 0.0;
        thinking_options.system_prompt = options.thinking.system_prompt.clone();
        thinking_options.stop_sequences.clear();
        thinking_options.speculative.enabled = false;

        let thinking_messages = [ChatMessage::user(original.to_string())];
        let rewritten = self.generate_chat(&thinking_messages, &thinking_options)?;
        let rewritten_prompt = clean_thinking_prompt(&rewritten.text);
        if rewritten_prompt.is_empty() {
            return Ok(messages.to_vec());
        }

        let mut prepared = messages.to_vec();
        prepared[target_idx].content = rewritten_prompt;
        Ok(prepared)
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
        let sliding_window = assistant.effective_sliding_window(options);
        let mut cache = KVCache::with_sliding_window(
            assistant.config.n_layers,
            kv_k_dim,
            kv_v_dim,
            assistant_cache_len,
            sliding_window,
        );
        let mut buf = DecodeBuffer::new(
            &assistant.config,
            max_head_dim,
            max_n_kv_heads,
            max_value_dim,
        );
        let mut logits = Vec::with_capacity(assistant.config.vocab_size);
        let t_draft = Instant::now();
        assistant.prefill_prompt_tokens(&mut cache, &mut buf, tokens, 0, &mut logits, options);

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
            let window_start = trailing_char_boundary_start(output, max_stop_len + text.len());
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
            LoadedWeights::NomicBert(_) => {
                unreachable!("nomic-bert is embedding-only; decode is gated before this call")
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
            LoadedWeights::NomicBert(_) => {
                unreachable!("nomic-bert uses forward_nomic_bert_hidden, not the decoder path")
            }
        }
    }

    /// Advances the model state for one prompt token without computing logits
    /// or a final normalized hidden state.
    fn forward_prefill_token(
        &self,
        cache: &mut KVCache,
        buf: &mut DecodeBuffer,
        token: u32,
        pos: usize,
    ) {
        match &self.weights {
            LoadedWeights::GptOss(weights) => {
                model::forward_prefill_gpt_oss(&self.config, weights, cache, buf, token, pos)
            }
            LoadedWeights::Gemma4(weights) => {
                model::forward_prefill_gemma4(&self.config, weights, cache, buf, token, pos)
            }
            LoadedWeights::Standard(weights) => {
                model::forward_prefill(&self.config, weights, cache, buf, token, pos)
            }
            LoadedWeights::NomicBert(_) => {
                unreachable!("nomic-bert is embedding-only; prefill is never invoked")
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
        let mut tokens = self.tok.encode(text);
        if tokens.is_empty() {
            return Err(String::from("embed: input tokenised to zero tokens"));
        }

        // nomic-bert / BERT encoders run a dedicated bidirectional forward with
        // no KV cache; mean-pool over all positions (incl. CLS/SEP), then
        // L2-normalise — matching llama.cpp's pooling_type=MEAN convention.
        if let LoadedWeights::NomicBert(weights) = &self.weights {
            if tokens.len() > self.config.max_seq_len {
                // Keep the trailing [SEP] when truncating an over-long input.
                let sep = *tokens.last().expect("non-empty tokens");
                tokens.truncate(self.config.max_seq_len);
                if let Some(last) = tokens.last_mut() {
                    *last = sep;
                }
            }
            let token_count = tokens.len();
            let dim = self.config.dim;
            let hs = model::forward_nomic_bert_hidden(&self.config, weights, &tokens);
            let mut sum = vec![0.0f32; dim];
            for i in 0..token_count {
                for j in 0..dim {
                    sum[j] += hs[i * dim + j];
                }
            }
            mean_pool_in_place(&mut sum, token_count);
            l2_normalize_in_place(&mut sum);
            return Ok(EmbeddingResult {
                embedding: sum,
                token_count,
            });
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
        } else if is_gemma_arch(&self.arch) {
            token == self.tok.eos_id
                || self.tok.special_id("<end_of_turn>") == Some(token)
                || self.tok.special_id("<turn|>") == Some(token)
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
        if matches!(self.chat_template_kind(), Some("gemma-turn")) {
            if let Some(tokens) = self.render_gemma_turn_messages(messages, system_prompt) {
                return tokens;
            }
        }
        if matches!(self.chat_template_kind(), Some("mistral3-inst")) {
            if let Some(tokens) = self.render_mistral3_inst_messages(messages, system_prompt) {
                return tokens;
            }
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

    /// Renders Gemma-style turn-delimited chat messages.
    fn render_gemma_turn_messages(
        &self,
        messages: &[ChatMessage],
        system_prompt: &str,
    ) -> Option<Vec<u32>> {
        let (start, end, system_as_turn) = if let (Some(start), Some(end)) = (
            self.tok.special_id("<start_of_turn>"),
            self.tok.special_id("<end_of_turn>"),
        ) {
            (start, end, false)
        } else if let (Some(start), Some(end)) = (
            self.tok.special_id("<|turn>"),
            self.tok.special_id("<turn|>"),
        ) {
            (start, end, true)
        } else {
            return None;
        };
        let mut tokens = Vec::new();
        if self.tok.adds_bos_token() {
            tokens.push(self.tok.bos_id);
        }

        let mut system_parts = Vec::new();
        if !system_prompt.trim().is_empty() {
            system_parts.push(system_prompt.trim().to_string());
        }
        for message in messages {
            if matches!(message.role, ChatRole::System) && !message.content.trim().is_empty() {
                system_parts.push(message.content.trim().to_string());
            }
        }
        let mut pending_system = system_parts.join("\n\n");
        let mut emitted_user = false;

        if system_as_turn && !pending_system.is_empty() {
            self.push_gemma_turn(start, end, "system", &pending_system, true, &mut tokens);
            pending_system.clear();
        }

        for message in messages {
            match message.role {
                ChatRole::System => {}
                ChatRole::User => {
                    let mut content = String::new();
                    if !system_as_turn && !pending_system.is_empty() {
                        content.push_str(&pending_system);
                        content.push_str("\n\n");
                        pending_system.clear();
                    }
                    content.push_str(message.content.trim());
                    self.push_gemma_turn(start, end, "user", &content, true, &mut tokens);
                    emitted_user = true;
                }
                ChatRole::Assistant => {
                    self.push_gemma_turn(
                        start,
                        end,
                        "model",
                        message.content.trim(),
                        true,
                        &mut tokens,
                    );
                }
            }
        }

        if !pending_system.is_empty() && !emitted_user {
            self.push_gemma_turn(start, end, "user", &pending_system, true, &mut tokens);
        }

        tokens.push(start);
        self.push_gemma_role("model", &mut tokens);
        tokens.extend(self.tok.encode_without_bos("\n"));
        Some(tokens)
    }

    /// Renders Mistral 3 / Ministral 3 instruction templates.
    ///
    /// Current Ministral 3 GGUFs expose a Jinja template of the form:
    /// BOS, optional `[SYSTEM_PROMPT]...[/SYSTEM_PROMPT]`, alternating
    /// `[INST]user[/INST]` turns, assistant text followed by EOS.
    fn render_mistral3_inst_messages(
        &self,
        messages: &[ChatMessage],
        system_prompt: &str,
    ) -> Option<Vec<u32>> {
        let inst = self.tok.special_id("[INST]")?;
        let end_inst = self.tok.special_id("[/INST]")?;
        let system_start = self.tok.special_id("[SYSTEM_PROMPT]")?;
        let system_end = self.tok.special_id("[/SYSTEM_PROMPT]")?;

        let mut tokens = Vec::new();
        tokens.push(self.tok.bos_id);

        let mut system_parts = Vec::new();
        if !system_prompt.trim().is_empty() {
            system_parts.push(system_prompt.trim().to_string());
        }
        for message in messages {
            if matches!(message.role, ChatRole::System) && !message.content.trim().is_empty() {
                system_parts.push(message.content.trim().to_string());
            }
        }
        let system = system_parts.join("\n\n");
        if !system.is_empty() {
            tokens.push(system_start);
            tokens.extend(self.tok.encode_without_bos(&system));
            tokens.push(system_end);
        }

        let mut last_role: Option<ChatRole> = None;
        for message in messages {
            match message.role {
                ChatRole::System => {}
                ChatRole::User => {
                    if matches!(last_role.as_ref(), Some(ChatRole::User)) {
                        tokens.extend(self.tok.encode_without_bos("\n"));
                    }
                    tokens.push(inst);
                    tokens.extend(self.tok.encode_without_bos(message.content.trim()));
                    tokens.push(end_inst);
                    last_role = Some(ChatRole::User);
                }
                ChatRole::Assistant => {
                    tokens.extend(self.tok.encode_without_bos(message.content.trim()));
                    tokens.push(self.tok.eos_id);
                    last_role = Some(ChatRole::Assistant);
                }
            }
        }

        Some(tokens)
    }

    fn push_gemma_role(&self, role: &str, out: &mut Vec<u32>) {
        if let Some(role_id) = self.tok.special_id(role) {
            out.push(role_id);
        } else {
            out.extend(self.tok.encode_without_bos(role));
        }
    }

    fn push_gemma_turn(
        &self,
        start: u32,
        end: u32,
        role: &str,
        content: &str,
        trailing_newline: bool,
        out: &mut Vec<u32>,
    ) {
        out.push(start);
        self.push_gemma_role(role, out);
        out.extend(self.tok.encode_without_bos("\n"));
        out.extend(self.tok.encode_without_bos(content));
        out.push(end);
        if trailing_newline {
            out.extend(self.tok.encode_without_bos("\n"));
        }
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
        Self::chat_template_kind_from_template(template)
    }

    fn chat_template_kind_from_template(template: &str) -> Option<&'static str> {
        if template.contains("<|start_header_id|>") && template.contains("<|eot_id|>") {
            Some("header-chat")
        } else if (template.contains("<start_of_turn>") && template.contains("<end_of_turn>"))
            || (template.contains("<|turn>") && template.contains("<turn|>"))
        {
            Some("gemma-turn")
        } else if template.contains("[SYSTEM_PROMPT]")
            && template.contains("[/SYSTEM_PROMPT]")
            && template.contains("[INST]")
            && template.contains("[/INST]")
        {
            Some("mistral3-inst")
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
        let sliding_window = (self.config.sliding_window > 0).then_some(self.config.sliding_window);
        let kv_cache = KVCache::with_sliding_window(
            self.config.n_layers,
            kv_k_dim,
            kv_v_dim,
            cap,
            sliding_window,
        );
        let decode_buf =
            DecodeBuffer::new(&self.config, max_head_dim, max_n_kv_heads, max_value_dim);
        crate::session::Session::new(kv_cache, decode_buf)
    }

    /// Generates chat text with session KV reuse and prompt-selected skills.
    #[cfg(all(not(target_family = "wasm"), feature = "server"))]
    pub fn generate_chat_with_session_with_skill_memory<F>(
        &self,
        messages: &[ChatMessage],
        options: &GenerationOptions,
        session: &mut crate::session::Session,
        on_token: F,
    ) -> Result<GenerationResult, String>
    where
        F: FnMut(&str),
    {
        let generation =
            self.options_with_skill_context(messages, options, &mut session.loaded_skill_paths)?;
        self.generate_chat_with_session(messages, &generation, session, on_token)
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
    pub fn generate_chat_with_session<F>(
        &self,
        messages: &[ChatMessage],
        options: &GenerationOptions,
        session: &mut crate::session::Session,
        on_token: F,
    ) -> Result<GenerationResult, String>
    where
        F: FnMut(&str),
    {
        options.validate()?;
        if matches!(self.weights, LoadedWeights::NomicBert(_)) {
            return Err(String::from(
                "nomic-bert is an encoder-only embedding model; use the embeddings endpoints (/v1/embeddings) instead of text generation.",
            ));
        }
        if options.thinking.enabled {
            let prepared_messages = self.apply_thinking_to_messages(messages, options)?;
            let mut generation = options.clone();
            generation.thinking.enabled = false;
            return self.generate_chat_with_session(
                &prepared_messages,
                &generation,
                session,
                on_token,
            );
        }

        let _guard = self
            .generation_lock
            .lock()
            .expect("generation lock poisoned");
        let mut on_token = on_token;

        if messages.is_empty() {
            return Err(String::from("No prompt provided."));
        }

        let total_start = Instant::now();
        session.last_used = total_start;

        let tokens = self.render_messages(messages, &options.system_prompt);
        if tokens.is_empty() {
            return Err(String::from("Prompt rendered to zero tokens."));
        }

        if session
            .kv_cache
            .set_sliding_window(self.effective_sliding_window(options))
        {
            session.reset();
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
        let mut logits = Vec::with_capacity(self.config.vocab_size);
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
            self.prefill_prompt_tokens(
                &mut session.kv_cache,
                &mut session.decode_buf,
                &tokens[prefix_len..],
                prefix_len,
                &mut logits,
                options,
            );
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
        let mut output = String::with_capacity(options.max_tokens.saturating_mul(4));
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
                let window_start = trailing_char_boundary_start(&output, max_stop_len + text.len());
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
    measure_kernel_raw(name, weight_dtype(weight), rows, cols, runs, || {
        weight.matvec_into(x, out);
        std::hint::black_box(out.len());
    })
}

/// Measures one routed expert matrix-vector kernel.
fn measure_expert_matvec(
    name: &str,
    weight: &ExpertWeight,
    expert: usize,
    x: &[f32],
    runs: usize,
    out: &mut Vec<f32>,
) -> KernelBenchRow {
    let expert = expert.min(weight.experts.saturating_sub(1));
    measure_kernel_raw(
        name,
        format!("{:?}", weight.dtype),
        weight.rows,
        weight.cols,
        runs,
        || {
            weight.matvec_expert_into(expert, x, out);
            std::hint::black_box(out.len());
        },
    )
}

/// Measures the fused gate/up projection used for one routed GPT-OSS expert.
fn measure_expert_matvec_pair(
    name: &str,
    gate: &ExpertWeight,
    up: &ExpertWeight,
    expert: usize,
    x: &[f32],
    runs: usize,
) -> Option<KernelBenchRow> {
    let expert = expert.min(gate.experts.saturating_sub(1));
    let mut gate_out = Vec::new();
    let mut up_out = Vec::new();
    if !gate.try_matvec_expert_pair_into(up, expert, x, &mut gate_out, &mut up_out) {
        return None;
    }
    Some(measure_kernel_raw(
        name,
        format!("{:?}+{:?}", gate.dtype, up.dtype),
        gate.rows + up.rows,
        gate.cols,
        runs,
        || {
            let ran = gate.try_matvec_expert_pair_into(up, expert, x, &mut gate_out, &mut up_out);
            std::hint::black_box(ran);
            std::hint::black_box(gate_out.len() + up_out.len());
        },
    ))
}

#[cfg(not(target_family = "wasm"))]
/// Measures a fused Q/K/V K-quant matvec when all three projections are eligible.
fn measure_fused_kquant3(
    name: &str,
    a: &Weight,
    b: &Weight,
    c: &Weight,
    x: &[f32],
    runs: usize,
) -> Option<KernelBenchRow> {
    let (
        Weight::Quantized {
            data: a_data,
            dtype: a_dtype,
            rows: a_rows,
            cols: a_cols,
        },
        Weight::Quantized {
            data: b_data,
            dtype: b_dtype,
            rows: b_rows,
            cols: b_cols,
        },
        Weight::Quantized {
            data: c_data,
            dtype: c_dtype,
            rows: c_rows,
            cols: c_cols,
        },
    ) = (a, b, c)
    else {
        return None;
    };
    let a_kind = kquant_matvec_kind(*a_dtype)?;
    let b_kind = kquant_matvec_kind(*b_dtype)?;
    let c_kind = kquant_matvec_kind(*c_dtype)?;
    if *a_cols != *b_cols || *a_cols != *c_cols || *a_cols != x.len() {
        return None;
    }

    let mut out_a = Vec::new();
    let mut out_b = Vec::new();
    let mut out_c = Vec::new();
    let dtype = format!("{:?}+{:?}+{:?}", a_dtype, b_dtype, c_dtype);
    Some(measure_kernel_raw(
        name,
        dtype,
        *a_rows + *b_rows + *c_rows,
        *a_cols,
        runs,
        || {
            let ran = crate::simd::matvec_kquant3_into(
                (a_kind, a_data.as_slice(), *a_rows, *a_cols),
                (b_kind, b_data.as_slice(), *b_rows, *b_cols),
                (c_kind, c_data.as_slice(), *c_rows, *c_cols),
                x,
                &mut out_a,
                &mut out_b,
                &mut out_c,
            );
            std::hint::black_box(ran);
            std::hint::black_box(out_a.len() + out_b.len() + out_c.len());
        },
    ))
}

#[cfg(target_family = "wasm")]
fn measure_fused_kquant3(
    _name: &str,
    _a: &Weight,
    _b: &Weight,
    _c: &Weight,
    _x: &[f32],
    _runs: usize,
) -> Option<KernelBenchRow> {
    None
}

#[cfg(not(target_family = "wasm"))]
/// Measures a fused gate/up K-quant matvec when both projections are eligible.
fn measure_fused_kquant2(
    name: &str,
    a: &Weight,
    b: &Weight,
    x: &[f32],
    runs: usize,
) -> Option<KernelBenchRow> {
    let (
        Weight::Quantized {
            data: a_data,
            dtype: a_dtype,
            rows: a_rows,
            cols: a_cols,
        },
        Weight::Quantized {
            data: b_data,
            dtype: b_dtype,
            rows: b_rows,
            cols: b_cols,
        },
    ) = (a, b)
    else {
        return None;
    };
    let a_kind = kquant_matvec_kind(*a_dtype)?;
    let b_kind = kquant_matvec_kind(*b_dtype)?;
    if *a_cols != *b_cols || *a_cols != x.len() {
        return None;
    }

    let mut out_a = Vec::new();
    let mut out_b = Vec::new();
    let mut out_c = Vec::new();
    let dtype = format!("{:?}+{:?}", a_dtype, b_dtype);
    Some(measure_kernel_raw(
        name,
        dtype,
        *a_rows + *b_rows,
        *a_cols,
        runs,
        || {
            let ran = crate::simd::matvec_kquant3_into(
                (a_kind, a_data.as_slice(), *a_rows, *a_cols),
                (b_kind, b_data.as_slice(), *b_rows, *b_cols),
                (b_kind, b_data.as_slice(), 0, *b_cols),
                x,
                &mut out_a,
                &mut out_b,
                &mut out_c,
            );
            std::hint::black_box(ran);
            std::hint::black_box(out_a.len() + out_b.len());
        },
    ))
}

#[cfg(target_family = "wasm")]
fn measure_fused_kquant2(
    _name: &str,
    _a: &Weight,
    _b: &Weight,
    _x: &[f32],
    _runs: usize,
) -> Option<KernelBenchRow> {
    None
}

#[cfg(not(target_family = "wasm"))]
/// Measures the fused Mistral FFN block when the Q4_K/Q4_K/Q6_K Metal path is available.
fn measure_mistral_ffn_block(
    name: &str,
    gate: &Weight,
    up: &Weight,
    down: &Weight,
    x: &[f32],
    runs: usize,
) -> Option<KernelBenchRow> {
    let (
        Weight::Quantized {
            data: gate_data,
            dtype: GGMLType::Q4_K,
            rows: gate_rows,
            cols: gate_cols,
        },
        Weight::Quantized {
            data: up_data,
            dtype: GGMLType::Q4_K,
            rows: up_rows,
            cols: up_cols,
        },
        Weight::Quantized {
            data: down_data,
            dtype: GGMLType::Q6_K,
            rows: down_rows,
            cols: down_cols,
        },
    ) = (gate, up, down)
    else {
        return None;
    };
    if *gate_cols != *up_cols
        || *gate_cols != x.len()
        || *gate_rows != *up_rows
        || *gate_rows != *down_cols
    {
        return None;
    }

    let mut out = Vec::new();
    if !crate::metal::q4k_q4k_q6k_ffn_into(
        (gate_data.as_slice(), *gate_rows, *gate_cols),
        (up_data.as_slice(), *up_rows, *up_cols),
        (down_data.as_slice(), *down_rows, *down_cols),
        x,
        &mut out,
    ) {
        return None;
    }
    Some(measure_kernel_raw(
        name,
        String::from("Q4_K+SwiGLU+Q6_K"),
        *down_rows,
        *gate_cols,
        runs,
        || {
            let ran = crate::metal::q4k_q4k_q6k_ffn_into(
                (gate_data.as_slice(), *gate_rows, *gate_cols),
                (up_data.as_slice(), *up_rows, *up_cols),
                (down_data.as_slice(), *down_rows, *down_cols),
                x,
                &mut out,
            );
            std::hint::black_box(ran);
            std::hint::black_box(out.len());
        },
    ))
}

#[cfg(target_family = "wasm")]
fn measure_mistral_ffn_block(
    _name: &str,
    _gate: &Weight,
    _up: &Weight,
    _down: &Weight,
    _x: &[f32],
    _runs: usize,
) -> Option<KernelBenchRow> {
    None
}

#[cfg(not(target_family = "wasm"))]
fn kquant_matvec_kind(dtype: GGMLType) -> Option<crate::simd::KQuantMatvecKind> {
    match dtype {
        GGMLType::Q4_K => Some(crate::simd::KQuantMatvecKind::Q4K),
        GGMLType::Q5_K => Some(crate::simd::KQuantMatvecKind::Q5K),
        GGMLType::Q6_K => Some(crate::simd::KQuantMatvecKind::Q6K),
        _ => None,
    }
}

/// Runs a benchmark closure repeatedly and summarizes timing.
fn measure_kernel<F>(
    name: &str,
    weight: &Weight,
    rows: usize,
    cols: usize,
    runs: usize,
    body: F,
) -> KernelBenchRow
where
    F: FnMut(),
{
    measure_kernel_raw(name, weight_dtype(weight), rows, cols, runs, body)
}

/// Runs a benchmark closure repeatedly and summarizes timing.
fn measure_kernel_raw<F>(
    name: &str,
    dtype: String,
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
        dtype,
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

/// Measures RMSNorm on a representative activation vector.
fn measure_rms_norm(
    name: &str,
    x: &[f32],
    weight: &[f32],
    eps: f32,
    runs: usize,
) -> KernelBenchRow {
    let mut out = Vec::new();
    measure_kernel_raw(name, String::from("F32"), x.len(), x.len(), runs, || {
        rms_norm_into(x, weight, eps, &mut out);
        std::hint::black_box(out.len());
    })
}

#[allow(clippy::too_many_arguments)]
/// Measures the Q/K rotary embedding transform.
fn measure_rope_qk(
    name: &str,
    q_len: usize,
    k_len: usize,
    head_dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
    inv_freq: &[f32],
    runs: usize,
) -> KernelBenchRow {
    let mut q = deterministic_bench_vector(q_len);
    let mut k = deterministic_bench_vector(k_len);
    measure_kernel_raw(
        name,
        String::from("F32"),
        q_len + k_len,
        head_dim,
        runs,
        || {
            apply_rope_qk(&mut q, &mut k, 32, head_dim, n_heads, n_kv_heads, inv_freq);
            std::hint::black_box(q.len() + k.len());
        },
    )
}

#[allow(clippy::too_many_arguments)]
/// Measures the per-layer attention scan against a synthetic cache.
fn measure_attention_scan(
    name: &str,
    n_heads: usize,
    n_kv_heads: usize,
    kv_mul: usize,
    head_dim: usize,
    value_dim: usize,
    ctx_len: usize,
    runs: usize,
) -> KernelBenchRow {
    let kv_k_dim = n_kv_heads * head_dim;
    let kv_v_dim = n_kv_heads * value_dim;
    let q = deterministic_bench_vector(n_heads * head_dim);
    let k = deterministic_bench_vector(ctx_len * kv_k_dim);
    let v = deterministic_bench_vector(ctx_len * kv_v_dim);
    let mut out = vec![0.0f32; n_heads * value_dim];
    let scale = 1.0 / (head_dim as f32).sqrt();
    measure_kernel_raw(name, String::from("F32"), ctx_len, n_heads, runs, || {
        for h in 0..n_heads {
            let kv_h = h / kv_mul.max(1);
            let q_off = h * head_dim;
            let out_off = h * value_dim;
            online_attention(
                &q[q_off..q_off + head_dim],
                &k[kv_h * head_dim..],
                &v[kv_h * value_dim..],
                kv_k_dim,
                kv_v_dim,
                ctx_len,
                head_dim,
                value_dim,
                0,
                ctx_len.saturating_sub(1),
                scale,
                &mut out[out_off..out_off + value_dim],
            );
        }
        std::hint::black_box(out.len());
    })
}

/// Measures the FFN activation path after the gate/up projections.
fn measure_ffn_activation(name: &str, hidden_dim: usize, runs: usize) -> KernelBenchRow {
    // Measures the production SwiGLU combine (crate::simd::silu_mul_into),
    // not a private scalar loop, so the row tracks what decode actually runs.
    let gate = deterministic_bench_vector(hidden_dim);
    let up = deterministic_bench_vector(hidden_dim);
    let mut hidden = Vec::with_capacity(hidden_dim);
    measure_kernel_raw(
        name,
        String::from("F32"),
        hidden_dim,
        hidden_dim,
        runs,
        || {
            crate::simd::silu_mul_into(&gate, &up, &mut hidden);
            std::hint::black_box(hidden.len());
        },
    )
}

/// Reconstructs the rotary inverse-frequency table used by the model.
fn deterministic_rope_inv_freq(
    theta: f32,
    head_dim: usize,
    scaling: f32,
    original_context_length: usize,
) -> Vec<f32> {
    let pair_count = head_dim / 2;
    let mut inv = vec![0.0f32; pair_count];
    if pair_count == 0 {
        return inv;
    }
    if scaling > 1.0 {
        let d_half = head_dim as f32 / 2.0;
        let low = d_half
            * ((original_context_length as f32 / (32.0 * 2.0 * std::f32::consts::PI)).ln()
                / theta.ln());
        let high = d_half
            * ((original_context_length as f32 / (1.0 * 2.0 * std::f32::consts::PI)).ln()
                / theta.ln());
        for (pair, slot) in inv.iter_mut().enumerate() {
            let i = (pair * 2) as f32;
            let base_freq = theta.powf(i / head_dim as f32);
            let idx = pair as f32;
            let ramp = ((idx - low) / (high - low)).clamp(0.0, 1.0);
            let mask = 1.0 - ramp;
            let interpolation = 1.0 / (scaling * base_freq);
            let extrapolation = 1.0 / base_freq;
            *slot = interpolation * (1.0 - mask) + extrapolation * mask;
        }
    } else {
        for (pair, slot) in inv.iter_mut().enumerate() {
            let i = (pair * 2) as f32;
            let base_freq = theta.powf(i / head_dim as f32);
            *slot = 1.0 / base_freq;
        }
    }
    inv
}

/// Builds a derived benchmark row from already-measured kernel timings.
fn estimated_kernel_row(name: &str, cols: usize, runs: usize, avg_ms: f64) -> KernelBenchRow {
    KernelBenchRow {
        name: name.to_string(),
        dtype: String::from("estimated"),
        rows: 0,
        cols,
        runs,
        avg_ms,
        total_ms: avg_ms * runs as f64,
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
        BackendPolicy, RECENT_TOKEN_LIMIT, RuntimeProfile, clean_thinking_prompt,
        cosine_similarity, l2_normalize_in_place, mean_pool_in_place, push_recent_token,
        recent_token_tail, trailing_char_boundary_start,
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
    /// Verifies that Thinking prompt cleanup removes common model wrappers.
    fn clean_thinking_prompt_removes_common_wrappers() {
        assert_eq!(
            clean_thinking_prompt("Prompt: Analyse Einsteins Relativitätstheorie."),
            "Analyse Einsteins Relativitätstheorie."
        );
        assert_eq!(
            clean_thinking_prompt("```text\nRewritten prompt: Compare TCP and UDP.\n```"),
            "Compare TCP and UDP."
        );
    }

    #[test]
    /// Verifies backend policy aliases used by the CLI/API.
    fn backend_policy_parses_runtime_aliases() {
        assert_eq!(BackendPolicy::parse("auto"), Some(BackendPolicy::Auto));
        assert_eq!(BackendPolicy::parse("cpu"), Some(BackendPolicy::Cpu));
        assert_eq!(BackendPolicy::parse("metal"), Some(BackendPolicy::Metal));
        assert_eq!(
            BackendPolicy::parse("metal_ultra"),
            Some(BackendPolicy::MetalUltra)
        );
        assert_eq!(BackendPolicy::parse("unknown"), None);
        assert_eq!(BackendPolicy::MetalUltra.as_str(), "metal-ultra");
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

    #[test]
    /// Verifies explicit Mistral ultra profile aliases parse to the aggressive profile.
    fn runtime_profile_parses_mistral_ultra_aliases() {
        for value in ["ultra", "mistral-ultra", "ministral_ultra"] {
            assert_eq!(
                RuntimeProfile::parse(value),
                Some(RuntimeProfile::MistralUltra)
            );
        }
        assert_eq!(RuntimeProfile::MistralUltra.as_str(), "mistral-ultra");
    }

    #[test]
    /// Verifies that Mistral 3 / Ministral 3 instruction templates use the native renderer.
    fn chat_template_kind_detects_mistral3_inst() {
        let template = "{%- if messages[0]['role'] == 'system' -%}\
            {{ '[SYSTEM_PROMPT]' }}{{ messages[0]['content'] }}{{ '[/SYSTEM_PROMPT]' }}\
            {%- endif -%}{{ '[INST]' + message['content'] + '[/INST]' }}";
        assert_eq!(
            super::Runner::chat_template_kind_from_template(template),
            Some("mistral3-inst")
        );
    }

    #[test]
    /// Verifies that Gemma and header-chat templates keep their established renderers.
    fn chat_template_kind_keeps_existing_renderers() {
        assert_eq!(
            super::Runner::chat_template_kind_from_template(
                "{{ '<start_of_turn>' + role + '<end_of_turn>' }}"
            ),
            Some("gemma-turn")
        );
        assert_eq!(
            super::Runner::chat_template_kind_from_template(
                "{{ '<|start_header_id|>' + role + '<|eot_id|>' }}"
            ),
            Some("header-chat")
        );
    }

    #[test]
    /// Verifies stop-sequence scan windows never split a multi-byte UTF-8 character.
    fn trailing_char_boundary_start_preserves_utf8_boundaries() {
        let text = " audngram \u{09AD}\u{09DF}\u{09BE}\u{09AC}\u{09B9}\u{09EF}";

        for max_bytes in 0..=text.len() + 4 {
            let start = trailing_char_boundary_start(text, max_bytes);
            assert!(text.is_char_boundary(start));
            let _ = &text[start..];
        }
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
