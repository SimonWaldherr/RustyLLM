#![allow(clippy::too_many_arguments)]

use crate::gguf::MetaValue;
use crate::runtime::{ChatMessage, ChatRole, GenerationOptions, Runner};
use crate::session::SessionStore;
#[cfg(feature = "tls")]
use rustls::pki_types::{CertificateDer, PrivateKeyDer};
#[cfg(feature = "tls")]
use rustls::{ServerConfig, ServerConnection, StreamOwned};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs;
#[cfg(feature = "tls")]
use std::fs::File;
#[cfg(feature = "tls")]
use std::io::BufReader;
use std::io::{self, BufRead, Read, Write};
use std::net::{Shutdown, TcpListener, TcpStream};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const MAX_HEADER_BYTES: usize = 16 * 1024;
const MAX_BODY_BYTES: usize = 2 * 1024 * 1024;
const IO_TIMEOUT: Duration = Duration::from_secs(30);

/// Default maximum number of concurrent sessions when `--max-sessions` is not specified.
pub const DEFAULT_MAX_SESSIONS: usize = 8;
/// Default per-session KV-cache capacity in tokens when `--max-cached-tokens` is not specified.
pub const DEFAULT_MAX_CACHED_TOKENS: usize = 2048;

#[derive(Clone)]
pub struct ServeOptions {
    pub addr: String,
    pub defaults: GenerationOptions,
    pub tls_cert_path: Option<String>,
    pub tls_key_path: Option<String>,
    pub max_concurrent_connections: usize,
    pub chat_ui: bool,
    pub model_catalog: Option<ModelCatalogSnapshot>,
    pub chat_history_path: Option<String>,
    pub chat_history_lock: Arc<Mutex<()>>,
    /// Session store for persistent KV-cache conversations.
    /// `None` when session caching is disabled (`--max-sessions 0`).
    pub session_store: Option<Arc<SessionStore>>,
}

#[derive(Clone, Debug)]
pub struct ModelCatalogSnapshot {
    pub model_dir: String,
    pub loaded_model_path: String,
    pub entries: Vec<ModelCatalogEntry>,
}

#[derive(Clone, Debug)]
pub struct ModelCatalogEntry {
    pub id: String,
    pub repository: String,
    pub file_name: String,
    pub path: String,
    pub size_bytes: u64,
    pub architecture: Option<String>,
    pub model_name: Option<String>,
    pub status: String,
    pub is_projector: bool,
    pub is_supported: bool,
}

impl From<crate::catalog::ModelEntry> for ModelCatalogEntry {
    fn from(entry: crate::catalog::ModelEntry) -> Self {
        let status = entry.status().to_string();
        let path = entry.path.display().to_string();
        Self {
            id: entry.id,
            repository: entry.repository,
            file_name: entry.file_name,
            path,
            size_bytes: entry.size_bytes,
            architecture: entry.architecture,
            model_name: entry.model_name,
            status,
            is_projector: entry.is_projector,
            is_supported: entry.is_supported,
        }
    }
}

impl ServeOptions {
    /// Reports whether both TLS certificate and key paths are configured.
    pub fn is_tls(&self) -> bool {
        #[cfg(feature = "tls")]
        {
            self.tls_cert_path.is_some() && self.tls_key_path.is_some()
        }
        #[cfg(not(feature = "tls"))]
        {
            false
        }
    }
}

// ─── Multimodal content types ─────────────────────────────────────────────────

/// A single part of a multimodal message content array.
#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ContentPart {
    Text { text: String },
    ImageUrl { image_url: ImageUrl },
}

#[derive(Deserialize)]
struct ImageUrl {
    url: String,
    #[allow(dead_code)]
    detail: Option<String>,
}

/// The `content` field of an API message can be either a plain string or an
/// array of content parts (OpenAI multimodal format).
#[derive(Deserialize)]
#[serde(untagged)]
enum ApiMessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

impl ApiMessageContent {
    /// Extract just the text, describing images with a placeholder.
    fn into_text(self) -> String {
        match self {
            ApiMessageContent::Text(s) => s,
            ApiMessageContent::Parts(parts) => {
                let mut out = String::new();
                for part in parts {
                    match part {
                        ContentPart::Text { text } => {
                            if !out.is_empty() {
                                out.push('\n');
                            }
                            out.push_str(&text);
                        }
                        ContentPart::ImageUrl { image_url } => {
                            if !out.is_empty() {
                                out.push('\n');
                            }
                            // Describe the image with its URL for context; a
                            // vision encoder would process it in a full
                            // multimodal pipeline.
                            let url = &image_url.url;
                            if url.starts_with("data:image/") {
                                out.push_str("[image: base64 data]");
                            } else {
                                out.push_str(&format!("[image: {}]", url));
                            }
                        }
                    }
                }
                out
            }
        }
    }
}

// ─── Stop sequence helper ─────────────────────────────────────────────────────

/// Accept `"stop"` as either a single string or an array of strings, matching
/// the OpenAI API contract.
#[derive(Deserialize)]
#[serde(untagged)]
enum StopSpec {
    One(String),
    Many(Vec<String>),
}

impl StopSpec {
    /// Normalizes a stop specification into a list of stop strings.
    fn into_vec(self) -> Vec<String> {
        match self {
            StopSpec::One(s) => vec![s],
            StopSpec::Many(v) => v,
        }
    }
}

// ─── Request types ────────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct GenerateRequest {
    prompt: Option<String>,
    messages: Option<Vec<ApiMessage>>,
    max_tokens: Option<usize>,
    temp: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<usize>,
    repeat_penalty: Option<f32>,
    seed: Option<u64>,
    system_prompt: Option<String>,
    thinking: Option<bool>,
    thinking_prompt: Option<String>,
    thinking_max_tokens: Option<usize>,
    stop: Option<StopSpec>,
    /// Optional conversation ID for persistent KV-cache sessions.
    conversation_id: Option<String>,
    /// When `true` (and `conversation_id` is set), use the persistent session.
    cache_prompt: Option<bool>,
}

#[derive(Deserialize)]
struct ApiMessage {
    role: String,
    content: ApiMessageContent,
}

#[derive(Deserialize)]
struct OpenAiCompletionsRequest {
    model: Option<String>,
    prompt: OpenAiPrompt,
    max_tokens: Option<usize>,
    /// Alias for max_tokens (OpenAI spec ≥ 2024-10).
    max_completion_tokens: Option<usize>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<usize>,
    repeat_penalty: Option<f32>,
    seed: Option<u64>,
    #[allow(dead_code)]
    stream: Option<bool>,
    system_prompt: Option<String>,
    thinking: Option<bool>,
    thinking_prompt: Option<String>,
    thinking_max_tokens: Option<usize>,
    stop: Option<StopSpec>,
    response_format: Option<serde_json::Value>,
    #[allow(dead_code)]
    tools: Option<Vec<serde_json::Value>>,
    #[allow(dead_code)]
    tool_choice: Option<serde_json::Value>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum OpenAiPrompt {
    Single(String),
    Batch(Vec<String>),
}

#[derive(Deserialize)]
struct OpenAiChatCompletionsRequest {
    model: Option<String>,
    messages: Vec<ApiMessage>,
    max_tokens: Option<usize>,
    /// Alias for max_tokens (OpenAI spec ≥ 2024-10).
    max_completion_tokens: Option<usize>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<usize>,
    repeat_penalty: Option<f32>,
    seed: Option<u64>,
    #[allow(dead_code)]
    stream: Option<bool>,
    system_prompt: Option<String>,
    thinking: Option<bool>,
    thinking_prompt: Option<String>,
    thinking_max_tokens: Option<usize>,
    stop: Option<StopSpec>,
    response_format: Option<serde_json::Value>,
    #[allow(dead_code)]
    tools: Option<Vec<serde_json::Value>>,
    #[allow(dead_code)]
    tool_choice: Option<serde_json::Value>,
    /// Optional conversation ID for persistent KV-cache sessions.
    conversation_id: Option<String>,
    /// When `true` (and `conversation_id` is set), use the persistent session.
    cache_prompt: Option<bool>,
}

#[derive(Deserialize)]
struct OpenAiResponsesRequest {
    model: Option<String>,
    input: Option<serde_json::Value>,
    instructions: Option<String>,
    max_output_tokens: Option<usize>,
    /// Compatibility alias accepted by some clients.
    max_completion_tokens: Option<usize>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<usize>,
    repeat_penalty: Option<f32>,
    seed: Option<u64>,
    thinking: Option<bool>,
    thinking_prompt: Option<String>,
    thinking_max_tokens: Option<usize>,
    #[allow(dead_code)]
    stream: Option<bool>,
    stop: Option<StopSpec>,
    text: Option<ResponsesTextConfig>,
    response_format: Option<serde_json::Value>,
    #[allow(dead_code)]
    tools: Option<Vec<serde_json::Value>>,
    #[allow(dead_code)]
    tool_choice: Option<serde_json::Value>,
    #[allow(dead_code)]
    previous_response_id: Option<String>,
    #[allow(dead_code)]
    conversation: Option<serde_json::Value>,
    #[allow(dead_code)]
    store: Option<bool>,
    #[allow(dead_code)]
    metadata: Option<serde_json::Value>,
}

#[derive(Deserialize)]
struct ResponsesTextConfig {
    format: Option<serde_json::Value>,
}

/// OpenAI-compatible `/v1/embeddings` request.
#[derive(Deserialize)]
struct EmbeddingsRequest {
    model: Option<String>,
    input: EmbeddingsInput,
    /// Ignored — always returns float embeddings.
    #[serde(default)]
    #[allow(dead_code)]
    encoding_format: Option<String>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum EmbeddingsInput {
    Single(String),
    Batch(Vec<String>),
}

#[derive(Deserialize)]
struct OllamaGenerateRequest {
    model: Option<String>,
    prompt: Option<String>,
    system: Option<String>,
    thinking: Option<bool>,
    thinking_prompt: Option<String>,
    thinking_max_tokens: Option<usize>,
    #[allow(dead_code)]
    stream: Option<bool>,
    options: Option<OllamaOptions>,
    stop: Option<StopSpec>,
}

#[derive(Deserialize)]
struct OllamaChatRequest {
    model: Option<String>,
    messages: Vec<OllamaMessage>,
    thinking: Option<bool>,
    thinking_prompt: Option<String>,
    thinking_max_tokens: Option<usize>,
    #[allow(dead_code)]
    stream: Option<bool>,
    options: Option<OllamaOptions>,
}

#[derive(Deserialize)]
struct OllamaMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct OllamaOptions {
    num_predict: Option<usize>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<usize>,
    repeat_penalty: Option<f32>,
    seed: Option<u64>,
    stop: Option<StopSpec>,
}

#[derive(Deserialize)]
struct OllamaEmbeddingRequest {
    model: Option<String>,
    prompt: Option<String>,
    input: Option<EmbeddingsInput>,
}

#[derive(Deserialize)]
struct ExplorerTokenizeRequest {
    input: String,
    add_bos: Option<bool>,
}

#[derive(Deserialize)]
struct ExplorerVectorRequest {
    input: Option<String>,
    token_id: Option<u32>,
}

#[derive(Deserialize)]
struct ExplorerNeighborsRequest {
    input: Option<String>,
    token_id: Option<u32>,
    limit: Option<usize>,
    include_special: Option<bool>,
}

// ─── Response types ───────────────────────────────────────────────────────────

/// Optional cache performance metrics included in responses when session
/// caching was active for the request.
#[derive(Serialize)]
struct CacheStats {
    cached_tokens: usize,
    evaluated_tokens: usize,
    prefill_ms: u128,
    decode_ms: u128,
}

#[derive(Serialize)]
struct GenerateResponse<'a> {
    text: &'a str,
    prompt_tokens: usize,
    generated_tokens: usize,
    prefill_ms: u128,
    decode_ms: u128,
    total_ms: u128,
    /// Present only when session caching was active.
    #[serde(skip_serializing_if = "Option::is_none")]
    cache_stats: Option<CacheStats>,
}

/// Build a [`CacheStats`] for a session-backed response.
///
/// Only returns `Some` when `use_session` is `true` AND a `conversation_id`
/// was provided; returns `None` for stateless requests so the field is
/// omitted from the JSON response.
fn make_cache_stats(
    stats: &crate::runtime::GenerationStats,
    use_session: bool,
    conv_id: Option<&str>,
) -> Option<CacheStats> {
    if use_session && conv_id.is_some() {
        Some(CacheStats {
            cached_tokens: stats.cached_tokens,
            evaluated_tokens: stats.prompt_tokens - stats.cached_tokens + stats.generated_tokens,
            prefill_ms: stats.prefill_time.as_millis(),
            decode_ms: stats.decode_time.as_millis(),
        })
    } else {
        None
    }
}

struct ActiveConnectionGuard {
    active_connections: Arc<AtomicUsize>,
}

impl Drop for ActiveConnectionGuard {
    /// Releases the resource represented by this guard or mapping.
    fn drop(&mut self) {
        self.active_connections.fetch_sub(1, Ordering::Release);
    }
}

#[derive(Debug)]
struct HttpError {
    status: u16,
    message: String,
}

impl HttpError {
    /// Creates an HTTP 400 parsing error.
    fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: 400,
            message: message.into(),
        }
    }

    /// Creates an HTTP 413 parsing error.
    fn payload_too_large(message: impl Into<String>) -> Self {
        Self {
            status: 413,
            message: message.into(),
        }
    }

    /// Creates an HTTP 408 parsing error.
    fn request_timeout(message: impl Into<String>) -> Self {
        Self {
            status: 408,
            message: message.into(),
        }
    }
}

#[derive(Serialize)]
struct OpenAiModelListResponse {
    object: &'static str,
    data: Vec<OpenAiModelInfo>,
}

#[derive(Serialize)]
struct OpenAiModelInfo {
    id: String,
    object: &'static str,
    created: u64,
    owned_by: &'static str,
}

#[derive(Serialize)]
struct OpenAiUsage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

#[derive(Serialize)]
struct OpenAiChatMessage {
    role: &'static str,
    content: String,
}

#[derive(Serialize)]
struct OpenAiChatChoice {
    index: usize,
    message: OpenAiChatMessage,
    finish_reason: &'static str,
}

#[derive(Serialize)]
struct OpenAiChatCompletionResponse {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<OpenAiChatChoice>,
    usage: OpenAiUsage,
    /// Present only when session caching was active.
    #[serde(skip_serializing_if = "Option::is_none")]
    cache_stats: Option<CacheStats>,
}

#[derive(Serialize)]
struct OpenAiCompletionChoice {
    text: String,
    index: usize,
    finish_reason: &'static str,
}

#[derive(Serialize)]
struct OpenAiCompletionResponse {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<OpenAiCompletionChoice>,
    usage: OpenAiUsage,
}

#[derive(Serialize)]
struct OpenAiResponsesUsage {
    input_tokens: usize,
    output_tokens: usize,
    total_tokens: usize,
}

#[derive(Serialize)]
struct OpenAiResponsesOutputText {
    r#type: &'static str,
    text: String,
    annotations: Vec<serde_json::Value>,
}

#[derive(Serialize)]
struct OpenAiResponsesOutputMessage {
    r#type: &'static str,
    id: String,
    status: &'static str,
    role: &'static str,
    content: Vec<OpenAiResponsesOutputText>,
}

#[derive(Serialize)]
struct OpenAiResponsesResponse {
    id: String,
    object: &'static str,
    created_at: u64,
    status: &'static str,
    model: String,
    output: Vec<OpenAiResponsesOutputMessage>,
    output_text: String,
    usage: OpenAiResponsesUsage,
    error: Option<serde_json::Value>,
    incomplete_details: Option<serde_json::Value>,
    parallel_tool_calls: bool,
}

#[derive(Clone)]
pub struct McpServeOptions {
    pub defaults: GenerationOptions,
    pub chat_history_path: Option<String>,
    pub chat_history_lock: Arc<Mutex<()>>,
    pub skill_memory: Arc<Mutex<HashSet<String>>>,
}

/// Starts the blocking HTTP or HTTPS server loop.
pub fn serve(runner: Arc<Runner>, options: ServeOptions) -> Result<(), String> {
    let listener = TcpListener::bind(&options.addr)
        .map_err(|err| format!("Failed to bind {}: {}", options.addr, err))?;
    let max_connections = options.max_concurrent_connections.max(1);
    let active_connections = Arc::new(AtomicUsize::new(0));

    // Keep the server loop deliberately small: accept a connection, hand it to
    // a worker thread, and let the handler own the request lifecycle.
    if options.is_tls() {
        #[cfg(not(feature = "tls"))]
        {
            return Err(String::from(
                "TLS serving requires a binary built with the `tls` feature.",
            ));
        }
        #[cfg(feature = "tls")]
        {
            let tls_config = Arc::new(load_tls_config(
                options.tls_cert_path.as_deref().unwrap_or(""),
                options.tls_key_path.as_deref().unwrap_or(""),
            )?);
            for stream in listener.incoming() {
                match stream {
                    Ok(stream) => {
                        let _ = stream.set_read_timeout(Some(IO_TIMEOUT));
                        let _ = stream.set_write_timeout(Some(IO_TIMEOUT));
                        let _ = stream.set_nodelay(true);
                        let Some(guard) = try_acquire_connection_slot(
                            Arc::clone(&active_connections),
                            max_connections,
                        ) else {
                            let _ = stream.shutdown(Shutdown::Both);
                            continue;
                        };

                        let runner = Arc::clone(&runner);
                        let options = options.clone();
                        let tls_config = Arc::clone(&tls_config);
                        thread::spawn(move || {
                            let _guard = guard;
                            if let Err(err) =
                                handle_tls_connection(stream, runner, &options, tls_config)
                            {
                                eprintln!("HTTPS connection error: {}", err);
                            }
                        });
                    }
                    Err(err) => eprintln!("Accept error: {}", err),
                }
            }
        }
    } else {
        for stream in listener.incoming() {
            match stream {
                Ok(mut stream) => {
                    let _ = stream.set_read_timeout(Some(IO_TIMEOUT));
                    let _ = stream.set_write_timeout(Some(IO_TIMEOUT));
                    let _ = stream.set_nodelay(true);
                    let Some(guard) = try_acquire_connection_slot(
                        Arc::clone(&active_connections),
                        max_connections,
                    ) else {
                        let body = json_error("Server overloaded: too many concurrent requests.");
                        let _ = write_http_response(&mut stream, 503, &body);
                        let _ = stream.shutdown(Shutdown::Both);
                        continue;
                    };

                    let runner = Arc::clone(&runner);
                    let options = options.clone();
                    thread::spawn(move || {
                        let _guard = guard;
                        if let Err(err) = handle_plain_connection(stream, runner, &options) {
                            eprintln!("HTTP connection error: {}", err);
                        }
                    });
                }
                Err(err) => eprintln!("Accept error: {}", err),
            }
        }
    }

    Ok(())
}

/// Runs a minimal MCP server over newline-delimited JSON-RPC on stdin/stdout.
pub fn serve_mcp_stdio(runner: Arc<Runner>, options: McpServeOptions) -> Result<(), String> {
    let stdin = io::stdin();
    let mut stdout = io::stdout();
    let reader = stdin.lock();
    let model_ids = advertised_model_ids(&runner);

    for line in reader.lines() {
        let line = line.map_err(|err| format!("Failed to read MCP message: {}", err))?;
        if line.trim().is_empty() {
            continue;
        }
        let request = match serde_json::from_str::<serde_json::Value>(&line) {
            Ok(value) => value,
            Err(err) => {
                let response = mcp_error(None, -32700, &format!("Parse error: {}", err));
                write_mcp_message(&mut stdout, &response)?;
                continue;
            }
        };
        let response = handle_mcp_request(&runner, &options, &model_ids, request);
        if let Some(response) = response {
            write_mcp_message(&mut stdout, &response)?;
        }
    }

    Ok(())
}

/// Handles one MCP JSON-RPC request. Notifications return `None`.
fn handle_mcp_request(
    runner: &Runner,
    options: &McpServeOptions,
    model_ids: &[String],
    request: serde_json::Value,
) -> Option<serde_json::Value> {
    let id = request.get("id").cloned();
    let method = match request.get("method").and_then(|v| v.as_str()) {
        Some(method) => method,
        None => {
            return id.map(|id| mcp_error(Some(id), -32600, "Invalid Request: missing method"));
        }
    };

    let id = id?;
    let params = request
        .get("params")
        .cloned()
        .unwrap_or(serde_json::Value::Null);
    let result = match method {
        "initialize" => Ok(mcp_initialize_result(&params)),
        "ping" => Ok(serde_json::json!({})),
        "tools/list" => Ok(mcp_tools_list()),
        "tools/call" => mcp_tools_call(runner, options, model_ids, &params),
        "resources/list" => Ok(serde_json::json!({ "resources": [] })),
        "prompts/list" => Ok(serde_json::json!({ "prompts": [] })),
        other => Err((-32601, format!("Method not found: {}", other))),
    };

    Some(match result {
        Ok(result) => serde_json::json!({
            "jsonrpc": "2.0",
            "id": id,
            "result": result,
        }),
        Err((code, message)) => mcp_error(Some(id), code, &message),
    })
}

/// Builds the MCP initialize result.
fn mcp_initialize_result(params: &serde_json::Value) -> serde_json::Value {
    let protocol_version = params
        .get("protocolVersion")
        .and_then(|v| v.as_str())
        .unwrap_or("2025-06-18");
    serde_json::json!({
        "protocolVersion": protocol_version,
        "serverInfo": {
            "name": "rusty-llm",
            "version": env!("CARGO_PKG_VERSION"),
        },
        "capabilities": {
            "tools": { "listChanged": false },
        },
    })
}

/// Returns the MCP tool inventory.
fn mcp_tools_list() -> serde_json::Value {
    serde_json::json!({
        "tools": [
            {
                "name": "generate",
                "description": "Generate a completion from a plain prompt with the loaded local GGUF model.",
                "inputSchema": {
                    "type": "object",
                    "required": ["prompt"],
                    "properties": {
                        "prompt": {"type": "string"},
                        "max_tokens": {"type": "integer", "minimum": 1},
                        "temperature": {"type": "number", "minimum": 0},
                        "top_p": {"type": "number"},
                        "top_k": {"type": "integer"},
                        "repeat_penalty": {"type": "number"},
                        "seed": {"type": "integer"},
                        "system_prompt": {"type": "string"},
                        "thinking": {"type": "boolean"},
                        "thinking_prompt": {"type": "string"},
                        "thinking_max_tokens": {"type": "integer", "minimum": 1},
                        "stop": {"oneOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}}]}
                    }
                }
            },
            {
                "name": "chat",
                "description": "Generate a chat response from system, user, and assistant messages.",
                "inputSchema": {
                    "type": "object",
                    "required": ["messages"],
                    "properties": {
                        "messages": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "required": ["role", "content"],
                                "properties": {
                                    "role": {"type": "string", "enum": ["system", "user", "assistant"]},
                                    "content": {"type": "string"}
                                }
                            }
                        },
                        "max_tokens": {"type": "integer", "minimum": 1},
                        "temperature": {"type": "number", "minimum": 0},
                        "top_p": {"type": "number"},
                        "top_k": {"type": "integer"},
                        "repeat_penalty": {"type": "number"},
                        "seed": {"type": "integer"},
                        "system_prompt": {"type": "string"},
                        "thinking": {"type": "boolean"},
                        "thinking_prompt": {"type": "string"},
                        "thinking_max_tokens": {"type": "integer", "minimum": 1},
                        "stop": {"oneOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}}]}
                    }
                }
            },
            {
                "name": "embed",
                "description": "Return an L2-normalized embedding for text using the loaded model.",
                "inputSchema": {
                    "type": "object",
                    "required": ["input"],
                    "properties": {
                        "input": {"type": "string"}
                    }
                }
            },
            {
                "name": "models",
                "description": "Return metadata for the loaded model and advertised compatibility aliases.",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            }
        ]
    })
}

/// Dispatches an MCP tools/call request.
fn mcp_tools_call(
    runner: &Runner,
    options: &McpServeOptions,
    model_ids: &[String],
    params: &serde_json::Value,
) -> Result<serde_json::Value, (i64, String)> {
    let name = params
        .get("name")
        .and_then(|v| v.as_str())
        .ok_or_else(|| (-32602, String::from("tools/call requires params.name")))?;
    let args = params
        .get("arguments")
        .cloned()
        .unwrap_or_else(|| serde_json::json!({}));

    match name {
        "generate" => mcp_tool_generate(runner, options, &args),
        "chat" => mcp_tool_chat(runner, options, &args),
        "embed" => mcp_tool_embed(runner, &args),
        "models" => Ok(mcp_tool_result_structured(
            serde_json::json!({
                "model_ids": model_ids,
                "model_name": runner.model_name(),
                "architecture": runner.architecture(),
                "format": "gguf",
                "quantization": dominant_quantization(runner),
                "context_length": runner.config().max_seq_len,
                "vocab_size": runner.tokenizer().vocab_size(),
            }),
            "Loaded RustyLLM model metadata.",
        )),
        other => Err((-32602, format!("Unknown tool: {}", other))),
    }
}

/// MCP generate tool implementation.
fn mcp_tool_generate(
    runner: &Runner,
    options: &McpServeOptions,
    args: &serde_json::Value,
) -> Result<serde_json::Value, (i64, String)> {
    let prompt = mcp_arg_string(args, "prompt")
        .ok_or_else(|| (-32602, String::from("generate requires a prompt string")))?;
    let generation = mcp_generation_options(&options.defaults, args);
    let mut skill_memory = options
        .skill_memory
        .lock()
        .expect("MCP skill memory lock poisoned");
    match runner.generate_with_skill_memory(&prompt, &generation, &mut skill_memory) {
        Ok(result) => {
            let _ = append_mcp_history(
                options,
                "mcp.generate",
                &[ChatMessage::user(prompt)],
                &result,
            );
            Ok(mcp_text_result(&result.text))
        }
        Err(err) => Ok(mcp_tool_error(&err)),
    }
}

/// MCP chat tool implementation.
fn mcp_tool_chat(
    runner: &Runner,
    options: &McpServeOptions,
    args: &serde_json::Value,
) -> Result<serde_json::Value, (i64, String)> {
    let messages = args
        .get("messages")
        .and_then(|v| v.as_array())
        .ok_or_else(|| (-32602, String::from("chat requires a messages array")))?;
    let messages = parse_mcp_messages(messages)?;
    let generation = mcp_generation_options(&options.defaults, args);
    let mut skill_memory = options
        .skill_memory
        .lock()
        .expect("MCP skill memory lock poisoned");
    match runner.generate_chat_with_skill_memory(&messages, &generation, &mut skill_memory) {
        Ok(result) => {
            let _ = append_mcp_history(options, "mcp.chat", &messages, &result);
            Ok(mcp_text_result(&result.text))
        }
        Err(err) => Ok(mcp_tool_error(&err)),
    }
}

/// MCP embed tool implementation.
fn mcp_tool_embed(
    runner: &Runner,
    args: &serde_json::Value,
) -> Result<serde_json::Value, (i64, String)> {
    let input = mcp_arg_string(args, "input")
        .or_else(|| mcp_arg_string(args, "text"))
        .ok_or_else(|| (-32602, String::from("embed requires an input string")))?;
    match runner.embed(&input) {
        Ok(result) => Ok(mcp_tool_result_structured(
            serde_json::json!({
                "embedding": result.embedding,
                "token_count": result.token_count,
            }),
            "Embedding generated.",
        )),
        Err(err) => Ok(mcp_tool_error(&err)),
    }
}

/// Converts MCP message JSON into runtime messages.
fn parse_mcp_messages(items: &[serde_json::Value]) -> Result<Vec<ChatMessage>, (i64, String)> {
    items
        .iter()
        .map(|item| {
            let role = item
                .get("role")
                .and_then(|v| v.as_str())
                .ok_or_else(|| (-32602, String::from("message.role must be a string")))?;
            let content = item
                .get("content")
                .and_then(|v| v.as_str())
                .ok_or_else(|| (-32602, String::from("message.content must be a string")))?;
            match role {
                "system" => Ok(ChatMessage {
                    role: ChatRole::System,
                    content: content.to_string(),
                }),
                "user" => Ok(ChatMessage::user(content.to_string())),
                "assistant" => Ok(ChatMessage::assistant(content.to_string())),
                other => Err((-32602, format!("Unsupported role: {}", other))),
            }
        })
        .collect()
}

/// Applies MCP generation overrides.
fn mcp_generation_options(
    defaults: &GenerationOptions,
    args: &serde_json::Value,
) -> GenerationOptions {
    let max_tokens =
        mcp_arg_usize(args, "max_tokens").or_else(|| mcp_arg_usize(args, "max_output_tokens"));
    let temperature = mcp_arg_f32(args, "temperature").or_else(|| mcp_arg_f32(args, "temp"));
    let stop = args.get("stop").and_then(stop_value_to_vec);
    apply_generation_overrides(
        defaults,
        max_tokens,
        temperature,
        mcp_arg_f32(args, "top_p"),
        mcp_arg_usize(args, "top_k"),
        mcp_arg_f32(args, "repeat_penalty"),
        mcp_arg_u64(args, "seed"),
        mcp_arg_string(args, "system_prompt").or_else(|| mcp_arg_string(args, "instructions")),
        stop,
        mcp_arg_bool(args, "thinking"),
        mcp_arg_string(args, "thinking_prompt")
            .or_else(|| mcp_arg_string(args, "thinking_system_prompt")),
        mcp_arg_usize(args, "thinking_max_tokens"),
    )
}

/// Extracts a string argument from a JSON object.
fn mcp_arg_string(args: &serde_json::Value, name: &str) -> Option<String> {
    args.get(name)
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
}

/// Extracts a usize argument from a JSON object.
fn mcp_arg_usize(args: &serde_json::Value, name: &str) -> Option<usize> {
    args.get(name)
        .and_then(|v| v.as_u64())
        .and_then(|v| usize::try_from(v).ok())
}

/// Extracts a u64 argument from a JSON object.
fn mcp_arg_u64(args: &serde_json::Value, name: &str) -> Option<u64> {
    args.get(name).and_then(|v| v.as_u64())
}

/// Extracts a bool argument from a JSON object.
fn mcp_arg_bool(args: &serde_json::Value, name: &str) -> Option<bool> {
    args.get(name).and_then(|v| v.as_bool())
}

/// Extracts an f32 argument from a JSON object.
fn mcp_arg_f32(args: &serde_json::Value, name: &str) -> Option<f32> {
    args.get(name).and_then(|v| v.as_f64()).map(|v| v as f32)
}

/// Normalizes a JSON stop value.
fn stop_value_to_vec(value: &serde_json::Value) -> Option<Vec<String>> {
    match value {
        serde_json::Value::String(text) => Some(vec![text.clone()]),
        serde_json::Value::Array(items) => Some(
            items
                .iter()
                .filter_map(|item| item.as_str().map(|s| s.to_string()))
                .collect(),
        ),
        _ => None,
    }
}

/// Builds a successful text-only MCP tool result.
fn mcp_text_result(text: &str) -> serde_json::Value {
    serde_json::json!({
        "content": [{"type": "text", "text": text}],
    })
}

/// Builds a successful MCP tool result with structured content.
fn mcp_tool_result_structured(data: serde_json::Value, summary: &str) -> serde_json::Value {
    serde_json::json!({
        "content": [{"type": "text", "text": summary}],
        "structuredContent": data,
    })
}

/// Builds a tool-level error result.
fn mcp_tool_error(message: &str) -> serde_json::Value {
    serde_json::json!({
        "content": [{"type": "text", "text": message}],
        "isError": true,
    })
}

/// Builds a JSON-RPC error response.
fn mcp_error(id: Option<serde_json::Value>, code: i64, message: &str) -> serde_json::Value {
    serde_json::json!({
        "jsonrpc": "2.0",
        "id": id.unwrap_or(serde_json::Value::Null),
        "error": {
            "code": code,
            "message": message,
        }
    })
}

/// Writes one MCP JSON-RPC message to stdout.
fn write_mcp_message<W: Write>(writer: &mut W, value: &serde_json::Value) -> Result<(), String> {
    serde_json::to_writer(&mut *writer, value)
        .map_err(|err| format!("Failed to serialize MCP response: {}", err))?;
    writeln!(writer).map_err(|err| format!("Failed to write MCP newline: {}", err))?;
    writer
        .flush()
        .map_err(|err| format!("Failed to flush MCP response: {}", err))
}

/// Appends MCP tool calls to the same optional chat history log as HTTP routes.
fn append_mcp_history(
    options: &McpServeOptions,
    source: &str,
    messages: &[ChatMessage],
    result: &crate::runtime::GenerationResult,
) -> Result<(), String> {
    let Some(path) = options.chat_history_path.as_deref() else {
        return Ok(());
    };
    append_chat_history_inner(
        Some(path),
        &options.chat_history_lock,
        source,
        "mcp",
        messages,
        result,
    )
}

/// Attempts to reserve capacity for one accepted connection.
fn try_acquire_connection_slot(
    active_connections: Arc<AtomicUsize>,
    max_connections: usize,
) -> Option<ActiveConnectionGuard> {
    loop {
        let current = active_connections.load(Ordering::Acquire);
        if current >= max_connections {
            return None;
        }
        if active_connections
            .compare_exchange(current, current + 1, Ordering::AcqRel, Ordering::Relaxed)
            .is_ok()
        {
            return Some(ActiveConnectionGuard { active_connections });
        }
    }
}

/// Handles one accepted TCP connection without TLS.
fn handle_plain_connection(
    stream: TcpStream,
    runner: Arc<Runner>,
    options: &ServeOptions,
) -> Result<(), String> {
    handle_connection(stream, runner, options)
}

#[cfg(feature = "tls")]
/// Wraps one TCP connection in TLS and handles the request.
fn handle_tls_connection(
    stream: TcpStream,
    runner: Arc<Runner>,
    options: &ServeOptions,
    tls_config: Arc<ServerConfig>,
) -> Result<(), String> {
    let conn = ServerConnection::new(tls_config).map_err(|err| err.to_string())?;
    let tls_stream = StreamOwned::new(conn, stream);
    handle_connection(tls_stream, runner, options)
}

/// Reads, routes, and responds to one HTTP request.
fn handle_connection<T>(
    mut stream: T,
    runner: Arc<Runner>,
    options: &ServeOptions,
) -> Result<(), String>
where
    T: Read + Write,
{
    let request = match read_http_request(&mut stream) {
        Ok(request) => request,
        Err(err) => {
            let body = json_error(&err.message);
            let _ = write_http_response(&mut stream, err.status, &body);
            return Ok(());
        }
    };

    // Streaming requests need direct access to the underlying write stream;
    // route them before falling through to the standard (status, body) path.
    if is_streaming_request(&request) {
        return route_streaming_request(&request, &mut stream, &runner, options)
            .map_err(|err| err.to_string());
    }

    if request.method == "GET" && swagger_ui_route(&request.path) {
        return write_http_response_with_content_type(
            &mut stream,
            200,
            "text/html; charset=utf-8",
            swagger_ui_html(),
        )
        .map_err(|err| err.to_string());
    }

    if request.method == "GET" && chat_ui_route(&request.path).is_some() {
        if options.chat_ui {
            let body = match chat_ui_route(&request.path) {
                Some(ChatUiRoute::Simple) => chat_ui_html(),
                Some(ChatUiRoute::Expert) => expert_chat_ui_html(),
                Some(ChatUiRoute::Explorer) => explorer_ui_html(),
                None => unreachable!(),
            };
            return write_http_response_with_content_type(
                &mut stream,
                200,
                "text/html; charset=utf-8",
                body,
            )
            .map_err(|err| err.to_string());
        }

        let body = json_error("Not found");
        return write_http_response(&mut stream, 404, &body).map_err(|err| err.to_string());
    }

    if request.method == "GET" && options.chat_ui {
        match request.path.as_str() {
            "/style.css" => {
                return write_http_response_with_content_type(
                    &mut stream,
                    200,
                    "text/css; charset=utf-8",
                    include_str!("web_ui/style.css"),
                )
                .map_err(|err| err.to_string());
            }
            "/script.js" => {
                return write_http_response_with_content_type(
                    &mut stream,
                    200,
                    "text/javascript; charset=utf-8",
                    include_str!("web_ui/script.js"),
                )
                .map_err(|err| err.to_string());
            }
            "/explorer.js" => {
                return write_http_response_with_content_type(
                    &mut stream,
                    200,
                    "text/javascript; charset=utf-8",
                    include_str!("web_ui/explorer.js"),
                )
                .map_err(|err| err.to_string());
            }
            _ => {}
        }
    }

    let (status, body) = route_request(&request, &runner, options);
    write_http_response(&mut stream, status, &body).map_err(|err| err.to_string())
}

enum ChatUiRoute {
    Simple,
    Expert,
    Explorer,
}

/// Returns true for the embedded Swagger UI route.
fn swagger_ui_route(path: &str) -> bool {
    matches!(path, "/docs" | "/swagger" | "/swagger-ui")
}

/// Maps a request path to one of the embedded chat UI assets.
fn chat_ui_route(path: &str) -> Option<ChatUiRoute> {
    match path {
        "/chat" => Some(ChatUiRoute::Simple),
        "/chat?expert" => Some(ChatUiRoute::Expert),
        "/explorer" | "/chat?explorer" => Some(ChatUiRoute::Explorer),
        _ => None,
    }
}

/// Returns true when the request body asks for SSE streaming.
fn is_streaming_request(request: &HttpRequest) -> bool {
    let method = request.method.as_str();
    let path = request.path.as_str();
    if method != "POST" {
        return false;
    }
    if !matches!(
        path,
        "/v1/chat/completions"
            | "/v1/completions"
            | "/v1/responses"
            | "/api/v0/chat/completions"
            | "/api/v0/completions"
            | "/api/generate"
            | "/api/chat"
    ) {
        return false;
    }
    // Quick JSON field scan — avoid full deserialisation just for the flag.
    if let Ok(v) = serde_json::from_slice::<serde_json::Value>(&request.body) {
        return v.get("stream").and_then(|s| s.as_bool()).unwrap_or(false);
    }
    false
}

/// Write the streaming SSE response directly to `stream`.
fn route_streaming_request<W: Write>(
    request: &HttpRequest,
    stream: &mut W,
    runner: &Runner,
    options: &ServeOptions,
) -> io::Result<()> {
    if request.path == "/v1/responses" {
        return route_openai_response_stream(request, stream, runner, options);
    }
    if request.path == "/api/generate" {
        return route_ollama_generate_stream(request, stream, runner, options);
    }
    if request.path == "/api/chat" {
        return route_ollama_chat_stream(request, stream, runner, options);
    }

    let model_ids = advertised_model_ids(runner);
    let created = unix_timestamp();
    let comp_id = format!(
        "{}rustyllm-{}",
        if request.path == "/v1/chat/completions" || request.path == "/api/v0/chat/completions" {
            "chatcmpl-"
        } else {
            "cmpl-"
        },
        created
    );
    let is_chat =
        request.path == "/v1/chat/completions" || request.path == "/api/v0/chat/completions";
    let model_name = model_ids
        .first()
        .cloned()
        .unwrap_or_else(|| runner.architecture().to_string());

    let (messages, generation, use_session, conv_id) = if is_chat {
        match serde_json::from_slice::<OpenAiChatCompletionsRequest>(&request.body) {
            Ok(payload) => {
                let messages = match parse_api_messages(payload.messages) {
                    Ok(m) => m,
                    Err(err) => {
                        write_http_response(stream, 400, &json_error(&err))?;
                        return Ok(());
                    }
                };
                let max_tok = payload.max_completion_tokens.or(payload.max_tokens);
                let mut generation_options = apply_generation_overrides(
                    &options.defaults,
                    max_tok,
                    payload.temperature,
                    payload.top_p,
                    payload.top_k,
                    payload.repeat_penalty,
                    payload.seed,
                    payload.system_prompt,
                    payload.stop.map(|s| s.into_vec()),
                    payload.thinking,
                    payload.thinking_prompt,
                    payload.thinking_max_tokens,
                );
                apply_response_format_hint(
                    &mut generation_options,
                    payload.response_format.as_ref(),
                );
                let use_session = payload.cache_prompt.unwrap_or(false);
                let conv_id = payload.conversation_id;
                (messages, generation_options, use_session, conv_id)
            }
            Err(err) => {
                write_http_response(stream, 400, &json_error(&format!("Invalid JSON: {}", err)))?;
                return Ok(());
            }
        }
    } else {
        match serde_json::from_slice::<OpenAiCompletionsRequest>(&request.body) {
            Ok(payload) => {
                let prompt = match payload.prompt {
                    OpenAiPrompt::Single(p) => p,
                    OpenAiPrompt::Batch(mut v) => {
                        if v.is_empty() {
                            write_http_response(
                                stream,
                                400,
                                &json_error("Prompt array must not be empty."),
                            )?;
                            return Ok(());
                        }
                        v.remove(0)
                    }
                };
                let max_tok = payload.max_completion_tokens.or(payload.max_tokens);
                let mut generation_options = apply_generation_overrides(
                    &options.defaults,
                    max_tok,
                    payload.temperature,
                    payload.top_p,
                    payload.top_k,
                    payload.repeat_penalty,
                    payload.seed,
                    payload.system_prompt,
                    payload.stop.map(|s| s.into_vec()),
                    payload.thinking,
                    payload.thinking_prompt,
                    payload.thinking_max_tokens,
                );
                apply_response_format_hint(
                    &mut generation_options,
                    payload.response_format.as_ref(),
                );
                (
                    vec![ChatMessage::user(prompt)],
                    generation_options,
                    false,
                    None,
                )
            }
            Err(err) => {
                write_http_response(stream, 400, &json_error(&format!("Invalid JSON: {}", err)))?;
                return Ok(());
            }
        }
    };

    // Write SSE response headers — no Content-Length since length is unknown.
    write!(
        stream,
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\nAccess-Control-Allow-Origin: *\r\n\r\n"
    )?;

    // Stream each token as a Server-Sent Event chunk.
    let result = generate_with_optional_session(
        runner,
        options,
        &messages,
        &generation,
        use_session,
        conv_id.as_deref(),
        |text| {
            let chunk = if is_chat {
                serde_json::json!({
                    "id": comp_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": text},
                        "finish_reason": null
                    }]
                })
            } else {
                serde_json::json!({
                    "id": comp_id,
                    "object": "text_completion",
                    "created": created,
                    "model": model_name,
                    "choices": [{
                        "index": 0,
                        "text": text,
                        "finish_reason": null
                    }]
                })
            };
            let line = format!("data: {}\n\n", chunk);
            let _ = stream.write_all(line.as_bytes());
        },
    );
    if let Ok(result) = &result {
        let _ = append_chat_history(options, "openai.stream", &model_name, &messages, result);
    }

    // Send final chunk with finish_reason and [DONE].
    let finish_reason = if result.is_ok() { "stop" } else { "error" };
    let final_chunk = if is_chat {
        serde_json::json!({
            "id": comp_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}]
        })
    } else {
        serde_json::json!({
            "id": comp_id,
            "object": "text_completion",
            "created": created,
            "model": model_name,
            "choices": [{"index": 0, "text": "", "finish_reason": finish_reason}]
        })
    };
    write!(stream, "data: {}\n\n", final_chunk)?;
    write!(stream, "data: [DONE]\n\n")?;
    stream.flush()
}

/// Writes an OpenAI Responses API SSE stream.
fn route_openai_response_stream<W: Write>(
    request: &HttpRequest,
    stream: &mut W,
    runner: &Runner,
    options: &ServeOptions,
) -> io::Result<()> {
    let model_ids = advertised_model_ids(runner);
    let payload = match serde_json::from_slice::<OpenAiResponsesRequest>(&request.body) {
        Ok(payload) => payload,
        Err(err) => {
            write_http_response(stream, 400, &json_error(&format!("Invalid JSON: {}", err)))?;
            return Ok(());
        }
    };
    let model = resolve_model(payload.model.as_deref(), &model_ids);
    let messages = match parse_responses_input(payload.input.as_ref()) {
        Ok(messages) => messages,
        Err(err) => {
            write_http_response(stream, 400, &json_error(&err))?;
            return Ok(());
        }
    };
    let max_tokens = payload.max_output_tokens.or(payload.max_completion_tokens);
    let mut generation = apply_generation_overrides(
        &options.defaults,
        max_tokens,
        payload.temperature,
        payload.top_p,
        payload.top_k,
        payload.repeat_penalty,
        payload.seed,
        payload.instructions,
        payload.stop.map(|s| s.into_vec()),
        payload.thinking,
        payload.thinking_prompt,
        payload.thinking_max_tokens,
    );
    let format_hint = payload
        .text
        .as_ref()
        .and_then(|text| text.format.as_ref())
        .or(payload.response_format.as_ref());
    apply_response_format_hint(&mut generation, format_hint);

    let created = unix_timestamp();
    let response_id = format!("resp-rustyllm-{}", created);
    let message_id = format!("msg-rustyllm-{}", created);
    write!(
        stream,
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\nAccess-Control-Allow-Origin: *\r\n\r\n"
    )?;
    let created_event = serde_json::json!({
        "type": "response.created",
        "response": {
            "id": response_id,
            "object": "response",
            "created_at": created,
            "status": "in_progress",
            "model": model,
        }
    });
    write_sse_event(stream, "response.created", &created_event)?;

    let result = generate_with_optional_session(
        runner,
        options,
        &messages,
        &generation,
        false,
        None,
        |text| {
            let chunk = serde_json::json!({
                "type": "response.output_text.delta",
                "response_id": response_id,
                "item_id": message_id,
                "output_index": 0,
                "content_index": 0,
                "delta": text,
            });
            let _ = write_sse_event(stream, "response.output_text.delta", &chunk);
        },
    );

    match result {
        Ok(result) => {
            let _ = append_chat_history(
                options,
                "openai.response.stream",
                &model,
                &messages,
                &result,
            );
            let completed = serde_json::json!({
                "type": "response.completed",
                "response": make_openai_response(&model, result),
            });
            write_sse_event(stream, "response.completed", &completed)?;
        }
        Err(err) => {
            let failed = serde_json::json!({
                "type": "response.failed",
                "response": {
                    "id": response_id,
                    "object": "response",
                    "created_at": created,
                    "status": "failed",
                    "model": model,
                    "error": {"message": err},
                }
            });
            write_sse_event(stream, "response.failed", &failed)?;
        }
    }
    write!(stream, "data: [DONE]\n\n")?;
    stream.flush()
}

/// Writes an Ollama-compatible newline-delimited generate stream.
fn route_ollama_generate_stream<W: Write>(
    request: &HttpRequest,
    stream: &mut W,
    runner: &Runner,
    options: &ServeOptions,
) -> io::Result<()> {
    let model_ids = advertised_model_ids(runner);
    let payload = match serde_json::from_slice::<OllamaGenerateRequest>(&request.body) {
        Ok(payload) => payload,
        Err(err) => {
            write_http_response(stream, 400, &json_error(&format!("Invalid JSON: {}", err)))?;
            return Ok(());
        }
    };
    let model = resolve_model(payload.model.as_deref(), &model_ids);
    let Some(prompt) = payload.prompt else {
        write_http_response(stream, 400, &json_error("Missing prompt."))?;
        return Ok(());
    };
    let mut generation = apply_ollama_options(&options.defaults, payload.options);
    if let Some(system) = payload.system {
        generation.system_prompt = system;
    }
    if let Some(stop) = payload.stop {
        generation.stop_sequences = stop.into_vec();
    }
    let started = Instant::now();
    write_ndjson_response_header(stream)?;
    let mut loaded_skills = HashSet::new();
    let result = runner.generate_stream_with_skill_memory(
        &prompt,
        &generation,
        &mut loaded_skills,
        |text| {
            let chunk = serde_json::json!({
                "model": model,
                "created_at": iso_timestamp(),
                "response": text,
                "done": false,
            });
            let _ = write_json_line(stream, &chunk);
        },
    );
    match result {
        Ok(result) => {
            let messages = [ChatMessage::user(prompt)];
            let _ = append_chat_history(
                options,
                "ollama.generate.stream",
                &model,
                &messages,
                &result,
            );
            let final_chunk = serde_json::json!({
                "model": model,
                "created_at": iso_timestamp(),
                "response": "",
                "done": true,
                "context": [],
                "total_duration": started.elapsed().as_nanos(),
                "load_duration": 0,
                "prompt_eval_count": result.stats.prompt_tokens,
                "prompt_eval_duration": result.stats.prefill_time.as_nanos(),
                "eval_count": result.stats.generated_tokens,
                "eval_duration": result.stats.decode_time.as_nanos(),
            });
            write_json_line(stream, &final_chunk)?;
        }
        Err(err) => {
            write_json_line(stream, &serde_json::json!({"error": err, "done": true}))?;
        }
    }
    stream.flush()
}

/// Writes an Ollama-compatible newline-delimited chat stream.
fn route_ollama_chat_stream<W: Write>(
    request: &HttpRequest,
    stream: &mut W,
    runner: &Runner,
    options: &ServeOptions,
) -> io::Result<()> {
    let model_ids = advertised_model_ids(runner);
    let payload = match serde_json::from_slice::<OllamaChatRequest>(&request.body) {
        Ok(payload) => payload,
        Err(err) => {
            write_http_response(stream, 400, &json_error(&format!("Invalid JSON: {}", err)))?;
            return Ok(());
        }
    };
    let model = resolve_model(payload.model.as_deref(), &model_ids);
    let messages = match parse_ollama_messages(payload.messages) {
        Ok(messages) => messages,
        Err(err) => {
            write_http_response(stream, 400, &json_error(&err))?;
            return Ok(());
        }
    };
    let generation = apply_ollama_options(&options.defaults, payload.options);
    let started = Instant::now();
    write_ndjson_response_header(stream)?;
    let mut loaded_skills = HashSet::new();
    let result = runner.generate_chat_stream_with_skill_memory(
        &messages,
        &generation,
        &mut loaded_skills,
        |text| {
            let chunk = serde_json::json!({
                "model": model,
                "created_at": iso_timestamp(),
                "message": {"role": "assistant", "content": text},
                "done": false,
            });
            let _ = write_json_line(stream, &chunk);
        },
    );
    match result {
        Ok(result) => {
            let _ = append_chat_history(options, "ollama.chat.stream", &model, &messages, &result);
            let final_chunk = serde_json::json!({
                "model": model,
                "created_at": iso_timestamp(),
                "message": {"role": "assistant", "content": ""},
                "done": true,
                "total_duration": started.elapsed().as_nanos(),
                "load_duration": 0,
                "prompt_eval_count": result.stats.prompt_tokens,
                "prompt_eval_duration": result.stats.prefill_time.as_nanos(),
                "eval_count": result.stats.generated_tokens,
                "eval_duration": result.stats.decode_time.as_nanos(),
            });
            write_json_line(stream, &final_chunk)?;
        }
        Err(err) => {
            write_json_line(stream, &serde_json::json!({"error": err, "done": true}))?;
        }
    }
    stream.flush()
}

/// Routes non-streaming HTTP requests.
fn route_request(request: &HttpRequest, runner: &Runner, options: &ServeOptions) -> (u16, String) {
    if request.method == "OPTIONS" {
        return (204, String::new());
    }

    let model_ids = advertised_model_ids(runner);
    match (request.method.as_str(), request.path.as_str()) {
        ("GET", "/") | ("GET", "/health") | ("GET", "/healthz") | ("GET", "/ready") => {
            (200, String::from("{\"status\":\"ok\"}"))
        }
        ("GET", "/api/version") => (200, String::from("{\"version\":\"rusty-llm-0.3.0\"}")),
        ("GET", "/api/explorer/model") => route_explorer_model(runner, options),
        ("GET", "/openapi.json") | ("GET", "/swagger.json") => {
            route_openapi_document(runner, &model_ids)
        }
        ("GET", "/api/tags") => route_ollama_tags(runner, &model_ids),
        ("GET", "/v1/models") | ("GET", "/api/v0/models") => {
            let created = unix_timestamp();
            let response = OpenAiModelListResponse {
                object: "list",
                data: model_ids
                    .into_iter()
                    .map(|id| OpenAiModelInfo {
                        id,
                        object: "model",
                        created,
                        owned_by: "rusty-llm",
                    })
                    .collect(),
            };
            json_response(response)
        }
        ("POST", path)
            if matches!(
                path,
                "/generate"
                    | "/v1/completions"
                    | "/v1/chat/completions"
                    | "/v1/responses"
                    | "/v1/embeddings"
                    | "/api/v0/completions"
                    | "/api/v0/chat/completions"
                    | "/api/v0/embeddings"
                    | "/api/generate"
                    | "/api/chat"
                    | "/api/embeddings"
                    | "/api/embed"
                    | "/api/explorer/tokenize"
                    | "/api/explorer/vector"
                    | "/api/explorer/neighbors"
            ) =>
        {
            if !request
                .content_type
                .as_deref()
                .map(is_json_content_type)
                .unwrap_or(false)
            {
                return (415, json_error("Content-Type must be application/json."));
            }
            match path {
                "/generate" => route_generate(&request.body, runner, options),
                "/v1/completions" | "/api/v0/completions" => {
                    route_openai_completion(&request.body, runner, options, &model_ids)
                }
                "/v1/chat/completions" | "/api/v0/chat/completions" => {
                    route_openai_chat(&request.body, runner, options, &model_ids)
                }
                "/v1/responses" => {
                    route_openai_response(&request.body, runner, options, &model_ids)
                }
                "/v1/embeddings" | "/api/v0/embeddings" => {
                    route_embeddings(&request.body, runner, &model_ids)
                }
                "/api/generate" => {
                    route_ollama_generate(&request.body, runner, options, &model_ids)
                }
                "/api/chat" => route_ollama_chat(&request.body, runner, options, &model_ids),
                "/api/embeddings" | "/api/embed" => {
                    route_ollama_embeddings(&request.body, runner, &model_ids)
                }
                "/api/explorer/tokenize" => route_explorer_tokenize(&request.body, runner),
                "/api/explorer/vector" => route_explorer_vector(&request.body, runner),
                "/api/explorer/neighbors" => route_explorer_neighbors(&request.body, runner),
                _ => (404, json_error("Not found")),
            }
        }
        ("GET", _) | ("POST", _) => (404, json_error("Not found")),
        _ => (405, json_error("Method not allowed.")),
    }
}

fn route_openapi_document(runner: &Runner, model_ids: &[String]) -> (u16, String) {
    let default_model = model_ids
        .first()
        .cloned()
        .unwrap_or_else(|| runner.architecture().to_string());
    json_response(serde_json::json!({
        "openapi": "3.1.0",
        "info": {
            "title": "RustyLLM API",
            "version": env!("CARGO_PKG_VERSION"),
            "description": "Local GGUF inference server with RustyLLM-native, OpenAI-compatible, LM Studio-compatible, Ollama-compatible, and explorer routes."
        },
        "servers": [{"url": "/"}],
        "tags": [
            {"name": "health"},
            {"name": "openai"},
            {"name": "ollama"},
            {"name": "native"},
            {"name": "explorer"}
        ],
        "paths": {
            "/": {
                "get": {
                    "tags": ["health"],
                    "summary": "Health check",
                    "responses": { "200": { "description": "Server is alive" } }
                }
            },
            "/health": {
                "get": {
                    "tags": ["health"],
                    "summary": "Health check",
                    "responses": { "200": { "description": "Server is alive" } }
                }
            },
            "/ready": {
                "get": {
                    "tags": ["health"],
                    "summary": "Readiness check",
                    "responses": { "200": { "description": "Loaded model is ready" } }
                }
            },
            "/openapi.json": {
                "get": {
                    "tags": ["native"],
                    "summary": "OpenAPI document",
                    "responses": { "200": { "description": "OpenAPI JSON" } }
                }
            },
            "/docs": {
                "get": {
                    "tags": ["native"],
                    "summary": "Swagger UI",
                    "responses": { "200": { "description": "Swagger UI HTML" } }
                }
            },
            "/generate": {
                "post": {
                    "tags": ["native"],
                    "summary": "RustyLLM-native text generation",
                    "requestBody": { "required": true, "content": { "application/json": { "schema": { "$ref": "#/components/schemas/GenerateRequest" } } } },
                    "responses": { "200": { "description": "Generated text" } }
                }
            },
            "/v1/models": {
                "get": {
                    "tags": ["openai"],
                    "summary": "List advertised model IDs",
                    "responses": { "200": { "description": "Model list" } }
                }
            },
            "/v1/chat/completions": {
                "post": {
                    "tags": ["openai"],
                    "summary": "OpenAI-compatible chat completion",
                    "requestBody": { "required": true, "content": { "application/json": { "schema": { "$ref": "#/components/schemas/OpenAIChatRequest" } } } },
                    "responses": { "200": { "description": "Chat completion, or SSE stream when stream=true" } }
                }
            },
            "/v1/completions": {
                "post": {
                    "tags": ["openai"],
                    "summary": "OpenAI-compatible text completion",
                    "requestBody": { "required": true, "content": { "application/json": { "schema": { "$ref": "#/components/schemas/OpenAICompletionRequest" } } } },
                    "responses": { "200": { "description": "Completion, or SSE stream when stream=true" } }
                }
            },
            "/v1/responses": {
                "post": {
                    "tags": ["openai"],
                    "summary": "OpenAI-compatible Responses API subset",
                    "description": "Accepts input, instructions, max_output_tokens, response_format, text.format, tools, and tool_choice. Tool definitions are accepted for client compatibility but not executed by the model server.",
                    "requestBody": { "required": true, "content": { "application/json": { "schema": { "$ref": "#/components/schemas/OpenAIResponsesRequest" } } } },
                    "responses": { "200": { "description": "Responses API object, or SSE stream when stream=true" } }
                }
            },
            "/v1/embeddings": {
                "post": {
                    "tags": ["openai"],
                    "summary": "OpenAI-compatible embeddings",
                    "requestBody": { "required": true, "content": { "application/json": { "schema": { "$ref": "#/components/schemas/EmbeddingsRequest" } } } },
                    "responses": { "200": { "description": "Embedding list" } }
                }
            },
            "/api/v0/models": {
                "get": {
                    "tags": ["openai"],
                    "summary": "LM Studio model-list alias",
                    "responses": { "200": { "description": "Model list" } }
                }
            },
            "/api/v0/completions": {
                "post": {
                    "tags": ["openai"],
                    "summary": "LM Studio completions alias",
                    "responses": { "200": { "description": "Completion" } }
                }
            },
            "/api/v0/chat/completions": {
                "post": {
                    "tags": ["openai"],
                    "summary": "LM Studio chat completions alias",
                    "responses": { "200": { "description": "Chat completion" } }
                }
            },
            "/api/v0/embeddings": {
                "post": {
                    "tags": ["openai"],
                    "summary": "LM Studio embeddings alias",
                    "responses": { "200": { "description": "Embedding list" } }
                }
            },
            "/api/version": {
                "get": {
                    "tags": ["ollama"],
                    "summary": "Ollama-compatible version",
                    "responses": { "200": { "description": "Version metadata" } }
                }
            },
            "/api/tags": {
                "get": {
                    "tags": ["ollama"],
                    "summary": "Ollama-compatible model tags",
                    "responses": { "200": { "description": "Model tags" } }
                }
            },
            "/api/generate": {
                "post": {
                    "tags": ["ollama"],
                    "summary": "Ollama-compatible generation",
                    "requestBody": { "required": true, "content": { "application/json": { "schema": { "$ref": "#/components/schemas/OllamaGenerateRequest" } } } },
                    "responses": { "200": { "description": "JSON response, or application/x-ndjson stream when stream=true" } }
                }
            },
            "/api/chat": {
                "post": {
                    "tags": ["ollama"],
                    "summary": "Ollama-compatible chat",
                    "requestBody": { "required": true, "content": { "application/json": { "schema": { "$ref": "#/components/schemas/OllamaChatRequest" } } } },
                    "responses": { "200": { "description": "JSON response, or application/x-ndjson stream when stream=true" } }
                }
            },
            "/api/embeddings": {
                "post": {
                    "tags": ["ollama"],
                    "summary": "Ollama-compatible embedding",
                    "responses": { "200": { "description": "Embedding response" } }
                }
            },
            "/api/embed": {
                "post": {
                    "tags": ["ollama"],
                    "summary": "Ollama-compatible batched embedding alias",
                    "responses": { "200": { "description": "Embedding response" } }
                }
            },
            "/api/explorer/model": {
                "get": {
                    "tags": ["explorer"],
                    "summary": "Inspect loaded GGUF metadata, tensors, and runtime config",
                    "responses": { "200": { "description": "Explorer model anatomy" } }
                }
            },
            "/api/explorer/tokenize": {
                "post": {
                    "tags": ["explorer"],
                    "summary": "Tokenize text with the loaded GGUF tokenizer",
                    "requestBody": { "required": true },
                    "responses": { "200": { "description": "Token list" } }
                }
            },
            "/api/explorer/vector": {
                "post": {
                    "tags": ["explorer"],
                    "summary": "Inspect a text or token vector in static token-embedding space",
                    "requestBody": { "required": true },
                    "responses": { "200": { "description": "Vector summary" } }
                }
            },
            "/api/explorer/neighbors": {
                "post": {
                    "tags": ["explorer"],
                    "summary": "Find nearest vocabulary tokens by cosine similarity",
                    "requestBody": { "required": true },
                    "responses": { "200": { "description": "Nearest token list" } }
                }
            }
        },
        "components": {
            "schemas": {
                "ChatMessage": {
                    "type": "object",
                    "required": ["role", "content"],
                    "properties": {
                        "role": { "type": "string", "enum": ["system", "user", "assistant"] },
                        "content": { "oneOf": [{ "type": "string" }, { "type": "array", "items": { "type": "object" } }] }
                    }
                },
                "GenerateRequest": {
                    "type": "object",
                    "properties": {
                        "prompt": { "type": "string" },
                        "messages": { "type": "array", "items": { "$ref": "#/components/schemas/ChatMessage" } },
                        "max_tokens": { "type": "integer", "default": 256 },
                        "temp": { "type": "number", "default": 0.7 },
                        "top_p": { "type": "number", "default": 0.9 },
                        "top_k": { "type": "integer", "default": 40 },
                        "repeat_penalty": { "type": "number", "default": 1.1 },
                        "thinking": { "type": "boolean" },
                        "thinking_prompt": { "type": "string" },
                        "thinking_max_tokens": { "type": "integer", "minimum": 1 },
                        "stop": { "oneOf": [{ "type": "string" }, { "type": "array", "items": { "type": "string" } }] },
                        "conversation_id": { "type": "string" },
                        "cache_prompt": { "type": "boolean" }
                    }
                },
                "OpenAICompletionRequest": {
                    "type": "object",
                    "required": ["prompt"],
                    "properties": {
                        "model": { "type": "string", "enum": model_ids },
                        "prompt": { "oneOf": [{ "type": "string" }, { "type": "array", "items": { "type": "string" } }] },
                        "max_tokens": { "type": "integer" },
                        "max_completion_tokens": { "type": "integer" },
                        "temperature": { "type": "number" },
                        "top_p": { "type": "number" },
                        "thinking": { "type": "boolean" },
                        "thinking_prompt": { "type": "string" },
                        "thinking_max_tokens": { "type": "integer", "minimum": 1 },
                        "stream": { "type": "boolean" },
                        "response_format": { "type": "object" },
                        "tools": { "type": "array", "items": { "type": "object" } },
                        "tool_choice": {}
                    }
                },
                "OpenAIChatRequest": {
                    "type": "object",
                    "required": ["messages"],
                    "properties": {
                        "model": { "type": "string", "enum": model_ids },
                        "messages": { "type": "array", "items": { "$ref": "#/components/schemas/ChatMessage" } },
                        "max_tokens": { "type": "integer" },
                        "max_completion_tokens": { "type": "integer" },
                        "temperature": { "type": "number" },
                        "top_p": { "type": "number" },
                        "thinking": { "type": "boolean" },
                        "thinking_prompt": { "type": "string" },
                        "thinking_max_tokens": { "type": "integer", "minimum": 1 },
                        "stream": { "type": "boolean" },
                        "response_format": { "type": "object" },
                        "tools": { "type": "array", "items": { "type": "object" } },
                        "tool_choice": {},
                        "conversation_id": { "type": "string" },
                        "cache_prompt": { "type": "boolean" }
                    }
                },
                "OpenAIResponsesRequest": {
                    "type": "object",
                    "required": ["input"],
                    "properties": {
                        "model": { "type": "string", "enum": model_ids },
                        "input": { "oneOf": [{ "type": "string" }, { "type": "array" }, { "type": "object" }] },
                        "instructions": { "type": "string" },
                        "max_output_tokens": { "type": "integer" },
                        "temperature": { "type": "number" },
                        "top_p": { "type": "number" },
                        "thinking": { "type": "boolean" },
                        "thinking_prompt": { "type": "string" },
                        "thinking_max_tokens": { "type": "integer", "minimum": 1 },
                        "stream": { "type": "boolean" },
                        "text": { "type": "object" },
                        "response_format": { "type": "object" },
                        "tools": { "type": "array", "items": { "type": "object" } },
                        "tool_choice": {}
                    }
                },
                "EmbeddingsRequest": {
                    "type": "object",
                    "required": ["input"],
                    "properties": {
                        "model": { "type": "string", "enum": model_ids },
                        "input": { "oneOf": [{ "type": "string" }, { "type": "array", "items": { "type": "string" } }] },
                        "encoding_format": { "type": "string", "default": "float" }
                    }
                },
                "OllamaGenerateRequest": {
                    "type": "object",
                    "properties": {
                        "model": { "type": "string", "enum": model_ids },
                        "prompt": { "type": "string" },
                        "system": { "type": "string" },
                        "thinking": { "type": "boolean" },
                        "thinking_prompt": { "type": "string" },
                        "thinking_max_tokens": { "type": "integer", "minimum": 1 },
                        "stream": { "type": "boolean" },
                        "options": { "type": "object" },
                        "stop": { "oneOf": [{ "type": "string" }, { "type": "array", "items": { "type": "string" } }] }
                    }
                },
                "OllamaChatRequest": {
                    "type": "object",
                    "properties": {
                        "model": { "type": "string", "enum": model_ids },
                        "messages": { "type": "array", "items": { "$ref": "#/components/schemas/ChatMessage" } },
                        "thinking": { "type": "boolean" },
                        "thinking_prompt": { "type": "string" },
                        "thinking_max_tokens": { "type": "integer", "minimum": 1 },
                        "stream": { "type": "boolean" },
                        "options": { "type": "object" }
                    }
                }
            }
        },
        "x-rustyllm": {
            "loaded_model": default_model,
            "architecture": runner.architecture(),
            "model_name": runner.model_name(),
            "model_ids": model_ids,
            "format": "gguf",
            "quantization": dominant_quantization(runner),
            "explorer": "/explorer"
        }
    }))
}

/// Generate a response, either stateless or via a persistent session.
///
/// When `use_session` is `true` and `conv_id` is `Some`, the request is
/// routed through the session store so that the KV cache is reused across
/// turns.  Falls back to stateless generation when the session store is
/// unavailable or disabled.
fn generate_with_optional_session<F>(
    runner: &Runner,
    options: &ServeOptions,
    messages: &[ChatMessage],
    generation: &GenerationOptions,
    use_session: bool,
    conv_id: Option<&str>,
    on_token: F,
) -> Result<crate::runtime::GenerationResult, String>
where
    F: FnMut(&str),
{
    if use_session {
        if let (Some(id), Some(store)) = (conv_id, &options.session_store) {
            let max_cached = store.max_cached_tokens();
            let session_arc = store.get_or_create(id, || runner.new_session(max_cached));
            let mut session = session_arc.lock().expect("session lock poisoned");
            return runner.generate_chat_with_session_with_skill_memory(
                messages,
                generation,
                &mut session,
                on_token,
            );
        }
    }
    // Stateless fallback.
    let mut loaded_skills = HashSet::new();
    runner.generate_chat_stream_with_skill_memory(
        messages,
        generation,
        &mut loaded_skills,
        on_token,
    )
}

/// Handles the native `/generate` JSON endpoint.
fn route_generate(body: &[u8], runner: &Runner, options: &ServeOptions) -> (u16, String) {
    match serde_json::from_slice::<GenerateRequest>(body) {
        Ok(payload) => {
            let generation = apply_generation_overrides(
                &options.defaults,
                payload.max_tokens,
                payload.temp,
                payload.top_p,
                payload.top_k,
                payload.repeat_penalty,
                payload.seed,
                payload.system_prompt,
                payload.stop.map(|s| s.into_vec()),
                payload.thinking,
                payload.thinking_prompt,
                payload.thinking_max_tokens,
            );

            let use_session = payload.cache_prompt.unwrap_or(false);
            let conv_id = payload.conversation_id.clone();

            let mut history_messages = Vec::new();
            let result = if let Some(messages) = payload.messages {
                match parse_api_messages(messages) {
                    Ok(messages) => generate_with_optional_session(
                        runner,
                        options,
                        &messages,
                        &generation,
                        use_session,
                        conv_id.as_deref(),
                        |_| {},
                    )
                    .inspect(|_| {
                        history_messages = messages;
                    }),
                    Err(err) => Err(err),
                }
            } else if let Some(prompt) = payload.prompt {
                history_messages = vec![ChatMessage::user(prompt.clone())];
                generate_with_optional_session(
                    runner,
                    options,
                    &[ChatMessage::user(prompt)],
                    &generation,
                    use_session,
                    conv_id.as_deref(),
                    |_| {},
                )
            } else {
                Err(String::from("Missing prompt or messages."))
            };

            match result {
                Ok(result) => {
                    let model = runner.model_name().unwrap_or(runner.architecture());
                    let _ = append_chat_history(
                        options,
                        "rusty.generate",
                        model,
                        &history_messages,
                        &result,
                    );
                    let cache_stats =
                        make_cache_stats(&result.stats, use_session, conv_id.as_deref());
                    json_response(GenerateResponse {
                        text: &result.text,
                        prompt_tokens: result.stats.prompt_tokens,
                        generated_tokens: result.stats.generated_tokens,
                        prefill_ms: result.stats.prefill_time.as_millis(),
                        decode_ms: result.stats.decode_time.as_millis(),
                        total_ms: result.stats.total_time.as_millis(),
                        cache_stats,
                    })
                }
                Err(err) => (400, json_error(&err)),
            }
        }
        Err(err) => (400, json_error(&format!("Invalid JSON: {}", err))),
    }
}

/// Handles OpenAI-compatible text completion requests.
fn route_openai_completion(
    body: &[u8],
    runner: &Runner,
    options: &ServeOptions,
    model_ids: &[String],
) -> (u16, String) {
    match serde_json::from_slice::<OpenAiCompletionsRequest>(body) {
        Ok(payload) => {
            let model = resolve_model(payload.model.as_deref(), model_ids);
            let prompt = match payload.prompt {
                OpenAiPrompt::Single(prompt) => prompt,
                OpenAiPrompt::Batch(mut prompts) => {
                    if prompts.is_empty() {
                        return (400, json_error("Prompt array must not be empty."));
                    }
                    prompts.remove(0)
                }
            };
            let max_tokens = payload.max_completion_tokens.or(payload.max_tokens);
            let mut generation = apply_generation_overrides(
                &options.defaults,
                max_tokens,
                payload.temperature,
                payload.top_p,
                payload.top_k,
                payload.repeat_penalty,
                payload.seed,
                payload.system_prompt,
                payload.stop.map(|s| s.into_vec()),
                payload.thinking,
                payload.thinking_prompt,
                payload.thinking_max_tokens,
            );
            apply_response_format_hint(&mut generation, payload.response_format.as_ref());
            let mut loaded_skills = HashSet::new();
            match runner.generate_with_skill_memory(&prompt, &generation, &mut loaded_skills) {
                Ok(result) => {
                    let messages = [ChatMessage::user(prompt)];
                    let _ = append_chat_history(
                        options,
                        "openai.completion",
                        &model,
                        &messages,
                        &result,
                    );
                    let created = unix_timestamp();
                    let usage = OpenAiUsage {
                        prompt_tokens: result.stats.prompt_tokens,
                        completion_tokens: result.stats.generated_tokens,
                        total_tokens: result.stats.prompt_tokens + result.stats.generated_tokens,
                    };
                    json_response(OpenAiCompletionResponse {
                        id: format!("cmpl-rustyllm-{}", created),
                        object: "text_completion",
                        created,
                        model,
                        choices: vec![OpenAiCompletionChoice {
                            text: result.text,
                            index: 0,
                            finish_reason: "stop",
                        }],
                        usage,
                    })
                }
                Err(err) => (400, json_error(&err)),
            }
        }
        Err(err) => (400, json_error(&format!("Invalid JSON: {}", err))),
    }
}

/// Handles OpenAI-compatible chat completion requests.
fn route_openai_chat(
    body: &[u8],
    runner: &Runner,
    options: &ServeOptions,
    model_ids: &[String],
) -> (u16, String) {
    match serde_json::from_slice::<OpenAiChatCompletionsRequest>(body) {
        Ok(payload) => {
            let model = resolve_model(payload.model.as_deref(), model_ids);
            let messages = match parse_api_messages(payload.messages) {
                Ok(messages) => messages,
                Err(err) => return (400, json_error(&err)),
            };
            let max_tokens = payload.max_completion_tokens.or(payload.max_tokens);
            let mut generation = apply_generation_overrides(
                &options.defaults,
                max_tokens,
                payload.temperature,
                payload.top_p,
                payload.top_k,
                payload.repeat_penalty,
                payload.seed,
                payload.system_prompt,
                payload.stop.map(|s| s.into_vec()),
                payload.thinking,
                payload.thinking_prompt,
                payload.thinking_max_tokens,
            );
            apply_response_format_hint(&mut generation, payload.response_format.as_ref());
            let use_session = payload.cache_prompt.unwrap_or(false);
            let conv_id = payload.conversation_id.clone();
            let result = generate_with_optional_session(
                runner,
                options,
                &messages,
                &generation,
                use_session,
                conv_id.as_deref(),
                |_| {},
            );
            match result {
                Ok(result) => {
                    let _ = append_chat_history(options, "openai.chat", &model, &messages, &result);
                    let created = unix_timestamp();
                    let usage = OpenAiUsage {
                        prompt_tokens: result.stats.prompt_tokens,
                        completion_tokens: result.stats.generated_tokens,
                        total_tokens: result.stats.prompt_tokens + result.stats.generated_tokens,
                    };
                    let cache_stats =
                        make_cache_stats(&result.stats, use_session, conv_id.as_deref());
                    json_response(OpenAiChatCompletionResponse {
                        id: format!("chatcmpl-rustyllm-{}", created),
                        object: "chat.completion",
                        created,
                        model,
                        choices: vec![OpenAiChatChoice {
                            index: 0,
                            message: OpenAiChatMessage {
                                role: "assistant",
                                content: result.text,
                            },
                            finish_reason: "stop",
                        }],
                        usage,
                        cache_stats,
                    })
                }
                Err(err) => (400, json_error(&err)),
            }
        }
        Err(err) => (400, json_error(&format!("Invalid JSON: {}", err))),
    }
}

/// Handles OpenAI-compatible Responses API requests.
fn route_openai_response(
    body: &[u8],
    runner: &Runner,
    options: &ServeOptions,
    model_ids: &[String],
) -> (u16, String) {
    match serde_json::from_slice::<OpenAiResponsesRequest>(body) {
        Ok(payload) => {
            let model = resolve_model(payload.model.as_deref(), model_ids);
            let messages = match parse_responses_input(payload.input.as_ref()) {
                Ok(messages) => messages,
                Err(err) => return (400, json_error(&err)),
            };
            let max_tokens = payload.max_output_tokens.or(payload.max_completion_tokens);
            let mut generation = apply_generation_overrides(
                &options.defaults,
                max_tokens,
                payload.temperature,
                payload.top_p,
                payload.top_k,
                payload.repeat_penalty,
                payload.seed,
                payload.instructions,
                payload.stop.map(|s| s.into_vec()),
                payload.thinking,
                payload.thinking_prompt,
                payload.thinking_max_tokens,
            );
            let format_hint = payload
                .text
                .as_ref()
                .and_then(|text| text.format.as_ref())
                .or(payload.response_format.as_ref());
            apply_response_format_hint(&mut generation, format_hint);

            match generate_with_optional_session(
                runner,
                options,
                &messages,
                &generation,
                false,
                None,
                |_| {},
            ) {
                Ok(result) => {
                    let _ =
                        append_chat_history(options, "openai.response", &model, &messages, &result);
                    json_response(make_openai_response(&model, result))
                }
                Err(err) => (400, json_error(&err)),
            }
        }
        Err(err) => (400, json_error(&format!("Invalid JSON: {}", err))),
    }
}

/// Handles OpenAI-compatible embedding requests.
fn route_embeddings(body: &[u8], runner: &Runner, model_ids: &[String]) -> (u16, String) {
    match serde_json::from_slice::<EmbeddingsRequest>(body) {
        Ok(payload) => {
            let model = resolve_model(payload.model.as_deref(), model_ids);
            let inputs = match payload.input {
                EmbeddingsInput::Single(input) => vec![input],
                EmbeddingsInput::Batch(inputs) => inputs,
            };
            if inputs.is_empty() {
                return (400, json_error("input must not be empty."));
            }

            let mut data = Vec::new();
            for (index, text) in inputs.iter().enumerate() {
                match runner.embed(text) {
                    Ok(result) => data.push(serde_json::json!({
                        "object": "embedding",
                        "embedding": result.embedding,
                        "index": index,
                    })),
                    Err(err) => return (400, json_error(&err)),
                }
            }

            let total_tokens = inputs
                .iter()
                .map(|text| runner.tokenizer().encode(text).len())
                .sum::<usize>();
            let response = serde_json::json!({
                "object": "list",
                "data": data,
                "model": model,
                "usage": {
                    "prompt_tokens": total_tokens,
                    "total_tokens": total_tokens,
                }
            });
            match serde_json::to_string(&response) {
                Ok(body) => (200, body),
                Err(err) => (500, json_error(&format!("Serialize error: {}", err))),
            }
        }
        Err(err) => (400, json_error(&format!("Invalid JSON: {}", err))),
    }
}

fn route_explorer_model(runner: &Runner, options: &ServeOptions) -> (u16, String) {
    let config = runner.config();
    let mut dtype_counts = std::collections::BTreeMap::<String, usize>::new();
    let mut family_counts = std::collections::BTreeMap::<String, usize>::new();
    for tensor in &runner.gguf().tensors {
        *dtype_counts
            .entry(format!("{:?}", tensor.dtype))
            .or_insert(0) += 1;
        *family_counts
            .entry(explorer_tensor_family(&tensor.name))
            .or_insert(0) += 1;
    }

    let mut metadata = serde_json::Map::new();
    let mut keys = runner.gguf().metadata.keys().cloned().collect::<Vec<_>>();
    keys.sort();
    for key in keys {
        if explorer_metadata_key(&key) {
            if let Some(value) = runner.gguf().metadata.get(&key) {
                metadata.insert(key, meta_value_json(value));
            }
        }
    }

    let tensors = runner
        .gguf()
        .tensors
        .iter()
        .map(|tensor| {
            serde_json::json!({
                "name": tensor.name,
                "dims": tensor.dims,
                "dtype": format!("{:?}", tensor.dtype),
                "family": explorer_tensor_family(&tensor.name),
                "elements": tensor.numel(),
                "offset": tensor.offset,
            })
        })
        .collect::<Vec<_>>();

    json_response(serde_json::json!({
        "model": {
            "name": runner.model_name(),
            "architecture": runner.architecture(),
            "vocab_size": runner.tokenizer().vocab_size(),
            "token_embedding_rows": runner.token_embedding_count(),
        },
        "config": {
            "dim": config.dim,
            "hidden_dim": config.hidden_dim,
            "layers": config.n_layers,
            "heads": config.n_heads,
            "kv_heads": config.n_kv_heads,
            "head_dim": config.head_dim,
            "value_dim": config.value_dim,
            "context_length": config.max_seq_len,
            "rope_theta": config.rope_theta,
            "sliding_window": config.sliding_window,
            "experts": config.expert_count,
            "experts_used": config.expert_used_count,
        },
        "gguf": {
            "metadata_count": runner.gguf().metadata.len(),
            "tensor_count": runner.gguf().tensors.len(),
            "data_offset": runner.gguf().data_offset,
            "dtype_counts": dtype_counts,
            "family_counts": family_counts,
            "metadata": metadata,
            "tensors": tensors,
        },
        "catalog": explorer_catalog_json(options),
    }))
}

fn explorer_catalog_json(options: &ServeOptions) -> serde_json::Value {
    let Some(catalog) = options.model_catalog.as_ref() else {
        return serde_json::Value::Null;
    };
    let loaded_model_path = catalog.loaded_model_path.as_str();
    let entries = catalog
        .entries
        .iter()
        .map(|entry| {
            serde_json::json!({
                "id": entry.id,
                "repository": entry.repository,
                "file_name": entry.file_name,
                "path": entry.path,
                "size_bytes": entry.size_bytes,
                "architecture": entry.architecture,
                "model_name": entry.model_name,
                "status": entry.status,
                "is_projector": entry.is_projector,
                "is_supported": entry.is_supported,
                "is_loaded": entry.path == loaded_model_path,
            })
        })
        .collect::<Vec<_>>();
    serde_json::json!({
        "model_dir": catalog.model_dir,
        "loaded_model_path": loaded_model_path,
        "entries": entries,
    })
}

fn route_explorer_tokenize(body: &[u8], runner: &Runner) -> (u16, String) {
    match serde_json::from_slice::<ExplorerTokenizeRequest>(body) {
        Ok(payload) => {
            let tokens = if payload.add_bos.unwrap_or(false) {
                runner.tokenizer().encode(&payload.input)
            } else {
                runner.tokenizer().encode_without_bos(&payload.input)
            };
            let token_json = tokens
                .iter()
                .map(|&id| explorer_token_json(runner, id))
                .collect::<Vec<_>>();
            json_response(serde_json::json!({
                "input": payload.input,
                "token_count": tokens.len(),
                "tokens": token_json,
            }))
        }
        Err(err) => (400, json_error(&format!("Invalid JSON: {}", err))),
    }
}

fn route_explorer_vector(body: &[u8], runner: &Runner) -> (u16, String) {
    match serde_json::from_slice::<ExplorerVectorRequest>(body) {
        Ok(payload) => {
            match explorer_query_vector(runner, payload.input.as_deref(), payload.token_id) {
                Ok(query) => json_response(serde_json::json!({
                    "source": query.source,
                    "tokens": query.tokens.iter().map(|&id| explorer_token_json(runner, id)).collect::<Vec<_>>(),
                    "vector": vector_summary_json(&query.vector),
                    "projection": projection_json(&query.vector),
                })),
                Err(err) => (400, json_error(&err)),
            }
        }
        Err(err) => (400, json_error(&format!("Invalid JSON: {}", err))),
    }
}

fn route_explorer_neighbors(body: &[u8], runner: &Runner) -> (u16, String) {
    match serde_json::from_slice::<ExplorerNeighborsRequest>(body) {
        Ok(payload) => {
            let limit = payload.limit.unwrap_or(24).clamp(1, 60);
            let include_special = payload.include_special.unwrap_or(false);
            let token_id = payload.token_id;
            match explorer_query_vector(runner, payload.input.as_deref(), token_id) {
                Ok(query) => {
                    let search_limit = if token_id.is_some() { limit + 1 } else { limit };
                    let mut neighbors = match runner.nearest_token_embeddings(
                        &query.vector,
                        search_limit,
                        include_special,
                    ) {
                        Ok(neighbors) => neighbors,
                        Err(err) => return (400, json_error(&err)),
                    };
                    if let Some(id) = token_id {
                        neighbors.retain(|neighbor| neighbor.id != id);
                        neighbors.truncate(limit);
                    }

                    let items = neighbors
                        .iter()
                        .filter_map(|neighbor| {
                            let vector = runner.token_embedding(neighbor.id).ok()?;
                            Some(serde_json::json!({
                                "token": explorer_token_json(runner, neighbor.id),
                                "score": neighbor.score,
                                "projection": projection_json(&vector),
                                "vector_preview": vector.iter().take(8).copied().collect::<Vec<_>>(),
                            }))
                        })
                        .collect::<Vec<_>>();

                    json_response(serde_json::json!({
                        "source": query.source,
                        "query_tokens": query.tokens.iter().map(|&id| explorer_token_json(runner, id)).collect::<Vec<_>>(),
                        "query_projection": projection_json(&query.vector),
                        "limit": limit,
                        "include_special": include_special,
                        "neighbors": items,
                    }))
                }
                Err(err) => (400, json_error(&err)),
            }
        }
        Err(err) => (400, json_error(&format!("Invalid JSON: {}", err))),
    }
}

struct ExplorerQueryVector {
    source: &'static str,
    tokens: Vec<u32>,
    vector: Vec<f32>,
}

fn explorer_query_vector(
    runner: &Runner,
    input: Option<&str>,
    token_id: Option<u32>,
) -> Result<ExplorerQueryVector, String> {
    if let Some(id) = token_id {
        return Ok(ExplorerQueryVector {
            source: "token_embedding",
            tokens: vec![id],
            vector: runner.token_embedding(id)?,
        });
    }

    let Some(text) = input.map(str::trim).filter(|text| !text.is_empty()) else {
        return Err(String::from("input or token_id is required."));
    };
    let (vector, tokens) = runner.token_embedding_query(text)?;
    Ok(ExplorerQueryVector {
        source: "mean_token_embedding",
        tokens,
        vector,
    })
}

fn explorer_token_json(runner: &Runner, token_id: u32) -> serde_json::Value {
    let raw = runner.tokenizer().raw_token(token_id).unwrap_or("");
    let decoded = runner.tokenizer().decode_token(token_id);
    serde_json::json!({
        "id": token_id,
        "raw": raw,
        "decoded": decoded,
        "score": runner.tokenizer().token_score(token_id),
        "special": explorer_token_is_special(runner, token_id, raw, &decoded),
    })
}

fn explorer_token_is_special(runner: &Runner, token_id: u32, raw: &str, decoded: &str) -> bool {
    if token_id == runner.tokenizer().bos_id || token_id == runner.tokenizer().eos_id {
        return true;
    }
    if decoded.trim().is_empty() {
        return true;
    }
    if raw.starts_with("<0x") {
        return true;
    }
    (raw.starts_with('<') && raw.ends_with('>')) || (raw.starts_with('[') && raw.ends_with(']'))
}

fn vector_summary_json(vector: &[f32]) -> serde_json::Value {
    let mut norm = 0.0f32;
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut sum = 0.0f32;
    for &value in vector {
        norm += value * value;
        min = min.min(value);
        max = max.max(value);
        sum += value;
    }
    let mut top = vector
        .iter()
        .enumerate()
        .map(|(index, &value)| (index, value))
        .collect::<Vec<_>>();
    top.sort_by(|a, b| {
        b.1.abs()
            .partial_cmp(&a.1.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let top_dimensions = top
        .into_iter()
        .take(12)
        .map(|(index, value)| serde_json::json!({ "index": index, "value": value }))
        .collect::<Vec<_>>();

    let dimensions = vector.len();
    serde_json::json!({
        "dimensions": dimensions,
        "l2_norm": norm.sqrt(),
        "min": if dimensions == 0 { 0.0 } else { min },
        "max": if dimensions == 0 { 0.0 } else { max },
        "mean": if dimensions == 0 { 0.0 } else { sum / dimensions as f32 },
        "preview": vector.iter().take(48).copied().collect::<Vec<_>>(),
        "top_dimensions": top_dimensions,
    })
}

fn projection_json(vector: &[f32]) -> serde_json::Value {
    let (x, y) = explorer_projection(vector);
    serde_json::json!({ "x": x, "y": y })
}

fn explorer_projection(vector: &[f32]) -> (f32, f32) {
    if vector.is_empty() {
        return (0.0, 0.0);
    }
    let mut x = 0.0f32;
    let mut y = 0.0f32;
    for (i, &value) in vector.iter().enumerate() {
        let p = i as f32 + 1.0;
        x += value * (p * 12.9898).sin();
        y += value * (p * 78.233).cos();
    }
    let scale = (vector.len() as f32).sqrt().max(1.0);
    (x / scale, y / scale)
}

fn explorer_metadata_key(key: &str) -> bool {
    key.starts_with("general.")
        || key.starts_with("tokenizer.ggml.model")
        || key.starts_with("tokenizer.ggml.pre")
        || key.starts_with("tokenizer.ggml.add_")
        || key.ends_with(".context_length")
        || key.ends_with(".embedding_length")
        || key.ends_with(".block_count")
        || key.ends_with(".feed_forward_length")
        || key.contains(".attention.")
        || key.contains(".rope.")
        || key.contains(".expert_")
}

fn explorer_tensor_family(name: &str) -> String {
    if name == "token_embd.weight" {
        return String::from("token embeddings");
    }
    if name.starts_with("output") {
        return String::from("output");
    }
    if !name.starts_with("blk.") {
        return String::from("other");
    }
    if name.contains(".attn_") || name.contains("attention") {
        String::from("attention")
    } else if name.contains(".ffn_") || name.contains(".moe_") || name.contains("expert") {
        String::from("feed-forward")
    } else if name.contains("norm") {
        String::from("normalization")
    } else {
        String::from("block other")
    }
}

fn meta_value_json(value: &MetaValue) -> serde_json::Value {
    match value {
        MetaValue::U8(v) => serde_json::json!(v),
        MetaValue::I8(v) => serde_json::json!(v),
        MetaValue::U16(v) => serde_json::json!(v),
        MetaValue::I16(v) => serde_json::json!(v),
        MetaValue::U32(v) => serde_json::json!(v),
        MetaValue::I32(v) => serde_json::json!(v),
        MetaValue::U64(v) => serde_json::json!(v),
        MetaValue::I64(v) => serde_json::json!(v),
        MetaValue::F32(v) => serde_json::json!(v),
        MetaValue::F64(v) => serde_json::json!(v),
        MetaValue::Bool(v) => serde_json::json!(v),
        MetaValue::Str(v) => serde_json::json!(v),
        MetaValue::Array(values) => {
            if values.len() > 16 {
                serde_json::json!({
                    "type": "array",
                    "len": values.len(),
                    "preview": values.iter().take(8).map(meta_value_json).collect::<Vec<_>>(),
                })
            } else {
                serde_json::Value::Array(values.iter().map(meta_value_json).collect())
            }
        }
    }
}

/// Handles the Ollama-compatible model listing route.
fn route_ollama_tags(runner: &Runner, model_ids: &[String]) -> (u16, String) {
    let modified_at = iso_timestamp();
    let size = runner.gguf().tensors.len();
    let models = model_ids
        .iter()
        .map(|id| {
            serde_json::json!({
                "name": id,
                "model": id,
                "modified_at": modified_at,
                "size": size,
                "digest": format!("rusty-llm-{}", runner.architecture()),
                "details": {
                    "format": "gguf",
                    "family": runner.architecture(),
                    "families": [runner.architecture()],
                    "parameter_size": "unknown",
                    "quantization_level": dominant_quantization(runner),
                }
            })
        })
        .collect::<Vec<_>>();
    json_response(serde_json::json!({ "models": models }))
}

/// Handles the Ollama-compatible generate route.
fn route_ollama_generate(
    body: &[u8],
    runner: &Runner,
    options: &ServeOptions,
    model_ids: &[String],
) -> (u16, String) {
    match serde_json::from_slice::<OllamaGenerateRequest>(body) {
        Ok(payload) => {
            let model = resolve_model(payload.model.as_deref(), model_ids);
            let Some(prompt) = payload.prompt else {
                return (400, json_error("Missing prompt."));
            };
            let mut generation = apply_ollama_options(&options.defaults, payload.options);
            if let Some(system) = payload.system {
                generation.system_prompt = system;
            }
            if let Some(stop) = payload.stop {
                generation.stop_sequences = stop.into_vec();
            }
            apply_thinking_overrides(
                &mut generation,
                payload.thinking,
                payload.thinking_prompt,
                payload.thinking_max_tokens,
            );
            let started = Instant::now();
            let mut loaded_skills = HashSet::new();
            match runner.generate_with_skill_memory(&prompt, &generation, &mut loaded_skills) {
                Ok(result) => {
                    let messages = [ChatMessage::user(prompt)];
                    let _ =
                        append_chat_history(options, "ollama.generate", &model, &messages, &result);
                    json_response(serde_json::json!({
                        "model": model,
                        "created_at": iso_timestamp(),
                        "response": result.text,
                        "done": true,
                        "context": [],
                        "total_duration": started.elapsed().as_nanos(),
                        "load_duration": 0,
                        "prompt_eval_count": result.stats.prompt_tokens,
                        "prompt_eval_duration": result.stats.prefill_time.as_nanos(),
                        "eval_count": result.stats.generated_tokens,
                        "eval_duration": result.stats.decode_time.as_nanos(),
                    }))
                }
                Err(err) => (400, json_error(&err)),
            }
        }
        Err(err) => (400, json_error(&format!("Invalid JSON: {}", err))),
    }
}

/// Handles the Ollama-compatible chat route.
fn route_ollama_chat(
    body: &[u8],
    runner: &Runner,
    options: &ServeOptions,
    model_ids: &[String],
) -> (u16, String) {
    match serde_json::from_slice::<OllamaChatRequest>(body) {
        Ok(payload) => {
            let model = resolve_model(payload.model.as_deref(), model_ids);
            let messages = match parse_ollama_messages(payload.messages) {
                Ok(messages) => messages,
                Err(err) => return (400, json_error(&err)),
            };
            let mut generation = apply_ollama_options(&options.defaults, payload.options);
            apply_thinking_overrides(
                &mut generation,
                payload.thinking,
                payload.thinking_prompt,
                payload.thinking_max_tokens,
            );
            let started = Instant::now();
            let mut loaded_skills = HashSet::new();
            match runner.generate_chat_with_skill_memory(&messages, &generation, &mut loaded_skills)
            {
                Ok(result) => {
                    let _ = append_chat_history(options, "ollama.chat", &model, &messages, &result);
                    json_response(serde_json::json!({
                        "model": model,
                        "created_at": iso_timestamp(),
                        "message": {
                            "role": "assistant",
                            "content": result.text,
                        },
                        "done": true,
                        "total_duration": started.elapsed().as_nanos(),
                        "load_duration": 0,
                        "prompt_eval_count": result.stats.prompt_tokens,
                        "prompt_eval_duration": result.stats.prefill_time.as_nanos(),
                        "eval_count": result.stats.generated_tokens,
                        "eval_duration": result.stats.decode_time.as_nanos(),
                    }))
                }
                Err(err) => (400, json_error(&err)),
            }
        }
        Err(err) => (400, json_error(&format!("Invalid JSON: {}", err))),
    }
}

/// Handles the Ollama-compatible embedding routes.
fn route_ollama_embeddings(body: &[u8], runner: &Runner, model_ids: &[String]) -> (u16, String) {
    match serde_json::from_slice::<OllamaEmbeddingRequest>(body) {
        Ok(payload) => {
            let model = resolve_model(payload.model.as_deref(), model_ids);
            let inputs = if let Some(input) = payload.input {
                match input {
                    EmbeddingsInput::Single(text) => vec![text],
                    EmbeddingsInput::Batch(texts) => texts,
                }
            } else if let Some(prompt) = payload.prompt {
                vec![prompt]
            } else {
                return (400, json_error("Missing prompt or input."));
            };
            if inputs.is_empty() {
                return (400, json_error("input must not be empty."));
            }
            let mut embeddings = Vec::with_capacity(inputs.len());
            for input in &inputs {
                match runner.embed(input) {
                    Ok(result) => embeddings.push(result.embedding),
                    Err(err) => return (400, json_error(&err)),
                }
            }
            if embeddings.len() == 1 {
                json_response(serde_json::json!({
                    "model": model,
                    "embedding": embeddings.remove(0),
                }))
            } else {
                json_response(serde_json::json!({
                    "model": model,
                    "embeddings": embeddings,
                }))
            }
        }
        Err(err) => (400, json_error(&format!("Invalid JSON: {}", err))),
    }
}

/// Copies Ollama option fields into RustyLLM generation options.
fn apply_ollama_options(
    defaults: &GenerationOptions,
    options: Option<OllamaOptions>,
) -> GenerationOptions {
    let Some(options) = options else {
        return defaults.clone();
    };
    apply_generation_overrides(
        defaults,
        options.num_predict,
        options.temperature,
        options.top_p,
        options.top_k,
        options.repeat_penalty,
        options.seed,
        None,
        options.stop.map(|s| s.into_vec()),
        None,
        None,
        None,
    )
}

/// Converts Ollama chat messages into runtime chat messages.
fn parse_ollama_messages(messages: Vec<OllamaMessage>) -> Result<Vec<ChatMessage>, String> {
    messages
        .into_iter()
        .map(|message| match message.role.as_str() {
            "system" => Ok(ChatMessage {
                role: ChatRole::System,
                content: message.content,
            }),
            "user" => Ok(ChatMessage::user(message.content)),
            "assistant" => Ok(ChatMessage::assistant(message.content)),
            other => Err(format!("Unsupported role: {}", other)),
        })
        .collect()
}

/// Finds the most representative quantization format among loaded weights.
fn dominant_quantization(runner: &Runner) -> String {
    let mut counts = std::collections::BTreeMap::<String, usize>::new();
    for tensor in &runner.gguf().tensors {
        *counts.entry(format!("{:?}", tensor.dtype)).or_insert(0) += 1;
    }
    counts
        .into_iter()
        .max_by_key(|(_, count)| *count)
        .map(|(dtype, _)| dtype)
        .unwrap_or_else(|| String::from("unknown"))
}

/// Appends server chat history to a JSON history file.
fn append_chat_history(
    options: &ServeOptions,
    source: &str,
    model: &str,
    messages: &[ChatMessage],
    result: &crate::runtime::GenerationResult,
) -> Result<(), String> {
    append_chat_history_inner(
        options.chat_history_path.as_deref(),
        &options.chat_history_lock,
        source,
        model,
        messages,
        result,
    )
}

/// Appends one chat history entry to a path guarded by a shared lock.
fn append_chat_history_inner(
    path: Option<&str>,
    lock: &Arc<Mutex<()>>,
    source: &str,
    model: &str,
    messages: &[ChatMessage],
    result: &crate::runtime::GenerationResult,
) -> Result<(), String> {
    let Some(path) = path else {
        return Ok(());
    };
    let _guard = lock
        .lock()
        .map_err(|_| String::from("chat history lock poisoned"))?;
    let mut entries = match fs::read_to_string(path) {
        Ok(text) if !text.trim().is_empty() => {
            serde_json::from_str::<Vec<serde_json::Value>>(&text)
                .map_err(|err| format!("Failed to parse chat history {}: {}", path, err))?
        }
        Ok(_) => Vec::new(),
        Err(err) if err.kind() == io::ErrorKind::NotFound => Vec::new(),
        Err(err) => return Err(format!("Failed to read chat history {}: {}", path, err)),
    };
    entries.push(serde_json::json!({
        "timestamp": unix_timestamp(),
        "source": source,
        "model": model,
        "messages": messages.iter().map(history_message_json).collect::<Vec<_>>(),
        "response": result.text,
        "usage": {
            "prompt_tokens": result.stats.prompt_tokens,
            "completion_tokens": result.stats.generated_tokens,
            "total_tokens": result.stats.prompt_tokens + result.stats.generated_tokens,
        },
        "timings_ms": {
            "prefill": result.stats.prefill_time.as_millis(),
            "decode": result.stats.decode_time.as_millis(),
            "total": result.stats.total_time.as_millis(),
        }
    }));
    if let Some(parent) = std::path::Path::new(path).parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .map_err(|err| format!("Failed to create {}: {}", parent.display(), err))?;
        }
    }
    let body = serde_json::to_string_pretty(&entries)
        .map_err(|err| format!("Failed to serialize chat history: {}", err))?;
    fs::write(path, body).map_err(|err| format!("Failed to write chat history {}: {}", path, err))
}

/// Serializes one chat message for the history log.
fn history_message_json(message: &ChatMessage) -> serde_json::Value {
    let role = match message.role {
        ChatRole::System => "system",
        ChatRole::User => "user",
        ChatRole::Assistant => "assistant",
    };
    serde_json::json!({ "role": role, "content": message.content })
}

/// Writes one named Server-Sent Event with a JSON body.
fn write_sse_event<T: Serialize, W: Write>(
    stream: &mut W,
    event: &str,
    data: &T,
) -> io::Result<()> {
    let body = serde_json::to_string(data)
        .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?;
    write!(stream, "event: {}\ndata: {}\n\n", event, body)?;
    stream.flush()
}

/// Writes HTTP headers for an Ollama-style NDJSON stream.
fn write_ndjson_response_header<W: Write>(stream: &mut W) -> io::Result<()> {
    write!(
        stream,
        "HTTP/1.1 200 OK\r\nContent-Type: application/x-ndjson\r\nCache-Control: no-cache\r\nConnection: close\r\nAccess-Control-Allow-Origin: *\r\n\r\n"
    )?;
    stream.flush()
}

/// Writes one JSON value followed by a newline.
fn write_json_line<T: Serialize, W: Write>(stream: &mut W, data: &T) -> io::Result<()> {
    let body = serde_json::to_string(data)
        .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?;
    writeln!(stream, "{}", body)?;
    stream.flush()
}

/// Serializes a response object with a JSON content type.
fn json_response<T: Serialize>(response: T) -> (u16, String) {
    match serde_json::to_string(&response) {
        Ok(body) => (200, body),
        Err(err) => (500, json_error(&format!("Serialize error: {}", err))),
    }
}

/// Checks whether a Content-Type header represents JSON.
fn is_json_content_type(content_type: &str) -> bool {
    content_type
        .split(';')
        .next()
        .map(|mime| mime.trim().eq_ignore_ascii_case("application/json"))
        .unwrap_or(false)
}

/// Applies request-provided sampling and length overrides.
fn apply_generation_overrides(
    defaults: &GenerationOptions,
    max_tokens: Option<usize>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<usize>,
    repeat_penalty: Option<f32>,
    seed: Option<u64>,
    system_prompt: Option<String>,
    stop_sequences: Option<Vec<String>>,
    thinking: Option<bool>,
    thinking_prompt: Option<String>,
    thinking_max_tokens: Option<usize>,
) -> GenerationOptions {
    let mut generation = defaults.clone();
    if let Some(max_tokens) = max_tokens {
        generation.max_tokens = max_tokens;
    }
    if let Some(temperature) = temperature {
        generation.sampler.temperature = temperature;
    }
    if let Some(top_p) = top_p {
        generation.sampler.top_p = top_p;
    }
    if let Some(top_k) = top_k {
        generation.sampler.top_k = top_k;
    }
    if let Some(repeat_penalty) = repeat_penalty {
        generation.sampler.repeat_penalty = repeat_penalty;
    }
    if let Some(seed) = seed {
        generation.seed = seed;
    }
    if let Some(system_prompt) = system_prompt {
        generation.system_prompt = system_prompt;
    }
    if let Some(stop) = stop_sequences {
        generation.stop_sequences = stop;
    }
    if let Some(enabled) = thinking {
        generation.thinking.enabled = enabled;
    }
    if let Some(prompt) = thinking_prompt {
        generation.thinking.system_prompt = prompt;
    }
    if let Some(max_tokens) = thinking_max_tokens {
        generation.thinking.max_tokens = max_tokens;
    }
    generation
}

fn apply_thinking_overrides(
    generation: &mut GenerationOptions,
    thinking: Option<bool>,
    thinking_prompt: Option<String>,
    thinking_max_tokens: Option<usize>,
) {
    if let Some(enabled) = thinking {
        generation.thinking.enabled = enabled;
    }
    if let Some(prompt) = thinking_prompt {
        generation.thinking.system_prompt = prompt;
    }
    if let Some(max_tokens) = thinking_max_tokens {
        generation.thinking.max_tokens = max_tokens;
    }
}

/// Adds a prompt-level JSON-output hint for OpenAI response-format requests.
///
/// This is compatibility, not grammar-constrained decoding: RustyLLM accepts the
/// standard API fields and asks the model for JSON, but the current sampler does
/// not enforce a JSON schema token by token.
fn apply_response_format_hint(
    generation: &mut GenerationOptions,
    format: Option<&serde_json::Value>,
) {
    let Some(format) = format else {
        return;
    };
    let format_type = format.get("type").and_then(|v| v.as_str()).unwrap_or("");
    if format_type != "json_object" && format_type != "json_schema" {
        return;
    }

    let mut hint = String::from(
        " Respond with valid JSON only. Do not wrap the JSON in markdown or explanatory text.",
    );
    if format_type == "json_schema" {
        let schema = format
            .get("json_schema")
            .or_else(|| format.get("schema"))
            .and_then(|schema| serde_json::to_string(schema).ok());
        if let Some(schema) = schema {
            hint.push_str(" Match this JSON schema as closely as possible: ");
            hint.push_str(&schema);
        }
    }
    if !generation.system_prompt.ends_with(' ') {
        generation.system_prompt.push(' ');
    }
    generation.system_prompt.push_str(&hint);
}

/// Builds a Responses API object from a completed generation result.
fn make_openai_response(
    model: &str,
    result: crate::runtime::GenerationResult,
) -> OpenAiResponsesResponse {
    let created = unix_timestamp();
    let text = result.text;
    OpenAiResponsesResponse {
        id: format!("resp-rustyllm-{}", created),
        object: "response",
        created_at: created,
        status: "completed",
        model: model.to_string(),
        output: vec![OpenAiResponsesOutputMessage {
            r#type: "message",
            id: format!("msg-rustyllm-{}", created),
            status: "completed",
            role: "assistant",
            content: vec![OpenAiResponsesOutputText {
                r#type: "output_text",
                text: text.clone(),
                annotations: Vec::new(),
            }],
        }],
        output_text: text,
        usage: OpenAiResponsesUsage {
            input_tokens: result.stats.prompt_tokens,
            output_tokens: result.stats.generated_tokens,
            total_tokens: result.stats.prompt_tokens + result.stats.generated_tokens,
        },
        error: None,
        incomplete_details: None,
        parallel_tool_calls: false,
    }
}

/// Converts the flexible Responses API `input` field into runtime messages.
fn parse_responses_input(input: Option<&serde_json::Value>) -> Result<Vec<ChatMessage>, String> {
    let Some(input) = input else {
        return Err(String::from("Missing input."));
    };
    match input {
        serde_json::Value::String(text) => Ok(vec![ChatMessage::user(text.clone())]),
        serde_json::Value::Array(items) => {
            let mut messages = Vec::new();
            let mut loose_text = Vec::new();
            for item in items {
                if let Some(message) = parse_response_message_item(item)? {
                    messages.push(message);
                } else if let Some(text) = response_value_text(item) {
                    loose_text.push(text);
                }
            }
            if !loose_text.is_empty() {
                messages.push(ChatMessage::user(loose_text.join("\n")));
            }
            if messages.is_empty() {
                Err(String::from("input array did not contain any text."))
            } else {
                Ok(messages)
            }
        }
        serde_json::Value::Object(_) => {
            if let Some(message) = parse_response_message_item(input)? {
                Ok(vec![message])
            } else if let Some(text) = response_value_text(input) {
                Ok(vec![ChatMessage::user(text)])
            } else {
                Err(String::from("input object did not contain any text."))
            }
        }
        _ => Err(String::from("input must be a string, object, or array.")),
    }
}

/// Parses one Responses API message-like item.
fn parse_response_message_item(value: &serde_json::Value) -> Result<Option<ChatMessage>, String> {
    let serde_json::Value::Object(obj) = value else {
        return Ok(None);
    };
    let item_type = obj.get("type").and_then(|v| v.as_str()).unwrap_or("");
    let role = obj.get("role").and_then(|v| v.as_str());
    if role.is_none() && item_type != "message" {
        return Ok(None);
    }
    let role = role.unwrap_or("user");
    let content = obj
        .get("content")
        .and_then(response_value_text)
        .or_else(|| obj.get("text").and_then(response_value_text))
        .or_else(|| obj.get("input_text").and_then(response_value_text))
        .unwrap_or_default();
    if content.is_empty() {
        return Ok(None);
    }
    match role {
        "system" | "developer" => Ok(Some(ChatMessage {
            role: ChatRole::System,
            content,
        })),
        "user" => Ok(Some(ChatMessage::user(content))),
        "assistant" => Ok(Some(ChatMessage::assistant(content))),
        other => Err(format!("Unsupported role: {}", other)),
    }
}

/// Extracts user-visible text from OpenAI-style content values.
fn response_value_text(value: &serde_json::Value) -> Option<String> {
    match value {
        serde_json::Value::String(text) => Some(text.clone()),
        serde_json::Value::Array(items) => {
            let parts = items
                .iter()
                .filter_map(response_value_text)
                .filter(|text| !text.is_empty())
                .collect::<Vec<_>>();
            if parts.is_empty() {
                None
            } else {
                Some(parts.join("\n"))
            }
        }
        serde_json::Value::Object(obj) => {
            if let Some(text) = obj.get("text").and_then(|v| v.as_str()) {
                return Some(text.to_string());
            }
            if let Some(text) = obj.get("input_text").and_then(|v| v.as_str()) {
                return Some(text.to_string());
            }
            if let Some(text) = obj.get("output_text").and_then(|v| v.as_str()) {
                return Some(text.to_string());
            }
            if let Some(image_url) = obj.get("image_url") {
                if let Some(url) = image_url.get("url").and_then(|v| v.as_str()) {
                    return Some(format!("[image: {}]", redact_data_image_url(url)));
                }
                if let Some(url) = image_url.as_str() {
                    return Some(format!("[image: {}]", redact_data_image_url(url)));
                }
            }
            if let Some(url) = obj.get("url").and_then(|v| v.as_str()) {
                let typ = obj.get("type").and_then(|v| v.as_str()).unwrap_or("image");
                if typ.contains("image") {
                    return Some(format!("[image: {}]", redact_data_image_url(url)));
                }
            }
            None
        }
        _ => None,
    }
}

/// Avoids echoing large base64 image payloads into prompts.
fn redact_data_image_url(url: &str) -> String {
    if url.starts_with("data:image/") {
        String::from("base64 data")
    } else {
        url.to_string()
    }
}

/// Converts OpenAI-compatible messages into runtime chat messages.
fn parse_api_messages(messages: Vec<ApiMessage>) -> Result<Vec<ChatMessage>, String> {
    messages
        .into_iter()
        .map(|message| {
            let content = message.content.into_text();
            match message.role.as_str() {
                "system" => Ok(ChatMessage {
                    role: ChatRole::System,
                    content,
                }),
                "user" => Ok(ChatMessage::user(content)),
                "assistant" => Ok(ChatMessage::assistant(content)),
                other => Err(format!("Unsupported role: {}", other)),
            }
        })
        .collect()
}

/// Resolve the requested model ID to a canonical name.
///
/// For single-model deployments (the most common case) we accept any model ID
/// the caller provides — it is silently mapped to the loaded model.  This
/// avoids breaking RAG pipelines that hard-code an arbitrary model name.
fn resolve_model(requested: Option<&str>, model_ids: &[String]) -> String {
    match model_ids.first() {
        None => String::from("unknown"),
        Some(default) => match requested {
            None => default.clone(),
            Some(model) if model_ids.iter().any(|c| c == model) => model.to_string(),
            // Unknown model name — fall back to the loaded model and warn so
            // operators can detect misconfigured clients.
            Some(model) => {
                eprintln!(
                    "Warning: requested model '{}' not found; using '{}' instead.",
                    model, default
                );
                default.clone()
            }
        },
    }
}

/// Builds the model IDs and aliases exposed by compatibility APIs.
fn advertised_model_ids(runner: &Runner) -> Vec<String> {
    let mut ids = Vec::new();
    if let Some(model_name) = runner.model_name() {
        let trimmed = model_name.trim();
        if !trimmed.is_empty() {
            ids.push(trimmed.to_string());
        }
    }
    ids.push(runner.architecture().to_string());
    ids.extend(
        model_aliases_for_arch(runner.architecture())
            .iter()
            .map(|id| id.to_string()),
    );

    let mut seen = HashSet::new();
    ids.retain(|id| seen.insert(id.clone()));
    ids
}

/// Returns compatibility aliases for a model architecture.
fn model_aliases_for_arch(arch: &str) -> &'static [&'static str] {
    match arch {
        "llama" | "llama2" | "llama3" => &["llama", "llama2", "llama3"],
        "mistral" | "mixtral" | "ministral" => &["mistral", "mixtral", "ministral"],
        "qwen2" | "qwen3" => &["qwen2", "qwen2.5", "qwen3"],
        "gpt-oss" => &["gpt-oss"],
        "gemma" | "gemma2" | "gemma3" | "gemma4" | "gemma4n" | "gemma4-assistant" => &[
            "gemma",
            "gemma2",
            "gemma3",
            "gemma4",
            "gemma4n",
            "gemma4-assistant",
            "google/gemma-4-12b-qat",
        ],
        "deepseek" | "deepseek-v2" | "deepseek2" => &["deepseek", "deepseek-v2", "deepseek2"],
        "nemotron" => &["nemotron"],
        "hermes" => &["hermes"],
        "phi" | "phi2" | "phi3" | "phi4" => &["phi", "phi2", "phi3", "phi4"],
        "falcon" | "falcon3" => &["falcon", "falcon3"],
        "stablelm" => &["stablelm"],
        "starcoder2" => &["starcoder2"],
        "command-r" | "cohere" => &["command-r", "cohere"],
        "internlm2" => &["internlm2"],
        "olmo" | "olmo2" => &["olmo", "olmo2"],
        "exaone" => &["exaone"],
        "solar" => &["solar"],
        "yi" => &["yi"],
        "arctic" => &["arctic"],
        "nomic-bert" | "nomic-embed" | "text-embedding-nomic-embed-text" => &[
            "nomic-bert",
            "nomic-embed",
            "text-embedding-nomic-embed-text",
        ],
        _ => &[],
    }
}

/// Returns the current Unix timestamp in seconds.
fn unix_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

/// Formats the current timestamp for lightweight logs.
fn iso_timestamp() -> String {
    format!("{}Z", unix_timestamp())
}

/// Formats an error message as a JSON object.
fn json_error(message: &str) -> String {
    serde_json::json!({ "error": message }).to_string()
}

#[derive(Debug)]
struct HttpRequest {
    method: String,
    path: String,
    content_type: Option<String>,
    body: Vec<u8>,
}

/// Reads and validates one bounded HTTP request from a stream.
fn read_http_request<T>(stream: &mut T) -> Result<HttpRequest, HttpError>
where
    T: Read,
{
    let mut header_bytes = Vec::new();
    let mut one = [0u8; 1];
    // This parser intentionally supports only small HTTP/1.1 requests with a
    // Content-Length body; it is enough for the local inference API.
    loop {
        match stream.read(&mut one) {
            Ok(0) => {
                return Err(HttpError::bad_request(
                    "Connection closed before request headers.",
                ));
            }
            Ok(_) => {
                header_bytes.push(one[0]);
                if header_bytes.ends_with(b"\r\n\r\n") {
                    break;
                }
                if header_bytes.len() > MAX_HEADER_BYTES {
                    return Err(HttpError::payload_too_large("HTTP header too large."));
                }
            }
            Err(err) if err.kind() == io::ErrorKind::TimedOut => {
                return Err(HttpError::request_timeout("Request timed out."));
            }
            Err(err) if err.kind() == io::ErrorKind::WouldBlock => {
                return Err(HttpError::request_timeout("Request timed out."));
            }
            Err(err) => {
                return Err(HttpError::bad_request(format!(
                    "Failed to read request header: {}",
                    err
                )));
            }
        }
    }

    let header_text = String::from_utf8_lossy(&header_bytes);
    let mut lines = header_text.split("\r\n");
    let request_line = lines
        .next()
        .ok_or_else(|| HttpError::bad_request("Missing request line."))?;
    let parts: Vec<&str> = request_line.split_whitespace().collect();
    if parts.len() != 3 {
        return Err(HttpError::bad_request("Malformed request line."));
    }
    let method = parts[0].to_uppercase();
    let path = parts[1].to_string();
    if !path.starts_with('/') {
        return Err(HttpError::bad_request("Malformed request path."));
    }

    let mut content_length = 0usize;
    let mut content_type = None::<String>;
    for line in lines {
        if line.is_empty() {
            continue;
        }
        let Some((name, value)) = line.split_once(':') else {
            return Err(HttpError::bad_request("Malformed header line."));
        };
        let name = name.trim();
        let value = value.trim();
        if name.eq_ignore_ascii_case("content-length") {
            content_length = value.parse::<usize>().map_err(|_| {
                HttpError::bad_request(format!("Invalid Content-Length value: {}", value))
            })?;
        } else if name.eq_ignore_ascii_case("content-type") {
            content_type = Some(value.to_string());
        }
    }

    if content_length > MAX_BODY_BYTES {
        return Err(HttpError::payload_too_large(format!(
            "Request body too large (max {} bytes).",
            MAX_BODY_BYTES
        )));
    }

    let mut body = vec![0u8; content_length];
    if content_length > 0 {
        match stream.read_exact(&mut body) {
            Ok(()) => {}
            Err(err) if err.kind() == io::ErrorKind::TimedOut => {
                return Err(HttpError::request_timeout(
                    "Timed out reading request body.",
                ));
            }
            Err(err) if err.kind() == io::ErrorKind::WouldBlock => {
                return Err(HttpError::request_timeout(
                    "Timed out reading request body.",
                ));
            }
            Err(err) => {
                return Err(HttpError::bad_request(format!(
                    "Failed to read request body: {}",
                    err
                )));
            }
        }
    }

    Ok(HttpRequest {
        method,
        path,
        content_type,
        body,
    })
}

/// Returns the embedded default chat UI HTML.
fn chat_ui_html() -> &'static str {
    include_str!("web_ui/chat.html")
}

/// Returns the embedded expert chat UI HTML.
fn expert_chat_ui_html() -> &'static str {
    include_str!("web_ui/expert.html")
}

/// Returns the embedded GGUF explorer UI HTML.
fn explorer_ui_html() -> &'static str {
    include_str!("web_ui/explorer.html")
}

/// Returns an embedded Swagger UI page backed by `/openapi.json`.
fn swagger_ui_html() -> &'static str {
    r#"<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RustyLLM API Docs</title>
  <link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist@5/swagger-ui.css">
  <style>
    body { margin: 0; background: #f7f7f7; }
    header { padding: 10px 16px; background: #1f2933; color: white; font: 14px system-ui, sans-serif; }
    header a { color: #b9e0ff; }
    #swagger-ui { max-width: 1320px; margin: 0 auto; }
  </style>
</head>
<body>
  <header>RustyLLM API Docs · Raw OpenAPI: <a href="/openapi.json">/openapi.json</a></header>
  <div id="swagger-ui"></div>
  <script src="https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
  <script>
    window.addEventListener('load', () => {
      if (!window.SwaggerUIBundle) {
        document.getElementById('swagger-ui').textContent = 'Swagger UI assets failed to load. Open /openapi.json for the raw document.';
        return;
      }
      SwaggerUIBundle({
        url: '/openapi.json',
        dom_id: '#swagger-ui',
        presets: [SwaggerUIBundle.presets.apis],
        layout: 'BaseLayout',
        tryItOutEnabled: true
      });
    });
  </script>
</body>
</html>"#
}

/// Writes a standard HTML response.
fn write_http_response<T>(stream: &mut T, status: u16, body: &str) -> io::Result<()>
where
    T: Write,
{
    write_http_response_with_content_type(stream, status, "application/json", body)
}

/// Writes an HTTP response with explicit type, length, and CORS headers.
fn write_http_response_with_content_type<T>(
    stream: &mut T,
    status: u16,
    content_type: &str,
    body: &str,
) -> io::Result<()>
where
    T: Write,
{
    let status_text = match status {
        200 => "OK",
        204 => "No Content",
        400 => "Bad Request",
        404 => "Not Found",
        405 => "Method Not Allowed",
        408 => "Request Timeout",
        413 => "Payload Too Large",
        415 => "Unsupported Media Type",
        500 => "Internal Server Error",
        503 => "Service Unavailable",
        _ => "Internal Server Error",
    };
    write!(
        stream,
        "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Methods: GET,POST,OPTIONS\r\nAccess-Control-Allow-Headers: Content-Type\r\nX-Content-Type-Options: nosniff\r\n\r\n{}",
        status,
        status_text,
        content_type,
        body.len(),
        body
    )?;
    stream.flush()
}

#[cfg(feature = "tls")]
/// Loads TLS certificate and private key files into a rustls server config.
fn load_tls_config(cert_path: &str, key_path: &str) -> Result<ServerConfig, String> {
    let mut cert_reader = BufReader::new(File::open(cert_path).map_err(|err| err.to_string())?);
    let certs: Vec<CertificateDer<'static>> = rustls_pemfile::certs(&mut cert_reader)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|err| err.to_string())?;
    if certs.is_empty() {
        return Err(String::from(
            "No certificates found in TLS certificate file.",
        ));
    }

    let mut key_reader = BufReader::new(File::open(key_path).map_err(|err| err.to_string())?);
    let mut pkcs8_keys = rustls_pemfile::pkcs8_private_keys(&mut key_reader);
    let Some(key) = pkcs8_keys
        .next()
        .transpose()
        .map_err(|err| err.to_string())?
    else {
        return Err(String::from("No PKCS#8 private key found."));
    };
    let key: PrivateKeyDer<'static> = key.into();

    ServerConfig::builder()
        .with_no_client_auth()
        .with_single_cert(certs, key)
        .map_err(|err| err.to_string())
}

#[cfg(test)]
mod tests {
    use super::{
        MAX_BODY_BYTES, mcp_initialize_result, mcp_tools_list, parse_responses_input,
        read_http_request, write_http_response,
    };
    use crate::runtime::ChatRole;
    use std::io::Cursor;

    #[test]
    /// Verifies that JSON content types with parameters are parsed as JSON requests.
    fn read_http_request_parses_json_content_type() {
        let req = b"POST /generate HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json; charset=utf-8\r\nContent-Length: 2\r\n\r\n{}";
        let mut cursor = Cursor::new(req.as_slice());
        let parsed = read_http_request(&mut cursor).expect("request should parse");
        assert_eq!(parsed.method, "POST");
        assert_eq!(parsed.path, "/generate");
        assert_eq!(
            parsed.content_type.as_deref(),
            Some("application/json; charset=utf-8")
        );
        assert_eq!(parsed.body, b"{}");
    }

    #[test]
    /// Verifies that invalid Content-Length values produce a bad-request error.
    fn read_http_request_rejects_invalid_content_length() {
        let req = b"POST /generate HTTP/1.1\r\nContent-Length: nope\r\n\r\n";
        let mut cursor = Cursor::new(req.as_slice());
        let err = read_http_request(&mut cursor).expect_err("request should fail");
        assert_eq!(err.status, 400);
    }

    #[test]
    /// Verifies that oversized request bodies are rejected before allocation.
    fn read_http_request_rejects_oversized_body() {
        let req = format!(
            "POST /generate HTTP/1.1\r\nContent-Length: {}\r\n\r\n",
            MAX_BODY_BYTES + 1
        );
        let mut cursor = Cursor::new(req.into_bytes());
        let err = read_http_request(&mut cursor).expect_err("request should fail");
        assert_eq!(err.status, 413);
    }

    #[test]
    /// Verifies that normal responses include permissive CORS headers.
    fn write_http_response_includes_cors_headers() {
        let mut out = Vec::new();
        write_http_response(&mut out, 200, "{\"status\":\"ok\"}").expect("response write");
        let text = String::from_utf8(out).expect("valid UTF-8 response");
        assert!(text.contains("Access-Control-Allow-Origin: *"));
        assert!(text.contains("Access-Control-Allow-Methods: GET,POST,OPTIONS"));
        assert!(text.contains("X-Content-Type-Options: nosniff"));
    }

    #[test]
    /// Verifies that Responses API string input maps to a user message.
    fn parse_responses_input_accepts_plain_string() {
        let input = serde_json::json!("Explain GGUF.");
        let messages = parse_responses_input(Some(&input)).expect("input should parse");
        assert_eq!(messages.len(), 1);
        assert_eq!(role_name(&messages[0].role), "user");
        assert_eq!(messages[0].content, "Explain GGUF.");
    }

    #[test]
    /// Verifies that Responses API message arrays preserve role and content text.
    fn parse_responses_input_accepts_message_array() {
        let input = serde_json::json!([
            {
                "type": "message",
                "role": "system",
                "content": [{"type": "input_text", "text": "Be terse."}]
            },
            {
                "role": "user",
                "content": "What is memory mapping?"
            }
        ]);
        let messages = parse_responses_input(Some(&input)).expect("input should parse");
        assert_eq!(messages.len(), 2);
        assert_eq!(role_name(&messages[0].role), "system");
        assert_eq!(messages[0].content, "Be terse.");
        assert_eq!(role_name(&messages[1].role), "user");
    }

    #[test]
    /// Verifies that MCP initialize echoes the requested protocol version.
    fn mcp_initialize_result_echoes_protocol_version() {
        let result = mcp_initialize_result(&serde_json::json!({
            "protocolVersion": "2025-06-18"
        }));
        assert_eq!(result["protocolVersion"], "2025-06-18");
        assert_eq!(result["serverInfo"]["name"], "rusty-llm");
        assert!(result["capabilities"]["tools"].is_object());
    }

    #[test]
    /// Verifies that the MCP tool list advertises the core local-model tools.
    fn mcp_tools_list_includes_core_tools() {
        let tools = mcp_tools_list();
        let names = tools["tools"]
            .as_array()
            .expect("tools array")
            .iter()
            .filter_map(|tool| tool["name"].as_str())
            .collect::<Vec<_>>();
        assert!(names.contains(&"generate"));
        assert!(names.contains(&"chat"));
        assert!(names.contains(&"embed"));
        assert!(names.contains(&"models"));
    }

    fn role_name(role: &ChatRole) -> &'static str {
        match role {
            ChatRole::System => "system",
            ChatRole::User => "user",
            ChatRole::Assistant => "assistant",
        }
    }
}
