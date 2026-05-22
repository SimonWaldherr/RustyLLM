#![allow(clippy::too_many_arguments)]

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
use std::io::{self, Read, Write};
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
    pub chat_history_path: Option<String>,
    pub chat_history_lock: Arc<Mutex<()>>,
    /// Session store for persistent KV-cache conversations.
    /// `None` when session caching is disabled (`--max-sessions 0`).
    pub session_store: Option<Arc<SessionStore>>,
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
    stop: Option<StopSpec>,
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
    stop: Option<StopSpec>,
    /// Optional conversation ID for persistent KV-cache sessions.
    conversation_id: Option<String>,
    /// When `true` (and `conversation_id` is set), use the persistent session.
    cache_prompt: Option<bool>,
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
    stream: Option<bool>,
    options: Option<OllamaOptions>,
    stop: Option<StopSpec>,
}

#[derive(Deserialize)]
struct OllamaChatRequest {
    model: Option<String>,
    messages: Vec<OllamaMessage>,
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

    if request.method == "GET" && chat_ui_route(&request.path).is_some() {
        if options.chat_ui {
            let body = match chat_ui_route(&request.path) {
                Some(ChatUiRoute::Simple) => chat_ui_html(),
                Some(ChatUiRoute::Expert) => expert_chat_ui_html(),
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
            _ => {}
        }
    }

    let (status, body) = route_request(&request, &runner, options);
    write_http_response(&mut stream, status, &body).map_err(|err| err.to_string())
}

enum ChatUiRoute {
    Simple,
    Expert,
}

/// Maps a request path to one of the embedded chat UI assets.
fn chat_ui_route(path: &str) -> Option<ChatUiRoute> {
    match path {
        "/chat" => Some(ChatUiRoute::Simple),
        "/chat?expert" => Some(ChatUiRoute::Expert),
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
            | "/api/v0/chat/completions"
            | "/api/v0/completions"
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
                let generation_options = apply_generation_overrides(
                    &options.defaults,
                    max_tok,
                    payload.temperature,
                    payload.top_p,
                    payload.top_k,
                    payload.repeat_penalty,
                    payload.seed,
                    payload.system_prompt,
                    payload.stop.map(|s| s.into_vec()),
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
                let generation_options = apply_generation_overrides(
                    &options.defaults,
                    max_tok,
                    payload.temperature,
                    payload.top_p,
                    payload.top_k,
                    payload.repeat_penalty,
                    payload.seed,
                    payload.system_prompt,
                    payload.stop.map(|s| s.into_vec()),
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
                    | "/v1/embeddings"
                    | "/api/v0/completions"
                    | "/api/v0/chat/completions"
                    | "/api/v0/embeddings"
                    | "/api/generate"
                    | "/api/chat"
                    | "/api/embeddings"
                    | "/api/embed"
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
                _ => (404, json_error("Not found")),
            }
        }
        ("GET", _) | ("POST", _) => (404, json_error("Not found")),
        _ => (405, json_error("Method not allowed.")),
    }
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
            return runner.generate_chat_with_session(messages, generation, &mut session, on_token);
        }
    }
    // Stateless fallback.
    runner.generate_chat_stream(messages, generation, on_token)
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
                    .map(|r| {
                        history_messages = messages;
                        r
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
            let generation = apply_generation_overrides(
                &options.defaults,
                max_tokens,
                payload.temperature,
                payload.top_p,
                payload.top_k,
                payload.repeat_penalty,
                payload.seed,
                payload.system_prompt,
                payload.stop.map(|s| s.into_vec()),
            );
            match runner.generate(&prompt, &generation) {
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
            let generation = apply_generation_overrides(
                &options.defaults,
                max_tokens,
                payload.temperature,
                payload.top_p,
                payload.top_k,
                payload.repeat_penalty,
                payload.seed,
                payload.system_prompt,
                payload.stop.map(|s| s.into_vec()),
            );
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
            if payload.stream.unwrap_or(false) {
                eprintln!(
                    "Warning: Ollama stream=true requested; returning a single final JSON response."
                );
            }
            let started = Instant::now();
            match runner.generate(&prompt, &generation) {
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
            let generation = apply_ollama_options(&options.defaults, payload.options);
            if payload.stream.unwrap_or(false) {
                eprintln!(
                    "Warning: Ollama stream=true requested; returning a single final JSON response."
                );
            }
            let started = Instant::now();
            match runner.generate_chat(&messages, &generation) {
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
    let Some(path) = options.chat_history_path.as_deref() else {
        return Ok(());
    };
    let _guard = options
        .chat_history_lock
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
    generation
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
        "gemma" | "gemma2" | "gemma4" => &["gemma", "gemma2", "gemma4"],
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
    use super::{MAX_BODY_BYTES, read_http_request, write_http_response};
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
}
