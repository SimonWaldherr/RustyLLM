use crate::runtime::{ChatMessage, ChatRole, GenerationOptions, Runner};
use rustls::pki_types::{CertificateDer, PrivateKeyDer};
use rustls::{ServerConfig, ServerConnection, StreamOwned};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{self, BufReader, Read, Write};
use std::net::{Shutdown, TcpListener, TcpStream};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

const MAX_HEADER_BYTES: usize = 16 * 1024;
const MAX_BODY_BYTES: usize = 2 * 1024 * 1024;
const IO_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Clone)]
pub struct ServeOptions {
    pub addr: String,
    pub defaults: GenerationOptions,
    pub tls_cert_path: Option<String>,
    pub tls_key_path: Option<String>,
    pub max_concurrent_connections: usize,
}

impl ServeOptions {
    pub fn is_tls(&self) -> bool {
        self.tls_cert_path.is_some() && self.tls_key_path.is_some()
    }
}

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
}

#[derive(Deserialize)]
struct ApiMessage {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct GenerateResponse<'a> {
    text: &'a str,
    prompt_tokens: usize,
    generated_tokens: usize,
    prefill_ms: u128,
    decode_ms: u128,
    total_ms: u128,
}

struct ActiveConnectionGuard {
    active_connections: Arc<AtomicUsize>,
}

impl Drop for ActiveConnectionGuard {
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
    fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: 400,
            message: message.into(),
        }
    }

    fn payload_too_large(message: impl Into<String>) -> Self {
        Self {
            status: 413,
            message: message.into(),
        }
    }

    fn request_timeout(message: impl Into<String>) -> Self {
        Self {
            status: 408,
            message: message.into(),
        }
    }
}

pub fn serve(runner: Arc<Runner>, options: ServeOptions) -> Result<(), String> {
    let listener = TcpListener::bind(&options.addr)
        .map_err(|err| format!("Failed to bind {}: {}", options.addr, err))?;
    let max_connections = options.max_concurrent_connections.max(1);
    let active_connections = Arc::new(AtomicUsize::new(0));

    // Keep the server loop deliberately small: accept a connection, hand it to
    // a worker thread, and let the handler own the request lifecycle.
    if options.is_tls() {
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

fn handle_plain_connection(
    stream: TcpStream,
    runner: Arc<Runner>,
    options: &ServeOptions,
) -> Result<(), String> {
    handle_connection(stream, runner, options)
}

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
    let (status, body) = route_request(&request, &runner, options);
    write_http_response(&mut stream, status, &body).map_err(|err| err.to_string())
}

fn route_request(request: &HttpRequest, runner: &Runner, options: &ServeOptions) -> (u16, String) {
    if request.method == "OPTIONS" {
        return (204, String::new());
    }

    match request.path.as_str() {
        "/health" => {
            if request.method != "GET" {
                return (405, json_error("Method not allowed."));
            }
            (200, String::from("{\"status\":\"ok\"}"))
        }
        "/generate" => {
            if request.method != "POST" {
                return (405, json_error("Method not allowed."));
            }
            if !request
                .content_type
                .as_deref()
                .map(is_json_content_type)
                .unwrap_or(false)
            {
                return (415, json_error("Content-Type must be application/json."));
            }

            match serde_json::from_slice::<GenerateRequest>(&request.body) {
                Ok(payload) => {
                    let mut generation = options.defaults.clone();
                    if let Some(max_tokens) = payload.max_tokens {
                        generation.max_tokens = max_tokens;
                    }
                    if let Some(temp) = payload.temp {
                        generation.sampler.temperature = temp;
                    }
                    if let Some(top_p) = payload.top_p {
                        generation.sampler.top_p = top_p;
                    }
                    if let Some(top_k) = payload.top_k {
                        generation.sampler.top_k = top_k;
                    }
                    if let Some(repeat_penalty) = payload.repeat_penalty {
                        generation.sampler.repeat_penalty = repeat_penalty;
                    }
                    if let Some(seed) = payload.seed {
                        generation.seed = seed;
                    }
                    if let Some(system_prompt) = payload.system_prompt {
                        generation.system_prompt = system_prompt;
                    }

                    if let Err(err) = generation.validate() {
                        return (400, json_error(&err));
                    }

                    let result = if let Some(messages) = payload.messages {
                        let messages: Result<Vec<ChatMessage>, String> = messages
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
                            .collect();
                        match messages {
                            Ok(messages) => runner.generate_chat(&messages, &generation),
                            Err(err) => Err(err),
                        }
                    } else if let Some(prompt) = payload.prompt {
                        runner.generate(&prompt, &generation)
                    } else {
                        Err(String::from("Missing prompt or messages."))
                    };

                    match result {
                        Ok(result) => {
                            let response = GenerateResponse {
                                text: &result.text,
                                prompt_tokens: result.stats.prompt_tokens,
                                generated_tokens: result.stats.generated_tokens,
                                prefill_ms: result.stats.prefill_time.as_millis(),
                                decode_ms: result.stats.decode_time.as_millis(),
                                total_ms: result.stats.total_time.as_millis(),
                            };
                            match serde_json::to_string(&response) {
                                Ok(body) => (200, body),
                                Err(err) => (500, json_error(&format!("Serialize error: {}", err))),
                            }
                        }
                        Err(err) => (400, json_error(&err)),
                    }
                }
                Err(err) => (400, json_error(&format!("Invalid JSON: {}", err))),
            }
        }
        _ => (404, json_error("Not found")),
    }
}

fn is_json_content_type(content_type: &str) -> bool {
    content_type
        .split(';')
        .next()
        .map(|mime| mime.trim().eq_ignore_ascii_case("application/json"))
        .unwrap_or(false)
}

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
                ))
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

fn write_http_response<T>(stream: &mut T, status: u16, body: &str) -> io::Result<()>
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
        "HTTP/1.1 {} {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Methods: GET,POST,OPTIONS\r\nAccess-Control-Allow-Headers: Content-Type\r\nX-Content-Type-Options: nosniff\r\n\r\n{}",
        status,
        status_text,
        body.len(),
        body
    )?;
    stream.flush()
}

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
    use super::{read_http_request, write_http_response, MAX_BODY_BYTES};
    use std::io::Cursor;

    #[test]
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
    fn read_http_request_rejects_invalid_content_length() {
        let req = b"POST /generate HTTP/1.1\r\nContent-Length: nope\r\n\r\n";
        let mut cursor = Cursor::new(req.as_slice());
        let err = read_http_request(&mut cursor).expect_err("request should fail");
        assert_eq!(err.status, 400);
    }

    #[test]
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
    fn write_http_response_includes_cors_headers() {
        let mut out = Vec::new();
        write_http_response(&mut out, 200, "{\"status\":\"ok\"}").expect("response write");
        let text = String::from_utf8(out).expect("valid UTF-8 response");
        assert!(text.contains("Access-Control-Allow-Origin: *"));
        assert!(text.contains("Access-Control-Allow-Methods: GET,POST,OPTIONS"));
        assert!(text.contains("X-Content-Type-Options: nosniff"));
    }
}
