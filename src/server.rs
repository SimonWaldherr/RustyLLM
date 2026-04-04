use crate::runtime::{ChatMessage, ChatRole, GenerationOptions, Runner};
use rustls::pki_types::{CertificateDer, PrivateKeyDer};
use rustls::{ServerConfig, ServerConnection, StreamOwned};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{self, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;
use std::thread;

#[derive(Clone)]
pub struct ServeOptions {
    pub addr: String,
    pub defaults: GenerationOptions,
    pub tls_cert_path: Option<String>,
    pub tls_key_path: Option<String>,
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

pub fn serve(runner: Arc<Runner>, options: ServeOptions) -> Result<(), String> {
    let listener = TcpListener::bind(&options.addr)
        .map_err(|err| format!("Failed to bind {}: {}", options.addr, err))?;

    // Keep the server loop deliberately small: accept a connection, hand it to
    // a worker thread, and let the handler own the request lifecycle.
    if options.is_tls() {
        let tls_config = Arc::new(load_tls_config(
            options.tls_cert_path.as_deref().unwrap(),
            options.tls_key_path.as_deref().unwrap(),
        )?);
        for stream in listener.incoming() {
            match stream {
                Ok(stream) => {
                    let runner = Arc::clone(&runner);
                    let options = options.clone();
                    let tls_config = Arc::clone(&tls_config);
                    thread::spawn(move || {
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
                Ok(stream) => {
                    let runner = Arc::clone(&runner);
                    let options = options.clone();
                    thread::spawn(move || {
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
    let request = read_http_request(&mut stream).map_err(|err| err.to_string())?;
    let (status, body) = route_request(&request, &runner, options);
    write_http_response(&mut stream, status, &body).map_err(|err| err.to_string())
}

fn route_request(request: &HttpRequest, runner: &Runner, options: &ServeOptions) -> (u16, String) {
    match (request.method.as_str(), request.path.as_str()) {
        ("GET", "/health") => (200, String::from("{\"status\":\"ok\"}")),
        ("POST", "/generate") => match serde_json::from_slice::<GenerateRequest>(&request.body) {
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
        },
        _ => (404, json_error("Not found")),
    }
}

fn json_error(message: &str) -> String {
    serde_json::json!({ "error": message }).to_string()
}

struct HttpRequest {
    method: String,
    path: String,
    body: Vec<u8>,
}

fn read_http_request<T>(stream: &mut T) -> io::Result<HttpRequest>
where
    T: Read,
{
    let mut header_bytes = Vec::new();
    let mut one = [0u8; 1];
    // This parser intentionally supports only small HTTP/1.1 requests with a
    // Content-Length body; it is enough for the local inference API.
    loop {
        let read = stream.read(&mut one)?;
        if read == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "connection closed",
            ));
        }
        header_bytes.push(one[0]);
        if header_bytes.ends_with(b"\r\n\r\n") {
            break;
        }
        if header_bytes.len() > 64 * 1024 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "header too large",
            ));
        }
    }

    let header_text = String::from_utf8_lossy(&header_bytes);
    let mut lines = header_text.split("\r\n");
    let request_line = lines
        .next()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing request line"))?;
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or("").to_string();
    let path = parts.next().unwrap_or("/").to_string();

    let mut content_length = 0usize;
    for line in lines {
        if line.is_empty() {
            continue;
        }
        if let Some((name, value)) = line.split_once(':') {
            if name.eq_ignore_ascii_case("content-length") {
                content_length = value.trim().parse().unwrap_or(0);
            }
        }
    }

    let mut body = vec![0u8; content_length];
    if content_length > 0 {
        stream.read_exact(&mut body)?;
    }

    Ok(HttpRequest { method, path, body })
}

fn write_http_response<T>(stream: &mut T, status: u16, body: &str) -> io::Result<()>
where
    T: Write,
{
    let status_text = match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        _ => "Internal Server Error",
    };
    write!(
        stream,
        "HTTP/1.1 {} {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\nAccess-Control-Allow-Origin: *\r\n\r\n{}",
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
