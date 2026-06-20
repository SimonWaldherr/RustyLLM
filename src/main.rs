#![allow(
    clippy::collapsible_if,
    clippy::collapsible_match,
    clippy::explicit_counter_loop,
    clippy::manual_checked_ops,
    clippy::manual_is_multiple_of,
    clippy::unnecessary_unwrap,
    clippy::useless_conversion
)]

#[cfg(all(not(target_family = "wasm"), feature = "server"))]
use rusty_llm::catalog::{ModelEntry, select_model};
#[cfg(not(target_family = "wasm"))]
use rusty_llm::catalog::{
    default_model_dir, discover_models, print_model_list, resolve_model_file_selector,
    resolve_model_path,
};
use rusty_llm::gguf::GGUFFile;
#[cfg(not(target_family = "wasm"))]
use rusty_llm::metal;
use rusty_llm::model::Config;
use rusty_llm::runtime::{
    ChatMessage, GenerationOptions, KvCacheDType, LoadInfo, Runner, RuntimeProfile,
    compatibility_report,
};
#[cfg(all(not(target_family = "wasm"), feature = "server"))]
use rusty_llm::server::{self, McpServeOptions, ServeOptions};
use rusty_llm::simd;
use std::collections::{BTreeMap, HashSet};
use std::env;
use std::fmt::Display;
use std::io::{self, BufRead, Read, Write};
use std::path::{Path, PathBuf};
#[cfg(all(not(target_family = "wasm"), feature = "server"))]
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Prints CLI usage text.
fn print_usage(name: &str) {
    eprintln!("rusty-llm v0.3.0");
    eprintln!();
    eprintln!(
        "Usage: {} [model.gguf|model-name|model-dir] [options]",
        name
    );
    eprintln!();
    eprintln!("Options:");
    eprintln!(
        "  --model <name>           Select a GGUF from --model-dir by repo, file, or metadata name"
    );
    eprintln!("  --model-dir <path>       Directory to recursively scan for GGUF files");
    eprintln!("  --list-models            List GGUF files in --model-dir and exit");
    eprintln!("  --prompt <text>           Input prompt (interactive if omitted)");
    eprintln!("  --repl                    Start an interactive REPL session");
    eprintln!("  --verbose                 Print startup timing details");
    #[cfg(feature = "server")]
    eprintln!("  --serve <addr>            Start HTTP(S) API server, e.g. 127.0.0.1:8080");
    #[cfg(feature = "server")]
    eprintln!("  --mcp                     Start a Model Context Protocol stdio server");
    #[cfg(feature = "server")]
    eprintln!("  --chat                    Enable Web UIs at /chat, /chat?expert, and /explorer");
    #[cfg(feature = "server")]
    eprintln!("  --tls-cert <path>         PEM certificate for HTTPS");
    #[cfg(feature = "server")]
    eprintln!("  --tls-key <path>          PEM private key for HTTPS");
    #[cfg(feature = "server")]
    eprintln!("  --max-connections <N>     Max concurrent server connections");
    #[cfg(feature = "server")]
    eprintln!(
        "  --max-sessions <N>        Max persistent conversation sessions (default: 8, 0=disable)"
    );
    #[cfg(feature = "server")]
    eprintln!("  --max-cached-tokens <N>   Max KV-cache tokens per session (default: 2048)");
    eprintln!("  --max-tokens <N>          Max tokens to generate (default: 256)");
    eprintln!("  --temp <F>                Temperature (default: 0.7, 0=greedy)");
    eprintln!("  --top-p <F>               Nucleus sampling threshold (default: 0.9)");
    eprintln!("  --top-k <N>               Top-K filtering (default: 40)");
    eprintln!("  --repeat-penalty <F>      Repetition penalty (default: 1.1)");
    eprintln!("  --seed <N>                RNG seed (default: time-based)");
    eprintln!("  --threads <N>             Override thread count");
    eprintln!("  --mtp-assistant <path>    Assistant GGUF for greedy speculative decoding");
    eprintln!("  --mtp-tokens <N>          Max speculative draft tokens (default: 4)");
    eprintln!("  --mtp-min-accept-rate <F> Disable MTP below this accept rate (default: 0.5)");
    eprintln!("  --no-mtp-adaptive         Keep --mtp-tokens fixed instead of adapting it");
    eprintln!("  --no-speculative          Disable speculative decoding");
    eprintln!("  --kv-cache-dtype <T>      KV cache dtype: auto, f32, bf16, q8");
    eprintln!("  --max-context <N>         Cap runtime context/KV-cache length");
    eprintln!("  --profile <name>          Runtime profile: auto, mistral, mistral-ultra, gemma");
    eprintln!("  --sliding-window <N>      Override sliding-window size for runtime planning");
    eprintln!("  --no-flash-attn           Disable online-softmax attention optimization marker");
    eprintln!("  --system-prompt <T>       Override the default system prompt");
    eprintln!(
        "  --thinking                Rewrite each prompt with the built-in meta-prompt before answering"
    );
    eprintln!("  --thinking-prompt <T>     Override the Thinking meta-prompt");
    eprintln!("  --thinking-max-tokens <N> Max tokens for the Thinking rewrite (default: 192)");
    eprintln!("  --skills-dir <path>       Directory containing prompt-selected SKILL.md files");
    eprintln!("  --max-skills <N>          Max new skills to load per prompt (default: 3)");
    eprintln!("  --skill-max-bytes <N>     Max bytes loaded from each SKILL.md (default: 16384)");
    eprintln!("  --stop <text>             Stop generation when this string appears");
    eprintln!("  --embed                   Embed prompt and print the vector (RAG mode)");
    eprintln!("  --bench                   Run a non-streaming generation benchmark");
    eprintln!("  --bench-json              Run benchmark and emit machine-readable JSON");
    eprintln!("  --bench-output            Include generated text for each benchmark run");
    eprintln!("  --bench-runs <N>          Number of benchmark runs (default: 3)");
    eprintln!("  --kernel-bench            Run isolated kernel benchmark");
    eprintln!("  --kernel-bench-json       Emit isolated kernel benchmark JSON");
    eprintln!("  --kernel-bench-runs <N>   Number of kernel benchmark runs (default: 25)");
    eprintln!("  --kernel-bench-layer <N>  Transformer layer to benchmark (default: 0)");
    eprintln!(
        "  --inspect                 Inspect GGUF metadata and compatibility without loading weights"
    );
    eprintln!("  --chat-history <path>     Append chat/generation turns to a JSON file");
    eprintln!("  --list-tensors            Print GGUF tensor inventory and exit");
}

/// Parses the value following a command-line flag.
fn parse_arg<T>(args: &[String], i: &mut usize, flag: &str) -> Result<T, String>
where
    T: std::str::FromStr,
    T::Err: Display,
{
    *i += 1;
    if *i >= args.len() {
        return Err(format!("Missing value for {}.", flag));
    }
    args[*i]
        .parse::<T>()
        .map_err(|err| format!("Invalid {} value '{}': {}", flag, args[*i], err))
}

/// Records a model selector while rejecting conflicting selectors.
fn set_model_selector(
    current: &mut Option<String>,
    value: String,
    source: &str,
) -> Result<(), String> {
    if current.is_some() {
        return Err(format!(
            "Multiple model selectors were provided; remove the extra {} value.",
            source
        ));
    }
    *current = Some(value);
    Ok(())
}

/// Reports whether a selector already names a concrete model file.
fn is_existing_file(selection: &str) -> bool {
    let path = Path::new(selection);
    path.exists() && path.is_file()
}

/// Prints a compact startup timing breakdown for performance tuning.
fn print_startup_timing(
    total_time: Duration,
    model_resolution_time: Duration,
    model_catalog_time: Option<Duration>,
    load_info: &LoadInfo,
) {
    eprintln!("Startup timing:");
    if let Some(duration) = model_catalog_time {
        eprintln!("  model catalog:       {}", format_duration_ms(duration));
    }
    eprintln!(
        "  model resolution:    {}",
        format_duration_ms(model_resolution_time)
    );
    eprintln!(
        "  mmap open:           {}",
        format_duration_ms(load_info.mmap_time)
    );
    eprintln!(
        "  GGUF parse:          {}",
        format_duration_ms(load_info.parse_time)
    );
    eprintln!(
        "  compatibility check: {}",
        format_duration_ms(load_info.compatibility_time)
    );
    eprintln!(
        "  tokenizer build:     {}",
        format_duration_ms(load_info.tokenizer_time)
    );
    eprintln!(
        "  weight views:        {}",
        format_duration_ms(load_info.weights_time)
    );
    eprintln!(
        "  runner init:         {}",
        format_duration_ms(load_info.runner_time)
    );
    eprintln!(
        "  model load total:    {}",
        format_duration_ms(load_info.load_time)
    );
    eprintln!("  process startup:     {}", format_duration_ms(total_time));
    eprintln!();
}

fn format_duration_ms(duration: Duration) -> String {
    format!("{:.2} ms", duration.as_secs_f64() * 1000.0)
}

#[cfg(all(not(target_family = "wasm"), feature = "server"))]
fn resolve_model_path_from_discovered(
    selection: Option<&str>,
    model_dir: &Path,
    entries: &[ModelEntry],
) -> Result<PathBuf, String> {
    if let Some(selection) = selection {
        let selected_path = Path::new(selection);
        if selected_path.exists() {
            if selected_path.is_file() {
                return Ok(selected_path.to_path_buf());
            }
            if selected_path.is_dir() {
                return choose_single_discovered_model(entries, selected_path);
            }
            return Err(format!(
                "Model path is neither a file nor a directory: {}",
                selection
            ));
        }

        return select_model(entries, selection).map(|entry| entry.path.clone());
    }

    choose_single_discovered_model(entries, model_dir)
}

#[cfg(all(not(target_family = "wasm"), feature = "server"))]
fn choose_single_discovered_model(entries: &[ModelEntry], dir: &Path) -> Result<PathBuf, String> {
    let usable = entries
        .iter()
        .filter(|entry| entry.is_supported && !entry.is_projector)
        .collect::<Vec<_>>();

    match usable.len() {
        0 => Err(format!(
            "No supported text GGUF models found in {}.",
            dir.display()
        )),
        1 => Ok(usable[0].path.clone()),
        _ => {
            let choices = usable
                .iter()
                .take(16)
                .map(|entry| format!("  - {}", entry.id))
                .collect::<Vec<_>>()
                .join("\n");
            Err(format!(
                "Found multiple GGUF models in {}. Choose one with --model <name> or pass an exact .gguf path.\n\n{}",
                dir.display(),
                choices
            ))
        }
    }
}

/// runs the CLI and prints fatal errors.
fn main() {
    if let Err(err) = run() {
        eprintln!("{}", err);
        std::process::exit(1);
    }
}

/// Executes a queued worker-pool job and waits for completion.
fn run() -> Result<(), String> {
    let startup_start = Instant::now();
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        print_usage(&args[0]);
        return Err(String::from(
            "Missing model selector. Use --list-models to inspect the configured model directory.",
        ));
    }

    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        print_usage(&args[0]);
        return Ok(());
    }

    let mut model_selector: Option<String> = None;
    let mut model_dir: PathBuf = default_model_dir();
    let mut prompt = String::new();
    let mut options = GenerationOptions::default();
    let mut list_tensors = false;
    let mut list_models = false;
    let mut threads_override: Option<usize> = None;
    let mut repl_mode = false;
    let mut serve_addr: Option<String> = None;
    let mut mcp_mode = false;
    let mut chat_ui = false;
    let mut tls_cert: Option<String> = None;
    let mut tls_key: Option<String> = None;
    let mut max_connections_override: Option<usize> = None;
    #[cfg(all(not(target_family = "wasm"), feature = "server"))]
    let mut max_sessions: usize = server::DEFAULT_MAX_SESSIONS;
    #[cfg(all(not(target_family = "wasm"), feature = "server"))]
    let mut max_cached_tokens: usize = server::DEFAULT_MAX_CACHED_TOKENS;
    let mut embed_mode = false;
    let mut bench_mode = false;
    let mut bench_json = false;
    let mut bench_output = false;
    let mut kernel_bench = false;
    let mut kernel_bench_json = false;
    let mut kernel_bench_runs = 25usize;
    let mut kernel_bench_layer = 0usize;
    let mut inspect_mode = false;
    let mut chat_history_path: Option<String> = None;
    let mut bench_runs = 3usize;
    let mut verbose = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                let value = parse_arg::<String>(&args, &mut i, "--model")?;
                if let Some(existing) = model_selector.as_deref() {
                    let existing_path = PathBuf::from(existing);
                    if existing_path.is_dir() {
                        model_dir = existing_path;
                        model_selector = Some(value);
                    } else {
                        return Err(String::from(
                            "Multiple model selectors were provided; remove the extra --model value.",
                        ));
                    }
                } else {
                    set_model_selector(&mut model_selector, value, "--model")?;
                }
            }
            "--model-dir" => {
                model_dir = PathBuf::from(parse_arg::<String>(&args, &mut i, "--model-dir")?);
            }
            "--list-models" => {
                list_models = true;
            }
            "--prompt" | "-p" => {
                prompt = parse_arg::<String>(&args, &mut i, "--prompt")?;
            }
            "--repl" => {
                repl_mode = true;
            }
            "--verbose" | "-v" => {
                verbose = true;
            }
            "--serve" => {
                serve_addr = Some(parse_arg::<String>(&args, &mut i, "--serve")?);
            }
            "--mcp" => {
                mcp_mode = true;
            }
            "--chat" => {
                chat_ui = true;
            }
            "--tls-cert" => {
                tls_cert = Some(parse_arg::<String>(&args, &mut i, "--tls-cert")?);
            }
            "--tls-key" => {
                tls_key = Some(parse_arg::<String>(&args, &mut i, "--tls-key")?);
            }
            "--max-connections" => {
                max_connections_override =
                    Some(parse_arg::<usize>(&args, &mut i, "--max-connections")?);
            }
            "--max-sessions" => {
                #[cfg(all(not(target_family = "wasm"), feature = "server"))]
                {
                    max_sessions = parse_arg::<usize>(&args, &mut i, "--max-sessions")?;
                }
                #[cfg(not(all(not(target_family = "wasm"), feature = "server")))]
                {
                    let _ = parse_arg::<usize>(&args, &mut i, "--max-sessions")?;
                }
            }
            "--max-cached-tokens" => {
                #[cfg(all(not(target_family = "wasm"), feature = "server"))]
                {
                    max_cached_tokens = parse_arg::<usize>(&args, &mut i, "--max-cached-tokens")?;
                }
                #[cfg(not(all(not(target_family = "wasm"), feature = "server")))]
                {
                    let _ = parse_arg::<usize>(&args, &mut i, "--max-cached-tokens")?;
                }
            }
            "--max-tokens" | "-n" => {
                options.max_tokens = parse_arg::<usize>(&args, &mut i, "--max-tokens")?;
            }
            "--temp" | "-t" => {
                options.sampler.temperature = parse_arg::<f32>(&args, &mut i, "--temp")?;
            }
            "--top-p" => {
                options.sampler.top_p = parse_arg::<f32>(&args, &mut i, "--top-p")?;
            }
            "--top-k" => {
                options.sampler.top_k = parse_arg::<usize>(&args, &mut i, "--top-k")?;
            }
            "--repeat-penalty" => {
                options.sampler.repeat_penalty =
                    parse_arg::<f32>(&args, &mut i, "--repeat-penalty")?;
            }
            "--seed" => {
                options.seed = parse_arg::<u64>(&args, &mut i, "--seed")?;
            }
            "--threads" => {
                threads_override = Some(parse_arg::<usize>(&args, &mut i, "--threads")?);
            }
            "--mtp-assistant" => {
                options.speculative.assistant_path =
                    Some(parse_arg::<String>(&args, &mut i, "--mtp-assistant")?);
            }
            "--mtp-tokens" => {
                options.speculative.max_draft_tokens =
                    parse_arg::<usize>(&args, &mut i, "--mtp-tokens")?;
            }
            "--mtp-min-accept-rate" => {
                options.speculative.min_accept_rate =
                    parse_arg::<f32>(&args, &mut i, "--mtp-min-accept-rate")?;
            }
            "--no-mtp-adaptive" => {
                options.speculative.adaptive = false;
            }
            "--no-speculative" => {
                options.speculative.enabled = false;
            }
            "--kv-cache-dtype" => {
                let value = parse_arg::<String>(&args, &mut i, "--kv-cache-dtype")?;
                options.runtime.kv_cache_dtype = KvCacheDType::parse(&value)
                    .ok_or_else(|| format!("Invalid --kv-cache-dtype value '{}'.", value))?;
            }
            "--max-context" => {
                options.runtime.max_context =
                    Some(parse_arg::<usize>(&args, &mut i, "--max-context")?);
            }
            "--profile" => {
                let value = parse_arg::<String>(&args, &mut i, "--profile")?;
                options.runtime.profile = RuntimeProfile::parse(&value)
                    .ok_or_else(|| format!("Invalid --profile value '{}'.", value))?;
            }
            "--sliding-window" => {
                options.runtime.sliding_window_size =
                    Some(parse_arg::<usize>(&args, &mut i, "--sliding-window")?);
            }
            "--flash-attn" => {
                options.runtime.flash_attention = true;
            }
            "--no-flash-attn" => {
                options.runtime.flash_attention = false;
            }
            "--system-prompt" => {
                options.system_prompt = parse_arg::<String>(&args, &mut i, "--system-prompt")?;
            }
            "--thinking" => {
                options.thinking.enabled = true;
            }
            "--no-thinking" => {
                options.thinking.enabled = false;
            }
            "--thinking-prompt" | "--thinking-system-prompt" => {
                options.thinking.system_prompt =
                    parse_arg::<String>(&args, &mut i, "--thinking-prompt")?;
            }
            "--thinking-max-tokens" => {
                options.thinking.max_tokens =
                    parse_arg::<usize>(&args, &mut i, "--thinking-max-tokens")?;
            }
            "--skills-dir" => {
                options.skills.directory =
                    Some(parse_arg::<String>(&args, &mut i, "--skills-dir")?);
            }
            "--max-skills" => {
                options.skills.max_skills = parse_arg::<usize>(&args, &mut i, "--max-skills")?;
            }
            "--skill-max-bytes" => {
                options.skills.max_bytes_per_skill =
                    parse_arg::<usize>(&args, &mut i, "--skill-max-bytes")?;
            }
            "--stop" => {
                options
                    .stop_sequences
                    .push(parse_arg::<String>(&args, &mut i, "--stop")?);
            }
            "--embed" => {
                embed_mode = true;
            }
            "--bench" => {
                bench_mode = true;
            }
            "--bench-json" | "—bench-json" => {
                bench_mode = true;
                bench_json = true;
            }
            "--bench-output" => {
                bench_output = true;
            }
            "--bench-runs" => {
                bench_runs = parse_arg::<usize>(&args, &mut i, "--bench-runs")?;
            }
            "--kernel-bench" => {
                kernel_bench = true;
            }
            "--kernel-bench-json" => {
                kernel_bench = true;
                kernel_bench_json = true;
            }
            "--kernel-bench-runs" => {
                kernel_bench_runs = parse_arg::<usize>(&args, &mut i, "--kernel-bench-runs")?;
            }
            "--kernel-bench-layer" => {
                kernel_bench_layer = parse_arg::<usize>(&args, &mut i, "--kernel-bench-layer")?;
            }
            "--inspect" => {
                inspect_mode = true;
            }
            "--chat-history" | "--chat-log" => {
                chat_history_path = Some(parse_arg::<String>(&args, &mut i, "--chat-history")?);
            }
            "--list-tensors" => {
                list_tensors = true;
            }
            other => {
                if other.starts_with('-') {
                    return Err(format!("Unknown option: {}", other));
                }
                let positional_path = PathBuf::from(other);
                if model_selector.is_some() && positional_path.is_dir() {
                    model_dir = positional_path;
                } else {
                    set_model_selector(&mut model_selector, other.to_string(), "positional model")?;
                }
            }
        }
        i += 1;
    }

    if list_models {
        let entries = discover_models(&model_dir)?;
        print_model_list(&entries);
        return Ok(());
    }

    if tls_cert.is_some() ^ tls_key.is_some() {
        return Err(String::from(
            "Both --tls-cert and --tls-key must be provided together.",
        ));
    }
    if chat_ui && serve_addr.is_none() {
        return Err(String::from("--chat requires --serve <addr>."));
    }
    if mcp_mode && serve_addr.is_some() {
        return Err(String::from("--mcp cannot be combined with --serve."));
    }
    if mcp_mode && (embed_mode || repl_mode) {
        return Err(String::from(
            "--mcp cannot be combined with --embed or --repl.",
        ));
    }
    #[cfg(not(feature = "tls"))]
    if tls_cert.is_some() || tls_key.is_some() {
        return Err(String::from(
            "--tls-cert/--tls-key require a binary built with the `tls` feature.",
        ));
    }
    if let Some(n) = threads_override {
        if n == 0 {
            return Err(String::from("--threads must be greater than 0."));
        }
    }
    if let Some(n) = max_connections_override {
        if n == 0 {
            return Err(String::from("--max-connections must be greater than 0."));
        }
    }
    if bench_runs == 0 {
        return Err(String::from("--bench-runs must be greater than 0."));
    }
    if kernel_bench_runs == 0 {
        return Err(String::from("--kernel-bench-runs must be greater than 0."));
    }
    options.validate()?;

    let n_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    eprintln!("System: {} threads", n_threads);
    #[cfg(target_arch = "aarch64")]
    eprintln!("SIMD: ARM NEON (native)");
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            eprintln!("SIMD: AVX2 + FMA");
        } else if is_x86_feature_detected!("sse4.1") {
            eprintln!("SIMD: SSE4.1 (AVX2 not available)");
        } else {
            eprintln!("SIMD: scalar fallback");
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    eprintln!("SIMD: scalar fallback");
    #[cfg(not(target_family = "wasm"))]
    if metal::enabled() {
        eprintln!("Metal: Q4_K matvec enabled, Q6_K output matvec enabled");
    } else if metal::requested() == Some(true) {
        eprintln!("Metal: unavailable, using CPU");
    } else if metal::requested() == Some(false) {
        eprintln!("Metal: disabled by RUSTY_LLM_METAL");
    }
    if std::env::var_os("RUSTY_LLM_FAST_ATTN").is_some() {
        eprintln!("Attention: fast approximation mode enabled");
    }
    if options.runtime.profile == RuntimeProfile::MistralUltra {
        eprintln!(
            "Mistral Ultra: aggressive Metal matvec/attention routing with native SIMD fallback"
        );
    }

    if let Some(n) = threads_override {
        simd::set_num_threads(n);
        eprintln!("Worker threads: {}", n);
    }

    let model_resolution_start = Instant::now();
    let mut model_catalog_time: Option<Duration> = None;
    #[cfg(all(not(target_family = "wasm"), feature = "server"))]
    let exact_model_file = model_selector
        .as_deref()
        .map(is_existing_file)
        .unwrap_or(false);
    #[cfg(all(not(target_family = "wasm"), feature = "server"))]
    let server_model_dir = model_selector
        .as_deref()
        .map(Path::new)
        .filter(|path| path.exists() && path.is_dir())
        .map(Path::to_path_buf)
        .unwrap_or_else(|| model_dir.clone());
    #[cfg(all(not(target_family = "wasm"), feature = "server"))]
    let fast_model_path = if exact_model_file {
        Some(PathBuf::from(
            model_selector.as_deref().expect("exact file has selector"),
        ))
    } else if let Some(selection) = model_selector.as_deref() {
        resolve_model_file_selector(selection, &server_model_dir)?
    } else {
        None
    };
    #[cfg(all(not(target_family = "wasm"), feature = "server"))]
    let discovered_server_models = if serve_addr.is_some() && fast_model_path.is_none() {
        let catalog_start = Instant::now();
        match discover_models(&server_model_dir) {
            Ok(entries) => {
                model_catalog_time = Some(catalog_start.elapsed());
                Some(entries)
            }
            Err(err) => {
                model_catalog_time = Some(catalog_start.elapsed());
                eprintln!("Warning: model catalog unavailable: {}", err);
                None
            }
        }
    } else {
        None
    };
    #[cfg(all(not(target_family = "wasm"), feature = "server"))]
    let model_path = if let Some(path) = fast_model_path {
        path
    } else if let Some(entries) = discovered_server_models.as_deref() {
        resolve_model_path_from_discovered(model_selector.as_deref(), &server_model_dir, entries)?
    } else {
        resolve_model_path(model_selector.as_deref(), &model_dir)?
    };
    #[cfg(not(all(not(target_family = "wasm"), feature = "server")))]
    let model_path = resolve_model_path(model_selector.as_deref(), &model_dir)?;
    let model_resolution_time = model_resolution_start.elapsed();
    #[cfg(all(not(target_family = "wasm"), feature = "server"))]
    let model_catalog = discovered_server_models.map(|entries| server::ModelCatalogSnapshot {
        model_dir: server_model_dir.display().to_string(),
        loaded_model_path: model_path.display().to_string(),
        entries: entries
            .into_iter()
            .map(server::ModelCatalogEntry::from)
            .collect(),
    });

    if inspect_mode {
        inspect_model_file(&model_path)?;
        return Ok(());
    }

    eprintln!("\nLoading: {}", model_path.display());
    let model_path_str = model_path
        .to_str()
        .ok_or_else(|| format!("Non-UTF-8 model path: {}", model_path.display()))?;
    let (mut runner, load_info) = Runner::from_path(model_path_str)?;
    let file_mb = load_info.file_size_bytes as f64 / (1024.0 * 1024.0);
    eprintln!("File size: {:.1} MB", file_mb);
    if let Some(name) = runner.model_name() {
        eprintln!("Model: {}", name);
    }
    eprintln!("Architecture: {}", runner.architecture());
    eprintln!(
        "Tokenizer: {} tokens, BOS={}, EOS={}",
        runner.tokenizer().vocab_size(),
        runner.tokenizer().bos_id,
        runner.tokenizer().eos_id
    );
    eprintln!(
        "Loaded in {:.2}s ({:.0} MB/s)\n",
        load_info.load_time.as_secs_f32(),
        file_mb / load_info.load_time.as_secs_f64()
    );
    if verbose {
        print_startup_timing(
            startup_start.elapsed(),
            model_resolution_time,
            model_catalog_time,
            &load_info,
        );
    }

    if options.speculative.enabled {
        if let Some(assistant_path) = options.speculative.assistant_path.as_deref() {
            eprintln!("Loading MTP assistant: {}", assistant_path);
            let (assistant, assistant_info) = Runner::from_path(assistant_path)?;
            runner.attach_speculative_assistant(assistant)?;
            let assistant_mb = assistant_info.file_size_bytes as f64 / (1024.0 * 1024.0);
            let assistant_ratio =
                assistant_info.file_size_bytes as f64 / load_info.file_size_bytes.max(1) as f64;
            eprintln!(
                "MTP assistant loaded in {:.2}s ({:.1} MB, {:.0}% of target size)",
                assistant_info.load_time.as_secs_f32(),
                assistant_mb,
                assistant_ratio * 100.0
            );
            if assistant_ratio >= 0.75 {
                eprintln!(
                    "Warning: MTP assistant is {:.0}% of target size; speculative decoding usually helps only with a much smaller assistant.",
                    assistant_ratio * 100.0
                );
            }
        }
    }
    eprintln!(
        "Optimizations: {}",
        runner.optimization_summary(&options).join(", ")
    );
    for warning in runner.optimization_warnings(&options) {
        eprintln!("Warning: {}", warning);
    }
    eprintln!();
    io::stderr().flush().map_err(|err| err.to_string())?;

    if list_tensors {
        for tensor in &runner.gguf().tensors {
            eprintln!("{} {:?} {:?}", tensor.name, tensor.dtype, tensor.dims);
        }
        return Ok(());
    }

    if bench_mode {
        if embed_mode || repl_mode || serve_addr.is_some() || mcp_mode {
            return Err(String::from(
                "--bench cannot be combined with --embed, --repl, --serve, or --mcp.",
            ));
        }
        let bench_prompt = if prompt.trim().is_empty() {
            "Explain local LLM inference performance in one concise paragraph."
        } else {
            prompt.trim()
        };
        run_benchmark(
            &runner,
            &load_info,
            bench_prompt,
            &options,
            bench_runs,
            bench_json,
            bench_output,
        )?;
        return Ok(());
    }

    if kernel_bench {
        if embed_mode || repl_mode || serve_addr.is_some() || mcp_mode {
            return Err(String::from(
                "--kernel-bench cannot be combined with --embed, --repl, --serve, or --mcp.",
            ));
        }
        run_kernel_benchmark(
            &runner,
            &model_path,
            &options,
            kernel_bench_runs,
            kernel_bench_layer,
            kernel_bench_json,
        )?;
        return Ok(());
    }

    // Embedding mode: run prefill, print the L2-normalised embedding vector.
    if embed_mode {
        if prompt.is_empty() {
            return Err(String::from("--embed requires --prompt <text>."));
        }
        let result = runner.embed(&prompt)?;
        eprintln!(
            "Embedding dim: {}, tokens: {}",
            result.embedding.len(),
            result.token_count
        );
        let floats: Vec<String> = result
            .embedding
            .iter()
            .map(|v| format!("{:.6}", v))
            .collect();
        println!("[{}]", floats.join(", "));
        return Ok(());
    }

    // Server mode takes over the process after the model is loaded.
    if let Some(addr) = serve_addr {
        #[cfg(not(feature = "server"))]
        {
            let _ = addr;
            return Err(String::from(
                "--serve requires a binary built with the `server` feature.",
            ));
        }

        #[cfg(feature = "server")]
        {
            let protocol = if tls_cert.is_some() && tls_key.is_some() {
                "HTTPS"
            } else {
                "HTTP"
            };
            let max_connections =
                max_connections_override.unwrap_or_else(|| (n_threads * 8).max(16));
            eprintln!("{} endpoint listening on {}", protocol, addr);
            if chat_ui {
                eprintln!(
                    "Routes: GET /chat, GET /chat?expert, GET /explorer, GET /docs, GET /health, POST /generate, GET /v1/models, POST /v1/completions, POST /v1/chat/completions, POST /v1/responses, POST /v1/embeddings."
                );
            } else {
                eprintln!(
                    "Routes: GET /docs, GET /health, POST /generate, GET /v1/models, POST /v1/completions, POST /v1/chat/completions, POST /v1/responses, POST /v1/embeddings."
                );
            }
            eprintln!("Max concurrent connections: {}", max_connections);
            let session_store = if max_sessions > 0 {
                use rusty_llm::session::SessionStore;
                eprintln!(
                    "Session cache: {} sessions, {} tokens/session",
                    max_sessions, max_cached_tokens
                );
                Some(Arc::new(SessionStore::new(max_sessions, max_cached_tokens)))
            } else {
                eprintln!("Session cache: disabled (--max-sessions 0)");
                None
            };
            let serve_options = ServeOptions {
                addr,
                defaults: options.clone(),
                tls_cert_path: tls_cert,
                tls_key_path: tls_key,
                max_concurrent_connections: max_connections,
                chat_ui,
                model_catalog,
                chat_history_path: chat_history_path.clone(),
                chat_history_lock: Arc::new(std::sync::Mutex::new(())),
                session_store,
            };
            server::serve(Arc::new(runner), serve_options)?;
            return Ok(());
        }
    }

    if mcp_mode {
        #[cfg(not(feature = "server"))]
        {
            return Err(String::from(
                "--mcp requires a binary built with the `server` feature.",
            ));
        }

        #[cfg(feature = "server")]
        {
            eprintln!("MCP stdio server ready");
            let mcp_options = McpServeOptions {
                defaults: options.clone(),
                chat_history_path: chat_history_path.clone(),
                chat_history_lock: Arc::new(std::sync::Mutex::new(())),
                skill_memory: Arc::new(std::sync::Mutex::new(HashSet::new())),
            };
            server::serve_mcp_stdio(Arc::new(runner), mcp_options)?;
            return Ok(());
        }
    }

    if repl_mode {
        run_repl(&runner, &options, chat_history_path.as_deref())?;
        return Ok(());
    }

    // Fall back to stdin when no prompt flag was provided so the binary works
    // both interactively and in shell pipelines.
    if prompt.is_empty() {
        if atty_is_stdin() {
            eprint!(">>> ");
            io::stderr().flush().map_err(|err| err.to_string())?;
            io::stdin()
                .lock()
                .read_line(&mut prompt)
                .map_err(|err| err.to_string())?;
            prompt = prompt.trim().to_string();
        } else {
            let mut buf = String::new();
            io::stdin()
                .read_to_string(&mut buf)
                .map_err(|err| err.to_string())?;
            prompt = buf.trim().to_string();
        }
    }

    if prompt.is_empty() {
        return Err(String::from("No prompt provided."));
    }

    let mut loaded_skills = HashSet::new();
    let result = runner.generate_stream_with_skill_memory(
        &prompt,
        &options,
        &mut loaded_skills,
        |text| {
            print!("{}", text);
            let _ = io::stdout().flush();
        },
    )?;
    append_cli_history(
        chat_history_path.as_deref(),
        &runner,
        "cli.generate",
        &[ChatMessage::user(prompt.clone())],
        &result,
    )?;

    eprintln!("\n\n─── Stats ───────────────────────────────");
    eprintln!("Prompt: {} tokens", result.stats.prompt_tokens);
    eprintln!("Generated: {} tokens", result.stats.generated_tokens);
    eprintln!(
        "Prefill: {:.2}s ({:.1} tok/s)",
        result.stats.prefill_time.as_secs_f32(),
        result.stats.prompt_tokens as f32 / result.stats.prefill_time.as_secs_f32().max(0.001)
    );
    eprintln!(
        "Decode: {:.2}s ({:.2} tok/s)",
        result.stats.decode_time.as_secs_f32(),
        result.stats.generated_tokens as f32 / result.stats.decode_time.as_secs_f32().max(0.001)
    );
    if let Some(spec) = &result.stats.speculative {
        eprintln!(
            "MTP: accept_rate={:.2}, drafted={}, accepted={}, draft={:.2} tok/s, effective={:.2} tok/s{}{}",
            spec.accept_rate(),
            spec.drafted_tokens,
            spec.accepted_tokens,
            spec.draft_tok_s(),
            result.stats.generated_tokens as f32
                / result.stats.decode_time.as_secs_f32().max(0.001),
            if spec.disabled { " (disabled)" } else { "" },
            if spec.accept_rate() < options.speculative.min_accept_rate {
                " recommendation=disable"
            } else {
                ""
            }
        );
    }
    eprintln!("Total: {:.2}s", result.stats.total_time.as_secs_f32());

    Ok(())
}

/// Prints a JSON inspection report for one GGUF model file.
fn inspect_model_file(path: &PathBuf) -> Result<(), String> {
    let path_str = path
        .to_str()
        .ok_or_else(|| format!("Non-UTF-8 model path: {}", path.display()))?;
    let mmap = rusty_llm::mmap::MmapFile::open(path_str).map_err(|err| err.to_string())?;
    let gguf = GGUFFile::parse_quiet(mmap.as_slice())?;
    let arch = gguf.get_str("general.architecture").unwrap_or("unknown");
    let config = Config::from_gguf(&gguf);
    let metadata_count = gguf.metadata.len();
    let tensor_count = gguf.tensors.len();
    let mut dtype_counts: BTreeMap<String, usize> = BTreeMap::new();

    for tensor in &gguf.tensors {
        *dtype_counts
            .entry(format!("{:?}", tensor.dtype))
            .or_insert(0) += 1;
    }

    let tokenizer_vocab = gguf
        .metadata
        .get("tokenizer.ggml.tokens")
        .and_then(|value| value.as_string_array())
        .map(|tokens| tokens.len())
        .unwrap_or(0);
    let file_size_bytes = std::fs::metadata(path)
        .map_err(|err| err.to_string())?
        .len();
    let compatibility = compatibility_report(&gguf);
    let supported_architecture = compatibility.supported_architecture;
    let status = compatibility.status();

    let report = serde_json::json!({
        "type": "rusty-llm.inspect",
        "path": path.display().to_string(),
        "file_size_bytes": file_size_bytes,
        "status": status,
        "model": {
            "name": gguf.get_str("general.name"),
            "architecture": arch,
            "supported_architecture": supported_architecture,
        },
        "config": {
            "dim": config.dim,
            "hidden_dim": config.hidden_dim,
            "layers": config.n_layers,
            "heads": config.n_heads,
            "kv_heads": config.n_kv_heads,
            "head_dim": config.head_dim,
            "value_dim": config.value_dim,
            "kv_dim": config.kv_dim,
            "context_length": config.max_seq_len,
            "vocab_size": config.vocab_size,
            "rope_theta": config.rope_theta,
            "rms_norm_eps": config.rms_norm_eps,
            "sliding_window": config.sliding_window,
            "expert_count": config.expert_count,
            "expert_used_count": config.expert_used_count,
        },
        "tokenizer": {
            "vocab_size": tokenizer_vocab,
            "chat_template": gguf.metadata.get("tokenizer.chat_template").and_then(|v| v.as_str()).is_some(),
        },
        "gguf": {
            "metadata_entries": metadata_count,
            "tensors": tensor_count,
            "data_offset": gguf.data_offset,
            "tensor_types": dtype_counts,
            "unsupported_tensor_examples": compatibility.unsupported_tensor_types.iter().take(16).collect::<Vec<_>>(),
            "unsupported_tensor_count": compatibility.unsupported_tensor_types.len(),
            "missing_tensor_examples": compatibility.missing_tensors.iter().take(16).collect::<Vec<_>>(),
            "missing_tensor_count": compatibility.missing_tensors.len(),
            "unsupported_layouts": compatibility.unsupported_layouts,
        },
        "api_compatibility": {
            "openai": ["/v1/models", "/v1/completions", "/v1/chat/completions", "/v1/responses", "/v1/embeddings"],
            "lm_studio": ["/api/v0/models", "/api/v0/completions", "/api/v0/chat/completions", "/api/v0/embeddings"],
            "ollama": ["/api/tags", "/api/generate", "/api/chat", "/api/embeddings"],
            "openapi": ["/openapi.json", "/swagger.json", "/docs"],
        }
    });
    let body = serde_json::to_string_pretty(&report)
        .map_err(|err| format!("Failed to serialize inspect JSON: {}", err))?;
    println!("{}", body);
    Ok(())
}

/// Runs a human-readable generation benchmark.
fn run_benchmark(
    runner: &Runner,
    load_info: &LoadInfo,
    prompt: &str,
    options: &GenerationOptions,
    runs: usize,
    json: bool,
    output: bool,
) -> Result<(), String> {
    if json {
        return run_benchmark_json(runner, load_info, prompt, options, runs, output);
    }

    println!("Benchmark");
    println!("model={}", runner.model_name().unwrap_or("unknown"));
    println!("architecture={}", runner.architecture());
    println!("load_ms={}", load_info.load_time.as_millis());
    println!("runs={}", runs);
    println!("max_tokens={}", options.max_tokens);
    println!();
    println!(
        "run,prompt_tokens,generated_tokens,prefill_ms,decode_ms,total_ms,wall_ms,decode_tok_s,mtp_accept_rate,mtp_draft_tok_s"
    );

    let mut total_prompt_tokens = 0usize;
    let mut total_generated_tokens = 0usize;
    let mut total_prefill = Duration::from_secs(0);
    let mut total_decode = Duration::from_secs(0);
    let mut total_model = Duration::from_secs(0);
    let mut total_wall = Duration::from_secs(0);

    for run in 0..runs {
        let mut run_options = options.clone();
        if options.seed != 0 {
            run_options.seed = options.seed.wrapping_add(run as u64);
        }

        let wall_start = Instant::now();
        let mut loaded_skills = HashSet::new();
        let result = runner.generate_with_skill_memory(prompt, &run_options, &mut loaded_skills)?;
        let wall = wall_start.elapsed();
        let decode_tok_s = result.stats.generated_tokens as f64
            / result.stats.decode_time.as_secs_f64().max(0.001);

        let mtp_accept_rate = result
            .stats
            .speculative
            .as_ref()
            .map(|s| s.accept_rate())
            .unwrap_or(0.0);
        let mtp_draft_tok_s = result
            .stats
            .speculative
            .as_ref()
            .map(|s| s.draft_tok_s())
            .unwrap_or(0.0);

        println!(
            "{},{},{},{},{},{},{},{:.2},{:.2},{:.2}",
            run + 1,
            result.stats.prompt_tokens,
            result.stats.generated_tokens,
            result.stats.prefill_time.as_millis(),
            result.stats.decode_time.as_millis(),
            result.stats.total_time.as_millis(),
            wall.as_millis(),
            decode_tok_s,
            mtp_accept_rate,
            mtp_draft_tok_s
        );
        if output {
            println!("run {} output: {}", run + 1, result.text);
        }

        total_prompt_tokens += result.stats.prompt_tokens;
        total_generated_tokens += result.stats.generated_tokens;
        total_prefill += result.stats.prefill_time;
        total_decode += result.stats.decode_time;
        total_model += result.stats.total_time;
        total_wall += wall;
    }

    println!();
    println!(
        "avg_prompt_tokens={:.1}",
        total_prompt_tokens as f64 / runs as f64
    );
    println!(
        "avg_generated_tokens={:.1}",
        total_generated_tokens as f64 / runs as f64
    );
    println!(
        "avg_prefill_ms={:.1}",
        total_prefill.as_secs_f64() * 1000.0 / runs as f64
    );
    println!(
        "avg_decode_ms={:.1}",
        total_decode.as_secs_f64() * 1000.0 / runs as f64
    );
    println!(
        "avg_total_ms={:.1}",
        total_model.as_secs_f64() * 1000.0 / runs as f64
    );
    println!(
        "avg_wall_ms={:.1}",
        total_wall.as_secs_f64() * 1000.0 / runs as f64
    );
    println!(
        "aggregate_decode_tok_s={:.2}",
        total_generated_tokens as f64 / total_decode.as_secs_f64().max(0.001)
    );

    Ok(())
}

/// Runs a generation benchmark and emits JSON.
fn run_benchmark_json(
    runner: &Runner,
    load_info: &LoadInfo,
    prompt: &str,
    options: &GenerationOptions,
    runs: usize,
    output: bool,
) -> Result<(), String> {
    let ultra_profile = options.runtime.profile == RuntimeProfile::MistralUltra;
    let metal_q4k_min_rows = if ultra_profile {
        metal::ultra_q4k_min_metal_rows()
    } else {
        metal::Q4K_MIN_METAL_ROWS
    };
    let metal_q6k_min_rows = if ultra_profile {
        metal::ultra_q6k_min_metal_rows()
    } else {
        metal::Q6K_MIN_METAL_ROWS
    };
    let mut total_prompt_tokens = 0usize;
    let mut total_generated_tokens = 0usize;
    let mut total_prefill = Duration::from_secs(0);
    let mut total_decode = Duration::from_secs(0);
    let mut total_model = Duration::from_secs(0);
    let mut total_wall = Duration::from_secs(0);
    let mut run_values = Vec::with_capacity(runs);

    for run in 0..runs {
        let mut run_options = options.clone();
        if options.seed != 0 {
            run_options.seed = options.seed.wrapping_add(run as u64);
        }

        let wall_start = Instant::now();
        let mut loaded_skills = HashSet::new();
        let result = runner.generate_with_skill_memory(prompt, &run_options, &mut loaded_skills)?;
        let wall = wall_start.elapsed();
        let decode_tok_s = result.stats.generated_tokens as f64
            / result.stats.decode_time.as_secs_f64().max(0.001);
        let prefill_tok_s =
            result.stats.prompt_tokens as f64 / result.stats.prefill_time.as_secs_f64().max(0.001);

        let mut run_value = serde_json::json!({
            "run": run + 1,
            "prompt_tokens": result.stats.prompt_tokens,
            "generated_tokens": result.stats.generated_tokens,
            "prefill_ms": result.stats.prefill_time.as_millis(),
            "decode_ms": result.stats.decode_time.as_millis(),
            "total_ms": result.stats.total_time.as_millis(),
            "wall_ms": wall.as_millis(),
            "prefill_tok_s": prefill_tok_s,
            "decode_tok_s": decode_tok_s,
        });
        if let Some(spec) = &result.stats.speculative {
            run_value["speculative"] = serde_json::json!({
                "drafted_tokens": spec.drafted_tokens,
                "accepted_tokens": spec.accepted_tokens,
                "rejected_tokens": spec.rejected_tokens,
                "accept_rate": spec.accept_rate(),
                "draft_ms": spec.draft_time.as_millis(),
                "draft_tok_s": spec.draft_tok_s(),
                "effective_tok_s": decode_tok_s,
                "disabled": spec.disabled,
            });
        }
        if output {
            run_value["text"] = serde_json::json!(result.text);
        }
        run_values.push(run_value);

        total_prompt_tokens += result.stats.prompt_tokens;
        total_generated_tokens += result.stats.generated_tokens;
        total_prefill += result.stats.prefill_time;
        total_decode += result.stats.decode_time;
        total_model += result.stats.total_time;
        total_wall += wall;
    }

    let response = serde_json::json!({
        "type": "rusty-llm.benchmark",
        "model": runner.model_name().unwrap_or("unknown"),
        "architecture": runner.architecture(),
        "load_ms": load_info.load_time.as_millis(),
        "file_size_bytes": load_info.file_size_bytes,
        "metal": {
            "available": metal::available(),
            "enabled": metal::enabled(),
            "ultra": ultra_profile,
            "nocopy": metal::nocopy_enabled(),
            "fused_ffn": metal::fused_ffn_enabled(),
            "q4_k": metal::enabled(),
            "q6_k": metal::q6k_enabled(),
            "q4_k_min_rows": metal_q4k_min_rows,
            "q4_k_min_cols": metal::Q4K_MIN_METAL_COLS,
            "q6_k_min_rows": metal_q6k_min_rows,
            "attention_min_tokens": metal::attention_min_metal_tokens(),
            "ultra_q4_k_min_rows": metal::ultra_q4k_min_metal_rows(),
            "ultra_q6_k_min_rows": metal::ultra_q6k_min_metal_rows(),
            "ultra_attention_min_tokens": metal::ultra_attention_min_metal_tokens(),
        },
        "runs": runs,
        "prompt": prompt,
        "options": {
            "max_tokens": options.max_tokens,
            "temperature": options.sampler.temperature,
            "top_p": options.sampler.top_p,
            "top_k": options.sampler.top_k,
            "repeat_penalty": options.sampler.repeat_penalty,
            "seed": options.seed,
            "stop_sequences": options.stop_sequences,
            "speculative": {
                "enabled": options.speculative.enabled && runner.has_speculative_assistant(),
                "assistant": options.speculative.assistant_path.as_deref(),
                "max_draft_tokens": options.speculative.max_draft_tokens,
                "adaptive": options.speculative.adaptive,
                "min_accept_rate": options.speculative.min_accept_rate,
            },
            "runtime": {
                "profile": options.runtime.profile.as_str(),
                "kv_cache_dtype": options.runtime.kv_cache_dtype.as_str(),
                "flash_attention": options.runtime.flash_attention,
                "sliding_window_size": options.runtime.sliding_window_size,
                "max_context": options.runtime.max_context,
            },
        },
        "results": run_values,
        "summary": {
            "avg_prompt_tokens": total_prompt_tokens as f64 / runs as f64,
            "avg_generated_tokens": total_generated_tokens as f64 / runs as f64,
            "avg_prefill_ms": total_prefill.as_secs_f64() * 1000.0 / runs as f64,
            "avg_decode_ms": total_decode.as_secs_f64() * 1000.0 / runs as f64,
            "avg_total_ms": total_model.as_secs_f64() * 1000.0 / runs as f64,
            "avg_wall_ms": total_wall.as_secs_f64() * 1000.0 / runs as f64,
            "aggregate_prefill_tok_s": total_prompt_tokens as f64 / total_prefill.as_secs_f64().max(0.001),
            "aggregate_decode_tok_s": total_generated_tokens as f64 / total_decode.as_secs_f64().max(0.001),
        }
    });
    let body = serde_json::to_string_pretty(&response)
        .map_err(|err| format!("Failed to serialize benchmark JSON: {}", err))?;
    println!("{}", body);
    Ok(())
}

/// Runs matrix-vector kernel benchmarks for the loaded model.
fn run_kernel_benchmark(
    runner: &Runner,
    model_path: &Path,
    options: &GenerationOptions,
    runs: usize,
    layer: usize,
    json: bool,
) -> Result<(), String> {
    let ultra_profile = options.runtime.profile == RuntimeProfile::MistralUltra;
    let metal_q4k_min_rows = if ultra_profile {
        metal::ultra_q4k_min_metal_rows()
    } else {
        metal::Q4K_MIN_METAL_ROWS
    };
    let metal_q6k_min_rows = if ultra_profile {
        metal::ultra_q6k_min_metal_rows()
    } else {
        metal::Q6K_MIN_METAL_ROWS
    };
    let (layer, rows) = runner.kernel_benchmark_with_options(runs, layer, options)?;
    if json {
        let kernels: Vec<_> = rows
            .iter()
            .map(|row| {
                serde_json::json!({
                    "name": row.name,
                    "dtype": row.dtype,
                    "rows": row.rows,
                    "cols": row.cols,
                    "runs": row.runs,
                    "avg_ms": row.avg_ms,
                    "total_ms": row.total_ms,
                })
            })
            .collect();
        let payload = serde_json::json!({
            "type": "rusty-llm.kernel_benchmark",
            "format": "llm-kernel-bench.v1",
            "runtime": "RustyLLM",
            "model": {
                "path": model_path.display().to_string(),
                "name": runner.model_name().unwrap_or(""),
                "arch": runner.architecture(),
                "dim": runner.config().dim,
                "hidden_dim": runner.config().hidden_dim,
                "layers": runner.config().n_layers,
            },
            "metal": {
                "available": metal::available(),
                "enabled": metal::enabled(),
                "ultra": ultra_profile,
                "nocopy": metal::nocopy_enabled(),
                "fused_ffn": metal::fused_ffn_enabled(),
                "q4_k": metal::enabled(),
                "q6_k": metal::q6k_enabled(),
                "q4_k_min_rows": metal_q4k_min_rows,
                "q4_k_min_cols": metal::Q4K_MIN_METAL_COLS,
                "q6_k_min_rows": metal_q6k_min_rows,
                "attention_min_tokens": metal::attention_min_metal_tokens(),
                "ultra_q4_k_min_rows": metal::ultra_q4k_min_metal_rows(),
                "ultra_q6_k_min_rows": metal::ultra_q6k_min_metal_rows(),
                "ultra_attention_min_tokens": metal::ultra_attention_min_metal_tokens(),
            },
            "profile": options.runtime.profile.as_str(),
            "layer": layer,
            "runs": runs,
            "kernels": kernels,
        });
        let body = serde_json::to_string_pretty(&payload)
            .map_err(|err| format!("Failed to serialize kernel benchmark JSON: {}", err))?;
        println!("{}", body);
        return Ok(());
    }

    println!(
        "Kernel benchmark format=llm-kernel-bench.v1 runtime=RustyLLM layer={} runs={}",
        layer, runs
    );
    println!(
        "Metal available={} enabled={} ultra={} nocopy={} fused_ffn={} q4_k={} q6_k={} q4_k_min_rows={} q4_k_min_cols={} q6_k_min_rows={} attention_min_tokens={} ultra_attention_min_tokens={}",
        metal::available(),
        metal::enabled(),
        ultra_profile,
        metal::nocopy_enabled(),
        metal::fused_ffn_enabled(),
        metal::enabled(),
        metal::q6k_enabled(),
        metal_q4k_min_rows,
        metal::Q4K_MIN_METAL_COLS,
        metal_q6k_min_rows,
        metal::attention_min_metal_tokens(),
        metal::ultra_attention_min_metal_tokens()
    );
    for row in rows {
        println!(
            "{} dtype={} rows={} cols={} avg={:.3}ms total={:.3}ms",
            row.name, row.dtype, row.rows, row.cols, row.avg_ms, row.total_ms
        );
    }
    Ok(())
}

/// Runs the interactive chat REPL.
fn run_repl(
    runner: &Runner,
    options: &GenerationOptions,
    chat_history_path: Option<&str>,
) -> Result<(), String> {
    eprintln!("REPL mode. Commands: /exit, /quit, /clear, /help");
    let stdin = io::stdin();
    let mut history: Vec<ChatMessage> = Vec::new();
    let mut loaded_skills = HashSet::new();

    loop {
        eprint!("repl> ");
        io::stderr().flush().map_err(|err| err.to_string())?;

        let mut line = String::new();
        if stdin
            .lock()
            .read_line(&mut line)
            .map_err(|err| err.to_string())?
            == 0
        {
            break;
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        match line {
            "/exit" | "/quit" => break,
            "/clear" => {
                history.clear();
                loaded_skills.clear();
                eprintln!("History cleared.");
                continue;
            }
            "/help" => {
                eprintln!("Enter a prompt to generate text.");
                eprintln!("/clear resets the chat history for this session.");
                continue;
            }
            _ => {}
        }

        history.push(ChatMessage::user(line));
        let result = runner.generate_chat_stream_with_skill_memory(
            &history,
            options,
            &mut loaded_skills,
            |text| {
                print!("{}", text);
                let _ = io::stdout().flush();
            },
        );

        match result {
            Ok(result) => {
                println!();
                append_cli_history(chat_history_path, runner, "cli.repl", &history, &result)?;
                history.push(ChatMessage::assistant(result.text));
                eprintln!(
                    "stats: prompt={} generated={} total={:.2}s",
                    result.stats.prompt_tokens,
                    result.stats.generated_tokens,
                    result.stats.total_time.as_secs_f32()
                );
            }
            Err(err) => {
                eprintln!("Generation error: {}", err);
                history.pop();
            }
        }
    }
    Ok(())
}

/// Appends one CLI chat turn to the history file.
fn append_cli_history(
    path: Option<&str>,
    runner: &Runner,
    source: &str,
    messages: &[ChatMessage],
    result: &rusty_llm::runtime::GenerationResult,
) -> Result<(), String> {
    let Some(path) = path else {
        return Ok(());
    };
    let mut entries = match std::fs::read_to_string(path) {
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
        "model": runner.model_name().unwrap_or(runner.architecture()),
        "architecture": runner.architecture(),
        "messages": messages.iter().map(chat_message_json).collect::<Vec<_>>(),
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
            std::fs::create_dir_all(parent)
                .map_err(|err| format!("Failed to create {}: {}", parent.display(), err))?;
        }
    }
    let body = serde_json::to_string_pretty(&entries)
        .map_err(|err| format!("Failed to serialize chat history: {}", err))?;
    std::fs::write(path, body)
        .map_err(|err| format!("Failed to write chat history {}: {}", path, err))
}

/// Serializes a chat message for history output.
fn chat_message_json(message: &ChatMessage) -> serde_json::Value {
    let role = match message.role {
        rusty_llm::runtime::ChatRole::System => "system",
        rusty_llm::runtime::ChatRole::User => "user",
        rusty_llm::runtime::ChatRole::Assistant => "assistant",
    };
    serde_json::json!({ "role": role, "content": message.content })
}

/// Returns the current Unix timestamp in seconds.
fn unix_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

/// Check if stdin is a terminal (not piped)
fn atty_is_stdin() -> bool {
    // Use isatty via libc — available on both macOS and Linux without deps
    unsafe extern "C" {
        /// Returns nonzero when the file descriptor is attached to a terminal.
        fn isatty(fd: i32) -> i32;
    }
    unsafe { isatty(0) != 0 }
}
