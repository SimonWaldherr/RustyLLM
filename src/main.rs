#[cfg(not(target_family = "wasm"))]
use rusty_llm::catalog::{
    default_model_dir, discover_models, print_model_list, resolve_model_path,
};
#[cfg(not(target_family = "wasm"))]
use rusty_llm::metal;
use rusty_llm::runtime::{ChatMessage, GenerationOptions, LoadInfo, Runner};
#[cfg(all(not(target_family = "wasm"), feature = "server"))]
use rusty_llm::server::{self, ServeOptions};
use rusty_llm::simd;
use std::env;
use std::fmt::Display;
use std::io::{self, BufRead, Read, Write};
use std::path::PathBuf;
#[cfg(all(not(target_family = "wasm"), feature = "server"))]
use std::sync::Arc;
use std::time::{Duration, Instant};

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
    #[cfg(feature = "server")]
    eprintln!("  --serve <addr>            Start HTTP(S) API server, e.g. 127.0.0.1:8080");
    #[cfg(feature = "server")]
    eprintln!("  --chat                    Enable the minimal Web UI at /chat with --serve");
    #[cfg(feature = "server")]
    eprintln!("  --tls-cert <path>         PEM certificate for HTTPS");
    #[cfg(feature = "server")]
    eprintln!("  --tls-key <path>          PEM private key for HTTPS");
    #[cfg(feature = "server")]
    eprintln!("  --max-connections <N>     Max concurrent server connections");
    eprintln!("  --max-tokens <N>          Max tokens to generate (default: 256)");
    eprintln!("  --temp <F>                Temperature (default: 0.7, 0=greedy)");
    eprintln!("  --top-p <F>               Nucleus sampling threshold (default: 0.9)");
    eprintln!("  --top-k <N>               Top-K filtering (default: 40)");
    eprintln!("  --repeat-penalty <F>      Repetition penalty (default: 1.1)");
    eprintln!("  --seed <N>                RNG seed (default: time-based)");
    eprintln!("  --threads <N>             Override thread count");
    eprintln!("  --system-prompt <T>       Override the default system prompt");
    eprintln!("  --stop <text>             Stop generation when this string appears");
    eprintln!("  --embed                   Embed prompt and print the vector (RAG mode)");
    eprintln!("  --bench                   Run a non-streaming generation benchmark");
    eprintln!("  --bench-runs <N>          Number of benchmark runs (default: 3)");
    eprintln!("  --list-tensors            Print GGUF tensor inventory and exit");
}

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

fn main() {
    if let Err(err) = run() {
        eprintln!("{}", err);
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
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
    let mut chat_ui = false;
    let mut tls_cert: Option<String> = None;
    let mut tls_key: Option<String> = None;
    let mut max_connections_override: Option<usize> = None;
    let mut embed_mode = false;
    let mut bench_mode = false;
    let mut bench_runs = 3usize;

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
            "--serve" => {
                serve_addr = Some(parse_arg::<String>(&args, &mut i, "--serve")?);
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
            "--system-prompt" => {
                options.system_prompt = parse_arg::<String>(&args, &mut i, "--system-prompt")?;
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
            "--bench-runs" => {
                bench_runs = parse_arg::<usize>(&args, &mut i, "--bench-runs")?;
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
        eprintln!("Metal: Q4_K matvec enabled");
    } else if std::env::var_os("RUSTY_LLM_METAL").is_some() {
        eprintln!("Metal: unavailable, using CPU");
    }
    if std::env::var_os("RUSTY_LLM_FAST_ATTN").is_some() {
        eprintln!("Attention: fast approximation mode enabled");
    }

    if let Some(n) = threads_override {
        simd::set_num_threads(n);
        eprintln!("Worker threads: {}", n);
    }

    let model_path = resolve_model_path(model_selector.as_deref(), &model_dir)?;

    eprintln!("\nLoading: {}", model_path.display());
    let model_path_str = model_path
        .to_str()
        .ok_or_else(|| format!("Non-UTF-8 model path: {}", model_path.display()))?;
    let (runner, load_info) = Runner::from_path(model_path_str)?;
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
    io::stderr().flush().map_err(|err| err.to_string())?;

    if list_tensors {
        for tensor in &runner.gguf().tensors {
            eprintln!("{} {:?} {:?}", tensor.name, tensor.dtype, tensor.dims);
        }
        return Ok(());
    }

    if bench_mode {
        if embed_mode || repl_mode || serve_addr.is_some() {
            return Err(String::from(
                "--bench cannot be combined with --embed, --repl, or --serve.",
            ));
        }
        let bench_prompt = if prompt.trim().is_empty() {
            "Explain local LLM inference performance in one concise paragraph."
        } else {
            prompt.trim()
        };
        run_benchmark(&runner, &load_info, bench_prompt, &options, bench_runs)?;
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
                    "Routes: GET /chat, GET /chat?expert, GET /health, POST /generate, GET /v1/models, POST /v1/completions, POST /v1/chat/completions, POST /v1/embeddings."
                );
            } else {
                eprintln!(
                    "Routes: GET /health, POST /generate, GET /v1/models, POST /v1/completions, POST /v1/chat/completions, POST /v1/embeddings."
                );
            }
            eprintln!("Max concurrent connections: {}", max_connections);
            let serve_options = ServeOptions {
                addr,
                defaults: options.clone(),
                tls_cert_path: tls_cert,
                tls_key_path: tls_key,
                max_concurrent_connections: max_connections,
                chat_ui,
            };
            server::serve(Arc::new(runner), serve_options)?;
            return Ok(());
        }
    }

    if repl_mode {
        run_repl(&runner, &options)?;
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

    let result = runner.generate_stream(&prompt, &options, |text| {
        print!("{}", text);
        let _ = io::stdout().flush();
    })?;

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
    eprintln!("Total: {:.2}s", result.stats.total_time.as_secs_f32());

    Ok(())
}

fn run_benchmark(
    runner: &Runner,
    load_info: &LoadInfo,
    prompt: &str,
    options: &GenerationOptions,
    runs: usize,
) -> Result<(), String> {
    println!("Benchmark");
    println!("model={}", runner.model_name().unwrap_or("unknown"));
    println!("architecture={}", runner.architecture());
    println!("load_ms={}", load_info.load_time.as_millis());
    println!("runs={}", runs);
    println!("max_tokens={}", options.max_tokens);
    println!();
    println!(
        "run,prompt_tokens,generated_tokens,prefill_ms,decode_ms,total_ms,wall_ms,decode_tok_s"
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
        let result = runner.generate(prompt, &run_options)?;
        let wall = wall_start.elapsed();
        let decode_tok_s = result.stats.generated_tokens as f64
            / result.stats.decode_time.as_secs_f64().max(0.001);

        println!(
            "{},{},{},{},{},{},{},{:.2}",
            run + 1,
            result.stats.prompt_tokens,
            result.stats.generated_tokens,
            result.stats.prefill_time.as_millis(),
            result.stats.decode_time.as_millis(),
            result.stats.total_time.as_millis(),
            wall.as_millis(),
            decode_tok_s
        );

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

fn run_repl(runner: &Runner, options: &GenerationOptions) -> Result<(), String> {
    eprintln!("REPL mode. Commands: /exit, /quit, /clear, /help");
    let stdin = io::stdin();
    let mut history: Vec<ChatMessage> = Vec::new();

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
        let result = runner.generate_chat_stream(&history, options, |text| {
            print!("{}", text);
            let _ = io::stdout().flush();
        });

        match result {
            Ok(result) => {
                println!();
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

/// Check if stdin is a terminal (not piped)
fn atty_is_stdin() -> bool {
    // Use isatty via libc — available on both macOS and Linux without deps
    unsafe extern "C" {
        fn isatty(fd: i32) -> i32;
    }
    unsafe { isatty(0) != 0 }
}
