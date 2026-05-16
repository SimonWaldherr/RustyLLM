use rusty_llm::runtime::{ChatMessage, GenerationOptions, Runner};
#[cfg(not(target_family = "wasm"))]
use rusty_llm::server::{self, ServeOptions};
use rusty_llm::simd;
use std::env;
use std::fmt::Display;
use std::io::{self, BufRead, Read, Write};
#[cfg(not(target_family = "wasm"))]
use std::sync::Arc;

fn print_usage(name: &str) {
    eprintln!("rusty-llm v0.2.0");
    eprintln!();
    eprintln!("Usage: {} <model.gguf> [options]", name);
    eprintln!();
    eprintln!("Options:");
    eprintln!("  --prompt <text>           Input prompt (interactive if omitted)");
    eprintln!("  --repl                    Start an interactive REPL session");
    eprintln!("  --serve <addr>            Start HTTP(S) API server, e.g. 127.0.0.1:8080");
    eprintln!("  --tls-cert <path>         PEM certificate for HTTPS");
    eprintln!("  --tls-key <path>          PEM private key for HTTPS");
    eprintln!("  --max-connections <N>     Max concurrent server connections");
    eprintln!("  --max-tokens <N>          Max tokens to generate (default: 256)");
    eprintln!("  --temp <F>                Temperature (default: 0.7, 0=greedy)");
    eprintln!("  --top-p <F>               Nucleus sampling threshold (default: 0.9)");
    eprintln!("  --top-k <N>               Top-K filtering (default: 40)");
    eprintln!("  --repeat-penalty <F>      Repetition penalty (default: 1.1)");
    eprintln!("  --seed <N>                RNG seed (default: time-based)");
    eprintln!("  --threads <N>             Override thread count");
    eprintln!("  --system-prompt <T>       Override the default system prompt");
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
        return Err(String::from("Missing required <model.gguf> path."));
    }

    if args[1] == "--help" || args[1] == "-h" {
        print_usage(&args[0]);
        return Ok(());
    }

    let model_path = &args[1];

    let mut prompt = String::new();
    let mut options = GenerationOptions::default();
    let mut list_tensors = false;
    let mut threads_override: Option<usize> = None;
    let mut repl_mode = false;
    let mut serve_addr: Option<String> = None;
    let mut tls_cert: Option<String> = None;
    let mut tls_key: Option<String> = None;
    let mut max_connections_override: Option<usize> = None;

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--prompt" | "-p" => {
                prompt = parse_arg::<String>(&args, &mut i, "--prompt")?;
            }
            "--repl" => {
                repl_mode = true;
            }
            "--serve" => {
                serve_addr = Some(parse_arg::<String>(&args, &mut i, "--serve")?);
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
            "--list-tensors" => {
                list_tensors = true;
            }
            other => {
                return Err(format!("Unknown option: {}", other));
            }
        }
        i += 1;
    }

    if tls_cert.is_some() ^ tls_key.is_some() {
        return Err(String::from(
            "Both --tls-cert and --tls-key must be provided together.",
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

    if let Some(n) = threads_override {
        simd::set_num_threads(n);
        eprintln!("Worker threads: {}", n);
    }

    eprintln!("\nLoading: {}", model_path);
    let (runner, load_info) = Runner::from_path(model_path)?;
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

    if list_tensors {
        for tensor in &runner.gguf().tensors {
            eprintln!("{} {:?} {:?}", tensor.name, tensor.dtype, tensor.dims);
        }
        return Ok(());
    }

    // Server mode takes over the process after the model is loaded.
    if let Some(addr) = serve_addr {
        let protocol = if tls_cert.is_some() && tls_key.is_some() {
            "HTTPS"
        } else {
            "HTTP"
        };
        let max_connections = max_connections_override.unwrap_or_else(|| (n_threads * 8).max(16));
        eprintln!("{} endpoint listening on {}", protocol, addr);
        eprintln!("POST /generate and GET /health are available.");
        eprintln!("Max concurrent connections: {}", max_connections);
        let serve_options = ServeOptions {
            addr,
            defaults: options.clone(),
            tls_cert_path: tls_cert,
            tls_key_path: tls_key,
            max_concurrent_connections: max_connections,
        };
        server::serve(Arc::new(runner), serve_options)?;
        return Ok(());
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
