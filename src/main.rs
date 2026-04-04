use rusty_llm::runtime::{ChatMessage, GenerationOptions, Runner};
#[cfg(not(target_family = "wasm"))]
use rusty_llm::server::{self, ServeOptions};
use rusty_llm::simd;
use std::env;
use std::io::{self, BufRead, Write};
#[cfg(not(target_family = "wasm"))]
use std::sync::Arc;

fn print_usage(name: &str) {
    eprintln!("rusty-llm v0.2.0");
    eprintln!();
    eprintln!("Usage: {} <model.gguf> [options]", name);
    eprintln!();
    eprintln!("Options:");
    eprintln!("  --prompt <text>       Input prompt (interactive if omitted)");
    eprintln!("  --repl                Start an interactive REPL session");
    eprintln!("  --serve <addr>        Start HTTP(S) API server, e.g. 127.0.0.1:8080");
    eprintln!("  --tls-cert <path>     PEM certificate for HTTPS");
    eprintln!("  --tls-key <path>      PEM private key for HTTPS");
    eprintln!("  --max-tokens <N>      Max tokens to generate (default: 256)");
    eprintln!("  --temp <F>            Temperature (default: 0.7, 0=greedy)");
    eprintln!("  --top-p <F>           Nucleus sampling threshold (default: 0.9)");
    eprintln!("  --top-k <N>           Top-K filtering (default: 40)");
    eprintln!("  --repeat-penalty <F>  Repetition penalty (default: 1.1)");
    eprintln!("  --seed <N>            RNG seed (default: time-based)");
    eprintln!("  --threads <N>         Override thread count");
    eprintln!("  --system-prompt <T>   Override the default system prompt");
    eprintln!("  --list-tensors        Print GGUF tensor inventory and exit");
}

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 || args[1] == "--help" || args[1] == "-h" {
        print_usage(&args[0]);
        std::process::exit(if args.len() < 2 { 1 } else { 0 });
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

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--prompt" | "-p" => {
                i += 1;
                prompt = args[i].clone();
            }
            "--repl" => {
                repl_mode = true;
            }
            "--serve" => {
                i += 1;
                serve_addr = Some(args[i].clone());
            }
            "--tls-cert" => {
                i += 1;
                tls_cert = Some(args[i].clone());
            }
            "--tls-key" => {
                i += 1;
                tls_key = Some(args[i].clone());
            }
            "--max-tokens" | "-n" => {
                i += 1;
                options.max_tokens = args[i].parse().expect("Invalid --max-tokens");
            }
            "--temp" | "-t" => {
                i += 1;
                options.sampler.temperature = args[i].parse().expect("Invalid --temp");
            }
            "--top-p" => {
                i += 1;
                options.sampler.top_p = args[i].parse().expect("Invalid --top-p");
            }
            "--top-k" => {
                i += 1;
                options.sampler.top_k = args[i].parse().expect("Invalid --top-k");
            }
            "--repeat-penalty" => {
                i += 1;
                options.sampler.repeat_penalty = args[i].parse().expect("Invalid --repeat-penalty");
            }
            "--seed" => {
                i += 1;
                options.seed = args[i].parse().expect("Invalid --seed");
            }
            "--threads" => {
                i += 1;
                threads_override = Some(args[i].parse().expect("Invalid --threads"));
            }
            "--system-prompt" => {
                i += 1;
                options.system_prompt = args[i].clone();
            }
            "--list-tensors" => {
                list_tensors = true;
            }
            other => {
                eprintln!("Unknown option: {}", other);
                std::process::exit(1);
            }
        }
        i += 1;
    }

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
        eprintln!("Worker threads: {}", n.max(1));
    }

    eprintln!("\nLoading: {}", model_path);
    let (runner, load_info) = Runner::from_path(model_path).unwrap_or_else(|err| {
        eprintln!("{}", err);
        std::process::exit(1);
    });
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
        return;
    }

    // Server mode takes over the process after the model is loaded.
    if let Some(addr) = serve_addr {
        let protocol = if tls_cert.is_some() && tls_key.is_some() {
            "HTTPS"
        } else {
            "HTTP"
        };
        eprintln!("{} endpoint listening on {}", protocol, addr);
        eprintln!("POST /generate and GET /health are available.");
        let serve_options = ServeOptions {
            addr,
            defaults: options.clone(),
            tls_cert_path: tls_cert,
            tls_key_path: tls_key,
        };
        server::serve(Arc::new(runner), serve_options).unwrap_or_else(|err| {
            eprintln!("Server error: {}", err);
            std::process::exit(1);
        });
        return;
    }

    if repl_mode {
        run_repl(&runner, &options);
        return;
    }

    // Fall back to stdin when no prompt flag was provided so the binary works
    // both interactively and in shell pipelines.
    if prompt.is_empty() {
        if atty_is_stdin() {
            eprint!(">>> ");
            io::stderr().flush().unwrap();
            io::stdin().lock().read_line(&mut prompt).unwrap();
            prompt = prompt.trim().to_string();
        } else {
            let mut buf = String::new();
            io::stdin().lock().read_to_string_linewise(&mut buf);
            prompt = buf.trim().to_string();
        }
    }

    if prompt.is_empty() {
        eprintln!("No prompt provided.");
        std::process::exit(1);
    }

    let result = runner
        .generate_stream(&prompt, &options, |text| {
            print!("{}", text);
            io::stdout().flush().unwrap();
        })
        .unwrap_or_else(|err| {
            eprintln!("Generation error: {}", err);
            std::process::exit(1);
        });

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
}

fn run_repl(runner: &Runner, options: &GenerationOptions) {
    eprintln!("REPL mode. Commands: /exit, /quit, /clear, /help");
    let stdin = io::stdin();
    let mut history: Vec<ChatMessage> = Vec::new();

    loop {
        eprint!("repl> ");
        io::stderr().flush().unwrap();

        let mut line = String::new();
        if stdin.lock().read_line(&mut line).unwrap_or(0) == 0 {
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
            io::stdout().flush().unwrap();
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
}

/// Check if stdin is a terminal (not piped)
fn atty_is_stdin() -> bool {
    // Use isatty via libc — available on both macOS and Linux without deps
    extern "C" {
        fn isatty(fd: i32) -> i32;
    }
    unsafe { isatty(0) != 0 }
}

/// Helper to read all lines from stdin
trait ReadAllLines {
    fn read_to_string_linewise(&mut self, buf: &mut String);
}

impl<T: BufRead> ReadAllLines for T {
    fn read_to_string_linewise(&mut self, buf: &mut String) {
        loop {
            let mut line = String::new();
            match self.read_line(&mut line) {
                Ok(0) => break,
                Ok(_) => buf.push_str(&line),
                Err(_) => break,
            }
        }
    }
}
